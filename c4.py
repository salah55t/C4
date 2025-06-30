import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
import base64
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException
from flask import Flask, request, Response, jsonify, render_template_string
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timedelta, timezone
from decouple import config
from typing import List, Dict, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler
from collections import deque
import warnings
import gc
from github import Github, GithubException, Repository

# --- تجاهل التحذيرات غير الهامة ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v6_with_sr.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV6_With_SR')

# ---------------------- تحميل متغيرات البيئة ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
    
    # --- GitHub Configuration (NEW) ---
    GITHUB_TOKEN: Optional[str] = config('GITHUB_TOKEN', default=None)
    GITHUB_REPO: str = config('GITHUB_REPO')
    RESULTS_FOLDER: str = 'ml_results'

except Exception as e:
     logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
     exit(1)

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
# --- V6 Model Constants ---
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V6_With_SR'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 90

# --- Indicator & Feature Parameters ---
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
BTC_CORR_PERIOD: int = 30

# --- Trading Logic Constants ---
MODEL_CONFIDENCE_THRESHOLD = 0.70
MAX_OPEN_TRADES: int = 5
TRADE_AMOUNT_USDT: float = 10.0
USE_DYNAMIC_SL_TP = True
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.0
USE_BTC_TREND_FILTER = True
BTC_SYMBOL = 'BTCUSDT'
BTC_TREND_TIMEFRAME = '4h'
BTC_TREND_EMA_PERIOD = 50

# --- المتغيرات العامة وقفل العمليات ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
github_repo_obj: Optional[Repository] = None 
ml_models_cache: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
current_prices: Dict[str, float] = {}
prices_lock = Lock()
notifications_cache = deque(maxlen=50)
notifications_lock = Lock()

# ---------------------- دوال GitHub (جديد) ----------------------
def init_github_repo():
    """Initializes the connection to the GitHub repository."""
    global github_repo_obj
    if not GITHUB_TOKEN or not GITHUB_REPO:
        logger.warning("⚠️ [GitHub] GitHub token or repo not configured. Model loading from GitHub will be skipped.")
        return
    try:
        g = Github(GITHUB_TOKEN)
        github_repo_obj = g.get_repo(GITHUB_REPO)
        logger.info(f"✅ [GitHub] Successfully connected to repository: {GITHUB_REPO}")
    except Exception as e:
        logger.error(f"❌ [GitHub] Failed to connect to GitHub repository: {e}")
        github_repo_obj = None

def load_ml_model_from_github(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Loads a model bundle (model, scaler, features) for a symbol from GitHub.
    Caches the model after the first successful load.
    """
    global ml_models_cache, github_repo_obj
    model_key = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_key in ml_models_cache:
        return ml_models_cache[model_key]

    if github_repo_obj is None:
        logger.error(f"❌ [GitHub Load] Cannot load model for {symbol} because GitHub repository object is not initialized.")
        return None

    model_filename = f"{RESULTS_FOLDER}/{symbol}_latest_model.pkl"
    logger.info(f"ℹ️ [GitHub Load] Attempting to load model for {symbol} from path: {model_filename}")

    try:
        file_content_object = github_repo_obj.get_contents(model_filename, ref="main")
        
        model_bytes = file_content_object.decoded_content

        if not model_bytes:
            logger.warning(f"⚠️ [GitHub Load] Model file for {symbol} at path '{model_filename}' is empty or could not be decoded. It will be skipped.")
            return None

        try:
            model_bundle = pickle.loads(model_bytes)
        except pickle.UnpicklingError as pickle_err:
            logger.error(f"❌ [GitHub Load] Failed to unpickle model for {symbol}. The file may be corrupted. Error: {pickle_err}")
            return None

        if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
            ml_models_cache[model_key] = model_bundle
            logger.info(f"✅ [GitHub Load] Successfully loaded and cached model for {symbol}.")
            return model_bundle
        else:
            logger.warning(f"⚠️ [GitHub Load] Model bundle for {symbol} is invalid or incomplete.")
            return None

    except GithubException as e:
        if e.status == 404:
            logger.warning(f"⚠️ [GitHub Load] Model file not found for {symbol} at path '{model_filename}'.")
        else:
            logger.error(f"❌ [GitHub Load] A GitHub error occurred while loading model for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ [GitHub Load] A general error occurred while loading model for {symbol}: {e}", exc_info=True)
        return None


# ---------------------- دوال قاعدة البيانات ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    global conn
    logger.info("[قاعدة البيانات] بدء تهيئة الاتصال...")
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION NOT NULL,
                        target_price DOUBLE PRECISION NOT NULL, stop_loss DOUBLE PRECISION NOT NULL,
                        status TEXT DEFAULT 'open', closing_price DOUBLE PRECISION, closed_at TIMESTAMP,
                        profit_percentage DOUBLE PRECISION, strategy_name TEXT, signal_details JSONB );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications ( id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL, message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS support_resistance_levels (
                        id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, level_price DOUBLE PRECISION NOT NULL,
                        level_type TEXT NOT NULL, timeframe TEXT NOT NULL, strength NUMERIC NOT NULL,
                        score NUMERIC DEFAULT 0, last_tested_at TIMESTAMP WITH TIME ZONE, details TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        CONSTRAINT unique_level UNIQUE (symbol, level_price, timeframe, level_type, details) );
                """)
            conn.commit()
            logger.info("✅ [قاعدة البيانات] تم تهيئة جداول قاعدة البيانات بنجاح.")
            return
        except Exception as e:
            logger.error(f"❌ [قاعدة البيانات] خطأ في الاتصال (المحاولة {attempt + 1}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: logger.critical("❌ [قاعدة البيانات] فشل الاتصال بعد عدة محاولات."); exit(1)

def check_db_connection() -> bool:
    global conn
    if conn is None or conn.closed != 0:
        logger.warning("[قاعدة البيانات] الاتصال مغلق، محاولة إعادة الاتصال...")
        init_db()
    try:
        if conn: conn.cursor().execute("SELECT 1;"); return True
        return False
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [قاعدة البيانات] فقدان الاتصال: {e}. محاولة إعادة الاتصال...")
        try: init_db(); return conn is not None and conn.closed == 0
        except Exception as retry_e: logger.error(f"❌ [قاعدة البيانات] فشل إعادة الاتصال: {retry_e}"); return False
    return False

def log_and_notify(level: str, message: str, notification_type: str):
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error, 'critical': logger.critical}
    log_methods.get(level.lower(), logger.info)(message)
    if not check_db_connection() or not conn: return
    try:
        new_notification = {"timestamp": datetime.now().isoformat(), "type": notification_type, "message": message}
        with notifications_lock: notifications_cache.appendleft(new_notification)
        with conn.cursor() as cur:
            cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [Notify DB] فشل حفظ التنبيه في قاعدة البيانات: {e}");
        if conn: conn.rollback()

# ---------------------- دوال Binance والبيانات ----------------------
def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    logger.info(f"ℹ️ [التحقق] قراءة الرموز من '{filename}' والتحقق منها مع Binance...")
    if not client: logger.error("❌ [التحقق] كائن Binance client غير مهيأ."); return []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        exchange_info = client.get_exchange_info()
        active = {s['symbol'] for s in exchange_info['symbols'] if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'}
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ [التحقق] سيقوم البوت بمراقبة {len(validated)} عملة معتمدة.")
        return validated
    except Exception as e:
        logger.error(f"❌ [التحقق] حدث خطأ أثناء التحقق من الرموز: {e}", exc_info=True); return []

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        start_dt = datetime.now(timezone.utc) - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        numeric_cols = {'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'float32'}
        df = df.astype(numeric_cols)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.dropna()
    except BinanceAPIException as e:
        logger.warning(f"⚠️ [API Binance] خطأ في جلب بيانات {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ [البيانات] خطأ أثناء جلب البيانات التاريخية لـ {symbol}: {e}")
        return None

def fetch_sr_levels_from_db(symbol: str) -> pd.DataFrame:
    if not check_db_connection() or not conn: return pd.DataFrame()
    query = "SELECT level_price, level_type, score FROM support_resistance_levels WHERE symbol = %s"
    try:
        with conn.cursor() as cur:
            cur.execute(query, (symbol,))
            levels = cur.fetchall()
            if not levels: return pd.DataFrame()
            return pd.DataFrame(levels)
    except Exception as e:
        logger.error(f"❌ [S/R Fetch Bot] Could not fetch S/R levels for {symbol}: {e}")
        if conn: conn.rollback()
        return pd.DataFrame()

# --- Feature Engineering Functions ---
def calculate_sr_features(df: pd.DataFrame, sr_levels_df: pd.DataFrame) -> pd.DataFrame:
    if sr_levels_df.empty:
        for col in ['dist_to_support', 'score_of_support', 'dist_to_resistance', 'score_of_resistance']:
            df[col] = 0.0
        return df

    supports = sr_levels_df[sr_levels_df['level_type'].str.contains('support|poc|confluence', case=False, na=False)]
    resistances = sr_levels_df[sr_levels_df['level_type'].str.contains('resistance|poc|confluence', case=False, na=False)]
    
    support_levels = supports['level_price'].to_numpy()
    resistance_levels = resistances['level_price'].to_numpy()
    support_scores = pd.Series(supports['score'].values, index=supports['level_price']).to_dict()
    resistance_scores = pd.Series(resistances['score'].values, index=resistances['level_price']).to_dict()

    results = []
    for price in df['close']:
        dist_support, score_support, dist_resistance, score_resistance = 1.0, 0.0, 1.0, 0.0
        
        if support_levels.size > 0:
            diffs = price - support_levels
            below_price = diffs[diffs >= 0]
            if below_price.size > 0:
                nearest_sup_price = support_levels[diffs == below_price.min()][0]
                dist_support = (price - nearest_sup_price) / price if price > 0 else 0
                score_support = support_scores.get(nearest_sup_price, 0)
        
        if resistance_levels.size > 0:
            diffs = resistance_levels - price
            above_price = diffs[diffs >= 0]
            if above_price.size > 0:
                nearest_res_price = resistance_levels[diffs == above_price.min()][0]
                dist_resistance = (nearest_res_price - price) / price if price > 0 else 0
                score_resistance = resistance_scores.get(nearest_res_price, 0)
                
        results.append((dist_support, score_support, dist_resistance, score_resistance))

    df[['dist_to_support', 'score_of_support', 'dist_to_resistance', 'score_of_resistance']] = results
    return df

def calculate_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
    
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    
    return df_calc.astype('float32', errors='ignore')

# ---------------------- دوال WebSocket والاستراتيجية ----------------------
def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
    global open_signals_cache, current_prices
    try:
        data = msg.get('data', msg) if isinstance(msg, dict) else msg
        if not isinstance(data, list): data = [data]
        
        for item in data:
            symbol = item.get('s')
            if not symbol: continue
            price = float(item.get('c', 0))
            if price == 0: continue
            with prices_lock: current_prices[symbol] = price
            signal_to_process, status, closing_price = None, None, None
            
            with signal_cache_lock:
                if symbol in open_signals_cache:
                    signal = open_signals_cache[symbol]
                    target_price = signal.get('target_price'); stop_loss_price = signal.get('stop_loss')
                    if not all(isinstance(p, (int, float)) for p in [price, target_price, stop_loss_price]): continue
                    if price >= target_price: status, closing_price, signal_to_process = 'target_hit', target_price, signal
                    elif price <= stop_loss_price: status, closing_price, signal_to_process = 'stop_loss_hit', stop_loss_price, signal
            
            if signal_to_process and status:
                logger.info(f"⚡ [المتتبع الفوري] تم تفعيل حدث '{status}' للعملة {symbol} عند سعر {price:.8f}")
                Thread(target=close_signal, args=(signal_to_process, status, closing_price, "auto")).start()
    except Exception as e:
        logger.error(f"❌ [متتبع WebSocket] خطأ في معالجة رسالة السعر الفورية: {e}", exc_info=True)

def run_websocket_manager() -> None:
    logger.info("ℹ️ [WebSocket] بدء مدير WebSocket...")
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start(); twm.start_ticker_socket(callback=handle_ticker_message)
    logger.info("✅ [WebSocket] تم الاتصال والاستماع بنجاح."); twm.join()

class TradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_from_github(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame, sr_levels_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            df_featured = calculate_base_features(df_15m)
            df_featured = calculate_sr_features(df_featured, sr_levels_df)
            
            df_featured['returns'] = df_featured['close'].pct_change()
            merged = df_featured.join(btc_df['btc_returns']).fillna(0)
            df_featured['btc_correlation'] = merged['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged['btc_returns'])
            
            df_featured['rsi_4h'] = calculate_base_features(df_4h)['rsi']
            df_featured.fillna(method='ffill', inplace=True)
            
            for col in self.feature_names:
                if col not in df_featured.columns: df_featured[col] = 0.0
            
            return df_featured[self.feature_names].dropna()
        except Exception as e:
            logger.error(f"❌ [{self.symbol}] فشل هندسة الميزات: {e}", exc_info=True)
            return None

    def generate_signal(self, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]):
            logger.debug(f"[{self.symbol}] Skipping signal generation: model/scaler/features not loaded.")
            return None
        if df_features.empty:
            return None
        
        last_row_df = df_features.iloc[[-1]]
        try:
            features_scaled = self.scaler.transform(last_row_df)
            features_scaled_df = pd.DataFrame(features_scaled, columns=self.feature_names)
            
            prediction = self.ml_model.predict(features_scaled_df)[0]
            prediction_proba = self.ml_model.predict_proba(features_scaled_df)[0]
            
            try:
                class_1_index = list(self.ml_model.classes_).index(1)
            except ValueError:
                return None
            prob_for_class_1 = prediction_proba[class_1_index]

            if prediction == 1 and prob_for_class_1 >= MODEL_CONFIDENCE_THRESHOLD:
                logger.info(f"✅ [العثور على إشارة] {self.symbol}: تنبأ النموذج 'شراء' (1) بثقة {prob_for_class_1:.2%}, وهي أعلى من الحد المطلوب ({MODEL_CONFIDENCE_THRESHOLD:.0%}).")
                return {'symbol': self.symbol, 'strategy_name': BASE_ML_MODEL_NAME, 'signal_details': {'ML_Probability_Buy': f"{prob_for_class_1:.2%}"}}
            return None
        except Exception as e:
            logger.warning(f"⚠️ [توليد إشارة] {self.symbol}: خطأ أثناء التوليد: {e}", exc_info=True)
            return None

# ---------------------- دوال التنبيهات والإدارة ----------------------
def send_telegram_message(target_chat_id: str, text: str):
    if not TELEGRAM_TOKEN or not target_chat_id: return
    url = f"[https://api.telegram.org/bot](https://api.telegram.org/bot){TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(target_chat_id), 'text': text, 'parse_mode': 'Markdown'}
    try: requests.post(url, json=payload, timeout=20).raise_for_status()
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

def send_new_signal_alert(signal_data: Dict[str, Any]) -> None:
    safe_symbol = signal_data['symbol'].replace('_', '\\_')
    entry, target, sl = signal_data['entry_price'], signal_data['target_price'], signal_data['stop_loss']
    profit_pct = ((target / entry) - 1) * 100
    message = (f"💡 *إشارة تداول جديدة ({BASE_ML_MODEL_NAME})* 💡\n\n"
               f"🪙 *العملة:* `{safe_symbol}`\n📈 *النوع:* شراء (LONG)\n\n"
               f"⬅️ *سعر الدخول:* `${entry:,.8g}`\n"
               f"🎯 *الهدف:* `${target:,.8g}` (ربح متوقع `{profit_pct:+.2f}%`)\n"
               f"🛑 *وقف الخسارة:* `${sl:,.8g}`\n\n"
               f"🔍 *ثقة النموذج:* {signal_data['signal_details']['ML_Probability_Buy']}")
    reply_markup = {"inline_keyboard": [[{"text": "📊 فتح لوحة التحكم", "url": WEBHOOK_URL or '#'}]]}
    url = f"[https://api.telegram.org/bot](https://api.telegram.org/bot){TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(CHAT_ID), 'text': message, 'parse_mode': 'Markdown', 'reply_markup': json.dumps(reply_markup)}
    try: requests.post(url, json=payload, timeout=20).raise_for_status()
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال تنبيه الإشارة الجديدة: {e}")
    log_and_notify('info', f"إشارة جديدة: {signal_data['symbol']} بسعر دخول ${entry:,.8g}", "NEW_SIGNAL")

def insert_signal_into_db(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not check_db_connection() or not conn: return None
    try:
        entry, target, sl = float(signal['entry_price']), float(signal['target_price']), float(signal['stop_loss'])
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;",
                (signal['symbol'], entry, target, sl, signal.get('strategy_name'), json.dumps(signal.get('signal_details', {})))
            )
            signal['id'] = cur.fetchone()['id']
        conn.commit()
        logger.info(f"✅ [قاعدة البيانات] تم إدراج الإشارة لـ {signal['symbol']} (ID: {signal['id']}).")
        return signal
    except Exception as e:
        logger.error(f"❌ [إدراج في قاعدة البيانات] خطأ في إدراج إشارة {signal['symbol']}: {e}", exc_info=True)
        if conn: conn.rollback(); return None

def close_signal(signal: Dict, status: str, closing_price: float, closed_by: str):
    symbol = signal['symbol']
    with signal_cache_lock:
        if symbol not in open_signals_cache or open_signals_cache[symbol]['id'] != signal['id']: return
    if not check_db_connection() or not conn: return
    try:
        db_closing_price = float(closing_price)
        db_profit_pct = float(((db_closing_price / signal['entry_price']) - 1) * 100)
        with conn.cursor() as update_cur:
            update_cur.execute(
                "UPDATE signals SET status = %s, closing_price = %s, closed_at = NOW(), profit_percentage = %s WHERE id = %s;",
                (status, db_closing_price, db_profit_pct, signal['id'])
            )
        conn.commit()
        with signal_cache_lock: del open_signals_cache[symbol]
        status_map = {'target_hit': '✅ تحقق الهدف', 'stop_loss_hit': '🛑 ضرب وقف الخسارة', 'manual_close': '🖐️ أُغلقت يدوياً'}
        status_message = status_map.get(status, status.replace('_', ' ').title())
        safe_symbol = signal['symbol'].replace('_', '\\_')
        alert_msg_tg = f"*{status_message}*\n`{safe_symbol}` | *الربح:* `{db_profit_pct:+.2f}%`"
        send_telegram_message(CHAT_ID, alert_msg_tg)
        alert_msg_db = f"{status_message}: {signal['symbol']} | الربح: {db_profit_pct:+.2f}%"
        log_and_notify('info', alert_msg_db, 'CLOSE_SIGNAL')
    except Exception as e:
        logger.error(f"❌ [إغلاق قاعدة البيانات] خطأ فادح أثناء إغلاق الإشارة {signal['id']} لـ {signal['symbol']}: {e}", exc_info=True)
        if conn: conn.rollback()

def load_open_signals_to_cache():
    if not check_db_connection() or not conn: return
    logger.info("ℹ️ [تحميل الذاكرة المؤقتة] جاري تحميل الإشارات المفتوحة سابقاً...")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status = 'open';")
            open_signals = cur.fetchall()
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in open_signals: open_signals_cache[signal['symbol']] = dict(signal)
            logger.info(f"✅ [تحميل الذاكرة المؤقتة] تم تحميل {len(open_signals)} إشارة مفتوحة.")
    except Exception as e: logger.error(f"❌ [تحميل الذاكرة المؤقتة] فشل تحميل الإشارات المفتوحة: {e}")

def load_notifications_to_cache():
    if not check_db_connection() or not conn: return
    logger.info("ℹ️ [تحميل الذاكرة المؤقتة] جاري تحميل آخر التنبيهات...")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 50;")
            recent = cur.fetchall()
            with notifications_lock:
                notifications_cache.clear()
                for n in reversed(recent): n['timestamp'] = n['timestamp'].isoformat(); notifications_cache.appendleft(dict(n))
            logger.info(f"✅ [تحميل الذاكرة المؤقتة] تم تحميل {len(notifications_cache)} تنبيه.")
    except Exception as e: logger.error(f"❌ [تحميل الذاكرة المؤقتة] فشل تحميل التنبيهات: {e}")

# ---------------------- حلقة العمل الرئيسية ----------------------
def get_btc_trend() -> Dict[str, Any]:
    if not client: return {"status": "error", "message": "Binance client not initialized", "is_uptrend": False}
    try:
        klines = client.get_klines(symbol=BTC_SYMBOL, interval=BTC_TREND_TIMEFRAME, limit=BTC_TREND_EMA_PERIOD * 2)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df['close'] = pd.to_numeric(df['close'])
        ema = df['close'].ewm(span=BTC_TREND_EMA_PERIOD, adjust=False).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        status, message = ("Uptrend", f"صاعد (السعر فوق EMA {BTC_TREND_EMA_PERIOD})") if current_price > ema else ("Downtrend", f"هابط (السعر تحت EMA {BTC_TREND_EMA_PERIOD})")
        return {"status": status, "message": message, "is_uptrend": (status == "Uptrend")}
    except Exception as e:
        logger.error(f"❌ [فلتر BTC] فشل تحديد اتجاه البيتكوين: {e}")
        return {"status": "Error", "message": str(e), "is_uptrend": False}

def get_btc_data_for_bot() -> Optional[pd.DataFrame]:
    logger.info("ℹ️ [بيانات BTC] جاري جلب بيانات البيتكوين لحساب المؤشرات...")
    btc_data = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
    if btc_data is None:
        logger.error("❌ [بيانات BTC] فشل جلب بيانات البيتكوين. سيتخطى البوت الارتباط.")
        return None
    btc_data['btc_returns'] = btc_data['close'].pct_change()
    return btc_data

def main_loop():
    logger.info("[الحلقة الرئيسية] انتظار اكتمال التهيئة الأولية...")
    time.sleep(15) 
    if not validated_symbols_to_scan:
        log_and_notify("critical", "لا توجد رموز معتمدة للمسح. لن يستمر البوت في العمل.", "SYSTEM"); return
    log_and_notify("info", f"بدء حلقة المسح الرئيسية لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")
    
    while True:
        try:
            if USE_BTC_TREND_FILTER:
                trend_data = get_btc_trend()
                if not trend_data.get("is_uptrend"):
                    logger.warning(f"⚠️ [إيقاف المسح] تم إيقاف البحث عن إشارات شراء بسبب الاتجاه الهابط للبيتكوين. {trend_data.get('message')}")
                    time.sleep(300); continue

            with signal_cache_lock: open_count = len(open_signals_cache)
            if open_count >= MAX_OPEN_TRADES:
                logger.info(f"ℹ️ [إيقاف مؤقت] تم الوصول للحد الأقصى للصفقات ({open_count}/{MAX_OPEN_TRADES}).")
                time.sleep(60); continue
            
            slots_available = MAX_OPEN_TRADES - open_count
            logger.info(f"ℹ️ [بدء المسح] بدء دورة مسح جديدة. المراكز المتاحة: {slots_available}")
            
            btc_data = get_btc_data_for_bot()
            if btc_data is None:
                logger.error("❌ فشل حاسم في جلب بيانات BTC. سيتم تخطي دورة المسح هذه."); time.sleep(120); continue
            
            for symbol in validated_symbols_to_scan:
                if slots_available <= 0: break
                with signal_cache_lock:
                    if symbol in open_signals_cache: continue
                
                try:
                    df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_15m is None or df_15m.empty or df_4h is None or df_4h.empty: continue
                    
                    sr_levels = fetch_sr_levels_from_db(symbol)
                    
                    strategy = TradingStrategy(symbol)
                    if not strategy.ml_model:
                        logger.debug(f"[{symbol}] Skipping due to model not being loaded from GitHub.")
                        continue
                        
                    df_features = strategy.get_features(df_15m, df_4h, btc_data, sr_levels)
                    
                    del df_15m, df_4h, sr_levels; gc.collect()
                    
                    if df_features is None or df_features.empty: continue
                    
                    potential_signal = strategy.generate_signal(df_features)
                    if potential_signal:
                        with prices_lock: current_price = current_prices.get(symbol)
                        if not current_price:
                             logger.warning(f"⚠️ {symbol}: لا يمكن الحصول على السعر الحالي. سيتم التخطي."); continue
                        
                        potential_signal['entry_price'] = current_price
                        if USE_DYNAMIC_SL_TP:
                            atr_value = df_features['atr'].iloc[-1]
                            potential_signal['stop_loss'] = current_price - (atr_value * ATR_SL_MULTIPLIER)
                            potential_signal['target_price'] = current_price + (atr_value * ATR_TP_MULTIPLIER)
                        else:
                            potential_signal['target_price'] = current_price * 1.02; potential_signal['stop_loss'] = current_price * 0.985
                        
                        saved_signal = insert_signal_into_db(potential_signal)
                        if saved_signal:
                            with signal_cache_lock: open_signals_cache[saved_signal['symbol']] = saved_signal
                            send_new_signal_alert(saved_signal)
                            slots_available -= 1
                except Exception as e:
                    logger.error(f"❌ [خطأ في المعالجة] حدث خطأ أثناء معالجة العملة {symbol}: {e}", exc_info=True)

            logger.info("ℹ️ [نهاية المسح] انتهت دورة المسح. في انتظار الدورة التالية..."); time.sleep(60)
        except (KeyboardInterrupt, SystemExit): break
        except Exception as main_err:
            log_and_notify("error", f"خطأ غير متوقع في الحلقة الرئيسية: {main_err}", "SYSTEM"); time.sleep(120)

# ---------------------- واجهة برمجة تطبيقات Flask (بدون تغيير) ----------------------
app = Flask(__name__)
CORS(app)

def get_fear_and_greed_index() -> Dict[str, Any]:
    classification_translation = {"Extreme Fear": "خوف شديد", "Fear": "خوف", "Neutral": "محايد", "Greed": "طمع", "Extreme Greed": "طمع شديد", "Error": "خطأ"}
    try:
        response = requests.get("[https://api.alternative.me/fng/?limit=1](https://api.alternative.me/fng/?limit=1)", timeout=10)
        response.raise_for_status()
        data = response.json()['data'][0]; original = data['value_classification']
        return {"value": int(data['value']), "classification": classification_translation.get(original, original)}
    except Exception as e:
        logger.error(f"❌ [مؤشر الخوف والطمع] فشل الاتصال بالـ API: {e}")
        return {"value": -1, "classification": classification_translation["Error"]}

@app.route('/')
def home():
    try:
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, 'index.html')
        with open(file_path, 'r', encoding='utf-8') as f: return render_template_string(f.read())
    except FileNotFoundError: return "<h1>ملف لوحة التحكم (index.html) غير موجود.</h1>", 404

@app.route('/api/market_status')
def get_market_status(): return jsonify({"btc_trend": get_btc_trend(), "fear_and_greed": get_fear_and_greed_index()})

@app.route('/api/stats')
def get_stats():
    if not check_db_connection() or not conn: return jsonify({"error": "فشل الاتصال بقاعدة البيانات"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT status, profit_percentage FROM signals WHERE status != 'open';")
            closed = cur.fetchall()
        wins = sum(1 for s in closed if s.get('profit_percentage', 0) > 0); losses = len(closed) - wins
        total_closed = len(closed); win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
        total_profit = sum(s['profit_percentage'] / 100 * TRADE_AMOUNT_USDT for s in closed if s.get('profit_percentage') is not None)
        return jsonify({"win_rate": win_rate, "wins": wins, "losses": losses, "total_profit_usdt": total_profit, "total_closed_trades": total_closed})
    except Exception as e:
        logger.error(f"❌ [API إحصائيات] خطأ: {e}"); return jsonify({"error": "تعذر جلب الإحصائيات"}), 500

@app.route('/api/signals')
def get_signals():
    if not check_db_connection() or not conn: return jsonify({"error": "فشل الاتصال بقاعدة البيانات"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals ORDER BY CASE WHEN status = 'open' THEN 0 ELSE 1 END, id DESC;")
            all_signals = cur.fetchall()
        for s in all_signals:
            if s.get('closed_at'): s['closed_at'] = s['closed_at'].isoformat()
            if s['status'] == 'open':
                with prices_lock: s['current_price'] = current_prices.get(s['symbol'])
        return jsonify(all_signals)
    except Exception as e:
        logger.error(f"❌ [API إشارات] خطأ: {e}"); return jsonify({"error": "تعذر جلب الإشارات"}), 500

@app.route('/api/close/<int:signal_id>', methods=['POST'])
def manual_close_signal(signal_id):
    logger.info(f"ℹ️ [API إغلاق] تم استلام طلب إغلاق يدوي للإشارة ID: {signal_id}")
    signal_to_close = None
    with signal_cache_lock:
        for s in open_signals_cache.values():
            if s['id'] == signal_id: signal_to_close = s.copy(); break
    if not signal_to_close: return jsonify({"error": "لم يتم العثور على الإشارة."}), 404
    symbol_to_close = signal_to_close['symbol']
    with prices_lock: closing_price = current_prices.get(symbol_to_close)
    if not closing_price: return jsonify({"error": f"تعذر الحصول على السعر الحالي لـ {symbol_to_close}."}), 500
    Thread(target=close_signal, args=(signal_to_close, 'manual_close', closing_price, "manual")).start()
    return jsonify({"message": f"جاري إغلاق الإشارة {signal_id} لـ {symbol_to_close}."})

@app.route('/api/notifications')
def get_notifications():
    with notifications_lock: return jsonify(list(notifications_cache))

def run_flask():
    host, port = "0.0.0.0", int(os.environ.get('PORT', 10000))
    log_and_notify("info", f"بدء تشغيل لوحة التحكم على {host}:{port}", "SYSTEM")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        logger.warning("⚠️ [Flask] مكتبة 'waitress' غير موجودة, سيتم استخدام خادم التطوير.")
        app.run(host=host, port=port)

# ---------------------- نقطة انطلاق البرنامج ----------------------
def initialize_bot_services():
    global client, validated_symbols_to_scan
    logger.info("🤖 [خدمات البوت] بدء التهيئة في الخلفية...")
    try:
        client = Client(API_KEY, API_SECRET)
        logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
        
        init_github_repo()
        
        init_db()
        load_open_signals_to_cache(); load_notifications_to_cache()
        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan:
            logger.critical("❌ لا توجد رموز معتمدة للمسح. الحلقات لن تبدأ."); return
        Thread(target=run_websocket_manager, daemon=True).start()
        Thread(target=main_loop, daemon=True).start()
        logger.info("✅ [خدمات البوت] تم بدء جميع خدمات الخلفية بنجاح.")
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حاسم أثناء التهيئة: {e}", "SYSTEM")
        pass

if __name__ == "__main__":
    logger.info(f"🚀 بدء تشغيل بوت التداول - إصدار {BASE_ML_MODEL_NAME}...")
    initialization_thread = Thread(target=initialize_bot_services, daemon=True)
    initialization_thread.start()
    run_flask()
    logger.info("👋 [إيقاف] تم إيقاف تشغيل البوت."); os._exit(0)
