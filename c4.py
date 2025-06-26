import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
import gc
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException
from flask import Flask, request, Response, jsonify, render_template_string
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timedelta, UTC
from decouple import config
from typing import List, Dict, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler
from collections import deque

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v5_advanced_sr.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV5_AdvancedSR')

# ---------------------- تحميل متغيرات البيئة ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
except Exception as e:
     logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
     exit(1)

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V5'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
DATA_FETCH_LOOKBACK_DAYS: int = 15

# --- Indicator & Feature Parameters ---
ADX_PERIOD: int = 14
BBANDS_PERIOD: int = 20
RSI_PERIOD: int = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD: int = 14
EMA_SLOW_PERIOD: int = 200
EMA_FAST_PERIOD: int = 50
BTC_CORR_PERIOD: int = 30
STOCH_RSI_PERIOD: int = 14
STOCH_K: int = 3
STOCH_D: int = 3
REL_VOL_PERIOD: int = 30
RSI_OVERBOUGHT: int = 70
RSI_OVERSOLD: int = 30
STOCH_RSI_OVERBOUGHT: int = 80
STOCH_RSI_OVERSOLD: int = 20
BTC_SYMBOL = 'BTCUSDT'

# --- Trading Logic Constants ---
MODEL_CONFIDENCE_THRESHOLD = 0.80
MAX_OPEN_TRADES: int = 5
USE_SR_LEVELS = True
MINIMUM_SR_SCORE = 30
ATR_SL_MULTIPLIER = 2.0
ATR_TP_MULTIPLIER = 2.5
USE_BTC_TREND_FILTER = True
BTC_TREND_TIMEFRAME = '4h'
BTC_TREND_EMA_PERIOD = 10

# --- ثوابت فلترة الصفقات ---
MINIMUM_PROFIT_PERCENTAGE = 0.5
MINIMUM_RISK_REWARD_RATIO = 1.2
MINIMUM_15M_VOLUME_USDT = 10_000

# --- المتغيرات العامة وقفل العمليات ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
current_prices: Dict[str, float] = {}
prices_lock = Lock()
notifications_cache = deque(maxlen=50)
notifications_lock = Lock()


# ---------------------- دوال قاعدة البيانات (مُعدَّلة) ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    global conn
    logger.info("[قاعدة البيانات] بدء تهيئة الاتصال...")
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signals ( id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION NOT NULL,
                        target_price DOUBLE PRECISION NOT NULL, stop_loss DOUBLE PRECISION NOT NULL, status TEXT DEFAULT 'open',
                        closing_price DOUBLE PRECISION, closed_at TIMESTAMP, profit_percentage DOUBLE PRECISION,
                        strategy_name TEXT, signal_details JSONB );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications ( id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL, message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE );
                """)
                cur.execute("""
                     CREATE TABLE IF NOT EXISTS ml_models ( id SERIAL PRIMARY KEY, model_name TEXT NOT NULL UNIQUE,
                        model_data BYTEA NOT NULL, trained_at TIMESTAMP DEFAULT NOW(), metrics JSONB );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS support_resistance_levels ( id SERIAL PRIMARY KEY, symbol TEXT NOT NULL,
                        level_price DOUBLE PRECISION NOT NULL, level_type TEXT NOT NULL, timeframe TEXT NOT NULL,
                        strength NUMERIC NOT NULL, score NUMERIC DEFAULT 0, last_tested_at TIMESTAMP WITH TIME ZONE,
                        details TEXT, created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        CONSTRAINT unique_level UNIQUE (symbol, level_price, timeframe, level_type) );
                """)
                cur.execute("SELECT 1 FROM information_schema.columns WHERE table_name='support_resistance_levels' AND column_name='score'")
                if not cur.fetchone():
                    cur.execute("ALTER TABLE support_resistance_levels ADD COLUMN score NUMERIC DEFAULT 0;")

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS pending_recommendations (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL UNIQUE,
                        original_entry_price DOUBLE PRECISION NOT NULL,
                        original_target_price DOUBLE PRECISION NOT NULL,
                        trigger_price DOUBLE PRECISION NOT NULL,
                        atr_at_creation DOUBLE PRECISION,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        signal_details JSONB
                    );
                """)

            conn.commit()
            logger.info("✅ [قاعدة البيانات] تم تهيئة جميع جداول قاعدة البيانات بنجاح (بما في ذلك pending_recommendations).")
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
        if conn:
            with conn.cursor() as cur:
                 cur.execute("SELECT 1;")
            return True
        return False
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [قاعدة البيانات] فقدان الاتصال: {e}. محاولة إعادة الاتصال...")
        try:
            init_db()
            return conn is not None and conn.closed == 0
        except Exception as retry_e:
            logger.error(f"❌ [قاعدة البيانات] فشل إعادة الاتصال: {retry_e}")
            return False
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

def fetch_sr_levels(symbol: str) -> Optional[List[Dict]]:
    if not check_db_connection() or not conn:
        logger.warning(f"⚠️ [{symbol}] لا يمكن جلب مستويات الدعم والمقاومة، اتصال قاعدة البيانات غير متاح.")
        return None
    try:
        with conn.cursor() as cur:
            cur.execute( "SELECT level_price, level_type, score FROM support_resistance_levels WHERE symbol = %s ORDER BY level_price ASC", (symbol,))
            levels = cur.fetchall()
            if not levels: return None
            for level in levels: level['score'] = float(level.get('score', 0))
            return levels
    except Exception as e:
        logger.error(f"❌ [{symbol}] خطأ أثناء جلب مستويات الدعم والمقاومة: {e}")
        if conn: conn.rollback()
        return None

# ---------------------- دوال Binance والبيانات (بدون تغيير) ----------------------
def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    logger.info(f"ℹ️ [التحقق] قراءة الرموز من '{filename}' والتحقق منها مع Binance...")
    if not client: logger.error("❌ [التحقق] كائن Binance client غير مهيأ."); return []
    try:
        script_dir = os.path.dirname(__file__)
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
        # *** FIX: Use datetime.now(UTC) instead of deprecated datetime.utcnow() ***
        start_str = (datetime.now(UTC) - timedelta(days=days + 1)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in numeric_cols:
            if df[col].dtype == 'float64': df[col] = df[col].astype('float32')
        return df[numeric_cols].dropna()
    except BinanceAPIException as e:
        logger.warning(f"⚠️ [API Binance] خطأ في جلب بيانات {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ [البيانات] خطأ أثناء جلب البيانات التاريخية لـ {symbol}: {e}")
        return None

def calculate_all_features(df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    df_calc = df_15m.copy()
    high_low = df_calc['high'] - df_calc['low']; high_close = (df_calc['high'] - df_calc['close'].shift()).abs(); low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    up_move = df_calc['high'].diff(); down_move = -df_calc['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr']
    minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr']
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean(); loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    ema_fast_macd = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean(); ema_slow_macd = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast_macd - ema_slow_macd; signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df_calc['macd_hist'] = macd_line - signal_line
    df_calc['macd_cross'] = 0
    df_calc.loc[(df_calc['macd_hist'].shift(1) < 0) & (df_calc['macd_hist'] >= 0), 'macd_cross'] = 1
    df_calc.loc[(df_calc['macd_hist'].shift(1) > 0) & (df_calc['macd_hist'] <= 0), 'macd_cross'] = -1
    sma = df_calc['close'].rolling(window=BBANDS_PERIOD).mean(); std_dev = df_calc['close'].rolling(window=BBANDS_PERIOD).std()
    upper_band = sma + (std_dev * 2); lower_band = sma - (std_dev * 2)
    df_calc['bb_width'] = (upper_band - lower_band) / (sma + 1e-9)
    rsi_stoch = df_calc['rsi']; min_rsi = rsi_stoch.rolling(window=STOCH_RSI_PERIOD).min(); max_rsi = rsi_stoch.rolling(window=STOCH_RSI_PERIOD).max()
    stoch_rsi_val = (rsi_stoch - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(window=STOCH_K).mean() * 100
    df_calc['stoch_rsi_d'] = df_calc['stoch_rsi_k'].rolling(window=STOCH_D).mean()
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)
    df_calc['market_condition'] = 0
    df_calc.loc[(df_calc['rsi'] > 70) | (df_calc['stoch_rsi_k'] > 80), 'market_condition'] = 1
    df_calc.loc[(df_calc['rsi'] < 30) | (df_calc['stoch_rsi_k'] < 20), 'market_condition'] = -1
    ema_fast_trend = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean(); ema_slow_trend = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    df_calc['price_vs_ema50'] = (df_calc['close'] / ema_fast_trend) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / ema_slow_trend) - 1
    df_calc['returns'] = df_calc['close'].pct_change()
    merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
    df_calc['btc_correlation'] = merged_df['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    df_calc['hour_of_day'] = df_calc.index.hour
    df_calc = calculate_candlestick_patterns(df_calc)
    delta_4h = df_4h['close'].diff()
    gain_4h = delta_4h.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean(); loss_4h = -delta_4h.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_4h['rsi_4h'] = 100 - (100 / (1 + (gain_4h / loss_4h.replace(0, 1e-9))))
    ema_fast_4h = df_4h['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    df_4h['price_vs_ema50_4h'] = (df_4h['close'] / ema_fast_4h) - 1
    mtf_features = df_4h[['rsi_4h', 'price_vs_ema50_4h']]
    df_featured = df_calc.join(mtf_features)
    # *** FIX: Use .ffill() instead of deprecated .fillna(method='ffill') ***
    df_featured[['rsi_4h', 'price_vs_ema50_4h']] = df_featured[['rsi_4h', 'price_vs_ema50_4h']].ffill()
    return df_featured.dropna()

def calculate_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df_patterns = df.copy()
    op, hi, lo, cl = df_patterns['open'], df_patterns['high'], df_patterns['low'], df_patterns['close']
    body = abs(cl - op); candle_range = hi - lo; candle_range[candle_range == 0] = 1e-9
    upper_wick = hi - pd.concat([op, cl], axis=1).max(axis=1)
    lower_wick = pd.concat([op, cl], axis=1).min(axis=1) - lo
    df_patterns['candlestick_pattern'] = 0
    df_patterns.loc[(body / candle_range) < 0.05, 'candlestick_pattern'] = 3
    df_patterns.loc[(body > candle_range * 0.1) & (lower_wick >= body * 2) & (upper_wick < body), 'candlestick_pattern'] = 2
    df_patterns.loc[(body > candle_range * 0.1) & (upper_wick >= body * 2) & (lower_wick < body), 'candlestick_pattern'] = -2
    df_patterns.loc[(cl.shift(1) < op.shift(1)) & (cl > op) & (cl >= op.shift(1)) & (op <= cl.shift(1)) & (body > body.shift(1)), 'candlestick_pattern'] = 1
    df_patterns.loc[(cl.shift(1) > op.shift(1)) & (cl < op) & (op >= cl.shift(1)) & (cl <= op.shift(1)) & (body > body.shift(1)), 'candlestick_pattern'] = -1
    df_patterns.loc[(cl > op) & (body / candle_range > 0.95) & (upper_wick < body * 0.1) & (lower_wick < body * 0.1), 'candlestick_pattern'] = 4
    df_patterns.loc[(op > cl) & (body / candle_range > 0.95) & (upper_wick < body * 0.1) & (lower_wick < body * 0.1), 'candlestick_pattern'] = -4
    return df_patterns

def load_ml_model_bundle_from_folder(symbol: str) -> Optional[Dict[str, Any]]:
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    model_dir = 'Mo'
    file_path = os.path.join(model_dir, f"{model_name}.pkl")
    if not os.path.isdir(model_dir):
        logger.warning(f"⚠️ [نموذج تعلم الآلة] مجلد النماذج '{model_dir}' غير موجود.")
        return None
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                model_bundle = pickle.load(f)
            if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
                return model_bundle
            else:
                logger.error(f"❌ [نموذج تعلم الآلة] حزمة النموذج في الملف '{file_path}' غير مكتملة.")
                return None
        except Exception as e:
            logger.error(f"❌ [نموذج تعلم الآلة] خطأ عند تحميل النموذج '{file_path}': {e}", exc_info=True)
            return None
    else:
        return None

# ---------------------- دوال WebSocket والاستراتيجية (بدون تغيير) ----------------------
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
                    target_price = signal.get('target_price')
                    stop_loss_price = signal.get('stop_loss')
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
    twm.start()
    twm.start_ticker_socket(callback=handle_ticker_message)
    logger.info("✅ [WebSocket] تم الاتصال والاستماع بنجاح.")
    twm.join()

class TradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_bundle_from_folder(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        return calculate_all_features(df_15m, df_4h, btc_df)

    def generate_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]):
            return None
        last_row = df_processed.iloc[-1]
        try:
            missing_features = [f for f in self.feature_names if f not in df_processed.columns]
            if missing_features: return None
            features_df = pd.DataFrame([last_row], columns=df_processed.columns)[self.feature_names]
            if features_df.isnull().values.any(): return None
            features_scaled = self.scaler.transform(features_df)
            features_scaled_df = pd.DataFrame(features_scaled, columns=self.feature_names)
            prediction = self.ml_model.predict(features_scaled_df)[0]
            prediction_proba = self.ml_model.predict_proba(features_scaled_df)[0]
            prob_for_class_1 = 0
            try:
                class_1_index = list(self.ml_model.classes_).index(1)
                prob_for_class_1 = prediction_proba[class_1_index]
            except ValueError:
                return None
            if prediction == 1 and prob_for_class_1 >= MODEL_CONFIDENCE_THRESHOLD:
                logger.info(f"✅ [العثور على إشارة أولية] {self.symbol}: تنبأ النموذج 'شراء' بثقة {prob_for_class_1:.2%}.")
                return { 'symbol': self.symbol, 'strategy_name': BASE_ML_MODEL_NAME,
                    'signal_details': {'ML_Probability_Buy': f"{prob_for_class_1:.2%}"} }
            else:
                return None
        except Exception as e:
            logger.warning(f"⚠️ [توليد إشارة] {self.symbol}: خطأ أثناء التوليد: {e}", exc_info=True)
            return None

# ---------------------- دوال التنبيهات والإدارة (مُعدَّلة) ----------------------
def send_telegram_message(target_chat_id: str, text: str):
    if not TELEGRAM_TOKEN or not target_chat_id: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(target_chat_id), 'text': text, 'parse_mode': 'Markdown'}
    try: requests.post(url, json=payload, timeout=20).raise_for_status()
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

def send_new_signal_alert(signal_data: Dict[str, Any]) -> None:
    safe_symbol = signal_data['symbol'].replace('_', '\\_')
    entry = signal_data['entry_price']
    sl = signal_data['stop_loss']
    signal_details = signal_data.get('signal_details', {})

    if 'TP1' in signal_details and 'TP2' in signal_details:
        target1 = signal_details['TP1']
        target2 = signal_details['TP2']
        profit_pct1 = ((target1 / entry) - 1) * 100 if entry > 0 else 0
        profit_pct2 = ((target2 / entry) - 1) * 100 if entry > 0 else 0
        target_text = (f"🎯 *الهدف 1:* `${target1:,.8g}` (ربح `{profit_pct1:+.2f}%`)\n"
                       f"🎯 *الهدف 2:* `${target2:,.8g}` (ربح `{profit_pct2:+.2f}%`)")
    else:
        target = signal_data['target_price']
        profit_pct = ((target / entry) - 1) * 100 if entry > 0 else 0
        target_text = f"🎯 *الهدف:* `${target:,.8g}` (ربح متوقع `{profit_pct:+.2f}%`)"

    rr_ratio_info = signal_details.get('risk_reward_ratio', 'N/A')
    volume_info = signal_details.get('last_15m_volume_usdt', 'N/A')
    strategy_name = signal_data.get('strategy_name', BASE_ML_MODEL_NAME)

    message = (f"💡 *إشارة تداول جديدة ({strategy_name})* 💡\n\n"
               f"🪙 *العملة:* `{safe_symbol}`\n"
               f"📈 *النوع:* شراء (LONG)\n\n"
               f"⬅️ *سعر الدخول:* `${entry:,.8g}`\n"
               f"{target_text}\n"
               f"🛑 *وقف الخسارة:* `${sl:,.8g}`\n\n"
               f"💧 *سيولة آخر 15د:* `{volume_info}`\n"
               f"🔍 *ثقة النموذج:* {signal_details.get('ML_Probability_Buy', 'N/A')}\n"
               f"⚖️ *المخاطرة/العائد:* `{rr_ratio_info}`\n"
               f"🛠️ *أساس الهدف/الوقف:* {signal_details.get('sr_info', 'ATR Default')}")

    reply_markup = {"inline_keyboard": [[{"text": "📊 فتح لوحة التحكم", "url": WEBHOOK_URL or '#'}]]}
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
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
                """INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details)
                   VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;""",
                (signal['symbol'], entry, target, sl, signal.get('strategy_name'), json.dumps(signal.get('signal_details', {})))
            )
            signal['id'] = cur.fetchone()['id']
        conn.commit()
        logger.info(f"✅ [قاعدة البيانات] تم إدراج الإشارة لـ {signal['symbol']} (ID: {signal['id']}).")
        return signal
    except Exception as e:
        logger.error(f"❌ [إدراج في قاعدة البيانات] خطأ في إدراج إشارة {signal['symbol']}: {e}", exc_info=True)
        if conn: conn.rollback()
        return None

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
                for n in reversed(recent):
                    if 'timestamp' in n and isinstance(n['timestamp'], datetime):
                        n['timestamp'] = n['timestamp'].isoformat()
                    notifications_cache.appendleft(dict(n))
            logger.info(f"✅ [تحميل الذاكرة المؤقتة] تم تحميل {len(notifications_cache)} تنبيه.")
    except Exception as e: logger.error(f"❌ [تحميل الذاكرة المؤقتة] فشل تحميل التنبيهات: {e}")

def save_pending_recommendation(signal: Dict[str, Any]) -> None:
    if not check_db_connection() or not conn:
        logger.warning(f"⚠️ [{signal['symbol']}] لا يمكن حفظ توصية قيد الانتظار، اتصال قاعدة البيانات غير متاح.")
        return

    try:
        # *** FIX: Convert all potential numpy types to standard Python floats ***
        original_entry = float(signal['entry_price'])
        original_target = float(signal['target_price'])
        trigger_price_val = float(signal['stop_loss'])
        atr_value = signal['signal_details'].get('atr_value')
        atr_value_float = float(atr_value) if atr_value is not None else None

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pending_recommendations (
                    symbol, original_entry_price, original_target_price, trigger_price,
                    atr_at_creation, signal_details
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol) DO UPDATE SET
                    original_entry_price = EXCLUDED.original_entry_price,
                    original_target_price = EXCLUDED.original_target_price,
                    trigger_price = EXCLUDED.trigger_price,
                    atr_at_creation = EXCLUDED.atr_at_creation,
                    signal_details = EXCLUDED.signal_details,
                    created_at = NOW();
                """,
                (
                    signal['symbol'],
                    original_entry,
                    original_target,
                    trigger_price_val,
                    atr_value_float,
                    json.dumps(signal.get('signal_details', {}))
                )
            )
        conn.commit()
        logger.info(f"💾 [{signal['symbol']}] تم حفظ/تحديث توصية قيد الانتظار بنجاح.")
        log_and_notify('info', f"توصية قيد الانتظار جديدة لـ {signal['symbol']}", "PENDING_SIGNAL")
    except Exception as e:
        logger.error(f"❌ [حفظ توصية معلقة] خطأ في حفظ توصية {signal['symbol']}: {e}", exc_info=True)
        if conn: conn.rollback()

# ---------------------- حلقة العمل الرئيسية والجديدة ----------------------
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

def main_loop():
    logger.info("[الحلقة الرئيسية] انتظار اكتمال التهيئة الأولية...")
    time.sleep(15)
    if not validated_symbols_to_scan:
        log_and_notify("critical", "لا توجد رموز معتمدة للمسح. لن يستمر البوت في العمل.", "SYSTEM")
        return
    log_and_notify("info", f"بدء حلقة المسح الرئيسية (لتوليد توصيات معلقة) لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")

    while True:
        try:
            if USE_BTC_TREND_FILTER:
                trend_data = get_btc_trend()
                if not trend_data.get("is_uptrend"):
                    logger.warning(f"⚠️ [إيقاف المسح] تم إيقاف البحث عن إشارات شراء بسبب الاتجاه الهابط للبيتكوين. {trend_data.get('message')}")
                    time.sleep(300); continue

            btc_data_cycle = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
            if btc_data_cycle is None:
                logger.error("❌ فشل في جلب بيانات BTC. سيتم تخطي دورة المسح هذه.")
                time.sleep(120); continue
            btc_data_cycle['btc_returns'] = btc_data_cycle['close'].pct_change()

            for symbol in validated_symbols_to_scan:
                with signal_cache_lock:
                    if symbol in open_signals_cache: continue

                try:
                    df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
                    df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
                    if df_15m is None or df_15m.empty or df_4h is None or df_4h.empty: continue

                    strategy = TradingStrategy(symbol)
                    df_features = strategy.get_features(df_15m, df_4h, btc_data_cycle)
                    if df_features is None or df_features.empty:
                        del df_15m, df_4h, strategy; gc.collect(); continue

                    potential_signal = strategy.generate_signal(df_features)
                    if potential_signal:
                        last_candle = df_features.iloc[-1]
                        last_15m_volume_usdt = last_candle['volume'] * last_candle['close']
                        if last_15m_volume_usdt < MINIMUM_15M_VOLUME_USDT:
                            continue

                        potential_signal['signal_details']['last_15m_volume_usdt'] = f"${last_15m_volume_usdt:,.0f}"
                        with prices_lock: current_price = current_prices.get(symbol)
                        if not current_price: continue

                        potential_signal['entry_price'] = current_price
                        atr_value = df_features['atr'].iloc[-1]
                        potential_signal['signal_details']['atr_value'] = atr_value

                        stop_loss = current_price - (atr_value * ATR_SL_MULTIPLIER)
                        target_price = current_price + (atr_value * ATR_TP_MULTIPLIER)
                        sr_info = "ATR Default"

                        if USE_SR_LEVELS:
                            all_levels = fetch_sr_levels(symbol)
                            if all_levels:
                                strong_levels = [lvl for lvl in all_levels if lvl.get('score', 0) >= MINIMUM_SR_SCORE]
                                strong_supports = [lvl for lvl in strong_levels if 'support' in lvl.get('level_type', '') and lvl['level_price'] < current_price]
                                strong_resistances = [lvl for lvl in strong_levels if 'resistance' in lvl.get('level_type', '') and lvl['level_price'] > current_price]
                                if strong_supports:
                                    closest_strong_support = max(strong_supports, key=lambda x: x['level_price'])
                                    stop_loss = closest_strong_support['level_price'] * 0.998
                                    sr_info = f"Strong S/R (Score > {MINIMUM_SR_SCORE})"
                                if strong_resistances:
                                    closest_strong_resistance = min(strong_resistances, key=lambda x: x['level_price'])
                                    target_price = closest_strong_resistance['level_price'] * 0.998
                                    sr_info = f"Strong S/R (Score > {MINIMUM_SR_SCORE})"

                        if target_price <= current_price or stop_loss >= current_price: continue
                        potential_profit_pct = ((target_price / current_price) - 1) * 100
                        if potential_profit_pct < MINIMUM_PROFIT_PERCENTAGE: continue
                        potential_risk = current_price - stop_loss
                        if potential_risk <= 0: continue
                        risk_reward_ratio = (target_price - current_price) / potential_risk
                        if risk_reward_ratio < MINIMUM_RISK_REWARD_RATIO: continue

                        potential_signal['stop_loss'] = stop_loss
                        potential_signal['target_price'] = target_price
                        potential_signal['signal_details']['sr_info'] = sr_info
                        potential_signal['signal_details']['risk_reward_ratio'] = f"{risk_reward_ratio:.2f} : 1"

                        save_pending_recommendation(potential_signal)

                    del df_15m, df_4h, df_features, strategy
                except Exception as e:
                    logger.error(f"❌ [خطأ في المعالجة] حدث خطأ أثناء معالجة العملة {symbol}: {e}", exc_info=True)

            del btc_data_cycle
            gc.collect()
            time.sleep(120)

        except (KeyboardInterrupt, SystemExit): break
        except Exception as main_err:
            log_and_notify("error", f"خطأ غير متوقع في الحلقة الرئيسية: {main_err}", "SYSTEM")
            time.sleep(120)

def monitor_pending_loop():
    logger.info("⏳ [مراقب التوصيات] بدء حلقة مراقبة التوصيات المعلقة...")
    time.sleep(25)

    while True:
        try:
            if not check_db_connection() or not conn:
                logger.warning("⚠️ [مراقب التوصيات] لا يمكن الاتصال بقاعدة البيانات، سيتم تخطي الدورة.")
                time.sleep(30); continue

            with signal_cache_lock:
                if len(open_signals_cache) >= MAX_OPEN_TRADES:
                    time.sleep(10); continue

            pending_recs = []
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM pending_recommendations;")
                pending_recs = cur.fetchall()

            if not pending_recs:
                time.sleep(10); continue

            for rec in pending_recs:
                symbol = rec['symbol']
                trigger_price = rec['trigger_price']
                with prices_lock: current_price = current_prices.get(symbol)
                if not current_price: continue

                if current_price <= trigger_price:
                    logger.info(f"💥 [{symbol}] تم تفعيل توصية معلقة! السعر الحالي ({current_price}) وصل لسعر التفعيل ({trigger_price}).")

                    with signal_cache_lock:
                        if len(open_signals_cache) >= MAX_OPEN_TRADES:
                            logger.warning(f"⚠️ [{symbol}] تم تفعيل التوصية ولكن لا توجد أماكن متاحة للصفقات. سيتم المحاولة لاحقاً.")
                            continue

                    new_entry_price = trigger_price
                    tp1 = rec['original_entry_price']
                    tp2 = rec['original_target_price']
                    atr_at_creation = rec.get('atr_at_creation')

                    new_stop_loss, sl_info = 0, "ATR Fallback"
                    strong_supports = []
                    if USE_SR_LEVELS:
                        all_levels = fetch_sr_levels(symbol)
                        if all_levels:
                            strong_lvls = [lvl for lvl in all_levels if lvl.get('score', 0) >= MINIMUM_SR_SCORE]
                            strong_supports = [lvl for lvl in strong_lvls if 'support' in lvl.get('level_type','') and lvl['level_price'] < new_entry_price]
                    if strong_supports:
                        closest_support = max(strong_supports, key=lambda x: x['level_price'])
                        new_stop_loss = closest_support['level_price'] * 0.998
                        sl_info = f"Strong Support (Score > {MINIMUM_SR_SCORE})"
                    elif atr_at_creation:
                        new_stop_loss = new_entry_price - (atr_at_creation * ATR_SL_MULTIPLIER)
                    else:
                        new_stop_loss = new_entry_price * 0.98
                        sl_info = "Failsafe 2%"

                    if tp2 <= new_entry_price or new_stop_loss >= new_entry_price:
                        logger.error(f"❌ [{symbol}] تم إلغاء التوصية المفعلة. الأهداف أو الوقف غير منطقية. سيتم حذفها.")
                        with conn.cursor() as del_cur:
                            del_cur.execute("DELETE FROM pending_recommendations WHERE id = %s;", (rec['id'],))
                        conn.commit(); continue

                    new_signal = { 'symbol': symbol, 'entry_price': new_entry_price, 'target_price': tp2,
                        'stop_loss': new_stop_loss, 'strategy_name': "Pending-Triggered",
                        'signal_details': { **json.loads(rec.get('signal_details', '{}')),
                            'TP1': tp1, 'TP2': tp2,
                            'trigger_event': 'SL of pending recommendation hit', 'sr_info': sl_info } }

                    saved_signal = insert_signal_into_db(new_signal)
                    if saved_signal:
                        with signal_cache_lock: open_signals_cache[saved_signal['symbol']] = saved_signal
                        send_new_signal_alert(saved_signal)
                        with conn.cursor() as del_cur:
                            del_cur.execute("DELETE FROM pending_recommendations WHERE id = %s;", (rec['id'],))
                        conn.commit()
                        logger.info(f"✅ [{symbol}] تم تحويل التوصية المعلقة إلى صفقة نشطة وحذفها من قائمة الانتظار.")

        except (OperationalError, InterfaceError) as db_err:
            logger.error(f"❌ [مراقب التوصيات] خطأ في الاتصال بقاعدة البيانات: {db_err}. محاولة إعادة الاتصال...")
            check_db_connection(); time.sleep(30)
        except Exception as e:
            logger.error(f"❌ [مراقب التوصيات] خطأ غير متوقع في حلقة المراقبة: {e}", exc_info=True)
            time.sleep(60)

        time.sleep(5)

# ---------------------- واجهة برمجة تطبيقات Flask (مُعدَّلة) ----------------------
app = Flask(__name__)
CORS(app)

def get_fear_and_greed_index() -> Dict[str, Any]:
    classification_translation = {"Extreme Fear": "خوف شديد", "Fear": "خوف", "Neutral": "محايد", "Greed": "طمع", "Extreme Greed": "طمع شديد", "Error": "خطأ"}
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        response.raise_for_status()
        data = response.json()['data'][0]
        original = data['value_classification']
        return {"value": int(data['value']), "classification": classification_translation.get(original, original)}
    except Exception as e:
        logger.error(f"❌ [مؤشر الخوف والطمع] فشل الاتصال بالـ API: {e}")
        return {"value": -1, "classification": classification_translation["Error"]}

@app.route('/')
def home():
    try:
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, 'index.html')
        if not os.path.exists(file_path):
             return "<h1>ملف لوحة التحكم (index.html) غير موجود.</h1><p>تأكد من وجود الملف في نفس مجلد السكريبت.</p>", 404
        with open(file_path, 'r', encoding='utf-8') as f: return render_template_string(f.read())
    except FileNotFoundError: return "<h1>ملف لوحة التحكم (index.html) غير موجود.</h1>", 404
    except Exception as e: return f"<h1>خطأ في تحميل لوحة التحكم:</h1><p>{e}</p>", 500

@app.route('/api/market_status')
def get_market_status(): return jsonify({"btc_trend": get_btc_trend(), "fear_and_greed": get_fear_and_greed_index()})

@app.route('/api/stats')
def get_stats():
    if not check_db_connection() or not conn: return jsonify({"error": "فشل الاتصال بقاعدة البيانات"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT status, profit_percentage FROM signals WHERE status != 'open';")
            closed = cur.fetchall()
        wins = sum(1 for s in closed if s.get('profit_percentage', 0) > 0)
        losses = len(closed) - wins
        total_closed = len(closed)
        win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
        total_profit_percent = sum(s['profit_percentage'] for s in closed if s.get('profit_percentage') is not None)
        return jsonify({"win_rate": win_rate, "wins": wins, "losses": losses, "total_profit_percent": total_profit_percent, "total_closed_trades": total_closed})
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
            if s.get('closed_at') and isinstance(s['closed_at'], datetime):
                s['closed_at'] = s['closed_at'].isoformat()
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
    with notifications_lock:
        notifications_list = []
        for n in list(notifications_cache):
            notif_copy = n.copy()
            if 'timestamp' in notif_copy and isinstance(notif_copy['timestamp'], (datetime, str)):
                 notif_copy['timestamp'] = pd.to_datetime(notif_copy['timestamp']).isoformat()
            notifications_list.append(notif_copy)
        return jsonify(notifications_list)

@app.route('/api/pending_recommendations')
def get_pending_recommendations():
    if not check_db_connection() or not conn:
        return jsonify({"error": "فشل الاتصال بقاعدة البيانات"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM pending_recommendations ORDER BY created_at DESC;")
            pending_recs = cur.fetchall()
        for rec in pending_recs:
            with prices_lock:
                rec['current_price'] = current_prices.get(rec['symbol'])
            if 'created_at' in rec and isinstance(rec['created_at'], datetime):
                 rec['created_at'] = rec['created_at'].isoformat()
        return jsonify(pending_recs)
    except Exception as e:
        logger.error(f"❌ [API توصيات معلقة] خطأ: {e}")
        return jsonify({"error": "تعذر جلب التوصيات المعلقة"}), 500

@app.route('/api/trigger_pending/<int:rec_id>', methods=['POST'])
def trigger_pending_recommendation(rec_id):
    logger.info(f"ℹ️ [API تفعيل فوري] تم استلام طلب تفعيل فوري للتوصية المعلقة ID: {rec_id}")
    if not check_db_connection() or not conn:
        return jsonify({"error": "فشل الاتصال بقاعدة البيانات"}), 500
    with signal_cache_lock:
        if len(open_signals_cache) >= MAX_OPEN_TRADES:
            return jsonify({"error": f"لا يمكن التفعيل، تم الوصول للحد الأقصى للصفقات ({MAX_OPEN_TRADES})."}), 400
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM pending_recommendations WHERE id = %s;", (rec_id,))
            rec = cur.fetchone()
        if not rec: return jsonify({"error": "لم يتم العثور على التوصية المعلقة."}), 404
        symbol = rec['symbol']
        with prices_lock:
            current_price = current_prices.get(symbol)
        if not current_price:
            return jsonify({"error": f"تعذر الحصول على السعر الحالي لـ {symbol}."}), 500

        new_entry_price = current_price
        tp1 = rec['original_entry_price']
        tp2 = rec['original_target_price']
        atr_at_creation = rec.get('atr_at_creation')

        new_stop_loss, sl_info = 0, "ATR Fallback"
        strong_supports = []
        if USE_SR_LEVELS:
            all_levels = fetch_sr_levels(symbol)
            if all_levels:
                strong_lvls = [lvl for lvl in all_levels if lvl.get('score', 0) >= MINIMUM_SR_SCORE]
                strong_supports = [lvl for lvl in strong_lvls if 'support' in lvl.get('level_type','') and lvl['level_price'] < new_entry_price]
        if strong_supports:
            closest_support = max(strong_supports, key=lambda x: x['level_price'])
            new_stop_loss = closest_support['level_price'] * 0.998
            sl_info = f"Strong Support (Score > {MINIMUM_SR_SCORE})"
        elif atr_at_creation:
            new_stop_loss = new_entry_price - (atr_at_creation * ATR_SL_MULTIPLIER)
        else:
            new_stop_loss = new_entry_price * 0.98

        if tp2 <= new_entry_price or new_stop_loss >= new_entry_price:
            return jsonify({"error": "فشل التفعيل. الهدف أو الوقف غير منطقي بالسعر الحالي."}), 400

        original_details = json.loads(rec.get('signal_details', '{}'))
        new_signal = {
            'symbol': symbol, 'entry_price': new_entry_price, 'target_price': tp2,
            'stop_loss': new_stop_loss, 'strategy_name': "Manual-Triggered",
            'signal_details': { **original_details, 'TP1': tp1, 'TP2': tp2,
                'trigger_event': 'Manual trigger from dashboard', 'sr_info': sl_info }
        }
        saved_signal = insert_signal_into_db(new_signal)
        if saved_signal:
            with signal_cache_lock:
                open_signals_cache[saved_signal['symbol']] = saved_signal
            send_new_signal_alert(saved_signal)
            with conn.cursor() as del_cur:
                del_cur.execute("DELETE FROM pending_recommendations WHERE id = %s;", (rec_id,))
            conn.commit()
            logger.info(f"✅ [{symbol}] تم تفعيل التوصية المعلقة يدوياً بنجاح.")
            return jsonify({"message": f"تم تفعيل الصفقة لـ {symbol} بنجاح."})
        else:
            return jsonify({"error": "فشل حفظ الصفقة الجديدة في قاعدة البيانات."}), 500
    except Exception as e:
        logger.error(f"❌ [API تفعيل فوري] خطأ فادح: {e}", exc_info=True)
        if conn: conn.rollback()
        return jsonify({"error": "حدث خطأ داخلي في الخادم."}), 500

def run_flask():
    host, port = "0.0.0.0", int(os.environ.get('PORT', 10000))
    log_and_notify("info", f"بدء تشغيل لوحة التحكم على {host}:{port}", "SYSTEM")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        logger.warning("⚠️ [Flask] مكتبة 'waitress' غير موجودة, سيتم استخدام خادم التطوير.")
        app.run(host=host, port=port)

# ---------------------- نقطة انطلاق البرنامج (مُعدَّلة) ----------------------
def initialize_bot_services():
    global client, validated_symbols_to_scan
    logger.info("🤖 [خدمات البوت] بدء التهيئة في الخلفية...")
    try:
        client = Client(API_KEY, API_SECRET)
        logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
        init_db()
        load_open_signals_to_cache()
        load_notifications_to_cache()
        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan:
            logger.critical("❌ لا توجد رموز معتمدة للمسح. الحلقات لن تبدأ.")
            return
        Thread(target=run_websocket_manager, daemon=True).start()
        Thread(target=main_loop, daemon=True).start()
        Thread(target=monitor_pending_loop, daemon=True).start()
        logger.info("✅ [خدمات البوت] تم بدء جميع خدمات الخلفية بنجاح.")
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حاسم أثناء التهيئة: {e}", "SYSTEM")
        pass

if __name__ == "__main__":
    logger.info(f"🚀 بدء تشغيل بوت التداول المتقدم - إصدار {BASE_ML_MODEL_NAME}...")
    initialization_thread = Thread(target=initialize_bot_services, daemon=True)
    initialization_thread.start()
    run_flask()
    logger.info("👋 [إيقاف] تم إيقاف تشغيل البوت.")
    os._exit(0)
