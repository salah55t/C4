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
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler
from collections import deque

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v6_reversal_entry.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV6_ReversalEntry')

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
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V6_Reversal'
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
BTC_SYMBOL = 'BTCUSDT'

# --- Trading Logic Constants ---
MODEL_CONFIDENCE_THRESHOLD = 0.80
MAX_OPEN_TRADES: int = 5
USE_SR_LEVELS = True
MINIMUM_SR_SCORE = 30
# ** تعديل: مضاعفات الأهداف والوقف عند الدخول الفعلي من التوصية **
# سيتم استخدامها عند تحول التوصية إلى صفقة فعلية
ATR_SL_MULTIPLIER_ON_ENTRY = 1.5 # وقف أضيق عند الدخول
ATR_TP_MULTIPLIER_ON_ENTRY = 2.0 # هدف واقعي بعد الانعكاس
USE_BTC_TREND_FILTER = True
BTC_TREND_TIMEFRAME = '4h'
BTC_TREND_EMA_PERIOD = 10

# --- ثوابت فلترة التوصيات ---
MINIMUM_PROFIT_PERCENTAGE = 0.5
MINIMUM_RISK_REWARD_RATIO = 1.2
MINIMUM_15M_VOLUME_USDT = 200_000

# --- المتغيرات العامة وقفل العمليات ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
validated_symbols_to_scan: List[str] = []

# --- **جديد**: فصل التوصيات عن الصفقات المفتوحة ---
pending_recommendations_cache: Dict[str, Dict] = {} # ذاكرة التوصيات المنتظرة
recommendations_cache_lock = Lock()

open_signals_cache: Dict[str, Dict] = {} # ذاكرة الصفقات المفتوحة فعلاً
signal_cache_lock = Lock()

current_prices: Dict[str, float] = {}
prices_lock = Lock()
notifications_cache = deque(maxlen=50)
notifications_lock = Lock()


# ---------------------- دوال قاعدة البيانات (مُعدَّلة) ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    """
    Initializes the database connection and ensures all required tables are created.
    """
    global conn
    logger.info("[قاعدة البيانات] بدء تهيئة الاتصال...")
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            with conn.cursor() as cur:
                # جدول الصفقات المفتوحة والمغلقة
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION NOT NULL,
                        target_price DOUBLE PRECISION NOT NULL, stop_loss DOUBLE PRECISION NOT NULL,
                        status TEXT DEFAULT 'open', closing_price DOUBLE PRECISION, closed_at TIMESTAMP,
                        profit_percentage DOUBLE PRECISION, strategy_name TEXT, signal_details JSONB );
                """)
                # **جديد**: جدول التوصيات المنتظرة
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS recommendations (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL UNIQUE,
                        generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        original_entry_price DOUBLE PRECISION NOT NULL,
                        original_target_price DOUBLE PRECISION NOT NULL,
                        entry_trigger_price DOUBLE PRECISION NOT NULL,
                        status TEXT DEFAULT 'waiting',
                        signal_details JSONB,
                        triggered_at TIMESTAMP WITH TIME ZONE
                    );
                """)
                # الجداول الأخرى بدون تغيير
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications ( id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(), type TEXT NOT NULL,
                        message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE );
                """)
                cur.execute("""
                     CREATE TABLE IF NOT EXISTS ml_models ( id SERIAL PRIMARY KEY, model_name TEXT NOT NULL UNIQUE, model_data BYTEA NOT NULL,
                        trained_at TIMESTAMP DEFAULT NOW(), metrics JSONB );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS support_resistance_levels (
                        id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, level_price DOUBLE PRECISION NOT NULL,
                        level_type TEXT NOT NULL, timeframe TEXT NOT NULL, strength NUMERIC NOT NULL, score NUMERIC DEFAULT 0,
                        last_tested_at TIMESTAMP WITH TIME ZONE, details TEXT, created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        CONSTRAINT unique_level UNIQUE (symbol, level_price, timeframe, level_type)
                    );
                """)
            conn.commit()
            logger.info("✅ [قاعدة البيانات] تم تهيئة جميع جداول قاعدة البيانات بنجاح (بما في ذلك جدول التوصيات).")
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
    # لم يتم تغيير هذه الدالة
    if not check_db_connection() or not conn:
        logger.warning(f"⚠️ [{symbol}] لا يمكن جلب مستويات الدعم والمقاومة، اتصال قاعدة البيانات غير متاح.")
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT level_price, level_type, score FROM support_resistance_levels WHERE symbol = %s ORDER BY level_price ASC",
                (symbol,)
            )
            levels = cur.fetchall()
            if not levels: return None
            for level in levels: level['score'] = float(level.get('score', 0))
            logger.info(f"📈 [{symbol}] تم جلب {len(levels)} مستوى دعم ومقاومة.")
            return levels
    except Exception as e:
        logger.error(f"❌ [{symbol}] خطأ أثناء جلب مستويات الدعم والمقاومة: {e}")
        if conn: conn.rollback()
        return None

# ---------------------- دوال Binance والبيانات (بدون تغيير كبير) ----------------------
def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    # لم يتم تغيير هذه الدالة
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
    # لم يتم تغيير هذه الدالة
    if not client: return None
    try:
        start_str = (datetime.utcnow() - timedelta(days=days + 1)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in numeric_cols:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
        return df[numeric_cols].dropna()
    except BinanceAPIException as e:
        logger.warning(f"⚠️ [API Binance] خطأ في جلب بيانات {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ [البيانات] خطأ أثناء جلب البيانات التاريخية لـ {symbol}: {e}")
        return None

# --- دوال حساب المؤشرات وتحميل النماذج (بدون تغيير) ---
def calculate_all_features(df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    # No change in this function
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
    sma = df_calc['close'].rolling(window=BBANDS_PERIOD).mean(); std_dev = df_calc['close'].rolling(window=BBANDS_PERIOD).std()
    upper_band = sma + (std_dev * 2); lower_band = sma - (std_dev * 2)
    df_calc['bb_width'] = (upper_band - lower_band) / (sma + 1e-9)
    df_calc['price_vs_ema200'] = (df_calc['close'] / df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()) - 1
    # ... other indicators ...
    return df_calc.dropna()

def load_ml_model_bundle_from_folder(symbol: str) -> Optional[Dict[str, Any]]:
    # No change in this function
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    model_dir = 'Mo'
    file_path = os.path.join(model_dir, f"{model_name}.pkl")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                model_bundle = pickle.load(f)
            if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
                logger.info(f"✅ [نموذج تعلم الآلة] تم تحميل النموذج '{model_name}' بنجاح من الملف.")
                return model_bundle
        except Exception as e:
            logger.error(f"❌ [نموذج تعلم الآلة] خطأ عند تحميل النموذج '{file_path}': {e}", exc_info=True)
    return None

class TradingStrategy:
    # No change in this class
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_bundle_from_folder(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        # This calls a simplified function for brevity in this example
        return calculate_all_features(df_15m, df_4h, btc_df)

    def generate_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]): return None
        last_row = df_processed.iloc[-1]
        try:
            features_df = pd.DataFrame([last_row], columns=df_processed.columns)[self.feature_names]
            if features_df.isnull().values.any(): return None
            features_scaled = self.scaler.transform(features_df)
            prediction = self.ml_model.predict(features_scaled)[0]
            prediction_proba = self.ml_model.predict_proba(features_scaled)[0]
            prob_for_class_1 = prediction_proba[list(self.ml_model.classes_).index(1)] if 1 in self.ml_model.classes_ else 0
            if prediction == 1 and prob_for_class_1 >= MODEL_CONFIDENCE_THRESHOLD:
                logger.info(f"✅ [العثور على توصية] {self.symbol}: تنبأ النموذج 'شراء' بثقة {prob_for_class_1:.2%}.")
                return {'symbol': self.symbol, 'strategy_name': BASE_ML_MODEL_NAME, 'signal_details': {'ML_Probability_Buy': f"{prob_for_class_1:.2%}"}}
            return None
        except Exception as e:
            logger.warning(f"⚠️ [توليد توصية] {self.symbol}: خطأ: {e}")
            return None

# ---------------------- دوال WebSocket والاستراتيجية (مُعدَّلة) ----------------------
def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
    """
    Handles incoming ticker messages from WebSocket, checks for TP/SL on open trades
    AND checks for entry triggers on pending recommendations.
    """
    global open_signals_cache, current_prices, pending_recommendations_cache
    try:
        data = msg.get('data', msg) if isinstance(msg, dict) else msg
        if not isinstance(data, list): data = [data]

        for item in data:
            symbol = item.get('s')
            if not symbol: continue
            price = float(item.get('c', 0))
            if price == 0: continue
            with prices_lock: current_prices[symbol] = price

            # 1. Check for TP/SL on active trades
            signal_to_process, status, closing_price = None, None, None
            with signal_cache_lock:
                if symbol in open_signals_cache:
                    signal = open_signals_cache[symbol]
                    if price >= signal['target_price']: status, closing_price, signal_to_process = 'target_hit', signal['target_price'], signal
                    elif price <= signal['stop_loss']: status, closing_price, signal_to_process = 'stop_loss_hit', signal['stop_loss'], signal
            if signal_to_process:
                logger.info(f"⚡ [صفقة نشطة] حدث '{status}' لـ {symbol} عند سعر {price:.8f}")
                Thread(target=close_signal, args=(signal_to_process, status, closing_price, "auto")).start()
                continue # Skip to next item in message

            # 2. **جديد**: Check for entry trigger on pending recommendations
            rec_to_trigger = None
            with recommendations_cache_lock:
                if symbol in pending_recommendations_cache:
                    rec = pending_recommendations_cache[symbol]
                    # الدخول عندما يلمس السعر أو ينزل تحت سعر التفعيل
                    if price <= rec['entry_trigger_price']:
                        rec_to_trigger = rec
            if rec_to_trigger:
                logger.info(f"🎯 [تفعيل توصية] تم الوصول لسعر الدخول لـ {symbol} عند {price:.8f} (السعر المستهدف: {rec_to_trigger['entry_trigger_price']:.8f})")
                # استخدام Thread لتجنب إيقاف معالجة الرسائل الأخرى
                Thread(target=open_trade_from_recommendation, args=(rec_to_trigger, price)).start()

    except Exception as e:
        logger.error(f"❌ [متتبع WebSocket] خطأ في معالجة رسالة السعر الفورية: {e}", exc_info=True)

def run_websocket_manager() -> None:
    # لم يتم تغيير هذه الدالة
    logger.info("ℹ️ [WebSocket] بدء مدير WebSocket...")
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    # تحويل الرموز إلى صيغة stream name المطلوبة
    streams = [f"{s.lower()}@ticker" for s in validated_symbols_to_scan] if validated_symbols_to_scan else []
    if not streams:
        logger.error("❌ [WebSocket] لا توجد رموز صالحة للاستماع إليها. لن يتم بدء الـ WebSocket.")
        return
        
    twm.start_multiplex_socket(callback=handle_ticker_message, streams=streams)
    logger.info(f"✅ [WebSocket] تم الاتصال والاستماع لـ {len(streams)} عملة بنجاح.")
    twm.join()

# ---------------------- دوال التنبيهات والإدارة (مُعدَّلة) ----------------------
def send_telegram_message(target_chat_id: str, text: str):
    if not TELEGRAM_TOKEN or not target_chat_id: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(target_chat_id), 'text': text, 'parse_mode': 'Markdown'}
    try: requests.post(url, json=payload, timeout=20).raise_for_status()
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

def send_new_recommendation_alert(rec_data: Dict[str, Any]) -> None:
    """
    **جديد**: إرسال تنبيه عند إنشاء توصية جديدة (وليس صفقة).
    """
    safe_symbol = rec_data['symbol'].replace('_', '\\_')
    original_entry = rec_data['original_entry_price']
    trigger_price = rec_data['entry_trigger_price']

    message = (f"⏳ *توصية جديدة قيد الانتظار* ⏳\n\n"
               f"🪙 *العملة:* `{safe_symbol}`\n"
               f"📈 *النوع:* شراء عند الانعكاس (LONG)\n\n"
               f"📉 *سعر تفعيل الدخول:* `${trigger_price:,.8g}`\n"
               f"🔍 *تم إنشاؤها عند سعر:* `${original_entry:,.8g}`\n"
               f"📊 *ثقة النموذج:* {rec_data['signal_details']['ML_Probability_Buy']}\n\n"
               f"_سيقوم البوت بمراقبة هذه العملة والدخول في صفقة تلقائياً إذا وصل السعر إلى مستوى التفعيل._")

    log_and_notify('info', f"توصية جديدة لـ {rec_data['symbol']} بانتظار سعر تفعيل عند ${trigger_price:,.8g}", "NEW_RECOMMENDATION")
    send_telegram_message(CHAT_ID, message)

def send_new_trade_alert(signal_data: Dict[str, Any]) -> None:
    """
    **معدل**: إرسال تنبيه عند فتح صفقة فعلية (بعد تفعيل التوصية).
    """
    safe_symbol = signal_data['symbol'].replace('_', '\\_')
    entry, target, sl = signal_data['entry_price'], signal_data['target_price'], signal_data['stop_loss']
    profit_pct = ((target / entry) - 1) * 100 if entry > 0 else 0

    message = (f"💡 *تم فتح صفقة جديدة تلقائياً* 💡\n\n"
               f"🪙 *العملة:* `{safe_symbol}`\n"
               f"📈 *النوع:* شراء (LONG)\n\n"
               f"⬅️ *سعر الدخول الفعلي:* `${entry:,.8g}`\n"
               f"🎯 *الهدف الجديد:* `${target:,.8g}` (ربح متوقع `{profit_pct:+.2f}%`)\n"
               f"🛑 *وقف الخسارة الجديد:* `${sl:,.8g}`\n\n"
               f"🔍 *ثقة النموذج الأصلية:* {signal_data['signal_details']['ML_Probability_Buy']}")

    log_and_notify('info', f"صفقة جديدة مفتوحة: {signal_data['symbol']} بسعر دخول ${entry:,.8g}", "NEW_SIGNAL")
    send_telegram_message(CHAT_ID, message)

def save_or_update_recommendation_in_db(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    **جديد**: حفظ توصية جديدة أو تحديث توصية موجودة لنفس العملة.
    """
    if not check_db_connection() or not conn: return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO recommendations (symbol, original_entry_price, original_target_price, entry_trigger_price, signal_details, status)
                VALUES (%s, %s, %s, %s, %s, 'waiting')
                ON CONFLICT (symbol) DO UPDATE SET
                    original_entry_price = EXCLUDED.original_entry_price,
                    original_target_price = EXCLUDED.original_target_price,
                    entry_trigger_price = EXCLUDED.entry_trigger_price,
                    signal_details = EXCLUDED.signal_details,
                    generated_at = NOW(),
                    status = 'waiting',
                    triggered_at = NULL
                RETURNING id;
                """,
                (rec['symbol'], rec['original_entry_price'], rec['original_target_price'], rec['entry_trigger_price'], json.dumps(rec.get('signal_details', {})))
            )
            rec['id'] = cur.fetchone()['id']
        conn.commit()
        logger.info(f"✅ [قاعدة بيانات التوصيات] تم حفظ/تحديث التوصية لـ {rec['symbol']} (ID: {rec['id']}).")
        return rec
    except Exception as e:
        logger.error(f"❌ [قاعدة بيانات التوصيات] خطأ في حفظ توصية {rec['symbol']}: {e}", exc_info=True)
        if conn: conn.rollback()
        return None

def insert_signal_into_db(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # لم يتم تغيير هذه الدالة
    if not check_db_connection() or not conn: return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;",
                (signal['symbol'], signal['entry_price'], signal['target_price'], signal['stop_loss'], signal.get('strategy_name'), json.dumps(signal.get('signal_details', {})))
            )
            signal['id'] = cur.fetchone()['id']
        conn.commit()
        logger.info(f"✅ [قاعدة البيانات] تم إدراج الصفقة لـ {signal['symbol']} (ID: {signal['id']}).")
        return signal
    except Exception as e:
        logger.error(f"❌ [إدراج في قاعدة البيانات] خطأ في إدراج صفقة {signal['symbol']}: {e}", exc_info=True)
        if conn: conn.rollback()
        return None

def open_trade_from_recommendation(rec: Dict, entry_price: float):
    """
    **جديد**: الدالة المحورية التي تحول توصية منتظرة إلى صفقة فعلية.
    """
    symbol = rec['symbol']
    
    # التأكد من أن التوصية لا تزال في الذاكرة (لم يتم تفعيلها من قبل thread آخر)
    with recommendations_cache_lock:
        if symbol not in pending_recommendations_cache:
            logger.warning(f"⚠️ [{symbol}] تم بالفعل تفعيل أو إزالة التوصية. إلغاء فتح الصفقة المكررة.")
            return

    # التحقق من حدود الصفقات المفتوحة
    with signal_cache_lock:
        if symbol in open_signals_cache:
            logger.warning(f"⚠️ [{symbol}] توجد صفقة مفتوحة بالفعل. إلغاء فتح صفقة جديدة.")
            # إزالة التوصية لتجنب التفعيل المتكرر
            with recommendations_cache_lock:
                del pending_recommendations_cache[symbol]
            return
        if len(open_signals_cache) >= MAX_OPEN_TRADES:
            logger.warning(f"⚠️ [{symbol}] تم الوصول للحد الأقصى للصفقات المفتوحة. لا يمكن فتح صفقة جديدة.")
            return
            
    logger.info(f"⚙️ [{symbol}] بدء عملية فتح صفقة فعلية من توصية...")
    
    # 1. جلب بيانات جديدة لحساب ATR دقيق لحظة الدخول
    df_fresh = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, 2) # أيام قليلة كافية
    if df_fresh is None or df_fresh.empty:
        logger.error(f"❌ [{symbol}] فشل جلب بيانات جديدة لحساب ATR. إلغاء فتح الصفقة.")
        return
        
    high_low = df_fresh['high'] - df_fresh['low']
    high_close = (df_fresh['high'] - df_fresh['close'].shift()).abs()
    low_close = (df_fresh['low'] - df_fresh['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_value_on_entry = tr.ewm(span=ATR_PERIOD, adjust=False).mean().iloc[-1]
    
    # 2. حساب وقف الخسارة والهدف الجديدين
    new_stop_loss = entry_price - (atr_value_on_entry * ATR_SL_MULTIPLIER_ON_ENTRY)
    new_target_price = entry_price + (atr_value_on_entry * ATR_TP_MULTIPLIER_ON_ENTRY)
    
    logger.info(f"📊 [{symbol}] الحسابات الجديدة: ATR={atr_value_on_entry:.5f}, SL={new_stop_loss:.8g}, TP={new_target_price:.8g}")

    # 3. تحديث حالة التوصية في قاعدة البيانات
    if check_db_connection() and conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE recommendations SET status = 'triggered', triggered_at = NOW() WHERE id = %s",
                    (rec['id'],)
                )
            conn.commit()
        except Exception as e:
            logger.error(f"❌ [{symbol}] فشل تحديث حالة التوصية في قاعدة البيانات: {e}")
            if conn: conn.rollback()
            # لا نوقف العملية، يمكن الاستمرار وفتح الصفقة
    
    # 4. إنشاء الصفقة الجديدة وإدراجها
    new_signal = {
        'symbol': symbol,
        'entry_price': entry_price,
        'target_price': new_target_price,
        'stop_loss': new_stop_loss,
        'strategy_name': rec.get('strategy_name', BASE_ML_MODEL_NAME),
        'signal_details': rec.get('signal_details', {})
    }
    
    saved_signal = insert_signal_into_db(new_signal)
    
    if saved_signal:
        # 5. تحديث الذاكرة المؤقتة وإرسال التنبيهات
        with recommendations_cache_lock:
            if symbol in pending_recommendations_cache:
                del pending_recommendations_cache[symbol]
        
        with signal_cache_lock:
            open_signals_cache[symbol] = saved_signal
            
        send_new_trade_alert(saved_signal)
    else:
        logger.error(f"❌ [{symbol}] فشل حاسم في حفظ الصفقة الجديدة في قاعدة البيانات. لن يتم فتح الصفقة.")


def close_signal(signal: Dict, status: str, closing_price: float, closed_by: str):
    # لم يتم تغيير هذه الدالة
    symbol = signal['symbol']
    with signal_cache_lock:
        if symbol not in open_signals_cache or open_signals_cache[symbol]['id'] != signal['id']: return
    if not check_db_connection() or not conn: return
    try:
        db_profit_pct = float(((closing_price / signal['entry_price']) - 1) * 100)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE signals SET status = %s, closing_price = %s, closed_at = NOW(), profit_percentage = %s WHERE id = %s;",
                (status, closing_price, db_profit_pct, signal['id'])
            )
        conn.commit()
        with signal_cache_lock: del open_signals_cache[symbol]
        status_map = {'target_hit': '✅ تحقق الهدف', 'stop_loss_hit': '🛑 ضرب وقف الخسارة', 'manual_close': '🖐️ أُغلقت يدوياً'}
        status_message = status_map.get(status, status)
        alert_msg = f"*{status_message}*\n`{symbol}` | *الربح:* `{db_profit_pct:+.2f}%`"
        send_telegram_message(CHAT_ID, alert_msg)
        log_and_notify('info', f"{status_message}: {symbol} | الربح: {db_profit_pct:+.2f}%", 'CLOSE_SIGNAL')
    except Exception as e:
        logger.error(f"❌ [إغلاق الصفقة] خطأ فادح أثناء إغلاق الصفقة {signal['id']}: {e}", exc_info=True)
        if conn: conn.rollback()

def load_data_to_cache():
    """
    **معدل**: تحميل كل من الصفقات المفتوحة والتوصيات المنتظرة إلى الذاكرة.
    """
    if not check_db_connection() or not conn: return
    
    # 1. تحميل الصفقات المفتوحة
    logger.info("ℹ️ [تحميل الذاكرة] جاري تحميل الصفقات المفتوحة...")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status = 'open';")
            open_signals = cur.fetchall()
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in open_signals: open_signals_cache[signal['symbol']] = dict(signal)
            logger.info(f"✅ [تحميل الذاكرة] تم تحميل {len(open_signals)} صفقة مفتوحة.")
    except Exception as e: logger.error(f"❌ [تحميل الذاكرة] فشل تحميل الصفقات المفتوحة: {e}")

    # 2. تحميل التوصيات المنتظرة
    logger.info("ℹ️ [تحميل الذاكرة] جاري تحميل التوصيات المنتظرة...")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM recommendations WHERE status = 'waiting';")
            pending_recs = cur.fetchall()
            with recommendations_cache_lock:
                pending_recommendations_cache.clear()
                for rec in pending_recs: pending_recommendations_cache[rec['symbol']] = dict(rec)
            logger.info(f"✅ [تحميل الذاكرة] تم تحميل {len(pending_recs)} توصية منتظرة.")
    except Exception as e: logger.error(f"❌ [تحميل الذاكرة] فشل تحميل التوصيات المنتظرة: {e}")

    # 3. تحميل التنبيهات (بدون تغيير)
    logger.info("ℹ️ [تحميل الذاكرة] جاري تحميل آخر التنبيهات...")
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
            logger.info(f"✅ [تحميل الذاكرة] تم تحميل {len(notifications_cache)} تنبيه.")
    except Exception as e: logger.error(f"❌ [تحميل الذاكرة] فشل تحميل التنبيهات: {e}")


# ---------------------- حلقة العمل الرئيسية (مُعدَّلة بشكل كبير) ----------------------
def get_btc_trend() -> Dict[str, Any]:
    # لم يتم تغيير هذه الدالة
    if not client: return {"status": "error", "is_uptrend": False}
    try:
        klines = client.get_klines(symbol=BTC_SYMBOL, interval=BTC_TREND_TIMEFRAME, limit=BTC_TREND_EMA_PERIOD * 2)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df['close'] = pd.to_numeric(df['close'])
        ema = df['close'].ewm(span=BTC_TREND_EMA_PERIOD, adjust=False).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        return {"status": "Uptrend" if current_price > ema else "Downtrend", "is_uptrend": current_price > ema}
    except Exception as e:
        logger.error(f"❌ [فلتر BTC] فشل تحديد اتجاه البيتكوين: {e}")
        return {"status": "Error", "is_uptrend": False}

def main_loop():
    logger.info("[الحلقة الرئيسية] انتظار اكتمال التهيئة الأولية...")
    time.sleep(15)
    if not validated_symbols_to_scan:
        log_and_notify("critical", "لا توجد رموز معتمدة للمسح. لن يستمر البوت في العمل.", "SYSTEM")
        return
    log_and_notify("info", f"بدء حلقة البحث عن توصيات جديدة لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")
    
    while True:
        try:
            if USE_BTC_TREND_FILTER:
                trend_data = get_btc_trend()
                if not trend_data.get("is_uptrend"):
                    logger.warning(f"⚠️ [إيقاف البحث] تم إيقاف البحث عن توصيات شراء بسبب الاتجاه الهابط للبيتكوين.")
                    time.sleep(300); continue

            logger.info(f"ℹ️ [بدء المسح] بدء دورة مسح جديدة للبحث عن توصيات.")
            
            btc_data_cycle = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
            if btc_data_cycle is None:
                logger.error("❌ فشل في جلب بيانات BTC. سيتم تخطي دورة المسح هذه.")
                time.sleep(120); continue
            btc_data_cycle['btc_returns'] = btc_data_cycle['close'].pct_change()
            
            for symbol in validated_symbols_to_scan:
                # لا نتحقق من عدد الصفقات المفتوحة هنا، لأننا نولد توصيات فقط
                # التحقق يتم عند تفعيل التوصية
                
                try:
                    df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
                    df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
                    if df_15m is None or df_15m.empty or df_4h is None or df_4h.empty: continue
                    
                    strategy = TradingStrategy(symbol)
                    df_features = strategy.get_features(df_15m, df_4h, btc_data_cycle)
                    if df_features is None or df_features.empty: continue
                    
                    potential_signal = strategy.generate_signal(df_features)
                    if potential_signal:
                        # --- فلاتر التوصية (سيولة، نسبة ربح، الخ) ---
                        last_candle = df_features.iloc[-1]
                        last_15m_volume_usdt = last_candle['volume'] * last_candle['close']
                        if last_15m_volume_usdt < MINIMUM_15M_VOLUME_USDT:
                            logger.info(f"📉 [{symbol}] تم تجاهل التوصية. حجم السيولة (${last_15m_volume_usdt:,.0f}) أقل من الحد الأدنى.")
                            continue
                        
                        potential_signal['signal_details']['last_15m_volume_usdt'] = f"${last_15m_volume_usdt:,.0f}"
                        
                        with prices_lock: current_price = current_prices.get(symbol)
                        if not current_price: continue
                        
                        atr_value = df_features['atr'].iloc[-1]
                        
                        # **المنطق الجديد**: تحديد سعر التفعيل (وقف الخسارة الأصلي) والهدف الأصلي
                        # هذه القيم ستُحفظ مع التوصية ولكن لن تُستخدم لفتح صفقة مباشرة
                        original_stop_loss = current_price - (atr_value * ATR_SL_MULTIPLIER_ON_ENTRY * 1.5) # نطاق أوسع مبدئياً
                        original_target_price = current_price + (atr_value * ATR_TP_MULTIPLIER_ON_ENTRY * 1.5)
                        sr_info = "ATR Default"

                        if USE_SR_LEVELS:
                            all_levels = fetch_sr_levels(symbol)
                            if all_levels:
                                strong_supports = [lvl for lvl in all_levels if 'support' in lvl.get('level_type', '') and lvl['level_price'] < current_price and lvl.get('score', 0) >= MINIMUM_SR_SCORE]
                                if strong_supports:
                                    closest_strong_support = max(strong_supports, key=lambda x: x['level_price'])
                                    original_stop_loss = closest_strong_support['level_price'] * 0.998
                                    sr_info = f"Strong S/R (Score > {MINIMUM_SR_SCORE})"
                        
                        # --- تطبيق الفلاتر على التوصية الأولية ---
                        if original_target_price <= current_price or original_stop_loss >= current_price: continue
                        potential_profit_pct = ((original_target_price / current_price) - 1) * 100
                        if potential_profit_pct < MINIMUM_PROFIT_PERCENTAGE: continue
                        risk_reward_ratio = (original_target_price - current_price) / (current_price - original_stop_loss)
                        if risk_reward_ratio < MINIMUM_RISK_REWARD_RATIO: continue
                        
                        # **الخطوة الحاسمة**: إنشاء التوصية لحفظها
                        recommendation_to_save = {
                            'symbol': symbol,
                            'original_entry_price': current_price,
                            'original_target_price': original_target_price,
                            'entry_trigger_price': original_stop_loss, # الوقف الأصلي هو سعر التفعيل
                            'signal_details': potential_signal['signal_details']
                        }
                        
                        saved_rec = save_or_update_recommendation_in_db(recommendation_to_save)
                        if saved_rec:
                            with recommendations_cache_lock:
                                pending_recommendations_cache[saved_rec['symbol']] = saved_rec
                            send_new_recommendation_alert(saved_rec)
                    
                except Exception as e:
                    logger.error(f"❌ [خطأ في المعالجة] حدث خطأ أثناء معالجة العملة {symbol}: {e}", exc_info=True)

            gc.collect()
            logger.info("ℹ️ [نهاية المسح] انتهت دورة البحث عن توصيات. في انتظار 180 ثانية...")
            time.sleep(180) 

        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as main_err:
            log_and_notify("error", f"خطأ غير متوقع في الحلقة الرئيسية: {main_err}", "SYSTEM")
            time.sleep(120)

# ---------------------- واجهة برمجة تطبيقات Flask (مُعدَّلة) ----------------------
app = Flask(__name__)
CORS(app)

# --- دوال API (مع إضافة قسم للتوصيات) ---
@app.route('/')
def home():
    try:
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, 'index.html')
        with open(file_path, 'r', encoding='utf-8') as f: return render_template_string(f.read())
    except Exception as e: return f"<h1>خطأ: {e}</h1>", 500

@app.route('/api/market_status')
def get_market_status():
    # A function to get fear and greed index would be here
    return jsonify({"btc_trend": get_btc_trend(), "fear_and_greed": {"value": 50, "classification": "محايد"}})

@app.route('/api/stats')
def get_stats():
    # No changes to this function
    if not check_db_connection() or not conn: return jsonify({"error": "فشل الاتصال بقاعدة البيانات"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT status, profit_percentage FROM signals WHERE status != 'open';")
            closed = cur.fetchall()
        wins = sum(1 for s in closed if s.get('profit_percentage', 0) > 0)
        total_closed = len(closed)
        win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
        return jsonify({"win_rate": win_rate, "wins": wins, "losses": len(closed) - wins})
    except Exception as e:
        logger.error(f"❌ [API إحصائيات] خطأ: {e}"); return jsonify({"error": "تعذر جلب الإحصائيات"}), 500

@app.route('/api/signals')
def get_signals():
    # No changes to this function, it returns active/closed trades
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
        
@app.route('/api/recommendations')
def get_recommendations():
    """
    **جديد**: نقطة نهاية API لجلب التوصيات المنتظرة.
    """
    if not check_db_connection() or not conn: return jsonify({"error": "فشل الاتصال بقاعدة البيانات"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM recommendations WHERE status = 'waiting' ORDER BY generated_at DESC;")
            all_recs = cur.fetchall()
        for r in all_recs:
            if r.get('generated_at'): r['generated_at'] = r['generated_at'].isoformat()
            with prices_lock: r['current_price'] = current_prices.get(r['symbol'])
        return jsonify(all_recs)
    except Exception as e:
        logger.error(f"❌ [API توصيات] خطأ: {e}"); return jsonify({"error": "تعذر جلب التوصيات"}), 500

@app.route('/api/close/<int:signal_id>', methods=['POST'])
def manual_close_signal(signal_id):
    # No changes to this function
    signal_to_close = None
    with signal_cache_lock:
        for s in open_signals_cache.values():
            if s['id'] == signal_id: signal_to_close = s.copy(); break
    if not signal_to_close: return jsonify({"error": "لم يتم العثور على الإشارة."}), 404
    with prices_lock: closing_price = current_prices.get(signal_to_close['symbol'])
    if not closing_price: return jsonify({"error": "تعذر الحصول على السعر الحالي."}), 500
    Thread(target=close_signal, args=(signal_to_close, 'manual_close', closing_price, "manual")).start()
    return jsonify({"message": f"جاري إغلاق الإشارة {signal_id}."})

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
        app.run(host=host, port=port)

# ---------------------- نقطة انطلاق البرنامج ----------------------
def initialize_bot_services():
    global client, validated_symbols_to_scan
    logger.info("🤖 [خدمات البوت] بدء التهيئة في الخلفية...")
    try:
        client = Client(API_KEY, API_SECRET)
        logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
        init_db()
        load_data_to_cache() # تحميل كل من الصفقات والتوصيات
        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan:
            logger.critical("❌ لا توجد رموز معتمدة للمسح. الحلقات لن تبدأ.")
            return
        Thread(target=run_websocket_manager, daemon=True).start()
        Thread(target=main_loop, daemon=True).start()
        logger.info("✅ [خدمات البوت] تم بدء جميع خدمات الخلفية بنجاح.")
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حاسم أثناء التهيئة: {e}", "SYSTEM")

if __name__ == "__main__":
    logger.info(f"🚀 بدء تشغيل بوت التداول بمنطق الدخول العكسي - إصدار {BASE_ML_MODEL_NAME}...")
    initialization_thread = Thread(target=initialize_bot_services, daemon=True)
    initialization_thread.start()
    run_flask()
    logger.info("👋 [إيقاف] تم إيقاف تشغيل البوت.")
    os._exit(0)
