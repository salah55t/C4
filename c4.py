# ملف c4.py - نسخة V10.0 (ثلاث استراتيجيات متخصصة)
# --- التغييرات الرئيسية (V10.0):
# 1. [إزالة] تمت إزالة جميع الاستراتيجيات القديمة
# 2. [إضافة] تمت إضافة استراتيجية الانعكاس الذكي للسوق الهابط
# 3. [إضافة] تمت إضافة استراتيجية الزخم المستدام للسوق الصاعد
# 4. [إضافة] تمت إضافة استراتيجية الترند الجانبي الذكي للسوق الجانبي
# 5. [تحسين] تم تحسين لوحة التحكم لعرض تفاصيل الاستراتيجيات الجديدة

import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
import redis
import re
import gc
import random
from decimal import Decimal, ROUND_DOWN
from urllib.parse import urlparse
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance.exceptions import BinanceAPIException
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timezone, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Set, Tuple
from sklearn.preprocessing import StandardScaler
from collections import deque, Counter
import warnings

# --- إعدادات التجاهل واللوجر ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v10_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV10.0')

# --- المشفر المخصص لأنواع بيانات NumPy ---
class NpEncoder(json.JSONEncoder):
    """ مشفر مخصص لأنواع بيانات NumPy """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        return super(NpEncoder, self).default(obj)

# --- تحميل متغيرات البيئة ---
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
    REDIS_URL: str = config('REDIS_URL', default='redis://localhost:6379/0')
    TELEGRAM_BOT_TOKEN: str = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID: str = config('TELEGRAM_CHAT_ID', default='')

except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
    exit(1)

# --- متغيرات عامة وإعدادات البوت ---
is_trading_enabled: bool = False
trading_status_lock = Lock()

# --- المتغيرات القابلة للتعديل ---
RISK_PER_TRADE_PERCENT: float = 0.85
risk_per_trade_lock = Lock()

BUY_CONFIDENCE_THRESHOLD = 0.53
buy_confidence_lock = Lock()

ORDER_BOOK_MIN_BID_ASK_RATIO: float = 1.15
order_book_ratio_lock = Lock()

VOLUME_FILTER_MULTIPLIER: float = 1.1
volume_filter_lock = Lock()

MIN_PROFIT_PERCENT: float = 0.8

# --- مفاتيح تفعيل الاستراتيجيات الجديدة ---
USE_SMART_REVERSAL_STRATEGY: bool = True  # للسوق الهابط
smart_reversal_strategy_lock = Lock()

USE_SUSTAINABLE_MOMENTUM_STRATEGY: bool = True  # للسوق الصاعد
sustainable_momentum_strategy_lock = Lock()

USE_SMART_SIDEWAYS_STRATEGY: bool = True  # للسوق الجانبي
smart_sideways_strategy_lock = Lock()

# --- إعدادات البوت ---
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V10_With_Microstructure'
MODEL_FOLDER: str = 'V10'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '1h' # الإطار الزمني الأعلى المستخدم في فلتر التأكيد
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 90
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices_v10"
TRADING_FEE_PERCENT: float = 0.1
STATS_TRADE_SIZE_USDT: float = 4.0
BTC_SYMBOL: str = 'BTCUSDT'
MAX_OPEN_TRADES: int = 5
SYMBOL_PROCESSING_BATCH_SIZE: int = 10

# --- إعدادات رحلة التداول الديناميكية ---
USE_DYNAMIC_JOURNEY = True

# --- إعدادات المؤشرات الفنية ---
EMA_FAST_PERIOD: int = 21
EMA_SLOW_PERIOD: int = 50
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
BTC_CORR_PERIOD: int = 30
REL_VOL_PERIOD: int = 30
MOMENTUM_PERIOD: int = 10 # أسرع
EMA_SLOPE_PERIOD: int = 5
SUPERTREND_ATR_PERIOD: int = 10
SUPERTREND_MULTIPLIER: float = 3.0
CANDLE_AVG_VOLUME_PERIOD: int = 15
SR_LOOKBACK_CANDLES: int = 60
SR_MIN_BOUNCES: int = 2

# --- إعدادات الفلاتر المتقدمة وإدارة الصفقات ---
ORDER_BOOK_DEPTH_LIMIT: int = 100
ORDER_BOOK_ANALYSIS_RANGE_PCT: float = 0.005
USE_ATR_TRAILING_STOP: bool = True
ATR_TS_PERIOD: int = 14
ATR_TS_MULTIPLIER: float = 2.2

# --- متغيرات الحالة والكاش ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
ml_models_cache: Dict[str, Any] = {}
exchange_info_map: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
notifications_cache = deque(maxlen=50)
notifications_lock = Lock()
rejection_logs_cache = deque(maxlen=100)
rejection_logs_lock = Lock()
current_market_state: Dict[str, Any] = {"overall_regime": "INITIALIZING", "trend_details_by_tf": {}, "last_updated": None}
market_state_lock = Lock()
last_market_state_check = 0
technical_signals_cache: Dict[str, Dict] = {}
TECHNICAL_SIGNAL_CACHE_DURATION: int = 60 * 5

# --- قاموس أسباب الرفض باللغة العربية ---
REJECTION_REASONS_AR = {
    "Market Volatility Filter Failed": "فلتر تقلب السوق رفض الدخول",
    "Trend Strength Filter Failed": "فلتر قوة الاتجاه رفض الدخول",
    "Smart Reversal Strategy Conditions Not Met": "شروط استراتيجية الانعكاس الذكي لم تتحقق",
    "Sustainable Momentum Strategy Conditions Not Met": "شروط استراتيجية الزخم المستدام لم تتحقق",
    "Smart Sideways Strategy Conditions Not Met": "شروط استراتيجية الترند الجانبي الذكي لم تتحقق",
    "Bullish Reversal Candle Pattern Failed": "لم يظهر نمط شمعة انعكاسية صاعدة",
    "Signal Candle Volume Too Low": "حجم تداول شمعة الإشارة منخفض",
    "Order Book Filter Failed": "فشل فلتر دفتر الطلبات (Bids/Asks)",
    "Order Book Fetch Failed": "فشل جلب دفتر الطلبات",
    "Invalid Position Size": "حجم الصفقة غير صالح",
    "Lot Size Adjustment Failed": "فشل ضبط حجم العقد",
    "Min Notional Filter": "قيمة الصفقة أقل من الحد الأدنى",
    "Insufficient Balance": "الرصيد غير كافٍ",
    "Insufficient data for TP/SL calculation": "بيانات غير كافية لحساب TP/SL",
    "Insufficient Historical Data": "بيانات تاريخية غير كافية للفحص",
    "HTF Trend Confirmation Failed": "فشل تأكيد الترند على الفريم الأعلى",
    "Short-Term Momentum Filter Failed": "فشل فلتر الزخم قصير الأجل",
}

# --- دالة إرسال رسائل تليجرام ---
def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("[تليجرام] Token أو Chat ID غير معين، تم تخطي الإرسال.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("[تليجرام] تم إرسال الرسالة بنجاح.")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [تليجرام] فشل إرسال الرسالة: {e}")

# --- دوال تهيئة الخدمات ---
def init_db(retries: int = 5, delay: int = 5) -> None:
    global conn
    logger.info("[قاعدة البيانات] تهيئة الاتصال...")
    db_url_to_use = DB_URL
    if 'postgres' in db_url_to_use and 'sslmode' not in db_url_to_use:
        db_url_to_use += f"{'?' if '?' not in db_url_to_use else '&'}sslmode=require"
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(db_url_to_use, connect_timeout=15, cursor_factory=RealDictCursor)
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION NOT NULL,
                        target_price DOUBLE PRECISION NOT NULL, stop_loss DOUBLE PRECISION NOT NULL,
                        status TEXT DEFAULT 'open', closing_price DOUBLE PRECISION, closed_at TIMESTAMP,
                        profit_percentage DOUBLE PRECISION, strategy_name TEXT, signal_details JSONB,
                        current_peak_price DOUBLE PRECISION, is_real_trade BOOLEAN DEFAULT FALSE,
                        quantity DOUBLE PRECISION, order_id TEXT, closing_reason TEXT
                    );
                """)
                cur.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS journey_state JSONB;")
                cur.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS original_quantity DOUBLE PRECISION;")
                cur.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS rr_ratio DOUBLE PRECISION;")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals (status);")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL, message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE
                    );
                """)
            conn.commit()
            logger.info("✅ [قاعدة البيانات] الاتصال وتحديث المخطط بنجاح.")
            return
        except Exception as e:
            logger.error(f"❌ [قاعدة البيانات] خطأ أثناء التهيئة (محاولة {attempt + 1}/{retries}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: logger.critical("❌ [قاعدة البيانات] فشل الاتصال.")

def check_db_connection() -> bool:
    global conn
    if conn is None or conn.closed != 0:
        logger.warning("[قاعدة البيانات] الاتصال مغلق، محاولة إعادة الاتصال...")
        init_db()
    try:
        if conn and conn.closed == 0:
            with conn.cursor() as cur: cur.execute("SELECT 1;")
            return True
        return False
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [قاعدة البيانات] فقدان الاتصال: {e}. إعادة الاتصال...")
        try:
            init_db()
            return conn is not None and conn.closed == 0
        except Exception as retry_e:
            logger.error(f"❌ [قاعدة البيانات] فشل إعادة الاتصال: {retry_e}")
            return False

def log_and_notify(level: str, message: str, notification_type: str):
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error, 'critical': logger.critical}
    log_methods.get(level.lower(), logger.info)(message)
    if not check_db_connection() or not conn: return
    try:
        new_notification = {"timestamp": datetime.now(timezone.utc).isoformat(), "type": notification_type, "message": message}
        with notifications_lock: notifications_cache.appendleft(new_notification)
        with conn.cursor() as cur: cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [قاعدة البيانات] فشل حفظ الإشعار: {e}")
        if conn: conn.rollback()

def log_rejection(symbol: str, reason_key: str, details: Optional[Dict] = None):
    reason_ar = REJECTION_REASONS_AR.get(reason_key, reason_key)
    log_message = f"🚫 [{symbol}] تم الرفض | السبب: {reason_ar} | تفاصيل: {details or {}}"
    logger.info(log_message)
    with rejection_logs_lock:
        rejection_logs_cache.appendleft({
            "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
            "reason": reason_ar, "details": json.loads(json.dumps(details, cls=NpEncoder)) or {}
        })

def init_redis() -> None:
    global redis_client
    logger.info("[Redis] تهيئة الاتصال...")
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] تم الاتصال بنجاح.")
    except redis.exceptions.ConnectionError as e:
        logger.critical(f"❌ [Redis] فشل الاتصال: {e}")
        exit(1)

def get_exchange_info_map() -> None:
    global exchange_info_map
    if not client: return
    logger.info("ℹ️ [معلومات المنصة] جاري جلب قواعد التداول...")
    try:
        info = client.get_exchange_info()
        exchange_info_map = {s['symbol']: s for s in info['symbols']}
        logger.info(f"✅ [معلومات المنصة] تم تحميل القواعد لـ {len(exchange_info_map)} عملة.")
    except Exception as e:
        logger.error(f"❌ [معلومات المنصة] فشل جلب المعلومات: {e}")

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    if not client: return []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)

        if not os.path.exists(file_path):
            logger.critical(f"❌ [التحقق من الرموز] ملف العملات '{filename}' غير موجود!")
            return []

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}

        if not raw_symbols:
            logger.warning(f"⚠️ [التحقق من الرموز] ملف العملات '{filename}' فارغ.")
            return []

        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        if not exchange_info_map: get_exchange_info_map()

        active = {s for s, info in exchange_info_map.items() if info.get('quoteAsset') == 'USDT' and info.get('status') == 'TRADING'}

        validated = sorted(list(formatted.intersection(active)))

        logger.info(f"✅ [التحقق من الرموز] تم العثور على {len(validated)} عملة صالحة للتداول.")
        if not validated:
             logger.warning(f"⚠️ [التحقق من الرموز] لا توجد عملات متطابقة في ملفك مع المتاح في المنصة.")
        else:
            logger.info(f"🔍 [التحقق من الرموز] عينة من العملات للمراقبة: {validated[:5]}")

        return validated
    except Exception as e:
        logger.error(f"❌ [التحقق من الرموز] خطأ: {e}", exc_info=True)
        return []

# --- دوال جلب البيانات وحساب المؤشرات ---
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        lookback_str = f"{days + 50} day" if 'd' in interval.lower() else f"{days * 24 + 200} hour"
        
        klines = client.get_historical_klines(symbol, interval, lookback_str)
        if not klines: return None
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(klines, columns=cols)
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base']
        df = df[required_cols]
        numeric_cols = {'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float', 'quote_volume': 'float', 'taker_buy_base': 'float'}
        df = df.astype(numeric_cols)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna()
    except Exception as e:
        logger.error(f"❌ [جلب البيانات] خطأ في جلب البيانات التاريخية لـ {symbol} ({interval}): {e}")
        return None

def calculate_advanced_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    highest_high = df['high'].rolling(window=14).max()
    lowest_low = df['low'].rolling(window=14).min()
    df['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low).replace(0, 1e-9)
    df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low).replace(0, 1e-9)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    bb_period = 20
    df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
    bb_std = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, 1e-9)
    df['kc_middle'] = df['close'].ewm(span=20, adjust=False).mean()
    if 'atr' in df.columns:
        df['kc_upper'] = df['kc_middle'] + (df['atr'] * 1.5)
        df['kc_lower'] = df['kc_middle'] - (df['atr'] * 1.5)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    money_ratio = positive_flow / negative_flow.replace(0, 1e-9)
    df['mfi'] = 100 - (100 / (1 + money_ratio))
    return df

def calculate_market_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ['taker_buy_base', 'volume', 'quote_volume', 'high', 'low', 'open', 'close']
    if not all(col in df.columns for col in required_cols): return df
    df['buy_pressure'] = df['taker_buy_base'] / df['volume'].replace(0, 1e-9)
    volume_ma = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / volume_ma.replace(0, 1e-9)
    df['price_impact'] = df['quote_volume'] / df['volume'].replace(0, 1e-9)
    log_hl = np.log(df['high'] / df['low'].replace(0, 1e-9))
    log_co = np.log(df['close'] / df['open'].replace(0, 1e-9))
    gk_vol_sq = (0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)).clip(lower=0)
    df['garman_klass_vol'] = np.sqrt(gk_vol_sq)
    log_hc = np.log(df['high'] / df['close'].replace(0, 1e-9))
    log_ho = np.log(df['high'] / df['open'].replace(0, 1e-9))
    log_lc = np.log(df['low'] / df['close'].replace(0, 1e-9))
    log_lo = np.log(df['low'] / df['open'].replace(0, 1e-9))
    rs_vol_sq = (log_hc * log_ho + log_lc * log_lo).clip(lower=0)
    df['rogers_satchell_vol'] = np.sqrt(rs_vol_sq)
    return df

def calculate_advanced_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    high_low = df['high'] - df['low']
    ema_high_low = high_low.ewm(span=10, adjust=False).mean()
    ema_high_low_shifted = ema_high_low.shift(10)
    df['chaikin_volatility'] = (ema_high_low - ema_high_low_shifted) / ema_high_low_shifted.replace(0, 1e-9) * 100
    period = 14
    max_close = df['close'].rolling(window=period).max()
    percentage_drawdown = 100 * (df['close'] - max_close) / max_close.replace(0, 1e-9)
    df['ulcer_index'] = np.sqrt((percentage_drawdown ** 2).rolling(window=period).mean())
    if 'atr' not in df.columns: return df
    high_low_tr = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift()).abs()
    low_close_prev = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low_tr, high_close_prev, low_close_prev], axis=1).max(axis=1)
    for p in [5, 10, 20]:
        atr_p = tr.ewm(span=p, adjust=False).mean()
        df[f'atr_ratio_{p}'] = df['atr'] / atr_p.replace(0, 1e-9)
    return df

def calculate_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['asia_session'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
    df['london_session'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
    df['ny_session'] = ((df.index.hour >= 13) & (df.index.hour < 21)).astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    return df

def calculate_supertrend(df: pd.DataFrame, atr_period: int, multiplier: float) -> pd.DataFrame:
    high = df['high']
    low = df['low']
    close = df['close']
    high_low = high - low
    high_close_prev = np.abs(high - close.shift(1))
    low_close_prev = np.abs(low - close.shift(1))
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = tr.ewm(com=atr_period - 1, min_periods=atr_period, adjust=False).mean()
    hl2 = (high + low) / 2
    final_upper_band = upper_band = hl2 + (multiplier * atr)
    final_lower_band = lower_band = hl2 - (multiplier * atr)
    supertrend = pd.Series(np.nan, index=df.index)
    supertrend_direction = pd.Series(np.nan, index=df.index)
    for i in range(1, len(df)):
        curr, prev = i, i - 1
        if close[curr] > final_upper_band[prev]:
            supertrend_direction[curr] = 1
        elif close[curr] < final_lower_band[prev]:
            supertrend_direction[curr] = -1
        else:
            supertrend_direction[curr] = supertrend_direction[prev]
            if supertrend_direction[curr] == -1 and final_upper_band[curr] < final_upper_band[prev]:
                final_upper_band[curr] = final_upper_band[curr]
            if supertrend_direction[curr] == 1 and final_lower_band[curr] > final_lower_band[prev]:
                final_lower_band[curr] = final_lower_band[curr]
        if supertrend_direction[curr] == 1:
            supertrend[curr] = final_lower_band[curr]
        else:
            supertrend[curr] = final_upper_band[curr]
    df['supertrend'] = supertrend
    df['supertrend_direction'] = supertrend_direction
    return df

def calculate_all_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df_calc = df.copy()
    df_calc['ema_9'] = df_calc['close'].ewm(span=9, adjust=False).mean()
    df_calc['ema_21'] = df_calc['close'].ewm(span=21, adjust=False).mean()
    df_calc['sma_50'] = df_calc['close'].rolling(window=50).mean()
    df_calc['sma_200'] = df_calc['close'].rolling(window=200).mean()
    df_calc['volume_sma_20'] = df_calc['volume'].rolling(window=20).mean()
    df_calc['ema_50'] = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    df_calc['ema_100'] = df_calc['close'].ewm(span=100, adjust=False).mean()
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    up_move = df_calc['high'].diff()
    down_move = -df_calc['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    df_calc['plus_di'] = plus_di
    df_calc['minus_di'] = minus_di
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    STOCH_RSI_PERIOD = 14
    STOCH_RSI_K_PERIOD = 3
    STOCH_RSI_D_PERIOD = 3
    rsi = df_calc['rsi']
    stoch_rsi_val = (rsi - rsi.rolling(STOCH_RSI_PERIOD).min()) / (rsi.rolling(STOCH_RSI_PERIOD).max() - rsi.rolling(STOCH_RSI_PERIOD).min()).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(STOCH_RSI_K_PERIOD).mean() * 100
    df_calc['stoch_rsi_d'] = df_calc['stoch_rsi_k'].rolling(STOCH_RSI_D_PERIOD).mean()
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)
    df_calc['price_vs_ema50'] = (df_calc['close'] / df_calc['ema_50']) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / df_calc['close'].ewm(span=200, adjust=False).mean()) - 1
    if btc_df is not None and not btc_df.empty:
        asset_returns = df_calc['close'].pct_change()
        if 'btc_returns' not in btc_df.columns:
            btc_df['btc_returns'] = btc_df['close'].pct_change()
        merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
        df_calc['btc_correlation'] = asset_returns.rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    else:
        df_calc['btc_correlation'] = 0.0
    df_calc = calculate_advanced_momentum_features(df_calc)
    df_calc['bb_width'] = (df_calc['bb_upper'] - df_calc['bb_lower']) / df_calc['bb_middle'].replace(0, 1e-9)
    df_calc = calculate_market_microstructure_features(df_calc)
    df_calc = calculate_advanced_volatility_features(df_calc)
    df_calc = calculate_temporal_features(df_calc)
    df_calc = calculate_supertrend(df_calc, SUPERTREND_ATR_PERIOD, SUPERTREND_MULTIPLIER)
    df_calc[f'roc_{MOMENTUM_PERIOD}'] = (df_calc['close'] / df_calc['close'].shift(MOMENTUM_PERIOD) - 1) * 100
    df_calc['roc_acceleration'] = df_calc[f'roc_{MOMENTUM_PERIOD}'].diff()
    ema_slope = df_calc['close'].ewm(span=EMA_SLOPE_PERIOD, adjust=False).mean()
    df_calc[f'ema_slope_{EMA_SLOPE_PERIOD}'] = (ema_slope - ema_slope.shift(1)) / ema_slope.shift(1).replace(0, 1e-9) * 100
    return df_calc.astype('float32', errors='ignore')

def get_session_state() -> Tuple[List[str], str, str]:
    sessions = {"London": (8, 17), "New York": (13, 22), "Tokyo": (0, 9)}
    active_sessions = []
    now_utc = datetime.now(timezone.utc)
    current_hour = now_utc.hour
    if now_utc.weekday() >= 5: return [], "WEEKEND", "عطلة نهاية الأسبوع"
    
    for session, (start, end) in sessions.items():
        if start > end:
            if current_hour >= start or current_hour < end:
                active_sessions.append(session)
        elif start <= current_hour < end:
            active_sessions.append(session)

    if "London" in active_sessions and "New York" in active_sessions:
        return active_sessions, "HIGH_LIQUIDITY", "تداخل لندن/نيويورك"
    elif len(active_sessions) >= 1:
        return active_sessions, "NORMAL_LIQUIDITY", f"{', '.join(active_sessions)}"
    else:
        return [], "LOW_LIQUIDITY", "خارج أوقات الذروة"

def get_btc_data_for_bot() -> Optional[pd.DataFrame]:
    btc_data = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
    if btc_data is not None: btc_data['btc_returns'] = btc_data['close'].pct_change()
    return btc_data

def load_open_signals_to_cache():
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status IN ('open', 'updated');")
            open_signals = cur.fetchall()
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in open_signals: open_signals_cache[signal['symbol']] = dict(signal)
            logger.info(f"✅ [تحميل] تم تحميل {len(open_signals)} صفقة مفتوحة إلى الذاكرة المؤقتة.")
    except Exception as e:
        logger.error(f"❌ [تحميل] فشل تحميل الصفقات المفتوحة: {e}")

def load_notifications_to_cache():
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 50;")
            recent = cur.fetchall()
            with notifications_lock:
                notifications_cache.clear()
                for n in reversed(recent):
                    n['timestamp'] = n['timestamp'].isoformat()
                    notifications_cache.appendleft(dict(n))
            logger.info(f"✅ [تحميل] تم تحميل {len(notifications_cache)} إشعار إلى الذاكرة المؤقتة.")
    except Exception as e:
        logger.error(f"❌ [تحميل] فشل تحميل الإشعارات: {e}")

# ---------------------- منطق التداول والفلاتر ----------------------

# --- فلاتر التأكيد ---
def check_market_volatility_filter(df: pd.DataFrame) -> bool:
    """فلتر لتجنب التداول في فترات التقلب الشديد أو المنخفض جدًا"""
    if len(df) < 50:
        return False
    
    last = df.iloc[-1]
    if 'atr' not in last or 'close' not in last or last['close'] == 0:
        return False
        
    atr_percent = (last['atr'] / last['close']) * 100
    
    if atr_percent < 0.5 or atr_percent > 5.0:
        log_rejection(df.name, "Market Volatility Filter Failed", {"atr_percent": f"{atr_percent:.2f}"})
        return False
    
    return True

def check_trend_strength_filter(df: pd.DataFrame) -> bool:
    """فلتر لتأكيد قوة الاتجاه الحالي"""
    if len(df) < 50:
        return False
        
    last = df.iloc[-1]
    
    if 'adx' not in last or f'roc_{MOMENTUM_PERIOD}' not in last:
        return False

    if last['adx'] < 18:
        log_rejection(df.name, "Trend Strength Filter Failed", {"reason": "ADX too low", "adx": f"{last['adx']:.2f}"})
        return False
        
    if abs(last[f'roc_{MOMENTUM_PERIOD}']) < 0.5:
        log_rejection(df.name, "Trend Strength Filter Failed", {"reason": "ROC too low", "roc_10": f"{last[f'roc_{MOMENTUM_PERIOD}']:.2f}"})
        return False
        
    return True

# --- الاستراتيجيات الجديدة ---
def check_smart_reversal_strategy(df: pd.DataFrame) -> bool:
    """
    استراتيجية الانعكاس الذكي - مناسبة للسوق الهابط لاقتناص الانعكاسات
    البحث عن إشارات انعكاس قوية بعد هبوط حاد مع تأكيدات متعددة
    """
    if len(df) < 30:
        return False
    
    last, prev, prev_prev = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    
    # الشرط 1: السعر قريب أو تحت الحد السفلي لبولينجر باند
    price_near_lower_bb = last['close'] <= last['bb_lower'] * 1.01  # ضمن 1% من الحد السفلي
    
    # الشرط 2: ستوكاستيك في منطقة التشبع البيعي ويبدأ في الارتفاع
    stoch_oversold = prev['stoch_rsi_k'] < 20 and last['stoch_rsi_k'] > prev['stoch_rsi_k']
    
    # الشرط 3: RSI في منطقة التشبع البيعي ويبدأ في الارتفاع
    rsi_oversold = prev['rsi'] < 30 and last['rsi'] > prev['rsi']
    
    # الشرط 4: وجود نمط شمعة انعكاسي (مطرقة، شهاب، إلخ)
    reversal_pattern = is_bullish_reversal_pattern(df)
    
    # الشرط 5: حجم التداول أعلى من المتوسط (تأكيد)
    volume_confirmation = last['volume'] > last['volume_sma_20'] * 1.2
    
    # الشرط 6: التحقق من وجود هبوط حاد في الشموع السابقة
    sharp_decline = (prev_prev['close'] - prev['close']) / prev_prev['close'] > 0.02  # هبوط أكثر من 2%
    
    # الشرط 7: الماكد يبدأ في الارتفاع من تحت الصفر
    macd_reversal = prev['macd'] < 0 and last['macd'] > prev['macd'] and last['macd_histogram'] > 0
    
    conditions = {
        "price_near_lower_bb": price_near_lower_bb,
        "stoch_oversold": stoch_oversold,
        "rsi_oversold": rsi_oversold,
        "reversal_pattern": reversal_pattern,
        "volume_confirmation": volume_confirmation,
        "sharp_decline": sharp_decline,
        "macd_reversal": macd_reversal
    }
    
    if all(conditions.values()):
        logger.info(f"  -> [{df.name}] ✅ إشارة استراتيجية الانعكاس الذكي.")
        return True
    
    # تسجيل الشروط الفاشلة للمساعدة في التحليل
    failed_conditions = {k: v for k, v in conditions.items() if not v}
    if price_near_lower_bb and len(failed_conditions) > 0:
         log_rejection(df.name, "Smart Reversal Strategy Conditions Not Met", {"failed": list(failed_conditions.keys())})
    
    return False

def check_sustainable_momentum_strategy(df: pd.DataFrame) -> bool:
    """
    استراتيجية الزخم المستدام - مناسبة للسوق الصاعد
    البحث عن زخم قوي ومستمر مع تأكيدات متعددة
    """
    if len(df) < 50:
        return False
    
    last, prev = df.iloc[-1], df.iloc[-2]
    
    # الشرط 1: اتجاه صاعد قوي (ADX فوق 25 و+DI فوق -DI)
    strong_trend = last['adx'] > 25 and last['plus_di'] > last['minus_di']
    
    # الشرط 2: السعر فوق المتوسطات الحركية الأسيّة
    price_above_ema = last['close'] > last['ema_21'] and last['close'] > last['ema_50']
    
    # الشرط 3: MACD إيجابي وفوق إشارته
    macd_positive = last['macd'] > last['macd_signal'] and last['macd'] > 0
    
    # الشرط 4: RSI فوق 50 ولكن ليس في منطقة التشبع الشرائي المفرط
    rsi_good = 50 < last['rsi'] < 70
    
    # الشرط 5: حجم التداول يؤكد الزخم
    volume_confirmation = last['volume'] > last['volume_sma_20'] * 1.1
    
    # الشرط 6: السعر يحقق قمم وقيعان أعلى (Higher Highs and Higher Lows)
    if len(df) >= 10:
        recent_highs = df['high'].iloc[-10:]
        recent_lows = df['low'].iloc[-10:]
        
        # التحقق من وجود قمم وقيعان أعلى
        higher_highs = recent_highs.iloc[-1] > recent_highs.iloc[-5] and recent_highs.iloc[-5] > recent_highs.iloc[-9]
        higher_lows = recent_lows.iloc[-1] > recent_lows.iloc[-5] and recent_lows.iloc[-5] > recent_lows.iloc[-9]
        
        price_structure = higher_highs and higher_lows
    else:
        price_structure = True
    
    # الشرط 7: مؤشر الزخم (ROC) إيجابي
    momentum_positive = last[f'roc_{MOMENTUM_PERIOD}'] > 0.5
    
    conditions = {
        "strong_trend": strong_trend,
        "price_above_ema": price_above_ema,
        "macd_positive": macd_positive,
        "rsi_good": rsi_good,
        "volume_confirmation": volume_confirmation,
        "price_structure": price_structure,
        "momentum_positive": momentum_positive
    }
    
    if all(conditions.values()):
        logger.info(f"  -> [{df.name}] ✅ إشارة استراتيجية الزخم المستدام.")
        return True
    
    # تسجيل الشروط الفاشلة للمساعدة في التحليل
    failed_conditions = {k: v for k, v in conditions.items() if not v}
    if strong_trend and len(failed_conditions) > 0:
         log_rejection(df.name, "Sustainable Momentum Strategy Conditions Not Met", {"failed": list(failed_conditions.keys())})
    
    return False

def check_smart_sideways_strategy(df: pd.DataFrame) -> bool:
    """
    استراتيجية الترند الجانبي الذكي - مناسبة للسوق الجانبي
    البحث عن فرص تداول عند حدود نطاق التداول
    """
    if len(df) < 100:
        return False
    
    last, prev = df.iloc[-1], df.iloc[-2]
    
    # الشرط 1: تحديد نطاق تداول جانبي
    # نحسب أعلى وأدنى مستويات في آخر 50 شمعة
    recent_high = df['high'].iloc[-50:].max()
    recent_low = df['low'].iloc[-50:].min()
    range_size = recent_high - recent_low
    range_threshold = last['close'] * 0.05  # 5% من السعر الحالي
    
    # نطاق تداول جانبي إذا كان الفرق بين الأعلى والأدنى أقل من 5%
    is_sideways = range_size < range_threshold
    
    # الشرط 2: السعر قريب من الحد السفلي للنطاق
    price_near_low = last['close'] <= recent_low * 1.01  # ضمن 1% من الحد الأدنى
    
    # الشرط 3: ستوكاستيك في منطقة التشبع البيعي ويبدأ في الارتفاع
    stoch_turning_up = prev['stoch_rsi_k'] < 30 and last['stoch_rsi_k'] > prev['stoch_rsi_k']
    
    # الشرط 4: RSI في منطقة التشبع البيعي ويبدأ في الارتفاع
    rsi_turning_up = prev['rsi'] < 40 and last['rsi'] > prev['rsi']
    
    # الشرط 5: حجم التداول يؤكد الإشارة
    volume_confirmation = last['volume'] > last['volume_sma_20'] * 1.15
    
    # الشرط 6: ADX منخفض (يشير إلى عدم وجود اتجاه قوي)
    no_strong_trend = last['adx'] < 20
    
    # الشرط 7: بولينجر باند ضيقة (تشير إلى تقلب منخفض)
    bb_squeeze = last['bb_width'] < df['bb_width'].rolling(50).mean().iloc[-1] * 0.8
    
    # الشرط 8: مؤشر القناة السلعية (CCI) يشير إلى ذروة بيع
    if 'cci' not in df.columns:
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.fabs(x - x.mean()).mean())
        df['cci'] = (tp - sma_tp) / (0.015 * mad)
    
    cci_oversold = prev['cci'] < -150 and last['cci'] > prev['cci']
    
    conditions = {
        "is_sideways": is_sideways,
        "price_near_low": price_near_low,
        "stoch_turning_up": stoch_turning_up,
        "rsi_turning_up": rsi_turning_up,
        "volume_confirmation": volume_confirmation,
        "no_strong_trend": no_strong_trend,
        "bb_squeeze": bb_squeeze,
        "cci_oversold": cci_oversold
    }
    
    if all(conditions.values()):
        logger.info(f"  -> [{df.name}] ✅ إشارة استراتيجية الترند الجانبي الذكي.")
        return True
    
    # تسجيل الشروط الفاشلة للمساعدة في التحليل
    failed_conditions = {k: v for k, v in conditions.items() if not v}
    if is_sideways and price_near_low and len(failed_conditions) > 0:
         log_rejection(df.name, "Smart Sideways Strategy Conditions Not Met", {"failed": list(failed_conditions.keys())})
    
    return False

# --- دالة تأكيد الترند على فريم أعلى ---
def is_htf_bullish_confirmation(symbol: str, htf: str = '1h', lookback: int = 200) -> bool:
    try:
        df = fetch_historical_data(symbol, htf, days=40) 
        if df is None or len(df) < lookback:
            logger.warning(f"  -> [HTF {htf}] {symbol} بيانات غير كافية للتأكيد ({len(df) if df is not None else 0} شمعة).")
            return False

        df['ema50']  = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

        tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low']  - df['close'].shift()).abs()], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        plus_dm = np.where((df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']), df['high'] - df['high'].shift(), 0)
        minus_dm = np.where((df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()), df['low'].shift() - df['low'], 0)
        plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / df['atr'].replace(0, 1e-9)
        minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / df['atr'].replace(0, 1e-9)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        df['adx'] = dx.rolling(14).mean()

        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        strong_uptrend = (last['ema50'] > last['ema200'] and last['adx'] > 25 and last['close'] > last['ema50'])
        macd_cross_up = prev['macd'] < prev['signal_line'] and last['macd'] > last['signal_line']
        ema_cross_up  = prev['ema50'] < prev['ema200'] and last['ema50'] > last['ema200']
        recent_bullish_flip = macd_cross_up and ema_cross_up

        is_confirmed = strong_uptrend or recent_bullish_flip
        logger.info(f"  -> [HTF {htf}] {symbol} تأكيد الترند: {is_confirmed} (قوي: {strong_uptrend} | تحول: {recent_bullish_flip})")
        return is_confirmed

    except Exception as e:
        logger.error(f"❌ [HTF Confirm] خطأ في {symbol}: {e}")
        return False

# --- دالة فلتر الزخم قصير الأجل ---
def passes_short_term_momentum_filter(symbol: str, df: pd.DataFrame) -> bool:
    if len(df) < 100:
        return False

    last = df.iloc[-1]
    
    bb_width = last.get('bb_width', 0)
    price_vs_bb_upper = abs(last['close'] - last['bb_upper']) / last['bb_upper'] if last['bb_upper'] > 0 else 0
    price_vs_bb_lower = abs(last['close'] - last['bb_lower']) / last['bb_lower'] if last['bb_lower'] > 0 else 0
    
    close_to_bands = price_vs_bb_upper < 0.005 or price_vs_bb_lower < 0.005
    
    with volume_filter_lock:
        vol_mult = VOLUME_FILTER_MULTIPLIER
    volume_spike = last.get('relative_volume', 0) > vol_mult
    
    macd_momentum = last['macd'] > last['macd_signal'] and last['macd'] > 0
    rsi_momentum  = last['rsi'] > 55
    
    squeeze_threshold = df['bb_width'].rolling(100).quantile(0.25).iloc[-1]
    is_squeeze = bb_width < squeeze_threshold
    
    price_momentum = last['close'] > last['ema_9'] and last['close'] > df['close'].iloc[-4] # Check against 3 candles ago
    
    trend_strength = last['adx'] > 20
    
    is_valid = (
        (is_squeeze or close_to_bands) and
        volume_spike and
        (macd_momentum or rsi_momentum) and
        price_momentum and
        trend_strength
    )
    
    logger.info(f"  -> [فلتر الزخم المحسن] {symbol}: Valid={is_valid}")
    return is_valid

# --- التعرف على أنماط الشموع ---
def is_bullish_reversal_pattern(df: pd.DataFrame) -> bool:
    if len(df) < 3: return False
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    patterns = {
        "Hammer": is_hammer(c3, c2), "Inverse Hammer": is_inverse_hammer(c3, c2),
        "Bullish Engulfing": is_bullish_engulfing(c3, c2), "Piercing Line": is_piercing_line(c3, c2),
        "Morning Star": is_morning_star(c1, c2, c3), "Three White Soldiers": is_three_white_soldiers(c1, c2, c3)
    }
    for pattern_name, is_present in patterns.items():
        if is_present:
            logger.info(f"  -> [{df.name}] ✅ نمط شمعة صاعدة: {pattern_name}")
            return True
    return False

def is_hammer(candle: pd.Series, prev_candle: pd.Series) -> bool:
    body = abs(candle['open'] - candle['close'])
    lower_wick = candle['close'] - candle['low'] if candle['open'] < candle['close'] else candle['open'] - candle['low']
    upper_wick = candle['high'] - candle['close'] if candle['open'] < candle['close'] else candle['high'] - candle['open']
    return body > 0 and lower_wick > 2 * body and upper_wick < body

def is_inverse_hammer(candle: pd.Series, prev_candle: pd.Series) -> bool:
    body = abs(candle['open'] - candle['close'])
    lower_wick = candle['close'] - candle['low'] if candle['open'] < candle['close'] else candle['open'] - candle['low']
    upper_wick = candle['high'] - candle['close'] if candle['open'] < candle['close'] else candle['high'] - candle['open']
    return body > 0 and upper_wick > 2 * body and lower_wick < body

def is_bullish_engulfing(candle: pd.Series, prev_candle: pd.Series) -> bool:
    return (prev_candle['close'] < prev_candle['open'] and
            candle['close'] > candle['open'] and
            candle['close'] > prev_candle['open'] and
            candle['open'] < prev_candle['close'])

def is_piercing_line(candle: pd.Series, prev_candle: pd.Series) -> bool:
    midpoint = (prev_candle['open'] + prev_candle['close']) / 2
    return (prev_candle['close'] < prev_candle['open'] and
            candle['close'] > candle['open'] and
            candle['open'] < prev_candle['low'] and
            candle['close'] > midpoint and
            candle['close'] < prev_candle['open'])

def is_morning_star(c1: pd.Series, c2: pd.Series, c3: pd.Series) -> bool:
    c1_body = abs(c1['open'] - c1['close'])
    c3_body = abs(c3['open'] - c3['close'])
    return (c1['close'] < c1['open'] and  # First candle red
            abs(c2['open'] - c2['close']) < c1_body * 0.5 and  # Second candle small body
            c3['close'] > c3['open'] and  # Third candle green
            c3['close'] > (c1['open'] + c1['close']) / 2)  # Third candle closes above midpoint of first

def is_three_white_soldiers(c1: pd.Series, c2: pd.Series, c3: pd.Series) -> bool:
    return (c1['close'] > c1['open'] and  # First candle green
            c2['close'] > c2['open'] and  # Second candle green
            c3['close'] > c3['open'] and  # Third candle green
            c2['close'] > c1['close'] and  # Second candle closes higher than first
            c3['close'] > c2['close'] and  # Third candle closes higher than second
            c1['open'] < c1['close'] and  # First candle opens lower than it closes
            c2['open'] > c1['open'] and c2['open'] < c1['close'] and  # Second candle opens within first candle body
            c3['open'] > c2['open'] and c3['open'] < c2['close'])  # Third candle opens within second candle body

# --- دالة التحقق من الاستراتيجيات ---
def check_strategies_for_symbol(symbol: str, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    التحقق من جميع الاستراتيجيات المفعلة للرمز المحدد
    """
    if df_15m is None or df_15m.empty:
        logger.warning(f"  -> [{symbol}] لا توجد بيانات للفحص.")
        return None
    
    df_15m.name = symbol  # إضافة اسم الرمز إلى DataFrame لاستخدامه في السجلات
    
    # تطبيق الفلاتر العامة أولاً
    if not check_market_volatility_filter(df_15m):
        return None
    
    if not check_trend_strength_filter(df_15m):
        return None
    
    # التحقق من الاستراتيجيات المفعلة
    strategy_triggered = False
    strategy_name = ""
    
    with smart_reversal_strategy_lock:
        if USE_SMART_REVERSAL_STRATEGY and check_smart_reversal_strategy(df_15m):
            strategy_triggered = True
            strategy_name = "Smart_Reversal"
    
    if not strategy_triggered:
        with sustainable_momentum_strategy_lock:
            if USE_SUSTAINABLE_MOMENTUM_STRATEGY and check_sustainable_momentum_strategy(df_15m):
                strategy_triggered = True
                strategy_name = "Sustainable_Momentum"
    
    if not strategy_triggered:
        with smart_sideways_strategy_lock:
            if USE_SMART_SIDEWAYS_STRATEGY and check_smart_sideways_strategy(df_15m):
                strategy_triggered = True
                strategy_name = "Smart_Sideways"
    
    if not strategy_triggered:
        return None
    
    # التحقق من تأكيد الترند على الفريم الأعلى
    if not is_htf_bullish_confirmation(symbol, HIGHER_TIMEFRAME):
        log_rejection(symbol, "HTF Trend Confirmation Failed")
        return None
    
    # التحقق من فلتر الزخم قصير الأجل
    if not passes_short_term_momentum_filter(symbol, df_15m):
        log_rejection(symbol, "Short-Term Momentum Filter Failed")
        return None
    
    # التحقق من نمط الشمعة الانعكاسية
    if not is_bullish_reversal_pattern(df_15m):
        log_rejection(symbol, "Bullish Reversal Candle Pattern Failed")
        return None
    
    # التحقق من حجم التداول
    last_candle = df_15m.iloc[-1]
    if last_candle['volume'] < last_candle['volume_sma_20'] * VOLUME_FILTER_MULTIPLIER:
        log_rejection(symbol, "Signal Candle Volume Too Low")
        return None
    
    # إذا تم اجتياز جميع الشروط، إنشاء إشارة
    logger.info(f"✅ [{symbol}] جميع الشروط مستوفاة لاستراتيجية {strategy_name}!")
    
    # إعداد تفاصيل الإشارة
    last = df_15m.iloc[-1]
    atr = last['atr']
    
    # حساب مستويات الدخول، الهدف، ووقف الخسارة
    entry_price = last['close']
    
    # حساب وقف الخسارة والهدف بناءً على ATR
    if USE_ATR_TRAILING_STOP:
        stop_loss = entry_price - (atr * ATR_TS_MULTIPLIER)
        target_price = entry_price + (atr * ATR_TS_MULTIPLIER * 2.5)  # نسبة 1:2.5
    else:
        # استخدام بولينجر باند كمرجع
        stop_loss = min(last['bb_lower'], last['low'] - (atr * 0.5))
        target_price = last['bb_upper'] + (atr * 0.5)
    
    # التأكد من أن وقف الخسارة والهدف منطقيان
    if stop_loss >= entry_price:
        stop_loss = entry_price * 0.98  # 2% تحت سعر الدخول
    
    if target_price <= entry_price:
        target_price = entry_price * 1.03  # 3% فوق سعر الدخول
    
    # حساب نسبة المخاطرة إلى العائد
    rr_ratio = (target_price - entry_price) / (entry_price - stop_loss)
    
    # إعداد تفاصيل الإشارة
    signal_details = {
        "strategy": strategy_name,
        "entry_price": float(entry_price),
        "target_price": float(target_price),
        "stop_loss": float(stop_loss),
        "rr_ratio": float(rr_ratio),
        "atr": float(atr),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_state": current_market_state.get("overall_regime", "UNKNOWN"),
        "indicators": {
            "rsi": float(last['rsi']),
            "stoch_k": float(last['stoch_rsi_k']),
            "macd": float(last['macd']),
            "adx": float(last['adx']),
            "bb_position": float(last['bb_position']),
            "volume_ratio": float(last['relative_volume'])
        }
    }
    
    return signal_details

# --- لوحة التحكم ---
app = Flask(__name__)
CORS(app)

# قالب HTML للوحة التحكم
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم بوت التداول V10.0</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #1a1a2e;
            color: #eee;
        }
        .navbar {
            background-color: #16213e !important;
        }
        .card {
            background-color: #0f3460;
            border: none;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
        }
        .card-header {
            background-color: #e94560;
            color: white;
            border-radius: 15px 15px 0 0 !important;
            font-weight: bold;
        }
        .table {
            color: #eee;
        }
        .table thead th {
            background-color: #16213e;
            border-color: #e94560;
        }
        .table tbody tr:hover {
            background-color: rgba(233, 69, 96, 0.1);
        }
        .badge {
            font-size: 0.8em;
        }
        .status-active {
            color: #4caf50;
        }
        .status-inactive {
            color: #f44336;
        }
        .strategy-card {
            transition: transform 0.3s;
        }
        .strategy-card:hover {
            transform: translateY(-5px);
        }
        .signal-row {
            border-left: 4px solid #e94560;
        }
        .notification-item {
            border-right: 3px solid #e94560;
            padding-right: 10px;
            margin-bottom: 10px;
        }
        .rejection-item {
            border-right: 3px solid #ff9800;
            padding-right: 10px;
            margin-bottom: 10px;
        }
        .market-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-left: 5px;
        }
        .bullish {
            background-color: #4caf50;
        }
        .bearish {
            background-color: #f44336;
        }
        .sideways {
            background-color: #ff9800;
        }
        .footer {
            background-color: #16213e;
            color: #aaa;
            text-align: center;
            padding: 20px 0;
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container">
            <a class="navbar-brand" href="#">
                <i class="fas fa-robot"></i> بوت التداول الآلي V10.0
            </a>
            <div class="ms-auto">
                <span class="navbar-text">
                    <span id="trading-status" class="badge bg-danger">متوقف</span>
                </span>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="row">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-chart-line"></i> حالة السوق الحالية
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-3">
                                <div class="text-center">
                                    <h5>النظام العام</h5>
                                    <h3 id="market-regime">جاري التحميل...</h3>
                                    <span id="market-indicator" class="market-indicator"></span>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="text-center">
                                    <h5>جلسة التداول</h5>
                                    <h3 id="session-state">جاري التحميل...</h3>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="text-center">
                                    <h5>العملات المراقبة</h5>
                                    <h3 id="symbols-count">0</h3>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="text-center">
                                    <h5>الصفقات المفتوحة</h5>
                                    <h3 id="open-trades-count">0</h3>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-md-4">
                <div class="card strategy-card">
                    <div class="card-header">
                        <i class="fas fa-arrow-down"></i> استراتيجية الانعكاس الذكي
                    </div>
                    <div class="card-body">
                        <p>مناسبة للسوق الهابط لاقتناص الانعكاسات</p>
                        <div class="d-flex justify-content-between">
                            <span>الحالة:</span>
                            <span id="reversal-strategy-status" class="badge bg-secondary">جاري التحميل...</span>
                        </div>
                        <div class="mt-3">
                            <button class="btn btn-sm btn-outline-primary w-100" onclick="toggleStrategy('reversal')">
                                تبديل الحالة
                            </button>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card strategy-card">
                    <div class="card-header">
                        <i class="fas fa-arrow-up"></i> استراتيجية الزخم المستدام
                    </div>
                    <div class="card-body">
                        <p>مناسبة للسوق الصاعد للاستفادة من الزخم</p>
                        <div class="d-flex justify-content-between">
                            <span>الحالة:</span>
                            <span id="momentum-strategy-status" class="badge bg-secondary">جاري التحميل...</span>
                        </div>
                        <div class="mt-3">
                            <button class="btn btn-sm btn-outline-primary w-100" onclick="toggleStrategy('momentum')">
                                تبديل الحالة
                            </button>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card strategy-card">
                    <div class="card-header">
                        <i class="fas fa-arrows-alt-h"></i> استراتيجية الترند الجانبي
                    </div>
                    <div class="card-body">
                        <p>مناسبة للسوق الجانبي للتداول عند الحدود</p>
                        <div class="d-flex justify-content-between">
                            <span>الحالة:</span>
                            <span id="sideways-strategy-status" class="badge bg-secondary">جاري التحميل...</span>
                        </div>
                        <div class="mt-3">
                            <button class="btn btn-sm btn-outline-primary w-100" onclick="toggleStrategy('sideways')">
                                تبديل الحالة
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-bell"></i> الإشعارات الأخيرة
                    </div>
                    <div class="card-body" style="max-height: 400px; overflow-y: auto;">
                        <div id="notifications-container">جاري التحميل...</div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-times-circle"></i> عمليات الرفض الأخيرة
                    </div>
                    <div class="card-body" style="max-height: 400px; overflow-y: auto;">
                        <div id="rejections-container">جاري التحميل...</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <i class="fas fa-exchange-alt"></i> الصفقات المفتوحة
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>الرمز</th>
                                        <th>الاستراتيجية</th>
                                        <th>سعر الدخول</th>
                                        <th>سعر الهدف</th>
                                        <th>وقف الخسارة</th>
                                        <th>نسبة المخاطرة</th>
                                        <th>الربح الحالي</th>
                                        <th>الإجراءات</th>
                                    </tr>
                                </thead>
                                <tbody id="open-trades-tbody">
                                    <tr>
                                        <td colspan="8" class="text-center">جاري التحميل...</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <footer class="footer">
        <div class="container">
            <p>&copy; 2023 بوت التداول الآلي V10.0 - جميع الحقوق محفوظة</p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // تحديث البيانات كل 5 ثواني
        setInterval(updateDashboard, 5000);
        
        // تحديث لوحة التحكم
        function updateDashboard() {
            // تحديث حالة السوق
            fetch('/api/market_state')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('market-regime').textContent = data.overall_regime_ar || data.overall_regime;
                    const indicator = document.getElementById('market-indicator');
                    indicator.className = 'market-indicator';
                    
                    if (data.overall_regime.includes('BULLISH')) {
                        indicator.classList.add('bullish');
                    } else if (data.overall_regime.includes('BEARISH')) {
                        indicator.classList.add('bearish');
                    } else {
                        indicator.classList.add('sideways');
                    }
                });
            
            // تحديث جلسة التداول
            fetch('/api/session_state')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('session-state').textContent = data.session_ar;
                });
            
            // تحديث حالة التداول
            fetch('/api/trading_status')
                .then(response => response.json())
                .then(data => {
                    const statusElement = document.getElementById('trading-status');
                    statusElement.textContent = data.enabled ? 'مفعل' : 'متوقف';
                    statusElement.className = data.enabled ? 'badge bg-success' : 'badge bg-danger';
                });
            
            // تحديث عدد العملات
            fetch('/api/symbols_count')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('symbols-count').textContent = data.count;
                });
            
            // تحديث عدد الصفقات المفتوحة
            fetch('/api/open_trades_count')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('open-trades-count').textContent = data.count;
                });
            
            // تحديث حالة الاستراتيجيات
            fetch('/api/strategies_status')
                .then(response => response.json())
                .then(data => {
                    updateStrategyStatus('reversal', data.reversal);
                    updateStrategyStatus('momentum', data.momentum);
                    updateStrategyStatus('sideways', data.sideways);
                });
            
            // تحديث الإشعارات
            fetch('/api/notifications')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('notifications-container');
                    container.innerHTML = '';
                    
                    if (data.length === 0) {
                        container.innerHTML = '<p class="text-center">لا توجد إشعارات</p>';
                        return;
                    }
                    
                    data.forEach(notification => {
                        const item = document.createElement('div');
                        item.className = 'notification-item';
                        item.innerHTML = `
                            <div class="d-flex justify-content-between">
                                <span>${notification.message}</span>
                                <small>${new Date(notification.timestamp).toLocaleString('ar-SA')}</small>
                            </div>
                        `;
                        container.appendChild(item);
                    });
                });
            
            // تحديث عمليات الرفض
            fetch('/api/rejections')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('rejections-container');
                    container.innerHTML = '';
                    
                    if (data.length === 0) {
                        container.innerHTML = '<p class="text-center">لا توجد عمليات رفض</p>';
                        return;
                    }
                    
                    data.forEach(rejection => {
                        const item = document.createElement('div');
                        item.className = 'rejection-item';
                        item.innerHTML = `
                            <div class="d-flex justify-content-between">
                                <span><strong>${rejection.symbol}:</strong> ${rejection.reason}</span>
                                <small>${new Date(rejection.timestamp).toLocaleString('ar-SA')}</small>
                            </div>
                        `;
                        container.appendChild(item);
                    });
                });
            
            // تحديث الصفقات المفتوحة
            fetch('/api/open_trades')
                .then(response => response.json())
                .then(data => {
                    const tbody = document.getElementById('open-trades-tbody');
                    tbody.innerHTML = '';
                    
                    if (data.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="8" class="text-center">لا توجد صفقات مفتوحة</td></tr>';
                        return;
                    }
                    
                    data.forEach(trade => {
                        const row = document.createElement('tr');
                        row.className = 'signal-row';
                        
                        const profit = ((trade.current_price - trade.entry_price) / trade.entry_price * 100).toFixed(2);
                        const profitClass = profit >= 0 ? 'text-success' : 'text-danger';
                        
                        row.innerHTML = `
                            <td>${trade.symbol}</td>
                            <td>${trade.strategy_name}</td>
                            <td>${trade.entry_price.toFixed(4)}</td>
                            <td>${trade.target_price.toFixed(4)}</td>
                            <td>${trade.stop_loss.toFixed(4)}</td>
                            <td>${trade.rr_ratio.toFixed(2)}</td>
                            <td class="${profitClass}">${profit}%</td>
                            <td>
                                <button class="btn btn-sm btn-danger" onclick="closeTrade('${trade.symbol}')">
                                    إغلاق
                                </button>
                            </td>
                        `;
                        tbody.appendChild(row);
                    });
                });
        }
        
        // تحديث حالة استراتيجية
        function updateStrategyStatus(strategy, enabled) {
            const statusElement = document.getElementById(`${strategy}-strategy-status`);
            statusElement.textContent = enabled ? 'مفعل' : 'معطل';
            statusElement.className = enabled ? 'badge bg-success' : 'badge bg-danger';
        }
        
        // تبديل حالة استراتيجية
        function toggleStrategy(strategy) {
            const strategyMap = {
                'reversal': 'USE_SMART_REVERSAL_STRATEGY',
                'momentum': 'USE_SUSTAINABLE_MOMENTUM_STRATEGY',
                'sideways': 'USE_SMART_SIDEWAYS_STRATEGY'
            };
            
            fetch('/api/toggle_strategy', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    strategy: strategyMap[strategy]
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updateDashboard();
                } else {
                    alert('فشل تحديث حالة الاستراتيجية');
                }
            });
        }
        
        // إغلاق صفقة
        function closeTrade(symbol) {
            if (confirm(`هل أنت متأكد من إغلاق صفقة ${symbol}؟`)) {
                fetch('/api/close_trade', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        symbol: symbol
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateDashboard();
                    } else {
                        alert('فشل إغلاق الصفقة');
                    }
                });
            }
        }
        
        // تبديل حالة التداول
        function toggleTrading() {
            fetch('/api/toggle_trading', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updateDashboard();
                } else {
                    alert('فشل تحديث حالة التداول');
                }
            });
        }
        
        // تحميل البيانات عند فتح الصفحة
        document.addEventListener('DOMContentLoaded', updateDashboard);
    </script>
</body>
</html>
"""

# نقاط نهاية API للوحة التحكم
@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/market_state')
def get_market_state():
    with market_state_lock:
        return jsonify(current_market_state)

@app.route('/api/session_state')
def get_session_state():
    sessions, liquidity, session_ar = get_session_state()
    return jsonify({
        "sessions": sessions,
        "liquidity": liquidity,
        "session_ar": session_ar
    })

@app.route('/api/trading_status')
def get_trading_status():
    with trading_status_lock:
        return jsonify({"enabled": is_trading_enabled})

@app.route('/api/toggle_trading', methods=['POST'])
def toggle_trading():
    global is_trading_enabled
    with trading_status_lock:
        is_trading_enabled = not is_trading_enabled
        status = "مفعل" if is_trading_enabled else "متوقف"
        log_and_notify("info", f"تم {status} التداول الآلي", "SYSTEM")
        return jsonify({"success": True, "enabled": is_trading_enabled})

@app.route('/api/symbols_count')
def get_symbols_count():
    return jsonify({"count": len(validated_symbols_to_scan)})

@app.route('/api/open_trades_count')
def get_open_trades_count():
    with signal_cache_lock:
        return jsonify({"count": len(open_signals_cache)})

@app.route('/api/strategies_status')
def get_strategies_status():
    with smart_reversal_strategy_lock:
        reversal_status = USE_SMART_REVERSAL_STRATEGY
    with sustainable_momentum_strategy_lock:
        momentum_status = USE_SUSTAINABLE_MOMENTUM_STRATEGY
    with smart_sideways_strategy_lock:
        sideways_status = USE_SMART_SIDEWAYS_STRATEGY
    
    return jsonify({
        "reversal": reversal_status,
        "momentum": momentum_status,
        "sideways": sideways_status
    })

@app.route('/api/toggle_strategy', methods=['POST'])
def toggle_strategy():
    data = request.get_json()
    strategy_name = data.get('strategy')
    
    if strategy_name == 'USE_SMART_REVERSAL_STRATEGY':
        with smart_reversal_strategy_lock:
            global USE_SMART_REVERSAL_STRATEGY
            USE_SMART_REVERSAL_STRATEGY = not USE_SMART_REVERSAL_STRATEGY
            status = USE_SMART_REVERSAL_STRATEGY
    elif strategy_name == 'USE_SUSTAINABLE_MOMENTUM_STRATEGY':
        with sustainable_momentum_strategy_lock:
            global USE_SUSTAINABLE_MOMENTUM_STRATEGY
            USE_SUSTAINABLE_MOMENTUM_STRATEGY = not USE_SUSTAINABLE_MOMENTUM_STRATEGY
            status = USE_SUSTAINABLE_MOMENTUM_STRATEGY
    elif strategy_name == 'USE_SMART_SIDEWAYS_STRATEGY':
        with smart_sideways_strategy_lock:
            global USE_SMART_SIDEWAYS_STRATEGY
            USE_SMART_SIDEWAYS_STRATEGY = not USE_SMART_SIDEWAYS_STRATEGY
            status = USE_SMART_SIDEWAYS_STRATEGY
    else:
        return jsonify({"success": False, "error": "استراتيجية غير معروفة"})
    
    strategy_display = {
        'USE_SMART_REVERSAL_STRATEGY': 'الانعكاس الذكي',
        'USE_SUSTAINABLE_MOMENTUM_STRATEGY': 'الزخم المستدام',
        'USE_SMART_SIDEWAYS_STRATEGY': 'الترند الجانبي'
    }
    
    status_text = "مفعل" if status else "معطل"
    log_and_notify("info", f"تم {status_text} استراتيجية {strategy_display[strategy_name]}", "SYSTEM")
    
    return jsonify({"success": True, "strategy": strategy_name, "enabled": status})

@app.route('/api/notifications')
def get_notifications():
    with notifications_lock:
        return jsonify(list(notifications_cache))

@app.route('/api/rejections')
def get_rejections():
    with rejection_logs_lock:
        return jsonify(list(rejection_logs_cache))

@app.route('/api/open_trades')
def get_open_trades():
    with signal_cache_lock:
        trades = []
        for symbol, signal in open_signals_cache.items():
            # الحصول على السعر الحالي من Redis
            current_price = 0
            if redis_client:
                try:
                    price_data = redis_client.hget(REDIS_PRICES_HASH_NAME, symbol)
                    if price_data:
                        current_price = float(price_data)
                except Exception as e:
                    logger.error(f"خطأ في جلب السعر الحالي لـ {symbol}: {e}")
            
            # إذا لم يتم العثور على السعر، استخدم سعر الدخول
            if current_price == 0:
                current_price = signal['entry_price']
            
            trades.append({
                "symbol": symbol,
                "strategy_name": signal['strategy_name'],
                "entry_price": signal['entry_price'],
                "target_price": signal['target_price'],
                "stop_loss": signal['stop_loss'],
                "rr_ratio": signal.get('rr_ratio', 0),
                "current_price": current_price
            })
        return jsonify(trades)

@app.route('/api/close_trade', methods=['POST'])
def close_trade():
    data = request.get_json()
    symbol = data.get('symbol')
    
    if not symbol:
        return jsonify({"success": False, "error": "الرمز مطلوب"})
    
    with signal_cache_lock:
        if symbol not in open_signals_cache:
            return jsonify({"success": False, "error": "الصفقة غير موجودة"})
        
        signal = open_signals_cache[symbol]
    
    # تحديث حالة الصفقة في قاعدة البيانات
    if not check_db_connection() or not conn:
        return jsonify({"success": False, "error": "فشل الاتصال بقاعدة البيانات"})
    
    try:
        with conn.cursor() as cur:
            # الحصول على السعر الحالي
            current_price = 0
            if redis_client:
                try:
                    price_data = redis_client.hget(REDIS_PRICES_HASH_NAME, symbol)
                    if price_data:
                        current_price = float(price_data)
                except Exception as e:
                    logger.error(f"خطأ في جلب السعر الحالي لـ {symbol}: {e}")
            
            # إذا لم يتم العثور على السعر، استخدم سعر الدخول
            if current_price == 0:
                current_price = signal['entry_price']
            
            # حساب الربح/الخسارة
            profit_percentage = ((current_price - signal['entry_price']) / signal['entry_price']) * 100
            
            # تحديث الصفقة في قاعدة البيانات
            cur.execute("""
                UPDATE signals 
                SET status = 'closed', closing_price = %s, closed_at = NOW(), 
                    profit_percentage = %s, closing_reason = 'Manual Close'
                WHERE symbol = %s AND status = 'open'
            """, (current_price, profit_percentage, symbol))
            
            conn.commit()
            
            # إزالة الصفقة من الكاش
            with signal_cache_lock:
                if symbol in open_signals_cache:
                    del open_signals_cache[symbol]
            
            log_and_notify("info", f"تم إغلاق صفقة {symbol} يدوياً", "TRADE")
            
            return jsonify({"success": True})
    
    except Exception as e:
        logger.error(f"خطأ في إغلاق صفقة {symbol}: {e}")
        if conn: conn.rollback()
        return jsonify({"success": False, "error": str(e)})

# --- دالة تشغيل البوت ---
def run_bot():
    global client, validated_symbols_to_scan
    
    # تهيئة العميل
    try:
        client = Client(API_KEY, API_SECRET)
        logger.info("✅ [بايننس] تم الاتصال بمنصة بايننس.")
    except Exception as e:
        logger.critical(f"❌ [بايننس] فشل الاتصال: {e}")
        return
    
    # تهيئة قاعدة البيانات
    init_db()
    
    # تهيئة Redis
    init_redis()
    
    # جلب معلومات المنصة
    get_exchange_info_map()
    
    # الحصول على قائمة العملات الصالحة
    validated_symbols_to_scan = get_validated_symbols()
    if not validated_symbols_to_scan:
        logger.critical("❌ لا توجد عملات صالحة للتداول.")
        return
    
    # تحميل الصفقات المفتوحة والإشعارات
    load_open_signals_to_cache()
    load_notifications_to_cache()
    
    # بدء تشغيل خادم Flask في خيط منفصل
    flask_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, threaded=True))
    flask_thread.daemon = True
    flask_thread.start()
    logger.info("✅ [لوحة التحكم] تم تشغيل لوحة التحكم على http://localhost:5000")
    
    # الحلقة الرئيسية للبوت
    while True:
        try:
            # تحديث حالة السوق
            update_market_state()
            
            # إذا كان التداول مفعلًا
            with trading_status_lock:
                trading_enabled = is_trading_enabled
            
            if trading_enabled and len(open_signals_cache) < MAX_OPEN_TRADES:
                # الحصول على بيانات BTC
                btc_df = get_btc_data_for_bot()
                
                # معالجة العملات في دفعات
                for i in range(0, len(validated_symbols_to_scan), SYMBOL_PROCESSING_BATCH_SIZE):
                    batch_symbols = validated_symbols_to_scan[i:i+SYMBOL_PROCESSING_BATCH_SIZE]
                    
                    for symbol in batch_symbols:
                        # تجاهل العملات التي لديها صفقات مفتوحة
                        with signal_cache_lock:
                            if symbol in open_signals_cache:
                                continue
                        
                        # جلب البيانات
                        df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, 40)
                        
                        if df_15m is None or df_4h is None:
                            continue
                        
                        # حساب المؤشرات
                        df_15m = calculate_all_features(df_15m, btc_df)
                        
                        # التحقق من الاستراتيجيات
                        signal_details = check_strategies_for_symbol(symbol, df_15m, df_4h, btc_df)
                        
                        if signal_details:
                            # حفظ الإشارة في قاعدة البيانات
                            save_signal_to_db(symbol, signal_details)
                            
                            # إضافة الإشارة إلى الكاش
                            with signal_cache_lock:
                                open_signals_cache[symbol] = {
                                    "symbol": symbol,
                                    "strategy_name": signal_details["strategy"],
                                    "entry_price": signal_details["entry_price"],
                                    "target_price": signal_details["target_price"],
                                    "stop_loss": signal_details["stop_loss"],
                                    "rr_ratio": signal_details["rr_ratio"],
                                    "signal_details": signal_details
                                }
                            
                            # إرسال إشعار
                            message = f"🚀 إشارة جديدة: {symbol}\n"
                            message += f"الاستراتيجية: {signal_details['strategy']}\n"
                            message += f"سعر الدخول: {signal_details['entry_price']:.4f}\n"
                            message += f"سعر الهدف: {signal_details['target_price']:.4f}\n"
                            message += f"وقف الخسارة: {signal_details['stop_loss']:.4f}\n"
                            message += f"نسبة المخاطرة: {signal_details['rr_ratio']:.2f}"
                            
                            log_and_notify("info", message, "SIGNAL")
                            send_telegram_message(message)
            
            # انتظار قبل الدورة التالية
            time.sleep(60)  # انتظار دقيقة واحدة
            
        except KeyboardInterrupt:
            logger.info("تم إيقاف البوت يدوياً.")
            break
        except Exception as e:
            logger.error(f"❌ خطأ غير متوقع في الحلقة الرئيسية: {e}", exc_info=True)
            time.sleep(30)

# --- دالة تحديث حالة السوق ---
def update_market_state():
    global current_market_state, last_market_state_check
    
    # تحديث الحالة كل 30 دقيقة
    current_time = time.time()
    if current_time - last_market_state_check < 1800:  # 30 دقيقة
        return
    
    last_market_state_check = current_time
    
    try:
        # جلب بيانات BTC
        btc_df = fetch_historical_data(BTC_SYMBOL, '1h', 10)
        if btc_df is None or len(btc_df) < 50:
            return
        
        # حساب المؤشرات
        btc_df = calculate_all_features(btc_df, None)
        
        # تحديد حالة السوق
        last = btc_df.iloc[-1]
        
        # حالة السوق بناءً على مؤشرات متعددة
        if last['close'] > last['ema_200'] and last['adx'] > 25 and last['plus_di'] > last['minus_di']:
            regime = "BULLISH"
            regime_ar = "صاعد"
        elif last['close'] < last['ema_200'] and last['adx'] > 25 and last['minus_di'] > last['plus_di']:
            regime = "BEARISH"
            regime_ar = "هابط"
        else:
            regime = "SIDEWAYS"
            regime_ar = "جانبي"
        
        # تحديث حالة السوق
        with market_state_lock:
            current_market_state = {
                "overall_regime": regime,
                "overall_regime_ar": regime_ar,
                "trend_details_by_tf": {
                    "1h": {
                        "price_vs_ema200": float(last['price_vs_ema200']),
                        "adx": float(last['adx']),
                        "rsi": float(last['rsi']),
                        "macd": float(last['macd'])
                    }
                },
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
        
        logger.info(f"📊 تم تحديث حالة السوق: {regime_ar}")
        
    except Exception as e:
        logger.error(f"❌ خطأ في تحديث حالة السوق: {e}", exc_info=True)

# --- دالة حفظ الإشارة في قاعدة البيانات ---
def save_signal_to_db(symbol: str, signal_details: Dict[str, Any]):
    if not check_db_connection() or not conn:
        logger.error(f"❌ [قاعدة البيانات] فشل الاتصال عند حفظ إشارة {symbol}")
        return
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, rr_ratio)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                symbol,
                signal_details['entry_price'],
                signal_details['target_price'],
                signal_details['stop_loss'],
                signal_details['strategy'],
                json.dumps(signal_details, cls=NpEncoder),
                signal_details['rr_ratio']
            ))
            conn.commit()
            logger.info(f"✅ [قاعدة البيانات] تم حفظ إشارة {symbol} بنجاح.")
    except Exception as e:
        logger.error(f"❌ [قاعدة البيانات] فشل حفظ إشارة {symbol}: {e}")
        if conn: conn.rollback()

# --- نقطة الدخول الرئيسية ---
if __name__ == "__main__":
    logger.info("🚀 بدء تشغيل بوت التداول الآلي V10.0")
    run_bot()