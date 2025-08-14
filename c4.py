# ملف c4.py - نسخة V10.0 (تحسينات شاملة واستراتيجيات جديدة)
# --- التغييرات الرئيسية (V10.0):
# 1. [تحسين شامل] تم تحديث جميع الاستراتيجيات الحالية (BB+Stoch, MACD+EMA, SR Breakout) بمنطق أكثر دقة وفلاتر إضافية.
# 2. [استراتيجيات جديدة] تمت إضافة ثلاث استراتيجيات جديدة: الارتداد من الدعم (Support Bounce)، اختراق المثلث (Triangle Breakout)، وتقارب الماكد (MACD Convergence).
# 3. [إدارة مخاطر متقدمة] تم تحسين دوال حساب وقف الخسارة وجني الأرباح لتكون ديناميكية وتعتمد على نوع الاستراتيجية وحالة السوق.
# 4. [فلاتر محسنة] تم تطوير فلاتر التقلب وقوة الاتجاه لتكون أكثر ذكاءً وتتكيف مع أنواع العملات المختلفة.
# 5. [تحسينات عامة] تم تعزيز دالة تأكيد الاتجاه على فريم أعلى (HTF) وفلتر الزخم قصير الأجل.
# 6. [لوحة التحكم] تم تحديث لوحة التحكم بالكامل لإضافة أزرار تفعيل/تعطيل لجميع الاستراتيجيات الجديدة والمحسنة.

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

# --- مفاتيح تفعيل الاستراتيجيات ---
USE_BB_STOCH_STRATEGY: bool = True
bb_stoch_strategy_lock = Lock()

USE_MACD_EMA_STRATEGY: bool = True
macd_ema_strategy_lock = Lock()

USE_SR_BREAKOUT_STRATEGY: bool = True
sr_breakout_strategy_lock = Lock()

# --- [جديد] مفاتيح تفعيل الاستراتيجيات الجديدة ---
USE_SUPPORT_BOUNCE_STRATEGY: bool = True
support_bounce_strategy_lock = Lock()

USE_TRIANGLE_BREAKOUT_STRATEGY: bool = True
triangle_breakout_strategy_lock = Lock()

USE_MACD_CONVERGENCE_STRATEGY: bool = True
macd_convergence_strategy_lock = Lock()


BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V9_With_Microstructure'
MODEL_FOLDER: str = 'V9'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '1h'
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
MOMENTUM_PERIOD: int = 10
EMA_SLOPE_PERIOD: int = 5
SUPERTREND_ATR_PERIOD: int = 10
SUPERTREND_MULTIPLIER: float = 3.0

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
    "Invalid Position Size": "حجم الصفقة غير صالح",
    "Lot Size Adjustment Failed": "فشل ضبط حجم العقد",
    "Min Notional Filter": "قيمة الصفقة أقل من الحد الأدنى",
    "Insufficient Balance": "الرصيد غير كافٍ",
    "Insufficient data for TP/SL calculation": "بيانات غير كافية لحساب TP/SL",
    "Insufficient Historical Data": "بيانات تاريخية غير كافية للفحص",
    "HTF Trend Confirmation Failed": "فشل تأكيد الترند على الفريم الأعلى",
    "Short-Term Momentum Filter Failed": "فشل فلتر الزخم قصير الأجل",
    "BB_Stoch Strategy Conditions Not Met": "شروط استراتيجية BB+Stoch (المحسنة) لم تتحقق",
    "MACD_EMA Strategy Conditions Not Met": "شروط استراتيجية MACD+EMA (المحسنة) لم تتحقق",
    "SR Breakout Strategy Conditions Not Met": "شروط استراتيجية اختراق المقاومة (المحسنة) لم تتحقق",
    "Support Bounce Strategy Conditions Not Met": "شروط استراتيجية الارتداد من الدعم لم تتحقق",
    "Triangle Breakout Strategy Conditions Not Met": "شروط استراتيجية اختراق المثلث لم تتحقق",
    "MACD Convergence Strategy Conditions Not Met": "شروط استراتيجية تقارب الماكد لم تتحقق",
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
    exp1 = df_calc['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_calc['close'].ewm(span=26, adjust=False).mean()
    df_calc['macd'] = exp1 - exp2
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()
    df_calc['macd_histogram'] = df_calc['macd'] - df_calc['macd_signal']
    bb_period = 20
    df_calc['bb_middle'] = df_calc['close'].rolling(window=bb_period).mean()
    bb_std = df_calc['close'].rolling(window=bb_period).std()
    df_calc['bb_upper'] = df_calc['bb_middle'] + (bb_std * 2)
    df_calc['bb_lower'] = df_calc['bb_middle'] - (bb_std * 2)
    df_calc['bb_width'] = (df_calc['bb_upper'] - df_calc['bb_lower']) / df_calc['bb_middle'].replace(0, 1e-9)
    df_calc[f'roc_{MOMENTUM_PERIOD}'] = (df_calc['close'] / df_calc['close'].shift(MOMENTUM_PERIOD) - 1) * 100
    
    return df_calc.astype('float32', errors='ignore')

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

# ---------------------- [محسن] فلاتر التأكيد العامة ----------------------
def check_market_volatility_filter(df: pd.DataFrame) -> bool:
    """فلتر لتجنب التداول في فترات التقلب الشديد أو المنخفض جدًا (محسن)"""
    if len(df) < 50:
        return False
    
    last = df.iloc[-1]
    
    if 'atr' not in last or 'close' not in last or last['close'] == 0:
        return False
        
    atr_percent = (last['atr'] / last['close']) * 100
    
    symbol = df.name
    
    high_vol_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'DOTUSDT']
    medium_vol_symbols = ['BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'LINKUSDT', 'MATICUSDT']
    low_vol_symbols = ['USDCUSDT', 'BUSDUSDT', 'TUSDUSDT']
    
    if symbol in high_vol_symbols:
        min_atr, max_atr = 1.0, 8.0
    elif symbol in medium_vol_symbols:
        min_atr, max_atr = 0.7, 5.0
    elif symbol in low_vol_symbols:
        min_atr, max_atr = 0.3, 2.0
    else:
        min_atr, max_atr = 0.5, 5.0

    if atr_percent < min_atr or atr_percent > max_atr:
        log_rejection(df.name, "Market Volatility Filter Failed", {"atr_percent": f"{atr_percent:.2f}"})
        return False
    
    if len(df) >= 2:
        gap_percent = abs(df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
        if gap_percent > 3.0:
            log_rejection(df.name, "Market Volatility Filter Failed", {"gap_percent": f"{gap_percent:.2f}"})
            return False
    
    return True

def check_trend_strength_filter(df: pd.DataFrame) -> bool:
    """فلتر لتأكيد قوة الاتجاه الحالي (محسن)"""
    if len(df) < 50:
        return False
        
    last = df.iloc[-1]
    
    if 'adx' not in last or f'roc_{MOMENTUM_PERIOD}' not in last:
        return False

    symbol = df.name
    
    high_liquidity_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    medium_liquidity_symbols = ['SOLUSDT', 'ADAUSDT', 'XRPUSDT', 'DOTUSDT', 'AVAXUSDT']
    
    if symbol in high_liquidity_symbols:
        min_adx = 20
    elif symbol in medium_liquidity_symbols:
        min_adx = 22
    else:
        min_adx = 25

    if last['adx'] < min_adx:
        log_rejection(df.name, "Trend Strength Filter Failed", {"reason": "ADX too low", "adx": f"{last['adx']:.2f}"})
        return False
        
    min_roc = 0.5
    if abs(last[f'roc_{MOMENTUM_PERIOD}']) < min_roc:
        log_rejection(df.name, "Trend Strength Filter Failed", {"reason": "ROC too low", "roc_10": f"{last[f'roc_{MOMENTUM_PERIOD}']:.2f}"})
        return False
    
    if len(df) >= 5:
        adx_trend = df['adx'].iloc[-5:].pct_change().sum()
        if adx_trend < -0.1:
            log_rejection(df.name, "Trend Strength Filter Failed", {"reason": "ADX trending down", "adx_trend": f"{adx_trend:.2f}"})
            return False
    
    return True

# ---------------------- [محسن] دوال منطق الاستراتيجيات الحالية ----------------------
def check_bb_stoch_strategy_revised(df: pd.DataFrame) -> bool:
    """استراتيجية BB+Stoch المعدلة للدخول المبكر مع تحسينات إضافية"""
    if len(df) < 21:
        return False

    last, prev = df.iloc[-1], df.iloc[-2]
    
    prev_candle_touch_bb = prev['low'] <= prev['bb_lower']
    current_candle_is_bullish = last['close'] > last['open']
    stoch_turning_up = (last['stoch_rsi_k'] < 40 and last['stoch_rsi_k'] > prev['stoch_rsi_k'])
    rsi_not_extreme = last['rsi'] > 20
    volume_increasing = last['volume'] > prev['volume'] * 1.1
    stoch_cross_up = (prev['stoch_rsi_k'] < prev['stoch_rsi_d'] and last['stoch_rsi_k'] > last['stoch_rsi_d'])
    macd_bullish = (last['macd'] > last['macd_signal'] or (last['macd_histogram'] > prev['macd_histogram'] and last['macd_histogram'] > 0))
    
    with market_state_lock:
        trend_ok = "DOWNTREND" not in current_market_state.get("overall_regime", "UNCERTAIN")
    
    price_above_ema100 = last['close'] > last.get('ema_100', last['close'])
    
    conditions = {
        "prev_candle_touch_bb": prev_candle_touch_bb,
        "current_candle_is_bullish": current_candle_is_bullish,
        "stoch_turning_up": stoch_turning_up,
        "stoch_cross_up": stoch_cross_up,
        "rsi_not_extreme": rsi_not_extreme,
        "volume_increasing": volume_increasing,
        "macd_bullish": macd_bullish,
        "price_above_ema100": price_above_ema100,
        "trend_ok": trend_ok
    }

    if all(conditions.values()):
        logger.info(f"  -> [{df.name}] ✅ إشارة BB+Stoch (المحسنة).")
        return True

    failed_conditions = {k: v for k, v in conditions.items() if not v}
    if prev_candle_touch_bb and current_candle_is_bullish and len(failed_conditions) <= 3:
         log_rejection(df.name, "BB_Stoch Strategy Conditions Not Met", {"failed": list(failed_conditions.keys())})

    return False

def check_macd_ema_strategy(df: pd.DataFrame) -> bool:
    """استراتيجية MACD+EMA مع تحسينات إضافية للدقة"""
    if len(df) < 3: return False
    last, prev, prev_prev = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    
    macd_cross_up = prev['macd'] < prev['macd_signal'] and last['macd'] > last['macd_signal']
    price_above_ema = last['close'] > last['ema_21']
    macd_strength = last['macd_histogram'] > 0 and last['macd_histogram'] > prev['macd_histogram']
    trend_strength = last['adx'] > 25
    macd_position = prev_prev['macd'] < 0
    
    no_bearish_divergence = True
    if len(df) >= 10:
        recent_prices = df['close'].iloc[-10:]
        recent_macd = df['macd'].iloc[-10:]
        price_high_idx = recent_prices.idxmax()
        macd_high_idx = recent_macd.idxmax()
        no_bearish_divergence = price_high_idx >= macd_high_idx
    
    volume_confirmation = last['volume'] > df['volume'].rolling(5).mean().iloc[-1] * 1.2
    rsi_confirmation = 40 < last['rsi'] < 70
    
    conditions = {
        "macd_cross_up": macd_cross_up,
        "price_above_ema": price_above_ema,
        "macd_strength": macd_strength,
        "trend_strength": trend_strength,
        "macd_position": macd_position,
        "no_bearish_divergence": no_bearish_divergence,
        "volume_confirmation": volume_confirmation,
        "rsi_confirmation": rsi_confirmation
    }
    
    if all(conditions.values()):
        logger.info(f"  -> [{df.name}] ✅ إشارة MACD+EMA (المحسنة).")
        return True
    
    if macd_cross_up:
        failed_conditions = {k: v for k, v in conditions.items() if not v}
        log_rejection(df.name, "MACD_EMA Strategy Conditions Not Met", {"failed": list(failed_conditions.keys())})
        
    return False

def check_support_resistance_strategy_enhanced(df: pd.DataFrame) -> bool:
    """استراتيجية اختراق الدعم والمقاومة المحسنة بدقة أكبر"""
    if len(df) < 50:
        return False
    
    last = df.iloc[-1]
    
    resistance_candidates = []
    for i in range(2, len(df)-2):
        if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
            df['high'].iloc[i] > df['high'].iloc[i+1] and
            df['high'].iloc[i] > df['high'].iloc[i-2] and
            df['high'].iloc[i] > df['high'].iloc[i+2]):
            resistance_candidates.append(df['high'].iloc[i])
    
    if not resistance_candidates:
        return False
        
    resistance_candidates.sort()
    resistance_zones = []
    current_zone = [resistance_candidates[0]]
    
    for price in resistance_candidates[1:]:
        if price - current_zone[-1] < last['close'] * 0.01:
            current_zone.append(price)
        else:
            resistance_zones.append(sum(current_zone) / len(current_zone))
            current_zone = [price]
    
    if current_zone:
        resistance_zones.append(sum(current_zone) / len(current_zone))
    
    current_price = last['close']
    next_resistance = None
    for resistance in sorted(resistance_zones):
        if resistance > current_price:
            next_resistance = resistance
            break
    
    if next_resistance is None:
        return False
    
    tolerance = next_resistance * 0.01
    resistance_tests = df[(df['high'] >= next_resistance - tolerance) & (df['high'] <= next_resistance + tolerance)]
    resistance_strength = len(resistance_tests)
    
    if resistance_strength < 2:
        return False
    
    breakout = last['close'] > next_resistance and df['close'].iloc[-2] <= next_resistance
    volume_confirmation = last['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.5
    trend_confirmation = (last['close'] > last['ema_21'] and last['ema_21'] > last['ema_50'])
    not_false_breakout = (last['close'] - next_resistance) / next_resistance > 0.005
    momentum_confirmation = (last['macd'] > last['macd_signal'] and last['rsi'] > 50 and last['rsi'] < 70)
    adx_confirmation = last['adx'] > 25
    
    conditions = {
        "breakout": breakout,
        "volume_confirmation": volume_confirmation,
        "trend_confirmation": trend_confirmation,
        "not_false_breakout": not_false_breakout,
        "momentum_confirmation": momentum_confirmation,
        "adx_confirmation": adx_confirmation,
        "resistance_strength": resistance_strength >= 2
    }
    
    if all(conditions.values()):
        logger.info(f"  -> [{df.name}] ✅ إشارة اختراق مقاومة (المحسنة) - قوة المستوى: {resistance_strength}")
        return True
        
    if breakout:
        failed_conditions = {k: v for k, v in conditions.items() if not v}
        log_rejection(df.name, "SR Breakout Strategy Conditions Not Met", {"failed": list(failed_conditions.keys())})
    
    return False

# ---------------------- [جديد] دوال منطق الاستراتيجيات الجديدة ----------------------
def check_support_bounce_strategy(df: pd.DataFrame) -> bool:
    """استراتيجية الهبوط على خط الدعم مع تأكيدات متعددة"""
    if len(df) < 50:
        return False
        
    last, prev = df.iloc[-1], df.iloc[-2]
    
    support_candidates = []
    for i in range(2, len(df)-2):
        if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
            df['low'].iloc[i] < df['low'].iloc[i+1] and
            df['low'].iloc[i] < df['low'].iloc[i-2] and
            df['low'].iloc[i] < df['low'].iloc[i+2]):
            support_candidates.append(df['low'].iloc[i])
    
    if not support_candidates:
        return False
        
    support_candidates.sort(reverse=True)
    support_zones = []
    current_zone = [support_candidates[0]]
    
    for price in support_candidates[1:]:
        if current_zone[-1] - price < last['close'] * 0.01:
            current_zone.append(price)
        else:
            support_zones.append(sum(current_zone) / len(current_zone))
            current_zone = [price]
    
    if current_zone:
        support_zones.append(sum(current_zone) / len(current_zone))
    
    current_price = last['close']
    next_support = None
    for support in sorted(support_zones, reverse=True):
        if support < current_price:
            next_support = support
            break
    
    if next_support is None:
        return False
    
    tolerance = next_support * 0.01
    support_tests = df[(df['low'] >= next_support - tolerance) & (df['low'] <= next_support + tolerance)]
    support_strength = len(support_tests)
    
    if support_strength < 2:
        return False
    
    support_touch = prev['low'] <= next_support
    bounce = last['close'] > last['open'] and last['close'] > next_support
    rsi_oversold = prev['rsi'] < 40 and last['rsi'] > prev['rsi']
    volume_increasing = last['volume'] > prev['volume'] * 1.1
    momentum_turning = (last['macd_histogram'] > prev['macd_histogram'] and last['stoch_rsi_k'] > prev['stoch_rsi_k'])
    
    with market_state_lock:
        trend_ok = "DOWNTREND" not in current_market_state.get("overall_regime", "UNCERTAIN")
    
    conditions = {
        "support_touch": support_touch,
        "bounce": bounce,
        "rsi_oversold": rsi_oversold,
        "volume_increasing": volume_increasing,
        "momentum_turning": momentum_turning,
        "trend_ok": trend_ok,
        "support_strength": support_strength >= 2
    }
    
    if all(conditions.values()):
        logger.info(f"  -> [{df.name}] ✅ إشارة ارتداد عن دعم (المحسنة) - قوة المستوى: {support_strength}")
        return True
    
    if support_touch and bounce:
        failed_conditions = {k: v for k, v in conditions.items() if not v}
        log_rejection(df.name, "Support Bounce Strategy Conditions Not Met", {"failed": list(failed_conditions.keys())})
        
    return False

def check_triangle_breakout_strategy(df: pd.DataFrame) -> bool:
    """استراتيجية الاختراق من مثلث صاعد"""
    if len(df) < 30:
        return False
        
    highs, lows = [], []
    
    for i in range(2, len(df)-2):
        if (df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i+1] and
            df['high'].iloc[i] > df['high'].iloc[i-2] and df['high'].iloc[i] > df['high'].iloc[i+2]):
            highs.append((i, df['high'].iloc[i]))
    
    for i in range(2, len(df)-2):
        if (df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i+1] and
            df['low'].iloc[i] < df['low'].iloc[i-2] and df['low'].iloc[i] < df['low'].iloc[i+2]):
            lows.append((i, df['low'].iloc[i]))
    
    if len(highs) < 2 or len(lows) < 2:
        return False
    
    resistance_highs = sorted(highs, key=lambda x: x[0], reverse=True)[:2]
    if len(resistance_highs) < 2: return False
        
    resistance_slope = (resistance_highs[0][1] - resistance_highs[1][1]) / (resistance_highs[0][0] - resistance_highs[1][0])
    resistance_intercept = resistance_highs[0][1] - resistance_slope * resistance_highs[0][0]
    
    support_lows = sorted(lows, key=lambda x: x[0], reverse=True)[:2]
    if len(support_lows) < 2: return False
        
    support_slope = (support_lows[0][1] - support_lows[1][1]) / (support_lows[0][0] - support_lows[1][0])
    support_intercept = support_lows[0][1] - support_slope * support_lows[0][0]
    
    if resistance_slope >= 0 or support_slope <= 0:
        return False
    
    last_idx, last = len(df) - 1, df.iloc[-1]
    resistance_at_last = resistance_slope * last_idx + resistance_intercept
    
    breakout = last['close'] > resistance_at_last and df['close'].iloc[-2] <= resistance_at_last
    volume_confirmation = last['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.5
    momentum_confirmation = (last['macd'] > last['macd_signal'] and last['rsi'] > 50 and last['rsi'] < 70)
    adx_confirmation = last['adx'] > 20
    
    conditions = {
        "breakout": breakout,
        "volume_confirmation": volume_confirmation,
        "momentum_confirmation": momentum_confirmation,
        "adx_confirmation": adx_confirmation
    }
    
    if all(conditions.values()):
        logger.info(f"  -> [{df.name}] ✅ إشارة اختراق مثلث صاعد (المحسنة)")
        return True
    
    if breakout:
        failed_conditions = {k: v for k, v in conditions.items() if not v}
        log_rejection(df.name, "Triangle Breakout Strategy Conditions Not Met", {"failed": list(failed_conditions.keys())})
        
    return False

def check_macd_convergence_strategy(df: pd.DataFrame) -> bool:
    """استراتيجية التقارب بين MACD وخط الإشارة"""
    if len(df) < 10:
        return False
        
    last = df.iloc[-1]
    
    macd_diff = df['macd'] - df['macd_signal']
    
    diff_decreasing = all(macd_diff.iloc[-i] < macd_diff.iloc[-i-1] for i in range(1, 5))
    macd_close_to_signal = abs(macd_diff.iloc[-1]) < abs(macd_diff.iloc[-5]) * 0.3
    price_above_ema = last['close'] > last['ema_21']
    volume_increasing = last['volume'] > df['volume'].rolling(5).mean().iloc[-1] * 1.1
    momentum_positive = last['macd_histogram'] > 0
    rsi_confirmation = 40 < last['rsi'] < 70
    
    conditions = {
        "diff_decreasing": diff_decreasing,
        "macd_close_to_signal": macd_close_to_signal,
        "price_above_ema": price_above_ema,
        "volume_increasing": volume_increasing,
        "momentum_positive": momentum_positive,
        "rsi_confirmation": rsi_confirmation
    }
    
    if all(conditions.values()):
        logger.info(f"  -> [{df.name}] ✅ إشارة تقارب MACD (المحسنة)")
        return True
    
    if diff_decreasing and macd_close_to_signal:
        failed_conditions = {k: v for k, v in conditions.items() if not v}
        log_rejection(df.name, "MACD Convergence Strategy Conditions Not Met", {"failed": list(failed_conditions.keys())})
        
    return False

# --- [محسن] دالة تأكيد الترند على فريم أعلى ---
def is_htf_bullish_confirmation(symbol: str, htf: str = '1h', lookback: int = 200) -> bool:
    """دالة تأكيد الترند على فريم أعلى (محسنة)"""
    try:
        df = fetch_historical_data(symbol, htf, days=40) 
        if df is None or len(df) < lookback:
            logger.warning(f"  -> [HTF {htf}] {symbol} بيانات غير كافية للتأكيد ({len(df) if df is not None else 0} شمعة).")
            return False

        df = calculate_all_features(df, None) # Use the main feature calculation function
        
        last, prev = df.iloc[-1], df.iloc[-2]

        strong_uptrend = (last['ema_50'] > last['ema_200'] and 
                          last['adx'] > 25 and 
                          last['close'] > last['ema_50'] and
                          last['plus_di'] > last['minus_di'])
        
        macd_cross_up = prev['macd'] < prev['macd_signal'] and last['macd'] > last['macd_signal']
        ema_cross_up  = prev['ema_50'] < prev['ema_200'] and last['ema_50'] > last['ema_200']
        recent_bullish_flip = macd_cross_up and ema_cross_up
        
        momentum_positive = last['macd'] > last['macd_signal'] and last['macd_histogram'] > 0
        volume_positive = last['volume'] > df['volume'].rolling(20).mean().iloc[-1]
        rsi_positive = 40 < last['rsi'] < 70
        
        is_confirmed = (strong_uptrend or recent_bullish_flip) and momentum_positive and volume_positive and rsi_positive
        
        logger.info(f"  -> [HTF {htf}] {symbol} تأكيد الترند: {is_confirmed} (قوي: {strong_uptrend} | تحول: {recent_bullish_flip})")
        return is_confirmed

    except Exception as e:
        logger.error(f"❌ [HTF Confirm] خطأ في {symbol}: {e}")
        return False

# --- [محسن] دالة فلتر الزخم قصير الأجل ---
def passes_short_term_momentum_filter(symbol: str, df: pd.DataFrame) -> bool:
    """دالة فلتر الزخم قصير الأجل (محسنة)"""
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
    
    price_momentum = last['close'] > last['ema_9'] and last['close'] > df['close'].iloc[-4]
    trend_strength = last['adx'] > 20
    price_trend = (df['close'].iloc[-1] > df['close'].iloc[-3] and df['close'].iloc[-3] > df['close'].iloc[-5])
    volume_trend = (df['volume'].iloc[-1] > df['volume'].iloc[-3] and df['volume'].iloc[-3] > df['volume'].iloc[-5])
    roc_momentum = last[f'roc_{MOMENTUM_PERIOD}'] > 0
    
    is_valid = (
        (is_squeeze or close_to_bands) and
        volume_spike and
        (macd_momentum or rsi_momentum) and
        price_momentum and
        trend_strength and
        price_trend and
        volume_trend and
        roc_momentum
    )
    
    if not is_valid:
        log_rejection(symbol, "Short-Term Momentum Filter Failed")

    logger.info(f"  -> [فلتر الزخم المحسن] {symbol}: Valid={is_valid}")
    return is_valid

# --- [محسن] دوال حساب الأهداف ووقف الخسارة ---
def calculate_optimal_stop_loss(df: pd.DataFrame, strategy_type: str) -> float:
    """حساب وقف خسارة مثالي بناءً على الاستراتيجية وظروف السوق"""
    last = df.iloc[-1]
    stop_loss = 0.0

    if strategy_type == "BB_Stoch":
        bb_lower = last['bb_lower']
        recent_low = df['low'].iloc[-5:].min()
        stop_loss = min(bb_lower, recent_low) * 0.99
        
    elif strategy_type == "MACD_EMA":
        ema_21 = last['ema_21']
        recent_low = df['low'].iloc[-5:].min()
        stop_loss = min(ema_21, recent_low) * 0.99
        
    elif strategy_type == "Support_Bounce":
        support_candidates = []
        for i in range(2, len(df)-2):
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i+1] and
                df['low'].iloc[i] < df['low'].iloc[i-2] and df['low'].iloc[i] < df['low'].iloc[i+2]):
                support_candidates.append(df['low'].iloc[i])
        
        if support_candidates:
            support_candidates.sort(reverse=True)
            support_zones = []
            current_zone = [support_candidates[0]]
            for price in support_candidates[1:]:
                if current_zone[-1] - price < last['close'] * 0.01:
                    current_zone.append(price)
                else:
                    support_zones.append(sum(current_zone) / len(current_zone))
                    current_zone = [price]
            if current_zone: support_zones.append(sum(current_zone) / len(current_zone))
            
            next_support = next((s for s in sorted(support_zones, reverse=True) if s < last['close']), None)
            if next_support: stop_loss = next_support * 0.99
            
    elif strategy_type == "Triangle_Breakout":
        lows = []
        for i in range(2, len(df)-2):
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i+1] and
                df['low'].iloc[i] < df['low'].iloc[i-2] and df['low'].iloc[i] < df['low'].iloc[i+2]):
                lows.append((i, df['low'].iloc[i]))
        
        if len(lows) >= 2:
            support_lows = sorted(lows, key=lambda x: x[0], reverse=True)[:2]
            support_slope = (support_lows[0][1] - support_lows[1][1]) / (support_lows[0][0] - support_lows[1][0])
            support_intercept = support_lows[0][1] - support_slope * support_lows[0][0]
            support_at_last = support_slope * (len(df) - 1) + support_intercept
            stop_loss = support_at_last * 0.99

    if stop_loss == 0.0: # Default fallback
        stop_loss = last['close'] - (last['atr'] * 1.5)
    
    return stop_loss

def calculate_optimal_take_profit(df: pd.DataFrame, strategy_type: str, entry_price: float, stop_loss: float) -> float:
    """حساب جني أرباح مثالي بناءً على الاستراتيجية ونسبة المخاطرة إلى العائد"""
    last = df.iloc[-1]
    
    risk_amount = entry_price - stop_loss
    if risk_amount <= 0: return entry_price * 1.02 # Default 2% profit if risk is invalid

    with market_state_lock:
        market_regime = current_market_state.get("overall_regime", "UNCERTAIN")
    
    if market_regime == "STRONG_UPTREND": rr_ratio = 3.0
    elif market_regime == "UPTREND": rr_ratio = 2.5
    elif market_regime == "RANGING": rr_ratio = 1.5
    else: rr_ratio = 2.0
    
    take_profit = entry_price + (risk_amount * rr_ratio)
    
    resistance_candidates = []
    for i in range(2, len(df)-2):
        if (df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i+1] and
            df['high'].iloc[i] > df['high'].iloc[i-2] and df['high'].iloc[i] > df['high'].iloc[i+2]):
            resistance_candidates.append(df['high'].iloc[i])
    
    if resistance_candidates:
        resistance_candidates.sort()
        resistance_zones = []
        current_zone = [resistance_candidates[0]]
        for price in resistance_candidates[1:]:
            if price - current_zone[-1] < last['close'] * 0.01:
                current_zone.append(price)
            else:
                resistance_zones.append(sum(current_zone) / len(current_zone))
                current_zone = [price]
        if current_zone: resistance_zones.append(sum(current_zone) / len(current_zone))
        
        next_resistance = next((r for r in sorted(resistance_zones) if r > entry_price), None)
        
        if next_resistance and next_resistance < take_profit:
            logger.info(f"  -> [{df.name}] تعديل الهدف ليتناسب مع المقاومة القريبة عند {next_resistance:.4f}")
            take_profit = next_resistance * 0.99
    
    return take_profit

# ---------------------- دوال إدارة الصفقات ----------------------
def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> Optional[Decimal]:
    try:
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info: return None
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = Decimal(lot_size_filter['stepSize'])
            return (Decimal(str(quantity)) // step_size) * step_size
        return Decimal(str(quantity))
    except Exception as e:
        logger.error(f"[{symbol}] خطأ في تعديل الكمية لـ LOT_SIZE: {e}", exc_info=True)
        return None

def calculate_position_size(symbol: str, entry_price: float, stop_loss_price: float) -> Optional[Decimal]:
    if not client: return None
    try:
        with risk_per_trade_lock: current_risk_percent = RISK_PER_TRADE_PERCENT
        balance_response = client.get_asset_balance(asset='USDT')
        available_balance = Decimal(balance_response['free'])
        risk_amount_usdt = available_balance * (Decimal(str(current_risk_percent)) / Decimal('100'))
        risk_per_coin = Decimal(str(entry_price)) - Decimal(str(stop_loss_price))
        if risk_per_coin <= 0: log_rejection(symbol, "Invalid Position Size"); return None
        initial_quantity = risk_amount_usdt / risk_per_coin
        adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(initial_quantity))
        if adjusted_quantity is None or adjusted_quantity <= 0: log_rejection(symbol, "Lot Size Adjustment Failed"); return None
        notional_value = adjusted_quantity * Decimal(str(entry_price))
        symbol_info = exchange_info_map.get(symbol)
        if symbol_info:
            for f in symbol_info['filters']:
                if f['filterType'] in ('MIN_NOTIONAL', 'NOTIONAL'):
                    min_notional = Decimal(f.get('minNotional', f.get('notional', '0')))
                    if notional_value < min_notional: log_rejection(symbol, "Min Notional Filter", {"value": f"{notional_value:.2f}"}); return None
        if notional_value > available_balance: log_rejection(symbol, "Insufficient Balance", {"required": f"{notional_value:.2f}"}); return None
        return adjusted_quantity
    except Exception as e:
        logger.error(f"❌ [{symbol}] خطأ في حساب حجم الصفقة: {e}", exc_info=True); return None

def place_order(symbol: str, side: str, quantity: Decimal, order_type: str = Client.ORDER_TYPE_MARKET) -> Optional[Dict]:
    if not client: return None
    logger.info(f"➡️ [{symbol}] محاولة تنفيذ أمر {side} حقيقي لكمية {quantity}.")
    try:
        order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=str(quantity))
        log_and_notify('info', f"صفقة حقيقية: تم وضع أمر {side} لـ {quantity} {symbol}.", "REAL_TRADE")
        return order
    except Exception as e:
        logger.error(f"❌ [{symbol}] خطأ من المنصة عند تنفيذ الأمر: {e}")
        log_and_notify('error', f"فشل صفقة حقيقية: {symbol} | {e}", "REAL_TRADE_ERROR")
        return None

def close_signal(signal_id: int, closing_price: float, reason: str) -> bool:
    with signal_cache_lock:
        signal_to_close, symbol_to_close = None, None
        for symbol, signal_data in open_signals_cache.items():
            if signal_data['id'] == signal_id:
                signal_to_close, symbol_to_close = signal_data, symbol
                break
        if not signal_to_close:
            logger.warning(f"⚠️ [إغلاق] محاولة إغلاق صفقة غير موجودة في الكاش (ID: {signal_id}). ربما أغلقت بالفعل.")
            return False

        entry_price = float(signal_to_close['entry_price'])
        profit_percentage = ((closing_price - entry_price) / entry_price) * 100

        if signal_to_close.get('is_real_trade'):
            try:
                base_asset = symbol_to_close.replace('USDT', '')
                balance_response = client.get_asset_balance(asset=base_asset)
                actual_free_balance = Decimal(balance_response['free'])
                
                if actual_free_balance > 0:
                    quantity_to_sell = adjust_quantity_to_lot_size(symbol_to_close, float(actual_free_balance))
                    if quantity_to_sell and quantity_to_sell > 0:
                        sell_order = place_order(symbol_to_close, Client.SIDE_SELL, quantity_to_sell)
                        if not sell_order:
                            logger.warning(f"⚠️ [{symbol_to_close}] فشل أمر البيع عند الإغلاق. سيتم إكمال عملية الإغلاق في قاعدة البيانات على أي حال.")
            except Exception as e:
                logger.error(f"❌ [{symbol_to_close}] خطأ أثناء محاولة بيع الرصيد عند الإغلاق: {e}", exc_info=True)
        
        if not check_db_connection() or not conn: return False

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE signals SET status = 'closed', closing_price = %s, closed_at = NOW(),
                    profit_percentage = %s, closing_reason = %s WHERE id = %s;
                """, (closing_price, profit_percentage, reason, signal_id))
            conn.commit()

            if symbol_to_close in open_signals_cache:
                del open_signals_cache[symbol_to_close]

            log_and_notify('info', f"تم الإغلاق: {symbol_to_close} عند {closing_price:.4f}. السبب: {reason}. الربح/الخسارة: {profit_percentage:.2f}%", "TRADE_CLOSED")
            return True
        except Exception as e:
            logger.error(f"❌ [قاعدة البيانات] فشل تحديث الصفقة المغلقة: {e}"); conn.rollback(); return False

def insert_signal_into_db(signal_data: Dict) -> Optional[Dict]:
    if not check_db_connection() or not conn: return None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, is_real_trade, quantity, original_quantity, order_id, current_peak_price, rr_ratio)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *;
            """, (
                signal_data['symbol'], float(signal_data['entry_price']), float(signal_data['target_price']), 
                float(signal_data['stop_loss']), signal_data['strategy_name'], 
                json.dumps(signal_data['signal_details'], cls=NpEncoder), 
                signal_data.get('is_real_trade', False),
                float(signal_data.get('quantity', 0)), float(signal_data.get('quantity', 0)), 
                signal_data.get('order_id'), float(signal_data['entry_price']),
                float(signal_data.get('rr_ratio', 0.0))
            ))
            saved_signal = cur.fetchone()
            conn.commit()
            logger.info(f"💾 [{signal_data['symbol']}] تم حفظ الإشارة الجديدة في قاعدة البيانات.")
            return dict(saved_signal)
    except Exception as e:
        logger.error(f"❌ [قاعدة البيانات] فشل إدراج الإشارة: {e}", exc_info=True); conn.rollback(); return None


# ---------------------- دوال النظام الأساسية ----------------------
def determine_market_state_enhanced():
    global current_market_state, last_market_state_check
    if time.time() - last_market_state_check < 180: return
    logger.info("🧠 [حالة السوق] جاري تحديث حالة السوق العامة...")
    try:
        trend_details = {}
        for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
            df = fetch_historical_data(BTC_SYMBOL, tf, 20)
            if df is not None and not df.empty:
                ema_fast = df['close'].ewm(span=12, adjust=False).mean().iloc[-1]
                ema_slow = df['close'].ewm(span=26, adjust=False).mean().iloc[-1]
                adx_features = calculate_all_features(df, None)
                adx = adx_features['adx'].iloc[-1] if not adx_features.empty else 0
                if ema_fast > ema_slow and adx > 25: trend = "Strong Uptrend"
                elif ema_fast > ema_slow: trend = "Uptrend"
                elif ema_fast < ema_slow and adx > 25: trend = "Strong Downtrend"
                elif ema_fast < ema_slow: trend = "Downtrend"
                else: trend = "Ranging"
                trend_details[tf] = {"trend": trend, "adx": float(adx)}
            else: trend_details[tf] = {"trend": "Uncertain", "adx": 0}
        trends = [d['trend'] for d in trend_details.values()]
        overall_regime = max(set(trends), key=trends.count) if trends else "Uncertain"
        with market_state_lock:
            current_market_state = {"overall_regime": overall_regime.upper().replace(" ", "_"), "trend_details_by_tf": trend_details, "last_updated": datetime.now(timezone.utc).isoformat()}
            last_market_state_check = time.time()
        logger.info(f"✅ [حالة السوق] الحالة العامة المحددة: {overall_regime}")
    except Exception as e:
        logger.error(f"❌ [حالة السوق] خطأ في التحديث: {e}", exc_info=True)

# ---------------------- واجهة الويب (Flask) ----------------------
app = Flask(__name__)
CORS(app)

def get_dashboard_html():
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم التداول V10 - استراتيجيات محسنة</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
    <style>
        :root { --bg-main: #0D1117; --bg-card: #161B22; --border-color: #30363D; --text-primary: #E6EDF3; --text-secondary: #848D97; --accent-blue: #58A6FF; --accent-green: #3FB950; --accent-red: #F85149; --accent-yellow: #D29922; --accent-purple: #A371F7;}
        body { font-family: 'Tajawal', sans-serif; background-color: var(--bg-main); color: var(--text-primary); }
        .card { background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 0.5rem; }
        .tab-btn.active { border-bottom-color: var(--accent-blue); }
        input:checked + .toggle-bg { background-color: var(--accent-green); }
        .input-field { background-color: #0D1117; border: 1px solid var(--border-color); border-radius: 0.375rem; padding: 0.5rem 0.75rem; color: var(--text-primary); }
        .save-btn { background-color: var(--accent-blue); color: white; padding: 0.5rem 1rem; border-radius: 0.375rem; font-weight: bold; transition: background-color 0.2s; }
        .save-btn:hover { background-color: #4a91e2; }
        .strategy-toggle { border-left: 4px solid var(--accent-blue); }
        .strategy-toggle-new { border-left: 4px solid var(--accent-yellow); }
    </style>
</head>
<body class="p-4 md:p-6">
    <div class="container mx-auto max-w-screen-2xl">
        <header class="mb-6 flex flex-wrap justify-between items-center gap-4">
            <h1 class="text-2xl md:text-3xl font-extrabold"><span class="text-accent-blue">لوحة تحكم</span><span class="text-text-secondary font-medium"> V10.0 (Enhanced)</span></h1>
        </header>
        <section class="mb-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">حالة السوق</h3><div id="overall-regime" class="text-2xl font-bold text-center">...</div></div>
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">الصفقات المفتوحة</h3><div id="open-trades-count" class="text-2xl font-bold text-center">...</div></div>
            <div class="card p-4 flex flex-col justify-center items-center"><h3 class="font-bold text-lg text-text-secondary mb-2">التداول الحقيقي</h3><div class="flex items-center space-x-3 space-x-reverse"><span id="trading-status-text" class="font-bold text-lg"></span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="trading-toggle" class="sr-only" onchange="toggleTrading()"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div><div class="mt-2 text-xs text-text-secondary">رصيد USDT: <span id="usdt-balance" class="font-mono">...</span></div></div>
        </section>
        <div class="mb-4 border-b border-border-color"><nav class="flex space-x-6 space-x-reverse -mb-px">
            <button onclick="showTab('signals', this)" class="tab-btn active text-white py-3 px-1 font-semibold">الصفقات</button>
            <button onclick="showTab('settings', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الإعدادات</button>
            <button onclick="showTab('notifications', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الإشعارات</button>
            <button onclick="showTab('rejections', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الصفقات المرفوضة</button>
        </nav></div>
        <main>
            <div id="signals-tab" class="tab-content"><div class="overflow-x-auto card p-0"><table class="min-w-full text-sm text-right"><thead class="border-b border-border-color bg-black/20"><tr><th class="p-4 font-semibold">العملة</th><th class="p-4 font-semibold">الربح/الخسارة</th><th class="p-4 font-semibold">الدخول/الحالي/الهدف</th><th class="p-4 font-semibold">إجراء</th></tr></thead><tbody id="signals-table"></tbody></table></div></div>
            <div id="settings-tab" class="tab-content hidden">
                <div class="card p-6">
                    <h4 class="text-xl font-bold mb-6 text-text-primary">الاستراتيجيات المفعّلة</h4>
                    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
                        <div class="flex items-center justify-between p-4 bg-black/20 rounded-lg strategy-toggle">
                            <span class="font-semibold">BB+Stoch (محسنة)</span>
                            <label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="bb-stoch-strategy-toggle" class="sr-only strategy-input"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label>
                        </div>
                        <div class="flex items-center justify-between p-4 bg-black/20 rounded-lg strategy-toggle">
                            <span class="font-semibold">MACD+EMA (محسنة)</span>
                            <label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="macd-ema-strategy-toggle" class="sr-only strategy-input"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label>
                        </div>
                        <div class="flex items-center justify-between p-4 bg-black/20 rounded-lg strategy-toggle">
                            <span class="font-semibold">S/R Breakout (محسنة)</span>
                            <label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="sr-breakout-strategy-toggle" class="sr-only strategy-input"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label>
                        </div>
                        <div class="flex items-center justify-between p-4 bg-black/20 rounded-lg strategy-toggle-new">
                            <span class="font-semibold">Support Bounce (جديدة)</span>
                            <label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="support-bounce-strategy-toggle" class="sr-only strategy-input"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label>
                        </div>
                        <div class="flex items-center justify-between p-4 bg-black/20 rounded-lg strategy-toggle-new">
                            <span class="font-semibold">Triangle Breakout (جديدة)</span>
                            <label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="triangle-breakout-strategy-toggle" class="sr-only strategy-input"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label>
                        </div>
                        <div class="flex items-center justify-between p-4 bg-black/20 rounded-lg strategy-toggle-new">
                            <span class="font-semibold">MACD Convergence (جديدة)</span>
                            <label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="macd-convergence-strategy-toggle" class="sr-only strategy-input"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label>
                        </div>
                    </div>
                    <div class="mt-8 text-left">
                        <button onclick="saveSettings()" class="save-btn">حفظ الإعدادات</button>
                    </div>
                    <div id="settings-feedback" class="mt-4 text-center"></div>
                </div>
            </div>
            <div id="notifications-tab" class="tab-content hidden"><div id="notifications-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-2"></div></div>
            <div id="rejections-tab" class="tab-content hidden"><div id="rejections-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-2"></div></div>
        </main>
    </div>
<script>
function showTab(tabId, el) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.add('hidden'));
    document.getElementById(tabId + '-tab').classList.remove('hidden');
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active', 'text-white'));
    el.classList.add('active', 'text-white');
}
async function fetchData(url) { try { const r = await fetch(url); return r.ok ? await r.json() : null; } catch (e) { console.error('Fetch Error:', e); return null; } }

function updateMarketStatus() {
    fetchData('/api/market_status').then(data => {
        if (!data) return;
        document.getElementById('overall-regime').textContent = (data.market_state?.overall_regime || 'UNCERTAIN').replace(/_/g, ' ');
        document.getElementById('open-trades-count').textContent = `${data.open_trades_count} / ${data.max_open_trades}`;
        const tradeToggle = document.getElementById('trading-toggle'), tradeText = document.getElementById('trading-status-text');
        tradeToggle.checked = data.is_trading_enabled;
        tradeText.textContent = data.is_trading_enabled ? 'مُفعَّل' : 'غير مُفعَّل';
        tradeText.className = `font-bold text-lg ${data.is_trading_enabled ? 'text-accent-green' : 'text-accent-red'}`;
        document.getElementById('usdt-balance').textContent = data.usdt_balance ? parseFloat(data.usdt_balance).toFixed(2) : 'N/A';

        if(data.settings) {
            document.getElementById('bb-stoch-strategy-toggle').checked = data.settings.use_bb_stoch_strategy;
            document.getElementById('macd-ema-strategy-toggle').checked = data.settings.use_macd_ema_strategy;
            document.getElementById('sr-breakout-strategy-toggle').checked = data.settings.use_sr_breakout_strategy;
            document.getElementById('support-bounce-strategy-toggle').checked = data.settings.use_support_bounce_strategy;
            document.getElementById('triangle-breakout-strategy-toggle').checked = data.settings.use_triangle_breakout_strategy;
            document.getElementById('macd-convergence-strategy-toggle').checked = data.settings.use_macd_convergence_strategy;
        }
    });
}
function updateSignals() {
    fetchData('/api/signals').then(data => {
        if (!data || !Array.isArray(data)) return;
        const tableBody = document.getElementById('signals-table');
        tableBody.innerHTML = '';
        data.filter(s => ['open', 'updated'].includes(s.status)).forEach(s => {
            const profit = parseFloat(s.profit_percentage || 0);
            const pClass = profit > 0 ? 'text-accent-green' : profit < 0 ? 'text-accent-red' : 'text-text-secondary';
            const entry = parseFloat(s.entry_price);
            const current = parseFloat(s.current_price || entry);
            const target = parseFloat(s.target_price);
            tableBody.innerHTML += `
            <tr class="border-b border-border-color hover:bg-white/5">
                <td class="p-4 font-bold">${s.symbol}<br><span class="text-xs text-text-secondary">${s.strategy_name.replace(/_/g, ' ')}</span></td>
                <td class="p-4 font-mono ${pClass}">${profit.toFixed(2)}%</td>
                <td class="p-4 font-mono text-xs">
                    <div><span class="text-text-secondary">الدخول:</span> ${entry.toFixed(4)}</div>
                    <div><span class="text-accent-blue">الحالي:</span> ${current.toFixed(4)}</div>
                    <div><span class="text-accent-green">الهدف:</span> ${target.toFixed(4)}</div>
                </td>
                <td class="p-4">
                    <button onclick="manualClose(${s.id}, '${s.symbol}')" class="bg-red-600 hover:bg-red-700 text-white font-bold py-1 px-3 rounded text-xs w-full">إغلاق</button>
                </td>
            </tr>`;
        });
    });
}
function updateNotifications() {
    fetchData('/api/notifications').then(data => {
        if (!data) return;
        document.getElementById('notifications-list').innerHTML = data.map(n => `<div class="p-2 border-b border-border-color"><span class="font-mono text-xs text-text-secondary">${new Date(n.timestamp).toLocaleString('ar-EG')}</span>: ${n.message}</div>`).join('');
    });
}
function updateRejections() {
    fetchData('/api/rejection_logs').then(data => {
        if (!data) return;
        document.getElementById('rejections-list').innerHTML = data.map(r => `<div class="p-2 border-b border-border-color"><span class="font-mono text-xs text-text-secondary">${new Date(r.timestamp).toLocaleString('ar-EG')}</span>: <strong class="text-accent-yellow">${r.symbol}</strong> - ${r.reason} <span class="text-xs text-gray-500">${JSON.stringify(r.details)}</span></div>`).join('');
    });
}
function manualClose(signalId, symbol) {
    if(confirm(\`هل أنت متأكد من رغبتك في إغلاق الصفقة لـ \${symbol} يدوياً؟\`)) {
        fetch(\`/api/signals/close/\${signalId}\`, { method: 'POST' })
            .then(res => res.json())
            .then(data => { if(data.success) { updateSignals(); } else { alert(data.message); } });
    }
}
function toggleTrading() { fetch('/api/trading/toggle', { method: 'POST' }).then(() => updateMarketStatus()); }

function saveSettings() {
    const settings = {
        use_bb_stoch_strategy: document.getElementById('bb-stoch-strategy-toggle').checked,
        use_macd_ema_strategy: document.getElementById('macd-ema-strategy-toggle').checked,
        use_sr_breakout_strategy: document.getElementById('sr-breakout-strategy-toggle').checked,
        use_support_bounce_strategy: document.getElementById('support-bounce-strategy-toggle').checked,
        use_triangle_breakout_strategy: document.getElementById('triangle-breakout-strategy-toggle').checked,
        use_macd_convergence_strategy: document.getElementById('macd-convergence-strategy-toggle').checked,
    };
    const feedbackEl = document.getElementById('settings-feedback');
    feedbackEl.textContent = 'جاري الحفظ...';
    fetch('/api/settings/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
    })
    .then(res => res.json())
    .then(data => {
        feedbackEl.textContent = data.success ? '✅ تم حفظ الإعدادات بنجاح!' : `❌ فشل الحفظ: ${data.message}`;
        setTimeout(() => { feedbackEl.textContent = ''; }, 3000);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    ['MarketStatus', 'Signals', 'Notifications', 'Rejections'].forEach(f => window[`update${f}`]());
    setInterval(updateMarketStatus, 5000); setInterval(updateSignals, 7000);
    setInterval(updateNotifications, 15000); setInterval(updateRejections, 15000);
});
</script>
</body></html>
"""

@app.route('/')
def home(): return render_template_string(get_dashboard_html())

@app.route('/api/market_status')
def get_market_status():
    with market_state_lock: state_copy = dict(current_market_state)
    with trading_status_lock: is_enabled = is_trading_enabled
    usdt_balance = None
    if client:
        try: usdt_balance = float(client.get_asset_balance(asset='USDT')['free'])
        except: usdt_balance = 'N/A'

    with bb_stoch_strategy_lock: use_bb_stoch = USE_BB_STOCH_STRATEGY
    with macd_ema_strategy_lock: use_macd_ema = USE_MACD_EMA_STRATEGY
    with sr_breakout_strategy_lock: use_sr_breakout = USE_SR_BREAKOUT_STRATEGY
    with support_bounce_strategy_lock: use_support_bounce = USE_SUPPORT_BOUNCE_STRATEGY
    with triangle_breakout_strategy_lock: use_triangle_breakout = USE_TRIANGLE_BREAKOUT_STRATEGY
    with macd_convergence_strategy_lock: use_macd_convergence = USE_MACD_CONVERGENCE_STRATEGY
    with signal_cache_lock: open_trades = len(open_signals_cache)

    return jsonify({
        "market_state": state_copy, "usdt_balance": usdt_balance,
        "is_trading_enabled": is_enabled,
        "open_trades_count": open_trades, "max_open_trades": MAX_OPEN_TRADES,
        "settings": {
            "use_bb_stoch_strategy": use_bb_stoch,
            "use_macd_ema_strategy": use_macd_ema,
            "use_sr_breakout_strategy": use_sr_breakout,
            "use_support_bounce_strategy": use_support_bounce,
            "use_triangle_breakout_strategy": use_triangle_breakout,
            "use_macd_convergence_strategy": use_macd_convergence,
        }
    })

@app.route('/api/signals')
def get_signals():
    if not (check_db_connection() and redis_client):
        return jsonify({"error": "Service connection failed"}), 500
    try:
        current_prices = redis_client.hgetall(REDIS_PRICES_HASH_NAME)
        with signal_cache_lock:
            signals_copy = list(open_signals_cache.values())
        
        for signal in signals_copy:
            current_price = current_prices.get(signal['symbol'])
            if current_price:
                signal['current_price'] = current_price
                signal['profit_percentage'] = ((float(current_price) - float(signal['entry_price'])) / float(signal['entry_price'])) * 100

        return jsonify(signals_copy)
    except Exception as e:
        logger.error(f"❌ [API إشارات] خطأ: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/notifications')
def get_notifications():
    with notifications_lock: return jsonify(list(notifications_cache))

@app.route('/api/rejection_logs')
def get_rejection_logs():
    with rejection_logs_lock: return jsonify(list(rejection_logs_cache))

@app.route('/api/trading/toggle', methods=['POST'])
def toggle_trading_status():
    global is_trading_enabled
    with trading_status_lock:
        is_trading_enabled = not is_trading_enabled
        status_msg = "مُفعّل" if is_trading_enabled else "مُعطّل"
        log_and_notify('warning', f"🚨 تم تغيير حالة التداول الحقيقي إلى: {status_msg}", "TRADING_STATUS_CHANGE")
        return jsonify({"message": f"Trading status set to {status_msg}"})

@app.route('/api/settings/update', methods=['POST'])
def update_settings():
    global USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_SR_BREAKOUT_STRATEGY, \
           USE_SUPPORT_BOUNCE_STRATEGY, USE_TRIANGLE_BREAKOUT_STRATEGY, USE_MACD_CONVERGENCE_STRATEGY
    try:
        data = request.get_json()
        
        with bb_stoch_strategy_lock: USE_BB_STOCH_STRATEGY = bool(data.get('use_bb_stoch_strategy', USE_BB_STOCH_STRATEGY))
        with macd_ema_strategy_lock: USE_MACD_EMA_STRATEGY = bool(data.get('use_macd_ema_strategy', USE_MACD_EMA_STRATEGY))
        with sr_breakout_strategy_lock: USE_SR_BREAKOUT_STRATEGY = bool(data.get('use_sr_breakout_strategy', USE_SR_BREAKOUT_STRATEGY))
        with support_bounce_strategy_lock: USE_SUPPORT_BOUNCE_STRATEGY = bool(data.get('use_support_bounce_strategy', USE_SUPPORT_BOUNCE_STRATEGY))
        with triangle_breakout_strategy_lock: USE_TRIANGLE_BREAKOUT_STRATEGY = bool(data.get('use_triangle_breakout_strategy', USE_TRIANGLE_BREAKOUT_STRATEGY))
        with macd_convergence_strategy_lock: USE_MACD_CONVERGENCE_STRATEGY = bool(data.get('use_macd_convergence_strategy', USE_MACD_CONVERGENCE_STRATEGY))

        log_and_notify('info', f"⚙️ تم تحديث إعدادات الاستراتيجيات من لوحة التحكم.", "SETTINGS_UPDATE")
        return jsonify({"success": True, "message": "Settings updated successfully"})
    except Exception as e:
        logger.error(f"❌ [API إعدادات] فشل تحديث الإعدادات: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 400


@app.route('/api/signals/close/<int:signal_id>', methods=['POST'])
def manual_close_trade_endpoint(signal_id):
    if not redis_client or not client: return jsonify({"success": False, "message": "Services not ready"}), 503
    with signal_cache_lock:
        signal_to_close = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)
    if not signal_to_close: return jsonify({"success": False, "message": "Signal not found"}), 404
    try:
        current_price = float(redis_client.hget(REDIS_PRICES_HASH_NAME, signal_to_close['symbol']))
    except (TypeError, ValueError):
        try: current_price = float(client.get_symbol_ticker(symbol=signal_to_close['symbol'])['price'])
        except Exception as e: return jsonify({"success": False, "message": f"Could not fetch price: {e}"}), 500

    if close_signal(signal_id, current_price, 'manual'):
        return jsonify({"success": True, "message": "Signal closed."})
    else:
        return jsonify({"success": False, "message": "Failed to close signal."}), 500

# ---------------------- حلقات النظام ----------------------
def trade_management_loop():
    logger.info("✅ [مدير الصفقات] بدء حلقة إدارة الصفقات...")
    while True:
        try:
            with signal_cache_lock:
                if not open_signals_cache:
                    time.sleep(5)
                    continue
                signals_to_check = list(open_signals_cache.values())

            if not redis_client:
                time.sleep(5)
                continue

            current_prices = redis_client.hgetall(REDIS_PRICES_HASH_NAME)

            for signal in signals_to_check:
                current_price_str = current_prices.get(signal['symbol'])
                if not current_price_str: continue

                current_price = float(current_price_str)
                signal_id, tp, sl = signal['id'], float(signal['target_price']), float(signal['stop_loss'])

                if current_price <= sl:
                    close_signal(signal_id, current_price, 'stop_loss')
                    continue

                if current_price >= tp:
                    close_signal(signal_id, current_price, 'take_profit')
                    continue
            time.sleep(2)
        except Exception as e:
            logger.error(f"❌ [مدير الصفقات] خطأ في حلقة الإدارة: {e}", exc_info=True)
            time.sleep(10)


def main_loop_enhanced():
    logger.info("[الحلقة الرئيسية] انتظار اكتمال التهيئة...")
    time.sleep(15)
    if not validated_symbols_to_scan:
        log_and_notify("critical", "قائمة العملات للمسح فارغة. يرجى التحقق من ملف 'crypto_list.txt'.", "SYSTEM_ERROR")
        return
    log_and_notify("info", f"✅ بدء حلقة المسح لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")

    while True:
        try:
            logger.info("🔄 [الحلقة الرئيسية] بدء دورة مسح جديدة...")
            determine_market_state_enhanced()
            btc_data = get_btc_data_for_bot()
            
            for symbol in random.sample(validated_symbols_to_scan, len(validated_symbols_to_scan)):
                try:
                    with signal_cache_lock:
                        if symbol in open_signals_cache or len(open_signals_cache) >= MAX_OPEN_TRADES:
                            continue
                    
                    df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    
                    if df_15m is None or len(df_15m) < 100:
                        continue
                    
                    df_with_indicators = calculate_all_features(df_15m, btc_data)
                    df_with_indicators.name = symbol
                    if df_with_indicators.empty:
                        continue
                    
                    if not check_market_volatility_filter(df_with_indicators): continue
                    if not check_trend_strength_filter(df_with_indicators): continue

                    signal_found, strategy_used, strategy_key = False, None, None

                    strategies_to_check = []
                    with bb_stoch_strategy_lock:
                        if USE_BB_STOCH_STRATEGY: strategies_to_check.append(('BB_Stoch', check_bb_stoch_strategy_revised, "BB_Stoch_Revised"))
                    with macd_ema_strategy_lock:
                        if USE_MACD_EMA_STRATEGY: strategies_to_check.append(('MACD_EMA', check_macd_ema_strategy, "MACD_EMA_Enhanced"))
                    with sr_breakout_strategy_lock:
                        if USE_SR_BREAKOUT_STRATEGY: strategies_to_check.append(('SR_Breakout', check_support_resistance_strategy_enhanced, "SR_Breakout_Enhanced"))
                    with support_bounce_strategy_lock:
                        if USE_SUPPORT_BOUNCE_STRATEGY: strategies_to_check.append(('Support_Bounce', check_support_bounce_strategy, "Support_Bounce_Enhanced"))
                    with triangle_breakout_strategy_lock:
                        if USE_TRIANGLE_BREAKOUT_STRATEGY: strategies_to_check.append(('Triangle_Breakout', check_triangle_breakout_strategy, "Triangle_Breakout_Enhanced"))
                    with macd_convergence_strategy_lock:
                        if USE_MACD_CONVERGENCE_STRATEGY: strategies_to_check.append(('MACD_Convergence', check_macd_convergence_strategy, "MACD_Convergence_Enhanced"))

                    for key, check_func, name in strategies_to_check:
                        if check_func(df_with_indicators):
                            signal_found, strategy_used, strategy_key = True, name, key
                            break
                    
                    if not signal_found:
                        continue

                    logger.info(f"  -> [{symbol}] إشارة ناجحة من {strategy_used}. جاري تحضير الصفقة...")
                    
                    try: entry_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                    except Exception as e: logger.error(f"❌ [{symbol}] فشل جلب سعر الدخول: {e}."); continue

                    stop_loss = calculate_optimal_stop_loss(df_with_indicators, strategy_key)
                    take_profit = calculate_optimal_take_profit(df_with_indicators, strategy_key, entry_price, stop_loss)

                    if stop_loss >= entry_price or take_profit <= entry_price:
                        log_rejection(symbol, "Insufficient data for TP/SL calculation", {"tp": take_profit, "sl": stop_loss, "entry": entry_price})
                        continue

                    rr_ratio = (take_profit - entry_price) / (entry_price - stop_loss)

                    new_signal = {
                        'symbol': symbol, 'strategy_name': strategy_used,
                        'entry_price': entry_price, 'target_price': take_profit, 'stop_loss': stop_loss,
                        'rr_ratio': rr_ratio, 'signal_details': {'rr_ratio': rr_ratio}
                    }

                    with trading_status_lock: is_enabled = is_trading_enabled
                    if is_enabled:
                        quantity = calculate_position_size(symbol, entry_price, new_signal['stop_loss'])
                        if quantity and quantity > 0:
                            order_result = place_order(symbol, Client.SIDE_BUY, quantity)
                            if order_result:
                                new_signal.update({'is_real_trade': True, 'quantity': float(quantity), 'order_id': order_result['orderId']})
                            else: continue
                        else: continue
                    
                    saved_signal = insert_signal_into_db(new_signal)
                    if saved_signal:
                        with signal_cache_lock: open_signals_cache[saved_signal['symbol']] = saved_signal
                        log_and_notify('info', f"إشارة: إشارة شراء جديدة لـ {symbol} من استراتيجية {strategy_used}", "NEW_SIGNAL")

                except Exception as e:
                    logger.error(f"❌ [خطأ معالجة] للرمز {symbol}: {e}", exc_info=True)
                finally:
                    time.sleep(0.2)
            
            gc.collect()
            sleep_duration = 60
            logger.info(f"✅ [نهاية الدورة] انتهت دورة المسح الكاملة. الانتظار {sleep_duration} ثانية...")
            time.sleep(sleep_duration)

        except (KeyboardInterrupt, SystemExit):
            log_and_notify("info", "إيقاف البوت.", "SYSTEM"); break
        except Exception as main_err:
            log_and_notify("error", f"خطأ حرج في الحلقة الرئيسية: {main_err}", "SYSTEM"); time.sleep(120)

def price_update_loop():
    if not redis_client: return
    while True:
        try:
            if validated_symbols_to_scan:
                tickers = client.get_symbol_ticker()
                prices_to_set = {t['symbol']: t['price'] for t in tickers if t['symbol'] in validated_symbols_to_scan}
                if prices_to_set: redis_client.hset(REDIS_PRICES_HASH_NAME, mapping=prices_to_set)
            time.sleep(1)
        except Exception as e: logger.error(f"خطأ في حلقة تحديث الأسعار: {e}"); time.sleep(10)

def initialize_bot_services():
    global client, validated_symbols_to_scan
    logger.info("🤖 [خدمات البوت] بدء التهيئة...")
    try:
        client = Client(API_KEY, API_SECRET)
        init_db()
        init_redis()
        get_exchange_info_map()
        load_open_signals_to_cache()
        load_notifications_to_cache()
        validated_symbols_to_scan = get_validated_symbols()
        Thread(target=main_loop_enhanced, daemon=True).start()
        Thread(target=price_update_loop, daemon=True).start()
        Thread(target=trade_management_loop, daemon=True).start()
        logger.info("✅ [خدمات البوت] تم بدء جميع الخدمات الخلفية بنجاح.")
        send_telegram_message("✅ *البوت قيد التشغيل الآن (نسخة V10.0 - محسنة)*")
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حرج أثناء التهيئة: {e}", "SYSTEM"); exit(1)

# ---------------------- نقطة الدخول ----------------------
if __name__ == "__main__":
    logger.info("🚀 إطلاق بوت التداول ولوحة التحكم (V10.0) 🚀")
    Thread(target=initialize_bot_services, daemon=True).start()
    port = int(os.environ.get('PORT', 10000))
    host = "0.0.0.0"
    logger.info(f"✅ بدء لوحة التحكم على {host}:{port}")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        app.run(host=host, port=port)
    logger.info("👋 [إيقاف] تم إيقاف تشغيل التطبيق.")
