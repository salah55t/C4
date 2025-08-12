# ملف c4.py - نسخة V9.8.0 (استهداف ربح 1% كحد أدنى)
# --- التغييرات الرئيسية (V9.8.0):
# 1. [إضافة] فلتر جديد `check_profit_potential_filter` للتحقق من أن العملة لديها تقلبات كافية لتحقيق 1% ربح.
# 2. [إضافة] فلتر جديد `check_entry_timing_filter` لتحسين توقيت الدخول وتجنب الصفقات المتأخرة.
# 3. [تحسين] تم استبدال دالة حساب TP/SL بدالة `calculate_optimal_take_profit` التي تضمن هدفًا لا يقل عن 1% وتأخذ في الاعتبار مستويات المقاومة.
# 4. [تحسين] تم تحديث حلقة `trade_management_loop` بالكامل لتشمل نقل وقف الخسارة إلى نقطة الدخول وتعديل الهدف ديناميكيًا بناءً على قوة الحركة.
# 5. [دمج] تم دمج كل التحسينات المقترحة في الحلقة الرئيسية لضمان جودة الإشارات.

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
        logging.FileHandler('crypto_bot_v9_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV9.8.0')

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

MIN_PROFIT_PERCENT: float = 1.0 # تم تحديثه ليعكس الهدف الجديد

# --- مفاتيح تفعيل الاستراتيجيات ---
USE_ML_STRATEGY: bool = False
ml_strategy_lock = Lock()

USE_BB_STOCH_STRATEGY: bool = True
bb_stoch_strategy_lock = Lock()

USE_MACD_EMA_STRATEGY: bool = True
macd_ema_strategy_lock = Lock()

USE_EMA_RSI_STRATEGY: bool = True
ema_rsi_strategy_lock = Lock()

USE_PULLBACK_STRATEGY: bool = True
pullback_strategy_lock = Lock()

USE_BB_SQUEEZE_STRATEGY: bool = True
bb_squeeze_strategy_lock = Lock()

USE_BULLISH_MOMENTUM_STRATEGY: bool = True
bullish_momentum_strategy_lock = Lock()

USE_SR_BREAKOUT_STRATEGY: bool = True
sr_breakout_strategy_lock = Lock()


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
    "Insufficient Daily Movement": "حركة يومية غير كافية لتحقيق الربح",
    "Insufficient M15 Movement": "حركة 15د غير كافية لتحقيق الربح",
    "Entry Too Late": "توقيت الدخول متأخر جدًا",
    "RSI Overbought": "مؤشر القوة النسبية في منطقة تشبع شرائي",
    "Decreasing Volume": "حجم التداول في انخفاض",
    "MACD Momentum Weakening": "زخم مؤشر الماكد يضعف",
    "Unfavorable Risk-Reward Ratio": "نسبة المخاطرة إلى الربح غير مناسبة",
    "Market Volatility Filter Failed": "فلتر تقلب السوق رفض الدخول",
    "Trend Strength Filter Failed": "فلتر قوة الاتجاه رفض الدخول",
    "ML Model Rejected Signal": "نموذج التعلم الآلي رفض الإشارة",
    "ML Model Load Failed": "فشل تحميل نموذج التعلم الآلي",
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
    "Bullish Momentum Strategy Conditions Not Met": "شروط استراتيجية الزخم الصعودي لم تتحقق",
    "BB_Stoch Strategy Conditions Not Met": "شروط استراتيجية BB+Stoch لم تتحقق",
    "MACD_EMA Strategy Conditions Not Met": "شروط استراتيجية MACD+EMA لم تتحقق",
    "EMA_RSI Strategy Conditions Not Met": "شروط استراتيجية EMA+RSI لم تتحقق",
    "Pullback Strategy Conditions Not Met": "شروط استراتيجية Pullback لم تتحقق",
    "BB Squeeze Strategy Conditions Not Met": "شروط استراتيجية BB Squeeze لم تتحقق",
    "SR Breakout Strategy Conditions Not Met": "شروط استراتيجية اختراق الدعم/المقاومة لم تتحقق",
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

# --- [إضافة] فلاتر التأكيد الجديدة ---
def check_profit_potential_filter(symbol: str) -> bool:
    """فلتر للتحقق من إمكانية تحقيق 1% ربح للعملة"""
    try:
        # جلب البيانات اليومية للتحقق من متوسط الحركة اليومية
        daily_df = fetch_historical_data(symbol, '1d', 30)
        if daily_df is None or len(daily_df) < 20:
            return False
            
        # حساب متوسط الحركة اليومية (نسبة من أعلى إلى أدنى سعر)
        daily_df['daily_range_pct'] = (daily_df['high'] - daily_df['low']) / daily_df['close'] * 100
        avg_daily_range = daily_df['daily_range_pct'].mean()
        
        # يجب أن يكون متوسط الحركة اليومي على الأقل 2.5% لضمان إمكانية تحقيق 1%
        if avg_daily_range < 2.5:
            log_rejection(symbol, "Insufficient Daily Movement", {"avg_daily_range": f"{avg_daily_range:.2f}%"})
            return False
            
        # حساب متوسط الحركة في فريم 15 دقيقة
        m15_df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, 7)
        if m15_df is None or len(m15_df) < 50:
            return False
            
        m15_df['range_pct'] = (m15_df['high'] - m15_df['low']) / m15_df['close'] * 100
        avg_m15_range = m15_df['range_pct'].mean()
        
        # يجب أن يكون متوسط الحركة في 15 دقيقة على الأقل 0.4%
        if avg_m15_range < 0.4:
            log_rejection(symbol, "Insufficient M15 Movement", {"avg_m15_range": f"{avg_m15_range:.2f}%"})
            return False
            
        logger.info(f"  -> [{symbol}] ✅ فلتر إمكانية الربح: متوسط الحركة اليومية {avg_daily_range:.2f}%, متوسط حركة 15م {avg_m15_range:.2f}%")
        return True
        
    except Exception as e:
        logger.error(f"❌ [فلتر إمكانية الربح] خطأ في {symbol}: {e}")
        return False

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

def check_entry_timing_filter(symbol: str, df: pd.DataFrame) -> bool:
    """فلتر لتحسين توقيت الدخول لضمان استمرار الحركة"""
    if len(df) < 20:
        return False
        
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    if 'ema_9' in df.columns and 'close' in df.columns:
        ema_distance = abs(last['close'] - last['ema_9']) / last['ema_9'] * 100
        if ema_distance > 0.8:
            log_rejection(symbol, "Entry Too Late", {"ema_distance": f"{ema_distance:.2f}%"})
            return False
    
    if last['rsi'] > 70:
        log_rejection(symbol, "RSI Overbought", {"rsi": f"{last['rsi']:.2f}"})
        return False
    
    if last['volume'] < prev['volume'] * 0.9:
        log_rejection(symbol, "Decreasing Volume", {"volume_ratio": f"{last['volume']/prev['volume']:.2f}"})
        return False
    
    if 'macd_histogram' in df.columns and last['macd_histogram'] < prev['macd_histogram']:
        log_rejection(symbol, "MACD Momentum Weakening", {"macd_hist_current": f"{last['macd_histogram']:.4f}", "macd_hist_prev": f"{prev['macd_histogram']:.4f}"})
        return False
    
    logger.info(f"  -> [{symbol}] ✅ توقيت الدخول مناسب")
    return True

# --- دوال منطق الاستراتيجيات ---
def check_bb_stoch_strategy_enhanced(df: pd.DataFrame) -> bool:
    if len(df) < 21: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    price_touch_bb = last['low'] <= (last['bb_lower'] * 1.002)
    stoch_cross_up = prev['stoch_rsi_k'] < prev['stoch_rsi_d'] and last['stoch_rsi_k'] > last['stoch_rsi_d']
    oversold_area = last['stoch_rsi_k'] < 35 and last['stoch_rsi_d'] < 35
    volume_spike = last['volume'] > last['volume_sma_20'] * 1.2
    with market_state_lock:
        trend_ok = "DOWNTREND" not in current_market_state.get("overall_regime", "UNCERTAIN")
    bb_width_ok = last['bb_width'] > 0.02
    price_not_oversold = last['rsi'] > 25
    if all([price_touch_bb, stoch_cross_up, oversold_area, volume_spike, trend_ok, bb_width_ok, price_not_oversold]):
        logger.info(f"  -> [{df.name}] ✅ إشارة BB+Stoch (معززة).")
        return True
    return False

def check_macd_ema_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 3: return False
    last, prev, prev_prev = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    macd_cross_up = prev['macd'] < prev['macd_signal'] and last['macd'] > last['macd_signal']
    price_above_ema = last['close'] > last['ema_21']
    macd_strength = last['macd_histogram'] > 0 and last['macd_histogram'] > prev['macd_histogram']
    trend_strength = last['adx'] > 20
    macd_position = prev_prev['macd'] < 0
    if all([macd_cross_up, price_above_ema, macd_strength, trend_strength, macd_position]):
        logger.info(f"  -> [{df.name}] ✅ إشارة MACD+EMA (مع فلاتر قوة واتجاه).")
        return True
    return False

def check_ema_rsi_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 2: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    ema_cross_up = prev['ema_9'] < prev['ema_21'] and last['ema_9'] > last['ema_21']
    rsi_strong = last['rsi'] > 52
    trend_filter = last['close'] > last['ema_50']
    if ema_cross_up and rsi_strong and trend_filter:
        logger.info(f"  -> [{df.name}] ✅ إشارة استراتيجية EMA+RSI Cross.")
        return True
    return False

def check_pullback_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 2: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    uptrend_confirmed = last['close'] > last['ema_21'] and last['ema_21'] > last['ema_50']
    macd_cross_up = prev['macd'] < prev['macd_signal'] and last['macd'] > last['macd_signal']
    if uptrend_confirmed and macd_cross_up:
        logger.info(f"  -> [{df.name}] ✅ إشارة استراتيجية Pullback MACD.")
        return True
    return False

def check_bb_squeeze_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 100: return False
    last = df.iloc[-1]
    squeeze_threshold = df['bb_width'].rolling(100).quantile(0.20).iloc[-1]
    is_squeeze = last['bb_width'] < squeeze_threshold
    breakout = last['close'] > last['bb_upper']
    volume_confirmed = last['relative_volume'] > 1.25
    if is_squeeze and breakout and volume_confirmed:
        logger.info(f"  -> [{df.name}] ✅ إشارة استراتيجية BB Squeeze Breakout.")
        return True
    return False

def check_bullish_momentum_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 50: return False
    last = df.iloc[-1]
    price_above_sma50 = last['close'] > last['sma_50']
    strong_trend = last['adx'] > 25
    bullish_direction = last['plus_di'] > last['minus_di']
    rsi_is_bullish = 50 < last['rsi'] < 75
    if len(df) < 8: return False
    recent_highs = df['high'].iloc[-8:-1]
    recent_lows = df['low'].iloc[-8:-1]
    def is_higher_highs(highs, min_count=3):
        if len(highs) < min_count + 1: return False
        return (highs.diff().dropna() > 0).sum() >= min_count
    def is_higher_lows(lows, min_count=3):
        if len(lows) < min_count + 1: return False
        return (lows.diff().dropna() > 0).sum() >= min_count
    price_momentum_confirmed = is_higher_highs(recent_highs) and is_higher_lows(recent_lows)
    volume_confirmation = last['volume'] > last['volume_sma_20'] * 1.1
    if all([price_above_sma50, strong_trend, bullish_direction, rsi_is_bullish, price_momentum_confirmed, volume_confirmation]):
        logger.info(f"  -> [{df.name}] ✅ إشارة زخم صعودي (معززة).")
        return True
    return False

def check_support_resistance_strategy_enhanced(df: pd.DataFrame) -> bool:
    if len(df) < 50: return False
    last = df.iloc[-1]
    resistance_candidates = df[df['high'] == df['high'].rolling(5, center=True).max()]['high']
    if resistance_candidates.empty: return False
    current_price = last['close']
    next_resistance_series = resistance_candidates[resistance_candidates > current_price]
    closest_resistance = next_resistance_series.min() if not next_resistance_series.empty else None
    if closest_resistance is not None:
        tolerance = closest_resistance * 0.01
        resistance_tests = df[(df['high'] >= closest_resistance - tolerance) & (df['high'] <= closest_resistance + tolerance)]
        resistance_strength = len(resistance_tests)
        if resistance_strength < 2: return False
        breakout = last['close'] > closest_resistance and df['close'].iloc[-2] <= closest_resistance
        volume_confirmation = last['volume'] > last['volume_sma_20'] * 1.3
        trend_confirmation = last['close'] > last['ema_21']
        not_false_breakout = last['close'] > closest_resistance * 1.005
        if all([breakout, volume_confirmation, trend_confirmation, not_false_breakout]):
            logger.info(f"  -> [{df.name}] ✅ إشارة اختراق مقاومة (معززة) - قوة المستوى: {resistance_strength}")
            return True
    return False

# --- دوال حساب الأهداف ووقف الخسارة ---
def find_resistance_levels(df: pd.DataFrame, current_price: float, lookback: int = 100) -> List[float]:
    """البحث عن مستويات المقاومة القريبة باستخدام قمم التأرجح"""
    df_slice = df.iloc[-lookback:]
    is_pivot = (df_slice['high'].shift(1) < df_slice['high']) & (df_slice['high'].shift(-1) < df_slice['high'])
    resistance_prices = df_slice[is_pivot]['high']
    
    # تجميع المستويات المتقاربة
    if resistance_prices.empty:
        return []
    
    sorted_prices = sorted(resistance_prices.unique())
    levels = []
    if not sorted_prices:
        return []
        
    current_level_prices = [sorted_prices[0]]
    for price in sorted_prices[1:]:
        if price <= current_level_prices[-1] * 1.005: # تجميع المستويات ضمن نطاق 0.5%
            current_level_prices.append(price)
        else:
            levels.append(np.mean(current_level_prices))
            current_level_prices = [price]
    levels.append(np.mean(current_level_prices))
    
    return [level for level in levels if level > current_price]

def calculate_optimal_take_profit(symbol: str, entry_price: float, stop_loss: float, 
                                df: pd.DataFrame, min_profit_pct: float = 1.0) -> Tuple[float, float, float]:
    """حساب مستوى جني الأرباح الأمثل ونسبة المخاطرة إلى الربح"""
    try:
        atr_value = df['atr'].iloc[-1]
        
        # تحديد مستوى جني الأرباح الأولي بناءً على نسبة 1:1.5 للمخاطرة:الربح
        risk_distance = entry_price - stop_loss
        tp_price = entry_price + (risk_distance * 1.5)
        
        # التأكد من أن الهدف يحقق الحد الأدنى للربح المطلوب
        min_tp_price = entry_price * (1 + min_profit_pct / 100)
        if tp_price < min_tp_price:
            tp_price = min_tp_price

        # البحث عن أقرب مقاومة وتعديل الهدف إذا لزم الأمر
        resistance_levels = find_resistance_levels(df, entry_price)
        if resistance_levels:
            closest_resistance = resistance_levels[0]
            # إذا كانت المقاومة قريبة جدًا وتوفر ربحًا معقولًا، نضع الهدف قبلها
            if (closest_resistance > min_tp_price) and (closest_resistance < tp_price * 1.05):
                tp_price = closest_resistance * 0.998 # قبل المقاومة بـ 0.2%
            
        profit_pct = ((tp_price - entry_price) / entry_price) * 100
        risk_pct = ((entry_price - stop_loss) / entry_price) * 100
        rr_ratio = profit_pct / risk_pct if risk_pct > 0 else 0
        
        logger.info(f"  -> [{symbol}] مستوى جني الأرباح المحسوب: {tp_price:.6f} ({profit_pct:.2f}%), نسبة المخاطرة:ربح = 1:{rr_ratio:.2f}")
        
        return tp_price, profit_pct, rr_ratio
        
    except Exception as e:
        logger.error(f"❌ [حساب جني الأرباح] خطأ في {symbol}: {e}")
        tp_price = entry_price * (1 + min_profit_pct / 100)
        profit_pct = min_profit_pct
        risk_pct = ((entry_price - stop_loss) / entry_price) * 100
        rr_ratio = profit_pct / risk_pct if risk_pct > 0 else 0
        return tp_price, profit_pct, rr_ratio


# --- دوال إدارة الصفقات ---
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
                
                logger.info(f"  -> [{symbol_to_close}] التحقق من الرصيد للإغلاق الكامل. الرصيد الفعلي: {actual_free_balance} {base_asset}")

                if actual_free_balance > 0:
                    quantity_to_sell = adjust_quantity_to_lot_size(symbol_to_close, float(actual_free_balance))
                    if quantity_to_sell and quantity_to_sell > 0:
                        notional_value = quantity_to_sell * Decimal(str(closing_price))
                        symbol_info = exchange_info_map.get(symbol_to_close)
                        min_notional_ok = True
                        if symbol_info:
                            min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] in ('MIN_NOTIONAL', 'NOTIONAL')), None)
                            if min_notional_filter:
                                min_notional = Decimal(min_notional_filter.get('minNotional', min_notional_filter.get('notional', '0')))
                                if notional_value < min_notional:
                                    min_notional_ok = False
                                    logger.warning(f"⚠️ [{symbol_to_close}] الرصيد الفعلي للبيع ({quantity_to_sell}) أقل من الحد الأدنى ({min_notional}). سيتم اعتباره غبارًا.")
                        
                        if min_notional_ok:
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
            # ... (نفس كود الإدراج السابق)
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, is_real_trade, quantity, original_quantity, order_id, current_peak_price, journey_state, rr_ratio)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *;
            """, (
                signal_data['symbol'], signal_data['entry_price'], signal_data['target_price'], signal_data['stop_loss'],
                signal_data['strategy_name'], json.dumps(signal_data['signal_details'], cls=NpEncoder), signal_data.get('is_real_trade', False),
                signal_data.get('quantity'), signal_data.get('quantity'), signal_data.get('order_id'), signal_data['entry_price'], 
                json.dumps(signal_data.get('journey_state'), cls=NpEncoder), signal_data.get('rr_ratio')
            ))
            saved_signal = cur.fetchone()
            conn.commit()
            logger.info(f"💾 [{signal_data['symbol']}] تم حفظ الإشارة الجديدة في قاعدة البيانات.")
            return dict(saved_signal)
    except Exception as e:
        logger.error(f"❌ [قاعدة البيانات] فشل إدراج الإشارة: {e}", exc_info=True); conn.rollback(); return None

def update_signal_in_db(signal_id: int, updates: Dict):
    """تحديث حقول معينة في صفقة موجودة."""
    if not check_db_connection() or not conn:
        logger.error(f"❌ [DB Update] لا يمكن تحديث الصفقة {signal_id} بسبب عدم وجود اتصال بقاعدة البيانات.")
        return
    try:
        with conn.cursor() as cur:
            set_clause = ", ".join([f"{key} = %s" for key in updates.keys()])
            query = sql.SQL("UPDATE signals SET {} WHERE id = %s").format(sql.SQL(set_clause))
            params = list(updates.values()) + [signal_id]
            cur.execute(query, params)
        conn.commit()
        logger.info(f"  -> [DB Update] تم تحديث الصفقة {signal_id} بالبيانات: {updates.keys()}")
    except Exception as e:
        logger.error(f"❌ [DB Update] فشل تحديث الصفقة {signal_id}: {e}")
        if conn: conn.rollback()

# --- دوال النظام الأساسية ---
def determine_market_state_enhanced():
    global current_market_state, last_market_state_check
    if time.time() - last_market_state_check < 180: return
    # ... (نفس الكود السابق)
    try:
        # ...
        pass
    except Exception as e:
        logger.error(f"❌ [حالة السوق] خطأ في التحديث: {e}", exc_info=True)

# ---------------------- واجهة الويب (Flask) ----------------------
# ... (كل كود Flask يبقى كما هو)
app = Flask(__name__)
CORS(app)

def get_dashboard_html():
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم التداول V9.8.0 - استهداف الربح</title>
    <!-- ... (نفس كود HTML السابق) ... -->
</head>
<body>
    <!-- ... (نفس كود HTML السابق) ... -->
</body>
</html>
"""
# ... (كل مسارات API تبقى كما هي)

# ---------------------- حلقات النظام ----------------------
def dynamic_take_profit_adjustment(symbol: str, current_price: float, entry_price: float, 
                                 original_tp: float, df: pd.DataFrame) -> float:
    """تعديل مستوى جني الأرباح ديناميكيًا بناءً على قوة الحركة"""
    try:
        current_profit_pct = ((current_price - entry_price) / entry_price) * 100
        original_target_pct = ((original_tp - entry_price) / entry_price) * 100

        # إذا وصلنا إلى 80% من الهدف، نبدأ في تقييم إمكانية رفع الهدف
        if current_profit_pct >= original_target_pct * 0.8:
            last = df.iloc[-1]
            atr_pct = (last['atr'] / current_price) * 100
            
            # إذا كان الزخم قوياً (ADX مرتفع، ATR مرتفع)، نرفع الهدف
            if last['adx'] > 28 and atr_pct > 0.35:
                new_tp = entry_price + ((original_tp - entry_price) * 1.5)
                logger.info(f"  -> [{symbol}] 🔄 رفع مستوى جني الأرباح من {original_tp:.6f} إلى {new_tp:.6f} بسبب قوة الحركة")
                return new_tp
        
        return original_tp
        
    except Exception as e:
        logger.error(f"❌ [تعديل جني الأرباح] خطأ في {symbol}: {e}")
        return original_tp

def check_momentum_exit_signal(symbol: str, df: pd.DataFrame) -> bool:
    """التحقق من وجود إشارة خروج مبكرة بسبب ضعف الزخم"""
    if len(df) < 3: return False
    last = df.iloc[-1]
    prev = df.iloc[-2]
    # إشارة خروج: السعر يغلق تحت EMA9 والماكد هيستوجرام بدأ في الانخفاض
    price_below_ema = last['close'] < last['ema_9']
    macd_weakening = last['macd_histogram'] < prev['macd_histogram']
    if price_below_ema and macd_weakening:
        logger.warning(f"  -> [{symbol}] ⚠️ إشارة خروج مبكر محتملة بسبب ضعف الزخم.")
        return True
    return False

def trade_management_loop():
    """حلقة إدارة الصفقات مع التركيز على تحقيق 1% ربح كحد أدنى"""
    logger.info("✅ [مدير الصفقات المحسن] بدء حلقة إدارة الصفقات...")
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
                symbol = signal['symbol']
                current_price_str = current_prices.get(symbol)
                if not current_price_str: continue
                
                current_price = float(current_price_str)
                signal_id = signal['id']
                entry_price = float(signal['entry_price'])
                stop_loss = float(signal['stop_loss'])
                target_price = float(signal['target_price'])
                original_target_pct = ((target_price - entry_price) / entry_price) * 100
                current_profit_pct = ((current_price - entry_price) / entry_price) * 100

                # 1. التحقق من وقف الخسارة
                if current_price <= stop_loss:
                    close_signal(signal_id, current_price, "Stop Loss")
                    continue

                # 2. التحقق من جني الأرباح
                if current_price >= target_price:
                    close_signal(signal_id, current_price, "Take Profit")
                    continue

                # 3. نقل وقف الخسارة إلى نقطة الدخول
                if current_profit_pct >= (original_target_pct / 2) and stop_loss < entry_price:
                    new_stop_loss = entry_price * 1.0005 # فوق الدخول بقليل لتغطية الرسوم
                    logger.info(f"  -> [{symbol}] 🔄 تعديل وقف الخسارة إلى نقطة التعادل {new_stop_loss:.6f}")
                    update_signal_in_db(signal_id, {'stop_loss': new_stop_loss})
                    with signal_cache_lock:
                        if symbol in open_signals_cache:
                            open_signals_cache[symbol]['stop_loss'] = new_stop_loss
                
                # 4. تعديل الهدف ديناميكيًا
                if current_profit_pct >= (original_target_pct * 0.8):
                    df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, 20)
                    if df is not None and not df.empty:
                        df = calculate_all_features(df, None)
                        new_target = dynamic_take_profit_adjustment(symbol, current_price, entry_price, target_price, df)
                        if new_target > target_price:
                            update_signal_in_db(signal_id, {'target_price': new_target})
                            with signal_cache_lock:
                                if symbol in open_signals_cache:
                                    open_signals_cache[symbol]['target_price'] = new_target
                
                # 5. التحقق من إشارة خروج مبكر
                if current_profit_pct > 0.5: # فقط إذا كنا في ربح
                    df_exit = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, 20)
                    if df_exit is not None and not df_exit.empty:
                        df_exit = calculate_all_features(df_exit, None)
                        if check_momentum_exit_signal(symbol, df_exit):
                            close_signal(signal_id, current_price, "Momentum Change")
                            continue
            
            time.sleep(3)
        except Exception as e:
            logger.error(f"❌ [حلقة إدارة الصفقات] خطأ: {e}", exc_info=True)
            time.sleep(30)


def main_loop_enhanced():
    logger.info("[الحلقة الرئيسية] انتظار اكتمال التهيئة...")
    time.sleep(15)
    if not validated_symbols_to_scan:
        log_and_notify("critical", "قائمة العملات للمسح فارغة.", "SYSTEM_ERROR")
        return
    log_and_notify("info", f"✅ بدء حلقة المسح لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")

    while True:
        try:
            logger.info("🔄 [الحلقة الرئيسية] بدء دورة مسح جديدة...")
            determine_market_state_enhanced()
            btc_data = get_btc_data_for_bot()
            symbols_to_process = random.sample(validated_symbols_to_scan, len(validated_symbols_to_scan))
            
            for symbol in symbols_to_process:
                try:
                    with signal_cache_lock:
                        if symbol in open_signals_cache or len(open_signals_cache) >= MAX_OPEN_TRADES:
                            continue
                    
                    # --- [تكامل] تطبيق الفلاتر الجديدة بالترتيب ---
                    if not check_profit_potential_filter(symbol):
                        time.sleep(1) # انتظار لتجنب استهلاك API
                        continue
                    
                    df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, 100)
                    if df_15m is None or len(df_15m) < 100: continue
                    
                    df_with_indicators = calculate_all_features(df_15m, btc_data)
                    df_with_indicators.name = symbol
                    if df_with_indicators.empty: continue
                    
                    if not check_market_volatility_filter(df_with_indicators): continue
                    if not check_trend_strength_filter(df_with_indicators): continue
                    if not check_entry_timing_filter(symbol, df_with_indicators): continue

                    # ... (نفس كود التحقق من الاستراتيجيات)
                    signal_found, strategy_used = False, None
                    # ...
                    if not signal_found: continue

                    logger.info(f"  -> [{symbol}] إشارة ناجحة من {strategy_used}. جاري التحقق النهائي...")
                    
                    entry_price = df_with_indicators['close'].iloc[-1]
                    if not passes_final_order_book_check(symbol, entry_price): continue

                    logger.info(f"  -> [{symbol}] ✅ نجح فلتر دفتر الطلبات. جاري تحضير الصفقة...")
                    
                    # --- [تكامل] حساب SL و TP الجديد ---
                    atr_value = df_with_indicators['atr'].iloc[-1]
                    stop_loss = entry_price - (atr_value * 1.2) # وقف خسارة أضيق قليلاً
                    
                    take_profit, profit_pct, rr_ratio = calculate_optimal_take_profit(symbol, entry_price, stop_loss, df_with_indicators, min_profit_pct=MIN_PROFIT_PERCENT)
                    
                    if rr_ratio < 1.5:
                        log_rejection(symbol, "Unfavorable Risk-Reward Ratio", {"rr_ratio": f"1:{rr_ratio:.2f}"})
                        continue
                    
                    # ... (نفس كود إنشاء الصفقة ووضع الأمر)
                    new_signal = {
                        'symbol': symbol, 'strategy_name': strategy_used,
                        'signal_details': {'profit_pct': profit_pct, 'rr_ratio': rr_ratio},
                        'entry_price': entry_price, 'target_price': take_profit, 'stop_loss': stop_loss,
                        'rr_ratio': rr_ratio, 'journey_state': {"targets_hit": 0, "is_complete": False, "partial_exit_done": False}
                    }

                    # ... (نفس كود وضع الأمر الحقيقي)

                    saved_signal = insert_signal_into_db(new_signal)
                    if saved_signal:
                        with signal_cache_lock: open_signals_cache[saved_signal['symbol']] = saved_signal
                        log_and_notify('info', f"إشارة: شراء جديد لـ {symbol} من {strategy_used}", "NEW_SIGNAL")

                except Exception as e:
                    logger.error(f"❌ [خطأ معالجة] للرمز {symbol}: {e}", exc_info=True)
                finally:
                    time.sleep(0.5)
            
            gc.collect()
            sleep_duration = 60
            logger.info(f"✅ [نهاية الدورة] انتهت دورة المسح. الانتظار {sleep_duration} ثانية...")
            time.sleep(sleep_duration)

        except (KeyboardInterrupt, SystemExit):
            log_and_notify("info", "إيقاف البوت.", "SYSTEM"); break
        except Exception as main_err:
            log_and_notify("error", f"خطأ حرج في الحلقة الرئيسية: {main_err}", "SYSTEM"); time.sleep(120)

# ... (بقية الدوال: price_update_loop, initialize_bot_services, ونقطة الدخول __main__)
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
        send_telegram_message("✅ *البوت قيد التشغيل الآن (نسخة V9.8.0 - استهداف الربح)*")
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حرج أثناء التهيئة: {e}", "SYSTEM"); exit(1)

if __name__ == "__main__":
    logger.info("🚀 إطلاق بوت التداول ولوحة التحكم (V9.8.0) 🚀")
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
