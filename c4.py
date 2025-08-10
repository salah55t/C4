# ملف c4.py - نسخة V9.7.0 (إضافة نظام مراقبة الأخبار الاقتصادية)
# --- التغييرات الرئيسية (V9.7.0):
# 1. [إضافة] نظام جلب الأخبار الاقتصادية من NewsData.io API.
# 2. [إضافة] حلقة مراقبة في الخلفية للتحقق من الأخبار القادمة عالية التأثير.
# 3. [إضافة] آلية لإيقاف البوت مؤقتاً لمدة 15 دقيقة قبل صدور الأخبار الهامة.
# 4. [إضافة] إرسال إشعارات تليجرام عند إيقاف واستئناف البوت بسبب الأخبار.
# 5. [إضافة] قسم جديد "الأخبار الاقتصادية" في واجهة الويب لعرض آخر الأخبار.
# 6. [إضافة] مؤشر لحالة الإيقاف بسبب الأخبار في لوحة التحكم.

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
logger = logging.getLogger('CryptoBotV9.7.0')

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
    # [إضافة] مفتاح API للأخبار
    NEWSDATA_API_KEY: str = config('NEWSDATA_API_KEY', default='pub_2c2d39760da740178b67ee1befd97206')

except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
    exit(1)

# --- متغيرات عامة وإعدادات البوت ---
is_trading_enabled: bool = False
trading_status_lock = Lock()

# --- [إضافة] متغيرات حالة إيقاف الأخبار ---
is_bot_paused_by_news: bool = False
news_pause_reason: str = ""
bot_pause_lock = Lock()

# --- المتغيرات القابلة للتعديل ---
RISK_PER_TRADE_PERCENT: float = 0.85
risk_per_trade_lock = Lock()

BUY_CONFIDENCE_THRESHOLD = 0.53
buy_confidence_lock = Lock()

ORDER_BOOK_MIN_BID_ASK_RATIO: float = 1.18
order_book_ratio_lock = Lock()

VOLUME_FILTER_MULTIPLIER: float = 1.07
volume_filter_lock = Lock()

MIN_PROFIT_PERCENT: float = 0.7

# --- إعدادات الفلاتر والاستراتيجيات ---
USE_CANDLESTICK_FILTER: bool = True
candle_filter_lock = Lock()

USE_VOLUME_FILTER: bool = True
# volume_filter_lock is used for both USE_VOLUME_FILTER and VOLUME_FILTER_MULTIPLIER

USE_ORDER_BOOK_FILTER: bool = True
order_book_filter_enable_lock = Lock()

# --- فلتر تأكيد الترند على فريم أعلى ---
USE_HTF_CONFIRMATION_FILTER: bool = True
htf_confirmation_lock = Lock()

# --- فلتر الزخم قصير الأجل ---
USE_SHORT_TERM_MOMENTUM_FILTER: bool = True
short_term_momentum_filter_lock = Lock()


# --- مفاتيح تفعيل الاستراتيجيات ---
USE_ML_STRATEGY: bool = False
ml_strategy_lock = Lock()

USE_BB_STOCH_STRATEGY: bool = True
bb_stoch_strategy_lock = Lock()

USE_MACD_EMA_STRATEGY: bool = True
macd_ema_strategy_lock = Lock()

USE_QQE_SSL_STRATEGY: bool = True
qqe_ssl_strategy_lock = Lock()

USE_EMA_RSI_STRATEGY: bool = True
ema_rsi_strategy_lock = Lock()

USE_PULLBACK_STRATEGY: bool = True
pullback_strategy_lock = Lock()

USE_BB_SQUEEZE_STRATEGY: bool = True
bb_squeeze_strategy_lock = Lock()

USE_BULLISH_MOMENTUM_STRATEGY: bool = True
bullish_momentum_strategy_lock = Lock()

USE_SMART_BREAKOUT_STRATEGY: bool = True
smart_breakout_strategy_lock = Lock()

USE_RSI_DIVERGENCE_STRATEGY: bool = True
rsi_divergence_strategy_lock = Lock()

USE_VWAP_REVERSAL_STRATEGY: bool = True
vwap_reversal_strategy_lock = Lock()


# --- تحديد الاستراتيجيات التي تعتبر سكالبينج لتخفيف الفلاتر ---
SCALPING_STRATEGIES = ["Pullback_MACD", "BB_Squeeze_Breakout", "QQE_SSL_Explosion"]


BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V9_With_Microstructure'
MODEL_FOLDER: str = 'V9'
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
EMA_FAST_PERIOD: int = 50
EMA_SLOW_PERIOD: int = 120
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
BTC_CORR_PERIOD: int = 30
REL_VOL_PERIOD: int = 30
MOMENTUM_PERIOD: int = 12
EMA_SLOPE_PERIOD: int = 5
SUPERTREND_ATR_PERIOD: int = 10
SUPERTREND_MULTIPLIER: float = 3.0
CANDLE_AVG_VOLUME_PERIOD: int = 15
CMF_PERIOD: int = 20
ASK_WALL_THRESHOLD_USDT: float = 20000.0
DIVERGENCE_LOOKBACK: int = 25

# --- إعدادات الفلاتر المتقدمة وإدارة الصفقات ---
ORDER_BOOK_DEPTH_LIMIT: int = 100
ORDER_BOOK_ANALYSIS_RANGE_PCT: float = 0.005
USE_ATR_TRAILING_STOP: bool = True
ATR_TS_PERIOD: int = 14
ATR_TS_MULTIPLIER: float = 2.2

# --- [إضافة] إعدادات نظام الأخبار ---
ENABLE_NEWS_PAUSE_SYSTEM: bool = True
NEWS_PAUSE_BEFORE_EVENT_MINUTES: int = 15
NEWS_RESUME_AFTER_EVENT_MINUTES: int = 15
HIGH_IMPACT_NEWS_KEYWORDS: List[str] = [
    "interest rate", "cpi", "inflation", "gdp", "non-farm payroll", "unemployment",
    "fomc", "ecb", "boe", "boj", "federal reserve", "central bank",
    "معدل الفائدة", "التضخم", "المركزي", "الفيدرالي", "الوظائف غير الزراعية", "الناتج المحلي الإجمالي"
]

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

# --- [إضافة] كاش الأخبار ---
economic_news_cache = deque(maxlen=20)
news_cache_lock = Lock()


# --- قاموس أسباب الرفض باللغة العربية ---
REJECTION_REASONS_AR = {
    "Bot Paused for News": "البوت متوقف مؤقتاً بسبب الأخبار",
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
    "Large Ask Wall Ahead": "يوجد جدار بيع كبير في الأمام",
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
    df_calc['ema_50'] = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    df_calc['ema_100'] = df_calc['close'].ewm(span=100, adjust=False).mean()
    df_calc['ema_120'] = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
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
    
    typical_price = (df_calc['high'] + df_calc['low'] + df_calc['close']) / 3
    df_calc['vwap'] = (typical_price * df_calc['volume']).cumsum() / df_calc['volume'].cumsum()
    df_calc['obv'] = (np.sign(df_calc['close'].diff()) * df_calc['volume']).fillna(0).cumsum()
    mfv = ((df_calc['close'] - df_calc['low']) - (df_calc['high'] - df_calc['close'])) / (df_calc['high'] - df_calc['low']).replace(0, 1e-9) * df_calc['volume']
    df_calc['cmf'] = mfv.rolling(CMF_PERIOD).sum() / df_calc['volume'].rolling(CMF_PERIOD).sum()

    # Other feature calculations would go here...
    # For brevity, I'm assuming the rest of the feature calculation functions are defined elsewhere
    # e.g., calculate_advanced_momentum_features, calculate_market_microstructure_features, etc.

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

# ---------------------- منطق التداول والفلاتر (مختصر) ----------------------
# All strategy and filter functions (check_bb_stoch_strategy, etc.) are assumed to be here as in the original file.
# For brevity, they are not repeated.

# ---------------------- دوال إدارة الصفقات (مختصر) ----------------------
# All trade management functions (adjust_quantity_to_lot_size, etc.) are assumed to be here as in the original file.
# For brevity, they are not repeated.

# --- [إضافة] دوال نظام الأخبار ---
def fetch_economic_news() -> List[Dict]:
    """Fetches economic news from NewsData.io API."""
    if not NEWSDATA_API_KEY:
        logger.warning("[الأخبار] مفتاح API لـ NewsData.io غير موجود.")
        return []
    
    url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&language=en&category=business,economics"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "success":
            articles = data.get("results", [])
            logger.info(f"[الأخبار] تم جلب {len(articles)} خبر اقتصادي جديد.")
            
            # Format and cache the articles
            formatted_articles = []
            for article in articles:
                # NewsData.io returns pubDate in 'YYYY-MM-DD HH:MM:SS' format, assuming UTC
                pub_date_str = article.get('pubDate')
                if not pub_date_str: continue
                
                try:
                    # Parse the string and make it timezone-aware (UTC)
                    pub_date_utc = datetime.strptime(pub_date_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                    
                    formatted_articles.append({
                        "title": article.get("title", "N/A"),
                        "pubDate": pub_date_utc,
                        "link": article.get("link", "#"),
                        "source": article.get("source_id", "N/A")
                    })
                except ValueError:
                    logger.warning(f"[الأخبار] تنسيق تاريخ غير صالح: {pub_date_str}")
                    continue

            return formatted_articles
        else:
            logger.error(f"[الأخبار] فشل جلب الأخبار، الحالة: {data.get('status')}, رسالة: {data.get('message')}")
            return []
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [الأخبار] خطأ في طلب API للأخبار: {e}")
        return []

def news_monitoring_loop():
    """Monitors for high-impact news and pauses the bot if necessary."""
    global is_bot_paused_by_news, news_pause_reason
    logger.info("✅ [مراقب الأخبار] بدء حلقة مراقبة الأخبار الاقتصادية...")
    
    pause_until_time = None

    while True:
        try:
            if not ENABLE_NEWS_PAUSE_SYSTEM:
                time.sleep(300) # Sleep for 5 minutes if disabled
                continue

            now_utc = datetime.now(timezone.utc)

            # Check if we need to resume the bot
            if is_bot_paused_by_news and pause_until_time and now_utc >= pause_until_time:
                with bot_pause_lock:
                    is_bot_paused_by_news = False
                    news_pause_reason = ""
                pause_until_time = None
                log_and_notify("info", "✅ استئناف التداول بعد انتهاء فترة تأثير الأخبار.", "NEWS_RESUME")
                send_telegram_message("✅ *استئناف التداول* | عادت عمليات البوت إلى طبيعتها.")

            # Fetch news only if not currently in a pause period
            if not is_bot_paused_by_news:
                articles = fetch_economic_news()
                
                # Update the global cache for the web UI
                with news_cache_lock:
                    economic_news_cache.clear()
                    for article in articles[:20]: # Add latest 20 to cache
                        # Convert datetime to string for JSON serialization
                        article_copy = article.copy()
                        article_copy['pubDate'] = article_copy['pubDate'].isoformat()
                        economic_news_cache.append(article_copy)

                for article in articles:
                    article_title_lower = article['title'].lower()
                    # Check if the article is high-impact
                    if any(keyword.lower() in article_title_lower for keyword in HIGH_IMPACT_NEWS_KEYWORDS):
                        event_time = article['pubDate']
                        time_to_event = event_time - now_utc
                        
                        # Check if the event is in the near future
                        if timedelta(minutes=0) < time_to_event <= timedelta(minutes=NEWS_PAUSE_BEFORE_EVENT_MINUTES + 5):
                            with bot_pause_lock:
                                if not is_bot_paused_by_news: # Pause only if not already paused
                                    is_bot_paused_by_news = True
                                    news_pause_reason = f"خبر قادم: {article['title']}"
                                    pause_start_time = event_time - timedelta(minutes=NEWS_PAUSE_BEFORE_EVENT_MINUTES)
                                    pause_until_time = event_time + timedelta(minutes=NEWS_RESUME_AFTER_EVENT_MINUTES)
                                    
                                    pause_duration = (pause_until_time - pause_start_time).total_seconds() / 60
                                    
                                    log_and_notify("warning", f"🚨 إيقاف البوت بسبب خبر اقتصادي قادم: {article['title']}. سيتم الإيقاف حتى {pause_until_time.strftime('%Y-%m-%d %H:%M:%S')} UTC.", "NEWS_PAUSE")
                                    send_telegram_message(
                                        f"🛑 *إيقاف مؤقت للتداول* 🛑\n\n"
                                        f"سيتم إيقاف البوت لمدة *{int(pause_duration)} دقيقة* بسبب الخبر التالي:\n\n"
                                        f"📰 *{article['title']}*\n"
                                        f"⏰ موعد الصدور: `{event_time.strftime('%Y-%m-%d %H:%M')}` UTC"
                                    )
                                    # Break after finding the first upcoming event to handle
                                    break 
            
            # Sleep for a while before the next check
            time.sleep(120) # Check every 2 minutes

        except Exception as e:
            logger.error(f"❌ [مراقب الأخبار] خطأ في حلقة المراقبة: {e}", exc_info=True)
            time.sleep(300) # Sleep longer on error

# ---------------------- دوال النظام الأساسية ----------------------
def determine_market_state_enhanced():
    global current_market_state, last_market_state_check
    if time.time() - last_market_state_check < 180: return
    logger.info("🧠 [حالة السوق] جاري تحديث حالة السوق العامة...")
    # ... (rest of the function is the same)
    # ...

# ---------------------- واجهة الويب (Flask) ----------------------
app = Flask(__name__)
CORS(app)

def get_dashboard_html():
    # This function now returns a more complex HTML string with the new elements
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم التداول V9.7.0 - نظام الأخبار</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
    <style>
        :root { --bg-main: #0D1117; --bg-card: #161B22; --border-color: #30363D; --text-primary: #E6EDF3; --text-secondary: #848D97; --accent-blue: #58A6FF; --accent-green: #3FB950; --accent-red: #F85149; --accent-yellow: #D29922; --accent-purple: #A371F7;}
        body { font-family: 'Tajawal', sans-serif; background-color: var(--bg-main); color: var(--text-primary); }
        .card { background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 0.5rem; }
        .tab-btn.active { border-bottom-color: var(--accent-blue); }
        input:checked + .toggle-bg { background-color: var(--accent-green); }
        .news-pause-active { background-color: var(--accent-yellow); color: #0D1117; animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
    </style>
</head>
<body class="p-4 md:p-6">
    <div class="container mx-auto max-w-screen-2xl">
        <header class="mb-6 flex flex-wrap justify-between items-center gap-4">
            <h1 class="text-2xl md:text-3xl font-extrabold"><span class="text-accent-blue">لوحة تحكم</span><span class="text-text-secondary font-medium"> V9.7.0</span></h1>
            <div id="news-pause-indicator" class="hidden font-bold px-4 py-2 rounded-lg text-center"></div>
        </header>
        <section class="mb-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">حالة السوق</h3><div id="overall-regime" class="text-2xl font-bold text-center">...</div></div>
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">الجلسات النشطة</h3><div id="active-sessions-list" class="flex flex-wrap gap-2 items-center justify-center pt-2">...</div></div>
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">الصفقات المفتوحة</h3><div id="open-trades-count" class="text-2xl font-bold text-center">...</div></div>
            <div class="card p-4 flex flex-col justify-center items-center"><h3 class="font-bold text-lg text-text-secondary mb-2">التداول الحقيقي</h3><div class="flex items-center space-x-3 space-x-reverse"><span id="trading-status-text" class="font-bold text-lg"></span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="trading-toggle" class="sr-only" onchange="toggleTrading()"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div><div class="mt-2 text-xs text-text-secondary">رصيد USDT: <span id="usdt-balance" class="font-mono">...</span></div></div>
        </section>
        <div class="mb-4 border-b border-border-color"><nav class="flex space-x-6 space-x-reverse -mb-px">
            <button onclick="showTab('signals', this)" class="tab-btn active text-white py-3 px-1 font-semibold">الصفقات</button>
            <button onclick="showTab('stats', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الإحصائيات</button>
            <button onclick="showTab('news', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الأخبار الاقتصادية</button>
            <button onclick="showTab('settings', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الإعدادات</button>
            <button onclick="showTab('notifications', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الإشعارات</button>
            <button onclick="showTab('rejections', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الصفقات المرفوضة</button>
        </nav></div>
        <main>
            <div id="signals-tab" class="tab-content">...</div>
            <div id="stats-tab" class="tab-content hidden">...</div>
            <div id="news-tab" class="tab-content hidden">
                <div id="news-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-3">
                    <p class="text-text-secondary">جاري تحميل الأخبار...</p>
                </div>
            </div>
            <div id="settings-tab" class="tab-content hidden">...</div>
            <div id="notifications-tab" class="tab-content hidden">...</div>
            <div id="rejections-tab" class="tab-content hidden">...</div>
        </main>
    </div>
<script>
// The script part is expanded to handle the new features
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
        // Update general status (same as before)
        document.getElementById('overall-regime').textContent = (data.market_state?.overall_regime || 'UNCERTAIN').replace(/_/g, ' ');
        document.getElementById('open-trades-count').textContent = `${data.open_trades_count} / ${data.max_open_trades}`;
        // ... other status updates
        
        // Update News Pause Indicator
        const newsIndicator = document.getElementById('news-pause-indicator');
        if (data.is_bot_paused_by_news) {
            newsIndicator.textContent = `🛑 متوقف بسبب الأخبار: ${data.news_pause_reason}`;
            newsIndicator.classList.remove('hidden');
            newsIndicator.classList.add('news-pause-active');
        } else {
            newsIndicator.classList.add('hidden');
            newsIndicator.classList.remove('news-pause-active');
        }

        const tradeToggle = document.getElementById('trading-toggle');
        tradeToggle.checked = data.is_trading_enabled;
        // ... rest of the function
    });
}

function updateEconomicNews() {
    fetchData('/api/economic_news').then(data => {
        if (!data || data.length === 0) {
            document.getElementById('news-list').innerHTML = '<p class="text-text-secondary">لا توجد أخبار حالياً.</p>';
            return;
        }
        const newsList = document.getElementById('news-list');
        newsList.innerHTML = data.map(n => `
            <div class="p-3 border-b border-border-color/50">
                <p class="font-bold">${n.title}</p>
                <div class="flex justify-between items-center mt-1">
                    <span class="text-xs text-text-secondary font-mono">${new Date(n.pubDate).toLocaleString('ar-EG', { timeZone: 'UTC' })} UTC</span>
                    <a href="${n.link}" target="_blank" class="text-xs text-accent-blue hover:underline">المصدر: ${n.source}</a>
                </div>
            </div>
        `).join('');
    });
}

function toggleTrading() { fetch('/api/trading/toggle', { method: 'POST' }).then(() => updateMarketStatus()); }

document.addEventListener('DOMContentLoaded', () => {
    // Initial data fetch for all tabs
    updateMarketStatus();
    updateEconomicNews();
    // ... other initial fetches (signals, stats, etc.)

    // Set intervals
    setInterval(updateMarketStatus, 5000);
    setInterval(updateEconomicNews, 60000); // Update news every minute
    // ... other intervals
});
</script>
</body></html>
"""

@app.route('/')
def home(): 
    # The HTML content is now very large, so it's better to keep it clean here
    # and assume the get_dashboard_html() function provides the full string.
    # For this example, I'll return a simplified version of the logic.
    html_content = get_dashboard_html() # This function should contain the full HTML
    # To keep this snippet clean, I'll simulate filling the tabs.
    # In a real app, you'd use a templating engine like Jinja2.
    # The actual content is filled by JavaScript on the client-side.
    return render_template_string(html_content)

@app.route('/api/market_status')
def get_market_status():
    with market_state_lock: state_copy = dict(current_market_state)
    with trading_status_lock: is_enabled = is_trading_enabled
    with bot_pause_lock: 
        is_paused = is_bot_paused_by_news
        reason = news_pause_reason
    
    # ... (rest of the function is the same, fetching balance, settings, etc.)
    
    with signal_cache_lock: open_trades = len(open_signals_cache)

    return jsonify({
        "market_state": state_copy, 
        "is_trading_enabled": is_enabled,
        "open_trades_count": open_trades, 
        "max_open_trades": MAX_OPEN_TRADES,
        "is_bot_paused_by_news": is_paused, # New status field
        "news_pause_reason": reason, # New status field
        # ... (rest of the JSON data)
    })

@app.route('/api/economic_news')
def get_economic_news():
    """New endpoint to serve cached economic news."""
    with news_cache_lock:
        return jsonify(list(economic_news_cache))

# ... (All other API endpoints like /api/stats, /api/signals, etc., remain the same)

def main_loop_enhanced():
    logger.info("[الحلقة الرئيسية] انتظار اكتمال التهيئة...")
    time.sleep(15)
    if not validated_symbols_to_scan:
        log_and_notify("critical", "قائمة العملات للمسح فارغة.", "SYSTEM_ERROR")
        return
    log_and_notify("info", f"✅ بدء حلقة المسح لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")

    while True:
        try:
            # [إضافة] التحقق من حالة الإيقاف بسبب الأخبار
            with bot_pause_lock:
                if is_bot_paused_by_news:
                    if (time.time() % 60) < 5: # Log status every minute
                        logger.info(f"⏸️ البوت متوقف مؤقتاً. السبب: {news_pause_reason}")
                    time.sleep(5)
                    continue

            logger.info("🔄 [الحلقة الرئيسية] بدء دورة مسح جديدة...")
            # ... (The rest of the main loop logic remains the same)
            # ...
            
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

def trade_management_loop():
    logger.info("✅ [مدير الصفقات] بدء حلقة إدارة الصفقات...")
    while True:
        try:
            # The logic for managing open trades remains the same
            time.sleep(2)
        except Exception as e:
            logger.error(f"❌ [مدير الصفقات] خطأ في حلقة الإدارة: {e}", exc_info=True)
            time.sleep(10)

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
        
        # Start all background threads
        Thread(target=main_loop_enhanced, daemon=True).start()
        Thread(target=price_update_loop, daemon=True).start()
        Thread(target=trade_management_loop, daemon=True).start()
        Thread(target=news_monitoring_loop, daemon=True).start() # [إضافة] بدء حلقة الأخبار

        logger.info("✅ [خدمات البوت] تم بدء جميع الخدمات الخلفية بنجاح.")
        send_telegram_message("✅ *البوت قيد التشغيل الآن (نسخة V9.7.0 مع نظام الأخبار)*")
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حرج أثناء التهيئة: {e}", "SYSTEM"); exit(1)

# ---------------------- نقطة الدخول ----------------------
if __name__ == "__main__":
    logger.info("🚀 إطلاق بوت التداول ولوحة التحكم (V9.7.0) 🚀")
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
