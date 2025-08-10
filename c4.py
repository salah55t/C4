# ملف c4.py - نسخة V9.7.2 (نسخة كاملة ومصححة)
# --- التغييرات الرئيسية (V9.7.2):
# 1. [إصلاح] تقديم الكود بالكامل دون أي اختصارات أو أجزاء مفقودة.
# 2. [مدمج] إصلاح خطأ فئة الأخبار غير المدعومة في NewsData.io API.
# 3. [مدمج] نظام مراقبة الأخبار الاقتصادية والإيقاف التلقائي.

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
logger = logging.getLogger('CryptoBotV9.7.2')

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
    NEWSDATA_API_KEY: str = config('NEWSDATA_API_KEY', default='pub_2c2d39760da740178b67ee1befd97206')

except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
    exit(1)

# --- متغيرات عامة وإعدادات البوت ---
is_trading_enabled: bool = False
trading_status_lock = Lock()

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
USE_ORDER_BOOK_FILTER: bool = True
order_book_filter_enable_lock = Lock()
USE_HTF_CONFIRMATION_FILTER: bool = True
htf_confirmation_lock = Lock()
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

SCALPING_STRATEGIES = ["Pullback_MACD", "BB_Squeeze_Breakout", "QQE_SSL_Explosion"]

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

USE_DYNAMIC_JOURNEY = True

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

ORDER_BOOK_DEPTH_LIMIT: int = 100
ORDER_BOOK_ANALYSIS_RANGE_PCT: float = 0.005
USE_ATR_TRAILING_STOP: bool = True
ATR_TS_PERIOD: int = 14
ATR_TS_MULTIPLIER: float = 2.2

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
    "Bullish Momentum Strategy Conditions Not Met": "شروط استراتيجية الزخم الصعودي لم تتحقق",
    "BB_Stoch Strategy Conditions Not Met": "شروط استراتيجية BB+Stoch لم تتحقق",
    "MACD_EMA Strategy Conditions Not Met": "شروط استراتيجية MACD+EMA لم تتحقق",
    "QQE_SSL Strategy Conditions Not Met": "شروط استراتيجية QQE+SSL لم تتحقق",
    "EMA_RSI Strategy Conditions Not Met": "شروط استراتيجية EMA+RSI لم تتحقق",
    "Pullback Strategy Conditions Not Met": "شروط استراتيجية Pullback لم تتحقق",
    "BB Squeeze Strategy Conditions Not Met": "شروط استراتيجية BB Squeeze لم تتحقق",
    "Smart Breakout Strategy Conditions Not Met": "شروط استراتيجية الاختراق الذكي لم تتحقق",
    "RSI Hidden Divergence Conditions Not Met": "شروط استراتيجية التباعد الخفي لم تتحقق",
    "VWAP Reversal Conditions Not Met": "شروط استراتيجية الارتداد من VWAP لم تتحقق",
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

def check_bb_stoch_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 2: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    price_touch_bb = last['low'] <= (last['bb_lower'] * 1.001)
    stoch_cross_up = prev['stoch_rsi_k'] < prev['stoch_rsi_d'] and last['stoch_rsi_k'] > last['stoch_rsi_d']
    oversold_area = last['stoch_rsi_k'] < 30 and last['stoch_rsi_d'] < 30
    if price_touch_bb and stoch_cross_up and oversold_area:
        logger.info(f"  -> [{df.name}] ✅ إشارة استراتيجية BB+Stoch.")
        return True
    return False

def check_macd_ema_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 2: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    macd_cross_up = prev['macd'] < prev['macd_signal'] and last['macd'] > last['macd_signal']
    price_above_ema = last['close'] > last['ema_50']
    if macd_cross_up and price_above_ema:
        logger.info(f"  -> [{df.name}] ✅ إشارة استراتيجية MACD+EMA.")
        return True
    return False

def check_qqe_ssl_strategy_approx(df: pd.DataFrame) -> bool:
    if len(df) < 2: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    ssl_flipped_green = prev['supertrend_direction'] == -1 and last['supertrend_direction'] == 1
    wae_explosion = last['relative_volume'] > 1.2
    qqe_bullish = last['rsi'] > 55
    if ssl_flipped_green and wae_explosion and qqe_bullish:
        logger.info(f"  -> [{df.name}] ✅ إشارة استراتيجية QQE+SSL (تقريبية).")
        return True
    return False

def check_ema_rsi_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 2: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    ema_cross_up = prev['ema_9'] < prev['ema_21'] and last['ema_9'] > last['ema_21']
    rsi_strong = last['rsi'] > 50
    trend_filter = last['close'] > last['ema_50']
    if ema_cross_up and rsi_strong and trend_filter:
        logger.info(f"  -> [{df.name}] ✅ إشارة استراتيجية EMA+RSI Cross.")
        return True
    return False

def check_pullback_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 2: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    uptrend_confirmed = last['close'] > last['ema_50'] and last['ema_50'] > last['ema_100']
    macd_cross_up = prev['macd'] < prev['macd_signal'] and last['macd'] > last['macd_signal']
    if uptrend_confirmed and macd_cross_up:
        logger.info(f"  -> [{df.name}] ✅ إشارة استراتيجية Pullback MACD.")
        return True
    return False

def check_bb_squeeze_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 100: return False
    last = df.iloc[-1]
    is_squeeze = last['bb_width'] < df['bb_width'].rolling(100).quantile(0.25).iloc[-1]
    breakout = last['close'] > last['bb_upper']
    volume_confirmed = last['relative_volume'] > 1.2
    if is_squeeze and breakout and volume_confirmed:
        logger.info(f"  -> [{df.name}] ✅ إشارة استراتيجية BB Squeeze Breakout.")
        return True
    return False

def check_bullish_momentum_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 50:
        return False
    last = df.iloc[-1]
    price_above_sma50 = last['close'] > last['sma_50']
    strong_trend = last['adx'] > 25
    bullish_direction = last['plus_di'] > last['minus_di']
    rsi_is_bullish = 50 < last['rsi'] < 75
    if len(df) < 6:
        return False
    recent_highs = df['high'].iloc[-6:-1]
    recent_lows = df['low'].iloc[-6:-1]
    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return False
    is_higher_highs = all(recent_highs.iloc[i] > recent_highs.iloc[i - 1] for i in range(1, len(recent_highs)))
    is_higher_lows = all(recent_lows.iloc[i] > recent_lows.iloc[i - 1] for i in range(1, len(recent_lows)))
    price_momentum_confirmed = is_higher_highs and is_higher_lows
    if all([price_above_sma50, strong_trend, bullish_direction, rsi_is_bullish, price_momentum_confirmed]):
        logger.info(f"  -> [{df.name}] ✅ إشارة استراتيجية زخم صعودي (مع تأكيد حركة السعر).")
        return True
    return False

def check_smart_breakout_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 2 or not all(k in df.columns for k in ['vwap', 'cmf', 'obv', 'relative_volume']):
        return False
    last = df.iloc[-1]
    prev = df.iloc[-2]
    breakout_occurred = last['close'] > last['vwap'] and prev['close'] <= prev['vwap']
    momentum_confirmed = last['cmf'] > 0.05 and last['obv'] > prev['obv']
    volume_confirmed = last['relative_volume'] > 1.5
    if breakout_occurred and momentum_confirmed and volume_confirmed:
        logger.info(f"  -> [{df.name}] ✅ إشارة استراتيجية الاختراق الذكي (Smart Breakout).")
        return True
    return False

def check_rsi_hidden_divergence_strategy(df: pd.DataFrame) -> bool:
    if len(df) < DIVERGENCE_LOOKBACK + 5 or not all(k in df.columns for k in ['rsi', 'low', 'ema_50']):
        return False
    last = df.iloc[-1]
    if last['close'] < last['ema_50']:
        return False
    search_df = df.iloc[-DIVERGENCE_LOOKBACK:-1]
    price_low_idx = search_df['low'].idxmin()
    price_low_val = search_df.loc[price_low_idx]['low']
    rsi_low_idx = search_df['rsi'].idxmin()
    rsi_low_val = search_df.loc[rsi_low_idx]['rsi']
    if (last['low'] < price_low_val and 
        last['rsi'] > rsi_low_val and 
        last['rsi'] > 30 and rsi_low_val > 30):
        logger.info(f"  -> [{df.name}] ✅ إشارة تباعد خفي: Price Low ({last['low']:.4f} < {price_low_val:.4f}), RSI Low ({last['rsi']:.2f} > {rsi_low_val:.2f})")
        return True
    return False

def check_vwap_reversal_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 2 or not all(k in df.columns for k in ['vwap', 'relative_volume', 'macd_histogram']):
        return False
    last = df.iloc[-1]
    prev = df.iloc[-2]
    vwap_touch = abs(last['close'] - last['vwap']) / last['vwap'] < 0.002
    high_volume = last['relative_volume'] > 1.5
    macd_reversal = last['macd_histogram'] > prev['macd_histogram'] and prev['macd_histogram'] < 0
    if vwap_touch and high_volume and macd_reversal:
        logger.info(f"  -> [{df.name}] ✅ إشارة ارتداد VWAP: Volume={last['relative_volume']:.2f}x, MACD Hist Reversal.")
        return True
    return False

def has_large_ask_wall_nearby(symbol: str, price: float) -> bool:
    if not client: return True
    try:
        order_book = client.get_order_book(symbol=symbol, limit=100)
        asks = pd.DataFrame(order_book['asks'], columns=['price', 'qty'], dtype=float)
        price_range_upper = price * (1 + 0.005) 
        relevant_asks = asks[asks['price'].between(price, price_range_upper)]
        if relevant_asks.empty:
            return False
        relevant_asks['value_usdt'] = relevant_asks['price'] * relevant_asks['qty']
        if relevant_asks['value_usdt'].max() > ASK_WALL_THRESHOLD_USDT:
            wall_price = relevant_asks.loc[relevant_asks['value_usdt'].idxmax()]
            logger.warning(f"  -> [{symbol}] 🧱 تم اكتشاف جدار بيع كبير: {wall_price['qty']:.2f} @ {wall_price['price']:.4f} (قيمته ${wall_price['value_usdt']:.0f})")
            return True
        return False
    except Exception as e:
        logger.error(f"❌ [{symbol}] خطأ في فلتر جدار البيع: {e}")
        return True

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

def passes_short_term_momentum_filter(symbol: str, df: pd.DataFrame) -> bool:
    try:
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
        is_valid = ((is_squeeze or close_to_bands) and volume_spike and (macd_momentum or rsi_momentum))
        logger.info(f"  -> [فلتر الزخم] {symbol}: Squeeze/CloseToBand={(is_squeeze or close_to_bands)}, Volume={volume_spike}, MACD/RSI={(macd_momentum or rsi_momentum)} → {is_valid}")
        return is_valid
    except Exception as e:
        logger.error(f"❌ [Short-Term Filter] خطأ في {symbol}: {e}", exc_info=True)
        return False

class EnhancedTradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ml_model, self.scaler, self.feature_names = None, None, None

    def load_model(self) -> bool:
        model_name = f"{BASE_ML_MODEL_NAME}_{self.symbol}"
        if model_name in ml_models_cache:
            model_bundle = ml_models_cache[model_name]
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir_path = os.path.join(script_dir, MODEL_FOLDER)
            model_path = os.path.join(model_dir_path, f"{model_name}.pkl")
            if not os.path.exists(model_path):
                return False
            try:
                with open(model_path, 'rb') as f:
                    model_bundle = pickle.load(f)
                ml_models_cache[model_name] = model_bundle
            except Exception as e:
                logger.error(f"❌ [نموذج ML] خطأ في تحميل النموذج لـ {self.symbol}: {e}")
                return False
        if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
            self.ml_model = model_bundle['model']
            self.scaler = model_bundle['scaler']
            self.feature_names = model_bundle['feature_names']
            return True
        else:
            logger.error(f"  -> [{self.symbol}] 🛑 ملف نموذج ML غير مكتمل.")
            return False

    def get_features_for_model(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.feature_names is None: return None
        try:
            df_featured = calculate_all_features(df_15m, btc_df)
            delta_4h = df_4h['close'].diff()
            gain_4h = delta_4h.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
            loss_4h = -delta_4h.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
            df_4h['rsi_4h'] = 100 - (100 / (1 + (gain_4h / loss_4h.replace(0, 1e-9))))
            ema_fast_4h = df_4h['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
            df_4h['price_vs_ema50_4h'] = (df_4h['close'] / ema_fast_4h) - 1
            mtf_features = df_4h[['rsi_4h', 'price_vs_ema50_4h']]
            df_featured = df_featured.join(mtf_features)
            df_featured[['rsi_4h', 'price_vs_ema50_4h']] = df_featured[['rsi_4h', 'price_vs_ema50_4h']].fillna(method='ffill')
            for col in self.feature_names:
                if col not in df_featured.columns:
                    df_featured[col] = 0.0
            df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_featured.dropna(subset=self.feature_names)
        except Exception as e:
            logger.error(f"❌ [{self.symbol}] فشل هندسة الميزات لنموذج ML: {e}", exc_info=True)
            return None

    def generate_prediction_result(self, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]) or df_features.empty:
            return None
        try:
            last_row_ordered_df = df_features.iloc[[-1]][self.feature_names]
            features_scaled = self.scaler.transform(last_row_ordered_df)
            prediction = self.ml_model.predict(features_scaled)[0]
            prediction_proba = self.ml_model.predict_proba(features_scaled)
            confidence = float(np.max(prediction_proba[0]))
            return {'prediction': int(prediction), 'confidence': confidence}
        except Exception as e:
            logger.warning(f"⚠️ [{self.symbol}] خطأ في توليد تنبؤ نموذج ML: {e}", exc_info=True)
            return None

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
    return (c1['close'] < c1['open'] and c1_body > c1.atr * 0.7 and
            abs(c2['open'] - c2['close']) < c1_body * 0.3 and
            c2['close'] < c1['close'] and c2['open'] < c1['close'] and
            c3['close'] > c3['open'] and
            c3['close'] > (c1['open'] + c1['close']) / 2)

def is_three_white_soldiers(c1: pd.Series, c2: pd.Series, c3: pd.Series) -> bool:
    return (c1['close'] > c1['open'] and c2['close'] > c2['open'] and c3['close'] > c3['open'] and
            c2['open'] > c1['open'] and c2['open'] < c1['close'] and c2['close'] > c1['close'] and
            c3['open'] > c2['open'] and c3['open'] < c2['close'] and c3['close'] > c2['close'])

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

def passes_final_order_book_check(symbol: str, entry_price: float) -> bool:
    if not client:
        log_rejection(symbol, "Order Book Fetch Failed", {"error": "Client not initialized"})
        return False
    try:
        with order_book_ratio_lock:
             current_ratio_threshold = ORDER_BOOK_MIN_BID_ASK_RATIO
        order_book = client.get_order_book(symbol=symbol, limit=ORDER_BOOK_DEPTH_LIMIT)
        bids = pd.DataFrame(order_book['bids'], columns=['price', 'qty'], dtype=float)
        asks = pd.DataFrame(order_book['asks'], columns=['price', 'qty'], dtype=float)
        price_range_upper = entry_price * (1 + ORDER_BOOK_ANALYSIS_RANGE_PCT)
        price_range_lower = entry_price * (1 - ORDER_BOOK_ANALYSIS_RANGE_PCT)
        relevant_bids_vol = bids[bids['price'].between(price_range_lower, entry_price)]['qty'].sum()
        relevant_asks_vol = asks[asks['price'].between(entry_price, price_range_upper)]['qty'].sum()
        if relevant_asks_vol == 0: return True
        bid_ask_ratio = relevant_bids_vol / relevant_asks_vol
        if bid_ask_ratio >= current_ratio_threshold:
            return True
        else:
            log_rejection(symbol, "Order Book Filter Failed", {"ratio": f"{bid_ask_ratio:.2f}", "required": f"{current_ratio_threshold}"})
            return False
    except Exception as e:
        log_rejection(symbol, "Order Book Fetch Failed", {"error": str(e)})
        return False

def find_sr_levels(df: pd.DataFrame, lookback: int = 50, min_bounces: int = 2) -> Dict[str, Optional[float]]:
    if len(df) < lookback:
        return {'support': None, 'resistance': None}
    df_slice = df.iloc[-lookback:]
    resistance_candidates = df_slice[df_slice['high'] == df_slice['high'].rolling(5, center=True).max()]['high']
    support_candidates = df_slice[df_slice['low'] == df_slice['low'].rolling(5, center=True).min()]['low']
    if resistance_candidates.empty or support_candidates.empty:
        return {'support': None, 'resistance': None}
    current_price = df_slice['close'].iloc[-1]
    next_resistance = resistance_candidates[resistance_candidates > current_price]
    closest_resistance = next_resistance.min() if not next_resistance.empty else None
    next_support = support_candidates[support_candidates < current_price]
    closest_support = next_support.max() if not next_support.empty else None
    return {'support': closest_support, 'resistance': closest_resistance}

def calculate_tp_sl(symbol: str, entry_price: float, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    try:
        if df.empty or len(df) < 50:
            log_rejection(symbol, "Insufficient data for TP/SL calculation")
            return None
        sr = find_sr_levels(df, lookback=50, min_bounces=2)
        resistance = sr['resistance']
        support    = sr['support']
        potential_profit_pct = 0
        if resistance is not None and resistance > entry_price:
            potential_profit_pct = ((resistance - entry_price) / entry_price) * 100
        with_min_profit = MIN_PROFIT_PERCENT
        if resistance is None or support is None or potential_profit_pct < with_min_profit:
            new_target_price = entry_price * (1 + (with_min_profit + 0.2) / 100)
            new_stop_loss = entry_price * (1 - 1.5 / 100)
            rr_ratio = ((new_target_price - entry_price) / (entry_price - new_stop_loss)) if (entry_price - new_stop_loss) > 0 else 0
            return {
                'target_price': round(new_target_price, 6), 'stop_loss': round(new_stop_loss, 6),
                'source': 'FIXED_PERCENTAGE', 'rr_ratio': round(rr_ratio, 2)
            }
        if support >= entry_price: support = entry_price * 0.98
        risk_pct = ((entry_price - support) / entry_price) * 100
        if risk_pct < 0.3: support = entry_price * (1 - 0.003)
        rr_ratio = ((resistance - entry_price) / (entry_price - support)) if (entry_price - support) > 0 else 0
        return {
            'target_price': round(resistance, 6), 'stop_loss': round(support, 6),
            'source': 'SR_LEVELS', 'rr_ratio': round(rr_ratio, 2)
        }
    except Exception as e:
        logger.error(f"❌ [{symbol}] خطأ في حساب TP/SL: {e}", exc_info=True)
        last_atr = df['atr'].iloc[-1] if 'atr' in df.columns and not df['atr'].empty else 0
        if last_atr > 0:
            target = entry_price + last_atr * 2.2
            stop = entry_price - last_atr * 1.5
            rr_ratio = (target - entry_price) / (entry_price - stop) if (entry_price - stop) > 0 else 0
            return {'target_price': target, 'stop_loss': stop, 'source': 'ATR_Fallback', 'rr_ratio': round(rr_ratio, 2)}
        return None

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

def verify_order_filled(symbol: str, order_id: str, timeout_seconds: int = 30) -> bool:
    if not client: return False
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            order_status = client.get_order(symbol=symbol, orderId=order_id)
            if order_status['status'] == 'FILLED': return True
            elif order_status['status'] in ['CANCELED', 'EXPIRED', 'REJECTED']: return False
            time.sleep(2)
        except Exception as e:
            logger.error(f"❌ [{symbol}] خطأ غير متوقع عند التحقق من الأمر {order_id}: {e}", exc_info=True)
            return False
    return False

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
            quantity_to_sell_str = signal_to_close.get('quantity')
            if quantity_to_sell_str and float(quantity_to_sell_str) > 0:
                logger.info(f"  -> [{symbol_to_close}] محاولة بيع الكمية المتبقية عند الإغلاق: {quantity_to_sell_str}")
                try:
                    sell_order = place_order(symbol_to_close, Client.SIDE_SELL, Decimal(str(quantity_to_sell_str)))
                    if not sell_order:
                        logger.warning(f"⚠️ [{symbol_to_close}] فشل أمر البيع عند الإغلاق (ربما تم البيع مسبقاً). سيتم إكمال عملية الإغلاق في قاعدة البيانات.")
                except Exception as e:
                    logger.error(f"❌ [{symbol_to_close}] خطأ أثناء إغلاق الجزء المتبقي من الصفقة: {e}", exc_info=True)
                    if 'insufficient' in str(e).lower() or 'min_notional' in str(e).lower():
                        logger.warning(f"⚠️ [{symbol_to_close}] الخطأ يشير إلى عدم وجود رصيد كافٍ، مما يؤكد أن الكمية بيعت. سيتم المتابعة بالإغلاق.")
                    else:
                        return False 
        
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

            reason_map = {
                'take_profit': '🎯 أخذ الربح', 'stop_loss': '🛑 وقف الخسارة', 'manual': '🖐️ إغلاق يدوي',
                'atr_trailing_stop': '🛡️ وقف خسارة متحرك', 'journey_completed': '🏁 اكتملت الرحلة',
                'take_profit_full_exit_on_small_size': '🎯 أخذ الربح (إغلاق كامل لصفقة صغيرة)'
            }
            emoji = "✅" if profit_percentage >= 0 else "🔻"
            trade_type = "حقيقية" if signal_to_close.get('is_real_trade') else "تجريبية"
            telegram_message = (
                f"{emoji} *إغلاق صفقة {trade_type}*\n\n"
                f"*العملة:* `{symbol_to_close}`\n*سبب الإغلاق:* {reason_map.get(reason, reason)}\n"
                f"*سعر الدخول:* `{entry_price:.4f}`\n*سعر الإغلاق:* `{closing_price:.4f}`\n"
                f"*الربح/الخسارة النهائي:* `{profit_percentage:.2f}%`"
            )
            send_telegram_message(telegram_message)
            return True
        except Exception as e:
            logger.error(f"❌ [قاعدة البيانات] فشل تحديث الصفقة المغلقة: {e}"); conn.rollback(); return False

def insert_signal_into_db(signal_data: Dict) -> Optional[Dict]:
    if not check_db_connection() or not conn: return None
    try:
        with conn.cursor() as cur:
            entry_price = float(signal_data['entry_price'])
            target_price = float(signal_data['target_price'])
            stop_loss = float(signal_data['stop_loss'])
            quantity = float(signal_data['quantity']) if signal_data.get('quantity') is not None else None
            rr_ratio = float(signal_data.get('rr_ratio', 0.0))
            journey_state = None
            if USE_DYNAMIC_JOURNEY:
                journey_state = {"targets_hit": 0, "is_complete": False, "partial_exit_done": False}
            signal_details_json = json.dumps(signal_data['signal_details'], cls=NpEncoder)
            journey_state_json = json.dumps(journey_state, cls=NpEncoder) if journey_state else None
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, is_real_trade, quantity, original_quantity, order_id, current_peak_price, journey_state, rr_ratio)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *;
            """, (
                signal_data['symbol'], entry_price, target_price, stop_loss,
                signal_data['strategy_name'], signal_details_json, signal_data.get('is_real_trade', False),
                quantity, quantity, signal_data.get('order_id'), entry_price, journey_state_json, rr_ratio
            ))
            saved_signal = cur.fetchone()
            conn.commit()
            logger.info(f"💾 [{signal_data['symbol']}] تم حفظ الإشارة الجديدة في قاعدة البيانات.")
            trade_type = "حقيقية" if signal_data.get('is_real_trade') else "تجريبية"
            telegram_message = (
                f"💡 *توصية شراء {trade_type} جديدة*\n\n"
                f"*العملة:* `{signal_data['symbol']}`\n*الاستراتيجية:* `{signal_data['strategy_name'].replace('_', ' ')}`\n"
                f"*سعر الدخول:* `{entry_price:.4f}`\n*الهدف الأول:* `{target_price:.4f}`\n"
                f"*وقف الخسارة:* `{stop_loss:.4f}`\n*RR Ratio:* `{rr_ratio:.2f}`\n\n"
                f"Confidence: {signal_data['signal_details'].get('ML_Confidence', 'N/A')}"
            )
            send_telegram_message(telegram_message)
            return dict(saved_signal)
    except Exception as e:
        logger.error(f"❌ [قاعدة البيانات] فشل إدراج الإشارة: {e}", exc_info=True); conn.rollback(); return None

# --- دوال نظام الأخبار ---
def fetch_economic_news() -> List[Dict]:
    if not NEWSDATA_API_KEY:
        logger.warning("[الأخبار] مفتاح API لـ NewsData.io غير موجود.")
        return []
    
    valid_categories = "business,politics,world"
    url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_API_KEY}&language=en&category={valid_categories}"
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "success":
            articles = data.get("results", [])
            logger.info(f"[الأخبار] تم جلب {len(articles)} خبر.")
            formatted_articles = []
            for article in articles:
                pub_date_str = article.get('pubDate')
                if not pub_date_str: continue
                try:
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
            logger.error(f"[الأخبار] فشل جلب الأخبار، الحالة: {data.get('status')}, رسالة: {data.get('results', {}).get('message', 'خطأ غير معروف')}")
            return []
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [الأخبار] خطأ في طلب API للأخبار: {e}")
        return []

def news_monitoring_loop():
    global is_bot_paused_by_news, news_pause_reason
    logger.info("✅ [مراقب الأخبار] بدء حلقة مراقبة الأخبار الاقتصادية...")
    pause_until_time = None
    while True:
        try:
            if not ENABLE_NEWS_PAUSE_SYSTEM:
                time.sleep(300)
                continue
            now_utc = datetime.now(timezone.utc)
            if is_bot_paused_by_news and pause_until_time and now_utc >= pause_until_time:
                with bot_pause_lock:
                    is_bot_paused_by_news = False
                    news_pause_reason = ""
                pause_until_time = None
                log_and_notify("info", "✅ استئناف التداول بعد انتهاء فترة تأثير الأخبار.", "NEWS_RESUME")
                send_telegram_message("✅ *استئناف التداول* | عادت عمليات البوت إلى طبيعتها.")
            if not is_bot_paused_by_news:
                articles = fetch_economic_news()
                with news_cache_lock:
                    economic_news_cache.clear()
                    for article in articles[:20]:
                        article_copy = article.copy()
                        article_copy['pubDate'] = article_copy['pubDate'].isoformat()
                        economic_news_cache.append(article_copy)
                for article in articles:
                    article_title_lower = article['title'].lower()
                    if any(keyword.lower() in article_title_lower for keyword in HIGH_IMPACT_NEWS_KEYWORDS):
                        event_time = article['pubDate']
                        time_to_event = event_time - now_utc
                        if timedelta(minutes=0) < time_to_event <= timedelta(minutes=NEWS_PAUSE_BEFORE_EVENT_MINUTES + 5):
                            with bot_pause_lock:
                                if not is_bot_paused_by_news:
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
                                    break 
            time.sleep(120)
        except Exception as e:
            logger.error(f"❌ [مراقب الأخبار] خطأ في حلقة المراقبة: {e}", exc_info=True)
            time.sleep(300)

# ---------------------- واجهة الويب (Flask) ----------------------
app = Flask(__name__)
CORS(app)

def get_dashboard_html():
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم التداول V9.7.2</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
    <style>
        :root { --bg-main: #0D1117; --bg-card: #161B22; --border-color: #30363D; --text-primary: #E6EDF3; --text-secondary: #848D97; --accent-blue: #58A6FF; --accent-green: #3FB950; --accent-red: #F85149; --accent-yellow: #D29922; --accent-purple: #A371F7; --accent-teal: #39D3BB; --accent-orange: #F78166; --accent-pink: #DB61A2;}
        body { font-family: 'Tajawal', sans-serif; background-color: var(--bg-main); color: var(--text-primary); }
        .card { background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 0.5rem; }
        .tab-btn.active { border-bottom-color: var(--accent-blue); }
        input:checked + .toggle-bg { background-color: var(--accent-green); }
        .news-pause-active { background-color: var(--accent-yellow); color: #0D1117; animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
        .input-field { background-color: #0D1117; border: 1px solid var(--border-color); border-radius: 0.375rem; padding: 0.5rem 0.75rem; color: var(--text-primary); }
        .save-btn { background-color: var(--accent-blue); color: white; padding: 0.5rem 1rem; border-radius: 0.375rem; font-weight: bold; transition: background-color 0.2s; }
        .save-btn:hover { background-color: #4a91e2; }
    </style>
</head>
<body class="p-4 md:p-6">
    <div id="modal-overlay" class="fixed inset-0 bg-black bg-opacity-70 hidden items-center justify-center z-50">
        <div id="modal-content" class="card p-6 rounded-lg shadow-xl max-w-sm w-full">
            <h3 id="modal-title" class="text-xl font-bold mb-4"></h3>
            <p id="modal-body" class="text-text-secondary mb-6"></p>
            <div class="flex justify-end gap-3">
                <button id="modal-cancel" class="px-4 py-2 rounded-md bg-gray-600 hover:bg-gray-700">إلغاء</button>
                <button id="modal-confirm" class="px-4 py-2 rounded-md bg-red-600 hover:bg-red-700">تأكيد</button>
            </div>
        </div>
    </div>
    <div class="container mx-auto max-w-screen-2xl">
        <header class="mb-6 flex flex-wrap justify-between items-center gap-4">
            <h1 class="text-2xl md:text-3xl font-extrabold"><span class="text-accent-blue">لوحة تحكم</span><span class="text-text-secondary font-medium"> V9.7.2</span></h1>
            <div id="news-pause-indicator" class="hidden font-bold px-4 py-2 rounded-lg text-center"></div>
        </header>
        <section class="mb-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">حالة السوق</h3><div id="overall-regime" class="text-2xl font-bold text-center">...</div></div>
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">الجلسات النشطة</h3><div id="active-sessions-list" class="flex flex-wrap gap-2 items-center justify-center pt-2">...</div></div>
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">الصفقات المفتوحة</h3><div id="open-trades-count" class="text-2xl font-bold text-center">...</div></div>
            <div class="card p-4 flex flex-col justify-center items-center"><h3 class="font-bold text-lg text-text-secondary mb-2">التداول الحقيقي</h3><div class="flex items-center space-x-3 space-x-reverse"><span id="trading-status-text" class="font-bold text-lg"></span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="trading-toggle" class="sr-only" onchange="toggleTrading()"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div><div class="mt-2 text-xs text-text-secondary">رصيد USDT: <span id="usdt-balance" class="font-mono">...</span></div></div>
        </section>
        <div class="mb-4 border-b border-border-color"><nav class="flex space-x-6 space-x-reverse -mb-px overflow-x-auto">
            <button onclick="showTab('signals', this)" class="tab-btn active text-white py-3 px-1 font-semibold whitespace-nowrap">الصفقات</button>
            <button onclick="showTab('stats', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1 whitespace-nowrap">الإحصائيات</button>
            <button onclick="showTab('news', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1 whitespace-nowrap">الأخبار الاقتصادية</button>
            <button onclick="showTab('settings', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1 whitespace-nowrap">الإعدادات</button>
            <button onclick="showTab('notifications', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1 whitespace-nowrap">الإشعارات</button>
            <button onclick="showTab('rejections', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1 whitespace-nowrap">الصفقات المرفوضة</button>
        </nav></div>
        <main>
            <div id="signals-tab" class="tab-content"><div class="overflow-x-auto card p-0"><table class="min-w-full text-sm text-right"><thead class="border-b border-border-color bg-black/20"><tr><th class="p-4 font-semibold">العملة</th><th class="p-4 font-semibold">الربح/الخسارة</th><th class="p-4 font-semibold">الدخول/الحالي/الهدف</th><th class="p-4 font-semibold">إجراء</th></tr></thead><tbody id="signals-table"></tbody></table></div></div>
            <div id="stats-tab" class="tab-content hidden"><div id="stats-container" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"></div></div>
            <div id="news-tab" class="tab-content hidden"><div id="news-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-3"><p class="text-text-secondary">جاري تحميل الأخبار...</p></div></div>
            <div id="settings-tab" class="tab-content hidden"><div class="card p-6">
                <h4 class="text-lg font-bold mb-4 text-text-secondary">الإعدادات العامة</h4>
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    <div><label for="risk-percent" class="block text-sm font-medium text-text-secondary mb-1">نسبة المخاطرة للصفقة (%)</label><input type="number" id="risk-percent" name="risk_percent" step="0.1" class="input-field w-full"></div>
                    <div><label for="ob-ratio" class="block text-sm font-medium text-text-secondary mb-1">نسبة فلتر دفتر الطلبات</label><input type="number" id="ob-ratio" name="ob_ratio" step="0.1" class="input-field w-full"></div>
                    <div><label for="vol-multiplier" class="block text-sm font-medium text-text-secondary mb-1">مضاعف فلتر حجم التداول</label><input type="number" id="vol-multiplier" name="vol_multiplier" step="0.01" class="input-field w-full"></div>
                    <div><label for="min-profit" class="block text-sm font-medium text-text-secondary mb-1">أدنى ربح مستهدف (%)</label><input type="number" id="min-profit" name="min_profit" step="0.1" class="input-field w-full"></div>
                </div>
                <hr class="border-border-color my-6">
                <h4 class="text-lg font-bold mb-4 text-text-secondary">الفلاتر</h4>
                <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                    <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg"><span class="font-semibold">فلتر نمط الشموع</span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="candle-filter-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div>
                    <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg"><span class="font-semibold">فلتر حجم التداول</span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="volume-filter-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div>
                    <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg"><span class="font-semibold">فلتر دفتر الطلبات</span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="ob-filter-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div>
                    <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg"><span class="font-semibold">فلتر تأكيد الترند (HTF)</span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="htf-confirmation-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div>
                    <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg"><span class="font-semibold">فلتر الزخم قصير الأجل</span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="short-term-momentum-filter-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div>
                </div>
                <hr class="border-border-color my-6">
                <h4 class="text-lg font-bold mb-4 text-text-secondary">الاستراتيجيات</h4>
                <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mt-6">
                    <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg"><span class="font-semibold">التعلم الآلي (ML)</span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="ml-strategy-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div>
                    <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg"><span class="font-semibold">الاختراق الذكي</span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="smart-breakout-strategy-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div>
                    <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg"><span class="font-semibold">تباعد RSI الخفي</span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="rsi-divergence-strategy-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div>
                    <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg"><span class="font-semibold">ارتداد VWAP</span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="vwap-reversal-strategy-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div>
                    <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg"><span class="font-semibold">BB+Stoch</span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="bb-stoch-strategy-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div>
                    <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg"><span class="font-semibold">MACD+EMA</span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="macd-ema-strategy-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div>
                    <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg"><span class="font-semibold">QQE+SSL</span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="qqe-ssl-strategy-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div>
                    <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg"><span class="font-semibold">EMA+RSI Cross</span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="ema-rsi-strategy-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div>
                    <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg"><span class="font-semibold">Pullback MACD</span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="pullback-strategy-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div>
                    <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg"><span class="font-semibold">BB Squeeze</span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="bb-squeeze-strategy-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div>
                    <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg"><span class="font-semibold">زخم صعودي</span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="bullish-momentum-strategy-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div>
                </div>
                <div class="mt-8 text-left"><button onclick="saveSettings()" class="save-btn">حفظ الإعدادات</button></div>
                <div id="settings-feedback" class="mt-4 text-center"></div>
            </div></div>
            <div id="notifications-tab" class="tab-content hidden"><div id="notifications-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-2"></div></div>
            <div id="rejections-tab" class="tab-content hidden"><div id="rejections-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-2"></div></div>
        </main>
    </div>
<script>
let confirmCallback = null;
const modal = { overlay: document.getElementById('modal-overlay'), title: document.getElementById('modal-title'), body: document.getElementById('modal-body'), confirmBtn: document.getElementById('modal-confirm'), cancelBtn: document.getElementById('modal-cancel'), };
modal.cancelBtn.onclick = () => { modal.overlay.classList.add('hidden'); };
modal.confirmBtn.onclick = () => { if(confirmCallback) confirmCallback(); modal.overlay.classList.add('hidden'); };
function showConfirmation(title, bodyText, onConfirm) { modal.title.textContent = title; modal.body.textContent = bodyText; confirmCallback = onConfirm; modal.overlay.classList.remove('hidden'); modal.overlay.classList.add('flex'); }
function showTab(tabId, el) { document.querySelectorAll('.tab-content').forEach(t => t.classList.add('hidden')); document.getElementById(tabId + '-tab').classList.remove('hidden'); document.querySelectorAll('.tab-btn').forEach(b => { b.classList.remove('active', 'text-white'); b.classList.add('text-text-secondary'); }); el.classList.add('active', 'text-white'); el.classList.remove('text-text-secondary'); }
async function fetchData(url) { try { const r = await fetch(url); return r.ok ? await r.json() : null; } catch (e) { console.error('Fetch Error:', e); return null; } }
function updateMarketStatus() {
    fetchData('/api/market_status').then(data => {
        if (!data) return;
        document.getElementById('overall-regime').textContent = (data.market_state?.overall_regime || 'UNCERTAIN').replace(/_/g, ' ');
        document.getElementById('open-trades-count').textContent = `${data.open_trades_count} / ${data.max_open_trades}`;
        const sessions = document.getElementById('active-sessions-list');
        sessions.innerHTML = data.active_sessions.length > 0 ? data.active_sessions.map(s => `<span class="bg-accent-blue/20 text-accent-blue text-xs font-bold px-2 py-1 rounded">${s}</span>`).join('') : `<span class="bg-gray-700 text-text-secondary text-xs font-bold px-2 py-1 rounded">لا توجد</span>`;
        const newsIndicator = document.getElementById('news-pause-indicator');
        if (data.is_bot_paused_by_news) {
            newsIndicator.textContent = `🛑 متوقف بسبب الأخبار: ${data.news_pause_reason}`;
            newsIndicator.classList.remove('hidden');
            newsIndicator.classList.add('news-pause-active');
        } else {
            newsIndicator.classList.add('hidden');
            newsIndicator.classList.remove('news-pause-active');
        }
        const tradeToggle = document.getElementById('trading-toggle'), tradeText = document.getElementById('trading-status-text');
        tradeToggle.checked = data.is_trading_enabled;
        tradeText.textContent = data.is_trading_enabled ? 'مُفعَّل' : 'غير مُفعَّل';
        tradeText.className = `font-bold text-lg ${data.is_trading_enabled ? 'text-accent-green' : 'text-accent-red'}`;
        document.getElementById('usdt-balance').textContent = data.usdt_balance ? parseFloat(data.usdt_balance).toFixed(2) : 'N/A';
        if(data.settings) {
            document.getElementById('risk-percent').value = data.settings.risk_percent;
            document.getElementById('ob-ratio').value = data.settings.ob_ratio;
            document.getElementById('vol-multiplier').value = data.settings.vol_multiplier;
            document.getElementById('min-profit').value = data.settings.min_profit;
            document.getElementById('candle-filter-toggle').checked = data.settings.use_candle_filter;
            document.getElementById('volume-filter-toggle').checked = data.settings.use_volume_filter;
            document.getElementById('ob-filter-toggle').checked = data.settings.use_order_book_filter;
            document.getElementById('htf-confirmation-toggle').checked = data.settings.use_htf_confirmation_filter;
            document.getElementById('short-term-momentum-filter-toggle').checked = data.settings.use_short_term_momentum_filter;
            document.getElementById('ml-strategy-toggle').checked = data.settings.use_ml_strategy;
            document.getElementById('smart-breakout-strategy-toggle').checked = data.settings.use_smart_breakout_strategy;
            document.getElementById('rsi-divergence-strategy-toggle').checked = data.settings.use_rsi_divergence_strategy;
            document.getElementById('vwap-reversal-strategy-toggle').checked = data.settings.use_vwap_reversal_strategy;
            document.getElementById('bb-stoch-strategy-toggle').checked = data.settings.use_bb_stoch_strategy;
            document.getElementById('macd-ema-strategy-toggle').checked = data.settings.use_macd_ema_strategy;
            document.getElementById('qqe-ssl-strategy-toggle').checked = data.settings.use_qqe_ssl_strategy;
            document.getElementById('ema-rsi-strategy-toggle').checked = data.settings.use_ema_rsi_strategy;
            document.getElementById('pullback-strategy-toggle').checked = data.settings.use_pullback_strategy;
            document.getElementById('bb-squeeze-strategy-toggle').checked = data.settings.use_bb_squeeze_strategy;
            document.getElementById('bullish-momentum-strategy-toggle').checked = data.settings.use_bullish_momentum_strategy;
        }
    });
}
function updateSignals() {
    fetchData('/api/signals').then(data => {
        if (!data) return;
        const tableBody = document.getElementById('signals-table');
        tableBody.innerHTML = '';
        data.filter(s => ['open', 'updated'].includes(s.status)).forEach(s => {
            const profit = parseFloat(s.profit_percentage || 0);
            const pClass = profit > 0 ? 'text-accent-green' : profit < 0 ? 'text-accent-red' : 'text-text-secondary';
            const entry = parseFloat(s.entry_price);
            const current = parseFloat(s.current_price || entry);
            const currentTarget = parseFloat(s.target_price);
            tableBody.innerHTML += `<tr class="border-b border-border-color hover:bg-white/5"><td class="p-4 font-bold">${s.symbol}<br><span class="text-xs text-text-secondary">${s.strategy_name.replace(/_/g, ' ')}</span></td><td class="p-4 font-mono ${pClass}">${profit.toFixed(2)}%</td><td class="p-4 font-mono text-xs"><div><span class="text-text-secondary">الدخول:</span> ${entry.toFixed(4)}</div><div><span class="text-accent-blue">الحالي:</span> ${current.toFixed(4)}</div><div><span class="text-accent-green">الهدف:</span> ${currentTarget.toFixed(4)}</div></td><td class="p-4"><button onclick="manualClose(${s.id}, '${s.symbol}')" class="bg-red-600 hover:bg-red-700 text-white font-bold py-1 px-3 rounded text-xs w-full">إغلاق</button></td></tr>`;
        });
    });
}
function updateStats() {
    fetchData('/api/stats').then(data => {
        if (!data) return;
        const container = document.getElementById('stats-container');
        if (data.error) { container.innerHTML = `<div class="card p-4 text-center col-span-full text-accent-red">${data.error}</div>`; return; }
        container.innerHTML = `<div class="card p-4 text-center"><h4 class="text-text-secondary">صافي الربح</h4><div class="text-2xl font-bold ${data.net_profit_usdt >= 0 ? 'text-accent-green' : 'text-accent-red'}">${parseFloat(data.net_profit_usdt).toFixed(2)}</div></div><div class="card p-4 text-center"><h4 class="text-text-secondary">معدل الربح</h4><div class="text-2xl font-bold">${parseFloat(data.win_rate).toFixed(2)}%</div></div><div class="card p-4 text-center"><h4 class="text-text-secondary">عامل الربح</h4><div class="text-2xl font-bold">${data.profit_factor === 'Infinity' ? '∞' : parseFloat(data.profit_factor).toFixed(2)}</div></div><div class="card p-4 text-center"><h4 class="text-text-secondary">الصفقات المغلقة</h4><div class="text-2xl font-bold">${data.total_closed_trades}</div></div>`;
    });
}
function updateEconomicNews() {
    fetchData('/api/economic_news').then(data => {
        if (!data || data.length === 0) { document.getElementById('news-list').innerHTML = '<p class="text-text-secondary">لا توجد أخبار حالياً.</p>'; return; }
        document.getElementById('news-list').innerHTML = data.map(n => `<div class="p-3 border-b border-border-color/50"><p class="font-bold">${n.title}</p><div class="flex justify-between items-center mt-1"><span class="text-xs text-text-secondary font-mono">${new Date(n.pubDate).toLocaleString('ar-EG', { timeZone: 'UTC' })} UTC</span><a href="${n.link}" target="_blank" class="text-xs text-accent-blue hover:underline">المصدر: ${n.source}</a></div></div>`).join('');
    });
}
function updateNotifications() { fetchData('/api/notifications').then(data => { if (!data) return; document.getElementById('notifications-list').innerHTML = data.map(n => `<div class="p-2 border-b border-border-color"><span class="font-mono text-xs text-text-secondary">${new Date(n.timestamp).toLocaleString('ar-EG')}</span>: ${n.message}</div>`).join(''); }); }
function updateRejections() { fetchData('/api/rejection_logs').then(data => { if (!data) return; document.getElementById('rejections-list').innerHTML = data.map(r => `<div class="p-2 border-b border-border-color"><span class="font-mono text-xs text-text-secondary">${new Date(r.timestamp).toLocaleString('ar-EG')}</span>: <strong class="text-accent-yellow">${r.symbol}</strong> - ${r.reason} <span class="text-xs text-gray-500">${JSON.stringify(r.details)}</span></div>`).join(''); }); }
function manualClose(signalId, symbol) { showConfirmation('تأكيد الإغلاق', `هل أنت متأكد من رغبتك في إغلاق الصفقة لـ ${symbol} يدوياً؟`, () => { fetch(`/api/signals/close/${signalId}`, { method: 'POST' }).then(res => res.json()).then(data => { if(data.success) { updateSignals(); } else { alert(data.message); } }); }); }
function toggleTrading() { fetch('/api/trading/toggle', { method: 'POST' }).then(() => updateMarketStatus()); }
function saveSettings() {
    const settings = {
        risk_percent: parseFloat(document.getElementById('risk-percent').value), ob_ratio: parseFloat(document.getElementById('ob-ratio').value), vol_multiplier: parseFloat(document.getElementById('vol-multiplier').value), min_profit: parseFloat(document.getElementById('min-profit').value),
        use_candle_filter: document.getElementById('candle-filter-toggle').checked, use_volume_filter: document.getElementById('volume-filter-toggle').checked, use_order_book_filter: document.getElementById('ob-filter-toggle').checked,
        use_htf_confirmation_filter: document.getElementById('htf-confirmation-toggle').checked, use_short_term_momentum_filter: document.getElementById('short-term-momentum-filter-toggle').checked,
        use_ml_strategy: document.getElementById('ml-strategy-toggle').checked, use_smart_breakout_strategy: document.getElementById('smart-breakout-strategy-toggle').checked, use_rsi_divergence_strategy: document.getElementById('rsi-divergence-strategy-toggle').checked,
        use_vwap_reversal_strategy: document.getElementById('vwap-reversal-strategy-toggle').checked, use_bb_stoch_strategy: document.getElementById('bb-stoch-strategy-toggle').checked, use_macd_ema_strategy: document.getElementById('macd-ema-strategy-toggle').checked,
        use_qqe_ssl_strategy: document.getElementById('qqe-ssl-strategy-toggle').checked, use_ema_rsi_strategy: document.getElementById('ema-rsi-strategy-toggle').checked, use_pullback_strategy: document.getElementById('pullback-strategy-toggle').checked,
        use_bb_squeeze_strategy: document.getElementById('bb-squeeze-strategy-toggle').checked, use_bullish_momentum_strategy: document.getElementById('bullish-momentum-strategy-toggle').checked,
    };
    const feedbackEl = document.getElementById('settings-feedback');
    feedbackEl.textContent = 'جاري الحفظ...'; feedbackEl.className = 'mt-4 text-center text-accent-yellow';
    fetch('/api/settings/update', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(settings) })
    .then(res => res.json()).then(data => {
        if (data.success) { feedbackEl.textContent = '✅ تم حفظ الإعدادات بنجاح!'; feedbackEl.className = 'mt-4 text-center text-accent-green'; }
        else { feedbackEl.textContent = `❌ فشل الحفظ: ${data.message}`; feedbackEl.className = 'mt-4 text-center text-accent-red'; }
        setTimeout(() => { feedbackEl.textContent = ''; }, 3000);
    }).catch(err => { feedbackEl.textContent = `❌ خطأ في الشبكة: ${err}`; feedbackEl.className = 'mt-4 text-center text-accent-red'; });
}
document.addEventListener('DOMContentLoaded', () => {
    ['MarketStatus', 'Signals', 'Stats', 'EconomicNews', 'Notifications', 'Rejections'].forEach(f => window[`update${f}`]());
    setInterval(updateMarketStatus, 5000); setInterval(updateSignals, 7000); setInterval(updateStats, 60000);
    setInterval(updateEconomicNews, 120000); setInterval(updateNotifications, 15000); setInterval(updateRejections, 15000);
});
</script>
</body></html>
"""

@app.route('/')
def home():
    return render_template_string(get_dashboard_html())

@app.route('/api/market_status')
def get_market_status():
    with market_state_lock: state_copy = dict(current_market_state)
    with trading_status_lock: is_enabled = is_trading_enabled
    with bot_pause_lock: 
        is_paused = is_bot_paused_by_news
        reason = news_pause_reason
    with signal_cache_lock: open_trades = len(open_signals_cache)
    usdt_balance = None
    if client:
        try: usdt_balance = float(client.get_asset_balance(asset='USDT')['free'])
        except: usdt_balance = 'N/A'
    active_sessions, _, _ = get_session_state()
    
    with risk_per_trade_lock: risk = RISK_PER_TRADE_PERCENT
    with order_book_ratio_lock: ob_ratio = ORDER_BOOK_MIN_BID_ASK_RATIO
    with volume_filter_lock:
        vol_mult = VOLUME_FILTER_MULTIPLIER
        use_volume = USE_VOLUME_FILTER
    with candle_filter_lock: use_candle = USE_CANDLESTICK_FILTER
    with order_book_filter_enable_lock: use_ob = USE_ORDER_BOOK_FILTER
    with htf_confirmation_lock: use_htf = USE_HTF_CONFIRMATION_FILTER
    with short_term_momentum_filter_lock: use_stm = USE_SHORT_TERM_MOMENTUM_FILTER
    with ml_strategy_lock: use_ml = USE_ML_STRATEGY
    with smart_breakout_strategy_lock: use_smart_breakout = USE_SMART_BREAKOUT_STRATEGY
    with rsi_divergence_strategy_lock: use_rsi_divergence = USE_RSI_DIVERGENCE_STRATEGY
    with vwap_reversal_strategy_lock: use_vwap_reversal = USE_VWAP_REVERSAL_STRATEGY
    with bb_stoch_strategy_lock: use_bb_stoch = USE_BB_STOCH_STRATEGY
    with macd_ema_strategy_lock: use_macd_ema = USE_MACD_EMA_STRATEGY
    with qqe_ssl_strategy_lock: use_qqe_ssl = USE_QQE_SSL_STRATEGY
    with ema_rsi_strategy_lock: use_ema_rsi = USE_EMA_RSI_STRATEGY
    with pullback_strategy_lock: use_pullback = USE_PULLBACK_STRATEGY
    with bb_squeeze_strategy_lock: use_bb_squeeze = USE_BB_SQUEEZE_STRATEGY
    with bullish_momentum_strategy_lock: use_bullish_momentum = USE_BULLISH_MOMENTUM_STRATEGY

    return jsonify({
        "market_state": state_copy, "active_sessions": active_sessions, "usdt_balance": usdt_balance,
        "is_trading_enabled": is_enabled, "open_trades_count": open_trades, "max_open_trades": MAX_OPEN_TRADES,
        "is_bot_paused_by_news": is_paused, "news_pause_reason": reason,
        "settings": {
            "risk_percent": risk, "ob_ratio": ob_ratio, "vol_multiplier": vol_mult, "min_profit": MIN_PROFIT_PERCENT,
            "use_candle_filter": use_candle, "use_volume_filter": use_volume, "use_order_book_filter": use_ob,
            "use_htf_confirmation_filter": use_htf, "use_short_term_momentum_filter": use_stm,
            "use_ml_strategy": use_ml, "use_smart_breakout_strategy": use_smart_breakout,
            "use_rsi_divergence_strategy": use_rsi_divergence, "use_vwap_reversal_strategy": use_vwap_reversal,
            "use_bb_stoch_strategy": use_bb_stoch, "use_macd_ema_strategy": use_macd_ema, "use_qqe_ssl_strategy": use_qqe_ssl,
            "use_ema_rsi_strategy": use_ema_rsi, "use_pullback_strategy": use_pullback, "use_bb_squeeze_strategy": use_bb_squeeze,
            "use_bullish_momentum_strategy": use_bullish_momentum,
        }
    })

@app.route('/api/stats')
def get_stats():
    if not check_db_connection() or not conn:
        return jsonify({"error": "DB connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT profit_percentage, is_real_trade, original_quantity, entry_price FROM signals WHERE status = 'closed';")
            closed_trades = cur.fetchall()
        if not closed_trades:
            return jsonify({"net_profit_usdt": 0, "win_rate": 0, "profit_factor": 0, "total_closed_trades": 0})
        total_net_profit_usdt = sum(
            ((float(t['profit_percentage']) - (2 * TRADING_FEE_PERCENT)) / 100) * (float(t['original_quantity']) * float(t['entry_price']) if t.get('is_real_trade') and t.get('original_quantity') and t.get('entry_price') else STATS_TRADE_SIZE_USDT)
            for t in closed_trades
        )
        wins = [float(s['profit_percentage']) for s in closed_trades if float(s['profit_percentage']) > 0]
        losses = [float(s['profit_percentage']) for s in closed_trades if float(s['profit_percentage']) < 0]
        win_rate = (len(wins) / len(closed_trades) * 100) if closed_trades else 0.0
        total_loss = abs(sum(losses))
        profit_factor = sum(wins) / total_loss if total_loss > 0 else "Infinity"
        return jsonify({
            "net_profit_usdt": total_net_profit_usdt, "win_rate": win_rate,
            "profit_factor": profit_factor, "total_closed_trades": len(closed_trades)
        })
    except Exception as e:
        logger.error(f"❌ [API إحصائيات] خطأ: {e}", exc_info=True)
        return jsonify({"error": "Internal server error fetching stats"}), 500

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

@app.route('/api/economic_news')
def get_economic_news():
    with news_cache_lock:
        return jsonify(list(economic_news_cache))

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
    global RISK_PER_TRADE_PERCENT, ORDER_BOOK_MIN_BID_ASK_RATIO, VOLUME_FILTER_MULTIPLIER, \
           MIN_PROFIT_PERCENT, USE_CANDLESTICK_FILTER, USE_VOLUME_FILTER, USE_ORDER_BOOK_FILTER, USE_HTF_CONFIRMATION_FILTER, \
           USE_SHORT_TERM_MOMENTUM_FILTER, USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_QQE_SSL_STRATEGY, USE_EMA_RSI_STRATEGY, \
           USE_PULLBACK_STRATEGY, USE_BB_SQUEEZE_STRATEGY, USE_BULLISH_MOMENTUM_STRATEGY, USE_ML_STRATEGY, USE_SMART_BREAKOUT_STRATEGY, \
           USE_RSI_DIVERGENCE_STRATEGY, USE_VWAP_REVERSAL_STRATEGY
    try:
        data = request.get_json()
        with risk_per_trade_lock: RISK_PER_TRADE_PERCENT = float(data.get('risk_percent', RISK_PER_TRADE_PERCENT))
        with order_book_ratio_lock: ORDER_BOOK_MIN_BID_ASK_RATIO = float(data.get('ob_ratio', ORDER_BOOK_MIN_BID_ASK_RATIO))
        with volume_filter_lock:
            VOLUME_FILTER_MULTIPLIER = float(data.get('vol_multiplier', VOLUME_FILTER_MULTIPLIER))
            USE_VOLUME_FILTER = bool(data.get('use_volume_filter', USE_VOLUME_FILTER))
        MIN_PROFIT_PERCENT = float(data.get('min_profit', MIN_PROFIT_PERCENT))
        with candle_filter_lock: USE_CANDLESTICK_FILTER = bool(data.get('use_candle_filter', USE_CANDLESTICK_FILTER))
        with order_book_filter_enable_lock: USE_ORDER_BOOK_FILTER = bool(data.get('use_order_book_filter', USE_ORDER_BOOK_FILTER))
        with htf_confirmation_lock: USE_HTF_CONFIRMATION_FILTER = bool(data.get('use_htf_confirmation_filter', USE_HTF_CONFIRMATION_FILTER))
        with short_term_momentum_filter_lock: USE_SHORT_TERM_MOMENTUM_FILTER = bool(data.get('use_short_term_momentum_filter', USE_SHORT_TERM_MOMENTUM_FILTER))
        with ml_strategy_lock: USE_ML_STRATEGY = bool(data.get('use_ml_strategy', USE_ML_STRATEGY))
        with smart_breakout_strategy_lock: USE_SMART_BREAKOUT_STRATEGY = bool(data.get('use_smart_breakout_strategy', USE_SMART_BREAKOUT_STRATEGY))
        with rsi_divergence_strategy_lock: USE_RSI_DIVERGENCE_STRATEGY = bool(data.get('use_rsi_divergence_strategy', USE_RSI_DIVERGENCE_STRATEGY))
        with vwap_reversal_strategy_lock: USE_VWAP_REVERSAL_STRATEGY = bool(data.get('use_vwap_reversal_strategy', USE_VWAP_REVERSAL_STRATEGY))
        with bb_stoch_strategy_lock: USE_BB_STOCH_STRATEGY = bool(data.get('use_bb_stoch_strategy', USE_BB_STOCH_STRATEGY))
        with macd_ema_strategy_lock: USE_MACD_EMA_STRATEGY = bool(data.get('use_macd_ema_strategy', USE_MACD_EMA_STRATEGY))
        with qqe_ssl_strategy_lock: USE_QQE_SSL_STRATEGY = bool(data.get('use_qqe_ssl_strategy', USE_QQE_SSL_STRATEGY))
        with ema_rsi_strategy_lock: USE_EMA_RSI_STRATEGY = bool(data.get('use_ema_rsi_strategy', USE_EMA_RSI_STRATEGY))
        with pullback_strategy_lock: USE_PULLBACK_STRATEGY = bool(data.get('use_pullback_strategy', USE_PULLBACK_STRATEGY))
        with bb_squeeze_strategy_lock: USE_BB_SQUEEZE_STRATEGY = bool(data.get('use_bb_squeeze_strategy', USE_BB_SQUEEZE_STRATEGY))
        with bullish_momentum_strategy_lock: USE_BULLISH_MOMENTUM_STRATEGY = bool(data.get('use_bullish_momentum_strategy', USE_BULLISH_MOMENTUM_STRATEGY))
        log_and_notify('info', f"⚙️ تم تحديث الإعدادات من لوحة التحكم.", "SETTINGS_UPDATE")
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
def main_loop_enhanced():
    logger.info("[الحلقة الرئيسية] انتظار اكتمال التهيئة...")
    time.sleep(15)
    if not validated_symbols_to_scan:
        log_and_notify("critical", "قائمة العملات للمسح فارغة. يرجى التحقق من ملف 'crypto_list.txt'.", "SYSTEM_ERROR")
        return
    log_and_notify("info", f"✅ بدء حلقة المسح لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")
    while True:
        try:
            with bot_pause_lock:
                if is_bot_paused_by_news:
                    if (time.time() % 60) < 5:
                        logger.info(f"⏸️ البوت متوقف مؤقتاً. السبب: {news_pause_reason}")
                    time.sleep(5)
                    continue
            logger.info("🔄 [الحلقة الرئيسية] بدء دورة مسح جديدة...")
            determine_market_state_enhanced()
            btc_data = get_btc_data_for_bot()
            symbols_to_process = random.sample(validated_symbols_to_scan, len(validated_symbols_to_scan))
            total_batches = (len(symbols_to_process) + SYMBOL_PROCESSING_BATCH_SIZE - 1) // SYMBOL_PROCESSING_BATCH_SIZE
            for i in range(0, len(symbols_to_process), SYMBOL_PROCESSING_BATCH_SIZE):
                batch = symbols_to_process[i:i + SYMBOL_PROCESSING_BATCH_SIZE]
                logger.info(f"🔄 جاري معالجة الدفعة {i // SYMBOL_PROCESSING_BATCH_SIZE + 1}/{total_batches}...")
                for symbol in batch:
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
                        signal_found, strategy_used = False, None
                        strategies_to_check = []
                        with smart_breakout_strategy_lock:
                            if USE_SMART_BREAKOUT_STRATEGY: strategies_to_check.append(('SMART_BREAKOUT', check_smart_breakout_strategy, "Smart_Breakout_VWAP"))
                        with rsi_divergence_strategy_lock:
                            if USE_RSI_DIVERGENCE_STRATEGY: strategies_to_check.append(('RSI_DIVERGENCE', check_rsi_hidden_divergence_strategy, "RSI_Hidden_Divergence"))
                        with vwap_reversal_strategy_lock:
                            if USE_VWAP_REVERSAL_STRATEGY: strategies_to_check.append(('VWAP_REVERSAL', check_vwap_reversal_strategy, "VWAP_Reversal"))
                        with macd_ema_strategy_lock:
                            if USE_MACD_EMA_STRATEGY: strategies_to_check.append(('MACD_EMA', check_macd_ema_strategy, "MACD_EMA_Crossover"))
                        with bb_stoch_strategy_lock:
                            if USE_BB_STOCH_STRATEGY: strategies_to_check.append(('BB_STOCH', check_bb_stoch_strategy, "BB_Stoch_Reversal"))
                        with qqe_ssl_strategy_lock:
                            if USE_QQE_SSL_STRATEGY: strategies_to_check.append(('QQE_SSL', check_qqe_ssl_strategy_approx, "QQE_SSL_Explosion"))
                        with ema_rsi_strategy_lock:
                            if USE_EMA_RSI_STRATEGY: strategies_to_check.append(('EMA_RSI', check_ema_rsi_strategy, "EMA_RSI_Cross"))
                        with pullback_strategy_lock:
                            if USE_PULLBACK_STRATEGY: strategies_to_check.append(('PULLBACK', check_pullback_strategy, "Pullback_MACD"))
                        with bb_squeeze_strategy_lock:
                            if USE_BB_SQUEEZE_STRATEGY: strategies_to_check.append(('BB_SQUEEZE', check_bb_squeeze_strategy, "BB_Squeeze_Breakout"))
                        with bullish_momentum_strategy_lock:
                            if USE_BULLISH_MOMENTUM_STRATEGY: strategies_to_check.append(('BULLISH_MOMENTUM', check_bullish_momentum_strategy, "Bullish_Momentum"))
                        for key, check_func, name in strategies_to_check:
                            if check_func(df_with_indicators):
                                signal_found, strategy_used = True, name
                                break
                        if not signal_found:
                            continue
                        logger.info(f"  -> [{symbol}] إشارة أولية من {strategy_used}. بدء الفلاتر النهائية...")
                        if strategy_used == "Smart_Breakout_VWAP":
                            if has_large_ask_wall_nearby(symbol, df_with_indicators.iloc[-1]['close']):
                                log_rejection(symbol, "Large Ask Wall Ahead", {"strategy": strategy_used})
                                continue
                        with htf_confirmation_lock: use_htf_filter = USE_HTF_CONFIRMATION_FILTER
                        with short_term_momentum_filter_lock: use_stm_filter = USE_SHORT_TERM_MOMENTUM_FILTER
                        strategies_for_htf = ["EMA_RSI_Cross", "Bullish_Momentum", "MACD_EMA_Crossover", "Pullback_MACD", "Smart_Breakout_VWAP", "RSI_Hidden_Divergence", "VWAP_Reversal"]
                        strategies_for_stm = ["BB_Squeeze_Breakout", "QQE_SSL_Explosion", "BB_Stoch_Reversal"]
                        if use_htf_filter and strategy_used in strategies_for_htf:
                            if not is_htf_bullish_confirmation(symbol, htf=HIGHER_TIMEFRAME):
                                log_rejection(symbol, "HTF Trend Confirmation Failed", {"strategy": strategy_used}); continue
                        if use_stm_filter and strategy_used in strategies_for_stm:
                            if not passes_short_term_momentum_filter(symbol, df_with_indicators):
                                log_rejection(symbol, "Short-Term Momentum Filter Failed", {"strategy": strategy_used}); continue
                        df_for_filtering = df_with_indicators.iloc[:-1]
                        df_for_filtering.name = symbol
                        with candle_filter_lock: use_filter = USE_CANDLESTICK_FILTER
                        if use_filter and (strategy_used not in SCALPING_STRATEGIES):
                            if not is_bullish_reversal_pattern(df_for_filtering):
                                log_rejection(symbol, "Bullish Reversal Candle Pattern Failed", {"strategy": strategy_used}); continue
                        try: entry_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                        except Exception as e: logger.error(f"❌ [{symbol}] فشل جلب سعر الدخول: {e}."); continue
                        with order_book_filter_enable_lock: use_filter = USE_ORDER_BOOK_FILTER
                        if use_filter:
                            if not passes_final_order_book_check(symbol, entry_price):
                                continue
                        logger.info(f"  -> [{symbol}] ✅ نجحت جميع الفلاتر. جاري تحضير الصفقة...")
                        tp_sl_data = calculate_tp_sl(symbol, entry_price, df_with_indicators)
                        if not tp_sl_data: continue
                        new_signal = {
                            'symbol': symbol, 'strategy_name': strategy_used,
                            'signal_details': {**tp_sl_data},
                            'entry_price': entry_price, **tp_sl_data
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
            _, session_liquidity, _ = get_session_state()
            sleep_duration = 45 if session_liquidity == 'HIGH_LIQUIDITY' else 60
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
            with signal_cache_lock:
                if not open_signals_cache:
                    time.sleep(5)
                    continue
                signals_to_check = list(open_signals_cache.values())
            if not redis_client:
                time.sleep(5)
                continue
            current_prices = redis_client.hgetall(REDIS_PRICES_HASH_NAME)
            _, session_liquidity, _ = get_session_state()
            for signal in signals_to_check:
                current_price_str = current_prices.get(signal['symbol'])
                if not current_price_str: continue
                current_price = float(current_price_str)
                signal_id, symbol = signal['id'], signal['symbol']
                tp, sl, entry = float(signal['target_price']), float(signal['stop_loss']), float(signal['entry_price'])
                if current_price <= sl:
                    reason = 'atr_trailing_stop' if USE_ATR_TRAILING_STOP and sl > float(signal.get('initial_stop_loss', sl)) else 'stop_loss'
                    close_signal(signal_id, current_price, reason)
                    continue
                if current_price >= tp:
                    if USE_DYNAMIC_JOURNEY and signal.get('journey_state'):
                        journey_state = signal['journey_state']
                        if journey_state.get('is_complete'): continue
                        logger.info(f"🎉 [{symbol}] الهدف عند {tp:.4f} تحقق بسعر {current_price:.4f}")
                        if not journey_state.get('partial_exit_done'):
                            rr_ratio = float(signal.get('rr_ratio', 0.0))
                            partial_exit_percent = 0.6 if rr_ratio >= 2.0 else 0.4
                            if signal.get('is_real_trade') and partial_exit_percent > 0:
                                original_quantity = Decimal(str(signal.get('original_quantity', '0')))
                                exit_percentage = Decimal(str(partial_exit_percent))
                                exit_quantity = adjust_quantity_to_lot_size(symbol, float(original_quantity * exit_percentage))
                                if exit_quantity and exit_quantity > 0:
                                    sell_order = place_order(symbol, Client.SIDE_SELL, exit_quantity)
                                    if sell_order:
                                        executed_quantity = Decimal(sell_order.get('executedQty', '0'))
                                        if executed_quantity == 0: executed_quantity = exit_quantity
                                        remaining_quantity = Decimal(str(signal['quantity'])) - executed_quantity
                                        signal['quantity'] = float(remaining_quantity)
                                        log_and_notify('info', f"↗️ [{symbol}] خروج جزئي ({partial_exit_percent*100}%): بيع {executed_quantity} عند {current_price:.4f}", "PARTIAL_EXIT")
                                        is_dust = False
                                        if remaining_quantity > 0:
                                            symbol_info = exchange_info_map.get(symbol)
                                            if symbol_info:
                                                min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] in ('MIN_NOTIONAL', 'NOTIONAL')), None)
                                                if min_notional_filter:
                                                    min_notional = Decimal(min_notional_filter.get('minNotional', min_notional_filter.get('notional', '0')))
                                                    if (remaining_quantity * Decimal(str(current_price))) < min_notional:
                                                        is_dust = True
                                                        logger.warning(f"⚠️ [{symbol}] الكمية المتبقية ({remaining_quantity}) أقل من الحد الأدنى. سيتم إغلاق الصفقة بالكامل.")
                                        if remaining_quantity <= 0 or is_dust:
                                            close_signal(signal_id, current_price, 'take_profit_full_exit_on_small_size')
                                            continue 
                            journey_state['partial_exit_done'] = True
                        journey_state['targets_hit'] = journey_state.get('targets_hit', 0) + 1
                        df_analysis = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, 100)
                        if df_analysis is not None:
                            df_with_features = calculate_all_features(df_analysis, None)
                            if analyze_path_for_extension(df_with_features):
                                new_sl = tp
                                next_target_atr_multiplier = 1.45 if session_liquidity == 'HIGH_LIQUIDITY' else 1.25
                                next_tp = find_next_resistance(df_with_features, current_price)
                                if next_tp is None:
                                    last_atr = df_with_features['atr'].iloc[-1]
                                    next_tp = current_price + (last_atr * next_target_atr_multiplier)
                                    logger.info(f"  -> [{symbol}] لم يتم العثور على مقاومة، تم تحديد الهدف التالي باستخدام ATR ({next_target_atr_multiplier}x): {next_tp:.4f}")
                                else:
                                    logger.info(f"  -> [{symbol}] تم العثور على مقاومة تالية عند: {next_tp:.4f}")
                                signal['stop_loss'] = new_sl
                                signal['target_price'] = next_tp
                                signal['journey_state'] = journey_state
                                logger.info(f"🎯 [{symbol}] تمديد الرحلة! الهدف التالي: {next_tp:.4f}, وقف الخسارة الجديد: {new_sl:.4f}")
                                with signal_cache_lock: open_signals_cache[symbol] = signal
                                try:
                                    if check_db_connection():
                                        with conn.cursor() as cur:
                                            cur.execute("UPDATE signals SET journey_state = %s, target_price = %s, stop_loss = %s, quantity = %s WHERE id = %s",
                                                        (json.dumps(journey_state, cls=NpEncoder), float(signal['target_price']), float(signal['stop_loss']), float(signal.get('quantity', 0)), signal_id))
                                        conn.commit()
                                except Exception as e:
                                    logger.error(f"خطأ في قاعدة البيانات عند تحديث رحلة الصفقة لـ {symbol}: {e}"); conn.rollback()
                                continue
                        logger.info(f"⏹️ [{symbol}] تحليل المسار لا يدعم التمديد أو فشل جلب البيانات. إغلاق الصفقة.")
                        journey_state['is_complete'] = True
                        close_signal(signal_id, current_price, 'journey_completed')
                    else:
                        close_signal(signal_id, current_price, 'take_profit')
                peak_price = float(signal.get('current_peak_price', entry))
                new_peak = max(peak_price, current_price)
                if new_peak > peak_price:
                    signal['current_peak_price'] = new_peak
                    if USE_ATR_TRAILING_STOP:
                        try:
                            df_atr = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, ATR_TS_PERIOD + 1)
                            if df_atr is not None and len(df_atr) >= ATR_TS_PERIOD:
                                high_low = df_atr['high'] - df_atr['low']
                                high_close_prev = (df_atr['high'] - df_atr['close'].shift()).abs()
                                low_close_prev = (df_atr['low'] - df_atr['close'].shift()).abs()
                                tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)
                                latest_atr = tr.ewm(span=ATR_TS_PERIOD, adjust=False).mean().iloc[-1]
                                if latest_atr > 0:
                                    new_trailing_stop_price = new_peak - (latest_atr * ATR_TS_MULTIPLIER)
                                    if new_trailing_stop_price > sl:
                                        signal['stop_loss'] = new_trailing_stop_price
                        except Exception as e:
                            logger.error(f"❌ [{symbol}] خطأ أثناء حساب وقف الخسارة المتحرك: {e}")
                    with signal_cache_lock: open_signals_cache[symbol] = signal
                    try:
                        if check_db_connection():
                            with conn.cursor() as cur:
                                cur.execute("UPDATE signals SET current_peak_price = %s, stop_loss = %s WHERE id = %s", (float(new_peak), float(signal['stop_loss']), signal_id))
                            conn.commit()
                    except Exception as e:
                        logger.error(f"خطأ في قاعدة البيانات عند تحديث سعر الذروة/الوقف لـ {symbol}: {e}"); conn.rollback()
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
        Thread(target=main_loop_enhanced, daemon=True).start()
        Thread(target=price_update_loop, daemon=True).start()
        Thread(target=trade_management_loop, daemon=True).start()
        Thread(target=news_monitoring_loop, daemon=True).start()
        logger.info("✅ [خدمات البوت] تم بدء جميع الخدمات الخلفية بنجاح.")
        send_telegram_message("✅ *البوت قيد التشغيل الآن (نسخة V9.7.2)*")
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حرج أثناء التهيئة: {e}", "SYSTEM"); exit(1)

# ---------------------- نقطة الدخول ----------------------
if __name__ == "__main__":
    logger.info("🚀 إطلاق بوت التداول ولوحة التحكم (V9.7.2) 🚀")
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
