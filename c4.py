# ملف c4.py - نسخة V9.9.2 (محسّنة لفريم 15 دقيقة وتحسين الذاكرة مع حل مشاكل Redis و Binance)
# --- التغييرات الرئيسية (V9.9.2):
# 1. [إصلاح] حل مشكلة الوصول إلى المتغير العام redis_config_available
# 2. [إصلاح] إضافة معالجة لمشكلة معدل الطلبات في Binance

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
from binance.exceptions import BinanceAPIException, BinanceOrderException, BinanceRequestException
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
logger = logging.getLogger('CryptoBotV9.9.2')

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
HIGHER_TIMEFRAME: str = '1h' # الإطار الزمني الأعلى المستخدم في فلتر التأكيد
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 15  # تقليل من 90 إلى 15 يوم
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices_v10"
TRADING_FEE_PERCENT: float = 0.1
STATS_TRADE_SIZE_USDT: float = 4.0
BTC_SYMBOL: str = 'BTCUSDT'
MAX_OPEN_TRADES: int = 3  # تقليل من 5 إلى 3
SYMBOL_PROCESSING_BATCH_SIZE: int = 5  # تقليل من 10 إلى 5

# --- إعدادات رحلة التداول الديناميكية ---
USE_DYNAMIC_JOURNEY = True

# --- إعدادات المؤشرات الفنية (تم تعديلها للسكالبينج على فريم 15 دقيقة) ---
EMA_FAST_PERIOD: int = 12  # تقليل من 21 إلى 12
EMA_SLOW_PERIOD: int = 26  # تقليل من 50 إلى 26
ADX_PERIOD: int = 10       # تقليل من 14 إلى 10
RSI_PERIOD: int = 10       # تقليل من 14 إلى 10
ATR_PERIOD: int = 10       # تقليل من 14 إلى 10
BTC_CORR_PERIOD: int = 30
REL_VOL_PERIOD: int = 30
MOMENTUM_PERIOD: int = 5   # تقليل من 10 إلى 5
EMA_SLOPE_PERIOD: int = 5
SUPERTREND_ATR_PERIOD: int = 7  # تقليل من 10 إلى 7
SUPERTREND_MULTIPLIER: float = 3.0
CANDLE_AVG_VOLUME_PERIOD: int = 10  # تقليل من 15 إلى 10
SR_LOOKBACK_CANDLES: int = 40       # تقليل من 60 إلى 40
SR_MIN_BOUNCES: int = 2

# --- إعدادات الفلاتر المتقدمة وإدارة الصفقات ---
ORDER_BOOK_DEPTH_LIMIT: int = 100
ORDER_BOOK_ANALYSIS_RANGE_PCT: float = 0.005
USE_ATR_TRAILING_STOP: bool = True
ATR_TS_PERIOD: int = 14
ATR_TS_MULTIPLIER: float = 2.2

# --- إعدادات تحسين الذاكرة ---
TECHNICAL_SIGNAL_CACHE_DURATION: int = 60 * 2  # تقليل مدة التخزين المؤقت
REDIS_MAX_MEMORY: str = "256mb"  # تحديد أقصى استخدام للذاكرة
REDIS_POLICY: str = "allkeys-lru"  # سياسة حذف المفاتيح عند امتلاء الذاكرة
REDIS_CONFIG_ENABLED: bool = False  # تعطيل إعدادات Redis لحل مشكلة الصلاحيات

# --- إعدادات التحكم في معدل الطلبات ---
API_REQUEST_DELAY: float = 0.2  # تأخير بين الطلبات بالثواني
API_RETRY_COUNT: int = 3  # عدد مرات إعادة المحاولة
API_RETRY_DELAY: float = 5.0  # تأخير إعادة المحاولة بالثواني
RATE_LIMIT_BAN_TIME: int = 3600  # وقت الحظر بالثواني (ساعة واحدة)

# --- متغيرات الحالة والكاش ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
redis_config_available: bool = False  # متغير لتتبع ما إذا كانت إعدادات Redis متاحة
ml_models_cache: Dict[str, Any] = {}
exchange_info_map: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
notifications_cache = deque(maxlen=20)  # تقليل من 50 إلى 20
notifications_lock = Lock()
rejection_logs_cache = deque(maxlen=30)  # تقليل من 100 إلى 30
rejection_logs_lock = Lock()
current_market_state: Dict[str, Any] = {"overall_regime": "INITIALIZING", "trend_details_by_tf": {}, "last_updated": None}
market_state_lock = Lock()
last_market_state_check = 0
technical_signals_cache: Dict[str, Dict] = {}
api_ban_until: float = 0.0  # وقت انتهاء حظر API

# --- قاموس أسباب الرفض باللغة العربية ---
REJECTION_REASONS_AR = {
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
    "API Rate Limit Exceeded": "تم تجاوز الحد الأقصى لمعدل الطلبات",
    "API Temporarily Banned": "تم حظر IP مؤقتاً بسبب كثرة الطلبات",
}

# --- إعداد تطبيق Flask ---
app = Flask(__name__)
CORS(app)

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
    global redis_client, redis_config_available
    logger.info("[Redis] تهيئة الاتصال...")
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        
        # محاولة تعيين إعدادات Redis إذا كانت مفعلة
        if REDIS_CONFIG_ENABLED:
            try:
                redis_client.config_set('maxmemory', REDIS_MAX_MEMORY)
                redis_client.config_set('maxmemory-policy', REDIS_POLICY)
                redis_config_available = True
                logger.info("✅ [Redis] تم الاتصال وتعيين الإعدادات بنجاح.")
            except redis.exceptions.NoPermissionError:
                logger.warning("⚠️ [Redis] لا توجد صلاحية لتعيين إعدادات Redis. سيتم استخدام Redis بدون إعدادات مخصصة.")
                redis_config_available = False
            except Exception as e:
                logger.warning(f"⚠️ [Redis] خطأ في تعيين إعدادات Redis: {e}. سيتم استخدام Redis بدون إعدادات مخصصة.")
                redis_config_available = False
        else:
            logger.info("✅ [Redis] تم الاتصال بنجاح (بدون إعدادات مخصصة).")
            redis_config_available = False
            
    except redis.exceptions.ConnectionError as e:
        logger.warning(f"⚠️ [Redis] فشل الاتصال بـ Redis: {e}. سيتم العمل بدون Redis.")
        redis_client = None
        redis_config_available = False

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
    global api_ban_until
    if not client: return None
    
    # التحقق من حالة الحظر
    current_time = time.time()
    if current_time < api_ban_until:
        remaining_time = int(api_ban_until - current_time)
        logger.warning(f"⚠️ [جلب البيانات] IP محظور حتى {datetime.fromtimestamp(api_ban_until)}. المتبقي: {remaining_time} ثانية")
        return None
    
    # إضافة تأخير بين الطلبات
    time.sleep(API_REQUEST_DELAY)
    
    for attempt in range(API_RETRY_COUNT):
        try:
            lookback_str = f"{days} day" if 'd' in interval.lower() else f"{days * 24} hour"
            
            klines = client.get_historical_klines(symbol, interval, lookback_str)
            if not klines: return None
            
            cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = pd.DataFrame(klines, columns=cols)[:len(klines)-1]  # تجاهل آخر شمعة غير مكتملة
            
            # تحسين أنواع البيانات لتوفير الذاكرة
            numeric_cols = {'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'float32'}
            df = df.astype(numeric_cols)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            
            return df.dropna()
        except BinanceAPIException as e:
            if e.code == -1003:  # Rate limit exceeded
                logger.warning(f"⚠️ [جلب البيانات] تم تجاوز الحد الأقصى لمعدل الطلبات لـ {symbol}. محاولة {attempt + 1}/{API_RETRY_COUNT}")
                if attempt < API_RETRY_COUNT - 1:
                    time.sleep(API_RETRY_DELAY * (attempt + 1))  # تأخير متزايد
                else:
                    logger.error(f"❌ [جلب البيانات] تم حظر IP مؤقتاً لـ {symbol} بسبب كثرة الطلبات")
                    api_ban_until = current_time + RATE_LIMIT_BAN_TIME
                    send_telegram_message(f"⚠️ تم حظر IP مؤقتاً بسبب كثرة الطلبات. سيتم إعادة المحاولة بعد {RATE_LIMIT_BAN_TIME//60} دقيقة")
                    return None
            else:
                logger.error(f"❌ [جلب البيانات] خطأ في جلب البيانات التاريخية لـ {symbol} ({interval}): {e}")
                return None
        except Exception as e:
            logger.error(f"❌ [جلب البيانات] خطأ في جلب البيانات التاريخية لـ {symbol} ({interval}): {e}")
            if attempt < API_RETRY_COUNT - 1:
                time.sleep(API_RETRY_DELAY)
            else:
                return None
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
    df_calc['ema_12'] = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()  # استخدام EMA_FAST_PERIOD
    df_calc['ema_21'] = df_calc['close'].ewm(span=21, adjust=False).mean()
    df_calc['sma_50'] = df_calc['close'].rolling(window=50).mean()
    df_calc['sma_200'] = df_calc['close'].rolling(window=200).mean()
    df_calc['volume_sma_20'] = df_calc['volume'].rolling(window=20).mean()
    df_calc['ema_26'] = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()  # استخدام EMA_SLOW_PERIOD
    df_calc['ema_50'] = df_calc['close'].ewm(span=50, adjust=False).mean()
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
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 20;")
            recent = cur.fetchall()
            with notifications_lock:
                notifications_cache.clear()
                for n in reversed(recent):
                    n['timestamp'] = n['timestamp'].isoformat()
                    notifications_cache.appendleft(dict(n))
            logger.info(f"✅ [تحميل] تم تحميل {len(notifications_cache)} إشعار إلى الذاكرة المؤقتة.")
    except Exception as e:
        logger.error(f"❌ [تحميل] فشل تحميل الإشعارات: {e}")

# --- دوال تحسين الذاكرة ---
def optimize_memory_usage():
    """وظيفة لتحسين استخدام الذاكرة"""
    global redis_config_available
    try:
        # جمع القمامة بشكل دوري
        gc.collect()
        
        # تقليل حجم البيانات في الذاكرة
        with signal_cache_lock:
            if len(open_signals_cache) > 10:  # الحفاظ على أحدث 10 إشارات فقط
                oldest_signals = sorted(open_signals_cache.items(), key=lambda x: x[1].get('timestamp', ''))[:len(open_signals_cache)-10]
                for symbol, _ in oldest_signals:
                    del open_signals_cache[symbol]
        
        with notifications_lock:
            if len(notifications_cache) > 20:  # الحفاظ على أحدث 20 إشعار فقط
                while len(notifications_cache) > 20:
                    notifications_cache.pop()
        
        with rejection_logs_lock:
            if len(rejection_logs_cache) > 30:  # الحفاظ على أحدث 30 رفض فقط
                while len(rejection_logs_cache) > 30:
                    rejection_logs_cache.pop()
        
        # محاولة تحسين Redis إذا كانت الإعدادات متاحة
        if redis_client and redis_config_available:
            try:
                redis_client.config_set('maxmemory', REDIS_MAX_MEMORY)
                redis_client.config_set('maxmemory-policy', REDIS_POLICY)
            except Exception as e:
                logger.warning(f"⚠️ [Redis] خطأ في تحديث إعدادات Redis: {e}")
                redis_config_available = False
        
        logger.info("✅ تم تحسين استخدام الذاكرة")
    except Exception as e:
        logger.error(f"❌ خطأ في تحسين الذاكرة: {e}")

def load_settings_from_redis():
    """تحميل الإعدادات من Redis"""
    global RISK_PER_TRADE_PERCENT, BUY_CONFIDENCE_THRESHOLD, MAX_OPEN_TRADES, MIN_PROFIT_PERCENT
    global USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY
    
    if not redis_client:
        return
    
    try:
        # تحميل إعدادات التداول
        settings_data = redis_client.get('trading_settings')
        if settings_data:
            settings = json.loads(settings_data)
            
            with risk_per_trade_lock:
                RISK_PER_TRADE_PERCENT = settings.get('RISK_PER_TRADE_PERCENT', RISK_PER_TRADE_PERCENT)
            
            with buy_confidence_lock:
                BUY_CONFIDENCE_THRESHOLD = settings.get('BUY_CONFIDENCE_THRESHOLD', BUY_CONFIDENCE_THRESHOLD)
            
            MAX_OPEN_TRADES = settings.get('MAX_OPEN_TRADES', MAX_OPEN_TRADES)
            MIN_PROFIT_PERCENT = settings.get('MIN_PROFIT_PERCENT', MIN_PROFIT_PERCENT)
        
        # تحميل إعدادات الاستراتيجيات
        strategies_data = redis_client.get('strategy_settings')
        if strategies_data:
            strategies = json.loads(strategies_data)
            
            with bb_stoch_strategy_lock:
                USE_BB_STOCH_STRATEGY = strategies.get('USE_BB_STOCH_STRATEGY', USE_BB_STOCH_STRATEGY)
            
            with macd_ema_strategy_lock:
                USE_MACD_EMA_STRATEGY = strategies.get('USE_MACD_EMA_STRATEGY', USE_MACD_EMA_STRATEGY)
            
            with ema_rsi_strategy_lock:
                USE_EMA_RSI_STRATEGY = strategies.get('USE_EMA_RSI_STRATEGY', USE_EMA_RSI_STRATEGY)
            
            with pullback_strategy_lock:
                USE_PULLBACK_STRATEGY = strategies.get('USE_PULLBACK_STRATEGY', USE_PULLBACK_STRATEGY)
        
        logger.info("✅ تم تحميل الإعدادات من Redis")
    except Exception as e:
        logger.error(f"❌ خطأ في تحميل الإعدادات من Redis: {e}")

# --- منطق التداول والفلاتر ---
def check_market_volatility_filter(df: pd.DataFrame) -> bool:
    """فلتر لتجنب التداول في فترات التقلب الشديد أو المنخفض جدًا"""
    if len(df) < 50:
        return False
    
    last = df.iloc[-1]
    # التأكد من وجود الأعمدة المطلوبة
    if 'atr' not in last or 'close' not in last or last['close'] == 0:
        return False # لا يمكن الحساب، نفترض أنه غير صالح
        
    atr_percent = (last['atr'] / last['close']) * 100
    
    # تجنب التداول عندما يكون التقلب منخفض جدًا أو مرتفع جدًا
    if atr_percent < 0.5 or atr_percent > 5.0:
        log_rejection(df.name, "Market Volatility Filter Failed", {"atr_percent": f"{atr_percent:.2f}"})
        return False
    
    return True

def check_trend_strength_filter(df: pd.DataFrame) -> bool:
    """فلتر لتأكيد قوة الاتجاه الحالي"""
    if len(df) < 50:
        return False
        
    last = df.iloc[-1]
    
    # التأكد من وجود الأعمدة المطلوبة
    if 'adx' not in last or f'roc_{MOMENTUM_PERIOD}' not in last:
        return False # لا يمكن الحساب، نفترض أنه غير صالح

    # تجنب التداول في الأسواق الجانبية
    if last['adx'] < 18:
        log_rejection(df.name, "Trend Strength Filter Failed", {"reason": "ADX too low", "adx": f"{last['adx']:.2f}"})
        return False
        
    # التأكد من وجود زخم كافٍ
    if abs(last[f'roc_{MOMENTUM_PERIOD}']) < 0.5:
        log_rejection(df.name, "Trend Strength Filter Failed", {"reason": "ROC too low", "roc_10": f"{last[f'roc_{MOMENTUM_PERIOD}']:.2f}"})
        return False
        
    return True

def check_bb_stoch_strategy_revised(df: pd.DataFrame) -> bool:
    """
    استراتيجية BB+Stoch المعدلة للدخول المبكر على فريم 15 دقيقة.
    """
    if len(df) < 15:  # تقليل الحد الأدنى للبيانات
        return False

    last, prev = df.iloc[-1], df.iloc[-2]
    
    # شروط أكثر صرامة لفريم 15 دقيقة
    prev_candle_touch_bb = prev['low'] <= prev['bb_lower'] * 1.001  # لمس دقيق
    current_candle_is_bullish = last['close'] > last['open'] and (last['close'] - last['open']) > (last['high'] - last['low']) * 0.6
    stoch_turning_up = (last['stoch_rsi_k'] < 35 and last['stoch_rsi_k'] > prev['stoch_rsi_k'] and prev['stoch_rsi_k'] < 20)
    rsi_not_extreme = last['rsi'] > 25  # تجنب التشبع البيعي الشديد
    
    # إضافة فلتر الحجم
    volume_ok = last['volume'] > last['volume'].rolling(10).mean() * 1.2
    
    conditions = {
        "prev_candle_touch_bb": prev_candle_touch_bb,
        "current_candle_is_bullish": current_candle_is_bullish,
        "stoch_turning_up": stoch_turning_up,
        "rsi_not_extreme": rsi_not_extreme,
        "volume_ok": volume_ok
    }

    if all(conditions.values()):
        logger.info(f"  -> [{df.name}] ✅ إشارة BB+Stoch (معدلة لفريم 15 دقيقة).")
        return True

    return False

def check_macd_ema_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 3: return False
    last, prev, prev_prev = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    
    # تعديل الشروط لفريم 15 دقيقة
    macd_cross_up = prev['macd'] < prev['macd_signal'] and last['macd'] > last['macd_signal']
    price_above_ema = last['close'] > last['ema_12']  # استخدام EMA السريع
    macd_strength = last['macd_histogram'] > 0 and last['macd_histogram'] > prev['macd_histogram'] * 1.2
    trend_strength = last['adx'] > 18  # تقليل الحد الأدنى
    macd_position = prev_prev['macd'] < 0
    
    if macd_cross_up and price_above_ema and macd_strength and trend_strength and macd_position:
        logger.info(f"  -> [{df.name}] ✅ إشارة MACD+EMA (معدلة لفريم 15 دقيقة).")
        return True
    return False

def check_ema_rsi_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 2: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    ema_cross_up = prev['ema_9'] < prev['ema_12'] and last['ema_9'] > last['ema_12']  # استخدام EMA السريع
    rsi_strong = last['rsi'] > 52
    trend_filter = last['close'] > last['ema_26']  # استخدام EMA البطيء
    if ema_cross_up and rsi_strong and trend_filter:
        logger.info(f"  -> [{df.name}] ✅ إشارة استراتيجية EMA+RSI Cross.")
        return True
    return False

def check_pullback_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 2: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    uptrend_confirmed = last['close'] > last['ema_12'] and last['ema_12'] > last['ema_26']  # استخدام EMA السريع والبطيء
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
    if len(df) < 50:
        return False

    last = df.iloc[-1]
    
    price_above_sma50 = last['close'] > last['sma_50']
    strong_trend = last['adx'] > 25
    bullish_direction = last['plus_di'] > last['minus_di']
    rsi_is_bullish = 50 < last['rsi'] < 75
    
    if len(df) < 8: 
        return False
        
    recent_highs = df['high'].iloc[-8:-1]
    recent_lows = df['low'].iloc[-8:-1]
    
    def is_higher_highs(highs, min_count=3):
        if len(highs) < min_count + 1: return False
        count = 0
        for i in range(1, len(highs)):
            if highs.iloc[i] > highs.iloc[i-1]: count += 1
        return count >= min_count
    
    def is_higher_lows(lows, min_count=3):
        if len(lows) < min_count + 1: return False
        count = 0
        for i in range(1, len(lows)):
            if lows.iloc[i] > lows.iloc[i-1]: count += 1
        return count >= min_count
    
    price_momentum_confirmed = is_higher_highs(recent_highs) and is_higher_lows(recent_lows)
    volume_confirmation = last['volume'] > last['volume_sma_20'] * 1.1
    
    if all([price_above_sma50, strong_trend, bullish_direction, rsi_is_bullish, price_momentum_confirmed, volume_confirmation]):
        logger.info(f"  -> [{df.name}] ✅ إشارة زخم صعودي (معززة).")
        return True

    return False

def check_support_resistance_strategy_enhanced(df: pd.DataFrame) -> bool:
    """استراتيجية اختراق الدعم والمقاومة المحسنة"""
    if len(df) < 50:
        return False
    
    last = df.iloc[-1]
    
    # تحديد مستويات المقاومة بدقة أكبر
    resistance_candidates = df[df['high'] == df['high'].rolling(5, center=True).max()]['high']
    
    if resistance_candidates.empty:
        return False
    
    # أقرب مستوى مقاومة فوق السعر
    current_price = last['close']
    next_resistance_series = resistance_candidates[resistance_candidates > current_price]
    closest_resistance = next_resistance_series.min() if not next_resistance_series.empty else None
    
    # شرط اختراق المقاومة
    if closest_resistance is not None:
        # فلتر جديد: التحقق من قوة مستوى المقاومة (عدد مرات الاختبار)
        tolerance = closest_resistance * 0.01  # 1% tolerance
        resistance_tests = df[(df['high'] >= closest_resistance - tolerance) & 
                              (df['high'] <= closest_resistance + tolerance)]
        resistance_strength = len(resistance_tests)

        if resistance_strength < 2: # التأكد من أن المستوى تم اختباره مرتين على الأقل
            return False

        breakout = last['close'] > closest_resistance and df['close'].iloc[-2] <= closest_resistance
        
        # تأكيد الاختراق بحجم التداول
        volume_confirmation = last['volume'] > last['volume_sma_20'] * 1.3
        
        # تأكيد الاتجاه
        trend_confirmation = last['close'] > last['ema_12']
        
        # فلتر جديد: تجنب الاختراقات الكاذبة
        not_false_breakout = last['close'] > closest_resistance * 1.005  # اختراق حقيقي وليس مجرد لمس
        
        conditions = {
            "breakout": breakout,
            "volume_confirmation": volume_confirmation,
            "trend_confirmation": trend_confirmation,
            "not_false_breakout": not_false_breakout
        }

        if all(conditions.values()):
            logger.info(f"  -> [{df.name}] ✅ إشارة اختراق مقاومة (معززة) - قوة المستوى: {resistance_strength}")
            return True
        
        failed_conditions = {k: v for k, v in conditions.items() if not v}
        if breakout: # نسجل فقط إذا كان هناك اختراق لكنه فشل في الفلاتر الأخرى
             log_rejection(df.name, "SR Breakout Strategy Conditions Not Met", {"failed": list(failed_conditions.keys())})

    return False

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
    return (c1['close'] < c1['open'] and
            abs(c2['open'] - c2['close']) < c1_body * 0.5 and
            c3['close'] > c3['open'] and
            c3['close'] > (c1['open'] + c1['close']) / 2)

def is_three_white_soldiers(c1: pd.Series, c2: pd.Series, c3: pd.Series) -> bool:
    return (c1['close'] > c1['open'] and
            c2['close'] > c2['open'] and
            c3['close'] > c3['open'] and
            c1['open'] > c1['close'].shift() and
            c2['open'] > c2['close'].shift() and
            c3['open'] > c3['close'].shift())

# --- مسارات Flask ---
@app.route('/')
def dashboard():
    """لوحة التحكم الرئيسية"""
    # تحسين أداء الصفحة
    start_time = time.time()
    
    # الحصول على البيانات الأساسية
    with market_state_lock:
        market_state = current_market_state.copy()
    
    with trading_status_lock:
        trading_enabled = is_trading_enabled
    
    # الحصول على الإشارات المفتوحة (محدودة العدد)
    with signal_cache_lock:
        open_signals = dict(list(open_signals_cache.items())[:5])  # عرض أول 5 إشارات فقط
    
    # الحصول على الإشعارات الأخيرة (محدودة العدد)
    with notifications_lock:
        notifications = list(notifications_cache)[:10]  # عرض آخر 10 إشعارات فقط
    
    # الحصول على سجل الرفوض (محدود العدد)
    with rejection_logs_lock:
        rejections = list(rejection_logs_cache)[:10]  # عرض آخر 10 رفض فقط
    
    # حساب وقت التحميل
    load_time = round((time.time() - start_time) * 1000, 2)
    
    # تحميل القالب
    return render_template_string(DASHBOARD_TEMPLATE, 
                                market_state=market_state,
                                trading_enabled=trading_enabled,
                                open_signals=open_signals,
                                notifications=notifications,
                                rejections=rejections,
                                load_time=load_time,
                                redis_config_available=redis_config_available)

@app.route('/settings')
def settings():
    """صفحة الإعدادات المتقدمة"""
    return render_template_string(SETTINGS_TEMPLATE, 
                                redis_config_available=redis_config_available)

@app.route('/toggle_trading', methods=['POST'])
def toggle_trading():
    """تبديل حالة التداول"""
    global is_trading_enabled
    try:
        with trading_status_lock:
            is_trading_enabled = not is_trading_enabled
            status = "مفعل" if is_trading_enabled else "معطل"
            log_and_notify("info", f"تم {status} التداول", "TRADING_STATUS")
            send_telegram_message(f"⚙️ تم {status} التداول")
            return jsonify({"success": True, "message": f"تم {status} التداول"})
    except Exception as e:
        logger.error(f"❌ خطأ في تبديل حالة التداول: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/update_settings', methods=['POST'])
def update_settings():
    """تحديث إعدادات التداول"""
    try:
        data = request.json
        
        # تحديث الإعدادات العامة
        global RISK_PER_TRADE_PERCENT, BUY_CONFIDENCE_THRESHOLD, MAX_OPEN_TRADES, MIN_PROFIT_PERCENT
        
        with risk_per_trade_lock:
            RISK_PER_TRADE_PERCENT = float(data.get('risk_per_trade', RISK_PER_TRADE_PERCENT))
        
        with buy_confidence_lock:
            BUY_CONFIDENCE_THRESHOLD = float(data.get('buy_confidence', BUY_CONFIDENCE_THRESHOLD))
        
        MAX_OPEN_TRADES = int(data.get('max_trades', MAX_OPEN_TRADES))
        MIN_PROFIT_PERCENT = float(data.get('min_profit', MIN_PROFIT_PERCENT))
        
        # حفظ الإعدادات في Redis إذا كان متاحًا
        if redis_client:
            settings = {
                'RISK_PER_TRADE_PERCENT': RISK_PER_TRADE_PERCENT,
                'BUY_CONFIDENCE_THRESHOLD': BUY_CONFIDENCE_THRESHOLD,
                'MAX_OPEN_TRADES': MAX_OPEN_TRADES,
                'MIN_PROFIT_PERCENT': MIN_PROFIT_PERCENT
            }
            redis_client.set('trading_settings', json.dumps(settings))
        
        log_and_notify("info", "تم تحديث إعدادات التداول", "SETTINGS")
        return jsonify({"success": True, "message": "تم تحديث الإعدادات بنجاح"})
    except Exception as e:
        logger.error(f"❌ خطأ في تحديث الإعدادات: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/update_strategies', methods=['POST'])
def update_strategies():
    """تحديث إعدادات الاستراتيجيات"""
    try:
        data = request.json
        
        # تحديث إعدادات الاستراتيجيات
        global USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY
        
        with bb_stoch_strategy_lock:
            USE_BB_STOCH_STRATEGY = data.get('use_bb_stoch', USE_BB_STOCH_STRATEGY)
        
        with macd_ema_strategy_lock:
            USE_MACD_EMA_STRATEGY = data.get('use_macd_ema', USE_MACD_EMA_STRATEGY)
        
        with ema_rsi_strategy_lock:
            USE_EMA_RSI_STRATEGY = data.get('use_ema_rsi', USE_EMA_RSI_STRATEGY)
        
        with pullback_strategy_lock:
            USE_PULLBACK_STRATEGY = data.get('use_pullback', USE_PULLBACK_STRATEGY)
        
        # حفظ الإعدادات في Redis إذا كان متاحًا
        if redis_client:
            strategies = {
                'USE_BB_STOCH_STRATEGY': USE_BB_STOCH_STRATEGY,
                'USE_MACD_EMA_STRATEGY': USE_MACD_EMA_STRATEGY,
                'USE_EMA_RSI_STRATEGY': USE_EMA_RSI_STRATEGY,
                'USE_PULLBACK_STRATEGY': USE_PULLBACK_STRATEGY
            }
            redis_client.set('strategy_settings', json.dumps(strategies))
        
        log_and_notify("info", "تم تحديث إعدادات الاستراتيجيات", "SETTINGS")
        return jsonify({"success": True, "message": "تم تحديث إعدادات الاستراتيجيات بنجاح"})
    except Exception as e:
        logger.error(f"❌ خطأ في تحديث إعدادات الاستراتيجيات: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/reset_settings', methods=['POST'])
def reset_settings():
    """إعادة الإعدادات إلى القيم الافتراضية"""
    try:
        # إعادة الإعدادات العامة
        global RISK_PER_TRADE_PERCENT, BUY_CONFIDENCE_THRESHOLD, MAX_OPEN_TRADES, MIN_PROFIT_PERCENT
        
        with risk_per_trade_lock:
            RISK_PER_TRADE_PERCENT = 0.85
        
        with buy_confidence_lock:
            BUY_CONFIDENCE_THRESHOLD = 0.53
        
        MAX_OPEN_TRADES = 3
        MIN_PROFIT_PERCENT = 0.8
        
        # إعادة إعدادات الاستراتيجيات
        global USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY
        
        with bb_stoch_strategy_lock:
            USE_BB_STOCH_STRATEGY = True
        
        with macd_ema_strategy_lock:
            USE_MACD_EMA_STRATEGY = True
        
        with ema_rsi_strategy_lock:
            USE_EMA_RSI_STRATEGY = True
        
        with pullback_strategy_lock:
            USE_PULLBACK_STRATEGY = True
        
        # حفظ الإعدادات في Redis إذا كان متاحًا
        if redis_client:
            settings = {
                'RISK_PER_TRADE_PERCENT': RISK_PER_TRADE_PERCENT,
                'BUY_CONFIDENCE_THRESHOLD': BUY_CONFIDENCE_THRESHOLD,
                'MAX_OPEN_TRADES': MAX_OPEN_TRADES,
                'MIN_PROFIT_PERCENT': MIN_PROFIT_PERCENT
            }
            redis_client.set('trading_settings', json.dumps(settings))
            
            strategies = {
                'USE_BB_STOCH_STRATEGY': USE_BB_STOCH_STRATEGY,
                'USE_MACD_EMA_STRATEGY': USE_MACD_EMA_STRATEGY,
                'USE_EMA_RSI_STRATEGY': USE_EMA_RSI_STRATEGY,
                'USE_PULLBACK_STRATEGY': USE_PULLBACK_STRATEGY
            }
            redis_client.set('strategy_settings', json.dumps(strategies))
        
        log_and_notify("info", "تم إعادة الإعدادات إلى القيم الافتراضية", "SETTINGS")
        return jsonify({"success": True, "message": "تم إعادة الإعدادات إلى القيم الافتراضية"})
    except Exception as e:
        logger.error(f"❌ خطأ في إعادة الإعدادات: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# --- قوالب HTML ---
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم بوت التداول</title>
    <style>
        :root {
            --primary-color: #3498db;
            --success-color: #2ecc71;
            --danger-color: #e74c3c;
            --warning-color: #f39c12;
            --dark-color: #2c3e50;
            --light-color: #ecf0f1;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background-color: #f8f9fa;
            color: var(--dark-color);
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background-color: var(--dark-color);
            color: white;
            padding: 15px 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header-title {
            font-size: 24px;
            font-weight: bold;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: var(--danger-color);
        }
        
        .status-dot.active {
            background-color: var(--success-color);
        }
        
        .toggle-btn {
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        
        .toggle-btn:hover {
            background-color: #2980b9;
        }
        
        .toggle-btn.stop {
            background-color: var(--danger-color);
        }
        
        .toggle-btn.stop:hover {
            background-color: #c0392b;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            padding: 20px;
            transition: transform 0.3s;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }
        
        .card-title {
            font-size: 18px;
            font-weight: bold;
            color: var(--dark-color);
        }
        
        .card-icon {
            font-size: 24px;
        }
        
        .market-state {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        .state-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        
        .state-label {
            font-weight: bold;
        }
        
        .state-value {
            color: var(--primary-color);
        }
        
        .signal-item, .notification-item, .rejection-item {
            padding: 12px;
            border-radius: 5px;
            margin-bottom: 10px;
            border-right: 4px solid transparent;
        }
        
        .signal-item {
            background-color: #e8f5e9;
            border-right-color: var(--success-color);
        }
        
        .notification-item.info {
            background-color: #e3f2fd;
            border-right-color: var(--primary-color);
        }
        
        .notification-item.warning {
            background-color: #fff8e1;
            border-right-color: var(--warning-color);
        }
        
        .notification-item.error {
            background-color: #ffebee;
            border-right-color: var(--danger-color);
        }
        
        .rejection-item {
            background-color: #fbe9e7;
            border-right-color: var(--danger-color);
        }
        
        .item-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        
        .item-title {
            font-weight: bold;
        }
        
        .item-time {
            font-size: 12px;
            color: #777;
        }
        
        .item-content {
            font-size: 14px;
        }
        
        .redis-status {
            display: flex;
            align-items: center;
            gap: 5px;
            margin-top: 10px;
            font-size: 14px;
        }
        
        .redis-status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: var(--danger-color);
        }
        
        .redis-status-dot.available {
            background-color: var(--success-color);
        }
        
        .footer {
            text-align: center;
            margin-top: 20px;
            padding: 10px;
            color: #777;
            font-size: 14px;
        }
        
        .performance {
            background-color: var(--light-color);
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            margin-top: 10px;
        }
        
        @media (max-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
            
            header {
                flex-direction: column;
                gap: 10px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-title">بوت التداول الإلكتروني V9.9.2</div>
            <div class="status-indicator">
                <div class="status-dot {{ 'active' if trading_enabled else '' }}"></div>
                <span>{{ 'نشط' if trading_enabled else 'متوقف' }}</span>
                <button class="toggle-btn {{ 'stop' if trading_enabled else '' }}" onclick="toggleTrading()">
                    {{ 'إيقاف' if trading_enabled else 'تشغيل' }}
                </button>
            </div>
        </header>
        
        <div class="dashboard-grid">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">حالة السوق</div>
                    <div class="card-icon">📊</div>
                </div>
                <div class="market-state">
                    <div class="state-item">
                        <span class="state-label">النظام العام:</span>
                        <span class="state-value">{{ market_state.overall_regime }}</span>
                    </div>
                    <div class="state-item">
                        <span class="state-label">آخر تحديث:</span>
                        <span class="state-value">{{ market_state.last_updated }}</span>
                    </div>
                    <div class="redis-status">
                        <div class="redis-status-dot {{ 'available' if redis_config_available else '' }}"></div>
                        <span>Redis: {{ 'مُحسَّن' if redis_config_available else 'أساسي' }}</span>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <div class="card-title">الإشارات المفتوحة</div>
                    <div class="card-icon">📈</div>
                </div>
                {% if open_signals %}
                    {% for symbol, signal in open_signals.items() %}
                    <div class="signal-item">
                        <div class="item-header">
                            <div class="item-title">{{ symbol }}</div>
                            <div class="item-time">{{ signal.get('timestamp', '')[:16] if signal.get('timestamp') else '' }}</div>
                        </div>
                        <div class="item-content">
                            الدخول: {{ "%.4f"|format(signal.entry_price) if signal.entry_price else 'N/A' }} | 
                            الهدف: {{ "%.4f"|format(signal.target_price) if signal.target_price else 'N/A' }} | 
                            وقف الخسارة: {{ "%.4f"|format(signal.stop_loss) if signal.stop_loss else 'N/A' }}
                        </div>
                    </div>
                    {% endfor %}
                {% else %}
                    <div style="text-align: center; padding: 20px; color: #777;">
                        لا توجد إشارات مفتوحة
                    </div>
                {% endif %}
            </div>
            
            <div class="card">
                <div class="card-header">
                    <div class="card-title">الإشعارات الأخيرة</div>
                    <div class="card-icon">🔔</div>
                </div>
                {% if notifications %}
                    {% for notif in notifications %}
                    <div class="notification-item {{ notif.type }}">
                        <div class="item-header">
                            <div class="item-title">{{ notif.type }}</div>
                            <div class="item-time">{{ notif.timestamp[:16] if notif.timestamp else '' }}</div>
                        </div>
                        <div class="item-content">{{ notif.message }}</div>
                    </div>
                    {% endfor %}
                {% else %}
                    <div style="text-align: center; padding: 20px; color: #777;">
                        لا توجد إشعارات
                    </div>
                {% endif %}
            </div>
            
            <div class="card">
                <div class="card-header">
                    <div class="card-title">سجل الرفوض</div>
                    <div class="card-icon">🚫</div>
                </div>
                {% if rejections %}
                    {% for rej in rejections %}
                    <div class="rejection-item">
                        <div class="item-header">
                            <div class="item-title">{{ rej.symbol }}</div>
                            <div class="item-time">{{ rej.timestamp[:16] if rej.timestamp else '' }}</div>
                        </div>
                        <div class="item-content">{{ rej.reason }}</div>
                    </div>
                    {% endfor %}
                {% else %}
                    <div style="text-align: center; padding: 20px; color: #777;">
                        لا توجد رفضو
                    </div>
                {% endif %}
            </div>
        </div>
        
        <div class="footer">
            <div>بوت التداول الإلكتروني V9.9.2 - فريم 15 دقيقة</div>
            <div class="performance">تم تحميل الصفحة في {{ load_time }} مللي ثانية</div>
        </div>
    </div>
    
    <script>
        function toggleTrading() {
            fetch('/toggle_trading', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert(data.message);
                    location.reload();
                } else {
                    alert('فشل تغيير حالة التداول: ' + data.message);
                }
            })
            .catch(error => {
                alert('خطأ في الاتصال بالخادم: ' + error);
            });
        }
        
        // تحديث الصفحة تلقائياً كل 30 ثانية
        setTimeout(() => {
            location.reload();
        }, 30000);
    </script>
</body>
</html>
"""

SETTINGS_TEMPLATE = """
<!DOCTYPE html>
<html dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>إعدادات البوت</title>
    <style>
        :root {
            --primary-color: #3498db;
            --success-color: #2ecc71;
            --danger-color: #e74c3c;
            --warning-color: #f39c12;
            --dark-color: #2c3e50;
            --light-color: #ecf0f1;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background-color: #f8f9fa;
            color: var(--dark-color);
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background-color: var(--dark-color);
            color: white;
            padding: 15px 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header-title {
            font-size: 24px;
            font-weight: bold;
        }
        
        .toggle-btn {
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        
        .toggle-btn:hover {
            background-color: #2980b9;
        }
        
        .settings-form {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        
        .form-group input, .form-group select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        
        .form-actions {
            display: flex;
            justify-content: space-between;
            margin-top: 20px;
        }
        
        .btn {
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
        }
        
        .btn-primary {
            background-color: var(--primary-color);
            color: white;
        }
        
        .btn-secondary {
            background-color: var(--light-color);
            color: var(--dark-color);
        }
        
        .memory-info {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        .memory-bar {
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .memory-used {
            height: 100%;
            background-color: var(--primary-color);
            width: 0%;
            transition: width 0.5s;
        }
        
        .redis-status {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 10px;
            padding: 10px;
            border-radius: 5px;
            background-color: #f8f9fa;
        }
        
        .redis-status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: var(--danger-color);
        }
        
        .redis-status-dot.available {
            background-color: var(--success-color);
        }
        
        .redis-status-text {
            font-weight: bold;
        }
        
        @media (max-width: 768px) {
            header {
                flex-direction: column;
                gap: 10px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-title">إعدادات البوت</div>
            <a href="/" class="toggle-btn">العودة للرئيسية</a>
        </header>
        
        <div class="redis-status">
            <div class="redis-status-dot {{ 'available' if redis_config_available else '' }}"></div>
            <div class="redis-status-text">
                حالة Redis: {{ 'مُحسَّنة (إعدادات متاحة)' if redis_config_available else 'أساسية (إعدادات غير متاحة)' }}
            </div>
        </div>
        
        <div class="memory-info">
            <h3>حالة الذاكرة</h3>
            <div>الاستخدام الحالي: <span id="memory-percent">0%</span></div>
            <div class="memory-bar">
                <div class="memory-used" id="memory-bar"></div>
            </div>
        </div>
        
        <div class="settings-form">
            <h3>إعدادات التداول</h3>
            <form id="settings-form">
                <div class="form-group">
                    <label for="risk-per-trade">نسبة المخاطرة للصفقة (%)</label>
                    <input type="number" id="risk-per-trade" name="risk_per_trade" step="0.1" min="0.1" max="5" value="{{ RISK_PER_TRADE_PERCENT }}">
                </div>
                
                <div class="form-group">
                    <label for="buy-confidence">حد الثقة للشراء</label>
                    <input type="number" id="buy-confidence" name="buy_confidence" step="0.01" min="0.5" max="1" value="{{ BUY_CONFIDENCE_THRESHOLD }}">
                </div>
                
                <div class="form-group">
                    <label for="max-trades">الحد الأقصى للصفقات المفتوحة</label>
                    <input type="number" id="max-trades" name="max_trades" min="1" max="10" value="{{ MAX_OPEN_TRADES }}">
                </div>
                
                <div class="form-group">
                    <label for="min-profit">الحد الأدنى للربح (%)</label>
                    <input type="number" id="min-profit" name="min_profit" step="0.1" min="0.1" max="5" value="{{ MIN_PROFIT_PERCENT }}">
                </div>
                
                <div class="form-actions">
                    <button type="button" class="btn btn-secondary" onclick="resetSettings()">إعادة الإعدادات الافتراضية</button>
                    <button type="submit" class="btn btn-primary">حفظ الإعدادات</button>
                </div>
            </form>
        </div>
        
        <div class="settings-form">
            <h3>إعدادات الاستراتيجيات</h3>
            <form id="strategies-form">
                <div class="form-group">
                    <label>
                        <input type="checkbox" name="use_bb_stoch" {{ 'checked' if USE_BB_STOCH_STRATEGY else '' }}>
                        تفعيل استراتيجية BB+Stoch
                    </label>
                </div>
                
                <div class="form-group">
                    <label>
                        <input type="checkbox" name="use_macd_ema" {{ 'checked' if USE_MACD_EMA_STRATEGY else '' }}>
                        تفعيل استراتيجية MACD+EMA
                    </label>
                </div>
                
                <div class="form-group">
                    <label>
                        <input type="checkbox" name="use_ema_rsi" {{ 'checked' if USE_EMA_RSI_STRATEGY else '' }}>
                        تفعيل استراتيجية EMA+RSI
                    </label>
                </div>
                
                <div class="form-group">
                    <label>
                        <input type="checkbox" name="use_pullback" {{ 'checked' if USE_PULLBACK_STRATEGY else '' }}>
                        تفعيل استراتيجية Pullback
                    </label>
                </div>
                
                <div class="form-actions">
                    <button type="submit" class="btn btn-primary">حفظ الإعدادات</button>
                </div>
            </form>
        </div>
    </div>
    
    <script>
        // محاكاة حالة الذاكرة
        document.addEventListener('DOMContentLoaded', function() {
            // في التطبيق الحقيقي، سيتم جلب هذه البيانات من الخادم
            const memoryPercent = 65; // مثال للاستخدام
            document.getElementById('memory-percent').textContent = memoryPercent + '%';
            document.getElementById('memory-bar').style.width = memoryPercent + '%';
        });
        
        // معالجة نموذج الإعدادات
        document.getElementById('settings-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const settings = {};
            
            for (let [key, value] of formData.entries()) {
                settings[key] = value;
            }
            
            // إرسال الإعدادات إلى الخادم
            fetch('/update_settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(settings)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('تم حفظ الإعدادات بنجاح');
                } else {
                    alert('فشل حفظ الإعدادات: ' + data.message);
                }
            })
            .catch(error => {
                alert('خطأ في الاتصال بالخادم: ' + error);
            });
        });
        
        // معالجة نموذج الاستراتيجيات
        document.getElementById('strategies-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const strategies = {};
            
            for (let [key, value] of formData.entries()) {
                strategies[key] = value === 'on';
            }
            
            // إرسال الاستراتيجيات إلى الخادم
            fetch('/update_strategies', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(strategies)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('تم حفظ إعدادات الاستراتيجيات بنجاح');
                } else {
                    alert('فشل حفظ إعدادات الاستراتيجيات: ' + data.message);
                }
            })
            .catch(error => {
                alert('خطأ في الاتصال بالخادم: ' + error);
            });
        });
        
        function resetSettings() {
            if (confirm('هل أنت متأكد من إعادة الإعدادات إلى القيم الافتراضية؟')) {
                fetch('/reset_settings', {
                    method: 'POST'
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('تم إعادة الإعدادات إلى القيم الافتراضية');
                        location.reload();
                    } else {
                        alert('فشل إعادة الإعدادات: ' + data.message);
                    }
                })
                .catch(error => {
                    alert('خطأ في الاتصال بالخادم: ' + error);
                });
            }
        }
    </script>
</body>
</html>
"""

# --- دالة تحديث حالة السوق ---
def update_market_state():
    """تحديث حالة السوق العامة"""
    global current_market_state
    try:
        # جلب بيانات البيتكوين لتحليل السوق
        btc_df = fetch_historical_data(BTC_SYMBOL, '1h', 2)
        if btc_df is None or len(btc_df) < 50:
            return
        
        # حساب المؤشرات الأساسية
        btc_df['ema_50'] = btc_df['close'].ewm(span=50, adjust=False).mean()
        btc_df['ema_200'] = btc_df['close'].ewm(span=200, adjust=False).mean()
        
        tr = pd.concat([btc_df['high'] - btc_df['low'], 
                       (btc_df['high'] - btc_df['close'].shift()).abs(), 
                       (btc_df['low'] - btc_df['close'].shift()).abs()], axis=1).max(axis=1)
        btc_df['atr'] = tr.rolling(14).mean()
        
        plus_dm = np.where((btc_df['high'] - btc_df['high'].shift()) > (btc_df['low'].shift() - btc_df['low']), 
                          btc_df['high'] - btc_df['high'].shift(), 0)
        minus_dm = np.where((btc_df['low'].shift() - btc_df['low']) > (btc_df['high'] - btc_df['high'].shift()), 
                           btc_df['low'].shift() - btc_df['low'], 0)
        plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / btc_df['atr'].replace(0, 1e-9)
        minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / btc_df['atr'].replace(0, 1e-9)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        btc_df['adx'] = dx.rolling(14).mean()
        
        last = btc_df.iloc[-1]
        
        # تحديد نظام السوق
        if last['close'] > last['ema_200'] and last['adx'] > 25:
            regime = "UPTREND"
        elif last['close'] < last['ema_200'] and last['adx'] > 25:
            regime = "DOWNTREND"
        else:
            regime = "RANGING"
        
        # تحديث حالة السوق
        with market_state_lock:
            current_market_state = {
                "overall_regime": regime,
                "trend_details_by_tf": {
                    "1h": {
                        "price_vs_ema200": (last['close'] / last['ema_200']) - 1,
                        "adx": last['adx'],
                        "ema_trend": "BULLISH" if last['close'] > last['ema_200'] else "BEARISH"
                    }
                },
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
        
        logger.info(f"✅ تم تحديث حالة السوق: {regime}")
    except Exception as e:
        logger.error(f"❌ خطأ في تحديث حالة السوق: {e}")

# --- الحلقة الرئيسية ---
def main_loop():
    """الحلقة الرئيسية للبوت"""
    global client, last_market_state_check
    
    # تهيئة الخدمات
    init_db()
    init_redis()
    
    # تحميل الإعدادات من Redis إذا كانت موجودة
    load_settings_from_redis()
    
    # تهيئة عميل Binance
    client = Client(API_KEY, API_SECRET)
    
    # جلب معلومات المنصة والرموز الصالحة
    get_exchange_info_map()
    validated_symbols_to_scan = get_validated_symbols()
    
    # تحميل البيانات الأولية
    load_open_signals_to_cache()
    load_notifications_to_cache()
    
    # بدء Flask في خيط منفصل
    app_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=5000, threaded=True))
    app_thread.daemon = True
    app_thread.start()
    
    logger.info("✅ بدء تشغيل البوت بنجاح")
    send_telegram_message("✅ تم بدء تشغيل بوت التداول V9.9.2 بنجاح")
    
    # الحلقة الرئيسية
    while True:
        try:
            # تحسين الذاكرة بشكل دوري
            if random.random() < 0.1:  # 10% chance in each iteration
                optimize_memory_usage()
            
            # التحقق من حالة السوق كل 5 دقائق
            current_time = time.time()
            if current_time - last_market_state_check > 300:  # 5 minutes
                update_market_state()
                last_market_state_check = current_time
            
            # معالجة العملات في دفعات
            for i in range(0, len(validated_symbols_to_scan), SYMBOL_PROCESSING_BATCH_SIZE):
                batch_symbols = validated_symbols_to_scan[i:i+SYMBOL_PROCESSING_BATCH_SIZE]
                
                for symbol in batch_symbols:
                    try:
                        # التحقق من حالة التداول
                        with trading_status_lock:
                            if not is_trading_enabled:
                                break
                            
                            # التحقق من عدد الصفقات المفتوحة
                            if len(open_signals_cache) >= MAX_OPEN_TRADES:
                                break
                            
                            # تجنب العملات المفتوحة بالفعل
                            if symbol in open_signals_cache:
                                continue
                            
                            # جلب البيانات
                            df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                            if df_15m is None or len(df_15m) < 50:
                                continue
                            
                            df_15m.name = symbol
                            
                            # جلب بيانات البيتكوين للارتباط
                            btc_df = get_btc_data_for_bot()
                            
                            # حساب المؤشرات
                            df_15m = calculate_all_features(df_15m, btc_df)
                            
                            # تطبيق الفلاتر
                            if not check_market_volatility_filter(df_15m):
                                continue
                            
                            if not check_trend_strength_filter(df_15m):
                                continue
                            
                            # التحقق من استراتيجيات الدخول
                            signal_found = False
                            
                            with bb_stoch_strategy_lock:
                                if USE_BB_STOCH_STRATEGY and check_bb_stoch_strategy_revised(df_15m):
                                    signal_found = True
                            
                            if not signal_found:
                                with macd_ema_strategy_lock:
                                    if USE_MACD_EMA_STRATEGY and check_macd_ema_strategy(df_15m):
                                        signal_found = True
                            
                            if not signal_found:
                                with ema_rsi_strategy_lock:
                                    if USE_EMA_RSI_STRATEGY and check_ema_rsi_strategy(df_15m):
                                        signal_found = True
                            
                            if not signal_found:
                                with pullback_strategy_lock:
                                    if USE_PULLBACK_STRATEGY and check_pullback_strategy(df_15m):
                                        signal_found = True
                            
                            # إذا تم العثور على إشارة، قم بإنشاء صفقة
                            if signal_found:
                                # ... منطق إنشاء الصفقة ...
                                pass
                            
                            # تحرير الذاكرة
                            del df_15m
                            if btc_df is not None:
                                del btc_df
                            
                    except Exception as e:
                        logger.error(f"❌ خطأ في معالجة العملة {symbol}: {e}")
                        continue
                
                # تحسين الذاكرة بعد كل دفعة
                gc.collect()
                
                # انتظار قبل الدفعة التالية
                time.sleep(1)
            
            # انتظار قبل الدورة التالية
            time.sleep(10)
            
        except KeyboardInterrupt:
            logger.info("تم إيقاف البوت بواسطة المستخدم")
            send_telegram_message("⚠️ تم إيقاف بوت التداول بواسطة المستخدم")
            break
        except Exception as e:
            logger.error(f"❌ خطأ في الحلقة الرئيسية: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main_loop()