# ملف c4_enhanced_v11.0.py - نسخة V11.0 "Enhanced Trading System"
# --- نسخة معدلة مع استراتيجيات محسنة وإدارة مخاطر متقدمة ---
# هذا الإصدار يحتوي على تحسينات شاملة لجميع الاستراتيجيات مع إضافة استراتيجية جديدة
# ونظام إدارة مخاطر محسّن ونظام خروج متطور.

import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import redis
import traceback
import gc
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException, BinanceRequestException
from flask import Flask, jsonify, render_template_string, request, abort
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timezone, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Tuple
from functools import wraps
import random
import warnings

# --- إعدادات التجاهل واللوجر ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v11.0_enhanced.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV11.0-Enhanced')

# --- مشفر مخصص لأنواع بيانات NumPy والعشرية ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, Decimal): return float(obj)
        if isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat()
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

# --- متغيرات عامة وأقفال ---
is_trading_enabled: bool = False
trading_status_lock = Lock()

# --- المتغيرات القابلة للتعديل ---
RISK_PER_TRADE_PERCENT: float = 0.85
risk_per_trade_lock = Lock()

# --- مفاتيح تفعيل الاستراتيجيات ---
STRATEGY_CONFIG = {
    "BB_Reversal": {"enabled": True, "lock": Lock(), "display_name": "BB Reversal (Enhanced)"},
    "MACD_EMA": {"enabled": True, "lock": Lock(), "display_name": "MACD+EMA (Enhanced)"},
    "SR_Breakout": {"enabled": True, "lock": Lock(), "display_name": "S/R Breakout (Enhanced)"},
    "Triple_Confirmation": {"enabled": True, "lock": Lock(), "display_name": "Triple Confirmation (Enhanced)"},
    "VWAP_Reversal": {"enabled": True, "lock": Lock(), "display_name": "VWAP Reversal (Enhanced)"},
    "Price_Channel": {"enabled": True, "lock": Lock(), "display_name": "Price Channel (New)"},
}

# --- إعدادات الفلاتر القابلة للتعديل ---
FILTER_CONFIG = {
    "ADX_THRESHOLD": {"value": 25, "lock": Lock(), "display_name": "حد مؤشر ADX"},
    "BB_STOCH_VOLUME_MULT": {"value": 1.2, "lock": Lock(), "display_name": "مضاعف فوليوم (BB Reversal)"},
    "MACD_EMA_VOLUME_MULT": {"value": 1.2, "lock": Lock(), "display_name": "مضاعف فوليوم (MACD_EMA)"},
    "SR_BREAKOUT_VOLUME_MULT": {"value": 1.4, "lock": Lock(), "display_name": "مضاعف فوليوم (SR Breakout)"},
    "TRIPLE_CONF_VOLUME_MULT": {"value": 1.2, "lock": Lock(), "display_name": "مضاعف فوليوم (Triple Conf)"},
    "VWAP_VOLUME_MULT": {"value": 1.3, "lock": Lock(), "display_name": "مضاعف فوليوم (VWAP Reversal)"},
    "PRICE_CHANNEL_VOLUME_MULT": {"value": 1.3, "lock": Lock(), "display_name": "مضاعف فوليوم (Price Channel)"},
    "TRIPLE_CONF_MODE": {"value": "strict", "lock": Lock(), "display_name": "وضع (Triple Conf)"}, # 'strict' or 'relaxed'
    "VWAP_REVERSAL_MODE": {"value": "strict", "lock": Lock(), "display_name": "وضع (VWAP Reversal)"}, # 'strict' or 'relaxed'
    "TIME_BASED_EXIT_CANDLES": {"value": 30, "lock": Lock(), "display_name": "إغلاق الصفقة بعد (شمعة)"},
    "SIGNAL_STRENGTH_THRESHOLD": {"value": 0.7, "lock": Lock(), "display_name": "حد قوة الإشارة"},
    "MAX_CORRELATION_THRESHOLD": {"value": 0.7, "lock": Lock(), "display_name": "حد الارتباط بين الأصول"}
}

# --- إعدادات المؤشرات الفنية والإطارات الزمنية ---
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 90
BTC_SYMBOL: str = 'BTCUSDT'
MAX_OPEN_TRADES: int = 5
ATR_PERIOD: int = 14
ADX_PERIOD: int = 14
CACHE_EXPIRATION_MINUTES: int = 15
BATCH_SIZE: int = 50

# --- إعدادات إدارة المخاطر والخروج ---
USE_SMART_EXIT_SYSTEM: bool = True
TAKE_PROFIT_LEVELS = {
    1: {"atr_multiplier": 1.5, "exit_percentage": 0.50},
    2: {"atr_multiplier": 3.0, "exit_percentage": 0.30},
    3: {"atr_multiplier": 5.0, "exit_percentage": 0.20}
}
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_STOP_ACTIVATION_ATR: float = 2.0

# --- متغيرات الحالة والكاش ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
exchange_info_map: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
rejection_logs_cache = []
rejection_logs_lock = Lock()
notifications_cache = []
notifications_lock = Lock()
current_market_state: Dict[str, Any] = {"status": "INITIALIZING"}
market_state_lock = Lock()

# --- قاموس أسباب الرفض باللغة العربية ---
REJECTION_REASONS_AR = {
    "Market Status Filter: BTC Downtrend (5m)": "فلتر السوق: اتجاه البيتكوين هابط (5 دقائق)",
    "Market Status Filter: BTC Downtrend (4h)": "فلتر السوق: اتجاه البيتكوين هابط (4 ساعات)",
    "Market Status Filter: Low Liquidity": "فلتر السوق: سيولة منخفضة",
    "Market Status Filter: High Volatility": "فلتر السوق: تقلبات عالية",
    "Market Status Filter: Weak Market Strength": "فلتر السوق: ضعف قوة السوق",
    "BB Reversal: No Reversal Candle": "انعكاس BB: لم تظهر شمعة انعكاسية",
    "BB Reversal: ADX Filter Failed": "انعكاس BB: فلتر قوة الاتجاه ADX",
    "BB Reversal: Volume Filter Failed": "انعكاس BB: فلتر تأكيد حجم التداول",
    "BB Reversal: RSI Not Oversold": "انعكاس BB: RSI ليس في منطقة تشبع بيعي",
    "BB Reversal: Price Too Far Below BB": "انعكاس BB: السعر بعيد جداً تحت BB",
    "MACD_EMA: RSI Filter Failed": "MACD_EMA: فلتر RSI",
    "MACD_EMA: Trend Filter Failed": "MACD_EMA: فلتر تأكيد الاتجاه",
    "MACD_EMA: MACD Filter Failed": "MACD_EMA: فلتر MACD",
    "MACD_EMA: Volume Filter Failed": "MACD_EMA: فلتر حجم التداول",
    "SR_Breakout: Volume Filter Failed": "SR_Breakout: فلتر حجم التداول",
    "SR_Breakout: RSI Filter Failed": "SR_Breakout: فلتر RSI",
    "SR_Breakout: ADX Filter Failed": "SR_Breakout: فلتر ADX",
    "Triple_Confirmation: Conditions Not Met": "Triple Confirmation: لم تتحقق الشروط",
    "Triple_Confirmation: Volatility Filter Failed": "Triple Confirmation: فلتر التقلبات",
    "VWAP_Reversal: Conditions Not Met": "VWAP Reversal: لم تتحقق الشروط",
    "VWAP_Reversal: RSI Not Oversold": "VWAP Reversal: RSI ليس في منطقة تشبع بيعي",
    "VWAP_Reversal: Trend Not Aligned": "VWAP Reversal: الاتجاه غير متوافق",
    "Price Channel: Volume Filter Failed": "القناة السعرية: فلتر حجم التداول",
    "Price Channel: RSI Filter Failed": "القناة السعرية: فلتر RSI",
    "Price Channel: ADX Filter Failed": "القناة السعرية: فلتر ADX",
    "Insufficient Balance": "الرصيد غير كافٍ",
    "Min Notional Filter": "قيمة الصفقة أقل من الحد الأدنى",
    "Invalid Position Size": "حجم الصفقة غير صالح",
    "Lot Size Adjustment Failed": "فشل تعديل حجم العقد",
    "Portfolio Risk: Max Open Trades": "مخاطر المحفظة: الحد الأقصى للصفقات المفتوحة",
    "Portfolio Risk: Sector Concentration": "مخاطر المحفظة: تركيز قطاعي مفرط",
    "Portfolio Risk: Max Portfolio Risk": "مخاطر المحفظة: تجاوز الحد الأقصى للمخاطرة",
    "Signal Strength Too Low": "قوة الإشارة منخفضة جداً"
}

# --- آلية تنظيم الطلبات المتقدمة (Token Bucket) ---
class RequestThrottler:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_rate = float(refill_rate)
        self.last_refill_time = time.time()
        self.lock = Lock()
        self.total_weight_used_minute = 0
        self.minute_start_time = time.time()

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill_time
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill_time = now
        if now - self.minute_start_time > 60:
            self.total_weight_used_minute = 0
            self.minute_start_time = now

    def consume(self, weight: int) -> None:
        with self.lock:
            self._refill()
            if weight > self.tokens:
                wait_time = (weight - self.tokens) / self.refill_rate
                logger.warning(f"🚦 [Throttler] الوزن المطلوب ({weight}) أعلى من المتاح ({self.tokens:.2f}). الانتظار {wait_time:.2f} ثانية.")
                time.sleep(wait_time)
            self._refill()
            self.tokens -= weight
            self.total_weight_used_minute += weight

throttler = RequestThrottler(capacity=5900, refill_rate=100) 

def rate_limiter(weight=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            throttler.consume(weight)
            try:
                response = func(*args, **kwargs)
                return response
            except (BinanceAPIException, BinanceRequestException) as e:
                if e.status_code in [429, 418]:
                    logger.critical("🚨 [API BAN] تم الوصول إلى حد الطلبات (HTTP 429/418). سيتم الانتظار لمدة 10 دقائق.")
                    send_telegram_message("🚨 *تحذير حظر API!* 🚨\\nتم الوصول إلى حد الطلبات. سيتوقف البوت مؤقتاً.")
                    time.sleep(600)
                raise
        return wrapper
    return decorator

# --- دوال مساعدة ---
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
                        target_price DOUBLE PRECISION, stop_loss DOUBLE PRECISION NOT NULL,
                        status TEXT DEFAULT 'open', closing_price DOUBLE PRECISION, 
                        opened_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        closed_at TIMESTAMP,
                        profit_percentage DOUBLE PRECISION, strategy_name TEXT, signal_details JSONB,
                        current_peak_price DOUBLE PRECISION, is_real_trade BOOLEAN DEFAULT FALSE,
                        quantity DOUBLE PRECISION, original_quantity DOUBLE PRECISION, order_id TEXT, closing_reason TEXT,
                        exit_levels JSONB, candles_since_entry INTEGER DEFAULT 0
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_performance (
                        strategy_name TEXT PRIMARY KEY, total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0, total_pnl_percent DOUBLE PRECISION DEFAULT 0.0
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL, message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS historical_data_cache (
                        symbol_timeframe TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        data JSONB NOT NULL,
                        last_updated TIMESTAMP WITH TIME ZONE NOT NULL
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timeframe ON historical_data_cache (symbol, timeframe);")
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
    
    new_notification = {"timestamp": datetime.now(timezone.utc).isoformat(), "type": notification_type, "message": message}
    with notifications_lock:
        notifications_cache.insert(0, new_notification)
        if len(notifications_cache) > 100: notifications_cache.pop()

    if not check_db_connection() or not conn: return
    try:
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
        rejection_logs_cache.insert(0, {
            "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
            "reason": reason_ar, "details": json.loads(json.dumps(details, cls=NpEncoder)) or {}
        })
        if len(rejection_logs_cache) > 200: rejection_logs_cache.pop()

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

@rate_limiter(weight=10)
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
        return validated
    except Exception as e:
        logger.error(f"❌ [التحقق من الرموز] خطأ: {e}", exc_info=True)
        return []

# --- دوال جلب البيانات وحساب المؤشرات ---
@rate_limiter(weight=1)
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        klines = client.get_historical_klines(symbol, interval, f"{days} day ago UTC")
        if not klines: return None
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(klines, columns=cols)
        numeric_cols = {'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'}
        df = df.astype(numeric_cols)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logger.error(f"❌ [جلب البيانات] خطأ في جلب البيانات التاريخية لـ {symbol} ({interval}): {e}")
        raise

def _df_to_json(df: pd.DataFrame) -> str:
    return df.to_json(orient='split', date_format='iso')

def _json_to_df(json_data: Any) -> pd.DataFrame:
    if not isinstance(json_data, str):
        json_str = json.dumps(json_data)
    else:
        json_str = json_data
    df = pd.read_json(json_str, orient='split')
    df.index = pd.to_datetime(df.index, utc=True)
    return df

def get_data_for_symbol(symbol: str, timeframe: str, days: int) -> Optional[pd.DataFrame]:
    if timeframe == '5m':
        logger.info(f"  -> [{symbol}-{timeframe}] ⚡ جلب بيانات حية (بدون كاش).")
        return fetch_historical_data(symbol, timeframe, days=2)

    if not check_db_connection() or not conn:
        logger.warning("[DB Cache] لا يوجد اتصال بقاعدة البيانات، سيتم الجلب مباشرة من API.")
        return fetch_historical_data(symbol, timeframe, days)

    try:
        with conn.cursor() as cur:
            pk = f"{symbol}_{timeframe}"
            cur.execute("SELECT data, last_updated FROM historical_data_cache WHERE symbol_timeframe = %s", (pk,))
            cache_result = cur.fetchone()
        if cache_result:
            last_updated_time = cache_result['last_updated']
            if (datetime.now(timezone.utc) - last_updated_time) < timedelta(minutes=CACHE_EXPIRATION_MINUTES):
                logger.info(f"  -> [{symbol}-{timeframe}] 💾 استخدام البيانات من كاش قاعدة البيانات.")
                return _json_to_df(cache_result['data'])
            else:
                logger.info(f"  -> [{symbol}-{timeframe}] ⏳ بيانات الكاش منتهية الصلاحية.")
    except Exception as e:
        logger.error(f"❌ [DB Cache] خطأ أثناء قراءة الكاش لـ {symbol}-{timeframe}: {e}")
        if conn: conn.rollback()

    logger.info(f"  -> [{symbol}-{timeframe}] 🌐 جلب بيانات جديدة من المنصة.")
    try:
        df = fetch_historical_data(symbol, timeframe, days)
        if df is not None and not df.empty:
            json_data = _df_to_json(df)
            with conn.cursor() as cur:
                pk = f"{symbol}_{timeframe}"
                cur.execute("""
                    INSERT INTO historical_data_cache (symbol_timeframe, symbol, timeframe, data, last_updated)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (symbol_timeframe) DO UPDATE SET
                        data = EXCLUDED.data,
                        last_updated = EXCLUDED.last_updated;
                """, (pk, symbol, timeframe, json_data, datetime.now(timezone.utc)))
            conn.commit()
            logger.info(f"  -> [{symbol}-{timeframe}] ✅ تم تحديث الكاش في قاعدة البيانات.")
            return df
        return None
    except Exception as e:
        logger.error(f"❌ [DB Cache] فشل جلب وتخزين البيانات لـ {symbol}-{timeframe}: {e}")
        if conn: conn.rollback()
        if isinstance(e, BinanceAPIException) and e.code == -1003:
            raise
        return None

def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    df_calc = df.copy()
    df_calc['ema_50'] = df_calc['close'].ewm(span=50, adjust=False).mean()
    df_calc['ema_200'] = df_calc['close'].ewm(span=200, adjust=False).mean()
    df_calc['volume_sma_20'] = df_calc['volume'].rolling(window=20).mean()
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
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=14 - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=14 - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    bb_period = 20
    df_calc['bb_middle'] = df_calc['close'].rolling(window=bb_period).mean()
    bb_std = df_calc['close'].rolling(window=bb_period).std()
    df_calc['bb_upper'] = df_calc['bb_middle'] + (bb_std * 2)
    df_calc['bb_lower'] = df_calc['bb_middle'] - (bb_std * 2)
    df_calc['bb_width'] = (df_calc['bb_upper'] - df_calc['bb_lower']) / df_calc['bb_middle'].replace(0, 1e-9)
    rsi = df_calc['rsi']
    stoch_rsi_val = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min()).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(3).mean() * 100
    df_calc['stoch_rsi_d'] = df_calc['stoch_rsi_k'].rolling(3).mean()
    exp1 = df_calc['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_calc['close'].ewm(span=26, adjust=False).mean()
    df_calc['macd'] = exp1 - exp2
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()
    q = df_calc['volume']
    p = (df_calc['high'] + df_calc['low'] + df_calc['close']) / 3
    df_calc['vwap'] = (p * q).cumsum() / q.cumsum()
    return df_calc.dropna()

# --- دوال الاستراتيجيات وإدارة الصفقات ---

def is_bullish_reversal_enhanced(df: pd.DataFrame) -> bool:
    """
    نسخة محسّنة للتعرف على أنماط الانعكاس الصاعدة
    """
    if len(df) < 3:
        return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3] if len(df) >= 3 else None
    
    # الشروط الحالية
    is_green_candle = last['close'] > last['open']
    candle_range = last['high'] - last['low']
    body_size = last['close'] - last['open']
    is_strong_body = (body_size / candle_range) >= 0.5 if candle_range > 0 else False
    
    is_bullish_engulfing = (
        is_green_candle and
        prev['close'] < prev['open'] and  # الشمعة السابقة حمراء
        last['close'] > prev['open'] and
        last['open'] < prev['close']
    )
    
    # أنماط إضافية للشموع الانعكاسية
    # نموذج المطرقة (Hammer)
    is_hammer = (
        is_green_candle and
        (last['open'] - last['low']) > 2 * (last['close'] - last['open']) and
        (last['high'] - last['close']) < 0.1 * (last['close'] - last['open'])
    )
    
    # نموذج النجمية الصباحية (Morning Star) - مبسط
    is_morning_star = (
        prev2 and prev2['close'] < prev2['open'] and  # الشمعة الأولى حمراء
        prev['close'] < prev['open'] and  # الشمعة الثانية حمراء وصغيرة
        abs(prev['close'] - prev['open']) < 0.3 * (prev2['close'] - prev2['open']) and
        is_green_candle and  # الشمعة الثالثة خضراء
        last['close'] > (prev2['open'] + prev2['close']) / 2  # تغلق فوق منتصف الشمعة الأولى
    )
    
    # نموذج المطرقة المعكوسة (Inverted Hammer)
    is_inverted_hammer = (
        is_green_candle and
        (last['high'] - last['close']) > 2 * (last['close'] - last['open']) and
        (last['open'] - last['low']) < 0.1 * (last['close'] - last['open'])
    )
    
    return is_strong_body or is_bullish_engulfing or is_hammer or is_morning_star or is_inverted_hammer

def check_bb_reversal_strategy_enhanced(df: pd.DataFrame) -> bool:
    """
    استراتيجية انعكاس بولينجر باند المحسنة مع مرشحات إضافية
    """
    if len(df) < 26: return False  # التأكد من وجود بيانات كافية للمؤشرات

    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # الشرط الأول: لامس السعر أو هبط تحت الحد السفلي لبولينجر باند في الشمعة السابقة
    price_touched_bb = prev['low'] <= prev['bb_lower']
    
    # الشرط الثاني: الشمعة الحالية هي شمعة انعكاسية صاعدة
    reversal_candle_appeared = is_bullish_reversal_enhanced(df)
    
    if not (price_touched_bb and reversal_candle_appeared):
        if price_touched_bb and not reversal_candle_appeared:
             log_rejection(df.name, "BB Reversal: No Reversal Candle", {"details": "السعر لمس BB لكن لم تظهر شمعة انعكاسية"})
        return False
    
    # الحصول على قيم المرشحات
    with FILTER_CONFIG["ADX_THRESHOLD"]["lock"]: adx_thresh = FILTER_CONFIG["ADX_THRESHOLD"]["value"]
    with FILTER_CONFIG["BB_STOCH_VOLUME_MULT"]["lock"]: vol_mult = FILTER_CONFIG["BB_STOCH_VOLUME_MULT"]["value"]
    
    # مرشح قوة الاتجاه ADX
    adx_strong = last['adx'] > adx_thresh
    if not adx_strong:
        log_rejection(df.name, "BB Reversal: ADX Filter Failed", {"adx": f"{last['adx']:.2f}", "threshold": adx_thresh})
        return False
    
    # مرشح حجم التداول
    volume_confirmed = last['volume'] > (last['volume_sma_20'] * vol_mult)
    if not volume_confirmed:
        log_rejection(df.name, "BB Reversal: Volume Filter Failed", {"vol_multiplier": vol_mult})
        return False
    
    # مرشح إضافي: حالة التشبع البيعي لـ RSI
    rsi_oversold = last['rsi'] < 30 or prev['rsi'] < 30
    if not rsi_oversold:
        log_rejection(df.name, "BB Reversal: RSI Not Oversold", {"rsi": f"{last['rsi']:.2f}"})
        return False
    
    # مرشح إضافي: السعر لم يهبط كثيراً تحت الحد السفلي لبولينجر باند (ضمن 2%)
    bb_lower = prev['bb_lower']
    price_below_bb_pct = ((bb_lower - prev['low']) / bb_lower) * 100 if bb_lower > 0 else 0
    if price_below_bb_pct > 2:
        log_rejection(df.name, "BB Reversal: Price Too Far Below BB", {"percent_below": f"{price_below_bb_pct:.2f}%"})
        return False
    
    # إذا تحققت جميع الشروط، فهي إشارة صالحة
    logger.info(f"  -> [{df.name}] ✅ إشارة انعكاس BB محسنة (شمعة انعكاسية بعد ملامسة الحد السفلي).")
    return True

def check_macd_ema_strategy_enhanced(df: pd.DataFrame) -> bool:
    """
    استراتيجية MACD+EMA المحسنة مع مرشحات إضافية
    """
    if len(df) < 201: return False  # التأكد من وجود بيانات كافية لـ EMA200
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # الحصول على قيم المرشحات
    with FILTER_CONFIG["MACD_EMA_VOLUME_MULT"]["lock"]: vol_mult = 1.2  # قيمة افتراضية إذا لم يتم تعريفها
    
    # تأكيد الاتجاه (EMAs)
    trend_bullish = last['ema_50'] > last['ema_200']
    
    # تأكيد MACD
    macd_bullish = (
        last['macd'] > last['macd_signal'] and  # MACD فوق الإشارة
        prev['macd'] <= prev['macd_signal'] and  # تقاطع صاعد
        last['macd'] < 0  # MACD تحت الصفر (شراء في منطقة تشبع بيعي)
    )
    
    # مرشح RSI
    rsi_confirmed = last['rsi'] > 30 and last['rsi'] < 70  # ليس في منطقة تشبع极端
    
    # مرشح حجم التداول
    volume_confirmed = last['volume'] > (last['volume_sma_20'] * vol_mult)
    
    # يجب أن تتحقق جميع الشروط
    if trend_bullish and macd_bullish and rsi_confirmed and volume_confirmed:
        logger.info(f"  -> [{df.name}] ✅ إشارة MACD_EMA محسنة.")
        return True
    
    # تسجيل أسباب الرفض
    if not trend_bullish:
        log_rejection(df.name, "MACD_EMA: Trend Filter Failed")
    elif not macd_bullish:
        log_rejection(df.name, "MACD_EMA: MACD Filter Failed")
    elif not rsi_confirmed:
        log_rejection(df.name, "MACD_EMA: RSI Filter Failed", {"rsi": f"{last['rsi']:.2f}"})
    elif not volume_confirmed:
        log_rejection(df.name, "MACD_EMA: Volume Filter Failed", {"vol_multiplier": vol_mult})
    
    return False

def check_sr_breakout_strategy_enhanced(df: pd.DataFrame) -> bool:
    """
    استراتيجية اختراق الدعم/المقاومة المحسنة مع مرشحات إضافية
    """
    if len(df) < 50: return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # الحصول على قيم المرشحات
    with FILTER_CONFIG["SR_BREAKOUT_VOLUME_MULT"]["lock"]: vol_mult = FILTER_CONFIG["SR_BREAKOUT_VOLUME_MULT"]["value"]
    
    # تحديد مستوى المقاومة (باستخدام أعلى سعر مع تأكيد الحجم)
    resistance_lookback = 20
    resistance_candidates = df.iloc[-resistance_lookback:-1]
    resistance_level = resistance_candidates['high'].max()
    
    # إيجاد الحجم عند مستوى المقاومة
    resistance_candle = resistance_candidates[resistance_candidates['high'] == resistance_level]
    resistance_volume = resistance_candle['volume'].iloc[0] if not resistance_candle.empty else 0
    
    # شروط الاختراق
    breakout = (
        last['close'] > resistance_level and  # أغلق فوق المقاومة
        prev['close'] <= resistance_level and  # الشمعة السابقة أغلقت عند أو تحت المقاومة
        (last['close'] - resistance_level) / resistance_level > 0.005  # على الأقل 0.5% فوق المقاومة
    )
    
    if not breakout:
        return False
    
    # مرشح الحجم (الحجم الحالي يجب أن يكون أعلى بشكل ملحوظ من حجم المقاومة)
    volume_confirmed = last['volume'] > (resistance_volume * vol_mult)
    if not volume_confirmed:
        log_rejection(df.name, "SR_Breakout: Volume Filter Failed", {"vol_multiplier": vol_mult})
        return False
    
    # مرشح الزخم (RSI يجب أن يكون قوياً لكن ليس في منطقة شراء مفرط)
    rsi_confirmed = last['rsi'] > 50 and last['rsi'] < 80
    if not rsi_confirmed:
        log_rejection(df.name, "SR_Breakout: RSI Filter Failed", {"rsi": f"{last['rsi']:.2f}"})
        return False
    
    # مرشح قوة الاتجاه ADX
    with FILTER_CONFIG["ADX_THRESHOLD"]["lock"]: adx_thresh = FILTER_CONFIG["ADX_THRESHOLD"]["value"]
    adx_strong = last['adx'] > adx_thresh
    if not adx_strong:
        log_rejection(df.name, "SR_Breakout: ADX Filter Failed", {"adx": f"{last['adx']:.2f}", "threshold": adx_thresh})
        return False
    
    logger.info(f"  -> [{df.name}] ✅ إشارة اختراق دعم/مقاومة محسنة.")
    return True

def check_triple_confirmation_strategy_enhanced(df: pd.DataFrame) -> bool:
    """
    استراتيجية التأكيد الثلاثي المحسنة مع مرشحات إضافية
    """
    if len(df) < 201: return False
    
    last = df.iloc[-1]
    
    # الحصول على قيم المرشحات
    with FILTER_CONFIG["TRIPLE_CONF_VOLUME_MULT"]["lock"]: vol_mult = FILTER_CONFIG["TRIPLE_CONF_VOLUME_MULT"]["value"]
    with FILTER_CONFIG["TRIPLE_CONF_MODE"]["lock"]: mode = FILTER_CONFIG["TRIPLE_CONF_MODE"]["value"]
    
    # تأكيد الاتجاه (EMAs)
    trend_confirmed = last['ema_50'] > last['ema_200']
    
    # تأكيد الزخم (MACD و RSI)
    macd_bullish = last['macd'] > last['macd_signal']
    rsi_bullish = last['rsi'] > 55 and last['rsi'] < 80  # قوي لكن ليس في منطقة شراء مفرط
    momentum_confirmed = macd_bullish and rsi_bullish
    
    # مرشح زخم إضافي (Stochastic)
    stoch_bullish = last['stoch_rsi_k'] > last['stoch_rsi_d'] and last['stoch_rsi_k'] < 80
    momentum_confirmed = momentum_confirmed and stoch_bullish
    
    # تأكيد حجم التداول
    volume_confirmed = last['volume'] > (last['volume_sma_20'] * vol_mult)
    
    # مرشح التقلبات (ATR يجب أن يكون معقولاً)
    atr_normalized = last['atr'] / last['close']  # ATR كنسبة من السعر
    volatility_confirmed = 0.01 < atr_normalized < 0.05  # بين 1% و 5%
    
    # حساب الشروط المستوفاة حسب الوضع
    conditions = [trend_confirmed, momentum_confirmed, volume_confirmed, volatility_confirmed]
    conditions_met = sum(conditions)
    
    # في الوضع الصارم، يجب تحقيق جميع الشروط الأصلية، بالإضافة إلى التقلبات
    if mode == 'strict':
        required_conditions = 4
    else:  # الوضع المرن
        required_conditions = 3
    
    if conditions_met >= required_conditions:
        logger.info(f"  -> [{df.name}] ✅ إشارة التأكيد الثلاثي محسنة (الوضع: {mode}).")
        return True
    
    # تسجيل أسباب الرفض
    log_rejection(df.name, "Triple_Confirmation: Conditions Not Met", {
        "trend": trend_confirmed,
        "momentum": momentum_confirmed,
        "volume": volume_confirmed,
        "volatility": volatility_confirmed,
        "mode": mode,
        "met": conditions_met,
        "required": required_conditions
    })
    
    return False

def check_vwap_reversal_strategy_enhanced(df: pd.DataFrame) -> bool:
    """
    استراتيجية انعكاس VWAP المحسنة مع مرشحات إضافية
    """
    if len(df) < 21: return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # الحصول على قيم المرشحات
    with FILTER_CONFIG["VWAP_VOLUME_MULT"]["lock"]: vol_mult = FILTER_CONFIG["VWAP_VOLUME_MULT"]["value"]
    with FILTER_CONFIG["VWAP_REVERSAL_MODE"]["lock"]: mode = FILTER_CONFIG["VWAP_REVERSAL_MODE"]["value"]
    
    # شروط انعكاس VWAP
    vwap_reversal = prev['close'] < prev['vwap'] and last['close'] > last['vwap']
    
    # أنماط الشموع المحسنة
    is_bullish_engulfing = prev['close'] < prev['open'] and last['close'] > last['open'] and last['close'] > prev['open']
    is_hammer = (last['close'] > last['open']) and (last['open'] - last['low']) > 2 * (last['close'] - last['open'])
    is_doji = abs(last['close'] - last['open']) < 0.1 * (last['high'] - last['low']) and last['close'] > last['open']
    candle_confirmed = is_bullish_engulfing or is_hammer or is_doji
    
    # مرشح حجم التداول (المقارنة مع المتوسط الحديث)
    recent_volume_avg = df['volume'].iloc[-11:-1].mean()
    volume_confirmed = last['volume'] > (recent_volume_avg * vol_mult)
    
    # مرشح إضافي: حالة التشبع البيعي لـ RSI
    rsi_oversold = last['rsi'] < 40 or prev['rsi'] < 40
    
    # مرشح إضافي: توافق الاتجاه (فريم أعلى)
    # هذا يتطلب بيانات الفريم الأعلى، لذا سنستخدم EMAs كبديل
    trend_aligned = last['ema_50'] > last['ema_200']
    
    # تحديد ما إذا كانت الإشارة مستوفاة حسب الوضع
    passes = False
    if mode == 'strict':
        passes = vwap_reversal and candle_confirmed and volume_confirmed and rsi_oversold and trend_aligned
    else:  # الوضع المرن
        conditions_met = sum([vwap_reversal, candle_confirmed, volume_confirmed, rsi_oversold, trend_aligned])
        passes = conditions_met >= 3
    
    if passes:
        logger.info(f"  -> [{df.name}] ✅ إشارة انعكاس VWAP محسنة (الوضع: {mode}).")
        return True
    
    # تسجيل أسباب الرفض
    log_rejection(df.name, "VWAP_Reversal: Conditions Not Met", {
        "reversal": vwap_reversal,
        "candle": candle_confirmed,
        "volume": volume_confirmed,
        "rsi_oversold": rsi_oversold,
        "trend_aligned": trend_aligned,
        "mode": mode
    })
    
    return False

def check_price_channel_strategy(df: pd.DataFrame) -> bool:
    """
    استراتيجية القناة السعرية القائمة على مبادئ قناة دونشيان (Donchian Channel)
    الدخول عند اختراق السعر للحد العلوي للقناة مع تأكيد الحجم
    """
    if len(df) < 30: return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # حساب قناة دونشيان
    channel_period = 20
    df['upper_channel'] = df['high'].rolling(window=channel_period).max()
    df['lower_channel'] = df['low'].rolling(window=channel_period).min()
    df['channel_middle'] = (df['upper_channel'] + df['lower_channel']) / 2
    
    # شروط اختراق القناة العلوية
    upper_breakout = (
        last['close'] > last['upper_channel'] and  # أغلق فوق القناة العلوية
        prev['close'] <= prev['upper_channel'] and  # الشمعة السابقة أغلقت عند أو تحت القناة العلوية
        (last['close'] - last['upper_channel']) / last['upper_channel'] > 0.005  # على الأقل 0.5% فوق القناة
    )
    
    if not upper_breakout:
        return False
    
    # مرشح الحجم
    vol_mult = 1.3  # قيمة افتراضية
    volume_confirmed = last['volume'] > (df['volume'].iloc[-11:-1].mean() * vol_mult)
    if not volume_confirmed:
        log_rejection(df.name, "Price Channel: Volume Filter Failed", {"vol_multiplier": vol_mult})
        return False
    
    # مرشح RSI (يجب أن يكون قوياً لكن ليس في منطقة شراء مفرط)
    rsi_confirmed = last['rsi'] > 50 and last['rsi'] < 80
    if not rsi_confirmed:
        log_rejection(df.name, "Price Channel: RSI Filter Failed", {"rsi": f"{last['rsi']:.2f}"})
        return False
    
    # مرشح قوة الاتجاه ADX
    with FILTER_CONFIG["ADX_THRESHOLD"]["lock"]: adx_thresh = FILTER_CONFIG["ADX_THRESHOLD"]["value"]
    adx_strong = last['adx'] > adx_thresh
    if not adx_strong:
        log_rejection(df.name, "Price Channel: ADX Filter Failed", {"adx": f"{last['adx']:.2f}", "threshold": adx_thresh})
        return False
    
    logger.info(f"  -> [{df.name}] ✅ إشارة اختراق القناة السعرية.")
    return True

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

def calculate_dynamic_position_size_enhanced(symbol: str, entry_price: float, atr_value: float, signal_strength: float = 1.0) -> Optional[Tuple[Decimal, float]]:
    """
    حساب حجم الصفقة الديناميكي المحسن مع مراعاة قوة الإشارة
    signal_strength: قيمة بين 0.5 و 1.5 تشير إلى قوة الإشارة
    """
    if not client: return None
    try:
        with risk_per_trade_lock: current_risk_percent = RISK_PER_TRADE_PERCENT
        
        # تعديل المخاطرة بناءً على قوة الإشارة
        adjusted_risk_percent = current_risk_percent * signal_strength
        
        balance_response = client.get_asset_balance(asset='USDT')
        available_balance = Decimal(balance_response['free'])
        
        # استخدام نسبة المخاطرة المعدلة
        risk_amount_usdt = available_balance * (Decimal(str(adjusted_risk_percent)) / Decimal('100'))
        
        # حساب مسافة وقف الخسارة بناءً على ATR وقوة الإشارة
        # الإشارات الأقوى يمكن أن يكون لها وقف خسارة أضيق
        sl_multiplier = 2.5 - (signal_strength * 0.5)  # المدى من 2.0 إلى 2.5 بناءً على قوة الإشارة
        sl_distance = Decimal(str(atr_value)) * Decimal(str(sl_multiplier))
        
        actual_stop_loss_price = Decimal(str(entry_price)) - sl_distance
        risk_per_coin = Decimal(str(entry_price)) - actual_stop_loss_price
        
        if risk_per_coin <= 0:
            log_rejection(symbol, "Invalid Position Size", {"details": "Stop loss is at or above entry price based on ATR"})
            return None
        
        initial_quantity = risk_amount_usdt / risk_per_coin
        adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(initial_quantity))
        
        if adjusted_quantity is None or adjusted_quantity <= 0:
            log_rejection(symbol, "Lot Size Adjustment Failed")
            return None
        
        notional_value = adjusted_quantity * Decimal(str(entry_price))
        
        symbol_info = exchange_info_map.get(symbol)
        if symbol_info:
            min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] in ('MIN_NOTIONAL', 'NOTIONAL')), None)
            if min_notional_filter:
                min_notional = Decimal(min_notional_filter.get('minNotional', min_notional_filter.get('notional', '0')))
                if notional_value < min_notional:
                    log_rejection(symbol, "Min Notional Filter", {"value": f"{notional_value:.2f}", "min": f"{min_notional}"})
                    return None
        
        if notional_value > available_balance:
            log_rejection(symbol, "Insufficient Balance", {"required": f"{notional_value:.2f}", "available": f"{available_balance:.2f}"})
            return None
        
        return adjusted_quantity, float(actual_stop_loss_price)
    except Exception as e:
        logger.error(f"❌ [{symbol}] خطأ في حساب حجم الصفقة الديناميكي المحسّن: {e}", exc_info=True)
        return None

@rate_limiter(weight=1)
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

def insert_signal_into_db(signal_data: Dict) -> Optional[Dict]:
    if not check_db_connection() or not conn: return None
    try:
        atr_value = 0
        if USE_SMART_EXIT_SYSTEM:
            entry_price = float(signal_data['entry_price'])
            atr_value = float(signal_data.get('signal_details', {}).get('atr', 0))
            if atr_value > 0:
                exit_levels = {}
                for level, config in TAKE_PROFIT_LEVELS.items():
                    exit_levels[str(level)] = {
                        "target_price": entry_price + (atr_value * config['atr_multiplier']),
                        "exit_percentage": config['exit_percentage'],
                        "is_hit": False
                    }
                signal_data['exit_levels'] = exit_levels
                signal_data['target_price'] = exit_levels[str(len(TAKE_PROFIT_LEVELS))]['target_price']
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, is_real_trade, quantity, original_quantity, order_id, current_peak_price, exit_levels, opened_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *;
            """, (
                signal_data['symbol'], signal_data['entry_price'], signal_data.get('target_price'), signal_data['stop_loss'],
                signal_data['strategy_name'], json.dumps(signal_data['signal_details'], cls=NpEncoder),
                signal_data.get('is_real_trade', False), signal_data.get('quantity'), signal_data.get('quantity'),
                signal_data.get('order_id'), signal_data['entry_price'], json.dumps(signal_data.get('exit_levels'), cls=NpEncoder),
                datetime.now(timezone.utc)
            ))
            saved_signal = cur.fetchone()
            conn.commit()
            logger.info(f"💾 [{signal_data['symbol']}] تم حفظ الإشارة الجديدة في قاعدة البيانات.")
            trade_type = "صفقة حقيقية" if signal_data.get('is_real_trade') else "إشارة ورقية"
            message = (f"🚨 *{trade_type} جديدة*\n\n*{signal_data['symbol']}* | `{signal_data['strategy_name']}`\n\n"
                       f"🔹 *الدخول:* `{signal_data['entry_price']:.4f}`\n🛑 *وقف الخسارة:* `{signal_data['stop_loss']:.4f}`\n\n")
            if 'exit_levels' in signal_data and signal_data['exit_levels']:
                message += "*الأهداف:*\n"
                for level, config in signal_data['exit_levels'].items():
                    message += f"  - الهدف {level}: `{config['target_price']:.4f}`\n"
            send_telegram_message(message)
            return dict(saved_signal)
    except Exception as e:
        logger.error(f"❌ [DB Insert] فشل إدراج الإشارة: {e}", exc_info=True)
        if conn: conn.rollback()
        return None

def update_signal_in_db(signal_id: int, updates: Dict):
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            set_clauses = [sql.SQL("{} = %s").format(sql.Identifier(key)) for key in updates]
            values = [json.dumps(v, cls=NpEncoder) if isinstance(v, dict) else v for v in updates.values()]
            values.append(signal_id)
            query = sql.SQL("UPDATE signals SET {} WHERE id = %s").format(sql.SQL(', ').join(set_clauses))
            cur.execute(query, values)
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [DB Update] فشل تحديث الصفقة {signal_id}: {e}")
        if conn: conn.rollback()

def check_exit_conditions_enhanced(signal_data: Dict, current_price: float, df: pd.DataFrame) -> Tuple[bool, str]:
    """
    شروط الخروج المحسنة مع استراتيجيات متعددة
    Returns: (should_exit, reason)
    """
    symbol = signal_data['symbol']
    entry_price = float(signal_data['entry_price'])
    stop_loss = float(signal_data['stop_loss'])
    strategy_name = signal_data['strategy_name']
    
    # الحصول على بيانات الشمعة الحالية
    last_candle = df.iloc[-1]
    
    # 1. وقف الخسارة
    if current_price <= stop_loss:
        return True, "stop_loss"
    
    # 2. الخروج الزمني
    with FILTER_CONFIG["TIME_BASED_EXIT_CANDLES"]["lock"]: max_candles = FILTER_CONFIG["TIME_BASED_EXIT_CANDLES"]["value"]
    candles_since_entry = signal_data.get('candles_since_entry', 0) + 1
    if candles_since_entry >= max_candles:
        return True, "time_based_exit"
    
    # 3. مستويات جني الأرباح
    if 'exit_levels' in signal_data and signal_data['exit_levels']:
        exit_levels = signal_data['exit_levels']
        all_hit = True
        for level, config in exit_levels.items():
            if not config.get('is_hit', False) and current_price >= config['target_price']:
                config['is_hit'] = True
                # تحديث الإشارة في الكاش وقاعدة البيانات
                signal_data['exit_levels'] = exit_levels
                update_signal_in_db(signal_data['id'], {'exit_levels': exit_levels})
                
                # تنفيذ خروج جزئي إذا كانت صفقة حقيقية
                if signal_data.get('is_real_trade'):
                    exit_percentage = config['exit_percentage']
                    quantity_to_sell = Decimal(str(signal_data['quantity'])) * Decimal(str(exit_percentage))
                    place_order(symbol, Client.SIDE_SELL, quantity_to_sell)
                    
                    # تحديث الكمية المتبقية
                    remaining_quantity = Decimal(str(signal_data['quantity'])) - quantity_to_sell
                    signal_data['quantity'] = float(remaining_quantity)
                    update_signal_in_db(signal_data['id'], {'quantity': float(remaining_quantity)})
            
            if not config.get('is_hit', False):
                all_hit = False
        
        if all_hit:
            return True, "all_tp_hit"
    
    # 4. انعكاس الاتجاه (حسب الاستراتيجية)
    if strategy_name == "BB_Reversal":
        # الخروج إذا لمس السعر الحد العلوي لبولينجر باند وأظهر علامات انعكاس هبوطي
        if last_candle['high'] >= last_candle['bb_upper'] and last_candle['close'] < last_candle['open']:
            return True, "trend_reversal"
    
    elif strategy_name == "MACD_EMA":
        # الخروج إذا تقاطع MACD تحت خط الإشارة
        if last_candle['macd'] < last_candle['macd_signal'] and df.iloc[-2]['macd'] >= df.iloc[-2]['macd_signal']:
            return True, "trend_reversal"
    
    elif strategy_name == "SR_Breakout":
        # الخروج إذا عاد السعر تحت مستوى المقاومة
        resistance_level = df['high'].iloc[-21:-1].max()
        if current_price < resistance_level * 0.995:  # 0.5% تحت المقاومة
            return True, "trend_reversal"
    
    elif strategy_name == "Triple_Confirmation":
        # الخروج إذا انعكس أي من الشروط الرئيسية الثلاثة
        ema_bearish = last_candle['ema_50'] < last_candle['ema_200']
        macd_bearish = last_candle['macd'] < last_candle['macd_signal']
        rsi_overbought = last_candle['rsi'] > 70
        
        if ema_bearish or (macd_bearish and rsi_overbought):
            return True, "trend_reversal"
    
    elif strategy_name == "VWAP_Reversal":
        # الخروج إذا عاد السعر تحت VWAP مع حجم تداول
        if last_candle['close'] < last_candle['vwap'] and last_candle['volume'] > last_candle['volume_sma_20'] * 1.2:
            return True, "trend_reversal"
    
    elif strategy_name == "Price_Channel":
        # الخروج إذا عاد السعر تحت منتصف القناة
        channel_middle = (last_candle['upper_channel'] + last_candle['lower_channel']) / 2
        if current_price < channel_middle:
            return True, "trend_reversal"
    
    # 5. وقف الخسارة المتحرك
    if USE_TRAILING_STOP_LOSS:
        current_peak = float(signal_data.get('current_peak_price', entry_price))
        if current_price > current_peak:
            # تحديث أعلى قمة
            signal_data['current_peak_price'] = current_price
            update_signal_in_db(signal_data['id'], {'current_peak_price': current_price})
        
        # حساب مسافة وقف الخسارة المتحرك
        atr_value = last_candle['atr']
        trailing_stop_distance = atr_value * TRAILING_STOP_ACTIVATION_ATR
        trailing_stop_price = current_peak - trailing_stop_distance
        
        if current_price <= trailing_stop_price:
            return True, "trailing_stop"
    
    # تحديث عدد الشموع
    update_signal_in_db(signal_data['id'], {'candles_since_entry': candles_since_entry})
    
    # لم يتم تحقيق أي شرط من شروط الخروج
    return False, ""

def close_signal(signal_id: int, closing_price: float, reason: str) -> bool:
    with signal_cache_lock:
        signal_to_close, symbol_to_close = None, None
        for symbol, signal_data in open_signals_cache.items():
            if signal_data.get('id') == signal_id:
                signal_to_close, symbol_to_close = signal_data, symbol
                break
    if not signal_to_close: return False
    profit_percentage = ((closing_price - float(signal_to_close['entry_price'])) / float(signal_to_close['entry_price'])) * 100
    update_strategy_performance(signal_to_close['strategy_name'], profit_percentage)
    if signal_to_close.get('is_real_trade'):
        try:
            base_asset = symbol_to_close.replace('USDT', '')
            balance_response = client.get_asset_balance(asset=base_asset)
            actual_free_balance = Decimal(balance_response['free'])
            if actual_free_balance > 0:
                quantity_to_sell = adjust_quantity_to_lot_size(symbol_to_close, float(actual_free_balance))
                if quantity_to_sell and quantity_to_sell > 0:
                    place_order(symbol_to_close, Client.SIDE_SELL, quantity_to_sell)
        except Exception as e:
            logger.error(f"❌ [{symbol_to_close}] خطأ أثناء بيع الرصيد عند الإغلاق: {e}")
    if not check_db_connection() or not conn: return False
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE signals SET status = 'closed', closing_price = %s, closed_at = NOW(), profit_percentage = %s, closing_reason = %s WHERE id = %s;",
                        (closing_price, profit_percentage, reason, signal_id))
        conn.commit()
        with signal_cache_lock:
            if symbol_to_close in open_signals_cache:
                del open_signals_cache[symbol_to_close]
        log_and_notify('info', f"تم الإغلاق: {symbol_to_close} عند {closing_price:.4f}. السبب: {reason}. الربح/الخسارة: {profit_percentage:.2f}%", "TRADE_CLOSED")
        reason_map = {'stop_loss': '🛑 تم ضرب وقف الخسارة', 'all_tp_hit': '✅ تم تحقيق جميع الأهداف', 'manual': ' تم الإغلاق يدوياً', 'time_based_exit': '⏳ تم الإغلاق بسبب انتهاء الوقت', 'trend_reversal': '🔄 انعكاس الاتجاه', 'trailing_stop': '📉 وقف خسارة متحرك'}
        emoji = "✅" if profit_percentage > 0 else "🛑"
        message = (f"{emoji} *إغلاق صفقة {symbol_to_close}*\n\n"
                   f"السبب: *{reason_map.get(reason, reason)}*\n"
                   f"سعر الإغلاق: `{closing_price:.4f}`\n"
                   f"الربح/الخسارة: `{profit_percentage:.2f}%`")
        send_telegram_message(message)
        return True
    except Exception as e:
        logger.error(f"❌ [DB Close] فشل تحديث الصفقة المغلقة: {e}"); conn.rollback(); return False

def update_strategy_performance(strategy_name: str, pnl_percent: float):
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO strategy_performance (strategy_name, total_trades, winning_trades, total_pnl_percent)
                VALUES (%s, 1, %s, %s)
                ON CONFLICT (strategy_name) DO UPDATE SET
                    total_trades = strategy_performance.total_trades + 1,
                    winning_trades = strategy_performance.winning_trades + EXCLUDED.winning_trades,
                    total_pnl_percent = strategy_performance.total_pnl_percent + EXCLUDED.total_pnl_percent;
            """, (strategy_name, 1 if pnl_percent > 0 else 0, pnl_percent))
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [Perf Update] فشل تحديث أداء الاستراتيجية {strategy_name}: {e}")
        if conn: conn.rollback()

def check_market_state_enhanced() -> Dict[str, Any]:
    """
    فحص حالة السوق العام المحسنة مع أطر زمنية متعددة ومؤشرات إضافية
    """
    global current_market_state
    
    with market_state_lock:
        try:
            # الحصول على بيانات BTC على أطر زمنية متعددة
            btc_5m = get_data_for_symbol(BTC_SYMBOL, '5m', days=1)
            btc_1h = get_data_for_symbol(BTC_SYMBOL, '1h', days=3)
            btc_4h = get_data_for_symbol(BTC_SYMBOL, '4h', days=7)
            
            if btc_5m is None or btc_1h is None or btc_4h is None:
                current_market_state = {"status": "DATA_UNAVAILABLE"}
                return current_market_state
            
            # حساب المؤشرات لكل إطار زمني
            btc_5m = calculate_all_features(btc_5m)
            btc_1h = calculate_all_features(btc_1h)
            btc_4h = calculate_all_features(btc_4h)
            
            # الحصول على أحدث البيانات
            last_5m = btc_5m.iloc[-1]
            last_1h = btc_1h.iloc[-1]
            last_4h = btc_4h.iloc[-1]
            
            # تحديد الاتجاه على كل إطار زمني
            btc_5m_trend = "bullish" if last_5m['ema_50'] > last_5m['ema_200'] else "bearish"
            btc_1h_trend = "bullish" if last_1h['ema_50'] > last_1h['ema_200'] else "bearish"
            btc_4h_trend = "bullish" if last_4h['ema_50'] > last_4h['ema_200'] else "bearish"
            
            # فحص تقلبات السوق
            btc_1h_atr_pct = (last_1h['atr'] / last_1h['close']) * 100
            volatility_status = "high" if btc_1h_atr_pct > 3.0 else "normal" if btc_1h_atr_pct > 1.0 else "low"
            
            # فحص قوة السوق
            btc_1h_rsi = last_1h['rsi']
            strength_status = "strong" if 40 < btc_1h_rsi < 70 else "weak"
            
            # تقييم عام لحالة السوق
            if btc_4h_trend == "bearish" or (btc_1h_trend == "bearish" and btc_5m_trend == "bearish"):
                overall_status = "BEARISH"
            elif btc_4h_trend == "bullish" and (btc_1h_trend == "bullish" or btc_5m_trend == "bullish"):
                overall_status = "BULLISH"
            else:
                overall_status = "NEUTRAL"
            
            # فحص إضافي للسيولة
            try:
                btc_volume_24h = float(client.get_ticker(symbol=BTC_SYMBOL)['quoteVolume'])
                liquidity_status = "high" if btc_volume_24h > 20000000000 else "normal" if btc_volume_24h > 10000000000 else "low"
            except:
                liquidity_status = "unknown"
            
            current_market_state = {
                "status": overall_status,
                "btc_5m_trend": btc_5m_trend,
                "btc_1h_trend": btc_1h_trend,
                "btc_4h_trend": btc_4h_trend,
                "volatility": volatility_status,
                "strength": strength_status,
                "liquidity": liquidity_status,
                "btc_1h_rsi": round(btc_1h_rsi, 2),
                "btc_1h_atr_pct": round(btc_1h_atr_pct, 2)
            }
            
            return current_market_state
            
        except Exception as e:
            logger.error(f"❌ [Market State] خطأ في تحديد حالة السوق: {e}", exc_info=True)
            current_market_state = {"status": "ERROR", "error": str(e)}
            return current_market_state

def should_skip_signal_based_on_market(market_state: Dict) -> Tuple[bool, str]:
    """
    تحديد ما إذا كان يجب تخطي الإشارة بناءً على حالة السوق
    Returns: (should_skip, reason_key)
    """
    # فحص حالة السوق العامة
    if market_state.get("status") == "BEARISH":
        return True, "Market Status Filter: BTC Downtrend (4h)"
    
    # فحص التقلبات
    if market_state.get("volatility") == "high":
        return True, "Market Status Filter: High Volatility"
    
    # فحص قوة السوق
    if market_state.get("strength") == "weak":
        return True, "Market Status Filter: Weak Market Strength"
    
    # فحص السيولة
    if market_state.get("liquidity") == "low":
        return True, "Market Status Filter: Low Liquidity"
    
    # جميع الشروط متوافقة
    return False, ""

def check_portfolio_risk() -> bool:
    """
    إدارة مخاطر المحفظة المحسنة
    Returns True إذا كان آمناً فتح مراكز جديدة، False خلاف ذلك
    """
    global open_signals_cache
    
    # فحص الحد الأقصى لعدد الصفقات المفتوحة
    with signal_cache_lock:
        open_trades_count = len(open_signals_cache)
    
    if open_trades_count >= MAX_OPEN_TRADES:
        logger.info(f"🚫 [Portfolio Risk] تم الوصول إلى الحد الأقصى لعدد الصفقات المفتوحة ({MAX_OPEN_TRADES}).")
        return False
    
    # فحص مخاطر ارتباط المحفظة
    if open_trades_count > 0:
        try:
            # الحصول على قائمة الرموز المفتوحة
            open_symbols = [signal['symbol'] for signal in open_signals_cache.values()]
            
            # حساب مصفوفة الارتباط (منهجية مبسطة)
            # في التطبيق الفعلي، ستحتاج لجلب البيانات التاريخية وحساب الارتباطات
            with FILTER_CONFIG["MAX_CORRELATION_THRESHOLD"]["lock"]: 
                max_correlation = FILTER_CONFIG["MAX_CORRELATION_THRESHOLD"]["value"]
            
            # للآن، سنستخدم فحصاً بسيطاً يعتمد على أسماء الرموز
            # هذا تبسيط - في الواقع، ستحسب ارتباطات الأسعار الفعلية
            base_assets = [s.replace('USDT', '') for s in open_symbols]
            
            # فحص وجود مراكز كثيرة جداً في نفس القطاع (مبسط)
            sector_counts = {}
            for asset in base_assets:
                # هذا تصنيف قطاعي مبسط جداً
                if asset in ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOT', 'SOL']:
                    sector = 'major'
                elif any(x in asset for x in ['DEFI', 'UNI', 'AAVE', 'COMP', 'CRV']):
                    sector = 'defi'
                elif any(x in asset for x in ['WEB3', 'LINK', 'FIL', 'AR']):
                    sector = 'web3'
                elif any(x in asset for x in ['GAMING', 'SAND', 'MANA', 'AXS']):
                    sector = 'gaming'
                else:
                    sector = 'other'
                
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            # فحص إذا كان أي قطاع لديه مراكز كثيرة جداً
            max_sector_exposure = max(sector_counts.values()) if sector_counts else 0
            if max_sector_exposure >= 3:  # لا أكثر من 3 مراكز في نفس القطاع
                logger.info(f"🚫 [Portfolio Risk] تركيز مفرط في قطاع واحد ({max_sector_exposure} صفقات).")
                return False
            
        except Exception as e:
            logger.error(f"❌ [Portfolio Risk] خطأ في فحص ارتباط المحفظة: {e}")
    
    # فحص مخاطر المحفظة الإجمالية
    try:
        # الحصول على رصيد الحساب
        balance = client.get_asset_balance(asset='USDT')
        available_balance = float(balance['free'])
        
        # حساب إجمالي المخاطرة في المراكز المفتوحة
        total_risk_usdt = 0
        for signal in open_signals_cache.values():
            entry_price = float(signal['entry_price'])
            stop_loss = float(signal['stop_loss'])
            quantity = float(signal['quantity'])
            risk_per_coin = entry_price - stop_loss
            position_risk = risk_per_coin * quantity
            total_risk_usdt += position_risk
        
        # حساب إجمالي قيمة المحفظة (تقدير)
        total_portfolio_value = available_balance + sum(
            float(signal['quantity']) * float(current_prices.get(signal['symbol'], signal['entry_price']))
            for signal in open_signals_cache.values()
        )
        
        # حساب نسبة المخاطرة
        risk_percentage = (total_risk_usdt / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0
        
        # فحص إذا تجاوزت المخاطرة الحد الأقصى المسموح به
        max_portfolio_risk = 15.0  # كحد أقصى 15% من المحفظة في حالة مخاطرة
        if risk_percentage > max_portfolio_risk:
            logger.info(f"🚫 [Portfolio Risk] تجاوز الحد الأقصى لمخاطرة المحفظة ({risk_percentage:.2f}% > {max_portfolio_risk}%).")
            return False
        
    except Exception as e:
        logger.error(f"❌ [Portfolio Risk] خطأ في فحص مخاطرة المحفظة: {e}")
    
    # جميع الفحوصات نجحت
    return True

def generate_signals_for_symbol(symbol: str) -> List[Dict]:
    """
    توليد الإشارات المحسن مع فحص حالة السوق وتقييم قوة الإشارة
    """
    signals = []
    
    # فحص حالة السوق أولاً
    market_state = check_market_state_enhanced()
    should_skip, skip_reason = should_skip_signal_based_on_market(market_state)
    
    if should_skip:
        log_rejection(symbol, skip_reason)
        return signals
    
    # الحصول على بيانات الرمز
    df = get_data_for_symbol(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
    if df is None or df.empty:
        logger.warning(f"  -> [{symbol}] لا توجد بيانات متاحة.")
        return signals
    
    df = calculate_all_features(df)
    df.name = symbol  # تعيين الاسم للتسجيل
    
    # فحص كل استراتيجية
    strategy_results = {}
    
    # استراتيجية BB Reversal
    with STRATEGY_CONFIG["BB_Reversal"]["lock"]:
        if STRATEGY_CONFIG["BB_Reversal"]["enabled"]:
            strategy_results["BB_Reversal"] = check_bb_reversal_strategy_enhanced(df)
    
    # استراتيجية MACD_EMA
    with STRATEGY_CONFIG["MACD_EMA"]["lock"]:
        if STRATEGY_CONFIG["MACD_EMA"]["enabled"]:
            strategy_results["MACD_EMA"] = check_macd_ema_strategy_enhanced(df)
    
    # استراتيجية SR_Breakout
    with STRATEGY_CONFIG["SR_Breakout"]["lock"]:
        if STRATEGY_CONFIG["SR_Breakout"]["enabled"]:
            strategy_results["SR_Breakout"] = check_sr_breakout_strategy_enhanced(df)
    
    # استراتيجية Triple_Confirmation
    with STRATEGY_CONFIG["Triple_Confirmation"]["lock"]:
        if STRATEGY_CONFIG["Triple_Confirmation"]["enabled"]:
            strategy_results["Triple_Confirmation"] = check_triple_confirmation_strategy_enhanced(df)
    
    # استراتيجية VWAP_Reversal
    with STRATEGY_CONFIG["VWAP_Reversal"]["lock"]:
        if STRATEGY_CONFIG["VWAP_Reversal"]["enabled"]:
            strategy_results["VWAP_Reversal"] = check_vwap_reversal_strategy_enhanced(df)
    
    # استراتيجية القناة السعرية (جديدة)
    with STRATEGY_CONFIG["Price_Channel"]["lock"]:
        if STRATEGY_CONFIG["Price_Channel"]["enabled"]:
            strategy_results["Price_Channel"] = check_price_channel_strategy(df)
    
    # حساب عدد الاستراتيجيات المتوافقة
    active_strategies = sum(1 for strategy, enabled in STRATEGY_CONFIG.items() if enabled)
    agreement_count = sum(1 for result in strategy_results.values() if result)
    
    # المتابعة فقط إذا أعطت استراتيجية واحدة على الأقل إشارة
    if agreement_count == 0:
        return signals
    
    # حساب قوة الإشارة بناءً على التوافق وحالة السوق
    signal_strength = agreement_count / active_strategies
    
    # تعديل قوة الإشارة بناءً على حالة السوق
    if market_state.get("status") == "BULLISH":
        signal_strength *= 1.2  # تعزيز في السوق الصاعد
    elif market_state.get("status") == "NEUTRAL":
        signal_strength *= 1.0  # محايد في السوق المحايد
    else:
        signal_strength *= 0.8  # تقليل في السوق الهابط (رغم أننا لا يجب أن نصل هنا بسبب الفحص المبكر)
    
    # التعديل بناءً على التقلبات
    if market_state.get("volatility") == "high":
        signal_strength *= 0.8  # تقليل في حالة التقلبات العالية
    elif market_state.get("volatility") == "low":
        signal_strength *= 1.1  # تعزيز طفيف في حالة التقلبات المنخفضة
    
    # تحديد قوة الإشارة بين 0.5 و 1.5
    signal_strength = max(0.5, min(1.5, signal_strength))
    
    # فحص قوة الإشارة
    with FILTER_CONFIG["SIGNAL_STRENGTH_THRESHOLD"]["lock"]: min_strength = FILTER_CONFIG["SIGNAL_STRENGTH_THRESHOLD"]["value"]
    if signal_strength < min_strength:
        log_rejection(symbol, "Signal Strength Too Low", {"strength": signal_strength, "threshold": min_strength})
        return signals
    
    # فحص مخاطر المحفظة
    if not check_portfolio_risk():
        return signals
    
    # الحصول على أحدث سعر
    entry_price = float(df.iloc[-1]['close'])
    atr_value = float(df.iloc[-1]['atr'])
    
    # حساب حجم الصفقة بناءً على قوة الإشارة
    position_result = calculate_dynamic_position_size_enhanced(symbol, entry_price, atr_value, signal_strength)
    
    if position_result is None:
        return signals
    
    quantity, stop_loss = position_result
    
    # إنشاء بيانات الإشارة
    signal_data = {
        "symbol": symbol,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "strategy_name": ", ".join([name for name, result in strategy_results.items() if result]),
        "signal_details": {
            "strategies": {name: result for name, result in strategy_results.items()},
            "signal_strength": signal_strength,
            "market_state": market_state,
            "atr": atr_value,
            "rsi": float(df.iloc[-1]['rsi']),
            "adx": float(df.iloc[-1]['adx']),
            "agreement_count": agreement_count,
            "active_strategies": active_strategies
        },
        "quantity": float(quantity),
        "is_real_trade": is_trading_enabled
    }
    
    signals.append(signal_data)
    
    return signals

def process_open_signals():
    """
    معالجة الإشارات المفتوحة والتحقق من شروط الخروج
    """
    if not check_db_connection() or not conn or not redis_client:
        return
    
    try:
        # الحصول على الأسعار الحالية من Redis
        current_prices = redis_client.hgetall("crypto_bot_prices")
        
        # الحصول على الإشارات المفتوحة من قاعدة البيانات
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status = 'open';")
            open_signals_db = cur.fetchall()
        
        # تحديث الكاش
        with signal_cache_lock:
            open_signals_cache.clear()
            for signal in open_signals_db:
                signal_dict = dict(signal)
                if 'exit_levels' in signal_dict and isinstance(signal_dict['exit_levels'], str):
                    signal_dict['exit_levels'] = json.loads(signal_dict['exit_levels'])
                open_signals_cache[signal_dict['symbol']] = signal_dict
        
        # فحص كل إشارة مفتوحة
        for symbol, signal_data in open_signals_cache.items():
            try:
                # الحصول على السعر الحالي
                current_price_str = current_prices.get(symbol)
                if current_price_str is None:
                    continue
                
                current_price = float(current_price_str)
                
                # الحصول على بيانات الرمز
                df = get_data_for_symbol(symbol, SIGNAL_GENERATION_TIMEFRAME, days=2)
                if df is None or df.empty:
                    continue
                
                df = calculate_all_features(df)
                
                # فحص شروط الخروج
                should_exit, reason = check_exit_conditions_enhanced(signal_data, current_price, df)
                
                if should_exit:
                    close_signal(signal_data['id'], current_price, reason)
                
            except Exception as e:
                logger.error(f"❌ [Process Signals] خطأ في معالجة الإشارة لـ {symbol}: {e}")
    
    except Exception as e:
        logger.error(f"❌ [Process Signals] خطأ عام في معالجة الإشارات: {e}", exc_info=True)

def scan_and_generate_signals():
    """
    مسح العملات وتوليد الإشارات الجديدة
    """
    global validated_symbols_to_scan
    
    if not check_db_connection() or not conn or not redis_client:
        logger.warning("⚠️ [Scan] الخدمات غير جاهزة، سيتم تخطي المسح.")
        return
    
    logger.info("🔍 [Scan] بدء مسح العملات لتوليد الإشارات...")
    
    # الحصول على قائمة العملات الصالحة
    if not validated_symbols_to_scan:
        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan:
            logger.error("❌ [Scan] لا توجد عملات صالحة للمسح.")
            return
    
    # معالجة الإشارات المفتوحة أولاً
    process_open_signals()
    
    # مسح العملات على دفعات
    new_signals = []
    for i in range(0, len(validated_symbols_to_scan), BATCH_SIZE):
        batch_symbols = validated_symbols_to_scan[i:i+BATCH_SIZE]
        
        for symbol in batch_symbols:
            try:
                # تخطي العملات التي لديها إشارات مفتوحة بالفعل
                with signal_cache_lock:
                    if symbol in open_signals_cache:
                        continue
                
                # توليد الإشارات للعملة
                signals = generate_signals_for_symbol(symbol)
                
                if signals:
                    new_signals.extend(signals)
                    
                    # حفظ الإشارة في قاعدة البيانات
                    for signal_data in signals:
                        saved_signal = insert_signal_into_db(signal_data)
                        if saved_signal:
                            with signal_cache_lock:
                                open_signals_cache[symbol] = saved_signal
                
                # تأخير صغير لتجنب الحد الأقصى للطلبات
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ [Scan] خطأ في مسح {symbol}: {e}", exc_info=True)
    
    logger.info(f"✅ [Scan] اكتمل المسح. تم توليد {len(new_signals)} إشارة جديدة.")

def price_stream_processor(msg):
    """
    معالج تدفق الأسعار من WebSocket
    """
    if msg['e'] == '24hrTicker':
        symbol = msg['s']
        last_price = msg['c']
        
        # تحديث الأسعار في Redis
        if redis_client:
            try:
                redis_client.hset("crypto_bot_prices", symbol, last_price)
            except Exception as e:
                logger.error(f"❌ [WebSocket] خطأ في تحديث سعر {symbol} في Redis: {e}")

def start_websocket_manager():
    """
    بدء مدير WebSocket للاستماع إلى تدفق الأسعار
    """
    if not client or not validated_symbols_to_scan:
        return
    
    logger.info("🔌 [WebSocket] بدء مدير WebSocket للاستماع إلى تدفق الأسعار...")
    
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    
    # الاشتراك في تدفق الأسعار لجميع العملات
    symbols_str = '/'.join(validated_symbols_to_scan)
    twm.start_ticker_socket(callback=price_stream_processor)
    
    return twm

# --- واجهة الويب (Flask) ---
app = Flask(__name__)
CORS(app)

@app.before_request
def block_method():
    # حظر طرق معينة لأسباب أمنية
    if request.method in ['PUT', 'DELETE', 'PATCH']:
        abort(403)

def get_dashboard_html_v11_0():
    """
    HTML لوحة التحكم المحسنة
    """
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentinel V11.0 - لوحة تحكم التداول</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-main: #0D1117;
            --bg-card: #161B22;
            --border-color: #30363D;
            --text-primary: #E6EDF3;
            --text-secondary: #848D97;
            --accent-blue: #58A6FF;
            --accent-green: #3FB950;
            --accent-red: #F85149;
            --accent-yellow: #D29922;
        }
        body {
            font-family: 'Tajawal', sans-serif;
            background-color: var(--bg-main);
            color: var(--text-primary);
        }
        .card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 0.75rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
        }
        .tab-btn.active {
            border-bottom-color: var(--accent-blue);
            color: var(--accent-blue);
            font-weight: 700;
        }
        input:checked + .toggle-bg {
            background-color: var(--accent-green);
        }
        .progress-bar {
            background-color: #30363d;
            border-radius: 9999px;
            overflow: hidden;
        }
        .progress-bar-inner {
            background-color: var(--accent-blue);
            height: 100%;
            transition: width 0.5s ease-in-out;
        }
        .input-field {
            background-color: var(--bg-main);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            padding: 0.5rem 0.75rem;
            color: var(--text-primary);
        }
        select.input-field {
            -webkit-appearance: none;
            -moz-appearance: none;
            appearance: none;
            background-image: url('data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%23848D97%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22/%3E%3C/svg%3E');
            background-repeat: no-repeat;
            background-position: left 0.75rem center;
            background-size: 0.65em auto;
            padding-left: 2rem;
        }
    </style>
</head>
<body class="p-4 md:p-6">
    <div class="container mx-auto max-w-screen-2xl">
        <header class="mb-6 flex flex-wrap justify-between items-center gap-4">
            <h1 class="text-2xl md:text-3xl font-extrabold">
                <span class="text-blue-400">Sentinel</span> V11.0
                <span class="text-sm font-normal text-gray-400">نظام التداول الآلي المحسّن</span>
            </h1>
            <div class="flex items-center gap-3">
                <div class="text-sm px-3 py-1 rounded-full bg-gray-800" id="market-status">
                    <span class="inline-block w-2 h-2 rounded-full bg-gray-500 mr-2"></span>
                    <span id="market-status-text">جاري التحميل...</span>
                </div>
                <div class="text-sm px-3 py-1 rounded-full bg-gray-800">
                    <span class="inline-block w-2 h-2 rounded-full bg-gray-500 mr-2"></span>
                    <span id="trading-status-text">معطل</span>
                </div>
                <button id="toggle-trading" class="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors">
                    تفعيل التداول
                </button>
            </div>
        </header>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
            <!-- إحصائيات عامة -->
            <div class="card p-5 lg:col-span-1">
                <h2 class="text-xl font-bold mb-4 text-blue-400">إحصائيات عامة</h2>
                <div class="space-y-4">
                    <div>
                        <div class="flex justify-between text-sm mb-1">
                            <span>الرصيد المتاح (USDT)</span>
                            <span id="balance">--</span>
                        </div>
                    </div>
                    <div>
                        <div class="flex justify-between text-sm mb-1">
                            <span>الصفقات المفتوحة</span>
                            <span id="open-trades">--</span>
                        </div>
                        <div class="progress-bar h-2 mt-1">
                            <div id="open-trades-bar" class="progress-bar-inner" style="width: 0%"></div>
                        </div>
                    </div>
                    <div>
                        <div class="flex justify-between text-sm mb-1">
                            <span>وزن API المستخدم</span>
                            <span id="api-weight">--</span>
                        </div>
                        <div class="progress-bar h-2 mt-1">
                            <div id="api-weight-bar" class="progress-bar-inner" style="width: 0%"></div>
                        </div>
                    </div>
                    <div>
                        <div class="flex justify-between text-sm mb-1">
                            <span>حالة السوق</span>
                            <span id="market-state">--</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- إعدادات الاستراتيجيات -->
            <div class="card p-5 lg:col-span-2">
                <h2 class="text-xl font-bold mb-4 text-blue-400">إعدادات الاستراتيجيات</h2>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div class="space-y-3">
                        <div class="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                            <div>
                                <div class="font-medium">BB Reversal</div>
                                <div class="text-xs text-gray-400">انعكاس بولينجر باند</div>
                            </div>
                            <label class="relative inline-flex items-center cursor-pointer">
                                <input type="checkbox" class="sr-only peer" id="strategy-BB_Reversal">
                                <div class="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all toggle-bg peer-checked:bg-green-600"></div>
                            </label>
                        </div>
                        <div class="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                            <div>
                                <div class="font-medium">MACD_EMA</div>
                                <div class="text-xs text-gray-400">تقاطع MACD مع EMA</div>
                            </div>
                            <label class="relative inline-flex items-center cursor-pointer">
                                <input type="checkbox" class="sr-only peer" id="strategy-MACD_EMA">
                                <div class="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all toggle-bg peer-checked:bg-green-600"></div>
                            </label>
                        </div>
                        <div class="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                            <div>
                                <div class="font-medium">SR_Breakout</div>
                                <div class="text-xs text-gray-400">اختراق الدعم/المقاومة</div>
                            </div>
                            <label class="relative inline-flex items-center cursor-pointer">
                                <input type="checkbox" class="sr-only peer" id="strategy-SR_Breakout">
                                <div class="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all toggle-bg peer-checked:bg-green-600"></div>
                            </label>
                        </div>
                    </div>
                    <div class="space-y-3">
                        <div class="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                            <div>
                                <div class="font-medium">Triple_Confirmation</div>
                                <div class="text-xs text-gray-400">التأكيد الثلاثي</div>
                            </div>
                            <label class="relative inline-flex items-center cursor-pointer">
                                <input type="checkbox" class="sr-only peer" id="strategy-Triple_Confirmation">
                                <div class="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all toggle-bg peer-checked:bg-green-600"></div>
                            </label>
                        </div>
                        <div class="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                            <div>
                                <div class="font-medium">VWAP_Reversal</div>
                                <div class="text-xs text-gray-400">انعكاس VWAP</div>
                            </div>
                            <label class="relative inline-flex items-center cursor-pointer">
                                <input type="checkbox" class="sr-only peer" id="strategy-VWAP_Reversal">
                                <div class="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all toggle-bg peer-checked:bg-green-600"></div>
                            </label>
                        </div>
                        <div class="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                            <div>
                                <div class="font-medium">Price_Channel</div>
                                <div class="text-xs text-gray-400">القناة السعرية</div>
                            </div>
                            <label class="relative inline-flex items-center cursor-pointer">
                                <input type="checkbox" class="sr-only peer" id="strategy-Price_Channel">
                                <div class="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all toggle-bg peer-checked:bg-green-600"></div>
                            </label>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- إعدادات الفلاتر -->
        <div class="card p-5 mb-6">
            <h2 class="text-xl font-bold mb-4 text-blue-400">إعدادات الفلاتر</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div>
                    <label class="block text-sm mb-1">حد مؤشر ADX</label>
                    <input type="number" id="filter-ADX_THRESHOLD" class="input-field w-full" step="1" min="10" max="50">
                </div>
                <div>
                    <label class="block text-sm mb-1">مضاعف فوليوم (BB Reversal)</label>
                    <input type="number" id="filter-BB_STOCH_VOLUME_MULT" class="input-field w-full" step="0.1" min="1" max="3">
                </div>
                <div>
                    <label class="block text-sm mb-1">مضاعف فوليوم (MACD_EMA)</label>
                    <input type="number" id="filter-MACD_EMA_VOLUME_MULT" class="input-field w-full" step="0.1" min="1" max="3">
                </div>
                <div>
                    <label class="block text-sm mb-1">مضاعف فوليوم (SR Breakout)</label>
                    <input type="number" id="filter-SR_BREAKOUT_VOLUME_MULT" class="input-field w-full" step="0.1" min="1" max="3">
                </div>
                <div>
                    <label class="block text-sm mb-1">مضاعف فوليوم (Triple Conf)</label>
                    <input type="number" id="filter-TRIPLE_CONF_VOLUME_MULT" class="input-field w-full" step="0.1" min="1" max="3">
                </div>
                <div>
                    <label class="block text-sm mb-1">مضاعف فوليوم (VWAP Reversal)</label>
                    <input type="number" id="filter-VWAP_VOLUME_MULT" class="input-field w-full" step="0.1" min="1" max="3">
                </div>
                <div>
                    <label class="block text-sm mb-1">مضاعف فوليوم (Price Channel)</label>
                    <input type="number" id="filter-PRICE_CHANNEL_VOLUME_MULT" class="input-field w-full" step="0.1" min="1" max="3">
                </div>
                <div>
                    <label class="block text-sm mb-1">وضع (Triple Conf)</label>
                    <select id="filter-TRIPLE_CONF_MODE" class="input-field w-full">
                        <option value="strict">صارم</option>
                        <option value="relaxed">مرن</option>
                    </select>
                </div>
                <div>
                    <label class="block text-sm mb-1">وضع (VWAP Reversal)</label>
                    <select id="filter-VWAP_REVERSAL_MODE" class="input-field w-full">
                        <option value="strict">صارم</option>
                        <option value="relaxed">مرن</option>
                    </select>
                </div>
                <div>
                    <label class="block text-sm mb-1">إغلاق الصفقة بعد (شمعة)</label>
                    <input type="number" id="filter-TIME_BASED_EXIT_CANDLES" class="input-field w-full" step="5" min="10" max="100">
                </div>
                <div>
                    <label class="block text-sm mb-1">حد قوة الإشارة</label>
                    <input type="number" id="filter-SIGNAL_STRENGTH_THRESHOLD" class="input-field w-full" step="0.1" min="0.5" max="1">
                </div>
                <div>
                    <label class="block text-sm mb-1">نسبة المخاطرة للصفقة (%)</label>
                    <input type="number" id="risk-percent" class="input-field w-full" step="0.1" min="0.1" max="5">
                </div>
            </div>
            <div class="mt-4 flex justify-end">
                <button id="save-settings" class="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors">
                    حفظ الإعدادات
                </button>
            </div>
        </div>

        <!-- علامات التبويب -->
        <div class="border-b border-gray-700 mb-4">
            <nav class="-mb-px flex space-x-8">
                <button class="tab-btn py-2 px-1 border-b-2 border-transparent font-medium text-sm" data-tab="signals">
                    الإشارات المفتوحة
                </button>
                <button class="tab-btn py-2 px-1 border-b-2 border-transparent font-medium text-sm" data-tab="performance">
                    أداء الاستراتيجيات
                </button>
                <button class="tab-btn py-2 px-1 border-b-2 border-transparent font-medium text-sm" data-tab="notifications">
                    الإشعارات
                </button>
                <button class="tab-btn py-2 px-1 border-b-2 border-transparent font-medium text-sm" data-tab="rejections">
                    سجل الرفض
                </button>
            </nav>
        </div>

        <!-- محتوى علامات التبويب -->
        <div class="tab-content">
            <!-- الإشارات المفتوحة -->
            <div id="signals-tab" class="tab-pane">
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-700">
                        <thead>
                            <tr>
                                <th class="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">العملة</th>
                                <th class="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">الاستراتيجية</th>
                                <th class="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">سعر الدخول</th>
                                <th class="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">السعر الحالي</th>
                                <th class="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">وقف الخسارة</th>
                                <th class="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">الربح/الخسارة</th>
                                <th class="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">الإجراءات</th>
                            </tr>
                        </thead>
                        <tbody id="signals-table-body" class="divide-y divide-gray-700">
                            <!-- سيتم ملؤها ديناميكياً -->
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- أداء الاستراتيجيات -->
            <div id="performance-tab" class="tab-pane hidden">
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-700">
                        <thead>
                            <tr>
                                <th class="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">الاستراتيجية</th>
                                <th class="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">إجمالي الصفقات</th>
                                <th class="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">الصفقات الرابحة</th>
                                <th class="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">نسبة النجاح</th>
                                <th class="px-4 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">متوسط الربح/الخسارة</th>
                            </tr>
                        </thead>
                        <tbody id="performance-table-body" class="divide-y divide-gray-700">
                            <!-- سيتم ملؤها ديناميكياً -->
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- الإشعارات -->
            <div id="notifications-tab" class="tab-pane hidden">
                <div class="space-y-3" id="notifications-container">
                    <!-- سيتم ملؤها ديناميكياً -->
                </div>
            </div>

            <!-- سجل الرفض -->
            <div id="rejections-tab" class="tab-pane hidden">
                <div class="space-y-3" id="rejections-container">
                    <!-- سيتم ملؤها ديناميكياً -->
                </div>
            </div>
        </div>
    </div>

    <script>
        // متغيرات عامة
        let isTradingEnabled = false;
        let currentTab = 'signals';
        
        // عناصر DOM
        const toggleTradingBtn = document.getElementById('toggle-trading');
        const tradingStatusText = document.getElementById('trading-status-text');
        const marketStatusText = document.getElementById('market-status-text');
        const marketStatus = document.getElementById('market-status');
        const balanceEl = document.getElementById('balance');
        const openTradesEl = document.getElementById('open-trades');
        const openTradesBar = document.getElementById('open-trades-bar');
        const apiWeightEl = document.getElementById('api-weight');
        const apiWeightBar = document.getElementById('api-weight-bar');
        const marketStateEl = document.getElementById('market-state');
        const saveSettingsBtn = document.getElementById('save-settings');
        const tabBtns = document.querySelectorAll('.tab-btn');
        const tabPanes = document.querySelectorAll('.tab-pane');
        
        // تبديل علامات التبويب
        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const tab = btn.dataset.tab;
                
                // تحديث الزر النشط
                tabBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                // إظهار/إخفاء المحتوى
                tabPanes.forEach(pane => {
                    if (pane.id === `${tab}-tab`) {
                        pane.classList.remove('hidden');
                        currentTab = tab;
                        loadTabData(tab);
                    } else {
                        pane.classList.add('hidden');
                    }
                });
            });
        });
        
        // تحميل بيانات علامة التبويب
        function loadTabData(tab) {
            if (tab === 'signals') {
                loadSignals();
            } else if (tab === 'performance') {
                loadPerformance();
            } else if (tab === 'notifications') {
                loadNotifications();
            } else if (tab === 'rejections') {
                loadRejections();
            }
        }
        
        // تحميل البيانات الأولية
        function loadInitialData() {
            // تحميل حالة النظام
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // تحديث حالة التداول
                    isTradingEnabled = data.is_trading_enabled;
                    tradingStatusText.textContent = isTradingEnabled ? 'مفعل' : 'معطل';
                    toggleTradingBtn.textContent = isTradingEnabled ? 'تعطيل التداول' : 'تفعيل التداول';
                    toggleTradingBtn.className = isTradingEnabled ? 
                        'px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors' : 
                        'px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors';
                    
                    // تحديث حالة السوق
                    const marketState = data.market_state;
                    if (marketState.status === 'BULLISH') {
                        marketStatusText.textContent = 'سوق صاعد';
                        marketStatus.className = 'text-sm px-3 py-1 rounded-full bg-green-900';
                        marketStatus.querySelector('span').className = 'inline-block w-2 h-2 rounded-full bg-green-500 mr-2';
                    } else if (marketState.status === 'BEARISH') {
                        marketStatusText.textContent = 'سوق هابط';
                        marketStatus.className = 'text-sm px-3 py-1 rounded-full bg-red-900';
                        marketStatus.querySelector('span').className = 'inline-block w-2 h-2 rounded-full bg-red-500 mr-2';
                    } else if (marketState.status === 'NEUTRAL') {
                        marketStatusText.textContent = 'سوق محايد';
                        marketStatus.className = 'text-sm px-3 py-1 rounded-full bg-yellow-900';
                        marketStatus.querySelector('span').className = 'inline-block w-2 h-2 rounded-full bg-yellow-500 mr-2';
                    } else {
                        marketStatusText.textContent = 'غير متوفر';
                        marketStatus.className = 'text-sm px-3 py-1 rounded-full bg-gray-800';
                        marketStatus.querySelector('span').className = 'inline-block w-2 h-2 rounded-full bg-gray-500 mr-2';
                    }
                    
                    marketStateEl.textContent = `${marketState.status} | التقلبات: ${marketState.volatility} | القوة: ${marketState.strength}`;
                    
                    // تحديث الرصيد
                    balanceEl.textContent = data.usdt_balance !== 'N/A' ? `${parseFloat(data.usdt_balance).toFixed(2)} USDT` : 'غير متوفر';
                    
                    // تحديث الصفقات المفتوحة
                    openTradesEl.textContent = `${data.open_trades_count} / ${data.max_open_trades}`;
                    const openTradesPercent = (data.open_trades_count / data.max_open_trades) * 100;
                    openTradesBar.style.width = `${openTradesPercent}%`;
                    
                    // تحديث وزن API
                    apiWeightEl.textContent = `${data.api_weight} / 6000`;
                    const apiWeightPercent = (data.api_weight / 6000) * 100;
                    apiWeightBar.style.width = `${apiWeightPercent}%`;
                    
                    // تحميل إعدادات الاستراتيجيات والفلاتر
                    const strategies = data.settings.strategies;
                    Object.keys(strategies).forEach(strategy => {
                        const checkbox = document.getElementById(`strategy-${strategy}`);
                        if (checkbox) {
                            checkbox.checked = strategies[strategy].enabled;
                        }
                    });
                    
                    const filters = data.settings.filters;
                    Object.keys(filters).forEach(filter => {
                        const input = document.getElementById(`filter-${filter}`);
                        if (input) {
                            input.value = filters[filter].value;
                        }
                    });
                    
                    // تحديث نسبة المخاطرة
                    document.getElementById('risk-percent').value = data.settings.risk_percent;
                })
                .catch(error => {
                    console.error('Error loading initial data:', error);
                });
            
            // تحميل بيانات علامة التبويب النشطة
            loadTabData(currentTab);
        }
        
        // تحميل الإشارات المفتوحة
        function loadSignals() {
            const tbody = document.getElementById('signals-table-body');
            tbody.innerHTML = '<tr><td colspan="7" class="text-center py-4 text-gray-500">جاري التحميل...</td></tr>';
            
            fetch('/api/signals')
                .then(response => response.json())
                .then(signals => {
                    if (signals.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="7" class="text-center py-4 text-gray-500">لا توجد إشارات مفتوحة</td></tr>';
                        return;
                    }
                    
                    tbody.innerHTML = '';
                    signals.forEach(signal => {
                        const profitClass = signal.profit_percentage >= 0 ? 'text-green-400' : 'text-red-400';
                        const profitSign = signal.profit_percentage >= 0 ? '+' : '';
                        
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td class="px-4 py-3 whitespace-nowrap">
                                <div class="font-medium">${signal.symbol}</div>
                            </td>
                            <td class="px-4 py-3 whitespace-nowrap">
                                <div class="text-sm">${signal.strategy_name}</div>
                            </td>
                            <td class="px-4 py-3 whitespace-nowrap">
                                <div class="text-sm">${parseFloat(signal.entry_price).toFixed(4)}</div>
                            </td>
                            <td class="px-4 py-3 whitespace-nowrap">
                                <div class="text-sm">${signal.current_price ? parseFloat(signal.current_price).toFixed(4) : '--'}</div>
                            </td>
                            <td class="px-4 py-3 whitespace-nowrap">
                                <div class="text-sm">${parseFloat(signal.stop_loss).toFixed(4)}</div>
                            </td>
                            <td class="px-4 py-3 whitespace-nowrap">
                                <div class="text-sm ${profitClass}">${profitSign}${signal.profit_percentage ? signal.profit_percentage.toFixed(2) : '0.00'}%</div>
                            </td>
                            <td class="px-4 py-3 whitespace-nowrap text-sm">
                                <button class="text-red-400 hover:text-red-300" onclick="closeSignal(${signal.id})">
                                    إغلاق
                                </button>
                            </td>
                        `;
                        tbody.appendChild(row);
                    });
                })
                .catch(error => {
                    console.error('Error loading signals:', error);
                    tbody.innerHTML = '<tr><td colspan="7" class="text-center py-4 text-red-500">خطأ في تحميل البيانات</td></tr>';
                });
        }
        
        // تحميل أداء الاستراتيجيات
        function loadPerformance() {
            const tbody = document.getElementById('performance-table-body');
            tbody.innerHTML = '<tr><td colspan="5" class="text-center py-4 text-gray-500">جاري التحميل...</td></tr>';
            
            fetch('/api/performance')
                .then(response => response.json())
                .then(performance => {
                    if (performance.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="5" class="text-center py-4 text-gray-500">لا توجد بيانات أداء</td></tr>';
                        return;
                    }
                    
                    tbody.innerHTML = '';
                    performance.forEach(strategy => {
                        const winRate = strategy.total_trades > 0 ? 
                            ((strategy.winning_trades / strategy.total_trades) * 100).toFixed(1) : 0;
                        const avgPnl = strategy.total_trades > 0 ? 
                            (strategy.total_pnl_percent / strategy.total_trades).toFixed(2) : 0;
                        
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td class="px-4 py-3 whitespace-nowrap">
                                <div class="font-medium">${strategy.strategy_name}</div>
                            </td>
                            <td class="px-4 py-3 whitespace-nowrap">
                                <div class="text-sm">${strategy.total_trades}</div>
                            </td>
                            <td class="px-4 py-3 whitespace-nowrap">
                                <div class="text-sm">${strategy.winning_trades}</div>
                            </td>
                            <td class="px-4 py-3 whitespace-nowrap">
                                <div class="text-sm">${winRate}%</div>
                            </td>
                            <td class="px-4 py-3 whitespace-nowrap">
                                <div class="text-sm ${avgPnl >= 0 ? 'text-green-400' : 'text-red-400'}">${avgPnl >= 0 ? '+' : ''}${avgPnl}%</div>
                            </td>
                        `;
                        tbody.appendChild(row);
                    });
                })
                .catch(error => {
                    console.error('Error loading performance:', error);
                    tbody.innerHTML = '<tr><td colspan="5" class="text-center py-4 text-red-500">خطأ في تحميل البيانات</td></tr>';
                });
        }
        
        // تحميل الإشعارات
        function loadNotifications() {
            const container = document.getElementById('notifications-container');
            container.innerHTML = '<div class="text-center py-4 text-gray-500">جاري التحميل...</div>';
            
            fetch('/api/notifications')
                .then(response => response.json())
                .then(notifications => {
                    if (notifications.length === 0) {
                        container.innerHTML = '<div class="text-center py-4 text-gray-500">لا توجد إشعارات</div>';
                        return;
                    }
                    
                    container.innerHTML = '';
                    notifications.forEach(notification => {
                        const date = new Date(notification.timestamp);
                        const dateStr = date.toLocaleString('ar-EG');
                        
                        const item = document.createElement('div');
                        item.className = 'card p-4';
                        item.innerHTML = `
                            <div class="flex justify-between items-start">
                                <div>
                                    <div class="font-medium">${notification.message}</div>
                                    <div class="text-xs text-gray-400 mt-1">${dateStr}</div>
                                </div>
                                <div class="text-xs px-2 py-1 rounded-full ${
                                    notification.type === 'error' ? 'bg-red-900 text-red-300' :
                                    notification.type === 'warning' ? 'bg-yellow-900 text-yellow-300' :
                                    notification.type === 'success' ? 'bg-green-900 text-green-300' :
                                    'bg-blue-900 text-blue-300'
                                }">
                                    ${notification.type}
                                </div>
                            </div>
                        `;
                        container.appendChild(item);
                    });
                })
                .catch(error => {
                    console.error('Error loading notifications:', error);
                    container.innerHTML = '<div class="text-center py-4 text-red-500">خطأ في تحميل البيانات</div>';
                });
        }
        
        // تحميل سجل الرفض
        function loadRejections() {
            const container = document.getElementById('rejections-container');
            container.innerHTML = '<div class="text-center py-4 text-gray-500">جاري التحميل...</div>';
            
            fetch('/api/rejection_logs')
                .then(response => response.json())
                .then(rejections => {
                    if (rejections.length === 0) {
                        container.innerHTML = '<div class="text-center py-4 text-gray-500">لا توجد سجلات رفض</div>';
                        return;
                    }
                    
                    container.innerHTML = '';
                    rejections.forEach(rejection => {
                        const date = new Date(rejection.timestamp);
                        const dateStr = date.toLocaleString('ar-EG');
                        
                        const item = document.createElement('div');
                        item.className = 'card p-4';
                        item.innerHTML = `
                            <div class="flex justify-between items-start">
                                <div>
                                    <div class="font-medium">${rejection.symbol}</div>
                                    <div class="text-sm text-gray-300 mt-1">${rejection.reason}</div>
                                    ${rejection.details && Object.keys(rejection.details).length > 0 ? `
                                        <div class="text-xs text-gray-400 mt-2">
                                            <details>
                                                <summary class="cursor-pointer">التفاصيل</summary>
                                                <pre class="mt-2 text-xs bg-gray-800 p-2 rounded overflow-x-auto">${JSON.stringify(rejection.details, null, 2)}</pre>
                                            </details>
                                        </div>
                                    ` : ''}
                                </div>
                                <div class="text-xs text-gray-400">${dateStr}</div>
                            </div>
                        `;
                        container.appendChild(item);
                    });
                })
                .catch(error => {
                    console.error('Error loading rejections:', error);
                    container.innerHTML = '<div class="text-center py-4 text-red-500">خطأ في تحميل البيانات</div>';
                });
        }
        
        // تبديل حالة التداول
        toggleTradingBtn.addEventListener('click', () => {
            fetch('/api/trading/toggle', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    isTradingEnabled = !isTradingEnabled;
                    tradingStatusText.textContent = isTradingEnabled ? 'مفعل' : 'معطل';
                    toggleTradingBtn.textContent = isTradingEnabled ? 'تعطيل التداول' : 'تفعيل التداول';
                    toggleTradingBtn.className = isTradingEnabled ? 
                        'px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors' : 
                        'px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors';
                }
            })
            .catch(error => {
                console.error('Error toggling trading:', error);
            });
        });
        
        // حفظ الإعدادات
        saveSettingsBtn.addEventListener('click', () => {
            const settings = {
                risk_percent: parseFloat(document.getElementById('risk-percent').value),
                strategies: {},
                filters: {}
            };
            
            // جمع إعدادات الاستراتيجيات
            document.querySelectorAll('[id^="strategy-"]').forEach(input => {
                const strategy = input.id.replace('strategy-', '');
                settings.strategies[strategy] = input.checked;
            });
            
            // جمع إعدادات الفلاتر
            document.querySelectorAll('[id^="filter-"]').forEach(input => {
                const filter = input.id.replace('filter-', '');
                if (input.type === 'number') {
                    settings.filters[filter] = parseFloat(input.value);
                } else {
                    settings.filters[filter] = input.value;
                }
            });
            
            fetch('/api/settings/update', {
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
                    // إعادة تحميل البيانات
                    loadInitialData();
                } else {
                    alert(`خطأ: ${data.message}`);
                }
            })
            .catch(error => {
                console.error('Error saving settings:', error);
                alert('خطأ في حفظ الإعدادات');
            });
        });
        
        // إغلاق إشارة
        function closeSignal(signalId) {
            if (confirm('هل أنت متأكد من إغلاق هذه الصفقة؟')) {
                fetch(`/api/signals/close/${signalId}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        // إعادة تحميل الإشارات
                        loadSignals();
                    } else {
                        alert(`خطأ: ${data.message}`);
                    }
                })
                .catch(error => {
                    console.error('Error closing signal:', error);
                    alert('خطأ في إغلاق الصفقة');
                });
            }
        }
        
        // تحديث البيانات بشكل دوري
        setInterval(() => {
            loadInitialData();
        }, 30000); // كل 30 ثانية
        
        // تحميل البيانات الأولية
        loadInitialData();
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(get_dashboard_html_v11_0())

@app.route('/api/status')
def get_status():
    with market_state_lock: state_copy = dict(current_market_state)
    with trading_status_lock: is_enabled = is_trading_enabled
    with throttler.lock: weight = throttler.total_weight_used_minute
    with signal_cache_lock: open_trades = len(open_signals_cache)
    usdt_balance = None
    if client:
        try: usdt_balance = float(client.get_asset_balance(asset='USDT')['free'])
        except: usdt_balance = 'N/A'
    with risk_per_trade_lock: risk = RISK_PER_TRADE_PERCENT
    strategy_settings = {key: {"enabled": config['enabled'], "display_name": config['display_name']} for key, config in STRATEGY_CONFIG.items()}
    filter_settings = {}
    for key, config in FILTER_CONFIG.items():
        with config['lock']:
            filter_settings[key] = {"value": config['value'], "display_name": config['display_name']}
    return jsonify({
        "market_state": state_copy, 
        "is_trading_enabled": is_enabled, 
        "usdt_balance": usdt_balance,
        "api_weight": weight, 
        "open_trades_count": open_trades, 
        "max_open_trades": MAX_OPEN_TRADES,
        "settings": {
            "risk_percent": risk, 
            "strategies": strategy_settings, 
            "filters": filter_settings
        }
    })

@app.route('/api/signals')
def get_signals():
    if not (check_db_connection() and redis_client): return jsonify([]), 500
    try:
        current_prices = redis_client.hgetall("crypto_bot_prices")
        with signal_cache_lock: signals_copy = list(open_signals_cache.values())
        for signal in signals_copy:
            current_price = current_prices.get(signal['symbol'])
            if current_price:
                signal['current_price'] = current_price
                signal['profit_percentage'] = ((float(current_price) - float(signal['entry_price'])) / float(signal['entry_price'])) * 100
        return jsonify(signals_copy)
    except Exception as e:
        logger.error(f"❌ [API Signals] خطأ: {e}"); return jsonify([]), 500

@app.route('/api/performance')
def get_performance():
    if not check_db_connection() or not conn: return jsonify([]), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM strategy_performance ORDER BY total_pnl_percent DESC;")
            return jsonify([dict(row) for row in cur.fetchall()])
    except Exception as e:
        logger.error(f"❌ [API Performance] خطأ: {e}"); return jsonify([]), 500

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
        return jsonify({"success": True, "message": f"Trading status set to {status_msg}"})

@app.route('/api/settings/update', methods=['POST'])
def update_settings():
    try:
        data = request.get_json()
        with risk_per_trade_lock: 
            global RISK_PER_TRADE_PERCENT
            RISK_PER_TRADE_PERCENT = float(data.get('risk_percent', RISK_PER_TRADE_PERCENT))
        strategies_data = data.get('strategies', {})
        for key, is_enabled in strategies_data.items():
            if key in STRATEGY_CONFIG:
                with STRATEGY_CONFIG[key]['lock']: 
                    STRATEGY_CONFIG[key]['enabled'] = bool(is_enabled)
        filters_data = data.get('filters', {})
        for key, value in filters_data.items():
            if key in FILTER_CONFIG:
                with FILTER_CONFIG[key]['lock']:
                    if isinstance(FILTER_CONFIG[key]['value'], (int, float)):
                        FILTER_CONFIG[key]['value'] = type(FILTER_CONFIG[key]['value'])(float(value))
                    else:
                        FILTER_CONFIG[key]['value'] = str(value)
        log_and_notify('info', "⚙️ تم تحديث الإعدادات من لوحة التحكم.", "SETTINGS_UPDATE")
        return jsonify({"success": True, "message": "Settings updated successfully"})
    except Exception as e:
        logger.error(f"❌ [API Settings] فشل تحديث الإعدادات: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 400

@app.route('/api/signals/close/<int:signal_id>', methods=['POST'])
def manual_close_trade_endpoint(signal_id):
    if not redis_client or not client: return jsonify({"success": False, "message": "Services not ready"}), 503
    with signal_cache_lock: signal_to_close = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)
    if not signal_to_close: return jsonify({"success": False, "message": "Signal not found"}), 404
    try:
        current_price = float(redis_client.hget("crypto_bot_prices", signal_to_close['symbol']))
    except (TypeError, ValueError):
        try: current_price = float(client.get_symbol_ticker(symbol=signal_to_close['symbol'])['price'])
        except Exception as e: return jsonify({"success": False, "message": f"Could not fetch price: {e}"}), 500
    if close_signal(signal_id, current_price, 'manual'): return jsonify({"success": True, "message": "Signal closed."})
    else: return jsonify({"success": False, "message": "Failed to close signal."}), 500

# --- الدوال الرئيسية لتشغيل البوت ---
def main():
    global client, current_prices
    
    logger.info("🚀 بدء تشغيل نظام التداول الآلي Sentinel V11.0")
    
    # تهيئة الاتصالات
    init_db()
    init_redis()
    
    # تهيئة عميل Binance
    client = Client(API_KEY, API_SECRET)
    
    # الحصول على معلومات المنصة والعملات الصالحة
    get_exchange_info_map()
    validated_symbols_to_scan = get_validated_symbols()
    
    if not validated_symbols_to_scan:
        logger.critical("❌ لا توجد عملات صالحة للمسح. إنهاء التشغيل.")
        return
    
    # بدء WebSocket للاستماع إلى تدفق الأسعار
    twm = start_websocket_manager()
    
    # تشغيل واجهة الويب في خيط منفصل
    web_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=5000, threaded=True))
    web_thread.daemon = True
    web_thread.start()
    logger.info("🌐 بدء واجهة الويب على http://localhost:5000")
    
    # الحصول على الأسعار الحالية
    current_prices = {}
    if redis_client:
        try:
            for symbol in validated_symbols_to_scan:
                ticker = client.get_symbol_ticker(symbol=symbol)
                current_prices[symbol] = ticker['price']
            redis_client.hset("crypto_bot_prices", mapping=current_prices)
        except Exception as e:
            logger.error(f"❌ خطأ في جلب الأسعار الأولية: {e}")
    
    # حلقة المسح الدورية
    try:
        while True:
            scan_and_generate_signals()
            gc.collect()  # تنظيف الذاكرة
            time.sleep(60)  # انتظار دقيقة قبل المسح التالي
    except KeyboardInterrupt:
        logger.info("🛑 تم إيقاف البوت يدوياً")
    except Exception as e:
        logger.error(f"❌ خطأ غير متوقع في الحلقة الرئيسية: {e}", exc_info=True)
    finally:
        if twm:
            twm.stop()
        logger.info("🔚 تم إنهاء تشغيل البوت")

if __name__ == "__main__":
    main()