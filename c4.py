# ملف c4_enhanced_v10.1_dashboard_complete.py - نسخة V10.1 "Phoenix"
# --- نسخة معدلة مع إصلاح جذري لمشكلة حدود الـ API ---
# هذا الملف يدمج جميع الدوال والوظائف في هيكل واحد متكامل،
# ويحتوي على كل التحسينات المطلوبة بما في ذلك لوحة التحكم المطورة.

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

# --- إعدادات التجاهل واللوجر ---
warnings = __import__('warnings')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v10.1_phoenix.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV10.1-Phoenix')

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
api_weight_lock = Lock()
api_used_weight = 0
last_api_weight_reset = time.time()

# --- المتغيرات القابلة للتعديل ---
RISK_PER_TRADE_PERCENT: float = 0.85
risk_per_trade_lock = Lock()

# --- مفاتيح تفعيل الاستراتيجيات ---
STRATEGY_CONFIG = {
    "BB_Stoch": {"enabled": True, "lock": Lock(), "display_name": "BB+Stoch (Enhanced)"},
    "MACD_EMA": {"enabled": True, "lock": Lock(), "display_name": "MACD+EMA (Enhanced)"},
    "SR_Breakout": {"enabled": True, "lock": Lock(), "display_name": "S/R Breakout (Enhanced)"},
    "Triple_Confirmation": {"enabled": True, "lock": Lock(), "display_name": "Triple Confirmation (New)"},
    "VWAP_Reversal": {"enabled": True, "lock": Lock(), "display_name": "VWAP Reversal (New)"},
}

# --- إعدادات الفلاتر القابلة للتعديل ---
FILTER_CONFIG = {
    "ADX_THRESHOLD": {"value": 20, "lock": Lock(), "display_name": "حد مؤشر ADX"},
    "BB_STOCH_VOLUME_MULT": {"value": 1.1, "lock": Lock(), "display_name": "مضاعف فوليوم (BB Stoch)"},
    "SR_BREAKOUT_VOLUME_MULT": {"value": 1.3, "lock": Lock(), "display_name": "مضاعف فوليوم (SR Breakout)"},
    "TRIPLE_CONF_VOLUME_MULT": {"value": 1.1, "lock": Lock(), "display_name": "مضاعف فوليوم (Triple Conf)"},
    "VWAP_VOLUME_MULT": {"value": 1.2, "lock": Lock(), "display_name": "مضاعف فوليوم (VWAP Reversal)"},
    "TRIPLE_CONF_MODE": {"value": "relaxed", "lock": Lock(), "display_name": "وضع (Triple Conf)"}, # 'strict' or 'relaxed'
    "VWAP_REVERSAL_MODE": {"value": "relaxed", "lock": Lock(), "display_name": "وضع (VWAP Reversal)"}, # 'strict' or 'relaxed'
    "TIME_BASED_EXIT_CANDLES": {"value": 20, "lock": Lock(), "display_name": "إغلاق الصفقة بعد (شمعة)"}
}


# --- إعدادات المؤشرات الفنية والإطارات الزمنية ---
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 30
BTC_SYMBOL: str = 'BTCUSDT'
MAX_OPEN_TRADES: int = 5
ATR_PERIOD: int = 14
ADX_PERIOD: int = 14

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
# --- FIX START: Initialize cache for historical data ---
historical_data_cache = {}
cache_lock = Lock()
CACHE_EXPIRATION_SECONDS = 15 * 60 # 15 minutes
# --- FIX END ---

# --- قاموس أسباب الرفض باللغة العربية (موسع) ---
REJECTION_REASONS_AR = {
    "Market Status Filter: BTC Downtrend": "فلتر السوق: اتجاه البيتكوين هابط",
    "Market Status Filter: Low Liquidity": "فلتر السوق: سيولة منخفضة",
    "BB_Stoch: ADX Filter Failed": "BB_Stoch: فلتر قوة الاتجاه ADX",
    "BB_Stoch: BBW Filter Failed": "BB_Stoch: فلتر توسع البولينجر BBW",
    "BB_Stoch: Volume Filter Failed": "BB_Stoch: فلتر تأكيد حجم التداول",
    "MACD_EMA: RSI Filter Failed": "MACD_EMA: فلتر RSI",
    "MACD_EMA: Trend Filter Failed": "MACD_EMA: فلتر تأكيد الاتجاه",
    "SR_Breakout: Retest Failed": "SR_Breakout: فشل إعادة اختبار المستوى",
    "Triple_Confirmation: Conditions Not Met": "Triple Confirmation: لم تتحقق الشروط",
    "VWAP_Reversal: Conditions Not Met": "VWAP Reversal: لم تتحقق الشروط",
    "Insufficient Balance": "الرصيد غير كافٍ",
    "Min Notional Filter": "قيمة الصفقة أقل من الحد الأدنى",
}

# --- نظام الحماية من تجاوز حدود API ---
def rate_limiter(weight=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global api_used_weight, last_api_weight_reset
            with api_weight_lock:
                current_time = time.time()
                if current_time - last_api_weight_reset > 60:
                    api_used_weight = 0
                    last_api_weight_reset = current_time
                
                if api_used_weight + weight > 5800:
                    sleep_time = 60 - (current_time - last_api_weight_reset)
                    logger.warning(f"⚠️ [API Limiter] الاقتراب من حد الوزن (المستخدم: {api_used_weight}). الانتظار {sleep_time:.2f} ثانية...")
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    api_used_weight = 0
                    last_api_weight_reset = time.time()

                try:
                    response = func(*args, **kwargs)
                    api_used_weight += weight
                    return response
                except (BinanceAPIException, BinanceRequestException) as e:
                    if e.status_code == 429 or e.status_code == 418:
                        logger.critical("🚨 [API Limiter] تم الوصول إلى حد الطلبات! (HTTP 429/418). سيتم الانتظار لمدة 5 دقائق.")
                        send_telegram_message("🚨 *تحذير حظر API!* 🚨\nتم الوصول إلى حد الطلبات. سيتوقف البوت مؤقتاً.")
                        time.sleep(300)
                    raise
        return wrapper
    return decorator

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

# --- دوال تهيئة الخدمات وقاعدة البيانات ---
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
                        quantity DOUBLE PRECISION, original_quantity DOUBLE PRECISION, order_id TEXT, closing_reason TEXT
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
                
                cur.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS exit_levels JSONB;")
                cur.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS candles_since_entry INTEGER DEFAULT 0;")
                cur.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS opened_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();")

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

def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    df_calc = df.copy()
    # EMAs and SMAs
    df_calc['ema_50'] = df_calc['close'].ewm(span=50, adjust=False).mean()
    df_calc['ema_200'] = df_calc['close'].ewm(span=200, adjust=False).mean()
    df_calc['volume_sma_20'] = df_calc['volume'].rolling(window=20).mean()
    # ATR
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    # ADX
    up_move = df_calc['high'].diff()
    down_move = -df_calc['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    # RSI
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=14 - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=14 - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    # Bollinger Bands
    bb_period = 20
    df_calc['bb_middle'] = df_calc['close'].rolling(window=bb_period).mean()
    bb_std = df_calc['close'].rolling(window=bb_period).std()
    df_calc['bb_upper'] = df_calc['bb_middle'] + (bb_std * 2)
    df_calc['bb_lower'] = df_calc['bb_middle'] - (bb_std * 2)
    df_calc['bb_width'] = (df_calc['bb_upper'] - df_calc['bb_lower']) / df_calc['bb_middle'].replace(0, 1e-9)
    # Stochastic RSI
    rsi = df_calc['rsi']
    stoch_rsi_val = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min()).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(3).mean() * 100
    df_calc['stoch_rsi_d'] = df_calc['stoch_rsi_k'].rolling(3).mean()
    # MACD
    exp1 = df_calc['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_calc['close'].ewm(span=26, adjust=False).mean()
    df_calc['macd'] = exp1 - exp2
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()
    # VWAP
    q = df_calc['volume']
    p = (df_calc['high'] + df_calc['low'] + df_calc['close']) / 3
    df_calc['vwap'] = (p * q).cumsum() / q.cumsum()
    return df_calc.dropna()

# --- FIX START: New function to get data using cache ---
def get_data_for_symbol(symbol: str) -> Optional[pd.DataFrame]:
    """
    Fetches historical data for a symbol, using a cache to avoid excessive API calls.
    """
    with cache_lock:
        cache_entry = historical_data_cache.get(symbol)
        # Check if cache entry exists and is not expired
        if cache_entry and (time.time() - cache_entry['timestamp']) < CACHE_EXPIRATION_SECONDS:
            logger.info(f"  -> [{symbol}] 🧠 Using cached data.")
            return cache_entry['data']

    # If not in cache or expired, fetch new data from API
    logger.info(f"  -> [{symbol}] 🌐 Fetching new historical data from API.")
    try:
        df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
        if df is not None and not df.empty:
            # Update the cache with the new data
            with cache_lock:
                historical_data_cache[symbol] = {
                    'timestamp': time.time(),
                    'data': df
                }
            return df
        return None
    except BinanceAPIException as e:
        # Re-raise the exception to be handled by the main loop's specific ban handler
        if e.code == -1003:
            raise
        logger.error(f"❌ [{symbol}] Unhandled API exception in get_data_for_symbol: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ [{symbol}] General exception in get_data_for_symbol: {e}")
        return None
# --- FIX END ---

# --- دوال منطق الاستراتيجيات المحسنة والجديدة ---
def check_bb_stoch_strategy_enhanced(df: pd.DataFrame) -> bool:
    if len(df) < 26: return False
    last = df.iloc[-1]
    
    with FILTER_CONFIG["ADX_THRESHOLD"]["lock"]: adx_thresh = FILTER_CONFIG["ADX_THRESHOLD"]["value"]
    with FILTER_CONFIG["BB_STOCH_VOLUME_MULT"]["lock"]: vol_mult = FILTER_CONFIG["BB_STOCH_VOLUME_MULT"]["value"]
    
    price_touch_bb = last['low'] <= last['bb_lower']
    stoch_oversold = last['stoch_rsi_k'] < 30 and last['stoch_rsi_d'] < 30
    adx_strong = last['adx'] > adx_thresh
    if not adx_strong: log_rejection(df.name, "BB_Stoch: ADX Filter Failed", {"adx": f"{last['adx']:.2f}", "threshold": adx_thresh}); return False
    volume_confirmed = last['volume'] > (last['volume_sma_20'] * vol_mult)
    if not volume_confirmed: log_rejection(df.name, "BB_Stoch: Volume Filter Failed", {"vol_multiplier": vol_mult}); return False
    
    if price_touch_bb and stoch_oversold: 
        logger.info(f"  -> [{df.name}] ✅ إشارة BB+Stoch.")
        return True
    return False

def check_sr_breakout_strategy_enhanced(df: pd.DataFrame) -> bool:
    if len(df) < 20: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    
    with FILTER_CONFIG["SR_BREAKOUT_VOLUME_MULT"]["lock"]: vol_mult = FILTER_CONFIG["SR_BREAKOUT_VOLUME_MULT"]["value"]
    
    resistance_level = df['high'].iloc[-11:-1].max()
    breakout = last['close'] > resistance_level and prev['close'] <= resistance_level
    if not breakout: return False
    
    volume_confirmed = last['volume'] > (last['volume_sma_20'] * vol_mult)
    if not volume_confirmed: log_rejection(df.name, "SR_Breakout: Volume Filter Failed", {"vol_multiplier": vol_mult}); return False
    
    logger.info(f"  -> [{df.name}] ✅ إشارة اختراق دعم/مقاومة.")
    return True

def check_triple_confirmation_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 201: return False
    last = df.iloc[-1]

    with FILTER_CONFIG["TRIPLE_CONF_VOLUME_MULT"]["lock"]: vol_mult = FILTER_CONFIG["TRIPLE_CONF_VOLUME_MULT"]["value"]
    with FILTER_CONFIG["TRIPLE_CONF_MODE"]["lock"]: mode = FILTER_CONFIG["TRIPLE_CONF_MODE"]["value"]

    trend_confirmed = last['ema_50'] > last['ema_200']
    momentum_confirmed = last['macd'] > last['macd_signal'] and last['rsi'] > 55
    volume_confirmed = last['volume'] > (last['volume_sma_20'] * vol_mult)
    
    conditions_met = sum([trend_confirmed, momentum_confirmed, volume_confirmed])
    required_conditions = 3 if mode == 'strict' else 2

    if conditions_met >= required_conditions:
        logger.info(f"  -> [{df.name}] ✅ إشارة التأكيد الثلاثي (الوضع: {mode}).")
        return True
        
    log_rejection(df.name, "Triple_Confirmation: Conditions Not Met", {"trend": trend_confirmed, "momentum": momentum_confirmed, "volume": volume_confirmed, "mode": mode})
    return False

def check_vwap_reversal_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 21: return False
    last, prev = df.iloc[-1], df.iloc[-2]

    with FILTER_CONFIG["VWAP_VOLUME_MULT"]["lock"]: vol_mult = FILTER_CONFIG["VWAP_VOLUME_MULT"]["value"]
    with FILTER_CONFIG["VWAP_REVERSAL_MODE"]["lock"]: mode = FILTER_CONFIG["VWAP_REVERSAL_MODE"]["value"]

    vwap_reversal = prev['close'] < prev['vwap'] and last['close'] > last['vwap']
    is_bullish_engulfing = prev['close'] < prev['open'] and last['close'] > last['open'] and last['close'] > prev['open']
    is_hammer = (last['close'] > last['open']) and (last['open'] - last['low']) > 2 * (last['close'] - last['open'])
    candle_confirmed = is_bullish_engulfing or is_hammer
    volume_confirmed = last['volume'] > (df['volume'].iloc[-11:-1].mean() * vol_mult)
    
    passes = False
    if mode == 'strict':
        passes = vwap_reversal and candle_confirmed and volume_confirmed
    else: # relaxed
        passes = vwap_reversal and (candle_confirmed or volume_confirmed)

    if passes:
        logger.info(f"  -> [{df.name}] ✅ إشارة انعكاس VWAP (الوضع: {mode}).")
        return True
        
    log_rejection(df.name, "VWAP_Reversal: Conditions Not Met", {"reversal": vwap_reversal, "candle": candle_confirmed, "volume": volume_confirmed, "mode": mode})
    return False

# --- دوال إدارة المخاطر والصفقات ---
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

def calculate_dynamic_position_size(symbol: str, entry_price: float, atr_value: float) -> Optional[Tuple[Decimal, float]]:
    if not client: return None
    try:
        with risk_per_trade_lock: current_risk_percent = RISK_PER_TRADE_PERCENT
        balance_response = client.get_asset_balance(asset='USDT')
        available_balance = Decimal(balance_response['free'])
        risk_amount_usdt = available_balance * (Decimal(str(current_risk_percent)) / Decimal('100'))
        sl_distance = Decimal(str(atr_value)) * Decimal('2.0')
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
        logger.error(f"❌ [{symbol}] خطأ في حساب حجم الصفقة الديناميكي: {e}", exc_info=True)
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
            message = (
                f"🚨 *{trade_type} جديدة*\n\n"
                f"*{signal_data['symbol']}* | `{signal_data['strategy_name']}`\n\n"
                f"🔹 *الدخول:* `{signal_data['entry_price']:.4f}`\n"
                f"🛑 *وقف الخسارة:* `{signal_data['stop_loss']:.4f}`\n\n"
            )
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
        
        reason_map = {
            'stop_loss': '🛑 تم ضرب وقف الخسارة',
            'all_tp_hit': '✅ تم تحقيق جميع الأهداف',
            'manual': ' تم الإغلاق يدوياً',
            'time_based_exit': '⏳ تم الإغلاق بسبب انتهاء الوقت'
        }
        emoji = "✅" if profit_percentage > 0 else "🛑"
        message = (
            f"{emoji} *إغلاق صفقة {symbol_to_close}*\n\n"
            f"السبب: *{reason_map.get(reason, reason)}*\n"
            f"سعر الإغلاق: `{closing_price:.4f}`\n"
            f"الربح/الخسارة: `{profit_percentage:.2f}%`"
        )
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


# --- واجهة الويب (Flask) المطورة ---
app = Flask(__name__)
CORS(app)

@app.before_request
def block_method():
    pass

def get_dashboard_html_v10_1():
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phoenix V10.1 - لوحة تحكم التداول</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
    <style>
        :root { --bg-main: #0D1117; --bg-card: #161B22; --border-color: #30363D; --text-primary: #E6EDF3; --text-secondary: #848D97; --accent-blue: #58A6FF; --accent-green: #3FB950; --accent-red: #F85149; --accent-yellow: #D29922; }
        body { font-family: 'Tajawal', sans-serif; background-color: var(--bg-main); color: var(--text-primary); }
        .card { background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 0.75rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1); }
        .tab-btn.active { border-bottom-color: var(--accent-blue); color: var(--accent-blue); font-weight: 700; }
        input:checked + .toggle-bg { background-color: var(--accent-green); }
        .progress-bar { background-color: #30363d; border-radius: 9999px; overflow: hidden; }
        .progress-bar-inner { background-color: var(--accent-blue); height: 100%; transition: width 0.5s ease-in-out; }
        .input-field { background-color: var(--bg-main); border: 1px solid var(--border-color); border-radius: 0.5rem; padding: 0.5rem 0.75rem; color: var(--text-primary); }
        select.input-field { -webkit-appearance: none; -moz-appearance: none; appearance: none; background-image: url('data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%23848D97%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22/%3E%3C/svg%3E'); background-repeat: no-repeat; background-position: left 0.75rem center; background-size: 0.65em auto; padding-left: 2rem; }
    </style>
</head>
<body class="p-4 md:p-6">
    <div class="container mx-auto max-w-screen-2xl">
        <header class="mb-6 flex flex-wrap justify-between items-center gap-4">
            <h1 class="text-2xl md:text-3xl font-extrabold">
                <span class="text-transparent bg-clip-text bg-gradient-to-r from-red-500 to-yellow-500">Phoenix</span>
                <span class="text-text-secondary font-medium">V10.1</span>
            </h1>
            <div class="flex items-center gap-x-6 bg-black/20 px-4 py-2 rounded-lg border border-border-color">
                <div class="w-32">
                    <div class="text-xs text-text-secondary mb-1">API Weight</div>
                    <div class="progress-bar h-2 w-full">
                        <div id="api-weight-bar" class="progress-bar-inner"></div>
                    </div>
                </div>
                <div id="market-trend-lights" class="flex items-center gap-x-4">
                </div>
            </div>
        </header>

        <section class="mb-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
            <div class="card p-4 flex flex-col justify-center items-center">
                <h3 class="font-bold text-lg text-text-secondary mb-2">التداول الحقيقي</h3>
                <div class="flex items-center space-x-3 space-x-reverse">
                    <span id="trading-status-text" class="font-bold text-lg"></span>
                    <label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="trading-toggle" class="sr-only" onchange="toggleTrading()"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label>
                </div>
                <div class="mt-2 text-xs text-text-secondary">رصيد USDT: <span id="usdt-balance" class="font-mono">...</span></div>
            </div>
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">الاتجاه العام (4h)</h3><div id="overall-regime" class="text-2xl font-bold text-center">...</div></div>
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">الجلسات النشطة</h3><div id="active-sessions-list" class="flex flex-wrap gap-2 items-center justify-center pt-2">...</div></div>
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">الصفقات المفتوحة</h3><div id="open-trades-count" class="text-2xl font-bold text-center">...</div></div>
        </section>

        <div class="mb-4 border-b border-border-color">
            <nav class="flex space-x-6 space-x-reverse -mb-px">
                <button onclick="showTab('signals', this)" class="tab-btn active text-white py-3 px-1">الصفقات المفتوحة</button>
                <button onclick="showTab('performance', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">أداء الاستراتيجيات</button>
                <button onclick="showTab('settings', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الإعدادات</button>
                <button onclick="showTab('logs', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">السجلات</button>
            </nav>
        </div>

        <main>
            <div id="signals-tab" class="tab-content grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4"></div>
            <div id="performance-tab" class="tab-content hidden"><div class="overflow-x-auto card p-0"><table class="min-w-full text-sm text-right"><thead class="border-b border-border-color bg-black/20"><tr><th class="p-4 font-semibold">الاستراتيجية</th><th class="p-4 font-semibold">إجمالي الصفقات</th><th class="p-4 font-semibold">معدل الربح</th><th class="p-4 font-semibold">إجمالي PNL</th></tr></thead><tbody id="performance-table"></tbody></table></div></div>
            <div id="settings-tab" class="tab-content hidden card p-6">
                 <h4 class="text-xl font-bold mb-6 text-text-primary">الإعدادات العامة</h4>
                 <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
                    <div>
                        <h5 class="text-lg font-bold mb-4 text-text-secondary">إدارة المخاطر</h5>
                        <div class="space-y-4">
                           <div>
                                <label for="risk-percent" class="block text-sm font-medium text-text-secondary mb-1">نسبة المخاطرة للصفقة (%)</label>
                                <input type="number" id="risk-percent" step="0.1" class="input-field w-full">
                           </div>
                        </div>
                    </div>
                    <div class="md:col-span-2">
                        <h5 class="text-lg font-bold mb-4 text-text-secondary">تفعيل الاستراتيجيات</h5>
                        <div id="strategy-toggles-container" class="grid grid-cols-2 gap-x-8 gap-y-3"></div>
                    </div>
                 </div>
                 <div class="border-t border-border-color my-8"></div>
                 <h4 class="text-xl font-bold mb-6 text-text-primary">إعدادات الفلاتر</h4>
                 <div id="filter-settings-container" class="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-8"></div>

                 <div class="mt-8 text-left border-t border-border-color pt-6">
                    <button onclick="saveSettings()" class="bg-accent-blue hover:bg-blue-500 text-white font-bold py-2 px-6 rounded-lg">حفظ الإعدادات</button>
                    <span id="settings-feedback" class="mr-4"></span>
                 </div>
            </div>
            <div id="logs-tab" class="tab-content hidden grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="card p-4"><h4 class="text-lg font-bold mb-3 text-text-secondary">الإشعارات الهامة</h4><div id="notifications-list" class="max-h-[60vh] overflow-y-auto space-y-2 text-sm"></div></div>
                <div class="card p-4"><h4 class="text-lg font-bold mb-3 text-text-secondary">الصفقات المرفوضة</h4><div id="rejections-list" class="max-h-[60vh] overflow-y-auto space-y-2 text-sm"></div></div>
            </div>
        </main>
    </div>

<script>
function showTab(tabId, el) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.add('hidden'));
    document.getElementById(tabId + '-tab').classList.remove('hidden');
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    el.classList.add('active');
}

async function fetchData(url, options = {}) {
    try {
        const response = await fetch(url, options);
        if (!response.ok) { console.error(`Fetch Error: ${response.status}`); return null; }
        return await response.json();
    } catch (e) { console.error('Network Error:', e); return null; }
}

function renderPriceBar(entry, current, stopLoss, target) {
    const range = target - stopLoss;
    if (range <= 0) return '<div class="w-full h-2 bg-gray-600 rounded"></div>';
    const progress = Math.max(0, Math.min(100, ((current - stopLoss) / range) * 100));
    const entryPoint = Math.max(0, Math.min(100, ((entry - stopLoss) / range) * 100));
    return `<div class="w-full h-2 bg-gray-600 rounded relative"><div class="absolute h-full bg-gradient-to-r from-green-500 to-teal-400 rounded" style="width: ${progress}%"></div><div class="absolute top-0 w-0.5 h-3 bg-white -translate-y-0.5 rounded" style="left: ${entryPoint}%" title="Entry: ${entry}"></div></div>`;
}

function updateSignals() {
    fetchData('/api/signals').then(data => {
        if (!data) return;
        const container = document.getElementById('signals-tab');
        container.innerHTML = '';
        if (data.length === 0) { container.innerHTML = '<p class="text-text-secondary col-span-full text-center py-10">لا توجد صفقات مفتوحة حالياً.</p>'; return; }
        data.forEach(s => {
            const profit = parseFloat(s.profit_percentage || 0);
            const pClass = profit > 0 ? 'text-accent-green' : profit < 0 ? 'text-accent-red' : 'text-text-secondary';
            const entry = parseFloat(s.entry_price);
            const current = parseFloat(s.current_price || entry);
            const stopLoss = parseFloat(s.stop_loss);
            const finalTarget = s.exit_levels ? Math.max(...Object.values(s.exit_levels).map(l => l.target_price)) : parseFloat(s.target_price);
            let tpLevelsHTML = '';
            if (s.exit_levels) {
                Object.entries(s.exit_levels).forEach(([level, config]) => {
                    const hitClass = config.is_hit ? 'text-accent-green' : 'text-text-secondary';
                    tpLevelsHTML += `<div class="flex justify-between items-center text-xs ${hitClass}"><span>الهدف ${level} (${config.exit_percentage * 100}%)</span> <span class="font-mono">${config.target_price.toFixed(4)}</span></div>`;
                });
            }
            container.innerHTML += `<div class="card p-4 flex flex-col justify-between"><div><div class="flex justify-between items-center mb-2"><h4 class="text-lg font-bold">${s.symbol}</h4><span class="font-mono text-lg ${pClass}">${profit.toFixed(2)}%</span></div><p class="text-xs text-text-secondary mb-3">${s.strategy_name.replace(/_/g, ' ')}</p><div class="font-mono text-xs space-y-1 mb-4"><div class="flex justify-between"><span>الدخول:</span> <span>${entry.toFixed(4)}</span></div><div class="flex justify-between text-accent-blue"><span>الحالي:</span> <span>${current.toFixed(4)}</span></div><div class="flex justify-between text-accent-red"><span>الوقف:</span> <span>${stopLoss.toFixed(4)}</span></div></div>${renderPriceBar(entry, current, stopLoss, finalTarget)}<div class="mt-4 space-y-1">${tpLevelsHTML}</div></div><div class="mt-4 pt-4 border-t border-border-color"><button onclick="manualClose(${s.id}, '${s.symbol}')" class="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-3 rounded-lg text-sm">إغلاق يدوي</button></div></div>`;
        });
    });
}

function updateStatus() {
    fetchData('/api/status').then(data => {
        if (!data) return;
        document.getElementById('overall-regime').textContent = (data.market_state?.primary_trend || '...').replace(/_/g, ' ');
        document.getElementById('open-trades-count').textContent = `${data.open_trades_count} / ${data.max_open_trades}`;
        document.getElementById('active-sessions-list').innerHTML = data.market_state?.session_status?.name || 'N/A';
        const tradeToggle = document.getElementById('trading-toggle');
        const tradeText = document.getElementById('trading-status-text');
        tradeToggle.checked = data.is_trading_enabled;
        tradeText.textContent = data.is_trading_enabled ? 'مُفعَّل' : 'غير مُفعَّل';
        tradeText.className = `font-bold text-lg ${data.is_trading_enabled ? 'text-accent-green' : 'text-accent-red'}`;
        document.getElementById('usdt-balance').textContent = data.usdt_balance ? parseFloat(data.usdt_balance).toFixed(2) : 'N/A';
        const weightPercent = (data.api_weight / 6000) * 100;
        document.getElementById('api-weight-bar').style.width = `${weightPercent}%`;
        
        const lightsContainer = document.getElementById('market-trend-lights');
        lightsContainer.innerHTML = '';
        const trends = data.market_state?.market_trends;
        if (trends) {
            const timeframes = ['15m', '1h', '4h'];
            timeframes.forEach(tf => {
                const trend = trends[tf] || 'RANGING';
                let lightClass = 'bg-accent-yellow';
                if (trend.includes('UPTREND')) lightClass = 'bg-accent-green';
                else if (trend.includes('DOWNTREND')) lightClass = 'bg-accent-red';
                lightsContainer.innerHTML += `<div class="flex items-center gap-2" title="Trend for ${tf} is ${trend}"><div class="w-3 h-3 rounded-full ${lightClass}"></div><span class="text-sm font-bold text-text-secondary">${tf}</span></div>`;
            });
        }

        if(data.settings) {
            document.getElementById('risk-percent').value = data.settings.risk_percent;
            const togglesContainer = document.getElementById('strategy-toggles-container');
            togglesContainer.innerHTML = '';
            Object.entries(data.settings.strategies).forEach(([key, config]) => {
                togglesContainer.innerHTML += `<div class="flex items-center justify-between"><span class="text-sm">${config.display_name}</span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="strategy-${key}" data-strategy-key="${key}" class="sr-only strategy-toggle-input" ${config.enabled ? 'checked' : ''}><div class="toggle-bg block bg-gray-600 w-10 h-6 rounded-full"></div></div></label></div>`;
            });
            
            const filtersContainer = document.getElementById('filter-settings-container');
            filtersContainer.innerHTML = '';
            Object.entries(data.settings.filters).forEach(([key, config]) => {
                let inputHTML = '';
                if (typeof config.value === 'number') {
                    inputHTML = `<input type="number" id="filter-${key}" data-filter-key="${key}" value="${config.value}" step="${key === 'TIME_BASED_EXIT_CANDLES' ? '1' : '0.1'}" class="input-field w-full filter-input">`;
                } else {
                    inputHTML = `<select id="filter-${key}" data-filter-key="${key}" class="input-field w-full filter-input">
                        <option value="strict" ${config.value === 'strict' ? 'selected' : ''}>صارم</option>
                        <option value="relaxed" ${config.value === 'relaxed' ? 'selected' : ''}>مخفف</option>
                    </select>`;
                }
                filtersContainer.innerHTML += `<div><label for="filter-${key}" class="block text-sm font-medium text-text-secondary mb-1">${config.display_name}</label>${inputHTML}</div>`;
            });
        }
    });
}

function updatePerformance() {
    fetchData('/api/performance').then(data => {
        if (!data) return;
        const tableBody = document.getElementById('performance-table');
        tableBody.innerHTML = '';
        data.forEach(p => {
            const pnlClass = p.total_pnl_percent > 0 ? 'text-accent-green' : p.total_pnl_percent < 0 ? 'text-accent-red' : 'text-text-secondary';
            const winRate = p.total_trades > 0 ? (p.winning_trades / p.total_trades) * 100 : 0;
            tableBody.innerHTML += `<tr class="border-b border-border-color hover:bg-white/5"><td class="p-4 font-bold">${p.strategy_name.replace(/_/g, ' ')}</td><td class="p-4 font-mono text-center">${p.total_trades}</td><td class="p-4 font-mono text-center">${winRate.toFixed(2)}%</td><td class="p-4 font-mono text-center ${pnlClass}">${p.total_pnl_percent.toFixed(2)}%</td></tr>`;
        });
    });
}

function updateLogs() {
    fetchData('/api/notifications').then(data => { if(data) document.getElementById('notifications-list').innerHTML = data.map(n => `<div class="p-2 border-b border-border-color/50"><span class="font-mono text-xs text-text-secondary/70">${new Date(n.timestamp).toLocaleTimeString('ar-EG')}</span>: ${n.message}</div>`).join(''); });
    fetchData('/api/rejection_logs').then(data => { if(data) document.getElementById('rejections-list').innerHTML = data.map(r => `<div class="p-2 border-b border-border-color/50"><span class="font-mono text-xs text-text-secondary/70">${new Date(r.timestamp).toLocaleTimeString('ar-EG')}</span>: <strong class="text-accent-yellow">${r.symbol}</strong> - ${r.reason}</div>`).join(''); });
}

function manualClose(signalId, symbol) {
    if (confirm(`هل أنت متأكد من رغبتك في إغلاق الصفقة لـ ${symbol} يدوياً؟`)) {
        fetchData(`/api/signals/close/${signalId}`, { method: 'POST' }).then(data => { if(data && data.success) { updateSignals(); } else { alert(data ? data.message : 'فشل الإغلاق'); } });
    }
}

function toggleTrading() { fetchData('/api/trading/toggle', { method: 'POST' }).then(() => updateStatus()); }

function saveSettings() {
    const strategies = {};
    document.querySelectorAll('.strategy-toggle-input').forEach(input => { strategies[input.dataset.strategyKey] = input.checked; });
    
    const filters = {};
    document.querySelectorAll('.filter-input').forEach(input => {
        const key = input.dataset.filterKey;
        const value = input.type === 'number' ? parseFloat(input.value) : input.value;
        filters[key] = value;
    });

    const settings = { 
        risk_percent: parseFloat(document.getElementById('risk-percent').value), 
        strategies: strategies,
        filters: filters
    };

    const feedbackEl = document.getElementById('settings-feedback');
    feedbackEl.textContent = 'جاري الحفظ...';
    feedbackEl.className = 'mr-4 text-accent-yellow';

    fetchData('/api/settings/update', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(settings) }).then(data => {
        if (data && data.success) { feedbackEl.textContent = '✅ تم الحفظ بنجاح!'; feedbackEl.className = 'mr-4 text-accent-green'; }
        else { feedbackEl.textContent = `❌ فشل الحفظ: ${data ? data.message : 'خطأ'}`; feedbackEl.className = 'mr-4 text-accent-red'; }
        setTimeout(() => { feedbackEl.textContent = ''; }, 3000);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    updateStatus(); updateSignals(); updatePerformance(); updateLogs();
    setInterval(updateStatus, 5000); setInterval(updateSignals, 7000);
    setInterval(updatePerformance, 30000); setInterval(updateLogs, 15000);
});
</script>
</body></html>
""";

@app.route('/')
def home(): return render_template_string(get_dashboard_html_v10_1())

@app.route('/api/status')
def get_status():
    with market_state_lock: state_copy = dict(current_market_state)
    with trading_status_lock: is_enabled = is_trading_enabled
    with api_weight_lock: weight = api_used_weight
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

# --- حلقات النظام الأساسية ---
def determine_market_state_enhanced():
    global current_market_state
    logger.info("🧠 [حالة السوق] جاري تحديث حالة السوق العامة...")
    
    def get_trend(df: pd.DataFrame) -> str:
        if df is None or df.empty or len(df) < 200:
            return "RANGING"
        last = df.iloc[-1]
        if last['ema_50'] > last['ema_200']:
            return "UPTREND"
        elif last['ema_50'] < last['ema_200']:
            return "DOWNTREND"
        return "RANGING"

    try:
        timeframes = {'15m': 10, '1h': 10, '4h': 20}
        market_trends = {}
        
        for tf, days in timeframes.items():
            df_btc = get_data_for_symbol(BTC_SYMBOL) # Use caching for BTC data as well
            if df_btc is not None:
                df_with_features = calculate_all_features(df_btc)
                market_trends[tf] = get_trend(df_with_features)
            else:
                market_trends[tf] = "N/A"


        primary_trend = market_trends.get('4h', 'RANGING')
        
        sessions = {"London": (8, 17), "New York": (13, 22), "Tokyo": (0, 9)}
        now_utc = datetime.now(timezone.utc)
        current_hour = now_utc.hour
        active_sessions = [s for s, (start, end) in sessions.items() if (start <= current_hour < end) or (start > end and (current_hour >= start or current_hour < end))]
        
        session_name = ", ".join(active_sessions) if active_sessions else "خارج الجلسات"
        liquidity = "HIGH" if "London" in active_sessions and "New York" in active_sessions else "NORMAL" if active_sessions else "LOW"

        with market_state_lock:
            current_market_state = {
                "status": "OK",
                "primary_trend": primary_trend,
                "market_trends": market_trends,
                "session_status": {"name": session_name, "liquidity": liquidity},
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
        logger.info(f"✅ [حالة السوق] 15m: {market_trends.get('15m', 'N/A')}, 1h: {market_trends.get('1h', 'N/A')}, 4h: {market_trends.get('4h', 'N/A')}")
    except Exception as e:
        logger.error(f"❌ [حالة السوق] خطأ في التحديث: {e}", exc_info=True)

def passes_comprehensive_market_filter() -> bool:
    with market_state_lock: state = current_market_state
    if state.get("status") != "OK": return False
    
    primary_trend = state.get('primary_trend', 'RANGING')
    if "DOWNTREND" in primary_trend: 
        log_rejection("GLOBAL", "Market Status Filter: BTC Downtrend", {"btc_trend_4h": primary_trend})
        return False
        
    liquidity = state.get('session_status', {}).get('liquidity', 'LOW')
    if liquidity == 'LOW': 
        log_rejection("GLOBAL", "Market Status Filter: Low Liquidity")
        return False

    logger.info("✅ [فلتر السوق] ظروف السوق العامة مناسبة للتداول.")
    return True

def trade_management_loop():
    logger.info("✅ [مدير الصفقات الذكي] بدء حلقة إدارة الصفقات...")
    interval_str = SIGNAL_GENERATION_TIMEFRAME
    interval_seconds = 0
    if 'm' in interval_str:
        interval_seconds = int(interval_str.replace('m', '')) * 60
    elif 'h' in interval_str:
        interval_seconds = int(interval_str.replace('h', '')) * 3600

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
            
            current_prices = redis_client.hgetall("crypto_bot_prices")
            
            with FILTER_CONFIG["TIME_BASED_EXIT_CANDLES"]["lock"]:
                time_based_exit_candles = FILTER_CONFIG["TIME_BASED_EXIT_CANDLES"]["value"]

            for signal in signals_to_check:
                current_price_str = current_prices.get(signal['symbol'])
                if not current_price_str:
                    continue
                
                current_price = float(current_price_str)
                symbol = signal['symbol']
                
                if current_price <= float(signal['stop_loss']):
                    close_signal(signal['id'], current_price, 'stop_loss')
                    continue
                
                if time_based_exit_candles > 0 and interval_seconds > 0:
                    opened_at_time = signal.get('opened_at')
                    if opened_at_time:
                        if opened_at_time.tzinfo is None:
                            opened_at_time = opened_at_time.replace(tzinfo=timezone.utc)
                        
                        time_since_open = (datetime.now(timezone.utc) - opened_at_time).total_seconds()
                        candles_passed = time_since_open / interval_seconds
                        
                        if candles_passed > time_based_exit_candles:
                            first_tp_hit = signal.get('exit_levels', {}).get('1', {}).get('is_hit', False)
                            if not first_tp_hit:
                                logger.info(f"⏳ [{symbol}] خروج زمني بعد {candles_passed:.1f} شمعة (الحد: {time_based_exit_candles}).")
                                close_signal(signal['id'], current_price, 'time_based_exit')
                                continue

                if USE_SMART_EXIT_SYSTEM and 'exit_levels' in signal and signal['exit_levels']:
                    if signal.get('quantity') is None:
                        continue
                    
                    try:
                        remaining_quantity = Decimal(str(signal['quantity']))
                    except (InvalidOperation, TypeError):
                        logger.warning(f"⚠️ [{symbol}] قيمة الكمية غير صالحة ({signal.get('quantity')}) للصفقة ID {signal.get('id')}. سيتم تخطي إدارة الخروج الجزئي.")
                        continue

                    exit_levels = signal['exit_levels']
                    
                    for level, config in sorted(exit_levels.items()):
                        if not config['is_hit'] and current_price >= config['target_price']:
                            logger.info(f"🎯 [{symbol}] تم الوصول إلى الهدف {level} عند {config['target_price']:.4f}")
                            config['is_hit'] = True
                            exit_qty_percent = Decimal(str(config['exit_percentage']))
                            original_quantity = Decimal(str(signal['original_quantity']))
                            quantity_to_sell = original_quantity * exit_qty_percent
                            if signal.get('is_real_trade') and place_order(symbol, Client.SIDE_SELL, quantity_to_sell) is None:
                                continue
                            remaining_quantity -= quantity_to_sell
                            signal['quantity'] = float(remaining_quantity)
                            log_and_notify('info', f"↗️ [{symbol}] خروج جزئي ({exit_qty_percent*100}%): بيع {quantity_to_sell} عند الهدف {level}", "PARTIAL_EXIT")
                            
                            message = (
                                f"↗️ *خروج جزئي من {symbol}*\n\n"
                                f"تم تحقيق الهدف {level} عند `{config['target_price']:.4f}`.\n"
                                f"تم بيع `{exit_qty_percent*100}%` من الكمية."
                            )
                            send_telegram_message(message)

                    signal['exit_levels'] = exit_levels
                    update_signal_in_db(signal['id'], {'exit_levels': exit_levels, 'quantity': float(remaining_quantity)})
                    if remaining_quantity <= Decimal('0.00000001'):
                        close_signal(signal['id'], current_price, 'all_tp_hit')
                        continue

                if USE_TRAILING_STOP_LOSS:
                    atr_value = signal.get('signal_details', {}).get('atr')
                    
                    if atr_value:
                        activation_price = float(signal['entry_price']) + (float(atr_value) * TRAILING_STOP_ACTIVATION_ATR)
                        if current_price >= activation_price:
                            new_stop_loss = current_price * 0.99
                            if new_stop_loss > float(signal['stop_loss']):
                                signal['stop_loss'] = new_stop_loss
                                update_signal_in_db(signal['id'], {'stop_loss': new_stop_loss})
                                logger.info(f"🛡️ [{symbol}] تم تحريك وقف الخسارة إلى {new_stop_loss:.4f}")
                    else:
                        logger.warning(f"⚠️ [{symbol}] مفتاح 'atr' غير موجود في تفاصيل الإشارة ID {signal.get('id')}. سيتم تخطي حساب وقف الخسارة المتحرك.")

            time.sleep(3)
        except Exception as e:
            logger.error(f"❌ [مدير الصفقات] خطأ في حلقة الإدارة: {e}", exc_info=True)
            time.sleep(10)

def main_loop_enhanced():
    logger.info("[الحلقة الرئيسية] انتظار اكتمال التهيئة...")
    time.sleep(15)
    if not validated_symbols_to_scan: log_and_notify("critical", "قائمة العملات فارغة.", "SYSTEM_ERROR"); return
    log_and_notify("info", f"✅ بدء حلقة المسح لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")

    while True:
        try:
            determine_market_state_enhanced()
            if not passes_comprehensive_market_filter():
                logger.info("⏸️ [الحلقة الرئيسية] السوق في حالة غير مناسبة. الانتظار 5 دقائق..."); time.sleep(300); continue
            
            symbols_to_process = random.sample(validated_symbols_to_scan, len(validated_symbols_to_scan))
            
            for symbol in symbols_to_process:
                with signal_cache_lock:
                    if symbol in open_signals_cache or len(open_signals_cache) >= MAX_OPEN_TRADES: continue
                
                # --- FIX START: Use the new caching function ---
                df_15m = get_data_for_symbol(symbol)
                # --- FIX END ---

                if df_15m is None or len(df_15m) < 201: 
                    time.sleep(0.5) # Shorter sleep, as calls are less frequent now
                    continue
                
                df_with_indicators = calculate_all_features(df_15m)
                df_with_indicators.name = symbol
                
                signal_found, strategy_used = False, None
                strategies_to_check = [
                    ('BB_Stoch', check_bb_stoch_strategy_enhanced),
                    ('SR_Breakout', check_sr_breakout_strategy_enhanced), 
                    ('Triple_Confirmation', check_triple_confirmation_strategy),
                    ('VWAP_Reversal', check_vwap_reversal_strategy)
                ]
                for key, func in strategies_to_check:
                    with STRATEGY_CONFIG[key]['lock']: is_enabled = STRATEGY_CONFIG[key]['enabled']
                    if is_enabled and func(df_with_indicators):
                        signal_found, strategy_used = True, key; break
                
                if signal_found:
                    logger.info(f"  -> [{symbol}] إشارة ناجحة من {strategy_used}. جاري التحقق النهائي...")
                    try: 
                        entry_price_str = redis_client.hget("crypto_bot_prices", symbol)
                        if not entry_price_str:
                            logger.warning(f"⚠️ [{symbol}] لم يتم العثور على السعر في Redis. سيتم جلبه عبر API.")
                            entry_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                        else:
                            entry_price = float(entry_price_str)
                    except Exception as e: 
                        logger.error(f"❌ [{symbol}] فشل جلب سعر الدخول: {e}."); continue
                    
                    last_atr = df_with_indicators.iloc[-1]['atr']
                    size_result = calculate_dynamic_position_size(symbol, entry_price, last_atr)
                    if not size_result: continue
                    quantity, stop_loss_price = size_result
                    
                    new_signal = {'symbol': symbol, 'strategy_name': strategy_used, 'entry_price': entry_price,
                                  'stop_loss': stop_loss_price, 'signal_details': {'atr': last_atr}}
                    
                    with trading_status_lock: is_enabled = is_trading_enabled
                    if is_enabled:
                        order_result = place_order(symbol, Client.SIDE_BUY, quantity)
                        if order_result:
                            new_signal.update({'is_real_trade': True, 'quantity': float(quantity), 'order_id': order_result['orderId']})
                        else: continue
                    
                    saved_signal = insert_signal_into_db(new_signal)
                    if saved_signal:
                        with signal_cache_lock: open_signals_cache[saved_signal['symbol']] = saved_signal
                
                time.sleep(0.5) 
            
            logger.info(f"✅ [نهاية الدورة] انتهت دورة المسح. الانتظار 5 دقائق...");
            time.sleep(300)
        except (KeyboardInterrupt, SystemExit):
            log_and_notify("info", "إيقاف البوت.", "SYSTEM")
            break
        except BinanceAPIException as e:
            if e.code == -1003:
                logger.critical("🚨 [API BAN] تم حظر الـ IP! سيتوقف البوت لمدة 30 دقيقة لاحترام الحظر.")
                send_telegram_message("🚨 *تم حظر الـ IP!* 🚨\nسيتوقف البوت مؤقتاً لمدة 30 دقيقة.")
                time.sleep(1800) # Sleep for 30 minutes
            else:
                log_and_notify("error", f"خطأ غير متوقع من Binance API في الحلقة الرئيسية: {e}", "SYSTEM")
                traceback.print_exc()
                time.sleep(120)
        except Exception as main_err:
            log_and_notify("error", f"خطأ حرج في الحلقة الرئيسية: {main_err}", "SYSTEM")
            traceback.print_exc()
            time.sleep(120)

# --- دوال WebSocket الجديدة والمعدلة ---
def handle_socket_message(msg: Dict[str, Any]):
    try:
        if msg.get('e') == 'error':
            logger.error(f"❌ [WebSocket] Error: {msg.get('m')}")
            return

        if 'stream' in msg and 'data' in msg:
            payload = msg['data']
            if isinstance(payload, dict) and 's' in payload and 'c' in payload:
                prices_to_set = {payload['s']: payload['c']}
                if prices_to_set and redis_client:
                    redis_client.hset("crypto_bot_prices", mapping=prices_to_set)
            return
        
        if isinstance(msg, list):
            prices_to_set = {item['s']: item['c'] for item in msg if 's' in item and 'c' in item}
            if prices_to_set and redis_client:
                redis_client.hset("crypto_bot_prices", mapping=prices_to_set)
            return

    except (KeyError, TypeError) as e:
        logger.error(f"❌ [WebSocket] Error processing message: {e} | Data: {msg}")
    except Exception as e:
        logger.error(f"❌ [WebSocket] Unexpected error in handler: {e}", exc_info=True)

def start_websocket_streams():
    if not validated_symbols_to_scan:
        logger.warning("⚠️ [WebSocket] No symbols to stream. Skipping WebSocket start.")
        return

    logger.info("✅ [WebSocket] Starting price streams...")
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()

    streams = [f"{s.lower()}@miniTicker" for s in validated_symbols_to_scan]
    if BTC_SYMBOL not in validated_symbols_to_scan:
         streams.append(f"{BTC_SYMBOL.lower()}@miniTicker")

    twm.start_multiplex_socket(callback=handle_socket_message, streams=streams)
    logger.info(f"✅ [WebSocket] Subscribed to {len(streams)} mini-ticker streams.")


def load_initial_data():
    global validated_symbols_to_scan
    get_exchange_info_map()
    validated_symbols_to_scan = get_validated_symbols()
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status = 'open';")
            open_signals = cur.fetchall()
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in open_signals:
                    if isinstance(signal.get('signal_details'), str):
                        try:
                            signal['signal_details'] = json.loads(signal['signal_details'])
                        except json.JSONDecodeError:
                            signal['signal_details'] = {}
                    elif signal.get('signal_details') is None:
                        signal['signal_details'] = {}
                    open_signals_cache[signal['symbol']] = dict(signal)
            logger.info(f"✅ [تحميل] تم تحميل {len(open_signals)} صفقة مفتوحة إلى الذاكرة المؤقتة.")
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 100;")
            with notifications_lock: notifications_cache.extend([dict(n) for n in cur.fetchall()])
    except Exception as e:
        logger.error(f"❌ [تحميل] فشل تحميل البيانات الأولية: {e}")

def initialize_bot_services():
    global client
    logger.info("🤖 [خدمات البوت] بدء التهيئة...")
    try:
        client = Client(API_KEY, API_SECRET)
        init_db()
        init_redis()
        load_initial_data()
        
        Thread(target=main_loop_enhanced, daemon=True).start()
        Thread(target=trade_management_loop, daemon=True).start()
        
        start_websocket_streams()

        logger.info("✅ [خدمات البوت] تم بدء جميع الخدمات الخلفية بنجاح.")
        send_telegram_message("✅ *البوت قيد التشغيل الآن (نسخة V10.1 - Phoenix)*")
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حرج أثناء التهيئة: {e}", "SYSTEM"); exit(1)

# --- نقطة الدخول الرئيسية ---
if __name__ == "__main__":
    logger.info("🚀 إطلاق بوت التداول V10.1 'Phoenix' مع لوحة التحكم 🚀")
    Thread(target=initialize_bot_services, daemon=True).start()
    port = int(os.environ.get('PORT', 10000))
    host = "0.0.0.0"
    logger.info(f"✅ بدء لوحة التحكم على http://{host}:{port}")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        app.run(host=host, port=port, debug=False)
    logger.info("👋 [إيقاف] تم إيقاف تشغيل التطبيق.")

