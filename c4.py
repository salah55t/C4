# ملف c4_enhanced_v11.8.py - نسخة V11.8 "Aggressive Memory Management"
# --- نسخة معدلة مع إدارة ذاكرة قوية لمنع التسرب وحل مشكلة نفاد الذاكرة ---

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
        logging.FileHandler('crypto_bot_v11.8_aggressive_gc.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV11.8-AggressiveGC')

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
    "SIGNAL_STRICTNESS": {"value": 75, "lock": Lock(), "display_name": "مقياس صرامة الشروط"}, # 0 (مرن) to 100 (صارم)
    "ADX_THRESHOLD": {"value": 25, "lock": Lock(), "display_name": "حد مؤشر ADX"},
    "TIME_BASED_EXIT_CANDLES": {"value": 30, "lock": Lock(), "display_name": "إغلاق الصفقة بعد (شمعة)"},
    "MAX_CORRELATION_THRESHOLD": {"value": 0.7, "lock": Lock(), "display_name": "حد الارتباط بين الأصول"},
    "LIQUIDITY_FILTER_STRICTNESS": {"value": "medium", "lock": Lock(), "display_name": "صرامة فلتر السيولة"}, # 'low', 'medium', 'high'
}

# --- إعدادات المؤشرات الفنية والإطارات الزمنية ---
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 90
BTC_SYMBOL: str = 'BTCUSDT'
MAX_OPEN_TRADES: int = 5
ATR_PERIOD: int = 14
ADX_PERIOD: int = 14
CACHE_EXPIRATION_MINUTES: int = 15
BATCH_SIZE: int = 50 # حجم الدفعة للمسح

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
    "Conditions Not Met": "لم تستوفِ الشروط درجة الصرامة المطلوبة",
    "Market Status Filter: BTC Downtrend (4h)": "فلتر السوق: اتجاه البيتكوين هابط (4 ساعات)",
    "Market Status Filter: Low Liquidity": "فلتر السوق: سيولة منخفضة",
    "Market Status Filter: High Volatility": "فلتر السوق: تقلبات عالية",
    "Market Status Filter: Weak Market Strength": "فلتر السوق: ضعف قوة السوق",
    "Insufficient Balance": "الرصيد غير كافٍ",
    "Min Notional Filter": "قيمة الصفقة أقل من الحد الأدنى",
    "Invalid Position Size": "حجم الصفقة غير صالح",
    "Lot Size Adjustment Failed": "فشل تعديل حجم العقد",
    "Portfolio Risk: Max Open Trades": "مخاطر المحفظة: الحد الأقصى للصفقات المفتوحة",
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
                return func(*args, **kwargs)
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
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload, timeout=10).raise_for_status()
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
                    CREATE TABLE IF NOT EXISTS strategy_performance (
                        strategy_name TEXT PRIMARY KEY, total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0, total_pnl_percent DOUBLE PRECISION DEFAULT 0.0
                    );
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL, message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE
                    );
                    CREATE TABLE IF NOT EXISTS historical_data_cache (
                        symbol_timeframe TEXT PRIMARY KEY, symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL, data JSONB NOT NULL,
                        last_updated TIMESTAMP WITH TIME ZONE NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_symbol_timeframe ON historical_data_cache (symbol, timeframe);
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
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error}
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
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
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

def _df_to_json(df: pd.DataFrame) -> str: return df.to_json(orient='split', date_format='iso')
def _json_to_df(json_data: Any) -> pd.DataFrame:
    json_str = json.dumps(json_data) if not isinstance(json_data, str) else json_data
    df = pd.read_json(json_str, orient='split')
    df.index = pd.to_datetime(df.index, utc=True)
    return df

def get_data_for_symbol(symbol: str, timeframe: str, days: int) -> Optional[pd.DataFrame]:
    if timeframe == '5m': return fetch_historical_data(symbol, timeframe, days=2)
    if not check_db_connection() or not conn: return fetch_historical_data(symbol, timeframe, days)
    try:
        with conn.cursor() as cur:
            pk = f"{symbol}_{timeframe}"
            cur.execute("SELECT data, last_updated FROM historical_data_cache WHERE symbol_timeframe = %s", (pk,))
            cache_result = cur.fetchone()
        if cache_result and (datetime.now(timezone.utc) - cache_result['last_updated']) < timedelta(minutes=CACHE_EXPIRATION_MINUTES):
            return _json_to_df(cache_result['data'])
    except Exception as e:
        logger.error(f"❌ [DB Cache] خطأ أثناء قراءة الكاش لـ {symbol}-{timeframe}: {e}")
        if conn: conn.rollback()
    try:
        df = fetch_historical_data(symbol, timeframe, days)
        if df is not None and not df.empty:
            with conn.cursor() as cur:
                pk = f"{symbol}_{timeframe}"
                cur.execute("""
                    INSERT INTO historical_data_cache (symbol_timeframe, symbol, timeframe, data, last_updated)
                    VALUES (%s, %s, %s, %s, %s) ON CONFLICT (symbol_timeframe) DO UPDATE SET
                    data = EXCLUDED.data, last_updated = EXCLUDED.last_updated;
                """, (pk, symbol, timeframe, _df_to_json(df), datetime.now(timezone.utc)))
            conn.commit()
            return df
        return None
    except Exception as e:
        logger.error(f"❌ [DB Cache] فشل جلب وتخزين البيانات لـ {symbol}-{timeframe}: {e}")
        if conn: conn.rollback()
        if isinstance(e, BinanceAPIException) and e.code == -1003: raise
        return None

def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    df_calc = df.copy()
    df_calc['ema_50'] = df_calc['close'].ewm(span=50, adjust=False).mean()
    df_calc['ema_200'] = df_calc['close'].ewm(span=200, adjust=False).mean()
    df_calc['volume_sma_20'] = df_calc['volume'].rolling(window=20).mean()
    tr = pd.concat([(df_calc['high'] - df_calc['low']), (df_calc['high'] - df_calc['close'].shift()).abs(), (df_calc['low'] - df_calc['close'].shift()).abs()], axis=1).max(axis=1, skipna=False)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    plus_dm = pd.Series(np.where((df_calc['high'].diff() > -df_calc['low'].diff()) & (df_calc['high'].diff() > 0), df_calc['high'].diff(), 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((-df_calc['low'].diff() > df_calc['high'].diff()) & (-df_calc['low'].diff() > 0), -df_calc['low'].diff(), 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    df_calc['adx'] = (100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))).ewm(span=ADX_PERIOD, adjust=False).mean()
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=14 - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=14 - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    df_calc['bb_middle'] = df_calc['close'].rolling(window=20).mean()
    bb_std = df_calc['close'].rolling(window=20).std()
    df_calc['bb_upper'] = df_calc['bb_middle'] + (bb_std * 2)
    df_calc['bb_lower'] = df_calc['bb_middle'] - (bb_std * 2)
    stoch_rsi_val = (df_calc['rsi'] - df_calc['rsi'].rolling(14).min()) / (df_calc['rsi'].rolling(14).max() - df_calc['rsi'].rolling(14).min()).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(3).mean() * 100
    df_calc['stoch_rsi_d'] = df_calc['stoch_rsi_k'].rolling(3).mean()
    df_calc['macd'] = df_calc['close'].ewm(span=12, adjust=False).mean() - df_calc['close'].ewm(span=26, adjust=False).mean()
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()
    df_calc['vwap'] = ((df_calc['high'] + df_calc['low'] + df_calc['close']) / 3 * df_calc['volume']).cumsum() / df_calc['volume'].cumsum()
    return df_calc.dropna()

# --- دوال الاستراتيجيات وإدارة الصفقات ---
def is_bullish_reversal_enhanced(df: pd.DataFrame) -> bool:
    if len(df) < 3: return False
    last, prev, prev2 = df.iloc[-1], df.iloc[-2], df.iloc[-3] if len(df) >= 3 else None
    is_green = bool(last['close'] > last['open'])
    body_size = float(abs(last['close'] - last['open']))
    candle_range = float(last['high'] - last['low'])
    is_strong_body = (body_size / candle_range) >= 0.5 if candle_range > 0 else False
    is_engulfing = is_green and bool(prev['close'] < prev['open']) and bool(last['close'] > prev['open']) and bool(last['open'] < prev['close'])
    is_hammer = is_green and bool((last['open'] - last['low']) > 2 * body_size) and bool((last['high'] - last['close']) < 0.1 * body_size)
    is_morning_star = prev2 is not None and bool(prev2['close'] < prev2['open']) and bool(abs(prev['close'] - prev['open']) < 0.3 * abs(prev2['close'] - prev2['open'])) and is_green and bool(last['close'] > (prev2['open'] + prev2['close']) / 2)
    return is_strong_body or is_engulfing or is_hammer or is_morning_star

def check_bb_reversal_strategy_enhanced(df: pd.DataFrame) -> Tuple[int, int]:
    total_conditions = 5
    if len(df) < 26: return 0, total_conditions
    last, prev = df.iloc[-1], df.iloc[-2]
    with FILTER_CONFIG["ADX_THRESHOLD"]["lock"]: adx_thresh = FILTER_CONFIG["ADX_THRESHOLD"]["value"]
    cond1 = bool(prev['low'] <= prev['bb_lower']) and is_bullish_reversal_enhanced(df)
    cond2 = bool(last['adx'] > adx_thresh)
    cond3 = bool(last['volume'] > (last['volume_sma_20'] * 1.2))
    cond4 = bool(last['rsi'] < 30 or prev['rsi'] < 30)
    price_below_bb_pct = ((float(prev['bb_lower']) - float(prev['low'])) / float(prev['bb_lower'])) * 100 if float(prev['bb_lower']) > 0 else 0
    cond5 = price_below_bb_pct <= 2
    return sum([cond1, cond2, cond3, cond4, cond5]), total_conditions

def check_macd_ema_strategy_enhanced(df: pd.DataFrame) -> Tuple[int, int]:
    total_conditions = 4
    if len(df) < 201: return 0, total_conditions
    last, prev = df.iloc[-1], df.iloc[-2]
    cond1 = bool(last['ema_50'] > last['ema_200'])
    cond2 = bool(last['macd'] > last['macd_signal']) and bool(prev['macd'] <= prev['macd_signal']) and bool(last['macd'] < 0)
    cond3 = bool(last['rsi'] > 30 and last['rsi'] < 70)
    cond4 = bool(last['volume'] > (last['volume_sma_20'] * 1.2))
    return sum([cond1, cond2, cond3, cond4]), total_conditions

def check_sr_breakout_strategy_enhanced(df: pd.DataFrame) -> Tuple[int, int]:
    total_conditions = 4
    if len(df) < 50: return 0, total_conditions
    last, prev = df.iloc[-1], df.iloc[-2]
    with FILTER_CONFIG["ADX_THRESHOLD"]["lock"]: adx_thresh = FILTER_CONFIG["ADX_THRESHOLD"]["value"]
    resistance_level = df.iloc[-21:-1]['high'].max()
    cond1 = bool(last['close'] > resistance_level) and bool(prev['close'] <= resistance_level)
    cond2 = bool(last['volume'] > (df.iloc[-21:-1]['volume'].mean() * 1.4))
    cond3 = bool(last['rsi'] > 50 and last['rsi'] < 80)
    cond4 = bool(last['adx'] > adx_thresh)
    return sum([cond1, cond2, cond3, cond4]), total_conditions

def check_triple_confirmation_strategy_enhanced(df: pd.DataFrame) -> Tuple[int, int]:
    total_conditions = 4
    if len(df) < 201: return 0, total_conditions
    last = df.iloc[-1]
    cond1 = bool(last['ema_50'] > last['ema_200'])
    cond2 = bool(last['macd'] > last['macd_signal']) and bool(last['rsi'] > 55 and last['rsi'] < 80) and bool(last['stoch_rsi_k'] > last['stoch_rsi_d'] and last['stoch_rsi_k'] < 80)
    cond3 = bool(last['volume'] > (last['volume_sma_20'] * 1.2))
    cond4 = bool(0.01 < (float(last['atr'] / last['close'])) < 0.05)
    return sum([cond1, cond2, cond3, cond4]), total_conditions

def check_vwap_reversal_strategy_enhanced(df: pd.DataFrame) -> Tuple[int, int]:
    total_conditions = 5
    if len(df) < 21: return 0, total_conditions
    last, prev = df.iloc[-1], df.iloc[-2]
    cond1 = bool(prev['close'] < prev['vwap'] and last['close'] > last['vwap'])
    cond2 = is_bullish_reversal_enhanced(df.iloc[-3:])
    cond3 = bool(last['volume'] > (df['volume'].iloc[-11:-1].mean() * 1.3))
    cond4 = bool(last['rsi'] < 40 or prev['rsi'] < 40)
    cond5 = bool(last['ema_50'] > last['ema_200'])
    return sum([cond1, cond2, cond3, cond4, cond5]), total_conditions

def check_price_channel_strategy(df: pd.DataFrame) -> Tuple[int, int]:
    total_conditions = 4
    if len(df) < 30: return 0, total_conditions
    channel_period = 20
    df['upper_channel'] = df['high'].rolling(window=channel_period).max()
    last, prev = df.iloc[-1], df.iloc[-2]
    with FILTER_CONFIG["ADX_THRESHOLD"]["lock"]: adx_thresh = FILTER_CONFIG["ADX_THRESHOLD"]["value"]
    cond1 = bool(last['close'] > last['upper_channel']) and bool(prev['close'] <= prev['upper_channel'])
    cond2 = bool(last['volume'] > (df['volume'].iloc[-11:-1].mean() * 1.3))
    cond3 = bool(last['rsi'] > 50 and last['rsi'] < 80)
    cond4 = bool(last['adx'] > adx_thresh)
    return sum([cond1, cond2, cond3, cond4]), total_conditions

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
    if not client: return None
    try:
        with risk_per_trade_lock: current_risk_percent = RISK_PER_TRADE_PERCENT
        
        if not is_trading_enabled:
            available_balance = Decimal('1000') 
            logger.info(f"[{symbol}] وضع التداول الورقي: استخدام رصيد افتراضي {available_balance} USDT لحساب حجم الصفقة.")
        else:
            balance_response = client.get_asset_balance(asset='USDT')
            available_balance = Decimal(balance_response['free'])

        risk_amount_usdt = available_balance * (Decimal(str(current_risk_percent)) / Decimal('100'))
        sl_multiplier = 2.5 - (signal_strength * 0.5)
        sl_distance = Decimal(str(atr_value)) * Decimal(str(sl_multiplier))
        actual_stop_loss_price = Decimal(str(entry_price)) - sl_distance
        risk_per_coin = Decimal(str(entry_price)) - actual_stop_loss_price
        
        if risk_per_coin <= 0:
            log_rejection(symbol, "Invalid Position Size", {"details": "وقف الخسارة أعلى من سعر الدخول"})
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
                min_notional = Decimal(min_notional_filter.get('minNotional', min_notional_filter.get('notional', '5.0')))
                
                if is_trading_enabled and notional_value < min_notional:
                    log_rejection(symbol, "Min Notional Filter", {"value": f"{notional_value:.2f}", "min": f"{min_notional}"})
                    return None
                elif not is_trading_enabled:
                     logger.info(f"[{symbol}] تداول ورقي: تم تجاوز فحص الحد الأدنى للقيمة. (المحسوبة: {notional_value:.2f}, الحد الأدنى: {min_notional})")

        if is_trading_enabled and notional_value > available_balance:
            log_rejection(symbol, "Insufficient Balance", {"required": f"{notional_value:.2f}", "available": f"{available_balance:.2f}"})
            return None
            
        return adjusted_quantity, float(actual_stop_loss_price)
        
    except Exception as e:
        logger.error(f"❌ [{symbol}] خطأ في حساب حجم الصفقة: {e}", exc_info=True)
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
        if USE_SMART_EXIT_SYSTEM:
            entry_price = float(signal_data['entry_price'])
            atr_value = float(signal_data.get('signal_details', {}).get('atr', 0))
            if atr_value > 0:
                exit_levels = {str(level): {"target_price": entry_price + (atr_value * config['atr_multiplier']), "exit_percentage": config['exit_percentage'], "is_hit": False} for level, config in TAKE_PROFIT_LEVELS.items()}
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
                message += "*الأهداف:*\n" + "".join([f"  - الهدف {level}: `{config['target_price']:.4f}`\n" for level, config in signal_data['exit_levels'].items()])
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
            values = [json.dumps(v, cls=NpEncoder) if isinstance(v, dict) else v for v in updates.values()] + [signal_id]
            query = sql.SQL("UPDATE signals SET {} WHERE id = %s").format(sql.SQL(', ').join(set_clauses))
            cur.execute(query, values)
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [DB Update] فشل تحديث الصفقة {signal_id}: {e}")
        if conn: conn.rollback()

def check_exit_conditions_enhanced(signal_data: Dict, current_price: float, df: pd.DataFrame) -> Tuple[bool, str]:
    entry_price, stop_loss = float(signal_data['entry_price']), float(signal_data['stop_loss'])
    last_candle = df.iloc[-1]
    if current_price <= stop_loss: return True, "stop_loss"
    with FILTER_CONFIG["TIME_BASED_EXIT_CANDLES"]["lock"]: max_candles = FILTER_CONFIG["TIME_BASED_EXIT_CANDLES"]["value"]
    candles_since_entry = signal_data.get('candles_since_entry', 0) + 1
    if candles_since_entry >= max_candles: return True, "time_based_exit"
    if 'exit_levels' in signal_data and signal_data['exit_levels']:
        exit_levels = signal_data['exit_levels']
        all_hit = all(config.get('is_hit', False) for config in exit_levels.values())
        if all_hit: return True, "all_tp_hit"
        for level, config in exit_levels.items():
            if not config.get('is_hit', False) and current_price >= config['target_price']:
                config['is_hit'] = True
                update_signal_in_db(signal_data['id'], {'exit_levels': exit_levels})
                if signal_data.get('is_real_trade'):
                    quantity_to_sell = Decimal(str(signal_data['quantity'])) * Decimal(str(config['exit_percentage']))
                    place_order(signal_data['symbol'], Client.SIDE_SELL, quantity_to_sell)
                    remaining_quantity = Decimal(str(signal_data['quantity'])) - quantity_to_sell
                    update_signal_in_db(signal_data['id'], {'quantity': float(remaining_quantity)})
    if USE_TRAILING_STOP_LOSS:
        current_peak = max(float(signal_data.get('current_peak_price', entry_price)), current_price)
        if current_peak > float(signal_data.get('current_peak_price', entry_price)):
            update_signal_in_db(signal_data['id'], {'current_peak_price': current_peak})
        if current_price <= (current_peak - (float(last_candle['atr']) * TRAILING_STOP_ACTIVATION_ATR)):
            return True, "trailing_stop"
    update_signal_in_db(signal_data['id'], {'candles_since_entry': candles_since_entry})
    return False, ""

def close_signal(signal_id: int, closing_price: float, reason: str) -> bool:
    with signal_cache_lock:
        signal_to_close = next((s for s in open_signals_cache.values() if s.get('id') == signal_id), None)
    if not signal_to_close: return False
    profit_percentage = ((closing_price - float(signal_to_close['entry_price'])) / float(signal_to_close['entry_price'])) * 100
    update_strategy_performance(signal_to_close['strategy_name'], profit_percentage)
    if signal_to_close.get('is_real_trade'):
        try:
            base_asset = signal_to_close['symbol'].replace('USDT', '')
            balance = Decimal(client.get_asset_balance(asset=base_asset)['free'])
            if balance > 0:
                quantity_to_sell = adjust_quantity_to_lot_size(signal_to_close['symbol'], float(balance))
                if quantity_to_sell and quantity_to_sell > 0: place_order(signal_to_close['symbol'], Client.SIDE_SELL, quantity_to_sell)
        except Exception as e:
            logger.error(f"❌ [{signal_to_close['symbol']}] خطأ أثناء بيع الرصيد عند الإغلاق: {e}")
    if not check_db_connection() or not conn: return False
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE signals SET status = 'closed', closing_price = %s, closed_at = NOW(), profit_percentage = %s, closing_reason = %s WHERE id = %s;",
                        (closing_price, profit_percentage, reason, signal_id))
        conn.commit()
        with signal_cache_lock:
            if signal_to_close['symbol'] in open_signals_cache: del open_signals_cache[signal_to_close['symbol']]
        log_and_notify('info', f"تم الإغلاق: {signal_to_close['symbol']} عند {closing_price:.4f}. السبب: {reason}. الربح/الخسارة: {profit_percentage:.2f}%", "TRADE_CLOSED")
        reason_map = {'stop_loss': '🛑 وقف الخسارة', 'all_tp_hit': '✅ تحقيق الأهداف', 'manual': 'إغلاق يدوي', 'time_based_exit': '⏳ انتهاء الوقت', 'trend_reversal': '🔄 انعكاس الاتجاه', 'trailing_stop': '📉 وقف متحرك'}
        emoji = "✅" if profit_percentage > 0 else "🛑"
        message = (f"{emoji} *إغلاق صفقة {signal_to_close['symbol']}*\n\n"
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
                VALUES (%s, 1, %s, %s) ON CONFLICT (strategy_name) DO UPDATE SET
                total_trades = strategy_performance.total_trades + 1,
                winning_trades = strategy_performance.winning_trades + EXCLUDED.winning_trades,
                total_pnl_percent = strategy_performance.total_pnl_percent + EXCLUDED.total_pnl_percent;
            """, (strategy_name, 1 if pnl_percent > 0 else 0, pnl_percent))
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [Perf Update] فشل تحديث أداء الاستراتيجية {strategy_name}: {e}")
        if conn: conn.rollback()

def check_market_state_enhanced() -> Dict[str, Any]:
    global current_market_state
    with market_state_lock:
        try:
            btc_1h = get_data_for_symbol(BTC_SYMBOL, '1h', days=3)
            btc_4h = get_data_for_symbol(BTC_SYMBOL, '4h', days=7)
            if btc_1h is None or btc_4h is None: return {"status": "DATA_UNAVAILABLE"}
            btc_1h, btc_4h = calculate_all_features(btc_1h), calculate_all_features(btc_4h)
            last_1h, last_4h = btc_1h.iloc[-1], btc_4h.iloc[-1]
            btc_1h_trend = "bullish" if bool(last_1h['ema_50'] > last_1h['ema_200']) else "bearish"
            btc_4h_trend = "bullish" if bool(last_4h['ema_50'] > last_4h['ema_200']) else "bearish"
            btc_1h_atr_pct = (float(last_1h['atr']) / float(last_1h['close'])) * 100
            volatility = "high" if btc_1h_atr_pct > 3.0 else "normal" if btc_1h_atr_pct > 1.0 else "low"
            overall_status = "BEARISH" if btc_4h_trend == "bearish" else "BULLISH" if btc_4h_trend == "bullish" and btc_1h_trend == "bullish" else "NEUTRAL"
            current_market_state = {"status": overall_status, "btc_1h_trend": btc_1h_trend, "btc_4h_trend": btc_4h_trend, "volatility": volatility}
            return current_market_state
        except Exception as e:
            logger.error(f"❌ [Market State] خطأ في تحديد حالة السوق: {e}", exc_info=True)
            return {"status": "ERROR", "error": str(e)}

def check_portfolio_risk() -> bool:
    with signal_cache_lock: open_trades_count = len(open_signals_cache)
    if open_trades_count >= MAX_OPEN_TRADES:
        logger.info(f"🚫 [Portfolio Risk] تم الوصول إلى الحد الأقصى لعدد الصفقات المفتوحة ({MAX_OPEN_TRADES}).")
        return False
    return True

# --- UPDATED: generate_signals_for_symbol with aggressive memory management ---
def generate_signals_for_symbol(symbol: str) -> List[Dict]:
    df = None # Initialize df to None
    try:
        signals = []
        df = get_data_for_symbol(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
        if df is None or df.empty:
            return signals
        
        df = calculate_all_features(df)
        df.name = symbol
        
        strategy_checkers = {
            "BB_Reversal": check_bb_reversal_strategy_enhanced, "MACD_EMA": check_macd_ema_strategy_enhanced,
            "SR_Breakout": check_sr_breakout_strategy_enhanced, "Triple_Confirmation": check_triple_confirmation_strategy_enhanced,
            "VWAP_Reversal": check_vwap_reversal_strategy_enhanced, "Price_Channel": check_price_channel_strategy,
        }
        total_met, total_possible, successful_strategies, strategy_details = 0, 0, [], {}
        for name, checker in strategy_checkers.items():
            if STRATEGY_CONFIG[name]["enabled"]:
                met, total = checker(df)
                total_met += met; total_possible += total
                strategy_details[name] = f"{met}/{total}"
                if met > 0: successful_strategies.append(name)

        if total_possible == 0: return signals
        actual_strength = (total_met / total_possible) * 100
        with FILTER_CONFIG["SIGNAL_STRICTNESS"]["lock"]: required_strength = FILTER_CONFIG["SIGNAL_STRICTNESS"]["value"]

        if actual_strength < required_strength:
            log_rejection(symbol, "Conditions Not Met", {"actual": f"{actual_strength:.2f}%", "required": f"{required_strength}%", "details": strategy_details})
            return signals
        if not check_portfolio_risk(): return signals
        
        entry_price, atr_value = float(df.iloc[-1]['close']), float(df.iloc[-1]['atr'])
        strength_for_sizing = max(0.1, (actual_strength - required_strength) / (100 - required_strength) if required_strength < 100 else 1.0)
        position_result = calculate_dynamic_position_size_enhanced(symbol, entry_price, atr_value, strength_for_sizing)
        
        if position_result is None: return signals
        quantity, stop_loss = position_result
        
        signal_data = {
            "symbol": symbol, "entry_price": entry_price, "stop_loss": stop_loss,
            "strategy_name": ", ".join(successful_strategies) or "Combined Signal",
            "signal_details": {"strategies": strategy_details, "actual_strength": actual_strength, "required_strength": required_strength, "atr": atr_value},
            "quantity": float(quantity), "is_real_trade": is_trading_enabled
        }
        signals.append(signal_data)
        logger.info(f"  -> [{symbol}] ✅ تم توليد إشارة (القوة: {actual_strength:.2f}% >= {required_strength}%)")
        return signals
    finally:
        # This block ensures the DataFrame is deleted, freeing up memory,
        # regardless of whether a signal was generated or an error occurred.
        del df

def process_open_signals():
    if not check_db_connection() or not conn or not redis_client: return
    try:
        current_prices = redis_client.hgetall("crypto_bot_prices")
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status = 'open';")
            open_signals_db = cur.fetchall()
        
        with signal_cache_lock:
            open_signals_cache.clear()
            for signal in open_signals_db:
                signal_dict = dict(signal)
                if 'exit_levels' in signal_dict and isinstance(signal_dict['exit_levels'], str):
                    signal_dict['exit_levels'] = json.loads(signal_dict['exit_levels'])
                open_signals_cache[signal_dict['symbol']] = signal_dict
        
        for symbol, signal_data in list(open_signals_cache.items()):
            df = None # Initialize for the finally block
            try:
                current_price = float(current_prices.get(symbol, 0))
                if current_price == 0: continue
                df = get_data_for_symbol(symbol, SIGNAL_GENERATION_TIMEFRAME, days=2)
                if df is None or df.empty: continue
                df = calculate_all_features(df)
                should_exit, reason = check_exit_conditions_enhanced(signal_data, current_price, df)
                if should_exit: close_signal(signal_data['id'], current_price, reason)
            except Exception as e:
                logger.error(f"❌ [Process Signals] خطأ في معالجة الإشارة لـ {symbol}: {e}")
            finally:
                del df # Clean up the DataFrame for the open signal check
    except Exception as e:
        logger.error(f"❌ [Process Signals] خطأ عام: {e}", exc_info=True)

def scan_and_generate_signals():
    global validated_symbols_to_scan
    if not (check_db_connection() and redis_client):
        logger.warning("⚠️ [Scan] الخدمات غير جاهزة، سيتم تخطي المسح.")
        return
    
    logger.info("🔍 [Scan] بدء دورة المسح الكاملة...")
    if not validated_symbols_to_scan:
        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan:
            logger.error("❌ [Scan] لا توجد عملات صالحة للمسح.")
            return

    process_open_signals()
    new_signals_total = 0
    
    for i in range(0, len(validated_symbols_to_scan), BATCH_SIZE):
        batch_symbols = validated_symbols_to_scan[i:i+BATCH_SIZE]
        num_batches = (len(validated_symbols_to_scan) + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(f" обрабатываю партию {i//BATCH_SIZE + 1}/{num_batches} ({len(batch_symbols)} символов)")
        
        for symbol in batch_symbols:
            try:
                with signal_cache_lock:
                    if symbol in open_signals_cache:
                        continue
                signals = generate_signals_for_symbol(symbol)
                if signals:
                    new_signals_total += len(signals)
                    for signal_data in signals:
                        saved_signal = insert_signal_into_db(signal_data)
                        if saved_signal:
                            with signal_cache_lock:
                                open_signals_cache[symbol] = saved_signal
                # A small sleep can help prevent overwhelming the system, though rate limiter is primary
                time.sleep(0.05) 
            except Exception as e:
                logger.error(f"❌ [Scan] خطأ في مسح {symbol}: {e}", exc_info=True)
        
        logger.info(f"🗑️ انتهت الدفعة. جاري استدعاء جامع القمامة لتحرير الذاكرة...")
        collected_count = gc.collect()
        logger.info(f"✅ تم تحرير {collected_count} كائن من الذاكرة.")

    logger.info(f"✅ [Scan] اكتمل المسح الكامل. تم توليد {new_signals_total} إشارة جديدة.")

def price_stream_processor(msg):
    for ticker in msg:
        if ticker.get('e') == '24hrTicker' and redis_client:
            try: redis_client.hset("crypto_bot_prices", ticker.get('s'), ticker.get('c'))
            except Exception as e: logger.error(f"❌ [WebSocket] خطأ في تحديث سعر {ticker.get('s')} في Redis: {e}")

def start_websocket_manager():
    if not (client and validated_symbols_to_scan): return None
    logger.info("🔌 [WebSocket] بدء مدير WebSocket...")
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    twm.start_ticker_socket(callback=price_stream_processor)
    return twm

# --- واجهة الويب (Flask) ---
app = Flask(__name__)
CORS(app)

@app.before_request
def block_method():
    if request.method in ['PUT', 'DELETE', 'PATCH']: abort(403)

def get_dashboard_html_v11_8():
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentinel V11.8 - لوحة تحكم التداول</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
    <style>
        :root { --bg-main: #0D1117; --bg-card: #161B22; --border-color: #30363D; --text-primary: #E6EDF3; --accent-blue: #58A6FF; --accent-green: #3FB950; --accent-red: #F85149; }
        body { font-family: 'Tajawal', sans-serif; background-color: var(--bg-main); color: var(--text-primary); }
        .card { background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 0.75rem; }
        .tab-btn.active { border-bottom-color: var(--accent-blue); color: var(--accent-blue); }
        .input-field { background-color: var(--bg-main); border: 1px solid var(--border-color); border-radius: 0.5rem; padding: 0.5rem 0.75rem; }
        input[type=range] { -webkit-appearance: none; background: transparent; }
        input[type=range]::-webkit-slider-runnable-track { height: 8px; background: var(--border-color); border-radius: 5px; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; height: 24px; width: 24px; border-radius: 50%; background: var(--accent-blue); margin-top: -8px; border: 4px solid var(--bg-card); }
    </style>
</head>
<body class="p-4 md:p-6">
    <div class="container mx-auto max-w-screen-2xl">
        <header class="mb-6 flex flex-wrap justify-between items-center gap-4">
            <h1 class="text-2xl md:text-3xl font-extrabold"><span class="text-blue-400">Sentinel</span> V11.8</h1>
            <div class="flex items-center gap-3">
                <div id="market-status" class="text-sm px-3 py-1 rounded-full bg-gray-800"><span id="market-status-text">...</span></div>
                <div class="text-sm px-3 py-1 rounded-full bg-gray-800"><span id="trading-status-text">معطل</span></div>
                <button id="toggle-trading" class="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg">تفعيل التداول</button>
            </div>
        </header>
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
            <div class="lg:col-span-1 flex flex-col gap-6">
                <div class="card p-5">
                    <h2 class="text-xl font-bold mb-4 text-blue-400">إحصائيات</h2>
                    <div class="space-y-4">
                        <div><div class="flex justify-between text-sm"><span>الرصيد (USDT)</span><span id="balance">--</span></div></div>
                        <div><div class="flex justify-between text-sm"><span>الصفقات المفتوحة</span><span id="open-trades">--</span></div></div>
                        <div><div class="flex justify-between text-sm"><span>وزن API</span><span id="api-weight">--</span></div></div>
                        <div><div class="flex justify-between text-sm"><span>حالة السوق</span><span id="market-state">--</span></div></div>
                    </div>
                </div>
                <div class="card p-5">
                    <h2 class="text-xl font-bold mb-4 text-blue-400">مقياس الصرامة</h2>
                    <div class="flex flex-col items-center">
                        <input type="range" id="filter-SIGNAL_STRICTNESS" min="0" max="100" value="75" class="w-full">
                        <div class="flex justify-between w-full text-xs mt-2"><span>مرن</span><span id="strictness-value" class="font-bold text-lg">75%</span><span>صارم</span></div>
                    </div>
                </div>
            </div>
            <div class="lg:col-span-2 card p-5">
                <h2 class="text-xl font-bold mb-4 text-blue-400">الإعدادات</h2>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <h3 class="font-bold mb-3">الاستراتيجيات</h3>
                        <div id="strategies-container" class="space-y-3"></div>
                    </div>
                    <div>
                        <h3 class="font-bold mb-3">الفلاتر</h3>
                        <div class="space-y-4">
                            <div><label class="block text-sm mb-1">نسبة المخاطرة للصفقة (%)</label><input type="number" id="risk-percent" class="input-field w-full" step="0.1"></div>
                            <div><label class="block text-sm mb-1">حد مؤشر ADX</label><input type="number" id="filter-ADX_THRESHOLD" class="input-field w-full" step="1"></div>
                            <div><label class="block text-sm mb-1">إغلاق بعد (شمعة)</label><input type="number" id="filter-TIME_BASED_EXIT_CANDLES" class="input-field w-full" step="5"></div>
                        </div>
                    </div>
                </div>
                <div class="mt-6 flex justify-end"><button id="save-settings" class="px-5 py-2.5 bg-green-600 hover:bg-green-700 rounded-lg font-bold">حفظ</button></div>
            </div>
        </div>
        <div class="border-b border-gray-700 mb-4">
            <nav class="-mb-px flex space-x-8" id="tabs-nav"></nav>
        </div>
        <div class="tab-content" id="tabs-content"></div>
    </div>
<script>
    const TABS = { signals: "الإشارات المفتوحة", performance: "أداء الاستراتيجيات", notifications: "الإشعارات", rejections: "سجل الرفض" };
    let currentTab = 'signals';

    function closeSignal(signalId) {
        if (confirm('هل أنت متأكد من رغبتك في إغلاق هذه الصفقة يدوياً؟')) {
            fetch(`/api/signals/close/${signalId}`, { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        alert('تم إرسال أمر الإغلاق بنجاح.');
                        loadTabData();
                    } else {
                        alert(`فشل إغلاق الصفقة: ${data.message || 'خطأ غير معروف'}`);
                    }
                })
                .catch(err => {
                    console.error('Error closing signal:', err);
                    alert('حدث خطأ أثناء محاولة إغلاق الصفقة.');
                });
        }
    }

    function setupTabs() {
        const nav = document.getElementById('tabs-nav');
        const content = document.getElementById('tabs-content');
        nav.innerHTML = Object.entries(TABS).map(([key, value]) => `<button class="tab-btn py-2 px-1 border-b-2 border-transparent" data-tab="${key}">${value}</button>`).join('');
        content.innerHTML = Object.keys(TABS).map(key => `<div id="${key}-tab" class="tab-pane hidden"></div>`).join('');
        nav.querySelector('.tab-btn').classList.add('active');
        content.querySelector('.tab-pane').classList.remove('hidden');
        nav.addEventListener('click', e => {
            if (e.target.matches('.tab-btn')) {
                currentTab = e.target.dataset.tab;
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                document.querySelectorAll('.tab-pane').forEach(p => p.classList.add('hidden'));
                document.getElementById(`${currentTab}-tab`).classList.remove('hidden');
                loadTabData();
            }
        });
    }

    function loadTabData() {
        const endpoint = { signals: '/api/signals', performance: '/api/performance', notifications: '/api/notifications', rejections: '/api/rejection_logs' }[currentTab];
        const container = document.getElementById(`${currentTab}-tab`);
        container.innerHTML = '<p class="text-center">جاري التحميل...</p>';
        fetch(endpoint)
            .then(res => res.json())
            .then(data => {
                if (!data || data.length === 0) {
                    container.innerHTML = '<p class="text-center">لا توجد بيانات.</p>';
                    return;
                }
                if (currentTab === 'signals') renderSignals(data);
                else if (currentTab === 'performance') renderPerformance(data);
                else if (currentTab === 'notifications') renderNotifications(data);
                else if (currentTab === 'rejections') renderRejections(data);
            })
            .catch(() => container.innerHTML = '<p class="text-center text-red-500">خطأ في تحميل البيانات.</p>');
    }

    function renderSignals(signals) {
        const table = `<div class="overflow-x-auto"><table class="min-w-full divide-y divide-gray-700"><thead><tr>${['العملة', 'الاستراتيجية', 'الدخول', 'الحالي', 'وقف الخسارة', 'الربح/الخسارة', 'إجراء'].map(h => `<th class="px-4 py-3 text-right text-xs font-medium uppercase">${h}</th>`).join('')}</tr></thead><tbody>${signals.map(s => { const profit = s.profit_percentage || 0; return `<tr><td class="px-4 py-3">${s.symbol}</td><td class="px-4 py-3 text-sm">${s.strategy_name}</td><td class="px-4 py-3">${(s.entry_price || 0).toFixed(4)}</td><td class="px-4 py-3">${(s.current_price || 0).toFixed(4)}</td><td class="px-4 py-3">${(s.stop_loss || 0).toFixed(4)}</td><td class="px-4 py-3 ${profit >= 0 ? 'text-green-400' : 'text-red-400'}">${profit.toFixed(2)}%</td><td class="px-4 py-3"><button class="text-red-400 hover:text-red-300" onclick="closeSignal(${s.id})">إغلاق</button></td></tr>`; }).join('')}</tbody></table></div>`;
        document.getElementById('signals-tab').innerHTML = table;
    }
    function renderPerformance(performance) {
        const table = `<div class="overflow-x-auto"><table class="min-w-full divide-y divide-gray-700"><thead><tr>${['الاستراتيجية', 'الإجمالي', 'الرابحة', 'نسبة النجاح', 'متوسط الربح/الخسارة'].map(h => `<th class="px-4 py-3 text-right text-xs font-medium uppercase">${h}</th>`).join('')}</tr></thead><tbody>${performance.map(p => { const winRate = p.total_trades > 0 ? ((p.winning_trades / p.total_trades) * 100).toFixed(1) : 0; const avgPnl = p.total_trades > 0 ? (p.total_pnl_percent / p.total_trades).toFixed(2) : 0; return `<tr><td class="px-4 py-3">${p.strategy_name}</td><td class="px-4 py-3">${p.total_trades}</td><td class="px-4 py-3">${p.winning_trades}</td><td class="px-4 py-3">${winRate}%</td><td class="px-4 py-3 ${avgPnl >= 0 ? 'text-green-400' : 'text-red-400'}">${avgPnl}%</td></tr>`; }).join('')}</tbody></table></div>`;
        document.getElementById('performance-tab').innerHTML = table;
    }
    function renderNotifications(notifications) {
        document.getElementById('notifications-tab').innerHTML = notifications.map(n => `<div class="card p-4 mb-3"><p>${n.message}</p><p class="text-xs text-gray-400 mt-1">${new Date(n.timestamp).toLocaleString()}</p></div>`).join('');
    }
    function renderRejections(rejections) {
        document.getElementById('rejections-tab').innerHTML = rejections.map(r => `<div class="card p-4 mb-3"><div><strong>${r.symbol}</strong>: ${r.reason}</div><details class="text-xs mt-1"><summary class="cursor-pointer">تفاصيل</summary><pre class="bg-gray-800 p-2 rounded mt-1 text-xs">${JSON.stringify(r.details, null, 2)}</pre></details><p class="text-xs text-gray-400 mt-1">${new Date(r.timestamp).toLocaleString()}</p></div>`).join('');
    }

    function loadInitialData() {
        fetch('/api/status').then(res => res.json()).then(data => {
            document.getElementById('balance').textContent = data.usdt_balance !== 'N/A' ? `${parseFloat(data.usdt_balance).toFixed(2)}` : '--';
            document.getElementById('open-trades').textContent = `${data.open_trades_count} / ${data.max_open_trades}`;
            document.getElementById('api-weight').textContent = `${data.api_weight} / 6000`;
            document.getElementById('market-state').textContent = data.market_state.status || '...';
            const tradingStatusEl = document.getElementById('trading-status-text');
            tradingStatusEl.textContent = data.is_trading_enabled ? 'مفعل' : 'معطل';
            
            const strategiesContainer = document.getElementById('strategies-container');
            strategiesContainer.innerHTML = Object.entries(data.settings.strategies).map(([key, val]) => `<div class="flex items-center justify-between p-2 bg-gray-900 rounded"><label for="strategy-${key}">${val.display_name}</label><input type="checkbox" id="strategy-${key}" ${val.enabled ? 'checked' : ''}></div>`).join('');
            
            Object.entries(data.settings.filters).forEach(([key, val]) => {
                const input = document.getElementById(`filter-${key}`);
                if (input) input.value = val.value;
            });
            document.getElementById('risk-percent').value = data.settings.risk_percent;
            const strictnessSlider = document.getElementById('filter-SIGNAL_STRICTNESS');
            document.getElementById('strictness-value').textContent = `${strictnessSlider.value}%`;
        }).catch(err => console.error("Error loading status:", err));
        loadTabData();
    }
    
    document.getElementById('toggle-trading').addEventListener('click', () => {
        fetch('/api/trading/toggle', { method: 'POST' }).then(() => loadInitialData());
    });

    document.getElementById('save-settings').addEventListener('click', () => {
        const settings = { strategies: {}, filters: {} };
        document.querySelectorAll('[id^="strategy-"]').forEach(el => settings.strategies[el.id.replace('strategy-', '')] = el.checked);
        document.querySelectorAll('[id^="filter-"]').forEach(el => settings.filters[el.id.replace('filter-', '')] = el.type === 'number' || el.type === 'range' ? parseFloat(el.value) : el.value);
        settings.risk_percent = parseFloat(document.getElementById('risk-percent').value);
        
        fetch('/api/settings/update', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(settings) })
            .then(res => res.json()).then(data => alert(data.success ? 'تم الحفظ' : 'خطأ'));
    });

    document.getElementById('filter-SIGNAL_STRICTNESS').addEventListener('input', e => {
        document.getElementById('strictness-value').textContent = `${e.target.value}%`;
    });

    document.addEventListener('DOMContentLoaded', () => {
        setupTabs();
        loadInitialData();
        setInterval(loadInitialData, 30000);
    });
</script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(get_dashboard_html_v11_8())

@app.route('/api/status')
def get_status():
    with market_state_lock: state_copy = dict(current_market_state)
    with trading_status_lock: is_enabled = is_trading_enabled
    with throttler.lock: weight = throttler.total_weight_used_minute
    with signal_cache_lock: open_trades = len(open_signals_cache)
    usdt_balance = 'N/A'
    if client:
        try: usdt_balance = float(client.get_asset_balance(asset='USDT')['free'])
        except: pass
    with risk_per_trade_lock: risk = RISK_PER_TRADE_PERCENT
    strategy_settings = {key: {"enabled": config['enabled'], "display_name": config['display_name']} for key, config in STRATEGY_CONFIG.items()}
    filter_settings = {key: {"value": config['value'], "display_name": config['display_name']} for key, config in FILTER_CONFIG.items()}
    return jsonify({
        "market_state": state_copy, "is_trading_enabled": is_enabled, "usdt_balance": usdt_balance,
        "api_weight": weight, "open_trades_count": open_trades, "max_open_trades": MAX_OPEN_TRADES,
        "settings": { "risk_percent": risk, "strategies": strategy_settings, "filters": filter_settings }
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
                signal['current_price'] = float(current_price)
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
        return jsonify({"success": True})

@app.route('/api/settings/update', methods=['POST'])
def update_settings():
    try:
        data = request.get_json()
        with risk_per_trade_lock: 
            global RISK_PER_TRADE_PERCENT
            RISK_PER_TRADE_PERCENT = float(data.get('risk_percent', RISK_PER_TRADE_PERCENT))
        for key, is_enabled in data.get('strategies', {}).items():
            if key in STRATEGY_CONFIG:
                with STRATEGY_CONFIG[key]['lock']: STRATEGY_CONFIG[key]['enabled'] = bool(is_enabled)
        for key, value in data.get('filters', {}).items():
            if key in FILTER_CONFIG:
                with FILTER_CONFIG[key]['lock']:
                    current_type = type(FILTER_CONFIG[key]['value'])
                    FILTER_CONFIG[key]['value'] = current_type(float(value)) if current_type in [int, float] else str(value)
        log_and_notify('info', "⚙️ تم تحديث الإعدادات من لوحة التحكم.", "SETTINGS_UPDATE")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"❌ [API Settings] فشل تحديث الإعدادات: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 400

@app.route('/api/signals/close/<int:signal_id>', methods=['POST'])
def manual_close_trade_endpoint(signal_id):
    if not (redis_client and client): return jsonify({"success": False, "message": "Services not ready"}), 503
    with signal_cache_lock: signal_to_close = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)
    if not signal_to_close: return jsonify({"success": False, "message": "Signal not found"}), 404
    try:
        current_price = float(redis_client.hget("crypto_bot_prices", signal_to_close['symbol']))
    except:
        try: current_price = float(client.get_symbol_ticker(symbol=signal_to_close['symbol'])['price'])
        except Exception as e: return jsonify({"success": False, "message": f"Could not fetch price: {e}"}), 500
    if close_signal(signal_id, current_price, 'manual'): return jsonify({"success": True})
    else: return jsonify({"success": False, "message": "Failed to close signal."}), 500

# --- الدوال الرئيسية لتشغيل البوت ---
def main():
    global client
    logger.info("🚀 بدء تشغيل نظام التداول الآلي Sentinel V11.8")
    init_db()
    init_redis()
    client = Client(API_KEY, API_SECRET)
    get_exchange_info_map()
    global validated_symbols_to_scan
    validated_symbols_to_scan = get_validated_symbols()
    if not validated_symbols_to_scan:
        logger.critical("❌ لا توجد عملات صالحة للمسح. إنهاء التشغيل.")
        return
    
    twm = start_websocket_manager()
    web_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=5000, threaded=True), daemon=True)
    web_thread.start()
    logger.info("🌐 بدء واجهة الويب على http://localhost:5000")
    
    if redis_client:
        try:
            initial_prices = {s['symbol']: s['price'] for s in client.get_all_tickers() if s['symbol'] in validated_symbols_to_scan}
            redis_client.hset("crypto_bot_prices", mapping=initial_prices)
        except Exception as e:
            logger.error(f"❌ خطأ في جلب الأسعار الأولية: {e}")
    
    try:
        while True:
            scan_and_generate_signals()
            gc.collect() 
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("🛑 تم إيقاف البوت يدوياً")
    finally:
        if twm: twm.stop()
        logger.info("🔚 تم إنهاء تشغيل البوت")

if __name__ == "__main__":
    main()
