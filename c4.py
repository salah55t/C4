# ملف c4.py - نسخة V13.0.0 (إعادة تصميم الواجهة)
# --- التغييرات الرئيسية (V13.0.0):
# 1. [واجهة مستخدم جديدة] إعادة تصميم كاملة للوحة التحكم باستخدام Tailwind CSS.
# 2. [وضع ليلي] تطبيق تصميم داكن احترافي لتحسين تجربة المستخدم.
# 3. [صفحة إعدادات] إضافة صفحة إعدادات مخصصة ومنفصلة للتحكم الكامل.
# 4. [تفعيل الاستراتيجيات] إضافة مفاتيح تبديل سهلة لتفعيل/إلغاء كل استراتيجية على حدة.
# 5. [تحسين الأداء] استخدام JavaScript لتحميل البيانات بشكل غير متزامن وتحسين استجابة الواجهة.

import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import redis
import gc
from decimal import Decimal
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timezone
from decouple import config
from typing import List, Dict, Optional, Any
from collections import deque
import warnings

# --- إعدادات التجاهل واللوجر ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v13_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV13.0.0')

# --- المشفر المخصص لأنواع بيانات NumPy ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
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

# --- متغيرات عامة وإعدادات البوت ---
is_trading_enabled: bool = False
trading_status_lock = Lock()
paper_trading_mode: bool = True

# --- المتغيرات القابلة للتعديل ---
RISK_PER_TRADE_PERCENT: float = 0.85
risk_per_trade_lock = Lock()
BUY_CONFIDENCE_THRESHOLD = 0.53
buy_confidence_lock = Lock()
MAX_OPEN_TRADES: int = 3
MIN_PROFIT_PERCENT: float = 0.8
PAPER_TRADE_SIZE_USDT: float = 10.0

# --- إعدادات إدارة الصفقات المتقدمة ---
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_STOP_TRIGGER_PERCENT: float = 0.4
TRAILING_STOP_DISTANCE_PERCENT: float = 0.5
USE_PARTIAL_TAKE_PROFIT: bool = True
PARTIAL_TP_RSI_THRESHOLD: float = 60
USE_VOLUME_PROFILE_STRATEGY: bool = True
volume_profile_strategy_lock = Lock()

# --- مفاتيح تفعيل الاستراتيجيات ---
USE_BB_STOCH_STRATEGY: bool = True
bb_stoch_strategy_lock = Lock()
USE_MACD_EMA_STRATEGY: bool = True
macd_ema_strategy_lock = Lock()
USE_EMA_RSI_STRATEGY: bool = True
ema_rsi_strategy_lock = Lock()
USE_PULLBACK_STRATEGY: bool = True
pullback_strategy_lock = Lock()

# --- إعدادات عامة ---
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '1h'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 15
BTC_SYMBOL: str = 'BTCUSDT'
SYMBOL_PROCESSING_BATCH_SIZE: int = 5
ATR_TS_MULTIPLIER: float = 2.2
TRADING_FEE_PERCENT: float = 0.1
API_REQUEST_DELAY: float = 0.5
API_RETRY_COUNT: int = 3
API_RETRY_DELAY: float = 5.0

# --- إعدادات المؤشرات الفنية ---
EMA_FAST_PERIOD: int = 12
EMA_SLOW_PERIOD: int = 26
ADX_PERIOD: int = 10
RSI_PERIOD: int = 10
ATR_PERIOD: int = 10
MOMENTUM_PERIOD: int = 5

# --- متغيرات الحالة والكاش ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
ws_manager: Optional[ThreadedWebsocketManager] = None
live_prices: Dict[str, float] = {}
live_prices_lock = Lock()
exchange_info_map: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
notifications_cache = deque(maxlen=20)
notifications_lock = Lock()
rejection_logs_cache = deque(maxlen=30)
rejection_logs_lock = Lock()
current_market_state: Dict[str, Any] = {"overall_regime": "INITIALIZING", "trend_details_by_tf": {}, "last_updated": "N/A"}
market_state_lock = Lock()

# --- قاموس أسباب الرفض باللغة العربية ---
REJECTION_REASONS_AR = {
    "Market Volatility Filter Failed": "فلتر تقلب السوق رفض الدخول",
    "Trend Strength Filter Failed": "فلتر قوة الاتجاه رفض الدخول",
    "HTF Trend Confirmation Failed": "فشل تأكيد الترند على الفريم الأعلى",
    "Bullish Reversal Candle Pattern Failed": "لم يظهر نمط شمعة انعكاسية صاعدة",
    "Insufficient Historical Data": "بيانات تاريخية غير كافية للفحص",
    "Volume Profile Strategy Failed": "استراتيجية ملف الحجم رفضت الدخول"
}

# --- إعداد تطبيق Flask ---
app = Flask(__name__)
CORS(app)

# --- دوال تهيئة الخدمات ---
def init_db(retries: int = 5, delay: int = 5) -> None:
    """تهيئة الاتصال بقاعدة البيانات PostgreSQL وإنشاء الجداول اللازمة."""
    global conn
    logger.info("[DB] Initializing database connection...")
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
                        is_real_trade BOOLEAN DEFAULT FALSE, quantity DOUBLE PRECISION, closing_reason TEXT,
                        target_price_2 DOUBLE PRECISION, initial_quantity DOUBLE PRECISION
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL, message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE
                    );
                """)
            conn.commit()
            logger.info("✅ [DB] Database connection and schema updated successfully.")
            return
        except Exception as e:
            logger.error(f"❌ [DB] Error during initialization (Attempt {attempt + 1}/{retries}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: logger.critical("❌ [DB] Failed to connect to the database. Exiting.")

def check_db_connection() -> bool:
    """يتحقق من حالة الاتصال بقاعدة البيانات ويعيد الاتصال إذا لزم الأمر."""
    global conn
    if conn is None or conn.closed != 0:
        logger.warning("[DB] Connection is closed. Attempting to reconnect...")
        init_db()
    try:
        if conn and conn.closed == 0:
            with conn.cursor() as cur: cur.execute("SELECT 1;")
            return True
        return False
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [DB] Connection lost: {e}. Reconnecting...")
        init_db()
        return conn is not None and conn.closed == 0

def init_redis() -> None:
    """تهيئة الاتصال بذاكرة التخزين المؤقت Redis."""
    global redis_client
    logger.info("[Redis] Initializing connection...")
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] Connected successfully.")
    except redis.exceptions.ConnectionError as e:
        logger.warning(f"⚠️ [Redis] Connection failed: {e}. The bot will run without Redis.")
        redis_client = None

# --- دوال المساعدة والإشعارات ---
def log_and_notify(level: str, message: str, notification_type: str):
    """يسجل رسالة في السجل ويحفظها كإشعار في قاعدة البيانات."""
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error}
    log_methods.get(level.lower(), logger.info)(message)
    if not check_db_connection() or not conn:
        logger.error(f"[DB] Could not save notification due to DB connection issue: {message}")
        return
    try:
        new_notification = {"timestamp": datetime.now(timezone.utc).isoformat(), "type": notification_type, "message": message}
        with notifications_lock: notifications_cache.appendleft(new_notification)
        with conn.cursor() as cur: cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [DB] Failed to save notification: {e}")
        if conn: conn.rollback()

def log_rejection(symbol: str, reason_key: str, details: Optional[Dict] = None):
    """يسجل سبب رفض إشارة دخول في الذاكرة المؤقتة."""
    reason_ar = REJECTION_REASONS_AR.get(reason_key, reason_key)
    with rejection_logs_lock:
        rejection_logs_cache.appendleft({
            "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
            "reason": reason_ar, "details": json.loads(json.dumps(details or {}, cls=NpEncoder))
        })

def send_telegram_message(message: str):
    """يرسل رسالة إلى تيليجرام."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [Telegram] Failed to send message: {e}")

# --- WebSocket Handler ---
def handle_socket_message(msg):
    """يعالج الرسائل الواردة من WebSocket لتحديث الأسعار الحية."""
    global live_prices
    if msg and 'e' in msg and msg['e'] == 'error':
        logger.error(f"❌ [WebSocket] Error: {msg['m']}")
        return
    if isinstance(msg, list):
        with live_prices_lock:
            for ticker in msg:
                if 's' in ticker and 'c' in ticker:
                    live_prices[ticker['s']] = float(ticker['c'])

def start_websocket():
    """يبدأ مدير WebSocket للاستماع إلى تحديثات الأسعار."""
    global ws_manager
    logger.info("🚀 [WebSocket] Starting WebSocket manager...")
    ws_manager = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    ws_manager.start()
    ws_manager.start_ticker_socket(callback=handle_socket_message)
    logger.info("✅ [WebSocket] Successfully subscribed to ticker stream (!ticker@arr).")

# --- دوال جلب البيانات وحساب المؤشرات ---
def get_exchange_info_map() -> None:
    """يجلب معلومات البورصة وينشئ خريطة للرموز."""
    global exchange_info_map
    if not client: return
    try:
        logger.info("[API] Fetching exchange info...")
        info = client.get_exchange_info()
        exchange_info_map = {s['symbol']: s for s in info['symbols']}
        logger.info(f"[API] Exchange info map created with {len(exchange_info_map)} symbols.")
    except BinanceAPIException as e:
        logger.error(f"❌ [API] Binance error fetching exchange info: {e}")
    except Exception as e:
        logger.error(f"❌ [API] Generic error fetching exchange info: {e}")

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """يقرأ قائمة الرموز ويتحقق من صحتها مقابل البورصة."""
    if not client: return []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        if not os.path.exists(file_path):
            logger.critical(f"❌ Symbol list file '{filename}' not found!"); return []
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        if not exchange_info_map: get_exchange_info_map()
        active = {s for s, info in exchange_info_map.items() if info.get('quoteAsset') == 'USDT' and info.get('status') == 'TRADING'}
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ Found {len(validated)} valid symbols for trading.")
        return validated
    except Exception as e:
        logger.error(f"❌ [Symbols] Error validating symbols: {e}"); return []

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """يجلب البيانات التاريخية للشموع من Binance."""
    if not client: return None
    time.sleep(API_REQUEST_DELAY)
    try:
        lookback_str = f"{days} day ago UTC"
        klines = client.get_historical_klines(symbol, interval, lookback_str)
        if not klines: return None
        processed_klines = [kline[:6] for kline in klines]
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(processed_klines, columns=cols)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna().astype(float)
    except BinanceAPIException as e:
        logger.error(f"❌ [API] Binance error fetching data for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ [Data] Generic error fetching data for {symbol}: {e}"); return None

def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """تحسب جميع المؤشرات الفنية الأساسية (EMA, ATR, ADX, RSI, MACD)."""
    df_calc = df.copy()
    df_calc['ema_9'] = df_calc['close'].ewm(span=9, adjust=False).mean()
    df_calc['ema_12'] = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    df_calc['ema_26'] = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
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
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    rsi_val = df_calc['rsi']
    stoch_rsi = (rsi_val - rsi_val.rolling(14).min()) / (rsi_val.rolling(14).max() - rsi_val.rolling(14).min()).replace(0, 1e-9)
    df_calc['stoch_rsi'] = stoch_rsi.rolling(3).mean() * 100
    df_calc['macd'] = df_calc['ema_12'] - df_calc['ema_26']
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()
    df_calc['macd_hist'] = df_calc['macd'] - df_calc['macd_signal']
    df_calc['lower_band'] = df_calc['close'].rolling(20).mean() - (df_calc['close'].rolling(20).std() * 2)
    df_calc['upper_band'] = df_calc['close'].rolling(20).mean() + (df_calc['close'].rolling(20).std() * 2)
    return df_calc.dropna().astype(float)

def calculate_market_trend(df: pd.DataFrame) -> str:
    """تحسب اتجاه السوق بناءً على EMA و ADX."""
    last = df.iloc[-1]
    last_but_one = df.iloc[-2]
    is_up_trend = (last['ema_9'] > last['ema_26'] and
                   last_but_one['ema_9'] > last_but_one['ema_26'] and
                   last['adx'] > 20)
    is_down_trend = (last['ema_9'] < last['ema_26'] and
                     last_but_one['ema_9'] < last_but_one['ema_26'] and
                     last['adx'] > 20)
    if is_up_trend: return 'اتجاه صاعد'
    if is_down_trend: return 'اتجاه هابط'
    return 'سوق عرضي'

def is_bullish_reversal_pattern(df: pd.DataFrame) -> bool:
    """تتحقق من وجود نمط شمعة انعكاسية صاعدة (مثال: مطرقة، ابتلاع صاعد)."""
    if len(df) < 2: return False
    last = df.iloc[-1]
    prev = df.iloc[-2]
    hammer = (last['close'] > last['open'] and last['low'] < last['open'] * 0.99 and (last['open'] - last['low']) > 2 * (last['close'] - last['open']))
    engulfing = (prev['close'] < prev['open'] and last['close'] > last['open'] and last['close'] > prev['open'] and last['open'] < prev['close'])
    return hammer or engulfing

def is_bearish_reversal_pattern(df: pd.DataFrame) -> bool:
    """تتحقق من وجود نمط شمعة انعكاسية هابطة (مثال: شهاب، ابتلاع هابط)."""
    if len(df) < 2: return False
    last = df.iloc[-1]
    prev = df.iloc[-2]
    shooting_star = (last['close'] < last['open'] and last['high'] > last['close'] * 1.01 and (last['high'] - last['close']) > 2 * (last['open'] - last['close']))
    engulfing = (prev['close'] > prev['open'] and last['close'] < last['open'] and last['close'] < prev['open'] and last['open'] > prev['close'])
    return shooting_star or engulfing

# --- استراتيجيات الدخول (جديد) ---
def calculate_volume_profile(df: pd.DataFrame):
    """يحسب ملف الحجم (Volume Profile) ونقطة التحكم (POC) ومنطقة القيمة (VA)."""
    price_bins = pd.cut(df['close'], bins=20, labels=False)
    volume_by_bin = df.groupby(price_bins)['volume'].sum()
    poc_bin = volume_by_bin.idxmax()
    poc = df['close'].iloc[price_bins[price_bins == poc_bin].index].mean()
    total_volume = volume_by_bin.sum()
    sorted_bins = volume_by_bin.sort_values(ascending=False)
    cumulative_volume = 0
    value_area_bins = []
    for bin_label, volume in sorted_bins.items():
        cumulative_volume += volume
        value_area_bins.append(bin_label)
        if cumulative_volume >= total_volume * 0.7:
            break
    value_area_prices = df['close'].iloc[price_bins[price_bins.isin(value_area_bins)].index]
    value_area_high = value_area_prices.max()
    value_area_low = value_area_prices.min()
    return poc, value_area_high, value_area_low

def check_volume_profile_strategy(df: pd.DataFrame) -> bool:
    """استراتيجية ملف الحجم: تتحقق من أن السعر فوق منطقة القيمة، مع حجم تداول وقوة اتجاه عالية."""
    last = df.iloc[-1]
    poc, value_area_high, value_area_low = calculate_volume_profile(df)
    above_value_area = last['close'] > value_area_high
    high_volume = last['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.5
    price_action = last['close'] > df['high'].iloc[-2]
    trend_strength = last['adx'] > 18
    return above_value_area and high_volume and price_action and trend_strength
    
# --- استراتيجيات الخروج المتقدمة (جديد) ---
def check_market_structure_change(df: pd.DataFrame, signal: Dict) -> bool:
    """استراتيجية خروج: تتحقق من تغير في هيكل السوق مثل كسر قاع مهم أو ظهور نمط انعكاسي قوي."""
    last = df.iloc[-1]
    market_structure_broken = False
    lows = df['low'].rolling(5).min()
    if last['close'] < lows.iloc[-2] * 0.995: market_structure_broken = True
    if is_bearish_reversal_pattern(df): market_structure_broken = True
    if last['adx'] < 15: market_structure_broken = True
    return market_structure_broken

def check_reversal_signal_exit(df: pd.DataFrame, signal: Dict) -> bool:
    """استراتيجية خروج: تتحقق من ظهور إشارات عكسية متعددة مثل تقاطع MACD أو ذروة شراء في RSI."""
    last = df.iloc[-1]
    macd_bearish_cross = (df['macd'].iloc[-2] > df['macd_signal'].iloc[-2] and last['macd'] < last['macd_signal'] and last['macd'] > 0)
    rsi_overbought = last['rsi'] > 70
    bearish_candle = is_bearish_reversal_pattern(df)
    high_volume_reversal = (last['volume'] > df['volume'].rolling(10).mean().iloc[-1] * 1.5 and last['close'] < last['open'] and (last['close'] - last['low']) > (last['high'] - last['close']) * 2)
    reversal_signals = [macd_bearish_cross, rsi_overbought, bearish_candle, high_volume_reversal]
    return sum(reversal_signals) >= 2

# --- دوال إدارة الصفقات (تحسين) ---
def calculate_trailing_stop_loss(df: pd.DataFrame, entry_price: float, initial_stop: float, highest_price: float) -> float:
    """يحسب وقف الخسارة المتحرك بناءً على ATR ونسبة الربح المحققة."""
    if not USE_TRAILING_STOP_LOSS: return initial_stop
    last = df.iloc[-1]
    atr = last['atr']
    profit_percent = ((highest_price - entry_price) / entry_price) * 100
    if profit_percent >= TRAILING_STOP_TRIGGER_PERCENT:
        trailing_distance = atr * TRAILING_STOP_DISTANCE_PERCENT
        trailing_stop = highest_price - trailing_distance
        return max(trailing_stop, initial_stop)
    return initial_stop

def check_partial_take_profit(df: pd.DataFrame, signal: Dict, current_price: float) -> Dict:
    """يتحقق من شروط أخذ الربح الجزئي، مع الأخذ في الاعتبار قوة الاتجاه عبر RSI."""
    if not USE_PARTIAL_TAKE_PROFIT: return {"action": "hold"}
    last = df.iloc[-1]
    entry_price = signal['entry_price']
    target_price_1 = signal['target_price']
    initial_quantity = signal['initial_quantity']
    current_quantity = signal['quantity']
    if current_price >= target_price_1 and current_quantity == initial_quantity:
        if last['rsi'] >= PARTIAL_TP_RSI_THRESHOLD:
            new_quantity = initial_quantity * 0.5
            with signal_cache_lock: open_signals_cache[signal['symbol']]['quantity'] = new_quantity
            message = (f"📊 *أخذ ربح جزئي*\\n"
                       f"💱 *العملة:* `{signal['symbol']}`\\n"
                       f"🎯 *الهدف 1 مُحقق:* `{target_price_1:.4f}`\\n"
                       f"📈 *الهدف 2:* `{signal['target_price_2']:.4f}`\\n"
                       f"📉 *الكمية المتبقية:* `{new_quantity:.4f}`")
            send_telegram_message(message)
            log_and_notify("info", f"تم أخذ ربح جزئي لـ {signal['symbol']} والاستمرار للهدف الثاني", "PARTIAL_TP")
            return {"action": "partial_tp", "new_quantity": new_quantity}
        else:
            return {"action": "close_all", "reason": "weak_trend"}
    return {"action": "hold"}
    
def check_for_close_conditions(df: pd.DataFrame, signal: Dict, current_price: float) -> bool:
    """يتحقق من جميع شروط الخروج الجديدة."""
    if check_market_structure_change(df, signal):
        logger.warning(f"❌ [Close Signal] إشارة خروج لـ {signal['symbol']} بسبب تغير هيكل السوق.")
        return True
    if check_reversal_signal_exit(df, signal):
        logger.warning(f"❌ [Close Signal] إشارة خروج لـ {signal['symbol']} بسبب ظهور إشارة عكسية.")
        return True
    stop_loss_hit = current_price <= signal['stop_loss']
    take_profit_hit = current_price >= signal['target_price']
    take_profit_2_hit = signal.get('target_price_2') and current_price >= signal['target_price_2']
    if stop_loss_hit:
        logger.warning(f"❌ [Close Signal] إشارة خروج لـ {signal['symbol']} بسبب ضرب وقف الخسارة.")
        return True
    if take_profit_hit:
        if take_profit_2_hit:
            logger.info(f"✅ [Close Signal] إشارة خروج لـ {signal['symbol']} بسبب تحقيق الهدف الثاني.")
            return True
        else:
            partial_tp_result = check_partial_take_profit(df, signal, current_price)
            if partial_tp_result['action'] == 'close_all':
                logger.info(f"✅ [Close Signal] إشارة خروج لـ {signal['symbol']} بسبب ضعف الترند عند الهدف الأول.")
                return True
    return False

# --- دوال التعامل مع قاعدة البيانات (تحديثات طفيفة) ---
def create_signal_db(signal_data: Dict) -> bool:
    """ينشئ صفقة جديدة في قاعدة البيانات."""
    if not check_db_connection(): return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, is_real_trade, quantity, target_price_2, initial_quantity)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
            """, (signal_data['symbol'], signal_data['entry_price'], signal_data['target_price'],
                  signal_data['stop_loss'], signal_data['strategy'], json.dumps(signal_data['details'], cls=NpEncoder),
                  not paper_trading_mode, signal_data['quantity'], signal_data['target_price_2'], signal_data['initial_quantity']))
            signal_id = cur.fetchone()['id']
            conn.commit()
            log_and_notify("info", f"✅ [DB] تم حفظ إشارة جديدة ID: {signal_id} لـ {signal_data['symbol']}", "DB_ACTION")
            return True
    except Exception as e:
        logger.error(f"❌ [DB] فشل حفظ الإشارة: {e}")
        if conn: conn.rollback()
        return False

def get_open_signals() -> List[Dict]:
    """يجلب جميع الإشارات المفتوحة من قاعدة البيانات."""
    if not check_db_connection(): return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status = 'open';")
            signals = cur.fetchall()
        return [dict(s) for s in signals]
    except Exception as e:
        logger.error(f"❌ [DB] فشل جلب الإشارات المفتوحة: {e}"); return []

def update_signal_db(signal_id: int, updates: Dict) -> bool:
    """يحدث صفقة موجودة في قاعدة البيانات."""
    if not check_db_connection(): return False
    try:
        with conn.cursor() as cur:
            set_clause = ", ".join([f"{key} = %s" for key in updates.keys()])
            query = f"UPDATE signals SET {set_clause} WHERE id = %s"
            cur.execute(query, list(updates.values()) + [signal_id])
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"❌ [DB] فشل تحديث الإشارة ID {signal_id}: {e}")
        if conn: conn.rollback()
        return False

def close_signal_db(signal: Dict, closing_price: float, closing_reason: str):
    """يغلق صفقة في قاعدة البيانات ويحسب الربح."""
    if not check_db_connection(): return
    try:
        entry_price = signal['entry_price']
        profit_percentage = ((closing_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        profit_percentage_after_fees = profit_percentage - (TRADING_FEE_PERCENT * 2)
        
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE signals SET status = 'closed', closing_price = %s, closed_at = NOW(),
                profit_percentage = %s, closing_reason = %s WHERE id = %s;
            """, (closing_price, profit_percentage_after_fees, closing_reason, signal['id']))
            conn.commit()
            log_and_notify("info", f"✅ [DB] تم إغلاق الإشارة ID: {signal['id']} لـ {signal['symbol']} بنجاح.", "DB_ACTION")
    except Exception as e:
        logger.error(f"❌ [DB] فشل إغلاق الإشارة ID {signal['id']}: {e}")
        if conn: conn.rollback()

def load_open_signals_to_cache() -> None:
    """يحمل الإشارات المفتوحة من قاعدة البيانات إلى الذاكرة المؤقتة عند بدء التشغيل."""
    global open_signals_cache
    if not check_db_connection(): return
    try:
        signals = get_open_signals()
        with signal_cache_lock:
            open_signals_cache = {s['symbol']: s for s in signals}
        logger.info(f"✅ [Cache] تم تحميل {len(signals)} إشارة مفتوحة إلى الذاكرة المؤقتة.")
    except Exception as e:
        logger.error(f"❌ [Cache] فشل تحميل الإشارات المفتوحة: {e}")

def load_notifications_to_cache() -> None:
    """يحمل الإشعارات الأخيرة من قاعدة البيانات إلى الذاكرة المؤقتة عند بدء التشغيل."""
    if not check_db_connection(): return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 20;")
            notifs = cur.fetchall()
        with notifications_lock:
            # تحويل RealDictRow إلى dict قبل الإضافة
            notifications_cache.extendleft([dict(n) for n in notifs])
        logger.info(f"✅ [Cache] تم تحميل {len(notifs)} إشعارًا إلى الذاكرة المؤقتة.")
    except Exception as e:
        logger.error(f"❌ [Cache] فشل تحميل الإشعارات: {e}")

# --- دوال إعدادات Redis (جديد) ---
def save_settings_to_redis() -> None:
    """يحفظ إعدادات البوت القابلة للتغيير إلى Redis."""
    if not redis_client: return
    try:
        settings = {
            'is_trading_enabled': is_trading_enabled,
            'paper_trading_mode': paper_trading_mode,
            'RISK_PER_TRADE_PERCENT': RISK_PER_TRADE_PERCENT,
            'BUY_CONFIDENCE_THRESHOLD': BUY_CONFIDENCE_THRESHOLD,
            'USE_TRAILING_STOP_LOSS': USE_TRAILING_STOP_LOSS,
            'USE_PARTIAL_TAKE_PROFIT': USE_PARTIAL_TAKE_PROFIT,
            'USE_VOLUME_PROFILE_STRATEGY': USE_VOLUME_PROFILE_STRATEGY,
            'USE_BB_STOCH_STRATEGY': USE_BB_STOCH_STRATEGY,
            'USE_MACD_EMA_STRATEGY': USE_MACD_EMA_STRATEGY,
            'USE_EMA_RSI_STRATEGY': USE_EMA_RSI_STRATEGY,
            'USE_PULLBACK_STRATEGY': USE_PULLBACK_STRATEGY,
        }
        redis_client.set('bot_settings', json.dumps(settings))
        logger.info("✅ [Redis] تم حفظ الإعدادات بنجاح.")
    except Exception as e:
        logger.error(f"❌ [Redis] فشل حفظ الإعدادات: {e}")

def load_settings_from_redis() -> None:
    """يحمل إعدادات البوت من Redis عند بدء التشغيل."""
    global is_trading_enabled, paper_trading_mode, RISK_PER_TRADE_PERCENT, BUY_CONFIDENCE_THRESHOLD
    global USE_TRAILING_STOP_LOSS, USE_PARTIAL_TAKE_PROFIT, USE_VOLUME_PROFILE_STRATEGY
    global USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY
    if not redis_client: return
    try:
        settings_json = redis_client.get('bot_settings')
        if settings_json:
            settings = json.loads(settings_json)
            is_trading_enabled = settings.get('is_trading_enabled', is_trading_enabled)
            paper_trading_mode = settings.get('paper_trading_mode', paper_trading_mode)
            RISK_PER_TRADE_PERCENT = settings.get('RISK_PER_TRADE_PERCENT', RISK_PER_TRADE_PERCENT)
            BUY_CONFIDENCE_THRESHOLD = settings.get('BUY_CONFIDENCE_THRESHOLD', BUY_CONFIDENCE_THRESHOLD)
            USE_TRAILING_STOP_LOSS = settings.get('USE_TRAILING_STOP_LOSS', USE_TRAILING_STOP_LOSS)
            USE_PARTIAL_TAKE_PROFIT = settings.get('USE_PARTIAL_TAKE_PROFIT', USE_PARTIAL_TAKE_PROFIT)
            USE_VOLUME_PROFILE_STRATEGY = settings.get('USE_VOLUME_PROFILE_STRATEGY', USE_VOLUME_PROFILE_STRATEGY)
            USE_BB_STOCH_STRATEGY = settings.get('USE_BB_STOCH_STRATEGY', USE_BB_STOCH_STRATEGY)
            USE_MACD_EMA_STRATEGY = settings.get('USE_MACD_EMA_STRATEGY', USE_MACD_EMA_STRATEGY)
            USE_EMA_RSI_STRATEGY = settings.get('USE_EMA_RSI_STRATEGY', USE_EMA_RSI_STRATEGY)
            USE_PULLBACK_STRATEGY = settings.get('USE_PULLBACK_STRATEGY', USE_PULLBACK_STRATEGY)
            logger.info("✅ [Redis] تم تحميل الإعدادات بنجاح.")
    except Exception as e:
        logger.error(f"❌ [Redis] فشل تحميل الإعدادات: {e}")

# --- دوال الواجهة (API) ---
@app.route('/status', methods=['GET'])
def get_status():
    """نقطة نهاية للحصول على حالة البوت الحالية."""
    with trading_status_lock:
        is_enabled = is_trading_enabled
    # استخدام json.dumps مع NpEncoder لضمان تحويل البيانات بشكل صحيح
    return jsonify(json.loads(json.dumps({
        'status': 'Running',
        'trading_enabled': is_enabled,
        'paper_trading_mode': paper_trading_mode,
        'open_trades_count': len(open_signals_cache),
        'max_open_trades': MAX_OPEN_TRADES,
        'risk_per_trade_percent': RISK_PER_TRADE_PERCENT,
        'buy_confidence_threshold': BUY_CONFIDENCE_THRESHOLD,
        'market_state': current_market_state,
        'strategies_enabled': {
            'bb_stoch': USE_BB_STOCH_STRATEGY,
            'macd_ema': USE_MACD_EMA_STRATEGY,
            'ema_rsi': USE_EMA_RSI_STRATEGY,
            'pullback': USE_PULLBACK_STRATEGY,
            'volume_profile': USE_VOLUME_PROFILE_STRATEGY
        },
        'advanced_features_enabled': {
            'trailing_stop_loss': USE_TRAILING_STOP_LOSS,
            'partial_take_profit': USE_PARTIAL_TAKE_PROFIT
        }
    }, cls=NpEncoder)))

@app.route('/open_trades', methods=['GET'])
def get_open_trades():
    """نقطة نهاية للحصول على الصفقات المفتوحة."""
    with signal_cache_lock:
        trades = list(open_signals_cache.values())
    return jsonify(json.loads(json.dumps({'open_trades': trades}, cls=NpEncoder)))

@app.route('/rejection_logs', methods=['GET'])
def get_rejection_logs():
    """نقطة نهاية للحصول على سجلات الرفض."""
    with rejection_logs_lock:
        logs = list(rejection_logs_cache)
    return jsonify(json.loads(json.dumps({'rejection_logs': logs}, cls=NpEncoder)))

@app.route('/notifications', methods=['GET'])
def get_notifications():
    """نقطة نهاية للحصول على الإشعارات."""
    with notifications_lock:
        notifs = list(notifications_cache)
    return jsonify(json.loads(json.dumps({'notifications': notifs}, cls=NpEncoder)))

@app.route('/settings', methods=['POST'])
def update_settings():
    """نقطة نهاية لتحديث إعدادات البوت."""
    global is_trading_enabled, paper_trading_mode, RISK_PER_TRADE_PERCENT, BUY_CONFIDENCE_THRESHOLD
    global USE_TRAILING_STOP_LOSS, USE_PARTIAL_TAKE_PROFIT, USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY, USE_VOLUME_PROFILE_STRATEGY

    try:
        data = request.json
        if data is None:
            return jsonify({"success": False, "message": "Invalid JSON payload."}), 400
            
        with trading_status_lock:
            if 'trading_enabled' in data: is_trading_enabled = bool(data['trading_enabled'])
        if 'paper_trading_mode' in data: paper_trading_mode = bool(data['paper_trading_mode'])
        if 'risk_per_trade_percent' in data: RISK_PER_TRADE_PERCENT = float(data['risk_per_trade_percent'])
        if 'buy_confidence_threshold' in data: BUY_CONFIDENCE_THRESHOLD = float(data['buy_confidence_threshold'])
        
        # Advanced features
        if 'use_trailing_stop_loss' in data: USE_TRAILING_STOP_LOSS = bool(data['use_trailing_stop_loss'])
        if 'use_partial_take_profit' in data: USE_PARTIAL_TAKE_PROFIT = bool(data['use_partial_take_profit'])
        
        # Strategies
        if 'use_volume_profile_strategy' in data: USE_VOLUME_PROFILE_STRATEGY = bool(data['use_volume_profile_strategy'])
        if 'use_bb_stoch_strategy' in data: USE_BB_STOCH_STRATEGY = bool(data['use_bb_stoch_strategy'])
        if 'use_macd_ema_strategy' in data: USE_MACD_EMA_STRATEGY = bool(data['use_macd_ema_strategy'])
        if 'use_ema_rsi_strategy' in data: USE_EMA_RSI_STRATEGY = bool(data['use_ema_rsi_strategy'])
        if 'use_pullback_strategy' in data: USE_PULLBACK_STRATEGY = bool(data['use_pullback_strategy'])
        
        save_settings_to_redis()
        
        log_and_notify("info", f"✅ [API] تم تحديث إعدادات البوت بنجاح.", "SETTINGS_UPDATE")
        return jsonify({"success": True, "message": "تم تحديث الإعدادات بنجاح."})
    except Exception as e:
        logger.error(f"❌ [API] فشل تحديث الإعدادات: {e}")
        return jsonify({"success": False, "message": "فشل تحديث الإعدادات."}), 500

@app.route('/')
def home():
    """نقطة نهاية لصفحة الويب الرئيسية الجديدة."""
    return render_template_string("""
        <!doctype html>
        <html lang="ar" dir="rtl">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>لوحة تحكم بوت التداول</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap" rel="stylesheet">
            <style>
                body { font-family: 'Cairo', sans-serif; }
                .toggle-checkbox:checked { right: 0; border-color: #4A5568; }
                .toggle-checkbox:checked + .toggle-label { background-color: #4A5568; }
                .sidebar-link.active { background-color: #4A5568; }
            </style>
        </head>
        <body class="bg-gray-900 text-gray-200">
            <div class="flex h-screen">
                <!-- Sidebar -->
                <aside class="w-64 bg-gray-800 p-6 flex flex-col justify-between">
                    <div>
                        <h1 class="text-2xl font-bold text-white mb-8">بوت التداول V13</h1>
                        <nav>
                            <ul>
                                <li><a href="#" id="nav-dashboard" class="sidebar-link active flex items-center py-3 px-4 rounded-lg hover:bg-gray-700 transition-colors">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="w-5 h-5 ml-3"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg>
                                    لوحة التحكم
                                </a></li>
                                <li><a href="#" id="nav-settings" class="sidebar-link flex items-center py-3 px-4 rounded-lg hover:bg-gray-700 transition-colors mt-2">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="w-5 h-5 ml-3"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 0 2l-.15.08a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.38a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1 0-2l.15-.08a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>
                                    الإعدادات
                                </a></li>
                            </ul>
                        </nav>
                    </div>
                    <div id="status-footer" class="text-center">
                        <div class="flex items-center justify-center">
                            <span id="botStatusDot" class="h-3 w-3 rounded-full bg-gray-500 ml-2"></span>
                            <span id="botStatusText">جاري التحميل...</span>
                        </div>
                        <span id="tradingModeText" class="text-sm text-gray-400 mt-1 block"></span>
                    </div>
                </aside>

                <!-- Main Content -->
                <main class="flex-1 p-8 overflow-y-auto">
                    <!-- Dashboard Page -->
                    <div id="page-dashboard">
                        <h2 class="text-3xl font-bold mb-6">لوحة التحكم</h2>
                        
                        <!-- Status Cards -->
                        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                            <div class="bg-gray-800 p-6 rounded-xl">
                                <h3 class="text-gray-400 text-lg">حالة السوق (BTC)</h3>
                                <p id="marketTrendText" class="text-2xl font-bold mt-2 text-white">جاري التقييم...</p>
                                <p id="marketTrendTime" class="text-xs text-gray-500 mt-1"></p>
                            </div>
                            <div class="bg-gray-800 p-6 rounded-xl">
                                <h3 class="text-gray-400 text-lg">الصفقات المفتوحة</h3>
                                <p id="openTradesCount" class="text-2xl font-bold mt-2 text-white">-</p>
                            </div>
                            <div class="bg-gray-800 p-6 rounded-xl">
                                <h3 class="text-gray-400 text-lg">تفعيل التداول</h3>
                                <button id="toggleTradingBtn" class="mt-3 w-full text-white font-bold py-2 px-4 rounded-lg transition-colors">
                                    ...
                                </button>
                            </div>
                        </div>

                        <!-- Tabs -->
                        <div class="bg-gray-800 rounded-xl p-6">
                            <div class="border-b border-gray-700">
                                <nav class="-mb-px flex space-x-6" dir="ltr">
                                    <a href="#" class="tab-link active-tab whitespace-nowrap py-4 px-1 border-b-2 font-medium text-lg" data-tab="open-trades">الصفقات المفتوحة</a>
                                    <a href="#" class="tab-link text-gray-400 hover:text-white whitespace-nowrap py-4 px-1 border-b-2 border-transparent font-medium text-lg" data-tab="rejection-logs">سجلات الرفض</a>
                                    <a href="#" class="tab-link text-gray-400 hover:text-white whitespace-nowrap py-4 px-1 border-b-2 border-transparent font-medium text-lg" data-tab="notifications">الإشعارات</a>
                                </nav>
                            </div>
                            <div class="mt-6">
                                <!-- Open Trades Tab Content -->
                                <div id="tab-content-open-trades" class="tab-content">
                                    <div class="overflow-x-auto">
                                        <table class="min-w-full">
                                            <thead class="text-gray-400">
                                                <tr>
                                                    <th class="py-3 px-4 text-right">الرمز</th>
                                                    <th class="py-3 px-4 text-right">سعر الدخول</th>
                                                    <th class="py-3 px-4 text-right">وقف الخسارة</th>
                                                    <th class="py-3 px-4 text-right">الهدف (1)</th>
                                                    <th class="py-3 px-4 text-right">الهدف (2)</th>
                                                    <th class="py-3 px-4 text-right">الكمية</th>
                                                    <th class="py-3 px-4 text-right">الاستراتيجية</th>
                                                </tr>
                                            </thead>
                                            <tbody id="openTradesTableBody">
                                                <!-- Rows will be injected by JS -->
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                                <!-- Rejection Logs Tab Content -->
                                <div id="tab-content-rejection-logs" class="tab-content hidden">
                                    <ul id="rejectionLogsList" class="space-y-3 max-h-96 overflow-y-auto">
                                        <!-- Items will be injected by JS -->
                                    </ul>
                                </div>
                                <!-- Notifications Tab Content -->
                                <div id="tab-content-notifications" class="tab-content hidden">
                                    <ul id="notificationsList" class="space-y-3 max-h-96 overflow-y-auto">
                                        <!-- Items will be injected by JS -->
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Settings Page -->
                    <div id="page-settings" class="hidden">
                        <h2 class="text-3xl font-bold mb-6">الإعدادات</h2>
                        
                        <div class="space-y-8">
                            <!-- General Settings -->
                            <div class="bg-gray-800 p-6 rounded-xl">
                                <h3 class="text-xl font-bold mb-4 border-b border-gray-700 pb-3">الإعدادات العامة</h3>
                                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div>
                                        <label for="riskPerTrade" class="block mb-2 text-sm font-medium text-gray-300">نسبة المخاطرة لكل صفقة (%)</label>
                                        <input type="number" id="riskPerTrade" step="0.05" class="bg-gray-700 border border-gray-600 text-white text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5">
                                    </div>
                                    <div>
                                        <label for="buyConfidence" class="block mb-2 text-sm font-medium text-gray-300">عتبة ثقة الشراء</label>
                                        <input type="number" id="buyConfidence" step="0.01" class="bg-gray-700 border border-gray-600 text-white text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5">
                                    </div>
                                    <div class="flex items-center justify-between bg-gray-700 p-3 rounded-lg">
                                        <span class="font-medium text-white">وضع التداول التجريبي</span>
                                        <label class="relative inline-flex items-center cursor-pointer">
                                            <input type="checkbox" id="paperTradingModeToggle" class="sr-only peer">
                                            <div class="w-11 h-6 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-green-600"></div>
                                        </label>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Trade Management -->
                            <div class="bg-gray-800 p-6 rounded-xl">
                                <h3 class="text-xl font-bold mb-4 border-b border-gray-700 pb-3">إدارة الصفقات المتقدمة</h3>
                                <div class="space-y-4">
                                    <div class="flex items-center justify-between">
                                        <span class="font-medium text-white">تفعيل وقف الخسارة المتحرك</span>
                                        <label class="relative inline-flex items-center cursor-pointer">
                                            <input type="checkbox" id="trailingStopLossToggle" class="sr-only peer">
                                            <div class="w-11 h-6 bg-gray-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-green-600"></div>
                                        </label>
                                    </div>
                                    <div class="flex items-center justify-between">
                                        <span class="font-medium text-white">تفعيل أخذ الربح الجزئي</span>
                                        <label class="relative inline-flex items-center cursor-pointer">
                                            <input type="checkbox" id="partialTakeProfitToggle" class="sr-only peer">
                                            <div class="w-11 h-6 bg-gray-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-green-600"></div>
                                        </label>
                                    </div>
                                </div>
                            </div>

                            <!-- Strategies Activation -->
                            <div class="bg-gray-800 p-6 rounded-xl">
                                <h3 class="text-xl font-bold mb-4 border-b border-gray-700 pb-3">تفعيل استراتيجيات التداول</h3>
                                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div class="flex items-center justify-between">
                                        <span class="font-medium text-white">استراتيجية Volume Profile</span>
                                        <label class="relative inline-flex items-center cursor-pointer">
                                            <input type="checkbox" id="volumeProfileStrategy" class="sr-only peer">
                                            <div class="w-11 h-6 bg-gray-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-green-600"></div>
                                        </label>
                                    </div>
                                    <div class="flex items-center justify-between">
                                        <span class="font-medium text-white">استراتيجية BB & Stoch</span>
                                        <label class="relative inline-flex items-center cursor-pointer">
                                            <input type="checkbox" id="bbStochStrategy" class="sr-only peer">
                                            <div class="w-11 h-6 bg-gray-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-green-600"></div>
                                        </label>
                                    </div>
                                    <div class="flex items-center justify-between">
                                        <span class="font-medium text-white">استراتيجية MACD & EMA</span>
                                        <label class="relative inline-flex items-center cursor-pointer">
                                            <input type="checkbox" id="macdEmaStrategy" class="sr-only peer">
                                            <div class="w-11 h-6 bg-gray-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-green-600"></div>
                                        </label>
                                    </div>
                                    <div class="flex items-center justify-between">
                                        <span class="font-medium text-white">استراتيجية EMA & RSI</span>
                                        <label class="relative inline-flex items-center cursor-pointer">
                                            <input type="checkbox" id="emaRsiStrategy" class="sr-only peer">
                                            <div class="w-11 h-6 bg-gray-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-green-600"></div>
                                        </label>
                                    </div>
                                    <div class="flex items-center justify-between">
                                        <span class="font-medium text-white">استراتيجية Pullback</span>
                                        <label class="relative inline-flex items-center cursor-pointer">
                                            <input type="checkbox" id="pullbackStrategy" class="sr-only peer">
                                            <div class="w-11 h-6 bg-gray-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-green-600"></div>
                                        </label>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="flex justify-end">
                                <button id="saveSettingsBtn" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg transition-colors">
                                    حفظ الإعدادات
                                </button>
                            </div>
                        </div>
                    </div>
                </main>
            </div>

            <script>
                document.addEventListener('DOMContentLoaded', () => {
                    const pageDashboard = document.getElementById('page-dashboard');
                    const pageSettings = document.getElementById('page-settings');
                    const navDashboard = document.getElementById('nav-dashboard');
                    const navSettings = document.getElementById('nav-settings');
                    
                    const showPage = (pageToShow) => {
                        [pageDashboard, pageSettings].forEach(p => p.classList.add('hidden'));
                        pageToShow.classList.remove('hidden');
                        
                        [navDashboard, navSettings].forEach(n => n.classList.remove('active'));
                        if (pageToShow === pageDashboard) navDashboard.classList.add('active');
                        else navSettings.classList.add('active');
                    };

                    navDashboard.addEventListener('click', (e) => { e.preventDefault(); showPage(pageDashboard); });
                    navSettings.addEventListener('click', (e) => { e.preventDefault(); showPage(pageSettings); });

                    const tabLinks = document.querySelectorAll('.tab-link');
                    const tabContents = document.querySelectorAll('.tab-content');

                    tabLinks.forEach(link => {
                        link.addEventListener('click', (e) => {
                            e.preventDefault();
                            const tabId = link.dataset.tab;

                            tabLinks.forEach(l => l.classList.remove('active-tab', 'text-white'));
                            tabLinks.forEach(l => l.classList.add('text-gray-400', 'hover:text-white', 'border-transparent'));
                            link.classList.add('active-tab', 'text-white');
                            link.classList.remove('text-gray-400', 'hover:text-white', 'border-transparent');

                            tabContents.forEach(content => content.classList.add('hidden'));
                            document.getElementById(`tab-content-${tabId}`).classList.remove('hidden');
                        });
                    });

                    // --- Data Fetching and UI Update Functions ---
                    const fetchStatus = async () => {
                        try {
                            const response = await fetch('/status');
                            if (!response.ok) throw new Error('Network response was not ok');
                            const data = await response.json();
                            
                            // Status Footer & Header
                            document.getElementById('botStatusDot').className = data.trading_enabled ? 'h-3 w-3 rounded-full bg-green-500 ml-2' : 'h-3 w-3 rounded-full bg-red-500 ml-2';
                            document.getElementById('botStatusText').innerText = data.trading_enabled ? 'يعمل' : 'متوقف';
                            document.getElementById('tradingModeText').innerText = data.paper_trading_mode ? 'وضع تجريبي' : 'وضع حقيقي';
                            
                            // Dashboard Cards
                            document.getElementById('marketTrendText').innerText = data.market_state.overall_regime || 'غير محدد';
                            document.getElementById('marketTrendTime').innerText = `آخر تحديث: ${data.market_state.last_updated || 'N/A'}`;
                            document.getElementById('openTradesCount').innerText = `${data.open_trades_count} / ${data.max_open_trades}`;
                            
                            const toggleBtn = document.getElementById('toggleTradingBtn');
                            toggleBtn.innerText = data.trading_enabled ? 'إيقاف التداول' : 'تفعيل التداول';
                            toggleBtn.className = data.trading_enabled ? 'w-full text-white font-bold py-2 px-4 rounded-lg transition-colors bg-red-600 hover:bg-red-700' : 'w-full text-white font-bold py-2 px-4 rounded-lg transition-colors bg-green-600 hover:bg-green-700';

                            // Settings Page Inputs
                            document.getElementById('riskPerTrade').value = data.risk_per_trade_percent;
                            document.getElementById('buyConfidence').value = data.buy_confidence_threshold;
                            document.getElementById('paperTradingModeToggle').checked = data.paper_trading_mode;
                            document.getElementById('trailingStopLossToggle').checked = data.advanced_features_enabled.trailing_stop_loss;
                            document.getElementById('partialTakeProfitToggle').checked = data.advanced_features_enabled.partial_take_profit;
                            document.getElementById('volumeProfileStrategy').checked = data.strategies_enabled.volume_profile;
                            document.getElementById('bbStochStrategy').checked = data.strategies_enabled.bb_stoch;
                            document.getElementById('macdEmaStrategy').checked = data.strategies_enabled.macd_ema;
                            document.getElementById('emaRsiStrategy').checked = data.strategies_enabled.ema_rsi;
                            document.getElementById('pullbackStrategy').checked = data.strategies_enabled.pullback;

                        } catch (error) {
                            console.error('Error fetching status:', error);
                            document.getElementById('botStatusText').innerText = 'خطأ بالاتصال';
                            document.getElementById('botStatusDot').className = 'h-3 w-3 rounded-full bg-yellow-500 ml-2';
                        }
                    };

                    const fetchOpenTrades = async () => {
                        try {
                            const response = await fetch('/open_trades');
                            if (!response.ok) throw new Error('Network response was not ok');
                            const data = await response.json();
                            const tableBody = document.getElementById('openTradesTableBody');
                            tableBody.innerHTML = '';
                            if (!data.open_trades || data.open_trades.length === 0) {
                                tableBody.innerHTML = '<tr><td colspan="7" class="text-center py-4 text-gray-500">لا توجد صفقات مفتوحة حالياً.</td></tr>';
                                return;
                            }
                            data.open_trades.forEach(trade => {
                                const row = document.createElement('tr');
                                row.className = 'border-b border-gray-700 hover:bg-gray-800';
                                row.innerHTML = `
                                    <td class="py-3 px-4 font-medium">${trade.symbol}</td>
                                    <td class="py-3 px-4">${trade.entry_price.toFixed(4)}</td>
                                    <td class="py-3 px-4 text-red-400">${trade.stop_loss.toFixed(4)}</td>
                                    <td class="py-3 px-4 text-green-400">${trade.target_price.toFixed(4)}</td>
                                    <td class="py-3 px-4 text-green-400">${(trade.target_price_2 || 0).toFixed(4)}</td>
                                    <td class="py-3 px-4">${(trade.quantity || 0).toFixed(4)}</td>
                                    <td class="py-3 px-4 text-gray-400">${trade.strategy_name || 'N/A'}</td>
                                `;
                                tableBody.appendChild(row);
                            });
                        } catch (error) {
                            console.error('Error fetching open trades:', error);
                            document.getElementById('openTradesTableBody').innerHTML = '<tr><td colspan="7" class="text-center py-4 text-red-500">فشل تحميل الصفقات.</td></tr>';
                        }
                    };

                    const fetchRejectionLogs = async () => {
                        try {
                            const response = await fetch('/rejection_logs');
                            if (!response.ok) throw new Error('Network response was not ok');
                            const data = await response.json();
                            const list = document.getElementById('rejectionLogsList');
                            list.innerHTML = '';
                            if (!data.rejection_logs || data.rejection_logs.length === 0) {
                                list.innerHTML = '<li class="text-center text-gray-500 py-4">لا توجد سجلات رفض حديثة.</li>';
                                return;
                            }
                            data.rejection_logs.forEach(log => {
                                const item = document.createElement('li');
                                item.className = 'bg-gray-700 p-3 rounded-lg flex justify-between items-center';
                                item.innerHTML = `
                                    <div>
                                        <span class="font-bold text-red-400">${log.symbol}</span>
                                        <span class="text-gray-300 mr-2">${log.reason}</span>
                                    </div>
                                    <span class="text-xs text-gray-500">${new Date(log.timestamp).toLocaleTimeString('ar-EG')}</span>
                                `;
                                list.appendChild(item);
                            });
                        } catch (error) {
                            console.error('Error fetching rejection logs:', error);
                            document.getElementById('rejectionLogsList').innerHTML = '<li class="text-center text-red-500 py-4">فشل تحميل سجلات الرفض.</li>';
                        }
                    };

                    const fetchNotifications = async () => {
                        try {
                            const response = await fetch('/notifications');
                            if (!response.ok) throw new Error('Network response was not ok');
                            const data = await response.json();
                            const list = document.getElementById('notificationsList');
                            list.innerHTML = '';
                            if (!data.notifications || data.notifications.length === 0) {
                                list.innerHTML = '<li class="text-center text-gray-500 py-4">لا توجد إشعارات جديدة.</li>';
                                return;
                            }
                            data.notifications.forEach(notif => {
                                let icon = '';
                                let colorClass = '';
                                if (notif.type.includes('SIGNAL')) {
                                    icon = '📈'; colorClass = 'text-green-400';
                                } else if (notif.type.includes('CLOSE') || notif.type.includes('STOP')) {
                                    icon = '🛑'; colorClass = 'text-red-400';
                                } else if (notif.type.includes('TP')) {
                                    icon = '🎯'; colorClass = 'text-blue-400';
                                } else {
                                    icon = 'ℹ️'; colorClass = 'text-yellow-400';
                                }
                                const item = document.createElement('li');
                                item.className = 'bg-gray-700 p-3 rounded-lg flex justify-between items-center';
                                item.innerHTML = `
                                    <div class="flex items-center">
                                        <span class="ml-3">${icon}</span>
                                        <div>
                                            <span class="font-bold ${colorClass}">${notif.type}</span>
                                            <p class="text-sm text-gray-300">${notif.message}</p>
                                        </div>
                                    </div>
                                    <span class="text-xs text-gray-500">${new Date(notif.timestamp).toLocaleTimeString('ar-EG')}</span>
                                `;
                                list.appendChild(item);
                            });
                        } catch (error) {
                            console.error('Error fetching notifications:', error);
                            document.getElementById('notificationsList').innerHTML = '<li class="text-center text-red-500 py-4">فشل تحميل الإشعارات.</li>';
                        }
                    };

                    const updateSettings = async (settings) => {
                        try {
                            const response = await fetch('/settings', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify(settings)
                            });
                            const data = await response.json();
                            if (data.success) {
                                console.log(data.message);
                                await fetchStatus(); 
                            } else {
                                console.error(data.message);
                            }
                        } catch (error) {
                            console.error('Error updating settings:', error);
                        }
                    };

                    document.getElementById('toggleTradingBtn').addEventListener('click', async () => {
                        const isEnabled = document.getElementById('botStatusText').innerText === 'يعمل';
                        await updateSettings({ trading_enabled: !isEnabled });
                    });

                    document.getElementById('saveSettingsBtn').addEventListener('click', async () => {
                        const settings = {
                            risk_per_trade_percent: parseFloat(document.getElementById('riskPerTrade').value),
                            buy_confidence_threshold: parseFloat(document.getElementById('buyConfidence').value),
                            paper_trading_mode: document.getElementById('paperTradingModeToggle').checked,
                            use_trailing_stop_loss: document.getElementById('trailingStopLossToggle').checked,
                            use_partial_take_profit: document.getElementById('partialTakeProfitToggle').checked,
                            use_volume_profile_strategy: document.getElementById('volumeProfileStrategy').checked,
                            use_bb_stoch_strategy: document.getElementById('bbStochStrategy').checked,
                            use_macd_ema_strategy: document.getElementById('macdEmaStrategy').checked,
                            use_ema_rsi_strategy: document.getElementById('emaRsiStrategy').checked,
                            use_pullback_strategy: document.getElementById('pullbackStrategy').checked,
                        };
                        
                        const btn = document.getElementById('saveSettingsBtn');
                        btn.innerText = 'جاري الحفظ...';
                        btn.disabled = true;
                        
                        await updateSettings(settings);

                        btn.innerText = 'حفظ الإعدادات';
                        btn.disabled = false;
                        // Simple feedback
                        btn.classList.add('bg-green-600');
                        setTimeout(() => { btn.classList.remove('bg-green-600'); }, 2000);
                    });
                    
                    const refreshDashboard = () => {
                        fetchStatus();
                        fetchOpenTrades();
                        fetchRejectionLogs();
                        fetchNotifications();
                    };

                    setInterval(refreshDashboard, 5000); // Refresh every 5 seconds
                    refreshDashboard(); // Initial load
                });
            </script>
        </body>
        </html>
    """)

def start_flask_app():
    """يبدأ خادم Flask."""
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# --- وظائف البوت الرئيسية ---
def process_symbol(symbol: str):
    """الدالة الرئيسية التي تحلل البيانات وتطبق استراتيجيات التداول."""
    logger.info(f"✨ [Scanner] جاري فحص الرمز: {symbol}")
    df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
    if df is None or len(df) < 50:
        log_rejection(symbol, "Insufficient Historical Data")
        return
    df_with_features = calculate_all_features(df)
    last = df_with_features.iloc[-1]
    
    if last['adx'] < 18 or last['atr'] < (last['close'] * 0.005):
        log_rejection(symbol, "Market Volatility Filter Failed")
        return
        
    htf_df = fetch_historical_data(symbol, HIGHER_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
    if htf_df is None or len(htf_df) < 50:
        log_rejection(symbol, "Insufficient Historical Data")
        return
    htf_df_with_features = calculate_all_features(htf_df)
    htf_trend = calculate_market_trend(htf_df_with_features)
    if htf_trend != 'اتجاه صاعد':
        log_rejection(symbol, "HTF Trend Confirmation Failed")
        return
        
    signal_found = False
    strategy_name = "N/A"
    signal_details = {}
    
    if USE_VOLUME_PROFILE_STRATEGY and check_volume_profile_strategy(df_with_features):
        signal_found, strategy_name = True, "Volume Profile"
        signal_details = {"poc": calculate_volume_profile(df_with_features)[0]}
    elif USE_BB_STOCH_STRATEGY and last['close'] < last['lower_band'] and last['stoch_rsi'] < 20:
        signal_found, strategy_name = True, "BB & Stoch"
    elif USE_MACD_EMA_STRATEGY and last['macd_hist'] > 0 and last['macd'] > last['macd_signal'] and last['ema_12'] > last['ema_26']:
        signal_found, strategy_name = True, "MACD & EMA"
    elif USE_EMA_RSI_STRATEGY and last['ema_12'] > last['ema_26'] and last['rsi'] > 50 and last['close'] > last['ema_26']:
        signal_found, strategy_name = True, "EMA & RSI"
    elif USE_PULLBACK_STRATEGY and last['close'] > last['ema_26'] and df_with_features.iloc[-2]['close'] < df_with_features.iloc[-2]['ema_26']:
        signal_found, strategy_name = True, "Pullback"
        
    if signal_found:
        entry_price = float(last['close'])
        stop_loss = entry_price - (entry_price * RISK_PER_TRADE_PERCENT / 100)
        target_price = entry_price + (entry_price * MIN_PROFIT_PERCENT / 100)
        target_price_2 = entry_price + (entry_price * (MIN_PROFIT_PERCENT + 0.5) / 100)
        quantity = PAPER_TRADE_SIZE_USDT / entry_price if paper_trading_mode else 0
        
        signal_data = {
            "symbol": symbol, "entry_price": entry_price, "target_price": target_price,
            "stop_loss": stop_loss, "strategy": strategy_name, "quantity": quantity,
            "is_real_trade": not paper_trading_mode, "details": signal_details,
            "target_price_2": target_price_2, "initial_quantity": quantity
        }
        
        if create_signal_db(signal_data):
            # جلب الإشارة من قاعدة البيانات للتأكد من وجود ID
            newly_created_signal = get_open_signals()[-1] # افتراض أن آخر إشارة هي التي تم إنشاؤها
            with signal_cache_lock:
                open_signals_cache[symbol] = newly_created_signal
            message = (f"📈 *إشارة شراء جديدة!*\\n"
                       f"💱 *العملة:* `{symbol}`\\n"
                       f"🛒 *سعر الدخول:* `{entry_price:.4f}`\\n"
                       f"🛑 *وقف الخسارة:* `{stop_loss:.4f}`\\n"
                       f"🎯 *الهدف 1:* `{target_price:.4f}`\\n"
                       f"🎯 *الهدف 2:* `{target_price_2:.4f}`\\n"
                       f"🧠 *الاستراتيجية:* `{strategy_name}`")
            send_telegram_message(message)

def run_signal_scanner():
    """الخيط الرئيسي الذي يفحص الرموز بحثًا عن إشارات جديدة."""
    while True:
        try:
            with trading_status_lock:
                if not is_trading_enabled:
                    time.sleep(60)
                    continue
            logger.info("🕵️ [Scanner] بدء دورة المسح...")
            with signal_cache_lock:
                open_trades_count = len(open_signals_cache)
            if open_trades_count >= MAX_OPEN_TRADES:
                logger.warning(f"⚠️ [Scanner] تجاوز الحد الأقصى للصفقات المفتوحة ({open_trades_count}/{MAX_OPEN_TRADES}). جاري التخطي.")
                time.sleep(60)
                continue
            symbols_to_process = [s for s in validated_symbols_to_scan if s not in open_signals_cache]
            if not symbols_to_process:
                logger.info("💤 [Scanner] لا توجد رموز جديدة للفحص. جاري الانتظار.")
                time.sleep(60)
                continue
            for i in range(0, len(symbols_to_process), SYMBOL_PROCESSING_BATCH_SIZE):
                batch = symbols_to_process[i:i + SYMBOL_PROCESSING_BATCH_SIZE]
                for symbol in batch:
                    try:
                        process_symbol(symbol)
                    except Exception as e:
                        logger.error(f"❌ [Scanner] خطأ في معالجة الرمز {symbol}: {e}")
                time.sleep(10) # تأخير بين الدفعات
            logger.info("✅ [Scanner] انتهت دورة المسح بنجاح.")
            time.sleep(60 * 2) # انتظار دقيقتين قبل الدورة التالية
        except Exception as e:
            logger.error(f"❌ [Scanner] حدث خطأ حرج: {e}", exc_info=True)
            time.sleep(60)
        finally:
            gc.collect()

def run_trade_manager():
    """الخيط الرئيسي الذي يدير الصفقات المفتوحة."""
    while True:
        try:
            time.sleep(15) # تأخير ثابت بين الدورات
            with trading_status_lock:
                if not is_trading_enabled: continue
            
            with signal_cache_lock:
                symbols_to_manage = list(open_signals_cache.keys())
            if not symbols_to_manage: continue
            
            logger.info("🔄 [Manager] بدء دورة إدارة الصفقات...")
            for symbol in symbols_to_manage:
                with signal_cache_lock:
                    if symbol not in open_signals_cache: continue
                    signal = open_signals_cache[symbol]
                with live_prices_lock:
                    current_price = live_prices.get(symbol)
                if current_price is None:
                    logger.warning(f"⚠️ [Manager] السعر الحي لـ {symbol} غير متوفر. جاري التخطي.")
                    continue
                
                df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, 50)
                if df is None or len(df) < 50: continue
                df_with_features = calculate_all_features(df)
                
                # تحديث وقف الخسارة المتحرك
                highest_price = max(signal.get('highest_price', signal['entry_price']), current_price)
                new_stop_loss = calculate_trailing_stop_loss(df_with_features, signal['entry_price'], signal['stop_loss'], highest_price)
                if new_stop_loss > signal['stop_loss']:
                    logger.info(f"✅ [Manager] تم تحديث وقف الخسارة المتحرك لـ {symbol} من {signal['stop_loss']:.4f} إلى {new_stop_loss:.4f}.")
                    update_signal_db(signal['id'], {'stop_loss': new_stop_loss})
                    with signal_cache_lock:
                        if symbol in open_signals_cache:
                            open_signals_cache[symbol]['stop_loss'] = new_stop_loss
                            open_signals_cache[symbol]['highest_price'] = highest_price

                # فحص شروط الخروج
                if check_for_close_conditions(df_with_features, signal, current_price):
                    reason = "Exit condition met" # يمكن تحسين هذا لاحقًا
                    close_signal_db(signal, current_price, reason)
                    with signal_cache_lock:
                        open_signals_cache.pop(symbol, None)
                        
            logger.info("✅ [Manager] انتهت دورة إدارة الصفقات بنجاح.")
        except Exception as e:
            logger.error(f"❌ [Manager] حدث خطأ حرج: {e}", exc_info=True)
        finally:
            gc.collect()

def update_market_state():
    """تحدث حالة السوق العامة واتجاه الترند على الفريمات المختلفة."""
    global current_market_state
    while True:
        try:
            btc_data = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
            if btc_data is None or len(btc_data) < 20:
                logger.warning("⚠️ [Market State] بيانات BTC غير كافية لتقييم حالة السوق.")
                time.sleep(60)
                continue
            
            btc_data_with_features = calculate_all_features(btc_data)
            overall_regime = calculate_market_trend(btc_data_with_features)
            
            trend_details = {}
            for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
                tf_data = fetch_historical_data(BTC_SYMBOL, tf, SIGNAL_GENERATION_LOOKBACK_DAYS)
                if tf_data is not None:
                    tf_data_with_features = calculate_all_features(tf_data)
                    trend_details[tf] = calculate_market_trend(tf_data_with_features)
            
            logger.info(f"✅ [Market State] تم تحديث حالة السوق: الاتجاه العام: {overall_regime}")
            with market_state_lock:
                current_market_state.update({'overall_regime': overall_regime, 'trend_details_by_tf': trend_details, 'last_updated': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')})
            time.sleep(60 * 5)
        except Exception as e:
            logger.error(f"❌ [Market State] A critical error occurred: {e}", exc_info=True)
            time.sleep(60)

# --- نقطة بداية البرنامج ---
if __name__ == '__main__':
    logger.info("="*50 + "\\n====== Starting Crypto Trading Bot V13.0.0 ======\\n" + "="*50)
    init_db()
    init_redis()
    try:
        client = Client(API_KEY, API_SECRET); client.ping()
        logger.info("✅ [Binance] API connection successful.")
    except Exception as e:
        logger.critical(f"❌ [Binance] API connection failed: {e}"); exit(1)
    get_exchange_info_map()
    validated_symbols_to_scan = get_validated_symbols()
    if not validated_symbols_to_scan:
        logger.critical("❌ No valid symbols to scan. The bot will exit."); exit(1)
    load_open_signals_to_cache()
    load_notifications_to_cache()
    load_settings_from_redis()
    start_websocket()
    
    # بدء الخيوط
    Thread(target=run_signal_scanner, daemon=True).start()
    Thread(target=run_trade_manager, daemon=True).start()
    Thread(target=update_market_state, daemon=True).start()
    Thread(target=start_flask_app, daemon=True).start()
    
    logger.info("✅ Bot is fully initialized and running. Dashboard is available at http://127.0.0.1:5000")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("👋 [Bot] Shutting down bot gracefully...")
    finally:
        if ws_manager: ws_manager.stop()
        if conn: conn.close()
        logger.info("✅ [Bot] Bot has been shut down.")
