# ملف c4.py - نسخة V14.0.0 (تحسين الفلاتر وإضافة لوحة تحكم متقدمة)
# --- التغييرات الرئيسية (V14.0.0):
# 1. [ميزة جديدة] تم إضافة نظام "ملفات تعريف الفلترة" (Filter Profiles) للتحكم في صرامة البوت.
#    - الملفات المتاحة: صارم (Strict), متوسط (Moderate), انعكاسي (Reversal), معطل (Disabled).
# 2. [ميزة جديدة] تم إضافة قسم جديد في لوحة التحكم (صفحة الإعدادات) لإدارة فلاتر كل استراتيجية على حدة.
#    - يمكن للمستخدم الآن اختيار ملف تعريف الفلترة وتحديد قيمة ADX لكل استراتيجية.
# 3. [تحسين] تم إعادة هيكلة منطق الفحص في الحلقة الرئيسية (`main_bot_loop`) لاستخدام الإعدادات الديناميكية للفلاتر.
# 4. [تحسين] تم تعديل دالة تأكيد الاتجاه على الفريم الأعلى (`is_htf_bullish_confirmation`) لتقبل أوضاع مختلفة (Strict, Relaxed).
# 5. [هيكلة] تم إضافة قاموس جديد `STRATEGY_FILTER_CONFIG` لإدارة إعدادات الفلاتر بشكل مركزي.
# 6. [تحسين] تم تحديث واجهة المستخدم وقوالب HTML لتعكس الإضافات الجديدة.

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
        logging.FileHandler('crypto_bot_v14_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV14.0.0')

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

# --- مفاتيح تفعيل الاستراتيجيات ---
USE_BB_STOCH_STRATEGY: bool = True
bb_stoch_strategy_lock = Lock()
USE_MACD_EMA_STRATEGY: bool = True
macd_ema_strategy_lock = Lock()
USE_EMA_RSI_STRATEGY: bool = True
ema_rsi_strategy_lock = Lock()
USE_PULLBACK_STRATEGY: bool = True
pullback_strategy_lock = Lock()
USE_MOMENTUM_VOLATILITY_STRATEGY: bool = True
momentum_volatility_strategy_lock = Lock()

# --- [جديد] إعدادات الفلاتر الديناميكية للاستراتيجيات ---
STRATEGY_NAMES = {
    "BB_Stoch_Strategy": "BB+Stoch (انعكاسية)",
    "MACD_EMA_Strategy": "MACD+EMA (اتجاهية)",
    "EMA_RSI_Strategy": "EMA+RSI (مختلطة)",
    "Pullback_Strategy": "Pullback (انعكاسية)",
    "Momentum_Volatility_Strategy": "Momentum (زخم)"
}

STRATEGY_FILTER_CONFIG = {
    "BB_Stoch_Strategy": {"profile": "Reversal", "adx_threshold": 18, "htf_confirmation_mode": "Disabled"},
    "MACD_EMA_Strategy": {"profile": "Strict", "adx_threshold": 22, "htf_confirmation_mode": "Strict"},
    "EMA_RSI_Strategy": {"profile": "Moderate", "adx_threshold": 20, "htf_confirmation_mode": "Relaxed"},
    "Pullback_Strategy": {"profile": "Reversal", "adx_threshold": 18, "htf_confirmation_mode": "Relaxed"},
    "Momentum_Volatility_Strategy": {"profile": "Strict", "adx_threshold": 25, "htf_confirmation_mode": "Strict"},
}
strategy_filters_lock = Lock()


# --- إعدادات عامة ---
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '1h'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 15
BTC_SYMBOL: str = 'BTCUSDT'
SYMBOL_PROCESSING_BATCH_SIZE: int = 5
TRADING_FEE_PERCENT: float = 0.1
API_REQUEST_DELAY: float = 1
API_RETRY_COUNT: int = 3
API_RETRY_DELAY: float = 5.0

# --- إعدادات المؤشرات الفنية ---
EMA_FAST_PERIOD: int = 12
EMA_SLOW_PERIOD: int = 26
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
MOMENTUM_PERIOD: int = 10

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
}

# --- إعداد تطبيق Flask ---
app = Flask(__name__)
CORS(app)

# --- دوال تهيئة الخدمات (بدون تغيير) ---
def init_db(retries: int = 5, delay: int = 5) -> None:
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
                        target_price DOUBLE PRECISION, stop_loss DOUBLE PRECISION NOT NULL,
                        status TEXT DEFAULT 'open', closing_price DOUBLE PRECISION, closed_at TIMESTAMP,
                        profit_percentage DOUBLE PRECISION, strategy_name TEXT, signal_details JSONB,
                        is_real_trade BOOLEAN DEFAULT FALSE, quantity DOUBLE PRECISION, closing_reason TEXT,
                        target_price_1 DOUBLE PRECISION, target_price_2 DOUBLE PRECISION, initial_quantity DOUBLE PRECISION
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
    global redis_client
    logger.info("[Redis] Initializing connection...")
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] Connected successfully.")
    except redis.exceptions.ConnectionError as e:
        logger.warning(f"⚠️ [Redis] Connection failed: {e}. The bot will run without Redis.")
        redis_client = None

# --- دوال المساعدة والإشعارات (بدون تغيير) ---
def log_and_notify(level: str, message: str, notification_type: str):
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
    reason_ar = REJECTION_REASONS_AR.get(reason_key, reason_key)
    with rejection_logs_lock:
        rejection_logs_cache.appendleft({
            "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
            "reason": reason_ar, "details": json.loads(json.dumps(details or {}, cls=NpEncoder))
        })

def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [Telegram] Failed to send message: {e}")

# --- WebSocket Handler (بدون تغيير) ---
def handle_socket_message(msg):
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
    global ws_manager
    logger.info("🚀 [WebSocket] Starting WebSocket manager...")
    ws_manager = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    ws_manager.start()
    ws_manager.start_ticker_socket(callback=handle_socket_message)
    logger.info("✅ [WebSocket] Successfully subscribed to ticker stream (!ticker@arr).")

# --- دوال جلب البيانات وحساب المؤشرات (بدون تغيير) ---
def get_exchange_info_map() -> None:
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

def compute_rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    return (typical_price * df['volume']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum().replace(0, 1e-9)

def compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    return 100 - (100 / (1 + positive_mf / negative_mf.replace(0, 1e-9)))

def compute_force_index(df: pd.DataFrame, period: int = 13) -> pd.Series:
    return ((df['close'] - df['close'].shift(1)) * df['volume']).ewm(span=period, adjust=False).mean()

def compute_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    highest_high = df['high'].rolling(window=period).max()
    lowest_low = df['low'].rolling(window=period).min()
    return -100 * (highest_high - df['close']) / (highest_high - lowest_low).replace(0, 1e-9)

def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    df_calc['ema9'] = df_calc['close'].ewm(span=9, adjust=False).mean()
    df_calc['ema12'] = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    df_calc['ema21'] = df_calc['close'].ewm(span=21, adjust=False).mean()
    df_calc['ema26'] = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    df_calc['ema50'] = df_calc['close'].ewm(span=50, adjust=False).mean()
    df_calc['ema200'] = df_calc['close'].ewm(span=200, adjust=False).mean()
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
    df_calc['rsi'] = compute_rsi(df_calc['close'], RSI_PERIOD)
    df_calc['rsi9'] = compute_rsi(df_calc['close'], 9)
    df_calc['rsi14'] = compute_rsi(df_calc['close'], 14)
    df_calc['rsi21'] = compute_rsi(df_calc['close'], 21)
    df_calc['rsi_avg'] = (df_calc['rsi9'] + df_calc['rsi14'] + df_calc['rsi21']) / 3
    df_calc['rsi_ma'] = df_calc['rsi'].rolling(window=5).mean()
    rsi_val = df_calc['rsi']
    stoch_rsi = (rsi_val - rsi_val.rolling(14).min()) / (rsi_val.rolling(14).max() - rsi_val.rolling(14).min()).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi.rolling(3).mean() * 100
    bb_period = 20
    df_calc['bb_middle'] = df_calc['close'].rolling(window=bb_period).mean()
    bb_std = df_calc['close'].rolling(window=bb_period).std()
    df_calc['bb_upper'] = df_calc['bb_middle'] + (bb_std * 2)
    df_calc['bb_lower'] = df_calc['bb_middle'] - (bb_std * 2)
    exp1 = df_calc['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_calc['close'].ewm(span=26, adjust=False).mean()
    df_calc['macd'] = exp1 - exp2
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()
    df_calc[f'roc_{MOMENTUM_PERIOD}'] = (df_calc['close'] / df_calc['close'].shift(MOMENTUM_PERIOD) - 1) * 100
    df_calc['atr_sma'] = df_calc['atr'].rolling(window=14).mean()
    df_calc['force_index'] = compute_force_index(df_calc, 13)
    df_calc['volume_sma'] = df_calc['volume'].rolling(window=10).mean()
    df_calc['money_flow_index'] = compute_mfi(df_calc, 14)
    df_calc['williams_r'] = compute_williams_r(df_calc, 14)
    df_calc['atr_percent'] = (df_calc['atr'] / df_calc['close']) * 100
    df_calc['atr_percent_sma'] = df_calc['atr_percent'].rolling(window=14).mean()
    df_calc['keltner_upper'] = df_calc['ema21'] + (df_calc['atr'] * 2)
    df_calc['keltner_lower'] = df_calc['ema21'] - (df_calc['atr'] * 2)
    df_calc['vwap'] = compute_vwap(df_calc, 20)
    df_calc['dc_upper'] = df_calc['high'].rolling(window=20).max()
    df_calc['dc_lower'] = df_calc['low'].rolling(window=20).min()
    return df_calc

# --- دوال تحميل البيانات الأولية ---
def load_open_signals_to_cache():
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status IN ('open', 'updated');")
            open_signals = cur.fetchall()
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in open_signals: open_signals_cache[signal['symbol']] = dict(signal)
            logger.info(f"✅ [Cache] Loaded {len(open_signals)} open signals into cache.")
    except Exception as e:
        logger.error(f"❌ [Cache] Failed to load open signals: {e}")

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
    except Exception as e:
        logger.error(f"❌ [Cache] Failed to load notifications: {e}")

def load_settings_from_redis():
    global RISK_PER_TRADE_PERCENT, BUY_CONFIDENCE_THRESHOLD, MAX_OPEN_TRADES, MIN_PROFIT_PERCENT
    global USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY, USE_MOMENTUM_VOLATILITY_STRATEGY
    global STRATEGY_FILTER_CONFIG
    if not redis_client: return
    try:
        settings_data = redis_client.get('trading_settings')
        if settings_data:
            settings = json.loads(settings_data)
            with risk_per_trade_lock: RISK_PER_TRADE_PERCENT = settings.get('RISK_PER_TRADE_PERCENT', 0.85)
            with buy_confidence_lock: BUY_CONFIDENCE_THRESHOLD = settings.get('BUY_CONFIDENCE_THRESHOLD', 0.53)
            MAX_OPEN_TRADES = settings.get('MAX_OPEN_TRADES', 3)
            MIN_PROFIT_PERCENT = settings.get('MIN_PROFIT_PERCENT', 0.8)
        
        strategies_data = redis_client.get('strategy_settings')
        if strategies_data:
            strategies = json.loads(strategies_data)
            with bb_stoch_strategy_lock: USE_BB_STOCH_STRATEGY = strategies.get('USE_BB_STOCH_STRATEGY', True)
            with macd_ema_strategy_lock: USE_MACD_EMA_STRATEGY = strategies.get('USE_MACD_EMA_STRATEGY', True)
            with ema_rsi_strategy_lock: USE_EMA_RSI_STRATEGY = strategies.get('USE_EMA_RSI_STRATEGY', True)
            with pullback_strategy_lock: USE_PULLBACK_STRATEGY = strategies.get('USE_PULLBACK_STRATEGY', True)
            with momentum_volatility_strategy_lock: USE_MOMENTUM_VOLATILITY_STRATEGY = strategies.get('USE_MOMENTUM_VOLATILITY_STRATEGY', True)

        # [جديد] تحميل إعدادات الفلاتر
        filters_data = redis_client.get('strategy_filter_config')
        if filters_data:
            with strategy_filters_lock:
                STRATEGY_FILTER_CONFIG = json.loads(filters_data)
                logger.info("✅ [Redis] Successfully loaded strategy filter settings from Redis.")
        else:
            logger.info("⚠️ [Redis] No strategy filter settings found in Redis, using defaults.")

    except Exception as e:
        logger.error(f"❌ [Redis] Error loading settings: {e}")

# --- [مُحَدَّث] منطق التداول والفلاتر الديناميكية ---
def check_market_volatility_filter(df: pd.DataFrame) -> bool:
    last = df.iloc[-1]
    atr_percent = (last['atr'] / last['close']) * 100
    # يمكن جعل هذه القيم قابلة للتعديل في المستقبل
    if atr_percent < 0.5 or atr_percent > 5.0:
        log_rejection(df.name, "Market Volatility Filter Failed", {"atr_percent": f"{atr_percent:.2f}"})
        return False
    return True

def check_trend_strength_filter(df: pd.DataFrame, adx_threshold: int) -> bool:
    last = df.iloc[-1]
    if last['adx'] < adx_threshold:
        log_rejection(df.name, "Trend Strength Filter Failed", {"adx": f"{last['adx']:.2f}", "threshold": adx_threshold})
        return False
    return True

def is_htf_bullish_confirmation(symbol: str, htf: str = '1h', mode: str = 'Strict') -> bool:
    if mode == 'Disabled':
        return True
    try:
        df = fetch_historical_data(symbol, htf, days=40) 
        if df is None or len(df) < 200: return False
        df['ema50']  = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        last = df.iloc[-1]
        
        if mode == 'Strict':
            return last['close'] > last['ema50'] and last['ema50'] > last['ema200']
        elif mode == 'Relaxed':
            # يكفي أن يكون السعر فوق المتوسط 50 (ليس في ترند هابط واضح)
            return last['close'] > last['ema50']
        
        return False # Default case
    except Exception as e:
        logger.warning(f"[HTF] Could not confirm HTF trend for {symbol} (Mode: {mode}): {e}")
        return False

# --- [جديد] دالة تطبيق الفلاتر الديناميكية ---
def apply_strategy_filters(symbol: str, df: pd.DataFrame, strategy_name: str) -> bool:
    """
    تطبق هذه الدالة الفلاتر بناءً على الإعدادات المخصصة لكل استراتيجية.
    """
    with strategy_filters_lock:
        config = STRATEGY_FILTER_CONFIG.get(strategy_name)
    
    if not config:
        logger.warning(f"[Filters] No filter configuration found for strategy '{strategy_name}'. Skipping filters.")
        return True

    profile = config.get("profile", "Strict")
    
    # إذا كان ملف التعريف معطلاً، نتجاوز كل الفلاتر
    if profile == "Disabled":
        return True

    # 1. فلتر التقلب (دائماً نشط إلا إذا كان الملف معطلاً)
    if not check_market_volatility_filter(df):
        return False

    # 2. فلتر قوة الاتجاه (ADX)
    adx_threshold = config.get("adx_threshold", 22)
    if not check_trend_strength_filter(df, adx_threshold):
        return False

    # 3. فلتر تأكيد الاتجاه على الفريم الأعلى (HTF)
    htf_mode = config.get("htf_confirmation_mode", "Strict")
    if not is_htf_bullish_confirmation(symbol, HIGHER_TIMEFRAME, htf_mode):
        log_rejection(symbol, "HTF Trend Confirmation Failed", {"mode": htf_mode})
        return False
        
    return True


# --- دوال مساعدة للاستراتيجيات (بدون تغيير) ---
def check_rsi_bullish_divergence(df: pd.DataFrame, lookback: int = 25) -> bool:
    if len(df) < lookback: return False
    try:
        subset = df.iloc[-lookback:]
        low_price_idx = subset['low'].idxmin()
        low_rsi_idx = subset['rsi'].idxmin()
        if low_price_idx == low_rsi_idx: return False
        price_before_low = subset.loc[:low_price_idx]['low'].iloc[:-1]
        if price_before_low.empty: return False
        second_low_price_idx = price_before_low.idxmin()
        first_low_price = subset.loc[second_low_price_idx]['low']
        second_low_price = subset.loc[low_price_idx]['low']
        first_low_rsi = subset.loc[second_low_price_idx]['rsi']
        second_low_rsi = subset.loc[low_price_idx]['rsi']
        if second_low_price < first_low_price and second_low_rsi > first_low_rsi:
            logger.info(f"[{df.name}] Bullish RSI Divergence detected.")
            return True
    except Exception as e:
        logger.debug(f"[Divergence] Error checking RSI divergence for {df.name}: {e}")
    return False

def check_fibonacci_pullback(df: pd.DataFrame, lookback: int = 50) -> bool:
    if len(df) < lookback: return False
    try:
        subset = df.iloc[-lookback:]
        swing_high = subset['high'].max()
        swing_low = subset['low'].min()
        if swing_high == swing_low: return False
        fib_50 = swing_high - 0.5 * (swing_high - swing_low)
        fib_618 = swing_high - 0.618 * (swing_high - swing_low)
        last_candle = df.iloc[-1]
        is_at_fib_50 = last_candle['low'] <= fib_50 and last_candle['close'] > fib_50
        is_at_fib_618 = last_candle['low'] <= fib_618 and last_candle['close'] > fib_618
        if is_at_fib_50 or is_at_fib_618:
            logger.info(f"[{df.name}] Fibonacci pullback bounce detected.")
            return True
    except Exception as e:
        logger.debug(f"[Fibonacci] Error checking fib pullback for {df.name}: {e}")
    return False

# --- استراتيجيات التداول (بدون تغيير) ---
def check_bb_stoch_strategy_enhanced(df: pd.DataFrame) -> bool:
    if len(df) < 21: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    bb_breakout = ((prev['low'] <= prev['bb_lower'] * 1.001) and (last['close'] > last['open']) and (last['close'] > last['bb_lower']) and (last['low'] > last['bb_lower'] * 0.995))
    stoch_signal = ((prev['stoch_rsi_k'] < 25) and (last['stoch_rsi_k'] > prev['stoch_rsi_k']) and (last['stoch_rsi_k'] < 45) and (last['rsi'] > 30) and (last['rsi'] < 60))
    volume_ok = ((last['volume'] > last['volume_sma'] * 1.8) and (last['volume'] > prev['volume'] * 1.2))
    momentum_ok = ((last[f'roc_{MOMENTUM_PERIOD}'] > 0) and (last['money_flow_index'] > 40))
    trend_ok = ((last['close'] > last['ema21']) and (last['ema21'] > prev['ema21']))
    return bb_breakout and stoch_signal and volume_ok and momentum_ok and trend_ok

def check_macd_ema_strategy_enhanced(df: pd.DataFrame) -> bool:
    if len(df) < 30: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    macd_cross = ((prev['macd'] < prev['macd_signal']) and (last['macd'] > last['macd_signal']) and (last['macd'] > 0) and (last['macd'] - last['macd_signal'] > 0.0001 * last['close']))
    price_above_ema = ((last['close'] > last['ema12']) and (last['close'] > last['ema26']) and (last['ema12'] > last['ema26']) and (last['ema12'] > prev['ema12']) and (last['ema26'] > prev['ema26']))
    trend_strength = ((last['adx'] > 22) and (last['adx'] > prev['adx']))
    volatility_ok = ((last['atr'] > last['atr_sma'] * 0.9) and (last['atr'] < last['atr_sma'] * 1.5))
    volume_ok = ((last['volume'] > last['volume_sma'] * 1.3) and (last['volume'] > prev['volume']))
    force_ok = ((last['force_index'] > 0) and (last['rsi'] > 50))
    return macd_cross and price_above_ema and trend_strength and volatility_ok and volume_ok and force_ok

def check_ema_rsi_strategy_enhanced(df: pd.DataFrame) -> bool:
    if len(df) < 30: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    ema_cross = ((prev['ema9'] < prev['ema12']) and (last['ema9'] > last['ema12']) and (last['ema9'] > prev['ema9']) and (last['ema12'] > prev['ema12']))
    rsi_signal = ((50 < last['rsi'] < 65) and (last['rsi'] > last['rsi_ma']) and (last['rsi'] > prev['rsi']))
    price_above_slow_ema = ((last['close'] > last['ema26']) and (last['close'] > last['ema50']) and (last['ema26'] > prev['ema26']))
    reversal_condition = (check_rsi_bullish_divergence(df) or (prev['williams_r'] < -80 and last['williams_r'] > prev['williams_r'] and last['williams_r'] < -50))
    volume_ok = ((last['volume'] > df['volume'].rolling(window=5).mean().iloc[-1] * 1.4) and (last['volume'] > prev['volume'] * 1.1))
    close_strong = ((last['close'] - last['low']) > (last['high'] - last['low']) * 0.6)
    return ema_cross and rsi_signal and price_above_slow_ema and (reversal_condition or volume_ok) and close_strong

def check_pullback_strategy_enhanced(df: pd.DataFrame) -> bool:
    if len(df) < 50: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    ema_trend = ((last['close'] > last['ema12']) and (last['close'] > last['ema26']) and (last['ema12'] > last['ema26']) and (last['ema12'] > prev['ema12']) and (last['ema26'] > prev['ema26']))
    macd_condition = (((prev['macd'] < prev['macd_signal']) and (last['macd'] > last['macd_signal'])) or ((last['macd'] > last['macd_signal']) and (last['macd'] > 0)))
    pullback_condition = (check_fibonacci_pullback(df) or (last['low'] <= last['keltner_lower'] * 1.01 and last['close'] > last['keltner_lower']))
    volume_pattern = ((df['volume'].rolling(window=5).mean().iloc[-2] > df['volume'].rolling(window=10).mean().iloc[-2]) and (prev['volume'] < df['volume'].rolling(window=5).mean().iloc[-2]) and (last['volume'] > prev['volume']))
    volatility_ok = ((last['atr_percent'] > 0.8) and (last['atr_percent'] < 3.0))
    reversal_candle = ((last['close'] > last['open']) and (last['close'] > (last['high'] + last['low']) / 2) and (last['close'] > prev['close']))
    return ema_trend and macd_condition and pullback_condition and volume_pattern and volatility_ok and reversal_candle

def check_momentum_volatility_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 50: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    rsi_condition = ((45 < last['rsi_avg'] < 55) and (last['rsi_avg'] > prev['rsi_avg']) and (last['rsi9'] > last['rsi14'] > last['rsi21']))
    volatility_condition = ((last['atr_percent'] > last['atr_percent_sma'] * 1.2) and (last['atr_percent'] > prev['atr_percent']))
    trend_condition = ((last['close'] > last['ema9'] > last['ema21'] > last['ema50']) and (last['ema9'] > prev['ema9']) and (last['ema21'] > prev['ema21']))
    volume_condition = ((last['volume'] > last['volume_sma'] * 1.5) and (last['volume'] > prev['volume']))
    breakout_condition = ((prev['close'] < prev['dc_upper']) and (last['close'] > last['dc_upper']) and (last['close'] > last['vwap']))
    confirmation_condition = ((last['close'] - last['low']) > (last['high'] - last['low']) * 0.66)
    return (rsi_condition and volatility_condition and trend_condition and volume_condition and breakout_condition and confirmation_condition)

# --- نظام إدارة الصفقات (بدون تغيير) ---
def calculate_trade_levels(df: pd.DataFrame, strategy_name: str) -> Dict[str, Any]:
    last = df.iloc[-1]
    atr = last['atr']
    close = last['close']
    base_stop_distance = atr * 1.5
    if "BB_Stoch" in strategy_name:
        stop_distance, target1_distance, target2_distance, trailing_activation = base_stop_distance * 0.9, atr * 2.2, atr * 3.8, 0.6
    elif "MACD_EMA" in strategy_name:
        stop_distance, target1_distance, target2_distance, trailing_activation = base_stop_distance * 1.0, atr * 2.0, atr * 3.5, 0.5
    elif "EMA_RSI" in strategy_name:
        stop_distance, target1_distance, target2_distance, trailing_activation = base_stop_distance * 1.1, atr * 1.8, atr * 3.2, 0.4
    elif "Pullback" in strategy_name:
        stop_distance, target1_distance, target2_distance, trailing_activation = base_stop_distance * 0.8, atr * 2.5, atr * 4.2, 0.7
    elif "Momentum_Volatility" in strategy_name:
        stop_distance, target1_distance, target2_distance, trailing_activation = base_stop_distance * 1.2, atr * 2.8, atr * 4.5, 0.5
    else:
        stop_distance, target1_distance, target2_distance, trailing_activation = base_stop_distance, atr * 2.0, atr * 3.5, 0.5
    entry_price = close
    stop_loss = entry_price - stop_distance
    target_price_1 = entry_price + target1_distance
    target_price_2 = entry_price + target2_distance
    partial_exit_levels = [
        {"level": entry_price + (target1_distance * 0.5), "percentage": 0.25, "executed": False},
        {"level": target_price_1, "percentage": 0.25, "executed": False},
        {"level": entry_price + (target1_distance + target2_distance) * 0.5, "percentage": 0.25, "executed": False},
        {"level": target_price_2, "percentage": 0.25, "executed": False},
    ]
    return {"entry_price": entry_price, "stop_loss": stop_loss, "target_price_1": target_price_1, "target_price_2": target_price_2, "partial_exit_levels": partial_exit_levels, "trailing_activation": trailing_activation, "atr": atr}

def create_paper_trade_signal(symbol: str, df: pd.DataFrame, strategy_name: str) -> None:
    try:
        trade_levels = calculate_trade_levels(df, strategy_name)
        entry_price, stop_loss = trade_levels['entry_price'], trade_levels['stop_loss']
        target_price_1, target_price_2 = trade_levels['target_price_1'], trade_levels['target_price_2']
        if entry_price <= stop_loss:
            logger.warning(f"[Signal] Could not create signal for {symbol}: Stop loss ({stop_loss}) would be at or above entry price ({entry_price}).")
            return
        quantity = PAPER_TRADE_SIZE_USDT / entry_price
        if not (check_db_connection() and conn):
            logger.error(f"❌ [DB] Cannot create signal for {symbol}, no database connection.")
            return
        signal_details = {
            "atr": trade_levels['atr'], "initial_atr": trade_levels['atr'],
            "partial_exit_levels": trade_levels['partial_exit_levels'],
            "trailing_activation": trade_levels['trailing_activation'],
            "current_trailing_stop": stop_loss, "is_trailing_active": False,
            "last_target_update": time.time()
        }
        new_id = None
        with conn.cursor() as cur:
            logger.info(f"[DB] Queuing new signal for {symbol}...")
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price_1, target_price_2, stop_loss, status, 
                                   strategy_name, is_real_trade, quantity, initial_quantity, signal_details) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
            """, (symbol, float(entry_price), float(target_price_1), float(target_price_2), float(stop_loss), 'open',
                  strategy_name, False, float(quantity), float(quantity), json.dumps(signal_details, cls=NpEncoder)))
            result = cur.fetchone()
            if result and 'id' in result: new_id = result['id']
            else:
                logger.error(f"❌ [DB] FAILED TO RETRIEVE ID for new signal {symbol}. Rolling back.")
                conn.rollback()
                return
        signal_data = {
            'id': new_id, 'symbol': symbol, 'entry_price': float(entry_price), 
            'target_price_1': float(target_price_1), 'target_price_2': float(target_price_2),
            'stop_loss': float(stop_loss), 'status': 'open', 'strategy_name': strategy_name, 
            'is_real_trade': False, 'quantity': float(quantity), 'initial_quantity': float(quantity),
            'signal_details': signal_details
        }
        with signal_cache_lock:
            open_signals_cache[symbol] = signal_data
            logger.info(f"[Cache] Updated cache for new signal {symbol} (ID: {new_id}).")
        conn.commit()
        logger.info(f"✅ [DB] Transaction committed. Signal {new_id} ({symbol}) is now live.")
        message = (f"📊 *فتح صفقة ورقية جديدة*\n"
                   f"💱 *العملة:* `{symbol}`\n"
                   f"📈 *الاستراتيجية:* {strategy_name}\n"
                   f"📌 *الدخول:* `{entry_price:.4f}`\n"
                   f"💰 *الكمية:* `{quantity:.4f}`\n"
                   f"🎯 *الهدف 1:* `{target_price_1:.4f}`\n"
                   f"🎯 *الهدف 2:* `{target_price_2:.4f}`\n"
                   f"🛑 *الوقف:* `{stop_loss:.4f}`")
        send_telegram_message(message)
        log_and_notify("info", f"Opened paper trade for {symbol}", "PAPER_TRADE_OPEN")
    except Exception as e:
        logger.error(f"❌ [Signal] CRITICAL ERROR creating paper trade for {symbol}: {e}", exc_info=True)
        if conn:
            logger.info(f"[DB] Rolling back transaction for {symbol} due to error.")
            conn.rollback()
        with signal_cache_lock:
            if symbol in open_signals_cache and open_signals_cache[symbol].get('id') == new_id:
                del open_signals_cache[symbol]
                logger.warning(f"[Cache] Removed inconsistent cache entry for {symbol} due to creation failure.")

# --- قوالب HTML (بدون تغيير في القالب الرئيسي) ---
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم بوت التداول</title>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #121212; --bg-surface: #1e1e1e; --primary: #BB86FC;
            --primary-variant: #3700B3; --secondary: #03DAC6; --text-light: #e0e0e0;
            --text-medium: #a0a0a0; --success: #4CAF50; --danger: #F44336;
            --warning: #FFC107; --bullish: #26a69a; --bearish: #ef5350;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background-color: var(--bg-dark); color: var(--text-light); font-family: 'Tajawal', sans-serif; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        header { background-color: var(--bg-surface); padding: 15px 25px; border-radius: 12px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: center; border: 1px solid #2a2a2a; }
        .header-title { font-size: 24px; font-weight: 700; color: var(--primary); }
        .status-indicator { display: flex; align-items: center; gap: 15px; }
        .status-dot { width: 12px; height: 12px; border-radius: 50%; background-color: var(--danger); box-shadow: 0 0 8px var(--danger); }
        .status-dot.active { background-color: var(--success); box-shadow: 0 0 8px var(--success); }
        .btn { background-color: var(--primary-variant); color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; transition: background-color 0.3s, transform 0.2s; font-weight: 700; text-decoration: none; }
        .btn:hover { background-color: var(--primary); transform: translateY(-2px); }
        .btn.stop { background-color: var(--danger); }
        .btn.stop:hover { background-color: #d32f2f; }
        .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 20px; }
        .card { background-color: var(--bg-surface); border-radius: 12px; padding: 20px; border: 1px solid #2a2a2a; display: flex; flex-direction: column; }
        .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #333; }
        .card-title { font-size: 18px; font-weight: 700; color: var(--text-light); }
        .scrollable-content { overflow-y: auto; max-height: 400px; padding-right: 10px; }
        .scrollable-content::-webkit-scrollbar { width: 6px; }
        .scrollable-content::-webkit-scrollbar-track { background: #2a2a2a; }
        .scrollable-content::-webkit-scrollbar-thumb { background: var(--primary); border-radius: 3px; }
        .item { padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid var(--primary); background-color: #252525; }
        .item-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
        .item-title { font-weight: 700; }
        .item-time { font-size: 12px; color: var(--text-medium); }
        .item-content { font-size: 13px; color: var(--text-light); line-height: 1.6; }
        .state-item { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px; }
        .state-label { font-weight: bold; color: var(--text-medium); }
        .state-value.Bullish { color: var(--bullish); font-weight: 700; }
        .state-value.Bearish { color: var(--bearish); font-weight: 700; }
        .state-value.Sideways { color: var(--text-medium); }
        .signal-item.paper { border-left-color: var(--secondary); }
        .signal-item.updated { border-left-color: var(--warning); }
        .notification-item.info { border-left-color: var(--primary); }
        .notification-item.warning { border-left-color: var(--warning); }
        .notification-item.error, .notification-item.trading_status { border-left-color: var(--danger); }
        .rejection-item { border-left-color: var(--warning); }
        .progress-bar-container { background-color: #333; border-radius: 10px; height: 10px; overflow: hidden; margin-top: 10px; direction: ltr; }
        .progress-bar { height: 100%; transition: width 0.4s ease-in-out; border-radius: 10px; }
        .footer { text-align: center; margin-top: 30px; padding: 15px; color: var(--text-medium); font-size: 14px; }
        .signal-actions { margin-top: 10px; display: flex; justify-content: flex-end; }
        .btn-close { background-color: var(--danger); color: white; border: none; padding: 5px 12px; font-size: 12px; border-radius: 6px; cursor: pointer; transition: background-color 0.2s; }
        .btn-close:hover { background-color: #c0392b; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-title">بوت التداول V14.0.0</div>
            <div class="status-indicator">
                <div class="status-dot {{ 'active' if trading_enabled else '' }}"></div>
                <span>{{ 'نشط' if trading_enabled else 'متوقف' }}</span>
                <button class="btn {{ 'stop' if trading_enabled else '' }}" onclick="toggleTrading()">{{ 'إيقاف' if trading_enabled else 'تشغيل' }}</button>
                <a href="/settings" class="btn">الإعدادات</a>
            </div>
        </header>
        <div class="dashboard-grid">
            <div class="card">
                <div class="card-header"><div class="card-title">حالة السوق (BTC)</div></div>
                <div class="item">
                    <div class="state-item"><span class="state-label">النظام العام:</span><span class="state-value {{ market_state.get('overall_regime', 'N/A') }}">{{ market_state.get('overall_regime', 'N/A') }}</span></div>
                    {% for tf, details in market_state.get('trend_details_by_tf', {}).items() %}
                    <div class="state-item"><span class="state-label">اتجاه {{ tf }}:</span><span class="state-value {{ details.get('trend', 'N/A') }}">{{ details.get('trend', 'N/A') }} (RSI: {{ "%.1f"|format(details.get('rsi', 0)) }})</span></div>
                    {% endfor %}
                    <div class="state-item" style="margin-top:10px; font-size: 12px; color: var(--text-medium);"><span>آخر تحديث:</span><span>{{ market_state.get('last_updated', 'N/A') }}</span></div>
                </div>
            </div>
            <div class="card">
                <div class="card-header"><div class="card-title">الإشارات المفتوحة ({{ open_signals|length }})</div></div>
                <div class="scrollable-content">
                {% if open_signals %}{% for symbol, signal in open_signals.items() %}
                <div class="item signal-item {{ 'paper' if not signal.get('is_real_trade') else '' }} {{ signal.get('status', 'open') }}">
                    <div class="item-header">
                        <div class="item-title">{{ symbol }} 
                            <span style="font-size:12px; color: var(--text-medium);">({{ 'ورقية' if not signal.get('is_real_trade') else 'حقيقية' }})</span>
                            {% if signal.get('status') == 'updated' %}<span style="font-size:11px; color: var(--warning); font-weight: bold;">(ربح جزئي)</span>{% endif %}
                        </div>
                        <div class="item-time">{{ signal.get('strategy_name', '') }}</div>
                    </div>
                    <div class="item-content">
                        دخول: {{ "%.4f"|format(signal.get('entry_price', 0)) }} | حالي: {{ "%.4f"|format(signal.get('current_price', 0)) }}<br>
                        هدف: {{ "%.4f"|format(signal.get('target_price_1', 0)) }} | وقف: {{ "%.4f"|format(signal.get('stop_loss', 0)) }}
                    </div>
                    <div class="progress-bar-container">
                        {% set progress = signal.get('progress', 0) %}{% if progress >= 0 %}<div class="progress-bar" style="width: {{ [progress, 100]|min }}%; background-color: var(--success);"></div>{% else %}<div class="progress-bar" style="width: {{ [progress|abs, 100]|min }}%; background-color: var(--danger); float: right;"></div>{% endif %}
                    </div>
                    <div class="signal-actions">
                        <button class="btn-close" onclick="manualClose({{ signal.id }})">إغلاق</button>
                    </div>
                </div>
                {% endfor %}{% else %}<div style="text-align: center; padding: 20px; color: var(--text-medium);">لا توجد إشارات مفتوحة</div>{% endif %}
                </div>
            </div>
            <div class="card">
                <div class="card-header"><div class="card-title">الإشعارات الأخيرة</div></div>
                <div class="scrollable-content">
                {% for notif in notifications %}<div class="item notification-item {{ notif.get('type', 'info').lower() }}"><div class="item-header"><div class="item-title">{{ notif.get('type', 'INFO') }}</div><div class="item-time">{{ notif.get('timestamp', '')[:16] }}</div></div><div class="item-content">{{ notif.get('message', '') }}</div></div>{% endfor %}
                </div>
            </div>
            <div class="card">
                <div class="card-header"><div class="card-title">سجل الرفض</div></div>
                <div class="scrollable-content">
                {% for rej in rejections %}<div class="item rejection-item"><div class="item-header"><div class="item-title">{{ rej.get('symbol', 'N/A') }}</div><div class="item-time">{{ rej.get('timestamp', '')[:16] }}</div></div><div class="item-content">{{ rej.get('reason', 'N/A') }}</div></div>{% endfor %}
                </div>
            </div>
        </div>
        <div class="footer"><div>بوت التداول الإلكتروني V14.0.0</div></div>
    </div>
    <script>
        function showAlert(message, type = 'info') {
            const alertBox = document.createElement('div');
            const bgColor = type === 'success' ? 'var(--success)' : type === 'error' ? 'var(--danger)' : 'var(--primary)';
            Object.assign(alertBox.style, { position: 'fixed', bottom: '20px', left: '20px', padding: '15px 25px', borderRadius: '8px', color: 'white', zIndex: '1000', backgroundColor: bgColor, boxShadow: '0 4px 15px rgba(0,0,0,0.2)', transform: 'translateY(100px)', opacity: '0', transition: 'transform 0.4s ease, opacity 0.4s ease' });
            alertBox.innerText = message;
            document.body.appendChild(alertBox);
            setTimeout(() => { alertBox.style.transform = 'translateY(0)'; alertBox.style.opacity = '1'; }, 10);
            setTimeout(() => { alertBox.style.transform = 'translateY(100px)'; alertBox.style.opacity = '0'; setTimeout(() => alertBox.remove(), 400); }, 4000);
        }
        function toggleTrading() {
            fetch('/toggle_trading', { method: 'POST' }).then(res => res.json()).then(data => {
                showAlert(data.message, data.success ? 'success' : 'error');
                if(data.success) setTimeout(() => location.reload(), 1500);
            });
        }
        function manualClose(signalId) {
            const modalHTML = `<div id="confirm-modal" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); display: flex; align-items: center; justify-content: center; z-index: 2000;"><div style="background: var(--bg-surface); padding: 25px; border-radius: 12px; text-align: center; border: 1px solid #333;"><p style="margin-bottom: 20px;">هل أنت متأكد من إغلاق الصفقة يدويًا؟</p><button id="confirm-yes" class="btn">نعم</button><button id="confirm-no" class="btn" style="margin-right: 10px; background-color: #555;">لا</button></div></div>`;
            document.body.insertAdjacentHTML('beforeend', modalHTML);
            document.getElementById('confirm-yes').onclick = () => {
                fetch('/close_signal/' + signalId, { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    showAlert(data.message, data.success ? 'success' : 'error');
                    if(data.success) setTimeout(() => location.reload(), 1500);
                });
                document.getElementById('confirm-modal').remove();
            };
            document.getElementById('confirm-no').onclick = () => {
                document.getElementById('confirm-modal').remove();
            };
        }
        setInterval(() => location.reload(), 60000);
    </script>
</body>
</html>
"""

# --- [مُحَدَّث] قالب صفحة الإعدادات ---
SETTINGS_TEMPLATE = """
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>إعدادات البوت</title>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #121212; --bg-surface: #1e1e1e; --primary: #BB86FC;
            --primary-variant: #3700B3; --secondary: #03DAC6; --text-light: #e0e0e0;
            --text-medium: #a0a0a0; --danger: #F44336;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background-color: var(--bg-dark); color: var(--text-light); font-family: 'Tajawal', sans-serif; }
        .container { max-width: 900px; margin: 0 auto; padding: 20px; }
        header { background-color: var(--bg-surface); padding: 15px 25px; border-radius: 12px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: center; border: 1px solid #2a2a2a; }
        .header-title { font-size: 24px; font-weight: 700; color: var(--primary); }
        .btn { background-color: var(--primary-variant); color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; transition: all 0.3s; font-weight: 700; text-decoration: none; }
        .btn:hover { background-color: var(--primary); transform: translateY(-2px); }
        .settings-form { background-color: var(--bg-surface); border-radius: 12px; padding: 25px; margin-bottom: 20px; border: 1px solid #2a2a2a; }
        .form-section-title { font-size: 20px; font-weight: 700; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid #333; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; margin-bottom: 8px; font-weight: bold; color: var(--text-medium); }
        .form-group input[type="number"], .form-group select { width: 100%; padding: 12px; border: 1px solid #333; border-radius: 8px; background-color: #252525; color: var(--text-light); font-size: 16px; }
        .form-actions { display: flex; justify-content: space-between; align-items: center; margin-top: 25px; gap: 15px; }
        .btn-secondary { background-color: #333; }
        .btn-secondary:hover { background-color: #444; }
        .checkbox-group { display: flex; align-items: center; gap: 10px; padding: 10px; border-radius: 8px; transition: background-color 0.2s; }
        .checkbox-group:hover { background-color: #252525; }
        .checkbox-group input { width: 18px; height: 18px; accent-color: var(--primary); }
        .checkbox-group label { margin-bottom: 0; cursor: pointer; }
        .toggle-section { text-align: center; padding: 20px; background-color: var(--bg-surface); border-radius: 12px; margin-bottom: 20px; border: 1px solid #2a2a2a; }
        .toggle-switch { display: inline-flex; align-items: center; gap: 10px; background-color: #252525; padding: 5px; border-radius: 20px; cursor: pointer; }
        .toggle-switch .label { padding: 5px 15px; border-radius: 15px; font-weight: bold; transition: all 0.3s; }
        #paper-trading-toggle:checked ~ .labels .paper { background-color: var(--primary); color: white; }
        #paper-trading-toggle:not(:checked) ~ .labels .real { background-color: var(--secondary); color: var(--bg-dark); }
        #paper-trading-toggle { display: none; }
        .filter-table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        .filter-table th, .filter-table td { padding: 12px; text-align: right; border-bottom: 1px solid #333; }
        .filter-table th { color: var(--text-medium); font-size: 14px; }
        .filter-table td { vertical-align: middle; }
        .filter-table select, .filter-table input { padding: 8px; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-title">إعدادات البوت</div>
            <a href="/" class="btn">العودة للرئيسية</a>
        </header>
        <div class="toggle-section">
            <h3 class="form-section-title">وضع التداول</h3>
            <label class="toggle-switch">
                <input type="checkbox" id="paper-trading-toggle" {{ 'checked' if paper_trading_mode else '' }}>
                <div class="labels">
                    <span class="label real">حقيقي</span>
                    <span class="label paper">ورقي</span>
                </div>
            </label>
        </div>
        <div class="settings-form">
            <h3 class="form-section-title">إعدادات التداول العامة</h3>
            <form id="settings-form">
                <div class="form-group"><label for="risk-per-trade">نسبة المخاطرة للصفقة (%)</label><input type="number" id="risk-per-trade" name="risk_per_trade" step="0.1" value="{{ RISK_PER_TRADE_PERCENT }}"></div>
                <div class="form-group"><label for="buy-confidence">حد الثقة للشراء</label><input type="number" id="buy-confidence" name="buy_confidence" step="0.01" value="{{ BUY_CONFIDENCE_THRESHOLD }}"></div>
                <div class="form-group"><label for="max-trades">الحد الأقصى للصفقات المفتوحة</label><input type="number" id="max-trades" name="max_trades" value="{{ MAX_OPEN_TRADES }}"></div>
                <div class="form-group"><label for="min-profit">الحد الأدنى للربح (%)</label><input type="number" id="min-profit" name="min_profit" step="0.1" value="{{ MIN_PROFIT_PERCENT }}"></div>
                <div class="form-actions"><button type="button" class="btn btn-secondary" onclick="resetSettings()">إعادة الافتراضي</button><button type="submit" class="btn">حفظ الإعدادات</button></div>
            </form>
        </div>
        <div class="settings-form">
            <h3 class="form-section-title">تفعيل الاستراتيجيات</h3>
            <form id="strategies-form">
                <div class="form-group checkbox-group"><input type="checkbox" id="use_bb_stoch" name="use_bb_stoch" {{ 'checked' if USE_BB_STOCH_STRATEGY else '' }}><label for="use_bb_stoch">استراتيجية BB+Stoch</label></div>
                <div class="form-group checkbox-group"><input type="checkbox" id="use_macd_ema" name="use_macd_ema" {{ 'checked' if USE_MACD_EMA_STRATEGY else '' }}><label for="use_macd_ema">استراتيجية MACD+EMA</label></div>
                <div class="form-group checkbox-group"><input type="checkbox" id="use_ema_rsi" name="use_ema_rsi" {{ 'checked' if USE_EMA_RSI_STRATEGY else '' }}><label for="use_ema_rsi">استراتيجية EMA+RSI</label></div>
                <div class="form-group checkbox-group"><input type="checkbox" id="use_pullback" name="use_pullback" {{ 'checked' if USE_PULLBACK_STRATEGY else '' }}><label for="use_pullback">استراتيجية Pullback</label></div>
                <div class="form-group checkbox-group"><input type="checkbox" id="use_momentum_volatility" name="use_momentum_volatility" {{ 'checked' if USE_MOMENTUM_VOLATILITY_STRATEGY else '' }}><label for="use_momentum_volatility">استراتيجية التقارب الزخمي-التقلبي</label></div>
                <div class="form-actions"><button type="submit" class="btn">حفظ الاستراتيجيات</button></div>
            </form>
        </div>
        <!-- [جديد] قسم إعدادات الفلاتر -->
        <div class="settings-form">
            <h3 class="form-section-title">إعدادات فلاتر الاستراتيجيات</h3>
            <form id="filters-form">
                <table class="filter-table">
                    <thead>
                        <tr>
                            <th>الاستراتيجية</th>
                            <th>ملف تعريف الفلتر</th>
                            <th>حد قوة الاتجاه (ADX)</th>
                            <th>تأكيد الفريم الأعلى (HTF)</th>
                        </tr>
                    </thead>
                    <tbody>
                    {% for key, config in STRATEGY_FILTER_CONFIG.items() %}
                        <tr>
                            <td>{{ STRATEGY_NAMES.get(key, key) }}</td>
                            <td>
                                <select name="{{ key }}_profile">
                                    <option value="Strict" {{ 'selected' if config.profile == 'Strict' }}>صارم</option>
                                    <option value="Moderate" {{ 'selected' if config.profile == 'Moderate' }}>متوسط</option>
                                    <option value="Reversal" {{ 'selected' if config.profile == 'Reversal' }}>انعكاسي</option>
                                    <option value="Disabled" {{ 'selected' if config.profile == 'Disabled' }}>معطل</option>
                                </select>
                            </td>
                            <td><input type="number" name="{{ key }}_adx_threshold" value="{{ config.adx_threshold }}" min="10" max="50"></td>
                            <td>
                                <select name="{{ key }}_htf_confirmation_mode">
                                    <option value="Strict" {{ 'selected' if config.htf_confirmation_mode == 'Strict' }}>صارم</option>
                                    <option value="Relaxed" {{ 'selected' if config.htf_confirmation_mode == 'Relaxed' }}>مخفف</option>
                                    <option value="Disabled" {{ 'selected' if config.htf_confirmation_mode == 'Disabled' }}>معطل</option>
                                </select>
                            </td>
                        </tr>
                    {% endfor %}
                    </tbody>
                </table>
                <div class="form-actions">
                    <button type="submit" class="btn">حفظ إعدادات الفلاتر</button>
                </div>
            </form>
        </div>
    </div>
    <script>
        function showAlert(message, type = 'info') {
            const alertBox = document.createElement('div');
            const bgColor = type === 'success' ? 'var(--success)' : type === 'error' ? 'var(--danger)' : 'var(--primary)';
            Object.assign(alertBox.style, { position: 'fixed', bottom: '20px', left: '20px', padding: '15px 25px', borderRadius: '8px', color: 'white', zIndex: '1000', backgroundColor: bgColor, boxShadow: '0 4px 15px rgba(0,0,0,0.2)', transform: 'translateY(100px)', opacity: '0', transition: 'transform 0.4s ease, opacity 0.4s ease' });
            alertBox.innerText = message; document.body.appendChild(alertBox);
            setTimeout(() => { alertBox.style.transform = 'translateY(0)'; alertBox.style.opacity = '1'; }, 10);
            setTimeout(() => { alertBox.style.transform = 'translateY(100px)'; alertBox.style.opacity = '0'; setTimeout(() => alertBox.remove(), 400); }, 4000);
        }
        document.getElementById('paper-trading-toggle').addEventListener('change', function() {
            fetch('/toggle_paper_trading', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ paper_trading_mode: this.checked }) }).then(res => res.json()).then(data => showAlert(data.message, data.success ? 'success' : 'error'));
        });
        document.getElementById('settings-form').addEventListener('submit', function(e) {
            e.preventDefault();
            const data = { risk_per_trade: this.risk_per_trade.value, buy_confidence: this.buy_confidence.value, max_trades: this.max_trades.value, min_profit: this.min_profit.value };
            fetch('/update_settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }).then(res => res.json()).then(data => showAlert(data.message, data.success ? 'success' : 'error'));
        });
        document.getElementById('strategies-form').addEventListener('submit', function(e) {
            e.preventDefault();
            const data = { 
                use_bb_stoch: this.use_bb_stoch.checked, use_macd_ema: this.use_macd_ema.checked, 
                use_ema_rsi: this.use_ema_rsi.checked, use_pullback: this.use_pullback.checked,
                use_momentum_volatility: this.use_momentum_volatility.checked
            };
            fetch('/update_strategies', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }).then(res => res.json()).then(data => showAlert(data.message, data.success ? 'success' : 'error'));
        });
        // [جديد] إرسال إعدادات الفلاتر
        document.getElementById('filters-form').addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            const data = {};
            for (let [key, value] of formData.entries()) {
                data[key] = value;
            }
            fetch('/update_filter_settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }).then(res => res.json()).then(data => showAlert(data.message, data.success ? 'success' : 'error'));
        });
        function resetSettings() {
            const modalHTML = `<div id="confirm-modal" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); display: flex; align-items: center; justify-content: center; z-index: 2000;"><div style="background: var(--bg-surface); padding: 25px; border-radius: 12px; text-align: center; border: 1px solid #333;"><p style="margin-bottom: 20px;">هل أنت متأكد من إعادة تعيين كافة الإعدادات؟</p><button id="confirm-yes" class="btn">نعم</button><button id="confirm-no" class="btn btn-secondary" style="margin-right: 10px;">لا</button></div></div>`;
            document.body.insertAdjacentHTML('beforeend', modalHTML);
            document.getElementById('confirm-yes').onclick = () => {
                fetch('/reset_settings', { method: 'POST' }).then(res => res.json()).then(data => { showAlert(data.message, data.success ? 'success' : 'error'); if(data.success) setTimeout(() => location.reload(), 1500); });
                document.getElementById('confirm-modal').remove();
            };
            document.getElementById('confirm-no').onclick = () => { document.getElementById('confirm-modal').remove(); };
        }
    </script>
</body>
</html>
"""

# --- [مُحَدَّث] مسارات Flask ---
@app.route('/')
def dashboard():
    with signal_cache_lock: open_signals = dict(sorted(open_signals_cache.items()))
    with market_state_lock: market_state = current_market_state.copy()
    with trading_status_lock: trading_enabled = is_trading_enabled
    with notifications_lock: notifications = list(notifications_cache)
    with rejection_logs_lock: rejections = list(rejection_logs_cache)
    with live_prices_lock: current_prices = live_prices.copy()
    for symbol, signal in open_signals.items():
        current_price = current_prices.get(symbol)
        if current_price:
            entry, target1, stop = signal.get('entry_price', 0), signal.get('target_price_1', 0), signal.get('stop_loss', 0)
            progress = 0
            if target1 > entry: progress = ((current_price - entry) / (target1 - entry)) * 100
            elif entry > stop: progress = ((current_price - entry) / (entry - stop)) * 100
            signal['current_price'] = current_price
            signal['progress'] = progress
    return render_template_string(DASHBOARD_TEMPLATE, market_state=market_state, trading_enabled=trading_enabled, open_signals=open_signals, notifications=notifications, rejections=rejections)

@app.route('/settings')
def settings():
    with risk_per_trade_lock: risk_val = RISK_PER_TRADE_PERCENT
    with buy_confidence_lock: buy_conf = BUY_CONFIDENCE_THRESHOLD
    with bb_stoch_strategy_lock: use_bb = USE_BB_STOCH_STRATEGY
    with macd_ema_strategy_lock: use_macd = USE_MACD_EMA_STRATEGY
    with ema_rsi_strategy_lock: use_ema = USE_EMA_RSI_STRATEGY
    with pullback_strategy_lock: use_pullback = USE_PULLBACK_STRATEGY
    with momentum_volatility_strategy_lock: use_momentum = USE_MOMENTUM_VOLATILITY_STRATEGY
    with strategy_filters_lock: filters_config = STRATEGY_FILTER_CONFIG.copy()
    return render_template_string(SETTINGS_TEMPLATE, paper_trading_mode=paper_trading_mode, 
                                RISK_PER_TRADE_PERCENT=risk_val, BUY_CONFIDENCE_THRESHOLD=buy_conf, 
                                MAX_OPEN_TRADES=MAX_OPEN_TRADES, MIN_PROFIT_PERCENT=MIN_PROFIT_PERCENT, 
                                USE_BB_STOCH_STRATEGY=use_bb, USE_MACD_EMA_STRATEGY=use_macd, 
                                USE_EMA_RSI_STRATEGY=use_ema, USE_PULLBACK_STRATEGY=use_pullback, 
                                USE_MOMENTUM_VOLATILITY_STRATEGY=use_momentum,
                                STRATEGY_FILTER_CONFIG=filters_config, STRATEGY_NAMES=STRATEGY_NAMES)

@app.route('/toggle_trading', methods=['POST'])
def toggle_trading():
    global is_trading_enabled
    with trading_status_lock:
        is_trading_enabled = not is_trading_enabled
        status = "مفعل" if is_trading_enabled else "معطل"
        log_and_notify("info", f"Trading has been {'enabled' if is_trading_enabled else 'disabled'}.", "TRADING_STATUS")
        send_telegram_message(f"⚙️ تم {status} التداول")
        return jsonify({"success": True, "message": f"تم {status} التداول"})

@app.route('/close_signal/<int:signal_id>', methods=['POST'])
def manual_close_signal_route(signal_id):
    logger.info(f"[Manual Close] Received request to close signal ID: {signal_id}")
    with signal_cache_lock:
        signal_to_close = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)
    if not signal_to_close:
        logger.warning(f"[Manual Close] Signal ID {signal_id} not found in open signals cache.")
        return jsonify({"success": False, "message": "لم يتم العثور على الصفقة"}), 404
    symbol = signal_to_close['symbol']
    with live_prices_lock: current_price = live_prices.get(symbol)
    if not current_price:
        logger.error(f"[Manual Close] No live price available for {symbol} to close signal {signal_id}.")
        return jsonify({"success": False, "message": "لا يوجد سعر حالي متاح للإغلاق"}), 500
    close_signal(signal_to_close, current_price, "MANUAL_CLOSE")
    return jsonify({"success": True, "message": f"تم إرسال أمر إغلاق لصفقة {symbol}"})

@app.route('/toggle_paper_trading', methods=['POST'])
def toggle_paper_trading():
    global paper_trading_mode
    try:
        data = request.json
        paper_trading_mode = data.get('paper_trading_mode', True)
        mode = "ورقي" if paper_trading_mode else "حقيقي"
        log_and_notify("info", f"Trading mode changed to: {'Paper' if paper_trading_mode else 'Real'}", "TRADING_MODE")
        return jsonify({"success": True, "message": f"تم تغيير وضع التداول إلى: {mode}"})
    except Exception as e:
        logger.error(f"[Settings] Error toggling paper trading mode: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/update_settings', methods=['POST'])
def update_settings():
    global RISK_PER_TRADE_PERCENT, BUY_CONFIDENCE_THRESHOLD, MAX_OPEN_TRADES, MIN_PROFIT_PERCENT
    try:
        data = request.json
        with risk_per_trade_lock: RISK_PER_TRADE_PERCENT = float(data['risk_per_trade'])
        with buy_confidence_lock: BUY_CONFIDENCE_THRESHOLD = float(data['buy_confidence'])
        MAX_OPEN_TRADES = int(data['max_trades'])
        MIN_PROFIT_PERCENT = float(data['min_profit'])
        if redis_client: redis_client.set('trading_settings', json.dumps({'RISK_PER_TRADE_PERCENT': RISK_PER_TRADE_PERCENT, 'BUY_CONFIDENCE_THRESHOLD': BUY_CONFIDENCE_THRESHOLD, 'MAX_OPEN_TRADES': MAX_OPEN_TRADES, 'MIN_PROFIT_PERCENT': MIN_PROFIT_PERCENT}))
        log_and_notify("info", "Trading settings have been updated.", "SETTINGS_UPDATE")
        return jsonify({"success": True, "message": "تم تحديث الإعدادات بنجاح"})
    except Exception as e:
        logger.error(f"[Settings] Error updating settings: {e}")
        return jsonify({"success": False, "message": "خطأ في تحديث الإعدادات"}), 500

@app.route('/update_strategies', methods=['POST'])
def update_strategies():
    global USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY, USE_MOMENTUM_VOLATILITY_STRATEGY
    try:
        data = request.json
        with bb_stoch_strategy_lock: USE_BB_STOCH_STRATEGY = data['use_bb_stoch']
        with macd_ema_strategy_lock: USE_MACD_EMA_STRATEGY = data['use_macd_ema']
        with ema_rsi_strategy_lock: USE_EMA_RSI_STRATEGY = data['use_ema_rsi']
        with pullback_strategy_lock: USE_PULLBACK_STRATEGY = data['use_pullback']
        with momentum_volatility_strategy_lock: USE_MOMENTUM_VOLATILITY_STRATEGY = data['use_momentum_volatility']
        if redis_client: redis_client.set('strategy_settings', json.dumps({'USE_BB_STOCH_STRATEGY': USE_BB_STOCH_STRATEGY, 'USE_MACD_EMA_STRATEGY': USE_MACD_EMA_STRATEGY, 'USE_EMA_RSI_STRATEGY': USE_EMA_RSI_STRATEGY, 'USE_PULLBACK_STRATEGY': USE_PULLBACK_STRATEGY, 'USE_MOMENTUM_VOLATILITY_STRATEGY': USE_MOMENTUM_VOLATILITY_STRATEGY}))
        log_and_notify("info", "Strategy activation settings have been updated.", "STRATEGY_UPDATE")
        return jsonify({"success": True, "message": "تم تحديث الاستراتيجيات بنجاح"})
    except Exception as e:
        logger.error(f"[Settings] Error updating strategies: {e}")
        return jsonify({"success": False, "message": "خطأ في تحديث الاستراتيجيات"}), 500

# [جديد] مسار تحديث إعدادات الفلاتر
@app.route('/update_filter_settings', methods=['POST'])
def update_filter_settings():
    global STRATEGY_FILTER_CONFIG
    try:
        data = request.json
        new_config = {}
        for key in STRATEGY_FILTER_CONFIG.keys():
            new_config[key] = {
                "profile": data.get(f"{key}_profile"),
                "adx_threshold": int(data.get(f"{key}_adx_threshold")),
                "htf_confirmation_mode": data.get(f"{key}_htf_confirmation_mode")
            }
        
        with strategy_filters_lock:
            STRATEGY_FILTER_CONFIG = new_config
        
        if redis_client:
            redis_client.set('strategy_filter_config', json.dumps(STRATEGY_FILTER_CONFIG))
            
        log_and_notify("info", "Strategy filter settings have been updated.", "FILTER_SETTINGS_UPDATE")
        return jsonify({"success": True, "message": "تم تحديث إعدادات الفلاتر بنجاح"})
    except Exception as e:
        logger.error(f"[Settings] Error updating filter settings: {e}")
        return jsonify({"success": False, "message": "خطأ في تحديث إعدادات الفلاتر"}), 500

@app.route('/reset_settings', methods=['POST'])
def reset_settings():
    global RISK_PER_TRADE_PERCENT, BUY_CONFIDENCE_THRESHOLD, MAX_OPEN_TRADES, MIN_PROFIT_PERCENT
    global USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY, USE_MOMENTUM_VOLATILITY_STRATEGY
    global STRATEGY_FILTER_CONFIG
    try:
        # إعادة الإعدادات العامة
        with risk_per_trade_lock: RISK_PER_TRADE_PERCENT = 0.85
        with buy_confidence_lock: BUY_CONFIDENCE_THRESHOLD = 0.53
        MAX_OPEN_TRADES, MIN_PROFIT_PERCENT = 3, 0.8
        # إعادة تفعيل الاستراتيجيات
        with bb_stoch_strategy_lock: USE_BB_STOCH_STRATEGY = True
        with macd_ema_strategy_lock: USE_MACD_EMA_STRATEGY = True
        with ema_rsi_strategy_lock: USE_EMA_RSI_STRATEGY = True
        with pullback_strategy_lock: USE_PULLBACK_STRATEGY = True
        with momentum_volatility_strategy_lock: USE_MOMENTUM_VOLATILITY_STRATEGY = True
        # [جديد] إعادة إعدادات الفلاتر
        with strategy_filters_lock:
            STRATEGY_FILTER_CONFIG = {
                "BB_Stoch_Strategy": {"profile": "Reversal", "adx_threshold": 18, "htf_confirmation_mode": "Disabled"},
                "MACD_EMA_Strategy": {"profile": "Strict", "adx_threshold": 22, "htf_confirmation_mode": "Strict"},
                "EMA_RSI_Strategy": {"profile": "Moderate", "adx_threshold": 20, "htf_confirmation_mode": "Relaxed"},
                "Pullback_Strategy": {"profile": "Reversal", "adx_threshold": 18, "htf_confirmation_mode": "Relaxed"},
                "Momentum_Volatility_Strategy": {"profile": "Strict", "adx_threshold": 25, "htf_confirmation_mode": "Strict"},
            }
        
        if redis_client: redis_client.delete('trading_settings', 'strategy_settings', 'strategy_filter_config')
        log_and_notify("info", "All settings have been reset to their default values.", "SETTINGS_RESET")
        return jsonify({"success": True, "message": "تمت إعادة تعيين الإعدادات"})
    except Exception as e:
        logger.error(f"[Settings] Error resetting settings: {e}")
        return jsonify({"success": False, "message": "خطأ في إعادة تعيين الإعدادات"}), 500

# --- [مُحَدَّث] حلقة العمل الرئيسية ---
def main_bot_loop():
    logger.info("🚀 [Main Loop] Starting signal scanning loop...")
    while True:
        try:
            with trading_status_lock:
                if not is_trading_enabled:
                    time.sleep(10); continue
            
            with signal_cache_lock:
                if len(open_signals_cache) >= MAX_OPEN_TRADES:
                    logger.info(f"[Main Loop] Max open trades ({MAX_OPEN_TRADES}) reached. Pausing scan.")
                    time.sleep(60 * 2)
                    continue

            logger.info("="*20 + " Starting New Scan Cycle " + "="*20)
            for i in range(0, len(validated_symbols_to_scan), SYMBOL_PROCESSING_BATCH_SIZE):
                batch = validated_symbols_to_scan[i:i + SYMBOL_PROCESSING_BATCH_SIZE]
                for symbol in batch:
                    with signal_cache_lock:
                        if symbol in open_signals_cache: continue
                    
                    df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df is None or df.empty or len(df) < 50:
                        log_rejection(symbol, "Insufficient Historical Data"); continue
                    
                    df_featured = calculate_all_features(df); df_featured.name = symbol
                    
                    strategy_found = None
                    if USE_BB_STOCH_STRATEGY and check_bb_stoch_strategy_enhanced(df_featured): strategy_found = "BB_Stoch_Strategy"
                    elif USE_MACD_EMA_STRATEGY and check_macd_ema_strategy_enhanced(df_featured): strategy_found = "MACD_EMA_Strategy"
                    elif USE_EMA_RSI_STRATEGY and check_ema_rsi_strategy_enhanced(df_featured): strategy_found = "EMA_RSI_Strategy"
                    elif USE_PULLBACK_STRATEGY and check_pullback_strategy_enhanced(df_featured): strategy_found = "Pullback_Strategy"
                    elif USE_MOMENTUM_VOLATILITY_STRATEGY and check_momentum_volatility_strategy(df_featured): strategy_found = "Momentum_Volatility_Strategy"

                    if strategy_found:
                        # --- [المنطق الجديد] ---
                        # بدلاً من الفلاتر الثابتة، نستخدم الدالة الديناميكية
                        if apply_strategy_filters(symbol, df_featured, strategy_found):
                            logger.info(f"🌟 [Signal Passed Filters] Confirmed signal for {symbol}! Strategy: {strategy_found}")
                            create_paper_trade_signal(symbol, df_featured, strategy_found)
                        else:
                            # سيتم تسجيل سبب الرفض داخل دالة apply_strategy_filters
                            logger.info(f"Filtered out signal for {symbol} based on dynamic strategy filters.")

            logger.info("="*20 + " Scan Cycle Completed " + "="*20)
            time.sleep(60 * 5)
        except Exception as e:
            logger.error(f"❌ [Main Loop] A critical error occurred: {e}", exc_info=True)
            time.sleep(60)

# --- دوال وحلقات أخرى (بدون تغيير) ---
def close_signal(signal: Dict, closing_price: float, reason: str):
    symbol = signal['symbol']
    signal_id = signal['id']
    entry_price = signal['entry_price']
    profit = ((closing_price - entry_price) / entry_price) * 100
    if not (check_db_connection() and conn):
        logger.error(f"❌ [DB] Cannot close signal {signal_id} for {symbol}, no database connection.")
        return
    try:
        logger.info(f"[DB] Attempting to close signal {signal_id} for {symbol} in database. Reason: {reason}")
        with conn.cursor() as cur:
            cur.execute("UPDATE signals SET status = 'closed', closing_price = %s, closed_at = %s, profit_percentage = %s, closing_reason = %s WHERE id = %s",
                        (closing_price, datetime.now(timezone.utc), profit, reason, signal_id))
        conn.commit()
        logger.info(f"✅ [DB] Successfully committed close status for signal {signal_id} ({symbol}).")
        log_and_notify("info", f"Closed trade for {symbol} due to: {reason}. Profit: {profit:.2f}%", "TRADE_CLOSED")
        send_telegram_message(f"✅ *إغلاق صفقة {symbol}*\n*السبب:* {reason}\n*الربح:* `{profit:.2f}%`")
    except Exception as e:
        logger.error(f"❌ [DB] Failed to update signal {signal_id} for {symbol} to closed status: {e}", exc_info=True)
        if conn: conn.rollback()
    finally:
        with signal_cache_lock:
            if symbol in open_signals_cache:
                del open_signals_cache[symbol]
                logger.info(f"[Cache] Removed signal {signal_id} ({symbol}) from active cache.")

def trade_management_loop():
    logger.info("🚀 [Trade Manager] Starting unified trade management loop...")
    while True:
        try:
            with signal_cache_lock:
                if not open_signals_cache:
                    time.sleep(2)
                    continue
                symbols_to_monitor = list(open_signals_cache.keys())
            for symbol in symbols_to_monitor:
                with signal_cache_lock:
                    if symbol not in open_signals_cache: continue
                    signal = open_signals_cache[symbol]
                with live_prices_lock:
                    current_price = live_prices.get(symbol)
                if not current_price: continue
                stop_loss = signal['stop_loss']
                target_price_2 = signal.get('target_price_2')
                if current_price <= stop_loss:
                    close_signal(signal, stop_loss, "SL_HIT")
                    continue
                if target_price_2 and current_price >= target_price_2:
                    close_signal(signal, target_price_2, "TP2_HIT")
                    continue
            time.sleep(1)
        except Exception as e:
            logger.error(f"❌ [Trade Manager] A critical error occurred in management loop: {e}", exc_info=True)
            time.sleep(10)

def update_market_state_loop():
    logger.info("🚀 [Market State] Starting market state update loop...")
    while True:
        try:
            trend_details, bullish_count = {}, 0
            for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
                days = 10 if tf == '15m' else 30 if tf == '1h' else 90
                btc_df = fetch_historical_data(BTC_SYMBOL, tf, days)
                if btc_df is None or len(btc_df) < 50:
                    trend_details[tf] = {"trend": "Unknown", "rsi": 50}; continue
                btc_df['ema_fast'] = btc_df['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
                btc_df['ema_slow'] = btc_df['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
                btc_df['rsi'] = compute_rsi(btc_df['close'], RSI_PERIOD)
                last = btc_df.iloc[-1]
                trend = "Sideways"
                if last['close'] > last['ema_slow'] and last['ema_fast'] > last['ema_slow'] and last['rsi'] > 55:
                    trend = "Bullish"; bullish_count += 1
                elif last['close'] < last['ema_slow'] and last['ema_fast'] < last['ema_slow'] and last['rsi'] < 45:
                    trend = "Bearish"
                trend_details[tf] = {"trend": trend, "rsi": last['rsi']}
            overall_regime = "Sideways"
            if bullish_count >= 2: overall_regime = "Bullish"
            elif bullish_count == 0: overall_regime = "Bearish"
            with market_state_lock:
                current_market_state.update({'overall_regime': overall_regime, 'trend_details_by_tf': trend_details, 'last_updated': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')})
            time.sleep(60 * 5)
        except Exception as e:
            logger.error(f"❌ [Market State] A critical error occurred: {e}", exc_info=True)
            time.sleep(60)

# --- نقطة بداية البرنامج ---
if __name__ == '__main__':
    logger.info("="*50 + "\n====== Starting Crypto Trading Bot V14.0.0 ======\n" + "="*50)
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
    Thread(target=main_bot_loop, daemon=True).start()
    Thread(target=trade_management_loop, daemon=True).start()
    Thread(target=update_market_state_loop, daemon=True).start()
    logger.info("🌐 [Flask] Starting user interface on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
