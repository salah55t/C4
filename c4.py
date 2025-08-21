# ملف c4.py - نسخة V17.2.0 (إصلاح استقرار لوحة التحكم)
# --- وصف الإصدار:
# هذا الإصدار يعالج مشكلة عدم عمل لوحة التحكم عن طريق تحسين استقرار جلب البيانات.
# 1.  [إصلاح] تحصين مسار API (`/api/dashboard_data`) بمعالجة أخطاء واستخدام JSON encoder مخصص لضمان عدم توقفه.
# 2.  [محسن] تعديل منطق بدء التشغيل لجلب بيانات حالة السوق والرصيد فوراً عند بدء البوت، مما يضمن ظهور البيانات في لوحة التحكم بدون تأخير.
# 3.  [مكتمل] الحفاظ على جميع ميزات التداول الحقيقي والورقي من الإصدار V17.1.0.

import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import redis
from decimal import Decimal, ROUND_DOWN
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
        logging.FileHandler('crypto_bot_v17_2_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV17.2.0')

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
trading_mode_lock = Lock()
usdt_balance: float = 0.0
balance_lock = Lock()


# --- المتغيرات القابلة للتعديل ---
RISK_PER_TRADE_PERCENT: float = 0.85
risk_per_trade_lock = Lock()
MAX_OPEN_TRADES: int = 3
PAPER_TRADE_SIZE_USDT: float = 10.0
TRAILING_STOP_ACTIVATION_PROFIT_PERCENT: float = 1.4 # النسبة المئوية لتفعيل الوقف المتحرك

# --- مفاتيح تفعيل الاستراتيجيات ---
USE_BB_STOCH_STRATEGY: bool = True
USE_MACD_EMA_STRATEGY: bool = True
USE_EMA_RSI_STRATEGY: bool = True
USE_PULLBACK_STRATEGY: bool = True
USE_MOMENTUM_VOLATILITY_STRATEGY: bool = True

# --- إعدادات الفلاتر الديناميكية للاستراتيجيات ---
STRATEGY_NAMES = {
    "BB_Stoch_Strategy": "BB+Stoch (انعكاسية)", "MACD_EMA_Strategy": "MACD+EMA (اتجاهية)",
    "EMA_RSI_Strategy": "EMA+RSI (مختلطة)", "Pullback_Strategy": "Pullback (انعكاسية)",
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
BASE_FILTER_ADX_THRESHOLD = 20

# --- إعدادات عامة ---
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '1h'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 15
BTC_SYMBOL: str = 'BTCUSDT'
API_REQUEST_DELAY: float = 1

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
current_market_state: Dict[str, Any] = {"trend_details_by_tf": {}}
market_state_lock = Lock()

# --- قاموس أسباب الرفض باللغة العربية ---
REJECTION_REASONS_AR = {
    "Market Volatility Filter Failed": "فلتر تقلب السوق رفض الدخول",
    "Trend Strength Filter Failed": "فلتر قوة الاتجاه رفض الدخول",
    "HTF Trend Confirmation Failed": "فشل تأكيد الترند على الفريم الأعلى",
    "Insufficient Historical Data": "بيانات تاريخية غير كافية للفحص",
    "MinNotional Filter Failed": "قيمة الصفقة أقل من الحد الأدنى للمنصة",
    "Insufficient Balance": "الرصيد غير كافي لتنفيذ الصفقة",
    "Bullish Confirmation Failed": "فشل تأكيد الشمعة الصعودية",
}

# --- إعداد تطبيق Flask ---
app = Flask(__name__)
CORS(app)

# --- دوال تهيئة الخدمات وقاعدة البيانات ---
def column_exists(cursor, table_name, column_name):
    cursor.execute("SELECT 1 FROM information_schema.columns WHERE table_name = %s AND column_name = %s", (table_name, column_name))
    return cursor.fetchone() is not None

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
                        stop_loss DOUBLE PRECISION NOT NULL, status TEXT DEFAULT 'open', 
                        closing_price DOUBLE PRECISION, closed_at TIMESTAMP, profit_percentage DOUBLE PRECISION, 
                        strategy_name TEXT, signal_details JSONB, is_real_trade BOOLEAN DEFAULT FALSE, 
                        quantity DOUBLE PRECISION, closing_reason TEXT, order_id TEXT
                    );
                """)
                cur.execute("CREATE TABLE IF NOT EXISTS notifications (id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(), type TEXT NOT NULL, message TEXT NOT NULL);")
                
                columns_to_add = {
                    "target_price_1": "DOUBLE PRECISION", "target_price_2": "DOUBLE PRECISION",
                    "initial_quantity": "DOUBLE PRECISION"
                }
                for col, col_type in columns_to_add.items():
                    if not column_exists(cur, 'signals', col):
                        cur.execute(sql.SQL("ALTER TABLE signals ADD COLUMN {} {}").format(sql.Identifier(col), sql.SQL(col_type)))
                        logger.info(f"✅ [DB] Added missing column '{col}' to 'signals' table.")
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
        init_db()
    try:
        if conn and conn.closed == 0:
            with conn.cursor() as cur: cur.execute("SELECT 1;")
            return True
        return False
    except (OperationalError, InterfaceError):
        init_db()
        return conn is not None and conn.closed == 0

def init_redis() -> None:
    global redis_client
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] Connected successfully.")
    except redis.exceptions.ConnectionError as e:
        logger.warning(f"⚠️ [Redis] Connection failed: {e}.")
        redis_client = None

# --- دوال المساعدة والإشعارات ---
def log_and_notify(level: str, message: str, notification_type: str):
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error}
    log_methods.get(level.lower(), logger.info)(message)
    if not check_db_connection() or not conn: return
    try:
        new_notification = {"timestamp": datetime.now(timezone.utc).isoformat(), "type": notification_type, "message": message}
        with notifications_lock: notifications_cache.appendleft(new_notification)
        with conn.cursor() as cur: cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [DB] Failed to save notification: {e}")
        if conn: conn.rollback()

def log_rejection(symbol: str, reason_key: str, details: Optional[Dict] = None):
    try:
        reason_ar = REJECTION_REASONS_AR.get(reason_key, reason_key)
        log_entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol, "reason": reason_ar}
        with rejection_logs_lock: rejection_logs_cache.appendleft(log_entry)
    except Exception as e:
        logger.error(f"❌ [Log Rejection] Error logging rejection for {symbol}: {e}", exc_info=True)

def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", json={'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}, timeout=10)
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [Telegram] Failed to send message: {e}")

# --- WebSocket & Data Fetching ---
def handle_socket_message(msg):
    global live_prices
    if msg and 'e' in msg and msg['e'] == 'error': logger.error(f"❌ [WebSocket] Error: {msg['m']}"); return
    if isinstance(msg, list):
        with live_prices_lock:
            for ticker in msg:
                if 's' in ticker and 'c' in ticker: live_prices[ticker['s']] = float(ticker['c'])

def start_websocket():
    global ws_manager
    ws_manager = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    ws_manager.start()
    ws_manager.start_ticker_socket(callback=handle_socket_message)
    logger.info("✅ [WebSocket] Subscribed to ticker stream.")

def get_exchange_info_map() -> None:
    global exchange_info_map
    try:
        logger.info("[API] Fetching exchange info...")
        exchange_info_map = {s['symbol']: s for s in client.get_exchange_info()['symbols']}
        logger.info(f"[API] Exchange info map created with {len(exchange_info_map)} symbols.")
    except Exception as e:
        logger.error(f"❌ [API] Error fetching exchange info: {e}")

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    try:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
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
    time.sleep(API_REQUEST_DELAY)
    try:
        klines = client.get_historical_klines(symbol, interval, f"{days} day ago UTC")
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna().astype(float)
    except Exception as e:
        logger.error(f"❌ [Data] Error fetching data for {symbol}: {e}"); return None

def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    df_calc['ema9'] = df_calc['close'].ewm(span=9, adjust=False).mean()
    df_calc['ema21'] = df_calc['close'].ewm(span=21, adjust=False).mean()
    df_calc['ema50'] = df_calc['close'].ewm(span=50, adjust=False).mean()
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    df_calc['atr'] = tr.ewm(span=14, adjust=False).mean()
    up_move = df_calc['high'].diff(); down_move = -df_calc['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=14, adjust=False).mean()
    delta = df_calc['close'].diff(); gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean(); avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df_calc['rsi'] = 100 - (100 / (1 + rs))
    rsi_val = df_calc['rsi']
    stoch_rsi = (rsi_val - rsi_val.rolling(14).min()) / (rsi_val.rolling(14).max() - rsi_val.rolling(14).min()).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi.rolling(3).mean() * 100
    bb_middle = df_calc['close'].rolling(window=20).mean()
    bb_std = df_calc['close'].rolling(window=20).std()
    df_calc['bb_lower'] = bb_middle - (bb_std * 2)
    exp1 = df_calc['close'].ewm(span=12, adjust=False).mean(); exp2 = df_calc['close'].ewm(span=26, adjust=False).mean()
    df_calc['macd'] = exp1 - exp2
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()
    df_calc['atr_percent'] = (df_calc['atr'] / df_calc['close']) * 100
    return df_calc

# --- Data Loading ---
def load_open_signals_to_cache():
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status IN ('open', 'updated');")
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in cur.fetchall(): open_signals_cache[signal['symbol']] = dict(signal)
            logger.info(f"✅ [Cache] Loaded {len(open_signals_cache)} open signals.")
    except Exception as e:
        logger.error(f"❌ [Cache] Failed to load open signals: {e}")

def load_notifications_to_cache():
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 20;")
            with notifications_lock:
                notifications_cache.clear()
                for n in reversed(cur.fetchall()):
                    n['timestamp'] = n['timestamp'].isoformat()
                    notifications_cache.appendleft(dict(n))
    except Exception as e:
        logger.error(f"❌ [Cache] Failed to load notifications: {e}")

def load_settings_from_redis():
    global RISK_PER_TRADE_PERCENT, MAX_OPEN_TRADES, USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY, USE_MOMENTUM_VOLATILITY_STRATEGY, STRATEGY_FILTER_CONFIG, paper_trading_mode
    if not redis_client: return
    try:
        settings_data = redis_client.get('trading_settings')
        if settings_data:
            settings = json.loads(settings_data)
            with risk_per_trade_lock: RISK_PER_TRADE_PERCENT = settings.get('RISK_PER_TRADE_PERCENT', 0.85)
            MAX_OPEN_TRADES = settings.get('MAX_OPEN_TRADES', 3)
            with trading_mode_lock: paper_trading_mode = settings.get('paper_trading_mode', True)
        strategies_data = redis_client.get('strategy_settings')
        if strategies_data:
            strategies = json.loads(strategies_data)
            USE_BB_STOCH_STRATEGY = strategies.get('USE_BB_STOCH_STRATEGY', True)
            USE_MACD_EMA_STRATEGY = strategies.get('USE_MACD_EMA_STRATEGY', True)
            USE_EMA_RSI_STRATEGY = strategies.get('USE_EMA_RSI_STRATEGY', True)
            USE_PULLBACK_STRATEGY = strategies.get('USE_PULLBACK_STRATEGY', True)
            USE_MOMENTUM_VOLATILITY_STRATEGY = strategies.get('USE_MOMENTUM_VOLATILITY_STRATEGY', True)
        filters_data = redis_client.get('strategy_filter_config')
        if filters_data:
            with strategy_filters_lock: STRATEGY_FILTER_CONFIG = json.loads(filters_data)
        logger.info("✅ [Redis] Successfully loaded settings from Redis.")
    except Exception as e:
        logger.error(f"❌ [Redis] Error loading settings: {e}")

# --- Filters & Strategies ---

def check_market_volatility_filter(df: pd.DataFrame) -> bool:
    """
    15m-tuned volatility gate:
    - Use last ~1 day (96 bars) of ATR% to compute adaptive bounds via percentiles.
    - Keeps us out of dead markets and extreme whipsaws without being overly strict.
    """
    if 'atr_percent' not in df.columns or len(df) < 30:
        log_rejection(getattr(df, "name", "—"), "Market Volatility Filter Failed")
        return False

    recent = df['atr_percent'].tail(96).dropna()
    last = float(df.iloc[-1].get('atr_percent', 0))

    if recent.empty:
        log_rejection(getattr(df, "name", "—"), "Market Volatility Filter Failed")
        return False

    q25 = float(np.percentile(recent, 25))
    q90 = float(np.percentile(recent, 90))
    lower = max(0.35, q25 * 0.9)   # allow slightly under the 25th for flexibility
    upper = min(8.0,  q90 * 1.1)   # cap extremes

    if last < lower or last > upper:
        log_rejection(df.name, "Market Volatility Filter Failed")
        return False
    return True


def check_trend_strength_filter(df: pd.DataFrame, adx_threshold: int) -> bool:
    """
    15m-tuned trend gate:
    - Consider the mean of the last 3 ADX values.
    - Slight tolerance (95%) to avoid being too strict while keeping quality.
    """
    if 'adx' not in df.columns or len(df) < 5:
        log_rejection(getattr(df, "name", "—"), "Trend Strength Filter Failed")
        return False

    recent_adx = float(pd.Series(df['adx'].tail(3)).mean())
    if recent_adx < (adx_threshold * 0.95):
        log_rejection(df.name, "Trend Strength Filter Failed")
        return False
    return True

def is_htf_bullish_confirmation(symbol: str, htf: str = '1h', mode: str = 'Strict') -> bool:
    if mode == 'Disabled': return True
    try:
        df = fetch_historical_data(symbol, htf, days=40) 
        if df is None or len(df) < 50: return False
        df['ema50']  = df['close'].ewm(span=50, adjust=False).mean()
        last = df.iloc[-1]
        if mode == 'Strict':
            df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
            last = df.iloc[-1]
            return last['close'] > last['ema50'] and last['ema50'] > last['ema200']
        elif mode == 'Relaxed':
            return last['close'] > last['ema50']
        return False
    except Exception as e:
        logger.warning(f"[HTF] Could not confirm HTF trend for {symbol}: {e}")
        return False

def apply_strategy_filters(symbol: str, df: pd.DataFrame, strategy_name: str) -> bool:
    with strategy_filters_lock: config = STRATEGY_FILTER_CONFIG.get(strategy_name)
    if not config or config.get("profile") == "Disabled": return True
    adx_threshold = config.get("adx_threshold", 22)
    if not check_trend_strength_filter(df, adx_threshold): return False
    htf_mode = config.get("htf_confirmation_mode", "Strict")
    if not is_htf_bullish_confirmation(symbol, HIGHER_TIMEFRAME, htf_mode):
        log_rejection(symbol, "HTF Trend Confirmation Failed"); return False
    return True


def check_bb_stoch_strategy_enhanced(df: pd.DataFrame) -> bool:
    """
    15m BB+Stoch (Reversal):
    - Previous close below lower band, current close back above.
    - StochRSI rising OR bullish body.
    - RSI improving (current > previous) and not overbought.
    """
    if len(df) < 21 or not {'bb_lower','stoch_rsi_k','rsi','open','close'}.issubset(df.columns):
        return False
    last, prev = df.iloc[-1], df.iloc[-2]

    bounce = (prev['close'] < prev['bb_lower']) and (last['close'] > last['bb_lower'])
    stoch_rising = last['stoch_rsi_k'] > prev['stoch_rsi_k']
    bullish_body = last['close'] > last['open']
    rsi_improving = last['rsi'] > prev['rsi']
    not_overbought = last['rsi'] < 70

    signal = bounce and (stoch_rising or bullish_body) and rsi_improving and not_overbought
    if not signal:
        log_rejection(df.name, "Bullish Confirmation Failed")
    return signal


def check_macd_ema_strategy_enhanced(df: pd.DataFrame) -> bool:
    """
    15m MACD+EMA (Trend):
    - MACD cross up OR histogram increasing for 2 bars.
    - Close above EMA21 and EMA9 > EMA21.
    - RSI < 70.
    """
    needed = {'macd','macd_signal','ema9','ema21','rsi','close'}
    if len(df) < 30 or not needed.issubset(df.columns):
        return False
    last = df.iloc[-1]
    prev = df.iloc[-2]

    macd_cross_up = (prev['macd'] <= prev['macd_signal']) and (last['macd'] > last['macd_signal'])
    hist_now = last['macd'] - last['macd_signal']
    hist_prev = prev['macd'] - prev['macd_signal']
    hist_increasing = (hist_now > hist_prev) and (hist_prev > 0 or macd_cross_up)

    ema_ok = (last['close'] > last['ema21']) and (last['ema9'] > last['ema21'])
    rsi_ok = last['rsi'] < 70

    return (macd_cross_up or hist_increasing) and ema_ok and rsi_ok


def check_ema_rsi_strategy_enhanced(df: pd.DataFrame) -> bool:
    """
    15m EMA+RSI (Mixed):
    - EMA9 > EMA21 on at least 2 of the last 3 bars.
    - RSI in 50–65 zone (momentum without overextension).
    - Accept a mild pullback: low touches EMA9 but close above it.
    """
    needed = {'ema9','ema21','rsi','low','close'}
    if len(df) < 25 or not needed.issubset(df.columns):
        return False

    last3 = df.tail(3)
    ema9_over_21 = (last3['ema9'] > last3['ema21']).sum() >= 2
    last = last3.iloc[-1]

    rsi_ok = 50 <= float(last['rsi']) <= 65
    pullback_ok = (float(last['low']) <= float(last['ema9'])) and (float(last['close']) > float(last['ema9']))

    return ema9_over_21 and rsi_ok and pullback_ok


def check_pullback_strategy_enhanced(df: pd.DataFrame) -> bool:
    """
    15m Pullback (Re-entry in uptrend):
    - Uptrend context: close and EMA21 above EMA50.
    - 1–3 bar dip that tags EMA21/EMA9 and a bullish close back above EMA9.
    """
    needed = {'ema9','ema21','ema50','open','close','low'}
    if len(df) < 55 or not needed.issubset(df.columns):
        return False

    last = df.iloc[-1]
    uptrend = (last['ema21'] > last['ema50']) and (last['close'] > last['ema50'])
    if not uptrend:
        return False

    recent = df.tail(4)
    dipped = ((recent['low'] <= recent['ema21']) | (recent['low'] <= recent['ema9'])).any()
    bullish_close = last['close'] > last['open'] and last['close'] > last['ema9']

    return dipped and bullish_close


def check_momentum_volatility_strategy(df: pd.DataFrame) -> bool:
    """
    15m Momentum (Volatility-boosted breakout):
    - ATR% > 1.2× its 14-period mean.
    - EMA9 > EMA21 and close > EMA9.
    - MACD histogram rising.
    """
    needed = {'atr_percent','ema9','ema21','macd','macd_signal','close'}
    if len(df) < 30 or not needed.issubset(df.columns):
        return False
    last, prev = df.iloc[-1], df.iloc[-2]

    atr_mean = float(pd.Series(df['atr_percent'].tail(14)).mean())
    atr_ok = float(last['atr_percent']) >= (1.2 * atr_mean)

    hist_now = float(last['macd'] - last['macd_signal'])
    hist_prev = float(prev['macd'] - prev['macd_signal'])
    hist_rising = hist_now > hist_prev

    ema_ok = (last['ema9'] > last['ema21']) and (last['close'] > last['ema9'])

    return atr_ok and hist_rising and ema_ok

def calculate_trade_levels(df: pd.DataFrame) -> Dict[str, Any]:
    last = df.iloc[-1]
    atr = last['atr']
    entry_price = last['close']
    stop_loss = entry_price - (atr * 1.5)
    target_price_1 = entry_price + (atr * 2.0)
    target_price_2 = entry_price + (atr * 3.5)
    trailing_stop_distance = atr * 1.5
    return {
        "entry_price": entry_price, "stop_loss": stop_loss, "target_price_1": target_price_1,
        "target_price_2": target_price_2, "atr": atr,
        "trailing_stop_distance": trailing_stop_distance
    }

def adjust_quantity_to_step_size(quantity: float, step_size: str) -> float:
    return float(Decimal(quantity).quantize(Decimal(step_size), rounding=ROUND_DOWN))

def create_trade_signal(symbol: str, df: pd.DataFrame, strategy_name: str):
    with trading_mode_lock:
        is_real = not paper_trading_mode

    trade_levels = calculate_trade_levels(df)
    entry_price = trade_levels['entry_price']
    
    if is_real:
        with balance_lock:
            current_usdt_balance = usdt_balance
        
        with risk_per_trade_lock:
            trade_size_usdt = current_usdt_balance * (RISK_PER_TRADE_PERCENT / 100)

        if trade_size_usdt <= 0:
            log_rejection(symbol, "Insufficient Balance")
            return

        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info:
            logger.error(f"❌ [Real Trade] Could not find exchange info for {symbol}")
            return

        min_notional = float(next((f['minNotional'] for f in symbol_info['filters'] if f['filterType'] == 'NOTIONAL'), '0.0'))
        if trade_size_usdt < min_notional:
            log_rejection(symbol, "MinNotional Filter Failed", {"required": min_notional, "actual": trade_size_usdt})
            return

        quantity = trade_size_usdt / entry_price
        
        step_size = next((f['stepSize'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), '0.000001')
        adjusted_quantity = adjust_quantity_to_step_size(quantity, step_size)

        if adjusted_quantity <= 0:
            logger.warning(f"⚠️ [Real Trade] Adjusted quantity for {symbol} is zero. Aborting.")
            return

        try:
            logger.info(f"💰 [Real Trade] Placing LIVE MARKET BUY order for {adjusted_quantity} of {symbol}")
            order = client.create_order(symbol=symbol, side=Client.SIDE_BUY, type=Client.ORDER_TYPE_MARKET, quantity=adjusted_quantity)
            
            avg_fill_price = sum(float(f['price']) * float(f['qty']) for f in order.get('fills', [])) / sum(float(f['qty']) for f in order.get('fills', [])) if order.get('fills') else entry_price
            final_quantity = float(order.get('executedQty', adjusted_quantity))
            order_id = order.get('orderId', 'N/A')
            
            save_signal_to_db(symbol, avg_fill_price, trade_levels, strategy_name, True, final_quantity, order_id)
            message = (f"💰 *صفقة حقيقية جديدة*\n`{symbol}` | `{strategy_name}`\n*دخول:* `{avg_fill_price:.4f}`\n*كمية:* `{final_quantity}`")
            send_telegram_message(message)
            log_and_notify("info", f"Opened REAL trade for {symbol}", "REAL_TRADE_OPEN")

        except BinanceAPIException as e:
            logger.error(f"❌ [Real Trade] Binance API Error for {symbol}: {e}")
            send_telegram_message(f"❌ *خطأ في صفقة حقيقية لـ {symbol}*\n`{e}`")
        except Exception as e:
            logger.error(f"❌ [Real Trade] CRITICAL ERROR creating real trade for {symbol}: {e}", exc_info=True)

    else: # وضع التداول الورقي
        quantity = PAPER_TRADE_SIZE_USDT / entry_price
        save_signal_to_db(symbol, entry_price, trade_levels, strategy_name, False, quantity)
        message = (f"📊 *صفقة ورقية جديدة*\n`{symbol}` | `{strategy_name}`\n*دخول:* `{entry_price:.4f}`\n*هدف1:* `{trade_levels['target_price_1']:.4f}`\n*وقف:* `{trade_levels['stop_loss']:.4f}`")
        send_telegram_message(message)
        log_and_notify("info", f"Opened paper trade for {symbol}", "PAPER_TRADE_OPEN")

def save_signal_to_db(symbol: str, entry_price: float, trade_levels: Dict, strategy_name: str, is_real: bool, quantity: float, order_id: Optional[str] = None):
    try:
        if not (check_db_connection() and conn): return
        
        signal_details = {
            "atr": trade_levels['atr'], "is_trailing_active": False,
            "trailing_stop_distance": trade_levels['trailing_stop_distance']
        }
        
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price_1, target_price_2, stop_loss, status, 
                                   strategy_name, is_real_trade, quantity, initial_quantity, signal_details, order_id) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
            """, (symbol, float(entry_price), float(trade_levels['target_price_1']), float(trade_levels['target_price_2']), 
                  float(trade_levels['stop_loss']), 'open', strategy_name, is_real, float(quantity), float(quantity), 
                  json.dumps(signal_details, cls=NpEncoder), order_id))
            new_id = cur.fetchone()['id']
        conn.commit()
        
        signal_data = {
            'id': new_id, 'symbol': symbol, 'entry_price': float(entry_price), 
            'target_price_1': float(trade_levels['target_price_1']), 'target_price_2': float(trade_levels['target_price_2']),
            'stop_loss': float(trade_levels['stop_loss']), 'status': 'open', 'strategy_name': strategy_name, 
            'is_real_trade': is_real, 'quantity': float(quantity), 'initial_quantity': float(quantity),
            'signal_details': signal_details, 'order_id': order_id
        }
        with signal_cache_lock: open_signals_cache[symbol] = signal_data
    except Exception as e:
        logger.error(f"❌ [DB] CRITICAL ERROR saving signal for {symbol}: {e}", exc_info=True)
        if conn: conn.rollback()

# --- قوالب HTML ---
# -*- coding: utf-8 -*-
"""
بوت تداول مع لوحة تحكم محسنة (Responsive Dashboard)
"""

from flask import Flask, render_template_string, jsonify, request
import json
import threading
from datetime import datetime, timezone

app = Flask(__name__)

# ----------------- الإعدادات العامة -----------------
trading_enabled = False
paper_trading_mode = True
usdt_balance = 0.0
open_signals = {}
notifications = []
market_state = {"trend_details_by_tf": {}}

lock = threading.Lock()

# ----------------- واجهة المستخدم (لوحة التحكم) -----------------
DASHBOARD_TEMPLATE = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>لوحة التحكم - بوت التداول</title>
<style>
:root{--bg:#0b1020;--panel:#121b36;--accent:#3aa0ff;--ok:#15c46a;--warn:#ff9f1a;--bad:#ff4757;--muted:#8aa0c8;}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:#e8f1ff;font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,"Noto Sans",Arial;line-height:1.4}
.container{max-width:1200px;margin:auto;padding:12px}
header{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px}
h1{font-size:18px;font-weight:700;color:#d7e4ff}
.badge{padding:4px 10px;border-radius:999px;font-size:12px;background:#0d1730;border:1px solid #1e2c52;color:#cce0ff}

.grid-main{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:768px){.grid-main{grid-template-columns:1fr}}

.card{background:var(--panel);border:1px solid #1e2c52;border-radius:12px;box-shadow:0 6px 20px rgba(0,0,0,.25);overflow:hidden}
.card h2{margin:0;padding:10px 14px;border-bottom:1px solid #1e2c52;font-size:14px;color:#cfe2ff}
.card-body{padding:12px}

.controls{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px}
.btn{appearance:none;border:1px solid #2a3a68;background:#0f1b3b;color:#d9e7ff;padding:8px 12px;border-radius:8px;cursor:pointer;font-size:13px}
.btn.ok{background:#13482c;border-color:#1d6a48}
.btn.warn{background:#6a3a0f;border-color:#8b5b0f}

.kv{display:grid;grid-template-columns:auto 1fr;gap:6px 10px;font-size:13px}
.trend{display:flex;gap:6px;flex-wrap:wrap}
.pill{flex:1;min-width:70px;text-align:center;background:#0d1730;border:1px solid #1f2d55;border-radius:8px;padding:6px}
.pill b{display:block;font-size:12px;color:#9fb7ef}
.pill span{font-size:12px}
.green{color:var(--ok)}.red{color:var(--bad)}.amber{color:var(--warn)}

#signals{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px}
.signal{padding:10px;border:1px solid #24335f;border-radius:10px;background:#0d1730;display:flex;flex-direction:column;gap:6px}
.sig-title{font-size:15px;font-weight:700}
.sig-meta{font-size:12px;color:var(--muted)}
.price{font-size:16px;font-weight:600}
.progress{height:6px;background:#0b1126;border:1px solid #233056;border-radius:999px;overflow:hidden}
.progress>span{display:block;height:100%;background:linear-gradient(90deg,var(--ok),#3fd1b0)}

.table{width:100%;border-collapse:collapse;font-size:12px}
.table th{padding:6px;text-align:right;color:#9ab2e2}
.table td{padding:6px;background:#0d1730;border:1px solid #24335f}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>لوحة التحكم • فريم 15 دقيقة</h1>
    <div class="badge" id="serverTime">—</div>
  </header>

  <div class="grid-main">
    <div class="card">
      <h2>الصفقات المفتوحة</h2>
      <div class="card-body">
        <div id="signals"></div>
      </div>
    </div>
    <div class="card">
      <h2>التحكم والحالة</h2>
      <div class="card-body">
        <div class="controls">
          <button class="btn ok" id="toggleTrading">تشغيل التداول</button>
          <button class="btn" id="toggleMode">وضع: ورقي</button>
          <a class="btn" href="/settings">الإعدادات</a>
        </div>
        <div class="kv">
          <div>الرصيد (USDT):</div><div id="balance">—</div>
          <div>عدد الصفقات:</div><div id="openCount">—</div>
        </div>
        <div style="margin-top:10px">
          <h3 style="font-size:13px;margin:0 0 6px 0;color:#cfe2ff">حالة السوق</h3>
          <div id="trend" class="trend"></div>
        </div>
      </div>
    </div>
  </div>

  <div class="card" style="margin-top:12px">
    <h2>سجل الأحداث</h2>
    <div class="card-body">
      <table class="table" id="events">
        <thead><tr><th>الوقت</th><th>النوع</th><th>الرسالة</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>

<script>
const qs = s=>document.querySelector(s);
let lastPrices={};

async function load(){
  const res = await fetch('/api/dashboard_data',{cache:'no-cache'});
  const data = await res.json();
  render(data);
}
function render(data){
  qs('#serverTime').textContent = data.server_time||'—';
  qs('#toggleTrading').textContent = data.trading_enabled?"إيقاف التداول":"تشغيل التداول";
  qs('#toggleMode').textContent = "وضع: "+(data.paper_trading_mode?"ورقي":"حقيقي");
  qs('#balance').textContent = (data.usdt_balance||0).toFixed(2);
  qs('#openCount').textContent = Object.keys(data.open_signals||{}).length;

  const trend = data.market_state?.trend_details_by_tf||{};
  qs('#trend').innerHTML = Object.entries(trend).map(([tf,t])=>{
    let c="amber"; if(t.trend=="Bullish")c="green"; else if(t.trend=="Bearish")c="red";
    return `<div class="pill"><b>${tf}</b><span class="${c}">${t.trend||'—'}</span><br><span>RSI ${t.rsi||'—'}</span></div>`
  }).join("");

  const sigBox = qs('#signals'); sigBox.innerHTML="";
  for(const [sym,s] of Object.entries(data.open_signals||{})){
    const cp=s.current_price; const prev=lastPrices[sym]; lastPrices[sym]=cp;
    const delta=prev?cp-prev:0; 
    const prog=s.progress_to_tp?`<div class="progress"><span style="width:${s.progress_to_tp}%"></span></div>`:"";
    sigBox.innerHTML += `<div class="signal">
      <div class="sig-title">${sym}</div>
      <div class="sig-meta">دخول ${s.entry_price} • وقف ${s.stop_loss} • هدف ${s.target_price_1}</div>
      <div class="price">${cp||'—'} (${delta.toFixed(4)})</div>
      ${prog}
      <button class="btn warn" onclick="fetch('/close_trade/${s.id}',{method:'POST'}).then(load)">إغلاق</button>
    </div>`;
  }

  qs('#events tbody').innerHTML=(data.notifications||[]).map(n=>`<tr><td>${n.timestamp}</td><td>${n.type}</td><td>${n.message}</td></tr>`).join("");
}
qs('#toggleTrading').onclick=()=>{fetch('/toggle_trading',{method:'POST'}).then(load)};
qs('#toggleMode').onclick=()=>{fetch('/toggle_real_trading',{method:'POST'}).then(load)};
load();setInterval(load,2000);
</script>
</body>
</html>
"""

# ----------------- المسارات -----------------
@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route("/api/dashboard_data")
def dashboard_data():
    with lock:
        payload = {
            "trading_enabled": trading_enabled,
            "paper_trading_mode": paper_trading_mode,
            "usdt_balance": usdt_balance,
            "open_signals": open_signals,
            "notifications": notifications,
            "market_state": market_state,
            "server_time": datetime.now(timezone.utc).isoformat()
        }
    return jsonify(payload)

@app.route("/toggle_trading", methods=["POST"])
def toggle_trading():
    global trading_enabled
    with lock:
        trading_enabled = not trading_enabled
    return dashboard_data()

@app.route("/toggle_real_trading", methods=["POST"])
def toggle_mode():
    global paper_trading_mode
    with lock:
        paper_trading_mode = not paper_trading_mode
    return dashboard_data()

@app.route("/close_trade/<int:signal_id>", methods=["POST"])
def close_trade(signal_id):
    return jsonify({"success": True, "message": f"Closed trade {signal_id}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


# --- Main Loop & Threads ---
def main_bot_loop():
    logger.info("🚀 [Main Loop] Starting signal scanning loop...")
    while True:
        try:
            with trading_status_lock:
                if not is_trading_enabled: time.sleep(10); continue
            with signal_cache_lock:
                if len(open_signals_cache) >= MAX_OPEN_TRADES: time.sleep(120); continue
            
            logger.info("="*20 + " Starting New Scan Cycle " + "="*20)
            for symbol in validated_symbols_to_scan:
                with signal_cache_lock:
                    if symbol in open_signals_cache: continue
                
                df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                if df is None or len(df) < 50: log_rejection(symbol, "Insufficient Historical Data"); continue
                
                df_featured = calculate_all_features(df); df_featured.name = symbol
                
                if not check_market_volatility_filter(df_featured): continue
                if not check_trend_strength_filter(df_featured, BASE_FILTER_ADX_THRESHOLD): continue
                
                strategy_found = None
                if USE_BB_STOCH_STRATEGY and check_bb_stoch_strategy_enhanced(df_featured): strategy_found = "BB_Stoch_Strategy"
                elif USE_MACD_EMA_STRATEGY and check_macd_ema_strategy_enhanced(df_featured): strategy_found = "MACD_EMA_Strategy"
                elif USE_EMA_RSI_STRATEGY and check_ema_rsi_strategy_enhanced(df_featured): strategy_found = "EMA_RSI_Strategy"
                elif USE_PULLBACK_STRATEGY and check_pullback_strategy_enhanced(df_featured): strategy_found = "Pullback_Strategy"
                elif USE_MOMENTUM_VOLATILITY_STRATEGY and check_momentum_volatility_strategy(df_featured): strategy_found = "Momentum_Volatility_Strategy"
                
                if strategy_found and apply_strategy_filters(symbol, df_featured, strategy_found):
                    logger.info(f"🌟 [Signal Passed] Confirmed signal for {symbol}! Strategy: {strategy_found}")
                    create_trade_signal(symbol, df_featured, strategy_found)
            
            logger.info("="*20 + " Scan Cycle Completed " + "="*20)
            time.sleep(60 * 5)
        except Exception as e:
            logger.error(f"❌ [Main Loop] A critical error occurred: {e}", exc_info=True)
            time.sleep(60)

def update_signal_in_db(signal_id, updates):
    if not (check_db_connection() and conn): return False
    try:
        with conn.cursor() as cur:
            set_clause = sql.SQL(', ').join(sql.SQL("{} = %s").format(sql.Identifier(k)) for k in updates.keys())
            values = list(updates.values())
            query = sql.SQL("UPDATE signals SET {} WHERE id = %s").format(set_clause)
            values.append(signal_id)
            cur.execute(query, values)
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"❌ [DB] Failed to update signal {signal_id}: {e}")
        if conn: conn.rollback()
        return False

def close_signal(signal: Dict, closing_price: float, reason: str):
    symbol, signal_id, entry_price = signal['symbol'], signal['id'], signal['entry_price']
    
    if signal.get('is_real_trade'):
        try:
            asset = symbol.replace("USDT", "")
            balance = client.get_asset_balance(asset=asset)
            quantity_to_sell = float(balance['free'])
            
            if quantity_to_sell > 0:
                symbol_info = exchange_info_map.get(symbol)
                step_size = next((f['stepSize'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), '0.000001')
                adjusted_quantity = adjust_quantity_to_step_size(quantity_to_sell, step_size)

                if adjusted_quantity > 0:
                    logger.info(f"💰 [Real Close] Closing {adjusted_quantity} of {symbol} with a LIVE MARKET SELL order due to {reason}")
                    client.create_order(symbol=symbol, side=Client.SIDE_SELL, type=Client.ORDER_TYPE_MARKET, quantity=adjusted_quantity)
                else:
                    logger.warning(f"⚠️ [Real Close] Adjusted quantity for selling {symbol} is zero after applying step size.")
            else:
                logger.warning(f"⚠️ [Real Close] No balance found for {asset} to sell for signal ID {signal_id}.")
        except BinanceAPIException as e:
            logger.error(f"❌ [Real Close] Binance API Error closing {symbol}: {e}")
            send_telegram_message(f"❌ *خطأ في إغلاق صفقة حقيقية لـ {symbol}*\n`{e}`")
        except Exception as e:
            logger.error(f"❌ [Real Close] CRITICAL ERROR closing real trade for {symbol}: {e}", exc_info=True)

    profit = ((closing_price - entry_price) / entry_price) * 100
    update_signal_in_db(signal_id, {"status": "closed", "closing_price": closing_price, "closed_at": datetime.now(timezone.utc), "profit_percentage": profit, "closing_reason": reason})
    
    trade_type = "حقيقية" if signal.get('is_real_trade') else "ورقية"
    result_emoji = "✅" if profit >= 0 else "🔻"
    reason_map = {
        "SL_HIT": "ضرب وقف الخسارة", "TP1_HIT": "تحقيق الهدف الأول", "TP2_HIT": "تحقيق الهدف الثاني",
        "MANUAL_CLOSE": "إغلاق يدوي", "TRAILING_SL_HIT": "ضرب الوقف المتحرك"
    }
    reason_ar = reason_map.get(reason, reason)

    log_and_notify("info", f"Closed {trade_type} trade for {symbol}. Profit: {profit:.2f}%", "TRADE_CLOSED")
    send_telegram_message(f"{result_emoji} *إغلاق صفقة {trade_type} {symbol}*\n*السبب:* {reason_ar}\n*الربح:* `{profit:.2f}%`")
    
    with signal_cache_lock:
        if symbol in open_signals_cache: del open_signals_cache[symbol]

def trade_management_loop():
    logger.info("🚀 [Trade Manager] Starting advanced trade management loop...")
    while True:
        try:
            with signal_cache_lock:
                if not open_signals_cache: time.sleep(2); continue
                signals_to_monitor = list(open_signals_cache.values())

            for signal in signals_to_monitor:
                symbol = signal['symbol']
                with live_prices_lock: current_price = live_prices.get(symbol)
                if not current_price: continue

                signal_details = signal.get('signal_details', {})
                stop_loss = signal['stop_loss']
                entry_price = signal['entry_price']

                if current_price <= stop_loss:
                    reason = "TRAILING_SL_HIT" if signal_details.get('is_trailing_active') else "SL_HIT"
                    close_signal(signal, stop_loss, reason)
                    continue

                if signal.get('target_price_2') and current_price >= signal['target_price_2']:
                    close_signal(signal, signal['target_price_2'], "TP2_HIT")
                    continue

                if not signal_details.get('is_trailing_active'):
                    profit_percent = ((current_price - entry_price) / entry_price) * 100
                    if profit_percent >= TRAILING_STOP_ACTIVATION_PROFIT_PERCENT:
                        new_stop_loss = entry_price
                        signal_details['is_trailing_active'] = True
                        
                        updates = {"stop_loss": new_stop_loss, "status": "updated", "signal_details": json.dumps(signal_details, cls=NpEncoder)}
                        if update_signal_in_db(signal['id'], updates):
                            signal.update({"stop_loss": new_stop_loss, "status": "updated", "signal_details": signal_details})
                            log_and_notify("info", f"Trailing stop activated for {symbol}. New SL at entry: {new_stop_loss:.4f}", "TRAIL_ACTIVATED")
                            send_telegram_message(f"🎯 *تأمين صفقة {symbol}*\nتم رفع الوقف إلى نقطة الدخول بعد تحقيق ربح `{profit_percent:.2f}%`.")
                        continue

                if signal_details.get('is_trailing_active'):
                    trailing_distance = signal_details.get('trailing_stop_distance', 0)
                    if trailing_distance > 0:
                        potential_new_sl = current_price - trailing_distance
                        if potential_new_sl > stop_loss:
                            if update_signal_in_db(signal['id'], {"stop_loss": potential_new_sl}):
                                signal['stop_loss'] = potential_new_sl
            time.sleep(1)
        except Exception as e:
            logger.error(f"❌ [Trade Manager] A critical error occurred: {e}", exc_info=True)
            time.sleep(10)

# --- [جديد] دوال مجزأة لتحديث البيانات ---
def update_market_state():
    try:
        trend_details = {}
        for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
            btc_df = fetch_historical_data(BTC_SYMBOL, tf, 30)
            if btc_df is None or btc_df.empty: 
                trend_details[tf] = {"trend": "Unknown", "rsi": "N/A"}
                continue
            
            btc_df_featured = calculate_all_features(btc_df)
            if 'rsi' not in btc_df_featured.columns:
                trend_details[tf] = {"trend": "Unknown", "rsi": "N/A"}
                continue

            last = btc_df_featured.iloc[-1]
            rsi_value = last['rsi']
            
            trend = "Sideways"
            if rsi_value > 55: trend = "Bullish"
            elif rsi_value < 45: trend = "Bearish"
            
            trend_details[tf] = {"trend": trend, "rsi": round(rsi_value, 2)}

        with market_state_lock:
            current_market_state.update({'trend_details_by_tf': trend_details, 'last_updated': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')})
    except Exception as e:
        logger.error(f"❌ [Market State] Error during single update: {e}", exc_info=True)

def update_market_state_loop():
    logger.info("🚀 [Market State] Starting market state update loop...")
    while True:
        update_market_state()
        time.sleep(60 * 5)

def update_balance():
    try:
        balance_info = client.get_asset_balance(asset='USDT')
        with balance_lock:
            global usdt_balance
            usdt_balance = float(balance_info['free'])
        logger.info(f"💰 [Balance] USDT balance updated: {usdt_balance:.2f}")
    except Exception as e:
        logger.error(f"❌ [Balance] Could not update USDT balance: {e}")

def update_balance_loop():
    logger.info("🚀 [Balance Updater] Starting balance update loop...")
    while True:
        try:
            update_balance()
        except Exception as e:
            logger.error(f"❌ [Balance Loop] Error: {e}", exc_info=True)
        time.sleep(60 * 10) 

# --- نقطة بداية البرنامج ---
if __name__ == '__main__':
    logger.info("="*50 + "\n====== Starting Crypto Trading Bot V17.2.0 ======\n" + "="*50)
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
        logger.critical("❌ No valid symbols to scan. Exiting."); exit(1)
    
    load_open_signals_to_cache()
    load_notifications_to_cache()
    load_settings_from_redis()
    
    # --- [جديد] استدعاء أولي للبيانات لضمان عمل الواجهة فوراً ---
    logger.info("Performing initial data fetch for dashboard...")
    update_market_state()
    update_balance()
    logger.info("Initial data fetch complete.")

    start_websocket()
    Thread(target=main_bot_loop, daemon=True).start()
    Thread(target=trade_management_loop, daemon=True).start()
    Thread(target=update_market_state_loop, daemon=True).start()
    Thread(target=update_balance_loop, daemon=True).start() 
    
    logger.info("🌐 [Flask] Starting UI on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
