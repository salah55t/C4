# ملف c4.py - نسخة V17.0.0 (تحسينات استراتيجية وإدارة الصفقات)
# --- وصف الإصدار:
# هذا الإصدار يضيف تحسينات جوهرية على منطق التداول وإدارة المخاطر.
# 1.  [جديد] إضافة شرط تأكيد شمعة صعودية (Bullish Candle) لاستراتيجيات الانعكاس لزيادة دقة الإشارات.
# 2.  [جديد] تفعيل وقف الخسارة المتحرك (Trailing Stop) تلقائياً عند تحقيق ربح +1.4% وتأمين الصفقة عند نقطة الدخول.
# 3.  [جديد] إضافة إشعارات تليجرام مفصلة عند إغلاق الصفقات (لجميع الأسباب: ربح، خسارة، يدوي).
# 4.  [مكتمل] الحفاظ على لوحة التحكم اللحظية التي تحدث البيانات كل ثانيتين عبر API.
# 5.  [مكتمل] زر الإغلاق اليدوي للصفقات.
# 6.  [مكتمل] منطق التداول الحقيقي والورقي مع عرض الرصيد.

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
        logging.FileHandler('crypto_bot_v17_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV17.0.0')

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
    last = df.iloc[-1]
    atr_percent = last.get('atr_percent', 0)
    if atr_percent < 0.5 or atr_percent > 7.0:
        log_rejection(df.name, "Market Volatility Filter Failed")
        return False
    return True

def check_trend_strength_filter(df: pd.DataFrame, adx_threshold: int) -> bool:
    last = df.iloc[-1]
    if last['adx'] < adx_threshold:
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
    if len(df) < 21: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    # الشرط الأساسي على الشمعة السابقة، والتأكيد على الشمعة الحالية
    signal_condition = (prev['close'] < prev['bb_lower']) and (last['close'] > last['bb_lower']) and (last['stoch_rsi_k'] < 30)
    bullish_confirmation = last['close'] > last['open']
    if signal_condition and not bullish_confirmation:
        log_rejection(df.name, "Bullish Confirmation Failed")
    return signal_condition and bullish_confirmation

def check_macd_ema_strategy_enhanced(df: pd.DataFrame) -> bool:
    if len(df) < 30: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    return (prev['macd'] < prev['macd_signal']) and (last['macd'] > last['macd_signal']) and (last['close'] > last['ema50'])

def check_ema_rsi_strategy_enhanced(df: pd.DataFrame) -> bool:
    if len(df) < 30: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    return (prev['ema9'] < prev['ema21']) and (last['ema9'] > last['ema21']) and (50 < last['rsi'] < 65)

def check_pullback_strategy_enhanced(df: pd.DataFrame) -> bool:
    if len(df) < 50: return False
    last = df.iloc[-1]
    # شرط الارتداد مع التأكيد بأن تكون شمعة الارتداد نفسها صعودية
    signal_condition = (last['close'] > last['ema50']) and (last['low'] < last['ema21']) and (last['close'] > last['ema21'])
    bullish_confirmation = last['close'] > last['open']
    if signal_condition and not bullish_confirmation:
        log_rejection(df.name, "Bullish Confirmation Failed")
    return signal_condition and bullish_confirmation

def check_momentum_volatility_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 50: return False
    last = df.iloc[-1]
    return (last['atr_percent'] > df['atr_percent'].rolling(14).mean().iloc[-1] * 1.5) and (last['close'] > last['ema9'])

# --- نظام إدارة الصفقات ---
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
            logger.info(f"💰 [Real Trade] Placing MARKET BUY order for {adjusted_quantity} of {symbol}")
            # استبدل بـ client.create_order في التداول الحقيقي
            order = client.create_test_order(symbol=symbol, side=Client.SIDE_BUY, type=Client.ORDER_TYPE_MARKET, quantity=adjusted_quantity)
            
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

    else:
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
DASHBOARD_TEMPLATE = """
<!DOCTYPE html><html dir="rtl" lang="ar"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>لوحة تحكم بوت التداول</title><link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap" rel="stylesheet"><style>:root{--bg-dark:#121212;--bg-surface:#1e1e1e;--primary:#BB86FC;--primary-variant:#3700B3;--text-light:#e0e0e0;--text-medium:#a0a0a0;--success:#4CAF50;--danger:#F44336;--warning:#FFC107;--info:#2196F3;}body{background-color:var(--bg-dark);color:var(--text-light);font-family:'Tajawal',sans-serif;}.container{max-width:1400px;margin:0 auto;padding:20px;}header{background-color:var(--bg-surface);padding:15px 25px;border-radius:12px;margin-bottom:25px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:15px;}.header-title{font-size:24px;font-weight:700;color:var(--primary);}.status-indicator{display:flex;align-items:center;gap:15px;}.status-dot{width:12px;height:12px;border-radius:50%;background-color:var(--danger);transition:background-color 0.5s ease;}.status-dot.active{background-color:var(--success);}.btn{background-color:var(--primary-variant);color:white;border:none;padding:10px 20px;border-radius:8px;cursor:pointer;text-decoration:none;font-size:14px;}.btn-small{padding:5px 10px;font-size:12px;}.btn.stop{background-color:var(--danger);}.btn.real-mode{background-color:var(--info);}.dashboard-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px;}.card{background-color:var(--bg-surface);border-radius:12px;padding:20px;display:flex;flex-direction:column;}.card-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:15px;padding-bottom:10px;border-bottom:1px solid #333;}.card-title{font-size:18px;font-weight:700;}.scrollable-content{overflow-y:auto;max-height:400px;flex-grow:1;}.item{padding:12px;border-radius:8px;margin-bottom:10px;border-left:4px solid var(--primary);background-color:#252525;}.item.real-trade-item{border-left-color:var(--info);}.item-header{display:flex;justify-content:space-between;align-items:center;}.item-title{font-weight:700;}.item-time{font-size:12px;color:var(--text-medium);}.item-content{font-size:13px;margin-top:5px;}.rejection-item{border-left-color:var(--warning);}.trend-container{display:flex;justify-content:space-around;align-items:center;padding:15px 0;}.trend-item{text-align:center;}.trend-label{font-size:14px;color:var(--text-medium);margin-bottom:8px;}.trend-status{font-size:18px;font-weight:700;padding:5px 15px;border-radius:20px;}.trend-up{color:var(--success);}.trend-down{color:var(--danger);}.trend-sideways{color:var(--danger);}.progress-bar-container{width:100%;background-color:#3c3c3c;border-radius:5px;height:10px;margin:8px 0;overflow:hidden;}.progress-bar{height:100%;transition:width 0.4s ease-in-out;}.progress-bar.profit{background-color:var(--success);}.progress-bar.loss{background-color:var(--danger);}.item-footer{display:flex;justify-content:space-between;font-size:12px;color:var(--text-medium);margin-top:4px;}.trade-mode-card{grid-column:1/-1;display:flex;justify-content:space-between;align-items:center;}.trade-mode-status{font-size:18px;}.trade-mode-status span{font-weight:700;padding:4px 12px;border-radius:8px;}.trade-mode-paper{color:var(--warning);background-color:rgba(255,193,7,0.1);}.trade-mode-real{color:var(--info);background-color:rgba(33,150,243,0.1);}.balance-display{font-size:16px;color:var(--text-medium);}.footer{text-align:center;margin-top:30px;padding:15px;color:var(--text-medium);}</style></head><body><div class="container"><header><div class="header-title">بوت التداول V17.0.0</div><div class="status-indicator"><div id="status-dot" class="status-dot"></div><span id="status-text">متوقف</span><button id="toggle-trading-btn" class="btn">تشغيل</button><a href="/settings" class="btn">الإعدادات</a></div></header><div class="dashboard-grid"><div class="card trade-mode-card"><div id="trade-mode-status" class="trade-mode-status"></div><div id="balance-display" class="balance-display"></div><button id="toggle-real-trading-btn" class="btn"></button></div><div class="card"><div class="card-header"><div class="card-title">اتجاه السوق (BTC)</div></div><div id="market-trend-container" class="trend-container"></div></div><div class="card"><div class="card-header"><div id="open-signals-title" class="card-title">الإشارات المفتوحة (0)</div></div><div id="open-signals-container" class="scrollable-content"><div style="text-align:center;color:var(--text-medium);">لا توجد إشارات مفتوحة</div></div></div><div class="card"><div class="card-header"><div class="card-title">الإشعارات</div></div><div id="notifications-container" class="scrollable-content"><div style="text-align:center;color:var(--text-medium);">لا توجد إشعارات</div></div></div><div class="card"><div class="card-header"><div class="card-title">سجل الرفض</div></div><div id="rejections-container" class="scrollable-content"><div style="text-align:center;color:var(--text-medium);">لا توجد سجلات رفض</div></div></div></div><div class="footer">بوت التداول V17.0.0</div></div><script>
function toggleTrading(){fetch('/toggle_trading',{method:'POST'}).then(res=>res.json()).then(updateUI).catch(err=>console.error('Error toggling trading:',err));}
function toggleRealTrading(){if(confirm('هل أنت متأكد من تغيير وضع التداول؟ قد يؤدي هذا إلى استخدام أموال حقيقية.')){fetch('/toggle_real_trading',{method:'POST'}).then(res=>res.json()).then(updateUI).catch(err=>console.error('Error toggling real trading:',err));}}
function closeTrade(signalId,symbol){if(confirm(`هل أنت متأكد من رغبتك في إغلاق الصفقة لـ ${symbol} يدويًا بسعر السوق؟`)){fetch(`/close_trade/${signalId}`,{method:'POST'}).then(res=>res.json()).then(data=>{alert(data.message);fetchData();}).catch(err=>{alert('حدث خطأ أثناء محاولة إغلاق الصفقة.');console.error(err);});}}

function updateUI(data){
    const statusDot=document.getElementById('status-dot');
    const statusText=document.getElementById('status-text');
    const toggleTradingBtn=document.getElementById('toggle-trading-btn');
    if(data.trading_enabled){statusDot.classList.add('active');statusText.textContent='نشط';toggleTradingBtn.textContent='إيقاف';toggleTradingBtn.classList.add('stop');}else{statusDot.classList.remove('active');statusText.textContent='متوقف';toggleTradingBtn.textContent='تشغيل';toggleTradingBtn.classList.remove('stop');}
    
    const tradeModeStatus=document.getElementById('trade-mode-status');
    const toggleRealBtn=document.getElementById('toggle-real-trading-btn');
    if(data.paper_trading_mode){tradeModeStatus.innerHTML='وضع التداول: <span class="trade-mode-paper">ورقي</span>';toggleRealBtn.textContent='تفعيل التداول الحقيقي';toggleRealBtn.className='btn real-mode';}else{tradeModeStatus.innerHTML='وضع التداول: <span class="trade-mode-real">حقيقي</span>';toggleRealBtn.textContent='العودة للتداول الورقي';toggleRealBtn.className='btn stop';}
    
    document.getElementById('balance-display').innerHTML=`الرصيد المتاح: <b>$${data.usdt_balance.toFixed(2)}</b>`;
    document.getElementById('open-signals-title').textContent=`الإشارات المفتوحة (${Object.keys(data.open_signals).length})`;
    
    const signalsContainer=document.getElementById('open-signals-container');
    signalsContainer.innerHTML='';
    if(Object.keys(data.open_signals).length>0){Object.entries(data.open_signals).forEach(([symbol,signal])=>{const isReal=signal.is_real_trade?'real-trade-item':'';const currentPrice=signal.current_price?signal.current_price.toFixed(4):'...';const progressBarHtml=signal.progress_to_tp>0?`<div class="progress-bar profit" style="width:${signal.progress_to_tp}%;"></div>`:signal.progress_to_sl>0?`<div class="progress-bar loss" style="width:${signal.progress_to_sl}%;"></div>`:'';signalsContainer.innerHTML+=`<div class="item ${isReal}"><div class="item-header"><div class="item-title">${symbol}</div><button class="btn btn-small stop" onclick="closeTrade(${signal.id},'${symbol}')">إغلاق</button></div><div class="item-content"><span>الدخول: ${signal.entry_price.toFixed(4)}</span> | <span id="price-${symbol}">الحالي: ${currentPrice}</span></div><div class="progress-bar-container">${progressBarHtml}</div><div class="item-footer"><span>الوقف: ${signal.stop_loss.toFixed(4)}</span><span>الهدف: ${signal.target_price_1.toFixed(4)}</span></div></div>`;});}else{signalsContainer.innerHTML='<div style="text-align:center;color:var(--text-medium);">لا توجد إشارات مفتوحة</div>';}
    
    const notificationsContainer=document.getElementById('notifications-container');
    notificationsContainer.innerHTML='';
    if(data.notifications.length>0){data.notifications.forEach(n=>{notificationsContainer.innerHTML+=`<div class="item"><div class="item-content">${n.message}</div></div>`;});}else{notificationsContainer.innerHTML='<div style="text-align:center;color:var(--text-medium);">لا توجد إشعارات</div>';}
    
    const rejectionsContainer=document.getElementById('rejections-container');
    rejectionsContainer.innerHTML='';
    if(data.rejections.length>0){data.rejections.forEach(r=>{rejectionsContainer.innerHTML+=`<div class="item rejection-item"><div class="item-header"><div class="item-title">${r.symbol}</div></div><div class="item-content">${r.reason}</div></div>`;});}else{rejectionsContainer.innerHTML='<div style="text-align:center;color:var(--text-medium);">لا توجد سجلات رفض</div>';}

    const marketTrendContainer = document.getElementById('market-trend-container');
    marketTrendContainer.innerHTML = '';
    if (Object.keys(data.market_state.trend_details_by_tf).length > 0) {
        Object.entries(data.market_state.trend_details_by_tf).forEach(([tf, trend_data]) => {
            let trendClass = 'trend-sideways';
            let trendText = 'متذبذب';
            if (trend_data.trend === 'Bullish') { trendClass = 'trend-up'; trendText = 'صاعد'; }
            else if (trend_data.trend === 'Bearish') { trendClass = 'trend-down'; trendText = 'هابط'; }
            marketTrendContainer.innerHTML += `<div class="trend-item"><div class="trend-label">${tf}</div><div class="trend-status ${trendClass}">${trendText}</div></div>`;
        });
    } else {
        marketTrendContainer.innerHTML = '<div style="text-align:center;color:var(--text-medium);">جاري تحميل بيانات السوق...</div>';
    }
}

async function fetchData(){try{const response=await fetch('/api/dashboard_data');const data=await response.json();updateUI(data);}catch(error){console.error('Failed to fetch dashboard data:',error);}}
document.addEventListener('DOMContentLoaded',()=>{document.getElementById('toggle-trading-btn').onclick=toggleTrading;document.getElementById('toggle-real-trading-btn').onclick=toggleRealTrading;fetchData();setInterval(fetchData,2000);});
</script></body></html>
"""
SETTINGS_TEMPLATE = """
<!DOCTYPE html><html dir="rtl" lang="ar"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>إعدادات البوت</title><link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap" rel="stylesheet"><style>:root{--bg-dark:#121212;--bg-surface:#1e1e1e;--primary:#BB86FC;--primary-variant:#3700B3;--text-light:#e0e0e0;--text-medium:#a0a0a0;}body{background-color:var(--bg-dark);color:var(--text-light);font-family:'Tajawal',sans-serif;}.container{max-width:900px;margin:0 auto;padding:20px;}header{background-color:var(--bg-surface);padding:15px 25px;border-radius:12px;margin-bottom:25px;display:flex;justify-content:space-between;align-items:center;}.header-title{font-size:24px;font-weight:700;color:var(--primary);}.btn{background-color:var(--primary-variant);color:white;border:none;padding:10px 20px;border-radius:8px;cursor:pointer;text-decoration:none;}.settings-form{background-color:var(--bg-surface);border-radius:12px;padding:25px;margin-bottom:20px;}.form-section-title{font-size:20px;font-weight:700;margin-bottom:20px;padding-bottom:10px;border-bottom:1px solid #333;}.form-group{margin-bottom:20px;}.form-group label{display:block;margin-bottom:8px;font-weight:bold;color:var(--text-medium);}.form-group input[type="number"],.form-group select{width:100%;padding:12px;border:1px solid #333;border-radius:8px;background-color:#252525;color:var(--text-light);}.checkbox-group{display:flex;align-items:center;gap:10px;padding:10px;}.filter-table{width:100%;border-collapse:collapse;}.filter-table th,.filter-table td{padding:12px;text-align:right;border-bottom:1px solid #333;}.filter-table select,.filter-table input{width:100%;padding:8px;}</style></head><body><div class="container"><header><div class="header-title">إعدادات البوت</div><a href="/" class="btn">العودة للرئيسية</a></header><div class="settings-form"><h3 class="form-section-title">إعدادات التداول العامة</h3><form id="settings-form"><div class="form-group"><label>نسبة المخاطرة للصفقة (%)</label><input type="number" name="risk_per_trade" step="0.1" value="{{RISK_PER_TRADE_PERCENT}}"></div><div class="form-group"><label>الحد الأقصى للصفقات المفتوحة</label><input type="number" name="max_trades" value="{{MAX_OPEN_TRADES}}"></div><button type="submit" class="btn">حفظ الإعدادات</button></form></div><div class="settings-form"><h3 class="form-section-title">تفعيل الاستراتيجيات</h3><form id="strategies-form"><div class="form-group checkbox-group"><input type="checkbox" id="use_bb_stoch" name="use_bb_stoch" {{'checked' if USE_BB_STOCH_STRATEGY else ''}}><label for="use_bb_stoch">BB+Stoch</label></div><div class="form-group checkbox-group"><input type="checkbox" id="use_macd_ema" name="use_macd_ema" {{'checked' if USE_MACD_EMA_STRATEGY else ''}}><label for="use_macd_ema">MACD+EMA</label></div><div class="form-group checkbox-group"><input type="checkbox" id="use_ema_rsi" name="use_ema_rsi" {{'checked' if USE_EMA_RSI_STRATEGY else ''}}><label for="use_ema_rsi">EMA+RSI</label></div><div class="form-group checkbox-group"><input type="checkbox" id="use_pullback" name="use_pullback" {{'checked' if USE_PULLBACK_STRATEGY else ''}}><label for="use_pullback">Pullback</label></div><div class="form-group checkbox-group"><input type="checkbox" id="use_momentum_volatility" name="use_momentum_volatility" {{'checked' if USE_MOMENTUM_VOLATILITY_STRATEGY else ''}}><label for="use_momentum_volatility">Momentum</label></div><button type="submit" class="btn">حفظ الاستراتيجيات</button></form></div><div class="settings-form"><h3 class="form-section-title">إعدادات فلاتر الاستراتيجيات</h3><form id="filters-form"><table class="filter-table"><thead><tr><th>الاستراتيجية</th><th>ملف تعريف الفلتر</th><th>حد ADX</th><th>تأكيد HTF</th></tr></thead><tbody>{%for key, config in STRATEGY_FILTER_CONFIG.items()%}<tr><td>{{STRATEGY_NAMES.get(key,key)}}</td><td><select name="{{key}}_profile"><option value="Strict" {{'selected' if config.profile=='Strict'}}>صارم</option><option value="Moderate" {{'selected' if config.profile=='Moderate'}}>متوسط</option><option value="Reversal" {{'selected' if config.profile=='Reversal'}}>انعكاسي</option><option value="Disabled" {{'selected' if config.profile=='Disabled'}}>معطل</option></select></td><td><input type="number" name="{{key}}_adx_threshold" value="{{config.adx_threshold}}"></td><td><select name="{{key}}_htf_confirmation_mode"><option value="Strict" {{'selected' if config.htf_confirmation_mode=='Strict'}}>صارم</option><option value="Relaxed" {{'selected' if config.htf_confirmation_mode=='Relaxed'}}>مخفف</option><option value="Disabled" {{'selected' if config.htf_confirmation_mode=='Disabled'}}>معطل</option></select></td></tr>{%endfor%}</tbody></table><button type="submit" class="btn">حفظ إعدادات الفلاتر</button></form></div></div><script>function setupForm(formId,url){document.getElementById(formId).addEventListener('submit',function(e){e.preventDefault();const formData=new FormData(this);const data=formId==='strategies-form'?Object.fromEntries([...formData.keys()].map(key=>[key,this.querySelector(`[name=${key}]`).checked])):Object.fromEntries(formData.entries());fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}).then(res=>res.json()).then(data=>alert(data.message));});}
setupForm('settings-form','/update_settings');setupForm('strategies-form','/update_strategies');setupForm('filters-form','/update_filter_settings');</script></body></html>
"""

# --- مسارات Flask ---
@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/api/dashboard_data')
def dashboard_data():
    with trading_status_lock: trading_enabled = is_trading_enabled
    with trading_mode_lock: is_paper_mode = paper_trading_mode
    with balance_lock: current_balance = usdt_balance
    with notifications_lock: notifications = list(notifications_cache)
    with rejection_logs_lock: rejections = list(rejection_logs_cache)
    with market_state_lock: market_state = dict(current_market_state)
    with live_prices_lock: live_prices_copy = dict(live_prices)

    open_signals_with_progress = {}
    with signal_cache_lock:
        sorted_symbols = sorted(open_signals_cache.keys())
        for symbol in sorted_symbols:
            signal = open_signals_cache[symbol]
            signal_data = signal.copy()
            current_price = live_prices_copy.get(symbol)
            
            signal_data['current_price'] = current_price
            signal_data['progress_to_tp'] = 0
            signal_data['progress_to_sl'] = 0

            if current_price:
                entry_price = signal.get('entry_price', 0)
                stop_loss = signal.get('stop_loss', 0)
                target_price_1 = signal.get('target_price_1', 0)

                if current_price > entry_price and target_price_1 > entry_price:
                    progress = ((current_price - entry_price) / (target_price_1 - entry_price)) * 100
                    signal_data['progress_to_tp'] = min(progress, 100)
                elif current_price < entry_price and entry_price > stop_loss:
                    progress = ((entry_price - current_price) / (entry_price - stop_loss)) * 100
                    signal_data['progress_to_sl'] = min(progress, 100)
            
            open_signals_with_progress[symbol] = signal_data

    return jsonify({
        "trading_enabled": trading_enabled,
        "paper_trading_mode": is_paper_mode,
        "usdt_balance": current_balance,
        "open_signals": open_signals_with_progress,
        "notifications": notifications,
        "rejections": rejections,
        "market_state": market_state
    })


@app.route('/settings')
def settings():
    return render_template_string(SETTINGS_TEMPLATE, 
        RISK_PER_TRADE_PERCENT=RISK_PER_TRADE_PERCENT, MAX_OPEN_TRADES=MAX_OPEN_TRADES,
        USE_BB_STOCH_STRATEGY=USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY=USE_MACD_EMA_STRATEGY,
        USE_EMA_RSI_STRATEGY=USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY=USE_PULLBACK_STRATEGY,
        USE_MOMENTUM_VOLATILITY_STRATEGY=USE_MOMENTUM_VOLATILITY_STRATEGY,
        STRATEGY_FILTER_CONFIG=STRATEGY_FILTER_CONFIG, STRATEGY_NAMES=STRATEGY_NAMES)

@app.route('/toggle_trading', methods=['POST'])
def toggle_trading():
    global is_trading_enabled
    with trading_status_lock: is_trading_enabled = not is_trading_enabled
    status_msg = "enabled" if is_trading_enabled else "disabled"
    log_and_notify("info", f"Trading has been {status_msg}.", "TRADING_STATUS")
    return dashboard_data()

@app.route('/toggle_real_trading', methods=['POST'])
def toggle_real_trading():
    global paper_trading_mode
    with trading_mode_lock:
        with trading_status_lock:
            if is_trading_enabled and not paper_trading_mode:
                log_and_notify("warning", "Cannot switch to paper mode while real trading is active. Stop the bot first.", "MODE_SWITCH_FAIL")
                return jsonify({"success": False, "message": "يجب إيقاف البوت أولاً للعودة للوضع الورقي"})
        
        paper_trading_mode = not paper_trading_mode
        mode_msg = "Paper" if paper_trading_mode else "Real"
        log_and_notify("info", f"Trading mode switched to {mode_msg}.", "TRADING_MODE_SWITCH")
        
        if redis_client:
            settings = {'RISK_PER_TRADE_PERCENT': RISK_PER_TRADE_PERCENT, 'MAX_OPEN_TRADES': MAX_OPEN_TRADES, 'paper_trading_mode': paper_trading_mode}
            redis_client.set('trading_settings', json.dumps(settings))
            
    return dashboard_data()

@app.route('/close_trade/<int:signal_id>', methods=['POST'])
def manual_close_trade(signal_id):
    with signal_cache_lock:
        signal_to_close = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)

    if not signal_to_close:
        return jsonify({"success": False, "message": "لم يتم العثور على الصفقة."}), 404

    symbol = signal_to_close['symbol']
    with live_prices_lock:
        current_price = live_prices.get(symbol)

    if not current_price:
        return jsonify({"success": False, "message": "لا يمكن الحصول على السعر الحالي للإغلاق."}), 500

    try:
        close_signal(signal_to_close, current_price, "MANUAL_CLOSE")
        return jsonify({"success": True, "message": f"تم إرسال أمر إغلاق لصفقة {symbol} بنجاح."})
    except Exception as e:
        logger.error(f"❌ [Manual Close] Error closing signal {signal_id}: {e}", exc_info=True)
        return jsonify({"success": False, "message": "حدث خطأ أثناء إغلاق الصفقة."}), 500

@app.route('/update_settings', methods=['POST'])
def update_settings():
    global RISK_PER_TRADE_PERCENT, MAX_OPEN_TRADES
    try:
        data = request.json
        with risk_per_trade_lock: RISK_PER_TRADE_PERCENT = float(data['risk_per_trade'])
        MAX_OPEN_TRADES = int(data['max_trades'])
        if redis_client: 
            with trading_mode_lock: is_paper = paper_trading_mode
            settings = {'RISK_PER_TRADE_PERCENT': RISK_PER_TRADE_PERCENT, 'MAX_OPEN_TRADES': MAX_OPEN_TRADES, 'paper_trading_mode': is_paper}
            redis_client.set('trading_settings', json.dumps(settings))
        log_and_notify("info", "Trading settings updated.", "SETTINGS_UPDATE")
        return jsonify({"success": True, "message": "تم تحديث الإعدادات العامة"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/update_strategies', methods=['POST'])
def update_strategies():
    global USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY, USE_MOMENTUM_VOLATILITY_STRATEGY
    try:
        data = request.json
        USE_BB_STOCH_STRATEGY = data.get('use_bb_stoch', False)
        USE_MACD_EMA_STRATEGY = data.get('use_macd_ema', False)
        USE_EMA_RSI_STRATEGY = data.get('use_ema_rsi', False)
        USE_PULLBACK_STRATEGY = data.get('use_pullback', False)
        USE_MOMENTUM_VOLATILITY_STRATEGY = data.get('use_momentum_volatility', False)
        if redis_client: redis_client.set('strategy_settings', json.dumps({
            'USE_BB_STOCH_STRATEGY': USE_BB_STOCH_STRATEGY, 'USE_MACD_EMA_STRATEGY': USE_MACD_EMA_STRATEGY,
            'USE_EMA_RSI_STRATEGY': USE_EMA_RSI_STRATEGY, 'USE_PULLBACK_STRATEGY': USE_PULLBACK_STRATEGY,
            'USE_MOMENTUM_VOLATILITY_STRATEGY': USE_MOMENTUM_VOLATILITY_STRATEGY
        }))
        log_and_notify("info", "Strategy settings updated.", "STRATEGY_UPDATE")
        return jsonify({"success": True, "message": "تم تحديث الاستراتيجيات"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/update_filter_settings', methods=['POST'])
def update_filter_settings():
    global STRATEGY_FILTER_CONFIG
    try:
        data = request.json
        new_config = {}
        for key in STRATEGY_FILTER_CONFIG.keys():
            new_config[key] = {
                "profile": data.get(f"{key}_profile"), "adx_threshold": int(data.get(f"{key}_adx_threshold")),
                "htf_confirmation_mode": data.get(f"{key}_htf_confirmation_mode")
            }
        with strategy_filters_lock: STRATEGY_FILTER_CONFIG = new_config
        if redis_client: redis_client.set('strategy_filter_config', json.dumps(STRATEGY_FILTER_CONFIG))
        log_and_notify("info", "Filter settings updated.", "FILTER_SETTINGS_UPDATE")
        return jsonify({"success": True, "message": "تم تحديث إعدادات الفلاتر"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

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
                    logger.info(f"💰 [Real Close] Closing {adjusted_quantity} of {symbol} due to {reason}")
                    # استبدل بـ client.create_order في التداول الحقيقي
                    client.create_test_order(symbol=symbol, side=Client.SIDE_SELL, type=Client.ORDER_TYPE_MARKET, quantity=adjusted_quantity)
                else:
                    logger.warning(f"⚠️ [Real Close] Adjusted quantity for selling {symbol} is zero.")
            else:
                logger.warning(f"⚠️ [Real Close] No balance found for {asset} to sell.")
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

                # 1. التحقق من وقف الخسارة الأساسي
                if current_price <= stop_loss:
                    reason = "TRAILING_SL_HIT" if signal_details.get('is_trailing_active') else "SL_HIT"
                    close_signal(signal, stop_loss, reason)
                    continue

                # 2. التحقق من الهدف النهائي
                if signal.get('target_price_2') and current_price >= signal['target_price_2']:
                    close_signal(signal, signal['target_price_2'], "TP2_HIT")
                    continue

                # 3. [جديد] تفعيل الوقف المتحرك عند +1.4%
                if not signal_details.get('is_trailing_active'):
                    profit_percent = ((current_price - entry_price) / entry_price) * 100
                    if profit_percent >= TRAILING_STOP_ACTIVATION_PROFIT_PERCENT:
                        new_stop_loss = entry_price  # نقل الوقف إلى نقطة الدخول
                        signal_details['is_trailing_active'] = True
                        
                        updates = {"stop_loss": new_stop_loss, "status": "updated", "signal_details": json.dumps(signal_details, cls=NpEncoder)}
                        if update_signal_in_db(signal['id'], updates):
                            signal.update({"stop_loss": new_stop_loss, "status": "updated", "signal_details": signal_details})
                            log_and_notify("info", f"Trailing stop activated for {symbol}. New SL at entry: {new_stop_loss:.4f}", "TRAIL_ACTIVATED")
                            send_telegram_message(f"🎯 *تأمين صفقة {symbol}*\nتم رفع الوقف إلى نقطة الدخول بعد تحقيق ربح `{profit_percent:.2f}%`.")
                        continue

                # 4. تحديث الوقف المتحرك إذا كان نشطاً
                if signal_details.get('is_trailing_active'):
                    trailing_distance = signal_details.get('trailing_stop_distance', 0)
                    if trailing_distance > 0:
                        potential_new_sl = current_price - trailing_distance
                        if potential_new_sl > stop_loss:
                            if update_signal_in_db(signal['id'], {"stop_loss": potential_new_sl}):
                                signal['stop_loss'] = potential_new_sl
                                # لا داعي للإشعار هنا لتجنب كثرة الرسائل
            time.sleep(1)
        except Exception as e:
            logger.error(f"❌ [Trade Manager] A critical error occurred: {e}", exc_info=True)
            time.sleep(10)

def update_market_state_loop():
    logger.info("🚀 [Market State] Starting market state update loop...")
    while True:
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
            
            time.sleep(60 * 5)
        except Exception as e:
            logger.error(f"❌ [Market State] Error: {e}", exc_info=True)
            time.sleep(60)

def update_balance_loop():
    logger.info("🚀 [Balance Updater] Starting balance update loop...")
    while True:
        try:
            balance_info = client.get_asset_balance(asset='USDT')
            with balance_lock:
                global usdt_balance
                usdt_balance = float(balance_info['free'])
            logger.info(f"💰 [Balance] USDT balance updated: {usdt_balance:.2f}")
        except Exception as e:
            logger.error(f"❌ [Balance] Could not update USDT balance: {e}")
        time.sleep(60 * 10) 

# --- نقطة بداية البرنامج ---
if __name__ == '__main__':
    logger.info("="*50 + "\n====== Starting Crypto Trading Bot V17.0.0 ======\n" + "="*50)
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
    
    start_websocket()
    Thread(target=main_bot_loop, daemon=True).start()
    Thread(target=trade_management_loop, daemon=True).start()
    Thread(target=update_market_state_loop, daemon=True).start()
    Thread(target=update_balance_loop, daemon=True).start() 
    
    logger.info("🌐 [Flask] Starting UI on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
