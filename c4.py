# ملف c4.py - نسخة V21.0.0 (نظام اختبار الاستراتيجيات المتكامل)
# --- وصف الإصدار:
# هذا الإصدار يضيف ميزة رئيسية جديدة وهي نظام اختبار الاستراتيجيات (Backtesting).
# 1.  [ميزة جديدة] إضافة صفحة "اختبار الاستراتيجيات" جديدة في واجهة المستخدم.
# 2.  [ميزة جديدة] يمكن للمستخدم اختيار استراتيجية، رمز عملة، وفترة زمنية للاختبار.
# 3.  [تحسين المنطق] تم تحسين منطق محاكاة الاختبار الخلفي لتجنب الانحياز للمستقبل، حيث يتم معالجة البيانات شمعة بشمعة.
# 4.  [واجهة المستخدم] تعرض صفحة الاختبار نتائج مفصلة تشمل مقاييس الأداء (معدل الربح، إجمالي الصفقات، إلخ) ورسم بياني لمنحنى رأس المال.
# 5.  [تكامل] إضافة مسارات API جديدة في Flask لمعالجة طلبات الاختبار الخلفي وتقديم النتائج للواجهة.

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
from flask_sock import Sock
from threading import Thread, Lock
from datetime import datetime, timezone, timedelta
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
        logging.FileHandler('crypto_bot_v21_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV21')

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
cooldowns_by_symbol = {}
cooldowns_lock = Lock()
consecutive_losses_by_symbol = {}
consecutive_losses_lock = Lock()
COOLDOWN_MINUTES_AFTER_SL = 20
PAPER_TRADE_INITIAL_BALANCE = 1000.0

# --- المتغيرات القابلة للتعديل ---
RISK_PER_TRADE_PERCENT: float = 0.85
risk_per_trade_lock = Lock()
MAX_OPEN_TRADES: int = 3
PAPER_TRADE_SIZE_USDT: float = 10.0
TRAILING_STOP_ACTIVATION_PROFIT_PERCENT: float = 1.4

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
    "Volume Filter Failed": "فلتر حجم التداول فشل",
    "MACD Momentum Failed": "فلتر زخم الماكد فشل",
    "Long-term Trend Filter Failed": "فلتر الاتجاه طويل الأجل فشل",
}

# --- إعداد تطبيق Flask و WebSocket ---
app = Flask(__name__)
CORS(app)
sock = Sock(app)
ws_clients: List[Any] = []
ws_clients_lock = Lock()

# --- دوال WebSocket ---
def broadcast(data: Dict):
    """إرسال البيانات إلى جميع عملاء WebSocket المتصلين."""
    with ws_clients_lock:
        clients_to_remove = []
        for client in ws_clients:
            try:
                client.send(json.dumps(data, cls=NpEncoder))
            except Exception:
                clients_to_remove.append(client)
        for client in clients_to_remove:
            ws_clients.remove(client)

def get_dashboard_payload() -> Dict:
    """تجميع بيانات لوحة التحكم لإرسالها."""
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
    return {
        "trading_enabled": trading_enabled, "paper_trading_mode": is_paper_mode,
        "usdt_balance": current_balance, "open_signals": open_signals_with_progress,
        "notifications": notifications, "rejections": rejections, "market_state": market_state,
        "server_time": datetime.now(timezone.utc).isoformat(), "live_prices": live_prices_copy
    }

def broadcast_dashboard_update():
    """تجميع وإرسال تحديثات لوحة التحكم عبر WebSocket."""
    payload = get_dashboard_payload()
    broadcast({"type": "dashboard_update", "payload": payload})

# --- دوال تهيئة الخدمات وقاعدة البيانات ---
def optimize_database():
    """تحسين أداء قاعدة البيانات بإضافة الفهارس اللازمة."""
    if not check_db_connection() or not conn:
        return
    try:
        with conn.cursor() as cur:
            logger.info("[DB] Optimizing database with indexes...")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol_status ON signals(symbol, status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_notifications_timestamp ON notifications(timestamp);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_status_closed_at ON signals(status, closed_at);")
            conn.commit()
            logger.info("✅ [DB] Database indexes optimized successfully.")
    except Exception as e:
        logger.error(f"❌ [DB] Error optimizing database: {e}")
        if conn: conn.rollback()

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
            optimize_database()
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
        broadcast_dashboard_update()
    except Exception as e:
        logger.error(f"❌ [DB] Failed to save notification: {e}")
        if conn: conn.rollback()

def log_rejection(symbol: str, reason_key: str, details: Optional[Dict] = None):
    try:
        reason_ar = REJECTION_REASONS_AR.get(reason_key, reason_key)
        log_entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol, "reason": reason_ar}
        with rejection_logs_lock: rejection_logs_cache.appendleft(log_entry)
        broadcast_dashboard_update()
    except Exception as e:
        logger.error(f"❌ [Log Rejection] Error logging rejection for {symbol}: {e}", exc_info=True)

def get_notification_settings() -> Dict:
    defaults = {'telegram_enabled': True, 'email_enabled': False, 'min_profit_notification': 1.0, 'max_loss_notification': -1.0}
    if not redis_client: return defaults
    try:
        settings_data = redis_client.get('notification_settings')
        if settings_data:
            settings = json.loads(settings_data)
            for key, value in defaults.items(): settings.setdefault(key, value)
            return settings
        return defaults
    except Exception as e:
        logger.error(f"❌ [Redis] Failed to get notification settings: {e}"); return defaults

def send_telegram_message(message: str, force: bool = False):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    settings = get_notification_settings()
    if not settings.get('telegram_enabled') and not force: return
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
        broadcast_dashboard_update()

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

# --- Risk Management ---
def dynamic_risk_management(symbol, df, consecutive_losses=0):
    with risk_per_trade_lock: base_risk = RISK_PER_TRADE_PERCENT
    atr_percent = df['atr_percent'].iloc[-1]
    if atr_percent > 5.0: volatility_factor = 0.7
    elif atr_percent < 1.5: volatility_factor = 1.2
    else: volatility_factor = 1.0
    loss_factor = max(0.5, 1.0 - (consecutive_losses * 0.15))
    final_risk = base_risk * volatility_factor * loss_factor
    final_risk_percent = min(final_risk, 2.0)
    logger.info(f"[Risk] Dynamic risk for {symbol}: {final_risk_percent:.2f}% (Base: {base_risk}%, Vol: {volatility_factor:.2f}, Loss: {loss_factor:.2f})")
    return final_risk_percent

# --- Filters & Strategies ---
def check_market_volatility_filter(df: pd.DataFrame) -> bool:
    if 'atr_percent' not in df.columns or len(df) < 30:
        log_rejection(getattr(df, "name", "—"), "Market Volatility Filter Failed"); return False
    recent = df['atr_percent'].tail(96).dropna()
    last = float(df.iloc[-1].get('atr_percent', 0))
    if recent.empty:
        log_rejection(getattr(df, "name", "—"), "Market Volatility Filter Failed"); return False
    q25 = float(np.percentile(recent, 25)); q90 = float(np.percentile(recent, 90))
    lower = max(0.35, q25 * 0.9); upper = min(8.0, q90 * 1.1)
    if last < lower or last > upper:
        log_rejection(df.name, "Market Volatility Filter Failed"); return False
    return True

def check_trend_strength_filter(df: pd.DataFrame, adx_threshold: int) -> bool:
    if 'adx' not in df.columns or len(df) < 5:
        log_rejection(getattr(df, "name", "—"), "Trend Strength Filter Failed"); return False
    recent_adx = float(pd.Series(df['adx'].tail(3)).mean())
    if recent_adx < (adx_threshold * 0.95):
        log_rejection(df.name, "Trend Strength Filter Failed"); return False
    return True

def check_volume_filter(df: pd.DataFrame, min_volume_percentile: float = 30) -> bool:
    if 'volume' not in df.columns or len(df) < 50: return False
    if df['volume'].tail(50).isnull().any(): return False
    current_volume = df['volume'].iloc[-1]
    volume_ma = df['volume'].rolling(20, min_periods=20).mean().iloc[-1]
    volume_percentile = df['volume'].rolling(50, min_periods=50).quantile(min_volume_percentile / 100).iloc[-1]
    if pd.isna(current_volume) or pd.isna(volume_ma) or pd.isna(volume_percentile): return False
    return current_volume > max(volume_ma, volume_percentile)

def is_htf_bullish_confirmation(symbol: str, htf: str = '1h', mode: str = 'Strict') -> bool:
    if mode == 'Disabled': return True
    try:
        df = fetch_historical_data(symbol, htf, days=40)
        if df is None or len(df) < 50: return False
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        last = df.iloc[-1]
        if mode == 'Strict':
            df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
            last = df.iloc[-1]
            return last['close'] > last['ema50'] and last['ema50'] > last['ema200']
        elif mode == 'Relaxed':
            return last['close'] > last['ema50']
        return False
    except Exception as e:
        logger.warning(f"[HTF] Could not confirm HTF trend for {symbol}: {e}"); return False

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
    if len(df) < 21 or not {'bb_lower', 'stoch_rsi_k', 'rsi', 'open', 'close', 'macd', 'macd_signal'}.issubset(df.columns): return False
    if not check_volume_filter(df, min_volume_percentile=40):
        log_rejection(df.name, "Volume Filter Failed"); return False
    if df['macd'].iloc[-1] < df['macd_signal'].iloc[-1]:
        log_rejection(df.name, "MACD Momentum Failed"); return False
    last, prev = df.iloc[-1], df.iloc[-2]
    bounce = (prev['close'] < prev['bb_lower']) and (last['close'] > last['bb_lower'])
    stoch_rising = last['stoch_rsi_k'] > prev['stoch_rsi_k']
    bullish_body = last['close'] > last['open']
    rsi_improving = last['rsi'] > prev['rsi']
    not_overbought = last['rsi'] < 65
    signal = bounce and (stoch_rising or bullish_body) and rsi_improving and not_overbought
    if not signal: log_rejection(df.name, "Bullish Confirmation Failed")
    return signal

def check_macd_ema_strategy_enhanced(df: pd.DataFrame) -> bool:
    needed = {'macd', 'macd_signal', 'ema9', 'ema21', 'rsi', 'close', 'adx'}
    if len(df) < 30 or not needed.issubset(df.columns): return False
    if df['adx'].iloc[-1] < 22:
        log_rejection(df.name, "Trend Strength Filter Failed"); return False
    last, prev = df.iloc[-1], df.iloc[-2]
    macd_cross_up = (prev['macd'] <= prev['macd_signal']) and (last['macd'] > last['macd_signal'])
    hist_now = last['macd'] - last['macd_signal']
    hist_prev = prev['macd'] - prev['macd_signal']
    hist_increasing = (hist_now > hist_prev) and (hist_prev > 0 or macd_cross_up)
    ema_ok = (last['close'] > last['ema21']) and (last['ema9'] > last['ema21'])
    rsi_ok = 40 <= last['rsi'] <= 65
    return (macd_cross_up or hist_increasing) and ema_ok and rsi_ok

def check_ema_rsi_strategy_enhanced(df: pd.DataFrame) -> bool:
    needed = {'ema9','ema21','rsi','low','close'}
    if len(df) < 25 or not needed.issubset(df.columns): return False
    last3 = df.tail(3)
    ema9_over_21 = (last3['ema9'] > last3['ema21']).sum() >= 2
    last = last3.iloc[-1]
    rsi_ok = 50 <= float(last['rsi']) <= 65
    pullback_ok = (float(last['low']) <= float(last['ema9'])) and (float(last['close']) > float(last['ema9']))
    return ema9_over_21 and rsi_ok and pullback_ok

def check_pullback_strategy_enhanced(df: pd.DataFrame) -> bool:
    needed = {'ema9','ema21','ema50','open','close','low'}
    if len(df) < 55 or not needed.issubset(df.columns): return False
    last = df.iloc[-1]
    uptrend = (last['ema21'] > last['ema50']) and (last['close'] > last['ema50'])
    if not uptrend: return False
    recent = df.tail(4)
    dipped = ((recent['low'] <= recent['ema21']) | (recent['low'] <= recent['ema9'])).any()
    bullish_close = last['close'] > last['open'] and last['close'] > last['ema9']
    return dipped and bullish_close

def check_momentum_volatility_strategy(df: pd.DataFrame) -> bool:
    needed = {'atr_percent','ema9','ema21','macd','macd_signal','close'}
    if len(df) < 200: return False
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    if df['close'].iloc[-1] < df['ema200'].iloc[-1]: log_rejection(df.name, "Long-term Trend Filter Failed"); return False
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
    try:
        with cooldowns_lock: until = cooldowns_by_symbol.get(symbol)
        if until and datetime.now(timezone.utc) < until:
            log_rejection(symbol, "Cooldown Active", {"until": until.isoformat()}); return
    except Exception: pass
    with trading_mode_lock: is_real = not paper_trading_mode
    trade_levels = calculate_trade_levels(df)
    entry_price = trade_levels['entry_price']
    if is_real:
        with balance_lock: current_usdt_balance = usdt_balance
        with consecutive_losses_lock: losses = consecutive_losses_by_symbol.get(symbol, 0)
        risk_percent = dynamic_risk_management(symbol, df, losses)
        max_risk_usdt = current_usdt_balance * (risk_percent / 100)
        stop_loss = trade_levels['stop_loss']
        risk_per_unit = max(entry_price - stop_loss, 1e-8)
        quantity = max_risk_usdt / risk_per_unit
        if quantity <= 0: log_rejection(symbol, "Insufficient Balance"); return
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info: logger.error(f"❌ [Real Trade] Could not find exchange info for {symbol}"); return
        step_size = next((f['stepSize'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), '0.000001')
        adjusted_quantity = adjust_quantity_to_step_size(quantity, step_size)
        notional = adjusted_quantity * entry_price
        min_notional = float(next((f['minNotional'] for f in symbol_info['filters'] if f['filterType'] == 'NOTIONAL'), '0.0'))
        if notional < min_notional:
            log_rejection(symbol, "MinNotional Filter Failed", {"required": min_notional, "actual": notional}); return
        try:
            logger.info(f"💰 [Real Trade] Placing LIVE MARKET BUY order for {adjusted_quantity} of {symbol}")
            order = client.create_order(symbol=symbol, side=Client.SIDE_BUY, type=Client.ORDER_TYPE_MARKET, quantity=adjusted_quantity)
            avg_fill_price = sum(float(f['price']) * float(f['qty']) for f in order.get('fills', [])) / sum(float(f['qty']) for f in order.get('fills', [])) if order.get('fills') else entry_price
            final_quantity = float(order.get('executedQty', adjusted_quantity))
            order_id = order.get('orderId', 'N/A')
            save_signal_to_db(symbol, avg_fill_price, trade_levels, strategy_name, True, final_quantity, order_id)
            message = (f"💰 *صفقة حقيقية جديدة*\n`{symbol}` | `{strategy_name}`\n*دخول:* `{avg_fill_price:.4f}`\n*كمية:* `{final_quantity}`")
            send_telegram_message(message, force=True)
            log_and_notify("info", f"Opened REAL trade for {symbol}", "REAL_TRADE_OPEN")
        except BinanceAPIException as e:
            logger.error(f"❌ [Real Trade] Binance API Error for {symbol}: {e}")
            send_telegram_message(f"❌ *خطأ في صفقة حقيقية لـ {symbol}*\n`{e}`", force=True)
        except Exception as e:
            logger.error(f"❌ [Real Trade] CRITICAL ERROR creating real trade for {symbol}: {e}", exc_info=True)
    else:
        quantity = PAPER_TRADE_SIZE_USDT / entry_price
        save_signal_to_db(symbol, entry_price, trade_levels, strategy_name, False, quantity)
        message = (f"📊 *صفقة ورقية جديدة*\n`{symbol}` | `{strategy_name}`\n*دخول:* `{entry_price:.4f}`\n*هدف1:* `{trade_levels['target_price_1']:.4f}`\n*وقف:* `{trade_levels['stop_loss']:.4f}`")
        send_telegram_message(message, force=True)
        log_and_notify("info", f"Opened paper trade for {symbol}", "PAPER_TRADE_OPEN")

def save_signal_to_db(symbol: str, entry_price: float, trade_levels: Dict, strategy_name: str, is_real: bool, quantity: float, order_id: Optional[str] = None):
    try:
        if not (check_db_connection() and conn): return
        signal_details = {
            "atr": trade_levels['atr'], "trailing_stop_activated": False,
            "trailing_stop_distance": trade_levels['trailing_stop_distance'], "tp1_done": False
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
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>لوحة التحكم - بوت التداول (V21)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
:root{--bg:#0b1020;--panel:#121b36;--accent:#3aa0ff;--ok:#15c46a;--warn:#ff9f1a;--bad:#ff4757;--muted:#8aa0c8;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:#e8f1ff;font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,"Noto Sans",Arial}
.container{max-width:1400px;margin:0 auto;padding:16px;display:flex;flex-direction:column;gap:16px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between}
h1{font-size:18px;margin:0;font-weight:700;color:#d7e4ff}
.badge{padding:6px 10px;border-radius:999px;font-size:12px;background:#0d1730;border:1px solid #1e2c52;color:#cce0ff}
.main-layout{display:grid;grid-template-columns:1fr;gap:16px;}
@media(min-width: 900px){.main-layout{grid-template-columns:1fr 320px;}}
.left-column{display:flex;flex-direction:column;gap:16px}
.right-column{display:flex;flex-direction:column;gap:16px}
.card{background:var(--panel);border:1px solid #1e2c52;border-radius:14px;box-shadow:0 8px 30px rgba(0,0,0,.25);overflow:hidden}
.card h2{margin:0;padding:12px 14px;border-bottom:1px solid #1e2c52;font-size:14px;color:#cfe2ff}
.card-body{padding:12px}
.controls{display:flex;gap:8px;flex-wrap:wrap}
.btn{appearance:none;border:1px solid #2a3a68;background:#0f1b3b;color:#d9e7ff;padding:10px 14px;border-radius:10px;cursor:pointer;font-weight:700;transition: background-color 0.2s, transform 0.2s; will-change: transform; text-decoration: none;}
.btn:hover{transform:translateY(-1px);border-color:#3a58a6}
.btn.warn{background:linear-gradient(180deg,#3b2a0f,#291b08);border-color:#8b5b0f}
.signals-grid{display:grid;grid-template-columns:repeat(auto-fill, minmax(280px, 1fr));gap:10px}
.signal{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;padding:10px;border:1px solid #24335f;border-radius:12px;background:#0d1730; will-change: opacity; transition: opacity 0.3s;}
.sig-title{font-weight:700}
.sig-meta{font-size:12px;color:var(--muted)}
.price{font-variant-numeric:tabular-nums;direction:ltr; transition: color 0.3s, background-color 0.3s;}
.price.flash-up{background-color:rgba(21, 196, 106, 0.2); color: #15c46a;}
.price.flash-down{background-color:rgba(255, 71, 87, 0.2); color: #ff4757;}
.progress{height:8px;background:#0b1126;border:1px solid #233056;border-radius:999px;overflow:hidden}
.progress>span{display:block;height:100%;}
.kv{display:grid;grid-template-columns:auto 1fr;gap:6px 10px}
.kv div:nth-child(odd){opacity:.8}
.trend{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}
.trend .pill{background:#0d1730;border:1px solid #1f2d55;border-radius:10px;padding:8px;text-align:center}
.pill b{display:block;font-size:12px;color:#9fb7ef}
.pill span{font-size:12px}
.green{color:var(--ok)}.red{color:var(--bad)}.amber{color:var(--warn)}
.table{width:100%;border-collapse:separate;border-spacing:0 8px}
.table th{font-size:12px;text-align:right;color:#9ab2e2;font-weight:600;padding:0 6px}
.table td{padding:8px;background:#0d1730;border:1px solid #24335f}
.switch{display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;border:1px solid #2a3a68;background:#0f1b3b;cursor:pointer;user-select:none}
.switch input{display:none}
.switch .dot{width:14px;height:14px;border-radius:50%;background:#6a7fb2;transition:.2s}
.switch input:checked + .dot{background:#24d08a;transform:translateX(2px) scale(1.1)}
.small{font-size:12px;color:#a8bfeb}
.metrics-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px;}
.metric-card {background: #0d1730; border: 1px solid #24335f; border-radius: 12px; padding: 16px; text-align: center; transition: transform 0.2s, box-shadow 0.2s;}
.metric-card:hover {transform: translateY(-2px); box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);}
.metric-title {font-size: 14px; color: #8aa0c8; margin-bottom: 8px;}
.metric-value {font-size: 24px; font-weight: 700; color: #3aa0ff;}
.trade-controls {display: flex; gap: 12px; margin-bottom: 24px;}
.trade-settings {background: #0d1730; border: 1px solid #24335f; border-radius: 12px; padding: 16px;}
.form-group { margin-bottom: 12px; }
.form-group label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
.form-group input { width: 100%; background: #0b1126; border: 1px solid #233056; color: #e8f1ff; padding: 8px; border-radius: 8px; }
.chart-container { height: 250px; }
@media (max-width: 768px) {
    .table { font-size: 12px; }
    .table th, .table td { padding: 6px 4px; }
    .metrics-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
</head>
<body>
<div class="container">
  <header><h1>لوحة التحكم • فريم 15 دقيقة</h1><div class="badge" id="serverTime">—</div></header>
  <div class="main-layout">
    <div class="left-column">
      <div class="card"><h2>الصفقات المفتوحة</h2><div class="card-body"><div id="signals" class="signals-grid"></div></div></div>
      <div class="card">
          <h2>تحليل الأداء المتقدم</h2>
          <div class="card-body">
              <div class="metrics-grid">
                  <div class="metric-card"><div class="metric-title">معدل الربح</div><div class="metric-value" id="winRate">—</div></div>
                  <div class="metric-card"><div class="metric-title">معدل المخاطرة/العائد</div><div class="metric-value" id="riskRewardRatio">—</div></div>
                  <div class="metric-card"><div class="metric-title">أكبر تراجع</div><div class="metric-value" id="maxDrawdown">—</div></div>
                  <div class="metric-card"><div class="metric-title">عامل الربح</div><div class="metric-value" id="profitFactor">—</div></div>
              </div>
              <div class="chart-container"><canvas id="drawdownChart"></canvas></div>
          </div>
      </div>
    </div>
    <div class="right-column">
      <div class="card">
        <h2>التحكم والحالة</h2>
        <div class="card-body">
          <div class="controls">
            <label class="switch"><input id="toggleTrading" type="checkbox" /><span class="dot"></span><span class="small">تشغيل التداول</span></label>
            <button class="btn" id="toggleMode">وضع: ورقي</button>
            <a class="btn" href="/settings">الإعدادات</a>
            <a class="btn" href="/backtest">الاختبار الخلفي</a>
          </div>
          <div class="kv" style="margin-top:12px">
            <div>الرصيد (USDT):</div><div id="balance">—</div><div>عدد الصفقات:</div><div id="openCount">—</div>
          </div>
          <div id="trend" class="trend"></div>
        </div>
      </div>
      <div class="card">
          <h2>إدارة الصفقات</h2>
          <div class="card-body">
              <div class="trade-controls">
                  <button class="btn" id="closeAllTrades">إغلاق جميع الصفقات</button>
                  <button class="btn warn" id="emergencyStop">إيقاف طوارئ</button>
              </div>
          </div>
      </div>
      <div class="card">
        <h2>سجل الأحداث</h2>
        <div class="card-body" style="padding:0">
          <table class="table" id="events"><thead><tr><th>الوقت</th><th>النوع</th><th>الرسالة</th></tr></thead><tbody></tbody></table>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
const qs = s => document.querySelector(s);
let lastPrices = {};
let drawdownChartInstance = null;
async function toggleTrading() { await fetch('/toggle_trading', {method:'POST'}); }
async function toggleMode() { await fetch('/toggle_real_trading', {method:'POST'}); }
async function closeAllTrades() { 
    if(confirm('هل أنت متأكد من رغبتك في إغلاق جميع الصفقات المفتوحة؟')) {
        await fetch('/close_all_trades', {method:'POST'}); 
    }
}
async function emergencyStop() { 
    if(confirm('تحذير: هذا سيوقف التداول ويغلق جميع الصفقات فوراً. هل أنت متأكد؟')) {
        await fetch('/emergency_stop', {method:'POST'}); 
    }
}
qs('#toggleTrading').addEventListener('change', toggleTrading);
qs('#toggleMode').addEventListener('click', toggleMode);
qs('#closeAllTrades').addEventListener('click', closeAllTrades);
qs('#emergencyStop').addEventListener('click', emergencyStop);
function fmt(n){ return n == null ? '—' : (+n).toLocaleString('en-US', {maximumFractionDigits: 6}); }
function clsByDelta(d){ if(d > 0) return 'green'; if(d < 0) return 'red'; return ''; }
function render(data){
  qs('#serverTime').textContent = new Date(data.server_time).toLocaleTimeString('ar-EG');
  qs('#toggleTrading').checked = !!data.trading_enabled;
  qs('#toggleMode').textContent = 'وضع: ' + (data.paper_trading_mode ? 'ورقي' : 'حقيقي');
  qs('#balance').textContent = fmt(data.usdt_balance);
  const sigs = data.open_signals || {};
  qs('#openCount').textContent = Object.keys(sigs).length;
  const trend = data.market_state?.trend_details_by_tf || {};
  const tfOrder = ['15m','1h','4h'];
  qs('#trend').innerHTML = tfOrder.map(tf => {
    const t = trend[tf] || {};
    const c = t.trend === 'Bullish' ? 'green' : (t.trend === 'Bearish' ? 'red' : 'amber');
    return `<div class="pill"><b>${tf}</b><span class="${c}">${t.trend||'—'}</span><br><span class="small">RSI ${fmt(t.rsi)}</span></div>`;
  }).join('');
  const box = qs('#signals');
  const existingSignalIds = new Set([...box.children].map(el => el.dataset.id));
  const incomingSignalIds = new Set(Object.values(sigs).map(s => String(s.id)));
  existingSignalIds.forEach(id => {
    if (!incomingSignalIds.has(id)) {
      const el = box.querySelector(`[data-id='${id}']`);
      if (el) el.remove();
    }
  });
  Object.values(sigs).forEach(s => {
    const cp = s.current_price;
    const prev = lastPrices[s.symbol] || cp;
    const delta = cp - prev;
    lastPrices[s.symbol] = cp;
    const pToTp = Math.min(100, s.progress_to_tp || 0);
    const pToSl = Math.min(100, s.progress_to_sl || 0);
    const progressBar = pToTp > 0 ? `<div class="progress" title="نحو الهدف"><span style="width:${pToTp}%; background:linear-gradient(90deg,var(--ok),#3fd1b0)"></span></div>` :
                                   `<div class="progress" title="نحو الوقف"><span style="width:${pToSl}%; background:linear-gradient(90deg,var(--bad),#ff7a7a)"></span></div>`;
    const meta = `دخول ${fmt(s.entry_price)} • وقف ${fmt(s.stop_loss)} • هدف ${fmt(s.target_price_1)}`;
    const btnClose = `<button class="btn warn" onclick="fetch('/close_trade/${s.id}',{method:'POST'})">إغلاق</button>`;
    let el = box.querySelector(`[data-id='${s.id}']`);
    if (!el) {
        el = document.createElement('div');
        el.className = 'signal';
        el.dataset.id = s.id;
        box.appendChild(el);
    }
    el.innerHTML = `<div><div class="sig-title">${s.symbol}</div><div class="sig-meta">${meta}</div>${progressBar}</div><div style="text-align:end"><div class="price">${fmt(cp)}</div><div class="small ${clsByDelta(delta)}">${delta>0?'▲':(delta<0?'▼':'•')} ${fmt(Math.abs(delta))}</div>${btnClose}</div>`;
    const priceEl = el.querySelector('.price');
    if (delta > 0) {
        priceEl.classList.add('flash-up');
        setTimeout(() => priceEl.classList.remove('flash-up'), 500);
    } else if (delta < 0) {
        priceEl.classList.add('flash-down');
        setTimeout(() => priceEl.classList.remove('flash-down'), 500);
    }
  });
  const tbody = qs('#events tbody');
  tbody.innerHTML = (data.notifications || []).map(n => `<tr><td>${new Date(n.timestamp).toLocaleTimeString('ar-EG')}</td><td>${n.type||''}</td><td>${n.message||''}</td></tr>`).join('');
}
function updateAdvancedPerformance(data) {
    qs('#winRate').textContent = data.winRate ? `${data.winRate.toFixed(2)}%` : '—';
    qs('#riskRewardRatio').textContent = data.riskRewardRatio ? `${data.riskRewardRatio.toFixed(2)}:1` : '—';
    qs('#maxDrawdown').textContent = data.maxDrawdown ? `${data.maxDrawdown.toFixed(2)}%` : '—';
    qs('#profitFactor').textContent = data.profitFactor ? data.profitFactor.toFixed(2) : '—';
    if (!drawdownChartInstance) {
        createDrawdownChart(data.drawdown_data);
    } else {
        drawdownChartInstance.data.labels = data.drawdown_data.labels;
        drawdownChartInstance.data.datasets[0].data = data.drawdown_data.values;
        drawdownChartInstance.update('none');
    }
}
function createDrawdownChart(chartData) {
    const ctx = document.getElementById('drawdownChart').getContext('2d');
    drawdownChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.labels,
            datasets: [{
                label: 'تراجع رأس المال', data: chartData.values, borderColor: '#ff4757',
                backgroundColor: 'rgba(255, 71, 87, 0.1)', tension: 0.4,
                fill: true, pointRadius: 0, borderWidth: 2
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: 'var(--muted)', autoSkip: true, maxTicksLimit: 10 }, grid: { display: false } },
                y: { ticks: { color: 'var(--muted)', callback: (v) => v + '%' }, grid: { color: 'rgba(255, 255, 255, 0.05)' } }
            }
        }
    });
}
function setupWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    const socket = new WebSocket(wsUrl);
    socket.onopen = (e) => console.log("WebSocket connection established");
    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'dashboard_update') {
            render(data.payload);
        } else if (data.type === 'advanced_performance_update') {
            updateAdvancedPerformance(data.payload);
        }
    };
    socket.onclose = (event) => {
        console.log("WebSocket connection closed, attempting to reconnect...");
        setTimeout(setupWebSocket, 3000);
    };
    socket.onerror = (error) => console.error("WebSocket error:", error);
}
async function initialLoad() {
    try {
        const res = await fetch('/api/dashboard_data');
        if (res.ok) render(await res.json());
        const perfRes = await fetch('/api/advanced_performance_data');
        if (perfRes.ok) updateAdvancedPerformance(await perfRes.json());
    } catch(error) {
        console.error("Error during initial load:", error);
    }
}
initialLoad();
setupWebSocket();
</script>
</body>
</html>
"""

BACKTEST_TEMPLATE = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>اختبار الاستراتيجيات (Backtesting)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
:root{--bg:#0b1020;--panel:#121b36;--accent:#3aa0ff;--ok:#15c46a;--warn:#ff9f1a;--bad:#ff4757;--muted:#8aa0c8;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:#e8f1ff;font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,"Noto Sans",Arial}
.container{max-width:1200px;margin:0 auto;padding:16px;display:flex;flex-direction:column;gap:16px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;padding-bottom:16px;border-bottom:1px solid #1e2c52;}
h1{font-size:18px;margin:0;font-weight:700;color:#d7e4ff}
.card{background:var(--panel);border:1px solid #1e2c52;border-radius:14px;box-shadow:0 8px 30px rgba(0,0,0,.25);overflow:hidden}
.card h2{margin:0;padding:12px 14px;border-bottom:1px solid #1e2c52;font-size:14px;color:#cfe2ff}
.card-body{padding:16px}
.form-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; align-items: end;}
.form-group label {display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px;}
.form-group input, .form-group select {width: 100%; background: #0b1126; border: 1px solid #233056; color: #e8f1ff; padding: 10px; border-radius: 8px;}
.btn{appearance:none;border:1px solid #2a3a68;background:#0f1b3b;color:#d9e7ff;padding:10px 14px;border-radius:10px;cursor:pointer;font-weight:700;transition: .18s; text-decoration: none;}
.btn.primary {background: var(--accent); color: #fff; border-color: var(--accent);}
.results-grid {display: grid; grid-template-columns: 1fr; gap: 16px; margin-top: 24px;}
@media(min-width: 900px){.results-grid{grid-template-columns: 1fr 1fr;}}
.metrics-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px;}
.metric-card {background: #0d1730; border-radius: 12px; padding: 16px; text-align: center;}
.metric-title {font-size: 14px; color: #8aa0c8; margin-bottom: 8px;}
.metric-value {font-size: 22px; font-weight: 700;}
.green{color:var(--ok)}.red{color:var(--bad)}
.table-container {max-height: 400px; overflow-y: auto;}
.table{width:100%;border-collapse:collapse; font-size: 12px;}
.table th, .table td{padding: 8px; text-align: right; border-bottom: 1px solid #1e2c52;}
.table th {font-weight: 600; color: #9ab2e2;}
#loader {text-align: center; padding: 40px; display: none;}
</style>
</head>
<body>
<div class="container">
    <header><h1>اختبار الاستراتيجيات (Backtesting)</h1><a href="/" class="btn">العودة للرئيسية</a></header>
    <div class="card">
        <div class="card-body">
            <form id="backtest-form">
                <div class="form-grid">
                    <div class="form-group">
                        <label for="strategy">اختر الاستراتيجية</label>
                        <select id="strategy" name="strategy">
                            {% for key, name in STRATEGY_NAMES.items() %}
                            <option value="{{ key }}">{{ name }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="symbol">رمز العملة (مثل BTCUSDT)</label>
                        <input type="text" id="symbol" name="symbol" value="BTCUSDT" required>
                    </div>
                    <div class="form-group">
                        <label for="days">أيام الاختبار</label>
                        <input type="number" id="days" name="days" value="90" required>
                    </div>
                    <button type="submit" class="btn primary">بدء الاختبار</button>
                </div>
            </form>
        </div>
    </div>
    <div id="loader">جاري تحميل النتائج...</div>
    <div id="results-container" style="display: none;">
        <div class="results-grid">
            <div class="card">
                <h2>ملخص الأداء</h2>
                <div class="card-body">
                    <div class="metrics-grid">
                        <div class="metric-card"><div class="metric-title">إجمالي الصفقات</div><div class="metric-value" id="totalTrades"></div></div>
                        <div class="metric-card"><div class="metric-title">معدل الربح</div><div class="metric-value" id="winRate"></div></div>
                        <div class="metric-card"><div class="metric-title">متوسط الربح/الخسارة</div><div class="metric-value" id="avgProfit"></div></div>
                        <div class="metric-card"><div class="metric-title">عامل الربح</div><div class="metric-value" id="profitFactor"></div></div>
                    </div>
                </div>
            </div>
            <div class="card">
                <h2>منحنى رأس المال</h2>
                <div class="card-body" style="height: 250px;"><canvas id="equityChart"></canvas></div>
            </div>
        </div>
        <div class="card" style="margin-top: 16px;">
            <h2>تفاصيل الصفقات</h2>
            <div class="card-body table-container">
                <table class="table">
                    <thead><tr><th>وقت الدخول</th><th>سعر الدخول</th><th>وقت الخروج</th><th>سعر الخروج</th><th>سبب الخروج</th><th>الربح %</th></tr></thead>
                    <tbody id="trades-table"></tbody>
                </table>
            </div>
        </div>
    </div>
</div>
<script>
const qs = s => document.querySelector(s);
let equityChartInstance = null;

qs('#backtest-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    qs('#loader').style.display = 'block';
    qs('#results-container').style.display = 'none';
    
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());

    try {
        const response = await fetch('/api/run_backtest', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        const results = await response.json();
        
        if (results.error) {
            alert('خطأ: ' + results.error);
            return;
        }
        
        displayResults(results);
    } catch (err) {
        alert('حدث خطأ غير متوقع. يرجى مراجعة الكونسول.');
        console.error(err);
    } finally {
        qs('#loader').style.display = 'none';
    }
});

function displayResults(data) {
    qs('#results-container').style.display = 'block';
    
    qs('#totalTrades').textContent = data.total_trades;
    qs('#winRate').textContent = `${data.win_rate.toFixed(2)}%`;
    qs('#avgProfit').textContent = `${data.avg_profit.toFixed(2)}%`;
    qs('#profitFactor').textContent = data.profit_factor.toFixed(2);
    
    const avgProfitEl = qs('#avgProfit');
    avgProfitEl.classList.toggle('green', data.avg_profit > 0);
    avgProfitEl.classList.toggle('red', data.avg_profit < 0);

    const tradesTable = qs('#trades-table');
    tradesTable.innerHTML = data.results.map(trade => `
        <tr>
            <td>${new Date(trade.entry_time).toLocaleString('ar-EG')}</td>
            <td>${trade.entry_price.toFixed(4)}</td>
            <td>${new Date(trade.exit_time).toLocaleString('ar-EG')}</td>
            <td>${trade.exit_price.toFixed(4)}</td>
            <td>${trade.exit_reason}</td>
            <td class="${trade.profit_percent > 0 ? 'green' : 'red'}">${trade.profit_percent.toFixed(2)}%</td>
        </tr>
    `).join('');
    
    updateEquityChart(data.equity_curve);
}

function updateEquityChart(equityData) {
    const ctx = document.getElementById('equityChart').getContext('2d');
    if (equityChartInstance) {
        equityChartInstance.destroy();
    }
    equityChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: equityData.map((_, i) => i),
            datasets: [{
                label: 'رأس المال',
                data: equityData,
                borderColor: '#3aa0ff',
                backgroundColor: 'rgba(58, 160, 255, 0.1)',
                tension: 0.1,
                fill: true,
                pointRadius: 0,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: 'var(--muted)', display: false }, grid: { display: false } },
                y: { ticks: { color: 'var(--muted)' }, grid: { color: 'rgba(255, 255, 255, 0.05)' } }
            }
        }
    });
}
</script>
</body>
</html>
"""

SETTINGS_TEMPLATE = """
<!DOCTYPE html><html dir="rtl" lang="ar"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>إعدادات البوت</title><link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap" rel="stylesheet"><style>:root{--bg-dark:#121212;--bg-surface:#1e1e1e;--primary:#BB86FC;--primary-variant:#3700B3;--text-light:#e0e0e0;--text-medium:#a0a0a0;}body{background-color:var(--bg-dark);color:var(--text-light);font-family:'Tajawal',sans-serif;}.container{max-width:900px;margin:0 auto;padding:20px;}header{background-color:var(--bg-surface);padding:15px 25px;border-radius:12px;margin-bottom:25px;display:flex;justify-content:space-between;align-items:center;}.header-title{font-size:24px;font-weight:700;color:var(--primary);}.btn{background-color:var(--primary-variant);color:white;border:none;padding:10px 20px;border-radius:8px;cursor:pointer;text-decoration:none;}.settings-form{background-color:var(--bg-surface);border-radius:12px;padding:25px;margin-bottom:20px;}.form-section-title{font-size:20px;font-weight:700;margin-bottom:20px;padding-bottom:10px;border-bottom:1px solid #333;}.form-group{margin-bottom:20px;}.form-group label{display:block;margin-bottom:8px;font-weight:bold;color:var(--text-medium);}.form-group input[type="number"],.form-group select,.form-group input[type="text"]{width:100%;padding:12px;border:1px solid #333;border-radius:8px;background-color:#252525;color:var(--text-light);}.checkbox-group{display:flex;align-items:center;gap:10px;padding:10px;}.filter-table{width:100%;border-collapse:collapse;}.filter-table th,.filter-table td{padding:12px;text-align:right;border-bottom:1px solid #333;}.filter-table select,.filter-table input{width:100%;padding:8px;}</style></head><body><div class="container"><header><div class="header-title">إعدادات البوت</div><a href="/" class="btn">العودة للرئيسية</a></header><div class="settings-form"><h3 class="form-section-title">إعدادات التداول العامة</h3><form id="settings-form"><div class="form-group"><label>نسبة المخاطرة الأساسية للصفقة (%)</label><input type="number" name="risk_per_trade" step="0.1" value="{{RISK_PER_TRADE_PERCENT}}"></div><div class="form-group"><label>الحد الأقصى للصفقات المفتوحة</label><input type="number" name="max_trades" value="{{MAX_OPEN_TRADES}}"></div><button type="submit" class="btn">حفظ الإعدادات</button></form></div><div class="settings-form"><h3 class="form-section-title">تفعيل الاستراتيجيات</h3><form id="strategies-form"><div class="form-group checkbox-group"><input type="checkbox" id="use_bb_stoch" name="use_bb_stoch" {{'checked' if USE_BB_STOCH_STRATEGY else ''}}><label for="use_bb_stoch">BB+Stoch</label></div><div class="form-group checkbox-group"><input type="checkbox" id="use_macd_ema" name="use_macd_ema" {{'checked' if USE_MACD_EMA_STRATEGY else ''}}><label for="use_macd_ema">MACD+EMA</label></div><div class="form-group checkbox-group"><input type="checkbox" id="use_ema_rsi" name="use_ema_rsi" {{'checked' if USE_EMA_RSI_STRATEGY else ''}}><label for="use_ema_rsi">EMA+RSI</label></div><div class="form-group checkbox-group"><input type="checkbox" id="use_pullback" name="use_pullback" {{'checked' if USE_PULLBACK_STRATEGY else ''}}><label for="use_pullback">Pullback</label></div><div class="form-group checkbox-group"><input type="checkbox" id="use_momentum_volatility" name="use_momentum_volatility" {{'checked' if USE_MOMENTUM_VOLATILITY_STRATEGY else ''}}><label for="use_momentum_volatility">Momentum</label></div><button type="submit" class="btn">حفظ الاستراتيجيات</button></form></div><div class="settings-form"><h3 class="form-section-title">إعدادات فلاتر الاستراتيجيات</h3><form id="filters-form"><table class="filter-table"><thead><tr><th>الاستراتيجية</th><th>ملف تعريف الفلتر</th><th>حد ADX</th><th>تأكيد HTF</th></tr></thead><tbody>{%for key, config in STRATEGY_FILTER_CONFIG.items()%}<tr><td>{{STRATEGY_NAMES.get(key,key)}}</td><td><select name="{{key}}_profile"><option value="Strict" {{'selected' if config.profile=='Strict'}}>صارم</option><option value="Moderate" {{'selected' if config.profile=='Moderate'}}>متوسط</option><option value="Reversal" {{'selected' if config.profile=='Reversal'}}>انعكاسي</option><option value="Disabled" {{'selected' if config.profile=='Disabled'}}>معطل</option></select></td><td><input type="number" name="{{key}}_adx_threshold" value="{{config.adx_threshold}}"></td><td><select name="{{key}}_htf_confirmation_mode"><option value="Strict" {{'selected' if config.htf_confirmation_mode=='Strict'}}>صارم</option><option value="Relaxed" {{'selected' if config.htf_confirmation_mode=='Relaxed'}}>مخفف</option><option value="Disabled" {{'selected' if config.htf_confirmation_mode=='Disabled'}}>معطل</option></select></td></tr>{%endfor%}</tbody></table><button type="submit" class="btn">حفظ إعدادات الفلاتر</button></form></div></div><script>function setupForm(formId,url){document.getElementById(formId).addEventListener('submit',function(e){e.preventDefault();const formData=new FormData(this);const data=formId==='strategies-form'?Object.fromEntries([...formData.keys()].map(key=>[key,this.querySelector(`[name=${key}]`).checked])):Object.fromEntries(formData.entries());fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}).then(res=>res.json()).then(data=>alert(data.message));});}
setupForm('settings-form','/update_settings');setupForm('strategies-form','/update_strategies');setupForm('filters-form','/update_filter_settings');</script></body></html>
"""

# --- مسارات Flask ---
@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/backtest')
def backtest_page():
    return render_template_string(BACKTEST_TEMPLATE, STRATEGY_NAMES=STRATEGY_NAMES)


@app.route('/api/dashboard_data')
def dashboard_data():
    try:
        return jsonify(get_dashboard_payload())
    except Exception as e:
        logger.error(f"❌ [API Error] Failed to generate dashboard data: {e}", exc_info=True)
        return jsonify({"error": "Failed to load dashboard data."}), 500

@sock.route('/ws')
def ws(ws_client):
    logger.info("WebSocket client connected.")
    with ws_clients_lock:
        ws_clients.append(ws_client)
    try:
        while True:
            time.sleep(1)
    except Exception:
        logger.info("WebSocket client disconnected.")
    finally:
        with ws_clients_lock:
            if ws_client in ws_clients:
                ws_clients.remove(ws_client)

@app.route('/api/advanced_performance_data')
def advanced_performance_data():
    if not check_db_connection() or not conn:
        return jsonify({"error": "DB connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT profit_percentage, entry_price, stop_loss, target_price_1 FROM signals WHERE status = 'closed'")
            trades = cur.fetchall()

        if not trades:
            return jsonify({
                "winRate": 0, "riskRewardRatio": 0, "maxDrawdown": 0, "profitFactor": 0,
                "drawdown_data": {"labels": [], "values": []}
            })

        profits = [t['profit_percentage'] for t in trades if t['profit_percentage'] is not None]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]

        win_rate = (len(wins) / len(profits) * 100) if profits else 0
        
        total_profit = sum(wins)
        total_loss = abs(sum(losses))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        potential_rewards = [(t['target_price_1'] - t['entry_price']) for t in trades if t['target_price_1'] and t['entry_price']]
        potential_risks = [(t['entry_price'] - t['stop_loss']) for t in trades if t['entry_price'] and t['stop_loss']]
        avg_reward = sum(potential_rewards) / len(potential_rewards) if potential_rewards else 0
        avg_risk = sum(potential_risks) / len(potential_risks) if potential_risks else 0
        risk_reward_ratio = avg_reward / avg_risk if avg_risk > 0 else 0

        equity_curve = [1000]
        for p in profits:
            equity_curve.append(equity_curve[-1] * (1 + p / 100))
        
        peak = equity_curve[0]
        max_drawdown = 0
        drawdown_values = []
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            drawdown_values.append(drawdown)

        return jsonify({
            "winRate": win_rate, "riskRewardRatio": risk_reward_ratio, "maxDrawdown": max_drawdown,
            "profitFactor": profit_factor, "drawdown_data": {"labels": list(range(len(drawdown_values))), "values": drawdown_values}
        })
    except Exception as e:
        logger.error(f"❌ [API] Error fetching advanced performance data: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


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
    return jsonify({"status": "success"})

@app.route('/toggle_real_trading', methods=['POST'])
def toggle_real_trading():
    global paper_trading_mode
    with trading_mode_lock:
        with trading_status_lock:
            if is_trading_enabled and not paper_trading_mode:
                log_and_notify("warning", "Cannot switch to paper mode while real trading is active.", "MODE_SWITCH_FAIL")
                return jsonify({"success": False, "message": "يجب إيقاف البوت أولاً"})
        paper_trading_mode = not paper_trading_mode
        mode_msg = "Paper" if paper_trading_mode else "Real (LIVE)"
        log_and_notify("info", f"Trading mode switched to {mode_msg}.", "TRADING_MODE_SWITCH")
        if redis_client:
            settings = {'RISK_PER_TRADE_PERCENT': RISK_PER_TRADE_PERCENT, 'MAX_OPEN_TRADES': MAX_OPEN_TRADES, 'paper_trading_mode': paper_trading_mode}
            redis_client.set('trading_settings', json.dumps(settings))
    return jsonify({"status": "success"})

@app.route('/close_trade/<int:signal_id>', methods=['POST'])
def manual_close_trade(signal_id):
    signal_to_close = None
    with signal_cache_lock:
        signal_to_close = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)
    if not signal_to_close:
        logger.warning(f"[Manual Close] Signal ID {signal_id} not in cache, querying DB.")
        if not check_db_connection() or not conn:
            return jsonify({"success": False, "message": "لا يمكن الاتصال بقاعدة البيانات."}), 500
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM signals WHERE id = %s AND status IN ('open', 'updated');", (signal_id,))
                signal_from_db = cur.fetchone()
                if signal_from_db: signal_to_close = dict(signal_from_db)
        except Exception as e:
            logger.error(f"❌ [Manual Close] DB query failed for signal {signal_id}: {e}", exc_info=True)
            return jsonify({"success": False, "message": "خطأ أثناء البحث في قاعدة البيانات."}), 500
    if not signal_to_close:
        return jsonify({"success": False, "message": "لم يتم العثور على الصفقة أو أنها مغلقة بالفعل."}), 404
    symbol = signal_to_close['symbol']
    with live_prices_lock: current_price = live_prices.get(symbol)
    if not current_price:
        try:
            ticker = client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            logger.info(f"[Manual Close] Fetched fallback price for {symbol}: {current_price}")
        except Exception as e:
            logger.error(f"❌ [Manual Close] Could not get live price for {symbol}: {e}")
            return jsonify({"success": False, "message": "لا يمكن الحصول على السعر الحالي."}), 500
    try:
        if isinstance(signal_to_close.get('signal_details'), str):
            signal_to_close['signal_details'] = json.loads(signal_to_close['signal_details'])
        elif signal_to_close.get('signal_details') is None:
             signal_to_close['signal_details'] = {}
        close_signal(signal_to_close, current_price, "MANUAL_CLOSE")
        return jsonify({"success": True, "message": f"تم إرسال أمر إغلاق لـ {symbol}"})
    except Exception as e:
        logger.error(f"❌ [Manual Close] Error closing signal {signal_id}: {e}", exc_info=True)
        return jsonify({"success": False, "message": "حدث خطأ أثناء عملية الإغلاق."}), 500

@app.route('/close_all_trades', methods=['POST'])
def close_all_trades_endpoint():
    try:
        with signal_cache_lock:
            open_trades = list(open_signals_cache.values())
        
        for trade in open_trades:
            symbol = trade['symbol']
            with live_prices_lock:
                current_price = live_prices.get(symbol)
            if current_price:
                close_signal(trade, current_price, "MANUAL_CLOSE_ALL")
        
        log_and_notify("info", "All open trades have been closed manually.", "CLOSE_ALL_TRADES")
        return jsonify({"success": True, "message": "تم إغلاق جميع الصفقات."})
    except Exception as e:
        logger.error(f"❌ [Close All] Error: {e}", exc_info=True)
        return jsonify({"success": False, "message": "حدث خطأ."}), 500

@app.route('/emergency_stop', methods=['POST'])
def emergency_stop_endpoint():
    global is_trading_enabled
    with trading_status_lock:
        is_trading_enabled = False
    
    close_all_trades_endpoint()
    log_and_notify("warning", "EMERGENCY STOP ACTIVATED. Trading disabled and all positions closed.", "EMERGENCY_STOP")
    return jsonify({"success": True, "message": "تم تفعيل إيقاف الطوارئ."})


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

# --- نظام اختبار الاستراتيجيات ---
def backtest_strategy(strategy_name, symbol, days=90):
    """
    اختبار استراتيجية على بيانات تاريخية بمنطق محاكاة دقيق.
    """
    logger.info(f"[Backtest] Starting for {strategy_name} on {symbol} for {days} days.")
    df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, days)
    if df is None or len(df) < 50:
        logger.error(f"[Backtest] Insufficient historical data for {symbol}.")
        return {"error": "Insufficient historical data."}
    
    df = calculate_all_features(df)
    
    results = []
    active_trade = None
    initial_balance = 1000
    equity_curve = [initial_balance]

    # استراتيجيات الفحص
    strategy_functions = {
        'BB_Stoch_Strategy': check_bb_stoch_strategy_enhanced,
        'MACD_EMA_Strategy': check_macd_ema_strategy_enhanced,
        'EMA_RSI_Strategy': check_ema_rsi_strategy_enhanced,
        'Pullback_Strategy': check_pullback_strategy_enhanced,
        'Momentum_Volatility_Strategy': check_momentum_volatility_strategy
    }
    check_strategy = strategy_functions.get(strategy_name)
    if not check_strategy:
        return {"error": f"Strategy '{strategy_name}' not found."}

    for i in range(50, len(df)):
        current_candle = df.iloc[i]
        
        # إدارة الصفقة المفتوحة
        if active_trade:
            exit_price = None
            exit_reason = None
            # تحقق من وقف الخسارة
            if current_candle['low'] <= active_trade['stop_loss']:
                exit_price = active_trade['stop_loss']
                exit_reason = 'stop_loss'
            # تحقق من الهدف الأول
            elif current_candle['high'] >= active_trade['target_price_1']:
                exit_price = active_trade['target_price_1']
                exit_reason = 'target_1'
            
            if exit_price:
                profit_percent = ((exit_price - active_trade['entry_price']) / active_trade['entry_price']) * 100
                active_trade['exit_time'] = current_candle.name
                active_trade['exit_price'] = exit_price
                active_trade['profit_percent'] = profit_percent
                active_trade['exit_reason'] = exit_reason
                results.append(active_trade)
                
                # تحديث منحنى رأس المال
                equity_curve.append(equity_curve[-1] * (1 + profit_percent / 100))
                active_trade = None

        # البحث عن إشارة دخول جديدة
        if not active_trade:
            # نستخدم البيانات حتى الشمعة *قبل* الحالية لتجنب الانحياز
            df_slice = df.iloc[:i]
            # ======================= FIX START =======================
            # إضافة اسم العملة إلى الـ DataFrame لضمان عمل الدوال بشكل صحيح
            df_slice.name = symbol
            # ======================== FIX END ========================
            if check_strategy(df_slice):
                trade_levels = calculate_trade_levels(df_slice)
                active_trade = {
                    'entry_time': current_candle.name,
                    'entry_price': current_candle['open'], # الدخول عند افتتاح الشمعة التالية
                    'stop_loss': trade_levels['stop_loss'],
                    'target_price_1': trade_levels['target_price_1'],
                }

    if not results:
        return {"error": "No trades were executed during this period."}

    total_trades = len(results)
    wins = [r for r in results if r['profit_percent'] > 0]
    losses = [r for r in results if r['profit_percent'] <= 0]
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    
    total_profit = sum(r['profit_percent'] for r in wins)
    total_loss = abs(sum(r['profit_percent'] for r in losses))
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    avg_profit = sum(r['profit_percent'] for r in results) / total_trades if total_trades > 0 else 0

    return {
        'strategy': strategy_name, 'symbol': symbol, 'total_trades': total_trades,
        'win_rate': win_rate, 'avg_profit': avg_profit, 'profit_factor': profit_factor,
        'results': results, 'equity_curve': equity_curve
    }

@app.route('/api/run_backtest', methods=['POST'])
def run_backtest():
    try:
        data = request.json
        strategy = data.get('strategy')
        symbol = data.get('symbol', '').upper()
        days = int(data.get('days', 90))

        if not all([strategy, symbol, days]):
            return jsonify({"error": "Missing parameters."}), 400

        results = backtest_strategy(strategy, symbol, days)
        return jsonify(results)
    except Exception as e:
        logger.error(f"❌ [Backtest API] Error: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred."}), 500


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
                if df is None or len(df) < 200:
                    if df is not None: log_rejection(symbol, "Insufficient Historical Data")
                    continue
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
        with signal_cache_lock:
            symbol = next((s['symbol'] for s in open_signals_cache.values() if s['id'] == signal_id), None)
            if symbol and symbol in open_signals_cache:
                open_signals_cache[symbol].update(updates)
                if 'signal_details' in updates and isinstance(updates['signal_details'], str):
                    open_signals_cache[symbol]['signal_details'] = json.loads(updates['signal_details'])
        broadcast_dashboard_update()
        return True
    except Exception as e:
        logger.error(f"❌ [DB] Failed to update signal {signal_id}: {e}")
        if conn: conn.rollback()
        return False

def execute_close_order(symbol: str, quantity: float, reason: str):
    with trading_mode_lock: is_real = not paper_trading_mode
    if is_real:
        try:
            symbol_info = exchange_info_map.get(symbol)
            if not symbol_info:
                logger.error(f"❌ [Execute Close] No exchange info for {symbol}"); return
            step_size = next((f['stepSize'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), '0.000001')
            adjusted_quantity = adjust_quantity_to_step_size(quantity, step_size)
            if adjusted_quantity > 0:
                logger.info(f"💰 [Real Close] Executing MARKET SELL for {adjusted_quantity} of {symbol} due to {reason}")
                client.create_order(symbol=symbol, side=Client.SIDE_SELL, type=Client.ORDER_TYPE_MARKET, quantity=adjusted_quantity)
            else:
                logger.warning(f"⚠️ [Execute Close] Adjusted quantity for {symbol} is zero.")
        except BinanceAPIException as e:
            logger.error(f"❌ [Execute Close] Binance API Error for {symbol}: {e}")
            send_telegram_message(f"❌ *خطأ في تنفيذ إغلاق لـ {symbol}*\n`{e}`", force=True)
        except Exception as e:
            logger.error(f"❌ [Execute Close] CRITICAL ERROR for {symbol}: {e}", exc_info=True)
    else:
        logger.info(f"📊 [Paper Close] Simulating close of {quantity} of {symbol} for reason: {reason}")

def close_signal(signal: Dict, closing_price: float, reason: str):
    symbol, signal_id, entry_price = signal['symbol'], signal['id'], signal['entry_price']
    remaining_quantity = signal.get('quantity', 0)
    if remaining_quantity > 0:
        execute_close_order(symbol, remaining_quantity, reason)
    profit = ((closing_price - entry_price) / entry_price) * 100
    with consecutive_losses_lock:
        if profit < 0:
            consecutive_losses_by_symbol[symbol] = consecutive_losses_by_symbol.get(symbol, 0) + 1
        else:
            consecutive_losses_by_symbol[symbol] = 0
    update_signal_in_db(signal_id, {"status": "closed", "closing_price": closing_price, "closed_at": datetime.now(timezone.utc), "profit_percentage": profit, "closing_reason": reason})
    with signal_cache_lock:
        if symbol in open_signals_cache: del open_signals_cache[symbol]
    broadcast_dashboard_update()
    trade_type = "حقيقية" if signal.get('is_real_trade') else "ورقية"
    result_emoji = "✅" if profit >= 0 else "🔻"
    reason_map = {
        "SL_HIT": "ضرب وقف الخسارة", "TP1_HIT": "تحقيق الهدف الأول", "TP2_HIT": "تحقيق الهدف الثاني",
        "MANUAL_CLOSE": "إغلاق يدوي", "TRAILING_SL_HIT": "ضرب الوقف المتحرك",
        "stop_loss": "ضرب وقف الخسارة", "target_2_reached": "تحقيق الهدف الثاني",
        "support_broken": "كسر الدعم (إشارة معاكسة)"
    }
    reason_ar = reason_map.get(reason, reason)
    log_and_notify("info", f"Closed {trade_type} trade for {symbol}. Profit: {profit:.2f}%", "TRADE_CLOSED")
    settings = get_notification_settings()
    profit_condition = profit >= settings['min_profit_notification']
    loss_condition = profit <= settings['max_loss_notification']
    if profit_condition or loss_condition:
        send_telegram_message(f"{result_emoji} *إغلاق صفقة {trade_type} {symbol}*\n*السبب:* {reason_ar}\n*الربح:* `{profit:.2f}%`")

def advanced_notification_system(signal, current_price, df):
    entry_price = signal['entry_price']
    stop_loss = signal['stop_loss']
    target_price_1 = signal['target_price_1']
    signal_details = signal.get('signal_details', {})
    if isinstance(signal_details, str):
        try: signal_details = json.loads(signal_details)
        except (json.JSONDecodeError, TypeError): signal_details = {}
    profit_percent = ((current_price - entry_price) / entry_price) * 100
    tp1_percent = ((target_price_1 - entry_price) / entry_price) * 100
    if tp1_percent > 0 and profit_percent >= (tp1_percent * 0.5) and not signal_details.get('notified_50_tp1', False):
        message = f"📈 {signal['symbol']} حقق 50% من الهدف الأول ({profit_percent:.2f}%)"
        send_telegram_message(message)
        signal_details['notified_50_tp1'] = True
    if tp1_percent > 0 and profit_percent >= (tp1_percent * 0.8) and not signal_details.get('notified_80_tp1', False):
        message = f"📈 {signal['symbol']} حقق 80% من الهدف الأول ({profit_percent:.2f}%)"
        send_telegram_message(message)
        signal_details['notified_80_tp1'] = True
    distance_to_sl = ((current_price - stop_loss) / current_price) * 100
    if distance_to_sl <= 0.5 and not signal_details.get('notified_sl_approaching', False):
        message = f"⚠️ {signal['symbol']} يقترب من وقف الخسارة (مسافة: {distance_to_sl:.2f}%)"
        send_telegram_message(message)
        signal_details['notified_sl_approaching'] = True
    strategy_name = signal['strategy_name']
    if strategy_name == 'BB_Stoch_Strategy':
        if current_price < df['bb_lower'].iloc[-1] and not signal_details.get('notified_bb_broken', False):
            message = f"🔴 {signal['symbol']} كسر دعم بولينجر السفلي"
            send_telegram_message(message)
            signal_details['notified_bb_broken'] = True
    return signal_details


def enhanced_exit_strategy(signal, current_price, df):
    entry_price = signal['entry_price']
    stop_loss = signal['stop_loss']
    target_price_1 = signal.get('target_price_1')
    target_price_2 = signal.get('target_price_2')
    quantity = signal['quantity']
    signal_details = signal.get('signal_details', {})
    if isinstance(signal_details, str):
        try: signal_details = json.loads(signal_details)
        except (json.JSONDecodeError, TypeError): signal_details = {}
    if current_price > entry_price * 1.015:
        new_stop_loss = entry_price * 1.005
        if new_stop_loss > stop_loss:
            stop_loss = new_stop_loss
            signal_details['trailing_stop_activated'] = True
    if current_price > entry_price * 1.03:
        atr = df['atr'].iloc[-1]
        new_stop_loss = current_price - (atr * 1.2)
        if new_stop_loss > stop_loss:
            stop_loss = new_stop_loss
            signal_details['trailing_stop_distance'] = atr * 1.2
    if target_price_1 and current_price >= target_price_1 and not signal_details.get('tp1_done', False):
        exit_quantity = quantity * 0.5
        remaining_quantity = quantity - exit_quantity
        signal_details['tp1_done'] = True
        new_stop_loss_after_tp1 = entry_price
        return {
            'action': 'partial_exit', 'quantity': exit_quantity, 'remaining_quantity': remaining_quantity,
            'new_stop_loss': new_stop_loss_after_tp1, 'updated_signal_details': signal_details
        }
    if (target_price_2 and current_price >= target_price_2) or current_price <= stop_loss:
        return {
            'action': 'full_exit', 'quantity': quantity,
            'reason': 'target_2_reached' if target_price_2 and current_price >= target_price_2 else 'stop_loss'
        }
    strategy_name = signal['strategy_name']
    if strategy_name == 'BB_Stoch_Strategy':
        if current_price < df['bb_lower'].iloc[-1]:
            return {'action': 'full_exit', 'quantity': quantity, 'reason': 'support_broken'}
    return {'action': 'hold', 'new_stop_loss': stop_loss, 'updated_signal_details': signal_details}

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
                df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, days=3)
                if df is None or df.empty:
                    logger.warning(f"[Trade Manager] Could not fetch data for {symbol} to make an exit decision."); continue
                df_featured = calculate_all_features(df)
                updated_details = advanced_notification_system(signal, current_price, df_featured)
                if updated_details != signal.get('signal_details'):
                    update_signal_in_db(signal['id'], {"signal_details": json.dumps(updated_details, cls=NpEncoder)})
                    signal['signal_details'] = updated_details
                exit_decision = enhanced_exit_strategy(signal, current_price, df_featured)
                action = exit_decision.get('action')
                if action == 'full_exit':
                    logger.info(f"[Exit Strategy] FULL EXIT for {symbol} triggered. Reason: {exit_decision.get('reason')}")
                    close_signal(signal, current_price, exit_decision.get('reason', 'STRATEGY_EXIT'))
                    if exit_decision.get('reason') == 'stop_loss':
                        with cooldowns_lock:
                            cooldowns_by_symbol[symbol] = datetime.now(timezone.utc) + timedelta(minutes=COOLDOWN_MINUTES_AFTER_SL)
                elif action == 'partial_exit':
                    logger.info(f"[Exit Strategy] PARTIAL EXIT for {symbol} triggered at TP1.")
                    execute_close_order(symbol, exit_decision['quantity'], "TP1_PARTIAL")
                    updates = {
                        "quantity": exit_decision['remaining_quantity'],
                        "stop_loss": exit_decision['new_stop_loss'],
                        "signal_details": json.dumps(exit_decision['updated_signal_details'], cls=NpEncoder)
                    }
                    if update_signal_in_db(signal['id'], updates):
                        msg = f"🎯 *جني ربح جزئي {symbol}*\nتم إغلاق 50% ونقل الوقف للدخول."
                        log_and_notify("info", f"Partial profit taken for {symbol} at TP1.", "TP1_HIT")
                        send_telegram_message(msg, force=True)
                elif action == 'hold':
                    new_stop_loss = exit_decision.get('new_stop_loss')
                    if new_stop_loss and new_stop_loss > signal['stop_loss']:
                        updates = {
                            "stop_loss": new_stop_loss,
                            "signal_details": json.dumps(exit_decision.get('updated_signal_details', {}), cls=NpEncoder)
                        }
                        if update_signal_in_db(signal['id'], updates):
                            logger.info(f"Updated trailing stop for {signal['symbol']} to {new_stop_loss:.4f}")
            time.sleep(2)
        except Exception as e:
            logger.error(f"❌ [Trade Manager] A critical error occurred: {e}", exc_info=True)
            time.sleep(10)

def multi_trade_management():
    global open_signals_cache
    with signal_cache_lock:
        open_trades = list(open_signals_cache.values())
    total_risk = sum(
        (trade['entry_price'] - trade['stop_loss']) * trade['quantity']
        for trade in open_trades if trade.get('status') == 'open'
    )
    with balance_lock: current_balance = usdt_balance
    with risk_per_trade_lock: risk_per_trade = RISK_PER_TRADE_PERCENT
    max_total_risk = current_balance * (risk_per_trade / 100) * MAX_OPEN_TRADES
    if total_risk > max_total_risk:
        logger.warning(f"[Risk Mgmt] Total risk {total_risk:.2f} exceeds max {max_total_risk:.2f}. Closing weakest trades.")
        with live_prices_lock: live_prices_copy = dict(live_prices)
        sorted_trades = sorted(
            open_trades,
            key=lambda t: ((live_prices_copy.get(t['symbol'], t['entry_price']) / t['entry_price']) - 1)
        )
        for trade in sorted_trades:
            if total_risk <= max_total_risk: break
            with live_prices_lock: current_price = live_prices.get(trade['symbol'])
            if current_price:
                log_and_notify("warning", f"Closing {trade['symbol']} due to overall risk management.", "RISK_MGMT_CLOSE")
                close_signal(trade, current_price, "Risk management")
                trade_risk = (trade['entry_price'] - trade['stop_loss']) * trade['quantity']
                total_risk -= trade_risk
    
def risk_management_loop():
    logger.info("🚀 [Risk Manager] Starting multi-trade risk management loop...")
    while True:
        try:
            multi_trade_management()
        except Exception as e:
            logger.error(f"❌ [Risk Manager] Error: {e}", exc_info=True)
        time.sleep(60 * 2)

def update_market_state():
    try:
        trend_details = {}
        for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
            btc_df = fetch_historical_data(BTC_SYMBOL, tf, 30)
            if btc_df is None or btc_df.empty:
                trend_details[tf] = {"trend": "Unknown", "rsi": "N/A"}; continue
            btc_df_featured = calculate_all_features(btc_df)
            if 'rsi' not in btc_df_featured.columns:
                trend_details[tf] = {"trend": "Unknown", "rsi": "N/A"}; continue
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
    logger.info("="*50 + "\n====== Starting Crypto Trading Bot V21.0.0 ======\n" + "="*50)
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

    logger.info("Performing initial data fetch for dashboard...")
    update_market_state()
    update_balance()
    logger.info("Initial data fetch complete.")

    start_websocket()
    Thread(target=main_bot_loop, daemon=True).start()
    Thread(target=trade_management_loop, daemon=True).start()
    Thread(target=update_market_state_loop, daemon=True).start()
    Thread(target=update_balance_loop, daemon=True).start()
    Thread(target=risk_management_loop, daemon=True).start()

    logger.info("🌐 [Flask] Starting UI on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
