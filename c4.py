# ملف c4.py - نسخة V23.2.5 (إصلاح LOT_SIZE وإضافة التحكم بحجم الصفقة)
# --- وصف الإصدار:
# 1.  [إصلاح] معالجة خطأ Binance API (code=-1013): Filter failure: LOT_SIZE عن طريق تعديل الكمية لتتوافق مع `stepSize`.
# 2.  [إضافة] متغير جديد `FIXED_TRADE_SIZE_USDT` بقيمة افتراضية 5.0 دولار للتحكم بحجم الصفقة.
# 3.  [تحديث] إضافة حقل في لوحة التحكم لتغيير حجم الصفقة بسهولة.
# 4.  [تحسين] إضافة فلتر للتحقق من `minNotional` لتجنب الصفقات الصغيرة جداً.
# 5.  [نتيجة] أصبح البوت الآن أكثر موثوقية للحسابات ذات الأرصدة الصغيرة ويوفر تحكمًا أفضل للمستخدم.

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
        logging.FileHandler('crypto_bot_v23_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV23')

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
RISK_PER_TRADE_PERCENT: float = 0.85 # يستخدم فقط في حالة عدم استخدام حجم الصفقة الثابت
risk_per_trade_lock = Lock()
MAX_OPEN_TRADES: int = 3
FIXED_TRADE_SIZE_USDT: float = 5.0 # [جديد] حجم الصفقة الثابت بالدولار
fixed_trade_size_lock = Lock()
TRAILING_STOP_ACTIVATION_PROFIT_PERCENT: float = 1.4
MIN_SIGNAL_QUALITY: int = 60
AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE: bool = True

min_quality_lock = Lock()


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
    "LOT_SIZE Filter Failed": "فشل تعديل حجم الصفقة",
    "Insufficient Balance": "الرصيد غير كافي لتنفيذ الصفقة",
    "Bullish Confirmation Failed": "فشل تأكيد الشمعة الصعودية",
    "Volume Filter Failed": "فلتر حجم التداول فشل",
    "MACD Momentum Failed": "فلتر زخم الماكد فشل",
    "Long-term Trend Filter Failed": "فلتر الاتجاه طويل الأجل فشل",
    "Low Quality Signal": "جودة الإشارة منخفضة"
}

# --- إعداد تطبيق Flask و WebSocket ---
app = Flask(__name__)
CORS(app)
sock = Sock(app)
ws_clients: List[Any] = []
ws_clients_lock = Lock()

# --- دوال WebSocket ---
def broadcast(data: Dict):
    with ws_clients_lock:
        clients_to_remove = []
        for client in ws_clients:
            try:
                client.send(json.dumps(data, cls=NpEncoder))
            except Exception as e:
                logger.warning(f"WebSocket send failed, removing client: {e}")
                clients_to_remove.append(client)
        
        for client in clients_to_remove:
            try:
                ws_clients.remove(client)
            except ValueError:
                pass

def get_dashboard_payload() -> Dict:
    with trading_status_lock: trading_enabled = is_trading_enabled
    with trading_mode_lock: is_paper_mode = paper_trading_mode
    with balance_lock: current_balance = usdt_balance
    with notifications_lock: notifications = list(notifications_cache)
    with rejection_logs_lock: rejections = list(rejection_logs_cache)
    with market_state_lock: market_state = dict(current_market_state)
    with min_quality_lock: min_quality = MIN_SIGNAL_QUALITY
    with fixed_trade_size_lock: trade_size = FIXED_TRADE_SIZE_USDT

    return {
        "trading_enabled": trading_enabled, 
        "paper_trading_mode": is_paper_mode,
        "usdt_balance": current_balance,
        "notifications": notifications, 
        "rejections": rejections, 
        "market_state": market_state,
        "min_signal_quality": min_quality,
        "fixed_trade_size": trade_size,
        "server_time": datetime.now(timezone.utc).isoformat()
    }

# --- دوال تهيئة الخدمات وقاعدة البيانات ---
def optimize_database():
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
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS performance_summary (
                        id SERIAL PRIMARY KEY,
                        trade_id INTEGER REFERENCES signals(id),
                        profit_percentage DOUBLE PRECISION,
                        drawdown DOUBLE PRECISION,
                        date DATE
                    );
                """)
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
        broadcast({"type": "new_notification", "payload": new_notification})
    except Exception as e:
        logger.error(f"❌ [DB] Failed to save notification: {e}")
        if conn: conn.rollback()

def log_rejection(symbol: str, reason_key: str, details: Optional[Dict] = None):
    try:
        reason_ar = REJECTION_REASONS_AR.get(reason_key, reason_key)
        if details:
            details_str = ", ".join([f"{k}: {v}" for k, v in details.items()])
            reason_ar = f"{reason_ar} ({details_str})"
        log_entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol, "reason": reason_ar}
        with rejection_logs_lock: rejection_logs_cache.appendleft(log_entry)
        broadcast({"type": "new_rejection", "payload": log_entry})
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

# ======================= START: تحسين وظيفة إرسال رسائل Telegram =======================
def send_telegram_message(message: str, force: bool = False):
    """
    إرسال رسالة إلى Telegram مع تحسينات
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    
    settings = get_notification_settings()
    if not settings.get('telegram_enabled') and not force:
        return
    
    # تقسيم الرسائل الطويلة
    max_length = 4096
    if len(message) <= max_length:
        messages = [message]
    else:
        messages = [message[i:i+max_length] for i in range(0, len(message), max_length)]
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    for msg in messages:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        
        for attempt in range(3):
            try:
                r = requests.post(url, data=payload, timeout=10)
                if r.status_code == 429:
                    try:
                        retry_after = int(r.json().get("parameters", {}).get("retry_after", 1))
                    except Exception:
                        retry_after = 1
                    time.sleep(min(5, retry_after))
                    continue
                if r.ok:
                    break
                else:
                    logger.warning(f"[Telegram] HTTP {r.status_code}: {r.text}")
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    logger.error(f"❌ [Telegram] Failed to send message after retries: {e}")
                time.sleep(1.5)
# ======================== END: تحسين وظيفة إرسال رسائل Telegram ========================

# ======================= START: تحسين إدارة الأخطاء في WebSocket =======================
def handle_socket_message(msg):
    """
    معالجة رسائل WebSocket مع تحسين إدارة الأخطاء
    """
    global live_prices
    try:
        if msg and 'e' in msg and msg['e'] == 'error': 
            logger.error(f"❌ [WebSocket] Error: {msg['m']}")
            return
        
        if isinstance(msg, list):
            price_updates = {}
            with live_prices_lock:
                for ticker in msg:
                    if 's' in ticker and 'c' in ticker: 
                        symbol = ticker['s']
                        try:
                            price = float(ticker['c'])
                            live_prices[symbol] = price
                            price_updates[symbol] = price
                        except (ValueError, TypeError):
                            logger.warning(f"[WebSocket] Invalid price data for {symbol}: {ticker.get('c')}")
            
            if price_updates:
                broadcast({"type": "price_update", "payload": price_updates})
    except Exception as e:
        logger.error(f"❌ [WebSocket] Error processing message: {e}", exc_info=True)
# ======================== END: تحسين إدارة الأخطاء في WebSocket ========================

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
    df_calc['ema200'] = df_calc['close'].ewm(span=200, adjust=False).mean()
    
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    df_calc['atr'] = tr.ewm(span=14, adjust=False).mean()
    df_calc['atr_percent'] = (df_calc['atr'] / df_calc['close']) * 100
    
    up_move = df_calc['high'].diff()
    down_move = -df_calc['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=14, adjust=False).mean()
    
    delta = df_calc['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df_calc['rsi'] = 100 - (100 / (1 + rs))
    
    rsi_val = df_calc['rsi']
    stoch_rsi = (rsi_val - rsi_val.rolling(14).min()) / (rsi_val.rolling(14).max() - rsi_val.rolling(14).min()).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi.rolling(3).mean() * 100
    
    bb_middle = df_calc['close'].rolling(window=20).mean()
    bb_std = df_calc['close'].rolling(window=20).std()
    df_calc['bb_lower'] = bb_middle - (bb_std * 2)
    df_calc['bb_upper'] = bb_middle + (bb_std * 2)
    
    exp1 = df_calc['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_calc['close'].ewm(span=26, adjust=False).mean()
    df_calc['macd'] = exp1 - exp2
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()
    df_calc['macd_hist'] = df_calc['macd'] - df_calc['macd_signal']
    
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
    global RISK_PER_TRADE_PERCENT, MAX_OPEN_TRADES, USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY, USE_MOMENTUM_VOLATILITY_STRATEGY, STRATEGY_FILTER_CONFIG, paper_trading_mode, MIN_SIGNAL_QUALITY, FIXED_TRADE_SIZE_USDT
    if not redis_client: return
    try:
        settings_data = redis_client.get('trading_settings')
        if settings_data:
            settings = json.loads(settings_data)
            with risk_per_trade_lock: RISK_PER_TRADE_PERCENT = settings.get('RISK_PER_TRADE_PERCENT', 0.85)
            MAX_OPEN_TRADES = settings.get('MAX_OPEN_TRADES', 3)
            with trading_mode_lock: paper_trading_mode = settings.get('paper_trading_mode', True)
            with fixed_trade_size_lock: FIXED_TRADE_SIZE_USDT = settings.get('FIXED_TRADE_SIZE_USDT', 5.0)

        quality_settings_data = redis_client.get('signal_quality_settings')
        if quality_settings_data:
            quality_settings = json.loads(quality_settings_data)
            with min_quality_lock: MIN_SIGNAL_QUALITY = quality_settings.get('min_quality', 60)

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
    with risk_per_trade_lock: 
        base_risk = RISK_PER_TRADE_PERCENT
    if df.empty or 'atr_percent' not in df.columns or len(df) < 14:
        logger.warning(f"[Risk] Insufficient data for dynamic risk calculation for {symbol}")
        return base_risk
    try:
        atr_percent = df['atr_percent'].iloc[-1]
        if atr_percent > 6.0: volatility_factor = 0.6
        elif atr_percent > 4.0: volatility_factor = 0.8
        elif atr_percent < 1.0: volatility_factor = 1.3
        elif atr_percent < 1.5: volatility_factor = 1.2
        else: volatility_factor = 1.0
        loss_factor = max(0.3, 1.0 - (consecutive_losses * 0.12))
        final_risk = base_risk * volatility_factor * loss_factor
        final_risk_percent = min(final_risk, 2.5)
        logger.info(f"[Risk] Dynamic risk for {symbol}: {final_risk_percent:.2f}% (Base: {base_risk}%, Vol: {volatility_factor:.2f}, Loss: {loss_factor:.2f})")
        return final_risk_percent
    except Exception as e:
        logger.error(f"[Risk] Error calculating dynamic risk for {symbol}: {e}")
        return base_risk

# --- Filters & Strategies ---
def calculate_signal_quality_score(symbol, df, strategy_name):
    score = 0
    if df.empty or len(df) < 50: 
        return 0
    
    last_row = df.iloc[-1]
    
    adx_value = last_row.get('adx', 0)
    if adx_value > 35: score += 25
    elif adx_value > 25: score += 20
    elif adx_value > 18: score += 15
    elif adx_value > 12: score += 10
    else: score += 5
    
    current_volume = last_row.get('volume', 0)
    volume_ma = df['volume'].rolling(20, min_periods=5).mean().iloc[-1]
    volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1
    
    if volume_ratio > 2.0: score += 15
    elif volume_ratio > 1.5: score += 12
    elif volume_ratio > 1.2: score += 8
    elif volume_ratio > 1.0: score += 5
    
    rsi = last_row.get('rsi', 50)
    macd_hist = last_row.get('macd_hist', 0)
    
    if 45 <= rsi <= 55 and macd_hist > 0: score += 20
    elif 40 <= rsi <= 60: score += 15
    elif 35 <= rsi <= 65: score += 10
    elif 30 <= rsi <= 70: score += 5
    
    ema9, ema21, ema50, ema200, close = last_row.get('ema9',0), last_row.get('ema21',0), last_row.get('ema50',0), last_row.get('ema200',0), last_row.get('close',0)
    
    if close > ema9 > ema21 > ema50 > ema200: score += 20
    elif close > ema9 > ema21 > ema50: score += 18
    elif close > ema9 > ema21: score += 15
    elif close > ema21: score += 10
    elif close > ema50: score += 5
    
    atr_percent = last_row.get('atr_percent', 0)
    if 2.0 <= atr_percent <= 4.0: score += 10
    elif 1.5 <= atr_percent <= 5.0: score += 8
    elif 1.0 <= atr_percent <= 6.0: score += 5
    
    if strategy_name == "Momentum_Volatility_Strategy" and adx_value > 30: score += 10
    elif strategy_name == "BB_Stoch_Strategy" and last_row.get('stoch_rsi_k', 50) < 25: score += 8
    elif strategy_name == "MACD_EMA_Strategy" and macd_hist > 0: score += 7
    
    return min(100, max(0, int(score)))

def dynamic_adx_threshold(symbol, df, base_threshold=20):
    atr_percent = df['atr_percent'].iloc[-1]
    if atr_percent > 4.0: return base_threshold * 0.85
    elif atr_percent < 1.5: return base_threshold * 1.15
    else: return base_threshold

def flexible_volume_filter(df, min_volume_percentile=30, strictness=0.8):
    if 'volume' not in df.columns or len(df) < 50: return False
    current_volume = df['volume'].iloc[-1]
    volume_ma = df['volume'].rolling(20, min_periods=20).mean().iloc[-1]
    volume_percentile = df['volume'].rolling(50, min_periods=50).quantile(min_volume_percentile / 100).iloc[-1]
    if pd.isna(current_volume) or pd.isna(volume_ma) or pd.isna(volume_percentile): return False
    volume_threshold = (volume_ma * strictness) + (volume_percentile * (1 - strictness))
    return current_volume > volume_threshold

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
    dynamic_threshold = dynamic_adx_threshold(df.name, df, base_threshold=adx_threshold)
    if recent_adx < (dynamic_threshold * 0.95):
        log_rejection(df.name, "Trend Strength Filter Failed"); return False
    return True

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
    if len(df) < 21 or not {'bb_lower', 'stoch_rsi_k', 'rsi', 'open', 'close', 'high', 'low', 'macd', 'macd_signal'}.issubset(df.columns): return False
    if not flexible_volume_filter(df, min_volume_percentile=30, strictness=0.7):
        log_rejection(df.name, "Volume Filter Failed"); return False
    last, prev = df.iloc[-1], df.iloc[-2]
    bounce = (prev['close'] < prev['bb_lower']) and (last['close'] > last['bb_lower'])
    stoch_rising = last['stoch_rsi_k'] > prev['stoch_rsi_k']
    bullish_body = last['close'] > (last['open'] + (last['high'] - last['low']) * 0.3)
    rsi_improving = last['rsi'] > prev['rsi']
    not_overbought = last['rsi'] < 70
    macd_ok = (last['macd'] > last['macd_signal']) or (last['macd'] - last['macd_signal'] > -0.1 * abs(last['macd']))
    signal = bounce and (stoch_rising or bullish_body) and rsi_improving and not_overbought and macd_ok
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
    needed = {'atr_percent','ema9','ema21','macd','macd_signal','close', 'ema200'}
    if len(df) < 200 or not needed.issubset(df.columns): return False
    if df['close'].iloc[-1] < df['ema200'].iloc[-1]: 
        log_rejection(df.name, "Long-term Trend Filter Failed"); return False
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
    try:
        step = Decimal(step_size)
        if step <= 0:
            logger.warning(f"Invalid step size: {step_size}")
            return quantity
        
        decimal_quantity = Decimal(str(quantity))
        adjusted = decimal_quantity.quantize(step, rounding=ROUND_DOWN)
        
        return float(adjusted)
    except Exception as e:
        logger.error(f"Error adjusting quantity: {e}")
        return quantity

# ======================= START: [FIX] Refactored Trade Creation Logic =======================
def create_trade_signal(symbol: str, df: pd.DataFrame, strategy_name: str):
    # --- 1. Pre-Trade Quality & Cooldown Checks ---
    try:
        quality_score = calculate_signal_quality_score(symbol, df, strategy_name)
        with min_quality_lock:
            min_score = MIN_SIGNAL_QUALITY
        
        if quality_score < min_score:
            log_rejection(symbol, "Low Quality Signal", {"score": quality_score, "min_required": min_score})
            return
        logger.info(f"⭐ [Signal Quality] {symbol} ({strategy_name}): {quality_score}/100")
        
        with cooldowns_lock: 
            until = cooldowns_by_symbol.get(symbol)
            if until and datetime.now(timezone.utc) < until:
                log_rejection(symbol, "Cooldown Active", {"until": until.isoformat()})
                return
    except Exception as e:
        logger.error(f"❌ [Signal Creation] Error during pre-checks for {symbol}: {e}", exc_info=True)
        return

    # --- 2. Setup Trade Parameters ---
    with trading_mode_lock: is_real = not paper_trading_mode
    trade_levels = calculate_trade_levels(df)
    entry_price = trade_levels['entry_price']
    
    signal_details = {
        "atr": trade_levels['atr'], "trailing_stop_activated": False,
        "trailing_stop_distance": trade_levels['trailing_stop_distance'], "tp1_done": False,
        "quality_score": quality_score
    }

    # --- 3. Execute Trade (Real or Paper) ---
    if is_real:
        with balance_lock: current_usdt_balance = usdt_balance
        with fixed_trade_size_lock: trade_size_usdt = FIXED_TRADE_SIZE_USDT

        # 3.1. Balance Check
        if trade_size_usdt > current_usdt_balance:
            log_rejection(symbol, "Insufficient Balance", {"required": trade_size_usdt, "available": round(current_usdt_balance, 2)})
            return

        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info:
            logger.error(f"❌ [Real Trade] Could not find exchange info for {symbol}")
            return

        # 3.2. MinNotional Check
        min_notional = 0.0
        for f in symbol_info.get('filters', []):
            if f.get('filterType') in ('NOTIONAL', 'MIN_NOTIONAL') and 'minNotional' in f:
                try:
                    min_notional = float(f['minNotional'])
                    break
                except (ValueError, TypeError): pass
        
        if trade_size_usdt < min_notional:
            logger.warning(f"⚠️ [Real Trade] Trade size ${trade_size_usdt} for {symbol} is below minNotional of ${min_notional}. Trade rejected.")
            log_rejection(symbol, "MinNotional Filter Failed", {"trade_size": trade_size_usdt, "min_notional": min_notional})
            return
        
        # 3.3. Quantity Calculation & LOT_SIZE Adjustment
        quantity = trade_size_usdt / entry_price
        step_size = next((f['stepSize'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), '0.000001')
        adjusted_quantity = adjust_quantity_to_step_size(quantity, step_size)

        if adjusted_quantity <= 0:
            logger.error(f"❌ [Real Trade] Adjusted quantity for {symbol} is 0 after applying LOT_SIZE filter. Trade rejected.")
            log_rejection(symbol, "LOT_SIZE Filter Failed", {"original_qty": quantity, "adjusted_qty": adjusted_quantity})
            return

        # 3.4. Place Order
        try:
            logger.info(f"💰 [Real Trade] Placing LIVE MARKET BUY order for {adjusted_quantity} of {symbol} (approx ${trade_size_usdt})")
            order = client.create_order(
                symbol=symbol,
                side=Client.SIDE_BUY,
                type=Client.ORDER_TYPE_MARKET,
                quantity=adjusted_quantity
            )
            
            avg_fill_price = sum(float(f['price']) * float(f['qty']) for f in order.get('fills', [])) / max(sum(float(f['qty']) for f in order.get('fills', [])), 1e-8) if order.get('fills') else entry_price
            final_quantity = float(order.get('executedQty', adjusted_quantity))
            order_id = order.get('orderId', 'N/A')

            save_signal_to_db(
                symbol, avg_fill_price, trade_levels,
                strategy_name, True, final_quantity,
                {**signal_details, "avg_fill": avg_fill_price, "trade_value_usdt": trade_size_usdt}, order_id
            )
            send_telegram_message(
                f"✅ *تم فتح صفقة حقيقية*\n"
                f"*العملة:* `{symbol}`\n"
                f"*الكمية:* `{final_quantity:.6f}`\n"
                f"*السعر:* `{avg_fill_price:.4f}`\n"
                f"*القيمة:* `~${trade_size_usdt:.2f}`",
                force=True
            )
            log_and_notify("info", f"Opened REAL trade for {symbol}", "REAL_TRADE_OPEN")
            return

        except BinanceAPIException as e:
            logger.error(f"❌ [Real Trade] Binance API Error for {symbol}: {e}")
            send_telegram_message(f"❌ *خطأ باينانس أثناء فتح صفقة {symbol}*\n`{e}`", force=True)
            return
        except Exception as e:
            logger.error(f"❌ [Real Trade] Failed to place order for {symbol}: {e}", exc_info=True)
            return

    else: # Paper Trading
        with fixed_trade_size_lock: trade_size_usdt = FIXED_TRADE_SIZE_USDT
        quantity = trade_size_usdt / entry_price
        save_signal_to_db(symbol, entry_price, trade_levels, strategy_name, False, quantity, signal_details)
        message = (f"📊 *صفقة ورقية جديدة*\n`{symbol}` | `{strategy_name}`\n*الجودة:* `{quality_score}/100`\n*دخول:* `{entry_price:.4f}`\n*قيمة الصفقة:* `${trade_size_usdt:.2f}`\n*وقف:* `{trade_levels['stop_loss']:.4f}`")
        send_telegram_message(message, force=True)
        log_and_notify("info", f"Opened paper trade for {symbol}", "PAPER_TRADE_OPEN")
# ======================== END: [FIX] Refactored Trade Creation Logic ========================

def save_signal_to_db(symbol: str, entry_price: float, trade_levels: Dict, strategy_name: str, is_real: bool, quantity: float, signal_details: Dict, order_id: Optional[str] = None):
    try:
        if not (check_db_connection() and conn): return
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
        broadcast({"type": "new_signal", "payload": signal_data})
    except Exception as e:
        logger.error(f"❌ [DB] CRITICAL ERROR saving signal for {symbol}: {e}", exc_info=True)
        if conn: conn.rollback()

# --- قوالب HTML ---
# ======================= START: تحديث واجهة المستخدم =======================
DASHBOARD_TEMPLATE = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>لوحة التحكم - بوت التداول (V23 - تفاعلي)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
:root{--bg:#0b1020;--panel:#121b36;--accent:#3aa0ff;--ok:#15c46a;--warn:#ff9f1a;--bad:#ff4757;--muted:#8aa0c8;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:#e8f1ff;font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,"Noto Sans",Arial}
.container{max-width:1600px;margin:0 auto;padding:16px;display:flex;flex-direction:column;gap:16px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between}
h1{font-size:18px;margin:0;font-weight:700;color:#d7e4ff}
.badge{padding:6px 10px;border-radius:999px;font-size:12px;background:#0d1730;border:1px solid #1e2c52;color:#cce0ff}
.main-layout{display:grid;grid-template-columns:1fr;gap:16px;}
@media(min-width: 1000px){.main-layout{grid-template-columns:1fr 350px;}}
.left-column{display:flex;flex-direction:column;gap:16px}
.right-column{display:flex;flex-direction:column;gap:16px}
.card{background:var(--panel);border:1px solid #1e2c52;border-radius:14px;box-shadow:0 8px 30px rgba(0,0,0,.25);overflow:hidden}
.card h2{margin:0;padding:12px 14px;border-bottom:1px solid #1e2c52;font-size:14px;color:#cfe2ff; display: flex; justify-content: space-between; align-items: center;}
.card-body{padding:12px}
.controls{display:flex;gap:8px;flex-wrap:wrap}
.btn{appearance:none;border:1px solid #2a3a68;background:#0f1b3b;color:#d9e7ff;padding:10px 14px;border-radius:10px;cursor:pointer;font-weight:700;transition: background-color 0.2s, transform 0.2s; will-change: transform; text-decoration: none;}
.btn:hover{transform:translateY(-1px);border-color:#3a58a6}
.btn.warn{background:linear-gradient(180deg,#3b2a0f,#291b08);border-color:#8b5b0f}
.btn.small{padding: 6px 10px; font-size: 12px;}
.signals-grid{display:grid;grid-template-columns:repeat(auto-fill, minmax(300px, 1fr));gap:10px; contain: layout style paint;}
.signal{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;padding:10px;border:1px solid #24335f;border-radius:12px;background:#0d1730; will-change: transform, opacity; transition: transform 0.2s ease, opacity 0.2s ease; grid-template-rows: auto auto;}
.signal > *:nth-child(1) { grid-column: 1 / 2; }
.signal > *:nth-child(2) { grid-column: 2 / 3; grid-row: 1 / 3; }
.signal > *:nth-child(3) { grid-column: 1 / 2; }
.sig-title{font-weight:700}
.sig-meta{font-size:12px;color:var(--muted)}
.price{font-variant-numeric:tabular-nums;direction:ltr; transition: color 0.3s, background-color 0.3s; font-size: 16px; font-weight: bold;}
.price.flash-up{background-color:rgba(21, 196, 106, 0.2); color: #15c46a;}
.price.flash-down{background-color:rgba(255, 71, 87, 0.2); color: #ff4757;}
.progress{height:8px;background:#0b1126;border:1px solid #233056;border-radius:999px;overflow:hidden; margin-top: 6px;}
.progress>span{display:block;height:100%;}
.kv{display:grid;grid-template-columns:auto 1fr;gap:6px 10px; align-items: center;}
.kv div:nth-child(odd){opacity:.8}
.trend{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}
.trend .pill{background:#0d1730;border:1px solid #1f2d55;border-radius:10px;padding:8px;text-align:center; display: flex; flex-direction: column; align-items: center; gap: 4px;}
.pill b{display:block;font-size:12px;color:#9fb7ef}
.pill span{font-size:12px}
.pill small {font-size: 10px; opacity: 0.8;}
.green{color:var(--ok)}.red{color:var(--bad)}.amber{color:var(--warn)}
.table{width:100%;border-collapse:separate;border-spacing:0 8px; table-layout: fixed;}
.table th{font-size:12px;text-align:right;color:#9ab2e2;font-weight:600;padding:0 6px}
.table td{padding:8px;background:#0d1730;border:1px solid #24335f; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}
.switch{display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;border:1px solid #2a3a68;background:#0f1b3b;cursor:pointer;user-select:none}
.switch input{display:none}
.switch .dot{width:14px;height:14px;border-radius:50%;background:#6a7fb2;transition:.2s}
.switch input:checked + .dot{background:#24d08a;transform:translateX(2px) scale(1.1)}
.small{font-size:12px;color:#a8bfeb}
.performance-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 16px;}
.metric-card {background: #0d1730; border: 1px solid #24335f; border-radius: 12px; padding: 12px; text-align: center;}
.metric-title {font-size: 12px; color: #8aa0c8; margin-bottom: 6px;}
.metric-value {font-size: 18px; font-weight: 700;}
.chart-container { height: 200px; }
.loading-spinner { border: 3px solid rgba(255, 255, 255, 0.1); border-radius: 50%; border-top: 3px solid #3aa0ff; width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 20px auto; }
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
.slider { -webkit-appearance: none; width: 100%; height: 6px; border-radius: 3px; background: #1e2c52; outline: none; }
.slider::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 16px; height: 16px; border-radius: 50%; background: #3aa0ff; cursor: pointer; }
.slider::-moz-range-thumb { width: 16px; height: 16px; border-radius: 50%; background: #3aa0ff; cursor: pointer; }
</style>
</head>
<body>
<div class="container">
  <header><h1>لوحة التحكم • بوت التداول V23 (تفاعلي)</h1><div class="badge" id="serverTime">—</div></header>
  <div class="main-layout">
    <div class="left-column">
      <div class="card">
        <h2>الصفقات المفتوحة <span class="small" id="signalCount">(0)</span></h2>
        <div class="card-body">
            <div class="controls" style="margin-bottom: 12px;">
                <button class="btn small" data-sort="quality_score">الترتيب حسب الجودة</button>
                <button class="btn small" data-sort="id">الترتيب حسب الأحدث</button>
                <button class="btn small" data-sort="strategy_name">الترتيب حسب الاستراتيجية</button>
            </div>
            <div id="signals" class="signals-grid"></div>
        </div>
      </div>
      <div class="card">
        <h2>مؤشرات الأداء</h2>
        <div class="card-body">
            <div class="performance-grid">
                <div class="metric-card"><div class="metric-title">معدل الربح (30 يوم)</div><div class="metric-value" id="winRate">—</div></div>
                <div class="metric-card"><div class="metric-title">متوسط الربح (30 يوم)</div><div class="metric-value" id="avgProfit">—</div></div>
                <div class="metric-card"><div class="metric-title">أكبر تراجع (30 يوم)</div><div class="metric-value" id="maxDrawdown">—</div></div>
                <div class="metric-card"><div class="metric-title">إجمالي الصفقات (30 يوم)</div><div class="metric-value" id="totalTrades">—</div></div>
            </div>
            <div class="chart-container"><canvas id="performanceChart"></canvas></div>
        </div>
      </div>
    </div>
    <div class="right-column">
      <div class="card">
        <h2>التحكم والحالة</h2>
        <div class="card-body">
          <div class="controls">
            <label class="switch"><input id="toggleTrading" type="checkbox" /><span class="dot"></span><span class="small">تشغيل التداول</span></label>
            <a class="btn" href="/settings">الإعدادات</a>
            <a class="btn" href="/backtest">الاختبار الخلفي</a>
          </div>
          <div class="kv" style="margin-top:12px">
            <div>الرصيد (USDT):</div><div id="balance">—</div><div>عدد الصفقات:</div><div id="openCount">—</div>
          </div>
        </div>
      </div>
      <div class="card">
        <h2>حالة السوق</h2>
        <div class="card-body">
          <div class="trend" id="marketTrends"><div class="loading-spinner"></div></div>
        </div>
      </div>
      <div class="card">
        <h2>إعدادات التداول</h2>
        <div class="card-body">
          <div class="kv">
            <div>وضع التداول:</div>
            <div>
              <label class="switch" id="tradingModeSwitch">
                <input type="checkbox" id="tradingModeToggle">
                <span class="dot"></span>
                <span id="tradingModeText">ورقي</span>
              </label>
            </div>
          </div>
          <div class="kv">
            <div>الحد الأدنى لجودة الإشارة:</div>
            <div>
              <input type="range" id="qualityFilter" min="30" max="90" value="60" class="slider">
              <span id="qualityValue">60</span>
            </div>
          </div>
          <div class="kv" style="margin-top: 12px;">
            <div>حجم الصفقة (USDT):</div>
            <input type="number" id="tradeSizeInput" value="5.0" step="0.5" min="1.0" style="width: 100%; background: #0b1126; border: 1px solid #233056; color: #e8f1ff; padding: 6px; border-radius: 8px; text-align: center;">
          </div>
        </div>
      </div>
      <div class="card">
        <h2>سجل الرفض</h2>
        <div class="card-body" style="padding:0; max-height: 250px; overflow-y: auto;">
          <table class="table" id="rejections"><thead><tr><th>الوقت</th><th>الرمز</th><th>السبب</th></tr></thead><tbody></tbody></table>
        </div>
      </div>
      <div class="card">
        <h2>سجل الأحداث</h2>
        <div class="card-body" style="padding:0; max-height: 250px; overflow-y: auto;">
          <table class="table" id="events"><thead><tr><th>الوقت</th><th>النوع</th><th>الرسالة</th></tr></thead><tbody></tbody></table>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
const qs = s => document.querySelector(s);
let lastPrices = {};
let performanceChartInstance = null;
let openSignals = {};

const debounce = (func, delay) => {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), delay);
    };
};
function fmt(n){ return n == null ? '—' : (+n).toLocaleString('en-US', {maximumFractionDigits: 6}); }
function showLoadingIndicator(containerId) {
    const container = qs(containerId);
    if(container) container.innerHTML = '<div class="loading-spinner"></div>';
}
function showNotification(message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);
}

function closeTrade(signalId) {
    if (!confirm('هل أنت متأكد من رغبتك في إغلاق هذه الصفقة يدويًا؟')) {
        return;
    }
    fetch(`/api/close_trade/${signalId}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({}) // Send empty JSON body
    })
    .then(res => {
        if (!res.ok) {
            return res.json().then(err => { throw new Error(err.message || 'Server error') });
        }
        return res.json();
    })
    .then(data => {
        if (data.success) {
            showNotification('تم إرسال أمر الإغلاق بنجاح.', 'success');
        } else {
            showNotification(`فشل إغلاق الصفقة: ${data.message}`, 'error');
        }
    })
    .catch(err => {
        showNotification(`حدث خطأ: ${err.message}`, 'error');
        console.error(err);
    });
}

function renderSignal(signal) {
    const cp = signal.current_price || lastPrices[signal.symbol] || signal.entry_price;
    const entry = signal.entry_price;
    const tp1 = signal.target_price_1;
    const sl = signal.stop_loss;

    let progress = 0;
    let color = 'transparent';
    let title = 'في انتظار حركة السعر';

    if (cp >= entry && tp1 > entry) {
        progress = Math.min(100, ((cp - entry) / (tp1 - entry)) * 100);
        color = 'linear-gradient(90deg, var(--ok), #3fd1b0)';
        title = `التقدم نحو الهدف: ${progress.toFixed(1)}%`;
    } else if (cp < entry && entry > sl) {
        progress = Math.min(100, ((entry - cp) / (entry - sl)) * 100);
        color = 'linear-gradient(90deg, var(--bad), #ff6b7a)';
        title = `الاقتراب من وقف الخسارة: ${progress.toFixed(1)}%`;
    }

    const qualityScore = signal.signal_details?.quality_score || 0;
    const qualityColor = qualityScore > 75 ? 'var(--ok)' : qualityScore > 55 ? 'var(--warn)' : 'var(--bad)';
    
    return `
        <div class="signal" id="signal-${signal.id}" data-symbol="${signal.symbol}">
            <div>
                <div class="sig-title">${signal.symbol}</div>
                <div class="sig-meta">${signal.strategy_name} | <span style="color: ${qualityColor}; font-weight: bold;">⭐ ${qualityScore}/100</span></div>
            </div>
            <div style="text-align:end">
                <div class="price">${fmt(cp)}</div>
                <div class="small price-delta"></div>
                <button class="btn warn small" onclick="closeTrade(${signal.id})">إغلاق</button>
            </div>
            <div class="progress" title="${title}">
                <span class="progress-bar" style="width:${progress.toFixed(2)}%; background:${color};"></span>
            </div>
        </div>`;
}

function renderAllSignals(signals) {
    const container = qs('#signals');
    if (!signals || signals.length === 0) {
        container.innerHTML = '<p style="text-align:center;color:var(--muted);">لا توجد صفقات مفتوحة حالياً.</p>';
        return;
    }
    container.innerHTML = signals.map(renderSignal).join('');
}

function updateSingleSignal(signal) {
    const existingElement = qs(`#signal-${signal.id}`);
    if (existingElement) {
        existingElement.outerHTML = renderSignal(signal);
    } else {
        qs('#signals').insertAdjacentHTML('afterbegin', renderSignal(signal));
    }
}

function updatePrices(priceData) {
    for (const [symbol, price] of Object.entries(priceData)) {
        const signalElements = document.querySelectorAll(`.signal[data-symbol="${symbol}"]`);
        signalElements.forEach(el => {
            const priceEl = el.querySelector('.price');
            const deltaEl = el.querySelector('.price-delta');
            const prevPrice = lastPrices[symbol] || price;
            const delta = price - prevPrice;
            
            if (priceEl) priceEl.textContent = fmt(price);
            if (deltaEl) {
                deltaEl.className = `small price-delta ${delta > 0 ? 'green' : (delta < 0 ? 'red' : '')}`;
                deltaEl.textContent = delta > 0 ? '▲' : (delta < 0 ? '▼' : '•');
            }
            
            const signalId = el.id.split('-')[1];
            const signalData = openSignals[signalId];
            if (signalData) {
                const entry = signalData.entry_price;
                const tp1 = signalData.target_price_1;
                const sl = signalData.stop_loss;

                let progress = 0;
                let color = 'transparent';
                let title = 'في انتظار حركة السعر';

                if (price >= entry && tp1 > entry) {
                    progress = Math.min(100, ((price - entry) / (tp1 - entry)) * 100);
                    color = 'linear-gradient(90deg, var(--ok), #3fd1b0)';
                    title = `التقدم نحو الهدف: ${progress.toFixed(1)}%`;
                } else if (price < entry && entry > sl) {
                    progress = Math.min(100, ((entry - price) / (entry - sl)) * 100);
                    color = 'linear-gradient(90deg, var(--bad), #ff6b7a)';
                    title = `الاقتراب من وقف الخسارة: ${progress.toFixed(1)}%`;
                }

                const progressBar = el.querySelector('.progress-bar');
                const progressContainer = el.querySelector('.progress');
                if(progressBar) {
                    progressBar.style.width = `${progress}%`;
                    progressBar.style.background = color;
                }
                if(progressContainer) {
                    progressContainer.title = title;
                }
            }
        });
        lastPrices[symbol] = price;
    }
}

function addNotification(notification) {
    const tbody = qs('#events tbody');
    const row = `<tr><td>${new Date(notification.timestamp).toLocaleTimeString('ar-EG')}</td><td>${notification.type||''}</td><td>${notification.message||''}</td></tr>`;
    tbody.insertAdjacentHTML('afterbegin', row);
    if (tbody.rows.length > 20) tbody.deleteRow(-1);
}

function addRejection(rejection) {
    const tbody = qs('#rejections tbody');
    const row = `<tr><td>${new Date(rejection.timestamp).toLocaleTimeString('ar-EG')}</td><td>${rejection.symbol||''}</td><td>${rejection.reason||''}</td></tr>`;
    tbody.insertAdjacentHTML('afterbegin', row);
    if (tbody.rows.length > 30) tbody.deleteRow(-1);
}

function updateMarketTrends(marketState) {
  const trendsContainer = document.getElementById('marketTrends');
  trendsContainer.innerHTML = '';
  
  if (marketState && marketState.trend_details_by_tf) {
    ['15m', '1h', '4h'].forEach(tf => {
      const trend = marketState.trend_details_by_tf[tf];
      if (trend) {
        let trendClass = 'amber';
        let trendText = 'جانبي';
        if (trend.trend === 'bullish') { trendClass = 'green'; trendText = 'صاعد'; } 
        else if (trend.trend === 'bearish') { trendClass = 'red'; trendText = 'هابط'; }
        
        trendsContainer.innerHTML += `
          <div class="pill">
            <b>${tf}</b>
            <span class="${trendClass}">${trendText}</span>
            <small>ADX: ${trend.adx?.toFixed(1) || '—'}</small>
            <small>RSI: ${trend.rsi?.toFixed(1) || '—'}</small>
          </div>`;
      }
    });
  }
}

async function initializeDashboard() {
    try {
        showLoadingIndicator('#signals');
        const [baseRes, signalsRes, metricsRes] = await Promise.all([
            fetch('/api/dashboard_data'),
            fetch('/api/open_signals'),
            fetch('/api/performance_metrics')
        ]);
        
        const baseData = await baseRes.json();
        const signalsData = await signalsRes.json();
        const metricsData = await metricsRes.json();
        
        qs('#serverTime').textContent = new Date(baseData.server_time).toLocaleTimeString('ar-EG');
        qs('#toggleTrading').checked = !!baseData.trading_enabled;
        qs('#balance').textContent = fmt(baseData.usdt_balance);
        
        const isPaper = baseData.paper_trading_mode;
        qs('#tradingModeToggle').checked = !isPaper;
        qs('#tradingModeText').textContent = isPaper ? 'ورقي' : 'حقيقي';

        qs('#qualityFilter').value = baseData.min_signal_quality;
        qs('#qualityValue').textContent = baseData.min_signal_quality;
        qs('#tradeSizeInput').value = baseData.fixed_trade_size;

        updateMarketTrends(baseData.market_state);
        
        openSignals = signalsData.signals.reduce((acc, s) => { acc[s.id] = s; return acc; }, {});
        renderAllSignals(signalsData.signals);
        qs('#openCount').textContent = signalsData.signals.length;
        qs('#signalCount').textContent = `(${signalsData.signals.length})`;
        
        qs('#winRate').textContent = `${metricsData.win_rate.toFixed(2)}%`;
        qs('#avgProfit').textContent = `${metricsData.avg_profit.toFixed(2)}%`;
        qs('#maxDrawdown').textContent = `${metricsData.max_drawdown.toFixed(2)}%`;
        qs('#totalTrades').textContent = metricsData.total_trades;

        loadAdditionalData();
        
    } catch (error) {
        console.error("فشل تحميل البيانات الأساسية:", error);
        qs('#signals').innerHTML = '<p>فشل تحميل البيانات. حاول تحديث الصفحة.</p>';
    }
}

async function loadAdditionalData() {
    try {
        const perfRes = await fetch('/api/advanced_performance_data');
        if (perfRes.ok) updateAdvancedPerformance(await perfRes.json());
    } catch (error) {
        console.error("Error loading additional data:", error);
    }
}

function setupWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    const socket = new WebSocket(wsUrl);
    
    socket.onopen = () => console.log("WebSocket connection established");
    
    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        switch(data.type) {
            case 'price_update': updatePrices(data.payload); break;
            case 'new_signal':
                openSignals[data.payload.id] = data.payload;
                updateSingleSignal(data.payload);
                break;
            case 'signal_update':
                openSignals[data.payload.id] = data.payload;
                updateSingleSignal(data.payload);
                break;
            case 'trade_closed':
                const el = qs(`#signal-${data.payload.signal_id}`);
                if (el) el.remove();
                delete openSignals[data.payload.signal_id];
                break;
            case 'new_notification': addNotification(data.payload); break;
            case 'new_rejection': addRejection(data.payload); break;
            case 'market_state_update': updateMarketTrends(data.payload); break;
            case 'trading_mode':
                const isPaper = data.payload.paper_trading;
                qs('#tradingModeToggle').checked = !isPaper;
                qs('#tradingModeText').textContent = isPaper ? 'ورقي' : 'حقيقي';
                break;
            case 'quality_filter':
                qs('#qualityFilter').value = data.payload.min_quality;
                qs('#qualityValue').textContent = data.payload.min_quality;
                break;
            case 'trade_size_update':
                const input = qs('#tradeSizeInput');
                if (input) input.value = data.payload.trade_size;
                break;
        }
    };
    
    socket.onclose = () => {
        console.log("WebSocket connection closed, reconnecting...");
        setTimeout(setupWebSocket, 3000);
    };
    socket.onerror = (error) => console.error("WebSocket error:", error);
}

function setupSorting() {
    const sortButtons = document.querySelectorAll('[data-sort]');
    const debouncedSort = debounce((sortBy) => {
        showLoadingIndicator('#signals');
        fetch(`/api/open_signals?sort=${sortBy}`)
            .then(res => res.json())
            .then(data => {
                openSignals = data.signals.reduce((acc, s) => { acc[s.id] = s; return acc; }, {});
                renderAllSignals(data.signals);
            })
            .catch(err => console.error("Sort failed:", err));
    }, 300);

    sortButtons.forEach(button => {
        button.addEventListener('click', () => {
            const sortBy = button.dataset.sort;
            debouncedSort(sortBy);
        });
    });
}

async function toggleTrading() { await fetch('/toggle_trading', {method:'POST'}); }
qs('#toggleTrading').addEventListener('change', toggleTrading);

qs('#tradingModeToggle').addEventListener('change', function() {
  const isPaper = !this.checked;
  const modeText = isPaper ? 'ورقي' : 'حقيقي';
  
  if (!isPaper) {
    if (!confirm('هل أنت متأكد من التبديل إلى التداول الحقيقي؟ هذا سيستخدم أموالاً حقيقية.')) {
      this.checked = false;
      return;
    }
  }
  
  fetch('/api/trading_mode', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({paper_trading: isPaper})
  })
  .then(res => res.json())
  .then(data => {
    if (data.success) {
      qs('#tradingModeText').textContent = modeText;
      showNotification(`تم التبديل إلى الوضع ${modeText}`, 'success');
    } else {
      showNotification('فشل تغيير وضع التداول', 'error');
      this.checked = !this.checked;
    }
  })
  .catch(error => {
    console.error('Error:', error);
    showNotification('خطأ في الاتصال بالخادم', 'error');
    this.checked = !this.checked;
  });
});

const debouncedQualityUpdate = debounce((value) => {
    fetch('/api/quality_filter', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({min_quality: parseInt(value)})
    }).catch(error => console.error('Error:', error));
}, 500);

qs('#qualityFilter').addEventListener('input', function() {
  const value = this.value;
  qs('#qualityValue').textContent = value;
  debouncedQualityUpdate(value);
});

const debouncedTradeSizeUpdate = debounce((value) => {
    fetch('/api/trade_size', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({trade_size: parseFloat(value)})
    }).catch(error => console.error('Error updating trade size:', error));
}, 800);

qs('#tradeSizeInput').addEventListener('input', function() {
  debouncedTradeSizeUpdate(this.value);
});


function updateAdvancedPerformance(data) {
    if (!performanceChartInstance && data.equity_curve && data.equity_curve.labels.length > 0) {
        createPerformanceChart(data.equity_curve);
    } else if (performanceChartInstance) {
        performanceChartInstance.data.labels = data.equity_curve.labels;
        performanceChartInstance.data.datasets[0].data = data.equity_curve.values;
        performanceChartInstance.update('none');
    }
}

function createPerformanceChart(chartData) {
    const ctx = document.getElementById('performanceChart').getContext('2d');
    performanceChartInstance = new Chart(ctx, {
        type: 'line',
        data: { labels: chartData.labels, datasets: [{ label: 'رأس المال', data: chartData.values, borderColor: '#3aa0ff', backgroundColor: 'rgba(58, 160, 255, 0.1)', tension: 0.4, fill: true, pointRadius: 0, borderWidth: 2 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { type: 'time', time: { unit: 'day' }, ticks: { color: 'var(--muted)', autoSkip: true, maxTicksLimit: 8 }, grid: { display: false } }, y: { ticks: { color: 'var(--muted)', callback: (v) => v.toFixed(0) }, grid: { color: 'rgba(255, 255, 255, 0.05)' } } } }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    initializeDashboard();
    setupWebSocket();
    setupSorting();
});
</script>
</body>
</html>
"""
# ======================== END: تحديث واجهة المستخدم ========================

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
.form-group input, .form-group textarea {width: 100%; background: #0b1126; border: 1px solid #233056; color: #e8f1ff; padding: 10px; border-radius: 8px;}
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
                        <label for="symbols">رموز العملات (مفصولة بفاصلة)</label>
                        <textarea id="symbols" name="symbols" rows="2">BTCUSDT,ETHUSDT,BNBUSDT</textarea>
                    </div>
                    <div class="form-group">
                        <label for="start_date">تاريخ البدء</label>
                        <input type="date" id="start_date" name="start_date" required>
                    </div>
                    <div class="form-group">
                        <label for="end_date">تاريخ الانتهاء</label>
                        <input type="date" id="end_date" name="end_date" required>
                    </div>
                    <div class="form-group">
                        <label for="initial_balance">الرصيد المبدئي</label>
                        <input type="number" id="initial_balance" name="initial_balance" value="10000" required>
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
                        <div class="metric-card"><div class="metric-title">الرصيد النهائي</div><div class="metric-value" id="finalBalance"></div></div>
                        <div class="metric-card"><div class="metric-title">إجمالي الصفقات</div><div class="metric-value" id="totalTrades"></div></div>
                        <div class="metric-card"><div class="metric-title">معدل الربح</div><div class="metric-value" id="winRate"></div></div>
                        <div class="metric-card"><div class="metric-title">عامل الربح</div><div class="metric-value" id="profitFactor"></div></div>
                        <div class="metric-card"><div class="metric-title">أكبر تراجع</div><div class="metric-value red" id="maxDrawdown"></div></div>
                        <div class="metric-card"><div class="metric-title">متوسط الربح/صفقة</div><div class="metric-value" id="avgProfit"></div></div>
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
                    <thead><tr><th>الرمز</th><th>الاستراتيجية</th><th>وقت الدخول</th><th>سعر الدخول</th><th>وقت الخروج</th><th>سعر الخروج</th><th>الربح %</th></tr></thead>
                    <tbody id="trades-table"></tbody>
                </table>
            </div>
        </div>
    </div>
</div>
<script>
const qs = s => document.querySelector(s);
let equityChartInstance = null;

const today = new Date();
const thirtyDaysAgo = new Date();
thirtyDaysAgo.setDate(today.getDate() - 30);
qs('#end_date').valueAsDate = today;
qs('#start_date').valueAsDate = thirtyDaysAgo;


qs('#backtest-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    qs('#loader').style.display = 'block';
    qs('#results-container').style.display = 'none';
    
    const formData = new FormData(e.target);
    const data = {
        symbols: formData.get('symbols').split(',').map(s => s.trim().toUpperCase()),
        start_date: formData.get('start_date'),
        end_date: formData.get('end_date'),
        initial_balance: parseFloat(formData.get('initial_balance'))
    };

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
    
    qs('#finalBalance').textContent = `$${data.final_balance.toFixed(2)}`;
    qs('#totalTrades').textContent = data.total_trades;
    qs('#winRate').textContent = `${(data.win_rate || 0).toFixed(2)}%`;
    qs('#profitFactor').textContent = (data.profit_factor || 0).toFixed(2);
    qs('#maxDrawdown').textContent = `${(data.max_drawdown || 0).toFixed(2)}%`;
    qs('#avgProfit').textContent = `${(data.avg_profit_per_trade || 0).toFixed(2)}%`;

    const tradesTable = qs('#trades-table');
    tradesTable.innerHTML = data.trades.map(trade => `
        <tr>
            <td>${trade.symbol}</td>
            <td>${trade.strategy}</td>
            <td>${new Date(trade.entry_date).toLocaleString('ar-EG')}</td>
            <td>${trade.entry_price.toFixed(4)}</td>
            <td>${new Date(trade.exit_date).toLocaleString('ar-EG')}</td>
            <td>${trade.exit_price.toFixed(4)}</td>
            <td class="${trade.profit_percentage > 0 ? 'green' : 'red'}">${trade.profit_percentage.toFixed(2)}%</td>
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
            labels: equityData.map(d => d.timestamp),
            datasets: [{
                label: 'رأس المال',
                data: equityData.map(d => d.balance),
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
                x: { type: 'time', time: { unit: 'day' }, ticks: { color: 'var(--muted)', autoSkip: true, maxTicksLimit: 8 }, grid: { display: false } },
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

@app.route('/api/health', methods=['GET'])
def api_health():
    try:
        with trading_status_lock: trading_enabled = is_trading_enabled
        with trading_mode_lock: is_paper = paper_trading_mode
        return jsonify({
            "status": "ok",
            "trading_enabled": trading_enabled,
            "mode": "PAPER" if is_paper else "REAL",
            "open_signals": len(open_signals_cache),
            "ws": {"connected": True}
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/open_signals')
def get_open_signals():
    if not check_db_connection():
        return jsonify({"error": "Database connection failed"}), 500
    
    sort_by = request.args.get('sort', 'id')
    allowed_sort_fields = ['id', 'symbol', 'entry_price', 'strategy_name', 'quality_score']
    if sort_by not in allowed_sort_fields:
        sort_by = 'id'

    order_direction = 'DESC' if sort_by in ['id', 'quality_score'] else 'ASC'
    sort_column_expression = sql.SQL("(signal_details->>'quality_score')::numeric")

    try:
        with conn.cursor() as cur:
            query = sql.SQL("""
                SELECT 
                    id, symbol, entry_price, target_price_1, target_price_2, 
                    stop_loss, strategy_name, is_real_trade, quantity, 
                    signal_details, 
                    {sort_expression} as quality_score
                FROM signals 
                WHERE status IN ('open', 'updated')
                ORDER BY {sort_col} {direction} NULLS LAST
            """).format(
                sort_expression=sort_column_expression,
                sort_col=sql.Identifier(sort_by) if sort_by != 'quality_score' else sql.SQL('quality_score'),
                direction=sql.SQL(order_direction)
            )
            cur.execute(query)
            signals = cur.fetchall()
        return jsonify({"signals": [dict(s) for s in signals]})
    except Exception as e:
        logger.error(f"Error fetching open signals: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance_metrics')
def get_performance_metrics():
    cache_key = "performance_metrics"
    if redis_client:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            return jsonify(json.loads(cached_data))
    
    if not check_db_connection():
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit_percentage > 0 THEN 1 ELSE 0 END) as winning_trades,
                    AVG(profit_percentage) as avg_profit,
                    MAX(drawdown) as max_drawdown
                FROM performance_summary
                WHERE date >= NOW() - INTERVAL '30 days'
            """)
            metrics = cur.fetchone()
        
        total_trades = metrics['total_trades'] if metrics['total_trades'] is not None else 0
        winning_trades = metrics['winning_trades'] if metrics['winning_trades'] is not None else 0
        
        result = {
            "total_trades": total_trades,
            "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            "avg_profit": metrics['avg_profit'] if metrics['avg_profit'] is not None else 0,
            "max_drawdown": metrics['max_drawdown'] if metrics['max_drawdown'] is not None else 0
        }
        
        if redis_client:
            redis_client.setex(cache_key, 300, json.dumps(result, cls=NpEncoder))
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/signals_history')
def get_signals_history():
    if not check_db_connection():
        return jsonify({"error": "Database connection failed"}), 500
        
    page = request.args.get('page', 1, type=int)
    per_page = 20
    offset = (page - 1) * per_page
    
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM signals WHERE status = 'closed' ORDER BY closed_at DESC LIMIT %s OFFSET %s", (per_page, offset))
        signals = cur.fetchall()
        
        cur.execute("SELECT COUNT(*) FROM signals WHERE status = 'closed'")
        total = cur.fetchone()['count']
    
    return jsonify({
        "signals": [dict(s) for s in signals],
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "pages": (total + per_page - 1) // per_page
        }
    })

@sock.route('/ws')
def ws(ws_client):
    logger.info("WebSocket client connected.")
    with ws_clients_lock:
        ws_clients.append(ws_client)
    try:
        ws_client.send(json.dumps({"type": "connection_established"}, cls=NpEncoder))
        while True:
            message = ws_client.receive(timeout=30)
            if message is None:
                ws_client.send(json.dumps({"type": "ping"}, cls=NpEncoder))
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
            cur.execute("SELECT profit_percentage, closed_at FROM signals WHERE status = 'closed' ORDER BY closed_at ASC")
            trades = cur.fetchall()

        if len(trades) < 2:
            return jsonify({
                "winRate": 0, "profitFactor": 0, "maxDrawdown": 0, "sharpeRatio": 0,
                "equity_curve": {"labels": [], "values": []}
            })

        profits = [t['profit_percentage'] for t in trades if t['profit_percentage'] is not None]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]

        win_rate = (len(wins) / len(profits) * 100) if profits else 0
        
        total_profit = sum(wins)
        total_loss = abs(sum(losses))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        equity_curve_values = [1000]
        for p in profits:
            equity_curve_values.append(equity_curve_values[-1] * (1 + p / 100))
        
        peak = equity_curve_values[0]
        max_drawdown = 0
        for equity in equity_curve_values:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        returns = np.array(profits) / 100
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(len(trades)) if np.std(returns) > 0 else 0

        equity_curve_labels = [t['closed_at'].isoformat() for t in trades]
        
        return jsonify({
            "winRate": win_rate,
            "profitFactor": profit_factor,
            "maxDrawdown": max_drawdown,
            "sharpeRatio": sharpe_ratio,
            "equity_curve": {"labels": equity_curve_labels, "values": equity_curve_values[1:]}
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

@app.route('/api/trading_mode', methods=['POST'])
def update_trading_mode():
    global paper_trading_mode
    try:
        data = request.json
        is_paper = data.get('paper_trading', True)
        
        with trading_mode_lock:
            paper_trading_mode = is_paper
        
        if redis_client:
            settings_data = redis_client.get('trading_settings')
            settings = json.loads(settings_data) if settings_data else {}
            settings['paper_trading_mode'] = is_paper
            redis_client.set('trading_settings', json.dumps(settings))
        
        broadcast({"type": "trading_mode", "payload": {"paper_trading": is_paper}})
        log_and_notify("info", f"Trading mode switched to {'Paper' if is_paper else 'Real'}.", "TRADING_MODE_SWITCH")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error updating trading mode: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/quality_filter', methods=['POST'])
def update_quality_filter():
    global MIN_SIGNAL_QUALITY
    try:
        data = request.json
        min_quality = data.get('min_quality', 60)
        
        with min_quality_lock:
            MIN_SIGNAL_QUALITY = int(min_quality)

        if redis_client:
            settings_data = redis_client.get('signal_quality_settings')
            settings = json.loads(settings_data) if settings_data else {}
            settings['min_quality'] = MIN_SIGNAL_QUALITY
            redis_client.set('signal_quality_settings', json.dumps(settings))
        
        broadcast({"type": "quality_filter", "payload": {"min_quality": MIN_SIGNAL_QUALITY}})
        log_and_notify("info", f"Minimum signal quality updated to {MIN_SIGNAL_QUALITY}.", "SETTINGS_UPDATE")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error updating quality filter: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# ======================= START: [NEW] API Endpoint for Trade Size =======================
@app.route('/api/trade_size', methods=['POST'])
def update_trade_size():
    global FIXED_TRADE_SIZE_USDT
    try:
        data = request.json
        trade_size = data.get('trade_size', 5.0)
        
        with fixed_trade_size_lock:
            FIXED_TRADE_SIZE_USDT = float(trade_size)

        if redis_client:
            settings_data = redis_client.get('trading_settings')
            settings = json.loads(settings_data) if settings_data else {}
            settings['FIXED_TRADE_SIZE_USDT'] = FIXED_TRADE_SIZE_USDT
            redis_client.set('trading_settings', json.dumps(settings))
        
        broadcast({"type": "trade_size_update", "payload": {"trade_size": FIXED_TRADE_SIZE_USDT}})
        log_and_notify("info", f"Fixed trade size updated to ${FIXED_TRADE_SIZE_USDT}.", "SETTINGS_UPDATE")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error updating trade size: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
# ======================== END: [NEW] API Endpoint for Trade Size ========================

def close_trade_manually(signal_id: int, closing_price: Optional[float] = None) -> bool:
    with signal_cache_lock:
        signal_to_close = next((dict(s) for s in open_signals_cache.values() if s['id'] == signal_id), None)

    if not signal_to_close:
        logger.warning(f"[Manual Close] Signal {signal_id} not found in active cache. It might have been closed automatically.")
        return False

    symbol = signal_to_close['symbol']

    if closing_price is None:
        with live_prices_lock:
            closing_price = live_prices.get(symbol)

        if closing_price is None:
            logger.error(f"[Manual Close] Could not get live price for {symbol} to close signal {signal_id}.")
            send_telegram_message(f"⚠️ *فشل الإغلاق اليدوي لـ {symbol}* \nلم يتمكن البوت من الحصول على السعر الحالي.", force=True)
            return False

    logger.info(f"[Manual Close] User initiated manual close for signal {signal_id} ({symbol}) at price {closing_price}")
    close_signal(signal_to_close, closing_price, "manual_close")
    return True

@app.route('/api/close_trade/<int:signal_id>', methods=['POST'])
def api_close_trade(signal_id):
    data = request.get_json(silent=True) or {}
    closing_price = data.get('closing_price')
    thread = Thread(target=close_trade_manually, args=(signal_id, closing_price))
    thread.start()
    return jsonify({"success": True, "message": "Trade close command received and is being processed."})

@app.route('/update_settings', methods=['POST'])
def update_settings():
    global RISK_PER_TRADE_PERCENT, MAX_OPEN_TRADES
    try:
        data = request.json
        with risk_per_trade_lock: RISK_PER_TRADE_PERCENT = float(data['risk_per_trade'])
        MAX_OPEN_TRADES = int(data['max_trades'])
        if redis_client:
            with trading_mode_lock: is_paper = paper_trading_mode
            with fixed_trade_size_lock: trade_size = FIXED_TRADE_SIZE_USDT
            settings = {'RISK_PER_TRADE_PERCENT': RISK_PER_TRADE_PERCENT, 'MAX_OPEN_TRADES': MAX_OPEN_TRADES, 'paper_trading_mode': is_paper, 'FIXED_TRADE_SIZE_USDT': trade_size}
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

def run_backtest(symbols: List[str], start_date: str, end_date: str, initial_balance: float = 10000.0) -> Dict:
    logger.info(f"[Backtest] Starting backtest from {start_date} to {end_date}")
    
    results = {
        "start_date": start_date, "end_date": end_date, "initial_balance": initial_balance,
        "final_balance": initial_balance, "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
        "total_profit": 0.0, "max_drawdown": 0.0, "max_drawdown_period": "", "trades": [],
        "balance_history": [], "equity_curve": [], "drawdown_curve": [], "strategy_performance": {}
    }
    
    current_balance = initial_balance
    max_balance = initial_balance
    max_drawdown = 0.0
    max_drawdown_date = start_date
    open_trades = {}
    balance_history = [(start_date, initial_balance)]
    
    for strategy in STRATEGY_NAMES.keys():
        results["strategy_performance"][strategy] = {
            "total_trades": 0, "winning_trades": 0, "total_profit": 0.0,
            "avg_profit": 0.0, "max_profit": 0.0, "max_loss": 0.0
        }
    
    for symbol in symbols:
        try:
            logger.info(f"[Backtest] Processing {symbol}")
            df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, 
                                      (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days + 5)
            if df is None or len(df) == 0:
                logger.warning(f"[Backtest] No data available for {symbol}"); continue
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            if len(df) < 200:
                logger.warning(f"[Backtest] Insufficient data for {symbol}"); continue
            
            df = calculate_all_features(df)
            df.name = symbol
            
            for i, (timestamp, row) in enumerate(df.iterrows()):
                trades_to_close = []
                for trade_id, trade in open_trades.items():
                    if trade['symbol'] == symbol:
                        if row['high'] >= trade['target_price_1']:
                            profit_percentage = ((trade['target_price_1'] - trade['entry_price']) / trade['entry_price']) * 100
                            trade_profit = current_balance * trade['risk_percent'] / 100 * (profit_percentage / 100)
                            current_balance += trade_profit
                            results["total_trades"] += 1; results["winning_trades"] += 1; results["total_profit"] += profit_percentage
                            strategy_name = trade['strategy']
                            strat_perf = results["strategy_performance"][strategy_name]
                            strat_perf["total_trades"] += 1; strat_perf["winning_trades"] += 1; strat_perf["total_profit"] += profit_percentage
                            strat_perf["avg_profit"] = strat_perf["total_profit"] / strat_perf["total_trades"]
                            strat_perf["max_profit"] = max(strat_perf["max_profit"], profit_percentage)
                            results["trades"].append({
                                "symbol": symbol, "strategy": strategy_name, "entry_date": trade['entry_date'],
                                "exit_date": timestamp.isoformat(), "entry_price": trade['entry_price'],
                                "exit_price": trade['target_price_1'], "profit_percentage": profit_percentage, "type": "win"
                            })
                            trades_to_close.append(trade_id)
                        elif row['low'] <= trade['stop_loss']:
                            loss_percentage = ((trade['stop_loss'] - trade['entry_price']) / trade['entry_price']) * 100
                            trade_loss = current_balance * trade['risk_percent'] / 100 * (loss_percentage / 100)
                            current_balance += trade_loss
                            results["total_trades"] += 1; results["losing_trades"] += 1; results["total_profit"] += loss_percentage
                            strategy_name = trade['strategy']
                            strat_perf = results["strategy_performance"][strategy_name]
                            strat_perf["total_trades"] += 1; strat_perf["max_loss"] = min(strat_perf["max_loss"], loss_percentage)
                            strat_perf["total_profit"] += loss_percentage
                            strat_perf["avg_profit"] = strat_perf["total_profit"] / strat_perf["total_trades"]
                            results["trades"].append({
                                "symbol": symbol, "strategy": strategy_name, "entry_date": trade['entry_date'],
                                "exit_date": timestamp.isoformat(), "entry_price": trade['entry_price'],
                                "exit_price": trade['stop_loss'], "profit_percentage": loss_percentage, "type": "loss"
                            })
                            trades_to_close.append(trade_id)
                for trade_id in trades_to_close: del open_trades[trade_id]
                
                if i >= 200:
                    current_df = df.iloc[:i+1]
                    def create_backtest_trade(strategy_name):
                        if apply_strategy_filters(symbol, current_df, strategy_name):
                            trade_levels = calculate_trade_levels(current_df)
                            risk_percent = dynamic_risk_management(symbol, current_df)
                            trade_id = f"{symbol}_{timestamp.isoformat()}_{strategy_name}"
                            open_trades[trade_id] = {
                                "symbol": symbol, "entry_date": timestamp.isoformat(), "entry_price": trade_levels['entry_price'],
                                "stop_loss": trade_levels['stop_loss'], "target_price_1": trade_levels['target_price_1'],
                                "target_price_2": trade_levels['target_price_2'], "strategy": strategy_name, "risk_percent": risk_percent
                            }
                    if USE_BB_STOCH_STRATEGY and check_bb_stoch_strategy_enhanced(current_df): create_backtest_trade("BB_Stoch_Strategy")
                    if USE_MACD_EMA_STRATEGY and check_macd_ema_strategy_enhanced(current_df): create_backtest_trade("MACD_EMA_Strategy")
                    if USE_EMA_RSI_STRATEGY and check_ema_rsi_strategy_enhanced(current_df): create_backtest_trade("EMA_RSI_Strategy")
                    if USE_PULLBACK_STRATEGY and check_pullback_strategy_enhanced(current_df): create_backtest_trade("Pullback_Strategy")
                    if USE_MOMENTUM_VOLATILITY_STRATEGY and check_momentum_volatility_strategy(current_df): create_backtest_trade("Momentum_Volatility_Strategy")

                balance_history.append((timestamp.isoformat(), current_balance))
                if current_balance > max_balance: max_balance = current_balance
                drawdown = (max_balance - current_balance) / max_balance * 100 if max_balance > 0 else 0
                if drawdown > max_drawdown:
                    max_drawdown = drawdown; max_drawdown_date = timestamp.isoformat()
                results["equity_curve"].append({"timestamp": timestamp.isoformat(), "balance": current_balance})
                results["drawdown_curve"].append({"timestamp": timestamp.isoformat(), "drawdown": drawdown})
        except Exception as e:
            logger.error(f"[Backtest] Error processing {symbol}: {e}", exc_info=True); continue
    
    for trade_id, trade in open_trades.items():
        symbol = trade['symbol']
        try:
            df_symbol = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, 1)
            if df_symbol is not None and len(df_symbol) > 0:
                last_price = df_symbol['close'].iloc[-1]
                profit_percentage = ((last_price - trade['entry_price']) / trade['entry_price']) * 100
                trade_profit = current_balance * trade['risk_percent'] / 100 * (profit_percentage / 100)
                current_balance += trade_profit
                results["total_trades"] += 1
                if profit_percentage > 0: results["winning_trades"] += 1
                else: results["losing_trades"] += 1
                results["total_profit"] += profit_percentage
                strategy_name = trade['strategy']
                strat_perf = results["strategy_performance"][strategy_name]
                strat_perf["total_trades"] += 1; strat_perf["total_profit"] += profit_percentage
                strat_perf["avg_profit"] = strat_perf["total_profit"] / strat_perf["total_trades"] if strat_perf["total_trades"] > 0 else 0
                if profit_percentage > 0:
                    strat_perf["winning_trades"] += 1; strat_perf["max_profit"] = max(strat_perf["max_profit"], profit_percentage)
                else:
                    strat_perf["max_loss"] = min(strat_perf["max_loss"], profit_percentage)
                results["trades"].append({
                    "symbol": symbol, "strategy": strategy_name, "entry_date": trade['entry_date'],
                    "exit_date": end_date, "entry_price": trade['entry_price'], "exit_price": last_price,
                    "profit_percentage": profit_percentage, "type": "win" if profit_percentage > 0 else "loss"
                })
        except Exception as e:
             logger.error(f"[Backtest] Error closing final trade for {symbol}: {e}")

    results["final_balance"] = current_balance
    results["max_drawdown"] = max_drawdown
    results["max_drawdown_period"] = max_drawdown_date
    results["balance_history"] = balance_history
    if results["total_trades"] > 0:
        results["win_rate"] = (results["winning_trades"] / results["total_trades"]) * 100
        results["avg_profit_per_trade"] = results["total_profit"] / results["total_trades"]
    else:
        results["win_rate"] = 0; results["avg_profit_per_trade"] = 0
    
    gross_profit = sum(trade["profit_percentage"] for trade in results["trades"] if trade["profit_percentage"] > 0)
    gross_loss = abs(sum(trade["profit_percentage"] for trade in results["trades"] if trade["profit_percentage"] < 0))
    if gross_loss > 0: results["profit_factor"] = gross_profit / gross_loss
    else: results["profit_factor"] = float('inf') if gross_profit > 0 else 0
    
    logger.info(f"[Backtest] Backtest completed. Final balance: {current_balance:.2f}, Win rate: {results.get('win_rate', 0):.2f}%")
    return results

@app.route('/api/run_backtest', methods=['POST'])
def api_run_backtest():
    try:
        data = request.get_json()
        symbols = data.get('symbols', validated_symbols_to_scan)
        start_date = data.get('start_date', (datetime.now(timezone.utc) - timedelta(days=30)).strftime('%Y-%m-%d'))
        end_date = data.get('end_date', datetime.now(timezone.utc).strftime('%Y-%m-%d'))
        initial_balance = float(data.get('initial_balance', 10000.0))
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        if start_dt >= end_dt: return jsonify({"error": "Start date must be before end date"}), 400
        if (end_dt - start_dt).days > 365 * 2: return jsonify({"error": "Backtest period cannot exceed 2 years"}), 400
        results = run_backtest(symbols, start_date, end_date, initial_balance)
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error running backtest: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

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
                broadcast({"type": "signal_update", "payload": open_signals_cache[symbol]})
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
    
    with signal_cache_lock:
        if symbol not in open_signals_cache or open_signals_cache[symbol]['id'] != signal_id:
            logger.warning(f"[Close Signal] Attempted to close already closed or non-existent signal {signal_id} for {symbol}.")
            return

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
    
    broadcast({"type": "trade_closed", "payload": {"signal_id": signal_id, "symbol": symbol, "reason": reason}})
    
    trade_type = "حقيقية" if signal.get('is_real_trade') else "ورقية"
    result_emoji = "✅" if profit >= 0 else "🔻"
    reason_map = {
        "SL_HIT": "ضرب وقف الخسارة", "TP1_HIT": "تحقيق الهدف الأول", "TP2_HIT": "تحقيق الهدف الثاني",
        "manual_close": "إغلاق يدوي", "TRAILING_SL_HIT": "ضرب الوقف المتحرك",
        "stop_loss": "ضرب وقف الخسارة", "target_2_reached": "تحقيق الهدف الثاني",
        "support_broken": "كسر الدعم (إشارة معاكسة)"
    }
    reason_ar = reason_map.get(reason, reason)
    log_and_notify("info", f"Closed {trade_type} trade for {symbol}. Profit: {profit:.2f}%", "TRADE_CLOSED")
    settings = get_notification_settings()
    profit_condition = profit >= settings['min_profit_notification']
    loss_condition = profit <= settings['max_loss_notification']
    if profit_condition or loss_condition or reason == "manual_close":
        send_telegram_message(f"{result_emoji} *إغلاق صفقة {trade_type} {symbol}*\n*السبب:* {reason_ar}\n*الربح:* `{profit:.2f}%`")


def trade_management_loop():
    logger.info("🚀 [Trade Manager] Starting advanced trade management loop...")
    while True:
        try:
            with signal_cache_lock:
                if not open_signals_cache:
                    time.sleep(2); continue
                signals_to_monitor = list(open_signals_cache.values())
            for signal in signals_to_monitor:
                symbol = signal['symbol']
                with live_prices_lock:
                    current_price = live_prices.get(symbol)
                if not current_price:
                    continue

                details = signal.get('signal_details')
                if isinstance(details, str):
                    try:
                        details = json.loads(details)
                    except Exception:
                        details = {}
                details = details or {}

                entry_price = float(signal.get('entry_price', 0))
                stop_loss = float(signal.get('stop_loss', 0))
                tp1 = float(signal.get('target_price_1') or 0)
                tp2 = float(signal.get('target_price_2') or 0)
                trail_dist = float(details.get('trailing_stop_distance') or 0)
                remaining_qty = float(signal.get('quantity') or 0)

                if stop_loss and current_price <= stop_loss:
                    close_signal(signal, stop_loss, "SL_HIT")
                    continue

                if tp2 and current_price >= tp2:
                    close_signal(signal, tp2, "TP2_HIT")
                    continue

                if tp1 and not details.get('tp1_done') and remaining_qty > 0 and current_price >= tp1:
                    part_qty = remaining_qty * 0.5
                    execute_close_order(symbol, part_qty, "TP1_HIT")
                    new_sl = max(stop_loss, entry_price)
                    updates = {
                        "quantity": remaining_qty - part_qty,
                        "stop_loss": new_sl,
                        "status": "updated",
                        "closing_reason": "TP1_HIT"
                    }
                    details['tp1_done'] = True
                    updates['signal_details'] = json.dumps(details)
                    update_signal_in_db(signal['id'], updates)
                    with signal_cache_lock:
                        if symbol in open_signals_cache:
                            open_signals_cache[symbol].update(updates)
                            open_signals_cache[symbol]['signal_details'] = details
                    send_telegram_message(f"🥇 *تحقق الهدف الأول* لـ `{symbol}`\nتم إقفال 50% من العقد وتحريك الوقف إلى نقطة الدخول.")
                    broadcast({"type": "signal_update", "payload": open_signals_cache.get(symbol, {})})
                    continue

                profit_pct = ((current_price - entry_price) / max(entry_price, 1e-8)) * 100 if entry_price else 0
                if trail_dist and not details.get('trailing_active') and profit_pct >= TRAILING_STOP_ACTIVATION_PROFIT_PERCENT:
                    details['trailing_active'] = True
                    update_signal_in_db(signal['id'], {"signal_details": json.dumps(details)})
                    with signal_cache_lock:
                        if symbol in open_signals_cache:
                            open_signals_cache[symbol]['signal_details'] = details
                    send_telegram_message(f"📈 *تفعيل الوقف المتحرك* لـ `{symbol}` عند ربح `{profit_pct:.2f}%`.")

                if details.get('trailing_active') and trail_dist:
                    new_sl = max(stop_loss, current_price - trail_dist)
                    if new_sl > stop_loss:
                        update_signal_in_db(signal['id'], {"stop_loss": new_sl})
                        with signal_cache_lock:
                            if symbol in open_signals_cache:
                                open_signals_cache[symbol]['stop_loss'] = new_sl
                        send_telegram_message(f"🔧 *تحديث الوقف المتحرك* لـ `{symbol}` → `{new_sl:.6f}`")
            time.sleep(1)
        except Exception as e:
            logger.error(f"❌ [Trade Manager] Loop error: {e}", exc_info=True)
            time.sleep(2)

def update_market_state():
    global current_market_state
    try:
        btc_df = fetch_historical_data(BTC_SYMBOL, '1h', days=10)
        if btc_df is None or len(btc_df) < 200:
            logger.warning("[Market State] Insufficient BTC data for full analysis")
            return
        
        btc_df = calculate_all_features(btc_df)
        last_btc = btc_df.iloc[-1]
        
        btc_trend = "sideways"
        if last_btc['close'] > last_btc['ema200'] and last_btc['macd_hist'] > 0:
            btc_trend = "bullish"
        elif last_btc['close'] < last_btc['ema200'] and last_btc['macd_hist'] < 0:
            btc_trend = "bearish"
        
        trend_details = {}
        for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
            try:
                tf_df = fetch_historical_data(BTC_SYMBOL, tf, days=15)
                if tf_df is not None and len(tf_df) >= 50:
                    tf_df = calculate_all_features(tf_df)
                    last_tf = tf_df.iloc[-1]
                    
                    tf_trend = "sideways"
                    if last_tf['close'] > last_tf['ema50'] and last_tf['adx'] > 20:
                        tf_trend = "bullish"
                    elif last_tf['close'] < last_tf['ema50'] and last_tf['adx'] > 20:
                        tf_trend = "bearish"
                    
                    trend_details[tf] = {
                        "trend": tf_trend,
                        "adx": last_tf.get('adx', 0),
                        "rsi": last_tf.get('rsi', 50),
                        "price_change": ((last_tf['close'] - tf_df.iloc[-10]['close']) / tf_df.iloc[-10]['close']) * 100 if len(tf_df) >= 10 else 0
                    }
            except Exception as e:
                logger.error(f"[Market State] Error analyzing {tf} timeframe: {e}")
        
        with market_state_lock:
            current_market_state = {
                "btc_trend": btc_trend,
                "btc_price": last_btc['close'],
                "btc_adx": last_btc.get('adx', 0),
                "btc_rsi": last_btc.get('rsi', 50),
                "trend_details_by_tf": trend_details,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
        
        broadcast({"type": "market_state_update", "payload": current_market_state})
        
    except Exception as e:
        logger.error(f"[Market State] Error updating market state: {e}", exc_info=True)

def start_market_state_updater():
    def update_loop():
        while True:
            try:
                update_market_state()
                time.sleep(300)
            except Exception as e:
                logger.error(f"[Market State Updater] Error in update loop: {e}")
                time.sleep(60)
    
    thread = Thread(target=update_loop, daemon=True)
    thread.start()
    logger.info("[Market State] Started market state updater thread")

def update_balance():
    try:
        balance_info = client.get_asset_balance(asset='USDT')
        with balance_lock:
            global usdt_balance
            usdt_balance = float(balance_info['free'])
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
    logger.info("="*50 + "\n====== Starting Crypto Trading Bot V23.2.5 (LOT_SIZE Fix) ======\n" + "="*50)
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

    logger.info("Initial data fetch complete.")

    start_websocket()
    Thread(target=main_bot_loop, daemon=True).start()
    Thread(target=trade_management_loop, daemon=True).start()
    start_market_state_updater()
    Thread(target=update_balance_loop, daemon=True).start()

    logger.info("🌐 [Flask] Starting UI on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
