# --- Crypto Trading Bot V35.0.0 (Advanced Strategy Integration) ---
#
# وصف التعديلات الرئيسية (V35):
# 1. [دمج الاستراتيجيات المتقدمة] تم استبدال دوال الاستراتيجيات القديمة بهيكل جديد قائم على الفئات (Classes)،
#    مما يوفر تحليلًا أكثر تفصيلاً لكل إشارة.
# 2. [نظام نقاط الجودة] كل استراتيجية الآن تحسب "نقاط جودة" (Quality Score) دقيقة للإشارة،
#    مما يسمح بفلترة أكثر فعالية للصفقات المحتملة.
# 3. [إعادة هيكلة حلقة الفحص] تم تحديث حلقة الفحص الرئيسية (`main_bot_loop`) لتتوافق مع
#    الهيكل الجديد، مما يحسن من كفاءة اكتشاف الإشارات.
# 4. [الحفاظ على الوظائف الأساسية] تم الاحتفاظ بجميع المكونات الأساسية للبوت مثل لوحة التحكم،
#    الاتصال بقاعدة البيانات، WebSocket، وإدارة الصفقات.
# 5. [مرشحات أساسية موحدة] تم دمج مرشحات السوق والاتجاه والحجم في فئة أساسية مشتركة
#    لضمان تطبيق نفس معايير الأمان على جميع الاستراتيجيات.

import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import redis
import statistics
import random
from decimal import Decimal, ROUND_DOWN, getcontext
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
from typing import List, Dict, Optional, Any, Tuple
from collections import deque
import warnings
from scipy.signal import argrelextrema

# --- إعدادات التجاهل واللوجر ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

getcontext().prec = 18

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v35_5min_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV35.0.0_5min')

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
COOLDOWN_MINUTES_AFTER_SL = 30

# --- المتغيرات القابلة للتعديل ---
PAPER_TRADE_FIXED_AMOUNT_USDT: float = 10.0
FIXED_TRADE_AMOUNT_MIN_USDT: float = 4.5
FIXED_TRADE_AMOUNT_MAX_USDT: float = 6.5
trade_amount_lock = Lock()
MAX_OPEN_TRADES: int = 3
TRAILING_STOP_ACTIVATION_PROFIT_PERCENT: float = 1.0
MIN_SIGNAL_QUALITY: int = 70
AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE: bool = True
min_quality_lock = Lock()

# --- مفاتيح تفعيل الاستراتيجيات ---
USE_BB_STOCH_STRATEGY: bool = True
USE_MACD_EMA_STRATEGY: bool = True
USE_EMA_RSI_STRATEGY: bool = True
USE_PULLBACK_STRATEGY: bool = True
USE_MOMENTUM_VOLATILITY_STRATEGY: bool = True
USE_ELLIOTT_WAVE_STRATEGY: bool = True
USE_RANGE_REVERSAL_STRATEGY: bool = True

# --- إعدادات عامة (معدلة لإطار 5 دقائق) ---
SIGNAL_GENERATION_TIMEFRAME: str = '5m'
HIGHER_TIMEFRAME: str = '15m'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['5m', '15m', '1h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 7
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
    # General Filters
    "Market Volatility Filter Failed": "فلتر تقلب السوق رفض الدخول",
    "Insufficient Historical Data": "بيانات تاريخية غير كافية للفحص",
    "MinNotional Filter Failed": "قيمة الصفقة أقل من الحد الأدنى للمنصة",
    "LOT_SIZE Filter Failed": "فشل تعديل حجم الصفقة",
    "Insufficient Balance": "الرصيد غير كافي لتنفيذ الصفقة",
    "Low Quality Signal": "جودة الإشارة منخفضة",
    "Invalid Position Size": "حجم الصفقة غير صالح (الوقف أعلى من الدخول)",
    "News Filter Failed": "فلتر الأخبار: تجنب التداول وقت الأخبار",
    "Liquidity Filter Failed": "فلتر السيولة: تجنب التداول في أوقات السيولة المنخفضة",
    "Correlation Filter Failed": "فلتر الارتباط: توجد صفقة مفتوحة على عملة مرتبطة",
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

# --- دوال تهيئة الخدمات وقاعدة البيانات ---
def optimize_database():
    if not check_db_connection() or not conn: return
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

def init_db(retries: int = 5, base_delay: int = 5) -> None:
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
                    "initial_quantity": "DOUBLE PRECISION", "created_at": "TIMESTAMP WITH TIME ZONE DEFAULT NOW()"
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
            if attempt < retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info(f"[DB] Retrying connection in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.critical("❌ [DB] Failed to connect to the database after all retries. Exiting.")
                exit(1)

def check_db_connection() -> bool:
    global conn
    if conn is None or conn.closed != 0:
        logger.warning("[DB] Connection is None or closed. Re-initializing...")
        init_db()
    try:
        if conn and conn.closed == 0:
            with conn.cursor() as cur: cur.execute("SELECT 1;")
            return True
        logger.warning("[DB] Connection check failed.")
        return False
    except (OperationalError, InterfaceError) as e:
        logger.error(f"[DB] Connection lost ({e}). Attempting to reconnect...")
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

def log_rejection(symbol: str, reason: str):
    try:
        reason_ar = REJECTION_REASONS_AR.get(reason, reason)
        log_entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol, "reason": reason_ar}
        with rejection_logs_lock: rejection_logs_cache.appendleft(log_entry)
        broadcast({"type": "new_rejection", "payload": log_entry})
    except Exception as e:
        logger.error(f"❌ [Log Rejection] Error logging rejection for {symbol}: {e}", exc_info=True)

def send_enhanced_telegram_message(message: str, force: bool = False):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    max_length = 4096
    messages = [message[i:i+max_length] for i in range(0, len(message), max_length)]
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for msg in messages:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown", "disable_web_page_preview": True}
        for attempt in range(3):
            try:
                r = requests.post(url, data=payload, timeout=10)
                if r.status_code == 429:
                    retry_after = int(r.json().get("parameters", {}).get("retry_after", 1))
                    time.sleep(min(5, retry_after))
                    continue
                if r.ok: break
                else: logger.warning(f"[Telegram] HTTP {r.status_code}: {r.text}")
            except requests.exceptions.RequestException as e:
                if attempt == 2: logger.error(f"❌ [Telegram] Failed to send message after retries: {e}")
                time.sleep(1.5)

def send_trade_open_notification(symbol: str, strategy_name: str, entry_price: float, stop_loss: float,
                                target1: float, target2: float, quantity: float, is_real: bool,
                                quality_score: int, atr_percent: float, notional_value: float):
    trade_type = "حقيقية" if is_real else "ورقية"
    emoji = "🔥" if is_real else "📊"
    message = (
        f"{emoji} *صفقة {trade_type} جديدة (5 دقائق)*\n\n"
        f"*العملة:* `{symbol}`\n"
        f"*الاستراتيجية:* `{strategy_name}`\n"
        f"*جودة الإشارة:* `{quality_score}/100`\n"
        f"*تقلب السوق:* `{atr_percent:.2f}%`\n\n"
        f"*سعر الدخول:* `{entry_price:.4f}`\n"
        f"*وقف الخسارة:* `{stop_loss:.4f}`\n"
        f"*الهدف الأول:* `{target1:.4f}`\n"
        f"*الهدف الثاني:* `{target2:.4f}`\n\n"
        f"*الكمية:* `{quantity:.4f}`\n"
        f"*قيمة الصفقة:* `${notional_value:.2f}`\n"
        f"*نسبة المخاطرة:* `{((entry_price - stop_loss) / entry_price * 100):.2f}%`"
    )
    send_enhanced_telegram_message(message, force=True)

def handle_socket_message(msg):
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
                            pass
            if price_updates:
                broadcast({"type": "price_update", "payload": price_updates})
    except Exception as e:
        logger.error(f"❌ [WebSocket] Error processing message: {e}", exc_info=True)

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
        logger.info(f"✅ [API] Exchange info map created with {len(exchange_info_map)} symbols.")
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
    df_calc['atr_percent'] = (df_calc['atr'] / df_calc['close'].replace(0, 1e-9)) * 100
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
    avg_gain = gain.rolling(window=7).mean()
    avg_loss = loss.rolling(window=7).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df_calc['rsi'] = 100 - (100 / (1 + rs))
    bb_middle = df_calc['close'].rolling(window=20).mean()
    bb_std = df_calc['close'].rolling(window=20).std()
    df_calc['bb_middle'] = bb_middle
    df_calc['bb_lower'] = bb_middle - (bb_std * 2)
    df_calc['bb_upper'] = bb_middle + (bb_std * 2)
    df_calc['bb_width'] = (df_calc['bb_upper'] - df_calc['bb_lower']) / df_calc['bb_middle'].replace(0, 1e-9)
    exp1 = df_calc['close'].ewm(span=8, adjust=False).mean()
    exp2 = df_calc['close'].ewm(span=17, adjust=False).mean()
    df_calc['macd'] = exp1 - exp2
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()
    df_calc['macd_hist'] = df_calc['macd'] - df_calc['macd_signal']
    low_14 = df_calc['low'].rolling(14).min()
    high_14 = df_calc['high'].rolling(14).max()
    meaningful_range = (high_14 - low_14) > (df_calc['close'] * 0.0001)
    df_calc['stoch_k'] = np.where(meaningful_range, 100 * ((df_calc['close'] - low_14) / (high_14 - low_14).replace(0, 1e-9)), 50)
    df_calc['stoch_d'] = df_calc['stoch_k'].rolling(3).mean()
    return df_calc

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

# --- تحسين منطق وقف الخسارة وجني الأرباح ---
def calculate_dynamic_stop_loss_enhanced(df: pd.DataFrame, entry_price: float) -> float:
    last = df.iloc[-1]
    atr_value = last.get('atr', 0)
    recent_low = df['low'].tail(5).min()
    # استخدام مزيج من أدنى سعر ومضاعف ATR
    stop_loss = min(recent_low * 0.995, entry_price - (atr_value * 2.0))
    # الحد الأقصى لمسافة وقف الخسارة
    max_stop_distance = entry_price * 0.05
    if entry_price - stop_loss > max_stop_distance:
        stop_loss = entry_price - max_stop_distance
    return stop_loss

def calculate_dynamic_take_profit_enhanced(entry_price: float, stop_loss: float) -> tuple:
    risk_amount = entry_price - stop_loss
    if risk_amount <= 0: return (entry_price * 1.015, entry_price * 1.025)
    # نسب المخاطرة إلى المكافأة Scalping
    rr1, rr2 = 1.7, 3.0
    target1 = entry_price + (risk_amount * rr1)
    target2 = entry_price + (risk_amount * rr2)
    return target1, target2

# ==============================================================================
# === استراتيجيات التداول المحسنة مع آليات الحماية المتقدمة (V35) ===
# ==============================================================================

class BaseStrategyFilter:
    """مرشحات أساسية مشتركة بين جميع الاستراتيجيات"""
    @staticmethod
    def check_basic_market_structure(df: pd.DataFrame) -> Tuple[bool, str]:
        if len(df) < 50: return False, "بيانات غير كافية"
        last = df.iloc[-1]
        required_indicators = ['close', 'volume', 'atr_percent', 'adx', 'rsi']
        for indicator in required_indicators:
            if pd.isna(last.get(indicator, np.nan)):
                return False, f"مؤشر {indicator} غير متوفر"
        atr_percent = last.get('atr_percent', 0)
        if not (0.3 <= atr_percent <= 3.5):
            return False, f"تقلب خطير: {atr_percent:.2f}%"
        if last['volume'] <= 0: return False, "حجم تداول صفر"
        return True, "البنية الأساسية سليمة"

    @staticmethod
    def check_trend_quality(df: pd.DataFrame) -> Tuple[bool, str, float]:
        last = df.iloc[-1]
        trend_score = 0
        adx = last.get('adx', 0)
        if adx > 25: trend_score += 30
        elif adx > 20: trend_score += 20
        elif adx > 15: trend_score += 10
        ema_consistent = (last.get('ema9', 0) > last.get('ema21', 0) > last.get('ema50', 0))
        if ema_consistent: trend_score += 25
        if last['close'] > last.get('ema21', 0): trend_score += 20
        macd_hist = last.get('macd_hist', 0)
        if macd_hist > 0: trend_score += 25
        min_trend_score = 60
        return trend_score >= min_trend_score, f"نقاط الاتجاه: {trend_score}/100", trend_score

    @staticmethod
    def check_volume_conviction(df: pd.DataFrame) -> Tuple[bool, str]:
        last = df.iloc[-1]
        volume_ma = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = last['volume']
        volume_ratio = current_volume / volume_ma if volume_ma > 0 else 0
        if volume_ratio < 1.2: return False, f"حجم ضعيف: {volume_ratio:.2f}x"
        recent_volumes = df['volume'].tail(3)
        recent_volume_ma = df['volume'].rolling(20).mean().tail(3)
        volume_consistency = sum(rv > rvm for rv, rvm in zip(recent_volumes, recent_volume_ma))
        if volume_consistency < 2: return False, "عدم اتساق في الحجم"
        return True, f"حجم مؤكد: {volume_ratio:.2f}x"

class EnhancedBBStochStrategy:
    """استراتيجية BB + Stochastic مع حماية متقدمة"""
    def __init__(self):
        self.name = "Enhanced_BB_Stoch"
        self.min_quality_score = 75
        self.base_filter = BaseStrategyFilter()

    def analyze(self, df: pd.DataFrame, mtf_trend: Dict) -> Tuple[bool, str, int]:
        basic_ok, basic_reason = self.base_filter.check_basic_market_structure(df)
        if not basic_ok: return False, basic_reason, 0
        bb_ok, bb_reason, bb_score = self._check_bollinger_conditions(df)
        if not bb_ok: return False, bb_reason, bb_score
        stoch_ok, stoch_reason, stoch_score = self._check_stochastic_conditions(df)
        if not stoch_ok: return False, stoch_reason, stoch_score
        confirm_ok, confirm_reason, confirm_score = self._check_confirmations(df, mtf_trend)
        if not confirm_ok: return False, confirm_reason, confirm_score
        total_score = (bb_score + stoch_score + confirm_score) // 3
        return total_score >= self.min_quality_score, f"تم تأكيد الإشارة: {total_score}/100", total_score

    def _check_bollinger_conditions(self, df: pd.DataFrame) -> Tuple[bool, str, int]:
        last, prev, score = df.iloc[-1], df.iloc[-2], 0
        bb_lower = last.get('bb_lower', 0)
        price_to_lower_ratio = last['close'] / bb_lower if bb_lower > 0 else 1
        if not (0.998 <= price_to_lower_ratio <= 1.02): return False, f"السعر بعيد عن BB السفلي: {price_to_lower_ratio:.4f}", 0
        score += 25
        bb_width, bb_width_ma = last.get('bb_width', 0), df['bb_width'].rolling(20).mean().iloc[-1]
        if bb_width < bb_width_ma * 0.7: return False, f"عرض BB ضيق جداً: {bb_width:.4f}", score
        score += 25
        touched_lower = any(df['low'].tail(5) <= df['bb_lower'].tail(5) * 1.002)
        if not touched_lower: return False, "لم يتم لمس الحد السفلي مؤخراً", score
        score += 25
        if last['close'] <= prev['close']: return False, "لا توجد إشارة انتعاش", score
        score += 25
        return True, f"شروط BB مُستوفاة: {score}/100", score

    def _check_stochastic_conditions(self, df: pd.DataFrame) -> Tuple[bool, str, int]:
        last, prev, prev2, score = df.iloc[-1], df.iloc[-2], df.iloc[-3], 0
        stoch_k, stoch_d, prev_stoch_k = last.get('stoch_k', 50), last.get('stoch_d', 50), prev.get('stoch_k', 50)
        if not (prev_stoch_k < 25 or prev2.get('stoch_k', 50) < 25): return False, "لم يصل لمنطقة تشبع بيعي", 0
        score += 30
        if not (stoch_k > prev_stoch_k and stoch_k > stoch_d): return False, "لا يوجد انتعاش في Stochastic", score
        score += 25
        if stoch_k > 75: return False, f"Stochastic مرتفع جداً: {stoch_k:.1f}", score
        score += 20
        stoch_momentum = stoch_k - prev_stoch_k
        if stoch_momentum < 3: return False, f"انتعاش ضعيف: +{stoch_momentum:.1f}", score
        score += 25
        return True, f"شروط Stochastic مُستوفاة: {score}/100", score

    def _check_confirmations(self, df: pd.DataFrame, mtf_trend: Dict) -> Tuple[bool, str, int]:
        last, score = df.iloc[-1], 0
        volume_ok, volume_reason = self.base_filter.check_volume_conviction(df)
        if volume_ok: score += 30
        else: return False, f"فشل تأكيد الحجم: {volume_reason}", 0
        rsi = last.get('rsi', 50)
        if 25 <= rsi <= 45: score += 25
        elif rsi > 70: return False, f"RSI مرتفع جداً: {rsi:.1f}", score
        if last.get('macd_hist', 0) > df.iloc[-2].get('macd_hist', 0): score += 20
        bullish_timeframes = sum(1 for trend in mtf_trend.values() if trend == 'bullish')
        total_timeframes = len(mtf_trend) or 1
        mtf_ratio = bullish_timeframes / total_timeframes
        if mtf_ratio >= 0.6: score += 25
        else: return False, f"ضعف في الأطر الزمنية: {mtf_ratio:.1%}", score
        return True, f"التأكيدات مُستوفاة: {score}/100", score

class EnhancedMACDEMAStrategy:
    """استراتيجية MACD + EMA مع فلاتر متقدمة"""
    def __init__(self):
        self.name = "Enhanced_MACD_EMA"
        self.min_quality_score = 80
        self.base_filter = BaseStrategyFilter()

    def analyze(self, df: pd.DataFrame, mtf_trend: Dict) -> Tuple[bool, str, int]:
        basic_ok, basic_reason = self.base_filter.check_basic_market_structure(df)
        if not basic_ok: return False, basic_reason, 0
        trend_ok, trend_reason, trend_score = self.base_filter.check_trend_quality(df)
        if not trend_ok: return False, f"اتجاه ضعيف: {trend_reason}", trend_score
        macd_ok, macd_reason, macd_score = self._check_macd_crossover(df)
        if not macd_ok: return False, macd_reason, macd_score
        ema_ok, ema_reason, ema_score = self._check_ema_alignment(df)
        if not ema_ok: return False, ema_reason, ema_score
        momentum_ok, momentum_reason, momentum_score = self._check_momentum_confirmations(df)
        if not momentum_ok: return False, momentum_reason, momentum_score
        total_score = (trend_score + macd_score + ema_score + momentum_score) // 4
        return total_score >= self.min_quality_score, f"MACD+EMA مؤكدة: {total_score}/100", total_score

    def _check_macd_crossover(self, df: pd.DataFrame) -> Tuple[bool, str, int]:
        if len(df) < 5: return False, "بيانات غير كافية للMACD", 0
        score, last, prev = 0, df.iloc[-1], df.iloc[-2]
        macd_hist, prev_macd_hist = last.get('macd_hist', 0), prev.get('macd_hist', 0)
        if not (prev_macd_hist <= 0 and macd_hist > 0): return False, "لا يوجد تقاطع MACD حديث", 0
        score += 40
        crossover_strength = macd_hist - prev_macd_hist
        if crossover_strength < abs(prev_macd_hist) * 0.5: return False, f"تقاطع ضعيف: {crossover_strength:.6f}", score
        score += 30
        if last.get('macd', 0) <= prev.get('macd', 0): return False, "MACD ليس في اتجاه صاعد", score
        score += 30
        return True, f"تقاطع MACD قوي: {score}/100", score

    def _check_ema_alignment(self, df: pd.DataFrame) -> Tuple[bool, str, int]:
        last, score = df.iloc[-1], 0
        ema9, ema21, ema50, ema200 = last.get('ema9', 0), last.get('ema21', 0), last.get('ema50', 0), last.get('ema200', 0)
        if not (ema9 > ema21 > ema50): return False, "ترتيب EMAs خاطئ", 0
        score += 40
        if last['close'] <= ema21: return False, "السعر تحت EMA21", score
        score += 30
        if ema50 <= ema200: return False, "الاتجاه العام هابط", score
        score += 30
        return True, f"ترتيب EMAs صحيح: {score}/100", score

    def _check_momentum_confirmations(self, df: pd.DataFrame) -> Tuple[bool, str, int]:
        last, score = df.iloc[-1], 0
        rsi = last.get('rsi', 50)
        if 45 <= rsi <= 70: score += 30
        elif rsi > 80: return False, f"RSI مُفرط الشراء: {rsi:.1f}", 0
        adx = last.get('adx', 0)
        if adx > 25: score += 35
        elif adx > 20: score += 25
        else: return False, f"ADX ضعيف: {adx:.1f}", score
        volume_ok, _ = self.base_filter.check_volume_conviction(df)
        if volume_ok: score += 35
        else: return False, "فشل تأكيد الحجم", score
        return True, f"الزخم مؤكد: {score}/100", score

class EnhancedEMARSIStrategy:
    """استراتيجية EMA + RSI مع حماية من الإشارات الكاذبة"""
    def __init__(self):
        self.name = "Enhanced_EMA_RSI"
        self.min_quality_score = 75
        self.base_filter = BaseStrategyFilter()

    def analyze(self, df: pd.DataFrame, mtf_trend: Dict) -> Tuple[bool, str, int]:
        basic_ok, basic_reason = self.base_filter.check_basic_market_structure(df)
        if not basic_ok: return False, basic_reason, 0
        rsi_ok, rsi_reason, rsi_score = self._check_rsi_conditions(df)
        if not rsi_ok: return False, rsi_reason, rsi_score
        ema_ok, ema_reason, ema_score = self._check_ema_structure(df)
        if not ema_ok: return False, ema_reason, ema_score
        convergence_ok, conv_reason, conv_score = self._check_convergence(df, mtf_trend)
        if not convergence_ok: return False, conv_reason, conv_score
        total_score = (rsi_score + ema_score + conv_score) // 3
        return total_score >= self.min_quality_score, f"EMA+RSI مؤكدة: {total_score}/100", total_score

    def _check_rsi_conditions(self, df: pd.DataFrame) -> Tuple[bool, str, int]:
        last, prev, score = df.iloc[-1], df.iloc[-2], 0
        rsi, prev_rsi, adx = last.get('rsi', 50), prev.get('rsi', 50), last.get('adx', 0)
        if adx > 30: optimal_range, score_range = (35, 75), (40, 70)
        elif adx > 20: optimal_range, score_range = (40, 70), (45, 65)
        else: optimal_range, score_range = (45, 65), (48, 62)
        if not (optimal_range[0] <= rsi <= optimal_range[1]): return False, f"RSI خارج النطاق {optimal_range}: {rsi:.1f}", 0
        if score_range[0] <= rsi <= score_range[1]: score += 40
        else: score += 25
        if rsi <= prev_rsi: return False, f"RSI ليس صاعد: {rsi:.1f} vs {prev_rsi:.1f}", score
        score += 30
        if rsi > 75 or rsi < 30: return False, f"RSI متطرف: {rsi:.1f}", score
        score += 30
        return True, f"RSI مناسب: {score}/100", score

    def _check_ema_structure(self, df: pd.DataFrame) -> Tuple[bool, str, int]:
        last, prev, score = df.iloc[-1], df.iloc[-2], 0
        ema9, ema21, ema50, ema200 = last.get('ema9', 0), last.get('ema21', 0), last.get('ema50', 0), last.get('ema200', 0)
        if not (ema9 > ema21 and ema50 > ema200): return False, "ترتيب EMAs غير صحيح", 0
        score += 30
        if last['close'] <= ema21: return False, "السعر تحت EMA21", score
        score += 25
        if ema21 <= prev.get('ema21', 0): return False, "EMA21 ليس صاعد", score
        score += 25
        ema_spread = (ema9 - ema21) / ema21 if ema21 > 0 else 0
        if abs(ema_spread) < 0.003: return False, f"EMAs متقاربة جداً: {ema_spread:.1%}", score
        score += 20
        return True, f"بنية EMAs سليمة: {score}/100", score

    def _check_convergence(self, df: pd.DataFrame, mtf_trend: Dict) -> Tuple[bool, str, int]:
        last, score = df.iloc[-1], 0
        if last.get('macd_hist', 0) <= 0: return False, "MACD سالب", 0
        score += 25
        volume_ok, volume_reason = self.base_filter.check_volume_conviction(df)
        if not volume_ok: return False, f"حجم ضعيف: {volume_reason}", score
        score += 30
        bullish_count = sum(1 for trend in mtf_trend.values() if trend == 'bullish')
        total_frames = len(mtf_trend) or 1
        mtf_ratio = bullish_count / total_frames
        if mtf_ratio >= 0.7: score += 25
        elif mtf_ratio >= 0.5: score += 15
        else: return False, f"ضعف في الأطر الزمنية: {mtf_ratio:.1%}", score
        adx = last.get('adx', 0)
        if adx <= 20: return False, f"ADX ضعيف: {adx:.1f}", score
        score += 20
        return True, f"تقارب الإشارات: {score}/100", score

class EnhancedPullbackStrategy:
    """استراتيجية Pullback مع فلاتر صارمة"""
    def __init__(self):
        self.name = "Enhanced_Pullback"
        self.min_quality_score = 80
        self.base_filter = BaseStrategyFilter()

    def analyze(self, df: pd.DataFrame, mtf_trend: Dict) -> Tuple[bool, str, int]:
        basic_ok, basic_reason = self.base_filter.check_basic_market_structure(df)
        if not basic_ok: return False, basic_reason, 0
        trend_ok, trend_reason, trend_score = self._check_strong_trend(df, mtf_trend)
        if not trend_ok: return False, trend_reason, trend_score
        pullback_ok, pullback_reason, pullback_score = self._check_pullback_quality(df)
        if not pullback_ok: return False, pullback_reason, pullback_score
        recovery_ok, recovery_reason, recovery_score = self._check_recovery_signals(df)
        if not recovery_ok: return False, recovery_reason, recovery_score
        total_score = (trend_score + pullback_score + recovery_score) // 3
        return total_score >= self.min_quality_score, f"Pullback مؤكد: {total_score}/100", total_score

    def _check_strong_trend(self, df: pd.DataFrame, mtf_trend: Dict) -> Tuple[bool, str, int]:
        trend_ok, _, trend_score = self.base_filter.check_trend_quality(df)
        if not trend_ok or trend_score < 75: return False, f"الاتجاه ليس قوياً بما يكفي: {trend_score}/100", int(trend_score)
        bullish_timeframes = sum(1 for trend in mtf_trend.values() if trend == 'bullish')
        if bullish_timeframes < len(mtf_trend): return False, "الأطر الزمنية الأعلى ليست كلها صاعدة", int(trend_score)
        return True, f"اتجاه قوي مؤكد: {trend_score}/100", int(trend_score)

    def _check_pullback_quality(self, df: pd.DataFrame) -> Tuple[bool, str, int]:
        score = 0
        ema21, ema50 = df['ema21'], df['ema50']
        touched_ema21 = any(df['low'].tail(5) <= ema21.tail(5) * 1.005)
        if not touched_ema21: return False, "لم يلمس EMA21", 0
        score += 30
        not_below_ema50 = all(df['low'].tail(5) > ema50.tail(5))
        if not not_below_ema50: return False, "انخفض تحت EMA50", score
        score += 30
        recent_volumes = df['volume'].tail(5)
        volume_ma20 = df['volume'].rolling(20).mean().tail(5)
        low_volume_pullback = all(v < v_ma for v, v_ma in zip(recent_volumes, volume_ma20))
        if not low_volume_pullback: return False, "التراجع بحجم تداول عالي", score
        score += 40
        return True, "تراجع عالي الجودة", score

    def _check_recovery_signals(self, df: pd.DataFrame) -> Tuple[bool, str, int]:
        last, prev, score = df.iloc[-1], df.iloc[-2], 0
        if not (last['close'] > last['open'] and last['close'] > prev['close']): return False, "لا توجد شمعة انعكاس", 0
        score += 30
        volume_ok, _ = self.base_filter.check_volume_conviction(df)
        if not volume_ok: return False, "حجم الانتعاش ضعيف", score
        score += 35
        if not last['macd_hist'] > prev['macd_hist']: return False, "زخم MACD لا يتزايد", score
        score += 35
        return True, "إشارات الانتعاش موجودة", score

# ==============================================================================
# --- نهاية قسم الاستراتيجيات المحسنة ---
# ==============================================================================

def get_formatted_quantity(symbol: str, quantity: Decimal) -> str:
    try:
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info: return f"{quantity.normalize()}"
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if not lot_size_filter: return f"{quantity.normalize()}"
        step_size = Decimal(lot_size_filter['stepSize'])
        formatted_quantity = quantity.quantize(step_size, rounding=ROUND_DOWN)
        return f"{formatted_quantity.normalize()}"
    except Exception as e:
        logger.error(f"❌ [{symbol}] Error formatting quantity: {e}.")
        return str(quantity)

def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> Optional[Decimal]:
    try:
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info: return None
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if not lot_size_filter: return Decimal(str(quantity))
        step_size = Decimal(lot_size_filter['stepSize'])
        min_qty = Decimal(lot_size_filter['minQty'])
        quantity_dec = Decimal(str(quantity))
        if quantity_dec < min_qty: return None
        adjusted_quantity = (quantity_dec - (quantity_dec % step_size))
        if adjusted_quantity < min_qty: return None
        return adjusted_quantity
    except Exception as e:
        logger.error(f"❌ [{symbol}] CRITICAL ERROR adjusting quantity: {e}", exc_info=True)
        return None

def calculate_position_size(symbol: str, entry_price: float, available_balance: float, is_real: bool) -> Optional[Decimal]:
    desired_usdt_amount = PAPER_TRADE_FIXED_AMOUNT_USDT if not is_real else random.uniform(FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT)
    try:
        dec_entry = Decimal(str(entry_price))
        if dec_entry <= 0: return None
        dec_balance = Decimal(str(available_balance))
        dec_desired_amount = Decimal(str(desired_usdt_amount))
        if is_real and dec_desired_amount > dec_balance:
            log_rejection(symbol, "Insufficient Balance")
            return None
        initial_quantity = dec_desired_amount / dec_entry
        adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(initial_quantity))
        if adjusted_quantity is None: adjusted_quantity = Decimal('0')
        notional_value = adjusted_quantity * dec_entry
        symbol_info = exchange_info_map.get(symbol)
        if symbol_info:
            min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] in ('MIN_NOTIONAL', 'NOTIONAL')), None)
            if min_notional_filter:
                min_notional = Decimal(min_notional_filter.get('minNotional', min_notional_filter.get('notional', '5.0')))
                if notional_value < min_notional:
                    required_notional = min_notional * Decimal('1.01')
                    if is_real and required_notional > dec_balance:
                        log_rejection(symbol, "Insufficient Balance")
                        return None
                    new_quantity = required_notional / dec_entry
                    adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(new_quantity))
                    if adjusted_quantity is None or adjusted_quantity <= 0:
                        log_rejection(symbol, "MinNotional Filter Failed")
                        return None
        if adjusted_quantity is None or adjusted_quantity <= 0: return None
        return adjusted_quantity
    except Exception as e:
        logger.error(f"❌ [{symbol}] Unhandled exception in calculate_position_size: {e}", exc_info=True)
        return None

def create_trade_signal(symbol: str, df: pd.DataFrame, strategy_name: str, quality_score: int):
    with min_quality_lock: min_score = MIN_SIGNAL_QUALITY
    if quality_score < min_score:
        log_rejection(symbol, f"جودة الإشارة ({quality_score}) أقل من المطلوب ({min_score})")
        return

    logger.info(f"✅ [Signal Found] {symbol} via {strategy_name} with score {quality_score}/100. Proceeding to create trade.")

    entry_price = df.iloc[-1]['close']
    stop_loss_price = calculate_dynamic_stop_loss_enhanced(df, entry_price)
    target_price_1, target_price_2 = calculate_dynamic_take_profit_enhanced(entry_price, stop_loss_price)

    if stop_loss_price >= entry_price:
        log_rejection(symbol, "Invalid Position Size")
        return

    with trading_mode_lock: is_real = not paper_trading_mode
    with balance_lock: current_real_balance = usdt_balance

    quantity_dec = calculate_position_size(symbol, entry_price, current_real_balance, is_real)
    if quantity_dec is None or quantity_dec <= 0:
        logger.error(f"❌ [{symbol}] Position size calculation failed. Trade rejected.")
        return

    notional_value = float(quantity_dec) * entry_price
    signal_details = {
        "quality_score": quality_score,
        "atr_percent": df.iloc[-1].get('atr_percent', 0)
    }

    if is_real:
        try:
            formatted_quantity = get_formatted_quantity(symbol, quantity_dec)
            logger.info(f"💰 [Real Trade] Placing LIVE MARKET BUY order for {formatted_quantity} of {symbol}")
            order = client.create_order(symbol=symbol, side=Client.SIDE_BUY, type=Client.ORDER_TYPE_MARKET, quantity=formatted_quantity)
            avg_fill_price = sum(Decimal(f['price']) * Decimal(f['qty']) for f in order.get('fills', [])) / max(sum(Decimal(f['qty']) for f in order.get('fills', [])), Decimal('1e-8'))
            final_quantity, order_id = Decimal(order.get('executedQty')), order.get('orderId')
            save_signal_to_db(symbol, float(avg_fill_price), stop_loss_price, target_price_1, target_price_2, strategy_name, True, float(final_quantity), {**signal_details, "avg_fill": float(avg_fill_price)}, order_id)
            send_trade_open_notification(symbol, strategy_name, float(avg_fill_price), stop_loss_price, target_price_1, target_price_2, float(final_quantity), is_real, quality_score, df.iloc[-1].get('atr_percent', 0), notional_value)
        except Exception as e:
            logger.error(f"❌ [Real Trade] CRITICAL ERROR creating real trade for {symbol}: {e}", exc_info=True)
    else: # Paper Trading
        save_signal_to_db(symbol, entry_price, stop_loss_price, target_price_1, target_price_2, strategy_name, False, float(quantity_dec), signal_details)
        send_trade_open_notification(symbol, strategy_name, entry_price, stop_loss_price, target_price_1, target_price_2, float(quantity_dec), is_real, quality_score, df.iloc[-1].get('atr_percent', 0), notional_value)

def save_signal_to_db(symbol: str, entry_price: float, stop_loss: float, target1: float, target2: float, strategy_name: str, is_real: bool, quantity: float, signal_details: Dict, order_id: Optional[str] = None):
    try:
        if not (check_db_connection() and conn): return
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price_1, target_price_2, stop_loss, status, strategy_name, is_real_trade, quantity, initial_quantity, signal_details, order_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
            """, (symbol, entry_price, target1, target2, stop_loss, 'open', strategy_name, is_real, quantity, quantity, json.dumps(signal_details, cls=NpEncoder), order_id))
            new_id = cur.fetchone()['id']
        conn.commit()
        signal_data = {'id': new_id, 'symbol': symbol, 'entry_price': entry_price, 'target_price_1': target1, 'target_price_2': target2, 'stop_loss': stop_loss, 'status': 'open', 'strategy_name': strategy_name, 'is_real_trade': is_real, 'quantity': quantity, 'initial_quantity': quantity, 'signal_details': signal_details, 'order_id': order_id, 'created_at': datetime.now(timezone.utc)}
        with signal_cache_lock: open_signals_cache[symbol] = signal_data
        broadcast({"type": "new_signal", "payload": signal_data})
    except Exception as e:
        logger.error(f"❌ [DB] CRITICAL ERROR saving signal for {symbol}: {e}", exc_info=True)
        if conn: conn.rollback()

# --- HTML Templates (remain unchanged) ---
DASHBOARD_TEMPLATE = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" />
<title>لوحة التحكم - بوت 5 دقائق (V35)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root{--bg:#0b1020;--panel:#121b36;--accent:#3aa0ff;--ok:#15c46a;--warn:#ff9f1a;--bad:#ff4757;--muted:#8aa0c8;}
body{margin:0;background:var(--bg);color:#e8f1ff;font-family:system-ui,sans-serif}
.container{max-width:1600px;margin:0 auto;padding:16px;display:flex;flex-direction:column;gap:16px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between}
h1{font-size:18px;margin:0;font-weight:700}
.main-layout{display:grid;grid-template-columns:1fr;gap:16px;}
@media(min-width: 1000px){.main-layout{grid-template-columns:1fr 350px;}}
.card{background:var(--panel);border:1px solid #1e2c52;border-radius:14px;box-shadow:0 8px 30px rgba(0,0,0,.25);overflow:hidden}
.card h2{margin:0;padding:12px 14px;border-bottom:1px solid #1e2c52;font-size:14px;}
.card-body{padding:12px}
.btn{border:1px solid #2a3a68;background:#0f1b3b;color:#d9e7ff;padding:10px 14px;border-radius:10px;cursor:pointer;font-weight:700;}
.signals-grid{display:grid;grid-template-columns:repeat(auto-fill, minmax(300px, 1fr));gap:10px;}
.signal{display:grid;grid-template-columns:1fr auto;gap:8px;padding:10px;border:1px solid #24335f;border-radius:12px;background:#0d1730;}
.sig-title{font-weight:700}.sig-meta{font-size:12px;color:var(--muted)}
.price{font-size: 16px; font-weight: bold;}
.progress{height:8px;background:#0b1126;border:1px solid #233056;border-radius:999px;overflow:hidden; margin-top: 6px;}
.progress>span{display:block;height:100%;}
.kv{display:grid;grid-template-columns:auto 1fr;gap:6px 10px; align-items: center;}
.table{width:100%;border-collapse:separate;border-spacing:0 8px; table-layout: fixed;}
.table td{padding:8px;background:#0d1730;border:1px solid #24335f; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}
.switch{display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;border:1px solid #2a3a68;background:#0f1b3b;cursor:pointer;}
.switch input{display:none}
.switch .dot{width:14px;height:14px;border-radius:50%;background:#6a7fb2;transition:.2s}
.switch input:checked + .dot{background:#24d08a;}
.small{font-size:12px;color:#a8bfeb}
.loading-spinner { border: 3px solid rgba(255, 255, 255, 0.1); border-radius: 50%; border-top: 3px solid var(--accent); width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 20px auto; }
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="container">
<header><h1>لوحة التحكم • بوت 5 دقائق V35</h1></header>
<div class="main-layout">
<div class="left-column">
<div class="card"><h2>الصفقات المفتوحة <span class="small" id="signalCount">(0)</span></h2><div class="card-body"><div id="signals" class="signals-grid"><div class="loading-spinner"></div></div></div></div>
</div>
<div class="right-column">
<div class="card"><h2>التحكم والحالة</h2><div class="card-body"><div style="display:flex;gap:8px;flex-wrap:wrap"><label class="switch"><input id="toggleTrading" type="checkbox" /><span class="dot"></span><span class="small">تشغيل التداول</span></label></div><div class="kv" style="margin-top:12px"><div>الرصيد (USDT):</div><div id="balance">—</div><div>عدد الصفقات:</div><div id="openCount">—</div></div></div></div>
<div class="card"><h2>سجل الرفض</h2><div class="card-body" style="padding:0; max-height: 250px; overflow-y: auto;"><table class="table" id="rejections"><tbody></tbody></table></div></div>
<div class="card"><h2>سجل الأحداث</h2><div class="card-body" style="padding:0; max-height: 250px; overflow-y: auto;"><table class="table" id="events"><tbody></tbody></table></div></div>
</div>
</div>
</div>
<script>
const qs=s=>document.querySelector(s);let lastPrices={};let openSignals={};
function fmt(n){return n==null?'—':(+n).toLocaleString('en-US',{maximumFractionDigits:6});}
function renderSignal(s){const c=s.current_price||lastPrices[s.symbol]||s.entry_price;const e=s.entry_price,t=s.target_price_1,l=s.stop_loss;let p=0,o='transparent',i='';if(c>=e&&t>e){p=Math.min(100,((c-e)/(t-e))*100);o='linear-gradient(90deg, var(--ok), #3fd1b0)';i=`التقدم: ${p.toFixed(1)}%`}else if(c<e&&e>l){p=Math.min(100,((e-c)/(e-l))*100);o='linear-gradient(90deg, var(--bad), #ff6b7a)';i=`اقتراب من الوقف: ${p.toFixed(1)}%`}const q=(s.signal_details&&s.signal_details.quality_score)?s.signal_details.quality_score:0;const n=q>75?'var(--ok)':q>55?'var(--warn)':'var(--bad)';const a=s.strategy_name.replace(/_/g," ").replace("Strategy","");return`<div class=signal id=signal-${s.id} data-symbol=${s.symbol}><div><div class=sig-title>${s.symbol}</div><div class=sig-meta>${a} | <span style="color:${n};font-weight:bold">⭐ ${q}/100</span></div></div><div style=text-align:end><div class=price>${fmt(c)}</div></div><div class=progress title="${i}"><span style=width:${p.toFixed(2)}%;background:${o};></span></div></div>`}
function renderAllSignals(s){const e=qs('#signals');if(!s||s.length===0){e.innerHTML='<p style=text-align:center;color:var(--muted);>لا توجد صفقات مفتوحة.</p>';return}e.innerHTML=s.map(renderSignal).join('')}
function updateSingleSignal(s){const e=qs(`#signal-${s.id}`);if(e){e.outerHTML=renderSignal(s)}else{qs('#signals').insertAdjacentHTML('afterbegin',renderSignal(s))}}
function updatePrices(p){for(const[s,c]of Object.entries(p)){const e=document.querySelectorAll(`.signal[data-symbol="${s}"]`);e.forEach(e=>{const t=e.querySelector('.price');if(t)t.textContent=fmt(c)});lastPrices[s]=c}}
function addNotification(n,p=true){const e=qs('#events tbody'),t=`<tr><td>${new Date(n.timestamp).toLocaleTimeString('ar-EG')}</td><td>${n.message||''}</td></tr>`;if(p){e.insertAdjacentHTML('afterbegin',t);if(e.rows.length>20)e.deleteRow(-1)}else{e.insertAdjacentHTML('beforeend',t)}}
function addRejection(r,p=true){const e=qs('#rejections tbody'),t=`<tr><td>${new Date(r.timestamp).toLocaleTimeString('ar-EG')}</td><td>${r.symbol||''}</td><td>${r.reason||''}</td></tr>`;if(p){e.insertAdjacentHTML('afterbegin',t);if(e.rows.length>30)e.deleteRow(-1)}else{e.insertAdjacentHTML('beforeend',t)}}
async function initDashboard(){try{const res=await fetch('/api/dashboard');const data=await res.json();qs('#toggleTrading').checked=!!data.trading_enabled;qs('#balance').textContent=fmt(data.usdt_balance);qs('#rejections tbody').innerHTML='';data.rejections.forEach(r=>addRejection(r,false));qs('#events tbody').innerHTML='';data.notifications.forEach(n=>addNotification(n,false));openSignals=data.open_trades.reduce((a,s)=>{a[s.id]=s;return a},{});renderAllSignals(data.open_trades);qs('#openCount').textContent=data.open_trades.length;qs('#signalCount').textContent=`(${data.open_trades.length})`}catch(e){console.error(e);qs('#signals').innerHTML='<p>فشل تحميل البيانات.</p>'}}
function setupWebSocket(){const p=window.location.protocol==='https:'?'wss:':'ws:';const u=`${p}//${window.location.host}/ws`;const s=new WebSocket(u);s.onopen=()=>console.log("WebSocket connected");s.onmessage=e=>{const d=JSON.parse(e.data);switch(d.type){case'price_update':updatePrices(d.payload);break;case'new_signal':openSignals[d.payload.id]=d.payload;updateSingleSignal(d.payload);break;case'trade_closed':const el=qs(`#signal-${d.payload.signal_id}`);if(el)el.remove();delete openSignals[d.payload.signal_id];break;case'new_notification':addNotification(d.payload);break;case'new_rejection':addRejection(d.payload);break}};s.onclose=()=>{console.log("WebSocket closed, reconnecting...");setTimeout(setupWebSocket,3000)};s.onerror=e=>console.error("WebSocket error:",e)}
async function toggleTrading(){await fetch('/toggle_trading',{method:'POST'});}
qs('#toggleTrading').addEventListener('change',toggleTrading);
document.addEventListener('DOMContentLoaded',()=>{initDashboard();setupWebSocket();});
</script>
</body>
</html>
"""

# --- مسارات Flask ---
@app.route('/')
def dashboard(): return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    with trading_status_lock: trading_enabled = is_trading_enabled
    with balance_lock: current_balance = usdt_balance
    with notifications_lock: notifications = list(notifications_cache)
    with rejection_logs_lock: rejections = list(rejection_logs_cache)
    with signal_cache_lock: open_trades = list(open_signals_cache.values())
    return jsonify({
        "trading_enabled": trading_enabled, "usdt_balance": current_balance,
        "notifications": notifications, "rejections": rejections,
        "open_trades": open_trades, "server_time": datetime.now(timezone.utc).isoformat()
    })

@app.route('/toggle_trading', methods=['POST'])
def toggle_trading():
    global is_trading_enabled
    with trading_status_lock: is_trading_enabled = not is_trading_enabled
    status_msg = "enabled" if is_trading_enabled else "disabled"
    log_and_notify("info", f"Trading has been {status_msg}.", "TRADING_STATUS")
    return jsonify({"status": "success", "trading_enabled": is_trading_enabled})

@sock.route('/ws')
def ws(ws_client):
    logger.info("WebSocket client connected.")
    with ws_clients_lock: ws_clients.append(ws_client)
    try:
        while True: ws_client.receive(timeout=30)
    except Exception: logger.info("WebSocket client disconnected.")
    finally:
        with ws_clients_lock:
            if ws_client in ws_clients: ws_clients.remove(ws_client)

def get_mtf_trend(symbol: str) -> Dict[str, str]:
    trends = {}
    timeframes = {'5m': 7, '15m': 10}
    for tf, days in timeframes.items():
        try:
            df = fetch_historical_data(symbol, tf, days)
            if df is None or len(df) < 50:
                trends[tf] = 'unknown'; continue
            df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
            if df.iloc[-1]['close'] > df.iloc[-1]['ema50']: trends[tf] = 'bullish'
            else: trends[tf] = 'bearish'
        except Exception: trends[tf] = 'unknown'
    return trends

def main_bot_loop():
    logger.info("🚀 [Main Loop] Starting signal scanning loop...")
    
    # Instantiate all strategies
    all_strategies = [
        EnhancedBBStochStrategy(),
        EnhancedMACDEMAStrategy(),
        EnhancedEMARSIStrategy(),
        EnhancedPullbackStrategy(),
        # Add wrappers for old strategies if they need to be kept
    ]

    while True:
        try:
            # Wait for the start of the next 5-minute candle
            now = datetime.now(timezone.utc)
            seconds_to_wait = (5 - (now.minute % 5)) * 60 - now.second - 2 # Scan 2 seconds before close
            if seconds_to_wait > 0:
                logger.info(f"Waiting {seconds_to_wait:.0f} seconds for the next candle...")
                time.sleep(seconds_to_wait)

            with trading_status_lock:
                if not is_trading_enabled:
                    time.sleep(5)
                    continue

            logger.info("="*20 + f" Starting New Scan Cycle at {datetime.now(timezone.utc)} " + "="*20)
            
            # Filter active strategies based on global flags
            active_strategies = []
            if USE_BB_STOCH_STRATEGY: active_strategies.append(all_strategies[0])
            if USE_MACD_EMA_STRATEGY: active_strategies.append(all_strategies[1])
            if USE_EMA_RSI_STRATEGY: active_strategies.append(all_strategies[2])
            if USE_PULLBACK_STRATEGY: active_strategies.append(all_strategies[3])

            for symbol in validated_symbols_to_scan:
                with signal_cache_lock:
                    if len(open_signals_cache) >= MAX_OPEN_TRADES:
                        logger.info(f"Max open trades ({MAX_OPEN_TRADES}) reached. Pausing scan.")
                        break
                    if symbol in open_signals_cache:
                        continue
                
                df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                if df is None or len(df) < 200:
                    continue
                
                df_featured = calculate_all_features(df)
                mtf_trend = get_mtf_trend(symbol)

                for strategy in active_strategies:
                    is_signal, reason, quality_score = strategy.analyze(df_featured, mtf_trend)
                    if is_signal:
                        logger.info(f"Signal found for {symbol} by {strategy.name}. Reason: {reason}")
                        create_trade_signal(symbol, df_featured, strategy.name, quality_score)
                        break # Move to the next symbol after finding a signal
                    else:
                        # Optional: Log rejections for debugging, can be noisy
                        # logger.debug(f"[{symbol}] Rejected by {strategy.name}. Reason: {reason}")
                        pass
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

def close_trade(symbol: str, signal_id: int, closing_price: float, reason: str):
    with signal_cache_lock:
        if symbol not in open_signals_cache or open_signals_cache[symbol]['id'] != signal_id: return
        signal = open_signals_cache[symbol]

    entry_price = signal['entry_price']
    profit = ((closing_price - entry_price) / entry_price) * 100
    
    if signal.get('is_real_trade'):
        try:
            quantity = Decimal(str(signal.get('quantity', 0)))
            if quantity > 0:
                adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(quantity))
                if adjusted_quantity and adjusted_quantity > 0:
                    formatted_qty = get_formatted_quantity(symbol, adjusted_quantity)
                    client.create_order(symbol=symbol, side=Client.SIDE_SELL, type=Client.ORDER_TYPE_MARKET, quantity=formatted_qty)
        except Exception as e:
            logger.error(f"❌ [Real Close] CRITICAL ERROR for {symbol}: {e}", exc_info=True)

    update_signal_in_db(signal_id, {"status": "closed", "closing_price": closing_price, "closed_at": datetime.now(timezone.utc), "profit_percentage": profit, "closing_reason": reason})
    with signal_cache_lock:
        if symbol in open_signals_cache: del open_signals_cache[symbol]
    broadcast({"type": "trade_closed", "payload": {"signal_id": signal_id}})
    log_and_notify("info", f"Closed trade for {symbol}. Profit: {profit:.2f}%", "TRADE_CLOSED")

def process_open_trades():
    with signal_cache_lock:
        signals_to_process = list(open_signals_cache.values())
    for signal in signals_to_process:
        symbol = signal['symbol']
        with live_prices_lock:
            current_price = live_prices.get(symbol)
        if current_price:
            if current_price <= signal['stop_loss']:
                close_trade(symbol, signal['id'], signal['stop_loss'], "stop_loss")
            elif current_price >= signal['target_price_2']:
                close_trade(symbol, signal['id'], signal['target_price_2'], "target_2")

def process_open_trades_periodically():
    logger.info("Starting open trades processor...")
    while True:
        try:
            process_open_trades()
            time.sleep(5)
        except Exception as e:
            logger.error(f"❌ [Process Open Trades] Error: {e}", exc_info=True)
            time.sleep(60)

def update_balance_loop():
    logger.info("🚀 [Balance Updater] Starting balance update loop...")
    while True:
        try:
            balance_info = client.get_asset_balance(asset='USDT')
            with balance_lock:
                global usdt_balance
                usdt_balance = float(balance_info['free'])
        except Exception as e:
            logger.error(f"❌ [Balance Loop] Error: {e}", exc_info=True)
        time.sleep(60 * 5)

if __name__ == '__main__':
    logger.info("="*50 + "\n====== Starting Crypto Trading Bot V35 (Advanced Strategies) ======\n" + "="*50)
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
    
    Thread(target=update_balance_loop, daemon=True).start()
    time.sleep(2) # Give time for first balance fetch
    
    start_websocket()
    Thread(target=main_bot_loop, daemon=True).start()
    Thread(target=process_open_trades_periodically, daemon=True).start()
    
    logger.info("🌐 [Flask] Starting UI on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
