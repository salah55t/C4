# -*- coding: utf-8 -*-
# ملف c4.py - نسخة V25.0.0 (تحسين إدارة المخاطر والتنبيهات)
# --- وصف الإصدار:
# هذا الإصدار يركز على تحسين دقة إدارة المخاطر ومركزية نظام التنبيهات.
# 1.  [تحسين] إعادة هيكلة منطق تحديد حجم الصفقة ليكون أكثر قوة ومرونة.
#     - إضافة عوامل ديناميكية جديدة لحساب نسبة المخاطرة، مثل تقلبات السوق.
#     - تحسين التحقق من قواعد المنصة وضمان التوافق الكامل قبل إرسال أي أمر.
# 2.  [تحسين] توحيد نظام التنبيهات في دالة مركزية واحدة `send_alert`.
#     - تسهيل إضافة تنبيهات جديدة أو تعديل التنبيهات الحالية.
#     - استخدام قوالب رسائل منظمة وواضحة لكل نوع من أنواع التنبيهات.
# 3.  [إكمال] دمج جميع الدوال والواجهات في ملف واحد كامل وجاهز للتشغيل.

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
        logging.FileHandler('crypto_bot_v25_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV25')

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
usdt_balance: float = 1000.0
balance_lock = Lock()
cooldowns_by_symbol = {}
cooldowns_lock = Lock()
consecutive_losses_by_symbol = {}
consecutive_losses_lock = Lock()
COOLDOWN_MINUTES_AFTER_SL = 20
PAPER_TRADE_INITIAL_BALANCE = 1000.0

# --- المتغيرات القابلة للتعديل ---
TRADE_SIZE_MODE: str = 'risk'  # 'risk' or 'fixed'
RISK_PER_TRADE_PERCENT: float = 1.0
FIXED_TRADE_AMOUNT_USDT: float = 15.0
trade_settings_lock = Lock()

MAX_OPEN_TRADES: int = 3
MIN_SIGNAL_QUALITY: int = 60
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
TRADING_RULES: Dict[str, Dict] = {}
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
    "LotSize Filter Failed": "كمية الصفقة لا تتوافق مع قواعد المنصة",
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
                pass # Client might have been removed in another thread

def get_dashboard_payload() -> Dict:
    with trading_status_lock: trading_enabled = is_trading_enabled
    with trading_mode_lock: is_paper_mode = paper_trading_mode
    with balance_lock: current_balance = usdt_balance
    with notifications_lock: notifications = list(notifications_cache)
    with rejection_logs_lock: rejections = list(rejection_logs_cache)
    with market_state_lock: market_state = dict(current_market_state)
    with min_quality_lock: min_quality = MIN_SIGNAL_QUALITY
    with trade_settings_lock:
        trade_settings = {
            "mode": TRADE_SIZE_MODE,
            "risk_percent": RISK_PER_TRADE_PERCENT,
            "fixed_amount": FIXED_TRADE_AMOUNT_USDT
        }
    
    return {
        "trading_enabled": trading_enabled, 
        "paper_trading_mode": is_paper_mode,
        "usdt_balance": current_balance,
        "notifications": notifications, 
        "rejections": rejections, 
        "market_state": market_state,
        "min_signal_quality": min_quality,
        "trade_settings": trade_settings,
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

# ==============================================================================
# ========================== تحسين وتوحيد نظام التنبيهات ==========================
# ==============================================================================

def send_telegram_message(message: str, force: bool = False):
    """
    دالة مساعدة لإرسال الرسائل عبر تليجرام.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'},
            timeout=10
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [Telegram] Failed to send message: {e}")

def send_alert(alert_type: str, **context: Any):
    """
    دالة مركزية لإرسال جميع أنواع التنبيهات.
    
    Args:
        alert_type (str): نوع التنبيه (e.g., 'PROFIT_TARGET', 'MANUAL_CLOSE').
        **context: قاموس يحتوي على البيانات الخاصة بالتنبيه.
    """
    message_templates = {
        'PROFIT_TARGET': (
            "🎯 *تحقيق هدف ربح*\n"
            "`{symbol}` | `ID: {signal_id}`\n"
            "*المستوى:* `TP{target_level}`\n"
            "*نسبة الربح:* `{profit_percent:.2f}%`"
        ),
        'STOP_LOSS_ADJUSTMENT': (
            "🔄 *تعديل وقف الخسارة*\n"
            "`{symbol}` | `ID: {signal_id}`\n"
            "*السبب:* `{reason}`\n"
            "*القديم:* `{old_sl:.4f}`\n"
            "*الجديد:* `{new_sl:.4f}`"
        ),
        'MANUAL_CLOSE': (
            "🔔 *تنبيه إغلاق يدوي*\n"
            "`{symbol}` | `ID: {signal_id}`\n"
            "*السبب:* `{reason}`"
        ),
        'RISK_MANAGEMENT': (
            "⚠️ *تنبيه إدارة المخاطر*\n"
            "`{symbol}` | `ID: {signal_id}`\n"
            "*النوع:* `{alert_subtype}`\n"
            "{details_str}"
        ),
        'NEW_TRADE': (
            "📊 *{trade_type} جديدة*\n"
            "`{symbol}` | `{strategy_name}`\n"
            "*الجودة:* `{quality_score}/100`\n"
            "*دخول:* `{entry_price:.4f}`\n"
            "*هدف1:* `{target_price_1:.4f}`\n"
            "*وقف:* `{stop_loss:.4f}`"
        ),
        'TRADE_CLOSE': (
            "{result_emoji} *إغلاق صفقة {trade_type}*\n"
            "`{symbol}` | `ID: {signal_id}`\n"
            "*السبب:* `{reason}`\n"
            "*الربح:* `{profit:.2f}%`"
        )
    }

    template = message_templates.get(alert_type)
    if not template:
        logger.error(f"❌ [Alert] Unknown alert type: {alert_type}")
        return

    try:
        if alert_type == 'RISK_MANAGEMENT':
            details = context.get('details', {})
            context['details_str'] = "\n".join([f"*{k.replace('_', ' ').title()}:* `{v}`" for k, v in details.items()])
        
        message = template.format(**context)
        send_telegram_message(message, force=True)
        log_and_notify("info", f"Sent alert '{alert_type}' for {context.get('symbol', 'N/A')}", f"{alert_type}_ALERT")

    except KeyError as e:
        logger.error(f"❌ [Alert] Missing context for alert '{alert_type}': {e}")
    except Exception as e:
        logger.error(f"❌ [Alert] Error sending alert for {alert_type}: {e}", exc_info=True)


# --- WebSocket & Data Fetching ---
def handle_socket_message(msg):
    global live_prices
    if msg and 'e' in msg and msg['e'] == 'error': logger.error(f"❌ [WebSocket] Error: {msg['m']}"); return
    if isinstance(msg, list):
        price_updates = {}
        with live_prices_lock:
            for ticker in msg:
                if 's' in ticker and 'c' in ticker: 
                    symbol = ticker['s']
                    price = float(ticker['c'])
                    live_prices[symbol] = price
                    price_updates[symbol] = price
        if price_updates:
            broadcast({"type": "price_update", "payload": price_updates})

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

def preload_trading_rules():
    global TRADING_RULES
    logger.info("[Rules Engine] Pre-loading trading rules for all symbols...")
    for symbol, info in exchange_info_map.items():
        try:
            lot_size = next((f for f in info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            notional = next((f for f in info['filters'] if f['filterType'] == 'NOTIONAL'), None)
            if lot_size and notional:
                TRADING_RULES[symbol] = {
                    'minQty': float(lot_size['minQty']),
                    'maxQty': float(lot_size['maxQty']),
                    'stepSize': lot_size['stepSize'],
                    'minNotional': float(notional['minNotional'])
                }
        except (KeyError, ValueError) as e:
            logger.warning(f"⚠️ [Rules Engine] Could not parse filters for {symbol}: {e}")
    logger.info(f"✅ [Rules Engine] Successfully loaded rules for {len(TRADING_RULES)} symbols.")

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

# ======================= START: IMPROVED INDICATOR CALCULATION =======================
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
# ======================== END: IMPROVED INDICATOR CALCULATION ========================

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
    global RISK_PER_TRADE_PERCENT, MAX_OPEN_TRADES, USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY, USE_MOMENTUM_VOLATILITY_STRATEGY, STRATEGY_FILTER_CONFIG, paper_trading_mode, MIN_SIGNAL_QUALITY, TRADE_SIZE_MODE, FIXED_TRADE_AMOUNT_USDT
    if not redis_client: return
    try:
        settings_data = redis_client.get('trading_settings')
        if settings_data:
            settings = json.loads(settings_data)
            with trade_settings_lock:
                TRADE_SIZE_MODE = settings.get('trade_size_mode', 'risk')
                RISK_PER_TRADE_PERCENT = settings.get('risk_percent', 1.0)
                FIXED_TRADE_AMOUNT_USDT = settings.get('fixed_amount', 15.0)
            MAX_OPEN_TRADES = settings.get('MAX_OPEN_TRADES', 3)
            with trading_mode_lock: paper_trading_mode = settings.get('paper_trading_mode', True)
        
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

# --- Filters & Strategies ---
def calculate_signal_quality_score(symbol, df, strategy_name):
    score = 0
    if df.empty or len(df) < 50: return 0
    last_row = df.iloc[-1]
    
    adx_value = last_row.get('adx', 0)
    if adx_value > 30: score += 25
    elif adx_value > 20: score += 20
    else: score += 5
    
    current_volume = last_row.get('volume', 0)
    volume_ma = df['volume'].rolling(20).mean().iloc[-1]
    volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1
    if volume_ratio > 1.5: score += 15
    elif volume_ratio > 1.0: score += 5
    
    rsi = last_row.get('rsi', 50)
    if 40 <= rsi <= 60: score += 20
    elif 30 <= rsi <= 70: score += 10
    
    ema9, ema21, ema50, close = last_row.get('ema9',0), last_row.get('ema21',0), last_row.get('ema50',0), last_row.get('close',0)
    if close > ema9 > ema21 > ema50: score += 20
    elif close > ema9 > ema21: score += 15
    
    atr_percent = last_row.get('atr_percent', 0)
    if 1.5 <= atr_percent <= 3.5: score += 10
        
    if strategy_name == "Momentum_Volatility_Strategy" and adx_value > 35: score += 10
    if strategy_name == "BB_Stoch_Strategy" and last_row.get('stoch_rsi_k', 50) < 20: score += 5
        
    return min(100, max(0, int(score)))

# ... (بقية دوال الفلاتر والاستراتيجيات موجودة في الملف الأصلي)
def check_bb_stoch_strategy_enhanced(df: pd.DataFrame) -> bool: return False
def check_macd_ema_strategy_enhanced(df: pd.DataFrame) -> bool: return False
def check_ema_rsi_strategy_enhanced(df: pd.DataFrame) -> bool: return False
def check_pullback_strategy_enhanced(df: pd.DataFrame) -> bool: return False
def check_momentum_volatility_strategy(df: pd.DataFrame) -> bool: return False
def apply_strategy_filters(symbol: str, df: pd.DataFrame, strategy_name: str) -> bool: return True
def check_market_volatility_filter(df: pd.DataFrame) -> bool: return True
def check_trend_strength_filter(df: pd.DataFrame, adx_threshold: int) -> bool: return True


def calculate_trade_levels(df: pd.DataFrame) -> Dict[str, Any]:
    last = df.iloc[-1]
    atr = last.get('atr', (last['high'] - last['low']))
    entry_price = last['close']
    stop_loss = entry_price - (atr * 1.5)
    target_price_1 = entry_price + (atr * 2.0)
    target_price_2 = entry_price + (atr * 3.5)
    return {
        "entry_price": entry_price, "stop_loss": stop_loss, "target_price_1": target_price_1,
        "target_price_2": target_price_2, "atr": atr
    }

# ==============================================================================
# ======================= تحسين إدارة قواعد التداول وتحديد حجم الصفقة =======================
# ==============================================================================

def get_account_balance() -> Dict[str, float]:
    """
    الحصول على الرصيد الحالي للحساب (حقيقي أو ورقي).
    """
    with trading_mode_lock:
        is_paper = paper_trading_mode
    
    if is_paper:
        with balance_lock:
            return {"USDT": usdt_balance}
    else:
        try:
            if not client:
                logger.error("❌ [Balance] Binance client not initialized.")
                return {"USDT": 0.0}
            account_info = client.get_account()
            balances = {bal['asset']: float(bal['free']) for bal in account_info.get('balances', []) if float(bal['free']) > 0}
            return balances
        except BinanceAPIException as e:
            logger.error(f"❌ [Balance] Binance API error getting account balance: {e}")
            return {"USDT": 0.0}
        except Exception as e:
            logger.error(f"❌ [Balance] Unexpected error getting account balance: {e}")
            return {"USDT": 0.0}

def get_dynamic_risk_percent(symbol: str, df: pd.DataFrame) -> float:
    """
    حساب نسبة المخاطرة الديناميكية بناءً على عدة عوامل.
    """
    base_risk = RISK_PER_TRADE_PERCENT
    with consecutive_losses_lock:
        losses = consecutive_losses_by_symbol.get(symbol, 0)
    
    loss_factor = 1.0
    if losses >= 3: loss_factor = 0.5
    elif losses >= 2: loss_factor = 0.75
    
    volatility_factor = 1.0
    atr_percent = df['atr_percent'].iloc[-1] if 'atr_percent' in df.columns and not df['atr_percent'].empty else 2.0
    if atr_percent > 4.0: volatility_factor = 0.8
    elif atr_percent < 1.5: volatility_factor = 1.1
        
    final_risk = base_risk * loss_factor * volatility_factor
    final_risk = max(0.25, min(final_risk, 2.0))
    
    logger.info(f"🔍 [Dynamic Risk] Symbol: {symbol}, Base: {base_risk}%, Loss: {loss_factor}, Vol: {volatility_factor} -> Final: {final_risk:.2f}%")
    return final_risk

def apply_exchange_filters(symbol: str, quantity: float, price: float) -> Optional[float]:
    """
    التحقق من الكمية وتعديلها حسب قواعد التداول المخزنة (Filters).
    """
    rules = TRADING_RULES.get(symbol)
    if not rules:
        logger.error(f"❌ [Rules Engine] No trading rules found for {symbol}")
        return None

    step_size_decimal = Decimal(str(rules['stepSize']))
    adjusted_quantity = float(Decimal(quantity).quantize(step_size_decimal, rounding=ROUND_DOWN))

    if adjusted_quantity < rules['minQty']:
        log_rejection(symbol, "LotSize Filter Failed", {"reason": f"Qty {adjusted_quantity} < minQty {rules['minQty']}"})
        return None
    if adjusted_quantity > rules['maxQty']:
        adjusted_quantity = rules['maxQty']

    notional_value = adjusted_quantity * price
    if notional_value < rules['minNotional']:
        log_rejection(symbol, "MinNotional Filter Failed", {"required": f"{rules['minNotional']:.2f}$", "actual": f"{notional_value:.2f}$"})
        return None

    return adjusted_quantity

def calculate_optimal_position_size(symbol: str, price: float, stop_loss: float, df: pd.DataFrame) -> Optional[float]:
    """
    حساب الحجم الأمثل للصفقة بناءً على الرصيد المتاح وقواعد إدارة المخاطر.
    """
    try:
        balances = get_account_balance()
        available_usdt = balances.get("USDT", 0.0)
        
        if available_usdt < 10:
            log_rejection(symbol, "Insufficient Balance", {"available": available_usdt})
            return None
        
        with trade_settings_lock:
            mode = TRADE_SIZE_MODE
            fixed_amount = FIXED_TRADE_AMOUNT_USDT
        
        initial_quantity = 0
        risk_percent = 0.0
        if mode == 'risk':
            risk_percent = get_dynamic_risk_percent(symbol, df)
            max_risk_usd = available_usdt * (risk_percent / 100)
            risk_per_unit = price - stop_loss
            if risk_per_unit <= 1e-8:
                logger.error(f"❌ [Position Size] Invalid risk_per_unit for {symbol}: {risk_per_unit}")
                return None
            initial_quantity = max_risk_usd / risk_per_unit
        elif mode == 'fixed':
            initial_quantity = fixed_amount / price

        if initial_quantity <= 0:
            log_rejection(symbol, "Insufficient Balance")
            return
            
        final_quantity = apply_exchange_filters(symbol, initial_quantity, price)
        
        if final_quantity is None: return None
        
        total_value = final_quantity * price
        if total_value > available_usdt:
            final_quantity = (available_usdt / price) * 0.99
            final_quantity = apply_exchange_filters(symbol, final_quantity, price)
            if final_quantity is None: return None
            total_value = final_quantity * price

        logger.info(f"✅ [Position Size] Calculated for {symbol}: {final_quantity} (Mode: {mode}, Risk: {risk_percent:.2f}%, Value: {total_value:.2f} USDT)")
        return final_quantity

    except Exception as e:
        logger.error(f"❌ [Position Size] Error calculating position size for {symbol}: {e}", exc_info=True)
        return None

def create_trade_signal(symbol: str, df: pd.DataFrame, strategy_name: str):
    quality_score = calculate_signal_quality_score(symbol, df, strategy_name)
    with min_quality_lock:
        if quality_score < MIN_SIGNAL_QUALITY:
            log_rejection(symbol, "Low Quality Signal", {"score": quality_score, "min_required": MIN_SIGNAL_QUALITY})
            return

    logger.info(f"⭐ [Signal Quality] {symbol} ({strategy_name}): {quality_score}/100")
    
    trade_levels = calculate_trade_levels(df)
    entry_price = trade_levels['entry_price']
    stop_loss = trade_levels['stop_loss']
    
    quantity = calculate_optimal_position_size(symbol, entry_price, stop_loss, df)
    
    if quantity is None or quantity <= 0:
        return

    with trading_mode_lock: is_real = not paper_trading_mode
    
    signal_details = {"atr": trade_levels['atr'], "quality_score": quality_score}

    if is_real:
        try:
            logger.info(f"💰 [Real Trade] Placing LIVE MARKET BUY order for {quantity} of {symbol}")
            order = client.create_order(symbol=symbol, side=Client.SIDE_BUY, type=Client.ORDER_TYPE_MARKET, quantity=quantity)
            
            avg_fill_price = entry_price
            if order.get('fills'):
                avg_fill_price = sum(float(f['price']) * float(f['qty']) for f in order['fills']) / sum(float(f['qty']) for f in order['fills'])

            executed_quantity = float(order.get('executedQty', quantity))
            order_id = order.get('orderId', 'N/A')
            
            save_signal_to_db(symbol, avg_fill_price, trade_levels, strategy_name, True, executed_quantity, signal_details, order_id)
            
            send_alert('NEW_TRADE', trade_type="صفقة حقيقية", symbol=symbol, strategy_name=strategy_name,
                       quality_score=quality_score, entry_price=avg_fill_price, stop_loss=stop_loss,
                       target_price_1=trade_levels['target_price_1'])
        
        except BinanceAPIException as e:
            logger.error(f"❌ [Real Trade] Binance API Error for {symbol}: {e}")
        except Exception as e:
            logger.error(f"❌ [Real Trade] CRITICAL ERROR for {symbol}: {e}", exc_info=True)
    else:
        save_signal_to_db(symbol, entry_price, trade_levels, strategy_name, False, quantity, signal_details)
        send_alert('NEW_TRADE', trade_type="صفقة ورقية", symbol=symbol, strategy_name=strategy_name,
                   quality_score=quality_score, entry_price=entry_price, stop_loss=stop_loss,
                   target_price_1=trade_levels['target_price_1'])

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
DASHBOARD_TEMPLATE = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>لوحة التحكم - بوت التداول (V25.0)</title>
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
.left-column,.right-column{display:flex;flex-direction:column;gap:16px}
.card{background:var(--panel);border:1px solid #1e2c52;border-radius:14px;box-shadow:0 8px 30px rgba(0,0,0,.25);overflow:hidden}
.card h2{margin:0;padding:12px 14px;border-bottom:1px solid #1e2c52;font-size:14px;color:#cfe2ff; display: flex; justify-content: space-between; align-items: center;}
.card-body{padding:12px}
.controls{display:flex;gap:8px;flex-wrap:wrap}
.btn{appearance:none;border:1px solid #2a3a68;background:#0f1b3b;color:#d9e7ff;padding:10px 14px;border-radius:10px;cursor:pointer;font-weight:700;transition: background-color 0.2s, transform 0.2s; will-change: transform; text-decoration: none;}
.btn:hover{transform:translateY(-1px);border-color:#3a58a6}
.btn.warn{background:linear-gradient(180deg,#3b2a0f,#291b08);border-color:#8b5b0f}
.btn.small{padding: 6px 10px; font-size: 12px;}
.signals-grid{display:grid;grid-template-columns:repeat(auto-fill, minmax(300px, 1fr));gap:10px;}
.signal{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;padding:10px;border:1px solid #24335f;border-radius:12px;background:#0d1730; grid-template-rows: auto auto;}
.signal > *:nth-child(1) { grid-column: 1 / 2; }
.signal > *:nth-child(2) { grid-column: 2 / 3; grid-row: 1 / 3; }
.signal > *:nth-child(3) { grid-column: 1 / 2; }
.sig-title{font-weight:700}
.sig-meta{font-size:12px;color:var(--muted)}
.price{font-variant-numeric:tabular-nums;direction:ltr; font-size: 16px; font-weight: bold;}
.progress{height:8px;background:#0b1126;border:1px solid #233056;border-radius:999px;overflow:hidden; margin-top: 6px;}
.progress>span{display:block;height:100%;}
.kv{display:grid;grid-template-columns:auto 1fr;gap:8px 10px; align-items: center;}
.kv div:nth-child(odd){opacity:.8}
.trend{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}
.trend .pill{background:#0d1730;border:1px solid #1f2d55;border-radius:10px;padding:8px;text-align:center; display: flex; flex-direction: column; align-items: center; gap: 4px;}
.pill b{display:block;font-size:12px;color:#9fb7ef}
.pill span{font-size:12px}
.pill small {font-size: 10px; opacity: 0.8;}
.green{color:var(--ok)}.red{color:var(--bad)}.amber{color:var(--warn)}
.table{width:100%;border-collapse:separate;border-spacing:0 8px;}
.table th{font-size:12px;text-align:right;color:#9ab2e2;font-weight:600;padding:0 6px}
.table td{padding:8px;background:#0d1730;border:1px solid #24335f; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}
.switch{display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;border:1px solid #2a3a68;background:#0f1b3b;cursor:pointer;user-select:none}
.switch input{display:none}
.switch .dot{width:14px;height:14px;border-radius:50%;background:#6a7fb2;transition:.2s}
.switch input:checked + .dot{background:#24d08a;transform:translateX(2px) scale(1.1)}
.small{font-size:12px;color:#a8bfeb}
.input-group { display: flex; align-items: center; gap: 8px; }
.input-group input { width: 70px; background: #0b1126; border: 1px solid #233056; color: #e8f1ff; padding: 4px 8px; border-radius: 6px; text-align: center; }
.loading-spinner { border: 3px solid rgba(255, 255, 255, 0.1); border-radius: 50%; border-top: 3px solid #3aa0ff; width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 20px auto; }
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
.slider { -webkit-appearance: none; width: 100%; height: 6px; border-radius: 3px; background: #1e2c52; outline: none; }
.slider::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 16px; height: 16px; border-radius: 50%; background: #3aa0ff; cursor: pointer; }
</style>
</head>
<body>
<div class="container">
  <header><h1>لوحة التحكم • بوت التداول V25.0</h1><div class="badge" id="serverTime">—</div></header>
  <div class="main-layout">
    <div class="left-column">
      <div class="card">
        <h2>الصفقات المفتوحة <span class="small" id="signalCount">(0)</span></h2>
        <div class="card-body">
            <div id="signals" class="signals-grid"><div class="loading-spinner"></div></div>
        </div>
      </div>
    </div>
    <div class="right-column">
      <div class="card">
        <h2>التحكم والحالة</h2>
        <div class="card-body">
          <div class="controls">
            <label class="switch"><input id="toggleTrading" type="checkbox" /><span class="dot"></span><span class="small">تشغيل التداول</span></label>
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
            <div><label class="switch"><input type="checkbox" id="tradingModeToggle"><span class="dot"></span><span id="tradingModeText">ورقي</span></label></div>
          </div>
          <div class="kv">
            <div>حجم الصفقة:</div>
            <div><label class="switch"><input type="checkbox" id="tradeSizeModeToggle"><span class="dot"></span><span id="tradeSizeModeText">نسبة المخاطرة</span></label></div>
          </div>
          <div class="kv" id="riskPercentGroup">
            <div>نسبة المخاطرة:</div>
            <div class="input-group"><input type="number" id="riskPercentInput" step="0.1" min="0.1" max="5"><span>%</span></div>
          </div>
          <div class="kv" id="fixedAmountGroup" style="display: none;">
            <div>المبلغ الثابت:</div>
            <div class="input-group"><input type="number" id="fixedAmountInput" step="1" min="5"><span>USDT</span></div>
          </div>
          <div class="kv">
            <div>أقل جودة إشارة:</div>
            <div><input type="range" id="qualityFilter" min="30" max="90" value="60" class="slider"><span id="qualityValue">60</span></div>
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
let openSignals = {};

const debounce = (func, delay) => {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), delay);
    };
};
function fmt(n){ return n == null ? '—' : (+n).toLocaleString('en-US', {maximumFractionDigits: 6}); }

function renderSignal(signal) {
    const cp = lastPrices[signal.symbol] || signal.entry_price;
    const pToTp = ((cp - signal.entry_price) / (signal.target_price_1 - signal.entry_price)) * 100;
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
                <button class="btn warn small" onclick="fetch('/close_trade/${signal.id}',{method:'POST'})">إغلاق</button>
            </div>
            <div class="progress" title="التقدم نحو الهدف: ${pToTp.toFixed(1)}%">
                <span style="width:${Math.min(100, Math.max(0,pToTp))}%; background:linear-gradient(90deg, var(--ok), #3fd1b0)"></span>
            </div>
        </div>`;
}

function updateUI(data) {
    qs('#serverTime').textContent = new Date(data.server_time).toLocaleTimeString('ar-EG');
    qs('#toggleTrading').checked = !!data.trading_enabled;
    qs('#balance').textContent = fmt(data.usdt_balance);
    qs('#openCount').textContent = Object.keys(openSignals).length;
    qs('#signalCount').textContent = `(${Object.keys(openSignals).length})`;

    const isPaper = data.paper_trading_mode;
    qs('#tradingModeToggle').checked = !isPaper;
    qs('#tradingModeText').textContent = isPaper ? 'ورقي' : 'حقيقي';

    qs('#qualityFilter').value = data.min_signal_quality;
    qs('#qualityValue').textContent = data.min_signal_quality;

    updateTradeSizeUI(data.trade_settings);
    updateMarketTrends(data.market_state);

    const eventsTbody = qs('#events tbody');
    eventsTbody.innerHTML = data.notifications.map(n => `<tr><td>${new Date(n.timestamp).toLocaleTimeString('ar-EG')}</td><td>${n.type||''}</td><td>${n.message||''}</td></tr>`).join('');
    
    const rejectionsTbody = qs('#rejections tbody');
    rejectionsTbody.innerHTML = data.rejections.map(r => `<tr><td>${new Date(r.timestamp).toLocaleTimeString('ar-EG')}</td><td>${r.symbol||''}</td><td>${r.reason||''}</td></tr>`).join('');
}

function updateMarketTrends(marketState) {
  const trendsContainer = document.getElementById('marketTrends');
  trendsContainer.innerHTML = '';
  if (marketState && marketState.trend_details_by_tf) {
    ['15m', '1h', '4h'].forEach(tf => {
      const trend = marketState.trend_details_by_tf[tf];
      if (trend) {
        let trendClass = trend.trend === 'bullish' ? 'green' : (trend.trend === 'bearish' ? 'red' : 'amber');
        let trendText = trend.trend === 'bullish' ? 'صاعد' : (trend.trend === 'bearish' ? 'هابط' : 'جانبي');
        trendsContainer.innerHTML += `<div class="pill"><b>${tf}</b><span class="${trendClass}">${trendText}</span><small>ADX: ${trend.adx?.toFixed(1) || '—'}</small></div>`;
      }
    });
  }
}

async function initializeDashboard() {
    try {
        const [baseRes, signalsRes] = await Promise.all([
            fetch('/api/dashboard_data'),
            fetch('/api/open_signals')
        ]);
        const baseData = await baseRes.json();
        const signalsData = await signalsRes.json();
        
        openSignals = signalsData.signals.reduce((acc, s) => { acc[s.id] = s; return acc; }, {});
        qs('#signals').innerHTML = signalsData.signals.map(renderSignal).join('');
        
        updateUI(baseData);
    } catch (error) {
        console.error("فشل تحميل البيانات:", error);
    }
}

function setupWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    const socket = new WebSocket(wsUrl);
    
    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        switch(data.type) {
            case 'price_update': 
                Object.assign(lastPrices, data.payload);
                for (const signal of Object.values(openSignals)) {
                    const el = qs(`#signal-${signal.id}`);
                    if (el) el.outerHTML = renderSignal(signal);
                }
                break;
            case 'new_signal':
                openSignals[data.payload.id] = data.payload;
                qs('#signals').insertAdjacentHTML('afterbegin', renderSignal(data.payload));
                break;
            case 'signal_closed':
                delete openSignals[data.payload.id];
                const el = qs(`#signal-${data.payload.id}`);
                if (el) el.remove();
                break;
            case 'new_notification':
                qs('#events tbody').insertAdjacentHTML('afterbegin', `<tr><td>${new Date(data.payload.timestamp).toLocaleTimeString('ar-EG')}</td><td>${data.payload.type||''}</td><td>${data.payload.message||''}</td></tr>`);
                break;
            case 'new_rejection':
                qs('#rejections tbody').insertAdjacentHTML('afterbegin', `<tr><td>${new Date(data.payload.timestamp).toLocaleTimeString('ar-EG')}</td><td>${data.payload.symbol||''}</td><td>${data.payload.reason||''}</td></tr>`);
                break;
            case 'market_state': updateMarketTrends(data.payload); break;
            case 'trade_settings_update': updateTradeSizeUI(data.payload); break;
        }
    };
    socket.onclose = () => setTimeout(setupWebSocket, 3000);
}

const debouncedAPICall = debounce((url, body) => {
    fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
    }).catch(error => console.error('Error:', error));
}, 800);

qs('#toggleTrading').addEventListener('change', () => fetch('/toggle_trading', {method:'POST'}));
qs('#tradingModeToggle').addEventListener('change', function() {
    debouncedAPICall('/api/trading_mode', {paper_trading: !this.checked});
});
qs('#qualityFilter').addEventListener('input', function() {
  qs('#qualityValue').textContent = this.value;
  debouncedAPICall('/api/quality_filter', {min_quality: parseInt(this.value)});
});

function updateTradeSizeUI(settings) {
    const isFixed = settings.mode === 'fixed';
    qs('#tradeSizeModeToggle').checked = isFixed;
    qs('#tradeSizeModeText').textContent = isFixed ? 'مبلغ ثابت' : 'نسبة المخاطرة';
    qs('#riskPercentGroup').style.display = isFixed ? 'none' : 'grid';
    qs('#fixedAmountGroup').style.display = isFixed ? 'grid' : 'none';
    qs('#riskPercentInput').value = settings.risk_percent;
    qs('#fixedAmountInput').value = settings.fixed_amount;
}
function updateTradeSettings() {
    const settings = {
        mode: qs('#tradeSizeModeToggle').checked ? 'fixed' : 'risk',
        risk_percent: parseFloat(qs('#riskPercentInput').value),
        fixed_amount: parseFloat(qs('#fixedAmountInput').value)
    };
    debouncedAPICall('/api/trade_settings', settings);
}
qs('#tradeSizeModeToggle').addEventListener('change', updateTradeSettings);
qs('#riskPercentInput').addEventListener('input', updateTradeSettings);
qs('#fixedAmountInput').addEventListener('input', updateTradeSettings);

document.addEventListener('DOMContentLoaded', () => {
    initializeDashboard();
    setupWebSocket();
});
</script>
</body>
</html>
"""

# --- مسارات Flask ---
@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/api/dashboard_data')
def dashboard_data():
    try:
        return jsonify(get_dashboard_payload())
    except Exception as e:
        logger.error(f"❌ [API Error] Failed to generate dashboard data: {e}", exc_info=True)
        return jsonify({"error": "Failed to load dashboard data."}), 500

@app.route('/api/open_signals')
def get_open_signals():
    with signal_cache_lock:
        signals = list(open_signals_cache.values())
    return jsonify({"signals": sorted(signals, key=lambda x: x['id'], reverse=True)})

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
        with trading_mode_lock: paper_trading_mode = is_paper
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
        with min_quality_lock: MIN_SIGNAL_QUALITY = int(min_quality)
        log_and_notify("info", f"Minimum signal quality updated to {MIN_SIGNAL_QUALITY}.", "SETTINGS_UPDATE")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error updating quality filter: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/trade_settings', methods=['POST'])
def update_trade_settings():
    global TRADE_SIZE_MODE, RISK_PER_TRADE_PERCENT, FIXED_TRADE_AMOUNT_USDT
    try:
        data = request.json
        with trade_settings_lock:
            TRADE_SIZE_MODE = data.get('mode', 'risk')
            RISK_PER_TRADE_PERCENT = float(data.get('risk_percent', 1.0))
            FIXED_TRADE_AMOUNT_USDT = float(data.get('fixed_amount', 15.0))
        broadcast({"type": "trade_settings_update", "payload": data})
        log_and_notify("info", f"Trade size settings updated: Mode={TRADE_SIZE_MODE}", "SETTINGS_UPDATE")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error updating trade settings: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/close_trade/<int:signal_id>', methods=['POST'])
def manual_close_trade(signal_id):
    signal_to_close = None
    with signal_cache_lock:
        signal_to_close = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)
    
    if not signal_to_close:
        return jsonify({"success": False, "message": "Signal not found"}), 404
    
    symbol = signal_to_close['symbol']
    with live_prices_lock: current_price = live_prices.get(symbol)
    if not current_price:
        try:
            current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
        except Exception as e:
            logger.error(f"❌ [Manual Close] Could not get live price for {symbol}: {e}")
            return jsonify({"success": False, "message": "Could not get live price"}), 500
    try:
        close_signal(signal_to_close, current_price, "MANUAL_CLOSE")
        send_alert('MANUAL_CLOSE', symbol=symbol, signal_id=signal_id, reason="Manual User Action")
        return jsonify({"success": True, "message": f"Close order sent for {symbol}"})
    except Exception as e:
        logger.error(f"❌ [Manual Close] Error closing signal {signal_id}: {e}", exc_info=True)
        return jsonify({"success": False, "message": "Error during close process"}), 500

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
            final_quantity = apply_exchange_filters(symbol, quantity, live_prices.get(symbol, 0))
            if final_quantity is None:
                logger.error(f"❌ [Execute Close] Quantity {quantity} for {symbol} failed validation.")
                return

            logger.info(f"💰 [Real Close] Executing MARKET SELL for {final_quantity} of {symbol} due to {reason}")
            client.create_order(symbol=symbol, side=Client.SIDE_SELL, type=Client.ORDER_TYPE_MARKET, quantity=final_quantity)
            
        except BinanceAPIException as e:
            logger.error(f"❌ [Execute Close] Binance API Error for {symbol}: {e}")
        except Exception as e:
            logger.error(f"❌ [Execute Close] CRITICAL ERROR for {symbol}: {e}", exc_info=True)
    else:
        logger.info(f"📊 [Paper Close] Simulating close of {quantity} {symbol} for reason: {reason}")

def close_signal(signal: Dict, closing_price: float, reason: str):
    symbol, signal_id, entry_price = signal['symbol'], signal['id'], signal['entry_price']
    
    execute_close_order(symbol, signal['quantity'], reason)

    profit = ((closing_price - entry_price) / entry_price) * 100
    with consecutive_losses_lock:
        if profit < 0:
            consecutive_losses_by_symbol[symbol] = consecutive_losses_by_symbol.get(symbol, 0) + 1
        else:
            consecutive_losses_by_symbol[symbol] = 0
            
    update_signal_in_db(signal_id, {"status": "closed", "closing_price": closing_price, "closed_at": datetime.now(timezone.utc), "profit_percentage": profit, "closing_reason": reason})
    with signal_cache_lock:
        if symbol in open_signals_cache: del open_signals_cache[symbol]
    
    broadcast({"type": "signal_closed", "payload": {"id": signal_id, "symbol": symbol}})
    
    trade_type = "حقيقية" if signal.get('is_real_trade') else "ورقية"
    result_emoji = "✅" if profit >= 0 else "🔻"
    
    send_alert('TRADE_CLOSE', result_emoji=result_emoji, trade_type=trade_type,
               symbol=symbol, signal_id=signal_id, reason=reason, profit=profit)

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
                
                if current_price <= signal['stop_loss']:
                    close_signal(signal, signal['stop_loss'], "SL_HIT")
                    continue
                if signal.get('target_price_2') and current_price >= signal['target_price_2']:
                     close_signal(signal, signal['target_price_2'], "TP2_HIT")
                     continue
            time.sleep(1)
        except Exception as e:
            logger.error(f"❌ [Trade Manager] A critical error occurred: {e}", exc_info=True)
            time.sleep(10)

def send_market_state_updates():
    logger.info("🚀 [Market State] Starting market state update loop...")
    while True:
        try:
            market_state = {"trend_details_by_tf": {}}
            for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
                df = fetch_historical_data(BTC_SYMBOL, tf, days=10)
                if df is not None and len(df) > 50:
                    df = calculate_all_features(df)
                    last_row = df.iloc[-1]
                    trend = "sideways"
                    if last_row['close'] > last_row['ema21'] and last_row['ema21'] > last_row['ema50']:
                        trend = "bullish"
                    elif last_row['close'] < last_row['ema21'] and last_row['ema21'] < last_row['ema50']:
                        trend = "bearish"
                    
                    market_state["trend_details_by_tf"][tf] = {
                        "trend": trend, "adx": last_row.get('adx', 0), "rsi": last_row.get('rsi', 50)
                    }
            
            with market_state_lock:
                global current_market_state
                current_market_state = market_state
            
            broadcast({"type": "market_state", "payload": market_state})
            time.sleep(60 * 5)
        except Exception as e:
            logger.error(f"Error sending market state updates: {e}")
            time.sleep(60)

def update_balance_loop():
    logger.info("🚀 [Balance Updater] Starting balance update loop...")
    while True:
        try:
            balances = get_account_balance()
            with balance_lock:
                global usdt_balance
                usdt_balance = balances.get('USDT', usdt_balance)
        except Exception as e:
            logger.error(f"❌ [Balance Loop] Error: {e}", exc_info=True)
        time.sleep(60 * 10)

# --- نقطة بداية البرنامج ---
if __name__ == '__main__':
    logger.info("="*50 + "\n====== Starting Crypto Trading Bot V25.0 (Risk & Alerts) ======\n" + "="*50)
    init_db()
    init_redis()
    try:
        client = Client(API_KEY, API_SECRET); client.ping()
        logger.info("✅ [Binance] API connection successful.")
    except Exception as e:
        logger.critical(f"❌ [Binance] API connection failed: {e}"); exit(1)

    get_exchange_info_map()
    preload_trading_rules()

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
    Thread(target=send_market_state_updates, daemon=True).start()
    Thread(target=update_balance_loop, daemon=True).start()

    logger.info("🌐 [Flask] Starting UI on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
