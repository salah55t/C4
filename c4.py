# ملف c4_5min_v34_0_5_enhanced.py - نسخة V34.0.5 محسنة بالكامل
# --- وصف التحسينات:
# 1. تحسين شامل لجميع استراتيجيات التداول مع فلتر حجم التداول وتأكيد الزخم
# 2. تحسين نظام إدارة المخاطر مع حساب ديناميكي لوقف الخسارة وجني الأرباح
# 3. تحسين الفلاتر الديناميكية لكل استراتيجية مع التكيف مع حالة السوق
# 4. تحسين نظام الإشعارات مع معالجة أفضل للأخطاء وتقارير مفصلة
# 5. تحسين نظام إدارة الصفقات مع وقف خسارة متحرك وإدارة أفضل للرصيد
# 6. تحسين نظام المسح والبحث عن الإشارات مع معالجة متوازية
# 7. تحسين نظام واجهة الويب مع نقاط نهاية API محسنة
# 8. تحسين نظام بدء التشغيل مع إدارة أفضل للاتصالات

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
from typing import List, Dict, Optional, Any
from collections import deque
import warnings
from scipy.signal import argrelextrema
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- إعدادات التجاهل واللوجر ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ضبط دقة النوع Decimal
getcontext().prec = 18

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v34_enhanced_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV34.0.5_Enhanced')

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

# --- إعدادات الفلاتر الديناميكية للاستراتيجيات ---
STRATEGY_NAMES = {
    "BB_Stoch_Strategy": "BB+Stoch (ارتداد مبكر)",
    "MACD_EMA_Strategy": "MACD+SMA (زخم وتقاطع)",
    "EMA_RSI_Strategy": "EMA+RSI (ارتداد سريع)",
    "Pullback_Strategy": "Pullback (ارتداد بحجم تداول)",
    "Momentum_Volatility_Strategy": "Momentum (زخم متزايد)",
    "Elliott_Wave_Strategy": "Elliott Wave (موجات إليوت)",
    "Range_Reversal_Strategy": "Range Reversal (انعكاس نطاقي)"
}
strategy_filters_lock = Lock()

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

    # Dynamic Filters Rejections
    "DYN_BB_WIDTH_LOW": "ديناميكي: عرض البولينجر ضيق جدًا",
    "DYN_STOCH_LOW": "ديناميكي: ستوكاستيك منخفض جدًا للسوق المتقلب",
    "DYN_VOLUME_LOW": "ديناميكي: حجم التداول منخفض بالنسبة للتقلبات",
    "DYN_ADX_LOW": "ديناميكي: قوة الاتجاه (ADX) ضعيفة للسوق الحالي",
    "DYN_MACD_MOMENTUM_LOW": "ديناميكي: زخم الماكد لا يتزايد بقوة كافية",
    "DYN_RSI_OOR": "ديناميكي: مؤشر القوة النسبية خارج النطاق المطلوب للاتجاه الحالي",
    "DYN_EMA_SPREAD_LOW": "ديناميكي: تباعد المتوسطات المتحركة ضعيف",
    "DYN_PULLBACK_SHALLOW": "ديناميكي: الارتداد ضحل جدًا للسوق المتقلب",
    "DYN_RECOVERY_FAIL": "ديناميكي: فشل السعر في التعافي بعد الارتداد",
    "DYN_VOLATILITY_OOR": "ديناميكي: التقلب خارج النطاق الأمثل للزخم",
    "DYN_MOMENTUM_SCORE_LOW": "ديناميكي: درجة الزخم الإجمالية منخفضة",
    "DYN_FIB_RETRACEMENT_OOR": "ديناميكي: تصحيح فيبوناتشي خارج النطاق المقبول للتقلب الحالي",

    # Strategy Specific Rejections
    "EMA_RSI: Bearish long-term trend": "EMA_RSI: اتجاه هابط طويل الأجل",
    "BB: Price below EMA50 (bearish trend)": "BB: السعر تحت EMA50 (اتجاه هابط)",
    "MACD: Strongly bearish trend": "MACD: الاتجاه هابط بقوة",
    "Pullback: Trend is not strongly bullish": "Pullback: الاتجاه ليس صاعدًا بقوة",
    "Momentum: EMAs not in bullish order": "Momentum: المتوسطات ليست في ترتيب صاعد",
    "Elliott Wave: Strongly bearish trend": "موجات إليوت: الاتجاه هابط بقوة",
    "Elliott Wave: Insufficient swing points": "موجات إليوت: نقاط تذبذب غير كافية",
    "Elliott Wave: Error in pattern detection": "موجات إليوت: خطأ في اكتشاف النمط",
    "Range Reversal: Trend too strong (ADX > 23)": "انعكاس نطاقي: الاتجاه قوي جدًا (ADX > 23)",
    "Range Reversal: RSI not in oversold zone": "انعكاس نطاقي: RSI ليس في منطقة تشبع بيعي",
    "MACD Momentum Negative": "زخم الماكد سلبي",
    "Bearish Trend": "اتجاه هابط",
    "RSI Out of Range": "مؤشر القوة النسبية خارج النطاق"
}

# --- إعداد تطبيق Flask و WebSocket ---
app = Flask(__name__)
CORS(app)
sock = Sock(app)
ws_clients: List[Any] = []
ws_clients_lock = Lock()

# --- فئة إدارة الاتصالات ---
class ConnectionManager:
    def __init__(self):
        self.db_connection_pool = None
        self.redis_connection = None
        self.binance_client = None
        self.ws_manager = None
        self.connection_locks = {
            'db': Lock(),
            'redis': Lock(),
            'binance': Lock(),
            'ws': Lock()
        }
    
    def get_db_connection(self):
        with self.connection_locks['db']:
            if not self.db_connection_pool or self.db_connection_pool.closed:
                self.db_connection_pool = self._create_db_connection_pool()
            return self.db_connection_pool
    
    def get_redis_connection(self):
        with self.connection_locks['redis']:
            if not self.redis_connection:
                self.redis_connection = self._create_redis_connection()
            return self.redis_connection
    
    def get_binance_client(self):
        with self.connection_locks['binance']:
            if not self.binance_client:
                self.binance_client = self._create_binance_client()
            return self.binance_client
    
    def get_ws_manager(self):
        with self.connection_locks['ws']:
            if not self.ws_manager:
                self.ws_manager = self._create_ws_manager()
            return self.ws_manager
    
    def _create_db_connection_pool(self):
        try:
            db_url_to_use = DB_URL
            if 'postgres' in db_url_to_use and 'sslmode' not in db_url_to_use:
                db_url_to_use += f"{'?' if '?' not in db_url_to_use else '&'}sslmode=require"
            
            conn = psycopg2.connect(
                db_url_to_use, 
                connect_timeout=15, 
                cursor_factory=RealDictCursor,
                application_name="CryptoBot_V34.0.5_Enhanced"
            )
            conn.autocommit = False
            
            with conn.cursor() as cur:
                cur.execute("SET statement_timeout TO 10000;")
                cur.execute("SET idle_in_transaction_session_timeout TO 20000;")
            
            return conn
        except Exception as e:
            logger.error(f"❌ [DB] Error creating database connection: {e}")
            raise
    
    def _create_redis_connection(self):
        try:
            redis_conn = redis.from_url(
                REDIS_URL, 
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            redis_conn.ping()
            return redis_conn
        except redis.exceptions.ConnectionError as e:
            logger.error(f"❌ [Redis] Connection failed: {e}")
            return None
    
    def _create_binance_client(self):
        try:
            return Client(
                API_KEY, 
                API_SECRET,
                testnet=paper_trading_mode,
                requests_params={'timeout': 10}
            )
        except Exception as e:
            logger.error(f"❌ [Binance] Client creation failed: {e}")
            raise
    
    def _create_ws_manager(self):
        try:
            ws = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
            ws.start()
            return ws
        except Exception as e:
            logger.error(f"❌ [WebSocket] Manager creation failed: {e}")
            raise

# إنشاء مدير الاتصالات
connection_manager = ConnectionManager()

# --- فئة إدارة الأخطاء ---
class ErrorHandler:
    @staticmethod
    def handle_database_error(error, operation="database operation"):
        logger.error(f"❌ [DB] Error during {operation}: {error}")
        
        try:
            global conn
            if conn:
                conn.close()
            init_db()
            return True
        except Exception as e:
            logger.critical(f"❌ [DB] Failed to reconnect after error: {e}")
            return False
    
    @staticmethod
    def handle_api_error(error, operation="API operation"):
        logger.error(f"❌ [API] Error during {operation}: {error}")
        
        try:
            global client
            client = connection_manager.get_binance_client()
            return True
        except Exception as e:
            logger.critical(f"❌ [API] Failed to recreate client after error: {e}")
            return False
    
    @staticmethod
    def handle_redis_error(error, operation="Redis operation"):
        logger.error(f"❌ [Redis] Error during {operation}: {error}")
        
        try:
            global redis_client
            redis_client = connection_manager.get_redis_connection()
            return True
        except Exception as e:
            logger.critical(f"❌ [Redis] Failed to reconnect after error: {e}")
            return False
    
    @staticmethod
    def handle_websocket_error(error, operation="WebSocket operation"):
        logger.error(f"❌ [WebSocket] Error during {operation}: {error}")
        
        try:
            global ws_manager
            if ws_manager:
                ws_manager.stop()
            ws_manager = connection_manager.get_ws_manager()
            ws_manager.start_ticker_socket(callback=handle_socket_message)
            return True
        except Exception as e:
            logger.critical(f"❌ [WebSocket] Failed to restart after error: {e}")
            return False

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
                            logger.warning(f"[WebSocket] Invalid price data for {symbol}: {ticker.get('c')}")
            
            if price_updates:
                broadcast({"type": "price_update", "payload": price_updates})
    except Exception as e:
        logger.error(f"❌ [WebSocket] Error processing message: {e}", exc_info=True)

# --- دوال تهيئة قاعدة البيانات ---
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
                    "initial_quantity": "DOUBLE PRECISION",
                    "created_at": "TIMESTAMP WITH TIME ZONE DEFAULT NOW()"
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
        logger.warning("[DB] Connection check failed. It might still be closed.")
        return False
    except (OperationalError, InterfaceError) as e:
        logger.error(f"[DB] Connection lost ({e}). Attempting to reconnect...")
        init_db()
        return conn is not None and conn.closed == 0

# --- دوال تهيئة Redis ---
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

def send_enhanced_telegram_message(message: str, force: bool = False, parse_mode: str = "Markdown"):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    
    settings = get_notification_settings()
    if not settings.get('telegram_enabled') and not force:
        return
    
    max_length = 4096
    messages = [message[i:i+max_length] for i in range(0, len(message), max_length)]
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    for i, msg in enumerate(messages):
        if i > 0:
            msg = f"*(الجزء {i+1}/{len(messages)})*\n\n{msg}"
        
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }
        
        for attempt in range(3):
            try:
                r = requests.post(url, data=payload, timeout=10)
                
                if r.status_code == 429:
                    retry_after = int(r.json().get("parameters", {}).get("retry_after", 1))
                    sleep_time = min(5, retry_after)
                    logger.warning(f"[Telegram] Rate limited. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                
                if not r.ok:
                    logger.warning(f"[Telegram] HTTP {r.status_code}: {r.text}")
                    if "can't parse entities" in r.text.lower() and parse_mode == "Markdown":
                        logger.info("[Telegram] Retrying without Markdown formatting...")
                        payload["parse_mode"] = None
                        continue
                
                if r.ok:
                    break
                    
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    logger.error(f"❌ [Telegram] Failed to send message after retries: {e}")
                time.sleep(1.5)

def send_trade_open_notification_enhanced(symbol: str, strategy_name: str, entry_price: float, stop_loss: float,
                                       target1: float, target2: float, quantity: float, is_real: bool,
                                       quality_score: int, atr_percent: float, notional_value: float,
                                       signal_details: Dict = None):
    trade_type = "حقيقية" if is_real else "ورقية"
    emoji = "🔥" if is_real else "📊"
    
    risk_pct = ((entry_price - stop_loss) / entry_price * 100)
    reward1_pct = ((target1 - entry_price) / entry_price * 100)
    reward2_pct = ((target2 - entry_price) / entry_price * 100)
    rr_ratio1 = reward1_pct / risk_pct if risk_pct > 0 else 0
    rr_ratio2 = reward2_pct / risk_pct if risk_pct > 0 else 0
    
    message = (
        f"{emoji} *صفقة {trade_type} جديدة (5 دقائق)*\n\n"
        f"*العملة:* `{symbol}`\n"
        f"*الاستراتيجية:* `{STRATEGY_NAMES.get(strategy_name, strategy_name)}`\n"
        f"*جودة الإشارة:* `{quality_score}/100`\n"
        f"*تقلب السوق:* `{atr_percent:.2f}%`\n\n"
        f"*سعر الدخول:* `{entry_price:.4f}`\n"
        f"*وقف الخسارة:* `{stop_loss:.4f}`\n"
        f"*الهدف الأول:* `{target1:.4f}`\n"
        f"*الهدف الثاني:* `{target2:.4f}`\n\n"
        f"*الكمية:* `{quantity:.4f}`\n"
        f"*قيمة الصفقة:* `${notional_value:.2f}`\n\n"
        f"*نسبة المخاطرة:* `{risk_pct:.2f}%`\n"
        f"*نسبة الربح 1:* `{reward1_pct:.2f}%` (RR: `{rr_ratio1:.2f}`)\n"
        f"*نسبة الربح 2:* `{reward2_pct:.2f}%` (RR: `{rr_ratio2:.2f}`)"
    )
    
    if signal_details:
        details_text = "\n\n*تفاصيل إضافية:*\n"
        for key, value in signal_details.items():
            if isinstance(value, float):
                details_text += f"{key}: `{value:.4f}`\n"
            else:
                details_text += f"{key}: `{value}`\n"
        message += details_text
    
    send_enhanced_telegram_message(message, force=True)

def send_trade_close_notification_enhanced(symbol: str, entry_price: float, close_price: float, 
                                       stop_loss: float, target1: float, target2: float,
                                       profit_pct: float, strategy_name: str, is_real: bool,
                                       close_reason: str, duration_minutes: int):
    trade_type = "حقيقية" if is_real else "ورقية"
    
    if profit_pct > 0:
        result_emoji = "✅"
        result_text = "ربح"
    else:
        result_emoji = "❌"
        result_text = "خسارة"
    
    if close_reason == "stop_loss":
        close_emoji = "🛑"
        close_text = "وقف الخسارة"
    elif close_reason == "target_1":
        close_emoji = "🎯"
        close_text = "الهدف الأول"
    elif close_reason == "target_2":
        close_emoji = "🏆"
        close_text = "الهدف الثاني"
    elif close_reason == "trailing_stop":
        close_emoji = "⬆️"
        close_text = "وقف متحرك"
    elif close_reason == "manual":
        close_emoji = "👤"
        close_text = "إغلاق يدوي"
    else:
        close_emoji = "❓"
        close_text = close_reason
    
    if duration_minutes < 60:
        duration_text = f"{duration_minutes} دقيقة"
    elif duration_minutes < 1440:
        hours = duration_minutes // 60
        minutes = duration_minutes % 60
        duration_text = f"{hours} ساعة و {minutes} دقيقة"
    else:
        days = duration_minutes // 1440
        hours = (duration_minutes % 1440) // 60
        duration_text = f"{days} يوم و {hours} ساعة"
    
    risk_pct = ((entry_price - stop_loss) / entry_price * 100)
    actual_reward_pct = ((close_price - entry_price) / entry_price * 100)
    actual_rr_ratio = actual_reward_pct / risk_pct if risk_pct > 0 else 0
    
    message = (
        f"{result_emoji} *إغلاق صفقة {trade_type} (5 دقائق)*\n\n"
        f"*العملة:* `{symbol}`\n"
        f"*الاستراتيجية:* `{STRATEGY_NAMES.get(strategy_name, strategy_name)}`\n"
        f"*النتيجة:* `{result_text} {profit_pct:.2f}%`\n"
        f"*سبب الإغلاق:* `{close_text}`\n"
        f"*مدة الصفقة:* `{duration_text}`\n\n"
        f"*سعر الدخول:* `{entry_price:.4f}`\n"
        f"*سعر الخروج:* `{close_price:.4f}`\n"
        f"*وقف الخسارة:* `{stop_loss:.4f}`\n"
        f"*الهدف الأول:* `{target1:.4f}`\n"
        f"*الهدف الثاني:* `{target2:.4f}`\n\n"
        f"*نسبة المخاطرة:* `{risk_pct:.2f}%`\n"
        f"*نسبة المكافأة الفعلية:* `{actual_reward_pct:.2f}%`\n"
        f"*نسبة المخاطرة/المكافأة:* `{actual_rr_ratio:.2f}`"
    )
    
    send_enhanced_telegram_message(message, force=True)

def send_daily_performance_report_enhanced():
    if not check_db_connection() or not conn:
        return
    
    try:
        with conn.cursor() as cur:
            today = datetime.now(timezone.utc).date()
            
            cur.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit_percentage > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN profit_percentage <= 0 THEN 1 ELSE 0 END) as losing_trades,
                    AVG(profit_percentage) as avg_profit,
                    SUM(profit_percentage) as total_profit,
                    MAX(profit_percentage) as best_trade,
                    MIN(profit_percentage) as worst_trade,
                    AVG(CASE WHEN profit_percentage > 0 THEN profit_percentage ELSE NULL END) as avg_win,
                    AVG(CASE WHEN profit_percentage <= 0 THEN profit_percentage ELSE NULL END) as avg_loss
                FROM signals
                WHERE closed_at::date = %s AND status = 'closed'
            """, (today,))
            
            stats = cur.fetchone()
            
            if not stats or stats['total_trades'] == 0:
                message = (
                    f"📊 *تقرير الأداء اليومي*\n\n"
                    f"*التاريخ:* `{today.strftime('%Y-%m-%d')}`\n\n"
                    f"لا توجد صفقات مغلقة اليوم."
                )
                send_enhanced_telegram_message(message, force=True)
                return
            
            cur.execute("""
                SELECT symbol, profit_percentage, strategy_name
                FROM signals
                WHERE closed_at::date = %s AND status = 'closed'
                ORDER BY profit_percentage DESC LIMIT 1
            """, (today,))
            best_trade = cur.fetchone()
            
            cur.execute("""
                SELECT symbol, profit_percentage, strategy_name
                FROM signals
                WHERE closed_at::date = %s AND status = 'closed'
                ORDER BY profit_percentage ASC LIMIT 1
            """, (today,))
            worst_trade = cur.fetchone()
            
            cur.execute("""
                SELECT strategy_name, 
                       COUNT(*) as trade_count,
                       AVG(profit_percentage) as avg_profit,
                       SUM(CASE WHEN profit_percentage > 0 THEN 1 ELSE 0 END) as wins
                FROM signals
                WHERE closed_at::date = %s AND status = 'closed'
                GROUP BY strategy_name
                ORDER BY avg_profit DESC
            """, (today,))
            
            strategy_stats = cur.fetchall()
            
            profit_factor = 0
            if stats['avg_loss'] != 0:
                profit_factor = abs(stats['avg_win'] / stats['avg_loss'])
            
            win_rate = (stats['winning_trades'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0
            
            message = (
                f"📈 *تقرير الأداء اليومي*\n\n"
                f"*التاريخ:* `{today.strftime('%Y-%m-%d')}`\n\n"
                f"*إجمالي الصفقات:* `{stats['total_trades']}`\n"
                f"*الصفقات الرابحة:* `{stats.get('winning_trades', 0)}`\n"
                f"*الصفقات الخاسرة:* `{stats.get('losing_trades', 0)}`\n"
                f"*نسبة الربح:* `{win_rate:.1f}%`\n"
                f"*متوسط الربح:* `{stats.get('avg_profit', 0):.2f}%`\n"
                f"*إجمالي الربح:* `{stats.get('total_profit', 0):.2f}%`\n"
                f"*معامل الربح:* `{profit_factor:.2f}`\n\n"
            )
            
            if best_trade:
                message += (
                    f"🏆 *أفضل صفقة:* `{best_trade['symbol']}` | "
                    f"الربح: `{best_trade['profit_percentage']:.2f}%` | "
                    f"الاستراتيجية: `{STRATEGY_NAMES.get(best_trade['strategy_name'], best_trade['strategy_name'])}`\n"
                )
            
            if worst_trade:
                message += (
                    f"📉 *أسوأ صفقة:* `{worst_trade['symbol']}` | "
                    f"الخسارة: `{worst_trade['profit_percentage']:.2f}%` | "
                    f"الاستراتيجية: `{STRATEGY_NAMES.get(worst_trade['strategy_name'], worst_trade['strategy_name'])}`\n\n"
                )
            
            if strategy_stats:
                message += "*أداء الاستراتيجيات:*\n"
                for strategy in strategy_stats:
                    strategy_win_rate = (strategy['wins'] / strategy['trade_count'] * 100) if strategy['trade_count'] > 0 else 0
                    message += (
                        f"`{STRATEGY_NAMES.get(strategy['strategy_name'], strategy['strategy_name'])}`: "
                        f"{strategy['trade_count']} صفقة | "
                        f"متوسط: {strategy['avg_profit']:.2f}% | "
                        f"ربح: {strategy_win_rate:.0f}%\n"
                    )
            
            send_enhanced_telegram_message(message, force=True)
            
    except Exception as e:
        logger.error(f"❌ [Daily Report] Error generating daily report: {e}", exc_info=True)

def send_market_state_notification():
    with market_state_lock:
        state = dict(current_market_state)
    
    if not state or not state.get("trend_details_by_tf"):
        return

    message = f"🌐 *تحديث حالة السوق*\n\n"
    for tf, details in state["trend_details_by_tf"].items():
        trend = details.get("trend", "N/A")
        emoji = "🟢" if trend == "bullish" else "🔴" if trend == "bearish" else "🟡"
        message += f"{emoji} *{tf}:* {trend.capitalize()} (ADX: {details.get('adx', 0):.1f}, RSI: {details.get('rsi', 0):.1f})\n"
    
    send_enhanced_telegram_message(message, force=False)

def schedule_periodic_reports():
    logger.info("Starting periodic reports scheduler...")
    while True:
        try:
            now = datetime.now(timezone.utc)
            if now.hour == 23 and now.minute == 59:
                send_daily_performance_report_enhanced()
                time.sleep(61)
            if now.hour % 6 == 0 and now.minute == 0:
                send_market_state_notification()
                time.sleep(61)
            time.sleep(30)
        except Exception as e:
            logger.error(f"❌ [Periodic Reports] Error in scheduler: {e}", exc_info=True)
            time.sleep(60)

def start_periodic_reports():
    reports_thread = Thread(target=schedule_periodic_reports, daemon=True)
    reports_thread.start()
    logger.info("✅ [Periodic Reports] Started periodic reports scheduler thread.")

# --- دوال الحصول على البيانات ---
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

# --- دوال حساب المؤشرات الفنية ---
def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    
    # --- SMA Calculations ---
    df_calc['sma7'] = df_calc['close'].rolling(window=7).mean()
    df_calc['sma200'] = df_calc['close'].rolling(window=200).mean()

    # --- EMA Calculations ---
    df_calc['ema9'] = df_calc['close'].ewm(span=9, adjust=False).mean()
    df_calc['ema13'] = df_calc['close'].ewm(span=13, adjust=False).mean()
    df_calc['ema21'] = df_calc['close'].ewm(span=21, adjust=False).mean()
    df_calc['ema34'] = df_calc['close'].ewm(span=34, adjust=False).mean()
    df_calc['ema50'] = df_calc['close'].ewm(span=50, adjust=False).mean()
    df_calc['ema100'] = df_calc['close'].ewm(span=100, adjust=False).mean()
    df_calc['ema200'] = df_calc['close'].ewm(span=200, adjust=False).mean()
    
    # --- ATR and ADX ---
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
    
    # --- RSI Calculation ---
    delta = df_calc['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=7).mean()
    avg_loss = loss.rolling(window=7).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df_calc['rsi'] = 100 - (100 / (1 + rs))
    
    # --- Bollinger Bands ---
    bb_middle = df_calc['close'].rolling(window=20).mean()
    bb_std = df_calc['close'].rolling(window=20).std()
    df_calc['bb_middle'] = bb_middle
    df_calc['bb_lower'] = bb_middle - (bb_std * 2)
    df_calc['bb_upper'] = bb_middle + (bb_std * 2)
    df_calc['bb_width'] = (df_calc['bb_upper'] - df_calc['bb_lower']) / df_calc['bb_middle'].replace(0, 1e-9)
    
    # --- MACD ---
    exp1 = df_calc['close'].ewm(span=8, adjust=False).mean()
    exp2 = df_calc['close'].ewm(span=17, adjust=False).mean()
    df_calc['macd'] = exp1 - exp2
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()
    df_calc['macd_hist'] = df_calc['macd'] - df_calc['macd_signal']
    
    # --- Stochastic ---
    low_14 = df_calc['low'].rolling(14).min()
    high_14 = df_calc['high'].rolling(14).max()
    high_low_range = high_14 - low_14
    meaningful_range = high_low_range > (df_calc['close'] * 0.0001)
    df_calc['stoch_k'] = np.where(
        meaningful_range,
        100 * ((df_calc['close'] - low_14) / high_low_range.replace(0, 1e-9)),
        50
    )
    df_calc['stoch_d'] = df_calc['stoch_k'].rolling(3).mean()
    
    # --- VWAP ---
    df_calc['vwap'] = (df_calc['close'] * df_calc['volume']).cumsum() / df_calc['volume'].cumsum()
    
    return df_calc

# --- دوال تحميل الإعدادات والبيانات ---
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
    global FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT, MAX_OPEN_TRADES, USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY, USE_MOMENTUM_VOLATILITY_STRATEGY, USE_ELLIOTT_WAVE_STRATEGY, USE_RANGE_REVERSAL_STRATEGY, paper_trading_mode, MIN_SIGNAL_QUALITY
    if not redis_client: return
    try:
        settings_data = redis_client.get('trading_settings')
        if settings_data:
            settings = json.loads(settings_data)
            with trade_amount_lock:
                FIXED_TRADE_AMOUNT_MIN_USDT = settings.get('FIXED_TRADE_AMOUNT_MIN_USDT', 4.5)
                FIXED_TRADE_AMOUNT_MAX_USDT = settings.get('FIXED_TRADE_AMOUNT_MAX_USDT', 6.5)
            MAX_OPEN_TRADES = settings.get('MAX_OPEN_TRADES', 3)
            with trading_mode_lock: paper_trading_mode = settings.get('paper_trading_mode', True)
            
        quality_settings_data = redis_client.get('signal_quality_settings')
        if quality_settings_data:
            quality_settings = json.loads(quality_settings_data)
            with min_quality_lock: MIN_SIGNAL_QUALITY = quality_settings.get('min_quality', 70)

        strategies_data = redis_client.get('strategy_settings')
        if strategies_data:
            strategies = json.loads(strategies_data)
            USE_BB_STOCH_STRATEGY = strategies.get('USE_BB_STOCH_STRATEGY', True)
            USE_MACD_EMA_STRATEGY = strategies.get('USE_MACD_EMA_STRATEGY', True)
            USE_EMA_RSI_STRATEGY = strategies.get('USE_EMA_RSI_STRATEGY', True)
            USE_PULLBACK_STRATEGY = strategies.get('USE_PULLBACK_STRATEGY', True)
            USE_MOMENTUM_VOLATILITY_STRATEGY = strategies.get('USE_MOMENTUM_VOLATILITY_STRATEGY', True)
            USE_ELLIOTT_WAVE_STRATEGY = strategies.get('USE_ELLIOTT_WAVE_STRATEGY', True)
            USE_RANGE_REVERSAL_STRATEGY = strategies.get('USE_RANGE_REVERSAL_STRATEGY', True)

        logger.info("✅ [Redis] Successfully loaded settings from Redis.")
    except Exception as e:
        logger.error(f"❌ [Redis] Error loading settings: {e}")

def save_settings_to_redis():
    if not redis_client:
        logger.warning("Redis client not available, cannot save settings")
        return False
    
    try:
        trading_settings = {
            'FIXED_TRADE_AMOUNT_MIN_USDT': FIXED_TRADE_AMOUNT_MIN_USDT,
            'FIXED_TRADE_AMOUNT_MAX_USDT': FIXED_TRADE_AMOUNT_MAX_USDT,
            'MAX_OPEN_TRADES': MAX_OPEN_TRADES,
            'paper_trading_mode': paper_trading_mode
        }
        redis_client.set('trading_settings', json.dumps(trading_settings))
        
        quality_settings = {'min_quality': MIN_SIGNAL_QUALITY}
        redis_client.set('signal_quality_settings', json.dumps(quality_settings))
        
        strategy_settings = {
            'USE_BB_STOCH_STRATEGY': USE_BB_STOCH_STRATEGY,
            'USE_MACD_EMA_STRATEGY': USE_MACD_EMA_STRATEGY,
            'USE_EMA_RSI_STRATEGY': USE_EMA_RSI_STRATEGY,
            'USE_PULLBACK_STRATEGY': USE_PULLBACK_STRATEGY,
            'USE_MOMENTUM_VOLATILITY_STRATEGY': USE_MOMENTUM_VOLATILITY_STRATEGY,
            'USE_ELLIOTT_WAVE_STRATEGY': USE_ELLIOTT_WAVE_STRATEGY,
            'USE_RANGE_REVERSAL_STRATEGY': USE_RANGE_REVERSAL_STRATEGY
        }
        redis_client.set('strategy_settings', json.dumps(strategy_settings))
        
        logger.info("Settings saved to Redis successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error saving settings to Redis: {e}")
        return False

# --- دوال الفلاتر الديناميكية ---
def get_wave_retracement(df: pd.DataFrame) -> float:
    try:
        highs = df['high'].values
        lows = df['low'].values
        peaks_idx = argrelextrema(highs, np.greater, order=5)[0]
        troughs_idx = argrelextrema(lows, np.less, order=5)[0]
        
        if len(peaks_idx) < 1 or len(troughs_idx) < 2: return 999.0
        
        last_trough_idx = troughs_idx[-1]
        prev_peak_idx = peaks_idx[peaks_idx < last_trough_idx][-1]
        prev_trough_idx = troughs_idx[troughs_idx < prev_peak_idx][-1]

        wave_start_price = lows[prev_trough_idx]
        wave_end_price = highs[prev_peak_idx]
        retracement_price = lows[last_trough_idx]

        wave_height = wave_end_price - wave_start_price
        if wave_height <= 0: return 999.0
        
        retracement = (wave_end_price - retracement_price) / wave_height
        return retracement
    except Exception:
        return 999.0

def check_bb_stoch_dynamic_filters_enhanced(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    
    bb_width = df['bb_width']
    bb_width_ma = bb_width.rolling(20).mean()
    bb_width_std = bb_width.rolling(20).std()
    
    if atr_percent > 2.5:
        bb_multiplier = 1.5
    elif atr_percent > 1.5:
        bb_multiplier = 1.3
    else:
        bb_multiplier = 1.1
    
    dynamic_bb_threshold = bb_width_ma.iloc[-1] + (bb_width_std.iloc[-1] * bb_multiplier)
    bb_width_ok = bb_width.iloc[-1] > dynamic_bb_threshold
    
    if atr_percent > 2.5:
        stoch_threshold = 28
    elif atr_percent > 1.5:
        stoch_threshold = 25
    else:
        stoch_threshold = 20
    
    stoch_ok = last_row['stoch_k'] > stoch_threshold
    
    volume_ma = df['volume'].rolling(20).mean()
    volume_std = df['volume'].rolling(20).std()
    
    if atr_percent > 2.5:
        volume_multiplier = 1.4
    elif atr_percent > 1.5:
        volume_multiplier = 1.2
    else:
        volume_multiplier = 1.0
    
    dynamic_volume_threshold = volume_ma.iloc[-1] + (volume_std.iloc[-1] * volume_multiplier)
    volume_ok = last_row['volume'] > dynamic_volume_threshold
    
    macd_hist = df['macd_hist']
    macd_hist_ma = macd_hist.rolling(10).mean()
    
    if atr_percent > 2.5:
        momentum_threshold = macd_hist_ma.iloc[-1] * 1.5
    elif atr_percent > 1.5:
        momentum_threshold = macd_hist_ma.iloc[-1] * 1.2
    else:
        momentum_threshold = macd_hist_ma.iloc[-1] * 1.0
    
    momentum_ok = macd_hist.iloc[-1] > momentum_threshold
    
    return {
        'bb_width_ok': bb_width_ok,
        'stoch_ok': stoch_ok,
        'volume_ok': volume_ok,
        'momentum_ok': momentum_ok
    }

def check_macd_ema_dynamic_filters_enhanced(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    
    adx = df['adx']
    adx_ma = adx.rolling(20).mean()
    adx_std = adx.rolling(20).std()
    
    if atr_percent > 2.5:
        adx_multiplier = 0.8
    elif atr_percent > 1.5:
        adx_multiplier = 0.9
    else:
        adx_multiplier = 1.0
    
    dynamic_adx_threshold = adx_ma.iloc[-1] + (adx_std.iloc[-1] * adx_multiplier)
    adx_ok = last_row['adx'] > dynamic_adx_threshold
    
    volume_ma = df['volume'].rolling(20).mean()
    volume_std = df['volume'].rolling(20).std()
    
    if atr_percent > 2.5:
        volume_multiplier = 1.5
    elif atr_percent > 1.5:
        volume_multiplier = 1.3
    else:
        volume_multiplier = 1.1
    
    dynamic_volume_threshold = volume_ma.iloc[-1] + (volume_std.iloc[-1] * volume_multiplier)
    volume_ok = last_row['volume'] > dynamic_volume_threshold
    
    macd_momentum = df['macd_hist'].diff()
    macd_momentum_ma = macd_momentum.rolling(10).mean()
    macd_momentum_std = macd_momentum.rolling(10).std()
    
    if atr_percent > 2.5:
        momentum_multiplier = 0.7
    elif atr_percent > 1.5:
        momentum_multiplier = 0.85
    else:
        momentum_multiplier = 1.0
    
    dynamic_momentum_threshold = macd_momentum_ma.iloc[-1] + (macd_momentum_std.iloc[-1] * momentum_multiplier)
    momentum_ok = macd_momentum.iloc[-1] > dynamic_momentum_threshold
    
    return {
        'adx_ok': adx_ok,
        'volume_ok': volume_ok,
        'momentum_ok': momentum_ok
    }

def check_ema_rsi_dynamic_filters_enhanced(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    adx = last_row.get('adx', 0)
    atr_percent = last_row.get('atr_percent', 0)
    
    if adx > 25:
        if atr_percent > 2.0:
            rsi_lower, rsi_upper = 45, 75
        else:
            rsi_lower, rsi_upper = 42, 72
    else:
        if atr_percent > 2.0:
            rsi_lower, rsi_upper = 50, 70
        else:
            rsi_lower, rsi_upper = 48, 68
    
    rsi_ok = rsi_lower < last_row['rsi'] < rsi_upper
    
    ema_spread = (df['ema9'] - df['ema21']) / df['ema21'].replace(0, 1e-9)
    ema_spread_ma = ema_spread.rolling(20).mean()
    ema_spread_std = ema_spread.rolling(20).std()
    
    if atr_percent > 2.5:
        spread_multiplier = 1.5
    elif atr_percent > 1.5:
        spread_multiplier = 1.3
    else:
        spread_multiplier = 1.1
    
    dynamic_ema_threshold = ema_spread_ma.iloc[-1] + (ema_spread_std.iloc[-1] * spread_multiplier)
    ema_ok = ema_spread.iloc[-1] > dynamic_ema_threshold
    
    volume_ma = df['volume'].rolling(20).mean()
    volume_std = df['volume'].rolling(20).std()
    
    trend_strength_multiplier = 1 + (adx / 100)
    
    if atr_percent > 2.5:
        volatility_adjustment = 1.3
    elif atr_percent > 1.5:
        volatility_adjustment = 1.15
    else:
        volatility_adjustment = 1.0
    
    dynamic_volume_threshold = volume_ma.iloc[-1] * trend_strength_multiplier * volatility_adjustment
    volume_ok = last_row['volume'] > dynamic_volume_threshold
    
    return {
        'rsi_ok': rsi_ok,
        'ema_ok': ema_ok,
        'volume_ok': volume_ok
    }

def check_pullback_dynamic_filters_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    
    if atr_percent > 2.5:
        pullback_depth = 0.04
    elif atr_percent > 1.5:
        pullback_depth = 0.03
    else:
        pullback_depth = 0.02
    
    if mtf_trend.get('5m') == 'bullish' and mtf_trend.get('15m') == 'bullish':
        pullback_depth *= 1.2
    
    recent_low = df['low'].tail(5).min()
    recovery_threshold = recent_low * (1 + (pullback_depth * 0.9))
    recovery_ok = last_row['close'] > recovery_threshold
    
    volume_ma = df['volume'].rolling(20).mean()
    volume_std = df['volume'].rolling(20).std()
    
    if atr_percent > 2.5:
        volume_multiplier = 1.4
    elif atr_percent > 1.5:
        volume_multiplier = 1.2
    else:
        volume_multiplier = 1.0
    
    dynamic_volume_threshold = volume_ma.iloc[-1] + (volume_std.iloc[-1] * volume_multiplier)
    volume_ok = last_row['volume'] > dynamic_volume_threshold
    
    rsi = df['rsi']
    rsi_prev = df['rsi'].shift(1)
    rsi_momentum = rsi - rsi_prev
    
    if atr_percent > 2.5:
        rsi_momentum_threshold = 2.0
    elif atr_percent > 1.5:
        rsi_momentum_threshold = 1.5
    else:
        rsi_momentum_threshold = 1.0
    
    rsi_momentum_ok = rsi_momentum.iloc[-1] > rsi_momentum_threshold
    
    return {
        'recovery_ok': recovery_ok,
        'volume_ok': volume_ok,
        'rsi_momentum_ok': rsi_momentum_ok
    }

def check_momentum_volatility_dynamic_filters_enhanced(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = df['atr_percent']
    
    volatility_ma = atr_percent.rolling(20).mean()
    volatility_std = atr_percent.rolling(20).std()
    
    if atr_percent.iloc[-5] > 2.0:
        dynamic_vol_min = volatility_ma.iloc[-1] - (volatility_std.iloc[-1] * 1.0)
        dynamic_vol_max = volatility_ma.iloc[-1] + (volatility_std.iloc[-1] * 2.0)
    else:
        dynamic_vol_min = volatility_ma.iloc[-1] - (volatility_std.iloc[-1] * 1.5)
        dynamic_vol_max = volatility_ma.iloc[-1] + (volatility_std.iloc[-1] * 1.5)
    
    volatility_ok = dynamic_vol_min <= atr_percent.iloc[-1] <= dynamic_vol_max
    
    rsi = df['rsi']
    macd_hist = df['macd_hist']
    
    if atr_percent.iloc[-1] > 2.0:
        rsi_threshold = 52
        macd_threshold = macd_hist.rolling(10).mean().iloc[-1] * 1.2
    else:
        rsi_threshold = 50
        macd_threshold = macd_hist.rolling(10).mean().iloc[-1] * 1.0
    
    is_momentum_ok = (rsi.iloc[-1] > rsi_threshold) and (macd_hist.iloc[-1] > macd_threshold)
    
    adx = df['adx']
    adx_ma = adx.rolling(20).mean()
    adx_std = adx.rolling(20).std()
    
    if atr_percent.iloc[-1] > 2.0:
        adx_multiplier = 0.8
    else:
        adx_multiplier = 1.0
    
    dynamic_adx_threshold = adx_ma.iloc[-1] + (adx_std.iloc[-1] * adx_multiplier)
    adx_ok = adx.iloc[-1] > dynamic_adx_threshold
    
    return {
        'volatility_ok': volatility_ok,
        'momentum_ok': is_momentum_ok,
        'adx_ok': adx_ok
    }

def check_elliott_wave_dynamic_filters_enhanced(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    
    if atr_percent > 2.5:
        fib_min, fib_max = 0.236, 0.786
    elif atr_percent > 1.5:
        fib_min, fib_max = 0.236, 0.618
    else:
        fib_min, fib_max = 0.382, 0.618
    
    fibonacci_ok = fib_min <= get_wave_retracement(df) <= fib_max
    
    volume_ma = df['volume'].rolling(20).mean()
    volume_std = df['volume'].rolling(20).std()
    
    if atr_percent > 2.5:
        volume_multiplier = 1.5
    elif atr_percent > 1.5:
        volume_multiplier = 1.3
    else:
        volume_multiplier = 1.1
    
    dynamic_volume_threshold = volume_ma.iloc[-1] + (volume_std.iloc[-1] * volume_multiplier)
    volume_ok = last_row['volume'] > dynamic_volume_threshold
    
    macd_momentum = df['macd_hist'].rolling(5).mean()
    macd_momentum_ma = macd_momentum.rolling(20).mean()
    macd_momentum_std = macd_momentum.rolling(20).std()
    
    if atr_percent > 2.5:
        momentum_multiplier = 0.7
    elif atr_percent > 1.5:
        momentum_multiplier = 0.85
    else:
        momentum_multiplier = 1.0
    
    dynamic_momentum_threshold = macd_momentum_ma.iloc[-1] + (macd_momentum_std.iloc[-1] * momentum_multiplier)
    momentum_ok = macd_momentum.iloc[-1] > dynamic_momentum_threshold
    
    wave_structure_ok = True
    
    try:
        highs = df['high'].values
        lows = df['low'].values
        
        peaks_idx = argrelextrema(highs, np.greater, order=5)[0]
        troughs_idx = argrelextrema(lows, np.less, order=5)[0]
        
        if len(peaks_idx) >= 3 and len(troughs_idx) >= 3:
            wave_heights = []
            for i in range(1, len(peaks_idx)):
                if i < len(troughs_idx):
                    wave_height = highs[peaks_idx[i]] - lows[troughs_idx[i-1]]
                    wave_heights.append(wave_height)
            
            if len(wave_heights) >= 2:
                wave_structure_ok = wave_heights[1] > wave_heights[0]
    except Exception:
        wave_structure_ok = True
    
    return {
        'fibonacci_ok': fibonacci_ok,
        'volume_ok': volume_ok,
        'momentum_ok': momentum_ok,
        'wave_structure_ok': wave_structure_ok
    }

def check_range_reversal_dynamic_filters_enhanced(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    adx = last_row.get('adx', 99)
    atr_percent = last_row.get('atr_percent', 0)
    
    if atr_percent > 2.0:
        adx_threshold = 25
    else:
        adx_threshold = 23
    
    adx_ok = adx < adx_threshold
    
    rsi = last_row.get('rsi', 50)
    
    if atr_percent > 2.0:
        rsi_threshold = 40
    else:
        rsi_threshold = 35
    
    rsi_ok = rsi < rsi_threshold
    
    bb_lower = last_row.get('bb_lower', 0)
    bb_upper = last_row.get('bb_upper', 0)
    current_price = last_row.get('close', 0)
    
    if bb_upper > bb_lower:
        price_position = (current_price - bb_lower) / (bb_upper - bb_lower)
        
        if atr_percent > 2.0:
            position_threshold = 0.3
        else:
            position_threshold = 0.25
        
        price_position_ok = price_position < position_threshold
    else:
        price_position_ok = True
    
    volume_ma = df['volume'].rolling(20).mean()
    volume_std = df['volume'].rolling(20).std()
    
    if atr_percent > 2.0:
        volume_multiplier = 1.2
    else:
        volume_multiplier = 1.0
    
    dynamic_volume_threshold = volume_ma.iloc[-1] + (volume_std.iloc[-1] * volume_multiplier)
    volume_ok = last_row['volume'] > dynamic_volume_threshold
    
    return {
        'adx_ok': adx_ok,
        'rsi_ok': rsi_ok,
        'price_position_ok': price_position_ok,
        'volume_ok': volume_ok
    }

# --- الفلاتر العامة ---
def add_news_filter() -> bool:
    news_hours = [(12, 30), (14, 0), (18, 30)]
    now = datetime.now(timezone.utc)
    for hour, minute in news_hours:
        if now.hour == hour and abs(now.minute - minute) <= 15:
            return False
    return True

def add_liquidity_filter() -> bool:
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5: return False
    if now.hour >= 22 or now.hour <= 2: return False
    return True

def add_correlation_filter(new_symbol: str) -> bool:
    correlated_groups = [
        {'BTCUSDT', 'ETHUSDT', 'BCHUSDT'}, {'ADAUSDT', 'DOTUSDT', 'LINKUSDT'},
        {'SOLUSDT', 'AVAXUSDT', 'MATICUSDT'},
    ]
    with signal_cache_lock: open_symbols = set(open_signals_cache.keys())
    if not open_symbols: return True
    for group in correlated_groups:
        if new_symbol in group and not open_symbols.isdisjoint(group):
            return False
    return True

def check_market_volatility_filter_enhanced(df: pd.DataFrame, symbol: str = "Unknown") -> bool:
    if 'atr_percent' not in df.columns or df['atr_percent'].isnull().all():
        log_rejection(symbol, "Market Volatility Filter Failed", {"reason": "No ATR data"})
        return False
    
    last_atr_percent = float(df.iloc[-1].get('atr_percent', 0))
    ATR_PERCENT_MIN = 0.5
    ATR_PERCENT_MAX = 2.8
    
    if not (ATR_PERCENT_MIN <= last_atr_percent <= ATR_PERCENT_MAX):
        log_rejection(symbol, "Market Volatility Filter Failed", {
            "atr": f"{last_atr_percent:.2f}%",
            "range": f"({ATR_PERCENT_MIN:.2f}-{ATR_PERCENT_MAX:.2f})%"
        })
        return False
    
    return True

# --- استراتيجيات التداول المحسنة ---
def check_bb_stoch_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> Dict:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 50:
        log_rejection(symbol_name, "Insufficient Historical Data")
        return {"signal": False, "quality": 0}
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    if not (last['ema50'] > last['ema200'] and last['close'] > last['ema21']):
        log_rejection(symbol_name, "BB: Price below EMA50 (bearish trend)")
        return {"signal": False, "quality": 0}
    
    bb_lower = last['bb_lower']
    bb_middle = last['bb_middle']
    bb_upper = last['bb_upper']
    bb_width = last['bb_width']
    
    price_above_lower = last['close'] > bb_lower
    price_near_lower = (last['close'] - bb_lower) / (bb_upper - bb_lower) < 0.25
    
    stoch_k = last['stoch_k']
    stoch_d = last['stoch_d']
    stoch_upward = stoch_k > stoch_d and prev['stoch_k'] <= prev['stoch_d']
    stoch_oversold = stoch_k < 25 and stoch_d < 30
    
    macd_hist = last['macd_hist']
    macd_hist_prev = prev['macd_hist']
    macd_upward = macd_hist > macd_hist_prev
    
    volume_ma = df['volume'].rolling(10).mean()
    volume_spike = last['volume'] > volume_ma.iloc[-1] * 1.5
    
    dynamic_filters = check_bb_stoch_dynamic_filters_enhanced(df)
    
    quality_score = 0
    
    if price_above_lower: quality_score += 15
    if price_near_lower: quality_score += 15
    if stoch_upward: quality_score += 15
    if stoch_oversold: quality_score += 10
    if macd_upward: quality_score += 10
    if volume_spike: quality_score += 10
    if dynamic_filters['bb_width_ok']: quality_score += 10
    if dynamic_filters['stoch_ok']: quality_score += 10
    if dynamic_filters['volume_ok']: quality_score += 5
    
    entry_conditions = (
        price_above_lower and
        price_near_lower and
        stoch_upward and
        stoch_oversold and
        macd_upward and
        volume_spike and
        dynamic_filters['bb_width_ok'] and
        dynamic_filters['stoch_ok'] and
        dynamic_filters['volume_ok']
    )
    
    if entry_conditions:
        return {
            "signal": True,
            "quality": min(quality_score, 100),
            "details": {
                "bb_width": bb_width,
                "stoch_k": stoch_k,
                "stoch_d": stoch_d,
                "volume_ratio": last['volume'] / volume_ma.iloc[-1],
                "macd_hist": macd_hist
            }
        }
    else:
        if not dynamic_filters['bb_width_ok']:
            log_rejection(symbol_name, "DYN_BB_WIDTH_LOW")
        elif not dynamic_filters['stoch_ok']:
            log_rejection(symbol_name, "DYN_STOCH_LOW")
        elif not dynamic_filters['volume_ok']:
            log_rejection(symbol_name, "DYN_VOLUME_LOW")
        elif not stoch_upward:
            log_rejection(symbol_name, "Stochastic not upward")
        elif not macd_upward:
            log_rejection(symbol_name, "MACD momentum negative")
        elif not volume_spike:
            log_rejection(symbol_name, "Volume spike not detected")
        
        return {"signal": False, "quality": quality_score}

def check_macd_ema_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> Dict:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 50:
        log_rejection(symbol_name, "Insufficient Historical Data")
        return {"signal": False, "quality": 0}
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    if not (last['ema21'] > last['ema50'] and last['ema50'] > last['ema200']):
        log_rejection(symbol_name, "MACD: Bearish trend")
        return {"signal": False, "quality": 0}
    
    macd = last['macd']
    macd_signal = last['macd_signal']
    macd_hist = last['macd_hist']
    macd_hist_prev = prev['macd_hist']
    
    macd_above_signal = macd > macd_signal
    macd_hist_positive = macd_hist > 0
    macd_hist_increasing = macd_hist > macd_hist_prev
    
    ema9 = last['ema9']
    ema13 = last['ema13']
    ema21 = last['ema21']
    
    ema_bullish_order = ema9 > ema13 > ema21
    price_above_ema9 = last['close'] > ema9
    
    adx = last['adx']
    adx_strong = adx > 20
    
    volume_ma = df['volume'].rolling(10).mean()
    volume_increasing = last['volume'] > volume_ma.iloc[-1] * 1.2
    
    dynamic_filters = check_macd_ema_dynamic_filters_enhanced(df)
    
    quality_score = 0
    
    if macd_above_signal: quality_score += 15
    if macd_hist_positive: quality_score += 10
    if macd_hist_increasing: quality_score += 15
    if ema_bullish_order: quality_score += 10
    if price_above_ema9: quality_score += 10
    if adx_strong: quality_score += 10
    if volume_increasing: quality_score += 5
    if dynamic_filters['adx_ok']: quality_score += 10
    if dynamic_filters['volume_ok']: quality_score += 10
    if dynamic_filters['momentum_ok']: quality_score += 5
    
    entry_conditions = (
        macd_above_signal and
        macd_hist_positive and
        macd_hist_increasing and
        ema_bullish_order and
        price_above_ema9 and
        adx_strong and
        volume_increasing and
        dynamic_filters['adx_ok'] and
        dynamic_filters['volume_ok'] and
        dynamic_filters['momentum_ok']
    )
    
    if entry_conditions:
        return {
            "signal": True,
            "quality": min(quality_score, 100),
            "details": {
                "macd": macd,
                "macd_signal": macd_signal,
                "macd_hist": macd_hist,
                "adx": adx,
                "volume_ratio": last['volume'] / volume_ma.iloc[-1]
            }
        }
    else:
        if not dynamic_filters['adx_ok']:
            log_rejection(symbol_name, "DYN_ADX_LOW")
        elif not dynamic_filters['volume_ok']:
            log_rejection(symbol_name, "DYN_VOLUME_LOW")
        elif not dynamic_filters['momentum_ok']:
            log_rejection(symbol_name, "DYN_MACD_MOMENTUM_LOW")
        elif not adx_strong:
            log_rejection(symbol_name, "ADX not strong enough")
        elif not macd_hist_increasing:
            log_rejection(symbol_name, "MACD momentum negative")
        
        return {"signal": False, "quality": quality_score}

def check_ema_rsi_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> Dict:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 50:
        log_rejection(symbol_name, "Insufficient Historical Data")
        return {"signal": False, "quality": 0}
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    if not (last['ema50'] > last['ema200']):
        log_rejection(symbol_name, "EMA_RSI: Bearish long-term trend")
        return {"signal": False, "quality": 0}
    
    ema9 = last['ema9']
    ema13 = last['ema13']
    ema21 = last['ema21']
    ema50 = last['ema50']
    
    ema_bullish_order = ema9 > ema13 > ema21 > ema50
    price_above_ema9 = last['close'] > ema9
    
    rsi = last['rsi']
    rsi_prev = prev['rsi']
    
    rsi_oversold = rsi < 40
    rsi_upward = rsi > rsi_prev
    rsi_below_70 = rsi < 70
    
    macd_hist = last['macd_hist']
    macd_hist_prev = prev['macd_hist']
    macd_upward = macd_hist > macd_hist_prev
    
    volume_ma = df['volume'].rolling(10).mean()
    volume_increasing = last['volume'] > volume_ma.iloc[-1] * 1.2
    
    dynamic_filters = check_ema_rsi_dynamic_filters_enhanced(df)
    
    quality_score = 0
    
    if ema_bullish_order: quality_score += 15
    if price_above_ema9: quality_score += 10
    if rsi_oversold: quality_score += 15
    if rsi_upward: quality_score += 10
    if rsi_below_70: quality_score += 5
    if macd_upward: quality_score += 10
    if volume_increasing: quality_score += 5
    if dynamic_filters['rsi_ok']: quality_score += 15
    if dynamic_filters['ema_ok']: quality_score += 10
    if dynamic_filters['volume_ok']: quality_score += 5
    
    entry_conditions = (
        ema_bullish_order and
        price_above_ema9 and
        rsi_oversold and
        rsi_upward and
        rsi_below_70 and
        macd_upward and
        volume_increasing and
        dynamic_filters['rsi_ok'] and
        dynamic_filters['ema_ok'] and
        dynamic_filters['volume_ok']
    )
    
    if entry_conditions:
        return {
            "signal": True,
            "quality": min(quality_score, 100),
            "details": {
                "rsi": rsi,
                "ema_spread": (ema9 - ema21) / ema21,
                "volume_ratio": last['volume'] / volume_ma.iloc[-1],
                "macd_hist": macd_hist
            }
        }
    else:
        if not dynamic_filters['rsi_ok']:
            log_rejection(symbol_name, "DYN_RSI_OOR")
        elif not dynamic_filters['ema_ok']:
            log_rejection(symbol_name, "DYN_EMA_SPREAD_LOW")
        elif not dynamic_filters['volume_ok']:
            log_rejection(symbol_name, "DYN_VOLUME_LOW")
        elif not rsi_upward:
            log_rejection(symbol_name, "RSI not upward")
        elif not macd_upward:
            log_rejection(symbol_name, "MACD momentum negative")
        
        return {"signal": False, "quality": quality_score}

def check_pullback_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> Dict:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 50:
        log_rejection(symbol_name, "Insufficient Historical Data")
        return {"signal": False, "quality": 0}
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    if not (last['ema21'] > last['ema50'] and last['ema50'] > last['ema200']):
        log_rejection(symbol_name, "Pullback: Trend is not strongly bullish")
        return {"signal": False, "quality": 0}
    
    highs = df['high'].values
    lows = df['low'].values
    
    try:
        peaks_idx = argrelextrema(highs, np.greater, order=3)[0]
        troughs_idx = argrelextrema(lows, np.less, order=3)[0]
    except Exception:
        log_rejection(symbol_name, "Error finding swing points")
        return {"signal": False, "quality": 0}
    
    if len(peaks_idx) < 1 or len(troughs_idx) < 2:
        log_rejection(symbol_name, "Insufficient swing points")
        return {"signal": False, "quality": 0}
    
    last_peak_idx = peaks_idx[-1]
    last_trough_idx = troughs_idx[-1]
    
    if last_peak_idx < last_trough_idx:
        log_rejection(symbol_name, "Invalid wave structure")
        return {"signal": False, "quality": 0}
    
    peak_price = highs[last_peak_idx]
    trough_price = lows[last_trough_idx]
    
    fib_levels = [0.236, 0.382, 0.5, 0.618]
    current_price = last['close']
    retracement = (peak_price - current_price) / (peak_price - trough_price)
    
    fib_retracement_ok = any(abs(retracement - level) < 0.05 for level in fib_levels)
    
    recent_low = df['low'].tail(3).min()
    recovery_ok = current_price > recent_low * 1.005
    
    volume_ma = df['volume'].rolling(10).mean()
    volume_increasing = last['volume'] > volume_ma.iloc[-1] * 1.3
    
    rsi = last['rsi']
    rsi_prev = prev['rsi']
    rsi_upward = rsi > rsi_prev
    rsi_not_oversold = rsi > 35
    
    dynamic_filters = check_pullback_dynamic_filters_enhanced(df, mtf_trend)
    
    quality_score = 0
    
    if fib_retracement_ok: quality_score += 20
    if recovery_ok: quality_score += 15
    if volume_increasing: quality_score += 15
    if rsi_upward: quality_score += 10
    if rsi_not_oversold: quality_score += 10
    if dynamic_filters['recovery_ok']: quality_score += 15
    if dynamic_filters['volume_ok']: quality_score += 15
    
    entry_conditions = (
        fib_retracement_ok and
        recovery_ok and
        volume_increasing and
        rsi_upward and
        rsi_not_oversold and
        dynamic_filters['recovery_ok'] and
        dynamic_filters['volume_ok']
    )
    
    if entry_conditions:
        return {
            "signal": True,
            "quality": min(quality_score, 100),
            "details": {
                "retracement": retracement,
                "fib_level": min([abs(retracement - level) for level in fib_levels]),
                "recovery_pct": (current_price - recent_low) / recent_low * 100,
                "volume_ratio": last['volume'] / volume_ma.iloc[-1],
                "rsi": rsi
            }
        }
    else:
        if not dynamic_filters['recovery_ok']:
            log_rejection(symbol_name, "DYN_RECOVERY_FAIL")
        elif not dynamic_filters['volume_ok']:
            log_rejection(symbol_name, "DYN_VOLUME_LOW")
        elif not fib_retracement_ok:
            log_rejection(symbol_name, "Fibonacci retracement not in valid range")
        elif not recovery_ok:
            log_rejection(symbol_name, "Price recovery not detected")
        
        return {"signal": False, "quality": quality_score}

def check_momentum_volatility_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> Dict:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 50:
        log_rejection(symbol_name, "Insufficient Historical Data")
        return {"signal": False, "quality": 0}
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    if not (last['ema21'] > last['ema50'] and last['ema50'] > last['ema200']):
        log_rejection(symbol_name, "Momentum: EMAs not in bullish order")
        return {"signal": False, "quality": 0}
    
    rsi = last['rsi']
    rsi_prev = prev['rsi']
    rsi_increasing = rsi > rsi_prev
    rsi_moderate = 45 < rsi < 70
    
    macd_hist = last['macd_hist']
    macd_hist_prev = prev['macd_hist']
    macd_increasing = macd_hist > macd_hist_prev
    macd_positive = macd_hist > 0
    
    atr_percent = last['atr_percent']
    atr_ma = df['atr_percent'].rolling(10).mean()
    volatility_increasing = atr_percent > atr_ma.iloc[-1]
    volatility_optimal = 0.8 < atr_percent < 2.5
    
    adx = last['adx']
    adx_increasing = adx > df['adx'].iloc[-5]
    adx_strong = adx > 18
    
    volume_ma = df['volume'].rolling(10).mean()
    volume_increasing = last['volume'] > volume_ma.iloc[-1] * 1.3
    volume_spike = last['volume'] > volume_ma.iloc[-1] * 1.5
    
    dynamic_filters = check_momentum_volatility_dynamic_filters_enhanced(df)
    
    quality_score = 0
    
    if rsi_increasing: quality_score += 10
    if rsi_moderate: quality_score += 10
    if macd_increasing: quality_score += 10
    if macd_positive: quality_score += 10
    if volatility_increasing: quality_score += 10
    if volatility_optimal: quality_score += 10
    if adx_increasing: quality_score += 10
    if adx_strong: quality_score += 10
    if volume_increasing: quality_score += 5
    if volume_spike: quality_score += 5
    if dynamic_filters['volatility_ok']: quality_score += 10
    if dynamic_filters['momentum_ok']: quality_score += 10
    if dynamic_filters['adx_ok']: quality_score += 10
    
    momentum_score = (
        (1 if rsi_increasing else 0) +
        (1 if rsi_moderate else 0) +
        (1 if macd_increasing else 0) +
        (1 if macd_positive else 0) +
        (1 if adx_increasing else 0) +
        (1 if adx_strong else 0)
    ) / 6 * 100
    
    entry_conditions = (
        rsi_increasing and
        rsi_moderate and
        macd_increasing and
        macd_positive and
        volatility_increasing and
        volatility_optimal and
        adx_increasing and
        adx_strong and
        volume_increasing and
        dynamic_filters['volatility_ok'] and
        dynamic_filters['momentum_ok'] and
        dynamic_filters['adx_ok'] and
        momentum_score > 70
    )
    
    if entry_conditions:
        return {
            "signal": True,
            "quality": min(quality_score, 100),
            "details": {
                "rsi": rsi,
                "macd_hist": macd_hist,
                "atr_percent": atr_percent,
                "adx": adx,
                "volume_ratio": last['volume'] / volume_ma.iloc[-1],
                "momentum_score": momentum_score
            }
        }
    else:
        if not dynamic_filters['volatility_ok']:
            log_rejection(symbol_name, "DYN_VOLATILITY_OOR")
        elif not dynamic_filters['momentum_ok']:
            log_rejection(symbol_name, "DYN_MOMENTUM_SCORE_LOW")
        elif not dynamic_filters['adx_ok']:
            log_rejection(symbol_name, "DYN_ADX_LOW")
        elif momentum_score <= 70:
            log_rejection(symbol_name, "Momentum score too low")
        elif not volatility_optimal:
            log_rejection(symbol_name, "Volatility not optimal")
        
        return {"signal": False, "quality": quality_score}

def check_elliott_wave_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> Dict:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 100:
        log_rejection(symbol_name, "Insufficient Historical Data")
        return {"signal": False, "quality": 0}
    
    last = df.iloc[-1]
    
    if not (last['ema50'] > last['ema200']):
        log_rejection(symbol_name, "Elliott Wave: Strongly bearish trend")
        return {"signal": False, "quality": 0}
    
    highs = df['high'].values
    lows = df['low'].values
    
    try:
        peaks_idx = argrelextrema(highs, np.greater, order=5)[0]
        troughs_idx = argrelextrema(lows, np.less, order=5)[0]
    except Exception:
        log_rejection(symbol_name, "Elliott Wave: Error in pattern detection")
        return {"signal": False, "quality": 0}
    
    if len(peaks_idx) < 3 or len(troughs_idx) < 3:
        log_rejection(symbol_name, "Elliott Wave: Insufficient swing points")
        return {"signal": False, "quality": 0}
    
    try:
        recent_peaks = peaks_idx[-3:]
        recent_troughs = troughs_idx[-3:]
        
        if (recent_troughs[0] < recent_peaks[0] < recent_troughs[1] < 
            recent_peaks[1] < recent_troughs[2] < recent_peaks[2]):
            
            wave3_high = highs[recent_peaks[1]]
            wave4_low = lows[recent_troughs[2]]
            
            wave2_high = highs[recent_peaks[0]]
            wave3_low = lows[recent_troughs[1]]
            
            wave3_height = wave3_high - wave3_low
            wave4_retracement = (wave3_high - wave4_low) / wave3_height
            
            current_price = last['close']
            wave5_start = wave4_low
            wave5_progress = (current_price - wave5_start) / wave3_height
            
            wave5_early = wave5_progress < 0.5
            
            volume_ma = df['volume'].rolling(10).mean()
            volume_increasing = last['volume'] > volume_ma.iloc[-1] * 1.2
            
            rsi = last['rsi']
            rsi_prev = df.iloc[-2]['rsi']
            rsi_increasing = rsi > rsi_prev
            rsi_moderate = 40 < rsi < 70
            
            macd_hist = last['macd_hist']
            macd_hist_prev = df.iloc[-2]['macd_hist']
            macd_increasing = macd_hist > macd_hist_prev
            
            dynamic_filters = check_elliott_wave_dynamic_filters_enhanced(df)
            
            quality_score = 0
            
            if dynamic_filters['fibonacci_ok']: quality_score += 20
            if wave5_early: quality_score += 15
            if volume_increasing: quality_score += 10
            if rsi_increasing: quality_score += 10
            if rsi_moderate: quality_score += 10
            if macd_increasing: quality_score += 10
            if dynamic_filters['volume_ok']: quality_score += 10
            if dynamic_filters['momentum_ok']: quality_score += 10
            
            entry_conditions = (
                dynamic_filters['fibonacci_ok'] and
                wave5_early and
                volume_increasing and
                rsi_increasing and
                rsi_moderate and
                macd_increasing and
                dynamic_filters['volume_ok'] and
                dynamic_filters['momentum_ok']
            )
            
            if entry_conditions:
                return {
                    "signal": True,
                    "quality": min(quality_score, 100),
                    "details": {
                        "wave4_retracement": wave4_retracement,
                        "wave5_progress": wave5_progress,
                        "volume_ratio": last['volume'] / volume_ma.iloc[-1],
                        "rsi": rsi,
                        "macd_hist": macd_hist
                    }
                }
            else:
                if not dynamic_filters['fibonacci_ok']:
                    log_rejection(symbol_name, "DYN_FIB_RETRACEMENT_OOR")
                elif not dynamic_filters['volume_ok']:
                    log_rejection(symbol_name, "DYN_VOLUME_LOW")
                elif not dynamic_filters['momentum_ok']:
                    log_rejection(symbol_name, "DYN_MOMENTUM_SCORE_LOW")
                elif not wave5_early:
                    log_rejection(symbol_name, "Wave 5 too advanced")
                
                return {"signal": False, "quality": quality_score}
        else:
            log_rejection(symbol_name, "Elliott Wave: Invalid wave pattern")
            return {"signal": False, "quality": 0}
    except Exception as e:
        logger.error(f"Error in Elliott Wave pattern detection for {symbol_name}: {e}")
        log_rejection(symbol_name, "Elliott Wave: Error in pattern detection")
        return {"signal": False, "quality": 0}

def check_range_reversal_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> Dict:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 50:
        log_rejection(symbol_name, "Insufficient Historical Data")
        return {"signal": False, "quality": 0}
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    adx = last['adx']
    if adx > 23:
        log_rejection(symbol_name, "Range Reversal: Trend too strong (ADX > 23)")
        return {"signal": False, "quality": 0}
    
    bb_upper = last['bb_upper']
    bb_lower = last['bb_lower']
    bb_middle = last['bb_middle']
    bb_width = last['bb_width']
    
    price_near_lower = (last['close'] - bb_lower) / (bb_upper - bb_lower) < 0.25
    
    rsi = last['rsi']
    rsi_prev = prev['rsi']
    rsi_oversold = rsi < 35
    rsi_upward = rsi > rsi_prev
    
    macd_hist = last['macd_hist']
    macd_hist_prev = prev['macd_hist']
    macd_upward = macd_hist > macd_hist_prev
    
    volume_ma = df['volume'].rolling(10).mean()
    volume_increasing = last['volume'] > volume_ma.iloc[-1] * 1.2
    
    ema9 = last['ema9']
    ema21 = last['ema21']
    price_above_ema9 = last['close'] > ema9
    ema9_above_ema21 = ema9 > ema21
    
    dynamic_filters = check_range_reversal_dynamic_filters_enhanced(df)
    
    quality_score = 0
    
    if adx < 23: quality_score += 15
    if price_near_lower: quality_score += 15
    if rsi_oversold: quality_score += 15
    if rsi_upward: quality_score += 10
    if macd_upward: quality_score += 10
    if volume_increasing: quality_score += 10
    if price_above_ema9: quality_score += 5
    if ema9_above_ema21: quality_score += 5
    if dynamic_filters['adx_ok']: quality_score += 15
    if dynamic_filters['rsi_ok']: quality_score += 15
    
    entry_conditions = (
        adx < 23 and
        price_near_lower and
        rsi_oversold and
        rsi_upward and
        macd_upward and
        volume_increasing and
        price_above_ema9 and
        ema9_above_ema21 and
        dynamic_filters['adx_ok'] and
        dynamic_filters['rsi_ok']
    )
    
    if entry_conditions:
        return {
            "signal": True,
            "quality": min(quality_score, 100),
            "details": {
                "adx": adx,
                "rsi": rsi,
                "bb_position": (last['close'] - bb_lower) / (bb_upper - bb_lower),
                "volume_ratio": last['volume'] / volume_ma.iloc[-1],
                "macd_hist": macd_hist
            }
        }
    else:
        if not dynamic_filters['adx_ok']:
            log_rejection(symbol_name, "Range Reversal: Trend too strong (ADX > 23)")
        elif not dynamic_filters['rsi_ok']:
            log_rejection(symbol_name, "Range Reversal: RSI not in oversold zone")
        elif not rsi_upward:
            log_rejection(symbol_name, "RSI not upward")
        elif not macd_upward:
            log_rejection(symbol_name, "MACD momentum negative")
        
        return {"signal": False, "quality": quality_score}

# --- دوال إدارة المخاطر ---
def calculate_dynamic_stop_loss_enhanced(df: pd.DataFrame, entry_price: float, strategy_name: str) -> float:
    last = df.iloc[-1]
    atr_value = last.get('atr', 0)
    atr_percent = last.get('atr_percent', 0)
    
    if atr_percent > 2.5:
        atr_multiplier = 2.8
    elif atr_percent > 1.5:
        atr_multiplier = 2.3
    elif atr_percent > 0.8:
        atr_multiplier = 1.8
    else:
        atr_multiplier = 1.3
    
    volume_ma = df['volume'].rolling(20).mean()
    volume_ratio = last['volume'] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1
    
    if volume_ratio > 2.5:
        atr_multiplier *= 0.7
    elif volume_ratio > 1.5:
        atr_multiplier *= 0.85
    elif volume_ratio < 0.5:
        atr_multiplier *= 1.15
    
    if strategy_name == "BB_Stoch_Strategy":
        recent_low = df['low'].tail(3).min()
        bb_stop = recent_low * 0.995
        atr_stop = entry_price - (atr_value * atr_multiplier)
        stop_loss = min(bb_stop, atr_stop)
    
    elif strategy_name == "MACD_EMA_Strategy":
        ema_stop = last['ema21'] * 0.995
        atr_stop = entry_price - (atr_value * atr_multiplier)
        stop_loss = min(ema_stop, atr_stop)
    
    elif strategy_name == "EMA_RSI_Strategy":
        ema_stop = last['ema21'] * 0.997
        atr_stop = entry_price - (atr_value * atr_multiplier * 0.9)
        stop_loss = min(ema_stop, atr_stop)
    
    elif strategy_name == "Pullback_Strategy":
        recent_low = df['low'].tail(5).min()
        pullback_stop = recent_low * 0.995
        atr_stop = entry_price - (atr_value * atr_multiplier * 0.75)
        stop_loss = min(pullback_stop, atr_stop)
    
    elif strategy_name == "Momentum_Volatility_Strategy":
        ema_stop = last['ema21'] * 0.993
        atr_stop = entry_price - (atr_value * atr_multiplier * 1.1)
        stop_loss = min(ema_stop, atr_stop)
    
    elif strategy_name == "Elliott_Wave_Strategy":
        lows = df['low'].values
        try:
            support_idx = argrelextrema(lows, np.less, order=5)[0]
            if len(support_idx) > 0:
                recent_support = lows[support_idx[-1]]
                wave_stop = recent_support * 0.995
                atr_stop = entry_price - (atr_value * atr_multiplier)
                stop_loss = min(wave_stop, atr_stop)
            else:
                ema_stop = last['ema50'] * 0.995
                atr_stop = entry_price - (atr_value * atr_multiplier)
                stop_loss = min(ema_stop, atr_stop)
        except Exception as e:
            logger.error(f"Error calculating stop loss for Elliott Wave: {e}")
            ema_stop = last['ema50'] * 0.995
            atr_stop = entry_price - (atr_value * atr_multiplier)
            stop_loss = min(ema_stop, atr_stop)
    
    elif strategy_name == "Range_Reversal_Strategy":
        recent_low = df['low'].tail(5).min()
        range_stop = recent_low * 0.99
        atr_stop = entry_price - (atr_value * atr_multiplier * 0.6)
        stop_loss = min(range_stop, atr_stop)
    
    else:
        stop_loss = entry_price - (atr_value * atr_multiplier)
    
    max_stop_distance = entry_price * 0.04
    if entry_price - stop_loss > max_stop_distance:
        stop_loss = entry_price - max_stop_distance
    
    min_stop_distance = entry_price * 0.01
    if entry_price - stop_loss < min_stop_distance:
        stop_loss = entry_price - min_stop_distance
    
    return stop_loss

def calculate_dynamic_take_profit_enhanced(df: pd.DataFrame, entry_price: float, stop_loss: float, strategy_name: str) -> tuple:
    risk_amount = entry_price - stop_loss
    if risk_amount <= 0:
        return (entry_price * 1.015, entry_price * 1.025)
    
    last = df.iloc[-1]
    atr_percent = last.get('atr_percent', 0)
    
    if atr_percent > 2.5:
        volatility_adjustment = 0.7
    elif atr_percent > 1.5:
        volatility_adjustment = 0.85
    elif atr_percent > 0.8:
        volatility_adjustment = 1.0
    else:
        volatility_adjustment = 1.2
    
    volume_ma = df['volume'].rolling(20).mean()
    volume_ratio = last['volume'] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1
    
    if volume_ratio > 2.5:
        volume_adjustment = 1.3
    elif volume_ratio > 1.5:
        volume_adjustment = 1.15
    elif volume_ratio < 0.5:
        volume_adjustment = 0.85
    else:
        volume_adjustment = 1.0
    
    adjustment_factor = volatility_adjustment * volume_adjustment
    
    if strategy_name == "BB_Stoch_Strategy": 
        rr1, rr2 = 1.8 * adjustment_factor, 3.0 * adjustment_factor
    elif strategy_name == "MACD_EMA_Strategy": 
        rr1, rr2 = 1.6 * adjustment_factor, 2.8 * adjustment_factor
    elif strategy_name == "EMA_RSI_Strategy": 
        rr1, rr2 = 1.7 * adjustment_factor, 3.0 * adjustment_factor
    elif strategy_name == "Pullback_Strategy": 
        rr1, rr2 = 1.8 * adjustment_factor, 3.2 * adjustment_factor
    elif strategy_name == "Momentum_Volatility_Strategy": 
        rr1, rr2 = 1.5 * adjustment_factor, 2.5 * adjustment_factor
    elif strategy_name == "Elliott_Wave_Strategy": 
        rr1, rr2 = 2.0 * adjustment_factor, 3.5 * adjustment_factor
    elif strategy_name == "Range_Reversal_Strategy":
        middle_band = df.iloc[-1].get('bb_middle', entry_price * 1.015)
        upper_band = df.iloc[-1].get('bb_upper', entry_price * 1.03)
        
        if atr_percent > 1.5:
            middle_band = entry_price + (middle_band - entry_price) * 0.8
            upper_band = entry_price + (upper_band - entry_price) * 0.8
        elif atr_percent < 0.8:
            middle_band = entry_price + (middle_band - entry_price) * 1.2
            upper_band = entry_price + (upper_band - entry_price) * 1.2
        
        return middle_band, upper_band
    else: 
        rr1, rr2 = 1.6 * adjustment_factor, 2.8 * adjustment_factor
    
    target1 = entry_price + (risk_amount * rr1)
    target2 = entry_price + (risk_amount * rr2)
    
    try:
        recent_highs = df['high'].tail(20).values
        resistance_levels = []
        
        for i in range(2, len(recent_highs)-2):
            if (recent_highs[i] > recent_highs[i-1] and 
                recent_highs[i] > recent_highs[i-2] and
                recent_highs[i] > recent_highs[i+1] and 
                recent_highs[i] > recent_highs[i+2]):
                resistance_levels.append(recent_highs[i])
        
        if resistance_levels:
            resistance_levels.sort()
            
            for resistance in resistance_levels:
                if resistance > entry_price and resistance < target1 * 1.05:
                    target1 = resistance * 0.995
                    break
            
            for resistance in resistance_levels:
                if resistance > entry_price and resistance < target2 * 1.05:
                    target2 = resistance * 0.995
                    break
    except Exception as e:
        logger.error(f"Error adjusting take profit based on resistance levels: {e}")
    
    if target1 <= entry_price:
        target1 = entry_price * 1.01
    
    if target2 <= target1:
        target2 = target1 * 1.015
    
    return target1, target2

def calculate_dynamic_position_size_enhanced(symbol: str, entry_price: float, stop_loss: float, is_real_trade: bool) -> float:
    if is_real_trade:
        with balance_lock:
            available_balance = usdt_balance
        
        try:
            df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, 1)
            if df is not None and not df.empty:
                atr_percent = df.iloc[-1].get('atr_percent', 1.0)
                volume_ma = df['volume'].rolling(20).mean()
                volume_ratio = df.iloc[-1]['volume'] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1
            else:
                atr_percent = 1.0
                volume_ratio = 1.0
        except Exception:
            atr_percent = 1.0
            volume_ratio = 1.0
        
        if atr_percent > 2.5:
            volatility_adjustment = 0.7
        elif atr_percent > 1.5:
            volatility_adjustment = 0.85
        elif atr_percent > 0.8:
            volatility_adjustment = 1.0
        else:
            volatility_adjustment = 1.15
        
        if volume_ratio > 2.5:
            volume_adjustment = 1.2
        elif volume_ratio > 1.5:
            volume_adjustment = 1.1
        elif volume_ratio < 0.5:
            volume_adjustment = 0.9
        else:
            volume_adjustment = 1.0
        
        with trade_amount_lock:
            base_amount = random.uniform(FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT)
        
        adjusted_amount = base_amount * volatility_adjustment * volume_adjustment
        
        adjusted_amount = max(FIXED_TRADE_AMOUNT_MIN_USDT * 0.8, min(adjusted_amount, FIXED_TRADE_AMOUNT_MAX_USDT * 1.2))
        
        if adjusted_amount > available_balance:
            adjusted_amount = available_balance * 0.95
        
        if adjusted_amount < FIXED_TRADE_AMOUNT_MIN_USDT * 0.5:
            adjusted_amount = FIXED_TRADE_AMOUNT_MIN_USDT * 0.5
        
        quantity = adjusted_amount / entry_price
        
        try:
            if not exchange_info_map:
                get_exchange_info_map()
            
            symbol_info = exchange_info_map.get(symbol, {})
            if symbol_info:
                lot_size_filter = None
                for filter in symbol_info.get('filters', []):
                    if filter.get('filterType') == 'LOT_SIZE':
                        lot_size_filter = filter
                        break
                
                if lot_size_filter:
                    min_qty = float(lot_size_filter.get('minQty', 0))
                    max_qty = float(lot_size_filter.get('maxQty', 0))
                    step_size = float(lot_size_filter.get('stepSize', 0))
                    
                    if step_size > 0:
                        quantity = round(quantity / step_size) * step_size
                    
                    quantity = max(min_qty, min(quantity, max_qty))
        except Exception as e:
            logger.error(f"Error adjusting quantity for {symbol}: {e}")
        
        return quantity
    else:
        paper_quantity = PAPER_TRADE_FIXED_AMOUNT_USDT / entry_price
        
        try:
            if not exchange_info_map:
                get_exchange_info_map()
            
            symbol_info = exchange_info_map.get(symbol, {})
            if symbol_info:
                lot_size_filter = None
                for filter in symbol_info.get('filters', []):
                    if filter.get('filterType') == 'LOT_SIZE':
                        lot_size_filter = filter
                        break
                
                if lot_size_filter:
                    min_qty = float(lot_size_filter.get('minQty', 0))
                    max_qty = float(lot_size_filter.get('maxQty', 0))
                    step_size = float(lot_size_filter.get('stepSize', 0))
                    
                    if step_size > 0:
                        paper_quantity = round(paper_quantity / step_size) * step_size
                    
                    paper_quantity = max(min_qty, min(paper_quantity, max_qty))
        except Exception as e:
            logger.error(f"Error adjusting paper quantity for {symbol}: {e}")
        
        return paper_quantity

def check_risk_management_rules(symbol: str, entry_price: float, stop_loss: float, is_real_trade: bool) -> bool:
    risk_percentage = ((entry_price - stop_loss) / entry_price) * 100
    MAX_RISK_PERCENTAGE = 3.0
    
    if risk_percentage > MAX_RISK_PERCENTAGE:
        log_rejection(symbol, "Risk percentage too high", {
            "risk_pct": f"{risk_percentage:.2f}%",
            "max_allowed": f"{MAX_RISK_PERCENTAGE}%"
        })
        return False
    
    with signal_cache_lock:
        open_trades_count = len(open_signals_cache)
    
    if open_trades_count >= MAX_OPEN_TRADES:
        log_rejection(symbol, "Maximum open trades reached", {
            "current": open_trades_count,
            "max_allowed": MAX_OPEN_TRADES
        })
        return False
    
    with cooldowns_lock:
        if symbol in cooldowns_by_symbol:
            cooldown_time = cooldowns_by_symbol[symbol]
            if datetime.now(timezone.utc) < cooldown_time:
                remaining_time = (cooldown_time - datetime.now(timezone.utc)).total_seconds() / 60
                log_rejection(symbol, "Cooldown period active", {
                    "remaining_minutes": f"{remaining_time:.1f}"
                })
                return False
    
    with consecutive_losses_lock:
        consecutive_losses = consecutive_losses_by_symbol.get(symbol, 0)
    
    if consecutive_losses >= 3:
        log_rejection(symbol, "Too many consecutive losses", {
            "consecutive_losses": consecutive_losses
        })
        return False
    
    if not add_correlation_filter(symbol):
        log_rejection(symbol, "Correlation Filter Failed")
        return False
    
    try:
        df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, 1)
        if df is not None and not df.empty:
            if not check_market_volatility_filter_enhanced(df, symbol):
                return False
    except Exception as e:
        logger.error(f"Error checking market volatility for {symbol}: {e}")
        return False
    
    if not add_news_filter():
        log_rejection(symbol, "News Filter Failed")
        return False
    
    if not add_liquidity_filter():
        log_rejection(symbol, "Liquidity Filter Failed")
        return False
    
    return True

# --- دوال إدارة الصفقات ---
def update_open_trades():
    if not check_db_connection() or not conn:
        return
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status = 'open';")
            open_trades = cur.fetchall()
            
            if not open_trades:
                return
            
            for trade in open_trades:
                symbol = trade['symbol']
                entry_price = trade['entry_price']
                stop_loss = trade['stop_loss']
                target1 = trade.get('target_price_1', entry_price * 1.02)
                target2 = trade.get('target_price_2', entry_price * 1.04)
                strategy_name = trade['strategy_name']
                is_real = trade['is_real_trade']
                
                with live_prices_lock:
                    current_price = live_prices.get(symbol, None)
                
                if not current_price:
                    logger.warning(f"Current price not available for {symbol}")
                    continue
                
                close_reason = None
                close_price = None
                
                if current_price <= stop_loss:
                    close_reason = "stop_loss"
                    close_price = stop_loss
                
                elif current_price >= target2:
                    close_reason = "target_2"
                    close_price = target2
                
                elif current_price >= target1:
                    close_reason = "target_1"
                    close_price = target1
                
                elif trade.get('trailing_stop') and current_price <= trade['trailing_stop']:
                    close_reason = "trailing_stop"
                    close_price = trade['trailing_stop']
                
                elif trade.get('trailing_stop_activation') and current_price >= trade['trailing_stop_activation']:
                    new_trailing_stop = current_price * (1 - TRAILING_STOP_ACTIVATION_PROFIT_PERCENT / 100)
                    
                    if not trade.get('trailing_stop') or new_trailing_stop > trade['trailing_stop']:
                        cur.execute("""
                            UPDATE signals 
                            SET trailing_stop = %s 
                            WHERE id = %s
                        """, (new_trailing_stop, trade['id']))
                        conn.commit()
                        
                        log_and_notify(
                            "info", 
                            f"تم تحديث وقف الخسارة المتحرك لـ {symbol} إلى {new_trailing_stop:.4f}", 
                            "trailing_stop_update"
                        )
                
                if close_reason and close_price:
                    profit_percentage = ((close_price - entry_price) / entry_price) * 100
                    
                    opened_at = trade['created_at']
                    duration_minutes = (datetime.now(timezone.utc) - opened_at).total_seconds() / 60
                    
                    cur.execute("""
                        UPDATE signals 
                        SET status = 'closed', 
                            closing_price = %s, 
                            closed_at = NOW(), 
                            profit_percentage = %s,
                            closing_reason = %s
                        WHERE id = %s
                    """, (close_price, profit_percentage, close_reason, trade['id']))
                    
                    if is_real:
                        with balance_lock:
                            global usdt_balance
                            usdt_balance += (usdt_balance * profit_percentage / 100)
                    
                    conn.commit()
                    
                    send_trade_close_notification_enhanced(
                        symbol, entry_price, close_price, stop_loss, target1, target2,
                        profit_percentage, strategy_name, is_real, close_reason, int(duration_minutes)
                    )
                    
                    with signal_cache_lock:
                        if symbol in open_signals_cache:
                            open_signals_cache[symbol]['status'] = 'closed'
                            open_signals_cache[symbol]['closing_price'] = close_price
                            open_signals_cache[symbol]['closed_at'] = datetime.now(timezone.utc).isoformat()
                            open_signals_cache[symbol]['profit_percentage'] = profit_percentage
                            open_signals_cache[symbol]['closing_reason'] = close_reason
                    
                    logger.info(f"Closed trade for {symbol} with {profit_percentage:.2f}% profit ({close_reason})")
                
                elif abs(current_price - entry_price) / entry_price > 0.01:
                    cur.execute("""
                        UPDATE signals 
                        SET status = 'updated' 
                        WHERE id = %s
                    """, (trade['id'],))
                    
                    conn.commit()
                    
                    with signal_cache_lock:
                        if symbol in open_signals_cache:
                            open_signals_cache[symbol]['status'] = 'updated'
                    
                    logger.info(f"Updated trade status for {symbol}")
    
    except Exception as e:
        logger.error(f"❌ [Trade Management] Error updating open trades: {e}", exc_info=True)
        if conn: conn.rollback()

# --- دوال المسح عن الإشارات ---
def process_symbol_enhanced(symbol: str, market_state: Dict) -> Optional[Dict]:
    try:
        df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
        if df is None or df.empty:
            return None
        
        df = calculate_all_features(df)
        df.name = symbol
        
        signals = []
        
        if USE_BB_STOCH_STRATEGY:
            bb_stoch_signal = check_bb_stoch_strategy_enhanced(df, market_state)
            if bb_stoch_signal['signal'] and bb_stoch_signal['quality'] >= MIN_SIGNAL_QUALITY:
                signals.append({
                    'strategy': 'BB_Stoch_Strategy',
                    'quality': bb_stoch_signal['quality'],
                    'details': bb_stoch_signal['details']
                })
        
        if USE_MACD_EMA_STRATEGY:
            macd_ema_signal = check_macd_ema_strategy_enhanced(df, market_state)
            if macd_ema_signal['signal'] and macd_ema_signal['quality'] >= MIN_SIGNAL_QUALITY:
                signals.append({
                    'strategy': 'MACD_EMA_Strategy',
                    'quality': macd_ema_signal['quality'],
                    'details': macd_ema_signal['details']
                })
        
        if USE_EMA_RSI_STRATEGY:
            ema_rsi_signal = check_ema_rsi_strategy_enhanced(df, market_state)
            if ema_rsi_signal['signal'] and ema_rsi_signal['quality'] >= MIN_SIGNAL_QUALITY:
                signals.append({
                    'strategy': 'EMA_RSI_Strategy',
                    'quality': ema_rsi_signal['quality'],
                    'details': ema_rsi_signal['details']
                })
        
        if USE_PULLBACK_STRATEGY:
            pullback_signal = check_pullback_strategy_enhanced(df, market_state)
            if pullback_signal['signal'] and pullback_signal['quality'] >= MIN_SIGNAL_QUALITY:
                signals.append({
                    'strategy': 'Pullback_Strategy',
                    'quality': pullback_signal['quality'],
                    'details': pullback_signal['details']
                })
        
        if USE_MOMENTUM_VOLATILITY_STRATEGY:
            momentum_signal = check_momentum_volatility_strategy_enhanced(df, market_state)
            if momentum_signal['signal'] and momentum_signal['quality'] >= MIN_SIGNAL_QUALITY:
                signals.append({
                    'strategy': 'Momentum_Volatility_Strategy',
                    'quality': momentum_signal['quality'],
                    'details': momentum_signal['details']
                })
        
        if USE_ELLIOTT_WAVE_STRATEGY:
            elliott_signal = check_elliott_wave_strategy_enhanced(df, market_state)
            if elliott_signal['signal'] and elliott_signal['quality'] >= MIN_SIGNAL_QUALITY:
                signals.append({
                    'strategy': 'Elliott_Wave_Strategy',
                    'quality': elliott_signal['quality'],
                    'details': elliott_signal['details']
                })
        
        if USE_RANGE_REVERSAL_STRATEGY:
            range_signal = check_range_reversal_strategy_enhanced(df, market_state)
            if range_signal['signal'] and range_signal['quality'] >= MIN_SIGNAL_QUALITY:
                signals.append({
                    'strategy': 'Range_Reversal_Strategy',
                    'quality': range_signal['quality'],
                    'details': range_signal['details']
                })
        
        if signals:
            best_signal = max(signals, key=lambda x: x['quality'])
            
            entry_price = df.iloc[-1]['close']
            stop_loss = calculate_dynamic_stop_loss_enhanced(df, entry_price, best_signal['strategy'])
            target1, target2 = calculate_dynamic_take_profit_enhanced(df, entry_price, stop_loss, best_signal['strategy'])
            
            return {
                'symbol': symbol,
                'strategy': best_signal['strategy'],
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target1': target1,
                'target2': target2,
                'quality': best_signal['quality'],
                'atr_percent': df.iloc[-1].get('atr_percent', 1.0),
                'details': best_signal['details']
            }
        
        return None
    
    except Exception as e:
        logger.error(f"Error processing symbol {symbol}: {e}", exc_info=True)
        return None

def scan_for_signals_enhanced():
    logger.info("Starting enhanced signal scan...")
    
    if not check_db_connection() or not conn:
        logger.error("Database connection not available for signal scan")
        return
    
    try:
        market_state = get_market_state()
        
        if not validated_symbols_to_scan:
            global validated_symbols_to_scan
            validated_symbols_to_scan = get_validated_symbols()
        
        if not validated_symbols_to_scan:
            logger.error("No valid symbols to scan")
            return
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {
                executor.submit(process_symbol_enhanced, symbol, market_state): symbol 
                for symbol in validated_symbols_to_scan[:50]
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    signal_data = future.result()
                    if signal_data and signal_data.get('signal'):
                        process_signal_enhanced(signal_data)
                except Exception as e:
                    logger.error(f"Error processing symbol {symbol}: {e}")
        
        logger.info("Enhanced signal scan completed")
    
    except Exception as e:
        logger.error(f"❌ [Signal Scan] Error during scan: {e}", exc_info=True)

def process_signal_enhanced(signal_data: Dict):
    try:
        symbol = signal_data['symbol']
        strategy_name = signal_data['strategy']
        entry_price = signal_data['entry_price']
        stop_loss = signal_data['stop_loss']
        target1 = signal_data['target1']
        target2 = signal_data['target2']
        quality_score = signal_data['quality']
        atr_percent = signal_data.get('atr_percent', 1.0)
        signal_details = signal_data.get('details', {})
        
        with signal_cache_lock:
            if symbol in open_signals_cache:
                logger.info(f"Signal for {symbol} already exists, skipping")
                return
        
        with trading_mode_lock:
            is_real = not paper_trading_mode
        
        if not check_risk_management_rules(symbol, entry_price, stop_loss, is_real):
            return
        
        quantity = calculate_dynamic_position_size_enhanced(symbol, entry_price, stop_loss, is_real)
        notional_value = quantity * entry_price
        
        MIN_NOTIONAL_VALUE = 5.0
        if notional_value < MIN_NOTIONAL_VALUE:
            log_rejection(symbol, "MinNotional Filter Failed", {
                "notional_value": f"{notional_value:.2f}",
                "min_required": f"{MIN_NOTIONAL_VALUE}"
            })
            return
        
        trailing_stop_activation = None
        if quality_score >= 85:
            trailing_stop_activation = entry_price * (1 + TRAILING_STOP_ACTIVATION_PROFIT_PERCENT / 100)
        
        if not check_db_connection() or not conn:
            logger.error("Database connection not available for saving signal")
            return
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO signals (
                        symbol, entry_price, stop_loss, target_price_1, target_price_2, 
                        status, strategy_name, signal_details, is_real_trade, 
                        quantity, trailing_stop_activation
                    ) VALUES (%s, %s, %s, %s, %s, 'open', %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    symbol, entry_price, stop_loss, target1, target2,
                    strategy_name, json.dumps(signal_details), is_real,
                    quantity, trailing_stop_activation
                ))
                
                signal_id = cur.fetchone()['id']
                conn.commit()
                
                with signal_cache_lock:
                    open_signals_cache[symbol] = {
                        'id': signal_id,
                        'symbol': symbol,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'target_price_1': target1,
                        'target_price_2': target2,
                        'status': 'open',
                        'strategy_name': strategy_name,
                        'signal_details': signal_details,
                        'is_real_trade': is_real,
                        'quantity': quantity,
                        'created_at': datetime.now(timezone.utc).isoformat(),
                        'trailing_stop_activation': trailing_stop_activation
                    }
                
                send_trade_open_notification_enhanced(
                    symbol, strategy_name, entry_price, stop_loss, target1, target2,
                    quantity, is_real, quality_score, atr_percent, notional_value, signal_details
                )
                
                logger.info(f"Opened new {'real' if is_real else 'paper'} trade for {symbol} with {strategy_name}")
                
                broadcast({
                    "type": "new_signal",
                    "payload": open_signals_cache[symbol]
                })
                
        except Exception as e:
            logger.error(f"❌ [Signal Processing] Error saving signal for {symbol}: {e}")
            if conn: conn.rollback()
    
    except Exception as e:
        logger.error(f"❌ [Signal Processing] Error processing signal: {e}", exc_info=True)

def get_market_state() -> Dict:
    try:
        btc_data = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, 2)
        if btc_data is None or btc_data.empty:
            return current_market_state
        
        btc_data = calculate_all_features(btc_data)
        
        trend_details = {}
        
        for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
            try:
                if tf == '5m':
                    df = btc_data
                else:
                    df = fetch_historical_data(BTC_SYMBOL, tf, 2)
                    if df is None or df.empty:
                        continue
                    df = calculate_all_features(df)
                
                last = df.iloc[-1]
                
                if last['ema50'] > last['ema200']:
                    trend = 'bullish'
                else:
                    trend = 'bearish'
                
                trend_details[tf] = {
                    'trend': trend,
                    'adx': last.get('adx', 0),
                    'rsi': last.get('rsi', 50)
                }
            except Exception as e:
                logger.error(f"Error getting market state for {tf}: {e}")
        
        with market_state_lock:
            current_market_state['trend_details_by_tf'] = trend_details
        
        return current_market_state
    
    except Exception as e:
        logger.error(f"Error getting market state: {e}")
        return current_market_state

# --- دوال واجهة الويب ---
@app.route('/api/settings', methods=['GET'])
def get_settings():
    try:
        with trading_mode_lock:
            current_paper_mode = paper_trading_mode
        
        with trade_amount_lock:
            current_min_amount = FIXED_TRADE_AMOUNT_MIN_USDT
            current_max_amount = FIXED_TRADE_AMOUNT_MAX_USDT
        
        with min_quality_lock:
            current_min_quality = MIN_SIGNAL_QUALITY
        
        settings = {
            "paper_trading_mode": current_paper_mode,
            "fixed_trade_amount_min": current_min_amount,
            "fixed_trade_amount_max": current_max_amount,
            "max_open_trades": MAX_OPEN_TRADES,
            "min_signal_quality": current_min_quality,
            "trailing_stop_activation_percent": TRAILING_STOP_ACTIVATION_PROFIT_PERCENT,
            "strategies": {
                "USE_BB_STOCH_STRATEGY": USE_BB_STOCH_STRATEGY,
                "USE_MACD_EMA_STRATEGY": USE_MACD_EMA_STRATEGY,
                "USE_EMA_RSI_STRATEGY": USE_EMA_RSI_STRATEGY,
                "USE_PULLBACK_STRATEGY": USE_PULLBACK_STRATEGY,
                "USE_MOMENTUM_VOLATILITY_STRATEGY": USE_MOMENTUM_VOLATILITY_STRATEGY,
                "USE_ELLIOTT_WAVE_STRATEGY": USE_ELLIOTT_WAVE_STRATEGY,
                "USE_RANGE_REVERSAL_STRATEGY": USE_RANGE_REVERSAL_STRATEGY
            }
        }
        
        return jsonify({"success": True, "settings": settings})
    
    except Exception as e:
        logger.error(f"Error getting settings: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/settings', methods=['POST'])
def update_settings():
    try:
        data = request.json
        
        if 'paper_trading_mode' in data:
            with trading_mode_lock:
                global paper_trading_mode
                paper_trading_mode = bool(data['paper_trading_mode'])
        
        if 'fixed_trade_amount_min' in data and 'fixed_trade_amount_max' in data:
            with trade_amount_lock:
                global FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT
                FIXED_TRADE_AMOUNT_MIN_USDT = float(data['fixed_trade_amount_min'])
                FIXED_TRADE_AMOUNT_MAX_USDT = float(data['fixed_trade_amount_max'])
                
                if FIXED_TRADE_AMOUNT_MIN_USDT > FIXED_TRADE_AMOUNT_MAX_USDT:
                    FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT = FIXED_TRADE_AMOUNT_MAX_USDT, FIXED_TRADE_AMOUNT_MIN_USDT
        
        if 'max_open_trades' in data:
            global MAX_OPEN_TRADES
            MAX_OPEN_TRADES = int(data['max_open_trades'])
        
        if 'min_signal_quality' in data:
            with min_quality_lock:
                global MIN_SIGNAL_QUALITY
                MIN_SIGNAL_QUALITY = int(data['min_signal_quality'])
        
        if 'trailing_stop_activation_percent' in data:
            global TRAILING_STOP_ACTIVATION_PROFIT_PERCENT
            TRAILING_STOP_ACTIVATION_PROFIT_PERCENT = float(data['trailing_stop_activation_percent'])
        
        if 'strategies' in data:
            strategies = data['strategies']
            global USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY
            global USE_PULLBACK_STRATEGY, USE_MOMENTUM_VOLATILITY_STRATEGY
            global USE_ELLIOTT_WAVE_STRATEGY, USE_RANGE_REVERSAL_STRATEGY
            
            USE_BB_STOCH_STRATEGY = bool(strategies.get('USE_BB_STOCH_STRATEGY', True))
            USE_MACD_EMA_STRATEGY = bool(strategies.get('USE_MACD_EMA_STRATEGY', True))
            USE_EMA_RSI_STRATEGY = bool(strategies.get('USE_EMA_RSI_STRATEGY', True))
            USE_PULLBACK_STRATEGY = bool(strategies.get('USE_PULLBACK_STRATEGY', True))
            USE_MOMENTUM_VOLATILITY_STRATEGY = bool(strategies.get('USE_MOMENTUM_VOLATILITY_STRATEGY', True))
            USE_ELLIOTT_WAVE_STRATEGY = bool(strategies.get('USE_ELLIOTT_WAVE_STRATEGY', True))
            USE_RANGE_REVERSAL_STRATEGY = bool(strategies.get('USE_RANGE_REVERSAL_STRATEGY', True))
        
        save_settings_to_redis()
        
        log_and_notify("info", "تم تحديث إعدادات البوت بنجاح", "settings_update")
        
        return jsonify({"success": True})
    
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/trading/toggle', methods=['POST'])
def toggle_trading():
    try:
        data = request.json
        enabled = bool(data.get('enabled', False))
        
        with trading_status_lock:
            global is_trading_enabled
            is_trading_enabled = enabled
        
        status = "مفعل" if enabled else "معطل"
        log_and_notify("info", f"تم {status} التداول", "trading_toggle")
        
        return jsonify({"success": True, "enabled": enabled})
    
    except Exception as e:
        logger.error(f"Error toggling trading: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/performance', methods=['GET'])
def get_performance():
    try:
        if not check_db_connection() or not conn:
            return jsonify({"success": False, "error": "Database connection not available"}), 500
        
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit_percentage > 0 THEN 1 ELSE 0 END) as winning_trades,
                    AVG(profit_percentage) as avg_profit,
                    SUM(profit_percentage) as total_profit,
                    MAX(profit_percentage) as best_trade,
                    MIN(profit_percentage) as worst_trade
                FROM signals
                WHERE status = 'closed'
            """)
            
            general_stats = cur.fetchone()
            
            today = datetime.now(timezone.utc).date()
            cur.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit_percentage > 0 THEN 1 ELSE 0 END) as winning_trades,
                    AVG(profit_percentage) as avg_profit,
                    SUM(profit_percentage) as total_profit
                FROM signals
                WHERE status = 'closed' AND closed_at::date = %s
            """, (today,))
            
            today_stats = cur.fetchone()
            
            week_start = today - timedelta(days=7)
            cur.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit_percentage > 0 THEN 1 ELSE 0 END) as winning_trades,
                    AVG(profit_percentage) as avg_profit,
                    SUM(profit_percentage) as total_profit
                FROM signals
                WHERE status = 'closed' AND closed_at::date >= %s
            """, (week_start,))
            
            week_stats = cur.fetchone()
            
            cur.execute("""
                SELECT strategy_name, 
                       COUNT(*) as trade_count,
                       AVG(profit_percentage) as avg_profit,
                       SUM(CASE WHEN profit_percentage > 0 THEN 1 ELSE 0 END) as wins
                FROM signals
                WHERE status = 'closed'
                GROUP BY strategy_name
                ORDER BY avg_profit DESC
            """)
            
            strategy_stats = cur.fetchall()
            
            cur.execute("""
                SELECT DATE_TRUNC('month', closed_at)::date as month,
                       COUNT(*) as trade_count,
                       AVG(profit_percentage) as avg_profit,
                       SUM(profit_percentage) as total_profit
                FROM signals
                WHERE status = 'closed'
                GROUP BY month
                ORDER BY month DESC
                LIMIT 12
            """)
            
            monthly_stats = cur.fetchall()
            
            with balance_lock:
                current_balance = usdt_balance
            
            return jsonify({
                "success": True,
                "performance": {
                    "general": general_stats,
                    "today": today_stats,
                    "week": week_stats,
                    "strategies": strategy_stats,
                    "monthly": monthly_stats,
                    "current_balance": current_balance
                }
            })
    
    except Exception as e:
        logger.error(f"Error getting performance data: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/')
def dashboard():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>CryptoBot V34.0.5 Enhanced Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { text-align: center; margin-bottom: 30px; }
                .card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .stats { display: flex; justify-content: space-around; text-align: center; }
                .stat-item { padding: 10px; }
                .stat-value { font-size: 24px; font-weight: bold; color: #333; }
                .stat-label { font-size: 14px; color: #666; }
                .trade-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
                .trade-table th, .trade-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                .trade-table th { background-color: #f2f2f2; }
                .profit { color: green; }
                .loss { color: red; }
                .btn { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
                .btn-danger { background-color: #f44336; }
                .settings-panel { margin-top: 20px; }
                .form-group { margin-bottom: 15px; }
                .form-group label { display: block; margin-bottom: 5px; }
                .form-group input, .form-group select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
                .strategy-toggle { display: flex; align-items: center; margin-bottom: 10px; }
                .strategy-toggle input { margin-right: 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>CryptoBot V34.0.5 Enhanced Dashboard</h1>
                    <p>نظام تداول ذكي للعملات الرقمية</p>
                </div>
                
                <div class="card">
                    <h2>حالة التداول</h2>
                    <div class="stats">
                        <div class="stat-item">
                            <div class="stat-value" id="balance">0</div>
                            <div class="stat-label">الرصيد (USDT)</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="open-trades">0</div>
                            <div class="stat-label">الصفقات المفتوحة</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="total-trades">0</div>
                            <div class="stat-label">إجمالي الصفقات</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" id="win-rate">0%</div>
                            <div class="stat-label">نسبة الربح</div>
                        </div>
                    </div>
                    <div style="text-align: center; margin-top: 20px;">
                        <button class="btn" id="toggle-trading">تفعيل التداول</button>
                        <button class="btn btn-danger" id="emergency-stop">إيقاف طوارئ</button>
                    </div>
                </div>
                
                <div class="grid">
                    <div class="card">
                        <h3>الصفقات المفتوحة</h3>
                        <div id="open-trades-list">جاري التحميل...</div>
                    </div>
                    
                    <div class="card">
                        <h3>آخر الإشعارات</h3>
                        <div id="notifications-list">جاري التحميل...</div>
                    </div>
                </div>
                
                <div class="card">
                    <h3>أداء الاستراتيجيات</h3>
                    <canvas id="strategy-chart" width="400" height="200"></canvas>
                </div>
                
                <div class="card">
                    <h3>الإعدادات</h3>
                    <div class="settings-panel">
                        <div class="form-group">
                            <label>نمط التداول</label>
                            <select id="trading-mode">
                                <option value="true">ورقي</option>
                                <option value="false">حقيقي</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>الحد الأدنى لحجم الصفقة (USDT)</label>
                            <input type="number" id="min-amount" step="0.1">
                        </div>
                        
                        <div class="form-group">
                            <label>الحد الأقصى لحجم الصفقة (USDT)</label>
                            <input type="number" id="max-amount" step="0.1">
                        </div>
                        
                        <div class="form-group">
                            <label>الحد الأقصى للصفقات المفتوحة</label>
                            <input type="number" id="max-trades">
                        </div>
                        
                        <div class="form-group">
                            <label>جودة الإشارة الدنيا</label>
                            <input type="number" id="min-quality" min="0" max="100">
                        </div>
                        
                        <h4>الاستراتيجيات</h4>
                        <div class="strategy-toggle">
                            <input type="checkbox" id="strategy-bb-stoch" checked>
                            <label for="strategy-bb-stoch">BB+Stoch</label>
                        </div>
                        <div class="strategy-toggle">
                            <input type="checkbox" id="strategy-macd-ema" checked>
                            <label for="strategy-macd-ema">MACD+EMA</label>
                        </div>
                        <div class="strategy-toggle">
                            <input type="checkbox" id="strategy-ema-rsi" checked>
                            <label for="strategy-ema-rsi">EMA+RSI</label>
                        </div>
                        <div class="strategy-toggle">
                            <input type="checkbox" id="strategy-pullback" checked>
                            <label for="strategy-pullback">Pullback</label>
                        </div>
                        <div class="strategy-toggle">
                            <input type="checkbox" id="strategy-momentum" checked>
                            <label for="strategy-momentum">Momentum</label>
                        </div>
                        <div class="strategy-toggle">
                            <input type="checkbox" id="strategy-elliott" checked>
                            <label for="strategy-elliott">Elliott Wave</label>
                        </div>
                        <div class="strategy-toggle">
                            <input type="checkbox" id="strategy-range" checked>
                            <label for="strategy-range">Range Reversal</label>
                        </div>
                        
                        <button class="btn" id="save-settings">حفظ الإعدادات</button>
                    </div>
                </div>
            </div>
            
            <script>
                // WebSocket connection
                const ws = new WebSocket('ws://' + window.location.host + '/ws');
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'price_update') {
                        // Update prices if needed
                    } else if (data.type === 'new_signal') {
                        updateOpenTrades();
                    } else if (data.type === 'new_notification') {
                        updateNotifications();
                    }
                };
                
                // Fetch initial data
                fetch('/api/settings')
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            document.getElementById('trading-mode').value = data.settings.paper_trading_mode.toString();
                            document.getElementById('min-amount').value = data.settings.fixed_trade_amount_min;
                            document.getElementById('max-amount').value = data.settings.fixed_trade_amount_max;
                            document.getElementById('max-trades').value = data.settings.max_open_trades;
                            document.getElementById('min-quality').value = data.settings.min_signal_quality;
                            
                            document.getElementById('strategy-bb-stoch').checked = data.settings.strategies.USE_BB_STOCH_STRATEGY;
                            document.getElementById('strategy-macd-ema').checked = data.settings.strategies.USE_MACD_EMA_STRATEGY;
                            document.getElementById('strategy-ema-rsi').checked = data.settings.strategies.USE_EMA_RSI_STRATEGY;
                            document.getElementById('strategy-pullback').checked = data.settings.strategies.USE_PULLBACK_STRATEGY;
                            document.getElementById('strategy-momentum').checked = data.settings.strategies.USE_MOMENTUM_VOLATILITY_STRATEGY;
                            document.getElementById('strategy-elliott').checked = data.settings.strategies.USE_ELLIOTT_WAVE_STRATEGY;
                            document.getElementById('strategy-range').checked = data.settings.strategies.USE_RANGE_REVERSAL_STRATEGY;
                        }
                    });
                
                fetch('/api/performance')
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            document.getElementById('balance').textContent = data.performance.current_balance.toFixed(2);
                            
                            if (data.performance.general) {
                                document.getElementById('total-trades').textContent = data.performance.general.total_trades;
                                const winRate = data.performance.general.total_trades > 0 
                                    ? (data.performance.general.winning_trades / data.performance.general.total_trades * 100).toFixed(1)
                                    : '0';
                                document.getElementById('win-rate').textContent = winRate + '%';
                            }
                            
                            updateOpenTrades();
                            updateNotifications();
                            updateStrategyChart(data.performance.strategies);
                        }
                    });
                
                function updateOpenTrades() {
                    fetch('/api/open-trades')
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                document.getElementById('open-trades').textContent = data.trades.length;
                                
                                let html = '<table class="trade-table"><tr><th>العملة</th><th>الاستراتيجية</th><th>سعر الدخول</th><th>الربح/الخسارة</th></tr>';
                                
                                data.trades.forEach(trade => {
                                    const profitClass = trade.profit_percentage >= 0 ? 'profit' : 'loss';
                                    html += `<tr>
                                        <td>${trade.symbol}</td>
                                        <td>${trade.strategy_name}</td>
                                        <td>${trade.entry_price.toFixed(4)}</td>
                                        <td class="${profitClass}">${trade.profit_percentage ? trade.profit_percentage.toFixed(2) + '%' : '-'}</td>
                                    </tr>`;
                                });
                                
                                html += '</table>';
                                document.getElementById('open-trades-list').innerHTML = html || 'لا توجد صفقات مفتوحة';
                            }
                        });
                }
                
                function updateNotifications() {
                    fetch('/api/notifications')
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                let html = '<ul>';
                                
                                data.notifications.forEach(notification => {
                                    html += `<li>${notification.timestamp}: ${notification.message}</li>`;
                                });
                                
                                html += '</ul>';
                                document.getElementById('notifications-list').innerHTML = html || 'لا توجد إشعارات';
                            }
                        });
                }
                
                function updateStrategyChart(strategies) {
                    const ctx = document.getElementById('strategy-chart').getContext('2d');
                    
                    const labels = strategies.map(s => s.strategy_name);
                    const data = strategies.map(s => s.avg_profit);
                    
                    new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: labels,
                            datasets: [{
                                label: 'متوسط الربح (%)',
                                data: data,
                                backgroundColor: data.map(d => d >= 0 ? 'rgba(75, 192, 192, 0.6)' : 'rgba(255, 99, 132, 0.6)'),
                                borderColor: data.map(d => d >= 0 ? 'rgba(75, 192, 192, 1)' : 'rgba(255, 99, 132, 1)'),
                                borderWidth: 1
                            }]
                        },
                        options: {
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            }
                        }
                    });
                }
                
                // Event listeners
                document.getElementById('toggle-trading').addEventListener('click', function() {
                    const enabled = this.textContent === 'تفعيل التداول';
                    
                    fetch('/api/trading/toggle', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ enabled: enabled })
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            this.textContent = enabled ? 'تعطيل التداول' : 'تفعيل التداول';
                        }
                    });
                });
                
                document.getElementById('emergency-stop').addEventListener('click', function() {
                    if (confirm('هل أنت متأكد من إيقاف جميع الصفقات؟')) {
                        fetch('/api/emergency-stop', {
                            method: 'POST'
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                alert('تم إيقاف جميع الصفقات بنجاح');
                                updateOpenTrades();
                            }
                        });
                    }
                });
                
                document.getElementById('save-settings').addEventListener('click', function() {
                    const settings = {
                        paper_trading_mode: document.getElementById('trading-mode').value === 'true',
                        fixed_trade_amount_min: parseFloat(document.getElementById('min-amount').value),
                        fixed_trade_amount_max: parseFloat(document.getElementById('max-amount').value),
                        max_open_trades: parseInt(document.getElementById('max-trades').value),
                        min_signal_quality: parseInt(document.getElementById('min-quality').value),
                        strategies: {
                            USE_BB_STOCH_STRATEGY: document.getElementById('strategy-bb-stoch').checked,
                            USE_MACD_EMA_STRATEGY: document.getElementById('strategy-macd-ema').checked,
                            USE_EMA_RSI_STRATEGY: document.getElementById('strategy-ema-rsi').checked,
                            USE_PULLBACK_STRATEGY: document.getElementById('strategy-pullback').checked,
                            USE_MOMENTUM_VOLATILITY_STRATEGY: document.getElementById('strategy-momentum').checked,
                            USE_ELLIOTT_WAVE_STRATEGY: document.getElementById('strategy-elliott').checked,
                            USE_RANGE_REVERSAL_STRATEGY: document.getElementById('strategy-range').checked
                        }
                    };
                    
                    fetch('/api/settings', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(settings)
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('تم حفظ الإعدادات بنجاح');
                        } else {
                            alert('خطأ في حفظ الإعدادات: ' + data.error);
                        }
                    });
                });
            </script>
        </body>
        </html>
    ''')

@app.route('/api/open-trades')
def get_open_trades():
    try:
        with signal_cache_lock:
            trades = list(open_signals_cache.values())
        
        return jsonify({"success": True, "trades": trades})
    except Exception as e:
        logger.error(f"Error getting open trades: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/notifications')
def get_notifications():
    try:
        with notifications_lock:
            notifications = list(notifications_cache)
        
        return jsonify({"success": True, "notifications": notifications})
    except Exception as e:
        logger.error(f"Error getting notifications: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/emergency-stop', methods=['POST'])
def emergency_stop():
    try:
        with trading_status_lock:
            global is_trading_enabled
            is_trading_enabled = False
        
        if not check_db_connection() or not conn:
            return jsonify({"success": False, "error": "Database connection not available"}), 500
        
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE signals SET status = 'closed', closing_reason = 'manual' WHERE status = 'open';")
                conn.commit()
            
            with signal_cache_lock:
                for symbol in open_signals_cache:
                    open_signals_cache[symbol]['status'] = 'closed'
                    open_signals_cache[symbol]['closing_reason'] = 'manual'
            
            log_and_notify("warning", "تم إيقاف جميع الصفقات يدويًا (إيقاف طوارئ)", "emergency_stop")
            
            return jsonify({"success": True})
        except Exception as e:
            logger.error(f"Error in emergency stop: {e}")
            if conn: conn.rollback()
            return jsonify({"success": False, "error": str(e)}), 500
    
    except Exception as e:
        logger.error(f"Error in emergency stop: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@sock.route('/ws')
def websocket_connection(ws):
    with ws_clients_lock:
        ws_clients.append(ws)
    
    try:
        while True:
            data = ws.receive()
            # Handle incoming WebSocket messages if needed
    except Exception:
        with ws_clients_lock:
            if ws in ws_clients:
                ws_clients.remove(ws)

# --- دوال بدء التشغيل ---
def start_websocket():
    global ws_manager
    ws_manager = connection_manager.get_ws_manager()
    ws_manager.start_ticker_socket(callback=handle_socket_message)
    logger.info("✅ [WebSocket] Subscribed to ticker stream.")

def update_account_balance():
    global usdt_balance
    try:
        with trading_mode_lock:
            if not paper_trading_mode and client:
                account_info = client.get_account()
                for balance in account_info['balances']:
                    if balance['asset'] == 'USDT':
                        usdt_balance = float(balance['free'])
                        break
                
                logger.info(f"✅ [Balance] Updated USDT balance: {usdt_balance}")
            else:
                if usdt_balance == 0:
                    usdt_balance = 1000.0
                    logger.info(f"✅ [Balance] Set paper trading balance: {usdt_balance}")
    except Exception as e:
        logger.error(f"❌ [Balance] Error updating balance: {e}")

def start_signal_scanner():
    def scanner():
        while True:
            try:
                if is_trading_enabled:
                    scan_for_signals_enhanced()
                time.sleep(60)
            except Exception as e:
                logger.error(f"❌ [Signal Scanner] Error: {e}", exc_info=True)
                time.sleep(60)
    
    scanner_thread = Thread(target=scanner, daemon=True)
    scanner_thread.start()
    logger.info("✅ [Signal Scanner] Started signal scanner thread.")

def start_trade_manager():
    def manager():
        while True:
            try:
                update_open_trades()
                time.sleep(30)
            except Exception as e:
                logger.error(f"❌ [Trade Manager] Error: {e}", exc_info=True)
                time.sleep(30)
    
    manager_thread = Thread(target=manager, daemon=True)
    manager_thread.start()
    logger.info("✅ [Trade Manager] Started trade manager thread.")

def initialize_system():
    logger.info("🚀 Starting CryptoBot V34.0.5 (Enhanced)...")
    
    try:
        logger.info("[System] Initializing database connection...")
        init_db()
        
        logger.info("[System] Initializing Redis connection...")
        init_redis()
        
        logger.info("[System] Loading settings from Redis...")
        load_settings_from_redis()
        
        logger.info("[System] Initializing Binance client...")
        global client
        client = connection_manager.get_binance_client()
        
        logger.info("[System] Fetching exchange information...")
        get_exchange_info_map()
        
        logger.info("[System] Validating symbols...")
        global validated_symbols_to_scan
        validated_symbols_to_scan = get_validated_symbols()
        
        logger.info("[System] Loading open signals...")
        load_open_signals_to_cache()
        
        logger.info("[System] Loading notifications...")
        load_notifications_to_cache()
        
        logger.info("[System] Fetching account balance...")
        update_account_balance()
        
        logger.info("[System] Starting WebSocket manager...")
        start_websocket()
        
        logger.info("[System] Starting periodic reports...")
        start_periodic_reports()
        
        logger.info("[System] Starting signal scanner...")
        start_signal_scanner()
        
        logger.info("[System] Starting trade manager...")
        start_trade_manager()
        
        logger.info("✅ [System] CryptoBot V34.0.5 (Enhanced) started successfully!")
        
        log_and_notify("info", "تم بدء تشغيل البوت بنجاح", "system_start")
        
    except Exception as e:
        logger.critical(f"❌ [System] Failed to initialize system: {e}", exc_info=True)
        exit(1)

if __name__ == '__main__':
    try:
        initialize_system()
        logger.info("[System] Starting Flask application...")
        app.run(host='0.0.0.0', port=5000, threaded=True)
    
    except KeyboardInterrupt:
        logger.info("🛑 [System] Received keyboard interrupt, shutting down...")
        try:
            if ws_manager:
                ws_manager.stop()
            
            if conn:
                conn.close()
            
            if redis_client:
                redis_client.close()
            
            logger.info("✅ [System] Shutdown completed successfully")
        except Exception as e:
            logger.error(f"❌ [System] Error during shutdown: {e}")
    
    except Exception as e:
        logger.critical(f"❌ [System] Fatal error: {e}", exc_info=True)
        exit(1)