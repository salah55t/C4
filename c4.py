# ملف crypto_bot_enhanced.py - (V35.0.0 مُحسّن)
# --- التحسينات الرئيسية:
# 1. إصلاح نظام WebSocket مع إعادة اتصال تلقائية ذكية
# 2. إضافة 5 استراتيجيات تداول متقدمة تغطي جميع الأوضاع السوقية
# 3. نظام ذكي لاختيار الاستراتيجية بناءً على حالة السوق
# 4. تحسين كفاءة الذاكرة والأداء
# 5. إضافة مرشحات سوقية متقدمة

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
from threading import Thread, Lock, Timer
from datetime import datetime, timezone, timedelta
from decouple import config
from typing import List, Dict, Optional, Any
from collections import deque
import warnings
from scipy.signal import argrelextrema

# --- إعدادات التجاهل واللوجر ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ضبط دقة النوع Decimal
getcontext().prec = 18

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v35_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBot_MultiStrategy_v35')

# --- المتغيرات العامة والإعدادات ---
API_KEY: str = config('BINANCE_API_KEY')
API_SECRET: str = config('BINANCE_API_SECRET')
DB_URL: str = config('DATABASE_URL')
REDIS_URL: str = config('REDIS_URL', default='redis://localhost:6379/0')
TELEGRAM_BOT_TOKEN: str = config('TELEGRAM_BOT_TOKEN', default='')
TELEGRAM_CHAT_ID: str = config('TELEGRAM_CHAT_ID', default='')

# --- متغيرات الحالة ---
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

# --- سجل الاستراتيجيات ---
STRATEGY_NAMES = {
    "RSI_Enhanced_Strategy": "RSI مُحسّن (الهابط/العكسي)",
    "EMA_Macd_Trend_Strategy": "EMA Cross + MACD (الاتجاه الصاعد)",
    "Bollinger_Stochastic_Range_Strategy": "Bollinger + Stochastic (الجانبي)",
    "ZigZag_Breakout_Strategy": "ZigZag Breakout (الاندفاعات)",
    "VWAP_Mean_Reversion_Strategy": "VWAP Mean Reversion (الانعكاس)"
}

strategy_registry = {}
strategies_lock = Lock()

# --- إعدادات عامة ---
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
current_market_state: Dict[str, Any] = {"trend_details_by_tf": {}, "overall_trend": "sideways"}
market_state_lock = Lock()

# --- متغيرات إعدادات الاستراتيجية ---
ENABLE_EMA_FILTER: bool = True
ENABLE_MACD_CONFIRMATION: bool = True
ENABLE_MFI_FILTER: bool = True
ENABLE_CANDLESTICK_PATTERNS: bool = True
REQUIRED_CONFIRMATIONS: int = 3
strategy_config_lock = Lock()

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
    "Liquidity Filter Failed": "فلتر السيولة: تجنب التداول في أوقات السيولة المنخفظة",
    "Correlation Filter Failed": "فلتر الارتباط: توجد صفقة مفتوحة على عملة مرتبطة",
    "Market Trend Mismatch": "فلتر الاتجاه: الاستراتيجية لا تتوافق مع الاتجاه الحالي",

    # Strategy Specific Rejections
    "RSI: No bullish crossover": "RSI: لم يحدث اختراق صعودي لمستوى 30",
    "EMA: No bullish crossover": "EMA: لم يحدث تقاطع صعودي",
    "MACD: No bullish momentum": "MACD: لا يوجد زخم صعودي",
    "MFI: Low volume pressure": "MFI: ضغط حجم ضعيف",
    "Candlestick: No bullish pattern": "الشموع: لا يوجد نمط صعودي مؤكد",
    "Insufficient confirmations": "تأكيدات غير كافية",
    "Multiple conditions failed": "عدة شروط تأكيد فشلت",
    "Bollinger: No bounce": "Bollinger: لا يوجد ارتداد من الحدود",
    "Stochastic: No oversold condition": "Stochastic: لم يصل إلى منطقة التشبع في البيع",
    "ZigZag: No breakout": "ZigZag: لا يوجد اختراق",
    "VWAP: Far from mean": "VWAP: السعر بعيد جداً عن المتوسط"
}

# --- إعداد تطبيق Flask و WebSocket ---
app = Flask(__name__)
CORS(app)
sock = Sock(app)
ws_clients: List[Any] = []
ws_clients_lock = Lock()

# --- WebSocket Health Monitor ---
ws_health_timer = None
WS_HEALTH_CHECK_INTERVAL = 30  # ثواني

def ensure_ws_connection():
    """مراقبة صحة اتصال WebSocket وإعادة الاتصال تلقائياً"""
    global ws_manager, ws_health_timer
    
    def check_and_reconnect():
        try:
            if ws_manager and ws_manager._conn and ws_manager._conn.connected:
                logger.debug("[WebSocket Health] Connection is healthy")
            else:
                logger.warning("[WebSocket Health] Connection lost, reconnecting...")
                reconnect_websocket()
        except Exception as e:
            logger.error(f"❌ [WebSocket Health] Check failed: {e}")
            reconnect_websocket()
        finally:
            # جدولة الفحص التالي
            ws_health_timer = Timer(WS_HEALTH_CHECK_INTERVAL, check_and_reconnect)
            ws_health_timer.daemon = True
            ws_health_timer.start()
    
    # بدء المراقبة
    check_and_reconnect()

def reconnect_websocket():
    """إعادة اتصال WebSocket مع تأخير تدريجي"""
    global ws_manager
    
    try:
        if ws_manager:
            ws_manager.stop()
            logger.info("[WebSocket] Stopped previous connection")
    except:
        pass
    
    time.sleep(2)
    
    try:
        ws_manager = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
        ws_manager.start()
        ws_manager.start_ticker_socket(callback=handle_socket_message)
        logger.info("✅ [WebSocket] Reconnected successfully")
    except Exception as e:
        logger.error(f"❌ [WebSocket] Reconnection failed: {e}")

def broadcast(data: Dict):
    """بث البيانات لجميع عملاء WebSocket مع معالجة الأخطاء"""
    with ws_clients_lock:
        if not ws_clients:
            logger.debug("[WebSocket] No clients connected to broadcast")
            return
        
        clients_to_remove = []
        for i, client in enumerate(ws_clients):
            try:
                client.send(json.dumps(data, cls=NpEncoder))
                logger.debug(f"[WebSocket] Sent {data['type']} to client {i}")
            except Exception as e:
                logger.warning(f"[WebSocket] Send failed to client {i}, removing: {e}")
                clients_to_remove.append(client)
        
        for client in clients_to_remove:
            try:
                ws_clients.remove(client)
                logger.info("[WebSocket] Removed disconnected client")
            except ValueError:
                pass

class NpEncoder(json.JSONEncoder):
    """معالج JSON لأنواع NumPy الخاصة"""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, pd.Timestamp): return obj.isoformat()
        else: return super(NpEncoder, self).default(obj)

def get_dashboard_payload() -> Dict:
    """جلب جميع بيانات لوحة التحكم مع التحقق من الاتصال بقاعدة البيانات"""
    try:
        with trading_status_lock: trading_enabled = is_trading_enabled
        with trading_mode_lock: is_paper_mode = paper_trading_mode
        with balance_lock: current_balance = usdt_balance
        with notifications_lock: notifications = list(notifications_cache)
        with rejection_logs_lock: rejections = list(rejection_logs_cache)
        with market_state_lock: market_state = dict(current_market_state)
        with min_quality_lock: min_quality = MIN_SIGNAL_QUALITY
        with trade_amount_lock:
            trade_amount_min = FIXED_TRADE_AMOUNT_MIN_USDT
            trade_amount_max = FIXED_TRADE_AMOUNT_MAX_USDT
        with strategy_config_lock:
            strategy_config = {
                "enable_ema_filter": ENABLE_EMA_FILTER,
                "enable_macd_confirmation": ENABLE_MACD_CONFIRMATION,
                "enable_mfi_filter": ENABLE_MFI_FILTER,
                "enable_candlestick_patterns": ENABLE_CANDLESTICK_PATTERNS,
                "required_confirmations": REQUIRED_CONFIRMATIONS
            }
        
        with strategies_lock:
            strategies = STRATEGY_NAMES
        
        return {
            "trading_enabled": trading_enabled,
            "paper_trading_mode": is_paper_mode,
            "usdt_balance": current_balance,
            "notifications": notifications,
            "rejections": rejections,
            "market_state": market_state,
            "min_signal_quality": min_quality,
            "trade_amount_min": trade_amount_min,
            "trade_amount_max": trade_amount_max,
            "strategy_config": strategy_config,
            "strategies": strategies,
            "server_time": datetime.now(timezone.utc).isoformat(),
            "open_signals_cache_count": len(open_signals_cache)
        }
    except Exception as e:
        logger.error(f"❌ [Dashboard] Error generating data: {e}", exc_info=True)
        return {"error": str(e)}

# --- دوال المساعدة ---
def optimize_database():
    """تحسين قاعدة البيانات بإضافة الفهارس"""
    if not check_db_connection() or not conn:
        logger.warning("[DB] Connection not available for optimization")
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
    """التحقق من وجود عمود في جدول"""
    cursor.execute("SELECT 1 FROM information_schema.columns WHERE table_name = %s AND column_name = %s", (table_name, column_name))
    return cursor.fetchone() is not None

def init_db(retries: int = 5, base_delay: int = 5) -> None:
    """تهيئة اتصال قاعدة البيانات مع إعادة محاولة تلقائية"""
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
                # إنشاء الجداول الأساسية
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION NOT NULL,
                        stop_loss DOUBLE PRECISION NOT NULL, status TEXT DEFAULT 'open',
                        closing_price DOUBLE PRECISION, closed_at TIMESTAMP, profit_percentage DOUBLE PRECISION,
                        strategy_name TEXT, signal_details JSONB, is_real_trade BOOLEAN DEFAULT FALSE,
                        quantity DOUBLE PRECISION, closing_reason TEXT, order_id TEXT
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(), type TEXT NOT NULL, message TEXT NOT NULL
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS performance_summary (
                        id SERIAL PRIMARY KEY,
                        trade_id INTEGER REFERENCES signals(id),
                        profit_percentage DOUBLE PRECISION,
                        drawdown DOUBLE PRECISION,
                        date DATE
                    );
                """)
                
                # إضافة الأعمدة المفقودة
                columns_to_add = {
                    "target_price_1": "DOUBLE PRECISION",
                    "target_price_2": "DOUBLE PRECISION",
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
            if attempt < retries - 1:
                delay = base_delay * (2  ** attempt)
                logger.info(f"[DB] Retrying connection in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.critical(" ❌ [DB] Failed to connect after all retries. Exiting.")
                exit(1)

def check_db_connection() -> bool:
    """التحقق من صحة اتصال قاعدة البيانات مع إعادة الاتصال التلقائي"""
    global conn
    if conn is None or conn.closed != 0:
        logger.warning("[DB] Connection is None or closed. Will attempt to reconnect...")
        try:
            init_db(retries=3, base_delay=2)
        except:
            logger.error(" [DB] Reconnection attempt failed.")
            return False
    
    try:
        if conn and conn.closed == 0:
            with conn.cursor() as cur: cur.execute("SELECT 1;")
            return True
        logger.warning("[DB] Connection check failed.")
        return False
    except (OperationalError, InterfaceError) as e:
        logger.error(f" [DB] Connection lost ({e}). Attempting to reconnect...")
        try:
            init_db(retries=3, base_delay=2)
        except:
            logger.error("❌ [DB] Reconnection attempt failed.")
        return conn is not None and conn.closed == 0

def init_redis() -> None:
    """تهيئة اتصال Redis"""
    global redis_client
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] Connected successfully.")
    except redis.exceptions.ConnectionError as e:
        logger.warning(f"⚠️ [Redis] Connection failed: {e}.")
        redis_client = None

def log_and_notify(level: str, message: str, notification_type: str):
    """تسجيل وإخطار الأحداث مع التحقق من الاتصال"""
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error}
    log_methods.get(level.lower(), logger.info)(message)
    
    if not check_db_connection() or not conn:
        logger.warning("[Log & Notify] Skipping DB save due to connection issues")
        return
    
    try:
        new_notification = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": notification_type,
            "message": message
        }
        with notifications_lock: notifications_cache.appendleft(new_notification)
        
        with conn.cursor() as cur:
            cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
        
        broadcast({"type": "new_notification", "payload": new_notification})
        logger.debug(f"[WebSocket] Broadcasted notification: {notification_type}")
    except Exception as e:
        logger.error(f"❌ [DB] Failed to save notification: {e}")
        if conn: conn.rollback()

def log_rejection(symbol: str, reason_key: str, details: Optional[Dict] = None):
    """تسجيل أسباب رفض الإشارات"""
    try:
        reason_ar = REJECTION_REASONS_AR.get(reason_key, reason_key)
        if details:
            details_str = ", ".join([f"{k}: {v}" for k, v in details.items()])
            reason_ar = f"{reason_ar} ({details_str})"
        
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "reason": reason_ar
        }
        
        with rejection_logs_lock:
            rejection_logs_cache.appendleft(log_entry)
        
        broadcast({"type": "new_rejection", "payload": log_entry})
        logger.debug(f"[Rejection] {symbol}: {reason_ar}")
    except Exception as e:
        logger.error(f"❌ [Log Rejection] Error logging rejection for {symbol}: {e}", exc_info=True)

def get_notification_settings() -> Dict:
    """جلب إعدادات الإشعارات من Redis"""
    defaults = {'telegram_enabled': True, 'email_enabled': False, 'min_profit_notification': 1.0, 'max_loss_notification': -1.0}
    if not redis_client: return defaults
    
    try:
        settings_data = redis_client.get('notification_settings')
        if settings_data:
            settings = json.loads(settings_data)
            for key, value in defaults.items(): settings.setdefault(key, value)
            return settings
    except Exception as e:
        logger.error(f"❌ [Redis] Failed to get notification settings: {e}")
        return defaults

def send_enhanced_telegram_message(message: str, force: bool = False):
    """إرسال رسائل Telegram مع معالجة الأخطاء"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("[Telegram] Tokens not configured")
        return
    
    settings = get_notification_settings()
    if not settings.get('telegram_enabled') and not force:
        return
    
    max_length = 4096
    messages = [message[i:i+max_length] for i in range(0, len(message), max_length)]
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    for i, msg in enumerate(messages):
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
                    retry_after = int(r.json().get("parameters", {}).get("retry_after", 1))
                    time.sleep(min(5, retry_after))
                    continue
                if r.ok:
                    logger.info(f"[Telegram] Message {i+1}/{len(messages)} sent successfully")
                    break
                else:
                    logger.warning(f"[Telegram] HTTP {r.status_code}: {r.text}")
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    logger.error(f"❌ [Telegram] Failed to send message after retries: {e}")
                time.sleep(1.5)

def send_trade_open_notification(symbol: str, strategy_name: str, entry_price: float, stop_loss: float,
                                target1: float, target2: float, quantity: float, is_real: bool,
                                quality_score: int, atr_percent: float, notional_value: float, confirmations: Dict):
    """إشعار بفتح صفقة جديدة"""
    trade_type = "حقيقية" if is_real else "ورقية"
    emoji = "🔥" if is_real else "📊"
    
    confirms_list = "\n".join([f"✅ {k.replace('_', ' ').title()}: {'نعم' if v else 'لا'}" for k, v in confirmations.items()])
    
    message = (
        f"{emoji} *صفقة {trade_type} جديدة (5 دقائق)*\n\n"
        f"*العملة:* `{symbol}`\n"
        f"*الاستراتيجية:* `{STRATEGY_NAMES.get(strategy_name, strategy_name)}`\n"
        f"*جودة الإشارة:* `{quality_score}/100`\n"
        f"*تقلب السوق:* `{atr_percent:.2f}%`\n\n"
        f"*التأكيدات:*\n{confirms_list}\n\n"
        f"*سعر الدخول:* `{entry_price:.4f}`\n"
        f"*وقف الخسارة:* `{stop_loss:.4f}`\n"
        f"*الهدف الأول:* `{target_price_1:.4f}`\n"
        f"*الهدف الثاني:* `{target_price_2:.4f}`\n\n"
        f"*الكمية:* `{quantity:.4f}`\n"
        f"*قيمة الصفقة:* `${notional_value:.2f}`\n"
        f"*نسبة المخاطرة:* `{((entry_price - stop_loss) / entry_price * 100):.2f}%`\n"
        f"*نسبة الربح المحتملة 1:* `{((target1 - entry_price) / entry_price * 100):.2f}%`\n"
        f"*نسبة الربح المحتملة 2:* `{((target2 - entry_price) / entry_price * 100):.2f}%`"
    )
    
    send_enhanced_telegram_message(message, force=True)

# --- الفلاتر الديناميكية ونظام السوق ---
def check_market_volatility_filter_enhanced(df: pd.DataFrame, symbol: str = "Unknown") -> bool:
    """فلتر تقلب السوق مع تسجيل واضح"""
    if 'atr_percent' not in df.columns or df['atr_percent'].isnull().all():
        log_rejection(symbol, "Market Volatility Filter Failed", {"reason": "No ATR data"})
        return False
    
    last_atr_percent = float(df.iloc[-1].get('atr_percent', 0))
    ATR_PERCENT_MIN = 0.35
    ATR_PERCENT_MAX = 3.2
    
    if not (ATR_PERCENT_MIN <= last_atr_percent <= ATR_PERCENT_MAX):
        log_rejection(symbol, "Market Volatility Filter Failed", {
            "atr": f"{last_atr_percent:.2f}%",
            "range": f"({ATR_PERCENT_MIN:.2f}-{ATR_PERCENT_MAX:.2f})%"
        })
        return False
    
    logger.debug(f"[Volatility Filter] {symbol} passed: {last_atr_percent:.2f}%")
    return True

def calculate_dynamic_stop_loss(df: pd.DataFrame, entry_price: float, strategy_name: str) -> float:
    """حساب وقف الخسارة الديناميكي"""
    last = df.iloc[-1]
    atr_value = last.get('atr', 0)
    
    stop_loss = entry_price - (atr_value * 2.0)
    
    if strategy_name == "RSI_Enhanced_Strategy":
        recent_low = df['low'].tail(5).min()
        stop_loss = min(recent_low * 0.995, entry_price - (atr_value * 1.8))
    
    max_stop_distance = entry_price * 0.05
    if entry_price - stop_loss > max_stop_distance:
        stop_loss = entry_price - max_stop_distance
    
    return stop_loss

def calculate_dynamic_take_profit(df: pd.DataFrame, entry_price: float, stop_loss: float, strategy_name: str) -> tuple:
    """حساب أهداف الربح الديناميكية"""
    risk_amount = entry_price - stop_loss
    if risk_amount <= 0: return (entry_price * 1.015, entry_price * 1.025)

    # نسب مخاطرة/ربح ديناميكية بناءً على الاستراتيجية
    if strategy_name == "EMA_Macd_Trend_Strategy":
        rr1, rr2 = 2.5, 4.0  # أهداف أعلى للاتجاهات القوية
    elif strategy_name == "Bollinger_Stochastic_Range_Strategy":
        rr1, rr2 = 1.5, 2.5  # أهداف محافظة للأسواق الجانبية
    else:
        rr1, rr2 = 1.8, 3.0  # افتراضي
    
    target1 = entry_price + (risk_amount * rr1)
    target2 = entry_price + (risk_amount * rr2)
    
    if 'r1' in df.columns:
        r1 = df.iloc[-1].get('r1', target1)
        r2 = df.iloc[-1].get('r2', target2)
        target1 = max(target1, r1 * 0.98)
        target2 = max(target2, r2 * 0.98)
    
    return target1, target2

# --- نظام اختيار استراتيجية ذكي ---
def determine_market_regime(df: pd.DataFrame) -> str:
    """تحديد نظام السوق الحالي: bullish, bearish, sideways, volatile"""
    if len(df) < 50:
        return "sideways"
    
    last = df.iloc[-1]
    
    # تحليل الاتجاه باستخدام EMA50 و ADX
    trend_strength = last.get('adx', 0)
    price_vs_ema = last['close'] > last.get('ema50', last['close'])
    
    if trend_strength > 25:
        if price_vs_ema:
            return "bullish"
        else:
            return "bearish"
    elif trend_strength > 20:
        return "volatile"
    else:
        return "sideways"

def select_optimal_strategy(df: pd.DataFrame, mtf_trend: Dict) -> Optional[str]:
    """اختيار الاستراتيجية الأمثل بناءً على حالة السوق"""
    market_regime = determine_market_regime(df)
    
    # تعيين الاستراتيجيات للأوضاع السوقية
    strategy_map = {
        "bullish": ["EMA_Macd_Trend_Strategy", "ZigZag_Breakout_Strategy"],
        "bearish": ["RSI_Enhanced_Strategy", "VWAP_Mean_Reversion_Strategy"],
        "sideways": ["Bollinger_Stochastic_Range_Strategy", "VWAP_Mean_Reversion_Strategy"],
        "volatile": ["ZigZag_Breakout_Strategy", "RSI_Enhanced_Strategy"]
    }
    
    candidates = strategy_map.get(market_regime, ["RSI_Enhanced_Strategy"])
    
    # اختيار أول استراتيجية تمر الفلاتر
    for strategy_name in candidates:
        if strategy_name in strategy_registry:
            logger.debug(f"[Strategy Selection] Trying {strategy_name} for {market_regime} market")
            if strategy_registry[strategy_name](df, mtf_trend):
                logger.info(f"✅ [Strategy Selected] {strategy_name} for {market_regime} market")
                return strategy_name
    
    return None

# --- استراتيجية 1: RSI المتقدم (الأصلية) ---
def check_rsi_enhanced_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """
    استراتيجية RSI المُحسّنة مع تأكيدات متعددة - تعمل بشكل جيد في الأسواق الهابطة أو الانعكاسية
    """
    if len(df) < 50:
        logger.debug(f"[RSI Strategy] {df.name if hasattr(df, 'name') else 'Unknown'}: Insufficient data ({len(df)} < 50)")
        return False

    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    confirmations = {
        'rsi_crossover': False,
        'ema_trend': False,
        'macd_momentum': False,
        'mfi_pressure': False,
        'candlestick_bullish': False
    }
    
    # 1. شرط RSI الأساسي
    confirmations['rsi_crossover'] = prev['rsi'] < 30 and last['rsi'] >= 30
    
    # 2. فلتر EMA21
    with strategy_config_lock:
        ema_enabled = ENABLE_EMA_FILTER
    if ema_enabled:
        confirmations['ema_trend'] = last['close'] > last['ema21']
    else:
        confirmations['ema_trend'] = True
    
    # 3. تأكيد MACD
    with strategy_config_lock:
        macd_enabled = ENABLE_MACD_CONFIRMATION
    if macd_enabled:
        confirmations['macd_momentum'] = last['macd_hist'] > 0 and prev['macd_hist'] < last['macd_hist']
    else:
        confirmations['macd_momentum'] = True
    
    # 4. فلتر MFI
    with strategy_config_lock:
        mfi_enabled = ENABLE_MFI_FILTER
    if mfi_enabled:
        confirmations['mfi_pressure'] = last['mfi'] > 20
    else:
        confirmations['mfi_pressure'] = True
    
    # 5. نمط الشموع
    with strategy_config_lock:
        candle_enabled = ENABLE_CANDLESTICK_PATTERNS
    if candle_enabled:
        confirmations['candlestick_bullish'] = detect_bullish_candlestick_pattern(df)
    else:
        confirmations['candlestick_bullish'] = True
    
    # حساب عدد التأكيدات
    active_confirmations = sum(confirmations.values())
    
    with strategy_config_lock:
        required = REQUIRED_CONFIRMATIONS
    
    # تسجيل أسباب الرفض
    if not confirmations['rsi_crossover']:
        log_rejection(df.name if hasattr(df, 'name') else 'Unknown', "RSI: No bullish crossover")
        return False
    
    if active_confirmations < required:
        failed_conditions = [k for k, v in confirmations.items() if not v]
        log_rejection(df.name if hasattr(df, 'name') else 'Unknown', 
                     f"Insufficient confirmations ({active_confirmations}/{required})", 
                     {"failed": ', '.join(failed_conditions)})
        return False
    
    logger.info(f"✅ [Enhanced RSI Signal] {df.name if hasattr(df, 'name') else 'Unknown'}: {active_confirmations}/{required} confirmations met")
    df.confirmations = confirmations
    return True

# --- استراتيجية 2: EMA + MACD للاتجاهات الصاعدة ---
def check_ema_macd_trend_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """
    استراتيجية تعقب الاتجاه الصاعدة - تعمل بشكل أفضل في الأسواق الصاعدة
    """
    if len(df) < 100:
        return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    confirmations = {
        'ema_golden_cross': False,
        'macd_bullish': False,
        'adx_strong': False,
        'volume_confirmed': False,
        'price_above_vwap': False
    }
    
    # 1. تقاطع EMA21 فوق EMA50 (Golden Cross)
    confirmations['ema_golden_cross'] = last['ema21'] > last['ema50'] and prev['ema21'] <= prev['ema50']
    
    # 2. MACD فوق خط الإشارة
    confirmations['macd_bullish'] = last['macd'] > last['macd_signal'] and last['macd_hist'] > 0
    
    # 3. ADX يشير إلى اتجاه قوي
    confirmations['adx_strong'] = last['adx'] > 25
    
    # 4. تأكيد الحجم (أعلى من المتوسط)
    avg_volume = df['volume'].rolling(20).mean().iloc[-1]
    confirmations['volume_confirmed'] = last['volume'] > avg_volume * 1.2
    
    # 5. السعر أعلى من VWAP
    confirmations['price_above_vwap'] = last['close'] > last['vwap']
    
    # متطلبات أقل للاتجاهات الصاعدة
    required_confirmations = max(3, REQUIRED_CONFIRMATIONS - 1)
    active_confirmations = sum(confirmations.values())
    
    if active_confirmations >= required_confirmations:
        logger.info(f"✅ [EMA MACD Trend] {df.name}: {active_confirmations}/{required_confirmations} confirmations")
        df.confirmations = confirmations
        return True
    
    failed_conditions = [k for k, v in confirmations.items() if not v]
    log_rejection(df.name, "Insufficient confirmations for trend strategy", {
        "failed": ', '.join(failed_conditions)
    })
    return False

# --- استراتيجية 3: Bollinger + Stochastic للأسواق الجانبية ---
def check_bollinger_stochastic_range_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """
    استراتيجية للأسواق الجانبية - تشتري عند الحدود السفلى لـ Bollinger Bands
    """
    if len(df) < 50:
        return False
    
    last = df.iloc[-1]
    
    confirmations = {
        'bb_lower_touch': False,
        'stochastic_oversold': False,
        'volume_contracting': False,
        'bb_width_narrow': False,
        'bullish_engulfing': False
    }
    
    # 1. السعر يمس أو يكسر الحد السفلي لـ Bollinger
    confirmations['bb_lower_touch'] = last['close'] <= last['bb_lower']
    
    # 2. Stochastic في منطقة التشبع في البيع
    confirmations['stochastic_oversold'] = last['stoch_k'] < 20
    
    # 3. حجم التداول متقلص (قبل الاختراق)
    bb_width = last['bb_width']
    avg_width = df['bb_width'].rolling(20).mean().iloc[-1]
    confirmations['bb_width_narrow'] = bb_width < avg_width * 0.8
    
    # 4. الحجم أقل من المتوسط
    avg_volume = df['volume'].rolling(20).mean().iloc[-1]
    confirmations['volume_contracting'] = last['volume'] < avg_volume * 0.8
    
    # 5. نمط شمعة صعودي Bullish Engulfing
    prev = df.iloc[-2]
    confirmations['bullish_engulfing'] = (
        prev['close'] < prev['open'] and 
        last['close'] > last['open'] and
        last['close'] > prev['open'] and
        last['open'] < prev['close']
    )
    
    required_confirmations = 4  # أكثر صرامة للأسواق الجانبية
    active_confirmations = sum(confirmations.values())
    
    if active_confirmations >= required_confirmations:
        logger.info(f"✅ [BB Stochastic Range] {df.name}: {active_confirmations}/{required_confirmations} confirmations")
        df.confirmations = confirmations
        return True
    
    failed_conditions = [k for k, v in confirmations.items() if not v]
    log_rejection(df.name, "Bollinger: No bounce", {
        "failed": ', '.join(failed_conditions)
    })
    return False

# --- استراتيجية 4: ZigZag Breakout ---
def check_zigzag_breakout_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """
    استراتيجية اختراق ZigZag - تعمل في الأسواق المتقلبة والاندفاعات
    """
    if len(df) < 100:
        return False
    
    # حساب ZigZag
    df['zigzag'] = calculate_zigzag(df, deviation=5)
    
    # إيجاد القمم والقيعان الأخيرة
    recent_peaks = df[df['zigzag'] > 0].tail(5)
    recent_troughs = df[df['zigzag'] < 0].tail(5)
    
    if len(recent_peaks) < 3 or len(recent_troughs) < 3:
        return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    confirmations = {
        'higher_highs': False,
        'higher_lows': False,
        'volume_surge': False,
        'breakout_confirmation': False,
        'adx_volatile': False
    }
    
    # 1. القمم تصاعدية
    highs = recent_peaks['high'].values
    confirmations['higher_highs'] = all(highs[i] < highs[i+1] for i in range(len(highs)-1))
    
    # 2. القيعان تصاعدية
    lows = recent_troughs['low'].values
    confirmations['higher_lows'] = all(lows[i] < lows[i+1] for i in range(len(lows)-1))
    
    # 3. حجم مرتفع جداً
    avg_volume = df['volume'].rolling(20).mean().iloc[-1]
    confirmations['volume_surge'] = last['volume'] > avg_volume * 2
    
    # 4. اختراق أعلى قمة سابقة
    last_peak = recent_peaks['high'].iloc[-1]
    confirmations['breakout_confirmation'] = last['close'] > last_peak
    
    # 5. ADX يشير إلى تقلب
    confirmations['adx_volatile'] = 20 < last['adx'] < 40
    
    required_confirmations = 4
    active_confirmations = sum(confirmations.values())
    
    if active_confirmations >= required_confirmations:
        logger.info(f"✅ [ZigZag Breakout] {df.name}: {active_confirmations}/{required_confirmations} confirmations")
        df.confirmations = confirmations
        return True
    
    failed_conditions = [k for k, v in confirmations.items() if not v]
    log_rejection(df.name, "ZigZag: No breakout", {
        "failed": ', '.join(failed_conditions)
    })
    return False

def calculate_zigzag(df: pd.DataFrame, deviation: float = 5) -> pd.Series:
    """حساب ZigZag (تبسيط)"""
    zigzag = pd.Series(0, index=df.index)
    
    last_pivot = df.iloc[0]['close']
    last_pivot_idx = 0
    
    for i in range(1, len(df)):
        current_price = df.iloc[i]['close']
        change = (current_price - last_pivot) / last_pivot * 100
        
        if abs(change) >= deviation:
            if change > 0:
                zigzag.iloc[i] = 1  # قمة
            else:
                zigzag.iloc[i] = -1  # قاع
            
            last_pivot = current_price
            last_pivot_idx = i
    
    return zigzag

# --- استراتيجية 5: VWAP Mean Reversion ---
def check_vwap_mean_reversion_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """
    استراتيجية انعكاس الوسيط - تعمل في جميع الأوضاع لكن بشكل أفضل في الانعكاسات
    """
    if len(df) < 50:
        return False
    
    last = df.iloc[-1]
    
    confirmations = {
        'far_from_vwap': False,
        'rsi_extreme': False,
        'stochastic_reversal': False,
        'volume_spike': False,
        'reversal_candle': False
    }
    
    # 1. السعر بعيد جداً عن VWAP
    vwap_deviation = abs(last['close'] - last['vwap']) / last['vwap'] * 100
    confirmations['far_from_vwap'] = vwap_deviation > 2.0
    
    # 2. RSI في منطقة التشبع
    confirmations['rsi_extreme'] = last['rsi'] < 25 or last['rsi'] > 75
    
    # 3. Stochastic يعطي إشارة انعكاس
    prev = df.iloc[-2]
    confirmations['stochastic_reversal'] = (
        (last['stoch_k'] > 20 and prev['stoch_k'] < 20) or
        (last['stoch_k'] < 80 and prev['stoch_k'] > 80)
    )
    
    # 4. حجم مرتفع (إشارة على الانعكاس)
    avg_volume = df['volume'].rolling(20).mean().iloc[-1]
    confirmations['volume_spike'] = last['volume'] > avg_volume * 1.5
    
    # 5. نمط انعكاس (Hammer أو Shooting Star)
    body = abs(last['close'] - last['open'])
    lower_shadow = min(last['open'], last['close']) - last['low']
    upper_shadow = last['high'] - max(last['open'], last['close'])
    
    is_hammer = body > 0 and lower_shadow > 2 * body and upper_shadow < 0.3 * body
    is_shooting_star = body > 0 and upper_shadow > 2 * body and lower_shadow < 0.3 * body
    
    confirmations['reversal_candle'] = is_hammer or is_shooting_star
    
    required_confirmations = 4
    active_confirmations = sum(confirmations.values())
    
    if active_confirmations >= required_confirmations:
        logger.info(f"✅ [VWAP Mean Reversion] {df.name}: {active_confirmations}/{required_confirmations} confirmations")
        df.confirmations = confirmations
        return True
    
    failed_conditions = [k for k, v in confirmations.items() if not v]
    log_rejection(df.name, "VWAP: Far from mean", {
        "failed": ', '.join(failed_conditions)
    })
    return False

# --- تسجيل جميع الاستراتيجيات ---
def register_strategies():
    """تسجيل جميع الاستراتيجيات في السجل"""
    global strategy_registry
    with strategies_lock:
        strategy_registry = {
            "RSI_Enhanced_Strategy": check_rsi_enhanced_strategy,
            "EMA_Macd_Trend_Strategy": check_ema_macd_trend_strategy,
            "Bollinger_Stochastic_Range_Strategy": check_bollinger_stochastic_range_strategy,
            "ZigZag_Breakout_Strategy": check_zigzag_breakout_strategy,
            "VWAP_Mean_Reversion_Strategy": check_vwap_mean_reversion_strategy
        }
    logger.info(f"✅ [Strategies] Registered {len(strategy_registry)} strategies")

def detect_bullish_candlestick_pattern(df: pd.DataFrame) -> bool:
    """كشف أنماط الشموع الصعودية"""
    if len(df) < 3:
        return False
    
    last = df.iloc[-1]
    
    # Hammer pattern
    body = abs(last['close'] - last['open'])
    lower_shadow = min(last['open'], last['close']) - last['low']
    upper_shadow = last['high'] - max(last['open'], last['close'])
    is_hammer = body > 0 and lower_shadow > 2 * body and upper_shadow < 0.3 * body
    
    # Bullish Engulfing
    prev = df.iloc[-2]
    prev_body = abs(prev['close'] - prev['open'])
    prev_is_red = prev['close'] < prev['open']
    current_is_green = last['close'] > last['open']
    is_engulfing = prev_is_red and current_is_green and last['close'] > prev['open'] and last['open'] < prev['close']
    
    return is_hammer or is_engulfing

# --- Data Loading & Settings Management ---
def load_open_signals_to_cache():
    """تحميل الإشارات المفتوحة من قاعدة البيانات إلى الكاش"""
    logger.info("[Cache] Starting to load open signals from database...")
    
    if not check_db_connection() or not conn:
        logger.error("[Cache] Cannot load signals: DB connection failed")
        return
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, symbol, entry_price, target_price_1, target_price_2, stop_loss, 
                       strategy_name, is_real_trade, quantity, initial_quantity, 
                       signal_details, status, order_id
                FROM signals 
                WHERE status IN ('open', 'updated')
                ORDER BY id DESC
            """)
            
            signals = cur.fetchall()
            logger.info(f"[Cache] Fetched {len(signals)} open signals from database")
            
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in signals:
                    if isinstance(signal['signal_details'], str):
                        try:
                            signal['signal_details'] = json.loads(signal['signal_details'])
                        except:
                            signal['signal_details'] = {}
                    
                    open_signals_cache[signal['symbol']] = dict(signal)
                    logger.debug(f"[Cache] Loaded signal: {signal['symbol']} (ID: {signal['id']})")
            
            logger.info(f"✅ [Cache] Successfully loaded {len(open_signals_cache)} open signals to cache")
            broadcast({"type": "signals_loaded", "count": len(open_signals_cache)})
            
    except Exception as e:
        logger.error(f"❌ [Cache] Failed to load open signals: {e}", exc_info=True)
        if conn: conn.rollback()

def load_notifications_to_cache():
    """تحميل الإشعارات الأخيرة من قاعدة البيانات"""
    if not check_db_connection() or not conn:
        logger.warning("[Cache] Cannot load notifications: DB connection failed")
        return
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 20;")
            with notifications_lock:
                notifications_cache.clear()
                for n in reversed(cur.fetchall()):
                    if hasattr(n['timestamp'], 'isoformat'):
                        n['timestamp'] = n['timestamp'].isoformat()
                    notifications_cache.appendleft(dict(n))
        logger.info(f"✅ [Cache] Loaded {len(notifications_cache)} notifications")
    except Exception as e:
        logger.error(f"❌ [Cache] Failed to load notifications: {e}")

def load_settings_from_redis():
    """تحميل الإعدادات من Redis"""
    global FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT, MAX_OPEN_TRADES, paper_trading_mode, MIN_SIGNAL_QUALITY
    global ENABLE_EMA_FILTER, ENABLE_MACD_CONFIRMATION, ENABLE_MFI_FILTER, ENABLE_CANDLESTICK_PATTERNS, REQUIRED_CONFIRMATIONS
    
    if not redis_client:
        logger.warning("[Redis] Client not available for loading settings")
        return
    
    try:
        settings_data = redis_client.get('trading_settings')
        if settings_data:
            settings = json.loads(settings_data)
            with trade_amount_lock:
                FIXED_TRADE_AMOUNT_MIN_USDT = settings.get('FIXED_TRADE_AMOUNT_MIN_USDT', 4.5)
                FIXED_TRADE_AMOUNT_MAX_USDT = settings.get('FIXED_TRADE_AMOUNT_MAX_USDT', 6.5)
            MAX_OPEN_TRADES = settings.get('MAX_OPEN_TRADES', 3)
            with trading_mode_lock:
                paper_trading_mode = settings.get('paper_trading_mode', True)
        
        quality_settings_data = redis_client.get('signal_quality_settings')
        if quality_settings_data:
            quality_settings = json.loads(quality_settings_data)
            with min_quality_lock:
                MIN_SIGNAL_QUALITY = quality_settings.get('min_quality', 70)
        
        strategy_config_data = redis_client.get('strategy_config')
        if strategy_config_data:
            strategy_config = json.loads(strategy_config_data)
            with strategy_config_lock:
                ENABLE_EMA_FILTER = strategy_config.get('enable_ema_filter', True)
                ENABLE_MACD_CONFIRMATION = strategy_config.get('enable_macd_confirmation', True)
                ENABLE_MFI_FILTER = strategy_config.get('enable_mfi_filter', True)
                ENABLE_CANDLESTICK_PATTERNS = strategy_config.get('enable_candlestick_patterns', True)
                REQUIRED_CONFIRMATIONS = strategy_config.get('required_confirmations', 3)
        
        logger.info("✅ [Redis] Successfully loaded settings")
    except Exception as e:
        logger.error(f"❌ [Redis] Error loading settings: {e}")

def save_settings_to_redis():
    """حفظ الإعدادات في Redis"""
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
        
        strategy_config = {
            'enable_ema_filter': ENABLE_EMA_FILTER,
            'enable_macd_confirmation': ENABLE_MACD_CONFIRMATION,
            'enable_mfi_filter': ENABLE_MFI_FILTER,
            'enable_candlestick_patterns': ENABLE_CANDLESTICK_PATTERNS,
            'required_confirmations': REQUIRED_CONFIRMATIONS
        }
        redis_client.set('strategy_config', json.dumps(strategy_config))
        
        logger.info("✅ [Redis] Settings saved successfully")
        return True
    
    except Exception as e:
        logger.error(f"❌ [Redis] Error saving settings: {e}")
        return False

# --- الحسابات الفنية ---
def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """حساب جميع المؤشرات الفنية"""
    df_calc = df.copy()
    
    # SMA
    df_calc['sma7'] = df_calc['close'].rolling(window=7).mean()
    df_calc['sma200'] = df_calc['close'].rolling(window=200).mean()

    # EMA (إضافة المزيد من الفترات)
    df_calc['ema9'] = df_calc['close'].ewm(span=9, adjust=False).mean()
    df_calc['ema13'] = df_calc['close'].ewm(span=13, adjust=False).mean()
    df_calc['ema21'] = df_calc['close'].ewm(span=21, adjust=False).mean()
    df_calc['ema34'] = df_calc['close'].ewm(span=34, adjust=False).mean()
    df_calc['ema50'] = df_calc['close'].ewm(span=50, adjust=False).mean()
    df_calc['ema100'] = df_calc['close'].ewm(span=100, adjust=False).mean()
    df_calc['ema200'] = df_calc['close'].ewm(span=200, adjust=False).mean()
    
    # ATR و ADX
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
    
    # RSI (14-period)
    delta = df_calc['close'].diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain_14 = gain.ewm(com=14 - 1, adjust=False).mean()
    avg_loss_14 = loss.ewm(com=14 - 1, adjust=False).mean()
    rs_14 = avg_gain_14 / avg_loss_14.replace(0, 1e-9)
    df_calc['rsi'] = 100.0 - (100.0 / (1.0 + rs_14))
    
    # Bollinger Bands
    bb_middle = df_calc['close'].rolling(window=20).mean()
    bb_std = df_calc['close'].rolling(window=20).std()
    df_calc['bb_middle'] = bb_middle
    df_calc['bb_lower'] = bb_middle - (bb_std * 2)
    df_calc['bb_upper'] = bb_middle + (bb_std * 2)
    df_calc['bb_width'] = (df_calc['bb_upper'] - df_calc['bb_lower']) / df_calc['bb_middle'].replace(0, 1e-9)
    
    # MACD
    exp1 = df_calc['close'].ewm(span=8, adjust=False).mean()
    exp2 = df_calc['close'].ewm(span=17, adjust=False).mean()
    df_calc['macd'] = exp1 - exp2
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()
    df_calc['macd_hist'] = df_calc['macd'] - df_calc['macd_signal']
    
    # Stochastic
    low_14 = df_calc['low'].rolling(14).min()
    high_14 = df_calc['high'].rolling(14).max()
    high_low_range = high_14 - low_14
    meaningful_range = high_low_range > (df_calc['close'] * 0.0001)
    df_calc['stoch_k'] = np.where(meaningful_range, 100 * ((df_calc['close'] - low_14) / high_low_range.replace(0, 1e-9)), 50)
    df_calc['stoch_d'] = df_calc['stoch_k'].rolling(3).mean()
    
    # VWAP
    df_calc['vwap'] = (df_calc['close'] * df_calc['volume']).cumsum() / df_calc['volume'].cumsum()
    
    # MFI
    typical_price = (df_calc['high'] + df_calc['low'] + df_calc['close']) / 3
    money_flow = typical_price * df_calc['volume']
    positive_flow = money_flow.where(typical_price.diff() > 0, 0)
    negative_flow = money_flow.where(typical_price.diff() < 0, 0)
    positive_flow_sum = positive_flow.rolling(14).sum()
    negative_flow_sum = negative_flow.rolling(14).sum()
    money_ratio = positive_flow_sum / negative_flow_sum.replace(0, 1e-9)
    df_calc['mfi'] = 100 - (100 / (1 + money_ratio))
    
    # Pivot Points
    df_calc['pivot'] = (df_calc['high'].shift(1) + df_calc['low'].shift(1) + df_calc['close'].shift(1)) / 3
    df_calc['r1'] = 2 * df_calc['pivot'] - df_calc['low'].shift(1)
    df_calc['s1'] = 2 * df_calc['pivot'] - df_calc['high'].shift(1)
    df_calc['r2'] = df_calc['pivot'] + (df_calc['high'].shift(1) - df_calc['low'].shift(1))
    df_calc['s2'] = df_calc['pivot'] - (df_calc['high'].shift(1) - df_calc['low'].shift(1))
    
    return df_calc

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """جلب البيانات التاريخية من Binance"""
    time.sleep(API_REQUEST_DELAY)
    try:
        klines = client.get_historical_klines(symbol, interval, f"{days} day ago UTC")
        if not klines:
            logger.warning(f"[Data] No klines returned for {symbol}")
            return None
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore'
        ])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        
        return df.dropna().astype(float)
    except Exception as e:
        logger.error(f"❌ [Data] Error fetching data for {symbol}: {e}")
        return None

# --- حساب حجم الصفقة ---
def get_formatted_quantity(symbol: str, quantity: Decimal) -> str:
    """تنسيق الكمية حسب متطلبات Binance"""
    try:
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info:
            logger.warning(f"[{symbol}] No exchange info for formatting")
            return f"{quantity.normalize()}"

        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if not lot_size_filter:
            logger.warning(f"[{symbol}] LOT_SIZE filter not found")
            return f"{quantity.normalize()}"
        
        step_size = Decimal(lot_size_filter['stepSize'])
        formatted_quantity = quantity.quantize(step_size, rounding=ROUND_DOWN)
        return f"{formatted_quantity.normalize()}"
    except Exception as e:
        logger.error(f"❌ [{symbol}] Error formatting quantity: {e}")
        return str(quantity)

def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> Optional[Decimal]:
    """تعديل الكمية لتتوافق مع LOT_SIZE"""
    try:
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info:
            logger.error(f"[{symbol}] No exchange info for LOT_SIZE")
            return None
        
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if not lot_size_filter:
            logger.warning(f"[{symbol}] LOT_SIZE filter not found")
            return Decimal(str(quantity))
        
        step_size = Decimal(lot_size_filter['stepSize'])
        min_qty = Decimal(lot_size_filter['minQty'])
        quantity_dec = Decimal(str(quantity))
        
        if quantity_dec < min_qty:
            log_rejection(symbol, "LOT_SIZE Filter Failed", {
                "reason": "Below minQty",
                "qty": f"{quantity_dec}",
                "min": f"{min_qty}"
            })
            return None
        
        adjusted_quantity = quantity_dec - (quantity_dec % step_size)
        if adjusted_quantity < min_qty:
            log_rejection(symbol, "LOT_SIZE Filter Failed", {
                "reason": "Adjusted below minQty",
                "qty": f"{adjusted_quantity}",
                "min": f"{min_qty}"
            })
            return None
        
        return adjusted_quantity
    except Exception as e:
        logger.error(f"❌ [{symbol}] CRITICAL ERROR adjusting quantity: {e}", exc_info=True)
        return None

def calculate_position_size(symbol: str, entry_price: float, available_balance: float, is_real: bool) -> Optional[Decimal]:
    """حساب حجم الصفقة مع معالجة الأخطاء الشاملة"""
    desired_usdt_amount = PAPER_TRADE_FIXED_AMOUNT_USDT if not is_real else random.uniform(FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT)
    
    try:
        dec_entry = Decimal(str(entry_price))
        if dec_entry <= 0:
            logger.error(f"[{symbol}] Invalid entry price")
            return None
        
        dec_balance = Decimal(str(available_balance))
        dec_desired_amount = Decimal(str(desired_usdt_amount))
        
        logger.info(f"[{symbol}] Position sizing: Desired=${dec_desired_amount:.2f}, Available=${dec_balance:.2f}")

        if is_real and dec_desired_amount > dec_balance:
            log_rejection(symbol, "Insufficient Balance", {
                "required": f"${dec_desired_amount:.2f}",
                "available": f"${dec_balance:.2f}"
            })
            return None

        initial_quantity = dec_desired_amount / dec_entry
        adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(initial_quantity))

        if adjusted_quantity is None or adjusted_quantity <= 0:
            logger.warning(f"[{symbol}] Quantity adjustment failed")
            adjusted_quantity = Decimal('0')

        notional_value = adjusted_quantity * dec_entry
        symbol_info = exchange_info_map.get(symbol)
        
        if symbol_info:
            min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] in ('MIN_NOTIONAL', 'NOTIONAL')), None)
            
            if min_notional_filter:
                min_notional_str = min_notional_filter.get('minNotional', min_notional_filter.get('notional', '5.0'))
                min_notional = Decimal(min_notional_str)
                
                if notional_value < min_notional:
                    logger.warning(f"[{symbol}] Notional ${notional_value:.2f} < min ${min_notional}")
                    
                    required_notional = min_notional * Decimal('1.01')
                    
                    if is_real and required_notional > dec_balance:
                        log_rejection(symbol, "Insufficient Balance", {
                            "reason": "Cannot meet min_notional",
                            "required": f"${required_notional:.2f}",
                            "available": f"${dec_balance:.2f}"
                        })
                        return None
                    
                    new_quantity = required_notional / dec_entry
                    adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(new_quantity))

                    if adjusted_quantity is None or adjusted_quantity <= 0:
                        log_rejection(symbol, "MinNotional Filter Failed", {
                            "reason": "Failed to adjust quantity"
                        })
                        return None

                    notional_value = adjusted_quantity * dec_entry
                    logger.info(f"[{symbol}] Adjusted for min_notional: {adjusted_quantity} (${notional_value:.2f})")

        if notional_value <= 0:
            log_rejection(symbol, "MinNotional Filter Failed", {
                "reason": "Final notional is zero"
            })
            return None

        if is_real and notional_value > dec_balance:
            log_rejection(symbol, "Insufficient Balance", {
                "required": f"{notional_value:.2f}",
                "available": f"${dec_balance:.2f}"
            })
            return None
        
        logger.info(f"[ {symbol}] Final quantity: {adjusted_quantity} (${notional_value:.2f})")
        return adjusted_quantity

    except Exception as e:
        logger.error(f"❌ [{symbol}] Unhandled exception in calculate_position_size: {e}", exc_info=True)
        return None

def calculate_dynamic_quality_score(df: pd.DataFrame, symbol: str) -> int:
    """حساب درجة جودة الإشارة"""
    if len(df) < 2:
        return 0
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    score = 0
    
    # 1. RSI Crossover Strength
    if prev['rsi'] < 30 and last['rsi'] >= 30:
        rsi_strength = min((last['rsi'] - 30) / 10 * 20, 20)
        score += rsi_strength
    
    # 2. EMA21 Distance
    ema_distance = ((last['close'] - last['ema21']) / last['ema21']) * 100
    if ema_distance > 0:
        ema_score = min(ema_distance * 4, 20)
        score += ema_score
    
    # 3. MACD Momentum
    macd_strength = last['macd_hist']
    if macd_strength > 0:
        macd_score = min(macd_strength / abs(last['macd_signal']) * 20, 20)
        score += macd_score
    
    # 4. MFI Volume Pressure
    if last['mfi'] > 20:
        mfi_score = min((last['mfi'] - 20) / 80 * 20, 20)
        score += mfi_score
    
    # 5. ATR Volatility
    atr_percent = last['atr_percent']
    if 0.5 <= atr_percent <= 1.5:
        score += 20
    elif atr_percent > 0.35 and atr_percent < 3.0:
        score += 10
    
    final_score = int(min(score, 100))
    logger.debug(f"[Quality Score] {symbol}: {final_score}/100")
    return final_score

# --- أنظمة الفلاتر ---
def add_news_filter() -> bool:
    """فلتر الأخبار"""
    news_hours = [(12, 30), (14, 0), (18, 30)]
    now = datetime.now(timezone.utc)
    for hour, minute in news_hours:
        if now.hour == hour and abs(now.minute - minute) <= 15:
            return False
    return True

def add_liquidity_filter() -> bool:
    """فلتر السيولة"""
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5: return False
    if now.hour >= 22 or now.hour <= 2: return False
    return True

def add_correlation_filter(new_symbol: str) -> bool:
    """فلتر الارتباط"""
    correlated_groups = [
        {'BTCUSDT', 'ETHUSDT', 'BCHUSDT'},
        {'ADAUSDT', 'DOTUSDT', 'LINKUSDT'},
        {'SOLUSDT', 'AVAXUSDT', 'MATICUSDT'},
    ]
    with signal_cache_lock:
        open_symbols = set(open_signals_cache.keys())
    
    if not open_symbols: return True
    
    for group in correlated_groups:
        if new_symbol in group and not open_symbols.isdisjoint(group):
            return False
    
    return True

def check_market_trend_filter(strategy_name: str, market_state: str) -> bool:
    """فلتر تطابق الاستراتيجية مع حالة السوق"""
    strategy_regime_map = {
        "RSI_Enhanced_Strategy": ["bearish", "sideways"],
        "EMA_Macd_Trend_Strategy": ["bullish", "volatile"],
        "Bollinger_Stochastic_Range_Strategy": ["sideways"],
        "ZigZag_Breakout_Strategy": ["volatile", "bullish"],
        "VWAP_Mean_Reversion_Strategy": ["bearish", "bullish", "sideways"]
    }
    
    allowed_regimes = strategy_regime_map.get(strategy_name, ["sideways"])
    
    if market_state in allowed_regimes:
        return True
    
    log_rejection("System", "Market Trend Mismatch", {
        "strategy": strategy_name,
        "market_state": market_state,
        "allowed": ', '.join(allowed_regimes)
    })
    return False

# --- إنشاء وإدارة الإشارات ---
def create_trade_signal(symbol: str, df: pd.DataFrame, strategy_name: str):
    """إنشاء إشارة تداول جديدة مع التحقق الكامل"""
    logger.info(f"🔍 [Signal] Processing potential signal for {symbol}...")
    
    df.strategy = strategy_name 
    
    # تطبيق الفلاتر
    if not check_market_volatility_filter_enhanced(df, symbol):
        logger.debug(f"[Signal] {symbol} failed volatility filter")
        return
    
    if not add_news_filter():
        log_rejection(symbol, "News Filter Failed")
        return
    
    if not add_liquidity_filter():
        log_rejection(symbol, "Liquidity Filter Failed")
        return
    
    if not add_correlation_filter(symbol):
        log_rejection(symbol, "Correlation Filter Failed")
        return

    # فلتر الاتجاه السوقي
    with market_state_lock:
        market_state = current_market_state.get("overall_trend", "sideways")
    
    if not check_market_trend_filter(strategy_name, market_state):
        return
    
    # حساب جودة الإشارة
    quality_score = calculate_dynamic_quality_score(df, symbol)
    
    with min_quality_lock:
        min_score = MIN_SIGNAL_QUALITY
    
    if quality_score < min_score:
        log_rejection(symbol, "Low Quality Signal", {
            "score": quality_score,
            "min_required": min_score
        })
        return
    
    logger.info(f"⭐ [Signal Quality] {symbol}: {quality_score}/100 (min: {min_score})")

    entry_price = df.iloc[-1]['close']
    stop_loss_price = calculate_dynamic_stop_loss(df, entry_price, strategy_name)
    target_price_1, target_price_2 = calculate_dynamic_take_profit(df, entry_price, stop_loss_price, strategy_name)
    
    if stop_loss_price >= entry_price:
        log_rejection(symbol, "Invalid Position Size", {
            "entry": entry_price,
            "sl": stop_loss_price
        })
        return

    with trading_mode_lock:
        is_real = not paper_trading_mode
    
    confirmations = getattr(df, 'confirmations', {})
    
    signal_details = {
        "atr": df.iloc[-1].get('atr', 0),
        "trailing_stop_activated": False,
        "tp1_done": False,
        "quality_score": quality_score,
        "atr_percent": df.iloc[-1].get('atr_percent', 0),
        "rsi_at_signal": df.iloc[-1].get('rsi', 0),
        "confirmations": confirmations,
        "mfi_at_signal": df.iloc[-1].get('mfi', 0),
        "macd_hist_at_signal": df.iloc[-1].get('macd_hist', 0),
        "market_regime": market_state
    }
    
    trade_levels = {
        "entry_price": entry_price,
        "stop_loss": stop_loss_price,
        "target_price_1": target_price_1,
        "target_price_2": target_price_2
    }

    current_real_balance = 0
    with balance_lock:
        current_real_balance = usdt_balance

    quantity_dec = calculate_position_size(symbol, entry_price, current_real_balance, is_real)

    if quantity_dec is None or quantity_dec <= 0:
        logger.error(f"❌ [{symbol}] Position size calculation failed")
        return
    
    notional_value = float(quantity_dec) * entry_price

    # حفظ الإشارة في قاعدة البيانات
    save_signal_to_db(
        symbol, entry_price, trade_levels, strategy_name,
        is_real, float(quantity_dec), signal_details
    )
    
    # إرسال إشعار
    send_trade_open_notification(
        symbol, strategy_name, entry_price, stop_loss_price,
        target_price_1, target_price_2, float(quantity_dec),
        is_real, quality_score, df.iloc[-1].get('atr_percent', 0),
        notional_value, confirmations
    )
    
    logger.info(f"✅ [Signal] {symbol} signal created and saved successfully")

def save_signal_to_db(symbol: str, entry_price: float, trade_levels: Dict, strategy_name: str,
                     is_real: bool, quantity: float, signal_details: Dict, order_id: Optional[str] = None):
    """حفظ الإشارة في قاعدة البيانات والكاش"""
    if not check_db_connection() or not conn:
        logger.error(f"❌ [DB] Cannot save signal for {symbol}: DB connection failed")
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (
                    symbol, entry_price, target_price_1, target_price_2, stop_loss, status,
                    strategy_name, is_real_trade, quantity, initial_quantity, signal_details, order_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                symbol, float(entry_price), float(trade_levels['target_price_1']),
                float(trade_levels['target_price_2']), float(trade_levels['stop_loss']),
                'open', strategy_name, is_real, float(quantity), float(quantity),
                json.dumps(signal_details, cls=NpEncoder), order_id
            ))
            
            new_id = cur.fetchone()['id']
        
        conn.commit()
        logger.info(f"✅ [DB] Signal saved for {symbol} with ID: {new_id}")
        
        # تحضير البيانات للكاش
        signal_data = {
            'id': new_id,
            'symbol': symbol,
            'entry_price': float(entry_price),
            'target_price_1': float(trade_levels['target_price_1']),
            'target_price_2': float(trade_levels['target_price_2']),
            'stop_loss': float(trade_levels['stop_loss']),
            'status': 'open',
            'strategy_name': strategy_name,
            'is_real_trade': is_real,
            'quantity': float(quantity),
            'initial_quantity': float(quantity),
            'signal_details': signal_details,
            'order_id': order_id
        }
        
        # تحديث الكاش
        with signal_cache_lock:
            open_signals_cache[symbol] = signal_data
        
        # بث التحديث
        broadcast({"type": "new_signal", "payload": signal_data})
        logger.info(f"✅ [Cache & WS] Signal {new_id} for {symbol} added to cache and broadcasted")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ [DB] Failed to save signal for {symbol}: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

# --- التقارير الدورية ---
def send_daily_performance_report():
    """إرسال تقرير الأداء اليومي"""
    if not check_db_connection() or not conn:
        return
    
    try:
        with conn.cursor() as cur:
            today = datetime.now(timezone.utc).date()
            cur.execute("""
                SELECT COUNT(*) as total_trades,
                       SUM(CASE WHEN profit_percentage > 0 THEN 1 ELSE 0 END) as winning_trades,
                       AVG(profit_percentage) as avg_profit,
                       SUM(profit_percentage) as total_profit
                FROM signals
                WHERE closed_at::date = %s AND status = 'closed'
            """, (today,))
            
            stats = cur.fetchone()
            
            if not stats or stats['total_trades'] == 0:
                logger.info("[Daily Report] No trades today")
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
            
            message = (
                f"📈 *تقرير الأداء اليومي*\n\n"
                f"*التاريخ:* `{today.strftime('%Y-%m-%d')}`\n\n"
                f"*إجمالي الصفقات:* `{stats['total_trades']}`\n"
                f"*الصفقات الرابحة:* `{stats.get('winning_trades', 0) or 0}`\n"
                f"*نسبة الربح:* `{(stats.get('winning_trades', 0) / stats['total_trades'] * 100):.1f}%`\n"
                f"*متوسط الربح:* `{stats.get('avg_profit', 0):.2f}%`\n"
                f"*إجمالي الربح:* `{stats.get('total_profit', 0):.2f}%`\n\n"
            )
            
            if best_trade:
                message += (
                    f"🏆 *أفضل صفقة:*\n"
                    f"العملة: `{best_trade['symbol']}` | الربح: `{best_trade['profit_percentage']:.2f}%`\n\n"
                )
            
            if worst_trade:
                message += (
                    f" 📉 *أسوأ صفقة:*\n"
                    f"العملة: `{worst_trade['symbol']}` | الخسارة: `{worst_trade['profit_percentage']:.2f}%`\n\n"
                )
            
            send_enhanced_telegram_message(message, force=True)
            
    except Exception as e:
        logger.error(f"❌ [Daily Report] Error: {e}", exc_info=True)

def schedule_periodic_reports():
    """جدولة التقارير الدورية"""
    logger.info("Starting periodic reports scheduler...")
    while True:
        try:
            now = datetime.now(timezone.utc)
            if now.hour == 23 and now.minute == 59:
                send_daily_performance_report()
                time.sleep(61)
            if now.hour % 6 == 0 and now.minute == 0:
                send_market_state_notification()
                time.sleep(61)
            time.sleep(30)
        except Exception as e:
            logger.error(f"❌ [Periodic Reports] Scheduler error: {e}", exc_info=True)
            time.sleep(60)

def send_market_state_notification():
    """إرسال إشعار حالة السوق"""
    try:
        with market_state_lock:
            state = dict(current_market_state)
        
        if not state.get('trend_details_by_tf'):
            return
        
        message = "📊 *حالة السوق الحالية*\n\n"
        for tf in ['5m', '15m', '1h']:
            trend = state['trend_details_by_tf'].get(tf, {})
            if trend:
                trend_emoji = "🟢" if trend['trend'] == 'bullish' else "🔴" if trend['trend'] == 'bearish' else "⚪"
                message += f"*{tf}:* {trend_emoji} {trend['trend']} | ADX: {trend.get('adx', 0):.1f}\n"
        
        send_enhanced_telegram_message(message, force=True)
    except Exception as e:
        logger.error(f"❌ [Market State Notification] Error: {e}")

def start_periodic_reports():
    """بدء خيط التقارير الدورية"""
    reports_thread = Thread(target=schedule_periodic_reports, daemon=True)
    reports_thread.start()
    logger.info("✅ [Periodic Reports] Started scheduler thread")

def handle_socket_message(msg):
    """معالجة رسائل WebSocket مع تحسين لتقليل البث غير الضروري"""
    global live_prices
    try:
        if msg and 'e' in msg and msg['e'] == 'error':
            logger.error(f"❌ [WebSocket] Error: {msg['m']}")
            return
        
        if isinstance(msg, list):
            price_updates = {}
            with live_prices_lock:
                # فقط معالجة الرموز التي لدينا صفقات مفتوحة لها
                with signal_cache_lock:
                    monitored_symbols = set(open_signals_cache.keys())
                
                for ticker in msg:
                    if 's' in ticker and 'c' in ticker:
                        symbol = ticker['s']
                        
                        # تخطي الرموز غير المرغوبة لتقليل المعالجة
                        if symbol not in monitored_symbols:
                            continue
                            
                        try:
                            price = float(ticker['c'])
                            live_prices[symbol] = price
                            price_updates[symbol] = price
                        except (ValueError, TypeError):
                            logger.warning(f"[WebSocket] Invalid price for {symbol}: {ticker.get('c')}")
            
            if price_updates:
                broadcast({"type": "price_update", "payload": price_updates})
                logger.debug(f"[WebSocket] Broadcasted price updates for {len(price_updates)} symbols")
    except Exception as e:
        logger.error(f"❌ [WebSocket] Error processing message: {e}", exc_info=True)

def start_websocket():
    """بدء اتصال WebSocket مع Binance مع إدارة صحية"""
    global ws_manager
    ws_manager = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    ws_manager.start()
    ws_manager.start_ticker_socket(callback=handle_socket_message)
    logger.info("✅ [WebSocket] Subscribed to ticker stream")
    
    # بدء مراقبة الصحة
    ensure_ws_connection()

def get_exchange_info_map() -> None:
    """جلب معلومات الصرف من Binance"""
    global exchange_info_map
    try:
        logger.info("[API] Fetching exchange info...")
        exchange_info_map = {s['symbol']: s for s in client.get_exchange_info()['symbols']}
        logger.info(f"✅ [API] Exchange info loaded: {len(exchange_info_map)} symbols")
    except Exception as e:
        logger.error(f"❌ [API] Error fetching exchange info: {e}")

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """التحقق والتحقق من رموز التداول"""
    try:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        if not os.path.exists(file_path):
            logger.critical(f"❌ Symbol list file '{filename}' not found!")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        
        if not exchange_info_map:
            get_exchange_info_map()
        
        active = {s for s, info in exchange_info_map.items() 
                 if info.get('quoteAsset') == 'USDT' and info.get('status') == 'TRADING'}
        
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ Found {len(validated)} valid symbols for trading")
        return validated
    except Exception as e:
        logger.error(f"❌ [Symbols] Error: {e}")
        return []

# --- مسارات Flask ---
@app.route('/')
def dashboard():
    """صفحة لوحة التحكم الرئيسية"""
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/backtest')
def backtest_page():
    """صفحة الاختبار الخلفي"""
    return render_template_string(BACKTEST_TEMPLATE, STRATEGY_NAMES=STRATEGY_NAMES)

@app.route('/settings')
def settings_page():
    """صفحة الإعدادات"""
    with trade_amount_lock:
        trade_amount_min = FIXED_TRADE_AMOUNT_MIN_USDT
        trade_amount_max = FIXED_TRADE_AMOUNT_MAX_USDT
    with trading_mode_lock: is_paper_mode = paper_trading_mode
    with min_quality_lock: min_quality = MIN_SIGNAL_QUALITY
    with strategy_config_lock:
        strategy_config = {
            "enable_ema_filter": ENABLE_EMA_FILTER,
            "enable_macd_confirmation": ENABLE_MACD_CONFIRMATION,
            "enable_mfi_filter": ENABLE_MFI_FILTER,
            "enable_candlestick_patterns": ENABLE_CANDLESTICK_PATTERNS,
            "required_confirmations": REQUIRED_CONFIRMATIONS
        }
    
    return render_template_string(SETTINGS_TEMPLATE,
                                  trade_amount_min=trade_amount_min,
                                  trade_amount_max=trade_amount_max,
                                  MAX_OPEN_TRADES=MAX_OPEN_TRADES,
                                  min_quality=min_quality,
                                  is_paper_mode=is_paper_mode,
                                  strategy_config=strategy_config)

@app.route('/api/dashboard_data')
def dashboard_data():
    """endpoint جلب بيانات لوحة التحكم"""
    try:
        payload = get_dashboard_payload()
        logger.info(f"[API] Dashboard data sent: {len(open_signals_cache)} open signals")
        return jsonify(payload)
    except Exception as e:
        logger.error(f"❌ [API Error] Dashboard data: {e}", exc_info=True)
        return jsonify({"error": "Failed to load dashboard data"}), 500

@app.route('/toggle_trading', methods=['POST'])
def toggle_trading():
    """تبديل حالة التداول"""
    global is_trading_enabled
    with trading_status_lock:
        is_trading_enabled = not is_trading_enabled
    
    status_msg = "enabled" if is_trading_enabled else "disabled"
    log_and_notify("info", f"Trading has been {status_msg}.", "TRADING_STATUS")
    logger.info(f"[Trading] Status changed to: {is_trading_enabled}")
    
    return jsonify({"status": "success", "trading_enabled": is_trading_enabled})

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """تحديث الإعدادات العامة"""
    try:
        data = request.json
        logger.info(f"[API] Updating settings: {list(data.keys())}")
        
        if 'FIXED_TRADE_AMOUNT_MIN_USDT' in data and 'FIXED_TRADE_AMOUNT_MAX_USDT' in data:
            with trade_amount_lock:
                global FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT
                FIXED_TRADE_AMOUNT_MIN_USDT = float(data['FIXED_TRADE_AMOUNT_MIN_USDT'])
                FIXED_TRADE_AMOUNT_MAX_USDT = float(data['FIXED_TRADE_AMOUNT_MAX_USDT'])
                broadcast({
                    "type": "trade_amount_update",
                    "payload": {
                        "min": FIXED_TRADE_AMOUNT_MIN_USDT,
                        "max": FIXED_TRADE_AMOUNT_MAX_USDT
                    }
                })
        
        if 'MAX_OPEN_TRADES' in data:
            global MAX_OPEN_TRADES
            MAX_OPEN_TRADES = int(data['MAX_OPEN_TRADES'])
        
        if 'paper_trading_mode' in data:
            with trading_mode_lock:
                global paper_trading_mode
                paper_trading_mode = bool(data['paper_trading_mode'])
        
        save_settings_to_redis()
        logger.info("✅ [Settings] Updated successfully")
        return jsonify({"success": True, "message": "Settings updated"})
    
    except Exception as e:
        logger.error(f"❌ [Settings] Update failed: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/strategy_config', methods=['POST'])
def update_strategy_config():
    """تحديث إعدادات الاستراتيجية"""
    try:
        data = request.json
        logger.info(f"[API] Updating strategy config: {list(data.keys())}")
        
        with strategy_config_lock:
            global ENABLE_EMA_FILTER, ENABLE_MACD_CONFIRMATION, ENABLE_MFI_FILTER, ENABLE_CANDLESTICK_PATTERNS, REQUIRED_CONFIRMATIONS
            ENABLE_EMA_FILTER = bool(data.get('enable_ema_filter', True))
            ENABLE_MACD_CONFIRMATION = bool(data.get('enable_macd_confirmation', True))
            ENABLE_MFI_FILTER = bool(data.get('enable_mfi_filter', True))
            ENABLE_CANDLESTICK_PATTERNS = bool(data.get('enable_candlestick_patterns', True))
            REQUIRED_CONFIRMATIONS = int(data.get('required_confirmations', 3))
        
        save_settings_to_redis()
        logger.info("✅ [Strategy Config] Updated successfully")
        return jsonify({"success": True, "message": "Strategy config updated"})
    
    except Exception as e:
        logger.error(f"❌ [Strategy Config] Update failed: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/health')
def api_health():
    """endpoint فحص صحة النظام"""
    try:
        with trading_status_lock: trading_enabled = is_trading_enabled
        with trading_mode_lock: is_paper = paper_trading_mode
        
        health_data = {
            "status": "ok",
            "trading_enabled": trading_enabled,
            "mode": "PAPER" if is_paper else "REAL",
            "open_signals": len(open_signals_cache),
            "open_signals_list": list(open_signals_cache.keys()),
            "ws": {"connected": len(ws_clients) > 0},
            "db": {"connected": check_db_connection()},
            "redis": {"connected": redis_client is not None},
            "active_strategies": len(STRATEGY_NAMES)
        }
        
        logger.info(f"[Health] System health check: {health_data}")
        return jsonify(health_data), 200
    
    except Exception as e:
        logger.error(f"❌ [Health] Check failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/open_signals')
def get_open_signals():
    """endpoint جلب الإشارات المفتوحة"""
    if not check_db_connection():
        logger.error("[API] Cannot fetch signals: DB connection failed")
        return jsonify({"error": "Database connection failed"}), 500
    
    sort_by = request.args.get('sort', 'id')
    allowed_sort_fields = ['id', 'symbol', 'entry_price', 'strategy_name']
    if sort_by not in allowed_sort_fields:
        sort_by = 'id'
    
    try:
        logger.info(f"[API] Fetching open signals, sort by: {sort_by}")
        
        with conn.cursor() as cur:
            query = sql.SQL("""
                SELECT id, symbol, entry_price, target_price_1, target_price_2,
                       stop_loss, strategy_name, is_real_trade, quantity,
                       signal_details, status
                FROM signals
                WHERE status IN ('open', 'updated')
                ORDER BY {} DESC
            """).format(sql.Identifier(sort_by))
            
            cur.execute(query)
            signals = cur.fetchall()
        
        signals_list = []
        for s in signals:
            signal_dict = dict(s)
            if isinstance(signal_dict.get('signal_details'), str):
                try:
                    signal_dict['signal_details'] = json.loads(signal_dict['signal_details'])
                except:
                    signal_dict['signal_details'] = {}
            signals_list.append(signal_dict)
        
        logger.info(f"✅ [API] Returned {len(signals_list)} open signals")
        
        # تحديث الكاش في الخلفية
        def update_cache():
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in signals_list:
                    open_signals_cache[signal['symbol']] = signal
        
        Thread(target=update_cache, daemon=True).start()
        
        return jsonify({"signals": signals_list})
    
    except Exception as e:
        logger.error(f"❌ [API] Error fetching signals: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance_metrics')
def get_performance_metrics():
    """endpoint جلب مقاييس الأداء"""
    cache_key = "performance_metrics_30d"
    
    if redis_client:
        try:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                return jsonify(json.loads(cached_data))
        except:
            pass
    
    if not check_db_connection():
        return jsonify({"error": "Database connection failed"}), 500
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit_percentage > 0 THEN 1 ELSE 0 END) as winning_trades,
                    AVG(profit_percentage) as avg_profit
                FROM signals
                WHERE status = 'closed' AND closed_at >= NOW() - INTERVAL '30 days'
            """)
            metrics = cur.fetchone()
        
        total_trades = metrics['total_trades'] or 0
        winning_trades = metrics['winning_trades'] or 0
        
        result = {
            "total_trades": total_trades,
            "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            "avg_profit": metrics['avg_profit'] or 0,
            "max_drawdown": 0
        }
        
        if redis_client:
            try:
                redis_client.setex(cache_key, 300, json.dumps(result, cls=NpEncoder))
            except:
                pass
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"❌ [Metrics] Error: {e}")
        return jsonify({"error": str(e)}), 500

@sock.route('/ws')
def ws(ws_client):
    """WebSocket connection handler"""
    logger.info(f"[WebSocket] Client connected from {request.remote_addr}")
    
    with ws_clients_lock:
        ws_clients.append(ws_client)
    
    try:
        # إرسال بيانات الترحيب
        ws_client.send(json.dumps({
            "type": "connection_established",
            "client_count": len(ws_clients)
        }, cls=NpEncoder))
        
        # إرسال الإشارات الحالية
        with signal_cache_lock:
            signals_snapshot = list(open_signals_cache.values())
        
        if signals_snapshot:
            ws_client.send(json.dumps({
                "type": "initial_signals",
                "payload": signals_snapshot
            }, cls=NpEncoder))
        
        logger.info(f"[WebSocket] Sent initial data to new client")
        
        # الانتظار مع ping
        while True:
            ws_client.send(json.dumps({"type": "ping"}, cls=NpEncoder))
            time.sleep(30)
    
    except Exception as e:
        logger.info(f"[WebSocket] Client disconnected: {e}")
    
    finally:
        with ws_clients_lock:
            if ws_client in ws_clients:
                ws_clients.remove(ws_client)
                logger.info(f"[WebSocket] Client removed. Total clients: {len(ws_clients)}")

# --- إغلاق الصفقات ---
def close_trade_manually(signal_id: int, closing_price: Optional[float] = None) -> bool:
    """إغلاق صفقة يدوياً"""
    logger.info(f"[Manual Close] Requested for signal ID: {signal_id}")
    
    # البحث في الكاش
    with signal_cache_lock:
        signal_to_close = None
        for symbol, signal in open_signals_cache.items():
            if signal['id'] == signal_id:
                signal_to_close = signal
                break
    
    if not signal_to_close:
        logger.warning(f"[Manual Close] Signal {signal_id} not found in cache")
        return False
    
    symbol = signal_to_close['symbol']
    
    # الحصول على سعر الإغلاق
    if closing_price is None:
        with live_prices_lock:
            closing_price = live_prices.get(symbol)
        
        if closing_price is None:
            logger.error(f"[Manual Close] No current price for {symbol}")
            return False
    
    logger.info(f"[Manual Close] Closing {symbol} at {closing_price}")
    close_signal(signal_to_close, closing_price, "manual_close")
    
    return True

@app.route('/api/close_trade/<int:signal_id>', methods=['POST'])
def api_close_trade(signal_id):
    """endpoint إغلاق صفقة يدوياً"""
    data = request.get_json(silent=True) or {}
    closing_price = data.get('closing_price')
    
    logger.info(f"[API] Close trade request for ID: {signal_id}")
    
    success = close_trade_manually(signal_id, closing_price)
    
    if success:
        return jsonify({"success": True, "message": "Trade close command executed"})
    else:
        return jsonify({"success": False, "message": "Trade not found or failed"}), 404

def update_signal_in_db(signal_id, updates):
    """تحديث بيانات الإشارة في قاعدة البيانات"""
    if not check_db_connection() or not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            set_clause = sql.SQL(', ').join(
                sql.SQL("{} = %s").format(sql.Identifier(k)) for k in updates.keys()
            )
            values = list(updates.values())
            query = sql.SQL("UPDATE signals SET {} WHERE id = %s").format(set_clause)
            values.append(signal_id)
            cur.execute(query, values)
        
        conn.commit()
        logger.info(f"✅ [DB] Signal {signal_id} updated: {list(updates.keys())}")
        
        # تحديث الكاش
        with signal_cache_lock:
            symbol = None
            for sym, sig in open_signals_cache.items():
                if sig['id'] == signal_id:
                    symbol = sym
                    break
            
            if symbol:
                open_signals_cache[symbol].update(updates)
                if 'signal_details' in updates and isinstance(updates['signal_details'], str):
                    open_signals_cache[symbol]['signal_details'] = json.loads(updates['signal_details'])
                
                broadcast({"type": "signal_update", "payload": open_signals_cache[symbol]})
        
        return True
    
    except Exception as e:
        logger.error(f"❌ [DB] Failed to update signal {signal_id}: {e}")
        if conn: conn.rollback()
        return False

def close_signal(signal: Dict, closing_price: float, reason: str):
    """إغلاق إشارة وتجديد الرصيد"""
    global usdt_balance
    
    symbol = signal['symbol']
    signal_id = signal['id']
    entry_price = signal['entry_price']
    is_real = signal.get('is_real_trade', False)
    
    logger.info(f"[Close] Processing {symbol} (ID: {signal_id}) at {closing_price}, reason: {reason}")
    
    # التحقق من وجود الإشارة في الكاش
    with signal_cache_lock:
        if symbol not in open_signals_cache or open_signals_cache[symbol]['id'] != signal_id:
            logger.warning(f"[Close] Signal {signal_id} not in cache or mismatch")
            return
    
    # تنفيذ أمر البيع للصفقات الحقيقية
    if is_real:
        try:
            asset = symbol.replace("USDT", "")
            asset_balance_info = client.get_asset_balance(asset=asset)
            available_on_exchange = Decimal(asset_balance_info.get('free', '0.0'))
            
            if available_on_exchange > 0:
                adjusted_qty = adjust_quantity_to_lot_size(symbol, float(available_on_exchange))
                if adjusted_qty and adjusted_qty > 0:
                    formatted_sell_quantity = get_formatted_quantity(symbol, adjusted_qty)
                    logger.info(f"💰 [Real Close] SELL {formatted_sell_quantity} of {symbol}")
                    client.create_order(
                        symbol=symbol,
                        side=Client.SIDE_SELL,
                        type=Client.ORDER_TYPE_MARKET,
                        quantity=formatted_sell_quantity
                    )
                else:
                    logger.warning(f"[Real Close] Adjusted quantity is zero for {symbol}")
            else:
                logger.warning(f"[Real Close] No balance to sell for {symbol}")
        
        except BinanceAPIException as e:
            logger.error(f"❌ [Real Close] Binance API Error: {e}")
            send_enhanced_telegram_message(f"❌ *خطأ إغلاق {symbol}*\n`{e}`", force=True)
        except Exception as e:
            logger.error(f"❌ [Real Close] Critical error: {e}", exc_info=True)
    
    # حساب الربح
    profit = ((closing_price - entry_price) / entry_price) * 100
    
    # تحديث قاعدة البيانات
    update_signal_in_db(signal['id'], {
        "status": "closed",
        "closing_price": closing_price,
        "closed_at": datetime.now(timezone.utc),
        "profit_percentage": profit,
        "closing_reason": reason
    })
    
    # إزالة من الكاش
    with signal_cache_lock:
        if symbol in open_signals_cache:
            del open_signals_cache[symbol]
    
    # بث الإغلاق
    broadcast({
        "type": "trade_closed",
        "payload": {
            "signal_id": signal_id,
            "symbol": symbol,
            "reason": reason,
            "profit": profit
        }
    })
    
    # تسجيل وإشعار
    trade_type = "حقيقية" if is_real else "ورقية"
    result_emoji = "✅" if profit >= 0 else "🔻"
    
    log_and_notify("info", f"Closed {trade_type} trade {symbol}: {profit:.2f}%", "TRADE_CLOSED")
    
    settings = get_notification_settings()
    if (profit >= settings['min_profit_notification'] or
        profit <= settings['max_loss_notification'] or
        reason == "manual_close"):
        
        send_enhanced_telegram_message(
            f"{result_emoji} *إغلاق صفقة {trade_type} {symbol}*\n"
            f"*السبب:* {reason}\n"
            f"*الربح:* `{profit:.2f}%`",
            force=True
        )
    
    logger.info(f"✅ [Close] {symbol} closed successfully with {profit:.2f}% profit")

# --- إدارة الصفقات ---
def trade_management_loop():
    """الحلقة الرئيسية لإدارة الصفقات المفتوحة"""
    logger.info("🚀 [Trade Manager] Starting trade management loop...")
    cycle_count = 0
    
    while True:
        try:
            with signal_cache_lock:
                if not open_signals_cache:
                    if cycle_count % 30 == 0:
                        logger.debug("[Trade Manager] No open signals to monitor")
                    time.sleep(2)
                    cycle_count += 1
                    continue
                
                signals_to_monitor = list(open_signals_cache.values())
            
            logger.debug(f"[Trade Manager] Monitoring {len(signals_to_monitor)} signals")
            
            for signal in signals_to_monitor:
                symbol = signal['symbol']
                
                # الحصول على السعر الحالي
                with live_prices_lock:
                    current_price = live_prices.get(symbol)
                
                if not current_price:
                    logger.debug(f"[Trade Manager] No price for {symbol}")
                    continue
                
                # تحليل تفاصيل الإشارة
                details = signal.get('signal_details', {})
                if isinstance(details, str):
                    try:
                        details = json.loads(details)
                    except:
                        details = {}
                
                entry_price = float(signal.get('entry_price', 0))
                stop_loss = float(signal.get('stop_loss', 0))
                tp1 = float(signal.get('target_price_1') or 0)
                tp2 = float(signal.get('target_price_2') or 0)
                initial_quantity = float(signal.get('initial_quantity') or 0)
                remaining_qty = float(signal.get('quantity') or 0)
                
                # التحقق من مستويات TP/SL
                if stop_loss and current_price <= stop_loss:
                    logger.info(f"[Trade Manager] {symbol} hit SL at {current_price}")
                    close_signal(signal, stop_loss, "SL_HIT")
                    continue
                
                if tp2 and current_price >= tp2:
                    logger.info(f"[Trade Manager] {symbol} hit TP2 at {current_price}")
                    close_signal(signal, tp2, "TP2_HIT")
                    continue
                
                if tp1 and not details.get('tp1_done') and remaining_qty > 0 and current_price >= tp1:
                    logger.info(f"[Trade Manager] {symbol} hit TP1 at {current_price}")
                    
                    # إغلاق جزئي
                    part_qty_to_close = initial_quantity * 0.5
                    
                    if signal.get('is_real_trade'):
                        adjusted_qty = adjust_quantity_to_lot_size(symbol, part_qty_to_close)
                        if adjusted_qty and adjusted_qty > 0:
                            try:
                                formatted_sell_quantity = get_formatted_quantity(symbol, adjusted_qty)
                                logger.info(f" 💰 [Partial Close] SELL {formatted_sell_quantity} of {symbol}")
                                client.create_order(
                                    symbol=symbol,
                                    side=Client.SIDE_SELL,
                                    type=Client.ORDER_TYPE_MARKET,
                                    quantity=formatted_sell_quantity
                                )
                            except Exception as e:
                                logger.error(f"❌ [Partial Close] Error: {e}")
                    
                    # تحديث وقف الخسارة
                    new_sl = max(stop_loss, entry_price)
                    new_remaining_qty = remaining_qty - part_qty_to_close
                    
                    details['tp1_done'] = True
                    details['partial_close_price'] = current_price
                    
                    update_signal_in_db(signal['id'], {
                        "quantity": new_remaining_qty,
                        "stop_loss": new_sl,
                        "status": "updated",
                        "signal_details": json.dumps(details)
                    })
                    
                    send_enhanced_telegram_message(
                        f"🥇 *تم تحقيق الهدف الأول* لـ `{symbol}`\n"
                        f"تم إغلاق 50% من العقد\n"
                        f"تحريك الوقف إلى: {new_sl:.6f}",
                        force=True
                    )
                    
                    continue
                
                # تتبع الوقف المتحرك
                profit_pct = ((current_price - entry_price) / max(entry_price, 1e-8)) * 100 if entry_price else 0
                
                if not details.get('trailing_active') and profit_pct >= TRAILING_STOP_ACTIVATION_PROFIT_PERCENT:
                    details['trailing_active'] = True
                    update_signal_in_db(signal['id'], {
                        "signal_details": json.dumps(details)
                    })
                    
                    send_enhanced_telegram_message(
                        f"📈 *تفعيل الوقف المتحرك* لـ `{symbol}`\n"
                        f"الربح الحالي: `{profit_pct:.2f}%`",
                        force=True
                    )
                
                if details.get('trailing_active'):
                    new_sl = max(stop_loss, current_price * (1 - (TRAILING_STOP_ACTIVATION_PROFIT_PERCENT / 200)))
                    if new_sl > stop_loss:
                        update_signal_in_db(signal['id'], {"stop_loss": new_sl})
                        logger.debug(f"[Trailing] {symbol} SL updated to {new_sl}")
            
            time.sleep(1)
            cycle_count += 1
            
        except Exception as e:
            logger.error(f"❌ [Trade Manager] Loop error: {e}", exc_info=True)
            time.sleep(5)

# --- تحديث حالة السوق ---
def update_market_state():
    """تحديث حالة السوق مع تحديد النظام السوقي"""
    global current_market_state
    
    try:
        btc_df = fetch_historical_data(BTC_SYMBOL, '1h', days=10)
        if btc_df is None or len(btc_df) < 200:
            logger.warning("[Market State] Insufficient BTC data")
            return
        
        btc_df = calculate_all_features(btc_df)
        last_btc = btc_df.iloc[-1]
        
        # تحديد الاتجاه العام لـ BTC
        btc_trend = "sideways"
        if last_btc['close'] > last_btc['ema200'] and last_btc['macd_hist'] > 0:
            btc_trend = "bullish"
        elif last_btc['close'] < last_btc['ema200'] and last_btc['macd_hist'] < 0:
            btc_trend = "bearish"
        
        # تحليل الأطر الزمنية المختلفة
        trend_details = {}
        adx_values = []
        
        for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
            try:
                tf_df = fetch_historical_data(BTC_SYMBOL, tf, days=15)
                if tf_df is not None and len(tf_df) >= 50:
                    tf_df = calculate_all_features(tf_df)
                    last_tf = tf_df.iloc[-1]
                    trend_strength = last_tf.get('adx', 0)
                    adx_values.append(trend_strength)
                    
                    tf_trend = "sideways"
                    if trend_strength > 25:
                        if last_tf['close'] > last_tf['ema50']:
                            tf_trend = "bullish"
                        else:
                            tf_trend = "bearish"
                    
                    trend_details[tf] = {
                        "trend": tf_trend,
                        "adx": trend_strength,
                        "rsi": last_tf.get('rsi', 50)
                    }
            except Exception as e:
                logger.error(f"[Market State] Error analyzing {tf}: {e}")
        
        # تحديد النظام السوقي الكلي
        avg_adx = np.mean(adx_values) if adx_values else 0
        
        if avg_adx > 30:
            overall_trend = "volatile"
        elif avg_adx > 20:
            overall_trend = btc_trend  # توافق مع BTC
        else:
            overall_trend = "sideways"
        
        with market_state_lock:
            current_market_state = {
                "btc_trend": btc_trend,
                "overall_trend": overall_trend,
                "btc_price": last_btc['close'],
                "btc_adx": last_btc.get('adx', 0),
                "btc_rsi": last_btc.get('rsi', 50),
                "trend_details_by_tf": trend_details,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
        
        broadcast({"type": "market_state_update", "payload": current_market_state})
        logger.debug("[Market State] Updated and broadcasted")
    
    except Exception as e:
        logger.error(f"❌ [Market State] Update error: {e}", exc_info=True)

def start_market_state_updater():
    """بدء خيط تحديث حالة السوق"""
    def update_loop():
        while True:
            try:
                update_market_state()
                time.sleep(300)  # كل 5 دقائق
            except Exception as e:
                logger.error(f"❌ [Market State Updater] Error: {e}")
                time.sleep(60)
    
    thread = Thread(target=update_loop, daemon=True)
    thread.start()
    logger.info("✅ [Market State] Started updater thread")

# --- تحديث الرصيد ---
def update_balance():
    """تحديث رصيد USDT الحقيقي"""
    global usdt_balance
    
    try:
        balance_info = client.get_asset_balance(asset='USDT')
        with balance_lock:
            usdt_balance = float(balance_info['free'])
        logger.info(f"💰 [Balance] Updated: ${usdt_balance:.2f}")
    except Exception as e:
        logger.error(f"❌ [Balance] Update failed: {e}")

def update_balance_loop():
    """الحلقة الدورية لتحديث الرصيد"""
    logger.info("🚀 [Balance] Starting balance updater...")
    
    while True:
        try:
            update_balance()
        except Exception as e:
            logger.error(f"❌ [Balance Loop] Error: {e}", exc_info=True)
        
        time.sleep(300)  # كل 5 دقائق

# --- الحلقة الرئيسية للبوت ---
def main_bot_loop():
    """الحلقة الرئيسية لإنشاء الإشارات"""
    logger.info("🚀 [Main Loop] Starting signal scanning loop (5-minute cycle)...")
    scan_count = 0
    
    register_strategies()  # تسجيل الاستراتيجيات
    
    while True:
        try:
            # الانتظار حتى بداية شمعة جديدة
            while True:
                now = datetime.now(timezone.utc)
                seconds_until_next_candle = (5 - (now.minute % 5)) * 60 - now.second
                
                with trading_status_lock:
                    if not is_trading_enabled:
                        logger.info("[Main Loop] Trading disabled, waiting...")
                        time.sleep(10)
                        continue
                
                if seconds_until_next_candle <= 1:
                    time.sleep(1)
                    break
                
                time.sleep(1)
            
            logger.info("="*50)
            logger.info(f"[Scan Cycle {scan_count}] Starting scan of {len(validated_symbols_to_scan)} symbols")
            logger.info("="*50)
            
            # فحص كل رمز
            for i, symbol in enumerate(validated_symbols_to_scan):
                logger.debug(f"[Scan] Processing {symbol} ({i+1}/{len(validated_symbols_to_scan)})")
                
                # التحقق من الحد الأقصى للصفقات المفتوحة
                with signal_cache_lock:
                    if len(open_signals_cache) >= MAX_OPEN_TRADES:
                        logger.info(f"[Scan] Max open trades reached: {MAX_OPEN_TRADES}")
                        break
                
                # التحقق من وجود صفقة مفتوحة مسبقاً
                with signal_cache_lock:
                    if symbol in open_signals_cache:
                        logger.debug(f"[Scan] {symbol} already has open signal")
                        continue
                
                try:
                    # جلب بيانات السوق
                    df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df is None or len(df) < 200:
                        if df is not None:
                            log_rejection(symbol, "Insufficient Historical Data", {
                                "len": len(df)
                            })
                        continue
                    
                    # حساب المؤشرات
                    df_featured = calculate_all_features(df)
                    df_featured.name = symbol
                    
                    # تحليل MTF Trend
                    mtf_trend = get_mtf_trend(symbol)
                    
                    # اختيار الاستراتيجية الذكية
                    selected_strategy = select_optimal_strategy(df_featured, mtf_trend)
                    
                    if selected_strategy:
                        logger.info(f"⭐ [Signal Detected] {symbol} via {selected_strategy}")
                        create_trade_signal(symbol, df_featured, selected_strategy)
                    else:
                        logger.debug(f"[Scan] {symbol} no suitable strategy")
                
                except Exception as e:
                    logger.error(f"❌ [Scan] Error processing {symbol}: {e}", exc_info=True)
                    continue
            
            scan_count += 1
            logger.info(f"[Scan Cycle {scan_count}] Completed")
            
        except Exception as e:
            logger.error(f"❌ [Main Loop] Critical error: {e}", exc_info=True)
            time.sleep(60)

def get_mtf_trend(symbol: str) -> Dict[str, str]:
    """تحليل الاتجاه متعدد الأطر الزمنية"""
    trends = {}
    timeframes = {'5m': 7, '15m': 10}
    
    for tf, days in timeframes.items():
        try:
            df = fetch_historical_data(symbol, tf, days)
            if df is not None and len(df) >= 50:
                df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
                df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
                last = df.iloc[-1]
                
                tf_trend = "sideways"
                if last['close'] > last['ema50'] and last['ema21'] > last['ema50']:
                    tf_trend = "bullish"
                elif last['close'] < last['ema50'] and last['ema21'] < last['ema50']:
                    tf_trend = "bearish"
                
                trends[tf] = tf_trend
        except Exception as e:
            logger.warning(f"[MTF] Could not determine {tf} trend for {symbol}: {e}")
            trends[tf] = 'unknown'
    
    return trends

# --- الاختبار الخلفي ---
@app.route('/api/run_backtest', methods=['POST'])
def api_run_backtest():
    """endpoint تشغيل الاختبار الخلفي"""
    try:
        data = request.json
        strategy = data.get('strategy')
        symbol = data.get('symbol', '').upper()
        days = int(data.get('days', 90))

        if not all([strategy, symbol, days]):
            return jsonify({"error": "Missing parameters"}), 400

        logger.info(f"[Backtest] Starting for {strategy} on {symbol} for {days} days")
        results = backtest_strategy(strategy, symbol, days)
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"❌ [Backtest API] Error: {e}", exc_info=True)
        return jsonify({"error": "Internal error occurred"}), 500

def backtest_strategy(strategy_name, symbol, days=90):
    """تنفيذ الاختبار الخلفي للاستراتيجية"""
    logger.info(f"[Backtest] Starting backtest: {strategy_name} on {symbol} for {days} days")
    
    df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, days)
    if df is None or len(df) < 200:
        logger.error(f"[Backtest] Insufficient data for {symbol}")
        return {"error": "Insufficient historical data"}
    
    df = calculate_all_features(df)
    
    results = []
    active_trade = None
    equity_curve = [1000]
    backtest_trade_amount = 10.0

    # التأكد من وجود الاستراتيجية
    with strategies_lock:
        check_strategy = strategy_registry.get(strategy_name)
    
    if not check_strategy:
        return {"error": f"Strategy '{strategy_name}' not found"}

    dummy_mtf = {'5m': 'bullish', '15m': 'bullish'}

    for i in range(200, len(df)):
        current_candle = df.iloc[i]
        
        # إدارة الصفقة المفتوحة
        if active_trade:
            exit_price = None
            exit_reason = None
            
            if current_candle['low'] <= active_trade['stop_loss']:
                exit_price = active_trade['stop_loss']
                exit_reason = 'Stop Loss'
            elif current_candle['high'] >= active_trade['target_price_2']:
                exit_price = active_trade['target_price_2']
                exit_reason = 'Target 2'
            elif current_candle['high'] >= active_trade['target_price_1']:
                exit_price = active_trade['target_price_1']
                exit_reason = 'Target 1'
            
            if exit_price:
                profit = (exit_price - active_trade['entry_price']) * active_trade['quantity']
                equity_curve.append(equity_curve[-1] + profit)
                
                active_trade.update({
                    'exit_time': current_candle.name.isoformat(),
                    'exit_price': exit_price,
                    'profit_percent': ((exit_price - active_trade['entry_price']) / active_trade['entry_price']) * 100,
                    'exit_reason': exit_reason
                })
                
                results.append(active_trade)
                active_trade = None

        # البحث عن إشارات جديدة
        if not active_trade:
            df_slice = df.iloc[:i]
            df_slice.name = symbol
            
            if check_strategy(df_slice, dummy_mtf):
                entry_price = current_candle['open']
                sl = calculate_dynamic_stop_loss(df_slice, entry_price, strategy_name)
                tp1, tp2 = calculate_dynamic_take_profit(df_slice, entry_price, sl, strategy_name)
                
                if sl >= entry_price:
                    continue

                quantity = backtest_trade_amount / entry_price
                
                active_trade = {
                    'entry_time': current_candle.name.isoformat(),
                    'entry_price': entry_price,
                    'stop_loss': sl,
                    'target_price_1': tp1,
                    'target_price_2': tp2,
                    'quantity': quantity
                }

    # حساب المقاييس
    total_trades = len(results)
    
    if total_trades == 0:
        return {"error": "No trades executed"}

    wins = [r for r in results if r['profit_percent'] > 0]
    win_rate = (len(wins) / total_trades) * 100
    
    total_profit = sum(r['profit_percent'] for r in wins)
    total_loss = abs(sum(r['profit_percent'] for r in results if r['profit_percent'] <= 0))
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    avg_profit = sum(r['profit_percent'] for r in results) / total_trades

    return {
        'strategy': strategy_name,
        'symbol': symbol,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'profit_factor': profit_factor,
        'results': results,
        'equity_curve': equity_curve
    }

# --- HTML Templates ---
DASHBOARD_TEMPLATE = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>لوحة التحكم - بوت 5 دقائق (V35.0.0)</title>
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
input[type=number] { -moz-appearance: textfield; }
input[type=number]::-webkit-inner-spin-button, input[type=number]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
.confirmations-grid {display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; font-size: 11px; margin-top: 6px;}
.confirmation-item {display: flex; align-items: center; gap: 4px;}
.confirmation-item .indicator {width: 8px; height: 8px; border-radius: 50%;}
.confirmation-item .indicator.true {background: var(--ok);}
.confirmation-item .indicator.false {background: var(--bad);}
.signal-error {background: #3a0f14; border-color: #ff4757; color: #ff6b7a; padding: 10px; border-radius: 8px; margin: 10px 0;}
.debug-info {font-size: 10px; color: var(--muted); margin-top: 8px; opacity: 0.7;}
.strategy-badge {font-size: 10px; background: #1e2c52; padding: 2px 6px; border-radius: 4px; margin-left: 4px;}
</style>
</head>
<body>
<div class="container">
  <header><h1>لوحة التحكم • بوت 5 دقائق V35.0.0</h1><div class="badge" id="serverTime">—</div></header>
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
            <div id="loading-indicator" class="loading-spinner" style="display: none;"></div>
            <div id="signals" class="signals-grid"></div>
            <div id="debug-info" class="debug-info"></div>
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
            <div>الرصيد الفعلي (USDT):</div><div id="balance">—</div><div>عدد الصفقات:</div><div id="openCount">—</div>
          </div>
          <div class="kv" style="margin-top:8px">
            <div>حالة الاتصال:</div><div id="connectionStatus">⚪ جاري الاتصال...</div>
          </div>
        </div>
      </div>
      <div class="card">
        <h2>حالة السوق</h2>
        <div class="card-body">
            <div id="marketTrends"><div class="loading-spinner"></div></div>
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
              <input type="range" id="qualityFilter" min="30" max="90" value="70" class="slider">
              <span id="qualityValue">70</span>
            </div>
          </div>
          <div class="kv" style="margin-top: 12px;">
            <div>قيمة الصفقة (الحقيقية):</div>
            <div id="tradeAmountDisplay">$4.5 - $6.5</div>
          </div>
           <div class="kv" style="margin-top: 12px;">
            <div>قيمة الصفقة (الورقية):</div>
            <div>$10.0 (ثابتة)</div>
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
const qsa = s => document.querySelectorAll(s);
let lastPrices = {};
let performanceChartInstance = null;
let openSignals = {};
let ws = null;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;

const debounce = (func, delay) => {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), delay);
    };
};

function fmt(n){ 
    if (n == null || isNaN(n)) return '—';
    return (+n).toLocaleString('en-US', {maximumFractionDigits: 6});
}

function showLoading(show = true) {
    qs('#loading-indicator').style.display = show ? 'block' : 'none';
}

function showNotification(message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);
    // يمكن إضافة نظام إشعارات مرئي هنا
}

function updateConnectionStatus(status, message) {
    const el = qs('#connectionStatus');
    if (!el) return;
    
    let emoji = '⚪';
    let color = '#8aa0c8';
    
    if (status === 'connected') {
        emoji = '🟢';
        color = '#15c46a';
    } else if (status === 'error') {
        emoji = '🔴';
        color = '#ff4757';
    } else if (status === 'warning') {
        emoji = '🟡';
        color = '#ff9f1a';
    }
    
    el.innerHTML = `${emoji} ${message}`;
    el.style.color = color;
}

function closeTrade(signalId) {
    if (!confirm('هل أنت متأكد من رغبتك في إغلاق هذه الصفقة يدويًا؟')) {
        return;
    }
    
    showLoading(true);
    
    fetch(`/api/close_trade/${signalId}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({})
    })
    .then(res => res.ok ? res.json() : res.json().then(err => { throw new Error(err.message || 'Server error') }))
    .then(data => {
        showLoading(false);
        if (data.success) {
            showNotification('تم إرسال أمر الإغلاق بنجاح.', 'success');
        } else {
            showNotification(`فشل إغلاق الصفقة: ${data.message}`, 'error');
        }
    })
    .catch(err => {
        showLoading(false);
        showNotification(`حدث خطأ: ${err.message}`, 'error');
        console.error(err);
    });
}

function renderSignal(signal) {
    try {
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
        const strategyName = (signal.strategy_name || '').replace(/_/g, " ").replace("Strategy", "");
        
        // عرض التأكيدات
        const confirmations = signal.signal_details?.confirmations || {};
        const confirmationsHtml = Object.entries(confirmations).map(([key, value]) => 
            `<div class="confirmation-item">
                <div class="indicator ${value}"></div>
                <span>${key.replace('_', ' ')}</span>
            </div>`
        ).join('');
        
        return `
            <div class="signal" id="signal-${signal.id}" data-symbol="${signal.symbol}">
                <div>
                    <div class="sig-title">${signal.symbol} <span class="strategy-badge">${strategyName}</span></div>
                    <div class="sig-meta"><span style="color: ${qualityColor}; font-weight: bold;">⭐ ${qualityScore}/100</span></div>
                    <div class="confirmations-grid">${confirmationsHtml}</div>
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
    } catch (err) {
        console.error('Error rendering signal:', err, signal);
        return `<div class="signal-error">خطأ في عرض الإشارة: ${signal.symbol || 'Unknown'}</div>`;
    }
}

function renderAllSignals(signals) {
    const container = qs('#signals');
    if (!signals || signals.length === 0) {
        container.innerHTML = '<p style="text-align:center;color:var(--muted);">لا توجد صفقات مفتوحة حالياً.</p>';
        qs('#signalCount').textContent = '(0)';
        qs('#openCount').textContent = '0';
        return;
    }
    
    container.innerHTML = signals.map(renderSignal).join('');
    qs('#signalCount').textContent = `(${signals.length})`;
    qs('#openCount').textContent = signals.length;
    
    // تحديث الكاش المحلي
    openSignals = signals.reduce((acc, s) => {
        acc[s.id] = s;
        return acc;
    }, {});
}

function updateSingleSignal(signal) {
    try {
        const existingElement = qs(`#signal-${signal.id}`);
        if (existingElement) {
            existingElement.outerHTML = renderSignal(signal);
        } else {
            qs('#signals').insertAdjacentHTML('afterbegin', renderSignal(signal));
        }
        
        // تحديث الكاش
        openSignals[signal.id] = signal;
    } catch (err) {
        console.error('Error updating signal:', err, signal);
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
            
            if (priceEl) {
                priceEl.textContent = fmt(price);
                // تأثير وميض عند تغيير السعر
                priceEl.classList.add(delta > 0 ? 'flash-up' : 'flash-down');
                setTimeout(() => {
                    priceEl.classList.remove('flash-up', 'flash-down');
                }, 300);
            }
            
            if (deltaEl) {
                deltaEl.className = `small price-delta ${delta > 0 ? 'green' : (delta < 0 ? 'red' : '')}`;
                deltaEl.textContent = delta > 0 ? '▲' : (delta < 0 ? '▼' : '•');
            }
            
            // تحديث شريط التقدم
            const signalId = el.id.split('-')[1];
            const signalData = openSignals[signalId];
            
            if (signalData) {
                const entry = signalData.entry_price;
                const tp1 = signalData.target_price_1;
                const sl = signalData.stop_loss;
                
                let progress = 0, color = 'transparent', title = 'في انتظار حركة السعر';
                
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

function addNotification(notification, prepend = true) {
    try {
        const tbody = qs('#events tbody');
        const time = new Date(notification.timestamp).toLocaleTimeString('ar-EG');
        const row = `<tr><td>${time}</td><td>${notification.type||''}</td><td>${notification.message||''}</td></tr>`;
        
        if (prepend) {
            tbody.insertAdjacentHTML('afterbegin', row);
            if (tbody.rows.length > 20) tbody.deleteRow(-1);
        } else {
            tbody.insertAdjacentHTML('beforeend', row);
        }
    } catch (err) {
        console.error('Error adding notification:', err);
    }
}

function addRejection(rejection, prepend = true) {
    try {
        const tbody = qs('#rejections tbody');
        const time = new Date(rejection.timestamp).toLocaleTimeString('ar-EG');
        const row = `<tr><td>${time}</td><td>${rejection.symbol||''}</td><td>${rejection.reason||''}</td></tr>`;
        
        if (prepend) {
            tbody.insertAdjacentHTML('afterbegin', row);
            if (tbody.rows.length > 30) tbody.deleteRow(-1);
        } else {
            tbody.insertAdjacentHTML('beforeend', row);
        }
    } catch (err) {
        console.error('Error adding rejection:', err);
    }
}

function updateMarketTrends(marketState) {
    const container = qs('#marketTrends');
    if (!container) return;
    
    if (!marketState || !marketState.trend_details_by_tf) {
        container.innerHTML = '<p style="text-align:center;color:var(--muted);">لا توجد بيانات</p>';
        return;
    }
    
    container.innerHTML = '';
    ['5m', '15m', '1h'].forEach(tf => {
        const trend = marketState.trend_details_by_tf[tf];
        if (trend) {
            let trendClass = 'amber', trendText = 'جانبي';
            if (trend.trend === 'bullish') { trendClass = 'green'; trendText = 'صاعد'; }
            else if (trend.trend === 'bearish') { trendClass = 'red'; trendText = 'هابط'; }
            
            container.innerHTML += `
                <div class="pill">
                    <b>${tf}</b>
                    <span class="${trendClass}">${trendText}</span>
                    <small>ADX: ${trend.adx?.toFixed(1) || '—'}</small>
                    <small>RSI: ${trend.rsi?.toFixed(1) || '—'}</small>
                </div>`;
        }
    });
}

async function initializeDashboard() {
    try {
        updateConnectionStatus('connecting', 'جاري التحميل...');
        showLoading(true);
        
        // جلب البيانات الأساسية
        const [baseRes, signalsRes, metricsRes] = await Promise.all([
            fetch('/api/dashboard_data'),
            fetch('/api/open_signals'),
            fetch('/api/performance_metrics')
        ]);
        
        if (!baseRes.ok || !signalsRes.ok || !metricsRes.ok) {
            throw new Error('Failed to fetch initial data');
        }
        
        const [baseData, signalsData, metricsData] = await Promise.all([
            baseRes.json(),
            signalsRes.json(),
            metricsRes.json()
        ]);
        
        // عرض البيانات
        qs('#serverTime').textContent = new Date(baseData.server_time).toLocaleTimeString('ar-EG');
        qs('#toggleTrading').checked = !!baseData.trading_enabled;
        qs('#balance').textContent = fmt(baseData.usdt_balance);
        
        const isPaper = baseData.paper_trading_mode;
        qs('#tradingModeToggle').checked = !isPaper;
        qs('#tradingModeText').textContent = isPaper ? 'ورقي' : 'حقيقي';
        
        qs('#qualityFilter').value = baseData.min_signal_quality;
        qs('#qualityValue').textContent = baseData.min_signal_quality;
        qs('#tradeAmountDisplay').textContent = `$${baseData.trade_amount_min} - $${baseData.trade_amount_max}`;
        
        updateMarketTrends(baseData.market_state);
        
        // تعبئة الجداول
        qs('#rejections tbody').innerHTML = '';
        baseData.rejections.forEach(r => addRejection(r, false));
        
        qs('#events tbody').innerHTML = '';
        baseData.notifications.forEach(n => addNotification(n, false));
        
        // عرض الإشارات
        renderAllSignals(signalsData.signals);
        
        updateConnectionStatus('connected', 'متصل');
        showLoading(false);
        
        // جلب بيانات إضافية
        loadAdditionalData();
        
    } catch (error) {
        console.error("❌ Failed to initialize dashboard:", error);
        updateConnectionStatus('error', 'فشل التحميل');
        qs('#signals').innerHTML = '<div class="signal-error">فشل تحميل البيانات. حاول تحديث الصفحة.</div>';
        showLoading(false);
    }
}

async function loadAdditionalData() {
    try {
        const perfRes = await fetch('/api/advanced_performance_data');
        if (perfRes.ok) {
            const advancedData = await perfRes.json();
            qs('#maxDrawdown').textContent = `${advancedData.maxDrawdown.toFixed(2)}%`;
            updateAdvancedPerformance(advancedData);
        }
    } catch (error) {
        console.error("Error loading additional data:", error);
    }
}

function setupWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        return;
    }
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    console.log(`[WebSocket] Connecting to ${wsUrl}...`);
    updateConnectionStatus('connecting', 'جاري الاتصال بـ WebSocket...');
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log("[WebSocket] Connected");
        updateConnectionStatus('connected', 'متصل بـ WebSocket');
        reconnectAttempts = 0;
    };
    
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            
            switch(data.type) {
                case 'price_update':
                    updatePrices(data.payload);
                    break;
                case 'new_signal':
                case 'signal_update':
                    openSignals[data.payload.id] = data.payload;
                    updateSingleSignal(data.payload);
                    break;
                case 'trade_closed':
                    const el = qs(`#signal-${data.payload.signal_id}`);
                    if (el) el.remove();
                    delete openSignals[data.payload.signal_id];
                    break;
                case 'new_notification':
                    addNotification(data.payload);
                    break;
                case 'new_rejection':
                    addRejection(data.payload);
                    break;
                case 'market_state_update':
                    updateMarketTrends(data.payload);
                    break;
                case 'trading_mode':
                    const isPaper = data.payload.paper_trading;
                    qs('#tradingModeToggle').checked = !isPaper;
                    qs('#tradingModeText').textContent = isPaper ? 'ورقي' : 'حقيقي';
                    break;
                case 'quality_filter':
                    qs('#qualityFilter').value = data.payload.min_quality;
                    qs('#qualityValue').textContent = data.payload.min_quality;
                    break;
                case 'trade_amount_update':
                    qs('#tradeAmountDisplay').textContent = `$${data.payload.min} - $${data.payload.max}`;
                    break;
            }
        } catch (err) {
            console.error('WebSocket message error:', err);
        }
    };
    
    ws.onclose = () => {
        console.warn("[WebSocket] Disconnected");
        updateConnectionStatus('warning', 'غير متصل بـ WebSocket');
        
        if (reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            console.log(`[WebSocket] Reconnecting... Attempt ${reconnectAttempts}/${maxReconnectAttempts}`);
            updateConnectionStatus('connecting', `إعادة الاتصال... المحاولة ${reconnectAttempts}`);
            setTimeout(setupWebSocket, Math.min(1000 * reconnectAttempts, 5000));
        } else {
            updateConnectionStatus('error', 'فشل الاتصال بـ WebSocket');
        }
    };
    
    ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        updateConnectionStatus('error', 'خطأ في WebSocket');
    };
}

function setupSorting() {
    const sortButtons = document.querySelectorAll('[data-sort]');
    
    sortButtons.forEach(button => {
        button.addEventListener('click', () => {
            showLoading(true);
            
            fetch(`/api/open_signals?sort=${button.dataset.sort}`)
                .then(res => res.json())
                .then(data => {
                    renderAllSignals(data.signals);
                    showLoading(false);
                })
                .catch(err => {
                    console.error("Sort failed:", err);
                    showLoading(false);
                });
        });
    });
}

function setupEventListeners() {
    // تبديل التداول
    qs('#toggleTrading').addEventListener('change', () => {
        fetch('/toggle_trading', {method: 'POST'})
            .then(res => res.json())
            .then(data => {
                showNotification(`Trading ${data.trading_enabled ? 'enabled' : 'disabled'}`, 'info');
            });
    });

    // تبديل وضع التداول
    qs('#tradingModeToggle').addEventListener('change', function() {
        const isPaper = !this.checked;
        
        if (!isPaper && !confirm('هل أنت متأكد من التبديل إلى التداول الحقيقي؟')) {
            this.checked = false;
            return;
        }
        
        fetch('/api/settings', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({paper_trading_mode: isPaper})
        }).then(res => res.json()).then(data => {
            if (data.success) {
                qs('#tradingModeText').textContent = isPaper ? 'ورقي' : 'حقيقي';
                showNotification(`تم التبديل إلى الوضع ${isPaper ? 'الورقي' : 'الحقيقي'}`, 'success');
            }
        });
    });

    // تحديث جودة الإشارة
    const debouncedQualityUpdate = debounce((value) => {
        fetch('/api/signal_quality', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({min_quality: parseInt(value)})
        }).catch(err => console.error('Error:', err));
    }, 500);

    qs('#qualityFilter').addEventListener('input', function() {
        qs('#qualityValue').textContent = this.value;
        debouncedQualityUpdate(this.value);
    });
}

function updateAdvancedPerformance(data) {
    if (!data || !data.equity_curve) return;
    
    if (!performanceChartInstance) {
        createPerformanceChart(data.equity_curve);
    } else {
        performanceChartInstance.data.labels = data.equity_curve.labels;
        performanceChartInstance.data.datasets[0].data = data.equity_curve.values;
        performanceChartInstance.update('none');
    }
}

function createPerformanceChart(chartData) {
    const ctx = document.getElementById('performanceChart').getContext('2d');
    if (!ctx) return;
    
    performanceChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.labels,
            datasets: [{
                label: 'رأس المال',
                data: chartData.values,
                borderColor: '#3aa0ff',
                backgroundColor: 'rgba(58, 160, 255, 0.1)',
                tension: 0.4,
                fill: true,
                pointRadius: 0,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    type: 'time',
                    time: { unit: 'day' },
                    ticks: { color: 'var(--muted)', autoSkip: true, maxTicksLimit: 8 },
                    grid: { display: false }
                },
                y: {
                    ticks: { color: 'var(--muted)', callback: (v) => v.toFixed(0) },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                }
            }
        }
    });
}

// تهيئة التطبيق
document.addEventListener('DOMContentLoaded', () => {
    console.log("[Dashboard] Initializing...");
    
    initializeDashboard().then(() => {
        setupWebSocket();
        setupSorting();
        setupEventListeners();
        
        // فحص دوري للاتصال
        setInterval(() => {
            fetch('/api/health')
                .then(res => res.json())
                .then(data => {
                    if (data.status === 'ok') {
                        updateConnectionStatus('connected', 'متصل');
                    } else {
                        updateConnectionStatus('warning', 'مشكلة في الاتصال');
                    }
                })
                .catch(() => {
                    updateConnectionStatus('error', 'فقدان الاتصال');
                });
        }, 10000);
        
        console.log("[Dashboard] Initialized successfully");
    }).catch(err => {
        console.error("[Dashboard] Initialization failed:", err);
        updateConnectionStatus('error', 'فشل التهيئة');
    });
});
</script>
</body>
</html>
"""

SETTINGS_TEMPLATE = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>الإعدادات - بوت 5 دقائق (V35.0.0)</title>
<style>
:root{--bg:#0b1020;--panel:#121b36;--accent:#3aa0ff;--ok:#15c46a;--warn:#ff9f1a;--bad:#ff4757;--muted:#8aa0c8;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:#e8f1ff;font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,"Noto Sans",Arial}
.container{max-width:900px;margin:0 auto;padding:16px;display:flex;flex-direction:column;gap:16px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;padding-bottom:16px;border-bottom:1px solid #1e2c52;}
h1{font-size:22px;margin:0;font-weight:700;color:#d7e4ff}
.card{background:var(--panel);border:1px solid #1e2c52;border-radius:14px;box-shadow:0 8px 30px rgba(0,0,0,.25);overflow:hidden}
.card h2{margin:0;padding:12px 14px;border-bottom:1px solid #1e2c52;font-size:16px;color:#cfe2ff;}
.card-body{padding:16px}
.form-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; align-items: end;}
.form-group label {display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px;}
.form-group input, .form-group select {
    background: #0b1126; border: 1px solid #233056; color: #e8f1ff; padding: 10px; border-radius: 8px; font-size: 14px;
}
.switch{display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;border:1px solid #2a3a68;background:#0f1b3b;cursor:pointer;user-select:none}
.switch input{display:none}
.switch .dot{width:14px;height:14px;border-radius:50%;background:#6a7fb2;transition:.2s}
.switch input:checked + .dot{background:#24d08a;transform:translateX(2px) scale(1.1)}
.btn{appearance:none;border:1px solid #2a3a68;background:#0f1b3b;color:#d9e7ff;padding:10px 14px;border-radius:10px;cursor:pointer;font-weight:700;transition: .18s; text-decoration: none;}
.btn.primary{background: linear-gradient(180deg, var(--accent), #2a80d3); border-color: #4aaeff;}
.footer-actions{display:flex;justify-content:flex-end;gap:12px;margin-top:24px;border-top:1px solid #1e2c52;padding-top:16px;}
.notification { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); background-color: #1e2c52; color: #e8f1ff; padding: 12px 20px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); z-index: 1000; opacity: 0; transition: opacity 0.3s, transform 0.3s; }
.notification.show { opacity: 1; transform: translateX(-50%) translateY(0); }
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>الإعدادات</h1>
        <a href="/" class="btn">العودة للوحة التحكم</a>
    </header>

    <form id="settingsForm">
        <div class="card">
            <h2>إعدادات التداول العامة</h2>
            <div class="card-body form-grid">
                <div class="form-group">
                    <label for="tradeAmountMinInput">أدنى قيمة للصفقة (الحقيقية)</label>
                    <input type="number" id="tradeAmountMinInput" name="FIXED_TRADE_AMOUNT_MIN_USDT" value="{{ trade_amount_min }}" step="0.1" min="1.0" max="50.0">
                </div>
                <div class="form-group">
                    <label for="tradeAmountMaxInput">أقصى قيمة للصفقة (الحقيقية)</label>
                    <input type="number" id="tradeAmountMaxInput" name="FIXED_TRADE_AMOUNT_MAX_USDT" value="{{ trade_amount_max }}" step="0.1" min="1.0" max="50.0">
                </div>
                <div class="form-group">
                    <label for="maxTradesInput">الحد الأقصى للصفقات المفتوحة</label>
                    <input type="number" id="maxTradesInput" name="MAX_OPEN_TRADES" value="{{ MAX_OPEN_TRADES }}" step="1" min="1" max="10">
                </div>
                <div class="form-group">
                    <label for="qualityFilter">الحد الأدنى لجودة الإشارة</label>
                    <input type="number" id="qualityFilter" name="min_quality" value="{{ min_quality }}" step="1" min="30" max="90">
                </div>
                <div class="form-group">
                    <label>وضع التداول</label>
                    <label class="switch">
                        <input type="checkbox" name="paper_trading_mode" {% if not is_paper_mode %}checked{% endif %}>
                        <span class="dot"></span>
                        <span id="tradingModeText">{% if is_paper_mode %}ورقي (Paper){% else %}حقيقي (Real){% endif %}</span>
                    </label>
                </div>
            </div>
        </div>

        <!-- إعدادات الاستراتيجية -->
        <div class="card" style="margin-top: 16px;">
            <h2>إعدادات الاستراتيجية المتقدمة</h2>
            <div class="card-body form-grid">
                <div class="form-group">
                    <label>تفعيل فلتر EMA21</label>
                    <label class="switch">
                        <input type="checkbox" name="enable_ema_filter" {% if strategy_config.enable_ema_filter %}checked{% endif %}>
                        <span class="dot"></span>
                    </label>
                </div>
                <div class="form-group">
                    <label>تفعيل تأكيد MACD</label>
                    <label class="switch">
                        <input type="checkbox" name="enable_macd_confirmation" {% if strategy_config.enable_macd_confirmation %}checked{% endif %}>
                        <span class="dot"></span>
                    </label>
                </div>
                <div class="form-group">
                    <label>تفعيل فلتر MFI</label>
                    <label class="switch">
                        <input type="checkbox" name="enable_mfi_filter" {% if strategy_config.enable_mfi_filter %}checked{% endif %}>
                        <span class="dot"></span>
                    </label>
                </div>
                <div class="form-group">
                    <label>تفعيل أنماط الشموع</label>
                    <label class="switch">
                        <input type="checkbox" name="enable_candlestick_patterns" {% if strategy_config.enable_candlestick_patterns %}checked{% endif %}>
                        <span class="dot"></span>
                    </label>
                </div>
                <div class="form-group">
                    <label for="requiredConfirmations">عدد التأكيدات المطلوبة</label>
                    <input type="number" id="requiredConfirmations" name="required_confirmations" value="{{ strategy_config.required_confirmations }}" step="1" min="1" max="5">
                </div>
            </div>
        </div>
        
        <div class="footer-actions">
            <button type="submit" class="btn primary">حفظ التغييرات</button>
        </div>
    </form>
</div>
<div id="notification" class="notification"></div>

<script>
document.getElementById('settingsForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const settings = {};
    const strategyConfig = {};

    for (const [key, value] of formData.entries()) {
        if (key.includes('enable_') || key === 'required_confirmations') {
            strategyConfig[key] = formData.has(key);
        } else if (key === 'paper_trading_mode') {
            settings[key] = false;
        } else {
            settings[key] = value;
        }
    }
    
    if (!formData.has('paper_trading_mode')) {
        settings['paper_trading_mode'] = true;
    }

    // إرسال جميع التحديثات في آن واحد
    Promise.all([
        fetch('/api/settings', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(settings)
        }),
        fetch('/api/strategy_config', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(strategyConfig)
        }),
        fetch('/api/signal_quality', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({min_quality: settings.min_quality})
        })
    ]).then(responses => {
        const allSuccess = responses.every(res => res.ok);
        
        if (allSuccess) {
            showNotification('تم حفظ جميع الإعدادات بنجاح!', 'success');
        } else {
            showNotification('حدث خطأ في بعض الإعدادات.', 'error');
        }
    }).catch(err => {
        console.error('Error saving settings:', err);
        showNotification('فشل الاتصال بالخادم.', 'error');
    });
});

document.querySelector('input[name="paper_trading_mode"]').addEventListener('change', function() {
    document.getElementById('tradingModeText').textContent = this.checked ? 'حقيقي (Real)' : 'ورقي (Paper)';
});

function showNotification(message, type = 'info') {
    const notification = document.getElementById('notification');
    notification.textContent = message;
    notification.className = `notification ${type} show`;
    
    setTimeout(() => {
        notification.classList.remove('show');
    }, 3000);
}
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
<title>الاختبار الخلفي - بوت التداول V35.0.0</title>
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
.form-group input, .form-group select {
    background: #0b1126; border: 1px solid #233056; color: #e8f1ff; padding: 10px; border-radius: 8px;
}
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
.loader-text { margin-top: 10px; color: var(--muted); }
.loading-spinner { border: 3px solid rgba(255, 255, 255, 0.1); border-radius: 50%; border-top: 3px solid #3aa0ff; width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 20px auto; }
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="container">
    <header><h1>الاختبار الخلفي للاستراتيجيات</h1><a href="/" class="btn">العودة للرئيسية</a></header>
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
    <div id="loader">
        <div class="loading-spinner"></div>
        <div class="loader-text">جاري تحميل البيانات وتنفيذ الاختبار... قد يستغرق هذا بعض الوقت.</div>
    </div>
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
    if (!data) return;
    
    qs('#results-container').style.display = 'block';
    
    qs('#totalTrades').textContent = data.total_trades || 0;
    qs('#winRate').textContent = `${(data.win_rate || 0).toFixed(2)}%`;
    qs('#avgProfit').textContent = `${(data.avg_profit || 0).toFixed(2)}%`;
    qs('#profitFactor').textContent = (data.profit_factor || 0).toFixed(2);
    
    const avgProfitEl = qs('#avgProfit');
    avgProfitEl.classList.toggle('green', data.avg_profit > 0);
    avgProfitEl.classList.toggle('red', data.avg_profit < 0);
    
    const tradesTable = qs('#trades-table');
    if (data.results && data.results.length > 0) {
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
    } else {
        tradesTable.innerHTML = '<tr><td colspan="6" style="text-align:center;">لا توجد صفقات</td></tr>';
    }
    
    updateEquityChart(data.equity_curve);
}

function updateEquityChart(equityData) {
    if (!equityData || equityData.length === 0) return;
    
    const ctx = document.getElementById('equityChart').getContext('2d');
    if (!ctx) return;
    
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

# --- نقطة البداية ---
if __name__ == '__main__':
    print("="*60)
    print("====== بوت التداول المُحسّن V35.0.0 - 5 دقائق متعدد الاستراتيجيات ======")
    print("="*60)
    
    # تهيئة قاعدة البيانات
    init_db()
    init_redis()
    
    # تهيئة Binance
    try:
        client = Client(API_KEY, API_SECRET)
        client.ping()
        logger.info("✅ [Binance] API connection successful")
    except Exception as e:
        logger.critical(f"❌ [Binance] API connection failed: {e}")
        exit(1)
    
    # جلب البيانات الأساسية
    get_exchange_info_map()
    validated_symbols_to_scan = get_validated_symbols()
    
    if not validated_symbols_to_scan:
        logger.critical("❌ No valid symbols found")
        exit(1)
    
    # تحميل البيانات الأولية
    load_open_signals_to_cache()
    load_notifications_to_cache()
    load_settings_from_redis()
    update_balance()
    
    # بدء الخيوط الخلفية
    start_websocket()
    Thread(target=main_bot_loop, daemon=True, name="MainBotLoop").start()
    Thread(target=trade_management_loop, daemon=True, name="TradeManager").start()
    start_market_state_updater()
    Thread(target=update_balance_loop, daemon=True, name="BalanceUpdater").start()
    start_periodic_reports()
    
    # بدء خادم Flask
    logger.info("🌐 [Flask] Starting UI on http://0.0.0.0:5000")
    logger.info("📊 Dashboard: http://localhost:5000")
    logger.info("⚙️  Settings: http://localhost:5000/settings")
    logger.info("🧪 Backtest: http://localhost:5000/backtest")
    logger.info("🚀 Bot is running with multi-strategy support!")
    
    app.run(host='0.0.0.0', port=5000, debug=False)