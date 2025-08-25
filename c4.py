# ملف c4.py - نسخة V25.4.0 (إدارة متقدمة)
# --- وصف الإصدار:
# 1.  [إدارة الصفقات] تمت إضافة آلية للتحقق من الحد الأقصى لعدد الصفقات المفتوحة (MAX_OPEN_TRADES) قبل البحث عن فرص جديدة.
# 2.  [تحكم مباشر] تمت إضافة زر تحكم في واجهة المستخدم لتشغيل أو إيقاف التداول بشكل فوري، مع تحديث الحالة لجميع العملاء عبر WebSocket.
# 3.  [تحسين الواجهة] تم عرض عدد الصفقات المفتوحة حالياً مقارنة بالحد الأقصى في لوحة التحكم.
# 4.  [إشعارات مرئية] تمت إضافة نظام إشعارات في الواجهة لتأكيد عمليات تشغيل/إيقاف التداول.

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
from flask import Flask, jsonify, render_template, render_template_string, request
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
usdt_balance: float = 0.0
balance_lock = Lock()
cooldowns_by_symbol = {}
cooldowns_lock = Lock()
consecutive_losses_by_symbol = {}
consecutive_losses_lock = Lock()
COOLDOWN_MINUTES_AFTER_SL = 20
PAPER_TRADE_INITIAL_BALANCE = 1000.0

# --- المتغيرات القابلة للتعديل ---
RISK_PER_TRADE_PERCENT: float = 1.0 # القيمة الافتراضية، سيتم تعديلها ديناميكياً
risk_per_trade_lock = Lock()
MAX_OPEN_TRADES: int = 3
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
    "Low Quality Signal": "جودة الإشارة منخفضة",
    "Invalid Position Size": "حجم الصفقة غير صالح (الوقف أعلى من الدخول)",
    "News Filter Failed": "فلتر الأخبار: تجنب التداول وقت الأخبار",
    "Liquidity Filter Failed": "فلتر السيولة: تجنب التداول في أوقات السيولة المنخفضة",
    "Correlation Filter Failed": "فلتر الارتباط: توجد صفقة مفتوحة على عملة مرتبطة"
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

def get_open_trades_count() -> int:
    """الحصول على عدد التوصيات المفتوحة حالياً"""
    with signal_cache_lock:
        return len(open_signals_cache)

def get_dashboard_payload() -> Dict:
    with trading_status_lock: trading_enabled = is_trading_enabled
    with trading_mode_lock: is_paper_mode = paper_trading_mode
    with balance_lock: current_balance = usdt_balance
    with notifications_lock: notifications = list(notifications_cache)
    with rejection_logs_lock: rejections = list(rejection_logs_cache)
    with market_state_lock: market_state = dict(current_market_state)
    with min_quality_lock: min_quality = MIN_SIGNAL_QUALITY
    with risk_per_trade_lock: risk_percent = RISK_PER_TRADE_PERCENT

    return {
        "trading_enabled": trading_enabled, 
        "paper_trading_mode": is_paper_mode,
        "usdt_balance": current_balance,
        "notifications": notifications, 
        "rejections": rejections, 
        "market_state": market_state,
        "min_signal_quality": min_quality,
        "risk_per_trade": risk_percent,
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

def send_enhanced_telegram_message(message: str, force: bool = False):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    
    settings = get_notification_settings()
    if not settings.get('telegram_enabled') and not force:
        return
    
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
                                quality_score: int, atr_percent: float):
    trade_type = "حقيقية" if is_real else "ورقية"
    emoji = "🔥" if is_real else "📊"
    
    message = (
        f"{emoji} *صفقة {trade_type} جديدة*\n\n"
        f"*العملة:* `{symbol}`\n"
        f"*الاستراتيجية:* `{strategy_name}`\n"
        f"*جودة الإشارة:* `{quality_score}/100`\n"
        f"*تقلب السوق:* `{atr_percent:.2f}%`\n\n"
        f"*سعر الدخول:* `{entry_price:.4f}`\n"
        f"*وقف الخسارة:* `{stop_loss:.4f}`\n"
        f"*الهدف الأول:* `{target1:.4f}`\n"
        f"*الهدف الثاني:* `{target2:.4f}`\n\n"
        f"*الكمية:* `{quantity:.4f}`\n"
        f"*نسبة المخاطرة:* `{((entry_price - stop_loss) / entry_price * 100):.2f}%`\n"
        f"*نسبة الربح المحتملة 1:* `{((target1 - entry_price) / entry_price * 100):.2f}%`\n"
        f"*نسبة الربح المحتملة 2:* `{((target2 - entry_price) / entry_price * 100):.2f}%`"
    )
    
    send_enhanced_telegram_message(message, force=True)

def send_daily_performance_report():
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
                logger.info("[Daily Report] No trades to report for today.")
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
                    f"📉 *أسوأ صفقة:*\n"
                    f"العملة: `{worst_trade['symbol']}` | الخسارة: `{worst_trade['profit_percentage']:.2f}%`\n\n"
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
                send_daily_performance_report()
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
    global RISK_PER_TRADE_PERCENT, MAX_OPEN_TRADES, USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY, USE_MOMENTUM_VOLATILITY_STRATEGY, STRATEGY_FILTER_CONFIG, paper_trading_mode, MIN_SIGNAL_QUALITY
    if not redis_client: return
    try:
        settings_data = redis_client.get('trading_settings')
        if settings_data:
            settings = json.loads(settings_data)
            with risk_per_trade_lock: RISK_PER_TRADE_PERCENT = settings.get('RISK_PER_TRADE_PERCENT', 1.0)
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

# --- Advanced Filters ---
def add_news_filter() -> bool:
    news_hours = [(12, 30), (14, 0), (18, 30)]
    now = datetime.now(timezone.utc)
    for hour, minute in news_hours:
        if now.hour == hour and abs(now.minute - minute) <= 30:
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

# --- Dynamic Risk Calculation ---
def calculate_dynamic_risk_per_trade() -> float:
    if not check_db_connection() or not conn:
        with risk_per_trade_lock: return RISK_PER_TRADE_PERCENT
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT profit_percentage FROM signals WHERE status = 'closed' ORDER BY closed_at DESC LIMIT 10")
            recent_trades = cur.fetchall()
            if len(recent_trades) < 5:
                with risk_per_trade_lock: return RISK_PER_TRADE_PERCENT
            profits = [trade['profit_percentage'] for trade in recent_trades]
            avg_profit = sum(profits) / len(profits)
            win_rate = sum(1 for p in profits if p > 0) / len(profits)
            with risk_per_trade_lock: base_risk = RISK_PER_TRADE_PERCENT
            if avg_profit > 1.5 and win_rate >= 0.6:
                new_risk = min(2.0, base_risk * 1.2)
                logger.info(f"[Dynamic Risk] Good performance detected. Increasing risk to {new_risk:.2f}%")
                return new_risk
            elif avg_profit < -0.5 or win_rate < 0.4:
                new_risk = max(0.5, base_risk * 0.8)
                logger.warning(f"[Dynamic Risk] Poor performance detected. Reducing risk to {new_risk:.2f}%")
                return new_risk
            else:
                return base_risk
    except Exception as e:
        logger.error(f"❌ [Dynamic Risk] Error calculating dynamic risk: {e}", exc_info=True)
        with risk_per_trade_lock: return RISK_PER_TRADE_PERCENT

# --- Risk Management & Filters ---
def calculate_signal_quality_score(symbol, df, strategy_name):
    score = 0
    if df.empty or len(df) < 50: return 0
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

# --- Enhanced Strategies ---
def check_bb_stoch_strategy_enhanced(df: pd.DataFrame) -> bool:
    if len(df) < 200 or not {'bb_lower', 'stoch_rsi_k', 'rsi', 'open', 'close', 'high', 'low', 'macd', 'macd_signal', 'ema50', 'ema200', 'volume'}.issubset(df.columns): return False
    if df['close'].iloc[-1] < df['ema200'].iloc[-1]: log_rejection(df.name, "Long-term Trend Filter Failed"); return False
    if not flexible_volume_filter(df, min_volume_percentile=30, strictness=0.7): log_rejection(df.name, "Volume Filter Failed"); return False
    if not check_market_volatility_filter(df): return False
    last, prev = df.iloc[-1], df.iloc[-2]
    bounce = (prev['close'] < prev['bb_lower']) and (last['close'] > last['bb_lower'])
    stoch_rising = last['stoch_rsi_k'] > prev['stoch_rsi_k']
    bullish_body = last['close'] > (last['open'] + (last['high'] - last['low']) * 0.3)
    rsi_improving = last['rsi'] > prev['rsi']
    not_overbought = last['rsi'] < 70
    macd_ok = (last['macd'] > last['macd_signal']) or (last['macd'] - last['macd_signal'] > -0.1 * abs(last['macd']))
    price_above_ema50 = last['close'] > last['ema50']
    ema50_above_ema200 = last['ema50'] > last['ema200']
    volume_increasing = last['volume'] > prev['volume']
    signal = (bounce and (stoch_rising or bullish_body) and rsi_improving and not_overbought and macd_ok and price_above_ema50 and ema50_above_ema200 and volume_increasing)
    if not signal: log_rejection(df.name, "Bullish Confirmation Failed")
    return signal

def check_macd_ema_strategy_enhanced(df: pd.DataFrame) -> bool:
    needed = {'macd', 'macd_signal', 'ema9', 'ema21', 'rsi', 'close', 'adx', 'ema200'}
    if len(df) < 200 or not needed.issubset(df.columns): return False
    if df['close'].iloc[-1] < df['ema200'].iloc[-1]: log_rejection(df.name, "Long-term Trend Filter Failed"); return False
    if df['adx'].iloc[-1] < 22: log_rejection(df.name, "Trend Strength Filter Failed"); return False
    last, prev = df.iloc[-1], df.iloc[-2]
    macd_cross_up = (prev['macd'] <= prev['macd_signal']) and (last['macd'] > last['macd_signal'])
    hist_now = last['macd'] - last['macd_signal']
    hist_prev = prev['macd'] - prev['macd_signal']
    hist_increasing = (hist_now > hist_prev) and (hist_prev > 0 or macd_cross_up)
    ema_ok = (last['close'] > last['ema21']) and (last['ema9'] > last['ema21'])
    rsi_ok = 40 <= last['rsi'] <= 65
    return (macd_cross_up or hist_increasing) and ema_ok and rsi_ok

def check_ema_rsi_strategy_enhanced(df: pd.DataFrame) -> bool:
    needed = {'ema9','ema21','rsi','low','close', 'ema200'}
    if len(df) < 200 or not needed.issubset(df.columns): return False
    if df['close'].iloc[-1] < df['ema200'].iloc[-1]: log_rejection(df.name, "Long-term Trend Filter Failed"); return False
    last3 = df.tail(3)
    ema9_over_21 = (last3['ema9'] > last3['ema21']).sum() >= 2
    last = last3.iloc[-1]
    rsi_ok = 50 <= float(last['rsi']) <= 65
    pullback_ok = (float(last['low']) <= float(last['ema9'])) and (float(last['close']) > float(last['ema9']))
    return ema9_over_21 and rsi_ok and pullback_ok

def check_pullback_strategy_enhanced(df: pd.DataFrame) -> bool:
    needed = {'ema9','ema21','ema50','open','close','low', 'ema200'}
    if len(df) < 200 or not needed.issubset(df.columns): return False
    if df['close'].iloc[-1] < df['ema200'].iloc[-1]: log_rejection(df.name, "Long-term Trend Filter Failed"); return False
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
    if df['close'].iloc[-1] < df['ema200'].iloc[-1]: log_rejection(df.name, "Long-term Trend Filter Failed"); return False
    last, prev = df.iloc[-1], df.iloc[-2]
    atr_mean = float(pd.Series(df['atr_percent'].tail(14)).mean())
    atr_ok = float(last['atr_percent']) >= (1.2 * atr_mean)
    hist_now = float(last['macd'] - last['macd_signal'])
    hist_prev = float(prev['macd'] - prev['macd_signal'])
    hist_rising = hist_now > hist_prev
    ema_ok = (last['ema9'] > last['ema21']) and (last['close'] > last['ema9'])
    return atr_ok and hist_rising and ema_ok

# --- Dynamic Trade Level Calculation ---
def calculate_trade_levels(df: pd.DataFrame) -> Dict[str, Any]:
    last = df.iloc[-1]
    atr = last['atr']
    entry_price = last['close']
    atr_percent = last['atr_percent']
    if atr_percent > 3.0: stop_loss_multiplier = 2.0
    elif atr_percent < 1.5: stop_loss_multiplier = 1.2
    else: stop_loss_multiplier = 1.5
    stop_loss = entry_price - (atr * stop_loss_multiplier)
    if atr_percent > 3.0: target1_multiplier, target2_multiplier = 2.5, 4.5
    elif atr_percent < 1.5: target1_multiplier, target2_multiplier = 1.8, 3.0
    else: target1_multiplier, target2_multiplier = 2.0, 3.5
    target_price_1 = entry_price + (atr * target1_multiplier)
    target_price_2 = entry_price + (atr * target2_multiplier)
    if atr_percent > 3.0: trailing_stop_multiplier = 2.0
    elif atr_percent < 1.5: trailing_stop_multiplier = 1.2
    else: trailing_stop_multiplier = 1.5
    trailing_stop_distance = atr * trailing_stop_multiplier
    return {"entry_price": entry_price, "stop_loss": stop_loss, "target_price_1": target_price_1, "target_price_2": target_price_2, "atr": atr, "trailing_stop_distance": trailing_stop_distance, "stop_loss_multiplier": stop_loss_multiplier, "target1_multiplier": target1_multiplier, "target2_multiplier": target2_multiplier, "trailing_stop_multiplier": trailing_stop_multiplier}

def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> Optional[Decimal]:
    try:
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info: logger.error(f"[{symbol}] No exchange info found for LOT_SIZE adjustment."); return None
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if not lot_size_filter: logger.warning(f"[{symbol}] LOT_SIZE filter not found. Using raw quantity."); return Decimal(str(quantity))
        step_size = Decimal(lot_size_filter['stepSize'])
        min_qty = Decimal(lot_size_filter['minQty'])
        quantity_dec = Decimal(str(quantity))
        if quantity_dec < min_qty: log_rejection(symbol, "LOT_SIZE Filter Failed", {"reason": "Below minQty", "qty": f"{quantity_dec}", "min": f"{min_qty}"}); return None
        adjusted_quantity = (quantity_dec // step_size) * step_size
        if adjusted_quantity < min_qty: log_rejection(symbol, "LOT_SIZE Filter Failed", {"reason": "Adjusted below minQty", "qty": f"{adjusted_quantity}", "min": f"{min_qty}"}); return None
        return adjusted_quantity
    except Exception as e:
        logger.error(f"❌ [{symbol}] CRITICAL ERROR adjusting quantity: {e}", exc_info=True)
        return None

def calculate_position_size(symbol: str, entry_price: float, stop_loss_price: float, risk_percent: float = None) -> Optional[Decimal]:
    if not client: return None
    available_balance_str, risk_per_coin_str = "N/A", "N/A"
    try:
        if risk_percent is None:
            with risk_per_trade_lock: risk_percent = RISK_PER_TRADE_PERCENT
        balance_response = client.get_asset_balance(asset='USDT')
        available_balance = Decimal(balance_response['free'])
        available_balance_str = str(available_balance)
        risk_amount_usdt = available_balance * (Decimal(str(risk_percent)) / Decimal('100'))
        risk_per_coin = Decimal(str(entry_price)) - Decimal(str(stop_loss_price))
        risk_per_coin_str = str(risk_per_coin)
        if risk_per_coin <= 0: log_rejection(symbol, "Invalid Position Size", {"reason": "Stop loss is not below entry price", "entry": entry_price, "sl": stop_loss_price}); return None
        initial_quantity = risk_amount_usdt / risk_per_coin
        adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(initial_quantity))
        if adjusted_quantity is None or adjusted_quantity <= 0: return None
        notional_value = adjusted_quantity * Decimal(str(entry_price))
        symbol_info = exchange_info_map.get(symbol)
        if symbol_info:
            for f in symbol_info['filters']:
                if f['filterType'] in ('MIN_NOTIONAL', 'NOTIONAL'):
                    min_notional = Decimal(f.get('minNotional', f.get('notional', '0')))
                    if notional_value < min_notional: log_rejection(symbol, "MinNotional Filter Failed", {"value": f"{notional_value:.2f}", "required": f"{min_notional}"}); return None
        if notional_value > available_balance: log_rejection(symbol, "Insufficient Balance", {"required": f"{notional_value:.2f}", "available": f"{available_balance}"}); return None
        return adjusted_quantity
    except Exception as e:
        logger.error(f"❌ [{symbol}] Error calculating position size: {e}", exc_info=True)
        logger.error(f"  └── DEBUG INFO for {symbol}: Entry={entry_price}, SL={stop_loss_price}, Balance={available_balance_str}, RiskPerCoin={risk_per_coin_str}, RiskPercent={risk_percent}")
        return None

def place_order(symbol: str, side: str, quantity: Decimal, order_type: str = Client.ORDER_TYPE_MARKET) -> Optional[Dict]:
    if not client: return None
    logger.info(f"➡️ [{symbol}] Attempting to place REAL {side} order for quantity {quantity}.")
    try:
        order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=str(quantity))
        log_and_notify('info', f"TRADE REAL: Placed {side} order for {quantity} {symbol}.", "REAL_TRADE")
        return order
    except BinanceAPIException as e:
        logger.error(f"❌ [{symbol}] Binance API Error on order placement: {e}")
        send_enhanced_telegram_message(f"❌ *خطأ باينانس عند وضع أمر لـ {symbol}*\n`{e}`", force=True)
        return None
    except Exception as e:
        logger.error(f"❌ [{symbol}] General error on order placement: {e}", exc_info=True)
        return None

# --- Centralized Trade Signal Creation ---
def create_trade_signal(symbol: str, df: pd.DataFrame, strategy_name: str):
    try:
        quality_score = calculate_signal_quality_score(symbol, df, strategy_name)
        with min_quality_lock: min_score = MIN_SIGNAL_QUALITY
        if quality_score < min_score: log_rejection(symbol, "Low Quality Signal", {"score": quality_score, "min_required": min_score}); return
        logger.info(f"⭐ [Signal Quality] {symbol} ({strategy_name}): {quality_score}/100")
        with cooldowns_lock: 
            until = cooldowns_by_symbol.get(symbol)
            if until and datetime.now(timezone.utc) < until: log_rejection(symbol, "Cooldown Active", {"until": until.isoformat()}); return
        if not add_news_filter(): log_rejection(symbol, "News Filter Failed"); return
        if not add_liquidity_filter(): log_rejection(symbol, "Liquidity Filter Failed"); return
        if not add_correlation_filter(symbol): log_rejection(symbol, "Correlation Filter Failed"); return
    except Exception as e:
        logger.error(f"❌ [Signal Creation] Error during pre-checks for {symbol}: {e}", exc_info=True)
        return

    with trading_mode_lock: is_real = not paper_trading_mode
    trade_levels = calculate_trade_levels(df)
    entry_price = trade_levels['entry_price']
    stop_loss_price = trade_levels['stop_loss']
    signal_details = {"atr": trade_levels['atr'], "trailing_stop_activated": False, "trailing_stop_distance": trade_levels['trailing_stop_distance'], "tp1_done": False, "quality_score": quality_score, "stop_loss_multiplier": trade_levels['stop_loss_multiplier'], "target1_multiplier": trade_levels['target1_multiplier'], "target2_multiplier": trade_levels['target2_multiplier'], "trailing_stop_multiplier": trade_levels['trailing_stop_multiplier']}

    if is_real:
        dynamic_risk_percent = calculate_dynamic_risk_per_trade()
        quantity_dec = calculate_position_size(symbol, entry_price, stop_loss_price, dynamic_risk_percent)
        if quantity_dec is None or quantity_dec <= 0: logger.error(f"❌ [Real Trade] Position size calculation failed for {symbol}. Trade rejected."); return
        order = place_order(symbol, Client.SIDE_BUY, quantity_dec)
        if order:
            avg_fill_price = sum(Decimal(f['price']) * Decimal(f['qty']) for f in order.get('fills', [])) / max(sum(Decimal(f['qty']) for f in order.get('fills', [])), Decimal('1e-8')) if order.get('fills') else Decimal(str(entry_price))
            final_quantity = Decimal(order.get('executedQty', str(quantity_dec)))
            order_id = order.get('orderId', 'N/A')
            save_signal_to_db(symbol, float(avg_fill_price), trade_levels, strategy_name, True, float(final_quantity), {**signal_details, "avg_fill": float(avg_fill_price)}, order_id)
            send_trade_open_notification(symbol, strategy_name, float(avg_fill_price), trade_levels['stop_loss'], trade_levels['target_price_1'], trade_levels['target_price_2'], float(final_quantity), is_real, quality_score, df['atr_percent'].iloc[-1])
            log_and_notify("info", f"Opened REAL trade for {symbol}", "REAL_TRADE_OPEN")
        else:
            logger.error(f"❌ [Real Trade] Order placement failed for {symbol}. Trade not opened.")
            return
    else: # Paper Trading
        risk_per_coin = entry_price - stop_loss_price
        if risk_per_coin <= 0: return
        with risk_per_trade_lock: paper_risk = RISK_PER_TRADE_PERCENT
        risk_amount_usdt = PAPER_TRADE_INITIAL_BALANCE * (paper_risk / 100.0)
        quantity = risk_amount_usdt / risk_per_coin
        save_signal_to_db(symbol, entry_price, trade_levels, strategy_name, False, quantity, signal_details)
        send_trade_open_notification(symbol, strategy_name, entry_price, trade_levels['stop_loss'], trade_levels['target_price_1'], trade_levels['target_price_2'], quantity, is_real, quality_score, df['atr_percent'].iloc[-1])
        log_and_notify("info", f"Opened paper trade for {symbol}", "PAPER_TRADE_OPEN")

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
        broadcast({"type": "open_trades_count_update", "payload": {"count": get_open_trades_count(), "max": MAX_OPEN_TRADES}})
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
<title>لوحة التحكم - بوت التداول (V25 - إدارة متقدمة)</title>
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
.badge.bg-success{background-color:rgba(21,196,106,.2);border-color:#15c46a;color:#15c46a}
.badge.bg-danger{background-color:rgba(255,71,87,.2);border-color:#ff4757;color:#ff4757}
.badge.bg-info{background-color:rgba(58,160,255,.2);border-color:#3aa0ff;color:#3aa0ff}
.main-layout{display:grid;grid-template-columns:1fr;gap:16px;}
@media(min-width: 1000px){.main-layout{grid-template-columns:1fr 350px;}}
.left-column,.right-column{display:flex;flex-direction:column;gap:16px}
.card{background:var(--panel);border:1px solid #1e2c52;border-radius:14px;box-shadow:0 8px 30px rgba(0,0,0,.25);overflow:hidden}
.card h2{margin:0;padding:12px 14px;border-bottom:1px solid #1e2c52;font-size:14px;color:#cfe2ff; display: flex; justify-content: space-between; align-items: center;}
.card-body{padding:12px}
.controls{display:flex;gap:8px;flex-wrap:wrap}
.btn{appearance:none;border:1px solid #2a3a68;background:#0f1b3b;color:#d9e7ff;padding:10px 14px;border-radius:10px;cursor:pointer;font-weight:700;transition: all 0.2s; will-change: transform; text-decoration: none;}
.btn:hover{transform:translateY(-1px);border-color:#3a58a6}
.btn.warn{background:linear-gradient(180deg,#3b2a0f,#291b08);border-color:#8b5b0f}
.btn.btn-success{background:linear-gradient(180deg,#103a25,#0b291a);border-color:#15c46a;color:#15c46a}
.btn.btn-danger{background:linear-gradient(180deg,#4d161c,#330e13);border-color:#ff4757;color:#ff4757}
.btn.small{padding: 6px 10px; font-size: 12px;}
.signals-grid{display:grid;grid-template-columns:repeat(auto-fill, minmax(300px, 1fr));gap:10px; contain: layout style paint;}
.signal{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;padding:10px;border:1px solid #24335f;border-radius:12px;background:#0d1730; will-change: transform, opacity; transition: transform 0.2s ease, opacity 0.2s ease; grid-template-rows: auto auto;}
.signal > *:nth-child(1){grid-column:1/2}.signal > *:nth-child(2){grid-column:2/3;grid-row:1/3}.signal > *:nth-child(3){grid-column:1/2}
.sig-title{font-weight:700}.sig-meta{font-size:12px;color:var(--muted)}
.price{font-variant-numeric:tabular-nums;direction:ltr; transition: color 0.3s, background-color 0.3s; font-size: 16px; font-weight: bold;}
.progress{height:8px;background:#0b1126;border:1px solid #233056;border-radius:999px;overflow:hidden; margin-top: 6px;}
.progress>span{display:block;height:100%;}
.kv{display:grid;grid-template-columns:auto 1fr;gap:6px 10px; align-items: center;}
.kv div:nth-child(odd){opacity:.8}
.trend{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}
.trend .pill{background:#0d1730;border:1px solid #1f2d55;border-radius:10px;padding:8px;text-align:center; display: flex; flex-direction: column; align-items: center; gap: 4px;}
.pill b{display:block;font-size:12px;color:#9fb7ef}.pill span{font-size:12px}.pill small{font-size:10px;opacity:.8}
.green{color:var(--ok)}.red{color:var(--bad)}.amber{color:var(--warn)}
.table{width:100%;border-collapse:separate;border-spacing:0 8px; table-layout: fixed;}
.table th{font-size:12px;text-align:right;color:#9ab2e2;font-weight:600;padding:0 6px}
.table td{padding:8px;background:#0d1730;border:1px solid #24335f; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}
.small{font-size:12px;color:#a8bfeb}
.chart-container{height:200px}
.loading-spinner{border:3px solid rgba(255,255,255,.1);border-radius:50%;border-top:3px solid #3aa0ff;width:30px;height:30px;animation:spin 1s linear infinite;margin:20px auto}
@keyframes spin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}
#notification-container{position:fixed;top:20px;left:50%;transform:translateX(-50%);z-index:1050;width:350px;max-width:90%}
.alert{padding:1rem;margin-bottom:1rem;border:1px solid transparent;border-radius:.75rem;position:relative;box-shadow:0 .5rem 1rem rgba(0,0,0,.15)}
.alert-info{color:#e8f1ff;background-color:var(--panel);border-color:#1e2c52}
.alert-danger{color:#ff4757;background-color:rgba(255,71,87,.15);border-color:rgba(255,71,87,.2)}
.btn-close{position:absolute;top:0;right:0;z-index:2;padding:1.25rem 1rem;background:transparent;border:0;opacity:.5;cursor:pointer}
</style>
</head>
<body>
<div id="notification-container"></div>
<div class="container">
  <header><h1>لوحة التحكم • بوت التداول V25 (إدارة متقدمة)</h1><div class="badge" id="serverTime">—</div></header>
  <div class="main-layout">
    <div class="left-column">
      <div class="card">
        <h2>الصفقات المفتوحة</h2>
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
        <div class="card-body"><div class="chart-container"><canvas id="performanceChart"></canvas></div></div>
      </div>
    </div>
    <div class="right-column">
      <div class="card">
        <h2>التحكم في البوت</h2>
        <div class="card-body">
            <div class="kv">
                <div>حالة التداول</div>
                <div><span id="trading-status" class="badge {% if trading_enabled %}bg-success{% else %}bg-danger{% endif %}">{% if trading_enabled %}مفعّل{% else %}معطّل{% endif %}</span></div>
                <div>الإجراءات</div>
                <div><button id="toggle-trading-btn" class="btn {% if trading_enabled %}btn-danger{% else %}btn-success{% endif %}">{% if trading_enabled %}إيقاف التداول{% else %}تشغيل التداول{% endif %}</button></div>
                <div>عدد التوصيات المفتوحة</div>
                <div><span id="open-trades-count" class="badge bg-info">{{ open_trades_count }}/{{ MAX_OPEN_TRADES }}</span></div>
            </div>
        </div>
      </div>
      <div class="card">
        <h2>حالة السوق</h2>
        <div class="card-body"><div class="trend" id="marketTrends"><div class="loading-spinner"></div></div></div>
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

function fmt(n){ return n == null ? '—' : (+n).toLocaleString('en-US', {maximumFractionDigits: 6}); }
function showLoadingIndicator(containerId) {
    const container = qs(containerId);
    if(container) container.innerHTML = '<div class="loading-spinner"></div>';
}

function showNotification(message, type) {
    const container = document.getElementById('notification-container');
    const notification = document.createElement('div');
    notification.className = `alert alert-${type === 'error' ? 'danger' : 'info'} alert-dismissible fade show`;
    notification.innerHTML = `${message}<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>`;
    container.appendChild(notification);
    const bsAlert = new bootstrap.Alert(notification);
    setTimeout(() => { bsAlert.close(); }, 3000);
}

function closeTrade(signalId) {
    fetch(`/api/close_trade/${signalId}`, { method: 'POST' })
    .then(res => res.json())
    .then(data => showNotification(data.message || 'تم إرسال أمر الإغلاق', data.success ? 'info' : 'error'))
    .catch(err => showNotification('فشل إغلاق الصفقة', 'error'));
}

function renderSignal(signal) {
    const cp = lastPrices[signal.symbol] || signal.entry_price;
    const entry = signal.entry_price;
    const tp1 = signal.target_price_1;
    const sl = signal.stop_loss;
    let progress = 0, color = 'transparent', title = 'في انتظار حركة السعر';
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
                <button class="btn warn small" onclick="closeTrade(${signal.id})">إغلاق</button>
            </div>
            <div class="progress" title="${title}"><span style="width:${progress.toFixed(2)}%; background:${color};"></span></div>
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

function updatePrices(priceData) {
    Object.entries(priceData).forEach(([symbol, price]) => {
        document.querySelectorAll(`.signal[data-symbol="${symbol}"]`).forEach(el => {
            el.querySelector('.price').textContent = fmt(price);
            const signalId = el.id.split('-')[1];
            const signalData = openSignals[signalId];
            if (signalData) {
                const entry = signalData.entry_price, tp1 = signalData.target_price_1, sl = signalData.stop_loss;
                let progress = 0, color = 'transparent';
                if (price >= entry && tp1 > entry) {
                    progress = Math.min(100, ((price - entry) / (tp1 - entry)) * 100);
                    color = 'linear-gradient(90deg, var(--ok), #3fd1b0)';
                } else if (price < entry && entry > sl) {
                    progress = Math.min(100, ((entry - price) / (entry - sl)) * 100);
                    color = 'linear-gradient(90deg, var(--bad), #ff6b7a)';
                }
                const progressBar = el.querySelector('.progress span');
                if(progressBar) { progressBar.style.width = `${progress}%`; progressBar.style.background = color; }
            }
        });
        lastPrices[symbol] = price;
    });
}

function addNotification(notification) {
    const tbody = qs('#events tbody');
    tbody.insertAdjacentHTML('afterbegin', `<tr><td>${new Date(notification.timestamp).toLocaleTimeString('ar-EG')}</td><td>${notification.type||''}</td><td>${notification.message||''}</td></tr>`);
    if (tbody.rows.length > 20) tbody.deleteRow(-1);
}

function addRejection(rejection) {
    const tbody = qs('#rejections tbody');
    tbody.insertAdjacentHTML('afterbegin', `<tr><td>${new Date(rejection.timestamp).toLocaleTimeString('ar-EG')}</td><td>${rejection.symbol||''}</td><td>${rejection.reason||''}</td></tr>`);
    if (tbody.rows.length > 30) tbody.deleteRow(-1);
}

function updateMarketTrends(marketState) {
  const trendsContainer = document.getElementById('marketTrends');
  trendsContainer.innerHTML = '';
  if (marketState && marketState.trend_details_by_tf) {
    ['15m', '1h', '4h'].forEach(tf => {
      const trend = marketState.trend_details_by_tf[tf];
      if (trend) {
        let trendClass = 'amber', trendText = 'جانبي';
        if (trend.trend === 'bullish') { trendClass = 'green'; trendText = 'صاعد'; } 
        else if (trend.trend === 'bearish') { trendClass = 'red'; trendText = 'هابط'; }
        trendsContainer.innerHTML += `<div class="pill"><b>${tf}</b><span class="${trendClass}">${trendText}</span><small>ADX: ${trend.adx?.toFixed(1) || '—'}</small><small>RSI: ${trend.rsi?.toFixed(1) || '—'}</small></div>`;
      }
    });
  }
}

function updateTradingStatusUI(tradingEnabled) {
    const btn = document.getElementById('toggle-trading-btn');
    const statusBadge = document.getElementById('trading-status');
    if (tradingEnabled) {
        btn.textContent = 'إيقاف التداول';
        btn.className = 'btn btn-danger';
        statusBadge.textContent = 'مفعّل';
        statusBadge.className = 'badge bg-success';
    } else {
        btn.textContent = 'تشغيل التداول';
        btn.className = 'btn btn-success';
        statusBadge.textContent = 'معطّل';
        statusBadge.className = 'badge bg-danger';
    }
}

async function initializeDashboard() {
    try {
        showLoadingIndicator('#signals');
        const [baseRes, signalsRes] = await Promise.all([
            fetch('/api/dashboard_data'), fetch('/api/open_signals')
        ]);
        const baseData = await baseRes.json();
        const signalsData = await signalsRes.json();
        qs('#serverTime').textContent = new Date(baseData.server_time).toLocaleTimeString('ar-EG');
        updateTradingStatusUI(baseData.trading_enabled);
        updateMarketTrends(baseData.market_state);
        openSignals = signalsData.signals.reduce((acc, s) => { acc[s.id] = s; return acc; }, {});
        renderAllSignals(signalsData.signals);
    } catch (error) {
        console.error("فشل تحميل البيانات الأساسية:", error);
        qs('#signals').innerHTML = '<p>فشل تحميل البيانات. حاول تحديث الصفحة.</p>';
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
            case 'new_signal': openSignals[data.payload.id] = data.payload; renderAllSignals(Object.values(openSignals)); break;
            case 'trade_closed': delete openSignals[data.payload.signal_id]; renderAllSignals(Object.values(openSignals)); break;
            case 'new_notification': addNotification(data.payload); break;
            case 'new_rejection': addRejection(data.payload); break;
            case 'market_state_update': updateMarketTrends(data.payload); break;
            case 'trading_status_changed': updateTradingStatusUI(data.payload.trading_enabled); break;
            case 'open_trades_count_update': qs('#open-trades-count').textContent = `${data.payload.count}/${data.payload.max}`; break;
        }
    };
    socket.onclose = () => { console.log("WebSocket connection closed, reconnecting..."); setTimeout(setupWebSocket, 3000); };
    socket.onerror = (error) => console.error("WebSocket error:", error);
}

document.getElementById('toggle-trading-btn').addEventListener('click', function() {
    fetch('/api/trading/toggle', { method: 'POST' })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateTradingStatusUI(data.trading_enabled);
            showNotification(`تم ${data.trading_enabled ? 'تفعيل' : 'تعطيل'} التداول`, 'info');
        } else {
             showNotification('فشل تغيير حالة التداول', 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('فشل تغيير حالة التداول', 'error');
    });
});

document.addEventListener('DOMContentLoaded', () => { 
    // Add bootstrap JS for alert dismissal
    var script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js';
    document.head.appendChild(script);
    script.onload = () => {
        initializeDashboard(); 
        setupWebSocket(); 
    };
});
</script>
</body>
</html>
"""

# --- مسارات Flask ---
@app.route('/')
def dashboard():
    with trading_status_lock: trading_enabled = is_trading_enabled
    open_trades_count = get_open_trades_count()
    return render_template_string(DASHBOARD_TEMPLATE, 
                                  trading_enabled=trading_enabled,
                                  open_trades_count=open_trades_count,
                                  MAX_OPEN_TRADES=MAX_OPEN_TRADES)

@app.route('/api/dashboard_data')
def dashboard_data():
    try: return jsonify(get_dashboard_payload())
    except Exception as e:
        logger.error(f"❌ [API Error] Failed to generate dashboard data: {e}", exc_info=True)
        return jsonify({"error": "Failed to load dashboard data."}), 500

@app.route('/api/open_signals')
def get_open_signals():
    if not check_db_connection(): return jsonify({"error": "Database connection failed"}), 500
    sort_by = request.args.get('sort', 'id')
    allowed_sort_fields = ['id', 'symbol', 'entry_price', 'strategy_name', 'quality_score']
    if sort_by not in allowed_sort_fields: sort_by = 'id'
    order_direction = 'DESC' if sort_by in ['id', 'quality_score'] else 'ASC'
    sort_column_expression = sql.SQL("(signal_details->>'quality_score')::numeric")
    try:
        with conn.cursor() as cur:
            query = sql.SQL("SELECT id, symbol, entry_price, target_price_1, target_price_2, stop_loss, strategy_name, is_real_trade, quantity, signal_details, {sort_expression} as quality_score FROM signals WHERE status IN ('open', 'updated') ORDER BY {sort_col} {direction} NULLS LAST").format(sort_expression=sort_column_expression, sort_col=sql.Identifier(sort_by) if sort_by != 'quality_score' else sql.SQL('quality_score'), direction=sql.SQL(order_direction))
            cur.execute(query)
            signals = cur.fetchall()
        return jsonify({"signals": [dict(s) for s in signals]})
    except Exception as e:
        logger.error(f"Error fetching open signals: {e}")
        return jsonify({"error": str(e)}), 500

@sock.route('/ws')
def ws(ws_client):
    logger.info("WebSocket client connected.")
    with ws_clients_lock: ws_clients.append(ws_client)
    try:
        ws_client.send(json.dumps({"type": "connection_established"}, cls=NpEncoder))
        while True:
            message = ws_client.receive(timeout=30)
            if message is None: ws_client.send(json.dumps({"type": "ping"}, cls=NpEncoder))
    except Exception: logger.info("WebSocket client disconnected.")
    finally:
        with ws_clients_lock:
            if ws_client in ws_clients: ws_clients.remove(ws_client)

@app.route('/api/trading/toggle', methods=['POST'])
def toggle_trading():
    """تبديل حالة تشغيل البوت"""
    global is_trading_enabled
    with trading_status_lock:
        is_trading_enabled = not is_trading_enabled
        status = "مفعّل" if is_trading_enabled else "معطّل"
    
    log_and_notify("info", f"تم {status} التداول", "trading_status")
    
    broadcast({
        "type": "trading_status_changed",
        "payload": {"trading_enabled": is_trading_enabled}
    })
    
    return jsonify({"success": True, "trading_enabled": is_trading_enabled})

@app.route('/api/close_trade/<int:signal_id>', methods=['POST'])
def api_close_trade(signal_id):
    thread = Thread(target=close_trade_manually, args=(signal_id,))
    thread.start()
    return jsonify({"success": True, "message": "Trade close command received and is being processed."})

def close_trade_manually(signal_id: int, closing_price: Optional[float] = None) -> bool:
    with signal_cache_lock:
        signal_to_close = next((dict(s) for s in open_signals_cache.values() if s['id'] == signal_id), None)
    if not signal_to_close:
        logger.warning(f"[Manual Close] Signal {signal_id} not found in active cache.")
        return False
    symbol = signal_to_close['symbol']
    if closing_price is None:
        with live_prices_lock: closing_price = live_prices.get(symbol)
        if closing_price is None:
            logger.error(f"[Manual Close] Could not get live price for {symbol} to close signal {signal_id}.")
            send_enhanced_telegram_message(f"⚠️ *فشل الإغلاق اليدوي لـ {symbol}* \nلم يتمكن البوت من الحصول على السعر الحالي.", force=True)
            return False
    logger.info(f"[Manual Close] User initiated manual close for signal {signal_id} ({symbol}) at price {closing_price}")
    close_signal(signal_to_close, closing_price, "manual_close")
    return True

# --- Main Logic & Loops ---
def check_max_open_trades() -> bool:
    """التحقق من عدد التوصيات المفتوحة وعدم تجاوز الحد الأقصى"""
    with signal_cache_lock:
        open_trades_count = len(open_signals_cache)
    
    if open_trades_count >= MAX_OPEN_TRADES:
        logger.info(f"[Max Trades] تم الوصول إلى الحد الأقصى من التوصيات المفتوحة: {open_trades_count}/{MAX_OPEN_TRADES}. سيتم تجاهل التوصيات الجديدة.")
        return False
    
    return True

def main_bot_loop():
    logger.info("🚀 [Main Loop] Starting signal scanning loop...")
    while True:
        try:
            with trading_status_lock:
                if not is_trading_enabled:
                    time.sleep(10)
                    continue
            
            if not check_max_open_trades():
                time.sleep(120)
                continue
            
            logger.info("="*20 + " Starting New Scan Cycle " + "="*20)
            for symbol in validated_symbols_to_scan:
                with signal_cache_lock:
                    if symbol in open_signals_cache: continue
                
                scan_symbol(symbol)
            
            logger.info("="*20 + " Scan Cycle Completed " + "="*20)
            time.sleep(60 * 5)
        except Exception as e:
            logger.error(f"❌ [Main Loop] A critical error occurred: {e}", exc_info=True)
            time.sleep(60)

def scan_symbol(symbol: str) -> None:
    """فحص عملة معينة للبحث عن فرص تداول"""
    # التحقق من حالة التداول (احتياطي)
    with trading_status_lock:
        if not is_trading_enabled: return
    
    df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
    if df is None or len(df) < 200:
        if df is not None: log_rejection(symbol, "Insufficient Historical Data")
        return
    df_featured = calculate_all_features(df); df_featured.name = symbol
    
    if not check_market_volatility_filter(df_featured): return
    if not check_trend_strength_filter(df_featured, BASE_FILTER_ADX_THRESHOLD): return
    
    strategy_found = None
    if USE_BB_STOCH_STRATEGY and check_bb_stoch_strategy_enhanced(df_featured): strategy_found = "BB_Stoch_Strategy"
    elif USE_MACD_EMA_STRATEGY and check_macd_ema_strategy_enhanced(df_featured): strategy_found = "MACD_EMA_Strategy"
    elif USE_EMA_RSI_STRATEGY and check_ema_rsi_strategy_enhanced(df_featured): strategy_found = "EMA_RSI_Strategy"
    elif USE_PULLBACK_STRATEGY and check_pullback_strategy_enhanced(df_featured): strategy_found = "Pullback_Strategy"
    elif USE_MOMENTUM_VOLATILITY_STRATEGY and check_momentum_volatility_strategy(df_featured): strategy_found = "Momentum_Volatility_Strategy"
    
    if strategy_found and apply_strategy_filters(symbol, df_featured, strategy_found):
        create_trade_signal(symbol, df_featured, strategy_found)

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

def close_signal(signal: Dict, closing_price: float, reason: str):
    symbol, signal_id, entry_price = signal['symbol'], signal['id'], signal['entry_price']
    
    with signal_cache_lock:
        if symbol not in open_signals_cache or open_signals_cache[symbol]['id'] != signal_id:
            logger.warning(f"[Close Signal] Attempted to close already closed or non-existent signal {signal_id} for {symbol}.")
            return

    if signal.get('is_real_trade'):
        try:
            quantity_in_bot = Decimal(str(signal.get('quantity', 0)))
            if quantity_in_bot > 0:
                asset = symbol.replace("USDT", "")
                asset_balance_info = client.get_asset_balance(asset=asset)
                available_on_exchange = Decimal(asset_balance_info.get('free', '0.0'))
                quantity_to_sell = min(quantity_in_bot, available_on_exchange)
                if quantity_to_sell > 0:
                    adjusted_quantity_to_sell = adjust_quantity_to_lot_size(symbol, float(quantity_to_sell))
                    if adjusted_quantity_to_sell and adjusted_quantity_to_sell > 0:
                        if not place_order(symbol, Client.SIDE_SELL, adjusted_quantity_to_sell):
                            log_and_notify('error', f"CRITICAL: Final sell order placement failed for {symbol}. Trade remains open.", "TRADE_ERROR")
                            return
        except Exception as e:
            logger.error(f"❌ [{symbol}] Critical error during real trade closure: {e}", exc_info=True)
            return

    profit = ((closing_price - entry_price) / entry_price) * 100
    
    with consecutive_losses_lock:
        if profit < 0: consecutive_losses_by_symbol[symbol] = consecutive_losses_by_symbol.get(symbol, 0) + 1
        else: consecutive_losses_by_symbol[symbol] = 0
            
    update_signal_in_db(signal_id, {"status": "closed", "closing_price": closing_price, "closed_at": datetime.now(timezone.utc), "profit_percentage": profit, "closing_reason": reason})
    
    with signal_cache_lock:
        if symbol in open_signals_cache: del open_signals_cache[symbol]
    
    broadcast({"type": "trade_closed", "payload": {"signal_id": signal_id, "symbol": symbol}})
    broadcast({"type": "open_trades_count_update", "payload": {"count": get_open_trades_count(), "max": MAX_OPEN_TRADES}})

    trade_type = "حقيقية" if signal.get('is_real_trade') else "ورقية"
    result_emoji = "✅" if profit >= 0 else "🔻"
    reason_map = {"SL_HIT": "ضرب وقف الخسارة", "TP1_HIT": "تحقيق الهدف الأول", "TP2_HIT": "تحقيق الهدف الثاني", "manual_close": "إغلاق يدوي", "TRAILING_SL_HIT": "ضرب الوقف المتحرك"}
    reason_ar = reason_map.get(reason, reason)
    log_and_notify("info", f"Closed {trade_type} trade for {symbol}. Profit: {profit:.2f}%", "TRADE_CLOSED")
    settings = get_notification_settings()
    if (profit >= settings['min_profit_notification'] or profit <= settings['max_loss_notification'] or reason == "manual_close"):
        send_enhanced_telegram_message(f"{result_emoji} *إغلاق صفقة {trade_type} {symbol}*\n*السبب:* {reason_ar}\n*الربح:* `{profit:.2f}%`")

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
                details = signal.get('signal_details')
                if isinstance(details, str):
                    try: details = json.loads(details)
                    except Exception: details = {}
                details = details or {}
                entry_price = float(signal.get('entry_price', 0))
                stop_loss = float(signal.get('stop_loss', 0))
                tp1 = float(signal.get('target_price_1') or 0)
                tp2 = float(signal.get('target_price_2') or 0)
                trail_dist = float(details.get('trailing_stop_distance') or 0)
                remaining_qty = float(signal.get('quantity') or 0)

                if stop_loss and current_price <= stop_loss: close_signal(signal, stop_loss, "SL_HIT"); continue
                if tp2 and current_price >= tp2: close_signal(signal, tp2, "TP2_HIT"); continue
                if tp1 and not details.get('tp1_done') and remaining_qty > 0 and current_price >= tp1:
                    part_qty_to_close = remaining_qty * 0.5
                    if signal.get('is_real_trade'):
                        asset = symbol.replace("USDT", "")
                        asset_balance_info = client.get_asset_balance(asset=asset)
                        available_on_exchange = Decimal(asset_balance_info.get('free', '0.0'))
                        quantity_to_sell = min(Decimal(str(part_qty_to_close)), available_on_exchange)
                        if quantity_to_sell > 0:
                            adjusted_qty = adjust_quantity_to_lot_size(symbol, float(quantity_to_sell))
                            if adjusted_qty and adjusted_qty > 0: place_order(symbol, Client.SIDE_SELL, adjusted_qty)
                    new_sl = max(stop_loss, entry_price)
                    updates = {"quantity": remaining_qty - part_qty_to_close, "stop_loss": new_sl, "status": "updated", "closing_reason": "TP1_HIT"}
                    details['tp1_done'] = True
                    updates['signal_details'] = json.dumps(details)
                    update_signal_in_db(signal['id'], updates)
                    with signal_cache_lock:
                        if symbol in open_signals_cache:
                            open_signals_cache[symbol].update(updates)
                            open_signals_cache[symbol]['signal_details'] = details
                    send_enhanced_telegram_message(f"🥇 *تحقق الهدف الأول* لـ `{symbol}`\nتم إقفال 50% من العقد وتحريك الوقف إلى نقطة الدخول.")
                    broadcast({"type": "signal_update", "payload": open_signals_cache.get(symbol, {})})
                    continue
                profit_pct = ((current_price - entry_price) / max(entry_price, 1e-8)) * 100 if entry_price else 0
                if trail_dist and not details.get('trailing_active') and profit_pct >= TRAILING_STOP_ACTIVATION_PROFIT_PERCENT:
                    details['trailing_active'] = True
                    update_signal_in_db(signal['id'], {"signal_details": json.dumps(details)})
                    with signal_cache_lock:
                        if symbol in open_signals_cache: open_signals_cache[symbol]['signal_details'] = details
                    send_enhanced_telegram_message(f"📈 *تفعيل الوقف المتحرك* لـ `{symbol}` عند ربح `{profit_pct:.2f}%`.")
                if details.get('trailing_active') and trail_dist:
                    new_sl = max(stop_loss, current_price - trail_dist)
                    if new_sl > stop_loss:
                        update_signal_in_db(signal['id'], {"stop_loss": new_sl})
                        with signal_cache_lock:
                            if symbol in open_signals_cache: open_signals_cache[symbol]['stop_loss'] = new_sl
                        send_enhanced_telegram_message(f"🔧 *تحديث الوقف المتحرك* لـ `{symbol}` → `{new_sl:.6f}`")
            time.sleep(1)
        except Exception as e:
            logger.error(f"❌ [Trade Manager] Loop error: {e}", exc_info=True)
            time.sleep(2)

def update_market_state():
    global current_market_state
    try:
        btc_df = fetch_historical_data(BTC_SYMBOL, '1h', days=10)
        if btc_df is None or len(btc_df) < 200: logger.warning("[Market State] Insufficient BTC data"); return
        btc_df = calculate_all_features(btc_df)
        last_btc = btc_df.iloc[-1]
        btc_trend = "sideways"
        if last_btc['close'] > last_btc['ema200'] and last_btc['macd_hist'] > 0: btc_trend = "bullish"
        elif last_btc['close'] < last_btc['ema200'] and last_btc['macd_hist'] < 0: btc_trend = "bearish"
        trend_details = {}
        for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
            try:
                tf_df = fetch_historical_data(BTC_SYMBOL, tf, days=15)
                if tf_df is not None and len(tf_df) >= 50:
                    tf_df = calculate_all_features(tf_df)
                    last_tf = tf_df.iloc[-1]
                    tf_trend = "sideways"
                    if last_tf['close'] > last_tf['ema50'] and last_tf['adx'] > 20: tf_trend = "bullish"
                    elif last_tf['close'] < last_tf['ema50'] and last_tf['adx'] > 20: tf_trend = "bearish"
                    trend_details[tf] = {"trend": tf_trend, "adx": last_tf.get('adx', 0), "rsi": last_tf.get('rsi', 50), "price_change": ((last_tf['close'] - tf_df.iloc[-10]['close']) / tf_df.iloc[-10]['close']) * 100 if len(tf_df) >= 10 else 0}
            except Exception as e: logger.error(f"[Market State] Error analyzing {tf} timeframe: {e}")
        with market_state_lock:
            current_market_state = {"btc_trend": btc_trend, "btc_price": last_btc['close'], "btc_adx": last_btc.get('adx', 0), "btc_rsi": last_btc.get('rsi', 50), "trend_details_by_tf": trend_details, "last_updated": datetime.now(timezone.utc).isoformat()}
        broadcast({"type": "market_state_update", "payload": current_market_state})
    except Exception as e: logger.error(f"[Market State] Error updating market state: {e}", exc_info=True)

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
    except Exception as e: logger.error(f"❌ [Balance] Could not update USDT balance: {e}")

def update_balance_loop():
    logger.info("🚀 [Balance Updater] Starting balance update loop...")
    while True:
        try: update_balance()
        except Exception as e: logger.error(f"❌ [Balance Loop] Error: {e}", exc_info=True)
        time.sleep(60 * 10)

# --- نقطة بداية البرنامج ---
if __name__ == '__main__':
    logger.info("="*50 + "\n====== Starting Crypto Trading Bot V25.4.0 (Advanced Management) ======\n" + "="*50)
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
    start_periodic_reports()
    logger.info("🌐 [Flask] Starting UI on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
