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
from waitress import serve  # =====> تم إضافة هذا السطر لحل الخطأ
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

# ضبط دقة النوع Decimal
getcontext().prec = 18

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v36_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV36')

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
MIN_SIGNAL_QUALITY: int = 60 # تم التخفيف: من 70 إلى 60
AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE: bool = True
min_quality_lock = Lock()
strategy_filters_lock = Lock()

# --- إعدادات عامة (معدلة لإطار 5 دقائق) ---
SIGNAL_GENERATION_TIMEFRAME: str = '5m'
HIGHER_TIMEFRAME: str = '15m'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['5m', '15m', '1h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 7
# --- NEW: قائمة العملات القيادية لتحديد اتجاه السوق ---
MARKET_LEADERS: List[str] = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
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
current_market_state: Dict[str, Any] = {"trend_details_by_tf": {}, "market_consensus": "sideways", "market_score": 0.5} # تم التعديل
market_state_lock = Lock()

# --- قاموس أسباب الرفض باللغة العربية ---
REJECTION_REASONS_AR = {
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
    "Smart Market Structure Filter Failed": "فلتر هيكل السوق الذكي رفض الدخول",
    "Advanced Market Structure Filter Failed": "فلتر هيكل السوق المتقدم V2 رفض الدخول",
    "Smart Liquidity Filter Failed": "فلتر السيولة الذكي رفض الدخول",
    "Smart Risk/Reward Filter Failed": "فلتر المخاطرة/العائد الذكي رفض الدخول",
    "Market Consensus Filter Failed": "فلتر اتجاه السوق العام رفض الدخول (السوق هابط)", # إضافة
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
    with trade_amount_lock:
        trade_amount_min = FIXED_TRADE_AMOUNT_MIN_USDT
        trade_amount_max = FIXED_TRADE_AMOUNT_MAX_USDT

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

def send_trade_open_notification(symbol: str, strategy_key: str, entry_price: float, stop_loss: float,
                                target1: float, target2: float, quantity: float, is_real: bool,
                                quality_score: int, atr_percent: float, notional_value: float):
    trade_type = "حقيقية" if is_real else "ورقية"
    emoji = "🔥" if is_real else "📊"
    # --- FIX: Check if STRATEGY_NAMES is defined, otherwise use key ---
    strategy_name_ar = STRATEGY_NAMES.get(strategy_key, strategy_key) if 'STRATEGY_NAMES' in globals() else strategy_key
    
    message = (
        f"{emoji} *صفقة {trade_type} جديدة (5 دقائق)*\n\n"
        f"*العملة:* `{symbol}`\n"
        f"*الاستراتيجية:* `{strategy_name_ar}`\n"
        f"*جودة الإشارة:* `{quality_score}/100`\n"
        f"*تقلب السوق:* `{atr_percent:.2f}%`\n\n"
        f"*سعر الدخول:* `{entry_price:.4f}`\n"
        f"*وقف الخسارة:* `{stop_loss:.4f}`\n"
        f"*الهدف الأول:* `{target1:.4f}`\n"
        f"*الهدف الثاني:* `{target2:.4f}`\n\n"
        f"*الكمية:* `{quantity:.4f}`\n"
        f"*قيمة الصفقة:* `${notional_value:.2f}`\n"
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
    
    if not state or not state.get("market_consensus"):
        return

    message = f"🌐 *تحديث حالة السوق*\n\n"
    
    # --- NEW: إرسال ملخص اتجاه السوق الجديد ---
    consensus = state.get("market_consensus", "sideways")
    score = state.get("market_score", 0.5) * 100
    emoji = "🟢" if consensus == "bullish" else "🔴" if consensus == "bearish" else "🟡"
    
    message += f"{emoji} *اتجاه السوق العام: {consensus.capitalize()}*\n"
    message += f"*قوة الاتجاه (٪ صاعد):* `{score:.1f}%`\n\n"
    
    message += "*تفاصيل العملات القيادية:*\n"
    for leader, details in state.get("leader_trends", {}).items():
        leader_emoji = "🟢" if details.get("trend") == "bullish" else "🔴" if details.get("trend") == "bearish" else "🟡"
        message += f"{leader_emoji} *{leader}:* {details.get('trend', 'N/A')}\n"

    send_enhanced_telegram_message(message, force=False)

def schedule_periodic_reports():
    logger.info("Starting periodic reports scheduler...")
    while True:
        try:
            now = datetime.now(timezone.utc)
            if now.hour == 23 and now.minute == 59:
                send_daily_performance_report()
                time.sleep(61)
            # --- NEW: إرسال تحديث السوق كل 4 ساعات ---
            if now.hour % 4 == 0 and now.minute == 1:
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
    # لا نضع تأخير هنا لأننا قد نحتاجه في حلقات سريعة
    # time.sleep(API_REQUEST_DELAY) 
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
    high_low_range = high_14 - high_14
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
    # ===== Signal Quality Scoring System (Relaxed) =====
def calculate_signal_quality_score(df: pd.DataFrame, mtf_trend: Dict) -> int:
    """
    يحسب نقاط جودة للإشارة (من 0 إلى 100) بناءً على عدة عوامل فنية.
    (تم تخفيف الشروط لزيادة الجودة المتوسطة)
    """
    score = 0
    last = df.iloc[-1]
    
    # 1. قوة الاتجاه (Trend Strength) - (Max 25 points)
    ema_spread = (last['ema21'] - last['ema50']) / last['close'] * 100
    if ema_spread > 0.4: # تخفيف: من 0.5
        score += 25
    elif ema_spread > 0.15: # تخفيف: من 0.2
        score += 15
    
    # 2. تأكيد الحجم (Volume Confirmation) - (Max 20 points)
    volume_ma20 = df['volume'].rolling(20).mean().iloc[-1]
    if last['volume'] > volume_ma20 * 1.8: # تخفيف: من 2.0
        score += 20
    elif last['volume'] > volume_ma20 * 1.3: # تخفيف: من 1.5
        score += 10
        
    # 3. زخم المؤشرات (Indicator Momentum) - (Max 25 points)
    # RSI
    if 55 < last['rsi'] < 75: # توسيع النطاق: من 60-70
        score += 15
    elif last['rsi'] > 50: # تخفيف: من 55
        score += 5
    # MACD
    if last['macd_hist'] > 0 and df['macd_hist'].iloc[-1] > df['macd_hist'].iloc[-2]:
        score += 10
        
    # 4. التوافق الزمني (Multi-Timeframe Alignment) - (Max 20 points)
    if mtf_trend.get('15m') in ['bullish', 'sideways']: # تخفيف: السماح بـ 'sideways'
        score += 10
    if mtf_trend.get('1h') in ['bullish', 'sideways']: # تخفيف: السماح بـ 'sideways'
        score += 10
        
    # 5. التقلب (Volatility) - (Max 10 points)
    atr_percent = last.get('atr_percent', 0)
    if 0.8 < atr_percent < 3.0: # توسيع النطاق: من 1.0-2.5
        score += 10
    elif 0.6 < atr_percent < 4.0: # توسيع النطاق: من 0.7-3.5
        score += 5

    return min(100, int(score))

# --- Data Loading & Settings Management ---
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
    global FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT, MAX_OPEN_TRADES, paper_trading_mode, MIN_SIGNAL_QUALITY
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
            with min_quality_lock: MIN_SIGNAL_QUALITY = quality_settings.get('min_quality', 60) # تم التعديل

        strategies_data = redis_client.get('strategy_settings')
        if strategies_data:
            strategies_enabled_status = json.loads(strategies_data)
            with strategy_filters_lock:
                # --- FIX: Check if ENHANCED_STRATEGIES is defined before access ---
                if 'ENHANCED_STRATEGIES' in globals():
                    for key, enabled in strategies_enabled_status.items():
                        if key in ENHANCED_STRATEGIES:
                            ENHANCED_STRATEGIES[key]['enabled'] = enabled
                else:
                    logger.warning("[Redis] ENHANCED_STRATEGIES not defined yet, skipping strategy settings load.")


        logger.info("✅ [Redis] Successfully loaded settings from Redis.")
    except Exception as e:
        logger.error(f"❌ [Redis] Error loading settings: {e}")

def save_settings_to_redis():
    if not redis_client:
        logger.warning("Redis client not available, cannot save settings")
        return False
    
    try:
        with trade_amount_lock:
            trade_amount_min = FIXED_TRADE_AMOUNT_MIN_USDT
            trade_amount_max = FIXED_TRADE_AMOUNT_MAX_USDT
        with trading_mode_lock:
            is_paper = paper_trading_mode
        with min_quality_lock:
            min_quality = MIN_SIGNAL_QUALITY

        trading_settings = {
            'FIXED_TRADE_AMOUNT_MIN_USDT': trade_amount_min,
            'FIXED_TRADE_AMOUNT_MAX_USDT': trade_amount_max,
            'MAX_OPEN_TRADES': MAX_OPEN_TRADES,
            'paper_trading_mode': is_paper
        }
        redis_client.set('trading_settings', json.dumps(trading_settings))
        
        quality_settings = {'min_quality': min_quality}
        redis_client.set('signal_quality_settings', json.dumps(quality_settings))
        
        with strategy_filters_lock:
            # --- FIX: Check if ENHANCED_STRATEGIES is defined before access ---
            if 'ENHANCED_STRATEGIES' in globals():
                strategy_settings = {key: info['enabled'] for key, info in ENHANCED_STRATEGIES.items()}
                redis_client.set('strategy_settings', json.dumps(strategy_settings))
            else:
                logger.warning("[Redis] ENHANCED_STRATEGIES not defined yet, skipping strategy settings save.")
        
        logger.info("Settings saved to Redis successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error saving settings to Redis: {e}")
        return False

# --- General Filters (Relaxed) ---
def add_news_filter() -> bool:
    # تم التعطيل مؤقتاً لزيادة الإشارات، يمكن إعادته لاحقاً
    # news_hours = [(12, 30), (14, 0), (18, 30)]
    # now = datetime.now(timezone.utc)
    # for hour, minute in news_hours:
    #     if now.hour == hour and abs(now.minute - minute) <= 15:
    #         return False
    return True

def add_liquidity_filter() -> bool:
    # تم التعطيل مؤقتاً لزيادة الإشارات
    # now = datetime.now(timezone.utc)
    # if now.weekday() >= 5: return False
    # if now.hour >= 22 or now.hour <= 2: return False
    return True

def add_correlation_filter(new_symbol: str) -> bool:
    # تم التعطيل مؤقتاً لزيادة الإشارات
    # correlated_groups = [
    #     {'BTCUSDT', 'ETHUSDT', 'BCHUSDT'}, {'ADAUSDT', 'DOTUSDT', 'LINKUSDT'},
    #     {'SOLUSDT', 'AVAXUSDT', 'MATICUSDT'},
    # ]
    # with signal_cache_lock: open_symbols = set(open_signals_cache.keys())
    # if not open_symbols: return True
    # for group in correlated_groups:
    #     if new_symbol in group and not open_symbols.isdisjoint(group):
    #         return False
    return True

def check_market_volatility_filter_enhanced(df: pd.DataFrame, symbol: str = "Unknown") -> bool:
    if 'atr_percent' not in df.columns or df['atr_percent'].isnull().all():
        log_rejection(symbol, "Market Volatility Filter Failed", {"reason": "No ATR data"})
        return False
    
    last_atr_percent = float(df.iloc[-1].get('atr_percent', 0))
    ATR_PERCENT_MIN = 0.25 # تخفيف: من 0.30
    ATR_PERCENT_MAX = 4.0 # تخفيف: من 3.2
    
    if not (ATR_PERCENT_MIN <= last_atr_percent <= ATR_PERCENT_MAX):
        log_rejection(symbol, "Market Volatility Filter Failed", {
            "atr": f"{last_atr_percent:.2f}%",
            "range": f"({ATR_PERCENT_MIN:.2f}-{ATR_PERCENT_MAX:.2f})%"
        })
        return False
    
    return True

# ===== NEW: ENHANCED TRADING STRATEGIES V36.0 (Relaxed) =====
# ===== استراتيجيات مخففة + استراتيجيات جديدة بسيطة =====

def check_smart_momentum_strategy(df: pd.DataFrame, mtf_trend: Dict, market_consensus: str) -> bool:
    """
    استراتيجية الزخم الذكي - (شروط مخففة)
    """
    if len(df) < 200: return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. التحقق من الاتجاه (تخفيف)
    if not (mtf_trend.get('15m') in ['bullish', 'sideways']): return False
    if market_consensus == 'bearish': return False # فلتر السوق العام

    # 2. ترتيب EMAs (تخفيف كبير)
    ema_alignment = (
        last['ema21'] > last['ema50'] and last['close'] > last['ema21']
    )
    if not ema_alignment: return False
    
    # 3. MACD (تخفيف)
    macd_strong = (
        last['macd_hist'] > 0 and
        last['macd_hist'] > prev['macd_hist'] # يكفي أن يكون الهيستوجرام متزايداً
    )
    if not macd_strong: return False
    
    # 4. ADX (تخفيف)
    if not last['adx'] > 20: return False # تخفيف: من 22
    
    # 5. RSI (تخفيف)
    if not (50 < last['rsi'] < 80): return False # توسيع النطاق: من 55-75
    
    # 6. حجم التداول (تخفيف)
    volume_ma = df['volume'].rolling(20).mean()
    if not last['volume'] > volume_ma.iloc[-1] * 1.1: return False # تخفيف: من 1.3
    
    # 7. السعر فوق VWAP (اختياري)
    # if not last['close'] > last.get('vwap', last['close']): return False
    
    # 8. تقلب مناسب (تخفيف)
    atr_percent = last.get('atr_percent', 0)
    if not (0.7 < atr_percent < 3.5): return False # توسيع: من 0.8-3.0
    
    return True
# ( ... (trading_bot_v36_part1.py) ... )

def check_advanced_pullback_strategy(df: pd.DataFrame, mtf_trend: Dict, market_consensus: str) -> bool:
    """
    استراتيجية الارتداد المتقدمة - (شروط مخففة)
    """
    if len(df) < 200: return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2] # إضافة prev
    
    # 1. اتجاه صاعد (تخفيف)
    if not (mtf_trend.get('15m') in ['bullish', 'sideways']): return False
    if market_consensus == 'bearish': return False

    # 2. EMAs في ترتيب صاعد (تخفيف)
    if not (last['ema21'] > last['ema50']): return False # يكفي أن يكون 21 فوق 50
    
    # 3. تحديد الارتداد: السعر لمس EMA21 (توسيع النطاق)
    recent_lows = df['low'].tail(8) # توسيع: من 5 إلى 8 شموع
    recent_ema21 = df['ema21'].tail(8)
    if not (recent_lows <= recent_ema21 * 1.008).any(): return False # توسيع: من 1.005 إلى 1.008
    
    # 4. قياس عمق الارتداد (تخفيف)
    recent_high = df['high'].tail(15).max() # توسيع: من 10 إلى 15
    pullback_low = recent_lows.min()
    pullback_depth = (recent_high - pullback_low) / recent_high
    atr_percent = last.get('atr_percent', 0)
    min_depth = 0.008 # تخفيف: من 0.010
    max_depth = 0.07 # تخفيف: من 0.05
    if not (min_depth <= pullback_depth <= max_depth): return False
    
    # 5. السعر بدأ في التعافي (تخفيف)
    if not (last['close'] > last['ema9'] or last['close'] > prev['close']): return False # يكفي إغلاق أعلى من 9 أو أعلى من الإغلاق السابق
    
    # 6. حجم تداول (اختياري)
    # volume_ma = df['volume'].rolling(20).mean()
    # if not last['volume'] > volume_ma.iloc[-1] * 1.2: return False
    
    # 7. RSI (تخفيف)
    if last.get('rsi', 50) > 78: return False # تخفيف: من 72
    
    # 8. Stochastic (تخفيف)
    stoch_reversal = (
        df['stoch_k'].iloc[-3:].min() < 40 and # تخفيف: من 35
        last['stoch_k'] > df['stoch_k'].iloc[-2]
    )
    if not stoch_reversal: return False
    
    return True


def check_breakout_retest_strategy(df: pd.DataFrame, mtf_trend: Dict, market_consensus: str) -> bool:
    """
    استراتيجية الاختراق وإعادة الاختبار - (شروط مخففة)
    """
    if len(df) < 100: return False
    last = df.iloc[-1]
    
    if market_consensus == 'bearish': return False

    try:
        highs = df['high'].values
        resistance_indices = argrelextrema(highs, np.greater, order=5)[0] # تخفيف: من 7 إلى 5
        if len(resistance_indices) < 2: return False
        
        latest_resistance_idx = resistance_indices[-1]
        resistance_price = highs[latest_resistance_idx]
        
        # 1. تم الاختراق (تخفيف)
        if not (df['high'].tail(15) > resistance_price * 1.001).any(): return False # توسيع: 10->15, تخفيف: 1.002->1.001
        
        # 2. إعادة الاختبار (تخفيف)
        retest_zone_upper = resistance_price * 1.012 # توسيع: 1.008 -> 1.012
        retest_zone_lower = resistance_price * 0.995 # توسيع: 0.998 -> 0.995
        if not (retest_zone_lower <= last['close'] <= retest_zone_upper): return False
        
        # 3. حجم الاختراق (تخفيف)
        breakout_candle_idx_series = df[df['high'] > resistance_price * 1.001].index
        if breakout_candle_idx_series.empty: return False # التأكد من وجود اختراق
        breakout_candle_idx = breakout_candle_idx_series[-1]
        
        breakout_volume = df.loc[breakout_candle_idx, 'volume']
        volume_ma = df['volume'].rolling(20).mean().loc[breakout_candle_idx]
        if not breakout_volume > volume_ma * 1.3: return False # تخفيف: 1.5 -> 1.3
        
        # 4. تأكيدات (تخفيف)
        if not (last['ema9'] > last['ema21']): return False # تخفيف: 9>21>50
        if last['macd_hist'] <= - (last['close'] * 0.0001): return False # السماح بـ MACD سلبي طفيف
        if last['adx'] < 18: return False # تخفيف: 20 -> 18
        
        return True
    except Exception:
        return False

# --- (تم تخطي الاستراتيجيات الأخرى المعقدة من الملف الأصلي لعدم تفعيلها) ---
# ... (check_volume_price_divergence_strategy, check_golden_cross_momentum_strategy, etc. are omitted as they were disabled) ...


# ===== NEW: Simple Strategies (V36) =====

def check_simple_ema_cross_strategy(df: pd.DataFrame, mtf_trend: Dict, market_consensus: str) -> bool:
    """
    استراتيجية جديدة: تقاطع EMA بسيط
    تبحث عن تقاطع EMA9 فوق EMA21 مع اتجاه عام مساعد
    """
    if len(df) < 50: return False
    last, prev = df.iloc[-1], df.iloc[-2]

    # 1. فلتر السوق العام
    if market_consensus == 'bearish': return False
    
    # 2. التحقق من التقاطع
    ema_cross = (prev['ema9'] <= prev['ema21'] and last['ema9'] > last['ema21'])
    if not ema_cross: return False

    # 3. السعر يجب أن يكون فوق EMA50 (تأكيد الاتجاه)
    if not last['close'] > last['ema50']: return False

    # 4. RSI في منطقة جيدة (ليس تشبع شرائي)
    if not (45 < last['rsi'] < 75): return False

    # 5. ADX (اختياري، لكن يفضل أن يكون فوق 18)
    if not last['adx'] > 18: return False

    return True

def check_simple_bollinger_bounce_strategy(df: pd.DataFrame, mtf_trend: Dict, market_consensus: str) -> bool:
    """
    استراتيجية جديدة: ارتداد بولينجر باند بسيط
    للأسواق الجانبية أو الصاعدة، تشتري عند الارتداد من الحد السفلي
    """
    if len(df) < 50: return False
    last, prev = df.iloc[-1], df.iloc[-2]

    # 1. فلتر السوق العام (لا يعمل في هبوط قوي)
    if market_consensus == 'bearish': return False

    # 2. السعر لمس أو اقترب من الحد السفلي للبولينجر
    bb_touch = (df['low'].tail(3) <= df['bb_lower'].tail(3) * 1.003).any()
    if not bb_touch: return False

    # 3. السعر بدأ في الارتداد (الإغلاق الحالي أعلى من السابق)
    if not last['close'] > prev['close']: return False
    
    # 4. RSI يظهر خروج من التشبع البيعي (تخفيف)
    rsi_ok = 25 < last.get('rsi', 50) < 50 # نطاق أوسع
    if not rsi_ok: return False
    
    # 5. لا يوجد اتجاه هابط قوي في الأطر الأعلى
    if mtf_trend.get('15m') == 'bearish': return False

    return True

# ===== ADVANCED SMART MARKET STRUCTURE FILTER V2 (Relaxed) =====

def detect_market_structure(df: pd.DataFrame) -> Dict[str, any]:
    """
    تحليل متقدم لهيكل السوق - (شروط مخففة)
    """
    if len(df) < 50:
        return {"structure_type": "unknown", "strength": 0}
    
    highs = df['high'].values
    lows = df['low'].values
    
    swing_high_indices = argrelextrema(highs, np.greater, order=3)[0] # تخفيف: 3
    swing_low_indices = argrelextrema(lows, np.less, order=3)[0] # تخفيف: 3
    
    structure_data = {
        "swing_highs": [(i, highs[i]) for i in swing_high_indices[-5:]] if len(swing_high_indices) >= 2 else [],
        "swing_lows": [(i, lows[i]) for i in swing_low_indices[-5:]] if len(swing_low_indices) >= 2 else [],
        "structure_type": "ranging",
        "strength": 0,
        "bos_detected": False,
        "choch_detected": False
    }
    
    if len(swing_high_indices) >= 2 and len(swing_low_indices) >= 2: # تخفيف: 3->2
        recent_highs = highs[swing_high_indices[-2:]] # تخفيف: 3->2
        recent_lows = lows[swing_low_indices[-2:]] # تخفيف: 3->2
        
        # التأكد من وجود عناصر للمقارنة
        highs_rising = False
        if len(recent_highs) > 1:
            highs_rising = all(recent_highs[i] < recent_highs[i+1] for i in range(len(recent_highs)-1))
        
        lows_rising = False
        if len(recent_lows) > 1:
            lows_rising = all(recent_lows[i] < recent_lows[i+1] for i in range(len(recent_lows)-1))
        
        highs_falling = False
        if len(recent_highs) > 1:
            highs_falling = all(recent_highs[i] > recent_highs[i+1] for i in range(len(recent_highs)-1))

        lows_falling = False
        if len(recent_lows) > 1:
            lows_falling = all(recent_lows[i] > recent_lows[i+1] for i in range(len(recent_lows)-1))

        
        if highs_rising and lows_rising:
            structure_data["structure_type"] = "bullish"
            structure_data["strength"] = 85
        elif highs_falling and lows_falling:
            structure_data["structure_type"] = "bearish"
            structure_data["strength"] = 15
        elif highs_rising: # تخفيف: السماح فقط بالقمم الصاعدة
            structure_data["structure_type"] = "weak_bullish"
            structure_data["strength"] = 65 # زيادة: 60->65
        elif lows_rising: # تخفيف: السماح فقط بالقيعان الصاعدة
            structure_data["structure_type"] = "weak_bullish"
            structure_data["strength"] = 60
        elif highs_falling:
            structure_data["structure_type"] = "weak_bearish"
            structure_data["strength"] = 40
        else:
            structure_data["structure_type"] = "ranging"
            structure_data["strength"] = 50 # زيادة: (السماح بالأسواق الجانبية)
    
    # ... (BOS/CHoCH logic) ...
    if len(swing_high_indices) >= 2 and len(swing_low_indices) >= 2:
        last_swing_high = highs[swing_high_indices[-2]]
        last_swing_low = lows[swing_low_indices[-2]]
        current_price = df['close'].iloc[-1]
        if current_price > last_swing_high * 1.002:
            structure_data["bos_detected"] = True
            structure_data["bos_direction"] = "bullish"
        if structure_data["structure_type"] in ["bullish", "weak_bullish"]:
            if current_price < last_swing_low * 0.998:
                structure_data["choch_detected"] = True
                structure_data["choch_direction"] = "bearish"

    return structure_data


def apply_advanced_market_structure_filter(df: pd.DataFrame, symbol: str) -> Tuple[bool, Optional[str]]:
    """
    فلتر هيكل السوق الذكي المتطور - (شروط مخففة)
    """
    if len(df) < 50:
        return False, "Insufficient data for structure analysis"
    
    structure = detect_market_structure(df)
    
    # 1. تخفيف القبول: (السماح بـ ranging و weak_bearish)
    if structure["strength"] < 40: # تخفيف: 55 -> 40
        return False, f"Weak structure (strength: {structure['strength']})"
    
    # 2. الرفض فقط في حالة CHoCH هابط وواضح
    if structure.get("choch_detected") and structure.get("choch_direction") == "bearish" and structure["strength"] < 45: # تخفيف: 30->45
        return False, "Bearish CHoCH detected"
    
    # 3. تحليل السيولة (تخفيف)
    liquidity_analysis = analyze_liquidity_zones(df, tolerance=0.008) # توسيع النطاق: 0.005 -> 0.008
    if not liquidity_analysis["safe_to_trade"]:
        return False, liquidity_analysis["reason"]
    
    # 4. تحليل EMAs (تخفيف)
    last = df.iloc[-1]
    if not all(k in df.columns for k in ['ema9', 'ema21', 'ema50']):
        df = calculate_all_features(df) # إعادة الحساب إذا لم تكن موجودة
        last = df.iloc[-1]

    # يكفي أن يكون السعر فوق EMA50
    ema_alignment = (last['close'] > last['ema50'])
    if not ema_alignment:
        return False, "Price is below EMA50"
    
    # 5. Divergence (تعطيل مؤقت لزيادة الإشارات)
    # if detect_bearish_divergence(df):
    #     return False, "Bearish divergence detected"
    
    return True, None


def analyze_liquidity_zones(df: pd.DataFrame, tolerance: float = 0.008) -> Dict[str, any]: # تم تعديل التسامح
    """
    تحليل مناطق السيولة - (شروط مخففة)
    """
    recent_highs = df['high'].tail(20)
    recent_lows = df['low'].tail(20)
    current_price = df['close'].iloc[-1]
    
    high_clusters = find_price_clusters(recent_highs.values, tolerance=0.005) # تخفيف: 0.003 -> 0.005
    low_clusters = find_price_clusters(recent_lows.values, tolerance=0.005)
    
    danger_zone = False
    reason = ""
    
    for cluster in high_clusters:
        if abs(current_price - cluster) / current_price < tolerance: # استخدام التسامح المعدل
            danger_zone = True
            reason = f"Too close to liquidity zone at {cluster:.4f}"
            break
    
    return {
        "safe_to_trade": not danger_zone,
        "reason": reason if danger_zone else "Clear liquidity path",
        "high_clusters": high_clusters,
        "low_clusters": low_clusters
    }


def find_price_clusters(prices: np.ndarray, tolerance: float = 0.005) -> List[float]: # تم تعديل التسامح
    """
    إيجاد مناطق تجمع السعر (Price Clusters) - (شروط مخففة)
    """
    clusters = []
    sorted_prices = np.sort(prices)
    
    i = 0
    while i < len(sorted_prices):
        cluster = [sorted_prices[i]]
        j = i + 1
        
        while j < len(sorted_prices):
            if abs(sorted_prices[j] - sorted_prices[i]) / sorted_prices[i] <= tolerance:
                cluster.append(sorted_prices[j])
                j += 1
            else:
                break
        
        if len(cluster) >= 2:
            clusters.append(np.mean(cluster))
        
        i = j if j > i else i + 1
    
    return clusters


def detect_bearish_divergence(df: pd.DataFrame) -> bool:
    """
    اكتشاف Divergence هابط بين السعر والـ RSI
    """
    if len(df) < 20: return False
    
    if 'rsi' not in df.columns:
        df = calculate_all_features(df)

    recent_df = df.tail(20)
    
    highs = recent_df['high'].values
    rsi_values = recent_df['rsi'].values
    
    high_indices = argrelextrema(highs, np.greater, order=3)[0]
    
    if len(high_indices) >= 2:
        last_high_idx = high_indices[-1]
        prev_high_idx = high_indices[-2]
        
        price_higher_high = highs[last_high_idx] > highs[prev_high_idx]
        rsi_lower_high = rsi_values[last_high_idx] < rsi_values[prev_high_idx]
        
        if price_higher_high and rsi_lower_high:
            return True
    
    return False

# ===== SMART DYNAMIC FILTERS (Relaxed) =====

def apply_smart_liquidity_filter(df: pd.DataFrame, symbol: str) -> bool:
    """
    فلتر السيولة الذكي - (شروط مخففة)
    """
    if len(df) < 20: return False
    
    last = df.iloc[-1]
    
    # 1. نسبة Spread (High - Low) (تخفيف)
    spread_ratio = (last['high'] - last['low']) / last['close']
    if spread_ratio > 0.10:  # تخفيف: 0.08 -> 0.10
        log_rejection(symbol, "Smart Liquidity Filter Failed", {"reason": f"Spread > 10% ({spread_ratio:.2f}%)"})
        return False
    
    # 2. تذبذب حجم التداول (تخفيف)
    volume_std = df['volume'].tail(20).std()
    volume_mean = df['volume'].tail(20).mean()
    cv = volume_std / volume_mean if volume_mean > 0 else 999
    
    if cv > 3.5:  # تخفيف: 3.0 -> 3.5
        is_bullish_volume_spike = (last['close'] > df.iloc[-2]['close']) and (last['volume'] > volume_mean * 1.5)
        if not is_bullish_volume_spike:
            log_rejection(symbol, "Smart Liquidity Filter Failed", {"reason": f"Volume CV > 3.5 ({cv:.2f})"})
            return False
        else:
            logger.info(f"[{symbol}] Liquidity filter bypassed due to bullish volume spike.")

    # 3. قفزات السعر (تخفيف)
    price_changes = df['close'].pct_change().tail(10).abs()
    max_price_jump = price_changes.max()
    if max_price_jump > 0.10:  # تخفيف: 0.08 -> 0.10
        log_rejection(symbol, "Smart Liquidity Filter Failed", {"reason": f"Price Jump > 10% ({max_price_jump:.2f}%)"})
        return False
    
    return True


def apply_smart_risk_reward_filter(entry_price: float, stop_loss: float, target1: float, target2: float) -> bool:
    """ فلتر نسبة المخاطرة/العائد الذكي (شروط مخففة) """
    risk = entry_price - stop_loss
    if risk <= 0: return False
    
    reward1 = target1 - entry_price
    reward2 = target2 - entry_price
    
    if risk == 0: return False # تجنب القسمة على صفر
    
    rr1 = reward1 / risk
    rr2 = reward2 / risk
    
    # تخفيف: من (1.2, 2.0) إلى (1.1, 1.8)
    if rr1 < 1.1 or rr2 < 1.8: return False
    if (risk / entry_price) > 0.04: return False  # تخفيف: 0.035 -> 0.04 (4% risk)
    
    return True


def calculate_market_regime(df: pd.DataFrame) -> str:
    """ تحديد نظام السوق الحالي (trending, ranging, volatile) """
    if len(df) < 50: return 'unknown'
    last = df.iloc[-1]
    
    adx = last.get('adx', 0)
    bb_width = last.get('bb_width', 0)
    bb_width_ma = df['bb_width'].rolling(50).mean().iloc[-1]
    atr_percent = last.get('atr_percent', 0)
    
    if adx > 22 and atr_percent < 3.0: return 'trending' # تخفيف: 25->22, 2.5->3.0
    elif adx < 22 and bb_width < bb_width_ma * 0.9: return 'ranging' # تخفيف: 20->22, 0.8->0.9
    elif atr_percent > 3.5: return 'volatile' # تخفيف: 3.0->3.5
    else: return 'mixed'

# ===== STRATEGY SELECTOR (Updated) =====

ENHANCED_STRATEGIES = {
    "Smart_Momentum_Strategy": {
        "name": "زخم ذكي (مخفف)",
        "check_function": check_smart_momentum_strategy,
        "enabled": True, "best_regime": ['trending', 'mixed'], "risk_level": 'medium'
    },
    "Advanced_Pullback_Strategy": {
        "name": "ارتداد متقدم (مخفف)",
        "check_function": check_advanced_pullback_strategy,
        "enabled": True, "best_regime": ['trending', 'mixed'], "risk_level": 'low'
    },
    "Breakout_Retest_Strategy": {
        "name": "اختراق وإعادة اختبار (مخفف)",
        "check_function": check_breakout_retest_strategy,
        "enabled": True, "best_regime": ['trending', 'mixed'], "risk_level": 'medium'
    },
    # --- (تم إيقاف الاستراتيجيات المعقدة الأخرى مؤقتاً) ---
    # "Volume_Divergence_Strategy": {
    #     "name": "تباين الحجم",
    #     "check_function": check_volume_price_divergence_strategy,
    #     "enabled": False, ...
    # },
    
    # ===== NEW: إضافة الاستراتيجيات الجديدة =====
    "Simple_EMA_Cross_Strategy": {
        "name": "تقاطع EMA بسيط (جديد)",
        "check_function": check_simple_ema_cross_strategy,
        "enabled": True, "best_regime": ['trending', 'mixed'], "risk_level": 'medium'
    },
    "Simple_Bollinger_Bounce_Strategy": {
        "name": "ارتداد بولينجر بسيط (جديد)",
        "check_function": check_simple_bollinger_bounce_strategy,
        "enabled": True, "best_regime": ['ranging', 'mixed'], "risk_level": 'low'
    },
}

STRATEGY_NAMES = {key: info['name'] for key, info in ENHANCED_STRATEGIES.items()}


def find_best_strategy(df: pd.DataFrame, mtf_trend: Dict, symbol: str) -> Optional[Tuple[str, str]]:
    """
    يبحث عن أفضل استراتيجية مناسبة للوضع الحالي (محدث)
    """
    market_regime = calculate_market_regime(df)
    
    # --- NEW: جلب اتجاه السوق العام ---
    with market_state_lock:
        market_consensus = current_market_state.get("market_consensus", "sideways")
        market_score = current_market_state.get("market_score", 0.5)

    # --- NEW: فلتر اتجاه السوق العام ---
    # إذا كان السوق هابطاً جداً (أقل من 35% صاعد)، نرفض الدخول
    if market_score < 0.35:
        log_rejection(symbol, "Market Consensus Filter Failed", {"score": f"{market_score*100:.0f}%", "consensus": market_consensus})
        return None

    # استخدام الفلتر المتقدم (المخفف)
    passed, reason = apply_advanced_market_structure_filter(df, symbol)
    if not passed:
        log_rejection(symbol, "Advanced Market Structure Filter Failed", {"reason": reason})
        return None

    with strategy_filters_lock:
        strategies_to_check = {k: v for k, v in ENHANCED_STRATEGIES.items()}

    for strategy_key, strategy_info in strategies_to_check.items():
        if not strategy_info['enabled']: continue
        
        try:
            # --- NEW: تمرير اتجاه السوق العام إلى الاستراتيجيات ---
            if strategy_info['check_function'](df, mtf_trend, market_consensus):
                return (strategy_key, strategy_info['name'])
        except Exception as e:
            logger.error(f"❌ [{symbol}] Error checking strategy {strategy_key}: {e}", exc_info=True)
            continue
    
    return None
    # ===== IMPROVED STOP LOSS & TAKE PROFIT (Relaxed) =====

def calculate_smart_stop_loss(df: pd.DataFrame, entry_price: float, strategy_name: str) -> float:
    """
    حساب وقف خسارة ذكي - (شروط مخففة)
    """
    last = df.iloc[-1]
    atr_value = last.get('atr', 0)
    if atr_value == 0: # معالجة حالة ATR=0
        atr_value = (last['high'] - last['low']) or (entry_price * 0.01) # استخدام الشمعة الحالية أو 1%
        
    multipliers = {
        'Simple_Bollinger_Bounce_Strategy': 1.6, # وقف قريب
        'Simple_EMA_Cross_Strategy': 2.2,
    }
    atr_multiplier = multipliers.get(strategy_name, 2.0) # الافتراضي 2.0

    atr_stop = entry_price - (atr_value * atr_multiplier)
    recent_low = df['low'].tail(10).min() # توسيع: 7 -> 10
    structure_stop = recent_low * 0.995 # تخفيف: 0.997 -> 0.995
    ema50_stop = last['ema50'] * 0.995 # استخدام EMA50 كدعم أخير
    
    stop_loss = min(atr_stop, structure_stop, ema50_stop) # استخدام min لإعطاء أوسع وقف
    
    # تخفيف الحد الأقصى للوقف
    max_stop_distance = entry_price * 0.04 # زيادة: 0.028 -> 0.04
    if entry_price - stop_loss > max_stop_distance:
        stop_loss = entry_price - max_stop_distance
    
    # تخفيف الحد الأدنى للوقف
    min_stop_distance = entry_price * 0.005 # تقليل: 0.006 -> 0.005
    if entry_price - stop_loss < min_stop_distance:
        stop_loss = entry_price - min_stop_distance
    
    # التأكد من أن الوقف ليس أعلى من الدخول
    if stop_loss >= entry_price:
        stop_loss = entry_price - (entry_price * 0.01) # وقف افتراضي 1%

    return stop_loss


def calculate_smart_take_profit(
    df: pd.DataFrame, 
    entry_price: float, 
    stop_loss: float, 
    strategy_name: str
) -> Tuple[float, float]:
    """
    حساب أهداف ربح ذكية - (نسب مخففة)
    """
    risk_amount = entry_price - stop_loss
    if risk_amount <= 0: # معالجة حالة الوقف غير الصحيح
        risk_amount = entry_price * 0.015 # افتراض مخاطرة 1.5%
        
    rr_ratios = {
        'Simple_Bollinger_Bounce_Strategy': (1.2, 2.0), # أهداف قريبة
        'Simple_EMA_Cross_Strategy': (1.5, 2.5),
    }
    
    # تخفيف النسب الافتراضية
    rr1, rr2 = rr_ratios.get(strategy_name, (1.3, 2.2)) # تخفيف: (1.8, 3.0) -> (1.3, 2.2)
    
    target1 = entry_price + (risk_amount * rr1)
    target2 = entry_price + (risk_amount * rr2)
    
    last = df.iloc[-1]
    atr_value = last.get('atr', 0)
    if atr_value == 0: # معالجة حالة ATR=0
        atr_value = (last['high'] - last['low']) or (entry_price * 0.01)
    
    # تعديل الحدود القصوى (تخفيف)
    max_target1 = entry_price + (atr_value * 5.0) # زيادة: 4.0 -> 5.0
    if target1 > max_target1: target1 = max_target1
    
    max_target2 = entry_price + (atr_value * 8.0) # زيادة: 7.0 -> 8.0
    if target2 > max_target2: target2 = max_target2
    
    # التأكد من أن الأهداف أعلى من الدخول
    if target1 <= entry_price:
        target1 = entry_price * 1.01
    if target2 <= target1:
        target2 = target1 * 1.01

    return (target1, target2)
    # ===== END OF NEW STRATEGY BLOCK =====

# ( ... (trading_bot_v36_part2.py) ... )

def get_formatted_quantity(symbol: str, quantity: Decimal) -> str:
    """
    Formats the quantity to the correct precision required by Binance API for a specific symbol.
    """
    try:
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info:
            logger.warning(f"[{symbol}] No exchange info for formatting. Using default format.")
            return f"{quantity.normalize()}"

        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if not lot_size_filter:
            logger.warning(f"[{symbol}] LOT_SIZE filter not found. Using default format.")
            return f"{quantity.normalize()}"
        
        step_size = Decimal(lot_size_filter['stepSize'])
        
        # Quantize the number to the step size (e.g., 0.01 for 2 decimal places)
        # This correctly formats the number by rounding down to the nearest valid trade amount.
        formatted_quantity = quantity.quantize(step_size, rounding=ROUND_DOWN)

        # Return as a plain string without scientific notation or extra trailing zeros.
        return f"{formatted_quantity.normalize()}"
        
    except Exception as e:
        logger.error(f"❌ [{symbol}] Error formatting quantity: {e}. Returning raw value string.")
        return str(quantity)

# ===== FIXED: Position Size Calculation =====
def adjust_quantity_to_lot_size(symbol: str, quantity: float, logger=logger) -> Optional[Decimal]:
    """
    ضبط الكمية حسب LOT_SIZE مع معالجة أفضل للأخطاء.
    """
    from decimal import Decimal, ROUND_DOWN
    
    try:
        # الوصول المباشر إلى المتغير العام لتجنب البيانات القديمة
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info:
            logger.error(f"[{symbol}] معلومات الرمز غير موجودة في exchange_info_map")
            return None
            
        lot_size_filter = next((f for f in symbol_info['filters'] 
                               if f['filterType'] == 'LOT_SIZE'), None)
        
        if not lot_size_filter:
            logger.warning(f"[{symbol}] LOT_SIZE filter غير موجود، استخدام الكمية الخام")
            return Decimal(str(quantity))
        
        step_size = Decimal(lot_size_filter['stepSize'])
        min_qty = Decimal(lot_size_filter['minQty'])
        max_qty = Decimal(lot_size_filter.get('maxQty', '9000000000'))
        
        quantity_dec = Decimal(str(quantity))
        
        # التحقق من min_qty
        if quantity_dec < min_qty:
            logger.warning(f"[{symbol}] الكمية {quantity_dec} أقل من minQty {min_qty}")
            return None
        
        # التحقق من max_qty
        if quantity_dec > max_qty:
            logger.warning(f"[{symbol}] الكمية {quantity_dec} أكبر من maxQty {max_qty}")
            quantity_dec = max_qty
        
        # ضبط الكمية حسب step_size
        adjusted_quantity = (quantity_dec // step_size) * step_size
        
        # التحقق النهائي
        if adjusted_quantity < min_qty:
            logger.warning(f"[{symbol}] الكمية المعدلة {adjusted_quantity} أقل من minQty {min_qty}")
            return None
        
        return adjusted_quantity
        
    except Exception as e:
        logger.error(f"❌ [{symbol}] خطأ في ضبط LOT_SIZE: {e}", exc_info=True)
        return None

def calculate_position_size_fixed(symbol: str, entry_price: float, 
                                  available_balance: float, is_real: bool,
                                  logger=logger,
                                  override_amount: Optional[float] = None) -> Optional[Decimal]:
    """
    حساب حجم الصفقة مع معالجة صحيحة لجميع الحالات (النسخة الأساسية الآمنة).
    """
    desired_usdt_amount = 0.0
    if override_amount is not None:
        desired_usdt_amount = override_amount
    elif not is_real:
        desired_usdt_amount = PAPER_TRADE_FIXED_AMOUNT_USDT
    else:
        with trade_amount_lock:
            desired_usdt_amount = random.uniform(FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT)

    try:
        dec_entry = Decimal(str(entry_price))
        if dec_entry <= 0:
            logger.error(f"[{symbol}] سعر الدخول غير صحيح: {entry_price}")
            return None
        
        dec_balance = Decimal(str(available_balance))
        dec_desired_amount = Decimal(str(desired_usdt_amount))
        
        logger.info(f"[{symbol}] حساب الكمية: المبلغ المطلوب ${dec_desired_amount:.2f}, الرصيد المتاح ${dec_balance:.2f}")

        if is_real and dec_desired_amount > dec_balance:
            logger.warning(f"[{symbol}] الرصيد غير كافٍ: مطلوب ${dec_desired_amount:.2f}, متاح ${dec_balance:.2f}")
            return None

        initial_quantity = dec_desired_amount / dec_entry
        adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(initial_quantity), logger=logger)

        if adjusted_quantity is None or adjusted_quantity <= 0:
            logger.warning(f"[{symbol}] فشل ضبط الكمية حسب LOT_SIZE")
            return None

        notional_value = adjusted_quantity * dec_entry
        
        # الوصول المباشر إلى المتغير العام
        symbol_info = exchange_info_map.get(symbol)
        if symbol_info:
            min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] in ('MIN_NOTIONAL', 'NOTIONAL')), None)
            
            if min_notional_filter:
                min_notional_str = min_notional_filter.get('minNotional', min_notional_filter.get('notional', '5.0'))
                min_notional = Decimal(min_notional_str)
                
                if notional_value < min_notional:
                    logger.warning(f"[{symbol}] القيمة الاسمية ${notional_value:.2f} أقل من min_notional ${min_notional}")
                    required_notional = min_notional * Decimal('1.01')
                    
                    if is_real and required_notional > dec_balance:
                        logger.error(f"[{symbol}] لا يمكن تلبية min_notional: مطلوب ${required_notional:.2f}, متاح ${dec_balance:.2f}")
                        return None
                        
                    new_quantity = required_notional / dec_entry
                    adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(new_quantity), logger=logger)

                    if adjusted_quantity is None or adjusted_quantity <= 0:
                        logger.error(f"[{symbol}] فشل ضبط الكمية لتلبية min_notional")
                        return None

                    notional_value = adjusted_quantity * dec_entry
                    logger.info(f"[{symbol}] تم تعديل الكمية لتلبية min_notional: كمية={adjusted_quantity}, قيمة=${notional_value:.2f}")

        if notional_value <= 0:
            logger.error(f"[{symbol}] القيمة الاسمية النهائية صفر أو سالبة!")
            return None

        if is_real and notional_value > dec_balance:
            logger.error(f"[{symbol}] القيمة النهائية ${notional_value:.2f} تتجاوز الرصيد ${dec_balance:.2f}")
            return None
            
        logger.info(f"[{symbol}] ✅ الكمية النهائية الصحيحة: {adjusted_quantity} (قيمة اسمية: ${notional_value:.2f})")
        return adjusted_quantity

    except Exception as e:
        logger.error(f"❌ [{symbol}] خطأ حرج في حساب حجم الصفقة: {e}", exc_info=True)
        return None

# ===== UPDATED: Position Size Calculation (Now Dynamic) =====
def calculate_dynamic_position_size(
    symbol: str, 
    entry_price: float, 
    available_balance: float, 
    is_real: bool,
    quality_score: int,
    atr_percent: float,
    logger
) -> Optional[Decimal]:
    """
    حساب حجم الصفقة بشكل ديناميكي بناءً على جودة الإشارة وتقلب السوق. (شروط مخففة)
    """
    base_usdt_amount = 0.0
    # 1. تحديد مبلغ الأساس بناءً على وضع التداول
    if not is_real:
        base_usdt_amount = PAPER_TRADE_FIXED_AMOUNT_USDT
    else:
        with trade_amount_lock:
            base_usdt_amount = random.uniform(FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT)

    # 2. تعديل المبلغ بناءً على جودة الإشارة (تخفيف)
    quality_modifier = 1.0
    if quality_score > 80: # تخفيف: 85 -> 80
        quality_modifier = 1.10  # تخفيف: 1.15 -> 1.10
    elif quality_score < 65: # تخفيف: 70 -> 65
        quality_modifier = 0.95   # تخفيف: 0.90 -> 0.95

    # 3. تعديل المبلغ بناءً على تقلب السوق (تخفيف)
    volatility_modifier = 1.0
    if atr_percent > 3.5: # تخفيف: 3.0 -> 3.5
        volatility_modifier = 0.80  # تخفيف: 0.75 -> 0.80
    elif atr_percent < 0.7: # تخفيف: 0.8 -> 0.7
        volatility_modifier = 1.05  # تخفيف: 1.10 -> 1.05

    # 4. حساب المبلغ النهائي المطلوب
    desired_usdt_amount = base_usdt_amount * quality_modifier * volatility_modifier
    
    logger.info(
        f"[{symbol}] Dynamic Size: Base=${base_usdt_amount:.2f}, "
        f"QualityMod={quality_modifier:.2f}, VolatilityMod={volatility_modifier:.2f} -> "
        f"Final Desired=${desired_usdt_amount:.2f}"
    )
    
    # 5. استخدام دالة حساب الحجم الآمنة مع المبلغ الديناميكي الجديد
    return calculate_position_size_fixed(
        symbol, entry_price, available_balance, is_real, 
        logger, override_amount=desired_usdt_amount
    )

def create_trade_signal(symbol: str, df: pd.DataFrame, strategy_key: str, mtf_trend: Dict):
    df.strategy = strategy_key 
    
    # General filters (Relaxed)
    if not check_market_volatility_filter_enhanced(df, symbol): return
    if not add_news_filter(): log_rejection(symbol, "News Filter Failed"); return
    if not add_liquidity_filter(): log_rejection(symbol, "Liquidity Filter Failed"); return
    if not add_correlation_filter(symbol): log_rejection(symbol, "Correlation Filter Failed"); return

    quality_score = calculate_signal_quality_score(df, mtf_trend)
    with min_quality_lock: min_score = MIN_SIGNAL_QUALITY
    if quality_score < min_score:
        log_rejection(symbol, "Low Quality Signal", {"score": quality_score, "min_required": min_score})
        return
    logger.info(f"⭐ [Signal Quality] {symbol} ({strategy_key}): {quality_score}/100")

    entry_price = df.iloc[-1]['close']
    stop_loss_price = calculate_smart_stop_loss(df, entry_price, strategy_key)
    target_price_1, target_price_2 = calculate_smart_take_profit(df, entry_price, stop_loss_price, strategy_key)
    
    # Apply Risk/Reward filter (Relaxed)
    if not apply_smart_risk_reward_filter(entry_price, stop_loss_price, target_price_1, target_price_2):
        log_rejection(symbol, "Smart Risk/Reward Filter Failed")
        return

    if stop_loss_price >= entry_price:
        log_rejection(symbol, "Invalid Position Size", {"entry": entry_price, "sl": stop_loss_price})
        return

    with trading_mode_lock: is_real = not paper_trading_mode
    
    atr_percent = df.iloc[-1].get('atr_percent', 0)
    signal_details = {
        "atr": df.iloc[-1].get('atr', 0), "trailing_stop_activated": False, "tp1_done": False,
        "quality_score": quality_score, "atr_percent": atr_percent,
        "early_exit_threshold": entry_price * 0.995
    }
    
    trade_levels = {
        "entry_price": entry_price, "stop_loss": stop_loss_price,
        "target_price_1": target_price_1, "target_price_2": target_price_2
    }

    current_real_balance = 0
    with balance_lock:
        current_real_balance = usdt_balance

    quantity_dec = calculate_dynamic_position_size(
        symbol, entry_price, current_real_balance, is_real, quality_score, atr_percent, logger
    )

    if quantity_dec is None or quantity_dec <= 0:
        logger.error(f"❌ [{symbol}] Position size calculation failed. Trade rejected.")
        return
    
    notional_value = float(quantity_dec) * entry_price

    if is_real:
        try:
            formatted_quantity = get_formatted_quantity(symbol, quantity_dec)
            logger.info(f"💰 [Real Trade] Placing LIVE MARKET BUY order for {formatted_quantity} of {symbol}")
            order = client.create_order(
                symbol=symbol, 
                side=Client.SIDE_BUY, 
                type=Client.ORDER_TYPE_MARKET, 
                quantity=formatted_quantity
            )
            avg_fill_price = sum(Decimal(f['price']) * Decimal(f['qty']) for f in order.get('fills', [])) / max(sum(Decimal(f['qty']) for f in order.get('fills', [])), Decimal('1e-8')) if order.get('fills') else Decimal(str(entry_price))
            final_quantity = Decimal(order.get('executedQty', str(quantity_dec)))
            order_id = order.get('orderId', 'N/A')
            save_signal_to_db(
                symbol, float(avg_fill_price), trade_levels,
                strategy_key, True, float(final_quantity),
                {**signal_details, "avg_fill": float(avg_fill_price)}, order_id
            )
            send_trade_open_notification(
                symbol, strategy_key, float(avg_fill_price),
                stop_loss_price, target_price_1, target_price_2, float(final_quantity),
                is_real, quality_score, atr_percent, notional_value
            )
        except BinanceAPIException as e:
            logger.error(f"❌ [Real Trade] Binance API Error for {symbol}: {e}")
            send_enhanced_telegram_message(f"❌ *خطأ في صفقة حقيقية لـ {symbol}*\n`{e}`", force=True)
        except Exception as e:
            logger.error(f"❌ [Real Trade] CRITICAL ERROR creating real trade for {symbol}: {e}", exc_info=True)
    else: # Paper Trading
        save_signal_to_db(symbol, entry_price, trade_levels, strategy_key, False, float(quantity_dec), signal_details)
        send_trade_open_notification(
            symbol, strategy_key, entry_price, stop_loss_price, target_price_1, target_price_2,
            float(quantity_dec), is_real, quality_score, atr_percent, notional_value
        )

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
        # ===== HTML Templates (Updated) =====
DASHBOARD_TEMPLATE = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>لوحة التحكم - بوت 5 دقائق (V36.0)</title>
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
.trend{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:12px}
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
</style>
</head>
<body>
<div class="container">
  <header><h1>لوحة التحكم • بوت 5 دقائق V36.0 (مخفف)</h1><div class="badge" id="serverTime">—</div></header>
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
            <div id="signals" class="signals-grid"><div class="loading-spinner"></div></div>
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
        </div>
      </div>
      <div class="card">
        <h2>حالة السوق (الجديد)</h2>
        <div class="card-body">
            <div class="metric-card" style="margin-bottom: 12px;">
                <div class="metric-title">إجماع السوق العام</div>
                <div class="metric-value" id="marketConsensus" style="font-size: 20px;">—</div>
                <div class="metric-title" style="margin-top: 5px;">قوة الاتجاه الصاعد</div>
                <div class="metric-value" id="marketScore" style="font-size: 16px;">—</div>
            </div>
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
let lastPrices = {};
let performanceChartInstance = null;
let openSignals = {};
const STRATEGY_NAMES = {{ STRATEGY_NAMES|tojson }};

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
    // --- FIX: إزالة alert() واستخدام console.log بدلاً من ذلك ---
    console.log(`[${type.toUpperCase()}] ${message}`);
    // يمكنك إضافة نظام إشعارات أفضل هنا لاحقاً
}

function closeTrade(signalId) {
    // --- FIX: Use a custom modal instead of confirm() ---
    // This is a simple placeholder. For a real app, you'd create a DOM element.
    // For this environment, we'll proceed assuming user confirmation
    // as confirm() is blocked.
    // تم إزالة النافذة المنبثقة `confirm()`
    console.warn("Triggering manual close for signal " + signalId + ". (Skipping confirm() dialog).");
    
    fetch(`/api/close_trade/${signalId}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({})
    })
    .then(res => res.ok ? res.json() : res.json().then(err => { throw new Error(err.message || 'Server error') }))
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
    const strategyName = STRATEGY_NAMES[signal.strategy_name] || signal.strategy_name.replace(/_/g, " ").replace("Strategy", "");
    return `
        <div class="signal" id="signal-${signal.id}" data-symbol="${signal.symbol}">
            <div>
                <div class="sig-title">${signal.symbol}</div>
                <div class="sig-meta">${strategyName} | <span style="color: ${qualityColor}; font-weight: bold;">⭐ ${qualityScore}/100</span></div>
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
            if (!signalId) return; // --- FIX: Add guard clause
            const signalData = openSignals[signalId];
            if (signalData) {
                const entry = signalData.entry_price, tp1 = signalData.target_price_1, sl = signalData.stop_loss;
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
                const progressBar = el.querySelector('.progress-bar'), progressContainer = el.querySelector('.progress');
                if(progressBar) { progressBar.style.width = `${progress}%`; progressBar.style.background = color; }
                if(progressContainer) { progressContainer.title = title; }
            }
        });
        lastPrices[symbol] = price;
    }
}

function addNotification(notification, prepend = true) {
    const tbody = qs('#events tbody');
    const row = `<tr><td>${new Date(notification.timestamp).toLocaleTimeString('ar-EG')}</td><td>${notification.type||''}</td><td>${notification.message||''}</td></tr>`;
    if (prepend) {
        tbody.insertAdjacentHTML('afterbegin', row);
        if (tbody.rows.length > 20) tbody.deleteRow(-1);
    } else {
        tbody.insertAdjacentHTML('beforeend', row);
    }
}

function addRejection(rejection, prepend = true) {
    const tbody = qs('#rejections tbody');
    const row = `<tr><td>${new Date(rejection.timestamp).toLocaleTimeString('ar-EG')}</td><td>${rejection.symbol||''}</td><td>${rejection.reason||''}</td></tr>`;
    if (prepend) {
        tbody.insertAdjacentHTML('afterbegin', row);
        if (tbody.rows.length > 30) tbody.deleteRow(-1);
    } else {
        tbody.insertAdjacentHTML('beforeend', row);
    }
}

function updateMarketTrends(marketState) {
    const trendsContainer = document.getElementById('marketTrends');
    trendsContainer.innerHTML = '';
    
    // --- NEW: Update Market Consensus ---
    const consensus = marketState.market_consensus || 'sideways';
    const score = (marketState.market_score || 0.5) * 100;
    const consensusEl = document.getElementById('marketConsensus');
    const scoreEl = document.getElementById('marketScore');
    
    let consensusClass = 'amber';
    if (consensus === 'bullish') consensusClass = 'green';
    else if (consensus === 'bearish') consensusClass = 'red';
    
    if (consensusEl) {
        consensusEl.textContent = consensus.charAt(0).toUpperCase() + consensus.slice(1);
        consensusEl.className = `metric-value ${consensusClass}`;
    }
    if (scoreEl) {
        scoreEl.textContent = `${score.toFixed(1)}%`;
        scoreEl.className = `metric-value ${consensusClass}`;
    }
    
    // --- NEW: Update Leader Trends ---
    const leaders = marketState.leader_trends || {};
    if (Object.keys(leaders).length > 0) {
        for (const [symbol, details] of Object.entries(leaders)) {
            let trendClass = 'amber', trendText = 'جانبي';
            if (details.trend === 'bullish') { trendClass = 'green'; trendText = 'صاعد'; }
            else if (details.trend === 'bearish') { trendClass = 'red'; trendText = 'هابط'; }
            
            const symbolShort = symbol.replace('USDT', '');
            trendsContainer.innerHTML += `
                <div class="pill">
                    <b>${symbolShort}</b>
                    <span class="${trendClass}">${trendText}</span>
                    <small>${(details.score * 100).toFixed(0)}%</small>
                </div>`;
        }
    } else {
        trendsContainer.innerHTML = '<div class="loading-spinner"></div>';
    }
}

async function initializeDashboard() {
    try {
        showLoadingIndicator('#signals');
        
        // ===== START OF FIX =====
        // --- FIX: Use Promise.allSettled to prevent one failed request from breaking the entire dashboard ---
        // This was the cause of the bug in the screenshot
        // هذا هو الإصلاح الرئيسي للمشكلة التي ظهرت في لقطة الشاشة
        const results = await Promise.allSettled([
            fetch('/api/dashboard_data'),
            fetch('/api/open_signals'),
            fetch('/api/performance_metrics')
        ]);

        const baseDataRes = results[0];
        const signalsRes = results[1];
        const metricsRes = results[2];

        // 1. Process Base Data (Most Important)
        if (baseDataRes.status === 'fulfilled' && baseDataRes.value.ok) {
            const baseData = await baseDataRes.value.json();
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
            
            qs('#rejections tbody').innerHTML = '';
            baseData.rejections.forEach(r => addRejection(r, false));
            qs('#events tbody').innerHTML = '';
            baseData.notifications.forEach(n => addNotification(n, false));
        } else {
            console.error("Failed to load base dashboard data:", baseDataRes.reason || (baseDataRes.value && baseDataRes.value.statusText));
            // Show a partial error, but don't stop
            showNotification('فشل تحميل البيانات الأساسية للوحة التحكم', 'error');
        }

        // 2. Process Open Signals
        if (signalsRes.status === 'fulfilled' && signalsRes.value.ok) {
            const signalsData = await signalsRes.value.json();
            openSignals = signalsData.signals.reduce((acc, s) => { acc[s.id] = s; return acc; }, {});
            renderAllSignals(signalsData.signals);
            qs('#openCount').textContent = signalsData.signals.length;
            qs('#signalCount').textContent = `(${signalsData.signals.length})`;
        } else {
            console.error("Failed to load open signals:", signalsRes.reason || (signalsRes.value && signalsRes.value.statusText));
            qs('#signals').innerHTML = '<p style="text-align:center;color:var(--muted);">فشل تحميل الصفقات المفتوحة.</p>';
            qs('#openCount').textContent = 'Error';
            qs('#signalCount').textContent = '(Error)';
        }

        // 3. Process Performance Metrics
        if (metricsRes.status === 'fulfilled' && metricsRes.value.ok) {
            const metricsData = await metricsRes.value.json();
            qs('#winRate').textContent = `${metricsData.win_rate.toFixed(2)}%`;
            qs('#avgProfit').textContent = `${metricsData.avg_profit.toFixed(2)}%`;
            qs('#totalTrades').textContent = metricsData.total_trades;
        } else {
            console.error("Failed to load performance metrics:", metricsRes.reason || (metricsRes.value && metricsRes.value.statusText));
            qs('#winRate').textContent = 'N/A';
            qs('#avgProfit').textContent = 'N/A';
            qs('#totalTrades').textContent = 'N/A';
        }
        // ===== END OF FIX =====

        // 4. Load secondary data (which was already separate, this is good)
        loadAdditionalData();

    } catch (error) {
        // This catch is for network-level errors before Promise.allSettled
        console.error("فشل تحميل البيانات الأساسية:", error);
        qs('#signals').innerHTML = '<p>فشل تحميل البيانات. حاول تحديث الصفحة.</p>';
    }
}
async function loadAdditionalData() {
    try {
        const perfRes = await fetch('/api/advanced_performance_data');
        if (perfRes.ok) {
            const advancedData = await perfRes.json();
            qs('#maxDrawdown').textContent = `${advancedData.maxDrawdown.toFixed(2)}%`;
            updateAdvancedPerformance(advancedData);
        } else {
            qs('#maxDrawdown').textContent = 'N/A';
        }
    } catch (error) { console.error("Error loading additional data:", error); }
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
            case 'new_signal': openSignals[data.payload.id] = data.payload; updateSingleSignal(data.payload); break;
            case 'signal_update': openSignals[data.payload.id] = data.payload; updateSingleSignal(data.payload); break;
            case 'trade_closed': const el = qs(`#signal-${data.payload.signal_id}`); if (el) el.remove(); delete openSignals[data.payload.signal_id]; break;
            case 'new_notification': addNotification(data.payload); break;
            case 'new_rejection': addRejection(data.payload); break;
            case 'market_state_update': updateMarketTrends(data.payload); break;
            case 'trading_mode': const isPaper = data.payload.paper_trading; qs('#tradingModeToggle').checked = !isPaper; qs('#tradingModeText').textContent = isPaper ? 'ورقي' : 'حقيقي'; break;
            case 'quality_filter': qs('#qualityFilter').value = data.payload.min_quality; qs('#qualityValue').textContent = data.payload.min_quality; break;
            case 'trade_amount_update': qs('#tradeAmountDisplay').textContent = `$${data.payload.min} - $${data.payload.max}`; break;
        }
    };
    socket.onclose = () => { console.log("WebSocket connection closed, reconnecting..."); setTimeout(setupWebSocket, 3000); };
    socket.onerror = (error) => console.error("WebSocket error:", error);
}

function setupSorting() {
    const sortButtons = document.querySelectorAll('[data-sort]');
    const debouncedSort = debounce((sortBy) => {
        showLoadingIndicator('#signals');
        fetch(`/api/open_signals?sort=${sortBy}`)
            .then(res => res.json()).then(data => {
                openSignals = data.signals.reduce((acc, s) => { acc[s.id] = s; return acc; }, {});
                renderAllSignals(data.signals);
            }).catch(err => console.error("Sort failed:", err));
    }, 300);
    sortButtons.forEach(button => { button.addEventListener('click', () => debouncedSort(button.dataset.sort)); });
}

async function toggleTrading() { await fetch('/toggle_trading', {method:'POST'}); }
qs('#toggleTrading').addEventListener('change', toggleTrading);

qs('#tradingModeToggle').addEventListener('change', function() {
  const isPaper = !this.checked, modeText = isPaper ? 'ورقي' : 'حقيقي';
  // --- FIX: Do not use confirm() ---
  if (!isPaper) {
      // تم إزالة النافذة المنبثقة `confirm()`
      console.warn("Switching to REAL trading. (Skipping confirm() dialog).");
  }
  fetch('/api/settings', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({paper_trading_mode: isPaper}) })
  .then(res => res.json()).then(data => {
    if (data.success) { qs('#tradingModeText').textContent = modeText; showNotification(`تم التبديل إلى الوضع ${modeText}`, 'success'); }
    else { showNotification('فشل تغيير وضع التداول', 'error'); this.checked = !this.checked; }
  }).catch(error => { console.error('Error:', error); showNotification('خطأ في الاتصال بالخادم', 'error'); this.checked = !this.checked; });
});

const debouncedQualityUpdate = debounce((value) => {
    fetch('/api/signal_quality', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({min_quality: parseInt(value)}) })
    .catch(error => console.error('Error:', error));
}, 500);
qs('#qualityFilter').addEventListener('input', function() { qs('#qualityValue').textContent = this.value; debouncedQualityUpdate(this.value); });

function updateAdvancedPerformance(data) {
    if (!performanceChartInstance && data.equity_curve && data.equity_curve.labels.length > 0) { createPerformanceChart(data.equity_curve); }
    else if (performanceChartInstance) {
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
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { type: 'time', time: { unit: 'day' }, ticks: { color: 'var(--muted)', autoSkip: true, maxTicksLimit: 8 }, grid: { display: false } }, y: { ticks: { color: 'var(--muted)', callback: (v) => v.toFixed(0) }, grid: { color: 'rgba(255, 255, 255, 0.05)' } } }
    });
}

document.addEventListener('DOMContentLoaded', () => { initializeDashboard(); setupWebSocket(); setupSorting(); });
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
<title>الإعدادات - بوت 5 دقائق (V36.0)</title>
<style>
:root{--bg:#0b1020;--panel:#121b36;--accent:#3aa0ff;--ok:#15c46a;--warn:#ff9f1a;--bad:#ff4757;--muted:#8aa0c8;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:#e8f1ff;font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,"Noto Sans",Arial}
.container{max-width:900px;margin:0 auto;padding:16px;display:flex;flex-direction:column;gap:16px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between; margin-bottom: 16px;}
h1{font-size:22px;margin:0;font-weight:700;color:#d7e4ff}
.card{background:var(--panel);border:1px solid #1e2c52;border-radius:14px;box-shadow:0 8px 30px rgba(0,0,0,.25);overflow:hidden}
.card h2{margin:0;padding:12px 14px;border-bottom:1px solid #1e2c52;font-size:16px;color:#cfe2ff;}
.card-body{padding:16px}
.form-grid{display:grid;grid-template-columns:1fr;gap:24px;}
@media(min-width: 600px){.form-grid{grid-template-columns:1fr 1fr;}}
.form-group{display:flex;flex-direction:column;gap:8px}
.form-group label{font-weight:600;color:var(--muted);font-size:14px}
.form-group .note{font-size:12px; color: var(--muted); opacity: 0.8;}
.form-group input, .form-group select {
    background: #0b1126; border: 1px solid #233056; color: #e8f1ff; padding: 10px; border-radius: 8px; font-size: 14px;
}
.switch{display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;border:1px solid #2a3a68;background:#0f1b3b;cursor:pointer;user-select:none}
.switch input{display:none}
.switch .dot{width:14px;height:14px;border-radius:50%;background:#6a7fb2;transition:.2s}
.switch input:checked + .dot{background:#24d08a;transform:translateX(2px) scale(1.1)}
.btn{appearance:none;border:1px solid #2a3a68;background:#0f1b3b;color:#d9e7ff;padding:10px 14px;border-radius:10px;cursor:pointer;font-weight:700;transition: background-color 0.2s, transform 0.2s; text-decoration: none;}
.btn:hover{transform:translateY(-1px);border-color:#3a58a6}
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
                    <small class="note">لا يؤثر على الصفقات الورقية</small>
                </div>
                <div class="form-group">
                    <label for="tradeAmountMaxInput">أقصى قيمة للصفقة (الحقيقية)</label>
                    <input type="number" id="tradeAmountMaxInput" name="FIXED_TRADE_AMOUNT_MAX_USDT" value="{{ trade_amount_max }}" step="0.1" min="1.0" max="50.0">
                     <small class="note">لا يؤثر على الصفقات الورقية</small>
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

        <div class="card" style="margin-top: 16px;">
            <h2>إعدادات الاستراتيجيات (محدث)</h2>
            <div class="card-body">
                {% for key, name in STRATEGY_NAMES.items() %}
                <div class="form-group" style="flex-direction: row; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e2c52; padding-bottom: 12px; margin-bottom: 12px;">
                    <label>{{ name }}</label>
                    <label class="switch">
                        <input type="checkbox" name="{{ key }}" {% if strategies_status[key] %}checked{% endif %}>
                        <span class="dot"></span>
                    </label>
                </div>
                {% endfor %}
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
    const strategies = {};
    const strategyKeys = {{ STRATEGY_NAMES.keys()|list|tojson }};

    for (const [key, value] of formData.entries()) {
        if (strategyKeys.includes(key)) {
            strategies[key] = true;
        } else if (key === 'paper_trading_mode') {
            settings[key] = false;
        } else {
            settings[key] = value;
        }
    }
    
    document.querySelectorAll('input[type="checkbox"][name]').forEach(cb => {
        if (strategyKeys.includes(cb.name) && !cb.checked) {
            strategies[cb.name] = false;
        }
    });
    
    if (!formData.has('paper_trading_mode')) {
        settings['paper_trading_mode'] = true;
    }

    Promise.all([
        fetch('/api/settings', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(settings)
        }),
        fetch('/api/strategies', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(strategies)
        }),
        fetch('/api/signal_quality', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({min_quality: settings.min_quality})
        })
    ]).then(responses => {
        if (responses.every(res => res.ok)) {
            showNotification('تم حفظ الإعدادات بنجاح!');
        } else {
            showNotification('حدث خطأ أثناء حفظ الإعدادات.');
        }
    }).catch(err => {
        console.error(err);
        showNotification('فشل الاتصال بالخادم.');
    });
});

document.querySelector('input[name="paper_trading_mode"]').addEventListener('change', function() {
    document.getElementById('tradingModeText').textContent = this.checked ? 'حقيقي (Real)' : 'ورقي (Paper)';
});

function showNotification(message) {
    const notification = document.getElementById('notification');
    notification.textContent = message;
    notification.classList.add('show');
    setTimeout(() => {
        notification.classList.remove('show');
    }, 3000);
}
</script>
</body>
</html>
"""

# ===== NEW: Intelligent Trailing Stop-Loss & Re-analysis Logic (Relaxed) =====
def manage_intelligent_trailing_stop(signal: Dict, current_price: float, df: pd.DataFrame) -> Optional[Dict]:
    """
    إدارة وقف الخسارة المتحرك الذكي.
    (شروط مخففة)
    """
    details = signal.get('signal_details', {})
    if not isinstance(details, dict): details = {}

    if not details.get('tp1_done'): return None

    current_stop_loss = float(signal['stop_loss'])
    new_potential_sl = None

    try:
        # استخدام قاع أبعد (أكثر تساهلاً)
        recent_low = df['low'].iloc[-6:-2].min() # توسيع: -4:-1 -> -6:-2
        potential_sl_swing = recent_low * Decimal('0.997') # تخفيف: 0.998 -> 0.997
        if potential_sl_swing > current_stop_loss:
            new_potential_sl = float(potential_sl_swing)
    except Exception:
        atr_value = df.iloc[-1].get('atr', 0)
        if atr_value > 0:
            potential_sl_atr = current_price - (atr_value * 3.0) # توسيع: 2.5 -> 3.0
            if potential_sl_atr > current_stop_loss:
                new_potential_sl = potential_sl_atr

    if new_potential_sl and new_potential_sl > current_stop_loss:
        logger.info(f"[{signal['symbol']}] Trailing SL Update: From {current_stop_loss:.5f} to {new_potential_sl:.5f}")
        return {"stop_loss": new_potential_sl}

    return None

def reanalyze_open_trade(signal: Dict, df: pd.DataFrame) -> Tuple[Optional[Dict], Optional[str]]:
    """
    إعادة تحليل صفقة مفتوحة لاتخاذ قرارات إدارية (خروج مبكر، رفع هدف).
    (شروط مخففة للخروج المبكر)
    """
    symbol = signal['symbol']
    last = df.iloc[-1]
    
    # 1. فحص علامات الضعف للخروج المبكر (تخفيف كبير)
    is_weak = False
    weakness_reason = ""
    # الخروج فقط إذا كسر EMA50 (بدلاً من EMA21)
    if (last['close'] < last['ema50']) and (df.iloc[-2]['close'] > df.iloc[-2]['ema50']):
        is_weak = True
        weakness_reason = "Price crossed below EMA50"
    # الخروج إذا كان MACD سلبياً لـ 3 شموع
    elif last['macd_hist'] < 0 and df['macd_hist'].iloc[-2] < 0 and df['macd_hist'].iloc[-3] < 0:
        is_weak = True
        weakness_reason = "MACD histogram negative for 3 bars"

    if is_weak:
        logger.warning(f"🚨 [{symbol}] Weakness detected: {weakness_reason}. Triggering early exit.")
        return None, "early_exit_weakness"

    # 2. فحص القوة الاستثنائية لرفع الهدف الثاني (تخفيف)
    details = signal.get('signal_details', {})
    if isinstance(details, str): details = json.loads(details)
    if details.get('tp1_done'):
        # شروط أسهل للزخم
        is_strong_momentum = last['rsi'] > 65 and last['adx'] > 25 # تخفيف: 68->65, 28->25
        if is_strong_momentum:
            current_tp2 = float(signal['target_price_2'])
            new_tp2 = current_tp2 + (last['atr'] * 1.8) # زيادة: 1.5 -> 1.8
            logger.info(f"🚀 [{symbol}] Exceptional strength detected. Raising TP2 from {current_tp2:.4f} to {new_tp2:.4f}")
            send_enhanced_telegram_message(f"🚀 *رفع الهدف* لـ `{symbol}`\nتم اكتشاف زخم قوي، تم رفع الهدف الثاني إلى `{new_tp2:.4f}`.")
            return {"target_price_2": new_tp2}, None

    return None, None

# --- Flask API Routes ---
@app.route('/')
def dashboard(): return render_template_string(DASHBOARD_TEMPLATE, STRATEGY_NAMES=STRATEGY_NAMES)

@app.route('/settings')
def settings_page():
    with trade_amount_lock:
        trade_amount_min = FIXED_TRADE_AMOUNT_MIN_USDT
        trade_amount_max = FIXED_TRADE_AMOUNT_MAX_USDT
    with trading_mode_lock: is_paper_mode = paper_trading_mode
    with min_quality_lock: min_quality = MIN_SIGNAL_QUALITY
    
    with strategy_filters_lock:
        # --- NEW: التأكد من أن القاموس المرجعي محدث ---
        strategies_status = {}
        for key, info in ENHANCED_STRATEGIES.items():
            strategies_status[key] = info['enabled']

    return render_template_string(SETTINGS_TEMPLATE, 
                                  trade_amount_min=trade_amount_min,
                                  trade_amount_max=trade_amount_max,
                                  MAX_OPEN_TRADES=MAX_OPEN_TRADES,
                                  min_quality=min_quality,
                                  is_paper_mode=is_paper_mode,
                                  STRATEGY_NAMES=STRATEGY_NAMES,
                                  strategies_status=strategies_status) # استخدام القاموس المحدث

@app.route('/api/dashboard_data')
def dashboard_data():
    try: return jsonify(get_dashboard_payload())
    except Exception as e:
        logger.error(f"❌ [API Error] Failed to generate dashboard data: {e}", exc_info=True)
        return jsonify({"error": "Failed to load dashboard data."}), 500

@app.route('/toggle_trading', methods=['POST'])
def toggle_trading():
    global is_trading_enabled
    with trading_status_lock: is_trading_enabled = not is_trading_enabled
    status_msg = "enabled" if is_trading_enabled else "disabled"
    log_and_notify("info", f"Trading has been {status_msg}.", "TRADING_STATUS")
    return jsonify({"status": "success", "trading_enabled": is_trading_enabled})

@app.route('/api/settings', methods=['POST'])
def update_settings():
    try:
        data = request.json
        if 'FIXED_TRADE_AMOUNT_MIN_USDT' in data and 'FIXED_TRADE_AMOUNT_MAX_USDT' in data:
            with trade_amount_lock:
                global FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT
                FIXED_TRADE_AMOUNT_MIN_USDT = float(data['FIXED_TRADE_AMOUNT_MIN_USDT'])
                FIXED_TRADE_AMOUNT_MAX_USDT = float(data['FIXED_TRADE_AMOUNT_MAX_USDT'])
                broadcast({"type": "trade_amount_update", "payload": {"min": FIXED_TRADE_AMOUNT_MIN_USDT, "max": FIXED_TRADE_AMOUNT_MAX_USDT}})
        if 'MAX_OPEN_TRADES' in data:
            global MAX_OPEN_TRADES
            MAX_OPEN_TRADES = int(data['MAX_OPEN_TRADES'])
        if 'paper_trading_mode' in data:
            with trading_mode_lock:
                global paper_trading_mode
                paper_trading_mode = bool(data['paper_trading_mode'])
                broadcast({"type": "trading_mode", "payload": {"paper_trading": paper_trading_mode}})
        
        save_settings_to_redis()
        return jsonify({"success": True, "message": "Settings updated successfully"})
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/health')
def api_health():
    try:
        with trading_status_lock: trading_enabled = is_trading_enabled
        with trading_mode_lock: is_paper = paper_trading_mode
        return jsonify({"status": "ok", "trading_enabled": trading_enabled, "mode": "PAPER" if is_paper else "REAL", "open_signals": len(open_signals_cache), "ws": {"connected": True}}), 200
    except Exception as e: return jsonify({"status": "error", "message": str(e)}), 500

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

@app.route('/api/performance_metrics')
def get_performance_metrics():
    cache_key = "performance_metrics_30d"
    if redis_client:
        cached_data = redis_client.get(cache_key)
        if cached_data: return jsonify(json.loads(cached_data))
    if not check_db_connection(): return jsonify({"error": "Database connection failed"}), 500
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
        if redis_client: redis_client.setex(cache_key, 300, json.dumps(result, cls=NpEncoder))
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/signals_history')
def get_signals_history():
    if not check_db_connection(): return jsonify({"error": "Database connection failed"}), 500
    page = request.args.get('page', 1, type=int)
    per_page = 20
    offset = (page - 1) * per_page
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM signals WHERE status = 'closed' ORDER BY closed_at DESC LIMIT %s OFFSET %s", (per_page, offset))
        signals = cur.fetchall()
        cur.execute("SELECT COUNT(*) FROM signals WHERE status = 'closed'")
        total = cur.fetchone()['count']
    return jsonify({"signals": [dict(s) for s in signals], "pagination": {"page": page, "per_page": per_page, "total": total, "pages": (total + per_page - 1) // per_page}})

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

@app.route('/api/advanced_performance_data')
def advanced_performance_data():
    if not check_db_connection() or not conn: return jsonify({"error": "DB connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT profit_percentage, closed_at FROM signals WHERE status = 'closed' AND closed_at >= NOW() - INTERVAL '30 days' ORDER BY closed_at ASC")
            trades = cur.fetchall()
        if len(trades) < 2:
            return jsonify({"winRate": 0, "profitFactor": 0, "maxDrawdown": 0, "sharpeRatio": 0, "equity_curve": {"labels": [], "values": []}})
        profits = [t['profit_percentage'] for t in trades if t['profit_percentage'] is not None]
        wins = [p for p in profits if p > 0]; losses = [p for p in profits if p < 0]
        win_rate = (len(wins) / len(profits) * 100) if profits else 0
        total_profit = sum(wins); total_loss = abs(sum(losses))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        equity_curve_values = [1000]
        for p in profits: equity_curve_values.append(equity_curve_values[-1] * (1 + p / 100))
        peak = equity_curve_values[0]; max_drawdown = 0
        for equity in equity_curve_values:
            if equity > peak: peak = equity
            drawdown = (peak - equity) / peak * 100
            if drawdown > max_drawdown: max_drawdown = drawdown
        returns = np.array(profits) / 100
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(len(trades)) if np.std(returns) > 0 else 0
        equity_curve_labels = [t['closed_at'].isoformat() for t in trades]
        return jsonify({
            "winRate": win_rate, "profitFactor": profit_factor, "maxDrawdown": max_drawdown,
            "sharpeRatio": sharpe_ratio, "equity_curve": {"labels": equity_curve_labels, "values": equity_curve_values[1:]}
        })
    except Exception as e:
        logger.error(f"❌ [API] Error fetching advanced performance data: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/strategies', methods=['POST'])
def update_strategies():
    try:
        data = request.json
        with strategy_filters_lock:
            for strategy_key in ENHANCED_STRATEGIES:
                if strategy_key in data:
                    ENHANCED_STRATEGIES[strategy_key]['enabled'] = bool(data[strategy_key])

        save_settings_to_redis()
        return jsonify({"success": True, "message": "Strategies updated successfully"})
    except Exception as e:
        logger.error(f"Error updating strategies: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/signal_quality', methods=['POST'])
def update_signal_quality():
    try:
        data = request.json
        if 'min_quality' in data:
            with min_quality_lock:
                global MIN_SIGNAL_QUALITY
                MIN_SIGNAL_QUALITY = int(data['min_quality'])
                broadcast({"type": "quality_filter", "payload": {"min_quality": MIN_SIGNAL_QUALITY}})
        save_settings_to_redis()
        return jsonify({"success": True, "message": "Signal quality settings updated successfully"})
    except Exception as e:
        logger.error(f"Error updating signal quality settings: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

def close_trade_manually(signal_id: int, closing_price: Optional[float] = None) -> bool:
    with signal_cache_lock:
        signal_to_close = next((dict(s) for s in open_signals_cache.values() if s['id'] == signal_id), None)

    if signal_to_close:
        symbol = signal_to_close['symbol']
        if closing_price is None:
            with live_prices_lock: closing_price = live_prices.get(symbol)
            if closing_price is None:
                logger.error(f"[Manual Close] لم يتم العثور على السعر الحالي لـ {symbol} لإغلاق الصفقة {signal_id}.")
                send_enhanced_telegram_message(f"⚠️ *فشل الإغلاق اليدوي لـ {symbol}* \nلم يتمكن البوت من الحصول على السعر الحالي.", force=True)
                return False
        
        logger.info(f"[Manual Close] بدأ المستخدم إغلاقاً يدوياً للصفقة {signal_id} ({symbol}) عند سعر {closing_price}")
        close_signal(signal_to_close, closing_price, "manual_close")
        return True
    
    logger.info(f"[Manual Close] لم يتم العثور على الصفقة {signal_id} في الكاش. يتم الآن البحث في قاعدة البيانات...")
    if not check_db_connection() or not conn:
        logger.error("[Manual Close] لا يمكن التحقق من قاعدة البيانات بسبب مشكلة في الاتصال.")
        return False

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, status, closing_reason FROM signals WHERE id = %s;", (signal_id,))
            db_signal = cur.fetchone()

        if db_signal:
            if db_signal['status'] == 'closed':
                reason = db_signal.get('closing_reason', 'غير معروف')
                logger.info(f"✅ [Manual Close] تم تجاهل طلب إغلاق الصفقة {signal_id} لأنها مغلقة بالفعل. سبب الإغلاق: {reason}")
                return True
            else:
                logger.warning(f"⚠️ [Manual Close] عدم تطابق في البيانات! الصفقة {signal_id} مفتوحة في قاعدة البيانات ولكنها غير موجودة في الكاش.")
                return False
        else:
            logger.warning(f"❌ [Manual Close] فشل الإغلاق: الصفقة {signal_id} غير موجودة في الذاكرة المؤقتة أو قاعدة البيانات.")
            return False
    except Exception as e:
        logger.error(f"❌ [Manual Close] حدث خطأ أثناء التحقق من قاعدة البيانات للصفقة {signal_id}: {e}", exc_info=True)
        return False

@app.route('/api/close_trade/<int:signal_id>', methods=['POST'])
def api_close_trade(signal_id):
    data = request.get_json(silent=True) or {}
    closing_price = data.get('closing_price')
    thread = Thread(target=close_trade_manually, args=(signal_id, closing_price))
    thread.start()
    return jsonify({"success": True, "message": "Trade close command received and is being processed."})

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
    global usdt_balance
    symbol, signal_id, entry_price = signal['symbol'], signal['id'], float(signal['entry_price'])
    
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
                logger.info(f"💰 [Real Close] For {symbol}: Bot wants to sell {quantity_in_bot}, Available on Binance: {available_on_exchange}")
                
                quantity_to_sell = available_on_exchange
                
                if quantity_to_sell > 0:
                    adjusted_quantity_to_sell = adjust_quantity_to_lot_size(symbol, float(quantity_to_sell), logger=logger)
                    
                    if adjusted_quantity_to_sell and adjusted_quantity_to_sell > 0:
                        formatted_sell_quantity = get_formatted_quantity(symbol, adjusted_quantity_to_sell)
                        logger.info(f"💰 [Real Close] Executing MARKET SELL for {formatted_sell_quantity} of {symbol} due to {reason}")
                        client.create_order(symbol=symbol, side=Client.SIDE_SELL, type=Client.ORDER_TYPE_MARKET, quantity=formatted_sell_quantity)
                    else:
                        logger.warning(f"⚠️ [Real Close] Adjusted sell quantity for {symbol} is zero or None. Skipping API sell call.")
                else:
                    logger.warning(f"⚠️ [Real Close] No available quantity of {asset} to sell for {symbol}. Closing in DB only.")
        except BinanceAPIException as e:
            logger.error(f"❌ [Real Close] Binance API Error for {symbol}: {e}")
            send_enhanced_telegram_message(f"❌ *خطأ في تنفيذ إغلاق لـ {symbol}*\n`{e}`", force=True)
        except Exception as e:
            logger.error(f"❌ [Real Close] CRITICAL ERROR for {symbol}: {e}", exc_info=True)

    profit = 0.0
    if entry_price > 0:
        profit = ((closing_price - entry_price) / entry_price) * 100
    
    with consecutive_losses_lock:
        if profit < 0: consecutive_losses_by_symbol[symbol] = consecutive_losses_by_symbol.get(symbol, 0) + 1
        else: consecutive_losses_by_symbol[symbol] = 0
    update_signal_in_db(signal_id, {"status": "closed", "closing_price": closing_price, "closed_at": datetime.now(timezone.utc), "profit_percentage": profit, "closing_reason": reason})
    with signal_cache_lock:
        if symbol in open_signals_cache: del open_signals_cache[symbol]
    broadcast({"type": "trade_closed", "payload": {"signal_id": signal_id, "symbol": symbol, "reason": reason}})
    trade_type = "حقيقية" if signal.get('is_real_trade') else "ورقية"
    result_emoji = "✅" if profit >= 0 else "🔻"
    reason_map = {
        "SL_HIT": "ضرب وقف الخسارة", "TP1_HIT": "تحقيق الهدف الأول", "TP2_HIT": "تحقيق الهدف الثاني",
        "manual_close": "إغلاق يدوي", "TRAILING_SL_HIT": "ضرب الوقف المتحرك",
        "early_exit_weakness": "خروج مبكر (ضعف الإشارة)"
    }
    reason_ar = reason_map.get(reason, reason)
    log_and_notify("info", f"Closed {trade_type} trade for {symbol}. Profit: {profit:.2f}%", "TRADE_CLOSED")
    settings = get_notification_settings()
    if (profit >= settings['min_profit_notification'] or profit <= settings['max_loss_notification'] or reason == "manual_close"):
        send_enhanced_telegram_message(f"{result_emoji} *إغلاق صفقة {trade_type} {symbol}*\n*السبب:* {reason_ar}\n*الربح:* `{profit:.2f}%`")

def get_mtf_trend(symbol: str) -> Dict[str, str]:
    trends = {}
    timeframes = {'5m': 7, '15m': 10, '1h': 12} 

    for tf, days in timeframes.items():
        try:
            df = fetch_historical_data(symbol, tf, days)
            if df is None or len(df) < 50:
                trends[tf] = 'unknown'
                continue

            df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
            if 'adx' not in df.columns: # التأكد من وجود ADX
                df = calculate_all_features(df)
            last = df.iloc[-1]

            if last['close'] > last['ema50'] and last['adx'] > 20:
                trends[tf] = 'bullish'
            elif last['close'] < last['ema50'] and last['adx'] > 20:
                trends[tf] = 'bearish'
            else:
                trends[tf] = 'sideways'
            time.sleep(API_REQUEST_DELAY / 2) # تأخير بسيط بين طلبات MTF
        except Exception as e:
            logger.warning(f"[MTF Trend] Could not determine trend for {symbol} on {tf}: {e}")
            trends[tf] = 'unknown'
            
    return trends
    
def main_bot_loop():
    logger.info("🚀 [Main Loop] Starting signal scanning loop (5-minute cycle)...")
    while True:
        try:
            while True:
                now = datetime.now(timezone.utc)
                # --- NEW: تعديل توقيت الفحص ---
                # الفحص قبل 10 ثواني من إغلاق الشمعة
                seconds_until_next_candle = (5 - (now.minute % 5)) * 60 - now.second
                
                is_enabled_now = False
                with trading_status_lock:
                    is_enabled_now = is_trading_enabled

                if is_enabled_now and seconds_until_next_candle <= 10:
                    time.sleep(seconds_until_next_candle + 1) # الانتظار حتى فتح الشمعة الجديدة
                    break 

                time.sleep(1)

            with trading_status_lock:
                if not is_trading_enabled:
                    logger.info("Trading was disabled during the wait. Skipping scan cycle.")
                    continue
            
            logger.info("="*20 + " Starting New 5-Min Scan Cycle " + "="*20)
            
            # --- NEW: تحديث قائمة العملات قبل كل فحص ---
            # هذا يضمن أننا نفحص أحدث قائمة
            global validated_symbols_to_scan
            validated_symbols_to_scan = get_validated_symbols()
            
            for symbol in validated_symbols_to_scan:
                with signal_cache_lock:
                    if len(open_signals_cache) >= MAX_OPEN_TRADES:
                        logger.info(f"Max open trades ({MAX_OPEN_TRADES}) reached. Pausing scan.")
                        break
                    if symbol in open_signals_cache:
                        continue
                
                # --- NEW: إضافة تأخير هنا للتحكم في معدل الطلبات ---
                time.sleep(API_REQUEST_DELAY)
                
                mtf_trend = get_mtf_trend(symbol)

                df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                if df is None or len(df) < 200:
                    if df is not None: log_rejection(symbol, "Insufficient Historical Data")
                    continue
                
                df_featured = calculate_all_features(df)
                df_featured.name = symbol
                
                strategy_found_tuple = find_best_strategy(df_featured, mtf_trend, symbol)

                if strategy_found_tuple:
                    strategy_key, strategy_name_ar = strategy_found_tuple
                    logger.info(f"✅ [{symbol}] Found suitable strategy: {strategy_name_ar} ({strategy_key})")
                    create_trade_signal(symbol, df_featured, strategy_key, mtf_trend)

        except Exception as e:
            logger.error(f"❌ [Main Loop] A critical error occurred: {e}", exc_info=True)
            time.sleep(60)

def trade_management_loop():
    logger.info("🚀 [Trade Manager] Starting advanced trade management loop...")
    while True:
        try:
            with signal_cache_lock:
                if not open_signals_cache:
                    time.sleep(2)
                    continue
                signals_to_monitor = list(open_signals_cache.values())

            for signal in signals_to_monitor:
                symbol = signal['symbol']
                with live_prices_lock:
                    current_price = live_prices.get(symbol)
                if not current_price:
                    continue
                
                details = signal.get('signal_details', {})
                if isinstance(details, str):
                    try: details = json.loads(details)
                    except Exception: details = {}
                
                # --- Standard SL/TP/Trailing SL checks ---
                entry_price = float(signal.get('entry_price', 0))
                stop_loss = float(signal.get('stop_loss', 0))
                tp1 = float(signal.get('target_price_1', 0))
                tp2 = float(signal.get('target_price_2', 0))
                initial_quantity = float(signal.get('initial_quantity', 0))
                
                if stop_loss and current_price <= stop_loss: close_signal(signal, stop_loss, "SL_HIT"); continue
                if tp2 and current_price >= tp2: close_signal(signal, tp2, "TP2_HIT"); continue
                
                if tp1 and not details.get('tp1_done') and current_price >= tp1:
                    # (منطق إغلاق 50% من العقد يمكن إضافته هنا إذا كان مطلوباً في التداول الحقيقي)
                    # حالياً، نقوم فقط بتحريك الوقف وتحديث الحالة
                    new_sl = max(stop_loss, entry_price) # تحريك الوقف للدخول
                    updates = {"stop_loss": new_sl, "status": "updated"}
                    details['tp1_done'] = True
                    updates['signal_details'] = json.dumps(details)
                    update_signal_in_db(signal['id'], updates)
                    send_enhanced_telegram_message(f"🥇 *تحقق الهدف الأول* لـ `{symbol}`\nتم تحريك الوقف إلى نقطة الدخول ({new_sl:.4f}).")
                    continue
                
                # --- Re-analysis & Active Management (Relaxed) ---
                now_utc = datetime.now(timezone.utc)
                last_analysis_str = details.get('last_reanalysis')
                last_analysis_time = datetime.fromisoformat(last_analysis_str) if last_analysis_str else now_utc - timedelta(minutes=6)

                if (now_utc - last_analysis_time).total_seconds() >= 300: # Re-analyze every 5 minutes
                    logger.info(f"🔄 [{symbol}] Re-analyzing open trade...")
                    df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, days=5)
                    if df is None or len(df) < 50: continue
                    df_featured = calculate_all_features(df)
                    
                    updates, action = reanalyze_open_trade(signal, df_featured)
                    
                    details['last_reanalysis'] = now_utc.isoformat()
                    final_updates = {'signal_details': json.dumps(details)}

                    if updates:
                        final_updates.update(updates)
                    
                    update_signal_in_db(signal['id'], final_updates)
                    
                    if action == "early_exit_weakness":
                        close_signal(signal, current_price, action)
                        continue

                # Trailing SL (only after TP1)
                if details.get('tp1_done'):
                    df_trail = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, days=2)
                    if df_trail is not None and not df_trail.empty:
                        df_trail_featured = calculate_all_features(df_trail)
                        trailing_update = manage_intelligent_trailing_stop(signal, current_price, df_trail_featured)
                        if trailing_update:
                            update_signal_in_db(signal['id'], trailing_update)

            time.sleep(1)
        except Exception as e:
            logger.error(f"❌ [Trade Manager] Loop error: {e}", exc_info=True)
            time.sleep(5)

# ===== NEW: Market Consensus State Updater =====
def update_market_state():
    """
    تحديث حالة السوق بناءً على سلة من العملات القيادية (BTC, ETH, SOL, BNB)
    """
    global current_market_state
    logger.info("[Market State] Updating market consensus...")
    
    leader_trends = {}
    total_score = 0.0
    leaders_processed = 0
    
    try:
        for symbol in MARKET_LEADERS:
            try:
                # استخدام إطار ساعة لتحليل الاتجاه العام
                df = fetch_historical_data(symbol, '1h', days=10)
                if df is None or len(df) < 50:
                    logger.warning(f"[Market State] Insufficient data for {symbol}")
                    continue
                
                df = calculate_all_features(df)
                last = df.iloc[-1]
                
                symbol_trend = "sideways"
                symbol_score = 0.5 # الافتراضي (جانبي)
                
                # --- منطق تحديد الاتجاه (بسيط ومباشر) ---
                # 1. فوق EMA50 و ADX قوي = صاعد
                if last['close'] > last['ema50'] and last['adx'] > 20:
                    symbol_trend = "bullish"
                    symbol_score = 1.0
                # 2. تحت EMA50 و ADX قوي = هابط
                elif last['close'] < last['ema50'] and last['adx'] > 20:
                    symbol_trend = "bearish"
                    symbol_score = 0.0
                # 3. حالات أخرى (ADX ضعيف) = جانبي
                
                leader_trends[symbol] = {"trend": symbol_trend, "score": symbol_score}
                total_score += symbol_score
                leaders_processed += 1
                
                # --- هام: إضافة تأخير بين الطلبات لتجنب حظر IP ---
                time.sleep(API_REQUEST_DELAY)
                
            except Exception as e:
                logger.error(f"❌ [Market State] Error processing leader {symbol}: {e}")
                
        if leaders_processed == 0:
            logger.error("❌ [Market State] Failed to process any market leaders.")
            return

        # حساب النتيجة النهائية
        market_score = total_score / leaders_processed
        market_consensus = "sideways"
        
        if market_score > 0.65: # أغلبية صاعدة
            market_consensus = "bullish"
        elif market_score < 0.35: # أغلبية هابطة
            market_consensus = "bearish"
            
        logger.info(f"✅ [Market State] Consensus Updated: {market_consensus.upper()} (Score: {market_score:.2f})")

        with market_state_lock:
            current_market_state = {
                "market_consensus": market_consensus,
                "market_score": market_score,
                "leader_trends": leader_trends,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
        
        broadcast({"type": "market_state_update", "payload": current_market_state})

    except Exception as e:
        logger.error(f"❌ [Market State] Error updating market state: {e}", exc_info=True)


def start_market_state_updater():
    def update_loop():
        while True:
            try:
                update_market_state()
                # تحديث كل 15 دقيقة (أسرع من قبل)
                time.sleep(60 * 15)
            except Exception as e:
                logger.error(f"[Market State Updater] Error in update loop: {e}")
                time.sleep(60)
    thread = Thread(target=update_loop, daemon=True)
    thread.start()
    logger.info("[Market State] Started market state updater thread (using leaders)")

def update_balance():
    try:
        balance_info = client.get_asset_balance(asset='USDT')
        with balance_lock:
            global usdt_balance
            usdt_balance = float(balance_info['free'])
    except Exception as e: 
        logger.error(f"❌ [Balance] Could not update REAL USDT balance: {e}")

def update_balance_loop():
    logger.info("🚀 [Balance Updater] Starting balance update loop...")
    while True:
        try: update_balance()
        except Exception as e: logger.error(f"❌ [Balance Loop] Error: {e}", exc_info=True)
        time.sleep(60 * 5) # تحديث الرصيد كل 5 دقائق

# --- نقطة بداية البرنامج ---
if __name__ == '__main__':
    logger.info("="*50 + "\n====== Starting Crypto Trading Bot V36.0 (Relaxed) ======\n" + "="*50)
    init_db()
    init_redis()
    try:
        client = Client(API_KEY, API_SECRET); client.ping()
        logger.info("✅ [Binance] API connection successful.")
    except Exception as e:
        logger.critical(f"❌ [Binance] API connection failed: {e}"); exit(1)
    
    # --- NEW: جلب معلومات المنصة قبل جلب العملات ---
    get_exchange_info_map()
    validated_symbols_to_scan = get_validated_symbols()
    if not validated_symbols_to_scan:
        logger.critical("❌ No valid symbols to scan. Exiting."); exit(1)
    
    # Load initial data
    load_open_signals_to_cache()
    load_notifications_to_cache()
    load_settings_from_redis()
    logger.info("Fetching initial real account balance...")
    update_balance()
    with balance_lock:
        logger.info(f"Initial real balance fetched: ${usdt_balance:.2f}")

    logger.info("Initial data fetch complete.")
    
    # Start background threads
    start_websocket()
    Thread(target=main_bot_loop, daemon=True).start()
    Thread(target=trade_management_loop, daemon=True).start()
    start_market_state_updater() # --- NEW: بدء محدث السوق الجديد ---
    Thread(target=update_balance_loop, daemon=True).start()
    start_periodic_reports()
    
    # Start Flask App
    logger.info("🌐 [Flask] Starting UI on http://0.0.0.0:5000")
    # --- FIX: Use 'waitress' for production as requested by error log ---
    # app.run(host='0.0.0.0', port=5000, debug=False)
    serve(app, host='0.0.0.0', port=5000)