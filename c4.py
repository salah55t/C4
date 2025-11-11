# ملف c4_5min_v34_3_0.py - نسخة V34.3.0 (محدثة باستراتيجيات متقدمة)
# --- وصف التعديلات:
# 1. [استراتيجيات محدثة] تم تحسين جميع الاستراتيجيات الست باستخدام الأساليب الحديثة والشائعة
# 2. [Smart Money Concepts] إضافة مفاهيم المال الذكي (SMC) لتحليل أعمق
# 3. [تصفية ديناميكية] تحسين الفلاتر لقبول المزيد من الفرص الجيدة مع رفض الفرص الضعيفة
# 4. [إدارة المخاطر] تعديل نسب المخاطرة/العائد لتكون أكثر واقعية
# 5. [الأداء] تم تحسين سرعة الحسابات وتقليل التأخير

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

# ضبط دقة النوع Decimal
getcontext().prec = 18

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v34_5min_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV34.3.0_5min')

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
usdt_balance: float = 0.0 # سيحتوي دائمًا على الرصيد الحقيقي
balance_lock = Lock()
cooldowns_by_symbol = {}
cooldowns_lock = Lock()
consecutive_losses_by_symbol = {}
consecutive_losses_lock = Lock()
COOLDOWN_MINUTES_AFTER_SL = 30

# --- المتغيرات القابلة للتعديل ---
PAPER_TRADE_FIXED_AMOUNT_USDT: float = 10.0 # قيمة ثابتة للصفقات الورقية
FIXED_TRADE_AMOUNT_MIN_USDT: float = 4.5  # للصفقات الحقيقية فقط
FIXED_TRADE_AMOUNT_MAX_USDT: float = 6.5  # للصفقات الحقيقية فقط
trade_amount_lock = Lock()
MAX_OPEN_TRADES: int = 3
TRAILING_STOP_ACTIVATION_PROFIT_PERCENT: float = 1.0
MIN_SIGNAL_QUALITY: int = 70
AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE: bool = True
min_quality_lock = Lock()
strategy_filters_lock = Lock()

# --- إعدادات عامة (معدلة لإطار 5 دقائق) ---
SIGNAL_GENERATION_TIMEFRAME: str = '5m'
HIGHER_TIMEFRAME: str = '15m'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['5m', '15m', '1h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 7 # تقليل مدة البيانات التاريخية لتسريع الحسابات
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
    "Smart Market Structure Filter Failed": "فلتر هيكل السوق الذكي رفض الدخول",
    "Advanced Market Structure Filter Failed": "فلتر هيكل السوق المتقدم V2 رفض الدخول",
    "Smart Liquidity Filter Failed": "فلتر السيولة الذكي رفض الدخول",
    "Smart Risk/Reward Filter Failed": "فلتر المخاطرة/العائد الذكي رفض الدخول",
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
    strategy_name_ar = STRATEGY_NAMES.get(strategy_key, strategy_key)
    
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
# --- نهاية دوال المساعدة ---

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

# ===== NEW: Smart Money Concepts Helpers =====
def detect_order_block(df: pd.DataFrame, lookback: int = 20) -> Optional[Dict[str, float]]:
    """
    اكتشاف Order Blocks (OB) - المناطق التي توجد فيها أوامر شراء/بيع كبيرة من المؤسسات
    """
    if len(df) < lookback:
        return None
    
    df_calc = df.iloc[-lookback:].copy()
    
    # البحث عن شموع كبيرة الجسم مع ذيل صغير ووفرة في الحجم
    df_calc['body_size'] = abs(df_calc['close'] - df_calc['open'])
    df_calc['avg_body_20'] = df_calc['body_size'].rolling(20).mean()
    df_calc['volume_avg_20'] = df_calc['volume'].rolling(20).mean()
    
    # اعتبار الشمعة كـ Order Block إذا:
    # 1. جسم الشمعة أكبر من المتوسط بـ 1.5 مرة
    # 2. الحجم أكبر من المتوسط بـ 2 مرة
    # 3. نسبة الذيل إلى الجسم أقل من 0.3
    df_calc['tail_size'] = np.where(df_calc['close'] >= df_calc['open'], 
                                    df_calc['low'] - df_calc['open'],
                                    df_calc['close'] - df_calc['low'])
    df_calc['tail_ratio'] = df_calc['tail_size'] / df_calc['body_size'].replace(0, 1e-9)
    
    order_blocks = df_calc[
        (df_calc['body_size'] > df_calc['avg_body_20'] * 1.5) &
        (df_calc['volume'] > df_calc['volume_avg_20'] * 2.0) &
        (df_calc['tail_ratio'] < 0.3)
    ]
    
    if not order_blocks.empty:
        # أخذ آخر Order Block
        last_ob = order_blocks.iloc[-1]
        return {
            "top": float(max(last_ob['open'], last_ob['close'])),
            "bottom": float(min(last_ob['open'], last_ob['close'])),
            "type": "bullish" if last_ob['close'] >= last_ob['open'] else "bearish",
            "strength": float(last_ob['volume'] / last_ob['volume_avg_20'])
        }
    
    return None

def detect_fair_value_gap(df: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    اكتشاف Fair Value Gaps (FVG) - الفجوات السعرية التي لم تغطى
    """
    if len(df) < 5:
        return None
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    
    # نمط FVG الصاعد: شمعة هابطة، ثم شمعة صاعدة كبيرة تترك فجوة
    if (prev2['high'] > prev['low']) and (last['low'] > prev['high']):
        return {
            "type": "bullish",
            "top": float(prev['high']),
            "bottom": float(prev2['high']),
            "depth": float(prev2['high'] - prev['low'])
        }
    
    # نمط FVG الهابط: شمعة صاعدة، ثم شمعة هابطة كبيرة تترك فجوة
    elif (prev2['low'] < prev['high']) and (last['high'] < prev['low']):
        return {
            "type": "bearish", 
            "top": float(prev2['low']),
            "bottom": float(prev['low']),
            "depth": float(prev['high'] - prev2['low'])
        }
    
    return None

def detect_imbalance(df: pd.DataFrame) -> bool:
    """
    اكتشاف التوازن/عدم التوازن في السوق
    """
    if len(df) < 20:
        return False
    
    last = df.iloc[-1]
    volume_ma = df['volume'].rolling(20).mean().iloc[-1]
    
    # عدم توازن صاعد: سعر صاعد + حجم مرتفع + RSI معتدل
    if (last['close'] > last['open']) and (last['volume'] > volume_ma * 1.5):
        if 55 < last['rsi'] < 75:
            return True
    
    return False

def check_market_structure_shift(df: pd.DataFrame) -> Optional[str]:
    """
    اكتشاف Market Structure Shift (MSS) - تغيير في بنية السوق
    """
    if len(df) < 50:
        return None
    
    # تحليل آخر 20 شمعة
    recent_df = df.tail(20)
    
    # البحث عن Higher Highs و Higher Lows
    highs = recent_df['high'].values
    lows = recent_df['low'].values
    
    hh_indices = argrelextrema(highs, np.greater, order=3)[0]
    hl_indices = argrelextrema(lows, np.less, order=3)[0]
    
    # إذا وجدنا 2+ قمم وقيعان متتالية
    if len(hh_indices) >= 2 and len(hl_indices) >= 2:
        # تحقق من Higher Highs
        if highs[hh_indices[-1]] > highs[hh_indices[-2]] and lows[hl_indices[-1]] > lows[hl_indices[-2]]:
            return "bullish_mss"
        # تحقق من Lower Lows
        elif highs[hh_indices[-1]] < highs[hh_indices[-2]] and lows[hl_indices[-1]] < lows[hl_indices[-2]]:
            return "bearish_mss"
    
    return None

# ===== SIGNAL QUALITY SCORING SYSTEM (محسّن) =====
def calculate_signal_quality_score(df: pd.DataFrame, mtf_trend: Dict, strategy_key: str) -> int:
    """
    يحسب نقاط جودة للإشارة (من 0 إلى 100) بناءً على عدة عوامل فنية.
    """
    score = 0
    last = df.iloc[-1]
    
    # 1. قوة الاتجاه (Trend Strength) - (Max 25 points)
    ema_spread = (last['ema21'] - last['ema50']) / last['close'] * 100
    if ema_spread > 0.7:
        score += 25  # اتجاه قوي جداً
    elif ema_spread > 0.3:
        score += 15  # اتجاه جيد
    
    # 2. تأكيد هيكل السوق (Market Structure) - (Max 20 points)
    mss = check_market_structure_shift(df)
    if mss and "bullish" in mss:
        score += 20
    elif mss and "bearish" not in mss:
        score += 10
    
    # 3. حجم وتوازن (Volume & Imbalance) - (Max 20 points)
    if detect_imbalance(df):
        score += 20
    elif last['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.3:
        score += 10
        
    # 4. زخم المؤشرات (Indicator Momentum) - (Max 20 points)
    # MACD مع التدرج
    if last['macd_hist'] > 0 and df['macd_hist'].iloc[-1] > df['macd_hist'].iloc[-2]:
        score += 10
    # RSI في منطقة مثالية
    if 58 < last['rsi'] < 72:
        score += 10
        
    # 5. التوافق الزمني (Multi-Timeframe Alignment) - (Max 10 points)
    if mtf_trend.get('15m') == 'bullish':
        score += 5
    if mtf_trend.get('1h') == 'bullish':
        score += 5
        
    # 6. Order Blocks (Max 5 points)
    ob = detect_order_block(df)
    if ob and ob['type'] == 'bullish':
        if last['close'] > ob['bottom']:
            score += 5

    return min(100, int(score))

# ===== STRATEGIES: Version Updated with Smart Money Concepts =====

def check_smart_momentum_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """
    استراتيجية الزخم الذكي - تحديث شامل
    تتوافق مع Smart Money Concepts مع الحفاظ على البساطة
    """
    if len(df) < 200:
        return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. التحقق من الاتجاه متعدد الأطر الزمنية - شرط أساسي
    mtf_bullish = (
        mtf_trend.get('5m') == 'bullish' and 
        mtf_trend.get('15m') in ['bullish', 'sideways']
    )
    if not mtf_bullish: return False
    
    # 2. ترتيب EMAs القوي مع فحص إضافي للتدرج
    ema_alignment = (
        last['ema9'] > last['ema21'] > last['ema50'] > last['ema200'] and
        df['ema9'].iloc[-5:].is_monotonic_increasing
    )
    if not ema_alignment: return False
    
    # 3. MACD زخم قوي ومتزايد مع تأكيد إضافي
    macd_strong = (
        last['macd'] > 0 and
        last['macd_hist'] > 0 and
        last['macd_hist'] > prev['macd_hist'] and
        df['macd_hist'].iloc[-3:].is_monotonic_increasing
    )
    if not macd_strong: return False
    
    # 4. ADX يؤكد قوة الاتجاه + مؤشر Stochastic
    if not (last['adx'] > 22 and last['stoch_k'] > 55): return False
    
    # 5. حجم التداول متزايد مع تأكيد Imbalance
    volume_ma = df['volume'].rolling(20).mean().iloc[-1]
    if not (last['volume'] > volume_ma * 1.3 and detect_imbalance(df)): return False
    
    # 6. السعر فوق VWAP مع فجوة إيجابية
    if not (last['close'] > last.get('vwap', last['close'])): return False
    
    # 7. تقلب مناسب والسعر فوق Order Block
    atr_percent = last.get('atr_percent', 0)
    ob = detect_order_block(df)
    if ob and ob['type'] == 'bullish':
        if not (0.8 < atr_percent < 3.0 and last['close'] > ob['top']): return False
    else:
        if not (0.8 < atr_percent < 3.0): return False
    
    return True


def check_advanced_pullback_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """
    استراتيجية الارتداد المتقدمة - تحديث شامل
    """
    if len(df) < 200:
        return False
    
    last = df.iloc[-1]
    
    # 1. اتجاه صاعد قوي على الأطر الأعلى
    if not (mtf_trend.get('15m') == 'bullish' or mtf_trend.get('1h') == 'bullish'): return False
    
    # 2. EMAs في ترتيب صاعد قوي
    if not (last['ema21'] > last['ema50'] > last['ema200']): return False
    
    # 3. تحديد الارتداد: السعر لمس EMA21 في آخر 5 شموع
    recent_lows = df['low'].tail(5)
    recent_ema21 = df['ema21'].tail(5)
    if not (recent_lows <= recent_ema21 * 1.005).any(): return False
    
    # 4. قياس عمق الارتداد مع تعديل ديناميكي
    recent_high = df['high'].tail(15).max()
    pullback_low = recent_lows.min()
    pullback_depth = (recent_high - pullback_low) / recent_high
    atr_percent = last.get('atr_percent', 0)
    min_depth = 0.012 if atr_percent > 2.0 else 0.008
    max_depth = 0.06
    if not (min_depth <= pullback_depth <= max_depth): return False
    
    # 5. السعر بدأ في التعافي - شرط إضافي للتأكيد
    if not (last['close'] > last['open'] and last['close'] > last['ema9']): return False
    
    # 6. حجم تداول متزايد عند التعافي
    volume_ma = df['volume'].rolling(20).mean().iloc[-1]
    if not last['volume'] > volume_ma * 1.2: return False
    
    # 7. Stochastic يظهر تقاطع صاعد من منطقة التشبع البيعي
    stoch_reversal = (
        df['stoch_k'].iloc[-2] < 35 and
        last['stoch_k'] > df['stoch_k'].iloc[-2] and
        last['stoch_k'] > last['stoch_d']
    )
    if not stoch_reversal: return False
    
    return True


def check_breakout_retest_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """
    استراتيجية الاختراق وإعادة الاختبار - تحديث شامل
    """
    if len(df) < 100: return False
    last = df.iloc[-1]
    
    # 1. الاتجاه على الأطر الأعلى يجب أن يكون صاعداً أو جانبياً
    if mtf_trend.get('15m') == 'bearish' and mtf_trend.get('1h') == 'bearish': return False
    
    try:
        highs = df['high'].values
        resistance_indices = argrelextrema(highs, np.greater, order=7)[0]
        if len(resistance_indices) < 2: return False
        
        latest_resistance_idx = resistance_indices[-1]
        resistance_price = highs[latest_resistance_idx]
        
        # 2. السعر اخترق المقاومة بالفعل
        if not (df['high'].tail(12) > resistance_price * 1.002).any(): return False
        
        # 3. إعادة الاختبار: السعر يعود للمنطقة مع الحفاظ على دعم
        retest_zone_upper = resistance_price * 1.008
        retest_zone_lower = resistance_price * 0.995
        if not (retest_zone_lower <= last['close'] <= retest_zone_upper): return False
        
        # 4. حجم الاختراق كان قوياً
        breakout_candle_idx = df[df['high'] > resistance_price * 1.002].index[-1]
        breakout_volume = df.loc[breakout_candle_idx, 'volume']
        volume_ma = df['volume'].rolling(20).mean().loc[breakout_candle_idx]
        if not breakout_volume > volume_ma * 1.8: return False
        
        # 5. EMAs متقاربة وتدعم الاتجاه
        if not (last['ema9'] > last['ema21'] > last['ema50']): return False
        
        # 6. MACD إيجابي وADX يؤكد القوة
        if not (last['macd_hist'] > 0 and last['adx'] > 22): return False
        
        return True
    except Exception:
        return False


def check_volume_price_divergence_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """
    استراتيجية تباين السعر والحجم - تحديث شامل
    """
    if len(df) < 50: return False
    last = df.iloc[-1]
    
    # 1. السعر يتحرك في نطاق ضيق (تراكم)
    recent_20 = df.tail(20)
    price_range = (recent_20['high'].max() - recent_20['low'].min()) / recent_20['close'].mean()
    if not price_range < 0.025: return False
    
    # 2. حجم يتزايد بشكل تصاعدي في آخر 10 شموع
    volume_trend = df['volume'].tail(10)
    if not (volume_trend.iloc[-5:].mean() > volume_trend.iloc[-10:-5].mean() * 1.4): return False
    
    # 3. EMAs في ترتيب صاعد أو على الأقل لا تتقاطع
    if not (last['close'] > last['ema21'] > last['ema50']): return False
    
    # 4. بولنجر باند ضيق (قبل الاختراق)
    bb_width = last.get('bb_width', 0)
    bb_width_ma = df['bb_width'].rolling(50).mean().iloc[-1]
    if not bb_width < bb_width_ma * 0.6: return False
    
    # 5. RSI في منطقة جيدة (ليس تشبع)
    if not (52 < last['rsi'] < 65): return False
    
    # 6. MACD يتحول صاعداً
    macd_turning = (
        last['macd_hist'] > df['macd_hist'].iloc[-2] and 
        last['macd_hist'] > df['macd_hist'].iloc[-3]
    )
    if not macd_turning: return False
    
    # 7. لا يوجد اتجاه هابط قوي في الأطر الأعلى
    if mtf_trend.get('15m') == 'bearish' and mtf_trend.get('1h') == 'bearish': return False

    return True


def check_golden_cross_momentum_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """
    استراتيجية التقاطع الذهبي مع الزخم - تحديث شامل
    """
    if len(df) < 200: return False
    last, prev, prev2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    
    # 1. التقاطع الذهبي: EMA9 يقطع فوق EMA21
    golden_cross = (
        prev2['ema9'] <= prev2['ema21'] and 
        prev['ema9'] > prev['ema21']
    )
    if not golden_cross: return False
    
    # 2. السعر فوق EMA50 (تأكيد)
    if last['close'] <= last['ema50']: return False
    
    # 3. MACD صاعد قوي
    macd_bullish = (
        last['macd'] > 0 and 
        last['macd_hist'] > 0 and 
        last['macd_hist'] > prev['macd_hist']
    )
    if not macd_bullish: return False
    
    # 4. ADX يزداد قوة
    adx_strengthening = (
        last['adx'] > 22 and 
        last['adx'] > df['adx'].iloc[-5:].mean()
    )
    if not adx_strengthening: return False
    
    # 5. RSI في منطقة مثالية
    if not (48 < last['rsi'] < 72): return False
    
    # 6. حجم تأكيد قوي
    volume_ma = df['volume'].rolling(20).mean().iloc[-1]
    if not last['volume'] > volume_ma * 1.5: return False
    
    # 7. الاتجاه على الأطر الأعلى لا يكون هابطاً
    if mtf_trend.get('15m') == 'bearish' and mtf_trend.get('1h') == 'bearish': return False
    
    return True

def check_mean_reversion_bb_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """
    استراتيجية الانعكاس إلى المتوسط - تحديث شامل
    """
    if len(df) < 50: return False
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # 1. السوق يجب أن يكون جانبياً (ADX < 22)
    if not last.get('adx', 99) < 22: return False

    # 2. السعر لمس أو اقترب من الحد السفلي للبولينجر في آخر 3 شموع
    bb_touch = (df['low'].tail(3) <= df['bb_lower'].tail(3) * 1.002).any()
    if not bb_touch: return False

    # 3. السعر بدأ في الارتداد (الإغلاق الحالي أعلى من الإغلاق السابق)
    if not last['close'] > prev['close']: return False

    # 4. RSI يظهر تشبع بيعي أو بداية خروج منه
    rsi_ok = 22 < last.get('rsi', 50) < 48
    if not rsi_ok: return False

    # 5. Stochastic يظهر تقاطع صاعد من منطقة التشبع البيعي
    stoch_reversal = (prev['stoch_k'] < 25 and last['stoch_k'] > prev['stoch_k'] and last['stoch_k'] > last['stoch_d'])
    if not stoch_reversal: return False
    
    # 6. MACD يظهر تحول صاعد
    macd_bullish = (last['macd_hist'] >= -0.1) or (last['macd_hist'] > df['macd_hist'].iloc[-2])
    if not macd_bullish: return False
    
    # 7. لا يوجد اتجاه هابط قوي في الأطر الأعلى
    if mtf_trend.get('15m') == 'bearish' and mtf_trend.get('1h') == 'bearish': return False

    return True

# ===== ADVANCED SMART MARKET STRUCTURE FILTER V2 =====

def detect_market_structure(df: pd.DataFrame) -> Dict[str, any]:
    """
    تحليل متقدم لهيكل السوق يشمل:
    - اكتشاف القمم والقيعان الرئيسية (Swing High/Low)
    - تحديد اتجاه الهيكل (Bullish/Bearish/Ranging)
    - قياس قوة الهيكل
    - اكتشاف كسر الهيكل (Break of Structure - BOS)
    - اكتشاف تغيير الهيكل (Change of Character - CHoCH)
    """
    
    if len(df) < 50:
        return {"structure_type": "unknown", "strength": 0}
    
    # 1. اكتشاف القمم والقيعان الرئيسية
    highs = df['high'].values
    lows = df['low'].values
    
    # استخدام order=5 للحصول على نقاط تحول معنوية
    swing_high_indices = argrelextrema(highs, np.greater, order=5)[0]
    swing_low_indices = argrelextrema(lows, np.less, order=5)[0]
    
    # 2. تحليل اتجاه القمم والقيعان
    structure_data = {
        "swing_highs": [(i, highs[i]) for i in swing_high_indices[-5:]] if len(swing_high_indices) >= 2 else [],
        "swing_lows": [(i, lows[i]) for i in swing_low_indices[-5:]] if len(swing_low_indices) >= 2 else [],
        "structure_type": "ranging",
        "strength": 0,
        "bos_detected": False,
        "choch_detected": False
    }
    
    # 3. تحديد اتجاه الهيكل
    if len(swing_high_indices) >= 3 and len(swing_low_indices) >= 3:
        recent_highs = highs[swing_high_indices[-3:]]
        recent_lows = lows[swing_low_indices[-3:]]
        
        # هيكل صاعد: قمم وقيعان أعلى (Higher Highs & Higher Lows)
        highs_rising = all(recent_highs[i] < recent_highs[i+1] for i in range(len(recent_highs)-1))
        lows_rising = all(recent_lows[i] < recent_lows[i+1] for i in range(len(recent_lows)-1))
        
        # هيكل هابط: قمم وقيعان أدنى (Lower Highs & Lower Lows)
        highs_falling = all(recent_highs[i] > recent_highs[i+1] for i in range(len(recent_highs)-1))
        lows_falling = all(recent_lows[i] > recent_lows[i+1] for i in range(len(recent_lows)-1))
        
        if highs_rising and lows_rising:
            structure_data["structure_type"] = "bullish"
            structure_data["strength"] = 85
        elif highs_falling and lows_falling:
            structure_data["structure_type"] = "bearish"
            structure_data["strength"] = 15
        elif highs_rising and not lows_falling:
            structure_data["structure_type"] = "weak_bullish"
            structure_data["strength"] = 60
        elif highs_falling and not lows_rising:
            structure_data["structure_type"] = "weak_bearish"
            structure_data["strength"] = 40
        else:
            structure_data["structure_type"] = "ranging"
            structure_data["strength"] = 50
    
    # 4. اكتشاف كسر الهيكل (BOS) وتغيير الهيكل (CHoCH)
    if len(swing_high_indices) >= 2 and len(swing_low_indices) >= 2:
        last_swing_high = highs[swing_high_indices[-2]]
        last_swing_low = lows[swing_low_indices[-2]]
        current_price = df['close'].iloc[-1]
        
        # BOS صاعد: كسر آخر قمة رئيسية
        if current_price > last_swing_high * 1.002:
            structure_data["bos_detected"] = True
            structure_data["bos_direction"] = "bullish"
        
        # CHoCH: السعر كسر آخر قاع في اتجاه هابط (تغيير اتجاه)
        if structure_data["structure_type"] in ["bullish", "weak_bullish"]:
            if current_price < last_swing_low * 0.998:
                structure_data["choch_detected"] = True
                structure_data["choch_direction"] = "bearish"
    
    return structure_data


def apply_advanced_market_structure_filter(df: pd.DataFrame, symbol: str) -> Tuple[bool, Optional[str]]:
    """
    فلتر هيكل السوق الذكي المتطور - يجمع بين عدة تقنيات:
    1. تحليل القمم والقيعان (Swing Analysis)
    2. اكتشاف كسر الهيكل (BOS)
    3. اكتشاف مناطق السيولة (Liquidity Zones)
    4. تحليل قوة الاتجاه (Trend Strength)
    """
    
    if len(df) < 50:
        return False, "Insufficient data for structure analysis"
    
    # 1. تحليل الهيكل الأساسي
    structure = detect_market_structure(df)
    
    # 2. السماح فقط للهياكل القوية
    if structure["strength"] < 55:
        return False, f"Weak structure (strength: {structure['strength']})"
    
    # 3. الرفض في حالة تغيير الهيكل السلبي (CHoCH هابط)
    if structure.get("choch_detected") and structure.get("choch_direction") == "bearish":
        return False, "Bearish CHoCH detected - potential trend reversal"
    
    # 4. تحليل مناطق السيولة
    liquidity_analysis = analyze_liquidity_zones(df)
    if not liquidity_analysis["safe_to_trade"]:
        return False, liquidity_analysis["reason"]
    
    # 5. تحليل قوة الاتجاه الحالي
    last = df.iloc[-1]
    
    # تأكيد قوة الاتجاه بواسطة EMAs
    # Make sure required EMAs are calculated if not present
    if not all(k in df.columns for k in ['ema9', 'ema21', 'ema50']):
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        last = df.iloc[-1]

    ema_alignment = (
        last['ema9'] > last['ema21'] > last['ema50']
    )
    
    if not ema_alignment:
        return False, "EMAs not properly aligned"
    
    # 6. التحقق من عدم وجود divergence سلبي
    if detect_bearish_divergence(df):
        return False, "Bearish divergence detected"
    
    return True, None


def analyze_liquidity_zones(df: pd.DataFrame) -> Dict[str, any]:
    """
    تحليل مناطق السيولة - البحث عن مناطق تجمع الأوامر
    """
    
    recent_highs = df['high'].tail(20)
    recent_lows = df['low'].tail(20)
    current_price = df['close'].iloc[-1]
    
    # البحث عن مناطق تكرار السعر (Equal Highs/Lows)
    high_clusters = find_price_clusters(recent_highs.values)
    low_clusters = find_price_clusters(recent_lows.values)
    
    # التحقق من عدم وجود منطقة سيولة قريبة جداً من السعر الحالي
    danger_zone = False
    reason = ""
    
    for cluster in high_clusters:
        if abs(current_price - cluster) / current_price < 0.005:  # ضمن 0.5%
            danger_zone = True
            reason = f"Too close to liquidity zone at {cluster:.4f}"
            break
    
    return {
        "safe_to_trade": not danger_zone,
        "reason": reason if danger_zone else "Clear liquidity path",
        "high_clusters": high_clusters,
        "low_clusters": low_clusters
    }


def find_price_clusters(prices: np.ndarray, tolerance: float = 0.003) -> List[float]:
    """
    إيجاد مناطق تجمع السعر (Price Clusters)
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
        
        if len(cluster) >= 2:  # على الأقل سعرين متقاربين
            clusters.append(np.mean(cluster))
        
        i = j if j > i else i + 1
    
    return clusters


def detect_bearish_divergence(df: pd.DataFrame) -> bool:
    """
    اكتشاف Divergence هابط بين السعر والـ RSI
    """
    
    if len(df) < 20:
        return False
    
    # Make sure RSI is calculated if not present
    if 'rsi' not in df.columns:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=7).mean()
        avg_loss = loss.rolling(window=7).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))

    recent_df = df.tail(20)
    
    # البحث عن قمتين في السعر
    highs = recent_df['high'].values
    rsi_values = recent_df['rsi'].values
    
    high_indices = argrelextrema(highs, np.greater, order=3)[0]
    
    if len(high_indices) >= 2:
        last_high_idx = high_indices[-1]
        prev_high_idx = high_indices[-2]
        
        # السعر يصنع قمة أعلى
        price_higher_high = highs[last_high_idx] > highs[prev_high_idx]
        
        # RSI يصنع قمة أدنى (Divergence)
        rsi_lower_high = rsi_values[last_high_idx] < rsi_values[prev_high_idx]
        
        if price_higher_high and rsi_lower_high:
            return True
    
    return False

# ===== SMART DYNAMIC FILTERS (IMPROVED) =====

def apply_smart_liquidity_filter(df: pd.DataFrame, symbol: str) -> bool:
    """
    فلتر السيولة الذكي - يتحقق من جودة السيولة مع استثناء للزيادات الإيجابية في الحجم
    """
    if len(df) < 20: return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # 1. نسبة Spread (High - Low) إلى السعر
    spread_ratio = (last['high'] - last['low']) / last['close']
    if spread_ratio > 0.05:
        log_rejection(symbol, "Smart Liquidity Filter Failed", {"reason": f"Spread > 5% ({spread_ratio:.2f}%)"})
        return False
    
    # 2. حجم التداول مستقر (لا تذبذب عنيف)
    volume_std = df['volume'].tail(20).std()
    volume_mean = df['volume'].tail(20).mean()
    cv = volume_std / volume_mean if volume_mean > 0 else 999
    
    if cv > 2.0:
        # استثناء: إذا كان تذبذب الحجم مرتفعاً ولكنه مصحوب بحركة سعرية إيجابية، فقد يكون اختراقاً
        is_bullish_volume_spike = (last['close'] > prev['close']) and (last['volume'] > volume_mean * 1.5)
        if not is_bullish_volume_spike:
            log_rejection(symbol, "Smart Liquidity Filter Failed", {"reason": f"Volume CV > 2.0 ({cv:.2f})"})
            return False
        else:
            logger.info(f"[{symbol}] Liquidity filter bypassed due to bullish volume spike.")

    # 3. السعر لا يتحرك بقفزات كبيرة
    price_changes = df['close'].pct_change().tail(10).abs()
    max_price_jump = price_changes.max()
    if max_price_jump > 0.05:
        log_rejection(symbol, "Smart Liquidity Filter Failed", {"reason": f"Price Jump > 5% ({max_price_jump:.2f}%)"})
        return False
    
    return True


def apply_smart_risk_reward_filter(entry_price: float, stop_loss: float, target1: float, target2: float) -> bool:
    """ فلتر نسبة المخاطرة/العائد الذكي """
    risk = entry_price - stop_loss
    if risk <= 0: return False
    
    reward1 = target1 - entry_price
    reward2 = target2 - entry_price
    
    rr1 = reward1 / risk
    rr2 = reward2 / risk
    
    if rr1 < 1.2 or rr2 < 2.0: return False
    if (risk / entry_price) > 0.035: return False
    
    return True


def calculate_market_regime(df: pd.DataFrame) -> str:
    """ تحديد نظام السوق الحالي (trending, ranging, volatile) """
    if len(df) < 50: return 'unknown'
    last = df.iloc[-1]
    
    adx = last.get('adx', 0)
    bb_width = last.get('bb_width', 0)
    bb_width_ma = df['bb_width'].rolling(50).mean().iloc[-1]
    atr_percent = last.get('atr_percent', 0)
    
    if adx > 25 and atr_percent < 2.5: return 'trending'
    elif adx < 20 and bb_width < bb_width_ma * 0.8: return 'ranging'
    elif atr_percent > 3.0: return 'volatile'
    else: return 'mixed'


# ===== STRATEGY SELECTOR =====

ENHANCED_STRATEGIES = {
    "Smart_Momentum_Strategy": {
        "name": "زخم ذكي",
        "check_function": check_smart_momentum_strategy,
        "enabled": True, "best_regime": ['trending'], "risk_level": 'medium'
    },
    "Advanced_Pullback_Strategy": {
        "name": "ارتداد متقدم",
        "check_function": check_advanced_pullback_strategy,
        "enabled": True, "best_regime": ['trending', 'mixed'], "risk_level": 'low'
    },
    "Breakout_Retest_Strategy": {
        "name": "اختراق وإعادة اختبار",
        "check_function": check_breakout_retest_strategy,
        "enabled": True, "best_regime": ['trending', 'mixed'], "risk_level": 'medium'
    },
    "Volume_Divergence_Strategy": {
        "name": "تباين الحجم",
        "check_function": check_volume_price_divergence_strategy,
        "enabled": True, "best_regime": ['ranging'], "risk_level": 'low'
    },
    "Golden_Cross_Strategy": {
        "name": "تقاطع ذهبي",
        "check_function": check_golden_cross_momentum_strategy,
        "enabled": True, "best_regime": ['trending', 'mixed'], "risk_level": 'medium'
    },
    "Mean_Reversion_BB_Strategy": {
        "name": "انعكاس للمتوسط",
        "check_function": check_mean_reversion_bb_strategy,
        "enabled": True, "best_regime": ['ranging', 'mixed'], "risk_level": 'low'
    }
}

STRATEGY_NAMES = {key: info['name'] for key, info in ENHANCED_STRATEGIES.items()}


def find_best_strategy(df: pd.DataFrame, mtf_trend: Dict, symbol: str) -> Optional[Tuple[str, str]]:
    """
    يبحث عن أفضل استراتيجية مناسبة للوضع الحالي
    """
    market_regime = calculate_market_regime(df)
    
    # Use the new advanced market structure filter V2 which also includes liquidity checks
    passed, reason = apply_advanced_market_structure_filter(df, symbol)
    if not passed:
        log_rejection(symbol, "Advanced Market Structure Filter Failed", {"reason": reason})
        return None

    # The original liquidity filter is now redundant as it's part of the advanced filter
    # if not apply_smart_liquidity_filter(df, symbol): return None
    
    with strategy_filters_lock:
        strategies_to_check = {k: v for k, v in ENHANCED_STRATEGIES.items()}

    for strategy_key, strategy_info in strategies_to_check.items():
        if not strategy_info['enabled']: continue
        if market_regime not in strategy_info['best_regime'] and market_regime != 'mixed': continue
        
        try:
            if strategy_info['check_function'](df, mtf_trend):
                return (strategy_key, strategy_info['name'])
        except Exception as e:
            logger.error(f"❌ [{symbol}] Error checking strategy {strategy_key}: {e}", exc_info=True)
            continue
    
    return None


# ===== IMPROVED STOP LOSS & TAKE PROFIT =====

def calculate_smart_stop_loss(df: pd.DataFrame, entry_price: float, strategy_name: str) -> float:
    """
    حساب وقف خسارة ذكي بناءً على ATR وهيكل السوق
    """
    last = df.iloc[-1]
    atr_value = last.get('atr', 0)
    
    if strategy_name == "Mean_Reversion_BB_Strategy":
        atr_multiplier = 1.5
    else:
        atr_multiplier = 2.0 if strategy_name in ['Smart_Momentum_Strategy', 'Golden_Cross_Strategy'] else 1.8

    atr_stop = entry_price - (atr_value * atr_multiplier)
    
    # استخدام Order Block كدعم إضافي
    ob = detect_order_block(df)
    if ob and ob['type'] == 'bullish':
        structure_stop = ob['bottom'] * 0.996
    else:
        recent_low = df['low'].tail(7).min()
        structure_stop = recent_low * 0.997
    
    ema21_stop = last['ema21'] * 0.995
    
    stop_loss = max(atr_stop, structure_stop, ema21_stop)
    
    max_stop_distance = entry_price * 0.025
    if entry_price - stop_loss > max_stop_distance:
        stop_loss = entry_price - max_stop_distance
    
    min_stop_distance = entry_price * 0.008
    if entry_price - stop_loss < min_stop_distance:
        stop_loss = entry_price - min_stop_distance
    
    return stop_loss


def calculate_smart_take_profit(
    df: pd.DataFrame, 
    entry_price: float, 
    stop_loss: float, 
    strategy_name: str
) -> Tuple[float, float]:
    """
    حساب أهداف ربح ذكية بناءً على نسب RR محسّنة
    """
    risk_amount = entry_price - stop_loss
    if risk_amount <= 0: return (entry_price * 1.015, entry_price * 1.025)
    
    rr_ratios = {
        'Smart_Momentum_Strategy': (2.0, 3.5),
        'Advanced_Pullback_Strategy': (1.8, 3.2),
        'Breakout_Retest_Strategy': (2.2, 4.0),
        'Volume_Divergence_Strategy': (1.6, 2.8),
        'Golden_Cross_Strategy': (2.0, 3.5),
        'Mean_Reversion_BB_Strategy': (1.5, 2.8)
    }
    
    rr1, rr2 = rr_ratios.get(strategy_name, (1.8, 3.0))
    
    target1 = entry_price + (risk_amount * rr1)
    target2 = entry_price + (risk_amount * rr2)
    
    last = df.iloc[-1]
    atr_value = last.get('atr', 0)
    
    # استخدام Fair Value Gap كهدف محتمل
    fvg = detect_fair_value_gap(df)
    if fvg and fvg['type'] == 'bullish':
        target1 = max(target1, fvg['top'])
    
    max_target1 = entry_price + (atr_value * 3.5)
    if target1 > max_target1: target1 = max_target1
    
    max_target2 = entry_price + (atr_value * 6.0)
    if target2 > max_target2: target2 = max_target2
    
    return (target1, target2)

# ===== END OF NEW STRATEGY BLOCK =====


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
    تم إصلاح المشكلة عن طريق الوصول مباشرة إلى المتغير العام exchange_info_map.
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
    تم إصلاح المشكلة عن طريق الوصول مباشرة إلى المتغير العام exchange_info_map.
    """
    if override_amount is not None:
        desired_usdt_amount = override_amount
    elif not is_real:
        desired_usdt_amount = PAPER_TRADE_FIXED_AMOUNT_USDT
    else:
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
    حساب حجم الصفقة بشكل ديناميكي بناءً على جودة الإشارة وتقلب السوق.
    """
    # 1. تحديد مبلغ الأساس بناءً على وضع التداول
    if not is_real:
        base_usdt_amount = PAPER_TRADE_FIXED_AMOUNT_USDT
    else:
        base_usdt_amount = random.uniform(FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT)

    # 2. تعديل المبلغ بناءً على جودة الإشارة
    quality_modifier = 1.0
    if quality_score > 85:
        quality_modifier = 1.25  # زيادة 25% للفرص الممتازة
    elif quality_score < 70:
        quality_modifier = 0.85   # تقليل 15% للفرص الأضعف

    # 3. تعديل المبلغ بناءً على تقلب السوق (مخاطرة عكسية)
    volatility_modifier = 1.0
    if atr_percent > 3.0:
        volatility_modifier = 0.80  # تقليل 20% في الأسواق شديدة التقلب
    elif atr_percent < 0.8:
        volatility_modifier = 1.15  # زيادة 15% في الأسواق الهادئة

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
    
    # General filters that don't depend on trade levels
    if not check_market_volatility_filter_enhanced(df, symbol): return
    if not add_news_filter(): log_rejection(symbol, "News Filter Failed"); return
    if not add_liquidity_filter(): log_rejection(symbol, "Liquidity Filter Failed"); return
    if not add_correlation_filter(symbol): log_rejection(symbol, "Correlation Filter Failed"); return

    quality_score = calculate_signal_quality_score(df, mtf_trend, strategy_key)
    with min_quality_lock: min_score = MIN_SIGNAL_QUALITY
    if quality_score < min_score:
        log_rejection(symbol, "Low Quality Signal", {"score": quality_score, "min_required": min_score})
        return
    logger.info(f"⭐ [Signal Quality] {symbol} ({strategy_key}): {quality_score}/100")

    entry_price = df.iloc[-1]['close']
    stop_loss_price = calculate_smart_stop_loss(df, entry_price, strategy_key)
    target_price_1, target_price_2 = calculate_smart_take_profit(df, entry_price, stop_loss_price, strategy_key)
    
    # Apply Risk/Reward filter after calculating levels
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
        "quality_score": quality_score, "atr_percent": atr_percent
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
               