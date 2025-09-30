# ملف c4_5min_v34_1_0.py - نسخة V34.1.1 (تحسينات الاستراتيجيات والفلاتر)
# --- وصف التعديلات:
# 1. [إصلاح فلتر فيبوناتشي] تم توسيع نطاق قبول تصحيح فيبوناتشي بشكل كبير ليكون أكثر مرونة ويقلل من حالات الرفض.
# 2. [إصلاح فلتر التقلب] تم تخفيض الحد الأدنى لتقلب السوق (ATR) للسماح للبوت بالعثور على صفقات في ظروف السوق الأقل تقلباً.
# 3. [تحسينات طفيفة] تم إجراء تعديلات طفيفة على فلاتر ADX لتكون أقل صرامة.
# 4. [الحفاظ على الهيكل] تم الحفاظ على جميع مكونات البوت الأساسية وهيكله العام.
# 5. [جديد] إضافة شرط التحقق من اتجاهين صاعدين لقبول التوصيات.
# 6. [جديد] تحسين شروط الدخول لجميع الاستراتيجيات مع التركيز على قوة الاتجاه وحجم التداول.
# 7. [إصلاح] إصلاح خطأ SyntaxError في دالة api_settings

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
logger = logging.getLogger('CryptoBotV34.1.1_5min')

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
    "Overall trend not bullish in 2 timeframes": "الاتجاه العام ليس صاعدًا في فريمين",

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
    "Range Reversal: No reversal pattern detected": "انعكاس نطاقي: لم يتم اكتشاف نمط انعكاس"
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

def send_trade_open_notification(symbol: str, strategy_name: str, entry_price: float, stop_loss: float,
                                target1: float, target2: float, quantity: float, is_real: bool,
                                quality_score: int, atr_percent: float, notional_value: float):
    trade_type = "حقيقية" if is_real else "ورقية"
    emoji = "🔥" if is_real else "📊"
    
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
    global MIN_SIGNAL_QUALITY, FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT, MAX_OPEN_TRADES
    global USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY
    global USE_PULLBACK_STRATEGY, USE_MOMENTUM_VOLATILITY_STRATEGY, USE_ELLIOTT_WAVE_STRATEGY, USE_RANGE_REVERSAL_STRATEGY
    global paper_trading_mode
    
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

# --- الفلاتر الديناميكية ونظام السوق ---
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

def check_overall_trend_bullish(mtf_trend: Dict) -> bool:
    """
    التحقق مما إذا كان الاتجاه العام صاعدًا في فريمين على الأقل
    """
    bullish_count = sum(1 for trend in mtf_trend.values() if trend == 'bullish')
    return bullish_count >= 2

def analyze_trend_multiple_timeframes(symbol: str) -> Dict:
    """
    تحليل الاتجاه في فريمات زمنية متعددة
    """
    trend_results = {}
    
    for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
        try:
            df = fetch_historical_data(symbol, tf, 3)  # 3 أيام من البيانات
            if df is None or len(df) < 50:
                trend_results[tf] = "unknown"
                continue
                
            df = calculate_all_features(df)
            last = df.iloc[-1]
            
            # تحديد الاتجاه بناءً على المتوسطات المتحركة و ADX
            if last['ema9'] > last['ema21'] > last['ema50'] and last['adx'] > 18:
                trend_results[tf] = "bullish"
            elif last['ema9'] < last['ema21'] < last['ema50'] and last['adx'] > 18:
                trend_results[tf] = "bearish"
            else:
                trend_results[tf] = "sideways"
                
        except Exception as e:
            logger.error(f"Error analyzing trend for {symbol} on {tf}: {e}")
            trend_results[tf] = "unknown"
    
    # تحديث حالة السوق
    with market_state_lock:
        current_market_state["trend_details_by_tf"] = {
            tf: {"trend": trend, "adx": 0, "rsi": 0} 
            for tf, trend in trend_results.items()
        }
    
    return trend_results

def calculate_signal_quality(df: pd.DataFrame, strategy_name: str, mtf_trend: Dict) -> int:
    """
    حساب جودة الإشارة بناءً على عوامل متعددة
    """
    quality_score = 50  # نقطة البداية
    
    last = df.iloc[-1]
    
    # إضافة نقاط لقوة الاتجاه
    if mtf_trend.get('5m') == 'bullish': quality_score += 10
    if mtf_trend.get('15m') == 'bullish': quality_score += 10
    if mtf_trend.get('1h') == 'bullish': quality_score += 5
    
    # إضافة نقاط لقوة ADX
    if last['adx'] > 25: quality_score += 10
    elif last['adx'] > 20: quality_score += 5
    
    # إضافة نقاط لحجم التداول
    volume_ma = df['volume'].tail(20).mean()
    if last['volume'] > volume_ma * 1.5: quality_score += 10
    elif last['volume'] > volume_ma * 1.2: quality_score += 5
    
    # إضافة نقاط لمؤشر RSI
    if 40 < last['rsi'] < 60: quality_score += 5
    elif 30 < last['rsi'] < 70: quality_score += 2
    
    # خصم نقاط للتقلب الزائد
    if last['atr_percent'] > 2.5: quality_score -= 5
    
    # ضمان النتيجة في النطاق 0-100
    return max(0, min(100, quality_score))

def check_bb_stoch_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    
    bb_width = df['bb_width']
    dynamic_bb_threshold = bb_width.rolling(20).mean() * 1.2

    stoch_threshold = 23 if atr_percent > 1.5 else 18 # Adjusted for 5m
    
    volume_ma = df['volume'].rolling(20).mean()
    volume_multiplier = 1.0 + (atr_percent / 100)
    
    return {
        'bb_width_ok': bb_width.iloc[-1] > dynamic_bb_threshold.iloc[-1],
        'stoch_ok': last_row['stoch_k'] > stoch_threshold,
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * volume_multiplier
    }

def check_macd_ema_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    
    # --- تحسين: تم تخفيض الحد الأدنى لـ ADX ليكون أقل صرامة ---
    default_adx_thresh = 18 if atr_percent > 1.5 else 15
    adx_threshold = default_adx_thresh
    
    volume_ma = df['volume'].rolling(20).mean()
    volatility_adjusted_volume = volume_ma * (1 + atr_percent / 75)
    
    macd_momentum = df['macd_hist'].diff()
    momentum_threshold = macd_momentum.rolling(10).std() * 0.3
    
    return {
        'adx_ok': last_row['adx'] > adx_threshold,
        'volume_ok': last_row['volume'] > volatility_adjusted_volume.iloc[-1],
        'momentum_ok': macd_momentum.iloc[-1] > momentum_threshold.iloc[-1],
    }

def check_ema_rsi_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    adx = last_row.get('adx', 0)
    
    if adx > 25:
        rsi_lower, rsi_upper = 42, 78
    else:
        rsi_lower, rsi_upper = 48, 72
    
    ema_spread = (df['ema9'] - df['ema21']) / df['ema21'].replace(0, 1e-9)
    dynamic_ema_threshold = ema_spread.rolling(20).std() * 1.7
    
    volume_ma = df['volume'].rolling(20).mean()
    trend_strength_multiplier = 1 + (adx / 100)
    
    return {
        'rsi_ok': rsi_lower < last_row['rsi'] < rsi_upper,
        'ema_ok': ema_spread.iloc[-1] > dynamic_ema_threshold.iloc[-1],
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * trend_strength_multiplier,
    }

def check_pullback_dynamic_filters(df: pd.DataFrame, mtf_trend: Dict) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    
    pullback_depth = 0.035 if atr_percent > 2.0 else 0.02
    if mtf_trend.get('5m') == 'bullish' and mtf_trend.get('15m') == 'bullish':
        pullback_depth *= 1.2
    
    recent_low = df['low'].tail(5).min()
    recovery_threshold = recent_low * (1 + (pullback_depth * 0.9))
    
    volume_ma = df['volume'].rolling(20).mean()
    recovery_volume_multiplier = 1.1 + (atr_percent / 100)
    
    return {
        'recovery_ok': last_row['close'] > recovery_threshold,
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * recovery_volume_multiplier,
    }

def check_momentum_volatility_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = df['atr_percent']
    volatility_ma = atr_percent.rolling(20).mean()
    volatility_std = atr_percent.rolling(20).std()
    
    dynamic_vol_min = volatility_ma.iloc[-1] - (volatility_std.iloc[-1] * 1.5)
    dynamic_vol_max = volatility_ma.iloc[-1] + (volatility_std.iloc[-1] * 1.5)
    
    is_momentum_ok = (last_row['rsi'] > 51) and (df['macd_hist'].iloc[-1] > df['macd_hist'].iloc[-2])

    adx_ma = df['adx'].rolling(20).mean()
    # --- تحسين: تم تخفيض مضاعف ADX ليكون أقل صرامة ---
    dynamic_adx_threshold = adx_ma.iloc[-1] * 0.80
    
    return {
        'volatility_ok': dynamic_vol_min <= atr_percent.iloc[-1] <= dynamic_vol_max,
        'momentum_ok': is_momentum_ok,
        'adx_ok': last_row['adx'] > dynamic_adx_threshold,
    }

def check_elliott_wave_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    
    # --- إصلاح رئيسي: تم توسيع نطاق قبول فيبوناتشي بشكل كبير لزيادة فرص الدخول ---
    fib_min, fib_max = 0.15, 0.95 # نطاق واسع ومرن جداً
    
    volume_ma = df['volume'].rolling(20).mean()
    wave_volume_multiplier = 1.3 + (atr_percent / 50)
    
    macd_momentum = df['macd_hist'].rolling(5).mean()
    momentum_threshold = macd_momentum.rolling(20).std() * 0.3
    
    return {
        'fibonacci_ok': fib_min <= get_wave_retracement(df) <= fib_max,
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * wave_volume_multiplier,
        'momentum_ok': macd_momentum.iloc[-1] > momentum_threshold.iloc[-1],
    }

def check_range_reversal_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    adx = last_row.get('adx', 99)
    adx_ok = adx < 23
    
    rsi = last_row.get('rsi', 50)
    atr_percent = last_row.get('atr_percent', 0)
    rsi_threshold = 35 if atr_percent < 2.5 else 40
    rsi_ok = rsi < rsi_threshold

    return {'adx_ok': adx_ok, 'rsi_ok': rsi_ok}

# --- General Filters ---
def add_news_filter() -> bool:
    news_hours = [(12, 30), (14, 0), (18, 30)]
    now = datetime.now(timezone.utc)
    for hour, minute in news_hours:
        if now.hour == hour and abs(now.minute - minute) <= 15: # Reduced window for 5m
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
    # --- إصلاح: تم تخفيض الحد الأدنى للسماح بالتداول في الأسواق الهادئة ---
    ATR_PERCENT_MIN = 0.35
    ATR_PERCENT_MAX = 3.2
    
    if not (ATR_PERCENT_MIN <= last_atr_percent <= ATR_PERCENT_MAX):
        log_rejection(symbol, "Market Volatility Filter Failed", {
            "atr": f"{last_atr_percent:.2f}%",
            "range": f"({ATR_PERCENT_MIN:.2f}-{ATR_PERCENT_MAX:.2f})%"
        })
        return False
    
    return True

# --- Dynamic Stop Loss & Take Profit ---
def calculate_dynamic_stop_loss(df: pd.DataFrame, entry_price: float, strategy_name: str) -> float:
    last = df.iloc[-1]
    atr_value = last.get('atr', 0)
    
    if strategy_name == "BB_Stoch_Strategy":
        recent_low = df['low'].tail(3).min()
        stop_loss = min(recent_low * 0.995, entry_price - (atr_value * 1.5))
    elif strategy_name == "MACD_EMA_Strategy":
        stop_loss = min(last['ema21'], entry_price - (atr_value * 2.0))
    elif strategy_name == "EMA_RSI_Strategy":
        stop_loss = min(last['ema21'], entry_price - (atr_value * 1.8))
    elif strategy_name == "Pullback_Strategy":
        recent_low = df['low'].tail(5).min()
        stop_loss = min(recent_low * 0.995, entry_price - (atr_value * 1.5))
    elif strategy_name == "Momentum_Volatility_Strategy":
        stop_loss = min(last['ema21'], entry_price - (atr_value * 2.2))
    elif strategy_name == "Elliott_Wave_Strategy":
        lows = df['low'].values
        try:
            support_idx = argrelextrema(lows, np.less, order=5)[0]
            if len(support_idx) > 0:
                recent_support = lows[support_idx[-1]]
                stop_loss = min(recent_support * 0.995, entry_price - (atr_value * 2.0))
            else:
                stop_loss = min(last['ema21'], entry_price - (atr_value * 2.0))
        except Exception as e:
            logger.error(f"Error calculating stop loss for Elliott Wave: {e}")
            stop_loss = entry_price - (atr_value * 2.0)
    elif strategy_name == "Range_Reversal_Strategy":
        recent_low = df['low'].tail(5).min()
        stop_loss = min(recent_low * 0.99, entry_price - (atr_value * 1.2))
    else:
        stop_loss = entry_price - (atr_value * 2.0)
    
    max_stop_distance = entry_price * 0.05
    if entry_price - stop_loss > max_stop_distance:
        stop_loss = entry_price - max_stop_distance
    
    return stop_loss

def calculate_dynamic_take_profit(df: pd.DataFrame, entry_price: float, stop_loss: float, strategy_name: str) -> tuple:
    risk_amount = entry_price - stop_loss
    if risk_amount <= 0: return (entry_price * 1.015, entry_price * 1.025) # Default for 5m

    # Risk-Reward Ratios adjusted for 5m timeframe (Scalping)
    if strategy_name == "BB_Stoch_Strategy": rr1, rr2 = 1.8, 3.0
    elif strategy_name == "MACD_EMA_Strategy": rr1, rr2 = 1.6, 2.8
    elif strategy_name == "EMA_RSI_Strategy": rr1, rr2 = 1.7, 3.0
    elif strategy_name == "Pullback_Strategy": rr1, rr2 = 1.8, 3.2
    elif strategy_name == "Momentum_Volatility_Strategy": rr1, rr2 = 1.5, 2.5
    elif strategy_name == "Elliott_Wave_Strategy": rr1, rr2 = 2.0, 3.5
    elif strategy_name == "Range_Reversal_Strategy":
        middle_band = df.iloc[-1].get('bb_middle', entry_price * 1.015)
        upper_band = df.iloc[-1].get('bb_upper', entry_price * 1.03)
        return middle_band, upper_band
    else: rr1, rr2 = 1.6, 2.8
        
    target1 = entry_price + (risk_amount * rr1)
    target2 = entry_price + (risk_amount * rr2)
    
    return target1, target2

# --- استراتيجيات التداول المحسنة ---
def check_ema_rsi_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 200: 
        log_rejection(symbol_name, "Insufficient Historical Data")
        return False

    # التحقق من شرط الاتجاه العام في فريمين
    if not check_overall_trend_bullish(mtf_trend):
        log_rejection(symbol_name, "Overall trend not bullish in 2 timeframes")
        return False

    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # تحسين شروط الاتجاه الصاعد
    if not (last['ema50'] > last['ema200'] and 
            last['ema9'] > last['ema21'] and
            last['close'] > last['ema9'] and
            prev['close'] <= prev['ema9']):  # إضافة شرط كسر المتوسط المتحرك
        log_rejection(symbol_name, "EMA_RSI: Bearish long-term trend")
        return False
        
    filters = check_ema_rsi_dynamic_filters(df)
    if not filters['rsi_ok']: 
        log_rejection(symbol_name, "DYN_RSI_OOR")
        return False
    if not filters['ema_ok']: 
        log_rejection(symbol_name, "DYN_EMA_SPREAD_LOW")
        return False
    if not filters['volume_ok']: 
        log_rejection(symbol_name, "DYN_VOLUME_LOW")
        return False
        
    return True

def check_bb_stoch_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 50: 
        log_rejection(symbol_name, "Insufficient Historical Data")
        return False
    
    # التحقق من شرط الاتجاه العام في فريمين
    if not check_overall_trend_bullish(mtf_trend):
        log_rejection(symbol_name, "Overall trend not bullish in 2 timeframes")
        return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # تحسين شروط الدخول
    if not (last['close'] > last['ema50'] and 
            (df['low'].tail(3) <= df['bb_lower'].tail(3)).any() and 
            last['close'] > last['bb_lower'] and
            last['volume'] > df['volume'].tail(20).mean() * 1.2):  # إضافة شرط الحجم
        log_rejection(symbol_name, "BB: Price below EMA50 (bearish trend)")
        return False
    
    # تحسين شرط ستوكاستيك
    if not ((prev['stoch_k'] < 30) and (last['stoch_k'] > prev['stoch_k']) and last['stoch_k'] < 70):
        log_rejection(symbol_name, "DYN_STOCH_LOW")
        return False

    filters = check_bb_stoch_dynamic_filters(df)
    if not filters['bb_width_ok']: 
        log_rejection(symbol_name, "DYN_BB_WIDTH_LOW")
        return False
    if not filters['volume_ok']: 
        log_rejection(symbol_name, "DYN_VOLUME_LOW")
        return False

    return True

def check_macd_ema_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 200: 
        log_rejection(symbol_name, "Insufficient Historical Data")
        return False
    
    # التحقق من شرط الاتجاه العام في فريمين
    if not check_overall_trend_bullish(mtf_trend):
        log_rejection(symbol_name, "Overall trend not bullish in 2 timeframes")
        return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # تحسين شروط التقاطع
    if not (last['ema9'] > last['ema21'] and 
            prev['ema9'] <= prev['ema21'] and
            last['close'] > last['ema50'] and
            last['macd'] > last['macd_signal'] and
            prev['macd'] <= prev['macd_signal']):
        log_rejection(symbol_name, "MACD: Strongly bearish trend")
        return False
    
    # التحقق من قوة الاتجاه
    if last['adx'] < 18:  # تخفيض الحد الأدنى لـ ADX
        log_rejection(symbol_name, "DYN_ADX_LOW")
        return False

    filters = check_macd_ema_dynamic_filters(df)
    if not filters['volume_ok']: 
        log_rejection(symbol_name, "DYN_VOLUME_LOW")
        return False
    if not filters['momentum_ok']: 
        log_rejection(symbol_name, "DYN_MACD_MOMENTUM_LOW")
        return False

    return True

def check_pullback_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 100: 
        log_rejection(symbol_name, "Insufficient Historical Data")
        return False
    
    # التحقق من شرط الاتجاه العام في فريمين
    if not check_overall_trend_bullish(mtf_trend):
        log_rejection(symbol_name, "Overall trend not bullish in 2 timeframes")
        return False
    
    last = df.iloc[-1]
    
    # تحسين شروط الاتجاه الصاعد
    if not (last['ema9'] > last['ema21'] > last['ema50'] and
            last['adx'] > 20 and  # تخفيض الحد الأدنى لـ ADX
            mtf_trend.get('15m') == 'bullish'):  # التأكد من اتجاه 15m صاعد
        log_rejection(symbol_name, "Pullback: Trend is not strongly bullish")
        return False
    
    # تحسين شروط الارتداد
    recent_high = df['high'].tail(10).max()
    pullback_percentage = (recent_high - last['close']) / recent_high * 100
    
    atr_percent = last.get('atr_percent', 0)
    min_pullback = 0.8 if atr_percent > 1.5 else 0.5
    
    if pullback_percentage < min_pullback:
        log_rejection(symbol_name, "DYN_PULLBACK_SHALLOW", {"pullback": f"{pullback_percentage:.2f}%"})
        return False
    
    filters = check_pullback_dynamic_filters(df, mtf_trend)
    if not filters['recovery_ok']: 
        log_rejection(symbol_name, "DYN_RECOVERY_FAIL")
        return False
    if not filters['volume_ok']: 
        log_rejection(symbol_name, "DYN_VOLUME_LOW")
        return False
    
    return True

def check_momentum_volatility_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 100: 
        log_rejection(symbol_name, "Insufficient Historical Data")
        return False
    
    # التحقق من شرط الاتجاه العام في فريمين
    if not check_overall_trend_bullish(mtf_trend):
        log_rejection(symbol_name, "Overall trend not bullish in 2 timeframes")
        return False
    
    last = df.iloc[-1]
    
    # تحسين شروط الزخم
    if not (last['ema9'] > last['ema21'] > last['ema50'] and
            last['rsi'] > 50 and
            last['macd'] > last['macd_signal'] and
            last['macd_hist'] > 0):
        log_rejection(symbol_name, "Momentum: EMAs not in bullish order")
        return False
    
    filters = check_momentum_volatility_dynamic_filters(df)
    if not filters['volatility_ok']: 
        log_rejection(symbol_name, "DYN_VOLATILITY_OOR")
        return False
    if not filters['momentum_ok']: 
        log_rejection(symbol_name, "DYN_MOMENTUM_SCORE_LOW")
        return False
    if not filters['adx_ok']: 
        log_rejection(symbol_name, "DYN_ADX_LOW")
        return False
    
    return True

def check_elliott_wave_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 200: 
        log_rejection(symbol_name, "Insufficient Historical Data")
        return False
    
    # التحقق من شرط الاتجاه العام في فريمين
    if not check_overall_trend_bullish(mtf_trend):
        log_rejection(symbol_name, "Overall trend not bullish in 2 timeframes")
        return False
    
    last = df.iloc[-1]
    
    # تحسين شروط الموجات
    if not (last['ema50'] > last['ema200'] and
            last['close'] > last['ema21'] and
            last['adx'] > 18):  # تخفيض الحد الأدنى لـ ADX
        log_rejection(symbol_name, "Elliott Wave: Strongly bearish trend")
        return False
    
    # تحسين شروط فيبوناتشي
    fib_retracement = get_wave_retracement(df)
    if not (0.236 <= fib_retracement <= 0.786):  # توسيع نطاق فيبوناتشي
        log_rejection(symbol_name, "DYN_FIB_RETRACEMENT_OOR", {"fib": f"{fib_retracement:.3f}"})
        return False
    
    filters = check_elliott_wave_dynamic_filters(df)
    if not filters['volume_ok']: 
        log_rejection(symbol_name, "DYN_VOLUME_LOW")
        return False
    if not filters['momentum_ok']: 
        log_rejection(symbol_name, "DYN_MACD_MOMENTUM_LOW")
        return False
    
    return True

def check_range_reversal_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 50: 
        log_rejection(symbol_name, "Insufficient Historical Data")
        return False
    
    # التحقق من شرط الاتجاه العام في فريمين
    if not check_overall_trend_bullish(mtf_trend):
        log_rejection(symbol_name, "Overall trend not bullish in 2 timeframes")
        return False
    
    last = df.iloc[-1]
    
    # تحسين شروط الانعكاس النطاقي
    if not (last['close'] > last['bb_lower'] and
            last['rsi'] < 40 and
            last['adx'] < 23):  # تخفيض الحد الأقصى لـ ADX
        log_rejection(symbol_name, "Range Reversal: Trend too strong (ADX > 23)")
        return False
    
    # التحقق من نمط الشموع
    prev = df.iloc[-2]
    if not (last['close'] > last['open'] and  # شمعة صاعدة
            prev['close'] < prev['open'] and  # شمعة هابطة
            last['close'] > prev['close']):  # انعكاس
        log_rejection(symbol_name, "Range Reversal: No reversal pattern detected")
        return False
    
    filters = check_range_reversal_dynamic_filters(df)
    if not filters['rsi_ok']: 
        log_rejection(symbol_name, "Range Reversal: RSI not in oversold zone")
        return False
    
    return True

# --- دالة التحقق من جميع الاستراتيجيات ---
def check_all_strategies(df: pd.DataFrame, mtf_trend: Dict) -> Dict[str, bool]:
    """
    التحقق من جميع الاستراتيجيات المفعلة
    """
    df.name = getattr(df, 'name', 'Unknown')  # إضافة اسم العملة للوصول إليه في الاستراتيجيات
    
    results = {}
    
    if USE_BB_STOCH_STRATEGY:
        results['BB_Stoch_Strategy'] = check_bb_stoch_strategy_enhanced(df, mtf_trend)
    
    if USE_MACD_EMA_STRATEGY:
        results['MACD_EMA_Strategy'] = check_macd_ema_strategy_enhanced(df, mtf_trend)
    
    if USE_EMA_RSI_STRATEGY:
        results['EMA_RSI_Strategy'] = check_ema_rsi_strategy_enhanced(df, mtf_trend)
    
    if USE_PULLBACK_STRATEGY:
        results['Pullback_Strategy'] = check_pullback_strategy_enhanced(df, mtf_trend)
    
    if USE_MOMENTUM_VOLATILITY_STRATEGY:
        results['Momentum_Volatility_Strategy'] = check_momentum_volatility_strategy_enhanced(df, mtf_trend)
    
    if USE_ELLIOTT_WAVE_STRATEGY:
        results['Elliott_Wave_Strategy'] = check_elliott_wave_strategy_enhanced(df, mtf_trend)
    
    if USE_RANGE_REVERSAL_STRATEGY:
        results['Range_Reversal_Strategy'] = check_range_reversal_strategy_enhanced(df, mtf_trend)
    
    return results

# --- دوال التداول الأساسية ---
def check_general_filters(symbol: str, df: pd.DataFrame) -> bool:
    """
    التحقق من الفلاتر العامة قبل تقييم الاستراتيجيات
    """
    if not check_market_volatility_filter_enhanced(df, symbol):
        return False
    
    if not add_news_filter():
        log_rejection(symbol, "News Filter Failed")
        return False
    
    if not add_liquidity_filter():
        log_rejection(symbol, "Liquidity Filter Failed")
        return False
    
    if not add_correlation_filter(symbol):
        log_rejection(symbol, "Correlation Filter Failed")
        return False
    
    return True

def generate_signal_for_symbol(symbol: str) -> Optional[Dict]:
    """
    توليد إشارة تداول لعملة معينة
    """
    try:
        # التحقق من الفلاتر العامة أولاً
        df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
        if df is None or len(df) < 50:
            log_rejection(symbol, "Insufficient Historical Data")
            return None
        
        df = calculate_all_features(df)
        df.name = symbol  # إضافة اسم العملة للوصول إليه في الاستراتيجيات
        
        if not check_general_filters(symbol, df):
            return None
        
        # تحليل الاتجاه في فريمات متعددة
        mtf_trend = analyze_trend_multiple_timeframes(symbol)
        
        # التحقق من جميع الاستراتيجيات
        strategy_results = check_all_strategies(df, mtf_trend)
        
        # البحث عن استراتيجية ناجحة
        for strategy_name, is_valid in strategy_results.items():
            if is_valid:
                # حساب جودة الإشارة
                quality_score = calculate_signal_quality(df, strategy_name, mtf_trend)
                
                if quality_score >= MIN_SIGNAL_QUALITY:
                    # حساب نقاط الدخول والخروج
                    entry_price = df.iloc[-1]['close']
                    stop_loss = calculate_dynamic_stop_loss(df, entry_price, strategy_name)
                    target1, target2 = calculate_dynamic_take_profit(df, entry_price, stop_loss, strategy_name)
                    
                    return {
                        'symbol': symbol,
                        'strategy': strategy_name,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'target1': target1,
                        'target2': target2,
                        'quality_score': quality_score,
                        'atr_percent': df.iloc[-1]['atr_percent'],
                        'mtf_trend': mtf_trend,
                        'signal_details': {
                            'rsi': df.iloc[-1]['rsi'],
                            'adx': df.iloc[-1]['adx'],
                            'volume': df.iloc[-1]['volume'],
                            'bb_width': df.iloc[-1]['bb_width'],
                            'macd': df.iloc[-1]['macd'],
                            'macd_signal': df.iloc[-1]['macd_signal'],
                            'ema9': df.iloc[-1]['ema9'],
                            'ema21': df.iloc[-1]['ema21'],
                            'ema50': df.iloc[-1]['ema50']
                        }
                    }
        
        return None
        
    except Exception as e:
        logger.error(f"❌ [Signal Generation] Error generating signal for {symbol}: {e}", exc_info=True)
        return None

# --- دوال إدارة الصفقات ---
def calculate_position_size(symbol: str, entry_price: float, stop_loss: float) -> Optional[float]:
    """
    حساب حجم الصفقة بناءً على إدارة المخاطر
    """
    try:
        risk_percent = 0.02  # 2% مخاطرة للصفقة الواحدة
        
        with trading_mode_lock:
            is_paper = paper_trading_mode
        
        if is_paper:
            trade_amount = PAPER_TRADE_FIXED_AMOUNT_USDT
        else:
            with trade_amount_lock:
                trade_amount = random.uniform(FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT)
        
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            return None
        
        position_size = trade_amount / risk_per_share
        
        # تعديل الحجم ليتوافق مع قواعد المنصة
        symbol_info = exchange_info_map.get(symbol, {})
        lot_size_filter = symbol_info.get('filters', [{}])[1] if symbol_info.get('filters') else {}
        step_size = float(lot_size_filter.get('stepSize', 0.00000001))
        
        position_size = int(position_size / step_size) * step_size
        
        # التحقق من الحد الأدنى للصفقة
        min_notional = float(symbol_info.get('filters', [{}])[5].get('minNotional', 10)) if symbol_info.get('filters') else 10
        if position_size * entry_price < min_notional:
            log_rejection(symbol, "MinNotional Filter Failed", {
                "notional": f"{position_size * entry_price:.2f}",
                "min": f"{min_notional:.2f}"
            })
            return None
        
        return position_size
        
    except Exception as e:
        logger.error(f"❌ [Position Size] Error calculating position size for {symbol}: {e}")
        return None

def execute_trade(signal: Dict) -> bool:
    """
    تنفيذ صفقة جديدة
    """
    try:
        symbol = signal['symbol']
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        
        # حساب حجم الصفقة
        quantity = calculate_position_size(symbol, entry_price, stop_loss)
        if quantity is None:
            return False
        
        # حساب قيمة الصفقة
        notional_value = quantity * entry_price
        
        with trading_mode_lock:
            is_paper = paper_trading_mode
        
        # حفظ الإشارة في قاعدة البيانات
        if not check_db_connection() or not conn:
            return False
        
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (
                    symbol, entry_price, stop_loss, target_price_1, target_price_2,
                    strategy_name, signal_details, is_real_trade, quantity, initial_quantity
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                symbol, entry_price, stop_loss, signal['target1'], signal['target2'],
                signal['strategy'], json.dumps(signal['signal_details']), not is_paper,
                quantity, quantity
            ))
            
            signal_id = cur.fetchone()['id']
            conn.commit()
            
            # تحديث الكاش
            with signal_cache_lock:
                open_signals_cache[symbol] = {
                    'id': signal_id,
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target1': signal['target1'],
                    'target2': signal['target2'],
                    'strategy': signal['strategy'],
                    'quantity': quantity,
                    'is_real_trade': not is_paper,
                    'status': 'open'
                }
            
            # إرسال إشعار
            send_trade_open_notification(
                symbol, signal['strategy'], entry_price, stop_loss,
                signal['target1'], signal['target2'], quantity, not is_paper,
                signal['quality_score'], signal['atr_percent'], notional_value
            )
            
            log_and_notify('info', f"✅ تم فتح صفقة جديدة على {symbol} باستخدام استراتيجية {signal['strategy']}", 'trade_open')
            
            return True
            
    except Exception as e:
        logger.error(f"❌ [Trade Execution] Error executing trade for {signal['symbol']}: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

# --- حلقة التداول الرئيسية ---
def trading_loop():
    """
    الحلقة الرئيسية لتوليد وتنفيذ الصفقات
    """
    logger.info("🚀 Starting trading loop...")
    
    while True:
        try:
            with trading_status_lock:
                if not is_trading_enabled:
                    time.sleep(10)
                    continue
            
            # التحقق من عدد الصفقات المفتوحة
            with signal_cache_lock:
                open_trades_count = len(open_signals_cache)
            
            if open_trades_count >= MAX_OPEN_TRADES:
                logger.info(f"📊 Maximum open trades reached ({MAX_OPEN_TRADES}). Waiting...")
                time.sleep(30)
                continue
            
            # المسح عبر جميع العملات
            for symbol in validated_symbols_to_scan:
                try:
                    # التحقق من وجود صفقة مفتوحة على نفس العملة
                    with signal_cache_lock:
                        if symbol in open_signals_cache:
                            continue
                    
                    # التحقق من فترة التهدئة
                    with cooldowns_lock:
                        if symbol in cooldowns_by_symbol:
                            if datetime.now(timezone.utc) < cooldowns_by_symbol[symbol]:
                                continue
                            else:
                                del cooldowns_by_symbol[symbol]
                    
                    # توليد إشارة
                    signal = generate_signal_for_symbol(symbol)
                    
                    if signal:
                        # تنفيذ الصفقة
                        if execute_trade(signal):
                            logger.info(f"✅ Successfully executed trade for {symbol}")
                        else:
                            logger.warning(f"⚠️ Failed to execute trade for {symbol}")
                    
                    # تأخير بين كل عملة
                    time.sleep(API_REQUEST_DELAY)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing {symbol}: {e}")
                    continue
            
            # انتظار قبل الدورة التالية
            time.sleep(60)
            
        except Exception as e:
            logger.error(f"❌ Error in trading loop: {e}", exc_info=True)
            time.sleep(30)

# --- دوال واجهة الويب ---
@app.route('/')
def index():
    return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crypto Bot Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .card { background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .header { text-align: center; color: #333; }
                .status { display: flex; justify-content: space-around; margin: 20px 0; }
                .status-item { text-align: center; }
                .status-value { font-size: 24px; font-weight: bold; }
                .notifications { max-height: 300px; overflow-y: auto; }
                .notification { padding: 10px; margin: 5px 0; border-left: 4px solid #007bff; background: #f8f9fa; }
                .rejection { border-left-color: #dc3545; }
                .controls { display: flex; gap: 10px; justify-content: center; margin: 20px 0; }
                .btn { padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
                .btn-primary { background: #007bff; color: white; }
                .btn-danger { background: #dc3545; color: white; }
                .btn-success { background: #28a745; color: white; }
                .market-state { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
                .trend-item { padding: 10px; border-radius: 4px; text-align: center; }
                .trend-bullish { background: #d4edda; color: #155724; }
                .trend-bearish { background: #f8d7da; color: #721c24; }
                .trend-sideways { background: #fff3cd; color: #856404; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="card">
                    <h1 class="header">🤖 Crypto Bot Dashboard</h1>
                    <div class="status">
                        <div class="status-item">
                            <div>Trading Status</div>
                            <div class="status-value" id="trading-status">Loading...</div>
                        </div>
                        <div class="status-item">
                            <div>Trading Mode</div>
                            <div class="status-value" id="trading-mode">Loading...</div>
                        </div>
                        <div class="status-item">
                            <div>USDT Balance</div>
                            <div class="status-value" id="usdt-balance">Loading...</div>
                        </div>
                        <div class="status-item">
                            <div>Open Trades</div>
                            <div class="status-value" id="open-trades">Loading...</div>
                        </div>
                    </div>
                    <div class="controls">
                        <button class="btn btn-primary" onclick="toggleTrading()">Toggle Trading</button>
                        <button class="btn btn-success" onclick="toggleMode()">Toggle Mode</button>
                        <button class="btn btn-danger" onclick="closeAllTrades()">Close All Trades</button>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Market State</h2>
                    <div class="market-state" id="market-state">
                        Loading...
                    </div>
                </div>
                
                <div class="card">
                    <h2>Notifications</h2>
                    <div class="notifications" id="notifications">
                        Loading...
                    </div>
                </div>
                
                <div class="card">
                    <h2>Rejection Logs</h2>
                    <div class="notifications" id="rejections">
                        Loading...
                    </div>
                </div>
            </div>
            
            <script>
                const ws = new WebSocket(`ws://${window.location.host}/ws`);
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'dashboard_update') {
                        updateDashboard(data.payload);
                    } else if (data.type === 'new_notification') {
                        addNotification(data.payload);
                    } else if (data.type === 'new_rejection') {
                        addRejection(data.payload);
                    } else if (data.type === 'price_update') {
                        // Update prices if needed
                    }
                };
                
                function updateDashboard(data) {
                    document.getElementById('trading-status').textContent = data.trading_enabled ? '🟢 Active' : '🔴 Inactive';
                    document.getElementById('trading-mode').textContent = data.paper_trading_mode ? '📊 Paper' : '🔥 Real';
                    document.getElementById('usdt-balance').textContent = `$${data.usdt_balance.toFixed(2)}`;
                    document.getElementById('open-trades').textContent = Object.keys(data.open_signals || {}).length;
                    
                    // Update market state
                    const marketState = document.getElementById('market-state');
                    marketState.innerHTML = '';
                    
                    if (data.market_state && data.market_state.trend_details_by_tf) {
                        for (const [tf, details] of Object.entries(data.market_state.trend_details_by_tf)) {
                            const div = document.createElement('div');
                            div.className = `trend-item trend-${details.trend}`;
                            div.innerHTML = `
                                <strong>${tf}</strong><br>
                                ${details.trend.toUpperCase()}<br>
                                ADX: ${details.adx.toFixed(1)}<br>
                                RSI: ${details.rsi.toFixed(1)}
                            `;
                            marketState.appendChild(div);
                        }
                    }
                }
                
                function addNotification(notification) {
                    const container = document.getElementById('notifications');
                    const div = document.createElement('div');
                    div.className = 'notification';
                    div.innerHTML = `
                        <strong>${new Date(notification.timestamp).toLocaleString()}</strong><br>
                        ${notification.message}
                    `;
                    container.insertBefore(div, container.firstChild);
                    
                    // Keep only last 20 notifications
                    while (container.children.length > 20) {
                        container.removeChild(container.lastChild);
                    }
                }
                
                function addRejection(rejection) {
                    const container = document.getElementById('rejections');
                    const div = document.createElement('div');
                    div.className = 'notification rejection';
                    div.innerHTML = `
                        <strong>${new Date(rejection.timestamp).toLocaleString()}</strong><br>
                        ${rejection.symbol}: ${rejection.reason}
                    `;
                    container.insertBefore(div, container.firstChild);
                    
                    // Keep only last 30 rejections
                    while (container.children.length > 30) {
                        container.removeChild(container.lastChild);
                    }
                }
                
                function toggleTrading() {
                    fetch('/api/toggle_trading', {method: 'POST'})
                        .then(response => response.json())
                        .then(data => console.log(data));
                }
                
                function toggleMode() {
                    fetch('/api/toggle_mode', {method: 'POST'})
                        .then(response => response.json())
                        .then(data => console.log(data));
                }
                
                function closeAllTrades() {
                    if (confirm('Are you sure you want to close all trades?')) {
                        fetch('/api/close_all_trades', {method: 'POST'})
                            .then(response => response.json())
                            .then(data => console.log(data));
                    }
                }
                
                // Load initial data
                fetch('/api/dashboard')
                    .then(response => response.json())
                    .then(data => {
                        updateDashboard(data);
                        data.notifications.forEach(addNotification);
                        data.rejections.forEach(addRejection);
                    });
            </script>
        </body>
        </html>
    """)

@app.route('/api/dashboard')
def api_dashboard():
    """واجهة API لبيانات لوحة التحكم"""
    return jsonify(get_dashboard_payload())

@app.route('/api/toggle_trading', methods=['POST'])
def api_toggle_trading():
    """تبديل حالة التداول"""
    global is_trading_enabled
    with trading_status_lock:
        is_trading_enabled = not is_trading_enabled
        status = "enabled" if is_trading_enabled else "disabled"
        log_and_notify('info', f"Trading {status}", 'system')
        return jsonify({"status": status})

@app.route('/api/toggle_mode', methods=['POST'])
def api_toggle_mode():
    """تبديل وضع التداول (ورقي/حقيقي)"""
    global paper_trading_mode
    with trading_mode_lock:
        paper_trading_mode = not paper_trading_mode
        mode = "paper" if paper_trading_mode else "real"
        log_and_notify('info', f"Trading mode changed to {mode}", 'system')
        save_settings_to_redis()
        return jsonify({"mode": mode})

@app.route('/api/close_all_trades', methods=['POST'])
def api_close_all_trades():
    """إغلاق جميع الصفقات المفتوحة"""
    try:
        with signal_cache_lock:
            symbols_to_close = list(open_signals_cache.keys())
        
        for symbol in symbols_to_close:
            # هنا يتم إغلاق الصفقة
            # في التطبيق الحقيقي، سيتم تنفيذ أمر إغلاق
            with signal_cache_lock:
                if symbol in open_signals_cache:
                    del open_signals_cache[symbol]
        
        log_and_notify('warning', "All trades closed manually", 'system')
        return jsonify({"status": "success", "closed_trades": len(symbols_to_close)})
    except Exception as e:
        logger.error(f"Error closing all trades: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    """إعدادات البوت"""
    if request.method == 'GET':
        return jsonify({
            "min_quality": MIN_SIGNAL_QUALITY,
            "trade_amount_min": FIXED_TRADE_AMOUNT_MIN_USDT,
            "trade_amount_max": FIXED_TRADE_AMOUNT_MAX_USDT,
            "max_open_trades": MAX_OPEN_TRADES,
            "strategies": {
                "BB_Stoch_Strategy": USE_BB_STOCH_STRATEGY,
                "MACD_EMA_Strategy": USE_MACD_EMA_STRATEGY,
                "EMA_RSI_Strategy": USE_EMA_RSI_STRATEGY,
                "Pullback_Strategy": USE_PULLBACK_STRATEGY,
                "Momentum_Volatility_Strategy": USE_MOMENTUM_VOLATILITY_STRATEGY,
                "Elliott_Wave_Strategy": USE_ELLIOTT_WAVE_STRATEGY,
                "Range_Reversal_Strategy": USE_RANGE_REVERSAL_STRATEGY
            }
        })
    else:
        try:
            data = request.get_json()
            
            # تحديث الإعدادات باستخدام global بشكل صحيح
            global MIN_SIGNAL_QUALITY, FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT, MAX_OPEN_TRADES
            global USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY
            global USE_PULLBACK_STRATEGY, USE_MOMENTUM_VOLATILITY_STRATEGY, USE_ELLIOTT_WAVE_STRATEGY, USE_RANGE_REVERSAL_STRATEGY
            
            with min_quality_lock:
                MIN_SIGNAL_QUALITY = int(data.get('min_quality', MIN_SIGNAL_QUALITY))
            
            with trade_amount_lock:
                FIXED_TRADE_AMOUNT_MIN_USDT = float(data.get('trade_amount_min', FIXED_TRADE_AMOUNT_MIN_USDT))
                FIXED_TRADE_AMOUNT_MAX_USDT = float(data.get('trade_amount_max', FIXED_TRADE_AMOUNT_MAX_USDT))
            
            MAX_OPEN_TRADES = int(data.get('max_open_trades', MAX_OPEN_TRADES))
            
            # تحديث الاستراتيجيات
            strategies = data.get('strategies', {})
            USE_BB_STOCH_STRATEGY = strategies.get('BB_Stoch_Strategy', USE_BB_STOCH_STRATEGY)
            USE_MACD_EMA_STRATEGY = strategies.get('MACD_EMA_Strategy', USE_MACD_EMA_STRATEGY)
            USE_EMA_RSI_STRATEGY = strategies.get('EMA_RSI_Strategy', USE_EMA_RSI_STRATEGY)
            USE_PULLBACK_STRATEGY = strategies.get('Pullback_Strategy', USE_PULLBACK_STRATEGY)
            USE_MOMENTUM_VOLATILITY_STRATEGY = strategies.get('Momentum_Volatility_Strategy', USE_MOMENTUM_VOLATILITY_STRATEGY)
            USE_ELLIOTT_WAVE_STRATEGY = strategies.get('Elliott_Wave_Strategy', USE_ELLIOTT_WAVE_STRATEGY)
            USE_RANGE_REVERSAL_STRATEGY = strategies.get('Range_Reversal_Strategy', USE_RANGE_REVERSAL_STRATEGY)
            
            # حفظ الإعدادات
            save_settings_to_redis()
            
            log_and_notify('info', "Settings updated successfully", 'system')
            return jsonify({"status": "success"})
            
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

@sock.route('/ws')
def websocket_connection(ws):
    """
    اتصال WebSocket للتحديثات الحية
    """
    with ws_clients_lock:
        ws_clients.append(ws)
    
    try:
        # إرسال البيانات الأولية
        ws.send(json.dumps({"type": "dashboard_update", "payload": get_dashboard_payload()}, cls=NpEncoder))
        
        # الحفاظ على الاتصال مفتوحًا
        while True:
            data = ws.receive()
            if data is None:
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        with ws_clients_lock:
            if ws in ws_clients:
                ws_clients.remove(ws)

# --- دالة البدء الرئيسية ---
def main():
    """
    دالة البدء الرئيسية للبوت
    """
    logger.info("🚀 Starting Crypto Bot V34.1.1...")
    
    # تهيئة الاتصالات
    init_db()
    init_redis()
    
    # تهيئة عميل Binance
    global client
    try:
        client = Client(API_KEY, API_SECRET)
        logger.info("✅ [API] Connected to Binance API")
    except Exception as e:
        logger.error(f"❌ [API] Failed to connect to Binance API: {e}")
        exit(1)
    
    # تحميل الإعدادات والبيانات
    load_settings_from_redis()
    load_open_signals_to_cache()
    load_notifications_to_cache()
    validated_symbols_to_scan.extend(get_validated_symbols())
    
    # بدء WebSocket
    start_websocket()
    
    # بدء التقارير الدورية
    start_periodic_reports()
    
    # بدء حلقة التداول في خيط منفصل
    trading_thread = Thread(target=trading_loop, daemon=True)
    trading_thread.start()
    
    # بدء تطبيق Flask
    logger.info("🌐 Starting web dashboard...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    main()