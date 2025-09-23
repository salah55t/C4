# ملف c4_5min_v35_0_0.py - نسخة V35.0.0 (تحسين متقدم لتناغم الاستراتيجيات وإدارة الصفقات)
# --- وصف التعديلات:
# 1. [نظام تقييم السوق المتكامل] إضافة نظام شامل لتقييم حالة السوق وتعديل الاستراتيجيات ديناميكيًا
# 2. [إدارة صفقات متقدمة] تحسين نظام تحديث وقف الخسارة والأهداف بشكل ديناميكي
# 3. [تحليل دوري للصفقات] إضافة نظام تحليل دوري للصفقات المفتوحة وتحديثها
# 4. [إشعارات تلغرام محسنة] تحسين دقة ووضوح إشعارات تلغرام
# 5. [لوحة تحكم متقدمة] إضافة مؤشرات وتفاصيل أكثر دقة للوحة التحكم

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
from threading import Thread, Lock, Event
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
current_market_state: Dict[str, Any] = {"trend_details_by_tf": {}, "market_regime": "unknown", "volatility_state": "medium"}
market_state_lock = Lock()

# --- متغيرات جديدة لإدارة الصفقات المتقدمة ---
trade_analysis_thread: Optional[Thread] = None
stop_trade_analysis = Event()
trailing_stop_updates: Dict[str, Dict] = {}  # لتتبع تحديثات وقف الخسارة المتحرك
dynamic_targets: Dict[str, Dict] = {}  # لتتبع تحديثات الأهداف الديناميكية
trade_analysis_lock = Lock()

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
                # **FIX**: Add created_at to the columns to check and add if missing
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
    
    # إضافة معلومات حالة السوق
    with market_state_lock:
        market_regime = current_market_state.get("market_regime", "unknown")
        volatility_state = current_market_state.get("volatility_state", "medium")
    
    regime_emoji = {"trending": "📈", "ranging": "↔️", "volatile": "🌪️", "unknown": "❓"}.get(market_regime, "❓")
    volatility_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(volatility_state, "🟡")
    
    message = (
        f"{emoji} *صفقة {trade_type} جديدة (5 دقائق)*\n\n"
        f"*العملة:* `{symbol}`\n"
        f"*الاستراتيجية:* `{STRATEGY_NAMES.get(strategy_name, strategy_name)}`\n"
        f"*جودة الإشارة:* `{quality_score}/100`\n"
        f"*حالة السوق:* {regime_emoji} `{market_regime}` {volatility_emoji} `{volatility_state}`\n"
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

def send_trade_update_notification(symbol: str, update_type: str, old_value: float, new_value: float, 
                                 current_price: float, profit_percent: float, is_real: bool):
    trade_type = "حقيقية" if is_real else "ورقية"
    emoji = "🔄"
    
    if update_type == "stop_loss":
        field_name = "وقف الخسارة"
        emoji = "🛡️"
    elif update_type == "target_1":
        field_name = "الهدف الأول"
        emoji = "🎯"
    elif update_type == "target_2":
        field_name = "الهدف الثاني"
        emoji = "🏆"
    
    profit_emoji = "📈" if profit_percent >= 0 else "📉"
    
    message = (
        f"{emoji} *تحديث صفقة {trade_type} (5 دقائق)*\n\n"
        f"*العملة:* `{symbol}`\n"
        f"*نوع التحديث:* `{field_name}`\n"
        f"*القيمة القديمة:* `{old_value:.4f}`\n"
        f"*القيمة الجديدة:* `{new_value:.4f}`\n"
        f"*السعر الحالي:* `{current_price:.4f}`\n"
        f"{profit_emoji} *الربح/الخسارة:* `{profit_percent:.2f}%`"
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
            
            # إضافة معلومات عن حالة السوق اليوم
            with market_state_lock:
                market_regime = current_market_state.get("market_regime", "unknown")
                volatility_state = current_market_state.get("volatility_state", "medium")
            
            regime_emoji = {"trending": "📈", "ranging": "↔️", "volatile": "🌪️", "unknown": "❓"}.get(market_regime, "❓")
            volatility_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(volatility_state, "🟡")
            
            message = (
                f"📈 *تقرير الأداء اليومي*\n\n"
                f"*التاريخ:* `{today.strftime('%Y-%m-%d')}`\n"
                f"*حالة السوق:* {regime_emoji} `{market_regime}` {volatility_emoji} `{volatility_state}`\n\n"
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
    
    # إضافة معلومات عامة عن حالة السوق
    market_regime = state.get("market_regime", "unknown")
    volatility_state = state.get("volatility_state", "medium")
    
    regime_emoji = {"trending": "📈", "ranging": "↔️", "volatile": "🌪️", "unknown": "❓"}.get(market_regime, "❓")
    volatility_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(volatility_state, "🟡")
    
    message += f"{regime_emoji} *نظام السوق:* `{market_regime}`\n"
    message += f"{volatility_emoji} *مستوى التقلب:* `{volatility_state}`\n\n"
    
    # إضافة تفاصيل الاتجاه لكل فريم
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
                for signal in cur.fetchall(): 
                    open_signals_cache[signal['symbol']] = dict(signal)
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

# --- نظام تقييم حالة السوق المتكامل ---
def analyze_market_regime(df_5m: pd.DataFrame, df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> Dict[str, str]:
    """
    تحليل نظام السوق (trending, ranging, volatile)
    """
    try:
        # تحليل تقلب السوق
        atr_percent_5m = df_5m['atr_percent'].iloc[-1]
        atr_percent_15m = df_15m['atr_percent'].iloc[-1]
        atr_percent_1h = df_1h['atr_percent'].iloc[-1]
        
        # تحديد حالة التقلب
        if atr_percent_5m > 2.5 or atr_percent_15m > 3.0 or atr_percent_1h > 3.5:
            volatility_state = "high"
        elif atr_percent_5m < 1.0 or atr_percent_15m < 1.2 or atr_percent_1h < 1.5:
            volatility_state = "low"
        else:
            volatility_state = "medium"
        
        # تحليل اتجاه السوق
        adx_5m = df_5m['adx'].iloc[-1]
        adx_15m = df_15m['adx'].iloc[-1]
        adx_1h = df_1h['adx'].iloc[-1]
        
        # تحديد قوة الاتجاه
        strong_trend = (adx_5m > 22 and adx_15m > 20 and adx_1h > 18)
        weak_trend = (adx_5m < 15 and adx_15m < 15 and adx_1h < 15)
        
        # تحليل نطاق التداول
        bb_width_5m = df_5m['bb_width'].iloc[-1]
        bb_width_15m = df_15m['bb_width'].iloc[-1]
        bb_width_1h = df_1h['bb_width'].iloc[-1]
        
        # تحديد حالة النطاق
        wide_range = (bb_width_5m > 0.05 or bb_width_15m > 0.06 or bb_width_1h > 0.07)
        narrow_range = (bb_width_5m < 0.02 or bb_width_15m < 0.025 or bb_width_1h < 0.03)
        
        # تحديد نظام السوق
        if volatility_state == "high" and wide_range:
            market_regime = "volatile"
        elif strong_trend:
            market_regime = "trending"
        elif weak_trend and narrow_range:
            market_regime = "ranging"
        else:
            market_regime = "unknown"
        
        return {
            "market_regime": market_regime,
            "volatility_state": volatility_state
        }
    except Exception as e:
        logger.error(f"❌ [Market Analysis] Error analyzing market regime: {e}")
        return {"market_regime": "unknown", "volatility_state": "medium"}

def update_market_state():
    """
    تحديث حالة السوق العامة
    """
    try:
        # جلب البيانات لفريمات مختلفة
        btc_5m = fetch_historical_data(BTC_SYMBOL, '5m', 2)
        btc_15m = fetch_historical_data(BTC_SYMBOL, '15m', 3)
        btc_1h = fetch_historical_data(BTC_SYMBOL, '1h', 5)
        
        if btc_5m is None or btc_15m is None or btc_1h is None:
            logger.warning("❌ [Market Analysis] Could not fetch market data")
            return
        
        # حساب المؤشرات
        btc_5m = calculate_all_features(btc_5m)
        btc_15m = calculate_all_features(btc_15m)
        btc_1h = calculate_all_features(btc_1h)
        
        # تحليل نظام السوق
        market_analysis = analyze_market_regime(btc_5m, btc_15m, btc_1h)
        
        # تحليل اتجاه كل فريم
        trend_details = {}
        
        for tf, df in [('5m', btc_5m), ('15m', btc_15m), ('1h', btc_1h)]:
            last = df.iloc[-1]
            
            # تحديد الاتجاه
            if last['ema50'] > last['ema200'] and last['close'] > last['ema50']:
                trend = "bullish"
            elif last['ema50'] < last['ema200'] and last['close'] < last['ema50']:
                trend = "bearish"
            else:
                trend = "neutral"
            
            trend_details[tf] = {
                "trend": trend,
                "adx": last['adx'],
                "rsi": last['rsi']
            }
        
        # تحديث حالة السوق
        with market_state_lock:
            current_market_state["trend_details_by_tf"] = trend_details
            current_market_state["market_regime"] = market_analysis["market_regime"]
            current_market_state["volatility_state"] = market_analysis["volatility_state"]
        
        # إرسال تحديث عبر WebSocket
        broadcast({
            "type": "market_state_update",
            "payload": {
                "trend_details_by_tf": trend_details,
                "market_regime": market_analysis["market_regime"],
                "volatility_state": market_analysis["volatility_state"]
            }
        })
        
        logger.info(f"✅ [Market Analysis] Updated market state: {market_analysis['market_regime']} regime, {market_analysis['volatility_state']} volatility")
        
    except Exception as e:
        logger.error(f"❌ [Market Analysis] Error updating market state: {e}")

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

def check_bb_stoch_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    
    # تعديل العتبات بناءً على حالة السوق
    with market_state_lock:
        market_regime = current_market_state.get("market_regime", "unknown")
        volatility_state = current_market_state.get("volatility_state", "medium")
    
    # تعديل عرض البولينجر بناءً على حالة السوق
    bb_width = df['bb_width']
    if volatility_state == "high":
        dynamic_bb_threshold = bb_width.rolling(20).mean() * 1.5
    elif volatility_state == "low":
        dynamic_bb_threshold = bb_width.rolling(20).mean() * 0.9
    else:
        dynamic_bb_threshold = bb_width.rolling(20).mean() * 1.2

    # تعديل عتبة الستوكاستيك بناءً على حالة السوق
    if volatility_state == "high":
        stoch_threshold = 28
    elif market_regime == "trending":
        stoch_threshold = 20
    else:
        stoch_threshold = 23
    
    # تعديل عتبة الحجم بناءً على حالة السوق
    volume_ma = df['volume'].rolling(20).mean()
    if volatility_state == "high":
        volume_multiplier = 1.0 + (atr_percent / 80)
    elif market_regime == "trending":
        volume_multiplier = 1.0 + (atr_percent / 120)
    else:
        volume_multiplier = 1.0 + (atr_percent / 100)
    
    return {
        'bb_width_ok': bb_width.iloc[-1] > dynamic_bb_threshold.iloc[-1],
        'stoch_ok': last_row['stoch_k'] > stoch_threshold,
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * volume_multiplier
    }

def check_macd_ema_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    
    # تعديل العتبات بناءً على حالة السوق
    with market_state_lock:
        market_regime = current_market_state.get("market_regime", "unknown")
        volatility_state = current_market_state.get("volatility_state", "medium")
    
    # تعديل عتبة ADX بناءً على حالة السوق
    if volatility_state == "high":
        default_adx_thresh = 22
    elif market_regime == "trending":
        default_adx_thresh = 18
    else:
        default_adx_thresh = 20
    
    adx_threshold = default_adx_thresh
    
    # تعديل عتبة الحجم بناءً على حالة السوق
    volume_ma = df['volume'].rolling(20).mean()
    if volatility_state == "high":
        volatility_adjusted_volume = volume_ma * (1 + atr_percent / 60)
    elif market_regime == "trending":
        volatility_adjusted_volume = volume_ma * (1 + atr_percent / 90)
    else:
        volatility_adjusted_volume = volume_ma * (1 + atr_percent / 75)
    
    # تعديل عتبة الزخم بناءً على حالة السوق
    macd_momentum = df['macd_hist'].diff()
    if volatility_state == "high":
        momentum_threshold = macd_momentum.rolling(10).std() * 0.4
    elif market_regime == "trending":
        momentum_threshold = macd_momentum.rolling(10).std() * 0.2
    else:
        momentum_threshold = macd_momentum.rolling(10).std() * 0.3
    
    return {
        'adx_ok': last_row['adx'] > adx_threshold,
        'volume_ok': last_row['volume'] > volatility_adjusted_volume.iloc[-1],
        'momentum_ok': macd_momentum.iloc[-1] > momentum_threshold.iloc[-1],
    }

def check_ema_rsi_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    adx = last_row.get('adx', 0)
    
    # تعديل العتبات بناءً على حالة السوق
    with market_state_lock:
        market_regime = current_market_state.get("market_regime", "unknown")
        volatility_state = current_market_state.get("volatility_state", "medium")
    
    # تعديل نطاق RSI بناءً على حالة السوق
    if adx > 25:
        if volatility_state == "high":
            rsi_lower, rsi_upper = 45, 80
        else:
            rsi_lower, rsi_upper = 42, 78
    else:
        if market_regime == "ranging":
            rsi_lower, rsi_upper = 40, 60
        else:
            rsi_lower, rsi_upper = 48, 72
    
    # تعديل عتبة تباعد المتوسطات بناءً على حالة السوق
    ema_spread = (df['ema9'] - df['ema21']) / df['ema21'].replace(0, 1e-9)
    if volatility_state == "high":
        dynamic_ema_threshold = ema_spread.rolling(20).std() * 2.0
    elif market_regime == "trending":
        dynamic_ema_threshold = ema_spread.rolling(20).std() * 1.5
    else:
        dynamic_ema_threshold = ema_spread.rolling(20).std() * 1.7
    
    # تعديل عتبة الحجم بناءً على حالة السوق
    volume_ma = df['volume'].rolling(20).mean()
    if volatility_state == "high":
        trend_strength_multiplier = 1 + (adx / 80)
    elif market_regime == "trending":
        trend_strength_multiplier = 1 + (adx / 120)
    else:
        trend_strength_multiplier = 1 + (adx / 100)
    
    return {
        'rsi_ok': rsi_lower < last_row['rsi'] < rsi_upper,
        'ema_ok': ema_spread.iloc[-1] > dynamic_ema_threshold.iloc[-1],
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * trend_strength_multiplier,
    }

def check_pullback_dynamic_filters(df: pd.DataFrame, mtf_trend: Dict) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    
    # تعديل العتبات بناءً على حالة السوق
    with market_state_lock:
        market_regime = current_market_state.get("market_regime", "unknown")
        volatility_state = current_market_state.get("volatility_state", "medium")
    
    # تعديل عمق الارتداد بناءً على حالة السوق
    if volatility_state == "high":
        pullback_depth = 0.04
    elif market_regime == "trending":
        pullback_depth = 0.025
    else:
        pullback_depth = 0.035
    
    if mtf_trend.get('5m') == 'bullish' and mtf_trend.get('15m') == 'bullish':
        pullback_depth *= 1.2
    
    # تعديل عتبة التعافي بناءً على حالة السوق
    recent_low = df['low'].tail(5).min()
    if volatility_state == "high":
        recovery_threshold = recent_low * (1 + (pullback_depth * 0.8))
    elif market_regime == "trending":
        recovery_threshold = recent_low * (1 + (pullback_depth * 1.0))
    else:
        recovery_threshold = recent_low * (1 + (pullback_depth * 0.9))
    
    # تعديل عتبة الحجم بناءً على حالة السوق
    volume_ma = df['volume'].rolling(20).mean()
    if volatility_state == "high":
        recovery_volume_multiplier = 1.2 + (atr_percent / 80)
    elif market_regime == "trending":
        recovery_volume_multiplier = 1.0 + (atr_percent / 120)
    else:
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
    
    # تعديل العتبات بناءً على حالة السوق
    with market_state_lock:
        market_regime = current_market_state.get("market_regime", "unknown")
        volatility_state = current_market_state.get("volatility_state", "medium")
    
    # تعديل نطاق التقلب بناءً على حالة السوق
    if volatility_state == "high":
        dynamic_vol_min = volatility_ma.iloc[-1] - (volatility_std.iloc[-1] * 1.8)
        dynamic_vol_max = volatility_ma.iloc[-1] + (volatility_std.iloc[-1] * 1.8)
    elif market_regime == "trending":
        dynamic_vol_min = volatility_ma.iloc[-1] - (volatility_std.iloc[-1] * 1.2)
        dynamic_vol_max = volatility_ma.iloc[-1] + (volatility_std.iloc[-1] * 1.2)
    else:
        dynamic_vol_min = volatility_ma.iloc[-1] - (volatility_std.iloc[-1] * 1.5)
        dynamic_vol_max = volatility_ma.iloc[-1] + (volatility_std.iloc[-1] * 1.5)
    
    # تعديل عتبة الزخم بناءً على حالة السوق
    if volatility_state == "high":
        is_momentum_ok = (last_row['rsi'] > 53) and (df['macd_hist'].iloc[-1] > df['macd_hist'].iloc[-2])
    elif market_regime == "trending":
        is_momentum_ok = (last_row['rsi'] > 50) and (df['macd_hist'].iloc[-1] > df['macd_hist'].iloc[-3])
    else:
        is_momentum_ok = (last_row['rsi'] > 51) and (df['macd_hist'].iloc[-1] > df['macd_hist'].iloc[-2])

    # تعديل عتبة ADX بناءً على حالة السوق
    adx_ma = df['adx'].rolling(20).mean()
    if volatility_state == "high":
        dynamic_adx_threshold = adx_ma.iloc[-1] * 0.9
    elif market_regime == "trending":
        dynamic_adx_threshold = adx_ma.iloc[-1] * 0.8
    else:
        dynamic_adx_threshold = adx_ma.iloc[-1] * 0.85
    
    return {
        'volatility_ok': dynamic_vol_min <= atr_percent.iloc[-1] <= dynamic_vol_max,
        'momentum_ok': is_momentum_ok,
        'adx_ok': last_row['adx'] > dynamic_adx_threshold,
    }

def check_elliott_wave_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    
    # تعديل العتبات بناءً على حالة السوق
    with market_state_lock:
        market_regime = current_market_state.get("market_regime", "unknown")
        volatility_state = current_market_state.get("volatility_state", "medium")
    
    # تعديل نطاق فيبوناتشي بناءً على حالة السوق
    if volatility_state == "high":
        fib_min, fib_max = 0.18, 0.95  # نطاق أوسع للتصحيحات العميقة
    elif market_regime == "trending":
        fib_min, fib_max = 0.236, 0.786  # نطاق أكثر تحفظًا
    else:
        fib_min, fib_max = 0.18, 0.886  # نطاق أوسع قليلاً
    
    # تعديل عتبة الحجم بناءً على حالة السوق
    volume_ma = df['volume'].rolling(20).mean()
    if volatility_state == "high":
        wave_volume_multiplier = 1.4 + (atr_percent / 40)
    elif market_regime == "trending":
        wave_volume_multiplier = 1.2 + (atr_percent / 60)
    else:
        wave_volume_multiplier = 1.3 + (atr_percent / 50)
    
    # تعديل عتبة الزخم بناءً على حالة السوق
    macd_momentum = df['macd_hist'].rolling(5).mean()
    if volatility_state == "high":
        momentum_threshold = macd_momentum.rolling(20).std() * 0.4
    elif market_regime == "trending":
        momentum_threshold = macd_momentum.rolling(20).std() * 0.2
    else:
        momentum_threshold = macd_momentum.rolling(20).std() * 0.3
    
    return {
        'fibonacci_ok': fib_min <= get_wave_retracement(df) <= fib_max,
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * wave_volume_multiplier,
        'momentum_ok': macd_momentum.iloc[-1] > momentum_threshold.iloc[-1],
    }

def check_range_reversal_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    adx = last_row.get('adx', 99)
    
    # تعديل العتبات بناءً على حالة السوق
    with market_state_lock:
        market_regime = current_market_state.get("market_regime", "unknown")
        volatility_state = current_market_state.get("volatility_state", "medium")
    
    # تعديل عتبة ADX بناءً على حالة السوق
    if volatility_state == "high":
        adx_threshold = 20
    elif market_regime == "ranging":
        adx_threshold = 25
    else:
        adx_threshold = 23
    
    adx_ok = adx < adx_threshold
    
    # تعديل عتبة RSI بناءً على حالة السوق
    rsi = last_row.get('rsi', 50)
    atr_percent = last_row.get('atr_percent', 0)
    
    if volatility_state == "high":
        rsi_threshold = 42
    elif market_regime == "ranging":
        rsi_threshold = 30
    else:
        rsi_threshold = 35
    
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
    
    # تعديل نطاق التقلب بناءً على حالة السوق
    with market_state_lock:
        volatility_state = current_market_state.get("volatility_state", "medium")
    
    if volatility_state == "high":
        ATR_PERCENT_MIN = 0.7
        ATR_PERCENT_MAX = 3.5
    elif volatility_state == "low":
        ATR_PERCENT_MIN = 0.3
        ATR_PERCENT_MAX = 2.0
    else:
        ATR_PERCENT_MIN = 0.5
        ATR_PERCENT_MAX = 2.8
    
    if not (ATR_PERCENT_MIN <= last_atr_percent <= ATR_PERCENT_MAX):
        log_rejection(symbol, "Market Volatility Filter Failed", {
            "atr": f"{last_atr_percent:.2f}%",
            "range": f"({ATR_PERCENT_MIN:.2f}-{ATR_PERCENT_MAX:.2f})%"
        })
        return False
    
    return True

# --- تحسين منطق وقف الخسارة وجني الأرباح ---
def calculate_dynamic_stop_loss_enhanced(df: pd.DataFrame, entry_price: float, strategy_name: str) -> float:
    last = df.iloc[-1]
    atr_value = last.get('atr', 0)
    atr_percent = last.get('atr_percent', 0)
    
    # تعديل مسافة وقف الخسارة بناءً على حالة السوق
    with market_state_lock:
        market_regime = current_market_state.get("market_regime", "unknown")
        volatility_state = current_market_state.get("volatility_state", "medium")
    
    # تعديل معامل ATR بناءً على حالة السوق
    if volatility_state == "high":
        atr_multiplier = 2.8
    elif market_regime == "trending":
        atr_multiplier = 2.2
    else:
        atr_multiplier = 2.5
    
    # تعديل مسافة وقف الخسارة بناءً على حجم التداول
    volume_ma = df['volume'].rolling(20).mean()
    volume_ratio = last['volume'] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1
    
    if volume_ratio > 2.0:  # حجم تداول عالي
        atr_multiplier *= 0.8  # تقليل مسافة وقف الخسارة
    elif volume_ratio < 0.5:  # حجم تداول منخفض
        atr_multiplier *= 1.2  # زيادة مسافة وقف الخسارة
    
    # حساب وقف الخسارة بناءً على الاستراتيجية
    if strategy_name == "BB_Stoch_Strategy":
        recent_low = df['low'].tail(3).min()
        stop_loss = min(recent_low * 0.995, entry_price - (atr_value * atr_multiplier))
    elif strategy_name == "MACD_EMA_Strategy":
        stop_loss = min(last['ema21'], entry_price - (atr_value * atr_multiplier))
    elif strategy_name == "EMA_RSI_Strategy":
        stop_loss = min(last['ema21'], entry_price - (atr_value * atr_multiplier * 0.9))
    elif strategy_name == "Pullback_Strategy":
        recent_low = df['low'].tail(5).min()
        stop_loss = min(recent_low * 0.995, entry_price - (atr_value * atr_multiplier * 0.75))
    elif strategy_name == "Momentum_Volatility_Strategy":
        stop_loss = min(last['ema21'], entry_price - (atr_value * atr_multiplier * 1.1))
    elif strategy_name == "Elliott_Wave_Strategy":
        lows = df['low'].values
        try:
            support_idx = argrelextrema(lows, np.less, order=5)[0]
            if len(support_idx) > 0:
                recent_support = lows[support_idx[-1]]
                stop_loss = min(recent_support * 0.995, entry_price - (atr_value * atr_multiplier))
            else:
                stop_loss = min(last['ema21'], entry_price - (atr_value * atr_multiplier))
        except Exception as e:
            logger.error(f"Error calculating stop loss for Elliott Wave: {e}")
            stop_loss = entry_price - (atr_value * atr_multiplier)
    elif strategy_name == "Range_Reversal_Strategy":
        recent_low = df['low'].tail(5).min()
        stop_loss = min(recent_low * 0.99, entry_price - (atr_value * atr_multiplier * 0.6))
    else:
        stop_loss = entry_price - (atr_value * atr_multiplier)
    
    # الحد الأقصى لمسافة وقف الخسارة
    max_stop_distance = entry_price * 0.05
    if entry_price - stop_loss > max_stop_distance:
        stop_loss = entry_price - max_stop_distance
    
    return stop_loss

def calculate_dynamic_take_profit_enhanced(df: pd.DataFrame, entry_price: float, stop_loss: float, strategy_name: str) -> tuple:
    risk_amount = entry_price - stop_loss
    if risk_amount <= 0: return (entry_price * 1.015, entry_price * 1.025) # Default for 5m

    last = df.iloc[-1]
    atr_percent = last.get('atr_percent', 0)
    
    # تعديل نسبة المخاطرة إلى المكافأة بناءً على حالة السوق
    with market_state_lock:
        market_regime = current_market_state.get("market_regime", "unknown")
        volatility_state = current_market_state.get("volatility_state", "medium")
    
    if volatility_state == "high":
        volatility_adjustment = 0.7
    elif market_regime == "trending":
        volatility_adjustment = 1.1
    else:
        volatility_adjustment = 1.0
    
    # تعديل نسبة المخاطرة إلى المكافأة بناءً على حجم التداول
    volume_ma = df['volume'].rolling(20).mean()
    volume_ratio = last['volume'] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1
    
    if volume_ratio > 2.0:  # حجم تداول عالي
        volume_adjustment = 1.3
    elif volume_ratio < 0.5:  # حجم تداول منخفض
        volume_adjustment = 0.7
    else:
        volume_adjustment = 1.0
    
    # حساب نسبة المخاطرة إلى المكافأة النهائية
    adjustment_factor = volatility_adjustment * volume_adjustment
    
    # Risk-Reward Ratios adjusted for 5m timeframe (Scalping)
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
        return middle_band, upper_band
    else: 
        rr1, rr2 = 1.6 * adjustment_factor, 2.8 * adjustment_factor
        
    target1 = entry_price + (risk_amount * rr1)
    target2 = entry_price + (risk_amount * rr2)
    
    return target1, target2

# --- نظام تحليل الصفقات المفتوحة وتحديثها ---
def analyze_open_trades():
    """
    تحليل دوري للصفقات المفتوحة وتحديث وقف الخسارة والأهداف
    """
    logger.info("🔄 [Trade Analysis] Starting open trades analysis...")
    
    while not stop_trade_analysis.is_set():
        try:
            # تحديث حالة السوق
            update_market_state()
            
            # تحليل الصفقات المفتوحة
            with signal_cache_lock:
                open_signals = dict(open_signals_cache)
            
            if not open_signals:
                time.sleep(60)  # انتظار دقيقة إذا لم تكن هناك صفقات مفتوحة
                continue
            
            for symbol, signal in open_signals.items():
                try:
                    # جلب البيانات الحالية
                    df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, 1)
                    if df is None or len(df) < 50:
                        continue
                    
                    # حساب المؤشرات
                    df = calculate_all_features(df)
                    
                    # الحصول على السعر الحالي
                    with live_prices_lock:
                        current_price = live_prices.get(symbol, df['close'].iloc[-1])
                    
                    # حساب الربح/الخسارة الحالي
                    entry_price = signal['entry_price']
                    profit_percent = ((current_price - entry_price) / entry_price) * 100
                    
                    # تحديث وقف الخسارة المتحرك
                    new_stop_loss = update_trailing_stop(symbol, signal, df, current_price, profit_percent)
                    
                    # تحديث الأهداف الديناميكية
                    new_target1, new_target2 = update_dynamic_targets(symbol, signal, df, current_price, profit_percent)
                    
                    # تحديث الصفقة في قاعدة البيانات والكاش
                    if new_stop_loss != signal['stop_loss']:
                        update_trade_stop_loss(symbol, new_stop_loss)
                        send_trade_update_notification(
                            symbol, "stop_loss", signal['stop_loss'], new_stop_loss,
                            current_price, profit_percent, signal['is_real_trade']
                        )
                    
                    if new_target1 != signal.get('target_price_1', 0) or new_target2 != signal.get('target_price_2', 0):
                        update_trade_targets(symbol, new_target1, new_target2)
                        if new_target1 != signal.get('target_price_1', 0):
                            send_trade_update_notification(
                                symbol, "target_1", signal.get('target_price_1', 0), new_target1,
                                current_price, profit_percent, signal['is_real_trade']
                            )
                        if new_target2 != signal.get('target_price_2', 0):
                            send_trade_update_notification(
                                symbol, "target_2", signal.get('target_price_2', 0), new_target2,
                                current_price, profit_percent, signal['is_real_trade']
                            )
                    
                    # إرسال تحديث للوحة التحكم
                    broadcast({
                        "type": "trade_update",
                        "payload": {
                            "symbol": symbol,
                            "current_price": current_price,
                            "profit_percent": profit_percent,
                            "stop_loss": new_stop_loss,
                            "target_price_1": new_target1,
                            "target_price_2": new_target2
                        }
                    })
                    
                except Exception as e:
                    logger.error(f"❌ [Trade Analysis] Error analyzing trade for {symbol}: {e}")
            
            # انتظار قبل التحليل التالي
            time.sleep(30)  # تحليل كل 30 ثانية
            
        except Exception as e:
            logger.error(f"❌ [Trade Analysis] Error in trade analysis loop: {e}")
            time.sleep(60)

def update_trailing_stop(symbol: str, signal: Dict, df: pd.DataFrame, current_price: float, profit_percent: float) -> float:
    """
    تحديث وقف الخسارة المتحرك بناءً على حركة السعر
    """
    try:
        entry_price = signal['entry_price']
        current_stop_loss = signal['stop_loss']
        strategy_name = signal['strategy_name']
        
        # لا تحديث وقف الخسارة إذا كان الربح أقل من عتبة التفعيل
        if profit_percent < TRAILING_STOP_ACTIVATION_PROFIT_PERCENT:
            return current_stop_loss
        
        # الحصول على معلومات وقف الخسارة المتحرك السابقة
        with trade_analysis_lock:
            trailing_info = trailing_stop_updates.get(symbol, {
                "highest_price": entry_price,
                "activation_percent": TRAILING_STOP_ACTIVATION_PROFIT_PERCENT,
                "trail_percent": 0.5  # مسافة التتبع الافتراضية
            })
        
        # تحديث أعلى سعر إذا كان السعر الحالي أعلى
        if current_price > trailing_info["highest_price"]:
            trailing_info["highest_price"] = current_price
        
        # حساب مسافة التتبع بناءً على حالة السوق والاستراتيجية
        with market_state_lock:
            volatility_state = current_market_state.get("volatility_state", "medium")
        
        if volatility_state == "high":
            trail_percent = 0.7
        elif volatility_state == "low":
            trail_percent = 0.3
        else:
            trail_percent = 0.5
        
        # تعديل مسافة التتبع بناءً على الاستراتيجية
        if strategy_name == "Elliott_Wave_Strategy":
            trail_percent *= 0.8  # تتبع أقوى لموجات إليوت
        elif strategy_name == "Range_Reversal_Strategy":
            trail_percent *= 1.2  # تتبع أوسع لانعكاس النطاق
        
        # حساب وقف الخسارة الجديد
        new_stop_loss = trailing_info["highest_price"] * (1 - trail_percent / 100)
        
        # التأكد من أن وقف الخسارة الجديد أعلى من القديم (للصفقات الطويلة)
        if new_stop_loss > current_stop_loss:
            # تحديث معلومات التتبع
            with trade_analysis_lock:
                trailing_stop_updates[symbol] = trailing_info
            
            return new_stop_loss
        
        return current_stop_loss
        
    except Exception as e:
        logger.error(f"❌ [Trailing Stop] Error updating trailing stop for {symbol}: {e}")
        return signal['stop_loss']

def update_dynamic_targets(symbol: str, signal: Dict, df: pd.DataFrame, current_price: float, profit_percent: float) -> Tuple[float, float]:
    """
    تحديث الأهداف الديناميكية بناءً على حركة السعر وحالة السوق
    """
    try:
        entry_price = signal['entry_price']
        current_target1 = signal.get('target_price_1', entry_price * 1.015)
        current_target2 = signal.get('target_price_2', entry_price * 1.025)
        strategy_name = signal['strategy_name']
        
        # الحصول على معلومات الأهداف الديناميكية السابقة
        with trade_analysis_lock:
            target_info = dynamic_targets.get(symbol, {
                "initial_target1": current_target1,
                "initial_target2": current_target2,
                "adjustment_factor": 1.0,
                "last_adjustment_time": datetime.now(timezone.utc)
            })
        
        # لا تحديث الأهداف إذا كان الربح أقل من 0.5%
        if profit_percent < 0.5:
            return current_target1, current_target2
        
        # تحديث الأهداف فقط كل 5 دقائق على الأقل
        last_adjustment = target_info["last_adjustment_time"]
        if (datetime.now(timezone.utc) - last_adjustment).total_seconds() < 300:
            return current_target1, current_target2
        
        # حساب معامل التعديل بناءً على حالة السوق
        with market_state_lock:
            market_regime = current_market_state.get("market_regime", "unknown")
            volatility_state = current_market_state.get("volatility_state", "medium")
        
        # تعديل الأهداف بناءً على زخم السعر
        last = df.iloc[-1]
        atr_percent = last.get('atr_percent', 0)
        
        # حساب معامل الزخم
        momentum_factor = 1.0
        if df['macd_hist'].iloc[-1] > df['macd_hist'].iloc[-2] and last['rsi'] > 50:
            momentum_factor = 1.0 + (atr_percent / 100)
        
        # تعديل معامل التعديل بناءً على نظام السوق
        if market_regime == "trending" and momentum_factor > 1.0:
            adjustment_factor = min(1.3, target_info["adjustment_factor"] * 1.05)
        elif volatility_state == "high":
            adjustment_factor = max(0.9, target_info["adjustment_factor"] * 0.98)
        else:
            adjustment_factor = target_info["adjustment_factor"]
        
        # حساب الأهداف الجديدة
        new_target1 = target_info["initial_target1"] * adjustment_factor
        new_target2 = target_info["initial_target2"] * adjustment_factor
        
        # التأكد من أن الأهداف الجديدة أعلى من السعر الحالي
        if new_target1 <= current_price:
            new_target1 = current_price * 1.01
        if new_target2 <= current_price:
            new_target2 = current_price * 1.02
        
        # التأكد من أن الهدف الثاني أعلى من الأول
        if new_target2 <= new_target1:
            new_target2 = new_target1 * 1.015
        
        # تحديث معلومات الأهداف
        with trade_analysis_lock:
            dynamic_targets[symbol] = {
                "initial_target1": target_info["initial_target1"],
                "initial_target2": target_info["initial_target2"],
                "adjustment_factor": adjustment_factor,
                "last_adjustment_time": datetime.now(timezone.utc)
            }
        
        return new_target1, new_target2
        
    except Exception as e:
        logger.error(f"❌ [Dynamic Targets] Error updating dynamic targets for {symbol}: {e}")
        return signal.get('target_price_1', signal['entry_price'] * 1.015), signal.get('target_price_2', signal['entry_price'] * 1.025)

def update_trade_stop_loss(symbol: str, new_stop_loss: float):
    """
    تحديث وقف الخسارة في قاعدة البيانات والكاش
    """
    if not check_db_connection() or not conn:
        return
    
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE signals SET stop_loss = %s, status = 'updated' WHERE symbol = %s AND status IN ('open', 'updated');", 
                       (new_stop_loss, symbol))
            conn.commit()
        
        with signal_cache_lock:
            if symbol in open_signals_cache:
                open_signals_cache[symbol]['stop_loss'] = new_stop_loss
                open_signals_cache[symbol]['status'] = 'updated'
        
        logger.info(f"✅ [Trade Update] Updated stop loss for {symbol} to {new_stop_loss}")
    except Exception as e:
        logger.error(f"❌ [Trade Update] Error updating stop loss for {symbol}: {e}")
        if conn: conn.rollback()

def update_trade_targets(symbol: str, new_target1: float, new_target2: float):
    """
    تحديث الأهداف في قاعدة البيانات والكاش
    """
    if not check_db_connection() or not conn:
        return
    
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE signals SET target_price_1 = %s, target_price_2 = %s, status = 'updated' WHERE symbol = %s AND status IN ('open', 'updated');", 
                       (new_target1, new_target2, symbol))
            conn.commit()
        
        with signal_cache_lock:
            if symbol in open_signals_cache:
                open_signals_cache[symbol]['target_price_1'] = new_target1
                open_signals_cache[symbol]['target_price_2'] = new_target2
                open_signals_cache[symbol]['status'] = 'updated'
        
        logger.info(f"✅ [Trade Update] Updated targets for {symbol} to {new_target1}, {new_target2}")
    except Exception as e:
        logger.error(f"❌ [Trade Update] Error updating targets for {symbol}: {e}")
        if conn: conn.rollback()

def start_trade_analysis():
    """
    بدء خيط تحليل الصفقات
    """
    global trade_analysis_thread
    
    if trade_analysis_thread and trade_analysis_thread.is_alive():
        logger.warning("[Trade Analysis] Trade analysis thread is already running")
        return
    
    stop_trade_analysis.clear()
    trade_analysis_thread = Thread(target=analyze_open_trades, daemon=True)
    trade_analysis_thread.start()
    logger.info("✅ [Trade Analysis] Started trade analysis thread")

def stop_trade_analysis_thread():
    """
    إيقاف خيط تحليل الصفقات
    """
    global trade_analysis_thread
    
    stop_trade_analysis.set()
    if trade_analysis_thread and trade_analysis_thread.is_alive():
        trade_analysis_thread.join(timeout=5)
        logger.info("✅ [Trade Analysis] Stopped trade analysis thread")

# --- استراتيجيات التداول المحسنة ---
def check_ema_rsi_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 200: return False

    last = df.iloc[-1]
    if not (last['ema50'] > last['ema200'] and last['close'] > last['ema9']):
        return False
        
    filters = check_ema_rsi_dynamic_filters(df)
    if not filters['rsi_ok']:
        log_rejection(symbol_name, "RSI Out of Range", {"rsi": last['rsi']})
        return False
    if not filters['ema_ok']:
        log_rejection(symbol_name, "DYN_EMA_SPREAD_LOW")
        return False
    if not filters['volume_ok']:
        log_rejection(symbol_name, "DYN_VOLUME_LOW")
        return False
        
    # التحقق من أن الاتجاه العام ليس هبوطيًا
    if mtf_trend.get('15m') == 'bearish' or mtf_trend.get('1h') == 'bearish':
        log_rejection(symbol_name, "EMA_RSI: Bearish long-term trend")
        return False
        
    return True

def check_bb_stoch_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 50: return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # الشرط الأساسي: السعر يلامس أو يقترب من الحد السفلي لبولينجر
    price_near_lower_bb = last['close'] <= last['bb_lower'] * 1.01
    
    # الشرط الثاني: ستوكاستيك في منطقة تشبع البيع وبدأ في الصعود
    stoch_oversold = last['stoch_k'] < 25 and prev['stoch_k'] < last['stoch_k']
    
    # الشرط الثالث: السعر فوق EMA50 (للتأكد من أن الاتجاه العام صاعد)
    price_above_ema50 = last['close'] > last['ema50']
    
    if not (price_near_lower_bb and stoch_oversold and price_above_ema50):
        if not price_near_lower_bb:
            log_rejection(symbol_name, "BB: Price not near lower band")
        if not stoch_oversold:
            log_rejection(symbol_name, "DYN_STOCH_LOW")
        if not price_above_ema50:
            log_rejection(symbol_name, "BB: Price below EMA50 (bearish trend)")
        return False
    
    # التحقق من الفلاتر الديناميكية
    filters = check_bb_stoch_dynamic_filters(df)
    if not filters['bb_width_ok']:
        log_rejection(symbol_name, "DYN_BB_WIDTH_LOW")
        return False
    if not filters['stoch_ok']:
        log_rejection(symbol_name, "DYN_STOCH_LOW")
        return False
    if not filters['volume_ok']:
        log_rejection(symbol_name, "DYN_VOLUME_LOW")
        return False
    
    return True

def check_macd_ema_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 50: return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # الشرط الأساسي: تقاطع MACD إيجابي
    macd_bullish_cross = (last['macd'] > last['macd_signal'] and 
                          prev['macd'] <= prev['macd_signal'] and 
                          last['macd_hist'] > 0)
    
    # الشرط الثاني: السعر فوق المتوسطات المتحركة الرئيسية
    price_above_emas = (last['close'] > last['ema21'] and 
                        last['close'] > last['ema50'])
    
    if not (macd_bullish_cross and price_above_emas):
        if not macd_bullish_cross:
            log_rejection(symbol_name, "MACD Momentum Negative")
        if not price_above_emas:
            log_rejection(symbol_name, "MACD: Strongly bearish trend")
        return False
    
    # التحقق من الفلاتر الديناميكية
    filters = check_macd_ema_dynamic_filters(df)
    if not filters['adx_ok']:
        log_rejection(symbol_name, "DYN_ADX_LOW")
        return False
    if not filters['volume_ok']:
        log_rejection(symbol_name, "DYN_VOLUME_LOW")
        return False
    if not filters['momentum_ok']:
        log_rejection(symbol_name, "DYN_MACD_MOMENTUM_LOW")
        return False
    
    return True

def check_pullback_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 50: return False
    
    last = df.iloc[-1]
    
    # الشرط الأساسي: اتجاه صاعد في الفريمات الأعلى
    if not (mtf_trend.get('5m') == 'bullish' and mtf_trend.get('15m') == 'bullish'):
        log_rejection(symbol_name, "Pullback: Trend is not strongly bullish")
        return False
    
    # الشرط الثاني: ارتداد عن مستوى مقاومة
    recent_high = df['high'].tail(10).max()
    pullback_occurred = (last['close'] < recent_high * 0.98 and 
                         last['close'] > recent_high * 0.95)
    
    if not pullback_occurred:
        log_rejection(symbol_name, "DYN_PULLBACK_SHALLOW")
        return False
    
    # التحقق من الفلاتر الديناميكية
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
    if len(df) < 50: return False
    
    last = df.iloc[-1]
    
    # الشرط الأساسي: المتوسطات المتحركة بترتيب صاعد
    emas_bullish_order = (last['ema9'] > last['ema21'] > last['ema50'] > last['ema200'])
    
    if not emas_bullish_order:
        log_rejection(symbol_name, "Momentum: EMAs not in bullish order")
        return False
    
    # التحقق من الفلاتر الديناميكية
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
    if len(df) < 100: return False
    
    last = df.iloc[-1]
    
    # الشرط الأساسي: اتجاه عام صاعد
    if mtf_trend.get('15m') == 'bearish' or mtf_trend.get('1h') == 'bearish':
        log_rejection(symbol_name, "Elliott Wave: Strongly bearish trend")
        return False
    
    # التحقق من وجود نقاط تذبذب كافية
    highs = df['high'].values
    lows = df['low'].values
    peaks_idx = argrelextrema(highs, np.greater, order=5)[0]
    troughs_idx = argrelextrema(lows, np.less, order=5)[0]
    
    if len(peaks_idx) < 2 or len(troughs_idx) < 2:
        log_rejection(symbol_name, "Elliott Wave: Insufficient swing points")
        return False
    
    # التحقق من الفلاتر الديناميكية
    filters = check_elliott_wave_dynamic_filters(df)
    if not filters['fibonacci_ok']:
        log_rejection(symbol_name, "DYN_FIB_RETRACEMENT_OOR")
        return False
    if not filters['volume_ok']:
        log_rejection(symbol_name, "DYN_VOLUME_LOW")
        return False
    if not filters['momentum_ok']:
        log_rejection(symbol_name, "DYN_MOMENTUM_SCORE_LOW")
        return False
    
    return True

def check_range_reversal_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 50: return False
    
    last = df.iloc[-1]
    
    # الشرط الأساسي: RSI في منطقة تشبع البيع
    rsi_oversold = last['rsi'] < 35
    
    # الشرط الثاني: السعر قرب الحد السفلي لبولينجر
    price_near_lower_bb = last['close'] <= last['bb_lower'] * 1.01
    
    if not (rsi_oversold and price_near_lower_bb):
        if not rsi_oversold:
            log_rejection(symbol_name, "Range Reversal: RSI not in oversold zone")
        if not price_near_lower_bb:
            log_rejection(symbol_name, "Range Reversal: Price not near lower band")
        return False
    
    # التحقق من الفلاتر الديناميكية
    filters = check_range_reversal_dynamic_filters(df)
    if not filters['adx_ok']:
        log_rejection(symbol_name, "Range Reversal: Trend too strong (ADX > 23)")
        return False
    if not filters['rsi_ok']:
        log_rejection(symbol_name, "Range Reversal: RSI not in oversold zone")
        return False
    
    return True

# --- دوال التداول الأساسية ---
def generate_signals():
    global is_trading_enabled, client, validated_symbols_to_scan
    
    if not is_trading_enabled:
        logger.info("[Signals] Trading is disabled. Skipping signal generation.")
        return
    
    if not client or not validated_symbols_to_scan:
        logger.warning("[Signals] Client or symbols not initialized. Skipping signal generation.")
        return
    
    logger.info("[Signals] Starting signal generation cycle...")
    
    # تحديث حالة السوق قبل توليد الإشارات
    update_market_state()
    
    # تحميل الإعدادات من Redis
    load_settings_from_redis()
    
    # الحصول على حالة التداول
    with trading_status_lock:
        trading_enabled = is_trading_enabled
    
    with trading_mode_lock:
        is_paper = paper_trading_mode
    
    # الحصول على الرصيد
    with balance_lock:
        balance = usdt_balance
    
    # الحصول على الصفقات المفتوحة
    with signal_cache_lock:
        open_signals = dict(open_signals_cache)
    
    # تحقق من عدد الصفقات المفتوحة
    if len(open_signals) >= MAX_OPEN_TRADES:
        logger.info(f"[Signals] Maximum number of open trades ({MAX_OPEN_TRADES}) reached. Skipping signal generation.")
        return
    
    # حساب عدد الصفقات الجديدة المسموح بها
    new_trades_allowed = MAX_OPEN_TRADES - len(open_signals)
    
    # تحديث حالة السوق
    update_market_state()
    
    # الحصول على حالة السوق
    with market_state_lock:
        market_regime = current_market_state.get("market_regime", "unknown")
        volatility_state = current_market_state.get("volatility_state", "medium")
    
    # تعديل معلمات الاستراتيجيات بناءً على حالة السوق
    if market_regime == "volatile":
        # في الأسواق المتقلبة، قلل جودة الإشارة المطلوبة
        with min_quality_lock:
            required_quality = max(60, MIN_SIGNAL_QUALITY - 10)
    elif market_regime == "trending":
        # في الأسواق المتجهة، زد جودة الإشارة المطلوبة
        with min_quality_lock:
            required_quality = min(85, MIN_SIGNAL_QUALITY + 5)
    else:
        with min_quality_lock:
            required_quality = MIN_SIGNAL_QUALITY
    
    # معالجة العملات
    processed_symbols = 0
    new_signals = 0
    
    for symbol in validated_symbols_to_scan:
        if new_signals >= new_trades_allowed:
            break
        
        if symbol in open_signals:
            continue
        
        try:
            # جلب البيانات
            df_5m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
            if df_5m is None or len(df_5m) < 50:
                log_rejection(symbol, "Insufficient Historical Data")
                continue
            
            # حساب المؤشرات
            df_5m = calculate_all_features(df_5m)
            df_5m.name = symbol
            
            # جلب بيانات الفريم الأعلى
            df_15m = fetch_historical_data(symbol, HIGHER_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS * 2)
            if df_15m is not None:
                df_15m = calculate_all_features(df_15m)
            
            # تحليل الاتجاه متعدد الفريمات
            mtf_trend = {}
            
            # تحليل اتجاه 5m
            last_5m = df_5m.iloc[-1]
            if last_5m['ema50'] > last_5m['ema200'] and last_5m['close'] > last_5m['ema50']:
                mtf_trend['5m'] = 'bullish'
            elif last_5m['ema50'] < last_5m['ema200'] and last_5m['close'] < last_5m['ema50']:
                mtf_trend['5m'] = 'bearish'
            else:
                mtf_trend['5m'] = 'neutral'
            
            # تحليل اتجاه 15m
            if df_15m is not None and len(df_15m) > 0:
                last_15m = df_15m.iloc[-1]
                if last_15m['ema50'] > last_15m['ema200'] and last_15m['close'] > last_15m['ema50']:
                    mtf_trend['15m'] = 'bullish'
                elif last_15m['ema50'] < last_15m['ema200'] and last_15m['close'] < last_15m['ema50']:
                    mtf_trend['15m'] = 'bearish'
                else:
                    mtf_trend['15m'] = 'neutral'
            
            # التحقق من فلتر تقلب السوق
            if not check_market_volatility_filter_enhanced(df_5m, symbol):
                continue
            
            # التحقق من فلتر الأخبار
            if not add_news_filter():
                log_rejection(symbol, "News Filter Failed")
                continue
            
            # التحقق من فلتر السيولة
            if not add_liquidity_filter():
                log_rejection(symbol, "Liquidity Filter Failed")
                continue
            
            # التحقق من فلتر الارتباط
            if not add_correlation_filter(symbol):
                log_rejection(symbol, "Correlation Filter Failed")
                continue
            
            # تقييم الاستراتيجيات
            strategy_signals = {}
            
            if USE_BB_STOCH_STRATEGY:
                strategy_signals["BB_Stoch_Strategy"] = check_bb_stoch_strategy_enhanced(df_5m, mtf_trend)
            
            if USE_MACD_EMA_STRATEGY:
                strategy_signals["MACD_EMA_Strategy"] = check_macd_ema_strategy_enhanced(df_5m, mtf_trend)
            
            if USE_EMA_RSI_STRATEGY:
                strategy_signals["EMA_RSI_Strategy"] = check_ema_rsi_strategy_enhanced(df_5m, mtf_trend)
            
            if USE_PULLBACK_STRATEGY:
                strategy_signals["Pullback_Strategy"] = check_pullback_strategy_enhanced(df_5m, mtf_trend)
            
            if USE_MOMENTUM_VOLATILITY_STRATEGY:
                strategy_signals["Momentum_Volatility_Strategy"] = check_momentum_volatility_strategy_enhanced(df_5m, mtf_trend)
            
            if USE_ELLIOTT_WAVE_STRATEGY:
                strategy_signals["Elliott_Wave_Strategy"] = check_elliott_wave_strategy_enhanced(df_5m, mtf_trend)
            
            if USE_RANGE_REVERSAL_STRATEGY:
                strategy_signals["Range_Reversal_Strategy"] = check_range_reversal_strategy_enhanced(df_5m, mtf_trend)
            
            # حساب جودة الإشارة
            active_strategies = sum(1 for signal in strategy_signals.values() if signal)
            quality_score = min(100, int((active_strategies / len(strategy_signals)) * 100)) if strategy_signals else 0
            
            # التحقق من جودة الإشارة
            if quality_score < required_quality:
                log_rejection(symbol, "Low Quality Signal", {"quality": f"{quality_score}/{required_quality}"})
                continue
            
            # اختيار أفضل استراتيجية
            active_strategy_names = [name for name, active in strategy_signals.items() if active]
            if not active_strategy_names:
                continue
            
            # اختيار الاستراتيجية الأولى النشطة (يمكن تحسين هذا الاختيار)
            strategy_name = active_strategy_names[0]
            
            # حساب نقاط الدخول والخروج
            entry_price = df_5m['close'].iloc[-1]
            stop_loss = calculate_dynamic_stop_loss_enhanced(df_5m, entry_price, strategy_name)
            target1, target2 = calculate_dynamic_take_profit_enhanced(df_5m, entry_price, stop_loss, strategy_name)
            
            # التحقق من صحة نقاط الدخول والخروج
            if stop_loss >= entry_price:
                log_rejection(symbol, "Invalid Position Size")
                continue
            
            # حساب حجم الصفقة
            if is_paper:
                trade_amount = PAPER_TRADE_FIXED_AMOUNT_USDT
            else:
                with trade_amount_lock:
                    trade_amount = random.uniform(FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT)
            
            # التحقق من الرصيد
            if not is_paper and balance < trade_amount:
                if AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE:
                    with trading_mode_lock:
                        paper_trading_mode = True
                        is_paper = True
                    trade_amount = PAPER_TRADE_FIXED_AMOUNT_USDT
                    log_and_notify("warning", f"Low balance (${balance:.2f}), falling back to paper trading for {symbol}", "balance_warning")
                else:
                    log_rejection(symbol, "Insufficient Balance", {"balance": f"${balance:.2f}", "required": f"${trade_amount:.2f}"})
                    continue
            
            # حساب الكمية
            quantity = trade_amount / entry_price
            
            # التحقق من متطلبات المنصة
            symbol_info = exchange_info_map.get(symbol, {})
            if not symbol_info:
                log_rejection(symbol, "Symbol info not found")
                continue
            
            # تطبيق قيود المنصة
            for filter_type in ['LOT_SIZE', 'MIN_NOTIONAL']:
                filter_data = next((f for f in symbol_info.get('filters', []) if f.get('filterType') == filter_type), {})
                if not filter_data:
                    continue
                
                if filter_type == 'LOT_SIZE':
                    min_qty = float(filter_data.get('minQty', 0))
                    max_qty = float(filter_data.get('maxQty', float('inf')))
                    step_size = float(filter_data.get('stepSize', 0.00000001))
                    
                    # تعديل الكمية لتناسب قيود المنصة
                    quantity = max(min_qty, min(max_qty, quantity))
                    quantity = round(quantity / step_size) * step_size
                
                elif filter_type == 'MIN_NOTIONAL':
                    min_notional = float(filter_data.get('minNotional', 0))
                    notional_value = quantity * entry_price
                    
                    if notional_value < min_notional:
                        log_rejection(symbol, "MinNotional Filter Failed", {
                            "notional": f"${notional_value:.2f}",
                            "min_required": f"${min_notional:.2f}"
                        })
                        continue
            
            # حساب القيمة الاسمية للصفقة
            notional_value = quantity * entry_price
            
            # حفظ الإشارة في قاعدة البيانات
            if not check_db_connection() or not conn:
                continue
            
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO signals (symbol, entry_price, stop_loss, target_price_1, target_price_2, 
                                          strategy_name, signal_details, is_real_trade, quantity, 
                                          initial_quantity)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id;
                    """, (
                        symbol, entry_price, stop_loss, target1, target2,
                        strategy_name, json.dumps({
                            "quality_score": quality_score,
                            "atr_percent": df_5m['atr_percent'].iloc[-1],
                            "market_regime": market_regime,
                            "volatility_state": volatility_state,
                            "mtf_trend": mtf_trend,
                            "active_strategies": active_strategy_names
                        }), not is_paper, quantity, quantity
                    ))
                    
                    signal_id = cur.fetchone()['id']
                    conn.commit()
                
                # تحديث الكاش
                with signal_cache_lock:
                    open_signals_cache[symbol] = {
                        "id": signal_id,
                        "symbol": symbol,
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "target_price_1": target1,
                        "target_price_2": target2,
                        "strategy_name": strategy_name,
                        "status": "open",
                        "is_real_trade": not is_paper,
                        "quantity": quantity,
                        "initial_quantity": quantity
                    }
                
                # إرسال إشعار
                send_trade_open_notification(
                    symbol, strategy_name, entry_price, stop_loss, target1, target2,
                    quantity, not is_paper, quality_score, df_5m['atr_percent'].iloc[-1], notional_value
                )
                
                # تحديث الرصيد للصفقات الحقيقية
                if not is_paper:
                    with balance_lock:
                        usdt_balance -= notional_value
                
                new_signals += 1
                logger.info(f"✅ [Signals] New signal for {symbol} using {strategy_name} (Quality: {quality_score}%)")
                
            except Exception as e:
                logger.error(f"❌ [Signals] Error saving signal for {symbol}: {e}")
                if conn: conn.rollback()
            
            processed_symbols += 1
            
        except Exception as e:
            logger.error(f"❌ [Signals] Error processing {symbol}: {e}", exc_info=True)
    
    logger.info(f"[Signals] Signal generation completed. Processed {processed_symbols} symbols, found {new_signals} new signals.")

# --- دوال Flask ---
@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crypto Trading Bot Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; }
                .sidebar { background-color: #343a40; color: white; min-height: 100vh; }
                .main-content { padding: 20px; }
                .card { margin-bottom: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                .card-header { font-weight: bold; border-radius: 10px 10px 0 0 !important; }
                .trade-card { border-left: 4px solid; }
                .profit { border-left-color: #28a745; }
                .loss { border-left-color: #dc3545; }
                .neutral { border-left-color: #6c757d; }
                .signal-quality { font-weight: bold; }
                .quality-high { color: #28a745; }
                .quality-medium { color: #ffc107; }
                .quality-low { color: #dc3545; }
                .market-regime { font-weight: bold; padding: 5px 10px; border-radius: 20px; }
                .regime-trending { background-color: #d4edda; color: #155724; }
                .regime-ranging { background-color: #fff3cd; color: #856404; }
                .regime-volatile { background-color: #f8d7da; color: #721c24; }
                .volatility-low { color: #28a745; }
                .volatility-medium { color: #ffc107; }
                .volatility-high { color: #dc3545; }
                .trend-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 5px; }
                .trend-bullish { background-color: #28a745; }
                .trend-bearish { background-color: #dc3545; }
                .trend-neutral { background-color: #6c757d; }
                .price-up { color: #28a745; }
                .price-down { color: #dc3545; }
                .notification-item { padding: 10px; margin-bottom: 10px; border-radius: 5px; }
                .notification-info { background-color: #d1ecf1; border-left: 4px solid #17a2b8; }
                .notification-warning { background-color: #fff3cd; border-left: 4px solid #ffc107; }
                .notification-error { background-color: #f8d7da; border-left: 4px solid #dc3545; }
                .rejection-item { padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: #f8f9fa; border-left: 4px solid #6c757d; }
                .status-indicator { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }
                .status-active { background-color: #28a745; }
                .status-inactive { background-color: #dc3545; }
                .tab-content { padding: 20px; }
                .nav-tabs .nav-link { color: #495057; }
                .nav-tabs .nav-link.active { font-weight: bold; }
                .settings-section { margin-bottom: 20px; }
                .strategy-switch { margin-right: 10px; }
            </style>
        </head>
        <body>
            <div class="container-fluid">
                <div class="row">
                    <!-- Sidebar -->
                    <div class="col-md-3 col-lg-2 sidebar p-3">
                        <h4 class="mb-4">Trading Bot</h4>
                        <ul class="nav flex-column">
                            <li class="nav-item">
                                <a class="nav-link active text-white" href="#" data-tab="dashboard">
                                    <i class="bi bi-speedometer2 me-2"></i> Dashboard
                                </a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link text-white" href="#" data-tab="trades">
                                    <i class="bi bi-graph-up me-2"></i> Open Trades
                                </a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link text-white" href="#" data-tab="market">
                                    <i class="bi bi-globe me-2"></i> Market State
                                </a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link text-white" href="#" data-tab="notifications">
                                    <i class="bi bi-bell me-2"></i> Notifications
                                </a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link text-white" href="#" data-tab="rejections">
                                    <i class="bi bi-x-circle me-2"></i> Rejections
                                </a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link text-white" href="#" data-tab="settings">
                                    <i class="bi bi-gear me-2"></i> Settings
                                </a>
                            </li>
                        </ul>
                    </div>
                    
                    <!-- Main Content -->
                    <div class="col-md-9 col-lg-10 main-content">
                        <!-- Dashboard Tab -->
                        <div id="dashboard-tab" class="tab-content">
                            <div class="d-flex justify-content-between align-items-center mb-4">
                                <h2>Dashboard</h2>
                                <div>
                                    <span id="trading-status" class="status-indicator status-inactive"></span>
                                    <span id="trading-status-text">Trading Disabled</span>
                                    <button id="toggle-trading" class="btn btn-sm btn-primary ms-2">Enable Trading</button>
                                </div>
                            </div>
                            
                            <div class="row">
                                <div class="col-md-3">
                                    <div class="card text-center">
                                        <div class="card-header bg-primary text-white">Balance</div>
                                        <div class="card-body">
                                            <h5 id="balance">$0.00</h5>
                                            <small id="balance-status">Paper Trading</small>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="card text-center">
                                        <div class="card-header bg-success text-white">Open Trades</div>
                                        <div class="card-body">
                                            <h5 id="open-trades-count">0</h5>
                                            <small>Max: <span id="max-trades">3</span></small>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="card text-center">
                                        <div class="card-header bg-info text-white">Market Regime</div>
                                        <div class="card-body">
                                            <h5 id="market-regime">Unknown</h5>
                                            <small id="volatility-state">Medium Volatility</small>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="card text-center">
                                        <div class="card-header bg-warning text-white">Signal Quality</div>
                                        <div class="card-body">
                                            <h5 id="signal-quality">70%</h5>
                                            <small>Minimum Required</small>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="row mt-4">
                                <div class="col-md-6">
                                    <div class="card">
                                        <div class="card-header">Performance Summary</div>
                                        <div class="card-body">
                                            <canvas id="performance-chart" height="150"></canvas>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="card">
                                        <div class="card-header">Strategy Distribution</div>
                                        <div class="card-body">
                                            <canvas id="strategy-chart" height="150"></canvas>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Open Trades Tab -->
                        <div id="trades-tab" class="tab-content" style="display: none;">
                            <h2 class="mb-4">Open Trades</h2>
                            <div id="open-trades-container">
                                <div class="text-center py-5">
                                    <div class="spinner-border text-primary" role="status">
                                        <span class="visually-hidden">Loading...</span>
                                    </div>
                                    <p class="mt-2">Loading open trades...</p>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Market State Tab -->
                        <div id="market-tab" class="tab-content" style="display: none;">
                            <h2 class="mb-4">Market State</h2>
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="card">
                                        <div class="card-header">Trend Analysis</div>
                                        <div class="card-body">
                                            <div id="trend-analysis">
                                                <div class="text-center py-5">
                                                    <div class="spinner-border text-primary" role="status">
                                                        <span class="visually-hidden">Loading...</span>
                                                    </div>
                                                    <p class="mt-2">Loading market analysis...</p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="card">
                                        <div class="card-header">Volatility Analysis</div>
                                        <div class="card-body">
                                            <canvas id="volatility-chart" height="200"></canvas>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Notifications Tab -->
                        <div id="notifications-tab" class="tab-content" style="display: none;">
                            <h2 class="mb-4">Notifications</h2>
                            <div id="notifications-container">
                                <div class="text-center py-5">
                                    <div class="spinner-border text-primary" role="status">
                                        <span class="visually-hidden">Loading...</span>
                                    </div>
                                    <p class="mt-2">Loading notifications...</p>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Rejections Tab -->
                        <div id="rejections-tab" class="tab-content" style="display: none;">
                            <h2 class="mb-4">Signal Rejections</h2>
                            <div id="rejections-container">
                                <div class="text-center py-5">
                                    <div class="spinner-border text-primary" role="status">
                                        <span class="visually-hidden">Loading...</span>
                                    </div>
                                    <p class="mt-2">Loading rejections...</p>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Settings Tab -->
                        <div id="settings-tab" class="tab-content" style="display: none;">
                            <h2 class="mb-4">Settings</h2>
                            
                            <div class="card settings-section">
                                <div class="card-header">Trading Settings</div>
                                <div class="card-body">
                                    <div class="row mb-3">
                                        <div class="col-md-6">
                                            <div class="form-check form-switch">
                                                <input class="form-check-input" type="checkbox" id="paper-trading-switch" checked>
                                                <label class="form-check-label" for="paper-trading-switch">
                                                    Paper Trading Mode
                                                </label>
                                            </div>
                                        </div>
                                        <div class="col-md-6">
                                            <div class="form-check form-switch">
                                                <input class="form-check-input" type="checkbox" id="auto-fallback-switch" checked>
                                                <label class="form-check-label" for="auto-fallback-switch">
                                                    Auto Fallback to Paper on Low Balance
                                                </label>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div class="row mb-3">
                                        <div class="col-md-6">
                                            <label for="min-trade-amount" class="form-label">Min Trade Amount (USDT)</label>
                                            <input type="number" class="form-control" id="min-trade-amount" value="4.5" step="0.1">
                                        </div>
                                        <div class="col-md-6">
                                            <label for="max-trade-amount" class="form-label">Max Trade Amount (USDT)</label>
                                            <input type="number" class="form-control" id="max-trade-amount" value="6.5" step="0.1">
                                        </div>
                                    </div>
                                    
                                    <div class="row mb-3">
                                        <div class="col-md-6">
                                            <label for="max-open-trades" class="form-label">Max Open Trades</label>
                                            <input type="number" class="form-control" id="max-open-trades" value="3" min="1" max="10">
                                        </div>
                                        <div class="col-md-6">
                                            <label for="signal-quality" class="form-label">Minimum Signal Quality (%)</label>
                                            <input type="number" class="form-control" id="signal-quality" value="70" min="50" max="100">
                                        </div>
                                    </div>
                                    
                                    <div class="row">
                                        <div class="col-12">
                                            <button id="save-trading-settings" class="btn btn-primary">Save Settings</button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="card settings-section">
                                <div class="card-header">Strategy Settings</div>
                                <div class="card-body">
                                    <div class="row">
                                        <div class="col-md-6">
                                            <div class="form-check form-switch strategy-switch">
                                                <input class="form-check-input" type="checkbox" id="bb-stoch-switch" checked>
                                                <label class="form-check-label" for="bb-stoch-switch">
                                                    BB + Stoch Strategy
                                                </label>
                                            </div>
                                        </div>
                                        <div class="col-md-6">
                                            <div class="form-check form-switch strategy-switch">
                                                <input class="form-check-input" type="checkbox" id="macd-ema-switch" checked>
                                                <label class="form-check-label" for="macd-ema-switch">
                                                    MACD + EMA Strategy
                                                </label>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div class="row mt-3">
                                        <div class="col-md-6">
                                            <div class="form-check form-switch strategy-switch">
                                                <input class="form-check-input" type="checkbox" id="ema-rsi-switch" checked>
                                                <label class="form-check-label" for="ema-rsi-switch">
                                                    EMA + RSI Strategy
                                                </label>
                                            </div>
                                        </div>
                                        <div class="col-md-6">
                                            <div class="form-check form-switch strategy-switch">
                                                <input class="form-check-input" type="checkbox" id="pullback-switch" checked>
                                                <label class="form-check-label" for="pullback-switch">
                                                    Pullback Strategy
                                                </label>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div class="row mt-3">
                                        <div class="col-md-6">
                                            <div class="form-check form-switch strategy-switch">
                                                <input class="form-check-input" type="checkbox" id="momentum-switch" checked>
                                                <label class="form-check-label" for="momentum-switch">
                                                    Momentum Strategy
                                                </label>
                                            </div>
                                        </div>
                                        <div class="col-md-6">
                                            <div class="form-check form-switch strategy-switch">
                                                <input class="form-check-input" type="checkbox" id="elliott-switch" checked>
                                                <label class="form-check-label" for="elliott-switch">
                                                    Elliott Wave Strategy
                                                </label>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div class="row mt-3">
                                        <div class="col-md-6">
                                            <div class="form-check form-switch strategy-switch">
                                                <input class="form-check-input" type="checkbox" id="range-switch" checked>
                                                <label class="form-check-label" for="range-switch">
                                                    Range Reversal Strategy
                                                </label>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div class="row mt-3">
                                        <div class="col-12">
                                            <button id="save-strategy-settings" class="btn btn-primary">Save Settings</button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="card settings-section">
                                <div class="card-header">Notification Settings</div>
                                <div class="card-body">
                                    <div class="row mb-3">
                                        <div class="col-md-6">
                                            <div class="form-check form-switch">
                                                <input class="form-check-input" type="checkbox" id="telegram-notifications-switch" checked>
                                                <label class="form-check-label" for="telegram-notifications-switch">
                                                    Telegram Notifications
                                                </label>
                                            </div>
                                        </div>
                                        <div class="col-md-6">
                                            <div class="form-check form-switch">
                                                <input class="form-check-input" type="checkbox" id="email-notifications-switch">
                                                <label class="form-check-label" for="email-notifications-switch">
                                                    Email Notifications
                                                </label>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div class="row mb-3">
                                        <div class="col-md-6">
                                            <label for="min-profit-notification" class="form-label">Min Profit for Notification (%)</label>
                                            <input type="number" class="form-control" id="min-profit-notification" value="1.0" step="0.1">
                                        </div>
                                        <div class="col-md-6">
                                            <label for="max-loss-notification" class="form-label">Max Loss for Notification (%)</label>
                                            <input type="number" class="form-control" id="max-loss-notification" value="-1.0" step="0.1">
                                        </div>
                                    </div>
                                    
                                    <div class="row">
                                        <div class="col-12">
                                            <button id="save-notification-settings" class="btn btn-primary">Save Settings</button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script>
                // Global variables
                let socket;
                let openTrades = {};
                let notifications = [];
                let rejections = [];
                let marketState = {};
                let performanceChart = null;
                let strategyChart = null;
                let volatilityChart = null;
                
                // Initialize WebSocket connection
                function initWebSocket() {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${protocol}//${window.location.host}/ws`;
                    
                    socket = new WebSocket(wsUrl);
                    
                    socket.onopen = function(e) {
                        console.log("WebSocket connection established");
                        
                        // Request initial data
                        socket.send(JSON.stringify({type: "get_initial_data"}));
                    };
                    
                    socket.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        handleWebSocketMessage(data);
                    };
                    
                    socket.onclose = function(event) {
                        console.log("WebSocket connection closed. Reconnecting in 5 seconds...");
                        setTimeout(initWebSocket, 5000);
                    };
                    
                    socket.onerror = function(error) {
                        console.error("WebSocket error:", error);
                    };
                }
                
                // Handle WebSocket messages
                function handleWebSocketMessage(data) {
                    switch(data.type) {
                        case "initial_data":
                            updateDashboard(data.payload);
                            updateOpenTrades(data.payload.open_trades);
                            updateNotifications(data.payload.notifications);
                            updateRejections(data.payload.rejections);
                            updateMarketState(data.payload.market_state);
                            break;
                            
                        case "trade_update":
                            updateTrade(data.payload);
                            break;
                            
                        case "new_trade":
                            addNewTrade(data.payload);
                            break;
                            
                        case "closed_trade":
                            removeTrade(data.payload.symbol);
                            break;
                            
                        case "new_notification":
                            addNotification(data.payload);
                            break;
                            
                        case "new_rejection":
                            addRejection(data.payload);
                            break;
                            
                        case "market_state_update":
                            updateMarketState(data.payload);
                            break;
                            
                        case "price_update":
                            updatePrices(data.payload);
                            break;
                            
                        case "settings_updated":
                            showSettingsUpdated();
                            break;
                    }
                }
                
                // Update dashboard with initial data
                function updateDashboard(data) {
                    // Update balance
                    document.getElementById('balance').textContent = `$${data.balance.toFixed(2)}`;
                    document.getElementById('balance-status').textContent = data.paper_trading ? 'Paper Trading' : 'Real Trading';
                    
                    // Update trading status
                    const tradingStatus = document.getElementById('trading-status');
                    const tradingStatusText = document.getElementById('trading-status-text');
                    const toggleTradingBtn = document.getElementById('toggle-trading');
                    
                    if (data.trading_enabled) {
                        tradingStatus.classList.remove('status-inactive');
                        tradingStatus.classList.add('status-active');
                        tradingStatusText.textContent = 'Trading Enabled';
                        toggleTradingBtn.textContent = 'Disable Trading';
                        toggleTradingBtn.classList.remove('btn-primary');
                        toggleTradingBtn.classList.add('btn-danger');
                    } else {
                        tradingStatus.classList.remove('status-active');
                        tradingStatus.classList.add('status-inactive');
                        tradingStatusText.textContent = 'Trading Disabled';
                        toggleTradingBtn.textContent = 'Enable Trading';
                        toggleTradingBtn.classList.remove('btn-danger');
                        toggleTradingBtn.classList.add('btn-primary');
                    }
                    
                    // Update open trades count
                    document.getElementById('open-trades-count').textContent = data.open_trades_count;
                    document.getElementById('max-trades').textContent = data.max_trades;
                    
                    // Update market regime
                    const marketRegimeEl = document.getElementById('market-regime');
                    const volatilityStateEl = document.getElementById('volatility-state');
                    
                    marketRegimeEl.textContent = data.market_state.market_regime.charAt(0).toUpperCase() + data.market_state.market_regime.slice(1);
                    marketRegimeEl.className = `market-regime regime-${data.market_state.market_regime}`;
                    
                    volatilityStateEl.textContent = `${data.market_state.volatility_state.charAt(0).toUpperCase() + data.market_state.volatility_state.slice(1)} Volatility`;
                    volatilityStateEl.className = `volatility-${data.market_state.volatility_state}`;
                    
                    // Update signal quality
                    document.getElementById('signal-quality').textContent = `${data.signal_quality}%`;
                    
                    // Update settings
                    document.getElementById('paper-trading-switch').checked = data.paper_trading;
                    document.getElementById('auto-fallback-switch').checked = data.auto_fallback;
                    document.getElementById('min-trade-amount').value = data.min_trade_amount;
                    document.getElementById('max-trade-amount').value = data.max_trade_amount;
                    document.getElementById('max-open-trades').value = data.max_trades;
                    document.getElementById('signal-quality').value = data.signal_quality;
                    
                    // Update strategy settings
                    document.getElementById('bb-stoch-switch').checked = data.strategies.USE_BB_STOCH_STRATEGY;
                    document.getElementById('macd-ema-switch').checked = data.strategies.USE_MACD_EMA_STRATEGY;
                    document.getElementById('ema-rsi-switch').checked = data.strategies.USE_EMA_RSI_STRATEGY;
                    document.getElementById('pullback-switch').checked = data.strategies.USE_PULLBACK_STRATEGY;
                    document.getElementById('momentum-switch').checked = data.strategies.USE_MOMENTUM_VOLATILITY_STRATEGY;
                    document.getElementById('elliott-switch').checked = data.strategies.USE_ELLIOTT_WAVE_STRATEGY;
                    document.getElementById('range-switch').checked = data.strategies.USE_RANGE_REVERSAL_STRATEGY;
                    
                    // Update notification settings
                    document.getElementById('telegram-notifications-switch').checked = data.notification_settings.telegram_enabled;
                    document.getElementById('email-notifications-switch').checked = data.notification_settings.email_enabled;
                    document.getElementById('min-profit-notification').value = data.notification_settings.min_profit_notification;
                    document.getElementById('max-loss-notification').value = data.notification_settings.max_loss_notification;
                    
                    // Initialize charts
                    initPerformanceChart(data.performance_data);
                    initStrategyChart(data.strategy_distribution);
                    initVolatilityChart(data.volatility_data);
                }
                
                // Update open trades
                function updateOpenTrades(trades) {
                    openTrades = {};
                    
                    const container = document.getElementById('open-trades-container');
                    container.innerHTML = '';
                    
                    if (trades.length === 0) {
                        container.innerHTML = '<div class="text-center py-5"><p>No open trades</p></div>';
                        return;
                    }
                    
                    trades.forEach(trade => {
                        openTrades[trade.symbol] = trade;
                        const tradeCard = createTradeCard(trade);
                        container.appendChild(tradeCard);
                    });
                }
                
                // Create a trade card element
                function createTradeCard(trade) {
                    const card = document.createElement('div');
                    card.className = `card trade-card ${trade.profit_percent >= 0 ? 'profit' : 'loss'}`;
                    card.id = `trade-${trade.symbol}`;
                    
                    const profitClass = trade.profit_percent >= 0 ? 'price-up' : 'price-down';
                    const profitSign = trade.profit_percent >= 0 ? '+' : '';
                    
                    card.innerHTML = `
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <span>${trade.symbol}</span>
                            <span class="${profitClass}">${profitSign}${trade.profit_percent.toFixed(2)}%</span>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <p><strong>Strategy:</strong> ${trade.strategy_name}</p>
                                    <p><strong>Entry Price:</strong> $${trade.entry_price.toFixed(4)}</p>
                                    <p><strong>Current Price:</strong> $${trade.current_price.toFixed(4)}</p>
                                    <p><strong>Stop Loss:</strong> $${trade.stop_loss.toFixed(4)}</p>
                                </div>
                                <div class="col-md-6">
                                    <p><strong>Target 1:</strong> $${trade.target_price_1.toFixed(4)}</p>
                                    <p><strong>Target 2:</strong> $${trade.target_price_2.toFixed(4)}</p>
                                    <p><strong>Quantity:</strong> ${trade.quantity.toFixed(4)}</p>
                                    <p><strong>Notional Value:</strong> $${(trade.quantity * trade.entry_price).toFixed(2)}</p>
                                </div>
                            </div>
                            <div class="progress mt-3" style="height: 10px;">
                                <div class="progress-bar ${trade.profit_percent >= 0 ? 'bg-success' : 'bg-danger'}" 
                                     role="progressbar" 
                                     style="width: ${Math.min(100, Math.max(0, trade.profit_percent + 50))}%">
                                </div>
                            </div>
                        </div>
                    `;
                    
                    return card;
                }
                
                // Update a single trade
                function updateTrade(tradeData) {
                    if (!openTrades[tradeData.symbol]) {
                        return;
                    }
                    
                    // Update the trade in our local object
                    openTrades[tradeData.symbol] = {
                        ...openTrades[tradeData.symbol],
                        ...tradeData
                    };
                    
                    // Update the card if it exists
                    const card = document.getElementById(`trade-${tradeData.symbol}`);
                    if (card) {
                        const newCard = createTradeCard(openTrades[tradeData.symbol]);
                        card.parentNode.replaceChild(newCard, card);
                    }
                }
                
                // Add a new trade
                function addNewTrade(trade) {
                    openTrades[trade.symbol] = trade;
                    
                    const container = document.getElementById('open-trades-container');
                    
                    // Remove the "no trades" message if it exists
                    const noTradesMsg = container.querySelector('.text-center');
                    if (noTradesMsg && noTradesMsg.textContent.includes('No open trades')) {
                        container.innerHTML = '';
                    }
                    
                    const tradeCard = createTradeCard(trade);
                    container.appendChild(tradeCard);
                    
                    // Update the open trades count
                    const count = Object.keys(openTrades).length;
                    document.getElementById('open-trades-count').textContent = count;
                }
                
                // Remove a trade
                function removeTrade(symbol) {
                    delete openTrades[symbol];
                    
                    const card = document.getElementById(`trade-${symbol}`);
                    if (card) {
                        card.remove();
                    }
                    
                    // Check if there are any trades left
                    const container = document.getElementById('open-trades-container');
                    if (Object.keys(openTrades).length === 0) {
                        container.innerHTML = '<div class="text-center py-5"><p>No open trades</p></div>';
                    }
                    
                    // Update the open trades count
                    const count = Object.keys(openTrades).length;
                    document.getElementById('open-trades-count').textContent = count;
                }
                
                // Update notifications
                function updateNotifications(notificationList) {
                    notifications = notificationList;
                    
                    const container = document.getElementById('notifications-container');
                    container.innerHTML = '';
                    
                    if (notifications.length === 0) {
                        container.innerHTML = '<div class="text-center py-5"><p>No notifications</p></div>';
                        return;
                    }
                    
                    notifications.forEach(notification => {
                        const notificationEl = createNotificationElement(notification);
                        container.appendChild(notificationEl);
                    });
                }
                
                // Create a notification element
                function createNotificationElement(notification) {
                    const div = document.createElement('div');
                    div.className = `notification-item notification-${notification.type}`;
                    
                    const time = new Date(notification.timestamp).toLocaleString();
                    
                    div.innerHTML = `
                        <div class="d-flex justify-content-between">
                            <div>${notification.message}</div>
                            <small>${time}</small>
                        </div>
                    `;
                    
                    return div;
                }
                
                // Add a new notification
                function addNotification(notification) {
                    notifications.unshift(notification);
                    
                    const container = document.getElementById('notifications-container');
                    
                    // Remove the "no notifications" message if it exists
                    const noNotificationsMsg = container.querySelector('.text-center');
                    if (noNotificationsMsg && noNotificationsMsg.textContent.includes('No notifications')) {
                        container.innerHTML = '';
                    }
                    
                    const notificationEl = createNotificationElement(notification);
                    container.insertBefore(notificationEl, container.firstChild);
                    
                    // Keep only the latest 20 notifications
                    while (container.children.length > 20) {
                        container.removeChild(container.lastChild);
                    }
                }
                
                // Update rejections
                function updateRejections(rejectionList) {
                    rejections = rejectionList;
                    
                    const container = document.getElementById('rejections-container');
                    container.innerHTML = '';
                    
                    if (rejections.length === 0) {
                        container.innerHTML = '<div class="text-center py-5"><p>No rejections</p></div>';
                        return;
                    }
                    
                    rejections.forEach(rejection => {
                        const rejectionEl = createRejectionElement(rejection);
                        container.appendChild(rejectionEl);
                    });
                }
                
                // Create a rejection element
                function createRejectionElement(rejection) {
                    const div = document.createElement('div');
                    div.className = 'rejection-item';
                    
                    const time = new Date(rejection.timestamp).toLocaleString();
                    
                    div.innerHTML = `
                        <div class="d-flex justify-content-between">
                            <div>
                                <strong>${rejection.symbol}:</strong> ${rejection.reason}
                            </div>
                            <small>${time}</small>
                        </div>
                    `;
                    
                    return div;
                }
                
                // Add a new rejection
                function addRejection(rejection) {
                    rejections.unshift(rejection);
                    
                    const container = document.getElementById('rejections-container');
                    
                    // Remove the "no rejections" message if it exists
                    const noRejectionsMsg = container.querySelector('.text-center');
                    if (noRejectionsMsg && noRejectionsMsg.textContent.includes('No rejections')) {
                        container.innerHTML = '';
                    }
                    
                    const rejectionEl = createRejectionElement(rejection);
                    container.insertBefore(rejectionEl, container.firstChild);
                    
                    // Keep only the latest 30 rejections
                    while (container.children.length > 30) {
                        container.removeChild(container.lastChild);
                    }
                }
                
                // Update market state
                function updateMarketState(state) {
                    marketState = state;
                    
                    // Update market regime
                    const marketRegimeEl = document.getElementById('market-regime');
                    const volatilityStateEl = document.getElementById('volatility-state');
                    
                    if (marketRegimeEl) {
                        marketRegimeEl.textContent = state.market_regime.charAt(0).toUpperCase() + state.market_regime.slice(1);
                        marketRegimeEl.className = `market-regime regime-${state.market_regime}`;
                    }
                    
                    if (volatilityStateEl) {
                        volatilityStateEl.textContent = `${state.volatility_state.charAt(0).toUpperCase() + state.volatility_state.slice(1)} Volatility`;
                        volatilityStateEl.className = `volatility-${state.volatility_state}`;
                    }
                    
                    // Update trend analysis
                    const trendAnalysisEl = document.getElementById('trend-analysis');
                    if (trendAnalysisEl) {
                        trendAnalysisEl.innerHTML = '';
                        
                        Object.entries(state.trend_details_by_tf).forEach(([timeframe, details]) => {
                            const trendClass = `trend-${details.trend}`;
                            
                            const timeframeEl = document.createElement('div');
                            timeframeEl.className = 'd-flex justify-content-between align-items-center mb-3 p-3 border rounded';
                            
                            timeframeEl.innerHTML = `
                                <div>
                                    <span class="trend-indicator ${trendClass}"></span>
                                    <strong>${timeframe}</strong>
                                </div>
                                <div>
                                    <span class="me-3">ADX: ${details.adx.toFixed(1)}</span>
                                    <span>RSI: ${details.rsi.toFixed(1)}</span>
                                </div>
                            `;
                            
                            trendAnalysisEl.appendChild(timeframeEl);
                        });
                    }
                    
                    // Update volatility chart if it exists
                    if (volatilityChart && state.volatility_data) {
                        volatilityChart.data.labels = state.volatility_data.labels;
                        volatilityChart.data.datasets[0].data = state.volatility_data.values;
                        volatilityChart.update();
                    }
                }
                
                // Update prices
                function updatePrices(prices) {
                    // Update open trades with new prices
                    Object.entries(prices).forEach(([symbol, price]) => {
                        if (openTrades[symbol]) {
                            const trade = openTrades[symbol];
                            const profitPercent = ((price - trade.entry_price) / trade.entry_price) * 100;
                            
                            // Update the trade with the new price and profit
                            updateTrade({
                                symbol: symbol,
                                current_price: price,
                                profit_percent: profitPercent
                            });
                        }
                    });
                }
                
                // Initialize performance chart
                function initPerformanceChart(data) {
                    const ctx = document.getElementById('performance-chart').getContext('2d');
                    
                    performanceChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: data.labels,
                            datasets: [{
                                label: 'Profit/Loss (%)',
                                data: data.values,
                                borderColor: data.values.map(v => v >= 0 ? '#28a745' : '#dc3545'),
                                backgroundColor: 'rgba(0, 0, 0, 0.1)',
                                borderWidth: 2,
                                fill: true,
                                tension: 0.4
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: {
                                    display: false
                                }
                            },
                            scales: {
                                y: {
                                    beginAtZero: false,
                                    grid: {
                                        color: 'rgba(0, 0, 0, 0.05)'
                                    }
                                },
                                x: {
                                    grid: {
                                        display: false
                                    }
                                }
                            }
                        }
                    });
                }
                
                // Initialize strategy chart
                function initStrategyChart(data) {
                    const ctx = document.getElementById('strategy-chart').getContext('2d');
                    
                    strategyChart = new Chart(ctx, {
                        type: 'doughnut',
                        data: {
                            labels: data.labels,
                            datasets: [{
                                data: data.values,
                                backgroundColor: [
                                    '#FF6384',
                                    '#36A2EB',
                                    '#FFCE56',
                                    '#4BC0C0',
                                    '#9966FF',
                                    '#FF9F40',
                                    '#8AC926'
                                ],
                                borderWidth: 0
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: {
                                    position: 'right'
                                }
                            }
                        }
                    });
                }
                
                // Initialize volatility chart
                function initVolatilityChart(data) {
                    const ctx = document.getElementById('volatility-chart').getContext('2d');
                    
                    volatilityChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: data.labels,
                            datasets: [{
                                label: 'Volatility (%)',
                                data: data.values,
                                borderColor: '#36A2EB',
                                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                                borderWidth: 2,
                                fill: true,
                                tension: 0.4
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: {
                                    display: false
                                }
                            },
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    grid: {
                                        color: 'rgba(0, 0, 0, 0.05)'
                                    }
                                },
                                x: {
                                    grid: {
                                        display: false
                                    }
                                }
                            }
                        }
                    });
                }
                
                // Show settings updated notification
                function showSettingsUpdated() {
                    const alert = document.createElement('div');
                    alert.className = 'alert alert-success alert-dismissible fade show position-fixed';
                    alert.style.top = '20px';
                    alert.style.right = '20px';
                    alert.style.zIndex = '9999';
                    alert.innerHTML = `
                        Settings updated successfully!
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    `;
                    
                    document.body.appendChild(alert);
                    
                    setTimeout(() => {
                        alert.classList.remove('show');
                        setTimeout(() => alert.remove(), 150);
                    }, 3000);
                }
                
                // Tab navigation
                document.addEventListener('DOMContentLoaded', function() {
                    // Initialize WebSocket
                    initWebSocket();
                    
                    // Tab navigation
                    const tabLinks = document.querySelectorAll('.nav-link');
                    const tabContents = document.querySelectorAll('.tab-content');
                    
                    tabLinks.forEach(link => {
                        link.addEventListener('click', function(e) {
                            e.preventDefault();
                            
                            const tabId = this.getAttribute('data-tab');
                            
                            // Update active tab link
                            tabLinks.forEach(l => l.classList.remove('active'));
                            this.classList.add('active');
                            
                            // Show corresponding tab content
                            tabContents.forEach(content => {
                                content.style.display = 'none';
                            });
                            
                            document.getElementById(`${tabId}-tab`).style.display = 'block';
                        });
                    });
                    
                    // Toggle trading button
                    document.getElementById('toggle-trading').addEventListener('click', function() {
                        const enabled = this.textContent === 'Disable Trading';
                        
                        socket.send(JSON.stringify({
                            type: 'toggle_trading',
                            payload: { enabled: !enabled }
                        }));
                    });
                    
                    // Save trading settings
                    document.getElementById('save-trading-settings').addEventListener('click', function() {
                        const settings = {
                            paper_trading: document.getElementById('paper-trading-switch').checked,
                            auto_fallback: document.getElementById('auto-fallback-switch').checked,
                            min_trade_amount: parseFloat(document.getElementById('min-trade-amount').value),
                            max_trade_amount: parseFloat(document.getElementById('max-trade-amount').value),
                            max_trades: parseInt(document.getElementById('max-open-trades').value),
                            signal_quality: parseInt(document.getElementById('signal-quality').value)
                        };
                        
                        socket.send(JSON.stringify({
                            type: 'update_trading_settings',
                            payload: settings
                        }));
                    });
                    
                    // Save strategy settings
                    document.getElementById('save-strategy-settings').addEventListener('click', function() {
                        const strategies = {
                            USE_BB_STOCH_STRATEGY: document.getElementById('bb-stoch-switch').checked,
                            USE_MACD_EMA_STRATEGY: document.getElementById('macd-ema-switch').checked,
                            USE_EMA_RSI_STRATEGY: document.getElementById('ema-rsi-switch').checked,
                            USE_PULLBACK_STRATEGY: document.getElementById('pullback-switch').checked,
                            USE_MOMENTUM_VOLATILITY_STRATEGY: document.getElementById('momentum-switch').checked,
                            USE_ELLIOTT_WAVE_STRATEGY: document.getElementById('elliott-switch').checked,
                            USE_RANGE_REVERSAL_STRATEGY: document.getElementById('range-switch').checked
                        };
                        
                        socket.send(JSON.stringify({
                            type: 'update_strategy_settings',
                            payload: strategies
                        }));
                    });
                    
                    // Save notification settings
                    document.getElementById('save-notification-settings').addEventListener('click', function() {
                        const settings = {
                            telegram_enabled: document.getElementById('telegram-notifications-switch').checked,
                            email_enabled: document.getElementById('email-notifications-switch').checked,
                            min_profit_notification: parseFloat(document.getElementById('min-profit-notification').value),
                            max_loss_notification: parseFloat(document.getElementById('max-loss-notification').value)
                        };
                        
                        socket.send(JSON.stringify({
                            type: 'update_notification_settings',
                            payload: settings
                        }));
                    });
                });
            </script>
        </body>
        </html>
    ''')

@app.route('/api/status')
def get_status():
    global is_trading_enabled, paper_trading_mode, usdt_balance, open_signals_cache, current_market_state
    
    with trading_status_lock:
        trading_enabled = is_trading_enabled
    
    with trading_mode_lock:
        is_paper = paper_trading_mode
    
    with balance_lock:
        balance = usdt_balance
    
    with signal_cache_lock:
        open_trades_count = len(open_signals_cache)
        open_trades = list(open_signals_cache.values())
    
    with market_state_lock:
        market_state = dict(current_market_state)
    
    # Get strategy settings
    strategies = {
        "USE_BB_STOCH_STRATEGY": USE_BB_STOCH_STRATEGY,
        "USE_MACD_EMA_STRATEGY": USE_MACD_EMA_STRATEGY,
        "USE_EMA_RSI_STRATEGY": USE_EMA_RSI_STRATEGY,
        "USE_PULLBACK_STRATEGY": USE_PULLBACK_STRATEGY,
        "USE_MOMENTUM_VOLATILITY_STRATEGY": USE_MOMENTUM_VOLATILITY_STRATEGY,
        "USE_ELLIOTT_WAVE_STRATEGY": USE_ELLIOTT_WAVE_STRATEGY,
        "USE_RANGE_REVERSAL_STRATEGY": USE_RANGE_REVERSAL_STRATEGY
    }
    
    # Get notification settings
    notification_settings = get_notification_settings()
    
    # Get min signal quality
    with min_quality_lock:
        min_quality = MIN_SIGNAL_QUALITY
    
    # Get trade amount settings
    with trade_amount_lock:
        min_trade_amount = FIXED_TRADE_AMOUNT_MIN_USDT
        max_trade_amount = FIXED_TRADE_AMOUNT_MAX_USDT
    
    # Get max open trades
    max_trades = MAX_OPEN_TRADES
    
    # Get auto fallback setting
    auto_fallback = AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE
    
    # Get performance data (mock data for now)
    performance_data = {
        "labels": ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"],
        "values": [1.2, -0.5, 2.1, 0.8, -1.2, 1.5, 0.9]
    }
    
    # Get strategy distribution (mock data for now)
    strategy_distribution = {
        "labels": ["BB+Stoch", "MACD+EMA", "EMA+RSI", "Pullback", "Momentum", "Elliott", "Range"],
        "values": [15, 20, 18, 12, 10, 15, 10]
    }
    
    # Get volatility data (mock data for now)
    volatility_data = {
        "labels": ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"],
        "values": [1.2, 0.9, 1.5, 2.1, 1.8, 1.4, 1.3]
    }
    
    return jsonify({
        "trading_enabled": trading_enabled,
        "paper_trading": is_paper,
        "balance": balance,
        "open_trades_count": open_trades_count,
        "max_trades": max_trades,
        "market_state": market_state,
        "signal_quality": min_quality,
        "open_trades": open_trades,
        "notifications": list(notifications_cache),
        "rejections": list(rejection_logs_cache),
        "strategies": strategies,
        "notification_settings": notification_settings,
        "auto_fallback": auto_fallback,
        "min_trade_amount": min_trade_amount,
        "max_trade_amount": max_trade_amount,
        "performance_data": performance_data,
        "strategy_distribution": strategy_distribution,
        "volatility_data": volatility_data
    })

@app.route('/api/toggle_trading', methods=['POST'])
def toggle_trading():
    global is_trading_enabled
    
    data = request.get_json()
    enabled = data.get('enabled', False)
    
    with trading_status_lock:
        is_trading_enabled = enabled
    
    action = "enabled" if enabled else "disabled"
    log_and_notify("info", f"Trading {action}", "trading_status")
    
    return jsonify({"success": True, "enabled": enabled})

@app.route('/api/update_trading_settings', methods=['POST'])
def update_trading_settings():
    global paper_trading_mode, AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE, FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT, MAX_OPEN_TRADES, MIN_SIGNAL_QUALITY
    
    data = request.get_json()
    
    with trading_mode_lock:
        paper_trading_mode = data.get('paper_trading', True)
    
    AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE = data.get('auto_fallback', True)
    
    with trade_amount_lock:
        FIXED_TRADE_AMOUNT_MIN_USDT = data.get('min_trade_amount', 4.5)
        FIXED_TRADE_AMOUNT_MAX_USDT = data.get('max_trade_amount', 6.5)
    
    MAX_OPEN_TRADES = data.get('max_trades', 3)
    
    with min_quality_lock:
        MIN_SIGNAL_QUALITY = data.get('signal_quality', 70)
    
    # Save settings to Redis
    save_settings_to_redis()
    
    log_and_notify("info", "Trading settings updated", "settings_update")
    
    return jsonify({"success": True})

@app.route('/api/update_strategy_settings', methods=['POST'])
def update_strategy_settings():
    global USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY, USE_MOMENTUM_VOLATILITY_STRATEGY, USE_ELLIOTT_WAVE_STRATEGY, USE_RANGE_REVERSAL_STRATEGY
    
    data = request.get_json()
    
    USE_BB_STOCH_STRATEGY = data.get('USE_BB_STOCH_STRATEGY', True)
    USE_MACD_EMA_STRATEGY = data.get('USE_MACD_EMA_STRATEGY', True)
    USE_EMA_RSI_STRATEGY = data.get('USE_EMA_RSI_STRATEGY', True)
    USE_PULLBACK_STRATEGY = data.get('USE_PULLBACK_STRATEGY', True)
    USE_MOMENTUM_VOLATILITY_STRATEGY = data.get('USE_MOMENTUM_VOLATILITY_STRATEGY', True)
    USE_ELLIOTT_WAVE_STRATEGY = data.get('USE_ELLIOTT_WAVE_STRATEGY', True)
    USE_RANGE_REVERSAL_STRATEGY = data.get('USE_RANGE_REVERSAL_STRATEGY', True)
    
    # Save settings to Redis
    save_settings_to_redis()
    
    log_and_notify("info", "Strategy settings updated", "settings_update")
    
    return jsonify({"success": True})

@app.route('/api/update_notification_settings', methods=['POST'])
def update_notification_settings():
    if not redis_client:
        return jsonify({"success": False, "error": "Redis not available"})
    
    data = request.get_json()
    
    try:
        redis_client.set('notification_settings', json.dumps(data))
        log_and_notify("info", "Notification settings updated", "settings_update")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"❌ [Settings] Error updating notification settings: {e}")
        return jsonify({"success": False, "error": str(e)})

# --- WebSocket endpoint ---
@sock.route('/ws')
def websocket_connection(ws):
    ws_clients.append(ws)
    
    try:
        # Send initial data
        with trading_status_lock:
            trading_enabled = is_trading_enabled
        
        with trading_mode_lock:
            is_paper = paper_trading_mode
        
        with balance_lock:
            balance = usdt_balance
        
        with signal_cache_lock:
            open_trades_count = len(open_signals_cache)
            open_trades = list(open_signals_cache.values())
        
        with market_state_lock:
            market_state = dict(current_market_state)
        
        with min_quality_lock:
            min_quality = MIN_SIGNAL_QUALITY
        
        # Get strategy settings
        strategies = {
            "USE_BB_STOCH_STRATEGY": USE_BB_STOCH_STRATEGY,
            "USE_MACD_EMA_STRATEGY": USE_MACD_EMA_STRATEGY,
            "USE_EMA_RSI_STRATEGY": USE_EMA_RSI_STRATEGY,
            "USE_PULLBACK_STRATEGY": USE_PULLBACK_STRATEGY,
            "USE_MOMENTUM_VOLATILITY_STRATEGY": USE_MOMENTUM_VOLATILITY_STRATEGY,
            "USE_ELLIOTT_WAVE_STRATEGY": USE_ELLIOTT_WAVE_STRATEGY,
            "USE_RANGE_REVERSAL_STRATEGY": USE_RANGE_REVERSAL_STRATEGY
        }
        
        # Get notification settings
        notification_settings = get_notification_settings()
        
        # Get trade amount settings
        with trade_amount_lock:
            min_trade_amount = FIXED_TRADE_AMOUNT_MIN_USDT
            max_trade_amount = FIXED_TRADE_AMOUNT_MAX_USDT
        
        # Get max open trades
        max_trades = MAX_OPEN_TRADES
        
        # Get auto fallback setting
        auto_fallback = AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE
        
        # Get performance data (mock data for now)
        performance_data = {
            "labels": ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"],
            "values": [1.2, -0.5, 2.1, 0.8, -1.2, 1.5, 0.9]
        }
        
        # Get strategy distribution (mock data for now)
        strategy_distribution = {
            "labels": ["BB+Stoch", "MACD+EMA", "EMA+RSI", "Pullback", "Momentum", "Elliott", "Range"],
            "values": [15, 20, 18, 12, 10, 15, 10]
        }
        
        # Get volatility data (mock data for now)
        volatility_data = {
            "labels": ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"],
            "values": [1.2, 0.9, 1.5, 2.1, 1.8, 1.4, 1.3]
        }
        
        initial_data = {
            "type": "initial_data",
            "payload": {
                "trading_enabled": trading_enabled,
                "paper_trading": is_paper,
                "balance": balance,
                "open_trades_count": open_trades_count,
                "max_trades": max_trades,
                "market_state": market_state,
                "signal_quality": min_quality,
                "open_trades": open_trades,
                "notifications": list(notifications_cache),
                "rejections": list(rejection_logs_cache),
                "strategies": strategies,
                "notification_settings": notification_settings,
                "auto_fallback": auto_fallback,
                "min_trade_amount": min_trade_amount,
                "max_trade_amount": max_trade_amount,
                "performance_data": performance_data,
                "strategy_distribution": strategy_distribution,
                "volatility_data": volatility_data
            }
        }
        
        ws.send(json.dumps(initial_data, cls=NpEncoder))
        
        # Keep the connection open
        while True:
            data = ws.receive()
            if not data:
                break
                
            try:
                message = json.loads(data)
                
                if message.get('type') == 'get_initial_data':
                    ws.send(json.dumps(initial_data, cls=NpEncoder))
                    
            except Exception as e:
                logger.error(f"❌ [WebSocket] Error processing message: {e}")
                
    except Exception as e:
        logger.error(f"❌ [WebSocket] Error in connection: {e}")
    finally:
        if ws in ws_clients:
            ws_clients.remove(ws)

# --- Main function ---
def main():
    global client, usdt_balance, validated_symbols_to_scan
    
    # Initialize database
    init_db()
    
    # Initialize Redis
    init_redis()
    
    # Initialize Binance client
    try:
        client = Client(API_KEY, API_SECRET)
        logger.info("✅ [API] Binance client initialized successfully.")
    except Exception as e:
        logger.error(f"❌ [API] Error initializing Binance client: {e}")
        exit(1)
    
    # Get exchange info
    get_exchange_info_map()
    
    # Get validated symbols
    validated_symbols_to_scan = get_validated_symbols()
    if not validated_symbols_to_scan:
        logger.critical("❌ [Symbols] No valid symbols found. Exiting.")
        exit(1)
    
    # Load settings from Redis
    load_settings_from_redis()
    
    # Get account balance
    try:
        account_info = client.get_account()
        for balance in account_info['balances']:
            if balance['asset'] == 'USDT':
                with balance_lock:
                    usdt_balance = float(balance['free'])
                break
        logger.info(f"✅ [Account] USDT balance: ${usdt_balance:.2f}")
    except Exception as e:
        logger.error(f"❌ [Account] Error getting account balance: {e}")
        with balance_lock:
            usdt_balance = 0.0
    
    # Load open signals to cache
    load_open_signals_to_cache()
    
    # Load notifications to cache
    load_notifications_to_cache()
    
    # Start WebSocket
    start_websocket()
    
    # Start periodic reports
    start_periodic_reports()
    
    # Start trade analysis thread
    start_trade_analysis()
    
    # Initialize market state
    update_market_state()
    
    # Start Flask app
    logger.info("✅ [App] Starting Flask app...")
    app.run(host='0.0.0.0', port=5000, threaded=True)

if __name__ == '__main__':
    main()