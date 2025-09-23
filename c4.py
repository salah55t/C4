# ملف c4_5min_v35_0_1.py - نسخة V35.0.1 (لوحة تحكم عربية وإصلاح تفعيل التداول)
# --- وصف التعديلات:
# 1. [تحويل لوحة التحكم] تحويل واجهة لوحة التحكم بالكامل إلى اللغة العربية
# 2. [إصلاح تفعيل التداول] إصلاح مشكلة عدم استجابة البوت عند تفعيل التداول من لوحة التحكم
# 3. [تحسين اتجاه النصوص] تعديل اتجاه النصوص والواجهة لتكون مناسبة للغة العربية
# 4. [تحسين الأداء] تحسين أداء WebSocket ومعالجة الطلبات
# 5. [إصلاح المتغيرات العامة] إصلاح أخطاء استخدام المتغيرات العامة قبل الإعلان عنها

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
logger = logging.getLogger('CryptoBotV35.0.1_5min')

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

# --- دوال البيانات والمؤشرات ---
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
            with trading_mode_lock: 
                paper_trading_mode = settings.get('paper_trading_mode', True)
            
        quality_settings_data = redis_client.get('signal_quality_settings')
        if quality_settings_data:
            quality_settings = json.loads(quality_settings_data)
            with min_quality_lock: 
                MIN_SIGNAL_QUALITY = quality_settings.get('min_quality', 70)

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
    global FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT, MAX_OPEN_TRADES, USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY, USE_MOMENTUM_VOLATILITY_STRATEGY, USE_ELLIOTT_WAVE_STRATEGY, USE_RANGE_REVERSAL_STRATEGY, paper_trading_mode, MIN_SIGNAL_QUALITY
    
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

# --- الفلاتر الديناميكية ---
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
        trend_strength_multiplier = 1.0 + (last_row['atr_percent'] / 50)
    elif market_regime == "trending":
        trend_strength_multiplier = 1.0 + (last_row['atr_percent'] / 80)
    else:
        trend_strength_multiplier = 1.0 + (last_row['atr_percent'] / 65)
    
    return {
        'rsi_in_range': rsi_lower <= last_row['rsi'] <= rsi_upper,
        'ema_spread_ok': ema_spread.iloc[-1] > dynamic_ema_threshold.iloc[-1],
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * trend_strength_multiplier,
    }

def check_pullback_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    
    # تعديل العتبات بناءً على حالة السوق
    with market_state_lock:
        market_regime = current_market_state.get("market_regime", "unknown")
        volatility_state = current_market_state.get("volatility_state", "medium")
    
    # تعديل عتبة الارتداد بناءً على حالة السوق
    if volatility_state == "high":
        pullback_threshold = 0.038  # 3.8%
    elif market_regime == "trending":
        pullback_threshold = 0.025  # 2.5%
    else:
        pullback_threshold = 0.032  # 3.2%
    
    # حساب نسبة الارتداد
    recent_high = df['high'].rolling(10).max().iloc[-1]
    pullback_percent = (recent_high - last_row['close']) / recent_high
    
    # تعديل عتبة التعافي بناءً على حالة السوق
    if volatility_state == "high":
        recovery_threshold = 0.015  # 1.5%
    elif market_regime == "trending":
        recovery_threshold = 0.008  # 0.8%
    else:
        recovery_threshold = 0.012  # 1.2%
    
    # حساب نسبة التعافي
    recent_low = df['low'].rolling(5).min().iloc[-1]
    recovery_percent = (last_row['close'] - recent_low) / recent_low
    
    # تعديل عتبة الحجم بناءً على حالة السوق
    volume_ma = df['volume'].rolling(20).mean()
    if volatility_state == "high":
        volume_multiplier = 1.2
    elif market_regime == "trending":
        volume_multiplier = 1.0
    else:
        volume_multiplier = 1.1
    
    return {
        'pullback_ok': pullback_percent >= pullback_threshold,
        'recovery_ok': recovery_percent >= recovery_threshold,
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * volume_multiplier,
    }

def check_momentum_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    
    # تعديل العتبات بناءً على حالة السوق
    with market_state_lock:
        market_regime = current_market_state.get("market_regime", "unknown")
        volatility_state = current_market_state.get("volatility_state", "medium")
    
    # تعديل عتبة الزخم بناءً على حالة السوق
    if volatility_state == "high":
        momentum_threshold = 0.015  # 1.5%
    elif market_regime == "trending":
        momentum_threshold = 0.008  # 0.8%
    else:
        momentum_threshold = 0.012  # 1.2%
    
    # حساب نسبة الزخم
    price_change = (last_row['close'] - df['close'].iloc[-5]) / df['close'].iloc[-5]
    
    # تعديل عتبة التقلب بناءً على حالة السوق
    if volatility_state == "high":
        volatility_threshold = 0.025  # 2.5%
    elif market_regime == "trending":
        volatility_threshold = 0.015  # 1.5%
    else:
        volatility_threshold = 0.020  # 2.0%
    
    # حساب نسبة التقلب
    atr_percent = last_row['atr_percent'] / 100
    
    # تعديل عتبة الحجم بناءً على حالة السوق
    volume_ma = df['volume'].rolling(20).mean()
    if volatility_state == "high":
        volume_multiplier = 1.3
    elif market_regime == "trending":
        volume_multiplier = 1.1
    else:
        volume_multiplier = 1.2
    
    return {
        'momentum_ok': price_change >= momentum_threshold,
        'volatility_ok': atr_percent <= volatility_threshold,
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * volume_multiplier,
    }

def check_elliott_wave_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    
    # تعديل العتبات بناءً على حالة السوق
    with market_state_lock:
        market_regime = current_market_state.get("market_regime", "unknown")
        volatility_state = current_market_state.get("volatility_state", "medium")
    
    # تعديل عتبة تصحيح فيبوناتشي بناءً على حالة السوق
    if volatility_state == "high":
        fib_lower, fib_upper = 0.382, 0.786
    elif market_regime == "trending":
        fib_lower, fib_upper = 0.5, 0.618
    else:
        fib_lower, fib_upper = 0.382, 0.618
    
    # حساب نسبة تصحيح فيبوناتشي
    fib_retracement = get_wave_retracement(df)
    
    # تعديل عتبة ADX بناءً على حالة السوق
    if volatility_state == "high":
        adx_threshold = 25
    elif market_regime == "trending":
        adx_threshold = 20
    else:
        adx_threshold = 22
    
    # تعديل عتبة الحجم بناءً على حالة السوق
    volume_ma = df['volume'].rolling(20).mean()
    if volatility_state == "high":
        volume_multiplier = 1.2
    elif market_regime == "trending":
        volume_multiplier = 1.0
    else:
        volume_multiplier = 1.1
    
    return {
        'fib_ok': fib_lower <= fib_retracement <= fib_upper,
        'adx_ok': last_row['adx'] > adx_threshold,
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * volume_multiplier,
    }

def check_range_reversal_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    
    # تعديل العتبات بناءً على حالة السوق
    with market_state_lock:
        market_regime = current_market_state.get("market_regime", "unknown")
        volatility_state = current_market_state.get("volatility_state", "medium")
    
    # تعديل عتبة ADX (يجب أن يكون منخفضًا للانعكاس النطاقي)
    if volatility_state == "high":
        adx_threshold = 20
    elif market_regime == "ranging":
        adx_threshold = 15
    else:
        adx_threshold = 18
    
    # تعديل عتبة RSI بناءً على حالة السوق
    if volatility_state == "high":
        rsi_threshold = 28
    elif market_regime == "ranging":
        rsi_threshold = 30
    else:
        rsi_threshold = 32
    
    # تعديل عتبة الحجم بناءً على حالة السوق
    volume_ma = df['volume'].rolling(20).mean()
    if volatility_state == "high":
        volume_multiplier = 1.3
    elif market_regime == "ranging":
        volume_multiplier = 1.1
    else:
        volume_multiplier = 1.2
    
    return {
        'adx_ok': last_row['adx'] < adx_threshold,
        'rsi_ok': last_row['rsi'] < rsi_threshold,
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * volume_multiplier,
    }

# --- استراتيجيات التداول ---
def bb_stoch_strategy(df: pd.DataFrame, symbol: str) -> Optional[Dict]:
    """
    استراتيجية BB + Stochastics
    """
    try:
        if not USE_BB_STOCH_STRATEGY:
            return None
            
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        # فحص الفلاتر الديناميكية
        dynamic_filters = check_bb_stoch_dynamic_filters(df)
        if not all(dynamic_filters.values()):
            if not dynamic_filters['bb_width_ok']:
                log_rejection(symbol, "DYN_BB_WIDTH_LOW")
            elif not dynamic_filters['stoch_ok']:
                log_rejection(symbol, "DYN_STOCH_LOW")
            elif not dynamic_filters['volume_ok']:
                log_rejection(symbol, "DYN_VOLUME_LOW")
            return None
        
        # شروط الاستراتيجية
        condition1 = last_row['close'] > last_row['bb_middle']  # السعر فوق منتصف البولينجر
        condition2 = last_row['stoch_k'] > last_row['stoch_d']  # ستوكاستيك صاعد
        condition3 = last_row['stoch_k'] < 80  # ستوكاستيك ليس في منطقة الشراء المفرط
        condition4 = prev_row['stoch_k'] <= prev_row['stoch_d']  # تقاطع صاعد في الستوكاستيك
        
        # فحص اتجاه السوق
        if last_row['ema50'] < last_row['ema200'] or last_row['close'] < last_row['ema50']:
            log_rejection(symbol, "BB: Price below EMA50 (bearish trend)")
            return None
        
        if condition1 and condition2 and condition3 and condition4:
            atr = last_row['atr']
            stop_loss = max(last_row['low'] - (atr * 0.5), last_row['close'] * 0.985)
            
            # حساب الأهداف
            target1 = last_row['close'] + (atr * 1.5)
            target2 = last_row['close'] + (atr * 3.0)
            
            # حساب جودة الإشارة
            quality_score = 70
            if last_row['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.2:
                quality_score += 10
            if last_row['rsi'] > 50 and last_row['rsi'] < 70:
                quality_score += 10
            if last_row['macd_hist'] > 0:
                quality_score += 10
                
            return {
                'symbol': symbol,
                'strategy_name': 'BB_Stoch_Strategy',
                'entry_price': last_row['close'],
                'stop_loss': stop_loss,
                'target_price_1': target1,
                'target_price_2': target2,
                'quality_score': min(quality_score, 100),
                'atr_percent': last_row['atr_percent']
            }
            
        return None
    except Exception as e:
        logger.error(f"❌ [Strategy] Error in BB_Stoch strategy for {symbol}: {e}")
        return None

def macd_ema_strategy(df: pd.DataFrame, symbol: str) -> Optional[Dict]:
    """
    استراتيجية MACD + EMA
    """
    try:
        if not USE_MACD_EMA_STRATEGY:
            return None
            
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        # فحص الفلاتر الديناميكية
        dynamic_filters = check_macd_ema_dynamic_filters(df)
        if not all(dynamic_filters.values()):
            if not dynamic_filters['adx_ok']:
                log_rejection(symbol, "DYN_ADX_LOW")
            elif not dynamic_filters['volume_ok']:
                log_rejection(symbol, "DYN_VOLUME_LOW")
            elif not dynamic_filters['momentum_ok']:
                log_rejection(symbol, "DYN_MACD_MOMENTUM_LOW")
            return None
        
        # شروط الاستراتيجية
        condition1 = last_row['macd'] > last_row['macd_signal']  # MACD فوق إشارته
        condition2 = prev_row['macd'] <= prev_row['macd_signal']  # تقاطع صاعد
        condition3 = last_row['macd_hist'] > 0  # MACD histogram موجب
        condition4 = last_row['close'] > last_row['ema9']  # السعر فوق EMA9
        
        # فحص اتجاه السوق
        if last_row['ema9'] < last_row['ema21'] or last_row['ema21'] < last_row['ema50']:
            log_rejection(symbol, "MACD: Strongly bearish trend")
            return None
        
        if condition1 and condition2 and condition3 and condition4:
            atr = last_row['atr']
            stop_loss = max(last_row['low'] - (atr * 0.7), last_row['close'] * 0.98)
            
            # حساب الأهداف
            target1 = last_row['close'] + (atr * 1.8)
            target2 = last_row['close'] + (atr * 3.5)
            
            # حساب جودة الإشارة
            quality_score = 70
            if last_row['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.3:
                quality_score += 10
            if last_row['rsi'] > 50 and last_row['rsi'] < 65:
                quality_score += 10
            if last_row['ema9'] > last_row['ema21'] and last_row['ema21'] > last_row['ema50']:
                quality_score += 10
                
            return {
                'symbol': symbol,
                'strategy_name': 'MACD_EMA_Strategy',
                'entry_price': last_row['close'],
                'stop_loss': stop_loss,
                'target_price_1': target1,
                'target_price_2': target2,
                'quality_score': min(quality_score, 100),
                'atr_percent': last_row['atr_percent']
            }
            
        return None
    except Exception as e:
        logger.error(f"❌ [Strategy] Error in MACD_EMA strategy for {symbol}: {e}")
        return None

def ema_rsi_strategy(df: pd.DataFrame, symbol: str) -> Optional[Dict]:
    """
    استراتيجية EMA + RSI
    """
    try:
        if not USE_EMA_RSI_STRATEGY:
            return None
            
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        # فحص الفلاتر الديناميكية
        dynamic_filters = check_ema_rsi_dynamic_filters(df)
        if not all(dynamic_filters.values()):
            if not dynamic_filters['rsi_in_range']:
                log_rejection(symbol, "DYN_RSI_OOR")
            elif not dynamic_filters['ema_spread_ok']:
                log_rejection(symbol, "DYN_EMA_SPREAD_LOW")
            elif not dynamic_filters['volume_ok']:
                log_rejection(symbol, "DYN_VOLUME_LOW")
            return None
        
        # شروط الاستراتيجية
        condition1 = last_row['rsi'] > 50  # RSI فوق 50
        condition2 = last_row['rsi'] < prev_row['rsi'] and prev_row['rsi'] < df['rsi'].iloc[-3]  # RSI كان في انخفاض والآن يرتفع
        condition3 = last_row['close'] > last_row['ema9']  # السعر فوق EMA9
        condition4 = last_row['ema9'] > last_row['ema21']  # EMA9 فوق EMA21
        
        # فحص اتجاه السوق طويل الأجل
        if last_row['ema50'] < last_row['ema200']:
            log_rejection(symbol, "EMA_RSI: Bearish long-term trend")
            return None
        
        if condition1 and condition2 and condition3 and condition4:
            atr = last_row['atr']
            stop_loss = max(last_row['low'] - (atr * 0.6), last_row['close'] * 0.982)
            
            # حساب الأهداف
            target1 = last_row['close'] + (atr * 1.6)
            target2 = last_row['close'] + (atr * 3.2)
            
            # حساب جودة الإشارة
            quality_score = 70
            if last_row['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.25:
                quality_score += 10
            if last_row['rsi'] > 45 and last_row['rsi'] < 60:
                quality_score += 10
            if last_row['macd_hist'] > 0:
                quality_score += 10
                
            return {
                'symbol': symbol,
                'strategy_name': 'EMA_RSI_Strategy',
                'entry_price': last_row['close'],
                'stop_loss': stop_loss,
                'target_price_1': target1,
                'target_price_2': target2,
                'quality_score': min(quality_score, 100),
                'atr_percent': last_row['atr_percent']
            }
            
        return None
    except Exception as e:
        logger.error(f"❌ [Strategy] Error in EMA_RSI strategy for {symbol}: {e}")
        return None

def pullback_strategy(df: pd.DataFrame, symbol: str) -> Optional[Dict]:
    """
    استراتيجية Pullback
    """
    try:
        if not USE_PULLBACK_STRATEGY:
            return None
            
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        # فحص الفلاتر الديناميكية
        dynamic_filters = check_pullback_dynamic_filters(df)
        if not all(dynamic_filters.values()):
            if not dynamic_filters['pullback_ok']:
                log_rejection(symbol, "DYN_PULLBACK_SHALLOW")
            elif not dynamic_filters['recovery_ok']:
                log_rejection(symbol, "DYN_RECOVERY_FAIL")
            elif not dynamic_filters['volume_ok']:
                log_rejection(symbol, "DYN_VOLUME_LOW")
            return None
        
        # شروط الاستراتيجية
        # تحديد قمة محلية
        recent_high = df['high'].rolling(10).max().iloc[-1]
        high_idx = df['high'].rolling(10).apply(lambda x: x.argmax()).iloc[-1]
        
        # حساب نسبة الارتداد
        pullback_percent = (recent_high - last_row['close']) / recent_high
        
        # حساب نسبة التعافي
        recent_low = df['low'].rolling(5).min().iloc[-1]
        recovery_percent = (last_row['close'] - recent_low) / recent_low
        
        # شروط الدخول
        condition1 = pullback_percent >= 0.025  # ارتداد 2.5% على الأقل
        condition2 = recovery_percent >= 0.01  # تعافي 1% على الأقل
        condition3 = last_row['close'] > last_row['ema21']  # السعر فوق EMA21
        condition4 = last_row['ema21'] > last_row['ema50']  # EMA21 فوق EMA50
        
        # فحص اتجاه السوق
        if last_row['ema50'] < last_row['ema200'] or last_row['adx'] < 18:
            log_rejection(symbol, "Pullback: Trend is not strongly bullish")
            return None
        
        if condition1 and condition2 and condition3 and condition4:
            atr = last_row['atr']
            stop_loss = max(recent_low - (atr * 0.3), last_row['close'] * 0.985)
            
            # حساب الأهداف
            target1 = last_row['close'] + (atr * 1.7)
            target2 = recent_high  # الهدف الثاني هو القمة المحلية
            
            # حساب جودة الإشارة
            quality_score = 70
            if last_row['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.2:
                quality_score += 10
            if last_row['rsi'] > 45 and last_row['rsi'] < 65:
                quality_score += 10
            if last_row['macd_hist'] > 0:
                quality_score += 10
                
            return {
                'symbol': symbol,
                'strategy_name': 'Pullback_Strategy',
                'entry_price': last_row['close'],
                'stop_loss': stop_loss,
                'target_price_1': target1,
                'target_price_2': target2,
                'quality_score': min(quality_score, 100),
                'atr_percent': last_row['atr_percent']
            }
            
        return None
    except Exception as e:
        logger.error(f"❌ [Strategy] Error in Pullback strategy for {symbol}: {e}")
        return None

def momentum_volatility_strategy(df: pd.DataFrame, symbol: str) -> Optional[Dict]:
    """
    استراتيجية الزخم والتقلب
    """
    try:
        if not USE_MOMENTUM_VOLATILITY_STRATEGY:
            return None
            
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        # فحص الفلاتر الديناميكية
        dynamic_filters = check_momentum_dynamic_filters(df)
        if not all(dynamic_filters.values()):
            if not dynamic_filters['momentum_ok']:
                log_rejection(symbol, "DYN_MOMENTUM_SCORE_LOW")
            elif not dynamic_filters['volatility_ok']:
                log_rejection(symbol, "DYN_VOLATILITY_OOR")
            elif not dynamic_filters['volume_ok']:
                log_rejection(symbol, "DYN_VOLUME_LOW")
            return None
        
        # حساب نسبة الزخم
        price_change = (last_row['close'] - df['close'].iloc[-5]) / df['close'].iloc[-5]
        
        # حساب نسبة التقلب
        atr_percent = last_row['atr_percent'] / 100
        
        # شروط الاستراتيجية
        condition1 = price_change >= 0.01  # زيادة سعرية 1% على الأقل
        condition2 = atr_percent <= 0.02  # تقلب معتدل (2% أو أقل)
        condition3 = last_row['close'] > last_row['ema9']  # السعر فوق EMA9
        condition4 = last_row['ema9'] > last_row['ema21'] and last_row['ema21'] > last_row['ema50']  # ترتيب EMA صاعد
        
        # فحص اتجاه السوق
        if last_row['ema50'] < last_row['ema200']:
            log_rejection(symbol, "Momentum: EMAs not in bullish order")
            return None
        
        if condition1 and condition2 and condition3 and condition4:
            atr = last_row['atr']
            stop_loss = max(last_row['low'] - (atr * 0.8), last_row['close'] * 0.98)
            
            # حساب الأهداف
            target1 = last_row['close'] + (atr * 2.0)
            target2 = last_row['close'] + (atr * 4.0)
            
            # حساب جودة الإشارة
            quality_score = 70
            if last_row['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.3:
                quality_score += 10
            if last_row['rsi'] > 50 and last_row['rsi'] < 70:
                quality_score += 10
            if last_row['macd_hist'] > 0 and last_row['macd_hist'] > df['macd_hist'].iloc[-2]:
                quality_score += 10
                
            return {
                'symbol': symbol,
                'strategy_name': 'Momentum_Volatility_Strategy',
                'entry_price': last_row['close'],
                'stop_loss': stop_loss,
                'target_price_1': target1,
                'target_price_2': target2,
                'quality_score': min(quality_score, 100),
                'atr_percent': last_row['atr_percent']
            }
            
        return None
    except Exception as e:
        logger.error(f"❌ [Strategy] Error in Momentum_Volatility strategy for {symbol}: {e}")
        return None

def elliott_wave_strategy(df: pd.DataFrame, symbol: str) -> Optional[Dict]:
    """
    استراتيجية موجات إليوت
    """
    try:
        if not USE_ELLIOTT_WAVE_STRATEGY:
            return None
            
        last_row = df.iloc[-1]
        
        # فحص الفلاتر الديناميكية
        dynamic_filters = check_elliott_wave_dynamic_filters(df)
        if not all(dynamic_filters.values()):
            if not dynamic_filters['fib_ok']:
                log_rejection(symbol, "DYN_FIB_RETRACEMENT_OOR")
            elif not dynamic_filters['adx_ok']:
                log_rejection(symbol, "DYN_ADX_LOW")
            elif not dynamic_filters['volume_ok']:
                log_rejection(symbol, "DYN_VOLUME_LOW")
            return None
        
        # حساب نسبة تصحيح فيبوناتشي
        fib_retracement = get_wave_retracement(df)
        
        # شروط الاستراتيجية
        condition1 = 0.382 <= fib_retracement <= 0.618  # تصحيح فيبوناتشي في النطاق المطلوب
        condition2 = last_row['close'] > last_row['ema21']  # السعر فوق EMA21
        condition3 = last_row['ema21'] > last_row['ema50']  # EMA21 فوق EMA50
        
        # فحص اتجاه السوق
        if last_row['ema50'] < last_row['ema200']:
            log_rejection(symbol, "Elliott Wave: Strongly bearish trend")
            return None
        
        if condition1 and condition2 and condition3:
            atr = last_row['atr']
            stop_loss = max(last_row['low'] - (atr * 0.7), last_row['close'] * 0.98)
            
            # حساب الأهداف
            target1 = last_row['close'] + (atr * 1.8)
            target2 = last_row['close'] + (atr * 3.6)
            
            # حساب جودة الإشارة
            quality_score = 70
            if last_row['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.2:
                quality_score += 10
            if last_row['rsi'] > 45 and last_row['rsi'] < 65:
                quality_score += 10
            if last_row['macd_hist'] > 0:
                quality_score += 10
                
            return {
                'symbol': symbol,
                'strategy_name': 'Elliott_Wave_Strategy',
                'entry_price': last_row['close'],
                'stop_loss': stop_loss,
                'target_price_1': target1,
                'target_price_2': target2,
                'quality_score': min(quality_score, 100),
                'atr_percent': last_row['atr_percent']
            }
            
        return None
    except Exception as e:
        logger.error(f"❌ [Strategy] Error in Elliott_Wave strategy for {symbol}: {e}")
        return None

def range_reversal_strategy(df: pd.DataFrame, symbol: str) -> Optional[Dict]:
    """
    استراتيجية انعكاس النطاق
    """
    try:
        if not USE_RANGE_REVERSAL_STRATEGY:
            return None
            
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        # فحص الفلاتر الديناميكية
        dynamic_filters = check_range_reversal_dynamic_filters(df)
        if not all(dynamic_filters.values()):
            if not dynamic_filters['adx_ok']:
                log_rejection(symbol, "Range Reversal: Trend too strong (ADX > 23)")
            elif not dynamic_filters['rsi_ok']:
                log_rejection(symbol, "Range Reversal: RSI not in oversold zone")
            elif not dynamic_filters['volume_ok']:
                log_rejection(symbol, "DYN_VOLUME_LOW")
            return None
        
        # شروط الاستراتيجية
        condition1 = last_row['adx'] < 23  # ADX منخفض (سوق جانبي)
        condition2 = last_row['rsi'] < 30  # RSI في منطقة تشبع البيع
        condition3 = last_row['close'] > prev_row['close']  # السعر يرتفع
        condition4 = last_row['close'] > last_row['bb_lower']  # السعر فوق حزام البولينجر السفلي
        
        if condition1 and condition2 and condition3 and condition4:
            atr = last_row['atr']
            stop_loss = max(last_row['low'] - (atr * 0.5), last_row['close'] * 0.985)
            
            # حساب الأهداف
            target1 = last_row['close'] + (atr * 1.5)
            target2 = last_row['close'] + (atr * 3.0)
            
            # حساب جودة الإشارة
            quality_score = 70
            if last_row['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.3:
                quality_score += 10
            if last_row['rsi'] > 20 and last_row['rsi'] < 35:
                quality_score += 10
            if last_row['macd_hist'] > df['macd_hist'].iloc[-3]:  # MACD histogram يتحسن
                quality_score += 10
                
            return {
                'symbol': symbol,
                'strategy_name': 'Range_Reversal_Strategy',
                'entry_price': last_row['close'],
                'stop_loss': stop_loss,
                'target_price_1': target1,
                'target_price_2': target2,
                'quality_score': min(quality_score, 100),
                'atr_percent': last_row['atr_percent']
            }
            
        return None
    except Exception as e:
        logger.error(f"❌ [Strategy] Error in Range_Reversal strategy for {symbol}: {e}")
        return None

# --- دوال إدارة الصفقات ---
def get_open_trades_count() -> int:
    global open_signals_cache
    with signal_cache_lock:
        return len(open_signals_cache)

def is_symbol_in_cooldown(symbol: str) -> bool:
    with cooldowns_lock:
        if symbol in cooldowns_by_symbol:
            cooldown_time = cooldowns_by_symbol[symbol]
            if datetime.now(timezone.utc) < cooldown_time:
                return True
            else:
                del cooldowns_by_symbol[symbol]
    return False

def set_symbol_cooldown(symbol: str, minutes: int = COOLDOWN_MINUTES_AFTER_SL):
    with cooldowns_lock:
        cooldowns_by_symbol[symbol] = datetime.now(timezone.utc) + timedelta(minutes=minutes)

def get_consecutive_losses(symbol: str) -> int:
    with consecutive_losses_lock:
        return consecutive_losses_by_symbol.get(symbol, 0)

def increment_consecutive_losses(symbol: str):
    with consecutive_losses_lock:
        consecutive_losses_by_symbol[symbol] = consecutive_losses_by_symbol.get(symbol, 0) + 1

def reset_consecutive_losses(symbol: str):
    with consecutive_losses_lock:
        if symbol in consecutive_losses_by_symbol:
            del consecutive_losses_by_symbol[symbol]

def calculate_position_size(symbol: str, entry_price: float, stop_loss: float, is_real_trade: bool) -> Tuple[float, float]:
    """
    حساب حجم الصفقة والقيمة الاسمية
    """
    global FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT, PAPER_TRADE_FIXED_AMOUNT_USDT
    
    risk_percent = ((entry_price - stop_loss) / entry_price) * 100
    
    if is_real_trade:
        with trade_amount_lock:
            # حساب حجم الصفقة بناءً على نسبة المخاطرة
            if risk_percent > 0:
                amount_usdt = FIXED_TRADE_AMOUNT_MIN_USDT + (FIXED_TRADE_AMOUNT_MAX_USDT - FIXED_TRADE_AMOUNT_MIN_USDT) * min(1.0, 2.0 / risk_percent)
            else:
                amount_usdt = FIXED_TRADE_AMOUNT_MIN_USDT
            
            # التأكد من أن القيمة ضمن النطاق المسموح به
            amount_usdt = max(FIXED_TRADE_AMOUNT_MIN_USDT, min(amount_usdt, FIXED_TRADE_AMOUNT_MAX_USDT))
    else:
        amount_usdt = PAPER_TRADE_FIXED_AMOUNT_USDT
    
    # حساب الكمية
    quantity = amount_usdt / entry_price
    
    # التأكد من أن الكمية ضمن حدود المنصة
    symbol_info = exchange_info_map.get(symbol, {})
    if symbol_info:
        lot_size_filter = next((f for f in symbol_info.get('filters', []) if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = float(lot_size_filter['stepSize'])
            min_qty = float(lot_size_filter['minQty'])
            
            # تعديل الكمية لتتوافق مع حجم الخطوة
            quantity = max(min_qty, (quantity // step_size) * step_size)
    
    # حساب القيمة الاسمية الفعلية
    notional_value = quantity * entry_price
    
    return quantity, notional_value

def check_min_notional(symbol: str, quantity: float, price: float) -> bool:
    """
    فحص ما إذا كانت قيمة الصفقة تفي بالحد الأدنى للمنصة
    """
    symbol_info = exchange_info_map.get(symbol, {})
    if not symbol_info:
        return True
    
    notional_filter = next((f for f in symbol_info.get('filters', []) if f['filterType'] == 'MIN_NOTIONAL'), None)
    if not notional_filter:
        return True
    
    min_notional = float(notional_filter['minNotional'])
    notional_value = quantity * price
    
    return notional_value >= min_notional

def execute_trade(signal: Dict, is_real_trade: bool) -> bool:
    """
    تنفيذ الصفقة (حقيقية أو ورقية)
    """
    global open_signals_cache
    
    symbol = signal['symbol']
    entry_price = signal['entry_price']
    stop_loss = signal['stop_loss']
    
    # حساب حجم الصفقة
    quantity, notional_value = calculate_position_size(symbol, entry_price, stop_loss, is_real_trade)
    
    # فحص الحد الأدنى للقيمة الاسمية
    if not check_min_notional(symbol, quantity, entry_price):
        log_rejection(symbol, "MinNotional Filter Failed")
        return False
    
    # فحص الرصيد للصفقات الحقيقية
    if is_real_trade:
        with balance_lock:
            if usdt_balance < notional_value:
                log_rejection(symbol, "Insufficient Balance")
                return False
    
    # فحص صحة حجم الصفقة
    if stop_loss >= entry_price:
        log_rejection(symbol, "Invalid Position Size")
        return False
    
    # حفظ الصفقة في قاعدة البيانات
    if not check_db_connection() or not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals 
                (symbol, entry_price, stop_loss, target_price_1, target_price_2, 
                 strategy_name, signal_details, is_real_trade, quantity, initial_quantity)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                symbol, entry_price, stop_loss, signal['target_price_1'], signal['target_price_2'],
                signal['strategy_name'], json.dumps(signal, cls=NpEncoder), is_real_trade, quantity, quantity
            ))
            
            trade_id = cur.fetchone()['id']
            conn.commit()
            
            # تحديث الكاش
            with signal_cache_lock:
                signal['id'] = trade_id
                signal['status'] = 'open'
                signal['is_real_trade'] = is_real_trade
                signal['quantity'] = quantity
                signal['initial_quantity'] = quantity
                signal['created_at'] = datetime.now(timezone.utc).isoformat()
                open_signals_cache[symbol] = signal.copy()
            
            # إرسال إشعار
            send_trade_open_notification(
                symbol, signal['strategy_name'], entry_price, stop_loss,
                signal['target_price_1'], signal['target_price_2'], quantity,
                is_real_trade, signal['quality_score'], signal['atr_percent'], notional_value
            )
            
            logger.info(f"✅ {'Real' if is_real_trade else 'Paper'} trade opened for {symbol} at {entry_price}")
            return True
            
    except Exception as e:
        logger.error(f"❌ [Trade] Error executing trade for {symbol}: {e}")
        if conn: conn.rollback()
        return False

def process_signal(signal: Dict) -> bool:
    """
    معالجة الإشارة وتنفيذ الصفقة
    """
    global is_trading_enabled, paper_trading_mode, MIN_SIGNAL_QUALITY
    
    symbol = signal['symbol']
    
    # فحص إذا كان التداول مفعلًا
    with trading_status_lock:
        if not is_trading_enabled:
            log_rejection(symbol, "Trading is disabled")
            return False
    
    # فحص جودة الإشارة
    with min_quality_lock:
        if signal['quality_score'] < MIN_SIGNAL_QUALITY:
            log_rejection(symbol, "Low Quality Signal")
            return False
    
    # فحص عدد الصفقات المفتوحة
    if get_open_trades_count() >= MAX_OPEN_TRADES:
        log_rejection(symbol, "Maximum open trades reached")
        return False
    
    # فحص فترة التبريد
    if is_symbol_in_cooldown(symbol):
        log_rejection(symbol, "Symbol in cooldown period")
        return False
    
    # فحص الخسائر المتتالية
    consecutive_losses = get_consecutive_losses(symbol)
    if consecutive_losses >= 3:
        log_rejection(symbol, "Too many consecutive losses")
        return False
    
    # تحديد نوع الصفقة (حقيقية أو ورقية)
    with trading_mode_lock:
        is_real_trade = not paper_trading_mode
    
    # تنفيذ الصفقة
    return execute_trade(signal, is_real_trade)

def update_trailing_stop(symbol: str, current_price: float):
    """
    تحديث وقف الخسارة المتحرك
    """
    global open_signals_cache, trailing_stop_updates
    
    with signal_cache_lock:
        if symbol not in open_signals_cache:
            return
        
        signal = open_signals_cache[symbol].copy()
    
    entry_price = signal['entry_price']
    stop_loss = signal['stop_loss']
    target_price_1 = signal.get('target_price_1', entry_price * 1.02)
    target_price_2 = signal.get('target_price_2', entry_price * 1.04)
    
    # حساب نسبة الربح الحالية
    profit_percent = ((current_price - entry_price) / entry_price) * 100
    
    # تفعيل وقف الخسارة المتحرك عند الوصول إلى نسبة ربح معينة
    if profit_percent >= TRAILING_STOP_ACTIVATION_PROFIT_PERCENT:
        # حساب مسافة وقف الخسارة المتحرك
        trail_distance = (entry_price - stop_loss) * 0.5
        
        # حساب وقف الخسارة الجديد
        new_stop_loss = current_price - trail_distance
        
        # التأكد من أن وقف الخسارة الجديد أعلى من القديم
        if new_stop_loss > stop_loss:
            # تحديث في قاعدة البيانات
            if not check_db_connection() or not conn:
                return
            
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE signals 
                        SET stop_loss = %s, status = 'updated'
                        WHERE symbol = %s AND status IN ('open', 'updated')
                    """, (new_stop_loss, symbol))
                    
                    conn.commit()
                    
                    # تحديث في الكاش
                    with signal_cache_lock:
                        open_signals_cache[symbol]['stop_loss'] = new_stop_loss
                        open_signals_cache[symbol]['status'] = 'updated'
                    
                    # تسجيل التحديث
                    with trade_analysis_lock:
                        trailing_stop_updates[symbol] = {
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'old_stop_loss': stop_loss,
                            'new_stop_loss': new_stop_loss,
                            'current_price': current_price,
                            'profit_percent': profit_percent
                        }
                    
                    # إرسال إشعار
                    send_trade_update_notification(
                        symbol, "stop_loss", stop_loss, new_stop_loss,
                        current_price, profit_percent, signal['is_real_trade']
                    )
                    
                    logger.info(f"✅ Trailing stop updated for {symbol}: {stop_loss:.4f} -> {new_stop_loss:.4f}")
                    
            except Exception as e:
                logger.error(f"❌ [Trade] Error updating trailing stop for {symbol}: {e}")
                if conn: conn.rollback()

def update_dynamic_targets(symbol: str, current_price: float):
    """
    تحديث الأهداف الديناميكية
    """
    global open_signals_cache, dynamic_targets
    
    with signal_cache_lock:
        if symbol not in open_signals_cache:
            return
        
        signal = open_signals_cache[symbol].copy()
    
    entry_price = signal['entry_price']
    target_price_1 = signal.get('target_price_1', entry_price * 1.02)
    target_price_2 = signal.get('target_price_2', entry_price * 1.04)
    
    # حساب نسبة الربح الحالية
    profit_percent = ((current_price - entry_price) / entry_price) * 100
    
    # تحديث الأهداف عند الوصول إلى نسبة ربح معينة
    if profit_percent >= TRAILING_STOP_ACTIVATION_PROFIT_PERCENT * 1.5:
        # حساب الأهداف الجديدة
        atr = signal.get('atr_percent', 1.0) / 100 * entry_price
        new_target_1 = current_price + (atr * 1.2)
        new_target_2 = current_price + (atr * 2.5)
        
        # التأكد من أن الأهداف الجديدة أعلى من القديمة
        if new_target_1 > target_price_1 or new_target_2 > target_price_2:
            # تحديث في قاعدة البيانات
            if not check_db_connection() or not conn:
                return
            
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE signals 
                        SET target_price_1 = %s, target_price_2 = %s, status = 'updated'
                        WHERE symbol = %s AND status IN ('open', 'updated')
                    """, (new_target_1, new_target_2, symbol))
                    
                    conn.commit()
                    
                    # تحديث في الكاش
                    with signal_cache_lock:
                        open_signals_cache[symbol]['target_price_1'] = new_target_1
                        open_signals_cache[symbol]['target_price_2'] = new_target_2
                        open_signals_cache[symbol]['status'] = 'updated'
                    
                    # تسجيل التحديث
                    with trade_analysis_lock:
                        dynamic_targets[symbol] = {
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'old_target_1': target_price_1,
                            'old_target_2': target_price_2,
                            'new_target_1': new_target_1,
                            'new_target_2': new_target_2,
                            'current_price': current_price,
                            'profit_percent': profit_percent
                        }
                    
                    # إرسال إشعار للهدف الأول إذا تم تحديثه
                    if new_target_1 > target_price_1:
                        send_trade_update_notification(
                            symbol, "target_1", target_price_1, new_target_1,
                            current_price, profit_percent, signal['is_real_trade']
                        )
                    
                    # إرسال إشعار للهدف الثاني إذا تم تحديثه
                    if new_target_2 > target_price_2:
                        send_trade_update_notification(
                            symbol, "target_2", target_price_2, new_target_2,
                            current_price, profit_percent, signal['is_real_trade']
                        )
                    
                    logger.info(f"✅ Dynamic targets updated for {symbol}: T1 {target_price_1:.4f}->{new_target_1:.4f}, T2 {target_price_2:.4f}->{new_target_2:.4f}")
                    
            except Exception as e:
                logger.error(f"❌ [Trade] Error updating dynamic targets for {symbol}: {e}")
                if conn: conn.rollback()

def check_trade_exit_conditions(symbol: str, current_price: float) -> Optional[str]:
    """
    فحص شروط الخروج من الصفقة
    """
    with signal_cache_lock:
        if symbol not in open_signals_cache:
            return None
        
        signal = open_signals_cache[symbol].copy()
    
    entry_price = signal['entry_price']
    stop_loss = signal['stop_loss']
    target_price_1 = signal.get('target_price_1', entry_price * 1.02)
    target_price_2 = signal.get('target_price_2', entry_price * 1.04)
    
    # فحص وقف الخسارة
    if current_price <= stop_loss:
        return "stop_loss"
    
    # فحص الأهداف
    if current_price >= target_price_2:
        return "target_2"
    elif current_price >= target_price_1:
        return "target_1"
    
    return None

def close_trade(symbol: str, exit_reason: str, current_price: float):
    """
    إغلاق الصفقة
    """
    global open_signals_cache
    
    with signal_cache_lock:
        if symbol not in open_signals_cache:
            return
        
        signal = open_signals_cache[symbol].copy()
    
    entry_price = signal['entry_price']
    is_real_trade = signal['is_real_trade']
    
    # حساب نسبة الربح/الخسارة
    profit_percent = ((current_price - entry_price) / entry_price) * 100
    
    # تحديث في قاعدة البيانات
    if not check_db_connection() or not conn:
        return
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE signals 
                SET status = 'closed', closing_price = %s, closed_at = NOW(), 
                    profit_percentage = %s, closing_reason = %s
                WHERE symbol = %s AND status IN ('open', 'updated')
            """, (current_price, profit_percent, exit_reason, symbol))
            
            conn.commit()
            
            # تحديث في الكاش
            with signal_cache_lock:
                open_signals_cache[symbol]['status'] = 'closed'
                open_signals_cache[symbol]['closing_price'] = current_price
                open_signals_cache[symbol]['closed_at'] = datetime.now(timezone.utc).isoformat()
                open_signals_cache[symbol]['profit_percentage'] = profit_percent
                open_signals_cache[symbol]['closing_reason'] = exit_reason
            
            # تحديث الرصيد للصفقات الحقيقية
            if is_real_trade:
                with balance_lock:
                    usdt_balance += (current_price * signal['quantity']) - (entry_price * signal['quantity'])
            
            # تحديث الخسائر المتتالية
            if profit_percent < 0:
                increment_consecutive_losses(symbol)
                
                # تفعيل فترة التبريد بعد الخسارة
                if exit_reason == "stop_loss":
                    set_symbol_cooldown(symbol)
            else:
                reset_consecutive_losses(symbol)
            
            # إرسال إشعار
            if exit_reason == "stop_loss":
                log_and_notify("warning", f"🛡️ صفقة {symbol} أغلقت عند وقف الخسارة ({profit_percent:.2f}%)", "trade_closed")
            elif exit_reason == "target_1":
                log_and_notify("info", f"🎯 صفقة {symbol} أغلقت عند الهدف الأول ({profit_percent:.2f}%)", "trade_closed")
            elif exit_reason == "target_2":
                log_and_notify("info", f"🏆 صفقة {symbol} أغلقت عند الهدف الثاني ({profit_percent:.2f}%)", "trade_closed")
            
            logger.info(f"✅ {'Real' if is_real_trade else 'Paper'} trade closed for {symbol} at {current_price} ({exit_reason}, {profit_percent:.2f}%)")
            
    except Exception as e:
        logger.error(f"❌ [Trade] Error closing trade for {symbol}: {e}")
        if conn: conn.rollback()

def analyze_open_trades():
    """
    تحليل الصفقات المفتوحة وتحديثها
    """
    global live_prices, open_signals_cache
    
    # نسخ الصفقات المفتوحة لتجنب تعديل القائمة أثناء التكرار
    with signal_cache_lock:
        open_symbols = list(open_signals_cache.keys())
    
    for symbol in open_symbols:
        try:
            # الحصول على السعر الحالي
            with live_prices_lock:
                if symbol not in live_prices:
                    continue
                
                current_price = live_prices[symbol]
            
            # تحديث وقف الخسارة المتحرك
            update_trailing_stop(symbol, current_price)
            
            # تحديث الأهداف الديناميكية
            update_dynamic_targets(symbol, current_price)
            
            # فحص شروط الخروج
            exit_reason = check_trade_exit_conditions(symbol, current_price)
            if exit_reason:
                close_trade(symbol, exit_reason, current_price)
                
        except Exception as e:
            logger.error(f"❌ [Trade Analysis] Error analyzing trade for {symbol}: {e}")

def start_trade_analysis_thread():
    """
    بدء خيط تحليل الصفقات
    """
    global trade_analysis_thread, stop_trade_analysis
    
    if trade_analysis_thread and trade_analysis_thread.is_alive():
        return
    
    stop_trade_analysis.clear()
    
    def trade_analysis_worker():
        while not stop_trade_analysis.is_set():
            try:
                analyze_open_trades()
                time.sleep(5)  # تحليل كل 5 ثواني
            except Exception as e:
                logger.error(f"❌ [Trade Analysis] Error in trade analysis thread: {e}")
                time.sleep(10)
    
    trade_analysis_thread = Thread(target=trade_analysis_worker, daemon=True)
    trade_analysis_thread.start()
    logger.info("✅ [Trade Analysis] Started trade analysis thread.")

def stop_trade_analysis_thread():
    """
    إيقاف خيط تحليل الصفقات
    """
    global stop_trade_analysis
    
    stop_trade_analysis.set()
    logger.info("✅ [Trade Analysis] Stopped trade analysis thread.")

# --- دوال المسح والبحث عن الإشارات ---
def scan_symbol_for_signals(symbol: str) -> Optional[Dict]:
    """
    البحث عن إشارات لعملة معينة
    """
    try:
        # جلب البيانات
        df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
        if df is None or len(df) < 50:
            log_rejection(symbol, "Insufficient Historical Data")
            return None
        
        # حساب المؤشرات
        df = calculate_all_features(df)
        if df is None or len(df) < 10:
            log_rejection(symbol, "Insufficient Historical Data")
            return None
        
        # تطبيق الاستراتيجيات
        strategies = [
            bb_stoch_strategy,
            macd_ema_strategy,
            ema_rsi_strategy,
            pullback_strategy,
            momentum_volatility_strategy,
            elliott_wave_strategy,
            range_reversal_strategy
        ]
        
        # البحث عن إشارة
        for strategy in strategies:
            try:
                signal = strategy(df, symbol)
                if signal:
                    return signal
            except Exception as e:
                logger.error(f"❌ [Strategy] Error in {strategy.__name__} for {symbol}: {e}")
        
        return None
        
    except Exception as e:
        logger.error(f"❌ [Scan] Error scanning {symbol} for signals: {e}")
        return None

def scan_all_symbols_for_signals():
    """
    البحث عن إشارات في جميع العملات
    """
    global validated_symbols_to_scan, live_prices
    
    # تحديث حالة السوق
    update_market_state()
    
    # نسخ قائمة العملات لتجنب التعديل أثناء التكرار
    symbols_to_scan = validated_symbols_to_scan.copy()
    
    # عشوائية ترتيب العملات لتجنب التحيز
    random.shuffle(symbols_to_scan)
    
    for symbol in symbols_to_scan:
        try:
            # فحص إذا كانت العملة مفتوحة بالفعل
            with signal_cache_lock:
                if symbol in open_signals_cache:
                    continue
            
            # البحث عن إشارة
            signal = scan_symbol_for_signals(symbol)
            if signal:
                # معالجة الإشارة
                process_signal(signal)
                
        except Exception as e:
            logger.error(f"❌ [Scan] Error processing {symbol}: {e}")
        
        # تأخير بين كل عملية بحث لتجنب الحظر
        time.sleep(0.5)

def start_signal_scanning_thread():
    """
    بدء خيط البحث عن الإشارات
    """
    def signal_scanning_worker():
        while True:
            try:
                scan_all_symbols_for_signals()
                # الانتظار حتى بداية الشمعة التالية (5 دقائق)
                now = datetime.now(timezone.utc)
                next_candle = now.replace(second=0, microsecond=0) + timedelta(minutes=5)
                sleep_time = (next_candle - now).total_seconds()
                time.sleep(max(1, sleep_time))
            except Exception as e:
                logger.error(f"❌ [Signal Scanning] Error in signal scanning thread: {e}")
                time.sleep(30)
    
    scanning_thread = Thread(target=signal_scanning_worker, daemon=True)
    scanning_thread.start()
    logger.info("✅ [Signal Scanning] Started signal scanning thread.")

# --- دوال واجهة الويب ---
@app.route('/')
def dashboard():
    """صفحة لوحة التحكم"""
    return render_template_string('''
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم البوت</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
        }
        .card {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .card-header {
            background-color: #0d6efd;
            color: white;
            border-radius: 10px 10px 0 0 !important;
            font-weight: bold;
        }
        .card-body {
            padding: 20px;
        }
        .table {
            margin-bottom: 0;
        }
        .badge {
            font-size: 0.8em;
        }
        .status-active {
            color: #198754;
        }
        .status-inactive {
            color: #dc3545;
        }
        .signal-quality {
            height: 10px;
            border-radius: 5px;
            background-color: #e9ecef;
            margin-top: 5px;
        }
        .signal-quality-fill {
            height: 100%;
            border-radius: 5px;
        }
        .notification-item {
            padding: 10px;
            border-bottom: 1px solid #e9ecef;
        }
        .notification-item:last-child {
            border-bottom: none;
        }
        .rejection-item {
            padding: 10px;
            border-bottom: 1px solid #e9ecef;
        }
        .rejection-item:last-child {
            border-bottom: none;
        }
        .market-state-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-left: 5px;
        }
        .trending {
            background-color: #198754;
        }
        .ranging {
            background-color: #ffc107;
        }
        .volatile {
            background-color: #dc3545;
        }
        .unknown {
            background-color: #6c757d;
        }
        .volatility-low {
            background-color: #198754;
        }
        .volatility-medium {
            background-color: #ffc107;
        }
        .volatility-high {
            background-color: #dc3545;
        }
        .trend-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-left: 5px;
        }
        .trend-bullish {
            background-color: #198754;
        }
        .trend-bearish {
            background-color: #dc3545;
        }
        .trend-neutral {
            background-color: #6c757d;
        }
        .btn-group-sm > .btn {
            padding: 0.25rem 0.5rem;
            font-size: 0.875rem;
            border-radius: 0.2rem;
        }
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h4 class="mb-0">لوحة تحكم البوت التداولي</h4>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-3 mb-3">
                                <div class="card bg-light">
                                    <div class="card-body text-center">
                                        <h5 class="card-title">حالة التداول</h5>
                                        <h3 id="trading-status" class="status-inactive">غير مفعل</h3>
                                        <button id="toggle-trading" class="btn btn-primary btn-sm mt-2">تفعيل التداول</button>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3 mb-3">
                                <div class="card bg-light">
                                    <div class="card-body text-center">
                                        <h5 class="card-title">نوع التداول</h5>
                                        <h3 id="trading-mode">ورقي</h3>
                                        <button id="toggle-mode" class="btn btn-outline-primary btn-sm mt-2">تبديل إلى حقيقي</button>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3 mb-3">
                                <div class="card bg-light">
                                    <div class="card-body text-center">
                                        <h5 class="card-title">الصفقات المفتوحة</h5>
                                        <h3 id="open-trades-count">0</h3>
                                        <div class="text-muted">الحد الأقصى: <span id="max-trades">3</span></div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3 mb-3">
                                <div class="card bg-light">
                                    <div class="card-body text-center">
                                        <h5 class="card-title">الرصيد الحالي</h5>
                                        <h3 id="balance">$0.00</h3>
                                        <div class="text-muted">نوع الرصيد: <span id="balance-type">حقيقي</span></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-md-6 mb-4">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">حالة السوق</h5>
                    </div>
                    <div class="card-body">
                        <div class="row mb-3">
                            <div class="col-md-6">
                                <div class="d-flex align-items-center">
                                    <span>نظام السوق:</span>
                                    <span id="market-regime" class="fw-bold me-2">غير معروف</span>
                                    <span id="market-regime-indicator" class="market-state-indicator unknown"></span>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="d-flex align-items-center">
                                    <span>مستوى التقلب:</span>
                                    <span id="volatility-state" class="fw-bold me-2">متوسط</span>
                                    <span id="volatility-indicator" class="market-state-indicator volatility-medium"></span>
                                </div>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-md-12">
                                <h6>اتجاه الفريمات:</h6>
                                <div class="row">
                                    <div class="col-md-4">
                                        <div class="d-flex align-items-center">
                                            <span>5 دقائق:</span>
                                            <span id="trend-5m" class="fw-bold me-2">محايد</span>
                                            <span id="trend-5m-indicator" class="trend-indicator trend-neutral"></span>
                                        </div>
                                        <div class="small text-muted">
                                            ADX: <span id="adx-5m">0</span>, RSI: <span id="rsi-5m">0</span>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="d-flex align-items-center">
                                            <span>15 دقيقة:</span>
                                            <span id="trend-15m" class="fw-bold me-2">محايد</span>
                                            <span id="trend-15m-indicator" class="trend-indicator trend-neutral"></span>
                                        </div>
                                        <div class="small text-muted">
                                            ADX: <span id="adx-15m">0</span>, RSI: <span id="rsi-15m">0</span>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="d-flex align-items-center">
                                            <span>ساعة:</span>
                                            <span id="trend-1h" class="fw-bold me-2">محايد</span>
                                            <span id="trend-1h-indicator" class="trend-indicator trend-neutral"></span>
                                        </div>
                                        <div class="small text-muted">
                                            ADX: <span id="adx-1h">0</span>, RSI: <span id="rsi-1h">0</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-6 mb-4">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">الإعدادات</h5>
                    </div>
                    <div class="card-body">
                        <form id="settings-form">
                            <div class="row mb-3">
                                <div class="col-md-6">
                                    <label for="min-amount" class="form-label">الحد الأدنى للصفقة (USDT)</label>
                                    <input type="number" class="form-control" id="min-amount" step="0.1" min="1">
                                </div>
                                <div class="col-md-6">
                                    <label for="max-amount" class="form-label">الحد الأقصى للصفقة (USDT)</label>
                                    <input type="number" class="form-control" id="max-amount" step="0.1" min="1">
                                </div>
                            </div>
                            <div class="row mb-3">
                                <div class="col-md-6">
                                    <label for="max-trades-setting" class="form-label">الحد الأقصى للصفقات المفتوحة</label>
                                    <input type="number" class="form-control" id="max-trades-setting" min="1" max="10">
                                </div>
                                <div class="col-md-6">
                                    <label for="min-quality" class="form-label">الحد الأدنى لجودة الإشارة</label>
                                    <input type="number" class="form-control" id="min-quality" min="1" max="100">
                                </div>
                            </div>
                            <div class="mb-3">
                                <label class="form-label">الاستراتيجيات المفعلة</label>
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" id="strategy-bb-stoch">
                                            <label class="form-check-label" for="strategy-bb-stoch">BB + Stoch</label>
                                        </div>
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" id="strategy-macd-ema">
                                            <label class="form-check-label" for="strategy-macd-ema">MACD + EMA</label>
                                        </div>
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" id="strategy-ema-rsi">
                                            <label class="form-check-label" for="strategy-ema-rsi">EMA + RSI</label>
                                        </div>
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" id="strategy-pullback">
                                            <label class="form-check-label" for="strategy-pullback">Pullback</label>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" id="strategy-momentum">
                                            <label class="form-check-label" for="strategy-momentum">Momentum</label>
                                        </div>
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" id="strategy-elliott">
                                            <label class="form-check-label" for="strategy-elliott">Elliott Wave</label>
                                        </div>
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" id="strategy-range">
                                            <label class="form-check-label" for="strategy-range">Range Reversal</label>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="d-grid">
                                <button type="submit" class="btn btn-primary">حفظ الإعدادات</button>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-md-8 mb-4">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">الصفقات المفتوحة</h5>
                        <span class="badge bg-primary" id="open-trades-badge">0</span>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-hover">
                                <thead>
                                    <tr>
                                        <th>العملة</th>
                                        <th>الاستراتيجية</th>
                                        <th>سعر الدخول</th>
                                        <th>الهدف 1</th>
                                        <th>الهدف 2</th>
                                        <th>وقف الخسارة</th>
                                        <th>الربح/الخسارة</th>
                                        <th>الإجراءات</th>
                                    </tr>
                                </thead>
                                <tbody id="open-trades-table">
                                    <tr>
                                        <td colspan="8" class="text-center text-muted">لا توجد صفقات مفتوحة</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-4 mb-4">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">آخر الإشعارات</h5>
                    </div>
                    <div class="card-body" style="max-height: 400px; overflow-y: auto;">
                        <div id="notifications-container">
                            <div class="notification-item text-center text-muted">
                                لا توجد إشعارات
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">سجل الرفض</h5>
                    </div>
                    <div class="card-body" style="max-height: 300px; overflow-y: auto;">
                        <div class="table-responsive">
                            <table class="table table-hover">
                                <thead>
                                    <tr>
                                        <th>الوقت</th>
                                        <th>العملة</th>
                                        <th>سبب الرفض</th>
                                    </tr>
                                </thead>
                                <tbody id="rejection-table">
                                    <tr>
                                        <td colspan="3" class="text-center text-muted">لا توجد سجلات رفض</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // الاتصال بـ WebSocket
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
        const socket = new WebSocket(wsUrl);
        
        // متغيرات عامة
        let openTrades = {};
        let notifications = [];
        let rejections = [];
        let marketState = {
            trend_details_by_tf: {},
            market_regime: "unknown",
            volatility_state: "medium"
        };
        
        // عند الاتصال بـ WebSocket
        socket.onopen = function(event) {
            console.log("WebSocket connected");
            loadInitialData();
        };
        
        // عند استلام رسالة من WebSocket
        socket.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            switch(data.type) {
                case 'price_update':
                    updatePrices(data.payload);
                    break;
                case 'new_notification':
                    addNotification(data.payload);
                    break;
                case 'new_rejection':
                    addRejection(data.payload);
                    break;
                case 'market_state_update':
                    updateMarketState(data.payload);
                    break;
                case 'trade_opened':
                    addOpenTrade(data.payload);
                    break;
                case 'trade_updated':
                    updateOpenTrade(data.payload);
                    break;
                case 'trade_closed':
                    removeOpenTrade(data.payload);
                    break;
            }
        };
        
        // عند انقطاع الاتصال بـ WebSocket
        socket.onclose = function(event) {
            console.log("WebSocket disconnected");
            // محاولة إعادة الاتصال بعد 5 ثواني
            setTimeout(function() {
                window.location.reload();
            }, 5000);
        };
        
        // تحميل البيانات الأولية
        function loadInitialData() {
            // طلب البيانات الأولية من الخادم
            fetch('/api/initial-data')
                .then(response => response.json())
                .then(data => {
                    // تحديث حالة التداول
                    updateTradingStatus(data.trading_enabled);
                    
                    // تحديث نوع التداول
                    updateTradingMode(data.paper_trading_mode);
                    
                    // تحديث الرصيد
                    updateBalance(data.balance);
                    
                    // تحديث الإعدادات
                    updateSettings(data.settings);
                    
                    // تحديث الصفقات المفتوحة
                    data.open_trades.forEach(trade => {
                        addOpenTrade(trade);
                    });
                    
                    // تحديث الإشعارات
                    data.notifications.forEach(notification => {
                        addNotification(notification);
                    });
                    
                    // تحديث سجل الرفض
                    data.rejections.forEach(rejection => {
                        addRejection(rejection);
                    });
                    
                    // تحديث حالة السوق
                    updateMarketState(data.market_state);
                })
                .catch(error => {
                    console.error('Error loading initial data:', error);
                });
        }
        
        // تحديث حالة التداول
        function updateTradingStatus(enabled) {
            const statusElement = document.getElementById('trading-status');
            const toggleButton = document.getElementById('toggle-trading');
            
            if (enabled) {
                statusElement.textContent = 'مفعل';
                statusElement.className = 'status-active';
                toggleButton.textContent = 'إيقاف التداول';
                toggleButton.className = 'btn btn-danger btn-sm mt-2';
            } else {
                statusElement.textContent = 'غير مفعل';
                statusElement.className = 'status-inactive';
                toggleButton.textContent = 'تفعيل التداول';
                toggleButton.className = 'btn btn-primary btn-sm mt-2';
            }
        }
        
        // تحديث نوع التداول
        function updateTradingMode(paperMode) {
            const modeElement = document.getElementById('trading-mode');
            const toggleButton = document.getElementById('toggle-mode');
            
            if (paperMode) {
                modeElement.textContent = 'ورقي';
                toggleButton.textContent = 'تبديل إلى حقيقي';
            } else {
                modeElement.textContent = 'حقيقي';
                toggleButton.textContent = 'تبديل إلى ورقي';
            }
        }
        
        // تحديث الرصيد
        function updateBalance(balance) {
            const balanceElement = document.getElementById('balance');
            balanceElement.textContent = `$${balance.toFixed(2)}`;
        }
        
        // تحديث الإعدادات
        function updateSettings(settings) {
            document.getElementById('min-amount').value = settings.FIXED_TRADE_AMOUNT_MIN_USDT;
            document.getElementById('max-amount').value = settings.FIXED_TRADE_AMOUNT_MAX_USDT;
            document.getElementById('max-trades-setting').value = settings.MAX_OPEN_TRADES;
            document.getElementById('min-quality').value = settings.MIN_SIGNAL_QUALITY;
            document.getElementById('max-trades').textContent = settings.MAX_OPEN_TRADES;
            
            document.getElementById('strategy-bb-stoch').checked = settings.USE_BB_STOCH_STRATEGY;
            document.getElementById('strategy-macd-ema').checked = settings.USE_MACD_EMA_STRATEGY;
            document.getElementById('strategy-ema-rsi').checked = settings.USE_EMA_RSI_STRATEGY;
            document.getElementById('strategy-pullback').checked = settings.USE_PULLBACK_STRATEGY;
            document.getElementById('strategy-momentum').checked = settings.USE_MOMENTUM_VOLATILITY_STRATEGY;
            document.getElementById('strategy-elliott').checked = settings.USE_ELLIOTT_WAVE_STRATEGY;
            document.getElementById('strategy-range').checked = settings.USE_RANGE_REVERSAL_STRATEGY;
        }
        
        // تحديث حالة السوق
        function updateMarketState(state) {
            marketState = state;
            
            // تحديث نظام السوق
            const regimeElement = document.getElementById('market-regime');
            const regimeIndicator = document.getElementById('market-regime-indicator');
            
            regimeElement.textContent = getMarketRegimeText(state.market_regime);
            regimeIndicator.className = `market-state-indicator ${state.market_regime}`;
            
            // تحديث مستوى التقلب
            const volatilityElement = document.getElementById('volatility-state');
            const volatilityIndicator = document.getElementById('volatility-indicator');
            
            volatilityElement.textContent = getVolatilityStateText(state.volatility_state);
            volatilityIndicator.className = `market-state-indicator volatility-${state.volatility_state}`;
            
            // تحديث اتجاه الفريمات
            if (state.trend_details_by_tf) {
                ['5m', '15m', '1h'].forEach(tf => {
                    if (state.trend_details_by_tf[tf]) {
                        const trend = state.trend_details_by_tf[tf].trend;
                        const adx = state.trend_details_by_tf[tf].adx;
                        const rsi = state.trend_details_by_tf[tf].rsi;
                        
                        document.getElementById(`trend-${tf}`).textContent = getTrendText(trend);
                        document.getElementById(`trend-${tf}-indicator`).className = `trend-indicator trend-${trend}`;
                        document.getElementById(`adx-${tf}`).textContent = adx.toFixed(1);
                        document.getElementById(`rsi-${tf}`).textContent = rsi.toFixed(1);
                    }
                });
            }
        }
        
        // الحصول على نص نظام السوق
        function getMarketRegimeText(regime) {
            switch(regime) {
                case 'trending': return 'موجه';
                case 'ranging': return 'جانبي';
                case 'volatile': return 'متقلب';
                default: return 'غير معروف';
            }
        }
        
        // الحصول على نص مستوى التقلب
        function getVolatilityStateText(state) {
            switch(state) {
                case 'low': return 'منخفض';
                case 'medium': return 'متوسط';
                case 'high': return 'مرتفع';
                default: return 'متوسط';
            }
        }
        
        // الحصول على نص الاتجاه
        function getTrendText(trend) {
            switch(trend) {
                case 'bullish': return 'صاعد';
                case 'bearish': return 'هابط';
                default: return 'محايد';
            }
        }
        
        // إضافة صفقة مفتوحة
        function addOpenTrade(trade) {
            openTrades[trade.symbol] = trade;
            updateOpenTradesTable();
        }
        
        // تحديث صفقة مفتوحة
        function updateOpenTrade(trade) {
            if (openTrades[trade.symbol]) {
                openTrades[trade.symbol] = {...openTrades[trade.symbol], ...trade};
                updateOpenTradesTable();
            }
        }
        
        // إزالة صفقة مفتوحة
        function removeOpenTrade(trade) {
            if (openTrades[trade.symbol]) {
                delete openTrades[trade.symbol];
                updateOpenTradesTable();
            }
        }
        
        // تحديث جدول الصفقات المفتوحة
        function updateOpenTradesTable() {
            const tableBody = document.getElementById('open-trades-table');
            const badge = document.getElementById('open-trades-badge');
            const count = document.getElementById('open-trades-count');
            
            const trades = Object.values(openTrades);
            badge.textContent = trades.length;
            count.textContent = trades.length;
            
            if (trades.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="8" class="text-center text-muted">لا توجد صفقات مفتوحة</td></tr>';
                return;
            }
            
            tableBody.innerHTML = trades.map(trade => {
                const profitPercent = ((trade.current_price || trade.entry_price) - trade.entry_price) / trade.entry_price * 100;
                const profitClass = profitPercent >= 0 ? 'text-success' : 'text-danger';
                const profitSign = profitPercent >= 0 ? '+' : '';
                
                return `
                    <tr>
                        <td>${trade.symbol}</td>
                        <td>${getStrategyName(trade.strategy_name)}</td>
                        <td>${trade.entry_price.toFixed(4)}</td>
                        <td>${trade.target_price_1.toFixed(4)}</td>
                        <td>${trade.target_price_2.toFixed(4)}</td>
                        <td>${trade.stop_loss.toFixed(4)}</td>
                        <td class="${profitClass}">${profitSign}${profitPercent.toFixed(2)}%</td>
                        <td>
                            <div class="btn-group btn-group-sm" role="group">
                                <button type="button" class="btn btn-outline-danger" onclick="closeTrade('${trade.symbol}')">
                                    <i class="bi bi-x-circle"></i>
                                </button>
                            </div>
                        </td>
                    </tr>
                `;
            }).join('');
        }
        
        // الحصول على اسم الاستراتيجية
        function getStrategyName(strategyKey) {
            const strategies = {
                'BB_Stoch_Strategy': 'BB + Stoch',
                'MACD_EMA_Strategy': 'MACD + EMA',
                'EMA_RSI_Strategy': 'EMA + RSI',
                'Pullback_Strategy': 'Pullback',
                'Momentum_Volatility_Strategy': 'Momentum',
                'Elliott_Wave_Strategy': 'Elliott Wave',
                'Range_Reversal_Strategy': 'Range Reversal'
            };
            
            return strategies[strategyKey] || strategyKey;
        }
        
        // إضافة إشعار
        function addNotification(notification) {
            notifications.unshift(notification);
            if (notifications.length > 20) {
                notifications = notifications.slice(0, 20);
            }
            updateNotificationsContainer();
        }
        
        // تحديث حاوية الإشعارات
        function updateNotificationsContainer() {
            const container = document.getElementById('notifications-container');
            
            if (notifications.length === 0) {
                container.innerHTML = '<div class="notification-item text-center text-muted">لا توجد إشعارات</div>';
                return;
            }
            
            container.innerHTML = notifications.map(notification => {
                const typeClass = getNotificationTypeClass(notification.type);
                const typeIcon = getNotificationTypeIcon(notification.type);
                const time = new Date(notification.timestamp).toLocaleTimeString('ar-SA');
                
                return `
                    <div class="notification-item">
                        <div class="d-flex">
                            <div class="me-2">
                                <i class="${typeIcon} ${typeClass}"></i>
                            </div>
                            <div class="flex-grow-1">
                                <div>${notification.message}</div>
                                <div class="small text-muted">${time}</div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }
        
        // الحصول على صنف نوع الإشعار
        function getNotificationTypeClass(type) {
            switch(type) {
                case 'info': return 'text-info';
                case 'warning': return 'text-warning';
                case 'error': return 'text-danger';
                case 'success': return 'text-success';
                default: return 'text-secondary';
            }
        }
        
        // الحصول على أيقونة نوع الإشعار
        function getNotificationTypeIcon(type) {
            switch(type) {
                case 'info': return 'bi bi-info-circle';
                case 'warning': return 'bi bi-exclamation-triangle';
                case 'error': return 'bi bi-x-circle';
                case 'success': return 'bi bi-check-circle';
                default: return 'bi bi-bell';
            }
        }
        
        // إضافة سجل رفض
        function addRejection(rejection) {
            rejections.unshift(rejection);
            if (rejections.length > 30) {
                rejections = rejections.slice(0, 30);
            }
            updateRejectionTable();
        }
        
        // تحديث جدول الرفض
        function updateRejectionTable() {
            const tableBody = document.getElementById('rejection-table');
            
            if (rejections.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="3" class="text-center text-muted">لا توجد سجلات رفض</td></tr>';
                return;
            }
            
            tableBody.innerHTML = rejections.map(rejection => {
                const time = new Date(rejection.timestamp).toLocaleTimeString('ar-SA');
                
                return `
                    <tr>
                        <td>${time}</td>
                        <td>${rejection.symbol}</td>
                        <td>${rejection.reason}</td>
                    </tr>
                `;
            }).join('');
        }
        
        // تحديث الأسعار
        function updatePrices(prices) {
            // تحديث الصفقات المفتوحة بالأسعار الجديدة
            Object.keys(openTrades).forEach(symbol => {
                if (prices[symbol]) {
                    openTrades[symbol].current_price = prices[symbol];
                }
            });
            
            updateOpenTradesTable();
        }
        
        // إغلاق صفقة
        function closeTrade(symbol) {
            if (confirm(`هل أنت متأكد من إغلاق صفقة ${symbol}؟`)) {
                fetch(`/api/close-trade?symbol=${symbol}`, {
                    method: 'POST'
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('تم إغلاق الصفقة بنجاح');
                    } else {
                        alert('فشل في إغلاق الصفقة: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('Error closing trade:', error);
                    alert('حدث خطأ أثناء إغلاق الصفقة');
                });
            }
        }
        
        // تبديل حالة التداول
        document.getElementById('toggle-trading').addEventListener('click', function() {
            const enabled = document.getElementById('trading-status').textContent === 'مفعل';
            
            fetch('/api/toggle-trading', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ enabled: !enabled })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updateTradingStatus(!enabled);
                } else {
                    alert('فشل في تغيير حالة التداول: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error toggling trading:', error);
                alert('حدث خطأ أثناء تغيير حالة التداول');
            });
        });
        
        // تبديل نوع التداول
        document.getElementById('toggle-mode').addEventListener('click', function() {
            const paperMode = document.getElementById('trading-mode').textContent === 'ورقي';
            
            fetch('/api/toggle-mode', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ paper_mode: !paperMode })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updateTradingMode(!paperMode);
                } else {
                    alert('فشل في تغيير نوع التداول: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error toggling trading mode:', error);
                alert('حدث خطأ أثناء تغيير نوع التداول');
            });
        });
        
        // حفظ الإعدادات
        document.getElementById('settings-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const settings = {
                FIXED_TRADE_AMOUNT_MIN_USDT: parseFloat(document.getElementById('min-amount').value),
                FIXED_TRADE_AMOUNT_MAX_USDT: parseFloat(document.getElementById('max-amount').value),
                MAX_OPEN_TRADES: parseInt(document.getElementById('max-trades-setting').value),
                MIN_SIGNAL_QUALITY: parseInt(document.getElementById('min-quality').value),
                USE_BB_STOCH_STRATEGY: document.getElementById('strategy-bb-stoch').checked,
                USE_MACD_EMA_STRATEGY: document.getElementById('strategy-macd-ema').checked,
                USE_EMA_RSI_STRATEGY: document.getElementById('strategy-ema-rsi').checked,
                USE_PULLBACK_STRATEGY: document.getElementById('strategy-pullback').checked,
                USE_MOMENTUM_VOLATILITY_STRATEGY: document.getElementById('strategy-momentum').checked,
                USE_ELLIOTT_WAVE_STRATEGY: document.getElementById('strategy-elliott').checked,
                USE_RANGE_REVERSAL_STRATEGY: document.getElementById('strategy-range').checked
            };
            
            fetch('/api/save-settings', {
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
                    document.getElementById('max-trades').textContent = settings.MAX_OPEN_TRADES;
                } else {
                    alert('فشل في حفظ الإعدادات: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error saving settings:', error);
                alert('حدث خطأ أثناء حفظ الإعدادات');
            });
        });
    </script>
</body>
</html>
    ''')

@app.route('/api/initial-data')
def get_initial_data():
    """الحصول على البيانات الأولية للوحة التحكم"""
    global open_signals_cache, notifications_cache, rejection_logs_cache, current_market_state
    
    try:
        with trading_status_lock:
            trading_enabled = is_trading_enabled
        
        with trading_mode_lock:
            paper_trading_mode = paper_trading_mode
        
        with balance_lock:
            balance = usdt_balance
        
        with trade_amount_lock:
            fixed_trade_amount_min = FIXED_TRADE_AMOUNT_MIN_USDT
            fixed_trade_amount_max = FIXED_TRADE_AMOUNT_MAX_USDT
        
        with min_quality_lock:
            min_signal_quality = MIN_SIGNAL_QUALITY
        
        with signal_cache_lock:
            open_trades = list(open_signals_cache.values())
        
        with notifications_lock:
            notifications = list(notifications_cache)
        
        with rejection_logs_lock:
            rejections = list(rejection_logs_cache)
        
        with market_state_lock:
            market_state = dict(current_market_state)
        
        settings = {
            'FIXED_TRADE_AMOUNT_MIN_USDT': fixed_trade_amount_min,
            'FIXED_TRADE_AMOUNT_MAX_USDT': fixed_trade_amount_max,
            'MAX_OPEN_TRADES': MAX_OPEN_TRADES,
            'MIN_SIGNAL_QUALITY': min_signal_quality,
            'USE_BB_STOCH_STRATEGY': USE_BB_STOCH_STRATEGY,
            'USE_MACD_EMA_STRATEGY': USE_MACD_EMA_STRATEGY,
            'USE_EMA_RSI_STRATEGY': USE_EMA_RSI_STRATEGY,
            'USE_PULLBACK_STRATEGY': USE_PULLBACK_STRATEGY,
            'USE_MOMENTUM_VOLATILITY_STRATEGY': USE_MOMENTUM_VOLATILITY_STRATEGY,
            'USE_ELLIOTT_WAVE_STRATEGY': USE_ELLIOTT_WAVE_STRATEGY,
            'USE_RANGE_REVERSAL_STRATEGY': USE_RANGE_REVERSAL_STRATEGY
        }
        
        return jsonify({
            'trading_enabled': trading_enabled,
            'paper_trading_mode': paper_trading_mode,
            'balance': balance,
            'settings': settings,
            'open_trades': open_trades,
            'notifications': notifications,
            'rejections': rejections,
            'market_state': market_state
        })
    
    except Exception as e:
        logger.error(f"❌ [API] Error getting initial data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/toggle-trading', methods=['POST'])
def toggle_trading():
    """تبديل حالة التداول"""
    global is_trading_enabled
    
    try:
        data = request.get_json()
        enabled = data.get('enabled', not is_trading_enabled)
        
        with trading_status_lock:
            is_trading_enabled = enabled
        
        log_and_notify('info', f"🔄 التداول {'مفعل' if enabled else 'معطل'} من لوحة التحكم", 'system')
        
        return jsonify({'success': True, 'enabled': enabled})
    
    except Exception as e:
        logger.error(f"❌ [API] Error toggling trading: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/toggle-mode', methods=['POST'])
def toggle_mode():
    """تبديل نوع التداول (حقيقي/ورقي)"""
    global paper_trading_mode
    
    try:
        data = request.get_json()
        paper_mode = data.get('paper_mode', not paper_trading_mode)
        
        with trading_mode_lock:
            paper_trading_mode = paper_mode
        
        log_and_notify('info', f"🔄 نوع التداول تغير إلى {'ورقي' if paper_mode else 'حقيقي'}", 'system')
        
        return jsonify({'success': True, 'paper_mode': paper_mode})
    
    except Exception as e:
        logger.error(f"❌ [API] Error toggling trading mode: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/save-settings', methods=['POST'])
def save_settings():
    """حفظ الإعدادات"""
    global FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT, MAX_OPEN_TRADES, MIN_SIGNAL_QUALITY
    global USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY
    global USE_MOMENTUM_VOLATILITY_STRATEGY, USE_ELLIOTT_WAVE_STRATEGY, USE_RANGE_REVERSAL_STRATEGY
    
    try:
        data = request.get_json()
        
        with trade_amount_lock:
            FIXED_TRADE_AMOUNT_MIN_USDT = data.get('FIXED_TRADE_AMOUNT_MIN_USDT', FIXED_TRADE_AMOUNT_MIN_USDT)
            FIXED_TRADE_AMOUNT_MAX_USDT = data.get('FIXED_TRADE_AMOUNT_MAX_USDT', FIXED_TRADE_AMOUNT_MAX_USDT)
        
        MAX_OPEN_TRADES = data.get('MAX_OPEN_TRADES', MAX_OPEN_TRADES)
        
        with min_quality_lock:
            MIN_SIGNAL_QUALITY = data.get('MIN_SIGNAL_QUALITY', MIN_SIGNAL_QUALITY)
        
        USE_BB_STOCH_STRATEGY = data.get('USE_BB_STOCH_STRATEGY', USE_BB_STOCH_STRATEGY)
        USE_MACD_EMA_STRATEGY = data.get('USE_MACD_EMA_STRATEGY', USE_MACD_EMA_STRATEGY)
        USE_EMA_RSI_STRATEGY = data.get('USE_EMA_RSI_STRATEGY', USE_EMA_RSI_STRATEGY)
        USE_PULLBACK_STRATEGY = data.get('USE_PULLBACK_STRATEGY', USE_PULLBACK_STRATEGY)
        USE_MOMENTUM_VOLATILITY_STRATEGY = data.get('USE_MOMENTUM_VOLATILITY_STRATEGY', USE_MOMENTUM_VOLATILITY_STRATEGY)
        USE_ELLIOTT_WAVE_STRATEGY = data.get('USE_ELLIOTT_WAVE_STRATEGY', USE_ELLIOTT_WAVE_STRATEGY)
        USE_RANGE_REVERSAL_STRATEGY = data.get('USE_RANGE_REVERSAL_STRATEGY', USE_RANGE_REVERSAL_STRATEGY)
        
        # حفظ الإعدادات في Redis
        save_settings_to_redis()
        
        log_and_notify('info', "🔄 تم تحديث الإعدادات من لوحة التحكم", 'system')
        
        return jsonify({'success': True})
    
    except Exception as e:
        logger.error(f"❌ [API] Error saving settings: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/close-trade', methods=['POST'])
def close_trade_api():
    """إغلاق صفقة"""
    global live_prices
    
    try:
        symbol = request.args.get('symbol')
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol is required'}), 400
        
        # الحصول على السعر الحالي
        with live_prices_lock:
            if symbol not in live_prices:
                return jsonify({'success': False, 'error': 'Price not available'}), 400
            
            current_price = live_prices[symbol]
        
        # إغلاق الصفقة
        close_trade(symbol, 'manual_close', current_price)
        
        return jsonify({'success': True})
    
    except Exception as e:
        logger.error(f"❌ [API] Error closing trade: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@sock.route('/ws')
def websocket_connection(ws):
    """معالجة اتصالات WebSocket"""
    with ws_clients_lock:
        ws_clients.append(ws)
    
    try:
        while True:
            # استلام الرسائل (ليست هناك حاجة لمعالجتها في هذه الحالة)
            data = ws.receive()
            # يمكن معالجة الرسائل الواردة هنا إذا لزم الأمر
    except Exception as e:
        logger.warning(f"WebSocket connection closed: {e}")
    finally:
        with ws_clients_lock:
            if ws in ws_clients:
                ws_clients.remove(ws)

# --- دالة رئيسية ---
def main():
    """دالة التشغيل الرئيسية"""
    global client
    
    try:
        # تهيئة العميل
        client = Client(API_KEY, API_SECRET)
        
        # تهيئة قاعدة البيانات
        init_db()
        
        # تهيئة Redis
        init_redis()
        
        # تحميل الإعدادات
        load_settings_from_redis()
        
        # تحميل العملات الصالحة
        global validated_symbols_to_scan
        validated_symbols_to_scan = get_validated_symbols()
        
        # تحميل الصفقات المفتوحة
        load_open_signals_to_cache()
        
        # تحميل الإشعارات
        load_notifications_to_cache()
        
        # بدء WebSocket
        start_websocket()
        
        # بدء تحليل الصفقات
        start_trade_analysis_thread()
        
        # بدء البحث عن الإشارات
        start_signal_scanning_thread()
        
        # بدء التقارير الدورية
        start_periodic_reports()
        
        # بدء تطبيق Flask
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except Exception as e:
        logger.critical(f"❌ [Main] Error in main function: {e}")
        exit(1)

if __name__ == '__main__':
    main()