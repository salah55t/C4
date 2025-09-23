# ملف c4_5min_v35_0_1.py - نسخة V35.0.1 (لوحة تحكم عربية وإصلاح تفعيل التداول)
# --- وصف التعديلات:
# 1. [تحويل لوحة التحكم] تحويل واجهة لوحة التحكم بالكامل إلى اللغة العربية
# 2. [إصلاح تفعيل التداول] إصلاح مشكلة عدم استجابة البوت عند تفعيل التداول من لوحة التحكم
# 3. [تحسين اتجاه النصوص] تعديل اتجاه النصوص والواجهة لتكون مناسبة للغة العربية
# 4. [تحسين الأداء] تحسين أداء WebSocket ومعالجة الطلبات

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
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * trend_strength_multiplier
    }

def check_pullback_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    
    # تعديل العتبات بناءً على حالة السوق
    with market_state_lock:
        market_regime = current_market_state.get("market_regime", "unknown")
        volatility_state = current_market_state.get("volatility_state", "medium")
    
    # تعديل عمق الارتداد بناءً على حالة السوق
    if volatility_state == "high":
        pullback_threshold = 0.038  # 3.8%
    elif market_regime == "trending":
        pullback_threshold = 0.025  # 2.5%
    else:
        pullback_threshold = 0.032  # 3.2%
    
    # حساب عمق الارتداد
    recent_high = df['high'].rolling(10).max().iloc[-1]
    pullback_depth = (recent_high - last_row['close']) / recent_high
    
    # تعديل عتبة الحجم بناءً على حالة السوق
    volume_ma = df['volume'].rolling(20).mean()
    if volatility_state == "high":
        volume_multiplier = 1.2
    elif market_regime == "trending":
        volume_multiplier = 1.0
    else:
        volume_multiplier = 1.1
    
    # تعديل عتبة التعافي بناءً على حالة السوق
    if volatility_state == "high":
        recovery_threshold = 0.015  # 1.5%
    elif market_regime == "trending":
        recovery_threshold = 0.008  # 0.8%
    else:
        recovery_threshold = 0.012  # 1.2%
    
    # حساب التعافي
    recent_low = df['low'].rolling(5).min().iloc[-1]
    recovery = (last_row['close'] - recent_low) / recent_low
    
    return {
        'pullback_depth_ok': pullback_depth > pullback_threshold,
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * volume_multiplier,
        'recovery_ok': recovery > recovery_threshold
    }

def check_momentum_volatility_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    
    # تعديل العتبات بناءً على حالة السوق
    with market_state_lock:
        market_regime = current_market_state.get("market_regime", "unknown")
        volatility_state = current_market_state.get("volatility_state", "medium")
    
    # تعديل عتبة الزخم بناءً على حالة السوق
    if volatility_state == "high":
        momentum_threshold = 0.025  # 2.5%
    elif market_regime == "trending":
        momentum_threshold = 0.015  # 1.5%
    else:
        momentum_threshold = 0.020  # 2.0%
    
    # حساب الزخم
    price_change = (last_row['close'] - df['close'].iloc[-5]) / df['close'].iloc[-5]
    
    # تعديل عتبة التقلب بناءً على حالة السوق
    if volatility_state == "high":
        volatility_multiplier = 1.2
    elif market_regime == "trending":
        volatility_multiplier = 0.8
    else:
        volatility_multiplier = 1.0
    
    # حساب التقلب
    atr_percent = last_row['atr_percent']
    volatility_ok = atr_percent > (1.5 * volatility_multiplier)
    
    # تعديل عتبة الحجم بناءً على حالة السوق
    volume_ma = df['volume'].rolling(20).mean()
    if volatility_state == "high":
        volume_multiplier = 1.3
    elif market_regime == "trending":
        volume_multiplier = 1.1
    else:
        volume_multiplier = 1.2
    
    return {
        'momentum_ok': price_change > momentum_threshold,
        'volatility_ok': volatility_ok,
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * volume_multiplier
    }

def check_elliott_wave_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    
    # تعديل العتبات بناءً على حالة السوق
    with market_state_lock:
        market_regime = current_market_state.get("market_regime", "unknown")
        volatility_state = current_market_state.get("volatility_state", "medium")
    
    # تعديل عتبة ADX بناءً على حالة السوق
    if volatility_state == "high":
        adx_threshold = 20
    elif market_regime == "trending":
        adx_threshold = 15
    else:
        adx_threshold = 18
    
    # تعديل عتبة تصحيح فيبوناتشي بناءً على حالة السوق
    if volatility_state == "high":
        fib_min, fib_max = 0.382, 0.786
    elif market_regime == "trending":
        fib_min, fib_max = 0.5, 0.618
    else:
        fib_min, fib_max = 0.382, 0.618
    
    # حساب تصحيح فيبوناتشي
    fib_retracement = get_wave_retracement(df)
    
    # تعديل عتبة RSI بناءً على حالة السوق
    if volatility_state == "high":
        rsi_min, rsi_max = 35, 65
    elif market_regime == "trending":
        rsi_min, rsi_max = 40, 60
    else:
        rsi_min, rsi_max = 30, 70
    
    return {
        'adx_ok': last_row['adx'] > adx_threshold,
        'fib_ok': fib_min <= fib_retracement <= fib_max,
        'rsi_ok': rsi_min <= last_row['rsi'] <= rsi_max
    }

def check_range_reversal_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    
    # تعديل العتبات بناءً على حالة السوق
    with market_state_lock:
        market_regime = current_market_state.get("market_regime", "unknown")
        volatility_state = current_market_state.get("volatility_state", "medium")
    
    # تعديل عتبة ADX (يجب أن يكون منخفضًا لاستراتيجية الانعكاس النطاقي)
    if volatility_state == "high":
        adx_threshold = 18
    elif market_regime == "ranging":
        adx_threshold = 15
    else:
        adx_threshold = 16
    
    # تعديل عتبة RSI بناءً على حالة السوق
    if volatility_state == "high":
        rsi_oversold = 25
    elif market_regime == "ranging":
        rsi_oversold = 30
    else:
        rsi_oversold = 28
    
    # تعديل عتبة عرض البولينجر بناءً على حالة السوق
    bb_width = df['bb_width']
    if volatility_state == "high":
        bb_threshold = bb_width.rolling(20).mean() * 0.8
    elif market_regime == "ranging":
        bb_threshold = bb_width.rolling(20).mean() * 0.6
    else:
        bb_threshold = bb_width.rolling(20).mean() * 0.7
    
    # تعديل عتبة الحجم بناءً على حالة السوق
    volume_ma = df['volume'].rolling(20).mean()
    if volatility_state == "high":
        volume_multiplier = 1.4
    elif market_regime == "ranging":
        volume_multiplier = 1.2
    else:
        volume_multiplier = 1.3
    
    return {
        'adx_ok': last_row['adx'] < adx_threshold,
        'rsi_ok': last_row['rsi'] < rsi_oversold,
        'bb_width_ok': bb_width.iloc[-1] < bb_threshold.iloc[-1],
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * volume_multiplier
    }

# --- دوال التحقق من الإشارات ---
def check_bb_stoch_strategy(symbol: str, df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    
    # تحقق من أن السعر فوق المتوسط المتحرك طويل الأجل
    if last_row['close'] < last_row['ema50']:
        log_rejection(symbol, "BB: Price below EMA50 (bearish trend)")
        return {"valid": False, "reason": "BB: Price below EMA50 (bearish trend)"}
    
    # تحقق من أن السعر قريب من الحد السفلي لبولينجر
    bb_position = (last_row['close'] - last_row['bb_lower']) / (last_row['bb_upper'] - last_row['bb_lower'])
    if bb_position > 0.3:
        log_rejection(symbol, "BB: Price not near lower band")
        return {"valid": False, "reason": "BB: Price not near lower band"}
    
    # تحقق من أن الستوكاستيك في منطقة تشبع البيع
    if last_row['stoch_k'] > 30 or last_row['stoch_d'] > 30:
        log_rejection(symbol, "Stoch: Not in oversold zone")
        return {"valid": False, "reason": "Stoch: Not in oversold zone"}
    
    # تحقق من أن الستوكاستيك يبدأ في الصعود
    if last_row['stoch_k'] < last_row['stoch_d']:
        log_rejection(symbol, "Stoch: Not crossing upwards")
        return {"valid": False, "reason": "Stoch: Not crossing upwards"}
    
    # تطبيق الفلاتر الديناميكية
    dynamic_filters = check_bb_stoch_dynamic_filters(df)
    if not dynamic_filters['bb_width_ok']:
        log_rejection(symbol, "DYN_BB_WIDTH_LOW")
        return {"valid": False, "reason": "DYN_BB_WIDTH_LOW"}
    
    if not dynamic_filters['stoch_ok']:
        log_rejection(symbol, "DYN_STOCH_LOW")
        return {"valid": False, "reason": "DYN_STOCH_LOW"}
    
    if not dynamic_filters['volume_ok']:
        log_rejection(symbol, "DYN_VOLUME_LOW")
        return {"valid": False, "reason": "DYN_VOLUME_LOW"}
    
    # حساب مستويات وقف الخسارة والأهداف
    atr = last_row['atr']
    stop_loss = last_row['close'] - (atr * 1.5)
    target_1 = last_row['close'] + (atr * 2.0)
    target_2 = last_row['close'] + (atr * 3.5)
    
    # حساب جودة الإشارة
    quality_score = 0
    
    # جودة الستوكاستيك
    if last_row['stoch_k'] < 20:
        quality_score += 20
    elif last_row['stoch_k'] < 25:
        quality_score += 15
    elif last_row['stoch_k'] < 30:
        quality_score += 10
    
    # جودة موقف السعر بالنسبة لبولينجر
    if bb_position < 0.15:
        quality_score += 20
    elif bb_position < 0.25:
        quality_score += 15
    elif bb_position < 0.3:
        quality_score += 10
    
    # جودة الحجم
    volume_ratio = last_row['volume'] / df['volume'].rolling(20).mean().iloc[-1]
    if volume_ratio > 1.5:
        quality_score += 20
    elif volume_ratio > 1.2:
        quality_score += 15
    elif volume_ratio > 1.0:
        quality_score += 10
    
    # جودة اتجاه السوق
    if last_row['ema9'] > last_row['ema21'] > last_row['ema50']:
        quality_score += 20
    elif last_row['ema9'] > last_row['ema21']:
        quality_score += 15
    elif last_row['close'] > last_row['ema50']:
        quality_score += 10
    
    # جودة تقلب السوق
    if 1.0 < last_row['atr_percent'] < 2.0:
        quality_score += 20
    elif 0.8 < last_row['atr_percent'] < 2.5:
        quality_score += 15
    elif 0.5 < last_row['atr_percent'] < 3.0:
        quality_score += 10
    
    return {
        "valid": True,
        "strategy": "BB_Stoch_Strategy",
        "entry_price": last_row['close'],
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "quality_score": min(100, quality_score),
        "atr_percent": last_row['atr_percent']
    }

def check_macd_ema_strategy(symbol: str, df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    
    # تحقق من أن MACD فوق إشارته
    if last_row['macd'] < last_row['macd_signal']:
        log_rejection(symbol, "MACD: Below signal line")
        return {"valid": False, "reason": "MACD: Below signal line"}
    
    # تحقق من أن MACD يتجه للأعلى
    if last_row['macd_hist'] < 0 and last_row['macd_hist'] < df['macd_hist'].iloc[-2]:
        log_rejection(symbol, "MACD Momentum Negative")
        return {"valid": False, "reason": "MACD Momentum Negative"}
    
    # تحقق من أن السعر فوق المتوسطات المتحركة
    if last_row['close'] < last_row['ema9'] or last_row['close'] < last_row['ema21']:
        log_rejection(symbol, "MACD: Price below EMAs")
        return {"valid": False, "reason": "MACD: Price below EMAs"}
    
    # تحقق من أن المتوسطات المتحركة في الترتيب الصحيح
    if not (last_row['ema9'] > last_row['ema21'] > last_row['ema50']):
        log_rejection(symbol, "MACD: EMAs not in bullish order")
        return {"valid": False, "reason": "MACD: EMAs not in bullish order"}
    
    # تطبيق الفلاتر الديناميكية
    dynamic_filters = check_macd_ema_dynamic_filters(df)
    if not dynamic_filters['adx_ok']:
        log_rejection(symbol, "DYN_ADX_LOW")
        return {"valid": False, "reason": "DYN_ADX_LOW"}
    
    if not dynamic_filters['volume_ok']:
        log_rejection(symbol, "DYN_VOLUME_LOW")
        return {"valid": False, "reason": "DYN_VOLUME_LOW"}
    
    if not dynamic_filters['momentum_ok']:
        log_rejection(symbol, "DYN_MACD_MOMENTUM_LOW")
        return {"valid": False, "reason": "DYN_MACD_MOMENTUM_LOW"}
    
    # حساب مستويات وقف الخسارة والأهداف
    atr = last_row['atr']
    stop_loss = last_row['close'] - (atr * 1.2)
    target_1 = last_row['close'] + (atr * 2.5)
    target_2 = last_row['close'] + (atr * 4.0)
    
    # حساب جودة الإشارة
    quality_score = 0
    
    # جودة MACD
    macd_distance = (last_row['macd'] - last_row['macd_signal']) / abs(last_row['macd_signal'])
    if macd_distance > 0.1:
        quality_score += 20
    elif macd_distance > 0.05:
        quality_score += 15
    elif macd_distance > 0.02:
        quality_score += 10
    
    # جودة زخم MACD
    if last_row['macd_hist'] > 0 and last_row['macd_hist'] > df['macd_hist'].iloc[-2]:
        quality_score += 20
    elif last_row['macd_hist'] > df['macd_hist'].iloc[-2]:
        quality_score += 15
    elif last_row['macd_hist'] > 0:
        quality_score += 10
    
    # جودة ترتيب المتوسطات المتحركة
    ema_spread = (last_row['ema9'] - last_row['ema50']) / last_row['ema50']
    if ema_spread > 0.02:
        quality_score += 20
    elif ema_spread > 0.01:
        quality_score += 15
    elif ema_spread > 0.005:
        quality_score += 10
    
    # جودة الحجم
    volume_ratio = last_row['volume'] / df['volume'].rolling(20).mean().iloc[-1]
    if volume_ratio > 1.5:
        quality_score += 20
    elif volume_ratio > 1.2:
        quality_score += 15
    elif volume_ratio > 1.0:
        quality_score += 10
    
    # جودة ADX
    if last_row['adx'] > 25:
        quality_score += 20
    elif last_row['adx'] > 20:
        quality_score += 15
    elif last_row['adx'] > 15:
        quality_score += 10
    
    return {
        "valid": True,
        "strategy": "MACD_EMA_Strategy",
        "entry_price": last_row['close'],
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "quality_score": min(100, quality_score),
        "atr_percent": last_row['atr_percent']
    }

def check_ema_rsi_strategy(symbol: str, df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    
    # تحقق من أن RSI في منطقة تشبع البيع
    if last_row['rsi'] > 40:
        log_rejection(symbol, "RSI Out of Range")
        return {"valid": False, "reason": "RSI Out of Range"}
    
    # تحقق من أن السعر فوق المتوسط المتحرك طويل الأجل
    if last_row['close'] < last_row['ema50']:
        log_rejection(symbol, "EMA_RSI: Bearish long-term trend")
        return {"valid": False, "reason": "EMA_RSI: Bearish long-term trend"}
    
    # تحقق من أن RSI يبدأ في الصعود
    if last_row['rsi'] < df['rsi'].iloc[-2]:
        log_rejection(symbol, "RSI: Not turning upwards")
        return {"valid": False, "reason": "RSI: Not turning upwards"}
    
    # تحقق من أن المتوسطات المتحركة في الترتيب الصحيح
    if not (last_row['ema9'] > last_row['ema21']):
        log_rejection(symbol, "EMA: Short-term EMAs not in bullish order")
        return {"valid": False, "reason": "EMA: Short-term EMAs not in bullish order"}
    
    # تطبيق الفلاتر الديناميكية
    dynamic_filters = check_ema_rsi_dynamic_filters(df)
    if not dynamic_filters['rsi_in_range']:
        log_rejection(symbol, "DYN_RSI_OOR")
        return {"valid": False, "reason": "DYN_RSI_OOR"}
    
    if not dynamic_filters['ema_spread_ok']:
        log_rejection(symbol, "DYN_EMA_SPREAD_LOW")
        return {"valid": False, "reason": "DYN_EMA_SPREAD_LOW"}
    
    if not dynamic_filters['volume_ok']:
        log_rejection(symbol, "DYN_VOLUME_LOW")
        return {"valid": False, "reason": "DYN_VOLUME_LOW"}
    
    # حساب مستويات وقف الخسارة والأهداف
    atr = last_row['atr']
    stop_loss = last_row['close'] - (atr * 1.3)
    target_1 = last_row['close'] + (atr * 2.2)
    target_2 = last_row['close'] + (atr * 3.8)
    
    # حساب جودة الإشارة
    quality_score = 0
    
    # جودة RSI
    if last_row['rsi'] < 30:
        quality_score += 20
    elif last_row['rsi'] < 35:
        quality_score += 15
    elif last_row['rsi'] < 40:
        quality_score += 10
    
    # جودة اتجاه RSI
    if last_row['rsi'] > df['rsi'].iloc[-2] and df['rsi'].iloc[-2] > df['rsi'].iloc[-3]:
        quality_score += 20
    elif last_row['rsi'] > df['rsi'].iloc[-2]:
        quality_score += 15
    elif last_row['rsi'] > df['rsi'].iloc[-3]:
        quality_score += 10
    
    # جودة تباعد المتوسطات المتحركة
    ema_spread = (last_row['ema9'] - last_row['ema21']) / last_row['ema21']
    if ema_spread > 0.01:
        quality_score += 20
    elif ema_spread > 0.005:
        quality_score += 15
    elif ema_spread > 0.002:
        quality_score += 10
    
    # جودة الحجم
    volume_ratio = last_row['volume'] / df['volume'].rolling(20).mean().iloc[-1]
    if volume_ratio > 1.5:
        quality_score += 20
    elif volume_ratio > 1.2:
        quality_score += 15
    elif volume_ratio > 1.0:
        quality_score += 10
    
    # جودة ADX
    if last_row['adx'] > 25:
        quality_score += 20
    elif last_row['adx'] > 20:
        quality_score += 15
    elif last_row['adx'] > 15:
        quality_score += 10
    
    return {
        "valid": True,
        "strategy": "EMA_RSI_Strategy",
        "entry_price": last_row['close'],
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "quality_score": min(100, quality_score),
        "atr_percent": last_row['atr_percent']
    }

def check_pullback_strategy(symbol: str, df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    
    # تحقق من أن الاتجاه العام صاعد
    if last_row['ema50'] < last_row['ema200']:
        log_rejection(symbol, "Pullback: Bearish long-term trend")
        return {"valid": False, "reason": "Pullback: Bearish long-term trend"}
    
    # تحقق من أن السعر يرتد من قمة حديثة
    recent_high = df['high'].rolling(10).max().iloc[-1]
    if (recent_high - last_row['close']) / recent_high < 0.02:  # أقل من 2% ارتداد
        log_rejection(symbol, "Pullback: Insufficient pullback depth")
        return {"valid": False, "reason": "Pullback: Insufficient pullback depth"}
    
    # تحقق من أن السعر بدأ في التعافي
    recent_low = df['low'].rolling(5).min().iloc[-1]
    if (last_row['close'] - recent_low) / recent_low < 0.01:  # أقل من 1% تعافي
        log_rejection(symbol, "Pullback: No recovery sign")
        return {"valid": False, "reason": "Pullback: No recovery sign"}
    
    # تحقق من أن الحجم يتزايد
    if last_row['volume'] < df['volume'].rolling(10).mean().iloc[-1]:
        log_rejection(symbol, "Pullback: Low volume")
        return {"valid": False, "reason": "Pullback: Low volume"}
    
    # تطبيق الفلاتر الديناميكية
    dynamic_filters = check_pullback_dynamic_filters(df)
    if not dynamic_filters['pullback_depth_ok']:
        log_rejection(symbol, "DYN_PULLBACK_SHALLOW")
        return {"valid": False, "reason": "DYN_PULLBACK_SHALLOW"}
    
    if not dynamic_filters['volume_ok']:
        log_rejection(symbol, "DYN_VOLUME_LOW")
        return {"valid": False, "reason": "DYN_VOLUME_LOW"}
    
    if not dynamic_filters['recovery_ok']:
        log_rejection(symbol, "DYN_RECOVERY_FAIL")
        return {"valid": False, "reason": "DYN_RECOVERY_FAIL"}
    
    # حساب مستويات وقف الخسارة والأهداف
    atr = last_row['atr']
    stop_loss = recent_low - (atr * 0.5)
    target_1 = last_row['close'] + (atr * 2.0)
    target_2 = recent_high + (atr * 1.0)
    
    # حساب جودة الإشارة
    quality_score = 0
    
    # جودة عمق الارتداد
    pullback_depth = (recent_high - last_row['close']) / recent_high
    if 0.03 < pullback_depth < 0.05:
        quality_score += 20
    elif 0.025 < pullback_depth < 0.06:
        quality_score += 15
    elif 0.02 < pullback_depth < 0.07:
        quality_score += 10
    
    # جودة التعافي
    recovery = (last_row['close'] - recent_low) / recent_low
    if 0.015 < recovery < 0.03:
        quality_score += 20
    elif 0.01 < recovery < 0.04:
        quality_score += 15
    elif 0.008 < recovery < 0.05:
        quality_score += 10
    
    # جودة الحجم
    volume_ratio = last_row['volume'] / df['volume'].rolling(20).mean().iloc[-1]
    if volume_ratio > 1.5:
        quality_score += 20
    elif volume_ratio > 1.2:
        quality_score += 15
    elif volume_ratio > 1.0:
        quality_score += 10
    
    # جودة الاتجاه العام
    if last_row['ema50'] > last_row['ema200'] * 1.02:
        quality_score += 20
    elif last_row['ema50'] > last_row['ema200'] * 1.01:
        quality_score += 15
    elif last_row['ema50'] > last_row['ema200']:
        quality_score += 10
    
    # جودة ADX
    if 20 < last_row['adx'] < 30:
        quality_score += 20
    elif 15 < last_row['adx'] < 35:
        quality_score += 15
    elif 10 < last_row['adx'] < 40:
        quality_score += 10
    
    return {
        "valid": True,
        "strategy": "Pullback_Strategy",
        "entry_price": last_row['close'],
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "quality_score": min(100, quality_score),
        "atr_percent": last_row['atr_percent']
    }

def check_momentum_volatility_strategy(symbol: str, df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    
    # تحقق من أن السعر فوق المتوسطات المتحركة
    if last_row['close'] < last_row['ema21']:
        log_rejection(symbol, "Momentum: Price below EMA21")
        return {"valid": False, "reason": "Momentum: Price below EMA21"}
    
    # تحقق من أن السعر يظهر زخمًا إيجابيًا
    if last_row['close'] < df['close'].iloc[-3]:
        log_rejection(symbol, "Momentum: No positive momentum")
        return {"valid": False, "reason": "Momentum: No positive momentum"}
    
    # تحقق من أن المتوسطات المتحركة في الترتيب الصحيح
    if not (last_row['ema9'] > last_row['ema21'] > last_row['ema50']):
        log_rejection(symbol, "Momentum: EMAs not in bullish order")
        return {"valid": False, "reason": "Momentum: EMAs not in bullish order"}
    
    # تحقق من أن التقلب في النطاق المطلوب
    if last_row['atr_percent'] < 1.0:
        log_rejection(symbol, "Momentum: Low volatility")
        return {"valid": False, "reason": "Momentum: Low volatility"}
    
    # تطبيق الفلاتر الديناميكية
    dynamic_filters = check_momentum_volatility_dynamic_filters(df)
    if not dynamic_filters['momentum_ok']:
        log_rejection(symbol, "DYN_MOMENTUM_SCORE_LOW")
        return {"valid": False, "reason": "DYN_MOMENTUM_SCORE_LOW"}
    
    if not dynamic_filters['volatility_ok']:
        log_rejection(symbol, "DYN_VOLATILITY_OOR")
        return {"valid": False, "reason": "DYN_VOLATILITY_OOR"}
    
    if not dynamic_filters['volume_ok']:
        log_rejection(symbol, "DYN_VOLUME_LOW")
        return {"valid": False, "reason": "DYN_VOLUME_LOW"}
    
    # حساب مستويات وقف الخسارة والأهداف
    atr = last_row['atr']
    stop_loss = last_row['close'] - (atr * 1.1)
    target_1 = last_row['close'] + (atr * 2.3)
    target_2 = last_row['close'] + (atr * 4.2)
    
    # حساب جودة الإشارة
    quality_score = 0
    
    # جودة الزخم
    price_change = (last_row['close'] - df['close'].iloc[-5]) / df['close'].iloc[-5]
    if price_change > 0.03:
        quality_score += 20
    elif price_change > 0.02:
        quality_score += 15
    elif price_change > 0.01:
        quality_score += 10
    
    # جودة التقلب
    if 1.5 < last_row['atr_percent'] < 2.5:
        quality_score += 20
    elif 1.2 < last_row['atr_percent'] < 3.0:
        quality_score += 15
    elif 1.0 < last_row['atr_percent'] < 3.5:
        quality_score += 10
    
    # جودة ترتيب المتوسطات المتحركة
    ema_spread = (last_row['ema9'] - last_row['ema50']) / last_row['ema50']
    if ema_spread > 0.015:
        quality_score += 20
    elif ema_spread > 0.01:
        quality_score += 15
    elif ema_spread > 0.005:
        quality_score += 10
    
    # جودة الحجم
    volume_ratio = last_row['volume'] / df['volume'].rolling(20).mean().iloc[-1]
    if volume_ratio > 1.5:
        quality_score += 20
    elif volume_ratio > 1.2:
        quality_score += 15
    elif volume_ratio > 1.0:
        quality_score += 10
    
    # جودة ADX
    if 20 < last_row['adx'] < 30:
        quality_score += 20
    elif 15 < last_row['adx'] < 35:
        quality_score += 15
    elif 10 < last_row['adx'] < 40:
        quality_score += 10
    
    return {
        "valid": True,
        "strategy": "Momentum_Volatility_Strategy",
        "entry_price": last_row['close'],
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "quality_score": min(100, quality_score),
        "atr_percent": last_row['atr_percent']
    }

def check_elliott_wave_strategy(symbol: str, df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    
    # تحقق من أن الاتجاه العام صاعد
    if last_row['ema50'] < last_row['ema200']:
        log_rejection(symbol, "Elliott Wave: Bearish long-term trend")
        return {"valid": False, "reason": "Elliott Wave: Bearish long-term trend"}
    
    # تحقق من أن السعر يرتد من قاع حديث
    recent_low = df['low'].rolling(5).min().iloc[-1]
    if (last_row['close'] - recent_low) / recent_low < 0.01:  # أقل من 1% تعافي
        log_rejection(symbol, "Elliott Wave: No recovery from recent low")
        return {"valid": False, "reason": "Elliott Wave: No recovery from recent low"}
    
    # حساب تصحيح فيبوناتشي
    fib_retracement = get_wave_retracement(df)
    if fib_retracement < 0.382 or fib_retracement > 0.618:
        log_rejection(symbol, "Elliott Wave: Invalid Fibonacci retracement")
        return {"valid": False, "reason": "Elliott Wave: Invalid Fibonacci retracement"}
    
    # تطبيق الفلاتر الديناميكية
    dynamic_filters = check_elliott_wave_dynamic_filters(df)
    if not dynamic_filters['adx_ok']:
        log_rejection(symbol, "DYN_ADX_LOW")
        return {"valid": False, "reason": "DYN_ADX_LOW"}
    
    if not dynamic_filters['fib_ok']:
        log_rejection(symbol, "DYN_FIB_RETRACEMENT_OOR")
        return {"valid": False, "reason": "DYN_FIB_RETRACEMENT_OOR"}
    
    if not dynamic_filters['rsi_ok']:
        log_rejection(symbol, "DYN_RSI_OOR")
        return {"valid": False, "reason": "DYN_RSI_OOR"}
    
    # حساب مستويات وقف الخسارة والأهداف
    atr = last_row['atr']
    stop_loss = recent_low - (atr * 0.5)
    target_1 = last_row['close'] + (atr * 2.5)
    target_2 = last_row['close'] + (atr * 4.5)
    
    # حساب جودة الإشارة
    quality_score = 0
    
    # جودة تصحيح فيبوناتشي
    if 0.5 <= fib_retracement <= 0.618:
        quality_score += 30
    elif 0.382 <= fib_retracement <= 0.5:
        quality_score += 20
    else:
        quality_score += 10
    
    # جودة التعافي
    recovery = (last_row['close'] - recent_low) / recent_low
    if 0.02 < recovery < 0.04:
        quality_score += 20
    elif 0.015 < recovery < 0.05:
        quality_score += 15
    elif 0.01 < recovery < 0.06:
        quality_score += 10
    
    # جودة RSI
    if 30 < last_row['rsi'] < 50:
        quality_score += 20
    elif 25 < last_row['rsi'] < 55:
        quality_score += 15
    elif 20 < last_row['rsi'] < 60:
        quality_score += 10
    
    # جودة ADX
    if 20 < last_row['adx'] < 30:
        quality_score += 20
    elif 15 < last_row['adx'] < 35:
        quality_score += 15
    elif 10 < last_row['adx'] < 40:
        quality_score += 10
    
    # جودة الحجم
    volume_ratio = last_row['volume'] / df['volume'].rolling(20).mean().iloc[-1]
    if volume_ratio > 1.5:
        quality_score += 10
    elif volume_ratio > 1.2:
        quality_score += 7
    elif volume_ratio > 1.0:
        quality_score += 5
    
    return {
        "valid": True,
        "strategy": "Elliott_Wave_Strategy",
        "entry_price": last_row['close'],
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "quality_score": min(100, quality_score),
        "atr_percent": last_row['atr_percent']
    }

def check_range_reversal_strategy(symbol: str, df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    
    # تحقق من أن ADX منخفض (سوق جانبي)
    if last_row['adx'] > 23:
        log_rejection(symbol, "Range Reversal: Trend too strong (ADX > 23)")
        return {"valid": False, "reason": "Range Reversal: Trend too strong (ADX > 23)"}
    
    # تحقق من أن RSI في منطقة تشبع البيع
    if last_row['rsi'] > 35:
        log_rejection(symbol, "Range Reversal: RSI not in oversold zone")
        return {"valid": False, "reason": "Range Reversal: RSI not in oversold zone"}
    
    # تحقق من أن السعر قريب من الحد السفلي لبولينجر
    bb_position = (last_row['close'] - last_row['bb_lower']) / (last_row['bb_upper'] - last_row['bb_lower'])
    if bb_position > 0.25:
        log_rejection(symbol, "Range Reversal: Price not near lower band")
        return {"valid": False, "reason": "Range Reversal: Price not near lower band"}
    
    # تحقق من أن عرض البولينجر ضيق (سوق جانبي)
    if last_row['bb_width'] > 0.04:
        log_rejection(symbol, "Range Reversal: BB width too wide")
        return {"valid": False, "reason": "Range Reversal: BB width too wide"}
    
    # تطبيق الفلاتر الديناميكية
    dynamic_filters = check_range_reversal_dynamic_filters(df)
    if not dynamic_filters['adx_ok']:
        log_rejection(symbol, "Range Reversal: ADX too high")
        return {"valid": False, "reason": "Range Reversal: ADX too high"}
    
    if not dynamic_filters['rsi_ok']:
        log_rejection(symbol, "Range Reversal: RSI not in oversold zone")
        return {"valid": False, "reason": "Range Reversal: RSI not in oversold zone"}
    
    if not dynamic_filters['bb_width_ok']:
        log_rejection(symbol, "Range Reversal: BB width too wide")
        return {"valid": False, "reason": "Range Reversal: BB width too wide"}
    
    if not dynamic_filters['volume_ok']:
        log_rejection(symbol, "DYN_VOLUME_LOW")
        return {"valid": False, "reason": "DYN_VOLUME_LOW"}
    
    # حساب مستويات وقف الخسارة والأهداف
    atr = last_row['atr']
    stop_loss = last_row['close'] - (atr * 1.0)
    target_1 = last_row['close'] + (atr * 1.5)
    target_2 = last_row['close'] + (atr * 2.5)
    
    # حساب جودة الإشارة
    quality_score = 0
    
    # جودة RSI
    if last_row['rsi'] < 25:
        quality_score += 30
    elif last_row['rsi'] < 30:
        quality_score += 25
    elif last_row['rsi'] < 35:
        quality_score += 20
    
    # جودة موقف السعر بالنسبة لبولينجر
    if bb_position < 0.15:
        quality_score += 30
    elif bb_position < 0.2:
        quality_score += 25
    elif bb_position < 0.25:
        quality_score += 20
    
    # جودة عرض البولينجر
    if last_row['bb_width'] < 0.02:
        quality_score += 20
    elif last_row['bb_width'] < 0.03:
        quality_score += 15
    elif last_row['bb_width'] < 0.04:
        quality_score += 10
    
    # جودة ADX
    if last_row['adx'] < 15:
        quality_score += 20
    elif last_row['adx'] < 20:
        quality_score += 15
    elif last_row['adx'] < 23:
        quality_score += 10
    
    return {
        "valid": True,
        "strategy": "Range_Reversal_Strategy",
        "entry_price": last_row['close'],
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "quality_score": min(100, quality_score),
        "atr_percent": last_row['atr_percent']
    }

# --- دوال التحقق من الشروط العامة ---
def check_general_filters(symbol: str, df: pd.DataFrame, signal_quality: int) -> Tuple[bool, Optional[str]]:
    last_row = df.iloc[-1]
    
    # فلتر جودة الإشارة
    with min_quality_lock:
        min_quality = MIN_SIGNAL_QUALITY
    
    if signal_quality < min_quality:
        log_rejection(symbol, "Low Quality Signal", {"quality": signal_quality, "min_required": min_quality})
        return False, "Low Quality Signal"
    
    # فلتر تقلب السوق
    if last_row['atr_percent'] > 5.0:
        log_rejection(symbol, "Market Volatility Filter Failed", {"atr_percent": last_row['atr_percent']})
        return False, "Market Volatility Filter Failed"
    
    # فلتر البيانات التاريخية
    if len(df) < 100:
        log_rejection(symbol, "Insufficient Historical Data", {"candles": len(df)})
        return False, "Insufficient Historical Data"
    
    # فلتر الحد الأدنى للصفقة
    symbol_info = exchange_info_map.get(symbol, {})
    if symbol_info:
        min_notional = float(symbol_info.get('filters', [{}])[6].get('minNotional', 0))
        if min_notional > 0:
            with trade_amount_lock:
                if paper_trading_mode:
                    trade_amount = PAPER_TRADE_FIXED_AMOUNT_USDT
                else:
                    trade_amount = random.uniform(FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT)
            
            if trade_amount < min_notional:
                log_rejection(symbol, "MinNotional Filter Failed", {"trade_amount": trade_amount, "min_notional": min_notional})
                return False, "MinNotional Filter Failed"
    
    # فلتر الرصيد
    if not paper_trading_mode:
        with balance_lock:
            balance = usdt_balance
        
        with trade_amount_lock:
            trade_amount = random.uniform(FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT)
        
        if balance < trade_amount:
            log_rejection(symbol, "Insufficient Balance", {"balance": balance, "required": trade_amount})
            
            # التبديل التلقائي للوضع الورقي إذا كان الرصيد منخفضًا
            if AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE:
                with trading_mode_lock:
                    global paper_trading_mode
                    paper_trading_mode = True
                log_and_notify("warning", f"الرصيد منخفض (${balance:.2f}). تم التبديل تلقائيًا إلى وضع التداول الورقي.", "system")
            
            return False, "Insufficient Balance"
    
    # فلتر حجم الصفقة
    if last_row['close'] <= 0:
        log_rejection(symbol, "Invalid Position Size", {"price": last_row['close']})
        return False, "Invalid Position Size"
    
    return True, None

def check_symbol_cooldown(symbol: str) -> bool:
    with cooldowns_lock:
        if symbol in cooldowns_by_symbol:
            cooldown_end = cooldowns_by_symbol[symbol]
            if datetime.now(timezone.utc) < cooldown_end:
                return True
    
    return False

def apply_symbol_cooldown(symbol: str, minutes: int = COOLDOWN_MINUTES_AFTER_SL):
    with cooldowns_lock:
        cooldowns_by_symbol[symbol] = datetime.now(timezone.utc) + timedelta(minutes=minutes)

def check_consecutive_losses(symbol: str) -> bool:
    with consecutive_losses_lock:
        if symbol not in consecutive_losses_by_symbol:
            consecutive_losses_by_symbol[symbol] = 0
        
        return consecutive_losses_by_symbol[symbol] >= 2

def update_consecutive_losses(symbol: str, is_win: bool):
    with consecutive_losses_lock:
        if symbol not in consecutive_losses_by_symbol:
            consecutive_losses_by_symbol[symbol] = 0
        
        if is_win:
            consecutive_losses_by_symbol[symbol] = 0
        else:
            consecutive_losses_by_symbol[symbol] += 1

# --- دوال تنفيذ الصفقات ---
def calculate_position_size(symbol: str, entry_price: float, stop_loss: float, is_real_trade: bool) -> Tuple[float, float]:
    """
    حساب حجم الصفقة والقيمة الاسمية
    """
    with trade_amount_lock:
        if is_real_trade:
            trade_amount = random.uniform(FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT)
        else:
            trade_amount = PAPER_TRADE_FIXED_AMOUNT_USDT
    
    # حساب حجم الصفقة
    risk_per_unit = entry_price - stop_loss
    if risk_per_unit <= 0:
        return 0.0, 0.0
    
    quantity = trade_amount / entry_price
    
    # تطبيق قيود حجم الصفقة
    symbol_info = exchange_info_map.get(symbol, {})
    if symbol_info:
        # تطبيق LOT_SIZE filter
        lot_size_filter = next((f for f in symbol_info.get('filters', []) if f.get('filterType') == 'LOT_SIZE'), {})
        if lot_size_filter:
            min_qty = float(lot_size_filter.get('minQty', 0))
            max_qty = float(lot_size_filter.get('maxQty', float('inf')))
            step_size = float(lot_size_filter.get('stepSize', 0))
            
            # تعديل الكمية لتكون متوافقة مع step_size
            if step_size > 0:
                quantity = max(min_qty, (quantity // step_size) * step_size)
            
            quantity = max(min_qty, min(max_qty, quantity))
    
    # حساب القيمة الاسمية
    notional_value = quantity * entry_price
    
    return quantity, notional_value

def execute_trade(symbol: str, strategy_name: str, entry_price: float, stop_loss: float,
                target_1: float, target_2: float, quality_score: int, atr_percent: float) -> bool:
    """
    تنفيذ الصفقة (حقيقية أو ورقية)
    """
    global is_trading_enabled
    with trading_status_lock:
        if not is_trading_enabled:
            return False
    
    # التحقق من الحد الأقصى لعدد الصفقات المفتوحة
    with signal_cache_lock:
        open_trades = [s for s in open_signals_cache.values() if s.get('status') in ['open', 'updated']]
        if len(open_trades) >= MAX_OPEN_TRADES:
            log_rejection(symbol, "Maximum open trades reached", {"current": len(open_trades), "max": MAX_OPEN_TRADES})
            return False
    
    # التحقق من وجود صفقة مفتوحة لنفس الرمز
    with signal_cache_lock:
        if symbol in open_signals_cache:
            log_rejection(symbol, "Trade already open for this symbol")
            return False
    
    # التحقق من فترة التبريد
    if check_symbol_cooldown(symbol):
        log_rejection(symbol, "Symbol in cooldown period")
        return False
    
    # التحقق من الخسائر المتتالية
    if check_consecutive_losses(symbol):
        log_rejection(symbol, "Too many consecutive losses")
        return False
    
    # تحديد نوع الصفقة (حقيقية أو ورقية)
    with trading_mode_lock:
        is_real_trade = not paper_trading_mode
    
    # حساب حجم الصفقة
    quantity, notional_value = calculate_position_size(symbol, entry_price, stop_loss, is_real_trade)
    if quantity <= 0 or notional_value <= 0:
        log_rejection(symbol, "Invalid position size", {"quantity": quantity, "notional_value": notional_value})
        return False
    
    # حفظ الصفقة في قاعدة البيانات
    if not check_db_connection() or not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, stop_loss, target_price_1, target_price_2, 
                                    status, strategy_name, signal_details, is_real_trade, quantity, 
                                    initial_quantity, created_at)
                VALUES (%s, %s, %s, %s, %s, 'open', %s, %s, %s, %s, %s, NOW())
                RETURNING id
            """, (
                symbol, entry_price, stop_loss, target_1, target_2,
                strategy_name,
                json.dumps({
                    "quality_score": quality_score,
                    "atr_percent": atr_percent,
                    "notional_value": notional_value
                }),
                is_real_trade, quantity, quantity
            ))
            
            signal_id = cur.fetchone()['id']
            conn.commit()
            
            # تحديث الكاش
            with signal_cache_lock:
                cur.execute("SELECT * FROM signals WHERE id = %s", (signal_id,))
                signal_data = cur.fetchone()
                open_signals_cache[symbol] = dict(signal_data)
            
            # إرسال إشعار
            send_trade_open_notification(
                symbol, strategy_name, entry_price, stop_loss,
                target_1, target_2, quantity, is_real_trade,
                quality_score, atr_percent, notional_value
            )
            
            logger.info(f"✅ [Trade] Opened {'real' if is_real_trade else 'paper'} trade for {symbol} with {strategy_name}")
            return True
    
    except Exception as e:
        logger.error(f"❌ [Trade] Error executing trade for {symbol}: {e}")
        if conn: conn.rollback()
        return False

def check_and_execute_signals():
    """
    التحقق من جميع الرموز وتنفيذ الصفقات عند وجود إشارات
    """
    global is_trading_enabled
    with trading_status_lock:
        if not is_trading_enabled:
            return
    
    logger.info("[Signal] Starting signal check...")
    
    # جلب الرموز الصالحة
    if not validated_symbols_to_scan:
        global validated_symbols_to_scan
        validated_symbols_to_scan = get_validated_symbols()
    
    if not validated_symbols_to_scan:
        logger.error("[Signal] No valid symbols to scan")
        return
    
    # تحديث حالة السوق
    update_market_state()
    
    # معالجة كل رمز
    processed_count = 0
    signals_found = 0
    
    for symbol in validated_symbols_to_scan:
        try:
            # جلب البيانات
            df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
            if df is None or len(df) < 100:
                continue
            
            # حساب المؤشرات
            df = calculate_all_features(df)
            if df is None or len(df) < 50:
                continue
            
            processed_count += 1
            
            # التحقق من الاستراتيجيات المفعلة
            strategies_to_check = []
            
            if USE_BB_STOCH_STRATEGY:
                strategies_to_check.append(check_bb_stoch_strategy)
            if USE_MACD_EMA_STRATEGY:
                strategies_to_check.append(check_macd_ema_strategy)
            if USE_EMA_RSI_STRATEGY:
                strategies_to_check.append(check_ema_rsi_strategy)
            if USE_PULLBACK_STRATEGY:
                strategies_to_check.append(check_pullback_strategy)
            if USE_MOMENTUM_VOLATILITY_STRATEGY:
                strategies_to_check.append(check_momentum_volatility_strategy)
            if USE_ELLIOTT_WAVE_STRATEGY:
                strategies_to_check.append(check_elliott_wave_strategy)
            if USE_RANGE_REVERSAL_STRATEGY:
                strategies_to_check.append(check_range_reversal_strategy)
            
            # التحقق من الإشارات
            for strategy_func in strategies_to_check:
                try:
                    signal_result = strategy_func(symbol, df)
                    
                    if signal_result.get('valid', False):
                        # التحقق من الفلاتر العامة
                        general_ok, general_reason = check_general_filters(
                            symbol, df, signal_result.get('quality_score', 0)
                        )
                        
                        if not general_ok:
                            continue
                        
                        # تنفيذ الصفقة
                        success = execute_trade(
                            symbol,
                            signal_result.get('strategy', ''),
                            signal_result.get('entry_price', 0),
                            signal_result.get('stop_loss', 0),
                            signal_result.get('target_1', 0),
                            signal_result.get('target_2', 0),
                            signal_result.get('quality_score', 0),
                            signal_result.get('atr_percent', 0)
                        )
                        
                        if success:
                            signals_found += 1
                            break  # الخروج من حلقة الاستراتيجيات بعد تنفيذ صفقة واحدة للرمز
                
                except Exception as e:
                    logger.error(f"❌ [Signal] Error checking strategy for {symbol}: {e}")
            
            # تأخير صغير بين معالجة الرموز
            time.sleep(0.1)
        
        except Exception as e:
            logger.error(f"❌ [Signal] Error processing {symbol}: {e}")
    
    logger.info(f"[Signal] Processed {processed_count}/{len(validated_symbols_to_scan)} symbols, found {signals_found} signals")

def monitor_open_trades():
    """
    مراقبة الصفقات المفتوحة وتحديثها
    """
    global is_trading_enabled
    with trading_status_lock:
        if not is_trading_enabled:
            return
    
    if not check_db_connection() or not conn:
        return
    
    try:
        with conn.cursor() as cur:
            # جلب الصفقات المفتوحة
            cur.execute("SELECT * FROM signals WHERE status IN ('open', 'updated')")
            open_trades = cur.fetchall()
            
            if not open_trades:
                return
            
            # جلب الأسعار الحالية
            with live_prices_lock:
                prices = dict(live_prices)
            
            # معالجة كل صفقة
            for trade in open_trades:
                symbol = trade['symbol']
                current_price = prices.get(symbol)
                
                if current_price is None:
                    continue
                
                # حساب الربح/الخسارة
                entry_price = trade['entry_price']
                profit_percent = ((current_price - entry_price) / entry_price) * 100
                
                # تحقق من شروط إغلاق الصفقة
                close_trade = False
                close_reason = ""
                closing_price = current_price
                
                # وقف الخسارة
                if current_price <= trade['stop_loss']:
                    close_trade = True
                    close_reason = "stop_loss"
                    apply_symbol_cooldown(symbol)
                    update_consecutive_losses(symbol, False)
                
                # الهدف الأول
                elif trade.get('target_price_1') and current_price >= trade['target_price_1']:
                    # تحريك وقف الخسارة إلى نقطة التعادل
                    new_stop_loss = entry_price
                    if new_stop_loss > trade['stop_loss']:
                        # تحديث وقف الخسارة
                        cur.execute("""
                            UPDATE signals 
                            SET stop_loss = %s, status = 'updated' 
                            WHERE id = %s
                        """, (new_stop_loss, trade['id']))
                        
                        # تحديث الكاش
                        with signal_cache_lock:
                            if symbol in open_signals_cache:
                                open_signals_cache[symbol]['stop_loss'] = new_stop_loss
                                open_signals_cache[symbol]['status'] = 'updated'
                        
                        # إرسال إشعار
                        send_trade_update_notification(
                            symbol, "stop_loss", trade['stop_loss'], new_stop_loss,
                            current_price, profit_percent, trade['is_real_trade']
                        )
                
                # الهدف الثاني
                elif trade.get('target_price_2') and current_price >= trade['target_price_2']:
                    close_trade = True
                    close_reason = "target_2"
                    update_consecutive_losses(symbol, True)
                
                # وقف خسارة متحرك
                elif profit_percent >= TRAILING_STOP_ACTIVATION_PROFIT_PERCENT:
                    # حساب وقف الخسارة المتحرك
                    atr_percent = trade.get('signal_details', {}).get('atr_percent', 1.0)
                    trailing_stop = current_price * (1 - (atr_percent / 100))
                    
                    if trailing_stop > trade['stop_loss']:
                        # تحديث وقف الخسارة
                        cur.execute("""
                            UPDATE signals 
                            SET stop_loss = %s, status = 'updated' 
                            WHERE id = %s
                        """, (trailing_stop, trade['id']))
                        
                        # تحديث الكاش
                        with signal_cache_lock:
                            if symbol in open_signals_cache:
                                open_signals_cache[symbol]['stop_loss'] = trailing_stop
                                open_signals_cache[symbol]['status'] = 'updated'
                        
                        # إرسال إشعار
                        send_trade_update_notification(
                            symbol, "stop_loss", trade['stop_loss'], trailing_stop,
                            current_price, profit_percent, trade['is_real_trade']
                        )
                
                # إغلاق الصفقة إذا لزم الأمر
                if close_trade:
                    cur.execute("""
                        UPDATE signals 
                        SET status = 'closed', closing_price = %s, closed_at = NOW(), 
                            profit_percentage = %s, closing_reason = %s
                        WHERE id = %s
                    """, (closing_price, profit_percent, close_reason, trade['id']))
                    
                    # تحديث الكاش
                    with signal_cache_lock:
                        if symbol in open_signals_cache:
                            del open_signals_cache[symbol]
                    
                    # إرسال إشعار
                    if close_reason == "stop_loss":
                        log_and_notify("warning", f"⚠️ صفقة {symbol} أغلقت بوقف خسارة. الخسارة: {profit_percent:.2f}%", "trade_closed")
                    elif close_reason.startswith("target"):
                        log_and_notify("info", f"✅ صفقة {symbol} أغلقت بتحقيق الهدف. الربح: {profit_percent:.2f}%", "trade_closed")
            
            conn.commit()
    
    except Exception as e:
        logger.error(f"❌ [Monitor] Error monitoring open trades: {e}")
        if conn: conn.rollback()

def signal_generation_loop():
    """
    حلقة توليد الإشارات الرئيسية
    """
    while True:
        try:
            # التحقق من حالة التداول
            global is_trading_enabled
            with trading_status_lock:
                if not is_trading_enabled:
                    time.sleep(30)
                    continue
            
            # تنفيذ المهام المجدولة
            now = datetime.now(timezone.utc)
            
            # تحديث الأسعار كل دقيقة
            if now.second < 10:
                monitor_open_trades()
            
            # البحث عن إشارات جديدة كل 5 دقائق
            if now.minute % 5 == 0 and now.second < 10:
                check_and_execute_signals()
            
            # النوم لفترة قصيرة
            time.sleep(10)
        
        except Exception as e:
            logger.error(f"❌ [Main Loop] Error in signal generation loop: {e}")
            time.sleep(30)

# --- دوال واجهة المستخدم ---
@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم البوت v35.0.1</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
        }
        .sidebar {
            min-height: 100vh;
            background-color: #212529;
            color: white;
        }
        .sidebar .nav-link {
            color: #adb5bd;
            margin-bottom: 5px;
        }
        .sidebar .nav-link:hover {
            color: white;
            background-color: #343a40;
        }
        .sidebar .nav-link.active {
            color: white;
            background-color: #0d6efd;
        }
        .card {
            border-radius: 10px;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
            margin-bottom: 20px;
        }
        .card-header {
            background-color: #0d6efd;
            color: white;
            border-radius: 10px 10px 0 0 !important;
            font-weight: bold;
        }
        .badge {
            font-size: 0.85em;
        }
        .table th {
            background-color: #f1f5f9;
            border-top: none;
        }
        .status-online {
            color: #198754;
        }
        .status-offline {
            color: #dc3545;
        }
        .signal-quality {
            height: 10px;
            border-radius: 5px;
        }
        .trend-up {
            color: #198754;
        }
        .trend-down {
            color: #dc3545;
        }
        .trend-neutral {
            color: #6c757d;
        }
        .market-regime-trending {
            color: #198754;
        }
        .market-regime-ranging {
            color: #ffc107;
        }
        .market-regime-volatile {
            color: #dc3545;
        }
        .volatility-low {
            color: #198754;
        }
        .volatility-medium {
            color: #ffc107;
        }
        .volatility-high {
            color: #dc3545;
        }
        .notification-item {
            border-bottom: 1px solid #eee;
            padding: 10px 0;
        }
        .notification-item:last-child {
            border-bottom: none;
        }
        .rejection-item {
            border-bottom: 1px solid #eee;
            padding: 8px 0;
            font-size: 0.9rem;
        }
        .rejection-item:last-child {
            border-bottom: none;
        }
        .strategy-card {
            cursor: pointer;
            transition: transform 0.2s;
        }
        .strategy-card:hover {
            transform: translateY(-5px);
        }
        .strategy-enabled {
            border-left: 5px solid #198754;
        }
        .strategy-disabled {
            border-left: 5px solid #dc3545;
            opacity: 0.7;
        }
        .trade-card {
            border-left: 5px solid #0d6efd;
        }
        .trade-profit {
            border-left-color: #198754;
        }
        .trade-loss {
            border-left-color: #dc3545;
        }
        .control-panel {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 50px;
            height: 24px;
        }
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 24px;
        }
        .slider:before {
            position: absolute;
            content: "";
            height: 16px;
            width: 16px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }
        input:checked + .slider {
            background-color: #0d6efd;
        }
        input:checked + .slider:before {
            transform: translateX(26px);
        }
        .loading-spinner {
            display: inline-block;
            width: 1rem;
            height: 1rem;
            border: 2px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Sidebar -->
            <div class="col-md-2 sidebar p-3">
                <h4 class="mb-4">البوت v35.0.1</h4>
                <ul class="nav flex-column">
                    <li class="nav-item">
                        <a class="nav-link active" href="#" data-tab="dashboard">
                            <i class="fas fa-tachometer-alt me-2"></i> لوحة التحكم
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#" data-tab="trades">
                            <i class="fas fa-exchange-alt me-2"></i> الصفقات
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#" data-tab="strategies">
                            <i class="fas fa-chess me-2"></i> الاستراتيجيات
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#" data-tab="settings">
                            <i class="fas fa-cog me-2"></i> الإعدادات
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#" data-tab="notifications">
                            <i class="fas fa-bell me-2"></i> الإشعارات
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#" data-tab="rejections">
                            <i class="fas fa-times-circle me-2"></i> الرفوض
                        </a>
                    </li>
                </ul>
                
                <div class="mt-4">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <span>حالة البوت</span>
                        <span id="bot-status" class="badge bg-danger">متوقف</span>
                    </div>
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <span>نوع التداول</span>
                        <span id="trading-mode" class="badge bg-warning">ورقي</span>
                    </div>
                    <div class="d-flex justify-content-between align-items-center">
                        <span>الصفقات المفتوحة</span>
                        <span id="open-trades-count" class="badge bg-info">0</span>
                    </div>
                </div>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 p-4">
                <!-- Dashboard Tab -->
                <div id="dashboard-tab" class="tab-content">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2>لوحة التحكم</h2>
                        <div class="d-flex align-items-center">
                            <span class="me-2">آخر تحديث: <span id="last-update">--:--:--</span></span>
                            <button id="refresh-btn" class="btn btn-outline-primary btn-sm">
                                <i class="fas fa-sync-alt"></i>
                            </button>
                        </div>
                    </div>
                    
                    <!-- Control Panel -->
                    <div class="control-panel">
                        <div class="row align-items-center">
                            <div class="col-md-4">
                                <div class="d-flex align-items-center">
                                    <label class="form-label me-3 mb-0">تفعيل التداول:</label>
                                    <label class="toggle-switch">
                                        <input type="checkbox" id="trading-toggle">
                                        <span class="slider"></span>
                                    </label>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="d-flex align-items-center">
                                    <label class="form-label me-3 mb-0">نوع التداول:</label>
                                    <div class="btn-group" role="group">
                                        <input type="radio" class="btn-check" name="trading-mode" id="paper-mode" value="paper" checked>
                                        <label class="btn btn-outline-warning" for="paper-mode">ورقي</label>
                                        
                                        <input type="radio" class="btn-check" name="trading-mode" id="real-mode" value="real">
                                        <label class="btn btn-outline-danger" for="real-mode">حقيقي</label>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <button id="scan-now-btn" class="btn btn-primary">
                                    <i class="fas fa-search me-2"></i>فحص الآن
                                </button>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Market Status -->
                    <div class="row mb-4">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    حالة السوق
                                </div>
                                <div class="card-body">
                                    <div class="row">
                                        <div class="col-md-6">
                                            <h5>نظام السوق</h5>
                                            <p class="fs-3">
                                                <span id="market-regime" class="badge bg-secondary">غير معروف</span>
                                            </p>
                                        </div>
                                        <div class="col-md-6">
                                            <h5>مستوى التقلب</h5>
                                            <p class="fs-3">
                                                <span id="volatility-state" class="badge bg-secondary">متوسط</span>
                                            </p>
                                        </div>
                                    </div>
                                    <hr>
                                    <h5>اتجاه السوق حسب الفريم</h5>
                                    <div class="row">
                                        <div class="col-md-4">
                                            <div class="text-center">
                                                <h6>5 دقائق</h6>
                                                <p id="trend-5m" class="fs-4 trend-neutral">
                                                    <i class="fas fa-minus"></i>
                                                </p>
                                                <small>ADX: <span id="adx-5m">--</span></small>
                                            </div>
                                        </div>
                                        <div class="col-md-4">
                                            <div class="text-center">
                                                <h6>15 دقيقة</h6>
                                                <p id="trend-15m" class="fs-4 trend-neutral">
                                                    <i class="fas fa-minus"></i>
                                                </p>
                                                <small>ADX: <span id="adx-15m">--</span></small>
                                            </div>
                                        </div>
                                        <div class="col-md-4">
                                            <div class="text-center">
                                                <h6>ساعة</h6>
                                                <p id="trend-1h" class="fs-4 trend-neutral">
                                                    <i class="fas fa-minus"></i>
                                                </p>
                                                <small>ADX: <span id="adx-1h">--</span></small>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    أداء البوت
                                </div>
                                <div class="card-body">
                                    <div class="row">
                                        <div class="col-md-6">
                                            <div class="text-center mb-3">
                                                <h5>إجمالي الصفقات</h5>
                                                <p class="fs-3" id="total-trades">0</p>
                                            </div>
                                        </div>
                                        <div class="col-md-6">
                                            <div class="text-center mb-3">
                                                <h5>نسبة النجاح</h5>
                                                <p class="fs-3" id="win-rate">0%</p>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="row">
                                        <div class="col-md-6">
                                            <div class="text-center mb-3">
                                                <h5>متوسط الربح</h5>
                                                <p class="fs-3 text-success" id="avg-profit">0%</p>
                                            </div>
                                        </div>
                                        <div class="col-md-6">
                                            <div class="text-center mb-3">
                                                <h5>إجمالي الربح</h5>
                                                <p class="fs-3 text-success" id="total-profit">0%</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Open Trades -->
                    <div class="card">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            الصفقات المفتوحة
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
                                            <th>السعر الحالي</th>
                                            <th>وقف الخسارة</th>
                                            <th>الهدف 1</th>
                                            <th>الهدف 2</th>
                                            <th>الربح/الخسارة</th>
                                            <th>النوع</th>
                                        </tr>
                                    </thead>
                                    <tbody id="open-trades-table">
                                        <tr>
                                            <td colspan="9" class="text-center">لا توجد صفقات مفتوحة</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Trades Tab -->
                <div id="trades-tab" class="tab-content" style="display: none;">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2>الصفقات</h2>
                        <div>
                            <button class="btn btn-outline-primary btn-sm me-2" id="export-trades-btn">
                                <i class="fas fa-download me-1"></i> تصدير
                            </button>
                            <button class="btn btn-outline-danger btn-sm" id="clear-trades-btn">
                                <i class="fas fa-trash me-1"></i> مسح
                            </button>
                        </div>
                    </div>
                    
                    <div class="card mb-4">
                        <div class="card-header">
                            فلتر الصفقات
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-3">
                                    <div class="mb-3">
                                        <label class="form-label">الحالة</label>
                                        <select class="form-select" id="trade-status-filter">
                                            <option value="all">الكل</option>
                                            <option value="open">مفتوحة</option>
                                            <option value="closed">مغلقة</option>
                                        </select>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="mb-3">
                                        <label class="form-label">النوع</label>
                                        <select class="form-select" id="trade-type-filter">
                                            <option value="all">الكل</option>
                                            <option value="paper">ورقية</option>
                                            <option value="real">حقيقية</option>
                                        </select>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="mb-3">
                                        <label class="form-label">الاستراتيجية</label>
                                        <select class="form-select" id="trade-strategy-filter">
                                            <option value="all">الكل</option>
                                            <option value="BB_Stoch_Strategy">BB+Stoch</option>
                                            <option value="MACD_EMA_Strategy">MACD+SMA</option>
                                            <option value="EMA_RSI_Strategy">EMA+RSI</option>
                                            <option value="Pullback_Strategy">Pullback</option>
                                            <option value="Momentum_Volatility_Strategy">Momentum</option>
                                            <option value="Elliott_Wave_Strategy">Elliott Wave</option>
                                            <option value="Range_Reversal_Strategy">Range Reversal</option>
                                        </select>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="mb-3">
                                        <label class="form-label">العملة</label>
                                        <input type="text" class="form-control" id="trade-symbol-filter" placeholder="ابحث عن عملة">
                                    </div>
                                </div>
                            </div>
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label class="form-label">من تاريخ</label>
                                        <input type="date" class="form-control" id="trade-from-date">
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label class="form-label">إلى تاريخ</label>
                                        <input type="date" class="form-control" id="trade-to-date">
                                    </div>
                                </div>
                            </div>
                            <div class="text-end">
                                <button class="btn btn-primary" id="apply-trade-filters-btn">
                                    <i class="fas fa-filter me-1"></i> تطبيق الفلتر
                                </button>
                                <button class="btn btn-outline-secondary" id="reset-trade-filters-btn">
                                    <i class="fas fa-redo me-1"></i> إعادة تعيين
                                </button>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            سجل الصفقات
                        </div>
                        <div class="card-body">
                            <div class="table-responsive">
                                <table class="table table-hover">
                                    <thead>
                                        <tr>
                                            <th>العملة</th>
                                            <th>الاستراتيجية</th>
                                            <th>سعر الدخول</th>
                                            <th>سعر الخروج</th>
                                            <th>الربح/الخسارة</th>
                                            <th>الحالة</th>
                                            <th>النوع</th>
                                            <th>التاريخ</th>
                                        </tr>
                                    </thead>
                                    <tbody id="all-trades-table">
                                        <tr>
                                            <td colspan="8" class="text-center">لا توجد صفقات</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                            <nav aria-label="Page navigation">
                                <ul class="pagination justify-content-center" id="trades-pagination">
                                    <li class="page-item disabled">
                                        <a class="page-link" href="#" tabindex="-1">السابق</a>
                                    </li>
                                    <li class="page-item active"><a class="page-link" href="#">1</a></li>
                                    <li class="page-item"><a class="page-link" href="#">2</a></li>
                                    <li class="page-item"><a class="page-link" href="#">3</a></li>
                                    <li class="page-item">
                                        <a class="page-link" href="#">التالي</a>
                                    </li>
                                </ul>
                            </nav>
                        </div>
                    </div>
                </div>
                
                <!-- Strategies Tab -->
                <div id="strategies-tab" class="tab-content" style="display: none;">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2>الاستراتيجيات</h2>
                        <div>
                            <button class="btn btn-outline-primary btn-sm me-2" id="save-strategies-btn">
                                <i class="fas fa-save me-1"></i> حفظ
                            </button>
                            <button class="btn btn-outline-secondary btn-sm" id="reset-strategies-btn">
                                <i class="fas fa-redo me-1"></i> إعادة تعيين
                            </button>
                        </div>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-4 mb-4">
                            <div class="card strategy-card strategy-enabled h-100" data-strategy="BB_Stoch_Strategy">
                                <div class="card-header">
                                    BB+Stoch (ارتداد مبكر)
                                </div>
                                <div class="card-body">
                                    <h5>بولينجر + ستوكاستيك</h5>
                                    <p class="card-text">تستخدم هذه الاستراتيجية مؤشر بولينجر لتحديد مستويات الدعم والمقاومة، مع مؤشر ستوكاستيك لتحديد نقاط الدخول عند تشبع البيع.</p>
                                    <div class="mb-3">
                                        <div class="d-flex justify-content-between">
                                            <span>الحالة</span>
                                            <div class="form-check form-switch">
                                                <input class="form-check-input strategy-toggle" type="checkbox" checked>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="progress mb-3" style="height: 10px;">
                                        <div class="progress-bar bg-success" role="progressbar" style="width: 75%"></div>
                                    </div>
                                    <div class="d-flex justify-content-between">
                                        <small>معدل النجاح</small>
                                        <small>75%</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4 mb-4">
                            <div class="card strategy-card strategy-enabled h-100" data-strategy="MACD_EMA_Strategy">
                                <div class="card-header">
                                    MACD+SMA (زخم وتقاطع)
                                </div>
                                <div class="card-body">
                                    <h5>ماكد + متوسطات متحركة</h5>
                                    <p class="card-text">تعتمد هذه الاستراتيجية على تقاطع خط الماكد مع إشارته، مع تأكيد من المتوسطات المتحركة لضمان اتجاه صاعد.</p>
                                    <div class="mb-3">
                                        <div class="d-flex justify-content-between">
                                            <span>الحالة</span>
                                            <div class="form-check form-switch">
                                                <input class="form-check-input strategy-toggle" type="checkbox" checked>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="progress mb-3" style="height: 10px;">
                                        <div class="progress-bar bg-success" role="progressbar" style="width: 68%"></div>
                                    </div>
                                    <div class="d-flex justify-content-between">
                                        <small>معدل النجاح</small>
                                        <small>68%</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4 mb-4">
                            <div class="card strategy-card strategy-enabled h-100" data-strategy="EMA_RSI_Strategy">
                                <div class="card-header">
                                    EMA+RSI (ارتداد سريع)
                                </div>
                                <div class="card-body">
                                    <h5>متوسطات متحركة + RSI</h5>
                                    <p class="card-text">تستخدم هذه الاستراتيجية مؤشر RSI لتحديد مناطق التشبع، مع تأكيد من المتوسطات المتحركة لضمان الاتجاه العام.</p>
                                    <div class="mb-3">
                                        <div class="d-flex justify-content-between">
                                            <span>الحالة</span>
                                            <div class="form-check form-switch">
                                                <input class="form-check-input strategy-toggle" type="checkbox" checked>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="progress mb-3" style="height: 10px;">
                                        <div class="progress-bar bg-success" role="progressbar" style="width: 72%"></div>
                                    </div>
                                    <div class="d-flex justify-content-between">
                                        <small>معدل النجاح</small>
                                        <small>72%</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4 mb-4">
                            <div class="card strategy-card strategy-enabled h-100" data-strategy="Pullback_Strategy">
                                <div class="card-header">
                                    Pullback (ارتداد بحجم تداول)
                                </div>
                                <div class="card-body">
                                    <h5>استراتيجية الارتداد</h5>
                                    <p class="card-text">تستهدف هذه الاستراتيجية الدخول بعد ارتداد السعر من قمة حديثة، مع التركيز على حجم التداول لتأكيد التعافي.</p>
                                    <div class="mb-3">
                                        <div class="d-flex justify-content-between">
                                            <span>الحالة</span>
                                            <div class="form-check form-switch">
                                                <input class="form-check-input strategy-toggle" type="checkbox" checked>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="progress mb-3" style="height: 10px;">
                                        <div class="progress-bar bg-success" role="progressbar" style="width: 70%"></div>
                                    </div>
                                    <div class="d-flex justify-content-between">
                                        <small>معدل النجاح</small>
                                        <small>70%</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4 mb-4">
                            <div class="card strategy-card strategy-enabled h-100" data-strategy="Momentum_Volatility_Strategy">
                                <div class="card-header">
                                    Momentum (زخم متزايد)
                                </div>
                                <div class="card-body">
                                    <h5>استراتيجية الزخم</h5>
                                    <p class="card-text">تعتمد هذه الاستراتيجية على تحديد الزخم الإيجابي في السعر، مع التأكد من وجود تقلب كافٍ لتحقيق أرباح.</p>
                                    <div class="mb-3">
                                        <div class="d-flex justify-content-between">
                                            <span>الحالة</span>
                                            <div class="form-check form-switch">
                                                <input class="form-check-input strategy-toggle" type="checkbox" checked>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="progress mb-3" style="height: 10px;">
                                        <div class="progress-bar bg-success" role="progressbar" style="width: 65%"></div>
                                    </div>
                                    <div class="d-flex justify-content-between">
                                        <small>معدل النجاح</small>
                                        <small>65%</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4 mb-4">
                            <div class="card strategy-card strategy-enabled h-100" data-strategy="Elliott_Wave_Strategy">
                                <div class="card-header">
                                    Elliott Wave (موجات إليوت)
                                </div>
                                <div class="card-body">
                                    <h5>نظرية موجات إليوت</h5>
                                    <p class="card-text">تستند هذه الاستراتيجية إلى نظرية موجات إليوت لتحديد نقاط الانعكاس المحتملة بناءً على تصحيحات فيبوناتشي.</p>
                                    <div class="mb-3">
                                        <div class="d-flex justify-content-between">
                                            <span>الحالة</span>
                                            <div class="form-check form-switch">
                                                <input class="form-check-input strategy-toggle" type="checkbox" checked>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="progress mb-3" style="height: 10px;">
                                        <div class="progress-bar bg-success" role="progressbar" style="width: 63%"></div>
                                    </div>
                                    <div class="d-flex justify-content-between">
                                        <small>معدل النجاح</small>
                                        <small>63%</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4 mb-4">
                            <div class="card strategy-card strategy-enabled h-100" data-strategy="Range_Reversal_Strategy">
                                <div class="card-header">
                                    Range Reversal (انعكاس نطاقي)
                                </div>
                                <div class="card-body">
                                    <h5>استراتيجية الانعكاس النطاقي</h5>
                                    <p class="card-text">تعمل هذه الاستراتيجية في الأسواق الجانبية، وتستهدف الانعكاسات من حدود النطاق مع تأكيد من مؤشر RSI.</p>
                                    <div class="mb-3">
                                        <div class="d-flex justify-content-between">
                                            <span>الحالة</span>
                                            <div class="form-check form-switch">
                                                <input class="form-check-input strategy-toggle" type="checkbox" checked>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="progress mb-3" style="height: 10px;">
                                        <div class="progress-bar bg-success" role="progressbar" style="width: 67%"></div>
                                    </div>
                                    <div class="d-flex justify-content-between">
                                        <small>معدل النجاح</small>
                                        <small>67%</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Settings Tab -->
                <div id="settings-tab" class="tab-content" style="display: none;">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2>الإعدادات</h2>
                        <div>
                            <button class="btn btn-primary" id="save-settings-btn">
                                <i class="fas fa-save me-1"></i> حفظ الإعدادات
                            </button>
                        </div>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    إعدادات التداول
                                </div>
                                <div class="card-body">
                                    <div class="mb-3">
                                        <label class="form-label">الحد الأدنى لحجم الصفقة (USDT)</label>
                                        <input type="number" class="form-control" id="min-trade-amount" value="4.5" step="0.1" min="1">
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label">الحد الأقصى لحجم الصفقة (USDT)</label>
                                        <input type="number" class="form-control" id="max-trade-amount" value="6.5" step="0.1" min="1">
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label">الحد الأقصى لعدد الصفقات المفتوحة</label>
                                        <input type="number" class="form-control" id="max-open-trades" value="3" step="1" min="1" max="10">
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label">حجم الصفقة الورقية (USDT)</label>
                                        <input type="number" class="form-control" id="paper-trade-amount" value="10.0" step="0.1" min="1">
                                    </div>
                                    <div class="mb-3">
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" id="auto-fallback" checked>
                                            <label class="form-check-label" for="auto-fallback">
                                                التبديل التلقائي للوضع الورقي عند انخفاض الرصيد
                                            </label>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    إعدادات الإشارات
                                </div>
                                <div class="card-body">
                                    <div class="mb-3">
                                        <label class="form-label">الحد الأدنى لجودة الإشارة (%)</label>
                                        <input type="range" class="form-range" id="signal-quality" min="50" max="100" value="70">
                                        <div class="d-flex justify-content-between">
                                            <small>50%</small>
                                            <small id="signal-quality-value">70%</small>
                                            <small>100%</small>
                                        </div>
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label">نسبة تفعيل وقف الخسارة المتحرك (%)</label>
                                        <input type="number" class="form-control" id="trailing-stop" value="1.0" step="0.1" min="0.5" max="5.0">
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label">مدة التبريد بعد وقف الخسارة (دقائق)</label>
                                        <input type="number" class="form-control" id="cooldown-minutes" value="30" step="5" min="5" max="120">
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label">الإطار الزمني للإشارات</label>
                                        <select class="form-select" id="signal-timeframe">
                                            <option value="5m" selected>5 دقائق</option>
                                            <option value="15m">15 دقيقة</option>
                                            <option value="1h">ساعة</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="row mt-4">
                        <div class="col-md-12">
                            <div class="card">
                                <div class="card-header">
                                    إعدادات الإشعارات
                                </div>
                                <div class="card-body">
                                    <div class="row">
                                        <div class="col-md-4">
                                            <div class="mb-3">
                                                <div class="form-check form-switch">
                                                    <input class="form-check-input" type="checkbox" id="telegram-notifications" checked>
                                                    <label class="form-check-label" for="telegram-notifications">
                                                        إشعارات تيليجرام
                                                    </label>
                                                </div>
                                            </div>
                                            <div class="mb-3">
                                                <div class="form-check form-switch">
                                                    <input class="form-check-input" type="checkbox" id="email-notifications">
                                                    <label class="form-check-label" for="email-notifications">
                                                        إشعارات البريد الإلكتروني
                                                    </label>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="col-md-4">
                                            <div class="mb-3">
                                                <label class="form-label">الحد الأدنى للربح للإشعار (%)</label>
                                                <input type="number" class="form-control" id="min-profit-notification" value="1.0" step="0.1">
                                            </div>
                                        </div>
                                        <div class="col-md-4">
                                            <div class="mb-3">
                                                <label class="form-label">الحد الأقصى للخسارة للإشعار (%)</label>
                                                <input type="number" class="form-control" id="max-loss-notification" value="-1.0" step="0.1">
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Notifications Tab -->
                <div id="notifications-tab" class="tab-content" style="display: none;">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2>الإشعارات</h2>
                        <div>
                            <button class="btn btn-outline-danger btn-sm" id="clear-notifications-btn">
                                <i class="fas fa-trash me-1"></i> مسح الكل
                            </button>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            سجل الإشعارات
                        </div>
                        <div class="card-body">
                            <div id="notifications-container">
                                <div class="notification-item">
                                    <div class="d-flex justify-content-between">
                                        <div>
                                            <span class="badge bg-primary me-2">معلومات</span>
                                            <span>تم بدء تشغيل البوت بنجاح</span>
                                        </div>
                                        <small class="text-muted">2023-06-15 10:30:45</small>
                                    </div>
                                </div>
                                <div class="notification-item">
                                    <div class="d-flex justify-content-between">
                                        <div>
                                            <span class="badge bg-success me-2">نجاح</span>
                                            <span>تم فتح صفقة جديدة لعملة BTCUSDT</span>
                                        </div>
                                        <small class="text-muted">2023-06-15 11:45:22</small>
                                    </div>
                                </div>
                                <div class="notification-item">
                                    <div class="d-flex justify-content-between">
                                        <div>
                                            <span class="badge bg-warning me-2">تحذير</span>
                                            <span>صفقة ETHUSDT أغلقت بوقف خسارة</span>
                                        </div>
                                        <small class="text-muted">2023-06-15 14:20:10</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Rejections Tab -->
                <div id="rejections-tab" class="tab-content" style="display: none;">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <h2>سجل الرفوض</h2>
                        <div>
                            <button class="btn btn-outline-danger btn-sm" id="clear-rejections-btn">
                                <i class="fas fa-trash me-1"></i> مسح الكل
                            </button>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            أسباب رفض الإشارات
                        </div>
                        <div class="card-body">
                            <div id="rejections-container">
                                <div class="rejection-item">
                                    <div class="d-flex justify-content-between">
                                        <div>
                                            <span class="me-2">BTCUSDT</span>
                                            <span class="text-muted">فلتر تقلب السوق رفض الدخول</span>
                                        </div>
                                        <small class="text-muted">2023-06-15 10:35:12</small>
                                    </div>
                                </div>
                                <div class="rejection-item">
                                    <div class="d-flex justify-content-between">
                                        <div>
                                            <span class="me-2">ETHUSDT</span>
                                            <span class="text-muted">جودة الإشارة منخفضة</span>
                                        </div>
                                        <small class="text-muted">2023-06-15 11:20:45</small>
                                    </div>
                                </div>
                                <div class="rejection-item">
                                    <div class="d-flex justify-content-between">
                                        <div>
                                            <span class="me-2">ADAUSDT</span>
                                            <span class="text-muted">ديناميكي: عرض البولينجر ضيق جدًا</span>
                                        </div>
                                        <small class="text-muted">2023-06-15 12:15:33</small>
                                    </div>
                                </div>
                                <div class="rejection-item">
                                    <div class="d-flex justify-content-between">
                                        <div>
                                            <span class="me-2">SOLUSDT</span>
                                            <span class="text-muted">الرصيد غير كافي لتنفيذ الصفقة</span>
                                        </div>
                                        <small class="text-muted">2023-06-15 13:40:18</small>
                                    </div>
                                </div>
                                <div class="rejection-item">
                                    <div class="d-flex justify-content-between">
                                        <div>
                                            <span class="me-2">DOTUSDT</span>
                                            <span class="text-muted">ديناميكي: زخم الماكد لا يتزايد بقوة كافية</span>
                                        </div>
                                        <small class="text-muted">2023-06-15 14:05:27</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Toast for notifications -->
    <div class="position-fixed bottom-0 end-0 p-3" style="z-index: 11">
        <div id="liveToast" class="toast" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header">
                <strong class="me-auto">إشعار جديد</strong>
                <small>الآن</small>
                <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body" id="toast-message">
                تم فتح صفقة جديدة!
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Global variables
        let socket;
        let botStatus = false;
        let tradingMode = 'paper';
        let openTrades = [];
        let allTrades = [];
        let notifications = [];
        let rejections = [];
        let marketState = {};
        
        // Initialize the page
        document.addEventListener('DOMContentLoaded', function() {
            // Connect to WebSocket
            connectWebSocket();
            
            // Load initial data
            loadDashboardData();
            loadTradesData();
            loadNotificationsData();
            loadRejectionsData();
            loadStrategiesData();
            loadSettingsData();
            
            // Setup event listeners
            setupEventListeners();
            
            // Setup tab navigation
            setupTabNavigation();
            
            // Update time
            updateTime();
            setInterval(updateTime, 1000);
        });
        
        // Connect to WebSocket
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            socket = new WebSocket(wsUrl);
            
            socket.onopen = function(e) {
                console.log("WebSocket connection established");
                updateConnectionStatus(true);
            };
            
            socket.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            socket.onclose = function(event) {
                console.log("WebSocket connection closed");
                updateConnectionStatus(false);
                
                // Try to reconnect after 5 seconds
                setTimeout(function() {
                    connectWebSocket();
                }, 5000);
            };
            
            socket.onerror = function(error) {
                console.error("WebSocket error:", error);
                updateConnectionStatus(false);
            };
        }
        
        // Handle WebSocket messages
        function handleWebSocketMessage(data) {
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
                case 'trade_update':
                    updateTrade(data.payload);
                    break;
                case 'market_state_update':
                    updateMarketState(data.payload);
                    break;
                case 'bot_status':
                    updateBotStatus(data.payload);
                    break;
            }
        }
        
        // Update connection status
        function updateConnectionStatus(connected) {
            const statusElement = document.getElementById('connection-status');
            if (statusElement) {
                if (connected) {
                    statusElement.innerHTML = '<span class="status-online"><i class="fas fa-circle"></i> متصل</span>';
                } else {
                    statusElement.innerHTML = '<span class="status-offline"><i class="fas fa-circle"></i> غير متصل</span>';
                }
            }
        }
        
        // Setup event listeners
        function setupEventListeners() {
            // Trading toggle
            document.getElementById('trading-toggle').addEventListener('change', function() {
                toggleTrading(this.checked);
            });
            
            // Trading mode
            document.querySelectorAll('input[name="trading-mode"]').forEach(radio => {
                radio.addEventListener('change', function() {
                    if (this.checked) {
                        changeTradingMode(this.value);
                    }
                });
            });
            
            // Scan now button
            document.getElementById('scan-now-btn').addEventListener('click', function() {
                scanNow();
            });
            
            // Refresh button
            document.getElementById('refresh-btn').addEventListener('click', function() {
                loadDashboardData();
                loadTradesData();
                loadNotificationsData();
                loadRejectionsData();
            });
            
            // Strategy toggles
            document.querySelectorAll('.strategy-toggle').forEach(toggle => {
                toggle.addEventListener('change', function() {
                    const strategyCard = this.closest('.strategy-card');
                    const strategy = strategyCard.dataset.strategy;
                    
                    if (this.checked) {
                        strategyCard.classList.remove('strategy-disabled');
                        strategyCard.classList.add('strategy-enabled');
                    } else {
                        strategyCard.classList.remove('strategy-enabled');
                        strategyCard.classList.add('strategy-disabled');
                    }
                });
            });
            
            // Save strategies button
            document.getElementById('save-strategies-btn').addEventListener('click', function() {
                saveStrategies();
            });
            
            // Reset strategies button
            document.getElementById('reset-strategies-btn').addEventListener('click', function() {
                resetStrategies();
            });
            
            // Save settings button
            document.getElementById('save-settings-btn').addEventListener('click', function() {
                saveSettings();
            });
            
            // Signal quality range
            document.getElementById('signal-quality').addEventListener('input', function() {
                document.getElementById('signal-quality-value').textContent = this.value + '%';
            });
            
            // Clear notifications button
            document.getElementById('clear-notifications-btn').addEventListener('click', function() {
                clearNotifications();
            });
            
            // Clear rejections button
            document.getElementById('clear-rejections-btn').addEventListener('click', function() {
                clearRejections();
            });
            
            // Apply trade filters button
            document.getElementById('apply-trade-filters-btn').addEventListener('click', function() {
                applyTradeFilters();
            });
            
            // Reset trade filters button
            document.getElementById('reset-trade-filters-btn').addEventListener('click', function() {
                resetTradeFilters();
            });
            
            // Export trades button
            document.getElementById('export-trades-btn').addEventListener('click', function() {
                exportTrades();
            });
            
            // Clear trades button
            document.getElementById('clear-trades-btn').addEventListener('click', function() {
                clearTrades();
            });
        }
        
        // Setup tab navigation
        function setupTabNavigation() {
            document.querySelectorAll('.nav-link').forEach(link => {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    
                    // Remove active class from all tabs and content
                    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
                    
                    // Add active class to clicked tab
                    this.classList.add('active');
                    
                    // Show corresponding content
                    const tabId = this.dataset.tab + '-tab';
                    const tabContent = document.getElementById(tabId);
                    if (tabContent) {
                        tabContent.style.display = 'block';
                    }
                });
            });
        }
        
        // Update time
        function updateTime() {
            const now = new Date();
            const timeString = now.toLocaleTimeString('ar-SA');
            document.getElementById('last-update').textContent = timeString;
        }
        
        // Toggle trading
        function toggleTrading(enabled) {
            fetch('/api/toggle_trading', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ enabled: enabled })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    botStatus = enabled;
                    updateBotStatusUI();
                    showToast(enabled ? 'تم تفعيل التداول' : 'تم إيقاف التداول', 'success');
                } else {
                    showToast('فشل تغيير حالة التداول', 'danger');
                    // Reset toggle
                    document.getElementById('trading-toggle').checked = botStatus;
                }
            })
            .catch(error => {
                console.error('Error toggling trading:', error);
                showToast('خطأ في الاتصال بالخادم', 'danger');
                // Reset toggle
                document.getElementById('trading-toggle').checked = botStatus;
            });
        }
        
        // Change trading mode
        function changeTradingMode(mode) {
            fetch('/api/change_trading_mode', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ mode: mode })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    tradingMode = mode;
                    updateTradingModeUI();
                    showToast(mode === 'paper' ? 'تم التبديل للوضع الورقي' : 'تم التبديل للوضع الحقيقي', 'success');
                } else {
                    showToast('فشل تغيير وضع التداول', 'danger');
                    // Reset radio buttons
                    document.querySelector(`input[name="trading-mode"][value="${tradingMode}"]`).checked = true;
                }
            })
            .catch(error => {
                console.error('Error changing trading mode:', error);
                showToast('خطأ في الاتصال بالخادم', 'danger');
                // Reset radio buttons
                document.querySelector(`input[name="trading-mode"][value="${tradingMode}"]`).checked = true;
            });
        }
        
        // Scan now
        function scanNow() {
            fetch('/api/scan_now', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showToast('تم بدء الفحص', 'success');
                    // Show loading state
                    const scanBtn = document.getElementById('scan-now-btn');
                    const originalText = scanBtn.innerHTML;
                    scanBtn.innerHTML = '<span class="loading-spinner"></span> جاري الفحص...';
                    scanBtn.disabled = true;
                    
                    // Reset after 10 seconds
                    setTimeout(() => {
                        scanBtn.innerHTML = originalText;
                        scanBtn.disabled = false;
                    }, 10000);
                } else {
                    showToast('فشل بدء الفحص', 'danger');
                }
            })
            .catch(error => {
                console.error('Error scanning:', error);
                showToast('خطأ في الاتصال بالخادم', 'danger');
            });
        }
        
        // Save strategies
        function saveStrategies() {
            const strategies = {};
            
            document.querySelectorAll('.strategy-toggle').forEach(toggle => {
                const strategyCard = toggle.closest('.strategy-card');
                const strategy = strategyCard.dataset.strategy;
                strategies[strategy] = toggle.checked;
            });
            
            fetch('/api/save_strategies', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ strategies: strategies })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showToast('تم حفظ الاستراتيجيات بنجاح', 'success');
                } else {
                    showToast('فشل حفظ الاستراتيجيات', 'danger');
                }
            })
            .catch(error => {
                console.error('Error saving strategies:', error);
                showToast('خطأ في الاتصال بالخادم', 'danger');
            });
        }
        
        // Reset strategies
        function resetStrategies() {
            if (confirm('هل أنت متأكد من إعادة تعيين الاستراتيجيات؟')) {
                document.querySelectorAll('.strategy-toggle').forEach(toggle => {
                    toggle.checked = true;
                    const strategyCard = toggle.closest('.strategy-card');
                    strategyCard.classList.remove('strategy-disabled');
                    strategyCard.classList.add('strategy-enabled');
                });
                
                showToast('تم إعادة تعيين الاستراتيجيات', 'info');
            }
        }
        
        // Save settings
        function saveSettings() {
            const settings = {
                min_trade_amount: parseFloat(document.getElementById('min-trade-amount').value),
                max_trade_amount: parseFloat(document.getElementById('max-trade-amount').value),
                max_open_trades: parseInt(document.getElementById('max-open-trades').value),
                paper_trade_amount: parseFloat(document.getElementById('paper-trade-amount').value),
                auto_fallback: document.getElementById('auto-fallback').checked,
                signal_quality: parseInt(document.getElementById('signal-quality').value),
                trailing_stop: parseFloat(document.getElementById('trailing-stop').value),
                cooldown_minutes: parseInt(document.getElementById('cooldown-minutes').value),
                signal_timeframe: document.getElementById('signal-timeframe').value,
                telegram_notifications: document.getElementById('telegram-notifications').checked,
                email_notifications: document.getElementById('email-notifications').checked,
                min_profit_notification: parseFloat(document.getElementById('min-profit-notification').value),
                max_loss_notification: parseFloat(document.getElementById('max-loss-notification').value)
            };
            
            fetch('/api/save_settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(settings)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showToast('تم حفظ الإعدادات بنجاح', 'success');
                } else {
                    showToast('فشل حفظ الإعدادات', 'danger');
                }
            })
            .catch(error => {
                console.error('Error saving settings:', error);
                showToast('خطأ في الاتصال بالخادم', 'danger');
            });
        }
        
        // Clear notifications
        function clearNotifications() {
            if (confirm('هل أنت متأكد من مسح جميع الإشعارات؟')) {
                fetch('/api/clear_notifications', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        notifications = [];
                        renderNotifications();
                        showToast('تم مسح الإشعارات', 'success');
                    } else {
                        showToast('فشل مسح الإشعارات', 'danger');
                    }
                })
                .catch(error => {
                    console.error('Error clearing notifications:', error);
                    showToast('خطأ في الاتصال بالخادم', 'danger');
                });
            }
        }
        
        // Clear rejections
        function clearRejections() {
            if (confirm('هل أنت متأكد من مسح جميع الرفوض؟')) {
                fetch('/api/clear_rejections', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        rejections = [];
                        renderRejections();
                        showToast('تم مسح الرفوض', 'success');
                    } else {
                        showToast('فشل مسح الرفوض', 'danger');
                    }
                })
                .catch(error => {
                    console.error('Error clearing rejections:', error);
                    showToast('خطأ في الاتصال بالخادم', 'danger');
                });
            }
        }
        
        // Apply trade filters
        function applyTradeFilters() {
            const filters = {
                status: document.getElementById('trade-status-filter').value,
                type: document.getElementById('trade-type-filter').value,
                strategy: document.getElementById('trade-strategy-filter').value,
                symbol: document.getElementById('trade-symbol-filter').value,
                from_date: document.getElementById('trade-from-date').value,
                to_date: document.getElementById('trade-to-date').value
            };
            
            fetch('/api/get_filtered_trades', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(filters)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    allTrades = data.trades;
                    renderTrades();
                    showToast('تم تطبيق الفلتر', 'success');
                } else {
                    showToast('فشل تطبيق الفلتر', 'danger');
                }
            })
            .catch(error => {
                console.error('Error applying trade filters:', error);
                showToast('خطأ في الاتصال بالخادم', 'danger');
            });
        }
        
        // Reset trade filters
        function resetTradeFilters() {
            document.getElementById('trade-status-filter').value = 'all';
            document.getElementById('trade-type-filter').value = 'all';
            document.getElementById('trade-strategy-filter').value = 'all';
            document.getElementById('trade-symbol-filter').value = '';
            document.getElementById('trade-from-date').value = '';
            document.getElementById('trade-to-date').value = '';
            
            loadTradesData();
            showToast('تم إعادة تعيين الفلاتر', 'info');
        }
        
        // Export trades
        function exportTrades() {
            fetch('/api/export_trades')
                .then(response => response.blob())
                .then(blob => {
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.style.display = 'none';
                    a.href = url;
                    a.download = `trades_${new Date().toISOString().slice(0, 10)}.csv`;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    showToast('تم تصدير البيانات', 'success');
                })
                .catch(error => {
                    console.error('Error exporting trades:', error);
                    showToast('خطأ في تصدير البيانات', 'danger');
                });
        }
        
        // Clear trades
        function clearTrades() {
            if (confirm('هل أنت متأكد من مسح جميع الصفقات؟ هذا الإجراء لا يمكن التراجع عنه.')) {
                fetch('/api/clear_trades', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        allTrades = [];
                        openTrades = [];
                        renderTrades();
                        renderOpenTrades();
                        updatePerformanceStats();
                        showToast('تم مسح الصفقات', 'success');
                    } else {
                        showToast('فشل مسح الصفقات', 'danger');
                    }
                })
                .catch(error => {
                    console.error('Error clearing trades:', error);
                    showToast('خطأ في الاتصال بالخادم', 'danger');
                });
            }
        }
        
        // Load dashboard data
        function loadDashboardData() {
            fetch('/api/dashboard_data')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        botStatus = data.bot_status;
                        tradingMode = data.trading_mode;
                        openTrades = data.open_trades;
                        marketState = data.market_state;
                        
                        updateBotStatusUI();
                        updateTradingModeUI();
                        updateOpenTradesCount();
                        updateMarketStateUI();
                        renderOpenTrades();
                        updatePerformanceStats(data.performance);
                    }
                })
                .catch(error => {
                    console.error('Error loading dashboard data:', error);
                });
        }
        
        // Load trades data
        function loadTradesData() {
            fetch('/api/trades_data')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        allTrades = data.trades;
                        renderTrades();
                    }
                })
                .catch(error => {
                    console.error('Error loading trades data:', error);
                });
        }
        
        // Load notifications data
        function loadNotificationsData() {
            fetch('/api/notifications_data')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        notifications = data.notifications;
                        renderNotifications();
                    }
                })
                .catch(error => {
                    console.error('Error loading notifications data:', error);
                });
        }
        
        // Load rejections data
        function loadRejectionsData() {
            fetch('/api/rejections_data')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        rejections = data.rejections;
                        renderRejections();
                    }
                })
                .catch(error => {
                    console.error('Error loading rejections data:', error);
                });
        }
        
        // Load strategies data
        function loadStrategiesData() {
            fetch('/api/strategies_data')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const strategies = data.strategies;
                        
                        document.querySelectorAll('.strategy-toggle').forEach(toggle => {
                            const strategyCard = toggle.closest('.strategy-card');
                            const strategy = strategyCard.dataset.strategy;
                            
                            if (strategy in strategies) {
                                toggle.checked = strategies[strategy];
                                
                                if (strategies[strategy]) {
                                    strategyCard.classList.remove('strategy-disabled');
                                    strategyCard.classList.add('strategy-enabled');
                                } else {
                                    strategyCard.classList.remove('strategy-enabled');
                                    strategyCard.classList.add('strategy-disabled');
                                }
                            }
                        });
                    }
                })
                .catch(error => {
                    console.error('Error loading strategies data:', error);
                });
        }
        
        // Load settings data
        function loadSettingsData() {
            fetch('/api/settings_data')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const settings = data.settings;
                        
                        document.getElementById('min-trade-amount').value = settings.min_trade_amount || 4.5;
                        document.getElementById('max-trade-amount').value = settings.max_trade_amount || 6.5;
                        document.getElementById('max-open-trades').value = settings.max_open_trades || 3;
                        document.getElementById('paper-trade-amount').value = settings.paper_trade_amount || 10.0;
                        document.getElementById('auto-fallback').checked = settings.auto_fallback !== false;
                        document.getElementById('signal-quality').value = settings.signal_quality || 70;
                        document.getElementById('signal-quality-value').textContent = (settings.signal_quality || 70) + '%';
                        document.getElementById('trailing-stop').value = settings.trailing_stop || 1.0;
                        document.getElementById('cooldown-minutes').value = settings.cooldown_minutes || 30;
                        document.getElementById('signal-timeframe').value = settings.signal_timeframe || '5m';
                        document.getElementById('telegram-notifications').checked = settings.telegram_notifications !== false;
                        document.getElementById('email-notifications').checked = settings.email_notifications || false;
                        document.getElementById('min-profit-notification').value = settings.min_profit_notification || 1.0;
                        document.getElementById('max-loss-notification').value = settings.max_loss_notification || -1.0;
                    }
                })
                .catch(error => {
                    console.error('Error loading settings data:', error);
                });
        }
        
        // Update bot status UI
        function updateBotStatusUI() {
            const statusElement = document.getElementById('bot-status');
            const toggleElement = document.getElementById('trading-toggle');
            
            if (botStatus) {
                statusElement.textContent = 'نشط';
                statusElement.className = 'badge bg-success';
                toggleElement.checked = true;
            } else {
                statusElement.textContent = 'متوقف';
                statusElement.className = 'badge bg-danger';
                toggleElement.checked = false;
            }
        }
        
        // Update trading mode UI
        function updateTradingModeUI() {
            const modeElement = document.getElementById('trading-mode');
            const paperModeElement = document.getElementById('paper-mode');
            const realModeElement = document.getElementById('real-mode');
            
            if (tradingMode === 'paper') {
                modeElement.textContent = 'ورقي';
                modeElement.className = 'badge bg-warning';
                paperModeElement.checked = true;
            } else {
                modeElement.textContent = 'حقيقي';
                modeElement.className = 'badge bg-danger';
                realModeElement.checked = true;
            }
        }
        
        // Update open trades count
        function updateOpenTradesCount() {
            const countElement = document.getElementById('open-trades-count');
            const badgeElement = document.getElementById('open-trades-badge');
            
            const count = openTrades.length;
            countElement.textContent = count;
            badgeElement.textContent = count;
        }
        
        // Update market state UI
        function updateMarketStateUI() {
            if (!marketState) return;
            
            // Market regime
            const regimeElement = document.getElementById('market-regime');
            if (marketState.market_regime) {
                regimeElement.textContent = getMarketRegimeText(marketState.market_regime);
                regimeElement.className = `badge ${getMarketRegimeClass(marketState.market_regime)}`;
            }
            
            // Volatility state
            const volatilityElement = document.getElementById('volatility-state');
            if (marketState.volatility_state) {
                volatilityElement.textContent = getVolatilityStateText(marketState.volatility_state);
                volatilityElement.className = `badge ${getVolatilityStateClass(marketState.volatility_state)}`;
            }
            
            // Trend details
            if (marketState.trend_details_by_tf) {
                for (const [tf, details] of Object.entries(marketState.trend_details_by_tf)) {
                    const trendElement = document.getElementById(`trend-${tf}`);
                    const adxElement = document.getElementById(`adx-${tf}`);
                    
                    if (trendElement) {
                        trendElement.innerHTML = getTrendIcon(details.trend);
                        trendElement.className = `fs-4 ${getTrendClass(details.trend)}`;
                    }
                    
                    if (adxElement) {
                        adxElement.textContent = details.adx ? details.adx.toFixed(1) : '--';
                    }
                }
            }
        }
        
        // Get market regime text
        function getMarketRegimeText(regime) {
            const regimeMap = {
                'trending': 'اتجاهي',
                'ranging': 'جانبي',
                'volatile': 'متقلب',
                'unknown': 'غير معروف'
            };
            return regimeMap[regime] || regime;
        }
        
        // Get market regime class
        function getMarketRegimeClass(regime) {
            const regimeClassMap = {
                'trending': 'market-regime-trending',
                'ranging': 'market-regime-ranging',
                'volatile': 'market-regime-volatile',
                'unknown': 'bg-secondary'
            };
            return regimeClassMap[regime] || 'bg-secondary';
        }
        
        // Get volatility state text
        function getVolatilityStateText(state) {
            const stateMap = {
                'low': 'منخفض',
                'medium': 'متوسط',
                'high': 'عالي'
            };
            return stateMap[state] || state;
        }
        
        // Get volatility state class
        function getVolatilityStateClass(state) {
            const stateClassMap = {
                'low': 'volatility-low',
                'medium': 'volatility-medium',
                'high': 'volatility-high'
            };
            return stateClassMap[state] || 'bg-secondary';
        }
        
        // Get trend icon
        function getTrendIcon(trend) {
            const trendIconMap = {
                'bullish': '<i class="fas fa-arrow-up"></i>',
                'bearish': '<i class="fas fa-arrow-down"></i>',
                'neutral': '<i class="fas fa-minus"></i>'
            };
            return trendIconMap[trend] || '<i class="fas fa-minus"></i>';
        }
        
        // Get trend class
        function getTrendClass(trend) {
            const trendClassMap = {
                'bullish': 'trend-up',
                'bearish': 'trend-down',
                'neutral': 'trend-neutral'
            };
            return trendClassMap[trend] || 'trend-neutral';
        }
        
        // Render open trades
        function renderOpenTrades() {
            const tableBody = document.getElementById('open-trades-table');
            
            if (openTrades.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="9" class="text-center">لا توجد صفقات مفتوحة</td></tr>';
                return;
            }
            
            let html = '';
            
            openTrades.forEach(trade => {
                const profitPercent = trade.profit_percentage || 0;
                const profitClass = profitPercent >= 0 ? 'text-success' : 'text-danger';
                const tradeType = trade.is_real_trade ? 'حقيقية' : 'ورقية';
                const tradeTypeClass = trade.is_real_trade ? 'bg-danger' : 'bg-warning';
                
                html += `
                    <tr>
                        <td>${trade.symbol}</td>
                        <td>${STRATEGY_NAMES[trade.strategy_name] || trade.strategy_name}</td>
                        <td>${trade.entry_price.toFixed(4)}</td>
                        <td>${trade.current_price ? trade.current_price.toFixed(4) : '--'}</td>
                        <td>${trade.stop_loss.toFixed(4)}</td>
                        <td>${trade.target_price_1 ? trade.target_price_1.toFixed(4) : '--'}</td>
                        <td>${trade.target_price_2 ? trade.target_price_2.toFixed(4) : '--'}</td>
                        <td class="${profitClass}">${profitPercent.toFixed(2)}%</td>
                        <td><span class="badge ${tradeTypeClass}">${tradeType}</span></td>
                    </tr>
                `;
            });
            
            tableBody.innerHTML = html;
        }
        
        // Render trades
        function renderTrades() {
            const tableBody = document.getElementById('all-trades-table');
            
            if (allTrades.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="8" class="text-center">لا توجد صفقات</td></tr>';
                return;
            }
            
            let html = '';
            
            allTrades.forEach(trade => {
                const profitPercent = trade.profit_percentage || 0;
                const profitClass = profitPercent >= 0 ? 'text-success' : 'text-danger';
                const statusClass = trade.status === 'open' ? 'bg-primary' : profitPercent >= 0 ? 'bg-success' : 'bg-danger';
                const statusText = trade.status === 'open' ? 'مفتوحة' : profitPercent >= 0 ? 'مغلقة بربح' : 'مغلقة بخسارة';
                const tradeType = trade.is_real_trade ? 'حقيقية' : 'ورقية';
                const tradeTypeClass = trade.is_real_trade ? 'bg-danger' : 'bg-warning';
                
                html += `
                    <tr>
                        <td>${trade.symbol}</td>
                        <td>${STRATEGY_NAMES[trade.strategy_name] || trade.strategy_name}</td>
                        <td>${trade.entry_price.toFixed(4)}</td>
                        <td>${trade.closing_price ? trade.closing_price.toFixed(4) : '--'}</td>
                        <td class="${profitClass}">${profitPercent.toFixed(2)}%</td>
                        <td><span class="badge ${statusClass}">${statusText}</span></td>
                        <td><span class="badge ${tradeTypeClass}">${tradeType}</span></td>
                        <td>${formatDateTime(trade.created_at)}</td>
                    </tr>
                `;
            });
            
            tableBody.innerHTML = html;
        }
        
        // Render notifications
        function renderNotifications() {
            const container = document.getElementById('notifications-container');
            
            if (notifications.length === 0) {
                container.innerHTML = '<div class="notification-item"><div class="text-center text-muted">لا توجد إشعارات</div></div>';
                return;
            }
            
            let html = '';
            
            notifications.forEach(notification => {
                const typeClass = getNotificationTypeClass(notification.type);
                const typeText = getNotificationTypeText(notification.type);
                
                html += `
                    <div class="notification-item">
                        <div class="d-flex justify-content-between">
                            <div>
                                <span class="badge ${typeClass} me-2">${typeText}</span>
                                <span>${notification.message}</span>
                            </div>
                            <small class="text-muted">${formatDateTime(notification.timestamp)}</small>
                        </div>
                    </div>
                `;
            });
            
            container.innerHTML = html;
        }
        
        // Render rejections
        function renderRejections() {
            const container = document.getElementById('rejections-container');
            
            if (rejections.length === 0) {
                container.innerHTML = '<div class="rejection-item"><div class="text-center text-muted">لا توجد رفوض</div></div>';
                return;
            }
            
            let html = '';
            
            rejections.forEach(rejection => {
                html += `
                    <div class="rejection-item">
                        <div class="d-flex justify-content-between">
                            <div>
                                <span class="me-2">${rejection.symbol}</span>
                                <span class="text-muted">${rejection.reason}</span>
                            </div>
                            <small class="text-muted">${formatDateTime(rejection.timestamp)}</small>
                        </div>
                    </div>
                `;
            });
            
            container.innerHTML = html;
        }
        
        // Get notification type class
        function getNotificationTypeClass(type) {
            const typeClassMap = {
                'info': 'bg-primary',
                'success': 'bg-success',
                'warning': 'bg-warning',
                'error': 'bg-danger',
                'system': 'bg-info',
                'trade_opened': 'bg-success',
                'trade_closed': 'bg-primary'
            };
            return typeClassMap[type] || 'bg-secondary';
        }
        
        // Get notification type text
        function getNotificationTypeText(type) {
            const typeTextMap = {
                'info': 'معلومات',
                'success': 'نجاح',
                'warning': 'تحذير',
                'error': 'خطأ',
                'system': 'نظام',
                'trade_opened': 'صفقة مفتوحة',
                'trade_closed': 'صفقة مغلقة'
            };
            return typeTextMap[type] || type;
        }
        
        // Update performance stats
        function updatePerformanceStats(performance) {
            if (!performance) return;
            
            document.getElementById('total-trades').textContent = performance.total_trades || 0;
            document.getElementById('win-rate').textContent = (performance.win_rate || 0).toFixed(1) + '%';
            document.getElementById('avg-profit').textContent = (performance.avg_profit || 0).toFixed(2) + '%';
            document.getElementById('total-profit').textContent = (performance.total_profit || 0).toFixed(2) + '%';
        }
        
        // Update prices
        function updatePrices(prices) {
            // Update open trades with new prices
            openTrades.forEach(trade => {
                if (prices[trade.symbol]) {
                    trade.current_price = prices[trade.symbol];
                    
                    // Calculate profit percentage
                    if (trade.entry_price > 0) {
                        trade.profit_percentage = ((prices[trade.symbol] - trade.entry_price) / trade.entry_price) * 100;
                    }
                }
            });
            
            renderOpenTrades();
        }
        
        // Add notification
        function addNotification(notification) {
            notifications.unshift(notification);
            if (notifications.length > 20) {
                notifications = notifications.slice(0, 20);
            }
            
            renderNotifications();
            
            // Show toast for important notifications
            if (notification.type === 'trade_opened' || notification.type === 'trade_closed') {
                showToast(notification.message, notification.type === 'trade_opened' ? 'success' : 'primary');
            }
        }
        
        // Add rejection
        function addRejection(rejection) {
            rejections.unshift(rejection);
            if (rejections.length > 30) {
                rejections = rejections.slice(0, 30);
            }
            
            renderRejections();
        }
        
        // Update trade
        function updateTrade(trade) {
            // Find and update the trade in openTrades
            const index = openTrades.findIndex(t => t.symbol === trade.symbol);
            
            if (index !== -1) {
                if (trade.status === 'closed') {
                    // Remove from open trades
                    openTrades.splice(index, 1);
                    
                    // Add to all trades
                    allTrades.unshift(trade);
                    
                    // Update performance stats
                    loadDashboardData();
                } else {
                    // Update the trade
                    openTrades[index] = { ...openTrades[index], ...trade };
                }
                
                renderOpenTrades();
                renderTrades();
            }
        }
        
        // Update market state
        function updateMarketState(state) {
            marketState = state;
            updateMarketStateUI();
        }
        
        // Update bot status
        function updateBotStatus(status) {
            botStatus = status.enabled;
            updateBotStatusUI();
        }
        
        // Format date time
        function formatDateTime(dateString) {
            const date = new Date(dateString);
            return date.toLocaleString('ar-SA');
        }
        
        // Show toast
        function showToast(message, type = 'info') {
            const toastElement = document.getElementById('liveToast');
            const toastMessage = document.getElementById('toast-message');
            
            // Set message
            toastMessage.textContent = message;
            
            // Set toast class based on type
            const toastHeader = toastElement.querySelector('.toast-header');
            toastHeader.className = 'toast-header';
            
            if (type === 'success') {
                toastHeader.classList.add('bg-success', 'text-white');
            } else if (type === 'danger') {
                toastHeader.classList.add('bg-danger', 'text-white');
            } else if (type === 'warning') {
                toastHeader.classList.add('bg-warning');
            } else {
                toastHeader.classList.add('bg-primary', 'text-white');
            }
            
            // Show toast
            const toast = new bootstrap.Toast(toastElement);
            toast.show();
        }
        
        // Strategy names mapping (for UI)
        const STRATEGY_NAMES = {
            "BB_Stoch_Strategy": "BB+Stoch",
            "MACD_EMA_Strategy": "MACD+SMA",
            "EMA_RSI_Strategy": "EMA+RSI",
            "Pullback_Strategy": "Pullback",
            "Momentum_Volatility_Strategy": "Momentum",
            "Elliott_Wave_Strategy": "Elliott Wave",
            "Range_Reversal_Strategy": "Range Reversal"
        };
    </script>
</body>
</html>
    ''')

# --- واجهات برمجة التطبيقات (API) ---
@app.route('/api/dashboard_data')
def get_dashboard_data():
    global is_trading_enabled, paper_trading_mode, open_signals_cache, current_market_state
    
    with trading_status_lock:
        trading_enabled = is_trading_enabled
    
    with trading_mode_lock:
        trading_mode = paper_trading_mode
    
    with signal_cache_lock:
        open_trades = list(open_signals_cache.values())
    
    with market_state_lock:
        market_state = dict(current_market_state)
    
    # حساب إحصائيات الأداء
    performance = {
        "total_trades": 0,
        "win_rate": 0,
        "avg_profit": 0,
        "total_profit": 0
    }
    
    if check_db_connection() and conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN profit_percentage > 0 THEN 1 ELSE 0 END) as winning_trades,
                        AVG(profit_percentage) as avg_profit,
                        SUM(profit_percentage) as total_profit
                    FROM signals 
                    WHERE status = 'closed'
                """)
                
                stats = cur.fetchone()
                
                if stats and stats['total_trades'] > 0:
                    performance = {
                        "total_trades": stats['total_trades'],
                        "win_rate": (stats['winning_trades'] / stats['total_trades']) * 100,
                        "avg_profit": stats['avg_profit'] or 0,
                        "total_profit": stats['total_profit'] or 0
                    }
        except Exception as e:
            logger.error(f"❌ [API] Error fetching performance stats: {e}")
    
    return jsonify({
        "success": True,
        "bot_status": trading_enabled,
        "trading_mode": "paper" if trading_mode else "real",
        "open_trades": open_trades,
        "market_state": market_state,
        "performance": performance
    })

@app.route('/api/toggle_trading', methods=['POST'])
def toggle_trading():
    global is_trading_enabled
    
    data = request.get_json()
    enabled = data.get('enabled', False)
    
    with trading_status_lock:
        is_trading_enabled = enabled
    
    log_and_notify("info", f"تم {'تفعيل' if enabled else 'إيقاف'} التداول", "system")
    
    return jsonify({"success": True, "enabled": enabled})

@app.route('/api/change_trading_mode', methods=['POST'])
def change_trading_mode():
    global paper_trading_mode
    
    data = request.get_json()
    mode = data.get('mode', 'paper')
    
    with trading_mode_lock:
        paper_trading_mode = (mode == 'paper')
    
    # حفظ الإعدادات في Redis
    save_settings_to_redis()
    
    log_and_notify("info", f"تم التبديل إلى وضع التداول {'الورقي' if mode == 'paper' else 'الحقيقي'}", "system")
    
    return jsonify({"success": True, "mode": mode})

@app.route('/api/scan_now', methods=['POST'])
def scan_now():
    # تشغيل الفحص في خلفية
    thread = Thread(target=check_and_execute_signals)
    thread.daemon = True
    thread.start()
    
    return jsonify({"success": True})

@app.route('/api/trades_data')
def get_trades_data():
    if not check_db_connection() or not conn:
        return jsonify({"success": False, "message": "Database connection error"})
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM signals 
                ORDER BY created_at DESC 
                LIMIT 100
            """)
            
            trades = [dict(trade) for trade in cur.fetchall()]
            
            # Format dates
            for trade in trades:
                if trade.get('created_at'):
                    trade['created_at'] = trade['created_at'].isoformat()
                if trade.get('closed_at'):
                    trade['closed_at'] = trade['closed_at'].isoformat()
            
            return jsonify({"success": True, "trades": trades})
    
    except Exception as e:
        logger.error(f"❌ [API] Error fetching trades data: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/notifications_data')
def get_notifications_data():
    with notifications_lock:
        notifications_list = list(notifications_cache)
    
    return jsonify({"success": True, "notifications": notifications_list})

@app.route('/api/rejections_data')
def get_rejections_data():
    with rejection_logs_lock:
        rejections_list = list(rejection_logs_cache)
    
    return jsonify({"success": True, "rejections": rejections_list})

@app.route('/api/strategies_data')
def get_strategies_data():
    strategies = {
        "USE_BB_STOCH_STRATEGY": USE_BB_STOCH_STRATEGY,
        "USE_MACD_EMA_STRATEGY": USE_MACD_EMA_STRATEGY,
        "USE_EMA_RSI_STRATEGY": USE_EMA_RSI_STRATEGY,
        "USE_PULLBACK_STRATEGY": USE_PULLBACK_STRATEGY,
        "USE_MOMENTUM_VOLATILITY_STRATEGY": USE_MOMENTUM_VOLATILITY_STRATEGY,
        "USE_ELLIOTT_WAVE_STRATEGY": USE_ELLIOTT_WAVE_STRATEGY,
        "USE_RANGE_REVERSAL_STRATEGY": USE_RANGE_REVERSAL_STRATEGY
    }
    
    return jsonify({"success": True, "strategies": strategies})

@app.route('/api/settings_data')
def get_settings_data():
    settings = {
        "min_trade_amount": FIXED_TRADE_AMOUNT_MIN_USDT,
        "max_trade_amount": FIXED_TRADE_AMOUNT_MAX_USDT,
        "max_open_trades": MAX_OPEN_TRADES,
        "paper_trade_amount": PAPER_TRADE_FIXED_AMOUNT_USDT,
        "auto_fallback": AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE,
        "signal_quality": MIN_SIGNAL_QUALITY,
        "trailing_stop": TRAILING_STOP_ACTIVATION_PROFIT_PERCENT,
        "cooldown_minutes": COOLDOWN_MINUTES_AFTER_SL,
        "signal_timeframe": SIGNAL_GENERATION_TIMEFRAME,
        "telegram_notifications": get_notification_settings().get('telegram_enabled', True),
        "email_notifications": get_notification_settings().get('email_enabled', False),
        "min_profit_notification": get_notification_settings().get('min_profit_notification', 1.0),
        "max_loss_notification": get_notification_settings().get('max_loss_notification', -1.0)
    }
    
    return jsonify({"success": True, "settings": settings})

@app.route('/api/save_strategies', methods=['POST'])
def save_strategies():
    global USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY, USE_MOMENTUM_VOLATILITY_STRATEGY, USE_ELLIOTT_WAVE_STRATEGY, USE_RANGE_REVERSAL_STRATEGY
    
    data = request.get_json()
    strategies = data.get('strategies', {})
    
    USE_BB_STOCH_STRATEGY = strategies.get('USE_BB_STOCH_STRATEGY', True)
    USE_MACD_EMA_STRATEGY = strategies.get('USE_MACD_EMA_STRATEGY', True)
    USE_EMA_RSI_STRATEGY = strategies.get('USE_EMA_RSI_STRATEGY', True)
    USE_PULLBACK_STRATEGY = strategies.get('USE_PULLBACK_STRATEGY', True)
    USE_MOMENTUM_VOLATILITY_STRATEGY = strategies.get('USE_MOMENTUM_VOLATILITY_STRATEGY', True)
    USE_ELLIOTT_WAVE_STRATEGY = strategies.get('USE_ELLIOTT_WAVE_STRATEGY', True)
    USE_RANGE_REVERSAL_STRATEGY = strategies.get('USE_RANGE_REVERSAL_STRATEGY', True)
    
    # حفظ الإعدادات في Redis
    save_settings_to_redis()
    
    log_and_notify("info", "تم تحديث إعدادات الاستراتيجيات", "system")
    
    return jsonify({"success": True})

@app.route('/api/save_settings', methods=['POST'])
def save_settings():
    global FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT, MAX_OPEN_TRADES, PAPER_TRADE_FIXED_AMOUNT_USDT, AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE, MIN_SIGNAL_QUALITY, TRAILING_STOP_ACTIVATION_PROFIT_PERCENT, COOLDOWN_MINUTES_AFTER_SL
    
    data = request.get_json()
    
    with trade_amount_lock:
        FIXED_TRADE_AMOUNT_MIN_USDT = data.get('min_trade_amount', 4.5)
        FIXED_TRADE_AMOUNT_MAX_USDT = data.get('max_trade_amount', 6.5)
    
    MAX_OPEN_TRADES = data.get('max_open_trades', 3)
    PAPER_TRADE_FIXED_AMOUNT_USDT = data.get('paper_trade_amount', 10.0)
    AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE = data.get('auto_fallback', True)
    
    with min_quality_lock:
        MIN_SIGNAL_QUALITY = data.get('signal_quality', 70)
    
    TRAILING_STOP_ACTIVATION_PROFIT_PERCENT = data.get('trailing_stop', 1.0)
    COOLDOWN_MINUTES_AFTER_SL = data.get('cooldown_minutes', 30)
    
    # حفظ إعدادات الإشعارات
    if redis_client:
        notification_settings = {
            'telegram_enabled': data.get('telegram_notifications', True),
            'email_enabled': data.get('email_notifications', False),
            'min_profit_notification': data.get('min_profit_notification', 1.0),
            'max_loss_notification': data.get('max_loss_notification', -1.0)
        }
        redis_client.set('notification_settings', json.dumps(notification_settings))
    
    # حفظ الإعدادات في Redis
    save_settings_to_redis()
    
    log_and_notify("info", "تم تحديث إعدادات البوت", "system")
    
    return jsonify({"success": True})

@app.route('/api/clear_notifications', methods=['POST'])
def clear_notifications():
    if not check_db_connection() or not conn:
        return jsonify({"success": False, "message": "Database connection error"})
    
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM notifications;")
            conn.commit()
        
        with notifications_lock:
            notifications_cache.clear()
        
        log_and_notify("info", "تم مسح جميع الإشعارات", "system")
        
        return jsonify({"success": True})
    
    except Exception as e:
        logger.error(f"❌ [API] Error clearing notifications: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/clear_rejections', methods=['POST'])
def clear_rejections():
    with rejection_logs_lock:
        rejection_logs_cache.clear()
    
    log_and_notify("info", "تم مسح سجل الرفوض", "system")
    
    return jsonify({"success": True})

@app.route('/api/get_filtered_trades', methods=['POST'])
def get_filtered_trades():
    if not check_db_connection() or not conn:
        return jsonify({"success": False, "message": "Database connection error"})
    
    data = request.get_json()
    
    status = data.get('status', 'all')
    type_filter = data.get('type', 'all')
    strategy = data.get('strategy', 'all')
    symbol = data.get('symbol', '')
    from_date = data.get('from_date', '')
    to_date = data.get('to_date', '')
    
    try:
        query = "SELECT * FROM signals WHERE 1=1"
        params = []
        
        if status != 'all':
            query += " AND status = %s"
            params.append(status)
        
        if type_filter != 'all':
            is_real = (type_filter == 'real')
            query += " AND is_real_trade = %s"
            params.append(is_real)
        
        if strategy != 'all':
            query += " AND strategy_name = %s"
            params.append(strategy)
        
        if symbol:
            query += " AND symbol ILIKE %s"
            params.append(f"%{symbol}%")
        
        if from_date:
            query += " AND created_at >= %s"
            params.append(from_date)
        
        if to_date:
            query += " AND created_at <= %s"
            params.append(to_date + " 23:59:59")
        
        query += " ORDER BY created_at DESC LIMIT 100"
        
        with conn.cursor() as cur:
            cur.execute(query, params)
            trades = [dict(trade) for trade in cur.fetchall()]
            
            # Format dates
            for trade in trades:
                if trade.get('created_at'):
                    trade['created_at'] = trade['created_at'].isoformat()
                if trade.get('closed_at'):
                    trade['closed_at'] = trade['closed_at'].isoformat()
            
            return jsonify({"success": True, "trades": trades})
    
    except Exception as e:
        logger.error(f"❌ [API] Error fetching filtered trades: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/export_trades')
def export_trades():
    if not check_db_connection() or not conn:
        return jsonify({"success": False, "message": "Database connection error"})
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT symbol, strategy_name, entry_price, closing_price, stop_loss, 
                       profit_percentage, status, is_real_trade, created_at, closed_at
                FROM signals 
                ORDER BY created_at DESC
            """)
            
            trades = cur.fetchall()
        
        # Create CSV
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Symbol', 'Strategy', 'Entry Price', 'Closing Price', 'Stop Loss',
            'Profit %', 'Status', 'Trade Type', 'Created At', 'Closed At'
        ])
        
        # Write data
        for trade in trades:
            writer.writerow([
                trade['symbol'],
                trade['strategy_name'],
                trade['entry_price'],
                trade['closing_price'] or '',
                trade['stop_loss'],
                trade['profit_percentage'] or '',
                trade['status'],
                'Real' if trade['is_real_trade'] else 'Paper',
                trade['created_at'].isoformat() if trade['created_at'] else '',
                trade['closed_at'].isoformat() if trade['closed_at'] else ''
            ])
        
        output.seek(0)
        
        # Create response
        response = Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=trades_{datetime.now().strftime("%Y%m%d")}.csv'
            }
        )
        
        return response
    
    except Exception as e:
        logger.error(f"❌ [API] Error exporting trades: {e}")
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/clear_trades', methods=['POST'])
def clear_trades():
    if not check_db_connection() or not conn:
        return jsonify({"success": False, "message": "Database connection error"})
    
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM signals;")
            conn.commit()
        
        with signal_cache_lock:
            open_signals_cache.clear()
        
        log_and_notify("warning", "تم مسح جميع الصفقات", "system")
        
        return jsonify({"success": True})
    
    except Exception as e:
        logger.error(f"❌ [API] Error clearing trades: {e}")
        return jsonify({"success": False, "message": str(e)})

# --- نقطة نهاية WebSocket ---
@sock.route('/ws')
def websocket_connection(ws):
    with ws_clients_lock:
        ws_clients.append(ws)
    
    try:
        while True:
            # Keep the connection alive
            data = ws.receive()
            if data is None:
                break
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
    finally:
        with ws_clients_lock:
            if ws in ws_clients:
                ws_clients.remove(ws)

# --- الدالة الرئيسية ---
def main():
    # تهيئة الاتصالات
    init_db()
    init_redis()
    
    # تهيئة عميل Binance
    global client
    try:
        client = Client(API_KEY, API_SECRET)
        logger.info("✅ [API] Binance client initialized successfully.")
    except Exception as e:
        logger.error(f"❌ [API] Error initializing Binance client: {e}")
        exit(1)
    
    # تحميل الإعدادات
    load_settings_from_redis()
    
    # تحميل البيانات الأولية
    load_open_signals_to_cache()
    load_notifications_to_cache()
    
    # بدء WebSocket
    start_websocket()
    
    # بدء تقارير دورية
    start_periodic_reports()
    
    # بدء حلقة توليد الإشارات
    signal_thread = Thread(target=signal_generation_loop)
    signal_thread.daemon = True
    signal_thread.start()
    
    # بدء تطبيق Flask
    logger.info("✅ Starting Flask application...")
    app.run(host='0.0.0.0', port=5000, threaded=True)

if __name__ == '__main__':
    main()