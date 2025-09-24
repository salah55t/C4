# ملف c4_5min_v35_0_2.py - نسخة V35.0.2 (إصلاح الاستقرار وزيادة فعالية الإشارات)
# --- وصف التعديلات:
# 1. [إصلاح WebSocket] تعديل طريقة تشغيل الخادم لحل خطأ "RuntimeError" وضمان استقرار لوحة التحكم.
# 2. [تحسين الإشارات] تخفيف صرامة بعض الفلاتر الديناميكية وشروط الاستراتيجيات لزيادة عدد الإشارات الصالحة.
# 3. [تسجيل مُحسَّن] إضافة سجلات تفصيلية لتوضيح سبب رفض الإشارات، مما يساعد في التحليل المستقبلي.
# 4. [تحسينات عامة] تحسينات طفيفة على الأداء واستقرار الكود.

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
logger = logging.getLogger('CryptoBotV35.0.2_5min')

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
        dynamic_bb_threshold = bb_width.rolling(20).mean() * 1.4 # Relaxed from 1.5
    elif volatility_state == "low":
        dynamic_bb_threshold = bb_width.rolling(20).mean() * 0.9
    else:
        dynamic_bb_threshold = bb_width.rolling(20).mean() * 1.1 # Relaxed from 1.2

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
        default_adx_thresh = 21 # Relaxed from 22
    elif market_regime == "trending":
        default_adx_thresh = 18
    else:
        default_adx_thresh = 19 # Relaxed from 20
    
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
            rsi_lower, rsi_upper = 40, 65 # Widened range
        else:
            rsi_lower, rsi_upper = 48, 75 # Widened range
    
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
        pullback_threshold = 0.035  # 3.5%
    elif market_regime == "trending":
        pullback_threshold = 0.022  # 2.2%
    else:
        pullback_threshold = 0.030  # 3.0%
    
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
            failed_reasons = {k for k, v in dynamic_filters.items() if not v}
            logger.debug(f"[Strategy Check] BB_Stoch for {symbol}: Dynamic filters failed. Reasons: {failed_reasons}")
            return None
        
        # شروط الاستراتيجية
        condition1 = last_row['close'] > last_row['bb_middle']  # السعر فوق منتصف البولينجر
        condition2 = last_row['stoch_k'] > last_row['stoch_d']  # ستوكاستيك صاعد
        condition3 = last_row['stoch_k'] < 80  # ستوكاستيك ليس في منطقة الشراء المفرط
        condition4 = prev_row['stoch_k'] <= prev_row['stoch_d']  # تقاطع صاعد في الستوكاستيك
        
        # فحص اتجاه السوق
        if last_row['ema50'] < last_row['ema200'] or last_row['close'] < last_row['ema50']:
            logger.debug(f"[Strategy Check] BB_Stoch for {symbol}: Rejected due to bearish trend.")
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
            failed_reasons = {k for k, v in dynamic_filters.items() if not v}
            logger.debug(f"[Strategy Check] MACD_EMA for {symbol}: Dynamic filters failed. Reasons: {failed_reasons}")
            return None
        
        # شروط الاستراتيجية
        condition1 = last_row['macd'] > last_row['macd_signal']  # MACD فوق إشارته
        condition2 = prev_row['macd'] <= prev_row['macd_signal']  # تقاطع صاعد
        condition3 = last_row['macd_hist'] > 0  # MACD histogram موجب
        condition4 = last_row['close'] > last_row['ema9']  # السعر فوق EMA9
        
        # فحص اتجاه السوق
        if last_row['ema9'] < last_row['ema21'] or last_row['ema21'] < last_row['ema50']:
            logger.debug(f"[Strategy Check] MACD_EMA for {symbol}: Rejected due to bearish trend.")
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
            failed_reasons = {k for k, v in dynamic_filters.items() if not v}
            logger.debug(f"[Strategy Check] EMA_RSI for {symbol}: Dynamic filters failed. Reasons: {failed_reasons}")
            return None
        
        # شروط الاستراتيجية
        condition1 = prev_row['rsi'] < 55 and last_row['rsi'] > 55 # تقاطع صاعد لمستوى 55
        condition2 = last_row['close'] > last_row['ema9']  # السعر فوق EMA9
        condition3 = last_row['ema9'] > last_row['ema21']  # EMA9 فوق EMA21
        
        # فحص اتجاه السوق طويل الأجل
        if last_row['ema50'] < last_row['ema200']:
            logger.debug(f"[Strategy Check] EMA_RSI for {symbol}: Rejected due to long-term bearish trend.")
            return None
        
        if condition1 and condition2 and condition3:
            atr = last_row['atr']
            stop_loss = max(last_row['low'] - (atr * 0.6), last_row['close'] * 0.982)
            
            # حساب الأهداف
            target1 = last_row['close'] + (atr * 1.6)
            target2 = last_row['close'] + (atr * 3.2)
            
            # حساب جودة الإشارة
            quality_score = 70
            if last_row['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.25:
                quality_score += 10
            if last_row['rsi'] > 50 and last_row['rsi'] < 65:
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
        
        # فحص الفلاتر الديناميكية
        dynamic_filters = check_pullback_dynamic_filters(df)
        if not all(dynamic_filters.values()):
            failed_reasons = {k for k, v in dynamic_filters.items() if not v}
            logger.debug(f"[Strategy Check] Pullback for {symbol}: Dynamic filters failed. Reasons: {failed_reasons}")
            return None
        
        # شروط الاستراتيجية
        recent_high = df['high'].rolling(10).max().iloc[-1]
        pullback_percent = (recent_high - last_row['close']) / recent_high
        recent_low = df['low'].rolling(5).min().iloc[-1]
        recovery_percent = (last_row['close'] - recent_low) / recent_low
        
        condition1 = pullback_percent >= 0.025  # ارتداد 2.5% على الأقل
        condition2 = recovery_percent >= 0.01  # تعافي 1% على الأقل
        condition3 = last_row['close'] > last_row['ema21']  # السعر فوق EMA21
        condition4 = last_row['ema21'] > last_row['ema50']  # EMA21 فوق EMA50
        
        # فحص اتجاه السوق
        if last_row['ema50'] < last_row['ema200'] or last_row['adx'] < 18:
            logger.debug(f"[Strategy Check] Pullback for {symbol}: Rejected due to weak trend.")
            return None
        
        if condition1 and condition2 and condition3 and condition4:
            atr = last_row['atr']
            stop_loss = max(recent_low - (atr * 0.3), last_row['close'] * 0.985)
            
            # حساب الأهداف
            target1 = last_row['close'] + (atr * 1.7)
            target2 = recent_high
            
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
        
        # فحص الفلاتر الديناميكية
        dynamic_filters = check_momentum_dynamic_filters(df)
        if not all(dynamic_filters.values()):
            failed_reasons = {k for k, v in dynamic_filters.items() if not v}
            logger.debug(f"[Strategy Check] Momentum for {symbol}: Dynamic filters failed. Reasons: {failed_reasons}")
            return None
        
        # شروط الاستراتيجية
        price_change = (last_row['close'] - df['close'].iloc[-5]) / df['close'].iloc[-5]
        atr_percent = last_row['atr_percent'] / 100
        
        condition1 = price_change >= 0.01
        condition2 = atr_percent <= 0.02
        condition3 = last_row['close'] > last_row['ema9']
        condition4 = last_row['ema9'] > last_row['ema21'] and last_row['ema21'] > last_row['ema50']
        
        # فحص اتجاه السوق
        if last_row['ema50'] < last_row['ema200']:
            logger.debug(f"[Strategy Check] Momentum for {symbol}: Rejected due to bearish EMA order.")
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
            failed_reasons = {k for k, v in dynamic_filters.items() if not v}
            logger.debug(f"[Strategy Check] Elliott Wave for {symbol}: Dynamic filters failed. Reasons: {failed_reasons}")
            return None
        
        # شروط الاستراتيجية
        fib_retracement = get_wave_retracement(df)
        condition1 = 0.382 <= fib_retracement <= 0.618
        condition2 = last_row['close'] > last_row['ema21']
        condition3 = last_row['ema21'] > last_row['ema50']
        
        # فحص اتجاه السوق
        if last_row['ema50'] < last_row['ema200']:
            logger.debug(f"[Strategy Check] Elliott Wave for {symbol}: Rejected due to bearish trend.")
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
            failed_reasons = {k for k, v in dynamic_filters.items() if not v}
            logger.debug(f"[Strategy Check] Range Reversal for {symbol}: Dynamic filters failed. Reasons: {failed_reasons}")
            return None
        
        # شروط الاستراتيجية
        condition1 = last_row['adx'] < 23
        condition2 = last_row['rsi'] < 30
        condition3 = last_row['close'] > prev_row['close']
        condition4 = last_row['close'] > last_row['bb_lower']
        
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
            if last_row['macd_hist'] > df['macd_hist'].iloc[-3]:
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
            if risk_percent > 0:
                amount_usdt = FIXED_TRADE_AMOUNT_MIN_USDT + (FIXED_TRADE_AMOUNT_MAX_USDT - FIXED_TRADE_AMOUNT_MIN_USDT) * min(1.0, 2.0 / risk_percent)
            else:
                amount_usdt = FIXED_TRADE_AMOUNT_MIN_USDT
            amount_usdt = max(FIXED_TRADE_AMOUNT_MIN_USDT, min(amount_usdt, FIXED_TRADE_AMOUNT_MAX_USDT))
    else:
        amount_usdt = PAPER_TRADE_FIXED_AMOUNT_USDT
    
    quantity = amount_usdt / entry_price
    
    symbol_info = exchange_info_map.get(symbol, {})
    if symbol_info:
        lot_size_filter = next((f for f in symbol_info.get('filters', []) if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = float(lot_size_filter['stepSize'])
            min_qty = float(lot_size_filter['minQty'])
            quantity = max(min_qty, (quantity // step_size) * step_size)
    
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
    
    quantity, notional_value = calculate_position_size(symbol, entry_price, stop_loss, is_real_trade)
    
    if not check_min_notional(symbol, quantity, entry_price):
        log_rejection(symbol, "MinNotional Filter Failed")
        return False
    
    if is_real_trade:
        with balance_lock:
            if usdt_balance < notional_value:
                log_rejection(symbol, "Insufficient Balance")
                return False
    
    if stop_loss >= entry_price:
        log_rejection(symbol, "Invalid Position Size")
        return False
    
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
            
            signal_copy = signal.copy()
            signal_copy.update({
                'id': trade_id, 'status': 'open', 'is_real_trade': is_real_trade,
                'quantity': quantity, 'initial_quantity': quantity,
                'created_at': datetime.now(timezone.utc).isoformat()
            })
            
            with signal_cache_lock:
                open_signals_cache[symbol] = signal_copy
            
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
    symbol = signal['symbol']
    
    with trading_status_lock:
        if not is_trading_enabled:
            return False
    
    with min_quality_lock:
        if signal['quality_score'] < MIN_SIGNAL_QUALITY:
            log_rejection(symbol, "Low Quality Signal", {'score': signal['quality_score'], 'min': MIN_SIGNAL_QUALITY})
            return False
    
    if get_open_trades_count() >= MAX_OPEN_TRADES:
        return False
    
    if is_symbol_in_cooldown(symbol):
        return False
    
    if get_consecutive_losses(symbol) >= 3:
        return False
    
    with trading_mode_lock:
        is_real_trade = not paper_trading_mode
    
    return execute_trade(signal, is_real_trade)

def update_trailing_stop(symbol: str, current_price: float):
    """
    تحديث وقف الخسارة المتحرك
    """
    with signal_cache_lock:
        if symbol not in open_signals_cache:
            return
        signal = open_signals_cache[symbol]
    
    entry_price = signal['entry_price']
    stop_loss = signal['stop_loss']
    
    profit_percent = ((current_price - entry_price) / entry_price) * 100
    
    if profit_percent >= TRAILING_STOP_ACTIVATION_PROFIT_PERCENT:
        trail_distance = (entry_price - stop_loss) * 0.5
        new_stop_loss = current_price - trail_distance
        
        if new_stop_loss > stop_loss:
            if not check_db_connection() or not conn: return
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE signals SET stop_loss = %s, status = 'updated'
                        WHERE symbol = %s AND status IN ('open', 'updated')
                    """, (new_stop_loss, symbol))
                    conn.commit()
                
                with signal_cache_lock:
                    open_signals_cache[symbol]['stop_loss'] = new_stop_loss
                    open_signals_cache[symbol]['status'] = 'updated'
                
                send_trade_update_notification(
                    symbol, "stop_loss", stop_loss, new_stop_loss,
                    current_price, profit_percent, signal['is_real_trade']
                )
                logger.info(f"✅ Trailing stop updated for {symbol}: {stop_loss:.4f} -> {new_stop_loss:.4f}")
            except Exception as e:
                logger.error(f"❌ [Trade] Error updating trailing stop for {symbol}: {e}")
                if conn: conn.rollback()

def check_trade_exit_conditions(symbol: str, current_price: float) -> Optional[str]:
    """
    فحص شروط الخروج من الصفقة
    """
    with signal_cache_lock:
        if symbol not in open_signals_cache:
            return None
        signal = open_signals_cache[symbol]
    
    if current_price <= signal['stop_loss']: return "stop_loss"
    if current_price >= signal['target_price_2']: return "target_2"
    if current_price >= signal['target_price_1']: return "target_1"
    
    return None

def close_trade(symbol: str, exit_reason: str, current_price: float):
    """
    إغلاق الصفقة
    """
    with signal_cache_lock:
        if symbol not in open_signals_cache: return
        signal = open_signals_cache[symbol]
    
    profit_percent = ((current_price - signal['entry_price']) / signal['entry_price']) * 100
    
    if not check_db_connection() or not conn: return
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE signals 
                SET status = 'closed', closing_price = %s, closed_at = NOW(), 
                    profit_percentage = %s, closing_reason = %s
                WHERE symbol = %s AND status IN ('open', 'updated')
            """, (current_price, profit_percent, exit_reason, symbol))
            conn.commit()
        
        with signal_cache_lock:
            if symbol in open_signals_cache:
                del open_signals_cache[symbol]
        
        if signal['is_real_trade']:
            with balance_lock:
                usdt_balance += (current_price * signal['quantity']) - (signal['entry_price'] * signal['quantity'])
        
        if profit_percent < 0:
            increment_consecutive_losses(symbol)
            if exit_reason == "stop_loss":
                set_symbol_cooldown(symbol)
        else:
            reset_consecutive_losses(symbol)
        
        log_and_notify("info", f"✅ صفقة {symbol} أغلقت: {exit_reason} ({profit_percent:.2f}%)", "trade_closed")
        logger.info(f"✅ {'Real' if signal['is_real_trade'] else 'Paper'} trade closed for {symbol} at {current_price} ({exit_reason}, {profit_percent:.2f}%)")
        
    except Exception as e:
        logger.error(f"❌ [Trade] Error closing trade for {symbol}: {e}")
        if conn: conn.rollback()

def analyze_open_trades():
    """
    تحليل الصفقات المفتوحة وتحديثها
    """
    with signal_cache_lock:
        open_symbols = list(open_signals_cache.keys())
    
    for symbol in open_symbols:
        with live_prices_lock:
            if symbol not in live_prices: continue
            current_price = live_prices[symbol]
        
        try:
            update_trailing_stop(symbol, current_price)
            exit_reason = check_trade_exit_conditions(symbol, current_price)
            if exit_reason:
                close_trade(symbol, exit_reason, current_price)
        except Exception as e:
            logger.error(f"❌ [Trade Analysis] Error analyzing trade for {symbol}: {e}")

def start_trade_analysis_thread():
    """
    بدء خيط تحليل الصفقات
    """
    def trade_analysis_worker():
        while True:
            try:
                analyze_open_trades()
                time.sleep(5)
            except Exception as e:
                logger.error(f"❌ [Trade Analysis] Error in thread: {e}")
                time.sleep(10)
    
    thread = Thread(target=trade_analysis_worker, daemon=True)
    thread.start()
    logger.info("✅ [Trade Analysis] Started trade analysis thread.")

# --- دوال المسح والبحث عن الإشارات ---
def scan_symbol_for_signals(symbol: str) -> Optional[Dict]:
    """
    البحث عن إشارات لعملة معينة
    """
    try:
        df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
        if df is None or len(df) < 50:
            return None
        
        df = calculate_all_features(df)
        if df is None or len(df) < 10:
            return None
        
        strategies = [
            bb_stoch_strategy, macd_ema_strategy, ema_rsi_strategy,
            pullback_strategy, momentum_volatility_strategy,
            elliott_wave_strategy, range_reversal_strategy
        ]
        
        for strategy in strategies:
            signal = strategy(df, symbol)
            if signal:
                logger.info(f"[Signal Found] Strategy {strategy.__name__} found a signal for {symbol}.")
                return signal
        
        return None
    except Exception as e:
        logger.error(f"❌ [Scan] Error scanning {symbol}: {e}")
        return None

def scan_all_symbols_for_signals():
    """
    البحث عن إشارات في جميع العملات
    """
    update_market_state()
    symbols_to_scan = validated_symbols_to_scan.copy()
    random.shuffle(symbols_to_scan)
    
    for symbol in symbols_to_scan:
        with signal_cache_lock:
            if symbol in open_signals_cache: continue
        
        signal = scan_symbol_for_signals(symbol)
        if signal:
            process_signal(signal)
        time.sleep(0.5)

def start_signal_scanning_thread():
    """
    بدء خيط البحث عن الإشارات
    """
    def signal_scanning_worker():
        while True:
            try:
                scan_all_symbols_for_signals()
                now = datetime.now(timezone.utc)
                next_candle_minute = (now.minute // 5 + 1) * 5
                if next_candle_minute >= 60:
                    next_candle = now.replace(hour=(now.hour + 1) % 24, minute=0, second=0, microsecond=0)
                else:
                    next_candle = now.replace(minute=next_candle_minute, second=0, microsecond=0)
                sleep_time = (next_candle - now).total_seconds()
                time.sleep(max(1, sleep_time))
            except Exception as e:
                logger.error(f"❌ [Signal Scanning] Error in thread: {e}")
                time.sleep(30)
    
    thread = Thread(target=signal_scanning_worker, daemon=True)
    thread.start()
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
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; }
        .card { border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .card-header { background-color: #0d6efd; color: white; border-radius: 10px 10px 0 0 !important; font-weight: bold; }
        .status-active { color: #198754; }
        .status-inactive { color: #dc3545; }
        .market-state-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 5px; }
        .trending { background-color: #198754; } .ranging { background-color: #ffc107; } .volatile { background-color: #dc3545; } .unknown { background-color: #6c757d; }
        .volatility-low { background-color: #198754; } .volatility-medium { background-color: #ffc107; } .volatility-high { background-color: #dc3545; }
        .trend-indicator { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }
        .trend-bullish { background-color: #198754; } .trend-bearish { background-color: #dc3545; } .trend-neutral { background-color: #6c757d; }
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <h3 class="mb-4">لوحة تحكم البوت التداولي</h3>
        <div class="row">
            <!-- Main Column -->
            <div class="col-lg-8">
                <!-- Status & Main Controls Card -->
                <div class="card">
                    <div class="card-body">
                        <div class="row text-center">
                            <div class="col-md-3 col-6 mb-3 mb-md-0">
                                <h5>حالة التداول</h5>
                                <h4 id="trading-status" class="fw-bold status-inactive">غير مفعل</h4>
                                <button id="toggle-trading" class="btn btn-primary btn-sm mt-2">تفعيل</button>
                            </div>
                            <div class="col-md-3 col-6 mb-3 mb-md-0">
                                <h5>نوع التداول</h5>
                                <h4 id="trading-mode" class="fw-bold">ورقي</h4>
                                <button id="toggle-mode" class="btn btn-outline-secondary btn-sm mt-2">تبديل</button>
                            </div>
                            <div class="col-md-3 col-6">
                                <h5>الصفقات المفتوحة</h5>
                                <h4><span id="open-trades-count">0</span> / <span id="max-trades">3</span></h4>
                            </div>
                            <div class="col-md-3 col-6">
                                <h5>الرصيد</h5>
                                <h4 id="balance">$0.00</h4>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Open Trades Card -->
                <div class="card">
                    <div class="card-header">الصفقات المفتوحة</div>
                    <div class="card-body p-0">
                        <div class="table-responsive">
                            <table class="table table-hover mb-0">
                                <thead>
                                    <tr>
                                        <th>العملة</th><th>الدخول</th><th>الهدف</th><th>الوقف</th><th>ربح/خسارة</th><th></th>
                                    </tr>
                                </thead>
                                <tbody id="open-trades-table">
                                    <tr><td colspan="6" class="text-center p-4 text-muted">لا توجد صفقات مفتوحة</td></tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                 <!-- Rejection Log Card -->
                <div class="card">
                    <div class="card-header">سجل الرفض</div>
                    <div class="card-body p-0" style="max-height: 250px; overflow-y: auto;">
                        <table class="table table-sm table-striped mb-0">
                            <tbody id="rejection-table">
                                <tr><td class="text-center p-3 text-muted">لا توجد سجلات رفض</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- Side Column -->
            <div class="col-lg-4">
                <!-- Market State Card -->
                <div class="card">
                    <div class="card-header">حالة السوق</div>
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span><strong>نظام السوق:</strong> <span id="market-regime">...</span></span>
                            <span id="market-regime-indicator" class="market-state-indicator"></span>
                        </div>
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <span><strong>مستوى التقلب:</strong> <span id="volatility-state">...</span></span>
                            <span id="volatility-indicator" class="market-state-indicator"></span>
                        </div>
                        <hr>
                        <div id="trend-lights-container"></div>
                    </div>
                </div>

                <!-- Settings Card -->
                <div class="card">
                    <div class="card-header">الإعدادات</div>
                    <div class="card-body">
                        <form id="settings-form">
                            <!-- Trade Amount -->
                            <div class="mb-2">
                                <label class="form-label small">مبلغ الصفقة (USDT)</label>
                                <div class="input-group input-group-sm">
                                    <input type="number" class="form-control" id="min-amount" step="0.1">
                                    <span class="input-group-text">-</span>
                                    <input type="number" class="form-control" id="max-amount" step="0.1">
                                </div>
                            </div>
                            <!-- Max Trades & Min Quality -->
                            <div class="row mb-3">
                                <div class="col">
                                    <label class="form-label small">أقصى عدد صفقات</label>
                                    <input type="number" class="form-control form-control-sm" id="max-trades-setting" min="1">
                                </div>
                                <div class="col">
                                    <label class="form-label small">أدنى جودة إشارة</label>
                                    <input type="number" class="form-control form-control-sm" id="min-quality" min="1" max="100">
                                </div>
                            </div>
                            <!-- Strategies -->
                            <div class="mb-3">
                                <a class="small" data-bs-toggle="collapse" href="#strategiesCollapse" role="button">الاستراتيجيات ▼</a>
                                <div class="collapse" id="strategiesCollapse">
                                    <div class="p-2 border rounded small" style="max-height: 150px; overflow-y: auto;">
                                        <!-- Checkboxes will be inserted by JS -->
                                        <div id="strategies-container"></div>
                                    </div>
                                </div>
                            </div>
                            <button type="submit" class="btn btn-primary btn-sm w-100">حفظ الإعدادات</button>
                        </form>
                    </div>
                </div>

                <!-- Notifications Card -->
                <div class="card">
                    <div class="card-header">الإشعارات</div>
                    <div id="notifications-container" class="card-body p-0" style="max-height: 300px; overflow-y: auto;">
                         <div class="p-3 text-center text-muted">لا توجد إشعارات</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        const STRATEGY_DEFINITIONS = {
            'USE_BB_STOCH_STRATEGY': 'BB + Stoch',
            'USE_MACD_EMA_STRATEGY': 'MACD + EMA',
            'USE_EMA_RSI_STRATEGY': 'EMA + RSI',
            'USE_PULLBACK_STRATEGY': 'Pullback',
            'USE_MOMENTUM_VOLATILITY_STRATEGY': 'Momentum',
            'USE_ELLIOTT_WAVE_STRATEGY': 'Elliott Wave',
            'USE_RANGE_REVERSAL_STRATEGY': 'Range Reversal'
        };

        function connectWebSocket() {
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
            const socket = new WebSocket(wsUrl);

            socket.onopen = () => console.log("WebSocket connected");
            socket.onmessage = (event) => handleWebSocketMessage(JSON.parse(event.data));
            socket.onclose = () => setTimeout(connectWebSocket, 5000);
            socket.onerror = (error) => {
                console.error("WebSocket error:", error);
                socket.close();
            };
        }

        let openTrades = {};

        function loadInitialData() {
            fetch('/api/initial-data')
                .then(res => res.ok ? res.json() : Promise.reject(`HTTP error ${res.status}`))
                .then(data => {
                    if (data.error) return console.error('API Error:', data.error);
                    updateDashboard(data);
                })
                .catch(err => console.error('Failed to load initial data:', err));
        }

        function handleWebSocketMessage(data) {
            // A simple and robust way to handle updates is to reload all data.
            // This prevents the UI from getting out of sync.
            if (['new_notification', 'new_rejection', 'trade_opened', 'trade_updated', 'trade_closed'].includes(data.type)) {
                loadInitialData();
            } else if (data.type === 'price_update') {
                updatePrices(data.payload);
            } else if (data.type === 'market_state_update') {
                updateMarketState(data.payload);
            }
        }
        
        function updateDashboard(data) {
            updateTradingStatus(data.trading_enabled);
            updateTradingMode(data.paper_trading_mode);
            updateBalance(data.balance);
            updateSettings(data.settings);

            openTrades = (data.open_trades || []).reduce((acc, trade) => {
                acc[trade.symbol] = trade;
                return acc;
            }, {});
            updateOpenTradesTable();

            updateNotificationsContainer(data.notifications || []);
            updateRejectionTable(data.rejections || []);
            updateMarketState(data.market_state || {});
        }

        function updateTradingStatus(enabled) {
            const statusEl = document.getElementById('trading-status');
            const buttonEl = document.getElementById('toggle-trading');
            statusEl.textContent = enabled ? 'مفعل' : 'غير مفعل';
            statusEl.className = `fw-bold ${enabled ? 'status-active' : 'status-inactive'}`;
            buttonEl.textContent = enabled ? 'إيقاف' : 'تفعيل';
            buttonEl.className = `btn btn-sm mt-2 ${enabled ? 'btn-danger' : 'btn-primary'}`;
        }

        function updateTradingMode(isPaper) {
            document.getElementById('trading-mode').textContent = isPaper ? 'ورقي' : 'حقيقي';
            document.getElementById('toggle-mode').textContent = isPaper ? 'تبديل لحقيقي' : 'تبديل لورقي';
        }
        
        function updateBalance(balance) {
            document.getElementById('balance').textContent = `$${(balance || 0).toFixed(2)}`;
        }

        function updateSettings(settings) {
            if (!settings) return;
            document.getElementById('min-amount').value = settings.FIXED_TRADE_AMOUNT_MIN_USDT;
            document.getElementById('max-amount').value = settings.FIXED_TRADE_AMOUNT_MAX_USDT;
            document.getElementById('max-trades-setting').value = settings.MAX_OPEN_TRADES;
            document.getElementById('min-quality').value = settings.MIN_SIGNAL_QUALITY;
            document.getElementById('max-trades').textContent = settings.MAX_OPEN_TRADES;
            
            const container = document.getElementById('strategies-container');
            container.innerHTML = Object.entries(STRATEGY_DEFINITIONS).map(([key, name]) => `
                <div class="form-check">
                    <input class="form-check-input" type="checkbox" id="${key}" ${settings[key] ? 'checked' : ''}>
                    <label class="form-check-label" for="${key}">${name}</label>
                </div>
            `).join('');
        }
        
        function updateMarketState(state) {
            const texts = {
                regime: { trending: 'موجه', ranging: 'جانبي', volatile: 'متقلب', unknown: 'غير معروف' },
                volatility: { low: 'منخفض', medium: 'متوسط', high: 'مرتفع' },
                trend: { bullish: 'صاعد', bearish: 'هابط', neutral: 'محايد' }
            };
            document.getElementById('market-regime').textContent = texts.regime[state.market_regime] || '...';
            document.getElementById('market-regime-indicator').className = `market-state-indicator ${state.market_regime || 'unknown'}`;
            document.getElementById('volatility-state').textContent = texts.volatility[state.volatility_state] || '...';
            document.getElementById('volatility-indicator').className = `market-state-indicator volatility-${state.volatility_state || 'medium'}`;

            const lightsContainer = document.getElementById('trend-lights-container');
            lightsContainer.innerHTML = (Object.entries(state.trend_details_by_tf || {})).map(([tf, details]) => `
                <div class="d-flex justify-content-between align-items-center small mb-1">
                    <span>${tf}</span>
                    <span>${texts.trend[details.trend]} (ADX: ${details.adx.toFixed(1)})</span>
                    <span class="trend-indicator trend-${details.trend}"></span>
                </div>
            `).join('');
        }

        function updateOpenTradesTable() {
            const tableBody = document.getElementById('open-trades-table');
            const trades = Object.values(openTrades);
            document.getElementById('open-trades-count').textContent = trades.length;

            if (trades.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="6" class="text-center p-4 text-muted">لا توجد صفقات مفتوحة</td></tr>';
                return;
            }
            tableBody.innerHTML = trades.map(trade => {
                const price = trade.current_price || trade.entry_price;
                const pnl = ((price - trade.entry_price) / trade.entry_price) * 100;
                return `
                    <tr>
                        <td><strong>${trade.symbol}</strong><br><small class="text-muted">${STRATEGY_DEFINITIONS[trade.strategy_name.replace('Strategy','STRATEGY').replace('_','_USE_')+'_STRATEGY'] || trade.strategy_name}</small></td>
                        <td>${trade.entry_price.toFixed(4)}</td>
                        <td>${trade.target_price_2.toFixed(4)}</td>
                        <td>${trade.stop_loss.toFixed(4)}</td>
                        <td class="fw-bold ${pnl >= 0 ? 'text-success' : 'text-danger'}">${pnl.toFixed(2)}%</td>
                        <td><button class="btn btn-danger btn-sm" onclick="closeTrade('${trade.symbol}')"><i class="bi bi-x-lg"></i></button></td>
                    </tr>
                `;
            }).join('');
        }

        function updateNotificationsContainer(notifications) {
            const container = document.getElementById('notifications-container');
            const icons = { info: 'bi-info-circle', warning: 'bi-exclamation-triangle', error: 'bi-x-circle', success: 'bi-check-circle', system: 'bi-gear' };
            const colors = { info: 'text-info', warning: 'text-warning', error: 'text-danger', success: 'text-success', system: 'text-primary' };
            container.innerHTML = notifications.length === 0 ? '<div class="p-3 text-center text-muted">لا توجد إشعارات</div>' : notifications.map(n => `
                <div class="d-flex align-items-center p-2 border-bottom">
                    <i class="bi ${icons[n.type] || 'bi-bell'} ${colors[n.type] || ''} me-2"></i>
                    <small>${n.message}<br><span class="text-muted">${new Date(n.timestamp).toLocaleTimeString('ar-EG')}</span></small>
                </div>
            `).join('');
        }
        
        function updateRejectionTable(rejections) {
            const tableBody = document.getElementById('rejection-table');
            tableBody.innerHTML = rejections.length === 0 ? '<tr><td class="text-center p-3 text-muted">لا توجد سجلات رفض</td></tr>' : rejections.map(r => `
                <tr><td><small>${new Date(r.timestamp).toLocaleTimeString('ar-EG')} <strong>${r.symbol}</strong>: ${r.reason}</small></td></tr>
            `).join('');
        }

        function updatePrices(prices) {
            if (Object.values(openTrades).length === 0) return;
            Object.keys(openTrades).forEach(symbol => {
                if (prices[symbol]) openTrades[symbol].current_price = prices[symbol];
            });
            updateOpenTradesTable();
        }

        function closeTrade(symbol) {
            if (!confirm(`هل أنت متأكد من رغبتك في إغلاق صفقة ${symbol} يدويًا؟`)) return;
            fetch(`/api/close-trade?symbol=${symbol}`, { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        alert('تم إرسال طلب إغلاق الصفقة.');
                        delete openTrades[symbol];
                        updateOpenTradesTable();
                    } else {
                        alert('فشل إغلاق الصفقة: ' + data.error);
                    }
                }).catch(err => console.error('Close trade error:', err));
        }

        document.getElementById('toggle-trading').addEventListener('click', () => {
            const isEnabled = document.getElementById('trading-status').classList.contains('status-active');
            fetch('/api/toggle-trading', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled: !isEnabled })
            }).then(res => res.json()).then(data => data.success && updateTradingStatus(data.enabled));
        });

        document.getElementById('toggle-mode').addEventListener('click', () => {
            const isPaper = document.getElementById('trading-mode').textContent === 'ورقي';
            fetch('/api/toggle-mode', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ paper_mode: !isPaper })
            }).then(res => res.json()).then(data => data.success && updateTradingMode(data.paper_mode));
        });

        document.getElementById('settings-form').addEventListener('submit', function(e) {
            e.preventDefault();
            const settings = {
                FIXED_TRADE_AMOUNT_MIN_USDT: parseFloat(document.getElementById('min-amount').value),
                FIXED_TRADE_AMOUNT_MAX_USDT: parseFloat(document.getElementById('max-amount').value),
                MAX_OPEN_TRADES: parseInt(document.getElementById('max-trades-setting').value),
                MIN_SIGNAL_QUALITY: parseInt(document.getElementById('min-quality').value)
            };
            Object.keys(STRATEGY_DEFINITIONS).forEach(key => {
                settings[key] = document.getElementById(key).checked;
            });

            fetch('/api/save-settings', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            }).then(res => res.json()).then(data => {
                if(data.success) {
                    alert('تم حفظ الإعدادات بنجاح.');
                    document.getElementById('max-trades').textContent = settings.MAX_OPEN_TRADES;
                } else alert('فشل حفظ الإعدادات: ' + data.error);
            });
        });

        document.addEventListener('DOMContentLoaded', () => {
            loadInitialData();
            connectWebSocket();
        });
    </script>
</body>
</html>
    ''')

@app.route('/api/initial-data')
def get_initial_data():
    """الحصول على البيانات الأولية للوحة التحكم"""
    try:
        with trading_status_lock: trading_enabled = is_trading_enabled
        with trading_mode_lock: current_paper_trading_mode = paper_trading_mode
        with balance_lock: balance = usdt_balance
        with trade_amount_lock:
            min_amount = FIXED_TRADE_AMOUNT_MIN_USDT
            max_amount = FIXED_TRADE_AMOUNT_MAX_USDT
        with min_quality_lock: min_quality = MIN_SIGNAL_QUALITY
        with signal_cache_lock: open_trades = list(open_signals_cache.values())
        with notifications_lock: notifications = list(notifications_cache)
        with rejection_logs_lock: rejections = list(rejection_logs_cache)
        with market_state_lock: market_state = dict(current_market_state)
        
        settings = {
            'FIXED_TRADE_AMOUNT_MIN_USDT': min_amount,
            'FIXED_TRADE_AMOUNT_MAX_USDT': max_amount,
            'MAX_OPEN_TRADES': MAX_OPEN_TRADES,
            'MIN_SIGNAL_QUALITY': min_quality,
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
            'paper_trading_mode': current_paper_trading_mode,
            'balance': balance,
            'settings': settings,
            'open_trades': open_trades,
            'notifications': notifications,
            'rejections': rejections,
            'market_state': market_state
        })
    except Exception as e:
        logger.error(f"❌ [API] Error getting initial data: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/toggle-trading', methods=['POST'])
def toggle_trading():
    """تبديل حالة التداول"""
    global is_trading_enabled
    try:
        data = request.get_json()
        enabled = data.get('enabled', not is_trading_enabled)
        with trading_status_lock: is_trading_enabled = enabled
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
        with trading_mode_lock: paper_trading_mode = paper_mode
        save_settings_to_redis()
        log_and_notify('info', f"🔄 نوع التداول تغير إلى {'ورقي' if paper_mode else 'حقيقي'}", 'system')
        return jsonify({'success': True, 'paper_mode': paper_mode})
    except Exception as e:
        logger.error(f"❌ [API] Error toggling trading mode: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/save-settings', methods=['POST'])
def save_settings():
    """حفظ الإعدادات"""
    try:
        data = request.get_json()
        with trade_amount_lock:
            globals()['FIXED_TRADE_AMOUNT_MIN_USDT'] = float(data['FIXED_TRADE_AMOUNT_MIN_USDT'])
            globals()['FIXED_TRADE_AMOUNT_MAX_USDT'] = float(data['FIXED_TRADE_AMOUNT_MAX_USDT'])
        globals()['MAX_OPEN_TRADES'] = int(data['MAX_OPEN_TRADES'])
        with min_quality_lock:
            globals()['MIN_SIGNAL_QUALITY'] = int(data['MIN_SIGNAL_QUALITY'])
        
        for key in STRATEGY_NAMES.keys():
            globals()[f"USE_{key.upper()}"] = bool(data.get(f"USE_{key.upper()}", False))

        save_settings_to_redis()
        log_and_notify('info', "🔄 تم تحديث الإعدادات من لوحة التحكم", 'system')
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"❌ [API] Error saving settings: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/close-trade', methods=['POST'])
def close_trade_api():
    """إغلاق صفقة"""
    try:
        symbol = request.args.get('symbol')
        if not symbol: return jsonify({'success': False, 'error': 'Symbol is required'}), 400
        
        with signal_cache_lock:
            if symbol not in open_signals_cache:
                return jsonify({'success': False, 'error': 'Trade not found'}), 404

        with live_prices_lock:
            current_price = live_prices.get(symbol)
        
        if not current_price:
            try:
                current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
            except Exception as e:
                return jsonify({'success': False, 'error': f'Could not fetch price: {e}'}), 400

        close_trade(symbol, 'manual_close', current_price)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"❌ [API] Error closing trade: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@sock.route('/ws')
def websocket_connection(ws):
    """معالجة اتصالات WebSocket"""
    with ws_clients_lock:
        ws_clients.append(ws)
    try:
        while ws.connected:
            # Keep the connection alive
            time.sleep(1)
    except Exception as e:
        logger.warning(f"WebSocket connection closed: {e}")
    finally:
        with ws_clients_lock:
            if ws in ws_clients: ws_clients.remove(ws)

# --- دالة رئيسية ومنطق التشغيل ---
def start_bot_logic():
    """دالة لتهيئة وبدء جميع عمليات البوت الخلفية"""
    global client, validated_symbols_to_scan
    try:
        client = Client(API_KEY, API_SECRET)
        init_db()
        init_redis()
        load_settings_from_redis()
        get_exchange_info_map()
        validated_symbols_to_scan = get_validated_symbols()
        load_open_signals_to_cache()
        load_notifications_to_cache()
        start_websocket()
        start_trade_analysis_thread()
        start_signal_scanning_thread()
        start_periodic_reports()
        logger.info("✅ Bot logic is fully initialized and running in the background.")
    except Exception as e:
        logger.critical(f"❌ [Main] A critical error occurred during initialization: {e}", exc_info=True)
        exit(1)

if __name__ == '__main__':
    # بدء منطق البوت في خيط منفصل
    bot_thread = Thread(target=start_bot_logic)
    bot_thread.daemon = True
    bot_thread.start()
    
    # تشغيل تطبيق الويب (لوحة التحكم)
    logger.info("🚀 Starting Flask web server for the dashboard...")
    # The default Flask server is fine for this kind of single-user dashboard application.
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
