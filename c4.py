# ملف c4_5min_v34_0_5.py - نسخة V34.0.5 (تحسين مرونة فيبوناتشي)
# --- وصف التعديلات:
# 1. [توسيع نطاق فيبوناتشي] تم تعديل الفلتر الديناميكي الخاص باستراتيجية موجات إليوت ليقبل نطاقًا أوسع من تصحيحات فيبوناتشي، مما يقلل من الصرامة ويزيد من فرص الدخول.
# 2. [عرض الرصيد الفعلي] يحتفظ البوت بميزة عرض رصيد USDT الحقيقي دائمًا.
# 3. [صفقات ورقية ثابتة] تظل الصفقات الورقية تستخدم قيمة ثابتة قدرها 10 USDT.
# 4. [إصلاح خطأ قاعدة البيانات] تم إصلاح خطأ "column s.created_at does not exist" عن طريق إضافة العمود المفقود تلقائيًا وتحسين معالجة أخطاء المعاملات.
# 5. [تحسين جودة التوصيات] تضييق نطاق فيبوناتشي، تشديد فلاتر حجم التداول، تحسين نسب المخاطرة، زيادة متطلبات ADX، إضافة فلتر التقلبات العشوائية، تحسين إدارة المخاطر، تحسين فلتر السيولة والارتباط.

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
logger = logging.getLogger('CryptoBotV34.0.5_5min')

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
    "Random Volatility Filter Failed": "فلتر التقلبات العشوائية: تجنب التداول في فترات التقلبات غير الطبيعية",

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

# تحديث: تضييق نطاق فيبوناتشي لاستراتيجية موجات إليوت
def check_elliott_wave_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    
    # تضييق نطاق فيبوناتشي لزيادة الدقة
    if atr_percent > 2.5:  # سوق متقلب
        fib_min, fib_max = 0.236, 0.786  # نطاق أضيق (كان 0.18, 0.94)
    else:  # سوق عادي
        fib_min, fib_max = 0.236, 0.618  # نطاق أضيق (كان 0.18, 0.886)
    
    volume_ma = df['volume'].rolling(20).mean()
    wave_volume_multiplier = 1.3 + (atr_percent / 50)
    
    macd_momentum = df['macd_hist'].rolling(5).mean()
    momentum_threshold = macd_momentum.rolling(20).std() * 0.3
    
    return {
        'fibonacci_ok': fib_min <= get_wave_retracement(df) <= fib_max,
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * wave_volume_multiplier,
        'momentum_ok': macd_momentum.iloc[-1] > momentum_threshold.iloc[-1],
    }

# تحديث: تشديد فلاتر حجم التداول
def check_bb_stoch_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    
    bb_width = df['bb_width']
    dynamic_bb_threshold = bb_width.rolling(20).mean() * 1.2

    stoch_threshold = 23 if atr_percent > 1.5 else 18 # Adjusted for 5m
    
    # زيادة متطلبات حجم التداول
    volume_ma = df['volume'].rolling(20).mean()
    volume_multiplier = 1.2 + (atr_percent / 80)  # كان 1.0 + (atr_percent / 100)
    
    return {
        'bb_width_ok': bb_width.iloc[-1] > dynamic_bb_threshold.iloc[-1],
        'stoch_ok': last_row['stoch_k'] > stoch_threshold,
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * volume_multiplier
    }

# تحديث: زيادة متطلبات ADX لقوة الاتجاه
def check_macd_ema_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    
    # زيادة متطلبات ADX
    default_adx_thresh = 22 if atr_percent > 1.5 else 18  # كانت 20 و 16
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
    dynamic_adx_threshold = adx_ma.iloc[-1] * 0.85
    
    return {
        'volatility_ok': dynamic_vol_min <= atr_percent.iloc[-1] <= dynamic_vol_max,
        'momentum_ok': is_momentum_ok,
        'adx_ok': last_row['adx'] > dynamic_adx_threshold,
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

# تحديث: إضافة فلتر زمني لتجنب فترات السيولة المنخفضة
def add_liquidity_filter() -> bool:
    now = datetime.now(timezone.utc)
    
    # تجنب عطلة نهاية الأسبوع
    if now.weekday() >= 5: 
        return False
    
    # تجنب ساعات السيولة المنخفضة (تم توسيع النافذة)
    if now.hour >= 21 or now.hour <= 3: 
        return False
    
    # تجنب فترة افتتاح سوق نيويورك (تقلبات عالية)
    if now.hour == 12 and (30 <= now.minute <= 45):
        return False
    
    return True

# تحديث: تحسين فلتر الارتباط بين العملات
def add_correlation_filter(new_symbol: str) -> bool:
    # توسيع مجموعات العملات المرتبطة
    correlated_groups = [
        {'BTCUSDT', 'ETHUSDT', 'BCHUSDT', 'LTCUSDT'}, 
        {'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'ATOMUSDT'},
        {'SOLUSDT', 'AVAXUSDT', 'MATICUSDT', 'NEARUSDT'},
        {'XRPUSDT', 'XLMUSDT', 'ALGOUSDT'},
    ]
    
    with signal_cache_lock: 
        open_symbols = set(open_signals_cache.keys())
    
    if not open_symbols: 
        return True
    
    for group in correlated_groups:
        if new_symbol in group and not open_symbols.isdisjoint(group):
            return False
    
    return True

# تحديث: إضافة فلتر جديد لتجنب التداول في فترات التقلبات العشوائية
def add_random_volatility_filter(df: pd.DataFrame) -> bool:
    """
    فلتر لتجنب التداول في فترات التقلبات العشوائية بدون اتجاه واضح
    """
    last_row = df.iloc[-1]
    
    # حساب تغير السعر خلال آخر 5 شموع
    price_change = abs(df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6]
    
    # حساب متوسط التغير خلال 20 شمعة سابقة
    avg_price_change = df['close'].pct_change().abs().rolling(20).mean().iloc[-1]
    
    # إذا كان التغير الحالي أعلى بكثير من المتوسط، قد يكون تقلباً عشوائياً
    if price_change > avg_price_change * 3:
        return False
    
    return True

def check_market_volatility_filter_enhanced(df: pd.DataFrame, symbol: str = "Unknown") -> bool:
    if 'atr_percent' not in df.columns or df['atr_percent'].isnull().all():
        log_rejection(symbol, "Market Volatility Filter Failed", {"reason": "No ATR data"})
        return False
    
    last_atr_percent = float(df.iloc[-1].get('atr_percent', 0))
    # Adjusted for 5-minute timeframe
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
# تحديث: تحسين إدارة المخاطر
def calculate_dynamic_stop_loss_enhanced(df: pd.DataFrame, entry_price: float, strategy_name: str) -> float:
    last = df.iloc[-1]
    atr_value = last.get('atr', 0)
    atr_percent = last.get('atr_percent', 0)
    
    # زيادة مسافة وقف الخسارة في الأسواق المتقلبة
    if atr_percent > 2.5:  # سوق شديد التقلب
        atr_multiplier = 3.0  # كان 2.5
    elif atr_percent > 1.5:  # سوق متقلب
        atr_multiplier = 2.3  # كان 2.0
    elif atr_percent > 1.0:  # سوق متوسط التقلب
        atr_multiplier = 1.8  # كان 1.5
    else:  # سوق منخفض التقلب
        atr_multiplier = 1.5
    
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

# تحديث: تحسين نسب المخاطرة إلى المكافأة
def calculate_dynamic_take_profit_enhanced(df: pd.DataFrame, entry_price: float, stop_loss: float, strategy_name: str) -> tuple:
    risk_amount = entry_price - stop_loss
    if risk_amount <= 0: return (entry_price * 1.015, entry_price * 1.025) # Default for 5m

    last = df.iloc[-1]
    atr_percent = last.get('atr_percent', 0)
    
    # تعديل نسبة المخاطرة إلى المكافأة بناءً على تقلب السوق
    if atr_percent > 2.0:  # سوق متقلب
        volatility_adjustment = 0.8
    elif atr_percent > 1.0:  # سوق متوسط التقلب
        volatility_adjustment = 1.0
    else:  # سوق منخفض التقلب
        volatility_adjustment = 1.2
    
    # تعديل نسبة المخاطرة إلى المكافأة بناءً على حجم التداول
    volume_ma = df['volume'].rolling(20).mean()
    volume_ratio = last['volume'] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1
    
    if volume_ratio > 2.0:  # حجم تداول عالي
        volume_adjustment = 1.2
    elif volume_ratio < 0.5:  # حجم تداول منخفض
        volume_adjustment = 0.8
    else:
        volume_adjustment = 1.0
    
    # حساب نسبة المخاطرة إلى المكافأة النهائية
    adjustment_factor = volatility_adjustment * volume_adjustment
    
    # Risk-Reward Ratios adjusted for 5m timeframe (Scalping)
    if strategy_name == "BB_Stoch_Strategy": 
        rr1, rr2 = 2.0, 3.5  # كانت 1.8, 3.0
    elif strategy_name == "MACD_EMA_Strategy": 
        rr1, rr2 = 1.8, 3.2  # كانت 1.6, 2.8
    elif strategy_name == "EMA_RSI_Strategy": 
        rr1, rr2 = 1.9, 3.5  # كانت 1.7, 3.0
    elif strategy_name == "Pullback_Strategy": 
        rr1, rr2 = 2.0, 3.8  # كانت 1.8, 3.2
    elif strategy_name == "Momentum_Volatility_Strategy": 
        rr1, rr2 = 1.7, 3.0  # كانت 1.5, 2.5
    elif strategy_name == "Elliott_Wave_Strategy": 
        rr1, rr2 = 2.2, 4.0  # كانت 2.0, 3.5
    elif strategy_name == "Range_Reversal_Strategy":
        middle_band = df.iloc[-1].get('bb_middle', entry_price * 1.015)
        upper_band = df.iloc[-1].get('bb_upper', entry_price * 1.03)
        return middle_band, upper_band
    else: 
        rr1, rr2 = 1.6, 2.8
        
    target1 = entry_price + (risk_amount * rr1)
    target2 = entry_price + (risk_amount * rr2)
    
    return target1, target2

# --- استراتيجيات التداول المحسنة ---
def check_ema_rsi_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 200: return False

    last = df.iloc[-1]
    if not (last['ema50'] > last['ema200'] and last['close'] > last['ema9']):
        return False
        
    filters = check_ema_rsi_dynamic_filters(df)
    if not filters.get('rsi_ok', False):
        log_rejection(symbol_name, "DYN_RSI_OOR")
        return False
    if not filters.get('ema_ok', False):
        log_rejection(symbol_name, "DYN_EMA_SPREAD_LOW")
        return False
    if not filters.get('volume_ok', False):
        log_rejection(symbol_name, "DYN_VOLUME_LOW")
        return False

    # إضافة فلتر التقلبات العشوائية
    if not add_random_volatility_filter(df):
        log_rejection(symbol_name, "Random Volatility Filter Failed")
        return False

    return True

def check_bb_stoch_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 50: return False

    last = df.iloc[-1]
    if last['close'] < last['ema50']:
        log_rejection(symbol_name, "BB: Price below EMA50 (bearish trend)")
        return False

    filters = check_bb_stoch_dynamic_filters(df)
    if not filters.get('bb_width_ok', False):
        log_rejection(symbol_name, "DYN_BB_WIDTH_LOW")
        return False
    if not filters.get('stoch_ok', False):
        log_rejection(symbol_name, "DYN_STOCH_LOW")
        return False
    if not filters.get('volume_ok', False):
        log_rejection(symbol_name, "DYN_VOLUME_LOW")
        return False

    # إضافة فلتر التقلبات العشوائية
    if not add_random_volatility_filter(df):
        log_rejection(symbol_name, "Random Volatility Filter Failed")
        return False

    return True

def check_macd_ema_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 100: return False

    last = df.iloc[-1]
    if last['macd'] < last['macd_signal'] or last['macd_hist'] < 0:
        log_rejection(symbol_name, "MACD Momentum Negative")
        return False

    filters = check_macd_ema_dynamic_filters(df)
    if not filters.get('adx_ok', False):
        log_rejection(symbol_name, "DYN_ADX_LOW")
        return False
    if not filters.get('volume_ok', False):
        log_rejection(symbol_name, "DYN_VOLUME_LOW")
        return False
    if not filters.get('momentum_ok', False):
        log_rejection(symbol_name, "DYN_MACD_MOMENTUM_LOW")
        return False

    # إضافة فلتر التقلبات العشوائية
    if not add_random_volatility_filter(df):
        log_rejection(symbol_name, "Random Volatility Filter Failed")
        return False

    return True

def check_pullback_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 100: return False

    if mtf_trend.get('5m') != 'bullish' or mtf_trend.get('15m') != 'bullish':
        log_rejection(symbol_name, "Pullback: Trend is not strongly bullish")
        return False

    filters = check_pullback_dynamic_filters(df, mtf_trend)
    if not filters.get('recovery_ok', False):
        log_rejection(symbol_name, "DYN_RECOVERY_FAIL")
        return False
    if not filters.get('volume_ok', False):
        log_rejection(symbol_name, "DYN_VOLUME_LOW")
        return False

    # إضافة فلتر التقلبات العشوائية
    if not add_random_volatility_filter(df):
        log_rejection(symbol_name, "Random Volatility Filter Failed")
        return False

    return True

def check_momentum_volatility_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 100: return False

    last = df.iloc[-1]
    if not (last['ema9'] > last['ema21'] > last['ema34']):
        log_rejection(symbol_name, "Momentum: EMAs not in bullish order")
        return False

    filters = check_momentum_volatility_dynamic_filters(df)
    if not filters.get('volatility_ok', False):
        log_rejection(symbol_name, "DYN_VOLATILITY_OOR")
        return False
    if not filters.get('momentum_ok', False):
        log_rejection(symbol_name, "DYN_MOMENTUM_SCORE_LOW")
        return False
    if not filters.get('adx_ok', False):
        log_rejection(symbol_name, "DYN_ADX_LOW")
        return False

    # إضافة فلتر التقلبات العشوائية
    if not add_random_volatility_filter(df):
        log_rejection(symbol_name, "Random Volatility Filter Failed")
        return False

    return True

def check_elliott_wave_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 100: return False

    if mtf_trend.get('5m') != 'bullish':
        log_rejection(symbol_name, "Elliott Wave: Strongly bearish trend")
        return False

    filters = check_elliott_wave_dynamic_filters(df)
    if not filters.get('fibonacci_ok', False):
        log_rejection(symbol_name, "DYN_FIB_RETRACEMENT_OOR")
        return False
    if not filters.get('volume_ok', False):
        log_rejection(symbol_name, "DYN_VOLUME_LOW")
        return False
    if not filters.get('momentum_ok', False):
        log_rejection(symbol_name, "DYN_MACD_MOMENTUM_LOW")
        return False

    # إضافة فلتر التقلبات العشوائية
    if not add_random_volatility_filter(df):
        log_rejection(symbol_name, "Random Volatility Filter Failed")
        return False

    return True

def check_range_reversal_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 100: return False

    filters = check_range_reversal_dynamic_filters(df)
    if not filters.get('adx_ok', False):
        log_rejection(symbol_name, "Range Reversal: Trend too strong (ADX > 23)")
        return False
    if not filters.get('rsi_ok', False):
        log_rejection(symbol_name, "Range Reversal: RSI not in oversold zone")
        return False

    # إضافة فلتر التقلبات العشوائية
    if not add_random_volatility_filter(df):
        log_rejection(symbol_name, "Random Volatility Filter Failed")
        return False

    return True

# --- دوال المساعدة للتحقق من الإشارات ---
def apply_general_filters(symbol: str, df: pd.DataFrame) -> bool:
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
    if not add_random_volatility_filter(df):
        log_rejection(symbol, "Random Volatility Filter Failed")
        return False
    return True

def get_market_trend_for_timeframes(df_dict: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    trend_results = {}
    for tf, df in df_dict.items():
        if df is None or len(df) < 50:
            trend_results[tf] = 'unknown'
            continue
            
        last = df.iloc[-1]
        if last['ema9'] > last['ema21'] > last['ema50']:
            trend_results[tf] = 'bullish'
        elif last['ema9'] < last['ema21'] < last['ema50']:
            trend_results[tf] = 'bearish'
        else:
            trend_results[tf] = 'neutral'
            
    return trend_results

def analyze_market_state() -> Dict[str, Any]:
    if not validated_symbols_to_scan or not client:
        return {"trend_details_by_tf": {}}
        
    btc_df = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, 2)
    if btc_df is None or len(btc_df) < 50:
        return {"trend_details_by_tf": {}}
        
    btc_df = calculate_all_features(btc_df)
    
    trend_details = {}
    for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
        if tf == SIGNAL_GENERATION_TIMEFRAME:
            df = btc_df
        else:
            df = fetch_historical_data(BTC_SYMBOL, tf, 3)
            if df is None or len(df) < 50:
                continue
            df = calculate_all_features(df)
            
        last = df.iloc[-1]
        trend = 'neutral'
        if last['ema9'] > last['ema21'] > last['ema50']:
            trend = 'bullish'
        elif last['ema9'] < last['ema21'] < last['ema50']:
            trend = 'bearish'
            
        trend_details[tf] = {
            'trend': trend,
            'adx': last.get('adx', 0),
            'rsi': last.get('rsi', 50),
            'atr_percent': last.get('atr_percent', 0)
        }
    
    return {"trend_details_by_tf": trend_details}

def update_market_state():
    global current_market_state
    new_state = analyze_market_state()
    with market_state_lock:
        current_market_state = new_state
    broadcast({"type": "market_state_update", "payload": new_state})

# --- دوال التداول الأساسية ---
def get_trade_amount_usdt() -> float:
    with trading_mode_lock:
        if paper_trading_mode:
            return PAPER_TRADE_FIXED_AMOUNT_USDT
    
    with balance_lock:
        if usdt_balance <= 0:
            logger.warning("USDT balance is 0 or negative. Using minimum trade amount.")
            return FIXED_TRADE_AMOUNT_MIN_USDT
            
        # Use a percentage of balance for real trading, but within min/max limits
        percentage_of_balance = 0.05  # 5% of balance
        amount = usdt_balance * percentage_of_balance
        
        with trade_amount_lock:
            amount = max(FIXED_TRADE_AMOUNT_MIN_USDT, min(amount, FIXED_TRADE_AMOUNT_MAX_USDT))
            
        return amount

def calculate_position_size(symbol: str, entry_price: float, stop_loss: float, usdt_amount: float) -> float:
    if entry_price <= stop_loss:
        log_rejection(symbol, "Invalid Position Size")
        return 0.0
        
    risk_per_unit = entry_price - stop_loss
    max_units = usdt_amount / entry_price
    
    # Calculate position size based on risk (1% of capital)
    risk_amount = usdt_amount * 0.01
    units_by_risk = risk_amount / risk_per_unit
    
    # Take the minimum between max_units and units_by_risk
    position_size = min(max_units, units_by_risk)
    
    # Check against exchange limits
    if symbol in exchange_info_map:
        symbol_info = exchange_info_map[symbol]
        for filter in symbol_info.get('filters', []):
            if filter.get('filterType') == 'LOT_SIZE':
                min_qty = float(filter.get('minQty', 0))
                max_qty = float(filter.get('maxQty', float('inf')))
                step_size = float(filter.get('stepSize', 0))
                
                # Adjust position size to comply with LOT_SIZE rules
                if position_size < min_qty:
                    log_rejection(symbol, "MinNotional Filter Failed", {
                        "requested": position_size,
                        "min": min_qty
                    })
                    return 0.0
                    
                if position_size > max_qty:
                    position_size = max_qty
                    
                # Round down to nearest step size
                position_size = position_size - (position_size % step_size)
                break
    
    return position_size

def execute_trade(symbol: str, strategy_name: str, entry_price: float, stop_loss: float, 
                 target1: float, target2: float, quality_score: int, atr_percent: float) -> bool:
    with trading_mode_lock:
        is_real_trade = not paper_trading_mode
    
    usdt_amount = get_trade_amount_usdt()
    quantity = calculate_position_size(symbol, entry_price, stop_loss, usdt_amount)
    
    if quantity <= 0:
        log_rejection(symbol, "Invalid Position Size")
        return False
    
    notional_value = quantity * entry_price
    
    # Check against min notional value
    if symbol in exchange_info_map:
        symbol_info = exchange_info_map[symbol]
        for filter in symbol_info.get('filters', []):
            if filter.get('filterType') == 'MIN_NOTIONAL':
                min_notional = float(filter.get('minNotional', 0))
                if notional_value < min_notional:
                    log_rejection(symbol, "MinNotional Filter Failed", {
                        "notional": notional_value,
                        "min": min_notional
                    })
                    return False
    
    # For real trading, check balance
    if is_real_trade:
        with balance_lock:
            if usdt_balance < notional_value:
                log_rejection(symbol, "Insufficient Balance", {
                    "balance": usdt_balance,
                    "required": notional_value
                })
                
                # Auto fallback to paper trading if enabled
                if AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE:
                    with trading_mode_lock:
                        paper_trading_mode = True
                        is_real_trade = False
                        logger.warning("Auto-switched to paper trading due to insufficient balance")
                else:
                    return False
    
    # Save signal to database
    if not check_db_connection() or not conn:
        return False
        
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, stop_loss, target_price_1, target_price_2, 
                                    strategy_name, quantity, is_real_trade, signal_details, atr_percent, quality_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                symbol, entry_price, stop_loss, target1, target2,
                strategy_name, quantity, is_real_trade,
                json.dumps({
                    "atr_percent": atr_percent,
                    "quality_score": quality_score,
                    "notional_value": notional_value
                }), atr_percent, quality_score
            ))
            
            signal_id = cur.fetchone()['id']
            conn.commit()
            
            # Update cache
            with signal_cache_lock:
                open_signals_cache[symbol] = {
                    'id': signal_id,
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target_price_1': target1,
                    'target_price_2': target2,
                    'strategy_name': strategy_name,
                    'quantity': quantity,
                    'is_real_trade': is_real_trade,
                    'status': 'open',
                    'created_at': datetime.now(timezone.utc).isoformat()
                }
            
            # Send notification
            send_trade_open_notification(
                symbol, strategy_name, entry_price, stop_loss,
                target1, target2, quantity, is_real_trade,
                quality_score, atr_percent, notional_value
            )
            
            # For real trading, execute the order on Binance
            if is_real_trade:
                try:
                    order = client.create_order(
                        symbol=symbol,
                        side=Client.SIDE_BUY,
                        type=Client.ORDER_TYPE_LIMIT,
                        timeInForce=Client.TIME_IN_FORCE_GTC,
                        quantity=quantity,
                        price=round(entry_price, 8)
                    )
                    
                    # Update signal with order ID
                    with conn.cursor() as cur:
                        cur.execute("UPDATE signals SET order_id = %s WHERE id = %s", 
                                   (order['orderId'], signal_id))
                        conn.commit()
                        
                    # Update balance (simplified)
                    with balance_lock:
                        usdt_balance -= notional_value
                        
                    logger.info(f"✅ Real trade executed for {symbol}: {order}")
                    
                except BinanceAPIException as e:
                    logger.error(f"❌ Failed to execute real trade for {symbol}: {e}")
                    
                    # Mark signal as failed
                    with conn.cursor() as cur:
                        cur.execute("""
                            UPDATE signals 
                            SET status = 'failed', closing_reason = %s, closed_at = NOW()
                            WHERE id = %s
                        """, (str(e), signal_id))
                        conn.commit()
                        
                    # Remove from cache
                    with signal_cache_lock:
                        if symbol in open_signals_cache:
                            del open_signals_cache[symbol]
                            
                    return False
            
            logger.info(f"✅ Signal created for {symbol} with strategy {strategy_name}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error executing trade for {symbol}: {e}")
        if conn: conn.rollback()
        return False

def check_and_generate_signals() -> None:
    if not is_trading_enabled:
        return
        
    if not validated_symbols_to_scan or not client:
        logger.warning("No validated symbols or Binance client not initialized")
        return
        
    with signal_cache_lock:
        open_symbols = set(open_signals_cache.keys())
        
    if len(open_symbols) >= MAX_OPEN_TRADES:
        logger.info(f"Maximum number of open trades ({MAX_OPEN_TRADES}) reached. Skipping signal generation.")
        return
        
    # Get market trend data
    btc_df = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, 2)
    if btc_df is None or len(btc_df) < 50:
        logger.warning("Failed to fetch BTC data for trend analysis")
        return
        
    btc_df = calculate_all_features(btc_df)
    mtf_trend = get_market_trend_for_timeframes({
        SIGNAL_GENERATION_TIMEFRAME: btc_df,
        HIGHER_TIMEFRAME: fetch_historical_data(BTC_SYMBOL, HIGHER_TIMEFRAME, 3)
    })
    
    # Update market state
    update_market_state()
    
    # Process symbols in random order to avoid bias
    symbols_to_process = validated_symbols_to_scan.copy()
    random.shuffle(symbols_to_process)
    
    for symbol in symbols_to_process:
        if len(open_symbols) >= MAX_OPEN_TRADES:
            logger.info(f"Maximum number of open trades ({MAX_OPEN_TRADES}) reached. Stopping signal generation.")
            break
            
        # Skip if we already have an open signal for this symbol
        if symbol in open_symbols:
            continue
            
        # Skip if symbol is in cooldown
        with cooldowns_lock:
            if symbol in cooldowns_by_symbol:
                cooldown_time = cooldowns_by_symbol[symbol]
                if datetime.now(timezone.utc) < cooldown_time:
                    continue
                else:
                    del cooldowns_by_symbol[symbol]
        
        # Fetch historical data
        df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
        if df is None or len(df) < 50:
            continue
            
        # Calculate indicators
        df = calculate_all_features(df)
        df.name = symbol  # Add symbol name for logging
        
        # Apply general filters first
        if not apply_general_filters(symbol, df):
            continue
            
        # Get last row for reference
        last_row = df.iloc[-1]
        atr_percent = last_row.get('atr_percent', 0)
        
        # Check each strategy
        signal_triggered = False
        strategy_name = None
        
        with min_quality_lock:
            min_quality = MIN_SIGNAL_QUALITY
            
        if USE_BB_STOCH_STRATEGY and check_bb_stoch_strategy_enhanced(df, mtf_trend):
            strategy_name = "BB_Stoch_Strategy"
            signal_triggered = True
        elif USE_MACD_EMA_STRATEGY and check_macd_ema_strategy_enhanced(df, mtf_trend):
            strategy_name = "MACD_EMA_Strategy"
            signal_triggered = True
        elif USE_EMA_RSI_STRATEGY and check_ema_rsi_strategy_enhanced(df, mtf_trend):
            strategy_name = "EMA_RSI_Strategy"
            signal_triggered = True
        elif USE_PULLBACK_STRATEGY and check_pullback_strategy_enhanced(df, mtf_trend):
            strategy_name = "Pullback_Strategy"
            signal_triggered = True
        elif USE_MOMENTUM_VOLATILITY_STRATEGY and check_momentum_volatility_strategy_enhanced(df, mtf_trend):
            strategy_name = "Momentum_Volatility_Strategy"
            signal_triggered = True
        elif USE_ELLIOTT_WAVE_STRATEGY and check_elliott_wave_strategy_enhanced(df, mtf_trend):
            strategy_name = "Elliott_Wave_Strategy"
            signal_triggered = True
        elif USE_RANGE_REVERSAL_STRATEGY and check_range_reversal_strategy_enhanced(df, mtf_trend):
            strategy_name = "Range_Reversal_Strategy"
            signal_triggered = True
            
        if signal_triggered and strategy_name:
            # Calculate entry price, stop loss, and take profit levels
            entry_price = last_row['close']
            stop_loss = calculate_dynamic_stop_loss_enhanced(df, entry_price, strategy_name)
            target1, target2 = calculate_dynamic_take_profit_enhanced(df, entry_price, stop_loss, strategy_name)
            
            # Calculate signal quality score (simplified)
            quality_score = min(100, int(50 + (atr_percent * 10) + (last_row['adx'] / 2)))
            
            if quality_score < min_quality:
                log_rejection(symbol, "Low Quality Signal", {
                    "quality": quality_score,
                    "min_required": min_quality
                })
                continue
                
            # Execute the trade
            if execute_trade(symbol, strategy_name, entry_price, stop_loss, target1, target2, quality_score, atr_percent):
                with signal_cache_lock:
                    open_symbols.add(symbol)
                logger.info(f"✅ Signal executed for {symbol} with strategy {strategy_name}")
            else:
                logger.warning(f"❌ Failed to execute signal for {symbol}")
                
        # Small delay between processing symbols to avoid API rate limits
        time.sleep(0.1)

def monitor_open_trades() -> None:
    if not check_db_connection() or not conn:
        return
        
    try:
        with conn.cursor() as cur:
            # Get all open signals
            cur.execute("SELECT * FROM signals WHERE status = 'open'")
            open_signals = cur.fetchall()
            
            for signal in open_signals:
                symbol = signal['symbol']
                signal_id = signal['id']
                entry_price = signal['entry_price']
                stop_loss = signal['stop_loss']
                target1 = signal.get('target_price_1', entry_price * 1.02)
                target2 = signal.get('target_price_2', entry_price * 1.03)
                is_real_trade = signal['is_real_trade']
                
                # Get current price
                with live_prices_lock:
                    current_price = live_prices.get(symbol, entry_price)
                
                # Check if stop loss or take profit is hit
                if current_price <= stop_loss:
                    # Stop loss hit
                    closing_reason = "stop_loss"
                    profit_percentage = ((current_price - entry_price) / entry_price) * 100
                    
                    # Update signal in database
                    cur.execute("""
                        UPDATE signals 
                        SET status = 'closed', closing_price = %s, closed_at = NOW(), 
                            profit_percentage = %s, closing_reason = %s
                        WHERE id = %s
                    """, (current_price, profit_percentage, closing_reason, signal_id))
                    
                    # For real trades, execute sell order
                    if is_real_trade:
                        try:
                            order = client.create_order(
                                symbol=symbol,
                                side=Client.SIDE_SELL,
                                type=Client.ORDER_TYPE_MARKET,
                                quantity=signal['quantity']
                            )
                            
                            # Update balance (simplified)
                            with balance_lock:
                                usdt_balance += signal['quantity'] * current_price
                            
                            logger.info(f"✅ Stop loss executed for {symbol}: {order}")
                            
                        except BinanceAPIException as e:
                            logger.error(f"❌ Failed to execute stop loss for {symbol}: {e}")
                    
                    # Add cooldown
                    with cooldowns_lock:
                        cooldown_time = datetime.now(timezone.utc) + timedelta(minutes=COOLDOWN_MINUTES_AFTER_SL)
                        cooldowns_by_symbol[symbol] = cooldown_time
                    
                    # Update consecutive losses counter
                    if profit_percentage < 0:
                        with consecutive_losses_lock:
                            consecutive_losses_by_symbol[symbol] = consecutive_losses_by_symbol.get(symbol, 0) + 1
                    
                    # Remove from cache
                    with signal_cache_lock:
                        if symbol in open_signals_cache:
                            del open_signals_cache[symbol]
                    
                    logger.info(f"✅ Stop loss hit for {symbol} at {current_price}")
                    
                elif current_price >= target2:
                    # Take profit 2 hit
                    closing_reason = "take_profit_2"
                    profit_percentage = ((current_price - entry_price) / entry_price) * 100
                    
                    # Update signal in database
                    cur.execute("""
                        UPDATE signals 
                        SET status = 'closed', closing_price = %s, closed_at = NOW(), 
                            profit_percentage = %s, closing_reason = %s
                        WHERE id = %s
                    """, (current_price, profit_percentage, closing_reason, signal_id))
                    
                    # For real trades, execute sell order
                    if is_real_trade:
                        try:
                            order = client.create_order(
                                symbol=symbol,
                                side=Client.SIDE_SELL,
                                type=Client.ORDER_TYPE_MARKET,
                                quantity=signal['quantity']
                            )
                            
                            # Update balance (simplified)
                            with balance_lock:
                                usdt_balance += signal['quantity'] * current_price
                            
                            logger.info(f"✅ Take profit 2 executed for {symbol}: {order}")
                            
                        except BinanceAPIException as e:
                            logger.error(f"❌ Failed to execute take profit for {symbol}: {e}")
                    
                    # Reset consecutive losses counter
                    with consecutive_losses_lock:
                        if symbol in consecutive_losses_by_symbol:
                            del consecutive_losses_by_symbol[symbol]
                    
                    # Remove from cache
                    with signal_cache_lock:
                        if symbol in open_signals_cache:
                            del open_signals_cache[symbol]
                    
                    logger.info(f"✅ Take profit 2 hit for {symbol} at {current_price}")
                    
                elif current_price >= target1:
                    # Take profit 1 hit - close half position and move stop loss to breakeven
                    closing_reason = "take_profit_1"
                    profit_percentage = ((current_price - entry_price) / entry_price) * 100
                    
                    # Update signal in database
                    cur.execute("""
                        UPDATE signals 
                        SET status = 'updated', closing_reason = %s
                        WHERE id = %s
                    """, (closing_reason, signal_id))
                    
                    # For real trades, execute sell order for half position
                    if is_real_trade:
                        try:
                            half_quantity = signal['quantity'] / 2
                            order = client.create_order(
                                symbol=symbol,
                                side=Client.SIDE_SELL,
                                type=Client.ORDER_TYPE_MARKET,
                                quantity=half_quantity
                            )
                            
                            # Update balance (simplified)
                            with balance_lock:
                                usdt_balance += half_quantity * current_price
                            
                            logger.info(f"✅ Take profit 1 executed for {symbol} (half position): {order}")
                            
                        except BinanceAPIException as e:
                            logger.error(f"❌ Failed to execute take profit for {symbol}: {e}")
                    
                    # Update stop loss to breakeven
                    new_stop_loss = entry_price
                    cur.execute("""
                        UPDATE signals 
                        SET stop_loss = %s, quantity = %s
                        WHERE id = %s
                    """, (new_stop_loss, signal['quantity'] / 2, signal_id))
                    
                    # Update cache
                    with signal_cache_lock:
                        if symbol in open_signals_cache:
                            open_signals_cache[symbol]['stop_loss'] = new_stop_loss
                            open_signals_cache[symbol]['quantity'] = signal['quantity'] / 2
                            open_signals_cache[symbol]['status'] = 'updated'
                    
                    logger.info(f"✅ Take profit 1 hit for {symbol} at {current_price}, stop loss moved to breakeven")
            
            conn.commit()
            
    except Exception as e:
        logger.error(f"❌ Error monitoring open trades: {e}")
        if conn: conn.rollback()

def update_balance() -> None:
    global usdt_balance
    if not client:
        return
        
    try:
        account = client.get_account()
        for balance in account['balances']:
            if balance['asset'] == 'USDT':
                with balance_lock:
                    usdt_balance = float(balance['free'])
                logger.info(f"✅ USDT balance updated: {usdt_balance}")
                break
    except BinanceAPIException as e:
        logger.error(f"❌ Failed to update balance: {e}")

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crypto Trading Bot V34.0.5</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
                .card { margin-bottom: 20px; }
                .signal-row { cursor: pointer; }
                .signal-row:hover { background-color: #f8f9fa; }
                .profit { color: green; }
                .loss { color: red; }
                .neutral { color: #6c757d; }
                .status-open { background-color: #d1ecf1; }
                .status-closed { background-color: #f8d7da; }
                .status-updated { background-color: #fff3cd; }
            </style>
        </head>
        <body>
            <div class="container mt-4">
                <div class="row">
                    <div class="col-12">
                        <h1 class="text-center">Crypto Trading Bot V34.0.5</h1>
                        <p class="text-center text-muted">5-Minute Timeframe</p>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5>Trading Controls</h5>
                            </div>
                            <div class="card-body">
                                <div class="form-check form-switch mb-3">
                                    <input class="form-check-input" type="checkbox" id="tradingEnabled" {{ 'checked' if is_trading_enabled else '' }}>
                                    <label class="form-check-label" for="tradingEnabled">Enable Trading</label>
                                </div>
                                
                                <div class="form-check form-switch mb-3">
                                    <input class="form-check-input" type="checkbox" id="paperTradingMode" {{ 'checked' if paper_trading_mode else '' }}>
                                    <label class="form-check-label" for="paperTradingMode">Paper Trading Mode</label>
                                </div>
                                
                                <div class="mb-3">
                                    <label for="maxOpenTrades" class="form-label">Max Open Trades</label>
                                    <input type="number" class="form-control" id="maxOpenTrades" value="{{ MAX_OPEN_TRADES }}" min="1" max="20">
                                </div>
                                
                                <div class="mb-3">
                                    <label for="minSignalQuality" class="form-label">Min Signal Quality</label>
                                    <input type="range" class="form-range" id="minSignalQuality" min="50" max="100" value="{{ MIN_SIGNAL_QUALITY }}">
                                    <div class="d-flex justify-content-between">
                                        <span>50</span>
                                        <span id="minSignalQualityValue">{{ MIN_SIGNAL_QUALITY }}</span>
                                        <span>100</span>
                                    </div>
                                </div>
                                
                                <div class="mb-3">
                                    <label for="tradeAmountMin" class="form-label">Min Trade Amount (USDT)</label>
                                    <input type="number" class="form-control" id="tradeAmountMin" value="{{ FIXED_TRADE_AMOUNT_MIN_USDT }}" min="1" step="0.1">
                                </div>
                                
                                <div class="mb-3">
                                    <label for="tradeAmountMax" class="form-label">Max Trade Amount (USDT)</label>
                                    <input type="number" class="form-control" id="tradeAmountMax" value="{{ FIXED_TRADE_AMOUNT_MAX_USDT }}" min="1" step="0.1">
                                </div>
                                
                                <button id="saveSettings" class="btn btn-primary">Save Settings</button>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5>Account Information</h5>
                            </div>
                            <div class="card-body">
                                <div class="mb-3">
                                    <label class="form-label">USDT Balance</label>
                                    <div class="form-control">{{ "%.2f"|format(usdt_balance) }} USDT</div>
                                </div>
                                
                                <div class="mb-3">
                                    <label class="form-label">Open Trades</label>
                                    <div class="form-control">{{ open_signals|length }} / {{ MAX_OPEN_TRADES }}</div>
                                </div>
                                
                                <div class="mb-3">
                                    <label class="form-label">Trading Mode</label>
                                    <div class="form-control">{{ "Paper Trading" if paper_trading_mode else "Real Trading" }}</div>
                                </div>
                                
                                <div class="mb-3">
                                    <label class="form-label">Bot Status</label>
                                    <div class="form-control">{{ "Enabled" if is_trading_enabled else "Disabled" }}</div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="card">
                            <div class="card-header">
                                <h5>Strategy Settings</h5>
                            </div>
                            <div class="card-body">
                                <div class="form-check form-switch mb-2">
                                    <input class="form-check-input strategy-toggle" type="checkbox" id="bbStochStrategy" data-strategy="USE_BB_STOCH_STRATEGY" {{ 'checked' if USE_BB_STOCH_STRATEGY else '' }}>
                                    <label class="form-check-label" for="bbStochStrategy">BB + Stoch Strategy</label>
                                </div>
                                
                                <div class="form-check form-switch mb-2">
                                    <input class="form-check-input strategy-toggle" type="checkbox" id="macdEmaStrategy" data-strategy="USE_MACD_EMA_STRATEGY" {{ 'checked' if USE_MACD_EMA_STRATEGY else '' }}>
                                    <label class="form-check-label" for="macdEmaStrategy">MACD + EMA Strategy</label>
                                </div>
                                
                                <div class="form-check form-switch mb-2">
                                    <input class="form-check-input strategy-toggle" type="checkbox" id="emaRsiStrategy" data-strategy="USE_EMA_RSI_STRATEGY" {{ 'checked' if USE_EMA_RSI_STRATEGY else '' }}>
                                    <label class="form-check-label" for="emaRsiStrategy">EMA + RSI Strategy</label>
                                </div>
                                
                                <div class="form-check form-switch mb-2">
                                    <input class="form-check-input strategy-toggle" type="checkbox" id="pullbackStrategy" data-strategy="USE_PULLBACK_STRATEGY" {{ 'checked' if USE_PULLBACK_STRATEGY else '' }}>
                                    <label class="form-check-label" for="pullbackStrategy">Pullback Strategy</label>
                                </div>
                                
                                <div class="form-check form-switch mb-2">
                                    <input class="form-check-input strategy-toggle" type="checkbox" id="momentumStrategy" data-strategy="USE_MOMENTUM_VOLATILITY_STRATEGY" {{ 'checked' if USE_MOMENTUM_VOLATILITY_STRATEGY else '' }}>
                                    <label class="form-check-label" for="momentumStrategy">Momentum Strategy</label>
                                </div>
                                
                                <div class="form-check form-switch mb-2">
                                    <input class="form-check-input strategy-toggle" type="checkbox" id="elliottStrategy" data-strategy="USE_ELLIOTT_WAVE_STRATEGY" {{ 'checked' if USE_ELLIOTT_WAVE_STRATEGY else '' }}>
                                    <label class="form-check-label" for="elliottStrategy">Elliott Wave Strategy</label>
                                </div>
                                
                                <div class="form-check form-switch mb-2">
                                    <input class="form-check-input strategy-toggle" type="checkbox" id="rangeStrategy" data-strategy="USE_RANGE_REVERSAL_STRATEGY" {{ 'checked' if USE_RANGE_REVERSAL_STRATEGY else '' }}>
                                    <label class="form-check-label" for="rangeStrategy">Range Reversal Strategy</label>
                                </div>
                                
                                <button id="saveStrategySettings" class="btn btn-primary mt-2">Save Strategy Settings</button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header d-flex justify-content-between align-items-center">
                                <h5>Open Signals</h5>
                                <button id="refreshSignals" class="btn btn-sm btn-outline-primary">Refresh</button>
                            </div>
                            <div class="card-body">
                                <div class="table-responsive">
                                    <table class="table table-striped table-hover">
                                        <thead>
                                            <tr>
                                                <th>Symbol</th>
                                                <th>Strategy</th>
                                                <th>Entry Price</th>
                                                <th>Current Price</th>
                                                <th>Stop Loss</th>
                                                <th>Target 1</th>
                                                <th>Target 2</th>
                                                <th>Profit/Loss</th>
                                                <th>Status</th>
                                                <th>Type</th>
                                            </tr>
                                        </thead>
                                        <tbody id="openSignalsTable">
                                            {% for signal in open_signals.values() %}
                                            <tr class="signal-row" data-symbol="{{ signal.symbol }}">
                                                <td>{{ signal.symbol }}</td>
                                                <td>{{ STRATEGY_NAMES.get(signal.strategy_name, signal.strategy_name) }}</td>
                                                <td>{{ "%.4f"|format(signal.entry_price) }}</td>
                                                <td class="current-price" data-symbol="{{ signal.symbol }}">Loading...</td>
                                                <td>{{ "%.4f"|format(signal.stop_loss) }}</td>
                                                <td>{{ "%.4f"|format(signal.target_price_1) }}</td>
                                                <td>{{ "%.4f"|format(signal.target_price_2) }}</td>
                                                <td class="profit-loss" data-entry="{{ signal.entry_price }}" data-symbol="{{ signal.symbol }}">Calculating...</td>
                                                <td>
                                                    <span class="badge bg-{{ 
                                                        'info' if signal.status == 'open' else 
                                                        'warning' if signal.status == 'updated' else 
                                                        'secondary' 
                                                    }}">
                                                        {{ signal.status }}
                                                    </span>
                                                </td>
                                                <td>
                                                    <span class="badge bg-{{ 'success' if signal.is_real_trade else 'secondary' }}">
                                                        {{ "Real" if signal.is_real_trade else "Paper" }}
                                                    </span>
                                                </td>
                                            </tr>
                                            {% endfor %}
                                        </tbody>
                                    </table>
                                </div>
                                {% if not open_signals %}
                                <div class="text-center text-muted">No open signals</div>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5>Recent Notifications</h5>
                            </div>
                            <div class="card-body" style="max-height: 300px; overflow-y: auto;">
                                <ul class="list-group" id="notificationsList">
                                    {% for notification in notifications_cache %}
                                    <li class="list-group-item">
                                        <div class="d-flex justify-content-between">
                                            <span>{{ notification.message }}</span>
                                            <small class="text-muted">{{ notification.timestamp[:19] }}</small>
                                        </div>
                                    </li>
                                    {% endfor %}
                                </ul>
                                {% if not notifications_cache %}
                                <div class="text-center text-muted">No notifications</div>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5>Rejection Logs</h5>
                            </div>
                            <div class="card-body" style="max-height: 300px; overflow-y: auto;">
                                <ul class="list-group" id="rejectionLogsList">
                                    {% for log in rejection_logs_cache %}
                                    <li class="list-group-item">
                                        <div class="d-flex justify-content-between">
                                            <div>
                                                <strong>{{ log.symbol }}:</strong> {{ log.reason }}
                                            </div>
                                            <small class="text-muted">{{ log.timestamp[:19] }}</small>
                                        </div>
                                    </li>
                                    {% endfor %}
                                </ul>
                                {% if not rejection_logs_cache %}
                                <div class="text-center text-muted">No rejection logs</div>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">
                                <h5>Market State</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    {% for tf, details in current_market_state.trend_details_by_tf.items() %}
                                    <div class="col-md-4">
                                        <div class="card mb-3">
                                            <div class="card-header text-center">
                                                <strong>{{ tf }}</strong>
                                            </div>
                                            <div class="card-body text-center">
                                                <div class="mb-2">
                                                    <span class="badge bg-{{ 
                                                        'success' if details.trend == 'bullish' else 
                                                        'danger' if details.trend == 'bearish' else 
                                                        'secondary' 
                                                    }}">
                                                        {{ details.trend|capitalize }}
                                                    </span>
                                                </div>
                                                <div>ADX: {{ "%.1f"|format(details.adx) }}</div>
                                                <div>RSI: {{ "%.1f"|format(details.rsi) }}</div>
                                                <div>Volatility: {{ "%.2f"|format(details.atr_percent) }}%</div>
                                            </div>
                                        </div>
                                    </div>
                                    {% endfor %}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
            <script>
                // WebSocket connection for real-time updates
                const ws = new WebSocket(`ws://${window.location.host}/ws`);
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'price_update') {
                        // Update current prices
                        for (const [symbol, price] of Object.entries(data.payload)) {
                            const priceElements = document.querySelectorAll(`.current-price[data-symbol="${symbol}"]`);
                            priceElements.forEach(el => {
                                el.textContent = parseFloat(price).toFixed(4);
                            });
                            
                            // Update profit/loss
                            const profitLossElements = document.querySelectorAll(`.profit-loss[data-symbol="${symbol}"]`);
                            profitLossElements.forEach(el => {
                                const entryPrice = parseFloat(el.dataset.entry);
                                const profitLoss = ((price - entryPrice) / entryPrice * 100).toFixed(2);
                                el.textContent = `${profitLoss}%`;
                                
                                // Update color based on profit/loss
                                if (profitLoss > 0) {
                                    el.className = 'profit-loss profit';
                                } else if (profitLoss < 0) {
                                    el.className = 'profit-loss loss';
                                } else {
                                    el.className = 'profit-loss neutral';
                                }
                            });
                        }
                    } else if (data.type === 'new_notification') {
                        // Add new notification to the top of the list
                        const notificationsList = document.getElementById('notificationsList');
                        const newNotification = document.createElement('li');
                        newNotification.className = 'list-group-item';
                        newNotification.innerHTML = `
                            <div class="d-flex justify-content-between">
                                <span>${data.payload.message}</span>
                                <small class="text-muted">${data.payload.timestamp.substring(0, 19)}</small>
                            </div>
                        `;
                        
                        if (notificationsList.firstChild) {
                            notificationsList.insertBefore(newNotification, notificationsList.firstChild);
                        } else {
                            notificationsList.appendChild(newNotification);
                        }
                        
                        // Keep only the latest 20 notifications
                        while (notificationsList.children.length > 20) {
                            notificationsList.removeChild(notificationsList.lastChild);
                        }
                    } else if (data.type === 'new_rejection') {
                        // Add new rejection log to the top of the list
                        const rejectionLogsList = document.getElementById('rejectionLogsList');
                        const newRejection = document.createElement('li');
                        newRejection.className = 'list-group-item';
                        newRejection.innerHTML = `
                            <div class="d-flex justify-content-between">
                                <div>
                                    <strong>${data.payload.symbol}:</strong> ${data.payload.reason}
                                </div>
                                <small class="text-muted">${data.payload.timestamp.substring(0, 19)}</small>
                            </div>
                        `;
                        
                        if (rejectionLogsList.firstChild) {
                            rejectionLogsList.insertBefore(newRejection, rejectionLogsList.firstChild);
                        } else {
                            rejectionLogsList.appendChild(newRejection);
                        }
                        
                        // Keep only the latest 30 rejection logs
                        while (rejectionLogsList.children.length > 30) {
                            rejectionLogsList.removeChild(rejectionLogsList.lastChild);
                        }
                    } else if (data.type === 'market_state_update') {
                        // Reload the page to update market state
                        location.reload();
                    }
                };
                
                // Handle settings form
                document.getElementById('saveSettings').addEventListener('click', function() {
                    const settings = {
                        tradingEnabled: document.getElementById('tradingEnabled').checked,
                        paperTradingMode: document.getElementById('paperTradingMode').checked,
                        maxOpenTrades: parseInt(document.getElementById('maxOpenTrades').value),
                        minSignalQuality: parseInt(document.getElementById('minSignalQuality').value),
                        tradeAmountMin: parseFloat(document.getElementById('tradeAmountMin').value),
                        tradeAmountMax: parseFloat(document.getElementById('tradeAmountMax').value)
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
                            alert('Settings saved successfully!');
                        } else {
                            alert('Error saving settings: ' + data.error);
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('Error saving settings');
                    });
                });
                
                // Handle strategy settings form
                document.getElementById('saveStrategySettings').addEventListener('click', function() {
                    const strategySettings = {};
                    
                    document.querySelectorAll('.strategy-toggle').forEach(toggle => {
                        strategySettings[toggle.dataset.strategy] = toggle.checked;
                    });
                    
                    fetch('/api/strategy-settings', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(strategySettings)
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('Strategy settings saved successfully!');
                        } else {
                            alert('Error saving strategy settings: ' + data.error);
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('Error saving strategy settings');
                    });
                });
                
                // Handle min signal quality slider
                document.getElementById('minSignalQuality').addEventListener('input', function() {
                    document.getElementById('minSignalQualityValue').textContent = this.value;
                });
                
                // Handle refresh signals button
                document.getElementById('refreshSignals').addEventListener('click', function() {
                    location.reload();
                });
            </script>
        </body>
        </html>
    """, 
    is_trading_enabled=is_trading_enabled,
    paper_trading_mode=paper_trading_mode,
    MAX_OPEN_TRADES=MAX_OPEN_TRADES,
    MIN_SIGNAL_QUALITY=MIN_SIGNAL_QUALITY,
    FIXED_TRADE_AMOUNT_MIN_USDT=FIXED_TRADE_AMOUNT_MIN_USDT,
    FIXED_TRADE_AMOUNT_MAX_USDT=FIXED_TRADE_AMOUNT_MAX_USDT,
    usdt_balance=usdt_balance,
    open_signals=open_signals_cache,
    STRATEGY_NAMES=STRATEGY_NAMES,
    USE_BB_STOCH_STRATEGY=USE_BB_STOCH_STRATEGY,
    USE_MACD_EMA_STRATEGY=USE_MACD_EMA_STRATEGY,
    USE_EMA_RSI_STRATEGY=USE_EMA_RSI_STRATEGY,
    USE_PULLBACK_STRATEGY=USE_PULLBACK_STRATEGY,
    USE_MOMENTUM_VOLATILITY_STRATEGY=USE_MOMENTUM_VOLATILITY_STRATEGY,
    USE_ELLIOTT_WAVE_STRATEGY=USE_ELLIOTT_WAVE_STRATEGY,
    USE_RANGE_REVERSAL_STRATEGY=USE_RANGE_REVERSAL_STRATEGY,
    notifications_cache=list(notifications_cache),
    rejection_logs_cache=list(rejection_logs_cache),
    current_market_state=current_market_state
)

@app.route('/api/settings', methods=['POST'])
def update_settings():
    global is_trading_enabled, paper_trading_mode, MAX_OPEN_TRADES, MIN_SIGNAL_QUALITY
    global FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT
    
    try:
        data = request.json
        
        with trading_status_lock:
            is_trading_enabled = data.get('tradingEnabled', is_trading_enabled)
        
        with trading_mode_lock:
            paper_trading_mode = data.get('paperTradingMode', paper_trading_mode)
        
        MAX_OPEN_TRADES = data.get('maxOpenTrades', MAX_OPEN_TRADES)
        
        with min_quality_lock:
            MIN_SIGNAL_QUALITY = data.get('minSignalQuality', MIN_SIGNAL_QUALITY)
        
        with trade_amount_lock:
            FIXED_TRADE_AMOUNT_MIN_USDT = data.get('tradeAmountMin', FIXED_TRADE_AMOUNT_MIN_USDT)
            FIXED_TRADE_AMOUNT_MAX_USDT = data.get('tradeAmountMax', FIXED_TRADE_AMOUNT_MAX_USDT)
        
        # Save to Redis
        save_settings_to_redis()
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/strategy-settings', methods=['POST'])
def update_strategy_settings():
    global USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY
    global USE_PULLBACK_STRATEGY, USE_MOMENTUM_VOLATILITY_STRATEGY
    global USE_ELLIOTT_WAVE_STRATEGY, USE_RANGE_REVERSAL_STRATEGY
    
    try:
        data = request.json
        
        USE_BB_STOCH_STRATEGY = data.get('USE_BB_STOCH_STRATEGY', USE_BB_STOCH_STRATEGY)
        USE_MACD_EMA_STRATEGY = data.get('USE_MACD_EMA_STRATEGY', USE_MACD_EMA_STRATEGY)
        USE_EMA_RSI_STRATEGY = data.get('USE_EMA_RSI_STRATEGY', USE_EMA_RSI_STRATEGY)
        USE_PULLBACK_STRATEGY = data.get('USE_PULLBACK_STRATEGY', USE_PULLBACK_STRATEGY)
        USE_MOMENTUM_VOLATILITY_STRATEGY = data.get('USE_MOMENTUM_VOLATILITY_STRATEGY', USE_MOMENTUM_VOLATILITY_STRATEGY)
        USE_ELLIOTT_WAVE_STRATEGY = data.get('USE_ELLIOTT_WAVE_STRATEGY', USE_ELLIOTT_WAVE_STRATEGY)
        USE_RANGE_REVERSAL_STRATEGY = data.get('USE_RANGE_REVERSAL_STRATEGY', USE_RANGE_REVERSAL_STRATEGY)
        
        # Save to Redis
        save_settings_to_redis()
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error updating strategy settings: {e}")
        return jsonify({'success': False, 'error': str(e)})

# --- WebSocket endpoint ---
@sock.route('/ws')
def websocket_connection(ws):
    with ws_clients_lock:
        ws_clients.append(ws)
    
    try:
        while True:
            # Just keep the connection open
            data = ws.receive()
            if data is None:
                break
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
    finally:
        with ws_clients_lock:
            if ws in ws_clients:
                ws_clients.remove(ws)

# --- Main functions ---
def main():
    # Initialize database
    init_db()
    
    # Initialize Redis
    init_redis()
    
    # Load settings from Redis
    load_settings_from_redis()
    
    # Initialize Binance client
    global client
    try:
        client = Client(API_KEY, API_SECRET)
        logger.info("✅ [API] Binance client initialized successfully.")
    except Exception as e:
        logger.error(f"❌ [API] Failed to initialize Binance client: {e}")
        exit(1)
    
    # Get exchange info
    get_exchange_info_map()
    
    # Get validated symbols
    global validated_symbols_to_scan
    validated_symbols_to_scan = get_validated_symbols()
    if not validated_symbols_to_scan:
        logger.error("❌ No valid symbols found. Exiting.")
        exit(1)
    
    # Load open signals and notifications from database
    load_open_signals_to_cache()
    load_notifications_to_cache()
    
    # Update balance
    update_balance()
    
    # Start WebSocket
    start_websocket()
    
    # Start periodic reports
    start_periodic_reports()
    
    # Start Flask app in a separate thread
    flask_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=5000, threaded=True), daemon=True)
    flask_thread.start()
    logger.info("✅ [Flask] Web interface started on http://localhost:5000")
    
    # Main loop
    logger.info("✅ Bot started successfully. Entering main loop...")
    
    while True:
        try:
            # Update balance every 5 minutes
            update_balance()
            
            # Check and generate signals
            check_and_generate_signals()
            
            # Monitor open trades
            monitor_open_trades()
            
            # Sleep for 30 seconds
            time.sleep(30)
            
        except KeyboardInterrupt:
            logger.info("🛑 Bot stopped by user.")
            break
        except Exception as e:
            logger.error(f"❌ Error in main loop: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()