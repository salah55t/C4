# ملف c4_fixed.py - نسخة V34.1.0 (مراجعة وإصلاح شامل)
# --- وصف التعديلات الرئيسية:
# 1. [إصلاح مشكلة بدء التداول] تم إصلاح الخلل الجذري الذي كان يمنع بدء التداول عند تفعيله من لوحة التحكم. الآن يتم حفظ حالة التداول (مفعّل/معطّل) واستعادتها عند إعادة تشغيل البوت.
# 2. [تحسين منطق الاستراتيجيات] تم استبدال نظام تقييم جودة الإشارة بنظام أكثر تطوراً يأخذ في الاعتبار قوة الاتجاه، حجم التداول، ومؤشرات فنية متعددة لمنح تقييم دقيق لكل فرصة.
# 3. [تطوير لوحة التحكم] تم استبدال النوافذ المنبثقة المزعجة (alert) بنظام إشعارات "Toast" أكثر سلاسة وحداثة لتأكيد حفظ الإعدادات.
# 4. [تحسينات منطقية إضافية] تم تحسين منطق الإشعارات عند تفعيل أو تعطيل التداول لضمان الدقة، مع إضافة تعليقات توضيحية لشرح التغييرات.

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
logger = logging.getLogger('CryptoBotV34.1.0_5min')

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
is_trading_enabled: bool = False # **مهم**: القيمة الأولية ستُحمّل من Redis
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
                columns_to_add = {
                    "target_price_1": "DOUBLE PRECISION", "target_price_2": "DOUBLE PRECISION",
                    "initial_quantity": "DOUBLE PRECISION",
                    "created_at": "TIMESTAMP WITH TIME ZONE DEFAULT NOW()",
                    "atr_percent": "DOUBLE PRECISION",
                    "quality_score": "INTEGER"
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
    
    risk_percent = ((entry_price - stop_loss) / entry_price * 100)
    reward1_percent = ((target1 - entry_price) / entry_price * 100)
    reward2_percent = ((target2 - entry_price) / entry_price * 100)
    rr_ratio1 = reward1_percent / risk_percent if risk_percent > 0 else 0
    rr_ratio2 = reward2_percent / risk_percent if risk_percent > 0 else 0
    
    message = (
        f"{emoji} *صفقة {trade_type} جديدة (5 دقائق)*\n\n"
        f"📊 *العملة:* `{symbol}`\n"
        f"📈 *الاستراتيجية:* `{STRATEGY_NAMES.get(strategy_name, strategy_name)}`\n"
        f"⭐ *جودة الإشارة:* `{quality_score}/100`\n"
        f"📉 *تقلب السوق:* `{atr_percent:.2f}%`\n\n"
        f"💰 *تفاصيل الصفقة:*\n"
        f"🔸 *سعر الدخول:* `{entry_price:.4f}`\n"
        f"🔸 *وقف الخسارة:* `{stop_loss:.4f}`\n"
        f"🔸 *الهدف الأول:* `{target1:.4f}`\n"
        f"🔸 *الهدف الثاني:* `{target2:.4f}`\n\n"
        f"📏 *الكمية:* `{quantity:.4f}`\n"
        f"💵 *قيمة الصفقة:* `${notional_value:.2f}`\n\n"
        f"📊 *نسب المخاطرة والمكافأة:*\n"
        f"🔸 *نسبة المخاطرة:* `{risk_percent:.2f}%`\n"
        f"🔸 *نسبة الربح 1:* `{reward1_percent:.2f}%` (RR: `{rr_ratio1:.2f}`)\n"
        f"🔸 *نسبة الربح 2:* `{reward2_percent:.2f}%` (RR: `{rr_ratio2:.2f}`)\n\n"
        f"⏰ *الوقت:* `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC`"
    )
    
    send_enhanced_telegram_message(message, force=True)

def send_trade_close_notification(symbol: str, strategy_name: str, entry_price: float, close_price: float,
                                profit_percent: float, close_reason: str, is_real: bool):
    trade_type = "حقيقية" if is_real else "ورقية"
    emoji = "✅" if profit_percent > 0 else "❌"
    
    close_reason_ar = {
        "stop_loss": "إيقاف الخسارة", "take_profit_1": "جني الأرباح الأول",
        "take_profit_2": "جني الأرباح الثاني", "manual_close": "إغلاق يدوي",
        "timeout": "انتهاء الوقت"
    }.get(close_reason, close_reason)
    
    message = (
        f"{emoji} *إغلاق صفقة {trade_type}*\n\n"
        f"📊 *العملة:* `{symbol}`\n"
        f"📈 *الاستراتيجية:* `{STRATEGY_NAMES.get(strategy_name, strategy_name)}`\n\n"
        f"💰 *تفاصيل الصفقة:*\n"
        f"🔸 *سعر الدخول:* `{entry_price:.4f}`\n"
        f"🔸 *سعر الإغلاق:* `{close_price:.4f}`\n"
        f"🔸 *سبب الإغلاق:* `{close_reason_ar}`\n\n"
        f"📊 *النتيجة:* `{profit_percent:.2f}%`\n"
        f"⏰ *الوقت:* `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC`"
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
                       SUM(profit_percentage) as total_profit,
                       MAX(profit_percentage) as max_profit,
                       MIN(profit_percentage) as max_loss
                FROM signals
                WHERE closed_at::date = %s AND status = 'closed'
            """, (today,))
            
            stats = cur.fetchone()
            if not stats or stats['total_trades'] == 0:
                logger.info("[Daily Report] No trades to report for today.")
                return
            
            win_rate = (stats['winning_trades'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0
            
            cur.execute("SELECT symbol, profit_percentage, strategy_name FROM signals WHERE closed_at::date = %s AND status = 'closed' ORDER BY profit_percentage DESC LIMIT 1", (today,))
            best_trade = cur.fetchone()
            
            cur.execute("SELECT symbol, profit_percentage, strategy_name FROM signals WHERE closed_at::date = %s AND status = 'closed' ORDER BY profit_percentage ASC LIMIT 1", (today,))
            worst_trade = cur.fetchone()
            
            message = (
                f"📈 *تقرير الأداء اليومي*\n\n📅 *التاريخ:* `{today.strftime('%Y-%m-%d')}`\n\n"
                f"📊 *إحصائيات التداول:*\n"
                f"🔸 *إجمالي الصفقات:* `{stats['total_trades']}`\n"
                f"🔸 *الصفقات الرابحة:* `{stats.get('winning_trades', 0) or 0}`\n"
                f"🔸 *نسبة الربح:* `{win_rate:.1f}%`\n"
                f"🔸 *متوسط الربح:* `{stats.get('avg_profit', 0):.2f}%`\n"
                f"🔸 *إجمالي الربح:* `{stats.get('total_profit', 0):.2f}%`\n"
                f"🔸 *أفضل صفقة:* `{stats.get('max_profit', 0):.2f}%`\n"
                f"🔸 *أسوأ صفقة:* `{stats.get('max_loss', 0):.2f}%`\n\n"
            )
            
            if best_trade:
                message += f"🏆 *أفضل صفقة اليوم:*\nالعملة: `{best_trade['symbol']}` | الربح: `{best_trade['profit_percentage']:.2f}%`\n\n"
            
            if worst_trade:
                message += f"📉 *أسوأ صفقة اليوم:*\nالعملة: `{worst_trade['symbol']}` | الخسارة: `{worst_trade['profit_percentage']:.2f}%`\n\n"
            
            send_enhanced_telegram_message(message, force=True)
            
    except Exception as e:
        logger.error(f"❌ [Daily Report] Error generating daily report: {e}", exc_info=True)

def send_market_state_notification():
    with market_state_lock:
        state = dict(current_market_state)
    
    if not state or not state.get("trend_details_by_tf"): return

    message = f"🌐 *تحديث حالة السوق*\n\n"
    for tf, details in state["trend_details_by_tf"].items():
        trend = details.get("trend", "N/A")
        emoji = "🟢" if trend == "bullish" else "🔴" if trend == "bearish" else "🟡"
        message += f"{emoji} *{tf}:* {trend.capitalize()} (ADX: {details.get('adx', 0):.1f}, RSI: {details.get('rsi', 0):.1f}, Vol: {details.get('atr_percent', 0):.2f}%)\n"
    
    send_enhanced_telegram_message(message, force=False)

def send_bot_status_notification():
    with trading_status_lock: status = "مفعّل" if is_trading_enabled else "معطّل"
    with trading_mode_lock: mode = "حقيقي" if not paper_trading_mode else "ورقي"
    with signal_cache_lock: open_trades = len(open_signals_cache)
    with balance_lock: balance = usdt_balance
    
    message = (
        f"🤖 *حالة البوت*\n\n"
        f"🔌 *الحالة:* `{status}`\n📝 *نوع التداول:* `{mode}`\n"
        f"📊 *الصفقات المفتوحة:* `{open_trades}/{MAX_OPEN_TRADES}`\n💰 *الرصيد:* `{balance:.2f} USDT`\n"
        f"⏰ *الوقت:* `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC`"
    )
    
    send_enhanced_telegram_message(message, force=True)

def schedule_periodic_reports():
    logger.info("Starting periodic reports scheduler...")
    while True:
        try:
            now = datetime.now(timezone.utc)
            if now.hour == 23 and now.minute == 59:
                send_daily_performance_report(); time.sleep(61)
            if now.hour % 6 == 0 and now.minute == 0:
                send_market_state_notification(); time.sleep(61)
            if now.minute == 0:
                send_bot_status_notification(); time.sleep(61)
            time.sleep(30)
        except Exception as e:
            logger.error(f"❌ [Periodic Reports] Error in scheduler: {e}", exc_info=True); time.sleep(60)

def start_periodic_reports():
    reports_thread = Thread(target=schedule_periodic_reports, daemon=True)
    reports_thread.start()
    logger.info("✅ [Periodic Reports] Started periodic reports scheduler thread.")

def handle_socket_message(msg):
    global live_prices
    try:
        if msg and 'e' in msg and msg['e'] == 'error':
            logger.error(f"❌ [WebSocket] Error: {msg['m']}"); return
        
        if isinstance(msg, list):
            price_updates = {}
            with live_prices_lock:
                for ticker in msg:
                    if 's' in ticker and 'c' in ticker:
                        symbol, price = ticker['s'], float(ticker['c'])
                        live_prices[symbol] = price
                        price_updates[symbol] = price
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
    
    # EMAs, SMAs
    for p in [7, 9, 13, 21, 34, 50, 100, 200]:
        df_calc[f'ema{p}'] = df_calc['close'].ewm(span=p, adjust=False).mean()
    df_calc['sma200'] = df_calc['close'].rolling(window=200).mean()
    
    # ATR & ADX
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    df_calc['atr'] = tr.ewm(span=14, adjust=False).mean()
    df_calc['atr_percent'] = (df_calc['atr'] / df_calc['close'].replace(0, 1e-9)) * 100
    plus_dm = (df_calc['high'].diff() > -df_calc['low'].diff()) * df_calc['high'].diff()
    minus_dm = (-df_calc['low'].diff() > df_calc['high'].diff()) * -df_calc['low'].diff()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / df_calc['atr'])
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / df_calc['atr'])
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(alpha=1/14, adjust=False).mean()
    
    # RSI
    delta = df_calc['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=7).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=7).mean()
    rs = gain / loss.replace(0, 1e-9)
    df_calc['rsi'] = 100 - (100 / (1 + rs))
    
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
    low_14, high_14 = df_calc['low'].rolling(14).min(), df_calc['high'].rolling(14).max()
    df_calc['stoch_k'] = 100 * ((df_calc['close'] - low_14) / (high_14 - low_14).replace(0, 1e-9))
    df_calc['stoch_d'] = df_calc['stoch_k'].rolling(3).mean()
    
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

# **FIX**: Load trading status from Redis to ensure persistence
def load_settings_from_redis():
    global FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT, MAX_OPEN_TRADES, paper_trading_mode, MIN_SIGNAL_QUALITY, is_trading_enabled, USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY, USE_MOMENTUM_VOLATILITY_STRATEGY, USE_ELLIOTT_WAVE_STRATEGY, USE_RANGE_REVERSAL_STRATEGY
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
            with trading_status_lock: is_trading_enabled = settings.get('is_trading_enabled', False)

        quality_settings_data = redis_client.get('signal_quality_settings')
        if quality_settings_data:
            with min_quality_lock: MIN_SIGNAL_QUALITY = json.loads(quality_settings_data).get('min_quality', 70)

        strategies_data = redis_client.get('strategy_settings')
        if strategies_data:
            s = json.loads(strategies_data)
            USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY, USE_MOMENTUM_VOLATILITY_STRATEGY, USE_ELLIOTT_WAVE_STRATEGY, USE_RANGE_REVERSAL_STRATEGY = s.get('USE_BB_STOCH_STRATEGY', True), s.get('USE_MACD_EMA_STRATEGY', True), s.get('USE_EMA_RSI_STRATEGY', True), s.get('USE_PULLBACK_STRATEGY', True), s.get('USE_MOMENTUM_VOLATILITY_STRATEGY', True), s.get('USE_ELLIOTT_WAVE_STRATEGY', True), s.get('USE_RANGE_REVERSAL_STRATEGY', True)

        logger.info("✅ [Redis] Successfully loaded settings from Redis.")
    except Exception as e:
        logger.error(f"❌ [Redis] Error loading settings: {e}")

# **FIX**: Save trading status to Redis to ensure persistence
def save_settings_to_redis():
    if not redis_client:
        logger.warning("Redis client not available, cannot save settings")
        return False
    try:
        trading_settings = {
            'FIXED_TRADE_AMOUNT_MIN_USDT': FIXED_TRADE_AMOUNT_MIN_USDT,
            'FIXED_TRADE_AMOUNT_MAX_USDT': FIXED_TRADE_AMOUNT_MAX_USDT,
            'MAX_OPEN_TRADES': MAX_OPEN_TRADES,
            'paper_trading_mode': paper_trading_mode,
            'is_trading_enabled': is_trading_enabled
        }
        redis_client.set('trading_settings', json.dumps(trading_settings))
        
        redis_client.set('signal_quality_settings', json.dumps({'min_quality': MIN_SIGNAL_QUALITY}))
        
        strategy_settings = {
            'USE_BB_STOCH_STRATEGY': USE_BB_STOCH_STRATEGY, 'USE_MACD_EMA_STRATEGY': USE_MACD_EMA_STRATEGY,
            'USE_EMA_RSI_STRATEGY': USE_EMA_RSI_STRATEGY, 'USE_PULLBACK_STRATEGY': USE_PULLBACK_STRATEGY,
            'USE_MOMENTUM_VOLATILITY_STRATEGY': USE_MOMENTUM_VOLATILITY_STRATEGY, 'USE_ELLIOTT_WAVE_STRATEGY': USE_ELLIOTT_WAVE_STRATEGY,
            'USE_RANGE_REVERSAL_STRATEGY': USE_RANGE_REVERSAL_STRATEGY
        }
        redis_client.set('strategy_settings', json.dumps(strategy_settings))
        logger.info("✅ [Redis] Settings saved to Redis successfully")
        return True
    except Exception as e:
        logger.error(f"❌ [Redis] Error saving settings to Redis: {e}"); return False

# --- نظام تقييم جودة الإشارة المحسن ---
def calculate_signal_quality(df: pd.DataFrame, strategy_name: str, mtf_trend: Dict) -> int:
    """
    يحسب درجة جودة إشارة أكثر تفصيلاً بناءً على عوامل متعددة.
    """
    score = 0
    last_row = df.iloc[-1]
    
    # 1. درجة أساسية للاستراتيجية (بعضها أكثر موثوقية)
    base_scores = {
        "Elliott_Wave_Strategy": 40, "Pullback_Strategy": 35, "MACD_EMA_Strategy": 30,
        "BB_Stoch_Strategy": 25, "Momentum_Volatility_Strategy": 25, "EMA_RSI_Strategy": 20,
        "Range_Reversal_Strategy": 20,
    }
    score += base_scores.get(strategy_name, 20)

    # 2. توافق الاتجاه (+20 نقطة)
    if mtf_trend.get('5m') == 'bullish' and mtf_trend.get('15m') == 'bullish': score += 20
    elif mtf_trend.get('5m') == 'bullish': score += 10

    # 3. تأكيد حجم التداول (+20 نقطة)
    volume_ma20 = df['volume'].rolling(20).mean().iloc[-1]
    if volume_ma20 > 0:
        if last_row['volume'] > volume_ma20 * 2: score += 20
        elif last_row['volume'] > volume_ma20 * 1.5: score += 15

    # 4. قوة اتجاه ADX (+15 نقطة)
    if last_row['adx'] > 25: score += 15
    elif last_row['adx'] > 20: score += 10

    # 5. التقلب (ATR) في النطاق الأمثل (+10 نقاط)
    atr_percent = last_row.get('atr_percent', 0)
    if 0.8 <= atr_percent <= 2.2: score += 10
        
    # 6. تأكيد RSI (+15 نقطة)
    if 55 < last_row['rsi'] < 70: score += 15
    elif last_row['rsi'] >= 70: score -= 5 # تشبع شرائي، خطورة طفيفة

    return min(100, int(score))

# --- الفلاتر الديناميكية ونظام السوق ---
def get_wave_retracement(df: pd.DataFrame) -> float:
    try:
        highs, lows = df['high'].values, df['low'].values
        peaks_idx = argrelextrema(highs, np.greater, order=5)[0]
        troughs_idx = argrelextrema(lows, np.less, order=5)[0]
        
        if len(peaks_idx) < 1 or len(troughs_idx) < 2: return 999.0
        
        last_trough_idx = troughs_idx[-1]
        prev_peak_idx = peaks_idx[peaks_idx < last_trough_idx][-1]
        prev_trough_idx = troughs_idx[troughs_idx < prev_peak_idx][-1]

        wave_start, wave_end, retracement_price = lows[prev_trough_idx], highs[prev_peak_idx], lows[last_trough_idx]

        wave_height = wave_end - wave_start
        if wave_height <= 0: return 999.0
        
        return (wave_end - retracement_price) / wave_height
    except Exception:
        return 999.0

def check_elliott_wave_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    fib_min, fib_max = (0.236, 0.786) if atr_percent > 2.5 else (0.236, 0.618)
    volume_ma = df['volume'].rolling(20).mean()
    wave_volume_multiplier = 1.3 + (atr_percent / 50)
    macd_momentum = df['macd_hist'].rolling(5).mean()
    momentum_threshold = macd_momentum.rolling(20).std() * 0.3
    
    return {
        'fibonacci_ok': fib_min <= get_wave_retracement(df) <= fib_max,
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * wave_volume_multiplier,
        'momentum_ok': macd_momentum.iloc[-1] > momentum_threshold.iloc[-1],
    }

def check_bb_stoch_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    bb_width = df['bb_width']
    dynamic_bb_threshold = bb_width.rolling(20).mean() * 1.2
    stoch_threshold = 23 if atr_percent > 1.5 else 18
    volume_ma = df['volume'].rolling(20).mean()
    volume_multiplier = 1.2 + (atr_percent / 80)
    
    return {
        'bb_width_ok': bb_width.iloc[-1] > dynamic_bb_threshold.iloc[-1],
        'stoch_ok': last_row['stoch_k'] > stoch_threshold,
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * volume_multiplier
    }

def check_macd_ema_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    adx_threshold = 22 if atr_percent > 1.5 else 18
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
    rsi_lower, rsi_upper = (42, 78) if adx > 25 else (48, 72)
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
    volatility_ma, volatility_std = atr_percent.rolling(20).mean(), atr_percent.rolling(20).std()
    dynamic_vol_min = volatility_ma.iloc[-1] - (volatility_std.iloc[-1] * 1.5)
    dynamic_vol_max = volatility_ma.iloc[-1] + (volatility_std.iloc[-1] * 1.5)
    is_momentum_ok = (last_row['rsi'] > 51) and (df['macd_hist'].iloc[-1] > df['macd_hist'].iloc[-2])
    dynamic_adx_threshold = df['adx'].rolling(20).mean().iloc[-1] * 0.85
    
    return {
        'volatility_ok': dynamic_vol_min <= atr_percent.iloc[-1] <= dynamic_vol_max,
        'momentum_ok': is_momentum_ok, 'adx_ok': last_row['adx'] > dynamic_adx_threshold,
    }

def check_range_reversal_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    rsi_threshold = 35 if atr_percent < 2.5 else 40
    return {'adx_ok': last_row.get('adx', 99) < 23, 'rsi_ok': last_row.get('rsi', 50) < rsi_threshold}

# --- General Filters ---
def add_news_filter() -> bool:
    news_hours = [(12, 30), (14, 0), (18, 30)]
    now = datetime.now(timezone.utc)
    for hour, minute in news_hours:
        if now.hour == hour and abs(now.minute - minute) <= 15: return False
    return True

def add_liquidity_filter() -> bool:
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5: return False
    if now.hour >= 21 or now.hour <= 3: return False
    if now.hour == 12 and (30 <= now.minute <= 45): return False
    return True

def add_correlation_filter(new_symbol: str) -> bool:
    correlated_groups = [
        {'BTCUSDT', 'ETHUSDT', 'BCHUSDT', 'LTCUSDT'}, {'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'ATOMUSDT'},
        {'SOLUSDT', 'AVAXUSDT', 'MATICUSDT', 'NEARUSDT'}, {'XRPUSDT', 'XLMUSDT', 'ALGOUSDT'},
    ]
    with signal_cache_lock: open_symbols = set(open_signals_cache.keys())
    if not open_symbols: return True
    for group in correlated_groups:
        if new_symbol in group and not open_symbols.isdisjoint(group): return False
    return True

def add_random_volatility_filter(df: pd.DataFrame) -> bool:
    price_change = abs(df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6]
    avg_price_change = df['close'].pct_change().abs().rolling(20).mean().iloc[-1]
    return not price_change > avg_price_change * 3

def check_market_volatility_filter_enhanced(df: pd.DataFrame, symbol: str = "Unknown") -> bool:
    if 'atr_percent' not in df.columns or df['atr_percent'].isnull().all():
        log_rejection(symbol, "Market Volatility Filter Failed", {"reason": "No ATR data"}); return False
    
    last_atr_percent = float(df.iloc[-1].get('atr_percent', 0))
    ATR_PERCENT_MIN, ATR_PERCENT_MAX = 0.5, 2.8
    if not (ATR_PERCENT_MIN <= last_atr_percent <= ATR_PERCENT_MAX):
        log_rejection(symbol, "Market Volatility Filter Failed", {"atr": f"{last_atr_percent:.2f}%", "range": f"({ATR_PERCENT_MIN:.2f}-{ATR_PERCENT_MAX:.2f})%"})
        return False
    return True

# --- تحسين منطق وقف الخسارة وجني الأرباح ---
def calculate_dynamic_stop_loss_enhanced(df: pd.DataFrame, entry_price: float, strategy_name: str) -> float:
    last = df.iloc[-1]
    atr_value = last.get('atr', 0)
    atr_percent = last.get('atr_percent', 0)
    
    if atr_percent > 2.5: atr_multiplier = 3.0
    elif atr_percent > 1.5: atr_multiplier = 2.3
    elif atr_percent > 1.0: atr_multiplier = 1.8
    else: atr_multiplier = 1.5
    
    volume_ma = df['volume'].rolling(20).mean().iloc[-1]
    volume_ratio = last['volume'] / volume_ma if volume_ma > 0 else 1
    if volume_ratio > 2.0: atr_multiplier *= 0.8
    elif volume_ratio < 0.5: atr_multiplier *= 1.2
    
    base_sl = entry_price - (atr_value * atr_multiplier)
    recent_low = df['low'].tail(5).min()

    if strategy_name in ["BB_Stoch_Strategy", "Pullback_Strategy", "Range_Reversal_Strategy"]:
        stop_loss = min(recent_low * 0.995, base_sl)
    elif strategy_name == "Elliott_Wave_Strategy":
        lows = df['low'].values
        support_idx = argrelextrema(lows, np.less, order=5)[0]
        recent_support = lows[support_idx[-1]] if len(support_idx) > 0 else recent_low
        stop_loss = min(recent_support * 0.995, base_sl)
    else:
        stop_loss = min(last.get('ema21', base_sl), base_sl)
    
    max_stop_distance = entry_price * 0.05
    return max(stop_loss, entry_price - max_stop_distance)

def calculate_dynamic_take_profit_enhanced(df: pd.DataFrame, entry_price: float, stop_loss: float, strategy_name: str) -> tuple:
    risk_amount = entry_price - stop_loss
    if risk_amount <= 0: return (entry_price * 1.015, entry_price * 1.025)

    if strategy_name == "Range_Reversal_Strategy":
        return df.iloc[-1].get('bb_middle', entry_price * 1.015), df.iloc[-1].get('bb_upper', entry_price * 1.03)

    rr_ratios = {
        "Elliott_Wave_Strategy": (2.2, 4.0), "Pullback_Strategy": (2.0, 3.8),
        "BB_Stoch_Strategy": (2.0, 3.5), "EMA_RSI_Strategy": (1.9, 3.5),
        "MACD_EMA_Strategy": (1.8, 3.2), "Momentum_Volatility_Strategy": (1.7, 3.0),
    }
    rr1, rr2 = rr_ratios.get(strategy_name, (1.6, 2.8))
    
    return entry_price + (risk_amount * rr1), entry_price + (risk_amount * rr2)

# --- استراتيجيات التداول المحسنة ---
def check_ema_rsi_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 200: return False
    last = df.iloc[-1]
    if not (last['ema50'] > last['ema200'] and last['close'] > last['ema9']): return False
    filters = check_ema_rsi_dynamic_filters(df)
    if not all(filters.values()): log_rejection(symbol_name, "DYN_EMA_RSI_FAIL"); return False
    if not add_random_volatility_filter(df): log_rejection(symbol_name, "Random Volatility Filter Failed"); return False
    return True

def check_bb_stoch_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 50: return False
    last = df.iloc[-1]
    if last['close'] < last['ema50']: log_rejection(symbol_name, "BB: Price below EMA50 (bearish trend)"); return False
    filters = check_bb_stoch_dynamic_filters(df)
    if not all(filters.values()): log_rejection(symbol_name, "DYN_BB_STOCH_FAIL"); return False
    if not add_random_volatility_filter(df): log_rejection(symbol_name, "Random Volatility Filter Failed"); return False
    return True

def check_macd_ema_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 100: return False
    last = df.iloc[-1]
    if last['macd'] < last['macd_signal'] or last['macd_hist'] < 0: log_rejection(symbol_name, "MACD Momentum Negative"); return False
    filters = check_macd_ema_dynamic_filters(df)
    if not all(filters.values()): log_rejection(symbol_name, "DYN_MACD_EMA_FAIL"); return False
    if not add_random_volatility_filter(df): log_rejection(symbol_name, "Random Volatility Filter Failed"); return False
    return True

def check_pullback_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 100: return False
    if mtf_trend.get('5m') != 'bullish' or mtf_trend.get('15m') != 'bullish': log_rejection(symbol_name, "Pullback: Trend is not strongly bullish"); return False
    filters = check_pullback_dynamic_filters(df, mtf_trend)
    if not all(filters.values()): log_rejection(symbol_name, "DYN_PULLBACK_FAIL"); return False
    if not add_random_volatility_filter(df): log_rejection(symbol_name, "Random Volatility Filter Failed"); return False
    return True

def check_momentum_volatility_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 100: return False
    last = df.iloc[-1]
    if not (last['ema9'] > last['ema21'] > last['ema34']): log_rejection(symbol_name, "Momentum: EMAs not in bullish order"); return False
    filters = check_momentum_volatility_dynamic_filters(df)
    if not all(filters.values()): log_rejection(symbol_name, "DYN_MOMENTUM_FAIL"); return False
    if not add_random_volatility_filter(df): log_rejection(symbol_name, "Random Volatility Filter Failed"); return False
    return True

def check_elliott_wave_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 100: return False
    if mtf_trend.get('5m') != 'bullish': log_rejection(symbol_name, "Elliott Wave: Strongly bearish trend"); return False
    filters = check_elliott_wave_dynamic_filters(df)
    if not all(filters.values()): log_rejection(symbol_name, "DYN_ELLIOTT_FAIL"); return False
    if not add_random_volatility_filter(df): log_rejection(symbol_name, "Random Volatility Filter Failed"); return False
    return True

def check_range_reversal_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 100: return False
    filters = check_range_reversal_dynamic_filters(df)
    if not all(filters.values()): log_rejection(symbol_name, "DYN_RANGE_REVERSAL_FAIL"); return False
    if not add_random_volatility_filter(df): log_rejection(symbol_name, "Random Volatility Filter Failed"); return False
    return True

# --- دوال المساعدة للتحقق من الإشارات ---
def apply_general_filters(symbol: str, df: pd.DataFrame) -> bool:
    if not check_market_volatility_filter_enhanced(df, symbol): return False
    if not add_news_filter(): log_rejection(symbol, "News Filter Failed"); return False
    if not add_liquidity_filter(): log_rejection(symbol, "Liquidity Filter Failed"); return False
    if not add_correlation_filter(symbol): log_rejection(symbol, "Correlation Filter Failed"); return False
    return True

def get_market_trend_for_timeframes(df_dict: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    trend_results = {}
    for tf, df in df_dict.items():
        if df is None or len(df) < 50:
            trend_results[tf] = 'unknown'
            continue
        last = df.iloc[-1]
        if last['ema9'] > last['ema21'] > last['ema50']: trend_results[tf] = 'bullish'
        elif last['ema9'] < last['ema21'] < last['ema50']: trend_results[tf] = 'bearish'
        else: trend_results[tf] = 'neutral'
    return trend_results

def analyze_market_state() -> Dict[str, Any]:
    if not validated_symbols_to_scan or not client: return {"trend_details_by_tf": {}}
    btc_df = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, 2)
    if btc_df is None or len(btc_df) < 50: return {"trend_details_by_tf": {}}
    btc_df = calculate_all_features(btc_df)
    
    trend_details = {}
    for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
        df = btc_df if tf == SIGNAL_GENERATION_TIMEFRAME else fetch_historical_data(BTC_SYMBOL, tf, 3)
        if df is None or len(df) < 50: continue
        if tf != SIGNAL_GENERATION_TIMEFRAME: df = calculate_all_features(df)
        last = df.iloc[-1]
        trend = 'neutral'
        if last['ema9'] > last['ema21'] > last['ema50']: trend = 'bullish'
        elif last['ema9'] < last['ema21'] < last['ema50']: trend = 'bearish'
        trend_details[tf] = {'trend': trend, 'adx': last.get('adx', 0), 'rsi': last.get('rsi', 50), 'atr_percent': last.get('atr_percent', 0)}
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
        if paper_trading_mode: return PAPER_TRADE_FIXED_AMOUNT_USDT
    with balance_lock:
        if usdt_balance <= 0: return FIXED_TRADE_AMOUNT_MIN_USDT
        amount = usdt_balance * 0.05
    with trade_amount_lock:
        return max(FIXED_TRADE_AMOUNT_MIN_USDT, min(amount, FIXED_TRADE_AMOUNT_MAX_USDT))

def calculate_position_size(symbol: str, entry_price: float, usdt_amount: float) -> Optional[float]:
    try:
        quantity = usdt_amount / entry_price
        info = exchange_info_map.get(symbol)
        if not info: return quantity

        for f in info.get('filters', []):
            if f.get('filterType') == 'LOT_SIZE':
                min_qty, step_size = float(f['minQty']), float(f['stepSize'])
                if quantity < min_qty: return None
                quantity = float(Decimal(str(quantity - (quantity % step_size))).quantize(Decimal(str(step_size))))
                break
        
        for f in info.get('filters', []):
            if f.get('filterType') == 'MIN_NOTIONAL':
                min_notional = float(f.get('minNotional', 0))
                if (quantity * entry_price) < min_notional: return None
                break
        
        return quantity
    except Exception as e:
        logger.error(f"Error calculating position size for {symbol}: {e}"); return None

def execute_trade(symbol: str, strategy_name: str, entry_price: float, stop_loss: float, 
                 target1: float, target2: float, quality_score: int, atr_percent: float) -> bool:
    with trading_status_lock:
        if not is_trading_enabled:
            logger.info("Trading is disabled. Skipping trade execution."); return False
    
    with trading_mode_lock: is_real_trade = not paper_trading_mode
    usdt_amount = get_trade_amount_usdt()
    quantity = calculate_position_size(symbol, entry_price, usdt_amount)
    
    if quantity is None or quantity <= 0:
        log_rejection(symbol, "LOT_SIZE Filter Failed"); return False
    
    notional_value = quantity * entry_price
    
    if is_real_trade:
        with balance_lock:
            if usdt_balance < notional_value:
                log_rejection(symbol, "Insufficient Balance", {"balance": usdt_balance, "required": notional_value})
                if AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE:
                    with trading_mode_lock: paper_trading_mode = True; is_real_trade = False
                    logger.warning("Auto-switched to paper trading due to insufficient balance")
                else: return False
    
    if not check_db_connection() or not conn: return False
        
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, stop_loss, target_price_1, target_price_2, strategy_name, quantity, is_real_trade, signal_details, atr_percent, quality_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
            """, (symbol, entry_price, stop_loss, target1, target2, strategy_name, quantity, is_real_trade, json.dumps({"atr_percent": atr_percent, "quality_score": quality_score, "notional_value": notional_value}), atr_percent, quality_score))
            signal_id = cur.fetchone()['id']
            conn.commit()
            
            with signal_cache_lock:
                open_signals_cache[symbol] = {
                    'id': signal_id, 'symbol': symbol, 'entry_price': entry_price, 'stop_loss': stop_loss,
                    'target_price_1': target1, 'target_price_2': target2, 'strategy_name': strategy_name,
                    'quantity': quantity, 'is_real_trade': is_real_trade, 'status': 'open',
                    'created_at': datetime.now(timezone.utc).isoformat()
                }
            
            send_trade_open_notification(symbol, strategy_name, entry_price, stop_loss, target1, target2, quantity, is_real_trade, quality_score, atr_percent, notional_value)
            
            if is_real_trade:
                try:
                    order = client.create_order(symbol=symbol, side=Client.SIDE_BUY, type=Client.ORDER_TYPE_MARKET, quantity=quantity)
                    with conn.cursor() as cur2:
                        cur2.execute("UPDATE signals SET order_id = %s WHERE id = %s", (order['orderId'], signal_id)); conn.commit()
                    update_balance()
                    logger.info(f"✅ Real trade executed for {symbol}: {order}")
                except BinanceAPIException as e:
                    logger.error(f"❌ Failed to execute real trade for {symbol}: {e}")
                    with conn.cursor() as cur2:
                        cur2.execute("UPDATE signals SET status = 'failed', closing_reason = %s, closed_at = NOW() WHERE id = %s", (str(e), signal_id)); conn.commit()
                    with signal_cache_lock:
                        if symbol in open_signals_cache: del open_signals_cache[symbol]
                    return False
            
            logger.info(f"✅ Signal created for {symbol} with strategy {strategy_name}"); return True
    except Exception as e:
        logger.error(f"❌ Error executing trade for {symbol}: {e}");
        if conn: conn.rollback()
        return False

def check_and_generate_signals() -> None:
    with trading_status_lock:
        if not is_trading_enabled: return
    
    if not validated_symbols_to_scan or not client: return
    with signal_cache_lock: open_symbols = set(open_signals_cache.keys())
    if len(open_symbols) >= MAX_OPEN_TRADES: return
        
    btc_df = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, 2)
    if btc_df is None or len(btc_df) < 50: return
    btc_df = calculate_all_features(btc_df)
    mtf_trend = get_market_trend_for_timeframes({SIGNAL_GENERATION_TIMEFRAME: btc_df, HIGHER_TIMEFRAME: fetch_historical_data(BTC_SYMBOL, HIGHER_TIMEFRAME, 3)})
    update_market_state()
    
    symbols_to_process = random.sample(validated_symbols_to_scan, len(validated_symbols_to_scan))
    
    strategies = {
        "BB_Stoch_Strategy": (USE_BB_STOCH_STRATEGY, check_bb_stoch_strategy_enhanced),
        "MACD_EMA_Strategy": (USE_MACD_EMA_STRATEGY, check_macd_ema_strategy_enhanced),
        "EMA_RSI_Strategy": (USE_EMA_RSI_STRATEGY, check_ema_rsi_strategy_enhanced),
        "Pullback_Strategy": (USE_PULLBACK_STRATEGY, check_pullback_strategy_enhanced),
        "Momentum_Volatility_Strategy": (USE_MOMENTUM_VOLATILITY_STRATEGY, check_momentum_volatility_strategy_enhanced),
        "Elliott_Wave_Strategy": (USE_ELLIOTT_WAVE_STRATEGY, check_elliott_wave_strategy_enhanced),
        "Range_Reversal_Strategy": (USE_RANGE_REVERSAL_STRATEGY, check_range_reversal_strategy_enhanced),
    }

    for symbol in symbols_to_process:
        if len(open_symbols) >= MAX_OPEN_TRADES: break
        if symbol in open_symbols: continue
        with cooldowns_lock:
            if symbol in cooldowns_by_symbol and datetime.now(timezone.utc) < cooldowns_by_symbol[symbol]: continue
            elif symbol in cooldowns_by_symbol: del cooldowns_by_symbol[symbol]
        
        df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
        if df is None or len(df) < 200: continue
        
        df = calculate_all_features(df)
        df.name = symbol
        
        if not apply_general_filters(symbol, df): continue
            
        for name, (is_enabled, func) in strategies.items():
            if is_enabled and func(df, mtf_trend):
                last_row = df.iloc[-1]
                entry_price = last_row['close']
                stop_loss = calculate_dynamic_stop_loss_enhanced(df, entry_price, name)
                if entry_price <= stop_loss: log_rejection(symbol, "Invalid Position Size"); continue
                target1, target2 = calculate_dynamic_take_profit_enhanced(df, entry_price, stop_loss, name)
                
                quality_score = calculate_signal_quality(df, name, mtf_trend)
                
                with min_quality_lock: min_quality = MIN_SIGNAL_QUALITY
                if quality_score < min_quality:
                    log_rejection(symbol, "Low Quality Signal", {"quality": quality_score, "min_required": min_quality}); continue
                
                if execute_trade(symbol, name, entry_price, stop_loss, target1, target2, quality_score, last_row.get('atr_percent', 0)):
                    with signal_cache_lock: open_symbols.add(symbol)
                break
        time.sleep(0.1)

def monitor_open_trades() -> None:
    if not check_db_connection() or not conn: return
    
    signals_to_close = []
    signals_to_update = []
    with signal_cache_lock:
        open_signals_copy = list(open_signals_cache.values())
        
    for signal in open_signals_copy:
        symbol = signal['symbol']
        with live_prices_lock: current_price = live_prices.get(symbol)
        if not current_price: continue
        
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        target1 = signal.get('target_price_1', entry_price * 1.02)
        target2 = signal.get('target_price_2', entry_price * 1.03)
        profit_percent = ((current_price - entry_price) / entry_price) * 100
        
        if current_price <= stop_loss:
            signals_to_close.append((signal, current_price, "stop_loss", profit_percent))
        elif current_price >= target2:
            signals_to_close.append((signal, current_price, "take_profit_2", profit_percent))
        elif current_price >= target1 and signal['status'] == 'open':
            signals_to_update.append((signal, current_price, "take_profit_1"))
            
    if not signals_to_close and not signals_to_update: return

    try:
        with conn.cursor() as cur:
            for signal, price, reason, profit in signals_to_close:
                cur.execute("UPDATE signals SET status = 'closed', closing_price = %s, closed_at = NOW(), profit_percentage = %s, closing_reason = %s WHERE id = %s", (price, profit, reason, signal['id']))
                send_trade_close_notification(signal['symbol'], signal['strategy_name'], signal['entry_price'], price, profit, reason, signal['is_real_trade'])
                if signal['is_real_trade']:
                    try:
                        client.create_order(symbol=signal['symbol'], side=Client.SIDE_SELL, type=Client.ORDER_TYPE_MARKET, quantity=signal['quantity'])
                    except BinanceAPIException as e:
                        logger.error(f"❌ Failed to execute closing order for {signal['symbol']}: {e}")
                with cooldowns_lock:
                    cooldowns_by_symbol[signal['symbol']] = datetime.now(timezone.utc) + timedelta(minutes=COOLDOWN_MINUTES_AFTER_SL)
                with signal_cache_lock:
                    if signal['symbol'] in open_signals_cache: del open_signals_cache[signal['symbol']]

            for signal, price, reason in signals_to_update:
                new_stop_loss = signal['entry_price']
                half_quantity = signal['quantity'] / 2
                cur.execute("UPDATE signals SET status = 'updated', closing_reason = %s, stop_loss = %s, quantity = %s WHERE id = %s", (reason, new_stop_loss, half_quantity, signal['id']))
                if signal['is_real_trade']:
                    try:
                        client.create_order(symbol=signal['symbol'], side=Client.SIDE_SELL, type=Client.ORDER_TYPE_MARKET, quantity=half_quantity)
                    except BinanceAPIException as e:
                        logger.error(f"❌ Failed to execute TP1 order for {signal['symbol']}: {e}")
                with signal_cache_lock:
                    if signal['symbol'] in open_signals_cache:
                        open_signals_cache[signal['symbol']].update({'status': 'updated', 'stop_loss': new_stop_loss, 'quantity': half_quantity})
                logger.info(f"✅ TP1 hit for {signal['symbol']}, SL moved to breakeven.")

        conn.commit()
        if signals_to_close or signals_to_update: update_balance()
    except Exception as e:
        logger.error(f"❌ Error during trade monitoring transaction: {e}"); conn.rollback()


def update_balance() -> None:
    global usdt_balance
    if not client: return
    try:
        account = client.get_account()
        for b in account['balances']:
            if b['asset'] == 'USDT':
                with balance_lock: usdt_balance = float(b['free'])
                logger.info(f"✅ USDT balance updated: {usdt_balance}")
                break
    except BinanceAPIException as e:
        logger.error(f"❌ Failed to update balance: {e}")

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="ar" dir="rtl">
        <head>
            <title>Crypto Trading Bot V34.1.0</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.rtl.min.css" rel="stylesheet">
            <style>
                body { font-family: 'Tahoma', sans-serif; }
                .card { margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.05); }
                .profit { color: green; } .loss { color: red; } .neutral { color: #6c757d; }
                .trading-enabled { border-right: 5px solid #198754; }
                .trading-disabled { border-right: 5px solid #dc3545; }
                #toast-container { position: fixed; bottom: 20px; left: 20px; z-index: 1050; }
                .toast-notification { padding: 15px 20px; border-radius: 5px; color: #fff; font-size: 16px; margin-top: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.2); opacity: 0; transition: opacity 0.3s, transform 0.3s; transform: translateY(20px); }
                .toast-notification.show { opacity: 1; transform: translateY(0); }
                .toast-success { background-color: #198754; } .toast-error { background-color: #dc3545; }
            </style>
        </head>
        <body>
            <div class="container mt-4">
                <h1 class="text-center">بوت تداول العملات الرقمية V34.1.0</h1>
                <p class="text-center text-muted">إطار زمني 5 دقائق</p>
                
                <div class="row">
                    <div class="col-lg-6">
                        <div class="card {{ 'trading-enabled' if is_trading_enabled else 'trading-disabled' }}">
                            <div class="card-header"><h5>لوحة التحكم</h5></div>
                            <div class="card-body">
                                <div class="d-grid gap-2">
                                <div class="form-check form-switch fs-5 mb-3">
                                    <input class="form-check-input" type="checkbox" id="tradingEnabled" {{ 'checked' if is_trading_enabled else '' }}>
                                    <label class="form-check-label" for="tradingEnabled">تفعيل التداول</label>
                                </div>
                                <div class="form-check form-switch mb-3">
                                    <input class="form-check-input" type="checkbox" id="paperTradingMode" {{ 'checked' if paper_trading_mode else '' }}>
                                    <label class="form-check-label" for="paperTradingMode">وضع التداول الورقي</label>
                                </div>
                                </div>
                                <div class="mb-3">
                                    <label for="maxOpenTrades" class="form-label">أقصى عدد للصفقات المفتوحة</label>
                                    <input type="number" class="form-control" id="maxOpenTrades" value="{{ MAX_OPEN_TRADES }}" min="1" max="20">
                                </div>
                                <div class="mb-3">
                                    <label for="minSignalQuality" class="form-label">أدنى جودة للإشارة (<span id="minSignalQualityValue">{{ MIN_SIGNAL_QUALITY }}</span>)</label>
                                    <input type="range" class="form-range" id="minSignalQuality" min="50" max="100" value="{{ MIN_SIGNAL_QUALITY }}">
                                </div>
                                <div class="mb-3">
                                    <label for="tradeAmountMin" class="form-label">أدنى قيمة للصفقة (USDT)</label>
                                    <input type="number" class="form-control" id="tradeAmountMin" value="{{ FIXED_TRADE_AMOUNT_MIN_USDT }}" min="1" step="0.1">
                                </div>
                                <div class="mb-3">
                                    <label for="tradeAmountMax" class="form-label">أقصى قيمة للصفقة (USDT)</label>
                                    <input type="number" class="form-control" id="tradeAmountMax" value="{{ FIXED_TRADE_AMOUNT_MAX_USDT }}" min="1" step="0.1">
                                </div>
                                <button id="saveSettings" class="btn btn-primary w-100">حفظ الإعدادات</button>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-lg-6">
                        <div class="card">
                            <div class="card-header"><h5>معلومات الحساب</h5></div>
                            <div class="card-body">
                                <ul class="list-group list-group-flush">
                                    <li class="list-group-item d-flex justify-content-between"><strong>رصيد USDT:</strong> <span>{{ "%.2f"|format(usdt_balance) }}</span></li>
                                    <li class="list-group-item d-flex justify-content-between"><strong>الصفقات المفتوحة:</strong> <span>{{ open_signals|length }} / {{ MAX_OPEN_TRADES }}</span></li>
                                    <li class="list-group-item d-flex justify-content-between"><strong>وضع التداول:</strong> <span>{{ "ورقي" if paper_trading_mode else "حقيقي" }}</span></li>
                                    <li class="list-group-item d-flex justify-content-between"><strong>حالة البوت:</strong> <span>{{ "يعمل" if is_trading_enabled else "متوقف" }}</span></li>
                                </ul>
                            </div>
                        </div>
                        <div class="card">
                            <div class="card-header"><h5>إعدادات الاستراتيجيات</h5></div>
                            <div class="card-body">
                                {% for key, name in STRATEGY_NAMES.items() %}
                                <div class="form-check form-switch mb-2">
                                    <input class="form-check-input strategy-toggle" type="checkbox" id="{{ key }}" data-strategy="USE_{{ key.upper() }}" {{ 'checked' if globals()['USE_' + key.upper()] else '' }}>
                                    <label class="form-check-label" for="{{ key }}">{{ name }}</label>
                                </div>
                                {% endfor %}
                                <button id="saveStrategySettings" class="btn btn-primary mt-2 w-100">حفظ الاستراتيجيات</button>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header"><h5>الصفقات المفتوحة</h5></div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-striped table-hover">
                                <thead><tr><th>العملة</th><th>الاستراتيجية</th><th>سعر الدخول</th><th>السعر الحالي</th><th>وقف الخسارة</th><th>ربح/خسارة</th><th>الحالة</th><th>النوع</th></tr></thead>
                                <tbody id="openSignalsTable">
                                    {% for signal in open_signals.values() %}
                                    <tr data-symbol="{{ signal.symbol }}">
                                        <td>{{ signal.symbol }}</td><td>{{ STRATEGY_NAMES.get(signal.strategy_name, signal.strategy_name) }}</td>
                                        <td>{{ "%.4f"|format(signal.entry_price) }}</td><td class="current-price" data-symbol="{{ signal.symbol }}">...</td>
                                        <td>{{ "%.4f"|format(signal.stop_loss) }}</td>
                                        <td class="profit-loss" data-entry="{{ signal.entry_price }}" data-symbol="{{ signal.symbol }}">...</td>
                                        <td><span class="badge bg-{{ 'info' if signal.status == 'open' else 'warning' }}">{{ signal.status }}</span></td>
                                        <td><span class="badge bg-{{ 'success' if signal.is_real_trade else 'secondary' }}">{{ "حقيقي" if signal.is_real_trade else "ورقي" }}</span></td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                        {% if not open_signals %}<div class="text-center text-muted">لا توجد صفقات مفتوحة</div>{% endif %}
                    </div>
                </div>
                
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header"><h5>آخر الإشعارات</h5></div>
                            <div class="card-body" style="max-height: 300px; overflow-y: auto;">
                                <ul class="list-group" id="notificationsList">
                                {% for n in notifications_cache %}<li class="list-group-item">{{ n.message }}<small class="text-muted float-start">{{ n.timestamp[:19] }}</small></li>{% endfor %}
                                </ul>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header"><h5>سجل الرفض</h5></div>
                            <div class="card-body" style="max-height: 300px; overflow-y: auto;">
                                <ul class="list-group" id="rejectionLogsList">
                                {% for log in rejection_logs_cache %}<li class="list-group-item"><strong>{{ log.symbol }}:</strong> {{ log.reason }}<small class="text-muted float-start">{{ log.timestamp[:19] }}</small></li>{% endfor %}
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div id="toast-container"></div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
            <script>
                // --- نظام الإشعارات المحسن ---
                function showToast(message, type = 'success') {
                    const container = document.getElementById('toast-container');
                    const toast = document.createElement('div');
                    toast.className = `toast-notification toast-${type}`;
                    toast.textContent = message;
                    container.appendChild(toast);
                    setTimeout(() => toast.classList.add('show'), 100);
                    setTimeout(() => {
                        toast.classList.remove('show');
                        setTimeout(() => container.removeChild(toast), 300);
                    }, 3000);
                }

                // --- WebSocket للتحديثات الفورية ---
                const ws = new WebSocket(`ws://${window.location.host}/ws`);
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'price_update') {
                        for (const [symbol, price] of Object.entries(data.payload)) {
                            document.querySelectorAll(`.current-price[data-symbol="${symbol}"]`).forEach(el => el.textContent = parseFloat(price).toFixed(4));
                            document.querySelectorAll(`.profit-loss[data-symbol="${symbol}"]`).forEach(el => {
                                const entry = parseFloat(el.dataset.entry);
                                const pnl = ((price - entry) / entry * 100).toFixed(2);
                                el.textContent = `${pnl}%`;
                                el.className = `profit-loss ${pnl > 0 ? 'profit' : 'loss'}`;
                            });
                        }
                    } else if (['new_notification', 'new_rejection'].includes(data.type)) {
                        const isNotification = data.type === 'new_notification';
                        const listId = isNotification ? 'notificationsList' : 'rejectionLogsList';
                        const list = document.getElementById(listId);
                        const newItem = document.createElement('li');
                        newItem.className = 'list-group-item';
                        const content = isNotification ? data.payload.message : `<strong>${data.payload.symbol}:</strong> ${data.payload.reason}`;
                        newItem.innerHTML = `${content}<small class="text-muted float-start">${data.payload.timestamp.substring(0, 19)}</small>`;
                        if (list.firstChild) list.insertBefore(newItem, list.firstChild);
                        else list.appendChild(newItem);
                        while (list.children.length > (isNotification ? 20 : 30)) list.removeChild(list.lastChild);
                    } else if (data.type === 'market_state_update') {
                        location.reload();
                    }
                };
                
                // --- حفظ الإعدادات ---
                document.getElementById('saveSettings').addEventListener('click', function() {
                    const settings = {
                        tradingEnabled: document.getElementById('tradingEnabled').checked,
                        paperTradingMode: document.getElementById('paperTradingMode').checked,
                        maxOpenTrades: parseInt(document.getElementById('maxOpenTrades').value),
                        minSignalQuality: parseInt(document.getElementById('minSignalQuality').value),
                        tradeAmountMin: parseFloat(document.getElementById('tradeAmountMin').value),
                        tradeAmountMax: parseFloat(document.getElementById('tradeAmountMax').value)
                    };
                    fetch('/api/settings', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(settings)})
                    .then(r => r.json()).then(d => {
                        if (d.success) { showToast('تم حفظ الإعدادات بنجاح', 'success'); setTimeout(() => location.reload(), 1500); } 
                        else { showToast('خطأ: ' + d.error, 'error'); }
                    }).catch(e => showToast('حدث خطأ غير متوقع', 'error'));
                });
                
                document.getElementById('saveStrategySettings').addEventListener('click', function() {
                    const settings = {};
                    document.querySelectorAll('.strategy-toggle').forEach(t => settings[t.dataset.strategy] = t.checked);
                    fetch('/api/strategy-settings', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(settings)})
                    .then(r => r.json()).then(d => {
                        if (d.success) { showToast('تم حفظ إعدادات الاستراتيجيات', 'success'); }
                        else { showToast('خطأ: ' + d.error, 'error'); }
                    }).catch(e => showToast('حدث خطأ غير متوقع', 'error'));
                });
                
                document.getElementById('minSignalQuality').addEventListener('input', e => document.getElementById('minSignalQualityValue').textContent = e.target.value);
            </script>
        </body>
        </html>
    """, 
    is_trading_enabled=is_trading_enabled, paper_trading_mode=paper_trading_mode,
    MAX_OPEN_TRADES=MAX_OPEN_TRADES, MIN_SIGNAL_QUALITY=MIN_SIGNAL_QUALITY,
    FIXED_TRADE_AMOUNT_MIN_USDT=FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT=FIXED_TRADE_AMOUNT_MAX_USDT,
    usdt_balance=usdt_balance, open_signals=open_signals_cache, STRATEGY_NAMES=STRATEGY_NAMES,
    globals=globals(), notifications_cache=list(notifications_cache), rejection_logs_cache=list(rejection_logs_cache),
    current_market_state=current_market_state
)

# **FIX**: Improved logic for updating settings and sending notifications
@app.route('/api/settings', methods=['POST'])
def update_settings():
    global is_trading_enabled, paper_trading_mode, MAX_OPEN_TRADES, MIN_SIGNAL_QUALITY, FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT
    try:
        data = request.json
        with trading_status_lock: old_trading_status = is_trading_enabled
        new_trading_status = data.get('tradingEnabled', old_trading_status)
        
        with trading_status_lock: is_trading_enabled = new_trading_status
        with trading_mode_lock: paper_trading_mode = data.get('paperTradingMode', paper_trading_mode)
        MAX_OPEN_TRADES = data.get('maxOpenTrades', MAX_OPEN_TRADES)
        with min_quality_lock: MIN_SIGNAL_QUALITY = data.get('minSignalQuality', MIN_SIGNAL_QUALITY)
        with trade_amount_lock:
            FIXED_TRADE_AMOUNT_MIN_USDT = data.get('tradeAmountMin', FIXED_TRADE_AMOUNT_MIN_USDT)
            FIXED_TRADE_AMOUNT_MAX_USDT = data.get('tradeAmountMax', FIXED_TRADE_AMOUNT_MAX_USDT)
        
        save_settings_to_redis()
        
        if new_trading_status != old_trading_status:
            status = "مفعّل" if new_trading_status else "معطّل"
            log_and_notify("info", f"تم {status} التداول من لوحة التحكم", "trading_status")
            send_bot_status_notification()
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error updating settings: {e}"); return jsonify({'success': False, 'error': str(e)})

@app.route('/api/strategy-settings', methods=['POST'])
def update_strategy_settings():
    global USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY, USE_MOMENTUM_VOLATILITY_STRATEGY, USE_ELLIOTT_WAVE_STRATEGY, USE_RANGE_REVERSAL_STRATEGY
    try:
        data = request.json
        USE_BB_STOCH_STRATEGY = data.get('USE_BB_STOCH_STRATEGY', USE_BB_STOCH_STRATEGY)
        USE_MACD_EMA_STRATEGY = data.get('USE_MACD_EMA_STRATEGY', USE_MACD_EMA_STRATEGY)
        USE_EMA_RSI_STRATEGY = data.get('USE_EMA_RSI_STRATEGY', USE_EMA_RSI_STRATEGY)
        USE_PULLBACK_STRATEGY = data.get('USE_PULLBACK_STRATEGY', USE_PULLBACK_STRATEGY)
        USE_MOMENTUM_VOLATILITY_STRATEGY = data.get('USE_MOMENTUM_VOLATILITY_STRATEGY', USE_MOMENTUM_VOLATILITY_STRATEGY)
        USE_ELLIOTT_WAVE_STRATEGY = data.get('USE_ELLIOTT_WAVE_STRATEGY', USE_ELLIOTT_WAVE_STRATEGY)
        USE_RANGE_REVERSAL_STRATEGY = data.get('USE_RANGE_REVERSAL_STRATEGY', USE_RANGE_REVERSAL_STRATEGY)
        save_settings_to_redis()
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error updating strategy settings: {e}"); return jsonify({'success': False, 'error': str(e)})

@sock.route('/ws')
def websocket_connection(ws):
    with ws_clients_lock: ws_clients.append(ws)
    try:
        while True:
            if ws.receive(timeout=10) is None: continue
    except Exception: pass
    finally:
        with ws_clients_lock:
            if ws in ws_clients: ws_clients.remove(ws)

def main():
    global client, validated_symbols_to_scan
    init_db()
    init_redis()
    load_settings_from_redis()
    
    try:
        client = Client(API_KEY, API_SECRET)
        logger.info("✅ [API] Binance client initialized successfully.")
    except Exception as e:
        logger.error(f"❌ [API] Failed to initialize Binance client: {e}"); exit(1)
    
    get_exchange_info_map()
    validated_symbols_to_scan = get_validated_symbols()
    if not validated_symbols_to_scan: logger.error("❌ No valid symbols found. Exiting."); exit(1)
    
    load_open_signals_to_cache()
    load_notifications_to_cache()
    update_balance()
    start_websocket()
    start_periodic_reports()
    
    Thread(target=lambda: app.run(host='0.0.0.0', port=5000, threaded=True), daemon=True).start()
    logger.info("✅ [Flask] Web interface started on http://localhost:5000")
    
    send_bot_status_notification()
    logger.info("✅ Bot started successfully. Entering main loop...")
    
    main_loop_counter = 0
    while True:
        try:
            if main_loop_counter % 10 == 0: # Update balance every 5 minutes (30s * 10)
                update_balance()
            
            check_and_generate_signals()
            monitor_open_trades()
            
            main_loop_counter += 1
            time.sleep(30)
        except KeyboardInterrupt:
            logger.info("🛑 Bot stopped by user."); break
        except Exception as e:
            logger.error(f"❌ Error in main loop: {e}", exc_info=True); time.sleep(60)

if __name__ == "__main__":
    main()
