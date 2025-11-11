# ملف crypto_bot_enhanced.py - (V35.0.0 محسّن ومصحّح)
# --- التحسينات الرئيسية:
# 1. إصلاح نظام تتبع وحفظ الإشارات المفتوحة
# 2. تخفيف صرامة استراتيجية RSI وإضافة استراتيجيات جديدة
# 3. إضافة 4 استراتيجيات تداول شائعة وفعالة لفريم 5 دقائق
# 4. تحسين لوحة التحكم مع مؤشرات أداء حقيقية
# 5. إضافة فلاتر ذكية ونظام جودة ديناميكي
# 6. تحسين آليات إدارة الصفقات والإغلاق

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
        logging.FileHandler('crypto_bot_v35_5min_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotRSI_5min')

# --- المتغيرات العامة والإعدادات ---
API_KEY: str = config('BINANCE_API_KEY')
API_SECRET: str = config('BINANCE_API_SECRET')
DB_URL: str = config('DATABASE_URL')
REDIS_URL: str = config('REDIS_URL', default='redis://localhost:6379/0')
TELEGRAM_BOT_TOKEN: str = config('TELEGRAM_BOT_TOKEN', default='')
TELEGRAM_CHAT_ID: str = config('TELEGRAM_CHAT_ID', default='')

# --- متغيرات الحالة ---
is_trading_enabled: bool = False
trading_status_lock = Lock()
paper_trading_mode: bool = True
trading_mode_lock = Lock()
usdt_balance: float = 1000.0  # رصيد افتراضي للورقي
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
MIN_SIGNAL_QUALITY: int = 65  # تم خفضه من 70
AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE: bool = True
min_quality_lock = Lock()

# --- مفاتيح تفعيل الاستراتيجيات ---
STRATEGY_NAMES = {
    "RSI_Enhanced_Strategy": "RSI المتقدم (ذروة البيع)",
    "EMA_Crossover_Strategy": "تقاطع EMA21/50",
    "MACD_Momentum_Strategy": "زخم MACD الصاعد",
    "Stochastic_Strategy": "ستوكاستيك ذروة البيع",
    "Bollinger_Bands_Strategy": "اختراق بولينجر لأسفل",
    "Multi_Indicator_Fusion": "مزيج المؤشرات الذكي"
}
strategy_filters_lock = Lock()

# --- إعدادات عامة ---
SIGNAL_GENERATION_TIMEFRAME: str = '5m'
HIGHER_TIMEFRAME: str = '15m'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['5m', '15m', '1h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 7
BTC_SYMBOL: str = 'BTCUSDT'
API_REQUEST_DELAY: float = 0.5  # تم تقليله لتحسين السرعة

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
notifications_cache = deque(maxlen=50)  # زيادة من 20
rejection_logs_cache = deque(maxlen=50)  # زيادة من 30
current_market_state: Dict[str, Any] = {"trend_details_by_tf": {}}
market_state_lock = Lock()

# --- متغيرات إعدادات الاستراتيجية ---
ENABLE_EMA_FILTER: bool = True
ENABLE_MACD_CONFIRMATION: bool = True
ENABLE_MFI_FILTER: bool = True
ENABLE_CANDLESTICK_PATTERNS: bool = True
REQUIRED_CONFIRMATIONS: int = 3
strategy_config_lock = Lock()

# --- قاموس أسباب الرفض باللغة العربية (محدث) ---
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
    "Liquidity Filter Failed": "فلتر السيولة: تجنب التداول في أوقات السيولة المنخفظة",
    "Correlation Filter Failed": "فلتر الارتباط: توجد صفقة مفتوحة على عملة مرتبطة",

    # Strategy Specific Rejections
    "RSI: No bullish recovery": "RSI: لم يحدث تعافٍ صعودي من ذروة البيع",
    "RSI: Not oversold": "RSI: لم يصل إلى منطقة ذروة البيع",
    "EMA: No crossover": "EMA: لم يحدث تقاطع صعودي",
    "MACD: No momentum": "MACD: لا يوجد زخم صعودي",
    "Stochastic: No crossover": "ستوكاستيك: لم يحدث تقاطع صعودي",
    "BB: No bounce": "بولينجر: لم يحدث ارتداد من الفرقة السفلى",
    "Fusion: Insufficient confirmations": "المزيج: تأكيدات غير كافية",
    "Multiple conditions failed": "عدة شروط تأكيد فشلت"
}

# --- إعداد تطبيق Flask و WebSocket ---
app = Flask(__name__)
CORS(app)
sock = Sock(app)
ws_clients: List[Any] = []
ws_clients_lock = Lock()

# --- دوال WebSocket ---
def broadcast(data: Dict):
    """بث البيانات لجميع عملاء WebSocket مع معالجة الأخطاء"""
    with ws_clients_lock:
        if not ws_clients:
            logger.debug("[WebSocket] No clients connected to broadcast")
            return
        
        clients_to_remove = []
        for i, client in enumerate(ws_clients):
            try:
                client.send(json.dumps(data, cls=NpEncoder))
                logger.debug(f"[WebSocket] Sent {data['type']} to client {i}")
            except Exception as e:
                logger.warning(f"[WebSocket] Send failed to client {i}, removing: {e}")
                clients_to_remove.append(client)
        
        for client in clients_to_remove:
            try:
                ws_clients.remove(client)
                logger.info("[WebSocket] Removed disconnected client")
            except ValueError:
                pass

def get_dashboard_payload() -> Dict:
    """جلب جميع بيانات لوحة التحكم مع التحقق من الاتصال بقاعدة البيانات"""
    try:
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
        with strategy_config_lock:
            ema_filter = ENABLE_EMA_FILTER
            macd_confirm = ENABLE_MACD_CONFIRMATION
            mfi_filter = ENABLE_MFI_FILTER
            candle_confirm = ENABLE_CANDLESTICK_PATTERNS
            req_confirm = REQUIRED_CONFIRMATIONS

        logger.info(f"[Dashboard] Fetching {len(open_signals_cache)} open signals from cache")
        
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
            "strategy_config": {
                "enable_ema_filter": ema_filter,
                "enable_macd_confirmation": macd_confirm,
                "enable_mfi_filter": mfi_filter,
                "enable_candlestick_patterns": candle_confirm,
                "required_confirmations": req_confirm
            },
            "server_time": datetime.now(timezone.utc).isoformat(),
            "open_signals_cache_count": len(open_signals_cache),
            "active_strategies": list(STRATEGY_NAMES.keys())
        }
    except Exception as e:
        logger.error(f"❌ [Dashboard] Error generating data: {e}", exc_info=True)
        return {"error": str(e)}

# --- دوال المساعدة ---
def optimize_database():
    """تحسين قاعدة البيانات بإضافة الفهارس"""
    if not check_db_connection() or not conn:
        logger.warning("[DB] Connection not available for optimization")
        return
    try:
        with conn.cursor() as cur:
            logger.info("[DB] Optimizing database with indexes...")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol_status ON signals(symbol, status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_notifications_timestamp ON notifications(timestamp);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_status_closed_at ON signals(status, closed_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy_name);")
            conn.commit()
            logger.info("✅ [DB] Database indexes optimized successfully.")
    except Exception as e:
        logger.error(f"❌ [DB] Error optimizing database: {e}")
        if conn: conn.rollback()

def column_exists(cursor, table_name, column_name):
    """التحقق من وجود عمود في جدول"""
    cursor.execute("SELECT 1 FROM information_schema.columns WHERE table_name = %s AND column_name = %s", (table_name, column_name))
    return cursor.fetchone() is not None

def init_db(retries: int = 5, base_delay: int = 5) -> None:
    """تهيئة اتصال قاعدة البيانات مع إعادة محاولة تلقائية"""
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
                # إنشاء الجداول الأساسية
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION NOT NULL,
                        stop_loss DOUBLE PRECISION NOT NULL, status TEXT DEFAULT 'open',
                        closing_price DOUBLE PRECISION, closed_at TIMESTAMP, profit_percentage DOUBLE PRECISION,
                        strategy_name TEXT, signal_details JSONB, is_real_trade BOOLEAN DEFAULT FALSE,
                        quantity DOUBLE PRECISION, closing_reason TEXT, order_id TEXT,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(), type TEXT NOT NULL, message TEXT NOT NULL
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS performance_summary (
                        id SERIAL PRIMARY KEY,
                        trade_id INTEGER REFERENCES signals(id),
                        profit_percentage DOUBLE PRECISION,
                        drawdown DOUBLE PRECISION,
                        date DATE
                    );
                """)
                
                # إضافة الأعمدة المفقودة
                columns_to_add = {
                    "target_price_1": "DOUBLE PRECISION",
                    "target_price_2": "DOUBLE PRECISION",
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
                delay = base_delay * (2  ** attempt)
                logger.info(f"[DB] Retrying connection in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.critical(" ❌ [DB] Failed to connect after all retries. Exiting.")
                exit(1)

def check_db_connection() -> bool:
    """التحقق من صحة اتصال قاعدة البيانات مع إعادة الاتصال التلقائي"""
    global conn
    if conn is None or conn.closed != 0:
        logger.warning("[DB] Connection is None or closed. Will attempt to reconnect...")
        try:
            init_db(retries=3, base_delay=2)
        except:
            logger.error(" [DB] Reconnection attempt failed.")
            return False
    
    try:
        if conn and conn.closed == 0:
            with conn.cursor() as cur: cur.execute("SELECT 1;")
            return True
        logger.warning("[DB] Connection check failed.")
        return False
    except (OperationalError, InterfaceError) as e:
        logger.error(f" [DB] Connection lost ({e}). Attempting to reconnect...")
        try:
            init_db(retries=3, base_delay=2)
        except:
            logger.error("❌ [DB] Reconnection attempt failed.")
        return conn is not None and conn.closed == 0

def init_redis() -> None:
    """تهيئة اتصال Redis"""
    global redis_client
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] Connected successfully.")
    except redis.exceptions.ConnectionError as e:
        logger.warning(f"⚠️ [Redis] Connection failed: {e}.")
        redis_client = None

def log_and_notify(level: str, message: str, notification_type: str):
    """تسجيل وإخطار الأحداث مع التحقق من الاتصال"""
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error}
    log_methods.get(level.lower(), logger.info)(message)
    
    if not check_db_connection() or not conn:
        logger.warning("[Log & Notify] Skipping DB save due to connection issues")
        return
    
    try:
        new_notification = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": notification_type,
            "message": message
        }
        with notifications_lock: notifications_cache.appendleft(new_notification)
        
        with conn.cursor() as cur:
            cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
        
        broadcast({"type": "new_notification", "payload": new_notification})
        logger.debug(f"[WebSocket] Broadcasted notification: {notification_type}")
    except Exception as e:
        logger.error(f"❌ [DB] Failed to save notification: {e}")
        if conn: conn.rollback()

def log_rejection(symbol: str, reason_key: str, details: Optional[Dict] = None):
    """تسجيل أسباب رفض الإشارات"""
    try:
        reason_ar = REJECTION_REASONS_AR.get(reason_key, reason_key)
        if details:
            details_str = ", ".join([f"{k}: {v}" for k, v in details.items()])
            reason_ar = f"{reason_ar} ({details_str})"
        
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "reason": reason_ar
        }
        
        with rejection_logs_lock:
            rejection_logs_cache.appendleft(log_entry)
        
        broadcast({"type": "new_rejection", "payload": log_entry})
        logger.debug(f"[Rejection] {symbol}: {reason_ar}")
    except Exception as e:
        logger.error(f"❌ [Log Rejection] Error logging rejection for {symbol}: {e}", exc_info=True)

def get_notification_settings() -> Dict:
    """جلب إعدادات الإشعارات من Redis"""
    defaults = {'telegram_enabled': True, 'email_enabled': False, 'min_profit_notification': 0.5, 'max_loss_notification': -0.5}
    if not redis_client: return defaults
    
    try:
        settings_data = redis_client.get('notification_settings')
        if settings_data:
            settings = json.loads(settings_data)
            for key, value in defaults.items(): settings.setdefault(key, value)
            return settings
        return defaults
    except Exception as e:
        logger.error(f"❌ [Redis] Failed to get notification settings: {e}")
        return defaults

def send_enhanced_telegram_message(message: str, force: bool = False):
    """إرسال رسائل Telegram مع معالجة الأخطاء"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("[Telegram] Tokens not configured")
        return
    
    settings = get_notification_settings()
    if not settings.get('telegram_enabled') and not force:
        return
    
    max_length = 4096
    messages = [message[i:i+max_length] for i in range(0, len(message), max_length)]
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    for i, msg in enumerate(messages):
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        for attempt in range(3):
            try:
                r = requests.post(url, data=payload, timeout=10)
                if r.status_code == 429:
                    retry_after = int(r.json().get("parameters", {}).get("retry_after", 1))
                    time.sleep(min(5, retry_after))
                    continue
                if r.ok:
                    logger.info(f"[Telegram] Message {i+1}/{len(messages)} sent successfully")
                    break
                else:
                    logger.warning(f"[Telegram] HTTP {r.status_code}: {r.text}")
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    logger.error(f"❌ [Telegram] Failed to send message after retries: {e}")
                time.sleep(1.5)

def send_trade_open_notification(symbol: str, strategy_name: str, entry_price: float, stop_loss: float,
                                target1: float, target2: float, quantity: float, is_real: bool,
                                quality_score: int, atr_percent: float, notional_value: float, confirmations: Dict):
    """إشعار بفتح صفقة جديدة"""
    trade_type = "حقيقية" if is_real else "ورقية"
    emoji = "🔥" if is_real else "📊"
    
    confirms_list = "\n".join([f"✅ {k.replace('_', ' ').title()}: {'نعم' if v else 'لا'}" for k, v in confirmations.items()])
    
    message = (
        f"{emoji} *صفقة {trade_type} جديدة (5 دقائق)*\n\n"
        f"*العملة:* `{symbol}`\n"
        f"*الاستراتيجية:* `{STRATEGY_NAMES.get(strategy_name, strategy_name)}`\n"
        f"*جودة الإشارة:* `{quality_score}/100`\n"
        f"*تقلب السوق:* `{atr_percent:.2f}%`\n\n"
        f"*التأكيدات:*\n{confirms_list}\n\n"
        f"*سعر الدخول:* `{entry_price:.6f}`\n"
        f"*وقف الخسارة:* `{stop_loss:.6f}`\n"
        f"*الهدف الأول:* `{target1:.6f}`\n"
        f"*الهدف الثاني:* `{target2:.6f}`\n\n"
        f"*الكمية:* `{quantity:.6f}`\n"
        f"*قيمة الصفقة:* `${notional_value:.2f}`\n"
        f"*نسبة المخاطرة:* `{((entry_price - stop_loss) / entry_price * 100):.2f}%`\n"
        f"*نسبة الربح المحتملة 1:* `{((target1 - entry_price) / entry_price * 100):.2f}%`\n"
        f"*نسبة الربح المحتملة 2:* `{((target2 - entry_price) / entry_price * 100):.2f}%`"
    )
    
    send_enhanced_telegram_message(message, force=True)

# --- الفلاتر الديناميكية ونظام السوق ---
def check_market_volatility_filter_enhanced(df: pd.DataFrame, symbol: str = "Unknown") -> bool:
    """فلتر تقلب السوق مع تسجيل واضح"""
    if 'atr_percent' not in df.columns or df['atr_percent'].isnull().all():
        log_rejection(symbol, "Market Volatility Filter Failed", {"reason": "No ATR data"})
        return False
    
    last_atr_percent = float(df.iloc[-1].get('atr_percent', 0))
    ATR_PERCENT_MIN = 0.25  # تخفيف من 0.35
    ATR_PERCENT_MAX = 4.0   # زيادة من 3.2
    
    if not (ATR_PERCENT_MIN <= last_atr_percent <= ATR_PERCENT_MAX):
        log_rejection(symbol, "Market Volatility Filter Failed", {
            "atr": f"{last_atr_percent:.2f}%",
            "range": f"({ATR_PERCENT_MIN:.2f}-{ATR_PERCENT_MAX:.2f})%"
        })
        return False
    
    logger.debug(f"[Volatility Filter] {symbol} passed: {last_atr_percent:.2f}%")
    return True

def calculate_dynamic_stop_loss(df: pd.DataFrame, entry_price: float, strategy_name: str) -> float:
    """حساب وقف الخسارة الديناميكي"""
    last = df.iloc[-1]
    atr_value = last.get('atr', 0)
    
    stop_loss = entry_price - (atr_value * 1.8)  # تخفيف من 2.0
    
    if strategy_name == "RSI_Enhanced_Strategy":
        recent_low = df['low'].tail(7).min()  # زيادة من 5
        stop_loss = min(recent_low * 0.996, entry_price - (atr_value * 1.6))  # تخفيف من 1.8
    
    max_stop_distance = entry_price * 0.06  # زيادة من 0.05
    if entry_price - stop_loss > max_stop_distance:
        stop_loss = entry_price - max_stop_distance
    
    return round(stop_loss, 6)

def calculate_dynamic_take_profit(df: pd.DataFrame, entry_price: float, stop_loss: float, strategy_name: str) -> tuple:
    """حساب أهداف الربح الديناميكية"""
    risk_amount = entry_price - stop_loss
    if risk_amount <= 0: return (entry_price * 1.012, entry_price * 1.025)  # زيادة طفيفة

    rr1, rr2 = 1.5, 2.5  # تخفيف من 1.8, 3.0
    target1 = entry_price + (risk_amount * rr1)
    target2 = entry_price + (risk_amount * rr2)
    
    if 'r1' in df.columns:
        r1 = df.iloc[-1].get('r1', target1)
        r2 = df.iloc[-1].get('r2', target2)
        target1 = max(target1, r1 * 0.985)  # تعديل من 0.98
        target2 = max(target2, r2 * 0.985)
    
    return round(target1, 6), round(target2, 6)

# --- استراتيجيات التداول (5 جديدة + RSI المحسن) ---
def check_rsi_enhanced_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """
    استراتيجية RSI المُحسّنة - أقل صرامة
    الشرط: RSI في منطقة ذروة البيع (<40) وفي اتجاه صعودي
    """
    if len(df) < 50:
        logger.debug(f"[RSI Strategy] {df.name if hasattr(df, 'name') else 'Unknown'}: Insufficient data ({len(df)} < 50)")
        return False

    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    confirmations = {
        'rsi_oversold_recovery': False,
        'ema_trend': False,
        'macd_momentum': False,
        'mfi_pressure': False,
        'candlestick_bullish': False
    }
    
    # 1. شرط RSI الأساسي - تخفيف الصرامة
    is_oversold = prev['rsi'] < 40  # تغيير من <30
    is_rising = last['rsi'] > prev['rsi']
    confirmations['rsi_oversold_recovery'] = is_oversold and is_rising
    
    # 2. فلتر EMA21
    with strategy_config_lock:
        ema_enabled = ENABLE_EMA_FILTER
    if ema_enabled:
        confirmations['ema_trend'] = last['close'] > last['ema21']
    else:
        confirmations['ema_trend'] = True
    
    # 3. تأكيد MACD
    with strategy_config_lock:
        macd_enabled = ENABLE_MACD_CONFIRMATION
    if macd_enabled:
        confirmations['macd_momentum'] = last['macd_hist'] > 0
    else:
        confirmations['macd_momentum'] = True
    
    # 4. فلتر MFI
    with strategy_config_lock:
        mfi_enabled = ENABLE_MFI_FILTER
    if mfi_enabled:
        confirmations['mfi_pressure'] = last['mfi'] > 25  # تغيير من >20
    else:
        confirmations['mfi_pressure'] = True
    
    # 5. نمط الشموع
    with strategy_config_lock:
        candle_enabled = ENABLE_CANDLESTICK_PATTERNS
    if candle_enabled:
        confirmations['candlestick_bullish'] = detect_bullish_candlestick_pattern(df)
    else:
        confirmations['candlestick_bullish'] = True
    
    # حساب عدد التأكيدات
    active_confirmations = sum(confirmations.values())
    
    with strategy_config_lock:
        required = REQUIRED_CONFIRMATIONS
    
    # تسجيل أسباب الرفض (محدث)
    if not confirmations['rsi_oversold_recovery']:
        if not is_oversold:
            log_rejection(df.name if hasattr(df, 'name') else 'Unknown', "RSI: Not oversold", {
                "rsi": f"{last['rsi']:.1f}",
                "required": "<40"
            })
        else:
            log_rejection(df.name if hasattr(df, 'name') else 'Unknown', "RSI: No bullish recovery", {
                "rsi_change": f"{last['rsi'] - prev['rsi']:.1f}"
            })
        return False
    
    if active_confirmations < required:
        failed_conditions = [k for k, v in confirmations.items() if not v]
        log_rejection(df.name if hasattr(df, 'name') else 'Unknown', 
                     f"Insufficient confirmations ({active_confirmations}/{required})", 
                     {"failed": ', '.join(failed_conditions)})
        return False
    
    logger.info(f"✅ [Enhanced RSI Signal] {df.name if hasattr(df, 'name') else 'Unknown'}: {active_confirmations}/{required} confirmations met")
    df.confirmations = confirmations
    return True

def check_ema_crossover_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """
    استراتيجية تقاطع EMA: EMA21 يعبر فوق EMA50
    """
    if len(df) < 60:
        return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # تأكيدات
    confirmations = {
        'ema_crossover': prev['ema21'] <= prev['ema50'] and last['ema21'] > last['ema50'],
        'price_above_ema21': last['close'] > last['ema21'],
        'macd_positive': last['macd_hist'] > 0,
        'volume_increase': last['volume'] > df['volume'].tail(20).mean(),
        'rsi_support': last['rsi'] < 65  # ليس في ذروة الشراء
    }
    
    active_confirmations = sum(confirmations.values())
    
    if not confirmations['ema_crossover']:
        log_rejection(df.name if hasattr(df, 'name') else 'Unknown', "EMA: No crossover")
        return False
    
    if active_confirmations < 3:  # 3 من 5 تأكيدات
        return False
    
    logger.info(f"✅ [EMA Crossover Signal] {df.name if hasattr(df, 'name') else 'Unknown'}")
    df.confirmations = confirmations
    return True

def check_macd_momentum_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """
    استراتيجية زخم MACD: MACD histogram يتحول إيجابي ومتزايد
    """
    if len(df) < 50:
        return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # تأكيدات
    confirmations = {
        'macd_histogram_positive': last['macd_hist'] > 0,
        'macd_histogram_increasing': last['macd_hist'] > prev['macd_hist'],
        'macd_signal_crossover': last['macd'] > last['macd_signal'],
        'price_trend': last['close'] > last['ema21'],
        'rsi_support': last['rsi'] > 45
    }
    
    active_confirmations = sum(confirmations.values())
    
    if not confirmations['macd_histogram_positive'] or not confirmations['macd_histogram_increasing']:
        log_rejection(df.name if hasattr(df, 'name') else 'Unknown', "MACD: No bullish momentum")
        return False
    
    if active_confirmations < 3:  # 3 من 5 تأكيدات
        return False
    
    logger.info(f"✅ [MACD Momentum Signal] {df.name if hasattr(df, 'name') else 'Unknown'}")
    df.confirmations = confirmations
    return True

def check_stochastic_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """
    استراتيجية ستوكاستيك: %K يعبر فوق %D في منطقة ذروة البيع
    """
    if len(df) < 20:
        return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # تأكيدات
    confirmations = {
        'stochastic_crossover': prev['stoch_k'] <= prev['stoch_d'] and last['stoch_k'] > last['stoch_d'],
        'stochastic_oversold': prev['stoch_k'] < 30,
        'rsi_support': last['rsi'] > 40,
        'price_above_ema': last['close'] > last['ema21'],
        'volume_check': last['volume'] > df['volume'].tail(20).mean()
    }
    
    active_confirmations = sum(confirmations.values())
    
    if not confirmations['stochastic_crossover'] or not confirmations['stochastic_oversold']:
        log_rejection(df.name if hasattr(df, 'name') else 'Unknown', "Stochastic: No crossover")
        return False
    
    if active_confirmations < 3:  # 3 من 5 تأكيدات
        return False
    
    logger.info(f"✅ [Stochastic Signal] {df.name if hasattr(df, 'name') else 'Unknown'}")
    df.confirmations = confirmations
    return True

def check_bollinger_bands_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """
    استراتيجية بولينجر: السعر يلمس أو يخترق الفرقة السفلى ثم يرتد
    """
    if len(df) < 30:
        return False
    
    last = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]
    
    # تأكيدات
    confirmations = {
        'price_at_lower_band': prev1['close'] <= prev1['bb_lower'] or prev2['close'] <= prev2['bb_lower'],
        'price_bounce': last['close'] > prev1['close'],
        'rsi_recovery': last['rsi'] > prev1['rsi'],
        'volume_increase': last['volume'] > prev1['volume'],
        'bb_width_normal': prev1['bb_width'] > 0.02  # تقلبات كافية
    }
    
    active_confirmations = sum(confirmations.values())
    
    if not confirmations['price_at_lower_band'] or not confirmations['price_bounce']:
        log_rejection(df.name if hasattr(df, 'name') else 'Unknown', "BB: No bounce")
        return False
    
    if active_confirmations < 3:  # 3 من 5 تأكيدات
        return False
    
    logger.info(f"✅ [Bollinger Bands Signal] {df.name if hasattr(df, 'name') else 'Unknown'}")
    df.confirmations = confirmations
    return True

def check_multi_indicator_fusion_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """
    استراتيجية المزيج الذكي: تجمع أفضل المؤشرات معًا
    """
    if len(df) < 50:
        return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # نقاط القوة لكل مؤشر (0-100)
    rsi_score = max(0, 40 - last['rsi']) * 2.5 if last['rsi'] < 50 else 0
    macd_score = min(100, max(0, last['macd_hist'] / abs(last['macd_signal']) * 50)) if last['macd_signal'] != 0 else 0
    mfi_score = (last['mfi'] - 20) * 1.25 if last['mfi'] > 20 else 0
    ema_score = 100 if last['close'] > last['ema21'] else 0
    stoch_score = max(0, 30 - last['stoch_k']) * 3.33 if last['stoch_k'] < 50 else 0
    
    # مجموع النقاط
    total_score = rsi_score + macd_score + mfi_score + ema_score + stoch_score
    
    # تأكيدات
    confirmations = {
        'rsi_favorable': last['rsi'] < 50,
        'macd_positive': last['macd_hist'] > 0,
        'mfi_support': last['mfi'] > 25,
        'price_above_ema': last['close'] > last['ema21'],
        'stochastic_favorable': last['stoch_k'] < 60
    }
    
    active_confirmations = sum(confirmations.values())
    
    # الحد الأدنى للنقاط والتأكيدات
    if total_score < 150 or active_confirmations < 3:
        log_rejection(df.name if hasattr(df, 'name') else 'Unknown', 
                     f"Fusion: Insufficient confirmations ({active_confirmations}/3, score: {total_score}/500)")
        return False
    
    logger.info(f"✅ [Multi-Indicator Fusion Signal] {df.name if hasattr(df, 'name') else 'Unknown'} - Score: {total_score}/500")
    df.confirmations = confirmations
    return True

def detect_bullish_candlestick_pattern(df: pd.DataFrame) -> bool:
    """كشف أنماط الشموع الصعودية (مُحسّن)"""
    if len(df) < 3:
        return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Hammer pattern
    body = abs(last['close'] - last['open'])
    lower_shadow = min(last['open'], last['close']) - last['low']
    upper_shadow = last['high'] - max(last['open'], last['close'])
    is_hammer = body > 0 and lower_shadow > 1.8 * body and upper_shadow < 0.4 * body
    
    # Bullish Engulfing
    prev_body = abs(prev['close'] - prev['open'])
    prev_is_red = prev['close'] < prev['open']
    current_is_green = last['close'] > last['open']
    is_engulfing = prev_is_red and current_is_green and last['close'] > prev['open'] and last['open'] < prev['close']
    
    # Morning Star (مُبسّط)
    is_morning_star = False
    if len(df) >= 3:
        day1 = df.iloc[-3]
        day2 = df.iloc[-2]
        day3 = df.iloc[-1]
        if day1['close'] < day1['open'] and abs(day2['close'] - day2['open']) < abs(day1['close'] - day1['open']) * 0.3 and day3['close'] > day3['open']:
            is_morning_star = True
    
    return is_hammer or is_engulfing or is_morning_star

# --- Data Loading & Settings Management ---
def load_open_signals_to_cache():
    """[محسّن] تحميل الإشارات المفتوحة من قاعدة البيانات إلى الكاش"""
    logger.info("[Cache] Starting to load open signals from database...")
    
    if not check_db_connection() or not conn:
        logger.error("[Cache] Cannot load signals: DB connection failed")
        return
    
    try:
        with conn.cursor() as cur:
            # استعلام محسّن لجلب جميع الإشارات المفتوحة
            cur.execute("""
                SELECT id, symbol, entry_price, target_price_1, target_price_2, stop_loss, 
                       strategy_name, is_real_trade, quantity, initial_quantity, 
                       signal_details, status, order_id
                FROM signals 
                WHERE status IN ('open', 'updated')
                ORDER BY id DESC
            """)
            
            signals = cur.fetchall()
            logger.info(f"[Cache] Fetched {len(signals)} open signals from database")
            
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in signals:
                    # تحويل signal_details من string إلى dict إذا لزم
                    if isinstance(signal['signal_details'], str):
                        try:
                            signal['signal_details'] = json.loads(signal['signal_details'])
                        except:
                            signal['signal_details'] = {}
                    
                    open_signals_cache[signal['symbol']] = dict(signal)
                    logger.debug(f"[Cache] Loaded signal: {signal['symbol']} (ID: {signal['id']})")
            
            logger.info(f"✅ [Cache] Successfully loaded {len(open_signals_cache)} open signals to cache")
            broadcast({"type": "signals_loaded", "count": len(open_signals_cache)})
            
    except Exception as e:
        logger.error(f"❌ [Cache] Failed to load open signals: {e}", exc_info=True)
        if conn: conn.rollback()

def load_notifications_to_cache():
    """تحميل الإشعارات الأخيرة من قاعدة البيانات"""
    if not check_db_connection() or not conn:
        logger.warning("[Cache] Cannot load notifications: DB connection failed")
        return
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 50;")
            with notifications_lock:
                notifications_cache.clear()
                for n in reversed(cur.fetchall()):
                    if hasattr(n['timestamp'], 'isoformat'):
                        n['timestamp'] = n['timestamp'].isoformat()
                    notifications_cache.appendleft(dict(n))
        logger.info(f"✅ [Cache] Loaded {len(notifications_cache)} notifications")
    except Exception as e:
        logger.error(f"❌ [Cache] Failed to load notifications: {e}")

def load_settings_from_redis():
    """    تحميل الإعدادات من Redis"""
    global FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT, MAX_OPEN_TRADES, paper_trading_mode, MIN_SIGNAL_QUALITY
    global ENABLE_EMA_FILTER, ENABLE_MACD_CONFIRMATION, ENABLE_MFI_FILTER, ENABLE_CANDLESTICK_PATTERNS, REQUIRED_CONFIRMATIONS
    
    if not redis_client:
        logger.warning("[Redis] Client not available for loading settings")
        return
    
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
                MIN_SIGNAL_QUALITY = quality_settings.get('min_quality', 65)
        
        # تحميل إعدادات الاستراتيجية
        strategy_config_data = redis_client.get('strategy_config')
        if strategy_config_data:
            strategy_config = json.loads(strategy_config_data)
            with strategy_config_lock:
                ENABLE_EMA_FILTER = strategy_config.get('enable_ema_filter', True)
                ENABLE_MACD_CONFIRMATION = strategy_config.get('enable_macd_confirmation', True)
                ENABLE_MFI_FILTER = strategy_config.get('enable_mfi_filter', True)
                ENABLE_CANDLESTICK_PATTERNS = strategy_config.get('enable_candlestick_patterns', True)
                REQUIRED_CONFIRMATIONS = strategy_config.get('required_confirmations', 3)
        
        logger.info("✅ [Redis] Successfully loaded settings")
    except Exception as e:
        logger.error(f"❌ [Redis] Error loading settings: {e}")

def save_settings_to_redis():
    """حفظ الإعدادات في Redis"""
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
        
        strategy_config = {
            'enable_ema_filter': ENABLE_EMA_FILTER,
            'enable_macd_confirmation': ENABLE_MACD_CONFIRMATION,
            'enable_mfi_filter': ENABLE_MFI_FILTER,
            'enable_candlestick_patterns': ENABLE_CANDLESTICK_PATTERNS,
            'required_confirmations': REQUIRED_CONFIRMATIONS
        }
        redis_client.set('strategy_config', json.dumps(strategy_config))
        
        logger.info("✅ [Redis] Settings saved successfully")
        return True
    
    except Exception as e:
        logger.error(f"❌ [Redis] Error saving settings: {e}")
        return False

# --- الحسابات الفنية ---
def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """حساب جميع المؤشرات الفنية (مُحسّن)"""
    df_calc = df.copy()
    
    # SMA
    df_calc['sma7'] = df_calc['close'].rolling(window=7).mean()
    df_calc['sma200'] = df_calc['close'].rolling(window=200).mean()

    # EMA
    df_calc['ema9'] = df_calc['close'].ewm(span=9, adjust=False).mean()
    df_calc['ema13'] = df_calc['close'].ewm(span=13, adjust=False).mean()
    df_calc['ema21'] = df_calc['close'].ewm(span=21, adjust=False).mean()
    df_calc['ema34'] = df_calc['close'].ewm(span=34, adjust=False).mean()
    df_calc['ema50'] = df_calc['close'].ewm(span=50, adjust=False).mean()
    df_calc['ema100'] = df_calc['close'].ewm(span=100, adjust=False).mean()
    df_calc['ema200'] = df_calc['close'].ewm(span=200, adjust=False).mean()
    
    # ATR و ADX
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
    
    # RSI (14-period)
    delta = df_calc['close'].diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain_14 = gain.ewm(com=14 - 1, adjust=False).mean()
    avg_loss_14 = loss.ewm(com=14 - 1, adjust=False).mean()
    rs_14 = avg_gain_14 / avg_loss_14.replace(0, 1e-9)
    df_calc['rsi'] = 100.0 - (100.0 / (1.0 + rs_14))
    
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
    low_14 = df_calc['low'].rolling(14).min()
    high_14 = df_calc['high'].rolling(14).max()
    high_low_range = high_14 - low_14
    meaningful_range = high_low_range > (df_calc['close'] * 0.0001)
    df_calc['stoch_k'] = np.where(meaningful_range, 100 * ((df_calc['close'] - low_14) / high_low_range.replace(0, 1e-9)), 50)
    df_calc['stoch_d'] = df_calc['stoch_k'].rolling(3).mean()
    
    # VWAP
    df_calc['vwap'] = (df_calc['close'] * df_calc['volume']).cumsum() / df_calc['volume'].cumsum()
    
    # MFI
    typical_price = (df_calc['high'] + df_calc['low'] + df_calc['close']) / 3
    money_flow = typical_price * df_calc['volume']
    positive_flow = money_flow.where(typical_price.diff() > 0, 0)
    negative_flow = money_flow.where(typical_price.diff() < 0, 0)
    positive_flow_sum = positive_flow.rolling(14).sum()
    negative_flow_sum = negative_flow.rolling(14).sum()
    money_ratio = positive_flow_sum / negative_flow_sum.replace(0, 1e-9)
    df_calc['mfi'] = 100 - (100 / (1 + money_ratio))
    
    # Pivot Points
    df_calc['pivot'] = (df_calc['high'].shift(1) + df_calc['low'].shift(1) + df_calc['close'].shift(1)) / 3
    df_calc['r1'] = 2 * df_calc['pivot'] - df_calc['low'].shift(1)
    df_calc['s1'] = 2 * df_calc['pivot'] - df_calc['high'].shift(1)
    df_calc['r2'] = df_calc['pivot'] + (df_calc['high'].shift(1) - df_calc['low'].shift(1))
    df_calc['s2'] = df_calc['pivot'] - (df_calc['high'].shift(1) - df_calc['low'].shift(1))
    
    return df_calc

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """جلب البيانات التاريخية من Binance (مُحسّن)"""
    time.sleep(API_REQUEST_DELAY)
    try:
        klines = client.get_historical_klines(symbol, interval, f"{days} day ago UTC")
        if not klines:
            logger.warning(f"[Data] No klines returned for {symbol}")
            return None
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore'
        ])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        
        return df.dropna().astype(float)
    except Exception as e:
        logger.error(f"❌ [Data] Error fetching data for {symbol}: {e}")
        return None

# --- حساب حجم الصفقة ---
def get_formatted_quantity(symbol: str, quantity: Decimal) -> str:
    """تنسيق الكمية حسب متطلبات Binance"""
    try:
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info:
            logger.warning(f"[{symbol}] No exchange info for formatting")
            return f"{quantity.normalize()}"

        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if not lot_size_filter:
            logger.warning(f"[{symbol}] LOT_SIZE filter not found")
            return f"{quantity.normalize()}"
        
        step_size = Decimal(lot_size_filter['stepSize'])
        formatted_quantity = quantity.quantize(step_size, rounding=ROUND_DOWN)
        return f"{formatted_quantity.normalize()}"
    except Exception as e:
        logger.error(f"❌ [{symbol}] Error formatting quantity: {e}")
        return str(quantity)

def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> Optional[Decimal]:
    """تعديل الكمية لتتوافق مع LOT_SIZE"""
    try:
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info:
            logger.error(f"[{symbol}] No exchange info for LOT_SIZE")
            return None
        
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if not lot_size_filter:
            logger.warning(f"[{symbol}] LOT_SIZE filter not found")
            return Decimal(str(quantity))
        
        step_size = Decimal(lot_size_filter['stepSize'])
        min_qty = Decimal(lot_size_filter['minQty'])
        quantity_dec = Decimal(str(quantity))
        
        if quantity_dec < min_qty:
            log_rejection(symbol, "LOT_SIZE Filter Failed", {
                "reason": "Below minQty",
                "qty": f"{quantity_dec}",
                "min": f"{min_qty}"
            })
            return None
        
        adjusted_quantity = quantity_dec - (quantity_dec % step_size)
        if adjusted_quantity < min_qty:
            log_rejection(symbol, "LOT_SIZE Filter Failed", {
                "reason": "Adjusted below minQty",
                "qty": f"{adjusted_quantity}",
                "min": f"{min_qty}"
            })
            return None
        
        return adjusted_quantity
    except Exception as e:
        logger.error(f"❌ [{symbol}] CRITICAL ERROR adjusting quantity: {e}", exc_info=True)
        return None

def calculate_position_size(symbol: str, entry_price: float, available_balance: float, is_real: bool) -> Optional[Decimal]:
    """حساب حجم الصفقة مع معالجة الأخطاء الشاملة"""
    desired_usdt_amount = PAPER_TRADE_FIXED_AMOUNT_USDT if not is_real else random.uniform(FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT)
    
    try:
        dec_entry = Decimal(str(entry_price))
        if dec_entry <= 0:
            logger.error(f"[{symbol}] Invalid entry price")
            return None
        
        dec_balance = Decimal(str(available_balance))
        dec_desired_amount = Decimal(str(desired_usdt_amount))
        
        logger.info(f"[{symbol}] Position sizing: Desired=${dec_desired_amount:.2f}, Available=${dec_balance:.2f}")

        if is_real and dec_desired_amount > dec_balance:
            log_rejection(symbol, "Insufficient Balance", {
                "required": f"${dec_desired_amount:.2f}",
                "available": f"${dec_balance:.2f}"
            })
            return None

        initial_quantity = dec_desired_amount / dec_entry
        adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(initial_quantity))

        if adjusted_quantity is None or adjusted_quantity <= 0:
            logger.warning(f"[{symbol}] Quantity adjustment failed")
            adjusted_quantity = Decimal('0')

        notional_value = adjusted_quantity * dec_entry
        symbol_info = exchange_info_map.get(symbol)
        
        if symbol_info:
            min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] in ('MIN_NOTIONAL', 'NOTIONAL')), None)
            
            if min_notional_filter:
                min_notional_str = min_notional_filter.get('minNotional', min_notional_filter.get('notional', '5.0'))
                min_notional = Decimal(min_notional_str)
                
                if notional_value < min_notional:
                    logger.warning(f"[{symbol}] Notional ${notional_value:.2f} < min ${min_notional}")
                    
                    required_notional = min_notional * Decimal('1.01')
                    
                    if is_real and required_notional > dec_balance:
                        log_rejection(symbol, "Insufficient Balance", {
                            "reason": "Cannot meet min_notional",
                            "required": f"${required_notional:.2f}",
                            "available": f"${dec_balance:.2f}"
                        })
                        return None
                    
                    new_quantity = required_notional / dec_entry
                    adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(new_quantity))

                    if adjusted_quantity is None or adjusted_quantity <= 0:
                        log_rejection(symbol, "MinNotional Filter Failed", {
                            "reason": "Failed to adjust quantity"
                        })
                        return None

                    notional_value = adjusted_quantity * dec_entry
                    logger.info(f"[{symbol}] Adjusted for min_notional: {adjusted_quantity} (${notional_value:.2f})")

        if notional_value <= 0:
            log_rejection(symbol, "MinNotional Filter Failed", {
                "reason": "Final notional is zero"
            })
            return None

        if is_real and notional_value > dec_balance:
            log_rejection(symbol, "Insufficient Balance", {
                "required": f"{notional_value:.2f}",
                "available": f"${dec_balance:.2f}"
            })
            return None
        
        logger.info(f"[ {symbol}] Final quantity: {adjusted_quantity} (${notional_value:.2f})")
        return adjusted_quantity

    except Exception as e:
        logger.error(f"❌ [{symbol}] Unhandled exception in calculate_position_size: {e}", exc_info=True)
        return None

def calculate_dynamic_quality_score(df: pd.DataFrame, symbol: str) -> int:
    """[مُحسّن] حساب درجة جودة الإشارة"""
    if len(df) < 2:
        return 0
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    score = 0
    
    # 1. RSI Strength (تعديل)
    rsi_strength = 0
    if last['rsi'] < 40:
        rsi_strength = (40 - last['rsi']) * 2.5  # أعلى نقاط في مناطق البيع القصوى
    elif last['rsi'] > 50:
        rsi_strength = (last['rsi'] - 50) * 1.5
    score += rsi_strength
    
    # 2. EMA Distance (تعديل)
    ema_distance = ((last['close'] - last['ema21']) / last['ema21']) * 100
    if ema_distance > 0:
        ema_score = min(ema_distance * 3, 15)  # تقليل الحد الأقصى
        score += ema_score
    
    # 3. MACD Momentum (تعديل)
    macd_strength = last['macd_hist']
    if macd_strength > 0:
        macd_score = min(macd_strength / abs(last['macd_signal']) * 15, 20)  # تعديل
        score += macd_score
    
    # 4. MFI Volume Pressure (تعديل)
    if last['mfi'] > 20:
        mfi_score = min((last['mfi'] - 20) / 80 * 15, 15)  # تعديل
        score += mfi_score
    
    # 5. ATR Volatility (تعديل)
    atr_percent = last['atr_percent']
    if 0.4 <= atr_percent <= 1.8:  # توسيع النطاق
        score += 25
    elif atr_percent > 0.25 and atr_percent < 4.0:  # تعديل
        score += 15
    
    # 6. Volume Surge (جديد)
    avg_volume = df['volume'].tail(20).mean()
    volume_ratio = last['volume'] / avg_volume if avg_volume > 0 else 1
    if volume_ratio > 1.2:
        score += 10
    
    final_score = int(min(score, 100))
    logger.debug(f"[Quality Score] {symbol}: {final_score}/100")
    return final_score

# --- أنظمة الفلاتر ---
def add_news_filter() -> bool:
    """فلتر الأخبار (مُحسّن)"""
    news_hours = [(12, 30), (14, 0), (18, 30), (22, 0)]  # أوقات الأخبار الرئيسية
    now = datetime.now(timezone.utc)
    for hour, minute in news_hours:
        if now.hour == hour and abs(now.minute - minute) <= 20:  # تقليل من 15
            return False
    return True

def add_liquidity_filter() -> bool:
    """فلتر السيولة (مُحسّن)"""
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:  # عطلة نهاية الأسبوع
        return False
    
    # ساعات السيولة الضعيفة (تعديل)
    if (now.hour >= 23 or now.hour <= 3):
        return False
    
    return True

def add_correlation_filter(new_symbol: str) -> bool:
    """فلتر الارتباط (مُحسّن)"""
    correlated_groups = [
        {'BTCUSDT', 'ETHUSDT', 'BCHUSDT', 'LTCUSDT'},
        {'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'ATOMUSDT'},
        {'SOLUSDT', 'AVAXUSDT', 'MATICUSDT', 'FTMUSDT'},
        {'BNBUSDT', 'FTTUSDT', 'HTUSDT'},
    ]
    with signal_cache_lock:
        open_symbols = set(open_signals_cache.keys())
    
    if not open_symbols:
        return True
    
    for group in correlated_groups:
        if new_symbol in group and not open_symbols.isdisjoint(group):
            log_rejection(new_symbol, "Correlation Filter Failed", {
                "group": list(group & open_symbols)
            })
            return False
    
    return True

# --- إنشاء وإدارة الإشارات ---
def create_trade_signal(symbol: str, df: pd.DataFrame, strategy_name: str):
    """[مُحسّن] إنشاء إشارة تداول جديدة مع التحقق الكامل"""
    logger.info(f"🔍 [Signal] Processing potential signal for {symbol} (Strategy: {strategy_name})...")
    
    df.strategy = strategy_name 
    
    # تطبيق الفلاتر
    if not check_market_volatility_filter_enhanced(df, symbol):
        logger.debug(f"[Signal] {symbol} failed volatility filter")
        return
    
    if not add_news_filter():
        log_rejection(symbol, "News Filter Failed")
        return
    
    if not add_liquidity_filter():
        log_rejection(symbol, "Liquidity Filter Failed")
        return
    
    if not add_correlation_filter(symbol):
        log_rejection(symbol, "Correlation Filter Failed")
        return

    # حساب جودة الإشارة
    quality_score = calculate_dynamic_quality_score(df, symbol)
    
    with min_quality_lock:
        min_score = MIN_SIGNAL_QUALITY
    
    if quality_score < min_score:
        log_rejection(symbol, "Low Quality Signal", {
            "score": quality_score,
            "min_required": min_score
        })
        return
    
    logger.info(f"⭐ [Signal Quality] {symbol}: {quality_score}/100 (min: {min_score})")

    entry_price = df.iloc[-1]['close']
    stop_loss_price = calculate_dynamic_stop_loss(df, entry_price, strategy_name)
    target_price_1, target_price_2 = calculate_dynamic_take_profit(df, entry_price, stop_loss_price, strategy_name)
    
    if stop_loss_price >= entry_price:
        log_rejection(symbol, "Invalid Position Size", {
            "entry": entry_price,
            "sl": stop_loss_price
        })
        return

    with trading_status_lock:
        if not is_trading_enabled:
            logger.warning(f"[Signal] Trading disabled, skipping {symbol}")
            return

    with trading_mode_lock:
        is_real = not paper_trading_mode
    
    confirmations = getattr(df, 'confirmations', {})
    
    signal_details = {
        "atr": df.iloc[-1].get('atr', 0),
        "trailing_stop_activated": False,
        "tp1_done": False,
        "quality_score": quality_score,
        "atr_percent": df.iloc[-1].get('atr_percent', 0),
        "rsi_at_signal": df.iloc[-1].get('rsi', 0),
        "confirmations": confirmations,
        "mfi_at_signal": df.iloc[-1].get('mfi', 0),
        "macd_hist_at_signal": df.iloc[-1].get('macd_hist', 0)
    }
    
    trade_levels = {
        "entry_price": entry_price,
        "stop_loss": stop_loss_price,
        "target_price_1": target_price_1,
        "target_price_2": target_price_2
    }

    current_real_balance = 0
    with balance_lock:
        current_real_balance = usdt_balance

    quantity_dec = calculate_position_size(symbol, entry_price, current_real_balance, is_real)

    if quantity_dec is None or quantity_dec <= 0:
        logger.error(f"❌ [{symbol}] Position size calculation failed")
        return
    
    notional_value = float(quantity_dec) * entry_price

    # حفظ الإشارة في قاعدة البيانات
    save_signal_to_db(
        symbol, entry_price, trade_levels, strategy_name,
        is_real, float(quantity_dec), signal_details
    )
    
    # إرسال إشعار
    send_trade_open_notification(
        symbol, strategy_name, entry_price, stop_loss_price,
        target_price_1, target_price_2, float(quantity_dec),
        is_real, quality_score, df.iloc[-1].get('atr_percent', 0),
        notional_value, confirmations
    )
    
    logger.info(f"✅ [Signal] {symbol} signal created and saved successfully")

def save_signal_to_db(symbol: str, entry_price: float, trade_levels: Dict, strategy_name: str,
                     is_real: bool, quantity: float, signal_details: Dict, order_id: Optional[str] = None):
    """[مُحسّن] حفظ الإشارة في قاعدة البيانات والكاش"""
    if not check_db_connection() or not conn:
        logger.error(f"❌ [DB] Cannot save signal for {symbol}: DB connection failed")
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (
                    symbol, entry_price, target_price_1, target_price_2, stop_loss, status,
                    strategy_name, is_real_trade, quantity, initial_quantity, signal_details, order_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                symbol, float(entry_price), float(trade_levels['target_price_1']),
                float(trade_levels['target_price_2']), float(trade_levels['stop_loss']),
                'open', strategy_name, is_real, float(quantity), float(quantity),
                json.dumps(signal_details, cls=NpEncoder), order_id
            ))
            
            new_id = cur.fetchone()['id']
        
        conn.commit()
        logger.info(f"✅ [DB] Signal saved for {symbol} with ID: {new_id}")
        
        # تحضير البيانات للكاش
        signal_data = {
            'id': new_id,
            'symbol': symbol,
            'entry_price': float(entry_price),
            'target_price_1': float(trade_levels['target_price_1']),
            'target_price_2': float(trade_levels['target_price_2']),
            'stop_loss': float(trade_levels['stop_loss']),
            'status': 'open',
            'strategy_name': strategy_name,
            'is_real_trade': is_real,
            'quantity': float(quantity),
            'initial_quantity': float(quantity),
            'signal_details': signal_details,
            'order_id': order_id,
            'current_price': entry_price  # إضافة سعر حالي
        }
        
        # تحديث الكاش
        with signal_cache_lock:
            open_signals_cache[symbol] = signal_data
        
        # بث التحديث
        broadcast({"type": "new_signal", "payload": signal_data})
        logger.info(f"✅ [Cache & WS] Signal {new_id} for {symbol} added to cache and broadcasted")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ [DB] Failed to save signal for {symbol}: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

# --- التقارير الدورية ---
def send_daily_performance_report():
    """إرسال تقرير الأداء اليومي (مُحسّن)"""
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
                logger.info("[Daily Report] No trades today")
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
                    f"العملة: `{best_trade['symbol']}` | الربح: `{best_trade['profit_percentage']:.2f}%`\n"
                    f"الاستراتيجية: `{STRATEGY_NAMES.get(best_trade['strategy_name'], best_trade['strategy_name'])}`\n\n"
                )
            
            if worst_trade:
                message += (
                    f"📉 *أسوأ صفقة:*\n"
                    f"العملة: `{worst_trade['symbol']}` | الخسارة: `{worst_trade['profit_percentage']:.2f}%`\n"
                    f"الاستراتيجية: `{STRATEGY_NAMES.get(worst_trade['strategy_name'], worst_trade['strategy_name'])}`\n\n"
                )
            
            send_enhanced_telegram_message(message, force=True)
            
    except Exception as e:
        logger.error(f"❌ [Daily Report] Error: {e}", exc_info=True)

def schedule_periodic_reports():
    """جدولة التقارير الدورية"""
    logger.info("Starting periodic reports scheduler...")
    while True:
        try:
            now = datetime.now(timezone.utc)
            if now.hour == 23 and now.minute == 55:  # تغيير من 23:59 لتجنب التداخل
                send_daily_performance_report()
                time.sleep(61)
            if now.hour % 6 == 0 and now.minute == 0:
                send_market_state_notification()
                time.sleep(61)
            time.sleep(30)
        except Exception as e:
            logger.error(f"❌ [Periodic Reports] Scheduler error: {e}", exc_info=True)
            time.sleep(60)

def start_periodic_reports():
    """بدء خيط التقارير الدورية"""
    reports_thread = Thread(target=schedule_periodic_reports, daemon=True)
    reports_thread.start()
    logger.info("✅ [Periodic Reports] Started scheduler thread")

def handle_socket_message(msg):
    """معالجة رسائل WebSocket مع تحسين لتقليل البث غير الضروري"""
    global live_prices
    try:
        if msg and 'e' in msg and msg['e'] == 'error':
            logger.error(f"❌ [WebSocket] Error: {msg['m']}")
            return
        
        if isinstance(msg, list):
            price_updates = {}
            with live_prices_lock:
                # معالجة تحديثات السعر فقط للرموز الموجودة في الكاش
                with signal_cache_lock:
                    monitored_symbols = set(open_signals_cache.keys())
                
                for ticker in msg:
                    if 's' in ticker and 'c' in ticker:
                        symbol = ticker['s']
                        
                        # تخطي الرموز غير المرغوبة لتقليل المعالجة
                        if symbol not in monitored_symbols and symbol not in validated_symbols_to_scan[:20]:
                            continue
                            
                        try:
                            price = float(ticker['c'])
                            live_prices[symbol] = price
                            price_updates[symbol] = price
                        except (ValueError, TypeError):
                            logger.warning(f"[WebSocket] Invalid price for {symbol}: {ticker.get('c')}")
            
            if price_updates:
                broadcast({"type": "price_update", "payload": price_updates})
                logger.debug(f"[WebSocket] Broadcasted price updates for {len(price_updates)} symbols")
    except Exception as e:
        logger.error(f"❌ [WebSocket] Error processing message: {e}", exc_info=True)

def start_websocket():
    """بدء اتصال WebSocket مع Binance"""
    global ws_manager
    ws_manager = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    ws_manager.start()
    ws_manager.start_ticker_socket(callback=handle_socket_message)
    logger.info("✅ [WebSocket] Subscribed to ticker stream")

def get_exchange_info_map() -> None:
    """جلب معلومات الصرف من Binance"""
    global exchange_info_map
    try:
        logger.info("[API] Fetching exchange info...")
        exchange_info_map = {s['symbol']: s for s in client.get_exchange_info()['symbols']}
        logger.info(f"✅ [API] Exchange info loaded: {len(exchange_info_map)} symbols")
    except Exception as e:
        logger.error(f"❌ [API] Error fetching exchange info: {e}")

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """التحقق والتحقق من رموز التداول"""
    try:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        if not os.path.exists(file_path):
            logger.critical(f"❌ Symbol list file '{filename}' not found!")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        
        if not exchange_info_map:
            get_exchange_info_map()
        
        active = {s for s, info in exchange_info_map.items() 
                 if info.get('quoteAsset') == 'USDT' and info.get('status') == 'TRADING'}
        
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ Found {len(validated)} valid symbols for trading")
        return validated
    except Exception as e:
        logger.error(f"❌ [Symbols] Error: {e}")
        return []

# --- مسارات Flask ---
@app.route('/')
def dashboard():
    """صفحة لوحة التحكم الرئيسية"""
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/backtest')
def backtest_page():
    """صفحة الاختبار الخلفي"""
    return render_template_string(BACKTEST_TEMPLATE, STRATEGY_NAMES=STRATEGY_NAMES)

@app.route('/settings')
def settings_page():
    """صفحة الإعدادات"""
    with trade_amount_lock:
        trade_amount_min = FIXED_TRADE_AMOUNT_MIN_USDT
        trade_amount_max = FIXED_TRADE_AMOUNT_MAX_USDT
    with trading_mode_lock: is_paper_mode = paper_trading_mode
    with min_quality_lock: min_quality = MIN_SIGNAL_QUALITY
    with strategy_config_lock:
        strategy_config = {
            "enable_ema_filter": ENABLE_EMA_FILTER,
            "enable_macd_confirmation": ENABLE_MACD_CONFIRMATION,
            "enable_mfi_filter": ENABLE_MFI_FILTER,
            "enable_candlestick_patterns": ENABLE_CANDLESTICK_PATTERNS,
            "required_confirmations": REQUIRED_CONFIRMATIONS
        }
    
    return render_template_string(SETTINGS_TEMPLATE,
                                  trade_amount_min=trade_amount_min,
                                  trade_amount_max=trade_amount_max,
                                  MAX_OPEN_TRADES=MAX_OPEN_TRADES,
                                  min_quality=min_quality,
                                  is_paper_mode=is_paper_mode,
                                  strategy_config=strategy_config)

@app.route('/api/dashboard_data')
def dashboard_data():
    """endpoint جلب بيانات لوحة التحكم"""
    try:
        payload = get_dashboard_payload()
        logger.info(f"[API] Dashboard data sent: {len(open_signals_cache)} open signals")
        return jsonify(payload)
    except Exception as e:
        logger.error(f"❌ [API Error] Dashboard data: {e}", exc_info=True)
        return jsonify({"error": "Failed to load dashboard data"}), 500

@app.route('/toggle_trading', methods=['POST'])
def toggle_trading():
    """تبديل حالة التداول"""
    global is_trading_enabled
    with trading_status_lock:
        is_trading_enabled = not is_trading_enabled
    
    status_msg = "enabled" if is_trading_enabled else "disabled"
    log_and_notify("info", f"Trading has been {status_msg}.", "TRADING_STATUS")
    logger.info(f"[Trading] Status changed to: {is_trading_enabled}")
    
    return jsonify({"status": "success", "trading_enabled": is_trading_enabled})

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """تحديث الإعدادات العامة"""
    try:
        data = request.json
        logger.info(f"[API] Updating settings: {list(data.keys())}")
        
        if 'FIXED_TRADE_AMOUNT_MIN_USDT' in data and 'FIXED_TRADE_AMOUNT_MAX_USDT' in data:
            with trade_amount_lock:
                global FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT
                FIXED_TRADE_AMOUNT_MIN_USDT = float(data['FIXED_TRADE_AMOUNT_MIN_USDT'])
                FIXED_TRADE_AMOUNT_MAX_USDT = float(data['FIXED_TRADE_AMOUNT_MAX_USDT'])
                broadcast({
                    "type": "trade_amount_update",
                    "payload": {
                        "min": FIXED_TRADE_AMOUNT_MIN_USDT,
                        "max": FIXED_TRADE_AMOUNT_MAX_USDT
                    }
                })
        
        if 'MAX_OPEN_TRADES' in data:
            global MAX_OPEN_TRADES
            MAX_OPEN_TRADES = int(data['MAX_OPEN_TRADES'])
        
        if 'paper_trading_mode' in data:
            with trading_mode_lock:
                global paper_trading_mode
                paper_trading_mode = bool(data['paper_trading_mode'])
        
        save_settings_to_redis()
        logger.info("✅ [Settings] Updated successfully")
        return jsonify({"success": True, "message": "Settings updated"})
    
    except Exception as e:
        logger.error(f"❌ [Settings] Update failed: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/strategy_config', methods=['POST'])
def update_strategy_config():
    """تحديث إعدادات الاستراتيجية"""
    try:
        data = request.json
        logger.info(f"[API] Updating strategy config: {list(data.keys())}")
        
        with strategy_config_lock:
            global ENABLE_EMA_FILTER, ENABLE_MACD_CONFIRMATION, ENABLE_MFI_FILTER, ENABLE_CANDLESTICK_PATTERNS, REQUIRED_CONFIRMATIONS
            ENABLE_EMA_FILTER = bool(data.get('enable_ema_filter', True))
            ENABLE_MACD_CONFIRMATION = bool(data.get('enable_macd_confirmation', True))
            ENABLE_MFI_FILTER = bool(data.get('enable_mfi_filter', True))
            ENABLE_CANDLESTICK_PATTERNS = bool(data.get('enable_candlestick_patterns', True))
            REQUIRED_CONFIRMATIONS = int(data.get('required_confirmations', 3))
        
        save_settings_to_redis()
        logger.info("✅ [Strategy Config] Updated successfully")
        return jsonify({"success": True, "message": "Strategy config updated"})
    
    except Exception as e:
        logger.error(f"❌ [Strategy Config] Update failed: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/signal_quality', methods=['POST'])
def update_signal_quality():
    """تحديث إعدادات جودة الإشارة"""
    try:
        data = request.json
        global MIN_SIGNAL_QUALITY
        with min_quality_lock:
            MIN_SIGNAL_QUALITY = int(data.get('min_quality', 65))
        
        save_settings_to_redis()
        broadcast({"type": "quality_filter", "payload": {"min_quality": MIN_SIGNAL_QUALITY}})
        logger.info(f"✅ [Quality Filter] Updated to {MIN_SIGNAL_QUALITY}")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"❌ [Quality Filter] Update failed: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/health')
def api_health():
    """endpoint فحص صحة النظام"""
    try:
        with trading_status_lock: trading_enabled = is_trading_enabled
        with trading_mode_lock: is_paper = paper_trading_mode
        
        health_data = {
            "status": "ok",
            "trading_enabled": trading_enabled,
            "mode": "PAPER" if is_paper else "REAL",
            "open_signals": len(open_signals_cache),
            "open_signals_list": list(open_signals_cache.keys()),
            "ws": {"connected": len(ws_clients) > 0},
            "db": {"connected": check_db_connection()},
            "redis": {"connected": redis_client is not None},
            "active_strategies": len(STRATEGY_NAMES),
            "symbols_to_scan": len(validated_symbols_to_scan)
        }
        
        logger.info(f"[Health] System health check: {health_data}")
        return jsonify(health_data), 200
    
    except Exception as e:
        logger.error(f"❌ [Health] Check failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/open_signals')
def get_open_signals():
    """[مُحسّن] endpoint جلب الإشارات المفتوحة"""
    if not check_db_connection():
        logger.error("[API] Cannot fetch signals: DB connection failed")
        return jsonify({"error": "Database connection failed"}), 500
    
    sort_by = request.args.get('sort', 'id')
    allowed_sort_fields = ['id', 'symbol', 'entry_price', 'strategy_name', 'quality_score']
    if sort_by not in allowed_sort_fields:
        sort_by = 'id'
    
    try:
        logger.info(f"[API] Fetching open signals, sort by: {sort_by}")
        
        with conn.cursor() as cur:
            # ✅ استعلام صحيح ومكتمل
            cur.execute("""
                SELECT id, symbol, entry_price, target_price_1, target_price_2, stop_loss, 
                       strategy_name, is_real_trade, quantity, initial_quantity, 
                       signal_details, status
                FROM signals 
                WHERE status IN ('open', 'updated')
                ORDER BY %s DESC
            """, (sort_by,))
            
            signals = cur.fetchall()
        
        # تحويل النتائج إلى قائمة من القواميس
        signals_list = []
        for s in signals:
            signal_dict = dict(s)
            # التأكد من أن signal_details هو dict
            if isinstance(signal_dict.get('signal_details'), str):
                try:
                    signal_dict['signal_details'] = json.loads(signal_dict['signal_details'])
                except:
                    signal_dict['signal_details'] = {}
            signals_list.append(signal_dict)
        
        logger.info(f"✅ [API] Returned {len(signals_list)} open signals")
        
        # تحديث الكاش في الخلفية
        def update_cache():
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in signals_list:
                    open_signals_cache[signal['symbol']] = signal
        
        Thread(target=update_cache, daemon=True).start()
        
        return jsonify({"signals": signals_list})
    
    except Exception as e:
        logger.error(f"❌ [API] Error fetching signals: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500