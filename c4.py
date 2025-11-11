# ملف crypto_bot_enhanced.py - إصدار Render (V35.1.0)
# --- التحسينات الرئيسية:
# 1. تهيئة محسّنة للمتغيرات البيئية
# 2. اتصال قاعدة بيانات أكثر قوة مع SSL
# 3. جعل Redis اختياريًا تمامًا
# 4. إعداد WSGI لـ Gunicorn
# 5. تسجيل محسّن للبيئات المستضافة
# 6. فحوصات صحة أفضل
# 7. معالجة أخطاء محسّنة

import os
import sys
import time
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import warnings
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
import random

# --- إعدادات التجاهل واللوجر ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ضبط دقة النوع Decimal
getcontext().prec = 18

# --- تكوين التسجيل لـ Render ---
LOG_LEVEL = config('LOG_LEVEL', default='INFO')
LOG_FORMAT = config('LOG_FORMAT', default='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# على Render، لا تكتب إلى ملف، استخدم stdout/stderr فقط
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('CryptoBotRSI_5min')

# --- دالة تهيئة المتغيرات البيئية بأمان ---
def get_env_or_default(key: str, default: Any, cast_type: type = str) -> Any:
    """جلب متغير بيئي مع قيمة افتراضية ونوع محدد"""
    try:
        value = config(key, default=default)
        if cast_type == bool:
            return str(value).lower() in ('true', '1', 'yes', 'y', 'on')
        return cast_type(value)
    except Exception as e:
        logger.warning(f"فشل تحميل متغير {key}: {e}, استخدام القيمة الافتراضية: {default}")
        return default

# --- المتغيرات البيئية مع قيم افتراضية آمنة ---
API_KEY: str = get_env_or_default('BINANCE_API_KEY', '', str)
API_SECRET: str = get_env_or_default('BINANCE_API_SECRET', '', str)
DB_URL: str = get_env_or_default('DATABASE_URL', '', str)
REDIS_URL: str = get_env_or_default('REDIS_URL', '', str)
TELEGRAM_BOT_TOKEN: str = get_env_or_default('TELEGRAM_BOT_TOKEN', '', str)
TELEGRAM_CHAT_ID: str = get_env_or_default('TELEGRAM_CHAT_ID', '', str)

# --- المتغيرات القابلة للتكوين ---
PAPER_TRADE_FIXED_AMOUNT_USDT: float = get_env_or_default('PAPER_TRADE_FIXED_AMOUNT_USDT', 10.0, float)
FIXED_TRADE_AMOUNT_MIN_USDT: float = get_env_or_default('FIXED_TRADE_AMOUNT_MIN_USDT', 4.5, float)
FIXED_TRADE_AMOUNT_MAX_USDT: float = get_env_or_default('FIXED_TRADE_AMOUNT_MAX_USDT', 6.5, float)
MAX_OPEN_TRADES: int = get_env_or_default('MAX_OPEN_TRADES', 3, int)
MIN_SIGNAL_QUALITY: int = get_env_or_default('MIN_SIGNAL_QUALITY', 65, int)
AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE: bool = get_env_or_default('AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE', True, bool)
COOLDOWN_MINUTES_AFTER_SL: int = get_env_or_default('COOLDOWN_MINUTES_AFTER_SL', 30, int)
TRAILING_STOP_ACTIVATION_PROFIT_PERCENT: float = get_env_or_default('TRAILING_STOP_ACTIVATION_PROFIT_PERCENT', 1.0, float)

# --- إعدادات الاستراتيجية ---
ENABLE_EMA_FILTER: bool = get_env_or_default('ENABLE_EMA_FILTER', True, bool)
ENABLE_MACD_CONFIRMATION: bool = get_env_or_default('ENABLE_MACD_CONFIRMATION', True, bool)
ENABLE_MFI_FILTER: bool = get_env_or_default('ENABLE_MFI_FILTER', True, bool)
ENABLE_CANDLESTICK_PATTERNS: bool = get_env_or_default('ENABLE_CANDLESTICK_PATTERNS', True, bool)
REQUIRED_CONFIRMATIONS: int = get_env_or_default('REQUIRED_CONFIRMATIONS', 3, int)

# --- إعدادات عامة ---
SIGNAL_GENERATION_TIMEFRAME: str = get_env_or_default('SIGNAL_GENERATION_TIMEFRAME', '5m', str)
HIGHER_TIMEFRAME: str = get_env_or_default('HIGHER_TIMEFRAME', '15m', str)
SIGNAL_GENERATION_LOOKBACK_DAYS: int = get_env_or_default('SIGNAL_GENERATION_LOOKBACK_DAYS', 7, int)
BTC_SYMBOL: str = get_env_or_default('BTC_SYMBOL', 'BTCUSDT', str)
API_REQUEST_DELAY: float = get_env_or_default('API_REQUEST_DELAY', 0.5, float)

# --- المتغيرات العامة والحالة ---
is_trading_enabled: bool = False
trading_status_lock = Lock()
paper_trading_mode: bool = True
trading_mode_lock = Lock()
usdt_balance: float = 1000.0
balance_lock = Lock()
cooldowns_by_symbol: Dict[str, datetime] = {}
cooldowns_lock = Lock()
consecutive_losses_by_symbol: Dict[str, int] = {}
consecutive_losses_lock = Lock()

# --- كيانات الاتصال ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[Any] = None
ws_manager: Optional[ThreadedWebsocketManager] = None
live_prices: Dict[str, float] = {}
live_prices_lock = Lock()
exchange_info_map: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
notifications_cache = deque(maxlen=50)
rejection_logs_cache = deque(maxlen=50)
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

# --- أسماء الاستراتيجيات ---
STRATEGY_NAMES = {
    "RSI_Enhanced_Strategy": "RSI المتقدم (ذروة البيع)",
    "EMA_Crossover_Strategy": "تقاطع EMA21/50",
    "MACD_Momentum_Strategy": "زخم MACD الصاعد",
    "Stochastic_Strategy": "ستوكاستيك ذروة البيع",
    "Bollinger_Bands_Strategy": "اختراق بولينجر لأسفل",
    "Multi_Indicator_Fusion": "مزيج المؤشرات الذكي"
}

# --- مساعدات Lock ---
notifications_lock = Lock()
rejection_logs_lock = Lock()
min_quality_lock = Lock()
trade_amount_lock = Lock()
strategy_config_lock = Lock()
strategy_filters_lock = Lock()

# --- إعداد تطبيق Flask و WebSocket ---
app = Flask(__name__)
app.config['SECRET_KEY'] = get_env_or_default('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
CORS(app)
sock = Sock(app)
ws_clients: List[Any] = []
ws_clients_lock = Lock()

# --- دالة مصنف NpEncoder ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# --- دوال WebSocket ---
def broadcast(data: Dict):
    """بث البيانات لجميع عملاء WebSocket"""
    with ws_clients_lock:
        if not ws_clients:
            return
        
        clients_to_remove = []
        for client in ws_clients:
            try:
                client.send(json.dumps(data, cls=NpEncoder))
            except Exception as e:
                logger.warning(f"[WebSocket] Failed to send to client: {e}")
                clients_to_remove.append(client)
        
        for client in clients_to_remove:
            try:
                ws_clients.remove(client)
            except ValueError:
                pass

def get_dashboard_payload() -> Dict:
    """جلب جميع بيانات لوحة التحكم"""
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
        with signal_cache_lock: signals_count = len(open_signals_cache)
        
        with strategy_config_lock:
            strategy_config = {
                "enable_ema_filter": ENABLE_EMA_FILTER,
                "enable_macd_confirmation": ENABLE_MACD_CONFIRMATION,
                "enable_mfi_filter": ENABLE_MFI_FILTER,
                "enable_candlestick_patterns": ENABLE_CANDLESTICK_PATTERNS,
                "required_confirmations": REQUIRED_CONFIRMATIONS
            }
        
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
            "strategy_config": strategy_config,
            "server_time": datetime.now(timezone.utc).isoformat(),
            "open_signals_cache_count": signals_count,
            "active_strategies": list(STRATEGY_NAMES.keys())
        }
    except Exception as e:
        logger.error(f"❌ [Dashboard] Error: {e}", exc_info=True)
        return {"error": str(e)}

# --- دوال المساعدة ---
def check_db_connection() -> bool:
    """التحقق من صحة اتصال قاعدة البيانات مع إعادة اتصال تلقائية"""
    global conn
    if conn is None or conn.closed != 0:
        try:
            init_db(retries=2, base_delay=2)
        except:
            return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
        return True
    except (OperationalError, InterfaceError):
        try:
            init_db(retries=2, base_delay=2)
        except:
            pass
        return conn is not None and conn.closed == 0

def init_db(retries: int = 3, base_delay: int = 3) -> None:
    """تهيئة اتصال قاعدة البيانات"""
    global conn
    logger.info("[DB] Initializing connection...")
    
    # إضافة معلمات SSL لـ Render
    db_url_to_use = DB_URL
    if DB_URL and 'postgres' in DB_URL:
        if 'sslmode' not in DB_URL:
            separator = '?' if '?' not in DB_URL else '&'
            db_url_to_use = f"{DB_URL}{separator}sslmode=require"
    
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(db_url_to_use, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            
            with conn.cursor() as cur:
                # إنشاء الجداول
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION NOT NULL,
                        stop_loss DOUBLE PRECISION NOT NULL, status TEXT DEFAULT 'open',
                        closing_price DOUBLE PRECISION, closed_at TIMESTAMP, profit_percentage DOUBLE PRECISION,
                        strategy_name TEXT, signal_details JSONB, is_real_trade BOOLEAN DEFAULT FALSE,
                        quantity DOUBLE PRECISION, closing_reason TEXT, order_id TEXT,
                        created_at TIMESTAMP DEFAULT NOW(),
                        target_price_1 DOUBLE PRECISION, target_price_2 DOUBLE PRECISION,
                        initial_quantity DOUBLE PRECISION
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
            
            conn.commit()
            
            # إضافة الفهارس
            optimize_database()
            logger.info("✅ [DB] Connection successful")
            return
            
        except Exception as e:
            logger.error(f"❌ [DB] Attempt {attempt + 1}/{retries}: {e}")
            if conn:
                conn.rollback()
            if attempt < retries - 1:
                time.sleep(base_delay * (attempt + 1))
    
    logger.critical("❌ [DB] All connection attempts failed")
    # لا ننهي البرنامج، نسمح له بالاستمرار بدون DB

def optimize_database():
    """تحسين قاعدة البيانات بإضافة الفهارس"""
    if not check_db_connection():
        return
    try:
        with conn.cursor() as cur:
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)",
                "CREATE INDEX IF NOT EXISTS idx_signals_symbol_status ON signals(symbol, status)",
                "CREATE INDEX IF NOT EXISTS idx_notifications_timestamp ON notifications(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_signals_status_closed_at ON signals(status, closed_at)",
                "CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy_name)"
            ]
            for index in indexes:
                cur.execute(index)
            conn.commit()
            logger.info("✅ [DB] Indexes optimized")
    except Exception as e:
        logger.error(f"❌ [DB] Optimization failed: {e}")
        if conn: conn.rollback()

def init_redis():
    """تهيئة Redis اختياريًا"""
    global redis_client
    if not REDIS_URL:
        logger.warning("[Redis] No REDIS_URL configured, using memory cache only")
        redis_client = None
        return
    
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=5)
        redis_client.ping()
        logger.info("✅ [Redis] Connected")
    except Exception as e:
        logger.warning(f"⚠️ [Redis] Connection failed: {e}. Continuing without Redis")
        redis_client = None

# --- التسجيل والإشعارات ---
def log_and_notify(level: str, message: str, notification_type: str):
    """تسجيل وإخطار الأحداث"""
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error}
    log_methods.get(level.lower(), logger.info)(message)
    
    if not check_db_connection():
        return
    
    try:
        new_notification = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": notification_type,
            "message": message
        }
        with notifications_lock:
            notifications_cache.appendleft(new_notification)
        
        with conn.cursor() as cur:
            cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", 
                       (notification_type, message))
        conn.commit()
        
        broadcast({"type": "new_notification", "payload": new_notification})
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
        logger.error(f"❌ [Log Rejection] Error: {e}", exc_info=True)

def send_enhanced_telegram_message(message: str, force: bool = False):
    """إرسال رسائل Telegram مع معالجة الأخطاء"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("[Telegram] Tokens not configured")
        return
    
    try:
        settings = get_notification_settings()
        if not settings.get('telegram_enabled') and not force:
            return
    except:
        pass  # تابع بغض النظر عن إعدادات Redis
    
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
                    logger.info(f"[Telegram] Message {i+1}/{len(messages)} sent")
                    break
            except Exception as e:
                if attempt == 2:
                    logger.error(f"❌ [Telegram] Failed: {e}")
                time.sleep(1.5)

def get_notification_settings() -> Dict:
    """جلب إعدادات الإشعارات من Redis أو الذاكرة"""
    defaults = {
        'telegram_enabled': True,
        'email_enabled': False,
        'min_profit_notification': 0.5,
        'max_loss_notification': -0.5
    }
    
    if not redis_client:
        return defaults
    
    try:
        settings_data = redis_client.get('notification_settings')
        if settings_data:
            settings = json.loads(settings_data)
            return {**defaults, **settings}
        return defaults
    except:
        return defaults

# --- الاستراتيجيات والمؤشرات ---
def check_market_volatility_filter_enhanced(df: pd.DataFrame, symbol: str = "Unknown") -> bool:
    """فلتر تقلب السوق"""
    if len(df) < 2 or 'atr_percent' not in df.columns or df['atr_percent'].isnull().all():
        log_rejection(symbol, "Market Volatility Filter Failed", {"reason": "No ATR data"})
        return False
    
    last_atr_percent = float(df.iloc[-1].get('atr_percent', 0))
    ATR_PERCENT_MIN = 0.25
    ATR_PERCENT_MAX = 4.0
    
    if not (ATR_PERCENT_MIN <= last_atr_percent <= ATR_PERCENT_MAX):
        log_rejection(symbol, "Market Volatility Filter Failed", {
            "atr": f"{last_atr_percent:.2f}%",
            "range": f"({ATR_PERCENT_MIN:.2f}-{ATR_PERCENT_MAX:.2f})%"
        })
        return False
    
    return True

def detect_bullish_candlestick_pattern(df: pd.DataFrame) -> bool:
    """كشف أنماط الشموع الصعودية"""
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
    prev_is_red = prev['close'] < prev['open']
    current_is_green = last['close'] > last['open']
    is_engulfing = prev_is_red and current_is_green and last['close'] > prev['open'] and last['open'] < prev['close']
    
    return is_hammer or is_engulfing

# --- الاستراتيجيات الست ---
def check_rsi_enhanced_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """استراتيجية RSI المُحسّنة"""
    if len(df) < 50:
        return False

    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    confirmations = {
        'rsi_oversold_recovery': prev['rsi'] < 40 and last['rsi'] > prev['rsi'],
        'ema_trend': last['close'] > last['ema21'] if ENABLE_EMA_FILTER else True,
        'macd_momentum': last['macd_hist'] > 0 if ENABLE_MACD_CONFIRMATION else True,
        'mfi_pressure': last['mfi'] > 25 if ENABLE_MFI_FILTER else True,
        'candlestick_bullish': detect_bullish_candlestick_pattern(df) if ENABLE_CANDLESTICK_PATTERNS else True
    }
    
    active_confirmations = sum(confirmations.values())
    
    if not confirmations['rsi_oversold_recovery']:
        log_rejection(df.name if hasattr(df, 'name') else 'Unknown', "RSI: Not oversold or no recovery")
        return False
    
    if active_confirmations < REQUIRED_CONFIRMATIONS:
        failed = [k for k, v in confirmations.items() if not v]
        log_rejection(df.name if hasattr(df, 'name') else 'Unknown', 
                     f"Insufficient confirmations ({active_confirmations}/{REQUIRED_CONFIRMATIONS})", 
                     {"failed": ', '.join(failed)})
        return False
    
    df.confirmations = confirmations
    return True

def check_ema_crossover_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """تقاطع EMA21/50"""
    if len(df) < 60:
        return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    confirmations = {
        'ema_crossover': prev['ema21'] <= prev['ema50'] and last['ema21'] > last['ema50'],
        'price_above_ema21': last['close'] > last['ema21'],
        'macd_positive': last['macd_hist'] > 0,
        'volume_increase': last['volume'] > df['volume'].tail(20).mean(),
        'rsi_support': last['rsi'] < 65
    }
    
    if not confirmations['ema_crossover']:
        log_rejection(df.name if hasattr(df, 'name') else 'Unknown', "EMA: No crossover")
        return False
    
    if sum(confirmations.values()) < 3:
        return False
    
    df.confirmations = confirmations
    return True

def check_macd_momentum_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """زخم MACD"""
    if len(df) < 50:
        return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    confirmations = {
        'macd_histogram_positive': last['macd_hist'] > 0,
        'macd_histogram_increasing': last['macd_hist'] > prev['macd_hist'],
        'macd_signal_crossover': last['macd'] > last['macd_signal'],
        'price_trend': last['close'] > last['ema21'],
        'rsi_support': last['rsi'] > 45
    }
    
    if not confirmations['macd_histogram_positive'] or not confirmations['macd_histogram_increasing']:
        log_rejection(df.name if hasattr(df, 'name') else 'Unknown', "MACD: No bullish momentum")
        return False
    
    if sum(confirmations.values()) < 3:
        return False
    
    df.confirmations = confirmations
    return True

def check_stochastic_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """ستوكاستيك"""
    if len(df) < 20:
        return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    confirmations = {
        'stochastic_crossover': prev['stoch_k'] <= prev['stoch_d'] and last['stoch_k'] > last['stoch_d'],
        'stochastic_oversold': prev['stoch_k'] < 30,
        'rsi_support': last['rsi'] > 40,
        'price_above_ema': last['close'] > last['ema21'],
        'volume_check': last['volume'] > df['volume'].tail(20).mean()
    }
    
    if not confirmations['stochastic_crossover'] or not confirmations['stochastic_oversold']:
        log_rejection(df.name if hasattr(df, 'name') else 'Unknown', "Stochastic: No crossover or not oversold")
        return False
    
    if sum(confirmations.values()) < 3:
        return False
    
    df.confirmations = confirmations
    return True

def check_bollinger_bands_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """بولينجر باندز"""
    if len(df) < 30:
        return False
    
    last = df.iloc[-1]
    prev1 = df.iloc[-2]
    prev2 = df.iloc[-3]
    
    confirmations = {
        'price_at_lower_band': prev1['close'] <= prev1['bb_lower'] or prev2['close'] <= prev2['bb_lower'],
        'price_bounce': last['close'] > prev1['close'],
        'rsi_recovery': last['rsi'] > prev1['rsi'],
        'volume_increase': last['volume'] > prev1['volume'],
        'bb_width_normal': prev1['bb_width'] > 0.02
    }
    
    if not confirmations['price_at_lower_band'] or not confirmations['price_bounce']:
        log_rejection(df.name if hasattr(df, 'name') else 'Unknown', "BB: No bounce")
        return False
    
    if sum(confirmations.values()) < 3:
        return False
    
    df.confirmations = confirmations
    return True

def check_multi_indicator_fusion_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """مزيج المؤشرات"""
    if len(df) < 50:
        return False
    
    last = df.iloc[-1]
    
    # نقاط القوة
    rsi_score = max(0, 40 - last['rsi']) * 2.5 if last['rsi'] < 50 else 0
    macd_score = min(100, max(0, last['macd_hist'] / abs(last['macd_signal']) * 50)) if last['macd_signal'] != 0 else 0
    mfi_score = (last['mfi'] - 20) * 1.25 if last['mfi'] > 20 else 0
    ema_score = 100 if last['close'] > last['ema21'] else 0
    stoch_score = max(0, 30 - last['stoch_k']) * 3.33 if last['stoch_k'] < 50 else 0
    
    total_score = rsi_score + macd_score + mfi_score + ema_score + stoch_score
    
    confirmations = {
        'rsi_favorable': last['rsi'] < 50,
        'macd_positive': last['macd_hist'] > 0,
        'mfi_support': last['mfi'] > 25,
        'price_above_ema': last['close'] > last['ema21'],
        'stochastic_favorable': last['stoch_k'] < 60
    }
    
    active_confirmations = sum(confirmations.values())
    
    if total_score < 150 or active_confirmations < 3:
        log_rejection(df.name if hasattr(df, 'name') else 'Unknown', 
                     f"Fusion: Score {total_score}/500, Confirmations {active_confirmations}/3")
        return False
    
    df.confirmations = confirmations
    df.quality_score = total_score
    return True

# --- الحسابات الفنية ---
def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """حساب جميع المؤشرات"""
    df_calc = df.copy()
    
    # SMA
    df_calc['sma7'] = df_calc['close'].rolling(window=7).mean()
    df_calc['sma200'] = df_calc['close'].rolling(window=200).mean()

    # EMA
    for span in [9, 13, 21, 34, 50, 100, 200]:
        df_calc[f'ema{span}'] = df_calc['close'].ewm(span=span, adjust=False).mean()
    
    # ATR
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    df_calc['atr'] = tr.ewm(span=14, adjust=False).mean()
    df_calc['atr_percent'] = (df_calc['atr'] / df_calc['close'].replace(0, 1e-9)) * 100
    
    # ADX
    up_move = df_calc['high'].diff()
    down_move = -df_calc['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=14, adjust=False).mean()
    
    # RSI
    delta = df_calc['close'].diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain_14 = gain.ewm(com=13, adjust=False).mean()
    avg_loss_14 = loss.ewm(com=13, adjust=False).mean()
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
    exp1 = df_calc['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_calc['close'].ewm(span=26, adjust=False).mean()
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
    df_calc['vwap'] = (df_calc['close'] * df_calc['volume']).cumsum() / df_calc['volume'].cumsum().replace(0, 1e-9)
    
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
    
    return df_calc.dropna()

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """جلب البيانات التاريخية"""
    try:
        time.sleep(API_REQUEST_DELAY)
        klines = client.get_historical_klines(symbol, interval, f"{days} day ago UTC")
        if not klines:
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
        df.name = symbol  # تعيين اسم الرمز للتصحيح
        
        return df.dropna().astype(float)
    except Exception as e:
        logger.error(f"❌ [Data] {symbol}: {e}")
        return None

# --- حساب حجم الصفقة ---
def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> Optional[Decimal]:
    """تعديل الكمية لتتوافق مع LOT_SIZE"""
    try:
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info:
            return None
        
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if not lot_size_filter:
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
        
        adjusted = quantity_dec - (quantity_dec % step_size)
        if adjusted < min_qty:
            return None
        
        return adjusted
    except Exception as e:
        logger.error(f"❌ [{symbol}] Quantity adjustment: {e}")
        return None

def calculate_position_size(symbol: str, entry_price: float, available_balance: float, is_real: bool) -> Optional[Decimal]:
    """حساب حجم الصفقة"""
    desired = PAPER_TRADE_FIXED_AMOUNT_USDT if not is_real else random.uniform(
        FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT
    )
    
    try:
        dec_entry = Decimal(str(entry_price))
        if dec_entry <= 0:
            return None
        
        dec_desired = Decimal(str(desired))
        dec_balance = Decimal(str(available_balance))
        
        if is_real and dec_desired > dec_balance:
            log_rejection(symbol, "Insufficient Balance", {
                "required": f"${dec_desired:.2f}",
                "available": f"${dec_balance:.2f}"
            })
            return None
        
        initial_quantity = dec_desired / dec_entry
        adjusted = adjust_quantity_to_lot_size(symbol, float(initial_quantity))
        
        if adjusted is None or adjusted <= 0:
            return None
        
        # فحص Min Notional
        notional = adjusted * dec_entry
        symbol_info = exchange_info_map.get(symbol)
        
        if symbol_info:
            min_notional_filter = next((f for f in symbol_info['filters'] 
                                      if f['filterType'] in ('MIN_NOTIONAL', 'NOTIONAL')), None)
            
            if min_notional_filter:
                min_notional = Decimal(str(min_notional_filter.get('minNotional', '5.0')))
                
                if notional < min_notional:
                    required_qty = (min_notional * Decimal('1.01')) / dec_entry
                    adjusted = adjust_quantity_to_lot_size(symbol, float(required_qty))
                    
                    if adjusted is None:
                        return None
                    
                    notional = adjusted * dec_entry
        
        if is_real and notional > dec_balance:
            log_rejection(symbol, "Insufficient Balance", {
                "required": f"${notional:.2f}",
                "available": f"${dec_balance:.2f}"
            })
            return None
        
        if notional <= 0:
            return None
        
        return adjusted
        
    except Exception as e:
        logger.error(f"❌ [{symbol}] Position size error: {e}")
        return None

# --- إدارة الإشارات ---
def load_open_signals_to_cache():
    """تحميل الإشارات المفتوحة من قاعدة البيانات"""
    logger.info("[Cache] Loading open signals...")
    
    if not check_db_connection():
        logger.warning("[Cache] DB not connected, skipping")
        return
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, symbol, entry_price, target_price_1, target_price_2, stop_loss, 
                       strategy_name, is_real_trade, quantity, initial_quantity, 
                       signal_details, status, order_id
                FROM signals 
                WHERE status IN ('open', 'updated')
                ORDER BY id DESC
            """)
            
            signals = cur.fetchall()
            logger.info(f"[Cache] Found {len(signals)} open signals")
            
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in signals:
                    if isinstance(signal['signal_details'], str):
                        try:
                            signal['signal_details'] = json.loads(signal['signal_details'])
                        except:
                            signal['signal_details'] = {}
                    open_signals_cache[signal['symbol']] = dict(signal)
            
            broadcast({"type": "signals_loaded", "count": len(open_signals_cache)})
            
    except Exception as e:
        logger.error(f"❌ [Cache] Failed to load signals: {e}", exc_info=True)
        if conn: conn.rollback()

def save_signal_to_db(symbol: str, entry_price: float, trade_levels: Dict, strategy_name: str,
                     is_real: bool, quantity: float, signal_details: Dict, order_id: Optional[str] = None) -> bool:
    """حفظ الإشارة في قاعدة البيانات"""
    if not check_db_connection():
        logger.error(f"❌ [DB] Cannot save {symbol}: DB not connected")
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
        logger.info(f"✅ [DB] Signal {new_id} saved for {symbol}")
        
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
            'order_id': order_id
        }
        
        with signal_cache_lock:
            open_signals_cache[symbol] = signal_data
        
        broadcast({"type": "new_signal", "payload": signal_data})
        return True
        
    except Exception as e:
        logger.error(f"❌ [DB] Failed to save signal: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

# --- إنشاء الإشارات ---
def calculate_dynamic_stop_loss(df: pd.DataFrame, entry_price: float, strategy_name: str) -> float:
    """حساب وقف الخسارة الديناميكي"""
    last = df.iloc[-1]
    atr_value = last.get('atr', 0)
    
    stop_loss = entry_price - (atr_value * 1.8)
    
    if strategy_name == "RSI_Enhanced_Strategy":
        recent_low = df['low'].tail(7).min()
        stop_loss = min(recent_low * 0.996, entry_price - (atr_value * 1.6))
    
    max_stop_distance = entry_price * 0.06
    if entry_price - stop_loss > max_stop_distance:
        stop_loss = entry_price - max_stop_distance
    
    return round(stop_loss, 6)

def calculate_dynamic_take_profit(df: pd.DataFrame, entry_price: float, stop_loss: float, strategy_name: str) -> tuple:
    """حساب أهداف الربح"""
    risk_amount = entry_price - stop_loss
    if risk_amount <= 0:
        return (entry_price * 1.012, entry_price * 1.025)
    
    target1 = entry_price + (risk_amount * 1.5)
    target2 = entry_price + (risk_amount * 2.5)
    
    if 'r1' in df.columns:
        r1 = df.iloc[-1].get('r1', target1)
        r2 = df.iloc[-1].get('r2', target2)
        target1 = max(target1, r1 * 0.985)
        target2 = max(target2, r2 * 0.985)
    
    return round(target1, 6), round(target2, 6)

def calculate_dynamic_quality_score(df: pd.DataFrame, symbol: str) -> int:
    """حساب درجة جودة الإشارة"""
    if len(df) < 2:
        return 0
    
    last = df.iloc[-1]
    
    score = 0
    
    # RSI
    if last['rsi'] < 40:
        score += (40 - last['rsi']) * 2.5
    elif last['rsi'] > 50:
        score += (last['rsi'] - 50) * 1.5
    
    # EMA Distance
    ema_distance = ((last['close'] - last['ema21']) / last['ema21']) * 100
    if ema_distance > 0:
        score += min(ema_distance * 3, 15)
    
    # MACD
    macd_strength = last['macd_hist']
    if macd_strength > 0:
        score += min(macd_strength / abs(last['macd_signal']) * 15, 20)
    
    # MFI
    if last['mfi'] > 20:
        score += min((last['mfi'] - 20) / 80 * 15, 15)
    
    # ATR
    atr_percent = last['atr_percent']
    if 0.4 <= atr_percent <= 1.8:
        score += 25
    elif atr_percent > 0.25 and atr_percent < 4.0:
        score += 15
    
    # Volume
    avg_volume = df['volume'].tail(20).mean()
    volume_ratio = last['volume'] / avg_volume if avg_volume > 0 else 1
    if volume_ratio > 1.2:
        score += 10
    
    return int(min(score, 100))

def create_trade_signal(symbol: str, df: pd.DataFrame, strategy_name: str):
    """إنشاء إشارة تداول جديدة"""
    logger.info(f"🔍 [Signal] Processing {symbol} with {strategy_name}")
    
    df.strategy = strategy_name
    
    # الفلاتر
    if not check_market_volatility_filter_enhanced(df, symbol):
        return
    
    # جودة الإشارة
    quality_score = calculate_dynamic_quality_score(df, symbol)
    min_score = MIN_SIGNAL_QUALITY
    
    if quality_score < min_score:
        log_rejection(symbol, "Low Quality Signal", {
            "score": quality_score,
            "min_required": min_score
        })
        return
    
    logger.info(f"⭐ [Signal] {symbol} quality: {quality_score}/100")
    
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
    
    with balance_lock:
        current_balance = usdt_balance
    
    quantity = calculate_position_size(symbol, entry_price, current_balance, is_real)
    
    if quantity is None or quantity <= 0:
        logger.error(f"❌ [{symbol}] Invalid quantity")
        return
    
    notional_value = float(quantity) * entry_price
    
    save_signal_to_db(
        symbol, entry_price, trade_levels, strategy_name,
        is_real, float(quantity), signal_details
    )
    
    send_trade_open_notification(
        symbol, strategy_name, entry_price, stop_loss_price,
        target_price_1, target_price_2, float(quantity),
        is_real, quality_score, df.iloc[-1].get('atr_percent', 0),
        notional_value, confirmations
    )

# --- الفلاتر الديناميكية ---
def add_news_filter() -> bool:
    """فلتر الأخبار (مهلة 20 دقيقة)"""
    try:
        news_hours = [(12, 30), (14, 0), (18, 30), (22, 0)]
        now = datetime.now(timezone.utc)
        for hour, minute in news_hours:
            if now.hour == hour and abs(now.minute - minute) <= 20:
                return False
        return True
    except:
        return True  # تابع في حال حدوث خطأ

def add_liquidity_filter() -> bool:
    """فلتر السيولة"""
    try:
        now = datetime.now(timezone.utc)
        if now.weekday() >= 5:  # عطلة نهاية الأسبوع
            return False
        if now.hour >= 23 or now.hour <= 3:  # ساعات السيولة الضعيفة
            return False
        return True
    except:
        return True

def add_correlation_filter(new_symbol: str) -> bool:
    """فلتر الارتباط"""
    try:
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
    except Exception as e:
        logger.error(f"❌ [Correlation Filter] Error: {e}")
        return True

# --- التقارير والإشعارات ---
def send_trade_open_notification(symbol: str, strategy_name: str, entry_price: float, stop_loss: float,
                                target1: float, target2: float, quantity: float, is_real: bool,
                                quality_score: int, atr_percent: float, notional_value: float, confirmations: Dict):
    """إشعار بفتح صفقة"""
    trade_type = "حقيقية" if is_real else "ورقية"
    emoji = "🔥" if is_real else "📊"
    
    confirms_list = "\n".join([f"✅ {k.replace('_', ' ').title()}: {'نعم' if v else 'لا'}" for k, v in confirmations.items()])
    
    message = (
        f"{emoji} *صفقة {trade_type} جديدة ({SIGNAL_GENERATION_TIMEFRAME})*\n\n"
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
        f"*نسبة المخاطرة:* `{((entry_price - stop_loss) / entry_price * 100):.2f}%`"
    )
    
    send_enhanced_telegram_message(message, force=True)

def send_daily_performance_report():
    """إرسال تقرير الأداء اليومي"""
    if not check_db_connection():
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
                return
            
            message = (
                f"📈 *تقرير الأداء اليومي*\n\n"
                f"*التاريخ:* `{today.strftime('%Y-%m-%d')}`\n\n"
                f"*إجمالي الصفقات:* `{stats['total_trades']}`\n"
                f"*الصفقات الرابحة:* `{stats.get('winning_trades', 0)}`\n"
                f"*نسبة الربح:* `{(stats.get('winning_trades', 0) / stats['total_trades'] * 100):.1f}%`\n"
                f"*متوسط الربح:* `{stats.get('avg_profit', 0):.2f}%`\n"
                f"*إجمالي الربح:* `{stats.get('total_profit', 0):.2f}%`"
            )
            
            send_enhanced_telegram_message(message, force=True)
            
    except Exception as e:
        logger.error(f"❌ [Daily Report] Error: {e}", exc_info=True)

def schedule_periodic_reports():
    """جدولة التقارير الدورية"""
    logger.info("Starting periodic reports...")
    while True:
        try:
            now = datetime.now(timezone.utc)
            if now.hour == 23 and now.minute == 55:
                send_daily_performance_report()
                time.sleep(61)
            time.sleep(30)
        except Exception as e:
            logger.error(f"❌ [Scheduler] Error: {e}")
            time.sleep(60)

def start_periodic_reports():
    """بدء خيط التقارير"""
    reports_thread = Thread(target=schedule_periodic_reports, daemon=True)
    reports_thread.start()
    logger.info("✅ [Scheduler] Started")

# --- WebSocket وBinance ---
def handle_socket_message(msg):
    """معالجة رسائل WebSocket"""
    try:
        if isinstance(msg, list):
            price_updates = {}
            with live_prices_lock:
                for ticker in msg:
                    if 's' in ticker and 'c' in ticker:
                        symbol = ticker['s']
                        with signal_cache_lock:
                            monitored_symbols = set(open_signals_cache.keys())
                        
                        if symbol not in monitored_symbols and symbol not in validated_symbols_to_scan[:20]:
                            continue
                        
                        try:
                            price_updates[symbol] = float(ticker['c'])
                        except:
                            pass
                
                if price_updates:
                    broadcast({"type": "price_update", "payload": price_updates})
    except Exception as e:
        logger.error(f"❌ [WebSocket] Error: {e}")

def start_websocket():
    """بدء WebSocket"""
    global ws_manager
    try:
        ws_manager = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
        ws_manager.start()
        ws_manager.start_ticker_socket(callback=handle_socket_message)
        logger.info("✅ [WebSocket] Started")
    except Exception as e:
        logger.error(f"❌ [WebSocket] Failed to start: {e}")

def get_exchange_info_map():
    """جلب معلومات الصرف"""
    global exchange_info_map
    try:
        logger.info("[API] Fetching exchange info...")
        exchange_info_map = {s['symbol']: s for s in client.get_exchange_info()['symbols']}
        logger.info(f"✅ [API] Loaded {len(exchange_info_map)} symbols")
    except Exception as e:
        logger.error(f"❌ [API] Exchange info error: {e}")

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """التحقق من رموز التداول"""
    try:
        # على Render، الملف قد لا يكون متاحًا
        # لذا نستخدم قائمة افتراضية إذا لم يُعثر على الملف
        if not os.path.exists(filename):
            logger.warning(f"[Symbols] {filename} not found, using default list")
            # قائمة افتراضية من العملات الرئيسية
            default_symbols = {
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT',
                'LINKUSDT', 'MATICUSDT', 'SOLUSDT', 'AVAXUSDT', 'ATOMUSDT'
            }
            
            if not exchange_info_map:
                get_exchange_info_map()
            
            active = {s for s, info in exchange_info_map.items() 
                     if info.get('quoteAsset') == 'USDT' and info.get('status') == 'TRADING'}
            
            validated = sorted(list(default_symbols.intersection(active)))
            return validated
        
        with open(filename, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        
        if not exchange_info_map:
            get_exchange_info_map()
        
        active = {s for s, info in exchange_info_map.items() 
                 if info.get('quoteAsset') == 'USDT' and info.get('status') == 'TRADING'}
        
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ [Symbols] Found {len(validated)} valid symbols")
        return validated
    except Exception as e:
        logger.error(f"❌ [Symbols] Error: {e}")
        return []

# --- مسارات Flask ---
@app.route('/')
def dashboard():
    """لوحة التحكم"""
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/api/dashboard_data')
def dashboard_data():
    """بيانات لوحة التحكم"""
    try:
        payload = get_dashboard_payload()
        return jsonify(payload)
    except Exception as e:
        logger.error(f"❌ [API] Dashboard error: {e}", exc_info=True)
        return jsonify({"error": "Failed"}), 500

@app.route('/toggle_trading', methods=['POST'])
def toggle_trading():
    """تبديل التداول"""
    global is_trading_enabled
    with trading_status_lock:
        is_trading_enabled = not is_trading_enabled
    
    status = "enabled" if is_trading_enabled else "disabled"
    log_and_notify("info", f"Trading {status}", "TRADING_STATUS")
    
    return jsonify({"status": "success", "trading_enabled": is_trading_enabled})

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """تحديث الإعدادات"""
    global FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT, MAX_OPEN_TRADES, paper_trading_mode
    global MIN_SIGNAL_QUALITY
    
    try:
        data = request.json
        logger.info(f"[API] Updating settings: {list(data.keys())}")
        
        if 'FIXED_TRADE_AMOUNT_MIN_USDT' in data and 'FIXED_TRADE_AMOUNT_MAX_USDT' in data:
            FIXED_TRADE_AMOUNT_MIN_USDT = float(data['FIXED_TRADE_AMOUNT_MIN_USDT'])
            FIXED_TRADE_AMOUNT_MAX_USDT = float(data['FIXED_TRADE_AMOUNT_MAX_USDT'])
        
        if 'MAX_OPEN_TRADES' in data:
            MAX_OPEN_TRADES = int(data['MAX_OPEN_TRADES'])
        
        if 'paper_trading_mode' in data:
            paper_trading_mode = bool(data['paper_trading_mode'])
        
        if 'min_quality' in data:
            MIN_SIGNAL_QUALITY = int(data['min_quality'])
        
        return jsonify({"success": True, "message": "Settings updated"})
    
    except Exception as e:
        logger.error(f"❌ [Settings] Update failed: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/health')
def api_health():
    """فحص صحة النظام"""
    try:
        return jsonify({
            "status": "ok",
            "trading_enabled": is_trading_enabled,
            "mode": "PAPER" if paper_trading_mode else "REAL",
            "open_signals": len(open_signals_cache),
            "ws_clients": len(ws_clients),
            "db_connected": check_db_connection(),
            "redis_connected": redis_client is not None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/open_signals')
def get_open_signals():
    """جلب الإشارات المفتوحة"""
    if not check_db_connection():
        return jsonify({"error": "DB not connected"}), 500
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, symbol, entry_price, target_price_1, target_price_2, stop_loss, 
                       strategy_name, is_real_trade, quantity, signal_details, status
                FROM signals 
                WHERE status IN ('open', 'updated')
                ORDER BY id DESC
            """)
            
            signals = cur.fetchall()
            signals_list = [dict(s) for s in signals]
        
        return jsonify({"signals": signals_list})
    
    except Exception as e:
        logger.error(f"❌ [API] Signals error: {e}")
        return jsonify({"error": str(e)}), 500

# --- قوالب HTML (مبسطة) ---
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Crypto Bot Dashboard</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        .header { display: flex; justify-content: space-between; align-items: center; }
        .status { padding: 10px; border-radius: 5px; }
        .status.enabled { background: #4CAF50; color: white; }
        .status.disabled { background: #f44336; color: white; }
        .signals-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; margin-top: 20px; }
        .signal-card { border: 1px solid #ddd; padding: 10px; border-radius: 5px; }
        .metric { display: inline-block; margin: 5px 10px 5px 0; }
        button { padding: 10px 15px; background: #2196F3; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0b7dda; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Crypto Bot Dashboard</h1>
            <div>
                <span id="trading-status" class="status disabled">توقف</span>
                <button onclick="toggleTrading()">تبديل التداول</button>
                <button onclick="refreshData()">تحديث</button>
            </div>
        </div>
        
        <div style="margin: 20px 0;">
            <h3>المعلومات العامة</h3>
            <p class="metric"><strong>الوضع:</strong> <span id="mode">-</span></p>
            <p class="metric"><strong>الرصيد:</strong> <span id="balance">-</span> USDT</p>
            <p class="metric"><strong>الصفقات المفتوحة:</strong> <span id="open-count">0</span></p>
            <p class="metric"><strong>جودة الإشارة الأدنى:</strong> <span id="min-quality">-</span></p>
        </div>
        
        <div>
            <h3>الصفقات المفتوحة</h3>
            <div id="signals-grid" class="signals-grid">جاري التحميل...</div>
        </div>
        
        <div style="margin-top: 30px;">
            <h3>الإشعارات الأخيرة</h3>
            <div id="notifications" style="max-height: 200px; overflow-y: auto;"></div>
        </div>
    </div>
    
    <script>
        let ws = null;
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'price_update') {
                    // معالجة تحديثات السعر
                } else if (data.type === 'new_signal') {
                    refreshData();
                }
            };
            
            ws.onclose = () => {
                setTimeout(connectWebSocket, 5000);
            };
        }
        
        async function toggleTrading() {
            const response = await fetch('/toggle_trading', { method: 'POST' });
            const result = await response.json();
            logAndNotify('info', result.trading_enabled ? 'تم تفعيل التداول' : 'تم إيقاف التداول');
            refreshData();
        }
        
        async function refreshData() {
            try {
                const [dashboard, signals] = await Promise.all([
                    fetch('/api/dashboard_data').then(r => r.json()),
                    fetch('/api/open_signals').then(r => r.json())
                ]);
                
                updateDashboard(dashboard);
                updateSignals(signals.signals);
            } catch (e) {
                console.error('Refresh error:', e);
            }
        }
        
        function updateDashboard(data) {
            document.getElementById('trading-status').textContent = data.trading_enabled ? 'مفعّل' : 'متوقف';
            document.getElementById('trading-status').className = data.trading_enabled ? 'status enabled' : 'status disabled';
            document.getElementById('mode').textContent = data.paper_trading_mode ? '📊 ورقي' : '🔥 حقيقي';
            document.getElementById('balance').textContent = data.usdt_balance.toFixed(2);
            document.getElementById('open-count').textContent = data.open_signals_cache_count;
            document.getElementById('min-quality').textContent = data.min_signal_quality;
        }
        
        function updateSignals(signals) {
            const grid = document.getElementById('signals-grid');
            if (signals.length === 0) {
                grid.innerHTML = '<p>لا توجد صفقات مفتوحة</p>';
                return;
            }
            
            grid.innerHTML = signals.map(s => `
                <div class="signal-card">
                    <h4>${s.symbol}</h4>
                    <p><strong>الاستراتيجية:</strong> ${s.strategy_name}</p>
                    <p><strong>الدخول:</strong> ${s.entry_price}</p>
                    <p><strong>وقف الخسارة:</strong> ${s.stop_loss}</p>
                    <p><strong>الكمية:</strong> ${s.quantity}</p>
                </div>
            `).join('');
        }
        
        function logAndNotify(level, message) {
            const div = document.getElementById('notifications');
            const entry = document.createElement('div');
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            div.insertBefore(entry, div.firstChild);
        }
        
        // بدء الاتصال
        connectWebSocket();
        refreshData();
        setInterval(refreshData, 30000); // تحديث كل 30 ثانية
    </script>
</body>
</html>
"""

# --- نقطة الدخول الرئيسية ---
def main():
    """الدالة الرئيسية لبدء التطبيق"""
    logger.info("=" * 60)
    logger.info("🚀 Starting Crypto Bot Service")
    logger.info("=" * 60)
    
    # فحص المتغيرات البيئية المطلوبة
    required_vars = ['BINANCE_API_KEY', 'BINANCE_API_SECRET', 'DATABASE_URL']
    missing = [var for var in required_vars if not config(var, default='')]
    
    if missing:
        logger.critical(f"❌ Missing required environment variables: {missing}")
        logger.critical("Please set them in Render dashboard: Settings > Environment Variables")
        # لا ننهي، نسمح للتطبيق بالبدء للتحقق من الصحة
    
    # تهيئة الاتصالات
    init_db()
    init_redis()
    
    global client
    if API_KEY and API_SECRET:
        try:
            client = Client(API_KEY, API_SECRET)
            logger.info("✅ Binance client initialized")
        except Exception as e:
            logger.error(f"❌ Binance client failed: {e}")
    else:
        logger.warning("⚠️ Binance API keys not configured")
    
    # تحميل البيانات
    load_open_signals_to_cache()
    
    # بدء الخدمات
    start_periodic_reports()
    
    if client:
        get_exchange_info_map()
        global validated_symbols_to_scan
        validated_symbols_to_scan = get_validated_symbols()
        start_websocket()
    
    logger.info("✅ Service initialization complete")
    logger.info(f"📊 Trading mode: {'PAPER' if paper_trading_mode else 'REAL'}")
    logger.info(f"🔔 Telegram: {'ENABLED' if TELEGRAM_BOT_TOKEN else 'DISABLED'}")
    logger.info(f"💾 Database: {'CONNECTED' if check_db_connection() else 'DISCONNECTED'}")
    logger.info(f"🔄 Redis: {'CONNECTED' if redis_client else 'DISABLED'}")

# --- تشغيل التطبيق ---
if __name__ == '__main__':
    main()
    
    # في التطوير المحلي
    app.run(
        debug=get_env_or_default('FLASK_DEBUG', False, bool),
        host='0.0.0.0',
        port=get_env_or_default('PORT', 5000, int),
        threaded=True
    )
else:
    # للإنتاج على Render
    main()