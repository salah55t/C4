# ملف c4_enhanced_v11.3.py - نسخة V11.3 "Enhanced Trading System with Fibonacci-Based Dynamic Targets"
# --- نسخة معدلة مع استراتيجيات محسنة وإدارة مخاطر متقدمة وتدقيق شامل للأخطاء ---

import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import redis
import traceback
import gc
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException, BinanceRequestException
from flask import Flask, jsonify, render_template_string, request, abort
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timezone, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Tuple
from functools import wraps
import random
import warnings

# --- إعدادات التجاهل واللوجر ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# إعداد اللوجر بشكل مفصل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v11.3_enhanced.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV11.3-Enhanced')

# --- مشفر مخصص لأنواع بيانات NumPy والعشرية ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, Decimal): return float(obj)
        if isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat()
        return super(NpEncoder, self).default(obj)

# --- تحميل متغيرات البيئة ---
try:
    API_KEY: str = config('BINANCE_API_KEY', default='')
    API_SECRET: str = config('BINANCE_API_SECRET', default='')
    DB_URL: str = config('DATABASE_URL', default='')
    REDIS_URL: str = config('REDIS_URL', default='redis://localhost:6379/0')
    TELEGRAM_BOT_TOKEN: str = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID: str = config('TELEGRAM_CHAT_ID', default='')
    
    logger.info("✅ تم تحميل متغيرات البيئة بنجاح")
    logger.info(f"API_KEY: {'موجود' if API_KEY else 'غير موجود'}")
    logger.info(f"API_SECRET: {'موجود' if API_SECRET else 'غير موجود'}")
    logger.info(f"DB_URL: {'موجود' if DB_URL else 'غير موجود'}")
    logger.info(f"REDIS_URL: {REDIS_URL}")
    logger.info(f"TELEGRAM_BOT_TOKEN: {'موجود' if TELEGRAM_BOT_TOKEN else 'غير موجود'}")
    logger.info(f"TELEGRAM_CHAT_ID: {'موجود' if TELEGRAM_CHAT_ID else 'غير موجود'}")
    
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
    exit(1)

# --- متغيرات عامة وأقفال ---
is_trading_enabled: bool = False
trading_status_lock = Lock()

# --- المتغيرات القابلة للتعديل ---
RISK_PER_TRADE_PERCENT: float = 0.85
risk_per_trade_lock = Lock()

# --- مفاتيح تفعيل الاستراتيجيات ---
STRATEGY_CONFIG = {
    "BB_Reversal": {"enabled": True, "lock": Lock(), "display_name": "BB Reversal (Enhanced)"},
    "MACD_EMA": {"enabled": True, "lock": Lock(), "display_name": "MACD+EMA (Enhanced)"},
    "SR_Breakout": {"enabled": True, "lock": Lock(), "display_name": "S/R Breakout (Enhanced)"},
    "Triple_Confirmation": {"enabled": True, "lock": Lock(), "display_name": "Triple Confirmation (Enhanced)"},
    "VWAP_Reversal": {"enabled": True, "lock": Lock(), "display_name": "VWAP Reversal (Enhanced)"},
    "Price_Channel": {"enabled": True, "lock": Lock(), "display_name": "Price Channel (New)"},
}

# --- إعدادات الفلاتر القابلة للتعديل ---
FILTER_CONFIG = {
    "ADX_THRESHOLD": {"value": 25, "lock": Lock(), "display_name": "حد مؤشر ADX"},
    "BB_STOCH_VOLUME_MULT": {"value": 1.2, "lock": Lock(), "display_name": "مضاعف فوليوم (BB Reversal)"},
    "MACD_EMA_VOLUME_MULT": {"value": 1.2, "lock": Lock(), "display_name": "مضاعف فوليوم (MACD_EMA)"},
    "SR_BREAKOUT_VOLUME_MULT": {"value": 1.4, "lock": Lock(), "display_name": "مضاعف فوليوم (SR Breakout)"},
    "TRIPLE_CONF_VOLUME_MULT": {"value": 1.2, "lock": Lock(), "display_name": "مضاعف فوليوم (Triple Conf)"},
    "VWAP_VOLUME_MULT": {"value": 1.3, "lock": Lock(), "display_name": "مضاعف فوليوم (VWAP Reversal)"},
    "PRICE_CHANNEL_VOLUME_MULT": {"value": 1.3, "lock": Lock(), "display_name": "مضاعف فوليوم (Price Channel)"},
    "TRIPLE_CONF_MODE": {"value": "strict", "lock": Lock(), "display_name": "وضع (Triple Conf)"}, # 'strict' or 'relaxed'
    "VWAP_REVERSAL_MODE": {"value": "strict", "lock": Lock(), "display_name": "وضع (VWAP Reversal)"}, # 'strict' or 'relaxed'
    "TIME_BASED_EXIT_CANDLES": {"value": 30, "lock": Lock(), "display_name": "إغلاق الصفقة بعد (شمعة)"},
    "SIGNAL_STRENGTH_THRESHOLD": {"value": 0.7, "lock": Lock(), "display_name": "حد قوة الإشارة"},
    "MAX_CORRELATION_THRESHOLD": {"value": 0.7, "lock": Lock(), "display_name": "حد الارتباط بين الأصول"},
    "LIQUIDITY_FILTER_STRICTNESS": {"value": "medium", "lock": Lock(), "display_name": "صرامة فلتر السيولة"}, # 'low', 'medium', 'high'
    # إعدادات فيبوناتشي الجديدة
    "FIBONACCI_LOOKBACK_PERIOD": {"value": 50, "lock": Lock(), "display_name": "فترة حساب فيبوناتشي"},
    "FIBONACCI_PARTIAL_PROFIT_LEVEL": {"value": 0.618, "lock": Lock(), "display_name": "مستوى جني الأرباح الجزئي"},
    "FIBONACCI_PARTIAL_PROFIT_PERCENT": {"value": 0.5, "lock": Lock(), "display_name": "نسبة جني الأرباح الجزئي"},
    "FIBONACCI_BREAKEVEN_LEVEL": {"value": 0.382, "lock": Lock(), "display_name": "مستوى تحريك وقف الخسارة"},
    "TREND_SENSITIVITY": {"value": 0.6, "lock": Lock(), "display_name": "حساسية اكتشاف الاتجاه"},
}

# --- إعدادات المؤشرات الفنية والإطارات الزمنية ---
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 90
BTC_SYMBOL: str = 'BTCUSDT'
MAX_OPEN_TRADES: int = 5
ATR_PERIOD: int = 14
ADX_PERIOD: int = 14
CACHE_EXPIRATION_MINUTES: int = 15
BATCH_SIZE: int = 50

# --- إعدادات إدارة المخاطر والخروج ---
USE_SMART_EXIT_SYSTEM: bool = True
TAKE_PROFIT_LEVELS = {
    1: {"atr_multiplier": 1.5, "exit_percentage": 0.50},
    2: {"atr_multiplier": 3.0, "exit_percentage": 0.30},
    3: {"atr_multiplier": 5.0, "exit_percentage": 0.20}
}
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_STOP_ACTIVATION_ATR: float = 2.0

# --- متغيرات الحالة والكاش ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
exchange_info_map: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
rejection_logs_cache = []
rejection_logs_lock = Lock()
notifications_cache = []
notifications_lock = Lock()
current_market_state: Dict[str, Any] = {"status": "INITIALIZING"}
market_state_lock = Lock()

# --- قاموس أسباب الرفض باللغة العربية ---
REJECTION_REASONS_AR = {
    "Market Status Filter: BTC Downtrend (5m)": "فلتر السوق: اتجاه البيتكوين هابط (5 دقائق)",
    "Market Status Filter: BTC Downtrend (4h)": "فلتر السوق: اتجاه البيتكوين هابط (4 ساعات)",
    "Market Status Filter: Low Liquidity": "فلتر السوق: سيولة منخفضة",
    "Market Status Filter: High Volatility": "فلتر السوق: تقلبات عالية",
    "Market Status Filter: Weak Market Strength": "فلتر السوق: ضعف قوة السوق",
    "BB Reversal: No Reversal Candle": "انعكاس BB: لم تظهر شمعة انعكاسية",
    "BB Reversal: ADX Filter Failed": "انعكاس BB: فلتر قوة الاتجاه ADX",
    "BB Reversal: Volume Filter Failed": "انعكاس BB: فلتر تأكيد حجم التداول",
    "BB Reversal: RSI Not Oversold": "انعكاس BB: RSI ليس في منطقة تشبع بيعي",
    "BB Reversal: Price Too Far Below BB": "انعكاس BB: السعر بعيد جداً تحت BB",
    "MACD_EMA: RSI Filter Failed": "MACD_EMA: فلتر RSI",
    "MACD_EMA: Trend Filter Failed": "MACD_EMA: فلتر تأكيد الاتجاه",
    "MACD_EMA: MACD Filter Failed": "MACD_EMA: فلتر MACD",
    "MACD_EMA: Volume Filter Failed": "MACD_EMA: فلتر حجم التداول",
    "SR_Breakout: Volume Filter Failed": "SR_Breakout: فلتر حجم التداول",
    "SR_Breakout: RSI Filter Failed": "SR_Breakout: فلتر RSI",
    "SR_Breakout: ADX Filter Failed": "SR_Breakout: فلتر ADX",
    "Triple_Confirmation: Conditions Not Met": "Triple Confirmation: لم تتحقق الشروط",
    "Triple_Confirmation: Volatility Filter Failed": "Triple Confirmation: فلتر التقلبات",
    "VWAP_Reversal: Conditions Not Met": "VWAP Reversal: لم تتحقق الشروط",
    "VWAP_Reversal: RSI Not Oversold": "VWAP Reversal: RSI ليس في منطقة تشبع بيعي",
    "VWAP_Reversal: Trend Not Aligned": "VWAP Reversal: الاتجاه غير متوافق",
    "Price Channel: Volume Filter Failed": "القناة السعرية: فلتر حجم التداول",
    "Price Channel: RSI Filter Failed": "القناة السعرية: فلتر RSI",
    "Price Channel: ADX Filter Failed": "القناة السعرية: فلتر ADX",
    "Insufficient Balance": "الرصيد غير كافٍ",
    "Min Notional Filter": "قيمة الصفقة أقل من الحد الأدنى",
    "Invalid Position Size": "حجم الصفقة غير صالح",
    "Lot Size Adjustment Failed": "فشل تعديل حجم العقد",
    "Portfolio Risk: Max Open Trades": "مخاطر المحفظة: الحد الأقصى للصفقات المفتوحة",
    "Portfolio Risk: Sector Concentration": "مخاطر المحفظة: تركيز قطاعي مفرط",
    "Portfolio Risk: Max Portfolio Risk": "مخاطر المحفظة: تجاوز الحد الأقصى للمخاطرة",
    "Signal Strength Too Low": "قوة الإشارة منخفضة جداً"
}

# --- آلية تنظيم الطلبات المتقدمة (Token Bucket) ---
class RequestThrottler:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_rate = float(refill_rate)
        self.last_refill_time = time.time()
        self.lock = Lock()
        self.total_weight_used_minute = 0
        self.minute_start_time = time.time()

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill_time
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill_time = now
        if now - self.minute_start_time > 60:
            self.total_weight_used_minute = 0
            self.minute_start_time = now

    def consume(self, weight: int) -> None:
        with self.lock:
            self._refill()
            if weight > self.tokens:
                wait_time = (weight - self.tokens) / self.refill_rate
                logger.warning(f"🚦 [Throttler] الوزن المطلوب ({weight}) أعلى من المتاح ({self.tokens:.2f}). الانتظار {wait_time:.2f} ثانية.")
                time.sleep(wait_time)
            self._refill()
            self.tokens -= weight
            self.total_weight_used_minute += weight

throttler = RequestThrottler(capacity=5900, refill_rate=100) 

def rate_limiter(weight=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            throttler.consume(weight)
            try:
                response = func(*args, **kwargs)
                return response
            except (BinanceAPIException, BinanceRequestException) as e:
                if e.status_code in [429, 418]:
                    logger.critical("🚨 [API BAN] تم الوصول إلى حد الطلبات (HTTP 429/418). سيتم الانتظار لمدة 10 دقائق.")
                    send_telegram_message("🚨 *تحذير حظر API!* 🚨\\nتم الوصول إلى حد الطلبات. سيتوقف البوت مؤقتاً.")
                    time.sleep(600)
                raise
        return wrapper
    return decorator

# --- دوال مساعدة ---
def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("[تليجرام] Token أو Chat ID غير معين، تم تخطي الإرسال.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("[تليجرام] تم إرسال الرسالة بنجاح.")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [تليجرام] فشل إرسال الرسالة: {e}")

def init_db(retries: int = 5, delay: int = 5) -> bool:
    global conn
    logger.info("[قاعدة البيانات] تهيئة الاتصال...")
    
    if not DB_URL:
        logger.error("❌ [قاعدة البيانات] DB_URL غير معرف في متغيرات البيئة")
        return False
    
    db_url_to_use = DB_URL
    if 'postgres' in db_url_to_use and 'sslmode' not in db_url_to_use:
        db_url_to_use += f"{'?' if '?' not in db_url_to_use else '&'}sslmode=require"
    
    for attempt in range(retries):
        try:
            logger.info(f"[قاعدة البيانات] محاولة الاتصال {attempt + 1}/{retries}...")
            conn = psycopg2.connect(db_url_to_use, connect_timeout=15, cursor_factory=RealDictCursor)
            conn.autocommit = False
            logger.info("✅ [قاعدة البيانات] تم الاتصال بنجاح.")
            
            with conn.cursor() as cur:
                logger.info("[قاعدة البيانات] إنشاء الجداول إذا لم تكن موجودة...")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION NOT NULL,
                        target_price DOUBLE PRECISION, stop_loss DOUBLE PRECISION NOT NULL,
                        status TEXT DEFAULT 'open', closing_price DOUBLE PRECISION, 
                        opened_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        closed_at TIMESTAMP,
                        profit_percentage DOUBLE PRECISION, strategy_name TEXT, signal_details JSONB,
                        current_peak_price DOUBLE PRECISION, is_real_trade BOOLEAN DEFAULT FALSE,
                        quantity DOUBLE PRECISION, original_quantity DOUBLE PRECISION, order_id TEXT, closing_reason TEXT,
                        exit_levels JSONB, candles_since_entry INTEGER DEFAULT 0,
                        fibonacci_levels JSONB, partial_profit_taken BOOLEAN DEFAULT FALSE
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_performance (
                        strategy_name TEXT PRIMARY KEY, total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0, total_pnl_percent DOUBLE PRECISION DEFAULT 0.0
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL, message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS historical_data_cache (
                        symbol_timeframe TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        data JSONB NOT NULL,
                        last_updated TIMESTAMP WITH TIME ZONE NOT NULL
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timeframe ON historical_data_cache (symbol, timeframe);")
                
                conn.commit()
                logger.info("✅ [قاعدة البيانات] تم إنشاء/تحديث الجداول بنجاح.")
                return True
        except OperationalError as e:
            logger.error(f"❌ [قاعدة البيانات] خطأ في الاتصال (محاولة {attempt + 1}/{retries}): {e}")
            if conn: 
                try:
                    conn.rollback()
                except:
                    pass
            if attempt < retries - 1: 
                logger.info(f"[قاعدة البيانات] الانتظار {delay} ثانية قبل المحاولة التالية...")
                time.sleep(delay)
            else: 
                logger.critical("❌ [قاعدة البيانات] فشل الاتصال بعد عدة محاولات.")
                return False
        except Exception as e:
            logger.error(f"❌ [قاعدة البيانات] خطأ غير متوقع (محاولة {attempt + 1}/{retries}): {e}")
            if conn: 
                try:
                    conn.rollback()
                except:
                    pass
            if attempt < retries - 1: 
                logger.info(f"[قاعدة البيانات] الانتظار {delay} ثانية قبل المحاولة التالية...")
                time.sleep(delay)
            else: 
                logger.critical("❌ [قاعدة البيانات] فشل الاتصال بعد عدة محاولات.")
                return False
    
    return False

def check_db_connection() -> bool:
    global conn
    if conn is None or conn.closed != 0:
        logger.warning("[قاعدة البيانات] الاتصال مغلق، محاولة إعادة الاتصال...")
        if not init_db():
            return False
    try:
        if conn and conn.closed == 0:
            with conn.cursor() as cur: 
                cur.execute("SELECT 1;")
                conn.commit()
            return True
        return False
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [قاعدة البيانات] فقدان الاتصال: {e}. إعادة الاتصال...")
        try:
            if init_db():
                return conn is not None and conn.closed == 0
            return False
        except Exception as retry_e:
            logger.error(f"❌ [قاعدة البيانات] فشل إعادة الاتصال: {retry_e}")
            return False

def log_and_notify(level: str, message: str, notification_type: str):
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error, 'critical': logger.critical}
    log_methods.get(level.lower(), logger.info)(message)
    
    new_notification = {"timestamp": datetime.now(timezone.utc).isoformat(), "type": notification_type, "message": message}
    with notifications_lock:
        notifications_cache.insert(0, new_notification)
        if len(notifications_cache) > 100: notifications_cache.pop()

    if not check_db_connection() or not conn: 
        logger.warning("[قاعدة البيانات] لا يوجد اتصال بقاعدة البيانات، تخطي حفظ الإشعار.")
        return
        
    try:
        with conn.cursor() as cur: 
            cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
            conn.commit()
    except Exception as e:
        logger.error(f"❌ [قاعدة البيانات] فشل حفظ الإشعار: {e}")
        if conn: conn.rollback()

def log_rejection(symbol: str, reason_key: str, details: Optional[Dict] = None):
    reason_ar = REJECTION_REASONS_AR.get(reason_key, reason_key)
    log_message = f"🚫 [{symbol}] تم الرفض | السبب: {reason_ar} | تفاصيل: {details or {}}"
    logger.info(log_message)
    with rejection_logs_lock:
        rejection_logs_cache.insert(0, {
            "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
            "reason": reason_ar, "details": json.loads(json.dumps(details, cls=NpEncoder)) or {}
        })
        if len(rejection_logs_cache) > 200: rejection_logs_cache.pop()

def init_redis() -> bool:
    global redis_client
    logger.info("[Redis] تهيئة الاتصال...")
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] تم الاتصال بنجاح.")
        return True
    except redis.exceptions.ConnectionError as e:
        logger.error(f"❌ [Redis] فشل الاتصال: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ [Redis] خطأ غير متوقع: {e}")
        return False

@rate_limiter(weight=10)
def get_exchange_info_map() -> bool:
    global exchange_info_map
    if not client: 
        logger.error("[معلومات المنصة] عميل Binance غير مهيأ")
        return False
        
    logger.info("ℹ️ [معلومات المنصة] جاري جلب قواعد التداول...")
    try:
        info = client.get_exchange_info()
        exchange_info_map = {s['symbol']: s for s in info['symbols']}
        logger.info(f"✅ [معلومات المنصة] تم تحميل القواعد لـ {len(exchange_info_map)} عملة.")
        return True
    except Exception as e:
        logger.error(f"❌ [معلومات المنصة] فشل جلب المعلومات: {e}")
        return False

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    if not client: 
        logger.error("[التحقق من الرموز] عميل Binance غير مهيأ")
        return []
        
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        
        if not os.path.exists(file_path):
            logger.warning(f"⚠️ [التحقق من الرموز] ملف العملات '{filename}' غير موجود! سيتم إنشاء ملف افتراضي.")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# قائمة العملات المراد تداولها\n")
                f.write("BTC\n")
                f.write("ETH\n")
                f.write("BNB\n")
                f.write("ADA\n")
                f.write("SOL\n")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        
        if not raw_symbols:
            logger.warning(f"⚠️ [التحقق من الرموز] ملف العملات '{filename}' فارغ.")
            return []
            
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        
        if not exchange_info_map: 
            if not get_exchange_info_map():
                logger.error("[التحقق من الرموز] فشل جلب معلومات المنصة")
                return []
                
        active = {s for s, info in exchange_info_map.items() if info.get('quoteAsset') == 'USDT' and info.get('status') == 'TRADING'}
        validated = sorted(list(formatted.intersection(active)))
        
        logger.info(f"✅ [التحقق من الرموز] تم العثور على {len(validated)} عملة صالحة للتداول.")
        return validated
    except Exception as e:
        logger.error(f"❌ [التحقق من الرموز] خطأ: {e}", exc_info=True)
        return []

# --- دوال جلب البيانات وحساب المؤشرات ---
@rate_limiter(weight=1)
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: 
        logger.error(f"[جلب البيانات] عميل Binance غير مهيأ للرمز {symbol}")
        return None
        
    try:
        logger.debug(f"[جلب البيانات] جلب بيانات {symbol} ({interval}) لآخر {days} يوم")
        klines = client.get_historical_klines(symbol, interval, f"{days} day ago UTC")
        
        if not klines: 
            logger.warning(f"[جلب البيانات] لا توجد بيانات لـ {symbol} ({interval})")
            return None
            
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(klines, columns=cols)
        numeric_cols = {'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'}
        df = df.astype(numeric_cols)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logger.error(f"❌ [جلب البيانات] خطأ في جلب البيانات التاريخية لـ {symbol} ({interval}): {e}")
        raise

def _df_to_json(df: pd.DataFrame) -> str:
    return df.to_json(orient='split', date_format='iso')

def _json_to_df(json_data: Any) -> pd.DataFrame:
    if not isinstance(json_data, str):
        json_str = json.dumps(json_data)
    else:
        json_str = json_data
    df = pd.read_json(json_str, orient='split')
    df.index = pd.to_datetime(df.index, utc=True)
    return df

def get_data_for_symbol(symbol: str, timeframe: str, days: int) -> Optional[pd.DataFrame]:
    if timeframe == '5m':
        logger.debug(f"  -> [{symbol}-{timeframe}] ⚡ جلب بيانات حية (بدون كاش).")
        return fetch_historical_data(symbol, timeframe, days=2)

    if not check_db_connection() or not conn:
        logger.warning("[DB Cache] لا يوجد اتصال بقاعدة البيانات، سيتم الجلب مباشرة من API.")
        return fetch_historical_data(symbol, timeframe, days)

    try:
        with conn.cursor() as cur:
            pk = f"{symbol}_{timeframe}"
            cur.execute("SELECT data, last_updated FROM historical_data_cache WHERE symbol_timeframe = %s", (pk,))
            cache_result = cur.fetchone()
            
        if cache_result:
            last_updated_time = cache_result['last_updated']
            if (datetime.now(timezone.utc) - last_updated_time) < timedelta(minutes=CACHE_EXPIRATION_MINUTES):
                logger.debug(f"  -> [{symbol}-{timeframe}] 💾 استخدام البيانات من كاش قاعدة البيانات.")
                return _json_to_df(cache_result['data'])
            else:
                logger.debug(f"  -> [{symbol}-{timeframe}] ⏳ بيانات الكاش منتهية الصلاحية.")
    except Exception as e:
        logger.error(f"❌ [DB Cache] خطأ أثناء قراءة الكاش لـ {symbol}-{timeframe}: {e}")
        if conn: conn.rollback()

    logger.debug(f"  -> [{symbol}-{timeframe}] 🌐 جلب بيانات جديدة من المنصة.")
    try:
        df = fetch_historical_data(symbol, timeframe, days)
        if df is not None and not df.empty:
            json_data = _df_to_json(df)
            with conn.cursor() as cur:
                pk = f"{symbol}_{timeframe}"
                cur.execute("""
                    INSERT INTO historical_data_cache (symbol_timeframe, symbol, timeframe, data, last_updated)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (symbol_timeframe) DO UPDATE SET
                        data = EXCLUDED.data,
                        last_updated = EXCLUDED.last_updated;
                """, (pk, symbol, timeframe, json_data, datetime.now(timezone.utc)))
            conn.commit()
            logger.debug(f"  -> [{symbol}-{timeframe}] ✅ تم تحديث الكاش في قاعدة البيانات.")
            return df
        return None
    except Exception as e:
        logger.error(f"❌ [DB Cache] فشل جلب وتخزين البيانات لـ {symbol}-{timeframe}: {e}")
        if conn: conn.rollback()
        if isinstance(e, BinanceAPIException) and e.code == -1003:
            raise
        return None

def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: 
        logger.warning("[حساب المؤشرات] DataFrame فارغ")
        return pd.DataFrame()
        
    try:
        df_calc = df.copy()
        df_calc['ema_50'] = df_calc['close'].ewm(span=50, adjust=False).mean()
        df_calc['ema_200'] = df_calc['close'].ewm(span=200, adjust=False).mean()
        df_calc['volume_sma_20'] = df_calc['volume'].rolling(window=20).mean()
        
        high_low = df_calc['high'] - df_calc['low']
        high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
        low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
        df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
        
        up_move = df_calc['high'].diff()
        down_move = -df_calc['low'].diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
        plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
        minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
        df_calc['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
        
        delta = df_calc['close'].diff()
        gain = delta.clip(lower=0).ewm(com=14 - 1, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(com=14 - 1, adjust=False).mean()
        df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
        
        bb_period = 20
        df_calc['bb_middle'] = df_calc['close'].rolling(window=bb_period).mean()
        bb_std = df_calc['close'].rolling(window=bb_period).std()
        df_calc['bb_upper'] = df_calc['bb_middle'] + (bb_std * 2)
        df_calc['bb_lower'] = df_calc['bb_middle'] - (bb_std * 2)
        df_calc['bb_width'] = (df_calc['bb_upper'] - df_calc['bb_lower']) / df_calc['bb_middle'].replace(0, 1e-9)
        
        rsi = df_calc['rsi']
        stoch_rsi_val = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min()).replace(0, 1e-9)
        df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(3).mean() * 100
        df_calc['stoch_rsi_d'] = df_calc['stoch_rsi_k'].rolling(3).mean()
        
        exp1 = df_calc['close'].ewm(span=12, adjust=False).mean()
        exp2 = df_calc['close'].ewm(span=26, adjust=False).mean()
        df_calc['macd'] = exp1 - exp2
        df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()
        
        q = df_calc['volume']
        p = (df_calc['high'] + df_calc['low'] + df_calc['close']) / 3
        df_calc['vwap'] = (p * q).cumsum() / q.cumsum()
        
        return df_calc.dropna()
    except Exception as e:
        logger.error(f"❌ [حساب المؤشرات] خطأ في حساب المؤشرات: {e}")
        return pd.DataFrame()

# --- دوال جديدة لتحليل الاتجاه والفيبوناتشي ---
def calculate_market_trend_enhanced(symbol: str) -> Dict[str, Any]:
    """
    حساب اتجاه السوق المحسن باستخدام أطر زمنية متعددة ومؤشرات متنوعة
    """
    logger.debug(f"[اتجاه السوق] حساب اتجاه السوق لـ {symbol}")
    
    # إذا لم يكن هناك اتصال بـ Binance، نرجع حالة افتراضية
    if not client:
        logger.warning(f"[اتجاه السوق] لا يوجد اتصال بـ Binance، استخدام حالة افتراضية لـ {symbol}")
        return {
            "symbol": symbol,
            "overall_score": 0,
            "market_state": "RANGING",
            "timeframe_scores": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": "No Binance connection"
        }
    
    timeframes = ['5m', '15m', '1h', '4h']
    trend_scores = {}
    
    with FILTER_CONFIG["TREND_SENSITIVITY"]["lock"]: sensitivity = FILTER_CONFIG["TREND_SENSITIVITY"]["value"]
    
    for tf in timeframes:
        try:
            logger.debug(f"[اتجاه السوق] جلب بيانات {symbol} للإطار الزمني {tf}")
            df = get_data_for_symbol(symbol, tf, days=2 if tf == '5m' else 5)
            if df is None or df.empty:
                logger.warning(f"[اتجاه السوق] لا توجد بيانات لـ {symbol} ({tf})")
                continue
                
            df = calculate_all_features(df)
            if df.empty:
                logger.warning(f"[اتجاه السوق] فشل حساب المؤشرات لـ {symbol} ({tf})")
                continue
                
            last = df.iloc[-1]
            
            # حساب درجة الاتجاه لهذا الإطار الزمني
            ema_score = 0
            if last['ema_50'] > last['ema_200']:
                ema_score = 1
            elif last['ema_50'] < last['ema_200']:
                ema_score = -1
                
            # حساب درجة RSI
            rsi_score = 0
            if last['rsi'] > 60:
                rsi_score = 1
            elif last['rsi'] < 40:
                rsi_score = -1
                
            # حساب درجة ADX
            adx_score = 0
            if last['adx'] > 25:
                adx_score = 1
                
            # حساب درجة MACD
            macd_score = 0
            if last['macd'] > last['macd_signal']:
                macd_score = 1
            elif last['macd'] < last['macd_signal']:
                macd_score = -1
                
            # حساب درجة السعر بالنسبة لـ VWAP
            vwap_score = 0
            if last['close'] > last['vwap']:
                vwap_score = 1
            elif last['close'] < last['vwap']:
                vwap_score = -1
                
            # حساب الدرجة الإجمالية لهذا الإطار الزمني
            total_score = (ema_score * 0.3 + rsi_score * 0.2 + adx_score * 0.2 + 
                          macd_score * 0.15 + vwap_score * 0.15)
            
            # تطبيق عامل الحساسية
            if abs(total_score) < sensitivity:
                total_score = 0
                
            trend_scores[tf] = {
                "score": total_score,
                "ema_50": last['ema_50'],
                "ema_200": last['ema_200'],
                "rsi": last['rsi'],
                "adx": last['adx'],
                "macd": last['macd'],
                "macd_signal": last['macd_signal'],
                "vwap": last['vwap']
            }
            
            logger.debug(f"[اتجاه السوق] {symbol} ({tf}): score={total_score:.2f}, ema={ema_score}, rsi={rsi_score}, adx={adx_score}, macd={macd_score}, vwap={vwap_score}")
            
        except Exception as e:
            logger.error(f"❌ [اتجاه السوق] خطأ في حساب الاتجاه لـ {symbol} ({tf}): {e}")
            continue
    
    # حساب متوسط الاتجاه مع إعطاء وزن أكبر للأطر الزمنية الأعلى
    weights = {'5m': 0.1, '15m': 0.2, '1h': 0.3, '4h': 0.4}
    weighted_score = 0
    total_weight = 0
    
    for tf, data in trend_scores.items():
        weighted_score += data['score'] * weights.get(tf, 0.25)
        total_weight += weights.get(tf, 0.25)
    
    if total_weight > 0:
        overall_score = weighted_score / total_weight
    else:
        overall_score = 0
    
    # تحديد حالة السوق
    if overall_score > 0.6:
        market_state = "STRONG_UPTREND"
    elif overall_score > 0.2:
        market_state = "UPTREND"
    elif overall_score < -0.6:
        market_state = "STRONG_DOWNTREND"
    elif overall_score < -0.2:
        market_state = "DOWNTREND"
    else:
        market_state = "RANGING"
    
    logger.info(f"[اتجاه السوق] {symbol}: overall_score={overall_score:.2f}, market_state={market_state}")
    
    return {
        "symbol": symbol,
        "overall_score": overall_score,
        "market_state": market_state,
        "timeframe_scores": trend_scores,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

def calculate_fibonacci_levels(df: pd.DataFrame, period: int = None) -> Dict[str, float]:
    """
    حساب مستويات فيبوناتشي بناءً على أعلى وأدنى نقاط في الفترة المحددة
    """
    if period is None:
        with FILTER_CONFIG["FIBONACCI_LOOKBACK_PERIOD"]["lock"]: period = FILTER_CONFIG["FIBONACCI_LOOKBACK_PERIOD"]["value"]
    
    if len(df) < period:
        period = len(df)
    
    recent_data = df.iloc[-period:]
    high_point = recent_data['high'].max()
    low_point = recent_data['low'].min()
    
    diff = high_point - low_point
    
    levels = {
        "0.0": low_point,
        "0.236": low_point + diff * 0.236,
        "0.382": low_point + diff * 0.382,
        "0.5": low_point + diff * 0.5,
        "0.618": low_point + diff * 0.618,
        "0.786": low_point + diff * 0.786,
        "1.0": high_point,
        "1.272": high_point + diff * 0.272,
        "1.618": high_point + diff * 0.618
    }
    
    return levels

def adjust_dynamic_targets(symbol: str, signal_data: Dict, current_price: float, df: pd.DataFrame) -> Dict[str, Any]:
    """
    ضبط الأهداف ووقف الخسارة ديناميكياً بناءً على مستويات فيبوناتشي وحركة السعر
    """
    entry_price = float(signal_data['entry_price'])
    stop_loss = float(signal_data['stop_loss'])
    target_price = float(signal_data.get('target_price', entry_price * 1.05))
    
    # الحصول على مستويات فيبوناتشي الحالية
    fib_levels = calculate_fibonacci_levels(df)
    
    # الحصول على إعدادات فيبوناتشي
    with FILTER_CONFIG["FIBONACCI_PARTIAL_PROFIT_LEVEL"]["lock"]: 
        partial_profit_level = FILTER_CONFIG["FIBONACCI_PARTIAL_PROFIT_LEVEL"]["value"]
    with FILTER_CONFIG["FIBONACCI_PARTIAL_PROFIT_PERCENT"]["lock"]: 
        partial_profit_percent = FILTER_CONFIG["FIBONACCI_PARTIAL_PROFIT_PERCENT"]["value"]
    with FILTER_CONFIG["FIBONACCI_BREAKEVEN_LEVEL"]["lock"]: 
        breakeven_level = FILTER_CONFIG["FIBONACCI_BREAKEVEN_LEVEL"]["value"]
    
    # تحديد ما إذا كانت الصفقة طويلة (شراء) أم قصيرة (بيع)
    is_long = entry_price < target_price
    
    # تحديث مستويات فيبوناتشي في بيانات الإشارة
    signal_data['fibonacci_levels'] = fib_levels
    
    # تحديد الإجراءات المطلوبة
    actions = []
    updates = {}
    
    if is_long:
        # صفقة شراء (طويلة)
        if current_price >= fib_levels[str(partial_profit_level)] and not signal_data.get('partial_profit_taken', False):
            # الوصول إلى مستوى جني الأرباح الجزئي
            actions.append(f"جني أرباح جزئي عند مستوى فيبوناتشي {partial_profit_level}")
            updates['partial_profit_taken'] = True
            # يمكن هنا إضافة كود لبيع جزء من الصفقة
            
        if current_price >= fib_levels[str(breakeven_level)] and stop_loss < entry_price:
            # تحريك وقف الخسارة إلى نقطة التعادل
            actions.append("تحريك وقف الخسارة إلى نقطة التعادل")
            updates['stop_loss'] = entry_price
            
        if current_price >= fib_levels["1.0"]:
            # الوصول إلى مستوى 100% فيبوناتشي - يمكن رفع الهدف إلى المستوى التالي
            if target_price < fib_levels["1.272"]:
                actions.append("رفع الهدف إلى مستوى 1.272 فيبوناتشي")
                updates['target_price'] = fib_levels["1.272"]
            elif target_price < fib_levels["1.618"]:
                actions.append("رفع الهدف إلى مستوى 1.618 فيبوناتشي")
                updates['target_price'] = fib_levels["1.618"]
    else:
        # صفقة بيع (قصيرة)
        if current_price <= fib_levels[str(partial_profit_level)] and not signal_data.get('partial_profit_taken', False):
            # الوصول إلى مستوى جني الأرباح الجزئي
            actions.append(f"جني أرباح جزئي عند مستوى فيبوناتشي {partial_profit_level}")
            updates['partial_profit_taken'] = True
            # يمكن هنا إضافة كود لشراء جزء من الصفقة
            
        if current_price <= fib_levels[str(breakeven_level)] and stop_loss > entry_price:
            # تحريك وقف الخسارة إلى نقطة التعادل
            actions.append("تحريك وقف الخسارة إلى نقطة التعادل")
            updates['stop_loss'] = entry_price
            
        if current_price <= fib_levels["0.0"]:
            # الوصول إلى مستوى 0% فيبوناتشي - يمكن خفض الهدف إلى المستوى التالي
            if target_price > fib_levels["0.236"]:
                actions.append("خفض الهدف إلى مستوى 0.236 فيبوناتشي")
                updates['target_price'] = fib_levels["0.236"]
            elif target_price > fib_levels["0.382"]:
                actions.append("خفض الهدف إلى مستوى 0.382 فيبوناتشي")
                updates['target_price'] = fib_levels["0.382"]
    
    # تحديث أعلى سعر وصلت له العملة (للصفقات الطويلة)
    if is_long and current_price > signal_data.get('current_peak_price', entry_price):
        updates['current_peak_price'] = current_price
    
    # تحديث أدنى سعر وصلت له العملة (للصفقات القصيرة)
    if not is_long and current_price < signal_data.get('current_peak_price', entry_price):
        updates['current_peak_price'] = current_price
    
    # تحديث عدد الشموع منذ فتح الصفقة
    updates['candles_since_entry'] = signal_data.get('candles_since_entry', 0) + 1
    
    return {
        "actions": actions,
        "updates": updates,
        "fibonacci_levels": fib_levels
    }

# --- دوال الاستراتيجيات وإدارة الصفقات ---
def is_bullish_reversal_enhanced(df: pd.DataFrame) -> bool:
    """
    نسخة محسّنة للتعرف على أنماط الانعكاس الصاعدة
    """
    if len(df) < 3:
        return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3] if len(df) >= 3 else None
    
    # الشروط الحالية
    is_green_candle = bool(last['close'] > last['open'])
    candle_range = float(last['high'] - last['low'])
    body_size = float(last['close'] - last['open'])
    is_strong_body = (body_size / candle_range) >= 0.5 if candle_range > 0 else False
    
    is_bullish_engulfing = (
        is_green_candle and
        bool(prev['close'] < prev['open']) and  # الشمعة السابقة حمراء
        bool(last['close'] > prev['open']) and
        bool(last['open'] < prev['close'])
    )
    
    # أنماط إضافية للشموع الانعكاسية
    # نموذج المطرقة (Hammer)
    is_hammer = (
        is_green_candle and
        bool((last['open'] - last['low']) > 2 * (last['close'] - last['open'])) and
        bool((last['high'] - last['close']) < 0.1 * (last['close'] - last['open']))
    )
    
    # نموذج النجمية الصباحية (Morning Star) - مبسط
    is_morning_star = False
    if prev2 is not None:
        is_morning_star = (
            bool(prev2['close'] < prev2['open']) and  # الشمعة الأولى حمراء
            bool(prev['close'] < prev['open']) and  # الشمعة الثانية حمراء وصغيرة
            bool(abs(prev['close'] - prev['open']) < 0.3 * (prev2['close'] - prev2['open'])) and
            is_green_candle and  # الشمعة الثالثة خضراء
            bool(last['close'] > (prev2['open'] + prev2['close']) / 2)  # تغلق فوق منتصف الشمعة الأولى
        )
    
    # نموذج المطرقة المعكوسة (Inverted Hammer)
    is_inverted_hammer = (
        is_green_candle and
        bool((last['high'] - last['close']) > 2 * (last['close'] - last['open'])) and
        bool((last['open'] - last['low']) < 0.1 * (last['close'] - last['open']))
    )
    
    return is_strong_body or is_bullish_engulfing or is_hammer or is_morning_star or is_inverted_hammer

def check_bb_reversal_strategy_enhanced(df: pd.DataFrame) -> bool:
    """
    استراتيجية انعكاس بولينجر باند المحسنة مع مرشحات إضافية
    """
    if len(df) < 26: return False  # التأكد من وجود بيانات كافية للمؤشرات

    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # الشرط الأول: لامس السعر أو هبط تحت الحد السفلي لبولينجر باند في الشمعة السابقة
    price_touched_bb = bool(prev['low'] <= prev['bb_lower'])
    
    # الشرط الثاني: الشمعة الحالية هي شمعة انعكاسية صاعدة
    reversal_candle_appeared = is_bullish_reversal_enhanced(df)
    
    if not (price_touched_bb and reversal_candle_appeared):
        if price_touched_bb and not reversal_candle_appeared:
             log_rejection(df.name, "BB Reversal: No Reversal Candle", {"details": "السعر لمس BB لكن لم تظهر شمعة انعكاسية"})
        return False
    
    # الحصول على قيم المرشحات
    with FILTER_CONFIG["ADX_THRESHOLD"]["lock"]: adx_thresh = FILTER_CONFIG["ADX_THRESHOLD"]["value"]
    with FILTER_CONFIG["BB_STOCH_VOLUME_MULT"]["lock"]: vol_mult = FILTER_CONFIG["BB_STOCH_VOLUME_MULT"]["value"]
    
    # مرشح قوة الاتجاه ADX
    adx_strong = bool(last['adx'] > adx_thresh)
    if not adx_strong:
        log_rejection(df.name, "BB Reversal: ADX Filter Failed", {"adx": f"{last['adx']:.2f}", "threshold": adx_thresh})
        return False
    
    # مرشح حجم التداول
    volume_confirmed = bool(last['volume'] > (last['volume_sma_20'] * vol_mult))
    if not volume_confirmed:
        log_rejection(df.name, "BB Reversal: Volume Filter Failed", {"vol_multiplier": vol_mult})
        return False
    
    # مرشح إضافي: حالة التشبع البيعي لـ RSI
    rsi_oversold = bool(last['rsi'] < 30 or prev['rsi'] < 30)
    if not rsi_oversold:
        log_rejection(df.name, "BB Reversal: RSI Not Oversold", {"rsi": f"{last['rsi']:.2f}"})
        return False
    
    # مرشح إضافي: السعر لم يهبط كثيراً تحت الحد السفلي لبولينجر باند (ضمن 2%)
    bb_lower = float(prev['bb_lower'])
    price_below_bb_pct = ((bb_lower - float(prev['low'])) / bb_lower) * 100 if bb_lower > 0 else 0
    if price_below_bb_pct > 2:
        log_rejection(df.name, "BB Reversal: Price Too Far Below BB", {"percent_below": f"{price_below_bb_pct:.2f}%"})
        return False
    
    # إذا تحققت جميع الشروط، فهي إشارة صالحة
    logger.info(f"  -> [{df.name}] ✅ إشارة انعكاس BB محسنة (شمعة انعكاسية بعد ملامسة الحد السفلي).")
    return True

def check_macd_ema_strategy_enhanced(df: pd.DataFrame) -> bool:
    """
    استراتيجية MACD+EMA المحسنة مع مرشحات إضافية
    """
    if len(df) < 201: return False  # التأكد من وجود بيانات كافية لـ EMA200
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # الحصول على قيم المرشحات
    with FILTER_CONFIG["MACD_EMA_VOLUME_MULT"]["lock"]: vol_mult = 1.2  # قيمة افتراضية إذا لم يتم تعريفها
    
    # تأكيد الاتجاه (EMAs)
    trend_bullish = bool(last['ema_50'] > last['ema_200'])
    
    # تأكيد MACD
    macd_bullish = (
        bool(last['macd'] > last['macd_signal']) and  # MACD فوق الإشارة
        bool(prev['macd'] <= prev['macd_signal']) and  # تقاطع صاعد
        bool(last['macd'] < 0)  # MACD تحت الصفر (شراء في منطقة تشبع بيعي)
    )
    
    # مرشح RSI
    rsi_confirmed = bool(last['rsi'] > 30 and last['rsi'] < 70)  # ليس في منطقة تشبع极端
    
    # مرشح حجم التداول
    volume_confirmed = bool(last['volume'] > (last['volume_sma_20'] * vol_mult))
    
    # يجب أن تتحقق جميع الشروط
    if trend_bullish and macd_bullish and rsi_confirmed and volume_confirmed:
        logger.info(f"  -> [{df.name}] ✅ إشارة MACD_EMA محسنة.")
        return True
    
    # تسجيل أسباب الرفض
    if not trend_bullish:
        log_rejection(df.name, "MACD_EMA: Trend Filter Failed")
    elif not macd_bullish:
        log_rejection(df.name, "MACD_EMA: MACD Filter Failed")
    elif not rsi_confirmed:
        log_rejection(df.name, "MACD_EMA: RSI Filter Failed", {"rsi": f"{last['rsi']:.2f}"})
    elif not volume_confirmed:
        log_rejection(df.name, "MACD_EMA: Volume Filter Failed", {"vol_multiplier": vol_mult})
    
    return False

def check_sr_breakout_strategy_enhanced(df: pd.DataFrame) -> bool:
    """
    استراتيجية اختراق الدعم/المقاومة المحسنة مع مرشحات إضافية
    """
    if len(df) < 50: return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # الحصول على قيم المرشحات
    with FILTER_CONFIG["SR_BREAKOUT_VOLUME_MULT"]["lock"]: vol_mult = FILTER_CONFIG["SR_BREAKOUT_VOLUME_MULT"]["value"]
    
    # تحديد مستوى المقاومة (باستخدام أعلى سعر مع تأكيد الحجم)
    resistance_lookback = 20
    resistance_candidates = df.iloc[-resistance_lookback:-1]
    resistance_level = resistance_candidates['high'].max()
    
    # إيجاد الحجم عند مستوى المقاومة
    resistance_candle = resistance_candidates[resistance_candidates['high'] == resistance_level]
    resistance_volume = resistance_candle['volume'].iloc[0] if not resistance_candle.empty else 0
    
    # شروط الاختراق
    breakout = (
        bool(last['close'] > resistance_level) and  # أغلق فوق المقاومة
        bool(prev['close'] <= resistance_level) and  # الشمعة السابقة أغلقت عند أو تحت المقاومة
        bool((last['close'] - resistance_level) / resistance_level > 0.005)  # على الأقل 0.5% فوق المقاومة
    )
    
    if not breakout:
        return False
    
    # مرشح الحجم (الحجم الحالي يجب أن يكون أعلى بشكل ملحوظ من حجم المقاومة)
    volume_confirmed = bool(last['volume'] > (resistance_volume * vol_mult))
    if not volume_confirmed:
        log_rejection(df.name, "SR_Breakout: Volume Filter Failed", {"vol_multiplier": vol_mult})
        return False
    
    # مرشح الزخم (RSI يجب أن يكون قوياً لكن ليس في منطقة شراء مفرط)
    rsi_confirmed = bool(last['rsi'] > 50 and last['rsi'] < 80)
    if not rsi_confirmed:
        log_rejection(df.name, "SR_Breakout: RSI Filter Failed", {"rsi": f"{last['rsi']:.2f}"})
        return False
    
    # مرشح قوة الاتجاه ADX
    with FILTER_CONFIG["ADX_THRESHOLD"]["lock"]: adx_thresh = FILTER_CONFIG["ADX_THRESHOLD"]["value"]
    adx_strong = bool(last['adx'] > adx_thresh)
    if not adx_strong:
        log_rejection(df.name, "SR_Breakout: ADX Filter Failed", {"adx": f"{last['adx']:.2f}", "threshold": adx_thresh})
        return False
    
    logger.info(f"  -> [{df.name}] ✅ إشارة اختراق دعم/مقاومة محسنة.")
    return True

def check_triple_confirmation_strategy_enhanced(df: pd.DataFrame) -> bool:
    """
    استراتيجية التأكيد الثلاثي المحسنة مع مرشحات إضافية
    """
    if len(df) < 201: return False
    
    last = df.iloc[-1]
    
    # الحصول على قيم المرشحات
    with FILTER_CONFIG["TRIPLE_CONF_VOLUME_MULT"]["lock"]: vol_mult = FILTER_CONFIG["TRIPLE_CONF_VOLUME_MULT"]["value"]
    with FILTER_CONFIG["TRIPLE_CONF_MODE"]["lock"]: mode = FILTER_CONFIG["TRIPLE_CONF_MODE"]["value"]
    
    # تأكيد الاتجاه (EMAs)
    trend_confirmed = bool(last['ema_50'] > last['ema_200'])
    
    # تأكيد الزخم (MACD و RSI)
    macd_bullish = bool(last['macd'] > last['macd_signal'])
    rsi_bullish = bool(last['rsi'] > 55 and last['rsi'] < 80)  # قوي لكن ليس في منطقة شراء مفرط
    momentum_confirmed = macd_bullish and rsi_bullish
    
    # مرشح زخم إضافي (Stochastic)
    stoch_bullish = bool(last['stoch_rsi_k'] > last['stoch_rsi_d'] and last['stoch_rsi_k'] < 80)
    momentum_confirmed = momentum_confirmed and stoch_bullish
    
    # تأكيد حجم التداول
    volume_confirmed = bool(last['volume'] > (last['volume_sma_20'] * vol_mult))
    
    # مرشح التقلبات (ATR يجب أن يكون معقولاً)
    atr_normalized = float(last['atr'] / last['close'])  # ATR كنسبة من السعر
    volatility_confirmed = bool(0.01 < atr_normalized < 0.05)  # بين 1% و 5%
    
    # حساب الشروط المستوفاة حسب الوضع
    conditions = [trend_confirmed, momentum_confirmed, volume_confirmed, volatility_confirmed]
    conditions_met = sum(conditions)
    
    # في الوضع الصارم، يجب تحقيق جميع الشروط الأصلية، بالإضافة إلى التقلبات
    if mode == 'strict':
        required_conditions = 4
    else:  # الوضع المرن
        required_conditions = 3
    
    if conditions_met >= required_conditions:
        logger.info(f"  -> [{df.name}] ✅ إشارة التأكيد الثلاثي محسنة (الوضع: {mode}).")
        return True
    
    # تسجيل أسباب الرفض
    log_rejection(df.name, "Triple_Confirmation: Conditions Not Met", {
        "trend": trend_confirmed,
        "momentum": momentum_confirmed,
        "volume": volume_confirmed,
        "volatility": volatility_confirmed,
        "mode": mode,
        "met": conditions_met,
        "required": required_conditions
    })
    
    return False

def check_vwap_reversal_strategy_enhanced(df: pd.DataFrame) -> bool:
    """
    استراتيجية انعكاس VWAP المحسنة مع مرشحات إضافية
    """
    if len(df) < 21: return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # الحصول على قيم المرشحات
    with FILTER_CONFIG["VWAP_VOLUME_MULT"]["lock"]: vol_mult = FILTER_CONFIG["VWAP_VOLUME_MULT"]["value"]
    with FILTER_CONFIG["VWAP_REVERSAL_MODE"]["lock"]: mode = FILTER_CONFIG["VWAP_REVERSAL_MODE"]["value"]
    
    # شروط انعكاس VWAP
    vwap_reversal = bool(prev['close'] < prev['vwap'] and last['close'] > last['vwap'])
    
    # أنماط الشموع المحسنة
    is_bullish_engulfing = bool(prev['close'] < prev['open'] and last['close'] > last['open'] and last['close'] > prev['open'])
    is_hammer = bool((last['close'] > last['open']) and (last['open'] - last['low']) > 2 * (last['close'] - last['open']))
    is_doji = bool(abs(last['close'] - last['open']) < 0.1 * (last['high'] - last['low']) and last['close'] > last['open'])
    candle_confirmed = is_bullish_engulfing or is_hammer or is_doji
    
    # مرشح حجم التداول (المقارنة مع المتوسط الحديث)
    recent_volume_avg = df['volume'].iloc[-11:-1].mean()
    volume_confirmed = bool(last['volume'] > (recent_volume_avg * vol_mult))
    
    # مرشح إضافي: حالة التشبع البيعي لـ RSI
    rsi_oversold = bool(last['rsi'] < 40 or prev['rsi'] < 40)
    
    # مرشح إضافي: توافق الاتجاه (فريم أعلى)
    # هذا يتطلب بيانات الفريم الأعلى، لذا سنستخدم EMAs كبديل
    trend_aligned = bool(last['ema_50'] > last['ema_200'])
    
    # تحديد ما إذا كانت الإشارة مستوفاة حسب الوضع
    passes = False
    if mode == 'strict':
        passes = vwap_reversal and candle_confirmed and volume_confirmed and rsi_oversold and trend_aligned
    else:  # الوضع المرن
        conditions_met = sum([vwap_reversal, candle_confirmed, volume_confirmed, rsi_oversold, trend_aligned])
        passes = conditions_met >= 3
    
    if passes:
        logger.info(f"  -> [{df.name}] ✅ إشارة انعكاس VWAP محسنة (الوضع: {mode}).")
        return True
    
    # تسجيل أسباب الرفض
    log_rejection(df.name, "VWAP_Reversal: Conditions Not Met", {
        "reversal": vwap_reversal,
        "candle": candle_confirmed,
        "volume": volume_confirmed,
        "rsi_oversold": rsi_oversold,
        "trend_aligned": trend_aligned,
        "mode": mode
    })
    
    return False

def check_price_channel_strategy(df: pd.DataFrame) -> bool:
    """
    استراتيجية القناة السعرية القائمة على مبادئ قناة دونشيان (Donchian Channel)
    الدخول عند اختراق السعر للحد العلوي للقناة مع تأكيد الحجم
    """
    if len(df) < 30: return False
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # حساب قناة دونشيان
    channel_period = 20
    df['upper_channel'] = df['high'].rolling(window=channel_period).max()
    df['lower_channel'] = df['low'].rolling(window=channel_period).min()
    df['channel_middle'] = (df['upper_channel'] + df['lower_channel']) / 2
    
    # شروط اختراق القناة العلوية
    upper_breakout = (
        bool(last['close'] > last['upper_channel']) and  # أغلق فوق القناة العلوية
        bool(prev['close'] <= prev['upper_channel']) and  # الشمعة السابقة أغلقت عند أو تحت القناة العلوية
        bool((last['close'] - last['upper_channel']) / last['upper_channel'] > 0.005)  # على الأقل 0.5% فوق القناة
    )
    
    if not upper_breakout:
        return False
    
    # مرشح الحجم
    vol_mult = 1.3  # قيمة افتراضية
    volume_confirmed = bool(last['volume'] > (df['volume'].iloc[-11:-1].mean() * vol_mult))
    if not volume_confirmed:
        log_rejection(df.name, "Price Channel: Volume Filter Failed", {"vol_multiplier": vol_mult})
        return False
    
    # مرشح RSI (يجب أن يكون قوياً لكن ليس في منطقة شراء مفرط)
    rsi_confirmed = bool(last['rsi'] > 50 and last['rsi'] < 80)
    if not rsi_confirmed:
        log_rejection(df.name, "Price Channel: RSI Filter Failed", {"rsi": f"{last['rsi']:.2f}"})
        return False
    
    # مرشح قوة الاتجاه ADX
    with FILTER_CONFIG["ADX_THRESHOLD"]["lock"]: adx_thresh = FILTER_CONFIG["ADX_THRESHOLD"]["value"]
    adx_strong = bool(last['adx'] > adx_thresh)
    if not adx_strong:
        log_rejection(df.name, "Price Channel: ADX Filter Failed", {"adx": f"{last['adx']:.2f}", "threshold": adx_thresh})
        return False
    
    logger.info(f"  -> [{df.name}] ✅ إشارة اختراق القناة السعرية.")
    return True

def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> Optional[Decimal]:
    try:
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info: 
            logger.warning(f"[تعديل الكمية] لم يتم العثور على معلومات للرمز {symbol}")
            return None
            
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = Decimal(lot_size_filter['stepSize'])
            return (Decimal(str(quantity)) // step_size) * step_size
        return Decimal(str(quantity))
    except Exception as e:
        logger.error(f"[{symbol}] خطأ في تعديل الكمية لـ LOT_SIZE: {e}", exc_info=True)
        return None

def calculate_dynamic_position_size_enhanced(symbol: str, entry_price: float, atr_value: float, signal_strength: float = 1.0) -> Optional[Tuple[Decimal, float]]:
    """
    حساب حجم الصفقة الديناميكي المحسن مع مراعاة قوة الإشارة
    signal_strength: قيمة بين 0.5 و 1.5 تشير إلى قوة الإشارة
    """
    if not client: 
        logger.error(f"[حجم الصفقة] عميل Binance غير مهيأ للرمز {symbol}")
        return None
        
    try:
        with risk_per_trade_lock: current_risk_percent = RISK_PER_TRADE_PERCENT
        
        # تعديل المخاطرة بناءً على قوة الإشارة
        adjusted_risk_percent = current_risk_percent * signal_strength
        
        balance_response = client.get_asset_balance(asset='USDT')
        available_balance = Decimal(balance_response['free'])
        
        # استخدام نسبة المخاطرة المعدلة
        risk_amount_usdt = available_balance * (Decimal(str(adjusted_risk_percent)) / Decimal('100'))
        
        # حساب مسافة وقف الخسارة بناءً على ATR وقوة الإشارة
        # الإشارات الأقوى يمكن أن يكون لها وقف خسارة أضيق
        sl_multiplier = 2.5 - (signal_strength * 0.5)  # المدى من 2.0 إلى 2.5 بناءً على قوة الإشارة
        sl_distance = Decimal(str(atr_value)) * Decimal(str(sl_multiplier))
        
        actual_stop_loss_price = Decimal(str(entry_price)) - sl_distance
        risk_per_coin = Decimal(str(entry_price)) - actual_stop_loss_price
        
        if risk_per_coin <= 0:
            log_rejection(symbol, "Invalid Position Size", {"details": "Stop loss is at or above entry price based on ATR"})
            return None
        
        initial_quantity = risk_amount_usdt / risk_per_coin
        adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(initial_quantity))
        
        if adjusted_quantity is None or adjusted_quantity <= 0:
            log_rejection(symbol, "Lot Size Adjustment Failed")
            return None
        
        notional_value = adjusted_quantity * Decimal(str(entry_price))
        
        symbol_info = exchange_info_map.get(symbol)
        if symbol_info:
            min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] in ('MIN_NOTIONAL', 'NOTIONAL')), None)
            if min_notional_filter:
                min_notional = Decimal(min_notional_filter.get('minNotional', min_notional_filter.get('notional', '0')))
                if notional_value < min_notional:
                    log_rejection(symbol, "Min Notional Filter", {"value": f"{notional_value:.2f}", "min": f"{min_notional}"})
                    return None
        
        if notional_value > available_balance:
            log_rejection(symbol, "Insufficient Balance", {"required": f"{notional_value:.2f}", "available": f"{available_balance:.2f}"})
            return None
        
        return adjusted_quantity, float(actual_stop_loss_price)
    except Exception as e:
        logger.error(f"❌ [{symbol}] خطأ في حساب حجم الصفقة الديناميكي المحسّن: {e}", exc_info=True)
        return None

@rate_limiter(weight=1)
def place_order(symbol: str, side: str, quantity: Decimal, order_type: str = Client.ORDER_TYPE_MARKET) -> Optional[Dict]:
    if not client: 
        logger.error(f"[تنفيذ الأمر] عميل Binance غير مهيأ للرمز {symbol}")
        return None
        
    logger.info(f"➡️ [{symbol}] محاولة تنفيذ أمر {side} حقيقي لكمية {quantity}.")
    try:
        order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=str(quantity))
        log_and_notify('info', f"صفقة حقيقية: تم وضع أمر {side} لـ {quantity} {symbol}.", "REAL_TRADE")
        return order
    except Exception as e:
        logger.error(f"❌ [{symbol}] خطأ من المنصة عند تنفيذ الأمر: {e}")
        log_and_notify('error', f"فشل صفقة حقيقية: {symbol} | {e}", "REAL_TRADE_ERROR")
        return None

def insert_signal_into_db(signal_data: Dict) -> Optional[Dict]:
    if not check_db_connection() or not conn: 
        logger.error("[قاعدة البيانات] لا يوجد اتصال بقاعدة البيانات، تخطي حفظ الإشارة.")
        return None
        
    try:
        # حساب مستويات فيبوناتشي للإشارة
        df = get_data_for_symbol(signal_data['symbol'], SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
        if df is not None and not df.empty:
            df = calculate_all_features(df)
            fib_levels = calculate_fibonacci_levels(df)
            signal_data['fibonacci_levels'] = fib_levels
            
            # تحديد الهدف ووقف الخسارة بناءً على فيبوناتشي
            entry_price = float(signal_data['entry_price'])
            
            # تحديد ما إذا كانت الصفقة طويلة (شراء) أم قصيرة (بيع)
            is_long = signal_data.get('target_price', entry_price * 1.05) > entry_price
            
            if is_long:
                # صفقة شراء (طويلة)
                signal_data['stop_loss'] = fib_levels["0.0"] * 0.995  # وقف خسارة قليلاً تحت مستوى 0%
                signal_data['target_price'] = fib_levels["0.618"]  # الهدف الأولي عند مستوى 61.8%
            else:
                # صفقة بيع (قصيرة)
                signal_data['stop_loss'] = fib_levels["1.0"] * 1.005  # وقف خسارة قليلاً فوق مستوى 100%
                signal_data['target_price'] = fib_levels["0.382"]  # الهدف الأولي عند مستوى 38.2%
        
        atr_value = 0
        if USE_SMART_EXIT_SYSTEM:
            entry_price = float(signal_data['entry_price'])
            atr_value = float(signal_data.get('signal_details', {}).get('atr', 0))
            if atr_value > 0:
                exit_levels = {}
                for level, config in TAKE_PROFIT_LEVELS.items():
                    exit_levels[str(level)] = {
                        "target_price": entry_price + (atr_value * config['atr_multiplier']),
                        "exit_percentage": config['exit_percentage'],
                        "is_hit": False
                    }
                signal_data['exit_levels'] = exit_levels
                if 'target_price' not in signal_data:
                    signal_data['target_price'] = exit_levels[str(len(TAKE_PROFIT_LEVELS))]['target_price']
        
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, is_real_trade, quantity, original_quantity, order_id, current_peak_price, exit_levels, opened_at, fibonacci_levels, partial_profit_taken)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *;
            """, (
                signal_data['symbol'], signal_data['entry_price'], signal_data.get('target_price'), signal_data['stop_loss'],
                signal_data['strategy_name'], json.dumps(signal_data['signal_details'], cls=NpEncoder),
                signal_data.get('is_real_trade', False), signal_data.get('quantity'), signal_data.get('quantity'),
                signal_data.get('order_id'), signal_data['entry_price'], json.dumps(signal_data.get('exit_levels'), cls=NpEncoder),
                datetime.now(timezone.utc), json.dumps(signal_data.get('fibonacci_levels'), cls=NpEncoder), False
            ))
            saved_signal = cur.fetchone()
            conn.commit()
            logger.info(f"💾 [{signal_data['symbol']}] تم حفظ الإشارة الجديدة في قاعدة البيانات.")
            trade_type = "صفقة حقيقية" if signal_data.get('is_real_trade') else "إشارة ورقية"
            message = (f"🚨 *{trade_type} جديدة*\\n\\n*{signal_data['symbol']}* | `{signal_data['strategy_name']}`\\n\\n"
                       f"🔹 *الدخول:* `{signal_data['entry_price']:.4f}`\\n🛑 *وقف الخسارة:* `{signal_data['stop_loss']:.4f}`\\n\\n")
            if 'target_price' in signal_data:
                message += f"🎯 *الهدف:* `{signal_data['target_price']:.4f}`\\n"
            if 'fibonacci_levels' in signal_data:
                message += "*مستويات فيبوناتشي:*\\n"
                for level, price in signal_data['fibonacci_levels'].items():
                    message += f"  - {level}: `{price:.4f}`\\n"
            send_telegram_message(message)
            return dict(saved_signal)
    except Exception as e:
        logger.error(f"❌ [DB Insert] فشل إدراج الإشارة: {e}", exc_info=True)
        if conn: conn.rollback()
        return None

def update_signal_in_db(signal_id: int, updates: Dict):
    if not check_db_connection() or not conn: 
        logger.error("[قاعدة البيانات] لا يوجد اتصال بقاعدة البيانات، تخطي تحديث الإشارة.")
        return
        
    try:
        with conn.cursor() as cur:
            set_clauses = [sql.SQL("{} = %s").format(sql.Identifier(key)) for key in updates]
            values = [json.dumps(v, cls=NpEncoder) if isinstance(v, dict) else v for v in updates.values()]
            values.append(signal_id)
            query = sql.SQL("UPDATE signals SET {} WHERE id = %s").format(sql.SQL(', ').join(set_clauses))
            cur.execute(query, values)
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [DB Update] فشل تحديث الصفقة {signal_id}: {e}")
        if conn: conn.rollback()

def check_exit_conditions_enhanced(signal_data: Dict, current_price: float, df: pd.DataFrame) -> Tuple[bool, str]:
    """
    شروط الخروج المحسنة مع استراتيجيات متعددة
    Returns: (should_exit, reason)
    """
    symbol = signal_data['symbol']
    entry_price = float(signal_data['entry_price'])
    stop_loss = float(signal_data['stop_loss'])
    target_price = float(signal_data.get('target_price', entry_price * 1.05))
    
    # تحديد ما إذا كانت الصفقة طويلة (شراء) أم قصيرة (بيع)
    is_long = target_price > entry_price
    
    # ضبط الأهداف ووقف الخسارة ديناميكياً
    adjustment_result = adjust_dynamic_targets(symbol, signal_data, current_price, df)
    
    # تحديث الإشارة في قاعدة البيانات إذا كان هناك تحديثات
    if adjustment_result['updates']:
        update_signal_in_db(signal_data['id'], adjustment_result['updates'])
    
    # تسجيل الإجراءات التي تم اتخاذها
    for action in adjustment_result['actions']:
        logger.info(f"🔄 [{symbol}] {action}")
    
    # شروط الخروج الأساسية
    if is_long:
        # صفقة شراء (طويلة)
        if current_price <= stop_loss:
            return True, "Stop Loss Hit"
        
        if current_price >= target_price:
            return True, "Target Reached"
    else:
        # صفقة بيع (قصيرة)
        if current_price >= stop_loss:
            return True, "Stop Loss Hit"
        
        if current_price <= target_price:
            return True, "Target Reached"
    
    # شرط الخروج الزمني
    with FILTER_CONFIG["TIME_BASED_EXIT_CANDLES"]["lock"]: max_candles = FILTER_CONFIG["TIME_BASED_EXIT_CANDLES"]["value"]
    if signal_data.get('candles_since_entry', 0) >= max_candles:
        return True, "Time-based Exit"
    
    # شرط الخروج بناءً على تغير الاتجاه
    market_trend = calculate_market_trend_enhanced(symbol)
    if is_long and market_trend['market_state'] in ['STRONG_DOWNTREND', 'DOWNTREND']:
        return True, "Market Trend Reversal"
    
    if not is_long and market_trend['market_state'] in ['STRONG_UPTREND', 'UPTREND']:
        return True, "Market Trend Reversal"
    
    return False, ""

# --- دوال إدارة الصفقات والمراقبة ---
def get_open_signals_from_db() -> List[Dict]:
    if not check_db_connection() or not conn: 
        logger.error("[قاعدة البيانات] لا يوجد اتصال بقاعدة البيانات، تخطي جلب الإشارات.")
        return []
        
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status = 'open' ORDER BY opened_at DESC;")
            open_signals = cur.fetchall()
        return [dict(signal) for signal in open_signals]
    except Exception as e:
        logger.error(f"❌ [DB] فشل جلب الإشارات المفتوحة: {e}")
        if conn: conn.rollback()
        return []

def close_signal_in_db(signal_id: int, closing_price: float, reason: str) -> Optional[Dict]:
    if not check_db_connection() or not conn: 
        logger.error("[قاعدة البيانات] لا يوجد اتصال بقاعدة البيانات، تخطي إغلاق الإشارة.")
        return None
        
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE signals 
                SET status = 'closed', closing_price = %s, closed_at = %s, closing_reason = %s 
                WHERE id = %s 
                RETURNING *;
            """, (closing_price, datetime.now(timezone.utc), reason, signal_id))
            closed_signal = cur.fetchone()
            conn.commit()
        
        if closed_signal:
            signal_dict = dict(closed_signal)
            entry_price = float(signal_dict['entry_price'])
            profit_pct = ((closing_price - entry_price) / entry_price) * 100
            signal_dict['profit_percentage'] = profit_pct
            
            # تحديث أداء الاستراتيجية
            strategy_name = signal_dict['strategy_name']
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO strategy_performance (strategy_name, total_trades, winning_trades, total_pnl_percent)
                    VALUES (%s, 1, %s, %s)
                    ON CONFLICT (strategy_name) DO UPDATE SET
                        total_trades = strategy_performance.total_trades + 1,
                        winning_trades = strategy_performance.winning_trades + %s,
                        total_pnl_percent = strategy_performance.total_pnl_percent + %s;
                """, (
                    strategy_name, 
                    1 if profit_pct > 0 else 0,
                    profit_pct,
                    1 if profit_pct > 0 else 0,
                    profit_pct
                ))
                conn.commit()
            
            # إرسال إشعار
            trade_type = "صفقة حقيقية" if signal_dict.get('is_real_trade') else "إشارة ورقية"
            profit_emoji = "📈" if profit_pct > 0 else "📉"
            message = (f"✅ *{trade_type} مغلقة*\\n\\n*{signal_dict['symbol']}* | `{signal_dict['strategy_name']}`\\n\\n"
                       f"🔹 *الدخول:* `{entry_price:.4f}`\\n"
                       f"🔸 *الخروج:* `{closing_price:.4f}`\\n"
                       f"{profit_emoji} *الربح/الخسارة:* `{profit_pct:.2f}%`\\n"
                       f"📝 *السبب:* `{reason}`")
            send_telegram_message(message)
            
            return signal_dict
    except Exception as e:
        logger.error(f"❌ [DB] فشل إغلاق الإشارة {signal_id}: {e}")
        if conn: conn.rollback()
    return None

def monitor_open_signals():
    """
    مراقبة الإشارات المفتوحة وتطبيق شروط الخروج
    """
    open_signals = get_open_signals_from_db()
    if not open_signals:
        return
    
    logger.info(f"🔍 مراقبة {len(open_signals)} إشارة مفتوحة...")
    
    for signal in open_signals:
        symbol = signal['symbol']
        try:
            # جلب البيانات الحالية
            df = get_data_for_symbol(symbol, SIGNAL_GENERATION_TIMEFRAME, 2)
            if df is None or df.empty:
                continue
                
            df = calculate_all_features(df)
            if df.empty:
                continue
                
            current_price = float(df.iloc[-1]['close'])
            
            # فحص شروط الخروج
            should_exit, reason = check_exit_conditions_enhanced(signal, current_price, df)
            
            if should_exit:
                logger.info(f"⚠️ [{symbol}] إغلاق الإشارة بسبب: {reason}")
                close_signal_in_db(signal['id'], current_price, reason)
        except Exception as e:
            logger.error(f"❌ [{symbol}] خطأ أثناء مراقبة الإشارة: {e}", exc_info=True)

# --- دوال توليد الإشارات ---
def generate_signals_for_symbol(symbol: str) -> List[Dict]:
    """
    توليد إشارات التداول لعملة معينة
    """
    signals = []
    
    try:
        # جلب البيانات
        df = get_data_for_symbol(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
        if df is None or df.empty:
            return signals
            
        df = calculate_all_features(df)
        if df.empty:
            return signals
        
        df.name = symbol  # لإضافة اسم العملة إلى DataFrame
        
        # فحص حالة السوق
        market_trend = calculate_market_trend_enhanced(symbol)
        with market_state_lock:
            current_market_state[symbol] = market_trend
        
        # فلتر حالة السوق
        if market_trend['market_state'] in ['STRONG_DOWNTREND', 'DOWNTREND']:
            log_rejection(symbol, "Market Status Filter: BTC Downtrend (4h)", {"trend": market_trend['market_state']})
            return signals
        
        # فحص الاستراتيجيات المفعلة
        for strategy_name, config in STRATEGY_CONFIG.items():
            if not config['enabled']:
                continue
                
            with config['lock']:
                # فحص ما إذا كانت هناك إشارة مفتوحة بالفعل لهذه العملة
                open_signals = get_open_signals_from_db()
                existing_signal = next((s for s in open_signals if s['symbol'] == symbol and s['strategy_name'] == strategy_name), None)
                if existing_signal:
                    continue
                
                # فحص الاستراتيجية
                strategy_passed = False
                if strategy_name == "BB_Reversal":
                    strategy_passed = check_bb_reversal_strategy_enhanced(df)
                elif strategy_name == "MACD_EMA":
                    strategy_passed = check_macd_ema_strategy_enhanced(df)
                elif strategy_name == "SR_Breakout":
                    strategy_passed = check_sr_breakout_strategy_enhanced(df)
                elif strategy_name == "Triple_Confirmation":
                    strategy_passed = check_triple_confirmation_strategy_enhanced(df)
                elif strategy_name == "VWAP_Reversal":
                    strategy_passed = check_vwap_reversal_strategy_enhanced(df)
                elif strategy_name == "Price_Channel":
                    strategy_passed = check_price_channel_strategy(df)
                
                if strategy_passed:
                    # إنشاء بيانات الإشارة
                    last_candle = df.iloc[-1]
                    entry_price = float(last_candle['close'])
                    atr_value = float(last_candle['atr'])
                    
                    # حساب حجم الصفقة
                    position_result = calculate_dynamic_position_size_enhanced(symbol, entry_price, atr_value)
                    if position_result is None:
                        continue
                    
                    quantity, stop_loss = position_result
                    
                    # إنشاء الإشارة
                    signal_data = {
                        'symbol': symbol,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'strategy_name': strategy_name,
                        'signal_details': {
                            'atr': atr_value,
                            'rsi': float(last_candle['rsi']),
                            'adx': float(last_candle['adx']),
                            'macd': float(last_candle['macd']),
                            'market_trend': market_trend['market_state']
                        },
                        'quantity': float(quantity),
                        'is_real_trade': is_trading_enabled
                    }
                    
                    signals.append(signal_data)
    except Exception as e:
        logger.error(f"❌ [{symbol}] خطأ في توليد الإشارات: {e}", exc_info=True)
    
    return signals

def scan_all_symbols():
    """
    مسح جميع العملات وتوليد الإشارات
    """
    global validated_symbols_to_scan
    
    if not validated_symbols_to_scan:
        validated_symbols_to_scan = get_validated_symbols()
    
    if not validated_symbols_to_scan:
        logger.warning("⚠️ لا توجد عملات صالحة للمسح.")
        return
    
    logger.info(f"🔍 بدء مسح {len(validated_symbols_to_scan)} عملة...")
    
    # معالجة دفعات لتجنب الحظر
    for i in range(0, len(validated_symbols_to_scan), BATCH_SIZE):
        batch = validated_symbols_to_scan[i:i+BATCH_SIZE]
        logger.info(f"🔍 معالجة الدفعة {i//BATCH_SIZE + 1}/{(len(validated_symbols_to_scan)-1)//BATCH_SIZE + 1}...")
        
        for symbol in batch:
            try:
                signals = generate_signals_for_symbol(symbol)
                for signal in signals:
                    saved_signal = insert_signal_into_db(signal)
                    if saved_signal and is_trading_enabled:
                        # تنفيذ الصفقة الحقيقية
                        order = place_order(
                            symbol=saved_signal['symbol'],
                            side=Client.SIDE_BUY,
                            quantity=Decimal(str(saved_signal['quantity']))
                        )
                        if order:
                            update_signal_in_db(saved_signal['id'], {'order_id': order['orderId']})
            except Exception as e:
                logger.error(f"❌ [{symbol}] خطأ في معالجة العملة: {e}", exc_info=True)
        
        # استراحة بين الدفعات
        if i + BATCH_SIZE < len(validated_symbols_to_scan):
            time.sleep(5)
    
    logger.info("✅ اكتمل مسح جميع العملات.")

# --- واجهة الويب ---
app = Flask(__name__)
CORS(app)

@app.route('/')
def dashboard():
    """صفحة لوحة التحكم الرئيسية"""
    html_template = """
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>لوحة تحكم نظام التداول الآلي</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; }
            .navbar-brand { font-weight: bold; }
            .card { margin-bottom: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
            .card-header { background-color: #0d6efd; color: white; font-weight: bold; }
            .table { font-size: 0.9rem; }
            .profit-positive { color: #198754; font-weight: bold; }
            .profit-negative { color: #dc3545; font-weight: bold; }
            .status-open { color: #0d6efd; font-weight: bold; }
            .status-closed { color: #6c757d; }
            .signal-card { transition: transform 0.2s; }
            .signal-card:hover { transform: translateY(-5px); }
            .market-state { padding: 5px 10px; border-radius: 20px; font-size: 0.8rem; }
            .state-uptrend { background-color: #d4edda; color: #155724; }
            .state-downtrend { background-color: #f8d7da; color: #721c24; }
            .state-ranging { background-color: #fff3cd; color: #856404; }
            .btn-custom { margin: 2px; }
            .footer { margin-top: 50px; padding: 20px 0; background-color: #343a40; color: white; text-align: center; }
            .loading { text-align: center; padding: 20px; }
            .error-message { color: #dc3545; font-weight: bold; }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
            <div class="container">
                <a class="navbar-brand" href="#">
                    <i class="fas fa-robot"></i> نظام التداول الآلي V11.3
                </a>
                <div class="navbar-nav ms-auto">
                    <span class="navbar-text" id="trading-status">
                        <i class="fas fa-circle text-danger"></i> التداول معطل
                    </span>
                </div>
            </div>
        </nav>

        <div class="container mt-4">
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <i class="fas fa-chart-line"></i> حالة السوق الحالية
                        </div>
                        <div class="card-body" id="market-state-container">
                            <div class="loading">جاري تحميل حالة السوق...</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <i class="fas fa-signal"></i> الإشارات المفتوحة
                        </div>
                        <div class="card-body">
                            <div class="table-responsive">
                                <table class="table table-striped" id="open-signals-table">
                                    <thead>
                                        <tr>
                                            <th>العملة</th>
                                            <th>الاستراتيجية</th>
                                            <th>سعر الدخول</th>
                                            <th>الهدف</th>
                                            <th>وقف الخسارة</th>
                                            <th>الربح/الخسارة</th>
                                            <th>الإجراءات</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <!-- سيتم ملؤها ديناميكياً -->
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <i class="fas fa-history"></i> آخر الإشارات المغلقة
                        </div>
                        <div class="card-body">
                            <div class="table-responsive">
                                <table class="table table-striped" id="closed-signals-table">
                                    <thead>
                                        <tr>
                                            <th>العملة</th>
                                            <th>الاستراتيجية</th>
                                            <th>سعر الدخول</th>
                                            <th>سعر الخروج</th>
                                            <th>الربح/الخسارة</th>
                                            <th>السبب</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <!-- سيتم ملؤها ديناميكياً -->
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row mt-4">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <i class="fas fa-cog"></i> إعدادات النظام
                        </div>
                        <div class="card-body">
                            <form id="settings-form">
                                <div class="row">
                                    <div class="col-md-4">
                                        <div class="mb-3">
                                            <label for="risk-per-trade" class="form-label">نسبة المخاطرة للصفقة (%)</label>
                                            <input type="number" class="form-control" id="risk-per-trade" step="0.1" min="0.1" max="5">
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="mb-3">
                                            <label for="max-open-trades" class="form-label">الحد الأقصى للصفقات المفتوحة</label>
                                            <input type="number" class="form-control" id="max-open-trades" min="1" max="20">
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="mb-3">
                                            <label for="trend-sensitivity" class="form-label">حساسية اكتشاف الاتجاه</label>
                                            <input type="number" class="form-control" id="trend-sensitivity" step="0.1" min="0.1" max="1">
                                        </div>
                                    </div>
                                </div>
                                <div class="row">
                                    <div class="col-md-12">
                                        <button type="submit" class="btn btn-primary">
                                            <i class="fas fa-save"></i> حفظ الإعدادات
                                        </button>
                                        <button type="button" class="btn btn-success" id="toggle-trading">
                                            <i class="fas fa-play"></i> تفعيل التداول
                                        </button>
                                        <button type="button" class="btn btn-warning" id="scan-now">
                                            <i class="fas fa-sync-alt"></i> مسح الآن
                                        </button>
                                    </div>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row mt-4">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <i class="fas fa-bell"></i> آخر الإشعارات
                        </div>
                        <div class="card-body">
                            <div id="notifications-container">
                                <!-- سيتم ملؤها ديناميكياً -->
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <footer class="footer">
            <div class="container">
                <p>&copy; 2023 نظام التداول الآلي V11.3. جميع الحقوق محفوظة.</p>
            </div>
        </footer>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // تحديث البيانات بشكل دوري
            function updateData() {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        // تحديث حالة التداول
                        const tradingStatus = document.getElementById('trading-status');
                        if (data.trading_enabled) {
                            tradingStatus.innerHTML = '<i class="fas fa-circle text-success"></i> التداول مفعل';
                        } else {
                            tradingStatus.innerHTML = '<i class="fas fa-circle text-danger"></i> التداول معطل';
                        }
                        
                        // تحديث حالة السوق
                        const marketStateContainer = document.getElementById('market-state-container');
                        marketStateContainer.innerHTML = '';
                        
                        if (data.market_states && Object.keys(data.market_states).length > 0) {
                            const btcState = data.market_states['BTCUSDT'];
                            const stateClass = btcState.market_state.includes('UPTREND') ? 'state-uptrend' : 
                                              btcState.market_state.includes('DOWNTREND') ? 'state-downtrend' : 'state-ranging';
                            
                            marketStateContainer.innerHTML = `
                                <div class="d-flex align-items-center">
                                    <div class="market-state ${stateClass} me-2">
                                        ${btcState.market_state}
                                    </div>
                                    <div class="ms-2">
                                        <small>الدرجة: ${btcState.overall_score.toFixed(2)}</small>
                                    </div>
                                </div>
                            `;
                        } else {
                            marketStateContainer.innerHTML = '<div class="error-message">لا توجد بيانات حالة السوق</div>';
                        }
                        
                        // تحديث الإشارات المفتوحة
                        const openSignalsTable = document.getElementById('open-signals-table').getElementsByTagName('tbody')[0];
                        openSignalsTable.innerHTML = '';
                        
                        if (data.open_signals && data.open_signals.length > 0) {
                            data.open_signals.forEach(signal => {
                                const profitClass = signal.profit_percentage >= 0 ? 'profit-positive' : 'profit-negative';
                                const profitText = signal.profit_percentage ? `${signal.profit_percentage.toFixed(2)}%` : '-';
                                
                                const row = openSignalsTable.insertRow();
                                row.innerHTML = `
                                    <td>${signal.symbol}</td>
                                    <td>${signal.strategy_name}</td>
                                    <td>${signal.entry_price.toFixed(4)}</td>
                                    <td>${signal.target_price ? signal.target_price.toFixed(4) : '-'}</td>
                                    <td>${signal.stop_loss.toFixed(4)}</td>
                                    <td class="${profitClass}">${profitText}</td>
                                    <td>
                                        <button class="btn btn-sm btn-danger btn-custom" onclick="closeSignal(${signal.id})">
                                            <i class="fas fa-times"></i> إغلاق
                                        </button>
                                    </td>
                                `;
                            });
                        } else {
                            const row = openSignalsTable.insertRow();
                            row.innerHTML = '<td colspan="7" class="text-center">لا توجد إشارات مفتوحة</td>';
                        }
                        
                        // تحديث الإشارات المغلقة
                        const closedSignalsTable = document.getElementById('closed-signals-table').getElementsByTagName('tbody')[0];
                        closedSignalsTable.innerHTML = '';
                        
                        if (data.closed_signals && data.closed_signals.length > 0) {
                            data.closed_signals.forEach(signal => {
                                const profitClass = signal.profit_percentage >= 0 ? 'profit-positive' : 'profit-negative';
                                
                                const row = closedSignalsTable.insertRow();
                                row.innerHTML = `
                                    <td>${signal.symbol}</td>
                                    <td>${signal.strategy_name}</td>
                                    <td>${signal.entry_price.toFixed(4)}</td>
                                    <td>${signal.closing_price.toFixed(4)}</td>
                                    <td class="${profitClass}">${signal.profit_percentage.toFixed(2)}%</td>
                                    <td>${signal.closing_reason}</td>
                                `;
                            });
                        } else {
                            const row = closedSignalsTable.insertRow();
                            row.innerHTML = '<td colspan="6" class="text-center">لا توجد إشارات مغلقة</td>';
                        }
                        
                        // تحديث الإعدادات
                        if (data.settings) {
                            document.getElementById('risk-per-trade').value = data.settings.risk_per_trade;
                            document.getElementById('max-open-trades').value = data.settings.max_open_trades;
                            document.getElementById('trend-sensitivity').value = data.settings.trend_sensitivity;
                        }
                        
                        // تحديث زر التداول
                        const toggleButton = document.getElementById('toggle-trading');
                        if (data.trading_enabled) {
                            toggleButton.innerHTML = '<i class="fas fa-pause"></i> تعطيل التداول';
                            toggleButton.classList.remove('btn-success');
                            toggleButton.classList.add('btn-danger');
                        } else {
                            toggleButton.innerHTML = '<i class="fas fa-play"></i> تفعيل التداول';
                            toggleButton.classList.remove('btn-danger');
                            toggleButton.classList.add('btn-success');
                        }
                        
                        // تحديث الإشعارات
                        const notificationsContainer = document.getElementById('notifications-container');
                        notificationsContainer.innerHTML = '';
                        
                        if (data.notifications && data.notifications.length > 0) {
                            data.notifications.forEach(notification => {
                                const notificationDiv = document.createElement('div');
                                notificationDiv.className = 'alert alert-info alert-dismissible fade show';
                                notificationDiv.innerHTML = `
                                    <strong>${notification.timestamp}:</strong> ${notification.message}
                                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                                `;
                                notificationsContainer.appendChild(notificationDiv);
                            });
                        } else {
                            notificationsContainer.innerHTML = '<div class="text-center text-muted">لا توجد إشعارات جديدة</div>';
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching data:', error);
                        document.getElementById('market-state-container').innerHTML = '<div class="error-message">خطأ في جلب البيانات</div>';
                    });
            }
            
            // إغلاق إشارة
            function closeSignal(signalId) {
                if (confirm('هل أنت متأكد من إغلاق هذه الإشارة؟')) {
                    fetch(`/api/close_signal/${signalId}`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('تم إغلاق الإشارة بنجاح');
                            updateData();
                        } else {
                            alert('فشل إغلاق الإشارة: ' + data.error);
                        }
                    })
                    .catch(error => console.error('Error closing signal:', error));
                }
            }
            
            // إرسال نموذج الإعدادات
            document.getElementById('settings-form').addEventListener('submit', function(e) {
                e.preventDefault();
                
                const settings = {
                    risk_per_trade: parseFloat(document.getElementById('risk-per-trade').value),
                    max_open_trades: parseInt(document.getElementById('max-open-trades').value),
                    trend_sensitivity: parseFloat(document.getElementById('trend-sensitivity').value)
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
                        alert('تم حفظ الإعدادات بنجاح');
                        updateData();
                    } else {
                        alert('فشل حفظ الإعدادات: ' + data.error);
                    }
                })
                .catch(error => console.error('Error saving settings:', error));
            });
            
            // تبديل حالة التداول
            document.getElementById('toggle-trading').addEventListener('click', function() {
                const action = this.textContent.includes('تفعيل') ? 'enable' : 'disable';
                
                fetch(`/api/trading/${action}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert(`تم ${action === 'enable' ? 'تفعيل' : 'تعطيل'} التداول بنجاح`);
                        updateData();
                    } else {
                        alert(`فشل ${action === 'enable' ? 'تفعيل' : 'تعطيل'} التداول: ` + data.error);
                    }
                })
                .catch(error => console.error('Error toggling trading:', error));
            });
            
            // مسح الآن
            document.getElementById('scan-now').addEventListener('click', function() {
                if (confirm('هل تريد بدء مسح جديد للعملات الآن؟')) {
                    fetch('/api/scan', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('تم بدء المسح بنجاح');
                            updateData();
                        } else {
                            alert('فشل بدء المسح: ' + data.error);
                        }
                    })
                    .catch(error => console.error('Error starting scan:', error));
                }
            });
            
            // تحديث البيانات عند تحميل الصفحة
            updateData();
            
            // تحديث البيانات كل 30 ثانية
            setInterval(updateData, 30000);
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/api/status')
def api_status():
    """واجهة برمجية للحصول على حالة النظام"""
    global is_trading_enabled, current_market_state, validated_symbols_to_scan
    
    try:
        # الحصول على الإشارات المفتوحة
        open_signals = get_open_signals_from_db()
        
        # الحصول على آخر 10 إشارات مغلقة
        if check_db_connection() and conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM signals WHERE status = 'closed' ORDER BY closed_at DESC LIMIT 10;")
                closed_signals = [dict(signal) for signal in cur.fetchall()]
        else:
            closed_signals = []
        
        # الحصول على الإعدادات الحالية
        with risk_per_trade_lock: risk_percent = RISK_PER_TRADE_PERCENT
        with FILTER_CONFIG["TREND_SENSITIVITY"]["lock"]: trend_sensitivity = FILTER_CONFIG["TREND_SENSITIVITY"]["value"]
        
        # الحصول على الإشعارات الأخيرة
        with notifications_lock:
            notifications = notifications_cache[:10]
        
        # تحديث حالة السوق لـ BTC إذا لم تكن محدثة
        if 'BTCUSDT' not in current_market_state:
            logger.info("[API] تحديث حالة السوق لـ BTCUSDT")
            try:
                btc_trend = calculate_market_trend_enhanced('BTCUSDT')
                with market_state_lock:
                    current_market_state['BTCUSDT'] = btc_trend
            except Exception as e:
                logger.error(f"❌ [API] خطأ في تحديث حالة السوق لـ BTCUSDT: {e}")
                with market_state_lock:
                    current_market_state['BTCUSDT'] = {
                        "symbol": "BTCUSDT",
                        "overall_score": 0,
                        "market_state": "RANGING",
                        "timeframe_scores": {},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "error": str(e)
                    }
        
        return jsonify({
            'trading_enabled': is_trading_enabled,
            'market_states': current_market_state,
            'open_signals': open_signals,
            'closed_signals': closed_signals,
            'settings': {
                'risk_per_trade': risk_percent,
                'max_open_trades': MAX_OPEN_TRADES,
                'trend_sensitivity': trend_sensitivity
            },
            'notifications': notifications
        })
    except Exception as e:
        logger.error(f"❌ [API] خطأ في جلب الحالة: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['POST'])
def api_settings():
    """واجهة برمجية لتحديث إعدادات النظام"""
    try:
        data = request.get_json()
        
        if 'risk_per_trade' in data:
            with risk_per_trade_lock:
                global RISK_PER_TRADE_PERCENT
                RISK_PER_TRADE_PERCENT = float(data['risk_per_trade'])
        
        if 'max_open_trades' in data:
            global MAX_OPEN_TRADES
            MAX_OPEN_TRADES = int(data['max_open_trades'])
        
        if 'trend_sensitivity' in data:
            with FILTER_CONFIG["TREND_SENSITIVITY"]["lock"]:
                FILTER_CONFIG["TREND_SENSITIVITY"]["value"] = float(data['trend_sensitivity'])
        
        log_and_notify('info', "تم تحديث إعدادات النظام بنجاح", "SETTINGS_UPDATED")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"❌ [API] خطأ في تحديث الإعدادات: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/trading/<action>', methods=['POST'])
def api_trading(action):
    """واجهة برمجية لتفعيل/تعطيل التداول"""
    global is_trading_enabled
    
    try:
        if action == 'enable':
            with trading_status_lock:
                is_trading_enabled = True
            log_and_notify('info', "تم تفعيل التداول الحقيقي", "TRADING_ENABLED")
            return jsonify({'success': True})
        elif action == 'disable':
            with trading_status_lock:
                is_trading_enabled = False
            log_and_notify('info', "تم تعطيل التداول الحقيقي", "TRADING_DISABLED")
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'إجراء غير معروف'}), 400
    except Exception as e:
        logger.error(f"❌ [API] خطأ في تغيير حالة التداول: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/close_signal/<int:signal_id>', methods=['POST'])
def api_close_signal(signal_id):
    """واجهة برمجية لإغلاق إشارة"""
    try:
        # الحصول على الإشارة
        if not check_db_connection() or not conn:
            return jsonify({'success': False, 'error': 'لا يوجد اتصال بقاعدة البيانات'}), 500
        
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE id = %s AND status = 'open';", (signal_id,))
            signal = cur.fetchone()
        
        if not signal:
            return jsonify({'success': False, 'error': 'الإشارة غير موجودة أو مغلقة بالفعل'}), 404
        
        # جلب السعر الحالي
        symbol = signal['symbol']
        df = get_data_for_symbol(symbol, SIGNAL_GENERATION_TIMEFRAME, 2)
        if df is None or df.empty:
            return jsonify({'success': False, 'error': 'فشل جلب بيانات السعر'}), 500
        
        current_price = float(df.iloc[-1]['close'])
        
        # إغلاق الإشارة
        closed_signal = close_signal_in_db(signal_id, current_price, "Manual Close")
        
        if closed_signal:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'فشل إغلاق الإشارة'}), 500
    except Exception as e:
        logger.error(f"❌ [API] خطأ في إغلاق الإشارة: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/scan', methods=['POST'])
def api_scan():
    """واجهة برمجية لبدء مسح العملات"""
    try:
        # تشغيل المسح في خيط منفصل
        scan_thread = Thread(target=scan_all_symbols)
        scan_thread.daemon = True
        scan_thread.start()
        
        return jsonify({'success': True, 'message': 'تم بدء المسح بنجاح'})
    except Exception as e:
        logger.error(f"❌ [API] خطأ في بدء المسح: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

# --- الدوال الرئيسية لتشغيل النظام ---
def background_worker():
    """
    العامل الخلفي الذي يقوم بالمهام الدورية
    """
    while True:
        try:
            # مراقبة الإشارات المفتوحة
            monitor_open_signals()
            
            # مسح العملات كل ساعة
            current_time = datetime.now(timezone.utc)
            if current_time.minute == 0:
                scan_all_symbols()
            
            # تنظيف الذاكرة
            gc.collect()
            
            # انتظار 5 دقائق
            time.sleep(300)
        except Exception as e:
            logger.error(f"❌ [Background Worker] خطأ في العامل الخلفي: {e}", exc_info=True)
            time.sleep(60)

def main():
    """
    الدالة الرئيسية لتشغيل النظام
    """
    logger.info("🚀 بدء تشغيل نظام التداول الآلي V11.3...")
    
    # تهيئة قاعدة البيانات
    if not init_db():
        logger.error("❌ فشل تهيئة قاعدة البيانات. سيتم تشغيل النظام بدون قاعدة بيانات.")
    
    # تهيئة Redis
    if not init_redis():
        logger.warning("⚠️ فشل تهيئة Redis. سيتم تشغيل النظام بدون Redis.")
    
    # تهيئة عميل Binance
    global client
    try:
        if not API_KEY or not API_SECRET:
            logger.warning("⚠️ مفاتيح API غير معرفة. سيتم تشغيل النظام في وضع المحاكاة.")
            client = None
        else:
            client = Client(API_KEY, API_SECRET)
            logger.info("✅ تم تهيئة عميل Binance بنجاح.")
    except Exception as e:
        logger.error(f"❌ فشل تهيئة عميل Binance: {e}")
        client = None
    
    # جلب معلومات المنصة
    if client:
        if not get_exchange_info_map():
            logger.warning("⚠️ فشل جلب معلومات المنصة.")
    
    # الحصول على قائمة العملات
    global validated_symbols_to_scan
    validated_symbols_to_scan = get_validated_symbols()
    
    if not validated_symbols_to_scan:
        logger.warning("⚠️ لا توجد عملات صالحة للتداول. سيتم استخدام قائمة افتراضية.")
        validated_symbols_to_scan = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
    
    logger.info(f"✅ تم العثور على {len(validated_symbols_to_scan)} عملة صالحة للتداول.")
    
    # تشغيل العامل الخلفي
    worker_thread = Thread(target=background_worker)
    worker_thread.daemon = True
    worker_thread.start()
    
    # تشغيل واجهة الويب
    logger.info("🌐 بدء تشغيل واجهة الويب...")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except Exception as e:
        logger.error(f"❌ فشل تشغيل واجهة الويب: {e}")

if __name__ == "__main__":
    main()