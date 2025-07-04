# --- الاستيرادات والإعداد الأساسي ---
import time
import os
import sys
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
import redis
import psutil
from urllib.parse import urlparse
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException
from flask import Flask, request, Response, jsonify, render_template_string
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timedelta, timezone
from decouple import config
from typing import List, Dict, Optional, Any, Set
from sklearn.preprocessing import StandardScaler
from collections import deque
import warnings
import gc

# تجاهل التحذيرات غير الهامة
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- إعداد التسجيل ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v7_with_ichimoku.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV7_With_Ichimoku')

# --- تحميل متغيرات البيئة ---
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
    REDIS_URL: str = config('REDIS_URL', default='redis://localhost:6379/0')
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
    exit(1)

# --- الثوابت والمتغيرات العامة ---
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V7_With_Ichimoku'
MODEL_FOLDER: str = 'V7'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 30
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices"
MODEL_BATCH_SIZE: int = 5
DIRECT_API_CHECK_INTERVAL: int = 10
MEMORY_CLEANUP_INTERVAL: int = 3600  # تنظيف كل ساعة

# --- ثوابت المؤشرات الفنية ---
ADX_PERIOD: int = 14
BBANDS_PERIOD: int = 20
RSI_PERIOD: int = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD: int = 14
EMA_SLOW_PERIOD: int = 200
EMA_FAST_PERIOD: int = 50
BTC_CORR_PERIOD: int = 30
STOCH_RSI_PERIOD: int = 14
STOCH_K: int = 3
STOCH_D: int = 3
REL_VOL_PERIOD: int = 30
RSI_OVERBOUGHT: int = 70
RSI_OVERSOLD: int = 30
STOCH_RSI_OVERBOUGHT: int = 80
STOCH_RSI_OVERSOLD: int = 20
MODEL_CONFIDENCE_THRESHOLD = 0.70
MAX_OPEN_TRADES: int = 10
TRADE_AMOUNT_USDT: float = 10.0
USE_DYNAMIC_SL_TP = True
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.0
USE_BTC_TREND_FILTER = True
BTC_SYMBOL = 'BTCUSDT'
BTC_TREND_TIMEFRAME = '4h'
BTC_TREND_EMA_PERIOD = 50
MIN_PROFIT_PERCENTAGE_FILTER: float = 1.0

# --- ثوابت فلتر السرعة والتوقيت ---
USE_SPEED_FILTER: bool = True
SPEED_FILTER_ADX_THRESHOLD: float = 20.0
SPEED_FILTER_REL_VOL_THRESHOLD: float = 1.0
SPEED_FILTER_RSI_MIN: float = 30.0
SPEED_FILTER_RSI_MAX: float = 70.0

# --- المتغيرات العامة وقفل العمليات ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
ml_models_cache: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
notifications_cache = deque(maxlen=50)
notifications_lock = Lock()
signals_pending_closure: Set[int] = set()
closure_lock = Lock()
last_api_check_time = time.time()
last_memory_cleanup = time.time()

# --- دوال قاعدة البيانات ---
def init_db(retries: int = 5, delay: int = 5) -> None:
    global conn
    logger.info("[قاعدة البيانات] بدء تهيئة الاتصال...")
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            with conn.cursor() as cur:
                # التحقق من وجود عمود 'severity' وإضافته إذا لم يكن موجودًا
                cur.execute("""
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name='notifications' AND column_name='severity';
                """)
                if cur.fetchone() is None:
                    logger.warning("[قاعدة البيانات] عمود 'severity' غير موجود في جدول 'notifications'. سيتم إضافته الآن.")
                    cur.execute("ALTER TABLE notifications ADD COLUMN severity TEXT DEFAULT 'info';")
                    logger.info("[قاعدة البيانات] تم إضافة عمود 'severity' بنجاح.")

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY, 
                        symbol TEXT NOT NULL,
                        entry_price DOUBLE PRECISION NOT NULL,
                        target_price DOUBLE PRECISION NOT NULL,
                        stop_loss DOUBLE PRECISION NOT NULL,
                        status TEXT DEFAULT 'open',
                        closing_price DOUBLE PRECISION,
                        closed_at TIMESTAMP,
                        profit_percentage DOUBLE PRECISION,
                        strategy_name TEXT,
                        signal_details JSONB,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL,
                        message TEXT NOT NULL,
                        is_read BOOLEAN DEFAULT FALSE,
                        severity TEXT DEFAULT 'info'
                    );
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);
                    CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
                    CREATE INDEX IF NOT EXISTS idx_notifications_timestamp ON notifications(timestamp);
                """)
            conn.commit()
            logger.info("✅ [قاعدة البيانات] تم تهيئة جداول قاعدة البيانات بنجاح.")
            return
        except Exception as e:
            logger.error(f"❌ [قاعدة البيانات] خطأ في الاتصال (المحاولة {attempt + 1}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: 
                logger.critical("❌ [قاعدة البيانات] فشل الاتصال بعد عدة محاولات.")
                exit(1)

def check_db_connection() -> bool:
    global conn
    if conn is None or conn.closed != 0:
        logger.warning("[قاعدة البيانات] الاتصال مغلق، محاولة إعادة الاتصال...")
        init_db()
    try:
        if conn: 
            conn.cursor().execute("SELECT 1;")
            return True
        return False
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [قاعدة البيانات] فقدان الاتصال: {e}. محاولة إعادة الاتصال...")
        try: 
            init_db()
            return conn is not None and conn.closed == 0
        except Exception as retry_e: 
            logger.error(f"❌ [قاعدة البيانات] فشل إعادة الاتصال: {retry_e}")
            return False

def log_and_notify(level: str, message: str, notification_type: str):
    log_methods = {
        'info': logger.info,
        'warning': logger.warning,
        'error': logger.error,
        'critical': logger.critical
    }
    log_methods.get(level.lower(), logger.info)(message)
    
    if not check_db_connection() or not conn:
        return
        
    try:
        new_notification = {
            "timestamp": datetime.now().isoformat(),
            "type": notification_type,
            "message": message,
            "severity": level
        }
        
        with notifications_lock:
            notifications_cache.appendleft(new_notification)
            
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO notifications (type, message, severity) VALUES (%s, %s, %s);",
                (notification_type, message, level)
            )
        conn.commit()
        
    except Exception as e:
        logger.error(f"❌ [Notify DB] فشل حفظ التنبيه في قاعدة البيانات: {e}")
        if conn: conn.rollback()

def init_redis() -> None:
    global redis_client
    logger.info("[Redis] بدء تهيئة الاتصال...")
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] تم الاتصال بنجاح بخادم Redis.")
    except redis.exceptions.ConnectionError as e:
        logger.critical(f"❌ [Redis] فشل الاتصال بـ Redis على {REDIS_URL}. الخطأ: {e}")
        exit(1)
    except Exception as e:
        logger.critical(f"❌ [Redis] حدث خطأ غير متوقع أثناء تهيئة Redis: {e}")
        exit(1)

def load_open_signals_to_cache():
    if not check_db_connection() or not conn:
        return
    logger.info("🔄 [استرداد] تحميل الصفقات المفتوحة إلى الذاكرة المؤقتة...")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status = 'open';")
            open_signals = cur.fetchall()
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in open_signals:
                    open_signals_cache[signal['symbol']] = dict(signal)
            logger.info(f"✅ [استرداد] تم تحميل {len(open_signals)} صفقة مفتوحة.")
    except Exception as e:
        logger.error(f"❌ [استرداد] فشل تحميل الصفقات المفتوحة: {e}")
        if conn: conn.rollback()

def load_notifications_to_cache():
    if not check_db_connection() or not conn:
        return
    logger.info("🔄 [استرداد] تحميل آخر الإشعارات إلى الذاكرة المؤقتة...")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 50;")
            notifications = cur.fetchall()
            with notifications_lock:
                notifications_cache.clear()
                for n in reversed(notifications):
                     # تحويل كائن RealDictRow إلى قاموس عادي
                    notification_dict = dict(n)
                    # تحويل كائن datetime إلى سلسلة نصية متوافقة مع ISO 8601
                    if 'timestamp' in notification_dict and isinstance(notification_dict['timestamp'], datetime):
                        notification_dict['timestamp'] = notification_dict['timestamp'].isoformat()
                    notifications_cache.append(notification_dict)
            logger.info(f"✅ [استرداد] تم تحميل {len(notifications)} إشعار.")
    except Exception as e:
        logger.error(f"❌ [استرداد] فشل تحميل الإشعارات: {e}")
        if conn: conn.rollback()

def recover_cache_state():
    """استرداد حالة الذاكرة المؤقتة في حالة الأخطاء"""
    logger.info("🔄 [استرداد] بدء استرداد حالة الذاكرة المؤقتة...")
    try:
        # استرداد الإشارات المفتوحة
        load_open_signals_to_cache()
        
        # استرداد التنبيهات
        load_notifications_to_cache()
        
        # تنظيف الإشارات العالقة
        with closure_lock:
            signals_pending_closure.clear()
            
        # تنظيف النماذج المخزنة مؤقتاً
        ml_models_cache.clear()
        
        # تنظيف الذاكرة
        gc.collect()
            
        logger.info("✅ [استرداد] تم استرداد حالة الذاكرة المؤقتة بنجاح")
    except Exception as e:
        logger.error(f"❌ [استرداد] فشل استرداد حالة الذاكرة المؤقتة: {e}")

def cleanup_memory():
    """تنظيف دوري للذاكرة"""
    global last_memory_cleanup
    
    current_time = time.time()
    if current_time - last_memory_cleanup > MEMORY_CLEANUP_INTERVAL:
        logger.info("🧹 [تنظيف الذاكرة] بدء التنظيف الدوري...")
        
        # تنظيف ذاكرة النماذج
        ml_models_cache.clear()
        
        # تشغيل جامع النفايات
        gc.collect()
        
        last_memory_cleanup = current_time
        logger.info("✅ [تنظيف الذاكرة] اكتمل التنظيف الدوري")

# --- دوال Binance والبيانات ---
def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    logger.info(f"ℹ️ [التحقق] قراءة الرموز من '{filename}' والتحقق منها مع Binance...")
    if not client: 
        logger.error("❌ [التحقق] كائن Binance client غير مهيأ.")
        return []
        
    try:
        # تحديد المسار بناءً على مكان تنفيذ السكربت
        if getattr(sys, 'frozen', False):
            # إذا كان السكربت مجمداً (e.g., via PyInstaller)
            script_dir = os.path.dirname(sys.executable)
        else:
            # الوضع العادي
            script_dir = os.path.dirname(os.path.abspath(__file__))
        
        file_path = os.path.join(script_dir, filename)
        
        if not os.path.exists(file_path):
            logger.error(f"❌ [التحقق] ملف العملات '{file_path}' غير موجود.")
            return []

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
            
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        
        try:
            exchange_info = client.get_exchange_info()
            active = {s['symbol'] for s in exchange_info['symbols'] 
                     if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'}
        except BinanceAPIException as e:
            logger.error(f"❌ [Binance API] فشل جلب معلومات التداول: {e}")
            return []
            
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ [التحقق] سيقوم البوت بمراقبة {len(validated)} عملة معتمدة.")
        return validated
        
    except Exception as e:
        logger.error(f"❌ [التحقق] حدث خطأ أثناء التحقق من الرموز: {e}", exc_info=True)
        return []

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client:
        return None
        
    try:
        start_dt = datetime.now(timezone.utc) - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines:
            return None
            
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 
                                         'volume', 'close_time', 'quote_volume', 'trades',
                                         'taker_buy_base', 'taker_buy_quote', 'ignore'])
                                         
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        numeric_cols = {
            'open': 'float32', 'high': 'float32', 'low': 'float32', 
            'close': 'float32', 'volume': 'float32'
        }
        df = df.astype(numeric_cols)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        
        return df.dropna()
        
    except BinanceAPIException as e:
        logger.warning(f"⚠️ [API Binance] خطأ في جلب بيانات {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ [البيانات] خطأ أثناء جلب البيانات التاريخية لـ {symbol}: {e}")
        return None
    finally:
        gc.collect()

# --- دوال إغلاق الإشارات ---
def close_signal(signal_id: int, status: str, closing_price: float, closed_by: str = "auto"):
    if not all([check_db_connection(), conn, signal_id, status, closing_price]):
        logger.error(f"❌ [إغلاق] بيانات غير كافية لإغلاق الإشارة ID: {signal_id}")
        return

    try:
        with conn.cursor() as cur:
            # جلب سعر الدخول لحساب الربح
            cur.execute("SELECT entry_price, symbol FROM signals WHERE id = %s;", (signal_id,))
            signal_data = cur.fetchone()

            if not signal_data:
                logger.error(f"❌ [إغلاق] لم يتم العثور على الإشارة ID: {signal_id} في قاعدة البيانات.")
                return

            entry_price = signal_data['entry_price']
            symbol = signal_data['symbol']

            # حساب نسبة الربح/الخسارة
            profit_percentage = ((closing_price - entry_price) / entry_price) * 100

            # تحديث الإشارة في قاعدة البيانات
            cur.execute("""
                UPDATE signals 
                SET status = %s, closing_price = %s, profit_percentage = %s, closed_at = NOW()
                WHERE id = %s;
            """, (status, closing_price, profit_percentage, signal_id))
            
            conn.commit()
            
            # إرسال إشعار
            message = (f"✅ إغلاق {symbol}: {status} | "
                       f"سعر الإغلاق: {closing_price:.4f} | "
                       f"الربح: {profit_percentage:.2f}% | "
                       f"بواسطة: {closed_by}")
            log_and_notify('info', message, 'CLOSE_SIGNAL')

            # إزالة الإشارة من الذاكرة المؤقتة
            with signal_cache_lock:
                open_signals_cache.pop(symbol, None)
            
            logger.info(f"✅ [إغلاق] تم إغلاق الإشارة {signal_id} ({symbol}) بنجاح.")

    except Exception as e:
        logger.error(f"❌ [إغلاق] فشل في إغلاق الإشارة {signal_id}: {e}")
        if conn: conn.rollback()
    finally:
        with closure_lock:
            signals_pending_closure.discard(signal_id)
        gc.collect()

def initiate_signal_closure(symbol: str, signal_to_close: Dict, status: str, closing_price: float):
    signal_id = signal_to_close.get('id')
    
    if not all([signal_id, symbol, status, closing_price]):
        logger.error(f"❌ [CLOSURE] بيانات غير صالحة لإغلاق الإشارة: ID={signal_id}, Symbol={symbol}")
        return
    
    with closure_lock:
        if signal_id in signals_pending_closure:
            logger.warning(f"⚠️ [CLOSURE] الإشارة {signal_id} قيد الإغلاق بالفعل")
            return
        signals_pending_closure.add(signal_id)
    
    try:
        with signal_cache_lock:
            signal_data = open_signals_cache.pop(symbol, None)
            
        if signal_data:
            logger.info(f"⚡ [CLOSURE] تم إزالة الإشارة {signal_id} من الذاكرة المؤقتة. بدء عملية الإغلاق...")
            # استخدام الدالة الجديدة لإغلاق الإشارة في خيط منفصل
            Thread(target=close_signal, args=(signal_id, status, closing_price, "auto_monitor")).start()
        else:
            logger.warning(f"⚠️ [CLOSURE] لم يتم العثور على الإشارة {signal_id} في الذاكرة المؤقتة")
            with closure_lock:
                signals_pending_closure.discard(signal_id)
                
    except Exception as e:
        logger.error(f"❌ [CLOSURE] خطأ في بدء إغلاق الإشارة {signal_id}: {e}")
        with closure_lock:
            signals_pending_closure.discard(signal_id)
    finally:
        gc.collect()

# --- دوال WebSocket ومراقبة التداول ---
def handle_price_update_message(msg: List[Dict[str, Any]]) -> None:
    global redis_client
    
    try:
        if not isinstance(msg, list):
            logger.warning(f"⚠️ [WebSocket] تم استلام رسالة بتنسيق غير متوقع: {type(msg)}")
            return
            
        if not redis_client:
            logger.error("❌ [WebSocket] كائن Redis غير مهيأ. لا يمكن حفظ الأسعار.")
            return

        price_updates = {
            item.get('s'): float(item.get('c', 0)) 
            for item in msg 
            if item.get('s') and item.get('c')
        }
        
        if price_updates:
            redis_client.hset(REDIS_PRICES_HASH_NAME, mapping=price_updates)
            
    except Exception as e:
        logger.error(f"❌ [WebSocket Price Updater] خطأ في معالجة رسالة السعر: {e}", exc_info=True)
    finally:
        gc.collect()

def trade_monitoring_loop():
    global last_api_check_time
    
    MAX_RECONNECT_ATTEMPTS = 3
    reconnect_attempts = 0
    
    logger.info("✅ [Trade Monitor] بدء خيط المراقبة")
    
    while True:
        try:
            cleanup_memory()

            with signal_cache_lock:
                signals_to_check = dict(open_signals_cache)

            if not signals_to_check:
                time.sleep(1)
                continue
                
            if not redis_client or not client:
                reconnect_attempts += 1
                if reconnect_attempts > MAX_RECONNECT_ATTEMPTS:
                    logger.error("❌ [Trade Monitor] فشل الاتصال بالخدمات الأساسية بعد عدة محاولات")
                    time.sleep(60)
                    reconnect_attempts = 0
                continue
                
            reconnect_attempts = 0

            perform_direct_api_check = (time.time() - last_api_check_time) > DIRECT_API_CHECK_INTERVAL
            if perform_direct_api_check:
                logger.debug(f"🔄 [Direct API Check] حان وقت الفحص المباشر من API.")
                last_api_check_time = time.time()

            symbols_to_fetch = list(signals_to_check.keys())
            redis_prices_list = redis_client.hmget(REDIS_PRICES_HASH_NAME, symbols_to_fetch)
            redis_prices = {symbol: price for symbol, price in zip(symbols_to_fetch, redis_prices_list)}

            for symbol, signal in signals_to_check.items():
                signal_id = signal.get('id')
                
                with closure_lock:
                    if signal_id in signals_pending_closure:
                        continue

                price = None
                price_source = "None"

                if perform_direct_api_check:
                    try:
                        ticker = client.get_symbol_ticker(symbol=symbol)
                        price = float(ticker['price'])
                        price_source = "Direct API"
                    except Exception as e:
                        logger.error(f"❌ [Direct API Check] فشل جلب السعر لـ {symbol}: {e}")
                        if redis_prices.get(symbol):
                            price = float(redis_prices[symbol])
                            price_source = "Redis (Fallback)"
                else:
                    if redis_prices.get(symbol):
                        price = float(redis_prices[symbol])
                        price_source = "Redis"

                target_price = float(signal.get('target_price', 0))
                stop_loss_price = float(signal.get('stop_loss', 0))

                logger.debug(f"[MONITOR] ID:{signal_id} | {symbol} | Price: {price} ({price_source}) | TP: {target_price} | SL: {stop_loss_price}")
                
                if not all([price, target_price > 0, stop_loss_price > 0]):
                    logger.warning(f"  -> [SKIP] بيانات غير صالحة أو سعر غير متوفر لـ {symbol} (ID: {signal_id}).")
                    continue
                
                status_to_set = None
                if price >= target_price:
                    status_to_set = 'target_hit'
                elif price <= stop_loss_price:
                    status_to_set = 'stop_loss_hit'

                if status_to_set:
                    logger.info(f"✅ [TRIGGER] ID:{signal_id} | {symbol} | تحقق شرط '{status_to_set}'.")
                    initiate_signal_closure(symbol, signal, status_to_set, price)

            time.sleep(0.2)

        except Exception as e:
            logger.error(f"❌ [Trade Monitor] خطأ في حلقة المراقبة: {e}", exc_info=True)
            time.sleep(5)
        finally:
            gc.collect()

def run_websocket_manager() -> None:
    logger.info("ℹ️ [WebSocket] بدء مدير WebSocket...")
    
    MAX_RECONNECT_ATTEMPTS = 5
    reconnect_delay = 10
    attempt = 0
    
    while attempt < MAX_RECONNECT_ATTEMPTS:
        try:
            twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
            twm.start()
            
            twm.start_miniticker_socket(callback=handle_price_update_message)
            
            logger.info("✅ [WebSocket] تم الاتصال والاستماع إلى 'All Market Mini Tickers' بنجاح.")
            twm.join()
            break
            
        except Exception as e:
            attempt += 1
            logger.error(f"❌ [WebSocket] فشل الاتصال (المحاولة {attempt}/{MAX_RECONNECT_ATTEMPTS}): {e}")
            
            if attempt < MAX_RECONNECT_ATTEMPTS:
                time.sleep(reconnect_delay)
                reconnect_delay *= 2
            else:
                logger.critical("❌ [WebSocket] فشل الاتصال بعد عدة محاولات. إيقاف البوت.")
                os._exit(1)

# --- منطق الاستراتيجية والتداول الرئيسي (بدون تغيير) ---
# ... (All strategy and feature calculation functions remain the same)
def fetch_sr_levels_from_db(symbol: str) -> pd.DataFrame: return pd.DataFrame()
def fetch_ichimoku_features_from_db(symbol: str, timeframe: str) -> pd.DataFrame: return pd.DataFrame()
def calculate_ichimoku_based_features(df: pd.DataFrame) -> pd.DataFrame: return df
def calculate_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame: return df
def calculate_sr_features(df: pd.DataFrame, sr_levels_df: pd.DataFrame) -> pd.DataFrame: return df
def calculate_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame: return df
def load_ml_model_bundle_from_folder(symbol: str) -> Optional[Dict[str, Any]]: return None
class TradingStrategy:
    def __init__(self, symbol: str): self.symbol = symbol
    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame, sr_levels_df: pd.DataFrame, ichimoku_df: pd.DataFrame) -> Optional[pd.DataFrame]: return None
    def generate_signal(self, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]: return None
def main_loop(): 
    logger.info("[الحلقة الرئيسية] تم تعطيلها مؤقتًا للتركيز على إصلاحات الواجهة.")
    while True: time.sleep(3600)
# ...

# --- تهيئة Flask وتعريف المسارات ---
app = Flask(__name__)
CORS(app)

@app.route('/api/health')
def health_check():
    try:
        db_ok = check_db_connection()
        checks = {
            'redis': bool(redis_client and redis_client.ping()),
            'database': db_ok,
            'binance': bool(client and client.ping()),
            'symbols_loaded': len(validated_symbols_to_scan) > 0,
            'memory_usage': f"{psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB"
        }
        
        status = 'healthy' if all(checks.values()) else 'degraded'
        return jsonify({
            'status': status,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'checks': checks
        }), 200 if status == 'healthy' else 503
        
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/notifications', methods=['GET'])
def get_notifications():
    try:
        limit = min(int(request.args.get('limit', 50)), 100)
        with notifications_lock:
            # تحويل كائنات deque إلى قائمة من القواميس
            recent_notifications = [dict(n) for n in list(notifications_cache)][:limit]
        return jsonify(recent_notifications)
    except Exception as e:
        logger.error(f"❌ [API] خطأ في جلب الإشعارات: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/signals', methods=['GET'])
def get_signals():
    if not check_db_connection() or not conn:
        return jsonify({'error': 'Database connection failed'}), 503
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals ORDER BY created_at DESC;")
            all_signals = [dict(s) for s in cur.fetchall()]
        
        # جلب الأسعار الحالية للصفقات المفتوحة
        open_symbols = [s['symbol'] for s in all_signals if s['status'] == 'open']
        current_prices = {}
        if open_symbols and redis_client:
            prices = redis_client.hmget(REDIS_PRICES_HASH_NAME, open_symbols)
            current_prices = {symbol: float(price) if price else None for symbol, price in zip(open_symbols, prices)}

        # إضافة السعر الحالي ونسبة الربح/الخسارة للصفقات المفتوحة
        for signal in all_signals:
            if signal['status'] == 'open':
                price = current_prices.get(signal['symbol'])
                signal['current_price'] = price
                if price and signal.get('entry_price'):
                    pnl = ((price - signal['entry_price']) / signal['entry_price']) * 100
                    signal['pnl_pct'] = pnl
                else:
                    signal['pnl_pct'] = 0
        
        return jsonify(all_signals)
    except Exception as e:
        logger.error(f"❌ [API] خطأ في جلب الإشارات: {e}")
        if conn: conn.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    if not check_db_connection() or not conn:
        return jsonify({'error': 'Database connection failed'}), 503
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT status, profit_percentage FROM signals;")
            signals = cur.fetchall()
        
        stats = {
            'open_trades_count': sum(1 for s in signals if s['status'] == 'open'),
            'total_profit_pct': sum(s['profit_percentage'] for s in signals if s['profit_percentage'] is not None),
            'targets_hit_all_time': sum(1 for s in signals if s['status'] == 'target_hit'),
            'stops_hit_all_time': sum(1 for s in signals if s['status'] == 'stop_loss_hit'),
        }
        return jsonify(stats)
    except Exception as e:
        logger.error(f"❌ [API] خطأ في جلب الإحصائيات: {e}")
        if conn: conn.rollback()
        return jsonify({'error': str(e)}), 500
        
@app.route('/api/market_status', methods=['GET'])
def get_market_status():
    try:
        btc_df = fetch_historical_data(BTC_SYMBOL, BTC_TREND_TIMEFRAME, 50)
        if btc_df is None or btc_df.empty:
            return jsonify({'error': 'Could not fetch BTC data'}), 500

        ema = btc_df['close'].ewm(span=BTC_TREND_EMA_PERIOD, adjust=False).mean()
        is_uptrend = btc_df['close'].iloc[-1] > ema.iloc[-1]
        
        return jsonify({
            'btc_trend': {
                'is_uptrend': is_uptrend,
                'last_price': btc_df['close'].iloc[-1],
                'ema_value': ema.iloc[-1]
            }
        })
    except Exception as e:
        logger.error(f"❌ [API] خطأ في جلب حالة السوق: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/close/<int:signal_id>', methods=['POST'])
def manual_close_signal(signal_id):
    if not redis_client:
        return jsonify({'error': 'Redis client not available'}), 503

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT symbol, status FROM signals WHERE id = %s", (signal_id,))
            signal = cur.fetchone()

        if not signal:
            return jsonify({'error': 'Signal not found'}), 404
        if signal['status'] != 'open':
            return jsonify({'error': 'Signal is not open'}), 400

        symbol = signal['symbol']
        current_price_str = redis_client.hget(REDIS_PRICES_HASH_NAME, symbol)
        
        if not current_price_str:
            # Fallback to API if Redis price is missing
            ticker = client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
        else:
            current_price = float(current_price_str)

        # إغلاق الإشارة في خيط منفصل لتجنب حظر الطلب
        Thread(target=close_signal, args=(signal_id, 'manual_close', current_price, 'dashboard')).start()
        
        return jsonify({'message': f'Close request for signal {signal_id} ({symbol}) accepted.'})

    except Exception as e:
        logger.error(f"❌ [API] خطأ في الإغلاق اليدوي للإشارة {signal_id}: {e}")
        return jsonify({'error': str(e)}), 500

# --- دوال التهيئة والتشغيل ---
def init_services():
    global client, validated_symbols_to_scan
    try:
        init_db()
        init_redis()
        client = Client(API_KEY, API_SECRET)
        validated_symbols_to_scan = get_validated_symbols()
        recover_cache_state()
        return True
    except Exception as e:
        logger.critical(f"❌ [تهيئة] فشل في تهيئة الخدمات: {e}", exc_info=True)
        return False

def start_background_tasks():
    try:
        Thread(target=trade_monitoring_loop, daemon=True, name="TradeMonitor").start()
        Thread(target=run_websocket_manager, daemon=True, name="WebSocketManager").start()
        # Thread(target=main_loop, daemon=True, name="MainLoop").start() # معطل مؤقتاً
        return True
    except Exception as e:
        logger.critical(f"❌ [المهام الخلفية] فشل في بدء المهام الخلفية: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    try:
        if not init_services():
            logger.critical("❌ فشل في تهيئة الخدمات الأساسية. إيقاف البوت.")
            sys.exit(1)
            
        if not start_background_tasks():
            logger.critical("❌ فشل في بدء المهام الخلفية. إيقاف البوت.")
            sys.exit(1)
            
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"🚀 بدء تشغيل خادم Flask على المنفذ {port}...")
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        logger.critical(f"❌ [التشغيل الرئيسي] خطأ حرج: {e}", exc_info=True)
        sys.exit(1)
