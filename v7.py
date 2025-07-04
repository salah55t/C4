# --- الاستيرادات والإعداد الأساسي ---
import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
import redis
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
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        
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
        
        # تحويل الأعمدة إلى نوع float32 لتوفير الذاكرة
        numeric_cols = {
            'open': 'float32',
            'high': 'float32',
            'low': 'float32', 
            'close': 'float32',
            'volume': 'float32'
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
        gc.collect()  # تنظيف الذاكرة بعد معالجة البيانات

def fetch_sr_levels_from_db(symbol: str) -> pd.DataFrame:
    if not check_db_connection() or not conn:
        return pd.DataFrame()
        
    query = """
        SELECT level_price, level_type, score 
        FROM support_resistance_levels 
        WHERE symbol = %s
        AND updated_at > NOW() - INTERVAL '24 HOURS';
    """
    
    try:
        with conn.cursor() as cur:
            cur.execute(query, (symbol,))
            levels = cur.fetchall()
            
            if not levels:
                return pd.DataFrame()
                
            return pd.DataFrame(levels)
            
    except Exception as e:
        logger.error(f"❌ [S/R Fetch Bot] Could not fetch S/R levels for {symbol}: {e}")
        if conn: conn.rollback()
        return pd.DataFrame()
    finally:
        gc.collect()

def fetch_ichimoku_features_from_db(symbol: str, timeframe: str) -> pd.DataFrame:
    if not check_db_connection() or not conn:
        return pd.DataFrame()
        
    query = """
        SELECT timestamp, tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
        FROM ichimoku_features
        WHERE symbol = %s 
        AND timeframe = %s
        AND timestamp > NOW() - INTERVAL '24 HOURS'
        ORDER BY timestamp;
    """
    
    try:
        with conn.cursor() as cur:
            cur.execute(query, (symbol, timeframe))
            features = cur.fetchall()
            
            if not features:
                return pd.DataFrame()
                
            df_ichimoku = pd.DataFrame(features)
            df_ichimoku['timestamp'] = pd.to_datetime(df_ichimoku['timestamp'], utc=True)
            df_ichimoku.set_index('timestamp', inplace=True)
            
            return df_ichimoku
            
    except Exception as e:
        logger.error(f"❌ [Ichimoku Fetch Bot] Could not fetch Ichimoku features for {symbol}: {e}")
        if conn: conn.rollback()
        return pd.DataFrame()
    finally:
        gc.collect()
        def calculate_ichimoku_based_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # حساب المؤشرات النسبية
        df['price_vs_tenkan'] = (df['close'] - df['tenkan_sen']) / df['tenkan_sen']
        df['price_vs_kijun'] = (df['close'] - df['kijun_sen']) / df['kijun_sen']
        df['tenkan_vs_kijun'] = (df['tenkan_sen'] - df['kijun_sen']) / df['kijun_sen']
        df['price_vs_kumo_a'] = (df['close'] - df['senkou_span_a']) / df['senkou_span_a']
        df['price_vs_kumo_b'] = (df['close'] - df['senkou_span_b']) / df['senkou_span_b']
        df['kumo_thickness'] = (df['senkou_span_a'] - df['senkou_span_b']).abs() / df['close']

        # تحديد موقع السعر بالنسبة للسحابة
        kumo_high = df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
        kumo_low = df[['senkou_span_a', 'senkou_span_b']].min(axis=1)
        
        df['price_above_kumo'] = (df['close'] > kumo_high).astype(int)
        df['price_below_kumo'] = (df['close'] < kumo_low).astype(int)
        df['price_in_kumo'] = ((df['close'] >= kumo_low) & (df['close'] <= kumo_high)).astype(int)
        
        # مؤشرات Chikou
        df['chikou_above_kumo'] = (df['chikou_span'] > kumo_high).astype(int)
        df['chikou_below_kumo'] = (df['chikou_span'] < kumo_low).astype(int)
        
        # تقاطعات Tenkan/Kijun
        df['tenkan_kijun_cross'] = 0
        cross_up = (df['tenkan_sen'].shift(1) < df['kijun_sen'].shift(1)) & (df['tenkan_sen'] > df['kijun_sen'])
        cross_down = (df['tenkan_sen'].shift(1) > df['kijun_sen'].shift(1)) & (df['tenkan_sen'] < df['kijun_sen'])
        df.loc[cross_up, 'tenkan_kijun_cross'] = 1
        df.loc[cross_down, 'tenkan_kijun_cross'] = -1
        
        return df
        
    except Exception as e:
        logger.error(f"❌ [Ichimoku] خطأ في حساب مؤشرات Ichimoku: {e}")
        return df
    finally:
        gc.collect()

def calculate_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df_patterns = df.copy()
        op, hi, lo, cl = df_patterns['open'], df_patterns['high'], df_patterns['low'], df_patterns['close']
        
        # حساب خصائص الشموع
        body = abs(cl - op)
        candle_range = hi - lo
        candle_range[candle_range == 0] = 1e-9  # تجنب القسمة على صفر
        
        upper_wick = hi - pd.concat([op, cl], axis=1).max(axis=1)
        lower_wick = pd.concat([op, cl], axis=1).min(axis=1) - lo
        
        df_patterns['candlestick_pattern'] = 0
        
        # تحديد أنماط الشموع
        is_bullish_marubozu = (cl > op) & (body / candle_range > 0.95) & (upper_wick < body * 0.1) & (lower_wick < body * 0.1)
        is_bearish_marubozu = (op > cl) & (body / candle_range > 0.95) & (upper_wick < body * 0.1) & (lower_wick < body * 0.1)
        
        is_bullish_engulfing = (cl.shift(1) < op.shift(1)) & (cl > op) & (cl >= op.shift(1)) & (op <= cl.shift(1)) & (body > body.shift(1))
        is_bearish_engulfing = (cl.shift(1) > op.shift(1)) & (cl < op) & (op >= cl.shift(1)) & (cl <= op.shift(1)) & (body > body.shift(1))
        
        is_hammer = (body > candle_range * 0.1) & (lower_wick >= body * 2) & (upper_wick < body)
        is_shooting_star = (body > candle_range * 0.1) & (upper_wick >= body * 2) & (lower_wick < body)
        
        is_doji = (body / candle_range) < 0.05
        
        # تعيين قيم الأنماط
        df_patterns.loc[is_doji, 'candlestick_pattern'] = 3
        df_patterns.loc[is_hammer, 'candlestick_pattern'] = 2
        df_patterns.loc[is_shooting_star, 'candlestick_pattern'] = -2
        df_patterns.loc[is_bullish_engulfing, 'candlestick_pattern'] = 1
        df_patterns.loc[is_bearish_engulfing, 'candlestick_pattern'] = -1
        df_patterns.loc[is_bullish_marubozu, 'candlestick_pattern'] = 4
        df_patterns.loc[is_bearish_marubozu, 'candlestick_pattern'] = -4
        
        return df_patterns
        
    except Exception as e:
        logger.error(f"❌ [Patterns] خطأ في تحليل أنماط الشموع: {e}")
        return df
    finally:
        gc.collect()

def calculate_sr_features(df: pd.DataFrame, sr_levels_df: pd.DataFrame) -> pd.DataFrame:
    if sr_levels_df.empty:
        df['dist_to_support'] = 0.0
        df['dist_to_resistance'] = 0.0
        df['score_of_support'] = 0.0
        df['score_of_resistance'] = 0.0
        return df
        
    try:
        # تحديد مستويات الدعم والمقاومة
        supports = sr_levels_df[sr_levels_df['level_type'].str.contains('support|poc|confluence', case=False)]['level_price'].sort_values().to_numpy()
        resistances = sr_levels_df[sr_levels_df['level_type'].str.contains('resistance|poc|confluence', case=False)]['level_price'].sort_values().to_numpy()
        
        # تحويل درجات القوة إلى قواميس للوصول السريع
        support_scores = sr_levels_df[sr_levels_df['level_type'].str.contains('support|poc|confluence', case=False)].set_index('level_price')['score'].to_dict()
        resistance_scores = sr_levels_df[sr_levels_df['level_type'].str.contains('resistance|poc|confluence', case=False)].set_index('level_price')['score'].to_dict()

        def get_sr_info(price):
            dist_support, score_support = 1.0, 0.0
            dist_resistance, score_resistance = 1.0, 0.0
            
            # حساب أقرب مستوى دعم
            if supports.size > 0:
                idx = np.searchsorted(supports, price, side='right') - 1
                if idx >= 0:
                    nearest_support_price = supports[idx]
                    dist_support = (price - nearest_support_price) / price if price > 0 else 0
                    score_support = support_scores.get(nearest_support_price, 0)
                    
            # حساب أقرب مستوى مقاومة
            if resistances.size > 0:
                idx = np.searchsorted(resistances, price, side='left')
                if idx < len(resistances):
                    nearest_resistance_price = resistances[idx]
                    dist_resistance = (nearest_resistance_price - price) / price if price > 0 else 0
                    score_resistance = resistance_scores.get(nearest_resistance_price, 0)
                    
            return dist_support, score_support, dist_resistance, score_resistance

        # تطبيق الحسابات على كل سعر
        results = df['close'].apply(get_sr_info)
        df[['dist_to_support', 'score_of_support', 'dist_to_resistance', 'score_of_resistance']] = pd.DataFrame(results.tolist(), index=df.index)
        
        return df
        
    except Exception as e:
        logger.error(f"❌ [S/R Features] خطأ في حساب خصائص الدعم والمقاومة: {e}")
        return df
    finally:
        gc.collect()
        def calculate_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    try:
        df_calc = df.copy()
        
        # حساب ATR
        high_low = df_calc['high'] - df_calc['low']
        high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
        low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
        
        # حساب ADX
        up_move = df_calc['high'].diff()
        down_move = -df_calc['low'].diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
        plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr']
        minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr']
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
        df_calc['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
        
        # حساب RSI
        delta = df_calc['close'].diff()
        gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
        df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
        
        # حساب MACD
        ema_fast = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        df_calc['macd_hist'] = macd_line - signal_line
        
        # تقاطعات MACD
        df_calc['macd_cross'] = 0
        df_calc.loc[(df_calc['macd_hist'].shift(1) < 0) & (df_calc['macd_hist'] >= 0), 'macd_cross'] = 1
        df_calc.loc[(df_calc['macd_hist'].shift(1) > 0) & (df_calc['macd_hist'] <= 0), 'macd_cross'] = -1
        
        # حساب Bollinger Bands
        sma = df_calc['close'].rolling(window=BBANDS_PERIOD).mean()
        std_dev = df_calc['close'].rolling(window=BBANDS_PERIOD).std()
        upper_band = sma + (std_dev * 2)
        lower_band = sma - (std_dev * 2)
        df_calc['bb_width'] = (upper_band - lower_band) / (sma + 1e-9)
        
        # حساب Stochastic RSI
        rsi = df_calc['rsi']
        min_rsi = rsi.rolling(window=STOCH_RSI_PERIOD).min()
        max_rsi = rsi.rolling(window=STOCH_RSI_PERIOD).max()
        stoch_rsi_val = (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-9)
        df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(window=STOCH_K).mean() * 100
        df_calc['stoch_rsi_d'] = df_calc['stoch_rsi_k'].rolling(window=STOCH_D).mean()
        
        # حجم التداول النسبي
        df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)
        
        # حالة السوق
        df_calc['market_condition'] = 0
        df_calc.loc[(df_calc['rsi'] > RSI_OVERBOUGHT) | (df_calc['stoch_rsi_k'] > STOCH_RSI_OVERBOUGHT), 'market_condition'] = 1
        df_calc.loc[(df_calc['rsi'] < RSI_OVERSOLD) | (df_calc['stoch_rsi_k'] < STOCH_RSI_OVERSOLD), 'market_condition'] = -1
        
        # المتوسطات المتحركة الأسية
        ema_fast_trend = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
        ema_slow_trend = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
        df_calc['price_vs_ema50'] = (df_calc['close'] / ema_fast_trend) - 1
        df_calc['price_vs_ema200'] = (df_calc['close'] / ema_slow_trend) - 1
        
        # العوائد والارتباط مع البيتكوين
        df_calc['returns'] = df_calc['close'].pct_change()
        merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
        df_calc['btc_correlation'] = merged_df['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
        
        # ساعة اليوم
        df_calc['hour_of_day'] = df_calc.index.hour
        
        # أنماط الشموع
        df_calc = calculate_candlestick_patterns(df_calc)
        
        return df_calc.astype('float32', errors='ignore')
        
    except Exception as e:
        logger.error(f"❌ [Features] خطأ في حساب المؤشرات الفنية: {e}")
        return df_calc
    finally:
        gc.collect()
        def load_ml_model_bundle_from_folder(symbol: str) -> Optional[Dict[str, Any]]:
    global ml_models_cache
    
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    
    # التحقق من وجود النموذج في الذاكرة المؤقتة
    if model_name in ml_models_cache:
        logger.debug(f"✅ [ML Model] استخدام النموذج '{model_name}' من الذاكرة المؤقتة.")
        return ml_models_cache[model_name]

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, MODEL_FOLDER, f"{model_name}.pkl")
        
        if not os.path.exists(model_path):
            logger.warning(f"⚠️ [ML Model] ملف النموذج '{model_path}' غير موجود للعملة {symbol}.")
            return None

        with open(model_path, 'rb') as f:
            model_bundle = pickle.load(f)
            
        if all(key in model_bundle for key in ['model', 'scaler', 'feature_names']):
            ml_models_cache[model_name] = model_bundle
            logger.info(f"✅ [ML Model] تم تحميل النموذج '{model_name}' بنجاح من الملف المحلي.")
            return model_bundle
        else:
            logger.error(f"❌ [ML Model] حزمة النموذج في '{model_path}' غير مكتملة.")
            return None
            
    except Exception as e:
        logger.error(f"❌ [ML Model] خطأ في تحميل حزمة النموذج من الملف للعملة {symbol}: {e}", exc_info=True)
        return None
    finally:
        gc.collect()

def handle_price_update_message(msg: List[Dict[str, Any]]) -> None:
    global redis_client
    
    try:
        if not isinstance(msg, list):
            logger.warning(f"⚠️ [WebSocket] تم استلام رسالة بتنسيق غير متوقع: {type(msg)}")
            return
            
        if not redis_client:
            logger.error("❌ [WebSocket] كائن Redis غير مهيأ. لا يمكن حفظ الأسعار.")
            return

        # تجميع تحديثات الأسعار في قاموس واحد
        price_updates = {
            item.get('s'): float(item.get('c', 0)) 
            for item in msg 
            if item.get('s') and item.get('c')
        }
        
        if price_updates:
            # تحديث Redis في عملية واحدة
            redis_client.hset(REDIS_PRICES_HASH_NAME, mapping=price_updates)
            
    except Exception as e:
        logger.error(f"❌ [WebSocket Price Updater] خطأ في معالجة رسالة السعر: {e}", exc_info=True)
    finally:
        gc.collect()

def initiate_signal_closure(symbol: str, signal_to_close: Dict, status: str, closing_price: float):
    signal_id = signal_to_close.get('id')
    
    # التحقق من صحة البيانات
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
            Thread(target=close_signal, args=(signal_data, status, closing_price, "auto_monitor")).start()
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
        def trade_monitoring_loop():
    global last_api_check_time
    
    # إضافة مؤقت للتحكم في عدد محاولات إعادة الاتصال
    MAX_RECONNECT_ATTEMPTS = 3
    reconnect_attempts = 0
    
    # إضافة مؤقت للتحكم في عمليات تنظيف الذاكرة
    last_memory_cleanup = time.time()
    
    logger.info("✅ [Trade Monitor] بدء خيط المراقبة")
    
    while True:
        try:
            # تنظيف دوري للذاكرة
            current_time = time.time()
            if current_time - last_memory_cleanup > MEMORY_CLEANUP_INTERVAL:
                logger.info("🧹 [Trade Monitor] تنظيف الذاكرة...")
                gc.collect()
                last_memory_cleanup = current_time

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
                
            reconnect_attempts = 0  # إعادة تعيين العداد عند نجاح الاتصال

            perform_direct_api_check = (time.time() - last_api_check_time) > DIRECT_API_CHECK_INTERVAL
            if perform_direct_api_check:
                logger.debug(f"🔄 [Direct API Check] حان وقت الفحص المباشر من API (كل {DIRECT_API_CHECK_INTERVAL} ثانية).")
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
                    logger.info(f"✅ [TRIGGER] ID:{signal_id} | {symbol} | تحقق شرط '{status_to_set}'. السعر {price} ({price_source}) تجاوز المستوى (TP: {target_price}, SL: {stop_loss_price}).")
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
            
            # تسجيل وظيفة معالجة تحديثات الأسعار
            twm.start_miniticker_socket(callback=handle_price_update_message)
            
            logger.info("✅ [WebSocket] تم الاتصال والاستماع إلى 'All Market Mini Tickers' بنجاح.")
            twm.join()
            break
            
        except Exception as e:
            attempt += 1
            logger.error(f"❌ [WebSocket] فشل الاتصال (المحاولة {attempt}/{MAX_RECONNECT_ATTEMPTS}): {e}")
            
            if attempt < MAX_RECONNECT_ATTEMPTS:
                time.sleep(reconnect_delay)
                reconnect_delay *= 2  # زيادة وقت الانتظار تدريجياً
            else:
                logger.critical("❌ [WebSocket] فشل الاتصال بعد عدة محاولات. إيقاف البوت.")
                os._exit(1)

class TradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_bundle_from_folder(symbol)
        self.ml_model = model_bundle.get('model') if model_bundle else None
        self.scaler = model_bundle.get('scaler') if model_bundle else None
        self.feature_names = model_bundle.get('feature_names') if model_bundle else None

    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame, 
                    sr_levels_df: pd.DataFrame, ichimoku_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            # إنشاء نسخ محلية لتجنب تعديل البيانات الأصلية
            df_featured = df_15m.copy()
            df_4h_local = df_4h.copy()
            
            # حساب المؤشرات
            df_featured = calculate_features(df_featured, btc_df)
            df_featured = calculate_sr_features(df_featured, sr_levels_df)
            
            if not ichimoku_df.empty:
                df_featured = df_featured.join(ichimoku_df, how='left')
                df_featured = calculate_ichimoku_based_features(df_featured)
                
            # إضافة مؤشرات الإطار الزمني الأعلى
            delta_4h = df_4h_local['close'].diff()
            gain_4h = delta_4h.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
            loss_4h = -delta_4h.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
            df_4h_local['rsi_4h'] = 100 - (100 / (1 + (gain_4h / loss_4h.replace(0, 1e-9))))
            
            ema_fast_4h = df_4h_local['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
            df_4h_local['price_vs_ema50_4h'] = (df_4h_local['close'] / ema_fast_4h) - 1
            
            mtf_features = df_4h_local[['rsi_4h', 'price_vs_ema50_4h']]
            df_featured = df_featured.join(mtf_features)
            df_featured[['rsi_4h', 'price_vs_ema50_4h']] = df_featured[['rsi_4h', 'price_vs_ema50_4h']].fillna(method='ffill')
            
            # التأكد من وجود جميع الخصائص المطلوبة
            for col in self.feature_names:
                if col not in df_featured.columns:
                    df_featured[col] = 0.0
            
            # معالجة القيم غير المحدودة
            df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            return df_featured[self.feature_names].dropna()
            
        except Exception as e:
            logger.error(f"❌ [{self.symbol}] فشل هندسة الميزات: {e}", exc_info=True)
            return None
        finally:
            # تنظيف الذاكرة
            del df_15m, df_4h, sr_levels_df, ichimoku_df
            gc.collect()

    def generate_signal(self, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]): 
            return None
            
        if df_features.empty:
            return None
            
        try:
            last_row_df = df_features.iloc[[-1]]
            features_scaled = self.scaler.transform(last_row_df)
            features_scaled_df = pd.DataFrame(features_scaled, columns=self.feature_names)
            
            prediction = self.ml_model.predict(features_scaled_df)[0]
            prediction_proba = self.ml_model.predict_proba(features_scaled_df)[0]
            
            try:
                class_1_index = list(self.ml_model.classes_).index(1)
            except ValueError:
                return None
                
            prob_for_class_1 = prediction_proba[class_1_index]
            
            if prediction == 1 and prob_for_class_1 >= MODEL_CONFIDENCE_THRESHOLD:
                logger.info(f"✅ [العثور على إشارة] {self.symbol}: تنبأ النموذج 'شراء' (1) بثقة {prob_for_class_1:.2%}")
                return {
                    'symbol': self.symbol,
                    'strategy_name': BASE_ML_MODEL_NAME,
                    'signal_details': {
                        'ML_Probability_Buy': f"{prob_for_class_1:.2%}"
                    }
                }
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ [توليد إشارة] {self.symbol}: خطأ أثناء التوليد: {e}", exc_info=True)
            return None
        finally:
            gc.collect()
            def main_loop():
    logger.info("[الحلقة الرئيسية] انتظار اكتمال التهيئة الأولية...")
    time.sleep(15)
    
    # إضافة مؤقت لتتبع آخر مرة تم فيها تنظيف الذاكرة
    last_memory_cleanup = time.time()
    MEMORY_CLEANUP_INTERVAL = 3600  # تنظيف كل ساعة
    
    if not validated_symbols_to_scan:
        log_and_notify("critical", "لا توجد رموز معتمدة للمسح. لن يستمر البوت في العمل.", "SYSTEM")
        return
        
    log_and_notify("info", f"بدء حلقة المسح الرئيسية لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")
    
    all_symbols = list(validated_symbols_to_scan)
    symbol_chunks = [all_symbols[i:i + MODEL_BATCH_SIZE] for i in range(0, len(all_symbols), MODEL_BATCH_SIZE)]
    strategies = {symbol: TradingStrategy(symbol) for symbol in all_symbols}

    while True:
        try:
            start_time = time.time()
            
            # تنظيف دوري للذاكرة
            if start_time - last_memory_cleanup > MEMORY_CLEANUP_INTERVAL:
                logger.info("🧹 [تنظيف الذاكرة] بدء التنظيف الدوري...")
                ml_models_cache.clear()
                gc.collect()
                last_memory_cleanup = start_time
                logger.info("✅ [تنظيف الذاكرة] اكتمل التنظيف الدوري")

            for symbol_batch in symbol_chunks:
                batch_signals = []
                
                for symbol in symbol_batch:
                    try:
                        # جلب البيانات التاريخية
                        df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        if df_15m is None or df_15m.empty:
                            logger.warning(f"⚠️ [{symbol}] لا توجد بيانات 15 دقيقة صالحة.")
                            continue
                            
                        df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        if df_4h is None or df_4h.empty:
                            logger.warning(f"⚠️ [{symbol}] لا توجد بيانات 4 ساعات صالحة.")
                            continue

                        btc_df = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        if btc_df is None or btc_df.empty:
                            logger.warning(f"⚠️ [{symbol}] لا توجد بيانات BTC صالحة.")
                            continue
                            
                        btc_df['btc_returns'] = btc_df['close'].pct_change()

                        # جلب مستويات الدعم والمقاومة
                        sr_levels_df = fetch_sr_levels_from_db(symbol)
                        
                        # جلب مؤشرات Ichimoku
                        ichimoku_df = fetch_ichimoku_features_from_db(symbol, SIGNAL_GENERATION_TIMEFRAME)

                        # استخدام استراتيجية التداول
                        strategy = strategies.get(symbol)
                        if not strategy:
                            logger.warning(f"⚠️ [{symbol}] لا توجد استراتيجية مهيأة.")
                            continue

                        # حساب المؤشرات وتوليد الإشارة
                        df_features = strategy.get_features(df_15m, df_4h, btc_df, sr_levels_df, ichimoku_df)
                        if df_features is None:
                            continue

                        signal = strategy.generate_signal(df_features)
                        if signal:
                            batch_signals.append(signal)

                    except Exception as e:
                        logger.error(f"❌ [{symbol}] خطأ في معالجة العملة: {e}", exc_info=True)
                    finally:
                        # تنظيف الذاكرة بعد كل عملة
                        gc.collect()

                # معالجة إشارات المجموعة
                if batch_signals:
                    logger.info(f"🎯 [Batch] تم العثور على {len(batch_signals)} إشارة في هذه المجموعة.")
                    for signal in batch_signals:
                        try:
                            # هنا يمكنك إضافة منطق معالجة الإشارات
                            pass
                        except Exception as e:
                            logger.error(f"❌ [معالجة الإشارة] خطأ في معالجة الإشارة: {e}", exc_info=True)

            # حساب وقت النوم المطلوب
            execution_time = time.time() - start_time
            sleep_time = max(1, 60 - execution_time)  # على الأقل ثانية واحدة
            logger.debug(f"💤 [الحلقة الرئيسية] اكتمل المسح في {execution_time:.2f} ثانية. النوم لمدة {sleep_time:.2f} ثانية.")
            time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"❌ [الحلقة الرئيسية] خطأ غير متوقع: {e}", exc_info=True)
            time.sleep(60)
        finally:
            gc.collect()
            # --- تهيئة Flask وتعريف المسارات ---
app = Flask(__name__)
CORS(app)

@app.route('/health')
def health_check():
    try:
        checks = {
            'redis': bool(redis_client and redis_client.ping()),
            'database': check_db_connection(),
            'binance': bool(client and client.ping()),
            'symbols_loaded': len(validated_symbols_to_scan) > 0,
            'models_loaded': len(ml_models_cache) > 0,
            'memory_usage': f"{psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB"
        }
        
        status = 'healthy' if all(v for k, v in checks.items() if k != 'memory_usage') else 'degraded'
        return jsonify({
            'status': status,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'checks': checks
        }), 200 if status == 'healthy' else 503
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 500

@app.route('/notifications', methods=['GET'])
def get_notifications():
    try:
        limit = min(int(request.args.get('limit', 50)), 100)
        with notifications_lock:
            recent_notifications = list(notifications_cache)[:limit]
        return jsonify(recent_notifications)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/signals', methods=['GET'])
def get_signals():
    try:
        with signal_cache_lock:
            signals = list(open_signals_cache.values())
        return jsonify(signals)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def metrics():
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        metrics_data = {
            'system': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_usage_mb': memory_info.rss / 1024 / 1024,
                'memory_percent': process.memory_percent(),
                'threads_count': process.num_threads(),
                'open_files': len(process.open_files()),
                'uptime_seconds': time.time() - process.create_time()
            },
            'application': {
                'open_signals_count': len(open_signals_cache),
                'cached_models_count': len(ml_models_cache),
                'monitored_symbols': len(validated_symbols_to_scan),
                'notifications_cached': len(notifications_cache)
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return jsonify(metrics_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def init_services():
    """تهيئة جميع الخدمات المطلوبة"""
    try:
        global client, validated_symbols_to_scan
        
        # تهيئة قاعدة البيانات
        init_db()
        
        # تهيئة Redis
        init_redis()
        
        # تهيئة عميل Binance
        client = Client(API_KEY, API_SECRET)
        
        # تحميل الرموز المعتمدة
        validated_symbols_to_scan = get_validated_symbols()
        
        # استرداد حالة الذاكرة المؤقتة
        recover_cache_state()
        
        return True
        
    except Exception as e:
        logger.critical(f"❌ [تهيئة] فشل في تهيئة الخدمات: {e}", exc_info=True)
        return False

def start_background_tasks():
    """بدء المهام الخلفية"""
    try:
        # بدء خيط مراقبة التداول
        Thread(target=trade_monitoring_loop, daemon=True, name="TradeMonitor").start()
        
        # بدء مدير WebSocket
        Thread(target=run_websocket_manager, daemon=True, name="WebSocketManager").start()
        
        # بدء الحلقة الرئيسية
        Thread(target=main_loop, daemon=True, name="MainLoop").start()
        
        return True
        
    except Exception as e:
        logger.critical(f"❌ [المهام الخلفية] فشل في بدء المهام الخلفية: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    try:
        # تهيئة الخدمات
        if not init_services():
            logger.critical("❌ فشل في تهيئة الخدمات الأساسية. إيقاف البوت.")
            sys.exit(1)
            
        # بدء المهام الخلفية
        if not start_background_tasks():
            logger.critical("❌ فشل في بدء المهام الخلفية. إيقاف البوت.")
            sys.exit(1)
            
        # تشغيل خادم Flask
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        logger.critical(f"❌ [التشغيل الرئيسي] خطأ حرج: {e}", exc_info=True)
        sys.exit(1)