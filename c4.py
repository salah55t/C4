import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle # Added for ML model deserialization
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException, BinanceRequestException
from flask import Flask, request, Response
from threading import Thread
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Tuple, Any, Union

# ---------------------- إعداد التسجيل ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_elliott_fib.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBot')

# ---------------------- تحميل المتغيرات البيئية ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
    # WEBHOOK_URL is optional, but Flask will always run for Render compatibility
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
except Exception as e:
     logger.critical(f"❌ فشل في تحميل المتغيرات البيئية الأساسية: {e}")
     exit(1)

logger.info(f"Binance API Key: {'Available' if API_KEY else 'Not available'}")
logger.info(f"Telegram Token: {TELEGRAM_TOKEN[:10]}...{'*' * (len(TELEGRAM_TOKEN)-10)}")
logger.info(f"Telegram Chat ID: {CHAT_ID}")
logger.info(f"Database URL: {'Available' if DB_URL else 'Not available'}")
logger.info(f"Webhook URL: {WEBHOOK_URL if WEBHOOK_URL else 'Not specified'} (Flask will always run for Render)")

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
TRADE_VALUE: float = 10.0
MAX_OPEN_TRADES: int = 5
SIGNAL_GENERATION_TIMEFRAME: str = '5m'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 3 # Should be enough for indicators, ML model uses more
SIGNAL_TRACKING_TIMEFRAME: str = '5m' # For tracking, not primary signal generation
SIGNAL_TRACKING_LOOKBACK_DAYS: int = 1

# Indicator Parameters (يجب أن تتطابق مع ml.py إذا كانت تؤثر على السمات المستخدمة في النموذج)
RSI_PERIOD: int = 9
RSI_OVERSOLD: int = 30
RSI_OVERBOUGHT: int = 70
EMA_SHORT_PERIOD: int = 8
EMA_LONG_PERIOD: int = 21
VWMA_PERIOD: int = 15
SWING_ORDER: int = 3 # For Elliott waves, not directly ML features
FIB_LEVELS_TO_CHECK: List[float] = [0.382, 0.5, 0.618] # For Elliott, not ML
FIB_TOLERANCE: float = 0.005 # For Elliott, not ML
LOOKBACK_FOR_SWINGS: int = 50 # For Elliott, not ML
ENTRY_ATR_PERIOD: int = 10
ENTRY_ATR_MULTIPLIER: float = 1.5
BOLLINGER_WINDOW: int = 20
BOLLINGER_STD_DEV: int = 2
MACD_FAST: int = 9
MACD_SLOW: int = 18
MACD_SIGNAL: int = 9
ADX_PERIOD: int = 10
SUPERTREND_PERIOD: int = 10
SUPERTREND_MULTIPLIER: float = 2.5

MIN_PROFIT_MARGIN_PCT: float = 1.0
MIN_VOLUME_15M_USDT: float = 250000.0

RECENT_EMA_CROSS_LOOKBACK: int = 2
MIN_ADX_TREND_STRENGTH: int = 20
MACD_HIST_INCREASE_CANDLES: int = 3
OBV_INCREASE_CANDLES: int = 3

TARGET_APPROACH_THRESHOLD_PCT: float = 0.005

BINANCE_FEE_RATE: float = 0.001

ML_MODEL_NAME: str = 'DecisionTree_Scalping_V1' # Must match the name used in train_ml_model.py

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
cur: Optional[psycopg2.extensions.cursor] = None
client: Optional[Client] = None
ticker_data: Dict[str, float] = {}
ml_model: Optional[Any] = None # Global variable to hold the loaded ML model
ml_model_features: List[str] = [] # Global variable to hold the feature names the ML model was trained on

# ---------------------- Binance Client Setup ----------------------
try:
    logger.info("ℹ️ [Binance] تهيئة عميل Binance...")
    client = Client(API_KEY, API_SECRET)
    client.ping()
    server_time = client.get_server_time()
    logger.info(f"✅ [Binance] تم تهيئة عميل Binance. وقت الخادم: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
except BinanceRequestException as req_err:
     logger.critical(f"❌ [Binance] خطأ في طلب Binance (مشكلة في الشبكة أو الطلب): {req_err}")
     exit(1)
except BinanceAPIException as api_err:
     logger.critical(f"❌ [Binance] خطأ في واجهة برمجة تطبيقات Binance (مفاتيح غير صالحة أو مشكلة في الخادم): {api_err}")
     exit(1)
except Exception as e:
    logger.critical(f"❌ [Binance] فشل غير متوقع في تهيئة عميل Binance: {e}")
    exit(1)

# ---------------------- Additional Indicator Functions ----------------------
def get_fear_greed_index() -> str:
    """Fetches the Fear & Greed Index from alternative.me and translates classification to Arabic."""
    classification_translation_ar = {
        "Extreme Fear": "خوف شديد", "Fear": "خوف", "Neutral": "محايد",
        "Greed": "جشع", "Extreme Greed": "جشع شديد",
    }
    url = "https://api.alternative.me/fng/"
    logger.debug(f"ℹ️ [Indicators] جلب مؤشر الخوف والجشع من {url}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        value = int(data["data"][0]["value"])
        classification_en = data["data"][0]["value_classification"]
        classification_ar = classification_translation_ar.get(classification_en, classification_en)
        logger.debug(f"✅ [Indicators] مؤشر الخوف والجشع: {value} ({classification_ar})")
        return f"{value} ({classification_ar})"
    except requests.exceptions.RequestException as e:
         logger.error(f"❌ [Indicators] خطأ في الشبكة أثناء جلب مؤشر الخوف والجشع: {e}")
         return "N/A (خطأ في الشبكة)"
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"❌ [Indicators] خطأ في تنسيق البيانات لمؤشر الخوف والجشع: {e}")
        return "N/A (خطأ في البيانات)"
    except Exception as e:
        logger.error(f"❌ [Indicators] خطأ غير متوقع أثناء جلب مؤشر الخوف والجشع: {e}", exc_info=True)
        return "N/A (خطأ غير معروف)"

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """Fetches historical candlestick data from Binance."""
    if not client:
        logger.error("❌ [Data] عميل Binance غير مهيأ لجلب البيانات.")
        return None
    try:
        # Calculate start_str. Binance API limit is 1000 klines.
        # For '1m' interval, 1000 klines is about 16.6 hours.
        # For '5m' interval, 1000 klines is about 3.47 days.
        # For '15m' interval, 1000 klines is about 10.4 days.
        # Adjust 'days' if it would result in more than 1000 klines for the given interval.
        limit = 1000
        if interval.endswith('m'):
            minutes = int(interval[:-1])
            max_days_for_limit = (limit * minutes) / (24 * 60)
            if days > max_days_for_limit:
                logger.debug(f"ℹ️ [Data] Reducing 'days' from {days} to {max_days_for_limit:.1f} for {symbol} ({interval}) to stay within 1000 kline limit.")
                days_to_fetch = int(max_days_for_limit) + 1 # Fetch slightly more to ensure enough data after drops
            else:
                days_to_fetch = days
        elif interval.endswith('h'):
            hours = int(interval[:-1])
            max_days_for_limit = (limit * hours) / 24
            if days > max_days_for_limit:
                logger.debug(f"ℹ️ [Data] Reducing 'days' from {days} to {max_days_for_limit:.1f} for {symbol} ({interval}) to stay within 1000 kline limit.")
                days_to_fetch = int(max_days_for_limit) + 1
            else:
                days_to_fetch = days
        else: # d, w, M
            days_to_fetch = days # Assume daily or longer intervals won't hit 1000 klines easily with typical 'days' values

        start_dt = datetime.utcnow() - timedelta(days=days_to_fetch + 1) # Add 1 day buffer
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")

        logger.debug(f"ℹ️ [Data] جلب بيانات {interval} لـ {symbol} منذ {start_str} (حد {limit} شمعة)...")
        klines = client.get_historical_klines(symbol, interval, start_str, limit=limit)

        if not klines:
            logger.warning(f"⚠️ [Data] لا توجد بيانات تاريخية ({interval}) لـ {symbol} للفترة المطلوبة.")
            return None

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[numeric_cols] # Select only relevant columns
        initial_len = len(df)
        df.dropna(subset=numeric_cols, inplace=True) # Drop rows with NaN in essential OHLCV

        if len(df) < initial_len:
            logger.debug(f"ℹ️ [Data] {symbol}: تم إسقاط {initial_len - len(df)} صفًا بسبب قيم NaN في بيانات OHLCV.")

        if df.empty:
            logger.warning(f"⚠️ [Data] DataFrame لـ {symbol} فارغ بعد إزالة قيم NaN الأساسية.")
            return None

        logger.debug(f"✅ [Data] تم جلب ومعالجة {len(df)} شمعة تاريخية ({interval}) لـ {symbol}.")
        return df

    except BinanceAPIException as api_err:
         logger.error(f"❌ [Data] خطأ في Binance API أثناء جلب البيانات لـ {symbol}: {api_err}")
         return None
    except BinanceRequestException as req_err:
         logger.error(f"❌ [Data] خطأ في الطلب أو الشبكة أثناء جلب البيانات لـ {symbol}: {req_err}")
         return None
    except Exception as e:
        logger.error(f"❌ [Data] خطأ غير متوقع أثناء جلب البيانات التاريخية لـ {symbol}: {e}", exc_info=True)
        return None


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculates Exponential Moving Average (EMA)."""
    if series is None or series.isnull().all() or len(series) < span:
        return pd.Series(index=series.index if series is not None else None, dtype=float)
    return series.ewm(span=span, adjust=False).mean()

def calculate_vwma(df: pd.DataFrame, period: int) -> pd.Series:
    """Calculates Volume Weighted Moving Average (VWMA)."""
    df_calc = df.copy()
    required_cols = ['close', 'volume']
    if not all(col in df_calc.columns for col in required_cols) or df_calc[required_cols].isnull().all().any():
        logger.warning(f"⚠️ [Indicator VWMA] {df_calc.index.name if df_calc.index.name else ''}: أعمدة 'close' أو 'volume' مفقودة أو فارغة.")
        return pd.Series(index=df_calc.index if df_calc is not None else None, dtype=float)
    if len(df_calc) < period:
        logger.warning(f"⚠️ [Indicator VWMA] {df_calc.index.name if df_calc.index.name else ''}: بيانات غير كافية ({len(df_calc)} < {period}) لحساب VWMA.")
        return pd.Series(index=df_calc.index if df_calc is not None else None, dtype=float)

    df_calc['price_volume'] = df_calc['close'] * df_calc['volume']
    rolling_price_volume_sum = df_calc['price_volume'].rolling(window=period, min_periods=period).sum()
    rolling_volume_sum = df_calc['volume'].rolling(window=period, min_periods=period).sum()
    vwma = rolling_price_volume_sum / rolling_volume_sum.replace(0, np.nan) # Avoid division by zero
    df_calc.drop(columns=['price_volume'], inplace=True, errors='ignore')
    return vwma

def get_btc_trend_4h() -> str:
    """Calculates Bitcoin trend on 4-hour timeframe using EMA20 and EMA50."""
    logger.debug("ℹ️ [Indicators] حساب اتجاه البيتكوين على 4 ساعات...")
    try:
        df = fetch_historical_data("BTCUSDT", interval=Client.KLINE_INTERVAL_4HOUR, days=20) # Increased days for more robust EMA
        if df is None or df.empty or len(df) < 50 + 1: # Need at least 50 periods for EMA50
            logger.warning("⚠️ [Indicators] بيانات BTC/USDT 4H غير كافية لحساب الاتجاه.")
            return "N/A (بيانات غير كافية)"

        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df.dropna(subset=['close'], inplace=True)
        if len(df) < 50: # Re-check after dropna
             logger.warning("⚠️ [Indicators] بيانات BTC/USDT 4H غير كافية بعد إزالة قيم NaN.")
             return "N/A (بيانات غير كافية)"

        ema20 = calculate_ema(df['close'], 20).iloc[-1]
        ema50 = calculate_ema(df['close'], 50).iloc[-1]
        current_close = df['close'].iloc[-1]

        if pd.isna(ema20) or pd.isna(ema50) or pd.isna(current_close):
            logger.warning("⚠️ [Indicators] قيم EMA أو السعر الحالي للبيتكوين هي NaN.")
            return "N/A (خطأ في الحساب)"

        diff_ema20_pct = abs(current_close - ema20) / current_close if current_close > 0 else 0

        if current_close > ema20 > ema50:
            trend = "صعود 📈"
        elif current_close < ema20 < ema50:
            trend = "هبوط 📉"
        elif diff_ema20_pct < 0.005: # Close to EMA20, consider stable/sideways
            trend = "استقرار 🔄"
        else:
            trend = "تذبذب 🔀" # Other conditions

        logger.debug(f"✅ [Indicators] اتجاه البيتكوين 4H: {trend} (Close: {current_close:.2f}, EMA20: {ema20:.2f}, EMA50: {ema50:.2f})")
        return trend
    except Exception as e:
        logger.error(f"❌ [Indicators] خطأ في حساب اتجاه البيتكوين على 4 ساعات: {e}", exc_info=True)
        return "N/A (خطأ)"

# ---------------------- Database Connection Setup ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    """Initializes database connection and creates tables if they don't exist."""
    global conn, cur
    logger.info("[DB] بدء تهيئة قاعدة البيانات...")
    for attempt in range(retries):
        try:
            logger.info(f"[DB] محاولة الاتصال بقاعدة البيانات (المحاولة {attempt + 1}/{retries})...")
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False # Use False for explicit commit/rollback
            cur = conn.cursor()
            logger.info("✅ [DB] تم الاتصال بقاعدة البيانات بنجاح.")

            # --- Create or update signals table (Modified schema) ---
            logger.info("[DB] التحقق من/إنشاء جدول 'signals'...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    initial_target DOUBLE PRECISION NOT NULL,
                    current_target DOUBLE PRECISION NOT NULL,
                    r2_score DOUBLE PRECISION, -- For general strategy score, not necessarily R-squared
                    volume_15m DOUBLE PRECISION,
                    achieved_target BOOLEAN DEFAULT FALSE,
                    closing_price DOUBLE PRECISION,
                    closed_at TIMESTAMP,
                    sent_at TIMESTAMP DEFAULT NOW(),
                    entry_time TIMESTAMP DEFAULT NOW(),
                    time_to_target INTERVAL,
                    profit_percentage DOUBLE PRECISION,
                    strategy_name TEXT,
                    signal_details JSONB -- To store ML prediction, other conditions
                );""")
            conn.commit()
            logger.info("✅ [DB] جدول 'signals' موجود أو تم إنشاؤه.")

            # --- Create ml_models table (NEW) ---
            logger.info("[DB] التحقق من/إنشاء جدول 'ml_models'...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ml_models (
                    id SERIAL PRIMARY KEY,
                    model_name TEXT NOT NULL UNIQUE,
                    model_data BYTEA NOT NULL,
                    trained_at TIMESTAMP DEFAULT NOW(),
                    metrics JSONB -- To store accuracy, features used, etc.
                );""")
            conn.commit()
            logger.info("✅ [DB] جدول 'ml_models' موجود أو تم إنشاؤه.")

            # --- Create market_dominance table (if it doesn't exist) ---
            logger.info("[DB] التحقق من/إنشاء جدول 'market_dominance'...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS market_dominance (
                    id SERIAL PRIMARY KEY,
                    recorded_at TIMESTAMP DEFAULT NOW(),
                    btc_dominance DOUBLE PRECISION,
                    eth_dominance DOUBLE PRECISION
                );
            """)
            conn.commit()
            logger.info("✅ [DB] جدول 'market_dominance' موجود أو تم إنشاؤه.")


            logger.info("✅ [DB] تم تهيئة قاعدة البيانات بنجاح.")
            return

        except OperationalError as op_err:
            logger.error(f"❌ [DB] خطأ تشغيلي في الاتصال (المحاولة {attempt + 1}): {op_err}")
            if conn: conn.rollback() # Rollback on error if connection was partially made
            if attempt == retries - 1:
                 logger.critical("❌ [DB] فشلت جميع محاولات الاتصال بقاعدة البيانات.")
                 raise op_err # Re-raise the last error
            time.sleep(delay)
        except Exception as e:
            logger.critical(f"❌ [DB] فشل غير متوقع في تهيئة قاعدة البيانات (المحاولة {attempt + 1}): {e}", exc_info=True)
            if conn: conn.rollback()
            if attempt == retries - 1:
                 logger.critical("❌ [DB] فشلت جميع محاولات الاتصال بقاعدة البيانات.")
                 raise e # Re-raise the last error
            time.sleep(delay)

    # If loop finishes without returning, it means connection failed
    logger.critical("❌ [DB] فشل الاتصال بقاعدة البيانات بعد عدة محاولات.")
    exit(1) # Exit if DB connection is critical and fails


def check_db_connection() -> bool:
    """Checks database connection status and re-initializes if necessary."""
    global conn, cur
    try:
        if conn is None or conn.closed != 0: # conn.closed is 0 if open, non-zero if closed
            logger.warning("⚠️ [DB] الاتصال مغلق أو غير موجود. إعادة التهيئة...")
            init_db() # This will raise an exception if it fails, handled by caller
            return True # If init_db succeeds
        else:
             # Perform a simple query to check if the connection is truly alive
             with conn.cursor() as check_cur: # Use a new cursor for the check
                  check_cur.execute("SELECT 1;")
                  check_cur.fetchone()
             return True
    except (OperationalError, InterfaceError) as e: # Specific errors indicating connection loss
        logger.error(f"❌ [DB] فقدان الاتصال بقاعدة البيانات ({e}). إعادة التهيئة...")
        try:
             init_db()
             return True
        except Exception as recon_err: # Catch errors from init_db()
            logger.error(f"❌ [DB] فشلت محاولة إعادة الاتصال بعد فقدان الاتصال: {recon_err}")
            return False
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"❌ [DB] خطأ غير متوقع أثناء التحقق من الاتصال: {e}", exc_info=True)
        try:
            init_db() # Try to re-initialize even on unexpected errors
            return True
        except Exception as recon_err:
             logger.error(f"❌ [DB] فشلت محاولة إعادة الاتصال بعد خطأ غير متوقع: {recon_err}")
             return False

def load_ml_model_from_db() -> Optional[Any]:
    """Loads the latest trained ML model from the database."""
    global ml_model, ml_model_features # To update the global variables
    if not check_db_connection() or not conn: # Ensure conn is not None
        logger.error("❌ [ML Model] لا يمكن تحميل نموذج ML بسبب مشكلة في اتصال قاعدة البيانات.")
        return None

    try:
        with conn.cursor() as db_cur: # Use a new cursor for this operation
            db_cur.execute("SELECT model_data, metrics FROM ml_models WHERE model_name = %s ORDER BY trained_at DESC LIMIT 1;", (ML_MODEL_NAME,))
            result = db_cur.fetchone()
            if result and result['model_data']:
                loaded_model = pickle.loads(result['model_data'])
                metrics = result.get('metrics', {})
                features_in_model = metrics.get('features', []) # Get features from stored metrics
                
                # CRITICAL: Assign loaded features to global variable
                ml_model_features.clear() # Clear existing list
                ml_model_features.extend(features_in_model) # Add new features
                
                logger.info(f"✅ [ML Model] تم تحميل نموذج ML '{ML_MODEL_NAME}' من قاعدة البيانات بنجاح.")
                logger.info(f"ℹ️ [ML Model] النموذج تم تدريبه باستخدام السمات: {ml_model_features}")
                logger.info(f"ℹ️ [ML Model] مقاييس النموذج المحمل: {json.dumps(metrics, indent=2, ensure_ascii=False)}")
                ml_model = loaded_model # Assign to global ml_model
                return ml_model
            else:
                logger.warning(f"⚠️ [ML Model] لم يتم العثور على نموذج ML باسم '{ML_MODEL_NAME}' في قاعدة البيانات. يرجى تدريب النموذج أولاً.")
                ml_model = None # Ensure global ml_model is None if not found
                ml_model_features.clear() # Clear features if model not found
                return None
    except psycopg2.Error as db_err:
        logger.error(f"❌ [ML Model] خطأ في قاعدة البيانات أثناء تحميل نموذج ML: {db_err}", exc_info=True)
        if conn: conn.rollback() # Rollback any transaction
        ml_model = None
        ml_model_features.clear()
        return None
    except pickle.UnpicklingError as unpickle_err:
        logger.error(f"❌ [ML Model] خطأ في فك تسلسل نموذج ML: {unpickle_err}. قد يكون النموذج تالفًا أو تم حفظه بإصدار مختلف.", exc_info=True)
        ml_model = None
        ml_model_features.clear()
        return None
    except Exception as e:
        logger.error(f"❌ [ML Model] خطأ غير متوقع أثناء تحميل نموذج ML: {e}", exc_info=True)
        ml_model = None
        ml_model_features.clear()
        return None


def convert_np_values(obj: Any) -> Any:
    """Converts NumPy data types to native Python types for JSON and DB compatibility."""
    if isinstance(obj, dict):
        return {k: convert_np_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_values(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Use np.integer for all integer types and np.floating for all float types
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj): # Check for pandas NaN
        return None
    else:
        return obj

# ---------------------- Reading and Validating Symbols List ----------------------
def get_crypto_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """
    Reads the list of currency symbols from a text file, then validates them
    as valid USDT pairs available for Spot trading on Binance.
    """
    raw_symbols: List[str] = []
    logger.info(f"ℹ️ [Data] قراءة قائمة الرموز من الملف '{filename}'...")
    try:
        # Try to find the file in the script's directory first
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)

        if not os.path.exists(file_path):
            # If not found, try the current working directory (fallback)
            file_path = os.path.abspath(filename)
            if not os.path.exists(file_path):
                 logger.error(f"❌ [Data] الملف '{filename}' غير موجود في دليل السكربت أو الدليل الحالي.")
                 return []
            else:
                 logger.warning(f"⚠️ [Data] الملف '{filename}' غير موجود في دليل السكربت. استخدام الملف في الدليل الحالي: '{file_path}'")

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = [f"{line.strip().upper().replace('USDT', '')}USDT" # Ensure it's a USDT pair
                           for line in f if line.strip() and not line.startswith('#')] # Ignore empty lines and comments
        raw_symbols = sorted(list(set(raw_symbols))) # Remove duplicates and sort
        logger.info(f"ℹ️ [Data] تم قراءة {len(raw_symbols)} رمزًا مبدئيًا من '{file_path}'.")

    except FileNotFoundError: # Should be caught by os.path.exists, but as a safeguard
         logger.error(f"❌ [Data] الملف '{filename}' غير موجود.")
         return []
    except Exception as e:
        logger.error(f"❌ [Data] خطأ في قراءة الملف '{filename}': {e}", exc_info=True)
        return []

    if not raw_symbols:
         logger.warning("⚠️ [Data] قائمة الرموز الأولية فارغة.")
         return []

    if not client:
        logger.error("❌ [Data Validation] عميل Binance غير مهيأ. لا يمكن التحقق من الرموز.")
        return raw_symbols # Return unvalidated list if client is not available

    try:
        logger.info("ℹ️ [Data Validation] التحقق من الرموز وحالة التداول من Binance API...")
        exchange_info = client.get_exchange_info()
        valid_trading_usdt_symbols = {
            s['symbol'] for s in exchange_info['symbols']
            if s.get('quoteAsset') == 'USDT' and # Ensure it's a USDT pair
               s.get('status') == 'TRADING' and   # Ensure it's currently trading
               s.get('isSpotTradingAllowed') is True # Ensure spot trading is allowed
        }
        logger.info(f"ℹ️ [Data Validation] تم العثور على {len(valid_trading_usdt_symbols)} زوج تداول USDT صالح في Spot على Binance.")
        validated_symbols = [symbol for symbol in raw_symbols if symbol in valid_trading_usdt_symbols]

        removed_count = len(raw_symbols) - len(validated_symbols)
        if removed_count > 0:
            removed_symbols = set(raw_symbols) - set(validated_symbols)
            logger.warning(f"⚠️ [Data Validation] تم إزالة {removed_count} رمز تداول USDT غير صالح أو غير متاح من القائمة: {', '.join(removed_symbols)}")

        logger.info(f"✅ [Data Validation] تم التحقق من الرموز. استخدام {len(validated_symbols)} رمزًا صالحًا.")
        return validated_symbols

    except (BinanceAPIException, BinanceRequestException) as binance_err:
         logger.error(f"❌ [Data Validation] خطأ في Binance API أو الشبكة أثناء التحقق من الرموز: {binance_err}")
         logger.warning("⚠️ [Data Validation] استخدام القائمة الأولية من الملف بدون التحقق من Binance.")
         return raw_symbols # Return raw symbols as a fallback
    except Exception as api_err: # Catch any other unexpected error during validation
         logger.error(f"❌ [Data Validation] خطأ غير متوقع أثناء التحقق من رموز Binance: {api_err}", exc_info=True)
         logger.warning("⚠️ [Data Validation] استخدام القائمة الأولية من الملف بدون التحقق من Binance.")
         return raw_symbols


# ---------------------- WebSocket Management for Ticker Prices ----------------------
def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
    """Handles incoming WebSocket messages for mini-ticker prices."""
    global ticker_data
    try:
        if isinstance(msg, list): # For !miniTicker@arr
            for ticker_item in msg:
                symbol = ticker_item.get('s')
                price_str = ticker_item.get('c') # Last price
                if symbol and 'USDT' in symbol and price_str: # Ensure it's a USDT pair and price exists
                    try:
                        ticker_data[symbol] = float(price_str)
                        logger.debug(f"✅ [WS] تم تحديث سعر {symbol}: {price_str}")
                    except ValueError:
                         logger.warning(f"⚠️ [WS] قيمة سعر غير صالحة للرمز {symbol}: '{price_str}'")
                else:
                    logger.debug(f"ℹ️ [WS] رسالة تيكر مصفوفة غير مكتملة أو غير ذات صلة: {ticker_item}")
        elif isinstance(msg, dict): # For individual streams or error messages
             if msg.get('e') == 'error': # Check for error message from WebSocket
                 logger.error(f"❌ [WS] رسالة خطأ من WebSocket: {msg.get('m', 'لا توجد تفاصيل خطأ')}")
             # Handling for combined streams if used (e.g., <streamName>/<streamName>)
             elif msg.get('stream') and msg.get('data'): # Structure for combined streams
                 data_content = msg['data']
                 if data_content.get('e') == '24hrMiniTicker': # Check event type for mini ticker
                    symbol = data_content.get('s')
                    price_str = data_content.get('c')
                    if symbol and 'USDT' in symbol and price_str:
                        try:
                            ticker_data[symbol] = float(price_str)
                            logger.debug(f"✅ [WS] تم تحديث سعر {symbol} في البث المجمع: {price_str}")
                        except ValueError:
                             logger.warning(f"⚠️ [WS] قيمة سعر غير صالحة للرمز {symbol} في البث المجمع: '{price_str}'")
                    else:
                        logger.debug(f"ℹ️ [WS] رسالة تيكر مجمعة غير مكتملة أو غير ذات صلة: {data_content}")
             else:
                 logger.debug(f"ℹ️ [WS] رسالة WebSocket بتنسيق غير متوقع (قاموس): {msg}")
        else:
             logger.warning(f"⚠️ [WS] تم استلام رسالة WebSocket بتنسيق غير متوقع: {type(msg)} - المحتوى: {str(msg)[:100]}...")


    except Exception as e:
        logger.error(f"❌ [WS] خطأ في معالجة رسالة التيكر: {e}", exc_info=True)


def run_ticker_socket_manager() -> None:
    """Runs and manages the WebSocket connection for mini-ticker."""
    while True:
        twm = None  # Initialize twm to None at the start of each loop iteration
        try:
            logger.info("ℹ️ [WS] بدء إدارة WebSocket لأسعار التيكر...")
            twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
            twm.start()  # This sets twm._running to True internally

            stream_name = twm.start_miniticker_socket(callback=handle_ticker_message)
            logger.info(f"✅ [WS] تم بدء بث WebSocket: {stream_name}")

            # Wait for ticker_data to be populated
            wait_attempts = 0
            max_wait_attempts = 30 # 30 * 1 second = 30 seconds
            while not ticker_data and wait_attempts < max_wait_attempts:
                logger.info(f"ℹ️ [WS] انتظار بيانات التيكر من WebSocket... ({len(ticker_data)} رموز حتى الآن)")
                time.sleep(1)
                wait_attempts += 1
            
            if not ticker_data:
                logger.warning("⚠️ [WS] لم يتم استلام أي بيانات تيكر بعد الانتظار. قد تكون هناك مشكلة في اتصال WebSocket أو لا توجد رموز نشطة.")
            else:
                logger.info(f"✅ [WS] تم استلام بيانات تيكر لـ {len(ticker_data)} رمزًا. متابعة.")


            twm.join()  # Wait for the socket manager to finish (blocks until stop or error)
            # If join returns, it means the TWM stopped.
            logger.warning("⚠️ [WS] توقفت إدارة WebSocket (join completed). إعادة التشغيل...")

        except Exception as e:  # Catches exceptions from twm.start(), start_miniticker_socket(), or twm.join()
            logger.error(f"❌ [WS] خطأ فادح في إدارة WebSocket: {e}. إعادة التشغيل في 15 ثانية...", exc_info=True)
        finally:
            if twm:  # Check if twm object was created
                # Check the internal _running flag before attempting to stop
                # Use getattr for safer access to a private attribute
                is_running = getattr(twm, '_running', False)
                logger.info(f"ℹ️ [WS] Attempting to stop TWM in finally block. Assumed TWM running state: {is_running}")
                try:
                    twm.stop()  # stop() should be idempotent
                    logger.info("✅ [WS] TWM stop called successfully in finally block.")
                except Exception as stop_err:
                    logger.error(f"❌ [WS] خطأ أثناء إيقاف TWM في finally block: {stop_err}", exc_info=True)
        
        time.sleep(15)  # Wait before retrying the loop

# ---------------------- Technical Indicator Functions ----------------------

def calculate_rsi_indicator(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """Calculates Relative Strength Index (RSI)."""
    df = df.copy()
    if 'close' not in df.columns or df['close'].isnull().all():
        logger.warning(f"⚠️ [Indicator RSI] {df.index.name if df.index.name else ''}: عمود 'close' مفقود أو فارغ.")
        df['rsi'] = np.nan
        return df
    if len(df) < period +1: # RSI needs at least 'period' differences, so period+1 data points
        logger.warning(f"⚠️ [Indicator RSI] {df.index.name if df.index.name else ''}: بيانات غير كافية ({len(df)} < {period+1}) لحساب RSI.")
        df['rsi'] = np.nan
        return df

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, adjust=False).mean() # Wilder's smoothing
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean() # Wilder's smoothing

    rs = avg_gain / avg_loss.replace(0, np.nan) # Avoid division by zero, replace with NaN

    rsi_series = 100 - (100 / (1 + rs))
    df['rsi'] = rsi_series.ffill().fillna(50) # Fill initial NaNs with 50 (neutral) or ffill

    return df

def calculate_atr_indicator(df: pd.DataFrame, period: int = ENTRY_ATR_PERIOD) -> pd.DataFrame:
    """Calculates Average True Range (ATR)."""
    df = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().all().any():
        logger.warning(f"⚠️ [Indicator ATR] {df.index.name if df.index.name else ''}: أعمدة 'high', 'low', 'close' مفقودة أو فارغة.")
        df['atr'] = np.nan
        return df
    if len(df) < period + 1: # ATR needs a shift, so period+1
        logger.warning(f"⚠️ [Indicator ATR] {df.index.name if df.index.name else ''}: بيانات غير كافية ({len(df)} < {period + 1}) لحساب ATR.")
        df['atr'] = np.nan
        return df

    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()

    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)
    # Use ewm for smoothed ATR (Wilder's smoothing)
    df['atr'] = tr.ewm(alpha=1/period, adjust=False).mean()
    return df


def calculate_bollinger_bands(df: pd.DataFrame, window: int = BOLLINGER_WINDOW, num_std: int = BOLLINGER_STD_DEV) -> pd.DataFrame:
    """Calculates Bollinger Bands."""
    df = df.copy()
    if 'close' not in df.columns or df['close'].isnull().all():
        logger.warning(f"⚠️ [Indicator BB] {df.index.name if df.index.name else ''}: عمود 'close' مفقود أو فارغ.")
        df['bb_middle'] = np.nan
        df['bb_upper'] = np.nan
        df['bb_lower'] = np.nan
        return df
    if len(df) < window:
         logger.warning(f"⚠️ [Indicator BB] {df.index.name if df.index.name else ''}: بيانات غير كافية ({len(df)} < {window}) لحساب BB.")
         df['bb_middle'] = np.nan
         df['bb_upper'] = np.nan
         df['bb_lower'] = np.nan
         return df

    df['bb_middle'] = df['close'].rolling(window=window, min_periods=window).mean()
    df['bb_std'] = df['close'].rolling(window=window, min_periods=window).std(ddof=0) # ddof=0 for population std dev if desired
    df['bb_upper'] = df['bb_middle'] + num_std * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - num_std * df['bb_std']
    return df


def calculate_macd(df: pd.DataFrame, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> pd.DataFrame:
    """Calculates MACD, Signal Line, and Histogram."""
    df = df.copy()
    if 'close' not in df.columns or df['close'].isnull().all():
        logger.warning(f"⚠️ [Indicator MACD] {df.index.name if df.index.name else ''}: عمود 'close' مفقود أو فارغ.")
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['macd_hist'] = np.nan
        return df
    min_len = max(fast, slow, signal) # Effective minimum length for all components
    if len(df) < min_len: # Check against the longest period needed for any component
        logger.warning(f"⚠️ [Indicator MACD] {df.index.name if df.index.name else ''}: بيانات غير كافية ({len(df)} < {min_len}) لحساب MACD.")
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['macd_hist'] = np.nan
        return df

    ema_fast = calculate_ema(df['close'], fast)
    ema_slow = calculate_ema(df['close'], slow)
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = calculate_ema(df['macd'], signal) # Signal is EMA of MACD line
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df


def calculate_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    """Calculates ADX, DI+ and DI-."""
    df_calc = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df_calc.columns for col in required_cols) or df_calc[required_cols].isnull().all().any():
        logger.warning(f"⚠️ [Indicator ADX] {df_calc.index.name if df_calc.index.name else ''}: أعمدة 'high', 'low', 'close' مفقودة أو فارغة.")
        df_calc['adx'] = np.nan
        df_calc['di_plus'] = np.nan
        df_calc['di_minus'] = np.nan
        return df_calc
    # ADX calculation involves smoothing of TR, +DM, -DM, then DX.
    # A common rule of thumb is needing at least 2 * period for ADX to stabilize.
    if len(df_calc) < period * 2:
        logger.warning(f"⚠️ [Indicator ADX] {df_calc.index.name if df_calc.index.name else ''}: بيانات غير كافية ({len(df_calc)} < {period * 2}) لحساب ADX.")
        df_calc['adx'] = np.nan
        df_calc['di_plus'] = np.nan
        df_calc['di_minus'] = np.nan
        return df_calc

    # True Range (TR)
    df_calc['h-l'] = df_calc['high'] - df_calc['low']
    df_calc['h-pc'] = abs(df_calc['high'] - df_calc['close'].shift(1))
    df_calc['l-pc'] = abs(df_calc['low'] - df_calc['close'].shift(1))
    df_calc['tr'] = df_calc[['h-l', 'h-pc', 'l-pc']].max(axis=1, skipna=False)
    df_calc.drop(['h-l', 'h-pc', 'l-pc'], axis=1, inplace=True)

    # Directional Movement (+DM, -DM)
    df_calc['dm_plus'] = np.where((df_calc['high'] - df_calc['high'].shift(1)) > (df_calc['low'].shift(1) - df_calc['low']),
                                 np.maximum(0, df_calc['high'] - df_calc['high'].shift(1)), 0)
    df_calc['dm_minus'] = np.where((df_calc['low'].shift(1) - df_calc['low']) > (df_calc['high'] - df_calc['high'].shift(1)),
                                  np.maximum(0, df_calc['low'].shift(1) - df_calc['low']), 0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing, equivalent to EMA with alpha=1/period)
    alpha = 1.0 / period
    df_calc['tr_smooth'] = df_calc['tr'].ewm(alpha=alpha, adjust=False).mean()
    df_calc['dm_plus_smooth'] = df_calc['dm_plus'].ewm(alpha=alpha, adjust=False).mean()
    df_calc['dm_minus_smooth'] = df_calc['dm_minus'].ewm(alpha=alpha, adjust=False).mean()

    # Directional Indicators (DI+, DI-)
    df_calc['di_plus'] = np.where(df_calc['tr_smooth'] > 0, 100 * (df_calc['dm_plus_smooth'] / df_calc['tr_smooth']), 0)
    df_calc['di_minus'] = np.where(df_calc['tr_smooth'] > 0, 100 * (df_calc['dm_minus_smooth'] / df_calc['tr_smooth']), 0)

    # Directional Movement Index (DX)
    di_sum = df_calc['di_plus'] + df_calc['di_minus']
    df_calc['dx'] = np.where(di_sum > 0, 100 * abs(df_calc['di_plus'] - df_calc['di_minus']) / di_sum, 0)

    # Average Directional Index (ADX)
    df_calc['adx'] = df_calc['dx'].ewm(alpha=alpha, adjust=False).mean()
    
    # Clean up intermediate columns
    df_calc.drop(['tr', 'dm_plus', 'dm_minus', 'tr_smooth', 'dm_plus_smooth', 'dm_minus_smooth', 'dx'], axis=1, inplace=True, errors='ignore')

    return df_calc[['adx', 'di_plus', 'di_minus']]


def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Volume Weighted Average Price (VWAP) - Resets daily."""
    df = df.copy()
    required_cols = ['high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().all().any():
        logger.warning(f"⚠️ [Indicator VWAP] {df.index.name if df.index.name else ''}: أعمدة 'high', 'low', 'close' أو 'volume' مفقودة أو فارغة.")
        df['vwap'] = np.nan
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
            logger.warning(f"⚠️ [Indicator VWAP] {df.index.name if df.index.name else ''}: تم تحويل الفهرس إلى DatetimeIndex.")
        except Exception:
            logger.error(f"❌ [Indicator VWAP] {df.index.name if df.index.name else ''}: فشل تحويل الفهرس إلى DatetimeIndex، لا يمكن حساب VWAP اليومي.")
            df['vwap'] = np.nan
            return df
    
    # Ensure timezone-aware UTC index for correct daily grouping
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
        logger.debug(f"ℹ️ [Indicator VWAP] {df.index.name if df.index.name else ''}: تم توطين الفهرس إلى UTC لإعادة الضبط اليومي.")
    elif df.index.tz.zone != 'UTC': # Convert if already tz-aware but not UTC
        df.index = df.index.tz_convert('UTC')
        logger.debug(f"ℹ️ [Indicator VWAP] {df.index.name if df.index.name else ''}: تم تحويل الفهرس إلى UTC لإعادة الضبط اليومي.")


    df['date'] = df.index.date # Group by date for daily reset
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol'] = df['typical_price'] * df['volume']

    try:
        # Calculate cumulative sums resetting at the start of each day
        df['cum_tp_vol'] = df.groupby('date')['tp_vol'].cumsum()
        df['cum_volume'] = df.groupby('date')['volume'].cumsum()
    except KeyError as e: # Should not happen if 'date' column is created correctly
        logger.error(f"❌ [Indicator VWAP] {df.index.name if df.index.name else ''}: خطأ في تجميع البيانات حسب التاريخ: {e}. قد يكون الفهرس غير صحيح.")
        df['vwap'] = np.nan
        df.drop(columns=['date', 'typical_price', 'tp_vol', 'cum_tp_vol', 'cum_volume'], inplace=True, errors='ignore')
        return df
    except Exception as e: # Catch any other grouping errors
         logger.error(f"❌ [Indicator VWAP] {df.index.name if df.index.name else ''}: خطأ غير متوقع في حساب VWAP: {e}", exc_info=True)
         df['vwap'] = np.nan
         df.drop(columns=['date', 'typical_price', 'tp_vol', 'cum_tp_vol', 'cum_volume'], inplace=True, errors='ignore')
         return df


    df['vwap'] = np.where(df['cum_volume'] > 0, df['cum_tp_vol'] / df['cum_volume'], np.nan) # Avoid division by zero

    # Forward fill VWAP within each day to handle initial NaNs if cum_volume is 0 at start of day
    # This might not be strictly necessary if data starts with non-zero volume
    df['vwap'] = df.groupby('date')['vwap'].bfill().ffill()


    df.drop(columns=['date', 'typical_price', 'tp_vol', 'cum_tp_vol', 'cum_volume'], inplace=True, errors='ignore')
    return df


def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates On-Balance Volume (OBV)."""
    df = df.copy()
    required_cols = ['close', 'volume']
    if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().all().any():
        logger.warning(f"⚠️ [Indicator OBV] {df.index.name if df.index.name else ''}: أعمدة 'close' أو 'volume' مفقودة أو فارغة.")
        df['obv'] = np.nan
        return df
    if not pd.api.types.is_numeric_dtype(df['close']) or not pd.api.types.is_numeric_dtype(df['volume']):
        logger.warning(f"⚠️ [Indicator OBV] {df.index.name if df.index.name else ''}: أعمدة 'close' أو 'volume' ليست رقمية.")
        df['obv'] = np.nan
        return df

    # Initialize OBV series. First value is 0 or volume depending on preference.
    # Standard approach: OBV starts at 0, or first day's volume if close > prev_close (not applicable for first point).
    # Simpler: start OBV at 0.
    obv = np.zeros(len(df), dtype=np.float64)
    close = df['close'].values
    volume = df['volume'].values

    # Calculate close price differences
    close_diff = df['close'].diff().values # diff()[0] will be NaN

    # Iterate from the second data point
    for i in range(1, len(df)):
        # Handle potential NaNs in inputs
        if np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(close_diff[i]):
            obv[i] = obv[i-1] # Carry forward previous OBV if current data is bad
            continue

        if close_diff[i] > 0: # Price increased
            obv[i] = obv[i-1] + volume[i]
        elif close_diff[i] < 0: # Price decreased
             obv[i] = obv[i-1] - volume[i]
        else: # Price unchanged
             obv[i] = obv[i-1]

    df['obv'] = obv
    return df


def calculate_supertrend(df: pd.DataFrame, period: int = SUPERTREND_PERIOD, multiplier: float = SUPERTREND_MULTIPLIER) -> pd.DataFrame:
    """Calculates the SuperTrend indicator."""
    df_st = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df_st.columns for col in required_cols) or df_st[required_cols].isnull().all().any():
        logger.warning(f"⚠️ [Indicator SuperTrend] {df_st.index.name if df_st.index.name else ''}: أعمدة 'high', 'low', 'close' مفقودة أو فارغة.")
        df_st['supertrend'] = np.nan
        df_st['supertrend_trend'] = 0 # 0 for no trend, 1 for uptrend, -1 for downtrend
        return df_st

    # ATR is needed for SuperTrend
    df_st = calculate_atr_indicator(df_st, period=period) # Use the same period for ATR as for SuperTrend


    if 'atr' not in df_st.columns or df_st['atr'].isnull().all():
         logger.warning(f"⚠️ [Indicator SuperTrend] {df_st.index.name if df_st.index.name else ''}: لا يمكن حساب SuperTrend بسبب قيم ATR غير صالحة أو مفقودة.")
         df_st['supertrend'] = np.nan
         df_st['supertrend_trend'] = 0
         return df_st
    if len(df_st) < period: # Need enough data for ATR and initial calculations
        logger.warning(f"⚠️ [Indicator SuperTrend] {df_st.index.name if df_st.index.name else ''}: بيانات غير كافية ({len(df_st)} < {period}) لحساب SuperTrend.")
        df_st['supertrend'] = np.nan
        df_st['supertrend_trend'] = 0
        return df_st

    hl2 = (df_st['high'] + df_st['low']) / 2
    df_st['basic_ub'] = hl2 + multiplier * df_st['atr'] # Basic Upper Band
    df_st['basic_lb'] = hl2 - multiplier * df_st['atr'] # Basic Lower Band

    # Initialize final bands and SuperTrend series
    df_st['final_ub'] = 0.0
    df_st['final_lb'] = 0.0
    df_st['supertrend'] = np.nan
    df_st['supertrend_trend'] = 0 # 1 for uptrend, -1 for downtrend

    # Convert to numpy for faster iteration if needed, though pandas apply might also work
    close = df_st['close'].values
    basic_ub_np = df_st['basic_ub'].values
    basic_lb_np = df_st['basic_lb'].values
    final_ub_np = df_st['final_ub'].values # Will be populated
    final_lb_np = df_st['final_lb'].values # Will be populated
    supertrend_np = df_st['supertrend'].values # Will be populated
    supertrend_trend_np = df_st['supertrend_trend'].values # Will be populated

    # Initial trend (can be inferred or set, e.g., based on first close vs hl2 or wait for first cross)
    # For simplicity, start with no trend or infer from first few points if possible.
    # Or, more commonly, the loop starts from period index.
    # Let's assume the first trend is set based on the first valid comparison.

    for i in range(period, len(df_st)): # Start from 'period' to ensure ATR is somewhat stable
        if pd.isna(basic_ub_np[i]) or pd.isna(basic_lb_np[i]) or pd.isna(close[i]) or pd.isna(close[i-1]):
            # Carry forward if current data is bad
            final_ub_np[i] = final_ub_np[i-1] if i > 0 else np.nan
            final_lb_np[i] = final_lb_np[i-1] if i > 0 else np.nan
            supertrend_np[i] = supertrend_np[i-1] if i > 0 else np.nan
            supertrend_trend_np[i] = supertrend_trend_np[i-1] if i > 0 else 0
            continue

        # Calculate Final Upper Band
        if basic_ub_np[i] < final_ub_np[i-1] or close[i-1] > final_ub_np[i-1]:
            final_ub_np[i] = basic_ub_np[i]
        else:
            final_ub_np[i] = final_ub_np[i-1]

        # Calculate Final Lower Band
        if basic_lb_np[i] > final_lb_np[i-1] or close[i-1] < final_lb_np[i-1]:
            final_lb_np[i] = basic_lb_np[i]
        else:
            final_lb_np[i] = final_lb_np[i-1]
        
        # Determine SuperTrend and Trend Direction
        if supertrend_trend_np[i-1] <= 0: # If previous trend was down or neutral
            if close[i] > final_ub_np[i]: # Breakout above final upper band
                supertrend_trend_np[i] = 1
                supertrend_np[i] = final_lb_np[i]
            else: # No change in trend (still down or neutral)
                supertrend_trend_np[i] = -1 # Or keep previous if neutral was intended as separate state
                supertrend_np[i] = final_ub_np[i]
        elif supertrend_trend_np[i-1] == 1: # If previous trend was up
            if close[i] < final_lb_np[i]: # Breakdown below final lower band
                supertrend_trend_np[i] = -1
                supertrend_np[i] = final_ub_np[i]
            else: # No change in trend (still up)
                supertrend_trend_np[i] = 1
                supertrend_np[i] = final_lb_np[i]
    
    # Assign back to DataFrame
    df_st['final_ub'] = final_ub_np
    df_st['final_lb'] = final_lb_np
    df_st['supertrend'] = supertrend_np
    df_st['supertrend_trend'] = supertrend_trend_np
    
    # Fill initial NaNs for supertrend and trend (e.g., first 'period' rows)
    # Updated to use .bfill() directly to avoid FutureWarning
    df_st['supertrend'] = df_st['supertrend'].bfill()
    df_st['supertrend_trend'] = df_st['supertrend_trend'].bfill()


    df_st.drop(columns=['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, errors='ignore')

    return df_st


# ---------------------- Candlestick Patterns ----------------------

def is_hammer(row: pd.Series) -> int:
    """Checks for Hammer pattern (bullish signal). Returns 100 if hammer, 0 otherwise."""
    o, h, l, c = row.get('open'), row.get('high'), row.get('low'), row.get('close')
    if pd.isna([o, h, l, c]).any(): return 0
    body = abs(c - o)
    candle_range = h - l
    if candle_range == 0: return 0 # Avoid division by zero if candle has no range
    lower_shadow = min(o, c) - l
    upper_shadow = h - max(o, c)
    
    # Conditions for a Hammer:
    # 1. Small body compared to the candle range.
    # 2. Long lower shadow (at least ~2 times the body).
    # 3. Very small or no upper shadow.
    # Typically occurs after a downtrend, but this function only checks candle shape.
    is_small_body = body < (candle_range * 0.35) # Body is less than 35% of total range
    is_long_lower_shadow = lower_shadow >= 1.8 * body if body > 0.00001 else lower_shadow > candle_range * 0.6 # Handle zero body case
    is_small_upper_shadow = upper_shadow <= body * 0.6 if body > 0.00001 else upper_shadow < candle_range * 0.15

    return 100 if is_small_body and is_long_lower_shadow and is_small_upper_shadow else 0

def is_shooting_star(row: pd.Series) -> int:
    """Checks for Shooting Star pattern (bearish signal). Returns -100 if shooting star, 0 otherwise."""
    o, h, l, c = row.get('open'), row.get('high'), row.get('low'), row.get('close')
    if pd.isna([o, h, l, c]).any(): return 0
    body = abs(c - o)
    candle_range = h - l
    if candle_range == 0: return 0
    lower_shadow = min(o, c) - l
    upper_shadow = h - max(o, c)

    # Conditions for a Shooting Star (inverse of Hammer):
    # 1. Small body.
    # 2. Long upper shadow.
    # 3. Very small or no lower shadow.
    # Typically occurs after an uptrend.
    is_small_body = body < (candle_range * 0.35)
    is_long_upper_shadow = upper_shadow >= 1.8 * body if body > 0.00001 else upper_shadow > candle_range * 0.6
    is_small_lower_shadow = lower_shadow <= body * 0.6 if body > 0.00001 else lower_shadow < candle_range * 0.15
    
    return -100 if is_small_body and is_long_upper_shadow and is_small_lower_shadow else 0

def is_doji(row: pd.Series) -> int:
    """Checks for Doji pattern (uncertainty). Returns 100 if doji, 0 otherwise."""
    o, h, l, c = row.get('open'), row.get('high'), row.get('low'), row.get('close')
    if pd.isna([o, h, l, c]).any(): return 0
    candle_range = h - l
    if candle_range == 0: return 0 # If no range, it's effectively a doji if o=c=h=l
    # Doji: Open and Close are very close. Body is very small relative to range.
    return 100 if abs(c - o) <= (candle_range * 0.1) else 0

def compute_engulfing(df: pd.DataFrame, idx: int) -> int:
    """
    Checks for Bullish or Bearish Engulfing pattern at index 'idx'.
    Returns 100 for Bullish Engulfing, -100 for Bearish Engulfing, 0 otherwise.
    """
    if idx == 0: return 0 # Cannot be engulfing at the first candle
    prev = df.iloc[idx - 1]
    curr = df.iloc[idx]

    # Ensure all necessary price data is available
    if pd.isna([prev['close'], prev['open'], prev['high'], prev['low'],
                curr['close'], curr['open'], curr['high'], curr['low']]).any():
        return 0

    # Avoid engulfing a doji or very small-bodied candle (can be noisy)
    # Define a minimum body size for the first candle relative to its range
    prev_body = abs(prev['close'] - prev['open'])
    prev_range = prev['high'] - prev['low']
    if prev_range > 0 and prev_body < (prev_range * 0.1): # Previous candle is too small (like a doji)
        return 0


    # Bullish Engulfing: Current green candle engulfs previous red candle
    is_bullish_engulfing = (prev['close'] < prev['open'] and # Previous candle is red
                            curr['close'] > curr['open'] and # Current candle is green
                            curr['open'] <= prev['close'] and # Current open is at or below previous close
                            curr['close'] >= prev['open'])    # Current close is at or above previous open

    # Bearish Engulfing: Current red candle engulfs previous green candle
    is_bearish_engulfing = (prev['close'] > prev['open'] and # Previous candle is green
                            curr['close'] < curr['open'] and # Current candle is red
                            curr['open'] >= prev['close'] and # Current open is at or above previous close
                            curr['close'] <= prev['open'])    # Current close is at or below previous open

    if is_bullish_engulfing: return 100
    if is_bearish_engulfing: return -100
    return 0

def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds candlestick pattern signals to the DataFrame."""
    df = df.copy()
    symbol_name = df.index.name if df.index.name else 'DataFrame' # Get symbol for logging if available
    logger.debug(f"ℹ️ [Indicators] {symbol_name}: كشف أنماط الشموع...")
    df['Hammer'] = df.apply(is_hammer, axis=1)
    df['ShootingStar'] = df.apply(is_shooting_star, axis=1)
    df['Doji'] = df.apply(is_doji, axis=1)
    
    # Engulfing needs to look at the previous row, so apply carefully
    engulfing_values = [compute_engulfing(df, i) for i in range(len(df))]
    df['Engulfing'] = engulfing_values # 100 for bullish, -100 for bearish

    # Combine into simple bullish/bearish signals
    df['BullishCandleSignal'] = df.apply(lambda row: 1 if (row['Hammer'] == 100 or row['Engulfing'] == 100) else 0, axis=1)
    df['BearishCandleSignal'] = df.apply(lambda row: 1 if (row['ShootingStar'] == -100 or row['Engulfing'] == -100) else 0, axis=1)
    logger.debug(f"✅ [Indicators] {symbol_name}: تم كشف أنماط الشموع.")
    return df

# ---------------------- Other Helper Functions (Elliott, Swings, Volume) ----------------------
# These are not directly used by the ML model features but are part of the original bot's logic.
# They can be kept if the bot uses them for other purposes or if future ML models might use them.

def detect_swings(prices: np.ndarray, order: int = SWING_ORDER) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Detects swing points (peaks and troughs) in a time series (numpy array)."""
    n = len(prices)
    if n < 2 * order + 1: return [], [] # Not enough data to find swings of given order

    maxima_indices = []
    minima_indices = []

    for i in range(order, n - order):
        window = prices[i - order : i + order + 1]
        center_val = prices[i]

        if np.isnan(window).any(): continue # Skip if NaN in window

        if np.all(center_val >= window):
            # Optional: ensure it's not too close to a previous max
            if not maxima_indices or i > maxima_indices[-1] + order: # Simple distance check
                 maxima_indices.append(i)
        elif np.all(center_val <= window):
            if not minima_indices or i > minima_indices[-1] + order:
                minima_indices.append(i)

    maxima = [(idx, prices[idx]) for idx in maxima_indices]
    minima = [(idx, prices[idx]) for idx in minima_indices]
    return maxima, minima

def detect_elliott_waves(df: pd.DataFrame, order: int = SWING_ORDER) -> List[Dict[str, Any]]:
    """Simple attempt to identify Elliott Waves based on MACD histogram swings."""
    symbol_name = df.index.name if df.index.name else 'DataFrame'
    if 'macd_hist' not in df.columns or df['macd_hist'].isnull().all():
        logger.warning(f"⚠️ [Elliott] {symbol_name}: عمود 'macd_hist' مفقود أو فارغ لحساب موجات إليوت.")
        return []

    macd_values = df['macd_hist'].dropna().values
    if len(macd_values) < 2 * order + 1: # Check after dropna
         logger.warning(f"⚠️ [Elliott] {symbol_name}: بيانات MACD hist غير كافية ({len(macd_values)}) بعد إزالة قيم NaN.")
         return []

    maxima, minima = detect_swings(macd_values, order=order)

    # Map indices back to original DataFrame index (timestamps)
    df_nonan_macd = df['macd_hist'].dropna() # Get the Series with original timestamps
    all_swings = sorted(
        [(df_nonan_macd.index[idx], val, 'max') for idx, val in maxima] +
        [(df_nonan_macd.index[idx], val, 'min') for idx, val in minima],
        key=lambda x: x[0] # Sort by timestamp
    )

    waves = []
    wave_number = 1
    for timestamp, val, typ in all_swings:
        # Basic classification: Impulse if MACD hist is positive (for max) or non-negative (for min turning up)
        # Correction if MACD hist is negative (for min) or non-positive (for max turning down)
        # This is a very simplified interpretation.
        wave_type = "Impulse" if (typ == 'max' and val > 0) or (typ == 'min' and val >= 0) else "Correction"
        waves.append({
            "wave": wave_number,
            "timestamp": str(timestamp), # Convert timestamp to string for JSON
            "macd_hist_value": float(val), # Ensure float
            "swing_type": typ,
            "classified_type": wave_type
        })
        wave_number += 1
    return waves


def fetch_recent_volume(symbol: str) -> float:
    """Fetches the trading volume in USDT for the last 15 minutes for the specified symbol."""
    if not client:
         logger.error(f"❌ [Data Volume] عميل Binance غير مهيأ لجلب الحجم لـ {symbol}.")
         return 0.0
    try:
        logger.debug(f"ℹ️ [Data Volume] جلب حجم 15 دقيقة لـ {symbol}...")
        # Get 15 one-minute klines
        klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=15)
        if not klines or len(klines) < 15: # Ensure we have 15 klines
             logger.warning(f"⚠️ [Data Volume] بيانات 1m غير كافية (أقل من 15 شمعة) لـ {symbol}.")
             return 0.0

        # Sum of 'quote_asset_volume' (index 7)
        volume_usdt = sum(float(k[7]) for k in klines if len(k) > 7 and k[7] is not None)
        logger.debug(f"✅ [Data Volume] سيولة آخر 15 دقيقة لـ {symbol}: {volume_usdt:.2f} USDT")
        return volume_usdt
    except (BinanceAPIException, BinanceRequestException) as binance_err:
         logger.error(f"❌ [Data Volume] خطأ في Binance API أو الشبكة أثناء جلب الحجم لـ {symbol}: {binance_err}")
         return 0.0
    except Exception as e:
        logger.error(f"❌ [Data Volume] خطأ غير متوقع أثناء جلب الحجم لـ {symbol}: {e}", exc_info=True)
        return 0.0

# ---------------------- Comprehensive Performance Report Generation Function ----------------------
def generate_performance_report() -> str:
    """Generates a comprehensive performance report from the database in Arabic, including recent closed trades and USD profit/loss."""
    logger.info("ℹ️ [Report] إنشاء تقرير الأداء...")
    if not check_db_connection() or not conn or not cur: # Ensure cur is also available
        return "❌ لا يمكن إنشاء التقرير، مشكلة في اتصال قاعدة البيانات."
    try:
        # Use a new cursor for report generation to avoid conflicts if cur is used elsewhere
        with conn.cursor() as report_cur:
            report_cur.execute("SELECT id, symbol, entry_price, entry_time FROM signals WHERE achieved_target = FALSE ORDER BY entry_time DESC;")
            open_signals = report_cur.fetchall() # List of dicts
            open_signals_count = len(open_signals)

            # Get stats for closed signals where target was achieved
            report_cur.execute("""
                SELECT
                    COUNT(*) AS total_closed_achieved,
                    COALESCE(SUM(profit_percentage), 0) AS total_profit_pct_sum_achieved,
                    COALESCE(AVG(profit_percentage), 0) AS avg_profit_pct_achieved,
                    COALESCE(SUM( (entry_price * (1 + profit_percentage/100.0)) - entry_price ), 0) AS total_gross_profit_value_achieved,
                    COALESCE(SUM(entry_price), 0) AS total_entry_value_achieved
                FROM signals
                WHERE achieved_target = TRUE;
            """)
            achieved_stats = report_cur.fetchone() or {} # Ensure it's a dict even if no rows

            total_closed_achieved = achieved_stats.get('total_closed_achieved', 0)
            avg_win_pct = achieved_stats.get('avg_profit_pct_achieved', 0.0)
            # Calculate total profit in USDT based on TRADE_VALUE for each trade
            # This assumes each trade was exactly TRADE_VALUE at entry
            total_gross_profit_usdt_achieved = sum(
                TRADE_VALUE * (s['profit_percentage'] / 100.0)
                for s in get_achieved_signals_for_report(report_cur) # Helper to fetch details
            )
            
            # For simplicity, assuming fixed trade value for fee calculation
            total_fees_usdt_achieved = (total_closed_achieved * TRADE_VALUE * BINANCE_FEE_RATE) + \
                                       (total_closed_achieved * (TRADE_VALUE * (1 + avg_win_pct/100.0)) * BINANCE_FEE_RATE if avg_win_pct else 0)


            net_profit_usdt_achieved = total_gross_profit_usdt_achieved - total_fees_usdt_achieved
            net_profit_pct_overall_achieved = (net_profit_usdt_achieved / (total_closed_achieved * TRADE_VALUE)) * 100 if total_closed_achieved * TRADE_VALUE > 0 else 0.0


        report = (
            f"📊 *تقرير الأداء الشامل:*\n"
            f"_(افتراض حجم الصفقة: ${TRADE_VALUE:,.2f} ورسوم Binance: {BINANCE_FEE_RATE*100:.2f}% لكل طرف من الصفقة)_ \n"
            f"——————————————\n"
            f"📈 الإشارات المفتوحة حالياً: *{open_signals_count}*\n"
        )

        if open_signals:
            report += "  • التفاصيل (آخر 5):\n"
            for signal in open_signals[:5]: # Show details for a few open signals
                # Sanitize symbol for Markdown
                safe_symbol = str(signal['symbol']).replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')
                entry_time_str = signal['entry_time'].strftime('%Y-%m-%d %H:%M') if signal['entry_time'] else 'N/A'
                report += f"    - `{safe_symbol}` (دخول: ${signal['entry_price']:.8g} | فتح: {entry_time_str})\n"
        else:
            report += "  • لا توجد إشارات مفتوحة حالياً.\n"

        report += (
            f"——————————————\n"
            f"📉 *إحصائيات الإشارات المغلقة (تم تحقيق الهدف فقط):*\n"
            f"  • إجمالي الإشارات المغلقة (الرابحة): *{total_closed_achieved}*\n"
            # Since only target_achieved=TRUE are considered, win rate is 100% of these.
            # A more comprehensive report would include losses if they were tracked.
            f"  ✅ نسبة نجاح هذه الفئة: *100.00%* (من الصفقات التي حققت الهدف)\n"
            f"——————————————\n"
            f"💰 *الربحية (للصفقات التي حققت الهدف):*\n"
            f"  • إجمالي الربح الإجمالي (USDT): *${total_gross_profit_usdt_achieved:+.2f}*\n"
            f"  • إجمالي الرسوم المدفوعة المقدرة (USDT): *${total_fees_usdt_achieved:,.2f}*\n"
            f"  • *الربح الصافي المقدر (USDT):* *${net_profit_usdt_achieved:+.2f}*\n"
            f"  • متوسط ربح الصفقة (الرابحة): *{avg_win_pct:+.2f}%*\n"
            f"  • النسبة المئوية للربح الصافي الإجمالي (على رأس المال المستثمر في هذه الصفقات): *{net_profit_pct_overall_achieved:+.2f}%*\n"
            f"——————————————\n"
        )

        report += "ℹ️ *ملاحظة: هذا التقرير يعرض فقط الصفقات التي حققت الهدف، حيث أن منطق وقف الخسارة لم يتم تضمينه بشكل كامل في التتبع الحالي للإغلاق الخاسر.*\n"
        report += "\n——————————————\n"


        report += (
            f"🕰️ _التقرير محدث حتى: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        )

        logger.info("✅ [Report] تم إنشاء تقرير الأداء بنجاح.")
        return report

    except psycopg2.Error as db_err:
        logger.error(f"❌ [Report] خطأ في قاعدة البيانات أثناء إنشاء تقرير الأداء: {db_err}")
        if conn: conn.rollback() # Rollback if transaction was started
        return "❌ خطأ في قاعدة البيانات أثناء إنشاء تقرير الأداء."
    except Exception as e:
        logger.error(f"❌ [Report] خطأ غير متوقع أثناء إنشاء تقرير الأداء: {e}", exc_info=True)
        return "❌ حدث خطأ غير متوقع أثناء إنشاء تقرير الأداء."

def get_achieved_signals_for_report(report_cur: psycopg2.extensions.cursor) -> List[Dict]:
    """Helper to fetch details of achieved signals for report calculations."""
    report_cur.execute("SELECT entry_price, profit_percentage FROM signals WHERE achieved_target = TRUE;")
    return report_cur.fetchall()

# ---------------------- Trading Strategy (Adjusted for Scalping & ML) -------------------

class ScalpingTradingStrategy:
    """
    Encapsulates the trading strategy logic, now heavily reliant on ML model predictions
    and ensuring feature consistency.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        # CRITICAL: This list MUST match the features the ML model was trained on.
        # It is now dynamically set from ml_model_features loaded from DB,
        # or falls back to a hardcoded list if the model isn't loaded yet.
        if ml_model_features:
            self.feature_columns_for_ml = ml_model_features
            logger.info(f"📈 [Strategy {self.symbol}] Using ML model features from DB: {self.feature_columns_for_ml}")
        else:
            # Fallback hardcoded list (should ideally be avoided in production if ML is critical)
            self.feature_columns_for_ml = [
                f'ema_{EMA_SHORT_PERIOD}', f'ema_{EMA_LONG_PERIOD}', 'vwma',
                'rsi', 'atr', 'bb_upper', 'bb_lower', 'bb_middle',
                'macd', 'macd_signal', 'macd_hist',
                'adx', 'di_plus', 'di_minus', 'vwap', 'obv',
                'supertrend', 'supertrend_trend',
                'close_lag1', 'close_lag2', 'close_lag3',
                'rsi_lag1', 'rsi_lag2',
                'macd_lag1', 'macd_lag2',
                'supertrend_trend_lag1', 'supertrend_trend_lag2'
            ]
            logger.warning(f"⚠️ [Strategy {self.symbol}] ML model features not loaded. Falling back to hardcoded list: {self.feature_columns_for_ml}")

        # Log the features this strategy instance expects
        logger.debug(f"📈 [Strategy {self.symbol}] Initialized. Expecting {len(self.feature_columns_for_ml)} ML features: {self.feature_columns_for_ml}")


        # Optional conditions and weights (can be used if ML model is not available or as secondary confirmation)
        # These are less critical if ML model is the primary driver.
        self.condition_weights = {
            'rsi_ok': 0.5,
            'bullish_candle': 1.5,
            'not_bb_extreme': 0.5,
            'obv_rising': 1.0,
            'macd_hist_increasing': 2.0, # Reduced weight as ML is primary
            'obv_increasing_recent': 2.0, # Reduced weight
            'above_vwap': 1.0,
        }
        self.total_possible_score_optional = sum(self.condition_weights.values())
        self.min_score_threshold_pct_optional = 0.60 # Threshold for optional conditions if ML is not used

        # Essential traditional conditions (can be used as fallback or pre-filter if ML is not overriding)
        self.essential_conditions_traditional = [
            'price_above_emas_and_vwma',
            'ema_short_above_ema_long',
            'supertrend_up',
            'macd_positive_or_cross',
            'adx_trending_bullish_strong',
        ]


    def populate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Calculates all required indicators for the strategy AND for ML model prediction.
        **CRITICAL**: Must generate features identical to ml.py's training process.
        """
        symbol_name = self.symbol # Use self.symbol for logging context
        logger.debug(f"ℹ️ [Strategy {symbol_name}] حساب المؤشرات والسمات لنموذج ML...")
        
        # Determine minimum length needed for all indicators and lags
        # Max of all periods + max lag + buffer
        min_len_required = max(
            EMA_LONG_PERIOD, VWMA_PERIOD, RSI_PERIOD, ENTRY_ATR_PERIOD,
            BOLLINGER_WINDOW, MACD_SLOW, ADX_PERIOD * 2, SUPERTREND_PERIOD,
            3 # Max lag for close
        ) + 5 # Buffer

        if len(df) < min_len_required:
            logger.warning(f"⚠️ [Strategy {symbol_name}] DataFrame قصير جدًا ({len(df)} < {min_len_required}) لحساب المؤشرات والسمات المتأخرة.")
            return None

        try:
            df_calc = df.copy()
            df_calc.index.name = symbol_name # Assign symbol name to index for better logging in indicators

            # Calculate base indicators
            df_calc = calculate_atr_indicator(df_calc, ENTRY_ATR_PERIOD) # ATR used by SuperTrend and for target
            df_calc = calculate_supertrend(df_calc, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
            df_calc[f'ema_{EMA_SHORT_PERIOD}'] = calculate_ema(df_calc['close'], EMA_SHORT_PERIOD)
            df_calc[f'ema_{EMA_LONG_PERIOD}'] = calculate_ema(df_calc['close'], EMA_LONG_PERIOD)
            df_calc['vwma'] = calculate_vwma(df_calc, VWMA_PERIOD)
            df_calc = calculate_rsi_indicator(df_calc, RSI_PERIOD)
            df_calc = calculate_bollinger_bands(df_calc, BOLLINGER_WINDOW, BOLLINGER_STD_DEV)
            df_calc = calculate_macd(df_calc, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            adx_df = calculate_adx(df_calc, ADX_PERIOD)
            df_calc = df_calc.join(adx_df) # Joins 'adx', 'di_plus', 'di_minus'
            df_calc = calculate_vwap(df_calc)
            df_calc = calculate_obv(df_calc)
            df_calc = detect_candlestick_patterns(df_calc) # For 'BullishCandleSignal' if used by optional score

            # ** ADD LAGGED FEATURES - Must match ml.py **
            logger.debug(f"ℹ️ [Strategy {symbol_name}] إضافة السمات المتأخرة...")
            for lag in range(1, 4): # Lags 1, 2, 3 for close
                df_calc[f'close_lag{lag}'] = df_calc['close'].shift(lag)
            for lag in range(1, 3): # Lags 1, 2 for rsi, macd, supertrend_trend
                df_calc[f'rsi_lag{lag}'] = df_calc['rsi'].shift(lag)
                df_calc[f'macd_lag{lag}'] = df_calc['macd'].shift(lag) # Lag MACD line itself
                df_calc[f'supertrend_trend_lag{lag}'] = df_calc['supertrend_trend'].shift(lag)
            
            logger.debug(f"ℹ️ [Strategy {symbol_name}] أعمدة متوفرة بعد إضافة السمات المتأخرة: {df_calc.columns.tolist()}")


            # Ensure all feature columns for ML exist and are numeric
            missing_ml_cols = [col for col in self.feature_columns_for_ml if col not in df_calc.columns]
            if missing_ml_cols:
                logger.error(f"❌ [Strategy {symbol_name}] أعمدة سمات ML مفقودة بعد الحسابات: {missing_ml_cols}. لا يمكن المتابعة لهذا الرمز.")
                return None # Critical error if expected ML features are not generated

            for col in self.feature_columns_for_ml:
                df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')

            initial_len = len(df_calc)
            # Drop rows with NaN in any of the features required by the ML model OR essential base columns
            # Also include 'BullishCandleSignal' if it's used in optional scoring
            cols_for_dropna = list(set(self.feature_columns_for_ml + ['open', 'high', 'low', 'close', 'volume', 'BullishCandleSignal']))
            
            df_cleaned = df_calc.dropna(subset=cols_for_dropna).copy()
            dropped_count = initial_len - len(df_cleaned)

            if dropped_count > 0:
                 logger.debug(f"ℹ️ [Strategy {symbol_name}] تم إسقاط {dropped_count} صفًا بسبب قيم NaN في المؤشرات أو سمات ML.")
            if df_cleaned.empty:
                logger.warning(f"⚠️ [Strategy {symbol_name}] DataFrame فارغ بعد إزالة قيم NaN للمؤشرات وسمات ML.")
                return None

            # Log a sample of the last row's features for ML to help verify
            if not df_cleaned.empty:
                last_row_sample = df_cleaned[self.feature_columns_for_ml].iloc[-1].to_dict()
                logger.debug(f"✅ [Strategy {symbol_name}] تم حساب المؤشرات والسمات. عينة من سمات ML للصف الأخير: {json.dumps(convert_np_values(last_row_sample), indent=2)}")
            return df_cleaned

        except KeyError as ke:
             logger.error(f"❌ [Strategy {symbol_name}] خطأ: لم يتم العثور على عمود مطلوب '{ke}' أثناء حساب المؤشر.", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"❌ [Strategy {symbol_name}] خطأ غير متوقع أثناء حساب المؤشر: {e}", exc_info=True)
            return None


    def generate_buy_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Generates a buy signal. Primarily uses ML model prediction.
        Falls back to traditional logic if ML model is unavailable or fails.
        """
        symbol_name = self.symbol
        logger.debug(f"ℹ️ [Strategy {symbol_name}] إنشاء إشارة شراء...")

        min_signal_data_len = max(RECENT_EMA_CROSS_LOOKBACK, MACD_HIST_INCREASE_CANDLES, OBV_INCREASE_CANDLES) + 1
        if df_processed is None or df_processed.empty or len(df_processed) < min_signal_data_len:
            logger.warning(f"⚠️ [Strategy {symbol_name}] DataFrame فارغ أو قصير جدًا (<{min_signal_data_len})، لا يمكن إنشاء إشارة.")
            return None

        # Ensure all required columns for signal generation, including ML features, are present
        # This check is somewhat redundant if populate_indicators handles it, but good as a safeguard.
        missing_cols_check = [col for col in self.feature_columns_for_ml if col not in df_processed.columns]
        if missing_cols_check:
            logger.error(f"⚠️ [Strategy {symbol_name}] DataFrame يفتقد أعمدة سمات ML المطلوبة للإشارة: {missing_cols_check}. لا يمكن إنشاء إشارة.")
            return None

        last_row = df_processed.iloc[-1]
        recent_df = df_processed.iloc[-min_signal_data_len:] # For conditions looking at recent candles

        # Check for NaNs in the last row for critical features
        if last_row[self.feature_columns_for_ml].isnull().any():
             logger.warning(f"⚠️ [Strategy {symbol_name}] الصف الأخير يحتوي على قيم NaN في سمات ML المطلوبة. لا يمكن إنشاء إشارة. البيانات: {last_row[self.feature_columns_for_ml].to_dict()}")
             return None


        signal_details = {} # Initialize to store reasons for signal generation

        # --- ML Model Prediction ---
        ml_prediction_made = False
        ml_prediction_is_bullish = False
        ml_prediction_result_text = "N/A (نموذج غير محمل أو خطأ)"
        ml_pred_proba_bullish = 0.0 # Store bullish probability

        if ml_model and ml_model_features: # Global ml_model loaded from DB and features are known
            try:
                # Ensure features are in the correct order and format using ml_model_features
                features_for_prediction_df = pd.DataFrame([last_row[self.feature_columns_for_ml].values], columns=self.feature_columns_for_ml)
                
                # Log the exact features being sent to the model
                logger.debug(f"ℹ️ [Strategy {symbol_name}] سمات تُرسل إلى نموذج ML للتنبؤ: {features_for_prediction_df.iloc[0].to_dict()}")

                ml_pred_proba_array = ml_model.predict_proba(features_for_prediction_df)[0] # Get probabilities [prob_class_0, prob_class_1]
                ml_pred_class = np.argmax(ml_pred_proba_array) # Get the class with the highest probability
                ml_pred_proba_bullish = ml_pred_proba_array[1] # Probability of class 1 (bullish)
                
                logger.info(f"✨ [Strategy {symbol_name}] تنبؤ نموذج ML: الفئة={ml_pred_class}, الاحتمالات={ml_pred_proba_array}")
                
                if ml_pred_class == 1: # Assuming 1 is the "bullish" or "target will be hit" class
                    ml_prediction_is_bullish = True
                    ml_prediction_result_text = f'صعودي (فئة 1، ثقة: {ml_pred_proba_bullish:.2%}) ✅'
                else:
                    ml_prediction_result_text = f'هبوطي/محايد (فئة 0، ثقة: {ml_pred_proba_array[0]:.2%}) ❌'
                ml_prediction_made = True
            except Exception as ml_err:
                logger.error(f"❌ [Strategy {symbol_name}] خطأ في تنبؤ نموذج ML: {ml_err}", exc_info=True)
                ml_prediction_result_text = "خطأ في التنبؤ"
        else:
            ml_prediction_result_text = "نموذج ML غير محمل أو لا توجد سمات"
        
        signal_details['ML_Prediction_Raw'] = ml_prediction_result_text # Store raw text for alert
        signal_details['ML_Model_Used'] = ML_MODEL_NAME if ml_model else "None"
        signal_details['ML_Bullish_Probability'] = f"{ml_pred_proba_bullish:.4f}" if ml_prediction_made else "N/A"


        # --- BTC Trend Filter (General Market Condition) ---
        btc_trend = get_btc_trend_4h()
        signal_details['BTC_Trend_4H'] = btc_trend
        if "هبوط" in btc_trend:
            logger.info(f"ℹ️ [Strategy {symbol_name}] التداول متوقف مؤقتًا بسبب اتجاه البيتكوين الهابط العام ({btc_trend}).")
            return None # Hard stop if BTC is bearish, regardless of ML
        elif "N/A" in btc_trend:
             logger.warning(f"⚠️ [Strategy {symbol_name}] لا يمكن تحديد اتجاه البيتكوين، سيتم تجاهل هذا الشرط العام للسوق.")
        

        # --- Decision Logic: Primarily based on ML, fallback to traditional ---
        final_signal_decision = False
        strategy_name_used = "ML_Driven_Scalp"

        if ml_prediction_made and ml_prediction_is_bullish:
            logger.info(f"✅ [Strategy {symbol_name}] إشارة شراء مؤكدة بناءً على تنبؤ ML الصعودي.")
            final_signal_decision = True
            signal_details['Decision_Basis'] = "ML Prediction Bullish"

        elif not ml_model or not ml_prediction_made: 
            logger.warning(f"⚠️ [Strategy {symbol_name}] نموذج ML غير متاح أو فشل التنبؤ. الرجوع إلى الاستراتيجية التقليدية.")
            strategy_name_used = "Traditional_Scalp_Fallback"
            signal_details['Decision_Basis'] = "Fallback: Traditional Logic"

            essential_passed_trad = True
            failed_essential_conditions_trad = []
            
            # Price vs EMAs and VWMA
            if not (pd.notna(last_row[f'ema_{EMA_SHORT_PERIOD}']) and pd.notna(last_row[f'ema_{EMA_LONG_PERIOD}']) and pd.notna(last_row['vwma']) and
                    last_row['close'] > last_row[f'ema_{EMA_SHORT_PERIOD}'] and
                    last_row['close'] > last_row[f'ema_{EMA_LONG_PERIOD}'] and
                    last_row['close'] > last_row['vwma']):
                essential_passed_trad = False; failed_essential_conditions_trad.append('Price Not Above EMAs/VWMA')
            signal_details['Trad_Price_MA_Alignment'] = 'Pass' if 'Price Not Above EMAs/VWMA' not in failed_essential_conditions_trad else 'Fail'

            # EMA Order
            if not (pd.notna(last_row[f'ema_{EMA_SHORT_PERIOD}']) and pd.notna(last_row[f'ema_{EMA_LONG_PERIOD}']) and
                    last_row[f'ema_{EMA_SHORT_PERIOD}'] > last_row[f'ema_{EMA_LONG_PERIOD}']):
                 essential_passed_trad = False; failed_essential_conditions_trad.append('Short EMA Not Above Long EMA')
            signal_details['Trad_EMA_Order'] = 'Pass' if 'Short EMA Not Above Long EMA' not in failed_essential_conditions_trad else 'Fail'
            
            # Supertrend
            if not (pd.notna(last_row['supertrend']) and last_row['close'] > last_row['supertrend'] and last_row['supertrend_trend'] == 1):
                 essential_passed_trad = False; failed_essential_conditions_trad.append('SuperTrend Not Bullish')
            signal_details['Trad_SuperTrend'] = 'Pass' if 'SuperTrend Not Bullish' not in failed_essential_conditions_trad else 'Fail'

            # MACD
            if not (pd.notna(last_row['macd_hist']) and pd.notna(last_row['macd']) and pd.notna(last_row['macd_signal']) and (last_row['macd_hist'] > 0 or last_row['macd'] > last_row['macd_signal'])):
                 essential_passed_trad = False; failed_essential_conditions_trad.append('MACD Not Bullish')
            signal_details['Trad_MACD'] = 'Pass' if 'MACD Not Bullish' not in failed_essential_conditions_trad else 'Fail'

            # ADX/DI
            if not (pd.notna(last_row['adx']) and pd.notna(last_row['di_plus']) and pd.notna(last_row['di_minus']) and last_row['adx'] > MIN_ADX_TREND_STRENGTH and last_row['di_plus'] > last_row['di_minus']):
                 essential_passed_trad = False; failed_essential_conditions_trad.append('ADX/DI Not Bullish Strong')
            signal_details['Trad_ADX_DI'] = 'Pass' if 'ADX/DI Not Bullish Strong' not in failed_essential_conditions_trad else 'Fail'


            if not essential_passed_trad:
                logger.debug(f"ℹ️ [Strategy {symbol_name}] (Fallback) فشلت الشروط الإلزامية التقليدية: {', '.join(failed_essential_conditions_trad)}. تم رفض الإشارة.")
                return None
            
            current_score_optional = 0.0
            if pd.notna(last_row['rsi']) and RSI_OVERSOLD < last_row['rsi'] < RSI_OVERBOUGHT : current_score_optional += self.condition_weights['rsi_ok']; signal_details['Fallback_RSI_OK'] = 'Pass'
            if last_row.get('BullishCandleSignal', 0) == 1: current_score_optional += self.condition_weights['bullish_candle']; signal_details['Fallback_BullishCandle'] = 'Pass'
            # ... (add other optional conditions for fallback scoring and signal_details) ...

            min_total_score_needed = self.total_possible_score_optional * self.min_score_threshold_pct_optional
            if current_score_optional >= min_total_score_needed:
                logger.info(f"✅ [Strategy {symbol_name}] (Fallback) إشارة شراء مؤكدة بناءً على الشروط التقليدية (الدرجة: {current_score_optional:.2f}).")
                final_signal_decision = True
                signal_details['Fallback_Score'] = f"{current_score_optional:.2f}/{self.total_possible_score_optional:.2f}"
            else:
                logger.debug(f"ℹ️ [Strategy {symbol_name}] (Fallback) لم يتم استيفاء درجة الشروط الاختيارية التقليدية ({current_score_optional:.2f} < {min_total_score_needed:.2f}). تم رفض الإشارة.")
                return None
        else: 
            logger.info(f"ℹ️ [Strategy {symbol_name}] تنبؤ نموذج ML ليس صعوديًا. تم رفض الإشارة.")
            return None


        if not final_signal_decision:
            return None 

        # --- Final Checks (Volume, Profit Margin) ---
        volume_recent = fetch_recent_volume(self.symbol)
        signal_details['Volume_15m_USDT'] = f"{volume_recent:,.0f}"
        if volume_recent < MIN_VOLUME_15M_USDT:
            logger.info(f"ℹ️ [Strategy {symbol_name}] السيولة ({volume_recent:,.0f} USDT) أقل من الحد الأدنى المطلوب ({MIN_VOLUME_15M_USDT:,.0f} USDT). تم رفض الإشارة.")
            return None

        current_price = last_row['close']
        current_atr = last_row.get('atr')

        if pd.isna(current_atr) or current_atr <= 0:
             logger.warning(f"⚠️ [Strategy {symbol_name}] قيمة ATR غير صالحة ({current_atr}) لحساب الهدف. لا يمكن إنشاء إشارة.")
             return None

        initial_target = current_price + (ENTRY_ATR_MULTIPLIER * current_atr)
        profit_margin_pct = ((initial_target / current_price) - 1) * 100 if current_price > 0 else 0
        signal_details['Calculated_Target_ATR_Based'] = f"{initial_target:.8g} ({profit_margin_pct:.2f}%)"

        if profit_margin_pct < MIN_PROFIT_MARGIN_PCT:
            logger.info(f"ℹ️ [Strategy {symbol_name}] هامش الربح المحسوب ({profit_margin_pct:.2f}%) أقل من الحد الأدنى المطلوب ({MIN_PROFIT_MARGIN_PCT:.2f}%). تم رفض الإشارة.")
            return None

        final_signal_details_serializable = convert_np_values(signal_details)

        signal_output = {
            'symbol': self.symbol,
            'entry_price': float(f"{current_price:.8g}"), 
            'initial_target': float(f"{initial_target:.8g}"),
            'current_target': float(f"{initial_target:.8g}"), 
            'r2_score': ml_pred_proba_bullish if ml_prediction_made else signal_details.get('Fallback_Score', 0.0), # Use ML bullish prob as score
            'strategy_name': strategy_name_used,
            'signal_details': final_signal_details_serializable, 
            'volume_15m': volume_recent,
            'trade_value': TRADE_VALUE, 
            'total_possible_score': self.total_possible_score_optional if "Fallback" in strategy_name_used else 1.0 # 1.0 for ML prob
        }

        logger.info(f"✅ [Strategy {symbol_name}] تم تأكيد إشارة الشراء النهائية. السعر: {current_price:.6f}, التفاصيل: {json.dumps(final_signal_details_serializable, ensure_ascii=False)}")
        return signal_output


# ---------------------- Telegram Functions ----------------------
def send_telegram_message(target_chat_id: str, text: str, reply_markup: Optional[Dict] = None, parse_mode: str = 'Markdown', disable_web_page_preview: bool = True, timeout: int = 20) -> Optional[Dict]:
    """Sends a message via Telegram Bot API with improved error handling."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': str(target_chat_id),
        'text': text,
        'parse_mode': parse_mode,
        'disable_web_page_preview': disable_web_page_preview
    }
    if reply_markup:
        try:
            # Ensure all values in reply_markup are JSON serializable
            payload['reply_markup'] = json.dumps(convert_np_values(reply_markup))
        except (TypeError, ValueError) as json_err:
             logger.error(f"❌ [Telegram] فشل تحويل reply_markup إلى JSON: {json_err} - Markup: {reply_markup}")
             # Optionally send message without markup or handle error
             return None # Or send without markup: payload.pop('reply_markup', None)

    logger.debug(f"ℹ️ [Telegram] إرسال رسالة إلى {target_chat_id}...")
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
        logger.info(f"✅ [Telegram] تم إرسال الرسالة بنجاح إلى {target_chat_id}.")
        return response.json()
    except requests.exceptions.Timeout:
         logger.error(f"❌ [Telegram] فشل إرسال الرسالة إلى {target_chat_id} (مهلة).")
         return None
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة إلى {target_chat_id} (خطأ HTTP: {http_err.response.status_code}).")
        try:
            error_details = http_err.response.json() # Try to get error details from Telegram API
            logger.error(f"❌ [Telegram] تفاصيل خطأ API: {error_err_details}")
        except json.JSONDecodeError: # If response is not JSON
            logger.error(f"❌ [Telegram] تعذر فك تشفير استجابة الخطأ: {http_err.response.text}")
        return None
    except requests.exceptions.RequestException as req_err: # For other network issues
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة إلى {target_chat_id} (خطأ في الطلب): {req_err}")
        return None
    except Exception as e: # Catch-all for any other unexpected error
         logger.error(f"❌ [Telegram] خطأ غير متوقع أثناء إرسال الرسالة: {e}", exc_info=True)
         return None

def send_telegram_alert(signal_data: Dict[str, Any], timeframe: str) -> None:
    """Formats and sends a new trading signal alert to Telegram in Arabic, displaying ML info."""
    logger.debug(f"ℹ️ [Telegram Alert] تنسيق وإرسال تنبيه للإشارة: {signal_data.get('symbol', 'N/A')}")
    try:
        entry_price = float(signal_data['entry_price'])
        target_price = float(signal_data['initial_target'])
        symbol = signal_data['symbol']
        strategy_name_raw = signal_data.get('strategy_name', 'N/A')
        # Clean up strategy name for display
        strategy_name_display = strategy_name_raw.replace('_', ' ').title()

        signal_score_display = signal_data.get('r2_score', 0.0) # This is now ML confidence or fallback score
        # total_possible_score = signal_data.get('total_possible_score', 0) # May not be relevant if ML driven

        volume_15m = signal_data.get('volume_15m', 0.0)
        trade_value_signal = signal_data.get('trade_value', TRADE_VALUE) # Use the value from signal if present
        
        # Get signal details, ensure it's a dict
        signal_details_dict = signal_data.get('signal_details', {})
        if not isinstance(signal_details_dict, dict):
            logger.warning(f"⚠️ [Telegram Alert] signal_details ليست قاموسًا لـ {symbol}. استخدام قاموس فارغ.")
            signal_details_dict = {}


        profit_pct = ((target_price / entry_price) - 1) * 100 if entry_price > 0 else 0

        # Calculate fees and net profit based on the trade_value from the signal
        entry_fee = trade_value_signal * BINANCE_FEE_RATE
        exit_value_gross = trade_value_signal * (1 + profit_pct / 100.0)
        exit_fee = exit_value_gross * BINANCE_FEE_RATE
        total_trade_fees = entry_fee + exit_fee

        profit_usdt_gross = trade_value_signal * (profit_pct / 100.0)
        profit_usdt_net = profit_usdt_gross - total_trade_fees

        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')

        fear_greed = get_fear_greed_index()
        # Get specific details from signal_details_dict with fallbacks
        ml_prediction_text = signal_details_dict.get('ML_Prediction_Raw', 'N/A')
        btc_trend_text = signal_details_dict.get('BTC_Trend_4H', 'N/A')
        decision_basis_text = signal_details_dict.get('Decision_Basis', 'غير محدد')
        model_used_text = signal_details_dict.get('ML_Model_Used', 'N/A')
        ml_confidence_text = signal_details_dict.get('ML_Bullish_Probability', 'N/A')


        message = (
            f"💡 *إشارة تداول جديدة ({strategy_name_display})* 💡\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"📈 **نوع الإشارة:** شراء (طويل)\n"
            f"🕰️ **الإطار الزمني للتحليل:** {timeframe}\n"
            f"📊 **أساس القرار:** *{decision_basis_text}*\n"
            f"🧠 **نموذج ML المستخدم:** `{model_used_text}`\n"
            f"💬 **تنبؤ النموذج:** *{ml_prediction_text}*\n"
            f"🎯 **احتمالية الصعود (ML):** *{ml_confidence_text}*\n"
            f"💧 **السيولة (15 دقيقة):** {volume_15m:,.0f} USDT\n"
            f"——————————————\n"
            f"➡️ **سعر الدخول المقترح:** `${entry_price:,.8g}`\n"
            f"🎯 **الهدف الأولي (مبني على ATR):** `${target_price:,.8g}`\n"
            f"💰 **الربح المتوقع (إجمالي):** ({profit_pct:+.2f}% / ≈ ${profit_usdt_gross:+.2f})\n"
            f"💸 **الرسوم المتوقعة (لكامل الصفقة):** ${total_trade_fees:,.2f}\n"
            f"📈 **الربح الصافي المتوقع:** ${profit_usdt_net:+.2f}\n"
            f"——————————————\n"
            f"😨/🤑 **مؤشر الخوف والجشع (عام):** {fear_greed}\n"
            f"₿ **اتجاه البيتكوين (4 ساعات - عام):** {btc_trend_text}\n"
            f"——————————————\n"
        )
        
        # Add a small sample of other signal details if they exist
        other_details_to_show = {
            k: v for k, v in signal_details_dict.items() 
            if k not in ['ML_Prediction_Raw', 'BTC_Trend_4H', 'Decision_Basis', 'ML_Model_Used', 'Volume_15m_USDT', 'Calculated_Target_ATR_Based', 'ML_Bullish_Probability']
        }
        if "Fallback" in strategy_name_raw and 'Fallback_Score' in signal_details_dict:
             message += f"คะแนน الاستراتيجية التقليدية: {signal_details_dict['Fallback_Score']}\n" # Arabic: Traditional Strategy Score

        if other_details_to_show:
            message += "📋 *تفاصيل إضافية من الاستراتيجية:*\n"
            for key, value in list(other_details_to_show.items())[:3]: # Show first 3 other details
                 # Sanitize key for Markdown if it contains problematic characters
                safe_key = str(key).replace('_', ' ').title()
                message += f"  - {safe_key}: {value}\n"
            if len(other_details_to_show) > 3:
                message += "  ...\n"
        
        message += f"——————————————\n⏰ {timestamp_str}"


        reply_markup = {
            "inline_keyboard": [
                [{"text": "📊 عرض تقرير الأداء", "callback_data": "get_report"}]
            ]
        }

        send_telegram_message(CHAT_ID, message, reply_markup=reply_markup, parse_mode='Markdown')

    except KeyError as ke:
        logger.error(f"❌ [Telegram Alert] بيانات الإشارة غير مكتملة للرمز {signal_data.get('symbol', 'N/A')}: مفتاح مفقود {ke}", exc_info=True)
    except Exception as e:
        logger.error(f"❌ [Telegram Alert] فشل إرسال تنبيه الإشارة للرمز {signal_data.get('symbol', 'N/A')}: {e}", exc_info=True)

def send_tracking_notification(details: Dict[str, Any]) -> None:
    """Formats and sends enhanced Telegram notifications for tracking events in Arabic."""
    symbol = details.get('symbol', 'N/A')
    signal_id = details.get('id', 'N/A')
    notification_type = details.get('type', 'unknown')
    message = ""
    # Sanitize symbol for Markdown
    safe_symbol = str(symbol).replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')
    
    closing_price = details.get('closing_price', 0.0)
    profit_pct = details.get('profit_pct', 0.0)
    current_price = details.get('current_price', 0.0) # For target updates
    time_to_target_str = details.get('time_to_target', 'N/A') # Already a string
    old_target = details.get('old_target', 0.0)
    new_target = details.get('new_target', 0.0)


    logger.debug(f"ℹ️ [Notification] تنسيق إشعار التتبع: ID={signal_id}, Type={notification_type}, Symbol={symbol}")

    if notification_type == 'target_hit':
        message = (
            f"✅ *تم الوصول إلى الهدف (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"🎯 **سعر الإغلاق (الهدف):** `${closing_price:,.8g}`\n"
            f"💰 **الربح المحقق:** {profit_pct:+.2f}%\n"
            f"⏱️ **الوقت المستغرق:** {time_to_target_str}" # time_to_target is already formatted string
        )
    elif notification_type == 'target_updated':
         message = (
             f"↗️ *تم تحديث الهدف (ID: {signal_id})*\n"
             f"——————————————\n"
             f"🪙 **الزوج:** `{safe_symbol}`\n"
             f"📈 **السعر الحالي:** `${current_price:,.8g}`\n"
             f"🎯 **الهدف السابق:** `${old_target:,.8g}`\n"
             f"🎯 **الهدف الجديد:** `${new_target:,.8g}`\n"
             f"ℹ️ *تم التحديث بناءً على استمرار الزخم الصعودي أو إعادة تقييم ATR.*"
         )
    # Add other notification types here if needed (e.g., stop_loss_hit)
    else:
        logger.warning(f"⚠️ [Notification] نوع إشعار غير معروف: {notification_type} للتفاصيل: {details}")
        return # Do not send if type is unknown

    if message:
        send_telegram_message(CHAT_ID, message, parse_mode='Markdown')

# ---------------------- Database Functions (Insert and Update) ----------------------
def insert_signal_into_db(signal: Dict[str, Any]) -> bool:
    """Inserts a new signal into the signals table with the weighted score and entry time."""
    if not check_db_connection() or not conn: # Ensure conn is not None
        logger.error(f"❌ [DB Insert] فشل إدراج الإشارة {signal.get('symbol', 'N/A')} بسبب مشكلة في اتصال قاعدة البيانات.")
        return False

    symbol_to_insert = signal.get('symbol', 'N/A')
    logger.debug(f"ℹ️ [DB Insert] محاولة إدراج إشارة لـ {symbol_to_insert}...")
    try:
        # Ensure all parts of the signal are JSON serializable before trying to dump signal_details
        signal_prepared_for_db = convert_np_values(signal)
        signal_details_json = json.dumps(signal_prepared_for_db.get('signal_details', {}), ensure_ascii=False) # ensure_ascii=False for Arabic

        with conn.cursor() as cur_ins: # Use a new cursor
            insert_query = sql.SQL("""
                INSERT INTO signals
                 (symbol, entry_price, initial_target, current_target,
                 r2_score, strategy_name, signal_details, volume_15m, entry_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW()); 
            """) # entry_time is set to NOW() by default in DB or here
            cur_ins.execute(insert_query, (
                signal_prepared_for_db['symbol'],
                signal_prepared_for_db['entry_price'],
                signal_prepared_for_db['initial_target'],
                signal_prepared_for_db['current_target'],
                signal_prepared_for_db.get('r2_score'), # ML confidence or other score
                signal_prepared_for_db.get('strategy_name', 'unknown'),
                signal_details_json, # JSONB column
                signal_prepared_for_db.get('volume_15m')
            ))
        conn.commit() # Commit the transaction
        logger.info(f"✅ [DB Insert] تم إدراج إشارة لـ {symbol_to_insert} في قاعدة البيانات (الاستراتيجية: {signal_prepared_for_db.get('strategy_name')}).")
        return True
    except psycopg2.Error as db_err:
        logger.error(f"❌ [DB Insert] خطأ في قاعدة البيانات أثناء إدراج إشارة لـ {symbol_to_insert}: {db_err}")
        if conn: conn.rollback() # Rollback on error
        return False
    except (TypeError, ValueError) as convert_err: # Error during convert_np_values or json.dumps
         logger.error(f"❌ [DB Insert] خطأ في تحويل بيانات الإشارة قبل الإدراج لـ {symbol_to_insert}: {convert_err} - بيانات الإشارة: {signal}")
         if conn: conn.rollback()
         return False
    except Exception as e:
        logger.error(f"❌ [DB Insert] خطأ غير متوقع أثناء إدراج إشارة لـ {symbol_to_insert}: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

# ---------------------- Open Signal Tracking Function ----------------------
def track_signals() -> None:
    """Tracks open signals and checks targets. Calculates time to target upon hit."""
    logger.info("ℹ️ [Tracker] بدء عملية تتبع الإشارات المفتوحة...")
    while True:
        active_signals_summary: List[str] = [] # For logging current state
        processed_in_cycle = 0
        try:
            if not check_db_connection() or not conn: # Ensure conn is not None
                logger.warning("⚠️ [Tracker] تخطي دورة التتبع بسبب مشكلة في اتصال قاعدة البيانات.")
                time.sleep(15) # Wait before retrying connection check
                continue

            # Use a new cursor for fetching open signals
            with conn.cursor() as track_cur:
                 track_cur.execute("""
                    SELECT id, symbol, entry_price, initial_target, current_target, entry_time, strategy_name, signal_details
                    FROM signals
                    WHERE achieved_target = FALSE; 
                """) # Fetch more details if needed for logic
                 open_signals: List[Dict] = track_cur.fetchall() # Returns list of dicts

            if not open_signals:
                # logger.debug("ℹ️ [Tracker] لا توجد إشارات مفتوحة حاليًا للتتبع.")
                time.sleep(10) # Sleep longer if no signals
                continue

            logger.debug(f"ℹ️ [Tracker] تتبع {len(open_signals)} إشارة مفتوحة...")

            for signal_row in open_signals:
                signal_id = signal_row['id']
                symbol = signal_row['symbol']
                processed_in_cycle += 1
                update_executed_in_loop = False # Flag for this specific signal in the loop

                try:
                    entry_price = float(signal_row['entry_price'])
                    entry_time = signal_row['entry_time'] # Already datetime object from DB
                    # CRITICAL FIX: Ensure current_target_db is converted to a native float
                    current_target_db = convert_np_values(signal_row['current_target']) 

                    current_price_ws = ticker_data.get(symbol) # Get live price from WebSocket data

                    if current_price_ws is None:
                         logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): السعر الحالي غير متاح في بيانات التيكر. تخطي هذا الفحص.")
                         continue # Skip this signal if no live price

                    active_signals_summary.append(f"{symbol}(ID:{signal_id}): P={current_price_ws:.4f} T={current_target_db:.4f}")

                    update_query_sql: Optional[sql.SQL] = None
                    update_params_tuple: Tuple = ()
                    log_message_str: Optional[str] = None
                    notification_details_dict: Dict[str, Any] = {'symbol': symbol, 'id': signal_id, 'current_price': current_price_ws}


                    # --- Check and Update Logic ---
                    # 1. Check for Target Hit
                    if current_price_ws >= current_target_db:
                        profit_pct_calc = ((current_target_db / entry_price) - 1) * 100 if entry_price > 0 else 0
                        closed_at_dt = datetime.now() # Timestamp of closing
                        time_to_target_td = closed_at_dt - entry_time if entry_time else timedelta(0)
                        
                        # Format timedelta to a readable string (e.g., "X days, HH:MM:SS")
                        days = time_to_target_td.days
                        seconds = time_to_target_td.seconds
                        hours = seconds // 3600
                        minutes = (seconds % 3600) // 60
                        secs = seconds % 60
                        time_to_target_formatted_str = f"{days}d {hours:02}:{minutes:02}:{secs:02}" if days > 0 else f"{hours:02}:{minutes:02}:{secs:02}"


                        update_query_sql = sql.SQL("UPDATE signals SET achieved_target = TRUE, closing_price = %s, closed_at = %s, profit_percentage = %s, time_to_target = %s WHERE id = %s;")
                        update_params_tuple = (convert_np_values(current_target_db), closed_at_dt, convert_np_values(profit_pct_calc), time_to_target_td, signal_id)
                        log_message_str = f"🎯 [Tracker] {symbol}(ID:{signal_id}): تم الوصول إلى الهدف عند {current_target_db:.8g} (الربح: {profit_pct_calc:+.2f}%, الوقت: {time_to_target_formatted_str})."
                        notification_details_dict.update({'type': 'target_hit', 'closing_price': current_target_db, 'profit_pct': profit_pct_calc, 'time_to_target': time_to_target_formatted_str})
                        update_executed_in_loop = True

                    # 2. Check for Target Extension (Trailing Target Logic - Simplified)
                    # This is an example; more sophisticated trailing logic could be added.
                    # For now, the target update is based on re-evaluating with fresh ATR if price is near target.
                    if not update_executed_in_loop: # Only if target not hit
                        # If price is approaching the target, consider re-evaluating for a new, higher target
                        if current_price_ws >= current_target_db * (1 - TARGET_APPROACH_THRESHOLD_PCT): # e.g., within 0.5% of target
                             logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): السعر ({current_price_ws:.8g}) قريب من الهدف ({current_target_db:.8g}). التحقق من إمكانية تمديد الهدف...")

                             # Fetch fresh data to calculate a new ATR and potentially a new target
                             df_continuation_check = fetch_historical_data(symbol, interval=SIGNAL_TRACKING_TIMEFRAME, days=SIGNAL_TRACKING_LOOKBACK_DAYS)

                             if df_continuation_check is not None and not df_continuation_check.empty:
                                 # We only need ATR from this fresh data
                                 df_atr_recalc = calculate_atr_indicator(df_continuation_check, ENTRY_ATR_PERIOD)
                                 if not df_atr_recalc.empty and 'atr' in df_atr_recalc.columns and pd.notna(df_atr_recalc['atr'].iloc[-1]):
                                     latest_atr_recalc = df_atr_recalc['atr'].iloc[-1]
                                     if latest_atr_recalc > 0:
                                         potential_new_target_val = current_price_ws + (ENTRY_ATR_MULTIPLIER * latest_atr_recalc)
                                         
                                         # Only update if the new potential target is meaningfully higher
                                         if potential_new_target_val > current_target_db * 1.001: # e.g., at least 0.1% higher
                                             old_target_val = current_target_db
                                             new_target_val = potential_new_target_val
                                             update_query_sql = sql.SQL("UPDATE signals SET current_target = %s WHERE id = %s;")
                                             update_params_tuple = (convert_np_values(new_target_val), signal_id)
                                             log_message_str = f"↗️ [Tracker] {symbol}(ID:{signal_id}): تم تحديث الهدف من {old_target_val:.8g} إلى {new_target_val:.8g} بناءً على إعادة تقييم ATR."
                                             notification_details_dict.update({'type': 'target_updated', 'old_target': old_target_val, 'new_target': new_target_val})
                                             update_executed_in_loop = True
                                         else:
                                             logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): الهدف الجديد المحتمل ({potential_new_target_val:.8g}) ليس أعلى بشكل كافٍ من الهدف الحالي ({current_target_db:.8g}).")
                                     else:
                                         logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): ATR المعاد حسابه غير صالح ({latest_atr_recalc}) لتمديد الهدف.")
                                 else:
                                     logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): فشل في إعادة حساب ATR لتمديد الهدف.")
                             else:
                                 logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): لا يمكن جلب بيانات حديثة للتحقق من تمديد الهدف.")
                    

                    # Execute DB Update if any changes were made
                    if update_executed_in_loop and update_query_sql:
                        try:
                             # Use a new cursor for the update operation
                             with conn.cursor() as update_cur_local:
                                  update_cur_local.execute(update_query_sql, update_params_tuple)
                             conn.commit() # Commit the change
                             if log_message_str: logger.info(log_message_str)
                             if notification_details_dict.get('type'): # Check if a notification type was set
                                send_tracking_notification(notification_details_dict)
                        except psycopg2.Error as db_err_update:
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ في قاعدة البيانات أثناء التحديث: {db_err_update}")
                            if conn: conn.rollback() # Rollback this specific update
                        except Exception as exec_err_update: # Catch other errors during update/notification
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ غير متوقع أثناء تنفيذ التحديث/الإشعار: {exec_err_update}", exc_info=True)
                            if conn: conn.rollback()

                except (TypeError, ValueError) as convert_err_loop: # Error converting signal_row data
                    logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ في تحويل قيم الإشارة الأولية: {convert_err_loop}")
                    continue # Skip to next signal
                except Exception as inner_loop_err_general: # Catch-all for errors within a single signal's processing
                     logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ غير متوقع أثناء معالجة الإشارة: {inner_loop_err_general}", exc_info=True)
                     continue # Skip to next signal

            if active_signals_summary: # Log summary if any signals were processed
                logger.debug(f"ℹ️ [Tracker] نهاية حالة الدورة ({processed_in_cycle} معالجة): {'; '.join(active_signals_summary)}")

            time.sleep(3) # Check every 3 seconds

        except psycopg2.Error as db_cycle_err_main: # Error in main tracking loop's DB interaction (e.g., fetching open_signals)
             logger.error(f"❌ [Tracker] خطأ في قاعدة البيانات في دورة التتبع الرئيسية: {db_cycle_err_main}. محاولة إعادة الاتصال...")
             if conn: conn.rollback() # Rollback any pending transaction
             time.sleep(30) # Wait longer before trying to re-establish DB
             # check_db_connection() will be called at the start of the next loop
        except Exception as cycle_err_main: # Catch-all for other errors in the main tracking loop
            logger.error(f"❌ [Tracker] خطأ غير متوقع في دورة تتبع الإشارة: {cycle_err_main}", exc_info=True)
            time.sleep(30) # Wait before retrying the loop

def get_interval_minutes(interval: str) -> int:
    """Helper function to convert Binance interval string to minutes."""
    try:
        if interval.endswith('m'):
            return int(interval[:-1])
        elif interval.endswith('h'):
            return int(interval[:-1]) * 60
        elif interval.endswith('d'):
            return int(interval[:-1]) * 60 * 24
        elif interval.endswith('w'):
            return int(interval[:-1]) * 60 * 24 * 7
        elif interval.endswith('M'): # Approx month, use 30 days for simplicity
            return int(interval[:-1]) * 60 * 24 * 30
    except ValueError:
        logger.error(f"⚠️ قيمة فاصل زمني غير صالحة: {interval}")
    return 0 # Default or error


# ---------------------- Flask Service (Optional for Webhook) ----------------------
app = Flask(__name__)

@app.route('/')
def home() -> Response:
    """Simple home page to show the bot is running."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Check thread aliveness safely
    ws_alive = 'ws_thread' in globals() and ws_thread is not None and ws_thread.is_alive()
    tracker_alive = 'tracker_thread' in globals() and tracker_thread is not None and tracker_thread.is_alive()
    main_bot_alive = 'main_bot_thread' in globals() and main_bot_thread is not None and main_bot_thread.is_alive()
    
    status_parts = []
    if ws_alive: status_parts.append("WS:OK")
    else: status_parts.append("WS:DOWN")
    if tracker_alive: status_parts.append("Tracker:OK")
    else: status_parts.append("Tracker:DOWN")
    if main_bot_alive: status_parts.append("BotLoop:OK")
    else: status_parts.append("BotLoop:DOWN")

    status_str = ", ".join(status_parts)
    return Response(f"📈 Crypto Signal Bot ({status_str}) - Last Check: {now}", status=200, mimetype='text/plain')

@app.route('/favicon.ico')
def favicon() -> Response:
    """Handles favicon request to avoid 404 errors in logs."""
    return Response(status=204) # No content response

@app.route('/webhook', methods=['POST'])
def webhook() -> Tuple[str, int]:
    """Handles incoming requests from Telegram (like button presses and commands)."""
    # Only process webhook if WEBHOOK_URL is configured from environment
    if not WEBHOOK_URL:
        logger.warning("⚠️ [Flask] تم استلام طلب webhook، ولكن WEBHOOK_URL غير مهيأ. تجاهل الطلب.")
        return "Webhook not configured by environment variable", 200 # Return OK to Telegram

    if not request.is_json:
        logger.warning("⚠️ [Flask] تم استلام طلب webhook غير JSON.")
        return "Invalid request format: Expected JSON", 400

    try:
        data = request.get_json()
        logger.debug(f"ℹ️ [Flask] تم استلام بيانات webhook: {json.dumps(data, ensure_ascii=False)[:200]}...") # Log first 200 chars

        if 'callback_query' in data:
            callback_query = data['callback_query']
            callback_id = callback_query['id'] # For answering callback query
            callback_data_str = callback_query.get('data')
            message_info_dict = callback_query.get('message')
            
            if not message_info_dict or not callback_data_str:
                 logger.warning(f"⚠️ [Flask] استعلام رد الاتصال (ID: {callback_id}) يفتقد الرسالة أو البيانات.")
                 # Answer callback to remove loading icon on button
                 try:
                     ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                     requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
                 except Exception as ack_err:
                     logger.warning(f"⚠️ [Flask] فشل تأكيد استعلام رد الاتصال غير الصالح {callback_id}: {ack_err}")
                 return "OK", 200 # Acknowledge Telegram
            
            chat_id_callback = message_info_dict.get('chat', {}).get('id')
            if not chat_id_callback:
                 logger.warning(f"⚠️ [Flask] استعلام رد الاتصال (ID: {callback_id}) يفتقد معرف الدردشة.")
                 try: # Answer callback
                     ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                     requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
                 except Exception as ack_err:
                     logger.warning(f"⚠️ [Flask] فشل تأكيد استعلام رد الاتصال غير الصالح {callback_id}: {ack_err}")
                 return "OK", 200
            # message_id = message_info_dict['message_id'] # If needed to edit the original message
            user_info_dict = callback_query.get('from', {})
            user_id_callback = user_info_dict.get('id')
            username_callback = user_info_dict.get('username', 'N/A')

            logger.info(f"ℹ️ [Flask] تم استلام استعلام رد الاتصال: البيانات='{callback_data_str}', المستخدم={username_callback}({user_id_callback}), الدردشة={chat_id_callback}")

            # Always answer the callback query to remove the "loading" state on the button
            try:
                ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
            except Exception as ack_err:
                 logger.warning(f"⚠️ [Flask] فشل تأكيد استعلام رد الاتصال {callback_id}: {ack_err}")

            # Handle callback data
            if callback_data_str == "get_report":
                # Run in a new thread to avoid blocking the webhook response
                report_thread_cb = Thread(target=lambda: send_telegram_message(chat_id_callback, generate_performance_report(), parse_mode='Markdown'))
                report_thread_cb.start()
            else:
                logger.warning(f"⚠️ [Flask] تم استلام بيانات رد اتصال غير معالجة: '{callback_data_str}'")


        elif 'message' in data: # Regular message
            message_data = data['message']
            chat_info_dict = message_data.get('chat')
            user_info_dict = message_data.get('from', {})
            text_msg_received = message_data.get('text', '').strip()

            if not chat_info_dict or not text_msg_received: # Ignore messages without chat info or text
                 logger.debug("ℹ️ [Flask] تم استلام رسالة بدون معلومات الدردشة أو النص.")
                 return "OK", 200

            chat_id_msg = chat_info_dict['id']
            # user_id_msg = user_info_dict.get('id')
            # username_msg = user_info_dict.get('username', 'N/A')

            # logger.info(f"ℹ️ [Flask] تم استلام رسالة: النص='{text_msg_received}', المستخدم={username_msg}({user_id_msg}), الدردشة={chat_id_msg}")

            # Handle commands
            if text_msg_received.lower() == '/report':
                 report_thread_cmd = Thread(target=lambda: send_telegram_message(chat_id_msg, generate_performance_report(), parse_mode='Markdown'))
                 report_thread_cmd.start()
            elif text_msg_received.lower() == '/status':
                 status_thread_cmd = Thread(target=handle_status_command, args=(chat_id_msg,))
                 status_thread_cmd.start()
            # Add other command handlers here if needed

        else:
            logger.debug("ℹ️ [Flask] تم استلام بيانات webhook بدون 'callback_query' أو 'message'.")

        return "OK", 200 # Acknowledge receipt to Telegram
    except Exception as e:
         logger.error(f"❌ [Flask] خطأ في معالجة webhook: {e}", exc_info=True)
         return "Internal Server Error", 500

def handle_status_command(chat_id_for_status: int) -> None:
    """Separate function to handle /status command to avoid blocking the Webhook, and allows editing."""
    logger.info(f"ℹ️ [Flask Status] معالجة أمر /status للدردشة {chat_id_for_status}")
    status_msg_initial = "⏳ جاري جلب حالة البوت..."
    msg_sent_dict = send_telegram_message(chat_id_for_status, status_msg_initial)
    
    message_id_to_edit_status: Optional[int] = None
    if msg_sent_dict and msg_sent_dict.get('ok') and msg_sent_dict.get('result'):
        message_id_to_edit_status = msg_sent_dict['result'].get('message_id')
    
    if message_id_to_edit_status is None:
        logger.error(f"❌ [Flask Status] فشل الحصول على message_id لتحديث الحالة في الدردشة {chat_id_for_status}. سيتم إرسال رسالة جديدة.")
        # Fallback to sending a new message if edit is not possible
    
    try:
        open_signals_count_status = 0
        if check_db_connection() and conn: # Ensure conn is not None
            with conn.cursor() as status_cur_local: # Use new cursor
                status_cur_local.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE;")
                result = status_cur_local.fetchone()
                open_signals_count_status = (result or {}).get('count', 0)
        
        # Check thread aliveness safely
        ws_status_str = 'نشط ✅' if 'ws_thread' in globals() and ws_thread is not None and ws_thread.is_alive() else 'غير نشط ❌'
        tracker_status_str = 'نشط ✅' if 'tracker_thread' in globals() and tracker_thread is not None and tracker_thread.is_alive() else 'غير نشط ❌'
        main_bot_status_str = 'نشط ✅' if 'main_bot_thread' in globals() and main_bot_thread is not None and main_bot_thread.is_alive() else 'غير نشط ❌'
        
        # ML Model Status
        ml_model_status_str = "محمل ✅" if ml_model else "غير محمل ⚠️"
        if ml_model:
            try: # Get model name from stored metrics if possible
                # Check if ml_model_features is populated to indicate successful loading of features
                if ml_model_features:
                    ml_model_status_str += f" (بـ {len(ml_model_features)} سمة)"
                else:
                    ml_model_status_str += f" ({ML_MODEL_NAME})"
            except: pass # Ignore errors fetching detailed model name


        final_status_msg_text = (
            f"🤖 *حالة البوت:*\n"
            f"- تتبع الأسعار (WS): {ws_status_str}\n"
            f"- تتبع الإشارات: {tracker_status_str}\n"
            f"- حلقة البوت الرئيسية: {main_bot_status_str}\n"
            f"- نموذج التعلم الآلي: {ml_model_status_str}\n"
            f"- الإشارات النشطة: *{open_signals_count_status}* / {MAX_OPEN_TRADES}\n"
            f"- وقت الخادم الحالي (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        if message_id_to_edit_status:
            edit_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText"
            edit_payload = {
                'chat_id': chat_id_for_status,
                'message_id': message_id_to_edit_status,
                'text': final_status_msg_text,
                'parse_mode': 'Markdown'
            }
            response = requests.post(edit_url, json=edit_payload, timeout=10)
            response.raise_for_status() # Check for errors during edit
            logger.info(f"✅ [Flask Status] تم تحديث الحالة للدردشة {chat_id_for_status}")
        else: # Fallback: send as new message
            send_telegram_message(chat_id_for_status, final_status_msg_text, parse_mode='Markdown')
            logger.info(f"✅ [Flask Status] تم إرسال الحالة كرسالة جديدة للدردشة {chat_id_for_status} (فشل التعديل).")


    except Exception as status_err_details:
        logger.error(f"❌ [Flask Status] خطأ في جلب/تعديل تفاصيل الحالة للدردشة {chat_id_for_status}: {status_err_details}", exc_info=True)
        # Send a simple error message if status update failed
        send_telegram_message(chat_id_for_status, "❌ حدث خطأ أثناء جلب تفاصيل الحالة.")


def run_flask() -> None:
    """Runs the Flask application to listen for the Webhook using a production server if available."""
    host = "0.0.0.0" # Listen on all available interfaces
    port = int(os.environ.get('PORT', 10000)) # Render.com sets PORT env var
    logger.info(f"ℹ️ [Flask] بدء تطبيق Flask على {host}:{port}...")
    try:
        from waitress import serve
        logger.info("✅ [Flask] استخدام خادم 'waitress' للإنتاج.")
        serve(app, host=host, port=port, threads=6) # Adjust threads as needed
    except ImportError:
         logger.warning("⚠️ [Flask] 'waitress' غير مثبت. الرجوع إلى خادم تطوير Flask (لا يوصى به للإنتاج).")
         try:
             app.run(host=host, port=port, debug=False) # debug=False for production-like behavior
         except Exception as flask_run_err:
              logger.critical(f"❌ [Flask] فشل بدء خادم التطوير: {flask_run_err}", exc_info=True)
    except Exception as serve_err: # Catch errors from serve()
         logger.critical(f"❌ [Flask] فشل بدء الخادم (waitress?): {serve_err}", exc_info=True)

# ---------------------- Main Loop and Check Function ----------------------
def main_loop() -> None:
    """Main loop to scan pairs and generate signals."""
    symbols_to_scan_list = get_crypto_symbols() # Load and validate symbols
    if not symbols_to_scan_list:
        logger.critical("❌ [Main] لا توجد رموز صالحة تم تحميلها أو التحقق منها. لا يمكن المتابعة.")
        return # Exit main_loop if no symbols

    logger.info(f"✅ [Main] تم تحميل {len(symbols_to_scan_list)} رمزًا صالحًا للمسح: {', '.join(symbols_to_scan_list[:5])}...")
    # last_full_scan_time = time.time() # Not currently used, but can be for adaptive sleep

    while True:
        try:
            scan_start_time_sec = time.time()
            logger.info("+" + "-"*70 + "+")
            logger.info(f"🔄 [Main] بدء دورة مسح السوق - {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC%z')}")
            logger.info("+" + "-"*70 + "+")

            if not check_db_connection() or not conn: # Ensure conn is not None
                logger.error("❌ [Main] تخطي دورة المسح بسبب فشل اتصال قاعدة البيانات.")
                time.sleep(60) # Wait before retrying DB connection
                continue

            # Check current number of open trades
            open_trades_count = 0
            try:
                 with conn.cursor() as cur_count_check: # Use new cursor
                    cur_count_check.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE;")
                    result = cur_count_check.fetchone()
                    open_trades_count = (result or {}).get('count', 0)
            except psycopg2.Error as db_err_count:
                 logger.error(f"❌ [Main] خطأ في قاعدة البيانات أثناء التحقق من عدد الإشارات المفتوحة: {db_err_count}. تخطي الدورة.")
                 if conn: conn.rollback()
                 time.sleep(60)
                 continue

            logger.info(f"ℹ️ [Main] الإشارات المفتوحة حالياً: {open_trades_count} / {MAX_OPEN_TRADES}")
            if open_trades_count >= MAX_OPEN_TRADES:
                logger.info(f"⚠️ [Main] تم الوصول إلى الحد الأقصى لعدد الإشارات المفتوحة ({MAX_OPEN_TRADES}). انتظار...")
                # Wait for the duration of one signal generation timeframe before checking again
                time.sleep(get_interval_minutes(SIGNAL_GENERATION_TIMEFRAME) * 60) 
                continue

            processed_symbols_in_loop = 0
            signals_generated_this_loop = 0
            slots_available_for_trades = MAX_OPEN_TRADES - open_trades_count

            for symbol_item in symbols_to_scan_list:
                 if slots_available_for_trades <= 0:
                      logger.info(f"ℹ️ [Main] تم الوصول إلى الحد الأقصى ({MAX_OPEN_TRADES}) أثناء المسح. إيقاف مسح الرموز لهذه الدورة.")
                      break # Exit symbol scanning loop

                 processed_symbols_in_loop += 1
                 logger.debug(f"🔍 [Main] مسح {symbol_item} ({processed_symbols_in_loop}/{len(symbols_to_scan_list)})...")

                 try:
                    # Check if this symbol already has an open signal
                    with conn.cursor() as symbol_open_check_cur:
                        symbol_open_check_cur.execute("SELECT 1 FROM signals WHERE symbol = %s AND achieved_target = FALSE LIMIT 1;", (symbol_item,))
                        if symbol_open_check_cur.fetchone():
                            logger.debug(f"ℹ️ [Main] {symbol_item} لديه بالفعل إشارة مفتوحة. تخطي.")
                            continue # Skip if already an open signal for this symbol

                    # Fetch historical data for this symbol
                    # Use SIGNAL_GENERATION_LOOKBACK_DAYS for populating indicators for new signals
                    # This might need to be longer if ML model was trained on more data,
                    # but populate_indicators will use what it needs from this df.
                    df_hist_current = fetch_historical_data(symbol_item, interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_hist_current is None or df_hist_current.empty:
                        logger.warning(f"⚠️ [Main] لا توجد بيانات تاريخية كافية لـ {symbol_item} في الإطار الزمني {SIGNAL_GENERATION_TIMEFRAME}. تخطي.")
                        continue

                    # Instantiate strategy and generate signal
                    strategy_instance = ScalpingTradingStrategy(symbol_item) # Pass symbol for context
                    df_indicators_populated = strategy_instance.populate_indicators(df_hist_current)
                    if df_indicators_populated is None or df_indicators_populated.empty:
                        logger.debug(f"ℹ️ [Main] لم يتمكن populate_indicators من إرجاع بيانات لـ {symbol_item}. تخطي.")
                        continue

                    potential_signal_dict = strategy_instance.generate_buy_signal(df_indicators_populated)

                    if potential_signal_dict:
                        logger.info(f"✨ [Main] تم العثور على إشارة محتملة لـ {symbol_item}! التفاصيل: {potential_signal_dict.get('signal_details', {}).get('Decision_Basis', 'N/A')}")
                        
                        # Final check on open trade count before inserting (race condition mitigation)
                        with conn.cursor() as final_count_cur:
                             final_count_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE;")
                             final_open_count_val = (final_count_cur.fetchone() or {}).get('count', 0)

                             if final_open_count_val < MAX_OPEN_TRADES:
                                 if insert_signal_into_db(potential_signal_dict):
                                     send_telegram_alert(potential_signal_dict, SIGNAL_GENERATION_TIMEFRAME)
                                     signals_generated_this_loop += 1
                                     slots_available_for_trades -= 1 # Decrement available slots
                                     time.sleep(2) # Small delay after sending alert/inserting
                                 else:
                                     logger.error(f"❌ [Main] فشل إدراج الإشارة لـ {symbol_item} في قاعدة البيانات بعد إنشائها.")
                             else:
                                 logger.warning(f"⚠️ [Main] تم الوصول إلى الحد الأقصى ({final_open_count_val}) قبل إدراج الإشارة لـ {symbol_item}. تم تجاهل الإشارة.")
                                 break # Stop scanning if max trades reached just before insert
                    
                 except psycopg2.Error as db_loop_err_sym: # DB errors specific to this symbol's processing
                      logger.error(f"❌ [Main] خطأ في قاعدة البيانات أثناء معالجة الرمز {symbol_item}: {db_loop_err_sym}. الانتقال إلى التالي...")
                      if conn: conn.rollback()
                      continue # Move to the next symbol
                 except Exception as symbol_proc_err_gen: # General errors for this symbol
                      logger.error(f"❌ [Main] خطأ عام في معالجة الرمز {symbol_item}: {symbol_proc_err_gen}", exc_info=True)
                      continue # Move to the next symbol

                 time.sleep(0.2) # Small delay between processing each symbol to be kind to API

            scan_duration_sec = time.time() - scan_start_time_sec
            logger.info(f"🏁 [Main] انتهت دورة المسح. الإشارات التي تم إنشاؤها: {signals_generated_this_loop}. مدة المسح: {scan_duration_sec:.2f} ثانية.")
            
            # Calculate wait time for the next cycle
            # Aim to start next scan roughly after one SIGNAL_GENERATION_TIMEFRAME has passed
            # Or a minimum wait time if scan was very fast
            timeframe_minutes_val = get_interval_minutes(SIGNAL_GENERATION_TIMEFRAME)
            desired_cycle_time_sec = timeframe_minutes_val * 60 if timeframe_minutes_val > 0 else 120 # Default to 2 mins if interval is 0
            wait_time_sec = max(desired_cycle_time_sec - scan_duration_sec, 60) # Ensure at least 60s wait
            
            logger.info(f"⏳ [Main] انتظار {wait_time_sec:.1f} ثانية ({wait_time_sec/60:.1f} دقيقة) للدورة التالية...")
            time.sleep(wait_time_sec)

        except KeyboardInterrupt:
             logger.info("🛑 [Main] تم طلب الإيقاف (KeyboardInterrupt). إيقاف التشغيل...")
             break # Exit the main while loop
        except psycopg2.Error as db_main_loop_err: # Catch DB errors in the outer loop
             logger.error(f"❌ [Main] خطأ فادح في قاعدة البيانات في الحلقة الرئيسية: {db_main_loop_err}. محاولة إعادة الاتصال...")
             if conn: conn.rollback()
             time.sleep(60) # Wait before trying to re-init DB
             try:
                 init_db() # Try to re-initialize DB
                 load_ml_model_from_db() # And reload model
             except Exception as recon_err_main:
                 logger.critical(f"❌ [Main] فشل إعادة الاتصال بقاعدة البيانات بعد خطأ الحلقة الرئيسية: {recon_err_main}. خروج...")
                 break # Exit if re-init fails
        except Exception as main_loop_err_gen: # Catch all other errors in the main loop
            logger.error(f"❌ [Main] خطأ غير متوقع في الحلقة الرئيسية: {main_loop_err_gen}", exc_info=True)
            logger.info("ℹ️ [Main] انتظار 120 ثانية قبل إعادة المحاولة...")
            time.sleep(120)

def cleanup_resources() -> None:
    """Closes used resources like the database connection."""
    global conn, ws_thread, tracker_thread, main_bot_thread, flask_thread # Access to thread globals
    logger.info("ℹ️ [Cleanup] إغلاق الموارد وإيقاف الخيوط...")

    # Attempt to stop WebSocket manager if it's running
    # This part is tricky as twm.stop() might need to be called from within its own control context
    # For daemon threads, they will exit when the main program exits.
    # If Flask thread is non-daemon, it keeps main alive.
    # Proper thread management for graceful shutdown can be complex.

    if conn:
        try:
            conn.close()
            logger.info("✅ [DB] تم إغلاق اتصال قاعدة البيانات.")
        except Exception as close_err:
            logger.error(f"⚠️ [DB] خطأ في إغلاق اتصال قاعدة البيانات: {close_err}")
    
    # For daemon threads, Python will handle their termination when the main thread (or last non-daemon thread) exits.
    # If flask_thread is non-daemon and joined, cleanup happens after it stops.
    logger.info("✅ [Cleanup] اكتمل تنظيف الموارد (الخيوط الخفية ستنتهي مع البرنامج).")


# ---------------------- Main Entry Point ----------------------
if __name__ == "__main__":
    logger.info("🚀 بدء بوت إشارات التداول بالتعلم الآلي...")
    logger.info(f"الوقت المحلي: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')} | وقت UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}Z")

    # Declare threads at this scope to be accessible by home route and cleanup
    ws_thread: Optional[Thread] = None
    tracker_thread: Optional[Thread] = None
    flask_thread: Optional[Thread] = None
    main_bot_thread: Optional[Thread] = None

    try:
        # 1. Initialize the database first - critical for model loading and operations
        init_db()

        # 2. Load the ML model from the database
        # This should be done after DB init and before starting main_loop or tracker that might use it.
        load_ml_model_from_db() # Populates global ml_model and ml_model_features
        if ml_model is None:
            logger.warning("⚠️ [Main Startup] لم يتم تحميل نموذج تعلم الآلة. ستعمل الاستراتيجية بوضع احتياطي إذا تم تكوينه لذلك.")
        else:
            logger.info(f"✅ [Main Startup] تم تحميل نموذج تعلم الآلة: {ML_MODEL_NAME}")


        # 3. Start WebSocket Ticker (Daemon thread)
        ws_thread = Thread(target=run_ticker_socket_manager, daemon=True, name="WebSocketThread")
        ws_thread.start()
        logger.info("✅ [Main Startup] تم بدء مؤشر أسعار WebSocket.")
        
        # Wait a bit for WebSocket to connect and populate initial ticker_data
        logger.info("ℹ️ [Main Startup] انتظار 5-10 ثوانٍ لتهيئة WebSocket واستلام بيانات أولية...")
        time.sleep(10) 
        if not ticker_data:
             logger.warning("⚠️ [Main Startup] لم يتم استلام بيانات أولية من WebSocket بعد الانتظار. قد تكون هناك مشكلة في الاتصال أو لا توجد رموز نشطة.")
        else:
             logger.info(f"✅ [Main Startup] تم استلام بيانات أولية من WebSocket لـ {len(ticker_data)} رمزًا.")


        # 4. Start Signal Tracker (Daemon thread)
        tracker_thread = Thread(target=track_signals, daemon=True, name="TrackerThread")
        tracker_thread.start()
        logger.info("✅ [Main Startup] تم بدء متتبع الإشارات.")

        # 5. Start the main bot logic in a separate thread (Daemon thread)
        main_bot_thread = Thread(target=main_loop, daemon=True, name="MainBotLoopThread")
        main_bot_thread.start()
        logger.info("✅ [Main Startup] تم بدء حلقة البوت الرئيسية في خيط منفصل.")

        # 6. Start Flask Server (Non-Daemon thread - this will keep the main program alive)
        # Ensure Flask runs on the port specified by Render (or other hosting) via PORT env var
        flask_thread = Thread(target=run_flask, daemon=False, name="FlaskThread")
        flask_thread.start()
        logger.info("✅ [Main Startup] تم بدء خادم Flask.")

        # Wait for the Flask thread to finish. Since it's non-daemon and typically runs indefinitely,
        # this means the main program will stay alive as long as Flask is running.
        flask_thread.join()
        logger.info("ℹ️ [Main] خادم Flask انتهى. بدء عملية الإغلاق...")


    except KeyboardInterrupt:
        logger.info("🛑 [Main Startup] تم طلب الإيقاف (KeyboardInterrupt) أثناء بدء التشغيل.")
    except Exception as startup_err:
        logger.critical(f"❌ [Main Startup] حدث خطأ فادح أثناء بدء التشغيل أو في الحلقة الرئيسية للخادم: {startup_err}", exc_info=True)
    finally:
        logger.info("🛑 [Main] يتم إيقاف تشغيل البرنامج...")
        cleanup_resources() # Attempt to clean up DB connection
        logger.info("👋 [Main] تم إيقاف بوت إشارات التداول.")
        # Use os._exit(0) for a more forceful exit if threads are hanging,
        # but try to let daemons terminate naturally first.
        # If Flask was the only non-daemon and it exited, daemons should also stop.
        os._exit(0) # Force exit if other threads are stuck, especially daemon ones.
