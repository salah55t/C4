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
from threading import Thread, Lock
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
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
except Exception as e:
     logger.critical(f"❌ فشل في تحميل المتغيرات البيئية الأساسية: {e}")
     exit(1)

logger.info(f"Binance API Key: {'Available' if API_KEY else 'Not available'}")
logger.info(f"Telegram Token: {'Available' if TELEGRAM_TOKEN else 'Not available'}")
logger.info(f"Chat ID: {'Available' if CHAT_ID else 'Not available'}")
logger.info(f"Database URL: {'Available' if DB_URL else 'Not available'}")
logger.info(f"Webhook URL: {WEBHOOK_URL if WEBHOOK_URL else 'Not specified'} (Flask will always run for Render)")


# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
SIGNAL_GENERATION_TIMEFRAME: str = '5m' # الإطار الزمني لتوليد الإشارة
DATA_LOOKBACK_DAYS: int = 90 # عدد الأيام لجلب البيانات التاريخية للمؤشرات
MIN_PROFIT_MARGIN_PCT: float = 0.005 # 0.5%
TARGET_APPROACH_THRESHOLD_PCT: float = 0.005 # 0.5%
MAX_OPEN_SIGNALS_PER_SYMBOL: int = 1 # الحد الأقصى للإشارات المفتوحة لكل رمز
MIN_VOLUME_15M_USDT: float = 50000 # الحد الأدنى لمتوسط حجم التداول لآخر 15 دقيقة (بالدولار الأمريكي)
ML_MODEL_NAME_PREFIX: str = 'DecisionTree_Scalping_V1_' # بادئة لاسم نموذج ML

# Indicator Parameters (يجب أن تتطابق مع ml.py)
RSI_PERIOD: int = 9
RSI_OVERSOLD: int = 30
RSI_OVERBOUGHT: int = 70
EMA_SHORT_PERIOD: int = 8
EMA_LONG_PERIOD: int = 21
VWMA_PERIOD: int = 15
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
VOLUME_LOOKBACK_CANDLES: int = 3 # عدد الشمعات لحساب متوسط الحجم (3 شمعات * 5 دقائق = 15 دقيقة)
RSI_MOMENTUM_LOOKBACK_CANDLES: int = 2 # عدد الشمعات للتحقق من تزايد RSI للزخم

# Constants related to strategy conditions
RECENT_EMA_CROSS_LOOKBACK: int = 2
MIN_ADX_TREND_STRENGTH: int = 20
MACD_HIST_INCREASE_CANDLES: int = 3
OBV_INCREASE_CANDLES: int = 3


# Global variables
conn: Optional[psycopg2.extensions.connection] = None
cur: Optional[psycopg2.extensions.cursor] = None
client: Optional[Client] = None
twm: Optional[ThreadedWebsocketManager] = None
ticker_data: Dict[str, Dict[str, Any]] = {} # لتخزين أحدث بيانات التيكر (السعر، الحجم)
all_symbols: List[str] = [] # قائمة الرموز التي يتم تداولها
open_signals: List[Dict[str, Any]] = [] # قائمة بالإشارات المفتوحة
open_signals_lock = Lock() # قفل لتأمين الوصول إلى open_signals
db_lock = Lock() # قفل لتأمين الوصول إلى قاعدة البيانات

# Global dictionary to store partial candle data for each symbol
# { 'SYMBOL': { 'open': price, 'high': price, 'low': price, 'close': price, 'volume': vol, 'start_time': timestamp_ms, 'last_update_time': timestamp_ms } }
current_5m_candles: Dict[str, Dict[str, Any]] = {}
# Global dictionary to store strategy instances for each symbol
symbol_strategies: Dict[str, 'ScalpingTradingStrategy'] = {}


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

# ---------------------- Database Connection Setup ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    """Initializes database connection and creates tables if they don't exist."""
    global conn, cur
    logger.info("[DB] بدء تهيئة قاعدة البيانات...")
    for attempt in range(retries):
        try:
            logger.info(f"[DB] محاولة الاتصال بقاعدة البيانات (المحاولة {attempt + 1}/{retries})...")
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            cur = conn.cursor()
            logger.info("✅ [DB] تم الاتصال بقاعدة البيانات بنجاح.")

            # --- Create or update signals table ---
            logger.info("[DB] التحقق من/إنشاء جدول 'signals'...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    initial_target DOUBLE PRECISION NOT NULL,
                    current_target DOUBLE PRECISION NOT NULL,
                    r2_score DOUBLE PRECISION,
                    volume_15m DOUBLE PRECISION,
                    achieved_target BOOLEAN DEFAULT FALSE,
                    closing_price DOUBLE PRECISION,
                    closed_at TIMESTAMP,
                    sent_at TIMESTAMP DEFAULT NOW(),
                    entry_time TIMESTAMP DEFAULT NOW(),
                    time_to_target INTERVAL,
                    profit_percentage DOUBLE PRECISION,
                    strategy_name TEXT,
                    signal_details JSONB
                );""")
            conn.commit()
            logger.info("✅ [DB] جدول 'signals' موجود أو تم إنشاؤه.")

            # --- Create ml_models table (for loading trained models) ---
            logger.info("[DB] التحقق من/إنشاء جدول 'ml_models'...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ml_models (
                    id SERIAL PRIMARY KEY,
                    model_name TEXT NOT NULL UNIQUE,
                    model_data BYTEA NOT NULL,
                    trained_at TIMESTAMP DEFAULT NOW(),
                    metrics JSONB
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
            if conn: conn.rollback()
            if attempt == retries - 1:
                 logger.critical("❌ [DB] فشلت جميع محاولات الاتصال بقاعدة البيانات.")
                 raise op_err
            time.sleep(delay)
        except Exception as e:
            logger.critical(f"❌ [DB] فشل غير متوقع في تهيئة قاعدة البيانات (المحاولة {attempt + 1}): {e}", exc_info=True)
            if conn: conn.rollback()
            if attempt == retries - 1:
                 logger.critical("❌ [DB] فشلت جميع محاولات الاتصال بقاعدة البيانات.")
                 raise e
            time.sleep(delay)

    logger.critical("❌ [DB] فشل الاتصال بقاعدة البيانات بعد عدة محاولات.")
    exit(1)


def check_db_connection() -> bool:
    """Checks database connection status and re-initializes if necessary."""
    global conn, cur
    try:
        if conn is None or conn.closed != 0:
            logger.warning("⚠️ [DB] الاتصال مغلق أو غير موجود. إعادة التهيئة...")
            init_db()
            return True
        else:
             with conn.cursor() as check_cur:
                  check_cur.execute("SELECT 1;")
                  check_cur.fetchone()
             return True
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [DB] فقدان الاتصال بقاعدة البيانات ({e}). إعادة التهيئة...")
        try:
             init_db()
             return True
        except Exception as recon_err:
            logger.error(f"❌ [DB] فشلت محاولة إعادة الاتصال بعد فقدان الاتصال: {recon_err}")
            return False
    except Exception as e:
        logger.error(f"❌ [DB] خطأ غير متوقع أثناء التحقق من الاتصال: {e}", exc_info=True)
        try:
            init_db()
            return True
        except Exception as recon_err:
             logger.error(f"❌ [DB] فشلت محاولة إعادة الاتصال بعد خطأ غير متوقع: {recon_err}")
             return False

def get_crypto_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """
    Reads the list of currency symbols from a text file, then validates them
    as valid USDT pairs available for Spot trading on Binance.
    """
    raw_symbols: List[str] = []
    logger.info(f"ℹ️ [Data] قراءة قائمة الرموز من الملف '{filename}'...")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)

        if not os.path.exists(file_path):
            file_path = os.path.abspath(filename)
            if not os.path.exists(file_path):
                 logger.error(f"❌ [Data] الملف '{filename}' غير موجود في دليل السكربت أو الدليل الحالي.")
                 return []
            else:
                 logger.warning(f"⚠️ [Data] الملف '{filename}' غير موجود في دليل السكربت. استخدام الملف في الدليل الحالي: '{file_path}'")

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = [f"{line.strip().upper().replace('USDT', '')}USDT"
                           for line in f if line.strip() and not line.startswith('#')]
        raw_symbols = sorted(list(set(raw_symbols)))
        logger.info(f"ℹ️ [Data] تم قراءة {len(raw_symbols)} رمزًا مبدئيًا من '{file_path}'.")

    except FileNotFoundError:
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
        return raw_symbols

    try:
        logger.info("ℹ️ [Data Validation] التحقق من الرموز وحالة التداول من Binance API...")
        exchange_info = client.get_exchange_info()
        valid_trading_usdt_symbols = {
            s['symbol'] for s in exchange_info['symbols']
            if s.get('quoteAsset') == 'USDT' and
               s.get('status') == 'TRADING' and
               s.get('isSpotTradingAllowed') is True
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
         return raw_symbols
    except Exception as api_err:
         logger.error(f"❌ [Data Validation] خطأ غير متوقع أثناء التحقق من رموز Binance: {api_err}", exc_info=True)
         logger.warning("⚠️ [Data Validation] استخدام القائمة الأولية من الملف بدون التحقق من Binance.")
         return raw_symbols


def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """
    Fetches historical candlestick data from Binance for a specified number of days.
    This function relies on python-binance's get_historical_klines to handle
    internal pagination for large data ranges.
    """
    if not client:
        logger.error("❌ [Data] عميل Binance غير مهيأ لجلب البيانات.")
        return None
    try:
        # Calculate the start date for the entire data range needed
        start_dt = datetime.utcnow() - timedelta(days=days + 1)
        start_str_overall = start_dt.strftime("%Y-%m-%d %H:%M:%S")

        logger.debug(f"ℹ️ [Data] جلب بيانات {interval} لـ {symbol} من {start_str_overall} حتى الآن...")

        klines = client.get_historical_klines(symbol, interval, start_str_overall)

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
        df = df[numeric_cols]
        initial_len = len(df)
        df.dropna(subset=numeric_cols, inplace=True)

        if len(df) < initial_len:
            logger.debug(f"ℹ️ [Data] {symbol}: تم إسقاط {initial_len - len(df)} صفًا بسبب قيم NaN في بيانات OHLCV.")

        if df.empty:
            logger.warning(f"⚠️ [Data] DataFrame لـ {symbol} فارغ بعد إزالة قيم NaN الأساسية.")
            return None

        df.sort_index(inplace=True)

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

def get_ml_model_from_db(model_name: str) -> Optional[Any]:
    """
    Loads a trained ML model from the database.
    """
    if not check_db_connection() or not conn:
        logger.error("❌ [DB Load] لا يمكن تحميل نموذج ML بسبب مشكلة في اتصال قاعدة البيانات.")
        return None

    logger.info(f"ℹ️ [DB Load] محاولة تحميل نموذج ML '{model_name}' من قاعدة البيانات...")
    try:
        with db_lock: # Use lock for DB access
            with conn.cursor() as db_cur:
                db_cur.execute("SELECT model_data FROM ml_models WHERE model_name = %s;", (model_name,))
                result = db_cur.fetchone()
                if result and result['model_data']:
                    model = pickle.loads(result['model_data'])
                    logger.info(f"✅ [DB Load] تم تحميل نموذج ML '{model_name}' بنجاح.")
                    return model
                else:
                    logger.warning(f"⚠️ [DB Load] لم يتم العثور على نموذج ML باسم '{model_name}'.")
                    return None
    except psycopg2.Error as db_err:
        logger.error(f"❌ [DB Load] خطأ في قاعدة البيانات أثناء تحميل نموذج ML: {db_err}", exc_info=True)
        return None
    except pickle.UnpicklingError as pickle_err:
        logger.error(f"❌ [DB Load] خطأ في فك تسلسل نموذج ML: {pickle_err}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"❌ [DB Load] خطأ غير متوقع أثناء تحميل نموذج ML: {e}", exc_info=True)
        return None


# ---------------------- Technical Indicator Functions ----------------------

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
        logger.warning("⚠️ [Indicator VWMA] أعمدة 'close' أو 'volume' مفقودة أو فارغة.")
        return pd.Series(index=df_calc.index if df_calc is not None else None, dtype=float)
    if len(df_calc) < period:
        logger.warning(f"⚠️ [Indicator VWMA] بيانات غير كافية ({len(df_calc)} < {period}) لحساب VWMA.")
        return pd.Series(index=df_calc.index if df_calc is not None else None, dtype=float)

    df_calc['price_volume'] = df_calc['close'] * df_calc['volume']
    rolling_price_volume_sum = df_calc['price_volume'].rolling(window=period, min_periods=period).sum()
    rolling_volume_sum = df_calc['volume'].rolling(window=period, min_periods=period).sum()
    vwma = rolling_price_volume_sum / rolling_volume_sum.replace(0, np.nan)
    df_calc.drop(columns=['price_volume'], inplace=True, errors='ignore')
    return vwma

def calculate_rsi_indicator(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """Calculates Relative Strength Index (RSI)."""
    df = df.copy()
    if 'close' not in df.columns or df['close'].isnull().all():
        logger.warning("⚠️ [Indicator RSI] عمود 'close' مفقود أو فارغ.")
        df['rsi'] = np.nan
        return df
    if len(df) < period:
        logger.warning(f"⚠️ [Indicator RSI] بيانات غير كافية ({len(df)} < {period}) لحساب RSI.")
        df['rsi'] = np.nan
        return df

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi_series = 100 - (100 / (1 + rs))
    df['rsi'] = rsi_series.ffill().fillna(50)

    return df

def calculate_atr_indicator(df: pd.DataFrame, period: int = ENTRY_ATR_PERIOD) -> pd.DataFrame:
    """Calculates Average True Range (ATR)."""
    df = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator ATR] أعمدة 'high', 'low', 'close' مفقودة أو فارغة.")
        df['atr'] = np.nan
        return df
    if len(df) < period + 1:
        logger.warning(f"⚠️ [Indicator ATR] بيانات غير كافية ({len(df)} < {period + 1}) لحساب ATR.")
        df['atr'] = np.nan
        return df

    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()

    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)

    df['atr'] = tr.ewm(span=period, adjust=False).mean()
    return df


def calculate_bollinger_bands(df: pd.DataFrame, window: int = BOLLINGER_WINDOW, num_std: int = BOLLINGER_STD_DEV) -> pd.DataFrame:
    """Calculates Bollinger Bands."""
    df = df.copy()
    if 'close' not in df.columns or df['close'].isnull().all():
        logger.warning("⚠️ [Indicator BB] عمود 'close' مفقود أو فارغ.")
        df['bb_middle'] = np.nan
        df['bb_upper'] = np.nan
        df['bb_lower'] = np.nan
        return df
    if len(df) < window:
         logger.warning(f"⚠️ [Indicator BB] بيانات غير كافية ({len(df)} < {window}) لحساب BB.)")
         df['bb_middle'] = np.nan
         df['bb_upper'] = np.nan
         df['bb_lower'] = np.nan
         return df

    df['bb_middle'] = df['close'].rolling(window=window).mean()
    df['bb_std'] = df['close'].rolling(window=window).std()
    df['bb_upper'] = df['bb_middle'] + num_std * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - num_std * df['bb_std']
    return df


def calculate_macd(df: pd.DataFrame, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> pd.DataFrame:
    """Calculates MACD, Signal Line, and Histogram."""
    df = df.copy()
    if 'close' not in df.columns or df['close'].isnull().all():
        logger.warning("⚠️ [Indicator MACD] عمود 'close' مفقود أو فارغ.")
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['macd_hist'] = np.nan
        return df
    min_len = max(fast, slow, signal)
    if len(df) < min_len:
        logger.warning(f"⚠️ [Indicator MACD] بيانات غير كافية ({len(df)} < {min_len}) لحساب MACD.")
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['macd_hist'] = np.nan
        return df

    ema_fast = calculate_ema(df['close'], fast)
    ema_slow = calculate_ema(df['close'], slow)
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = calculate_ema(df['macd'], signal)
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df


def calculate_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    """Calculates ADX, DI+ and DI-."""
    df_calc = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df_calc.columns for col in required_cols) or df_calc[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator ADX] أعمدة 'high', 'low', 'close' مفقودة أو فارغة.")
        df_calc['adx'] = np.nan
        df_calc['di_plus'] = np.nan
        df_calc['di_minus'] = np.nan
        return df_calc
    if len(df_calc) < period * 2: # ADX requires more data than simple EMAs
        logger.warning(f"⚠️ [Indicator ADX] بيانات غير كافية ({len(df_calc)} < {period * 2}) لحساب ADX.")
        df_calc['adx'] = np.nan
        df_calc['di_plus'] = np.nan
        df_calc['di_minus'] = np.nan
        return df_calc

    df_calc['high-low'] = df_calc['high'] - df_calc['low']
    df_calc['high-prev_close'] = abs(df_calc['high'] - df_calc['close'].shift(1))
    df_calc['low-prev_close'] = abs(df_calc['low'] - df_calc['close'].shift(1))
    df_calc['tr'] = df_calc[['high-low', 'high-prev_close', 'low-prev_close']].max(axis=1, skipna=False)

    df_calc['up_move'] = df_calc['high'] - df_calc['high'].shift(1)
    df_calc['down_move'] = df_calc['low'].shift(1) - df_calc['low']
    df_calc['+dm'] = np.where((df_calc['up_move'] > df_calc['down_move']) & (df_calc['up_move'] > 0), df_calc['up_move'], 0)
    df_calc['-dm'] = np.where((df_calc['down_move'] > df_calc['up_move']) & (df_calc['down_move'] > 0), df_calc['down_move'], 0)

    alpha = 1 / period
    df_calc['tr_smooth'] = df_calc['tr'].ewm(alpha=alpha, adjust=False).mean()
    df_calc['+dm_smooth'] = df_calc['+dm'].ewm(alpha=alpha, adjust=False).mean()
    df_calc['di_minus_smooth'] = df_calc['-dm'].ewm(alpha=alpha, adjust=False).mean()

    df_calc['di_plus'] = np.where(df_calc['tr_smooth'] > 0, 100 * (df_calc['+dm_smooth'] / df_calc['tr_smooth']), 0)
    df_calc['di_minus'] = np.where(df_calc['tr_smooth'] > 0, 100 * (df_calc['di_minus_smooth'] / df_calc['tr_smooth']), 0)

    di_sum = df_calc['di_plus'] + df_calc['di_minus']
    df_calc['dx'] = np.where(di_sum > 0, 100 * abs(df_calc['di_plus'] - df_calc['di_minus']) / di_sum, 0)

    df_calc['adx'] = df_calc['dx'].ewm(alpha=alpha, adjust=False).mean()

    return df_calc[['adx', 'di_plus', 'di_minus']]


def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Volume Weighted Average Price (VWAP) - Resets daily."""
    df = df.copy()
    required_cols = ['high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator VWAP] أعمدة 'high', 'low', 'close' أو 'volume' مفقودة أو فارغة.")
        df['vwap'] = np.nan
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
            logger.warning("⚠️ [Indicator VWAP] تم تحويل الفهرس إلى DatetimeIndex.")
        except Exception:
            logger.error("❌ [Indicator VWAP] فشل تحويل الفهرس إلى DatetimeIndex، لا يمكن حساب VWAP اليومي.")
            df['vwap'] = np.nan
            return df
    if df.index.tz is not None:
        df.index = df.index.tz_convert('UTC')
        logger.debug("ℹ️ [Indicator VWAP] تم تحويل الفهرس إلى UTC لإعادة الضبط اليومي.")
    else:
        df.index = df.index.tz_localize('UTC')
        logger.debug("ℹ️ [Indicator VWAP] تم توطين الفهرس إلى UTC لإعادة الضبط اليومي.")


    df['date'] = df.index.date
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol'] = df['typical_price'] * df['volume']

    try:
        df['cum_tp_vol'] = df.groupby('date')['tp_vol'].cumsum()
        df['cum_volume'] = df.groupby('date')['volume'].cumsum()
    except KeyError as e:
        logger.error(f"❌ [Indicator VWAP] خطأ في تجميع البيانات حسب التاريخ: {e}. قد يكون الفهرس غير صحيح.")
        df['vwap'] = np.nan
        df.drop(columns=['date', 'typical_price', 'tp_vol', 'cum_tp_vol', 'cum_volume'], inplace=True, errors='ignore')
        return df
    except Exception as e:
         logger.error(f"❌ [Indicator VWAP] خطأ غير متوقع في حساب VWAP: {e}", exc_info=True)
         df['vwap'] = np.nan
         df.drop(columns=['date', 'typical_price', 'tp_vol', 'cum_tp_vol', 'cum_volume'], inplace=True, errors='ignore')
         return df


    df['vwap'] = np.where(df['cum_volume'] > 0, df['cum_tp_vol'] / df['cum_volume'], np.nan)

    df['vwap'] = df['vwap'].bfill()

    df.drop(columns=['date', 'typical_price', 'tp_vol', 'cum_tp_vol', 'cum_volume'], inplace=True, errors='ignore')
    return df


def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates On-Balance Volume (OBV)."""
    df = df.copy()
    required_cols = ['close', 'volume']
    if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator OBV] أعمدة 'close' أو 'volume' مفقودة أو فارغة.")
        df['obv'] = np.nan
        return df
    if not pd.api.types.is_numeric_dtype(df['close']) or not pd.api.types.is_numeric_dtype(df['volume']):
        logger.warning("⚠️ [Indicator OBV] أعمدة 'close' أو 'volume' ليست رقمية.")
        df['obv'] = np.nan
        return df

    obv = np.zeros(len(df), dtype=np.float64)
    close = df['close'].values
    volume = df['volume'].values

    close_diff = df['close'].diff().values

    for i in range(1, len(df)):
        if np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(close_diff[i]):
            obv[i] = obv[i-1]
            continue

        if close_diff[i] > 0:
            obv[i] = obv[i-1] + volume[i]
        elif close_diff[i] < 0:
             obv[i] = obv[i-1] - volume[i]
        else:
             obv[i] = obv[i-1]

    df['obv'] = obv
    return df


def calculate_supertrend(df: pd.DataFrame, period: int = SUPERTREND_PERIOD, multiplier: float = SUPERTREND_MULTIPLIER) -> pd.DataFrame:
    """Calculates the SuperTrend indicator."""
    df_st = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df_st.columns for col in required_cols) or df_st[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator SuperTrend] أعمدة 'high', 'low', 'close' مفقودة أو فارغة.")
        df_st['supertrend'] = np.nan
        df_st['supertrend_trend'] = 0
        return df_st

    df_st = calculate_atr_indicator(df_st, period=SUPERTREND_PERIOD)


    if 'atr' not in df_st.columns or df_st['atr'].isnull().all():
         logger.warning("⚠️ [Indicator SuperTrend] لا يمكن حساب SuperTrend بسبب قيم ATR غير صالحة أو مفقودة.")
         df_st['supertrend'] = np.nan
         df_st['supertrend_trend'] = 0
         return df_st
    if len(df_st) < SUPERTREND_PERIOD:
        logger.warning(f"⚠️ [Indicator SuperTrend] بيانات غير كافية ({len(df_st)} < {SUPERTREND_PERIOD}) لحساب SuperTrend.")
        df_st['supertrend'] = np.nan
        df_st['supertrend_trend'] = 0
        return df_st

    hl2 = (df_st['high'] + df_st['low']) / 2
    df_st['basic_ub'] = hl2 + multiplier * df_st['atr']
    df_st['basic_lb'] = hl2 - multiplier * df_st['atr']

    df_st['final_ub'] = 0.0
    df_st['final_lb'] = 0.0
    df_st['supertrend'] = np.nan
    df_st['supertrend_trend'] = 0

    close = df_st['close'].values
    basic_ub = df_st['basic_ub'].values
    basic_lb = df_st['basic_lb'].values
    final_ub = df_st['final_ub'].values
    final_lb = df_st['final_lb'].values
    st = df_st['supertrend'].values
    st_trend = df_st['supertrend_trend'].values

    for i in range(1, len(df_st)):
        if pd.isna(basic_ub[i]) or pd.isna(basic_lb[i]) or pd.isna(close[i]):
            final_ub[i] = final_ub[i-1]
            final_lb[i] = final_lb[i-1]
            st[i] = st[i-1]
            st_trend[i] = st_trend[i-1]
            continue

        if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]

        if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]

        if st_trend[i-1] == -1:
            if close[i] <= final_ub[i]:
                st[i] = final_ub[i]
                st_trend[i] = -1
            else:
                st[i] = final_lb[i]
                st_trend[i] = 1
        elif st_trend[i-1] == 1:
            if close[i] >= final_lb[i]:
                st[i] = final_lb[i]
                st_trend[i] = 1
            else:
                st[i] = final_ub[i]
                st_trend[i] = -1
        else:
             if close[i] > final_ub[i]:
                 st[i] = final_lb[i]
                 st_trend[i] = 1
             elif close[i] < final_ub[i]:
                  st[i] = final_ub[i]
                  st_trend[i] = -1
             else:
                  st[i] = np.nan
                  st_trend[i] = 0


    df_st['final_ub'] = final_ub
    df_st['final_lb'] = final_lb
    df_st['supertrend'] = st
    df_st['supertrend_trend'] = st_trend

    df_st.drop(columns=['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, errors='ignore')

    return df_st


# ---------------------- Candlestick Patterns ----------------------

def is_hammer(row: pd.Series) -> int:
    """Checks for Hammer pattern (bullish signal)."""
    o, h, l, c = row.get('open'), row.get('high'), row.get('low'), row.get('close')
    if pd.isna([o, h, l, c]).any(): return 0
    body = abs(c - o)
    candle_range = h - l
    if candle_range == 0: return 0
    lower_shadow = min(o, c) - l
    upper_shadow = h - max(o, c)
    is_small_body = body < (candle_range * 0.35)
    is_long_lower_shadow = lower_shadow >= 1.8 * body if body > 0 else lower_shadow > candle_range * 0.6
    is_small_upper_shadow = upper_shadow <= body * 0.6 if body > 0 else upper_shadow < candle_range * 0.15
    return 100 if is_small_body and is_long_lower_shadow and is_small_upper_shadow else 0

def is_shooting_star(row: pd.Series) -> int:
    """Checks for Shooting Star pattern (bearish signal)."""
    o, h, l, c = row.get('open'), row.get('high'), row.get('low'), row.get('close')
    if pd.isna([o, h, l, c]).any(): return 0
    body = abs(c - o)
    candle_range = h - l
    if candle_range == 0: return 0
    lower_shadow = min(o, c) - l
    upper_shadow = h - max(o, c)
    is_small_body = body < (candle_range * 0.35)
    is_long_upper_shadow = upper_shadow >= 1.8 * body if body > 0 else upper_shadow > candle_range * 0.6
    is_small_lower_shadow = lower_shadow <= body * 0.6 if body > 0 else upper_shadow < candle_range * 0.15
    return -100 if is_small_body and is_long_upper_shadow and is_small_lower_shadow else 0

def is_doji(row: pd.Series) -> int:
    """Checks for Doji pattern (uncertainty)."""
    o, h, l, c = row.get('open'), row.get('high'), row.get('low'), row.get('close')
    if pd.isna([o, h, l, c]).any(): return 0
    candle_range = h - l
    if candle_range == 0: return 0
    return 100 if abs(c - o) <= (candle_range * 0.1) else 0

def compute_engulfing(df: pd.DataFrame, idx: int) -> int:
    """Checks for Bullish or Bearish Engulfing pattern."""
    if idx == 0: return 0
    prev = df.iloc[idx - 1]
    curr = df.iloc[idx]
    if pd.isna([prev['close'], prev['open'], curr['close'], curr['open']]).any():
        return 0
    if abs(prev['close'] - prev['open']) < (prev['high'] - prev['low']) * 0.1: # Filter out very small previous candles
        return 0

    is_bullish = (prev['close'] < prev['open'] and curr['close'] > curr['open'] and
                  curr['open'] <= prev['close'] and curr['close'] >= prev['open'])
    is_bearish = (prev['close'] > prev['open'] and curr['close'] < curr['open'] and
                  curr['open'] >= prev['close'] and curr['close'] <= prev['open'])

    if is_bullish: return 100
    if is_bearish: return -100
    return 0

def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds candlestick pattern signals to the DataFrame."""
    df = df.copy()
    logger.debug("ℹ️ [Indicators] كشف أنماط الشموع...")
    df['Hammer'] = df.apply(is_hammer, axis=1)
    df['ShootingStar'] = df.apply(is_shooting_star, axis=1)
    df['Doji'] = df.apply(is_doji, axis=1)
    engulfing_values = [compute_engulfing(df, i) for i in range(len(df))]
    df['Engulfing'] = engulfing_values
    df['BullishCandleSignal'] = df.apply(lambda row: 1 if (row['Hammer'] == 100 or row['Engulfing'] == 100) else 0, axis=1)
    df['BearishCandleSignal'] = df.apply(lambda row: 1 if (row['ShootingStar'] == -100 or row['Engulfing'] == -100) else 0, axis=1)
    logger.debug("✅ [Indicators] تم كشف أنماط الشموع.")
    return df


class ScalpingTradingStrategy:
    """
    Implements a scalping trading strategy using various technical indicators
    and an ML model for signal generation.
    """
    def __init__(self, symbol: str, client: Client, conn: Any, cur: Any):
        self.symbol = symbol
        self.client = client
        self.conn = conn
        self.cur = cur
        self.ml_model: Optional[Any] = None
        self.feature_names: List[str] = []
        self._load_ml_model()

    def _load_ml_model(self) -> None:
        """Loads the pre-trained ML model for the specific symbol."""
        model_name = f"{ML_MODEL_NAME_PREFIX}{self.symbol}"
        model_data = get_ml_model_from_db(model_name)
        if model_data:
            self.ml_model = model_data
            # Assuming the model's metrics or an attribute stores feature names
            # This is crucial for consistent feature order during prediction
            if hasattr(self.ml_model, 'feature_names_in_'):
                self.feature_names = self.ml_model.feature_names_in_.tolist()
            elif hasattr(self.ml_model, 'metrics') and 'feature_names' in self.ml_model.metrics:
                 self.feature_names = self.ml_model.metrics['feature_names']
            else:
                # Fallback to a predefined list if not found in model (less robust)
                self.feature_names = [
                    'atr', 'supertrend', 'supertrend_trend',
                    f'ema_{EMA_SHORT_PERIOD}', f'ema_{EMA_LONG_PERIOD}', 'vwma', 'rsi',
                    'bb_middle', 'bb_upper', 'bb_lower',
                    'macd', 'macd_signal', 'macd_hist',
                    'adx', 'di_plus', 'di_minus', 'vwap', 'obv',
                    'Hammer', 'ShootingStar', 'Doji', 'Engulfing',
                    'BullishCandleSignal', 'BearishCandleSignal',
                    'volume_15m_avg', 'rsi_momentum_bullish'
                ]
            logger.info(f"✅ [Strategy] تم تحميل نموذج ML لـ {self.symbol} بـ {len(self.feature_names)} ميزة.")
        else:
            logger.warning(f"⚠️ [Strategy] لم يتم العثور على نموذج ML لـ {self.symbol}. ستعتمد الإشارات على المؤشرات فقط.")

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates all necessary technical indicators for the DataFrame."""
        if df.empty:
            logger.warning(f"⚠️ [Indicators] DataFrame فارغ لـ {self.symbol}. لا يمكن حساب المؤشرات.")
            return df

        df_copy = df.copy() # Work on a copy

        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

        # Calculate all indicators
        df_copy = calculate_atr_indicator(df_copy, ENTRY_ATR_PERIOD)
        df_copy = calculate_supertrend(df_copy, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
        df_copy[f'ema_{EMA_SHORT_PERIOD}'] = calculate_ema(df_copy['close'], EMA_SHORT_PERIOD)
        df_copy[f'ema_{EMA_LONG_PERIOD}'] = calculate_ema(df_copy['close'], EMA_LONG_PERIOD)
        df_copy['vwma'] = calculate_vwma(df_copy, VWMA_PERIOD)
        df_copy = calculate_rsi_indicator(df_copy, RSI_PERIOD)
        df_copy = calculate_bollinger_bands(df_copy, BOLLINGER_WINDOW, BOLLINGER_STD_DEV)
        df_copy = calculate_macd(df_copy, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        adx_df = calculate_adx(df_copy, ADX_PERIOD)
        df_copy = df_copy.join(adx_df)
        df_copy = calculate_vwap(df_copy)
        df_copy = calculate_obv(df_copy)
        df_copy = detect_candlestick_patterns(df_copy)

        # Add volume_15m_avg (average volume over last 3 5-min candles)
        df_copy['volume_15m_avg'] = df_copy['volume'].rolling(window=VOLUME_LOOKBACK_CANDLES, min_periods=1).mean()

        # Add RSI momentum bullish
        df_copy['rsi_momentum_bullish'] = 0
        if len(df_copy) >= RSI_MOMENTUM_LOOKBACK_CANDLES + 1:
            for i in range(RSI_MOMENTUM_LOOKBACK_CANDLES, len(df_copy)):
                rsi_slice = df_copy['rsi'].iloc[i - RSI_MOMENTUM_LOOKBACK_CANDLES : i + 1]
                if not rsi_slice.isnull().any() and np.all(np.diff(rsi_slice) > 0) and rsi_slice.iloc[-1] > 50:
                    df_copy.loc[df_copy.index[i], 'rsi_momentum_bullish'] = 1

        logger.debug(f"✅ [Indicators] تم حساب جميع المؤشرات لـ {self.symbol}. حجم DataFrame: {len(df_copy)}")
        return df_copy

    def generate_buy_signal(self, latest_candle_data: pd.Series, current_real_time_price: float) -> Optional[Dict[str, Any]]:
        """
        Generates a buy signal based on technical indicators and ML model prediction.
        Uses the current real-time price for entry.
        """
        if latest_candle_data.empty:
            logger.warning(f"⚠️ [Signal Gen] بيانات الشمعة الأخيرة فارغة لـ {self.symbol}. لا يمكن توليد إشارة.")
            return None

        # Check for open signals for this symbol
        with open_signals_lock:
            open_signals_for_symbol = [s for s in open_signals if s['symbol'] == self.symbol and not s['achieved_target']]
            if len(open_signals_for_symbol) >= MAX_OPEN_SIGNALS_PER_SYMBOL:
                logger.info(f"ℹ️ [Signal Gen] يوجد بالفعل {len(open_signals_for_symbol)} إشارة مفتوحة لـ {self.symbol}. تخطي توليد إشارة جديدة.")
                return None

        # Prepare features for ML prediction
        ml_features = {}
        for feature in self.feature_names:
            ml_features[feature] = latest_candle_data.get(feature)
            if pd.isna(ml_features[feature]):
                ml_features[feature] = 0.0 # Handle NaN features for ML model gracefully

        features_df = pd.DataFrame([ml_features])

        ml_prediction = 0
        if self.ml_model:
            try:
                # Ensure feature order matches training
                features_for_prediction = features_df[self.feature_names].values
                ml_prediction = self.ml_model.predict(features_for_prediction)[0]
                logger.debug(f"ℹ️ [Signal Gen] تنبؤ ML لـ {self.symbol}: {ml_prediction}")
            except Exception as e:
                logger.error(f"❌ [Signal Gen] خطأ في تنبؤ نموذج ML لـ {self.symbol}: {e}", exc_info=True)
                ml_prediction = 0 # Fallback to no ML signal

        # BTC Trend Filter (assuming BTC data is available in ticker_data)
        btc_price = ticker_data.get('BTCUSDT', {}).get('last_price')
        if btc_price:
            # For simplicity, we'll just check if BTC is generally trending up.
            # In a real scenario, you'd calculate BTC's EMA cross or SuperTrend.
            # For now, a placeholder: if BTC is below a recent high, consider it not bullish enough.
            # This needs actual BTC historical data and indicator calculation.
            # For this example, let's assume a simple check for now, or remove if not critical.
            # A more robust solution would involve fetching BTC candles and running indicators.
            # For now, let's assume BTC trend is generally bullish or ignore this filter if no complex BTC logic is implemented.
            pass # Placeholder, implement robust BTC trend check if needed.

        # Essential Conditions
        is_ema_short_above_long = latest_candle_data.get(f'ema_{EMA_SHORT_PERIOD}', -np.inf) > latest_candle_data.get(f'ema_{EMA_LONG_PERIOD}', -np.inf)
        is_price_above_emas = (current_real_time_price > latest_candle_data.get(f'ema_{EMA_SHORT_PERIOD}', -np.inf) and
                               current_real_time_price > latest_candle_data.get(f'ema_{EMA_LONG_PERIOD}', -np.inf))
        is_price_above_vwma = current_real_time_price > latest_candle_data.get('vwma', -np.inf)
        is_supertrend_bullish = latest_candle_data.get('supertrend_trend', 0) == 1 and current_real_time_price > latest_candle_data.get('supertrend', -np.inf)
        is_macd_bullish = (latest_candle_data.get('macd_hist', -np.inf) > 0 or
                           (latest_candle_data.get('macd', -np.inf) > latest_candle_data.get('macd_signal', -np.inf) and
                            latest_candle_data.get('macd', -np.inf) > latest_candle_data.iloc[-2].get('macd', -np.inf))) # MACD cross up
        is_adx_strong_trend = latest_candle_data.get('adx', 0) > MIN_ADX_TREND_STRENGTH and latest_candle_data.get('di_plus', 0) > latest_candle_data.get('di_minus', 0)
        
        # Volume Filter
        is_sufficient_volume = latest_candle_data.get('volume_15m_avg', 0) >= MIN_VOLUME_15M_USDT


        # Conditions check
        if ml_prediction == 1:
            logger.info(f"✅ [Signal Gen] ML Model predicts BUY for {self.symbol}. Bypassing essential conditions.")
            is_essential_conditions_met = True # ML overrides
        else:
            is_essential_conditions_met = (
                is_ema_short_above_long and
                is_price_above_emas and
                is_price_above_vwma and
                is_supertrend_bullish and
                is_macd_bullish and
                is_adx_strong_trend
            )
            if not is_essential_conditions_met:
                logger.debug(f"ℹ️ [Signal Gen] الشروط الأساسية غير مستوفاة لـ {self.symbol}.")
                return None

        if not is_sufficient_volume:
            logger.debug(f"ℹ️ [Signal Gen] حجم التداول غير كافٍ لـ {self.symbol}.")
            return None

        # Optional Conditions Scoring (only if essential conditions are met or bypassed by ML)
        score = 0
        max_score = 0

        # VWAP
        if current_real_time_price > latest_candle_data.get('vwap', -np.inf):
            score += 10
        max_score += 10

        # RSI Range
        rsi_val = latest_candle_data.get('rsi')
        if rsi_val is not None and (RSI_OVERSOLD < rsi_val < RSI_OVERBOUGHT):
            score += 10
        max_score += 10

        # Bullish Candlestick Pattern
        if latest_candle_data.get('BullishCandleSignal', 0) == 1:
            score += 20
        max_score += 20

        # Not near Bollinger Upper Band
        if current_real_time_price < latest_candle_data.get('bb_upper', np.inf) * 0.99: # 1% buffer
            score += 10
        max_score += 10

        # OBV increasing
        if latest_candle_data.get('obv', -np.inf) > latest_candle_data.iloc[-2].get('obv', -np.inf):
            score += 10
        max_score += 10

        # RSI Breakout Filter (RSI between 50 and 80)
        if rsi_val is not None and (50 <= rsi_val <= 80):
            score += 10
        max_score += 10

        # MACD Histogram positive
        if latest_candle_data.get('macd_hist', -np.inf) > 0:
            score += 10
        max_score += 10

        # MACD Histogram increasing for last N candles
        macd_hist_series = latest_candle_data.index.to_series().apply(lambda x: df_copy.loc[x, 'macd_hist'])
        if len(macd_hist_series) >= MACD_HIST_INCREASE_CANDLES:
            if np.all(np.diff(macd_hist_series.iloc[-MACD_HIST_INCREASE_CANDLES:]) > 0):
                score += 10
        max_score += 10

        # OBV increasing for last N candles
        obv_series = latest_candle_data.index.to_series().apply(lambda x: df_copy.loc[x, 'obv'])
        if len(obv_series) >= OBV_INCREASE_CANDLES:
            if np.all(np.diff(obv_series.iloc[-OBV_INCREASE_CANDLES:]) > 0):
                score += 10
        max_score += 10


        min_score_threshold = max_score * 0.70 # 70% of max score
        is_optional_conditions_met = (score >= min_score_threshold)
        
        if ml_prediction == 0 and not is_optional_conditions_met:
            logger.debug(f"ℹ️ [Signal Gen] الشروط الاختيارية غير مستوفاة (الدرجة: {score}/{max_score}) لـ {self.symbol}.")
            return None

        # Calculate initial target price
        atr_value = latest_candle_data.get('atr')
        if atr_value is None or atr_value <= 0:
            logger.warning(f"⚠️ [Signal Gen] قيمة ATR غير صالحة لـ {self.symbol}. لا يمكن حساب الهدف.")
            return None

        initial_target_price = current_real_time_price + (ENTRY_ATR_MULTIPLIER * atr_value)

        # Check profit margin
        profit_margin_pct = (initial_target_price / current_real_time_price) - 1
        if profit_margin_pct < MIN_PROFIT_MARGIN_PCT:
            logger.debug(f"ℹ️ [Signal Gen] هامش الربح المتوقع ({profit_margin_pct:.4f}%) أقل من الحد الأدنى ({MIN_PROFIT_MARGIN_PCT:.4f}%) لـ {self.symbol}.")
            return None

        signal_details = {
            'ml_prediction': int(ml_prediction), # Convert numpy.int64 to int
            'score': score,
            'max_score': max_score,
            'is_essential_conditions_met': is_essential_conditions_met,
            'is_sufficient_volume': is_sufficient_volume,
            'current_atr': atr_value,
            'current_rsi': rsi_val,
            'current_macd_hist': latest_candle_data.get('macd_hist'),
            'current_supertrend_trend': latest_candle_data.get('supertrend_trend'),
            'current_vwma': latest_candle_data.get('vwma'),
            'current_ema_short': latest_candle_data.get(f'ema_{EMA_SHORT_PERIOD}'),
            'current_ema_long': latest_candle_data.get(f'ema_{EMA_LONG_PERIOD}'),
            'current_adx': latest_candle_data.get('adx'),
            'current_di_plus': latest_candle_data.get('di_plus'),
            'current_di_minus': latest_candle_data.get('di_minus'),
            'current_vwap': latest_candle_data.get('vwap'),
            'current_obv': latest_candle_data.get('obv'),
            'current_volume_15m_avg': latest_candle_data.get('volume_15m_avg'),
            'current_rsi_momentum_bullish': latest_candle_data.get('rsi_momentum_bullish'),
            'latest_candle_close': latest_candle_data['close'],
            'latest_candle_open': latest_candle_data['open'],
            'latest_candle_high': latest_candle_data['high'],
            'latest_candle_low': latest_candle_data['low'],
            'latest_candle_volume': latest_candle_data['volume'],
            'timestamp': latest_candle_data.name.isoformat() # Timestamp of the candle
        }

        signal = {
            'symbol': self.symbol,
            'entry_price': current_real_time_price,
            'initial_target': initial_target_price,
            'current_target': initial_target_price,
            'r2_score': model_metrics.get('r2_score', None) if self.ml_model else None, # Assuming ML model has R2 score
            'volume_15m': latest_candle_data.get('volume_15m_avg'),
            'strategy_name': 'ScalpingTradingStrategy_V1',
            'signal_details': signal_details
        }
        return signal


def save_signal_to_db(signal: Dict[str, Any]) -> bool:
    """Saves a generated signal to the database."""
    if not check_db_connection() or not conn:
        logger.error("❌ [DB Save] لا يمكن حفظ الإشارة بسبب مشكلة في اتصال قاعدة البيانات.")
        return False

    logger.info(f"ℹ️ [DB Save] حفظ إشارة لـ {signal['symbol']} في قاعدة البيانات...")
    try:
        with db_lock: # Use lock for DB access
            with conn.cursor() as db_cur:
                insert_query = sql.SQL("""
                    INSERT INTO signals (
                        symbol, entry_price, initial_target, current_target,
                        r2_score, volume_15m, strategy_name, signal_details
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                """)
                db_cur.execute(insert_query, (
                    signal['symbol'],
                    signal['entry_price'],
                    signal['initial_target'],
                    signal['current_target'],
                    signal.get('r2_score'),
                    signal.get('volume_15m'),
                    signal['strategy_name'],
                    json.dumps(signal['signal_details']) # Convert dict to JSON string
                ))
            conn.commit()
        logger.info(f"✅ [DB Save] تم حفظ إشارة {signal['symbol']} بنجاح.")
        return True
    except psycopg2.Error as db_err:
        logger.error(f"❌ [DB Save] خطأ في قاعدة البيانات أثناء حفظ الإشارة: {db_err}", exc_info=True)
        if conn: conn.rollback()
        return False
    except Exception as e:
        logger.error(f"❌ [DB Save] خطأ غير متوقع أثناء حفظ الإشارة: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

def get_open_signals_from_db() -> List[Dict[str, Any]]:
    """Fetches all open signals from the database."""
    if not check_db_connection() or not conn:
        logger.error("❌ [DB Fetch] لا يمكن جلب الإشارات المفتوحة بسبب مشكلة في اتصال قاعدة البيانات.")
        return []

    logger.debug("ℹ️ [DB Fetch] جلب الإشارات المفتوحة من قاعدة البيانات...")
    try:
        with db_lock: # Use lock for DB access
            with conn.cursor() as db_cur:
                db_cur.execute("SELECT * FROM signals WHERE achieved_target = FALSE;")
                signals = db_cur.fetchall()
                logger.debug(f"✅ [DB Fetch] تم جلب {len(signals)} إشارة مفتوحة.")
                return signals
    except psycopg2.Error as db_err:
        logger.error(f"❌ [DB Fetch] خطأ في قاعدة البيانات أثناء جلب الإشارات المفتوحة: {db_err}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"❌ [DB Fetch] خطأ غير متوقع أثناء جلب الإشارات المفتوحة: {e}", exc_info=True)
        return []

def update_signal_in_db(signal_id: int, updates: Dict[str, Any]) -> bool:
    """Updates an existing signal in the database."""
    if not check_db_connection() or not conn:
        logger.error("❌ [DB Update] لا يمكن تحديث الإشارة بسبب مشكلة في اتصال قاعدة البيانات.")
        return False

    logger.info(f"ℹ️ [DB Update] تحديث الإشارة ID: {signal_id}...")
    try:
        with db_lock: # Use lock for DB access
            with conn.cursor() as db_cur:
                set_clauses = [sql.SQL(f"{k} = %s") for k in updates.keys()]
                update_query = sql.SQL("UPDATE signals SET {} WHERE id = %s;").format(
                    sql.SQL(', ').join(set_clauses)
                )
                db_cur.execute(update_query, (*updates.values(), signal_id))
            conn.commit()
        logger.info(f"✅ [DB Update] تم تحديث الإشارة ID: {signal_id} بنجاح.")
        return True
    except psycopg2.Error as db_err:
        logger.error(f"❌ [DB Update] خطأ في قاعدة البيانات أثناء تحديث الإشارة ID: {signal_id}: {db_err}", exc_info=True)
        if conn: conn.rollback()
        return False
    except Exception as e:
        logger.error(f"❌ [DB Update] خطأ غير متوقع أثناء تحديث الإشارة ID: {signal_id}: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

def send_telegram_alert(message: str) -> None:
    """Sends a message to the configured Telegram chat."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.warning("⚠️ [Telegram] لم يتم تكوين رمز Telegram المميز أو معرف الدردشة. تخطي إرسال التنبيه.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown' # Use Markdown for formatting
    }
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        logger.info("✅ [Telegram] تم إرسال تنبيه Telegram بنجاح.")
    except requests.exceptions.Timeout:
        logger.error("❌ [Telegram] انتهت مهلة طلب Telegram.")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [Telegram] خطأ في إرسال تنبيه Telegram: {e}")
    except Exception as e:
        logger.error(f"❌ [Telegram] خطأ غير متوقع أثناء إرسال تنبيه Telegram: {e}", exc_info=True)


# ---------------------- WebSocket Data Processing ----------------------

def process_ticker_data(msg: Dict[str, Any]) -> None:
    """
    Processes incoming WebSocket ticker data.
    Updates global ticker_data and aggregates 5-minute candles.
    Triggers signal generation when a 5-minute candle closes.
    """
    global ticker_data, current_5m_candles, symbol_strategies

    event_type = msg.get('e')
    if event_type == '24hrTicker':
        symbol = msg['s']
        last_price = float(msg['c'])
        volume = float(msg['v']) # Base asset volume
        quote_volume = float(msg['q']) # Quote asset volume (USDT)
        event_time_ms = msg['E'] # Event time in milliseconds

        # Update global real-time ticker data
        ticker_data[symbol] = {
            'last_price': last_price,
            'volume': volume,
            'quote_volume': quote_volume,
            'event_time': event_time_ms
        }
        logger.debug(f"Received ticker for {symbol}: {last_price}")

        # --- Candle Aggregation Logic ---
        # Calculate the start time for the current 5-minute candle window
        current_5m_window_start_ms = (event_time_ms // (5 * 60 * 1000)) * (5 * 60 * 1000)

        if symbol not in current_5m_candles:
            # Initialize new candle for this symbol if it's the first data point for this symbol
            current_5m_candles[symbol] = {
                'open': last_price,
                'high': last_price,
                'low': last_price,
                'close': last_price,
                'volume': 0.0,
                'start_time': current_5m_window_start_ms,
                'last_update_time': event_time_ms
            }
            logger.debug(f"ℹ️ [Candle Aggregator] بدء شمعة 5m جديدة لـ {symbol} عند {datetime.fromtimestamp(current_5m_window_start_ms / 1000)}")

        candle = current_5m_candles[symbol]

        # Check if the current event falls into a new 5-minute window
        # This means the previous candle is complete
        if current_5m_window_start_ms > candle['start_time']:
            logger.info(f"ℹ️ [Candle Aggregator] إغلاق شمعة 5m سابقة لـ {symbol} عند {datetime.fromtimestamp(candle['start_time'] / 1000)}")
            # Finalize the previous candle and process it
            closed_candle_data = {
                'timestamp': candle['start_time'],
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle['volume']
            }
            # Asynchronously process the closed candle to avoid blocking WebSocket
            # Use a separate thread for processing to keep WebSocket listener responsive
            Thread(target=process_closed_candle, args=(symbol, closed_candle_data), daemon=True).start()

            # Start a new candle with the current price as its open
            current_5m_candles[symbol] = {
                'open': last_price,
                'high': last_price,
                'low': last_price,
                'close': last_price,
                'volume': 0.0, # Reset volume for new candle
                'start_time': current_5m_window_start_ms,
                'last_update_time': event_time_ms
            }
            logger.debug(f"ℹ️ [Candle Aggregator] بدء شمعة 5m جديدة لـ {symbol} عند {datetime.fromtimestamp(current_5m_window_start_ms / 1000)}")

        # Update the current candle with the latest tick data
        candle['high'] = max(candle['high'], last_price)
        candle['low'] = min(candle['low'], last_price)
        candle['close'] = last_price
        candle['volume'] += volume # Accumulate volume for the current candle
        candle['last_update_time'] = event_time_ms

    else:
        logger.warning(f"⚠️ [WebSocket] نوع رسالة غير متوقع: {event_type}")


def process_closed_candle(symbol: str, closed_candle: Dict[str, Any]) -> None:
    """
    Processes a newly closed 5-minute candle for a specific symbol.
    Fetches recent historical data, appends the new candle, calculates indicators,
    and generates a buy signal if conditions are met.
    """
    logger.info(f"⏳ [Signal Gen] معالجة شمعة 5m مغلقة لـ {symbol}...")
    try:
        # 1. Fetch recent historical data (e.g., last 200 candles)
        # This is crucial for indicator calculation that needs lookback periods.
        # Fetch slightly more than needed to ensure enough data after potential NaNs.
        # Use DATA_LOOKBACK_DAYS for consistency with ml.py's training data range
        df_hist = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS)
        
        if df_hist is None or df_hist.empty:
            logger.warning(f"⚠️ [Signal Gen] لا توجد بيانات تاريخية كافية لـ {symbol} لمعالجة الشمعة المغلقة.")
            return

        # Convert closed_candle timestamp to datetime for DataFrame indexing
        closed_candle_timestamp = pd.to_datetime(closed_candle['timestamp'], unit='ms')

        # Check if the closed candle is already the very last candle in df_hist
        # This can happen if get_historical_klines returns data up to the current moment.
        # If so, we update it. Otherwise, append.
        if not df_hist.empty and df_hist.index[-1] == closed_candle_timestamp:
            logger.debug(f"ℹ️ [Signal Gen] الشمعة المغلقة لـ {symbol} موجودة بالفعل في البيانات التاريخية (آخر شمعة). تحديثها.")
            df_hist.loc[closed_candle_timestamp] = [
                closed_candle['open'], closed_candle['high'],
                closed_candle['low'], closed_candle['close'], closed_candle['volume']
            ]
        elif closed_candle_timestamp not in df_hist.index: # Only append if not already present
            # Append the new closed candle to the historical DataFrame
            new_candle_df = pd.DataFrame([closed_candle], index=[closed_candle_timestamp], columns=['open', 'high', 'low', 'close', 'volume'])
            df_hist = pd.concat([df_hist, new_candle_df]).sort_index()
            logger.debug(f"ℹ️ [Signal Gen] تم إضافة شمعة 5m مغلقة جديدة لـ {symbol} إلى البيانات التاريخية.")
        else:
            logger.debug(f"ℹ️ [Signal Gen] الشمعة المغلقة لـ {symbol} موجودة بالفعل في البيانات التاريخية (ليست الأحدث). تخطي الإضافة.")


        # Get the strategy instance for this symbol
        strategy = symbol_strategies.get(symbol)
        if not strategy:
            logger.error(f"❌ [Signal Gen] لم يتم العثور على استراتيجية لـ {symbol}. تخطي توليد الإشارة.")
            return

        # Calculate indicators on the updated DataFrame
        # Pass a copy to populate_indicators to avoid modifying the original df_hist
        df_with_indicators = strategy.populate_indicators(df_hist.copy())
        
        # Ensure we have enough data after indicator calculation for the latest candle
        if df_with_indicators.empty or len(df_with_indicators) < 2:
            logger.warning(f"⚠️ [Signal Gen] DataFrame لـ {symbol} فارغ أو غير كافٍ بعد حساب المؤشرات. تخطي.")
            return

        # Get the latest candle's data with calculated indicators
        latest_candle_data = df_with_indicators.iloc[-1]
        
        # Get the current real-time price from ticker_data for the entry price
        current_real_time_price = ticker_data.get(symbol, {}).get('last_price')
        if current_real_time_price is None:
            logger.warning(f"⚠️ [Signal Gen] لم يتم العثور على سعر الوقت الفعلي لـ {symbol}. استخدام سعر الإغلاق للشمعة المغلقة كـ سعر دخول.")
            current_real_time_price = latest_candle_data['close']

        # Generate buy signal
        signal = strategy.generate_buy_signal(latest_candle_data, current_real_time_price)

        if signal:
            logger.info(f"🔥 [Signal Gen] تم توليد إشارة شراء لـ {symbol}!")
            # Save signal to DB and send Telegram alert
            if save_signal_to_db(signal):
                logger.info(f"✅ [Signal Gen] تم حفظ إشارة {symbol} في قاعدة البيانات.")
                send_telegram_alert(f"🔥 إشارة شراء جديدة لـ {signal['symbol']}!\n"
                                    f"السعر: {signal['entry_price']:.4f}\n"
                                    f"الهدف: {signal['initial_target']:.4f}\n"
                                    f"الربح المتوقع: {signal['profit_percentage']:.2f}%")
            else:
                logger.error(f"❌ [Signal Gen] فشل حفظ إشارة {symbol} في قاعدة البيانات.")
        else:
            logger.debug(f"ℹ️ [Signal Gen] لا توجد إشارة شراء لـ {symbol} في هذه الشمعة.")

    except Exception as e:
        logger.error(f"❌ [Signal Gen] خطأ في معالجة الشمعة المغلقة لـ {symbol}: {e}", exc_info=True)


# ---------------------- Signal Tracking and Target Update ----------------------

def track_signals() -> None:
    """
    Periodically fetches and tracks open signals, updating their status
    and target prices based on real-time data.
    """
    global open_signals, ticker_data

    logger.info("✅ [Tracker] بدء مؤشر الإشارة.")
    while True:
        try:
            # Refresh open signals from DB
            with open_signals_lock:
                open_signals = get_open_signals_from_db()

            if not open_signals:
                logger.debug("ℹ️ [Tracker] لا توجد إشارات مفتوحة للتتبع.")
                time.sleep(10) # Sleep longer if no signals
                continue

            for signal in open_signals:
                symbol = signal['symbol']
                signal_id = signal['id']
                current_target = signal['current_target']
                entry_price = signal['entry_price']

                # Get the latest real-time price
                current_price_data = ticker_data.get(symbol)
                if not current_price_data:
                    logger.warning(f"⚠️ [Tracker] لا توجد بيانات تيكر لـ {symbol}. تخطي تتبع الإشارة ID: {signal_id}.")
                    continue

                current_price = current_price_data['last_price']

                # Check if target is achieved
                if current_price >= current_target:
                    profit_percentage = ((current_price / entry_price) - 1) * 100
                    updates = {
                        'achieved_target': True,
                        'closing_price': current_price,
                        'closed_at': datetime.utcnow(),
                        'profit_percentage': profit_percentage,
                        'time_to_target': datetime.utcnow() - signal['entry_time'] # Calculate duration
                    }
                    if update_signal_in_db(signal_id, updates):
                        logger.info(f"🎯 [Tracker] تم الوصول إلى الهدف لـ {symbol} (ID: {signal_id}). السعر: {current_price:.4f}، الربح: {profit_percentage:.2f}%")
                        send_telegram_alert(f"🎯 *تم الوصول إلى الهدف!* لـ {symbol}\n"
                                            f"سعر الدخول: {entry_price:.4f}\n"
                                            f"سعر الإغلاق: {current_price:.4f}\n"
                                            f"الربح: {profit_percentage:.2f}%")
                else:
                    # Logic for target extension/trailing (if price approaches target but not hit)
                    # This part needs to re-evaluate the strategy based on current market conditions
                    # and potentially extend the target if the trend continues.
                    # This is similar to the signal generation logic but for open positions.
                    if current_price >= current_target * (1 - TARGET_APPROACH_THRESHOLD_PCT):
                        logger.debug(f"ℹ️ [Tracker] السعر يقترب من الهدف لـ {symbol}. إعادة تقييم...")
                        
                        # Get the strategy instance for this symbol
                        strategy = symbol_strategies.get(symbol)
                        if not strategy:
                            logger.error(f"❌ [Tracker] لم يتم العثور على استراتيجية لـ {symbol} لإعادة التقييم.")
                            continue

                        # Fetch fresh historical data for indicator recalculation
                        df_hist_re_eval = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS)
                        if df_hist_re_eval is None or df_hist_re_eval.empty:
                            logger.warning(f"⚠️ [Tracker] لا توجد بيانات تاريخية كافية لـ {symbol} لإعادة تقييم الهدف.")
                            continue

                        df_with_indicators_re_eval = strategy.populate_indicators(df_hist_re_eval.copy())
                        if df_with_indicators_re_eval.empty or len(df_with_indicators_re_eval) < 2:
                            logger.warning(f"⚠️ [Tracker] DataFrame لـ {symbol} فارغ أو غير كافٍ بعد حساب المؤشرات لإعادة تقييم الهدف.")
                            continue
                        
                        latest_candle_data_re_eval = df_with_indicators_re_eval.iloc[-1]
                        
                        # Re-run a simplified signal generation check to see if conditions are still bullish
                        # This is to check if a new, higher target would be valid
                        # We don't want to create a new signal, just update the existing one's target
                        
                        # Prepare features for ML prediction (if ML is used for re-evaluation)
                        ml_features_re_eval = {}
                        for feature in strategy.feature_names:
                            ml_features_re_eval[feature] = latest_candle_data_re_eval.get(feature)
                            if pd.isna(ml_features_re_eval[feature]):
                                ml_features_re_eval[feature] = 0.0 # Handle NaN features gracefully

                        features_df_re_eval = pd.DataFrame([ml_features_re_eval])

                        ml_prediction_re_eval = 0
                        if strategy.ml_model:
                            try:
                                features_for_prediction_re_eval = features_df_re_eval[strategy.feature_names].values
                                ml_prediction_re_eval = strategy.ml_model.predict(features_for_prediction_re_eval)[0]
                            except Exception as e:
                                logger.error(f"❌ [Tracker] خطأ في تنبؤ نموذج ML لإعادة تقييم الهدف لـ {symbol}: {e}", exc_info=True)
                                ml_prediction_re_eval = 0

                        # Check essential conditions again (or rely on ML prediction)
                        is_ema_short_above_long = latest_candle_data_re_eval.get(f'ema_{EMA_SHORT_PERIOD}', -np.inf) > latest_candle_data_re_eval.get(f'ema_{EMA_LONG_PERIOD}', -np.inf)
                        is_price_above_emas = (current_price > latest_candle_data_re_eval.get(f'ema_{EMA_SHORT_PERIOD}', -np.inf) and
                                               current_price > latest_candle_data_re_eval.get(f'ema_{EMA_LONG_PERIOD}', -np.inf))
                        is_price_above_vwma = current_price > latest_candle_data_re_eval.get('vwma', -np.inf)
                        is_supertrend_bullish = latest_candle_data_re_eval.get('supertrend_trend', 0) == 1 and current_price > latest_candle_data_re_eval.get('supertrend', -np.inf)
                        is_macd_bullish = (latest_candle_data_re_eval.get('macd_hist', -np.inf) > 0 or
                                           (latest_candle_data_re_eval.get('macd', -np.inf) > latest_candle_data_re_eval.get('macd_signal', -np.inf)))
                        is_adx_strong_trend = latest_candle_data_re_eval.get('adx', 0) > MIN_ADX_TREND_STRENGTH and latest_candle_data_re_eval.get('di_plus', 0) > latest_candle_data_re_eval.get('di_minus', 0)
                        
                        is_re_evaluation_conditions_met = (ml_prediction_re_eval == 1) or (
                            is_ema_short_above_long and
                            is_price_above_emas and
                            is_price_above_vwma and
                            is_supertrend_bullish and
                            is_macd_bullish and
                            is_adx_strong_trend
                        )

                        if is_re_evaluation_conditions_met:
                            atr_value_re_eval = latest_candle_data_re_eval.get('atr')
                            if atr_value_re_eval is not None and atr_value_re_eval > 0:
                                new_potential_target = current_price + (ENTRY_ATR_MULTIPLIER * atr_value_re_eval)
                                if new_potential_target > current_target:
                                    updates = {'current_target': new_potential_target}
                                    if update_signal_in_db(signal_id, updates):
                                        logger.info(f"⬆️ [Tracker] تم تحديث الهدف لـ {symbol} (ID: {signal_id}) من {current_target:.4f} إلى {new_potential_target:.4f}.")
                                        send_telegram_alert(f"⬆️ *تم تحديث الهدف!* لـ {symbol}\n"
                                                            f"الهدف القديم: {current_target:.4f}\n"
                                                            f"الهدف الجديد: {new_potential_target:.4f}\n"
                                                            f"السعر الحالي: {current_price:.4f}")
                        else:
                            logger.debug(f"ℹ️ [Tracker] الظروف لم تعد مواتية لتمديد الهدف لـ {symbol}.")

            time.sleep(5) # Check open signals every 5 seconds
        except Exception as e:
            logger.critical(f"❌ [Tracker] حدث خطأ فادح في تتبع الإشارات: {e}", exc_info=True)
            time.sleep(30) # Longer sleep on error


# ---------------------- Flask Service (for Render uptime) ----------------------
app = Flask(__name__)

@app.route('/')
def home() -> Response:
    """Simple home page to show the bot is running."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # You can add more status details here if needed
    status_message = (
        f"🤖 *Crypto Bot Service Status:*\n"
        f"- Current Time: {now}\n"
        f"- Service: Running and monitoring markets.\n"
        f"- Open Signals: {len(open_signals)}\n"
        f"- Ticker Data Symbols: {len(ticker_data)}\n"
    )
    return Response(status_message, status=200, mimetype='text/plain')

@app.route('/favicon.ico')
def favicon() -> Response:
    """Handles favicon request to avoid 404 errors in logs."""
    return Response(status=204)

def run_flask_service() -> None:
    """Runs the Flask application."""
    host = "0.0.0.0"
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"ℹ️ [Flask] بدء تطبيق Flask على {host}:{port}...")
    try:
        from waitress import serve
        logger.info("✅ [Flask] استخدام خادم 'waitress'.")
        serve(app, host=host, port=port, threads=6)
    except ImportError:
        logger.warning("⚠️ [Flask] 'waitress' غير مثبت. الرجوع إلى خادم تطوير Flask (لا يوصى به للإنتاج).")
        try:
            app.run(host=host, port=port)
        except Exception as flask_run_err:
            logger.critical(f"❌ [Flask] فشل بدء خادم التطوير: {flask_run_err}", exc_info=True)
    except Exception as serve_err:
        logger.critical(f"❌ [Flask] فشل بدء الخادم (waitress؟): {serve_err}", exc_info=True)


# ---------------------- Main Execution ----------------------
def main_loop() -> None:
    """
    Initializes strategy instances and loads ML models for all symbols.
    Signal generation is handled by real-time candle processing via WebSocket.
    """
    global all_symbols, symbol_strategies

    logger.info("ℹ️ [Main Loop] بدء تهيئة الاستراتيجيات وتحميل نماذج ML...")

    for symbol in all_symbols:
        try:
            # Initialize strategy for each symbol. It will load its specific ML model.
            strategy = ScalpingTradingStrategy(symbol, client, conn, cur)
            symbol_strategies[symbol] = strategy
            logger.info(f"✅ [Main Loop] تم تهيئة الاستراتيجية لـ {symbol} وتحميل نموذج ML.")
        except Exception as e:
            logger.error(f"❌ [Main Loop] فشل تهيئة الاستراتيجية لـ {symbol}: {e}", exc_info=True)
            # Decide whether to continue without this symbol or exit.
            # For now, we'll just log and continue.

    logger.info("✅ [Main Loop] اكتمل تهيئة الاستراتيجيات وتحميل نماذج ML.")
    # The main loop now just keeps running, waiting for WebSocket events to trigger signal generation.
    # It might also have a periodic check for overall health or re-initialization if needed.
    while True:
        time.sleep(60) # Keep the thread alive, but actual work is event-driven
        logger.debug("ℹ️ [Main Loop] خيط الحلقة الرئيسية لا يزال نشطًا.")


if __name__ == "__main__":
    logger.info("🚀 بدء بوت تداول العملات المشفرة...")
    logger.info(f"الوقت المحلي: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | وقت UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. بدء خدمة Flask في خيط منفصل أولاً
    flask_thread = Thread(target=run_flask_service, daemon=False, name="FlaskServiceThread")
    flask_thread.start()
    logger.info("✅ [Main] تم بدء خدمة Flask.")
    time.sleep(2) # إعطاء بعض الوقت لـ Flask للبدء

    try:
        # 2. تهيئة قاعدة البيانات
        init_db()

        # 3. جلب قائمة الرموز
        all_symbols = get_crypto_symbols()
        if not all_symbols:
            logger.critical("❌ [Main] لا توجد رموز صالحة للتدفق. يرجى التحقق من 'crypto_list.txt'.")
            exit(1)

        # 4. بدء WebSocket Manager
        twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
        twm.start()
        logger.info("✅ [Main] تم بدء ThreadedWebsocketManager.")

        # 5. الاشتراك في تدفقات التيكر لجميع الرموز
        # استخدام 'allTickers' لتدفق جميع الرموز
        twm.start_all_ticker_futures_socket(callback=process_ticker_data) # Changed to futures for all tickers
        logger.info(f"✅ [Main] تم الاشتراك في تدفقات التيكر لجميع الرموز.")

        # 6. إعطاء بعض الوقت لـ WebSocket لاستقبال البيانات الأولية
        logger.info("ℹ️ [Main] انتظار استقبال بيانات أولية من WebSocket...")
        time.sleep(5)
        if not ticker_data:
             logger.warning("⚠️ [Main] لم يتم استلام بيانات أولية من WebSocket بعد 5 ثوانٍ.")
        else:
             logger.info(f"✅ [Main] تم استلام بيانات أولية من WebSocket لـ {len(ticker_data)} رمزًا.")


        # 7. Start Signal Tracker
        tracker_thread = Thread(target=track_signals, daemon=True, name="TrackerThread")
        tracker_thread.start()
        logger.info("✅ [Main] تم بدء مؤشر الإشارة.")

        # 8. Start the main bot logic in a separate thread (now primarily for strategy initialization)
        main_bot_thread = Thread(target=main_loop, daemon=True, name="MainBotLoopThread")
        main_bot_thread.start()
        logger.info("✅ [Main] تم بدء حلقة البوت الرئيسية في خيط منفصل.")

        # Wait for the Flask thread to finish (it usually won't unless there's an error),
        # keeping the main program alive.
        flask_thread.join()

    except Exception as startup_err:
        logger.critical(f"❌ [Main] حدث خطأ فادح أثناء بدء التشغيل: {startup_err}", exc_info=True)
    finally:
        logger.info("🛑 [Main] يتم إيقاف تشغيل البوت...")
        if twm:
            twm.stop()
            logger.info("✅ [WebSocket] تم إيقاف ThreadedWebsocketManager.")
        cleanup_resources()
        logger.info("👋 [Main] تم إيقاف بوت تداول العملات المشفرة.")

