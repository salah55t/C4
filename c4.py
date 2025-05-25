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
MAX_OPEN_TRADES: int = 20
SIGNAL_GENERATION_TIMEFRAME: str = '5m'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 3
SIGNAL_TRACKING_TIMEFRAME: str = '5m'
SIGNAL_TRACKING_LOOKBACK_DAYS: int = 1

# Indicator Parameters
RSI_PERIOD: int = 9
RSI_OVERSOLD: int = 30
RSI_OVERBOUGHT: int = 70
EMA_SHORT_PERIOD: int = 8
EMA_LONG_PERIOD: int = 21
VWMA_PERIOD: int = 15
SWING_ORDER: int = 3
FIB_LEVELS_TO_CHECK: List[float] = [0.382, 0.5, 0.618]
FIB_TOLERANCE: float = 0.005
LOOKBACK_FOR_SWINGS: int = 50
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
        start_dt = datetime.utcnow() - timedelta(days=days + 1)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        logger.debug(f"ℹ️ [Data] جلب بيانات {interval} لـ {symbol} منذ {start_str} (حد 1000 شمعة)...")

        klines = client.get_historical_klines(symbol, interval, start_str, limit=1000)

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

def get_btc_trend_4h() -> str:
    """Calculates Bitcoin trend on 4-hour timeframe using EMA20 and EMA50."""
    logger.debug("ℹ️ [Indicators] حساب اتجاه البيتكوين على 4 ساعات...")
    try:
        df = fetch_historical_data("BTCUSDT", interval=Client.KLINE_INTERVAL_4HOUR, days=10)
        if df is None or df.empty or len(df) < 50 + 1:
            logger.warning("⚠️ [Indicators] بيانات BTC/USDT 4H غير كافية لحساب الاتجاه.")
            return "N/A (بيانات غير كافية)"

        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df.dropna(subset=['close'], inplace=True)
        if len(df) < 50:
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
        elif diff_ema20_pct < 0.005:
            trend = "استقرار 🔄"
        else:
            trend = "تذبذب 🔀"

        logger.debug(f"✅ [Indicators] اتجاه البيتكوين 4H: {trend}")
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
            conn.autocommit = False
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

            # --- Create ml_models table (NEW) ---
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

def load_ml_model_from_db() -> Optional[Any]:
    """Loads the latest trained ML model from the database."""
    global ml_model
    if not check_db_connection() or not conn:
        logger.error("❌ [ML Model] لا يمكن تحميل نموذج ML بسبب مشكلة في اتصال قاعدة البيانات.")
        return None

    try:
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT model_data FROM ml_models WHERE model_name = %s ORDER BY trained_at DESC LIMIT 1;", (ML_MODEL_NAME,))
            result = db_cur.fetchone()
            if result and result['model_data']:
                ml_model = pickle.loads(result['model_data'])
                logger.info(f"✅ [ML Model] تم تحميل نموذج ML '{ML_MODEL_NAME}' من قاعدة البيانات بنجاح.")
                return ml_model
            else:
                logger.warning(f"⚠️ [ML Model] لم يتم العثور على نموذج ML باسم '{ML_MODEL_NAME}' في قاعدة البيانات. يرجى تدريب النموذج أولاً.")
                return None
    except psycopg2.Error as db_err:
        logger.error(f"❌ [ML Model] خطأ في قاعدة البيانات أثناء تحميل نموذج ML: {db_err}", exc_info=True)
        return None
    except pickle.UnpicklingError as unpickle_err:
        logger.error(f"❌ [ML Model] خطأ في فك تسلسل نموذج ML: {unpickle_err}. قد يكون النموذج تالفًا أو تم حفظه بإصدار مختلف.", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"❌ [ML Model] خطأ غير متوقع أثناء تحميل نموذج ML: {e}", exc_info=True)
        return None


def convert_np_values(obj: Any) -> Any:
    """Converts NumPy data types to native Python types for JSON and DB compatibility."""
    if isinstance(obj, dict):
        return {k: convert_np_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_values(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif pd.isna(obj):
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


# ---------------------- WebSocket Management for Ticker Prices ----------------------
def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
    """Handles incoming WebSocket messages for mini-ticker prices."""
    global ticker_data
    try:
        if isinstance(msg, list):
            for ticker_item in msg:
                symbol = ticker_item.get('s')
                price_str = ticker_item.get('c')
                if symbol and 'USDT' in symbol and price_str:
                    try:
                        ticker_data[symbol] = float(price_str)
                    except ValueError:
                         logger.warning(f"⚠️ [WS] قيمة سعر غير صالحة للرمز {symbol}: '{price_str}'")
        elif isinstance(msg, dict):
             if msg.get('e') == 'error':
                 logger.error(f"❌ [WS] رسالة خطأ من WebSocket: {msg.get('m', 'لا توجد تفاصيل خطأ')}")
             elif msg.get('stream') and msg.get('data'):
                 for ticker_item in msg.get('data', []):
                    symbol = ticker_item.get('s')
                    price_str = ticker_item.get('c')
                    if symbol and 'USDT' in symbol and price_str:
                        try:
                            ticker_data[symbol] = float(price_str)
                        except ValueError:
                             logger.warning(f"⚠️ [WS] قيمة سعر غير صالحة للرمز {symbol} في البث المجمع: '{price_str}'")
        else:
             logger.warning(f"⚠️ [WS] تم استلام رسالة WebSocket بتنسيق غير متوقع: {type(msg)}")

    except Exception as e:
        logger.error(f"❌ [WS] خطأ في معالجة رسالة التيكر: {e}", exc_info=True)


def run_ticker_socket_manager() -> None:
    """Runs and manages the WebSocket connection for mini-ticker."""
    while True:
        try:
            logger.info("ℹ️ [WS] بدء إدارة WebSocket لأسعار التيكر...")
            twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
            twm.start()

            stream_name = twm.start_miniticker_socket(callback=handle_ticker_message)
            logger.info(f"✅ [WS] تم بدء بث WebSocket: {stream_name}")

            twm.join()
            logger.warning("⚠️ [WS] توقفت إدارة WebSocket. إعادة التشغيل...")

        except Exception as e:
            logger.error(f"❌ [WS] خطأ فادح في إدارة WebSocket: {e}. إعادة التشغيل في 15 ثانية...", exc_info=True)

        time.sleep(15)

# ---------------------- Technical Indicator Functions ----------------------

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
         logger.warning(f"⚠️ [Indicator BB] بيانات غير كافية ({len(df)} < {window}) لحساب BB.")
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
    if len(df_calc) < period * 2:
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
    if abs(prev['close'] - prev['open']) < (prev['high'] - prev['low']) * 0.1:
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

# ---------------------- Other Helper Functions (Elliott, Swings, Volume) ----------------------
def detect_swings(prices: np.ndarray, order: int = SWING_ORDER) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Detects swing points (peaks and troughs) in a time series (numpy array)."""
    n = len(prices)
    if n < 2 * order + 1: return [], []

    maxima_indices = []
    minima_indices = []

    for i in range(order, n - order):
        window = prices[i - order : i + order + 1]
        center_val = prices[i]

        if np.isnan(window).any(): continue

        is_max = np.all(center_val >= window)
        is_min = np.all(center_val <= window)
        is_unique_max = is_max and (np.sum(window == center_val) == 1)
        is_unique_min = is_min and (np.sum(window == center_val) == 1)

        if is_unique_max:
            if not maxima_indices or i > maxima_indices[-1] + order:
                 maxima_indices.append(i)
        elif is_unique_min:
            if not minima_indices or i > minima_indices[-1] + order:
                minima_indices.append(i)

    maxima = [(idx, prices[idx]) for idx in maxima_indices]
    minima = [(idx, prices[idx]) for idx in minima_indices]
    return maxima, minima

def detect_elliott_waves(df: pd.DataFrame, order: int = SWING_ORDER) -> List[Dict[str, Any]]:
    """Simple attempt to identify Elliott Waves based on MACD histogram swings."""
    if 'macd_hist' not in df.columns or df['macd_hist'].isnull().all():
        logger.warning("⚠️ [Elliott] عمود 'macd_hist' مفقود أو فارغ لحساب موجات إليوت.")
        return []

    macd_values = df['macd_hist'].dropna().values
    if len(macd_values) < 2 * order + 1:
         logger.warning("⚠️ [Elliott] بيانات MACD hist غير كافية بعد إزالة قيم NaN.")
         return []

    maxima, minima = detect_swings(macd_values, order=order)

    df_nonan_macd = df['macd_hist'].dropna()
    all_swings = sorted(
        [(df_nonan_macd.index[idx], val, 'max') for idx, val in maxima] +
        [(df_nonan_macd.index[idx], val, 'min') for idx, val in minima],
        key=lambda x: x[0]
    )

    waves = []
    wave_number = 1
    for timestamp, val, typ in all_swings:
        wave_type = "Impulse" if (typ == 'max' and val > 0) or (typ == 'min' and val >= 0) else "Correction"
        waves.append({
            "wave": wave_number,
            "timestamp": str(timestamp),
            "macd_hist_value": float(val),
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
        klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=15)
        if not klines or len(klines) < 15:
             logger.warning(f"⚠️ [Data Volume] بيانات 1m غير كافية (أقل من 15 شمعة) لـ {symbol}.")
             return 0.0

        volume_usdt = sum(float(k[7]) for k in klines if len(k) > 7 and k[7])
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
    if not check_db_connection() or not conn or not cur:
        return "❌ لا يمكن إنشاء التقرير، مشكلة في اتصال قاعدة البيانات."
    try:
        with conn.cursor() as report_cur:
            report_cur.execute("SELECT id, symbol, entry_price, entry_time FROM signals WHERE achieved_target = FALSE ORDER BY entry_time DESC;")
            open_signals = report_cur.fetchall()
            open_signals_count = len(open_signals)

            report_cur.execute("""
                SELECT
                    COUNT(*) AS total_closed,
                    COUNT(*) AS winning_signals,
                    COALESCE(SUM(profit_percentage), 0) AS total_profit_pct_sum,
                    COALESCE(AVG(profit_percentage), 0) AS avg_profit_pct,
                    COALESCE(AVG(profit_percentage), 0) AS avg_win_pct,
                    COALESCE(SUM(entry_price * (1 + profit_percentage/100.0)), 0) AS total_exit_value
                FROM signals
                WHERE achieved_target = TRUE;
            """)
            closed_stats = report_cur.fetchone() or {}

            total_closed = closed_stats.get('total_closed', 0)
            winning_signals = closed_stats.get('winning_signals', 0)
            losing_signals = 0
            total_profit_pct_sum = closed_stats.get('total_profit_pct_sum', 0.0)
            gross_profit_pct_sum = total_profit_pct_sum
            avg_win_pct = closed_stats.get('avg_profit_pct', 0.0)
            total_exit_value = closed_stats.get('total_exit_value', 0.0)

            gross_profit_usd = (total_profit_pct_sum / 100.0) * TRADE_VALUE
            total_fees_usd = (total_closed * TRADE_VALUE * BINANCE_FEE_RATE) + (total_exit_value * BINANCE_FEE_RATE)
            net_profit_usd = gross_profit_usd - total_fees_usd
            net_profit_pct = (net_profit_usd / (total_closed * TRADE_VALUE)) * 100 if total_closed * TRADE_VALUE > 0 else 0.0

            gross_loss_pct_sum = 0.0
            avg_loss_pct = 0.0

            win_rate = 100.0 if total_closed > 0 else 0.0
            profit_factor = float('inf') if gross_loss_pct_sum == 0 else (gross_profit_pct_sum / abs(gross_loss_pct_sum))

        report = (
            f"📊 *تقرير الأداء الشامل:*\n"
            f"_(افتراض حجم الصفقة: ${TRADE_VALUE:,.2f} ورسوم Binance: {BINANCE_FEE_RATE*100:.2f}% لكل صفقة)_ \n"
            f"——————————————\n"
            f"📈 الإشارات المفتوحة حالياً: *{open_signals_count}*\n"
        )

        if open_signals:
            report += "  • التفاصيل:\n"
            for signal in open_signals:
                safe_symbol = str(signal['symbol']).replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')
                entry_time_str = signal['entry_time'].strftime('%Y-%m-%d %H:%M') if signal['entry_time'] else 'N/A'
                report += f"    - `{safe_symbol}` (دخول: ${signal['entry_price']:.8g} | فتح: {entry_time_str})\n"
        else:
            report += "  • لا توجد إشارات مفتوحة حالياً.\n"

        report += (
            f"——————————————\n"
            f"📉 *إحصائيات الإشارات المغلقة (تم تحقيق الهدف فقط):*\n"
            f"  • إجمالي الإشارات المغلقة: *{total_closed}*\n"
            f"  ✅ إشارات رابحة: *{winning_signals}* ({win_rate:.2f}%)\n"
            f"  ❌ إشارات خاسرة: *{losing_signals}*\n"
            f"——————————————\n"
            f"💰 *الربحية الإجمالية (للصفقات التي حققت الهدف):*\n"
            f"  • إجمالي الربح الإجمالي: *{gross_profit_pct_sum:+.2f}%* (≈ *${gross_profit_usd:+.2f}*)\n"
            f"  • إجمالي الرسوم المدفوعة: *${total_fees_usd:,.2f}*\n"
            f"  • *الربح الصافي:* *{net_profit_pct:+.2f}%* (≈ *${net_profit_usd:+.2f}*)\n"
            f"  • متوسط الصفقة الرابحة: *{avg_win_pct:+.2f}%*\n"
            f"  • عامل الربح: *{'∞' if profit_factor == float('inf') else f'{profit_factor:.2f}'}*\n"
            f"——————————————\n"
        )

        report += "ℹ️ *ملاحظة: هذا التقرير يعرض فقط الصفقات التي حققت الهدف، حيث تم إزالة منطق وقف الخسارة.*"
        report += "\n——————————————\n"


        report += (
            f"🕰️ _التقرير محدث حتى: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        )

        logger.info("✅ [Report] تم إنشاء تقرير الأداء بنجاح.")
        return report

    except psycopg2.Error as db_err:
        logger.error(f"❌ [Report] خطأ في قاعدة البيانات أثناء إنشاء تقرير الأداء: {db_err}")
        if conn: conn.rollback()
        return "❌ خطأ في قاعدة البيانات أثناء إنشاء تقرير الأداء."
    except Exception as e:
        logger.error(f"❌ [Report] خطأ غير متوقع أثناء إنشاء تقرير الأداء: {e}", exc_info=True)
        return "❌ حدث خطأ غير متوقع أثناء إنشاء تقرير الأداء."

# ---------------------- Trading Strategy (Adjusted for Scalping) -------------------

class ScalpingTradingStrategy:
    """Encapsulates the trading strategy logic and associated indicators with a scoring system and mandatory conditions, adjusted for Scalping and targeting early momentum."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.feature_columns_for_ml = [ # Features expected by the ML model
            f'ema_{EMA_SHORT_PERIOD}', f'ema_{EMA_LONG_PERIOD}', 'vwma',
            'rsi', 'atr', 'bb_upper', 'bb_lower', 'bb_middle',
            'macd', 'macd_signal', 'macd_hist',
            'adx', 'di_plus', 'di_minus', 'vwap', 'obv',
            'supertrend', 'supertrend_trend'
        ]

        self.condition_weights = {
            'rsi_ok': 0.5,
            'bullish_candle': 1.5,
            'not_bb_extreme': 0.5,
            'obv_rising': 1.0,
            'rsi_filter_breakout': 1.0,
            'macd_filter_breakout': 1.0,
            'macd_hist_increasing': 3.0,
            'obv_increasing_recent': 3.0,
            'above_vwap': 1.0,
            # 'ml_prediction_bullish': 2.5 # Removed from weights, now a mandatory override
        }

        self.essential_conditions = [
            'price_above_emas_and_vwma',
            'ema_short_above_ema_long',
            'supertrend_up',
            'macd_positive_or_cross',
            'adx_trending_bullish_strong',
        ]

        self.total_possible_score = sum(self.condition_weights.values())
        self.min_score_threshold_pct = 0.70
        self.min_signal_score = self.total_possible_score * self.min_score_threshold_pct


    def populate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculates all required indicators for the strategy."""
        logger.debug(f"ℹ️ [Strategy {self.symbol}] حساب المؤشرات...")
        min_len_required = max(EMA_SHORT_PERIOD, EMA_LONG_PERIOD, VWMA_PERIOD, RSI_PERIOD, ENTRY_ATR_PERIOD, BOLLINGER_WINDOW, MACD_SLOW, ADX_PERIOD*2, SUPERTREND_PERIOD, RECENT_EMA_CROSS_LOOKBACK, MACD_HIST_INCREASE_CANDLES, OBV_INCREASE_CANDLES) + 5

        if len(df) < min_len_required:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame قصير جدًا ({len(df)} < {min_len_required}) لحساب المؤشرات.")
            return None

        try:
            df_calc = df.copy()
            df_calc = calculate_atr_indicator(df_calc, ENTRY_ATR_PERIOD)
            df_calc = calculate_supertrend(df_calc, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
            df_calc[f'ema_{EMA_SHORT_PERIOD}'] = calculate_ema(df_calc['close'], EMA_SHORT_PERIOD)
            df_calc[f'ema_{EMA_LONG_PERIOD}'] = calculate_ema(df_calc['close'], EMA_LONG_PERIOD)
            df_calc['vwma'] = calculate_vwma(df_calc, VWMA_PERIOD)
            df_calc = calculate_rsi_indicator(df_calc, RSI_PERIOD)
            df_calc = calculate_bollinger_bands(df_calc, BOLLINGER_WINDOW, BOLLINGER_STD_DEV)
            df_calc = calculate_macd(df_calc, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            adx_df = calculate_adx(df_calc, ADX_PERIOD)
            df_calc = df_calc.join(adx_df)
            df_calc = calculate_vwap(df_calc)
            df_calc = calculate_obv(df_calc)
            df_calc = detect_candlestick_patterns(df_calc)

            # Ensure all feature columns for ML exist and are numeric
            for col in self.feature_columns_for_ml:
                if col not in df_calc.columns:
                    logger.warning(f"⚠️ [Strategy {self.symbol}] عمود الميزة المفقود لنموذج ML: {col}")
                    df_calc[col] = np.nan # Add missing column as NaN
                else:
                    df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')

            initial_len = len(df_calc)
            # Use all required columns for dropna, including ML features
            all_required_cols = list(set(self.feature_columns_for_ml + [
                'open', 'high', 'low', 'close', 'volume', 'BullishCandleSignal', 'BearishCandleSignal'
            ]))
            df_cleaned = df_calc.dropna(subset=all_required_cols).copy()
            dropped_count = initial_len - len(df_cleaned)

            if dropped_count > 0:
                 logger.debug(f"ℹ️ [Strategy {self.symbol}] تم إسقاط {dropped_count} صفًا بسبب قيم NaN في المؤشرات.")
            if df_cleaned.empty:
                logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame فارغ بعد إزالة قيم NaN للمؤشرات.")
                return None

            latest = df_cleaned.iloc[-1]
            logger.debug(f"✅ [Strategy {self.symbol}] تم حساب المؤشرات. أحدث EMA{EMA_SHORT_PERIOD}: {latest.get(f'ema_{EMA_SHORT_PERIOD}', np.nan):.4f}, EMA{EMA_LONG_PERIOD}: {latest.get(f'ema_{EMA_LONG_PERIOD}', np.nan):.4f}, VWMA: {latest.get('vwma', np.nan):.4f}, MACD Hist: {latest.get('macd_hist', np.nan):.4f}, SuperTrend: {latest.get('supertrend', np.nan):.4f}")
            return df_cleaned

        except KeyError as ke:
             logger.error(f"❌ [Strategy {self.symbol}] خطأ: لم يتم العثور على عمود مطلوب أثناء حساب المؤشر: {ke}", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"❌ [Strategy {self.symbol}] خطأ غير متوقع أثناء حساب المؤشر: {e}", exc_info=True)
            return None


    def generate_buy_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Generates a buy signal based on the processed DataFrame.
        If the ML model predicts bullish, it overrides all other essential conditions.
        """
        logger.debug(f"ℹ️ [Strategy {self.symbol}] إنشاء إشارة شراء...")

        min_signal_data_len = max(RECENT_EMA_CROSS_LOOKBACK, MACD_HIST_INCREASE_CANDLES, OBV_INCREASE_CANDLES) + 1
        if df_processed is None or df_processed.empty or len(df_processed) < min_signal_data_len:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame فارغ أو قصير جدًا (<{min_signal_data_len})، لا يمكن إنشاء إشارة.")
            return None

        # Ensure all required columns for signal generation, including ML features, are present
        required_cols_for_signal = list(set(self.feature_columns_for_ml + [
            'close', 'rsi', 'atr', 'bb_upper', 'BullishCandleSignal'
        ]))
        missing_cols = [col for col in required_cols_for_signal if col not in df_processed.columns]
        if missing_cols:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame يفتقد أعمدة مطلوبة للإشارة: {missing_cols}.")
            return None

        last_row = df_processed.iloc[-1]
        recent_df = df_processed.iloc[-min_signal_data_len:]

        if recent_df[required_cols_for_signal].isnull().values.any():
             logger.warning(f"⚠️ [Strategy {self.symbol}] البيانات الحديثة تحتوي على قيم NaN في أعمدة الإشارة المطلوبة. لا يمكن إنشاء إشارة.")
             return None

        # --- ML Model Prediction (NEW: Mandatory Override) ---
        ml_overrides_essentials = False
        ml_prediction_result_text = "N/A (نموذج غير محمل)"
        if ml_model:
            try:
                features_for_prediction = pd.DataFrame([last_row[self.feature_columns_for_ml].values], columns=self.feature_columns_for_ml)
                ml_pred = ml_model.predict(features_for_prediction)[0]
                if ml_pred == 1: # If ML model predicts upward movement
                    ml_overrides_essentials = True
                    ml_prediction_result_text = 'صعودي (تجاوز الشروط الأساسية) ✅'
                    logger.info(f"✨ [Strategy {self.symbol}] تنبؤ نموذج ML صعودي. تجاوز الشروط الأساسية الأخرى.")
                else:
                    ml_prediction_result_text = 'هابط (لا يوجد تجاوز)'
            except Exception as ml_err:
                logger.error(f"❌ [Strategy {self.symbol}] خطأ في تنبؤ نموذج ML: {ml_err}", exc_info=True)
                ml_prediction_result_text = "خطأ في التنبؤ (0)"
        signal_details = {'ML_Prediction': ml_prediction_result_text} # Initialize signal_details with ML prediction

        # --- Check BTC Trend (Still applies even with ML override, as a general market filter) ---
        btc_trend = get_btc_trend_4h()
        if "هبوط" in btc_trend:
            logger.info(f"ℹ️ [Strategy {self.symbol}] التداول متوقف بسبب اتجاه البيتكوين الهابط ({btc_trend}).")
            signal_details['BTC_Trend'] = f'هبوط ({btc_trend}) ❌'
            return None
        elif "N/A" in btc_trend:
             logger.warning(f"⚠️ [Strategy {self.symbol}] لا يمكن تحديد اتجاه البيتكوين، سيتم تجاهل هذا الشرط.")
             signal_details['BTC_Trend'] = 'غير متاح (تجاهل)'
        else:
             signal_details['BTC_Trend'] = f'صعود أو استقرار ({btc_trend}) ✅'


        # --- Check Mandatory Conditions (Bypassed if ML overrides) ---
        essential_passed = True
        failed_essential_conditions = []

        if not ml_overrides_essentials: # Only check if ML model did NOT override
            if not (pd.notna(last_row[f'ema_{EMA_SHORT_PERIOD}']) and pd.notna(last_row[f'ema_{EMA_LONG_PERIOD}']) and pd.notna(last_row['vwma']) and
                    last_row['close'] > last_row[f'ema_{EMA_SHORT_PERIOD}'] and
                    last_row['close'] > last_row[f'ema_{EMA_LONG_PERIOD}'] and
                    last_row['close'] > last_row['vwma']):
                essential_passed = False
                failed_essential_conditions.append('Price Above EMAs and VWMA')
                detail_ma = f"Close:{last_row['close']:.4f}, EMA{EMA_SHORT_PERIOD}:{last_row[f'ema_{EMA_SHORT_PERIOD}']:.4f}, EMA{EMA_LONG_PERIOD}:{last_row[f'ema_{EMA_LONG_PERIOD}']:.4f}, VWMA:{last_row['vwma']:.4f}"
                signal_details['Price_MA_Alignment'] = f'فشل: السعر ليس فوق جميع المتوسطات ({detail_ma})'
            else:
                signal_details['Price_MA_Alignment'] = f'نجاح: السعر فوق جميع المتوسطات'

            if not (pd.notna(last_row[f'ema_{EMA_SHORT_PERIOD}']) and pd.notna(last_row[f'ema_{EMA_LONG_PERIOD}']) and
                    last_row[f'ema_{EMA_SHORT_PERIOD}'] > last_row[f'ema_{EMA_LONG_PERIOD}']):
                 essential_passed = False
                 failed_essential_conditions.append('Short EMA Above Long EMA')
                 detail_ema_cross = f"EMA{EMA_SHORT_PERIOD}:{last_row[f'ema_{EMA_SHORT_PERIOD}']:.4f}, EMA{EMA_LONG_PERIOD}:{last_row[f'ema_{EMA_LONG_PERIOD}']:.4f}"
                 signal_details['EMA_Order'] = f'فشل: EMA القصير ليس فوق EMA الطويل ({detail_ema_cross})'
            else:
                 signal_details['EMA_Order'] = f'نجاح: EMA القصير فوق EMA الطويل'

            if not (pd.notna(last_row['supertrend']) and last_row['close'] > last_row['supertrend'] and last_row['supertrend_trend'] == 1):
                 essential_passed = False
                 failed_essential_conditions.append('SuperTrend (اتجاه صعودي والسعر فوقه)')
                 detail_st = f'ST:{last_row.get("supertrend", np.nan):.4f}, Trend:{last_row.get("supertrend_trend", 0)}'
                 signal_details['SuperTrend'] = f'فشل: ليس اتجاه صعودي أو السعر ليس فوقه ({detail_st})'
            else:
                signal_details['SuperTrend'] = f'نجاح: اتجاه صعودي والسعر فوقه'

            if not (pd.notna(last_row['macd_hist']) and pd.notna(last_row['macd']) and pd.notna(last_row['macd_signal']) and (last_row['macd_hist'] > 0 or last_row['macd'] > last_row['macd_signal'])):
                 essential_passed = False
                 failed_essential_conditions.append('MACD (هيستوجرام إيجابي أو تقاطع صعودي)')
                 detail_macd = f'Hist: {last_row.get("macd_hist", np.nan):.4f}, MACD: {last_row.get("macd", np.nan):.4f}, Signal: {last_row.get("macd_signal", np.nan):.4f}'
                 signal_details['MACD'] = f'فشل: ليس هيستوجرام إيجابي ولا يوجد تقاطع صعودي ({detail_macd})'
            else:
                 detail_macd = f'Hist > 0 ({last_row["macd_hist"]:.4f})' if last_row['macd_hist'] > 0 else ''
                 detail_macd += ' & ' if detail_macd and last_row['macd'] > last_row['macd_signal'] else ''
                 detail_macd += 'تقاطع صعودي' if last_row['macd'] > last_row['macd_signal'] else ''
                 signal_details['MACD'] = f'نجاح: {detail_macd}'

            if not (pd.notna(last_row['adx']) and pd.notna(last_row['di_plus']) and pd.notna(last_row['di_minus']) and last_row['adx'] > MIN_ADX_TREND_STRENGTH and last_row['di_plus'] > last_row['di_minus']):
                 essential_passed = False
                 failed_essential_conditions.append(f'ADX/DI (اتجاه صعودي قوي، ADX > {MIN_ADX_TREND_STRENGTH})')
                 detail_adx = f'ADX:{last_row.get("adx", np.nan):.1f}, DI+:{last_row.get("di_plus", np.nan):.1f}, DI-:{last_row.get("di_minus", np.nan):.1f}'
                 signal_details['ADX/DI'] = f'فشل: ليس اتجاه صعودي قوي (ADX <= {MIN_ADX_TREND_STRENGTH} أو DI+ <= DI-) ({detail_adx})'
            else:
                 signal_details['ADX/DI'] = f'نجاح: اتجاه صعودي قوي (ADX:{last_row["adx"]:.1f}, DI+>DI-)'

        # If ML did not override AND essential conditions failed, then no signal
        if not essential_passed and not ml_overrides_essentials:
            logger.debug(f"ℹ️ [Strategy {self.symbol}] فشلت الشروط الإلزامية (ولم يتم تجاوزها بواسطة ML): {', '.join(failed_essential_conditions)}. تم رفض الإشارة.")
            return None
        elif ml_overrides_essentials:
            logger.info(f"✅ [Strategy {self.symbol}] تم تجاوز الشروط الأساسية بواسطة تنبؤ ML الصعودي.")
            # If ML overrides, set essential conditions to "ML Override" in details for clarity
            for cond_name in self.essential_conditions:
                if cond_name not in signal_details:
                    signal_details[cond_name.replace(' ', '_')] = 'تم تجاوزها بواسطة ML' # Add a placeholder for overridden conditions


        # --- Calculate Score for Optional Conditions ---
        current_score = 0.0

        if pd.notna(last_row['vwap']) and last_row['close'] > last_row['vwap']:
            current_score += self.condition_weights.get('above_vwap', 0)
            signal_details['VWAP_Daily'] = f'فوق VWAP اليومي (+{self.condition_weights.get("above_vwap", 0)})'
        else:
             signal_details['VWAP_Daily'] = f'تحت VWAP اليومي (0)'

        if pd.notna(last_row['rsi']) and last_row['rsi'] < RSI_OVERBOUGHT and last_row['rsi'] > RSI_OVERSOLD:
            current_score += self.condition_weights.get('rsi_ok', 0)
            signal_details['RSI_Basic'] = f'مقبول ({RSI_OVERSOLD}<{last_row["rsi"]:.1f}<{RSI_OVERBOUGHT}) (+{self.condition_weights.get("rsi_ok", 0)})'
        else:
             signal_details['RSI_Basic'] = f'RSI ({last_row["rsi"]:.1f}) غير مقبول (0)'

        if last_row.get('BullishCandleSignal', 0) == 1:
            current_score += self.condition_weights.get('bullish_candle', 0)
            signal_details['Candle'] = f'نمط شمعة صعودي (+{self.condition_weights.get("bullish_candle", 0)})'
        else:
             signal_details['Candle'] = f'لا يوجد نمط شمعة صعودي (0)'

        if pd.notna(last_row['bb_upper']) and last_row['close'] < last_row['bb_upper'] * 0.995:
             current_score += self.condition_weights.get('not_bb_extreme', 0)
             signal_details['Bollinger_Basic'] = f'ليس عند الحد العلوي لبولينجر (+{self.condition_weights.get("not_bb_extreme", 0)})'
        else:
             signal_details['Bollinger_Basic'] = f'عند أو فوق الحد العلوي لبولينجر (0)'

        if len(df_processed) >= 2 and pd.notna(df_processed.iloc[-2]['obv']) and pd.notna(last_row['obv']) and last_row['obv'] > df_processed.iloc[-2]['obv']:
            current_score += self.condition_weights.get('obv_rising', 0)
            signal_details['OBV_Last'] = f'يرتفع في الشمعة الأخيرة (+{self.condition_weights.get("obv_rising", 0)})'
        else:
             signal_details['OBV_Last'] = f'لا يرتفع في الشمعة الأخيرة (0)'

        if pd.notna(last_row['rsi']) and last_row['rsi'] >= 50 and last_row['rsi'] <= 80:
             current_score += self.condition_weights.get('rsi_filter_breakout', 0)
             signal_details['RSI_Filter_Breakout'] = f'RSI ({last_row["rsi"]:.1f}) في النطاق الصعودي (50-80) (+{self.condition_weights.get("rsi_filter_breakout", 0)})'
        else:
             signal_details['RSI_Filter_Breakout'] = f'RSI ({last_row["rsi"]:.1f}) ليس في النطاق الصعودي (0)'

        if pd.notna(last_row['macd_hist']) and last_row['macd_hist'] > 0:
             current_score += self.condition_weights.get('macd_filter_breakout', 0)
             signal_details['MACD_Filter_Breakout'] = f'هيستوجرام MACD إيجابي ({last_row["macd_hist"]:.4f}) (+{self.condition_weights.get("macd_filter_breakout", 0)})'
        else:
             signal_details['MACD_Filter_Breakout'] = f'هيستوجرام MACD ليس إيجابي (0)'

        macd_hist_increasing = False
        if len(recent_df) >= MACD_HIST_INCREASE_CANDLES + 1:
             macd_hist_slice = recent_df['macd_hist'].iloc[-MACD_HIST_INCREASE_CANDLES-1:]
             if not macd_hist_slice.isnull().any() and np.all(np.diff(macd_hist_slice) > 0):
                  macd_hist_increasing = True
        if macd_hist_increasing:
             current_score += self.condition_weights.get('macd_hist_increasing', 0)
             signal_details['MACD_Hist_Increasing'] = f'هيستوجرام MACD يتزايد خلال آخر {MACD_HIST_INCREASE_CANDLES} شمعات (+{self.condition_weights.get("macd_hist_increasing", 0)})'
        else:
             signal_details['MACD_Hist_Increasing'] = f'هيستوجرام MACD لا يتزايد خلال آخر {MACD_HIST_INCREASE_CANDLES} شمعات (0)'

        obv_increasing_recent = False
        if len(recent_df) >= OBV_INCREASE_CANDLES + 1:
             obv_slice = recent_df['obv'].iloc[-OBV_INCREASE_CANDLES-1:]
             if not obv_slice.isnull().any() and np.all(np.diff(obv_slice) > 0):
                  obv_increasing_recent = True
        if obv_increasing_recent:
             current_score += self.condition_weights.get('obv_increasing_recent', 0)
             signal_details['OBV_Increasing_Recent'] = f'OBV يتزايد خلال آخر {OBV_INCREASE_CANDLES} شمعات (+{self.condition_weights.get("obv_increasing_recent", 0)})'
        else:
             signal_details['OBV_Increasing_Recent'] = f'OBV لا يتزايد خلال آخر {OBV_INCREASE_CANDLES} شمعات (0)'


        if current_score < self.min_signal_score and not ml_overrides_essentials: # Score check still applies if ML didn't override
            logger.debug(f"ℹ️ [Strategy {self.symbol}] لم يتم استيفاء درجة الإشارة المطلوبة من الشروط الاختيارية (الدرجة: {current_score:.2f} / {self.total_possible_score:.2f}, الحد الأدنى: {self.min_signal_score:.2f}). تم رفض الإشارة.")
            return None

        volume_recent = fetch_recent_volume(self.symbol)
        if volume_recent < MIN_VOLUME_15M_USDT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] السيولة ({volume_recent:,.0f} USDT) أقل من الحد الأدنى المطلوب ({MIN_VOLUME_15M_USDT:,.0f} USDT). تم رفض الإشارة.")
            return None

        current_price = last_row['close']
        current_atr = last_row.get('atr')

        if pd.isna(current_atr) or current_atr <= 0:
             logger.warning(f"⚠️ [Strategy {self.symbol}] قيمة ATR غير صالحة ({current_atr}) لحساب الهدف.")
             return None

        target_multiplier = ENTRY_ATR_MULTIPLIER
        initial_target = current_price + (target_multiplier * current_atr)

        profit_margin_pct = ((initial_target / current_price) - 1) * 100 if current_price > 0 else 0
        if profit_margin_pct < MIN_PROFIT_MARGIN_PCT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] هامش الربح ({profit_margin_pct:.2f}%) أقل من الحد الأدنى المطلوب ({MIN_PROFIT_MARGIN_PCT:.2f}%). تم رفض الإشارة.")
            return None

        signal_output = {
            'symbol': self.symbol,
            'entry_price': float(f"{current_price:.8g}"),
            'initial_target': float(f"{initial_target:.8g}"),
            'current_target': float(f"{initial_target:.8g}"),
            'r2_score': float(f"{current_score:.2f}"),
            'strategy_name': 'Scalping_Momentum_Trend_ML_Override', # Updated strategy name
            'signal_details': signal_details,
            'volume_15m': volume_recent,
            'trade_value': TRADE_VALUE,
            'total_possible_score': float(f"{self.total_possible_score:.2f}")
        }

        logger.info(f"✅ [Strategy {self.symbol}] تم تأكيد إشارة الشراء. السعر: {current_price:.6f}, الدرجة (اختياري): {current_score:.2f}/{self.total_possible_score:.2f}, ATR: {current_atr:.6f}, الحجم: {volume_recent:,.0f}, تنبؤ ML: {ml_prediction_result_text}")
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
            payload['reply_markup'] = json.dumps(convert_np_values(reply_markup))
        except (TypeError, ValueError) as json_err:
             logger.error(f"❌ [Telegram] فشل تحويل reply_markup إلى JSON: {json_err} - Markup: {reply_markup}")
             return None

    logger.debug(f"ℹ️ [Telegram] إرسال رسالة إلى {target_chat_id}...")
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        logger.info(f"✅ [Telegram] تم إرسال الرسالة بنجاح إلى {target_chat_id}.")
        return response.json()
    except requests.exceptions.Timeout:
         logger.error(f"❌ [Telegram] فشل إرسال الرسالة إلى {target_chat_id} (مهلة).")
         return None
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة إلى {target_chat_id} (خطأ HTTP: {http_err.response.status_code}).")
        try:
            error_details = http_err.response.json()
            logger.error(f"❌ [Telegram] تفاصيل خطأ API: {error_details}")
        except json.JSONDecodeError:
            logger.error(f"❌ [Telegram] تعذر فك تشفير استجابة الخطأ: {http_err.response.text}")
        return None
    except requests.exceptions.RequestException as req_err:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة إلى {target_chat_id} (خطأ في الطلب): {req_err}")
        return None
    except Exception as e:
         logger.error(f"❌ [Telegram] خطأ غير متوقع أثناء إرسال الرسالة: {e}", exc_info=True)
         return None

def send_telegram_alert(signal_data: Dict[str, Any], timeframe: str) -> None:
    """Formats and sends a new trading signal alert to Telegram in Arabic, displaying the score."""
    logger.debug(f"ℹ️ [Telegram Alert] تنسيق وإرسال تنبيه للإشارة: {signal_data.get('symbol', 'N/A')}")
    try:
        entry_price = float(signal_data['entry_price'])
        target_price = float(signal_data['initial_target'])
        symbol = signal_data['symbol']
        strategy_name = signal_data.get('strategy_name', 'N/A')
        signal_score = signal_data.get('r2_score', 0.0)
        total_possible_score = signal_data.get('total_possible_score', 10.0)
        volume_15m = signal_data.get('volume_15m', 0.0)
        trade_value_signal = signal_data.get('trade_value', TRADE_VALUE)
        signal_details = signal_data.get('signal_details', {})

        profit_pct = ((target_price / entry_price) - 1) * 100 if entry_price > 0 else 0

        entry_fee = trade_value_signal * BINANCE_FEE_RATE
        exit_value = trade_value_signal * (1 + profit_pct / 100.0)
        exit_fee = exit_value * BINANCE_FEE_RATE
        total_trade_fees = entry_fee + exit_fee

        profit_usdt_gross = trade_value_signal * (profit_pct / 100)
        profit_usdt_net = profit_usdt_gross - total_trade_fees

        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')

        fear_greed = get_fear_greed_index()
        btc_trend = signal_details.get('BTC_Trend', 'N/A') # Get BTC trend from signal_details
        ml_prediction_status = signal_details.get('ML_Prediction', 'N/A') # Get ML prediction status

        message = (
            f"💡 *إشارة تداول جديدة ({strategy_name.replace('_', ' ').title()})* 💡\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"📈 **نوع الإشارة:** شراء (طويل)\n"
            f"🕰️ **الإطار الزمني:** {timeframe}\n"
            f"📊 **قوة الإشارة (النقاط - اختيارية):** *{signal_score:.1f} / {total_possible_score:.1f}*\n"
            f"💧 **السيولة (15 دقيقة):** {volume_15m:,.0f} USDT\n"
            f"——————————————\n"
            f"➡️ **سعر الدخول المقترح:** `${entry_price:,.8g}`\n"
            f"🎯 **الهدف الأولي:** `${target_price:,.8g}`\n"
            f"💰 **الربح المتوقع (إجمالي):** ({profit_pct:+.2f}% / ≈ ${profit_usdt_gross:+.2f})\n"
            f"💸 **الرسوم المتوقعة:** ${total_trade_fees:,.2f}\n"
            f"📈 **الربح الصافي المتوقع:** ${profit_usdt_net:+.2f}\n"
            f"——————————————\n"
            f"🤖 *تنبؤ نموذج ML:* *{ml_prediction_status}*\n" # Highlight ML prediction
            f"✅ *الشروط الإلزامية المحققة:*\n"
            f"  - السعر فوق المتوسطات (EMA{EMA_SHORT_PERIOD}, EMA{EMA_LONG_PERIOD}, VWMA): {signal_details.get('Price_MA_Alignment', 'N/A')}\n"
            f"  - المتوسط القصير فوق الطويل (EMA{EMA_SHORT_PERIOD} > EMA{EMA_LONG_PERIOD}): {signal_details.get('EMA_Order', 'N/A')}\n"
            f"  - سوبر ترند: {signal_details.get('SuperTrend', 'N/A')}\n"
            f"  - ماكد: {signal_details.get('MACD', 'N/A')}\n"
            f"  - مؤشر الاتجاه (ADX/DI): {signal_details.get('ADX/DI', 'N/A')}\n"
            f"——————————————\n"
            f"✨ *شروط النقاط الإضافية (الاختيارية):*\n"
            f"  - فوق متوسط الحجم الموزون اليومي (VWAP): {signal_details.get('VWAP_Daily', 'N/A')}\n"
            f"  - مؤشر القوة النسبية (RSI): {signal_details.get('RSI_Basic', 'N/A')}\n"
            f"  - نمط شمعة صعودي: {signal_details.get('Candle', 'N/A')}\n"
            f"  - ليس عند الحد العلوي لبولينجر: {signal_details.get('Bollinger_Basic', 'N/A')}\n"
            f"  - حجم التوازن (OBV) يرتفع (شمعة أخيرة): {signal_details.get('OBV_Last', 'N/A')}\n"
            f"  - فلتر RSI للاختراق: {signal_details.get('RSI_Filter_Breakout', 'N/A')}\n"
            f"  - فلتر MACD للاختراق: {signal_details.get('MACD_Filter_Breakout', 'N/A')}\n"
            f"  - هيستوجرام MACD يتزايد ({MACD_HIST_INCREASE_CANDLES} شمعات): {signal_details.get('MACD_Hist_Increasing', 'N/A')}\n"
            f"  - حجم التوازن (OBV) يتزايد مؤخراً ({OBV_INCREASE_CANDLES} شمعات): {signal_details.get('OBV_Increasing_Recent', 'N/A')}\n"
            f"——————————————\n"
            f"😨/🤑 **مؤشر الخوف والجشع:** {fear_greed}\n"
            f"₿ **اتجاه البيتكوين (4 ساعات):** {btc_trend}\n"
            f"——————————————\n"
            f"⏰ {timestamp_str}"
        )

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
    safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')
    closing_price = details.get('closing_price', 0.0)
    profit_pct = details.get('profit_pct', 0.0)
    current_price = details.get('current_price', 0.0)
    time_to_target = details.get('time_to_target', 'N/A')
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
            f"⏱️ **الوقت المستغرق:** {time_to_target}"
        )
    elif notification_type == 'target_updated':
         message = (
             f"↗️ *تم تحديث الهدف (ID: {signal_id})*\n"
             f"——————————————\n"
             f"🪙 **الزوج:** `{safe_symbol}`\n"
             f"📈 **السعر الحالي:** `${current_price:,.8g}`\n"
             f"🎯 **الهدف السابق:** `${old_target:,.8g}`\n"
             f"🎯 **الهدف الجديد:** `${new_target:,.8g}`\n"
             f"ℹ️ *تم التحديث بناءً على استمرار الزخم الصعودي.*"
         )
    else:
        logger.warning(f"⚠️ [Notification] نوع إشعار غير معروف: {notification_type} للتفاصيل: {details}")
        return

    if message:
        send_telegram_message(CHAT_ID, message, parse_mode='Markdown')

# ---------------------- Database Functions (Insert and Update) ----------------------
def insert_signal_into_db(signal: Dict[str, Any]) -> bool:
    """Inserts a new signal into the signals table with the weighted score and entry time."""
    if not check_db_connection() or not conn:
        logger.error(f"❌ [DB Insert] فشل إدراج الإشارة {signal.get('symbol', 'N/A')} بسبب مشكلة في اتصال قاعدة البيانات.")
        return False

    symbol = signal.get('symbol', 'N/A')
    logger.debug(f"ℹ️ [DB Insert] محاولة إدراج إشارة لـ {symbol}...")
    try:
        signal_prepared = convert_np_values(signal)
        signal_details_json = json.dumps(signal_prepared.get('signal_details', {}))

        with conn.cursor() as cur_ins:
            insert_query = sql.SQL("""
                INSERT INTO signals
                 (symbol, entry_price, initial_target, current_target,
                 r2_score, strategy_name, signal_details, volume_15m, entry_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW());
            """)
            cur_ins.execute(insert_query, (
                signal_prepared['symbol'],
                signal_prepared['entry_price'],
                signal_prepared['initial_target'],
                signal_prepared['current_target'],
                signal_prepared.get('r2_score'),
                signal_prepared.get('strategy_name', 'unknown'),
                signal_details_json,
                signal_prepared.get('volume_15m')
            ))
        conn.commit()
        logger.info(f"✅ [DB Insert] تم إدراج إشارة لـ {symbol} في قاعدة البيانات (الدرجة: {signal_prepared.get('r2_score')}).")
        return True
    except psycopg2.Error as db_err:
        logger.error(f"❌ [DB Insert] خطأ في قاعدة البيانات أثناء إدراج إشارة لـ {symbol}: {db_err}")
        if conn: conn.rollback()
        return False
    except (TypeError, ValueError) as convert_err:
         logger.error(f"❌ [DB Insert] خطأ في تحويل بيانات الإشارة قبل الإدراج لـ {symbol}: {convert_err} - بيانات الإشارة: {signal}")
         if conn: conn.rollback()
         return False
    except Exception as e:
        logger.error(f"❌ [DB Insert] خطأ غير متوقع أثناء إدراج إشارة لـ {symbol}: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

# ---------------------- Open Signal Tracking Function ----------------------
def track_signals() -> None:
    """Tracks open signals and checks targets. Calculates time to target upon hit."""
    logger.info("ℹ️ [Tracker] بدء عملية تتبع الإشارات المفتوحة...")
    while True:
        active_signals_summary: List[str] = []
        processed_in_cycle = 0
        try:
            if not check_db_connection() or not conn:
                logger.warning("⚠️ [Tracker] تخطي دورة التتبع بسبب مشكلة في اتصال قاعدة البيانات.")
                time.sleep(15)
                continue

            with conn.cursor() as track_cur:
                 track_cur.execute("""
                    SELECT id, symbol, entry_price, initial_target, current_target, entry_time
                    FROM signals
                    WHERE achieved_target = FALSE;
                """)
                 open_signals: List[Dict] = track_cur.fetchall()

            if not open_signals:
                time.sleep(10)
                continue

            logger.debug(f"ℹ️ [Tracker] تتبع {len(open_signals)} إشارة مفتوحة...")

            for signal_row in open_signals:
                signal_id = signal_row['id']
                symbol = signal_row['symbol']
                processed_in_cycle += 1
                update_executed = False

                try:
                    entry_price = float(signal_row['entry_price'])
                    entry_time = signal_row['entry_time']
                    current_target = float(signal_row['current_target'])

                    current_price = ticker_data.get(symbol)

                    if current_price is None:
                         logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): السعر الحالي غير متاح في بيانات التيكر.")
                         continue

                    active_signals_summary.append(f"{symbol}({signal_id}): P={current_price:.4f} T={current_target:.4f}")

                    update_query: Optional[sql.SQL] = None
                    update_params: Tuple = ()
                    log_message: Optional[str] = None
                    notification_details: Dict[str, Any] = {'symbol': symbol, 'id': signal_id, 'current_price': current_price}


                    # --- Check and Update Logic ---
                    # 1. Check for Target Hit
                    if current_price >= current_target:
                        profit_pct = ((current_target / entry_price) - 1) * 100 if entry_price > 0 else 0
                        closed_at = datetime.now()
                        time_to_target_duration = closed_at - entry_time if entry_time else timedelta(0)
                        time_to_target_str = str(time_to_target_duration)

                        update_query = sql.SQL("UPDATE signals SET achieved_target = TRUE, closing_price = %s, closed_at = %s, profit_percentage = %s, time_to_target = %s WHERE id = %s;")
                        update_params = (current_target, closed_at, profit_pct, time_to_target_duration, signal_id)
                        log_message = f"🎯 [Tracker] {symbol}(ID:{signal_id}): تم الوصول إلى الهدف عند {current_target:.8g} (الربح: {profit_pct:+.2f}%, الوقت: {time_to_target_str})."
                        notification_details.update({'type': 'target_hit', 'closing_price': current_target, 'profit_pct': profit_pct, 'time_to_target': time_to_target_str})
                        update_executed = True

                    # 2. Check for Target Extension (Only if Target not hit)
                    if not update_executed:
                        if current_price >= current_target * (1 - TARGET_APPROACH_THRESHOLD_PCT):
                             logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): السعر قريب من الهدف ({current_price:.8g} مقابل {current_target:.8g}). التحقق من إشارة الاستمرار...")

                             df_continuation = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)

                             if df_continuation is not None and not df_continuation.empty:
                                 continuation_strategy = ScalpingTradingStrategy(symbol)
                                 df_continuation_indicators = continuation_strategy.populate_indicators(df_continuation)

                                 if df_continuation_indicators is not None:
                                     continuation_signal = continuation_strategy.generate_buy_signal(df_continuation_indicators)

                                     if continuation_signal:
                                         latest_row = df_continuation_indicators.iloc[-1]
                                         current_atr_for_new_target = latest_row.get('atr')

                                         if pd.notna(current_atr_for_new_target) and current_atr_for_new_target > 0:
                                             potential_new_target = current_price + (ENTRY_ATR_MULTIPLIER * current_atr_for_new_target)

                                             if potential_new_target > current_target:
                                                 old_target = current_target
                                                 new_target = potential_new_target
                                                 update_query = sql.SQL("UPDATE signals SET current_target = %s WHERE id = %s;")
                                                 update_params = (new_target, signal_id)
                                                 log_message = f"↗️ [Tracker] {symbol}(ID:{signal_id}): تم تحديث الهدف من {old_target:.8g} إلى {new_target:.8g} بناءً على إشارة الاستمرار."
                                                 notification_details.update({'type': 'target_updated', 'old_target': old_target, 'new_target': new_target})
                                                 update_executed = True
                                             else:
                                                 logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): تم اكتشاف إشارة استمرار، لكن الهدف الجديد ({potential_new_target:.8g}) ليس أعلى من الهدف الحالي ({current_target:.8g}). عدم تحديث الهدف.")
                                         else:
                                             logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): لا يمكن حساب الهدف الجديد بسبب ATR غير صالح ({current_atr_for_new_target}) من بيانات الاستمرار.")
                                     else:
                                         logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): السعر قريب من الهدف، ولكن لم يتم إنشاء إشارة استمرار.")
                                 else:
                                     logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): فشل في ملء المؤشرات للتحقق من الاستمرار.")
                             else:
                                 logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): لا يمكن جلب البيانات التاريخية للتحقق من الاستمرار.")


                    if update_executed and update_query:
                        try:
                             with conn.cursor() as update_cur:
                                  update_cur.execute(update_query, update_params)
                             conn.commit()
                             if log_message: logger.info(log_message)
                             if notification_details.get('type'):
                                send_tracking_notification(notification_details)
                        except psycopg2.Error as db_err:
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ في قاعدة البيانات أثناء التحديث: {db_err}")
                            if conn: conn.rollback()
                        except Exception as exec_err:
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ غير متوقع أثناء تنفيذ التحديث/الإشعار: {exec_err}", exc_info=True)
                            if conn: conn.rollback()

                except (TypeError, ValueError) as convert_err:
                    logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ في تحويل قيم الإشارة الأولية: {convert_err}")
                    continue
                except Exception as inner_loop_err:
                     logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ غير متوقع أثناء معالجة الإشارة: {inner_loop_err}", exc_info=True)
                     continue

            if active_signals_summary:
                logger.debug(f"ℹ️ [Tracker] نهاية حالة الدورة ({processed_in_cycle} معالجة): {'; '.join(active_signals_summary)}")

            time.sleep(3)

        except psycopg2.Error as db_cycle_err:
             logger.error(f"❌ [Tracker] خطأ في قاعدة البيانات في دورة التتبع الرئيسية: {db_cycle_err}. محاولة إعادة الاتصال...")
             if conn: conn.rollback()
             time.sleep(30)
             check_db_connection()
        except Exception as cycle_err:
            logger.error(f"❌ [Tracker] خطأ غير متوقع في دورة تتبع الإشارة: {cycle_err}", exc_info=True)
            time.sleep(30)

def get_interval_minutes(interval: str) -> int:
    """Helper function to convert Binance interval string to minutes."""
    if interval.endswith('m'):
        return int(interval[:-1])
    elif interval.endswith('h'):
        return int(interval[:-1]) * 60
    elif interval.endswith('d'):
        return int(interval[:-1]) * 60 * 24
    return 0


# ---------------------- Flask Service (Optional for Webhook) ----------------------
app = Flask(__name__)

@app.route('/')
def home() -> Response:
    """Simple home page to show the bot is running."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ws_alive = ws_thread.is_alive() if 'ws_thread' in globals() and ws_thread else False
    tracker_alive = tracker_thread.is_alive() if 'tracker_thread' in globals() and tracker_thread else False
    main_bot_alive = main_bot_thread.is_alive() if 'main_bot_thread' in globals() and main_bot_thread else False
    status = "running" if ws_alive and tracker_alive and main_bot_alive else "partially running"
    return Response(f"📈 Crypto Signal Bot ({status}) - Last Check: {now}", status=200, mimetype='text/plain')

@app.route('/favicon.ico')
def favicon() -> Response:
    """Handles favicon request to avoid 404 errors in logs."""
    return Response(status=204)

@app.route('/webhook', methods=['POST'])
def webhook() -> Tuple[str, int]:
    """Handles incoming requests from Telegram (like button presses and commands)."""
    # Only process webhook if WEBHOOK_URL is configured
    if not WEBHOOK_URL:
        logger.warning("⚠️ [Flask] تم استلام طلب webhook، ولكن WEBHOOK_URL غير مهيأ. تجاهل الطلب.")
        return "Webhook not configured", 200 # Return OK to Telegram to avoid repeated attempts

    if not request.is_json:
        logger.warning("⚠️ [Flask] تم استلام طلب webhook غير JSON.")
        return "Invalid request format", 400

    try:
        data = request.get_json()
        logger.debug(f"ℹ️ [Flask] تم استلام بيانات webhook: {json.dumps(data)[:200]}...")

        if 'callback_query' in data:
            callback_query = data['callback_query']
            callback_id = callback_query['id']
            callback_data = callback_query.get('data')
            message_info = callback_query.get('message')
            if not message_info or not callback_data:
                 logger.warning(f"⚠️ [Flask] استعلام رد الاتصال (ID: {callback_id}) يفتقد الرسالة أو البيانات.")
                 try:
                     ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                     requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
                 except Exception as ack_err:
                     logger.warning(f"⚠️ [Flask] فشل تأكيد استعلام رد الاتصال غير الصالح {callback_id}: {ack_err}")
                 return "OK", 200
            chat_id_callback = message_info.get('chat', {}).get('id')
            if not chat_id_callback:
                 logger.warning(f"⚠️ [Flask] استعلام رد الاتصال (ID: {callback_id}) يفتقد معرف الدردشة.")
                 try:
                     ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                     requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
                 except Exception as ack_err:
                     logger.warning(f"⚠️ [Flask] فشل تأكيد استعلام رد الاتصال غير الصالح {callback_id}: {ack_err}")
                 return "OK", 200


            message_id = message_info['message_id']
            user_info = callback_query.get('from', {})
            user_id = user_info.get('id')
            username = user_info.get('username', 'N/A')

            logger.info(f"ℹ️ [Flask] تم استلام استعلام رد الاتصال: البيانات='{callback_data}', المستخدم={username}({user_id}), الدردشة={chat_id_callback}")

            try:
                ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
            except Exception as ack_err:
                 logger.warning(f"⚠️ [Flask] فشل تأكيد استعلام رد الاتصال {callback_id}: {ack_err}")

            if callback_data == "get_report":
                report_thread = Thread(target=lambda: send_telegram_message(chat_id_callback, generate_performance_report(), parse_mode='Markdown'))
                report_thread.start()
            else:
                logger.warning(f"⚠️ [Flask] تم استلام بيانات رد اتصال غير معالجة: '{callback_data}'")


        elif 'message' in data:
            message_data = data['message']
            chat_info = message_data.get('chat')
            user_info = message_data.get('from', {})
            text_msg = message_data.get('text', '').strip()

            if not chat_info or not text_msg:
                 logger.debug("ℹ️ [Flask] تم استلام رسالة بدون معلومات الدردشة أو النص.")
                 return "OK", 200

            chat_id_msg = chat_info['id']
            user_id = user_info.get('id')
            username = user_info.get('username', 'N/A')

            logger.info(f"ℹ️ [Flask] تم استلام رسالة: النص='{text_msg}', المستخدم={username}({user_id}), الدردشة={chat_id_msg}")

            if text_msg.lower() == '/report':
                 report_thread = Thread(target=lambda: send_telegram_message(chat_id_msg, generate_performance_report(), parse_mode='Markdown'))
                 report_thread.start()
            elif text_msg.lower() == '/status':
                 status_thread = Thread(target=handle_status_command, args=(chat_id_msg,))
                 status_thread.start()

        else:
            logger.debug("ℹ️ [Flask] تم استلام بيانات webhook بدون 'callback_query' أو 'message'.")

        return "OK", 200
    except Exception as e:
         logger.error(f"❌ [Flask] خطأ في معالجة webhook: {e}", exc_info=True)
         return "Internal Server Error", 500

def handle_status_command(chat_id_msg: int) -> None:
    """Separate function to handle /status command to avoid blocking the Webhook."""
    logger.info(f"ℹ️ [Flask Status] معالجة أمر /status للدردشة {chat_id_msg}")
    status_msg = "⏳ جلب الحالة..."
    msg_sent = send_telegram_message(chat_id_msg, status_msg)
    if not (msg_sent and msg_sent.get('ok')):
         logger.error(f"❌ [Flask Status] فشل إرسال رسالة الحالة الأولية إلى {chat_id_msg}")
         return
    message_id_to_edit = msg_sent['result']['message_id'] if msg_sent and msg_sent.get('result') else None

    if message_id_to_edit is None:
        logger.error(f"❌ [Flask Status] فشل الحصول على message_id لتحديث الحالة في الدردشة {chat_id_msg}")
        return


    try:
        open_count = 0
        if check_db_connection() and conn:
            with conn.cursor() as status_cur:
                status_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE;")
                open_count = (status_cur.fetchone() or {}).get('count', 0)

        ws_status = 'نشط ✅' if 'ws_thread' in globals() and ws_thread and ws_thread.is_alive() else 'غير نشط ❌'
        tracker_status = 'نشط ✅' if 'tracker_thread' in globals() and tracker_thread and tracker_thread.is_alive() else 'غير نشط ❌'
        main_bot_alive = 'نشط ✅' if 'main_bot_thread' in globals() and main_bot_thread and main_bot_thread.is_alive() else 'غير نشط ❌'
        final_status_msg = (
            f"🤖 *حالة البوت:*\n"
            f"- تتبع الأسعار (WS): {ws_status}\n"
            f"- تتبع الإشارات: {tracker_status}\n"
            f"- حلقة البوت الرئيسية: {main_bot_alive}\n" # Added main bot loop status
            f"- الإشارات النشطة: *{open_count}* / {MAX_OPEN_TRADES}\n"
            f"- وقت الخادم الحالي: {datetime.now().strftime('%H:%M:%S')}"
        )
        edit_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText"
        edit_payload = {
            'chat_id': chat_id_msg,
             'message_id': message_id_to_edit,
            'text': final_status_msg,
            'parse_mode': 'Markdown'
        }
        response = requests.post(edit_url, json=edit_payload, timeout=10)
        response.raise_for_status()
        logger.info(f"✅ [Flask Status] تم تحديث الحالة للدردشة {chat_id_msg}")

    except Exception as status_err:
        logger.error(f"❌ [Flask Status] خطأ في جلب/تعديل تفاصيل الحالة للدردشة {chat_id_msg}: {status_err}", exc_info=True)
        send_telegram_message(chat_id_msg, "❌ حدث خطأ أثناء جلب تفاصيل الحالة.")


def run_flask() -> None:
    """Runs the Flask application to listen for the Webhook using a production server if available."""
    host = "0.0.0.0"
    port = int(os.environ.get('PORT', 10000)) # Use PORT environment variable or default value
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

# ---------------------- Main Loop and Check Function ----------------------
def main_loop() -> None:
    """Main loop to scan pairs and generate signals."""
    symbols_to_scan = get_crypto_symbols()
    if not symbols_to_scan:
        logger.critical("❌ [Main] لا توجد رموز صالحة تم تحميلها أو التحقق منها. لا يمكن المتابعة.")
        return

    logger.info(f"✅ [Main] تم تحميل {len(symbols_to_scan)} رمزًا صالحًا للمسح.")
    last_full_scan_time = time.time()

    while True:
        try:
            scan_start_time = time.time()
            logger.info("+" + "-"*60 + "+")
            logger.info(f"🔄 [Main] بدء دورة مسح السوق - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("+" + "-"*60 + "+")

            if not check_db_connection() or not conn:
                logger.error("❌ [Main] تخطي دورة المسح بسبب فشل اتصال قاعدة البيانات.")
                time.sleep(60)
                continue

            open_count = 0
            try:
                 with conn.cursor() as cur_check:
                    cur_check.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE;")
                    open_count = (cur_check.fetchone() or {}).get('count', 0)
            except psycopg2.Error as db_err:
                 logger.error(f"❌ [Main] خطأ في قاعدة البيانات أثناء التحقق من عدد الإشارات المفتوحة: {db_err}. تخطي الدورة.")
                 if conn: conn.rollback()
                 time.sleep(60)
                 continue

            logger.info(f"ℹ️ [Main] الإشارات المفتوحة حالياً: {open_count} / {MAX_OPEN_TRADES}")
            if open_count >= MAX_OPEN_TRADES:
                logger.info(f"⚠️ [Main] تم الوصول إلى الحد الأقصى لعدد الإشارات المفتوحة. انتظار...")
                time.sleep(get_interval_minutes(SIGNAL_GENERATION_TIMEFRAME) * 60)
                continue

            processed_in_loop = 0
            signals_generated_in_loop = 0
            slots_available = MAX_OPEN_TRADES - open_count

            for symbol in symbols_to_scan:
                 if slots_available <= 0:
                      logger.info(f"ℹ️ [Main] تم الوصول إلى الحد الأقصى ({MAX_OPEN_TRADES}) أثناء المسح. إيقاف مسح الرموز لهذه الدورة.")
                      break

                 processed_in_loop += 1
                 logger.debug(f"🔍 [Main] مسح {symbol} ({processed_in_loop}/{len(symbols_to_scan)})...")

                 try:
                    with conn.cursor() as symbol_cur:
                        symbol_cur.execute("SELECT 1 FROM signals WHERE symbol = %s AND achieved_target = FALSE LIMIT 1;", (symbol,))
                        if symbol_cur.fetchone():
                            continue

                    df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_hist is None or df_hist.empty:
                        continue

                    strategy = ScalpingTradingStrategy(symbol)
                    df_indicators = strategy.populate_indicators(df_hist)
                    if df_indicators is None:
                        continue

                    potential_signal = strategy.generate_buy_signal(df_indicators)

                    if potential_signal:
                        logger.info(f"✨ [Main] تم العثور على إشارة محتملة لـ {symbol}! (الدرجة: {potential_signal.get('r2_score', 0):.2f}) التحقق النهائي والإدراج...")
                        with conn.cursor() as final_check_cur:
                             final_check_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE;")
                             final_open_count = (final_check_cur.fetchone() or {}).get('count', 0)

                             if final_open_count < MAX_OPEN_TRADES:
                                 if insert_signal_into_db(potential_signal):
                                     send_telegram_alert(potential_signal, SIGNAL_GENERATION_TIMEFRAME)
                                     signals_generated_in_loop += 1
                                     slots_available -= 1
                                     time.sleep(2)
                                 else:
                                     logger.error(f"❌ [Main] فشل إدراج الإشارة لـ {symbol} في قاعدة البيانات.")
                             else:
                                 logger.warning(f"⚠️ [Main] تم الوصول إلى الحد الأقصى ({final_open_count}) قبل إدراج الإشارة لـ {symbol}. تم تجاهل الإشارة.")
                                 break

                 except psycopg2.Error as db_loop_err:
                      logger.error(f"❌ [Main] خطأ في قاعدة البيانات أثناء معالجة الرمز {symbol}: {db_loop_err}. الانتقال إلى التالي...")
                      if conn: conn.rollback()
                      continue
                 except Exception as symbol_proc_err:
                      logger.error(f"❌ [Main] خطأ عام في معالجة الرمز {symbol}: {symbol_proc_err}", exc_info=True)
                      continue

                 time.sleep(0.1)

            scan_duration = time.time() - scan_start_time
            logger.info(f"🏁 [Main] انتهت دورة المسح. الإشارات التي تم إنشاؤها: {signals_generated_in_loop}. مدة المسح: {scan_duration:.2f} ثانية.")
            frame_minutes = get_interval_minutes(SIGNAL_GENERATION_TIMEFRAME)
            wait_time = max(frame_minutes * 60, 120 - scan_duration)
            logger.info(f"⏳ [Main] انتظار {wait_time:.1f} ثانية للدورة التالية...")
            time.sleep(wait_time)

        except KeyboardInterrupt:
             logger.info("🛑 [Main] تم طلب الإيقاف (KeyboardInterrupt). إيقاف التشغيل...")
             break
        except psycopg2.Error as db_main_err:
             logger.error(f"❌ [Main] خطأ فادح في قاعدة البيانات في الحلقة الرئيسية: {db_main_err}. محاولة إعادة الاتصال...")
             if conn: conn.rollback()
             time.sleep(60)
             try:
                 init_db()
             except Exception as recon_err:
                 logger.critical(f"❌ [Main] فشل إعادة الاتصال بقاعدة البيانات: {recon_err}. خروج...")
                 break
        except Exception as main_err:
            logger.error(f"❌ [Main] خطأ غير متوقع في الحلقة الرئيسية: {main_err}", exc_info=True)
            logger.info("ℹ️ [Main] انتظار 120 ثانية قبل إعادة المحاولة...")
            time.sleep(120)

def cleanup_resources() -> None:
    """Closes used resources like the database connection."""
    global conn
    logger.info("ℹ️ [Cleanup] إغلاق الموارد...")
    if conn:
        try:
            conn.close()
            logger.info("✅ [DB] تم إغلاق اتصال قاعدة البيانات.")
        except Exception as close_err:
            logger.error(f"⚠️ [DB] خطأ في إغلاق اتصال قاعدة البيانات: {close_err}")
    logger.info("✅ [Cleanup] اكتمل تنظيف الموارد.")


# ---------------------- Main Entry Point ----------------------
if __name__ == "__main__":
    logger.info("🚀 بدء بوت إشارات التداول...")
    logger.info(f"الوقت المحلي: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | وقت UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    ws_thread: Optional[Thread] = None
    tracker_thread: Optional[Thread] = None
    flask_thread: Optional[Thread] = None
    main_bot_thread: Optional[Thread] = None # New thread for main_loop

    try:
        # 1. Initialize the database first
        init_db()

        # 2. Load the ML model from the database
        load_ml_model_from_db()
        if ml_model is None:
            logger.warning("⚠️ [Main] لم يتم تحميل نموذج تعلم الآلة. ستعمل الإستراتيجية بدون تنبؤات التعلم الآلي كشرط تجاوز.")

        # 3. Start WebSocket Ticker
        ws_thread = Thread(target=run_ticker_socket_manager, daemon=True, name="WebSocketThread")
        ws_thread.start()
        logger.info("✅ [Main] تم بدء مؤشر WebSocket.")
        logger.info("ℹ️ [Main] انتظار 5 ثوانٍ لتهيئة WebSocket...")
        time.sleep(5)
        if not ticker_data:
             logger.warning("⚠️ [Main] لم يتم استلام بيانات أولية من WebSocket بعد 5 ثوانٍ.")
        else:
             logger.info(f"✅ [Main] تم استلام بيانات أولية من WebSocket لـ {len(ticker_data)} رمزًا.")


        # 4. Start Signal Tracker
        tracker_thread = Thread(target=track_signals, daemon=True, name="TrackerThread")
        tracker_thread.start()
        logger.info("✅ [Main] تم بدء مؤشر الإشارة.")

        # 5. Start the main bot logic in a separate thread
        main_bot_thread = Thread(target=main_loop, daemon=True, name="MainBotLoopThread")
        main_bot_thread.start()
        logger.info("✅ [Main] تم بدء حلقة البوت الرئيسية في خيط منفصل.")

        # 6. Start Flask Server (ALWAYS run, daemon=False so it keeps the main program alive)
        flask_thread = Thread(target=run_flask, daemon=False, name="FlaskThread")
        flask_thread.start()
        logger.info("✅ [Main] تم بدء خادم Flask.")

        # Wait for the Flask thread to finish (it usually won't unless there's an error)
        flask_thread.join()

    except Exception as startup_err:
        logger.critical(f"❌ [Main] حدث خطأ فادح أثناء بدء التشغيل أو في الحلقة الرئيسية: {startup_err}", exc_info=True)
    finally:
        logger.info("🛑 [Main] يتم إيقاف تشغيل البرنامج...")
        cleanup_resources()
        logger.info("👋 [Main] تم إيقاف بوت إشارات التداول.")
        os._exit(0)
