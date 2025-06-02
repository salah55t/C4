import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Tuple

# استيراد مكتبات Flask والخيوط
from flask import Flask, request, Response
from threading import Thread

# ---------------------- إعداد التسجيل ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_model_trainer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLTrainer')

# ---------------------- تحميل المتغيرات البيئية ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None) # إضافة WEBHOOK_URL
except Exception as e:
     logger.critical(f"❌ فشل في تحميل المتغيرات البيئية الأساسية: {e}")
     exit(1)

logger.info(f"Binance API Key: {'Available' if API_KEY else 'Not available'}")
logger.info(f"Database URL: {'Available' if DB_URL else 'Not available'}")
logger.info(f"Webhook URL: {WEBHOOK_URL if WEBHOOK_URL else 'Not specified'} (Flask will always run for Render)")


# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
SIGNAL_GENERATION_TIMEFRAME: str = '5m' # تم التغيير إلى 5 دقائق ليتناسب مع 3 شمعات = 15 دقيقة
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 90 # 3 أشهر من البيانات
BASE_ML_MODEL_NAME: str = 'DecisionTree_Scalping_V1' # اسم أساسي للنموذج، سيتم إضافة الرمز إليه

# Indicator Parameters (نسخ من c4.py لضمان الاتساق)
# تم الاحتفاظ فقط بالمعلمات المطلوبة لحساب الميزات الجديدة
VOLUME_LOOKBACK_CANDLES: int = 3 # عدد الشمعات لحساب متوسط الحجم (3 شمعات * 5 دقائق = 15 دقيقة)
RSI_PERIOD: int = 9 # مطلوب لحساب RSI الذي يعتمد عليه RSI Momentum
RSI_MOMENTUM_LOOKBACK_CANDLES: int = 2 # عدد الشمعات للتحقق من تزايد RSI للزخم

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
cur: Optional[psycopg2.extensions.cursor] = None
client: Optional[Client] = None

# متغيرات لتتبع حالة التدريب
# تم تعريفها هنا كمتغيرات عامة
training_status: str = "Idle"
last_training_time: Optional[datetime] = None
last_training_metrics: Dict[str, Any] = {}
training_error: Optional[str] = None


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

# ---------------------- Database Connection Setup (نسخ من c4.py) ----------------------
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

def get_crypto_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """
    Reads the list of currency symbols from a text file, then validates them
    as valid USDT pairs available for Spot trading on Binance. (نسخ من c4.py)
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

        # Call get_historical_klines for the entire period.
        # The python-binance library is designed to handle internal pagination
        # if the requested range exceeds the API's single-request limit (e.g., 1000 klines).
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

        # Sort by index (timestamp) to ensure chronological order
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

# ---------------------- Technical Indicator Functions (Only those needed for ML features) ----------------------
def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculates Exponential Moving Average (EMA)."""
    if series is None or series.isnull().all() or len(series) < span:
        return pd.Series(index=series.index if series is not None else None, dtype=float)
    return series.ewm(span=span, adjust=False).mean()

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

# NEW: Function to calculate numerical Bitcoin trend feature
def _calculate_btc_trend_feature(df_btc: pd.DataFrame) -> Optional[pd.Series]:
    """
    Calculates a numerical representation of Bitcoin's trend based on EMA20 and EMA50.
    Returns 1 for bullish (صعودي), -1 for bearish (هبوطي), 0 for neutral/sideways (محايد/تذبذب).
    """
    logger.debug("ℹ️ [Indicators] حساب اتجاه البيتكوين للميزات...")
    # Need enough data for EMA50, plus a few extra candles for robustness
    min_data_for_ema = 50 + 5 # 50 for EMA50, 5 buffer

    if df_btc is None or df_btc.empty or len(df_btc) < min_data_for_ema:
        logger.warning(f"⚠️ [Indicators] بيانات BTC/USDT غير كافية ({len(df_btc) if df_btc is not None else 0} < {min_data_for_ema}) لحساب اتجاه البيتكوين للميزات.")
        # Return a series of zeros (neutral) with the original index if data is insufficient
        return pd.Series(index=df_btc.index if df_btc is not None else None, data=0.0)

    df_btc_copy = df_btc.copy()
    df_btc_copy['close'] = pd.to_numeric(df_btc_copy['close'], errors='coerce')
    df_btc_copy.dropna(subset=['close'], inplace=True)

    if len(df_btc_copy) < min_data_for_ema:
        logger.warning(f"⚠️ [Indicators] بيانات BTC/USDT غير كافية بعد إزالة قيم NaN لحساب الاتجاه.")
        return pd.Series(index=df_btc.index, data=0.0) # Return neutral if not enough data after dropna

    ema20 = calculate_ema(df_btc_copy['close'], 20)
    ema50 = calculate_ema(df_btc_copy['close'], 50)

    # Combine EMAs and close into a single DataFrame for easier comparison
    ema_df = pd.DataFrame({'ema20': ema20, 'ema50': ema50, 'close': df_btc_copy['close']})
    ema_df.dropna(inplace=True) # Drop rows where any EMA or close is NaN

    if ema_df.empty:
        logger.warning("⚠️ [Indicators] DataFrame EMA فارغ بعد إزالة قيم NaN. لا يمكن حساب اتجاه البيتكوين.")
        return pd.Series(index=df_btc.index, data=0.0) # Return neutral if no valid EMA data

    # Initialize trend column with neutral (0.0)
    trend_series = pd.Series(index=ema_df.index, data=0.0)

    # Apply trend logic:
    # Bullish: current_close > ema20 > ema50
    trend_series[(ema_df['close'] > ema_df['ema20']) & (ema_df['ema20'] > ema_df['ema50'])] = 1.0
    # Bearish: current_close < ema20 < ema50
    trend_series[(ema_df['close'] < ema_df['ema20']) & (ema_df['ema20'] < ema_df['ema50'])] = -1.0

    # Reindex to original df_btc index and fill any remaining NaNs with 0 (neutral)
    # This ensures the series has the same index as the altcoin DataFrame for merging
    final_trend_series = trend_series.reindex(df_btc.index).fillna(0.0)
    logger.debug(f"✅ [Indicators] تم حساب ميزة اتجاه البيتكوين. أمثلة: {final_trend_series.tail().tolist()}")
    return final_trend_series


# ---------------------- وظائف تدريب وحفظ النموذج ----------------------
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

def prepare_data_for_ml(df: pd.DataFrame, symbol: str, target_period: int = 5) -> Optional[pd.DataFrame]:
    """
    يجهز البيانات لتدريب نموذج التعلم الآلي.
    يضيف المؤشرات (حجم السيولة، مؤشر الزخم، واتجاه البيتكوين) ويزيل الصفوف التي تحتوي على قيم NaN.
    يضيف عمود الهدف 'target' الذي يشير إلى ما إذا كان السعر سيرتفع في الشموع القادمة.
    """
    logger.info(f"ℹ️ [ML Prep] تجهيز البيانات لنموذج التعلم الآلي لـ {symbol} (حجم السيولة، الزخم، واتجاه البيتكوين)...")

    # تحديد الحد الأدنى لطول البيانات المطلوبة فقط للميزات المستخدمة
    # 50 + 5 for BTC EMA calculation, plus target_period, plus some buffer
    min_len_required = max(VOLUME_LOOKBACK_CANDLES, RSI_PERIOD + RSI_MOMENTUM_LOOKBACK_CANDLES, 55) + target_period + 5

    if len(df) < min_len_required:
        logger.warning(f"⚠️ [ML Prep] DataFrame لـ {symbol} قصير جدًا ({len(df)} < {min_len_required}) لتجهيز البيانات.")
        return None

    df_calc = df.copy()

    # حساب الميزات المطلوبة فقط: متوسط حجم السيولة لآخر 15 دقيقة (3 شمعات 5m)
    df_calc['volume_15m_avg'] = df_calc['volume'].rolling(window=VOLUME_LOOKBACK_CANDLES, min_periods=1).mean()
    logger.debug(f"ℹ️ [ML Prep] تم حساب متوسط حجم 15 دقيقة لـ {symbol}.")

    # حساب مؤشر القوة النسبية (RSI) لأنه مطلوب لمؤشر الزخم
    df_calc = calculate_rsi_indicator(df_calc, RSI_PERIOD)

    # إضافة مؤشر زخم صعودي (RSI Momentum)
    df_calc['rsi_momentum_bullish'] = 0
    if len(df_calc) >= RSI_MOMENTUM_LOOKBACK_CANDLES + 1:
        # Check if RSI is increasing over the last N candles and is above 50 (bullish territory)
        for i in range(RSI_MOMENTUM_LOOKBACK_CANDLES, len(df_calc)):
            rsi_slice = df_calc['rsi'].iloc[i - RSI_MOMENTUM_LOOKBACK_CANDLES : i + 1]
            if not rsi_slice.isnull().any() and np.all(np.diff(rsi_slice) > 0) and rsi_slice.iloc[-1] > 50:
                df_calc.loc[df_calc.index[i], 'rsi_momentum_bullish'] = 1
    logger.debug(f"ℹ️ [ML Prep] تم حساب مؤشر زخم RSI الصعودي لـ {symbol}.")

    # --- NEW: Fetch and calculate BTC trend feature ---
    # Fetch BTC data for the same period and interval
    btc_df = fetch_historical_data("BTCUSDT", interval=SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING)
    btc_trend_series = None
    if btc_df is not None and not btc_df.empty:
        btc_trend_series = _calculate_btc_trend_feature(btc_df)
        if btc_trend_series is not None:
            # Merge BTC trend with the current symbol's DataFrame based on timestamp index
            # Using 'left' merge to keep all rows of df_calc. Fill NaNs with 0 (neutral).
            df_calc = df_calc.merge(btc_trend_series.rename('btc_trend_feature'),
                                    left_index=True, right_index=True, how='left')
            df_calc['btc_trend_feature'].fillna(0.0, inplace=True) # Fill missing BTC trend with neutral
            logger.debug(f"ℹ️ [ML Prep] تم دمج ميزة اتجاه البيتكوين لـ {symbol}.")
        else:
            logger.warning(f"⚠️ [ML Prep] فشل حساب ميزة اتجاه البيتكوين. سيتم استخدام 0 كقيمة افتراضية لـ 'btc_trend_feature'.")
            df_calc['btc_trend_feature'] = 0.0
    else:
        logger.warning(f"⚠️ [ML Prep] فشل جلب البيانات التاريخية للبيتكوين. سيتم استخدام 0 كقيمة افتراضية لـ 'btc_trend_feature'.")
        df_calc['btc_trend_feature'] = 0.0


    # تعريف أعمدة الميزات التي سيستخدمها النموذج (فقط الميزات الجديدة)
    feature_columns = [
        'volume_15m_avg', # ميزة جديدة: متوسط حجم السيولة لآخر 15 دقيقة
        'rsi_momentum_bullish', # ميزة جديدة: زخم RSI الصعودي
        'btc_trend_feature' # ميزة جديدة: اتجاه البيتكوين (1: صعودي, -1: هبوطي, 0: محايد)
    ]

    # التأكد من وجود جميع أعمدة الميزات وتحويلها إلى أرقام
    for col in feature_columns:
        if col not in df_calc.columns:
            logger.warning(f"⚠️ [ML Prep] عمود الميزة المفقود: {col}. سيتم إضافته كـ NaN.")
            df_calc[col] = np.nan
        else:
            df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')

    # إنشاء عمود الهدف: هل السعر سيصعد بنسبة معينة في الشموع القادمة؟
    # على سبيل المثال، إذا كان السعر سيصعد بنسبة 0.5% خلال الشموع الخمس القادمة
    price_change_threshold = 0.005 # 0.5%
    # Ensure 'close' column is numeric before shifting
    df_calc['close'] = pd.to_numeric(df_calc['close'], errors='coerce')
    df_calc['future_max_close'] = df_calc['close'].shift(-target_period).rolling(window=target_period, min_periods=1).max()
    # Corrected target calculation: check if future max close is significantly higher than current close
    df_calc['target'] = ((df_calc['future_max_close'] / df_calc['close']) - 1 > price_change_threshold).astype(int)


    # إسقاط الصفوف التي تحتوي على قيم NaN بعد حساب المؤشرات والهدف
    initial_len = len(df_calc)
    df_cleaned = df_calc.dropna(subset=feature_columns + ['target']).copy()
    dropped_count = initial_len - len(df_cleaned)

    if dropped_count > 0:
        logger.info(f"ℹ️ [ML Prep] لـ {symbol}: تم إسقاط {dropped_count} صفًا بسبب قيم NaN بعد حساب المؤشرات والهدف.")
    if df_cleaned.empty:
        logger.warning(f"⚠️ [ML Prep] DataFrame لـ {symbol} فارغ بعد إزالة قيم NaN لتجهيز ML.")
        return None

    logger.info(f"✅ [ML Prep] تم تجهيز البيانات لـ {symbol} بنجاح. عدد الصفوف: {len(df_cleaned)}")
    return df_cleaned[feature_columns + ['target']]


def train_and_evaluate_model(data: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
    """
    يقوم بتدريب نموذج Decision Tree ويقيم أداءه.
    """
    logger.info("ℹ️ [ML Train] بدء تدريب وتقييم النموذج...")

    if data.empty:
        logger.error("❌ [ML Train] DataFrame فارغ للتدريب.")
        return None, {}

    X = data.drop('target', axis=1)
    y = data['target']

    if X.empty or y.empty:
        logger.error("❌ [ML Train] ميزات أو أهداف فارغة للتدريب.")
        return None, {}

    # تقسيم البيانات إلى مجموعات تدريب واختبار
    # استخدام stratify=y لضمان توزيع متوازن للفئات في مجموعات التدريب والاختبار
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError as ve:
        logger.warning(f"⚠️ [ML Train] لا يمكن استخدام stratify بسبب وجود فئة واحدة في الهدف: {ve}. سيتم المتابعة بدون stratify.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # التحجيم (Scaling) للميزات (مهم لبعض النماذج، وليس بالضرورة لـ Decision Tree ولكن ممارسة جيدة)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back to DataFrames with feature names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # تدريب نموذج Decision Tree Classifier
    model = DecisionTreeClassifier(random_state=42, max_depth=10) # يمكن تعديل المعلمات
    model.fit(X_train_scaled_df, y_train) # Fit with DataFrame
    logger.info("✅ [ML Train] تم تدريب النموذج بنجاح.")

    # التقييم
    y_pred = model.predict(X_test_scaled_df) # Predict with DataFrame
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'num_samples_trained': len(X_train),
        'num_samples_tested': len(X_test),
        'feature_names': X.columns.tolist() # حفظ أسماء الميزات لضمان التناسق عند التحميل
    }

    logger.info(f"📊 [ML Train] مقاييس أداء النموذج:")
    logger.info(f"  - الدقة (Accuracy): {accuracy:.4f}")
    logger.info(f"  - الدقة (Precision): {precision:.4f}")
    logger.info(f"  - الاستدعاء (Recall): {recall:.4f}")
    logger.info(f"  - مقياس F1: {f1:.4f}")

    return model, metrics

def save_ml_model_to_db(model: Any, model_name: str, metrics: Dict[str, Any]) -> bool:
    """
    يحفظ النموذج المدرب وبياناته الوصفية (المقاييس) في قاعدة البيانات.
    """
    logger.info(f"ℹ️ [DB Save] التحقق من اتصال قاعدة البيانات قبل الحفظ...")
    if not check_db_connection() or not conn:
        logger.error("❌ [DB Save] لا يمكن حفظ نموذج ML بسبب مشكلة في اتصال قاعدة البيانات.")
        return False

    logger.info(f"ℹ️ [DB Save] محاولة حفظ نموذج ML '{model_name}' في قاعدة البيانات...")
    try:
        # تسلسل النموذج باستخدام pickle
        model_binary = pickle.dumps(model)
        logger.info(f"✅ [DB Save] تم تسلسل النموذج بنجاح. حجم البيانات: {len(model_binary)} بايت.")

        # تحويل المقاييس إلى JSONB
        metrics_json = json.dumps(convert_np_values(metrics))
        logger.info(f"✅ [DB Save] تم تحويل المقاييس إلى JSON بنجاح.")

        with conn.cursor() as db_cur:
            # التحقق مما إذا كان النموذج موجودًا بالفعل (للتحديث أو الإدراج)
            db_cur.execute("SELECT id FROM ml_models WHERE model_name = %s;", (model_name,))
            existing_model = db_cur.fetchone()

            if existing_model:
                logger.info(f"ℹ️ [DB Save] النموذج '{model_name}' موجود بالفعل. سيتم تحديثه.")
                update_query = sql.SQL("""
                    UPDATE ml_models
                    SET model_data = %s, trained_at = NOW(), metrics = %s
                    WHERE id = %s;
                """)
                db_cur.execute(update_query, (model_binary, metrics_json, existing_model['id']))
                logger.info(f"✅ [DB Save] تم تحديث نموذج ML '{model_name}' في قاعدة البيانات بنجاح.")
            else:
                logger.info(f"ℹ️ [DB Save] النموذج '{model_name}' غير موجود. سيتم إدراجه كنموذج جديد.")
                insert_query = sql.SQL("""
                    INSERT INTO ml_models (model_name, model_data, trained_at, metrics)
                    VALUES (%s, %s, NOW(), %s);
                """)
                db_cur.execute(insert_query, (model_name, model_binary, metrics_json))
                logger.info(f"✅ [DB Save] تم حفظ نموذج ML '{model_name}' جديد في قاعدة البيانات بنجاح.")
        conn.commit()
        logger.info(f"✅ [DB Save] تم تنفيذ commit لقاعدة البيانات بنجاح.")
        return True
    except psycopg2.Error as db_err:
        logger.error(f"❌ [DB Save] خطأ في قاعدة البيانات أثناء حفظ نموذج ML: {db_err}", exc_info=True)
        if conn: conn.rollback()
        return False
    except pickle.PicklingError as pickle_err:
        logger.error(f"❌ [DB Save] خطأ في تسلسل نموذج ML: {pickle_err}", exc_info=True)
        if conn: conn.rollback()
        return False
    except Exception as e:
        logger.error(f"❌ [DB Save] خطأ غير متوقع أثناء حفظ نموذج ML: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

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


# ---------------------- Flask Service ----------------------
app = Flask(__name__)

@app.route('/')
def home() -> Response:
    """Simple home page to show the bot is running."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    status_message = (
        f"🤖 *ML Trainer Service Status:*\n"
        f"- Current Time: {now}\n"
        f"- Training Status: *{training_status}*\n"
    )
    if last_training_time:
        status_message += f"- Last Training Time: {last_training_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    if last_training_metrics:
        status_message += f"- Last Training Metrics (Accuracy): {last_training_metrics.get('accuracy', 'N/A'):.4f}\n"
    if training_error:
        status_message += f"- Last Error: {training_error}\n"

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


# ---------------------- نقطة الدخول الرئيسية ----------------------
if __name__ == "__main__":
    logger.info("🚀 بدء سكريبت تدريب نموذج التعلم الآلي...")
    logger.info(f"الوقت المحلي: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | وقت UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    flask_thread: Optional[Thread] = None

    try:
        # 1. بدء خدمة Flask في خيط منفصل أولاً
        # هذا يضمن أن الخدمة ستكون متاحة للاستجابة لطلبات Uptime Monitor
        # بينما يتم تنفيذ عملية التدريب في الخيط الرئيسي.
        flask_thread = Thread(target=run_flask_service, daemon=False, name="FlaskServiceThread")
        flask_thread.start()
        logger.info("✅ [Main] تم بدء خدمة Flask.")
        time.sleep(2) # إعطاء بعض الوقت لـ Flask للبدء

        # 2. تهيئة قاعدة البيانات
        init_db()

        # 3. جلب قائمة الرموز
        symbols = get_crypto_symbols()
        if not symbols:
            logger.critical("❌ [Main] لا توجد رموز صالحة للتدريب. يرجى التحقق من 'crypto_list.txt'.")
            training_status = "Failed: No valid symbols"
            exit(1)

        training_status = "In Progress: Training Models"
        training_error = None # Reset error
        
        overall_metrics: Dict[str, Any] = {
            'total_models_trained': 0,
            'successful_models': 0,
            'failed_models': 0,
            'avg_accuracy': 0.0,
            'avg_precision': 0.0,
            'avg_recall': 0.0,
            'avg_f1_score': 0.0,
            'details_per_symbol': {}
        }
        
        total_accuracy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1_score = 0.0


        # 4. تدريب نموذج لكل رمز بشكل منفصل
        for symbol in symbols:
            current_model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
            overall_metrics['total_models_trained'] += 1
            logger.info(f"\n--- ⏳ [Main] بدء تدريب النموذج لـ {symbol} ({current_model_name}) ---")
            
            try:
                # جلب البيانات التاريخية للرمز الحالي
                df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING)
                if df_hist is None or df_hist.empty:
                    logger.warning(f"⚠️ [Main] لم يتمكن من جلب بيانات كافية لـ {symbol}. تخطي تدريب هذا النموذج.")
                    overall_metrics['failed_models'] += 1
                    overall_metrics['details_per_symbol'][symbol] = {'status': 'Failed: No data', 'error': 'No sufficient historical data'}
                    continue

                # تجهيز البيانات لنموذج التعلم الآلي
                df_processed = prepare_data_for_ml(df_hist, symbol)
                if df_processed is None or df_processed.empty:
                    logger.warning(f"⚠️ [Main] لا توجد بيانات جاهزة للتدريب لـ {symbol} بعد المعالجة المسبقة للمؤشرات. تخطي.")
                    overall_metrics['failed_models'] += 1
                    overall_metrics['details_per_symbol'][symbol] = {'status': 'Failed: No processed data', 'error': 'No sufficient processed data'}
                    continue

                # تدريب وتقييم النموذج
                trained_model, model_metrics = train_and_evaluate_model(df_processed)

                if trained_model is None:
                    logger.error(f"❌ [Main] فشل تدريب النموذج لـ {symbol}. لا يمكن حفظه.")
                    overall_metrics['failed_models'] += 1
                    overall_metrics['details_per_symbol'][symbol] = {'status': 'Failed: Training failed', 'error': 'Model training returned None'}
                    continue

                # حفظ النموذج في قاعدة البيانات
                if save_ml_model_to_db(trained_model, current_model_name, model_metrics):
                    logger.info(f"✅ [Main] تم حفظ النموذج '{current_model_name}' بنجاح في قاعدة البيانات.")
                    overall_metrics['successful_models'] += 1
                    overall_metrics['details_per_symbol'][symbol] = {'status': 'Completed Successfully', 'metrics': model_metrics}
                    
                    total_accuracy += model_metrics.get('accuracy', 0.0)
                    total_precision += model_metrics.get('precision', 0.0)
                    total_recall += model_metrics.get('recall', 0.0)
                    total_f1_score += model_metrics.get('f1_score', 0.0)
                else:
                    logger.error(f"❌ [Main] فشل حفظ النموذج '{current_model_name}' في قاعدة البيانات.")
                    overall_metrics['failed_models'] += 1
                    overall_metrics['details_per_symbol'][symbol] = {'status': 'Completed with Errors: Model save failed', 'error': 'Failed to save model to DB'}

            except Exception as e:
                logger.critical(f"❌ [Main] حدث خطأ فادح أثناء تدريب النموذج لـ {symbol}: {e}", exc_info=True)
                overall_metrics['failed_models'] += 1
                overall_metrics['details_per_symbol'][symbol] = {'status': 'Failed: Unhandled exception', 'error': str(e)}
            
            logger.info(f"--- ✅ [Main] انتهى تدريب النموذج لـ {symbol} ---")
            time.sleep(1) # تأخير بسيط بين تدريب النماذج

        # تحديث الحالة العامة للتدريب
        if overall_metrics['successful_models'] > 0:
            overall_metrics['avg_accuracy'] = total_accuracy / overall_metrics['successful_models']
            overall_metrics['avg_precision'] = total_precision / overall_metrics['successful_models']
            overall_metrics['avg_recall'] = total_recall / overall_metrics['successful_models']
            overall_metrics['avg_f1_score'] = total_f1_score / overall_metrics['successful_models']

        if overall_metrics['successful_models'] == overall_metrics['total_models_trained']:
            training_status = "Completed Successfully (All Models Trained)"
        elif overall_metrics['successful_models'] > 0:
            training_status = "Completed with Errors (Some Models Failed)"
        else:
            training_status = "Failed (No Models Trained Successfully)"
        
        last_training_time = datetime.now()
        last_training_metrics = overall_metrics


        # انتظر خيط Flask لإنهاء (مما يبقي البرنامج قيد التشغيل)
        if flask_thread:
            flask_thread.join()

    except Exception as e:
        logger.critical(f"❌ [Main] حدث خطأ فادح أثناء تشغيل سكريبت التدريب الرئيسي: {e}", exc_info=True)
        training_status = "Failed: Unhandled exception in main loop"
        training_error = str(e)
    finally:
        logger.info("🛑 [Main] يتم إيقاف تشغيل سكريبت التدريب...")
        cleanup_resources()
        logger.info("👋 [Main] تم إيقاف سكريبت تدريب نموذج التعلم الآلي.")
        # os._exit(0) # لا تستخدم os._exit(0) هنا إذا كنت تريد أن يبقى Flask يعمل
