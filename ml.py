import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle # For serializing/deserializing the model
from psycopg2 import sql, OperationalError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Tuple, Any, Union

# Scikit-learn imports for the ML model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------- إعداد التسجيل ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLTrainer')

# ---------------------- تحميل المتغيرات البيئية ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
except Exception as e:
     logger.critical(f"❌ فشل في تحميل المتغيرات البيئية الأساسية: {e}")
     exit(1)

logger.info(f"Binance API Key: {'Available' if API_KEY else 'Not available'}")
logger.info(f"Database URL: {'Available' if DB_URL else 'Not available'}")

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
# Parameters for data collection for ML training
ML_TRAINING_TIMEFRAME: str = '5m' # Use the same timeframe as signal generation
ML_TRAINING_LOOKBACK_DAYS: int = 30 # Fetch more historical data for training (e.g., 30-90 days)
ML_MODEL_NAME: str = 'DecisionTree_Scalping_V1' # Name for the ML model in DB

# Indicator parameters (should match those in c4.py)
RSI_PERIOD: int = 9
EMA_SHORT_PERIOD: int = 8
EMA_LONG_PERIOD: int = 21
VWMA_PERIOD: int = 15
ENTRY_ATR_PERIOD: int = 10
BOLLINGER_WINDOW: int = 20
BOLLINGER_STD_DEV: int = 2
MACD_FAST: int = 9
MACD_SLOW: int = 18
MACD_SIGNAL: int = 9
ADX_PERIOD: int = 10
SUPERTREND_PERIOD: int = 10
SUPERTREND_MULTIPLIER: float = 2.5

# Global variables
client: Optional[Client] = None
conn: Optional[psycopg2.extensions.connection] = None
cur: Optional[psycopg2.extensions.cursor] = None

# ---------------------- Binance Client Setup ----------------------
def init_binance_client() -> None:
    """Initializes Binance client."""
    global client
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
        logger.critical(f"❌ [Binance] فشل غير متوقع في تهيئة عميل Binance: {e}", exc_info=True)
        exit(1)

# ---------------------- Database Connection Setup ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    """Initializes database connection and creates ml_models table if it doesn't exist."""
    global conn, cur
    logger.info("[DB] بدء تهيئة قاعدة البيانات...")
    for attempt in range(retries):
        try:
            logger.info(f"[DB] محاولة الاتصال بقاعدة البيانات (المحاولة {attempt + 1}/{retries})...")
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            cur = conn.cursor()
            logger.info("✅ [DB] تم الاتصال بقاعدة البيانات بنجاح.")

            # --- Create ml_models table ---
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

def cleanup_resources() -> None:
    """Closes database connection."""
    global conn
    logger.info("ℹ️ [Cleanup] إغلاق الموارد...")
    if conn:
        try:
            conn.close()
            logger.info("✅ [DB] تم إغلاق اتصال قاعدة البيانات.")
        except Exception as close_err:
            logger.error(f"⚠️ [DB] خطأ أثناء إغلاق اتصال قاعدة البيانات: {close_err}")

# ---------------------- Data Fetching and Indicator Calculation Functions ----------------------
# Replicating necessary functions from c4.py for self-containment of the training script
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
        df.dropna(subset=numeric_cols, inplace=True)

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
        return pd.Series(index=df_calc.index if df_calc is not None else None, dtype=float)
    if len(df_calc) < period:
        return pd.Series(index=df_calc.index if df_calc is not None else None, dtype=float)

    df_calc['price_volume'] = df_calc['close'] * df_calc['volume']
    rolling_price_volume_sum = df_calc['price_volume'].rolling(window=period, min_periods=period).sum()
    rolling_volume_sum = df_calc['volume'].rolling(window=period, min_periods=period).sum()
    vwma = rolling_price_volume_sum / rolling_volume_sum.replace(0, np.nan)
    return vwma

def calculate_rsi_indicator(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """Calculates Relative Strength Index (RSI)."""
    df = df.copy()
    if 'close' not in df.columns or df['close'].isnull().all():
        df['rsi'] = np.nan
        return df
    if len(df) < period:
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
        df['atr'] = np.nan
        return df
    if len(df) < period + 1:
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
        df['bb_middle'] = np.nan
        df['bb_upper'] = np.nan
        df['bb_lower'] = np.nan
        return df
    if len(df) < window:
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
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['macd_hist'] = np.nan
        return df
    min_len = max(fast, slow, signal)
    if len(df) < min_len:
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
        df_calc['adx'] = np.nan
        df_calc['di_plus'] = np.nan
        df_calc['di_minus'] = np.nan
        return df_calc
    if len(df_calc) < period * 2:
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
        df['vwap'] = np.nan
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            df['vwap'] = np.nan
            return df
    if df.index.tz is not None:
        df.index = df.index.tz_convert('UTC')
    else:
        df.index = df.index.tz_localize('UTC')

    df['date'] = df.index.date
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol'] = df['typical_price'] * df['volume']

    try:
        df['cum_tp_vol'] = df.groupby('date')['tp_vol'].cumsum()
        df['cum_volume'] = df.groupby('date')['volume'].cumsum()
    except KeyError:
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
        df['obv'] = np.nan
        return df
    if not pd.api.types.is_numeric_dtype(df['close']) or not pd.api.types.is_numeric_dtype(df['volume']):
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
        df_st['supertrend'] = np.nan
        df_st['supertrend_trend'] = 0
        return df_st

    df_st = calculate_atr_indicator(df_st, period=SUPERTREND_PERIOD)

    if 'atr' not in df_st.columns or df_st['atr'].isnull().all():
         df_st['supertrend'] = np.nan
         df_st['supertrend_trend'] = 0
         return df_st
    if len(df_st) < SUPERTREND_PERIOD:
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
             elif close[i] < final_lb[i]:
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

def populate_all_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Calculates all required indicators for the strategy."""
    if df is None or df.empty:
        return None

    df_calc = df.copy()
    try:
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
    except Exception as e:
        logger.error(f"❌ خطأ أثناء حساب المؤشرات: {e}", exc_info=True)
        return None

    # Define the features that will be used for the ML model
    # These should be the same as the ones used in c4.py for prediction
    feature_columns = [
        f'ema_{EMA_SHORT_PERIOD}', f'ema_{EMA_LONG_PERIOD}', 'vwma',
        'rsi', 'atr', 'bb_upper', 'bb_lower', 'bb_middle',
        'macd', 'macd_signal', 'macd_hist',
        'adx', 'di_plus', 'di_minus', 'vwap', 'obv',
        'supertrend', 'supertrend_trend'
    ]

    # Ensure all feature columns exist and are numeric
    for col in feature_columns:
        if col not in df_calc.columns:
            logger.warning(f"⚠️ عمود الميزة المفقود: {col}")
            df_calc[col] = np.nan # Add missing column as NaN
        else:
            df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')

    # Drop rows with NaN values in feature columns
    initial_len = len(df_calc)
    df_cleaned = df_calc.dropna(subset=feature_columns).copy()
    if len(df_cleaned) < initial_len:
        logger.debug(f"ℹ️ تم إسقاط {initial_len - len(df_cleaned)} صفًا بسبب قيم NaN في المؤشرات.")

    return df_cleaned

# ---------------------- Main Training Logic ----------------------
def get_crypto_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """Reads the list of currency symbols from a text file."""
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
        return raw_symbols
    except Exception as e:
        logger.error(f"❌ [Data] خطأ في قراءة الملف '{filename}': {e}", exc_info=True)
        return []

def train_and_save_model() -> None:
    """
    Fetches data, calculates indicators, trains a Decision Tree Classifier,
    and saves the model and its metrics to the database.
    """
    logger.info("🚀 بدء عملية تدريب نموذج تعلم الآلة...")
    init_binance_client()
    init_db()

    symbols = get_crypto_symbols()
    if not symbols:
        logger.critical("❌ لا توجد رموز صالحة للتدريب. إنهاء.")
        return

    all_features = []
    all_targets = []
    processed_symbols_count = 0

    # Define the features that will be used for the ML model
    # This list MUST match what's expected by the model in c4.py
    feature_columns = [
        f'ema_{EMA_SHORT_PERIOD}', f'ema_{EMA_LONG_PERIOD}', 'vwma',
        'rsi', 'atr', 'bb_upper', 'bb_lower', 'bb_middle',
        'macd', 'macd_signal', 'macd_hist',
        'adx', 'di_plus', 'di_minus', 'vwap', 'obv',
        'supertrend', 'supertrend_trend'
    ]

    for symbol in symbols:
        logger.info(f"ℹ️ جلب البيانات وحساب المؤشرات لـ {symbol}...")
        df = fetch_historical_data(symbol, ML_TRAINING_TIMEFRAME, ML_TRAINING_LOOKBACK_DAYS)
        if df is None or df.empty:
            logger.warning(f"⚠️ تخطي {symbol} بسبب عدم كفاية البيانات.")
            continue

        df_processed = populate_all_indicators(df)
        if df_processed is None or df_processed.empty:
            logger.warning(f"⚠️ تخطي {symbol} بسبب عدم كفاية البيانات بعد حساب المؤشرات.")
            continue

        # Create the target variable: 1 if next close > current close, 0 otherwise
        # Shift(-1) means the close price of the *next* candle
        df_processed['target'] = (df_processed['close'].shift(-1) > df_processed['close']).astype(int)

        # Drop the last row as its target will be NaN
        df_processed.dropna(subset=['target'] + feature_columns, inplace=True)

        if df_processed.empty:
            logger.warning(f"⚠️ تخطي {symbol} بسبب عدم كفاية البيانات بعد إعداد الهدف وإزالة NaN.")
            continue

        # Ensure all feature columns are present before appending
        missing_features = [col for col in feature_columns if col not in df_processed.columns]
        if missing_features:
            logger.error(f"❌ الميزات المفقودة لـ {symbol}: {missing_features}. تخطي هذا الرمز.")
            continue

        all_features.append(df_processed[feature_columns])
        all_targets.append(df_processed['target'])
        processed_symbols_count += 1

    if not all_features:
        logger.critical("❌ لا توجد بيانات كافية لتدريب النموذج بعد معالجة الرموز.")
        return

    X = pd.concat(all_features)
    y = pd.concat(all_targets)

    if X.empty or y.empty:
        logger.critical("❌ مجموعات الميزات أو الأهداف فارغة بعد التجميع. لا يمكن التدريب.")
        return

    logger.info(f"✅ تم تجميع بيانات التدريب من {processed_symbols_count} رمزًا. حجم البيانات: {len(X)} عينة.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info(f"ℹ️ حجم بيانات التدريب: {len(X_train)}، حجم بيانات الاختبار: {len(X_test)}")

    # Train the Decision Tree Classifier
    logger.info("ℹ️ تدريب نموذج Decision Tree...")
    model = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_leaf=5) # Add some regularization
    model.fit(X_train, y_train)
    logger.info("✅ تم تدريب النموذج.")

    # Evaluate the model
    logger.info("ℹ️ تقييم أداء النموذج...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist() # Convert to list for JSONB

    logger.info(f"✅ دقة النموذج: {accuracy:.4f}")
    logger.info(f"تقرير التصنيف:\n{json.dumps(report, indent=2)}")
    logger.info(f"مصفوفة الارتباك:\n{json.dumps(conf_matrix, indent=2)}")

    # Prepare metrics for storage
    model_metrics = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'features': feature_columns, # Store feature names for consistency
        'training_symbols_count': processed_symbols_count
    }

    # Serialize the model
    pickled_model = pickle.dumps(model)
    logger.info(f"ℹ️ حجم النموذج المختار: {len(pickled_model) / (1024*1024):.2f} MB")

    # Save to database
    try:
        with conn.cursor() as db_cur:
            # Check if a model with this name already exists
            db_cur.execute("SELECT id FROM ml_models WHERE model_name = %s;", (ML_MODEL_NAME,))
            existing_model = db_cur.fetchone()

            if existing_model:
                logger.info(f"ℹ️ تحديث النموذج الحالي '{ML_MODEL_NAME}' في قاعدة البيانات.")
                update_query = sql.SQL("""
                    UPDATE ml_models
                    SET model_data = %s, trained_at = NOW(), metrics = %s
                    WHERE model_name = %s;
                """)
                db_cur.execute(update_query, (psycopg2.Binary(pickled_model), json.dumps(model_metrics), ML_MODEL_NAME))
            else:
                logger.info(f"ℹ️ إدراج نموذج جديد '{ML_MODEL_NAME}' في قاعدة البيانات.")
                insert_query = sql.SQL("""
                    INSERT INTO ml_models (model_name, model_data, trained_at, metrics)
                    VALUES (%s, %s, NOW(), %s);
                """)
                db_cur.execute(insert_query, (ML_MODEL_NAME, psycopg2.Binary(pickled_model), json.dumps(model_metrics)))
        conn.commit()
        logger.info("✅ تم حفظ النموذج ومقاييسه في قاعدة البيانات بنجاح.")
    except psycopg2.Error as db_err:
        logger.error(f"❌ خطأ في قاعدة البيانات أثناء حفظ النموذج: {db_err}", exc_info=True)
        if conn: conn.rollback()
    except Exception as e:
        logger.error(f"❌ خطأ غير متوقع أثناء حفظ النموذج: {e}", exc_info=True)
        if conn: conn.rollback()

    cleanup_resources()
    logger.info("🏁 انتهت عملية تدريب نموذج تعلم الآلة.")

if __name__ == "__main__":
    train_and_save_model()
