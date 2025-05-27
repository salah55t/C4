import os
import json
import logging
import time
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from threading import Thread
from apscheduler.schedulers.background import BackgroundScheduler
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
import psycopg2
from psycopg2 import sql, OperationalError
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
import pickle
from decouple import config
from typing import List, Tuple, Any # Import Tuple and Any for type hints

# Scikit-learn imports for the ML model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV # For hyperparameter tuning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight # For calculating class weights manually if needed

# ---------------------- Logging Setup ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('FlaskAppML')

# ---------------------- Load Environment Variables ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
    FLASK_PORT: int = int(os.environ.get('PORT', 5000)) # Default to 5000, use environment variable if set
except Exception as e:
    logger.critical(f"❌ Failed to load essential environment variables: {e}")
    exit(1)

logger.info(f"Binance API Key: {'Available' if API_KEY else 'Not available'}")
logger.info(f"Database URL: {'Available' if DB_URL else 'Not available'}")

# ---------------------- Global Constants and Variables ----------------------
ML_TRAINING_TIMEFRAME: str = '5m'
ML_TRAINING_LOOKBACK_DAYS: int = 30
ML_MODEL_NAME: str = 'DecisionTree_Scalping_V1'
TARGET_PERCENT_CHANGE: float = 0.01 # 1% price increase
TARGET_LOOKAHEAD_CANDLES: int = 3 # Look 3 candles ahead for target

# Indicator parameters (should match those in ml.py and c4.py)
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

# Global variables for Binance client and DB connection
binance_client: Client = None
db_conn: psycopg2.extensions.connection = None

# Define the features that will be used for the ML model
# This list will be dynamically extended with lagged features
FEATURE_COLUMNS = [
    f'ema_{EMA_SHORT_PERIOD}', f'ema_{EMA_LONG_PERIOD}', 'vwma',
    'rsi', 'atr', 'bb_upper', 'bb_lower', 'bb_middle',
    'macd', 'macd_signal', 'macd_hist',
    'adx', 'di_plus', 'di_minus', 'vwap', 'obv',
    'supertrend', 'supertrend_trend'
]

# ---------------------- Binance Client Setup ----------------------
def init_binance_client() -> None:
    """Initializes Binance client."""
    global binance_client
    try:
        if binance_client is None:
            logger.info("ℹ️ [Binance] Initializing Binance client...")
            binance_client = Client(API_KEY, API_SECRET)
            binance_client.ping()
            server_time = binance_client.get_server_time()
            logger.info(f"✅ [Binance] Binance client initialized. Server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
    except BinanceRequestException as req_err:
        logger.critical(f"❌ [Binance] Binance request error (network or request issue): {req_err}")
        binance_client = None # Reset client on failure
    except BinanceAPIException as api_err:
        logger.critical(f"❌ [Binance] Binance API error (invalid keys or server issue): {api_err}")
        binance_client = None # Reset client on failure
    except Exception as e:
        logger.critical(f"❌ [Binance] Unexpected failure in Binance client initialization: {e}", exc_info=True)
        binance_client = None # Reset client on failure

def get_exchange_info() -> dict:
    """Fetches exchange information from Binance."""
    init_binance_client()
    if binance_client:
        try:
            logger.info("ℹ️ [Binance] Fetching exchange information...")
            info = binance_client.get_exchange_info()
            logger.info("✅ [Binance] Exchange information fetched successfully.")
            return info
        except BinanceAPIException as e:
            logger.error(f"❌ [Binance] Error fetching exchange info: {e}")
        except BinanceRequestException as e:
            logger.error(f"❌ [Binance] Network error fetching exchange info: {e}")
        except Exception as e:
            logger.error(f"❌ [Binance] Unexpected error fetching exchange info: {e}", exc_info=True)
    return {}

def get_all_trading_symbols() -> list[str]:
    """Retrieves all currently trading symbols from Binance."""
    exchange_info = get_exchange_info()
    if exchange_info and 'symbols' in exchange_info:
        trading_symbols = [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING']
        return trading_symbols
    return []

# ---------------------- Database Connection Setup ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    """Initializes database connection and creates ml_models table if it doesn't exist."""
    global db_conn
    logger.info("[DB] Starting database initialization...")
    for attempt in range(retries):
        try:
            logger.info(f"[DB] Attempting to connect to database (Attempt {attempt + 1}/{retries})...")
            db_conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            db_conn.autocommit = False
            with db_conn.cursor() as cur:
                logger.info("✅ [DB] Successfully connected to database.")
                # --- Create ml_models table ---
                logger.info("[DB] Checking for/creating 'ml_models' table...")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ml_models (
                        id SERIAL PRIMARY KEY,
                        model_name TEXT NOT NULL UNIQUE,
                        model_data BYTEA NOT NULL,
                        trained_at TIMESTAMP DEFAULT NOW(),
                        metrics JSONB
                    );""")
                db_conn.commit()
                logger.info("✅ [DB] 'ml_models' table exists or created.")
                logger.info("✅ [DB] Database initialized successfully.")
            return
        except OperationalError as op_err:
            logger.error(f"❌ [DB] Operational error connecting to DB (Attempt {attempt + 1}): {op_err}")
            if db_conn: db_conn.rollback()
            if attempt == retries - 1:
                logger.critical("❌ [DB] All database connection attempts failed.")
                raise op_err
            time.sleep(delay)
        except Exception as e:
            logger.critical(f"❌ [DB] Unexpected failure in database initialization (Attempt {attempt + 1}): {e}", exc_info=True)
            if db_conn: db_conn.rollback()
            if attempt == retries - 1:
                logger.critical("❌ [DB] All database connection attempts failed.")
                raise e
            time.sleep(delay)
    logger.critical("❌ [DB] Failed to connect to database after multiple attempts.")
    exit(1)

def cleanup_resources() -> None:
    """Closes database connection."""
    global db_conn
    logger.info("ℹ️ [Cleanup] Closing resources...")
    if db_conn:
        try:
            db_conn.close()
            db_conn = None # Set to None after closing
            logger.info("✅ [DB] Database connection closed.")
        except Exception as close_err:
            logger.error(f"⚠️ [DB] Error closing database connection: {close_err}")

# ---------------------- Data Fetching and Indicator Calculation Functions ----------------------
def fetch_historical_data(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fetches historical candlestick data from Binance."""
    if not binance_client:
        logger.error("❌ [Data] Binance client not initialized for data fetching.")
        return pd.DataFrame()
    try:
        start_dt = datetime.utcnow() - timedelta(days=days + 1)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        logger.debug(f"ℹ️ [Data] Fetching {interval} data for {symbol} since {start_str} (limit 1000 candles)...")

        klines = binance_client.get_historical_klines(symbol, interval, start_str, limit=1000)

        if not klines:
            logger.warning(f"⚠️ [Data] No historical data ({interval}) for {symbol} for the requested period.")
            return pd.DataFrame()

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
            logger.warning(f"⚠️ [Data] DataFrame for {symbol} is empty after dropping essential NaN values.")
            return pd.DataFrame()

        logger.debug(f"✅ [Data] Fetched and processed {len(df)} historical candles ({interval}) for {symbol}.")
        return df

    except BinanceAPIException as api_err:
        logger.error(f"❌ [Data] Binance API error while fetching data for {symbol}: {api_err}")
        return pd.DataFrame()
    except BinanceRequestException as req_err:
        logger.error(f"❌ [Data] Request or network error while fetching data for {symbol}: {req_err}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"❌ [Data] Unexpected error while fetching historical data for {symbol}: {e}", exc_info=True)
        return pd.DataFrame()

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
            final_ub[i] = final_ub[i-1] if i > 0 else np.nan
            final_lb[i] = final_lb[i-1] if i > 0 else np.nan
            st[i] = st[i-1] if i > 0 else np.nan
            st_trend[i] = st_trend[i-1] if i > 0 else 0
            continue

        # Update final_ub
        if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]

        # Update final_lb
        if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]

        # Determine SuperTrend and Trend Direction
        if st_trend[i-1] == -1: # If previous trend was downtrend
            if close[i] > final_ub[i]: # Price crosses above final_ub, trend changes to uptrend
                st_trend[i] = 1
                st[i] = final_lb[i]
            else: # Price remains below final_ub, trend remains downtrend
                st_trend[i] = -1
                st[i] = final_ub[i]
        elif st_trend[i-1] == 1: # If previous trend was uptrend
            if close[i] < final_lb[i]: # Price crosses below final_lb, trend changes to downtrend
                st_trend[i] = -1
                st[i] = final_ub[i]
            else: # Price remains above final_lb, trend remains uptrend
                st_trend[i] = 1
                st[i] = final_lb[i]
        else: # Initial state or no strong previous trend (can be 0 or NaN)
            if close[i] > final_ub[i]:
                st_trend[i] = 1
                st[i] = final_lb[i]
            elif close[i] < final_lb[i]:
                st_trend[i] = -1
                st[i] = final_ub[i]
            else:
                st_trend[i] = 0 # No clear trend
                st[i] = np.nan # Keep as NaN or previous value if desired

    df_st['final_ub'] = final_ub
    df_st['final_lb'] = final_lb
    df_st['supertrend'] = st
    df_st['supertrend_trend'] = st_trend
    
    # Fill initial NaNs for supertrend and trend (e.g., first 'period' rows)
    df_st['supertrend'] = df_st['supertrend'].bfill()
    df_st['supertrend_trend'] = df_st['supertrend_trend'].bfill()

    df_st.drop(columns=['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, errors='ignore')
    return df_st

def populate_all_indicators(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calculates all required indicators and adds lagged features for the strategy.
    Returns the processed DataFrame and the list of feature columns used.
    """
    if df.empty:
        return pd.DataFrame(), []

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

        # Add lagged features
        for lag in range(1, 4): # Lags 1, 2, 3
            df_calc[f'close_lag{lag}'] = df_calc['close'].shift(lag)
        for lag in range(1, 3): # Lags 1, 2
            df_calc[f'rsi_lag{lag}'] = df_calc['rsi'].shift(lag)
            df_calc[f'macd_lag{lag}'] = df_calc['macd'].shift(lag)
            df_calc[f'supertrend_trend_lag{lag}'] = df_calc['supertrend_trend'].shift(lag)

    except Exception as e:
        logger.error(f"❌ Error calculating indicators or lagged features: {e}", exc_info=True)
        return pd.DataFrame(), []

    # Dynamically build the list of feature columns that will be used for the ML model
    current_feature_columns_generated = list(FEATURE_COLUMNS) # Start with original features
    for lag in range(1, 4):
        current_feature_columns_generated.append(f'close_lag{lag}')
    for lag in range(1, 3):
        current_feature_columns_generated.append(f'rsi_lag{lag}')
        current_feature_columns_generated.append(f'macd_lag{lag}')
        current_feature_columns_generated.append(f'supertrend_trend_lag{lag}')


    # Ensure all feature columns exist and are numeric
    for col in current_feature_columns_generated:
        if col not in df_calc.columns:
            logger.warning(f"⚠️ Missing feature column after calculation: {col}. Adding as NaN.")
            df_calc[col] = np.nan # Add missing column as NaN
        else:
            df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')

    # Drop rows with NaN values in feature columns
    initial_len = len(df_calc)
    df_cleaned = df_calc.dropna(subset=current_feature_columns_generated).copy()
    if len(df_cleaned) < initial_len:
        logger.debug(f"ℹ️ Dropped {initial_len - len(df_cleaned)} rows due to NaN values in indicators or lagged features.")

    # Return the cleaned DataFrame and the list of features
    return df_cleaned, current_feature_columns_generated

# ---------------------- Main Training Logic ----------------------
def get_crypto_symbols(filename: str = 'crypto_list.txt') -> list[str]:
    """
    Reads the list of currency symbols from a text file (assumes base symbol per line, e.g., 'BTC'),
    appends 'USDT' to each, and validates them against Binance.
    """
    base_symbols: list[str] = []
    logger.info(f"ℹ️ [Data] Reading base symbol list from '{filename}'...")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)

        if not os.path.exists(file_path):
            file_path = os.path.abspath(filename)
            if not os.path.exists(file_path):
                logger.error(f"❌ [Data] File '{filename}' not found in script directory or current directory.")
                return []
            else:
                logger.warning(f"⚠️ [Data] File '{filename}' not found in script directory. Using file in current directory: '{file_path}'")

        with open(file_path, 'r', encoding='utf-8') as f:
            # Read each line, strip whitespace, convert to uppercase, and append 'USDT'
            base_symbols = [f"{line.strip().upper()}USDT"
                           for line in f if line.strip() and not line.startswith('#')]
        base_symbols = sorted(list(set(base_symbols)))
        logger.info(f"ℹ️ [Data] Read {len(base_symbols)} initial symbols (with USDT appended) from '{file_path}'.")

        # Validate symbols against Binance
        logger.info("ℹ️ [Binance] Validating symbols against Binance available trading pairs...")
        all_binance_symbols = get_all_trading_symbols()
        valid_symbols = [s for s in base_symbols if s in all_binance_symbols]
        invalid_symbols = [s for s in base_symbols if s not in all_binance_symbols]

        if invalid_symbols:
            logger.warning(f"⚠️ The following symbols (after appending USDT) from '{filename}' are not active trading pairs on Binance and will be skipped: {', '.join(invalid_symbols)}")
        if not valid_symbols:
            logger.critical("❌ No valid trading symbols found after Binance validation. Please check your 'crypto_list.txt'.")

        logger.info(f"✅ [Data] {len(valid_symbols)} valid symbols found for training.")
        return valid_symbols
    except Exception as e:
        logger.error(f"❌ [Data] Error reading file '{filename}': {e}", exc_info=True)
        return []

def train_and_save_model() -> None:
    """
    Fetches data, calculates indicators, trains a Decision Tree Classifier with hyperparameter tuning,
    and saves the model and its metrics to the database.
    """
    logger.info("🚀 Starting ML model training process...")
    # Ensure init_binance_client and init_db are called before this function
    # if this function is called directly on startup.
    # If called by scheduler, they are already handled.

    symbols = get_crypto_symbols()
    if not symbols:
        logger.critical("❌ No valid symbols for training. Exiting training process.")
        cleanup_resources()
        return

    all_features_df = pd.DataFrame()
    all_targets_series = pd.Series(dtype=int)
    processed_symbols_count = 0
    
    # This will hold the final list of feature columns used for training
    final_feature_columns_for_model: List[str] = []

    for symbol in symbols:
        logger.info(f"ℹ️ Fetching data and calculating indicators for {symbol}...")
        df = fetch_historical_data(symbol, ML_TRAINING_TIMEFRAME, ML_TRAINING_LOOKBACK_DAYS)
        if df.empty:
            logger.warning(f"⚠️ Skipping {symbol} due to insufficient data.")
            continue

        df_processed, current_features_for_this_symbol = populate_all_indicators(df)
        
        # CRITICAL DEBUGGING STEP: Add explicit checks before proceeding
        if df_processed is None:
            logger.error(f"FATAL ERROR: populate_all_indicators returned None for {symbol}. Skipping.")
            continue
        if df_processed.empty:
            logger.warning(f"⚠️ Skipping {symbol} due to insufficient data after indicator calculation (df_processed is empty).")
            continue
        if 'close' not in df_processed.columns:
            logger.error(f"FATAL ERROR: 'close' column missing from df_processed for {symbol}. Columns: {df_processed.columns.tolist()}. Skipping.")
            continue
        # End CRITICAL DEBUGGING STEP

        # Set the final_feature_columns_for_model based on the first successfully processed symbol
        if not final_feature_columns_for_model:
            final_feature_columns_for_model = current_features_for_this_symbol
            logger.info(f"ℹ️ Set final feature columns for model training: {final_feature_columns_for_model}")
        else:
            # Optional: Add a check here to ensure subsequent symbols have the same features
            if set(final_feature_columns_for_model) != set(current_features_for_this_symbol):
                logger.warning(f"⚠️ Feature columns mismatch for {symbol}. Expected {len(final_feature_columns_for_model)}, got {len(current_features_for_this_symbol)}. Skipping.")
                continue
            # No need to reorder df_processed here, as it retains all original columns + new ones.
            # We will select the features explicitly for X later.


        # Define the target variable: 1 if close price increases by TARGET_PERCENT_CHANGE
        # within the next TARGET_LOOKAHEAD_CANDLES, 0 otherwise.
        # We look for the maximum close price in the future window.
        future_max_close = df_processed['close'].rolling(window=TARGET_LOOKAHEAD_CANDLES, min_periods=1).max().shift(-(TARGET_LOOKAHEAD_CANDLES - 1))
        df_processed['target'] = ((future_max_close / df_processed['close']) >= (1 + TARGET_PERCENT_CHANGE)).astype(int)

        # Drop rows with NaN values in target or feature columns
        df_processed.dropna(subset=['target'] + final_feature_columns_for_model, inplace=True)

        if df_processed.empty:
            logger.warning(f"⚠️ Skipping {symbol} due to insufficient data after target setup and NaN removal.")
            continue

        # Now, select only the features for X. This ensures X has only the expected columns.
        all_features_df = pd.concat([all_features_df, df_processed[final_feature_columns_for_model]])
        all_targets_series = pd.concat([all_targets_series, df_processed['target']])
        processed_symbols_count += 1

    if all_features_df.empty or all_targets_series.empty:
        logger.critical("❌ No sufficient data to train the model after processing symbols. Exiting training.")
        cleanup_resources()
        return

    # Ensure data is sorted by index (timestamp) for time-series aware splitting
    all_features_df.sort_index(inplace=True)
    all_targets_series.sort_index(inplace=True)

    X = all_features_df
    y = all_targets_series

    logger.info(f"✅ Aggregated training data from {processed_symbols_count} symbols. Data size: {len(X)} samples.")
    
    # Log class distribution
    class_distribution = y.value_counts(normalize=True)
    logger.info(f"ℹ️ Target Class Distribution:\n{class_distribution.to_string()}")
    if 1 not in class_distribution or class_distribution[1] < 0.05: # Warn if bullish class is less than 5%
        logger.warning("⚠️ Warning: Bullish class (1) is severely underrepresented in the training data. Consider data augmentation or more aggressive class weighting.")


    # Time-series aware split: 80% for training, 20% for testing
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    logger.info(f"ℹ️ Training data size: {len(X_train)}, Test data size: {len(X_test)}")

    if X_train.empty or y_train.empty:
        logger.critical("❌ Training sets are empty after time-series split. Cannot train.")
        cleanup_resources()
        return
    if X_test.empty or y_test.empty:
        logger.warning("⚠️ Test sets are empty after time-series split. Model will be evaluated on training data only.")
        # In a real scenario, you might want to skip evaluation or log a critical error.
        # For now, we'll proceed but note the issue.

    # Hyperparameter tuning for Decision Tree Classifier
    logger.info("ℹ️ Performing Grid Search for Decision Tree hyperparameters...")
    param_grid = {
        'max_depth': [5, 10, 15, 20],
        'min_samples_leaf': [1, 5, 10, 20],
        'criterion': ['gini', 'entropy']
    }
    # MODIFICATION: Add class_weight='balanced' to handle class imbalance
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42, class_weight='balanced'), param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    logger.info(f"✅ Best Decision Tree parameters found: {best_params}")
    logger.info(f"✅ Best cross-validation accuracy: {best_score:.4f}")
    logger.info("✅ Model trained with best parameters.")

    # Evaluate the model on the test set
    logger.info("ℹ️ Evaluating model performance on test set...")
    if not X_test.empty:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # Ensure classification_report includes all labels (0 and 1) even if one is missing in y_pred
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0, labels=[0, 1])
        conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
    else:
        # Fallback to training data evaluation if test set is empty
        logger.warning("⚠️ Test set is empty, evaluating on training data instead.")
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        report = classification_report(y_train, y_pred, output_dict=True, zero_division=0, labels=[0, 1])
        conf_matrix = confusion_matrix(y_train, y_pred, labels=[0, 1]).tolist()


    logger.info(f"✅ Model Accuracy (Test/Train): {accuracy:.4f}")
    # Log classification report in a more readable JSON format
    logger.info(f"Classification Report (Test/Train):\n{json.dumps(report, indent=2, ensure_ascii=False)}")
    logger.info(f"Confusion Matrix (Test/Train):\n{json.dumps(conf_matrix, indent=2, ensure_ascii=False)}")

    # Prepare metrics for storage
    model_metrics = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'features': final_feature_columns_for_model, # Store feature names for consistency
        'training_symbols_count': processed_symbols_count,
        'last_trained_utc': datetime.utcnow().isoformat(),
        'target_definition': {
            'percent_change': TARGET_PERCENT_CHANGE,
            'lookahead_candles': TARGET_LOOKAHEAD_CANDLES
        },
        'best_hyperparameters': best_params,
        'best_cv_score': best_score
    }

    # Serialize the model
    pickled_model = pickle.dumps(model)
    logger.info(f"ℹ️ Pickled model size: {len(pickled_model) / (1024*1024):.2f} MB")

    # Save to database
    try:
        if db_conn: # Ensure db_conn is not None before using
            with db_conn.cursor() as db_cur:
                # Check if a model with this name already exists
                db_cur.execute("SELECT id FROM ml_models WHERE model_name = %s;", (ML_MODEL_NAME,))
                existing_model = db_cur.fetchone()

                if existing_model:
                    logger.info(f"ℹ️ Updating existing model '{ML_MODEL_NAME}' in database.")
                    update_query = sql.SQL("""
                        UPDATE ml_models
                        SET model_data = %s, trained_at = NOW(), metrics = %s
                        WHERE model_name = %s;
                    """)
                    db_cur.execute(update_query, (psycopg2.Binary(pickled_model), json.dumps(model_metrics, ensure_ascii=False), ML_MODEL_NAME))
                else:
                    logger.info(f"ℹ️ Inserting new model '{ML_MODEL_NAME}' into database.")
                    insert_query = sql.SQL("""
                        INSERT INTO ml_models (model_name, model_data, trained_at, metrics)
                        VALUES (%s, %s, NOW(), %s);
                    """)
                    db_cur.execute(insert_query, (ML_MODEL_NAME, psycopg2.Binary(pickled_model), json.dumps(model_metrics, ensure_ascii=False)))
            db_conn.commit()
            logger.info("✅ Model and its metrics saved to database successfully.")
        else:
            logger.error("❌ Database connection is not active. Cannot save model.")
    except psycopg2.Error as db_err:
        logger.error(f"❌ Database error while saving model: {db_err}", exc_info=True)
        if db_conn: db_conn.rollback()
    except Exception as e:
        logger.error(f"❌ Unexpected error while saving model: {e}", exc_info=True)
        if db_conn: db_conn.rollback()

    cleanup_resources()
    logger.info("🏁 ML model training process finished.")

# ---------------------- Flask Application ----------------------
app = Flask(__name__)

@app.route('/')
def home():
    """Basic home route to keep the service alive."""
    logger.info("✅ Home route accessed.")
    return "ML Training Service is Running!"

@app.route('/status')
def status():
    """Provides status of the service and last training run."""
    logger.info("ℹ️ Status route accessed.")
    status_info = {
        "service_status": "Running",
        "last_training_attempt_utc": "N/A",
        "next_training_schedule_utc": "N/A",
        "binance_client_initialized": binance_client is not None,
        "database_connection_active": db_conn is not None
    }
    scheduler_jobs = scheduler.get_jobs()
    if scheduler_jobs:
        for job in scheduler_jobs:
            if job.id == 'daily_ml_training':
                status_info["next_training_schedule_utc"] = job.next_run_time.isoformat() if job.next_run_time else "N/A"
                break

    # Fetch last training time from DB
    try:
        init_db() # Ensure connection is open
        with db_conn.cursor() as cur:
            cur.execute("SELECT trained_at, metrics FROM ml_models WHERE model_name = %s ORDER BY trained_at DESC LIMIT 1;", (ML_MODEL_NAME,))
            result = cur.fetchone()
            if result:
                status_info["last_training_attempt_utc"] = result['trained_at'].isoformat()
                # Ensure metrics are JSON serializable (e.g., handle numpy types)
                status_info["last_training_metrics"] = json.loads(result['metrics']) if isinstance(result['metrics'], str) else result['metrics']
            else:
                status_info["last_training_metrics"] = "No model trained yet."
    except Exception as e:
        logger.error(f"❌ Error fetching last training status from DB: {e}", exc_info=True)
        status_info["last_training_attempt_utc"] = "Error fetching from DB"
        status_info["last_training_metrics"] = "Error fetching from DB"
    finally:
        cleanup_resources() # Close connection after use

    return jsonify(status_info)

@app.route('/trigger_training', methods=['POST'])
def trigger_training_manual():
    """Manually triggers the ML model training."""
    logger.info("ℹ️ Manual training trigger received.")
    try:
        # Run training in a separate thread to avoid blocking the Flask app
        training_thread = Thread(target=train_and_save_model)
        training_thread.start()
        return jsonify({"message": "ML training initiated in background.", "status": "success"}), 202
    except Exception as e:
        logger.error(f"❌ Error initiating manual training: {e}", exc_info=True)
        return jsonify({"message": f"Failed to initiate training: {e}", "status": "error"}), 500

# ---------------------- Scheduler Setup ----------------------
scheduler = BackgroundScheduler()

def start_scheduler():
    """Starts the APScheduler."""
    # Schedule the training to run daily at a specific time (e.g., 00:00 UTC)
    # This time can be adjusted based on when you want the training to occur.
    # 'cron' allows for precise scheduling.
    scheduler.add_job(train_and_save_model, 'cron', hour=0, minute=0, id='daily_ml_training', replace_existing=True)
    logger.info("✅ Scheduled daily ML training for 00:00 UTC.")
    scheduler.start()
    logger.info("✅ Scheduler started.")

# ---------------------- Application Entry Point ----------------------
if __name__ == '__main__':
    # Initialize Binance client and DB connection on app startup
    # These will be used by the scheduler and manual trigger
    init_binance_client()
    init_db()

    # MODIFICATION: Run training immediately on startup
    logger.info("🚀 Running ML model training on startup...")
    train_and_save_model() # This will train and save the model to DB

    # Start the scheduler in a separate thread to avoid blocking the Flask app
    # This is crucial for Render's free tier, as the web server needs to respond.
    scheduler_thread = Thread(target=start_scheduler)
    scheduler_thread.start()

    logger.info(f"🚀 Starting Flask app on port {FLASK_PORT}...")
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=False) # debug=False for production
