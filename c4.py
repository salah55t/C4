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

# ---------------------- Logging Setup ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_elliott_fib.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBot')

# ---------------------- Load Environment Variables ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
    # WEBHOOK_URL is optional, but Flask will always run for Render compatibility
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
except Exception as e:
     logger.critical(f"❌ Failed to load essential environment variables: {e}")
     exit(1)

logger.info(f"Binance API Key: {'Available' if API_KEY else 'Not available'}")
logger.info(f"Telegram Token: {TELEGRAM_TOKEN[:10]}...{'*' * (len(TELEGRAM_TOKEN)-10)}")
logger.info(f"Telegram Chat ID: {CHAT_ID}")
logger.info(f"Database URL: {'Available' if DB_URL else 'Not available'}")
logger.info(f"Webhook URL: {WEBHOOK_URL if WEBHOOK_URL else 'Not specified'} (Flask will always run for Render)")

# ---------------------- Constants and Global Variables Setup ----------------------
TRADE_VALUE: float = 10.0
MAX_OPEN_TRADES: int = 5
SIGNAL_GENERATION_TIMEFRAME: str = '15m' # Changed to 15 minutes
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 3
SIGNAL_TRACKING_TIMEFRAME: str = '15m' # Changed to 15 minutes
SIGNAL_TRACKING_LOOKBACK_DAYS: int = 1

# Indicator Parameters (MUST match ml.py)
RSI_PERIOD: int = 9 
RSI_OVERSOLD: int = 30 # Not directly used for signal, but good to keep for context if needed later
RSI_OVERBOUGHT: int = 70 # Not directly used for signal, but good to keep for context if needed later
VOLUME_LOOKBACK_CANDLES: int = 1 
RSI_MOMENTUM_LOOKBACK_CANDLES: int = 2 

ENTRY_ATR_PERIOD: int = 10 
ENTRY_ATR_MULTIPLIER: float = 1.5 

SUPERTRAND_PERIOD: int = 10 
SUPERTRAND_MULTIPLIER: float = 3.0 

# Ichimoku Cloud Parameters (MUST match ml.py)
TENKAN_PERIOD: int = 9
KIJUN_PERIOD: int = 26
SENKOU_SPAN_B_PERIOD: int = 52
CHIKOU_LAG: int = 26 

# Fibonacci & S/R Parameters (MUST match ml.py)
FIB_SR_LOOKBACK_WINDOW: int = 50 

MIN_PROFIT_MARGIN_PCT: float = 1.0 
MIN_VOLUME_15M_USDT: float = 50000.0 

TARGET_APPROACH_THRESHOLD_PCT: float = 0.005

BINANCE_FEE_RATE: float = 0.001

BASE_ML_MODEL_NAME: str = 'DecisionTree_Scalping_V1' # Must match the base name used in ml.py

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
cur: Optional[psycopg2.extensions.cursor] = None
client: Optional[Client] = None
ticker_data: Dict[str, float] = {}
ml_models: Dict[str, Any] = {} # Global dictionary to hold loaded ML models, keyed by symbol

# ---------------------- Binance Client Setup ----------------------
try:
    logger.info("ℹ️ [Binance] Initializing Binance client...")
    client = Client(API_KEY, API_SECRET)
    client.ping()
    server_time = client.get_server_time()
    logger.info(f"✅ [Binance] Binance client initialized. Server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
except BinanceRequestException as req_err:
     logger.critical(f"❌ [Binance] Binance request error (network or request issue): {req_err}")
     exit(1)
except BinanceAPIException as api_err:
     logger.critical(f"❌ [Binance] Binance API error (invalid keys or server issue): {api_err}")
     exit(1)
except Exception as e:
    logger.critical(f"❌ [Binance] Unexpected failure in Binance client initialization: {e}")
    exit(1)

# ---------------------- Additional Indicator Functions ----------------------
def get_fear_greed_index() -> str:
    """Fetches the Fear & Greed Index from alternative.me and translates classification to Arabic."""
    classification_translation_ar = {
        "Extreme Fear": "خوف شديد", "Fear": "خوف", "Neutral": "محايد",
        "Greed": "جشع", "Extreme Greed": "جشع شديد",
    }
    url = "https://api.alternative.me/fng/" 
    logger.debug(f"ℹ️ [Indicators] Fetching Fear & Greed Index from {url}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        value = int(data["data"][0]["value"])
        classification_en = data["data"][0]["value_classification"]
        classification_ar = classification_translation_ar.get(classification_en, classification_en)
        logger.debug(f"✅ [Indicators] Fear & Greed Index: {value} ({classification_ar})")
        return f"{value} ({classification_ar})"
    except requests.exceptions.RequestException as e:
         logger.error(f"❌ [Indicators] Network error while fetching Fear & Greed Index: {e}")
         return "N/A (Network Error)"
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"❌ [Indicators] Data format error for Fear & Greed Index: {e}")
        return "N/A (Data Error)"
    except Exception as e:
        logger.error(f"❌ [Indicators] Unexpected error while fetching Fear & Greed Index: {e}", exc_info=True)
        return "N/A (Unknown Error)"

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """
    Fetches historical candlestick data from Binance for a specified number of days.
    This function relies on python-binance's get_historical_klines to handle
    internal pagination for large data ranges.
    """
    if not client:
        logger.error("❌ [Data] Binance client not initialized for data fetching.")
        return None
    try:
        # Calculate the start date for the entire data range needed
        start_dt = datetime.utcnow() - timedelta(days=days + 1)
        start_str_overall = start_dt.strftime("%Y-%m-%d %H:%M:%S")

        logger.debug(f"ℹ️ [Data] Fetching {interval} data for {symbol} from {start_str_overall} onwards...")

        # Map interval string to Binance client constant
        binance_interval = None
        if interval == '15m':
            binance_interval = Client.KLINE_INTERVAL_15MINUTE
        elif interval == '5m':
            binance_interval = Client.KLINE_INTERVAL_5MINUTE
        elif interval == '1h':
            binance_interval = Client.KLINE_INTERVAL_1HOUR
        elif interval == '4h':
            binance_interval = Client.KLINE_INTERVAL_4HOUR
        elif interval == '1d':
            binance_interval = Client.KLINE_INTERVAL_1DAY
        else:
            logger.error(f"❌ [Data] Unsupported interval: {interval}")
            return None

        # Call get_historical_klines for the entire period.
        # The python-binance library is designed to handle internal pagination
        # if the requested range exceeds the API's single-request limit (e.g., 1000 klines).
        klines = client.get_historical_klines(symbol, binance_interval, start_str_overall)

        if not klines:
            logger.warning(f"⚠️ [Data] No historical ({interval}) data for {symbol} for the requested period.")
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
            logger.debug(f"ℹ️ [Data] {symbol}: Dropped {initial_len - len(df)} rows due to NaN values in OHLCV data.")

        if df.empty:
            logger.warning(f"⚠️ [Data] DataFrame for {symbol} is empty after removing essential NaN values.")
            return None

        # Sort by index (timestamp) to ensure chronological order
        df.sort_index(inplace=True)

        logger.debug(f"✅ [Data] Fetched and processed {len(df)} historical ({interval}) candles for {symbol}.")
        return df

    except BinanceAPIException as api_err:
         logger.error(f"❌ [Data] Binance API error while fetching data for {symbol}: {api_err}")
         return None
    except BinanceRequestException as req_err:
         logger.error(f"❌ [Data] Request or network error while fetching data for {symbol}: {req_err}")
         return None
    except Exception as e:
        logger.error(f"❌ [Data] Unexpected error while fetching historical data for {symbol}: {e}", exc_info=True)
        return None


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculates Exponential Moving Average (EMA)."""
    if series is None or series.isnull().all() or len(series) < span:
        return pd.Series(index=series.index if series is not None else None, dtype=float)
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi_indicator(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """Calculates Relative Strength Index (RSI)."""
    df = df.copy()
    if 'close' not in df.columns or df['close'].isnull().all():
        logger.warning("⚠️ [Indicator RSI] 'close' column is missing or empty.")
        df['rsi'] = np.nan
        return df
    if len(df) < period:
        logger.warning(f"⚠️ [Indicator RSI] Insufficient data ({len(df)} < {period}) to calculate RSI.")
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
        logger.warning("⚠️ [Indicator ATR] 'high', 'low', 'close' columns are missing or empty.")
        df['atr'] = np.nan
        return df
    if len(df) < period + 1:
        logger.warning(f"⚠️ [Indicator ATR] Insufficient data ({len(df)} < {period + 1}) to calculate ATR.")
        df['atr'] = np.nan
        return df

    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()

    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)

    df['atr'] = tr.ewm(span=period, adjust=False).mean()
    return df

def calculate_supertrend(df: pd.DataFrame, period: int = SUPERTRAND_PERIOD, multiplier: float = SUPERTRAND_MULTIPLIER) -> pd.DataFrame:
    """Calculates the Supertrend indicator."""
    df = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator Supertrend] 'high', 'low', 'close' columns are missing or empty. Cannot calculate Supertrend.")
        df['supertrend'] = np.nan
        df['supertrend_direction'] = 0 # Neutral if cannot calculate
        return df

    # Ensure ATR is already calculated
    if 'atr' not in df.columns:
        df = calculate_atr_indicator(df, period=period) # Use Supertrend period for ATR if not already calculated
        if 'atr' not in df.columns or df['atr'].isnull().all().any():
            logger.warning("⚠️ [Indicator Supertrend] ATR calculation failed. Cannot calculate Supertrend.")
            df['supertrend'] = np.nan
            df['supertrend_direction'] = 0
            return df

    # Calculate Basic Upper and Lower Bands
    df['basic_upper_band'] = ((df['high'] + df['low']) / 2) + (multiplier * df['atr'])
    df['basic_lower_band'] = ((df['high'] + df['low']) / 2) - (multiplier * df['atr'])

    # Initialize Final Upper and Lower Bands
    df['final_upper_band'] = 0.0
    df['final_lower_band'] = 0.0

    # Initialize Supertrend and Direction
    df['supertrend'] = 0.0
    df['supertrend_direction'] = 0 # 1 for uptrend, -1 for downtrend, 0 for neutral/flat

    # Determine Supertrend value and direction
    for i in range(1, len(df)):
        # Final Upper Band
        if df['basic_upper_band'].iloc[i] < df['final_upper_band'].iloc[i-1] or \
           df['close'].iloc[i-1] > df['final_upper_band'].iloc[i-1]:
            df.loc[df.index[i], 'final_upper_band'] = df['basic_upper_band'].iloc[i]
        else:
            df.loc[df.index[i], 'final_upper_band'] = df['final_upper_band'].iloc[i-1]

        # Final Lower Band
        if df['basic_lower_band'].iloc[i] > df['final_lower_band'].iloc[i-1] or \
           df['close'].iloc[i-1] < df['final_lower_band'].iloc[i-1]:
            df.loc[df.index[i], 'final_lower_band'] = df['basic_lower_band'].iloc[i]
        else:
            df.loc[df.index[i], 'final_lower_band'] = df['final_lower_band'].iloc[i-1]

        # Supertrend logic
        if df['supertrend_direction'].iloc[i-1] == 1: # Previous was uptrend
            if df['close'].iloc[i] < df['final_upper_band'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['final_upper_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = -1 # Change to downtrend
            else:
                df.loc[df.index[i], 'supertrend'] = df['final_lower_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = 1 # Remain uptrend
        elif df['supertrend_direction'].iloc[i-1] == -1: # Previous was downtrend
            if df['close'].iloc[i] > df['final_lower_band'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['final_lower_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = 1 # Change to uptrend
            else:
                df.loc[df.index[i], 'supertrend'] = df['final_upper_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = -1 # Remain downtrend
        else: # Initial state or neutral
            if df['close'].iloc[i] > df['final_lower_band'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['final_lower_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = 1
            elif df['close'].iloc[i] < df['final_upper_band'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['final_upper_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = -1
            else:
                df.loc[df.index[i], 'supertrend'] = df['close'].iloc[i] # Fallback
                df.loc[df.index[i], 'supertrend_direction'] = 0


    # Drop temporary columns
    df.drop(columns=['basic_upper_band', 'basic_lower_band', 'final_upper_band', 'final_lower_band'], inplace=True, errors='ignore')
    logger.debug(f"✅ [Indicator Supertrend] Supertrend calculated.")
    return df

def _calculate_btc_trend_feature(df_btc: pd.DataFrame) -> Optional[pd.Series]:
    """
    Calculates a numerical representation of Bitcoin's trend based on EMA20 and EMA50.
    Returns 1 for bullish (صعودي), -1 for bearish (هبوطي), 0 for neutral/sideways (محايد/تذبذب).
    """
    logger.debug("ℹ️ [Indicators] Calculating Bitcoin trend for features...")
    # Need enough data for EMA50, plus a few extra candles for robustness
    min_data_for_ema = 50 + 5 # 50 for EMA50, 5 buffer

    if df_btc is None or df_btc.empty or len(df_btc) < min_data_for_ema:
        logger.warning(f"⚠️ [Indicators] Insufficient BTC/USDT data ({len(df_btc) if df_btc is not None else 0} < {min_data_for_ema}) to calculate Bitcoin trend for features.")
        # Return a series of zeros (neutral) with the original index if data is insufficient
        return pd.Series(index=df_btc.index if df_btc is not None else None, data=0.0)

    df_btc_copy = df_btc.copy()
    df_btc_copy['close'] = pd.to_numeric(df_btc_copy['close'], errors='coerce')
    df_btc_copy.dropna(subset=['close'], inplace=True)

    if len(df_btc_copy) < min_data_for_ema:
        logger.warning(f"⚠️ [Indicators] Insufficient BTC/USDT data after NaN removal to calculate trend.")
        return pd.Series(index=df_btc.index, data=0.0) # Return neutral if not enough data after dropna

    ema20 = calculate_ema(df_btc_copy['close'], 20)
    ema50 = calculate_ema(df_btc_copy['close'], 50)

    # Combine EMAs and close into a single DataFrame for easier comparison
    ema_df = pd.DataFrame({'ema20': ema20, 'ema50': ema50, 'close': df_btc_copy['close']})
    ema_df.dropna(inplace=True) # Drop rows where any EMA or close is NaN

    if ema_df.empty:
        logger.warning("⚠️ [Indicators] EMA DataFrame is empty after NaN removal. Cannot calculate Bitcoin trend.")
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
    logger.debug(f"✅ [Indicators] Bitcoin trend feature calculated. Examples: {final_trend_series.tail().tolist()}")
    return final_trend_series


# NEW: Ichimoku Cloud Calculation (Copied from ml.py)
def calculate_ichimoku_cloud(df: pd.DataFrame, tenkan_period: int = TENKAN_PERIOD, kijun_period: int = KIJUN_PERIOD, senkou_span_b_period: int = SENKOU_SPAN_B_PERIOD, chikou_lag: int = CHIKOU_LAG) -> pd.DataFrame:
    """Calculates Ichimoku Cloud components and derived features."""
    df_ichimoku = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df_ichimoku.columns for col in required_cols) or df_ichimoku[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator Ichimoku] Missing or empty OHLC columns. Cannot calculate Ichimoku.")
        for col in ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span',
                    'ichimoku_tenkan_kijun_cross_signal', 'ichimoku_price_cloud_position', 'ichimoku_cloud_outlook']:
            df_ichimoku[col] = np.nan
        return df_ichimoku

    # Convert to numeric
    for col in required_cols:
        df_ichimoku[col] = pd.to_numeric(df_ichimoku[col], errors='coerce')

    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    df_ichimoku['tenkan_sen'] = (df_ichimoku['high'].rolling(window=tenkan_period, min_periods=1).max() +
                                 df_ichimoku['low'].rolling(window=tenkan_period, min_periods=1).min()) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    df_ichimoku['kijun_sen'] = (df_ichimoku['high'].rolling(window=kijun_period, min_periods=1).max() +
                                df_ichimoku['low'].rolling(window=kijun_period, min_periods=1).min()) / 2

    # Senkou Span A (Leading Span A): (Conversion Line + Base Line) / 2 plotted 26 periods ahead
    df_ichimoku['senkou_span_a'] = ((df_ichimoku['tenkan_sen'] + df_ichimoku['kijun_sen']) / 2).shift(kijun_period)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 plotted 26 periods ahead
    df_ichimoku['senkou_span_b'] = ((df_ichimoku['high'].rolling(window=senkou_span_b_period, min_periods=1).max() +
                                     df_ichimoku['low'].rolling(window=senkou_span_b_period, min_periods=1).min()) / 2).shift(kijun_period)

    # Chikou Span (Lagging Span): Close plotted 26 periods back
    df_ichimoku['chikou_span'] = df_ichimoku['close'].shift(-chikou_lag)

    # --- Derived Ichimoku Features ---
    # Tenkan/Kijun Cross Signal
    df_ichimoku['ichimoku_tenkan_kijun_cross_signal'] = 0
    if len(df_ichimoku) > 1:
        # Bullish cross: Tenkan-sen crosses above Kijun-sen
        df_ichimoku.loc[(df_ichimoku['tenkan_sen'].shift(1) < df_ichimoku['kijun_sen'].shift(1)) &
                        (df_ichimoku['tenkan_sen'] > df_ichimoku['kijun_sen']), 'ichimoku_tenkan_kijun_cross_signal'] = 1
        # Bearish cross: Tenkan-sen crosses below Kijun-sen
        df_ichimoku.loc[(df_ichimoku['tenkan_sen'].shift(1) > df_ichimoku['kijun_sen'].shift(1)) &
                        (df_ichimoku['tenkan_sen'] < df_ichimoku['kijun_sen']), 'ichimoku_tenkan_kijun_cross_signal'] = -1

    # Price vs Cloud Position (using current close price vs future cloud)
    df_ichimoku['ichimoku_price_cloud_position'] = 0 # 0 for inside, 1 for above, -1 for below
    # Price above cloud
    df_ichimoku.loc[(df_ichimoku['close'] > df_ichimoku[['senkou_span_a', 'senkou_span_b']].max(axis=1)), 'ichimoku_price_cloud_position'] = 1
    # Price below cloud
    df_ichimoku.loc[(df_ichimoku['close'] < df_ichimoku[['senkou_span_a', 'senkou_span_b']].min(axis=1)), 'ichimoku_price_cloud_position'] = -1

    # Cloud Outlook (future cloud's color)
    df_ichimoku['ichimoku_cloud_outlook'] = 0 # 0 for flat/mixed, 1 for bullish (green), -1 for bearish (red)
    df_ichimoku.loc[(df_ichimoku['senkou_span_a'] > df_ichimoku['senkou_span_b']), 'ichimoku_cloud_outlook'] = 1 # Green Cloud
    df_ichimoku.loc[(df_ichimoku['senkou_span_a'] < df_ichimoku['senkou_span_b']), 'ichimoku_cloud_outlook'] = -1 # Red Cloud

    logger.debug(f"✅ [Indicator Ichimoku] Ichimoku Cloud and derived features calculated.")
    return df_ichimoku


# NEW: Fibonacci Retracement Features (Copied from ml.py)
def calculate_fibonacci_features(df: pd.DataFrame, lookback_window: int = FIB_SR_LOOKBACK_WINDOW) -> pd.DataFrame:
    """
    Calculates Fibonacci Retracement levels from a recent swing (max/min in lookback window)
    and generates features based on current price position relative to these levels.
    """
    df_fib = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df_fib.columns for col in required_cols) or df_fib[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator Fibonacci] Missing or empty OHLC columns. Cannot calculate Fibonacci features.")
        for col in ['fib_236_retrace_dist_norm', 'fib_382_retrace_dist_norm', 'fib_618_retrace_dist_norm', 'is_price_above_fib_50']:
            df_fib[col] = np.nan
        return df_fib
    if len(df_fib) < lookback_window:
        logger.warning(f"⚠️ [Indicator Fibonacci] Insufficient data ({len(df_fib)} < {lookback_window}) for Fibonacci calculation.")
        for col in ['fib_236_retrace_dist_norm', 'fib_382_retrace_dist_norm', 'fib_618_retrace_dist_norm', 'is_price_above_fib_50']:
            df_fib[col] = np.nan
        return df_fib

    # Convert to numeric
    for col in required_cols:
        df_fib[col] = pd.to_numeric(df_fib[col], errors='coerce')

    df_fib['fib_236_retrace_dist_norm'] = np.nan
    df_fib['fib_382_retrace_dist_norm'] = np.nan
    df_fib['fib_618_retrace_dist_norm'] = np.nan
    df_fib['is_price_above_fib_50'] = 0

    for i in range(lookback_window - 1, len(df_fib)):
        window_df = df_fib.iloc[i - lookback_window + 1 : i + 1]
        swing_high = window_df['high'].max()
        swing_low = window_df['low'].min()
        current_close = df_fib['close'].iloc[i]

        price_range = swing_high - swing_low

        if price_range > 0:
            # For Uptrend Retracement (price drops from high to low)
            # Retracement levels are calculated from (Swing High - (Swing High - Swing Low) * Fib Level)
            fib_0_236 = swing_high - (price_range * 0.236)
            fib_0_382 = swing_high - (price_range * 0.382)
            fib_0_500 = swing_high - (price_range * 0.500)
            fib_0_618 = swing_high - (price_range * 0.618)

            # Features: Normalized distance from current price to key Fib levels
            if price_range != 0:
                df_fib.loc[df_fib.index[i], 'fib_236_retrace_dist_norm'] = (current_close - fib_0_236) / price_range
                df_fib.loc[df_fib.index[i], 'fib_382_retrace_dist_norm'] = (current_close - fib_0_382) / price_range
                df_fib.loc[df_fib.index[i], 'fib_618_retrace_dist_norm'] = (current_close - fib_0_618) / price_range

            # Is price above 0.5 Fibonacci retracement level?
            if current_close > fib_0_500:
                df_fib.loc[df_fib.index[i], 'is_price_above_fib_50'] = 1
            else:
                df_fib.loc[df_fib.index[i], 'is_price_above_fib_50'] = 0

    logger.debug(f"✅ [Indicator Fibonacci] Fibonacci features calculated.")
    return df_fib


# NEW: Support and Resistance Features (Copied from ml.py)
def calculate_support_resistance_features(df: pd.DataFrame, lookback_window: int = FIB_SR_LOOKBACK_WINDOW) -> pd.DataFrame:
    """
    Calculates simplified support and resistance features based on the lowest low and highest high
    within a rolling lookback window.
    """
    df_sr = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df_sr.columns for col in required_cols) or df_sr[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator S/R] Missing or empty OHLC columns. Cannot calculate S/R features.")
        for col in ['price_distance_to_recent_low_norm', 'price_distance_to_recent_high_norm']:
            df_sr[col] = np.nan
        return df_sr
    if len(df_sr) < lookback_window:
        logger.warning(f"⚠️ [Indicator S/R] Insufficient data ({len(df_sr)} < {lookback_window}) for S/R calculation.")
        for col in ['price_distance_to_recent_low_norm', 'price_distance_to_recent_high_norm']:
            df_sr[col] = np.nan
        return df_sr

    # Convert to numeric
    for col in required_cols:
        df_sr[col] = pd.to_numeric(df_sr[col], errors='coerce')

    df_sr['price_distance_to_recent_low_norm'] = np.nan
    df_sr['price_distance_to_recent_high_norm'] = np.nan

    for i in range(lookback_window - 1, len(df_sr)):
        window_df = df_sr.iloc[i - lookback_window + 1 : i + 1]
        recent_high = window_df['high'].max()
        recent_low = window_df['low'].min()
        current_close = df_sr['close'].iloc[i]

        price_range = recent_high - recent_low

        if price_range > 0:
            df_sr.loc[df_sr.index[i], 'price_distance_to_recent_low_norm'] = (current_close - recent_low) / price_range
            df_sr.loc[df_sr.index[i], 'price_distance_to_recent_high_norm'] = (recent_high - current_close) / price_range
        else:
            df_sr.loc[df_sr.index[i], 'price_distance_to_recent_low_norm'] = 0.0 # Price is at the low
            df_sr.loc[df_sr.index[i], 'price_distance_to_recent_high_norm'] = 0.0 # Price is at the high (if range is 0)

    logger.debug(f"✅ [Indicator S/R] Support and Resistance features calculated.")
    return df_sr


# ---------------------- Database Connection Setup ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    """Initializes database connection and creates tables if they don't exist."""
    global conn, cur
    logger.info("[DB] Starting database initialization...")
    for attempt in range(retries):
        try:
            logger.info(f"[DB] Attempting to connect to database (Attempt {attempt + 1}/{retries})...")
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            cur = conn.cursor()
            logger.info("✅ [DB] Successfully connected to database.")

            # --- Create or update signals table (Modified schema) ---
            logger.info("[DB] Checking/creating 'signals' table...")
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
                    signal_details JSONB,
                    stop_loss DOUBLE PRECISION  -- Added stop loss column
                );""")
            conn.commit()
            logger.info("✅ [DB] 'signals' table exists or created.")

            # --- Create ml_models table (NEW) ---
            logger.info("[DB] Checking/creating 'ml_models' table...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ml_models (
                    id SERIAL PRIMARY KEY,
                    model_name TEXT NOT NULL UNIQUE,
                    model_data BYTEA NOT NULL,
                    trained_at TIMESTAMP DEFAULT NOW(),
                    metrics JSONB
                );""")
            conn.commit()
            logger.info("✅ [DB] 'ml_models' table exists or created.")

            # --- Create market_dominance table (if it doesn't exist) ---
            logger.info("[DB] Checking/creating 'market_dominance' table...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS market_dominance (
                    id SERIAL PRIMARY KEY,
                    recorded_at TIMESTAMP DEFAULT NOW(),
                    btc_dominance DOUBLE PRECISION,
                    eth_dominance DOUBLE PRECISION
                );
            """)
            conn.commit()
            logger.info("✅ [DB] 'market_dominance' table exists or created.")

            logger.info("✅ [DB] Database initialized successfully.")
            return

        except OperationalError as op_err:
            logger.error(f"❌ [DB] Operational error connecting (Attempt {attempt + 1}): {op_err}")
            if conn: conn.rollback()
            if attempt == retries - 1:
                 logger.critical("❌ [DB] All database connection attempts failed.")
                 raise op_err
            time.sleep(delay)
        except Exception as e:
            logger.critical(f"❌ [DB] Unexpected failure in database initialization (Attempt {attempt + 1}): {e}", exc_info=True)
            if conn: conn.rollback()
            if attempt == retries - 1:
                 logger.critical("❌ [DB] All database connection attempts failed.")
                 raise e
            time.sleep(delay)

    logger.critical("❌ [DB] Failed to connect to the database after multiple attempts.")
    exit(1)


def check_db_connection() -> bool:
    """Checks database connection status and re-initializes if necessary."""
    global conn, cur
    try:
        if conn is None or conn.closed != 0:
            logger.warning("⚠️ [DB] Connection closed or non-existent. Re-initializing...")
            init_db()
            return True
        else:
             with conn.cursor() as check_cur:
                  check_cur.execute("SELECT 1;")
                  check_cur.fetchone()
             return True
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [DB] Database connection lost ({e}). Re-initializing...")
        try:
             init_db()
             return True
        except Exception as recon_err:
            logger.error(f"❌ [DB] Re-connection attempt failed after loss: {recon_err}")
            return False
    except Exception as e:
        logger.error(f"❌ [DB] Unexpected error while checking connection: {e}", exc_info=True)
        try:
            init_db()
            return True
        except Exception as recon_err:
             logger.error(f"❌ [DB] Re-connection attempt failed after unexpected error: {recon_err}")
             return False

def load_ml_model_from_db(symbol: str) -> Optional[Any]:
    """Loads the latest trained ML model for a specific symbol from the database."""
    global ml_models
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"

    if model_name in ml_models:
        logger.debug(f"ℹ️ [ML Model] Model '{model_name}' already in memory.")
        return ml_models[model_name]

    if not check_db_connection() or not conn:
        logger.error(f"❌ [ML Model] Cannot load ML model for {symbol} due to database connection issue.")
        return None

    try:
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT model_data FROM ml_models WHERE model_name = %s ORDER BY trained_at DESC LIMIT 1;", (model_name,))
            result = db_cur.fetchone()
            if result and result['model_data']:
                model = pickle.loads(result['model_data'])
                ml_models[model_name] = model # Store in global dictionary
                logger.info(f"✅ [ML Model] Successfully loaded ML model '{model_name}' from database.")
                return model
            else:
                logger.warning(f"⚠️ [ML Model] No ML model found with name '{model_name}' in database. Please train the model first.")
                return None
    except psycopg2.Error as db_err:
        logger.error(f"❌ [ML Model] Database error while loading ML model for {symbol}: {db_err}", exc_info=True)
        return None
    except pickle.UnpicklingError as unpickle_err:
        logger.error(f"❌ [ML Model] Error unpickling ML model for {symbol}: {unpickle_err}. Model might be corrupted or saved with a different version.", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"❌ [ML Model] Unexpected error while loading ML model for {symbol}: {e}", exc_info=True)
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
                         logger.warning(f"⚠️ [WS] Invalid price value for symbol {symbol}: '{price_str}'")
        elif isinstance(msg, dict):
             if msg.get('e') == 'error':
                 logger.error(f"❌ [WS] Error message from WebSocket: {msg.get('m', 'No error details')}")
             elif msg.get('stream') and msg.get('data'):
                 for ticker_item in msg.get('data', []):
                    symbol = ticker_item.get('s')
                    price_str = ticker_item.get('c')
                    if symbol and 'USDT' in symbol and price_str:
                        try:
                            ticker_data[symbol] = float(price_str)
                        except ValueError:
                             logger.warning(f"⚠️ [WS] Invalid price value for symbol {symbol} in combined stream: '{price_str}'")
        else:
             logger.warning(f"⚠️ [WS] Received WebSocket message in unexpected format: {type(msg)}")

    except Exception as e:
        logger.error(f"❌ [WS] Error processing ticker message: {e}", exc_info=True)


def run_ticker_socket_manager() -> None:
    """Runs and manages the WebSocket connection for mini-ticker."""
    while True:
        try:
            logger.info("ℹ️ [WS] Starting WebSocket manager for ticker prices...")
            twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
            twm.start()

            stream_name = twm.start_miniticker_socket(callback=handle_ticker_message)
            logger.info(f"✅ [WS] WebSocket stream started: {stream_name}")

            twm.join()
            logger.warning("⚠️ [WS] WebSocket manager stopped. Restarting...")

        except Exception as e:
            logger.error(f"❌ [WS] Fatal error in WebSocket manager: {e}. Restarting in 15 seconds...", exc_info=True)

        time.sleep(15)

# ---------------------- Other Helper Functions (Volume) ----------------------
def fetch_recent_volume(symbol: str, interval: str = SIGNAL_GENERATION_TIMEFRAME, num_candles: int = VOLUME_LOOKBACK_CANDLES) -> float:
    """Fetches the trading volume in USDT for the last `num_candles` of the specified `interval`."""
    if not client:
         logger.error(f"❌ [Data Volume] Binance client not initialized for fetching volume for {symbol}.")
         return 0.0
    try:
        logger.debug(f"ℹ️ [Data Volume] Fetching volume for last {num_candles} {interval} candles for {symbol}...")

        # Map interval string to Binance client constant
        binance_interval = None
        if interval == '15m':
            binance_interval = Client.KLINE_INTERVAL_15MINUTE
        elif interval == '5m':
            binance_interval = Client.KLINE_INTERVAL_5MINUTE
        elif interval == '1h':
            binance_interval = Client.KLINE_INTERVAL_1HOUR
        elif interval == '4h':
            binance_interval = Client.KLINE_INTERVAL_4HOUR
        elif interval == '1d':
            binance_interval = Client.KLINE_INTERVAL_1DAY
        else:
            logger.error(f"❌ [Data Volume] Unsupported interval: {interval}")
            return 0.0

        klines = client.get_klines(symbol=symbol, interval=binance_interval, limit=num_candles)
        if not klines or len(klines) < num_candles:
             logger.warning(f"⚠️ [Data Volume] Insufficient {interval} data (less than {num_candles} candles) for {symbol}.")
             return 0.0

        # k[7] is the quote asset volume (e.g., USDT volume)
        volume_usdt = sum(float(k[7]) for k in klines if len(k) > 7 and k[7])
        logger.debug(f"✅ [Data Volume] Liquidity for last {num_candles} {interval} candles for {symbol}: {volume_usdt:.2f} USDT")
        return volume_usdt
    except (BinanceAPIException, BinanceRequestException) as binance_err:
         logger.error(f"❌ [Data Volume] Binance API or network error while fetching volume for {symbol}: {binance_err}")
         return 0.0
    except Exception as e:
        logger.error(f"❌ [Data Volume] Unexpected error while fetching volume for {symbol}: {e}", exc_info=True)
        return 0.0

# ---------------------- Reading and Validating Symbols List ----------------------
def get_crypto_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """
    Reads the list of currency symbols from a text file, appends 'USDT' to each,
    then validates them as valid USDT pairs available for Spot trading on Binance.
    """
    raw_symbols: List[str] = []
    logger.info(f"ℹ️ [Data] Reading symbol list from '{filename}' file...")
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
            # Append USDT to each symbol if not already present
            raw_symbols = [f"{line.strip().upper()}USDT" if not line.strip().upper().endswith('USDT') else line.strip().upper()
                           for line in f if line.strip() and not line.startswith('#')]
        raw_symbols = sorted(list(set(raw_symbols)))
        logger.info(f"ℹ️ [Data] Read {len(raw_symbols)} initial symbols from '{file_path}'.")

    except FileNotFoundError:
         logger.error(f"❌ [Data] File '{filename}' not found.")
         return []
    except Exception as e:
        logger.error(f"❌ [Data] Error reading file '{filename}': {e}", exc_info=True)
        return []

    if not raw_symbols:
         logger.warning("⚠️ [Data] Initial symbol list is empty.")
         return []

    if not client:
        logger.error("❌ [Data Validation] Binance client not initialized. Cannot validate symbols.")
        return raw_symbols

    try:
        logger.info("ℹ️ [Data Validation] Validating symbols and trading status from Binance API...")
        exchange_info = client.get_exchange_info()
        valid_trading_usdt_symbols = {
            s['symbol'] for s in exchange_info['symbols']
            if s.get('quoteAsset') == 'USDT' and
               s.get('status') == 'TRADING' and
               s.get('isSpotTradingAllowed') is True
        }
        logger.info(f"ℹ️ [Data Validation] Found {len(valid_trading_usdt_symbols)} valid USDT Spot trading pairs on Binance.")
        validated_symbols = [symbol for symbol in raw_symbols if symbol in valid_trading_usdt_symbols]

        removed_count = len(raw_symbols) - len(validated_symbols)
        if removed_count > 0:
            removed_symbols = set(raw_symbols) - set(validated_symbols)
            logger.warning(f"⚠️ [Data Validation] Removed {removed_count} invalid or unavailable USDT trading symbols from list: {', '.join(removed_symbols)}")

        logger.info(f"✅ [Data Validation] Symbols validated. Using {len(validated_symbols)} valid symbols.")
        return validated_symbols

    except (BinanceAPIException, BinanceRequestException) as binance_err:
         logger.error(f"❌ [Data Validation] Binance API or network error while validating symbols: {binance_err}")
         logger.warning("⚠️ [Data Validation] Using initial list from file without Binance validation.")
         return raw_symbols
    except Exception as api_err:
         logger.error(f"❌ [Data Validation] Unexpected error while validating Binance symbols: {api_err}", exc_info=True)
         logger.warning("⚠️ [Data Validation] Using initial list from file without Binance validation.")
         return raw_symbols

# ---------------------- Comprehensive Performance Report Generation Function ----------------------
def generate_performance_report() -> str:
    """Generates a comprehensive performance report from the database in Arabic, including recent closed trades and USD profit/loss."""
    logger.info("ℹ️ [Report] Generating performance report...")
    if not check_db_connection() or not conn or not cur:
        logger.error("❌ [Report] Cannot generate report, database connection issue.")
        return "❌ Cannot generate report, database connection issue."
    try:
        with conn.cursor() as report_cur:
            # Modify query to include current_target and add current price from ticker_data
            report_cur.execute("SELECT id, symbol, entry_price, current_target, entry_time FROM signals WHERE achieved_target = FALSE ORDER BY entry_time DESC;")
            open_signals = report_cur.fetchall()
            open_signals_count = len(open_signals)

            report_cur.execute("""
                SELECT
                    COUNT(*) AS total_closed,
                    COUNT(*) FILTER (WHERE profit_percentage > 0) AS winning_signals,
                    COUNT(*) FILTER (WHERE profit_percentage <= 0) AS losing_signals,
                    COALESCE(SUM(profit_percentage) FILTER (WHERE profit_percentage > 0), 0) AS gross_profit_pct_sum,
                    COALESCE(SUM(profit_percentage) FILTER (WHERE profit_percentage <= 0), 0) AS gross_loss_pct_sum,
                    COALESCE(AVG(profit_percentage) FILTER (WHERE profit_percentage > 0), 0) AS avg_win_pct,
                    COALESCE(AVG(profit_percentage) FILTER (WHERE profit_percentage <= 0), 0) AS avg_loss_pct
                FROM signals
                WHERE achieved_target = TRUE;
            """)
            closed_stats = report_cur.fetchone() or {}

            total_closed = closed_stats.get('total_closed', 0)
            winning_signals = closed_stats.get('winning_signals', 0)
            losing_signals = closed_stats.get('losing_signals', 0)
            gross_profit_pct_sum = closed_stats.get('gross_profit_pct_sum', 0.0)
            gross_loss_pct_sum = closed_stats.get('gross_loss_pct_sum', 0.0)
            avg_win_pct = closed_stats.get('avg_win_pct', 0.0)
            avg_loss_pct = closed_stats.get('avg_loss_pct', 0.0)

            # Calculate USD profit/loss based on a fixed TRADE_VALUE
            gross_profit_usd = (gross_profit_pct_sum / 100.0) * TRADE_VALUE
            gross_loss_usd = (gross_loss_pct_sum / 100.0) * TRADE_VALUE

            # Total fees for all closed trades (entry and exit)
            total_fees_usd = total_closed * (TRADE_VALUE * BINANCE_FEE_RATE + (TRADE_VALUE * (1 + (avg_win_pct / 100.0 if avg_win_pct > 0 else 0))) * BINANCE_FEE_RATE)

            net_profit_usd = gross_profit_usd + gross_loss_usd - total_fees_usd # gross_loss_usd is already negative
            net_profit_pct = (net_profit_usd / (total_closed * TRADE_VALUE)) * 100 if total_closed * TRADE_VALUE > 0 else 0.0


            win_rate = (winning_signals / total_closed) * 100 if total_closed > 0 else 0.0
            profit_factor = float('inf') if gross_loss_pct_sum == 0 else (gross_profit_pct_sum / abs(gross_loss_pct_sum))

        report = f"""📊 *Comprehensive Performance Report:*
_(Assumed Trade Value: ${TRADE_VALUE:,.2f} and Binance Fees: {BINANCE_FEE_RATE*100:.2f}% per trade)_ 
——————————————
📈 Currently Open Signals: *{open_signals_count}*
"""

        if open_signals:
            report += "  • Details:\n"
            for i, signal in enumerate(open_signals):
                safe_symbol = str(signal['symbol']).replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')
                entry_time_str = signal['entry_time'].strftime('%Y-%m-%d %H:%M') if signal['entry_time'] else 'N/A'
                
                # Get current price from ticker data
                current_price = ticker_data.get(signal['symbol'], 0.0)
                
                # Calculate progress towards target
                progress_pct = 0.0
                if current_price > 0 and signal['entry_price'] > 0 and signal['current_target'] > signal['entry_price']:
                    progress_pct = ((current_price - signal['entry_price']) / (signal['current_target'] - signal['entry_price'])) * 100
                
                # Determine progress icon based on percentage
                progress_icon = "🔴"  # Less than 25%
                if progress_pct >= 75:
                    progress_icon = "🟢"  # More than 75%
                elif progress_pct >= 50:
                    progress_icon = "🟡"  # Between 50% and 75%
                elif progress_pct >= 25:
                    progress_icon = "🟠"  # Between 25% and 50%
                
                # Add entry price, target, and current price to report in an organized format
                report += f"""    *{i+1}. {safe_symbol}*
       💲 *Entry:* `${signal['entry_price']:.8g}`
       🎯 *Target:* `${signal['current_target']:.8g}`
       💵 *Current Price:* `${current_price:.8g}`
       {progress_icon} *Progress:* `{progress_pct:.1f}%`
       ⏰ *Opened:* `{entry_time_str}`
"""
                # Add separator between signals unless it's the last signal
                if i < len(open_signals) - 1:
                    report += "       ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n"
        else:
            report += "  • No open signals currently.\n"

        report += f"""——————————————
📉 *Closed Signal Statistics:*
  • Total Closed Signals: *{total_closed}*
  ✅ Winning Signals: *{winning_signals}* ({win_rate:.2f}%)
  ❌ Losing Signals: *{losing_signals}*
——————————————
💰 *Overall Profitability:*
  • Gross Profit: *{gross_profit_pct_sum:+.2f}%* (≈ *${gross_profit_usd:+.2f}*)
  • Gross Loss: *{gross_loss_pct_sum:+.2f}%* (≈ *${gross_loss_usd:+.2f}*)
  • Estimated Total Fees: *${total_fees_usd:,.2f}*
  • *Net Profit:* *{net_profit_pct:+.2f}%* (≈ *${net_profit_usd:+.2f}*)
  • Avg. Winning Trade: *{avg_win_pct:+.2f}%*
  • Avg. Losing Trade: *{avg_loss_pct:+.2f}%*
  • Profit Factor: *{'∞' if profit_factor == float('inf') else f'{profit_factor:.2f}'}*
——————————————
🕰️ _Report updated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"""

        logger.info("✅ [Report] Performance report generated successfully.")
        return report

    except psycopg2.Error as db_err:
        logger.error(f"❌ [Report] Database error while generating performance report: {db_err}")
        if conn: conn.rollback()
        return "❌ Database error while generating performance report."
    except Exception as e:
        logger.error(f"❌ [Report] Unexpected error while generating performance report: {e}", exc_info=True)
        return "❌ An unexpected error occurred while generating performance report."

# ---------------------- Trading Strategy (Adjusted for ML-Only) -------------------

class ScalpingTradingStrategy:
    """Encapsulates the trading strategy logic, now relying solely on ML model prediction for buy signals."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ml_model = load_ml_model_from_db(symbol) # Load model specific to this symbol
        if self.ml_model is None:
            logger.warning(f"⚠️ [Strategy {self.symbol}] ML model for {symbol} not loaded. Strategy will not be able to generate signals.")

        # Updated feature columns to include all new indicators (MUST match ml.py)
        self.feature_columns_for_ml = [ 
            'volume_15m_avg',
            'rsi_momentum_bullish',
            'btc_trend_feature',
            'supertrend_direction',
            # Ichimoku features
            'ichimoku_tenkan_kijun_cross_signal',
            'ichimoku_price_cloud_position',
            'ichimoku_cloud_outlook',
            # Fibonacci features
            'fib_236_retrace_dist_norm',
            'fib_382_retrace_dist_norm',
            'fib_618_retrace_dist_norm',
            'is_price_above_fib_50',
            # Support/Resistance features
            'price_distance_to_recent_low_norm',
            'price_distance_to_recent_high_norm'
        ]

    def populate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculates all required indicators for the ML model's features."""
        logger.debug(f"ℹ️ [Strategy {self.symbol}] Calculating indicators for ML model...")
        
        # min_len_required should reflect the max lookback of all indicators
        min_len_required = max(
            VOLUME_LOOKBACK_CANDLES,
            RSI_PERIOD,
            RSI_MOMENTUM_LOOKBACK_CANDLES,
            ENTRY_ATR_PERIOD,
            SUPERTRAND_PERIOD,
            TENKAN_PERIOD,
            KIJUN_PERIOD,
            SENKOU_SPAN_B_PERIOD,
            CHIKOU_LAG, 
            FIB_SR_LOOKBACK_WINDOW,
            55 # For BTC EMA calculation (50 + buffer)
        ) + 5 # Additional buffer for safe calculations

        if len(df) < min_len_required:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame too short ({len(df)} < {min_len_required}) for ML indicator calculation.")
            return None

        try:
            df_calc = df.copy()
            # Calculate RSI as it's a prerequisite for rsi_momentum_bullish
            df_calc = calculate_rsi_indicator(df_calc, RSI_PERIOD)
            # Calculate ATR for target price calculation and Supertrend
            df_calc = calculate_atr_indicator(df_calc, ENTRY_ATR_PERIOD)
            # Calculate Supertrend
            df_calc = calculate_supertrend(df_calc, SUPERTRAND_PERIOD, SUPERTRAND_MULTIPLIER)

            # Add new features: 15-minute average liquidity volume (1 15m candle)
            df_calc['volume_15m_avg'] = df_calc['volume'].rolling(window=VOLUME_LOOKBACK_CANDLES, min_periods=1).mean()

            # Add bullish RSI Momentum indicator
            df_calc['rsi_momentum_bullish'] = 0
            if len(df_calc) >= RSI_MOMENTUM_LOOKBACK_CANDLES + 1:
                for i in range(RSI_MOMENTUM_LOOKBACK_CANDLES, len(df_calc)):
                    rsi_slice = df_calc['rsi'].iloc[i - RSI_MOMENTUM_LOOKBACK_CANDLES : i + 1]
                    if not rsi_slice.isnull().any() and np.all(np.diff(rsi_slice) > 0) and rsi_slice.iloc[-1] > 50:
                        df_calc.loc[df_calc.index[i], 'rsi_momentum_bullish'] = 1

            # Fetch and calculate BTC trend feature
            btc_df = fetch_historical_data("BTCUSDT", interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
            btc_trend_series = None
            if btc_df is not None and not btc_df.empty:
                btc_trend_series = _calculate_btc_trend_feature(btc_df)
                if btc_trend_series is not None:
                    # Merge BTC trend with the current symbol's DataFrame based on timestamp index
                    df_calc = df_calc.merge(btc_trend_series.rename('btc_trend_feature'),
                                            left_index=True, right_index=True, how='left')
                    df_calc['btc_trend_feature'] = df_calc['btc_trend_feature'].fillna(0.0)
                    logger.debug(f"ℹ️ [Strategy {self.symbol}] Bitcoin trend feature merged.")
                else:
                    logger.warning(f"⚠️ [Strategy {self.symbol}] Bitcoin trend feature calculation failed. Defaulting 'btc_trend_feature' to 0.")
                    df_calc['btc_trend_feature'] = 0.0
            else:
                logger.warning(f"⚠️ [Strategy {self.symbol}] Failed to fetch Bitcoin historical data. Defaulting 'btc_trend_feature' to 0.")
                df_calc['btc_trend_feature'] = 0.0

            # NEW: Calculate Ichimoku Cloud components and features
            df_calc = calculate_ichimoku_cloud(df_calc, TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_SPAN_B_PERIOD, CHIKOU_LAG)
            logger.debug(f"ℹ️ [Strategy {self.symbol}] Ichimoku Cloud features calculated.")

            # NEW: Calculate Fibonacci Retracement features
            df_calc = calculate_fibonacci_features(df_calc, FIB_SR_LOOKBACK_WINDOW)
            logger.debug(f"ℹ️ [Strategy {self.symbol}] Fibonacci Retracement features calculated.")

            # NEW: Calculate Support and Resistance features
            df_calc = calculate_support_resistance_features(df_calc, FIB_SR_LOOKBACK_WINDOW)
            logger.debug(f"ℹ️ [Strategy {self.symbol}] Support and Resistance features calculated.")


            # Ensure all feature columns for ML exist and are numeric
            for col in self.feature_columns_for_ml:
                if col not in df_calc.columns:
                    logger.warning(f"⚠️ [Strategy {self.symbol}] Missing feature column for ML model: {col}")
                    df_calc[col] = np.nan # Add missing column as NaN
                else:
                    df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')

            initial_len = len(df_calc)
            # Use all required columns for dropna, including ML features and ATR for target
            all_required_cols = list(set(self.feature_columns_for_ml + [
                'open', 'high', 'low', 'close', 'volume', 'atr', 'supertrend' # 'supertrend' for debugging, not strictly for ML features
            ]))
            df_cleaned = df_calc.dropna(subset=all_required_cols).copy()
            dropped_count = initial_len - len(df_cleaned)

            if dropped_count > 0:
                 logger.debug(f"ℹ️ [Strategy {self.symbol}] Dropped {dropped_count} rows due to NaN values in indicators.")
            if df_cleaned.empty:
                logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame is empty after removing NaN values for indicators.")
                return None

            latest = df_cleaned.iloc[-1]
            logger.debug(f"✅ [Strategy {self.symbol}] ML indicators calculated. Latest: {latest.to_dict()}") # Log full latest row
            return df_cleaned

        except KeyError as ke:
             logger.error(f"❌ [Strategy {self.symbol}] Error: Required column not found during indicator calculation: {ke}", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"❌ [Strategy {self.symbol}] Unexpected error during indicator calculation: {e}", exc_info=True)
            return None


    def generate_buy_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Generates a buy signal based solely on the ML model's bullish prediction,
        followed by essential filters (volume, profit margin).
        """
        logger.debug(f"ℹ️ [Strategy {self.symbol}] Generating buy signal (ML-only based)...")

        # min_signal_data_len should reflect the max lookback of all indicators
        min_signal_data_len = max(
            VOLUME_LOOKBACK_CANDLES,
            RSI_PERIOD,
            RSI_MOMENTUM_LOOKBACK_CANDLES,
            ENTRY_ATR_PERIOD,
            SUPERTRAND_PERIOD,
            TENKAN_PERIOD,
            KIJUN_PERIOD,
            SENKOU_SPAN_B_PERIOD,
            CHIKOU_LAG, 
            FIB_SR_LOOKBACK_WINDOW,
            55 # For BTC EMA calculation (50 + buffer)
        ) + 1 # At least one candle for the latest calculations

        if df_processed is None or df_processed.empty or len(df_processed) < min_signal_data_len:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame is empty or too short (<{min_signal_data_len}), cannot generate signal.")
            return None

        # Ensure all required columns for signal generation, including ML features, are present
        required_cols_for_signal = list(set(self.feature_columns_for_ml + [
            'close', 'atr', 'supertrend' 
        ]))
        missing_cols = [col for col in required_cols_for_signal if col not in df_processed.columns]
        if missing_cols:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame missing required columns for signal: {missing_cols}.")
            return None

        last_row = df_processed.iloc[-1]

        # --- Get current real-time price from ticker_data ---
        current_price = ticker_data.get(self.symbol)
        if current_price is None:
            logger.warning(f"⚠️ [Strategy {self.symbol}] Current price not available from ticker data. Cannot generate signal.")
            return None

        if last_row[self.feature_columns_for_ml].isnull().values.any() or pd.isna(last_row.get('atr')) or pd.isna(last_row.get('supertrend')):
             logger.warning(f"⚠️ [Strategy {self.symbol}] Historical data contains NaN values in required indicator columns. Cannot generate signal.")
             return None

        signal_details = {} # Initialize signal_details

        # --- ML Model Prediction (Primary decision maker) ---
        ml_prediction_result_text = "N/A (Model not loaded)"
        ml_is_bullish = False

        if self.ml_model: # Use self.ml_model which is loaded per symbol
            try:
                # Ensure the order of features for prediction matches the training order
                features_for_prediction = pd.DataFrame([last_row[self.feature_columns_for_ml].values], columns=self.feature_columns_for_ml)
                ml_pred = self.ml_model.predict(features_for_prediction)[0]
                if ml_pred == 1: # If ML model predicts upward movement
                    ml_is_bullish = True
                    ml_prediction_result_text = 'Bullish ✅'
                    logger.info(f"✨ [Strategy {self.symbol}] ML model prediction is bullish.")
                else:
                    ml_prediction_result_text = 'Bearish ❌'
                    logger.info(f"ℹ️ [Strategy {self.symbol}] ML model prediction is bearish. Signal rejected.")
            except Exception as ml_err:
                logger.error(f"❌ [Strategy {self.symbol}] Error in ML model prediction: {ml_err}", exc_info=True)
                ml_prediction_result_text = "Prediction Error (0)"
        
        signal_details['ML_Prediction'] = ml_prediction_result_text
        # Add values of relevant features to signal_details for logging/reporting
        signal_details['BTC_Trend_Feature_Value'] = last_row.get('btc_trend_feature', 0.0)
        signal_details['Supertrend_Direction_Value'] = last_row.get('supertrend_direction', 0)
        signal_details['Ichimoku_Cross_Signal'] = last_row.get('ichimoku_tenkan_kijun_cross_signal', 0)
        signal_details['Ichimoku_Price_Cloud_Position'] = last_row.get('ichimoku_price_cloud_position', 0)
        signal_details['Ichimoku_Cloud_Outlook'] = last_row.get('ichimoku_cloud_outlook', 0)
        signal_details['Fib_Above_50'] = last_row.get('is_price_above_fib_50', 0)
        signal_details['Dist_to_Recent_Low_Norm'] = last_row.get('price_distance_to_recent_low_norm', np.nan)
        signal_details['Dist_to_Recent_High_Norm'] = last_row.get('price_distance_to_recent_high_norm', np.nan)


        # If ML model is not bullish, or was not loaded, reject the signal early.
        if not ml_is_bullish:
            logger.info(f"ℹ️ [Strategy {self.symbol}] ML model did not predict bullish. Signal rejected.")
            return None

        # --- NEW: Add additional filters (these are hard-coded rules on top of ML prediction) ---
        # Filter 1: Supertrend must be bullish
        current_supertrend_direction = last_row.get('supertrend_direction')
        if current_supertrend_direction != 1:
            logger.info(f"ℹ️ [Strategy {self.symbol}] Supertrend is not bullish ({current_supertrend_direction}). Signal rejected.")
            signal_details['Supertrend_Filter'] = f'Failed: Supertrend not bullish ({current_supertrend_direction})'
            return None
        else:
            signal_details['Supertrend_Filter'] = f'Success: Supertrend bullish ({current_supertrend_direction})'

        # Filter 2: Bitcoin trend should not be strongly bearish (allow neutral or bullish)
        current_btc_trend = last_row.get('btc_trend_feature')
        if current_btc_trend == -1.0:
            logger.info(f"ℹ️ [Strategy {self.symbol}] Bitcoin trend is bearish ({current_btc_trend}). Signal rejected.")
            signal_details['BTC_Trend_Filter'] = f'Failed: Bitcoin trend bearish ({current_btc_trend})'
            return None
        else:
            signal_details['BTC_Trend_Filter'] = f'Success: Bitcoin trend not bearish ({current_btc_trend})'

        # --- Volume Check (Essential filter) ---
        volume_recent = fetch_recent_volume(self.symbol, interval=SIGNAL_GENERATION_TIMEFRAME, num_candles=VOLUME_LOOKBACK_CANDLES)
        if volume_recent < MIN_VOLUME_15M_USDT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] Liquidity ({volume_recent:,.0f} USDT) is below required minimum ({MIN_VOLUME_15M_USDT:,.0f} USDT). Signal rejected.")
            signal_details['Volume_Check'] = f'Failed: Insufficient liquidity ({volume_recent:,.0f} USDT)'
            return None
        else:
            signal_details['Volume_Check'] = f'Success: Sufficient liquidity ({volume_recent:,.0f} USDT)'


        current_atr = last_row.get('atr')
        current_supertrend_value = last_row.get('supertrend') # Get the actual Supertrend line value

        if pd.isna(current_atr) or current_atr <= 0 or pd.isna(current_supertrend_value):
             logger.warning(f"⚠️ [Strategy {self.symbol}] Invalid ATR or Supertrend value ({current_atr}, {current_supertrend_value}) for target/stop calculation. Cannot generate signal.")
             return None

        # --- Calculate Initial Target ---
        target_multiplier = ENTRY_ATR_MULTIPLIER
        initial_target = current_price + (target_multiplier * current_atr)

        # --- Profit Margin Check (Essential filter) ---
        profit_margin_pct = ((initial_target / current_price) - 1) * 100 if current_price > 0 else 0
        if profit_margin_pct < MIN_PROFIT_MARGIN_PCT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] Profit margin ({profit_margin_pct:.2f}%) is below required minimum ({MIN_PROFIT_MARGIN_PCT:.2f}%). Signal rejected.")
            signal_details['Profit_Margin_Check'] = f'Failed: Insufficient profit margin ({profit_margin_pct:.2f}%)'
            return None
        else:
            signal_details['Profit_Margin_Check'] = f'Success: Sufficient profit margin ({profit_margin_pct:.2f}%)'

        # --- Calculate Initial Stop Loss ---
        # Ensure Supertrend value is below the current price for a long signal
        if current_supertrend_value < current_price:
            initial_stop_loss = current_supertrend_value
            signal_details['Stop_Loss_Method'] = f'Supertrend ({current_supertrend_value:.8g})'
        else:
            # Fallback to ATR if Supertrend is above entry (shouldn't happen if direction is 1, but as safety)
            stop_loss_atr_multiplier = 1.0
            initial_stop_loss = current_price - (stop_loss_atr_multiplier * current_atr)
            signal_details['Stop_Loss_Method'] = f'ATR Fallback ({initial_stop_loss:.8g})'
            logger.warning(f"⚠️ [Strategy {self.symbol}] Supertrend ({current_supertrend_value:.8g}) is above entry price ({current_price:.8g}) despite bullish direction. Using ATR for stop loss.")

        # Ensure stop loss is not negative
        initial_stop_loss = max(0.00000001, initial_stop_loss) # Avoid zero or negative stop loss


        signal_output = {
            'symbol': self.symbol,
            'entry_price': float(f"{current_price:.8g}"),
            'initial_target': float(f"{initial_target:.8g}"),
            'current_target': float(f"{initial_target:.8g}"),
            'stop_loss': float(f"{initial_stop_loss:.8g}"),
            'r2_score': 1.0, # Placeholder score as it's ML-driven now
            'strategy_name': 'Scalping_ML_Enhanced_Filtered', # Updated strategy name
            'signal_details': signal_details,
            'volume_15m': volume_recent,
            'trade_value': TRADE_VALUE,
            'total_possible_score': 1.0 # Placeholder
        }

        logger.info(f"✅ [Strategy {self.symbol}] Buy signal confirmed (ML + Filters). Price: {current_price:.6f}, Target: {initial_target:.6f}, Stop Loss: {initial_stop_loss:.6f}, ATR: {current_atr:.6f}, Volume: {volume_recent:,.0f}, ML Prediction: {ml_prediction_result_text}, BTC Trend: {signal_details.get('BTC_Trend_Feature_Value')}, Supertrend Dir: {signal_details.get('Supertrend_Direction_Value')}, Ichimoku Cross: {signal_details.get('Ichimoku_Cross_Signal')}, Price Cloud Pos: {signal_details.get('Ichimoku_Price_Cloud_Position')}, Cloud Outlook: {signal_details.get('Ichimoku_Cloud_Outlook')}, Fib Above 50: {signal_details.get('Fib_Above_50')}")
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
             logger.error(f"❌ [Telegram] Failed to convert reply_markup to JSON: {json_err} - Markup: {reply_markup}")
             return None

    logger.debug(f"ℹ️ [Telegram] Sending message to {target_chat_id}...")
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        logger.info(f"✅ [Telegram] Message sent successfully to {target_chat_id}.")
        return response.json()
    except requests.exceptions.Timeout:
         logger.error(f"❌ [Telegram] Failed to send message to {target_chat_id} (timeout).")
         return None
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"❌ [Telegram] Failed to send message to {target_chat_id} (HTTP error: {http_err.response.status_code}).")
        try:
            error_details = http_err.response.json()
            logger.error(f"❌ [Telegram] API error details: {error_details}")
        except json.JSONDecodeError:
            logger.error(f"❌ [Telegram] Could not decode error response: {http_err.response.text}")
        return None
    except requests.exceptions.RequestException as req_err:
        logger.error(f"❌ [Telegram] Failed to send message to {target_chat_id} (request error): {req_err}")
        return None
    except Exception as e:
         logger.error(f"❌ [Telegram] Unexpected error while sending message: {e}", exc_info=True)
         return None

def send_telegram_alert(signal_data: Dict[str, Any], timeframe: str) -> None:
    """Formats and sends a new trading signal alert to Telegram in Arabic, displaying the ML prediction and new indicator details."""
    logger.debug(f"ℹ️ [Telegram Alert] Formatting and sending signal alert: {signal_data.get('symbol', 'N/A')}")
    try:
        entry_price = float(signal_data['entry_price'])
        target_price = float(signal_data['initial_target'])
        stop_loss_price = float(signal_data['stop_loss'])
        symbol = signal_data['symbol']
        strategy_name = signal_data.get('strategy_name', 'N/A')
        volume_15m = signal_data.get('volume_15m', 0.0)
        trade_value_signal = signal_data.get('trade_value', TRADE_VALUE)
        signal_details = signal_data.get('signal_details', {})

        profit_pct = ((target_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        loss_pct = ((stop_loss_price / entry_price) - 1) * 100 if entry_price > 0 else 0

        entry_fee = trade_value_signal * BINANCE_FEE_RATE
        exit_value_target = trade_value_signal * (1 + profit_pct / 100.0)
        exit_value_stoploss = trade_value_signal * (1 + loss_pct / 100.0)
        
        exit_fee_target = exit_value_target * BINANCE_FEE_RATE
        exit_fee_stoploss = exit_value_stoploss * BINANCE_FEE_RATE
        
        total_trade_fees_target = entry_fee + exit_fee_target
        total_trade_fees_stoploss = entry_fee + exit_fee_stoploss

        profit_usdt_gross = trade_value_signal * (profit_pct / 100)
        profit_usdt_net = profit_usdt_gross - total_trade_fees_target
        
        loss_usdt_gross = trade_value_signal * (loss_pct / 100) 
        loss_usdt_net = loss_usdt_gross - total_trade_fees_stoploss

        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')

        fear_greed = get_fear_greed_index()
        ml_prediction_status = signal_details.get('ML_Prediction', 'N/A')
        btc_trend_feature_value = signal_details.get('BTC_Trend_Feature_Value', 0.0)
        btc_trend_display = "Bullish 📈" if btc_trend_feature_value == 1.0 else ("Bearish 📉" if btc_trend_feature_value == -1.0 else "Neutral 🔄")
        supertrend_direction_value = signal_details.get('Supertrend_Direction_Value', 0)
        supertrend_display = "Uptrend ⬆️" if supertrend_direction_value == 1 else ("Downtrend ⬇️" if supertrend_direction_value == -1 else "Neutral ↔️")

        # Ichimoku display
        ichimoku_cross_signal = signal_details.get('Ichimoku_Cross_Signal', 0)
        ichimoku_cross_display = "Bullish Cross (TK) ✅" if ichimoku_cross_signal == 1 else ("Bearish Cross (TK) ❌" if ichimoku_cross_signal == -1 else "No Cross ↔️")
        ichimoku_price_cloud_pos = signal_details.get('Ichimoku_Price_Cloud_Position', 0)
        ichimoku_price_cloud_display = "Above Cloud ☁️⬆️" if ichimoku_price_cloud_pos == 1 else ("Below Cloud ☁️⬇️" if ichimoku_price_cloud_pos == -1 else "Inside Cloud ☁️↔️")
        ichimoku_cloud_outlook = signal_details.get('Ichimoku_Cloud_Outlook', 0)
        ichimoku_cloud_outlook_display = "Bullish Cloud 🟩" if ichimoku_cloud_outlook == 1 else ("Bearish Cloud 🟥" if ichimoku_cloud_outlook == -1 else "Flat Cloud ⬜")

        # Fibonacci & S/R display (simplified, for context)
        fib_above_50 = signal_details.get('Fib_Above_50', 0)
        fib_above_50_display = "Above 50% Fib 🟢" if fib_above_50 == 1 else "Below 50% Fib 🔴"
        dist_to_recent_low = signal_details.get('Dist_to_Recent_Low_Norm', np.nan)
        dist_to_recent_high = signal_details.get('Dist_to_Recent_High_Norm', np.nan)
        
        sr_display_content = ""
        if not pd.isna(dist_to_recent_low) and not pd.isna(dist_to_recent_high):
            sr_display_content = f"  - Dist to Recent Low: {dist_to_recent_low:.2f} | Dist to Recent High: {dist_to_recent_high:.2f}"

        message = f"""💡 *New Trading Signal (ML-Only Based)* 💡
——————————————
🪙 **Pair:** `{safe_symbol}`
📈 **Signal Type:** Buy (Long)
🕰️ **Timeframe:** {timeframe}
💧 **Liquidity (last 15m):** {volume_15m:,.0f} USDT
——————————————
➡️ **Suggested Entry Price:** `${entry_price:,.8g}`
🎯 **Initial Target:** `${target_price:,.8g}`
🛑 **Stop Loss:** `${stop_loss_price:,.8g}`
💰 **Expected Profit (Gross):** ({profit_pct:+.2f}% / ≈ ${profit_usdt_gross:+.2f})
💸 **Expected Loss (Gross):** ({loss_pct:+.2f}% / ≈ ${loss_usdt_gross:+.2f})
📈 **Net Profit (Expected):** ${profit_usdt_net:+.2f}
📉 **Net Loss (Expected):** ${loss_usdt_net:+.2f}
——————————————
🤖 *ML Model Prediction:* *{ml_prediction_status}*
✅ *Additional Conditions Met:*
  - Liquidity Check: {signal_details.get('Volume_Check', 'N/A')}
  - Profit Margin Check: {signal_details.get('Profit_Margin_Check', 'N/A')}
  - Supertrend Filter: {signal_details.get('Supertrend_Filter', 'N/A')}
  - BTC Trend Filter: {signal_details.get('BTC_Trend_Filter', 'N/A')}
——————————————
📊 *Indicator Insights:*
  - Fear & Greed Index: {fear_greed}
  - Bitcoin Trend: {btc_trend_display}
  - Supertrend Direction: {supertrend_display}
  - Ichimoku Tenkan/Kijun: {ichimoku_cross_display}
  - Ichimoku Price vs Cloud: {ichimoku_price_cloud_display}
  - Ichimoku Cloud Outlook: {ichimoku_cloud_outlook_display}
  - Fibonacci Retracement (50%): {fib_above_50_display}
{sr_display_content + '\n' if sr_display_content else ''}——————————————
⏰ {timestamp_str}"""

        reply_markup = {
            "inline_keyboard": [
                [{"text": "📊 View Performance Report", "callback_data": "get_report"}]
            ]
        }

        send_telegram_message(CHAT_ID, message, reply_markup=reply_markup, parse_mode='Markdown')

    except KeyError as ke:
        logger.error(f"❌ [Telegram Alert] Incomplete signal data for symbol {signal_data.get('symbol', 'N/A')}: missing key {ke}", exc_info=True)
    except Exception as e:
        logger.error(f"❌ [Telegram Alert] Failed to send signal alert for symbol {signal_data.get('symbol', 'N/A')}: {e}", exc_info=True)

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
    old_stop_loss = details.get('old_stop_loss', 0.0)
    new_stop_loss = details.get('new_stop_loss', 0.0)


    logger.debug(f"ℹ️ [Notification] Formatting tracking notification: ID={signal_id}, Type={notification_type}, Symbol={symbol}")

    if notification_type == 'target_hit':
        message = f"""✅ *Target Reached (ID: {signal_id})*
——————————————
🪙 **Pair:** `{safe_symbol}`
🎯 **Closing Price (Target):** `${closing_price:,.8g}`
💰 **Realized Profit:** {profit_pct:+.2f}%
⏱️ **Time Taken:** {time_to_target}"""
    elif notification_type == 'stop_loss_hit':
        message = f"""🛑 *Stop Loss Hit (ID: {signal_id})*
——————————————
🪙 **Pair:** `{safe_symbol}`
📉 **Closing Price (Stop Loss):** `${closing_price:,.8g}`
💔 **Realized Loss:** {profit_pct:+.2f}%
⏱️ **Time Taken:** {time_to_target}"""
    elif notification_type == 'target_stoploss_updated':
         update_parts_formatted = [] # Renamed for clarity
         if 'old_target' in details and 'new_target' in details:
             update_parts_formatted.append(f"  🎯 *Target:* `${old_target:,.8g}` -> `${new_target:,.8g}`")
         if 'old_stop_loss' in details and 'new_stop_loss' in details:
             update_parts_formatted.append(f"  🛑 *Stop Loss:* `${old_stop_loss:,.8g}` -> `${new_stop_loss:,.8g}`")

         update_block = "\n".join(update_parts_formatted) # Pre-join to avoid backslash in f-string expression

         message = f"""🔄 *Signal Update (ID: {signal_id})*
——————————————
🪙 **Pair:** `{safe_symbol}`
📈 **Current Price:** `${current_price:,.8g}`
{update_block}
ℹ️ *Updated based on continued bullish momentum or market conditions.*"""
    else:
        logger.warning(f"⚠️ [Notification] Unknown notification type: {notification_type} for details: {details}")
        return

    if message:
        send_telegram_message(CHAT_ID, message, parse_mode='Markdown')

# ---------------------- Database Functions (Insert and Update) ----------------------
def insert_signal_into_db(signal: Dict[str, Any]) -> bool:
    """Inserts a new signal into the signals table with the weighted score and entry time."""
    if not check_db_connection() or not conn:
        logger.error(f"❌ [DB Insert] Failed to insert signal {signal.get('symbol', 'N/A')} due to database connection issue.")
        return False

    symbol = signal.get('symbol', 'N/A')
    logger.debug(f"ℹ️ [DB Insert] Attempting to insert signal for {symbol}...")
    try:
        signal_prepared = convert_np_values(signal)
        signal_details_json = json.dumps(signal_prepared.get('signal_details', {}))

        with conn.cursor() as cur_ins:
            insert_query = sql.SQL("""
                INSERT INTO signals
                 (symbol, entry_price, initial_target, current_target, stop_loss,
                 r2_score, strategy_name, signal_details, volume_15m, entry_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW());
            """)
            cur_ins.execute(insert_query, (
                signal_prepared['symbol'],
                signal_prepared['entry_price'],
                signal_prepared['initial_target'],
                signal_prepared['current_target'],
                signal_prepared['stop_loss'],
                signal_prepared.get('r2_score'),
                signal_prepared.get('strategy_name', 'unknown'),
                signal_details_json,
                signal_prepared.get('volume_15m')
            ))
        conn.commit()
        logger.info(f"✅ [DB Insert] Signal for {symbol} inserted into database (Score: {signal_prepared.get('r2_score')}).")
        return True
    except psycopg2.Error as db_err:
        logger.error(f"❌ [DB Insert] Database error while inserting signal for {symbol}: {db_err}")
        if conn: conn.rollback()
        return False
    except (TypeError, ValueError) as convert_err:
         logger.error(f"❌ [DB Insert] Error converting signal data before insertion for {symbol}: {convert_err} - Signal data: {signal}")
         if conn: conn.rollback()
         return False
    except Exception as e:
        logger.error(f"❌ [DB Insert] Unexpected error while inserting signal for {symbol}: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

# ---------------------- Open Signal Tracking Function ----------------------
def track_signals() -> None:
    """Tracks open signals and checks targets. Calculates time to target upon hit."""
    logger.info("ℹ️ [Tracker] Starting open signal tracking process...")
    while True:
        active_signals_summary: List[str] = []
        processed_in_cycle = 0
        try:
            if not check_db_connection() or not conn:
                logger.warning("⚠️ [Tracker] Skipping tracking cycle due to database connection issue.")
                time.sleep(15)
                continue

            with conn.cursor() as track_cur:
                 track_cur.execute("""
                    SELECT id, symbol, entry_price, initial_target, current_target, entry_time, stop_loss
                    FROM signals
                    WHERE achieved_target = FALSE AND closing_price is NULL;
                """)
                 open_signals: List[Dict] = track_cur.fetchall()

            if not open_signals:
                time.sleep(10)
                continue

            logger.debug(f"ℹ️ [Tracker] Tracking {len(open_signals)} open signals...")

            for signal_row in open_signals:
                signal_id = signal_row['id']
                symbol = signal_row['symbol']
                processed_in_cycle += 1
                update_executed = False

                try:
                    entry_price = float(signal_row['entry_price'])
                    entry_time = signal_row['entry_time']
                    current_target = float(signal_row["current_target"])
                    current_stop_loss = float(signal_row["stop_loss"]) if signal_row.get("stop_loss") is not None else None # Fetch stop loss

                    current_price = ticker_data.get(symbol)

                    if current_price is None:
                         logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): Current price not available in ticker data.")
                         continue

                    active_signals_summary.append(f"{symbol}({signal_id}): P={current_price:.4f} T={current_target:.4f} SL={current_stop_loss if current_stop_loss else 'N/A'}") # Include SL in summary

                    update_query: Optional[sql.SQL] = None
                    update_params: Tuple = ()
                    log_message: Optional[str] = None
                    notification_details: Dict[str, Any] = {
                        'symbol': symbol,
                        'id': signal_id,
                        'current_price': current_price,
                        'entry_price': entry_price,
                        'current_target': current_target,
                        'stop_loss': current_stop_loss
                    }


                    # --- Check and Update Logic ---
                    # 0. Check for Stop Loss Hit (PRIORITY)
                    if current_stop_loss is not None and current_price <= current_stop_loss:
                        profit_pct = ((current_stop_loss / entry_price) - 1) * 100 if entry_price > 0 else 0
                        closed_at = datetime.now()
                        time_to_close = closed_at - entry_time if entry_time else timedelta(0)
                        time_to_close_str = str(time_to_close)

                        update_query = sql.SQL("UPDATE signals SET achieved_target = FALSE, closing_price = %s, closed_at = %s, profit_percentage = %s, time_to_target = %s WHERE id = %s;") # achieved_target is FALSE for stop loss
                        update_params = (current_stop_loss, closed_at, profit_pct, time_to_close, signal_id)
                        log_message = f"🛑 [Tracker] {symbol}(ID:{signal_id}): Stop Loss hit at {current_stop_loss:.8g} (Loss: {profit_pct:+.2f}%, Time: {time_to_close_str})."
                        notification_details.update({
                            'type': 'stop_loss_hit',
                            'closing_price': current_stop_loss,
                            'profit_pct': profit_pct,
                            'time_to_target': time_to_close_str # Reusing field name for duration
                        })
                        update_executed = True

                    # 1. Check for Target Hit (Only if Stop Loss not hit)
                    if not update_executed and current_price >= current_target:
                        profit_pct = ((current_target / entry_price) - 1) * 100 if entry_price > 0 else 0
                        closed_at = datetime.now()
                        time_to_target_duration = closed_at - entry_time if entry_time else timedelta(0)
                        time_to_target_str = str(time_to_target_duration)

                        update_query = sql.SQL("UPDATE signals SET achieved_target = TRUE, closing_price = %s, closed_at = %s, profit_percentage = %s, time_to_target = %s WHERE id = %s;")
                        update_params = (current_target, closed_at, profit_pct, time_to_target_duration, signal_id)
                        log_message = f"🎯 [Tracker] {symbol}(ID:{signal_id}): Target reached at {current_target:.8g} (Profit: {profit_pct:+.2f}%, Time: {time_to_target_str})."
                        notification_details.update({
                            'type': 'target_hit',
                            'closing_price': current_target,
                            'profit_pct': profit_pct,
                            'time_to_target': time_to_target_str
                        })
                        update_executed = True

                    # 2. Check for Target/Stop Loss Update (Dynamic Target & Trailing Stop) (Only if not closed)
                    if not update_executed:
                        # Condition to check for update: e.g., price made significant progress towards target
                        progress_to_target = (current_price - entry_price) / (current_target - entry_price) if (current_target - entry_price) != 0 else 0
                        should_check_update = current_price >= current_target * (1 - TARGET_APPROACH_THRESHOLD_PCT)

                        if should_check_update:
                             logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): Price is near target ({current_price:.8g} vs {current_target:.8g}). Checking for continuation signal to update target/stop loss...")

                             df_continuation = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)

                             if df_continuation is not None and not df_continuation.empty:
                                 continuation_strategy = ScalpingTradingStrategy(symbol)
                                 if continuation_strategy.ml_model is None:
                                     logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): ML model not loaded for continuation strategy. Skipping target/stop loss update.")
                                     continue

                                 df_continuation_indicators = continuation_strategy.populate_indicators(df_continuation)

                                 if df_continuation_indicators is not None and not df_continuation_indicators.empty:
                                     # Use generate_buy_signal to check if conditions *still* hold for a buy
                                     # We don't need the full signal output, just whether it passes filters
                                     continuation_signal_check = continuation_strategy.generate_buy_signal(df_continuation_indicators)

                                     if continuation_signal_check: # If conditions still look bullish
                                         latest_row = df_continuation_indicators.iloc[-1]
                                         current_atr_for_update = latest_row.get('atr')
                                         current_supertrend_for_update = latest_row.get('supertrend') # Supertrend value for trailing stop

                                         if pd.notna(current_atr_for_update) and current_atr_for_update > 0 and pd.notna(current_supertrend_for_update):
                                             # --- Calculate Potential New Target ---
                                             potential_new_target = current_price + (ENTRY_ATR_MULTIPLIER * current_atr_for_update)

                                             # --- Calculate Potential New Stop Loss (Trailing Stop) ---
                                             # Option 1: Trail using Supertrend
                                             potential_new_stop_loss = current_supertrend_for_update
                                             
                                             # Ensure new stop loss is higher than the current one AND below current price
                                             new_stop_loss_valid = potential_new_stop_loss > (current_stop_loss or 0) and potential_new_stop_loss < current_price

                                             # --- Decide whether to update Target and/or Stop Loss ---
                                             update_target = potential_new_target > current_target
                                             update_stop_loss = new_stop_loss_valid

                                             if update_target or update_stop_loss:
                                                 old_target = current_target
                                                 old_stop_loss = current_stop_loss
                                                 new_target = potential_new_target if update_target else current_target
                                                 new_stop_loss = potential_new_stop_loss if update_stop_loss else current_stop_loss

                                                 update_fields = []
                                                 update_params_list = []
                                                 log_parts = []
                                                 notification_details.update({'type': 'target_stoploss_updated'}) # Assume both might update

                                                 if update_target:
                                                     update_fields.append("current_target = %s")
                                                     update_params_list.append(new_target)
                                                     log_parts.append(f"Target from {old_target:.8g} to {new_target:.8g}")
                                                     notification_details['old_target'] = old_target
                                                     notification_details['new_target'] = new_target

                                                 if update_stop_loss:
                                                     update_fields.append("stop_loss = %s")
                                                     update_params_list.append(new_stop_loss)
                                                     log_parts.append(f"Stop Loss from {old_stop_loss if old_stop_loss else 'N/A'} to {new_stop_loss:.8g}")
                                                     notification_details['old_stop_loss'] = old_stop_loss
                                                     notification_details['new_stop_loss'] = new_stop_loss

                                                 update_params_list.append(signal_id)
                                                 update_query = sql.SQL(f"UPDATE signals SET {', '.join(update_fields)} WHERE id = %s;")
                                                 update_params = tuple(update_params_list)
                                                 log_message = f"↔️ [Tracker] {symbol}(ID:{signal_id}): Updated {' and '.join(log_parts)} based on signal continuation."
                                                 update_executed = True
                                             else:
                                                 logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): Continuation signal detected, but new target ({potential_new_target:.8g}) or new stop loss ({potential_new_stop_loss:.8g}) does not warrant an update.")
                                         else:
                                             logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): Cannot calculate new target/stop loss due to invalid ATR/Supertrend ({current_atr_for_update}, {current_supertrend_for_update}) from continuation data.")
                                     else:
                                         logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): Price near target, but continuation signal not confirmed (filters or ML prediction failed). Not updating target/stop loss.")
                                 else:
                                     logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): Failed to populate indicators for continuation check.")
                             else:
                                 logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): Could not fetch historical data for continuation check.")


                    if update_executed and update_query:
                        try:
                             with conn.cursor() as update_cur:
                                  update_cur.execute(update_query, update_params)
                             conn.commit()
                             if log_message: logger.info(log_message)
                             if notification_details.get('type'):
                                send_tracking_notification(notification_details)
                        except psycopg2.Error as db_err:
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): Database error during update: {db_err}")
                            if conn: conn.rollback()
                        except Exception as exec_err:
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): Unexpected error during update/notification execution: {exec_err}", exc_info=True)
                            if conn: conn.rollback()

                except (TypeError, ValueError) as convert_err:
                    logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): Error converting initial signal values: {convert_err}")
                    continue
                except Exception as inner_loop_err:
                     logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): Unexpected error while processing signal: {inner_loop_err}", exc_info=True)
                     continue

            if active_signals_summary:
                logger.debug(f"ℹ️ [Tracker] End of cycle state ({processed_in_cycle} processed): {'; '.join(active_signals_summary)}")

            time.sleep(3)

        except psycopg2.Error as db_cycle_err:
             logger.error(f"❌ [Tracker] Database error in main tracking cycle: {db_cycle_err}. Attempting to reconnect...")
             if conn: conn.rollback()
             time.sleep(30)
             check_db_connection()
        except Exception as cycle_err:
            logger.error(f"❌ [Tracker] Unexpected error in signal tracking cycle: {cycle_err}", exc_info=True)
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
        logger.warning("⚠️ [Flask] Webhook request received, but WEBHOOK_URL is not configured. Ignoring request.")
        return "Webhook not configured", 200 # Return OK to Telegram to avoid repeated attempts

    if not request.is_json:
        logger.warning("⚠️ [Flask] Non-JSON webhook request received.")
        return "Invalid request format", 400

    try:
        data = request.get_json()
        logger.info(f"✅ [Flask] Webhook data received. Data size: {len(json.dumps(data))} bytes.")
        logger.debug(f"ℹ️ [Flask] Full webhook data: {json.dumps(data)}") # Log full payload for debugging


        if 'callback_query' in data:
            callback_query = data['callback_query']
            callback_id = callback_query['id']
            callback_data = callback_query.get('data')
            message_info = callback_query.get('message')

            logger.info(f"ℹ️ [Flask] Callback Query received. ID: {callback_id}, Data: '{callback_data}'")

            if not message_info or not callback_data:
                 logger.warning(f"⚠️ [Flask] Callback query (ID: {callback_id}) missing message or data. Ignoring.")
                 try:
                     ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                     requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
                 except Exception as ack_err:
                     logger.warning(f"⚠️ [Flask] Failed to acknowledge invalid callback query {callback_id}: {ack_err}")
                 return "OK", 200
            chat_id_callback = message_info.get('chat', {}).get('id')
            if not chat_id_callback:
                 logger.warning(f"⚠️ [Flask] Callback query (ID: {callback_id}) missing chat ID. Ignoring.")
                 try:
                     ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                     requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
                 except Exception as ack_err:
                     logger.warning(f"⚠️ [Flask] Failed to acknowledge invalid callback query {callback_id}: {ack_err}")
                 return "OK", 200


            message_id = message_info['message_id']
            user_info = callback_query.get('from', {})
            user_id = user_info.get('id')
            username = user_info.get('username', 'N/A')

            logger.info(f"ℹ️ [Flask] Processing callback query: Data='{callback_data}', User={username}({user_id}), Chat={chat_id_callback}")

            try:
                # Always acknowledge the callback query to remove the loading animation from the button
                ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
                logger.debug(f"✅ [Flask] Callback query {callback_id} acknowledged.")
            except Exception as ack_err:
                 logger.warning(f"⚠️ [Flask] Failed to acknowledge callback query {callback_id}: {ack_err}")

            if callback_data == "get_report":
                logger.info(f"ℹ️ [Flask] Received 'get_report' request from chat {chat_id_callback}. Generating report...")
                report_content = generate_performance_report()
                logger.info(f"✅ [Flask] Report generated. Report length: {len(report_content)} characters.")
                report_thread = Thread(target=lambda: send_telegram_message(chat_id_callback, report_content, parse_mode='Markdown'))
                report_thread.start()
                logger.info(f"✅ [Flask] Started report sending thread for chat {chat_id_callback}.")
            else:
                logger.warning(f"⚠️ [Flask] Unhandled callback data received: '{callback_data}'")


        elif 'message' in data:
            message_data = data['message']
            chat_info = message_data.get('chat')
            user_info = message_data.get('from', {})
            text_msg = message_data.get('text', '').strip()

            if not chat_info or not text_msg:
                 logger.debug("ℹ️ [Flask] Message received without chat info or text.")
                 return "OK", 200

            chat_id_msg = chat_info['id']
            user_id = user_info.get('id')
            username = user_info.get('username', 'N/A')

            logger.info(f"ℹ️ [Flask] Message received: Text='{text_msg}', User={username}({user_id}), Chat={chat_id_msg}")

            if text_msg.lower() == '/report':
                 report_thread = Thread(target=lambda: send_telegram_message(chat_id_msg, generate_performance_report(), parse_mode='Markdown'))
                 report_thread.start()
            elif text_msg.lower() == '/status':
                 status_thread = Thread(target=handle_status_command, args=(chat_id_msg,))
                 status_thread.start()

        else:
            logger.debug("ℹ️ [Flask] Webhook data received without 'callback_query' or 'message'.")

        return "OK", 200
    except Exception as e:
         logger.error(f"❌ [Flask] Error processing webhook: {e}", exc_info=True)
         return "Internal Server Error", 500

def handle_status_command(chat_id_msg: int) -> None:
    """Separate function to handle /status command to avoid blocking the Webhook."""
    logger.info(f"ℹ️ [Flask Status] Processing /status command for chat {chat_id_msg}")
    status_msg = "⏳ Fetching status..."
    msg_sent = send_telegram_message(chat_id_msg, status_msg)
    if not (msg_sent and msg_sent.get('ok')):
         logger.error(f"❌ [Flask Status] Failed to send initial status message to {chat_id_msg}")
         return
    message_id_to_edit = msg_sent['result']['message_id'] if msg_sent and msg_sent.get('result') else None

    if message_id_to_edit is None:
        logger.error(f"❌ [Flask Status] Failed to get message_id to update status in chat {chat_id_msg}")
        return


    try:
        open_count = 0
        if check_db_connection() and conn:
            with conn.cursor() as status_cur:
                status_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE;")
                open_count = (status_cur.fetchone() or {}).get('count', 0)

        ws_status = 'Active ✅' if 'ws_thread' in globals() and ws_thread and ws_thread.is_alive() else 'Inactive ❌'
        tracker_status = 'Active ✅' if 'tracker_thread' in globals() and tracker_thread and tracker_thread.is_alive() else 'Inactive ❌'
        main_bot_alive = 'Active ✅' if 'main_bot_thread' in globals() and main_bot_thread and main_bot_thread.is_alive() else 'Inactive ❌'
        final_status_msg = f"""🤖 *Bot Status:*
- Price Tracking (WS): {ws_status}
- Signal Tracking: {tracker_status}
- Main Bot Loop: {main_bot_alive}
- Active Signals: *{open_count}* / {MAX_OPEN_TRADES}
- Current Server Time: {datetime.now().strftime('%H:%M:%S')}"""
        edit_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText"
        edit_payload = {
            'chat_id': chat_id_msg,
             'message_id': message_id_to_edit,
            'text': final_status_msg,
            'parse_mode': 'Markdown'
        }
        response = requests.post(edit_url, json=edit_payload, timeout=10)
        response.raise_for_status()
        logger.info(f"✅ [Flask Status] Status updated for chat {chat_id_msg}")

    except Exception as status_err:
        logger.error(f"❌ [Flask Status] Error fetching/editing status details for chat {chat_id_msg}: {status_err}", exc_info=True)
        send_telegram_message(chat_id_msg, "❌ An error occurred while fetching status details.")


def run_flask() -> None:
    """Runs the Flask application to listen for the Webhook using a production server if available."""
    host = "0.0.0.0"
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"ℹ️ [Flask] Starting Flask app on {host}:{port}...")
    try:
        from waitress import serve
        logger.info("✅ [Flask] Using 'waitress' server.")
        serve(app, host=host, port=port, threads=6)
    except ImportError:
         logger.warning("⚠️ [Flask] 'waitress' not installed. Falling back to Flask development server (not recommended for production).")
         try:
             app.run(host=host, port=port)
         except Exception as flask_run_err:
              logger.critical(f"❌ [Flask] Failed to start development server: {flask_run_err}", exc_info=True)
    except Exception as serve_err:
         logger.critical(f"❌ [Flask] Failed to start server (waitress?): {serve_err}", exc_info=True)

# ---------------------- Main Loop and Check Function ----------------------
def main_loop() -> None:
    """Main loop to scan pairs and generate signals."""
    symbols_to_scan = get_crypto_symbols()
    if not symbols_to_scan:
        logger.critical("❌ [Main] No valid symbols loaded or validated. Cannot proceed.")
        return

    logger.info(f"✅ [Main] Loaded {len(symbols_to_scan)} valid symbols for scanning.")
    last_full_scan_time = time.time()

    while True:
        try:
            scan_start_time = time.time()
            logger.info("+" + "-"*60 + "+")
            logger.info(f"🔄 [Main] Starting market scan cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("+" + "-"*60 + "+")

            if not check_db_connection() or not conn:
                logger.error("❌ [Main] Skipping scan cycle due to database connection failure.")
                time.sleep(60)
                continue

            open_count = 0
            try:
                 with conn.cursor() as cur_check:
                    cur_check.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE;")
                    open_count = (cur_check.fetchone() or {}).get('count', 0)
            except psycopg2.Error as db_err:
                 logger.error(f"❌ [Main] Database error while checking open signal count: {db_err}. Skipping cycle.")
                 if conn: conn.rollback()
                 time.sleep(60)
                 continue

            logger.info(f"ℹ️ [Main] Currently open signals: {open_count} / {MAX_OPEN_TRADES}")
            if open_count >= MAX_OPEN_TRADES:
                logger.info(f"⚠️ [Main] Maximum number of open signals reached. Waiting...")
                time.sleep(get_interval_minutes(SIGNAL_GENERATION_TIMEFRAME) * 60)
                continue

            processed_in_loop = 0
            signals_generated_in_loop = 0
            slots_available = MAX_OPEN_TRADES - open_count

            for symbol in symbols_to_scan:
                 if slots_available <= 0:
                      logger.info(f"ℹ️ [Main] Max open trades ({MAX_OPEN_TRADES}) reached during scan. Stopping symbol scan for this cycle.")
                      break

                 processed_in_loop += 1
                 logger.debug(f"🔍 [Main] Scanning {symbol} ({processed_in_loop}/{len(symbols_to_scan)})...")

                 try:
                    with conn.cursor() as symbol_cur:
                        symbol_cur.execute("SELECT 1 FROM signals WHERE symbol = %s AND achieved_target = FALSE LIMIT 1;", (symbol,))
                        if symbol_cur.fetchone():
                            continue

                    df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_hist is None or df_hist.empty:
                        continue

                    strategy = ScalpingTradingStrategy(symbol) # ML model loaded here
                    # Check if ML model was loaded successfully for this symbol
                    if strategy.ml_model is None:
                        logger.warning(f"⚠️ [Main] Skipping {symbol} because its ML model was not loaded successfully.")
                        continue

                    df_indicators = strategy.populate_indicators(df_hist)
                    if df_indicators is None:
                        continue

                    potential_signal = strategy.generate_buy_signal(df_indicators)

                    if potential_signal:
                        logger.info(f"✨ [Main] Potential signal found for {symbol}! Final check and insertion...")
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
                                     logger.error(f"❌ [Main] Failed to insert signal for {symbol} into database.")
                             else:
                                 logger.warning(f"⚠️ [Main] Max open trades ({final_open_count}) reached before inserting signal for {symbol}. Signal ignored.")
                                 break

                 except psycopg2.Error as db_loop_err:
                      logger.error(f"❌ [Main] Database error while processing symbol {symbol}: {db_loop_err}. Moving to next...")
                      if conn: conn.rollback()
                      continue
                 except Exception as symbol_proc_err:
                      logger.error(f"❌ [Main] General error processing symbol {symbol}: {symbol_proc_err}", exc_info=True)
                      continue

                 time.sleep(0.1)

            scan_duration = time.time() - scan_start_time
            logger.info(f"🏁 [Main] Scan cycle finished. Signals generated: {signals_generated_in_loop}. Scan duration: {scan_duration:.2f} seconds.")
            frame_minutes = get_interval_minutes(SIGNAL_GENERATION_TIMEFRAME)
            wait_time = max(frame_minutes * 60, 120 - scan_duration)
            logger.info(f"⏳ [Main] Waiting {wait_time:.1f} seconds for next cycle...")
            time.sleep(wait_time)

        except KeyboardInterrupt:
             logger.info("🛑 [Main] Stop requested (KeyboardInterrupt). Shutting down...")
             break
        except psycopg2.Error as db_main_err:
             logger.error(f"❌ [Main] Fatal database error in main loop: {db_main_err}. Attempting to reconnect...")
             if conn: conn.rollback()
             time.sleep(60)
             try:
                 init_db()
             except Exception as recon_err:
                 logger.critical(f"❌ [Main] Failed to reconnect to database: {recon_err}. Exiting...")
                 break
        except Exception as main_err:
            logger.error(f"❌ [Main] Unexpected error in main loop: {main_err}", exc_info=True)
            logger.info("ℹ️ [Main] Waiting 120 seconds before retrying...")
            time.sleep(120)

def cleanup_resources() -> None:
    """Closes used resources like the database connection."""
    global conn
    logger.info("ℹ️ [Cleanup] Closing resources...")
    if conn:
        try:
            conn.close()
            logger.info("✅ [DB] Database connection closed.")
        except Exception as close_err:
            logger.error(f"⚠️ [DB] Error closing database connection: {close_err}")
    logger.info("✅ [Cleanup] Resource cleanup complete.")


# ---------------------- Main Entry Point ----------------------
if __name__ == "__main__":
    logger.info("🚀 Starting crypto trading signal bot...")
    logger.info(f"Local Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | UTC Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    ws_thread: Optional[Thread] = None
    tracker_thread: Optional[Thread] = None
    flask_thread: Optional[Thread] = None
    main_bot_thread: Optional[Thread] = None # New thread for main_loop

    try:
        # 1. Initialize the database first
        init_db()

        # 2. Start WebSocket Ticker
        ws_thread = Thread(target=run_ticker_socket_manager, daemon=True, name="WebSocketThread")
        ws_thread.start()
        logger.info("✅ [Main] WebSocket ticker started.")
        logger.info("ℹ️ [Main] Waiting 5 seconds for WebSocket to initialize...")
        time.sleep(5)
        if not ticker_data:
             logger.warning("⚠️ [Main] No initial data received from WebSocket after 5 seconds.")
        else:
             logger.info(f"✅ [Main] Initial WebSocket data received for {len(ticker_data)} symbols.")


        # 3. Start Signal Tracker
        tracker_thread = Thread(target=track_signals, daemon=True, name="TrackerThread")
        tracker_thread.start()
        logger.info("✅ [Main] Signal tracker started.")

        # 4. Start the main bot logic in a separate thread
        main_bot_thread = Thread(target=main_loop, daemon=True, name="MainBotLoopThread")
        main_bot_thread.start()
        logger.info("✅ [Main] Main bot loop started in a separate thread.")

        # 5. Start Flask Server (ALWAYS run, daemon=False so it keeps the main program alive)
        flask_thread = Thread(target=run_flask, daemon=False, name="FlaskThread")
        flask_thread.start()
        logger.info("✅ [Main] Flask server started.")

        # Wait for the Flask thread to finish (it usually won't unless there's an error)
        flask_thread.join()

    except Exception as startup_err:
        logger.critical(f"❌ [Main] A fatal error occurred during startup or in the main loop: {startup_err}", exc_info=True)
    finally:
        logger.info("🛑 [Main] Shutting down program...")
        cleanup_resources()
        logger.info("👋 [Main] Crypto trading signal bot stopped.")
        os._exit(0)
