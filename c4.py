import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql, OperationalError, InterfaceError # For safe queries and specific errors
from psycopg2.extras import RealDictCursor # To get results as dictionaries
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException, BinanceRequestException # Specific Binance errors
from flask import Flask, request, Response
from threading import Thread
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Tuple, Any, Union # For Type Hinting

# ---------------------- Logging Setup ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # Add logger name
    handlers=[
        logging.FileHandler('crypto_bot_dynamic_tracking.log', encoding='utf-8'), # Changed log file name
        logging.StreamHandler()
    ]
)
# Use a specific logger name instead of the root
logger = logging.getLogger('CryptoBotDynamic') # Changed logger name slightly

# ---------------------- Load Environment Variables ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
    # Use a default value of None if the variable is not set
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
except Exception as e:
     logger.critical(f"❌ Failed to load essential environment variables: {e}")
     exit(1) # Use a non-zero exit code to indicate an error

logger.info(f"Binance API Key: {'Available' if API_KEY else 'Not available'}")
logger.info(f"Telegram Token: {TELEGRAM_TOKEN[:10]}...{'*' * (len(TELEGRAM_TOKEN)-10)}")
logger.info(f"Telegram Chat ID: {CHAT_ID}")
logger.info(f"Database URL: {'Available' if DB_URL else 'Not available'}")
logger.info(f"Webhook URL: {WEBHOOK_URL if WEBHOOK_URL else 'Not specified'}")

# ---------------------- Constants and Global Variables ----------------------
TRADE_VALUE: float = 10.0         # Default trade value in USDT
MAX_OPEN_TRADES: int = 4          # Maximum number of open trades simultaneously
SIGNAL_GENERATION_TIMEFRAME: str = '30m' # Timeframe for signal generation
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 5 # Historical data lookback in days for signal generation
SIGNAL_TRACKING_TIMEFRAME: str = '30m' # Timeframe for signal tracking and stop loss updates
SIGNAL_TRACKING_LOOKBACK_DAYS: int = 5   # Historical data lookback in days for signal tracking (used for swings)
TRACKING_CYCLE_SLEEP_SECONDS: int = 30 # Sleep time between tracking cycles (Increased due to historical data fetching)

# =============================================================================
# --- Indicator Parameters ---
# =============================================================================
RSI_PERIOD: int = 14          # RSI Period
RSI_OVERSOLD: int = 30        # Oversold threshold
RSI_OVERBOUGHT: int = 70      # Overbought threshold
EMA_SHORT_PERIOD: int = 13      # Short EMA period
EMA_LONG_PERIOD: int = 34       # Long EMA period
VWMA_PERIOD: int = 20           # VWMA Period
SWING_ORDER: int = 5          # Order for swing point detection (for Fib and Tracking)
FIB_LEVELS_TO_CHECK: List[float] = [0.382, 0.5, 0.618] # Fibonacci levels for entry check
FIB_TOLERANCE: float = 0.007 # Tolerance for checking price near Fib level (0.7%)
LOOKBACK_FOR_SWINGS: int = 100 # How many candles back to look for swing points for Fib calculation
ENTRY_ATR_PERIOD: int = 14     # ATR Period for entry and initial SL/TP
ENTRY_ATR_MULTIPLIER: float = 3.5 # ATR Multiplier for initial target/stop

# Adjusted TP Multipliers to potentially meet R:R >= 1.5
TP1_ATR_MULTIPLIER: float = 3.0 # Increased from 1.5 - Used for R:R calc and BE trigger
TP2_ATR_MULTIPLIER: float = 4.5 # Increased from 2.5
TP3_ATR_MULTIPLIER: float = 6.0 # Increased from 3.5

BOLLINGER_WINDOW: int = 20     # Bollinger Bands Window
BOLLINGER_STD_DEV: int = 2       # Bollinger Bands Standard Deviation
MACD_FAST: int = 12            # MACD Fast Period
MACD_SLOW: int = 26            # MACD Slow Period
MACD_SIGNAL: int = 9             # MACD Signal Line Period
ADX_PERIOD: int = 14            # ADX Period
SUPERTREND_PERIOD: int = 10     # SuperTrend Period
SUPERTREND_MULTIPLIER: float = 3.0 # SuperTrend Multiplier

# --- Dynamic Trailing Stop (Market Structure Based) ---
SWING_SL_BUFFER_PCT: float = 0.002 # Percentage buffer below swing low for SL (0.2%)
# ------------------------------------------------------

# Additional Signal Conditions
# MIN_PROFIT_MARGIN_PCT check will now use TP1 price (calculated with TP1_ATR_MULTIPLIER)
MIN_PROFIT_MARGIN_PCT: float = 2 # Minimum required profit margin percentage for initial TP (based on TP1)
MIN_VOLUME_15M_USDT: float = 180000.0 # Minimum liquidity in the last 15 minutes in USDT
MIN_RR_RATIO: float = 1.5 # Minimum Risk:Reward ratio required for a signal (New)
# =============================================================================
# --- End Indicator Parameters ---
# =============================================================================

# Global variables (will be initialized later)
conn: Optional[psycopg2.extensions.connection] = None
cur: Optional[psycopg2.extensions.cursor] = None
client: Optional[Client] = None
ticker_data: Dict[str, float] = {} # Dictionary to store the latest closing prices for symbols

# ---------------------- Binance Client Setup ----------------------
try:
    logger.info("ℹ️ [Binance] Initializing Binance client...")
    client = Client(API_KEY, API_SECRET)
    client.ping() # Check connection and keys validity
    server_time = client.get_server_time()
    logger.info(f"✅ [Binance] Binance client initialized. Server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
except BinanceRequestException as req_err:
     logger.critical(f"❌ [Binance] Binance request error (network or request issue): {req_err}")
     exit(1)
except BinanceAPIException as api_err:
     logger.critical(f"❌ [Binance] Binance API error (invalid keys or server issue): {api_err}")
     exit(1)
except Exception as e:
    logger.critical(f"❌ [Binance] Unexpected failure initializing Binance client: {e}")
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
         logger.error(f"❌ [Indicators] Network error fetching Fear & Greed Index: {e}")
         return "N/A (Network Error)"
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"❌ [Indicators] Data format error for Fear & Greed Index: {e}")
        return "N/A (Data Error)"
    except Exception as e:
        logger.error(f"❌ [Indicators] Unexpected error fetching Fear & Greed Index: {e}", exc_info=True)
        return "N/A (Unknown Error)"

def fetch_historical_data(symbol: str, interval: str = SIGNAL_GENERATION_TIMEFRAME, days: int = SIGNAL_GENERATION_LOOKBACK_DAYS, limit_override: Optional[int] = None) -> Optional[pd.DataFrame]:
    """Fetches historical candlestick data from Binance. Allows overriding the limit."""
    if not client:
        logger.error("❌ [Data] Binance client not initialized for data fetching.")
        return None
    try:
        if limit_override:
            limit = limit_override
            # Estimate start time based on limit and interval
            interval_duration_minutes = {
                '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
                '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200 # Approximate
            }.get(interval, 60)
            start_dt = datetime.utcnow() - timedelta(minutes=(limit * interval_duration_minutes * 1.1)) # Add 10% buffer
            start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            logger.debug(f"ℹ️ [Data] Fetching {interval} data for {symbol} (Limit Override: {limit})...")
        else:
            # Calculate start time based on days and interval (approximate number of candles)
            interval_duration_minutes = {
                '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
                '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200 # Approximate
            }.get(interval, 60) # Default to 1 hour if interval not found

            # Calculate approximate number of candles needed, add buffer
            candles_needed = int((days * 24 * 60) / interval_duration_minutes)
            # Ensure enough for swings, max 1000 unless overridden
            limit = min(max(candles_needed + 50, LOOKBACK_FOR_SWINGS + SWING_ORDER * 2 + 50), 1000) # Ensure enough for swing detection

            start_dt = datetime.utcnow() - timedelta(minutes=(limit * interval_duration_minutes))
            start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            logger.debug(f"ℹ️ [Data] Fetching {interval} data for {symbol} since {start_str} (limit {limit} candles)...")


        klines = client.get_historical_klines(symbol, interval, start_str, limit=limit)

        if not klines:
            logger.warning(f"⚠️ [Data] No historical data ({interval}) for {symbol} for the requested period.")
            return None

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])

        # Define essential numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce') # coerce invalid values to NaN

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Select only the required columns
        df = df[numeric_cols]

        initial_len = len(df)
        df.dropna(subset=numeric_cols, inplace=True) # Drop rows with NaN in essential columns

        if len(df) < initial_len:
            logger.debug(f"ℹ️ [Data] {symbol}: Dropped {initial_len - len(df)} rows due to NaN in OHLCV data.")

        if df.empty:
            logger.warning(f"⚠️ [Data] DataFrame for {symbol} is empty after removing essential NaNs.")
            return None

        # Ensure index is sorted chronologically
        df = df.sort_index()

        logger.debug(f"✅ [Data] Fetched and processed {len(df)} historical candles ({interval}) for {symbol}.")
        return df

    except BinanceAPIException as api_err:
         logger.error(f"❌ [Data] Binance API error fetching data for {symbol}: {api_err}")
         return None
    except BinanceRequestException as req_err:
         logger.error(f"❌ [Data] Request or network error fetching data for {symbol}: {req_err}")
         return None
    except Exception as e:
        logger.error(f"❌ [Data] Unexpected error fetching historical data for {symbol}: {e}", exc_info=True)
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
        logger.warning("⚠️ [Indicator VWMA] 'close' or 'volume' columns missing or empty.")
        return pd.Series(index=df_calc.index if df_calc is not None else None, dtype=float)
    if len(df_calc) < period:
        logger.warning(f"⚠️ [Indicator VWMA] Insufficient data ({len(df_calc)} < {period}) to calculate VWMA.")
        return pd.Series(index=df_calc.index if df_calc is not None else None, dtype=float)
    df_calc['price_volume'] = df_calc['close'] * df_calc['volume']
    rolling_price_volume_sum = df_calc['price_volume'].rolling(window=period, min_periods=period).sum()
    rolling_volume_sum = df_calc['volume'].rolling(window=period, min_periods=period).sum()
    vwma = rolling_price_volume_sum / rolling_volume_sum.replace(0, np.nan)
    df_calc.drop(columns=['price_volume'], inplace=True, errors='ignore')
    return vwma

def get_btc_trend_4h() -> str:
    """Calculates Bitcoin trend on 4-hour timeframe using EMA20 and EMA50."""
    logger.debug("ℹ️ [Indicators] Calculating Bitcoin 4-hour trend...")
    try:
        df = fetch_historical_data("BTCUSDT", interval=Client.KLINE_INTERVAL_4HOUR, days=10)
        if df is None or df.empty or len(df) < 50 + 1:
            logger.warning("⚠️ [Indicators] Insufficient BTC/USDT 4H data to calculate trend.")
            return "N/A (Insufficient Data)"
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df.dropna(subset=['close'], inplace=True)
        if len(df) < 50:
             logger.warning("⚠️ [Indicators] Insufficient BTC/USDT 4H data after removing NaNs.")
             return "N/A (Insufficient Data)"
        ema20 = calculate_ema(df['close'], 20).iloc[-1]
        ema50 = calculate_ema(df['close'], 50).iloc[-1]
        current_close = df['close'].iloc[-1]
        if pd.isna(ema20) or pd.isna(ema50) or pd.isna(current_close):
            logger.warning("⚠️ [Indicators] BTC EMA or current price values are NaN.")
            return "N/A (Calculation Error)"
        diff_ema20_pct = abs(current_close - ema20) / current_close if current_close > 0 else 0
        if current_close > ema20 > ema50: trend = "صعود 📈"
        elif current_close < ema20 < ema50: trend = "هبوط 📉"
        elif diff_ema20_pct < 0.005: trend = "استقرار 🔄"
        else: trend = "تذبذب 🔀"
        logger.debug(f"✅ [Indicators] Bitcoin 4H Trend: {trend}")
        return trend
    except Exception as e:
        logger.error(f"❌ [Indicators] Error calculating Bitcoin 4-hour trend: {e}", exc_info=True)
        return "N/A (Error)"

# ---------------------- Database Connection Setup ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    """ Initializes database connection and creates/updates tables if they don't exist. """
    global conn, cur
    logger.info("[DB] Starting database initialization...")
    for attempt in range(retries):
        try:
            logger.info(f"[DB] Attempting to connect to database (Attempt {attempt + 1}/{retries})...")
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False # Manual commit/rollback control
            cur = conn.cursor()
            logger.info("✅ [DB] Successfully connected to database.")

            # --- Create or update signals table ---
            logger.info("[DB] Checking/Creating 'signals' table...")
            # Added columns for TP2, TP3, and R:R ratio
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    initial_target DOUBLE PRECISION NOT NULL, -- Original target based on entry ATR (Kept for historical compatibility, but TP1 is now primary)
                    initial_stop_loss DOUBLE PRECISION NOT NULL, -- Original SL based on entry ATR or Swing
                    current_target DOUBLE PRECISION NOT NULL, -- Might become dynamic or less relevant
                    current_stop_loss DOUBLE PRECISION NOT NULL, -- This will be updated dynamically
                    r2_score DOUBLE PRECISION, -- Weighted signal score
                    volume_15m DOUBLE PRECISION,
                    achieved_target BOOLEAN DEFAULT FALSE, -- Flag if the primary target (e.g., TP1) was hit - Renamed to tp1_hit
                    hit_stop_loss BOOLEAN DEFAULT FALSE, -- Flag if the final stop loss was hit
                    closing_price DOUBLE PRECISION, -- Price at which the trade was fully closed
                    closed_at TIMESTAMP, -- Timestamp when the trade was fully closed
                    sent_at TIMESTAMP DEFAULT NOW(),
                    profit_percentage DOUBLE PRECISION, -- Final profit/loss percentage for the closed trade
                    profitable_stop_loss BOOLEAN DEFAULT FALSE, -- Was the SL hit above entry?
                    is_trailing_active BOOLEAN DEFAULT FALSE, -- Is the dynamic trailing stop active?
                    strategy_name TEXT,
                    signal_details JSONB,
                    last_trailing_update_price DOUBLE PRECISION, -- Price when SL was last updated (used for ATR trailing, less relevant for swing)

                    -- === COLUMNS FOR DYNAMIC TRACKING & PARTIAL PROFITS ===
                    tp1_price DOUBLE PRECISION, -- Price for the first take profit (triggers break-even)
                    tp1_hit BOOLEAN DEFAULT FALSE,
                    tp2_price DOUBLE PRECISION, -- Price for the second take profit
                    tp2_hit BOOLEAN DEFAULT FALSE,
                    tp3_price DOUBLE PRECISION, -- Price for the third take profit
                    tp3_hit BOOLEAN DEFAULT FALSE,
                    stop_loss_at_breakeven BOOLEAN DEFAULT FALSE,
                    last_swing_low_price DOUBLE PRECISION, -- Stores the price of the swing low defining the current SL
                    last_swing_high_price DOUBLE PRECISION, -- Stores the price of the swing high used for initial activation
                    initial_atr DOUBLE PRECISION, -- Store ATR at entry for reference
                    risk_reward_ratio DOUBLE PRECISION -- Store R:R ratio (New)
                    -- ============================================================
                );""")
            conn.commit()
            logger.info("✅ [DB] 'signals' table exists or was created.")

            # --- Check and add missing columns (if necessary) ---
            # This section needs manual adjustment if you add columns later.
            required_columns = {
                "symbol", "entry_price", "initial_target", "initial_stop_loss",
                "current_target", "current_stop_loss", "r2_score", "volume_15m",
                # "achieved_target", # Replaced by tp1_hit
                "hit_stop_loss", "closing_price", "closed_at",
                "sent_at", "profit_percentage", "profitable_stop_loss",
                "is_trailing_active", "strategy_name", "signal_details",
                "last_trailing_update_price",
                # Add new columns here for the check
                "tp1_price", "tp1_hit", "tp2_price", "tp2_hit", "tp3_price", "tp3_hit",
                "stop_loss_at_breakeven", "last_swing_low_price",
                "last_swing_high_price", "initial_atr", "risk_reward_ratio"
            }
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'signals' AND table_schema = 'public';")
            existing_columns = {row['column_name'] for row in cur.fetchall()}
            missing_columns = required_columns - existing_columns

            if missing_columns:
                logger.warning(f"⚠️ [DB] Following columns are missing in 'signals' table: {missing_columns}. Please add them manually using ALTER TABLE commands.")
                # Example commands (execute these manually via psql or a DB tool):
                # ALTER TABLE signals ADD COLUMN tp1_price DOUBLE PRECISION;
                # ALTER TABLE signals ADD COLUMN tp1_hit BOOLEAN DEFAULT FALSE;
                # ALTER TABLE signals ADD COLUMN tp2_price DOUBLE PRECISION;
                # ALTER TABLE signals ADD COLUMN tp2_hit BOOLEISION DEFAULT FALSE;
                # ALTER TABLE signals ADD COLUMN tp3_price DOUBLE PRECISION;
                # ALTER TABLE signals ADD COLUMN tp3_hit BOOLEAN DEFAULT FALSE;
                # ALTER TABLE signals ADD COLUMN stop_loss_at_breakeven BOOLEAN DEFAULT FALSE;
                # ALTER TABLE signals ADD COLUMN last_swing_low_price DOUBLE PRECISION;
                # ALTER TABLE signals ADD COLUMN last_swing_high_price DOUBLE PRECISION;
                # ALTER TABLE signals ADD COLUMN initial_atr DOUBLE PRECISION;
                # ALTER TABLE signals ADD COLUMN risk_reward_ratio DOUBLE PRECISION;
            else:
                logger.info("✅ [DB] All required columns (including dynamic tracking and partial profits) exist in 'signals' table.")

            # --- Create market_dominance table (if it doesn't exist) ---
            logger.info("[DB] Checking/Creating 'market_dominance' table...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS market_dominance (
                    id SERIAL PRIMARY KEY,
                    recorded_at TIMESTAMP DEFAULT NOW(),
                    btc_dominance DOUBLE PRECISION,
                    eth_dominance DOUBLE PRECISION
                );
            """)
            conn.commit()
            logger.info("✅ [DB] 'market_dominance' table exists or was created.")

            logger.info("✅ [DB] Database initialization successful.")
            return # Connection and initialization successful

        except OperationalError as op_err:
            logger.error(f"❌ [DB] Operational error connecting (Attempt {attempt + 1}): {op_err}")
            if conn: conn.rollback()
            if attempt == retries - 1:
                 logger.critical("❌ [DB] All database connection attempts failed.")
                 raise op_err # Re-raise the error after all attempts fail
            time.sleep(delay)
        except Exception as e:
            logger.critical(f"❌ [DB] Unexpected failure initializing database (Attempt {attempt + 1}): {e}", exc_info=True)
            if conn: conn.rollback()
            if attempt == retries - 1:
                 logger.critical("❌ [DB] All database connection attempts failed.")
                 raise e
            time.sleep(delay)

    # If code reaches here, all attempts failed
    logger.critical("❌ [DB] Database connection failed after multiple attempts.")
    exit(1)


def check_db_connection() -> bool:
    """Checks database connection status and re-initializes if necessary."""
    global conn, cur
    try:
        if conn is None or conn.closed != 0:
            logger.warning("⚠️ [DB] Connection closed or not found. Re-initializing...")
            init_db() # Attempt to reconnect and initialize
            return True # Assume successful initialization (init_db will raise error if it fails)
        else:
             # Check that the connection is still working by sending a simple query
             with conn.cursor() as check_cur: # Use a temporary cursor
                  check_cur.execute("SELECT 1;")
                  check_cur.fetchone()
             # logger.debug("[DB] Connection is active.") # Uncomment for frequent checks
             return True
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [DB] Database connection lost ({e}). Re-initializing...")
        try:
             init_db()
             return True
        except Exception as recon_err:
            logger.error(f"❌ [DB] Reconnection attempt failed after connection loss: {recon_err}")
            return False
    except Exception as e:
        logger.error(f"❌ [DB] Unexpected error during connection check: {e}", exc_info=True)
        # Attempt to reconnect as a precautionary measure
        try:
            init_db()
            return True
        except Exception as recon_err:
             logger.error(f"❌ [DB] Reconnection attempt failed after unexpected error: {recon_err}")
             return False


def convert_np_values(obj: Any) -> Any:
    """Converts NumPy data types to native Python types for JSON and DB compatibility."""
    if isinstance(obj, dict):
        return {k: convert_np_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_values(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int_)): # np.int_ is old but still works
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)): # np.float64 used directly
        # Handle potential infinity or NaN from calculations
        if np.isinf(obj):
            return None # Or a very large number, depending on context
        elif np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif pd.isna(obj): # Handle NaT from Pandas as well
        return None
    else:
        return obj

# ---------------------- Fetching and Validating Symbols from Binance ----------------------
def get_crypto_symbols_from_binance() -> List[str]:
    """
    Fetches all valid USDT Spot trading pairs directly from Binance API.
    Replaces reading from a local file.
    """
    if not client:
        logger.error("❌ [Data] Binance client not initialized. Cannot fetch symbols from API.")
        return [] # Return empty list if client is not available

    logger.info("ℹ️ [Data] Fetching valid USDT Spot trading symbols from Binance API...")
    try:
        exchange_info = client.get_exchange_info()
        valid_trading_usdt_symbols = sorted([
            s['symbol'] for s in exchange_info['symbols']
            if s.get('quoteAsset') == 'USDT' and
               s.get('status') == 'TRADING' and
               s.get('isSpotTradingAllowed') is True
        ])

        logger.info(f"✅ [Data] Fetched {len(valid_trading_usdt_symbols)} valid USDT Spot trading pairs from Binance.")
        return valid_trading_usdt_symbols

    except (BinanceAPIException, BinanceRequestException) as binance_err:
         logger.error(f"❌ [Data] Binance API or network error while fetching symbols: {binance_err}")
         return []
    except Exception as api_err:
         logger.error(f"❌ [Data] Unexpected error while fetching Binance symbols: {api_err}", exc_info=True)
         return []

# ---------------------- WebSocket Management for Ticker Prices ----------------------
def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
    """Handles incoming WebSocket messages for mini-ticker prices."""
    global ticker_data
    try:
        if isinstance(msg, list):
            for ticker_item in msg:
                symbol = ticker_item.get('s')
                price_str = ticker_item.get('c') # Last closing price as string
                if symbol and 'USDT' in symbol and price_str:
                    try:
                        ticker_data[symbol] = float(price_str)
                    except ValueError:
                         logger.warning(f"⚠️ [WS] Invalid price value for symbol {symbol}: '{price_str}'")
        elif isinstance(msg, dict):
             if msg.get('e') == 'error':
                 logger.error(f"❌ [WS] Error message from WebSocket: {msg.get('m', 'No error details')}")
             elif msg.get('stream') and msg.get('data'): # Handle combined streams format
                 for ticker_item in msg.get('data', []):
                    symbol = ticker_item.get('s')
                    price_str = ticker_item.get('c')
                    if symbol and 'USDT' in symbol and price_str:
                        try:
                            ticker_data[symbol] = float(price_str)
                        except ValueError:
                             logger.warning(f"⚠️ [WS] Invalid price value for symbol {symbol} in combined stream: '{price_str}'")
        else:
             logger.warning(f"⚠️ [WS] Received WebSocket message with unexpected format: {type(msg)}")

    except Exception as e:
        logger.error(f"❌ [WS] Error processing ticker message: {e}", exc_info=True)

def run_ticker_socket_manager() -> None:
    """Runs and manages the WebSocket connection for mini-ticker."""
    while True:
        try:
            logger.info("ℹ️ [WS] Starting WebSocket Manager for Ticker prices...")
            twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
            twm.start() # Start the manager
            stream_name = twm.start_miniticker_socket(callback=handle_ticker_message)
            logger.info(f"✅ [WS] WebSocket stream started: {stream_name}")
            twm.join() # Wait until the manager stops
            logger.warning("⚠️ [WS] WebSocket Manager stopped. Restarting...")
        except Exception as e:
            logger.error(f"❌ [WS] Fatal error in WebSocket Manager: {e}. Restarting in 15 seconds...", exc_info=True)
        time.sleep(15)

# ---------------------- Technical Indicator Functions ----------------------
def calculate_atr_indicator(df: pd.DataFrame, period: int) -> pd.DataFrame: # Added period parameter type hint
    """Calculates Average True Range (ATR). Uses the provided period."""
    df = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().all().any():
        logger.warning(f"⚠️ [Indicator ATR period={period}] 'high', 'low', 'close' columns missing or empty.")
        df['atr'] = np.nan
        return df
    if len(df) < period + 1: # We need one extra candle for shift(1)
        logger.warning(f"⚠️ [Indicator ATR period={period}] Insufficient data ({len(df)} < {period + 1}) to calculate ATR.")
        df['atr'] = np.nan
        return df
    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)
    # Use the specific period passed to the function
    df['atr'] = tr.ewm(span=period, adjust=False).mean()
    return df

def calculate_rsi_indicator(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """Calculates Relative Strength Index (RSI)."""
    df = df.copy()
    if 'close' not in df.columns or df['close'].isnull().all():
        logger.warning("⚠️ [Indicator RSI] 'close' column missing or empty.")
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

def calculate_bollinger_bands(df: pd.DataFrame, window: int = BOLLINGER_WINDOW, num_std: int = BOLLINGER_STD_DEV) -> pd.DataFrame:
    """Calculates Bollinger Bands."""
    df = df.copy()
    if 'close' not in df.columns or df['close'].isnull().all():
        logger.warning("⚠️ [Indicator BB] 'close' column missing or empty.")
        df['bb_middle'] = np.nan; df['bb_upper'] = np.nan; df['bb_lower'] = np.nan
        return df
    if len(df) < window:
         logger.warning(f"⚠️ [Indicator BB] Insufficient data ({len(df)} < {window}) to calculate BB.")
         df['bb_middle'] = np.nan; df['bb_upper'] = np.nan; df['bb_lower'] = np.nan
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
        logger.warning("⚠️ [Indicator MACD] 'close' column missing or empty.")
        df['macd'] = np.nan; df['macd_signal'] = np.nan; df['macd_hist'] = np.nan
        return df
    min_len = max(fast, slow, signal)
    if len(df) < min_len:
        logger.warning(f"⚠️ [Indicator MACD] Insufficient data ({len(df)} < {min_len}) to calculate MACD.")
        df['macd'] = np.nan; df['macd_signal'] = np.nan; df['macd_hist'] = np.nan
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
        logger.warning("⚠️ [Indicator ADX] 'high', 'low', 'close' columns missing or empty.")
        df_calc['adx'] = np.nan; df_calc['di_plus'] = np.nan; df_calc['di_minus'] = np.nan
        return df_calc
    if len(df_calc) < period * 2:
        logger.warning(f"⚠️ [Indicator ADX] Insufficient data ({len(df_calc)} < {period * 2}) to calculate ADX.")
        df_calc['adx'] = np.nan; df_calc['di_plus'] = np.nan; df_calc['di_minus'] = np.nan
        return df_calc
    df_calc['high-low'] = df_calc['high'] - df_calc['low']
    df_calc['high-prev_close'] = abs(df_calc['high'] - df_calc['close'].shift(1))
    df_calc['low-prev_close'] = abs(df_calc['low'] - df_calc['close'].shift(1))
    df_calc['tr'] = df_calc[['high-low', 'high-prev_close', 'low-prev_close']].max(axis=1, skipna=False)
    df_calc['up_move'] = df_calc['high'] - df_calc['high'].shift(1)
    df_calc['down_move'] = df_calc['low'].shift(1) - df_calc['low']
    df_calc['+dm'] = np.where((df_calc['up_move'] > df_calc['down_move']) & (df_calc['up_move'] > 0), df_calc['up_move'], 0)
    # Corrected typo: df_move should be df_calc
    df_calc['-dm'] = np.where((df_calc['down_move'] > df_calc['up_move']) & (df_calc['down_move'] > 0), df_calc['down_move'], 0)
    alpha = 1 / period
    df_calc['tr_smooth'] = df_calc['tr'].ewm(alpha=alpha, adjust=False).mean()
    df_calc['+dm_smooth'] = df_calc['+dm'].ewm(alpha=alpha, adjust=False).mean()
    # Corrected typo: df_move should be df_calc
    df_calc['-dm_smooth'] = df_calc['-dm'].ewm(alpha=alpha, adjust=False).mean()
    df_calc['di_plus'] = np.where(df_calc['tr_smooth'] > 0, 100 * (df_calc['+dm_smooth'] / df_calc['tr_smooth']), 0)
    df_calc['di_minus'] = np.where(df_calc['tr_smooth'] > 0, 100 * (df_calc['-dm_smooth'] / df_calc['tr_smooth']), 0)
    di_sum = df_calc['di_plus'] + df_calc['di_minus']
    df_calc['dx'] = np.where(di_sum > 0, 100 * abs(df_calc['di_plus'] - df_calc['di_minus']) / di_sum, 0)
    df_calc['adx'] = df_calc['dx'].ewm(alpha=alpha, adjust=False).mean()
    return df_calc[['adx', 'di_plus', 'di_minus']]

def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Volume Weighted Average Price (VWAP) - Resets daily."""
    df = df.copy()
    required_cols = ['high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator VWAP] Required columns missing or empty.")
        df['vwap'] = np.nan; return df
    if not isinstance(df.index, pd.DatetimeIndex):
        try: df.index = pd.to_datetime(df.index)
        except Exception: logger.error("❌ [Indicator VWAP] Failed to convert index to DatetimeIndex."); df['vwap'] = np.nan; return df
    df['date'] = df.index.date
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol'] = df['typical_price'] * df['volume']
    try:
        df['cum_tp_vol'] = df.groupby('date')['tp_vol'].cumsum()
        df['cum_volume'] = df.groupby('date')['volume'].cumsum()
    except Exception as e: logger.error(f"❌ [Indicator VWAP] Error grouping: {e}"); df['vwap'] = np.nan; df.drop(columns=['date', 'typical_price', 'tp_vol', 'cum_tp_vol', 'cum_volume'], inplace=True, errors='ignore'); return df
    df['vwap'] = np.where(df['cum_volume'] > 0, df['cum_tp_vol'] / df['cum_volume'], np.nan)
    df['vwap'] = df['vwap'].bfill()
    df.drop(columns=['date', 'typical_price', 'tp_vol', 'cum_tp_vol', 'cum_volume'], inplace=True, errors='ignore')
    return df

def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates On-Balance Volume (OBV)."""
    df = df.copy()
    required_cols = ['close', 'volume']
    if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator OBV] 'close' or 'volume' columns missing or empty.")
        df['obv'] = np.nan; return df
    if not pd.api.types.is_numeric_dtype(df['close']) or not pd.api.types.is_numeric_dtype(df['volume']):
        logger.warning("⚠️ [Indicator OBV] 'close' or 'volume' columns are not numeric.")
        df['obv'] = np.nan; return df
    obv = np.zeros(len(df), dtype=np.float64)
    close = df['close'].values; volume = df['volume'].values
    close_diff = df['close'].diff().values
    for i in range(1, len(df)):
        if np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(close_diff[i]): obv[i] = obv[i-1]; continue
        if close_diff[i] > 0: obv[i] = obv[i-1] + volume[i]
        elif close_diff[i] < 0: obv[i] = obv[i-1] - volume[i]
        else: obv[i] = obv[i-1]
    df['obv'] = obv
    return df

def calculate_supertrend(df: pd.DataFrame, period: int = SUPERTREND_PERIOD, multiplier: float = SUPERTREND_MULTIPLIER) -> pd.DataFrame:
    """Calculates the SuperTrend indicator."""
    df_st = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df_st.columns for col in required_cols) or df_st[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator SuperTrend] Required columns missing or empty.")
        df_st['supertrend'] = np.nan; df_st['supertrend_trend'] = 0; return df_st
    # Ensure ATR is calculated for the SuperTrend period
    if 'atr' not in df_st.columns or df_st['atr'].isnull().all() or calculate_atr_indicator(df_st, period)['atr'].isnull().all():
        logger.debug(f"ℹ️ [Indicator SuperTrend] Calculating ATR (period={period}) for SuperTrend...")
        df_st = calculate_atr_indicator(df_st, period=period) # Use the correct ATR period
    if 'atr' not in df_st.columns or df_st['atr'].isnull().all():
         logger.warning("⚠️ [Indicator SuperTrend] Cannot calculate SuperTrend due to missing valid ATR values.")
         df_st['supertrend'] = np.nan; df_st['supertrend_trend'] = 0; return df_st
    if len(df_st) < period:
        logger.warning(f"⚠️ [Indicator SuperTrend] Insufficient data ({len(df_st)} < {period}) to calculate SuperTrend.")
        df_st['supertrend'] = np.nan; df_st['supertrend_trend'] = 0; return df_st
    hl2 = (df_st['high'] + df_st['low']) / 2
    df_st['basic_ub'] = hl2 + multiplier * df_st['atr']
    df_st['basic_lb'] = hl2 - multiplier * df_st['atr']
    df_st['final_ub'] = 0.0; df_st['final_lb'] = 0.0; df_st['supertrend'] = np.nan; df_st['supertrend_trend'] = 0
    close = df_st['close'].values; basic_ub = df_st['basic_ub'].values; basic_lb = df_st['basic_lb'].values
    final_ub = df_st['final_ub'].values; final_lb = df_st['final_lb'].values; st = df_st['supertrend'].values; st_trend = df_st['supertrend_trend'].values
    for i in range(1, len(df_st)):
        if pd.isna(basic_ub[i]) or pd.isna(basic_lb[i]) or pd.isna(close[i]):
            final_ub[i] = final_ub[i-1]; final_lb[i] = final_lb[i-1]; st[i] = st[i-1]; st_trend[i] = st_trend[i-1]; continue
        if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]: final_ub[i] = basic_ub[i]
        else: final_ub[i] = final_ub[i-1]
        if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]: final_lb[i] = basic_lb[i]
        else: final_lb[i] = final_lb[i-1]
        if st_trend[i-1] == -1:
            if close[i] <= final_ub[i]: st[i] = final_ub[i]; st_trend[i] = -1
            else: st[i] = final_lb[i]; st_trend[i] = 1
        elif st_trend[i-1] == 1:
            if close[i] >= final_lb[i]: st[i] = final_lb[i]; st_trend[i] = 1
            else: st[i] = final_ub[i]; st_trend[i] = -1
        else:
             if close[i] > final_ub[i]: st[i] = final_lb[i]; st_trend[i] = 1
             elif close[i] < final_lb[i]: st[i] = final_ub[i]; st_trend[i] = -1
             else:
                  if close[i] > basic_ub[i]: st[i] = basic_lb[i]; st_trend[i] = 1
                  elif close[i] < basic_lb[i]: st[i] = basic_ub[i]; st_trend[i] = -1
                  else: st[i] = np.nan; st_trend[i] = 0
    df_st['final_ub'] = final_ub; df_st['final_lb'] = final_lb; df_st['supertrend'] = st; df_st['supertrend_trend'] = st_trend
    df_st.drop(columns=['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, errors='ignore')
    return df_st

# --- Candlestick Pattern Functions ---
def is_hammer(row: pd.Series) -> int:
    o, h, l, c = row.get('open'), row.get('high'), row.get('low'), row.get('close')
    if pd.isna([o, h, l, c]).any(): return 0
    body = abs(c - o); candle_range = h - l
    if candle_range == 0: return 0
    lower_shadow = min(o, c) - l; upper_shadow = h - max(o, c)
    is_small_body = body < (candle_range * 0.35)
    is_long_lower_shadow = lower_shadow >= 1.8 * body if body > 0 else lower_shadow > candle_range * 0.6
    is_small_upper_shadow = upper_shadow <= body * 0.6 if body > 0 else upper_shadow < candle_range * 0.15
    return 100 if is_small_body and is_long_lower_shadow and is_small_upper_shadow else 0

def is_shooting_star(row: pd.Series) -> int:
    o, h, l, c = row.get('open'), row.get('high'), row.get('low'), row.get('close')
    if pd.isna([o, h, l, c]).any(): return 0
    body = abs(c - o); candle_range = h - l
    if candle_range == 0: return 0
    lower_shadow = min(o, c) - l; upper_shadow = h - max(o, c)
    is_small_body = body < (candle_range * 0.35)
    is_long_upper_shadow = upper_shadow >= 1.8 * body if body > 0 else upper_shadow > candle_range * 0.6
    is_small_lower_shadow = lower_shadow <= body * 0.6 if body > 0 else lower_shadow < candle_range * 0.15
    return -100 if is_small_body and is_long_upper_shadow and is_small_lower_shadow else 0

def is_doji(row: pd.Series) -> int:
    o, h, l, c = row.get('open'), row.get('high'), row.get('low'), row.get('close')
    if pd.isna([o, h, l, c]).any(): return 0
    candle_range = h - l
    if candle_range == 0: return 0
    return 100 if abs(c - o) <= (candle_range * 0.1) else 0

def compute_engulfing(df: pd.DataFrame, idx: int) -> int:
    if idx == 0: return 0
    prev = df.iloc[idx - 1]; curr = df.iloc[idx]
    if pd.isna([prev['close'], prev['open'], curr['close'], curr['open']]).any(): return 0
    is_bullish = (prev['close'] < prev['open'] and curr['close'] > curr['open'] and curr['open'] <= prev['close'] and curr['close'] >= prev['open'])
    is_bearish = (prev['close'] > prev['open'] and curr['close'] < curr['open'] and curr['open'] >= prev['close'] and curr['close'] <= prev['open'])
    if is_bullish: return 100
    if is_bearish: return -100
    return 0

def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    logger.debug("ℹ️ [Indicators] Detecting candlestick patterns...")
    df['Hammer'] = df.apply(is_hammer, axis=1)
    df['ShootingStar'] = df.apply(is_shooting_star, axis=1)
    df['Doji'] = df.apply(is_doji, axis=1)
    engulfing_values = [compute_engulfing(df, i) for i in range(len(df))]
    df['Engulfing'] = engulfing_values
    df['BullishCandleSignal'] = df.apply(lambda row: 1 if (row['Hammer'] == 100 or row['Engulfing'] == 100) else 0, axis=1)
    df['BearishCandleSignal'] = df.apply(lambda row: 1 if (row['ShootingStar'] == -100 or row['Engulfing'] == -100) else 0, axis=1)
    logger.debug("✅ [Indicators] Candlestick patterns detected.")
    return df

# ---------------------- Other Helper Functions (Swings, Fibonacci, Volume) ----------------------
def detect_swings(prices: pd.Series, order: int = SWING_ORDER) -> List[Tuple[Any, float]]:
    """
    Detects swing points (local maxima or minima) in a pandas Series.
    Returns a list of tuples: (index, value) for each swing point found.
    Handles potential NaNs.
    """
    n = len(prices)
    # Need at least order candles before and after the center candle
    if n < 2 * order + 1:
        return []

    price_values = prices.values
    indices = prices.index
    swings = []

    # Iterate through the series, excluding the first and last 'order' candles
    for i in range(order, n - order):
        window = price_values[i - order : i + order + 1]
        center_val = price_values[i]

        # Skip if center value or window contains NaN
        if pd.isna(center_val) or np.isnan(window).any():
            continue

        # Check if the center value is the max or min in the window
        is_max = np.all(center_val >= window)
        is_min = np.all(center_val <= window)

        # To avoid duplicates on flat lines, check if the center value is unique in the window
        is_unique_max = is_max and (np.sum(np.isclose(window, center_val, atol=1e-9)) == 1) # Use atol for float comparison
        is_unique_min = is_min and (np.sum(np.isclose(window, center_val, atol=1e-9)) == 1) # Use atol for float comparison

        if is_unique_max or is_unique_min:
             swings.append((indices[i], center_val))

    # Ensure swings are sorted by index (time)
    swings.sort(key=lambda x: x[0])
    return swings


def find_relevant_swing_points(df_history: pd.DataFrame, current_index: Any) -> Tuple[Optional[Tuple[Any, float]], Optional[Tuple[Any, float]]]:
    """
    Finds the most recent significant swing low and swing high before or at the current index.
    Used for Fibonacci calculation and dynamic SL placement.
    Returns (last_swing_low, last_swing_high) as tuples (index, price).
    """
    if df_history is None or df_history.empty or len(df_history) < 2 * SWING_ORDER + 1:
        return None, None

    # Ensure data is sorted by time
    df_history = df_history.sort_index()

    # Detect swings on high and low prices within the lookback period up to the current index
    # Filter data up to and including the current index
    df_relevant = df_history.loc[df_history.index <= current_index].copy()

    if df_relevant.empty or len(df_relevant) < 2 * SWING_ORDER + 1:
         return None, None

    all_high_swings = detect_swings(df_relevant['high'], order=SWING_ORDER)
    all_low_swings = detect_swings(df_relevant['low'], order=SWING_ORDER)

    last_swing_high = max(all_high_swings, key=lambda item: item[0]) if all_high_swings else None
    last_swing_low = max(all_low_swings, key=lambda item: item[0]) if all_low_swings else None

    # Basic validation: Ensure the swing high is above the swing low
    if last_swing_high and last_swing_low and last_swing_high[1] <= last_swing_low[1]:
        logger.debug(f"ℹ️ [Swings] Swing high ({last_swing_high[1]:.4f}) not above swing low ({last_swing_low[1]:.4f}). Invalid structure for retracement.")
        # In this case, we might still return the most recent high and low, but acknowledge the structure isn't a clear uptrend wave
        # For Fibonacci retracement specifically, this structure is required.
        # For SL tracking based on lows, we still need the last low.
        # Let's return them but the calling function (generate_buy_signal) should handle the validity for Fib.
        pass # Keep the found last_swing_high and last_swing_low

    logger.debug(f"ℹ️ [Swings] Found last swing low: {last_swing_low}, last swing high: {last_swing_high}")
    return last_swing_low, last_swing_high


def calculate_fibonacci_retracements(swing_low_val: float, swing_high_val: float) -> Dict[float, float]:
    """Calculates Fibonacci retracement levels based on a swing low and high value."""
    if pd.isna(swing_low_val) or pd.isna(swing_high_val) or swing_high_val <= swing_low_val:
        logger.warning(f"⚠️ [Fib] Invalid swing points for Fibonacci: Low={swing_low_val}, High={swing_high_val}")
        return {} # Cannot calculate if high is not above low or values are NaN

    diff = swing_high_val - swing_low_val
    levels = {}
    for level in FIB_LEVELS_TO_CHECK:
        # Fibonacci retracement levels are calculated from the high back towards the low
        levels[level] = swing_high_val - (diff * level)
    logger.debug(f"ℹ️ [Fib] Calculated Fib levels from Low {swing_low_val:.4f} to High {swing_high_val:.4f}: {levels}")
    return levels

def fetch_recent_volume(symbol: str) -> float:
    """Fetches the trading volume in USDT for the last 15 minutes for the specified symbol."""
    if not client:
         logger.error(f"❌ [Data Volume] Binance client not initialized to fetch volume for {symbol}.")
         return 0.0
    try:
        logger.debug(f"ℹ️ [Data Volume] Fetching 15-minute volume for {symbol}...")
        klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=15)
        if not klines or len(klines) < 15:
             logger.warning(f"⚠️ [Data Volume] Insufficient 1m data (less than 15 candles) for {symbol}.")
             return 0.0
        volume_usdt = sum(float(k[7]) for k in klines if len(k) > 7 and k[7])
        logger.debug(f"✅ [Data Volume] Last 15 minutes liquidity for {symbol}: {volume_usdt:,.2f} USDT")
        return volume_usdt
    except (BinanceAPIException, BinanceRequestException) as binance_err:
         logger.error(f"❌ [Data Volume] Binance API or network error fetching volume for {symbol}: {binance_err}")
         return 0.0
    except Exception as e:
        logger.error(f"❌ [Data Volume] Unexpected error fetching volume for {symbol}: {e}", exc_info=True)
        return 0.0

# ---------------------- Comprehensive Performance Report Generation Function ----------------------
def generate_performance_report() -> str:
    """Generates a comprehensive performance report from the database in Arabic."""
    logger.info("ℹ️ [Report] Generating performance report...")
    if not check_db_connection() or not conn or not cur:
        return "❌ لا يمكن إنشاء التقرير، مشكلة في اتصال قاعدة البيانات."
    try:
        with conn.cursor() as report_cur: # Uses RealDictCursor
            # Count signals that are not yet fully closed (hit_stop_loss = FALSE)
            report_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE hit_stop_loss = FALSE;")
            open_signals_count = (report_cur.fetchone() or {}).get('count', 0)

            # Count signals that hit TP1 but are not yet fully closed
            report_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE tp1_hit = TRUE AND hit_stop_loss = FALSE;")
            tp1_hit_open_count = (report_cur.fetchone() or {}).get('count', 0)

            # Count signals that hit TP2 but are not yet fully closed
            report_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE tp2_hit = TRUE AND hit_stop_loss = FALSE;")
            tp2_hit_open_count = (report_cur.fetchone() or {}).get('count', 0)

             # Count signals that hit TP3 but are not yet fully closed
            report_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE tp3_hit = TRUE AND hit_stop_loss = FALSE;")
            tp3_hit_open_count = (report_cur.fetchone() or {}).get('count', 0)

            report_cur.execute("""
                SELECT
                    COUNT(*) AS total_closed,
                    COUNT(CASE WHEN profit_percentage > 0 THEN 1 END) AS winning_signals,
                    COUNT(CASE WHEN profit_percentage < 0 THEN 1 END) AS losing_signals,
                    COUNT(CASE WHEN profit_percentage = 0 THEN 1 END) AS neutral_signals,
                    COALESCE(SUM(profit_percentage), 0) AS total_profit_pct,
                    COALESCE(AVG(profit_percentage), 0) AS avg_profit_pct,
                    COALESCE(SUM(CASE WHEN profit_percentage > 0 THEN profit_percentage ELSE 0 END), 0) AS gross_profit_pct,
                    COALESCE(SUM(CASE WHEN profit_percentage < 0 THEN profit_percentage ELSE 0 END), 0) AS gross_loss_pct,
                    COALESCE(AVG(CASE WHEN profit_percentage > 0 THEN profit_percentage END), 0) AS avg_win_pct,
                    COALESCE(AVG(CASE WHEN profit_percentage < 0 THEN profit_percentage END), 0) AS avg_loss_pct,
                    COALESCE(AVG(risk_reward_ratio), 0) AS avg_rr_ratio -- Average R:R Ratio (New)
                FROM signals
                WHERE hit_stop_loss = TRUE; -- Only consider trades fully closed by SL
            """)
            closed_stats_sl = report_cur.fetchone() or {}
            total_closed_sl = closed_stats_sl.get('total_closed', 0)
            winning_signals_sl = closed_stats_sl.get('winning_signals', 0)
            losing_signals_sl = closed_stats_sl.get('losing_signals', 0)
            total_profit_pct_sl = closed_stats_sl.get('total_profit_pct', 0.0)
            gross_profit_pct_sl = closed_stats_sl.get('gross_profit_pct', 0.0)
            gross_loss_pct_sl = closed_stats_sl.get('gross_loss_pct', 0.0)
            avg_win_pct_sl = closed_stats_sl.get('avg_win_pct', 0.0)
            avg_loss_pct_sl = closed_stats_sl.get('avg_loss_pct', 0.0)
            avg_rr_ratio = closed_stats_sl.get('avg_rr_ratio', 0.0)

            # Note: The current code only marks trades as 'closed' when SL is hit.
            # If you implement full target exits later, you'll need to adjust the
            # query to include trades closed by target as well. For now, 'achieved_target'
            # is effectively replaced by tp1_hit, tp2_hit, tp3_hit which don't close the trade.

            win_rate_sl = (winning_signals_sl / total_closed_sl * 100) if total_closed_sl > 0 else 0.0
            profit_factor_sl = (gross_profit_pct_sl / abs(gross_loss_pct_sl)) if gross_loss_pct_sl != 0 else float('inf')

        report = (
            f"📊 *تقرير الأداء الشامل:*\n"
            f"——————————————\n"
            f"📈 الإشارات المفتوحة حالياً: *{open_signals_count}*\n"
            f"  • ضربت TP1: *{tp1_hit_open_count}*\n"
            f"  • ضربت TP2: *{tp2_hit_open_count}*\n"
            f"  • ضربت TP3: *{tp3_hit_open_count}*\n"
            f"——————————————\n"
            f"📉 *إحصائيات الإشارات المغلقة (بواسطة وقف الخسارة):*\n"
            f"  • إجمالي الإشارات المغلقة (SL): *{total_closed_sl}*\n"
            f"  ✅ إشارات رابحة (SL): *{winning_signals_sl}* ({win_rate_sl:.2f}%)\n"
            f"  ❌ إشارات خاسرة (SL): *{losing_signals_sl}*\n"
            f"——————————————\n"
            f"💰 *الربحية (للصفقات المغلقة بوقف الخسارة):*\n"
            f"  • صافي الربح/الخسارة (الإجمالي %): *{total_profit_pct_sl:+.2f}%*\n"
            f"  • إجمالي الربح (%): *{gross_profit_pct_sl:+.2f}%*\n"
            f"  • إجمالي الخسارة (%): *{gross_loss_pct_sl:.2f}%*\n"
            f"  • متوسط الصفقة الرابحة (%): *{avg_win_pct_sl:+.2f}%*\n"
            f"  • متوسط الصفقة الخاسرة (%): *{avg_loss_pct_sl:.2f}%*\n"
            f"  • عامل الربح: *{'∞' if profit_factor_sl == float('inf') else f'{profit_factor_sl:.2f}'}*\n"
            f"  • متوسط نسبة المخاطرة/العائد: *{avg_rr_ratio:.2f}*\n" # Display Avg R:R
            f"——————————————\n"
            f"😨/🤑 **الخوف والجشع:** {get_fear_greed_index()}\n" # Include F&G Index
            f"₿ **اتجاه البيتكوين (4H):** {get_btc_trend_4h()}\n" # Include BTC Trend
            f"——————————————\n"
            f"🕰️ _التقرير محدث حتى: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        )
        logger.info("✅ [Report] Performance report generated successfully.")
        return report
    except psycopg2.Error as db_err:
        logger.error(f"❌ [Report] Database error generating performance report: {db_err}")
        if conn: conn.rollback()
        return "❌ خطأ في قاعدة البيانات أثناء إنشاء تقرير الأداء."
    except Exception as e:
        logger.error(f"❌ [Report] Unexpected error generating performance report: {e}", exc_info=True)
        return "❌ حدث خطأ غير متوقع أثناء إنشاء تقرير الأداء."

# ---------------------- Trading Strategy (Fibonacci Entry Check) -------------------
class ConservativeTradingStrategy:
    """Encapsulates the trading strategy logic and associated indicators with a scoring system and mandatory conditions."""
    def __init__(self, symbol: str):
        self.symbol = symbol
        # Added tp2_price, tp3_price to required_cols_indicators if needed for scoring/validation
        self.required_cols_indicators = [
            'open', 'high', 'low', 'close', 'volume', 'ema_13', 'ema_34', 'vwma',
            'rsi', 'atr', 'bb_upper', 'bb_lower', 'bb_middle', 'macd', 'macd_signal', 'macd_hist',
            'adx', 'di_plus', 'di_minus', 'vwap', 'obv', 'supertrend', 'supertrend_trend',
            'BullishCandleSignal', 'BearishCandleSignal'
        ]
        self.required_cols_buy_signal = [
            'high', 'low', 'close', 'ema_13', 'ema_34', 'vwma', 'rsi', 'atr',
            'macd', 'macd_signal', 'macd_hist', 'supertrend_trend', 'adx', 'di_plus', 'di_minus',
            'vwap', 'bb_upper', 'BullishCandleSignal', 'obv'
        ]
        # Added weight for R:R ratio check (although it's a filter, can give a score boost)
        self.condition_weights = {
            'rsi_ok': 0.5, 'bullish_candle': 1.0, 'not_bb_extreme': 0.5, 'obv_rising': 1.5,
            'rsi_filter_breakout': 1.5, 'macd_filter_breakout': 1.5, 'near_fib_level': 2.0,
            # 'good_rr_ratio': 1.0 # R:R is now a mandatory filter, not a score component
        }
        self.essential_conditions = [
            'ema_cross_bullish', 'supertrend_up', 'macd_positive_or_cross',
            'adx_trending_bullish', 'breakout_bb_upper', 'above_vwma',
            'valid_rr_ratio' # Added R:R as an essential condition
        ]
        self.total_possible_score = sum(self.condition_weights.values())
        self.min_score_threshold_pct = 0.50
        self.min_signal_score = self.total_possible_score * self.min_score_threshold_pct

    def populate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculates all required indicators for the strategy."""
        logger.debug(f"ℹ️ [Strategy {self.symbol}] Calculating indicators...")
        # Ensure enough data for all indicators, including swings for Fib
        min_len_required = max(EMA_SHORT_PERIOD, EMA_LONG_PERIOD, VWMA_PERIOD, RSI_PERIOD, ENTRY_ATR_PERIOD, BOLLINGER_WINDOW, MACD_SLOW, ADX_PERIOD*2, SUPERTREND_PERIOD, LOOKBACK_FOR_SWINGS, SWING_ORDER*2) + 5
        if len(df) < min_len_required:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame too short ({len(df)} < {min_len_required}) to calculate indicators.")
            return None
        try:
            df_calc = df.copy()
            # Calculate ATR first as it might be needed by other indicators (like SuperTrend)
            df_calc = calculate_atr_indicator(df_calc, ENTRY_ATR_PERIOD) # Use ENTRY_ATR_PERIOD here
            # SuperTrend uses its own period for ATR calculation internally now
            df_calc = calculate_supertrend(df_calc, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
            df_calc['ema_13'] = calculate_ema(df_calc['close'], EMA_SHORT_PERIOD)
            df_calc['ema_34'] = calculate_ema(df_calc['close'], EMA_LONG_PERIOD)
            df_calc['vwma'] = calculate_vwma(df_calc, VWMA_PERIOD)
            df_calc = calculate_rsi_indicator(df_calc, RSI_PERIOD)
            df_calc = calculate_bollinger_bands(df_calc, BOLLINGER_WINDOW, BOLLINGER_STD_DEV)
            df_calc = calculate_macd(df_calc, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            adx_df = calculate_adx(df_calc, ADX_PERIOD)
            df_calc = df_calc.join(adx_df)
            df_calc = calculate_vwap(df_calc)
            df_calc = calculate_obv(df_calc)
            df_calc = detect_candlestick_patterns(df_calc)

            missing_cols = [col for col in self.required_cols_indicators if col not in df_calc.columns]
            if missing_cols:
                 logger.error(f"❌ [Strategy {self.symbol}] Required indicator columns missing after calculation: {missing_cols}")
                 return None

            initial_len = len(df_calc)
            df_cleaned = df_calc.dropna(subset=self.required_cols_indicators).copy()
            dropped_count = initial_len - len(df_cleaned)
            if dropped_count > 0: logger.debug(f"ℹ️ [Strategy {self.symbol}] Dropped {dropped_count} rows due to NaN in indicators.")
            if df_cleaned.empty: logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame is empty after removing indicator NaNs."); return None

            latest = df_cleaned.iloc[-1]
            logger.debug(f"✅ [Strategy {self.symbol}] Indicators calculated. Latest EMA13: {latest.get('ema_13', np.nan):.4f}, VWMA: {latest.get('vwma', np.nan):.4f}")
            return df_cleaned
        except KeyError as ke: logger.error(f"❌ [Strategy {self.symbol}] Error: Required column not found: {ke}", exc_info=True); return None
        except Exception as e: logger.error(f"❌ [Strategy {self.symbol}] Unexpected error during indicator calculation: {e}", exc_info=True); return None

    def generate_buy_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generates a buy signal based on conditions, Fibonacci, and scoring."""
        logger.debug(f"ℹ️ [Strategy {self.symbol}] Generating buy signal...")
        # Ensure enough data for indicators and swing detection
        min_len_required = max(len(self.required_cols_buy_signal), LOOKBACK_FOR_SWINGS + SWING_ORDER * 2) + 5
        if df_processed is None or df_processed.empty or len(df_processed) < min_len_required:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame too short ({len(df_processed)} < {min_len_required}) for signal generation.")
            return None

        required_cols_full = list(set(self.required_cols_buy_signal + ['bb_upper', 'bb_lower', 'rsi', 'macd_hist', 'vwma', 'atr'])) # Ensure ATR is included
        missing_cols = [col for col in required_cols_full if col not in df_processed.columns]
        if missing_cols: logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame missing required columns: {missing_cols}."); return None

        btc_trend = get_btc_trend_4h()
        if "هبوط" in btc_trend: logger.info(f"ℹ️ [Strategy {self.symbol}] Trading paused due to bearish Bitcoin trend ({btc_trend})."); return None
        elif "N/A" in btc_trend: logger.warning(f"⚠️ [Strategy {self.symbol}] Cannot determine Bitcoin trend, ignoring condition.")

        last_row = df_processed.iloc[-1]
        prev_row = df_processed.iloc[-2] if len(df_processed) >= 2 else None
        last_row_check = last_row[required_cols_full]
        if last_row_check.isnull().any(): logger.warning(f"⚠️ [Strategy {self.symbol}] Last row contains NaN in required signal columns: {last_row_check[last_row_check.isnull()].index.tolist()}."); return None
        if prev_row is None or pd.isna(prev_row['obv']): logger.warning(f"⚠️ [Strategy {self.symbol}] Previous OBV value is NaN or previous row missing.")

        current_price = last_row['close']
        current_atr = last_row.get('atr')
        if pd.isna(current_atr) or current_atr <= 0: logger.warning(f"⚠️ [Strategy {self.symbol}] Invalid ATR ({current_atr}) for SL/TP calc."); return None

        # --- Check Mandatory Conditions ---
        essential_passed = True
        failed_essential_conditions = []
        signal_details = {}

        # EMA Cross Bullish
        if not (last_row['ema_13'] > last_row['ema_34']): essential_passed = False; failed_essential_conditions.append('EMA Cross'); signal_details['EMA_Cross'] = 'Failed'
        else: signal_details['EMA_Cross'] = 'Passed'

        # SuperTrend Up
        if not (pd.notna(last_row['supertrend']) and last_row['close'] > last_row['supertrend'] and last_row['supertrend_trend'] == 1): essential_passed = False; failed_essential_conditions.append('SuperTrend'); signal_details['SuperTrend'] = 'Failed'
        else: signal_details['SuperTrend'] = 'Passed'

        # MACD Positive or Bullish Cross
        if not (last_row['macd_hist'] > 0 or (prev_row is not None and pd.notna(prev_row['macd']) and pd.notna(prev_row['macd_signal']) and last_row['macd'] > last_row['macd_signal'] and prev_row['macd'] <= prev_row['macd_signal'])): essential_passed = False; failed_essential_conditions.append('MACD'); signal_details['MACD'] = 'Failed'
        else: signal_details['MACD'] = 'Passed'

        # ADX Trending Bullish (ADX > 20 and DI+ > DI-)
        if not (last_row['adx'] > 20 and last_row['di_plus'] > last_row['di_minus']): essential_passed = False; failed_essential_conditions.append('ADX/DI'); signal_details['ADX/DI'] = 'Passed'
        else: signal_details['ADX/DI'] = 'Failed'

        # Breakout above Bollinger Upper Band
        if not (pd.notna(last_row['bb_upper']) and last_row['close'] > last_row['bb_upper']): essential_passed = False; failed_essential_conditions.append('Breakout BB'); signal_details['Breakout_BB'] = 'Passed'
        else: signal_details['Breakout_BB'] = 'Failed'

        # Above VWMA
        if not (pd.notna(last_row['vwma']) and last_row['close'] > last_row['vwma']): essential_passed = False; failed_essential_conditions.append('Above VWMA'); signal_details['VWMA_Mandatory'] = 'Failed'
        else: signal_details['VWMA_Mandatory'] = 'Passed'

        # --- Determine Initial Stop Loss (Based on Market Structure or ATR) ---
        # Find the most recent swing low BEFORE the current candle close
        # We use data up to the second to last candle for swing detection to avoid lookahead bias
        lookback_data_for_sl = df_processed.iloc[:-1].tail(LOOKBACK_FOR_SWINGS + SWING_ORDER * 2) # Data before current candle
        last_swing_low_for_sl, _ = find_relevant_swing_points(lookback_data_for_sl, df_processed.index[-2]) # Swings before the last candle

        atr_based_stop_loss = current_price - (ENTRY_ATR_MULTIPLIER * current_atr)
        swing_based_stop_loss = None

        if last_swing_low_for_sl is not None:
            swing_based_stop_loss = last_swing_low_for_sl[1] * (1 - SWING_SL_BUFFER_PCT)
            signal_details['Initial_SL_Basis'] = f'Swing Low ({last_swing_low_for_sl[1]:.8g})'
            # Choose the lower (safer) stop loss between ATR and Swing
            initial_stop_loss = min(atr_based_stop_loss, swing_based_stop_loss)
            signal_details['Initial_SL_Calc'] = f'Min(ATR: {atr_based_stop_loss:.8g}, Swing: {swing_based_stop_loss:.8g})'
            logger.debug(f"ℹ️ [Strategy {self.symbol}] Swing low found for initial SL: {last_swing_low_for_sl[1]:.8g}. ATR SL: {atr_based_stop_loss:.8g}. Chosen SL: {initial_stop_loss:.8g}")
        else:
            # If no valid swing low found, default to ATR based SL
            initial_stop_loss = atr_based_stop_loss
            signal_details['Initial_SL_Basis'] = 'ATR'
            signal_details['Initial_SL_Calc'] = f'ATR: {atr_based_stop_loss:.8g}'
            logger.warning(f"⚠️ [Strategy {self.symbol}] No relevant swing low found for initial SL. Using ATR based SL: {initial_stop_loss:.8g}")


        # --- Calculate Take Profit Levels (Adjusted Multipliers) ---
        tp1_price_calc = current_price + (TP1_ATR_MULTIPLIER * current_atr)
        tp2_price_calc = current_price + (TP2_ATR_MULTIPLIER * current_atr) # New TP2
        tp3_price_calc = current_price + (TP3_ATR_MULTIPLIER * current_atr) # New TP3
        # initial_target_atr = current_price + (ENTRY_ATR_MULTIPLIER * current_atr) # Keep original ATR target for reference/display - Removed as TP1 is now the primary target


        # --- SL/TP Validation & R:R Check ---
        # Ensure SL is not zero or above entry
        if initial_stop_loss <= 0 or initial_stop_loss >= current_price:
             logger.warning(f"⚠️ [Strategy {self.symbol}] Calculated initial SL ({initial_stop_loss:.8g}) is invalid. Price: {current_price:.8g}. Signal rejected.")
             signal_details['Warning_SL'] = 'Invalid calculated SL'
             # essential_passed = False; failed_essential_conditions.append('Valid SL Price') # Make this mandatory? Or just reject? Reject seems safer.
             return None # Reject signal if initial SL is invalid

        # Check max loss percentage
        max_allowed_loss_pct = 0.10 # Example: 10% max loss
        max_sl_price = current_price * (1 - max_allowed_loss_pct)
        if initial_stop_loss < max_sl_price:
             logger.warning(f"⚠️ [Strategy {self.symbol}] Initial SL ({initial_stop_loss:.8g}) is too wide (>{max_allowed_loss_pct*100}% loss). Signal rejected.")
             signal_details['Warning_SL'] = f'Initial SL too wide (> {max_allowed_loss_pct*100}%)'
             return None # Reject signal if SL is too wide

        # Check minimum profit margin for TP1 (using the adjusted TP1 multiplier)
        # Now using tp1_price_calc for this check
        profit_margin_pct = ((tp1_price_calc / current_price) - 1) * 100 if current_price > 0 else 0
        if profit_margin_pct < MIN_PROFIT_MARGIN_PCT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] Initial Profit margin ({profit_margin_pct:.2f}%) below minimum ({MIN_PROFIT_MARGIN_PCT}%). Signal rejected.")
            signal_details['Warning_ProfitMargin'] = f'Profit margin ({profit_margin_pct:.2f}%) below min ({MIN_PROFIT_MARGIN_PCT}%)'
            return None # Reject signal if initial profit margin is too small

        # Calculate Risk:Reward Ratio
        risk = current_price - initial_stop_loss
        # Use TP1 for R:R calculation as it's the first significant target
        reward = tp1_price_calc - current_price
        rr_ratio = reward / risk if risk > 0 else 0.0

        if rr_ratio < MIN_RR_RATIO:
            logger.info(f"ℹ️ [Strategy {self.symbol}] Risk:Reward Ratio ({rr_ratio:.2f}) below minimum ({MIN_RR_RATIO}). Signal rejected.")
            signal_details['Warning_RRRatio'] = f'R:R ({rr_ratio:.2f}) below min ({MIN_RR_RATIO})'
            essential_passed = False; failed_essential_conditions.append('Valid R:R Ratio') # Make this a mandatory filter
        else:
             signal_details['Risk_Reward_Ratio'] = float(f"{rr_ratio:.2f}")
             signal_details['Valid_RRRatio'] = 'Passed'


        if not essential_passed: logger.debug(f"ℹ️ [Strategy {self.symbol}] Mandatory conditions failed: {', '.join(failed_essential_conditions)}. Signal rejected."); return None

        # --- Calculate Score for Optional Conditions ---
        current_score = 0.0

        # Fibonacci Check (Now a scoring condition, not mandatory)
        fib_level_found = None
        try:
            # Use lookback data up to the current candle for Fib check
            lookback_data_for_fib = df_processed.tail(LOOKBACK_FOR_SWINGS + SWING_ORDER * 2)
            # Find swings on the full lookback data
            fib_swing_low, fib_swing_high = find_relevant_swing_points(lookback_data_for_fib, last_row.name) # Swings up to last candle

            if fib_swing_low and fib_swing_high and fib_swing_low[0] < fib_swing_high[0] and fib_swing_high[1] > fib_swing_low[1]: # Ensure valid swing structure for Fib
                 fib_levels = calculate_fibonacci_retracements(fib_swing_low[1], fib_swing_high[1])
                 current_price_fib = last_row['close']
                 for level_pct, level_price in fib_levels.items():
                     if level_price * (1 - FIB_TOLERANCE) <= current_price_fib <= level_price * (1 + FIB_TOLERANCE):
                         fib_level_found = level_pct
                         current_score += self.condition_weights.get('near_fib_level', 0)
                         signal_details['Fibonacci'] = f'Near {fib_level_found*100:.1f}% ({level_price:.4f}) (+{self.condition_weights.get("near_fib_level", 0)})'
                         logger.debug(f"ℹ️ [Strategy {self.symbol}] Price {current_price_fib:.4f} near Fib {fib_level_found*100:.1f}% level ({level_price:.4f}).")
                         break
                 if not fib_level_found: signal_details['Fibonacci'] = 'Not Near Key Levels (0)'
            else: signal_details['Fibonacci'] = 'No Valid Swing Found (0)'
        except Exception as fib_err: logger.error(f"❌ [Strategy {self.symbol}] Error during Fibonacci calc: {fib_err}", exc_info=True); signal_details['Fibonacci'] = 'Error (0)'


        # Other Optional Conditions
        # RSI not overbought/oversold (between 30 and 70)
        if last_row['rsi'] < RSI_OVERBOUGHT and last_row['rsi'] > RSI_OVERSOLD: current_score += self.condition_weights.get('rsi_ok', 0); signal_details['RSI_Basic'] = f'OK (+{self.condition_weights.get("rsi_ok", 0)})'
        else: signal_details['RSI_Basic'] = 'Not OK (0)'

        # Bullish Candle Pattern (Hammer or Bullish Engulfing)
        if last_row['BullishCandleSignal'] == 1: current_score += self.condition_weights.get('bullish_candle', 0); signal_details['Candle'] = f'Bullish (+{self.condition_weights.get("bullish_candle", 0)})'
        else: signal_details['Candle'] = 'Neutral (0)'

        # Price not at extreme of Bollinger Bands (e.g., not touching upper band)
        if pd.notna(last_row['bb_upper']) and last_row['close'] < last_row['bb_upper'] * 0.995: current_score += self.condition_weights.get('not_bb_extreme', 0); signal_details['Bollinger_Basic'] = f'Not Extreme (+{self.condition_weights.get("not_bb_extreme", 0)})'
        else: signal_details['Bollinger_Basic'] = 'Extreme (0)'

        # OBV Rising (compared to previous candle)
        if prev_row is not None and pd.notna(prev_row['obv']) and pd.notna(last_row['obv']) and last_row['obv'] > prev_row['obv']: current_score += self.condition_weights.get('obv_rising', 0); signal_details['OBV'] = f'Rising (+{self.condition_weights.get("obv_rising", 0)})'
        else: signal_details['OBV'] = 'Not Rising (0)'

        # RSI Filter for Breakout (e.g., RSI between 55 and 75)
        if pd.notna(last_row['rsi']) and 55 <= last_row['rsi'] <= 75: current_score += self.condition_weights.get('rsi_filter_breakout', 0); signal_details['RSI_Filter_Breakout'] = f'OK (+{self.condition_weights.get("rsi_filter_breakout", 0)})'
        else: signal_details['RSI_Filter_Breakout'] = 'Not OK (0)'

        # MACD Histogram Positive (already part of mandatory, but can add score for strength)
        if pd.notna(last_row['macd_hist']) and last_row['macd_hist'] > 0: current_score += self.condition_weights.get('macd_filter_breakout', 0); signal_details['MACD_Filter_Breakout'] = f'Positive (+{self.condition_weights.get("macd_filter_breakout", 0)})'
        else: signal_details['MACD_Filter_Breakout'] = 'Negative (0)'


        # Final Score Check
        if current_score < self.min_signal_score: logger.debug(f"ℹ️ [Strategy {self.symbol}] Optional score not met (Score: {current_score:.2f}/{self.total_possible_score:.2f}). Signal rejected."); return None

        # --- Final Liquidity Check ---
        volume_recent = fetch_recent_volume(self.symbol)
        if volume_recent < MIN_VOLUME_15M_USDT: logger.info(f"ℹ️ [Strategy {self.symbol}] Liquidity ({volume_recent:,.0f} USDT) below threshold ({MIN_VOLUME_15M_USDT:,.0f} USDT). Signal rejected."); return None


        # --- Compile Signal ---
        signal_output = {
            'symbol': self.symbol,
            'entry_price': float(f"{current_price:.8g}"),
            'initial_target': float(f"{tp1_price_calc:.8g}"), # Use TP1 as the initial target for display/reference
            'initial_stop_loss': float(f"{initial_stop_loss:.8g}"), # This is the chosen initial SL (Min of ATR/Swing)
            'current_target': float(f"{tp1_price_calc:.8g}"), # Keep initial target for now, dynamic TPs handled in tracking
            'current_stop_loss': float(f"{initial_stop_loss:.8g}"), # Initial current SL is the chosen initial SL
            'r2_score': float(f"{current_score:.2f}"),
            'strategy_name': 'Breakout_Fib_VWMA_Dynamic', # Updated name
            'signal_details': signal_details,
            'volume_15m': volume_recent,
            'trade_value': TRADE_VALUE,
            'total_possible_score': float(f"{self.total_possible_score:.2f}"),
            'tp1_price': float(f"{tp1_price_calc:.8g}"), # Store calculated TP1
            'tp2_price': float(f"{tp2_price_calc:.8g}"), # Store calculated TP2 (New)
            'tp3_price': float(f"{tp3_price_calc:.8g}"), # Store calculated TP3 (New)
            'initial_atr': float(f"{current_atr:.8g}"), # Store initial ATR
            'risk_reward_ratio': float(f"{rr_ratio:.2f}") # Store R:R Ratio (New)
        }
        logger.info(f"✅ [Strategy {self.symbol}] Confirmed buy signal. Price: {current_price:.6f}, Score: {current_score:.2f}/{self.total_possible_score:.2f}, ATR: {current_atr:.6f}, TP1: {tp1_price_calc:.6f}, TP2: {tp2_price_calc:.6f}, TP3: {tp3_price_calc:.6f}, R:R: {rr_ratio:.2f}")
        return signal_output


# ---------------------- Telegram Functions ----------------------
def send_telegram_message(target_chat_id: str, text: str, reply_markup: Optional[Dict] = None, parse_mode: str = 'Markdown', disable_web_page_preview: bool = True, timeout: int = 20) -> Optional[Dict]:
    """Sends a message via Telegram Bot API with improved error handling."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(target_chat_id), 'text': text, 'parse_mode': parse_mode, 'disable_web_page_preview': disable_web_page_preview}
    if reply_markup:
        try: payload['reply_markup'] = json.dumps(convert_np_values(reply_markup))
        except (TypeError, ValueError) as json_err: logger.error(f"❌ [Telegram] Failed to convert reply_markup: {json_err}"); return None
    logger.debug(f"ℹ️ [Telegram] Sending message to {target_chat_id}...")
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        logger.info(f"✅ [Telegram] Message sent successfully to {target_chat_id}.")
        return response.json()
    except requests.exceptions.Timeout: logger.error(f"❌ [Telegram] Timeout sending to {target_chat_id}."); return None
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"❌ [Telegram] HTTP Error {http_err.response.status_code} sending to {target_chat_id}.")
        try: logger.error(f"❌ [Telegram] API error details: {http_err.response.json()}")
        except json.JSONDecodeError: logger.error(f"❌ [Telegram] Could not decode error response: {http_err.response.text}")
        return None
    except requests.exceptions.RequestException as req_err: logger.error(f"❌ [Telegram] Request Error sending to {target_chat_id}: {req_err}"); return None
    except Exception as e: logger.error(f"❌ [Telegram] Unexpected error sending message: {e}", exc_info=True); return None

def send_telegram_alert(signal_data: Dict[str, Any], timeframe: str) -> None:
    """Formats and sends a new trading signal alert to Telegram in Arabic."""
    logger.debug(f"ℹ️ [Telegram Alert] Formatting alert for signal: {signal_data.get('symbol', 'N/A')}")
    try:
        entry_price = float(signal_data['entry_price'])
        # Use tp1_price for the initial target display as it's now the primary first target
        initial_target_price_display = float(signal_data['tp1_price'])
        stop_loss_price = float(signal_data['initial_stop_loss'])
        tp1_price = signal_data.get('tp1_price', 0.0)
        tp2_price = signal_data.get('tp2_price', 0.0) # New
        tp3_price = signal_data.get('tp3_price', 0.0) # New
        symbol = signal_data['symbol']
        strategy_name = signal_data.get('strategy_name', 'N/A')
        signal_score = signal_data.get('r2_score', 0.0)
        total_possible_score = signal_data.get('total_possible_score', 10.0)
        volume_15m = signal_data.get('volume_15m', 0.0)
        trade_value_signal = signal_data.get('trade_value', TRADE_VALUE)
        rr_ratio = signal_data.get('risk_reward_ratio', 0.0) # New
        signal_details = signal_data.get('signal_details', {})

        # Calculate percentages relative to entry price
        # Calculate profit percentage for the displayed initial target (TP1)
        profit_pct_initial_target_display = ((initial_target_price_display / entry_price) - 1) * 100 if entry_price > 0 else 0
        loss_pct_sl = ((stop_loss_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        profit_pct_tp1 = ((tp1_price / entry_price) - 1) * 100 if entry_price > 0 and tp1_price else 0
        profit_pct_tp2 = ((tp2_price / entry_price) - 1) * 100 if entry_price > 0 and tp2_price else 0 # New
        profit_pct_tp3 = ((tp3_price / entry_price) - 1) * 100 if entry_price > 0 and tp3_price else 0 # New


        loss_usdt = abs(trade_value_signal * (loss_pct_sl / 100))
        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')
        fear_greed = get_fear_greed_index()
        btc_trend = get_btc_trend_4h()

        message = (
            f"💡 *إشارة تداول جديدة ({strategy_name.replace('_', ' ').title()})* 💡\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"📈 **نوع الإشارة:** شراء (طويل)\n"
            f"🕰️ **الإطار الزمني:** {timeframe}\n"
            f"📊 **قوة الإشارة (اختياري):** *{signal_score:.1f} / {total_possible_score:.1f}*\n"
            f"💧 **السيولة (15 د):** {volume_15m:,.0f} USDT\n"
            f"⚖️ **المخاطرة/العائد (R:R):** *{rr_ratio:.2f}*\n" # Display R:R
            f"——————————————\n"
            f"➡️ **سعر الدخول المقترح:** `${entry_price:,.8g}`\n"
            f"🛡️ **وقف الخسارة الأولي:** `${stop_loss_price:,.8g}` ({loss_pct_sl:.2f}% / ≈ ${loss_usdt:.2f})\n"
            # Display TP1 as the primary target with its calculated price and percentage
            f"🎯 **الهدف الأول (للتعادل):** `${tp1_price:,.8g}` ({profit_pct_tp1:+.2f}%)\n"
            f"🎯 **الهدف الثاني (جزئي):** `${tp2_price:,.8g}` ({profit_pct_tp2:+.2f}%)\n" # Show TP2 (New)
            f"🎯 **الهدف الثالث (جزئي):** `${tp3_price:,.8g}` ({profit_pct_tp3:+.2f}%)\n" # Show TP3 (New)
            f"——————————————\n"
            f"✅ *الشروط الإلزامية:*\n"
            f"  - عبور EMA: {'✅' if signal_details.get('EMA_Cross') == 'Passed' else '❌'}\n"
            f"  - سوبر ترند صاعد: {'✅' if signal_details.get('SuperTrend') == 'Passed' else '❌'}\n"
            f"  - MACD إيجابي/تقاطع: {'✅' if signal_details.get('MACD') == 'Passed' else '❌'}\n"
            f"  - ADX/DI اتجاه صاعد: {'✅' if signal_details.get('ADX/DI') == 'Passed' else '❌'}\n"
            f"  - اختراق البولينجر العلوي: {'✅' if signal_details.get('Breakout_BB') == 'Passed' else '❌'}\n"
            f"  - فوق VWMA: {'✅' if signal_details.get('VWMA_Mandatory') == 'Passed' else '❌'}\n"
            f"  - نسبة مخاطرة/عائد مقبولة: {'✅' if signal_details.get('Valid_RRRatio') == 'Passed' else '❌'}\n"
            f"➕ *الشروط الاختيارية المحققة:*\n"
            f"  - فيبوناتشي: {signal_details.get('Fibonacci', 'N/A')}\n"
            f"  - RSI: {signal_details.get('RSI_Basic', 'N/A')}\n"
            f"  - شمعة: {signal_details.get('Candle', 'N/A')}\n"
            f"  - OBV: {signal_details.get('OBV', 'N/A')}\n"
            f"  - RSI فلتر: {signal_details.get('RSI_Filter_Breakout', 'N/A')}\n"
            f"  - MACD فلتر: {signal_details.get('MACD_Filter_Breakout', 'N/A')}\n"
            f"——————————————\n"
            f"😨/🤑 **الخوف والجشع:** {fear_greed}\n"
            f"₿ **اتجاه البيتكوين (4H):** {btc_trend}\n"
            f"——————————————\n"
            f"⏰ {timestamp_str}"
        )
        reply_markup = {"inline_keyboard": [[{"text": "📊 عرض تقرير الأداء", "callback_data": "get_report"}]]}
        send_telegram_message(CHAT_ID, message, reply_markup=reply_markup, parse_mode='Markdown')
    except KeyError as ke: logger.error(f"❌ [Telegram Alert] Signal data incomplete for {signal_data.get('symbol', 'N/A')}: Missing key {ke}", exc_info=True)
    except Exception as e: logger.error(f"❌ [Telegram Alert] Failed to send signal alert for {signal_data.get('symbol', 'N/A')}: {e}", exc_info=True)

def send_tracking_notification(details: Dict[str, Any]) -> None:
    """Formats and sends enhanced Telegram notifications for tracking events in Arabic."""
    symbol = details.get('symbol', 'N/A')
    signal_id = details.get('id', 'N/A')
    notification_type = details.get('type', 'unknown')
    message = ""
    safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')
    closing_price = details.get('closing_price') # Can be None if not closed
    profit_pct = details.get('profit_pct') # Can be None if not closed
    current_price = details.get('current_price')
    new_stop_loss = details.get('new_stop_loss')
    old_stop_loss = details.get('old_stop_loss')
    entry_price = details.get('entry_price')
    swing_price = details.get('swing_price') # Price of the swing high/low used
    target_price = details.get('target_price') # Price of the TP hit (TP1, TP2, or TP3)

    logger.debug(f"ℹ️ [Notification] Formatting tracking notification: ID={signal_id}, Type={notification_type}, Symbol={symbol}")

    if notification_type == 'stop_loss_hit':
        sl_type_msg_ar = "بربح ✅" if details.get('profitable_sl', False) else "بخسارة ❌"
        message = (
            f"🛑 *تم ضرب وقف الخسارة (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"🚫 **سعر الإغلاق (الوقف):** `${closing_price:,.8g}`\n"
            f"📉 **النتيجة:** {profit_pct:.2f}% ({sl_type_msg_ar})"
        )
    elif notification_type == 'tp1_hit_breakeven':
        message = (
            f"🛡️ *تم الوصول للهدف الأول ونقل الوقف للتعادل (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"📈 **السعر الحالي (عند الهدف 1):** `${current_price:,.8g}`\n"
            f"🎯 **سعر الهدف الأول:** `${target_price:,.8g}`\n" # Use target_price
            f"➡️ **وقف الخسارة الجديد:** `${new_stop_loss:,.8g}` (نقطة الدخول)" # Use new_stop_loss (which is entry)
        )
    elif notification_type == 'tp_hit': # Generic for TP2, TP3
        target_level = details.get('target_level', 'هدف')
        message = (
            f"🎯 *تم الوصول إلى {target_level} (ID: {signal_id})*\n" # e.g., الهدف الثاني
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"📈 **السعر الحالي:** `${current_price:,.8g}`\n"
            f"🎯 **سعر {target_level}:** `${target_price:,.8g}`" # Use target_price
        )
    elif notification_type == 'trailing_activated_swing':
        message = (
            f"⬆️ *تم تفعيل الوقف المتحرك (كسر قمة) (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"📈 **السعر الحالي (عند الكسر):** `${current_price:,.8g}`\n"
            f"⛰️ **القمة المكسورة:** `${swing_price:,.8g}`\n"
            f"🛡️ **وقف الخسارة المبدئي (تحت القاع):** `${new_stop_loss:,.8g}`"
        )
    elif notification_type == 'trailing_updated_swing':
        message = (
            f"➡️ *تم تحديث الوقف المتحرك (قاع أعلى) (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"📈 **السعر الحالي:** `${current_price:,.8g}`\n"
            f"⚓ **القاع الجديد:** `${swing_price:,.8g}`\n"
            f"🔒 **الوقف السابق:** `${old_stop_loss:,.8g}`\n"
            f"🛡️ **وقف الخسارة الجديد:** `${new_stop_loss:,.8g}`"
        )
    else:
        logger.warning(f"⚠️ [Notification] Unknown notification type: {notification_type} for details: {details}")
        return

    if message:
        send_telegram_message(CHAT_ID, message, parse_mode='Markdown')


# ---------------------- Database Functions (Insert and Update) ----------------------
def insert_signal_into_db(signal: Dict[str, Any]) -> bool:
    """Inserts a new signal into the signals table, including TP1, TP2, TP3, initial ATR, and R:R."""
    if not check_db_connection() or not conn:
        logger.error(f"❌ [DB Insert] Failed to insert signal {signal.get('symbol', 'N/A')} due to DB connection issue.")
        return False

    symbol = signal.get('symbol', 'N/A')
    logger.debug(f"ℹ️ [DB Insert] Attempting to insert signal for {symbol}...")
    try:
        signal_prepared = convert_np_values(signal)
        signal_details_json = json.dumps(signal_prepared.get('signal_details', {}))

        with conn.cursor() as cur_ins:
            # Ensure all columns match the (potentially manually updated) table schema
            insert_query = sql.SQL("""
                INSERT INTO signals
                 (symbol, entry_price, initial_target, initial_stop_loss, current_target, current_stop_loss,
                 r2_score, strategy_name, signal_details, volume_15m,
                 tp1_price, tp2_price, tp3_price, initial_atr, risk_reward_ratio) -- Added new columns
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """)
            cur_ins.execute(insert_query, (
                signal_prepared['symbol'],
                signal_prepared['entry_price'],
                # Use tp1_price as initial_target in DB for consistency with display/checks
                signal_prepared['tp1_price'],
                signal_prepared['initial_stop_loss'],
                signal_prepared['tp1_price'], # Keep initial target for now, dynamic TPs handled in tracking
                signal_prepared['current_stop_loss'],
                signal_prepared.get('r2_score'),
                signal_prepared.get('strategy_name', 'unknown'),
                signal_details_json,
                signal_prepared.get('volume_15m'),
                signal_prepared.get('tp1_price'), # Insert TP1 price
                signal_prepared.get('tp2_price'), # Insert TP2 price (New)
                signal_prepared.get('tp3_price'), # Insert TP3 price (New)
                signal_prepared.get('initial_atr'), # Insert initial ATR
                signal_prepared.get('risk_reward_ratio') # Insert R:R Ratio (New)
            ))
        conn.commit()
        logger.info(f"✅ [DB Insert] Signal for {symbol} inserted (Score: {signal_prepared.get('r2_score')}, TP1: {signal_prepared.get('tp1_price')}, TP2: {signal_prepared.get('tp2_price')}, TP3: {signal_prepared.get('tp3_price')}, R:R: {signal_prepared.get('risk_reward_ratio')}).")
        return True
    except psycopg2.Error as db_err:
        logger.error(f"❌ [DB Insert] Database error inserting signal for {symbol}: {db_err}")
        if conn: conn.rollback()
        return False
    except (TypeError, ValueError) as convert_err:
         logger.error(f"❌ [DB Insert] Error converting signal data for {symbol}: {convert_err} - Data: {signal}")
         if conn: conn.rollback()
         return False
    except Exception as e:
        logger.error(f"❌ [DB Insert] Unexpected error inserting signal for {symbol}: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

# ---------------------- Open Signal Tracking Function (Dynamic Logic) ----------------------
def track_signals() -> None:
    """
    Tracks open signals with dynamic logic:
    - Checks for Stop Loss Hit.
    - Checks for TP1, TP2, TP3 Hits and sends notifications.
    - Moves SL to Break-Even when TP1 is hit (if BE > current SL).
    - Activates trailing stop based on breaking a recent swing high (above entry).
    - Updates trailing stop based on newly formed swing lows (always keeping SL at or above BE if BE is set).
    """
    logger.info("ℹ️ [Tracker] Starting open signal tracking process (Dynamic)...")
    while True:
        active_signals_summary: List[str] = []
        processed_in_cycle = 0
        try:
            if not check_db_connection() or not conn:
                logger.warning("⚠️ [Tracker] Skipping tracking cycle due to DB connection issue.")
                time.sleep(TRACKING_CYCLE_SLEEP_SECONDS)
                continue

            # Fetch open signals with necessary columns for dynamic tracking
            with conn.cursor() as track_cur:
                 # Ensure all needed columns (including new ones) are selected
                 track_cur.execute("""
                    SELECT id, symbol, entry_price, initial_stop_loss, current_target, current_stop_loss,
                           is_trailing_active, last_swing_low_price, last_swing_high_price,
                           tp1_price, tp1_hit, tp2_price, tp2_hit, tp3_price, tp3_hit,
                           stop_loss_at_breakeven
                    FROM signals
                    WHERE hit_stop_loss = FALSE; -- Only track signals that haven't hit the final SL
                """)
                 open_signals: List[Dict] = track_cur.fetchall()


            if not open_signals:
                # logger.debug("ℹ️ [Tracker] No open signals to track.")
                time.sleep(TRACKING_CYCLE_SLEEP_SECONDS // 2) # Wait less if no signals
                continue

            logger.debug(f"ℹ️ [Tracker] Tracking {len(open_signals)} open signals...")

            for signal_row in open_signals:
                signal_id = signal_row['id']
                symbol = signal_row['symbol']
                processed_in_cycle += 1
                update_executed = False # Flag to prevent multiple DB updates per signal per cycle

                try:
                    # Safely extract data from the row
                    entry_price = float(signal_row['entry_price'])
                    current_stop_loss = float(signal_row['current_stop_loss'])
                    is_trailing_active = signal_row['is_trailing_active']
                    last_sl_swing_low_price = signal_row.get('last_swing_low_price') # Price of swing low defining current SL
                    last_activation_swing_high_price = signal_row.get('last_swing_high_price') # Price of swing high used for initial activation
                    tp1_price = signal_row.get('tp1_price')
                    tp1_hit = signal_row.get('tp1_hit', False)
                    tp2_price = signal_row.get('tp2_price') # New
                    tp2_hit = signal_row.get('tp2_hit', False) # New
                    tp3_price = signal_row.get('tp3_price') # New
                    tp3_hit = signal_row.get('tp3_hit', False) # New
                    stop_loss_at_breakeven = signal_row.get('stop_loss_at_breakeven', False)

                    # Get current price
                    current_price = ticker_data.get(symbol)
                    if current_price is None:
                         logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): Current price unavailable.")
                         continue

                    active_signals_summary.append(f"{symbol}({signal_id}): P={current_price:.4f} SL={current_stop_loss:.4f} TP1={'✅' if tp1_hit else '❌'} TP2={'✅' if tp2_hit else '❌'} TP3={'✅' if tp3_hit else '❌'} BE={'✅' if stop_loss_at_breakeven else '❌'} Trail={'On' if is_trailing_active else 'Off'}")

                    # --- Define DB Update variables ---
                    update_query: Optional[sql.SQL] = None
                    update_params: Tuple = ()
                    log_message: Optional[str] = None
                    notification_details: Dict[str, Any] = {'symbol': symbol, 'id': signal_id, 'entry_price': entry_price, 'current_price': current_price}

                    # ======================================
                    # 1. Check for Stop Loss Hit FIRST
                    # ======================================
                    if current_price <= current_stop_loss:
                        loss_pct = ((current_stop_loss / entry_price) - 1) * 100 if entry_price > 0 else 0
                        profitable_sl = current_stop_loss > entry_price
                        sl_type_msg = "at a profit ✅" if profitable_sl else "at a loss ❌"
                        # Close the trade completely
                        update_query = sql.SQL("""
                            UPDATE signals
                            SET hit_stop_loss = TRUE, closing_price = %s, closed_at = NOW(),
                                profit_percentage = %s, profitable_stop_loss = %s
                            WHERE id = %s;
                        """)
                        update_params = (current_stop_loss, loss_pct, profitable_sl, signal_id)
                        log_message = f"🔻 [Tracker] {symbol}(ID:{signal_id}): Stop Loss hit ({sl_type_msg}) at {current_stop_loss:.8g} (Percentage: {loss_pct:.2f}%)."
                        notification_details.update({'type': 'stop_loss_hit', 'closing_price': current_stop_loss, 'profit_pct': loss_pct, 'profitable_sl': profitable_sl})
                        update_executed = True
                        # Go to next signal after SL hit - Execute update below

                    # ==================================================
                    # 2. Check for Take Profit Hits (TP1, TP2, TP3)
                    #    Only if SL not hit.
                    # ==================================================
                    else: # Only check TPs if SL is not hit
                        # Check TP3 first (highest target)
                        if not tp3_hit and tp3_price is not None and current_price >= tp3_price:
                            update_query = sql.SQL("UPDATE signals SET tp3_hit = TRUE WHERE id = %s;")
                            update_params = (signal_id,)
                            log_message = f"🎯 [Tracker] {symbol}(ID:{signal_id}): TP3 hit at {current_price:.8g} (>= {tp3_price:.8g})."
                            notification_details.update({'type': 'tp_hit', 'target_level': 'الهدف الثالث', 'target_price': tp3_price})
                            update_executed = True
                        # Check TP2
                        elif not tp2_hit and tp2_price is not None and current_price >= tp2_price:
                            update_query = sql.SQL("UPDATE signals SET tp2_hit = TRUE WHERE id = %s;")
                            update_params = (signal_id,)
                            log_message = f"🎯 [Tracker] {symbol}(ID:{signal_id}): TP2 hit at {current_price:.8g} (>= {tp2_price:.8g})."
                            notification_details.update({'type': 'tp_hit', 'target_level': 'الهدف الثاني', 'target_price': tp2_price})
                            update_executed = True
                        # Check TP1 (triggers Break-Even)
                        elif not tp1_hit and tp1_price is not None and current_price >= tp1_price:
                            new_stop_loss_be = entry_price # Move SL to entry
                            # Only update SL if the new BE SL is higher than the current one
                            if new_stop_loss_be > current_stop_loss:
                                update_query = sql.SQL("""
                                    UPDATE signals
                                    SET tp1_hit = TRUE, stop_loss_at_breakeven = TRUE, current_stop_loss = %s
                                    WHERE id = %s;
                                """)
                                update_params = (new_stop_loss_be, signal_id)
                                log_message = f"🛡️ [Tracker] {symbol}(ID:{signal_id}): TP1 hit at {current_price:.8g} (>= {tp1_price:.8g}). Moving SL to Break-Even ({new_stop_loss_be:.8g})."
                                notification_details.update({'type': 'tp1_hit_breakeven', 'target_price': tp1_price, 'new_stop_loss': new_stop_loss_be})
                                update_executed = True
                            else:
                                 logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): TP1 hit, but Break-Even SL ({entry_price:.8g}) is not higher than current SL ({current_stop_loss:.8g}). Keeping current SL.")
                                 # Still mark TP1 as hit, but don't change SL yet
                                 update_query = sql.SQL("UPDATE signals SET tp1_hit = TRUE WHERE id = %s;")
                                 update_params = (signal_id,)
                                 log_message = f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): TP1 hit at {current_price:.8g}, but BE SL not higher. Marked TP1 hit."
                                 # No notification needed if SL didn't move
                                 update_executed = True


                        # ==================================================
                        # 3. Check for Trailing Stop Activation/Update
                        #    Only if SL not hit.
                        #    This logic now runs regardless of TP1 hit status,
                        #    but respects the Break-Even level as a floor.
                        # ==================================================
                        # Fetch recent data for swing analysis up to the current candle
                        df_track_hist = fetch_historical_data(symbol, interval=SIGNAL_TRACKING_TIMEFRAME, limit_override=LOOKBACK_FOR_SWINGS + SWING_ORDER * 2)

                        if df_track_hist is not None and not df_track_hist.empty:
                            # Find the latest swing low and high up to the current candle
                            last_swing_low_point, last_swing_high_point = find_relevant_swing_points(df_track_hist, df_track_hist.index[-1])

                            # Determine the floor for the stop loss (entry price if BE hit, otherwise initial SL)
                            sl_floor = entry_price if stop_loss_at_breakeven else float(signal_row['initial_stop_loss']) # Use initial_stop_loss from DB

                            potential_new_sl_from_swing = current_stop_loss # Start with current SL

                            # --- a) Update Trailing Stop based on New Swing Low ---
                            if last_swing_low_point is not None:
                                last_swing_low_price = last_swing_low_point[1]
                                # Calculate potential new SL based on the latest swing low
                                potential_new_sl_from_swing = last_swing_low_price * (1 - SWING_SL_BUFFER_PCT)

                                # Only consider this new swing-based SL if it's higher than the PREVIOUSLY recorded swing low price
                                # and also higher than the current stop loss. This prevents unnecessary updates on the same swing.
                                if last_sl_swing_low_price is None or last_swing_low_price > last_sl_swing_low_price * (1 + SWING_SL_BUFFER_PCT/2): # Add a small buffer to avoid float comparison issues
                                     logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): Found potentially new swing low at {last_swing_low_price:.8g}. Previous SL Swing Low: {last_sl_swing_low_price:.8g if last_sl_swing_low_price is not None else 'N/A'}")
                                else:
                                     logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): Latest swing low ({last_swing_low_price:.8g}) is not significantly higher than the one defining current SL.")
                                     # If the latest swing low is not higher, we don't update based on it in this step.
                                     # The potential_new_sl_from_swing remains the current_stop_loss for now.
                                     potential_new_sl_from_swing = current_stop_loss # Revert if not a new higher low


                            # --- Determine the final new Stop Loss ---
                            # The new SL is the maximum of:
                            # 1. The current stop loss (to only move up)
                            # 2. The break-even price (if TP1 was hit)
                            # 3. The potential SL based on the latest significant swing low
                            new_stop_loss = max(current_stop_loss, sl_floor, potential_new_sl_from_swing)

                            # --- Execute SL Update if it's higher ---
                            if new_stop_loss > current_stop_loss:
                                update_query = sql.SQL("""
                                    UPDATE signals
                                    SET current_stop_loss = %s, is_trailing_active = TRUE,
                                        last_swing_low_price = %s -- Store the price of the swing low that caused this update
                                    WHERE id = %s;
                                """)
                                # Store the swing low price ONLY if it was the factor that increased the SL
                                swing_low_price_to_store = last_swing_low_point[1] if last_swing_low_point is not None and new_stop_loss == potential_new_sl_from_swing else last_sl_swing_low_price

                                update_params = (new_stop_loss, swing_low_price_to_store, signal_id)

                                if not is_trailing_active:
                                     log_message = f"⬆️ [Tracker] {symbol}(ID:{signal_id}): Trailing Activated (SL moved up). New SL={new_stop_loss:.8g}. Old SL={current_stop_loss:.8g}"
                                     notification_details.update({'type': 'trailing_activated_swing', 'new_stop_loss': new_stop_loss, 'old_stop_loss': current_stop_loss, 'swing_price': swing_low_price_to_store}) # Notify Activation
                                else:
                                    log_message = f"➡️ [Tracker] {symbol}(ID:{signal_id}): Trailing Stop Updated (SL moved up). New SL={new_stop_loss:.8g}. Old SL={current_stop_loss:.8g}"
                                    notification_details.update({'type': 'trailing_updated_swing', 'new_stop_loss': new_stop_loss, 'old_stop_loss': current_stop_loss, 'swing_price': swing_low_price_to_store}) # Notify Update

                                update_executed = True

                            # --- Initial Trailing Activation Check (Swing High Break above Entry) ---
                            # This is a secondary trigger for activation, in case the price runs up fast
                            # and forms a new high before a clear higher low appears.
                            # Only check this if trailing is NOT active and BE is NOT set.
                            elif not is_trailing_active and not stop_loss_at_breakeven and last_swing_high_point is not None:
                                last_swing_high_price = last_swing_high_point[1]
                                # Check if current price broke the last swing high AND this high is above entry
                                if current_price > last_swing_high_price and last_swing_high_price > entry_price and \
                                   (last_activation_swing_high_price is None or not np.isclose(last_swing_high_price, last_activation_swing_high_price)):

                                    # Find the swing low that formed *before* this broken high
                                    lows_before_break = [s for s in detect_swings(df_track_hist['low'], SWING_ORDER) if s[0] < last_swing_high_point[0]]
                                    if lows_before_break:
                                        activating_swing_low = max(lows_before_break, key=lambda item: item[0])
                                        activating_swing_low_price = activating_swing_low[1]
                                        potential_new_sl_swing_activation = activating_swing_low_price * (1 - SWING_SL_BUFFER_PCT)

                                        # Only activate if the new SL is higher than the current one (which is likely initial_stop_loss here)
                                        if potential_new_sl_swing_activation > current_stop_loss:
                                            update_query = sql.SQL("""
                                                UPDATE signals
                                                SET is_trailing_active = TRUE, current_stop_loss = %s,
                                                    last_swing_low_price = %s, last_swing_high_price = %s -- Store the high that triggered activation
                                                WHERE id = %s;
                                            """)
                                            update_params = (potential_new_sl_swing_activation, activating_swing_low_price, last_swing_high_price, signal_id)
                                            log_message = f"⬆️ [Tracker] {symbol}(ID:{signal_id}): Trailing Activated (Swing High Break). Price={current_price:.8g} > High={last_swing_high_price:.8g}. New SL={potential_new_sl_swing_activation:.8g} (Below Low={activating_swing_low_price:.8g})"
                                            notification_details.update({'type': 'trailing_activated_swing', 'current_price': current_price, 'swing_price': last_swing_high_price, 'new_stop_loss': potential_new_sl_swing_activation}) # Notify Activation
                                            update_executed = True
                                        else:
                                             logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): Swing High Break detected, but new SL ({potential_new_sl_swing_activation:.8g}) based on preceding low ({activating_swing_low_price:.8g}) is not higher than current SL ({current_stop_loss:.8g}). Not activating yet.")
                                    else:
                                         logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): Swing High Break detected, but couldn't find preceding swing low to set initial trailing SL.")
                                else:
                                     logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): Price didn't break the last swing high above entry, or already activated based on this high.")
                            # else: logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): Trailing active or BE set, skipping initial activation check.")

                        else:
                             logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): Could not fetch historical data for swing analysis.")


                    # ======================================
                    # Execute DB Update (if any)
                    # ======================================
                    if update_executed and update_query:
                        try:
                             with conn.cursor() as update_cur:
                                  update_cur.execute(update_query, update_params)
                             conn.commit()
                             if log_message: logger.info(log_message)
                             # Send notification only if type is set
                             if notification_details.get('type'):
                                send_tracking_notification(notification_details)
                        except psycopg2.Error as db_err:
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): DB error during update: {db_err}")
                            if conn: conn.rollback() # Rollback failed update
                        except Exception as exec_err:
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): Unexpected error during update/notification: {exec_err}", exc_info=True)
                            if conn: conn.rollback()

                except (TypeError, ValueError) as convert_err:
                    logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): Error converting signal values: {convert_err} - Row: {signal_row}")
                    continue # Skip this signal
                except Exception as inner_loop_err:
                     logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): Unexpected error processing signal: {inner_loop_err}", exc_info=True)
                     continue # Skip this signal

            if active_signals_summary:
                logger.debug(f"ℹ️ [Tracker] End of cycle status ({processed_in_cycle} processed): {'; '.join(active_signals_summary)}")

            # Wait before the next tracking cycle
            time.sleep(TRACKING_CYCLE_SLEEP_SECONDS)

        except psycopg2.Error as db_cycle_err:
             logger.error(f"❌ [Tracker] Database error in main tracking cycle: {db_cycle_err}. Attempting to reconnect...")
             if conn: conn.rollback()
             time.sleep(TRACKING_CYCLE_SLEEP_SECONDS * 2) # Wait longer after DB error
             check_db_connection() # Try to re-init
        except Exception as cycle_err:
            logger.error(f"❌ [Tracker] Unexpected error in signal tracking cycle: {cycle_err}", exc_info=True)
            logger.info("ℹ️ [Tracker] Waiting 120s before retrying tracking cycle...")
            time.sleep(120)


# ---------------------- Flask Service (Optional for Webhook) ----------------------
app = Flask(__name__)

@app.route('/')
def home() -> Response:
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ws_alive = ws_thread.is_alive() if 'ws_thread' in globals() and ws_thread else False
    tracker_alive = tracker_thread.is_alive() if 'tracker_thread' in globals() and tracker_thread else False
    status = "running" if ws_alive and tracker_alive else "partially running"
    return Response(f"📈 Crypto Signal Bot ({status}) - Last Check: {now}", status=200, mimetype='text/plain')

@app.route('/favicon.ico')
def favicon() -> Response: return Response(status=204)

@app.route('/webhook', methods=['POST'])
def webhook() -> Tuple[str, int]:
    if not request.is_json: logger.warning("⚠️ [Flask] Received non-JSON webhook request."); return "Invalid request format", 400
    try:
        data = request.get_json()
        logger.debug(f"ℹ️ [Flask] Received webhook data: {json.dumps(data)[:200]}...")
        if 'callback_query' in data:
            callback_query = data['callback_query']
            callback_id = callback_query['id']
            callback_data = callback_query.get('data')
            message_info = callback_query.get('message')
            if not message_info or not callback_data: logger.warning(f"⚠️ [Flask] Callback query (ID: {callback_id}) missing message or data."); return "OK", 200 # Ack handled below
            chat_id_callback = message_info.get('chat', {}).get('id')
            if not chat_id_callback: logger.warning(f"⚠️ [Flask] Callback query (ID: {callback_id}) missing chat ID."); return "OK", 200
            user_info = callback_query.get('from', {}); username = user_info.get('username', 'N/A'); user_id = user_info.get('id')
            logger.info(f"ℹ️ [Flask] Received callback: Data='{callback_data}', User={username}({user_id}), Chat={chat_id_callback}")
            try: # Acknowledge quickly
                ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
            except Exception as ack_err: logger.warning(f"⚠️ [Flask] Failed to ack callback {callback_id}: {ack_err}")
            # Process callback data in a thread
            if callback_data == "get_report":
                Thread(target=lambda: send_telegram_message(chat_id_callback, generate_performance_report(), parse_mode='Markdown')).start()
            else: logger.warning(f"⚠️ [Flask] Unhandled callback data: '{callback_data}'")
        elif 'message' in data:
            message_data = data['message']
            chat_info = message_data.get('chat'); text_msg = message_data.get('text', '').strip()
            if not chat_info or not text_msg: logger.debug("ℹ️ [Flask] Received message without chat/text."); return "OK", 200
            chat_id_msg = chat_info['id']; user_info = message_data.get('from', {}); username = user_info.get('username', 'N/A'); user_id = user_info.get('id')
            logger.info(f"ℹ️ [Flask] Received message: Text='{text_msg}', User={username}({user_id}), Chat={chat_id_msg}")
            # Process commands in threads
            if text_msg.lower() == '/report': Thread(target=lambda: send_telegram_message(chat_id_msg, generate_performance_report(), parse_mode='Markdown')).start()
            elif text_msg.lower() == '/status': Thread(target=handle_status_command, args=(chat_id_msg,)).start()
        else: logger.debug("ℹ️ [Flask] Received webhook without 'callback_query' or 'message'.")
        return "OK", 200
    except Exception as e: logger.error(f"❌ [Flask] Error processing webhook: {e}", exc_info=True); return "Internal Server Error", 500

def handle_status_command(chat_id_msg: int) -> None:
    """Separate function to handle /status command."""
    logger.info(f"ℹ️ [Flask Status] Handling /status for chat {chat_id_msg}")
    msg_sent = send_telegram_message(chat_id_msg, "⏳ جلب الحالة...")
    if not (msg_sent and msg_sent.get('ok')): logger.error(f"❌ [Flask Status] Failed to send initial status msg to {chat_id_msg}"); return
    message_id_to_edit = msg_sent['result']['message_id']
    try:
        open_count = 0
        tp1_hit_count = 0
        tp2_hit_count = 0
        tp3_hit_count = 0

        if check_db_connection() and conn:
            with conn.cursor() as status_cur:
                status_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE hit_stop_loss = FALSE;") # Count all not stopped
                open_count = (status_cur.fetchone() or {}).get('count', 0)
                status_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE tp1_hit = TRUE AND hit_stop_loss = FALSE;")
                tp1_hit_count = (status_cur.fetchone() or {}).get('count', 0)
                status_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE tp2_hit = TRUE AND hit_stop_loss = FALSE;")
                tp2_hit_count = (status_cur.fetchone() or {}).get('count', 0)
                status_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE tp3_hit = TRUE AND hit_stop_loss = FALSE;")
                tp3_hit_count = (status_cur.fetchone() or {}).get('count', 0)

        ws_status = 'نشط ✅' if 'ws_thread' in globals() and ws_thread and ws_thread.is_alive() else 'غير نشط ❌'
        tracker_status = 'نشط ✅' if 'tracker_thread' in globals() and tracker_thread and tracker_thread.is_alive() else 'غير نشط ❌'
        final_status_msg = (
            f"🤖 *حالة البوت:*\n"
            f"- تتبع الأسعار (WS): {ws_status}\n"
            f"- تتبع الإشارات: {tracker_status}\n"
            f"- الإشارات النشطة: *{open_count}* / {MAX_OPEN_TRADES}\n"
            f"  • ضربت TP1: {tp1_hit_count}\n"
            f"  • ضربت TP2: {tp2_hit_count}\n"
            f"  • ضربت TP3: {tp3_hit_count}\n"
            f"- وقت الخادم: {datetime.now().strftime('%H:%M:%S')}"
        )
        edit_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText"
        edit_payload = {'chat_id': chat_id_msg, 'message_id': message_id_to_edit, 'text': final_status_msg, 'parse_mode': 'Markdown'}
        response = requests.post(edit_url, json=edit_payload, timeout=10)
        response.raise_for_status()
        logger.info(f"✅ [Flask Status] Status updated for chat {chat_id_msg}")
    except Exception as status_err:
        logger.error(f"❌ [Flask Status] Error getting/editing status for {chat_id_msg}: {status_err}", exc_info=True)
        send_telegram_message(chat_id_msg, "❌ خطأ أثناء جلب الحالة.")

def run_flask() -> None:
    """Runs the Flask application."""
    if not WEBHOOK_URL: logger.info("ℹ️ [Flask] Webhook URL not configured. Flask server not starting."); return
    host = "0.0.0.0"; port = int(config('PORT', default=10000))
    logger.info(f"ℹ️ [Flask] Starting Flask app on {host}:{port}...")
    try:
        from waitress import serve
        logger.info("✅ [Flask] Using 'waitress' server.")
        serve(app, host=host, port=port, threads=6)
    except ImportError:
         logger.warning("⚠️ [Flask] 'waitress' not installed. Using Flask dev server (NOT FOR PRODUCTION).")
         try: app.run(host=host, port=port)
         except Exception as flask_run_err: logger.critical(f"❌ [Flask] Failed to start dev server: {flask_run_err}", exc_info=True)
    except Exception as serve_err: logger.critical(f"❌ [Flask] Failed to start server: {serve_err}", exc_info=True)

# ---------------------- Main Loop and Check Function ----------------------
def main_loop() -> None:
    """Main loop to scan pairs and generate signals."""
    # Call the new function to get symbols directly from Binance
    symbols_to_scan = get_crypto_symbols_from_binance()
    if not symbols_to_scan: logger.critical("❌ [Main] No valid symbols loaded from Binance. Cannot proceed."); return
    logger.info(f"✅ [Main] Loaded {len(symbols_to_scan)} valid symbols for scanning from Binance.")

    while True:
        try:
            scan_start_time = time.time()
            logger.info("+" + "-"*60 + "+")
            logger.info(f"🔄 [Main] Starting Market Scan Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("+" + "-"*60 + "+")

            if not check_db_connection() or not conn:
                logger.error("❌ [Main] Skipping scan cycle due to DB connection failure."); time.sleep(60); continue

            open_count = 0
            try:
                 with conn.cursor() as cur_check:
                    cur_check.execute("SELECT COUNT(*) AS count FROM signals WHERE hit_stop_loss = FALSE;") # Count non-closed signals
                    open_count = (cur_check.fetchone() or {}).get('count', 0)
            except psycopg2.Error as db_err:
                logger.error(f"❌ [Main] DB error checking open count: {db_err}. Skipping.")
                if conn:
                    conn.rollback()
                time.sleep(60)
                continue

            logger.info(f"ℹ️ [Main] Currently Open Signals: {open_count} / {MAX_OPEN_TRADES}")
            if open_count >= MAX_OPEN_TRADES: logger.info(f"⚠️ [Main] Max open signals reached. Waiting..."); time.sleep(60); continue

            processed_in_loop = 0; signals_generated_in_loop = 0; slots_available = MAX_OPEN_TRADES - open_count

            for symbol in symbols_to_scan:
                 if slots_available <= 0: logger.info(f"ℹ️ [Main] Max limit reached during scan."); break
                 processed_in_loop += 1
                 logger.debug(f"🔍 [Main] Scanning {symbol} ({processed_in_loop}/{len(symbols_to_scan)})...")
                 try:
                    with conn.cursor() as symbol_cur: # Check if already open
                        symbol_cur.execute("SELECT 1 FROM signals WHERE symbol = %s AND hit_stop_loss = FALSE LIMIT 1;", (symbol,))
                        if symbol_cur.fetchone():
                            logger.debug(f"ℹ️ [Main] {symbol} already has an open signal. Skipping.")
                            continue

                    df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_hist is None or df_hist.empty:
                        logger.debug(f"ℹ️ [Main] No historical data for {symbol}. Skipping.")
                        continue

                    strategy = ConservativeTradingStrategy(symbol)
                    df_indicators = strategy.populate_indicators(df_hist)
                    if df_indicators is None:
                        logger.debug(f"ℹ️ [Main] Failed to populate indicators for {symbol}. Skipping.")
                        continue

                    potential_signal = strategy.generate_buy_signal(df_indicators)

                    if potential_signal:
                        logger.info(f"✨ [Main] Potential signal for {symbol}! (Score: {potential_signal.get('r2_score', 0):.2f}, R:R: {potential_signal.get('risk_reward_ratio', 0):.2f}) Final check...")
                        with conn.cursor() as final_check_cur: # Final check before insert
                             final_check_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE hit_stop_loss = FALSE;")
                             final_open_count = (final_check_cur.fetchone() or {}).get('count', 0)
                             if final_open_count < MAX_OPEN_TRADES:
                                 if insert_signal_into_db(potential_signal):
                                     send_telegram_alert(potential_signal, SIGNAL_GENERATION_TIMEFRAME)
                                     signals_generated_in_loop += 1; slots_available -= 1
                                     logger.info(f"✅ [Main] Signal inserted and alert sent for {symbol}.")
                                     time.sleep(2) # Small delay after sending alert
                                 else: logger.error(f"❌ [Main] Failed to insert signal for {symbol}.")
                             else: logger.warning(f"⚠️ [Main] Max limit reached before inserting {symbol}. Ignored."); break
                    else:
                         logger.debug(f"ℹ️ [Main] No signal generated for {symbol} after checks.")

                 except psycopg2.Error as db_loop_err:
                     logger.error(f"❌ [Main] DB error processing {symbol}: {db_loop_err}. Moving next...")
                     if conn:
                         conn.rollback()
                     continue
                 except Exception as symbol_proc_err: logger.error(f"❌ [Main] General error processing {symbol}: {symbol_proc_proc_err}", exc_info=True); continue
                 time.sleep(0.3) # Small delay between symbols

            scan_duration = time.time() - scan_start_time
            logger.info(f"🏁 [Main] Scan cycle finished. Signals generated: {signals_generated_in_loop}. Duration: {scan_duration:.2f}s.")
            wait_time = max(60, 300 - scan_duration) # Wait 5 mins total or at least 1 min
            logger.info(f"⏳ [Main] Waiting {wait_time:.1f}s for next cycle...")
            time.sleep(wait_time)

        except KeyboardInterrupt: logger.info("🛑 [Main] Stop requested (KeyboardInterrupt)."); break
        except psycopg2.Error as db_main_err:
             logger.error(f"❌ [Main] Fatal DB error in main loop: {db_main_err}. Reconnecting...")
             if conn: conn.rollback(); time.sleep(60)
             try: init_db()
             except Exception as recon_err: logger.critical(f"❌ [Main] Failed to reconnect DB: {recon_err}. Exiting..."); break
        except Exception as main_err:
            logger.error(f"❌ [Main] Unexpected error in main loop: {main_err}", exc_info=True)
            logger.info("ℹ️ [Main] Waiting 120s before retrying...")
            time.sleep(120)

def cleanup_resources() -> None:
    """Closes used resources."""
    global conn
    logger.info("ℹ️ [Cleanup] Closing resources...")
    if conn:
        try: conn.close(); logger.info("✅ [DB] Database connection closed.")
        except Exception as close_err: logger.error(f"⚠️ [DB] Error closing DB connection: {close_err}")
    logger.info("✅ [Cleanup] Resource cleanup complete.")

# ---------------------- Main Entry Point ----------------------
if __name__ == "__main__":
    logger.info("🚀 Starting trading signal bot (Dynamic Tracking Version)...")
    logger.info(f"Local Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | UTC Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    ws_thread: Optional[Thread] = None
    tracker_thread: Optional[Thread] = None
    flask_thread: Optional[Thread] = None

    try:
        init_db() # Initialize DB first

        ws_thread = Thread(target=run_ticker_socket_manager, daemon=True, name="WebSocketThread")
        ws_thread.start()
        logger.info("✅ [Main] WebSocket Ticker thread started.")
        logger.info("ℹ️ [Main] Waiting 5s for WebSocket init...")
        time.sleep(5)
        if not ticker_data: logger.warning("⚠️ [Main] No initial data from WebSocket.")
        else: logger.info(f"✅ [Main] Initial data received for {len(ticker_data)} symbols.")

        tracker_thread = Thread(target=track_signals, daemon=True, name="TrackerThread")
        tracker_thread.start()
        logger.info("✅ [Main] Signal Tracker thread (Dynamic) started.")

        if WEBHOOK_URL:
            flask_thread = Thread(target=run_flask, daemon=True, name="FlaskThread")
            flask_thread.start()
            logger.info("✅ [Main] Flask Webhook thread started.")
        else: logger.info("ℹ️ [Main] Webhook URL not configured, Flask server not starting.")

        main_loop() # Start main signal generation loop

    except Exception as startup_err:
        logger.critical(f"❌ [Main] Fatal error during startup/main loop: {startup_err}", exc_info=True)
    finally:
        logger.info("🛑 [Main] Program shutting down...")
        # send_telegram_message(CHAT_ID, "⚠️ Alert: Trading bot is shutting down now.")
        cleanup_resources()
        logger.info("👋 [Main] Trading signal bot stopped.")
        os._exit(0) # Force exit if threads are stuck

