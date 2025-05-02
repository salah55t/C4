import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql, OperationalError, InterfaceError # لاستخدام استعلامات آمنة وأخطاء محددة
from psycopg2.extras import RealDictCursor # للحصول على النتائج كقواميس
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException, BinanceRequestException # أخطاء Binance المحددة
from flask import Flask, request, Response
from threading import Thread
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Tuple, Any, Union # لإضافة Type Hinting

# ---------------------- إعداد التسجيل ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # إضافة اسم المسجل
    handlers=[
        logging.FileHandler('crypto_bot_momentum_pullback.log', encoding='utf-8'), # تحديث اسم ملف السجل
        logging.StreamHandler()
    ]
)
# استخدام اسم محدد للمسجل بدلاً من الجذر
logger = logging.getLogger('CryptoBot')

# ---------------------- تحميل المتغيرات البيئية ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
    # استخدام قيمة افتراضية None إذا لم يكن المتغير موجودًا
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
except Exception as e:
     logger.critical(f"❌ Failed to load essential environment variables: {e}")
     exit(1) # استخدام رمز خروج غير صفري للإشارة إلى خطأ

logger.info(f"Binance API Key: {'Available' if API_KEY else 'Not available'}")
logger.info(f"Telegram Token: {TELEGRAM_TOKEN[:10]}...{'*' * (len(TELEGRAM_TOKEN)-10)}")
logger.info(f"Telegram Chat ID: {CHAT_ID}")
logger.info(f"Database URL: {'Available' if DB_URL else 'Not available'}")
logger.info(f"Webhook URL: {WEBHOOK_URL if WEBHOOK_URL else 'Not specified'}")

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
TRADE_VALUE: float = 10.0         # Default trade value in USDT
MAX_OPEN_TRADES: int = 4          # Maximum number of open trades simultaneously
SIGNAL_GENERATION_TIMEFRAME: str = '30m' # Timeframe for signal generation
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 5 # Historical data lookback in days for signal generation
SIGNAL_TRACKING_TIMEFRAME: str = '30m' # Timeframe for signal tracking and stop loss updates (يمكن أن تكون مختلفة عن التوليد)
SIGNAL_TRACKING_LOOKBACK_DAYS: int = 5   # Historical data lookback in days for signal tracking

# =============================================================================
# --- Indicator Parameters ---
# These values are shared and used by indicator calculation functions
# =============================================================================
RSI_PERIOD: int = 14          # RSI Period
# RSI_OVERSOLD/OVERBOUGHT thresholds are now used as filters/points, not global constants
EMA_SHORT_PERIOD: int = 13      # Short EMA period
EMA_LONG_PERIOD: int = 34       # Long EMA period
VWMA_PERIOD: int = 20           # VWMA Period
SWING_ORDER: int = 5          # Order for swing point detection (For Elliott/Fib - not strictly used in the new strategy logic but kept)
FIB_LEVELS_TO_CHECK: List[float] = [0.382, 0.5, 0.618] # Not strictly used in the new strategy logic but kept
FIB_TOLERANCE: float = 0.007 # Not strictly used in the new strategy logic but kept
LOOKBACK_FOR_SWINGS: int = 100 # Not strictly used in the new strategy logic but kept
ENTRY_ATR_PERIOD: int = 14     # ATR Period for entry/tracking (This is for the ATR calculation itself)
# ATR multipliers for initial TP/SL are now defined INSIDE the strategy class
BOLLINGER_WINDOW: int = 20     # Bollinger Bands Window
BOLLINGER_STD_DEV: int = 2       # Bollinger Bands Standard Deviation
MACD_FAST: int = 12            # MACD Fast Period
MACD_SLOW: int = 26            # MACD Slow Period
MACD_SIGNAL: int = 9             # MACD Signal Line Period
ADX_PERIOD: int = 14            # ADX Period
SUPERTREND_PERIOD: int = 10     # SuperTrend Period
SUPERTREND_MULTIPLIER: float = 3.0 # SuperTrend Multiplier

# Trailing Stop Loss (These can remain global as they apply AFTER entry)
TRAILING_STOP_ACTIVATION_PROFIT_PCT: float = 0.015 # Profit percentage to activate trailing stop (1.5%)
TRAILING_STOP_ATR_MULTIPLIER: float = 2.0        # ATR Multiplier for trailing stop (تعديل مقترح: 2.0)
TRAILING_STOP_MOVE_INCREMENT_PCT: float = 0.002  # Price increase percentage to move trailing stop (تعديل مقترح: 0.2%)

# Additional Signal Conditions (Mandatory Filters)
MIN_PROFIT_MARGIN_PCT: float = 1.5 # Minimum required profit margin percentage (تعديل مقترح: 1.5%)
MIN_VOLUME_15M_USDT: float = 200000.0 # Minimum liquidity in the last 15 minutes in USDT (تعديل مقترح: 200k)
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

def fetch_historical_data(symbol: str, interval: str = SIGNAL_GENERATION_TIMEFRAME, days: int = SIGNAL_GENERATION_LOOKBACK_DAYS) -> Optional[pd.DataFrame]:
    """Fetches historical candlestick data from Binance."""
    if not client:
        logger.error("❌ [Data] Binance client not initialized for data fetching.")
        return None
    try:
        start_dt = datetime.utcnow() - timedelta(days=days + 1) # Add an extra day as buffer
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        logger.debug(f"ℹ️ [Data] Fetching {interval} data for {symbol} since {start_str} (limit 1000 candles)...")

        klines = client.get_historical_klines(symbol, interval, start_str, limit=1000)

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
        # Return an empty series with the same index if possible to maintain compatibility
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

    # Calculate price * volume
    df_calc['price_volume'] = df_calc['close'] * df_calc['volume']

    # Calculate rolling sum of price * volume and rolling sum of volume
    # Use min_periods=period to ensure we have enough data points for the initial calculation
    rolling_price_volume_sum = df_calc['price_volume'].rolling(window=period, min_periods=period).sum()
    rolling_volume_sum = df_calc['volume'].rolling(window=period, min_periods=period).sum()

    # Calculate VWMA, avoiding division by zero
    vwma = rolling_price_volume_sum / rolling_volume_sum.replace(0, np.nan)

    # Drop the temporary column
    df_calc.drop(columns=['price_volume'], inplace=True, errors='ignore')

    return vwma

def get_btc_trend_4h() -> str:
    """Calculates Bitcoin trend on 4-hour timeframe using EMA20 and EMA50."""
    # Note: This function still uses EMA20 and EMA50 internally, you might want to unify it with the general EMA_PERIOD if desired
    logger.debug("ℹ️ [Indicators] Calculating Bitcoin 4-hour trend...")
    try:
        df = fetch_historical_data("BTCUSDT", interval=Client.KLINE_INTERVAL_4HOUR, days=10) # Request a bit more days
        if df is None or df.empty or len(df) < 50 + 1: # Ensure enough data for EMA50
            logger.warning("⚠️ [Indicators] Insufficient BTC/USDT 4H data to calculate trend.")
            return "N/A (Insufficient Data)"

        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df.dropna(subset=['close'], inplace=True)
        if len(df) < 50:
             logger.warning("⚠️ [Indicators] Insufficient BTC/USDT 4H data after removing NaNs.")
             return "N/A (Insufficient Data)"

        ema20 = calculate_ema(df['close'], 20).iloc[-1] # Still uses 20 here
        ema50 = calculate_ema(df['close'], 50).iloc[-1] # Still uses 50 here
        current_close = df['close'].iloc[-1]

        if pd.isna(ema20) or pd.isna(ema50) or pd.isna(current_close):
            logger.warning("⚠️ [Indicators] BTC EMA or current price values are NaN.")
            return "N/A (Calculation Error)"

        # Using a slightly wider spread for trend definition compared to the old code's 0.5%
        # Consider using cross or clear separation instead of just relative position
        if current_close > ema20 and ema20 > ema50:
            trend = "صعود 📈" # Uptrend
        elif current_close < ema20 and ema20 < ema50:
            trend = "هبوط 📉" # Downtrend
        else: # Crossover or unclear divergence
            trend = "تذبذب/استقرار 🔀" # Volatile / Sideways

        logger.debug(f"✅ [Indicators] Bitcoin 4H Trend: {trend}")
        return trend
    except Exception as e:
        logger.error(f"❌ [Indicators] Error calculating Bitcoin 4-hour trend: {e}", exc_info=True)
        return "N/A (Error)"

# ---------------------- Database Connection Setup ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    """Initializes database connection and creates tables if they don't exist."""
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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    initial_target DOUBLE PRECISION NOT NULL,
                    initial_stop_loss DOUBLE PRECISION NOT NULL,
                    current_target DOUBLE PRECISION NOT NULL,
                    current_stop_loss DOUBLE PRECISION NOT NULL,
                    r2_score DOUBLE PRECISION, -- Represents the signal score
                    volume_15m DOUBLE PRECISION,
                    achieved_target BOOLEAN DEFAULT FALSE,
                    hit_stop_loss BOOLEAN DEFAULT FALSE,
                    closing_price DOUBLE PRECISION,
                    closed_at TIMESTAMP,
                    sent_at TIMESTAMP DEFAULT NOW(),
                    profit_percentage DOUBLE PRECISION,
                    profitable_stop_loss BOOLEAN DEFAULT FALSE,
                    is_trailing_active BOOLEAN DEFAULT FALSE,
                    strategy_name TEXT,
                    signal_details JSONB,
                    last_trailing_update_price DOUBLE PRECISION,
                    total_possible_score DOUBLE PRECISION -- Add total possible score column
                );""")
            conn.commit()
            logger.info("✅ [DB] 'signals' table exists or was created.")

            # --- Check and add missing columns (if necessary) ---
            required_columns = {
                "symbol", "entry_price", "initial_target", "initial_stop_loss",
                "current_target", "current_stop_loss", "r2_score", "volume_15m",
                "achieved_target", "hit_stop_loss", "closing_price", "closed_at",
                "sent_at", "profit_percentage", "profitable_stop_loss",
                "is_trailing_active", "strategy_name", "signal_details",
                "last_trailing_update_price", "total_possible_score" # Include new column
            }
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'signals' AND table_schema = 'public';")
            existing_columns = {row['column_name'] for row in cur.fetchall()}
            missing_columns = required_columns - existing_columns

            if missing_columns:
                logger.warning(f"⚠️ [DB] Following columns are missing in 'signals' table: {missing_columns}. Attempting to add them...")
                # Add ALTER TABLE statements here for each missing column
                for col in missing_columns:
                    col_type = "DOUBLE PRECISION" if col in ["r2_score", "volume_15m", "profit_percentage", "last_trailing_update_price", "total_possible_score"] else \
                               "BOOLEAN" if col in ["achieved_target", "hit_stop_loss", "profitable_stop_loss", "is_trailing_active"] else \
                               "TEXT" if col == "strategy_name" else \
                               "JSONB" if col == "signal_details" else \
                               "TIMESTAMP" if col == "closed_at" else \
                               "TEXT" # Default to TEXT if type not specified
                    default_val = "DEFAULT FALSE" if col in ["achieved_target", "hit_stop_loss", "profitable_stop_loss", "is_trailing_active"] else \
                                  "DEFAULT NULL" if col in ["closed_at", "closing_price", "profit_percentage", "last_trailing_update_price", "r2_score", "volume_15m", "total_possible_score"] else \
                                  "DEFAULT ''" if col == "strategy_name" else "" # Default empty string for text, or NULL for others

                    try:
                        # Check if column already exists before adding to avoid errors on re-runs
                        cur.execute(f"SELECT 1 FROM information_schema.columns WHERE table_name = 'signals' AND column_name = '{col}';")
                        if not cur.fetchone():
                            alter_stmt = sql.SQL("ALTER TABLE signals ADD COLUMN {} {};").format(sql.Identifier(col), sql.SQL(col_type + " " + default_val))
                            cur.execute(alter_stmt)
                            conn.commit()
                            logger.info(f"✅ [DB] Added column '{col}' to 'signals' table.")
                        else:
                            logger.warning(f"⚠️ [DB] Column '{col}' already exists, skipping addition.")
                    except Exception as alter_err:
                         logger.error(f"❌ [DB] Failed to add column '{col}': {alter_err}", exc_info=True)
                         conn.rollback() # Rollback the current alter statement if it fails

                logger.info("✅ [DB] Attempted to add missing columns to 'signals' table. Please verify manually if needed.")
            else:
                logger.info("✅ [DB] All required columns exist in 'signals' table.")

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
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif pd.isna(obj): # Handle NaT/NaN from Pandas as well
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
    logger.info(f"ℹ️ [Data] Reading symbols list from file '{filename}'...")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)

        if not os.path.exists(file_path):
            file_path = os.path.abspath(filename) # Try current path if not next to script
            if not os.path.exists(file_path):
                 logger.error(f"❌ [Data] File '{filename}' not found in script directory or current directory.")
                 return [] # Return empty list if file not found
            else:
                 logger.warning(f"⚠️ [Data] File '{filename}' not found in script directory. Using file in current directory: '{file_path}'")

        with open(file_path, 'r', encoding='utf-8') as f:
            # Clean and format symbols: remove spaces, convert to uppercase, ensure ends with USDT
            raw_symbols = [f"{line.strip().upper().replace('USDT', '')}USDT"
                           for line in f if line.strip() and not line.startswith('#')]
        raw_symbols = sorted(list(set(raw_symbols))) # Remove duplicates and sort
        logger.info(f"ℹ️ [Data] Read {len(raw_symbols)} initial symbols from '{file_path}'.")

    except FileNotFoundError:
         logger.error(f"❌ [Data] File '{filename}' not found.")
         return []
    except Exception as e:
        logger.error(f"❌ [Data] Error reading file '{filename}': {e}", exc_info=True)
        return [] # Return empty list in case of error

    if not raw_symbols:
         logger.warning("⚠️ [Data] Initial symbols list is empty.")
         return []

    # --- Validate symbols against Binance API ---
    if not client:
        logger.error("❌ [Data Validation] Binance client not initialized. Cannot validate symbols.")
        return raw_symbols # Return unfiltered list if client is not ready

    try:
        logger.info("ℹ️ [Data Validation] Validating symbols and trading status from Binance API...")
        exchange_info = client.get_exchange_info()
        # Build a set of valid trading USDT symbols for faster lookup
        valid_trading_usdt_symbols = {
            s['symbol'] for s in exchange_info['symbols']
            if s.get('quoteAsset') == 'USDT' and    # Ensure quote asset is USDT
               s.get('status') == 'TRADING' and         # Ensure status is TRADING
               s.get('isSpotTradingAllowed') is True    # Ensure Spot trading is allowed
        }
        logger.info(f"ℹ️ [Data Validation] Found {len(valid_trading_usdt_symbols)} valid USDT Spot trading pairs on Binance.")

        # Filter the list read from the file based on the valid list from Binance
        validated_symbols = [symbol for symbol in raw_symbols if symbol in valid_trading_usdt_symbols]

        removed_count = len(raw_symbols) - len(validated_symbols)
        if removed_count > 0:
            removed_symbols = set(raw_symbols) - set(validated_symbols)
            logger.warning(f"⚠️ [Data Validation] Removed {removed_count} invalid or unavailable USDT Spot trading symbols from the list: {', '.join(removed_symbols)}")

        logger.info(f"✅ [Data Validation] Symbols validated. Using {len(validated_symbols)} valid symbols.")
        return validated_symbols

    except (BinanceAPIException, BinanceRequestException) as binance_err:
         logger.error(f"❌ [Data Validation] Binance API or network error while validating symbols: {binance_err}")
         logger.warning("⚠️ [Data Validation] Using initial list from file without Binance validation.")
         return raw_symbols # Return unfiltered list in case of API error
    except Exception as api_err:
         logger.error(f"❌ [Data Validation] Unexpected error while validating Binance symbols: {api_err}", exc_info=True)
         logger.warning("⚠️ [Data Validation] Using initial list from file without Binance validation.")
         return raw_symbols # Return unfiltered list in case of API error


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
                 # Ensure data is a list, even if only one item
                 data_items = msg.get('data', []) if isinstance(msg.get('data'), list) else [msg.get('data')] if msg.get('data') else []
                 for ticker_item in data_items:
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

            # Using start_miniticker_socket covers all symbols and is suitable here
            # It sends updates for all symbols, not just specific ones
            stream_name = twm.start_miniticker_socket(callback=handle_ticker_message)
            logger.info(f"✅ [WS] WebSocket stream started: {stream_name}")

            twm.join() # Wait until the manager stops (usually due to an error or stop)
            logger.warning("⚠️ [WS] WebSocket Manager stopped. Restarting...")

        except Exception as e:
            logger.error(f"❌ [WS] Fatal error in WebSocket Manager: {e}. Restarting in 15 seconds...", exc_info=True)

        # Wait before retrying to avoid resource exhaustion or IP banning
        time.sleep(15)

# ---------------------- Technical Indicator Functions ----------------------

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

    # Use ewm to calculate exponential moving average of gains and losses
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    # Calculate RS and avoid division by zero
    # Add a small epsilon to avg_loss to prevent division by exactly zero when all losses are 0
    rs = avg_gain / (avg_loss + 1e-10)

    # Calculate RSI
    rsi_series = 100 - (100 / (1 + rs))

    # Fill initial NaN values (resulting from diff or avg_loss=0) with 50 (neutral)
    # and use forward fill to fill any gaps if they exist (rare with adjust=False)
    df['rsi'] = rsi_series.ffill().fillna(50)

    return df

def calculate_atr_indicator(df: pd.DataFrame, period: int = ENTRY_ATR_PERIOD) -> pd.DataFrame:
    """Calculates Average True Range (ATR)."""
    df = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator ATR] 'high', 'low', 'close' columns missing or empty.")
        df['atr'] = np.nan
        return df
    # Need at least period + 1 candles for the first TR calculation (current high/low vs previous close)
    if len(df) < period + 1:
        logger.warning(f"⚠️ [Indicator ATR] Insufficient data ({len(df)} < {period + 1}) to calculate ATR.")
        df['atr'] = np.nan
        return df

    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()

    # Calculate True Range (TR) - Ignore NaN during max calculation, if close.shift(1) is NaN, TR is also NaN
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)

    # Calculate ATR using EMA (using span gives a result closer to TradingView than com=period-1)
    # Use min_periods to ensure we have enough data for the first calculation
    df['atr'] = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    return df


def calculate_bollinger_bands(df: pd.DataFrame, window: int = BOLLINGER_WINDOW, num_std: int = BOLLINGER_STD_DEV) -> pd.DataFrame:
    """Calculates Bollinger Bands."""
    df = df.copy()
    if 'close' not in df.columns or df['close'].isnull().all():
        logger.warning("⚠️ [Indicator BB] 'close' column missing or empty.")
        df['bb_middle'] = np.nan
        df['bb_upper'] = np.nan
        df['bb_lower'] = np.nan
        df['bb_std'] = np.nan # Also add std dev column
        return df
    if len(df) < window:
         logger.warning(f"⚠️ [Indicator BB] Insufficient data ({len(df)} < {window}) to calculate BB.")
         df['bb_middle'] = np.nan
         df['bb_upper'] = np.nan
         df['bb_lower'] = np.nan
         df['bb_std'] = np.nan
         return df

    # Use min_periods=window to ensure enough data points for initial calculation
    df['bb_middle'] = df['close'].rolling(window=window, min_periods=window).mean()
    df['bb_std'] = df['close'].rolling(window=window, min_periods=window).std()
    df['bb_upper'] = df['bb_middle'] + num_std * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - num_std * df['bb_std']
    return df


def calculate_macd(df: pd.DataFrame, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> pd.DataFrame:
    """Calculates MACD, Signal Line, and Histogram."""
    df = df.copy()
    if 'close' not in df.columns or df['close'].isnull().all():
        logger.warning("⚠️ [Indicator MACD] 'close' column missing or empty.")
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['macd_hist'] = np.nan
        return df
    min_len = max(fast, slow, signal)
    if len(df) < min_len:
        logger.warning(f"⚠️ [Indicator MACD] Insufficient data ({len(df)} < {min_len}) to calculate MACD.")
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['macd_hist'] = np.nan
        return df

    ema_fast = calculate_ema(df['close'], fast)
    ema_slow = calculate_ema(df['close'], slow)
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = calculate_ema(df['macd'], signal) # Calculate EMA of the MACD line
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df


def calculate_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    """Calculates ADX, DI+ and DI-."""
    df_calc = df.copy() # Work on a copy
    required_cols = ['high', 'low', 'close']
    if not all(col in df_calc.columns for col in required_cols) or df_calc[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator ADX] 'high', 'low', 'close' columns missing or empty.")
        df_calc['adx'] = np.nan
        df_calc['di_plus'] = np.nan
        df_calc['di_minus'] = np.nan
        return df_calc
    # ADX requires period + an additional period for smoothing
    if len(df_calc) < period * 2:
        logger.warning(f"⚠️ [Indicator ADX] Insufficient data ({len(df_calc)} < {period * 2}) to calculate ADX.")
        df_calc['adx'] = np.nan
        df_calc['di_plus'] = np.nan
        df_calc['di_minus'] = np.nan
        return df_calc

    # Calculate True Range (TR)
    df_calc['high-low'] = df_calc['high'] - df_calc['low']
    df_calc['high-prev_close'] = abs(df_calc['high'] - df_calc['close'].shift(1))
    df_calc['low-prev_close'] = abs(df_calc['low'] - df_calc['close'].shift(1))
    # Ensure TR is NaN if any of the components requiring shift(1) are NaN
    df_calc['tr'] = df_calc[['high-low', 'high-prev_close', 'low-prev_close']].max(axis=1, skipna=False)

    # Calculate Directional Movement (+DM, -DM)
    df_calc['up_move'] = df_calc['high'] - df_calc['high'].shift(1)
    df_calc['down_move'] = df_calc['low'].shift(1) - df_calc['low']
    df_calc['+dm'] = np.where((df_calc['up_move'] > df_calc['down_move']) & (df_calc['up_move'] > 0), df_calc['up_move'], 0)
    df_calc['-dm'] = np.where((df_calc['down_move'] > df_calc['up_move']) & (df_calc['down_move'] > 0), df_calc['down_move'], 0)

    # Use EMA to calculate smoothed values (alpha = 1/period) with min_periods
    alpha = 1 / period
    df_calc['tr_smooth'] = df_calc['tr'].ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    df_calc['+dm_smooth'] = df_calc['+dm'].ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    df_calc['-dm_smooth'] = df_calc['-dm'].ewm(alpha=alpha, adjust=False, min_periods=period).mean()


    # Calculate Directional Indicators (DI+, DI-) and avoid division by zero
    # Use a small epsilon to prevent division by exactly zero
    df_calc['di_plus'] = 100 * (df_calc['+dm_smooth'] / (df_calc['tr_smooth'] + 1e-10))
    df_calc['di_minus'] = 100 * (df_calc['-dm_smooth'] / (df_calc['tr_smooth'] + 1e-10))

    # Calculate Directional Movement Index (DX)
    di_sum = df_calc['di_plus'] + df_calc['di_minus']
     # Add a small epsilon to di_sum to prevent division by exactly zero
    df_calc['dx'] = 100 * abs(df_calc['di_plus'] - df_calc['di_minus']) / (di_sum + 1e-10)

    # Calculate Average Directional Index (ADX) using EMA with min_periods
    df_calc['adx'] = df_calc['dx'].ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    # Return DataFrame with only the new columns (or it can be merged with the original)
    return df_calc[['adx', 'di_plus', 'di_minus']]


def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Volume Weighted Average Price (VWAP) - Resets daily."""
    df = df.copy()
    required_cols = ['high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator VWAP] 'high', 'low', 'close' or 'volume' columns missing or empty.")
        df['vwap'] = np.nan
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            # Attempt to convert index if it's not a DatetimeIndex
            df.index = pd.to_datetime(df.index)
            logger.warning("⚠️ [Indicator VWAP] Index converted to DatetimeIndex.")
        except Exception:
            logger.error("❌ [Indicator VWAP] Failed to convert index to DatetimeIndex, cannot calculate daily VWAP.")
            df['vwap'] = np.nan
            return df

    df['date'] = df.index.date
    # Calculate typical price and volume * typical price
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol'] = df['typical_price'] * df['volume']

    # Calculate cumulative sums within each day
    try:
        # Group by date and calculate cumulative sums
        # Ensure cumulative sums start from the first non-NaN value for that day
        df['cum_tp_vol'] = df.groupby('date')['tp_vol'].cumsum()
        df['cum_volume'] = df.groupby('date')['volume'].cumsum()
    except KeyError as e:
        logger.error(f"❌ [Indicator VWAP] Error grouping data by date: {e}. Index might be incorrect.")
        df['vwap'] = np.nan
        # Drop temporary columns if they exist
        df.drop(columns=['date', 'typical_price', 'tp_vol', 'cum_tp_vol', 'cum_volume'], inplace=True, errors='ignore')
        return df
    except Exception as e:
         logger.error(f"❌ [Indicator VWAP] Unexpected error in VWAP calculation: {e}", exc_info=True)
         df['vwap'] = np.nan
         df.drop(columns=['date', 'typical_price', 'tp_vol', 'cum_tp_vol', 'cum_volume'], inplace=True, errors='ignore')
         return df


    # Calculate VWAP and avoid division by zero
    # Add a small epsilon to cum_volume to prevent division by exactly zero
    df['vwap'] = df['cum_tp_vol'] / (df['cum_volume'] + 1e-10)

    # Backfill initial NaN values at the start of each day with the next calculated value
    # Since daily VWAP accumulates, the first value might be NaN, we use the next calculated value
    df['vwap'] = df['vwap'].bfill()

    # Remove helper columns
    df.drop(columns=['date', 'typical_price', 'tp_vol', 'cum_tp_vol', 'cum_volume'], inplace=True, errors='ignore')
    return df


def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates On-Balance Volume (OBV)."""
    df = df.copy()
    required_cols = ['close', 'volume']
    if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator OBV] 'close' or 'volume' columns missing or empty.")
        df['obv'] = np.nan
        return df
    if not pd.api.types.is_numeric_dtype(df['close']) or not pd.api.types.is_numeric_dtype(df['volume']):
        logger.warning("⚠️ [Indicator OBV] 'close' or 'volume' columns are not numeric.")
        df['obv'] = np.nan
        return df

    obv = np.zeros(len(df), dtype=np.float64) # Use numpy array for faster processing
    close = df['close'].values
    volume = df['volume'].values

    # Calculate close changes once
    close_diff = df['close'].diff().values

    for i in range(1, len(df)):
        # Check for NaN in current close, volume, or close_diff
        if np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(close_diff[i]):
            obv[i] = obv[i-1] # Keep previous value in case of NaN
            continue

        # Check if previous OBV is NaN before adding/subtracting
        prev_obv = obv[i-1] if i > 0 else 0 # Initialize with 0 if it's the first iteration

        if close_diff[i] > 0: # Price increased
            obv[i] = prev_obv + volume[i]
        elif close_diff[i] < 0: # Price decreased
             obv[i] = prev_obv - volume[i]
        else: # Price unchanged
             obv[i] = prev_obv

    df['obv'] = obv
    return df


def calculate_supertrend(df: pd.DataFrame, period: int = SUPERTREND_PERIOD, multiplier: float = SUPERTREND_MULTIPLIER) -> pd.DataFrame:
    """Calculates the SuperTrend indicator."""
    df_st = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df_st.columns for col in required_cols) or df_st[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator SuperTrend] 'high', 'low', 'close' columns missing or empty.")
        df_st['supertrend'] = np.nan
        df_st['supertrend_trend'] = 0 # 0: unknown, 1: uptrend, -1: downtrend
        return df_st

    # Ensure ATR column exists or calculate it with min_periods
    # Use the ATR period specific to SuperTrend here (usually the same as SuperTrend period)
    df_st['atr'] = calculate_atr_indicator(df_st, period=period)['atr']


    if 'atr' not in df_st.columns or df_st['atr'].isnull().all():
         logger.warning("⚠️ [Indicator SuperTrend] Cannot calculate SuperTrend due to missing valid ATR values.")
         df_st['supertrend'] = np.nan
         df_st['supertrend_trend'] = 0
         return df_st
    if len(df_st) < period:
        logger.warning(f"⚠️ [Indicator SuperTrend] Insufficient data ({len(df_st)} < {period}) to calculate SuperTrend.")
        df_st['supertrend'] = np.nan
        df_st['supertrend_trend'] = 0
        return df_st

    # Calculate basic upper and lower bands
    hl2 = (df_st['high'] + df_st['low']) / 2
    df_st['basic_ub'] = hl2 + multiplier * df_st['atr']
    df_st['basic_lb'] = hl2 - multiplier * df_st['atr']

    # Initialize final columns with NaN or 0
    df_st['final_ub'] = np.nan
    df_st['final_lb'] = np.nan
    df_st['supertrend'] = np.nan
    df_st['supertrend_trend'] = 0 # 1 for uptrend, -1 for downtrend

    # Use .values for faster access within the loop
    close = df_st['close'].values
    basic_ub = df_st['basic_ub'].values
    basic_lb = df_st['basic_lb'].values
    final_ub = df_st['final_ub'].values # Will be modified within the loop
    final_lb = df_st['final_lb'].values # Will be modified within the loop
    st = df_st['supertrend'].values     # Will be modified within the loop
    st_trend = df_st['supertrend_trend'].values # Will be modified within the loop

    # Initialize the first values where sufficient data is available (after the ATR period)
    # Find the first non-NaN index for basic_ub/lb (which depend on ATR)
    first_valid_idx = df_st['basic_ub'].first_valid_index()
    if first_valid_idx is not None:
        first_idx_pos = df_st.index.get_loc(first_valid_idx)
        if first_idx_pos is not None: # Ensure we got a valid integer position
            # Initialize the first valid final bands and trend based on the first valid basic bands
            final_ub[first_idx_pos] = basic_ub[first_idx_pos]
            final_lb[first_idx_pos] = basic_lb[first_idx_pos]
            # Initial trend determination for the first valid candle
            if close[first_idx_pos] > final_ub[first_idx_pos]:
                 st[first_idx_pos] = final_lb[first_idx_pos]
                 st_trend[first_idx_pos] = 1
            elif close[first_idx_pos] < final_lb[first_idx_pos]:
                 st[first_idx_pos] = final_ub[first_idx_pos]
                 st_trend[first_idx_pos] = -1
            else:
                 st[first_idx_pos] = np.nan # Undefined if within bands initially
                 st_trend[first_idx_pos] = 0 # Undefined trend


    # Start the loop from the first valid index + 1 (comparing current with previous)
    if first_valid_idx is not None:
        start_loop_idx = df_st.index.get_loc(first_valid_idx) + 1
    else:
        start_loop_idx = 1 # Fallback, though data should be checked earlier

    for i in range(start_loop_idx, len(df_st)):
        # Handle NaN in essential inputs for this candle
        if pd.isna(basic_ub[i]) or pd.isna(basic_lb[i]) or pd.isna(close[i]) or pd.isna(final_ub[i-1]) or pd.isna(final_lb[i-1]) or pd.isna(close[i-1]):
            # In case of NaN, keep previous values for final bands, supertrend, and trend
            final_ub[i] = final_ub[i-1] if i > 0 else np.nan
            final_lb[i] = final_lb[i-1] if i > 0 else np.nan
            st[i] = st[i-1] if i > 0 else np.nan
            st_trend[i] = st_trend[i-1] if i > 0 else 0
            continue

        # Calculate Final Upper Band
        if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]

        # Calculate Final Lower Band
        if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]

        # Determine SuperTrend line and trend
        if st_trend[i-1] == -1: # If previous trend was downtrend (-1)
            if close[i] <= final_ub[i]: # Continue downtrend
                st[i] = final_ub[i]
                st_trend[i] = -1
            else: # Trend changes to uptrend
                st[i] = final_lb[i]
                st_trend[i] = 1
        elif st_trend[i-1] == 1: # If previous trend was uptrend (1)
            if close[i] >= final_lb[i]: # Continue uptrend
                st[i] = final_lb[i]
                st_trend[i] = 1
            else: # Trend changes to downtrend
                st[i] = final_ub[i]
                st_trend[i] = -1
        else: # Initial state (or if previous value was NaN or 0)
             # Try to infer trend from current price vs calculated final bands
             if close[i] > final_ub[i]: # Start of uptrend
                 st[i] = final_lb[i]
                 st_trend[i] = 1
             elif close[i] < final_lb[i]: # Start of downtrend
                  st[i] = final_ub[i]
                  st_trend[i] = -1
             else: # Still between bands or unclear
                  st[i] = np.nan # Undefined
                  st_trend[i] = 0


    # Assign calculated values back to DataFrame
    df_st['final_ub'] = final_ub
    df_st['final_lb'] = final_lb
    df_st['supertrend'] = st
    df_st['supertrend_trend'] = st_trend

    # Remove helper columns (optional)
    # df_st.drop(columns=['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, errors='ignore')

    return df_st


# ---------------------- Candlestick Patterns ----------------------

def is_hammer(row: pd.Series) -> bool:
    """Checks for Hammer pattern (bullish signal). Returns True or False."""
    o, h, l, c = row.get('open'), row.get('high'), row.get('low'), row.get('close')
    if pd.isna([o, h, l, c]).any(): return False
    body = abs(c - o)
    candle_range = h - l
    if candle_range == 0: return False
    lower_shadow = min(o, c) - l
    upper_shadow = h - max(o, c)
    # More standard hammer conditions
    is_small_body = body <= (candle_range * 0.3) # Body is small
    is_long_lower_shadow = lower_shadow >= 2.0 * body if body > 0 else lower_shadow > (candle_range * 0.6) # Lower shadow at least twice the body
    is_very_small_upper_shadow = upper_shadow < (body * 0.1) if body > 0 else upper_shadow < (candle_range * 0.05) # Upper shadow is very small or absent
    # Optional: Hammer forms after a downtrend (previous close < previous open) - could add this for stricter criteria
    # if prev_row is not None and pd.notna([prev_row['open'], prev_row['close']]).all() and prev_row['close'] < prev_row['open']:
    return bool(is_small_body and is_long_lower_shadow and is_very_small_upper_shadow) # Removed the downtrend check for broader applicability


def is_bullish_engulfing(df: pd.DataFrame, idx: int) -> bool:
    """Checks for Bullish Engulfing pattern. Returns True or False."""
    if idx == 0: return False
    prev = df.iloc[idx - 1]
    curr = df.iloc[idx]
    # Check for NaN in required values
    if pd.isna([prev['close'], prev['open'], curr['close'], curr['open']]).any():
        return False

    # Bullish Engulfing: Previous candle bearish, current bullish engulfing previous body
    is_prev_bearish = prev['close'] < prev['open']
    is_curr_bullish = curr['close'] > curr['open']
    is_engulfing = curr['open'] <= prev['close'] and curr['close'] >= prev['open']

    return bool(is_prev_bearish and is_curr_bullish and is_engulfing) # Ensure boolean output


def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Applies candlestick pattern detection functions to the DataFrame."""
    df = df.copy()
    logger.debug("ℹ️ [Indicators] Detecting candlestick patterns...")

    # Apply single-row patterns
    df['Hammer'] = df.apply(is_hammer, axis=1)

    # Engulfing requires access to the previous row
    # Use a list comprehension to apply the function row by row with index
    bullish_engulfing_values = [is_bullish_engulfing(df, i) for i in range(len(df))]
    df['BullishEngulfing'] = bullish_engulfing_values


    # Aggregate strong bullish candle signals
    # A bullish candle signal is present if Hammer OR Bullish Engulfing is True
    df['BullishCandleSignal'] = df.apply(lambda row: row['Hammer'] or row['BullishEngulfing'], axis=1)


    # Drop individual pattern columns if not needed later
    # df.drop(columns=['Hammer', 'BullishEngulfing'], inplace=True, errors='ignore')
    logger.debug("✅ [Indicators] Candlestick patterns detected.")
    return df


# ---------------------- Other Helper Functions (Elliott, Swings, Volume) ----------------------
# (Keeping these functions, though Elliott/Fib might not be used in the new strategy directly)
def detect_swings(prices: np.ndarray, order: int = SWING_ORDER) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Detects swing points (peaks and troughs) in a time series (numpy array)."""
    n = len(prices)
    if n < 2 * order + 1: return [], []

    maxima_indices = []
    minima_indices = []

    # Improve performance by avoiding loop on unnecessary edges and using numpy operations
    # Ensure prices array does not contain NaN within the window check
    for i in range(order, n - order):
        window = prices[i - order : i + order + 1]
        if np.isnan(window).any(): continue

        center_val = prices[i]

        is_max = np.all(center_val >= window)
        is_min = np.all(center_val <= window)

        # Ensure it's the only peak/trough in the window to avoid duplicates in flat areas
        # Consider a tolerance for flat tops/bottoms if needed, but exact match is simpler
        is_unique_max = is_max and (np.sum(window == center_val) == 1)
        is_unique_min = is_min and (np.sum(window == center_val) == 1)


        if is_unique_max:
            # Ensure no peak is too close (within 'order' distance)
            # This check helps filter out minor wiggles
            if not maxima_indices or i > maxima_indices[-1] + order:
                 maxima_indices.append(i)
        elif is_unique_min:
            # Ensure no trough is too close
            if not minima_indices or i > minima_indices[-1] + order:
                minima_indices.append(i)

    maxima = [(idx, prices[idx]) for idx in maxima_indices]
    minima = [(idx, prices[idx]) for idx in minima_indices]
    return maxima, minima

def detect_elliott_waves(df: pd.DataFrame, order: int = SWING_ORDER) -> List[Dict[str, Any]]:
    """Simple attempt to identify Elliott Waves based on MACD histogram swings."""
    if 'macd_hist' not in df.columns or df['macd_hist'].isnull().all():
        logger.warning("⚠️ [Elliott] 'macd_hist' column missing or empty for Elliott Wave calculation.")
        return []

    # Use only non-null values and their original index
    df_nonan_macd = df['macd_hist'].dropna()
    macd_values = df_nonan_macd.values

    if len(macd_values) < 2 * order + 1:
         logger.warning("⚠️ [Elliott] Insufficient MACD hist data after removing NaNs.")
         return []

    maxima_indices_local, minima_indices_local = detect_swings(macd_values, order=order)

    # Map local indices back to original DataFrame indices
    maxima = [(df_nonan_macd.index[idx], macd_values[idx]) for idx in maxima_indices_local]
    minima = [(df_nonan_macd.index[idx], macd_values[idx]) for idx in minima_indices_local]


    # Merge and sort all swing points by original index (timestamp)
    all_swings = sorted(
        [(timestamp, val, 'max') for timestamp, val in maxima] +
        [(timestamp, val, 'min') for timestamp, val in minima],
        key=lambda x: x[0] # Sort by time (original index)
    )

    waves = []
    wave_number = 1
    for timestamp, val, typ in all_swings:
        # Very basic classification, may not strictly follow Elliott rules
        # This classification might need significant refinement for actual EW analysis
        wave_type = "Impulse" if (typ == 'max' and val > 0) or (typ == 'min' and val >= 0) else "Correction"
        waves.append({
            "wave": wave_number,
            "timestamp": str(timestamp), # Convert Timestamp to string for JSON
            "macd_hist_value": float(val), # Ensure float
            "swing_type": typ,
            "classified_type": wave_type
        })
        wave_number += 1
    return waves


def fetch_recent_volume(symbol: str) -> float:
    """Fetches the trading volume in USDT for the last 15 minutes for the specified symbol."""
    if not client:
         logger.error(f"❌ [Data Volume] Binance client not initialized to fetch volume for {symbol}.")
         return 0.0
    try:
        logger.debug(f"ℹ️ [Data Volume] Fetching 15-minute volume for {symbol}...")
        # Fetch 1-minute data for the last 15 minutes
        klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=15)
        if not klines or len(klines) < 15:
             logger.warning(f"⚠️ [Data Volume] Insufficient 1m data (less than 15 candles) for {symbol}.")
             # Fallback: Try fetching 15m candle data if 1m is not enough
             klines_15m = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=1)
             if klines_15m and len(klines_15m) > 0:
                  # Use the quote volume of the last 15m candle
                  volume_usdt_15m = float(klines_15m[0][7]) if len(klines_15m[0]) > 7 and klines_15m[0][7] else 0.0
                  logger.debug(f"ℹ️ [Data Volume] Using last 15m candle volume as fallback for {symbol}: {volume_usdt_15m:.2f} USDT")
                  return volume_usdt_15m
             logger.warning(f"⚠️ [Data Volume] No 15m candle data available for {symbol} either.")
             return 0.0

        # Quote Asset Volume is the 8th field (index 7)
        volume_usdt = sum(float(k[7]) for k in klines if len(k) > 7 and k[7])
        logger.debug(f"✅ [Data Volume] Last 15 minutes liquidity for {symbol}: {volume_usdt:.2f} USDT")
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
        # Use a new cursor within the function to ensure no interference
        with conn.cursor() as report_cur: # Uses RealDictCursor
            # 1. Open Signals
            report_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;")
            open_signals_count = (report_cur.fetchone() or {}).get('count', 0)

            # 2. Closed Signals Statistics
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
                    COALESCE(AVG(CASE WHEN profit_percentage < 0 THEN profit_percentage END), 0) AS avg_loss_pct
                FROM signals
                WHERE achieved_target = TRUE OR hit_stop_loss = TRUE;
            """)
            closed_stats = report_cur.fetchone() or {} # Handle case with no results

            total_closed = closed_stats.get('total_closed', 0)
            winning_signals = closed_stats.get('winning_signals', 0)
            losing_signals = closed_stats.get('losing_signals', 0)
            total_profit_pct = closed_stats.get('total_profit_pct', 0.0)
            gross_profit_pct = closed_stats.get('gross_profit_pct', 0.0)
            gross_loss_pct = closed_stats.get('gross_loss_pct', 0.0) # Will be negative or zero
            avg_win_pct = closed_stats.get('avg_win_pct', 0.0)
            avg_loss_pct = closed_stats.get('avg_loss_pct', 0.0) # Will be negative or zero

            # 3. Calculate Derived Metrics
            win_rate = (winning_signals / total_closed * 100) if total_closed > 0 else 0.0
             # Profit Factor: Total Profit / Absolute Total Loss
            profit_factor = (gross_profit_pct / abs(gross_loss_pct)) if gross_loss_pct != 0 else float('inf')

        # 4. Format the report in Arabic
        report = (
            f"📊 *تقرير الأداء الشامل:*\n"
            f"——————————————\n"
            f"📈 الإشارات المفتوحة حالياً: *{open_signals_count}*\n"
            f"——————————————\n"
            f"📉 *إحصائيات الإشارات المغلقة:*\n"
            f"  • إجمالي الإشارات المغلقة: *{total_closed}*\n"
            f"  ✅ إشارات رابحة: *{winning_signals}* ({win_rate:.2f}%)\n" # Add win rate here
            f"  ❌ إشارات خاسرة: *{losing_signals}*\n"
            f"——————————————\n"
            f"💰 *الربحية:*\n"
            f"  • صافي الربح/الخسارة (الإجمالي %): *{total_profit_pct:+.2f}%*\n"
            f"  • إجمالي الربح (%): *{gross_profit_pct:+.2f}%*\n"
            f"  • إجمالي الخسارة (%): *{gross_loss_pct:.2f}%*\n"
            f"  • متوسط الصفقة الرابحة (%): *{avg_win_pct:+.2f}%*\n"
            f"  • متوسط الصفقة الخاسرة (%): *{avg_loss_pct:.2f}%*\n"
            f"  • عامل الربح: *{'∞' if profit_factor == float('inf') else f'{profit_factor:.2f}'}*\n"
            f"——————————————\n"
            f"🕰️ _التقرير محدث حتى: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        )
        logger.info("✅ [Report] Performance report generated successfully.")
        return report

    except psycopg2.Error as db_err:
        logger.error(f"❌ [Report] Database error generating performance report: {db_err}")
        if conn: conn.rollback() # Rollback any potentially open transaction
        return "❌ خطأ في قاعدة البيانات أثناء إنشاء تقرير الأداء."
    except Exception as e:
        logger.error(f"❌ [Report] Unexpected error generating performance report: {e}", exc_info=True)
        return "❌ حدث خطأ غير متوقع أثناء إنشاء تقرير الأداء."

# ---------------------- Trading Strategy (Momentum Pullback to VWMA & BB) -------------------

class MomentumPullbackStrategy:
    """Encapsulates the Momentum Pullback trading strategy logic."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.strategy_name = 'Momentum_Pullback_VWMA_BB' # اسم الاستراتيجية الجديدة

        # Required columns for indicator calculation
        self.required_cols_indicators = [
            'open', 'high', 'low', 'close', 'volume',
            'ema_13', 'ema_34', 'vwma',
            'rsi', 'atr', 'bb_upper', 'bb_lower', 'bb_middle',
            'macd', 'macd_signal', 'macd_hist',
            'adx', 'di_plus', 'di_minus',
            'vwap', # VWAP is calculated but used as an optional filter/point
            'obv',
            'supertrend', 'supertrend_trend',
            'BullishCandleSignal', # Aggregated bullish candle signal
            'Hammer', 'BullishEngulfing' # Keep individual pattern columns for detailed checking
        ]
        # Required columns for buy signal generation (ensure these are in df_processed)
        self.required_cols_buy_signal = [
            'close', 'open', 'low', # Added open, low for candle checks and bounce check
            'ema_13', 'ema_34', 'vwma',
            'rsi', 'atr',
            'macd', 'macd_signal', 'macd_hist',
            'supertrend_trend', 'adx', 'di_plus', 'di_minus', 'vwap',
            'bb_upper', 'bb_lower', 'bb_middle', # Ensure BB components are here
            'BullishCandleSignal', 'Hammer', 'BullishEngulfing', # Ensure candle signals are here
            'obv'
        ]


        # =====================================================================
        # --- Mandatory Entry Conditions (All must be met) ---
        # شروط الدخول الإلزامية (يجب تحقيقها جميعاً)
        # =====================================================================
        # Note: BTC Trend and Minimum Volume are checked in the main loop as filters
        self.essential_conditions = [
            'overall_bullish_trend', # EMA cross, SuperTrend trend & price above ST
            'adx_trending_bullish',  # ADX > 20 and DI+ > DI-
            'pullback_to_vwma_or_bblower', # Price touches/near VWMA or Lower BB
            'price_below_bb_middle', # Price is in the lower half of BB range (during pullback)
            'macd_positive_or_cross', # MACD momentum confirmation
            'rsi_filtered',           # RSI above 50, below 65
        ]
        # =====================================================================

        # =====================================================================
        # --- Scoring System (Weights) for Optional Conditions ---
        # نظام النقاط (الأوزان) للشروط الاختيارية (تساهم في قوة الإشارة)
        # =====================================================================
        self.condition_weights = {
            'rsi_mid_range': 1.5,         # RSI between 55 and 65 (stronger momentum confirmation for pullback)
            'bullish_candle_at_pullback': 1.5, # Bullish pattern (Hammer/Engulfing) near VWMA/BB Lower
            'obv_rising': 1.0,             # OBV is rising
            'price_touches_bb_lower': 1.0, # Price explicitly touches/closes very near Lower BB
            'price_bounces_from_vwma': 1.0, # Price shows bounce sign after touching VWMA
            'macd_hist_positive_score': 0.8, # MACD histogram is positive (score, not mandatory filter)
            # 'above_vwap_optional': 0.5,   # Price above Daily VWAP (can be an optional point) - Keep or remove
        }
        # =====================================================================

        # Calculate total possible score for *optional* conditions
        self.total_possible_score = sum(self.condition_weights.values())

        # Required signal score threshold for *optional* conditions (as a fixed value or percentage)
        # تم اختيار قيمة ثابتة كنقطة بداية
        self.min_signal_score = 3.5 # مثال: الحد الأدنى المطلوب 3.5 نقطة من النقاط الاختيارية

        # ATR Multipliers for Initial TP and SL (Specific to THIS strategy)
        self.initial_tp_multiplier = 2.5 # ATR multiplier for Take Profit
        self.initial_sl_multiplier = 1.8 # ATR multiplier for Stop Loss (تم تعديله ليكون أضيق)
        self.vwma_bb_tolerance_pct = 0.003 # Tolerance for price being "near" VWMA/BB bands (0.3%)

    def populate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculates all required indicators for the strategy."""
        logger.debug(f"ℹ️ [Strategy {self.symbol}] Calculating indicators...")
        # Update minimum required rows based on the largest period of used indicators
        min_len_required = max(EMA_SHORT_PERIOD, EMA_LONG_PERIOD, VWMA_PERIOD, RSI_PERIOD, ENTRY_ATR_PERIOD, BOLLINGER_WINDOW, MACD_SLOW, ADX_PERIOD*2, SUPERTREND_PERIOD) + 10 # Add a buffer


        if len(df) < min_len_required:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame too short ({len(df)} < {min_len_required}) to calculate indicators.")
            return None

        try:
            df_calc = df.copy()
            # Ensure essential OHLCV columns are numeric before calculations
            for col in ['open', 'high', 'low', 'close', 'volume']:
                 df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')


            # ATR is required for SuperTrend and Stop Loss/Target
            df_calc = calculate_atr_indicator(df_calc, ENTRY_ATR_PERIOD)
            # SuperTrend needs ATR calculated with its own period
            df_calc = calculate_supertrend(df_calc, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)

            # --- EMA Calculation ---
            df_calc['ema_13'] = calculate_ema(df_calc['close'], EMA_SHORT_PERIOD)
            df_calc['ema_34'] = calculate_ema(df_calc['close'], EMA_LONG_PERIOD)
            # ----------------------

            # --- VWMA Calculation ---
            df_calc['vwma'] = calculate_vwma(df_calc, VWMA_PERIOD)
            # ----------------------

            # Rest of the indicators
            df_calc = calculate_rsi_indicator(df_calc, RSI_PERIOD)
            df_calc = calculate_bollinger_bands(df_calc, BOLLINGER_WINDOW, BOLLINGER_STD_DEV)
            df_calc = calculate_macd(df_calc, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            adx_df = calculate_adx(df_calc, ADX_PERIOD)
            # Ensure index alignment when joining
            df_calc = df_calc.join(adx_df[['adx', 'di_plus', 'di_minus']], how='left')

            df_calc = calculate_vwap(df_calc)
            df_calc = calculate_obv(df_calc)
            df_calc = detect_candlestick_patterns(df_calc) # Adds BullishCandleSignal, Hammer, BullishEngulfing

            # Check for required columns after calculation
            missing_cols = [col for col in self.required_cols_indicators if col not in df_calc.columns]
            if missing_cols:
                 logger.error(f"❌ [Strategy {self.symbol}] Required indicator columns missing after calculation: {missing_cols}")
                 logger.debug(f"Columns present: {df_calc.columns.tolist()}")
                 return None

            # Handle NaNs after indicator calculation - drop rows where essential columns are NaN
            initial_len = len(df_calc)
            # Use required_cols_buy_signal which contains indicators + OHLC needed for signal
            df_cleaned = df_calc.dropna(subset=self.required_cols_buy_signal).copy()
            dropped_count = initial_len - len(df_cleaned)

            if dropped_count > 0:
                 logger.debug(f"ℹ️ [Strategy {self.symbol}] Dropped {dropped_count} rows due to NaN in essential signal data/indicators.")
            if df_cleaned.empty or len(df_cleaned) < 2:
                logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame is empty or too short (<2) after removing NaNs.")
                return None

            latest = df_cleaned.iloc[-1]
            logger.debug(f"✅ [Strategy {self.symbol}] Indicators calculated. Latest Close: {latest.get('close', np.nan):.4f}, EMA13: {latest.get('ema_13', np.nan):.4f}, EMA34: {latest.get('ema_34', np.nan):.4f}, VWMA: {latest.get('vwma', np.nan):.4f}, BB Lower: {latest.get('bb_lower', np.nan):.4f}, BB Middle: {latest.get('bb_middle', np.nan):.4f}, MACD Hist: {latest.get('macd_hist', np.nan):.4f}, RSI: {latest.get('rsi', np.nan):.1f}, ADX: {latest.get('adx', np.nan):.1f}, ST Trend: {latest.get('supertrend_trend', 0)}")
            return df_cleaned

        except KeyError as ke:
             logger.error(f"❌ [Strategy {self.symbol}] Error: Required column not found during indicator calculation: {ke}", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"❌ [Strategy {self.symbol}] Unexpected error during indicator calculation: {e}", exc_info=True)
            return None


    def generate_buy_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Generates a buy signal based on the processed DataFrame, mandatory conditions, and scoring system
        for the Momentum Pullback strategy.
        """
        logger.debug(f"ℹ️ [Strategy {self.symbol}] Generating buy signal...")

        # Check DataFrame and columns
        # Ensure df_processed has at least 2 rows to compare current and previous
        if df_processed is None or df_processed.empty or len(df_processed) < 2:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame is empty or too short (<2), cannot generate signal.")
            return None

        missing_cols = [col for col in self.required_cols_buy_signal if col not in df_processed.columns]
        if missing_cols:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame missing required columns for signal: {missing_cols}. Cannot generate signal.")
            return None

        # Check Bitcoin trend (still a mandatory filter from the main loop, but log here)
        btc_trend = get_btc_trend_4h()
        # Allow signal even if Bitcoin trend is neutral, volatile, or unknown, but not bearish
        if "هبوط" in btc_trend: # Downtrend
            logger.debug(f"ℹ️ [Strategy {self.symbol}] Trading paused due to bearish Bitcoin trend ({btc_trend}).")
            return None
        # Do not reject if "N/A" or "استقرار" (Sideways) or "تذبذب" (Volatile)
        elif "N/A" in btc_trend or "استقرار" in btc_trend or "تذبذب" in btc_trend:
             logger.debug(f"ℹ️ [Strategy {self.symbol}] Bitcoin trend is {btc_trend}. Proceeding with signal check.") # Log if not rejecting


        # Extract latest and previous candle data
        last_row = df_processed.iloc[-1]
        prev_row = df_processed.iloc[-2]

        # Check for NaN in essential columns required for the signal on the last row
        last_row_essential_check = last_row[self.required_cols_buy_signal]
        if last_row_essential_check.isnull().any():
            nan_cols = last_row_essential_check[last_row_essential_check.isnull()].index.tolist()
            logger.warning(f"⚠️ [Strategy {self.symbol}] Last row contains NaN in required signal columns: {nan_cols}. Cannot generate signal.")
            return None

        # Also check previous row for OBV and close/open for candle/bounce logic
        if pd.isna(prev_row.get('obv')) or pd.isna(prev_row.get('close')) or pd.isna(prev_row.get('open')) or pd.isna(prev_row.get('low')):
             logger.debug(f"ℹ️ [Strategy {self.symbol}] Previous row contains NaN in OBV/close/open/low. Some optional conditions might not be met.")
             # Continue, but some optional conditions might evaluate to False due to NaN

        # =====================================================================
        # --- Check Mandatory Conditions First ---
        # If any mandatory condition fails, the signal is rejected immediately
        # =====================================================================
        essential_passed = True
        failed_essential_conditions = []
        signal_details = {} # To store details of checked conditions (mandatory and optional)
        current_price = last_row['close']


        # 1. Overall Bullish Trend
        # Combine EMA cross, SuperTrend trend, and price above ST line
        ema_crossed_up = last_row['ema_13'] > last_row['ema_34']
        st_is_up_and_price_above = pd.notna(last_row['supertrend']) and last_row['supertrend_trend'] == 1 and last_row['close'] > last_row['supertrend']

        if not (ema_crossed_up and st_is_up_and_price_above):
            essential_passed = False
            failed_essential_conditions.append('Overall Bullish Trend (EMA/ST)')
            detail_trend = f'EMA13>EMA34: {ema_crossed_up}, ST Up/PriceAbove: {st_is_up_and_price_above}'
            signal_details['Overall_Bullish_Trend'] = f'Failed: {detail_trend}'
        else:
             signal_details['Overall_Bullish_Trend'] = 'Passed'


        # 2. ADX and DI+ > DI- (Trending Bullish)
        if not (pd.notna(last_row['adx']) and pd.notna(last_row['di_plus']) and pd.notna(last_row['di_minus']) and last_row['adx'] > 20 and last_row['di_plus'] > last_row['di_minus']):
             essential_passed = False
             failed_essential_conditions.append('ADX/DI (Trending Bullish)')
             detail_adx = f'ADX:{last_row.get("adx", np.nan):.1f}, DI+:{last_row.get("di_plus", np.nan):.1f}, DI-:{last_row.get("di_minus", np.nan):.1f}'
             signal_details['ADX/DI'] = f'Failed: Not Trending Bullish (ADX <= 20 or DI+ <= DI-) ({detail_adx})'
        else:
             signal_details['ADX/DI'] = f'Passed: Trending Bullish (ADX:{last_row["adx"]:.1f}, DI+>DI-)'


        # 3. Pullback to VWMA or BB Lower Band
        # Price is near VWMA OR Price is near/touches BB Lower Band
        vwma_tolerance = last_row['vwma'] * self.vwma_bb_tolerance_pct
        bb_lower_tolerance = last_row['bb_lower'] * self.vwma_bb_tolerance_pct

        price_near_vwma = pd.notna(last_row['vwma']) and abs(current_price - last_row['vwma']) <= vwma_tolerance
        price_near_bb_lower = pd.notna(last_row['bb_lower']) and (current_price <= last_row['bb_lower'] * (1 + self.vwma_bb_tolerance_pct) and current_price >= last_row['bb_lower'] * (1 - self.vwma_bb_tolerance_pct))


        if not (price_near_vwma or price_near_bb_lower):
             essential_passed = False
             failed_essential_conditions.append('Pullback (Near VWMA or BB Lower)')
             detail_pullback = f'NearVWMA: {price_near_vwma}, NearBBLower: {price_near_bb_lower}'
             signal_details['Pullback_Zone'] = f'Failed: Not Near VWMA or BB Lower ({detail_pullback})'
        else:
             signal_details['Pullback_Zone'] = f'Passed: Near VWMA ({price_near_vwma}) or BB Lower ({price_near_bb_lower})'


        # 4. Price Below Middle Bollinger Band (Confirms it's in the pullback portion of the band)
        if not (pd.notna(last_row['bb_middle']) and current_price < last_row['bb_middle']):
             essential_passed = False
             failed_essential_conditions.append('Price Below BB Middle')
             detail_bb_mid = f'Close:{current_price:.4f}, BB Middle:{last_row.get("bb_middle", np.nan):.4f}'
             signal_details['Below_BB_Middle'] = f'Failed: Not Below BB Middle ({detail_bb_mid})'
        else:
             signal_details['Below_BB_Middle'] = 'Passed'


        # 5. MACD Momentum Confirmation (Hist positive OR bullish cross)
        macd_hist_positive = pd.notna(last_row['macd_hist']) and last_row['macd_hist'] > 0
        macd_bullish_cross = pd.notna(last_row['macd']) and pd.notna(last_row['macd_signal']) and last_row['macd'] > last_row['macd_signal']

        if not (macd_hist_positive or macd_bullish_cross):
             essential_passed = False
             failed_essential_conditions.append('MACD (Momentum)')
             detail_macd = f'Hist Pos: {macd_hist_positive}, Bullish Cross: {macd_bullish_cross}'
             signal_details['MACD_Momentum'] = f'Failed: No Positive Hist OR Bullish Cross ({detail_macd})'
        else:
             signal_details['MACD_Momentum'] = 'Passed'


        # 6. RSI Filtered (Above 50 but below 65 - indicates bullish bias without being overextended)
        if not (pd.notna(last_row['rsi']) and last_row['rsi'] > 50 and last_row['rsi'] < 65):
             essential_passed = False
             failed_essential_conditions.append('RSI Filter (50-65)')
             detail_rsi = f'RSI:{last_row.get("rsi", np.nan):.1f}'
             signal_details['RSI_Filtered'] = f'Failed: Not in 50-65 Range ({detail_rsi})'
        else:
             signal_details['RSI_Filtered'] = f'Passed: In 50-65 Range (RSI:{last_row["rsi"]:.1f})'


        # If any mandatory condition failed, reject the signal immediately
        if not essential_passed:
            logger.debug(f"ℹ️ [Strategy {self.symbol}] Mandatory conditions failed: {', '.join(failed_essential_conditions)}. Signal rejected.")
            # Can add failed condition details to the log here if needed
            return None
        # =====================================================================


        # =====================================================================
        # --- Calculate Score for Optional Conditions (if mandatory passed) ---
        # حساب نقاط الشروط الاختيارية
        # =====================================================================
        current_score = 0.0
        signal_details['Optional_Scores'] = {} # To log scores from optional conditions


        # RSI between 55 and 65 (Already checked as mandatory, but can still contribute points if desired, or adjust range)
        # Let's refine: if RSI is between 55 and 65, add points. If it's >65 but <70 (still acceptable but less ideal for pullback), add fewer points?
        # Keeping the simpler check: if RSI is in the tighter range (55-65) add the score.
        if pd.notna(last_row['rsi']) and last_row['rsi'] >= 55 and last_row['rsi'] <= 65:
             score_val = self.condition_weights.get('rsi_mid_range', 0)
             current_score += score_val
             signal_details['Optional_Scores']['RSI_Mid_Range'] = f'RSI ({last_row["rsi"]:.1f}) in 55-65 Range (+{score_val})'
        else:
             signal_details['Optional_Scores']['RSI_Mid_Range'] = f'RSI ({last_row["rsi"]:.1f}) Not in 55-65 Range (0)'


        # Bullish candlestick pattern present at or near pullback zone
        # Check if BullishCandleSignal is True AND the candle low is near VWMA or BB Lower
        candle_low_near_vwma = pd.notna(last_row['low']) and pd.notna(last_row['vwma']) and abs(last_row['low'] - last_row['vwma']) <= vwma_tolerance
        candle_low_near_bb_lower = pd.notna(last_row['low']) and pd.notna(last_row['bb_lower']) and abs(last_row['low'] - last_row['bb_lower']) <= bb_lower_tolerance

        if last_row['BullishCandleSignal'] and (candle_low_near_vwma or candle_low_near_bb_lower):
             score_val = self.condition_weights.get('bullish_candle_at_pullback', 0)
             current_score += score_val
             signal_details['Optional_Scores']['Bullish_Candle_at_Pullback'] = f'Bullish Pattern Present Near Pullback Zone (+{score_val})'
        else:
             signal_details['Optional_Scores']['Bullish_Candle_at_Pullback'] = 'No Bullish Pattern or Not Near Pullback Zone (0)'


        # OBV is rising (current OBV > previous OBV)
        if pd.notna(last_row['obv']) and pd.notna(prev_row.get('obv')) and last_row['obv'] > prev_row.get('obv'):
            score_val = self.condition_weights.get('obv_rising', 0)
            current_score += score_val
            signal_details['Optional_Scores']['OBV_Rising'] = f'Rising (+{score_val})'
        else:
             signal_details['Optional_Scores']['OBV_Rising'] = 'Not Rising (0)'


        # Price explicitly touches/closes very near Lower BB
        if pd.notna(last_row['bb_lower']) and current_price <= last_row['bb_lower'] * (1 + self.vwma_bb_tolerance_pct) and current_price >= last_row['bb_lower'] * (1 - self.vwma_bb_tolerance_pct):
             score_val = self.condition_weights.get('price_touches_bb_lower', 0)
             current_score += score_val
             signal_details['Optional_Scores']['Price_Touches_BB_Lower'] = f'Touches BB Lower (+{score_val})'
        else:
             signal_details['Optional_Scores']['Price_Touches_BB_Lower'] = 'Does Not Touch BB Lower (0)'


        # Price shows bounce sign from VWMA
        # Check if price touched VWMA and the current candle closed higher than its low
        price_touched_vwma = pd.notna(last_row['vwma']) and last_row['low'] <= last_row['vwma'] * (1 + self.vwma_bb_tolerance_pct) and last_row['high'] >= last_row['vwma'] * (1 - self.vwma_bb_tolerance_pct) # Did any part of the candle touch VWMA zone
        closed_above_low = pd.notna(last_row['close']) and pd.notna(last_row['low']) and last_row['close'] > last_row['low']

        if price_touched_vwma and closed_above_low:
             score_val = self.condition_weights.get('price_bounces_from_vwma', 0)
             current_score += score_val
             signal_details['Optional_Scores']['Price_Bounces_from_VWMA'] = f'Touched VWMA and Closed Above Low (+{score_val})'
        else:
             signal_details['Optional_Scores']['Price_Bounces_from_VWMA'] = 'No Bounce Sign from VWMA (0)'


        # MACD histogram is positive (score, if not already covered by mandatory or wants more points)
        # We already have a mandatory check for positive or cross. This adds points specifically for positive hist.
        if pd.notna(last_row['macd_hist']) and last_row['macd_hist'] > 0:
             score_val = self.condition_weights.get('macd_hist_positive_score', 0)
             current_score += score_val
             signal_details['Optional_Scores']['MACD_Hist_Positive'] = f'MACD Hist Positive (+{score_val})'
        else:
             signal_details['Optional_Scores']['MACD_Hist_Positive'] = 'MACD Hist Not Positive (0)'


        # Optional: Price above Daily VWAP
        # if pd.notna(last_row['vwap']) and last_row['close'] > last_row['vwap']:
        #      score_val = self.condition_weights.get('above_vwap_optional', 0)
        #      current_score += score_val
        #      signal_details['Optional_Scores']['Above_Daily_VWAP'] = f'Above Daily VWAP (+{score_val})'
        # else:
        #      signal_details['Optional_Scores']['Above_Daily_VWAP'] = 'Not Above Daily VWAP (0)'


        # ------------------------------------------

        # Final buy decision based on the score of optional conditions
        if current_score < self.min_signal_score:
            logger.debug(f"ℹ️ [Strategy {self.symbol}] Required signal score from optional conditions not met (Score: {current_score:.2f} / {self.total_possible_score:.2f}, Threshold: {self.min_signal_score:.2f}). Signal rejected.")
            signal_details['Score_Met'] = f'Failed (Score: {current_score:.2f} < Threshold: {self.min_signal_score:.2f})'
            return None
        else:
             signal_details['Score_Met'] = f'Passed (Score: {current_score:.2f} >= Threshold: {self.min_signal_score:.2f})'


        # Check trading volume (liquidity) - still a mandatory filter
        volume_recent = fetch_recent_volume(self.symbol)
        if volume_recent < MIN_VOLUME_15M_USDT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] Liquidity ({volume_recent:,.0f} USDT) is below the minimum threshold ({MIN_VOLUME_15M_USDT:,.0f} USDT). Signal rejected.")
            signal_details['Volume_Check'] = f'Failed (Volume: {volume_recent:,.0f} < Min: {MIN_VOLUME_15M_USDT:,.0f})'
            return None
        else:
             signal_details['Volume_Check'] = f'Passed (Volume: {volume_recent:,.0f} >= Min: {MIN_VOLUME_15M_USDT:,.0f})'


        # Calculate initial target and stop loss based on ATR and the strategy's specific multipliers
        current_atr = last_row.get('atr')

        # Ensure ATR is not NaN before using it
        if pd.isna(current_atr) or current_atr <= 0:
             logger.warning(f"⚠️ [Strategy {self.symbol}] Invalid ATR value ({current_atr}) for calculating target and stop loss.")
             signal_details['ATR_Check'] = 'Failed (Invalid ATR)'
             return None
        else:
             signal_details['ATR_Check'] = f'Passed (ATR: {current_atr:.6f})'


        initial_target = current_price + (self.initial_tp_multiplier * current_atr)
        initial_stop_loss = current_price - (self.initial_sl_multiplier * current_atr)


        # Ensure stop loss is not zero or negative and is below the entry price
        if initial_stop_loss <= 0 or initial_stop_loss >= current_price:
            # Use a percentage as a minimum stop loss if the initial calculation is invalid
            # Example: 1.5% below current price as a minimum
            min_sl_price_pct = current_price * (1 - 0.015) # Example: 1.5% below entry
            initial_stop_loss = max(min_sl_price_pct, current_price * 0.0001) # Ensure it's not too close to zero
            logger.warning(f"⚠️ [Strategy {self.symbol}] Calculated stop loss ({initial_stop_loss:.8g}) is invalid or above entry price. Adjusted to {initial_stop_loss:.8f}")
            signal_details['SL_Adjustment'] = f'Initial SL adjusted (was <= 0 or >= entry, set to {initial_stop_loss:.8f})'
        else:
             # Ensure the initial stop loss is not too wide (optional filter)
             # Example: Initial loss should not exceed 10% (or adjust based on strategy testing)
             max_allowed_loss_pct = 0.10
             max_sl_price = current_price * (1 - max_allowed_loss_pct)
             if initial_stop_loss < max_sl_price:
                  logger.warning(f"⚠️ [Strategy {self.symbol}] Calculated stop loss ({initial_stop_loss:.8g}) is too wide. Adjusted to {max_sl_price:.8f}")
                  initial_stop_loss = max_sl_price
                  signal_details['SL_Adjustment'] = f'Initial SL adjusted (was too wide, set to {initial_stop_loss:.8f})' # Use the new value here


        # Check minimum profit margin (after calculating final target and stop loss) - still a mandatory filter
        profit_margin_pct = ((initial_target / current_price) - 1) * 100 if current_price > 0 else 0
        if profit_margin_pct < MIN_PROFIT_MARGIN_PCT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] Profit margin ({profit_margin_pct:.2f}%) is below the minimum required ({MIN_PROFIT_MARGIN_PCT:.2f}%). Signal rejected.")
            signal_details['Min_Profit_Margin_Check'] = f'Failed ({profit_margin_pct:.2f}% < Min: {MIN_PROFIT_MARGIN_PCT:.2f}%)'
            return None
        else:
             signal_details['Min_Profit_Margin_Check'] = f'Passed ({profit_margin_pct:.2f}% >= Min: {MIN_PROFIT_MARGIN_PCT:.2f}%)'


        # Compile final signal data
        signal_output = {
            'symbol': self.symbol,
            'entry_price': float(f"{current_price:.8g}"), # Use g format for precision display
            'initial_target': float(f"{initial_target:.8g}"),
            'initial_stop_loss': float(f"{initial_stop_loss:.8g}"),
            'current_target': float(f"{initial_target:.8g}"), # Current starts as initial
            'current_stop_loss': float(f"{initial_stop_loss:.8g}"), # Current starts as initial
            'r2_score': float(f"{current_score:.2f}"), # Weighted score of optional conditions
            'strategy_name': self.strategy_name,
            'signal_details': signal_details, # Now contains details of mandatory and optional conditions
            'volume_15m': volume_recent,
            'trade_value': TRADE_VALUE,
            'total_possible_score': float(f"{self.total_possible_score:.2f}") # Total points for optional conditions
        }

        logger.info(f"✨ [Strategy {self.symbol}] Confirmed buy signal ({self.strategy_name}). Price: {current_price:.6f}, Score (Optional): {current_score:.2f}/{self.total_possible_score:.2f}, ATR: {current_atr:.6f}, Volume: {volume_recent:,.0f}")
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
         logger.error(f"❌ [Telegram] Failed to send message to {target_chat_id} (Timeout).")
         return None
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"❌ [Telegram] Failed to send message to {target_chat_id} (HTTP Error: {http_err.response.status_code}).")
        try:
            error_details = http_err.response.json()
            logger.error(f"❌ [Telegram] API error details: {error_details}")
        except json.JSONDecodeError:
            logger.error(f"❌ [Telegram] Could not decode error response: {http_err.response.text}")
        return None
    except requests.exceptions.RequestException as req_err:
        logger.error(f"❌ [Telegram] Failed to send message to {target_chat_id} (Request Error): {req_err}")
        return None
    except Exception as e:
         logger.error(f"❌ [Telegram] Unexpected error sending message: {e}", exc_info=True)
         return None

def send_telegram_alert(signal_data: Dict[str, Any], timeframe: str) -> None:
    """Formats and sends a new trading signal alert to Telegram in Arabic, displaying the score."""
    logger.debug(f"ℹ️ [Telegram Alert] Formatting and sending alert for signal: {signal_data.get('symbol', 'N/A')}")
    try:
        entry_price = float(signal_data['entry_price'])
        target_price = float(signal_data['initial_target'])
        stop_loss_price = float(signal_data['initial_stop_loss'])
        symbol = signal_data['symbol']
        strategy_name = signal_data.get('strategy_name', 'N/A')
        signal_score = signal_data.get('r2_score', 0.0) # Weighted score of optional conditions
        total_possible_score = signal_data.get('total_possible_score', 0.0) # Total points for optional conditions
        volume_15m = signal_data.get('volume_15m', 0.0)
        trade_value_signal = signal_data.get('trade_value', TRADE_VALUE)
        signal_details = signal_data.get('signal_details', {}) # Details dictionary

        # Calculate profit/loss percentages and USDT values
        profit_pct = ((target_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        loss_pct = ((stop_loss_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        profit_usdt = trade_value_signal * (profit_pct / 100)
        loss_usdt = abs(trade_value_signal * (loss_pct / 100))

        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Escape special characters for Markdown v2 (Telegram default)
        # Characters needing escaping: _, *, [, ], (, ), ~, `, >, #, +, -, =, |, {, }, ., !
        def escape_markdown_v2(text: str) -> str:
             if not isinstance(text, str):
                 return str(text) # Convert non-strings to string
             escape_chars = '_*[]()~`>#+-=|{}.!'
             return ''.join('\\' + char if char in escape_chars else char for char in text)

        safe_symbol = escape_markdown_v2(symbol)
        safe_strategy_name = escape_markdown_v2(strategy_name.replace('_', ' ').title())

        fear_greed = get_fear_greed_index()
        btc_trend = get_btc_trend_4h()

        # Build the message in Arabic with score and condition details
        message = (
            f"💡 *إشارة تداول جديدة ({safe_strategy_name})* 💡\n"
            f"—————————————————\n" # Adjusted line length
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"📈 **نوع الإشارة:** شراء (طويل)\n"
            f"🕰️ **الإطار الزمني:** {timeframe}\n"
            # --- Display Mandatory Condition Summary ---
            f"✅ *الشروط الإلزامية:*\n"
            f"  - الاتجاه الصاعد العام (EMA/ST): {signal_details.get('Overall_Bullish_Trend', 'N/A').replace('Passed', 'متحقق ✅').replace('Failed', 'غير متحقق ❌')}\n"
            f"  - مؤشر الاتجاه (ADX/DI): {signal_details.get('ADX/DI', 'N/A').replace('Passed', 'متحقق ✅').replace('Failed', 'غير متحقق ❌')}\n"
            f"  - منطقة الارتداد (VWMA/BB): {signal_details.get('Pullback_Zone', 'N/A').replace('Passed', 'متحقق ✅').replace('Failed', 'غير متحقق ❌')}\n"
            f"  - السعر تحت الحد الأوسط لبولينجر: {signal_details.get('Below_BB_Middle', 'N/A').replace('Passed', 'متحقق ✅').replace('Failed', 'غير متحقق ❌')}\n"
            f"  - زخم MACD: {signal_details.get('MACD_Momentum', 'N/A').replace('Passed', 'متحقق ✅').replace('Failed', 'غير متحقق ❌')}\n"
            f"  - فلتر RSI (50-65): {signal_details.get('RSI_Filtered', 'N/A').replace('Passed', 'متحقق ✅').replace('Failed', 'غير متحقق ❌')}\n"
            f"  - فحص السيولة (15 دقيقة): {signal_details.get('Volume_Check', 'N/A').replace('Passed', 'متحقق ✅').replace('Failed', 'غير متحقق ❌')}\n"
            f"  - الحد الأدنى للربح: {signal_details.get('Min_Profit_Margin_Check', 'N/A').replace('Passed', 'متحقق ✅').replace('Failed', 'غير متحقق ❌')}\n"

            # --- Display Optional Condition Score ---
            f"📊 **قوة الإشارة (نقاط إضافية):** *{signal_score:.1f} / {total_possible_score:.1f}*\n"
            # Optionally list which optional conditions contributed points
            # if signal_details.get('Optional_Scores'):
            #      optional_score_details = ", ".join([f"{k}: {v}" for k, v in signal_details['Optional_Scores'].items()])
            #      message += f"  _تفاصيل النقاط:_ {optional_score_details}\n"
            f"—————————————————\n" # Adjusted line length
            f"➡️ **سعر الدخول المقترح:** `${entry_price:,.8g}`\n"
            f"🎯 **الهدف الأولي:** `${target_price:,.8g}` ({profit_pct:+.2f}% / ≈ ${profit_usdt:+.2f})\n"
            f"🛑 **وقف الخسارة الأولي:** `${stop_loss_price:,.8g}` ({loss_pct:.2f}% / ≈ ${loss_usdt:.2f})\n" # Use loss_usdt variable
            f"—————————————————\n" # Adjusted line length
            f"😨/🤑 **مؤشر الخوف والجشع:** {fear_greed}\n"
            f"₿ **اتجاه البيتكوين (4 ساعات):** {btc_trend}\n"
            f"—————————————————\n" # Adjusted line length
            f"⏰ {timestamp_str}"
        )

        reply_markup = {
            "inline_keyboard": [
                [{"text": "📊 عرض تقرير الأداء", "callback_data": "get_report"}]
            ]
        }

        send_telegram_message(CHAT_ID, message, reply_markup=reply_markup, parse_mode='MarkdownV2') # Use MarkdownV2

    except KeyError as ke:
        logger.error(f"❌ [Telegram Alert] Signal data incomplete for symbol {signal_data.get('symbol', 'N/A')}: Missing key {ke}", exc_info=True)
        # Fallback: send a basic error message
        send_telegram_message(CHAT_ID, f"❌ خطأ في تنسيق إشارة التداول لـ {signal_data.get('symbol', 'N/A')}. التفاصيل مفقودة: {ke}")
    except Exception as e:
        logger.error(f"❌ [Telegram Alert] Failed to send signal alert for symbol {signal_data.get('symbol', 'N/A')}: {e}", exc_info=True)
        # Fallback: send a basic error message
        send_telegram_message(CHAT_ID, f"❌ حدث خطأ أثناء إرسال إشارة التداول لـ {signal_data.get('symbol', 'N/A')}.")


def send_tracking_notification(details: Dict[str, Any]) -> None:
    """Formats and sends enhanced Telegram notifications for tracking events in Arabic."""
    symbol = details.get('symbol', 'N/A')
    signal_id = details.get('id', 'N/A')
    notification_type = details.get('type', 'unknown')
    message = ""
    # Escape special characters for Markdown v2
    def escape_markdown_v2(text: str) -> str:
         if not isinstance(text, str):
             return str(text) # Convert non-strings to string
         escape_chars = '_*[]()~`>#+-=|{}.!'
         return ''.join('\\' + char if char in escape_chars else char for char in text)

    safe_symbol = escape_markdown_v2(symbol)


    closing_price = details.get('closing_price', 0.0)
    profit_pct = details.get('profit_pct', 0.0)
    current_price = details.get('current_price', 0.0)
    atr_value = details.get('atr_value', 0.0)
    new_stop_loss = details.get('new_stop_loss', 0.0)
    old_stop_loss = details.get('old_stop_loss', 0.0)
    strategy_name = details.get('strategy_name', 'N/A') # Get strategy name for notification
    last_trailing_update_price = details.get('last_trailing_update_price', None) # Get the price from the details


    logger.debug(f"ℹ️ [Notification] Formatting tracking notification: ID={signal_id}, Type={notification_type}, Symbol={symbol}")

    # Format numeric values using 'g' or fixed point as appropriate
    closing_price_str = f"{closing_price:,.8g}" if closing_price else "N/A"
    profit_pct_str = f"{profit_pct:+.2f}%" if profit_pct is not None else "N/A"
    current_price_str = f"{current_price:,.8g}" if current_price else "N/A"
    atr_value_str = f"{atr_value:,.8g}" if atr_value else "N/A"
    new_stop_loss_str = f"{new_stop_loss:,.8g}" if new_stop_loss else "N/A"
    old_stop_loss_str = f"{old_stop_loss:,.8g}" if old_stop_loss else "N/A"


    if notification_type == 'target_hit':
        message = (
            f"✅ *تم الوصول إلى الهدف ({escape_markdown_v2(strategy_name)})*\n"
            f"—————————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"🎯 **سعر الإغلاق (الهدف):** `${closing_price_str}`\n"
            f"💰 **الربح المحقق:** {profit_pct_str}"
        )
    elif notification_type == 'stop_loss_hit':
        sl_type_msg_ar = "بربح ✅" if details.get('profitable_sl', False) else "بخسارة ❌" # Use profitable_sl flag
        message = (
            f"🛑 *تم ضرب وقف الخسارة ({escape_markdown_v2(strategy_name)})*\n"
            f"—————————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"🚫 **سعر الإغلاق (الوقف):** `${closing_price_str}`\n"
            f"📉 **النتيجة:** {profit_pct_str} ({sl_type_msg_ar})"
        )
    elif notification_type == 'trailing_activated':
        activation_profit_pct = details.get('activation_profit_pct', TRAILING_STOP_ACTIVATION_PROFIT_PCT * 100)
        message = (
            f"⬆️ *تم تفعيل وقف الخسارة المتحرك ({escape_markdown_v2(strategy_name)})*\n"
            f"—————————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"📈 **السعر الحالي (عند التفعيل):** `${current_price_str}` (الربح \> {activation_profit_pct:.1f}%)\n"
            f"📊 **قيمة ATR ({ENTRY_ATR_PERIOD}):** `{atr_value_str}` (المضاعف: {TRAILING_STOP_ATR_MULTIPLIER})\n"
            f"🛡️ **وقف الخسارة الجديد:** `${new_stop_loss_str}`"
        )
    elif notification_type == 'trailing_updated':
        # Calculate the percentage change since the last trailing stop update price
        # Ensure last_trailing_update_price is not None or zero to avoid division errors
        price_change_since_last_update_pct = ((current_price - last_trailing_update_price) / last_trailing_update_price) * 100 if last_trailing_update_price and last_trailing_update_price != 0 else 0.0

        # Build the percentage string with the conditional backslash separately
        percentage_str = f"{price_change_since_last_update_pct:+.1f}%"
        if price_change_since_last_update_pct > 0:
            percentage_str = '\\' + percentage_str # Add backslash if positive for MarkdownV2 escaping

        message = (
            f"➡️ *تم تحديث وقف الخسارة المتحرك ({escape_markdown_v2(strategy_name)})*\n"
            f"—————————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            # Embed the pre-built percentage string
            f"📈 **السعر الحالي (عند التحديث):** `${current_price_str}` ({percentage_str} منذ آخر تحديث)\n"
            f"📊 **قيمة ATR ({ENTRY_ATR_PERIOD}):** `{atr_value_str}` (المضاعف: {TRAILING_STOP_ATR_MULTIPLIER})\n"
            f"🔒 **الوقف السابق:** `${old_stop_loss_str}`\n"
            f"🛡️ **وقف الخسارة الجديد:** `${new_stop_loss_str}`"
        )
    else:
        logger.warning(f"⚠️ [Notification] Unknown notification type: {notification_type} for details: {details}")
        return # Don't send anything if type is unknown

    if message:
        # Use chat_id from details if available, otherwise use global CHAT_ID
        target_chat = details.get('chat_id', CHAT_ID)
        send_telegram_message(target_chat, message, parse_mode='MarkdownV2') # Use MarkdownV2

# ---------------------- Database Functions (Insert and Update) ----------------------
def insert_signal_into_db(signal: Dict[str, Any]) -> bool:
    """Inserts a new signal into the signals table with the weighted score."""
    if not check_db_connection() or not conn:
        logger.error(f"❌ [DB Insert] Failed to insert signal {signal.get('symbol', 'N/A')} due to DB connection issue.")
        return False

    symbol = signal.get('symbol', 'N/A')
    logger.debug(f"ℹ️ [DB Insert] Attempting to insert signal for {symbol}...")
    try:
        # Convert signal data to a format suitable for DB (Python native types)
        signal_prepared = convert_np_values(signal)
        # Convert signal details to JSON (ensure it doesn't contain numpy types)
        signal_details_json = json.dumps(signal_prepared.get('signal_details', {}))

        with conn.cursor() as cur_ins:
            insert_query = sql.SQL("""
                INSERT INTO signals
                 (symbol, entry_price, initial_target, initial_stop_loss, current_target, current_stop_loss,
                 r2_score, strategy_name, signal_details, last_trailing_update_price, volume_15m, total_possible_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id; -- Return the new signal ID
            """)
            cur_ins.execute(insert_query, (
                signal_prepared.get('symbol'),
                signal_prepared.get('entry_price'),
                signal_prepared.get('initial_target'),
                signal_prepared.get('initial_stop_loss'),
                signal_prepared.get('current_target'),
                signal_prepared.get('current_stop_loss'),
                signal_prepared.get('r2_score'),
                signal_prepared.get('strategy_name', 'unknown'),
                signal_details_json,
                signal_prepared.get('last_trailing_update_price'), # Should be None initially
                signal_prepared.get('volume_15m'),
                signal_prepared.get('total_possible_score') # Insert total possible score
            ))
            # Fetch the ID of the newly inserted row
            signal_id = cur_ins.fetchone()['id']
        conn.commit()
        logger.info(f"✅ [DB Insert] Signal for {symbol} inserted into database (ID: {signal_id}, Strategy: {signal_prepared.get('strategy_name')}, Score: {signal_prepared.get('r2_score')}).")
        return True
    except psycopg2.Error as db_err:
        logger.error(f"❌ [DB Insert] Database error inserting signal for {symbol}: {db_err}")
        if conn: conn.rollback()
        return False
    except (TypeError, ValueError) as convert_err:
         logger.error(f"❌ [DB Insert] Error converting signal data before insertion for {symbol}: {convert_err} - Signal Data: {signal}")
         if conn: conn.rollback()
         return False
    except Exception as e:
        logger.error(f"❌ [DB Insert] Unexpected error inserting signal for {symbol}: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

# ---------------------- Open Signal Tracking Function ----------------------
def track_signals() -> None:
    """Tracks open signals, checks targets and stop losses, and applies trailing stop."""
    logger.info("ℹ️ [Tracker] Starting open signal tracking process...")
    while True:
        active_signals_summary: List[str] = []
        processed_in_cycle = 0
        try:
            if not check_db_connection() or not conn:
                logger.warning("⚠️ [Tracker] Skipping tracking cycle due to DB connection issue.")
                time.sleep(15) # Wait a bit longer before retrying
                continue

            # Use a cursor with context manager to fetch open signals
            with conn.cursor() as track_cur: # Uses RealDictCursor
                 track_cur.execute("""
                    SELECT id, symbol, entry_price, initial_stop_loss, current_target, current_stop_loss,
                           is_trailing_active, last_trailing_update_price, strategy_name
                    FROM signals
                    WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;
                """)
                 open_signals: List[Dict] = track_cur.fetchall()

            if not open_signals:
                # logger.debug("ℹ️ [Tracker] No open signals to track.")
                time.sleep(5) # Wait less if no signals
                continue

            logger.debug(f"ℹ️ [Tracker] Tracking {len(open_signals)} open signals...")

            for signal_row in open_signals:
                signal_id = signal_row['id']
                symbol = signal_row['symbol']
                strategy_name = signal_row['strategy_name'] # Get strategy name
                processed_in_cycle += 1
                update_executed = False # To track if this signal was updated in the current cycle

                try:
                    # Extract and safely convert numeric data
                    entry_price = float(signal_row['entry_price'])
                    initial_stop_loss = float(signal_row['initial_stop_loss'])
                    current_target = float(signal_row['current_target'])
                    current_stop_loss = float(signal_row['current_stop_loss'])
                    is_trailing_active = signal_row['is_trailing_active']
                    last_update_px = signal_row.get('last_trailing_update_price') # Use .get for safety
                    last_trailing_update_price = float(last_update_px) if last_update_px is not None else None

                    # Get current price from WebSocket Ticker data
                    current_price = ticker_data.get(symbol)

                    if current_price is None:
                         # logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): Current price not available in Ticker data. Skipping for this cycle.")
                         continue # Skip this signal in this cycle if price is missing

                    active_signals_summary.append(f"{symbol}({signal_id}): P={current_price:.4f} T={current_target:.4f} SL={current_stop_loss:.4f} Trail={'On' if is_trailing_active else 'Off'}")

                    update_query: Optional[sql.SQL] = None
                    update_params: Tuple = ()
                    log_message: Optional[str] = None
                    # Include strategy_name in notification details
                    notification_details: Dict[str, Any] = {'symbol': symbol, 'id': signal_id, 'strategy_name': strategy_name}


                    # --- Check and Update Logic ---
                    # 1. Check for Target Hit
                    if current_price >= current_target:
                        profit_pct = ((current_target / entry_price) - 1) * 100 if entry_price > 0 else 0
                        update_query = sql.SQL("UPDATE signals SET achieved_target = TRUE, closing_price = %s, closed_at = NOW(), profit_percentage = %s WHERE id = %s;")
                        update_params = (current_target, profit_pct, signal_id)
                        log_message = f"🎯 [Tracker] {symbol}(ID:{signal_id},{strategy_name}): Target reached at {current_target:.8g} (Profit: {profit_pct:+.2f}%)."
                        notification_details.update({'type': 'target_hit', 'closing_price': current_target, 'profit_pct': profit_pct})
                        update_executed = True

                    # 2. Check for Stop Loss Hit (Must be after Target check)
                    elif current_price <= current_stop_loss:
                        loss_pct = ((current_stop_loss / entry_price) - 1) * 100 if entry_price > 0 else 0
                        profitable_sl = current_stop_loss > entry_price
                        sl_type_msg = "at a profit ✅" if profitable_sl else "at a loss ❌"
                        update_query = sql.SQL("UPDATE signals SET hit_stop_loss = TRUE, closing_price = %s, closed_at = NOW(), profit_percentage = %s, profitable_stop_loss = %s WHERE id = %s;")
                        update_params = (current_stop_loss, loss_pct, profitable_sl, signal_id)
                        log_message = f"🔻 [Tracker] {symbol}(ID:{signal_id},{strategy_name}): Stop Loss hit ({sl_type_msg}) at {current_stop_loss:.8g} (Percentage: {loss_pct:.2f}%)."
                        notification_details.update({'type': 'stop_loss_hit', 'closing_price': current_stop_loss, 'profit_pct': loss_pct, 'profitable_sl': profitable_sl}) # Pass the profitable_sl flag
                        update_executed = True

                    # 3. Check for Trailing Stop Activation or Update (Only if Target or SL not hit)
                    else:
                        activation_threshold_price = entry_price * (1 + TRAILING_STOP_ACTIVATION_PROFIT_PCT)
                        # a. Activate Trailing Stop
                        if not is_trailing_active and current_price >= activation_threshold_price:
                            logger.info(f"ℹ️ [Tracker] {symbol}(ID:{signal_id},{strategy_name}): Price {current_price:.8g} reached trailing activation threshold ({activation_threshold_price:.8g}). Fetching ATR...")
                            # Use the specified tracking timeframe
                            df_atr = fetch_historical_data(symbol, interval=SIGNAL_TRACKING_TIMEFRAME, days=SIGNAL_TRACKING_LOOKBACK_DAYS)
                            if df_atr is not None and not df_atr.empty:
                                # Use the ATR period designated for entry/tracking
                                df_atr = calculate_atr_indicator(df_atr, period=ENTRY_ATR_PERIOD)
                                if not df_atr.empty and 'atr' in df_atr.columns and pd.notna(df_atr['atr'].iloc[-1]):
                                    current_atr_val = df_atr['atr'].iloc[-1]
                                    if current_atr_val > 0:
                                         # New trailing stop is Current Price - (ATR Multiplier * ATR Value)
                                         # Ensure the new stop is not below the initial stop loss
                                         # Ensure the new stop is not below entry price * 1.001 (a tiny profit)
                                         new_stop_loss_calc = current_price - (TRAILING_STOP_ATR_MULTIPLIER * current_atr_val)
                                         new_stop_loss = max(new_stop_loss_calc, initial_stop_loss, entry_price * (1 + 0.001)) # Ensure a very small profit or keep initial stop if higher


                                         if new_stop_loss > current_stop_loss: # Only update if the new stop is actually higher than the current stop
                                            update_query = sql.SQL("UPDATE signals SET is_trailing_active = TRUE, current_stop_loss = %s, last_trailing_update_price = %s WHERE id = %s;")
                                            update_params = (new_stop_loss, current_price, signal_id)
                                            log_message = f"⬆️✅ [Tracker] {symbol}(ID:{signal_id},{strategy_name}): Trailing stop activated. Price={current_price:.8g}, ATR={current_atr_val:.8g}. New Stop: {new_stop_loss:.8g}"
                                            notification_details.update({'type': 'trailing_activated', 'current_price': current_price, 'atr_value': current_atr_val, 'new_stop_loss': new_stop_loss, 'activation_profit_pct': TRAILING_STOP_ACTIVATION_PROFIT_PCT * 100})
                                            update_executed = True
                                         else:
                                            logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id},{strategy_name}): Calculated trailing stop ({new_stop_loss:.8g}) is not higher than current stop ({current_stop_loss:.8g}). Not activating.")
                                    else: logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id},{strategy_name}): Invalid ATR value ({current_atr_val}) for trailing stop activation.")
                                else: logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id},{strategy_name}): Cannot calculate ATR for trailing stop activation.")
                            else: logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id},{strategy_name}): Cannot fetch data to calculate ATR for trailing stop activation.")

                        # b. Update Trailing Stop (if active)
                        elif is_trailing_active and last_trailing_update_price is not None:
                            # Calculate the threshold price for updating the trailing stop
                            update_threshold_price = last_trailing_update_price * (1 + TRAILING_STOP_MOVE_INCREMENT_PCT)
                            if current_price >= update_threshold_price:
                                logger.info(f"ℹ️ [Tracker] {symbol}(ID:{signal_id},{strategy_name}): Price {current_price:.8g} reached trailing update threshold ({update_threshold_price:.8g}). Fetching ATR...")
                                df_recent = fetch_historical_data(symbol, interval=SIGNAL_TRACKING_TIMEFRAME, days=SIGNAL_TRACKING_LOOKBACK_DAYS)
                                if df_recent is not None and not df_recent.empty:
                                    df_recent = calculate_atr_indicator(df_recent, period=ENTRY_ATR_PERIOD)
                                    if not df_recent.empty and 'atr' in df_recent.columns and pd.notna(df_recent['atr'].iloc[-1]):
                                         current_atr_val_update = df_recent['atr'].iloc[-1]
                                         if current_atr_val_update > 0:
                                             # Calculate potential new stop loss
                                             potential_new_stop_loss = current_price - (TRAILING_STOP_ATR_MULTIPLIER * current_atr_val_update)
                                             # The new stop loss must be higher than the current stop loss
                                             if potential_new_stop_loss > current_stop_loss:
                                                new_stop_loss_update = potential_new_stop_loss
                                                update_query = sql.SQL("UPDATE signals SET current_stop_loss = %s, last_trailing_update_price = %s WHERE id = %s;")
                                                update_params = (new_stop_loss_update, current_price, signal_id)
                                                log_message = f"➡️🔼 [Tracker] {symbol}(ID:{signal_id},{strategy_name}): Trailing stop updated. Price={current_price:.8g}, ATR={current_atr_val_update:.8g}. Old={current_stop_loss:.8g}, New: {new_stop_loss_update:.8g}"
                                                notification_details.update({'type': 'trailing_updated', 'current_price': current_price, 'atr_value': current_atr_val_update, 'old_stop_loss': current_stop_loss, 'new_stop_loss': new_stop_loss_update, 'last_trailing_update_price': last_trailing_update_price}) # Pass last_trailing_update_price
                                                update_executed = True
                                             else:
                                                 logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id},{strategy_name}): Calculated trailing stop ({potential_new_stop_loss:.8g}) is not higher than current ({current_stop_loss:.8g}). Not updating.")
                                         else: logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id},{strategy_name}): Invalid ATR value ({current_atr_val_update}) for update.")
                                    else: logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id},{strategy_name}): Cannot calculate ATR for update.")
                                else: logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id},{strategy_name}): Cannot fetch data to calculate ATR for update.")

                    # --- Execute Database Update and Send Notification ---
                    if update_executed and update_query:
                        try:
                             with conn.cursor() as update_cur:
                                  update_cur.execute(update_query, update_params)
                             conn.commit()
                             if log_message: logger.info(log_message)
                             if notification_details.get('type'):
                                send_tracking_notification(notification_details)
                        except psycopg2.Error as db_err:
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id},{strategy_name}): DB error during update: {db_err}")
                            if conn: conn.rollback()
                        except Exception as exec_err:
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id},{strategy_name}): Unexpected error during update execution/notification: {exec_err}", exc_info=True)
                            if conn: conn.rollback()

                except (TypeError, ValueError) as convert_err:
                    logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id},{strategy_name}): Error converting initial signal values: {convert_err}")
                    continue
                except Exception as inner_loop_err:
                     logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id},{strategy_name}): Unexpected error processing signal: {inner_loop_err}", exc_info=True)
                     continue

            if active_signals_summary:
                logger.debug(f"ℹ️ [Tracker] End of cycle status ({processed_in_cycle} processed): {'; '.join(active_signals_summary)}")

            time.sleep(5) # Wait between tracking cycles (Adjust if needed)

        except psycopg2.Error as db_cycle_err:
             logger.error(f"❌ [Tracker] Database error in main tracking cycle: {db_cycle_err}. Attempting to reconnect...")
             if conn: conn.rollback()
             time.sleep(30)
             check_db_connection()
        except Exception as cycle_err:
            logger.error(f"❌ [Tracker] Unexpected error in signal tracking cycle: {cycle_err}", exc_info=True)
            time.sleep(30)


# ---------------------- Flask Service (Optional for Webhook) ----------------------
app = Flask(__name__)

@app.route('/')
def home() -> Response:
    """Simple home page to show the bot is running."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ws_alive = ws_thread.is_alive() if 'ws_thread' in globals() and ws_thread else False
    tracker_alive = tracker_thread.is_alive() if 'tracker_thread' in globals() and tracker_thread else False
    flask_alive = flask_thread.is_alive() if 'flask_thread' in globals() and flask_thread else False

    status_parts = []
    status_parts.append(f"Trading Bot: {'Running ✅' if (ws_alive and tracker_alive and (not WEBHOOK_URL or flask_alive)) else 'Partial/Down ❌'}")
    status_parts.append(f"WebSocket: {'Active ✅' if ws_alive else 'Inactive ❌'}")
    status_parts.append(f"Signal Tracker: {'Active ✅' if tracker_alive else 'Inactive ❌'}")
    if WEBHOOK_URL:
         status_parts.append(f"Webhook Listener: {'Active ✅' if flask_alive else 'Inactive ❌'}")

    status_text = " | ".join(status_parts)

    return Response(f"📈 Crypto Signal Bot Status: {status_text} - Last Check: {now}", status=200, mimetype='text/plain')

@app.route('/favicon.ico')
def favicon() -> Response:
    """Handles favicon request to avoid 404 errors in logs."""
    return Response(status=204) # No Content

@app.route('/webhook', methods=['POST'])
def webhook() -> Tuple[str, int]:
    """Handles incoming requests from Telegram (like button presses and commands)."""
    if not request.is_json:
        logger.warning("⚠️ [Flask] Received non-JSON webhook request.")
        return "Invalid request format", 400 # Bad Request

    try:
        data = request.get_json()
        # logger.debug(f"ℹ️ [Flask] Received webhook data: {json.dumps(data)[:200]}...") # Log only part of the data, can be noisy

        # Handle Callback Queries (Button Responses)
        if 'callback_query' in data:
            callback_query = data['callback_query']
            callback_id = callback_query['id']
            callback_data = callback_query.get('data')
            message_info = callback_query.get('message')
            if not message_info or not callback_data:
                 logger.warning(f"⚠️ [Flask] Callback query (ID: {callback_id}) missing message or data.")
                 try:
                     # Acknowledge the callback query even if it's invalid
                     ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                     requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
                 except Exception as ack_err:
                     logger.warning(f"⚠️ [Flask] Failed to acknowledge invalid callback query {callback_id}: {ack_err}")
                 return "OK", 200
            # Check if chat_id is available before accessing it
            chat_id_callback = message_info.get('chat', {}).get('id')
            if not chat_id_callback:
                 logger.warning(f"⚠️ [Flask] Callback query (ID: {callback_id}) missing chat ID.")
                 try:
                     # Acknowledge the callback query even if chat_id is missing
                     ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                     requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
                 except Exception as ack_err:
                     logger.warning(f"⚠️ [Flask] Failed to acknowledge invalid callback query {callback_id}: {ack_err}")
                 return "OK", 200

            message_id = message_info['message_id']
            user_info = callback_query.get('from', {})
            user_id = user_info.get('id')
            username = user_info.get('username', 'N/A')

            logger.info(f"ℹ️ [Flask] Received callback query: Data='{callback_data}', User={username}({user_id}), Chat={chat_id_callback}")

            # Send acknowledgment quickly
            try:
                ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
            except Exception as ack_err:
                 logger.warning(f"⚠️ [Flask] Failed to acknowledge callback query {callback_id}: {ack_err}")

            # Process received button data
            if callback_data == "get_report":
                # Run report generation in a separate thread to avoid blocking webhook response
                report_thread = Thread(target=lambda: send_telegram_message(chat_id_callback, generate_performance_report(), parse_mode='Markdown')) # Report uses Markdown
                report_thread.start()
            else:
                logger.warning(f"⚠️ [Flask] Received unhandled callback data: '{callback_data}'")


        # Handle Text Messages (Commands)
        elif 'message' in data:
            message_data = data['message']
            chat_info = message_data.get('chat')
            user_info = message_data.get('from', {})
            text_msg = message_data.get('text', '').strip()

            if not chat_info or not text_msg:
                 logger.debug("ℹ️ [Flask] Received message without chat info or text.")
                 return "OK", 200

            chat_id_msg = chat_info['id']
            user_id = user_info.get('id')
            username = user_info.get('username', 'N/A')

            logger.info(f"ℹ️ [Flask] Received message: Text='{text_msg}', User={username}({user_id}), Chat={chat_id_msg}")

            # Process known commands
            if text_msg.lower() == '/report':
                 report_thread = Thread(target=lambda: send_telegram_message(chat_id_msg, generate_performance_report(), parse_mode='Markdown')) # Report uses Markdown
                 report_thread.start()
            elif text_msg.lower() == '/status':
                 status_thread = Thread(target=handle_status_command, args=(chat_id_msg,))
                 status_thread.start()
            # Add other commands here if needed

        else:
            # logger.debug("ℹ️ [Flask] Received webhook data without 'callback_query' or 'message'.") # Can be noisy
            pass # Ignore messages that are not text or callback_query

        return "OK", 200
    except Exception as e:
         logger.error(f"❌ [Flask] Error processing webhook: {e}", exc_info=True)
         # Optionally send an error message back to the user if chat_id is known and it's a message
         # if 'message' in data and data['message'].get('chat', {}).get('id'):
         #     send_telegram_message(data['message']['chat']['id'], "❌ حدث خطأ داخلي أثناء معالجة طلبك.")
         return "Internal Server Error", 500

def handle_status_command(chat_id_msg: int) -> None:
    """Separate function to handle /status command to avoid blocking the Webhook."""
    logger.info(f"ℹ️ [Flask Status] Handling /status command for chat {chat_id_msg}")
    status_msg = "⏳ جلب الحالة..."
    msg_sent = send_telegram_message(chat_id_msg, status_msg, parse_mode='Markdown') # Send initial message using Markdown
    if not (msg_sent and msg_sent.get('ok')):
         logger.error(f"❌ [Flask Status] Failed to send initial status message to {chat_id_msg}")
         return

    message_id_to_edit = msg_sent['result']['message_id']
    try:
        open_count = 0
        if check_db_connection() and conn:
            with conn.cursor() as status_cur:
                status_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;")
                open_count = (status_cur.fetchone() or {}).get('count', 0)

        # Check if threads exist before accessing them
        ws_alive = 'ws_thread' in globals() and ws_thread and ws_thread.is_alive()
        tracker_alive = 'tracker_thread' in globals() and tracker_thread and tracker_thread.is_alive()
        flask_alive = 'flask_thread' in globals() and flask_thread and flask_thread.is_alive()

        ws_status = 'نشط ✅' if ws_alive else 'غير نشط ❌'
        tracker_status = 'نشط ✅' if tracker_alive else 'غير نشط ❌'
        flask_status = 'نشط ✅' if (WEBHOOK_URL and flask_alive) else ('غير مفعل ℹ️' if not WEBHOOK_URL else 'غير نشط ❌')


        final_status_msg = (
            f"🤖 *حالة البوت:*\n"
            f"- تتبع الأسعار (WS): {ws_status}\n"
            f"- تتبع الإشارات: {tracker_status}\n"
            f"- الويب هوك/الخادم: {flask_status}\n"
            f"- الإشارات النشطة: *{open_count}* / {MAX_OPEN_TRADES}\n"
            f"- وقت الخادم الحالي: {datetime.now().strftime('%H:%M:%S UTC')}" # Use UTC time for clarity
        )
        # Edit the original message using MarkdownV2
        edit_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText"
        edit_payload = {
            'chat_id': chat_id_msg,
             'message_id': message_id_to_edit,
            'text': final_status_msg,
            'parse_mode': 'MarkdownV2' # Use MarkdownV2 for editing
        }
        response = requests.post(edit_url, json=edit_payload, timeout=10)
        response.raise_for_status()
        logger.info(f"✅ [Flask Status] Status updated for chat {chat_id_msg}")

    except Exception as status_err:
        logger.error(f"❌ [Flask Status] Error getting/editing status details for chat {chat_id_msg}: {status_err}", exc_info=True)
        # Attempt to send a new error message if editing failed
        send_telegram_message(chat_id_msg, "❌ حدث خطأ أثناء جلب تفاصيل الحالة.", parse_mode='Markdown')


def run_flask() -> None:
    """Runs the Flask application to listen for the Webhook using a production server if available."""
    if not WEBHOOK_URL:
        logger.info("ℹ️ [Flask] Webhook URL not configured. Flask server will not start.")
        return

    host = "0.0.0.0"
    port = int(config('PORT', default=10000)) # Use PORT environment variable or default value
    logger.info(f"ℹ️ [Flask] Starting Flask app on {host}:{port}...")
    try:
        from waitress import serve
        logger.info("✅ [Flask] Using 'waitress' server.")
        # Use threads for handling requests concurrently
        serve(app, host=host, port=port, threads=8) # Increased threads slightly
    except ImportError:
         logger.warning("⚠️ [Flask] 'waitress' not installed. Falling back to Flask development server (NOT recommended for production).")
         try:
             # Disable debug mode for potentially clearer logging in production fallback
             app.run(host=host, port=port, debug=False)
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
        # Optionally send a Telegram alert here
        send_telegram_message(CHAT_ID, "❌ البوت توقف عن العمل: لم يتم تحميل رموز عملات صالحة للفحص.")
        return

    logger.info(f"✅ [Main] Loaded {len(symbols_to_scan)} valid symbols for scanning.")
    # last_full_scan_time = time.time() # Not strictly needed for the current logic

    while True:
        try:
            scan_start_time = time.time()
            logger.info("+" + "-"*60 + "+")
            logger.info(f"🔄 [Main] Starting Market Scan Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}") # Use UTC
            logger.info("+" + "-"*60 + "+")

            if not check_db_connection() or not conn:
                logger.error("❌ [Main] Skipping scan cycle due to database connection failure.")
                time.sleep(60)
                continue

            # 1. Check the current number of open signals
            open_count = 0
            try:
                 with conn.cursor() as cur_check:
                    cur_check.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;")
                    open_count = (cur_check.fetchone() or {}).get('count', 0)
            except psycopg2.Error as db_err:
                 logger.error(f"❌ [Main] DB error checking open signal count: {db_err}. Skipping cycle.")
                 if conn: conn.rollback()
                 time.sleep(60)
                 continue

            logger.info(f"ℹ️ [Main] Currently Open Signals: {open_count} / {MAX_OPEN_TRADES}")
            if open_count >= MAX_OPEN_TRADES:
                logger.info(f"⚠️ [Main] Maximum number of open signals reached ({MAX_OPEN_TRADES}). Waiting for signals to close...")
                time.sleep(60) # Wait longer if max trades are open
                continue

            # 2. Iterate through the list of symbols and scan them
            processed_in_loop = 0
            signals_generated_in_loop = 0
            slots_available = MAX_OPEN_TRADES - open_count

            for symbol in symbols_to_scan:
                 if slots_available <= 0:
                      logger.info(f"ℹ️ [Main] Maximum limit ({MAX_OPEN_TRADES}) reached during scan. Stopping symbol scan for this cycle.")
                      break

                 processed_in_loop += 1
                 # logger.debug(f"🔍 [Main] Scanning {symbol} ({processed_in_loop}/{len(symbols_to_scan)})...") # Can be noisy

                 try:
                    # a. Check if there is already an open signal for this symbol
                    with conn.cursor() as symbol_cur:
                        symbol_cur.execute("SELECT 1 FROM signals WHERE symbol = %s AND achieved_target = FALSE AND hit_stop_loss = FALSE LIMIT 1;", (symbol,))
                        if symbol_cur.fetchone():
                            # logger.debug(f"ℹ️ [Main] Skipping {symbol}: Already has an open signal.") # Can be noisy
                            continue

                    # b. Fetch historical data
                    df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_hist is None or df_hist.empty:
                        # logger.debug(f"ℹ️ [Main] Skipping {symbol}: Failed to fetch historical data.") # Can be noisy
                        continue

                    # c. Apply the strategy and generate signal
                    # Use the NEW strategy class
                    strategy = MomentumPullbackStrategy(symbol)
                    df_indicators = strategy.populate_indicators(df_hist)
                    if df_indicators is None:
                        # logger.debug(f"ℹ️ [Main] Skipping {symbol}: Failed to populate indicators.") # Can be noisy
                        continue

                    potential_signal = strategy.generate_buy_signal(df_indicators)

                    # d. Insert signal and send alert
                    if potential_signal:
                        logger.info(f"✨ [Main] Potential signal found for {symbol} ({strategy.strategy_name})! Score: {potential_signal.get('r2_score', 0):.2f}. Final check and insertion...")
                        # Re-check open signal count just before inserting to avoid exceeding limit
                        with conn.cursor() as final_check_cur:
                             final_check_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;")
                             final_open_count = (final_check_cur.fetchone() or {}).get('count', 0)

                             if final_open_count < MAX_OPEN_TRADES:
                                 if insert_signal_into_db(potential_signal):
                                     send_telegram_alert(potential_signal, SIGNAL_GENERATION_TIMEFRAME)
                                     signals_generated_in_loop += 1
                                     slots_available -= 1
                                     time.sleep(2) # Small delay after sending alert
                                 else:
                                     logger.error(f"❌ [Main] Failed to insert signal for {symbol} into database.")
                             else:
                                 logger.warning(f"⚠️ [Main] Maximum limit ({final_open_count}) reached before inserting signal for {symbol}. Signal ignored.")
                                 break # Stop scanning symbols in this cycle

                 except psycopg2.Error as db_loop_err:
                      logger.error(f"❌ [Main] DB error processing symbol {symbol}: {db_loop_err}. Moving to next...")
                      if conn: conn.rollback()
                      continue
                 except Exception as symbol_proc_err:
                      logger.error(f"❌ [Main] General error processing symbol {symbol}: {symbol_proc_err}", exc_info=True)
                      continue

                 # Small delay between processing symbols to avoid hitting API limits rapidly
                 time.sleep(0.2) # Adjusted from 0.3

            # 3. Wait before starting the next cycle
            scan_duration = time.time() - scan_start_time
            logger.info(f"🏁 [Main] Scan cycle finished. Signals generated: {signals_generated_in_loop}. Scan duration: {scan_duration:.2f} seconds.")
            # Wait until the next full minute or a minimum time
            wait_time = max(60, 300 - scan_duration) # Wait 5 minutes total or at least 1 minute
            logger.info(f"⏳ [Main] Waiting {wait_time:.1f} seconds for the next cycle...")
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
                 # Send a critical alert via Telegram
                 send_telegram_message(CHAT_ID, f"❌ خطأ فادح في قاعدة البيانات. فشل إعادة الاتصال. البوت سيتوقف: {recon_err}")
                 break
        except Exception as main_err:
            logger.error(f"❌ [Main] Unexpected error in main loop: {main_err}", exc_info=True)
            logger.info("ℹ️ [Main] Waiting 120 seconds before retrying...")
            # Send a general error alert via Telegram
            send_telegram_message(CHAT_ID, f"⚠️ حدث خطأ غير متوقع في حلقة الفحص الرئيسية: {main_err}")
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
    # Ensure WebSocket manager thread is stopped if it's still running
    # (This might require adding a stop method to the manager thread or making twm global)
    # For now, rely on daemon=True to terminate with the main process

    logger.info("✅ [Cleanup] Resource cleanup complete.")


# ---------------------- Main Entry Point ----------------------
if __name__ == "__main__":
    logger.info("🚀 Starting trading signal bot...")
    logger.info(f"Local Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | UTC Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize Threads to be available as global variables
    ws_thread: Optional[Thread] = None
    tracker_thread: Optional[Thread] = None
    flask_thread: Optional[Thread] = None

    try:
        # 1. Initialize the database first
        init_db()

        # 2. Start WebSocket Ticker
        ws_thread = Thread(target=run_ticker_socket_manager, daemon=True, name="WebSocketThread")
        ws_thread.start()
        logger.info("✅ [Main] WebSocket Ticker thread started.")
        logger.info("ℹ️ [Main] Waiting 5 seconds for WebSocket initialization...")
        time.sleep(5)
        if not ticker_data:
             logger.warning("⚠️ [Main] No initial data received from WebSocket after 5 seconds.")
             # Decide if this is critical enough to exit or just warn
             # For now, it's a warning, bot might still get data later
        else:
             logger.info(f"✅ [Main] Received initial data from WebSocket for {len(ticker_data)} symbols.")


        # 3. Start Signal Tracker
        tracker_thread = Thread(target=track_signals, daemon=True, name="TrackerThread")
        tracker_thread.start()
        logger.info("✅ [Main] Signal Tracker thread started.")

        # 4. Start Flask Server (if Webhook configured)
        if WEBHOOK_URL:
            # Note: Flask's default server is not suitable for production.
            # Using waitress if available, otherwise fallbacks.
            flask_thread = Thread(target=run_flask, daemon=True, name="FlaskThread")
            flask_thread.start()
            logger.info("✅ [Main] Flask Webhook thread started.")
        else:
             logger.info("ℹ️ [Main] Webhook URL not configured, Flask server will not start.")

        # 5. Start the main loop (signal generation)
        main_loop()

    except Exception as startup_err:
        logger.critical(f"❌ [Main] A fatal error occurred during startup or in the main loop: {startup_err}", exc_info=True)
        # Send a critical error alert via Telegram
        send_telegram_message(CHAT_ID, f"❌ خطأ فادح أثناء بدء تشغيل البوت أو في الحلقة الرئيسية: {startup_err}. البوت سيتوقف.")
    finally:
        logger.info("🛑 [Main] Program is shutting down...")
        # send_telegram_message(CHAT_ID, "⚠️ Alert: Trading bot is shutting down now.") # Uncomment to send alert on graceful shutdown
        cleanup_resources()
        logger.info("👋 [Main] Trading signal bot stopped.")
        # Use os._exit(0) to ensure all threads (especially daemon ones) are terminated
        os._exit(0) # Exit cleanly
