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
        logging.FileHandler('crypto_bot_bottom_fishing.log', encoding='utf-8'), # تغيير اسم ملف السجل
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
     logger.critical(f"❌ فشل في تحميل المتغيرات البيئية الأساسية: {e}")
     exit(1) # استخدام رمز خروج غير صفري للإشارة إلى خطأ

logger.info(f"Binance API Key: {'Available' if API_KEY else 'Not available'}")
logger.info(f"Telegram Token: {TELEGRAM_TOKEN[:10]}...{'*' * (len(TELEGRAM_TOKEN)-10)}")
logger.info(f"Telegram Chat ID: {CHAT_ID}")
logger.info(f"Database URL: {'Available' if DB_URL else 'Not available'}")
logger.info(f"Webhook URL: {WEBHOOK_URL if WEBHOOK_URL else 'Not specified'}")

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
TRADE_VALUE: float = 10.0         # Default trade value in USDT
MAX_OPEN_TRADES: int = 4          # Maximum number of open trades simultaneously
SIGNAL_GENERATION_TIMEFRAME: str = '30m' # Timeframe for signal generation (كما طلب المستخدم، سنستخدمها للتوليد)
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 10 # Historical data lookback in days for signal generation (زيادة الأيام لالتقاط المزيد من القيعان المحتملة)
SIGNAL_TRACKING_TIMEFRAME: str = '1m' # Timeframe for signal tracking and stop loss updates (إطار زمني أصغر لتتبع أكثر آنية)
SIGNAL_TRACKING_LOOKBACK_DAYS: int = 2   # Historical data lookback in days for signal tracking (أيام أقل للتتبع)

# =============================================================================
# --- Indicator Parameters ---
# Adjusted parameters for bottom fishing strategy
# =============================================================================
RSI_PERIOD: int = 14          # RSI Period
RSI_OVERSOLD: int = 35        # Increased Oversold threshold slightly to catch earlier signs
RSI_OVERBOUGHT: int = 65      # Decreased Overbought threshold
EMA_SHORT_PERIOD: int = 10      # Short EMA period (faster)
EMA_LONG_PERIOD: int = 20       # Long EMA period (faster)
VWMA_PERIOD: int = 20           # VWMA Period
SWING_ORDER: int = 5          # Order for swing point detection (for Elliott, etc.)
# ... (Other constants remain the same) ...
FIB_LEVELS_TO_CHECK: List[float] = [0.382, 0.5, 0.618] # قد تكون أقل أهمية لاستراتيجية القيعان البسيطة
FIB_TOLERANCE: float = 0.007
LOOKBACK_FOR_SWINGS: int = 100
ENTRY_ATR_PERIOD: int = 14     # ATR Period for entry/tracking
ENTRY_ATR_MULTIPLIER: float = 2.0 # Reduced ATR Multiplier for initial target/stop (أكثر تحفظًا في الهدف الأولي)
BOLLINGER_WINDOW: int = 20     # Bollinger Bands Window
BOLLINGER_STD_DEV: int = 2       # Bollinger Bands Standard Deviation
MACD_FAST: int = 12            # MACD Fast Period
MACD_SLOW: int = 26            # MACD Slow Period
MACD_SIGNAL: int = 9             # MACD Signal Line Period
ADX_PERIOD: int = 14            # ADX Period
SUPERTREND_PERIOD: int = 10     # SuperTrend Period (قد يكون أقل أهمية في القيعان)
SUPERTREND_MULTIPLIER: float = 2.5 # SuperTrend Multiplier (أقل شراسة)

# Trailing Stop Loss (Adjusted for tighter stops)
TRAILING_STOP_ACTIVATION_PROFIT_PCT: float = 0.01 # Profit percentage to activate trailing stop (1%)
TRAILING_STOP_ATR_MULTIPLIER: float = 1.5        # Reduced ATR Multiplier for tighter trailing stop
TRAILING_STOP_MOVE_INCREMENT_PCT: float = 0.002  # Price increase percentage to move trailing stop (0.2%) - Increased slightly for less frequent updates

# Additional Signal Conditions
MIN_PROFIT_MARGIN_PCT: float = 1.5 # Minimum required profit margin percentage (Reduced for potential faster trades)
MIN_VOLUME_15M_USDT: float = 50000.0 # Minimum liquidity in the last 15 minutes in USDT (Increased slightly)
MAX_TRADE_DURATION_HOURS: int = 72 # Maximum time to keep a trade open (72 hours = 3 days)
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
    logger.info("ℹ️ [Binance] تهيئة عميل Binance...")
    client = Client(API_KEY, API_SECRET)
    client.ping() # Check connection and keys validity
    server_time = client.get_server_time()
    logger.info(f"✅ [Binance] تم تهيئة عميل Binance بنجاح. وقت الخادم: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
except BinanceRequestException as req_err:
     logger.critical(f"❌ [Binance] خطأ طلب Binance (مشكلة شبكة أو طلب): {req_err}")
     exit(1)
except BinanceAPIException as api_err:
     logger.critical(f"❌ [Binance] خطأ API Binance (مفاتيح غير صالحة أو مشكلة خادم): {api_err}")
     exit(1)
except Exception as e:
    logger.critical(f"❌ [Binance] فشل غير متوقع في تهيئة عميل Binance: {e}", exc_info=True)
    exit(1)

# ---------------------- Additional Indicator Functions (Keep existing) ----------------------
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

        diff_ema20_pct = abs(current_close - ema20) / current_close if current_close > 0 else 0

        if current_close > ema20 > ema50:
            trend = "صعود 📈" # Uptrend
        elif current_close < ema20 < ema50:
            trend = "هبوط 📉" # Downtrend
        elif diff_ema20_pct < 0.005: # Less than 0.5% difference, considered stable
            trend = "استقرار 🔄" # Sideways
        else: # Crossover or unclear divergence
            trend = "تذبذب 🔀" # Volatile

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
                    r2_score DOUBLE PRECISION, -- Now represents the weighted signal score
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
                    last_trailing_update_price DOUBLE PRECISION
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
                "last_trailing_update_price"
            }
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'signals' AND table_schema = 'public';")
            existing_columns = {row['column_name'] for row in cur.fetchall()}
            missing_columns = required_columns - existing_columns

            if missing_columns:
                logger.warning(f"⚠️ [DB] Following columns are missing in 'signals' table: {missing_columns}. Attempting to add them...")
                # (Original code to add columns was fine, can keep or improve here if needed)
                # ... (ALTER TABLE code can be added here if you anticipate future changes) ...
                logger.warning("⚠️ [DB] Automatic addition of missing columns is not implemented in this enhanced version. Please check manually if needed.")
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
    elif pd.isna(obj): # Handle NaT from Pandas as well
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

            # Using start_miniticker_socket covers all symbols and is suitable here
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
    rs = avg_gain / avg_loss.replace(0, np.nan) # Replace zero with NaN to avoid division by zero

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
    if len(df) < period + 1: # We need one extra candle for shift(1)
        logger.warning(f"⚠️ [Indicator ATR] Insufficient data ({len(df)} < {period + 1}) to calculate ATR.")
        df['atr'] = np.nan
        return df

    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()

    # Calculate True Range (TR) - Ignore NaN during max calculation
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)

    # Calculate ATR using EMA (using span gives a result closer to TradingView than com=period-1)
    df['atr'] = tr.ewm(span=period, adjust=False).mean()
    return df


def calculate_bollinger_bands(df: pd.DataFrame, window: int = BOLLINGER_WINDOW, num_std: int = BOLLINGER_STD_DEV) -> pd.DataFrame:
    """Calculates Bollinger Bands."""
    df = df.copy()
    if 'close' not in df.columns or df['close'].isnull().all():
        logger.warning("⚠️ [Indicator BB] 'close' column missing or empty.")
        df['bb_middle'] = np.nan
        df['bb_upper'] = np.nan
        df['bb_lower'] = np.nan
        return df
    if len(df) < window:
         logger.warning(f"⚠️ [Indicator BB] Insufficient data ({len(df)} < {window}) to calculate BB.")
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
    df_calc['tr'] = df_calc[['high-low', 'high-prev_close', 'low-prev_close']].max(axis=1, skipna=False)

    # Calculate Directional Movement (+DM, -DM)
    df_calc['up_move'] = df_calc['high'] - df_calc['high'].shift(1)
    df_calc['down_move'] = df_calc['low'].shift(1) - df_calc['low']
    df_calc['+dm'] = np.where((df_calc['up_move'] > df_calc['down_move']) & (df_calc['up_move'] > 0), df_calc['up_move'], 0)
    df_calc['-dm'] = np.where((df_calc['down_move'] > df_calc['up_move']) & (df_calc['down_move'] > 0), df_calc['down_move'], 0)

    # Use EMA to calculate smoothed values (alpha = 1/period)
    alpha = 1 / period
    df_calc['tr_smooth'] = df_calc['tr'].ewm(alpha=alpha, adjust=False).mean()
    df_calc['+dm_smooth'] = df_calc['+dm'].ewm(alpha=alpha, adjust=False).mean()
    df_calc['-dm_smooth'] = df_calc['-dm'].ewm(alpha=alpha, adjust=False).mean()

    # Calculate Directional Indicators (DI+, DI-) and avoid division by zero
    df_calc['di_plus'] = np.where(df_calc['tr_smooth'] > 0, 100 * (df_calc['+dm_smooth'] / df_calc['tr_smooth']), 0)
    df_calc['di_minus'] = np.where(df_calc['tr_smooth'] > 0, 100 * (df_calc['-dm_smooth'] / df_calc['tr_smooth']), 0)

    # Calculate Directional Movement Index (DX)
    di_sum = df_calc['di_plus'] + df_calc['di_minus']
    df_calc['dx'] = np.where(di_sum > 0, 100 * abs(df_calc['di_plus'] - df_calc['di_minus']) / di_sum, 0)

    # Calculate Average Directional Index (ADX) using EMA
    df_calc['adx'] = df_calc['dx'].ewm(alpha=alpha, adjust=False).mean()

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
    df['vwap'] = np.where(df['cum_volume'] > 0, df['cum_tp_vol'] / df['cum_volume'], np.nan)

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
        if np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(close_diff[i]):
            obv[i] = obv[i-1] # Keep previous value in case of NaN
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
        logger.warning("⚠️ [Indicator SuperTrend] 'high', 'low', 'close' columns missing or empty.")
        df_st['supertrend'] = np.nan
        df_st['supertrend_trend'] = 0 # 0: unknown, 1: uptrend, -1: downtrend
        return df_st

    # Ensure ATR column exists or calculate it
    if 'atr' not in df_st.columns or df_st['atr'].isnull().all():
        logger.debug(f"ℹ️ [Indicator SuperTrend] Calculating ATR (period={period}) for SuperTrend...")
        # Use the ATR period specific to SuperTrend here
        df_st = calculate_atr_indicator(df_st, period=period)

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

    # Initialize final columns
    df_st['final_ub'] = 0.0
    df_st['final_lb'] = 0.0
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

    # Start from the second candle (index 1) as we compare with the previous
    for i in range(1, len(df_st)):
        # Handle NaN in essential inputs for this candle
        if pd.isna(basic_ub[i]) or pd.isna(basic_lb[i]) or pd.isna(close[i]):
            # In case of NaN, keep previous values for final bands, supertrend, and trend
            final_ub[i] = final_ub[i-1]
            final_lb[i] = final_lb[i-1]
            st[i] = st[i-1]
            st_trend[i] = st_trend[i-1]
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
             if close[i] > final_ub[i]: # Start of uptrend
                 st[i] = final_lb[i]
                 st_trend[i] = 1
             elif close[i] < final_lb[i]: # Start of downtrend
                  st[i] = final_ub[i]
                  st_trend[i] = -1
             else: # Still between bands
                  st[i] = np.nan # Or can use previous value if available
                  st_trend[i] = 0


    # Assign calculated values back to DataFrame
    df_st['final_ub'] = final_ub
    df_st['final_lb'] = final_lb
    df_st['supertrend'] = st
    df_st['supertrend_trend'] = st_trend

    # Remove helper columns
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
    is_small_body = body < (candle_range * 0.35) # Slightly larger tolerance for body
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
    is_small_lower_shadow = lower_shadow <= body * 0.6 if body > 0 else lower_shadow < candle_range * 0.15
    return -100 if is_small_body and is_long_upper_shadow and is_small_lower_shadow else 0 # Negative signal

def is_doji(row: pd.Series) -> int:
    """Checks for Doji pattern (uncertainty)."""
    o, h, l, c = row.get('open'), row.get('high'), row.get('low'), row.get('close')
    if pd.isna([o, h, l, c]).any(): return 0
    candle_range = h - l
    if candle_range == 0: return 0
    return 100 if abs(c - o) <= (candle_range * 0.1) else 0 # Very small body

def compute_engulfing(df: pd.DataFrame, idx: int) -> int:
    """Checks for Bullish or Bearish Engulfing pattern."""
    if idx == 0: return 0
    prev = df.iloc[idx - 1]
    curr = df.iloc[idx]
    # Check for NaN in required values
    if pd.isna([prev['close'], prev['open'], curr['close'], curr['open']]).any():
        return 0

    # Bullish Engulfing: Previous candle bearish, current bullish engulfing previous body
    is_bullish = (prev['close'] < prev['open'] and curr['close'] > curr['open'] and
                  curr['open'] <= prev['close'] and curr['close'] >= prev['open'])
    # Bearish Engulfing: Previous candle bullish, current bearish engulfing previous body
    is_bearish = (prev['close'] > prev['open'] and curr['close'] < curr['open'] and
                  curr['open'] >= prev['close'] and curr['close'] <= prev['open'])

    if is_bullish: return 100
    if is_bearish: return -100
    return 0

def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Applies candlestick pattern detection functions to the DataFrame."""
    df = df.copy()
    logger.debug("ℹ️ [Indicators] Detecting candlestick patterns...")
    # Apply single-row patterns
    df['Hammer'] = df.apply(is_hammer, axis=1)
    df['ShootingStar'] = df.apply(is_shooting_star, axis=1)
    df['Doji'] = df.apply(is_doji, axis=1)
    # df['SpinningTop'] = df.apply(is_spinning_top, axis=1) # Can be added if needed

    # Engulfing requires access to the previous row
    engulfing_values = [compute_engulfing(df, i) for i in range(len(df))]
    df['Engulfing'] = engulfing_values

    # Aggregate strong bullish and bearish candle signals
    # Note: Signal value here is 100 or 0, weight will be applied later in the strategy
    df['BullishCandleSignal'] = df.apply(lambda row: 1 if (row['Hammer'] == 100 or row['Engulfing'] == 100) else 0, axis=1)
    df['BearishCandleSignal'] = df.apply(lambda row: 1 if (row['ShootingStar'] == -100 or row['Engulfing'] == -100) else 0, axis=1)

    # Drop individual pattern columns if not needed later
    # df.drop(columns=['Hammer', 'ShootingStar', 'Doji', 'Engulfing'], inplace=True, errors='ignore')
    logger.debug("✅ [Indicators] Candlestick patterns detected.")
    return df

# ---------------------- Other Helper Functions (Elliott, Swings, Volume) ----------------------
def detect_swings(prices: np.ndarray, order: int = SWING_ORDER) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Detects swing points (peaks and troughs) in a time series (numpy array)."""
    n = len(prices)
    if n < 2 * order + 1: return [], []

    maxima_indices = []
    minima_indices = []

    # Improve performance by avoiding loop on unnecessary edges
    for i in range(order, n - order):
        window = prices[i - order : i + order + 1]
        center_val = prices[i]

        # Check for NaN in the window
        if np.isnan(window).any(): continue

        is_max = np.all(center_val >= window) # Is it greater than or equal to all?
        is_min = np.all(center_val <= window) # Is it less than or equal to all?
        # Ensure it's the only peak/trough in the window (to avoid duplicates in flat areas)
        is_unique_max = is_max and (np.sum(window == center_val) == 1)
        is_unique_min = is_min and (np.sum(window == center_val) == 1)

        if is_unique_max:
            # Ensure no peak is too close (within 'order' distance)
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

    # Use only non-null values
    macd_values = df['macd_hist'].dropna().values
    if len(macd_values) < 2 * order + 1:
         logger.warning("⚠️ [Elliott] Insufficient MACD hist data after removing NaNs.")
         return []

    maxima, minima = detect_swings(macd_values, order=order)

    # Merge and sort all swing points by original index
    # (Need to link back to original index from df after dropping NaNs)
    df_nonan_macd = df['macd_hist'].dropna()
    all_swings = sorted(
        [(df_nonan_macd.index[idx], val, 'max') for idx, val in maxima] +
        [(df_nonan_macd.index[idx], val, 'min') for idx, val in minima],
        key=lambda x: x[0] # Sort by time (original index)
    )

    waves = []
    wave_number = 1
    for timestamp, val, typ in all_swings:
        # Very basic classification, may not strictly follow Elliott rules
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
         logger.error(f"❌ [Data Volume] Binance client not initialized to fetch volume for {symbol}.")
         return 0.0
    try:
        logger.debug(f"ℹ️ [Data Volume] Fetching 15-minute volume for {symbol}...")
        # Fetch 1-minute data for the last 15 minutes
        klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=15)
        if not klines or len(klines) < 15:
             logger.warning(f"⚠️ [Data Volume] Insufficient 1m data (less than 15 candles) for {symbol}.")
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

# ---------------------- Trading Strategy (Modified for Bottom Fishing) -------------------

class BottomFishingStrategy:
    """Encapsulates the trading strategy logic focused on capturing potential bottoms using a scoring system."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        # Required columns for indicator calculation (ensure all needed indicators are listed)
        self.required_cols_indicators = [
            'open', 'high', 'low', 'close', 'volume',
            'ema_10', 'ema_20', 'vwma', # Changed EMA periods
            'rsi', 'atr', 'bb_upper', 'bb_lower', 'bb_middle',
            'macd', 'macd_signal', 'macd_hist',
            'adx', 'di_plus', 'di_minus',
            'vwap', 'obv', 'supertrend', 'supertrend_trend',
            'BullishCandleSignal', 'BearishCandleSignal'
        ]
        # Required columns for buy signal generation
        self.required_cols_buy_signal = [
            'close',
            'ema_10', 'ema_20', 'vwma',
            'rsi', 'atr',
            'macd', 'macd_signal', 'macd_hist',
            'supertrend_trend', 'adx', 'di_plus', 'di_minus', 'vwap', 'bb_lower', # Changed bb_upper to bb_lower
            'BullishCandleSignal', 'obv'
        ]

        # =====================================================================
        # --- Scoring System (Weights) for Optional Conditions ---
        # Adjusted weights and conditions for bottom fishing
        # Removed 'rsi_bouncing_up' from optional as it's mandatory
        # Increased weights for MACD momentum shift, ADX/DI cross, and OBV rising
        # =====================================================================
        self.condition_weights = {
            'price_near_bb_lower': 2.0,   # Price touching or below lower Bollinger Band (High importance)
            'macd_bullish_momentum_shift': 2.5, # MACD showing upward momentum shift (Increased importance)
            'price_crossing_vwma_up': 1.5, # Price crosses above VWMA
            'adx_low_and_di_cross': 1.5,  # Low ADX and DI+ crossing above DI- (Increased importance)
            'price_crossing_ema10_up': 1.0, # Price crosses above faster EMA
            'obv_rising': 1.5,            # OBV is rising (Increased importance)

            # Removed conditions that conflict with bottom fishing or breakout
            # 'ema_cross_bullish': 0,
            # 'supertrend_up': 0,
            # 'above_vwma': 0, # Now looking for cross up from below
            # 'macd_positive_or_cross': 0, # Looking for shift from below zero
            # 'adx_trending_bullish': 0, # Looking for low ADX turning
            # 'breakout_bb_upper': 0,
            # 'rsi_ok': 0, # Replaced by rsi_oversold_or_bouncing (mandatory)
            # 'not_bb_extreme': 0, # Replaced by price_near_bb_lower
            # 'rsi_filter_breakout': 0,
            # 'macd_filter_breakout': 0
        }
        # =====================================================================

        # =====================================================================
        # --- Mandatory Entry Conditions (All must be met for bottom fishing) ---
        # Focused on oversold state and bullish reversal candle
        # Kept 'rsi_oversold_or_bouncing' as mandatory only
        # =====================================================================
        self.essential_conditions = [
            'rsi_oversold_or_bouncing', # RSI is oversold OR just bounced from oversold
            'bullish_reversal_candle', # Presence of a bullish reversal candle
            'price_has_dropped_recently' # Add a simple check for recent price drop (needs implementation)
        ]
        # =====================================================================


        # Calculate total possible score for *optional* conditions based on the new weights
        self.total_possible_score = sum(self.condition_weights.values())

        # Required signal score threshold for *optional* conditions (as a percentage)
        # Adjusted based on the new total possible score
        self.min_score_threshold_pct = 0.40 # Example: 40% of optional points (adjustable)
        self.min_signal_score = self.total_possible_score * self.min_score_threshold_pct


    def populate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculates all required indicators for the strategy."""
        logger.debug(f"ℹ️ [Strategy {self.symbol}] حساب المؤشرات...")
        # Update minimum required rows based on the largest period of used indicators
        min_len_required = max(EMA_SHORT_PERIOD, EMA_LONG_PERIOD, VWMA_PERIOD, RSI_PERIOD, ENTRY_ATR_PERIOD, BOLLINGER_WINDOW, MACD_SLOW, ADX_PERIOD*2, SUPERTREND_PERIOD) + 5 # Add a small buffer

        if len(df) < min_len_required:
            logger.warning(f"⚠️ [Strategy {self.symbol}] بيانات غير كافية ({len(df)} < {min_len_required}) لحساب المؤشرات.")
            return None

        try:
            df_calc = df.copy()
            # ATR is required for SuperTrend and Stop Loss/Target
            df_calc = calculate_atr_indicator(df_calc, ENTRY_ATR_PERIOD)
            # SuperTrend needs ATR calculated with its own period
            df_calc = calculate_supertrend(df_calc, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)

            # --- EMA Calculation (using adjusted periods) ---
            df_calc['ema_10'] = calculate_ema(df_calc['close'], EMA_SHORT_PERIOD) # Add EMA 10
            df_calc['ema_20'] = calculate_ema(df_calc['close'], EMA_LONG_PERIOD) # Add EMA 20
            # ----------------------

            # --- VWMA Calculation ---
            df_calc['vwma'] = calculate_vwma(df_calc, VWMA_PERIOD) # Calculate VWMA
            # ----------------------

            # Rest of the indicators
            df_calc = calculate_rsi_indicator(df_calc, RSI_PERIOD)
            df_calc = calculate_bollinger_bands(df_calc, BOLLINGER_WINDOW, BOLLINGER_STD_DEV)
            df_calc = calculate_macd(df_calc, MACD_FAST, MACD_SLOW, MACD_SIGNAL) # Ensure macd_hist is calculated here
            adx_df = calculate_adx(df_calc, ADX_PERIOD)
            df_calc = df_calc.join(adx_df)
            df_calc = calculate_vwap(df_calc) # Note: VWAP resets daily, VWMA is a rolling average
            df_calc = calculate_obv(df_calc)
            df_calc = detect_candlestick_patterns(df_calc)

            # Check for required columns after calculation
            missing_cols = [col for col in self.required_cols_indicators if col not in df_calc.columns]
            if missing_cols:
                 logger.error(f"❌ [Strategy {self.symbol}] أعمدة المؤشرات المطلوبة مفقودة بعد الحساب: {missing_cols}")
                 logger.debug(f"الأعمدة المتوفرة: {df_calc.columns.tolist()}")
                 return None

            # Handle NaNs after indicator calculation
            initial_len = len(df_calc)
            # Use required_cols_indicators which contains all calculated columns
            df_cleaned = df_calc.dropna(subset=self.required_cols_indicators).copy()
            dropped_count = initial_len - len(df_cleaned)

            if dropped_count > 0:
                 logger.debug(f"ℹ️ [Strategy {self.symbol}] تم حذف {dropped_count} صف بسبب قيم NaN في المؤشرات.")
            if df_cleaned.empty:
                logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame فارغ بعد إزالة قيم NaN للمؤشرات.")
                return None

            latest = df_cleaned.iloc[-1]
            logger.debug(f"✅ [Strategy {self.symbol}] تم حساب المؤشرات. آخر قيم - EMA10: {latest.get('ema_10', np.nan):.4f}, EMA20: {latest.get('ema_20', np.nan):.4f}, VWMA: {latest.get('vwma', np.nan):.4f}, RSI: {latest.get('rsi', np.nan):.1f}, MACD Hist: {latest.get('macd_hist', np.nan):.4f}")
            return df_cleaned

        except KeyError as ke:
             logger.error(f"❌ [Strategy {self.symbol}] خطأ: العمود المطلوب غير موجود أثناء حساب المؤشر: {ke}", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"❌ [Strategy {self.symbol}] خطأ غير متوقع أثناء حساب المؤشرات: {e}", exc_info=True)
            return None


    def generate_buy_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Generates a buy signal based on the processed DataFrame, mandatory bottom-fishing conditions,
        and a scoring system for optional conditions.
        """
        logger.debug(f"ℹ️ [Strategy {self.symbol}] توليد إشارة شراء (صيد القيعان)...")

        # Check DataFrame and columns
        if df_processed is None or df_processed.empty or len(df_processed) < max(2, MACD_SLOW + 1): # Need at least 2 for diff, and enough for indicators
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame فارغ أو قصير جدًا، لا يمكن توليد الإشارة.")
            return None
        # Add required columns for signal if not already present
        required_cols_for_signal = list(set(self.required_cols_buy_signal))
        missing_cols = [col for col in required_cols_for_signal if col not in df_processed.columns]
        if missing_cols:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame يفتقد الأعمدة المطلوبة للإشارة: {missing_cols}.")
            return None

        # Check Bitcoin trend (still a mandatory filter)
        btc_trend = get_btc_trend_4h()
        # Only allow signal if Bitcoin trend is bullish, neutral, or unknown (not bearish)
        if "هبوط" in btc_trend: # Downtrend
            logger.info(f"ℹ️ [Strategy {self.symbol}] التداول متوقف بسبب اتجاه البيتكوين الهبوطي ({btc_trend}).")
            return None
        # Do not reject if "N/A" or "استقرار" (Sideways) or "تذبذب" (Volatile)
        elif "N/A" in btc_trend:
             logger.warning(f"⚠️ [Strategy {self.symbol}] لا يمكن تحديد اتجاه البيتكوين، سيتم تجاهل هذا الشرط.")


        # Extract latest and previous candle data
        last_row = df_processed.iloc[-1]
        prev_row = df_processed.iloc[-2] if len(df_processed) >= 2 else pd.Series() # Handle case with only one row

        # Check for NaN in essential columns required for the signal (only last row needs full check)
        last_row_check = last_row[required_cols_for_signal]
        if last_row_check.isnull().any():
            nan_cols = last_row_check[last_row_check.isnull()].index.tolist()
            logger.warning(f"⚠️ [Strategy {self.symbol}] الصف الأخير يحتوي على قيم NaN في أعمدة الإشارة المطلوبة: {nan_cols}. لا يمكن توليد الإشارة.")
            return None
        # Check previous values needed for conditions
        if len(df_processed) < 2:
             logger.warning(f"⚠️ [Strategy {self.symbol}] بيانات غير كافية (أقل من شمعتين) للتحقق من الشروط التي تتطلب البيانات السابقة.")
             # Some conditions below might fail due to missing prev_row, they will contribute 0 points.

        # =====================================================================
        # --- Check Mandatory Bottom-Fishing Conditions First ---
        # If any mandatory condition fails, the signal is rejected immediately
        # =====================================================================
        essential_passed = True
        failed_essential_conditions = []
        signal_details = {} # To store details of checked conditions (mandatory and optional)

        # Mandatory Condition 1: RSI Oversold or Bouncing Up from Oversold
        # Check if RSI is currently oversold OR (was oversold and is now higher)
        is_oversold = last_row['rsi'] <= RSI_OVERSOLD
        bounced_from_oversold = (len(df_processed) >= 2 and
                                   pd.notna(prev_row.get('rsi')) and
                                   prev_row['rsi'] <= RSI_OVERSOLD and
                                   last_row['rsi'] > prev_row['rsi'])

        if not (is_oversold or bounced_from_oversold):
            essential_passed = False
            failed_essential_conditions.append('RSI Oversold or Bouncing')
            signal_details['RSI_Mandatory'] = f'فشل: RSI={last_row["rsi"]:.1f} (ليس في منطقة البيع المفرط أو لم يرتد)'
        else:
             signal_details['RSI_Mandatory'] = f'نجاح: RSI={last_row["rsi"]:.1f} (في منطقة البيع المفرط أو يرتد)'


        # Mandatory Condition 2: Bullish Reversal Candlestick Pattern
        if last_row['BullishCandleSignal'] != 1:
            essential_passed = False
            failed_essential_conditions.append('Bullish Reversal Candle')
            signal_details['Candle_Mandatory'] = 'فشل: لا يوجد نموذج شموع انعكاسي صعودي'
        else:
             signal_details['Candle_Mandatory'] = 'نجاح: يوجد نموذج شموع انعكاسي صعودي'

        # Mandatory Condition 3: Simple check for recent price drop (e.g., price is below EMA20 from a few bars ago)
        # This is a simplified way to check if the price was higher recently
        if len(df_processed) < EMA_LONG_PERIOD: # Need enough data for this check
             logger.warning(f"⚠️ [Strategy {self.symbol}] بيانات غير كافية للشمعات السابقة للتحقق من انخفاض السعر.")
             # This mandatory condition will be considered failed if we don't have enough history
             essential_passed = False
             failed_essential_conditions.append('Recent Price Drop (Insufficient Data)')
             signal_details['Recent_Drop_Mandatory'] = 'فشل: بيانات غير كافية للتحقق من انخفاض السعر الأخير'

        else:
            # Check if the current price is significantly lower than the price 'n' bars ago (e.g., EMA_LONG_PERIOD bars ago)
            price_n_bars_ago = df_processed['close'].iloc[-EMA_LONG_PERIOD]
            price_drop_threshold = price_n_bars_ago * 0.97 # Example: at least a 3% drop from n bars ago

            if not (last_row['close'] < price_drop_threshold):
                 essential_passed = False
                 failed_essential_conditions.append('Recent Price Drop')
                 signal_details['Recent_Drop_Mandatory'] = f'فشل: السعر الحالي ({last_row["close"]:.4f}) ليس أقل بكثير من سعر {EMA_LONG_PERIOD} شمعة سابقة ({price_n_bars_ago:.4f})'
            else:
                 signal_details['Recent_Drop_Mandatory'] = f'نجاح: السعر الحالي ({last_row["close"]:.4f}) أقل بشكل ملحوظ من سعر {EMA_LONG_PERIOD} شمعة سابقة ({price_n_bars_ago:.4f})'


        # If any mandatory condition failed, reject the signal immediately
        if not essential_passed:
            logger.debug(f"ℹ️ [Strategy {self.symbol}] فشلت الشروط الإلزامية: {', '.join(failed_essential_conditions)}. تم رفض الإشارة.")
            signal_details['Mandatory_Status'] = 'فشل'
            signal_details['Failed_Mandatory'] = failed_essential_conditions
            return None
        else:
             signal_details['Mandatory_Status'] = 'نجاح'

        # =====================================================================
        # --- Calculate Score for Optional Conditions (if mandatory passed) ---
        # =====================================================================
        current_score = 0.0

        # Optional Condition 1: Price near or below lower Bollinger Band
        if pd.notna(last_row.get('bb_lower')) and last_row['close'] <= last_row['bb_lower'] * 1.005: # Within 0.5% of lower band or below
             current_score += self.condition_weights.get('price_near_bb_lower', 0)
             signal_details['BB_Lower_Score'] = f'السعر قريب أو تحت الحد السفلي لبولينجر (+{self.condition_weights.get("price_near_bb_lower", 0)})'
        else:
             signal_details['BB_Lower_Score'] = f'السعر ليس قريبا من الحد السفلي لبولينجر (0)'


        # Optional Condition 2: MACD bullish momentum shift (hist turning positive or bullish cross from below zero)
        if len(df_processed) >= 2 and pd.notna(prev_row.get('macd_hist')) and pd.notna(last_row.get('macd_hist')) and pd.notna(prev_row.get('macd')) and pd.notna(last_row.get('macd_signal')):
            macd_hist_turning_up = last_row['macd_hist'] > prev_row['macd_hist']
            macd_cross_from_below_zero = (last_row['macd'] > last_row['macd_signal'] and
                                            prev_row['macd'] <= prev_row['macd_signal'] and
                                            last_row['macd'] < 0) # Added condition that MACD is still below zero or just crossed

            if macd_hist_turning_up or macd_cross_from_below_zero:
                 current_score += self.condition_weights.get('macd_bullish_momentum_shift', 0)
                 detail_macd_score = f'MACD Hist يتحول للإيجابية' if macd_hist_turning_up else ''
                 detail_macd_score += ' و ' if detail_macd_score and macd_cross_from_below_zero else ''
                 detail_macd_score += f'تقاطع صعودي تحت الصفر' if macd_cross_from_below_zero else ''
                 signal_details['MACD_Score'] = f'تحول زخم MACD الصعودي (+{self.condition_weights.get("macd_bullish_momentum_shift", 0)}) ({detail_macd_score})'
            else:
                 signal_details['MACD_Score'] = f'لا يوجد تحول زخم MACD صعودي (0)'
        else:
             signal_details['MACD_Score'] = f'بيانات MACD غير كافية أو NaN (0)'


        # Optional Condition 3: Price crosses above VWMA
        if len(df_processed) >= 2 and pd.notna(last_row.get('close')) and pd.notna(last_row.get('vwma')) and pd.notna(prev_row.get('close')) and pd.notna(prev_row.get('vwma')):
            if last_row['close'] > last_row['vwma'] and prev_row['close'] <= prev_row['vwma']:
                 current_score += self.condition_weights.get('price_crossing_vwma_up', 0)
                 signal_details['VWMA_Cross_Score'] = f'السعر يتقاطع فوق VWMA (+{self.condition_weights.get("price_crossing_vwma_up", 0)})'
            else:
                 signal_details['VWMA_Cross_Score'] = f'السعر لم يتقاطع فوق VWMA (0)'
        else:
             signal_details['VWMA_Cross_Score'] = f'بيانات VWMA غير كافية أو NaN (0)'


        # Optional Condition 4: Low ADX and DI+ crossing above DI-
        if len(df_processed) >= 2 and pd.notna(last_row.get('adx')) and pd.notna(last_row.get('di_plus')) and pd.notna(last_row.get('di_minus')) and pd.notna(prev_row.get('di_plus')) and pd.notna(prev_row.get('di_minus')):
             if last_row['adx'] < 25 and last_row['di_plus'] > last_row['di_minus'] and prev_row['di_plus'] <= prev_row['di_minus']: # Slightly increased low ADX threshold
                 current_score += self.condition_weights.get('adx_low_and_di_cross', 0)
                 signal_details['ADX_DI_Score'] = f'ADX منخفض وتقاطع DI+ (+{self.condition_weights.get("adx_low_and_di_cross", 0)})'
             else:
                 signal_details['ADX_DI_Score'] = f'ADX ليس منخفضا أو لا يوجد تقاطع DI+ (0)'
        else:
             signal_details['ADX_DI_Score'] = f'بيانات ADX/DI غير كافية أو NaN (0)'


        # Optional Condition 5: Price crosses above EMA10
        if len(df_processed) >= 2 and pd.notna(last_row.get('close')) and pd.notna(last_row.get('ema_10')) and pd.notna(prev_row.get('close')) and pd.notna(prev_row.get('ema_10')):
            if last_row['close'] > last_row['ema_10'] and prev_row['close'] <= prev_row['ema_10']:
                 current_score += self.condition_weights.get('price_crossing_ema10_up', 0)
                 signal_details['EMA10_Cross_Score'] = f'السعر يتقاطع فوق EMA10 (+{self.condition_weights.get("price_crossing_ema10_up", 0)})'
            else:
                 signal_details['EMA10_Cross_Score'] = f'السعر لم يتقاطع فوق EMA10 (0)'
        else:
             signal_details['EMA10_Cross_Score'] = f'بيانات EMA10 غير كافية أو NaN (0)'

        # Optional Condition 6: OBV is rising
        if len(df_processed) >= 2 and pd.notna(last_row.get('obv')) and pd.notna(prev_row.get('obv')):
            if last_row['obv'] > prev_row['obv']:
                 current_score += self.condition_weights.get('obv_rising', 0)
                 signal_details['OBV_Score'] = f'OBV يرتفع (+{self.condition_weights.get("obv_rising", 0)})'
            else:
                 signal_details['OBV_Score'] = f'OBV لا يرتفع (0)'
        else:
             signal_details['OBV_Score'] = f'بيانات OBV غير كافية أو NaN (0)'

        # ------------------------------------------

        # Final buy decision based on the score of optional conditions
        if current_score < self.min_signal_score:
            logger.debug(f"ℹ️ [Strategy {self.symbol}] نقاط الإشارة المطلوبة من الشروط الاختيارية لم يتم تحقيقها (النقاط: {current_score:.2f} / {self.total_possible_score:.2f}, الحد الأدنى: {self.min_signal_score:.2f}). تم رفض الإشارة.")
            signal_details['Optional_Score_Status'] = 'فشل'
            signal_details['Calculated_Score'] = float(f"{current_score:.2f}")
            signal_details['Min_Required_Score'] = float(f"{self.min_signal_score:.2f}")
            return None
        else:
             signal_details['Optional_Score_Status'] = 'نجاح'
             signal_details['Calculated_Score'] = float(f"{current_score:.2f}")
             signal_details['Min_Required_Score'] = float(f"{self.min_signal_score:.2f}")


        # Check trading volume (liquidity) - still a mandatory filter
        volume_recent = fetch_recent_volume(self.symbol)
        if volume_recent < MIN_VOLUME_15M_USDT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] السيولة ({volume_recent:,.0f} USDT) أقل من الحد الأدنى المطلوب ({MIN_VOLUME_15M_USDT:,.0f} USDT). تم رفض الإشارة.")
            signal_details['Liquidity_Status'] = 'فشل'
            signal_details['Volume_15m'] = volume_recent
            signal_details['Min_Volume_15m'] = MIN_VOLUME_15M_USDT
            return None
        else:
             signal_details['Liquidity_Status'] = 'نجاح'
             signal_details['Volume_15m'] = volume_recent
             signal_details['Min_Volume_15m'] = MIN_VOLUME_15M_USDT


        # Calculate initial target and stop loss based on ATR
        current_price = last_row['close']
        current_atr = last_row.get('atr')

        # Ensure ATR is not NaN before using it
        if pd.isna(current_atr) or current_atr <= 0:
             logger.warning(f"⚠️ [Strategy {self.symbol}] قيمة ATR غير صالحة ({current_atr}) لحساب الهدف ووقف الخسارة.")
             signal_details['ATR_Status'] = 'فشل'
             return None
        else:
             signal_details['ATR_Status'] = 'نجاح'
             signal_details['Current_ATR'] = float(f"{current_atr:.8g}")


        # These multipliers can be adjusted based on ADX or other factors for a more dynamic strategy if desired
        target_multiplier = ENTRY_ATR_MULTIPLIER
        stop_loss_multiplier = ENTRY_ATR_MULTIPLIER

        initial_target = current_price + (target_multiplier * current_atr)
        initial_stop_loss = current_price - (stop_loss_multiplier * current_atr)

        # Ensure stop loss is not zero or negative and is below the entry price
        if initial_stop_loss <= 0 or initial_stop_loss >= current_price:
            # Use a percentage as a minimum stop loss if the initial calculation is invalid
            min_sl_price_pct = current_price * (1 - 0.015) # Example: 1.5% below entry
            initial_stop_loss = max(min_sl_price_pct, current_price * 0.001) # Ensure it's not too close to zero
            logger.warning(f"⚠️ [Strategy {self.symbol}] وقف الخسارة المحسوب ({initial_stop_loss:.8g}) غير صالح أو أعلى من سعر الدخول. تم تعديله إلى {initial_stop_loss:.8f}")
            signal_details['Warning'] = f'تم تعديل وقف الخسارة الأولي (كان <= 0 أو >= الدخول، تم تعيينه إلى {initial_stop_loss:.8f})'
        else:
             # Ensure the initial stop loss is not too wide (optional)
             max_allowed_loss_pct = 0.10 # Example: Initial loss should not exceed 10%
             max_sl_price = current_price * (1 - max_allowed_loss_pct)
             if initial_stop_loss < max_sl_price:
                  logger.warning(f"⚠️ [Strategy {self.symbol}] وقف الخسارة المحسوب ({initial_stop_loss:.8g}) واسع جدًا. تم تعديله إلى {max_sl_price:.8f}")
                  initial_stop_loss = max_sl_price
                  signal_details['Warning'] = f'تم تعديل وقف الخسارة الأولي (كان واسعًا جدًا، تم تعيينه إلى {initial_stop_loss:.8f})' # Use the new value here


        # Check minimum profit margin (after calculating final target and stop loss) - still a mandatory filter
        profit_margin_pct = ((initial_target / current_price) - 1) * 100 if current_price > 0 else 0
        if profit_margin_pct < MIN_PROFIT_MARGIN_PCT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] هامش الربح ({profit_margin_pct:.2f}%) أقل من الحد الأدنى المطلوب ({MIN_PROFIT_MARGIN_PCT:.2f}%). تم رفض الإشارة.")
            signal_details['Profit_Margin_Status'] = 'فشل'
            signal_details['Profit_Margin_Pct'] = float(f"{profit_margin_pct:.2f}")
            signal_details['Min_Profit_Margin_Pct'] = MIN_PROFIT_MARGIN_PCT
            return None
        else:
             signal_details['Profit_Margin_Status'] = 'نجاح'
             signal_details['Profit_Margin_Pct'] = float(f"{profit_margin_pct:.2f}")
             signal_details['Min_Profit_Margin_Pct'] = MIN_PROFIT_MARGIN_PCT


        # Compile final signal data
        signal_output = {
            'symbol': self.symbol,
            'entry_price': float(f"{current_price:.8g}"),
            'initial_target': float(f"{initial_target:.8g}"),
            'initial_stop_loss': float(f"{initial_stop_loss:.8g}"),
            'current_target': float(f"{initial_target:.8g}"),
            'current_stop_loss': float(f"{initial_stop_loss:.8g}"),
            'r2_score': float(f"{current_score:.2f}"), # Weighted score of optional conditions
            'strategy_name': 'Bottom_Fishing_Filtered', # اسم الاستراتيجية الجديد
            'signal_details': signal_details, # Now contains details of mandatory and optional conditions
            'volume_15m': volume_recent,
            'trade_value': TRADE_VALUE,
            'total_possible_score': float(f"{self.total_possible_score:.2f}") # Total points for optional conditions
        }

        logger.info(f"✅ [Strategy {self.symbol}] تم تأكيد إشارة الشراء (صيد القيعان). السعر: {current_price:.6f}, النقاط (اختيارية): {current_score:.2f}/{self.total_possible_score:.2f}, ATR: {current_atr:.6f}, السيولة: {volume_recent:,.0f}")
        return signal_output


# ---------------------- Telegram Functions (Adjusted message format) ----------------------
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
    """Formats and sends a new trading signal alert (Bottom Fishing) to Telegram in Arabic."""
    logger.debug(f"ℹ️ [Telegram Alert] تنسيق وإرسال تنبيه الإشارة: {signal_data.get('symbol', 'N/A')}")
    try:
        entry_price = float(signal_data['entry_price'])
        target_price = float(signal_data['initial_target'])
        stop_loss_price = float(signal_data['initial_stop_loss'])
        symbol = signal_data['symbol']
        strategy_name = signal_data.get('strategy_name', 'N/A').replace('_', ' ').title() # تنسيق اسم الاستراتيجية
        signal_score = signal_data.get('r2_score', 0.0) # Weighted score for optional conditions
        total_possible_score = signal_data.get('total_possible_score', 10.0) # Total points for optional conditions
        volume_15m = signal_data.get('volume_15m', 0.0)
        trade_value_signal = signal_data.get('trade_value', TRADE_VALUE)
        signal_details = signal_data.get('signal_details', {}) # تفاصيل الشروط

        profit_pct = ((target_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        loss_pct = ((stop_loss_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        profit_usdt = trade_value_signal * (profit_pct / 100)
        loss_usdt = abs(trade_value_signal * (loss_pct / 100))

        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Escape special characters for Markdown
        safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')

        fear_greed = get_fear_greed_index()
        btc_trend = get_btc_trend_4h()

        # Build the message in Arabic with weighted score and condition details
        message = (
            f"💡 *إشارة تداول جديدة ({strategy_name})* 💡\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"📈 **نوع الإشارة:** شراء (صيد القيعان)\n"
            f"🕰️ **الإطار الزمني:** {timeframe}\n"
            # --- إضافة عرض النقاط ---
            f"📊 **قوة الإشارة (نقاط الشروط الاختيارية):** *{signal_score:.1f} / {total_possible_score:.1f}*\n"
            f"💧 **السيولة (15 دقيقة):** {volume_15m:,.0f} USDT\n"
            f"——————————————\n"
            f"➡️ **سعر الدخول المقترح:** `${entry_price:,.8g}`\n"
            f"🎯 **الهدف الأولي:** `${target_price:,.8g}` ({profit_pct:+.2f}% / ≈ ${profit_usdt:+.2f})\n"
            f"🛑 **وقف الخسارة الأولي:** `${stop_loss_price:,.8g}` ({loss_pct:.2f}% / ≈ ${loss_usdt:.2f})\n"
            f"——————————————\n"
            f"✅ *الشروط الإلزامية المحققة:*\n"
            f"  - RSI: {signal_details.get('RSI_Mandatory', 'N/A')}\n"
            f"  - نموذج الشموع: {signal_details.get('Candle_Mandatory', 'N/A')}\n"
            f"  - انخفاض أخير في السعر: {signal_details.get('Recent_Drop_Mandatory', 'N/A')}\n"
            f"——————————————\n"
            f"⭐ *نقاط الشروط الاختيارية:*\n" # قسم جديد لتفاصيل النقاط الاختيارية
            # Removed RSI_Bounce_Score from optional section as it's now only mandatory
            f"  - السعر قرب BB السفلي: {signal_details.get('BB_Lower_Score', 'N/A')}\n"
            f"  - تحول زخم MACD: {signal_details.get('MACD_Score', 'N/A')}\n"
            f"  - تقاطع السعر فوق VWMA: {signal_details.get('VWMA_Cross_Score', 'N/A')}\n"
            f"  - ADX منخفض وتقاطع DI: {signal_details.get('ADX_DI_Score', 'N/A')}\n"
            f"  - تقاطع السعر فوق EMA10: {signal_details.get('EMA10_Cross_Score', 'N/A')}\n"
            f"  - OBV يرتفع: {signal_details.get('OBV_Score', 'N/A')}\n"
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
        logger.error(f"❌ [Telegram Alert] بيانات الإشارة غير مكتملة للزوج {signal_data.get('symbol', 'N/A')}: مفتاح مفقود {ke}", exc_info=True)
    except Exception as e:
        logger.error(f"❌ [Telegram Alert] فشل إرسال تنبيه الإشارة للزوج {signal_data.get('symbol', 'N/A')}: {e}", exc_info=True)

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
    atr_value = details.get('atr_value', 0.0)
    new_stop_loss = details.get('new_stop_loss', 0.0)
    old_stop_loss = details.get('old_stop_loss', 0.0)

    logger.debug(f"ℹ️ [Notification] Formatting tracking notification: ID={signal_id}, Type={notification_type}, Symbol={symbol}")

    if notification_type == 'target_hit':
        message = (
            f"✅ *تم الوصول إلى الهدف (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"🎯 **سعر الإغلاق (الهدف):** `${closing_price:,.8g}`\n"
            f"💰 **الربح المحقق:** {profit_pct:+.2f}%"
        )
    elif notification_type == 'stop_loss_hit':
        sl_type_msg_ar = "بربح ✅" if details.get('profitable_sl', False) else "بخسارة ❌" # Use profitable_sl flag
        message = (
            f"🛑 *تم ضرب وقف الخسارة (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"🚫 **سعر الإغلاق (الوقف):** `${closing_price:,.8g}`\n"
            f"📉 **النتيجة:** {profit_pct:.2f}% ({sl_type_msg_ar})"
        )
    elif notification_type == 'trailing_activated':
        activation_profit_pct = details.get('activation_profit_pct', TRAILING_STOP_ACTIVATION_PROFIT_PCT * 100)
        message = (
            f"⬆️ *تم تفعيل وقف الخسارة المتحرك (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"📈 **السعر الحالي (عند التفعيل):** `${current_price:,.8g}` (الربح > {activation_profit_pct:.1f}%)\n"
            f"📊 **قيمة ATR ({ENTRY_ATR_PERIOD}):** `{atr_value:,.8g}` (المضاعف: {TRAILING_STOP_ATR_MULTIPLIER})\n"
            f"🛡️ **وقف الخسارة الجديد:** `${new_stop_loss:,.8g}`"
        )
    elif notification_type == 'trailing_updated':
        trigger_price_increase_pct = details.get('trigger_price_increase_pct', TRAILING_STOP_MOVE_INCREMENT_PCT * 100)
        message = (
            f"➡️ *تم تحديث وقف الخسارة المتحرك (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"📈 **السعر الحالي (عند التحديث):** `${current_price:,.8g}` (+{trigger_price_increase_pct:.1f}% منذ آخر تحديث)\n"
            f"📊 **قيمة ATR ({ENTRY_ATR_PERIOD}):** `{atr_value:,.8g}` (المضاعف: {TRAILING_STOP_ATR_MULTIPLIER})\n"
            f"🔒 **الوقف السابق:** `${old_stop_loss:,.8g}`\n"
            f"🛡️ **وقف الخسارة الجديد:** `${new_stop_loss:,.8g}`"
        )
    else:
        logger.warning(f"⚠️ [Notification] Unknown notification type: {notification_type} for details: {details}")
        return # Don't send anything if type is unknown

    if message:
        send_telegram_message(CHAT_ID, message, parse_mode='Markdown')


# ---------------------- Database Functions (Insert and Update) (Keep existing) ----------------------
def insert_signal_into_db(signal: Dict[str, Any]) -> bool:
    """Inserts a new signal into the signals table with the weighted score."""
    if not check_db_connection() or not conn:
        logger.error(f"❌ [DB Insert] Failed to insert signal {signal.get('symbol', 'N/A')} due to DB connection issue.")
        return False

    symbol = signal.get('symbol', 'N/A')
    logger.debug(f"ℹ️ [DB Insert] Attempting to insert signal for {symbol}...")
    try:
        signal_prepared = convert_np_values(signal)
        # Convert signal details to JSON (ensure it doesn't contain numpy types)
        signal_details_json = json.dumps(signal_prepared.get('signal_details', {}))

        with conn.cursor() as cur_ins:
            insert_query = sql.SQL("""
                INSERT INTO signals
                 (symbol, entry_price, initial_target, initial_stop_loss, current_target, current_stop_loss,
                 r2_score, strategy_name, signal_details, last_trailing_update_price, volume_15m)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """)
            cur_ins.execute(insert_query, (
                signal_prepared['symbol'],
                signal_prepared['entry_price'],
                signal_prepared['initial_target'],
                signal_prepared['initial_stop_loss'],
                signal_prepared['current_target'],
                signal_prepared['current_stop_loss'],
                signal_prepared.get('r2_score'), # Weighted score
                signal_prepared.get('strategy_name', 'unknown'),
                signal_details_json,
                None, # last_trailing_update_price
                signal_prepared.get('volume_15m')
            ))
        conn.commit()
        logger.info(f"✅ [DB Insert] Signal for {symbol} inserted into database (Score: {signal_prepared.get('r2_score')}).")
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

# ---------------------- Open Signal Tracking Function (Add max trade duration check) ----------------------
def track_signals() -> None:
    """Tracks open signals, checks targets and stop losses, applies trailing stop, and checks max trade duration."""
    logger.info("ℹ️ [Tracker] بدء عملية تتبع الإشارات المفتوحة...")
    while True:
        active_signals_summary: List[str] = []
        processed_in_cycle = 0
        try:
            if not check_db_connection() or not conn:
                logger.warning("⚠️ [Tracker] تخطي دورة التتبع بسبب مشكلة اتصال قاعدة البيانات.")
                time.sleep(15) # Wait a bit longer before retrying
                continue

            # Use a cursor with context manager to fetch open signals
            with conn.cursor() as track_cur: # Uses RealDictCursor
                 track_cur.execute("""
                    SELECT id, symbol, entry_price, initial_stop_loss, current_target, current_stop_loss,
                           is_trailing_active, last_trailing_update_price, sent_at
                    FROM signals
                    WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;
                """)
                 open_signals: List[Dict] = track_cur.fetchall()

            if not open_signals:
                # logger.debug("ℹ️ [Tracker] لا توجد إشارات مفتوحة لتتبعها.")
                time.sleep(10) # Wait less if no signals
                continue

            logger.debug(f"ℹ️ [Tracker] تتبع {len(open_signals)} إشارة مفتوحة...")

            for signal_row in open_signals:
                signal_id = signal_row['id']
                symbol = signal_row['symbol']
                processed_in_cycle += 1
                update_executed = False # To track if this signal was updated in the current cycle

                try:
                    # Extract and safely convert numeric data
                    entry_price = float(signal_row['entry_price'])
                    initial_stop_loss = float(signal_row['initial_stop_loss'])
                    current_target = float(signal_row['current_target'])
                    current_stop_loss = float(signal_row['current_stop_loss'])
                    is_trailing_active = signal_row['is_trailing_active']
                    last_update_px = signal_row['last_trailing_update_price']
                    last_trailing_update_price = float(last_update_px) if last_update_px is not None else None
                    sent_at = signal_row['sent_at'] # Get signal sent timestamp

                    # Get current price from WebSocket Ticker data
                    current_price = ticker_data.get(symbol)

                    if current_price is None:
                         logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): السعر الحالي غير متوفر في بيانات Ticker.")
                         continue # Skip this signal in this cycle

                    active_signals_summary.append(f"{symbol}({signal_id}): P={current_price:.4f} T={current_target:.4f} SL={current_stop_loss:.4f} Trail={'On' if is_trailing_active else 'Off'}")

                    update_query: Optional[sql.SQL] = None
                    update_params: Tuple = ()
                    log_message: Optional[str] = None
                    notification_details: Dict[str, Any] = {'symbol': symbol, 'id': signal_id}

                    # --- Check for Max Trade Duration ---
                    if MAX_TRADE_DURATION_HOURS > 0:
                         trade_duration = datetime.now() - sent_at
                         if trade_duration > timedelta(hours=MAX_TRADE_DURATION_HOURS):
                              logger.info(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): تجاوز الحد الأقصى لمدة الصفقة ({MAX_TRADE_DURATION_HOURS} ساعات). إغلاق الصفقة.")
                              # Close the trade at the current price
                              closing_price = current_price
                              profit_pct = ((closing_price / entry_price) - 1) * 100 if entry_price > 0 else 0
                              profitable_close = closing_price > entry_price

                              update_query = sql.SQL("UPDATE signals SET hit_stop_loss = TRUE, closing_price = %s, closed_at = NOW(), profit_percentage = %s, profitable_stop_loss = %s WHERE id = %s;")
                              update_params = (closing_price, profit_pct, profitable_close, signal_id) # Use hit_stop_loss flag for closure
                              log_message = f"⏳ [Tracker] {symbol}(ID:{signal_id}): تم الإغلاق بسبب تجاوز المدة القصوى ({trade_duration}). السعر: {closing_price:.8g} (النسبة: {profit_pct:.2f}%)."
                              notification_details.update({'type': 'stop_loss_hit', 'closing_price': closing_price, 'profit_pct': profit_pct, 'profitable_sl': profitable_close}) # Reuse stop_loss_hit type but include profitable status
                              update_executed = True
                              # If trade duration closes, skip other checks for this signal
                              if update_executed:
                                   try:
                                        with conn.cursor() as update_cur:
                                            update_cur.execute(update_query, update_params)
                                        conn.commit()
                                        if log_message: logger.info(log_message)
                                        if notification_details.get('type'):
                                           send_tracking_notification(notification_details)
                                   except psycopg2.Error as db_err:
                                       logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ قاعدة بيانات أثناء تحديث تجاوز المدة القصوى: {db_err}")
                                       if conn: conn.rollback()
                                   except Exception as exec_err:
                                       logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ غير متوقع أثناء تنفيذ التحديث/الإشعار لتجاوز المدة: {exec_err}", exc_info=True)
                                       if conn: conn.rollback()
                                   continue # Move to the next signal after closing

                    # --- Check and Update Logic (Only if not closed by max duration) ---
                    if not update_executed:
                        # 1. Check for Target Hit
                        if current_price >= current_target:
                            profit_pct = ((current_target / entry_price) - 1) * 100 if entry_price > 0 else 0
                            update_query = sql.SQL("UPDATE signals SET achieved_target = TRUE, closing_price = %s, closed_at = NOW(), profit_percentage = %s WHERE id = %s;")
                            update_params = (current_target, profit_pct, signal_id)
                            log_message = f"🎯 [Tracker] {symbol}(ID:{signal_id}): تم الوصول إلى الهدف عند {current_target:.8g} (الربح: {profit_pct:+.2f}%)."
                            notification_details.update({'type': 'target_hit', 'closing_price': current_target, 'profit_pct': profit_pct})
                            update_executed = True

                        # 2. Check for Stop Loss Hit (Must be after Target check)
                        elif current_price <= current_stop_loss:
                            loss_pct = ((current_stop_loss / entry_price) - 1) * 100 if entry_price > 0 else 0
                            profitable_sl = current_stop_loss > entry_price
                            sl_type_msg = "بربح ✅" if profitable_sl else "بخسارة ❌"
                            update_query = sql.SQL("UPDATE signals SET hit_stop_loss = TRUE, closing_price = %s, closed_at = NOW(), profit_percentage = %s, profitable_stop_loss = %s WHERE id = %s;")
                            update_params = (current_stop_loss, loss_pct, profitable_sl, signal_id)
                            log_message = f"🔻 [Tracker] {symbol}(ID:{signal_id}): تم ضرب وقف الخسارة ({sl_type_msg}) عند {current_stop_loss:.8g} (النسبة: {loss_pct:.2f}%)."
                            notification_details.update({'type': 'stop_loss_hit', 'closing_price': current_stop_loss, 'profit_pct': loss_pct, 'profitable_sl': profitable_sl}) # Pass the profitable_sl flag
                            update_executed = True

                        # 3. Check for Trailing Stop Activation or Update (Only if Target or SL not hit)
                        else:
                            activation_threshold_price = entry_price * (1 + TRAILING_STOP_ACTIVATION_PROFIT_PCT)
                            # a. Activate Trailing Stop
                            if not is_trailing_active and current_price >= activation_threshold_price:
                                logger.info(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): السعر {current_price:.8g} وصل إلى عتبة تفعيل الوقف المتحرك ({activation_threshold_price:.8g}). جلب ATR...")
                                # Use the specified tracking timeframe
                                df_atr = fetch_historical_data(symbol, interval=SIGNAL_TRACKING_TIMEFRAME, days=SIGNAL_TRACKING_LOOKBACK_DAYS)
                                if df_atr is not None and not df_atr.empty:
                                    # Use the ATR period designated for entry/tracking
                                    df_atr = calculate_atr_indicator(df_atr, period=ENTRY_ATR_PERIOD)
                                    if not df_atr.empty and 'atr' in df_atr.columns and pd.notna(df_atr['atr'].iloc[-1]):
                                        current_atr_val = df_atr['atr'].iloc[-1]
                                        if current_atr_val > 0:
                                             # Calculate new stop loss based on current price and ATR
                                             new_stop_loss_calc = current_price - (TRAILING_STOP_ATR_MULTIPLIER * current_atr_val)
                                             # The new stop loss must be HIGHER than the current stop loss AND higher than the entry price (to lock in profit)
                                             new_stop_loss = max(new_stop_loss_calc, current_stop_loss, entry_price * (1 + 0.0001)) # Ensure at least a tiny profit

                                             if new_stop_loss > current_stop_loss: # Only if the new stop is actually higher than the *previous* stop
                                                update_query = sql.SQL("UPDATE signals SET is_trailing_active = TRUE, current_stop_loss = %s, last_trailing_update_price = %s WHERE id = %s;")
                                                update_params = (new_stop_loss, current_price, signal_id)
                                                log_message = f"⬆️✅ [Tracker] {symbol}(ID:{signal_id}): تم تفعيل الوقف المتحرك. السعر={current_price:.8g}, ATR={current_atr_val:.8g}. الوقف الجديد: {new_stop_loss:.8g}"
                                                notification_details.update({'type': 'trailing_activated', 'current_price': current_price, 'atr_value': current_atr_val, 'new_stop_loss': new_stop_loss, 'activation_profit_pct': TRAILING_STOP_ACTIVATION_PROFIT_PCT * 100})
                                                update_executed = True
                                             else:
                                                logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): الوقف المتحرك المحسوب ({new_stop_loss:.8g}) ليس أعلى من الوقف الحالي ({current_stop_loss:.8g}). لن يتم التفعيل.")
                                    else: logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): قيمة ATR غير صالحة ({current_atr_val}) لتفعيل الوقف المتحرك.")
                                else: logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): لا يمكن حساب ATR لتفعيل الوقف المتحرك.")
                            # b. Update Trailing Stop (Only if trailing is already active)
                            elif is_trailing_active and last_trailing_update_price is not None:
                                update_threshold_price = last_trailing_update_price * (1 + TRAILING_STOP_MOVE_INCREMENT_PCT)
                                if current_price >= update_threshold_price: # Check if price has increased enough since last update
                                    logger.info(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): السعر {current_price:.8g} وصل إلى عتبة تحديث الوقف المتحرك ({update_threshold_price:.8g}). جلب ATR...")
                                    df_recent = fetch_historical_data(symbol, interval=SIGNAL_TRACKING_TIMEFRAME, days=SIGNAL_TRACKING_LOOKBACK_DAYS)
                                    if df_recent is not None and not df_recent.empty:
                                        df_recent = calculate_atr_indicator(df_recent, period=ENTRY_ATR_PERIOD)
                                        if not df_recent.empty and 'atr' in df_recent.columns and pd.notna(df_recent['atr'].iloc[-1]):
                                             current_atr_val_update = df_recent['atr'].iloc[-1]
                                             if current_atr_val_update > 0:
                                                 # Calculate the new potential stop loss
                                                 potential_new_stop_loss = current_price - (TRAILING_STOP_ATR_MULTIPLIER * current_atr_val_update)
                                                 # The new stop loss must be higher than the current stop loss
                                                 if potential_new_stop_loss > current_stop_loss:
                                                    new_stop_loss_update = potential_new_stop_loss
                                                    update_query = sql.SQL("UPDATE signals SET current_stop_loss = %s, last_trailing_update_price = %s WHERE id = %s;")
                                                    update_params = (new_stop_loss_update, current_price, signal_id)
                                                    log_message = f"➡️🔼 [Tracker] {symbol}(ID:{signal_id}): تم تحديث الوقف المتحرك. السعر={current_price:.8g}, ATR={current_atr_val_update:.8g}. السابق={current_stop_loss:.8g}, الجديد: {new_stop_loss_update:.8g}"
                                                    notification_details.update({'type': 'trailing_updated', 'current_price': current_price, 'atr_value': current_atr_val_update, 'old_stop_loss': current_stop_loss, 'new_stop_loss': new_stop_loss_update, 'trigger_price_increase_pct': TRAILING_STOP_MOVE_INCREMENT_PCT * 100})
                                                    update_executed = True
                                                 else:
                                                     logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): الوقف المتحرك المحسوب ({potential_new_stop_loss:.8g}) ليس أعلى من الوقف الحالي ({current_stop_loss:.8g}). لن يتم التحديث.")
                                             else:
                                                  logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): قيمة ATR غير صالحة ({current_atr_val_update}) للتحديث.")
                                        else:
                                             logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): لا يمكن حساب ATR للتحديث.")
                                    else:
                                         logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): لا يمكن جلب البيانات لحساب ATR للتحديث.")
                                else:
                                     logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): السعر ({current_price:.8g}) لم يصل إلى عتبة تحديث الوقف المتحرك ({update_threshold_price:.8g}) منذ آخر تحديث عند ({last_trailing_update_price:.8g}).")


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
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ قاعدة بيانات أثناء التحديث: {db_err}")
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
                logger.debug(f"ℹ️ [Tracker] حالة نهاية الدورة ({processed_in_cycle} معالجة): {'; '.join(active_signals_summary)}")

            time.sleep(3) # Wait between tracking cycles (real-time tracking needs short intervals)

        except psycopg2.Error as db_cycle_err:
             logger.error(f"❌ [Tracker] خطأ قاعدة بيانات في دورة التتبع الرئيسية: {db_cycle_err}. محاولة إعادة الاتصال...")
             if conn: conn.rollback()
             time.sleep(30) # Wait longer on DB error
             check_db_connection()
        except Exception as cycle_err:
            logger.error(f"❌ [Tracker] خطأ غير متوقع في دورة تتبع الإشارات: {cycle_err}", exc_info=True)
            time.sleep(30) # Wait longer on unexpected error


# ---------------------- Flask Service (Optional for Webhook) (Keep existing) ----------------------
app = Flask(__name__)

@app.route('/')
def home() -> Response:
    """Simple home page to show the bot is running."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ws_alive = ws_thread.is_alive() if 'ws_thread' in globals() and ws_thread else False
    tracker_alive = tracker_thread.is_alive() if 'tracker_thread' in globals() and tracker_thread else False
    status = "running" if ws_alive and tracker_alive else "partially running"
    return Response(f"📈 Crypto Signal Bot ({status}) - Last Check: {now}", status=200, mimetype='text/plain')

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
        logger.debug(f"ℹ️ [Flask] Received webhook data: {json.dumps(data)[:200]}...") # Log only part of the data

        # Handle Callback Queries (Button Responses)
        if 'callback_query' in data:
            callback_query = data['callback_query']
            callback_id = callback_query['id']
            callback_data = callback_query.get('data')
            message_info = callback_query.get('message')
            if not message_info or not callback_data:
                 logger.warning(f"⚠️ [Flask] Callback query (ID: {callback_id}) missing message or data.")
                 try:
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
                report_thread = Thread(target=lambda: send_telegram_message(chat_id_callback, generate_performance_report(), parse_mode='Markdown'))
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
                 report_thread = Thread(target=lambda: send_telegram_message(chat_id_msg, generate_performance_report(), parse_mode='Markdown'))
                 report_thread.start()
            elif text_msg.lower() == '/status':
                 status_thread = Thread(target=handle_status_command, args=(chat_id_msg,))
                 status_thread.start()

        else:
            logger.debug("ℹ️ [Flask] Received webhook data without 'callback_query' or 'message'.")

        return "OK", 200
    except Exception as e:
         logger.error(f"❌ [Flask] Error processing webhook: {e}", exc_info=True)
         return "Internal Server Error", 500

def handle_status_command(chat_id_msg: int) -> None:
    """Separate function to handle /status command to avoid blocking the Webhook."""
    logger.info(f"ℹ️ [Flask Status] Handling /status command for chat {chat_id_msg}")
    status_msg = "⏳ جلب الحالة..."
    msg_sent = send_telegram_message(chat_id_msg, status_msg)
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

        # Check if variables exist before accessing them
        ws_status = 'نشط ✅' if 'ws_thread' in globals() and ws_thread and ws_thread.is_alive() else 'غير نشط ❌'
        tracker_status = 'نشط ✅' if 'tracker_thread' in globals() and tracker_thread and tracker_thread.is_alive() else 'غير نشط ❌'
        final_status_msg = (
            f"🤖 *حالة البوت:*\n"
            f"- تتبع الأسعار (WS): {ws_status}\n"
            f"- تتبع الإشارات: {tracker_status}\n"
            f"- الإشارات النشطة: *{open_count}* / {MAX_OPEN_TRADES}\n"
            f"- وقت الخادم الحالي: {datetime.now().strftime('%H:%M:%S')}"
        )
        # Edit the original message
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
        logger.error(f"❌ [Flask Status] Error getting/editing status details for chat {chat_id_msg}: {status_err}", exc_info=True)
        send_telegram_message(chat_id_msg, "❌ حدث خطأ أثناء جلب تفاصيل الحالة.")


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
        serve(app, host=host, port=port, threads=6)
    except ImportError:
         logger.warning("⚠️ [Flask] 'waitress' not installed. Falling back to Flask development server (NOT recommended for production).")
         try:
             app.run(host=host, port=port)
         except Exception as flask_run_err:
              logger.critical(f"❌ [Flask] Failed to start development server: {flask_run_err}", exc_info=True)
    except Exception as serve_err:
         logger.critical(f"❌ [Flask] Failed to start server (waitress?): {serve_err}", exc_info=True)

# ---------------------- Main Loop and Check Function (Adjusted scan frequency) ----------------------
def main_loop() -> None:
    """Main loop to scan pairs and generate signals."""
    symbols_to_scan = get_crypto_symbols()
    if not symbols_to_scan:
        logger.critical("❌ [Main] لم يتم تحميل أو التحقق من أي أزواج صالحة. لا يمكن المتابعة.")
        return

    logger.info(f"✅ [Main] تم تحميل {len(symbols_to_scan)} زوجًا صالحًا للفحص.")
    # No need for last_full_scan_time if we use a fixed sleep time
    # last_full_scan_time = time.time()

    while True:
        try:
            scan_start_time = time.time()
            logger.info("+" + "-"*60 + "+")
            logger.info(f"🔄 [Main] بدء دورة فحص السوق - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("+" + "-"*60 + "+")

            if not check_db_connection() or not conn:
                logger.error("❌ [Main] تخطي دورة الفحص بسبب فشل اتصال قاعدة البيانات.")
                time.sleep(60)
                continue

            # 1. Check the current number of open signals
            open_count = 0
            try:
                 with conn.cursor() as cur_check:
                    cur_check.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;")
                    open_count = (cur_check.fetchone() or {}).get('count', 0)
            except psycopg2.Error as db_err:
                 logger.error(f"❌ [Main] خطأ قاعدة بيانات أثناء التحقق من عدد الإشارات المفتوحة: {db_err}. تخطي الدورة.")
                 if conn: conn.rollback()
                 time.sleep(60)
                 continue

            logger.info(f"ℹ️ [Main] الإشارات المفتوحة حالياً: {open_count} / {MAX_OPEN_TRADES}")
            if open_count >= MAX_OPEN_TRADES:
                logger.info(f"⚠️ [Main] تم الوصول إلى الحد الأقصى لعدد الإشارات المفتوحة. في انتظار...")
                time.sleep(30) # Wait 30 seconds before re-checking open count
                continue

            # 2. Iterate through the list of symbols and scan them
            processed_in_loop = 0
            signals_generated_in_loop = 0
            slots_available = MAX_OPEN_TRADES - open_count

            for symbol in symbols_to_scan:
                 if slots_available <= 0:
                      logger.info(f"ℹ️ [Main] تم الوصول إلى الحد الأقصى ({MAX_OPEN_TRADES}) أثناء الفحص. إيقاف فحص الأزواج لهذه الدورة.")
                      break

                 processed_in_loop += 1
                 logger.debug(f"🔍 [Main] فحص {symbol} ({processed_in_loop}/{len(symbols_to_scan)})...")

                 try:
                    # a. Check if there is already an open signal for this symbol
                    with conn.cursor() as symbol_cur:
                        symbol_cur.execute("SELECT 1 FROM signals WHERE symbol = %s AND achieved_target = FALSE AND hit_stop_loss = FALSE LIMIT 1;", (symbol,))
                        if symbol_cur.fetchone():
                            logger.debug(f"ℹ️ [Main] توجد إشارة مفتوحة بالفعل للزوج {symbol}. تخطي.")
                            continue

                    # b. Fetch historical data (using SIGNAL_GENERATION_TIMEFRAME)
                    df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_hist is None or df_hist.empty:
                        logger.debug(f"ℹ️ [Main] لا تتوفر بيانات تاريخية كافية للزوج {symbol}.")
                        continue

                    # c. Apply the strategy and generate signal
                    # Use the new BottomFishingStrategy
                    strategy = BottomFishingStrategy(symbol)
                    df_indicators = strategy.populate_indicators(df_hist)
                    if df_indicators is None:
                        logger.debug(f"ℹ️ [Main] فشل حساب المؤشرات للزوج {symbol}.")
                        continue

                    potential_signal = strategy.generate_buy_signal(df_indicators)

                    # d. Insert signal and send alert
                    if potential_signal:
                        logger.info(f"✨ [Main] تم العثور على إشارة محتملة للزوج {symbol}! (النقاط: {potential_signal.get('r2_score', 0):.2f}) التحقق النهائي والإدراج...")
                        # Re-check open count just before inserting to avoid exceeding the limit due to concurrent signals
                        with conn.cursor() as final_check_cur:
                             final_check_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;")
                             final_open_count = (final_check_cur.fetchone() or {}).get('count', 0)

                             if final_open_count < MAX_OPEN_TRADES:
                                 if insert_signal_into_db(potential_signal):
                                     send_telegram_alert(potential_signal, SIGNAL_GENERATION_TIMEFRAME)
                                     signals_generated_in_loop += 1
                                     slots_available -= 1
                                     # Add a small delay after generating a signal to avoid rapid-fire signals
                                     time.sleep(5)
                                 else:
                                     logger.error(f"❌ [Main] فشل في إدراج إشارة الزوج {symbol} في قاعدة البيانات.")
                             else:
                                 logger.warning(f"⚠️ [Main] تم الوصول إلى الحد الأقصى ({final_open_count}) قبل إدراج إشارة الزوج {symbol}. تم تجاهل الإشارة.")
                                 # Break out of the symbol loop if the limit is reached
                                 break

                 except psycopg2.Error as db_loop_err:
                      logger.error(f"❌ [Main] خطأ قاعدة بيانات أثناء معالجة الزوج {symbol}: {db_loop_err}. الانتقال إلى الزوج التالي...")
                      if conn: conn.rollback()
                      continue
                 except Exception as symbol_proc_err:
                      logger.error(f"❌ [Main] خطأ عام أثناء معالجة الزوج {symbol}: {symbol_proc_err}", exc_info=True)
                      continue

                 # Small delay between processing symbols to reduce load
                 time.sleep(0.1)

            # 3. Wait before starting the next cycle
            scan_duration = time.time() - scan_start_time
            # Adjust wait time to achieve a cycle of approximately 30 seconds
            # Ensure the wait time is not negative if scan_duration is long
            wait_time = max(0, 30 - scan_duration) # Target 30 seconds total cycle duration

            logger.info(f"🏁 [Main] انتهت دورة الفحص. الإشارات التي تم توليدها: {signals_generated_in_loop}. مدة الفحص: {scan_duration:.2f} ثانية.")
            logger.info(f"⏳ [Main] انتظار {wait_time:.1f} ثانية للدورة التالية...")
            time.sleep(wait_time)

        except KeyboardInterrupt:
             logger.info("🛑 [Main] تم طلب الإيقاف (KeyboardInterrupt). إغلاق...")
             break
        except psycopg2.Error as db_main_err:
             logger.error(f"❌ [Main] خطأ قاعدة بيانات قاتل في الدورة الرئيسية: {db_main_err}. محاولة إعادة الاتصال...")
             if conn: conn.rollback()
             time.sleep(60) # Wait longer on fatal DB error
             try:
                 init_db()
             except Exception as recon_err:
                 logger.critical(f"❌ [Main] فشل إعادة الاتصال بقاعدة البيانات: {recon_err}. الخروج...")
                 break
        except Exception as main_err:
            logger.error(f"❌ [Main] خطأ غير متوقع في الدورة الرئيسية: {main_err}", exc_info=True)
            logger.info("ℹ️ [Main] انتظار 60 ثانية قبل إعادة المحاولة...") # Reduce wait time on general error
            time.sleep(60)

def cleanup_resources() -> None:
    """Closes used resources like the database connection."""
    global conn
    logger.info("ℹ️ [Cleanup] إغلاق الموارد...")
    if conn:
        try:
            conn.close()
            logger.info("✅ [DB] تم إغلاق اتصال قاعدة البيانات.")
        except Exception as close_err:
            logger.error(f"⚠️ [DB] خطأ أثناء إغلاق اتصال قاعدة البيانات: {close_err}")
    logger.info("✅ [Cleanup] اكتمل تنظيف الموارد.")


# ---------------------- Main Entry Point ----------------------
if __name__ == "__main__":
    logger.info("🚀 بدء تشغيل بوت إشارات التداول...")
    logger.info(f"الوقت المحلي: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | وقت UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

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
        logger.info("✅ [Main] تم بدء تشغيل مؤشر WebSocket.")
        logger.info("ℹ️ [Main] انتظار 5 ثوانٍ لتهيئة WebSocket...")
        time.sleep(5) # Give WebSocket a moment to connect and receive initial data
        if not ticker_data:
             logger.warning("⚠️ [Main] لم يتم تلقي بيانات أولية من WebSocket بعد 5 ثوانٍ.")
        else:
             logger.info(f"✅ [Main] تم تلقي بيانات أولية من WebSocket لـ {len(ticker_data)} زوجًا.")


        # 3. Start Signal Tracker
        tracker_thread = Thread(target=track_signals, daemon=True, name="TrackerThread")
        tracker_thread.start()
        logger.info("✅ [Main] تم بدء تشغيل متتبع الإشارات.")

        # 4. Start Flask Server (if Webhook configured)
        if WEBHOOK_URL:
            flask_thread = Thread(target=run_flask, daemon=True, name="FlaskThread")
            flask_thread.start()
            logger.info("✅ [Main] تم بدء تشغيل ثريد Flask Webhook.")
        else:
             logger.info("ℹ️ [Main] لم يتم تكوين Webhook URL، لن يتم بدء تشغيل خادم Flask.")

        # 5. Start the main loop
        main_loop()

    except Exception as startup_err:
        logger.critical(f"❌ [Main] حدث خطأ فادح أثناء بدء التشغيل أو في الدورة الرئيسية: {startup_err}", exc_info=True)
    finally:
        logger.info("🛑 [Main] يتم إيقاف تشغيل البرنامج...")
        # send_telegram_message(CHAT_ID, "⚠️ Alert: Trading bot is shutting down now.") # Uncomment to send alert on shutdown
        cleanup_resources()
        logger.info("👋 [Main] تم إيقاف تشغيل بوت إشارات التداول.")
        os._exit(0)

