import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql, OperationalError, InterfaceError # For secure queries and specific errors
from psycopg2.extras import RealDictCursor # To get results as dictionaries
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException, BinanceRequestException # Specific Binance errors
from flask import Flask, request, Response
from threading import Thread
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Tuple, Any, Union # To add Type Hinting

# ---------------------- Logging Setup ----------------------
logging.basicConfig(
    level=logging.INFO, # Can change this to logging.DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # Add logger name
    handlers=[
        logging.FileHandler('crypto_bot_elliott_fib.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
# Use a specific logger name instead of the root
logger = logging.getLogger('CryptoBot')

# ---------------------- Load Environment Variables ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
    # Use a default value of None if the variable does not exist
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
except Exception as e:
     logger.critical(f"❌ Failed to load essential environment variables: {e}")
     exit(1) # Use a non-zero exit code to indicate an error

logger.info(f"Binance API Key: {'Available' if API_KEY else 'Not Available'}")
logger.info(f"Telegram Token: {TELEGRAM_TOKEN[:10]}...{'*' * (len(TELEGRAM_TOKEN)-10)}")
logger.info(f"Telegram Chat ID: {CHAT_ID}")
logger.info(f"Database URL: {'Available' if DB_URL else 'Not Available'}")
logger.info(f"Webhook URL: {WEBHOOK_URL if WEBHOOK_URL else 'Not Set'}")

# ---------------------- Constants and Global Variables (Adjusted for 15m frame check) ----------------------
TRADE_VALUE: float = 10.0         # Default trade value in USDT (Keep small for testing)
MAX_OPEN_TRADES: int = 5          # Maximum number of open trades simultaneously (Increased slightly for scalping)
SIGNAL_GENERATION_TIMEFRAME: str = '15m' # Timeframe for signal generation (Changed to 15m)
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 7 # Increased historical data lookback for 15m timeframe
SIGNAL_TRACKING_TIMEFRAME: str = '15m' # Timeframe for signal tracking and target updates (Changed to 15m)
SIGNAL_TRACKING_LOOKBACK_DAYS: int = 3   # Increased historical data lookback in days for signal tracking

# --- New Constants for Multi-Timeframe Confirmation ---
CONFIRMATION_TIMEFRAME: str = '30m' # Larger timeframe for trend confirmation (Changed to 30m)
CONFIRMATION_LOOKBACK_DAYS: int = 14 # Historical data lookback for confirmation timeframe (Increased for 30m)

# --- Parameters for Improved Entry Point ---
# --- MODIFIED: Increased tolerance slightly for less strict entry ---
ENTRY_POINT_EMA_PROXIMITY_PCT: float = 0.003 # Price must be within this % of signal timeframe EMA_SHORT (Increased tolerance slightly)
ENTRY_POINT_RECENT_CANDLE_LOOKBACK: int = 2 # Look back this many candles on signal timeframe for bullish sign (Reduced lookback)

# =============================================================================
# --- Indicator Parameters (Adjusted for 15m Signal and 30m Confirmation) ---
# =============================================================================
RSI_PERIOD: int = 14 # Standard RSI period
RSI_OVERSOLD: int = 30
RSI_OVERBOUGHT: int = 70
EMA_SHORT_PERIOD: int = 13 # Adjusted for 15m
EMA_LONG_PERIOD: int = 34 # Adjusted for 15m
VWMA_PERIOD: int = 21 # Adjusted for 15m
SWING_ORDER: int = 3 # Not used in current strategy logic
FIB_LEVELS_TO_CHECK: List[float] = [0.382, 0.5, 0.618] # Not used in current strategy logic
FIB_TOLERANCE: float = 0.005 # Not used in current strategy logic
LOOKBACK_FOR_SWINGS: int = 50 # Not used in current strategy logic
ENTRY_ATR_PERIOD: int = 14 # Adjusted for 15m
ENTRY_ATR_MULTIPLIER: float = 1.75 # ATR Multiplier for initial target (Adjusted slightly)
BOLLINGER_WINDOW: int = 20 # Standard Bollinger period
BOLLINGER_STD_DEV: int = 2 # Standard Bollinger std dev
MACD_FAST: int = 12 # Standard MACD fast period
MACD_SLOW: int = 26 # Standard MACD slow period
MACD_SIGNAL: int = 9 # Standard MACD signal period
ADX_PERIOD: int = 14 # Standard ADX period
SUPERTREND_PERIOD: int = 10 # Standard Supertrend period
SUPERTREND_MULTIPLIER: float = 3.0 # Adjusted Supertrend multiplier slightly

# --- Parameters for Dynamic Target Update ---
DYNAMIC_TARGET_APPROACH_PCT: float = 0.003 # Percentage proximity to target to trigger re-evaluation (e.g., 0.3%) (Increased slightly)
DYNAMIC_TARGET_EXTENSION_ATR_MULTIPLIER: float = 1.0 # ATR multiplier for extending the target (Increased)
MAX_DYNAMIC_TARGET_UPDATES: int = 3 # Maximum number of times a target can be dynamically updated for a single signal (Increased)
MIN_ADX_FOR_DYNAMIC_UPDATE: int = 25 # Minimum ADX value to consider dynamic target update (Increased slightly)

# --- MODIFIED: Lowered minimum profit margin for less strictness ---
MIN_PROFIT_MARGIN_PCT: float = 1.0 # Increased minimum profit margin (Lowered for less strictness)
# --- MODIFIED: Lowered minimum volume check for less strictness ---
MIN_VOLUME_15M_USDT: float = 400000.0 # Increased minimum volume check (using 15m data now) (Lowered for less strictness)

RECENT_EMA_CROSS_LOOKBACK: int = 3 # Adjusted for 15m
# --- MODIFIED: Lowered minimum ADX trend strength for less strictness ---
MIN_ADX_TREND_STRENGTH: int = 20 # Increased minimum ADX trend strength for essential condition (Lowered for less strictness)
MACD_HIST_INCREASE_CANDLES: int = 2 # Reduced lookback for MACD Hist increase
OBV_INCREASE_CANDLES: int = 2 # Reduced lookback for OBV increase
# =============================================================================

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
cur: Optional[psycopg2.extensions.cursor] = None
client: Optional[Client] = None
ticker_data: Dict[str, float] = {}

# ---------------------- Binance Client Setup ----------------------
try:
    logger.info("ℹ️ [Binance] Initializing Binance client...")
    client = Client(API_KEY, API_SECRET)
    client.ping()
    server_time = client.get_server_time()
    logger.info(f"✅ [Binance] Binance client initialized successfully. Server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
except (BinanceRequestException, BinanceAPIException) as binance_err:
     logger.critical(f"❌ [Binance] Binance API/Request Error: {binance_err}")
     exit(1)
except Exception as e:
    logger.critical(f"❌ [Binance] Unexpected failure initializing Binance client: {e}", exc_info=True)
    exit(1)

# ---------------------- Additional Indicator Functions ----------------------
def get_fear_greed_index() -> str:
    classification_translation_ar = {
        "Extreme Fear": "خوف شديد", "Fear": "خوف", "Neutral": "محايد",
        "Greed": "جشع", "Extreme Greed": "جشع شديد",
    }
    url = "https://api.alternative.me/fng/"
    logger.debug(f"ℹ️ [Indicators] Fetching Fear and Greed Index from {url}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        value = int(data["data"][0]["value"])
        classification_en = data["data"][0]["value_classification"]
        classification_ar = classification_translation_ar.get(classification_en, classification_en)
        logger.debug(f"✅ [Indicators] Fear and Greed Index: {value} ({classification_ar})")
        return f"{value} ({classification_ar})"
    except requests.exceptions.RequestException as e:
         logger.error(f"❌ [Indicators] Network error while fetching Fear and Greed Index: {e}")
         return "N/A (Network Error)"
    except Exception as e:
        logger.error(f"❌ [Indicators] Error while fetching Fear and Greed Index: {e}", exc_info=True)
        return "N/A (Error)"

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """Fetches historical data for a given pair and timeframe."""
    if not client:
        logger.error(f"❌ [Data] Binance client not initialized to fetch data for {symbol}.")
        return None
    try:
        start_dt = datetime.utcnow() - timedelta(days=days + 1)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        logger.debug(f"ℹ️ [Data] Fetching {interval} data for {symbol} since {start_str} (max 1000 klines)...")
        klines = client.get_historical_klines(symbol, interval, start_str, limit=1000)
        if not klines:
            logger.warning(f"⚠️ [Data] No historical data ({interval}) for {symbol}.")
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
            logger.warning(f"⚠️ [Data] DataFrame for {symbol} is empty after dropping NaN values.")
            return None
        logger.debug(f"✅ [Data] Fetched {len(df)} klines ({interval}) for {symbol}.")
        return df
    except (BinanceAPIException, BinanceRequestException) as binance_err:
         logger.error(f"❌ [Data] Binance error while fetching data for {symbol}: {binance_err}")
         return None
    except Exception as e:
        logger.error(f"❌ [Data] Error while fetching historical data for {symbol}: {e}", exc_info=True)
        return None

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculates Exponential Moving Average (EMA)."""
    if series is None or series.isnull().all() or len(series) < span:
        logger.debug(f"⚠️ [Indicators] Insufficient data to calculate EMA span={span}.")
        return pd.Series(index=series.index if series is not None else None, dtype=float)
    ema = series.ewm(span=span, adjust=False).mean()
    logger.debug(f"✅ [Indicators] Calculated EMA span={span}.")
    return ema

def calculate_vwma(df: pd.DataFrame, period: int) -> pd.Series:
    """Calculates Volume Weighted Moving Average (VWMA)."""
    df_calc = df.copy()
    if not all(col in df_calc.columns for col in ['close', 'volume']) or df_calc[['close', 'volume']].isnull().all().any() or len(df_calc) < period:
        logger.debug(f"⚠️ [Indicators] Insufficient data to calculate VWMA period={period}.")
        return pd.Series(index=df_calc.index, dtype=float)
    df_calc['price_volume'] = df_calc['close'] * df_calc['volume']
    rolling_price_volume_sum = df_calc['price_volume'].rolling(window=period, min_periods=period).sum()
    rolling_volume_sum = df_calc['volume'].rolling(window=period, min_periods=period).sum()
    vwma = rolling_price_volume_sum / rolling_volume_sum.replace(0, np.nan)
    logger.debug(f"✅ [Indicators] Calculated VWMA period={period}.")
    return vwma

def get_btc_trend_4h() -> str:
    """Calculates the Bitcoin trend on the 4-hour timeframe."""
    logger.debug("ℹ️ [Indicators] Calculating Bitcoin trend on 4-hour timeframe...")
    try:
        df = fetch_historical_data("BTCUSDT", interval=Client.KLINE_INTERVAL_4HOUR, days=10)
        if df is None or df.empty or len(df) < 51: # Ensure enough data for EMA50
            logger.warning("⚠️ [Indicators] Insufficient BTC/USDT 4H data to determine trend.")
            return "N/A (Insufficient Data)"
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df.dropna(subset=['close'], inplace=True)
        if len(df) < 50:
            logger.warning("⚠️ [Indicators] Insufficient BTC/USDT 4H data after NaN removal.")
            return "N/A (Insufficient Data)"
        ema20 = calculate_ema(df['close'], 20).iloc[-1]
        ema50 = calculate_ema(df['close'], 50).iloc[-1]
        current_close = df['close'].iloc[-1]
        if pd.isna(ema20) or pd.isna(ema50) or pd.isna(current_close):
            logger.warning("⚠️ [Indicators] Error calculating EMA20/EMA50 for BTC/USDT 4H.")
            return "N/A (Calculation Error)"
        diff_ema20_pct = abs(current_close - ema20) / current_close if current_close > 0 else 0
        if current_close > ema20 > ema50: trend = "Uptrend 📈"
        elif current_close < ema20 < ema50: trend = "Downtrend 📉"
        elif diff_ema20_pct < 0.005: trend = "Sideways 🔄" # Sideways
        else: trend = "Volatile 🔀" # Volatile
        logger.debug(f"✅ [Indicators] Bitcoin 4H Trend: {trend}")
        return trend
    except Exception as e:
        logger.error(f"❌ [Indicators] Error while calculating Bitcoin trend on 4-hour timeframe: {e}", exc_info=True)
        return "N/A (Error)"

# ---------------------- Database Connection Setup ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    """Initializes the database connection and creates tables if they don't exist."""
    global conn, cur
    logger.info("[DB] Starting database initialization...")
    for attempt in range(retries):
        try:
            logger.info(f"[DB] Attempting to connect (Attempt {attempt + 1}/{retries})..." )
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            cur = conn.cursor()
            logger.info("✅ [DB] Successfully connected to the database.")

            logger.info("[DB] Checking for/creating 'signals' table...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    initial_target DOUBLE PRECISION NOT NULL,
                    initial_stop_loss DOUBLE PRECISION DEFAULT 0.0,
                    current_target DOUBLE PRECISION NOT NULL,
                    current_stop_loss DOUBLE PRECISION DEFAULT 0.0,
                    r2_score DOUBLE PRECISION,
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
                    time_to_target_seconds BIGINT,
                    dynamic_updates_count INTEGER DEFAULT 0
                );""")
            conn.commit()
            logger.info("✅ [DB] Checked for/created 'signals' table.")

            columns_to_add = {
                "time_to_target_seconds": "BIGINT",
                "dynamic_updates_count": "INTEGER DEFAULT 0"
            }
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'signals' AND table_schema = 'public';")
            existing_columns = {row['column_name'] for row in cur.fetchall()}

            for col_name, col_type in columns_to_add.items():
                if col_name not in existing_columns:
                    logger.info(f"[DB] Adding column '{col_name}' to 'signals' table...")
                    cur.execute(sql.SQL(f"ALTER TABLE signals ADD COLUMN IF NOT EXISTS {col_name} {col_type};"))
                    conn.commit()
                    logger.info(f"✅ [DB] Added column '{col_name}'.")

            logger.info("[DB] Checking for/creating 'market_dominance' table...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS market_dominance (
                    id SERIAL PRIMARY KEY,
                    recorded_at TIMESTAMP DEFAULT NOW(),
                    btc_dominance DOUBLE PRECISION,
                    eth_dominance DOUBLE PRECISION
                );
            """)
            conn.commit()
            logger.info("✅ [DB] Checked for/created 'market_dominance' table.")
            logger.info("✅ [DB] Database initialization successful.")
            return
        except OperationalError as op_err:
            logger.error(f"❌ [DB] Operational Error (Attempt {attempt + 1}): {op_err}")
            if conn: conn.rollback()
            if attempt == retries - 1: raise op_err
            time.sleep(delay)
        except Exception as e:
            logger.critical(f"❌ [DB] Unexpected failure initializing database (Attempt {attempt + 1}): {e}", exc_info=True)
            if conn: conn.rollback()
            if attempt == retries - 1: raise e
            time.sleep(delay)
    logger.critical("❌ [DB] Failed to connect to the database after multiple attempts.")
    exit(1)

def check_db_connection() -> bool:
    """Checks if the database connection is active and re-initializes it if necessary."""
    global conn, cur
    try:
        if conn is None or conn.closed != 0:
            logger.warning("⚠️ [DB] Connection closed/non-existent. Re-initializing...")
            init_db()
            return True
        else:
             with conn.cursor() as check_cur: # Use a temporary cursor
                  check_cur.execute("SELECT 1;")
                  check_cur.fetchone()
             return True
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [DB] Database connection lost ({e}). Re-initializing...")
        try: init_db(); return True
        except Exception as recon_err: logger.error(f"❌ [DB] Failed to re-connect: {recon_err}"); return False
    except Exception as e:
        logger.error(f"❌ [DB] Unexpected error while checking connection: {e}", exc_info=True)
        try: init_db(); return True
        except Exception as recon_err: logger.error(f"❌ [DB] Failed to re-connect: {recon_err}"); return False

def convert_np_values(obj: Any) -> Any:
    """Converts NumPy values to native Python types for serialization."""
    if isinstance(obj, dict): return {k: convert_np_values(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_np_values(item) for item in obj]
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, (np.integer, np.int_)): return int(obj)
    elif isinstance(obj, (np.floating, np.float64)): return float(obj)
    elif isinstance(obj, (np.bool_)): return bool(obj)
    elif pd.isna(obj): return None
    else: return obj

# ---------------------- Reading and Validating Symbols List ----------------------
def get_crypto_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """Reads and validates the list of cryptocurrency symbols."""
    raw_symbols: List[str] = []
    logger.info(f"ℹ️ [Data] Reading symbol list from '{filename}'...")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        if not os.path.exists(file_path): file_path = os.path.abspath(filename)
        if not os.path.exists(file_path): logger.error(f"❌ [Data] File '{filename}' not found."); return []
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = [f"{line.strip().upper().replace('USDT', '')}USDT" for line in f if line.strip() and not line.startswith('#')]
        raw_symbols = sorted(list(set(raw_symbols)))
        logger.info(f"ℹ️ [Data] Read {len(raw_symbols)} symbols from '{file_path}'.")
    except Exception as e: logger.error(f"❌ [Data] Error reading file '{filename}': {e}", exc_info=True); return []
    if not raw_symbols: logger.warning("⚠️ [Data] Symbol list is empty."); return []

    if not client: logger.error("❌ [Data Validation] Binance client not initialized."); return raw_symbols
    try:
        logger.info("ℹ️ [Data Validation] Validating symbols against Binance API...")
        exchange_info = client.get_exchange_info()
        valid_trading_usdt_symbols = {s['symbol'] for s in exchange_info['symbols'] if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING' and s.get('isSpotTradingAllowed') is True}
        validated_symbols = [symbol for symbol in raw_symbols if symbol in valid_trading_usdt_symbols]
        removed_count = len(raw_symbols) - len(validated_symbols)
        if removed_count > 0: logger.warning(f"⚠️ [Data Validation] Removed {removed_count} invalid symbols.")
        logger.info(f"✅ [Data Validation] Using {len(validated_symbols)} valid symbols.")
        return validated_symbols
    except Exception as api_err:
         logger.error(f"❌ [Data Validation] Error validating Binance symbols: {api_err}", exc_info=True)
         return raw_symbols

# ---------------------- WebSocket Management for Ticker Prices ----------------------
def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
    """Handles WebSocket messages for spot prices."""
    global ticker_data
    try:
        if isinstance(msg, list):
            for ticker_item in msg:
                symbol = ticker_item.get('s')
                price_str = ticker_item.get('c')
                if symbol and 'USDT' in symbol and price_str:
                    try: ticker_data[symbol] = float(price_str)
                    except ValueError: logger.warning(f"⚠️ [WS] Invalid price for {symbol}: '{price_str}'")
        elif isinstance(msg, dict) and msg.get('stream') and msg.get('data'): # Handle combined streams format
            for ticker_item in msg.get('data', []):
                symbol = ticker_item.get('s')
                price_str = ticker_item.get('c')
                if symbol and 'USDT' in symbol and price_str:
                    try: ticker_data[symbol] = float(price_str)
                    except ValueError: logger.warning(f"⚠️ [WS] Invalid price for {symbol} in combined stream: '{price_str}'")
        elif isinstance(msg, dict) and msg.get('e') == 'error':
            logger.error(f"❌ [WS] Error from WebSocket: {msg.get('m', 'No details')}")
    except Exception as e: logger.error(f"❌ [WS] Error processing ticker message: {e}", exc_info=True)

def run_ticker_socket_manager() -> None:
    """Manages the WebSocket connection for ticker prices."""
    while True:
        try:
            logger.info("ℹ️ [WS] Starting WebSocket manager for ticker prices...")
            twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
            twm.start()
            stream_name = twm.start_miniticker_socket(callback=handle_ticker_message)
            logger.info(f"✅ [WS] Started WebSocket stream: {stream_name}")
            twm.join()
            logger.warning("⚠️ [WS] WebSocket manager stopped. Restarting...")
        except Exception as e: logger.error(f"❌ [WS] Fatal error in WebSocket manager: {e}. Restarting in 15 seconds...", exc_info=True)
        time.sleep(15)

# ---------------------- Technical Indicator Functions ----------------------
def calculate_rsi_indicator(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """Calculates Relative Strength Index (RSI)."""
    df = df.copy()
    if 'close' not in df.columns or df['close'].isnull().all() or len(df) < period:
        logger.debug(f"⚠️ [Indicators] Insufficient data to calculate RSI period={period}.")
        df['rsi'] = np.nan; return df
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = (100 - (100 / (1 + rs))).ffill().fillna(50)
    logger.debug(f"✅ [Indicators] Calculated RSI period={period}.")
    return df

def calculate_atr_indicator(df: pd.DataFrame, period: int = ENTRY_ATR_PERIOD) -> pd.DataFrame:
    """Calculates Average True Range (ATR)."""
    df = df.copy()
    if not all(col in df.columns for col in ['high', 'low', 'close']) or df[['high', 'low', 'close']].isnull().all().any() or len(df) < period + 1:
        logger.debug(f"⚠️ [Indicators] Insufficient data to calculate ATR period={period}.")
        df['atr'] = np.nan; return df
    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)
    df['atr'] = tr.ewm(span=period, adjust=False).mean()
    logger.debug(f"✅ [Indicators] Calculated ATR period={period}.")
    return df

def calculate_bollinger_bands(df: pd.DataFrame, window: int = BOLLINGER_WINDOW, num_std: int = BOLLINGER_STD_DEV) -> pd.DataFrame:
    """Calculates Bollinger Bands."""
    df = df.copy()
    if 'close' not in df.columns or df['close'].isnull().all() or len(df) < window:
        logger.debug(f"⚠️ [Indicators] Insufficient data to calculate Bollinger Bands window={window}.")
        df['bb_middle'] = np.nan; df['bb_upper'] = np.nan; df['bb_lower'] = np.nan; return df
    df['bb_middle'] = df['close'].rolling(window=window).mean()
    df['bb_std'] = df['close'].rolling(window=window).std()
    df['bb_upper'] = df['bb_middle'] + num_std * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - num_std * df['bb_std']
    logger.debug(f"✅ [Indicators] Calculated Bollinger Bands window={window}.")
    return df

def calculate_macd(df: pd.DataFrame, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> pd.DataFrame:
    """Calculates MACD indicator."""
    df = df.copy()
    min_len = max(fast, slow, signal)
    if 'close' not in df.columns or df['close'].isnull().all() or len(df) < min_len:
        logger.debug(f"⚠️ [Indicators] Insufficient data to calculate MACD fast={fast}, slow={slow}, signal={signal}.")
        df['macd'] = np.nan; df['macd_signal'] = np.nan; df['macd_hist'] = np.nan; return df
    ema_fast = calculate_ema(df['close'], fast)
    ema_slow = calculate_ema(df['close'], slow)
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = calculate_ema(df['macd'], signal)
    df['macd_hist'] = df['macd'] - df['macd_signal']
    logger.debug(f"✅ [Indicators] Calculated MACD fast={fast}, slow={slow}, signal={signal}.")
    return df

def calculate_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    """Calculates ADX indicator."""
    df_calc = df.copy() # Work on a copy
    if not all(col in df_calc.columns for col in ['high', 'low', 'close']) or df_calc[['high', 'low', 'close']].isnull().all().any() or len(df_calc) < period * 2:
        logger.debug(f"⚠️ [Indicators] Insufficient data to calculate ADX period={period}.")
        df_calc['adx'] = np.nan; df_calc['di_plus'] = np.nan; df_calc['di_minus'] = np.nan; return df_calc
    df_calc['tr'] = pd.concat([df_calc['high'] - df_calc['low'], abs(df_calc['high'] - df_calc['close'].shift(1)), abs(df_calc['low'] - df_calc['close'].shift(1))], axis=1).max(axis=1, skipna=False)
    df_calc['up_move'] = df_calc['high'] - df_calc['high'].shift(1)
    df_calc['down_move'] = df_calc['low'].shift(1) - df_calc['low']
    df_calc['+dm'] = np.where((df_calc['up_move'] > df_calc['down_move']) & (df_calc['up_move'] > 0), df_calc['up_move'], 0)
    df_calc['-dm'] = np.where((df_calc['down_move'] > df_calc['up_move']) & (df_calc['down_move'] > 0), df_calc['down_move'], 0)
    alpha = 1 / period
    df_calc['tr_smooth'] = df_calc['tr'].ewm(alpha=alpha, adjust=False).mean()
    df_calc['+dm_smooth'] = df_calc['+dm'].ewm(alpha=alpha, adjust=False).mean()
    df_calc['-dm_smooth'] = df_calc['-dm'].ewm(alpha=alpha, adjust=False).mean()
    df_calc['di_plus'] = np.where(df_calc['tr_smooth'] > 0, 100 * (df_calc['+dm_smooth'] / df_calc['tr_smooth']), 0)
    df_calc['di_minus'] = np.where(df_calc['tr_smooth'] > 0, 100 * (df_calc['-dm_smooth'] / df_calc['tr_smooth']), 0)
    di_sum = df_calc['di_plus'] + df_calc['di_minus']
    df_calc['dx'] = np.where(di_sum > 0, 100 * abs(df_calc['di_plus'] - df_calc['di_minus']) / di_sum, 0)
    df_calc['adx'] = df_calc['dx'].ewm(alpha=alpha, adjust=False).mean()
    logger.debug(f"✅ [Indicators] Calculated ADX period={period}.")
    return df_calc[['adx', 'di_plus', 'di_minus']]

def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Volume Weighted Average Price (VWAP)."""
    df = df.copy()
    if not all(col in df.columns for col in ['high', 'low', 'close', 'volume']) or df[['high', 'low', 'close', 'volume']].isnull().all().any() or not isinstance(df.index, pd.DatetimeIndex):
        logger.debug("⚠️ [Indicators] Insufficient data to calculate VWAP.")
        df['vwap'] = np.nan; return df
    try:
        df['date'] = df.index.date
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_vol'] = df['typical_price'] * df['volume']
        df['cum_tp_vol'] = df.groupby('date')['tp_vol'].cumsum()
        df['cum_volume'] = df.groupby('date')['volume'].cumsum()
        df['vwap'] = np.where(df['cum_volume'] > 0, df['cum_tp_vol'] / df['cum_volume'], np.nan)
        df['vwap'] = df['vwap'].bfill()
        df.drop(columns=['date', 'typical_price', 'tp_vol', 'cum_tp_vol', 'cum_volume'], inplace=True, errors='ignore')
        logger.debug("✅ [Indicators] Calculated VWAP.")
    except Exception as e: logger.error(f"❌ [Indicator VWAP] Error: {e}"); df['vwap'] = np.nan
    return df

def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates On-Balance Volume (OBV)."""
    df = df.copy()
    if not all(col in df.columns for col in ['close', 'volume']) or df[['close', 'volume']].isnull().all().any() or not pd.api.types.is_numeric_dtype(df['close']) or not pd.api.types.is_numeric_dtype(df['volume']):
        logger.debug("⚠️ [Indicators] Insufficient data to calculate OBV.")
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
    logger.debug("✅ [Indicators] Calculated OBV.")
    return df

def calculate_supertrend(df: pd.DataFrame, period: int = SUPERTREND_PERIOD, multiplier: float = SUPERTREND_MULTIPLIER) -> pd.DataFrame:
    """Calculates Supertrend indicator."""
    df_st = df.copy()
    if not all(col in df_st.columns for col in ['high', 'low', 'close']) or df_st[['high', 'low', 'close']].isnull().all().any():
        logger.debug(f"⚠️ [Indicators] Insufficient data to calculate Supertrend period={period}, multiplier={multiplier}.")
        df_st['supertrend'] = np.nan; df_st['supertrend_trend'] = 0; return df_st
    df_st = calculate_atr_indicator(df_st, period=period) # Use Supertrend's own period for ATR
    if 'atr' not in df_st.columns or df_st['atr'].isnull().all() or len(df_st) < period:
        logger.debug(f"⚠️ [Indicators] Insufficient ATR data to calculate Supertrend period={period}.")
        df_st['supertrend'] = np.nan; df_st['supertrend_trend'] = 0; return df_st
    hl2 = (df_st['high'] + df_st['low']) / 2
    df_st['basic_ub'] = hl2 + multiplier * df_st['atr']
    df_st['basic_lb'] = hl2 - multiplier * df_st['atr']
    df_st['final_ub'] = 0.0; df_st['final_lb'] = 0.0
    df_st['supertrend'] = np.nan; df_st['supertrend_trend'] = 0
    close = df_st['close'].values; basic_ub = df_st['basic_ub'].values; basic_lb = df_st['basic_lb'].values
    final_ub = df_st['final_ub'].values; final_lb = df_st['final_lb'].values
    st = df_st['supertrend'].values; st_trend = df_st['supertrend_trend'].values
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
        else: # Initial state
            if close[i] > final_ub[i]: st[i] = final_lb[i]; st_trend[i] = 1
            elif close[i] < final_lb[i]: st[i] = final_ub[i]; st_trend[i] = -1
            else: st[i] = np.nan; st_trend[i] = 0 # Or use previous if available
    df_st['final_ub'] = final_ub; df_st['final_lb'] = final_lb
    df_st['supertrend'] = st; df_st['supertrend_trend'] = st_trend
    df_st.drop(columns=['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, errors='ignore')
    logger.debug(f"✅ [Indicators] Calculated Supertrend period={period}, multiplier={multiplier}.")
    return df_st

# ---------------------- Candlestick Patterns ----------------------
def is_bullish_candle(row: pd.Series) -> bool:
    """Checks if the candle is bullish."""
    o, c = row.get('open'), row.get('close')
    return pd.notna(o) and pd.notna(c) and c > o

def is_hammer(row: pd.Series) -> int:
    """Checks if the candle is a Hammer pattern."""
    o, h, l, c = row.get('open'), row.get('high'), row.get('low'), row.get('close')
    if pd.isna([o, h, l, c]).any():
        return 0
    body = abs(c - o)
    candle_range = h - l
    if candle_range == 0:
        return 0
    lower_shadow = min(o, c) - l
    upper_shadow = h - max(o, c)
    is_small_body = body < (candle_range * 0.35)
    is_long_lower_shadow = lower_shadow >= 1.8 * body if body > 0 else lower_shadow > candle_range * 0.6
    is_small_upper_shadow = upper_shadow <= body * 0.6 if body > 0 else upper_shadow < candle_range * 0.15
    return 100 if is_small_body and is_long_lower_shadow and is_small_upper_shadow else 0

def is_shooting_star(row: pd.Series) -> int:
    """Checks if the candle is a Shooting Star pattern."""
    o, h, l, c = row.get('open'), row.get('high'), row.get('low'), row.get('close')
    if pd.isna([o, h, l, c]).any():
        return 0
    body = abs(c - o)
    candle_range = h - l
    if candle_range == 0:
        return 0
    lower_shadow = min(o, c) - l
    upper_shadow = h - max(o, c)
    is_small_body = body < (candle_range * 0.35)
    is_long_upper_shadow = upper_shadow >= 1.8 * body if body > 0 else upper_shadow < candle_range * 0.6
    is_small_lower_shadow = lower_shadow <= body * 0.6 if body > 0 else lower_shadow < candle_range * 0.15
    return -100 if is_small_body and is_long_upper_shadow and is_small_lower_shadow else 0

def compute_engulfing(df: pd.DataFrame, idx: int) -> int:
    """Checks if there is an Engulfing pattern."""
    if idx == 0: return 0
    prev = df.iloc[idx - 1]; curr = df.iloc[idx]
    if pd.isna([prev['close'], prev['open'], curr['close'], curr['open']]).any() or abs(prev['close'] - prev['open']) < (prev['high'] - prev['low']) * 0.1: return 0 # Prev is doji-like
    is_bullish = (prev['close'] < prev['open'] and curr['close'] > curr['open'] and curr['open'] <= prev['close'] and curr['close'] >= prev['open'])
    is_bearish = (prev['close'] > prev['open'] and curr['close'] < curr['open'] and curr['open'] >= prev['close'] and curr['close'] <= prev['open'])
    if is_bullish: return 100
    if is_bearish: return -100
    return 0

def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detects candlestick patterns."""
    df = df.copy()
    logger.debug("ℹ️ [Indicators] Detecting candlestick patterns...")
    df['Hammer'] = df.apply(is_hammer, axis=1)
    df['ShootingStar'] = df.apply(is_shooting_star, axis=1)
    df['Engulfing'] = [compute_engulfing(df, i) for i in range(len(df))]
    df['BullishCandleSignal'] = df.apply(lambda row: 1 if (row['Hammer'] == 100 or row['Engulfing'] == 100) else 0, axis=1)
    df['BearishCandleSignal'] = df.apply(lambda row: 1 if (row['ShootingStar'] == -100 or row['Engulfing'] == -100) else 0, axis=1)
    logger.debug("✅ [Indicators] Candlestick patterns detected.")
    return df

# ---------------------- Other Helper Functions ----------------------
def fetch_recent_volume(symbol: str, interval: str = '15m') -> float:
    """Fetches the trading volume for the last specified time period."""
    if not client:
        logger.error(f"❌ [Data Volume] Binance client not initialized for {symbol}."); return 0.0
    try:
        logger.debug(f"ℹ️ [Data Volume] Fetching last {interval} volume for {symbol}...")
        # Fetch klines for the specified interval
        klines = client.get_klines(symbol=symbol, interval=interval, limit=1)
        if not klines or len(klines) < 1:
            logger.warning(f"⚠️ [Data Volume] Insufficient {interval} data for {symbol}."); return 0.0
        # Use quote asset volume (index 7) for USDT volume
        volume_usdt = float(klines[0][7]) if len(klines[0]) > 7 and klines[0][7] else 0.0
        logger.debug(f"✅ [Data Volume] Liquidity for last {interval} for {symbol}: {volume_usdt:.2f} USDT")
        return volume_usdt
    except Exception as e:
        logger.error(f"❌ [Data Volume] Error while fetching volume for {symbol}: {e}", exc_info=True)
        return 0.0

# ---------------------- Performance Report Function ----------------------
def generate_performance_report() -> str:
    """Generates a comprehensive performance report."""
    logger.info("ℹ️ [Report] Generating performance report...")
    if not check_db_connection() or not conn or not cur:
        logger.error("❌ [Report] Database connection issue for report generation.")
        return "❌ Cannot generate report, database connection issue."
    try:
        with conn.cursor() as report_cur: # Uses RealDictCursor
            report_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;")
            open_signals_count = (report_cur.fetchone() or {}).get('count', 0)
            report_cur.execute("""
                SELECT
                    COUNT(*) AS total_closed,
                    COUNT(CASE WHEN profit_percentage > 0 THEN 1 END) AS winning_signals,
                    COUNT(CASE WHEN profit_percentage <= 0 THEN 1 END) AS losing_signals, -- Includes break-even as losing for simplicity here
                    COALESCE(SUM(profit_percentage), 0) AS total_profit_pct_sum,
                    COALESCE(AVG(profit_percentage), 0) AS avg_profit_pct,
                    COALESCE(SUM(CASE WHEN profit_percentage > 0 THEN profit_percentage ELSE 0 END), 0) AS gross_profit_pct_sum,
                    COALESCE(SUM(CASE WHEN profit_percentage < 0 THEN profit_percentage ELSE 0 END), 0) AS gross_loss_pct_sum,
                    COALESCE(AVG(CASE WHEN profit_percentage > 0 THEN profit_percentage END), 0) AS avg_win_pct,
                    COALESCE(AVG(CASE WHEN profit_percentage < 0 THEN profit_percentage END), 0) AS avg_loss_pct,
                    COALESCE(AVG(time_to_target_seconds), 0) AS avg_time_to_target_seconds
                FROM signals
                WHERE achieved_target = TRUE; -- Only count achieved targets for win/loss stats now
            """)
            closed_stats = report_cur.fetchone() or {}
            total_closed = closed_stats.get('total_closed', 0) # This now means total targets hit
            winning_signals = closed_stats.get('winning_signals', 0) # Should be same as total_closed if only target hits are counted
            losing_signals = 0 # Explicitly set to 0 as stop loss is removed

            total_profit_pct_sum = closed_stats.get('total_profit_pct_sum', 0.0)
            gross_profit_pct_sum = closed_stats.get('gross_profit_pct_sum', 0.0)
            gross_loss_pct_sum = 0.0
            avg_win_pct = closed_stats.get('avg_win_pct', 0.0)
            avg_loss_pct = 0.0
            avg_time_to_target_seconds = closed_stats.get('avg_time_to_target_seconds', 0.0)

            total_profit_usd = (total_profit_pct_sum / 100.0) * TRADE_VALUE
            gross_profit_usd = (gross_profit_pct_sum / 100.0) * TRADE_VALUE
            gross_loss_usd = 0.0

            win_rate = (winning_signals / total_closed) * 100 if total_closed > 0 else 0.0
            profit_factor = float('inf') if gross_loss_pct_sum == 0 else abs(gross_profit_pct_sum / gross_loss_pct_sum)

            avg_time_to_target_formatted = "N/A"
            if avg_time_to_target_seconds > 0:
                avg_time_delta = timedelta(seconds=avg_time_to_target_seconds)
                hours, remainder = divmod(avg_time_delta.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                avg_time_to_target_formatted = f"{int(hours)}h, {int(minutes)}m, {int(seconds)}s"


        report = (
            f"📊 *Comprehensive Performance Report (No Stop Loss):*\n"
            f"_(Assuming Trade Value: ${TRADE_VALUE:,.2f})_\n"
            f"——————————————\n"
            f"📈 Currently Open Signals: *{open_signals_count}*\n"
            f"——————————————\n"
            f"🎯 *Target Hit Statistics:*\n"
            f"  • Total Targets Hit: *{total_closed}*\n"
            f"  ⏳ Avg Time to Target: *{avg_time_to_target_formatted}*\n"
            f"——————————————\n"
            f"💰 *Overall Profitability (from Target Hits):*\n"
            f"  • Net Profit: *{total_profit_pct_sum:+.2f}%* (≈ *${total_profit_usd:+.2f}*)\n"
            f"  • Avg Winning Trade: *{avg_win_pct:+.2f}%*\n"
            f"  • Profit Factor: *{'∞' if profit_factor == float('inf') else f'{profit_factor:.2f}'}*\n"
            f"——————————————\n"
            f"ℹ️ *Note: Stop loss has been removed. Statistics only reflect trades that reached the target.*\n"
            f"🕰️ _Report updated as of: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        )
        logger.info("✅ [Report] Performance report generated.")
        return report
    except psycopg2.Error as db_err:
        logger.error(f"❌ [Report] Database error while generating report: {db_err}")
        if conn: conn.rollback()
        return "❌ Database error while generating report."
    except Exception as e:
        logger.error(f"❌ [Report] Unexpected error while generating report: {e}", exc_info=True)
        return "❌ Unexpected error while generating report."

# ---------------------- Trading Strategy -------------------
class ScalpingTradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        # Required columns for signal timeframe (15m) indicators - Removed VWAP, BB, OBV
        self.required_cols_signal_indicators = [
            'open', 'high', 'low', 'close', 'volume',
            f'ema_{EMA_SHORT_PERIOD}', f'ema_{EMA_LONG_PERIOD}', 'vwma',
            'rsi', 'atr', # Removed bb_upper, bb_lower, bb_middle
            'macd', 'macd_signal', 'macd_hist',
            'adx', 'di_plus', 'di_minus',
            # Removed vwap, obv
            'supertrend', 'supertrend_trend',
            'BullishCandleSignal', 'BearishCandleSignal' # Keep candle patterns for entry quality check
        ]
        # Required columns for confirmation timeframe (30m) indicators
        self.required_cols_confirmation_indicators = [
             'open', 'high', 'low', 'close',
             f'ema_{EMA_SHORT_PERIOD}', f'ema_{EMA_LONG_PERIOD}',
             'macd', 'macd_signal', 'macd_hist',
             'adx', 'di_plus', 'di_minus',
             'supertrend', 'supertrend_trend'
        ]
        # Required columns for buy signal generation - Removed VWAP, BB, OBV
        self.required_cols_buy_signal = [
            'close', f'ema_{EMA_SHORT_PERIOD}', f'ema_{EMA_LONG_PERIOD}', 'vwma',
            'rsi', 'atr', 'macd', 'macd_signal', 'macd_hist',
            'supertrend', 'supertrend_trend', 'adx', 'di_plus', 'di_minus',
            # Removed vwap, bb_upper, obv
            'BullishCandleSignal'
        ]
        # Adjusted weights for remaining optional conditions to increase sensitivity
        self.condition_weights = {
            'rsi_ok': 1.0, # Increased weight
            'bullish_candle': 2.0, # Increased weight
            # Removed 'not_bb_extreme': 0.5,
            # Removed 'obv_rising': 1.0,
            'rsi_filter_breakout': 1.5, # Increased weight
            'macd_filter_breakout': 1.5, # Increased weight
            'macd_hist_increasing': 4.0, # Increased weight
            # Removed 'obv_increasing_recent': 3.0,
            # Removed 'above_vwap': 1.0
        }
        # Essential conditions remain the same
        self.essential_conditions = [
            'price_above_emas_and_vwma', 'ema_short_above_ema_long',
            'supertrend_up', 'macd_positive_or_cross', 'adx_trending_bullish_strong',
        ]
        self.total_possible_score = sum(self.condition_weights.values())
        # --- MODIFIED: Lowered threshold for less strict optional condition scoring ---
        self.min_score_threshold_pct = 0.55 # Slightly lowered threshold for increased sensitivity (Lowered for less strictness)
        self.min_signal_score = self.total_possible_score * self.min_score_threshold_pct

    def populate_indicators(self, df: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
        """Populates indicators for a given dataframe and timeframe."""
        logger.debug(f"ℹ️ [Strategy {self.symbol}] Calculating {timeframe} frame indicators...")
        # Adjust min_len_required based on timeframe and indicators used
        min_len_required = max(EMA_LONG_PERIOD, VWMA_PERIOD, RSI_PERIOD, ENTRY_ATR_PERIOD, BOLLINGER_WINDOW, MACD_SLOW, ADX_PERIOD*2, SUPERTREND_PERIOD) + 5
        if timeframe == CONFIRMATION_TIMEFRAME:
             min_len_required = max(EMA_LONG_PERIOD, MACD_SLOW, ADX_PERIOD*2, SUPERTREND_PERIOD) + 5

        if len(df) < min_len_required:
            logger.warning(f"⚠️ [Strategy {self.symbol}] {timeframe} frame DataFrame is too short ({len(df)} < {min_len_required}).")
            return None

        try:
            df_calc = df.copy()
            # Calculate indicators relevant to both timeframes or specific to one
            df_calc[f'ema_{EMA_SHORT_PERIOD}'] = calculate_ema(df_calc['close'], EMA_SHORT_PERIOD)
            df_calc[f'ema_{EMA_LONG_PERIOD}'] = calculate_ema(df_calc['close'], EMA_LONG_PERIOD)
            df_calc = calculate_macd(df_calc, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            adx_df = calculate_adx(df_calc, ADX_PERIOD); df_calc = df_calc.join(adx_df)
            df_calc = calculate_supertrend(df_calc, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)

            if timeframe == SIGNAL_GENERATION_TIMEFRAME:
                # Add indicators specific to the signal timeframe (15m) - Removed VWAP, BB, OBV
                df_calc = calculate_atr_indicator(df_calc, ENTRY_ATR_PERIOD)
                df_calc['vwma'] = calculate_vwma(df_calc, VWMA_PERIOD)
                df_calc = calculate_rsi_indicator(df_calc, RSI_PERIOD)
                # df_calc = calculate_bollinger_bands(df_calc, BOLLINGER_WINDOW, BOLLINGER_STD_DEV) # Removed
                # df_calc = calculate_vwap(df_calc) # Removed
                # df_calc = calculate_obv(df_calc) # Removed
                df_calc = detect_candlestick_patterns(df_calc)
                required_cols = self.required_cols_signal_indicators
            elif timeframe == CONFIRMATION_TIMEFRAME:
                 required_cols = self.required_cols_confirmation_indicators
            else:
                 logger.error(f"❌ [Strategy {self.symbol}] Unknown timeframe '{timeframe}' for indicator calculation.")
                 return None

            missing_cols = [col for col in required_cols if col not in df_calc.columns]
            if missing_cols:
                logger.error(f"❌ [Strategy {self.symbol}] Missing {timeframe} frame indicator columns: {missing_cols}")
                return None

            # Drop rows with NaNs only for the required columns for this timeframe
            df_cleaned = df_calc.dropna(subset=required_cols).copy()
            if df_cleaned.empty:
                logger.warning(f"⚠️ [Strategy {self.symbol}] {timeframe} frame DataFrame is empty after dropping NaN values.")
                return None

            logger.debug(f"✅ [Strategy {self.symbol}] {timeframe} frame indicators calculated.")
            return df_cleaned
        except Exception as e:
            logger.error(f"❌ [Strategy {self.symbol}] Error while calculating {timeframe} frame indicators: {e}", exc_info=True)
            return None

    def check_confirmation_conditions(self) -> Tuple[bool, Dict[str, Any]]:
        """Checks for bullish trend confirmation on the larger timeframe (30m)."""
        logger.debug(f"ℹ️ [Strategy {self.symbol}] Checking confirmation conditions on {CONFIRMATION_TIMEFRAME} frame...")
        df_conf = fetch_historical_data(self.symbol, interval=CONFIRMATION_TIMEFRAME, days=CONFIRMATION_LOOKBACK_DAYS)
        confirmation_details = {}

        if df_conf is None or df_conf.empty:
            logger.warning(f"⚠️ [Strategy {self.symbol}] Could not fetch or process {CONFIRMATION_TIMEFRAME} frame data for confirmation.")
            confirmation_details['Status'] = f"Failed: No {CONFIRMATION_TIMEFRAME} frame data"
            return False, confirmation_details

        df_conf_processed = self.populate_indicators(df_conf, CONFIRMATION_TIMEFRAME)

        if df_conf_processed is None or df_conf_processed.empty:
            logger.warning(f"⚠️ [Strategy {self.symbol}] Could not calculate indicators for {CONFIRMATION_TIMEFRAME} frame confirmation.")
            confirmation_details['Status'] = f"Failed: Error in {CONFIRMATION_TIMEFRAME} frame indicators"
            return False, confirmation_details

        last_row_conf = df_conf_processed.iloc[-1]
        logger.debug(f"ℹ️ [Strategy {self.symbol}] Last candle on {CONFIRMATION_TIMEFRAME} frame: Close={last_row_conf['close']:.4f}, EMA_Short={last_row_conf[f'ema_{EMA_SHORT_PERIOD}']:.4f}, EMA_Long={last_row_conf[f'ema_{EMA_LONG_PERIOD}']:.4f}, Supertrend_Trend={last_row_conf.get('supertrend_trend')}, MACD_Hist={last_row_conf.get('macd_hist'):.4f}, ADX={last_row_conf.get('adx'):.1f}, DI+={last_row_conf.get('di_plus'):.1f}, DI-={last_row_conf.get('di_minus'):.1f}")


        # Confirmation Conditions: Price above EMAs, Supertrend up, MACD bullish, ADX trending
        price_above_emas_conf = (pd.notna(last_row_conf['close']) and
                                 pd.notna(last_row_conf[f'ema_{EMA_SHORT_PERIOD}']) and
                                 pd.notna(last_row_conf[f'ema_{EMA_LONG_PERIOD}']) and
                                 last_row_conf['close'] > last_row_conf[f'ema_{EMA_SHORT_PERIOD}'] and
                                 last_row_conf['close'] > last_row_conf[f'ema_{EMA_LONG_PERIOD}'])
        confirmation_details['Price_Above_EMAs_Conf'] = f"Passed ({last_row_conf['close']:.4f} > {last_row_conf[f'ema_{EMA_SHORT_PERIOD}']:.4f}, {last_row_conf['close']:.4f} > {last_row_conf[f'ema_{EMA_LONG_PERIOD}']:.4f})" if price_above_emas_conf else f"Failed ({last_row_conf['close']:.4f} not above EMA {EMA_SHORT_PERIOD}/{EMA_LONG_PERIOD})"

        supertrend_up_conf = (pd.notna(last_row_conf['supertrend_trend']) and last_row_conf['supertrend_trend'] == 1)
        confirmation_details['SuperTrend_Conf'] = "Passed (Uptrend)" if supertrend_up_conf else "Failed (Not Uptrend)"

        macd_bullish_conf = (pd.notna(last_row_conf['macd_hist']) and last_row_conf['macd_hist'] > 0)
        confirmation_details['MACD_Conf'] = f"Passed (Hist > 0: {last_row_conf['macd_hist']:.4f})" if macd_bullish_conf else f"Failed (Hist <= 0: {last_row_conf.get('macd_hist', np.nan):.4f})"

        adx_trending_bullish_conf = (pd.notna(last_row_conf['adx']) and last_row_conf['adx'] > MIN_ADX_TREND_STRENGTH and
                                     pd.notna(last_row_conf['di_plus']) and pd.notna(last_row_conf['di_minus']) and
                                     last_row_conf['di_plus'] > last_row_conf['di_minus'])
        confirmation_details['ADX_DI_Conf'] = f"Passed (ADX:{last_row_conf['adx']:.1f}, DI+>DI-)" if adx_trending_bullish_conf else f"Failed (ADX <= {MIN_ADX_TREND_STRENGTH} or DI+ <= DI-)"

        all_confirmed = price_above_emas_conf and supertrend_up_conf and macd_bullish_conf and adx_trending_bullish_conf

        confirmation_details['Status'] = "Confirmed" if all_confirmed else "Confirmation Failed"
        logger.debug(f"✅ [Strategy {self.symbol}] {CONFIRMATION_TIMEFRAME} frame confirmation status: {confirmation_details['Status']}")

        return all_confirmed, confirmation_details

    def check_entry_point_quality(self, df_processed_signal: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Checks if the current price offers a good entry point on the signal timeframe (15m)."""
        logger.debug(f"ℹ️ [Strategy {self.symbol}] Checking entry point quality on {SIGNAL_GENERATION_TIMEFRAME} frame...")
        entry_point_details = {}

        if df_processed_signal is None or df_processed_signal.empty or len(df_processed_signal) < ENTRY_POINT_RECENT_CANDLE_LOOKBACK + 1:
            logger.warning(f"⚠️ [Strategy {self.symbol}] Insufficient {SIGNAL_GENERATION_TIMEFRAME} frame data for entry point check.")
            entry_point_details['Status'] = f"Failed: Insufficient {SIGNAL_GENERATION_TIMEFRAME} frame data"
            return False, entry_point_details

        last_row_signal = df_processed_signal.iloc[-1]
        recent_df_signal = df_processed_signal.iloc[-ENTRY_POINT_RECENT_CANDLE_LOOKBACK-1:]

        if recent_df_signal[['close', 'open', f'ema_{EMA_SHORT_PERIOD}']].isnull().values.any():
             logger.warning(f"⚠️ [Strategy {self.symbol}] Recent {SIGNAL_GENERATION_TIMEFRAME} frame data contains NaN for entry point check.")
             entry_point_details['Status'] = f"Failed: NaN in recent {SIGNAL_GENERATION_TIMEFRAME} frame data"
             return False, entry_point_details

        current_price = last_row_signal['close']
        ema_short_signal = last_row_signal[f'ema_{EMA_SHORT_PERIOD}']

        # Condition 1: Price is close to the signal timeframe EMA_SHORT
        # --- MODIFIED: Using the potentially increased ENTRY_POINT_EMA_PROXIMITY_PCT ---
        price_near_ema_short = abs(current_price - ema_short_signal) / ema_short_signal <= ENTRY_POINT_EMA_PROXIMITY_PCT if ema_short_signal > 0 else False
        entry_point_details['Price_Near_EMA_Short_SignalTF'] = f"Passed (within {ENTRY_POINT_EMA_PROXIMITY_PCT*100:.2f}%)" if price_near_ema_short else f"Failed (Distance: {abs(current_price - ema_short_signal) / ema_short_signal * 100:.2f}%)"
        logger.debug(f"ℹ️ [Strategy {self.symbol}] EMA {EMA_SHORT_PERIOD} ({SIGNAL_GENERATION_TIMEFRAME}) proximity check: Price {current_price:.4f}, EMA {EMA_SHORT_PERIOD} {ema_short_signal:.4f}, Near: {price_near_ema_short}")


        # Condition 2: Last candle is bullish or a hammer
        last_candle_bullish_or_hammer = is_bullish_candle(last_row_signal) or is_hammer(last_row_signal) == 100
        entry_point_details['Last_Candle_Bullish_or_Hammer_SignalTF'] = "Passed" if last_candle_bullish_or_hammer else "Failed"
        logger.debug(f"ℹ️ [Strategy {self.symbol}] Last candle ({SIGNAL_GENERATION_TIMEFRAME}) check: Bullish or Hammer: {last_candle_bullish_or_hammer}")


        # Combine conditions for a good entry point
        is_good_entry = price_near_ema_short and last_candle_bullish_or_hammer

        entry_point_details['Status'] = "Good Entry Point" if is_good_entry else "Entry Point Not Ideal"
        logger.debug(f"✅ [Strategy {self.symbol}] Entry point quality status on {SIGNAL_GENERATION_TIMEFRAME} frame: {entry_point_details['Status']}")

        return is_good_entry, entry_point_details


    def generate_buy_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generates a buy signal based on strategy conditions."""
        logger.debug(f"ℹ️ [Strategy {self.symbol}] Starting buy signal generation...")
        min_signal_data_len = max(RECENT_EMA_CROSS_LOOKBACK, MACD_HIST_INCREASE_CANDLES, OBV_INCREASE_CANDLES, ENTRY_POINT_RECENT_CANDLE_LOOKBACK) + 1
        if df_processed is None or df_processed.empty or len(df_processed) < min_signal_data_len:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame too short for signal generation."); return None
        missing_cols = [col for col in self.required_cols_buy_signal if col not in df_processed.columns]
        if missing_cols: logger.warning(f"⚠️ [Strategy {self.symbol}] Missing columns for signal: {missing_cols}."); return None

        # --- Step 1: Check Multi-Timeframe Confirmation (30m) ---
        is_confirmed_on_larger_tf, confirmation_details = self.check_confirmation_conditions()
        if not is_confirmed_on_larger_tf:
             logger.debug(f"ℹ️ [Strategy {self.symbol}] Confirmation failed on {CONFIRMATION_TIMEFRAME} frame. Cancelling signal.")
             return None # Do not generate signal if larger timeframe is not confirmed
        logger.debug(f"✅ [Strategy {self.symbol}] Confirmed on {CONFIRMATION_TIMEFRAME} frame. Proceeding.")


        # --- Step 2: Check Entry Point Quality on Signal Timeframe (15m) ---
        is_good_entry_point, entry_point_details = self.check_entry_point_quality(df_processed)
        if not is_good_entry_point:
             logger.debug(f"ℹ️ [Strategy {self.symbol}] Entry point quality on {SIGNAL_GENERATION_TIMEFRAME} frame is not ideal. Cancelling signal.")
             return None # Do not generate signal if entry point is not ideal
        logger.debug(f"✅ [Strategy {self.symbol}] Entry point quality on {SIGNAL_GENERATION_TIMEFRAME} frame is good. Proceeding.")


        # --- Step 3: Proceed with Signal Generation if Confirmed and Entry is Good ---
        btc_trend = get_btc_trend_4h()
        # --- MODIFIED: Relaxed BTC trend check - only cancel on clear downtrend ---
        if "Downtrend" in btc_trend: logger.info(f"ℹ️ [Strategy {self.symbol}] Trading paused (BTC trend is downtrend: {btc_trend})."); return None
        if "N/A" in btc_trend: logger.warning(f"⚠️ [Strategy {self.symbol}] BTC trend not available, condition ignored.")

        last_row = df_processed.iloc[-1]; recent_df = df_processed.iloc[-min_signal_data_len:]
        if recent_df[self.required_cols_buy_signal].isnull().values.any():
            logger.warning(f"⚠️ [Strategy {self.symbol}] Recent data contains NaN in required columns."); return None

        essential_passed = True; failed_essential_conditions = []; signal_details = {}
        # Mandatory Conditions Check (on 15m timeframe)
        # Corrected typo: EMA_LONG_PERIOD instead of LONG_PERIOD
        if not (pd.notna(last_row[f'ema_{EMA_SHORT_PERIOD}']) and pd.notna(last_row[f'ema_{EMA_LONG_PERIOD}']) and pd.notna(last_row['vwma']) and last_row['close'] > last_row[f'ema_{EMA_SHORT_PERIOD}'] and last_row['close'] > last_row[f'ema_{EMA_LONG_PERIOD}'] and last_row['close'] > last_row['vwma']):
            essential_passed = False; failed_essential_conditions.append('Price above MAs and VWMA'); signal_details['Price_MA_Alignment_SignalTF'] = 'Failed: Price not above all MAs on signal frame'
        else: signal_details['Price_MA_Alignment_SignalTF'] = 'Passed: Price above all MAs on signal frame'

        if not (pd.notna(last_row[f'ema_{EMA_SHORT_PERIOD}']) and pd.notna(last_row[f'ema_{EMA_LONG_PERIOD}']) and last_row[f'ema_{EMA_SHORT_PERIOD}'] > last_row[f'ema_{EMA_LONG_PERIOD}']):
            essential_passed = False; failed_essential_conditions.append('Short EMA > Long EMA'); signal_details['EMA_Order_SignalTF'] = 'Failed: Short EMA not above Long EMA on signal frame'
        else: signal_details['EMA_Order_SignalTF'] = 'Passed: Short EMA above Long EMA on signal frame'

        if not (pd.notna(last_row['supertrend']) and last_row['close'] > last_row['supertrend'] and last_row['supertrend_trend'] == 1):
            essential_passed = False; failed_essential_conditions.append('SuperTrend Uptrend'); signal_details['SuperTrend_SignalTF'] = 'Failed: SuperTrend not uptrend or price not above it on signal frame'
        else: signal_details['SuperTrend_SignalTF'] = 'Passed: SuperTrend uptrend and price above it on signal frame'

        if not (pd.notna(last_row['macd_hist']) and (last_row['macd_hist'] > 0 or (pd.notna(last_row['macd']) and pd.notna(last_row['macd_signal']) and last_row['macd'] > last_row['macd_signal']))):
            essential_passed = False; failed_essential_conditions.append('MACD Bullish'); signal_details['MACD_SignalTF'] = 'Failed: MACD Hist not positive and no bullish cross on signal frame'
        else: signal_details['MACD_SignalTF'] = 'Passed: MACD Hist positive or bullish cross on signal frame'

        # --- MODIFIED: Using the potentially lowered MIN_ADX_TREND_STRENGTH ---
        if not (pd.notna(last_row['adx']) and pd.notna(last_row['di_plus']) and pd.notna(last_row['di_minus']) and last_row['adx'] > MIN_ADX_TREND_STRENGTH and last_row['di_plus'] > last_row['di_minus']):
            essential_passed = False; failed_essential_conditions.append(f'Strong Bullish ADX (>{MIN_ADX_TREND_STRENGTH})'); signal_details['ADX_DI_SignalTF'] = f'Failed: Not strongly bullish on signal frame (ADX <= {MIN_ADX_TREND_STRENGTH} or DI+ <= DI-)'
        else: signal_details['ADX_DI_SignalTF'] = f'Passed: Strongly bullish on signal frame (ADX:{last_row["adx"]:.1f}, DI+>DI-)'


        if not essential_passed:
            logger.debug(f"ℹ️ [Strategy {self.symbol}] {SIGNAL_GENERATION_TIMEFRAME} frame essential conditions failed: {', '.join(failed_essential_conditions)}.");
            # Include essential conditions check results in details even if failed
            signal_details['Essential_Conditions_SignalTF_Status'] = 'Failed'
            signal_details['Failed_Essential_Conditions_SignalTF'] = failed_essential_conditions
            return None

        signal_details['Essential_Conditions_SignalTF_Status'] = 'Passed'
        current_score = 0.0 # Optional Conditions Scoring (Updated based on remaining conditions)

        if pd.notna(last_row['rsi']) and RSI_OVERSOLD < last_row['rsi'] < RSI_OVERBOUGHT : current_score += self.condition_weights.get('rsi_ok', 0); signal_details['RSI_Basic_SignalTF'] = f'Acceptable ({RSI_OVERSOLD}<{last_row["rsi"]:.1f}<{RSI_OVERBOUGHT}) (+{self.condition_weights.get("rsi_ok",0)})'
        else: signal_details['RSI_Basic_SignalTF'] = f'RSI ({last_row.get("rsi", np.nan):.1f}) Not Acceptable (0)'

        if last_row.get('BullishCandleSignal', 0) == 1: current_score += self.condition_weights.get('bullish_candle', 0); signal_details['Candle_SignalTF'] = f'Bullish candle pattern (+{self.condition_weights.get("bullish_candle",0)})'
        else: signal_details['Candle_SignalTF'] = 'No bullish candle pattern (0)'

        # Removed: if pd.notna(last_row['bb_upper']) and last_row['close'] < last_row['bb_upper'] * 0.995 : current_score += self.condition_weights.get('not_bb_extreme', 0); signal_details['Bollinger_Basic_SignalTF'] = f'Not at upper band (+{self.condition_weights.get("not_bb_extreme",0)})'
        # Removed: else: signal_details['Bollinger_Basic_SignalTF'] = 'At or above upper band (0)'

        # Removed: if len(df_processed) >= 2 and pd.notna(df_processed.iloc[-2]['obv']) and pd.notna(last_row['obv']) and last_row['obv'] > df_processed.iloc[-2]['obv']: current_score += self.condition_weights.get('obv_rising', 0); signal_details['OBV_Last_SignalTF'] = f'Rising in last candle (+{self.condition_weights.get("obv_rising",0)})'
        # Removed: else: signal_details['OBV_Last_SignalTF'] = 'Not rising in last candle (0)'

        if pd.notna(last_row['rsi']) and 50 <= last_row['rsi'] <= 80: current_score += self.condition_weights.get('rsi_filter_breakout', 0); signal_details['RSI_Filter_Breakout_SignalTF'] = f'RSI ({last_row["rsi"]:.1f}) in bullish range (50-80) (+{self.condition_weights.get("rsi_filter_breakout",0)})'
        else: signal_details['RSI_Filter_Breakout_SignalTF'] = f'RSI ({last_row.get("rsi", np.nan):.1f}) not in bullish range (0)'

        if pd.notna(last_row['macd_hist']) and last_row['macd_hist'] > 0: current_score += self.condition_weights.get('macd_filter_breakout', 0); signal_details['MACD_Filter_Breakout_SignalTF'] = f'MACD Hist positive ({last_row["macd_hist"]:.4f}) (+{self.condition_weights.get("macd_filter_breakout",0)})'
        else: signal_details['MACD_Filter_Breakout_SignalTF'] = 'MACD Hist not positive (0)'

        if len(recent_df) >= MACD_HIST_INCREASE_CANDLES + 1 and not recent_df['macd_hist'].iloc[-MACD_HIST_INCREASE_CANDLES-1:].isnull().any() and np.all(np.diff(recent_df['macd_hist'].iloc[-MACD_HIST_INCREASE_CANDLES-1:]) > 0): current_score += self.condition_weights.get('macd_hist_increasing', 0); signal_details['MACD_Hist_Increasing_SignalTF'] = f'MACD Hist increasing ({MACD_HIST_INCREASE_CANDLES} candles) (+{self.condition_weights.get("macd_hist_increasing",0)})'
        else: signal_details['MACD_Hist_Increasing_SignalTF'] = f'MACD Hist not increasing ({MACD_HIST_INCREASE_CANDLES} candles) (0)'

        # Removed: if len(recent_df) >= OBV_INCREASE_CANDLES + 1 and not recent_df['obv'].iloc[-OBV_INCREASE_CANDLES-1:].isnull().any() and np.all(np.diff(recent_df['obv'].iloc[-OBV_INCREASE_CANDLES-1:]) > 0): current_score += self.condition_weights.get('obv_increasing_recent', 0); signal_details['OBV_Increasing_Recent_SignalTF'] = f'OBV increasing ({OBV_INCREASE_CANDLES} candles) (+{self.condition_weights.get("obv_increasing_recent",0)})'
        # Removed: else: signal_details['OBV_Increasing_Recent_SignalTF'] = f'OBV not increasing ({OBV_INCREASE_CANDLES} candles) (0)'

        # Removed: if pd.notna(last_row['vwap']) and last_row['close'] > last_row['vwap']: current_score += self.condition_weights.get('above_vwap', 0); signal_details['VWAP_Daily'] = f'Above daily VWAP (+{self.condition_weights.get("above_vwap",0)})'
        # Removed: else: signal_details['VWAP_Daily'] = 'Below daily VWAP (0)'

        # --- MODIFIED: Using the potentially lowered min_score_threshold_pct ---
        if current_score < self.min_signal_score:
            logger.debug(f"ℹ️ [Strategy {self.symbol}] Optional score too low ({current_score:.2f} < {self.min_signal_score:.2f}). Cancelling signal.");
            signal_details['Optional_Score_Status'] = f'Failed: Score {current_score:.2f} is below minimum {self.min_signal_score:.2f}'
            return None

        signal_details['Optional_Score_Status'] = f'Passed: Score {current_score:.2f}'

        # Fetch volume for the signal timeframe (15m)
        volume_recent = fetch_recent_volume(self.symbol, interval=SIGNAL_GENERATION_TIMEFRAME)
        # --- MODIFIED: Using the potentially lowered MIN_VOLUME_15M_USDT ---
        if volume_recent < MIN_VOLUME_15M_USDT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] Liquidity too low ({volume_recent:,.0f} < {MIN_VOLUME_15M_USDT:,.0f}). Cancelling signal.");
            signal_details['Liquidity_Check'] = f'Failed: Liquidity {volume_recent:,.0f} below minimum {MIN_VOLUME_15M_USDT:,.0f}'
            return None

        signal_details['Liquidity_Check'] = f'Passed: Liquidity {volume_recent:,.0f}'

        current_price = last_row['close']; current_atr = last_row.get('atr')
        if pd.isna(current_atr) or current_atr <= 0:
            logger.warning(f"⚠️ [Strategy {self.symbol}] Invalid ATR ({current_atr}). Cancelling signal.");
            signal_details['ATR_Check'] = f'Failed: Invalid ATR ({current_atr})'
            return None

        signal_details['ATR_Check'] = f'Passed: Valid ATR ({current_atr:.4f})'

        initial_target = current_price + (ENTRY_ATR_MULTIPLIER * current_atr)
        initial_stop_loss = 0.0 # Stop Loss is removed

        profit_margin_pct = ((initial_target / current_price) - 1) * 100 if current_price > 0 else 0
        # --- MODIFIED: Using the potentially lowered MIN_PROFIT_MARGIN_PCT ---
        if profit_margin_pct < MIN_PROFIT_MARGIN_PCT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] Profit margin too low ({profit_margin_pct:.2f}% < {MIN_PROFIT_MARGIN_PCT:.2f}%). Cancelling signal.");
            signal_details['Profit_Margin_Check'] = f'Failed: Profit margin {profit_margin_pct:.2f}% below minimum {MIN_PROFIT_MARGIN_PCT:.2f}%'
            return None

        signal_details['Profit_Margin_Check'] = f'Passed: Profit margin {profit_margin_pct:.2f}%'

        # Include confirmation and entry point details in the signal details
        signal_details['Confirmation_Details'] = confirmation_details
        signal_details['Entry_Point_Details_SignalTF'] = entry_point_details


        signal_output = {
            'symbol': self.symbol, 'entry_price': float(f"{current_price:.8g}"),
            'initial_target': float(f"{initial_target:.8g}"),
            'initial_stop_loss': initial_stop_loss, # Stop loss removed
            'current_target': float(f"{initial_target:.8g}"),
            'current_stop_loss': initial_stop_loss, # Stop loss removed
            'r2_score': float(f"{current_score:.2f}"),
            'strategy_name': 'Scalping_Momentum_Trend_MultiTF_EnhancedEntry_V2_LessStrict', # Updated strategy name to reflect changes
            'signal_details': signal_details, 'volume_15m': volume_recent, # Using the renamed key
            'trade_value': TRADE_VALUE, 'total_possible_score': float(f"{self.total_possible_score:.2f}")
        }
        logger.info(f"✅ [Strategy {self.symbol}] Buy signal confirmed. Price: {current_price:.6f}, Score: {current_score:.2f}, ATR: {current_atr:.6f}")
        return signal_output

    def analyze_target_continuation(self, df_processed: pd.DataFrame, current_price: float, current_target: float) -> Optional[float]:
        """Analyzes if the target should be extended."""
        logger.debug(f"ℹ️ [Strategy {self.symbol}] Analyzing target continuation...")
        if df_processed is None or df_processed.empty:
            logger.debug(f"⚠️ [Strategy {self.symbol}] Insufficient data for target continuation analysis.")
            return None
        last_row = df_processed.iloc[-1]

        # Conditions for continuation (example: strong momentum) - Using signal timeframe indicators
        macd_hist_ok = pd.notna(last_row['macd_hist']) and last_row['macd_hist'] > 0.1 # Example: histogram still positive and strong
        adx_ok = pd.notna(last_row['adx']) and last_row['adx'] > MIN_ADX_FOR_DYNAMIC_UPDATE and pd.notna(last_row['di_plus']) and pd.notna(last_row['di_minus']) and last_row['di_plus'] > last_row['di_minus']
        rsi_ok = pd.notna(last_row['rsi']) and last_row['rsi'] < (RSI_OVERBOUGHT + 10) # Allow slightly higher RSI for continuation

        logger.debug(f"ℹ️ [Strategy {self.symbol}] Dynamic target check: MACD Hist={last_row.get('macd_hist', np.nan):.4f} (OK:{macd_hist_ok}), ADX={last_row.get('adx', np.nan):.1f} (OK:{adx_ok}), RSI={last_row.get('rsi', np.nan):.1f} (OK:{rsi_ok})")

        if macd_hist_ok and adx_ok and rsi_ok:
            current_atr = last_row.get('atr')
            if pd.notna(current_atr) and current_atr > 0:
                new_target = current_target + (current_atr * DYNAMIC_TARGET_EXTENSION_ATR_MULTIPLIER)
                logger.info(f"🎯 [Strategy {self.symbol}] Dynamic target extension approved. Old: {current_target:.6f}, New: {new_target:.6f}")
                return new_target
            else:
                logger.warning(f"⚠️ [Strategy {self.symbol}] Invalid ATR for dynamic target extension.")
        else:
            logger.debug(f"ℹ️ [Strategy {self.symbol}] Conditions not met for dynamic target extension.")
        return None

# ---------------------- Telegram Functions ----------------------
def send_telegram_message(target_chat_id: str, text: str, reply_markup: Optional[Dict] = None, parse_mode: str = 'Markdown', disable_web_page_preview: bool = True, timeout: int = 20) -> Optional[Dict]:
    """Sends a message to Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(target_chat_id), 'text': text, 'parse_mode': parse_mode, 'disable_web_page_preview': disable_web_page_preview}
    if reply_markup: payload['reply_markup'] = json.dumps(convert_np_values(reply_markup))
    logger.debug(f"ℹ️ [Telegram] Sending message to {target_chat_id}...")
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        logger.info(f"✅ [Telegram] Message sent to {target_chat_id}.")
        return response.json()
    except Exception as e: logger.error(f"❌ [Telegram] Failed to send message to {target_chat_id}: {e}", exc_info=True); return None

def send_telegram_alert(signal_data: Dict[str, Any], timeframe: str) -> None:
    """Sends a trading signal alert to Telegram."""
    logger.debug(f"ℹ️ [Telegram Alert] Formatting alert for {signal_data.get('symbol', 'N/A')}")
    try:
        entry_price = float(signal_data['entry_price']); target_price = float(signal_data['initial_target'])
        symbol = signal_data['symbol']; strategy_name = signal_data.get('strategy_name', 'N/A')
        signal_score = signal_data.get('r2_score', 0.0); total_possible_score = signal_data.get('total_possible_score', 10.0)
        volume_signal_tf = signal_data.get('volume_15m', 0.0); # Using the renamed key
        trade_value_signal = signal_data.get('trade_value', TRADE_VALUE)

        # Extract and format confirmation and entry point details
        confirmation_details = signal_data.get('signal_details', {}).get('Confirmation_Details', {})
        entry_point_details = signal_data.get('signal_details', {}).get('Entry_Point_Details_SignalTF', {}) # Using the renamed key

        confirmation_text = "\n".join([f"    - {k.replace('_', ' ').title()}: {v}" for k,v in confirmation_details.items()])
        entry_point_text = "\n".join([f"    - {k.replace('_', ' ').title()}: {v}" for k,v in entry_point_details.items()])

        # Construct a readable string from other signal_details for conditions met
        other_signal_details_text = "\n".join([f"  - {k.replace('_', ' ').title()}: {v}" for k,v in signal_data.get('signal_details', {}).items() if k not in ['Confirmation_Details', 'Entry_Point_Details_SignalTF'] and ('Passed' in str(v) or 'Failed' not in str(v) or '+' in str(v) or '(0)' not in str(v))]) # Updated key


        profit_pct = ((target_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        profit_usdt = trade_value_signal * (profit_pct / 100)

        generation_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


        safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')
        fear_greed = get_fear_greed_index(); btc_trend = get_btc_trend_4h()

        message = (
            f"💡 *New Trading Signal ({strategy_name.replace('_', ' ').title()})* 💡\n"
            f"🕰️ *Recommendation Generation Time:* {generation_time_str}\n"
            f"——————————————\n"
            f"🪙 **Pair:** `{safe_symbol}`\n"
            f"📈 **Signal Type:** Buy (Long)\n"
            f"🕰️ **Signal Timeframe:** {timeframe}\n"
            f"📊 **Signal Strength (Score):** *{signal_score:.1f} / {total_possible_score:.1f}*\n"
            f"💧 **Liquidity ({timeframe}):** {volume_signal_tf:,.0f} USDT\n" # Updated volume interval
            f"——————————————\n"
            f"➡️ **Suggested Entry Price:** `${entry_price:,.8g}`\n"
            f"🎯 **Initial Target:** `${target_price:,.8g}` ({profit_pct:+.2f}% / ≈ ${profit_usdt:+.2f})\n"
            f"——————————————\n"
            f"✅ *Larger Timeframe Confirmation ({CONFIRMATION_TIMEFRAME}):*\n" # Updated confirmation timeframe
            f"{confirmation_text if confirmation_text else '    - No confirmation details available.'}\n"
            f"——————————————\n"
            f"📍 *Entry Point Quality ({SIGNAL_GENERATION_TIMEFRAME}):*\n" # Updated signal timeframe
            f"{entry_point_text if entry_point_text else '    - No entry point details available.'}\n"
            f"——————————————\n"
            f"📋 *Other Signal Condition Details:*\n"
            f"{other_signal_details_text if other_signal_details_text else '  - No additional condition details available.'}\n"
            f"——————————————\n"
            f"😨/🤑 **Fear and Greed Index:** {fear_greed}\n"
            f"₿ **Bitcoin Trend (4h):** {btc_trend}\n"
            f"——————————————\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        reply_markup = {"inline_keyboard": [[{"text": "📊 View Performance Report", "callback_data": "get_report"}]]}
        send_telegram_message(CHAT_ID, message, reply_markup=reply_markup, parse_mode='Markdown')
        logger.info(f"✅ [Telegram Alert] Signal alert sent for {symbol}.")
    except Exception as e: logger.error(f"❌ [Telegram Alert] Failed to send alert for {signal_data.get('symbol', 'N/A')}: {e}", exc_info=True)

def format_duration(seconds: Optional[int]) -> str:
    """Formats duration from seconds into a human-readable string."""
    if seconds is None or seconds < 0:
        return "N/A"
    delta = timedelta(seconds=seconds)
    hours, remainder = divmod(delta.total_seconds(), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours)}h, {int(minutes)}m, {int(secs)}s"

def send_tracking_notification(details: Dict[str, Any]) -> None:
    """Sends a signal tracking notification (e.g., target hit)."""
    symbol = details.get('symbol', 'N/A'); signal_id = details.get('id', 'N/A')
    notification_type = details.get('type', 'unknown'); message = ""
    safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')
    closing_price = details.get('closing_price', 0.0); profit_pct = details.get('profit_pct', 0.0)
    time_to_target_str = format_duration(details.get('time_to_target_seconds'))

    logger.debug(f"ℹ️ [Notification] Formatting tracking notification: ID={signal_id}, Type={notification_type}, Symbol={symbol}")

    if notification_type == 'target_hit':
        message = (
            f"✅ *Target Reached (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **Pair:** `{safe_symbol}`\n"
            f"🎯 **Closing Price (Target):** `${closing_price:,.8g}`\n"
            f"💰 **Profit Achieved:** {profit_pct:+.2f}%\n"
            f"⏱️ **Time Taken to Reach Target:** {time_to_target_str}"
        )
        logger.info(f"✅ [Notification] Target reached notification sent for {symbol} (ID: {signal_id}).")
    elif notification_type == 'target_updated_dynamically':
        old_target = details.get('old_target', 0.0)
        new_target = details.get('new_target', 0.0)
        current_price = details.get('current_price', 0.0)
        message = (
            f"🔄 *Target Dynamically Updated (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **Pair:** `{safe_symbol}`\n"
            f"📈 **Current Price:** `${current_price:,.8g}`\n"
            f"📊 **Analysis:** Continuation signals detected\n"
            f"🏹 **Old Target:** `${old_target:,.8g}`\n"
            f"🎯 **New Target:** `${new_target:,.8g}`"
        )
        logger.info(f"🔄 [Notification] Dynamic target update notification sent for {symbol} (ID: {signal_id}).")
    else: logger.warning(f"⚠️ [Notification] Unknown notification type: {notification_type} for {details}"); return

    if message: send_telegram_message(CHAT_ID, message, parse_mode='Markdown')

# ---------------------- Database Functions (Insert and Update) ----------------------
def insert_signal_into_db(signal: Dict[str, Any]) -> bool:
    """Inserts a new signal into the database."""
    if not check_db_connection() or not conn:
        logger.error(f"❌ [DB Insert] Database connection issue for inserting signal {signal.get('symbol', 'N/A')}.")
        return False
    symbol = signal.get('symbol', 'N/A')
    logger.debug(f"ℹ️ [DB Insert] Inserting signal for {symbol}...")
    try:
        signal_prepared = convert_np_values(signal)
        signal_details_json = json.dumps(signal_prepared.get('signal_details', {}))
        with conn.cursor() as cur_ins:
            insert_query = sql.SQL("""
                INSERT INTO signals
                 (symbol, entry_price, initial_target, initial_stop_loss, current_target, current_stop_loss,
                 r2_score, strategy_name, signal_details, volume_15m, sent_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW());
            """) # sent_at is recommendation generation time
            cur_ins.execute(insert_query, (
                signal_prepared['symbol'], signal_prepared['entry_price'],
                signal_prepared['initial_target'], signal_prepared.get('initial_stop_loss', 0.0), # Default SL to 0.0
                signal_prepared['current_target'], signal_prepared.get('current_stop_loss', 0.0), # Default SL to 0.0
                signal_prepared.get('r2_score'), signal_prepared.get('strategy_name', 'unknown'),
                signal_details_json, signal_prepared.get('volume_15m') # Using the renamed key
            ))
        conn.commit()
        logger.info(f"✅ [DB Insert] Signal inserted for {symbol} (Score: {signal_prepared.get('r2_score')}).")
        return True
    except psycopg2.Error as db_err: # Specific Psycopg2 error
        logger.error(f"❌ [DB Insert] Database error while inserting signal for {symbol}: {db_err}")
        if conn:
            conn.rollback()
        return False
    except Exception as e: # General exception
        logger.error(f"❌ [DB Insert] Error while inserting signal for {symbol}: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return False

# ---------------------- Open Signal Tracking Function ----------------------
def track_signals() -> None:
    """Tracks open signals and updates their status (target hit, target update)."""
    logger.info("ℹ️ [Tracker] Starting open signal tracking process...")
    while True:
        active_signals_summary: List[str] = []
        processed_in_cycle = 0
        try:
            if not check_db_connection() or not conn:
                logger.warning("⚠️ [Tracker] Skipping tracking: Database connection issue."); time.sleep(15); continue

            with conn.cursor() as track_cur: # Uses RealDictCursor
                 track_cur.execute("""
                    SELECT id, symbol, entry_price, current_target, sent_at, dynamic_updates_count
                    FROM signals
                    WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;
                """) # Removed stop loss fields from select
                 open_signals: List[Dict] = track_cur.fetchall()

            if not open_signals:
                logger.debug("ℹ️ [Tracker] No open signals to track.")
                time.sleep(10); continue # Wait less if no signals

            logger.debug(f"ℹ️ [Tracker] Tracking {len(open_signals)} open signals...")

            for signal_row in open_signals:
                signal_id = signal_row['id']; symbol = signal_row['symbol']; processed_in_cycle += 1
                update_executed = False # To track if this signal was updated in the current cycle
                try:
                    entry_price = float(signal_row['entry_price'])
                    current_target = float(signal_row['current_target'])
                    sent_at_timestamp = signal_row['sent_at'] # Recommendation generation time
                    dynamic_updates_count = signal_row.get('dynamic_updates_count', 0)

                    current_price = ticker_data.get(symbol)
                    if current_price is None:
                        logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): Price not available in ticker.");
                        continue # Skip if price not available

                    active_signals_summary.append(f"{symbol}({signal_id}): P={current_price:.4f} T={current_target:.4f} DynUpd={dynamic_updates_count}")
                    logger.debug(f"ℹ️ [Tracker] Processing signal ID:{signal_id}, Symbol:{symbol}, Current Price:{current_price:.4f}, Current Target:{current_target:.4f}")


                    update_query: Optional[sql.SQL] = None; update_params: Tuple = (); log_message: Optional[str] = None
                    notification_details: Dict[str, Any] = {'symbol': symbol, 'id': signal_id, 'current_price': current_price}

                    # 1. Check for Target Hit
                    if current_price >= current_target:
                        profit_pct = ((current_target / entry_price) - 1) * 100 if entry_price > 0 else 0
                        closed_at_time = datetime.now()
                        time_to_target_delta = closed_at_time - sent_at_timestamp
                        time_to_target_seconds = int(time_to_target_delta.total_seconds())

                        update_query = sql.SQL("""
                            UPDATE signals SET achieved_target = TRUE, closing_price = %s,
                                         closed_at = %s, profit_percentage = %s,
                                         time_to_target_seconds = %s
                            WHERE id = %s;
                        """)
                        update_params = (current_target, closed_at_time, profit_pct, time_to_target_seconds, signal_id)
                        log_message = f"🎯 [Tracker] {symbol}(ID:{signal_id}): Target hit at {current_target:.8g} (Profit: {profit_pct:+.2f}%). Time: {format_duration(time_to_target_seconds)}."
                        notification_details.update({'type': 'target_hit', 'closing_price': current_target, 'profit_pct': profit_pct, 'time_to_target_seconds': time_to_target_seconds})
                        update_executed = True
                        logger.info(f"🎯 [Tracker] Target hit for {symbol} (ID: {signal_id}).")

                    # 3. Dynamic Target Update (Only if Target not hit and updates allowed)
                    elif not update_executed and dynamic_updates_count < MAX_DYNAMIC_TARGET_UPDATES and \
                         current_price >= (current_target * (1 - DYNAMIC_TARGET_APPROACH_PCT)) and \
                         current_price < current_target: # Price is near target but hasn't hit it

                        logger.info(f"🔍 [Tracker] {symbol}(ID:{signal_id}): Price is near target. Re-evaluating for dynamic update (Update #{dynamic_updates_count + 1}).")
                        # Fetch data for the signal tracking timeframe (15m) for dynamic update analysis
                        df_dynamic = fetch_historical_data(symbol, interval=SIGNAL_TRACKING_TIMEFRAME, days=SIGNAL_TRACKING_LOOKBACK_DAYS)
                        if df_dynamic is not None and not df_dynamic.empty:
                            strategy = ScalpingTradingStrategy(symbol) # Re-initialize strategy to use its methods
                            # Populate indicators for the signal tracking timeframe (15m)
                            df_indicators_dynamic = strategy.populate_indicators(df_dynamic, SIGNAL_TRACKING_TIMEFRAME)
                            if df_indicators_dynamic is not None and not df_indicators_dynamic.empty:
                                new_dynamic_target = strategy.analyze_target_continuation(df_indicators_dynamic, current_price, current_target)
                                if new_dynamic_target and new_dynamic_target > current_target:
                                    update_query = sql.SQL("""
                                        UPDATE signals SET current_target = %s,
                                                       dynamic_updates_count = dynamic_updates_count + 1,
                                                       signal_details = signal_details || %s::jsonb
                                        WHERE id = %s;
                                    """)
                                    update_details_json = json.dumps({
                                        f"dynamic_update_{dynamic_updates_count+1}": {
                                            "timestamp": str(datetime.now()),
                                            "old_target": current_target,
                                            "new_target": new_dynamic_target,
                                            "price_at_update": current_price
                                        }
                                    })
                                    update_params = (new_dynamic_target, update_details_json, signal_id)
                                    log_message = f"🔄 [Tracker] {symbol}(ID:{signal_id}): Dynamic target update! Old: {current_target:.6f}, New: {new_dynamic_target:.6f}"
                                    notification_details.update({'type': 'target_updated_dynamically',
                                                                 'old_target': current_target,
                                                                 'new_target': new_dynamic_target,
                                                                 'current_price': current_price})
                                    update_executed = True
                                    logger.info(f"🔄 [Tracker] Target dynamically updated for {symbol} (ID: {signal_id}).")
                                else:
                                    logger.info(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): Conditions not met for dynamic target extension or new target is not higher.")
                            else:
                                logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): Could not get indicators for dynamic update.")
                        else:
                            logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): Could not fetch data for dynamic update.")

                    if update_executed and update_query:
                        try:
                             with conn.cursor() as update_cur:
                                 update_cur.execute(update_query, update_params)
                             conn.commit()
                             if log_message:
                                 logger.info(log_message)
                             if notification_details.get('type'):
                                 send_tracking_notification(notification_details)
                        except psycopg2.Error as db_err:
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): Database error during update: {db_err}")
                            if conn:
                                conn.rollback()
                        except Exception as exec_err:
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): Error during update/notification: {exec_err}", exc_info=True)
                            if conn:
                                conn.rollback()
                except Exception as inner_loop_err: logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): Error processing signal: {inner_loop_err}", exc_info=True)

            if active_signals_summary: logger.debug(f"ℹ️ [Tracker] End of cycle ({processed_in_cycle} processed): {'; '.join(active_signals_summary)}")
            time.sleep(5) # Increased wait time between tracking cycles
        except psycopg2.Error as db_cycle_err:
            logger.error(f"❌ [Tracker] Database error in tracking cycle: {db_cycle_err}. Reconnecting...")
            if conn:
                conn.rollback()
            time.sleep(30)
            check_db_connection()
        except Exception as cycle_err:
            logger.error(f"❌ [Tracker] Error in tracking cycle: {cycle_err}", exc_info=True)
            time.sleep(30)

def get_interval_minutes(interval: str) -> int:
    """Converts interval from string to minutes."""
    if interval.endswith('m'): return int(interval[:-1])
    elif interval.endswith('h'): return int(interval[:-1]) * 60
    elif interval.endswith('d'): return int(interval[:-1]) * 60 * 24
    return 0

# ---------------------- Flask Service (Optional for Webhook) ----------------------
app = Flask(__name__)
@app.route('/')
def home() -> Response:
    """Main status page."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ws_alive = ws_thread.is_alive() if 'ws_thread' in globals() and ws_thread else False
    tracker_alive = tracker_thread.is_alive() if 'tracker_thread' in globals() and tracker_thread else False
    status = "Running" if ws_alive and tracker_alive else "Partially Running"
    return Response(f"📈 Crypto Signal Bot ({status}) - Last Check: {now}", status=200, mimetype='text/plain')

@app.route('/favicon.ico')
def favicon() -> Response: return Response(status=204) # No Content

@app.route('/webhook', methods=['POST'])
def webhook() -> Tuple[str, int]:
    """Webhook endpoint to receive Telegram updates."""
    if not request.is_json: logger.warning("⚠️ [Flask] Webhook request is not JSON."); return "Invalid request", 400
    try:
        data = request.get_json(); logger.debug(f"ℹ️ [Flask] Webhook data: {json.dumps(data)[:200]}...")
        if 'callback_query' in data:
            callback_query = data['callback_query']; callback_id = callback_query['id']
            callback_data = callback_query.get('data'); message_info = callback_query.get('message')
            if not message_info or not callback_data: logger.warning(f"⚠️ [Flask] Callback {callback_id} missing message/data."); return "OK", 200
            chat_id_callback = message_info.get('chat', {}).get('id')
            if not chat_id_callback: logger.warning(f"⚠️ [Flask] Callback {callback_id} missing chat ID."); return "OK", 200
            try: requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery", json={'callback_query_id': callback_id}, timeout=5)
            except Exception as ack_err: logger.warning(f"⚠️ [Flask] Failed to acknowledge callback {callback_id}: {ack_err}")
            if callback_data == "get_report":
                Thread(target=lambda: send_telegram_message(chat_id_callback, generate_performance_report(), parse_mode='Markdown')).start()
                logger.info(f"ℹ️ [Flask] Triggered '/report' command from callback query for {chat_id_callback}.")
        elif 'message' in data:
            message_data = data['message']; chat_info = message_data.get('chat'); text_msg = message_data.get('text', '').strip()
            if not chat_info or not text_msg: logger.debug("ℹ️ [Flask] Message without chat/text."); return "OK", 200
            chat_id_msg = chat_info['id']
            if text_msg.lower() == '/report':
                Thread(target=lambda: send_telegram_message(chat_id_msg, generate_performance_report(), parse_mode='Markdown')).start()
                logger.info(f"ℹ️ [Flask] Triggered '/report' command from message for {chat_id_msg}.")
            elif text_msg.lower() == '/status':
                Thread(target=handle_status_command, args=(chat_id_msg,)).start()
                logger.info(f"ℹ️ [Flask] Triggered '/status' command from message for {chat_id_msg}.")
            else:
                logger.debug(f"ℹ️ [Flask] Unhandled message from {chat_id_msg}: '{text_msg}'")
        return "OK", 200
    except Exception as e: logger.error(f"❌ [Flask] Error processing Webhook: {e}", exc_info=True); return "Error", 500

def handle_status_command(chat_id_msg: int) -> None:
    """Handles the /status command from Telegram."""
    logger.info(f"ℹ️ [Flask Status] Handling /status command for chat {chat_id_msg}")
    status_msg = "⏳ Fetching status..."
    msg_sent = send_telegram_message(chat_id_msg, status_msg)
    if not (msg_sent and msg_sent.get('ok')): logger.error(f"❌ [Flask Status] Failed to send initial status to {chat_id_msg}"); return
    message_id_to_edit = msg_sent['result']['message_id'] if msg_sent and msg_sent.get('result') else None
    if not message_id_to_edit: logger.error(f"❌ [Flask Status] No message_id to update status in {chat_id_msg}"); return

    try:
        open_count = 0
        if check_db_connection() and conn:
            with conn.cursor() as status_cur:
                status_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;")
                open_count = (status_cur.fetchone() or {}).get('count', 0)
        ws_status = 'Active ✅' if 'ws_thread' in globals() and ws_thread and ws_thread.is_alive() else 'Inactive ❌'
        tracker_status = 'Active ✅' if 'tracker_thread' in globals() and tracker_thread and tracker_thread.is_alive() else 'Inactive ❌'
        final_status_msg = (
            f"🤖 *Bot Status:*\n"
            f"- Price Tracking (WS): {ws_status}\n"
            f"- Signal Tracking: {tracker_status}\n"
            f"- Active Signals: *{open_count}* / {MAX_OPEN_TRADES}\n"
            f"- Server Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        edit_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText"
        edit_payload = {'chat_id': chat_id_msg, 'message_id': message_id_to_edit, 'text': final_status_msg, 'parse_mode': 'Markdown'}
        requests.post(edit_url, json=edit_payload, timeout=10).raise_for_status()
        logger.info(f"✅ [Flask Status] Status updated for chat {chat_id_msg}")
    except Exception as status_err:
        logger.error(f"❌ [Flask Status] Error while fetching/editing status for {chat_id_msg}: {status_err}", exc_info=True)
        send_telegram_message(chat_id_msg, "❌ Error fetching status.")


def run_flask() -> None:
    """Runs the Flask server to handle Webhooks."""
    if not WEBHOOK_URL:
        logger.info("ℹ️ [Flask] Webhook URL not configured. Flask will not run.")
        return
    host = "0.0.0.0"; port = int(config('PORT', default=10000))
    logger.info(f"ℹ️ [Flask] Starting Flask app on {host}:{port}...")
    try:
        from waitress import serve
        logger.info("✅ [Flask] Using 'waitress'.")
        serve(app, host=host, port=port, threads=6)
    except ImportError:
        logger.warning("⚠️ [Flask] 'waitress' not installed. Using Flask development server.")
        app.run(host=host, port=port)
    except Exception as serve_err:
        logger.critical(f"❌ [Flask] Failed to start server: {serve_err}", exc_info=True)


# ---------------------- Main Loop and Check Function ----------------------
def main_loop() -> None:
    """Main loop for scanning the market and generating signals."""
    symbols_to_scan = get_crypto_symbols()
    if not symbols_to_scan: logger.critical("❌ [Main] No valid symbols loaded. Cannot proceed."); return
    logger.info(f"✅ [Main] Loaded {len(symbols_to_scan)} symbols for scanning.")

    while True:
        try:
            scan_start_time = time.time()
            logger.info(f"🔄 [Main] Starting market scan cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if not check_db_connection() or not conn:
                logger.error("❌ [Main] Database connection failed. Skipping scan.")
                time.sleep(60)
                continue

            open_count = 0
            try:
                 with conn.cursor() as cur_check:
                    cur_check.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;")
                    open_count = (cur_check.fetchone() or {}).get('count', 0)
            except psycopg2.Error as db_err:
                logger.error(f"❌ [Main] Database error while checking open signals: {db_err}. Skipping.")
                if conn:
                    conn.rollback()
                time.sleep(60)
                continue

            logger.info(f"ℹ️ [Main] Open Signals: {open_count} / {MAX_OPEN_TRADES}")
            if open_count >= MAX_OPEN_TRADES:
                logger.info(f"⚠️ [Main] Maximum open signals reached. Waiting...")
                time.sleep(get_interval_minutes(SIGNAL_GENERATION_TIMEFRAME) * 60) # Wait for the signal timeframe duration
                continue

            processed_in_loop = 0; signals_generated_in_loop = 0; slots_available = MAX_OPEN_TRADES - open_count
            logger.info(f"ℹ️ [Main] Slots available for new signals: {slots_available}")

            for symbol in symbols_to_scan:
                 if slots_available <= 0:
                     logger.info(f"ℹ️ [Main] Max reached during scan. Stopping.")
                     break
                 processed_in_loop += 1
                 logger.debug(f"🔍 [Main] Scanning symbol {symbol} ({processed_in_loop}/{len(symbols_to_scan)})...")
                 try:
                    with conn.cursor() as symbol_cur: # Check for existing open signal for symbol
                        symbol_cur.execute("SELECT 1 FROM signals WHERE symbol = %s AND achieved_target = FALSE AND hit_stop_loss = FALSE LIMIT 1;", (symbol,))
                        if symbol_cur.fetchone():
                            logger.debug(f"ℹ️ [Main] Open signal already exists for {symbol}. Skipping.")
                            continue
                    logger.debug(f"ℹ️ [Main] No open signal for {symbol}. Proceeding with analysis.")

                    # Fetch and process data for the signal timeframe (15m)
                    df_hist_signal_tf = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_hist_signal_tf is None or df_hist_signal_tf.empty:
                        logger.debug(f"⚠️ [Main] Could not fetch {SIGNAL_GENERATION_TIMEFRAME} frame data for {symbol}. Skipping.")
                        continue

                    strategy = ScalpingTradingStrategy(symbol)
                    # Populate indicators for the signal timeframe (15m)
                    df_indicators_signal_tf = strategy.populate_indicators(df_hist_signal_tf, SIGNAL_GENERATION_TIMEFRAME)
                    if df_indicators_signal_tf is None:
                        logger.debug(f"⚠️ [Main] Could not calculate {SIGNAL_GENERATION_TIMEFRAME} frame indicators for {symbol}. Skipping.")
                        continue

                    # Generate potential signal (which now includes the multi-TF and entry point checks internally)
                    potential_signal = strategy.generate_buy_signal(df_indicators_signal_tf)

                    if potential_signal:
                        logger.info(f"✨ [Main] Potential signal for {symbol}! (Score: {potential_signal.get('r2_score', 0):.2f})")
                        with conn.cursor() as final_check_cur: # Final check on open slots before inserting
                             final_check_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;")
                             final_open_count = (final_check_cur.fetchone() or {}).get('count', 0)
                             if final_open_count < MAX_OPEN_TRADES:
                                 logger.info(f"ℹ️ [Main] Slot available ({final_open_count}/{MAX_OPEN_TRADES}). Inserting signal for {symbol}.")
                                 if insert_signal_into_db(potential_signal):
                                     send_telegram_alert(potential_signal, SIGNAL_GENERATION_TIMEFRAME)
                                     signals_generated_in_loop += 1
                                     slots_available -= 1
                                     logger.info(f"✅ [Main] Signal inserted and sent for {symbol}. Remaining slots: {slots_available}.")
                                     time.sleep(2) # Small delay after sending alert
                                 else:
                                     logger.error(f"❌ [Main] Failed to insert signal for {symbol}.")
                             else:
                                 logger.warning(f"⚠️ [Main] Max reached before inserting {symbol}. Signal ignored.")
                                 break # Stop scanning if max open trades is reached during the loop
                    else:
                        logger.debug(f"ℹ️ [Main] No signal generated for {symbol} in this cycle.")

                 except psycopg2.Error as db_loop_err:
                     logger.error(f"❌ [Main] Database error for {symbol}: {db_loop_err}. Moving to next symbol...")
                     if conn:
                         conn.rollback()
                     continue
                 except Exception as symbol_proc_err:
                     logger.error(f"❌ [Main] General error while processing symbol {symbol}: {symbol_proc_err}", exc_info=True)
                     continue
                 time.sleep(0.1) # Small delay between processing symbols

            scan_duration = time.time() - scan_start_time
            logger.info(f"🏁 [Main] Scan finished. Signals generated: {signals_generated_in_loop}. Duration: {scan_duration:.2f} seconds.")
            frame_minutes = get_interval_minutes(SIGNAL_GENERATION_TIMEFRAME)
            # Wait until the next candle close for the signal timeframe, or at least 2 minutes
            wait_time = max(frame_minutes * 60 - (scan_duration % (frame_minutes * 60)), 120 - scan_duration) if scan_duration < 120 else frame_minutes * 60
            logger.info(f"⏳ [Main] Waiting {wait_time:.1f} seconds for next cycle...")
            time.sleep(wait_time)
        except KeyboardInterrupt:
            logger.info("🛑 [Main] Stop requested. Shutting down...")
            break
        except psycopg2.Error as db_main_err:
            logger.error(f"❌ [Main] Fatal database error: {db_main_err}. Reconnecting...")
            if conn:
                conn.rollback()
            time.sleep(60)
            try:
                init_db()
            except Exception as recon_err:
                logger.critical(f"❌ [Main] Failed to re-connect to database: {recon_err}. Exiting...")
                break
        except Exception as main_err:
            logger.error(f"❌ [Main] Unexpected error in main loop: {main_err}", exc_info=True)
            logger.info("ℹ️ [Main] Waiting 120 seconds before retrying...")
            time.sleep(120)


def cleanup_resources() -> None:
    """Cleans up resources before exiting."""
    global conn
    logger.info("ℹ️ [Cleanup] Closing resources...")
    if conn:
        try:
            conn.close()
            logger.info("✅ [DB] Database connection closed.")
        except Exception as close_err:
            logger.error(f"⚠️ [DB] Error closing database: {close_err}")
    logger.info("✅ [Cleanup] Resource cleanup complete.")

# ---------------------- Main Entry Point ----------------------
if __name__ == "__main__":
    logger.info("🚀 Starting Trading Signal Bot (No Stop Loss Version)...")
    logger.info(f"Local: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    ws_thread: Optional[Thread] = None
    tracker_thread: Optional[Thread] = None
    flask_thread: Optional[Thread] = None
    try:
        init_db()
        ws_thread = Thread(target=run_ticker_socket_manager, daemon=True, name="WebSocketThread")
        ws_thread.start()
        logger.info("✅ [Main] WebSocket Ticker thread started. Waiting 5 seconds for initialization...")
        time.sleep(5)
        if not ticker_data:
            logger.warning("⚠️ [Main] No initial data from WebSocket after 5 seconds.")
        else:
            logger.info(f"✅ [Main] Initial data from WebSocket for {len(ticker_data)} symbols.")
        tracker_thread = Thread(target=track_signals, daemon=True, name="TrackerThread")
        tracker_thread.start()
        logger.info("✅ [Main] Signal Tracker thread started.")
        if WEBHOOK_URL:
            flask_thread = Thread(target=run_flask, daemon=True, name="FlaskThread")
            flask_thread.start()
            logger.info("✅ [Main] Flask Webhook thread started.")
        else:
            logger.info("ℹ️ [Main] Webhook URL not configured, Flask server will not run.")
        main_loop()
    except Exception as startup_err:
        logger.critical(f"❌ [Main] Fatal error during startup/main loop: {startup_err}", exc_info=True)
    finally:
        logger.info("🛑 [Main] Shutting down program...")
        cleanup_resources()
        logger.info("👋 [Main] Trading Signal Bot stopped.")
        os._exit(0) # Force exit if threads are stuck

