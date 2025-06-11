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
from typing import List, Dict, Optional, Any, Tuple, Union

# استيراد مكتبات Flask والخيوط
from flask import Flask, request, Response
from threading import Thread

# ---------------------- إعداد التسجيل ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',\
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
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
    TELEGRAM_TOKEN: Optional[str] = config('TELEGRAM_BOT_TOKEN', default=None)
    CHAT_ID: Optional[str] = config('TELEGRAM_CHAT_ID', default=None)
except Exception as e:
     logger.critical(f"❌ فشل في تحميل المتغيرات البيئية الأساسية: {e}")
     exit(1)

logger.info(f"Binance API Key: {'Available' if API_KEY else 'Not available'}")
logger.info(f"Database URL: {'Available' if DB_URL else 'Not available'}")
logger.info(f"Webhook URL: {WEBHOOK_URL if WEBHOOK_URL else 'Not specified'} (Flask will always run for Render)")
logger.info(f"Telegram Token: {'Available' if TELEGRAM_TOKEN else 'Not available'}")
logger.info(f"Telegram Chat ID: {'Available' if CHAT_ID else 'Not available'}")


# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
SIGNAL_GENERATION_TIMEFRAME: str = '15m' # تم التغيير إلى 15 دقيقة
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 90 # 3 أشهر من البيانات
BASE_ML_MODEL_NAME: str = 'DecisionTree_Scalping_V1' # اسم أساسي للنموذج، سيتم إضافة الرمز إليه

# Indicator Parameters
VOLUME_LOOKBACK_CANDLES: int = 1 # عدد الشمعات لحساب متوسط الحجم (1 شمعة * 15 دقيقة = 15 دقيقة)
RSI_PERIOD: int = 9 
RSI_MOMENTUM_LOOKBACK_CANDLES: int = 2 
ENTRY_ATR_PERIOD: int = 10 
SUPERTRAND_PERIOD: int = 10 
SUPERTRAND_MULTIPLIER: float = 3.0 

# Ichimoku Cloud Parameters
TENKAN_PERIOD: int = 9
KIJUN_PERIOD: int = 26
SENKOU_SPAN_B_PERIOD: int = 52
CHIKOU_LAG: int = 26 # Lagging Span is plotted 26 periods back

# Fibonacci & S/R Parameters
FIB_SR_LOOKBACK_WINDOW: int = 50 # Lookback for identifying swing highs/lows for Fib & S/R

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
cur: Optional[psycopg2.extensions.cursor] = None
client: Optional[Client] = None

# متغيرات لتتبع حالة التدريب
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
                    signal_details JSONB,
                    stop_loss DOUBLE PRECISION -- Added stop loss column
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

# ---------------------- Technical Indicator Functions ----------------------
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


# NEW: Ichimoku Cloud Calculation
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


# NEW: Fibonacci Retracement Features
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
            # Retracement levels are calculated from (Swing Low + (Swing High - Swing Low) * Fib Level)
            # This is more for identifying resistance in an uptrend, or support in a pullback
            fib_0_236 = swing_high - (price_range * 0.236)
            fib_0_382 = swing_high - (price_range * 0.382)
            fib_0_500 = swing_high - (price_range * 0.500)
            fib_0_618 = swing_high - (price_range * 0.618)
            fib_0_786 = swing_high - (price_range * 0.786)

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


# NEW: Support and Resistance Features
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


# ---------------------- Model Training and Saving Functions ----------------------
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

def prepare_data_for_ml(df: pd.DataFrame, symbol: str, target_period: int = 5) -> Optional[pd.DataFrame]:
    """
    Prepares data for ML model training.
    Adds indicators (Volume, RSI Momentum, Bitcoin Trend, Supertrend, Ichimoku, Fibonacci, S/R)
    and removes rows with NaN values.
    Adds a 'target' column indicating if the price will rise in the coming candles.
    """
    logger.info(f"ℹ️ [ML Prep] Preparing data for ML model for {symbol} (All Indicators)...")

    # Determine minimum data length required for all features
    # Ichimoku requires KIJUN_PERIOD (26) for tenkan/kijun, SENKOU_SPAN_B_PERIOD (52) for span B, and CHIKOU_LAG (26)
    # Fibonacci/S/R require FIB_SR_LOOKBACK_WINDOW (50)
    # So, min_len_required should be the max of all periods + some buffer + target_period
    min_len_required = max(
        VOLUME_LOOKBACK_CANDLES,
        RSI_PERIOD,
        RSI_MOMENTUM_LOOKBACK_CANDLES,
        ENTRY_ATR_PERIOD,
        SUPERTRAND_PERIOD,
        TENKAN_PERIOD,
        KIJUN_PERIOD,
        SENKOU_SPAN_B_PERIOD,
        CHIKOU_LAG, # For Chikou Span to be valid (it's lagged)
        FIB_SR_LOOKBACK_WINDOW,
        55 # For BTC EMA calculation (50 + buffer)
    ) + target_period + 5 # Additional buffer for safe calculations

    if len(df) < min_len_required:
        logger.warning(f"⚠️ [ML Prep] DataFrame for {symbol} is too short ({len(df)} < {min_len_required}) for data preparation.")
        return None

    df_calc = df.copy()

    # Calculate required features only: 15m volume average (1 candle 15m)
    df_calc['volume_15m_avg'] = df_calc['volume'].rolling(window=VOLUME_LOOKBACK_CANDLES, min_periods=1).mean()
    logger.debug(f"ℹ️ [ML Prep] 15m volume average calculated for {symbol}.")

    # Calculate Relative Strength Index (RSI) as it's required for Momentum indicator
    df_calc = calculate_rsi_indicator(df_calc, RSI_PERIOD)

    # Add Bullish RSI Momentum indicator
    df_calc['rsi_momentum_bullish'] = 0
    if len(df_calc) >= RSI_MOMENTUM_LOOKBACK_CANDLES + 1:
        # Check if RSI is increasing over the last N candles and is above 50 (bullish territory)
        for i in range(RSI_MOMENTUM_LOOKBACK_CANDLES, len(df_calc)):
            rsi_slice = df_calc['rsi'].iloc[i - RSI_MOMENTUM_LOOKBACK_CANDLES : i + 1]
            if not rsi_slice.isnull().any() and np.all(np.diff(rsi_slice) > 0) and rsi_slice.iloc[-1] > 50:
                df_calc.loc[df_calc.index[i], 'rsi_momentum_bullish'] = 1
    logger.debug(f"ℹ️ [ML Prep] Bullish RSI momentum calculated for {symbol}.")

    # Calculate ATR (required for Supertrend)
    df_calc = calculate_atr_indicator(df_calc, ENTRY_ATR_PERIOD)

    # Calculate Supertrend
    df_calc = calculate_supertrend(df_calc, SUPERTRAND_PERIOD, SUPERTRAND_MULTIPLIER)
    logger.debug(f"ℹ️ [ML Prep] Supertrend indicator calculated for {symbol}.")


    # Fetch and calculate BTC trend feature
    btc_df = fetch_historical_data("BTCUSDT", interval=SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING)
    btc_trend_series = None
    if btc_df is not None and not btc_df.empty:
        btc_trend_series = _calculate_btc_trend_feature(btc_df)
        if btc_trend_series is not None:
            df_calc = df_calc.merge(btc_trend_series.rename('btc_trend_feature'),
                                    left_index=True, right_index=True, how='left')
            df_calc['btc_trend_feature'] = df_calc['btc_trend_feature'].fillna(0.0)
            logger.debug(f"ℹ️ [ML Prep] Bitcoin trend feature merged for {symbol}.")
        else:
            logger.warning(f"⚠️ [ML Prep] Bitcoin trend feature calculation failed. Defaulting 'btc_trend_feature' to 0.")
            df_calc['btc_trend_feature'] = 0.0
    else:
        logger.warning(f"⚠️ [ML Prep] Failed to fetch Bitcoin historical data. Defaulting 'btc_trend_feature' to 0.")
        df_calc['btc_trend_feature'] = 0.0

    # NEW: Calculate Ichimoku Cloud components and features
    df_calc = calculate_ichimoku_cloud(df_calc, TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_SPAN_B_PERIOD, CHIKOU_LAG)
    logger.debug(f"ℹ️ [ML Prep] Ichimoku Cloud features calculated for {symbol}.")

    # NEW: Calculate Fibonacci Retracement features
    df_calc = calculate_fibonacci_features(df_calc, FIB_SR_LOOKBACK_WINDOW)
    logger.debug(f"ℹ️ [ML Prep] Fibonacci Retracement features calculated for {symbol}.")

    # NEW: Calculate Support and Resistance features
    df_calc = calculate_support_resistance_features(df_calc, FIB_SR_LOOKBACK_WINDOW)
    logger.debug(f"ℹ️ [ML Prep] Support and Resistance features calculated for {symbol}.")


    # Define ALL feature columns that the model will use
    feature_columns = [
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

    # Ensure all feature columns exist and convert them to numeric
    for col in feature_columns:
        if col not in df_calc.columns:
            logger.warning(f"⚠️ [ML Prep] Missing feature column: {col}. Adding as NaN.")
            df_calc[col] = np.nan
        else:
            df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')

    # Create target column: Will the price rise by a certain percentage in the next candles?
    price_change_threshold = 0.005 # 0.5%
    df_calc['close'] = pd.to_numeric(df_calc['close'], errors='coerce')
    df_calc['future_max_close'] = df_calc['close'].shift(-target_period).rolling(window=target_period, min_periods=1).max()
    df_calc['target'] = ((df_calc['future_max_close'] / df_calc['close']) - 1 > price_change_threshold).astype(int)


    # Drop rows with NaN values after calculating indicators and target
    initial_len = len(df_calc)
    df_cleaned = df_calc.dropna(subset=feature_columns + ['target']).copy()
    dropped_count = initial_len - len(df_cleaned)

    if dropped_count > 0:
        logger.info(f"ℹ️ [ML Prep] For {symbol}: Dropped {dropped_count} rows due to NaN values after indicator and target calculation.")
    if df_cleaned.empty:
        logger.warning(f"⚠️ [ML Prep] DataFrame for {symbol} is empty after NaN removal for ML preparation.")
        return None

    logger.info(f"✅ [ML Prep] Data for {symbol} prepared successfully. Number of rows: {len(df_cleaned)}")
    return df_cleaned[feature_columns + ['target']]


def train_and_evaluate_model(data: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
    """
    Trains a Decision Tree model and evaluates its performance.
    """
    logger.info("ℹ️ [ML Train] Starting model training and evaluation...")

    if data.empty:
        logger.error("❌ [ML Train] Empty DataFrame for training.")
        return None, {}

    X = data.drop('target', axis=1)
    y = data['target']

    if X.empty or y.empty:
        logger.error("❌ [ML Train] Empty features or targets for training.")
        return None, {}

    # Split data into training and testing sets
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError as ve:
        logger.warning(f"⚠️ [ML Train] Cannot use stratify due to single class in target: {ve}. Proceeding without stratify.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Scaling features (important for some models, not necessarily for Decision Tree but good practice)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back to DataFrames with feature names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # Train a Decision Tree Classifier model
    model = DecisionTreeClassifier(random_state=42, max_depth=10) # Parameters can be tuned
    model.fit(X_train_scaled_df, y_train) # Fit with DataFrame
    logger.info("✅ [ML Train] Model trained successfully.")

    # Evaluation
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
        'feature_names': X.columns.tolist() # Save feature names to ensure consistency when loading
    }

    logger.info(f"📊 [ML Train] Model performance metrics:")
    logger.info(f"  - Accuracy: {accuracy:.4f}")
    logger.info(f"  - Precision: {precision:.4f}")
    logger.info(f"  - Recall: {recall:.4f}")
    logger.info(f"  - F1-Score: {f1:.4f}")

    return model, metrics

def save_ml_model_to_db(model: Any, model_name: str, metrics: Dict[str, Any]) -> bool:
    """
    Saves the trained model and its metadata (metrics) to the database.
    """
    logger.info(f"ℹ️ [DB Save] Checking database connection before saving...")
    if not check_db_connection() or not conn:
        logger.error("❌ [DB Save] Cannot save ML model due to database connection issue.")
        return False

    logger.info(f"ℹ️ [DB Save] Attempting to save ML model '{model_name}' to database...")
    try:
        # Serialize the model using pickle
        model_binary = pickle.dumps(model)
        logger.info(f"✅ [DB Save] Model serialized successfully. Data size: {len(model_binary)} bytes.")

        # Convert metrics to JSONB
        metrics_json = json.dumps(convert_np_values(metrics))
        logger.info(f"✅ [DB Save] Metrics converted to JSON successfully.")

        with conn.cursor() as db_cur:
            # Check if the model already exists (for update or insert)
            db_cur.execute("SELECT id FROM ml_models WHERE model_name = %s;", (model_name,))
            existing_model = db_cur.fetchone()

            if existing_model:
                logger.info(f"ℹ️ [DB Save] Model '{model_name}' already exists. Updating it.")
                update_query = sql.SQL("""
                    UPDATE ml_models
                    SET model_data = %s, trained_at = NOW(), metrics = %s
                    WHERE id = %s;
                """)
                db_cur.execute(update_query, (model_binary, metrics_json, existing_model['id']))
                logger.info(f"✅ [DB Save] ML model '{model_name}' updated in database successfully.")
            else:
                logger.info(f"ℹ️ [DB Save] Model '{model_name}' does not exist. Inserting as new model.")
                insert_query = sql.SQL("""
                    INSERT INTO ml_models (model_name, model_data, trained_at, metrics)
                    VALUES (%s, %s, NOW(), %s);
                """)
                db_cur.execute(insert_query, (model_name, model_binary, metrics_json))
                logger.info(f"✅ [DB Save] New ML model '{model_name}' saved to database successfully.")
        conn.commit()
        logger.info(f"✅ [DB Save] Database commit successful.")
        return True
    except psycopg2.Error as db_err:
        logger.error(f"❌ [DB Save] Database error while saving ML model: {db_err}", exc_info=True)
        if conn: conn.rollback()
        return False
    except pickle.PicklingError as pickle_err:
        logger.error(f"❌ [DB Save] Error serializing ML model: {pickle_err}", exc_info=True)
        if conn: conn.rollback()
        return False
    except Exception as e:
        logger.error(f"❌ [DB Save] Unexpected error while saving ML model: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

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

# ---------------------- Telegram Functions (Copied from c4.py) ----------------------
def send_telegram_message(target_chat_id: str, text: str, reply_markup: Optional[Dict] = None, parse_mode: str = 'Markdown', disable_web_page_preview: bool = True, timeout: int = 20) -> Optional[Dict]:
    """Sends a message via Telegram Bot API with improved error handling."""
    if not TELEGRAM_TOKEN or not target_chat_id:
        logger.warning("⚠️ [Telegram] Cannot send Telegram message: TELEGRAM_TOKEN or CHAT_ID is missing.")
        return None

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
        status_message += f"- Last Training Metrics (Average Accuracy): {last_training_metrics.get('avg_accuracy', 'N/A'):.4f}\n"
        status_message += f"- Successful Models: {last_training_metrics.get('successful_models', 'N/A')}/{last_training_metrics.get('total_models_trained', 'N/A')}\n"
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


# ---------------------- Main Entry Point ----------------------
if __name__ == "__main__":
    logger.info("🚀 Starting ML model training script...")
    logger.info(f"Local Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | UTC Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    flask_thread: Optional[Thread] = None
    initial_training_start_time = datetime.now() # Track overall training duration

    try:
        # 1. Start Flask service in a separate thread first
        # This ensures the service will be available to respond to Uptime Monitor requests
        # while the training process is running in the main thread.
        flask_thread = Thread(target=run_flask_service, daemon=False, name="FlaskServiceThread")
        flask_thread.start()
        logger.info("✅ [Main] Flask service started.")
        time.sleep(2) # Give Flask some time to start

        # 2. Initialize the database
        init_db()

        # 3. Fetch the list of symbols
        symbols = get_crypto_symbols()
        if not symbols:
            logger.critical("❌ [Main] No valid symbols to train. Please check 'crypto_list.txt'.")
            training_status = "Failed: No valid symbols"
            # Send Telegram notification for failure
            if TELEGRAM_TOKEN and CHAT_ID:
                send_telegram_message(CHAT_ID,
                                      f"❌ *ML Model Training Failed to Start:*\n"
                                      f"No valid symbols to train. Please check `crypto_list.txt`.",
                                      parse_mode='Markdown')
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

        # Send Telegram notification for training start
        if TELEGRAM_TOKEN and CHAT_ID:
            send_telegram_message(CHAT_ID,
                                  f"🚀 *ML Model Training Started:*\n"
                                  f"Training models for {len(symbols)} symbols.\n"
                                  f"Time: {initial_training_start_time.strftime('%Y-%m-%d %H:%M:%S')}",
                                  parse_mode='Markdown')


        # 4. Train a model for each symbol separately
        for symbol in symbols:
            current_model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
            overall_metrics['total_models_trained'] += 1
            logger.info(f"\n--- ⏳ [Main] Starting model training for {symbol} ({current_model_name}) ---")
            
            try:
                # Fetch historical data for the current symbol
                df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING)
                if df_hist is None or df_hist.empty:
                    logger.warning(f"⚠️ [Main] Could not fetch sufficient data for {symbol}. Skipping model training for this symbol.")
                    overall_metrics['failed_models'] += 1
                    overall_metrics['details_per_symbol'][symbol] = {'status': 'Failed: No data', 'error': 'No sufficient historical data'}
                    continue

                # Prepare data for ML model
                df_processed = prepare_data_for_ml(df_hist, symbol)
                if df_processed is None or df_processed.empty:
                    logger.warning(f"⚠️ [Main] No processed data available for training for {symbol} after indicator preprocessing. Skipping.")
                    overall_metrics['failed_models'] += 1
                    overall_metrics['details_per_symbol'][symbol] = {'status': 'Failed: No processed data', 'error': 'No sufficient processed data'}
                    continue

                # Train and evaluate the model
                trained_model, model_metrics = train_and_evaluate_model(df_processed)

                if trained_model is None:
                    logger.error(f"❌ [Main] Model training failed for {symbol}. Cannot save.")
                    overall_metrics['failed_models'] += 1
                    overall_metrics['details_per_symbol'][symbol] = {'status': 'Failed: Training failed', 'error': 'Model training returned None'}
                    continue

                # Save the model to the database
                if save_ml_model_to_db(trained_model, current_model_name, model_metrics):
                    logger.info(f"✅ [Main] Model '{current_model_name}' saved successfully to database.")
                    overall_metrics['successful_models'] += 1
                    overall_metrics['details_per_symbol'][symbol] = {'status': 'Completed Successfully', 'metrics': model_metrics}
                    
                    total_accuracy += model_metrics.get('accuracy', 0.0)
                    total_precision += model_metrics.get('precision', 0.0)
                    total_recall += model_metrics.get('recall', 0.0)
                    total_f1_score += model_metrics.get('f1_score', 0.0)
                else:
                    logger.error(f"❌ [Main] Failed to save model '{current_model_name}' to database.")
                    overall_metrics['failed_models'] += 1
                    overall_metrics['details_per_symbol'][symbol] = {'status': 'Completed with Errors: Model save failed', 'error': 'Failed to save model to DB'}

            except Exception as e:
                logger.critical(f"❌ [Main] A fatal error occurred during model training for {symbol}: {e}", exc_info=True)
                overall_metrics['failed_models'] += 1
                overall_metrics['details_per_symbol'][symbol] = {'status': 'Failed: Unhandled exception', 'error': str(e)}
            
            logger.info(f"--- ✅ [Main] Model training for {symbol} finished ---")
            time.sleep(1) # Small delay between model training

        # Update overall training status
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

        # Calculate total training duration
        training_duration = last_training_time - initial_training_start_time
        training_duration_str = str(training_duration).split('.')[0] # Remove microseconds

        # Send Telegram notification for training completion/failure
        if TELEGRAM_TOKEN and CHAT_ID:
            if training_status == "Completed Successfully (All Models Trained)":
                message_title = "✅ *ML Model Training Completed Successfully!*"
            elif training_status == "Completed with Errors (Some Models Failed)":
                message_title = "⚠️ *ML Model Training Completed with Errors!*"
            else:
                message_title = "❌ *ML Model Training Failed!*"
            
            telegram_message = (
                f"{message_title}\n"
                f"——————————————\n"
                f"📊 *Summary:*\n"
                f"- Total Models Trained: {overall_metrics['total_models_trained']}\n"
                f"- Successful Models: {overall_metrics['successful_models']}\n"
                f"- Failed Models: {overall_metrics['failed_models']}\n"
                f"- Average Accuracy: {overall_metrics['avg_accuracy']:.4f}\n"
                f"- Average Precision: {overall_metrics['avg_precision']:.4f}\n"
                f"- Average Recall: {overall_metrics['avg_recall']:.4f}\n"
                f"- Average F1-Score: {overall_metrics['avg_f1_score']:.4f}\n"
                f"——————————————\n"
                f"⏱️ *Total Training Duration:* {training_duration_str}\n"
                f"⏰ *Completion Time:* {last_training_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            if training_error:
                telegram_message += f"\n\n🚨 *General Error:* {training_error}"
            
            send_telegram_message(CHAT_ID, telegram_message, parse_mode='Markdown')

        # Wait for the Flask thread to finish (it usually won't unless there's an error)
        if flask_thread:
            flask_thread.join()

    except Exception as e:
        logger.critical(f"❌ [Main] A fatal error occurred during main training script execution: {e}", exc_info=True)
        training_status = "Failed: Unhandled exception in main loop"
        training_error = str(e)
        # Send Telegram notification for critical unhandled error
        if TELEGRAM_TOKEN and CHAT_ID:
            error_message = (
                f"🚨 *Fatal Error in ML Model Training Script:*\n"
                f"An unexpected error occurred causing the script to stop.\n"
                f"Details: `{e}`\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            send_telegram_message(CHAT_ID, error_message, parse_mode='Markdown')
    finally:
        logger.info("🛑 [Main] Shutting down training script...")
        cleanup_resources()
        logger.info("👋 [Main] ML model training script stopped.")
        # os._exit(0) # Do not use os._exit(0) here if you want Flask to keep running
