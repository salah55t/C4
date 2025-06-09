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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',\
    handlers=[
        logging.FileHandler('crypto_bot_enhanced.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotEnhanced')

# ---------------------- تحميل المتغيرات البيئية ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
except Exception as e:
     logger.critical(f"❌ فشل في تحميل المتغيرات البيئية الأساسية: {e}")
     exit(1)

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
TRADE_VALUE: float = 10.0
MAX_OPEN_TRADES: int = 5
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 3
SIGNAL_TRACKING_TIMEFRAME: str = '15m'
SIGNAL_TRACKING_LOOKBACK_DAYS: int = 1

# Indicator Parameters
RSI_PERIOD: int = 14
VOLUME_LOOKBACK_CANDLES: int = 2
RSI_MOMENTUM_LOOKBACK_CANDLES: int = 3
ENTRY_ATR_PERIOD: int = 14
ENTRY_ATR_MULTIPLIER: float = 2.0
SUPERTREND_PERIOD: int = 10
SUPERTREND_MULTIPLIER: float = 3.0
# NEW: MACD Parameters
MACD_FAST_PERIOD: int = 12
MACD_SLOW_PERIOD: int = 26
MACD_SIGNAL_PERIOD: int = 9
# NEW: Bollinger Bands Parameters
BB_PERIOD: int = 20
BB_STD_DEV: int = 2
# NEW: ADX Parameters
ADX_PERIOD: int = 14

MIN_PROFIT_MARGIN_PCT: float = 1.0
MIN_VOLUME_15M_USDT: float = 75000.0
TARGET_APPROACH_THRESHOLD_PCT: float = 0.005
BINANCE_FEE_RATE: float = 0.001

BASE_ML_MODEL_NAME: str = 'DecisionTree_Scalping_V2_Enhanced' # Updated model name

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
cur: Optional[psycopg2.extensions.cursor] = None
client: Optional[Client] = None
ticker_data: Dict[str, float] = {}
ml_models: Dict[str, Any] = {}

# ---------------------- Binance Client Setup ----------------------
try:
    logger.info("ℹ️ [Binance] Initializing Binance client...")
    client = Client(API_KEY, API_SECRET)
    client.ping()
    server_time = client.get_server_time()
    logger.info(f"✅ [Binance] Binance client initialized. Server Time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
except (BinanceAPIException, BinanceRequestException) as e:
    logger.critical(f"❌ [Binance] Failed to initialize Binance client: {e}")
    exit(1)

# ---------------------- Additional Indicator Functions (including new ones) ----------------------

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client:
        logger.error("❌ [Data] Binance client not initialized for fetching data.")
        return None
    try:
        start_str_overall = (datetime.utcnow() - timedelta(days=days + 1)).strftime("%Y-%m-%d %H:%M:%S")
        logger.debug(f"ℹ️ [Data] Fetching {interval} data for {symbol} from {start_str_overall}...")
        
        interval_map = {
            '15m': Client.KLINE_INTERVAL_15MINUTE, '5m': Client.KLINE_INTERVAL_5MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR, '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY
        }
        binance_interval = interval_map.get(interval)
        if not binance_interval:
            logger.error(f"❌ [Data] Unsupported timeframe: {interval}")
            return None

        klines = client.get_historical_klines(symbol, binance_interval, start_str_overall)
        if not klines:
            logger.warning(f"⚠️ [Data] No historical data ({interval}) found for {symbol}.")
            return None

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[numeric_cols].dropna()
        df.sort_index(inplace=True)
        
        logger.debug(f"✅ [Data] Fetched and processed {len(df)} historical klines ({interval}) for {symbol}.")
        return df
    except (BinanceAPIException, BinanceRequestException) as e:
        logger.error(f"❌ [Data] Binance error fetching data for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ [Data] Unexpected error fetching historical data for {symbol}: {e}", exc_info=True)
        return None


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculates Exponential Moving Average (EMA)."""
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi_indicator(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """Calculates Relative Strength Index (RSI)."""
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)
    return df

def calculate_atr_indicator(df: pd.DataFrame, period: int = ENTRY_ATR_PERIOD) -> pd.DataFrame:
    """Calculates Average True Range (ATR)."""
    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)
    df['atr'] = tr.ewm(span=period, adjust=False).mean()
    return df

def calculate_supertrend(df: pd.DataFrame, period: int = SUPERTREND_PERIOD, multiplier: float = SUPERTREND_MULTIPLIER) -> pd.DataFrame:
    """Calculates the Supertrend indicator."""
    if 'atr' not in df.columns:
        df = calculate_atr_indicator(df, period)
    
    hl2 = (df['high'] + df['low']) / 2
    df['upper_band'] = hl2 + (multiplier * df['atr'])
    df['lower_band'] = hl2 - (multiplier * df['atr'])
    df['in_uptrend'] = True

    for current in range(1, len(df.index)):
        previous = current - 1
        if df['close'].iloc[current] > df['upper_band'].iloc[previous]:
            df.loc[df.index[current], 'in_uptrend'] = True
        elif df['close'].iloc[current] < df['lower_band'].iloc[previous]:
            df.loc[df.index[current], 'in_uptrend'] = False
        else:
            df.loc[df.index[current], 'in_uptrend'] = df['in_uptrend'].iloc[previous]
            if df['in_uptrend'].iloc[current] and df['lower_band'].iloc[current] < df['lower_band'].iloc[previous]:
                df.loc[df.index[current], 'lower_band'] = df['lower_band'].iloc[previous]
            if not df['in_uptrend'].iloc[current] and df['upper_band'].iloc[current] > df['upper_band'].iloc[previous]:
                df.loc[df.index[current], 'upper_band'] = df['upper_band'].iloc[previous]
    
    df['supertrend'] = np.where(df['in_uptrend'], df['lower_band'], df['upper_band'])
    df['supertrend_direction'] = np.where(df['in_uptrend'], 1, -1)
    df.drop(['upper_band', 'lower_band', 'in_uptrend'], axis=1, inplace=True)
    return df

def _calculate_btc_trend_feature(df_btc: pd.DataFrame) -> Optional[pd.Series]:
    """Calculates a numerical representation of Bitcoin's trend."""
    min_data_for_ema = 55
    if df_btc is None or len(df_btc) < min_data_for_ema:
        return pd.Series(index=df_btc.index if df_btc is not None else None, data=0.0)
    
    ema20 = calculate_ema(df_btc['close'], 20)
    ema50 = calculate_ema(df_btc['close'], 50)
    
    trend_series = pd.Series(index=ema20.index, data=0.0)
    trend_series[(df_btc['close'] > ema20) & (ema20 > ema50)] = 1.0
    trend_series[(df_btc['close'] < ema20) & (ema20 < ema50)] = -1.0
    
    return trend_series.reindex(df_btc.index).fillna(0.0)

# NEW: Function to calculate MACD
def calculate_macd(df: pd.DataFrame, fast_period: int = MACD_FAST_PERIOD, slow_period: int = MACD_SLOW_PERIOD, signal_period: int = MACD_SIGNAL_PERIOD) -> pd.DataFrame:
    """Calculates MACD, MACD Signal, and MACD Histogram."""
    df['ema_fast'] = calculate_ema(df['close'], span=fast_period)
    df['ema_slow'] = calculate_ema(df['close'], span=slow_period)
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = calculate_ema(df['macd'], span=signal_period)
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df.drop(['ema_fast', 'ema_slow'], axis=1, inplace=True)
    return df

# NEW: Function to calculate Bollinger Bands
def calculate_bollinger_bands(df: pd.DataFrame, period: int = BB_PERIOD, std_dev: int = BB_STD_DEV) -> pd.DataFrame:
    """Calculates Bollinger Bands."""
    df['bb_ma'] = df['close'].rolling(window=period).mean()
    df['bb_std'] = df['close'].rolling(window=period).std()
    df['bb_upper'] = df['bb_ma'] + (df['bb_std'] * std_dev)
    df['bb_lower'] = df['bb_ma'] - (df['bb_std'] * std_dev)
    df.drop(['bb_ma', 'bb_std'], axis=1, inplace=True)
    return df

# NEW: Function to calculate ADX
def calculate_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    """Calculates Average Directional Index (ADX)."""
    df['plus_dm'] = (df['high'].diff() > df['low'].diff(-1)).astype(int) * df['high'].diff()
    df['minus_dm'] = (df['low'].diff() < df['high'].diff(-1)).astype(int) * df['low'].diff(-1)
    
    df['plus_dm'] = df['plus_dm'].clip(lower=0)
    df['minus_dm'] = df['minus_dm'].clip(lower=0)
    
    df['tr'] = calculate_atr_indicator(df, period)['atr']
    
    df['atr_smoothed'] = df['tr'].ewm(alpha=1/period, adjust=False).mean()
    df['plus_di'] = 100 * (df['plus_dm'].ewm(alpha=1/period, adjust=False).mean() / df['atr_smoothed'])
    df['minus_di'] = 100 * (df['minus_dm'].ewm(alpha=1/period, adjust=False).mean() / df['atr_smoothed'])
    
    df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']).replace(0, 1))
    df['adx'] = df['dx'].ewm(alpha=1/period, adjust=False).mean()

    df.drop(['plus_dm', 'minus_dm', 'tr', 'atr_smoothed', 'plus_di', 'minus_di', 'dx'], axis=1, inplace=True)
    return df

# ---------------------- Database Connection Setup ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    """Initializes database connection and creates tables if they don't exist."""
    global conn, cur
    logger.info("[DB] Initializing database...")
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            cur = conn.cursor()
            logger.info("✅ [DB] Database connected successfully.")

            # Create tables
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION NOT NULL,
                    initial_target DOUBLE PRECISION NOT NULL, current_target DOUBLE PRECISION NOT NULL,
                    r2_score DOUBLE PRECISION, volume_15m DOUBLE PRECISION, achieved_target BOOLEAN DEFAULT FALSE,
                    closing_price DOUBLE PRECISION, closed_at TIMESTAMP, sent_at TIMESTAMP DEFAULT NOW(),
                    entry_time TIMESTAMP DEFAULT NOW(), time_to_target INTERVAL, profit_percentage DOUBLE PRECISION,
                    strategy_name TEXT, signal_details JSONB, stop_loss DOUBLE PRECISION
                );
                CREATE TABLE IF NOT EXISTS ml_models (
                    id SERIAL PRIMARY KEY, model_name TEXT NOT NULL UNIQUE, model_data BYTEA NOT NULL,
                    trained_at TIMESTAMP DEFAULT NOW(), metrics JSONB
                );
            """)
            conn.commit()
            logger.info("✅ [DB] Tables 'signals' and 'ml_models' are ready.")
            return
        except (OperationalError, Exception) as e:
            logger.error(f"❌ [DB] Connection attempt {attempt + 1} failed: {e}")
            if conn: conn.rollback()
            if attempt == retries - 1:
                 logger.critical("❌ [DB] All database connection attempts failed.")
                 raise e
            time.sleep(delay)

def check_db_connection() -> bool:
    """Checks and re-initializes database connection if needed."""
    global conn, cur
    try:
        if conn is None or conn.closed != 0:
            logger.warning("⚠️ [DB] Connection is closed or non-existent. Re-initializing...")
            init_db()
        else:
             with conn.cursor() as check_cur:
                  check_cur.execute("SELECT 1;")
        return True
    except (OperationalError, InterfaceError, Exception) as e:
        logger.error(f"❌ [DB] Database connection lost ({e}). Attempting to reconnect...")
        try:
             init_db()
             return True
        except Exception as recon_err:
            logger.error(f"❌ [DB] Reconnect attempt failed: {recon_err}")
            return False

def load_ml_model_from_db(symbol: str) -> Optional[Any]:
    """Loads the latest trained ML model for a specific symbol from the database."""
    global ml_models
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_name in ml_models:
        return ml_models[model_name]

    if not check_db_connection() or not conn:
        logger.error(f"❌ [ML Model] Cannot load model for {symbol}, DB connection issue.")
        return None

    try:
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT model_data FROM ml_models WHERE model_name = %s ORDER BY trained_at DESC LIMIT 1;", (model_name,))
            result = db_cur.fetchone()
            if result and result['model_data']:
                model = pickle.loads(result['model_data'])
                ml_models[model_name] = model
                logger.info(f"✅ [ML Model] Loaded model '{model_name}' from DB.")
                return model
            else:
                logger.warning(f"⚠️ [ML Model] Model '{model_name}' not found in DB.")
                return None
    except (psycopg2.Error, pickle.UnpicklingError, Exception) as e:
        logger.error(f"❌ [ML Model] Failed to load model for {symbol}: {e}", exc_info=True)
        return None

def convert_np_values(obj: Any) -> Any:
    """Converts NumPy data types to native Python types for JSON and DB compatibility."""
    if isinstance(obj, (np.integer, np.int_)): return int(obj)
    if isinstance(obj, (np.floating, np.float_)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_np_values(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_np_values(item) for item in obj]
    if pd.isna(obj): return None
    return obj

# ---------------------- WebSocket Management ----------------------
def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
    """Handles incoming WebSocket messages for mini-ticker prices."""
    global ticker_data
    try:
        data_list = msg if isinstance(msg, list) else msg.get('data', [])
        for item in data_list:
            symbol = item.get('s')
            price_str = item.get('c')
            if symbol and 'USDT' in symbol and price_str:
                ticker_data[symbol] = float(price_str)
    except (ValueError, Exception) as e:
        logger.error(f"❌ [WS] Error processing ticker message: {e}", exc_info=True)

def run_ticker_socket_manager() -> None:
    """Runs and manages the WebSocket connection for mini-ticker."""
    while True:
        try:
            logger.info("ℹ️ [WS] Starting WebSocket manager...")
            twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
            twm.start()
            stream_name = twm.start_miniticker_socket(callback=handle_ticker_message)
            logger.info(f"✅ [WS] WebSocket stream started: {stream_name}")
            twm.join()
            logger.warning("⚠️ [WS] WebSocket manager stopped. Restarting...")
        except Exception as e:
            logger.error(f"❌ [WS] WebSocket manager crashed: {e}. Restarting in 15s...", exc_info=True)
        time.sleep(15)

# ---------------------- Other Helper Functions ----------------------
def fetch_recent_volume(symbol: str, interval: str = SIGNAL_GENERATION_TIMEFRAME, num_candles: int = VOLUME_LOOKBACK_CANDLES) -> float:
    """Fetches recent trading volume in USDT."""
    if not client: return 0.0
    try:
        interval_map = {'15m': Client.KLINE_INTERVAL_15MINUTE}
        klines = client.get_klines(symbol=symbol, interval=interval_map.get(interval, Client.KLINE_INTERVAL_15MINUTE), limit=num_candles)
        return sum(float(k[7]) for k in klines)
    except (BinanceAPIException, BinanceRequestException, Exception) as e:
        logger.error(f"❌ [Volume] Failed to fetch volume for {symbol}: {e}")
        return 0.0

def get_crypto_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """Reads and validates crypto symbols from a file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            raw_symbols = [f"{line.strip().upper()}USDT" for line in f if line.strip() and not line.startswith('#')]
        
        if not client:
            logger.error("❌ [Symbols] Binance client not ready for symbol validation.")
            return raw_symbols

        exchange_info = client.get_exchange_info()
        valid_symbols = {s['symbol'] for s in exchange_info['symbols'] if s.get('status') == 'TRADING' and s.get('isSpotTradingAllowed')}
        validated_symbols = [s for s in raw_symbols if s in valid_symbols]
        logger.info(f"✅ [Symbols] Validated {len(validated_symbols)} symbols for trading.")
        return validated_symbols
    except (FileNotFoundError, Exception) as e:
        logger.error(f"❌ [Symbols] Failed to read or validate symbols from '{filename}': {e}", exc_info=True)
        return []

# ---------------------- Trading Strategy (ENHANCED) -------------------
class EnhancedTradingStrategy:
    """Encapsulates the enhanced trading strategy logic."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ml_model = load_ml_model_from_db(symbol)
        if self.ml_model is None:
            logger.warning(f"⚠️ [Strategy {self.symbol}] ML model not loaded. Strategy will not generate signals.")

        self.feature_columns_for_ml = [
            'volume_15m_avg', 'rsi_momentum_bullish', 'btc_trend_feature', 
            'supertrend_direction', 'macd_hist', 'bb_upper_dist', 'bb_lower_dist', 'adx'
        ]

    def populate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculates all required indicators for the ML model."""
        logger.debug(f"ℹ️ [Strategy {self.symbol}] Calculating indicators for ML model...")
        try:
            df_calc = df.copy()
            # Calculate base indicators
            df_calc = calculate_rsi_indicator(df_calc, RSI_PERIOD)
            df_calc = calculate_atr_indicator(df_calc, ENTRY_ATR_PERIOD)
            df_calc = calculate_supertrend(df_calc, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
            
            # NEW: Calculate MACD, Bollinger Bands, ADX
            df_calc = calculate_macd(df_calc)
            df_calc = calculate_bollinger_bands(df_calc)
            df_calc = calculate_adx(df_calc)

            # Feature Engineering
            df_calc['volume_15m_avg'] = df_calc['volume'].rolling(window=VOLUME_LOOKBACK_CANDLES).mean()
            df_calc['rsi_momentum_bullish'] = ((df_calc['rsi'].diff(RSI_MOMENTUM_LOOKBACK_CANDLES) > 0) & (df_calc['rsi'] > 50)).astype(int)
            
            # NEW: BB Features
            df_calc['bb_upper_dist'] = (df_calc['bb_upper'] - df_calc['close']) / df_calc['close']
            df_calc['bb_lower_dist'] = (df_calc['close'] - df_calc['bb_lower']) / df_calc['close']

            # Fetch and merge BTC trend feature
            btc_df = fetch_historical_data("BTCUSDT", interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
            if btc_df is not None:
                btc_trend_series = _calculate_btc_trend_feature(btc_df)
                df_calc = df_calc.merge(btc_trend_series.rename('btc_trend_feature'), left_index=True, right_index=True, how='left')
                df_calc['btc_trend_feature'].fillna(0.0, inplace=True)
            else:
                df_calc['btc_trend_feature'] = 0.0

            # Clean up and ensure all feature columns are numeric
            for col in self.feature_columns_for_ml:
                if col not in df_calc.columns:
                    df_calc[col] = np.nan
            
            df_cleaned = df_calc.dropna(subset=self.feature_columns_for_ml + ['atr']).copy()
            logger.debug(f"✅ [Strategy {self.symbol}] Indicators calculated. Cleaned DF length: {len(df_cleaned)}")
            return df_cleaned

        except (KeyError, Exception) as e:
            logger.error(f"❌ [Strategy {self.symbol}] Error during indicator population: {e}", exc_info=True)
            return None

    def generate_buy_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generates a buy signal based on ML prediction and strong filter conditions."""
        if df_processed is None or df_processed.empty or self.ml_model is None:
            return None

        last_row = df_processed.iloc[-1]
        current_price = ticker_data.get(self.symbol)
        if current_price is None:
            logger.warning(f"⚠️ [Strategy {self.symbol}] Current price not available from ticker.")
            return None

        signal_details = {}

        # --- ML Model Prediction ---
        try:
            features_for_prediction = pd.DataFrame([last_row[self.feature_columns_for_ml]], columns=self.feature_columns_for_ml)
            ml_pred = self.ml_model.predict(features_for_prediction)[0]
            ml_is_bullish = ml_pred == 1
            signal_details['ML_Prediction'] = 'Bullish ✅' if ml_is_bullish else 'Bearish ❌'
        except Exception as ml_err:
            logger.error(f"❌ [Strategy {self.symbol}] ML prediction error: {ml_err}")
            return None

        # --- Strict Filtering Conditions ---
        # Condition 1: ML model must predict bullish
        if not ml_is_bullish:
            logger.info(f"ℹ️ [Strategy {self.symbol}] Signal rejected: ML prediction is not bullish.")
            return None

        # Condition 2: Supertrend must be bullish
        supertrend_is_bullish = last_row.get('supertrend_direction') == 1
        signal_details['Supertrend_Filter'] = f'Pass ({last_row.get("supertrend_direction")})' if supertrend_is_bullish else 'Fail'
        if not supertrend_is_bullish: return None

        # Condition 3: MACD Histogram must be positive and rising
        macd_hist_positive_rising = last_row.get('macd_hist', -1) > 0 and last_row.get('macd_hist', -1) > df_processed['macd_hist'].iloc[-2]
        signal_details['MACD_Filter'] = 'Pass' if macd_hist_positive_rising else 'Fail'
        if not macd_hist_positive_rising: return None
        
        # Condition 4: ADX must show a strong trend
        adx_strong_trend = last_row.get('adx', 0) > 25
        signal_details['ADX_Filter'] = f'Pass ({last_row.get("adx", 0):.1f})' if adx_strong_trend else 'Fail'
        if not adx_strong_trend: return None

        # Condition 5: Volume Check
        volume_recent = fetch_recent_volume(self.symbol)
        volume_ok = volume_recent >= MIN_VOLUME_15M_USDT
        signal_details['Volume_Check'] = f'Pass ({volume_recent:,.0f})' if volume_ok else 'Fail'
        if not volume_ok: return None

        # --- Target and Stop Loss Calculation ---
        current_atr = last_row.get('atr')
        if pd.isna(current_atr) or current_atr <= 0: return None
        initial_target = current_price + (ENTRY_ATR_MULTIPLIER * current_atr)
        initial_stop_loss = last_row.get('supertrend')

        # Final Profit Margin Check
        profit_margin_pct = ((initial_target / current_price) - 1) * 100
        profit_margin_ok = profit_margin_pct >= MIN_PROFIT_MARGIN_PCT
        signal_details['Profit_Margin_Check'] = f'Pass ({profit_margin_pct:.2f}%)' if profit_margin_ok else 'Fail'
        if not profit_margin_ok: return None

        signal_output = {
            'symbol': self.symbol, 'entry_price': float(current_price),
            'initial_target': float(initial_target), 'current_target': float(initial_target),
            'stop_loss': float(initial_stop_loss), 'strategy_name': 'Scalping_ML_Enhanced_V2',
            'signal_details': signal_details, 'volume_15m': volume_recent
        }
        logger.info(f"✅ [Strategy {self.symbol}] CONFIRMED Enhanced Buy Signal. Price: {current_price:.6f}, Target: {initial_target:.6f}, SL: {initial_stop_loss:.6f}")
        return signal_output


# ---------------------- Telegram Functions & Main Loop ----------------------
def send_telegram_message(target_chat_id: str, text: str, **kwargs):
    """Sends a message via Telegram Bot API."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(target_chat_id), 'text': text, 'parse_mode': 'Markdown', **kwargs}
    try:
        response = requests.post(url, json=payload, timeout=20)
        response.raise_for_status()
        return response.json()
    except (requests.exceptions.RequestException, Exception) as e:
        logger.error(f"❌ [Telegram] Failed to send message: {e}")
        return None

# ... (Rest of the functions like send_telegram_alert, generate_performance_report, track_signals, main_loop, Flask app etc. can be reused from your original script)
# ... MAKE SURE to adjust send_telegram_alert to display the new signal_details from the enhanced strategy.

# This is a placeholder for the rest of the file. You should integrate the functions above
# with the existing `c4.py` logic (tracking, reporting, main loop).

def main_loop() -> None:
    """Main loop to scan pairs and generate signals."""
    symbols_to_scan = get_crypto_symbols()
    if not symbols_to_scan:
        logger.critical("❌ [Main] No valid symbols loaded. Exiting.")
        return

    logger.info(f"✅ [Main] Loaded {len(symbols_to_scan)} valid symbols for scanning.")

    while True:
        try:
            scan_start_time = time.time()
            logger.info(f"🔄 [Main] Starting market scan cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            if not check_db_connection() or not conn:
                logger.error("❌ [Main] Skipping scan cycle due to DB connection failure.")
                time.sleep(60)
                continue

            with conn.cursor() as cur_check:
                cur_check.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE;")
                open_count = (cur_check.fetchone() or {}).get('count', 0)

            logger.info(f"ℹ️ [Main] Currently open trades: {open_count} / {MAX_OPEN_TRADES}")
            if open_count >= MAX_OPEN_TRADES:
                logger.info(f"⚠️ [Main] Max open trades reached. Waiting...")
                time.sleep(get_interval_minutes(SIGNAL_GENERATION_TIMEFRAME) * 60)
                continue

            slots_available = MAX_OPEN_TRADES - open_count
            for symbol in symbols_to_scan:
                if slots_available <= 0:
                    logger.info("ℹ️ [Main] All available slots filled. Ending scan for this cycle.")
                    break
                
                logger.debug(f"🔍 [Main] Scanning {symbol}...")
                try:
                    with conn.cursor() as symbol_cur:
                        symbol_cur.execute("SELECT 1 FROM signals WHERE symbol = %s AND achieved_target = FALSE LIMIT 1;", (symbol,))
                        if symbol_cur.fetchone():
                            continue

                    df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_hist is None or df_hist.empty:
                        continue

                    # Use the new Enhanced Strategy
                    strategy = EnhancedTradingStrategy(symbol)
                    if strategy.ml_model is None:
                        logger.debug(f"ℹ️ [Main] Skipping {symbol}, no ML model loaded.")
                        continue

                    df_indicators = strategy.populate_indicators(df_hist)
                    if df_indicators is None:
                        continue

                    potential_signal = strategy.generate_buy_signal(df_indicators)
                    if potential_signal:
                        # Final check before inserting
                        with conn.cursor() as final_check_cur:
                            final_check_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE;")
                            current_open_count = (final_check_cur.fetchone() or {}).get('count', 0)

                        if current_open_count < MAX_OPEN_TRADES:
                            # Here you would call insert_signal_into_db and send_telegram_alert
                            # insert_signal_into_db(potential_signal)
                            # send_telegram_alert(potential_signal, SIGNAL_GENERATION_TIMEFRAME)
                            logger.info(f"✅✅✅ SIGNAL FOUND AND VERIFIED FOR {symbol} ✅✅✅")
                            slots_available -= 1
                            time.sleep(2) # Stagger signal sending
                        else:
                            logger.warning(f"⚠️ [Main] Signal for {symbol} was found but trade limit was reached before insertion.")
                            break
                except (psycopg2.Error, Exception) as symbol_err:
                    logger.error(f"❌ [Main] Error processing symbol {symbol}: {symbol_err}", exc_info=True)
                    if conn: conn.rollback()
                    continue
                time.sleep(0.2)

            scan_duration = time.time() - scan_start_time
            wait_time = max(60, get_interval_minutes(SIGNAL_GENERATION_TIMEFRAME) * 60 - scan_duration)
            logger.info(f"🏁 [Main] Scan cycle finished in {scan_duration:.2f}s. Waiting {wait_time:.1f}s for next cycle.")
            time.sleep(wait_time)

        except KeyboardInterrupt:
            logger.info("🛑 [Main] Shutdown requested. Exiting...")
            break
        except Exception as main_err:
            logger.critical(f"❌ [Main] An unexpected error occurred in the main loop: {main_err}", exc_info=True)
            time.sleep(120)

# Placeholder for get_interval_minutes function, assuming it exists
def get_interval_minutes(interval: str) -> int:
    if interval.endswith('m'): return int(interval[:-1])
    if interval.endswith('h'): return int(interval[:-1]) * 60
    return 0

# --- Main Entry Point ---
if __name__ == "__main__":
    logger.info("🚀 Starting Enhanced Trading Bot...")
    init_db()
    
    # Start WebSocket in a daemon thread
    ws_thread = Thread(target=run_ticker_socket_manager, daemon=True, name="WebSocketThread")
    ws_thread.start()
    logger.info("✅ [Main] WebSocket manager started. Waiting 5s for initial data...")
    time.sleep(5)

    # Start the main bot loop
    main_loop()

    logger.info("👋 [Main] Bot has been shut down.")

