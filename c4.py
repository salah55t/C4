import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Tuple, Any, Union
# Note: psycopg2, Flask, ThreadedWebsocketManager, Binance API Client are removed/modified for backtesting
# We will simulate data fetching and trade execution

# ---------------------- Logging Setup ----------------------
logging.basicConfig(
    level=logging.INFO, # Keep INFO for general backtest progress, change to DEBUG for detailed candle-by-candle logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_backtest.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Backtester')

# ---------------------- Load Environment Variables (Only needed for API if fetching fresh data) ----------------------
# In a real backtest, you might load historical data from a file or database
# For this example, we'll simulate fetching data using a mock client
try:
    # API_KEY and API_SECRET might still be needed if fetching data from Binance API for backtest
    API_KEY: str = config('BINANCE_API_KEY', default='YOUR_BINANCE_API_KEY') # Use default if not set
    API_SECRET: str = config('BINANCE_API_SECRET', default='YOUR_BINANCE_API_SECRET') # Use default if not set
    # TELEGRAM_TOKEN and CHAT_ID are not needed for backtesting
    # DB_URL is not needed for backtesting results storage (we'll use a list/DataFrame)
except Exception as e:
     logger.warning(f"⚠️ Could not load environment variables. API keys might be needed for data fetching: {e}")
     # We will proceed, but data fetching might fail if keys are required by the mock client

# ---------------------- Constants and Global Variables (Copied/Modified from c4.py) ----------------------
TRADE_VALUE: float = 100.0         # Simulated trade value in USDT
MAX_OPEN_TRADES: int = 5          # Maximum number of simulated open trades simultaneously
SIGNAL_GENERATION_TIMEFRAME: str = '15m' # Timeframe for signal generation
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 7 # Historical data lookback for signal generation

# --- Confirmation Timeframe ---
CONFIRMATION_TIMEFRAME: str = '30m' # Larger timeframe for trend confirmation
CONFIRMATION_LOOKBACK_DAYS: int = 14 # Historical data lookback for confirmation timeframe

# --- Parameters for Improved Entry Point (Less Strict Version) ---
ENTRY_POINT_EMA_PROXIMITY_PCT: float = 0.003 # Price must be within this % of signal timeframe EMA_SHORT
ENTRY_POINT_RECENT_CANDLE_LOOKBACK: int = 2 # Look back this many candles on signal timeframe for bullish sign

# =============================================================================
# --- Indicator Parameters (Adjusted for 15m Signal and 30m Confirmation) ---
# =============================================================================
RSI_PERIOD: int = 14 # Standard RSI period
RSI_OVERSOLD: int = 30
RSI_OVERBOUGHT: int = 70
EMA_SHORT_PERIOD: int = 13 # Adjusted for 15m
EMA_LONG_PERIOD: int = 34 # Adjusted for 15m
VWMA_PERIOD: int = 21 # Adjusted for 15m
ENTRY_ATR_PERIOD: int = 14 # Adjusted for 15m
ENTRY_ATR_MULTIPLIER: float = 1.75 # ATR Multiplier for initial target
BOLLINGER_WINDOW: int = 20 # Standard Bollinger period
BOLLINGER_STD_DEV: int = 2 # Standard Bollinger std dev
MACD_FAST: int = 12 # Standard MACD fast period
MACD_SLOW: int = 26 # Standard MACD slow period
MACD_SIGNAL: int = 9 # Standard MACD signal period
ADX_PERIOD: int = 14 # Standard ADX period
SUPERTREND_PERIOD: int = 10 # Standard Supertrend period
SUPERTREND_MULTIPLIER: float = 3.0 # Adjusted Supertrend multiplier slightly

# --- Parameters for Dynamic Target Update ---
DYNAMIC_TARGET_APPROACH_PCT: float = 0.003 # Percentage proximity to target to trigger re-evaluation
DYNAMIC_TARGET_EXTENSION_ATR_MULTIPLIER: float = 1.0 # ATR multiplier for extending the target
MAX_DYNAMIC_TARGET_UPDATES: int = 3 # Maximum number of times a target can be dynamically updated for a single signal
MIN_ADX_FOR_DYNAMIC_UPDATE: int = 25 # Minimum ADX value to consider dynamic target update

# --- Less Strict Parameters ---
MIN_PROFIT_MARGIN_PCT: float = 1.0 # Minimum profit margin
MIN_VOLUME_15M_USDT: float = 400000.0 # Minimum volume check (using 15m data now)
RECENT_EMA_CROSS_LOOKBACK: int = 3 # Adjusted for 15m
MIN_ADX_TREND_STRENGTH: int = 20 # Minimum ADX trend strength for essential condition
MACD_HIST_INCREASE_CANDLES: int = 2 # Lookback for MACD Hist increase
OBV_INCREASE_CANDLES: int = 2 # Lookback for OBV increase

# --- Optional Condition Weights (Less Strict Version) ---
CONDITION_WEIGHTS = {
    'rsi_ok': 1.0,
    'bullish_candle': 2.0,
    'rsi_filter_breakout': 1.5,
    'macd_filter_breakout': 1.5,
    'macd_hist_increasing': 4.0,
}
TOTAL_POSSIBLE_SCORE = sum(CONDITION_WEIGHTS.values())
MIN_SCORE_THRESHOLD_PCT = 0.55 # Lowered threshold for less strictness
MIN_SIGNAL_SCORE = TOTAL_POSSIBLE_SCORE * MIN_SCORE_THRESHOLD_PCT
# =============================================================================

# Mock Binance Client and Data Fetcher for Backtesting
class MockBinanceClient:
    """A mock Binance client to simulate fetching historical data."""
    def __init__(self, data_path="historical_data"):
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)

    def get_historical_klines(self, symbol: str, interval: str, start_str: str, end_str: Optional[str] = None, limit: int = 1000):
        """Simulates fetching historical klines from a local file or generating mock data."""
        # In a real backtest, you would load pre-downloaded data here
        # For this example, we'll just return empty data or mock data structure
        logger.debug(f"Simulating fetching {interval} klines for {symbol} from {start_str} to {end_str}")
        # This is a placeholder. You would replace this with actual data loading logic.
        # Example: Load from a CSV file based on symbol, interval, and date range.
        # For demonstration, returning an empty list or a minimal structure.
        return [] # Replace with actual data loading

    def get_klines(self, symbol: str, interval: str, limit: int = 1):
        """Simulates fetching recent klines."""
         # This is a placeholder for backtesting. In a real backtest, you'd get this from your historical data.
         # For volume check, we might need the last candle's volume from the historical data slice.
         # This function is not ideal for backtesting volume checks.
         # We will adjust the volume check logic in the backtester.
        return [] # Replace with logic to get last kline from current backtest slice

# Use the mock client
mock_client = MockBinanceClient()

# Modified fetch_historical_data for backtesting
def fetch_historical_data_backtest(symbol: str, interval: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """Fetches historical data for backtesting from a source (e.g., CSV files)."""
    logger.info(f"ℹ️ [Data] Fetching historical data for {symbol} - {interval} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    # --- REPLACE THIS WITH YOUR ACTUAL DATA LOADING LOGIC ---
    # Example: Load from CSV files
    try:
        # Assuming your data is in CSV files named like 'BTCUSDT_15m.csv'
        file_path = os.path.join(mock_client.data_path, f"{symbol.replace('/', '')}_{interval}.csv")
        if not os.path.exists(file_path):
            logger.error(f"❌ [Data] Historical data file not found: {file_path}")
            return None

        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp']) # Assuming 'timestamp' column exists
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']] # Adjust columns as per your CSV
        df.dropna(inplace=True)

        # Filter data for the specified date range
        df_filtered = df.loc[start_date:end_date].copy()

        if df_filtered.empty:
            logger.warning(f"⚠️ [Data] No data found for {symbol} - {interval} in the specified date range after filtering.")
            return None

        logger.info(f"✅ [Data] Fetched {len(df_filtered)} data points for {symbol} - {interval}.")
        return df_filtered

    except Exception as e:
        logger.error(f"❌ [Data] Error fetching historical data for {symbol}: {e}", exc_info=True)
        return None
    # --- END OF REPLACEABLE DATA LOADING LOGIC ---


# ---------------------- Technical Indicator Functions (Copied from c4.py) ----------------------
# These functions remain mostly the same, they operate on pandas DataFrames
def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculates Exponential Moving Average (EMA)."""
    if series is None or series.isnull().all() or len(series) < span:
        # logger.debug(f"⚠️ [Indicators] Insufficient data to calculate EMA span={span}.") # Reduce logging in backtest indicators
        return pd.Series(index=series.index if series is not None else None, dtype=float)
    ema = series.ewm(span=span, adjust=False).mean()
    # logger.debug(f"✅ [Indicators] Calculated EMA span={span}.") # Reduce logging
    return ema

def calculate_vwma(df: pd.DataFrame, period: int) -> pd.Series:
    """Calculates Volume Weighted Moving Average (VWMA)."""
    df_calc = df.copy()
    if not all(col in df_calc.columns for col in ['close', 'volume']) or df_calc[['close', 'volume']].isnull().all().any() or len(df_calc) < period:
        # logger.debug(f"⚠️ [Indicators] Insufficient data to calculate VWMA period={period}.") # Reduce logging
        return pd.Series(index=df_calc.index, dtype=float)
    df_calc['price_volume'] = df_calc['close'] * df_calc['volume']
    rolling_price_volume_sum = df_calc['price_volume'].rolling(window=period, min_periods=period).sum()
    rolling_volume_sum = df_calc['volume'].rolling(window=period, min_periods=period).sum()
    vwma = rolling_price_volume_sum / rolling_volume_sum.replace(0, np.nan)
    # logger.debug(f"✅ [Indicators] Calculated VWMA period={period}.") # Reduce logging
    return vwma

def calculate_atr_indicator(df: pd.DataFrame, period: int = ENTRY_ATR_PERIOD) -> pd.DataFrame:
    """Calculates Average True Range (ATR)."""
    df = df.copy()
    if not all(col in df.columns for col in ['high', 'low', 'close']) or df[['high', 'low', 'close']].isnull().all().any() or len(df) < period + 1:
        # logger.debug(f"⚠️ [Indicators] Insufficient data to calculate ATR period={period}.") # Reduce logging
        df['atr'] = np.nan; return df
    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)
    df['atr'] = tr.ewm(span=period, adjust=False).mean()
    # logger.debug(f"✅ [Indicators] Calculated ATR period={period}.") # Reduce logging
    return df

def calculate_macd(df: pd.DataFrame, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> pd.DataFrame:
    """Calculates MACD indicator."""
    df = df.copy()
    min_len = max(fast, slow, signal)
    if 'close' not in df.columns or df['close'].isnull().all() or len(df) < min_len:
        # logger.debug(f"⚠️ [Indicators] Insufficient data to calculate MACD fast={fast}, slow={slow}, signal={signal}.") # Reduce logging
        df['macd'] = np.nan; df['macd_signal'] = np.nan; df['macd_hist'] = np.nan; return df
    ema_fast = calculate_ema(df['close'], fast)
    ema_slow = calculate_ema(df['close'], slow)
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = calculate_ema(df['macd'], signal)
    df['macd_hist'] = df['macd'] - df['macd_signal']
    # logger.debug(f"✅ [Indicators] Calculated MACD fast={fast}, slow={slow}, signal={signal}.") # Reduce logging
    return df

def calculate_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    """Calculates ADX indicator."""
    df_calc = df.copy() # Work on a copy
    if not all(col in df_calc.columns for col in ['high', 'low', 'close']) or df_calc[['high', 'low', 'close']].isnull().all().any() or len(df_calc) < period * 2:
        # logger.debug(f"⚠️ [Indicators] Insufficient data to calculate ADX period={period}.") # Reduce logging
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
    # logger.debug(f"✅ [Indicators] Calculated ADX period={period}.") # Reduce logging
    return df_calc[['adx', 'di_plus', 'di_minus']]

def calculate_supertrend(df: pd.DataFrame, period: int = SUPERTREND_PERIOD, multiplier: float = SUPERTREND_MULTIPLIER) -> pd.DataFrame:
    """Calculates Supertrend indicator."""
    df_st = df.copy()
    if not all(col in df_st.columns for col in ['high', 'low', 'close']) or df_st[['high', 'low', 'close']].isnull().all().any():
        # logger.debug(f"⚠️ [Indicators] Insufficient data to calculate Supertrend period={period}, multiplier={multiplier}.") # Reduce logging
        df_st['supertrend'] = np.nan; df_st['supertrend_trend'] = 0; return df_st
    df_st = calculate_atr_indicator(df_st, period=period) # Use Supertrend's own period for ATR
    if 'atr' not in df_st.columns or df_st['atr'].isnull().all() or len(df_st) < period:
        # logger.debug(f"⚠️ [Indicators] Insufficient ATR data to calculate Supertrend period={period}.") # Reduce logging
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
    # logger.debug(f"✅ [Indicators] Calculated Supertrend period={period}, multiplier={multiplier}.") # Reduce logging
    return df_st

# ---------------------- Candlestick Patterns (Copied from c4.py) ----------------------
def is_bullish_candle(row: pd.Series) -> bool:
    """Checks if the candle is bullish."""
    o, c = row.get('open'), row.get('close')
    return pd.notna(o) and pd.notna(c) and c > o

def is_hammer(row: pd.Series) -> int:
    """Checks if the candle is a Hammer pattern."""
    o, h, l, c = row.get('open'), row.get('high'), row.get('low'), row.get('close')
    if pd.isna([o, h, l, c]).any(): return 0
    body = abs(c - o)
    candle_range = h - l
    if candle_range == 0: return 0
    lower_shadow = min(o, c) - l
    upper_shadow = h - max(o, c)
    is_small_body = body < (candle_range * 0.35)
    is_long_lower_shadow = lower_shadow >= 1.8 * body if body > 0 else lower_shadow > candle_range * 0.6
    is_small_upper_shadow = upper_shadow <= body * 0.6 if body > 0 else upper_shadow < candle_range * 0.15
    return 100 if is_small_body and is_long_lower_shadow and is_small_upper_shadow else 0

def is_shooting_star(row: pd.Series) -> int:
    """Checks if the candle is a Shooting Star pattern."""
    o, h, l, c = row.get('open'), row.get('high'), row.get('low'), row.get('close')
    if pd.isna([o, h, l, c]).any(): return 0
    body = abs(c - o)
    candle_range = h - l
    if candle_range == 0: return 0
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
    # logger.debug("ℹ️ [Indicators] Detecting candlestick patterns...") # Reduce logging
    df['Hammer'] = df.apply(is_hammer, axis=1)
    df['ShootingStar'] = df.apply(is_shooting_star, axis=1)
    df['Engulfing'] = [compute_engulfing(df, i) for i in range(len(df))]
    df['BullishCandleSignal'] = df.apply(lambda row: 1 if (row['Hammer'] == 100 or row['Engulfing'] == 100) else 0, axis=1)
    df['BearishCandleSignal'] = df.apply(lambda row: 1 if (row['ShootingStar'] == -100 or row['Engulfing'] == -100) else 0, axis=1)
    # logger.debug("✅ [Indicators] Candlestick patterns detected.") # Reduce logging
    return df

# ---------------------- Other Helper Functions (Modified for Backtesting) ----------------------
# get_fear_greed_index and get_btc_trend_4h are not used in the core signal logic in the provided code,
# but they were in the alert message. For backtesting, we can either skip them or simulate.
# Let's skip them for simplicity in the backtest logic itself.

# Modified fetch_recent_volume for backtesting - get volume from the current candle in the backtest slice
def get_volume_for_backtest_candle(df: pd.DataFrame, current_index: int) -> float:
    """Gets the volume for the current candle in the backtest."""
    if current_index < 0 or current_index >= len(df):
        return 0.0
    return df.iloc[current_index].get('volume', 0.0)

def get_interval_minutes(interval: str) -> int:
    """Converts interval from string to minutes."""
    if interval.endswith('m'): return int(interval[:-1])
    elif interval.endswith('h'): return int(interval[:-1]) * 60
    elif interval.endswith('d'): return int(interval[:-1]) * 60 * 24
    return 0

# ---------------------- Trading Strategy (Modified for Backtesting) -------------------
class ScalpingTradingStrategyBacktest:
    def __init__(self, symbol: str):
        self.symbol = symbol
        # Required columns for signal timeframe (15m) indicators - Removed VWAP, BB, OBV
        self.required_cols_signal_indicators = [
            'open', 'high', 'low', 'close', 'volume',
            f'ema_{EMA_SHORT_PERIOD}', f'ema_{EMA_LONG_PERIOD}', 'vwma',
            'rsi', 'atr',
            'macd', 'macd_signal', 'macd_hist',
            'adx', 'di_plus', 'di_minus',
            'supertrend', 'supertrend_trend',
            'BullishCandleSignal', 'BearishCandleSignal'
        ]
        # Required columns for confirmation timeframe (30m) indicators
        self.required_cols_confirmation_indicators = [
             'open', 'high', 'low', 'close',
             f'ema_{EMA_SHORT_PERIOD}', f'ema_{EMA_LONG_PERIOD}',
             'macd', 'macd_signal', 'macd_hist',
             'adx', 'di_plus', 'di_minus',
             'supertrend', 'supertrend_trend'
        ]
        # Required columns for buy signal generation
        self.required_cols_buy_signal = [
            'close', f'ema_{EMA_SHORT_PERIOD}', f'ema_{EMA_LONG_PERIOD}', 'vwma',
            'rsi', 'atr', 'macd', 'macd_signal', 'macd_hist',
            'supertrend', 'supertrend_trend', 'adx', 'di_plus', 'di_minus',
            'BullishCandleSignal'
        ]
        # Optional Condition Weights (Less Strict Version)
        self.condition_weights = CONDITION_WEIGHTS
        # Essential conditions remain the same
        self.essential_conditions = [
            'price_above_emas_and_vwma', 'ema_short_above_ema_long',
            'supertrend_up', 'macd_positive_or_cross', 'adx_trending_bullish_strong',
        ]
        self.total_possible_score = TOTAL_POSSIBLE_SCORE
        self.min_score_threshold_pct = MIN_SCORE_THRESHOLD_PCT
        self.min_signal_score = self.total_possible_score * self.min_score_threshold_pct

    def populate_indicators(self, df: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
        """Populates indicators for a given dataframe and timeframe."""
        # logger.debug(f"ℹ️ [Strategy {self.symbol}] Calculating {timeframe} frame indicators...") # Reduce logging
        min_len_required = max(EMA_LONG_PERIOD, VWMA_PERIOD, RSI_PERIOD, ENTRY_ATR_PERIOD, BOLLINGER_WINDOW, MACD_SLOW, ADX_PERIOD*2, SUPERTREND_PERIOD) + 5
        if timeframe == CONFIRMATION_TIMEFRAME:
             min_len_required = max(EMA_LONG_PERIOD, MACD_SLOW, ADX_PERIOD*2, SUPERTREND_PERIOD) + 5

        if len(df) < min_len_required:
            # logger.debug(f"⚠️ [Strategy {self.symbol}] DataFrame {timeframe} too short ({len(df)} < {min_len_required}).") # Reduce logging
            return None

        try:
            df_calc = df.copy()
            df_calc[f'ema_{EMA_SHORT_PERIOD}'] = calculate_ema(df_calc['close'], EMA_SHORT_PERIOD)
            df_calc[f'ema_{EMA_LONG_PERIOD}'] = calculate_ema(df_calc['close'], EMA_LONG_PERIOD)
            df_calc = calculate_macd(df_calc, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            adx_df = calculate_adx(df_calc, ADX_PERIOD); df_calc = df_calc.join(adx_df)
            df_calc = calculate_supertrend(df_calc, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)

            if timeframe == SIGNAL_GENERATION_TIMEFRAME:
                df_calc = calculate_atr_indicator(df_calc, ENTRY_ATR_PERIOD)
                df_calc['vwma'] = calculate_vwma(df_calc, VWMA_PERIOD)
                df_calc = calculate_rsi_indicator(df_calc, RSI_PERIOD)
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

            df_cleaned = df_calc.dropna(subset=required_cols).copy()
            if df_cleaned.empty:
                # logger.debug(f"⚠️ [Strategy {self.symbol}] DataFrame {timeframe} empty after dropping NaN.") # Reduce logging
                return None

            # logger.debug(f"✅ [Strategy {self.symbol}] {timeframe} frame indicators calculated.") # Reduce logging
            return df_cleaned
        except Exception as e:
            logger.error(f"❌ [Strategy {self.symbol}] Error calculating {timeframe} indicators: {e}", exc_info=True)
            return None

    def check_confirmation_conditions(self, df_conf_processed: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Checks for bullish trend confirmation on the larger timeframe (30m) using provided data."""
        # logger.debug(f"ℹ️ [Strategy {self.symbol}] Checking confirmation conditions on {CONFIRMATION_TIMEFRAME} frame...") # Reduce logging
        confirmation_details = {}

        if df_conf_processed is None or df_conf_processed.empty:
            # logger.debug(f"⚠️ [Strategy {self.symbol}] Insufficient processed {CONFIRMATION_TIMEFRAME} data for confirmation.") # Reduce logging
            confirmation_details['Status'] = f"Failed: No processed {CONFIRMATION_TIMEFRAME} data"
            return False, confirmation_details

        last_row_conf = df_conf_processed.iloc[-1]

        # Confirmation Conditions: Price above EMAs, Supertrend up, MACD bullish, ADX trending
        price_above_emas_conf = (pd.notna(last_row_conf['close']) and
                                 pd.notna(last_row_conf[f'ema_{EMA_SHORT_PERIOD}']) and
                                 pd.notna(last_row_conf[f'ema_{EMA_LONG_PERIOD}']) and
                                 last_row_conf['close'] > last_row_conf[f'ema_{EMA_SHORT_PERIOD}'] and
                                 last_row_conf['close'] > last_row_conf[f'ema_{EMA_LONG_PERIOD}'])
        confirmation_details['Price_Above_EMAs_Conf'] = "Passed" if price_above_emas_conf else "Failed"

        supertrend_up_conf = (pd.notna(last_row_conf['supertrend_trend']) and last_row_conf['supertrend_trend'] == 1)
        confirmation_details['SuperTrend_Conf'] = "Passed" if supertrend_up_conf else "Failed"

        macd_bullish_conf = (pd.notna(last_row_conf['macd_hist']) and last_row_conf['macd_hist'] > 0)
        confirmation_details['MACD_Conf'] = "Passed" if macd_bullish_conf else "Failed"

        adx_trending_bullish_conf = (pd.notna(last_row_conf['adx']) and last_row_conf['adx'] > MIN_ADX_TREND_STRENGTH and
                                     pd.notna(last_row_conf['di_plus']) and pd.notna(last_row['di_minus']) and
                                     last_row_conf['di_plus'] > last_row_conf['di_minus'])
        confirmation_details['ADX_DI_Conf'] = "Passed" if adx_trending_bullish_conf else "Failed"

        all_confirmed = price_above_emas_conf and supertrend_up_conf and macd_bullish_conf and adx_trending_bullish_conf

        confirmation_details['Status'] = "Confirmed" if all_confirmed else "Confirmation Failed"
        # logger.debug(f"✅ [Strategy {self.symbol}] {CONFIRMATION_TIMEFRAME} frame confirmation status: {confirmation_details['Status']}") # Reduce logging

        return all_confirmed, confirmation_details

    def check_entry_point_quality(self, df_processed_signal: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Checks if the current price offers a good entry point on the signal timeframe (15m) using provided data."""
        # logger.debug(f"ℹ️ [Strategy {self.symbol}] Checking entry point quality on {SIGNAL_GENERATION_TIMEFRAME} frame...") # Reduce logging
        entry_point_details = {}

        if df_processed_signal is None or df_processed_signal.empty or len(df_processed_signal) < ENTRY_POINT_RECENT_CANDLE_LOOKBACK + 1:
            # logger.debug(f"⚠️ [Strategy {self.symbol}] Insufficient {SIGNAL_GENERATION_TIMEFRAME} data for entry point check.") # Reduce logging
            entry_point_details['Status'] = f"Failed: Insufficient {SIGNAL_GENERATION_TIMEFRAME} data"
            return False, entry_point_details

        last_row_signal = df_processed_signal.iloc[-1]
        recent_df_signal = df_processed_signal.iloc[-ENTRY_POINT_RECENT_CANDLE_LOOKBACK-1:]

        if recent_df_signal[['close', 'open', f'ema_{EMA_SHORT_PERIOD}']].isnull().values.any():
             # logger.debug(f"⚠️ [Strategy {self.symbol}] Recent {SIGNAL_GENERATION_TIMEFRAME} data contains NaN for entry point check.") # Reduce logging
             entry_point_details['Status'] = f"Failed: NaN in recent {SIGNAL_GENERATION_TIMEFRAME} data"
             return False, entry_point_details

        current_price = last_row_signal['close']
        ema_short_signal = last_row_signal[f'ema_{EMA_SHORT_PERIOD}']

        # Condition 1: Price is close to the signal timeframe EMA_SHORT
        price_near_ema_short = abs(current_price - ema_short_signal) / ema_short_signal <= ENTRY_POINT_EMA_PROXIMITY_PCT if ema_short_signal > 0 else False
        entry_point_details['Price_Near_EMA_Short_SignalTF'] = "Passed" if price_near_ema_short else "Failed"
        # logger.debug(f"ℹ️ [Strategy {self.symbol}] EMA {EMA_SHORT_PERIOD} ({SIGNAL_GENERATION_TIMEFRAME}) proximity check: Near: {price_near_ema_short}") # Reduce logging


        # Condition 2: Last candle is bullish or a hammer
        last_candle_bullish_or_hammer = is_bullish_candle(last_row_signal) or is_hammer(last_row_signal) == 100
        entry_point_details['Last_Candle_Bullish_or_Hammer_SignalTF'] = "Passed" if last_candle_bullish_or_hammer else "Failed"
        # logger.debug(f"ℹ️ [Strategy {self.symbol}] Last candle ({SIGNAL_GENERATION_TIMEFRAME}) check: Bullish or Hammer: {last_candle_bullish_or_hammer}") # Reduce logging


        # Combine conditions for a good entry point
        is_good_entry = price_near_ema_short and last_candle_bullish_or_hammer

        entry_point_details['Status'] = "Good Entry Point" if is_good_entry else "Entry Point Not Ideal"
        # logger.debug(f"✅ [Strategy {self.symbol}] Entry point quality status on {SIGNAL_GENERATION_TIMEFRAME} frame: {entry_point_details['Status']}") # Reduce logging

        return is_good_entry, entry_point_details


    def generate_buy_signal(self, df_signal_tf: pd.DataFrame, df_conf_tf: pd.DataFrame, current_index: int) -> Optional[Dict[str, Any]]:
        """Generates a buy signal based on strategy conditions using historical data slice."""
        # logger.debug(f"ℹ️ [Strategy {self.symbol}] Starting buy signal generation for index {current_index}...") # Reduce logging

        # Ensure enough data is available for lookback periods for both timeframes
        min_signal_data_len = max(RECENT_EMA_CROSS_LOOKBACK, MACD_HIST_INCREASE_CANDLES, OBV_INCREASE_CANDLES, ENTRY_POINT_RECENT_CANDLE_LOOKBACK) + 1
        min_conf_data_len = max(EMA_LONG_PERIOD, MACD_SLOW, ADX_PERIOD*2, SUPERTREND_PERIOD) + 5 # From populate_indicators

        if current_index < max(min_signal_data_len, min_conf_data_len) -1 :
             # logger.debug(f"ℹ️ [Strategy {self.symbol}] Not enough historical data yet at index {current_index} for signal generation.") # Reduce logging
             return None

        # Get the data slice up to the current candle (inclusive)
        df_processed_signal = self.populate_indicators(df_signal_tf.iloc[:current_index+1], SIGNAL_GENERATION_TIMEFRAME)
        if df_processed_signal is None or df_processed_signal.empty or len(df_processed_signal) < min_signal_data_len:
             # logger.debug(f"⚠️ [Strategy {self.symbol}] Processed signal DataFrame too short at index {current_index}."); # Reduce logging
             return None

        # Get corresponding data slice for confirmation timeframe
        # Find the timestamp of the current signal candle
        current_signal_timestamp = df_signal_tf.index[current_index]
        # Find the index of the corresponding candle in the confirmation timeframe data
        # This assumes confirmation data is aligned or contains the signal timeframe timestamps
        # A more robust approach might involve resampling or finding the closest candle
        conf_index_at_signal_time = df_conf_tf.index.searchsorted(current_signal_timestamp, side='right') - 1
        if conf_index_at_signal_time < min_conf_data_len - 1:
             # logger.debug(f"ℹ️ [Strategy {self.symbol}] Not enough historical data on confirmation timeframe at index {current_index} ({current_signal_timestamp}).") # Reduce logging
             return None

        df_processed_conf = self.populate_indicators(df_conf_tf.iloc[:conf_index_at_signal_time+1], CONFIRMATION_TIMEFRAME)
        if df_processed_conf is None or df_processed_conf.empty or len(df_processed_conf) < min_conf_data_len:
             # logger.debug(f"⚠️ [Strategy {self.symbol}] Processed confirmation DataFrame too short at index {conf_index_at_signal_time}."); # Reduce logging
             return None


        missing_cols_signal = [col for col in self.required_cols_buy_signal if col not in df_processed_signal.columns]
        if missing_cols_signal: logger.warning(f"⚠️ [Strategy {self.symbol}] Missing signal columns at index {current_index}: {missing_cols_signal}."); return None

        missing_cols_conf = [col for col in self.required_cols_confirmation_indicators if col not in df_processed_conf.columns]
        if missing_cols_conf: logger.warning(f"⚠️ [Strategy {self.symbol}] Missing confirmation columns at index {current_index}: {missing_cols_conf}."); return None


        # --- Step 1: Check Multi-Timeframe Confirmation (30m) ---
        is_confirmed_on_larger_tf, confirmation_details = self.check_confirmation_conditions(df_processed_conf)
        if not is_confirmed_on_larger_tf:
             # logger.debug(f"ℹ️ [Strategy {self.symbol}] Confirmation failed on {CONFIRMATION_TIMEFRAME} frame at index {current_index}. Cancelling signal.") # Reduce logging
             return None

        # --- Step 2: Check Entry Point Quality on Signal Timeframe (15m) ---
        is_good_entry_point, entry_point_details = self.check_entry_point_quality(df_processed_signal)
        if not is_good_entry_point:
             # logger.debug(f"ℹ️ [Strategy {self.symbol}] Entry point quality on {SIGNAL_GENERATION_TIMEFRAME} frame is not ideal at index {current_index}. Cancelling signal.") # Reduce logging
             return None

        # --- Step 3: Proceed with Signal Generation if Confirmed and Entry is Good ---
        # BTC trend check is skipped in backtest for simplicity unless you have BTC historical data

        last_row_signal = df_processed_signal.iloc[-1]
        recent_df_signal = df_processed_signal.iloc[-min_signal_data_len:]

        if recent_df_signal[self.required_cols_buy_signal].isnull().values.any():
            logger.warning(f"⚠️ [Strategy {self.symbol}] Recent signal data contains NaN in required columns at index {current_index}."); return None

        essential_passed = True; failed_essential_conditions = []; signal_details = {}
        # Mandatory Conditions Check (on 15m timeframe)
        if not (pd.notna(last_row_signal[f'ema_{EMA_SHORT_PERIOD}']) and pd.notna(last_row_signal[f'ema_{EMA_LONG_PERIOD}']) and pd.notna(last_row_signal['vwma']) and last_row_signal['close'] > last_row_signal[f'ema_{EMA_SHORT_PERIOD}'] and last_row_signal['close'] > last_row_signal[f'ema_{LONG_PERIOD}'] and last_row_signal['close'] > last_row_signal['vwma']):
            essential_passed = False; failed_essential_conditions.append('Price above MAs and VWMA'); signal_details['Price_MA_Alignment_SignalTF'] = 'Failed'
        else: signal_details['Price_MA_Alignment_SignalTF'] = 'Passed'

        if not (pd.notna(last_row_signal[f'ema_{EMA_SHORT_PERIOD}']) and pd.notna(last_row_signal[f'ema_{EMA_LONG_PERIOD}']) and last_row_signal[f'ema_{EMA_SHORT_PERIOD}'] > last_row_signal[f'ema_{EMA_LONG_PERIOD}']):
            essential_passed = False; failed_essential_conditions.append('Short EMA > Long EMA'); signal_details['EMA_Order_SignalTF'] = 'Failed'
        else: signal_details['EMA_Order_SignalTF'] = 'Passed'

        if not (pd.notna(last_row_signal['supertrend']) and last_row_signal['close'] > last_row_signal['supertrend'] and last_row_signal['supertrend_trend'] == 1):
            essential_passed = False; failed_essential_conditions.append('SuperTrend Uptrend'); signal_details['SuperTrend_SignalTF'] = 'Failed'
        else: signal_details['SuperTrend_SignalTF'] = 'Passed'

        if not (pd.notna(last_row_signal['macd_hist']) and (last_row_signal['macd_hist'] > 0 or (pd.notna(last_row_signal['macd']) and pd.notna(last_row_signal['macd_signal']) and last_row_signal['macd'] > last_row_signal['macd_signal']))):
            essential_passed = False; failed_essential_conditions.append('MACD Bullish'); signal_details['MACD_SignalTF'] = 'Failed'
        else: signal_details['MACD_SignalTF'] = 'Passed'

        if not (pd.notna(last_row_signal['adx']) and pd.notna(last_row_signal['di_plus']) and pd.notna(last_row_signal['di_minus']) and last_row_signal['adx'] > MIN_ADX_TREND_STRENGTH and last_row_signal['di_plus'] > last_row_signal['di_minus']):
            essential_passed = False; failed_essential_conditions.append(f'Strong Bullish ADX (>{MIN_ADX_TREND_STRENGTH})'); signal_details['ADX_DI_SignalTF'] = 'Failed'
        else: signal_details['ADX_DI_SignalTF'] = 'Passed'

        if not essential_passed:
            # logger.debug(f"ℹ️ [Strategy {self.symbol}] {SIGNAL_GENERATION_TIMEFRAME} essential conditions failed at index {current_index}: {', '.join(failed_essential_conditions)}."); # Reduce logging
            signal_details['Essential_Conditions_SignalTF_Status'] = 'Failed'
            signal_details['Failed_Essential_Conditions_SignalTF'] = failed_essential_conditions
            return None

        signal_details['Essential_Conditions_SignalTF_Status'] = 'Passed'
        current_score = 0.0 # Optional Conditions Scoring

        if pd.notna(last_row_signal['rsi']) and RSI_OVERSOLD < last_row_signal['rsi'] < RSI_OVERBOUGHT : current_score += self.condition_weights.get('rsi_ok', 0); signal_details['RSI_Basic_SignalTF'] = 'Passed'
        else: signal_details['RSI_Basic_SignalTF'] = 'Failed'

        if last_row_signal.get('BullishCandleSignal', 0) == 1: current_score += self.condition_weights.get('bullish_candle', 0); signal_details['Candle_SignalTF'] = 'Passed'
        else: signal_details['Candle_SignalTF'] = 'Failed'

        if pd.notna(last_row_signal['rsi']) and 50 <= last_row_signal['rsi'] <= 80: current_score += self.condition_weights.get('rsi_filter_breakout', 0); signal_details['RSI_Filter_Breakout_SignalTF'] = 'Passed'
        else: signal_details['RSI_Filter_Breakout_SignalTF'] = 'Failed'

        if pd.notna(last_row_signal['macd_hist']) and last_row_signal['macd_hist'] > 0: current_score += self.condition_weights.get('macd_filter_breakout', 0); signal_details['MACD_Filter_Breakout_SignalTF'] = 'Passed'
        else: signal_details['MACD_Filter_Breakout_SignalTF'] = 'Failed'

        if len(recent_df_signal) >= MACD_HIST_INCREASE_CANDLES + 1 and not recent_df_signal['macd_hist'].iloc[-MACD_HIST_INCREASE_CANDLES-1:].isnull().any() and np.all(np.diff(recent_df_signal['macd_hist'].iloc[-MACD_HIST_INCREASE_CANDLES-1:]) > 0): current_score += self.condition_weights.get('macd_hist_increasing', 0); signal_details['MACD_Hist_Increasing_SignalTF'] = 'Passed'
        else: signal_details['MACD_Hist_Increasing_SignalTF'] = 'Failed'


        if current_score < self.min_signal_score:
            # logger.debug(f"ℹ️ [Strategy {self.symbol}] Optional score too low at index {current_index} ({current_score:.2f} < {self.min_signal_score:.2f}). Cancelling signal."); # Reduce logging
            signal_details['Optional_Score_Status'] = 'Failed'
            return None

        signal_details['Optional_Score_Status'] = 'Passed'

        # Get volume from the current candle in the backtest slice
        volume_recent = last_row_signal.get('volume', 0.0) * last_row_signal.get('close', 0.0) # Approximate USDT volume
        if volume_recent < MIN_VOLUME_15M_USDT:
            # logger.debug(f"ℹ️ [Strategy {self.symbol}] Liquidity too low at index {current_index} ({volume_recent:,.0f} < {MIN_VOLUME_15M_USDT:,.0f}). Cancelling signal."); # Reduce logging
            signal_details['Liquidity_Check'] = 'Failed'
            return None

        signal_details['Liquidity_Check'] = 'Passed'

        current_price = last_row_signal['close']; current_atr = last_row_signal.get('atr')
        if pd.isna(current_atr) or current_atr <= 0:
            logger.warning(f"⚠️ [Strategy {self.symbol}] Invalid ATR ({current_atr}) at index {current_index}. Cancelling signal.");
            signal_details['ATR_Check'] = 'Failed'
            return None

        signal_details['ATR_Check'] = 'Passed'

        initial_target = current_price + (ENTRY_ATR_MULTIPLIER * current_atr)
        initial_stop_loss = 0.0 # Stop Loss is removed in the live script

        profit_margin_pct = ((initial_target / current_price) - 1) * 100 if current_price > 0 else 0
        if profit_margin_pct < MIN_PROFIT_MARGIN_PCT:
            # logger.debug(f"ℹ️ [Strategy {self.symbol}] Profit margin too low at index {current_index} ({profit_margin_pct:.2f}% < {MIN_PROFIT_MARGIN_PCT:.2f}%). Cancelling signal."); # Reduce logging
            signal_details['Profit_Margin_Check'] = 'Failed'
            return None

        signal_details['Profit_Margin_Check'] = 'Passed'

        # Include confirmation and entry point details in the signal details
        signal_details['Confirmation_Details'] = confirmation_details
        signal_details['Entry_Point_Details_SignalTF'] = entry_point_details


        signal_output = {
            'symbol': self.symbol,
            'entry_price': float(f"{current_price:.8g}"),
            'initial_target': float(f"{initial_target:.8g}"),
            'initial_stop_loss': initial_stop_loss,
            'current_target': float(f"{initial_target:.8g}"), # For backtest, current target is initially the same as initial
            'current_stop_loss': initial_stop_loss,
            'r2_score': float(f"{current_score:.2f}"),
            'strategy_name': 'Scalping_Momentum_Trend_MultiTF_EnhancedEntry_V2_LessStrict',
            'signal_details': signal_details,
            'volume_15m': volume_recent,
            'trade_value': TRADE_VALUE,
            'total_possible_score': float(f"{self.total_possible_score:.2f}"),
            'entry_time': df_signal_tf.index[current_index], # Record entry time
            'entry_candle_index': current_index # Record candle index for tracking
        }
        # logger.info(f"✨ [Strategy {self.symbol}] Potential buy signal generated at {signal_output['entry_time']}. Price: {current_price:.6f}, Score: {current_score:.2f}") # Reduce logging
        return signal_output

    # Dynamic target update logic is complex to simulate candle-by-candle accurately without lookahead.
    # For this basic backtest, we will not simulate dynamic target updates.
    # Trades will only close on hitting the initial target.


# ---------------------- Backtesting Engine ----------------------
class Backtester:
    def __init__(self, symbol: str, signal_interval: str, backtest_start: datetime, backtest_end: datetime):
        self.symbol = symbol
        self.signal_interval = signal_interval
        self.backtest_start = backtest_start
        self.backtest_end = backtest_end
        self.strategy = ScalpingTradingStrategyBacktest(symbol)

        self.historical_data_signal_tf: Optional[pd.DataFrame] = None
        self.historical_data_conf_tf: Optional[pd.DataFrame] = None

        self.open_trades: List[Dict[str, Any]] = []
        self.closed_trades: List[Dict[str, Any]] = []

        self.load_historical_data()

    def load_historical_data(self):
        """Loads historical data required for the backtest."""
        logger.info(f"⏳ Loading historical data for {self.symbol}...")

        # Determine the required start date for historical data based on lookback periods
        max_lookback_days = max(SIGNAL_GENERATION_LOOKBACK_DAYS, CONFIRMATION_LOOKBACK_DAYS)
        data_fetch_start_date = self.backtest_start - timedelta(days=max_lookback_days + 5) # Add buffer

        self.historical_data_signal_tf = fetch_historical_data_backtest(
            self.symbol, self.signal_interval, data_fetch_start_date, self.backtest_end
        )

        # Determine the corresponding confirmation timeframe interval
        conf_interval_minutes = get_interval_minutes(CONFIRMATION_TIMEFRAME)
        signal_interval_minutes = get_interval_minutes(self.signal_interval)
        if conf_interval_minutes % signal_interval_minutes != 0:
             logger.error(f"❌ Confirmation interval ({CONFIRMATION_TIMEFRAME}) is not a multiple of signal interval ({self.signal_interval}). Cannot perform multi-timeframe analysis.")
             self.historical_data_signal_tf = None # Invalidate data if intervals are incompatible
             return

        self.historical_data_conf_tf = fetch_historical_data_backtest(
            self.symbol, CONFIRMATION_TIMEFRAME, data_fetch_start_date, self.backtest_end
        )

        if self.historical_data_signal_tf is None or self.historical_data_conf_tf is None:
            logger.error("❌ Failed to load all required historical data. Backtest cannot proceed.")
            self.historical_data_signal_tf = None # Ensure both are None if loading failed

        logger.info("✅ Historical data loading complete.")


    def run_backtest(self):
        """Runs the backtesting simulation."""
        if self.historical_data_signal_tf is None or self.historical_data_conf_tf is None:
            logger.error("❌ Historical data not loaded. Cannot run backtest.")
            return

        logger.info(f"▶️ Starting backtest for {self.symbol} from {self.backtest_start.strftime('%Y-%m-%d')} to {self.backtest_end.strftime('%Y-%m-%d')}...")

        # Find the starting index for the backtest period within the loaded data
        backtest_start_index = self.historical_data_signal_tf.index.searchsorted(self.backtest_start, side='left')
        if backtest_start_index >= len(self.historical_data_signal_tf):
             logger.warning("⚠️ Backtest start date is beyond available historical data. Adjusting start date.")
             backtest_start_index = len(self.historical_data_signal_tf) - 1
             self.backtest_start = self.historical_data_signal_tf.index[backtest_start_index]


        total_candles = len(self.historical_data_signal_tf)
        logger.info(f"ℹ️ Total candles in data: {total_candles}. Starting simulation from index {backtest_start_index} ({self.historical_data_signal_tf.index[backtest_start_index].strftime('%Y-%m-%d %H:%M')})...")


        for i in range(backtest_start_index, total_candles):
            current_candle_time = self.historical_data_signal_tf.index[i]
            current_candle_close = self.historical_data_signal_tf.iloc[i]['close']
            current_candle_high = self.historical_data_signal_tf.iloc[i]['high']
            current_candle_low = self.historical_data_signal_tf.iloc[i]['low']

            # Skip if current candle is beyond backtest end date
            if current_candle_time > self.backtest_end:
                 logger.info(f"⏹️ Reached backtest end date {self.backtest_end.strftime('%Y-%m-%d')}. Stopping simulation.")
                 break

            # logger.debug(f"⏳ Simulating candle {i+1}/{total_candles} at {current_candle_time.strftime('%Y-%m-%d %H:%M')}...") # Use DEBUG for candle-by-candle detail

            # --- Track Open Trades ---
            trades_to_close = []
            for trade in self.open_trades:
                # Check if target is hit within the current candle's high/low range
                # Simplification: Assume target hit if high >= target
                if current_candle_high >= trade['current_target']:
                    trade['closing_price'] = trade['current_target'] # Assume exact target hit
                    trade['closed_at'] = current_candle_time # Assume hit at candle close for simplicity
                    trade['profit_percentage'] = ((trade['closing_price'] / trade['entry_price']) - 1) * 100
                    trade['achieved_target'] = True
                    trade['time_to_target_seconds'] = (trade['closed_at'] - trade['entry_time']).total_seconds()
                    self.closed_trades.append(trade)
                    trades_to_close.append(trade)
                    logger.debug(f"✅ Target hit for trade opened at {trade['entry_time'].strftime('%Y-%m-%d %H:%M')} at {current_candle_time.strftime('%Y-%m-%d %H:%M')}. Profit: {trade['profit_percentage']:.2f}%")

                # Note: Stop loss logic would go here if implemented.
                # elif current_candle_low <= trade['current_stop_loss']:
                #    ... simulate stop loss hit ...

            # Remove closed trades from open trades list
            for trade in trades_to_close:
                self.open_trades.remove(trade)

            # --- Generate New Signals ---
            if len(self.open_trades) < MAX_OPEN_TRADES:
                # Pass the data slices up to the current index for signal generation
                potential_signal = self.strategy.generate_buy_signal(
                    self.historical_data_signal_tf.iloc[:i+1],
                    self.historical_data_conf_tf, # Pass the full conf data, strategy will slice internally
                    i # Pass current index in signal timeframe
                )

                if potential_signal:
                    # Simulate opening a trade at the close of the signal candle
                    simulated_entry_price = self.historical_data_signal_tf.iloc[i]['close']
                    potential_signal['entry_price'] = simulated_entry_price # Use actual close price for entry
                    potential_signal['entry_time'] = current_candle_time # Use actual candle time for entry
                    potential_signal['current_target'] = simulated_entry_price + (ENTRY_ATR_MULTIPLIER * self.historical_data_signal_tf.iloc[i]['atr']) # Recalculate target based on actual entry price

                    self.open_trades.append(potential_signal)
                    logger.debug(f"✨ Signal generated and trade opened at {potential_signal['entry_time'].strftime('%Y-%m-%d %H:%M')}. Entry: {potential_signal['entry_price']:.6f}, Target: {potential_signal['current_target']:.6f}")


        # Close any remaining open trades at the end of the backtest period
        logger.info("Closing remaining open trades at the end of the backtest period.")
        for trade in self.open_trades:
            trade['closing_price'] = self.historical_data_signal_tf.iloc[-1]['close'] # Close at the last available price
            trade['closed_at'] = self.historical_data_signal_tf.index[-1]
            trade['profit_percentage'] = ((trade['closing_price'] / trade['entry_price']) - 1) * 100
            trade['achieved_target'] = False # Did not hit target within backtest period
            trade['time_to_target_seconds'] = None # Target not hit
            self.closed_trades.append(trade)
            logger.debug(f"Trade for {trade['symbol']} opened at {trade['entry_time'].strftime('%Y-%m-%d %H:%M')} closed at end of backtest. Profit: {trade['profit_percentage']:.2f}%")

        logger.info("🏁 Backtest simulation finished.")


    def generate_backtest_report(self):
        """Generates a summary report of the backtest results."""
        logger.info("📊 Generating backtest report...")

        if not self.closed_trades:
            report = "❌ No trades were closed during the backtest period."
            logger.warning(report)
            return report

        df_results = pd.DataFrame(self.closed_trades)

        total_trades = len(df_results)
        target_hits = df_results['achieved_target'].sum()
        # Since no stop loss, all others are considered "timed out" or closed at end
        timed_out_trades = total_trades - target_hits

        winning_trades = df_results[df_results['profit_percentage'] > 0]
        losing_trades = df_results[df_results['profit_percentage'] <= 0] # Includes break-even

        win_rate = (target_hits / total_trades) * 100 if total_trades > 0 else 0.0
        total_profit_pct = df_results['profit_percentage'].sum()
        total_profit_usd = (total_profit_pct / 100.0) * TRADE_VALUE # Assuming each trade was of TRADE_VALUE

        avg_profit_per_trade = df_results['profit_percentage'].mean() if total_trades > 0 else 0.0
        avg_win_pct = winning_trades['profit_percentage'].mean() if not winning_trades.empty else 0.0
        avg_loss_pct = losing_trades['profit_percentage'].mean() if not losing_trades.empty else 0.0

        gross_profit_pct_sum = winning_trades['profit_percentage'].sum()
        gross_loss_pct_sum = losing_trades['profit_percentage'].sum()

        profit_factor = abs(gross_profit_pct_sum / gross_loss_pct_sum) if gross_loss_pct_sum != 0 else float('inf')

        avg_time_to_target_seconds = df_results[df_results['achieved_target'] == True]['time_to_target_seconds'].mean()
        avg_time_to_target_formatted = format_duration(int(avg_time_to_target_seconds)) if pd.notna(avg_time_to_target_seconds) else "N/A"


        report = (
            f"📊 *Backtest Report for {self.symbol} ({self.signal_interval})*\n"
            f"🗓️ *Period:* {self.backtest_start.strftime('%Y-%m-%d')} to {self.backtest_end.strftime('%Y-%m-%d')}\n"
            f"_(Simulated Trade Value: ${TRADE_VALUE:,.2f})_\n"
            f"——————————————\n"
            f"📈 Total Trades: *{total_trades}*\n"
            f"🎯 Target Hits: *{target_hits}*\n"
            f"⏱️ Timed Out/Closed at End: *{timed_out_trades}*\n"
            f"✅ Winning Trades (Profit > 0%): *{len(winning_trades)}*\n"
            f"❌ Losing Trades (Profit <= 0%): *{len(losing_trades)}*\n"
            f"——————————————\n"
            f"🏆 Win Rate (based on Target Hits): *{win_rate:.2f}%*\n"
            f"💰 Total Profit (%): *{total_profit_pct:+.2f}%*\n"
            f"💵 Total Profit (USD): *${total_profit_usd:+.2f}*\n"
            f"——————————————\n"
            f"📈 Average Profit per Trade: *{avg_profit_per_trade:+.2f}%*\n"
            f"📊 Average Winning Trade: *{avg_win_pct:+.2f}%*\n"
            f"📉 Average Losing Trade: *{avg_loss_pct:+.2f}%*\n"
            f"Factor الربح: *{'∞' if profit_factor == float('inf') else f'{profit_factor:.2f}'}*\n"
            f"⏳ Average Time to Target: *{avg_time_to_target_formatted}*\n"
            f"——————————————\n"
            f"ℹ️ *Note: This backtest simulates target hits only, as per the current live script logic. No stop loss is simulated.*\n"
            f"⏰ _Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        )
        logger.info("✅ Backtest report generated.")
        print("\n" + report) # Print report to console
        return report

# ---------------------- Main Execution Block ----------------------
if __name__ == "__main__":
    logger.info("🚀 Starting Backtesting Script...")

    # --- Configure Backtest Parameters ---
    SYMBOL_TO_BACKTEST = "BTCUSDT" # Choose the symbol
    SIGNAL_TIMEFRAME_TO_BACKTEST = "15m" # Choose the signal timeframe
    BACKTEST_START_DATE = datetime(2023, 1, 1) # Start date of backtest period
    BACKTEST_END_DATE = datetime(2023, 12, 31) # End date of backtest period
    # -------------------------------------

    # Ensure you have historical data files in the 'historical_data' directory
    # For example, 'historical_data/BTCUSDT_15m.csv' and 'historical_data/BTCUSDT_30m.csv'
    # These files should have columns: 'timestamp', 'open', 'high', 'low', 'close', 'volume'

    backtester = Backtester(SYMBOL_TO_BACKTEST, SIGNAL_TIMEFRAME_TO_BACKTEST, BACKTEST_START_DATE, BACKTEST_END_DATE)
    backtester.run_backtest()
    backtester.generate_backtest_report()

    logger.info("🏁 Backtesting Script finished.")
