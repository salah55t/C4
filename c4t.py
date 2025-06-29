import os
import gc
import pickle
import logging
import warnings
import pandas as pd
import numpy as np
import psycopg2
from decouple import config
from binance.client import Client
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta, timezone
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from tqdm import tqdm

# --- Ignore non-critical warnings ---
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# ---------------------- Logging Setup ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtester_v6_with_sr.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Backtester_V6_With_SR')

# ---------------------- Environment & Constants Setup ----------------------
try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
except Exception as e:
    logger.critical(f"❌ Critical failure loading environment variables: {e}")
    exit(1)

# --- Backtesting Parameters ---
INITIAL_CASH = 100000.0  # Initial portfolio value for simulation
TRADE_AMOUNT_USDT = 10.0 # Fixed amount for each trade
FEE = 0.001  # 0.1% Binance spot trading fee
SLIPPAGE = 0.0005 # 0.05% simulated slippage on trades
COMMISSION = FEE + SLIPPAGE # Combined commission for backtesting.py
BACKTEST_PERIOD_DAYS = 90 # Historical data period for backtesting

# --- Strategy & Model Constants (Must match the bot and trainer) ---
BASE_ML_MODEL_NAME = 'LightGBM_Scalping_V6_With_SR'
SIGNAL_GENERATION_TIMEFRAME = '15m'
HIGHER_TIMEFRAME = '4h'
BTC_SYMBOL = 'BTCUSDT'
MODEL_CONFIDENCE_THRESHOLD = 0.70 # Signal confidence threshold
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.0

# --- Indicator Parameters ---
ADX_PERIOD, BBANDS_PERIOD, RSI_PERIOD = 14, 20, 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD, EMA_SLOW_PERIOD, EMA_FAST_PERIOD = 14, 200, 50
BTC_CORR_PERIOD, STOCH_RSI_PERIOD, STOCH_K, STOCH_D, REL_VOL_PERIOD = 30, 14, 3, 3, 30
RSI_OVERBOUGHT, RSI_OVERSOLD = 70, 30
STOCH_RSI_OVERBOUGHT, STOCH_RSI_OVERSOLD = 80, 20

# Global connection objects
conn = None
client = None

# ---------------------- Database Functions ----------------------
def init_db():
    global conn
    try:
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        logger.info("✅ [DB] Database connection initialized successfully.")
    except Exception as e:
        logger.critical(f"❌ [DB] Failed to connect to the database: {e}")
        exit(1)

def load_ml_model_bundle_from_db(symbol: str) -> dict | None:
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if not conn:
        logger.error("[DB] Database connection is not available.")
        return None
    try:
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT model_data FROM ml_models WHERE model_name = %s ORDER BY trained_at DESC LIMIT 1;", (model_name,))
            result = db_cur.fetchone()
            if result and result['model_data']:
                model_bundle = pickle.loads(result['model_data'])
                logger.info(f"✅ [ML Model] Loaded '{model_name}' for {symbol} from DB.")
                return model_bundle
        logger.warning(f"⚠️ [ML Model] Model '{model_name}' not found in DB for {symbol}.")
        return None
    except Exception as e:
        logger.error(f"❌ [ML Model] Error loading model bundle for {symbol}: {e}")
        return None

def fetch_sr_levels_from_db(symbol: str) -> pd.DataFrame:
    if not conn: return pd.DataFrame()
    query = "SELECT level_price, level_type, score FROM support_resistance_levels WHERE symbol = %s"
    try:
        df = pd.read_sql(query, conn, params=(symbol,))
        if not df.empty:
            logger.info(f"✅ [S/R Levels] Fetched {len(df)} S/R levels for {symbol} from DB.")
        return df
    except Exception as e:
        logger.error(f"❌ [S/R Levels] Could not fetch S/R levels for {symbol}: {e}")
        return pd.DataFrame()

# ---------------------- Data Fetching & Preparation ----------------------
def fetch_historical_data(symbol: str, interval: str, days: int) -> pd.DataFrame | None:
    if not client: return None
    try:
        start_dt = datetime.now(timezone.utc) - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        numeric_cols = {'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'float32'}
        df = df.astype(numeric_cols)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        # --- RENAME COLUMNS FOR BACKTESTING.PY ---
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        return df.dropna()
    except Exception as e:
        logger.error(f"❌ [Data] Error fetching historical data for {symbol}: {e}")
        return None

# ---------------------- Feature Engineering Functions (Copied from Bot/Trainer) ----------------------
def calculate_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    op, hi, lo, cl = df['Open'], df['High'], df['Low'], df['Close']
    body = abs(cl - op); candle_range = hi - lo
    candle_range[candle_range == 0] = 1e-9
    upper_wick = hi - pd.concat([op, cl], axis=1).max(axis=1)
    lower_wick = pd.concat([op, cl], axis=1).min(axis=1) - lo
    df['candlestick_pattern'] = 0
    is_bullish_engulfing = (cl.shift(1) < op.shift(1)) & (cl > op) & (cl >= op.shift(1)) & (op <= cl.shift(1)) & (body > body.shift(1))
    is_bearish_engulfing = (cl.shift(1) > op.shift(1)) & (cl < op) & (op >= cl.shift(1)) & (cl <= op.shift(1)) & (body > body.shift(1))
    df.loc[is_bullish_engulfing, 'candlestick_pattern'] = 1
    df.loc[is_bearish_engulfing, 'candlestick_pattern'] = -1
    return df

def calculate_sr_features(df: pd.DataFrame, sr_levels_df: pd.DataFrame) -> pd.DataFrame:
    if sr_levels_df.empty:
        df['dist_to_support'] = 0.0; df['dist_to_resistance'] = 0.0
        df['score_of_support'] = 0.0; df['score_of_resistance'] = 0.0
        return df
    supports = sr_levels_df[sr_levels_df['level_type'].str.contains('support|poc|confluence', case=False)]['level_price'].sort_values().to_numpy()
    resistances = sr_levels_df[sr_levels_df['level_type'].str.contains('resistance|poc|confluence', case=False)]['level_price'].sort_values().to_numpy()
    support_scores = pd.Series(sr_levels_df['score'].values, index=sr_levels_df['level_price']).to_dict()
    def get_sr_info(price):
        dist_support, score_support, dist_resistance, score_resistance = 1.0, 0.0, 1.0, 0.0
        if supports.size > 0:
            idx = np.searchsorted(supports, price, side='right') - 1
            if idx >= 0:
                nearest_support = supports[idx]
                dist_support = (price - nearest_support) / price if price > 0 else 0
                score_support = support_scores.get(nearest_support, 0)
        if resistances.size > 0:
            idx = np.searchsorted(resistances, price, side='left')
            if idx < len(resistances):
                nearest_resistance = resistances[idx]
                dist_resistance = (nearest_resistance - price) / price if price > 0 else 0
                score_resistance = support_scores.get(nearest_resistance, 0)
        return dist_support, score_support, dist_resistance, score_resistance
    results = df['Close'].apply(get_sr_info)
    df[['dist_to_support', 'score_of_support', 'dist_to_resistance', 'score_of_resistance']] = pd.DataFrame(results.tolist(), index=df.index)
    return df

def create_all_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    high_low = df_calc['High'] - df_calc['Low']; high_close = (df_calc['High'] - df_calc['Close'].shift()).abs(); low_close = (df_calc['Low'] - df_calc['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    up_move = df_calc['High'].diff(); down_move = -df_calc['Low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr']
    minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr']
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    delta = df_calc['Close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    ema_fast = df_calc['Close'].ewm(span=MACD_FAST, adjust=False).mean(); ema_slow = df_calc['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow; signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df_calc['macd_hist'] = macd_line - signal_line
    df_calc['macd_cross'] = 0
    df_calc.loc[(df_calc['macd_hist'].shift(1) < 0) & (df_calc['macd_hist'] >= 0), 'macd_cross'] = 1
    df_calc.loc[(df_calc['macd_hist'].shift(1) > 0) & (df_calc['macd_hist'] <= 0), 'macd_cross'] = -1
    sma = df_calc['Close'].rolling(window=BBANDS_PERIOD).mean(); std_dev = df_calc['Close'].rolling(window=BBANDS_PERIOD).std()
    upper_band = sma + (std_dev * 2); lower_band = sma - (std_dev * 2)
    df_calc['bb_width'] = (upper_band - lower_band) / (sma + 1e-9)
    rsi_val = df_calc['rsi']
    min_rsi = rsi_val.rolling(window=STOCH_RSI_PERIOD).min(); max_rsi = rsi_val.rolling(window=STOCH_RSI_PERIOD).max()
    stoch_rsi_val = (rsi_val - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(window=STOCH_K).mean() * 100
    df_calc['stoch_rsi_d'] = df_calc['stoch_rsi_k'].rolling(window=STOCH_D).mean()
    df_calc['relative_volume'] = df_calc['Volume'] / (df_calc['Volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)
    df_calc['market_condition'] = 0
    df_calc.loc[(df_calc['rsi'] > RSI_OVERBOUGHT) | (df_calc['stoch_rsi_k'] > STOCH_RSI_OVERBOUGHT), 'market_condition'] = 1
    df_calc.loc[(df_calc['rsi'] < RSI_OVERSOLD) | (df_calc['stoch_rsi_k'] < STOCH_RSI_OVERSOLD), 'market_condition'] = -1
    ema_fast_trend = df_calc['Close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    ema_slow_trend = df_calc['Close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    df_calc['price_vs_ema50'] = (df_calc['Close'] / ema_fast_trend) - 1
    df_calc['price_vs_ema200'] = (df_calc['Close'] / ema_slow_trend) - 1
    df_calc['returns'] = df_calc['Close'].pct_change()
    merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
    df_calc['btc_correlation'] = merged_df['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    df_calc['hour_of_day'] = df_calc.index.hour
    df_calc = calculate_candlestick_patterns(df_calc)
    return df_calc

# ---------------------- Backtesting.py Strategy Class ----------------------
class MLStrategy(Strategy):
    # Pass strategy-specific parameters here
    ml_model = None
    scaler = None
    feature_names = None

    def init(self):
        # init is called once before the backtest loop
        # We pre-calculate indicators, so not much is needed here.
        # The data passed to Backtest() will already have all feature columns.
        pass

    def next(self):
        # next is called for each candle in the historical data
        
        # If a position is already open, do nothing. SL/TP are handled by the broker.
        if self.position:
            return

        # Get the feature values for the current candle
        # Note: self.data contains the pre-computed features
        try:
            features = self.data.df.loc[self.data.index[-1], self.feature_names]
            if features.isnull().any():
                return # Skip if data is missing
        except (KeyError, IndexError):
            return # Skip if row or columns are missing

        # Reshape for the scaler and model
        features_df = pd.DataFrame([features])
        features_scaled = self.scaler.transform(features_df)
        
        # Get model prediction and probability
        prediction = self.ml_model.predict(features_scaled)[0]
        prediction_proba = self.ml_model.predict_proba(features_scaled)[0]
        
        try:
            # Find the probability of the "buy" class (1)
            class_1_index = list(self.ml_model.classes_).index(1)
            prob_for_class_1 = prediction_proba[class_1_index]
        except ValueError:
            return # Class '1' not found in model

        # --- TRADING LOGIC ---
        if prediction == 1 and prob_for_class_1 >= MODEL_CONFIDENCE_THRESHOLD:
            # Get current ATR for dynamic SL/TP
            current_atr = self.data.atr[-1]
            if pd.isna(current_atr) or current_atr == 0:
                return # Can't set SL/TP without ATR

            current_price = self.data.Close[-1]
            
            # Define Stop Loss and Take Profit levels
            stop_loss_price = current_price - (current_atr * ATR_SL_MULTIPLIER)
            take_profit_price = current_price + (current_atr * ATR_TP_MULTIPLIER)

            # Calculate position size to be $10
            size = TRADE_AMOUNT_USDT / current_price

            # Place the buy order with SL and TP
            self.buy(size=size, sl=stop_loss_price, tp=take_profit_price)

# ---------------------- Main Execution Block ----------------------
def run_backtest():
    global client, conn
    logger.info("🚀 Starting Advanced Backtest for V6 Strategy...")
    
    # Initialize connections
    init_db()
    client = Client(API_KEY, API_SECRET)
    
    # Get symbols to test
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'crypto_list.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            symbols_to_test = [line.strip().upper() + "USDT" for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        logger.error("❌ 'crypto_list.txt' not found. Exiting.")
        return

    logger.info("ℹ️ Fetching global BTC data for the backtest period...")
    btc_df_full = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, BACKTEST_PERIOD_DAYS + 10)
    if btc_df_full is None:
        logger.critical("❌ Failed to fetch BTC data. Cannot proceed."); return
    btc_df_full['btc_returns'] = btc_df_full['Close'].pct_change()

    all_stats = []
    
    for symbol in tqdm(symbols_to_test, desc="Backtesting Symbols"):
        logger.info(f"\n--- ⏳ Processing Symbol: {symbol} ---")
        
        # 1. Load ML Model for the symbol
        model_bundle = load_ml_model_bundle_from_db(symbol)
        if not model_bundle:
            logger.warning(f"⚠️ Skipping {symbol}: No model found.")
            continue
        
        # 2. Fetch all required data
        df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, BACKTEST_PERIOD_DAYS)
        df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, BACKTEST_PERIOD_DAYS * 5) # Fetch more 4h data for EMA
        
        if df_15m is None or df_15m.empty or df_4h is None or df_4h.empty:
            logger.warning(f"⚠️ Skipping {symbol}: Insufficient historical data.")
            continue
            
        sr_levels = fetch_sr_levels_from_db(symbol)

        # 3. Prepare the master DataFrame with all features
        logger.info(f"Engineering features for {symbol}...")
        
        # Create base features
        data = create_all_features(df_15m, btc_df_full)
        
        # Create and merge MTF features
        delta_4h = df_4h['Close'].diff()
        gain_4h = delta_4h.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
        loss_4h = -delta_4h.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
        df_4h['rsi_4h'] = 100 - (100 / (1 + (gain_4h / loss_4h.replace(0, 1e-9))))
        ema_fast_4h = df_4h['Close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
        df_4h['price_vs_ema50_4h'] = (df_4h['Close'] / ema_fast_4h) - 1
        mtf_features = df_4h[['rsi_4h', 'price_vs_ema50_4h']]
        data = data.join(mtf_features, how='left').fillna(method='ffill')

        # Create and merge S/R features
        data = calculate_sr_features(data, sr_levels)
        
        data.dropna(inplace=True)
        
        if data.empty:
            logger.warning(f"⚠️ Skipping {symbol}: DataFrame is empty after feature engineering.")
            continue

        # 4. Run the backtest for the symbol
        logger.info(f"Running backtest for {symbol}...")
        bt = Backtest(
            data,
            MLStrategy,
            cash=INITIAL_CASH,
            commission=COMMISSION, # Includes fee + slippage
            exclusive_orders=True # Prevent multiple orders at the same time
        )
        
        # Pass the loaded model and scaler to the strategy class
        stats = bt.run(
            ml_model=model_bundle['model'],
            scaler=model_bundle['scaler'],
            feature_names=model_bundle['feature_names']
        )
        
        logger.info(f"\n--- Backtest Results for {symbol} ---")
        print(stats)
        all_stats.append(stats)
        
        # Optional: Plot the results
        # Uncomment the line below to generate an HTML plot for each symbol
        # bt.plot(filename=f"backtest_plot_{symbol}.html", open_browser=False)

        del data, df_15m, df_4h, sr_levels, model_bundle
        gc.collect()

    # 5. Display final aggregated results
    logger.info("\n\n--- 🏁 Overall Backtest Summary 🏁 ---")
    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        print(summary_df)
        
        # Calculate and print aggregate metrics
        total_trades = summary_df['# Trades'].sum()
        total_profit = summary_df['Equity Final [$]'].sum() - summary_df['Equity Start [$]'].sum()
        avg_win_rate = summary_df['Win Rate [%]'].mean()
        avg_profit_factor = summary_df['Profit Factor'].mean()
        
        print("\n--- Aggregated Metrics ---")
        print(f"Total Symbols Tested: {len(summary_df)}")
        print(f"Total Number of Trades: {total_trades}")
        print(f"Total Net Profit/Loss: ${total_profit:,.2f}")
        print(f"Average Win Rate: {avg_win_rate:.2f}%")
        print(f"Average Profit Factor: {avg_profit_factor:.2f}")
    else:
        print("No backtests were successfully completed.")
        
    if conn:
        conn.close()

if __name__ == "__main__":
    run_backtest()
