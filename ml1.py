import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
import warnings
import gc
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from binance.client import Client
from datetime import datetime, timedelta, timezone
from decouple import config
from typing import List, Dict, Optional, Any, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from flask import Flask
from threading import Thread

# --- Keras and TensorFlow Imports for GRU Model ---
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


# ---------------------- تجاهل التحذيرات المستقبلية من Pandas ----------------------
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
# optuna.logging.set_verbosity(optuna.logging.WARNING) # Optuna is removed
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_model_trainer_gru_v1.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLTrainer_GRU_V1_Microstructure')
# Set TensorFlow logging to ERROR to avoid excessive logs
tf.get_logger().setLevel('ERROR')


# ---------------------- تحميل متغيرات البيئة ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
    TELEGRAM_TOKEN: Optional[str] = config('TELEGRAM_BOT_TOKEN', default=None)
    CHAT_ID: Optional[str] = config('TELEGRAM_CHAT_ID', default=None)
except Exception as e:
     logger.critical(f"❌ فشل في تحميل المتغيرات البيئية الأساسية: {e}")
     exit(1)

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
BASE_ML_MODEL_NAME: str = 'GRU_Scalping_V1_With_Microstructure' # Changed model name
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 60
BTC_SYMBOL = 'BTCUSDT'

# --- GRU Model Parameters ---
TIMESTEPS: int = 30 # Number of past periods to use for prediction
EPOCHS: int = 50
BATCH_SIZE: int = 64

# --- Indicator & Feature Parameters ---
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
EMA_SLOW_PERIOD: int = 200
EMA_FAST_PERIOD: int = 50
BTC_CORR_PERIOD: int = 30
REL_VOL_PERIOD: int = 30
MOMENTUM_PERIOD: int = 12
EMA_SLOPE_PERIOD: int = 5

# Triple-Barrier Method Parameters
TP_ATR_MULTIPLIER: float = 2.0
SL_ATR_MULTIPLIER: float = 1.5
MAX_HOLD_PERIOD: int = 24

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
btc_data_cache: Optional[pd.DataFrame] = None

# --- دوال الاتصال والتحقق ---
def init_db():
    global conn
    try:
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ml_models (
                    id SERIAL PRIMARY KEY, model_name TEXT NOT NULL UNIQUE,
                    model_data BYTEA NOT NULL, trained_at TIMESTAMP DEFAULT NOW(), metrics JSONB );
            """)
        conn.commit()
        logger.info("✅ [DB] تم تهيئة قاعدة البيانات بنجاح.")
    except Exception as e:
        logger.critical(f"❌ [DB] فشل الاتصال بقاعدة البيانات: {e}"); exit(1)

def keep_db_alive():
    if not conn: return
    try:
        with conn.cursor() as cur: cur.execute("SELECT 1;")
        logger.debug("[DB Keep-Alive] Ping successful.")
    except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
        logger.error(f"❌ [DB Keep-Alive] انقطع اتصال قاعدة البيانات: {e}. محاولة إعادة الاتصال...")
        if conn: conn.close()
        init_db()
    except Exception as e:
        logger.error(f"❌ [DB Keep-Alive] خطأ غير متوقع أثناء فحص الاتصال: {e}")
        if conn: conn.rollback()

def get_trained_symbols_from_db() -> set:
    if not conn: return set()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT model_name FROM ml_models WHERE model_name LIKE %s;", (f"{BASE_ML_MODEL_NAME}_%",))
            trained_models = cur.fetchall()
            prefix_to_remove = f"{BASE_ML_MODEL_NAME}_"
            trained_symbols = {row['model_name'].replace(prefix_to_remove, '') for row in trained_models if row['model_name'].startswith(prefix_to_remove)}
            logger.info(f"✅ [DB Check] تم العثور على {len(trained_symbols)} نموذج مدرب مسبقاً في قاعدة البيانات.")
            return trained_symbols
    except Exception as e:
        logger.error(f"❌ [DB Check] لا يمكن جلب الرموز المدربة من قاعدة البيانات: {e}")
        if conn: conn.rollback()
        return set()

def get_binance_client():
    global client
    try:
        client = Client(API_KEY, API_SECRET)
        client.ping()
        logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
    except Exception as e:
        logger.critical(f"❌ [Binance] فشل تهيئة عميل Binance: {e}"); exit(1)

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    if not client: return []
    try:
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            symbols = {s.strip().upper() for s in f if s.strip() and not s.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in symbols}
        info = client.get_exchange_info()
        active = {s['symbol'] for s in info['symbols'] if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT'}
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ [Validation] تم العثور على {len(validated)} عملة صالحة للتداول.")
        return validated
    except FileNotFoundError:
        logger.error(f"❌ [Validation] ملف قائمة العملات '{filename}' غير موجود.")
        return []
    except Exception as e:
        logger.error(f"❌ [Validation] خطأ في التحقق من الرموز: {e}"); return []

# --- دالة تحسين استهلاك الذاكرة ---
def optimize_memory_usage(df: pd.DataFrame, log_prefix: str = "") -> pd.DataFrame:
    """تقليل استهلاك الذاكرة للبيانات عن طريق تحويل أنواع البيانات."""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    end_mem = df.memory_usage().sum() / 1024**2
    if start_mem > end_mem:
         logger.info(f"🧠 [{log_prefix}] Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction).")
    return df

# --- دوال جلب ومعالجة البيانات ---
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    try:
        start_dt = datetime.now(timezone.utc) - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(klines, columns=cols)
        
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base']
        df = df[required_cols]
        
        numeric_cols = {
            'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 
            'volume': 'float', 'quote_volume': 'float', 'taker_buy_base': 'float'
        }
        df = df.astype(numeric_cols)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df.dropna(inplace=True)
        
        return optimize_memory_usage(df, log_prefix=f"Fetch {symbol} {interval}")
    except Exception as e:
        logger.error(f"❌ [Data] خطأ أثناء جلب البيانات لـ {symbol} على إطار {interval}: {e}"); return None

def fetch_and_cache_btc_data():
    global btc_data_cache
    logger.info("ℹ️ [BTC Data] جاري جلب بيانات البيتكوين وتخزينها...")
    btc_data_cache = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
    if btc_data_cache is None:
        logger.critical("❌ [BTC Data] فشل جلب بيانات البيتكوين."); exit(1)
    btc_data_cache['btc_returns'] = btc_data_cache['close'].pct_change()

# --- دوال حساب الميزات ---

def calculate_advanced_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    highest_high = df['high'].rolling(window=14).max()
    lowest_low = df['low'].rolling(window=14).min()
    df['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low).replace(0, 1e-9)
    df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low).replace(0, 1e-9)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    bb_period = 20
    df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
    bb_std = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, 1e-9)
    df['kc_middle'] = df['close'].ewm(span=20, adjust=False).mean()
    if 'atr' in df.columns:
        df['kc_upper'] = df['kc_middle'] + (df['atr'] * 1.5)
        df['kc_lower'] = df['kc_middle'] - (df['atr'] * 1.5)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    money_ratio = positive_flow / negative_flow.replace(0, 1e-9)
    df['mfi'] = 100 - (100 / (1 + money_ratio))
    return df

def calculate_market_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ['taker_buy_base', 'volume', 'quote_volume', 'high', 'low', 'open', 'close']
    if not all(col in df.columns for col in required_cols):
        logger.warning("⚠️ [Microstructure] Missing required columns for microstructure features. Skipping.")
        return df
    df['buy_pressure'] = df['taker_buy_base'] / df['volume'].replace(0, 1e-9)
    volume_ma = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / volume_ma.replace(0, 1e-9)
    df['price_impact'] = df['quote_volume'] / df['volume'].replace(0, 1e-9)
    log_hl = np.log(df['high'] / df['low'].replace(0, 1e-9))
    log_co = np.log(df['close'] / df['open'].replace(0, 1e-9))
    gk_vol_sq = (0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)).clip(lower=0)
    df['garman_klass_vol'] = np.sqrt(gk_vol_sq)
    log_hc = np.log(df['high'] / df['close'].replace(0, 1e-9))
    log_ho = np.log(df['high'] / df['open'].replace(0, 1e-9))
    log_lc = np.log(df['low'] / df['close'].replace(0, 1e-9))
    log_lo = np.log(df['low'] / df['open'].replace(0, 1e-9))
    rs_vol_sq = (log_hc * log_ho + log_lc * log_lo).clip(lower=0)
    df['rogers_satchell_vol'] = np.sqrt(rs_vol_sq)
    return df

def calculate_advanced_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    high_low = df['high'] - df['low']
    ema_high_low = high_low.ewm(span=10, adjust=False).mean()
    ema_high_low_shifted = ema_high_low.shift(10)
    df['chaikin_volatility'] = (ema_high_low - ema_high_low_shifted) / ema_high_low_shifted.replace(0, 1e-9) * 100
    period = 14
    max_close = df['close'].rolling(window=period).max()
    percentage_drawdown = 100 * (df['close'] - max_close) / max_close.replace(0, 1e-9)
    df['ulcer_index'] = np.sqrt((percentage_drawdown ** 2).rolling(window=period).mean())
    if 'atr' not in df.columns: return df
    high_low_tr = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift()).abs()
    low_close_prev = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low_tr, high_close_prev, low_close_prev], axis=1).max(axis=1)
    for p in [5, 10, 20]:
        atr_p = tr.ewm(span=p, adjust=False).mean()
        df[f'atr_ratio_{p}'] = df['atr'] / atr_p.replace(0, 1e-9)
    return df

def calculate_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['asia_session'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
    df['london_session'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
    df['ny_session'] = ((df.index.hour >= 13) & (df.index.hour < 21)).astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    return df

def calculate_all_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """
    دالة محسنة وشاملة لحساب جميع الميزات مع التقليل من استهلاك الذاكرة.
    """
    logger.info("ℹ️ [Features] Calculating all features in a unified function...")
    df_calc = df.copy()

    # --- 1. Standard Features (Optimized) ---
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    up_move = df_calc['high'].diff()
    down_move = -df_calc['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)
    df_calc['price_vs_ema50'] = (df_calc['close'] / df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()) - 1
    asset_returns = df_calc['close'].pct_change()
    merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
    df_calc['btc_correlation'] = asset_returns.rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    
    # --- 2. Advanced & Other Features (Chained) ---
    df_calc = calculate_advanced_momentum_features(df_calc)
    df_calc = calculate_market_microstructure_features(df_calc)
    df_calc = calculate_advanced_volatility_features(df_calc)
    df_calc = calculate_temporal_features(df_calc)

    # --- 3. Basic Momentum ---
    df_calc[f'roc_{MOMENTUM_PERIOD}'] = (df_calc['close'] / df_calc['close'].shift(MOMENTUM_PERIOD) - 1) * 100
    df_calc['roc_acceleration'] = df_calc[f'roc_{MOMENTUM_PERIOD}'].diff()
    ema_slope = df_calc['close'].ewm(span=EMA_SLOPE_PERIOD, adjust=False).mean()
    df_calc[f'ema_slope_{EMA_SLOPE_PERIOD}'] = (ema_slope - ema_slope.shift(1)) / ema_slope.shift(1).replace(0, 1e-9) * 100

    # --- 4. Cleanup and Finalization ---
    del high_low, high_close, low_close, tr, up_move, down_move, plus_dm, minus_dm, plus_di, minus_di, dx, delta, gain, loss, asset_returns, merged_df, ema_slope
    gc.collect()
    
    logger.info("✅ [Features] All features calculated successfully.")
    return optimize_memory_usage(df_calc, log_prefix="All Features")

# --- دوال إعداد البيانات والتدريب ---

def get_triple_barrier_labels(prices: pd.Series, atr: pd.Series) -> pd.Series:
    labels = pd.Series(0, index=prices.index)
    for i in tqdm(range(len(prices) - MAX_HOLD_PERIOD), desc="Labeling", leave=False):
        entry_price = prices.iloc[i]
        current_atr = atr.iloc[i]
        if pd.isna(current_atr) or current_atr == 0: continue
        upper_barrier = entry_price + (current_atr * TP_ATR_MULTIPLIER)
        lower_barrier = entry_price - (current_atr * SL_ATR_MULTIPLIER)
        for j in range(1, MAX_HOLD_PERIOD + 1):
            if i + j >= len(prices): break
            if prices.iloc[i + j] >= upper_barrier:
                labels.iloc[i] = 1; break
            if prices.iloc[i + j] <= lower_barrier:
                labels.iloc[i] = -1; break
    return labels

def prepare_data_for_ml(df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    logger.info(f"ℹ️ [ML Prep] Preparing data for {symbol}...")
    
    # --- 1. Feature Engineering (Unified Call) ---
    df_featured = calculate_all_features(df_15m, btc_df)
    
    # --- 2. MTF Features ---
    delta_4h = df_4h['close'].diff()
    gain_4h = delta_4h.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss_4h = -delta_4h.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_4h['rsi_4h'] = 100 - (100 / (1 + (gain_4h / loss_4h.replace(0, 1e-9))))
    ema_fast_4h = df_4h['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    df_4h['price_vs_ema50_4h'] = (df_4h['close'] / ema_fast_4h) - 1
    
    mtf_features = df_4h[['rsi_4h', 'price_vs_ema50_4h']]
    df_featured = df_featured.join(mtf_features)
    df_featured[['rsi_4h', 'price_vs_ema50_4h']] = df_featured[['rsi_4h', 'price_vs_ema50_4h']].fillna(method='ffill')
    df_featured = optimize_memory_usage(df_featured, log_prefix="After MTF")
    
    # --- 3. Target Labeling ---
    df_featured['target'] = get_triple_barrier_labels(df_featured['close'], df_featured['atr'])
    
    # --- 4. Feature List and Cleaning ---
    feature_columns = [
        'rsi', 'adx', 'atr', 'relative_volume', 'price_vs_ema50', 'price_vs_ema200', 'btc_correlation',
        'rsi_4h', 'price_vs_ema50_4h',
        f'roc_{MOMENTUM_PERIOD}', 'roc_acceleration', f'ema_slope_{EMA_SLOPE_PERIOD}',
        'williams_r', 'stoch_k', 'stoch_d', 'macd', 'macd_signal', 'macd_histogram', 'bb_position', 'mfi',
        'buy_pressure', 'volume_ratio', 'price_impact', 'garman_klass_vol', 'rogers_satchell_vol',
        'chaikin_volatility', 'ulcer_index', 'atr_ratio_5', 'atr_ratio_10', 'atr_ratio_20',
        'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend', 'asia_session', 'london_session', 'ny_session', 'month_sin', 'month_cos'
    ]
    
    df_cleaned = df_featured.dropna(subset=feature_columns + ['target']).copy()
    df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cleaned.dropna(subset=feature_columns, inplace=True)

    # Map target from {-1, 0, 1} to {0, 1, 2} for Keras to_categorical
    # -1 (Sell) -> 0
    #  0 (Hold) -> 1
    #  1 (Buy)  -> 2
    df_cleaned['target'] = df_cleaned['target'].map({-1: 0, 0: 1, 1: 2})

    if df_cleaned.empty or df_cleaned['target'].nunique() < 2:
        logger.warning(f"⚠️ [ML Prep] Data for {symbol} has less than 2 classes after cleaning. Skipping.")
        return None
        
    logger.info(f"📊 [ML Prep] Target distribution for {symbol} (0=Sell, 1=Hold, 2=Buy):\n{df_cleaned['target'].value_counts(normalize=True)}")
    X = df_cleaned[feature_columns]
    y = df_cleaned['target']
    return X, y, feature_columns

def create_sequences(X: np.ndarray, y: np.ndarray, time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences for time series forecasting.
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def build_gru_model(input_shape: tuple, num_classes: int) -> tf.keras.Model:
    """
    Builds and compiles the GRU model.
    """
    model = Sequential([
        Input(shape=input_shape),
        GRU(128, return_sequences=True),
        Dropout(0.3),
        GRU(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_gru_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    logger.info(f"🧠 [GRU Train] Starting GRU model training and evaluation...")
    
    n_features = X.shape[1]
    
    # Walk-forward validation for metrics
    all_preds_final, all_true_final = [], []
    tscv_final = TimeSeriesSplit(n_splits=5)
    
    for train_index, test_index in tscv_final.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create sequences
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, TIMESTEPS)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, TIMESTEPS)
        
        if X_train_seq.shape[0] == 0 or X_test_seq.shape[0] == 0:
            logger.warning("⚠️ [GRU Train] Not enough data in split to create sequences. Skipping split.")
            continue
            
        # One-hot encode targets
        y_train_cat = to_categorical(y_train_seq, num_classes=3)
        y_test_cat = to_categorical(y_test_seq, num_classes=3)
        
        # Build and train model for this fold
        model = build_gru_model(input_shape=(TIMESTEPS, n_features), num_classes=3)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        model.fit(X_train_seq, y_train_cat,
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  validation_split=0.1,
                  callbacks=[early_stopping],
                  verbose=0) # Set to 1 for detailed logs per epoch
        
        # Predict and evaluate
        y_pred_prob = model.predict(X_test_seq)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        all_preds_final.extend(y_pred)
        all_true_final.extend(y_test_seq)
        
        # Clean up memory
        del model, X_train_seq, y_train_seq, X_test_seq, y_test_seq, y_train_cat, y_test_cat
        tf.keras.backend.clear_session()
        gc.collect()

    if not all_true_final:
        logger.error("❌ [GRU Train] Walk-forward validation failed to produce any results.")
        return None, None, None

    # --- Final Model Training on All Data ---
    logger.info("ℹ️ [GRU Train] Retraining final model on all available data...")
    final_scaler = StandardScaler()
    X_scaled_full = final_scaler.fit_transform(X)
    X_seq_full, y_seq_full = create_sequences(X_scaled_full, y.values, TIMESTEPS)
    y_cat_full = to_categorical(y_seq_full, num_classes=3)
    
    final_model = build_gru_model(input_shape=(TIMESTEPS, n_features), num_classes=3)
    final_model.fit(X_seq_full, y_cat_full, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    
    # Calculate final metrics from walk-forward validation
    # Note: Class labels are {0: Sell, 1: Hold, 2: Buy}
    final_report = classification_report(all_true_final, all_preds_final, output_dict=True, zero_division=0, labels=[0, 1, 2], target_names=['Sell', 'Hold', 'Buy'])
    
    final_metrics = {
        'accuracy': accuracy_score(all_true_final, all_preds_final),
        'precision_class_2_buy': final_report.get('Buy', {}).get('precision', 0),
        'recall_class_2_buy': final_report.get('Buy', {}).get('recall', 0),
        'f1_score_class_2_buy': final_report.get('Buy', {}).get('f1-score', 0),
        'precision_class_0_sell': final_report.get('Sell', {}).get('precision', 0),
        'num_samples_trained': len(X),
    }
    
    metrics_log_str = f"Accuracy: {final_metrics['accuracy']:.4f}, P(Buy): {final_metrics['precision_class_2_buy']:.4f}, R(Buy): {final_metrics['recall_class_2_buy']:.4f}"
    logger.info(f"📊 [GRU Train] Final Walk-Forward Performance: {metrics_log_str}")

    return final_model, final_scaler, final_metrics


def save_ml_model_to_db(model_bundle: Dict[str, Any], model_name: str, metrics: Dict[str, Any]):
    logger.info(f"ℹ️ [DB Save] Saving model bundle '{model_name}'...")
    try:
        # For Keras, we serialize the architecture and weights separately
        # The bundle should contain model_json and model_weights
        pickled_bundle = pickle.dumps(model_bundle)
        metrics_json = json.dumps(metrics)
        
        with conn.cursor() as db_cur:
            db_cur.execute("""
                INSERT INTO ml_models (model_name, model_data, trained_at, metrics) 
                VALUES (%s, %s, NOW(), %s) ON CONFLICT (model_name) DO UPDATE SET 
                model_data = EXCLUDED.model_data, trained_at = NOW(), metrics = EXCLUDED.metrics;
            """, (model_name, pickled_bundle, metrics_json))
        conn.commit()
        logger.info(f"✅ [DB Save] Model bundle '{model_name}' saved successfully.")
    except Exception as e:
        logger.error(f"❌ [DB Save] Error saving model bundle: {e}"); conn.rollback()

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try: requests.post(url, json={'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

def run_training_job():
    logger.info(f"🚀 Starting GRU ML model training job ({BASE_ML_MODEL_NAME})...")
    init_db()
    get_binance_client()
    fetch_and_cache_btc_data()
    
    all_valid_symbols = get_validated_symbols(filename='crypto_list.txt')
    if not all_valid_symbols:
        logger.critical("❌ [Main] لم يتم العثور على رموز صالحة. سيتم الخروج."); return
    
    trained_symbols = get_trained_symbols_from_db()
    symbols_to_train = [s for s in all_valid_symbols if s not in trained_symbols]
    
    if not symbols_to_train:
        logger.info("✅ [Main] جميع الرموز مدربة بالفعل ومحدثة.");
        if conn: conn.close()
        return

    logger.info(f"ℹ️ [Main] Total: {len(all_valid_symbols)}. Trained: {len(trained_symbols)}. To Train: {len(symbols_to_train)}.")
    send_telegram_message(f"🚀 *{BASE_ML_MODEL_NAME} Training Started*\nWill train models for {len(symbols_to_train)} new symbols.")
    
    successful_models, failed_models = 0, 0
    for symbol in symbols_to_train:
        logger.info(f"\n--- ⏳ [Main] بدء تدريب النموذج لـ {symbol} ---")
        try:
            df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
            df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
            
            if df_15m is None or df_15m.empty or df_4h is None or df_4h.empty:
                logger.warning(f"⚠️ [Main] لا توجد بيانات كافية لـ {symbol}, سيتم التجاوز."); failed_models += 1; continue
            
            prepared_data = prepare_data_for_ml(df_15m, df_4h, btc_data_cache, symbol)
            del df_15m, df_4h; gc.collect()

            if prepared_data is None:
                failed_models += 1; continue
            X, y, feature_names = prepared_data
            
            training_result = train_gru_model(X, y)
            if not all(training_result):
                 logger.warning(f"⚠️ [Main] فشل تدريب النموذج لـ {symbol}."); failed_models += 1
                 del X, y, prepared_data; gc.collect()
                 continue
            final_model, final_scaler, model_metrics = training_result
            
            # We check for precision of the "Buy" signal (class 2)
            if final_model and final_scaler and model_metrics.get('precision_class_2_buy', 0) > 0.35:
                # Create the bundle for saving
                model_bundle = {
                    'model_json': final_model.to_json(),
                    'model_weights': final_model.get_weights(), # get_weights returns a list of numpy arrays
                    'scaler': final_scaler, 
                    'feature_names': feature_names,
                    'timesteps': TIMESTEPS
                }
                model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
                save_ml_model_to_db(model_bundle, model_name, model_metrics)
                successful_models += 1
            else:
                logger.warning(f"⚠️ [Main] النموذج الخاص بـ {symbol} غير مفيد (Buy Precision < 0.35). سيتم تجاهله."); failed_models += 1
            
            del X, y, prepared_data, training_result, final_model, final_scaler, model_metrics; 
            tf.keras.backend.clear_session()
            gc.collect()

        except Exception as e:
            logger.critical(f"❌ [Main] حدث خطأ فادح للرمز {symbol}: {e}", exc_info=True); failed_models += 1
            tf.keras.backend.clear_session()
            gc.collect()

        keep_db_alive()
        time.sleep(1)

    completion_message = (f"✅ *{BASE_ML_MODEL_NAME} Training Finished*\n"
                        f"- Successfully trained: {successful_models} new models\n"
                        f"- Failed/Discarded: {failed_models} models\n"
                        f"- Processed this run: {len(symbols_to_train)}")
    send_telegram_message(completion_message)
    logger.info(completion_message)

    if conn: conn.close()
    logger.info("👋 [Main] انتهت مهمة تدريب النماذج.")

app = Flask(__name__)

@app.route('/')
def health_check():
    return "ML GRU Trainer (with Microstructure features) service is running and healthy.", 200

if __name__ == "__main__":
    # Note: Keras/TensorFlow can have issues with multithreading on some systems.
    # Running directly might be more stable if issues arise.
    training_thread = Thread(target=run_training_job)
    training_thread.daemon = True
    training_thread.start()
    
    port = int(os.environ.get("PORT", 10001))
    logger.info(f"🌍 Starting web server on port {port} to keep the service alive...")
    app.run(host='0.0.0.0', port=port)
