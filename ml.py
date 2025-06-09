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
from typing import List, Dict, Optional, Any, Tuple

# Import ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier # Using a more robust model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ---------------------- إعداد التسجيل ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',\
    handlers=[
        logging.FileHandler('ml_trainer_enhanced.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLTrainerEnhanced')

# ---------------------- تحميل المتغيرات البيئية ----------------------
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
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 120 # Using more data for better training
BASE_ML_MODEL_NAME: str = 'DecisionTree_Scalping_V2_Enhanced' # Updated model name

# Indicator Parameters (matching c4.py)
RSI_PERIOD: int = 14
VOLUME_LOOKBACK_CANDLES: int = 2
RSI_MOMENTUM_LOOKBACK_CANDLES: int = 3
ENTRY_ATR_PERIOD: int = 14
SUPERTREND_PERIOD: int = 10
SUPERTREND_MULTIPLIER: float = 3.0
MACD_FAST_PERIOD: int = 12
MACD_SLOW_PERIOD: int = 26
MACD_SIGNAL_PERIOD: int = 9
BB_PERIOD: int = 20
BB_STD_DEV: int = 2
ADX_PERIOD: int = 14

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
cur: Optional[psycopg2.extensions.cursor] = None
client: Optional[Client] = None

# ---------------------- Binance Client & DB Setup (Reusing functions) ----------------------
# (Assuming init_db, check_db_connection, fetch_historical_data, and indicator functions are defined here as in c4.py)

# --- Indicator Functions (Copied from c4.py for consistency) ---
def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi_indicator(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)
    return df

def calculate_atr_indicator(df: pd.DataFrame, period: int = ENTRY_ATR_PERIOD) -> pd.DataFrame:
    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)
    df['atr'] = tr.ewm(span=period, adjust=False).mean()
    return df

def calculate_supertrend(df: pd.DataFrame, period: int = SUPERTREND_PERIOD, multiplier: float = SUPERTREND_MULTIPLIER) -> pd.DataFrame:
    if 'atr' not in df.columns: df = calculate_atr_indicator(df, period)
    hl2 = (df['high'] + df['low']) / 2
    df['upper_band'] = hl2 + (multiplier * df['atr'])
    df['lower_band'] = hl2 - (multiplier * df['atr'])
    df['in_uptrend'] = True
    for current in range(1, len(df.index)):
        previous = current - 1
        if df['close'].iloc[current] > df['upper_band'].iloc[previous]: df.loc[df.index[current], 'in_uptrend'] = True
        elif df['close'].iloc[current] < df['lower_band'].iloc[previous]: df.loc[df.index[current], 'in_uptrend'] = False
        else:
            df.loc[df.index[current], 'in_uptrend'] = df['in_uptrend'].iloc[previous]
            if df['in_uptrend'].iloc[current] and df['lower_band'].iloc[current] < df['lower_band'].iloc[previous]: df.loc[df.index[current], 'lower_band'] = df['lower_band'].iloc[previous]
            if not df['in_uptrend'].iloc[current] and df['upper_band'].iloc[current] > df['upper_band'].iloc[previous]: df.loc[df.index[current], 'upper_band'] = df['upper_band'].iloc[previous]
    df['supertrend_direction'] = np.where(df['in_uptrend'], 1, -1)
    df.drop(['upper_band', 'lower_band', 'in_uptrend'], axis=1, inplace=True, errors='ignore')
    return df

def calculate_macd(df: pd.DataFrame, fast_period: int = MACD_FAST_PERIOD, slow_period: int = MACD_SLOW_PERIOD, signal_period: int = MACD_SIGNAL_PERIOD) -> pd.DataFrame:
    df['ema_fast'] = calculate_ema(df['close'], span=fast_period)
    df['ema_slow'] = calculate_ema(df['close'], span=slow_period)
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = calculate_ema(df['macd'], span=signal_period)
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df.drop(['ema_fast', 'ema_slow'], axis=1, inplace=True)
    return df

def calculate_bollinger_bands(df: pd.DataFrame, period: int = BB_PERIOD, std_dev: int = BB_STD_DEV) -> pd.DataFrame:
    df['bb_ma'] = df['close'].rolling(window=period).mean()
    df['bb_std'] = df['close'].rolling(window=period).std()
    df['bb_upper'] = df['bb_ma'] + (df['bb_std'] * std_dev)
    df['bb_lower'] = df['bb_ma'] - (df['bb_std'] * std_dev)
    df.drop(['bb_ma', 'bb_std'], axis=1, inplace=True, errors='ignore')
    return df
    
def calculate_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    if 'atr' not in df.columns:
        df = calculate_atr_indicator(df, period)
    
    df['plus_dm'] = df['high'].diff()
    df['minus_dm'] = df['low'].diff().mul(-1)

    df['plus_dm'] = np.where((df['plus_dm'] > df['minus_dm']) & (df['plus_dm'] > 0), df['plus_dm'], 0)
    df['minus_dm'] = np.where((df['minus_dm'] > df['plus_dm']) & (df['minus_dm'] > 0), df['minus_dm'], 0)

    df['plus_di'] = 100 * (df['plus_dm'].ewm(alpha=1/period).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].ewm(alpha=1/period).mean() / df['atr'])
    
    df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']).replace(0, 1))
    df['adx'] = df['dx'].ewm(alpha=1/period).mean()
    
    df.drop(['plus_dm', 'minus_dm', 'plus_di', 'minus_di', 'dx'], axis=1, inplace=True, errors='ignore')
    return df


def _calculate_btc_trend_feature(df_btc: pd.DataFrame) -> Optional[pd.Series]:
    min_data_for_ema = 55
    if df_btc is None or len(df_btc) < min_data_for_ema: return None
    ema20 = calculate_ema(df_btc['close'], 20)
    ema50 = calculate_ema(df_btc['close'], 50)
    trend_series = pd.Series(index=ema20.index, data=0.0)
    trend_series[(df_btc['close'] > ema20) & (ema20 > ema50)] = 1.0
    trend_series[(df_btc['close'] < ema20) & (ema20 < ema50)] = -1.0
    return trend_series.reindex(df_btc.index).fillna(0.0)

# --- DB and Client Setup ---
# Assume these functions are copied from c4.py or are available
def init_db(): pass
def check_db_connection(): pass
def fetch_historical_data(symbol, interval, days): pass
def get_crypto_symbols(filename='crypto_list.txt'): pass
def save_ml_model_to_db(model, model_name, metrics): pass
def send_telegram_message(chat_id, text, **kwargs): pass
def convert_np_values(obj): pass

# ---------------------- ML Data Preparation (ENHANCED) ----------------------
def prepare_data_for_ml(df: pd.DataFrame, symbol: str, target_period: int = 8, profit_threshold: float = 0.015) -> Optional[pd.DataFrame]:
    """
    Prepares data for ML training with enhanced features.
    """
    logger.info(f"ℹ️ [ML Prep] Preparing ENHANCED data for {symbol}...")
    try:
        df_calc = df.copy()

        # Calculate all indicators
        df_calc = calculate_rsi_indicator(df_calc)
        df_calc = calculate_atr_indicator(df_calc)
        df_calc = calculate_supertrend(df_calc)
        df_calc = calculate_macd(df_calc)
        df_calc = calculate_bollinger_bands(df_calc)
        df_calc = calculate_adx(df_calc)

        # Feature Engineering
        df_calc['volume_15m_avg'] = df_calc['volume'].rolling(window=VOLUME_LOOKBACK_CANDLES).mean()
        df_calc['rsi_momentum_bullish'] = ((df_calc['rsi'].diff(RSI_MOMENTUM_LOOKBACK_CANDLES) > 0) & (df_calc['rsi'] > 50)).astype(int)
        df_calc['bb_upper_dist'] = (df_calc['bb_upper'] - df_calc['close']) / df_calc['close']
        df_calc['bb_lower_dist'] = (df_calc['close'] - df_calc['bb_lower']) / df_calc['close']

        # Fetch and merge BTC trend feature
        btc_df = fetch_historical_data("BTCUSDT", interval=SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING)
        if btc_df is not None:
            btc_trend_series = _calculate_btc_trend_feature(btc_df)
            df_calc = df_calc.merge(btc_trend_series.rename('btc_trend_feature'), left_index=True, right_index=True, how='left')
            df_calc['btc_trend_feature'].fillna(0.0, inplace=True)
        else:
            df_calc['btc_trend_feature'] = 0.0

        # Define the target: Did the price reach the profit threshold within the target period?
        df_calc['future_high'] = df_calc['high'].shift(-target_period).rolling(window=target_period).max()
        df_calc['target'] = (df_calc['future_high'] >= df_calc['close'] * (1 + profit_threshold)).astype(int)
        
        feature_columns = [
            'volume_15m_avg', 'rsi_momentum_bullish', 'btc_trend_feature', 
            'supertrend_direction', 'macd_hist', 'bb_upper_dist', 'bb_lower_dist', 'adx'
        ]

        # Clean up data
        df_cleaned = df_calc.dropna(subset=feature_columns + ['target']).copy()
        
        logger.info(f"✅ [ML Prep] Data prepared for {symbol}. Rows: {len(df_cleaned)}, Positive targets: {df_cleaned['target'].sum()}")
        return df_cleaned[feature_columns + ['target']]
    except Exception as e:
        logger.error(f"❌ [ML Prep] Error preparing data for {symbol}: {e}", exc_info=True)
        return None

def train_and_evaluate_model(data: pd.DataFrame) -> Tuple[Any, Dict[str, Any]]:
    """
    Trains and evaluates a RandomForestClassifier with hyperparameter tuning.
    """
    logger.info("ℹ️ [ML Train] Starting model training and evaluation with RandomForest...")
    X = data.drop('target', axis=1)
    y = data['target']

    if X.empty or y.empty or y.nunique() < 2:
        logger.error("❌ [ML Train] Not enough data or target classes to train.")
        return None, {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Using RandomForest for better performance
    model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=150, max_depth=15)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        'model_type': 'RandomForestClassifier',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'feature_names': X.columns.tolist()
    }
    
    logger.info(f"📊 [ML Train] Model Performance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float): logger.info(f"  - {key.capitalize()}: {value:.4f}")
    
    return (model, scaler), metrics # Return model and scaler together

# ---------------------- Main Training Script ----------------------
if __name__ == "__main__":
    logger.info("🚀 Starting Enhanced ML Model Training Script...")
    # Mock implementations for standalone run
    client = Client(API_KEY, API_SECRET)
    init_db() # Needs real implementation

    symbols = get_crypto_symbols()
    if not symbols:
        logger.critical("❌ [Main] No valid symbols to train. Check 'crypto_list.txt'.")
        exit(1)

    overall_summary = []

    for symbol in symbols:
        model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
        logger.info(f"\n--- ⏳ [Main] Starting training for {symbol} ({model_name}) ---")
        
        try:
            df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING)
            if df_hist is None or df_hist.empty:
                logger.warning(f"⚠️ [Main] Could not fetch sufficient data for {symbol}. Skipping.")
                continue

            df_ml = prepare_data_for_ml(df_hist, symbol)
            if df_ml is None or df_ml.empty or df_ml['target'].sum() < 10:
                logger.warning(f"⚠️ [Main] No usable training data for {symbol} after prep. Skipping.")
                continue

            model_bundle, model_metrics = train_and_evaluate_model(df_ml)
            if model_bundle is None:
                logger.error(f"❌ [Main] Model training failed for {symbol}.")
                continue

            if save_ml_model_to_db(model_bundle, model_name, model_metrics):
                logger.info(f"✅ [Main] Successfully trained and saved model '{model_name}'.")
                summary = f"✅ {symbol}: Success | Precision: {model_metrics['precision']:.2f}, Recall: {model_metrics['recall']:.2f}"
                overall_summary.append(summary)
            else:
                logger.error(f"❌ [Main] Failed to save model '{model_name}' to the database.")

        except Exception as e:
            logger.critical(f"❌ [Main] A fatal error occurred during training for {symbol}: {e}", exc_info=True)
        
        time.sleep(2) # Pause between symbols

    # Send final summary to Telegram
    if TELEGRAM_TOKEN and CHAT_ID:
        summary_message = "🤖 *ML Model Training Complete*\n\n" + "\n".join(overall_summary)
        send_telegram_message(CHAT_ID, summary_message)
    
    logger.info("👋 [Main] Enhanced ML training script finished.")
