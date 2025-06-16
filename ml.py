import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
import lightgbm as lgb
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Tuple
from flask import Flask, request, Response
from threading import Thread
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# ---------------------- إعداد التسجيل ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_model_trainer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLTrainer')

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
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 90
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V4' # <-- تم تحديث إصدار النموذج ليشمل الميزات الجديدة

# Indicator Parameters
RSI_PERIOD: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
BBANDS_PERIOD: int = 20
BBANDS_STD_DEV: float = 2.0
ATR_PERIOD: int = 14

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
training_status: str = "Idle"

# ---------------------- Binance Client & DB Setup ----------------------
try:
    logger.info("ℹ️ [Binance] تهيئة عميل Binance...")
    client = Client(API_KEY, API_SECRET)
    client.ping()
except Exception as e:
    logger.critical(f"❌ [Binance] فشل غير متوقع في تهيئة عميل Binance: {e}")
    exit(1)

# --- Database and Symbol Validation Functions ---
def init_db(retries: int = 5, delay: int = 5) -> None:
    global conn; logger.info("[DB] بدء تهيئة قاعدة البيانات...")
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor); conn.autocommit = False; cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS ml_models (id SERIAL PRIMARY KEY, model_name TEXT NOT NULL UNIQUE, model_data BYTEA NOT NULL, trained_at TIMESTAMP DEFAULT NOW(), metrics JSONB);")
            conn.commit(); logger.info("✅ [DB] تم تهيئة قاعدة البيانات بنجاح."); return
        except Exception as e:
            logger.error(f"❌ [DB] خطأ في الاتصال (المحاولة {attempt + 1}): {e}");
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: exit(1)

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    logger.info(f"ℹ️ [Validation] Reading symbols from '{filename}' and validating with Binance...")
    try:
        with open(os.path.join(os.path.dirname(__file__), filename), 'r', encoding='utf-8') as f:
            formatted_symbols = {f"{s.strip().upper()}USDT" if not s.strip().upper().endswith('USDT') else s.strip().upper() for s in f if s.strip() and not s.startswith('#')}
        logger.info(f"ℹ️ [Validation] Found {len(formatted_symbols)} unique symbols in the file.")
        exchange_info = client.get_exchange_info()
        active_binance_symbols = {s['symbol'] for s in exchange_info['symbols'] if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'}
        logger.info(f"ℹ️ [Validation] Found {len(active_binance_symbols)} actively trading USDT pairs on Binance.")
        validated_symbols = sorted(list(formatted_symbols.intersection(active_binance_symbols)))
        ignored_symbols = formatted_symbols - active_binance_symbols
        if ignored_symbols: logger.warning(f"⚠️ [Validation] Ignored {len(ignored_symbols)} symbols: {', '.join(ignored_symbols)}")
        logger.info(f"✅ [Validation] Proceeding with {len(validated_symbols)} validated symbols.")
        return validated_symbols
    except Exception as e: logger.error(f"❌ [Validation] An error occurred: {e}"); return []

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    try:
        start_str = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms'); df.set_index('timestamp', inplace=True)
        return df[numeric_cols].dropna()
    except Exception as e: logger.error(f"❌ [Data] خطأ أثناء جلب البيانات لـ {symbol}: {e}"); return None

# --- Indicator and Feature Calculation ---
def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates all technical indicators and new requested features."""
    df_calc = df.copy()
    
    # ATR
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    # RSI
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df_calc['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df_calc['macd'] = ema_fast - ema_slow
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df_calc['macd_hist'] = df_calc['macd'] - df_calc['macd_signal']
    
    # --- NEW: MACD Crossover Feature ---
    # 1 for bullish crossover (MACD crosses above Signal)
    # -1 for bearish crossover (MACD crosses below Signal)
    # 0 for no crossover
    macd_above = df_calc['macd'] > df_calc['macd_signal']
    macd_below = df_calc['macd'] < df_calc['macd_signal']
    df_calc['macd_cross'] = 0
    # Bullish cross: was below in the previous candle, is now above
    df_calc.loc[macd_above & macd_below.shift(1), 'macd_cross'] = 1
    # Bearish cross: was above in the previous candle, is now below
    df_calc.loc[macd_below & macd_above.shift(1), 'macd_cross'] = -1

    # Bollinger Bands
    sma = df_calc['close'].rolling(window=BBANDS_PERIOD).mean()
    std = df_calc['close'].rolling(window=BBANDS_PERIOD).std()
    df_calc['bb_upper'] = sma + (std * BBANDS_STD_DEV)
    df_calc['bb_lower'] = sma - (std * BBANDS_STD_DEV)
    df_calc['bb_width'] = (df_calc['bb_upper'] - df_calc['bb_lower']) / sma
    df_calc['bb_pos'] = (df_calc['close'] - sma) / std.replace(0, np.nan)
    
    # --- NEW: Time-based Features ---
    # Day of the week (Monday=0, Sunday=6)
    df_calc['day_of_week'] = df_calc.index.dayofweek
    # Hour of the day (0-23)
    df_calc['hour_of_day'] = df_calc.index.hour

    # Candlestick features
    df_calc['candle_body_size'] = (df_calc['close'] - df_calc['open']).abs()
    df_calc['upper_wick'] = df_calc['high'] - df_calc[['open', 'close']].max(axis=1)
    df_calc['lower_wick'] = df_calc[['open', 'close']].min(axis=1) - df_calc['low']

    # Relative volume
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=30, min_periods=1).mean() + 1e-9)

    return df_calc

# --- Model Training Logic ---

def prepare_data_for_ml(df: pd.DataFrame, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    logger.info(f"ℹ️ [ML Prep] Preparing data for ML model for {symbol} with new features...")
    df_featured = calculate_features(df)

    # --- Target Definition (Unchanged) ---
    # We still want to predict a price rise, but now with more context about bearish conditions.
    profit_target_pct = 0.015  # 1.5% profit
    look_forward_period = 12 # How many 15-min candles to look into the future (3 hours)
    
    future_high = df_featured['high'].rolling(window=look_forward_period).max().shift(-look_forward_period)
    df_featured['target'] = ((future_high / df_featured['close']) - 1 > profit_target_pct).astype(int)
    
    # --- UPDATED: Feature Columns ---
    feature_columns = [
        'volume', 'relative_volume', 'rsi', 'macd_hist', 'bb_width', 'bb_pos', 'atr',
        'candle_body_size', 'upper_wick', 'lower_wick',
        # --- NEW FEATURES ADDED FOR TRAINING ---
        'macd_cross',
        'day_of_week',
        'hour_of_day'
    ]
    
    df_cleaned = df_featured.dropna(subset=feature_columns + ['target']).copy()
    
    if df_cleaned.empty or df_cleaned['target'].nunique() < 2:
        logger.warning(f"⚠️ [ML Prep] DataFrame for {symbol} is empty or has only one class after feature calculation. Skipping.")
        return None
        
    logger.info(f"📊 [ML Prep] Target distribution for {symbol}:\n{df_cleaned['target'].value_counts(normalize=True)}")
    
    if df_cleaned['target'].value_counts(normalize=True).get(1, 0) < 0.01:
        logger.warning(f"⚠️ [ML Prep] Target class '1' is less than 1% of data for {symbol}. Model may struggle.")

    X = df_cleaned[feature_columns]
    y = df_cleaned['target']
    
    return X, y, feature_columns

def train_and_evaluate_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    logger.info("ℹ️ [ML Train] Starting model training and evaluation with LightGBM...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    model = lgb.LGBMClassifier(
        objective='binary',
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=-1,
        class_weight='balanced',
        colsample_bytree=0.8,
        subsample=0.8
    )
    
    model.fit(X_train_scaled, y_train,
              eval_set=[(X_test_scaled, y_test)],
              eval_metric='logloss',
              callbacks=[lgb.early_stopping(25, verbose=False)])
    
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'num_samples_trained': len(X_train),
    }
    metrics_log_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, float)])
    logger.info(f"📊 [ML Train] LightGBM performance: {metrics_log_str}")
    
    return model, scaler, metrics

def save_ml_model_to_db(model_bundle: Dict[str, Any], model_name: str, metrics: Dict[str, Any]) -> bool:
    logger.info(f"ℹ️ [DB Save] Attempting to save ML model bundle '{model_name}'...")
    if not conn: return False
    try:
        model_binary = pickle.dumps(model_bundle)
        metrics_json = json.dumps(metrics)
        with conn.cursor() as db_cur:
            db_cur.execute("INSERT INTO ml_models (model_name, model_data, trained_at, metrics) VALUES (%s, %s, NOW(), %s) ON CONFLICT (model_name) DO UPDATE SET model_data = EXCLUDED.model_data, trained_at = NOW(), metrics = EXCLUDED.metrics;", (model_name, model_binary, metrics_json))
        conn.commit()
        logger.info(f"✅ [DB Save] Model bundle '{model_name}' saved successfully.")
        return True
    except Exception as e:
        logger.error(f"❌ [DB Save] Unexpected error while saving ML model bundle: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

# --- Main Execution ---
def send_telegram_message(target_chat_id: str, text: str):
    if not TELEGRAM_TOKEN or not target_chat_id: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try: requests.post(url, json={'chat_id': str(target_chat_id), 'text': text, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

if __name__ == "__main__":
    logger.info(f"🚀 Starting ML model training script ({BASE_ML_MODEL_NAME})...")
    app = Flask(__name__)
    @app.route('/')
    def home(): return Response(f"ML Trainer Service: {training_status}", status=200)
    flask_thread = Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': int(os.environ.get('PORT', 10001))}, daemon=True)
    flask_thread.start()

    try:
        init_db()
        symbols_to_train = get_validated_symbols()
        if not symbols_to_train:
            logger.critical("❌ [Main] No valid symbols to train after validation. Exiting.")
            exit(1)
        
        training_status = "In Progress"
        send_telegram_message(CHAT_ID, f"🚀 *{BASE_ML_MODEL_NAME} Training Started*\nWill train models for {len(symbols_to_train)} symbols.")
        
        successful_models, failed_models = 0, 0
        for symbol in symbols_to_train:
            logger.info(f"\n--- ⏳ [Main] Starting model training for {symbol} ---")
            try:
                df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING)
                if df_hist is None or df_hist.empty: continue
                
                prepared_data = prepare_data_for_ml(df_hist, symbol)
                if prepared_data is None: failed_models += 1; continue
                
                X, y, feature_names = prepared_data
                trained_model, trained_scaler, model_metrics = train_and_evaluate_model(X, y)
                
                if trained_model and trained_scaler and model_metrics.get('precision', 0) > 0.1:
                    model_bundle = {'model': trained_model, 'scaler': trained_scaler, 'feature_names': feature_names}
                    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
                    if save_ml_model_to_db(model_bundle, model_name, model_metrics):
                        successful_models += 1
                    else:
                        failed_models += 1
                else:
                    logger.warning(f"⚠️ [Main] Model for {symbol} is not useful (precision <= 0.1). Discarding.")
                    failed_models += 1

            except Exception as e:
                logger.critical(f"❌ [Main] A fatal error occurred for {symbol}: {e}", exc_info=True)
                failed_models += 1
            time.sleep(1)

        training_status = f"Completed. {successful_models} successful, {failed_models} failed/discarded."
        send_telegram_message(CHAT_ID, f"✅ *{BASE_ML_MODEL_NAME} Training Finished*\nSuccessfully trained {successful_models}/{len(symbols_to_train)} useful models.")
        
    except Exception as e:
        logger.critical(f"❌ [Main] A fatal error occurred in main script: {e}", exc_info=True)
        training_status = f"Failed: {e}"
        send_telegram_message(CHAT_ID, f"🚨 *Fatal Error in ML Training Script*\n`{e}`")
    finally:
        if conn: conn.close()
        logger.info("👋 [Main] ML training script finished.")
        # Do not join the flask_thread to allow the script to exit
        # This is suitable for a script that runs, trains, and then exits.
