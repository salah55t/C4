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
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V2'

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

def init_db(retries: int = 5, delay: int = 5) -> None:
    global conn
    logger.info("[DB] بدء تهيئة قاعدة البيانات...")
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ml_models (
                    id SERIAL PRIMARY KEY, 
                    model_name TEXT NOT NULL UNIQUE,
                    model_data BYTEA NOT NULL, 
                    trained_at TIMESTAMP DEFAULT NOW(), 
                    metrics JSONB
                );
            """)
            conn.commit()
            logger.info("✅ [DB] تم تهيئة قاعدة البيانات بنجاح.")
            return
        except Exception as e:
            logger.error(f"❌ [DB] خطأ في الاتصال (المحاولة {attempt + 1}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: exit(1)

# ---------------------- Symbol Validation ----------------------
def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """
    Reads symbols from a file, validates them against Binance to ensure they are actively trading.
    """
    logger.info(f"ℹ️ [Validation] Reading symbols from '{filename}' and validating with Binance...")
    try:
        # Step 1: Read symbols from the local file
        with open(os.path.join(os.path.dirname(__file__), filename), 'r', encoding='utf-8') as f:
            raw_symbols_from_file = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        
        # Ensure symbols are formatted correctly as USDT pairs
        formatted_symbols = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols_from_file}
        logger.info(f"ℹ️ [Validation] Found {len(formatted_symbols)} unique symbols in the file.")

        # Step 2: Get all actively trading symbols from Binance
        exchange_info = client.get_exchange_info()
        active_binance_symbols = {
            s['symbol'] for s in exchange_info['symbols'] 
            if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'
        }
        logger.info(f"ℹ️ [Validation] Found {len(active_binance_symbols)} actively trading USDT pairs on Binance.")

        # Step 3: Find the intersection - symbols that are in your file AND active on Binance
        validated_symbols = sorted(list(formatted_symbols.intersection(active_binance_symbols)))
        
        # Step 4: Log the results
        ignored_symbols = formatted_symbols - active_binance_symbols
        if ignored_symbols:
            logger.warning(f"⚠️ [Validation] Ignored {len(ignored_symbols)} symbols not found or not active on Binance: {', '.join(ignored_symbols)}")
        
        logger.info(f"✅ [Validation] Proceeding with {len(validated_symbols)} validated symbols for training.")
        return validated_symbols
        
    except FileNotFoundError:
        logger.error(f"❌ [Validation] Critical error: The file '{filename}' was not found.")
        return []
    except Exception as e:
        logger.error(f"❌ [Validation] An error occurred during symbol validation: {e}")
        return []

# All other functions (fetch_historical_data, indicator calculations, prepare_data_for_ml, train_and_evaluate_model, etc.)
# remain the same as the previous version. They are included here for completeness.

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        start_str = (datetime.utcnow() - timedelta(days=days + 1)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df[numeric_cols].dropna()
    except Exception as e:
        logger.error(f"❌ [Data] خطأ أثناء جلب البيانات التاريخية لـ {symbol}: {e}")
        return None

def calculate_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.Series:
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_macd(df: pd.DataFrame, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> Tuple[pd.Series, pd.Series]:
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def calculate_bollinger_bands(df: pd.DataFrame, period: int = BBANDS_PERIOD, std_dev: float = BBANDS_STD_DEV) -> Tuple[pd.Series, pd.Series]:
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, lower_band

def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calculate_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    df['candle_body_size'] = (df['close'] - df['open']).abs()
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    return df

def prepare_data_for_ml(df: pd.DataFrame, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    logger.info(f"ℹ️ [ML Prep] Preparing data for ML model for {symbol}...")
    df_calc = df.copy()

    df_calc['atr'] = calculate_atr(df_calc, ATR_PERIOD)
    df_calc['rsi'] = calculate_rsi(df_calc, RSI_PERIOD)
    df_calc['macd'], df_calc['macd_signal'] = calculate_macd(df_calc)
    df_calc['macd_hist'] = df_calc['macd'] - df_calc['macd_signal']
    df_calc['bb_upper'], df_calc['bb_lower'] = calculate_bollinger_bands(df_calc)
    df_calc['bb_width'] = (df_calc['bb_upper'] - df_calc['bb_lower']) / df_calc['close']
    df_calc = calculate_candlestick_features(df_calc)
    df_calc['relative_volume'] = df_calc['volume'] / df_calc['volume'].rolling(window=30, min_periods=1).mean()

    profit_target_pct, stop_loss_pct, look_forward_period = 0.015, 0.01, 12
    df_calc['target'] = 0
    
    for i in range(len(df_calc) - look_forward_period):
        entry_price = df_calc['close'].iloc[i]
        target_price, stop_loss_price = entry_price * (1 + profit_target_pct), entry_price * (1 - stop_loss_pct)
        future_prices = df_calc.iloc[i+1 : i+1+look_forward_period]
        hit_target_time = future_prices[future_prices['high'] >= target_price].index.min()
        hit_stop_time = future_prices[future_prices['low'] <= stop_loss_price].index.min()

        if pd.notna(hit_target_time) and (pd.isna(hit_stop_time) or hit_target_time < hit_stop_time):
            df_calc.loc[df_calc.index[i], 'target'] = 1
    
    feature_columns = ['volume', 'relative_volume', 'rsi', 'macd_hist', 'bb_width', 'atr', 'candle_body_size', 'upper_wick', 'lower_wick']
    df_cleaned = df_calc.dropna(subset=feature_columns + ['target']).copy()

    if df_cleaned.empty or df_cleaned['target'].nunique() < 2:
        logger.warning(f"⚠️ [ML Prep] DataFrame for {symbol} is empty or has only one class after cleaning.")
        return None

    return df_cleaned[feature_columns], df_cleaned['target'], feature_columns

def train_and_evaluate_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    logger.info("ℹ️ [ML Train] Starting model training and evaluation with LightGBM...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)
    
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1] if y_train.value_counts()[1] > 0 else 1
    model = lgb.LGBMClassifier(objective='binary', random_state=42, n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=7, n_jobs=-1, colsample_bytree=0.8, subsample=0.8, reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=scale_pos_weight)
    model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], eval_metric='logloss', callbacks=[lgb.early_stopping(20, verbose=False)])
    
    y_pred = model.predict(X_test_scaled)
    metrics = {'accuracy': accuracy_score(y_test, y_pred), 'precision': precision_score(y_test, y_pred, zero_division=0), 'recall': recall_score(y_test, y_pred, zero_division=0), 'f1_score': f1_score(y_test, y_pred, zero_division=0), 'num_samples_trained': len(X_train)}
    logger.info(f"📊 [ML Train] LightGBM performance: { {k: f'{v:.4f}' for k, v in metrics.items() if isinstance(v, float)} }")
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

def send_telegram_message(target_chat_id: str, text: str):
    if not TELEGRAM_TOKEN or not target_chat_id: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try: requests.post(url, json={'chat_id': str(target_chat_id), 'text': text, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

if __name__ == "__main__":
    logger.info("🚀 Starting ML model training script with LightGBM (V2)...")
    app = Flask(__name__)
    @app.route('/')
    def home(): return Response(f"ML Trainer Service: {training_status}", status=200)
    flask_thread = Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': int(os.environ.get('PORT', 10001))}, daemon=True)
    flask_thread.start()

    try:
        init_db()
        symbols_to_train = get_validated_symbols() # <-- Use the validated list
        if not symbols_to_train:
            logger.critical("❌ [Main] No valid symbols to train after validation. Exiting.")
            exit(1)
        
        training_status = "In Progress"
        send_telegram_message(CHAT_ID, f"🚀 *LightGBM Model Training Started (V2)*\nWill train models for {len(symbols_to_train)} validated symbols.")
        
        successful_models, failed_models = 0, 0
        for symbol in symbols_to_train:
            logger.info(f"\n--- ⏳ [Main] Starting model training for {symbol} ---")
            try:
                df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING)
                if df_hist is None or df_hist.empty: continue
                
                prepared_data = prepare_data_for_ml(df_hist, symbol)
                if prepared_data is None: continue
                
                X, y, feature_names = prepared_data
                trained_model, trained_scaler, model_metrics = train_and_evaluate_model(X, y)
                
                if trained_model and trained_scaler:
                    model_bundle = {'model': trained_model, 'scaler': trained_scaler, 'feature_names': feature_names}
                    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
                    if save_ml_model_to_db(model_bundle, model_name, model_metrics):
                        successful_models += 1
                    else:
                        failed_models += 1
                else:
                    logger.warning(f"⚠️ [Main] Training did not produce a valid model for {symbol}.")
                    failed_models += 1
            except Exception as e:
                logger.critical(f"❌ [Main] A fatal error occurred for {symbol}: {e}", exc_info=True)
                failed_models += 1
            time.sleep(1)

        training_status = f"Completed. {successful_models} successful, {failed_models} failed."
        send_telegram_message(CHAT_ID, f"✅ *LightGBM Training Finished (V2)*\nSuccessfully trained {successful_models}/{len(symbols_to_train)} models.")
        
    except Exception as e:
        logger.critical(f"❌ [Main] A fatal error occurred in main script: {e}", exc_info=True)
        training_status = f"Failed: {e}"
        send_telegram_message(CHAT_ID, f"🚨 *Fatal Error in ML Training Script*\n`{e}`")
    finally:
        if conn: conn.close()
        logger.info("👋 [Main] ML training script finished. Flask server continues to run.")
        flask_thread.join()
