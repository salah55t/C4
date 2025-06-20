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
import pandas_ta as ta
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from binance.client import Client
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from flask import Flask
from threading import Thread

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_model_trainer_v7.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLTrainer_V7')

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
# --- V7 Model Constants ---
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V7' # تحديث اسم النموذج
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 200 # زيادة فترة التدريب

# --- Indicator & Feature Parameters ---
RSI_PERIOD: int = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD: int = 14
BOLLINGER_PERIOD: int = 20
ADX_PERIOD: int = 14
MOMENTUM_PERIOD: int = 10 # ميزة جديدة
EMA_SLOW_PERIOD: int = 200
EMA_FAST_PERIOD: int = 50
BTC_CORR_PERIOD: int = 30
BTC_SYMBOL = 'BTCUSDT'

# --- Triple-Barrier Method Parameters (V7) ---
TP_ATR_MULTIPLIER: float = 1.8 # تعديل مضاعف الربح
SL_ATR_MULTIPLIER: float = 1.2 # تعديل مضاعف الخسارة
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

def get_binance_client():
    global client
    try:
        client = Client(API_KEY, API_SECRET)
        client.ping()
        logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
    except Exception as e:
        logger.critical(f"❌ [Binance] فشل تهيئة عميل Binance: {e}"); exit(1)

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    if not client:
        logger.error("❌ [Validation] عميل Binance لم يتم تهيئته.")
        return []
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

# --- دوال جلب ومعالجة البيانات ---
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    try:
        start_str = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df[numeric_cols].dropna()
    except Exception as e:
        logger.error(f"❌ [Data] خطأ أثناء جلب البيانات لـ {symbol}: {e}"); return None

def fetch_and_cache_btc_data():
    global btc_data_cache
    logger.info("ℹ️ [BTC Data] جاري جلب بيانات البيتكوين وتخزينها...")
    btc_data_cache = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
    if btc_data_cache is None:
        logger.critical("❌ [BTC Data] فشل جلب بيانات البيتكوين."); exit(1)
    btc_data_cache['btc_returns'] = btc_data_cache['close'].pct_change()

def calculate_features_v7(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.astype('float64')
    
    strategy = ta.Strategy(
        name="V7_Features",
        ta=[
            {"kind": "ema", "length": EMA_FAST_PERIOD},
            {"kind": "ema", "length": EMA_SLOW_PERIOD},
            {"kind": "atr", "length": ATR_PERIOD},
            {"kind": "bbands", "length": BOLLINGER_PERIOD},
            {"kind": "rsi", "length": RSI_PERIOD},
            {"kind": "macd", "fast": MACD_FAST, "slow": MACD_SLOW, "signal": MACD_SIGNAL},
            {"kind": "obv"},
            {"kind": "adx", "length": ADX_PERIOD},
        ]
    )
    df_calc.ta.strategy(strategy)

    # V7 - New Features
    df_calc['log_returns'] = ta.log_return(close=df_calc['close'])
    df_calc['volatility'] = df_calc['log_returns'].rolling(window=ATR_PERIOD).std()
    df_calc['momentum'] = ta.mom(close=df_calc['close'], length=MOMENTUM_PERIOD)

    df_calc['price_vs_ema_fast'] = (df_calc['close'] / df_calc[f'EMA_{EMA_FAST_PERIOD}']) - 1
    df_calc['price_vs_ema_slow'] = (df_calc['close'] / df_calc[f'EMA_{EMA_SLOW_PERIOD}']) - 1
    df_calc['bollinger_width'] = df_calc[f'BBB_{BOLLINGER_PERIOD}_2.0']
    
    btc_df_float = btc_df.astype({'btc_returns': 'float64'})
    merged_df = pd.merge(df_calc, btc_df_float[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
    df_calc['btc_correlation'] = df_calc['log_returns'].rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    
    df_calc['day_of_week'] = df_calc.index.dayofweek
    df_calc['hour_of_day'] = df_calc.index.hour
    
    df_calc.columns = [col.upper() for col in df_calc.columns]
    
    return df_calc


def get_triple_barrier_labels(prices: pd.Series, atr: pd.Series) -> pd.Series:
    labels = pd.Series(0, index=prices.index, dtype='int8')
    for i in tqdm(range(len(prices) - MAX_HOLD_PERIOD), desc="Labeling", leave=False, ncols=100):
        entry_price = prices.iloc[i]
        current_atr = atr.iloc[i]
        if pd.isna(current_atr) or current_atr == 0: continue
        
        upper_barrier = entry_price + (current_atr * TP_ATR_MULTIPLIER)
        lower_barrier = entry_price - (current_atr * SL_ATR_MULTIPLIER)
        
        for j in range(1, MAX_HOLD_PERIOD + 1):
            if i + j >= len(prices): break
            
            future_price = prices.iloc[i + j]
            
            if future_price >= upper_barrier:
                labels.iloc[i] = 1
                break
            if future_price <= lower_barrier:
                labels.iloc[i] = -1
                break
    return labels

def prepare_data_for_ml(df: pd.DataFrame, btc_df: pd.DataFrame, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    logger.info(f"ℹ️ [ML Prep V7] Preparing data for {symbol}...")
    df_featured = calculate_features_v7(df, btc_df)
    
    atr_series_name = f'ATRR_{ATR_PERIOD}'.upper()
    if atr_series_name not in df_featured.columns:
        standard_atr_name = f'ATR_{ATR_PERIOD}'.upper()
        if standard_atr_name in df_featured.columns:
            atr_series_name = standard_atr_name
        else:
            logger.error(f"FATAL: ATR column not found for {symbol}.")
            return None
        
    df_featured['TARGET'] = get_triple_barrier_labels(df_featured['CLOSE'], df_featured[atr_series_name])
    
    # V7 feature list
    feature_columns = [
        f'RSI_{RSI_PERIOD}', f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}', 
        f'MACDH_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}',
        atr_series_name, 'BOLLINGER_WIDTH', 'OBV',
        f'ADX_{ADX_PERIOD}', 'PRICE_VS_EMA_FAST', 'PRICE_VS_EMA_SLOW',
        'BTC_CORRELATION', 'VOLATILITY', 'MOMENTUM',
        'DAY_OF_WEEK', 'HOUR_OF_DAY'
    ]
    feature_columns = [col.upper() for col in feature_columns]

    df_cleaned = df_featured.dropna(subset=feature_columns + ['TARGET']).copy()
    
    df_cleaned = df_cleaned[df_cleaned['TARGET'] != 0]

    if df_cleaned.empty or df_cleaned['TARGET'].nunique() < 2:
        logger.warning(f"⚠️ [ML Prep] Data for {symbol} has less than 2 classes after filtering. Skipping.")
        return None
    
    df_cleaned['TARGET'] = df_cleaned['TARGET'].replace(-1, 0)
    
    # Check for class imbalance before proceeding
    target_counts = df_cleaned['TARGET'].value_counts(normalize=True)
    logger.info(f"📊 [ML Prep] Target distribution for {symbol} (after filtering):\n{target_counts}")
    if target_counts.min() < 0.1: # If one class is less than 10%
        logger.warning(f"⚠️ [ML Prep] Severe class imbalance for {symbol}. Min class is {target_counts.min():.2%}. Skipping training.")
        return None

    X = df_cleaned[feature_columns]
    y = df_cleaned['TARGET']
    return X, y, feature_columns

def train_with_walk_forward_validation(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    logger.info("ℹ️ [ML Train V7] Starting training with Walk-Forward Validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    # --- FIX 2: Updated model parameters for more flexibility ---
    # These parameters make the model more complex and less constrained,
    # which can help overcome the "No further splits" warning.
    lgb_params = {
        'objective': 'binary',
        'metric': 'logloss',
        'random_state': 42,
        'verbosity': -1,             # Suppress verbose warnings
        'n_estimators': 1500,        # Give it more trees to build
        'learning_rate': 0.01,       # Lower learning rate requires more estimators
        'num_leaves': 31,            # Default complexity
        'max_depth': -1,             # No limit on depth
        'class_weight': 'balanced',
        'reg_alpha': 0.0,            # Turn off L1 regularization for this test
        'reg_lambda': 0.0,           # Turn off L2 regularization for this test
        'n_jobs': -1,
        'colsample_bytree': 0.8,     # Use a bit more features per tree
        'min_child_samples': 10,     # Allow splits that result in smaller leaves
        'boosting_type': 'gbdt',
    }
    
    final_model, final_scaler = None, None

    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        if len(y_train) == 0 or len(y_test) == 0:
            logger.warning(f"--- Fold {i+1}: Skipping due to empty train/test set.")
            continue
        
        scaler = StandardScaler().fit(X_train)
        
        X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
        
        model = lgb.LGBMClassifier(**lgb_params)
        
        # FIX 1: Added eval_metric='logloss' to the fit method.
        model.fit(X_train_scaled, y_train, 
                  eval_set=[(X_test_scaled, y_test)],
                  eval_metric='logloss',
                  callbacks=[lgb.early_stopping(100, verbose=False)]) # Increased patience for early stopping
        
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        logger.info(f"--- Fold {i+1}: Accuracy: {accuracy_score(y_test, y_pred):.4f}, "
                    f"P(Win): {report.get('1', {}).get('precision', 0):.4f}, "
                    f"P(Loss): {report.get('0', {}).get('precision', 0):.4f}")
        
        final_model, final_scaler = model, scaler

    if not final_model or not final_scaler:
        logger.error("❌ [ML Train] Training failed, no model was created.")
        return None, None, None

    all_preds = final_model.predict(final_scaler.transform(X))
    final_report = classification_report(y, all_preds, output_dict=True, zero_division=0)
    avg_metrics = {
        'accuracy': accuracy_score(y, all_preds),
        'precision_win': final_report.get('1', {}).get('precision', 0),
        'recall_win': final_report.get('1', {}).get('recall', 0),
        'f1_score_win': final_report.get('1', {}).get('f1-score', 0),
        'num_samples_trained': len(X),
    }

    metrics_log_str = ', '.join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
    logger.info(f"📊 [ML Train] Final Model Performance on All Data: {metrics_log_str}")
    return final_model, final_scaler, avg_metrics

def save_ml_model_to_db(model_bundle: Dict[str, Any], model_name: str, metrics: Dict[str, Any]):
    logger.info(f"ℹ️ [DB Save] Saving model bundle '{model_name}'...")
    try:
        if conn is None or conn.closed:
            logger.warning("[DB Save] DB connection is closed. Re-initializing.")
            init_db()

        model_binary = pickle.dumps(model_bundle)
        metrics_json = json.dumps(metrics)
        with conn.cursor() as db_cur:
            db_cur.execute("""
                INSERT INTO ml_models (model_name, model_data, trained_at, metrics) 
                VALUES (%s, %s, NOW(), %s) ON CONFLICT (model_name) DO UPDATE SET 
                model_data = EXCLUDED.model_data, trained_at = NOW(), metrics = EXCLUDED.metrics;
            """, (model_name, model_binary, metrics_json))
        conn.commit()
        logger.info(f"✅ [DB Save] Model bundle '{model_name}' saved successfully.")
    except Exception as e:
        logger.error(f"❌ [DB Save] Error saving model bundle: {e}"); 
        if conn: conn.rollback()

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try: requests.post(url, json={'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

def run_training_job():
    logger.info(f"🚀 Starting ML model training job ({BASE_ML_MODEL_NAME})...")
    init_db()
    get_binance_client()
    fetch_and_cache_btc_data()
    symbols_to_train = get_validated_symbols(filename='crypto_list.txt')
    if not symbols_to_train:
        logger.critical("❌ [Main] No valid symbols found. Exiting.")
        return
        
    send_telegram_message(f"🚀 *{BASE_ML_MODEL_NAME} Training Started*\nWill train models for {len(symbols_to_train)} symbols.")
    
    successful_models, failed_models = 0, 0
    for symbol in symbols_to_train:
        logger.info(f"\n--- ⏳ [Main] Starting model training for {symbol} ---")
        try:
            df_hist = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
            if df_hist is None or df_hist.empty:
                logger.warning(f"⚠️ [Main] No data for {symbol}, skipping."); failed_models += 1; continue
            
            prepared_data = prepare_data_for_ml(df_hist, btc_data_cache, symbol)
            if prepared_data is None:
                failed_models += 1; continue
            X, y, feature_names = prepared_data
            
            training_result = train_with_walk_forward_validation(X, y)
            if not all(training_result):
                 logger.warning(f"⚠️ [Main] Training did not produce a valid model for {symbol}. Skipping."); failed_models += 1; continue
            final_model, final_scaler, model_metrics = training_result
            
            # V7 - Stricter condition to save the model
            if final_model and final_scaler and model_metrics.get('precision_win', 0) > 0.52 and model_metrics.get('f1_score_win', 0) > 0.5:
                model_bundle = {'model': final_model, 'scaler': final_scaler, 'feature_names': feature_names}
                model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
                save_ml_model_to_db(model_bundle, model_name, model_metrics)
                successful_models += 1
            else:
                precision = model_metrics.get('precision_win', 0)
                f1_score = model_metrics.get('f1_score_win', 0)
                logger.warning(f"⚠️ [Main] Model for {symbol} is not useful (Precision {precision:.2f}, F1-Score: {f1_score:.2f}). Discarding."); failed_models += 1
        except Exception as e:
            logger.critical(f"❌ [Main] A fatal error occurred for {symbol}: {e}", exc_info=True); failed_models += 1
        time.sleep(1)

    completion_message = (f"✅ *{BASE_ML_MODEL_NAME} Training Finished*\n"
                        f"- Successfully trained: {successful_models} models\n"
                        f"- Failed/Discarded: {failed_models} models\n"
                        f"- Total symbols: {len(symbols_to_train)}")
    send_telegram_message(completion_message)
    logger.info(completion_message)

    if conn: conn.close()
    logger.info("👋 [Main] ML training job finished.")

app = Flask(__name__)

@app.route('/')
def health_check():
    return "ML Trainer service (V7) is running and healthy.", 200

if __name__ == "__main__":
    training_thread = Thread(target=run_training_job)
    training_thread.daemon = True
    training_thread.start()
    
    port = int(os.environ.get("PORT", 10001))
    logger.info(f"🌍 Starting web server on port {port} to keep the service alive...")
    app.run(host='0.0.0.0', port=port)
