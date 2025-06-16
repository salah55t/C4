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

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_model_trainer_v5.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLTrainer_V5')

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
# !!! اسم النموذج الجديد بعد التحديثات الجذرية !!!
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V5'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 120 # زيادة المدة للحصول على بيانات كافية للميزات الجديدة
BTC_SYMBOL = 'BTCUSDT'

# Indicator & Feature Parameters
RSI_PERIOD: int = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BBANDS_PERIOD: int = 20
ATR_PERIOD: int = 14
EMA_SLOW_PERIOD: int = 200
EMA_FAST_PERIOD: int = 50
BTC_CORR_PERIOD: int = 30 # فترة حساب الارتباط مع البيتكوين

# Triple-Barrier Method Parameters
# سيتم تحديد الهدف بناءً على ATR
# كم ضعف من ATR سيتم استخدامه كهدف ربح أو وقف خسارة
TP_ATR_MULTIPLIER: float = 2.0
SL_ATR_MULTIPLIER: float = 1.5
# أقصى مدة للصفقة (عدد الشموع) قبل إغلاقها
MAX_HOLD_PERIOD: int = 24 # 24 * 15m = 6 hours

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
                    id SERIAL PRIMARY KEY,
                    model_name TEXT NOT NULL UNIQUE,
                    model_data BYTEA NOT NULL,
                    trained_at TIMESTAMP DEFAULT NOW(),
                    metrics JSONB
                );
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
        logger.error(f"❌ [Data] خطأ أثناء جلب البيانات لـ {symbol}: {e}")
        return None

def fetch_and_cache_btc_data():
    """يجلب بيانات البيتكوين ويخزنها مؤقتاً لتجنب الطلبات المتكررة."""
    global btc_data_cache
    logger.info("ℹ️ [BTC Data] جاري جلب بيانات البيتكوين وتخزينها...")
    btc_data_cache = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
    if btc_data_cache is None:
        logger.critical("❌ [BTC Data] فشل جلب بيانات البيتكوين. لا يمكن المتابعة بدونها.")
        exit(1)
    # Calculate BTC returns for correlation
    btc_data_cache['btc_returns'] = btc_data_cache['close'].pct_change()

def calculate_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """
    يحسب جميع المؤشرات والميزات بما في ذلك الميزات الجديدة المعتمدة على الاتجاه والارتباط.
    """
    df_calc = df.copy()

    # --- Standard Indicators (from V4) ---
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    df_calc['rsi'] = 100 - (100 / (1 + rs))

    ema_fast = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df_calc['macd_hist'] = (ema_fast - ema_slow) - (ema_fast - ema_slow).ewm(span=MACD_SIGNAL, adjust=False).mean()
    
    # --- !!! NEW: Trend Features (EMAs) !!! ---
    ema_fast_trend = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    ema_slow_trend = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    df_calc['price_vs_ema50'] = (df_calc['close'] / ema_fast_trend) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / ema_slow_trend) - 1
    
    # --- !!! NEW: BTC Correlation Feature !!! ---
    df_calc['returns'] = df_calc['close'].pct_change()
    # Merge BTC returns to align timestamps
    merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
    df_calc['btc_correlation'] = merged_df['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])

    # Other features
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=30, min_periods=1).mean() + 1e-9)
    df_calc['hour_of_day'] = df_calc.index.hour
    
    return df_calc

# --- !!! NEW: Triple-Barrier Labeling Function !!! ---
def get_triple_barrier_labels(prices: pd.Series, atr: pd.Series) -> pd.Series:
    """
    تحدد الهدف (1, -1, 0) بناءً على طريقة الحاجز الثلاثي.
    - 1: حاجز الربح تم لمسه.
    - -1: حاجز وقف الخسارة تم لمسه.
    - 0: انتهى الوقت.
    """
    labels = pd.Series(0, index=prices.index)
    
    for i in tqdm(range(len(prices) - MAX_HOLD_PERIOD), desc="Labeling", leave=False):
        entry_price = prices.iloc[i]
        current_atr = atr.iloc[i]
        
        if pd.isna(current_atr) or current_atr == 0:
            continue
            
        upper_barrier = entry_price + (current_atr * TP_ATR_MULTIPLIER)
        lower_barrier = entry_price - (current_atr * SL_ATR_MULTIPLIER)
        
        for j in range(1, MAX_HOLD_PERIOD + 1):
            future_price_high = prices.iloc[i + j]
            future_price_low = prices.iloc[i + j] # Simplified for kline data
            
            # Check for TP hit
            if future_price_high >= upper_barrier:
                labels.iloc[i] = 1
                break
            # Check for SL hit
            if future_price_low <= lower_barrier:
                labels.iloc[i] = -1
                break
    return labels

def prepare_data_for_ml(df: pd.DataFrame, btc_df: pd.DataFrame, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    logger.info(f"ℹ️ [ML Prep] Preparing data for {symbol} with advanced features...")
    df_featured = calculate_features(df, btc_df)

    # --- !!! NEW: Labeling using Triple-Barrier ---
    df_featured['target'] = get_triple_barrier_labels(df_featured['close'], df_featured['atr'])
    
    feature_columns = [
        'rsi', 'macd_hist', 'atr', 'relative_volume', 'hour_of_day',
        'price_vs_ema50', 'price_vs_ema200', 'btc_correlation'
    ]
    
    df_cleaned = df_featured.dropna(subset=feature_columns + ['target']).copy()
    
    # For multiclass, we need to ensure all classes are present in the dataset
    if df_cleaned.empty or df_cleaned['target'].nunique() < 2:
        logger.warning(f"⚠️ [ML Prep] DataFrame for {symbol} has less than 2 classes after feature calculation. Skipping.")
        return None
        
    logger.info(f"📊 [ML Prep] Target distribution for {symbol}:\n{df_cleaned['target'].value_counts(normalize=True)}")
    
    # Filter out entries where the model should not trade (target=0 or -1)
    # We want to train the model on WHEN to enter a trade, so we only look for "buy" signals
    # The model will predict one of the 3 outcomes for a potential trade.
    # The BOT will only act on a prediction of "1" (TP hit).
    X = df_cleaned[feature_columns]
    y = df_cleaned['target']
    
    return X, y, feature_columns

# --- !!! NEW: Walk-Forward Validation and Training Function !!! ---
def train_with_walk_forward_validation(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    logger.info("ℹ️ [ML Train] Starting training with Walk-Forward Validation...")
    
    tscv = TimeSeriesSplit(n_splits=5)
    all_metrics = []
    final_model = None
    final_scaler = None

    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        scaler = StandardScaler().fit(X_train)
        X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
        
        # --- Model configured for Multi-class classification ---
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3, # (1 for TP, -1 for SL, 0 for Timeout)
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            class_weight='balanced',
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train,
                  eval_set=[(X_test_scaled, y_test)],
                  eval_metric='multi_logloss',
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        
        y_pred = model.predict(X_test_scaled)
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"--- Fold {i+1} ---")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Class 1 (TP) Precision: {report.get('1', {}).get('precision', 0):.4f}, Recall: {report.get('1', {}).get('recall', 0):.4f}")
        logger.info(f"Class -1 (SL) Precision: {report.get('-1', {}).get('precision', 0):.4f}, Recall: {report.get('-1', {}).get('recall', 0):.4f}")

        all_metrics.append(report)
        final_model = model
        final_scaler = scaler
    
    # Aggregate metrics from all folds
    avg_metrics = {
        'accuracy': np.mean([accuracy_score(y.iloc[test_index], final_model.predict(scaler.transform(X.iloc[test_index]))) for _, test_index in tscv.split(X)]),
        'precision_class_1': np.mean([classification_report(y.iloc[test_index], final_model.predict(scaler.transform(X.iloc[test_index])), output_dict=True, zero_division=0).get('1', {}).get('precision', 0) for _, test_index in tscv.split(X)]),
        'recall_class_1': np.mean([classification_report(y.iloc[test_index], final_model.predict(scaler.transform(X.iloc[test_index])), output_dict=True, zero_division=0).get('1', {}).get('recall', 0) for _, test_index in tscv.split(X)]),
        'num_samples_trained': len(X_train), # From the last fold
    }

    metrics_log_str = ', '.join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
    logger.info(f"📊 [ML Train] Average Walk-Forward Performance: {metrics_log_str}")
    
    return final_model, final_scaler, avg_metrics


def save_ml_model_to_db(model_bundle: Dict[str, Any], model_name: str, metrics: Dict[str, Any]):
    logger.info(f"ℹ️ [DB Save] Saving model bundle '{model_name}'...")
    try:
        model_binary = pickle.dumps(model_bundle)
        metrics_json = json.dumps(metrics)
        with conn.cursor() as db_cur:
            db_cur.execute("""
                INSERT INTO ml_models (model_name, model_data, trained_at, metrics) 
                VALUES (%s, %s, NOW(), %s) 
                ON CONFLICT (model_name) DO UPDATE SET 
                    model_data = EXCLUDED.model_data, 
                    trained_at = NOW(), 
                    metrics = EXCLUDED.metrics;
            """, (model_name, model_binary, metrics_json))
        conn.commit()
        logger.info(f"✅ [DB Save] Model bundle '{model_name}' saved successfully.")
    except Exception as e:
        logger.error(f"❌ [DB Save] Error saving model bundle: {e}"); conn.rollback()

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    logger.info(f"🚀 Starting ADVANCED ML model training script ({BASE_ML_MODEL_NAME})...")
    
    init_db()
    get_binance_client()
    
    # 1. Fetch BTC data once
    fetch_and_cache_btc_data()
    
    # 2. Get all symbols to train
    symbols_to_train = get_validated_symbols(filename='crypto_list.txt')
    if not symbols_to_train:
        logger.critical("❌ [Main] No valid symbols found. Exiting.")
        exit(1)
        
    send_telegram_message(f"🚀 *{BASE_ML_MODEL_NAME} Training Started*\nWill train models for {len(symbols_to_train)} symbols using advanced methods.")
    
    successful_models, failed_models = 0, 0
    for symbol in symbols_to_train:
        logger.info(f"\n--- ⏳ [Main] Starting model training for {symbol} ---")
        try:
            # 3. Fetch data for the current symbol
            df_hist = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
            if df_hist is None or df_hist.empty:
                logger.warning(f"⚠️ [Main] No data for {symbol}, skipping.")
                failed_models += 1
                continue
            
            # 4. Prepare data (features + labels)
            prepared_data = prepare_data_for_ml(df_hist, btc_data_cache, symbol)
            if prepared_data is None:
                failed_models += 1
                continue
            
            X, y, feature_names = prepared_data
            
            # 5. Train and validate using Walk-Forward
            final_model, final_scaler, model_metrics = train_with_walk_forward_validation(X, y)
            
            # 6. Save the model if it's useful
            # We check if the precision for predicting a "win" (class 1) is acceptable
            if final_model and final_scaler and model_metrics.get('precision_class_1', 0) > 0.35:
                model_bundle = {
                    'model': final_model, 
                    'scaler': final_scaler, 
                    'feature_names': feature_names
                }
                model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
                save_ml_model_to_db(model_bundle, model_name, model_metrics)
                successful_models += 1
            else:
                logger.warning(f"⚠️ [Main] Model for {symbol} is not useful (Precision for wins <= 0.35). Discarding.")
                failed_models += 1

        except Exception as e:
            logger.critical(f"❌ [Main] A fatal error occurred for {symbol}: {e}", exc_info=True)
            failed_models += 1
        time.sleep(1) # Small delay

    completion_message = f"""
✅ *{BASE_ML_MODEL_NAME} Training Finished*
- *Successfully trained:* {successful_models} models
- *Failed/Discarded:* {failed_models} models
- *Total symbols:* {len(symbols_to_train)}
    """
    send_telegram_message(completion_message)
    logger.info(completion_message)

    if conn:
        conn.close()
    logger.info("👋 [Main] ML training script finished.")
