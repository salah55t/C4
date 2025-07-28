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
import optuna
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
# --- الإضافات الجديدة لاختيار الميزات ---
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from tqdm import tqdm
from flask import Flask
from threading import Thread

# ---------------------- تجاهل التحذيرات المستقبلية من Pandas ----------------------
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_model_trainer_v12.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLTrainer_V12_Structured_Features')

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
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V12_Structured_Features'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 90
HYPERPARAM_TUNING_TRIALS: int = 5
BTC_SYMBOL = 'BTCUSDT'
K_BEST_FEATURES_PERCENTAGE = 0.9 # النسبة المئوية للميزات التي سيتم اختيارها بواسطة SelectKBest

# --- Indicator & Feature Parameters ---
ATR_PERIOD: int = 14
RSI_PERIOD: int = 14 # For 4h RSI calculation
EMA_FAST_PERIOD: int = 50
EMA_SLOW_PERIOD: int = 200
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
    except (psycopg2.InterfaceError, psycopg2.OperationalError):
        logger.error(f"❌ [DB Keep-Alive] انقطع اتصال قاعدة البيانات. محاولة إعادة الاتصال...")
        if conn: conn.close()
        init_db()

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

# --- دالة تحسين استهلاك الذاكرة ---
def optimize_memory_usage(df: pd.DataFrame, log_prefix: str = "") -> pd.DataFrame:
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
        
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base', 'taker_buy_quote']
        df = df[required_cols]
        
        numeric_cols = {col: 'float' for col in required_cols if col != 'timestamp'}
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

def calculate_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    # Flow of Funds Index (FOFI)
    df['fofi'] = df['taker_buy_quote'] / df['quote_volume'].replace(0, 1e-9)
    # Delta of Delta
    df['delta_of_delta'] = df['taker_buy_base'].diff().diff()
    # Bid-Ask Spread Proxy (%)
    df['spread_proxy_percent'] = (df['high'] - df['low']) / df['close'].replace(0, 1e-9) * 100
    return df

def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    # Williams %R
    highest_high = df['high'].rolling(window=14).max()
    lowest_low = df['low'].rolling(window=14).min()
    df['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low).replace(0, 1e-9)
    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    df['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    # Rate of Change (ROC)
    df[f'roc_{MOMENTUM_PERIOD}'] = (df['close'] / df['close'].shift(MOMENTUM_PERIOD) - 1) * 100
    # EMA Slope
    ema_slope = df['close'].ewm(span=EMA_SLOPE_PERIOD, adjust=False).mean()
    df[f'ema_slope_{EMA_SLOPE_PERIOD}'] = (ema_slope - ema_slope.shift(1)) / ema_slope.shift(1).replace(0, 1e-9) * 100
    # Bollinger Bands Position
    bb_period = 20
    bb_middle = df['close'].rolling(window=bb_period).mean()
    bb_std = df['close'].rolling(window=bb_period).std()
    df['bb_position'] = (df['close'] - (bb_middle - (bb_std * 2))) / ((bb_middle + (bb_std * 2)) - (bb_middle - (bb_std * 2))).replace(0, 1e-9)
    return df

def calculate_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_of_week'] = df.index.dayofweek
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    return df

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

def prepare_data_for_ml(df_15m: pd.DataFrame, df_4h: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    logger.info(f"ℹ️ [ML Prep] Preparing data...")
    
    # --- Feature Calculation ---
    df_featured = calculate_technical_features(df_15m)
    df_featured = calculate_microstructure_features(df_featured)
    df_featured = calculate_temporal_features(df_featured)
    
    # --- MTF & Other Features ---
    delta_4h = df_4h['close'].diff()
    gain_4h = delta_4h.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss_4h = -delta_4h.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_4h['rsi_4h'] = 100 - (100 / (1 + (gain_4h / loss_4h.replace(0, 1e-9))))
    
    mtf_features = df_4h[['rsi_4h']]
    df_featured = df_featured.join(mtf_features).fillna(method='ffill')
    
    df_featured = optimize_memory_usage(df_featured, log_prefix="After Features")
    
    # --- Target Labeling ---
    df_featured['target'] = get_triple_barrier_labels(df_featured['close'], df_featured['atr'])
    
    # --- Final Structured Feature List ---
    feature_columns = [
        # Micro-Structure Features (available)
        'fofi', 'delta_of_delta', 'spread_proxy_percent',
        
        # Technical Clean Features
        'williams_r', 'rsi_4h', 'atr', f'roc_{MOMENTUM_PERIOD}', f'ema_slope_{EMA_SLOPE_PERIOD}', 'bb_position',
        
        # Temporal Features
        'hour_sin', 'hour_cos', 'day_of_week', 'month_sin', 'month_cos'
    ]
    
    df_cleaned = df_featured.dropna(subset=feature_columns + ['target']).copy()
    df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cleaned.dropna(subset=feature_columns, inplace=True)

    if df_cleaned.empty or df_cleaned['target'].nunique() < 2:
        logger.warning(f"⚠️ [ML Prep] Data has less than 2 classes after cleaning. Skipping.")
        return None
        
    logger.info(f"📊 [ML Prep] Target distribution:\n{df_cleaned['target'].value_counts(normalize=True)}")
    X = df_cleaned[feature_columns]
    y = df_cleaned['target']

    # --- بداية قسم اختيار الميزات (Feature Selection) ---
    logger.info(f"🔍 [Feature Selection] Starting feature selection process...")
    logger.info(f"Initial number of features: {X.shape[1]}")

    # 1. VarianceThreshold: إزالة الميزات ذات التباين الصفري
    selector_var = VarianceThreshold()
    X_var = selector_var.fit_transform(X)
    
    # الاحتفاظ فقط بالأعمدة التي لم يتم حذفها
    cols_retained_var = X.columns[selector_var.get_support()]
    X = pd.DataFrame(X_var, index=X.index, columns=cols_retained_var)
    logger.info(f"Features after VarianceThreshold: {X.shape[1]}")

    # 2. SelectKBest: اختيار أفضل الميزات بناءً على mutual_info_classif
    # التأكد من وجود ميزات كافية لتطبيق SelectKBest
    if X.shape[1] > 1:
        # تحديد عدد الميزات المراد اختيارها (k)
        k_val = max(1, int(X.shape[1] * K_BEST_FEATURES_PERCENTAGE))
        logger.info(f"Selecting top {k_val} features using SelectKBest ({K_BEST_FEATURES_PERCENTAGE*100}%)...")
        
        selector_kbest = SelectKBest(mutual_info_classif, k=k_val)
        X_kbest = selector_kbest.fit_transform(X, y)
        
        # الاحتفاظ فقط بالأعمدة التي تم اختيارها
        cols_retained_kbest = X.columns[selector_kbest.get_support()]
        X = pd.DataFrame(X_kbest, index=X.index, columns=cols_retained_kbest)
        logger.info(f"Features after SelectKBest: {X.shape[1]}")
    else:
        logger.warning("⚠️ [Feature Selection] Skipping SelectKBest as there is only 1 feature left.")

    # تحديث قائمة أسماء الميزات النهائية
    final_feature_columns = X.columns.tolist()
    logger.info(f"✅ [Feature Selection] Final selected features ({len(final_feature_columns)}): {final_feature_columns}")
    # --- نهاية قسم اختيار الميزات ---

    return X, y, final_feature_columns


def tune_and_train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    logger.info(f"optimizing_hyperparameters [ML Train] Starting hyperparameter optimization...")

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
            'verbosity': -1, 'boosting_type': 'gbdt', 'class_weight': 'balanced', 'random_state': 42,
            'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }

        all_preds, all_true = [], []
        tscv = TimeSeriesSplit(n_splits=4)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], callbacks=[lgb.early_stopping(20, verbose=False)])
            y_pred = model.predict(X_test_scaled)
            all_preds.extend(y_pred)
            all_true.extend(y_test)

        report = classification_report(all_true, all_preds, output_dict=True, zero_division=0)
        return report.get('1', {}).get('precision', 0)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=HYPERPARAM_TUNING_TRIALS, show_progress_bar=True)
    best_params = study.best_params
    logger.info(f"🏆 [ML Train] Best hyperparameters found: {best_params}")
    
    logger.info("ℹ️ [ML Train] Retraining model with best parameters on all data...")
    final_model_params = {'objective': 'multiclass', 'num_class': 3, 'class_weight': 'balanced', 'random_state': 42, 'verbosity': -1, **best_params}
    
    final_scaler = StandardScaler()
    X_scaled_full = pd.DataFrame(final_scaler.fit_transform(X), index=X.index, columns=X.columns)
    final_model = lgb.LGBMClassifier(**final_model_params)
    final_model.fit(X_scaled_full, y)
    
    # Walk-forward validation for final metrics
    all_preds_final, all_true_final = [], []
    tscv_final = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv_final.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
        model = lgb.LGBMClassifier(**final_model_params)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        all_preds_final.extend(y_pred)
        all_true_final.extend(y_test)
        
    final_report = classification_report(all_true_final, all_preds_final, output_dict=True, zero_division=0)
    final_metrics = {
        'accuracy': accuracy_score(all_true_final, all_preds_final),
        'precision_class_1': final_report.get('1', {}).get('precision', 0),
        'recall_class_1': final_report.get('1', {}).get('recall', 0),
        'f1_score_class_1': final_report.get('1', {}).get('f1-score', 0),
        'precision_class_-1': final_report.get('-1', {}).get('precision', 0),
        'num_samples_trained': len(X),
        'best_hyperparameters': json.dumps(best_params)
    }
    
    metrics_log_str = f"Accuracy: {final_metrics['accuracy']:.4f}, P(1): {final_metrics['precision_class_1']:.4f}, R(1): {final_metrics['recall_class_1']:.4f}"
    logger.info(f"📊 [ML Train] Final Walk-Forward Performance: {metrics_log_str}")

    return final_model, final_scaler, final_metrics

def save_ml_model_to_db(model_bundle: Dict[str, Any], model_name: str, metrics: Dict[str, Any]):
    logger.info(f"ℹ️ [DB Save] Saving model bundle '{model_name}'...")
    try:
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
        logger.error(f"❌ [DB Save] Error saving model bundle: {e}"); conn.rollback()

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try: requests.post(url, json={'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

def run_training_job():
    logger.info(f"🚀 Starting STRUCTURED ML model training job ({BASE_ML_MODEL_NAME})...")
    init_db()
    get_binance_client()
    
    all_valid_symbols = get_validated_symbols(filename='crypto_list.txt')
    if not all_valid_symbols:
        logger.critical("❌ [Main] لم يتم العثور على رموز صالحة. سيتم الخروج."); return
    
    # No need to check for previously trained models for this specific run
    symbols_to_train = all_valid_symbols
    logger.info(f"ℹ️ [Main] Will attempt to train models for all {len(symbols_to_train)} valid symbols.")
    send_telegram_message(f"🚀 *{BASE_ML_MODEL_NAME} Training Started*\nWill process {len(symbols_to_train)} symbols.")
    
    successful_models, failed_models = 0, 0
    for symbol in symbols_to_train:
        logger.info(f"\n--- ⏳ [Main] بدء تدريب النموذج لـ {symbol} ---")
        try:
            df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
            df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
            
            if df_15m is None or df_15m.empty or df_4h is None or df_4h.empty:
                logger.warning(f"⚠️ [Main] لا توجد بيانات كافية لـ {symbol}, سيتم التجاوز."); failed_models += 1; continue
            
            prepared_data = prepare_data_for_ml(df_15m, df_4h)
            del df_15m, df_4h; gc.collect()

            if prepared_data is None:
                failed_models += 1; continue
            X, y, feature_names = prepared_data
            
            training_result = tune_and_train_model(X, y)
            if not all(training_result):
                 logger.warning(f"⚠️ [Main] فشل تدريب النموذج لـ {symbol}."); failed_models += 1
                 del X, y, prepared_data; gc.collect()
                 continue
            final_model, final_scaler, model_metrics = training_result
            
            if final_model and final_scaler and model_metrics.get('precision_class_1', 0) > 0.35:
                model_bundle = {'model': final_model, 'scaler': final_scaler, 'feature_names': feature_names}
                model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
                save_ml_model_to_db(model_bundle, model_name, model_metrics)
                successful_models += 1
            else:
                logger.warning(f"⚠️ [Main] النموذج الخاص بـ {symbol} غير مفيد (Precision < 0.35). سيتم تجاهله."); failed_models += 1
            
            del X, y, prepared_data, training_result, final_model, final_scaler, model_metrics; gc.collect()

        except Exception as e:
            logger.critical(f"❌ [Main] حدث خطأ فادح للرمز {symbol}: {e}", exc_info=True); failed_models += 1
            gc.collect()

        keep_db_alive()
        time.sleep(1)

    completion_message = (f"✅ *{BASE_ML_MODEL_NAME} Training Finished*\n"
                        f"- Successfully trained: {successful_models} models\n"
                        f"- Failed/Discarded: {failed_models} models\n"
                        f"- Total processed: {len(symbols_to_train)}")
    send_telegram_message(completion_message)
    logger.info(completion_message)

    if conn: conn.close()
    logger.info("👋 [Main] انتهت مهمة تدريب النماذج.")

app = Flask(__name__)

@app.route('/')
def health_check():
    return "ML Trainer (Structured Features) service is running and healthy.", 200

if __name__ == "__main__":
    training_thread = Thread(target=run_training_job)
    training_thread.daemon = True
    training_thread.start()
    
    port = int(os.environ.get("PORT", 10001))
    logger.info(f"🌍 Starting web server on port {port} to keep the service alive...")
    app.run(host='0.0.0.0', port=port)
