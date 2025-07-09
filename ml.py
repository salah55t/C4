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
        logging.FileHandler('ml_model_trainer_v8.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLTrainer_V8_Momentum')

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
# تم تحديث اسم النموذج ليعكس الميزات الجديدة
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V8_With_Momentum'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 90
HYPERPARAM_TUNING_TRIALS: int = 15
BTC_SYMBOL = 'BTCUSDT'

# --- Indicator & Feature Parameters ---
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
EMA_SLOW_PERIOD: int = 200
EMA_FAST_PERIOD: int = 50
BTC_CORR_PERIOD: int = 30
REL_VOL_PERIOD: int = 30
# New Momentum and Velocity Parameters
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

# --- دوال جلب ومعالجة البيانات ---
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    try:
        start_dt = datetime.now(timezone.utc) - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        numeric_cols = {'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'float32'}
        df = df.astype(numeric_cols)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna()
    except Exception as e:
        logger.error(f"❌ [Data] خطأ أثناء جلب البيانات لـ {symbol} على إطار {interval}: {e}"); return None

def fetch_and_cache_btc_data():
    global btc_data_cache
    logger.info("ℹ️ [BTC Data] جاري جلب بيانات البيتكوين وتخزينها...")
    btc_data_cache = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
    if btc_data_cache is None:
        logger.critical("❌ [BTC Data] فشل جلب بيانات البيتكوين."); exit(1)
    btc_data_cache['btc_returns'] = btc_data_cache['close'].pct_change()


def calculate_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to calculate all technical analysis features for the model.
    Includes new features: Momentum (ROC), Velocity (ROC Acceleration), and Market Direction (EMA Slope).
    """
    df_calc = df.copy()

    # --- Existing Features ---
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
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
    
    merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
    df_calc['btc_correlation'] = df_calc['close'].pct_change().rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    
    # --- ✨ New Features: Momentum, Velocity, and Market Direction ---
    
    # 1. Momentum (Rate of Change)
    df_calc[f'roc_{MOMENTUM_PERIOD}'] = (df_calc['close'] / df_calc['close'].shift(MOMENTUM_PERIOD) - 1) * 100
    
    # 2. Velocity (Acceleration of Momentum)
    df_calc['roc_acceleration'] = df_calc[f'roc_{MOMENTUM_PERIOD}'].diff()
    
    # 3. Market Direction (Short-term EMA Slope)
    ema_slope = df_calc['close'].ewm(span=EMA_SLOPE_PERIOD, adjust=False).mean()
    df_calc[f'ema_slope_{EMA_SLOPE_PERIOD}'] = (ema_slope - ema_slope.shift(1)) / ema_slope.shift(1).replace(0, 1e-9) * 100

    df_calc['hour_of_day'] = df_calc.index.hour

    return df_calc.astype('float32', errors='ignore')


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
    df_featured = calculate_features(df_15m, btc_df)
    
    # --- MTF Features ---
    delta_4h = df_4h['close'].diff()
    gain_4h = delta_4h.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss_4h = -delta_4h.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_4h['rsi_4h'] = 100 - (100 / (1 + (gain_4h / loss_4h.replace(0, 1e-9))))
    ema_fast_4h = df_4h['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    df_4h['price_vs_ema50_4h'] = (df_4h['close'] / ema_fast_4h) - 1
    
    mtf_features = df_4h[['rsi_4h', 'price_vs_ema50_4h']]
    df_featured = df_featured.join(mtf_features)
    df_featured[['rsi_4h', 'price_vs_ema50_4h']] = df_featured[['rsi_4h', 'price_vs_ema50_4h']].fillna(method='ffill')
    
    # --- Target Labeling ---
    df_featured['target'] = get_triple_barrier_labels(df_featured['close'], df_featured['atr'])
    
    # --- ✨ Updated Feature List ---
    feature_columns = [
        'rsi', 'adx', 'atr', 'relative_volume', 'hour_of_day',
        'price_vs_ema50', 'price_vs_ema200', 'btc_correlation',
        'rsi_4h', 'price_vs_ema50_4h',
        # New Features
        f'roc_{MOMENTUM_PERIOD}', 
        'roc_acceleration', 
        f'ema_slope_{EMA_SLOPE_PERIOD}'
    ]
    
    df_cleaned = df_featured.dropna(subset=feature_columns + ['target']).copy()
    
    # Replace any remaining infinite values with NaN and then drop them
    df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cleaned.dropna(subset=feature_columns, inplace=True)

    if df_cleaned.empty or df_cleaned['target'].nunique() < 2:
        logger.warning(f"⚠️ [ML Prep] Data for {symbol} has less than 2 classes after cleaning. Skipping.")
        return None
        
    logger.info(f"📊 [ML Prep] Target distribution for {symbol}:\n{df_cleaned['target'].value_counts(normalize=True)}")
    X = df_cleaned[feature_columns]
    y = df_cleaned['target']
    return X, y, feature_columns


def tune_and_train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    logger.info(f"optimizing_hyperparameters [ML Train] Starting hyperparameter optimization...")

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
            'verbosity': -1, 'boosting_type': 'gbdt', 'class_weight': 'balanced',
            'random_state': 42,
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
            X_train_scaled_df = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
            X_test_scaled_df = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train_scaled_df, y_train,
                      eval_set=[(X_test_scaled_df, y_test)],
                      callbacks=[lgb.early_stopping(20, verbose=False)])
            
            y_pred = model.predict(X_test_scaled_df)
            all_preds.extend(y_pred)
            all_true.extend(y_test)

        report = classification_report(all_true, all_preds, output_dict=True, zero_division=0)
        return report.get('1', {}).get('precision', 0)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=HYPERPARAM_TUNING_TRIALS, show_progress_bar=True)
    best_params = study.best_params
    logger.info(f"🏆 [ML Train] Best hyperparameters found: {best_params}")
    
    logger.info("ℹ️ [ML Train] Retraining model with best parameters on all data...")
    final_model_params = {
        'objective': 'multiclass', 'num_class': 3, 'class_weight': 'balanced',
        'random_state': 42, 'verbosity': -1, **best_params
    }
    
    all_preds_final, all_true_final = [], []
    tscv_final = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv_final.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        scaler = StandardScaler()
        X_train_scaled_df = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
        X_test_scaled_df = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

        model = lgb.LGBMClassifier(**final_model_params)
        model.fit(X_train_scaled_df, y_train)
        y_pred = model.predict(X_test_scaled_df)
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
    
    final_scaler = StandardScaler()
    X_scaled_full = pd.DataFrame(final_scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    final_model = lgb.LGBMClassifier(**final_model_params)
    final_model.fit(X_scaled_full, y)
    
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
    logger.info(f"🚀 Starting ADVANCED ML model training job ({BASE_ML_MODEL_NAME})...")
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
    return "ML Trainer (with Momentum features) service is running and healthy.", 200

if __name__ == "__main__":
    training_thread = Thread(target=run_training_job)
    training_thread.daemon = True
    training_thread.start()
    
    port = int(os.environ.get("PORT", 10001))
    logger.info(f"🌍 Starting web server on port {port} to keep the service alive...")
    app.run(host='0.0.0.0', port=port)
