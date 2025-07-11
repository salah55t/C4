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
        logging.FileHandler('ml_model_trainer_smc_v1.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLTrainer_SMC_V1')

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
BASE_ML_MODEL_NAME: str = 'LightGBM_SMC_V1' # <-- اسم النموذج الجديد
MODEL_FOLDER: str = 'SMC_V1' # <-- مجلد النموذج الجديد
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 90
HYPERPARAM_TUNING_TRIALS: int = 10 # زيادة عدد المحاولات لتحسين أفضل
BTC_SYMBOL = 'BTCUSDT'

# --- مؤشرات فنية (تُستخدم الآن فقط للمساعدة في تحديد الهدف) ---
ATR_PERIOD: int = 14

# --- معلمات طريقة الحاجز الثلاثي (Triple-Barrier) ---
TP_ATR_MULTIPLIER: float = 2.0
SL_ATR_MULTIPLIER: float = 1.5
MAX_HOLD_PERIOD: int = 24 # عدد الشموع كحد أقصى للصفقة

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
btc_data_cache: Optional[pd.DataFrame] = None

# --- دوال الاتصال والتحقق (بدون تغيير) ---
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

# --- ✨ دوال حساب ميزات SMC الجديدة ✨ ---
def find_swing_highs_lows(data: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Finds swing highs and lows using a simple n-period rule."""
    data['sh'] = data['high'][(data['high'].shift(n) < data['high']) & (data['high'].shift(-n) < data['high'])]
    data['sl'] = data['low'][(data['low'].shift(n) > data['low']) & (data['low'].shift(-n) > data['low'])]
    return data

def identify_order_blocks(data: pd.DataFrame) -> pd.DataFrame:
    """Identifies potential bullish and bearish order blocks."""
    data['bullish_ob'] = np.nan
    data['bearish_ob'] = np.nan
    
    is_up_trend = data['close'] > data['open']
    is_down_trend = data['close'] < data['open']
    
    # Bullish OB: Last down candle before an up move
    bullish_ob_indices = data.index[is_down_trend & is_up_trend.shift(-1)]
    if not bullish_ob_indices.empty:
        # For simplicity, we store the range [low, high] as a JSON string
        bullish_ob_values = data.loc[bullish_ob_indices, ['low', 'high']]
        data.loc[bullish_ob_indices, 'bullish_ob'] = bullish_ob_values.to_json(orient='records')

    # Bearish OB: Last up candle before a down move
    bearish_ob_indices = data.index[is_up_trend & is_down_trend.shift(-1)]
    if not bearish_ob_indices.empty:
        bearish_ob_values = data.loc[bearish_ob_indices, ['low', 'high']]
        data.loc[bearish_ob_indices, 'bearish_ob'] = bearish_ob_values.to_json(orient='records')

    return data

def identify_fvg(data: pd.DataFrame) -> pd.DataFrame:
    """Identifies Fair Value Gaps (Imbalances)."""
    # Bullish FVG: gap between high of candle i-1 and low of candle i+1
    bullish_fvg_mask = data['low'] > data['high'].shift(2)
    data.loc[bullish_fvg_mask, 'bullish_fvg'] = data['high'].shift(2)
    data.loc[bullish_fvg_mask, 'bullish_fvg_top'] = data['low']

    # Bearish FVG: gap between low of candle i-1 and high of candle i+1
    bearish_fvg_mask = data['high'] < data['low'].shift(2)
    data.loc[bearish_fvg_mask, 'bearish_fvg'] = data['low'].shift(2)
    data.loc[bearish_fvg_mask, 'bearish_fvg_top'] = data['high']
    
    return data

def calculate_smc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates all SMC features and engineers them for the model.
    """
    df_smc = df.copy()
    
    # 1. Market Structure (Swing Points)
    df_smc = find_swing_highs_lows(df_smc, n=5)
    
    swing_highs = df_smc[df_smc['sh'].notna()]['sh'].dropna()
    swing_lows = df_smc[df_smc['sl'].notna()]['sl'].dropna()
    
    # 2. BOS/CHoCH (Break of Structure / Change of Character)
    df_smc['bos'] = 0
    df_smc['choch'] = 0
    
    if len(swing_highs) > 1 and len(swing_lows) > 1:
        # Bullish BOS: price breaks above the last swing high
        bos_bull_mask = df_smc['high'] > swing_highs.rolling(2).max().shift(1)
        df_smc.loc[bos_bull_mask, 'bos'] = 1
        
        # Bearish BOS: price breaks below the last swing low
        bos_bear_mask = df_smc['low'] < swing_lows.rolling(2).min().shift(1)
        df_smc.loc[bos_bear_mask, 'bos'] = -1

        # Bullish CHoCH: price breaks a previous swing high after a downtrend
        choch_bull_mask = (df_smc['low'].shift(1) < swing_lows.shift(1)) & (df_smc['high'] > swing_highs.shift(1))
        df_smc.loc[choch_bull_mask, 'choch'] = 1
        
        # Bearish CHoCH: price breaks a previous swing low after an uptrend
        choch_bear_mask = (df_smc['high'].shift(1) > swing_highs.shift(1)) & (df_smc['low'] < swing_lows.shift(1))
        df_smc.loc[choch_bear_mask, 'choch'] = -1

    # 3. Order Blocks and FVG
    df_smc = identify_order_blocks(df_smc)
    df_smc = identify_fvg(df_smc)

    # 4. Feature Engineering from SMC concepts
    df_smc['is_in_bullish_fvg'] = ((df_smc['low'] < df_smc['bullish_fvg_top'].ffill()) & (df_smc['high'] > df_smc['bullish_fvg'].ffill())).astype(int)
    df_smc['is_in_bearish_fvg'] = ((df_smc['low'] < df_smc['bearish_fvg'].ffill()) & (df_smc['high'] > df_smc['bearish_fvg_top'].ffill())).astype(int)
    
    # Distance to nearest OB/FVG
    last_price = df_smc['close']
    
    # Use ffill to get the last known OB/FVG level
    bull_ob_level = pd.Series(df_smc['bullish_ob'].ffill().apply(lambda x: json.loads(x)[0]['low'] if pd.notna(x) else np.nan))
    bear_ob_level = pd.Series(df_smc['bearish_ob'].ffill().apply(lambda x: json.loads(x)[0]['high'] if pd.notna(x) else np.nan))
    bull_fvg_level = df_smc['bullish_fvg'].ffill()
    bear_fvg_level = df_smc['bearish_fvg'].ffill()

    df_smc['dist_to_bull_ob'] = (last_price - bull_ob_level) / last_price
    df_smc['dist_to_bear_ob'] = (bear_ob_level - last_price) / last_price
    df_smc['dist_to_bull_fvg'] = (last_price - bull_fvg_level) / last_price
    df_smc['dist_to_bear_fvg'] = (bear_fvg_level - last_price) / last_price

    # Liquidity (simple version: number of recent swing points)
    df_smc['liquidity_highs_nearby'] = df_smc['sh'].rolling(window=20, min_periods=1).count()
    df_smc['liquidity_lows_nearby'] = df_smc['sl'].rolling(window=20, min_periods=1).count()
    
    # Add ATR for labeling and context
    high_low = df_smc['high'] - df_smc['low']
    high_close = (df_smc['high'] - df_smc['close'].shift()).abs()
    low_close = (df_smc['low'] - df_smc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_smc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    return df_smc

def get_triple_barrier_labels(prices: pd.Series, atr: pd.Series) -> pd.Series:
    labels = pd.Series(0, index=prices.index, dtype=int)
    for i in tqdm(range(len(prices) - MAX_HOLD_PERIOD), desc="Labeling", leave=False):
        entry_price = prices.iloc[i]
        current_atr = atr.iloc[i]
        if pd.isna(current_atr) or current_atr == 0: continue
        
        upper_barrier = entry_price + (current_atr * TP_ATR_MULTIPLIER)
        lower_barrier = entry_price - (current_atr * SL_ATR_MULTIPLIER)
        
        for j in range(1, MAX_HOLD_PERIOD + 1):
            if i + j >= len(prices): break
            future_price = prices.iloc[i + j]
            if future_price >= upper_barrier:
                labels.iloc[i] = 1  # Buy signal
                break
            if future_price <= lower_barrier:
                # We are only predicting buys, so stop loss hit is a neutral event for training
                labels.iloc[i] = 0 # Neutral signal
                break
    return labels

def prepare_data_for_ml(df_15m: pd.DataFrame, df_4h: pd.DataFrame, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    logger.info(f"ℹ️ [ML Prep] Preparing SMC data for {symbol}...")
    
    # Calculate features on both timeframes
    df_featured_15m = calculate_smc_features(df_15m)
    df_featured_4h = calculate_smc_features(df_4h)
    df_featured_4h = df_featured_4h.rename(columns=lambda c: f"{c}_4h")
    
    # Join features
    df_combined = df_featured_15m.join(df_featured_4h, how='outer')
    df_combined.ffill(inplace=True)
    
    # --- Target Labeling (only on 15m timeframe) ---
    df_combined['target'] = get_triple_barrier_labels(df_combined['close'], df_combined['atr'])
    
    # --- ✨ Updated SMC Feature List ---
    feature_columns = [
        'bos', 'choch', 'is_in_bullish_fvg', 'is_in_bearish_fvg',
        'dist_to_bull_ob', 'dist_to_bear_ob', 'dist_to_bull_fvg', 'dist_to_bear_fvg',
        'liquidity_highs_nearby', 'liquidity_lows_nearby', 'atr',
        # 4H features
        'bos_4h', 'choch_4h', 'is_in_bullish_fvg_4h', 'is_in_bearish_fvg_4h',
        'dist_to_bull_ob_4h', 'dist_to_bear_ob_4h', 'dist_to_bull_fvg_4h', 'dist_to_bear_fvg_4h',
        'liquidity_highs_nearby_4h', 'liquidity_lows_nearby_4h', 'atr_4h'
    ]
    
    df_cleaned = df_combined.dropna(subset=feature_columns + ['target']).copy()
    df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cleaned.dropna(subset=feature_columns, inplace=True)

    # We only want to predict 'buy' signals (label 1) vs. 'do nothing' (label 0)
    df_cleaned = df_cleaned[df_cleaned['target'].isin([0, 1])]

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
        # We are predicting Buy (1) vs No-Buy (0), so it's a binary classification
        params = {
            'objective': 'binary', 'metric': 'auc',
            'verbosity': -1, 'boosting_type': 'gbdt',
            'is_unbalance': True, # Handles imbalanced data
            'random_state': 42,
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }

        all_preds, all_true = [], []
        tscv = TimeSeriesSplit(n_splits=4)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train_scaled, y_train,
                      eval_set=[(X_test_scaled, y_test)],
                      callbacks=[lgb.early_stopping(25, verbose=False)])
            
            y_pred = model.predict(X_test_scaled)
            all_preds.extend(y_pred)
            all_true.extend(y_test)

        report = classification_report(all_true, all_preds, output_dict=True, zero_division=0)
        # Optimize for precision of the 'buy' class (label 1)
        return report.get('1', {}).get('precision', 0)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=HYPERPARAM_TUNING_TRIALS, show_progress_bar=True)
    best_params = study.best_params
    logger.info(f"🏆 [ML Train] Best hyperparameters found: {best_params}")
    
    logger.info("ℹ️ [ML Train] Retraining model with best parameters on all data...")
    final_model_params = {
        'objective': 'binary', 'metric': 'auc', 'is_unbalance': True,
        'random_state': 42, 'verbosity': -1, **best_params
    }
    
    final_scaler = StandardScaler()
    X_scaled_full = final_scaler.fit_transform(X)
    
    final_model = lgb.LGBMClassifier(**final_model_params)
    final_model.fit(X_scaled_full, y)
    
    # Final evaluation using walk-forward
    all_preds_final, all_true_final = [], []
    tscv_final = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv_final.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

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
        'num_samples_trained': len(X),
        'best_hyperparameters': json.dumps(best_params)
    }
    
    metrics_log_str = f"Accuracy: {final_metrics['accuracy']:.4f}, P(1): {final_metrics['precision_class_1']:.4f}, R(1): {final_metrics['recall_class_1']:.4f}"
    logger.info(f"📊 [ML Train] Final Walk-Forward Performance: {metrics_log_str}")

    return final_model, final_scaler, final_metrics

def save_ml_model_to_folder(model_bundle: Dict[str, Any], model_name: str, metrics: Dict[str, Any]):
    """Saves the model bundle to a local folder instead of DB."""
    logger.info(f"💾 [File Save] Saving model bundle '{model_name}' to local folder...")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir_path = os.path.join(script_dir, MODEL_FOLDER)
        if not os.path.exists(model_dir_path):
            os.makedirs(model_dir_path)
        
        model_path = os.path.join(model_dir_path, f"{model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_bundle, f)
            
        metrics_path = os.path.join(model_dir_path, f"{model_name}_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
            
        logger.info(f"✅ [File Save] Model bundle '{model_name}' saved successfully to '{model_dir_path}'.")
    except Exception as e:
        logger.error(f"❌ [File Save] Error saving model bundle: {e}")

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try: requests.post(url, json={'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

def run_training_job():
    logger.info(f"🚀 Starting SMC ML model training job ({BASE_ML_MODEL_NAME})...")
    get_binance_client()
    
    all_valid_symbols = get_validated_symbols(filename='crypto_list.txt')
    if not all_valid_symbols:
        logger.critical("❌ [Main] لم يتم العثور على رموز صالحة. سيتم الخروج."); return
    
    # For SMC, it's better to retrain all models to ensure consistency
    symbols_to_train = all_valid_symbols
    
    logger.info(f"ℹ️ [Main] Will attempt to train/retrain SMC models for {len(symbols_to_train)} symbols.")
    send_telegram_message(f"🚀 *{BASE_ML_MODEL_NAME} Training Started*\nWill train models for {len(symbols_to_train)} symbols.")
    
    successful_models, failed_models = 0, 0
    for symbol in symbols_to_train:
        logger.info(f"\n--- ⏳ [Main] بدء تدريب النموذج لـ {symbol} ---")
        try:
            df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
            df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
            
            if df_15m is None or df_15m.empty or df_4h is None or df_4h.empty:
                logger.warning(f"⚠️ [Main] لا توجد بيانات كافية لـ {symbol}, سيتم التجاوز."); failed_models += 1; continue
            
            prepared_data = prepare_data_for_ml(df_15m, df_4h, symbol)
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
            
            # Set a reasonable precision threshold for the buy signal
            if final_model and final_scaler and model_metrics.get('precision_class_1', 0) > 0.45:
                model_bundle = {'model': final_model, 'scaler': final_scaler, 'feature_names': feature_names}
                model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
                # Save to local folder
                save_ml_model_to_folder(model_bundle, model_name, model_metrics)
                successful_models += 1
            else:
                precision = model_metrics.get('precision_class_1', 0)
                logger.warning(f"⚠️ [Main] النموذج الخاص بـ {symbol} غير مفيد (Precision: {precision:.2f} < 0.45). سيتم تجاهله."); failed_models += 1
            
            del X, y, prepared_data, training_result, final_model, final_scaler, model_metrics; gc.collect()

        except Exception as e:
            logger.critical(f"❌ [Main] حدث خطأ فادح للرمز {symbol}: {e}", exc_info=True); failed_models += 1
            gc.collect()
        
        time.sleep(1)

    completion_message = (f"✅ *{BASE_ML_MODEL_NAME} Training Finished*\n"
                        f"- Successfully trained: {successful_models} new models\n"
                        f"- Failed/Discarded: {failed_models} models\n"
                        f"- Processed this run: {len(symbols_to_train)}")
    send_telegram_message(completion_message)
    logger.info(completion_message)

    logger.info("👋 [Main] انتهت مهمة تدريب النماذج.")

app = Flask(__name__)

@app.route('/')
def health_check():
    return "SMC ML Trainer service is running and healthy.", 200

if __name__ == "__main__":
    training_thread = Thread(target=run_training_job)
    training_thread.daemon = True
    training_thread.start()
    
    port = int(os.environ.get("PORT", 10001))
    logger.info(f"🌍 Starting web server on port {port} to keep the service alive...")
    app.run(host='0.0.0.0', port=port)
