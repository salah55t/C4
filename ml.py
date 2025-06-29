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
import base64
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
from github import Github, GithubException, Repository

# ---------------------- Ignore FutureWarnings from Pandas ----------------------
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------------- Setup Logging ----------------------
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_model_trainer_v6.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLTrainer_V6_With_SR')

# ---------------------- Load Environment Variables ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
    TELEGRAM_TOKEN: Optional[str] = config('TELEGRAM_BOT_TOKEN', default=None)
    CHAT_ID: Optional[str] = config('TELEGRAM_CHAT_ID', default=None)
    
    # --- GitHub Configuration ---
    GITHUB_TOKEN: Optional[str] = config('GITHUB_TOKEN', default=None)
    GITHUB_REPO: Optional[str] = config('GITHUB_REPO', default=None) # e.g., 'your-username/your-repo-name'
    RESULTS_FOLDER: str = config('RESULTS_FOLDER', default='ml_results_v6') # Folder within the repo to save results

except Exception as e:
     logger.critical(f"❌ فشل في تحميل المتغيرات البيئية الأساسية: {e}")
     exit(1)

# ---------------------- Constants and Global Variables ----------------------
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V6_With_SR'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 90
HYPERPARAM_TUNING_TRIALS: int = 5
BTC_SYMBOL = 'BTCUSDT'

# --- Indicator & Feature Parameters ---
ADX_PERIOD: int = 14
BBANDS_PERIOD: int = 20
RSI_PERIOD: int = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD: int = 14
EMA_SLOW_PERIOD: int = 200
EMA_FAST_PERIOD: int = 50
BTC_CORR_PERIOD: int = 30
STOCH_RSI_PERIOD: int = 14
STOCH_K: int = 3
STOCH_D: int = 3
REL_VOL_PERIOD: int = 30
RSI_OVERBOUGHT: int = 70
RSI_OVERSOLD: int = 30
STOCH_RSI_OVERBOUGHT: int = 80
STOCH_RSI_OVERSOLD: int = 20

# Triple-Barrier Method Parameters
TP_ATR_MULTIPLIER: float = 2.0
SL_ATR_MULTIPLIER: float = 1.5
MAX_HOLD_PERIOD: int = 24

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
btc_data_cache: Optional[pd.DataFrame] = None

# --- GitHub Integration Functions ---
def get_github_repo() -> Optional[Repository]:
    """Initializes and returns a connection to the specified GitHub repository."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        logger.warning("⚠️ [GitHub] GitHub token or repo not configured. Skipping results upload.")
        return None
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(GITHUB_REPO)
        logger.info(f"✅ [GitHub] Successfully connected to repository: {GITHUB_REPO}")
        return repo
    except Exception as e:
        logger.error(f"❌ [GitHub] Failed to connect to GitHub repository: {e}")
        return None

def save_results_to_github(repo: Repository, symbol: str, metrics: Dict[str, Any], model_bundle: Dict[str, Any]):
    """Saves model metrics and the pickled model file to the GitHub repository."""
    if not repo:
        return

    # 1. Save metrics as a JSON file
    try:
        metrics_filename = f"{RESULTS_FOLDER}/{symbol}_latest_metrics.json"
        metrics_content = json.dumps(metrics, indent=4)
        commit_message_metrics = f"feat: Update training metrics for {symbol} on {datetime.now(timezone.utc).date()}"

        try:
            contents = repo.get_contents(metrics_filename, ref="main")
            repo.update_file(contents.path, commit_message_metrics, metrics_content, contents.sha, branch="main")
            logger.info(f"✅ [GitHub] Updated metrics for {symbol} in {metrics_filename}")
        except GithubException as e:
            if e.status == 404: # Not Found
                repo.create_file(metrics_filename, commit_message_metrics, metrics_content, branch="main")
                logger.info(f"✅ [GitHub] Created metrics file for {symbol} in {metrics_filename}")
            else:
                logger.error(f"❌ [GitHub] GitHub API error while saving metrics: {e}")
    except Exception as e:
        logger.error(f"❌ [GitHub] Failed to process and save metrics for {symbol}: {e}")

    # 2. Save the pickled model file
    try:
        model_filename = f"{RESULTS_FOLDER}/{symbol}_latest_model.pkl"
        model_content_bytes = pickle.dumps(model_bundle)
        commit_message_model = f"feat: Update trained model for {symbol} on {datetime.now(timezone.utc).date()}"

        try:
            contents = repo.get_contents(model_filename, ref="main")
            repo.update_file(contents.path, commit_message_model, model_content_bytes, contents.sha, branch="main")
            logger.info(f"✅ [GitHub] Updated model file for {symbol} in {model_filename}")
        except GithubException as e:
            if e.status == 404: # Not Found
                repo.create_file(model_filename, commit_message_model, model_content_bytes, branch="main")
                logger.info(f"✅ [GitHub] Created model file for {symbol} in {model_filename}")
            else:
                logger.error(f"❌ [GitHub] GitHub API error while saving model: {e}")
    except Exception as e:
        logger.error(f"❌ [GitHub] Failed to process and save model for {symbol}: {e}")


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
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
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

# --- FIXED DTYPE ISSUE ---
def fetch_sr_levels(symbol: str, db_conn: psycopg2.extensions.connection) -> pd.DataFrame:
    """Fetches S/R levels and ensures correct data types."""
    logger.info(f"🔍 [S/R Fetch] Fetching S/R levels for {symbol} from database...")
    query = "SELECT level_price, level_type, score FROM support_resistance_levels WHERE symbol = %s"
    try:
        with db_conn.cursor() as cur:
            cur.execute(query, (symbol,))
            levels = cur.fetchall()
            if not levels:
                logger.warning(f"⚠️ [S/R Fetch] No S/R levels found for {symbol}.")
                return pd.DataFrame()
            df_levels = pd.DataFrame(levels)
            # Ensure 'score' is numeric. Psycopg2 might return Decimal, which pandas can treat as object.
            df_levels['score'] = pd.to_numeric(df_levels['score'])
            logger.info(f"✅ [S/R Fetch] Found {len(df_levels)} levels for {symbol}.")
            return df_levels
    except Exception as e:
        logger.error(f"❌ [S/R Fetch] Could not fetch S/R levels for {symbol}: {e}")
        if db_conn: db_conn.rollback()
        return pd.DataFrame()

# --- ADDED DTYPE CASTING FOR ROBUSTNESS ---
def calculate_sr_features(df: pd.DataFrame, sr_levels_df: pd.DataFrame) -> pd.DataFrame:
    """Engineers features from S/R levels and ensures final columns are float."""
    if sr_levels_df.empty:
        df['dist_to_support'] = 0.0
        df['score_of_support'] = 0.0
        df['dist_to_resistance'] = 0.0
        df['score_of_resistance'] = 0.0
        return df

    supports = sr_levels_df[sr_levels_df['level_type'].str.contains('support|poc|confluence', case=False, na=False)]['level_price'].sort_values().to_numpy()
    resistances = sr_levels_df[sr_levels_df['level_type'].str.contains('resistance|poc|confluence', case=False, na=False)]['level_price'].sort_values().to_numpy()
    
    # Ensure scores are numeric before creating the dictionary
    level_scores = pd.Series(pd.to_numeric(sr_levels_df['score']).values, index=sr_levels_df['level_price']).to_dict()

    def get_sr_info(price):
        dist_support, score_support, dist_resistance, score_resistance = 1.0, 0.0, 1.0, 0.0

        if supports.size > 0:
            idx = np.searchsorted(supports, price, side='right') - 1
            if idx >= 0:
                nearest_support_price = supports[idx]
                dist_support = (price - nearest_support_price) / price if price > 0 else 0
                score_support = level_scores.get(nearest_support_price, 0.0)

        if resistances.size > 0:
            idx = np.searchsorted(resistances, price, side='left')
            if idx < len(resistances):
                nearest_resistance_price = resistances[idx]
                dist_resistance = (nearest_resistance_price - price) / price if price > 0 else 0
                score_resistance = level_scores.get(nearest_resistance_price, 0.0)
        
        return dist_support, score_support, dist_resistance, score_resistance

    results = df['close'].apply(get_sr_info)
    sr_features_df = pd.DataFrame(results.tolist(), index=df.index, columns=['dist_to_support', 'score_of_support', 'dist_to_resistance', 'score_of_resistance'])
    
    # Final explicit cast to float32 for model compatibility
    for col in sr_features_df.columns:
        df[col] = pd.to_numeric(sr_features_df[col], errors='coerce').fillna(0).astype('float32')

    return df

def calculate_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df_patterns = df.copy()
    op, hi, lo, cl = df_patterns['open'], df_patterns['high'], df_patterns['low'], df_patterns['close']
    body = abs(cl - op)
    candle_range = hi - lo
    candle_range[candle_range == 0] = 1e-9
    
    df_patterns['candlestick_pattern'] = 0
    is_doji = (body / candle_range) < 0.05
    df_patterns.loc[is_doji, 'candlestick_pattern'] = 3 # Neutral
    
    return df_patterns

def calculate_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()

    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))

    ema_fast = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df_calc['macd_hist'] = ema_fast - ema_slow

    df_calc['returns'] = df_calc['close'].pct_change()
    merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
    df_calc['btc_correlation'] = merged_df['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    df_calc['hour_of_day'] = df_calc.index.hour
    
    df_calc = calculate_candlestick_patterns(df_calc)

    return df_calc.astype('float32', errors='ignore')

def get_triple_barrier_labels(prices: pd.Series, atr: pd.Series) -> pd.Series:
    labels = pd.Series(0, index=prices.index, dtype='int8')
    for i in tqdm(range(len(prices) - MAX_HOLD_PERIOD), desc="Labeling", leave=False, ncols=100):
        entry_price = prices.iloc[i]
        current_atr = atr.iloc[i]
        if pd.isna(current_atr) or current_atr == 0: continue
        upper_barrier = entry_price + (current_atr * TP_ATR_MULTIPLIER)
        lower_barrier = entry_price - (current_atr * SL_ATR_MULTIPLIER)
        
        future_prices = prices.iloc[i+1 : i+1+MAX_HOLD_PERIOD]
        
        touched_upper = (future_prices >= upper_barrier).cummax()
        touched_lower = (future_prices <= lower_barrier).cummax()

        upper_idx = touched_upper.idxmax() if touched_upper.any() else None
        lower_idx = touched_lower.idxmax() if touched_lower.any() else None

        if upper_idx and lower_idx:
            labels.iloc[i] = 1 if upper_idx <= lower_idx else -1
        elif upper_idx:
            labels.iloc[i] = 1
        elif lower_idx:
            labels.iloc[i] = -1
            
    return labels

def prepare_data_for_ml(df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame, sr_levels: pd.DataFrame, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    logger.info(f"ℹ️ [ML Prep] Preparing data for {symbol}...")
    df_featured = calculate_features(df_15m, btc_df)
    df_featured = calculate_sr_features(df_featured, sr_levels)
    
    df_4h['rsi_4h'] = calculate_features(df_4h, btc_df)['rsi']
    df_featured = df_featured.join(df_4h['rsi_4h'].rename('rsi_4h')).fillna(method='ffill')
    
    df_featured['target'] = get_triple_barrier_labels(df_featured['close'], df_featured['atr'])
    
    feature_columns = [col for col in df_featured.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target', 'returns', 'btc_returns']]
    
    df_cleaned = df_featured.dropna(subset=feature_columns + ['target']).copy()
    df_cleaned = df_cleaned.replace([np.inf, -np.inf], np.nan).dropna()

    if df_cleaned.empty or df_cleaned['target'].nunique() < 2:
        logger.warning(f"⚠️ [ML Prep] Data for {symbol} has insufficient valid data or less than 2 classes. Skipping.")
        return None
        
    logger.info(f"📊 [ML Prep] Target distribution for {symbol}:\n{df_cleaned['target'].value_counts(normalize=True)}")
    X = df_cleaned[feature_columns]
    y = df_cleaned['target']
    return X, y, feature_columns


def tune_and_train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    logger.info("optimizing_hyperparameters [ML Train] Starting hyperparameter optimization...")
    # Simplified objective for faster execution
    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
            'verbosity': -1, 'boosting_type': 'gbdt', 'class_weight': 'balanced', 'random_state': 42,
            'n_estimators': trial.suggest_int('n_estimators', 200, 500, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 30, 100),
            'max_depth': trial.suggest_int('max_depth', 5, 8),
        }
        tscv = TimeSeriesSplit(n_splits=3)
        train_indices, test_indices = list(tscv.split(X))[-1]
        
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(15, verbose=False)])
        preds = model.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        return report.get('1', {}).get('precision', 0)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=HYPERPARAM_TUNING_TRIALS, show_progress_bar=True)
    best_params = study.best_params
    logger.info(f"🏆 [ML Train] Best hyperparameters found: {best_params}")
    
    final_scaler = StandardScaler().fit(X)
    X_scaled = final_scaler.transform(X)
    
    final_model = lgb.LGBMClassifier(objective='multiclass', num_class=3, class_weight='balanced', random_state=42, verbosity=-1, **best_params)
    final_model.fit(X_scaled, y)
    
    final_metrics = { 'best_hyperparameters': json.dumps(best_params) }
    
    logger.info("📊 [ML Train] Final model trained on all available data.")

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
    github_repo = get_github_repo() # Initialize GitHub repo object at the start
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
            
            sr_levels = fetch_sr_levels(symbol, conn)
            
            prepared_data = prepare_data_for_ml(df_15m, df_4h, btc_data_cache, sr_levels, symbol)
            del df_15m, df_4h, sr_levels; gc.collect()

            if prepared_data is None:
                failed_models += 1; continue
            X, y, feature_names = prepared_data
            
            training_result = tune_and_train_model(X, y)
            if not all(training_result):
                 logger.warning(f"⚠️ [Main] فشل تدريب النموذج لـ {symbol}."); failed_models += 1
                 del X, y, prepared_data; gc.collect()
                 continue
            final_model, final_scaler, model_metrics = training_result
            
            # Use a dummy metric check for now, as full evaluation was simplified
            if final_model and final_scaler:
                model_bundle = {'model': final_model, 'scaler': final_scaler, 'feature_names': feature_names}
                model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
                save_ml_model_to_db(model_bundle, model_name, model_metrics)
                # --- Save results to GitHub ---
                save_results_to_github(github_repo, symbol, model_metrics, model_bundle)
                successful_models += 1
            else:
                logger.warning(f"⚠️ [Main] النموذج الخاص بـ {symbol} لم يتم إنشاؤه بنجاح."); failed_models += 1
            
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
    return "ML Trainer (with S/R features & GitHub integration) service is running and healthy.", 200

if __name__ == "__main__":
    training_thread = Thread(target=run_training_job)
    training_thread.daemon = True
    training_thread.start()
    
    port = int(os.environ.get("PORT", 10001))
    logger.info(f"🌍 Starting web server on port {port} to keep the service alive...")
    app.run(host='0.0.0.0', port=port)
