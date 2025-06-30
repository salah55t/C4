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
    GITHUB_REPO: Optional[str] = config('GITHUB_REPO', default=None)
    RESULTS_FOLDER: str = config('RESULTS_FOLDER', default='ml_results')

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
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
BTC_CORR_PERIOD: int = 30

# --- Triple-Barrier Method Parameters ---
TP_ATR_MULTIPLIER: float = 2.0
SL_ATR_MULTIPLIER: float = 1.5
MAX_HOLD_PERIOD: int = 24

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
btc_data_cache: Optional[pd.DataFrame] = None

# --- GitHub Integration Functions ---
def get_github_repo() -> Optional[Repository]:
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
    if not repo: return
    try:
        commit_message = f"feat: Update results for {symbol} on {datetime.now(timezone.utc).date()}"
        
        # --- Save metrics (JSON files are fine as strings) ---
        metrics_filename = f"{RESULTS_FOLDER}/{symbol}_latest_metrics.json"
        metrics_content = json.dumps(metrics, indent=4)
        try:
            contents = repo.get_contents(metrics_filename)
            repo.update_file(contents.path, commit_message, metrics_content, contents.sha)
            logger.info(f"✅ [GitHub] Updated metrics for {symbol}")
        except GithubException as e:
            if e.status == 404:
                repo.create_file(metrics_filename, commit_message, metrics_content)
                logger.info(f"✅ [GitHub] Created metrics file for {symbol}")
            else: raise e
        
        # --- Save model (Pickle file needs base64 encoding) ---
        model_filename = f"{RESULTS_FOLDER}/{symbol}_latest_model.pkl"
        
        model_bytes = pickle.dumps(model_bundle)
        model_base64_content = base64.b64encode(model_bytes).decode('utf-8')

        if not model_base64_content:
            logger.error(f"❌ [GitHub Save] Generated model content for {symbol} is empty. Aborting upload.")
            return

        try:
            contents = repo.get_contents(model_filename)
            repo.update_file(contents.path, commit_message, model_base64_content, contents.sha, branch="main")
            logger.info(f"✅ [GitHub] Updated model for {symbol}")
        except GithubException as e:
            if e.status == 404:
                repo.create_file(model_filename, commit_message, model_base64_content, branch="main")
                logger.info(f"✅ [GitHub] Created model file for {symbol}")
            else: raise e
            
    except Exception as e:
        logger.error(f"❌ [GitHub] Failed to save results for {symbol}: {e}", exc_info=True)


# --- DB & API Functions ---
def init_db():
    global conn
    try:
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        logger.info("✅ [DB] Database initialized.")
    except Exception as e:
        logger.critical(f"❌ [DB] Database connection failed: {e}"); exit(1)

def get_binance_client():
    global client
    try:
        client = Client(API_KEY, API_SECRET)
        logger.info("✅ [Binance] Client initialized.")
    except Exception as e:
        logger.critical(f"❌ [Binance] Client initialization failed: {e}"); exit(1)

# --- NEW FUNCTION ---
def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """
    Reads symbols from a text file and validates them against Binance.
    """
    logger.info(f"ℹ️ [Symbol Validation] Reading symbols from '{filename}' and validating with Binance...")
    if not client:
        logger.error("❌ [Symbol Validation] Binance client is not initialized.")
        return []
    try:
        # Get the absolute path to the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)

        if not os.path.exists(file_path):
            logger.error(f"❌ [Symbol Validation] The file '{filename}' was not found in the directory.")
            return []

        # Read symbols from the file
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read non-empty, non-comment lines and convert to uppercase
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        
        # Ensure all symbols end with USDT
        formatted_symbols = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        
        # Get all actively trading USDT pairs from Binance
        exchange_info = client.get_exchange_info()
        active_symbols = {s['symbol'] for s in exchange_info['symbols'] if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'}
        
        # Find the intersection of symbols from the file and active symbols on Binance
        validated_list = sorted(list(formatted_symbols.intersection(active_symbols)))
        
        logger.info(f"✅ [Symbol Validation] Will train models for {len(validated_list)} validated symbols.")
        return validated_list
    except Exception as e:
        logger.error(f"❌ [Symbol Validation] An error occurred while validating symbols: {e}", exc_info=True)
        return []
        
# --- Data Fetching & Processing ---
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    try:
        start_str = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.dropna()
    except Exception as e:
        logger.error(f"❌ [Data] Error fetching {symbol}: {e}"); return None

def fetch_sr_levels(symbol: str, db_conn: psycopg2.extensions.connection) -> pd.DataFrame:
    query = "SELECT level_price, level_type, score FROM support_resistance_levels WHERE symbol = %s"
    try:
        with db_conn.cursor() as cur:
            cur.execute(query, (symbol,))
            levels = cur.fetchall()
            if not levels: return pd.DataFrame()
            df_levels = pd.DataFrame(levels)
            df_levels['score'] = pd.to_numeric(df_levels['score'], errors='coerce').fillna(0)
            return df_levels
    except Exception as e:
        logger.error(f"❌ [S/R Fetch] Could not fetch S/R levels for {symbol}: {e}")
        if db_conn: db_conn.rollback()
        return pd.DataFrame()

def calculate_sr_features(df: pd.DataFrame, sr_levels_df: pd.DataFrame) -> pd.DataFrame:
    if sr_levels_df.empty:
        for col in ['dist_to_support', 'score_of_support', 'dist_to_resistance', 'score_of_resistance']:
            df[col] = 0.0
        return df

    supports = sr_levels_df[sr_levels_df['level_type'].str.contains('support|poc|confluence', case=False, na=False)]
    resistances = sr_levels_df[sr_levels_df['level_type'].str.contains('resistance|poc|confluence', case=False, na=False)]
    
    for _, row in df.iterrows():
        price = row['close']
        
        sup_dists = supports['level_price'] - price
        res_dists = resistances['level_price'] - price
        
        nearest_sup = supports.loc[sup_dists[sup_dists <= 0].idxmax()] if not sup_dists[sup_dists <= 0].empty else None
        nearest_res = resistances.loc[res_dists[res_dists >= 0].idxmin()] if not res_dists[res_dists >= 0].empty else None
        
        df.loc[row.name, 'dist_to_support'] = (price - nearest_sup['level_price']) / price if nearest_sup is not None else 1.0
        df.loc[row.name, 'score_of_support'] = nearest_sup['score'] if nearest_sup is not None else 0.0
        df.loc[row.name, 'dist_to_resistance'] = (nearest_res['level_price'] - price) / price if nearest_res is not None else 1.0
        df.loc[row.name, 'score_of_resistance'] = nearest_res['score'] if nearest_res is not None else 0.0

    return df

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
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
        
        hit_upper_idx = future_prices[future_prices >= upper_barrier].index.min()
        hit_lower_idx = future_prices[future_prices <= lower_barrier].index.min()
        
        if pd.notna(hit_upper_idx) and pd.notna(hit_lower_idx):
            labels.iloc[i] = 1 if hit_upper_idx < hit_lower_idx else -1
        elif pd.notna(hit_upper_idx):
            labels.iloc[i] = 1
        elif pd.notna(hit_lower_idx):
            labels.iloc[i] = -1
    return labels

def prepare_data_for_ml(df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame, sr_levels: pd.DataFrame, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    logger.info(f"ℹ️ [ML Prep] Preparing data for {symbol}...")
    
    df_featured = calculate_features(df_15m)
    df_featured['atr'] = (df_15m['high'] - df_15m['low']).rolling(window=ATR_PERIOD).mean()
    df_featured = calculate_sr_features(df_featured, sr_levels)
    df_featured['returns'] = df_featured['close'].pct_change()
    merged = df_featured.join(btc_df['close'].pct_change().rename('btc_returns')).fillna(0)
    df_featured['btc_correlation'] = merged['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged['btc_returns'])
    df_featured['rsi_4h'] = calculate_features(df_4h)['rsi']
    df_featured.fillna(method='ffill', inplace=True)
    
    df_featured['target'] = get_triple_barrier_labels(df_featured['close'], df_featured['atr'])
    
    feature_columns = [
        'rsi', 'atr', 'dist_to_support', 'score_of_support', 'dist_to_resistance', 
        'score_of_resistance', 'btc_correlation', 'rsi_4h'
    ]
    
    df_to_clean = df_featured[feature_columns + ['target']].copy()
    for col in feature_columns:
        df_to_clean[col] = pd.to_numeric(df_to_clean[col], errors='coerce')

    df_cleaned = df_to_clean.dropna()
    if df_cleaned.empty or df_cleaned['target'].nunique() < 2:
        logger.warning(f"⚠️ [ML Prep] Not enough valid data for {symbol} after cleaning. Skipping.")
        return None
        
    logger.info(f"📊 [ML Prep] Target distribution for {symbol}:\n{df_cleaned['target'].value_counts(normalize=True)}")
    
    X = df_cleaned[feature_columns]
    y = df_cleaned['target']
    
    if X.select_dtypes(include=['object']).shape[1] > 0:
        bad_cols = X.select_dtypes(include=['object']).columns.tolist()
        logger.critical(f"❌ [FATAL PREP] Object dtypes still exist in final feature matrix for {symbol}: {bad_cols}")
        return None

    return X, y, feature_columns

def tune_and_train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    logger.info("optimizing_hyperparameters [ML Train] Starting hyperparameter optimization...")
    
    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
            'verbosity': -1, 'boosting_type': 'gbdt', 'class_weight': 'balanced', 'random_state': 42,
            'n_estimators': trial.suggest_int('n_estimators', 200, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        }
        tscv = TimeSeriesSplit(n_splits=3)
        train_idx, test_idx = list(tscv.split(X))[-1]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
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
    final_model = lgb.LGBMClassifier(objective='multiclass', num_class=3, class_weight='balanced', random_state=42, **best_params)
    final_model.fit(X_scaled, y)
    
    return final_model, final_scaler, {'best_hyperparameters': best_params}

def run_training_job():
    logger.info(f"🚀 Starting training job ({BASE_ML_MODEL_NAME})...")
    init_db()
    get_binance_client()
    github_repo = get_github_repo()
    
    # --- MODIFIED: Get symbols from crypto_list.txt instead of all symbols ---
    symbols_to_train = get_validated_symbols()
    if not symbols_to_train:
        logger.critical("❌ [Main] No validated symbols found from crypto_list.txt. Exiting training job.")
        return
    
    btc_data = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
    if btc_data is None:
        logger.critical("❌ [Main] Could not fetch BTC data. Exiting."); return

    for symbol in symbols_to_train:
        logger.info(f"\n--- ⏳ [Main] Processing {symbol} ---")
        try:
            df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
            df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
            if df_15m is None or df_4h is None:
                logger.warning(f"⚠️ [Main] Insufficient data for {symbol}. Skipping."); continue
            
            sr_levels = fetch_sr_levels(symbol, conn)
            
            prepared_data = prepare_data_for_ml(df_15m, df_4h, btc_data, sr_levels, symbol)
            if prepared_data is None: continue
                
            X, y, feature_names = prepared_data
            training_result = tune_and_train_model(X, y)
            
            if training_result[0]:
                final_model, final_scaler, model_metrics = training_result
                model_bundle = {'model': final_model, 'scaler': final_scaler, 'feature_names': feature_names}
                save_results_to_github(github_repo, symbol, model_metrics, model_bundle)
            else:
                logger.warning(f"⚠️ [Main] Model training failed for {symbol}.")
            
            gc.collect()

        except Exception as e:
            logger.critical(f"❌ [Main] Critical error for {symbol}: {e}", exc_info=True)
        time.sleep(1)

    logger.info("✅ [Main] Training job finished.")

app = Flask(__name__)
@app.route('/')
def health_check():
    return "ML Trainer service is running.", 200

if __name__ == "__main__":
    training_thread = Thread(target=run_training_job)
    training_thread.daemon = True
    training_thread.start()
    port = int(os.environ.get("PORT", 10001))
    app.run(host='0.0.0.0', port=port)
