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
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from flask import Flask
from threading import Thread
from github import Github, GithubException, Repository

# --- إعدادات أساسية ---
warnings.simplefilter(action='ignore', category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_model_trainer_v7_ichimoku.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
# تم تغيير اسم المسجل ليعكس استخدام إيشيموكو
logger = logging.getLogger('MLTrainer_V7_With_Ichimoku')

# --- تحميل متغيرات البيئة ---
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
    TELEGRAM_TOKEN: Optional[str] = config('TELEGRAM_BOT_TOKEN', default=None)
    CHAT_ID: Optional[str] = config('TELEGRAM_CHAT_ID', default=None)
    GITHUB_TOKEN: Optional[str] = config('GITHUB_TOKEN', default=None)
    GITHUB_REPO: Optional[str] = config('GITHUB_REPO', default=None)
    RESULTS_FOLDER: str = config('RESULTS_FOLDER', default='ml_results')
except Exception as e:
    logger.critical(f"❌ فشل في تحميل المتغيرات البيئية الأساسية: {e}")
    exit(1)

# --- ثوابت الاستراتيجية ---
# تم تغيير اسم النموذج
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V7_With_SR_Ichimoku'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 90
HYPERPARAM_TUNING_TRIALS: int = 5
BTC_SYMBOL = 'BTCUSDT'
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
BTC_CORR_PERIOD: int = 30
TP_ATR_MULTIPLIER: float = 2.0
SL_ATR_MULTIPLIER: float = 1.5
MAX_HOLD_PERIOD: int = 24

# --- متغيرات عامة ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None

# ... (جميع الدوال المساعدة من السكريبت الأصلي تبقى كما هي) ...
# get_github_repo, save_results_to_github, init_db, get_binance_client,
# get_validated_symbols, fetch_historical_data, fetch_sr_levels,
# calculate_sr_features, calculate_features, get_triple_barrier_labels,
# tune_and_train_model

def get_github_repo() -> Optional[Repository]:
    """Initialize and return the GitHub repository object."""
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
    """Saves metrics and the model bundle to the specified GitHub repository."""
    if not repo:
        return
    try:
        commit_message = f"feat: Update results for {symbol} on {datetime.now(timezone.utc).date()}"
        
        metrics_filename = f"{RESULTS_FOLDER}/{symbol}_latest_metrics.json"
        metrics_content = json.dumps(metrics, indent=4)
        try:
            contents = repo.get_contents(metrics_filename, ref="main")
            repo.update_file(contents.path, commit_message, metrics_content, contents.sha, branch="main")
            logger.info(f"✅ [GitHub] Updated metrics for {symbol}")
        except GithubException as e:
            if e.status == 404:
                repo.create_file(metrics_filename, commit_message, metrics_content, branch="main")
                logger.info(f"✅ [GitHub] Created metrics file for {symbol}")
            else:
                raise e
        
        model_filename = f"{RESULTS_FOLDER}/{symbol}_latest_model.pkl"
        if not model_bundle or 'model' not in model_bundle:
            logger.error(f"❌ [GitHub Save] Model bundle for {symbol} is incomplete. Aborting.")
            return
            
        model_bytes = pickle.dumps(model_bundle)
        model_base64_content = base64.b64encode(model_bytes).decode('utf-8')

        if not model_base64_content:
            logger.error(f"❌ [GitHub Save] Base64 encoded model for {symbol} is empty. Aborting upload.")
            return

        try:
            contents = repo.get_contents(model_filename, ref="main")
            repo.update_file(contents.path, commit_message, model_base64_content, contents.sha, branch="main")
            logger.info(f"✅ [GitHub] Updated model for {symbol}")
        except GithubException as e:
            if e.status == 404:
                repo.create_file(model_filename, commit_message, model_base64_content, branch="main")
                logger.info(f"✅ [GitHub] Created model file for {symbol}")
            else:
                raise e
    except Exception as e:
        logger.error(f"❌ [GitHub Save] Failed to save results for {symbol}: {e}", exc_info=True)

def init_db():
    """Initializes the database connection."""
    global conn
    try:
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        conn.autocommit = True
        logger.info("✅ [DB] Database initialized.")
    except Exception as e:
        logger.critical(f"❌ [DB] Database connection failed: {e}"); exit(1)

def get_binance_client():
    """Initializes the Binance client."""
    global client
    try:
        client = Client(API_KEY, API_SECRET)
        logger.info("✅ [Binance] Client initialized.")
    except Exception as e:
        logger.critical(f"❌ [Binance] Client initialization failed: {e}"); exit(1)

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    if not client: return []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        if not os.path.exists(file_path):
            logger.error(f"❌ Symbol list file not found at {file_path}")
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        formatted_symbols = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        exchange_info = client.get_exchange_info()
        active_symbols = {s['symbol'] for s in exchange_info['symbols'] if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'}
        validated_list = sorted(list(formatted_symbols.intersection(active_symbols)))
        logger.info(f"✅ [Symbol Validation] Will train models for {len(validated_list)} validated symbols.")
        return validated_list
    except Exception as e:
        logger.error(f"❌ [Symbol Validation] Error: {e}", exc_info=True)
        return []
        
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        start_str = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna()
    except Exception as e:
        logger.error(f"❌ [Data] Error fetching {symbol}: {e}"); return None

def fetch_sr_levels(symbol: str, db_conn: psycopg2.extensions.connection) -> pd.DataFrame:
    if not db_conn: return pd.DataFrame()
    query = "SELECT level_price, level_type, score FROM support_resistance_levels WHERE symbol = %s"
    try:
        with db_conn.cursor() as cur:
            cur.execute(query, (symbol,))
            levels = cur.fetchall()
            if not levels: return pd.DataFrame()
            df_levels = pd.DataFrame([dict(row) for row in levels])
            df_levels['score'] = pd.to_numeric(df_levels['score'], errors='coerce').fillna(0)
            return df_levels
    except Exception as e:
        logger.error(f"❌ [DB] Error fetching S/R levels for {symbol}: {e}")
        return pd.DataFrame()

# --- START: NEW AND MODIFIED FUNCTIONS ---

def fetch_ichimoku_features(symbol: str, timeframe: str, db_conn: psycopg2.extensions.connection) -> pd.DataFrame:
    """
    N E W   F U N C T I O N
    Fetches pre-calculated Ichimoku features from the database.
    """
    if not db_conn: return pd.DataFrame()
    query = """
        SELECT timestamp, tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
        FROM ichimoku_features 
        WHERE symbol = %s AND timeframe = %s
        ORDER BY timestamp ASC
    """
    try:
        with db_conn.cursor() as cur:
            cur.execute(query, (symbol, timeframe))
            data = cur.fetchall()
            if not data:
                logger.warning(f"⚠️ No Ichimoku data found in DB for {symbol} on {timeframe}.")
                return pd.DataFrame()
            
            df = pd.DataFrame([dict(row) for row in data])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            logger.info(f"✅ Successfully fetched {len(df)} Ichimoku records for {symbol} from DB.")
            return df
    except Exception as e:
        logger.error(f"❌ [DB] Error fetching Ichimoku features for {symbol}: {e}")
        return pd.DataFrame()

def calculate_derived_ichimoku_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    N E W   F U N C T I O N
    Creates meaningful features from raw Ichimoku values.
    """
    close = df['close']
    tenkan = df['tenkan_sen']
    kijun = df['kijun_sen']
    span_a = df['senkou_span_a']
    span_b = df['senkou_span_b']

    # Position of price relative to Ichimoku lines (normalized)
    df['price_vs_tenkan'] = (close - tenkan) / close
    df['price_vs_kijun'] = (close - kijun) / close
    df['tenkan_vs_kijun'] = (tenkan - kijun) / close

    # Position of price relative to the Kumo cloud
    df['price_vs_cloud_top'] = (close - df[['senkou_span_a', 'senkou_span_b']].max(axis=1)) / close
    df['price_vs_cloud_bottom'] = (close - df[['senkou_span_a', 'senkou_span_b']].min(axis=1)) / close
    
    # Kumo cloud thickness (normalized)
    df['cloud_thickness'] = (span_a - span_b).abs() / close

    # Categorical features
    df['tenkan_kijun_cross'] = np.sign(tenkan - kijun).fillna(0)
    df['is_above_cloud'] = np.where(close > span_a, np.where(close > span_b, 1, 0), 0)
    df['is_below_cloud'] = np.where(close < span_a, np.where(close < span_b, 1, 0), 0)

    return df

def calculate_sr_features(df: pd.DataFrame, sr_levels_df: pd.DataFrame) -> pd.DataFrame:
    if sr_levels_df.empty:
        for col in ['dist_to_support', 'score_of_support', 'dist_to_resistance', 'score_of_resistance']:
            df[col] = 0.0
        return df

    supports = sr_levels_df[sr_levels_df['level_type'].str.contains('support|poc|confluence', case=False, na=False)]
    resistances = sr_levels_df[sr_levels_df['level_type'].str.contains('resistance|poc|confluence', case=False, na=False)]
    
    support_levels = supports['level_price'].to_numpy()
    resistance_levels = resistances['level_price'].to_numpy()
    support_scores = pd.Series(supports['score'].values, index=supports['level_price']).to_dict()
    resistance_scores = pd.Series(resistances['score'].values, index=resistances['level_price']).to_dict()

    results = []
    for price in df['close']:
        dist_s, score_s, dist_r, score_r = 1.0, 0.0, 1.0, 0.0
        
        if support_levels.size > 0:
            diffs = price - support_levels
            below_or_at = diffs[diffs >= 0]
            if below_or_at.size > 0:
                nearest_support_price = support_levels[np.argmin(below_or_at)]
                dist_s = (price - nearest_support_price) / price if price > 0 else 0
                score_s = support_scores.get(nearest_support_price, 0.0)
                
        if resistance_levels.size > 0:
            diffs = resistance_levels - price
            above_or_at = diffs[diffs >= 0]
            if above_or_at.size > 0:
                nearest_resistance_price = resistance_levels[np.argmin(above_or_at)]
                dist_r = (nearest_resistance_price - price) / price if price > 0 else 0
                score_r = resistance_scores.get(nearest_resistance_price, 0.0)
                
        results.append((dist_s, score_s, dist_r, score_r))

    df[['dist_to_support', 'score_of_support', 'dist_to_resistance', 'score_of_resistance']] = results
    return df

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    df_calc['rsi'] = 100 - (100 / (1 + rs))
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

def prepare_data_for_ml(df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame, sr_levels: pd.DataFrame, ichimoku_df: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    """
    M O D I F I E D   F U N C T I O N
    Prepares the final DataFrame with all features (including Ichimoku) and labels.
    """
    try:
        # 1. Calculate original features
        df_featured = calculate_features(df_15m)
        df_featured['atr'] = (df_15m['high'] - df_15m['low']).rolling(window=ATR_PERIOD).mean()
        df_featured = calculate_sr_features(df_featured, sr_levels)
        
        df_featured['returns'] = df_featured['close'].pct_change()
        btc_returns = btc_df['close'].pct_change().rename('btc_returns')
        merged_temp = df_featured.join(btc_returns).fillna(0)
        df_featured['btc_correlation'] = merged_temp['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged_temp['btc_returns'])
        
        df_featured['rsi_4h'] = calculate_features(df_4h)['rsi']
        
        # 2. Join Ichimoku data
        if not ichimoku_df.empty:
            df_featured = df_featured.join(ichimoku_df, how='left')
            # 3. Calculate derived features from Ichimoku
            df_featured = calculate_derived_ichimoku_features(df_featured)
        
        # 4. Fill any missing values
        df_featured.fillna(method='ffill', inplace=True)
        df_featured.fillna(method='bfill', inplace=True)
        
        # 5. Generate labels
        df_featured['target'] = get_triple_barrier_labels(df_featured['close'], df_featured['atr'])
        
        # 6. Define all feature columns
        feature_columns = [
            'rsi', 'atr', 'dist_to_support', 'score_of_support', 
            'dist_to_resistance', 'score_of_resistance', 'btc_correlation', 'rsi_4h'
        ]
        
        # Add Ichimoku features if they exist
        ichimoku_feature_cols = [
            'price_vs_tenkan', 'price_vs_kijun', 'tenkan_vs_kijun', 'price_vs_cloud_top',
            'price_vs_cloud_bottom', 'cloud_thickness', 'tenkan_kijun_cross',
            'is_above_cloud', 'is_below_cloud'
        ]
        
        # Only add columns that actually exist in the dataframe
        for col in ichimoku_feature_cols:
            if col in df_featured.columns:
                feature_columns.append(col)

        df_to_clean = df_featured[feature_columns + ['target']].copy()
        df_to_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_cleaned = df_to_clean.dropna()

        if df_cleaned.empty or df_cleaned['target'].nunique() < 2:
            logger.warning("Not enough data or unique targets after cleaning.")
            return None

        X = df_cleaned[feature_columns]
        y = df_cleaned['target']
        
        return X, y, feature_columns
    except Exception as e:
        logger.error(f"Error in data preparation: {e}", exc_info=True)
        return None

def tune_and_train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
            'verbosity': -1, 'boosting_type': 'gbdt', 'class_weight': 'balanced', 'random_state': 42,
            'n_estimators': trial.suggest_int('n_estimators', 200, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 50),
            'max_depth': trial.suggest_int('max_depth', -1, 10),
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

    logger.info("Starting hyperparameter tuning with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=HYPERPARAM_TUNING_TRIALS, show_progress_bar=True)
    
    best_params = study.best_params
    logger.info(f"Best hyperparameters found: {best_params}")
    
    final_scaler = StandardScaler().fit(X)
    X_scaled = final_scaler.transform(X)
    
    final_model = lgb.LGBMClassifier(objective='multiclass', num_class=3, class_weight='balanced', random_state=42, **best_params)
    final_model.fit(X_scaled, y)
    
    return final_model, final_scaler, {'best_hyperparameters': best_params}

def run_training_job():
    """
    M O D I F I E D   F U N C T I O N
    Main function to run the entire training pipeline, now including Ichimoku features.
    """
    logger.info(f"🚀 Starting training job ({BASE_ML_MODEL_NAME})...")
    init_db()
    get_binance_client()
    github_repo = get_github_repo()
    
    symbols_to_train = get_validated_symbols()
    if not symbols_to_train:
        logger.error("No validated symbols found to train. Exiting.")
        return
        
    btc_data = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
    if btc_data is None:
        logger.error("Could not fetch BTC data. Exiting.")
        return

    for symbol in symbols_to_train:
        logger.info(f"\n--- ⏳ [Main] Processing {symbol} ---")
        try:
            df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
            df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
            if df_15m is None or df_4h is None:
                logger.warning(f"Could not fetch historical data for {symbol}. Skipping.")
                continue
            
            sr_levels = fetch_sr_levels(symbol, conn)
            if sr_levels.empty:
                 logger.warning(f"No S/R levels found for {symbol}. Proceeding without them.")

            # Fetch Ichimoku features from the database
            ichimoku_features = fetch_ichimoku_features(symbol, SIGNAL_GENERATION_TIMEFRAME, conn)

            # Pass all dataframes to the preparation function
            prepared_data = prepare_data_for_ml(df_15m, df_4h, btc_data, sr_levels, ichimoku_features)
            if prepared_data is None:
                logger.warning(f"Data preparation failed for {symbol}. Skipping.")
                continue

            X, y, feature_names = prepared_data
            
            if len(X) < 100:
                logger.warning(f"Not enough training samples for {symbol} after preparation ({len(X)}). Skipping.")
                continue

            logger.info(f"Training model for {symbol} with {len(X)} samples and {len(feature_names)} features.")
            logger.debug(f"Features used: {feature_names}")
            training_result = tune_and_train_model(X, y)
            
            if training_result and training_result[0]:
                final_model, final_scaler, model_metrics = training_result
                model_bundle = {'model': final_model, 'scaler': final_scaler, 'feature_names': feature_names}
                
                logger.info(f"Saving results for {symbol} to GitHub...")
                save_results_to_github(github_repo, symbol, model_metrics, model_bundle)
            else:
                logger.error(f"Model training failed for {symbol}.")
                
            gc.collect()
        except Exception as e:
            logger.critical(f"❌ [Main] Critical error during training for {symbol}: {e}", exc_info=True)
        time.sleep(1)
        
    if conn:
        conn.close()
    logger.info("✅ [Main] Training job finished.")

# --- Flask App for Health Check ---
app = Flask(__name__)
@app.route('/')
def health_check():
    return "ML Trainer service is running.", 200

if __name__ == "__main__":
    training_thread = Thread(target=run_training_job)
    training_thread.daemon = True
    training_thread.start()
    
    port = int(os.environ.get("PORT", 10001))
    logger.info(f"Starting health check server on port {port}")
    app.run(host='0.0.0.0', port=port)
