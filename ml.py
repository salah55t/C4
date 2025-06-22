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

# ---------------------- تجاهل التحذيرات المستقبلية من Pandas ----------------------
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
optuna.logging.set_verbosity(optuna.logging.WARNING)
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
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V5'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 120
HYPERPARAM_TUNING_TRIALS: int = 5
BTC_SYMBOL = 'BTCUSDT'
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
TP_ATR_MULTIPLIER: float = 2.0
SL_ATR_MULTIPLIER: float = 1.5
MAX_HOLD_PERIOD: int = 24

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
btc_data_cache: Optional[pd.DataFrame] = None

# --- دوال الاتصال والتحقق ---

def get_db_connection(force_reconnect: bool = False) -> Optional[psycopg2.extensions.connection]:
    """
    يوفر اتصالاً صالحًا بقاعدة البيانات، ويعيد الاتصال إذا لزم الأمر.
    """
    global conn
    if force_reconnect or conn is None or conn.closed != 0:
        if conn:
            try:
                conn.close()
            except psycopg2.Error:
                pass  # تجاهل الأخطاء عند إغلاق اتصال ميت بالفعل

        try:
            logger.info("ℹ️ [DB] محاولة (إعادة) الاتصال بقاعدة البيانات...")
            conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ml_models (
                        id SERIAL PRIMARY KEY, model_name TEXT NOT NULL UNIQUE,
                        model_data BYTEA NOT NULL, trained_at TIMESTAMP DEFAULT NOW(), metrics JSONB );
                """)
            conn.commit()
            logger.info("✅ [DB] تم إنشاء اتصال قاعدة البيانات والتحقق من الجدول بنجاح.")
        except psycopg2.OperationalError as e:
            logger.critical(f"❌ [DB] لا يمكن إنشاء اتصال بقاعدة البيانات: {e}")
            conn = None
        except Exception as e:
            logger.critical(f"❌ [DB] حدث خطأ غير متوقع أثناء تهيئة قاعدة البيانات: {e}")
            conn = None
    return conn


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
        logger.error(f"❌ [Data] خطأ أثناء جلب البيانات لـ {symbol} على إطار {interval}: {e}"); return None

def fetch_and_cache_btc_data():
    global btc_data_cache
    logger.info("ℹ️ [BTC Data] جاري جلب بيانات البيتكوين وتخزينها...")
    btc_data_cache = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
    if btc_data_cache is None:
        logger.critical("❌ [BTC Data] فشل جلب بيانات البيتكوين."); exit(1)
    btc_data_cache['btc_returns'] = btc_data_cache['close'].pct_change()

def calculate_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df_patterns = df.copy()
    op, hi, lo, cl = df_patterns['open'], df_patterns['high'], df_patterns['low'], df_patterns['close']
    body = abs(cl - op)
    candle_range = hi - lo
    candle_range[candle_range == 0] = 1e-9
    upper_wick = hi - pd.concat([op, cl], axis=1).max(axis=1)
    lower_wick = pd.concat([op, cl], axis=1).min(axis=1) - lo
    
    df_patterns['candlestick_pattern'] = 0
    is_bullish_marubozu = (cl > op) & (body / candle_range > 0.95)
    is_bearish_marubozu = (op > cl) & (body / candle_range > 0.95)
    is_bullish_engulfing = (cl.shift(1) < op.shift(1)) & (cl > op) & (cl >= op.shift(1)) & (op <= cl.shift(1))
    is_bearish_engulfing = (cl.shift(1) > op.shift(1)) & (cl < op) & (op >= cl.shift(1)) & (cl <= op.shift(1))
    is_hammer = (lower_wick >= body * 2) & (upper_wick < body)
    is_shooting_star = (upper_wick >= body * 2) & (lower_wick < body)
    is_doji = (body / candle_range) < 0.05

    df_patterns.loc[is_doji, 'candlestick_pattern'] = 3
    df_patterns.loc[is_hammer, 'candlestick_pattern'] = 2
    df_patterns.loc[is_shooting_star, 'candlestick_pattern'] = -2
    df_patterns.loc[is_bullish_engulfing, 'candlestick_pattern'] = 1
    df_patterns.loc[is_bearish_engulfing, 'candlestick_pattern'] = -1
    df_patterns.loc[is_bullish_marubozu, 'candlestick_pattern'] = 4
    df_patterns.loc[is_bearish_marubozu, 'candlestick_pattern'] = -4
    return df_patterns

def calculate_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    high_low = df_calc['high'] - df_calc['low']
    tr = pd.concat([high_low, (df_calc['high'] - df_calc['close'].shift()).abs(), (df_calc['low'] - df_calc['close'].shift()).abs()], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    plus_dm = (df_calc['high'].diff() > -df_calc['low'].diff()) & (df_calc['high'].diff() > 0)
    minus_dm = (-df_calc['low'].diff() > df_calc['high'].diff()) & (-df_calc['low'].diff() > 0)
    plus_di = 100 * (plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'])
    minus_di = 100 * (minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'])
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    ema_fast = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df_calc['macd_hist'] = macd_line - signal_line
    df_calc['macd_cross'] = np.select([(df_calc['macd_hist'].shift(1) < 0) & (df_calc['macd_hist'] >= 0), (df_calc['macd_hist'].shift(1) > 0) & (df_calc['macd_hist'] <= 0)], [1, -1], default=0)
    sma = df_calc['close'].rolling(window=BBANDS_PERIOD).mean()
    std_dev = df_calc['close'].rolling(window=BBANDS_PERIOD).std()
    df_calc['bb_width'] = ((sma + (std_dev * 2)) - (sma - (std_dev * 2))) / (sma + 1e-9)
    min_rsi = df_calc['rsi'].rolling(window=STOCH_RSI_PERIOD).min()
    max_rsi = df_calc['rsi'].rolling(window=STOCH_RSI_PERIOD).max()
    stoch_rsi_val = (df_calc['rsi'] - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(window=STOCH_K).mean() * 100
    df_calc['stoch_rsi_d'] = df_calc['stoch_rsi_k'].rolling(window=STOCH_D).mean()
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)
    df_calc['market_condition'] = np.select([(df_calc['rsi'] > RSI_OVERBOUGHT) | (df_calc['stoch_rsi_k'] > STOCH_RSI_OVERBOUGHT), (df_calc['rsi'] < RSI_OVERSOLD) | (df_calc['stoch_rsi_k'] < STOCH_RSI_OVERSOLD)], [1, -1], default=0)
    df_calc['price_vs_ema50'] = (df_calc['close'] / df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()) - 1
    df_calc['returns'] = df_calc['close'].pct_change()
    merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
    df_calc['btc_correlation'] = merged_df['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    df_calc['hour_of_day'] = df_calc.index.hour
    df_calc = calculate_candlestick_patterns(df_calc)
    return df_calc

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
            if prices.iloc[i + j] >= upper_barrier:
                labels.iloc[i] = 1; break
            if prices.iloc[i + j] <= lower_barrier:
                labels.iloc[i] = -1; break
    return labels

def prepare_data_for_ml(df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    logger.info(f"ℹ️ [ML Prep] Preparing data for {symbol} with Multi-Timeframe Analysis...")
    df_featured = calculate_features(df_15m, btc_df)
    delta_4h = df_4h['close'].diff()
    gain_4h = delta_4h.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss_4h = -delta_4h.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_4h['rsi_4h'] = 100 - (100 / (1 + (gain_4h / loss_4h.replace(0, 1e-9))))
    df_4h['price_vs_ema50_4h'] = (df_4h['close'] / df_4h['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()) - 1
    df_featured = df_featured.join(df_4h[['rsi_4h', 'price_vs_ema50_4h']]).fillna(method='ffill')
    df_featured['target'] = get_triple_barrier_labels(df_featured['close'], df_featured['atr'])
    feature_columns = ['rsi', 'macd_hist', 'atr', 'relative_volume', 'hour_of_day', 'price_vs_ema50', 'price_vs_ema200', 'btc_correlation', 'stoch_rsi_k', 'stoch_rsi_d', 'macd_cross', 'market_condition', 'bb_width', 'adx', 'candlestick_pattern', 'rsi_4h', 'price_vs_ema50_4h']
    df_cleaned = df_featured.dropna(subset=feature_columns + ['target']).copy()
    if df_cleaned.empty or df_cleaned['target'].nunique() < 2:
        logger.warning(f"⚠️ [ML Prep] Data for {symbol} has less than 2 classes. Skipping."); return None
    logger.info(f"📊 [ML Prep] Target distribution for {symbol}:\n{df_cleaned['target'].value_counts(normalize=True)}")
    return df_cleaned[feature_columns], df_cleaned['target'], feature_columns

def tune_and_train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    logger.info(f"optimizing_hyperparameters [ML Train] Starting hyperparameter optimization...")
    def objective(trial: optuna.trial.Trial) -> float:
        params = {'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss', 'verbosity': -1, 'boosting_type': 'gbdt', 'class_weight': 'balanced', 'random_state': 42, 'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=50), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2), 'num_leaves': trial.suggest_int('num_leaves', 20, 300), 'max_depth': trial.suggest_int('max_depth', 4, 12), 'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0), 'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), 'subsample': trial.suggest_float('subsample', 0.6, 1.0), 'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)}
        all_preds, all_true = [], []
        for train_index, test_index in TimeSeriesSplit(n_splits=5).split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], callbacks=[lgb.early_stopping(30, verbose=False)])
            all_preds.extend(model.predict(X_test_scaled)); all_true.extend(y_test)
        report = classification_report(all_true, all_preds, output_dict=True, zero_division=0)
        return report.get('1', {}).get('precision', 0)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=HYPERPARAM_TUNING_TRIALS, show_progress_bar=True)
    best_params = study.best_params
    logger.info(f"🏆 [ML Train] Best hyperparameters found: {best_params}")

    logger.info("ℹ️ [ML Train] Retraining model with best parameters on all data...")
    final_model_params = {'objective': 'multiclass', 'num_class': 3, 'class_weight': 'balanced', 'random_state': 42, 'verbosity': -1, **best_params}
    all_preds_final, all_true_final = [], []
    for train_index, test_index in TimeSeriesSplit(n_splits=5).split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
        model = lgb.LGBMClassifier(**final_model_params)
        model.fit(X_train_scaled, y_train)
        all_preds_final.extend(model.predict(X_test_scaled)); all_true_final.extend(y_test)
    
    final_report = classification_report(all_true_final, all_preds_final, output_dict=True, zero_division=0)
    final_metrics = {'accuracy': accuracy_score(all_true_final, all_preds_final), 'precision_class_1': final_report.get('1', {}).get('precision', 0), 'recall_class_1': final_report.get('1', {}).get('recall', 0), 'f1_score_class_1': final_report.get('1', {}).get('f1-score', 0), 'precision_class_-1': final_report.get('-1', {}).get('precision', 0), 'num_samples_trained': len(X), 'best_hyperparameters': json.dumps(best_params)}
    
    final_scaler = StandardScaler()
    X_scaled_full = pd.DataFrame(final_scaler.fit_transform(X), columns=X.columns)
    final_model = lgb.LGBMClassifier(**final_model_params)
    final_model.fit(X_scaled_full, y)
    logger.info(f"📊 [ML Train] Final Walk-Forward Performance: Acc: {final_metrics['accuracy']:.4f}, P(1): {final_metrics['precision_class_1']:.4f}")
    return final_model, final_scaler, final_metrics

def save_ml_model_to_db(model_bundle: Dict[str, Any], model_name: str, metrics: Dict[str, Any]):
    local_conn = get_db_connection()
    if not local_conn:
        logger.error(f"❌ [DB Save] No database connection for '{model_name}'. Skipping save.")
        return

    logger.info(f"ℹ️ [DB Save] Saving model bundle '{model_name}'...")
    try:
        with local_conn.cursor() as db_cur:
            db_cur.execute("""
                INSERT INTO ml_models (model_name, model_data, metrics) VALUES (%s, %s, %s) 
                ON CONFLICT (model_name) DO UPDATE SET model_data = EXCLUDED.model_data, 
                trained_at = NOW(), metrics = EXCLUDED.metrics;
            """, (model_name, pickle.dumps(model_bundle), json.dumps(metrics)))
        local_conn.commit()
        logger.info(f"✅ [DB Save] Model bundle '{model_name}' saved successfully.")
    except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
        logger.error(f"❌ [DB Save] Connection lost while saving '{model_name}': {e}. Forcing reconnect.")
        get_db_connection(force_reconnect=True)
    except Exception as e:
        logger.error(f"❌ [DB Save] Error saving model bundle '{model_name}': {e}")
        try:
            if not local_conn.closed:
                local_conn.rollback()
        except psycopg2.Error as db_err:
            logger.error(f"❌ [DB Save] Failed to rollback: {db_err}. Forcing reconnect.")
            get_db_connection(force_reconnect=True)

def check_if_model_exists(model_name: str) -> bool:
    local_conn = get_db_connection()
    if not local_conn:
        logger.error("❌ [DB Check] No DB connection. Assuming model doesn't exist.")
        return False
    try:
        with local_conn.cursor() as cur:
            cur.execute("SELECT 1 FROM ml_models WHERE model_name = %s LIMIT 1;", (model_name,))
            return cur.fetchone() is not None
    except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
        logger.error(f"❌ [DB Check] Connection lost checking for '{model_name}': {e}. Forcing reconnect.")
        get_db_connection(force_reconnect=True)
        return False
    except Exception as e:
        logger.error(f"❌ [DB Check] Error checking for model '{model_name}': {e}")
        return False

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    try: requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

def run_training_job():
    logger.info(f"🚀 Starting ADVANCED ML model training job ({BASE_ML_MODEL_NAME})...")
    get_binance_client()
    fetch_and_cache_btc_data()
    symbols_to_train = get_validated_symbols(filename='crypto_list.txt')
    if not symbols_to_train:
        logger.critical("❌ [Main] No valid symbols. Exiting."); return
        
    send_telegram_message(f"🚀 *{BASE_ML_MODEL_NAME} Training Started*\nWill process {len(symbols_to_train)} symbols.")
    successful_models, failed_models, skipped_models = 0, 0, 0
    
    for symbol in symbols_to_train:
        model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
        if check_if_model_exists(model_name):
            logger.info(f"⏭️ [Main] Model for {symbol} already exists. Skipping.")
            skipped_models += 1; continue
        
        logger.info(f"\n--- ⏳ [Main] Starting training for {symbol} ---")
        try:
            df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
            df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
            if df_15m is None or df_4h is None or df_15m.empty or df_4h.empty:
                logger.warning(f"⚠️ [Main] Not enough data for {symbol}."); failed_models += 1; continue
            
            prepared_data = prepare_data_for_ml(df_15m, df_4h, btc_data_cache, symbol)
            if prepared_data is None:
                failed_models += 1; continue
            X, y, feature_names = prepared_data
            
            training_result = tune_and_train_model(X, y)
            if not all(training_result):
                logger.warning(f"⚠️ [Main] Training failed for {symbol}."); failed_models += 1; continue
            final_model, final_scaler, model_metrics = training_result
            
            if final_model and final_scaler and model_metrics.get('precision_class_1', 0) > 0.35:
                model_bundle = {'model': final_model, 'scaler': final_scaler, 'feature_names': feature_names}
                save_ml_model_to_db(model_bundle, model_name, model_metrics)
                successful_models += 1
            else:
                logger.warning(f"⚠️ [Main] Model for {symbol} not useful (Precision < 0.35)."); failed_models += 1
        except Exception as e:
            logger.critical(f"❌ [Main] Fatal error for {symbol}: {e}", exc_info=True); failed_models += 1
        time.sleep(1)

    completion_message = (f"✅ *{BASE_ML_MODEL_NAME} Training Finished*\n"
                        f"- ✅ Successfully trained: {successful_models}\n"
                        f"- ❌ Failed/Discarded: {failed_models}\n"
                        f"- ⏭️ Already trained (Skipped): {skipped_models}\n"
                        f"- 📊 Total symbols processed: {len(symbols_to_train)}")
    send_telegram_message(completion_message)
    logger.info(completion_message)

    global conn
    if conn and not conn.closed: conn.close()
    logger.info("👋 [Main] ML training job finished.")

app = Flask(__name__)

@app.route('/')
def health_check():
    return "ML Trainer service is running and healthy.", 200

if __name__ == "__main__":
    training_thread = Thread(target=run_training_job)
    training_thread.daemon = True
    training_thread.start()
    port = int(os.environ.get("PORT", 10001))
    logger.info(f"🌍 Starting web server on port {port} to keep the service alive...")
    app.run(host='0.0.0.0', port=port)
