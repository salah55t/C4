import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import pandas_ta as ta
import psycopg2
import pickle
import lightgbm as lgb
from psycopg2.extras import RealDictCursor
from binance.client import Client
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Tuple
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from flask import Flask
from threading import Thread
from sklearn.metrics import accuracy_score, precision_score

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_trainer_sequential.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SequentialCryptoMLTrainer')

# ---------------------- تحميل متغيرات البيئة ----------------------
try:
    API_KEY = config('BINANCE_API_KEY')
    API_SECRET = config('BINANCE_API_SECRET')
    DB_URL = config('DATABASE_URL')
    TELEGRAM_TOKEN = config('TELEGRAM_BOT_TOKEN', default=None)
    CHAT_ID = config('TELEGRAM_CHAT_ID', default=None)
except Exception as e:
    logger.critical(f"❌ فشل في تحميل متغيرات البيئة الأساسية: {e}")
    exit(1)

# ---------------------- إعداد الثوابت ----------------------
BASE_ML_MODEL_NAME = 'LightGBM_Crypto_Predictor_V10_Sequential'
SIGNAL_TIMEFRAME = '15m'
DATA_LOOKBACK_DAYS = 180
BTC_SYMBOL = 'BTCUSDT'

TP_ATR_MULTIPLIER = 1.8
SL_ATR_MULTIPLIER = 1.2
MAX_HOLD_PERIOD = 24

# ---------------------- دوال جلب ومعالجة البيانات ----------------------
def fetch_historical_data(client: Client, symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    try:
        start_str = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines:
            return None
        
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(klines, columns=cols + ['_'] * 6)
        
        for col in cols[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df[cols[1:]].dropna()
    except Exception as e:
        logger.error(f"❌ [Data Fetch] Error fetching data for {symbol}: {e}")
        return None

def engineer_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    df.ta.atr(length=14, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['relative_volume'] = df['volume'] / (df['volume'].rolling(window=30, min_periods=1).mean() + 1e-9)
    
    merged = df.join(btc_df['btc_log_return'], how='left')
    df['btc_correlation'] = merged['log_return'].rolling(window=50).corr(merged['btc_log_return']).fillna(0)

    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    
    df.rename(columns={'ATRr_14': 'atr', 'RSI_14': 'rsi', 'MACDh_12_26_9': 'macd_hist'}, inplace=True)
    
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def get_vectorized_labels(prices: pd.Series, atr: pd.Series) -> pd.Series:
    upper_barrier = prices + (atr * TP_ATR_MULTIPLIER)
    lower_barrier = prices - (atr * SL_ATR_MULTIPLIER)
    
    future_highs = prices.shift(-1).rolling(window=MAX_HOLD_PERIOD, min_periods=1).max()
    future_lows = prices.shift(-1).rolling(window=MAX_HOLD_PERIOD, min_periods=1).min()
    
    profit_hit = future_highs >= upper_barrier
    loss_hit = future_lows <= lower_barrier
    
    labels = pd.Series(0, index=prices.index, dtype=int)
    labels.loc[profit_hit & ~loss_hit] = 1
    labels.loc[~profit_hit & loss_hit] = -1
    labels.loc[profit_hit & loss_hit] = -1
    
    return labels

def train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    lgbm_params = {
        'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
        'boosting_type': 'gbdt', 'n_estimators': 500, 'learning_rate': 0.05,
        'num_leaves': 31, 'seed': 42, 'n_jobs': -1, 'verbose': -1,
    }
    model = lgb.LGBMClassifier(**lgbm_params)
    
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler()
    X.loc[:, numerical_features] = scaler.fit_transform(X[numerical_features])
    
    categorical_features = ['hour', 'day_of_week']
    for col in categorical_features:
        if col in X.columns:
            X.loc[:, col] = X[col].astype('category')
            
    model.fit(X, y, categorical_feature=categorical_features)
    
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision_profit = precision_score(y, y_pred, labels=[2], average='macro', zero_division=0)
    
    metrics = {
        'in_sample_accuracy': accuracy,
        'precision_for_profit_class': precision_profit,
        'num_samples': len(X)
    }
    
    return model, scaler, metrics

# --- دالة معالجة العملة الواحدة ---
def process_symbol(symbol: str, client: Client, conn, btc_df: pd.DataFrame):
    """
    الدالة التي تقوم بكامل عملية التدريب لرمز واحد.
    """
    try:
        logger.info(f"⚙️ [Process] Starting to process {symbol}...")
        
        hist_data = fetch_historical_data(client, symbol, SIGNAL_TIMEFRAME, DATA_LOOKBACK_DAYS)
        if hist_data is None or hist_data.empty:
            logger.warning(f"⚠️ [{symbol}] No historical data found.")
            return (symbol, 'No Data', None)

        df_featured = engineer_features(hist_data, btc_df)
        
        df_featured['target'] = get_vectorized_labels(df_featured['close'], df_featured['atr'])
        df_featured['target_mapped'] = df_featured['target'].map({-1: 0, 0: 1, 1: 2})
        
        feature_columns = ['atr', 'rsi', 'macd_hist', 'log_return', 'relative_volume', 'btc_correlation', 'hour', 'day_of_week']
        
        df_cleaned = df_featured.dropna(subset=feature_columns + ['target_mapped'])
        if df_cleaned.empty or df_cleaned['target_mapped'].nunique() < 3:
            logger.warning(f"⚠️ [{symbol}] Insufficient data after cleaning.")
            return (symbol, 'Insufficient Data', None)

        X = df_cleaned[feature_columns].copy()
        y = df_cleaned['target_mapped']

        model, scaler, metrics = train_model(X, y)
        
        if model and metrics and metrics.get('precision_for_profit_class', 0) > 0.50:
            model_bundle = {'model': model, 'scaler': scaler, 'feature_names': list(X.columns)}
            model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
            
            model_binary = pickle.dumps(model_bundle)
            metrics_json = json.dumps(metrics)
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ml_models (model_name, model_data, metrics) VALUES (%s, %s, %s)
                    ON CONFLICT (model_name) DO UPDATE SET model_data = EXCLUDED.model_data,
                    trained_at = NOW(), metrics = EXCLUDED.metrics;
                """, (model_name, model_binary, metrics_json))
            conn.commit()
            
            logger.info(f"✅ [{symbol}] Model trained and saved successfully.")
            return (symbol, 'Success', metrics)
        else:
            logger.warning(f"⚠️ [{symbol}] Model did not meet performance criteria.")
            return (symbol, 'Low Performance', metrics)
            
    except Exception as e:
        logger.critical(f"❌ [{symbol}] Critical error in process_symbol: {e}", exc_info=True)
        # محاولة إعادة الاتصال بقاعدة البيانات في حالة حدوث خطأ
        if conn and conn.closed:
            logger.info("Reconnecting to the database...")
            conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        return (symbol, 'Error', None)

def send_telegram_notification(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      json={'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e:
        logger.error(f"❌ [Telegram] Failed to send notification: {e}")

def filter_tradable_symbols(client: Client, symbols_to_check: List[str]) -> List[str]:
    logger.info("ℹ️ [Validation] Validating tradable symbols on Binance...")
    try:
        exchange_info = client.get_exchange_info()
        available_symbols = {
            s['symbol'] for s in exchange_info['symbols']
            if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING'
        }
        symbols_set = set(symbols_to_check)
        tradable = list(symbols_set.intersection(available_symbols))
        untradable = list(symbols_set.difference(available_symbols))
        
        if untradable:
            logger.warning(f"⚠️ [Validation] Skipping untradable symbols: {', '.join(untradable)}")
        
        logger.info(f"✅ [Validation] Found {len(tradable)} tradable symbols out of {len(symbols_to_check)}.")
        return tradable
        
    except Exception as e:
        logger.error(f"❌ [Validation] Error during symbol validation: {e}. Skipping validation.")
        return symbols_to_check

# --- ✨ دالة التدريب الرئيسية التسلسلية ✨ ---
def sequential_training_job():
    logger.info(f"🚀 Starting SEQUENTIAL training process ({BASE_ML_MODEL_NAME})...")
    
    client = Client(API_KEY, API_SECRET)
    conn = None
    try:
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        logger.info("✅ Database connection established.")
    except Exception as e:
        logger.critical(f"❌ Could not connect to the database: {e}")
        return

    # جلب بيانات البيتكوين مرة واحدة
    logger.info("ℹ️ [BTC Data] Fetching Bitcoin data...")
    btc_df = fetch_historical_data(client, BTC_SYMBOL, SIGNAL_TIMEFRAME, DATA_LOOKBACK_DAYS)
    if btc_df is None:
        logger.critical("❌ [BTC Data] Failed to fetch Bitcoin data. Exiting.")
        return
    btc_df['btc_log_return'] = np.log(btc_df['close'] / btc_df['close'].shift(1))
    btc_df.dropna(inplace=True)
    logger.info("✅ [BTC Data] Bitcoin data fetched successfully.")

    try:
        with open('crypto_list.txt', 'r', encoding='utf-8') as f:
            symbols_from_file = {s.strip().upper() + 'USDT' for s in f if s.strip()}
    except FileNotFoundError:
        logger.critical("❌ [Main] 'crypto_list.txt' not found. Exiting."); return

    tradable_symbols = filter_tradable_symbols(client, list(symbols_from_file))
    if not tradable_symbols:
        logger.warning("⚠️ [Main] No tradable symbols found from the list. Exiting.")
        return

    send_telegram_notification(f"🚀 *Starting sequential training for {len(tradable_symbols)} symbols*...")
    
    results = []
    # استخدام tqdm لإظهار شريط التقدم
    for symbol in tqdm(tradable_symbols, desc="Training Symbols"):
        result = process_symbol(symbol, client, conn, btc_df)
        results.append(result)

    successful = sum(1 for r in results if r[1] == 'Success')
    failed = len(tradable_symbols) - successful
    
    summary_msg = (f"🏁 *Sequential training process completed*\n"
                   f"- Successful models: {successful}\n"
                   f"- Failed/Skipped models: {failed}")
    send_telegram_notification(summary_msg)
    logger.info(summary_msg)

    # إغلاق الاتصال بقاعدة البيانات في النهاية
    if conn:
        conn.close()
        logger.info("Database connection closed.")


# --- خادم Flask للبقاء نشطًا ---
app = Flask(__name__)
@app.route('/')
def health_check():
    return "Sequential model training service is running.", 200

if __name__ == "__main__":
    train_thread = Thread(target=sequential_training_job)
    train_thread.daemon = True
    train_thread.start()
    
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🌍 Web server running on port {port}...")
    app.run(host='0.0.0.0', port=port)
