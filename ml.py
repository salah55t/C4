import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
import lightgbm as lgb # <-- استيراد LightGBM
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Tuple, Union
from flask import Flask, request, Response
from threading import Thread
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# ---------------------- إعداد التسجيل ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_model_trainer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLTrainer')

# ---------------------- تحميل المتغيرات البيئية ----------------------
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
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 90
# <-- تغيير اسم النموذج ليعكس استخدام LightGBM
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V1' 

# Indicator Parameters
VOLUME_LOOKBACK_CANDLES: int = 1
RSI_PERIOD: int = 9
RSI_MOMENTUM_LOOKBACK_CANDLES: int = 2
ENTRY_ATR_PERIOD: int = 10
SUPERTRAND_PERIOD: int = 10
SUPERTRAND_MULTIPLIER: float = 3.0
TENKAN_PERIOD: int = 9
KIJUN_PERIOD: int = 26
SENKOU_SPAN_B_PERIOD: int = 52
CHIKOU_LAG: int = 26
FIB_SR_LOOKBACK_WINDOW: int = 50

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
training_status: str = "Idle"

# ---------------------- Binance Client & DB Setup ----------------------
try:
    logger.info("ℹ️ [Binance] تهيئة عميل Binance...")
    client = Client(API_KEY, API_SECRET)
    client.ping()
except Exception as e:
    logger.critical(f"❌ [Binance] فشل غير متوقع في تهيئة عميل Binance: {e}")
    exit(1)

def init_db(retries: int = 5, delay: int = 5) -> None:
    global conn
    logger.info("[DB] بدء تهيئة قاعدة البيانات...")
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION NOT NULL,
                    initial_target DOUBLE PRECISION NOT NULL, current_target DOUBLE PRECISION NOT NULL,
                    achieved_target BOOLEAN DEFAULT FALSE, closing_price DOUBLE PRECISION, closed_at TIMESTAMP,
                    profit_percentage DOUBLE PRECISION, strategy_name TEXT, signal_details JSONB, stop_loss DOUBLE PRECISION);
                CREATE TABLE IF NOT EXISTS ml_models (id SERIAL PRIMARY KEY, model_name TEXT NOT NULL UNIQUE,
                    model_data BYTEA NOT NULL, trained_at TIMESTAMP DEFAULT NOW(), metrics JSONB);
            """)
            conn.commit()
            logger.info("✅ [DB] تم تهيئة قاعدة البيانات بنجاح.")
            return
        except Exception as e:
            logger.error(f"❌ [DB] خطأ في الاتصال (المحاولة {attempt + 1}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: exit(1)

def check_db_connection() -> bool:
    global conn
    try:
        if conn is None or conn.closed != 0: init_db()
        else: conn.cursor().execute("SELECT 1;")
        return True
    except (OperationalError, InterfaceError):
        try: init_db()
        except Exception: return False
        return True
    return False

# (Helper functions like get_crypto_symbols, fetch_historical_data, and all indicator calculations remain the same)
# ...
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        start_str = (datetime.utcnow() - timedelta(days=days + 1)).strftime("%Y-%m-%d %H:%M:%S")
        binance_interval = Client.KLINE_INTERVAL_15MINUTE
        klines = client.get_historical_klines(symbol, binance_interval, start_str)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume', 'quote_volume']].dropna()
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        logger.error(f"❌ [Data] خطأ أثناء جلب البيانات التاريخية لـ {symbol}: {e}")
        return None

# ... All indicator functions (calculate_rsi, calculate_atr, etc.) go here ...
def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    if series is None or series.isnull().all() or len(series) < span: return pd.Series(index=series.index if series is not None else None, dtype=float)
    return series.ewm(span=span, adjust=False).mean()
def calculate_rsi_indicator(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    df = df.copy(); delta = df['close'].diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0); avg_gain = gain.ewm(com=period - 1, adjust=False).mean(); avg_loss = loss.ewm(com=period - 1, adjust=False).mean(); rs = avg_gain / avg_loss.replace(0, np.nan); df['rsi'] = (100 - (100 / (1 + rs))).ffill().fillna(50); return df
def calculate_atr_indicator(df: pd.DataFrame, period: int = ENTRY_ATR_PERIOD) -> pd.DataFrame:
    df = df.copy(); high_low = df['high'] - df['low']; high_close_prev = (df['high'] - df['close'].shift(1)).abs(); low_close_prev = (df['low'] - df['close'].shift(1)).abs(); tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False); df['atr'] = tr.ewm(span=period, adjust=False).mean(); return df
def calculate_supertrend(df: pd.DataFrame, period: int = SUPERTRAND_PERIOD, multiplier: float = SUPERTRAND_MULTIPLIER) -> pd.DataFrame:
    df = df.copy();
    if 'atr' not in df.columns: df = calculate_atr_indicator(df, period=period)
    df['basic_upper_band'] = ((df['high'] + df['low']) / 2) + (multiplier * df['atr']); df['basic_lower_band'] = ((df['high'] + df['low']) / 2) - (multiplier * df['atr']); df['final_upper_band'], df['final_lower_band'] = 0.0, 0.0; df['supertrend'], df['supertrend_direction'] = 0.0, 0
    for i in range(1, len(df)):
        if df['basic_upper_band'].iloc[i] < df['final_upper_band'].iloc[i-1] or df['close'].iloc[i-1] > df['final_upper_band'].iloc[i-1]: df.loc[df.index[i], 'final_upper_band'] = df['basic_upper_band'].iloc[i]
        else: df.loc[df.index[i], 'final_upper_band'] = df['final_upper_band'].iloc[i-1]
        if df['basic_lower_band'].iloc[i] > df['final_lower_band'].iloc[i-1] or df['close'].iloc[i-1] < df['final_lower_band'].iloc[i-1]: df.loc[df.index[i], 'final_lower_band'] = df['basic_lower_band'].iloc[i]
        else: df.loc[df.index[i], 'final_lower_band'] = df['final_lower_band'].iloc[i-1]
        if df['supertrend_direction'].iloc[i-1] == 1:
            if df['close'].iloc[i] < df['final_upper_band'].iloc[i]: df.loc[df.index[i], 'supertrend'], df.loc[df.index[i], 'supertrend_direction'] = df['final_upper_band'].iloc[i], -1
            else: df.loc[df.index[i], 'supertrend'], df.loc[df.index[i], 'supertrend_direction'] = df['final_lower_band'].iloc[i], 1
        elif df['supertrend_direction'].iloc[i-1] == -1:
            if df['close'].iloc[i] > df['final_lower_band'].iloc[i]: df.loc[df.index[i], 'supertrend'], df.loc[df.index[i], 'supertrend_direction'] = df['final_lower_band'].iloc[i], 1
            else: df.loc[df.index[i], 'supertrend'], df.loc[df.index[i], 'supertrend_direction'] = df['final_upper_band'].iloc[i], -1
        else:
            if df['close'].iloc[i] > df['final_lower_band'].iloc[i]: df.loc[df.index[i], 'supertrend_direction'] = 1
            elif df['close'].iloc[i] < df['final_upper_band'].iloc[i]: df.loc[df.index[i], 'supertrend_direction'] = -1
            df.loc[df.index[i], 'supertrend'] = df.loc[df.index[i], 'final_lower_band'] if df.loc[df.index[i], 'supertrend_direction'] == 1 else df.loc[df.index[i], 'final_upper_band']
    df.drop(columns=['basic_upper_band', 'basic_lower_band', 'final_upper_band', 'final_lower_band'], inplace=True, errors='ignore'); return df
def _calculate_btc_trend_feature(df_btc: pd.DataFrame) -> Optional[pd.Series]:
    if df_btc is None or df_btc.empty or len(df_btc) < 55: return None
    ema20, ema50 = calculate_ema(df_btc['close'], 20), calculate_ema(df_btc['close'], 50); ema_df = pd.DataFrame({'ema20': ema20, 'ema50': ema50, 'close': df_btc['close']}).dropna(); trend_series = pd.Series(index=ema_df.index, data=0.0)
    trend_series[(ema_df['close'] > ema_df['ema20']) & (ema_df['ema20'] > ema_df['ema50'])] = 1.0; trend_series[(ema_df['close'] < ema_df['ema20']) & (ema_df['ema20'] < ema_df['ema50'])] = -1.0; return trend_series.reindex(df_btc.index).fillna(0.0)
def calculate_ichimoku_cloud(df: pd.DataFrame, tenkan_period: int = TENKAN_PERIOD, kijun_period: int = KIJUN_PERIOD, senkou_span_b_period: int = SENKOU_SPAN_B_PERIOD, chikou_lag: int = CHIKOU_LAG) -> pd.DataFrame:
    df_ichimoku = df.copy(); df_ichimoku['tenkan_sen'] = (df_ichimoku['high'].rolling(window=tenkan_period).max() + df_ichimoku['low'].rolling(window=tenkan_period).min()) / 2; df_ichimoku['kijun_sen'] = (df_ichimoku['high'].rolling(window=kijun_period).max() + df_ichimoku['low'].rolling(window=kijun_period).min()) / 2; df_ichimoku['senkou_span_a'] = ((df_ichimoku['tenkan_sen'] + df_ichimoku['kijun_sen']) / 2).shift(kijun_period); df_ichimoku['senkou_span_b'] = ((df_ichimoku['high'].rolling(window=senkou_span_b_period).max() + df_ichimoku['low'].rolling(window=senkou_span_b_period).min()) / 2).shift(kijun_period); df_ichimoku['chikou_span'] = df_ichimoku['close'].shift(-chikou_lag); df_ichimoku['ichimoku_tenkan_kijun_cross_signal'] = np.where((df_ichimoku['tenkan_sen'] > df_ichimoku['kijun_sen']) & (df_ichimoku['tenkan_sen'].shift(1) < df_ichimoku['kijun_sen'].shift(1)), 1, np.where((df_ichimoku['tenkan_sen'] < df_ichimoku['kijun_sen']) & (df_ichimoku['tenkan_sen'].shift(1) > df_ichimoku['kijun_sen'].shift(1)), -1, 0)); df_ichimoku['ichimoku_price_cloud_position'] = np.where(df_ichimoku['close'] > df_ichimoku[['senkou_span_a', 'senkou_span_b']].max(axis=1), 1, np.where(df_ichimoku['close'] < df_ichimoku[['senkou_span_a', 'senkou_span_b']].min(axis=1), -1, 0)); df_ichimoku['ichimoku_cloud_outlook'] = np.where(df_ichimoku['senkou_span_a'] > df_ichimoku['senkou_span_b'], 1, np.where(df_ichimoku['senkou_span_a'] < df_ichimoku['senkou_span_b'], -1, 0)); return df_ichimoku
def calculate_fibonacci_features(df: pd.DataFrame, lookback_window: int = FIB_SR_LOOKBACK_WINDOW) -> pd.DataFrame:
    df_fib = df.copy()
    for i in range(lookback_window - 1, len(df_fib)):
        window_df = df_fib.iloc[i - lookback_window + 1 : i + 1]; swing_high, swing_low = window_df['high'].max(), window_df['low'].min(); price_range = swing_high - swing_low
        if price_range > 0:
            current_close = df_fib['close'].iloc[i]; fib_0_500 = swing_high - (price_range * 0.500)
            df_fib.loc[df_fib.index[i], 'fib_236_retrace_dist_norm'] = (current_close - (swing_high - price_range * 0.236)) / price_range
            df_fib.loc[df_fib.index[i], 'fib_382_retrace_dist_norm'] = (current_close - (swing_high - price_range * 0.382)) / price_range
            df_fib.loc[df_fib.index[i], 'fib_618_retrace_dist_norm'] = (current_close - (swing_high - price_range * 0.618)) / price_range
            df_fib.loc[df_fib.index[i], 'is_price_above_fib_50'] = 1 if current_close > fib_0_500 else 0
    return df_fib
def calculate_support_resistance_features(df: pd.DataFrame, lookback_window: int = FIB_SR_LOOKBACK_WINDOW) -> pd.DataFrame:
    df_sr = df.copy()
    for i in range(lookback_window - 1, len(df_sr)):
        window_df = df_sr.iloc[i - lookback_window + 1 : i + 1]; recent_high, recent_low = window_df['high'].max(), window_df['low'].min(); price_range = recent_high - recent_low
        if price_range > 0:
            current_close = df_sr['close'].iloc[i]
            df_sr.loc[df_sr.index[i], 'price_distance_to_recent_low_norm'] = (current_close - recent_low) / price_range
            df_sr.loc[df_sr.index[i], 'price_distance_to_recent_high_norm'] = (recent_high - current_close) / price_range
    return df_sr
def get_crypto_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    try:
        with open(os.path.join(os.path.dirname(__file__), filename), 'r', encoding='utf-8') as f: raw_symbols = [line.strip().upper() for line in f if line.strip() and not line.startswith('#')]
        raw_symbols = sorted([f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols]); exchange_info = client.get_exchange_info()
        valid_symbols = {s['symbol'] for s in exchange_info['symbols'] if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'}; return [s for s in raw_symbols if s in valid_symbols]
    except Exception as e: logger.error(f"❌ [Data Validation] خطأ أثناء التحقق من الرموز: {e}"); return []
def convert_np_values(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_np_values(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_np_values(i) for i in obj]
    if pd.isna(obj): return None
    return obj

# ---------------------- Model Training and Saving Functions ----------------------
def prepare_data_for_ml(df: pd.DataFrame, symbol: str, target_period: int = 5) -> Optional[pd.DataFrame]:
    logger.info(f"ℹ️ [ML Prep] Preparing data for ML model for {symbol}...")
    min_len_required = max(FIB_SR_LOOKBACK_WINDOW, SENKOU_SPAN_B_PERIOD, 55) + target_period + 5
    if len(df) < min_len_required:
        logger.warning(f"⚠️ [ML Prep] DataFrame for {symbol} is too short ({len(df)} < {min_len_required}).")
        return None

    df_calc = df.copy()
    df_calc['volume_15m_avg'] = df_calc['quote_volume'].rolling(window=VOLUME_LOOKBACK_CANDLES, min_periods=1).mean()
    df_calc = calculate_rsi_indicator(df_calc, RSI_PERIOD)
    df_calc['rsi_momentum_bullish'] = 0
    for i in range(RSI_MOMENTUM_LOOKBACK_CANDLES, len(df_calc)):
        rsi_slice = df_calc['rsi'].iloc[i - RSI_MOMENTUM_LOOKBACK_CANDLES : i + 1]
        if not rsi_slice.isnull().any() and np.all(np.diff(rsi_slice) > 0) and rsi_slice.iloc[-1] > 50:
            df_calc.loc[df_calc.index[i], 'rsi_momentum_bullish'] = 1
    df_calc = calculate_atr_indicator(df_calc, ENTRY_ATR_PERIOD)
    df_calc = calculate_supertrend(df_calc, SUPERTRAND_PERIOD, SUPERTRAND_MULTIPLIER)
    btc_df = fetch_historical_data("BTCUSDT", interval=SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING)
    if btc_df is not None:
        btc_trend = _calculate_btc_trend_feature(btc_df)
        df_calc = df_calc.merge(btc_trend.rename('btc_trend_feature'), left_index=True, right_index=True, how='left').fillna(0.0)
    else: df_calc['btc_trend_feature'] = 0.0
    
    df_calc = calculate_ichimoku_cloud(df_calc)
    df_calc = calculate_fibonacci_features(df_calc)
    df_calc = calculate_support_resistance_features(df_calc)

    feature_columns = [
        'volume_15m_avg', 'rsi_momentum_bullish', 'btc_trend_feature', 'supertrend_direction',
        'ichimoku_tenkan_kijun_cross_signal', 'ichimoku_price_cloud_position', 'ichimoku_cloud_outlook',
        'fib_236_retrace_dist_norm', 'fib_382_retrace_dist_norm', 'fib_618_retrace_dist_norm',
        'is_price_above_fib_50', 'price_distance_to_recent_low_norm', 'price_distance_to_recent_high_norm'
    ]
    for col in feature_columns:
        if col not in df_calc.columns: df_calc[col] = np.nan
    
    price_change_threshold = 0.005
    df_calc['future_max_close'] = df_calc['close'].shift(-target_period).rolling(window=target_period).max()
    df_calc['target'] = ((df_calc['future_max_close'] / df_calc['close']) - 1 > price_change_threshold).astype(int)
    
    df_cleaned = df_calc.dropna(subset=feature_columns + ['target']).copy()
    if df_cleaned.empty:
        logger.warning(f"⚠️ [ML Prep] DataFrame for {symbol} is empty after NaN removal.")
        return None
    
    return df_cleaned[feature_columns + ['target']]

def train_and_evaluate_model(data: pd.DataFrame) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    logger.info("ℹ️ [ML Train] Starting model training and evaluation with LightGBM...")
    if data.empty:
        logger.error("❌ [ML Train] Empty DataFrame for training.")
        return None, None, {}

    X = data.drop('target', axis=1)
    y = data['target']
    if y.nunique() < 2:
        logger.warning(f"⚠️ [ML Train] Target column has only one class. Cannot train model.")
        return None, None, {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # <-- استبدال النموذج بـ LightGBM
    model = lgb.LGBMClassifier(
        objective='binary',
        random_state=42,
        n_estimators=250,      # زيادة عدد الأشجار
        learning_rate=0.05,    # معدل تعلم أصغر
        num_leaves=31,         # عدد الأوراق في الشجرة
        max_depth=-1,          # لا يوجد حد للعمق
        n_jobs=-1,             # استخدام كل أنوية المعالج
        colsample_bytree=0.8,  # استخدام 80% من الميزات لكل شجرة
        subsample=0.8          # استخدام 80% من البيانات لكل شجرة
    )
    
    model.fit(X_train_scaled, y_train,
              eval_set=[(X_test_scaled, y_test)],
              eval_metric='logloss',
              callbacks=[lgb.early_stopping(10, verbose=False)]) # التوقف المبكر لتحسين الأداء
    
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'num_samples_trained': len(X_train),
        'feature_names': X.columns.tolist()
    }
    logger.info(f"📊 [ML Train] LightGBM performance: { {k: f'{v:.4f}' for k, v in metrics.items() if isinstance(v, float)} }")
    
    return model, scaler, metrics

def save_ml_model_to_db(model_bundle: Dict[str, Any], model_name: str, metrics: Dict[str, Any]) -> bool:
    logger.info(f"ℹ️ [DB Save] Attempting to save ML model bundle '{model_name}'...")
    if not check_db_connection() or not conn:
        logger.error("❌ [DB Save] Cannot save model due to database connection issue.")
        return False
    try:
        model_binary = pickle.dumps(model_bundle)
        metrics_json = json.dumps(convert_np_values(metrics))
        
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT id FROM ml_models WHERE model_name = %s;", (model_name,))
            existing_model = db_cur.fetchone()
            if existing_model:
                db_cur.execute("UPDATE ml_models SET model_data = %s, trained_at = NOW(), metrics = %s WHERE id = %s;",
                               (model_binary, metrics_json, existing_model['id']))
            else:
                db_cur.execute("INSERT INTO ml_models (model_name, model_data, trained_at, metrics) VALUES (%s, %s, NOW(), %s);",
                               (model_name, model_binary, metrics_json))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"❌ [DB Save] Unexpected error while saving ML model bundle: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

# ---------------------- Main Entry Point ----------------------
def send_telegram_message(target_chat_id: str, text: str):
    if not TELEGRAM_TOKEN or not target_chat_id: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try: requests.post(url, json={'chat_id': str(target_chat_id), 'text': text, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

if __name__ == "__main__":
    logger.info("🚀 Starting ML model training script with LightGBM...")
    # Start Flask in a daemon thread
    app = Flask(__name__)
    @app.route('/')
    def home(): return Response(f"ML Trainer Service: {training_status}", status=200)
    flask_thread = Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': int(os.environ.get('PORT', 10000))}, daemon=True)
    flask_thread.start()

    try:
        init_db()
        symbols = get_crypto_symbols()
        if not symbols:
            logger.critical("❌ [Main] No valid symbols to train. Exiting.")
            exit(1)
        
        training_status = "In Progress"
        send_telegram_message(CHAT_ID, f"🚀 *LightGBM Model Training Started*\nTraining models for {len(symbols)} symbols.")
        
        successful_models = 0
        for symbol in symbols:
            logger.info(f"\n--- ⏳ [Main] Starting LightGBM model training for {symbol} ---")
            try:
                df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING)
                if df_hist is None or df_hist.empty: continue
                
                df_processed = prepare_data_for_ml(df_hist, symbol)
                if df_processed is None or df_processed.empty: continue
                
                trained_model, trained_scaler, model_metrics = train_and_evaluate_model(df_processed)
                
                if trained_model and trained_scaler:
                    model_bundle = {'model': trained_model, 'scaler': trained_scaler}
                    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
                    if save_ml_model_to_db(model_bundle, model_name, model_metrics):
                        successful_models += 1
                else:
                    logger.warning(f"⚠️ [Main] Training did not produce a valid model for {symbol}.")
            except Exception as e:
                logger.critical(f"❌ [Main] A fatal error occurred for {symbol}: {e}", exc_info=True)
            time.sleep(1)

        training_status = f"Completed. {successful_models}/{len(symbols)} models trained."
        send_telegram_message(CHAT_ID, f"✅ *LightGBM Training Finished*\nSuccessfully trained {successful_models}/{len(symbols)} models.")
        
    except Exception as e:
        logger.critical(f"❌ [Main] A fatal error occurred in main script: {e}", exc_info=True)
        training_status = f"Failed: {e}"
        send_telegram_message(CHAT_ID, f"🚨 *Fatal Error in ML Training Script*\n`{e}`")
    finally:
        if conn: conn.close()
        logger.info("👋 [Main] ML training script finished. Flask server continues to run.")
        flask_thread.join() # Keep the main thread alive for Flask
