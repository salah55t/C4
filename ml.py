# ml_corrected.py
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
import pandas_ta as ta
import sys # <--- التعديل: استيراد sys للخروج الآمن
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

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_model_trainer_v7.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLTrainer_V7')

# ---------------------- تحميل متغيرات البيئة ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
    TELEGRAM_TOKEN: Optional[str] = config('TELEGRAM_BOT_TOKEN', default=None)
    CHAT_ID: Optional[str] = config('TELEGRAM_CHAT_ID', default=None)
except Exception as e:
     logger.critical(f"❌ فشل في تحميل المتغيرات البيئية الأساسية: {e}")
     sys.exit(1) # <--- التعديل: استخدام sys.exit

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V7_EnhancedFeatures'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 200

RSI_PERIODS: List[int] = [14, 21]
MACD_PARAMS: Dict[str, int] = {"fast": 12, "slow": 26, "signal": 9}
ATR_PERIOD: int = 14
BOLLINGER_PERIOD: int = 20
ADX_PERIOD: int = 14
MOM_PERIOD: int = 10
EMA_FAST_PERIODS: List[int] = [12, 50]
EMA_SLOW_PERIODS: List[int] = [26, 200]
BTC_CORR_PERIOD: int = 30
BTC_SYMBOL = 'BTCUSDT'

TP_ATR_MULTIPLIER: float = 1.8
SL_ATR_MULTIPLIER: float = 1.2
MAX_HOLD_PERIOD: int = 24

conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
btc_data_cache: Optional[pd.DataFrame] = None

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
        logger.critical(f"❌ [DB] فشل الاتصال بقاعدة البيانات: {e}"); sys.exit(1)

def get_binance_client():
    global client
    try:
        client = Client(API_KEY, API_SECRET)
        client.ping()
        logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
    except Exception as e:
        logger.critical(f"❌ [Binance] فشل تهيئة عميل Binance: {e}"); sys.exit(1)

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    if not client:
        logger.error("❌ [Validation] عميل Binance لم يتم تهيئته.")
        return []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__)) # <--- التعديل: استخدام abspath لمزيد من الموثوقية
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
        logger.error(f"❌ [Data] خطأ أثناء جلب البيانات لـ {symbol}: {e}"); return None

def fetch_and_cache_btc_data():
    global btc_data_cache
    logger.info("ℹ️ [BTC Data] جاري جلب بيانات البيتكوين وتخزينها...")
    btc_data_cache = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING + 1) # Add buffer
    if btc_data_cache is None:
        logger.critical("❌ [BTC Data] فشل جلب بيانات البيتكوين."); sys.exit(1)
    btc_data_cache['btc_log_returns'] = np.log(btc_data_cache['close'] / btc_data_cache['close'].shift(1))

# ... (دالة calculate_features_v7 تبقى كما هي) ...
# دالة حساب الميزات لم تتغير بشكل كبير، ولكن التحسينات في الدوال الأخرى تجعلها أكثر قوة
def calculate_features_v7(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy().astype('float64')

    df_calc['log_returns'] = np.log(df_calc['close'] / df_calc['close'].shift(1))
    df_calc['candle_range'] = (df_calc['high'] - df_calc['low']) / df_calc['close']
    # معالجة القسمة على صفر في النسب
    candle_range_nonzero = df_calc['high'] - df_calc['low']
    candle_range_nonzero[candle_range_nonzero == 0] = 1e-12 # قيمة صغيرة جدًا لتجنب القسمة على صفر
    df_calc['upper_shadow_ratio'] = (df_calc['high'] - np.maximum(df_calc['open'], df_calc['close'])) / candle_range_nonzero
    df_calc['lower_shadow_ratio'] = (np.minimum(df_calc['open'], df_calc['close']) - df_calc['low']) / candle_range_nonzero
    df_calc[['upper_shadow_ratio', 'lower_shadow_ratio']] = df_calc[['upper_shadow_ratio', 'lower_shadow_ratio']].fillna(0)
    df_calc['volume_change'] = df_calc['volume'].pct_change()

    df_calc.ta.ema(close=df_calc['close'], length=EMA_FAST_PERIODS[0], append=True, col_names=(f'EMA_{EMA_FAST_PERIODS[0]}',))
    df_calc.ta.ema(close=df_calc['close'], length=EMA_FAST_PERIODS[1], append=True, col_names=(f'EMA_{EMA_FAST_PERIODS[1]}',))
    df_calc.ta.ema(close=df_calc['close'], length=EMA_SLOW_PERIODS[0], append=True, col_names=(f'EMA_{EMA_SLOW_PERIODS[0]}',))
    df_calc.ta.ema(close=df_calc['close'], length=EMA_SLOW_PERIODS[1], append=True, col_names=(f'EMA_{EMA_SLOW_PERIODS[1]}',))
    df_calc.ta.macd(close=df_calc['close'], fast=MACD_PARAMS["fast"], slow=MACD_PARAMS["slow"], signal=MACD_PARAMS["signal"], append=True)
    df_calc.ta.adx(high=df_calc['high'], low=df_calc['low'], close=df_calc['close'], length=ADX_PERIOD, append=True)
    df_calc.ta.rsi(close=df_calc['close'], length=RSI_PERIODS[0], append=True, col_names=(f'RSI_{RSI_PERIODS[0]}',))
    df_calc.ta.rsi(close=df_calc['close'], length=RSI_PERIODS[1], append=True, col_names=(f'RSI_{RSI_PERIODS[1]}',))
    df_calc.ta.stoch(high=df_calc['high'], low=df_calc['low'], close=df_calc['close'], k=14, d=3, append=True, col_names=('STOCH_K_14_3','STOCH_D_14_3'))
    df_calc.ta.mom(close=df_calc['close'], length=MOM_PERIOD, append=True, col_names=('MOMENTUM',))
    df_calc.ta.obv(close=df_calc['close'], volume=df_calc['volume'], append=True)
    df_calc.ta.atr(high=df_calc['high'], low=df_calc['low'], close=df_calc['close'], length=ATR_PERIOD, append=True)
    df_calc.ta.bbands(close=df_calc['close'], length=BOLLINGER_PERIOD, append=True)

    lag_periods = [1, 2, 3, 5, 10]
    for lag in lag_periods:
        df_calc[f'CLOSE_LAG_{lag}'] = df_calc['close'].shift(lag)
        df_calc[f'VOLUME_LAG_{lag}'] = df_calc['volume'].shift(lag)
        df_calc[f'LOG_RETURNS_LAG_{lag}'] = df_calc['log_returns'].shift(lag)
        if f'RSI_{RSI_PERIODS[0]}' in df_calc.columns:
            df_calc[f'RSI_{RSI_PERIODS[0]}_LAG_{lag}'] = df_calc[f'RSI_{RSI_PERIODS[0]}'].shift(lag)
        if f'MACDH_{MACD_PARAMS["fast"]}_{MACD_PARAMS["slow"]}_{MACD_PARAMS["signal"]}' in df_calc.columns:
            df_calc[f'MACDH_{MACD_PARAMS["fast"]}_{MACD_PARAMS["slow"]}_{MACD_PARAMS["signal"]}_LAG_{lag}'] = df_calc[f'MACDH_{MACD_PARAMS["fast"]}_{MACD_PARAMS["slow"]}_{MACD_PARAMS["signal"]}'].shift(lag)

    df_calc['day_of_week'] = df_calc.index.dayofweek
    df_calc['hour_of_day'] = df_calc.index.hour
    df_calc['is_weekend'] = (df_calc.index.dayofweek >= 5).astype(int)

    if btc_df is not None and not btc_df.empty:
        df_calc['BTC_LOG_RETURNS'] = btc_df['btc_log_returns'].reindex(df_calc.index, method='nearest')
        if df_calc['log_returns'].std() > 0 and df_calc['BTC_LOG_RETURNS'].std() > 0:
            df_calc[f'BTC_CORRELATION_{BTC_CORR_PERIOD}'] = df_calc['log_returns'].rolling(window=BTC_CORR_PERIOD, min_periods=BTC_CORR_PERIOD//2).corr(df_calc['BTC_LOG_RETURNS'])
            df_calc[f'BTC_CORRELATION_{BTC_CORR_PERIOD}'].fillna(0, inplace=True)
            df_calc['BTC_CORRELATION_SQUARED'] = df_calc[f'BTC_CORRELATION_{BTC_CORR_PERIOD}'] ** 2 

    if f'EMA_{EMA_FAST_PERIODS[0]}' in df_calc.columns:
        df_calc['PRICE_VS_EMA_FAST_RATIO'] = (df_calc['close'] / df_calc[f'EMA_{EMA_FAST_PERIODS[0]}']) - 1
    if f'EMA_{EMA_SLOW_PERIODS[0]}' in df_calc.columns:
        df_calc['PRICE_VS_EMA_SLOW_RATIO'] = (df_calc['close'] / df_calc[f'EMA_{EMA_SLOW_PERIODS[0]}']) - 1
    if f'ATRr_{ATR_PERIOD}' in df_calc.columns:
         df_calc['VOLUME_X_VOLATILITY'] = df_calc['volume'] * df_calc[f'ATRr_{ATR_PERIOD}']

    df_calc.columns = [col.upper().replace(f'_{BOLLINGER_PERIOD}_2.0', '') for col in df_calc.columns] # <--- التعديل: تنظيف الأسماء بشكل أفضل
    return df_calc

def get_triple_barrier_labels(prices: pd.Series, atr: pd.Series) -> pd.Series:
    # <--- التعديل: تحسين منطق توليد التسميات بالكامل
    labels = pd.Series(np.nan, index=prices.index) # تهيئة بـ NaN بدلاً من 0

    for i in tqdm(range(len(prices) - MAX_HOLD_PERIOD), desc="Labeling", leave=False, ncols=100):
        entry_price = prices.iloc[i]
        current_atr = atr.iloc[i]
        
        if pd.isna(current_atr) or current_atr <= 0:
            continue
        
        upper_barrier = entry_price + (current_atr * TP_ATR_MULTIPLIER)
        lower_barrier = entry_price - (current_atr * SL_ATR_MULTIPLIER)
        
        path = prices.iloc[i+1 : i+1+MAX_HOLD_PERIOD]
        
        # التحقق من ملامسة الحواجز
        hit_upper = path[path >= upper_barrier].first_valid_index()
        hit_lower = path[path <= lower_barrier].first_valid_index()
        
        if hit_upper and (not hit_lower or hit_upper < hit_lower):
            labels.iloc[i] = 1  # ربح
        elif hit_lower and (not hit_upper or hit_lower < hit_upper):
            labels.iloc[i] = -1 # خسارة
        else:
            labels.iloc[i] = 0 # انتهاء الوقت
            
    return labels

def prepare_data_for_ml(df: pd.DataFrame, btc_df: pd.DataFrame, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    logger.info(f"ℹ️ [ML Prep V7] Preparing data for {symbol}...")
    df_featured = calculate_features_v7(df, btc_df)
    
    atr_series_name = f'ATRR_{ATR_PERIOD}'.upper()
    if atr_series_name not in df_featured.columns or df_featured[atr_series_name].isnull().all():
        logger.error(f"FATAL: ATR column '{atr_series_name}' not found or is all NaN for {symbol}.")
        return None
        
    df_featured['TARGET'] = get_triple_barrier_labels(df_featured['CLOSE'], df_featured[atr_series_name])
    
    # <--- التعديل: إنشاء قائمة الميزات ديناميكيًا
    base_features = list(df_featured.columns)
    # استثناء الأعمدة غير المرغوب فيها كميزات
    features_to_exclude = ['TARGET', 'OPEN', 'HIGH', 'LOW'] # استبعاد أسعار الشمعة الحالية لمنع تسرب البيانات
    feature_columns = [col for col in base_features if col not in features_to_exclude]

    # إزالة كل الصفوف التي تحتوي على أي قيمة فارغة في الميزات أو الهدف
    df_cleaned = df_featured.dropna(subset=feature_columns + ['TARGET']).copy()
    
    # تصفية الصفقات التي انتهى وقتها (TARGET=0)
    df_cleaned = df_cleaned[df_cleaned['TARGET'] != 0].copy()

    if df_cleaned.empty or df_cleaned['TARGET'].nunique() < 2:
        logger.warning(f"⚠️ [ML Prep] Data for {symbol} has less than 2 classes after filtering or is empty. Skipping.")
        return None
    
    # تحويل -1 (خسارة) إلى 0 ليتناسب مع التصنيف الثنائي
    df_cleaned['TARGET'] = df_cleaned['TARGET'].replace(-1, 0)
    
    target_counts = df_cleaned['TARGET'].value_counts(normalize=True)
    logger.info(f"📊 [ML Prep] Target distribution for {symbol} (after filtering):\n{target_counts}")

    if target_counts.min() < 0.1:
        logger.warning(f"⚠️ [ML Prep] Severe class imbalance for {symbol}. Min class is {target_counts.min():.2%}. Skipping training.")
        return None

    final_feature_columns = [col for col in feature_columns if col in df_cleaned.columns]

    X = df_cleaned[final_feature_columns]
    y = df_cleaned['TARGET']
    return X, y, final_feature_columns


def train_with_walk_forward_validation(X: pd.DataFrame, y: pd.Series, feature_names: List[str]) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    # <--- التعديل: إعادة هيكلة الدالة لتدريب النموذج النهائي على كامل البيانات
    logger.info("ℹ️ [ML Train V7] Starting training with Walk-Forward Validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    lgb_params = {
        'objective': 'binary', 'metric': 'logloss', 'random_state': 42,
        'verbosity': -1, 'n_estimators': 1500, 'learning_rate': 0.01,
        'num_leaves': 31, 'max_depth': -1, 'class_weight': 'balanced',
        'reg_alpha': 0.0, 'reg_lambda': 0.0, 'n_jobs': -1,
        'colsample_bytree': 0.8, 'min_child_samples': 10, 'boosting_type': 'gbdt',
    }
    
    all_y_true, all_y_pred = [], []
    
    # الخطوة 1: التحقق من الأداء باستخدام Walk-Forward Validation
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        if len(y_train) == 0 or len(y_test) == 0:
            logger.warning(f"--- Fold {i+1}: Skipping due to empty train/test set.")
            continue
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(X_train_scaled, y_train, 
                  eval_set=[(X_test_scaled, y_test)],
                  eval_metric='logloss',
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        
        y_pred = model.predict(X_test_scaled)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    if not all_y_true:
        logger.error("❌ [ML Train] Validation failed, no data to generate report.")
        return None, None, None

    # تقرير الأداء المجمع من كل مراحل التحقق
    final_report = classification_report(all_y_true, all_y_pred, output_dict=True, zero_division=0)
    avg_metrics = {
        'accuracy': accuracy_score(all_y_true, all_y_pred),
        'precision_win': final_report.get('1', {}).get('precision', 0),
        'recall_win': final_report.get('1', {}).get('recall', 0),
        'f1_score_win': final_report.get('1', {}).get('f1-score', 0),
        'num_samples_trained': len(X),
    }
    logger.info(f"📊 [ML Validate] Aggregated performance: {', '.join([f'{k}: {v:.4f}' for k, v in avg_metrics.items()])}")

    # الخطوة 2: تدريب النموذج النهائي على كل البيانات
    logger.info("ℹ️ [ML Train] Retraining final model on the entire dataset...")
    final_scaler = StandardScaler()
    X_scaled_full = final_scaler.fit_transform(X) # تدريب المحول على كل البيانات
    
    final_model = lgb.LGBMClassifier(**lgb_params)
    final_model.fit(X_scaled_full, y) # تدريب النموذج على كل البيانات

    logger.info("✅ [ML Train] Final model training complete.")
    return final_model, final_scaler, avg_metrics


# ... (دوال الحفظ والإرسال تبقى كما هي) ...

def run_training_job():
    logger.info(f"🚀 Starting ML model training job ({BASE_ML_MODEL_NAME})...")
    init_db()
    get_binance_client()
    fetch_and_cache_btc_data()
    symbols_to_train = get_validated_symbols(filename='crypto_list.txt')
    if not symbols_to_train:
        logger.critical("❌ [Main] No valid symbols found. Exiting.")
        return
        
    send_telegram_message(f"🚀 *{BASE_ML_MODEL_NAME} Training Started*\nWill train models for {len(symbols_to_train)} symbols.")
    
    successful_models, failed_models = 0, 0
    for symbol in symbols_to_train:
        logger.info(f"\n--- ⏳ [Main] Starting model training for {symbol} ---")
        try:
            df_hist = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
            if df_hist is None or df_hist.empty:
                logger.warning(f"⚠️ [Main] No data for {symbol}, skipping."); failed_models += 1; continue
            
            prepared_data = prepare_data_for_ml(df_hist, btc_data_cache, symbol)
            if prepared_data is None:
                failed_models += 1; continue
            X, y, feature_names = prepared_data
            
            # <--- التعديل: تمرير قائمة الميزات إلى دالة التدريب
            training_result = train_with_walk_forward_validation(X, y, feature_names)
            if not all(training_result):
                 logger.warning(f"⚠️ [Main] Training did not produce a valid model for {symbol}. Skipping."); failed_models += 1; continue
            final_model, final_scaler, model_metrics = training_result
            
            if final_model and final_scaler and model_metrics.get('precision_win', 0) > 0.52 and model_metrics.get('f1_score_win', 0) > 0.5:
                model_bundle = {'model': final_model, 'scaler': final_scaler, 'feature_names': feature_names}
                model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
                save_ml_model_to_db(model_bundle, model_name, model_metrics)
                successful_models += 1
            else:
                precision = model_metrics.get('precision_win', 0)
                f1_score = model_metrics.get('f1_score_win', 0)
                logger.warning(f"⚠️ [Main] Model for {symbol} is not useful (Precision {precision:.2f}, F1-Score: {f1_score:.2f}). Discarding."); failed_models += 1
        except Exception as e:
            logger.critical(f"❌ [Main] A fatal error occurred for {symbol}: {e}", exc_info=True); failed_models += 1
        time.sleep(1)

    completion_message = (f"✅ *{BASE_ML_MODEL_NAME} Training Finished*\n"
                        f"- تم التدريب بنجاح: {successful_models} نموذج\n"
                        f"- فشل/تم التجاهل: {failed_models} نموذج\n"
                        f"- إجمالي العملات: {len(symbols_to_train)}")
    send_telegram_message(completion_message)
    logger.info(completion_message)

    if conn: conn.close()
    logger.info("👋 [Main] ML training job finished.")

app = Flask(__name__)
@app.route('/')
def health_check():
    return "ML Trainer service (V7) is running and healthy.", 200

if __name__ == "__main__":
    # <--- التعديل: تشغيل التدريب أولاً ثم الخادم لمزيد من الأمان
    logger.info("Starting training job before launching web server.")
    run_training_job()
    
    port = int(os.environ.get("PORT", 10001))
    logger.info(f"🌍 Training complete. Starting web server on port {port} to keep the service alive...")
    app.run(host='0.0.0.0', port=port)