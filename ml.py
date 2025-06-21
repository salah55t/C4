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
from multiprocessing import Pool, cpu_count, Manager
from sklearn.metrics import accuracy_score, precision_score


# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_trainer_optimized.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('OptimizedCryptoMLTrainer')

# ---------------------- تحميل متغيرات البيئة ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
    TELEGRAM_TOKEN: Optional[str] = config('TELEGRAM_BOT_TOKEN', default=None)
    CHAT_ID: Optional[str] = config('TELEGRAM_CHAT_ID', default=None)
except Exception as e:
     logger.critical(f"❌ فشل في تحميل متغيرات البيئة الأساسية: {e}")
     exit(1)

# ---------------------- إعداد الثوابت (معدلة للسرعة) ----------------------
BASE_ML_MODEL_NAME: str = 'LightGBM_Crypto_Predictor_V9_Optimized'
SIGNAL_TIMEFRAME: str = '15m'
DATA_LOOKBACK_DAYS: int = 180 
BTC_SYMBOL = 'BTCUSDT'

# --- معلمات طريقة الحاجز الثلاثي ---
TP_ATR_MULTIPLIER: float = 1.8
SL_ATR_MULTIPLIER: float = 1.2
MAX_HOLD_PERIOD: int = 24

# --- متغيرات عالمية مشتركة بين العمليات ---
manager = Manager()
btc_data_cache = manager.dict()

# --- دوال الاتصال والتحقق ---
def get_db_connection():
    """إنشاء اتصال جديد بقاعدة البيانات لكل عملية."""
    return psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)

def get_binance_client():
    """إنشاء عميل Binance جديد لكل عملية."""
    return Client(API_KEY, API_SECRET)


# --- دوال جلب ومعالجة البيانات ---
def fetch_historical_data(client, symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """جلب البيانات التاريخية من Binance."""
    try:
        start_str = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(klines, columns=cols + ['_'] * 6)
        
        for col in cols[1:]: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df[cols[1:]].dropna()
    except Exception as e:
        logger.error(f"❌ [Data Fetch] خطأ في جلب بيانات {symbol}: {e}")
        return None

def fetch_and_cache_btc_data_global():
    """جلب بيانات BTC مرة واحدة وتخزينها في القاموس المشترك."""
    logger.info("ℹ️ [BTC Data] جاري جلب بيانات البيتكوين وتخزينها...")
    client = get_binance_client()
    btc_df = fetch_historical_data(client, BTC_SYMBOL, SIGNAL_TIMEFRAME, DATA_LOOKBACK_DAYS)
    if btc_df is None:
        logger.critical("❌ [BTC Data] فشل جلب بيانات البيتكوين."); exit(1)
        
    btc_df['btc_log_return'] = np.log(btc_df['close'] / btc_df['close'].shift(1))
    btc_data_cache['df'] = btc_df.dropna()
    logger.info("✅ [BTC Data] تم تخزين بيانات البيتكوين بنجاح.")

def engineer_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """
    هندسة الميزات باستخدام pandas-ta للسرعة.
    """
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
    """
    النسخة الموجهة (Vectorized) والأسرع لتوليد الأهداف.
    """
    upper_barrier = prices + (atr * TP_ATR_MULTIPLIER)
    lower_barrier = prices - (atr * SL_ATR_MULTIPLIER)
    
    future_highs = prices.shift(-1).rolling(window=MAX_HOLD_PERIOD, min_periods=1).max()
    future_lows = prices.shift(-1).rolling(window=MAX_HOLD_PERIOD, min_periods=1).min()
    
    profit_hit = future_highs >= upper_barrier
    loss_hit = future_lows <= lower_barrier
    
    labels = pd.Series(0, index=prices.index, dtype=int)
    labels.loc[profit_hit & ~loss_hit] = 1
    labels.loc[~profit_hit & loss_hit] = -1
    
    both_hit = profit_hit & loss_hit
    labels.loc[both_hit] = -1 
    
    return labels

def train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    """تدريب النموذج بمعلمات خفيفة للسرعة."""
    lgbm_params = {
        'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
        'boosting_type': 'gbdt', 
        'n_estimators': 500,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'seed': 42, 'n_jobs': 1,
        'verbose': -1,
    }

    model = lgb.LGBMClassifier(**lgbm_params)
    
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler()
    X.loc[:, numerical_features] = scaler.fit_transform(X[numerical_features])
    
    categorical_features = ['hour', 'day_of_week']
    for col in categorical_features:
        if col in X.columns: X[col] = X[col].astype('category')
            
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

# --- الدالة الرئيسية التي سيتم تشغيلها لكل عملية ---
def process_symbol(symbol: str):
    """
    الدالة التي تقوم بكامل عملية التدريب لرمز واحد.
    """
    try:
        logger.info(f"⚙️ [Process] بدء معالجة {symbol}...")
        client = get_binance_client()
        conn = get_db_connection()
        
        hist_data = fetch_historical_data(client, symbol, SIGNAL_TIMEFRAME, DATA_LOOKBACK_DAYS)
        if hist_data is None or hist_data.empty:
            logger.warning(f"⚠️ [{symbol}] لا توجد بيانات تاريخية.")
            return (symbol, 'No Data', None)

        btc_df_from_cache = btc_data_cache.get('df')
        if btc_df_from_cache is None:
             logger.error(f"❌ [{symbol}] لم يتم العثور على بيانات BTC في الذاكرة المؤقتة.")
             return (symbol, 'BTC Data Missing', None)
             
        df_featured = engineer_features(hist_data, btc_df_from_cache)
        
        df_featured['target'] = get_vectorized_labels(df_featured['close'], df_featured['atr'])
        df_featured['target_mapped'] = df_featured['target'].map({-1: 0, 0: 1, 1: 2})
        
        feature_columns = [col for col in df_featured.columns if col in ['atr', 'rsi', 'macd_hist', 'log_return', 'relative_volume', 'btc_correlation', 'hour', 'day_of_week']]
        
        df_cleaned = df_featured.dropna(subset=feature_columns + ['target_mapped'])
        if df_cleaned.empty or df_cleaned['target_mapped'].nunique() < 3:
            logger.warning(f"⚠️ [{symbol}] بيانات غير كافية بعد التنظيف.")
            return (symbol, 'Insufficient Data', None)

        X = df_cleaned[feature_columns]
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
            
            logger.info(f"✅ [{symbol}] تم تدريب وحفظ النموذج بنجاح.")
            return (symbol, 'Success', metrics)
        else:
            logger.warning(f"⚠️ [{symbol}] النموذج لم يحقق الأداء المطلوب.")
            return (symbol, 'Low Performance', metrics)
            
    except Exception as e:
        logger.critical(f"❌ [{symbol}] خطأ فادح في معالجة الرمز: {e}", exc_info=True)
        return (symbol, 'Error', None)
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def send_telegram_notification(text: str):
    """إرسال إشعار إلى تيليجرام."""
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      json={'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e:
        logger.error(f"❌ [Telegram] فشل إرسال الإشعار: {e}")

# ==============================================================================
# ✨ دالة جديدة للتحقق من العملات المتاحة ✨
# ==============================================================================
def filter_tradable_symbols(client: Client, symbols_to_check: List[str]) -> List[str]:
    """
    تقوم بتصفية قائمة العملات للتأكد من أنها متاحة وقابلة للتداول على Binance.
    """
    logger.info("ℹ️ [Validation] جاري التحقق من العملات المتاحة على Binance...")
    try:
        exchange_info = client.get_exchange_info()
        
        # إنشاء مجموعة (set) من العملات المتاحة للتداول مع USDT للبحث السريع
        available_symbols = {
            s['symbol'] 
            for s in exchange_info['symbols'] 
            if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING'
        }
        
        # تحويل قائمة الإدخال إلى مجموعة لإزالة التكرارات
        symbols_set = set(symbols_to_check)
        
        # العثور على العملات القابلة للتداول
        tradable = list(symbols_set.intersection(available_symbols))
        
        # العثور على العملات غير المتاحة
        untradable = list(symbols_set.difference(available_symbols))
        
        if untradable:
            logger.warning(f"⚠️ [Validation] العملات التالية تم تخطيها لأنها غير متاحة للتداول: {', '.join(untradable)}")
        
        logger.info(f"✅ [Validation] تم العثور على {len(tradable)} عملة قابلة للتداول من أصل {len(symbols_to_check)}.")
        return tradable
        
    except Exception as e:
        logger.error(f"❌ [Validation] خطأ أثناء التحقق من العملة من Binance: {e}. سيتم تخطي عملية التحقق.")
        # في حالة الفشل، قم بإرجاع القائمة الأصلية لتجنب إيقاف العملية بأكملها
        return symbols_to_check

# --- دالة التدريب الرئيسية التي تدير العمليات المتوازية ---
def parallel_training_job():
    logger.info(f"🚀 بدء عملية التدريب المحسنة ({BASE_ML_MODEL_NAME})...")
    
    fetch_and_cache_btc_data_global()
    
    try:
        with open('crypto_list.txt', 'r', encoding='utf-8') as f:
            symbols_from_file = {s.strip().upper() + 'USDT' for s in f if s.strip()}
    except FileNotFoundError:
        logger.critical("❌ [Main] ملف 'crypto_list.txt' غير موجود."); return

    # --- ✨ التحقق من العملات قبل البدء ✨ ---
    client = get_binance_client() 
    if not client:
        logger.critical("❌ [Main] فشل في إنشاء عميل Binance للتحقق من العملات."); return

    tradable_symbols = filter_tradable_symbols(client, list(symbols_from_file))
    if not tradable_symbols:
        logger.warning("⚠️ [Main] لم يتم العثور على أي عملات قابلة للتداول من القائمة المحددة.")
        return
    # --- نهاية قسم التحقق ---

    send_telegram_notification(f"🚀 *بدء تدريب محسن لـ {len(tradable_symbols)} عملة*...")
    
    num_processes = cpu_count()
    logger.info(f"🖥️ استخدام {num_processes} عمليات متوازية للتدريب.")
    
    results = []
    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(tradable_symbols), desc="Training Symbols") as pbar:
            for result in pool.imap_unordered(process_symbol, tradable_symbols):
                results.append(result)
                pbar.update()

    successful = sum(1 for r in results if r[1] == 'Success')
    failed = len(tradable_symbols) - successful
    
    summary_msg = (f"🏁 *اكتملت عملية التدريب المحسنة*\n"
                   f"- النماذج الناجحة: {successful}\n"
                   f"- النماذج الفاشلة/المتجاهَلة: {failed}")
    send_telegram_notification(summary_msg)
    logger.info(summary_msg)

# --- خادم Flask للبقاء نشطًا على Render ---
app = Flask(__name__)
@app.route('/')
def health_check():
    return "خدمة تدريب النماذج المحسنة تعمل.", 200

if __name__ == "__main__":
    train_thread = Thread(target=parallel_training_job)
    train_thread.daemon = True
    train_thread.start()
    
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🌍 تشغيل خادم الويب على المنفذ {port}...")
    app.run(host='0.0.0.0', port=port)
