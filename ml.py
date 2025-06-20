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
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from binance.client import Client
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Tuple
from sklearn.model_selection import TimeSeriesSplit # سنستخدمها بعناية
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from flask import Flask
from threading import Thread

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
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
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V6_Enhanced' # تحديث اسم النموذج
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 180 # زيادة فترة البحث عن البيانات
BTC_SYMBOL = 'BTCUSDT'

# Indicator & Feature Parameters (محدثة مع ميزات إضافية ونوافذ متعددة)
RSI_PERIODS: List[int] = [14, 21] # فترات RSI متعددة
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD: int = 14
EMA_TREND_PERIODS: Dict[str, int] = {'fast': 50, 'slow': 200, 'mid': 100} # EMA جديدة للاتجاه
VOL_MA_PERIOD: int = 30 # لمتوسط الحجم المتحرك
CORR_PERIODS: List[int] = [30, 60] # فترات ارتباط متعددة
ROLLING_VOLATILITY_PERIODS: List[int] = [10, 20, 50] # فترات تقلبات متدحرجة

# Triple-Barrier Method Parameters
TP_ATR_MULTIPLIER: float = 2.0
SL_ATR_MULTIPLIER: float = 1.5
MAX_HOLD_PERIOD: int = 48 # زيادة فترة الاحتفاظ القصوى (48 * 15 دقيقة = 12 ساعة)

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
        # استخدام datetime.now() بدلاً من utcnow() لتجنب المشاكل الزمنية المحتملة مع binance
        start_str = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"ℹ️ [Data] Fetching {symbol} data from {start_str} with interval {interval}")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines:
            logger.warning(f"⚠️ [Data] No klines data returned for {symbol}.")
            return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        # إزالة الأعمدة غير المستخدمة قبل الإرجاع
        return df[numeric_cols].dropna()
    except Exception as e:
        logger.error(f"❌ [Data] خطأ أثناء جلب البيانات لـ {symbol}: {e}", exc_info=True)
        return None

def fetch_and_cache_btc_data():
    global btc_data_cache
    logger.info("ℹ️ [BTC Data] جاري جلب بيانات البيتكوين وتخزينها...")
    btc_data_cache = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
    if btc_data_cache is None:
        logger.critical("❌ [BTC Data] فشل جلب بيانات البيتكوين."); exit(1)
    # إضافة ميزة عوائد البيتكوين مباشرة هنا
    btc_data_cache['btc_log_returns'] = np.log(btc_data_cache['close'] / btc_data_cache['close'].shift(1))
    btc_data_cache.dropna(inplace=True) # تنظيف أي NaNs بعد حساب العوائد

def calculate_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()

    # --- ميزات السعر والحجم الأساسية ---
    df_calc['log_return_1_period'] = np.log(df_calc['close'] / df_calc['close'].shift(1))
    df_calc['log_return_2_period'] = np.log(df_calc['close'] / df_calc['close'].shift(2))
    df_calc['log_return_5_period'] = np.log(df_calc['close'] / df_calc['close'].shift(5))
    df_calc['volume_change'] = df_calc['volume'].diff() / df_calc['volume'].shift(1)
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=VOL_MA_PERIOD, min_periods=1).mean() + 1e-9)

    # --- ميزات التحليل الفني ---
    # ATR
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    # RSI (فترات متعددة)
    for period in RSI_PERIODS:
        delta = df_calc['close'].diff()
        gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(com=period - 1, adjust=False).mean()
        # تجنب القسمة على صفر أو قيم صغيرة جداً
        rs = gain / loss.replace(0, 1e-9)
        df_calc[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df_calc['macd_hist'] = macd_line - macd_signal_line

    # EMA Crossings / Deviations
    df_calc['ema_fast_trend'] = df_calc['close'].ewm(span=EMA_TREND_PERIODS['fast'], adjust=False).mean()
    df_calc['ema_mid_trend'] = df_calc['close'].ewm(span=EMA_TREND_PERIODS['mid'], adjust=False).mean()
    df_calc['ema_slow_trend'] = df_calc['close'].ewm(span=EMA_TREND_PERIODS['slow'], adjust=False).mean()

    df_calc['price_vs_ema_fast'] = (df_calc['close'] / df_calc['ema_fast_trend']) - 1
    df_calc['price_vs_ema_mid'] = (df_calc['close'] / df_calc['ema_mid_trend']) - 1
    df_calc['price_vs_ema_slow'] = (df_calc['close'] / df_calc['ema_slow_trend']) - 1
    df_calc['ema_fast_vs_mid'] = (df_calc['ema_fast_trend'] / df_calc['ema_mid_trend']) - 1
    df_calc['ema_mid_vs_slow'] = (df_calc['ema_mid_trend'] / df_calc['ema_slow_trend']) - 1


    # --- ميزات التقلب (Volatility Features) ---
    for period in ROLLING_VOLATILITY_PERIODS:
        df_calc[f'rolling_vol_{period}'] = df_calc['log_return_1_period'].rolling(window=period).std()

    # --- ميزات العلاقات بين الأصول (مع BTC) ---
    # دمج بيانات BTC مع بيانات العملة الحالية بناءً على الفهرس الزمني
    # التأكد من أن الفهرس متسق بين df_calc و btc_df
    temp_df = pd.merge(df_calc[['log_return_1_period']], btc_df[['btc_log_returns']], left_index=True, right_index=True, how='left')
    
    for period in CORR_PERIODS:
        # حساب الارتباط المتدحرج.fillna(0) مهم للتعامل مع الفترات التي لا يوجد فيها ارتباط
        df_calc[f'btc_correlation_{period}'] = temp_df['log_return_1_period'].rolling(window=period).corr(temp_df['btc_log_returns']).fillna(0)

    # --- ميزات الوقت ---
    df_calc['hour_of_day'] = df_calc.index.hour.astype('category')
    df_calc['day_of_week'] = df_calc.index.dayofweek.astype('category')
    df_calc['day_of_month'] = df_calc.index.day.astype('category')
    df_calc['month'] = df_calc.index.month.astype('category')

    # ميزات دورية للوقت (Sine/Cosine)
    df_calc['hour_sin'] = np.sin(2 * np.pi * df_calc.index.hour / 24)
    df_calc['hour_cos'] = np.cos(2 * np.pi * df_calc.index.hour / 24)
    df_calc['day_of_week_sin'] = np.sin(2 * np.pi * df_calc.index.dayofweek / 7)
    df_calc['day_of_week_cos'] = np.cos(2 * np.pi * df_calc.index.dayofweek / 7)


    # تنظيف أي أعمدة مؤقتة استخدمت لحساب الميزات
    df_calc = df_calc.drop(columns=[col for col in ['ema_fast_trend', 'ema_mid_trend', 'ema_slow_trend'] if col in df_calc.columns], errors='ignore')

    return df_calc.dropna() # إزالة أي صفوف تحتوي على NaN بعد حساب جميع الميزات

def get_triple_barrier_labels(prices: pd.Series, atr: pd.Series) -> pd.Series:
    labels = pd.Series(0, index=prices.index, dtype=int) # تأكد من نوع البيانات int
    # التأكد من أن prices و atr لها نفس الفهرس
    prices = prices.copy()
    atr = atr.copy()

    # استخدام to_numpy() لتحسين الأداء
    prices_np = prices.to_numpy()
    atr_np = atr.to_numpy()

    for i in tqdm(range(len(prices) - MAX_HOLD_PERIOD), desc="Labeling", leave=False):
        entry_price = prices_np[i]
        current_atr = atr_np[i]

        if np.isnan(current_atr) or current_atr <= 0: # التعامل مع ATR = 0
            continue

        upper_barrier = entry_price + (current_atr * TP_ATR_MULTIPLIER)
        lower_barrier = entry_price - (current_atr * SL_ATR_MULTIPLIER)

        # البحث عن اختراق الحاجز أو انتهاء فترة الاحتفاظ
        # استخدام نافذة على numpy array لتحسين الأداء
        window_prices = prices_np[i + 1 : i + 1 + MAX_HOLD_PERIOD]

        # تحقق من اختراق الحاجز العلوي (ربح)
        if (window_prices >= upper_barrier).any():
            labels.iloc[i] = 1 # Profit
            continue
        
        # تحقق من اختراق الحاجز السفلي (خسارة)
        if (window_prices <= lower_barrier).any():
            labels.iloc[i] = -1 # Loss
            continue
        
        # إذا لم يتم اختراق أي حاجز خلال MAX_HOLD_PERIOD، يبقى 0 (لا شيء)
        # هذا هو السلوك الافتراضي الذي تم تعيينه في البداية

    return labels

def prepare_data_for_ml(df: pd.DataFrame, btc_df: pd.DataFrame, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str], List[str]]]:
    logger.info(f"ℹ️ [ML Prep] Preparing data for {symbol}...")
    df_featured = calculate_features(df, btc_df)

    # قبل حساب الهدف، تأكد من أن 'atr' ليس فارغًا
    if 'atr' not in df_featured.columns or df_featured['atr'].isnull().all():
        logger.warning(f"⚠️ [ML Prep] 'atr' feature is missing or all NaN for {symbol}. Cannot generate labels.")
        return None

    # تطبيق دالة الحواجز الثلاثية للحصول على الهدف (-1, 0, 1)
    df_featured['target'] = get_triple_barrier_labels(df_featured['close'], df_featured['atr'])

    # قائمة الميزات الرقمية والفئوية
    # سيتم إنشاء هذه القائمة ديناميكيًا بناءً على الميزات المحسوبة
    
    numerical_features = [
        'log_return_1_period', 'log_return_2_period', 'log_return_5_period',
        'volume_change', 'relative_volume', 'atr', 'macd_hist',
        'price_vs_ema_fast', 'price_vs_ema_mid', 'price_vs_ema_slow',
        'ema_fast_vs_mid', 'ema_mid_vs_slow',
        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos'
    ]
    # إضافة ميزات RSI و Rolling Volatility و BTC Correlation ديناميكيًا
    for period in RSI_PERIODS:
        numerical_features.append(f'rsi_{period}')
    for period in ROLLING_VOLATILITY_PERIODS:
        numerical_features.append(f'rolling_vol_{period}')
    for period in CORR_PERIODS:
        numerical_features.append(f'btc_correlation_{period}')

    categorical_features_for_lgbm = [
        'hour_of_day', 'day_of_week', 'day_of_month', 'month'
    ]

    # تصفية الميزات للتأكد من وجودها بعد عملية حساب الميزات وإزالة NaNs
    final_feature_columns = [col for col in numerical_features + categorical_features_for_lgbm if col in df_featured.columns]

    df_cleaned = df_featured.dropna(subset=final_feature_columns + ['target']).copy()

    # إزالة الفئات التي لا تحتوي على عينات كافية بعد التنظيف
    for cat_col in categorical_features_for_lgbm:
        if cat_col in df_cleaned.columns:
            # قم بتحويل العمود إلى نوع 'category' إذا لم يكن كذلك بالفعل
            df_cleaned[cat_col] = df_cleaned[cat_col].astype('category')
            # إزالة أي فئات لا تظهر في البيانات بعد التنظيف
            df_cleaned[cat_col] = df_cleaned[cat_col].cat.remove_unused_categories()

    if df_cleaned.empty or df_cleaned['target'].nunique() < 2:
        logger.warning(f"⚠️ [ML Prep] Data for {symbol} is empty or has less than 2 target classes after feature engineering. Skipping.")
        return None

    # إعادة تعيين قيم الهدف إلى 0, 1, 2 إذا كانت -1, 0, 1، لأن LightGBM يفضل 0-indexed للتصنيف المتعدد
    # Mapping: -1 -> 0 (Loss), 0 -> 1 (Neutral), 1 -> 2 (Profit)
    df_cleaned['target_mapped'] = df_cleaned['target'].map({-1: 0, 0: 1, 1: 2})
    
    # تأكد من أن جميع الميزات الفئوية معرفة بشكل صحيح لـ LightGBM
    # LightGBM يمكنه التعامل مع 'category' dtype مباشرة
    for col in categorical_features_for_lgbm:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype('category')

    X = df_cleaned[final_feature_columns]
    y = df_cleaned['target_mapped'] # استخدام الهدف المعدل

    logger.info(f"📊 [ML Prep] Target distribution for {symbol} (mapped: 0,1,2):\n{y.value_counts(normalize=True)}")
    
    return X, y, numerical_features, categorical_features_for_lgbm # تمرير قوائم الميزات المنفصلة

# --- دالة التدريب الرئيسية مع تحسينات LightGBM ---
def train_with_walk_forward_validation(X: pd.DataFrame, y: pd.Series, numerical_features: List[str], categorical_features: List[str]) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    logger.info("ℹ️ [ML Train] Starting training with Walk-Forward Validation...")
    
    # استخدام TimeSeriesSplit بشكل صحيح مع عدد كافٍ من الانقسامات.
    # n_splits=5 يعني 5 أضعاف، كل ضعف يستخدم جزءًا أكبر من البيانات للتدريب.
    # max_train_size يمكن أن يحد من حجم نافذة التدريب
    tscv = TimeSeriesSplit(n_splits=5) 
    
    final_model, final_scaler = None, None
    fold_metrics = [] # لتخزين مقاييس كل ضعف

    # معلمات LightGBM المقترحة لتحقيق أداء "قياسي"
    lgbm_params = {
        'objective': 'multiclass', # لأنه لدينا 3 فئات (-1, 0, 1) التي تم تعيينها إلى (0, 1, 2)
        'num_class': 3,            # 3 فئات
        'metric': 'multi_logloss', # مقياس مناسب للتصنيف المتعدد
        'boosting_type': 'gbdt',
        'num_leaves': 63,          # قيمة جيدة، تسمح بتعقيد معقول
        'max_depth': -1,           # السماح لـ num_leaves بالتحكم
        'learning_rate': 0.02,     # معدل تعلم صغير
        'feature_fraction': 0.7,   # أخذ عينات من 70% من الميزات
        'bagging_fraction': 0.7,   # أخذ عينات من 70% من البيانات
        'bagging_freq': 1,         # تكرار الـ bagging
        'lambda_l1': 0.1,          # تقييد L1
        'lambda_l2': 0.1,          # تقييد L2
        'min_child_samples': 200,  # عدد كبير في العقدة الطرفية لتقليل التجاوز
        'verbose': -1,             # إخفاء المخرجات
        'n_jobs': -1,              # استخدام جميع النوى
        'seed': 42,                # للتحقق
        'is_unbalance': True       # مهم للفئات غير المتوازنة
    }

    # تحديد الميزات الفئوية لـ LightGBM (يجب أن تكون كـ 'category' dtype)
    lgbm_categorical_features = [col for col in categorical_features if col in X.columns]
    
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # التأكد من وجود عينات كافية في كل مجموعة فرعية للتدريب
        if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
            logger.warning(f"⚠️ [ML Train] Fold {i+1} has empty train/test sets. Skipping.")
            continue
        
        # التأكد من وجود أكثر من فئة واحدة في y_train و y_test
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            logger.warning(f"⚠️ [ML Train] Fold {i+1} train/test target has less than 2 classes. Skipping.")
            continue

        # تطبيق StandardScaler فقط على الميزات الرقمية
        scaler = StandardScaler()
        # Fit scaler only on numerical features of the training set
        scaler.fit(X_train[numerical_features])
        
        # Transform both train and test sets, retaining original column names
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[numerical_features] = scaler.transform(X_train[numerical_features])
        X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
        
        # إعادة تعيين الميزات الفئوية إلى نوع 'category' بعد العمليات السابقة (إن لم تكن كذلك)
        for col in lgbm_categorical_features:
            if col in X_train_scaled.columns:
                X_train_scaled[col] = X_train_scaled[col].astype('category')
            if col in X_test_scaled.columns:
                X_test_scaled[col] = X_test_scaled[col].astype('category')

        # تدريب LightGBM
        model = lgb.train(
            lgbm_params,
            lgb.Dataset(X_train_scaled, y_train, categorical_feature=lgbm_categorical_features),
            num_boost_round=2000, # عدد كبير للسماح بالتوقف المبكر
            valid_sets=[lgb.Dataset(X_test_scaled, y_test, categorical_feature=lgbm_categorical_features)],
            callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)], # توقف مبكر
            # feature_name = list(X.columns) # للتأكد من استخدام أسماء الميزات الصحيحة
        )
        
        # التنبؤ والتقييم للضعف الحالي
        y_pred = model.predict(X_test_scaled, num_iteration=model.best_iteration)
        y_pred_labels = np.argmax(y_pred, axis=1) # للحصول على الفئة المتوقعة من الاحتمالات

        # مقاييس لكل ضعف
        acc = accuracy_score(y_test, y_pred_labels)
        # Precision و Recall و F1 لكل فئة على حدة
        # نركز على فئة الربح (mapped to 2) أو فئة الخسارة (mapped to 0)
        precision_profit = precision_score(y_test, y_pred_labels, labels=[2], average='macro', zero_division=0)
        recall_profit = recall_score(y_test, y_pred_labels, labels=[2], average='macro', zero_division=0)
        f1_profit = f1_score(y_test, y_pred_labels, labels=[2], average='macro', zero_division=0)
        
        # يمكنك إضافة مقاييس لفئة الخسارة إذا أردت
        # precision_loss = precision_score(y_test, y_pred_labels, labels=[0], average='macro', zero_division=0)

        fold_metrics.append({
            'accuracy': acc,
            'precision_profit': precision_profit,
            'recall_profit': recall_profit,
            'f1_profit': f1_profit,
            'best_iteration': model.best_iteration
        })

        logger.info(f"--- Fold {i+1}: Accuracy: {acc:.4f}, Precision (Profit): {precision_profit:.4f}, Recall (Profit): {recall_profit:.4f}, F1 (Profit): {f1_profit:.4f}, Best Iteration: {model.best_iteration}")
        
        final_model, final_scaler = model, scaler # حفظ آخر نموذج ومدرب

    if not final_model or not final_scaler:
        logger.error("❌ [ML Train] Training failed, no model was created or all folds skipped.")
        return None, None, None

    # حساب المتوسطات للمقاييس عبر جميع الأضعاف
    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
    avg_precision_profit = np.mean([m['precision_profit'] for m in fold_metrics])
    avg_recall_profit = np.mean([m['recall_profit'] for m in fold_metrics])
    avg_f1_profit = np.mean([m['f1_profit'] for m in fold_metrics])

    final_metrics = {
        'avg_accuracy': avg_accuracy,
        'avg_precision_profit': avg_precision_profit,
        'avg_recall_profit': avg_recall_profit,
        'avg_f1_profit': avg_f1_profit,
        'num_samples_trained': len(X),
        'num_folds': len(fold_metrics)
    }

    metrics_log_str = ', '.join([f"{k}: {v:.4f}" for k, v in final_metrics.items() if isinstance(v, (int, float))])
    logger.info(f"📊 [ML Train] Average Walk-Forward Performance: {metrics_log_str}")
    
    # ملاحظة: يتم حفظ آخر نموذج ومدرب. لنهج أكثر قوة، يمكنك تدريب نموذج نهائي على مجموعة بيانات أكبر
    # أو متوسط النماذج، ولكن هذا يعقد العملية. لهذا النهج "القياسي"، نستخدم آخر نموذج كـ "أفضل نموذج"
    # إذا كان أداؤه هو الأفضل عبر جميع الأضعاف.
    # بدلاً من ذلك، يمكنك اختيار النموذج الذي حقق أفضل مقياس على مجموعة التحقق من بين جميع الأضعاف.
    # للتبسيط، نستخدم آخر نموذج مدرب هنا.

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
        logger.error(f"❌ [DB Save] Error saving model bundle: {e}", exc_info=True); conn.rollback()

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.warning("⚠️ [Telegram] TELEGRAM_TOKEN or CHAT_ID not configured. Skipping Telegram message.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(url, json={'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}, timeout=10)
        response.raise_for_status() # يثير استثناء لأخطاء HTTP
        logger.info("✅ [Telegram] Telegram message sent successfully.")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [Telegram] Failed to send Telegram message: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"❌ [Telegram] An unexpected error occurred while sending Telegram message: {e}", exc_info=True)


# --- دالة التدريب الرئيسية للعمل في خيط منفصل ---
def run_training_job():
    logger.info(f"🚀 Starting ADVANCED ML model training job ({BASE_ML_MODEL_NAME})...")
    init_db()
    get_binance_client()
    fetch_and_cache_btc_data()
    symbols_to_train = get_validated_symbols(filename='crypto_list.txt')
    if not symbols_to_train:
        logger.critical("❌ [Main] No valid symbols found. Exiting training job.")
        if conn: conn.close()
        return
        
    send_telegram_message(f"🚀 *{BASE_ML_MODEL_NAME} Training Started*\nWill train models for {len(symbols_to_train)} symbols.")
    
    successful_models, failed_models = 0, 0
    # استخدام tqdm لعرض شريط تقدم لتدريب الرموز
    for symbol in tqdm(symbols_to_train, desc="Overall Symbol Training"):
        logger.info(f"\n--- ⏳ [Main] Starting model training for {symbol} ---")
        try:
            df_hist = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
            if df_hist is None or df_hist.empty:
                logger.warning(f"⚠️ [Main] No historical data for {symbol}, skipping."); failed_models += 1; continue
            
            # تمرير قوائم الميزات الرقمية والفئوية إلى prepare_data_for_ml
            prepared_data = prepare_data_for_ml(df_hist, btc_data_cache, symbol)
            if prepared_data is None:
                failed_models += 1; continue
            X, y, numerical_features, categorical_features = prepared_data
            
            # التأكد من أن y يحتوي على قيم صالحة
            if y.isnull().any() or y.nunique() < 2:
                logger.warning(f"⚠️ [Main] Target for {symbol} contains NaN or has less than 2 unique classes after preparation. Skipping.")
                failed_models += 1
                continue

            # تمرير قوائم الميزات إلى دالة التدريب
            training_result = train_with_walk_forward_validation(X, y, numerical_features, categorical_features)
            if not all(res is not None for res in training_result): # التحقق من أن جميع العناصر ليست None
                 logger.warning(f"⚠️ [Main] Training for {symbol} resulted in None values. Skipping."); failed_models += 1; continue
            
            final_model, final_scaler, model_metrics = training_result
            
            # تقييم الأداء بناءً على مقياس الربح (precision_profit) وليس فقط الدقة
            # يمكن تعديل هذا العتبة بناءً على مدى عدوانية استراتيجيتك
            if final_model and final_scaler and model_metrics.get('avg_precision_profit', 0) > 0.40: # عتبة أعلى للتدريب القياسي
                model_bundle = {
                    'model': final_model,
                    'scaler': final_scaler,
                    'feature_names': list(X.columns), # حفظ أسماء جميع الميزات لضمان التناسق
                    'numerical_features': numerical_features,
                    'categorical_features': categorical_features
                }
                model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
                save_ml_model_to_db(model_bundle, model_name, model_metrics)
                successful_models += 1
                send_telegram_message(f"✅ *Model Trained for {symbol}*\n_Avg Precision (Profit): {model_metrics['avg_precision_profit']:.4f}_")
            else:
                logger.warning(f"⚠️ [Main] Model for {symbol} did not meet performance criteria (Avg Precision Profit < 0.40). Discarding."); failed_models += 1
        except Exception as e:
            logger.critical(f"❌ [Main] A fatal error occurred for {symbol}: {e}", exc_info=True); failed_models += 1
        time.sleep(1) # تأخير قصير بين تدريب الرموز

    completion_message = (f"✅ *{BASE_ML_MODEL_NAME} Training Finished*\n"
                        f"- Successfully trained: {successful_models} models\n"
                        f"- Failed/Discarded: {failed_models} models\n"
                        f"- Total symbols: {len(symbols_to_train)}")
    send_telegram_message(completion_message)
    logger.info(completion_message)

    if conn:
        conn.close()
        logger.info("👋 [Main] Database connection closed.")
    logger.info("👋 [Main] ML training job finished.")

# --- إضافة خادم Flask للعمل على Render ---
app = Flask(__name__)

@app.route('/')
def health_check():
    """Endpoint for Render health checks."""
    return "ML Trainer service is running and healthy.", 200

if __name__ == "__main__":
    # بدء عملية التدريب في خيط منفصل حتى لا تمنع الخادم من العمل
    training_thread = Thread(target=run_training_job)
    training_thread.daemon = True # سيتم إنهاء الخيط عند إنهاء البرنامج الرئيسي
    training_thread.start()
    
    # تشغيل خادم الويب
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🌍 Starting web server on port {port} to keep the service alive...")
    # debug=True لا ينصح به في بيئات الإنتاج، قم بإزالته عند النشر الفعلي
    app.run(host='0.0.0.0', port=port)