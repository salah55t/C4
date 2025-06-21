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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from flask import Flask
from threading import Thread

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
# إعداد المسجل لتتبع عمليات البرنامج وحفظها في ملف
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_model_trainer_enhanced.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoMLTrainer')

# ---------------------- تحميل متغيرات البيئة ----------------------
# تحميل المتغيرات الحساسة مثل مفاتيح API من ملف .env
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
    TELEGRAM_TOKEN: Optional[str] = config('TELEGRAM_BOT_TOKEN', default=None)
    CHAT_ID: Optional[str] = config('TELEGRAM_CHAT_ID', default=None)
except Exception as e:
     logger.critical(f"❌ فشل في تحميل متغيرات البيئة الأساسية. تأكد من وجود ملف .env: {e}")
     exit(1)

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
BASE_ML_MODEL_NAME: str = 'LightGBM_Crypto_Predictor_V7'
SIGNAL_TIMEFRAME: str = '15m'  # الإطار الزمني للتحليل
DATA_LOOKBACK_DAYS: int = 200 # زيادة فترة جلب البيانات لتدريب أفضل
BTC_SYMBOL = 'BTCUSDT'

# --- معلمات هندسة الميزات ---
RSI_PERIODS: List[int] = [14, 28]
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD: int = 14
EMA_PERIODS: Dict[str, int] = {'fast': 50, 'slow': 200}
VOLUME_MA_PERIOD: int = 20
BTC_CORR_PERIOD: int = 50

# --- معلمات طريقة الحاجز الثلاثي لتحديد الهدف ---
TP_ATR_MULTIPLIER: float = 2.0  # مضاعف ATR لتحديد هدف الربح
SL_ATR_MULTIPLIER: float = 1.5  # مضاعف ATR لتحديد وقف الخسارة
MAX_HOLD_PERIOD: int = 48 # أقصى عدد شموع للاحتفاظ بالصفقة (48 * 15m = 12 hours)

# --- متغيرات عالمية ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
btc_data_cache: Optional[pd.DataFrame] = None

# --- دوال الاتصال والتحقق ---
def initialize_database():
    """تهيئة الاتصال بقاعدة البيانات وإنشاء جدول النماذج إذا لم يكن موجودًا."""
    global conn
    try:
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ml_models (
                    id SERIAL PRIMARY KEY,
                    model_name TEXT NOT NULL UNIQUE,
                    model_data BYTEA NOT NULL,
                    trained_at TIMESTAMP DEFAULT NOW(),
                    metrics JSONB
                );
            """)
        conn.commit()
        logger.info("✅ [DB] تم تهيئة قاعدة البيانات بنجاح.")
    except Exception as e:
        logger.critical(f"❌ [DB] فشل الاتصال بقاعدة البيانات: {e}"); exit(1)

def initialize_binance_client():
    """تهيئة عميل Binance للاتصال بـ API."""
    global client
    try:
        client = Client(API_KEY, API_SECRET)
        client.ping()
        logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
    except Exception as e:
        logger.critical(f"❌ [Binance] فشل تهيئة عميل Binance: {e}"); exit(1)

# --- دوال جلب ومعالجة البيانات ---
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """جلب البيانات التاريخية من Binance لرمز معين."""
    try:
        start_str = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"ℹ️ [Data] جلب بيانات {symbol} من تاريخ {start_str} بإطار زمني {interval}")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines:
            logger.warning(f"⚠️ [Data] لم يتم العثور على بيانات لـ {symbol}.")
            return None
            
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df[numeric_cols].dropna()
    except Exception as e:
        logger.error(f"❌ [Data] خطأ أثناء جلب البيانات لـ {symbol}: {e}")
        return None

def fetch_and_cache_btc_data():
    """جلب بيانات البيتكوين وتخزينها مؤقتًا لتسريع العمليات."""
    global btc_data_cache
    logger.info("ℹ️ [BTC Data] جاري جلب بيانات البيتكوين وتخزينها...")
    btc_data_cache = fetch_historical_data(BTC_SYMBOL, SIGNAL_TIMEFRAME, DATA_LOOKBACK_DAYS)
    if btc_data_cache is None:
        logger.critical("❌ [BTC Data] فشل جلب بيانات البيتكوين، لا يمكن المتابعة."); exit(1)
    # حساب عوائد البيتكوين اللوغاريتمية وتخزينها
    btc_data_cache['btc_log_return'] = np.log(btc_data_cache['close'] / btc_data_cache['close'].shift(1))
    btc_data_cache.dropna(inplace=True)

def engineer_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """
    الدالة الرئيسية لهندسة الميزات.
    تقوم بحساب جميع المؤشرات الفنية والميزات المطلوبة للنموذج.
    """
    df_feat = df.copy()

    # --- 1. ميزات السعر والحجم الأساسية ---
    df_feat['log_return'] = np.log(df_feat['close'] / df_feat['close'].shift(1))
    df_feat['volume_change_ratio'] = df_feat['volume'].diff() / df_feat['volume'].shift(1).replace(0, 1e-9)
    # سيولة الشمعة الحالية مقارنة بالمتوسط
    df_feat['relative_volume'] = df_feat['volume'] / (df_feat['volume'].rolling(window=VOLUME_MA_PERIOD, min_periods=1).mean() + 1e-9)

    # --- 2. ميزات المؤشرات الفنية (RSI, MACD, ATR, EMA) ---
    # ATR (Average True Range) لقياس التقلب
    high_low = df_feat['high'] - df_feat['low']
    high_close = (df_feat['high'] - df_feat['close'].shift()).abs()
    low_close = (df_feat['low'] - df_feat['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_feat['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    # RSI (Relative Strength Index) بفترات متعددة
    for period in RSI_PERIODS:
        delta = df_feat['close'].diff()
        gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(com=period - 1, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        df_feat[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    ema_fast = df_feat['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df_feat['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df_feat['macd_hist'] = macd_line - signal_line
    
    # *** ميزة تقاطعات الماكد (جديدة) ***
    # إشارة موجبة (1) لتقاطع صعودي، وسالبة (-1) لتقاطع هبوطي
    macd_cross_above = (macd_line.shift(1) < signal_line.shift(1)) & (macd_line > signal_line)
    macd_cross_below = (macd_line.shift(1) > signal_line.shift(1)) & (macd_line < signal_line)
    df_feat['macd_cross'] = 0
    df_feat.loc[macd_cross_above, 'macd_cross'] = 1
    df_feat.loc[macd_cross_below, 'macd_cross'] = -1

    # EMA (Exponential Moving Average) لتحديد الاتجاه
    df_feat['ema_fast'] = df_feat['close'].ewm(span=EMA_PERIODS['fast'], adjust=False).mean()
    df_feat['ema_slow'] = df_feat['close'].ewm(span=EMA_PERIODS['slow'], adjust=False).mean()
    df_feat['price_vs_ema_slow'] = (df_feat['close'] / df_feat['ema_slow']) - 1
    df_feat['ema_fast_vs_ema_slow'] = (df_feat['ema_fast'] / df_feat['ema_slow']) - 1

    # --- 3. ميزات هيكل الشمعة (Candle Structure) ---
    candle_range = df_feat['high'] - df_feat['low']
    candle_body = (df_feat['close'] - df_feat['open']).abs()
    df_feat['candle_body_ratio'] = candle_body / (candle_range + 1e-9)
    df_feat['upper_wick'] = df_feat['high'] - np.maximum(df_feat['open'], df_feat['close'])
    df_feat['lower_wick'] = np.minimum(df_feat['open'], df_feat['close']) - df_feat['low']
    df_feat['upper_wick_ratio'] = df_feat['upper_wick'] / (candle_range + 1e-9)

    # --- 4. ميزات العلاقة مع البيتكوين ---
    # دمج بيانات العملة مع بيانات البيتكوين لحساب الارتباط
    merged_df = pd.merge(df_feat[['log_return']], btc_df[['btc_log_return']], left_index=True, right_index=True, how='left')
    df_feat[f'btc_correlation_{BTC_CORR_PERIOD}'] = merged_df['log_return'].rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_log_return']).fillna(0)

    # --- 5. ميزات الوقت ---
    df_feat['hour'] = df_feat.index.hour
    df_feat['day_of_week'] = df_feat.index.dayofweek

    # تنظيف الأعمدة المؤقتة
    df_feat.drop(columns=['ema_fast', 'ema_slow', 'upper_wick', 'lower_wick'], inplace=True, errors='ignore')
    
    return df_feat.dropna()


def get_triple_barrier_labels(prices: pd.Series, atr: pd.Series) -> pd.Series:
    """
    تحديد الهدف (1: ربح, -1: خسارة, 0: محايد) باستخدام طريقة الحاجز الثلاثي.
    """
    labels = pd.Series(0, index=prices.index, dtype=int)
    prices_np = prices.to_numpy()
    atr_np = atr.to_numpy()

    for i in tqdm(range(len(prices) - MAX_HOLD_PERIOD), desc="Labeling Data", leave=False, ncols=80):
        entry_price = prices_np[i]
        current_atr = atr_np[i]

        if np.isnan(current_atr) or current_atr <= 1e-9:
            continue

        upper_barrier = entry_price + (current_atr * TP_ATR_MULTIPLIER)
        lower_barrier = entry_price - (current_atr * SL_ATR_MULTIPLIER)
        
        # البحث في نافذة الأسعار المستقبلية
        future_prices = prices_np[i + 1 : i + 1 + MAX_HOLD_PERIOD]
        
        # تحقق من لمس حاجز الربح
        profit_touch_indices = np.where(future_prices >= upper_barrier)[0]
        # تحقق من لمس حاجز الخسارة
        loss_touch_indices = np.where(future_prices <= lower_barrier)[0]

        first_profit_touch = profit_touch_indices[0] if len(profit_touch_indices) > 0 else None
        first_loss_touch = loss_touch_indices[0] if len(loss_touch_indices) > 0 else None

        if first_profit_touch is not None and (first_loss_touch is None or first_profit_touch < first_loss_touch):
            labels.iloc[i] = 1  # ربح
        elif first_loss_touch is not None and (first_profit_touch is None or first_loss_touch < first_profit_touch):
            labels.iloc[i] = -1 # خسارة
        # إذا لم يلمس أي حاجز، ستبقى القيمة 0 (محايد)

    return labels


def prepare_data_for_ml(df: pd.DataFrame, btc_df: pd.DataFrame, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """تجهيز البيانات النهائية للنموذج: هندسة الميزات، تحديد الهدف، والتنظيف."""
    logger.info(f"ℹ️ [ML Prep] تجهيز بيانات {symbol} للنموذج...")
    
    df_featured = engineer_features(df, btc_df)
    
    if 'atr' not in df_featured.columns or df_featured['atr'].isnull().all():
        logger.warning(f"⚠️ [ML Prep] ميزة 'atr' غير موجودة لـ {symbol}. لا يمكن توليد الأهداف.")
        return None

    # تطبيق دالة الحاجز الثلاثي لتوليد الهدف
    df_featured['target'] = get_triple_barrier_labels(df_featured['close'], df_featured['atr'])
    
    # تحويل الهدف من (-1, 0, 1) إلى (0, 1, 2) ليتوافق مع LightGBM
    # 0: خسارة, 1: محايد, 2: ربح
    df_featured['target_mapped'] = df_featured['target'].map({-1: 0, 0: 1, 1: 2})
    
    # تحديد أعمدة الميزات المستخدمة في التدريب
    feature_columns = [col for col in df_featured.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target', 'target_mapped']]
    
    df_cleaned = df_featured.dropna(subset=feature_columns + ['target_mapped']).copy()

    if df_cleaned.empty or df_cleaned['target_mapped'].nunique() < 2:
        logger.warning(f"⚠️ [ML Prep] بيانات {symbol} فارغة أو تحتوي على أقل من فئتين بعد هندسة الميزات.")
        return None

    X = df_cleaned[feature_columns]
    y = df_cleaned['target_mapped']
    
    # تحويل الميزات الفئوية إلى النوع الصحيح لـ LightGBM
    categorical_features = ['hour', 'day_of_week', 'macd_cross']
    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].astype('category')

    logger.info(f"📊 [ML Prep] توزيع أهداف {symbol} (0=خسارة, 1=محايد, 2=ربح):\n{y.value_counts(normalize=True)}")
    
    return X, y

# --- دالة التدريب الرئيسية ---
def train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    """تدريب النموذج باستخدام التحقق المتقاطع للسلاسل الزمنية (Walk-Forward)."""
    logger.info("ℹ️ [ML Train] بدء تدريب النموذج باستخدام Walk-Forward Validation...")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    # معلمات LightGBM محسنة للموازنة بين السرعة والدقة
    lgbm_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'num_leaves': 40,
        'max_depth': 7,
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1,
        'colsample_bytree': 0.7,
        'subsample': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }

    final_model, final_scaler = None, None
    all_preds, all_true = [], []

    categorical_features_in_X = [col for col in ['hour', 'day_of_week', 'macd_cross'] if col in X.columns]

    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        if y_train.nunique() < 3: # نحتاج كل الفئات للتدريب
             logger.warning(f"⚠️ [ML Train] Fold {i+1} لا يحتوي على كل الفئات الثلاث. جاري التخطي.")
             continue
        
        # فصل الميزات الرقمية والفئوية للتطبيع
        numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
        
        scaler = StandardScaler()
        X_train.loc[:, numerical_features] = scaler.fit_transform(X_train[numerical_features])
        X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])
        
        model = lgb.LGBMClassifier(**lgbm_params)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  eval_metric='multi_logloss',
                  callbacks=[lgb.early_stopping(100, verbose=False)],
                  categorical_feature=categorical_features_in_X)

        y_pred = model.predict(X_test)
        all_preds.extend(y_pred)
        all_true.extend(y_test)
        
        final_model, final_scaler = model, scaler # حفظ آخر نموذج ومُطبِّع

    if not all_true:
        logger.error("❌ [ML Train] فشل التدريب، لم يتم إكمال أي طية بنجاح.")
        return None, None, None
        
    # حساب المقاييس النهائية على جميع بيانات الاختبار المجمعة
    accuracy = accuracy_score(all_true, all_preds)
    # حساب الدقة (Precision) لفئة الربح (2) فقط
    precision_profit = precision_score(all_true, all_preds, labels=[2], average='macro', zero_division=0)
    
    metrics = {
        'overall_accuracy': accuracy,
        'precision_for_profit_class': precision_profit,
        'num_samples_trained': len(X)
    }
    logger.info(f"📊 [ML Train] الأداء النهائي: Accuracy={accuracy:.4f}, Precision (Profit)={precision_profit:.4f}")
    
    return final_model, final_scaler, metrics

def save_model_to_db(model_bundle: Dict[str, Any], model_name: str, metrics: Dict[str, Any]):
    """حفظ حزمة النموذج (النموذج، المطبع، الميزات) في قاعدة البيانات."""
    logger.info(f"ℹ️ [DB Save] حفظ النموذج '{model_name}'...")
    try:
        model_binary = pickle.dumps(model_bundle)
        metrics_json = json.dumps(metrics)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO ml_models (model_name, model_data, metrics)
                VALUES (%s, %s, %s) ON CONFLICT (model_name) DO UPDATE SET
                model_data = EXCLUDED.model_data, trained_at = NOW(), metrics = EXCLUDED.metrics;
            """, (model_name, model_binary, metrics_json))
        conn.commit()
        logger.info(f"✅ [DB Save] تم حفظ النموذج '{model_name}' بنجاح.")
    except Exception as e:
        logger.error(f"❌ [DB Save] خطأ في حفظ النموذج: {e}"); conn.rollback()

def send_telegram_notification(text: str):
    """إرسال إشعار إلى تيليجرام."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e:
        logger.error(f"❌ [Telegram] فشل إرسال الإشعار: {e}")

# --- دالة التدريب الرئيسية التي تعمل في خيط منفصل ---
def training_job():
    """الوظيفة الرئيسية لتشغيل عملية التدريب لجميع الرموز."""
    logger.info(f"🚀 بدء عملية تدريب النماذج ({BASE_ML_MODEL_NAME})...")
    initialize_database()
    initialize_binance_client()
    fetch_and_cache_btc_data()
    
    try:
        with open('crypto_list.txt', 'r', encoding='utf-8') as f:
            symbols = {s.strip().upper() + 'USDT' for s in f if s.strip()}
    except FileNotFoundError:
        logger.critical("❌ [Main] ملف 'crypto_list.txt' غير موجود. لا يمكن المتابعة."); return

    send_telegram_notification(f"🚀 *بدء تدريب نماذج {BASE_ML_MODEL_NAME}*\nسيتم تدريب نماذج لـ {len(symbols)} عملة.")
    
    successful, failed = 0, 0
    for symbol in tqdm(symbols, desc="تدريب العملات"):
        logger.info(f"\n--- ⏳ [Main] بدء تدريب النموذج لـ {symbol} ---")
        try:
            hist_data = fetch_historical_data(symbol, SIGNAL_TIMEFRAME, DATA_LOOKBACK_DAYS)
            if hist_data is None or hist_data.empty:
                logger.warning(f"⚠️ [Main] لا توجد بيانات تاريخية لـ {symbol}, جاري التخطي."); failed += 1; continue
            
            prepared_data = prepare_data_for_ml(hist_data, btc_data_cache, symbol)
            if prepared_data is None:
                failed += 1; continue
            X, y = prepared_data
            
            model, scaler, metrics = train_model(X, y)
            
            # حفظ النموذج فقط إذا كانت دقته لفئة الربح جيدة
            if model and metrics and metrics.get('precision_for_profit_class', 0) > 0.45:
                model_bundle = {
                    'model': model, 'scaler': scaler, 'feature_names': list(X.columns)
                }
                model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
                save_model_to_db(model_bundle, model_name, metrics)
                successful += 1
                send_telegram_notification(f"✅ *تم تدريب نموذج {symbol}*\n_Precision (Profit): {metrics['precision_for_profit_class']:.3f}_")
            else:
                logger.warning(f"⚠️ [Main] نموذج {symbol} لم يحقق الأداء المطلوب (Precision < 0.45). تم تجاهله."); failed += 1
        except Exception as e:
            logger.critical(f"❌ [Main] خطأ فادح أثناء تدريب {symbol}: {e}", exc_info=True); failed += 1
        time.sleep(2) # تأخير بسيط بين العملات لتجنب إغراق الـ API

    summary_msg = (f"🏁 *اكتملت عملية تدريب {BASE_ML_MODEL_NAME}*\n"
                   f"- النماذج الناجحة: {successful}\n"
                   f"- النماذج الفاشلة/المتجاهَلة: {failed}")
    send_telegram_notification(summary_msg)
    logger.info(summary_msg)

    if conn: conn.close(); logger.info("👋 [Main] تم إغلاق الاتصال بقاعدة البيانات.")

# --- خادم Flask للبقاء نشطًا على Render ---
app = Flask(__name__)
@app.route('/')
def health_check():
    return "خدمة تدريب النماذج تعمل.", 200

if __name__ == "__main__":
    # بدء عملية التدريب في خيط منفصل حتى لا يتعارض مع خادم الويب
    train_thread = Thread(target=training_job)
    train_thread.daemon = True
    train_thread.start()
    
    # تشغيل خادم الويب
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🌍 تشغيل خادم الويب على المنفذ {port} لإبقاء الخدمة نشطة...")
    app.run(host='0.0.0.0', port=port)
