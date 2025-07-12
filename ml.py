import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
import optuna
import warnings
import gc
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from binance.client import Client
from datetime import datetime, timedelta, timezone
from decouple import config
from typing import List, Dict, Optional, Any, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
        logging.FileHandler('ml_model_trainer_smc_v3_sr_ob.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MLTrainer_SMC_V3')

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
# تم تحديث اسم النموذج ليعكس استخدام معالم الدعم/المقاومة، المعنويات، ودفتر الأوامر
BASE_ML_MODEL_NAME: str = 'SMC_Scalping_V3_With_SR_Sentiment_OB'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 90
HYPERPARAM_TUNING_TRIALS: int = 5
BTC_SYMBOL = 'BTCUSDT'

# --- Indicator & Feature Parameters ---
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
EMA_SLOW_PERIOD: int = 200
EMA_FAST_PERIOD: int = 50
BTC_CORR_PERIOD: int = 30
REL_VOL_PERIOD: int = 30
MOMENTUM_PERIOD: int = 12
EMA_SLOPE_PERIOD: int = 5

# Triple-Barrier Method Parameters
TP_ATR_MULTIPLIER: float = 2.0
SL_ATR_MULTIPLIER: float = 1.5
MAX_HOLD_PERIOD: int = 24

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
btc_data_cache: Optional[pd.DataFrame] = None
fg_data_cache: Optional[pd.DataFrame] = None # [جديد] ذاكرة تخزين مؤقت لمؤشر الخوف والطمع

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

def get_trained_symbols_from_db() -> set:
    if not conn: return set()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT model_name FROM ml_models WHERE model_name LIKE %s;", (f"{BASE_ML_MODEL_NAME}_%",))
            trained_models = cur.fetchall()
            prefix_to_remove = f"{BASE_ML_MODEL_NAME}_"
            trained_symbols = {row['model_name'].replace(prefix_to_remove, '') for row in trained_models if row['model_name'].startswith(prefix_to_remove)}
            logger.info(f"✅ [DB Check] تم العثور على {len(trained_symbols)} نموذج مدرب مسبقاً بالاسم الجديد.")
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
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
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

def fetch_fear_and_greed_data(days: int) -> Optional[pd.DataFrame]:
    """[جديد] جلب بيانات مؤشر الخوف والطمع من واجهة برمجة تطبيقات خارجية."""
    global fg_data_cache
    logger.info(f"ℹ️ [F&G] جاري جلب بيانات مؤشر الخوف والطمع لآخر {days} يوم.")
    try:
        # رابط API لجلب بيانات المؤشر
        url = f"https://api.alternative.me/fng/?limit={days}&format=json"
        response = requests.get(url, timeout=10)
        response.raise_for_status() # التأكد من نجاح الطلب
        data = response.json()['data']
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.normalize()
        df = df[['timestamp', 'value']]
        df.rename(columns={'value': 'fear_greed_value'}, inplace=True)
        df['fear_greed_value'] = pd.to_numeric(df['fear_greed_value'])
        df.set_index('timestamp', inplace=True)
        
        fg_data_cache = df
        logger.info(f"✅ [F&G] تم جلب وتخزين {len(df)} سجل لمؤشر الخوف والطمع.")
        return fg_data_cache
    except Exception as e:
        logger.error(f"❌ [F&G] فشل جلب بيانات مؤشر الخوف والطمع: {e}")
        return None

def calculate_order_book_features(symbol: str) -> Dict[str, float]:
    """[جديد] حساب معالم من دفتر الأوامر الحالي للعملة."""
    default_features = {
        'bid_ask_spread': 0.0, 
        'order_book_imbalance': 0.0,
        'liquidity_density': 0.0
    }
    try:
        # جلب أفضل 20 مستوى من دفتر الأوامر
        order_book = client.get_order_book(symbol=symbol, limit=20)
        
        if not order_book or not order_book['bids'] or not order_book['asks']:
            return default_features

        # تحويل البيانات إلى DataFrame لسهولة الحساب
        bids = pd.DataFrame(order_book['bids'], columns=['price', 'qty'], dtype=float)
        asks = pd.DataFrame(order_book['asks'], columns=['price', 'qty'], dtype=float)

        # حساب المعالم
        best_bid = bids['price'].iloc[0]
        best_ask = asks['price'].iloc[0]

        # 1. الفارق بين العرض والطلب (Bid-Ask Spread)
        spread = (best_ask - best_bid) / best_ask if best_ask > 0 else 0.0

        # 2. عدم توازن دفتر الأوامر (Order Book Imbalance)
        total_bid_volume = (bids['price'] * bids['qty']).sum()
        total_ask_volume = (asks['price'] * asks['qty']).sum()
        total_volume = total_bid_volume + total_ask_volume
        imbalance = (total_bid_volume - total_ask_volume) / total_volume if total_volume > 0 else 0.0

        # 3. كثافة السيولة (Liquidity Density)
        avg_bid_qty = bids['qty'].mean()
        avg_ask_qty = asks['qty'].mean()
        density = (avg_bid_qty + avg_ask_qty) / 2

        return {
            'bid_ask_spread': spread,
            'order_book_imbalance': imbalance,
            'liquidity_density': density
        }

    except Exception as e:
        logger.warning(f"⚠️ [Order Book] لم نتمكن من حساب معالم دفتر الأوامر لـ {symbol}: {e}")
        return default_features

def fetch_sr_levels(symbol: str) -> Optional[pd.DataFrame]:
    if not conn: return None
    query = sql.SQL("SELECT level_price, level_type, score FROM support_resistance_levels WHERE symbol = %s AND score > 20;")
    try:
        with conn.cursor() as cur:
            cur.execute(query, (symbol,))
            levels = cur.fetchall()
        if not levels: return None
        df_levels = pd.DataFrame(levels)
        df_levels['level_price'] = df_levels['level_price'].astype(float)
        df_levels['score'] = df_levels['score'].astype(float)
        return df_levels
    except Exception as e:
        logger.error(f"❌ [S/R Fetch] خطأ أثناء جلب مستويات الدعم/المقاومة لـ {symbol}: {e}")
        if conn: conn.rollback()
        return None

def add_sr_features(df: pd.DataFrame, sr_levels: Optional[pd.DataFrame]) -> pd.DataFrame:
    if sr_levels is None or sr_levels.empty:
        df['dist_to_support'] = 1.0; df['score_of_support'] = 0
        df['dist_to_resistance'] = 1.0; df['score_of_resistance'] = 0
        return df
    support_prices = np.sort(sr_levels[sr_levels['level_type'].str.contains('support|poc|confluence', case=False, na=False)]['level_price'].unique())
    resistance_prices = np.sort(sr_levels[sr_levels['level_type'].str.contains('resistance|poc|confluence', case=False, na=False)]['level_price'].unique())
    price_to_score = sr_levels.set_index('level_price')['score'].to_dict()
    dist_support_list, score_support_list, dist_resistance_list, score_resistance_list = [], [], [], []
    for price in df['close']:
        if len(support_prices) > 0:
            idx = np.searchsorted(support_prices, price)
            if idx > 0:
                nearest_support_price = support_prices[idx - 1]
                dist_support_list.append((price - nearest_support_price) / price)
                score_support_list.append(price_to_score.get(nearest_support_price, 0))
            else: dist_support_list.append(1.0); score_support_list.append(0)
        else: dist_support_list.append(1.0); score_support_list.append(0)
        if len(resistance_prices) > 0:
            idx = np.searchsorted(resistance_prices, price)
            if idx < len(resistance_prices):
                nearest_resistance_price = resistance_prices[idx]
                dist_resistance_list.append((nearest_resistance_price - price) / price)
                score_resistance_list.append(price_to_score.get(nearest_resistance_price, 0))
            else: dist_resistance_list.append(1.0); score_resistance_list.append(0)
        else: dist_resistance_list.append(1.0); score_resistance_list.append(0)
    df['dist_to_support'] = dist_support_list; df['score_of_support'] = score_support_list
    df['dist_to_resistance'] = dist_resistance_list; df['score_of_resistance'] = score_resistance_list
    return df

def calculate_technical_indicators(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    up_move = df_calc['high'].diff(); down_move = -df_calc['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)
    df_calc['price_vs_ema50'] = (df_calc['close'] / df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()) - 1
    merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
    df_calc['btc_correlation'] = df_calc['close'].pct_change().rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    df_calc[f'roc_{MOMENTUM_PERIOD}'] = (df_calc['close'] / df_calc['close'].shift(MOMENTUM_PERIOD) - 1) * 100
    df_calc['roc_acceleration'] = df_calc[f'roc_{MOMENTUM_PERIOD}'].diff()
    ema_slope = df_calc['close'].ewm(span=EMA_SLOPE_PERIOD, adjust=False).mean()
    df_calc[f'ema_slope_{EMA_SLOPE_PERIOD}'] = (ema_slope - ema_slope.shift(1)) / ema_slope.shift(1).replace(0, 1e-9) * 100
    df_calc['hour_of_day'] = df_calc.index.hour
    return df_calc.astype('float32', errors='ignore')

def get_triple_barrier_labels(prices: pd.Series, atr: pd.Series) -> pd.Series:
    labels = pd.Series(0, index=prices.index)
    for i in tqdm(range(len(prices) - MAX_HOLD_PERIOD), desc="Labeling", leave=False):
        entry_price = prices.iloc[i]; current_atr = atr.iloc[i]
        if pd.isna(current_atr) or current_atr == 0: continue
        upper_barrier = entry_price + (current_atr * TP_ATR_MULTIPLIER)
        lower_barrier = entry_price - (current_atr * SL_ATR_MULTIPLIER)
        for j in range(1, MAX_HOLD_PERIOD + 1):
            if i + j >= len(prices): break
            if prices.iloc[i + j] >= upper_barrier: labels.iloc[i] = 1; break
            if prices.iloc[i + j] <= lower_barrier: labels.iloc[i] = -1; break
    return labels

def prepare_data_for_ml(df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    logger.info(f"ℹ️ [ML Prep] Preparing data for {symbol}...")
    
    # الخطوة 1: حساب المؤشرات الفنية
    df_featured = calculate_technical_indicators(df_15m, btc_df)
    
    # الخطوة 2: جلب وإضافة معالم الدعم والمقاومة
    sr_levels = fetch_sr_levels(symbol)
    df_featured = add_sr_features(df_featured, sr_levels)

    # [جديد] الخطوة 3: إضافة معلم الخوف والطمع
    if fg_data_cache is not None:
        df_featured = pd.merge(df_featured, fg_data_cache, left_on=df_featured.index.date, right_on=fg_data_cache.index.date, how='left')
        df_featured.index = df_15m.index # استعادة الفهرس الأصلي
        df_featured['fear_greed_value'].fillna(method='ffill', inplace=True)
        df_featured['fear_greed_value'].fillna(method='bfill', inplace=True)
        df_featured['fear_greed_value'].fillna(50, inplace=True) # تعبئة القيمة المحايدة إذا لم يوجد شيء
    else:
        df_featured['fear_greed_value'] = 50

    # [جديد] الخطوة 4: حساب وإضافة معالم دفتر الأوامر
    ob_features = calculate_order_book_features(symbol)
    for key, value in ob_features.items():
        df_featured[key] = value # تطبيق نفس القيمة على كل الصفوف

    # الخطوة 5: إضافة معالم الفريم الأعلى (MTF)
    delta_4h = df_4h['close'].diff()
    gain_4h = delta_4h.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss_4h = -delta_4h.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_4h['rsi_4h'] = 100 - (100 / (1 + (gain_4h / loss_4h.replace(0, 1e-9))))
    ema_fast_4h = df_4h['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    df_4h['price_vs_ema50_4h'] = (df_4h['close'] / ema_fast_4h) - 1
    mtf_features = df_4h[['rsi_4h', 'price_vs_ema50_4h']]
    df_featured = df_featured.join(mtf_features)
    df_featured[['rsi_4h', 'price_vs_ema50_4h']] = df_featured[['rsi_4h', 'price_vs_ema50_4h']].fillna(method='ffill')
    
    # الخطوة 6: حساب الهدف وتحديد قائمة المعالم النهائية
    df_featured['target'] = get_triple_barrier_labels(df_featured['close'], df_featured['atr'])
    
    feature_columns = [
        'rsi', 'adx', 'atr', 'relative_volume', 'hour_of_day',
        'price_vs_ema50', 'price_vs_ema200', 'btc_correlation',
        'rsi_4h', 'price_vs_ema50_4h',
        f'roc_{MOMENTUM_PERIOD}', 'roc_acceleration', f'ema_slope_{EMA_SLOPE_PERIOD}',
        'dist_to_support', 'score_of_support', 'dist_to_resistance', 'score_of_resistance',
        # --- المعالم الجديدة المضافة ---
        'fear_greed_value', 'bid_ask_spread', 'order_book_imbalance', 'liquidity_density'
    ]
    
    # الخطوة 7: تنظيف البيانات وإرجاعها
    df_cleaned = df_featured.dropna(subset=feature_columns + ['target']).copy()
    df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cleaned.dropna(subset=feature_columns, inplace=True)
    if df_cleaned.empty or df_cleaned['target'].nunique() < 2:
        logger.warning(f"⚠️ [ML Prep] البيانات لـ {symbol} تحتوي على أقل من فئتين بعد التنظيف. سيتم التخطي.")
        return None
    logger.info(f"📊 [ML Prep] توزيع الأهداف لـ {symbol}:\n{df_cleaned['target'].value_counts(normalize=True)}")
    X = df_cleaned[feature_columns]
    y = df_cleaned['target']
    return X, y, feature_columns

def tune_and_train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    logger.info("⏳ [ML Train] بدء تحسين الهايبرباراميترز لنموذج SMC (SVC)...")
    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
            'gamma': trial.suggest_float('gamma', 1e-4, 1e-1, log=True),
            'kernel': 'rbf', 'class_weight': 'balanced', 'probability': True, 'random_state': 42
        }
        all_preds, all_true = [], []
        tscv = TimeSeriesSplit(n_splits=4)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
            model = SVC(**params); model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            all_preds.extend(y_pred); all_true.extend(y_test)
        report = classification_report(all_true, all_preds, output_dict=True, zero_division=0)
        return report.get('1', {}).get('precision', 0)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=HYPERPARAM_TUNING_TRIALS, show_progress_bar=True)
    best_params = study.best_params
    logger.info(f"🏆 [ML Train] أفضل هايبرباراميترز تم العثور عليها: {best_params}")
    logger.info("ℹ️ [ML Train] إعادة تدريب النموذج باستخدام أفضل الباراميترز على كل البيانات...")
    final_model_params = {'class_weight': 'balanced', 'probability': True, 'random_state': 42, 'kernel': 'rbf', **best_params}
    all_preds_final, all_true_final = [], []
    tscv_final = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv_final.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
        model = SVC(**final_model_params); model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        all_preds_final.extend(y_pred); all_true_final.extend(y_test)
    final_report = classification_report(all_true_final, all_preds_final, output_dict=True, zero_division=0)
    final_metrics = {
        'accuracy': accuracy_score(all_true_final, all_preds_final),
        'precision_class_1': final_report.get('1', {}).get('precision', 0),
        'recall_class_1': final_report.get('1', {}).get('recall', 0),
        'f1_score_class_1': final_report.get('1', {}).get('f1-score', 0),
        'num_samples_trained': len(X), 'best_hyperparameters': json.dumps(best_params)
    }
    final_scaler = StandardScaler(); X_scaled_full = final_scaler.fit_transform(X)
    final_model = SVC(**final_model_params); final_model.fit(X_scaled_full, y)
    logger.info(f"📊 [ML Train] الأداء النهائي: Acc: {final_metrics['accuracy']:.4f}, P(1): {final_metrics['precision_class_1']:.4f}")
    return final_model, final_scaler, final_metrics

def save_ml_model_to_db(model_bundle: Dict[str, Any], model_name: str, metrics: Dict[str, Any]):
    logger.info(f"ℹ️ [DB Save] جاري حفظ حزمة النموذج '{model_name}'...")
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
        logger.info(f"✅ [DB Save] تم حفظ حزمة النموذج '{model_name}' بنجاح.")
    except Exception as e:
        logger.error(f"❌ [DB Save] خطأ أثناء حفظ حزمة النموذج: {e}"); conn.rollback()

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try: requests.post(url, json={'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

def run_training_job():
    logger.info(f"🚀 بدء مهمة تدريب النماذج ({BASE_ML_MODEL_NAME})...")
    init_db(); get_binance_client(); fetch_and_cache_btc_data()
    fetch_fear_and_greed_data(days=DATA_LOOKBACK_DAYS_FOR_TRAINING + 5) # جلب بيانات المؤشر مرة واحدة
    
    all_valid_symbols = get_validated_symbols(filename='crypto_list.txt')
    if not all_valid_symbols: logger.critical("❌ [Main] لم يتم العثور على رموز صالحة. سيتم الخروج."); return
    
    trained_symbols = get_trained_symbols_from_db()
    symbols_to_train = [s for s in all_valid_symbols if s not in trained_symbols]
    
    if not symbols_to_train:
        logger.info("✅ [Main] جميع الرموز مدربة بالفعل ومحدثة.");
        if conn: conn.close(); return

    logger.info(f"ℹ️ [Main] الإجمالي: {len(all_valid_symbols)}. مدرب: {len(trained_symbols)}. للتدريب: {len(symbols_to_train)}.")
    send_telegram_message(f"🚀 *{BASE_ML_MODEL_NAME} Training Started*\nWill train models for {len(symbols_to_train)} new symbols.")
    
    successful_models, failed_models = 0, 0
    for symbol in symbols_to_train:
        logger.info(f"\n--- ⏳ [Main] بدء تدريب النموذج لـ {symbol} ---")
        try:
            df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
            df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
            
            if df_15m is None or df_15m.empty or df_4h is None or df_4h.empty:
                logger.warning(f"⚠️ [Main] لا توجد بيانات كافية لـ {symbol}, سيتم التجاوز."); failed_models += 1; continue
            
            prepared_data = prepare_data_for_ml(df_15m, df_4h, btc_data_cache, symbol)
            del df_15m, df_4h; gc.collect()

            if prepared_data is None: failed_models += 1; continue
            X, y, feature_names = prepared_data
            
            training_result = tune_and_train_model(X, y)
            if not all(training_result):
                 logger.warning(f"⚠️ [Main] فشل تدريب النموذج لـ {symbol}."); failed_models += 1; del X, y, prepared_data; gc.collect(); continue
            final_model, final_scaler, model_metrics = training_result
            
            if final_model and final_scaler and model_metrics.get('precision_class_1', 0) > 0.35:
                model_bundle = {'model': final_model, 'scaler': final_scaler, 'feature_names': feature_names}
                model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
                save_ml_model_to_db(model_bundle, model_name, model_metrics)
                successful_models += 1
            else:
                logger.warning(f"⚠️ [Main] النموذج الخاص بـ {symbol} غير مفيد (Precision < 0.35). سيتم تجاهله."); failed_models += 1
            
            del X, y, prepared_data, training_result, final_model, final_scaler, model_metrics; gc.collect()
        except Exception as e:
            logger.critical(f"❌ [Main] حدث خطأ فادح للرمز {symbol}: {e}", exc_info=True); failed_models += 1; gc.collect()
        keep_db_alive(); time.sleep(1)

    completion_message = (f"✅ *{BASE_ML_MODEL_NAME} Training Finished*\n"
                        f"- Successfully trained: {successful_models} new models\n"
                        f"- Failed/Discarded: {failed_models} models\n"
                        f"- Processed this run: {len(symbols_to_train)}")
    send_telegram_message(completion_message); logger.info(completion_message)
    if conn: conn.close()
    logger.info("👋 [Main] انتهت مهمة تدريب النماذج.")

app = Flask(__name__)
@app.route('/')
def health_check():
    return "SMC ML Trainer (v3 - SR/Sentiment/OB) service is running.", 200

if __name__ == "__main__":
    training_thread = Thread(target=run_training_job); training_thread.daemon = True; training_thread.start()
    port = int(os.environ.get("PORT", 10001))
    logger.info(f"🌍 Starting web server on port {port} to keep the service alive...")
    app.run(host='0.0.0.0', port=port)
