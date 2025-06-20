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
import pandas_ta as ta # لتوليد المؤشرات الفنية
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
     exit(1)

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
# --- V7 Model Constants ---
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V7_EnhancedFeatures' # تحديث اسم النموذج
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 200 # زيادة فترة التدريب

# --- Indicator & Feature Parameters ---
# فترات المؤشرات الفنية المستخدمة
RSI_PERIODS: List[int] = [14, 21]
MACD_PARAMS: Dict[str, int] = {"fast": 12, "slow": 26, "signal": 9}
ATR_PERIOD: int = 14
BOLLINGER_PERIOD: int = 20
ADX_PERIOD: int = 14
MOM_PERIOD: int = 10 # فترة الزخم
EMA_FAST_PERIODS: List[int] = [12, 50] # فترات EMA سريعة
EMA_SLOW_PERIODS: List[int] = [26, 200] # فترات EMA بطيئة
BTC_CORR_PERIOD: int = 30 # فترة الارتباط بالبيتكوين
BTC_SYMBOL = 'BTCUSDT'

# --- Triple-Barrier Method Parameters (V7) ---
TP_ATR_MULTIPLIER: float = 1.8 # تعديل مضاعف الربح
SL_ATR_MULTIPLIER: float = 1.2 # تعديل مضاعف الخسارة
MAX_HOLD_PERIOD: int = 24 # أقصى فترة احتفاظ بالشمعات (24 شمعة = 6 ساعات لـ 15 دقيقة)

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
    btc_data_cache = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, DATA_LOOKBACK_DAYS_FOR_TRAINING)
    if btc_data_cache is None:
        logger.critical("❌ [BTC Data] فشل جلب بيانات البيتكوين."); exit(1)
    # حساب العوائد اللوغاريتمية للبيتكوين
    btc_data_cache['btc_log_returns'] = np.log(btc_data_cache['close'] / btc_data_cache['close'].shift(1))

def calculate_features_v7(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """
    يضيف ميزات هندسية جديدة إلى DataFrame للبيانات التاريخية.
    """
    df_calc = df.copy().astype('float64')

    # 1. الميزات الأساسية للسعر والحجم (Core Price & Volume Features)
    # تم الاحتفاظ بـ 'open', 'high', 'low', 'close', 'volume' كأعمدة أساسية
    
    # 2. ميزات التغيرات والنسب (Change & Ratio Features)
    df_calc['log_returns'] = np.log(df_calc['close'] / df_calc['close'].shift(1))
    df_calc['candle_range'] = (df_calc['high'] - df_calc['low']) / df_calc['close']
    df_calc['upper_shadow_ratio'] = (df_calc['high'] - np.maximum(df_calc['open'], df_calc['close'])) / (df_calc['high'] - df_calc['low'])
    df_calc['lower_shadow_ratio'] = (np.minimum(df_calc['open'], df_calc['close']) - df_calc['low']) / (df_calc['high'] - df_calc['low'])
    # استبدال NaN التي قد تنتج عن قسمة على صفر بـ 0 (خاصة لـ shadow ratios عندما High == Low)
    df_calc[['upper_shadow_ratio', 'lower_shadow_ratio']] = df_calc[['upper_shadow_ratio', 'lower_shadow_ratio']].fillna(0)
    df_calc['volume_change'] = df_calc['volume'].pct_change().fillna(0) #fillna(0) للقيم الأولى/القيم المفقودة

    # 3. المؤشرات الفنية (Technical Indicators) باستخدام pandas_ta
    # مؤشرات الاتجاه
    for period in EMA_FAST_PERIODS:
        df_calc[f'EMA_{period}'] = ta.ema(close=df_calc['close'], length=period)
    for period in EMA_SLOW_PERIODS:
        df_calc[f'EMA_{period}'] = ta.ema(close=df_calc['close'], length=period)

    # MACD - جعلها أكثر قوة
    macd_cols = [f'MACD_{MACD_PARAMS["fast"]}_{MACD_PARAMS["slow"]}_{MACD_PARAMS["signal"]}', 
                 f'MACDH_{MACD_PARAMS["fast"]}_{MACD_PARAMS["slow"]}_{MACD_PARAMS["signal"]}',
                 f'MACDS_{MACD_PARAMS["fast"]}_{MACD_PARAMS["slow"]}_{MACD_PARAMS["signal"]}']
    macd_data = ta.macd(close=df_calc['close'], fast=MACD_PARAMS["fast"], slow=MACD_PARAMS["slow"], signal=MACD_PARAMS["signal"])
    
    if macd_data is not None and not macd_data.empty:
        for col in macd_cols:
            if col in macd_data.columns:
                df_calc[col] = macd_data[col]
            else:
                df_calc[col] = np.nan # في حال عدم وجود عمود معين
    else:
        logger.warning("MACD data not generated by pandas_ta or empty. Filling MACD features with NaN.")
        for col in macd_cols:
            df_calc[col] = np.nan # تعبئة بقيم فارغة لتجنب الأخطاء لاحقًا
        

    adx_col_name = f'ADX_{ADX_PERIOD}'
    adx_data = ta.adx(high=df_calc['high'], low=df_calc['low'], close=df_calc['close'], length=ADX_PERIOD)
    if isinstance(adx_data, pd.DataFrame) and adx_col_name in adx_data.columns:
        df_calc[adx_col_name] = adx_data[adx_col_name]
    elif isinstance(adx_data, pd.Series):
        df_calc[adx_col_name] = adx_data
    else:
        logger.warning(f"ADX data not generated by pandas_ta or empty. Filling {adx_col_name} with NaN.")
        df_calc[adx_col_name] = np.nan


    # مؤشرات الزخم
    for period in RSI_PERIODS:
        df_calc[f'RSI_{period}'] = ta.rsi(close=df_calc['close'], length=period)
    
    stoch_cols = ['STOCHk_14_3_3', 'STOCHd_14_3_3'] # Column names as returned by pandas_ta
    stoch_data = ta.stoch(high=df_calc['high'], low=df_calc['low'], close=df_calc['close'], k=14, d=3)
    if stoch_data is not None and not stoch_data.empty:
        df_calc['STOCH_K_14_3'] = stoch_data.get('STOCHk_14_3_3', np.nan) # استخدام .get لسلامة العمود
        df_calc['STOCH_D_14_3'] = stoch_data.get('STOCHd_14_3_3', np.nan)
    else:
        logger.warning("Stochastic data not generated by pandas_ta or empty. Filling Stochastic features with NaN.")
        df_calc['STOCH_K_14_3'] = np.nan
        df_calc['STOCH_D_14_3'] = np.nan

    df_calc['MOMENTUM'] = ta.mom(close=df_calc['close'], length=MOM_PERIOD) # هذا المؤشر عادة لا يسبب مشاكل
    df_calc['OBV'] = ta.obv(close=df_calc['close'], volume=df_calc['volume']) # هذا المؤشر عادة لا يسبب مشاكل

    # مؤشرات التقلب
    atr_col_name = f'ATR_{ATR_PERIOD}'
    atr_data = ta.atr(high=df_calc['high'], low=df_calc['low'], close=df_calc['close'], length=ATR_PERIOD)
    if isinstance(atr_data, pd.DataFrame) and atr_col_name in atr_data.columns:
        df_calc[atr_col_name] = atr_data[atr_col_name]
    elif isinstance(atr_data, pd.Series): # If pandas_ta returns a Series
        df_calc[atr_col_name] = atr_data
    else:
        logger.warning(f"ATR data not generated correctly by pandas_ta. Filling {atr_col_name} with NaN.")
        df_calc[atr_col_name] = np.nan

    bbands_cols = [f'BBL_{BOLLINGER_PERIOD}_2.0', f'BBM_{BOLLINGER_PERIOD}_2.0', f'BBU_{BOLLINGER_PERIOD}_2.0', f'BBB_{BOLLINGER_PERIOD}_2.0']
    bbands_data = ta.bbands(close=df_calc['close'], length=BOLLINGER_PERIOD)
    if bbands_data is not None and not bbands_data.empty:
        for col in bbands_cols:
            if col in bbands_data.columns:
                df_calc[col] = bbands_data[col]
            else:
                df_calc[col] = np.nan # في حال عدم وجود عمود معين
    else:
        logger.warning("Bollinger Bands data not generated by pandas_ta or empty. Filling BB features with NaN.")
        for col in bbands_cols:
            df_calc[col] = np.nan

    # 4. الميزات المتأخرة (Lagged Features)
    lag_periods = [1, 2, 3, 5, 10]
    for lag in lag_periods:
        df_calc[f'CLOSE_LAG_{lag}'] = df_calc['close'].shift(lag)
        df_calc[f'VOLUME_LAG_{lag}'] = df_calc['volume'].shift(lag)
        df_calc[f'LOG_RETURNS_LAG_{lag}'] = df_calc['log_returns'].shift(lag)
        
        # التأكد من وجود أعمدة RSI و MACDH قبل عمل lagging
        # إذا لم يتم إنشاء RSI_14 على سبيل المثال، سيتم إرجاع NaN، وهو أمر مقبول
        rsi_col = f'RSI_{RSI_PERIODS[0]}'
        if rsi_col in df_calc.columns:
            df_calc[f'{rsi_col}_LAG_{lag}'] = df_calc[rsi_col].shift(lag)
        else:
            df_calc[f'{rsi_col}_LAG_{lag}'] = np.nan # التأكد من إنشاء العمود حتى لو كان RSI غير موجود

        macdh_col = f'MACDH_{MACD_PARAMS["fast"]}_{MACD_PARAMS["slow"]}_{MACD_PARAMS["signal"]}'
        if macdh_col in df_calc.columns:
            df_calc[f'{macdh_col}_LAG_{lag}'] = df_calc[macdh_col].shift(lag)
        else:
            df_calc[f'{macdh_col}_LAG_{lag}'] = np.nan # التأكد من إنشاء العمود

    # 5. الميزات المستندة إلى الوقت (Time-Based Features)
    df_calc['day_of_week'] = df_calc.index.dayofweek
    df_calc['hour_of_day'] = df_calc.index.hour
    df_calc['day_of_month'] = df_calc.index.day
    df_calc['month'] = df_calc.index.month
    df_calc['is_weekend'] = ((df_calc.index.dayofweek == 5) | (df_calc.index.dayofweek == 6)).astype(int)

    # 6. ميزات التفاعل (Interaction Features)
    # دمج بيانات البيتكوين
    if btc_df is not None and not btc_df.empty:
        # التأكد من أن الفهرس متطابق قبل الدمج
        # استخدام .reindex().fillna(0) بدلاً من merge لتبسيط التعامل مع الفجوات الزمنية
        df_calc['BTC_LOG_RETURNS'] = btc_df['btc_log_returns'].reindex(df_calc.index, method='nearest').fillna(0)
        
        # حساب الارتباط فقط إذا كان هناك تقلب في log_returns أو btc_log_returns
        if df_calc['log_returns'].std() > 0 and df_calc['BTC_LOG_RETURNS'].std() > 0:
            df_calc[f'BTC_CORRELATION_{BTC_CORR_PERIOD}'] = df_calc['log_returns'].rolling(window=BTC_CORR_PERIOD).corr(df_calc['BTC_LOG_RETURNS'])
            df_calc['BTC_CORRELATION_SQUARED'] = df_calc[f'BTC_CORRELATION_{BTC_CORR_PERIOD}'] ** 2 
        else:
            df_calc[f'BTC_CORRELATION_{BTC_CORR_PERIOD}'] = np.nan
            df_calc['BTC_CORRELATION_SQUARED'] = np.nan
    else:
        # إذا لم تكن بيانات البيتكوين متاحة، قم بإنشاء الأعمدة بـ NaN
        df_calc['BTC_LOG_RETURNS'] = np.nan
        df_calc[f'BTC_CORRELATION_{BTC_CORR_PERIOD}'] = np.nan
        df_calc['BTC_CORRELATION_SQUARED'] = np.nan


    # Ensure EMA columns exist before creating ratios
    ema_fast_col_name = f'EMA_{EMA_FAST_PERIODS[0]}'
    ema_slow_col_name = f'EMA_{EMA_SLOW_PERIODS[0]}'
    
    if ema_fast_col_name in df_calc.columns and not df_calc[ema_fast_col_name].isnull().all() and (df_calc[ema_fast_col_name] != 0).any():
        df_calc['PRICE_VS_EMA_FAST_RATIO'] = (df_calc['close'] / df_calc[ema_fast_col_name]) - 1
    else:
        df_calc['PRICE_VS_EMA_FAST_RATIO'] = np.nan

    if ema_slow_col_name in df_calc.columns and not df_calc[ema_slow_col_name].isnull().all() and (df_calc[ema_slow_col_name] != 0).any():
        df_calc['PRICE_VS_EMA_SLOW_RATIO'] = (df_calc['close'] / df_calc[ema_slow_col_name]) - 1
    else:
        df_calc['PRICE_VS_EMA_SLOW_RATIO'] = np.nan

    atr_col_for_interaction = f'ATR_{ATR_PERIOD}'
    if atr_col_for_interaction in df_calc.columns and not df_calc[atr_col_for_interaction].isnull().all():
        df_calc['VOLUME_X_VOLATILITY'] = df_calc['volume'] * df_calc[atr_col_for_interaction]
    else:
        df_calc['VOLUME_X_VOLATILITY'] = np.nan


    # تنظيف أسماء الأعمدة للتوافق مع LightGBM (تحويلها إلى حروف كبيرة)
    df_calc.columns = [col.upper() for col in df_calc.columns]
    
    return df_calc

def get_triple_barrier_labels(prices: pd.Series, atr: pd.Series) -> pd.Series:
    """
    تطبق طريقة الحاجز الثلاثي لإنشاء تسميات (labels) للتدريب.
    """
    labels = pd.Series(0, index=prices.index, dtype='int8')
    # لتجنب تحيز التطلع، يجب أن يكون ATR المستخدم من نفس الشمعة التي يتم دخولها
    # وبالتالي، يجب أن يكون ATR for price[i] هو atr.iloc[i]
    # loop over prices starting from the first price to the last price minus MAX_HOLD_PERIOD
    for i in tqdm(range(len(prices) - MAX_HOLD_PERIOD), desc="Labeling", leave=False, ncols=100):
        entry_price = prices.iloc[i]
        current_atr = atr.iloc[i]
        
        # تخطي إذا كان ATR غير صالح لتجنب الأخطاء
        if pd.isna(current_atr) or current_atr == 0:
            labels.iloc[i] = np.nan # وضع NaN للقيم التي لا يمكن حسابها
            continue
        
        upper_barrier = entry_price + (current_atr * TP_ATR_MULTIPLIER)
        lower_barrier = entry_price - (current_atr * SL_ATR_MULTIPLIER)
        
        triggered = False
        # ابدأ البحث عن الحاجز من الشمعة التالية (i+1)
        for j in range(1, MAX_HOLD_PERIOD + 1):
            if i + j >= len(prices): # التأكد من عدم تجاوز حدود DataFrame
                break
            
            future_price = prices.iloc[i + j]
            
            if future_price >= upper_barrier:
                labels.iloc[i] = 1  # ربح (شراء)
                triggered = True
                break
            if future_price <= lower_barrier:
                labels.iloc[i] = -1 # خسارة (شراء)
                triggered = True
                break
        
        # إذا لم يتم تشغيل أي حاجز خلال MAX_HOLD_PERIOD، يتم تعيينه 0 (لا تغيير كبير)
        if not triggered:
            labels.iloc[i] = 0
            
    return labels


def prepare_data_for_ml(df: pd.DataFrame, btc_df: pd.DataFrame, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
    logger.info(f"ℹ️ [ML Prep V7] Preparing data for {symbol}...")
    df_featured = calculate_features_v7(df, btc_df)
    
    # تحديد عمود ATR الصحيح
    atr_series_name = f'ATR_{ATR_PERIOD}'.upper()
    if atr_series_name not in df_featured.columns or df_featured[atr_series_name].isnull().all():
        logger.error(f"FATAL: ATR column '{atr_series_name}' not found or is all NaN for {symbol}. Cannot generate labels.")
        return None
        
    df_featured['TARGET'] = get_triple_barrier_labels(df_featured['CLOSE'], df_featured[atr_series_name])
    
    # قائمة الميزات الجديدة التي تم هندستها (تأكد من مطابقتها لما يتم إنشاؤه)
    # هذه القائمة يجب أن تحتوي على جميع الأعمدة التي نتوقع وجودها في df_featured
    feature_columns = [
        'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', # الميزات الأساسية
        'LOG_RETURNS', 'CANDLE_RANGE', 'UPPER_SHADOW_RATIO', 'LOWER_SHADOW_RATIO', 'VOLUME_CHANGE', # ميزات التغيرات والنسب
    ]
    
    # إضافة ميزات المؤشرات الفنية
    for period in EMA_FAST_PERIODS:
        feature_columns.append(f'EMA_{period}')
    for period in EMA_SLOW_PERIODS:
        feature_columns.append(f'EMA_{period}')
    feature_columns.extend([
        f'MACD_{MACD_PARAMS["fast"]}_{MACD_PARAMS["slow"]}_{MACD_PARAMS["signal"]}', 
        f'MACDH_{MACD_PARAMS["fast"]}_{MACD_PARAMS["slow"]}_{MACD_PARAMS["signal"]}',
        f'MACDS_{MACD_PARAMS["fast"]}_{MACD_PARAMS["slow"]}_{MACD_PARAMS["signal"]}',
        f'ADX_{ADX_PERIOD}'
    ])
    for period in RSI_PERIODS:
        feature_columns.append(f'RSI_{period}')
    feature_columns.extend([
        'STOCH_K_14_3', 'STOCH_D_14_3', 'MOMENTUM', 'OBV',
        f'ATR_{ATR_PERIOD}', f'BBL_{BOLLINGER_PERIOD}_2.0', f'BBM_{BOLLINGER_PERIOD}_2.0', 
        f'BBU_{BOLLINGER_PERIOD}_2.0', f'BBB_{BOLLINGER_PERIOD}_2.0'
    ])
    
    # إضافة الميزات المتأخرة
    lag_periods = [1, 2, 3, 5, 10]
    for lag in lag_periods:
        feature_columns.append(f'CLOSE_LAG_{lag}')
        feature_columns.append(f'VOLUME_LAG_{lag}')
        feature_columns.append(f'LOG_RETURNS_LAG_{lag}')
        feature_columns.append(f'RSI_{RSI_PERIODS[0]}_LAG_{lag}')
        feature_columns.append(f'MACDH_{MACD_PARAMS["fast"]}_{MACD_PARAMS["slow"]}_{MACD_PARAMS["signal"]}_LAG_{lag}')

    # إضافة الميزات المستندة إلى الوقت
    feature_columns.extend([
        'DAY_OF_WEEK', 'HOUR_OF_DAY', 'DAY_OF_MONTH', 'MONTH', 'IS_WEEKEND'
    ])

    # إضافة ميزات التفاعل
    # يتم إضافة هذه الميزات فقط إذا كانت بيانات البيتكوين قد تم جلبها
    if btc_df is not None and not btc_df.empty:
        feature_columns.extend([
            'BTC_LOG_RETURNS', # تم إضافة هذه الميزة مباشرةً في calculate_features_v7
            f'BTC_CORRELATION_{BTC_CORR_PERIOD}', 'BTC_CORRELATION_SQUARED'
        ])
    
    feature_columns.extend([
        'PRICE_VS_EMA_FAST_RATIO', 'PRICE_VS_EMA_SLOW_RATIO', 'VOLUME_X_VOLATILITY'
    ])
    
    # تنظيف DataFrame من الصفوف التي تحتوي على قيم NaN بعد حساب الميزات (خاصةً Lagged Features والمؤشرات)
    # وأيضًا إسقاط الصفوف التي لم يتم فيها تعيين TARGET (حيث تم تعيينها إلى NaN في get_triple_barrier_labels)
    df_cleaned = df_featured.dropna(subset=feature_columns + ['TARGET']).copy()
    
    # إزالة الصفوف ذات TARGET = 0 إذا كان النموذج مخصصًا فقط لتنبؤ الصعود/الهبوط الواضح
    # إذا كنت تريد تضمين الفئة "لا تغيير"، فقم بإزالة هذا السطر.
    # في هذا السيناريو، نحن نركز على تنبؤات الشراء (1) والخسارة (-1)
    df_cleaned = df_cleaned[df_cleaned['TARGET'] != 0]

    if df_cleaned.empty or df_cleaned['TARGET'].nunique() < 2:
        logger.warning(f"⚠️ [ML Prep] Data for {symbol} has less than 2 classes after filtering or is empty. Skipping.")
        return None
    
    # إعادة تسمية -1 إلى 0 ليتناسب مع مشكلة التصنيف الثنائي (0 = خسارة/هبوط، 1 = ربح/صعود)
    df_cleaned['TARGET'] = df_cleaned['TARGET'].replace(-1, 0)
    
    target_counts = df_cleaned['TARGET'].value_counts(normalize=True)
    logger.info(f"📊 [ML Prep] Target distribution for {symbol} (after filtering):\n{target_counts}")
    # التحقق من وجود عدم توازن كبير في الفئات
    if target_counts.min() < 0.1: # إذا كانت أصغر فئة تمثل أقل من 10%
        logger.warning(f"⚠️ [ML Prep] Severe class imbalance for {symbol}. Min class is {target_counts.min():.2%}. Skipping training.")
        return None

    # التحقق النهائي من أن جميع الأعمدة المحددة كـ features موجودة في DataFrame
    missing_features = [col for col in feature_columns if col not in df_cleaned.columns]
    if missing_features:
        logger.error(f"❌ [ML Prep] Missing features in DataFrame for {symbol} after cleanup: {missing_features}")
        return None
    
    # التأكد من أن الأعمدة موجودة وجميع قيمها ليست NaN
    for col in feature_columns:
        if df_cleaned[col].isnull().all():
            logger.warning(f"⚠️ [ML Prep] Feature '{col}' for {symbol} is all NaN after cleanup. This feature will not be useful.")

    X = df_cleaned[feature_columns]
    y = df_cleaned['TARGET']
    return X, y, feature_columns


def train_with_walk_forward_validation(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    logger.info("ℹ️ [ML Train V7] Starting training with Walk-Forward Validation...")
    tscv = TimeSeriesSplit(n_splits=5) # 5 أقسام للتحقق المتتالي
    
    lgb_params = {
        'objective': 'binary', # هدف التصنيف الثنائي
        'metric': 'logloss',   # مقياس تقييم
        'random_state': 42,    # لضمان قابلية التكرار
        'verbosity': -1,       # إيقاف إخراج التفاصيل أثناء التدريب
        'n_estimators': 1500,  # عدد المقدرين (الأشجار)
        'learning_rate': 0.01, # معدل التعلم
        'num_leaves': 31,      # الحد الأقصى لعدد الأوراق في كل شجرة
        'max_depth': -1,       # لا يوجد حد أقصى للعمق
        'class_weight': 'balanced', # للتعامل مع عدم توازن الفئات
        'reg_alpha': 0.0,      # L1 regularization
        'reg_lambda': 0.0,     # L2 regularization
        'n_jobs': -1,          # استخدام جميع النوى المتاحة
        'colsample_bytree': 0.8, # نسبة الميزات التي يتم أخذ عينات منها لكل شجرة
        'min_child_samples': 10, # الحد الأدنى لعدد العينات المطلوبة لإنشاء ورقة جديدة
        'boosting_type': 'gbdt', # نوع التعزيز
    }
    
    final_model, final_scaler = None, None
    all_y_true, all_y_pred = pd.Series(dtype=int), pd.Series(dtype=int)

    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        if len(y_train) == 0 or len(y_test) == 0:
            logger.warning(f"--- Fold {i+1}: Skipping due to empty train/test set.")
            continue
        
        # Scaling features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
        
        model = lgb.LGBMClassifier(**lgb_params)
        
        model.fit(X_train_scaled, y_train, 
                  eval_set=[(X_test_scaled, y_test)],
                  eval_metric='logloss',
                  callbacks=[lgb.early_stopping(100, verbose=False)]) # توقف مبكر
        
        y_pred = model.predict(X_test_scaled)
        
        # تجميع النتائج لتقرير نهائي شامل
        all_y_true = pd.concat([all_y_true, y_test])
        all_y_pred = pd.concat([all_y_pred, pd.Series(y_pred, index=y_test.index)])

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        logger.info(f"--- Fold {i+1}: Accuracy: {accuracy_score(y_test, y_pred):.4f}, "
                    f"P(Win): {report.get('1', {}).get('precision', 0):.4f}, "
                    f"P(Loss): {report.get('0', {}).get('precision', 0):.4f}")
        
        final_model, final_scaler = model, scaler # الاحتفاظ بالنموذج وال scaler من الطية الأخيرة

    if not final_model or not final_scaler or all_y_true.empty:
        logger.error("❌ [ML Train] Training failed, no valid model or data for final report.")
        return None, None, None

    # تقرير الأداء النهائي على جميع البيانات المختبرة عبر الطيات
    final_report = classification_report(all_y_true, all_y_pred, output_dict=True, zero_division=0)
    avg_metrics = {
        'accuracy': accuracy_score(all_y_true, all_y_pred),
        'precision_win': final_report.get('1', {}).get('precision', 0),
        'recall_win': final_report.get('1', {}).get('recall', 0),
        'f1_score_win': final_report.get('1', {}).get('f1-score', 0),
        'num_samples_trained': len(X),
    }

    metrics_log_str = ', '.join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
    logger.info(f"📊 [ML Train] Final Model Performance on All Test Data (Aggregated): {metrics_log_str}")
    return final_model, final_scaler, avg_metrics

def save_ml_model_to_db(model_bundle: Dict[str, Any], model_name: str, metrics: Dict[str, Any]):
    logger.info(f"ℹ️ [DB Save] Saving model bundle '{model_name}'...")
    try:
        if conn is None or conn.closed:
            logger.warning("[DB Save] DB connection is closed. Re-initializing.")
            init_db()

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
        logger.error(f"❌ [DB Save] Error saving model bundle: {e}"); 
        if conn: conn.rollback()

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try: requests.post(url, json={'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}, timeout=10)
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

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
            
            training_result = train_with_walk_forward_validation(X, y)
            if not all(training_result):
                 logger.warning(f"⚠️ [Main] Training did not produce a valid model for {symbol}. Skipping."); failed_models += 1; continue
            final_model, final_scaler, model_metrics = training_result
            
            # معايير قبول النموذج (يمكن تعديلها حسب الحاجة)
            # نتحقق من أن دقة الفوز (precision_win) ودرجة F1 (f1_score_win) أعلى من عتبة معينة.
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
    training_thread = Thread(target=run_training_job)
    training_thread.daemon = True # سيسمح هذا للخيط بالخروج عند إغلاق التطبيق الرئيسي
    training_thread.start()
    
    port = int(os.environ.get("PORT", 10001))
    logger.info(f"🌍 Starting web server on port {port} to keep the service alive...")
    app.run(host='0.0.0.0', port=port)
