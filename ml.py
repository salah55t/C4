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
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Tuple, Union

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# استيراد مكتبات Flask والخيوط
from flask import Flask, Response
from threading import Thread

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

logger.info(f"Binance API Key: {'Available' if API_KEY else 'Not available'}")
logger.info(f"Database URL: {'Available' if DB_URL else 'Not available'}")
logger.info(f"Telegram Token: {'Available' if TELEGRAM_TOKEN else 'Not available'}")
logger.info(f"Telegram Chat ID: {'Available' if CHAT_ID else 'Not available'}")

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
# NEW: Changed model name to reflect LightGBM
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V2'
TRAINING_TIMEFRAME: str = '15m'
DATA_LOOKBACK_DAYS_FOR_TRAINING: int = 90 # 3 months of data for robust training

# Indicator Parameters (Must match the bot script c4.py)
VOLUME_LOOKBACK_CANDLES: int = 1
RSI_PERIOD: int = 9
RSI_MOMENTUM_LOOKBACK_CANDLES: int = 2
ENTRY_ATR_PERIOD: int = 10
SUPERTREND_PERIOD: int = 10
SUPERTREND_MULTIPLIER: float = 3.0

# Ichimoku Cloud Parameters
TENKAN_PERIOD: int = 9
KIJUN_PERIOD: int = 26
SENKOU_SPAN_B_PERIOD: int = 52
CHIKOU_LAG: int = 26

# Fibonacci & S/R Parameters
FIB_SR_LOOKBACK_WINDOW: int = 50

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
cur: Optional[psycopg2.extensions.cursor] = None
client: Optional[Client] = None

# Training status variables
training_status: str = "Idle"
last_training_time: Optional[datetime] = None
last_training_summary: Dict[str, Any] = {}
training_error: Optional[str] = None

# ---------------------- Binance Client Setup ----------------------
try:
    logger.info("ℹ️ [Binance] تهيئة عميل Binance...")
    client = Client(API_KEY, API_SECRET)
    client.ping()
    server_time = client.get_server_time()
    logger.info(f"✅ [Binance] تم تهيئة عميل Binance. وقت الخادم: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
except (BinanceAPIException, BinanceRequestException) as e:
    logger.critical(f"❌ [Binance] خطأ في واجهة برمجة تطبيقات أو طلب Binance: {e}")
    exit(1)
except Exception as e:
    logger.critical(f"❌ [Binance] فشل غير متوقع في تهيئة عميل Binance: {e}")
    exit(1)

# ---------------------- Database Functions ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    """Initializes database connection and creates tables if they don't exist."""
    global conn, cur
    logger.info("[DB] بدء تهيئة قاعدة البيانات...")
    for attempt in range(retries):
        try:
            logger.info(f"[DB] محاولة الاتصال بقاعدة البيانات (المحاولة {attempt + 1}/{retries})...")
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False # Use transactions
            cur = conn.cursor()
            logger.info("✅ [DB] تم الاتصال بقاعدة البيانات بنجاح.")

            # Create ml_models table
            logger.info("[DB] التحقق من/إنشاء جدول 'ml_models'...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ml_models (
                    id SERIAL PRIMARY KEY,
                    model_name TEXT NOT NULL UNIQUE,
                    model_data BYTEA NOT NULL,
                    trained_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metrics JSONB,
                    feature_importance JSONB
                );""")
            conn.commit()
            logger.info("✅ [DB] جدول 'ml_models' موجود أو تم إنشاؤه.")

            # Create other necessary tables if they don't exist
            logger.info("[DB] التحقق من/إنشاء جدول 'signals'...")
            cur.execute("""
                 CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    initial_target DOUBLE PRECISION NOT NULL,
                    current_target DOUBLE PRECISION NOT NULL,
                    stop_loss DOUBLE PRECISION,
                    r2_score DOUBLE PRECISION,
                    volume_15m DOUBLE PRECISION,
                    achieved_target BOOLEAN DEFAULT FALSE,
                    closing_price DOUBLE PRECISION,
                    closed_at TIMESTAMP WITH TIME ZONE,
                    sent_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    entry_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    time_to_target INTERVAL,
                    profit_percentage DOUBLE PRECISION,
                    strategy_name TEXT,
                    signal_details JSONB
                );""")
            conn.commit()
            logger.info("✅ [DB] جدول 'signals' موجود أو تم إنشاؤه.")
            return

        except OperationalError as op_err:
            logger.error(f"❌ [DB] خطأ في التشغيل أثناء الاتصال (المحاولة {attempt + 1}): {op_err}")
            if conn: conn.rollback()
        except Exception as e:
            logger.critical(f"❌ [DB] فشل غير متوقع في تهيئة قاعدة البيانات (المحاولة {attempt + 1}): {e}", exc_info=True)
            if conn: conn.rollback()

        if attempt < retries - 1:
            logger.info(f"[DB] الانتظار {delay} ثواني قبل إعادة المحاولة...")
            time.sleep(delay)
        else:
            logger.critical("❌ [DB] فشلت جميع محاولات الاتصال بقاعدة البيانات.")
            raise e

def check_db_connection() -> bool:
    """Checks database connection status and re-initializes if necessary."""
    global conn, cur
    try:
        if conn is None or conn.closed != 0:
            logger.warning("⚠️ [DB] تم إغلاق الاتصال أو غير موجود. إعادة التهيئة...")
            init_db()
            return True
        else:
             with conn.cursor() as check_cur:
                  check_cur.execute("SELECT 1;") # Simple query to check connection
                  check_cur.fetchone()
             return True
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [DB] فقد الاتصال بقاعدة البيانات ({e}). إعادة التهيئة...")
        try:
             init_db()
             return True
        except Exception as recon_err:
            logger.error(f"❌ [DB] فشلت محاولة إعادة الاتصال بعد الفقدان: {recon_err}")
            return False
    except Exception as e:
        logger.error(f"❌ [DB] خطأ غير متوقع أثناء فحص الاتصال: {e}", exc_info=True)
        return False

# ---------------------- Data Fetching & Preparation ----------------------
def get_crypto_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """Reads and validates crypto symbols from a file against Binance spot trading pairs."""
    raw_symbols: List[str] = []
    logger.info(f"ℹ️ [Data] قراءة قائمة الرموز من الملف '{filename}'...")
    try:
        # Build path relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)

        if not os.path.exists(file_path):
            logger.error(f"❌ [Data] الملف '{filename}' غير موجود في '{script_dir}'.")
            return []

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = [f"{line.strip().upper()}USDT" if not line.strip().upper().endswith('USDT') else line.strip().upper()
                           for line in f if line.strip() and not line.startswith('#')]
        raw_symbols = sorted(list(set(raw_symbols)))
        logger.info(f"ℹ️ [Data] تم قراءة {len(raw_symbols)} رمزًا مبدئيًا.")

    except Exception as e:
        logger.error(f"❌ [Data] خطأ في قراءة الملف '{filename}': {e}", exc_info=True)
        return []

    if not client:
        logger.error("❌ [Data Validation] عميل Binance غير مهيأ. لا يمكن التحقق من الرموز.")
        return raw_symbols

    try:
        logger.info("ℹ️ [Data Validation] التحقق من الرموز وحالة التداول من Binance API...")
        exchange_info = client.get_exchange_info()
        valid_trading_usdt_symbols = {
            s['symbol'] for s in exchange_info['symbols']
            if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING' and s.get('isSpotTradingAllowed')
        }
        validated_symbols = [symbol for symbol in raw_symbols if symbol in valid_trading_usdt_symbols]
        removed_count = len(raw_symbols) - len(validated_symbols)
        if removed_count > 0:
            removed_symbols = set(raw_symbols) - set(validated_symbols)
            logger.warning(f"⚠️ [Data Validation] تم إزالة {removed_count} رمز غير صالح: {', '.join(removed_symbols)}")

        logger.info(f"✅ [Data Validation] تم التحقق من الرموز. استخدام {len(validated_symbols)} رمزًا صالحًا.")
        return validated_symbols

    except (BinanceAPIException, BinanceRequestException) as e:
         logger.error(f"❌ [Data Validation] خطأ في Binance API أو الشبكة أثناء التحقق من الرموز: {e}")
         return raw_symbols # Fallback to unvalidated list

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """Fetches and processes historical k-line data from Binance."""
    if not client: return None
    try:
        start_str = (datetime.utcnow() - timedelta(days=days)).strftime("%d %b %Y %H:%M:%S")
        logger.debug(f"ℹ️ [Data] جلب بيانات {interval} لـ {symbol} من {start_str}...")

        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines:
            logger.warning(f"⚠️ [Data] لا توجد بيانات تاريخية ({interval}) لـ {symbol}.")
            return None

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[numeric_cols]
        df.dropna(inplace=True)
        df.sort_index(inplace=True)

        logger.debug(f"✅ [Data] تم جلب ومعالجة {len(df)} شمعة تاريخية ({interval}) لـ {symbol}.")
        return df

    except (BinanceAPIException, BinanceRequestException) as e:
         logger.error(f"❌ [Data] خطأ في Binance API أو الشبكة أثناء جلب البيانات لـ {symbol}: {e}")
         return None
    except Exception as e:
        logger.error(f"❌ [Data] خطأ غير متوقع أثناء جلب البيانات التاريخية لـ {symbol}: {e}", exc_info=True)
        return None

# ---------------------- Technical Indicator Functions (Identical to Bot) ----------------------
def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calculate_atr_indicator(df: pd.DataFrame, period: int) -> pd.DataFrame:
    df_copy = df.copy()
    high_low = df_copy['high'] - df_copy['low']
    high_close_prev = (df_copy['high'] - df_copy['close'].shift(1)).abs()
    low_close_prev = (df_copy['low'] - df_copy['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)
    df_copy['atr'] = tr.ewm(span=period, adjust=False).mean()
    return df_copy

def calculate_supertrend(df: pd.DataFrame, period: int, multiplier: float) -> pd.DataFrame:
    df_st = df.copy()
    if 'atr' not in df_st.columns:
        df_st = calculate_atr_indicator(df_st, period)

    df_st['basic_upper_band'] = ((df_st['high'] + df_st['low']) / 2) + (multiplier * df_st['atr'])
    df_st['basic_lower_band'] = ((df_st['high'] + df_st['low']) / 2) - (multiplier * df_st['atr'])
    df_st['final_upper_band'] = 0.0
    df_st['final_lower_band'] = 0.0
    df_st['supertrend_direction'] = 0

    for i in range(1, len(df_st)):
        if df_st['basic_upper_band'].iloc[i] < df_st['final_upper_band'].iloc[i-1] or df_st['close'].iloc[i-1] > df_st['final_upper_band'].iloc[i-1]:
            df_st.loc[df_st.index[i], 'final_upper_band'] = df_st['basic_upper_band'].iloc[i]
        else:
            df_st.loc[df_st.index[i], 'final_upper_band'] = df_st['final_upper_band'].iloc[i-1]

        if df_st['basic_lower_band'].iloc[i] > df_st['final_lower_band'].iloc[i-1] or df_st['close'].iloc[i-1] < df_st['final_lower_band'].iloc[i-1]:
            df_st.loc[df_st.index[i], 'final_lower_band'] = df_st['basic_lower_band'].iloc[i]
        else:
            df_st.loc[df_st.index[i], 'final_lower_band'] = df_st['final_lower_band'].iloc[i-1]

        if df_st['supertrend_direction'].iloc[i-1] in [0, 1] and df_st['close'].iloc[i] <= df_st['final_lower_band'].iloc[i-1]:
             df_st.loc[df_st.index[i], 'supertrend_direction'] = -1
        elif df_st['supertrend_direction'].iloc[i-1] in [0, -1] and df_st['close'].iloc[i] >= df_st['final_upper_band'].iloc[i-1]:
             df_st.loc[df_st.index[i], 'supertrend_direction'] = 1
        else:
            df_st.loc[df_st.index[i], 'supertrend_direction'] = df_st['supertrend_direction'].iloc[i-1]

    df_st.drop(['basic_upper_band', 'basic_lower_band', 'final_upper_band', 'final_lower_band'], axis=1, inplace=True)
    return df_st

def _calculate_btc_trend_feature(df_btc: pd.DataFrame) -> Optional[pd.Series]:
    min_data_for_ema = 55
    if df_btc is None or len(df_btc) < min_data_for_ema:
        logger.warning(f"⚠️ بيانات BTC/USDT غير كافية ({len(df_btc) if df_btc is not None else 0}) لحساب اتجاه البيتكوين.")
        return pd.Series(index=df_btc.index if df_btc is not None else None, data=0.0)

    df_btc_copy = df_btc.copy()
    ema20 = calculate_ema(df_btc_copy['close'], 20)
    ema50 = calculate_ema(df_btc_copy['close'], 50)
    ema_df = pd.DataFrame({'ema20': ema20, 'ema50': ema50, 'close': df_btc_copy['close']}).dropna()

    trend_series = pd.Series(index=ema_df.index, data=0.0)
    trend_series[(ema_df['close'] > ema_df['ema20']) & (ema_df['ema20'] > ema_df['ema50'])] = 1.0  # Bullish
    trend_series[(ema_df['close'] < ema_df['ema20']) & (ema_df['ema20'] < ema_df['ema50'])] = -1.0 # Bearish
    return trend_series.reindex(df_btc.index).fillna(0.0)

# Other indicator functions (RSI, Ichimoku, Fib, S/R) are assumed here for brevity,
# but they MUST be identical to the ones in c4.py. Let's include the full prep function.

# ---------------------- ML Data Preparation ----------------------
def prepare_data_for_ml(df: pd.DataFrame, symbol: str, target_period: int = 5, price_change_threshold: float = 0.01) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """Prepares the complete feature set and target variable for ML training."""
    logger.info(f"ℹ️ [ML Prep] إعداد البيانات لنموذج ML لـ {symbol}...")

    min_len_required = max(SENKOU_SPAN_B_PERIOD, FIB_SR_LOOKBACK_WINDOW) + target_period + 5
    if len(df) < min_len_required:
        logger.warning(f"⚠️ [ML Prep] DataFrame لـ {symbol} قصير جدًا ({len(df)} < {min_len_required}).")
        return None

    df_calc = df.copy()

    # --- Feature Engineering (must match bot's feature calculation) ---
    # Volume
    df_calc['volume_15m_avg'] = df_calc['volume'].rolling(window=VOLUME_LOOKBACK_CANDLES, min_periods=1).mean()

    # RSI & RSI Momentum
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df_calc['rsi'] = (100 - (100 / (1 + rs))).ffill().fillna(50)
    df_calc['rsi_momentum_bullish'] = ((df_calc['rsi'] > df_calc['rsi'].shift(1)) & (df_calc['rsi'].shift(1) > df_calc['rsi'].shift(2)) & (df_calc['rsi'] > 50)).astype(int)

    # ATR, Supertrend
    df_calc = calculate_atr_indicator(df_calc, ENTRY_ATR_PERIOD)
    df_calc = calculate_supertrend(df_calc, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)

    # BTC Trend
    btc_df = fetch_historical_data("BTCUSDT", interval=TRAINING_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING)
    if btc_df is not None:
        btc_trend_series = _calculate_btc_trend_feature(btc_df)
        if btc_trend_series is not None:
            df_calc = df_calc.join(btc_trend_series.rename('btc_trend_feature'), how='left')
            df_calc['btc_trend_feature'].fillna(0.0, inplace=True)
    if 'btc_trend_feature' not in df_calc.columns:
        df_calc['btc_trend_feature'] = 0.0

    # Ichimoku Cloud
    # ... (Full function copied from original script)
    high_9 = df_calc['high'].rolling(window=TENKAN_PERIOD).max()
    low_9 = df_calc['low'].rolling(window=TENKAN_PERIOD).min()
    df_calc['tenkan_sen'] = (high_9 + low_9) / 2
    high_26 = df_calc['high'].rolling(window=KIJUN_PERIOD).max()
    low_26 = df_calc['low'].rolling(window=KIJUN_PERIOD).min()
    df_calc['kijun_sen'] = (high_26 + low_26) / 2
    df_calc['senkou_span_a'] = ((df_calc['tenkan_sen'] + df_calc['kijun_sen']) / 2).shift(KIJUN_PERIOD)
    high_52 = df_calc['high'].rolling(window=SENKOU_SPAN_B_PERIOD).max()
    low_52 = df_calc['low'].rolling(window=SENKOU_SPAN_B_PERIOD).min()
    df_calc['senkou_span_b'] = ((high_52 + low_52) / 2).shift(KIJUN_PERIOD)
    df_calc['ichimoku_tenkan_kijun_cross_signal'] = np.where(df_calc['tenkan_sen'] > df_calc['kijun_sen'], 1, -1)
    df_calc['ichimoku_price_cloud_position'] = np.where(df_calc['close'] > df_calc[['senkou_span_a', 'senkou_span_b']].max(axis=1), 1, np.where(df_calc['close'] < df_calc[['senkou_span_a', 'senkou_span_b']].min(axis=1), -1, 0))
    df_calc['ichimoku_cloud_outlook'] = np.where(df_calc['senkou_span_a'] > df_calc['senkou_span_b'], 1, -1)

    # Fibonacci and S/R
    # ... (Full function copied from original script)
    rolling_high = df_calc['high'].rolling(window=FIB_SR_LOOKBACK_WINDOW)
    rolling_low = df_calc['low'].rolling(window=FIB_SR_LOOKBACK_WINDOW)
    swing_high = rolling_high.max()
    swing_low = rolling_low.min()
    price_range = swing_high - swing_low
    price_range[price_range == 0] = np.nan # Avoid division by zero
    df_calc['price_distance_to_recent_low_norm'] = (df_calc['close'] - swing_low) / price_range
    df_calc['price_distance_to_recent_high_norm'] = (swing_high - df_calc['close']) / price_range
    fib_50 = swing_high - (price_range * 0.5)
    df_calc['is_price_above_fib_50'] = (df_calc['close'] > fib_50).astype(int)

    # --- Define Target Variable ---
    df_calc['future_max_high'] = df_calc['high'].shift(-target_period).rolling(window=target_period).max()
    df_calc['target'] = ((df_calc['future_max_high'] / df_calc['close']) - 1 > price_change_threshold).astype(int)

    # --- Finalize Features and Target ---
    feature_columns = [
        'volume_15m_avg', 'rsi_momentum_bullish', 'btc_trend_feature', 'supertrend_direction',
        'ichimoku_tenkan_kijun_cross_signal', 'ichimoku_price_cloud_position', 'ichimoku_cloud_outlook',
        'price_distance_to_recent_low_norm', 'price_distance_to_recent_high_norm', 'is_price_above_fib_50'
    ]
    # Dropping columns that are not features and have NaNs from shifting
    df_final = df_calc.dropna(subset=feature_columns + ['target'])
    if df_final.empty:
        logger.warning(f"⚠️ [ML Prep] DataFrame لـ {symbol} فارغ بعد إزالة قيم NaN.")
        return None

    X = df_final[feature_columns]
    y = df_final['target']

    logger.info(f"✅ [ML Prep] إعداد البيانات لـ {symbol} اكتمل. {len(X)} عينة جاهزة للتدريب.")
    return X, y

# ---------------------- Model Training & Saving ----------------------
def train_and_evaluate_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[lgb.LGBMClassifier], Dict[str, Any]]:
    """Trains a LightGBM model and evaluates its performance."""
    logger.info("ℹ️ [ML Train] بدء تدريب وتقييم نموذج LightGBM...")

    if X.empty or y.empty:
        logger.error("❌ [ML Train] ميزات أو أهداف فارغة للتدريب.")
        return None, {}

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError: # Handle cases with only one class
        logger.warning("⚠️ [ML Train] لا يمكن استخدام stratify بسبب وجود فئة واحدة في الهدف. المتابعة بدونها.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use LightGBM Classifier
    model = lgb.LGBMClassifier(
        random_state=42,
        n_estimators=200,      # More trees
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        is_unbalanced=True,    # Handle class imbalance
        n_jobs=-1              # Use all available CPU cores
    )

    model.fit(X_train_scaled, y_train)
    logger.info("✅ [ML Train] تم تدريب النموذج بنجاح.")

    y_pred = model.predict(X_test_scaled)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'num_samples_trained': len(X_train),
        'num_samples_tested': len(X_test),
        'feature_names': X.columns.tolist()
    }
    feature_importance = dict(zip(X.columns, model.feature_importances_))

    logger.info(f"📊 [ML Train] مقاييس أداء النموذج:")
    for key, value in metrics.items():
        if isinstance(value, float): logger.info(f"  - {key.capitalize()}: {value:.4f}")

    return model, scaler, metrics, feature_importance

def save_ml_model_to_db(model: Any, scaler: Any, model_name: str, metrics: Dict[str, Any], feature_importance: Dict) -> bool:
    """Saves the trained model, scaler, and metadata to the database."""
    if not check_db_connection() or not conn:
        logger.error("❌ [DB Save] لا يمكن حفظ نموذج ML بسبب مشكلة في الاتصال بقاعدة البيانات.")
        return False

    logger.info(f"ℹ️ [DB Save] محاولة حفظ نموذج ML '{model_name}'...")
    try:
        # Pickle the model and scaler together
        model_and_scaler = {'model': model, 'scaler': scaler}
        model_binary = pickle.dumps(model_and_scaler)
        metrics_json = json.dumps(metrics, default=str)
        feature_importance_json = json.dumps(feature_importance, default=str)

        with conn.cursor() as db_cur:
            db_cur.execute("SELECT id FROM ml_models WHERE model_name = %s;", (model_name,))
            existing_model = db_cur.fetchone()
            if existing_model:
                logger.info(f"ℹ️ [DB Save] النموذج '{model_name}' موجود بالفعل. سيتم تحديثه.")
                update_query = sql.SQL("UPDATE ml_models SET model_data = %s, trained_at = NOW(), metrics = %s, feature_importance = %s WHERE id = %s;")
                db_cur.execute(update_query, (model_binary, metrics_json, feature_importance_json, existing_model['id']))
            else:
                insert_query = sql.SQL("INSERT INTO ml_models (model_name, model_data, metrics, feature_importance) VALUES (%s, %s, %s, %s);")
                db_cur.execute(insert_query, (model_name, model_binary, metrics_json, feature_importance_json))
        conn.commit()
        logger.info(f"✅ [DB Save] تم حفظ النموذج '{model_name}' في قاعدة البيانات بنجاح.")
        return True
    except Exception as e:
        logger.error(f"❌ [DB Save] خطأ غير متوقع أثناء حفظ نموذج ML: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

# ---------------------- Helper & Flask Functions ----------------------
def send_telegram_message(text: str) -> None:
    """Sends a notification message to the configured Telegram chat."""
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload, timeout=10)
        logger.info(f"✅ [Telegram] تم إرسال إشعار الحالة بنجاح.")
    except Exception as e:
        logger.error(f"❌ [Telegram] فشل إرسال إشعار الحالة: {e}")

app = Flask(__name__)
@app.route('/')
def home() -> Response:
    """Simple health check endpoint for Render."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    status_message = (
        f"🤖 **خدمة تدريب نماذج ML**\n"
        f"- الحالة الحالية: **{training_status}**\n"
        f"- آخر تحديث: {now}\n"
    )
    if last_training_time:
        status_message += f"- وقت آخر تدريب: {last_training_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    if last_training_summary:
        status_message += (
            f"- نماذج ناجحة: {last_training_summary.get('successful_models', 'N/A')}/{last_training_summary.get('total_models_attempted', 'N/A')}\n"
            f"- متوسط الدقة: {last_training_summary.get('avg_accuracy', 0.0):.2%}\n"
        )
    if training_error:
        status_message += f"- آخر خطأ: {training_error}\n"

    return Response(status_message, status=200, mimetype='text/plain; charset=utf-8')

def run_flask_service() -> None:
    """Runs the Flask application using Waitress for production."""
    host = "0.0.0.0"
    port = int(os.environ.get('PORT', 10001)) # Use a different port if running alongside bot
    logger.info(f"ℹ️ [Flask] بدء تطبيق Flask على {host}:{port}...")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=4)
    except ImportError:
        logger.warning("⚠️ [Flask] 'waitress' غير مثبت. الرجوع إلى خادم تطوير Flask.")
        app.run(host=host, port=port)
    except Exception as e:
        logger.critical(f"❌ [Flask] فشل بدء الخادم: {e}", exc_info=True)

# ---------------------- Main Execution Logic ----------------------
def main():
    global training_status, last_training_time, last_training_summary, training_error

    logger.info("🚀 بدء سكريبت تدريب نماذج ML...")
    start_time = datetime.now()
    send_telegram_message(f"🚀 *بدء تدريب نماذج ML*\n- الوقت: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        init_db()
        symbols = get_crypto_symbols()
        if not symbols:
            raise ValueError("قائمة الرموز فارغة أو غير صالحة. تحقق من `crypto_list.txt`.")

        training_status = f"جاري العمل: تدريب {len(symbols)} نموذج..."
        summary = {
            'total_models_attempted': len(symbols),
            'successful_models': 0,
            'failed_models': 0,
            'accuracies': [],
            'precisions': [],
        }

        for i, symbol in enumerate(symbols):
            logger.info(f"\n--- ⏳ ({i+1}/{len(symbols)}) بدء تدريب النموذج لـ {symbol} ---")
            try:
                df_hist = fetch_historical_data(symbol, interval=TRAINING_TIMEFRAME, days=DATA_LOOKBACK_DAYS_FOR_TRAINING)
                if df_hist is None or df_hist.empty:
                    logger.warning(f"⚠️ تخطي {symbol}: لا توجد بيانات تاريخية كافية.")
                    summary['failed_models'] += 1
                    continue

                prepared_data = prepare_data_for_ml(df_hist, symbol)
                if prepared_data is None:
                    logger.warning(f"⚠️ تخطي {symbol}: فشل في إعداد البيانات.")
                    summary['failed_models'] += 1
                    continue

                X, y = prepared_data
                model, scaler, metrics, feature_importance = train_and_evaluate_model(X, y)
                if model and metrics:
                    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
                    if save_ml_model_to_db(model, scaler, model_name, metrics, feature_importance):
                        summary['successful_models'] += 1
                        summary['accuracies'].append(metrics.get('accuracy', 0))
                        summary['precisions'].append(metrics.get('precision', 0))
                    else:
                         summary['failed_models'] += 1
                else:
                    summary['failed_models'] += 1

            except Exception as e:
                logger.error(f"❌ خطأ فادح أثناء تدريب النموذج لـ {symbol}: {e}", exc_info=True)
                summary['failed_models'] += 1

        end_time = datetime.now()
        last_training_time = end_time
        training_status = "مكتمل"
        if summary['failed_models'] > 0:
            training_status += " مع وجود أخطاء"

        # Final Summary
        summary['avg_accuracy'] = np.mean(summary['accuracies']) if summary['accuracies'] else 0
        summary['avg_precision'] = np.mean(summary['precisions']) if summary['precisions'] else 0
        last_training_summary = summary

        duration = str(end_time - start_time).split('.')[0]
        summary_text = (
            f"✅ *اكتمل تدريب نماذج ML*\n"
            f"----------------------------------\n"
            f"⏱️ **المدة:** {duration}\n"
            f"📈 **النماذج الناجحة:** {summary['successful_models']}/{summary['total_models_attempted']}\n"
            f"📉 **النماذج الفاشلة:** {summary['failed_models']}\n"
            f"🎯 **متوسط الدقة:** {summary['avg_accuracy']:.2%}\n"
            f" precision : {summary['avg_precision']:.2%}\n"
            f"----------------------------------"
        )
        logger.info(summary_text.replace('*', '').replace('_', ''))
        send_telegram_message(summary_text)

    except Exception as e:
        logger.critical(f"❌ [Main] حدث خطأ فادح في السكريبت: {e}", exc_info=True)
        training_status = "فشل"
        training_error = str(e)
        send_telegram_message(f"❌ *فشل تدريب نماذج ML*\n- خطأ: `{e}`")

    finally:
        logger.info("🛑 [Main] إيقاف تشغيل سكريبت التدريب...")
        if conn: conn.close()
        logger.info("👋 [Main] انتهى سكريبت تدريب نماذج ML.")
        # The script will end here. The Flask thread will be terminated by Render.


if __name__ == "__main__":
    # Run Flask in a background thread so the main training logic can execute
    flask_thread = Thread(target=run_flask_service, daemon=True, name="FlaskServiceThread")
    flask_thread.start()

    # Run the main training logic
    main()

