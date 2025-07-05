import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
import redis
from urllib.parse import urlparse
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException
from flask import Flask, request, Response, jsonify, render_template_string
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timedelta, timezone
from decouple import config
from typing import List, Dict, Optional, Any, Set
from sklearn.preprocessing import StandardScaler
from collections import deque
import warnings
import gc

# --- تجاهل التحذيرات غير الهامة ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v7_with_ichimoku.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV7_With_Ichimoku')

# ---------------------- تحميل متغيرات البيئة ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
    REDIS_URL: str = config('REDIS_URL', default='redis://localhost:6379/0')

except Exception as e:
     logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
     exit(1)

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V7_With_Ichimoku'
MODEL_FOLDER: str = 'V7'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 30
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices"
MODEL_BATCH_SIZE: int = 5
DIRECT_API_CHECK_INTERVAL: int = 10 

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
MODEL_CONFIDENCE_THRESHOLD = 0.70
MAX_OPEN_TRADES: int = 10
TRADE_AMOUNT_USDT: float = 10.0
USE_DYNAMIC_SL_TP = True
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.0
USE_BTC_TREND_FILTER = True
BTC_SYMBOL = 'BTCUSDT'
BTC_TREND_TIMEFRAME = '4h'
BTC_TREND_EMA_PERIOD = 50
MIN_PROFIT_PERCENTAGE_FILTER: float = 1.0

# --- ✨ جديد: ثوابت فلتر السرعة والتوقيت ---
# هذا الفلتر يهدف إلى اختيار الإشارات ذات الزخم الأعلى لضمان وصول أسرع للهدف
USE_SPEED_FILTER: bool = True  # تفعيل أو تعطيل الفلتر
SPEED_FILTER_ADX_THRESHOLD: float = 20.0  # الحد الأدنى لمؤشر ADX للدلالة على وجود اتجاه قوي
SPEED_FILTER_REL_VOL_THRESHOLD: float = 1.2  # الحد الأدنى لحجم التداول النسبي للدلالة على اهتمام المتداولين
SPEED_FILTER_RSI_MIN: float = 45.0  # الحد الأدنى لمؤشر RSI (يجب أن يكون في منطقة الزخم الصاعد)
SPEED_FILTER_RSI_MAX: float = 70.0  # الحد الأقصى لمؤشر RSI (لتجنب مناطق التشبع الشرائي الشديد)


# --- المتغيرات العامة وقفل العمليات ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
ml_models_cache: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
notifications_cache = deque(maxlen=50)
notifications_lock = Lock()
signals_pending_closure: Set[int] = set()
closure_lock = Lock()
last_api_check_time = time.time()


# ---------------------- دوال قاعدة البيانات (تبقى كما هي) ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    global conn
    logger.info("[قاعدة البيانات] بدء تهيئة الاتصال...")
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION NOT NULL,
                        target_price DOUBLE PRECISION NOT NULL, stop_loss DOUBLE PRECISION NOT NULL,
                        status TEXT DEFAULT 'open', closing_price DOUBLE PRECISION, closed_at TIMESTAMP,
                        profit_percentage DOUBLE PRECISION, strategy_name TEXT, signal_details JSONB );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications ( id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL, message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE );
                """)
            conn.commit()
            logger.info("✅ [قاعدة البيانات] تم تهيئة جداول قاعدة البيانات بنجاح.")
            return
        except Exception as e:
            logger.error(f"❌ [قاعدة البيانات] خطأ في الاتصال (المحاولة {attempt + 1}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: logger.critical("❌ [قاعدة البيانات] فشل الاتصال بعد عدة محاولات."); exit(1)

def check_db_connection() -> bool:
    global conn
    if conn is None or conn.closed != 0:
        logger.warning("[قاعدة البيانات] الاتصال مغلق، محاولة إعادة الاتصال...")
        init_db()
    try:
        if conn: conn.cursor().execute("SELECT 1;"); return True
        return False
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [قاعدة البيانات] فقدان الاتصال: {e}. محاولة إعادة الاتصال...")
        try: init_db(); return conn is not None and conn.closed == 0
        except Exception as retry_e: logger.error(f"❌ [قاعدة البيانات] فشل إعادة الاتصال: {retry_e}"); return False
    return False

def log_and_notify(level: str, message: str, notification_type: str):
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error, 'critical': logger.critical}
    log_methods.get(level.lower(), logger.info)(message)
    if not check_db_connection() or not conn: return
    try:
        new_notification = {"timestamp": datetime.now().isoformat(), "type": notification_type, "message": message}
        with notifications_lock: notifications_cache.appendleft(new_notification)
        with conn.cursor() as cur:
            cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [Notify DB] فشل حفظ التنبيه في قاعدة البيانات: {e}");
        if conn: conn.rollback()

def init_redis() -> None:
    global redis_client
    logger.info("[Redis] بدء تهيئة الاتصال...")
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] تم الاتصال بنجاح بخادم Redis.")
    except redis.exceptions.ConnectionError as e:
        logger.critical(f"❌ [Redis] فشل الاتصال بـ Redis على {REDIS_URL}. الخطأ: {e}")
        exit(1)
    except Exception as e:
        logger.critical(f"❌ [Redis] حدث خطأ غير متوقع أثناء تهيئة Redis: {e}")
        exit(1)

# ---------------------- دوال Binance والبيانات (تبقى كما هي) ----------------------
def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    logger.info(f"ℹ️ [التحقق] قراءة الرموز من '{filename}' والتحقق منها مع Binance...")
    if not client: logger.error("❌ [التحقق] كائن Binance client غير مهيأ."); return []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        exchange_info = client.get_exchange_info()
        active = {s['symbol'] for s in exchange_info['symbols'] if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'}
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ [التحقق] سيقوم البوت بمراقبة {len(validated)} عملة معتمدة.")
        return validated
    except Exception as e:
        logger.error(f"❌ [التحقق] حدث خطأ أثناء التحقق من الرموز: {e}", exc_info=True); return []

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
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
    except BinanceAPIException as e:
        logger.warning(f"⚠️ [API Binance] خطأ في جلب بيانات {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ [البيانات] خطأ أثناء جلب البيانات التاريخية لـ {symbol}: {e}")
        return None

def fetch_sr_levels_from_db(symbol: str) -> pd.DataFrame:
    if not check_db_connection() or not conn: return pd.DataFrame()
    query = "SELECT level_price, level_type, score FROM support_resistance_levels WHERE symbol = %s"
    try:
        with conn.cursor() as cur:
            cur.execute(query, (symbol,))
            levels = cur.fetchall()
            if not levels: return pd.DataFrame()
            return pd.DataFrame(levels)
    except Exception as e:
        logger.error(f"❌ [S/R Fetch Bot] Could not fetch S/R levels for {symbol}: {e}")
        if conn: conn.rollback()
        return pd.DataFrame()

def fetch_ichimoku_features_from_db(symbol: str, timeframe: str) -> pd.DataFrame:
    if not check_db_connection() or not conn: return pd.DataFrame()
    query = """
        SELECT timestamp, tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
        FROM ichimoku_features
        WHERE symbol = %s AND timeframe = %s
        ORDER BY timestamp;
    """
    try:
        with conn.cursor() as cur:
            cur.execute(query, (symbol, timeframe))
            features = cur.fetchall()
            if not features: return pd.DataFrame()
            df_ichimoku = pd.DataFrame(features)
            df_ichimoku['timestamp'] = pd.to_datetime(df_ichimoku['timestamp'], utc=True)
            df_ichimoku.set_index('timestamp', inplace=True)
            return df_ichimoku
    except Exception as e:
        logger.error(f"❌ [Ichimoku Fetch Bot] Could not fetch Ichimoku features for {symbol}: {e}")
        if conn: conn.rollback()
        return pd.DataFrame()

def calculate_ichimoku_based_features(df: pd.DataFrame) -> pd.DataFrame:
    df['price_vs_tenkan'] = (df['close'] - df['tenkan_sen']) / df['tenkan_sen']
    df['price_vs_kijun'] = (df['close'] - df['kijun_sen']) / df['kijun_sen']
    df['tenkan_vs_kijun'] = (df['tenkan_sen'] - df['kijun_sen']) / df['kijun_sen']
    df['price_vs_kumo_a'] = (df['close'] - df['senkou_span_a']) / df['senkou_span_a']
    df['price_vs_kumo_b'] = (df['close'] - df['senkou_span_b']) / df['senkou_span_b']
    df['kumo_thickness'] = (df['senkou_span_a'] - df['senkou_span_b']).abs() / df['close']
    kumo_high = df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
    kumo_low = df[['senkou_span_a', 'senkou_span_b']].min(axis=1)
    df['price_above_kumo'] = (df['close'] > kumo_high).astype(int)
    df['price_below_kumo'] = (df['close'] < kumo_low).astype(int)
    df['price_in_kumo'] = ((df['close'] >= kumo_low) & (df['close'] <= kumo_high)).astype(int)
    df['chikou_above_kumo'] = (df['chikou_span'] > kumo_high).astype(int)
    df['chikou_below_kumo'] = (df['chikou_span'] < kumo_low).astype(int)
    df['tenkan_kijun_cross'] = 0
    cross_up = (df['tenkan_sen'].shift(1) < df['kijun_sen'].shift(1)) & (df['tenkan_sen'] > df['kijun_sen'])
    cross_down = (df['tenkan_sen'].shift(1) > df['kijun_sen'].shift(1)) & (df['tenkan_sen'] < df['kijun_sen'])
    df.loc[cross_up, 'tenkan_kijun_cross'] = 1
    df.loc[cross_down, 'tenkan_kijun_cross'] = -1
    return df

def calculate_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df_patterns = df.copy()
    op, hi, lo, cl = df_patterns['open'], df_patterns['high'], df_patterns['low'], df_patterns['close']
    body = abs(cl - op)
    candle_range = hi - lo
    candle_range[candle_range == 0] = 1e-9
    upper_wick = hi - pd.concat([op, cl], axis=1).max(axis=1)
    lower_wick = pd.concat([op, cl], axis=1).min(axis=1) - lo
    df_patterns['candlestick_pattern'] = 0
    is_bullish_marubozu = (cl > op) & (body / candle_range > 0.95) & (upper_wick < body * 0.1) & (lower_wick < body * 0.1)
    is_bearish_marubozu = (op > cl) & (body / candle_range > 0.95) & (upper_wick < body * 0.1) & (lower_wick < body * 0.1)
    is_bullish_engulfing = (cl.shift(1) < op.shift(1)) & (cl > op) & (cl >= op.shift(1)) & (op <= cl.shift(1)) & (body > body.shift(1))
    is_bearish_engulfing = (cl.shift(1) > op.shift(1)) & (cl < op) & (op >= cl.shift(1)) & (cl <= op.shift(1)) & (body > body.shift(1))
    is_hammer = (body > candle_range * 0.1) & (lower_wick >= body * 2) & (upper_wick < body)
    is_shooting_star = (body > candle_range * 0.1) & (upper_wick >= body * 2) & (lower_wick < body)
    is_doji = (body / candle_range) < 0.05
    df_patterns.loc[is_doji, 'candlestick_pattern'] = 3
    df_patterns.loc[is_hammer, 'candlestick_pattern'] = 2
    df_patterns.loc[is_shooting_star, 'candlestick_pattern'] = -2
    df_patterns.loc[is_bullish_engulfing, 'candlestick_pattern'] = 1
    df_patterns.loc[is_bearish_engulfing, 'candlestick_pattern'] = -1
    df_patterns.loc[is_bullish_marubozu, 'candlestick_pattern'] = 4
    df_patterns.loc[is_bearish_marubozu, 'candlestick_pattern'] = -4
    return df_patterns

def calculate_sr_features(df: pd.DataFrame, sr_levels_df: pd.DataFrame) -> pd.DataFrame:
    if sr_levels_df.empty:
        df['dist_to_support'] = 0.0; df['dist_to_resistance'] = 0.0
        df['score_of_support'] = 0.0; df['score_of_resistance'] = 0.0
        return df
    supports = sr_levels_df[sr_levels_df['level_type'].str.contains('support|poc|confluence', case=False)]['level_price'].sort_values().to_numpy()
    resistances = sr_levels_df[sr_levels_df['level_type'].str.contains('resistance|poc|confluence', case=False)]['level_price'].sort_values().to_numpy()
    support_scores = sr_levels_df[sr_levels_df['level_type'].str.contains('support|poc|confluence', case=False)].set_index('level_price')['score'].to_dict()
    resistance_scores = sr_levels_df[sr_levels_df['level_type'].str.contains('resistance|poc|confluence', case=False)].set_index('level_price')['score'].to_dict()

    def get_sr_info(price):
        dist_support, score_support, dist_resistance, score_resistance = 1.0, 0.0, 1.0, 0.0
        if supports.size > 0:
            idx = np.searchsorted(supports, price, side='right') - 1
            if idx >= 0:
                nearest_support_price = supports[idx]
                dist_support = (price - nearest_support_price) / price if price > 0 else 0
                score_support = support_scores.get(nearest_support_price, 0)
        if resistances.size > 0:
            idx = np.searchsorted(resistances, price, side='left')
            if idx < len(resistances):
                nearest_resistance_price = resistances[idx]
                dist_resistance = (nearest_resistance_price - price) / price if price > 0 else 0
                score_resistance = resistance_scores.get(nearest_resistance_price, 0)
        return dist_support, score_support, dist_resistance, score_resistance
    results = df['close'].apply(get_sr_info)
    df[['dist_to_support', 'score_of_support', 'dist_to_resistance', 'score_of_resistance']] = pd.DataFrame(results.tolist(), index=df.index)
    return df

def calculate_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    up_move = df_calc['high'].diff(); down_move = -df_calc['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr']
    minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr']
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    ema_fast = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow; signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df_calc['macd_hist'] = macd_line - signal_line
    df_calc['macd_cross'] = 0
    df_calc.loc[(df_calc['macd_hist'].shift(1) < 0) & (df_calc['macd_hist'] >= 0), 'macd_cross'] = 1
    df_calc.loc[(df_calc['macd_hist'].shift(1) > 0) & (df_calc['macd_hist'] <= 0), 'macd_cross'] = -1
    sma = df_calc['close'].rolling(window=BBANDS_PERIOD).mean()
    std_dev = df_calc['close'].rolling(window=BBANDS_PERIOD).std()
    upper_band = sma + (std_dev * 2); lower_band = sma - (std_dev * 2)
    df_calc['bb_width'] = (upper_band - lower_band) / (sma + 1e-9)
    rsi = df_calc['rsi']
    min_rsi = rsi.rolling(window=STOCH_RSI_PERIOD).min(); max_rsi = rsi.rolling(window=STOCH_RSI_PERIOD).max()
    stoch_rsi_val = (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(window=STOCH_K).mean() * 100
    df_calc['stoch_rsi_d'] = df_calc['stoch_rsi_k'].rolling(window=STOCH_D).mean()
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)
    df_calc['market_condition'] = 0
    df_calc.loc[(df_calc['rsi'] > RSI_OVERBOUGHT) | (df_calc['stoch_rsi_k'] > STOCH_RSI_OVERBOUGHT), 'market_condition'] = 1
    df_calc.loc[(df_calc['rsi'] < RSI_OVERSOLD) | (df_calc['stoch_rsi_k'] < STOCH_RSI_OVERSOLD), 'market_condition'] = -1
    ema_fast_trend = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    ema_slow_trend = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    df_calc['price_vs_ema50'] = (df_calc['close'] / ema_fast_trend) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / ema_slow_trend) - 1
    df_calc['returns'] = df_calc['close'].pct_change()
    merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
    df_calc['btc_correlation'] = merged_df['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    df_calc['hour_of_day'] = df_calc.index.hour
    df_calc = calculate_candlestick_patterns(df_calc)
    return df_calc.astype('float32', errors='ignore')

def load_ml_model_bundle_from_folder(symbol: str) -> Optional[Dict[str, Any]]:
    global ml_models_cache
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_name in ml_models_cache:
        return ml_models_cache[model_name]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FOLDER, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        logger.warning(f"⚠️ [نموذج تعلم الآلة] ملف النموذج '{model_path}' غير موجود للعملة {symbol}.")
        return None
    try:
        with open(model_path, 'rb') as f:
            model_bundle = pickle.load(f)
        if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
            ml_models_cache[model_name] = model_bundle
            logger.info(f"✅ [نموذج تعلم الآلة] تم تحميل النموذج '{model_name}' بنجاح من الملف المحلي.")
            return model_bundle
        else:
            logger.error(f"❌ [نموذج تعلم الآلة] حزمة النموذج في '{model_path}' غير مكتملة.")
            return None
    except Exception as e:
        logger.error(f"❌ [نموذج تعلم الآلة] خطأ في تحميل حزمة النموذج من الملف للعملة {symbol}: {e}", exc_info=True)
        return None

# ---------------------- دوال WebSocket والاستراتيجية ----------------------

def handle_price_update_message(msg: List[Dict[str, Any]]) -> None:
    global redis_client
    try:
        if not isinstance(msg, list):
            logger.warning(f"⚠️ [WebSocket] تم استلام رسالة بتنسيق غير متوقع: {type(msg)}")
            return
        if not redis_client:
            logger.error("❌ [WebSocket] كائن Redis غير مهيأ. لا يمكن حفظ الأسعار.")
            return

        price_updates = {item.get('s'): float(item.get('c', 0)) for item in msg if item.get('s') and item.get('c')}
        
        if price_updates:
            redis_client.hset(REDIS_PRICES_HASH_NAME, mapping=price_updates)
            
    except Exception as e:
        logger.error(f"❌ [WebSocket Price Updater] خطأ في معالجة رسالة السعر: {e}", exc_info=True)


def initiate_signal_closure(symbol: str, signal_to_close: Dict, status: str, closing_price: float):
    signal_id = signal_to_close.get('id')
    
    with closure_lock:
        if signal_id in signals_pending_closure:
            logger.warning(f"⚠️ [CLOSURE] Attempted to re-initiate closure for signal {signal_id} which is already pending.")
            return
        signals_pending_closure.add(signal_id)
    
    with signal_cache_lock:
        signal_data_for_thread = open_signals_cache.pop(symbol, None)

    if signal_data_for_thread:
        logger.info(f"⚡ [CLOSURE] Signal {signal_id} for {symbol} removed from cache. Starting close_signal thread.")
        Thread(target=close_signal, args=(signal_data_for_thread, status, closing_price, "auto_monitor")).start()
    else:
        logger.warning(f"⚠️ [CLOSURE] Signal {signal_id} for {symbol} was identified for closure, but was not found in the open_signals_cache when removal was attempted.")
        with closure_lock:
            signals_pending_closure.discard(signal_id)


def trade_monitoring_loop():
    global last_api_check_time
    logger.info("✅ [Trade Monitor] خيط مراقبة الصفقات الهجين بدأ بالعمل.")
    
    while True:
        try:
            with signal_cache_lock:
                signals_to_check = dict(open_signals_cache)

            if not signals_to_check or not redis_client or not client:
                time.sleep(1)
                continue

            perform_direct_api_check = (time.time() - last_api_check_time) > DIRECT_API_CHECK_INTERVAL
            if perform_direct_api_check:
                logger.debug(f"🔄 [Direct API Check] حان وقت الفحص المباشر من API (كل {DIRECT_API_CHECK_INTERVAL} ثانية).")
                last_api_check_time = time.time()

            symbols_to_fetch = list(signals_to_check.keys())
            redis_prices_list = redis_client.hmget(REDIS_PRICES_HASH_NAME, symbols_to_fetch)
            redis_prices = {symbol: price for symbol, price in zip(symbols_to_fetch, redis_prices_list)}

            for symbol, signal in signals_to_check.items():
                signal_id = signal.get('id')
                
                with closure_lock:
                    if signal_id in signals_pending_closure:
                        continue

                price = None
                price_source = "None"

                if perform_direct_api_check:
                    try:
                        ticker = client.get_symbol_ticker(symbol=symbol)
                        price = float(ticker['price'])
                        price_source = "Direct API"
                    except Exception as e:
                        logger.error(f"❌ [Direct API Check] فشل جلب السعر لـ {symbol}: {e}")
                        if redis_prices.get(symbol):
                            price = float(redis_prices[symbol])
                            price_source = "Redis (Fallback)"
                else:
                    if redis_prices.get(symbol):
                        price = float(redis_prices[symbol])
                        price_source = "Redis"

                target_price = float(signal.get('target_price', 0))
                stop_loss_price = float(signal.get('stop_loss', 0))

                logger.debug(f"[MONITOR] ID:{signal_id} | {symbol} | Price: {price} ({price_source}) | TP: {target_price} | SL: {stop_loss_price}")
                
                if not all([price, target_price > 0, stop_loss_price > 0]):
                    logger.warning(f"  -> [SKIP] بيانات غير صالحة أو سعر غير متوفر لـ {symbol} (ID: {signal_id}).")
                    continue
                
                status_to_set = None
                if price >= target_price:
                    status_to_set = 'target_hit'
                elif price <= stop_loss_price:
                    status_to_set = 'stop_loss_hit'

                if status_to_set:
                    logger.info(f"✅ [TRIGGER] ID:{signal_id} | {symbol} | Condition '{status_to_set}' met. Price {price} ({price_source}) breached level (TP: {target_price}, SL: {stop_loss_price}).")
                    initiate_signal_closure(symbol, signal, status_to_set, price)

            time.sleep(0.2)

        except Exception as e:
            logger.error(f"❌ [Trade Monitor] خطأ فادح في حلقة المراقبة: {e}", exc_info=True)
            time.sleep(5)


def run_websocket_manager() -> None:
    logger.info("ℹ️ [WebSocket] بدء مدير WebSocket...")
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    twm.start_miniticker_socket(callback=handle_price_update_message)
    logger.info("✅ [WebSocket] تم الاتصال والاستماع إلى 'All Market Mini Tickers' بنجاح.")
    twm.join()

class TradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_bundle_from_folder(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame, sr_levels_df: pd.DataFrame, ichimoku_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            df_featured = calculate_features(df_15m, btc_df)
            df_featured = calculate_sr_features(df_featured, sr_levels_df)
            if not ichimoku_df.empty:
                df_featured = df_featured.join(ichimoku_df, how='left')
                df_featured = calculate_ichimoku_based_features(df_featured)
            
            delta_4h = df_4h['close'].diff()
            gain_4h = delta_4h.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
            loss_4h = -delta_4h.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
            df_4h['rsi_4h'] = 100 - (100 / (1 + (gain_4h / loss_4h.replace(0, 1e-9))))
            ema_fast_4h = df_4h['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
            df_4h['price_vs_ema50_4h'] = (df_4h['close'] / ema_fast_4h) - 1
            mtf_features = df_4h[['rsi_4h', 'price_vs_ema50_4h']]
            df_featured = df_featured.join(mtf_features)
            df_featured[['rsi_4h', 'price_vs_ema50_4h']] = df_featured[['rsi_4h', 'price_vs_ema50_4h']].fillna(method='ffill')
            
            for col in self.feature_names:
                if col not in df_featured.columns:
                    df_featured[col] = 0.0
            
            df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_featured[self.feature_names].dropna()

        except Exception as e:
            logger.error(f"❌ [{self.symbol}] فشل هندسة الميزات: {e}", exc_info=True)
            return None

    def generate_signal(self, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]): return None
        if df_features.empty: return None
        last_row_df = df_features.iloc[[-1]]
        try:
            features_scaled = self.scaler.transform(last_row_df)
            features_scaled_df = pd.DataFrame(features_scaled, columns=self.feature_names)
            prediction = self.ml_model.predict(features_scaled_df)[0]
            prediction_proba = self.ml_model.predict_proba(features_scaled_df)[0]
            try: class_1_index = list(self.ml_model.classes_).index(1)
            except ValueError: return None
            prob_for_class_1 = prediction_proba[class_1_index]
            if prediction == 1 and prob_for_class_1 >= MODEL_CONFIDENCE_THRESHOLD:
                logger.info(f"✅ [العثور على إشارة] {self.symbol}: تنبأ النموذج 'شراء' (1) بثقة {prob_for_class_1:.2%}, وهي أعلى من الحد المطلوب ({MODEL_CONFIDENCE_THRESHOLD:.0%}).")
                return {'symbol': self.symbol, 'strategy_name': BASE_ML_MODEL_NAME, 'signal_details': {'ML_Probability_Buy': f"{prob_for_class_1:.2%}"}}
            return None
        except Exception as e:
            logger.warning(f"⚠️ [توليد إشارة] {self.symbol}: خطأ أثناء التوليد: {e}", exc_info=True)
            return None

# ---------------------- دوال التنبيهات والإدارة ----------------------
def send_telegram_message(target_chat_id: str, text: str):
    if not TELEGRAM_TOKEN or not target_chat_id: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(target_chat_id), 'text': text, 'parse_mode': 'Markdown'}
    try: requests.post(url, json=payload, timeout=20).raise_for_status()
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

def send_new_signal_alert(signal_data: Dict[str, Any]) -> None:
    safe_symbol = signal_data['symbol'].replace('_', '\\_')
    entry, target, sl = signal_data['entry_price'], signal_data['target_price'], signal_data['stop_loss']
    profit_pct = ((target / entry) - 1) * 100
    message = (f"💡 *إشارة تداول جديدة ({BASE_ML_MODEL_NAME})* 💡\n\n"
               f"🪙 *العملة:* `{safe_symbol}`\n📈 *النوع:* شراء (LONG)\n\n"
               f"⬅️ *سعر الدخول:* `${entry:,.8g}`\n"
               f"🎯 *الهدف:* `${target:,.8g}` (ربح متوقع `{profit_pct:+.2f}%`)\n"
               f"🛑 *وقف الخسارة:* `${sl:,.8g}`\n\n"
               f"🔍 *ثقة النموذج:* {signal_data['signal_details']['ML_Probability_Buy']}")
    reply_markup = {"inline_keyboard": [[{"text": "📊 فتح لوحة التحكم", "url": WEBHOOK_URL or '#'}]]}
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(CHAT_ID), 'text': message, 'parse_mode': 'Markdown', 'reply_markup': json.dumps(reply_markup)}
    try: requests.post(url, json=payload, timeout=20).raise_for_status()
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال تنبيه الإشارة الجديدة: {e}")
    log_and_notify('info', f"إشارة جديدة: {signal_data['symbol']} بسعر دخول ${entry:,.8g}", "NEW_SIGNAL")

def insert_signal_into_db(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not check_db_connection() or not conn: return None
    try:
        entry, target, sl = float(signal['entry_price']), float(signal['target_price']), float(signal['stop_loss'])
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;",
                (signal['symbol'], entry, target, sl, signal.get('strategy_name'), json.dumps(signal.get('signal_details', {})))
            )
            signal['id'] = cur.fetchone()['id']
        conn.commit()
        logger.info(f"✅ [قاعدة البيانات] تم إدراج الإشارة لـ {signal['symbol']} (ID: {signal['id']}).")
        return signal
    except Exception as e:
        logger.error(f"❌ [إدراج في قاعدة البيانات] خطأ في إدراج إشارة {signal['symbol']}: {e}", exc_info=True)
        if conn: conn.rollback(); return None

def close_signal(signal: Dict, status: str, closing_price: float, closed_by: str):
    signal_id = signal.get('id')
    symbol = signal.get('symbol')
    logger.info(f"Closing process started for Signal ID {signal_id} ({symbol}) with status '{status}'")
    
    try:
        if not check_db_connection() or not conn:
            raise OperationalError(f"DB connection failed for closing signal {signal_id}.")

        db_closing_price = float(closing_price)
        db_profit_pct = float(((db_closing_price / signal['entry_price']) - 1) * 100)
        
        with conn.cursor() as update_cur:
            update_cur.execute(
                "UPDATE signals SET status = %s, closing_price = %s, closed_at = NOW(), profit_percentage = %s WHERE id = %s AND status = 'open';",
                (status, db_closing_price, db_profit_pct, signal_id)
            )
            if update_cur.rowcount == 0:
                logger.warning(f"⚠️ [DB Close] Signal {signal_id} was not found or already closed. No recovery needed.")
                return 
        conn.commit()
        
        status_map = {'target_hit': '✅ تحقق الهدف', 'stop_loss_hit': '🛑 ضرب وقف الخسارة', 'manual_close': '🖐️ أُغلقت يدوياً'}
        status_message = status_map.get(status, status.replace('_', ' ').title())
        safe_symbol = signal['symbol'].replace('_', '\\_')
        alert_msg_tg = f"*{status_message}*\n`{safe_symbol}` | *الربح:* `{db_profit_pct:+.2f}%`"
        send_telegram_message(CHAT_ID, alert_msg_tg)
        alert_msg_db = f"{status_message}: {signal['symbol']} | الربح: {db_profit_pct:+.2f}%"
        log_and_notify('info', alert_msg_db, 'CLOSE_SIGNAL')
        logger.info(f"✅ [DB Close] Successfully closed signal {signal_id}.")

    except Exception as e:
        logger.error(f"❌ [DB Close] Critical error during signal close for ID {signal_id}: {e}", exc_info=True)
        if conn: 
            try: conn.rollback()
            except Exception as rb_e: logger.error(f"❌ [DB Close] Error during rollback: {rb_e}")

        if symbol:
            with signal_cache_lock:
                if symbol not in open_signals_cache:
                    open_signals_cache[symbol] = signal
                    logger.info(f"🔄 [Recovery] Signal {signal_id} for {symbol} has been returned to the open signals cache due to a closing error.")
    finally:
        with closure_lock:
            signals_pending_closure.discard(signal_id)
            logger.info(f"Signal ID {signal_id} removed from pending closure set.")

def load_open_signals_to_cache():
    if not check_db_connection() or not conn: return
    logger.info("ℹ️ [تحميل الذاكرة المؤقتة] جاري تحميل الإشارات المفتوحة سابقاً...")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status = 'open';")
            open_signals = cur.fetchall()
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in open_signals: open_signals_cache[signal['symbol']] = dict(signal)
            logger.info(f"✅ [تحميل الذاكرة المؤقتة] تم تحميل {len(open_signals)} إشارة مفتوحة.")
    except Exception as e: logger.error(f"❌ [تحميل الذاكرة المؤقتة] فشل تحميل الإشارات المفتوحة: {e}")

def load_notifications_to_cache():
    if not check_db_connection() or not conn: return
    logger.info("ℹ️ [تحميل الذاكرة المؤقتة] جاري تحميل آخر التنبيهات...")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 50;")
            recent = cur.fetchall()
            with notifications_lock:
                notifications_cache.clear()
                for n in reversed(recent): n['timestamp'] = n['timestamp'].isoformat(); notifications_cache.appendleft(dict(n))
            logger.info(f"✅ [تحميل الذاكرة المؤقتة] تم تحميل {len(notifications_cache)} تنبيه.")
    except Exception as e: logger.error(f"❌ [تحميل الذاكرة المؤقتة] فشل تحميل التنبيهات: {e}")

# ---------------------- حلقة العمل الرئيسية ----------------------
def get_btc_trend() -> Dict[str, Any]:
    if not client: return {"status": "error", "message": "Binance client not initialized", "is_uptrend": False}
    try:
        klines = client.get_klines(symbol=BTC_SYMBOL, interval=BTC_TREND_TIMEFRAME, limit=BTC_TREND_EMA_PERIOD * 2)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df['close'] = pd.to_numeric(df['close'])
        ema = df['close'].ewm(span=BTC_TREND_EMA_PERIOD, adjust=False).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        status, message = ("Uptrend", f"صاعد (السعر فوق EMA {BTC_TREND_EMA_PERIOD})") if current_price > ema else ("Downtrend", f"هابط (السعر تحت EMA {BTC_TREND_EMA_PERIOD})")
        return {"status": status, "message": message, "is_uptrend": (status == "Uptrend")}
    except Exception as e:
        logger.error(f"❌ [فلتر BTC] فشل تحديد اتجاه البيتكوين: {e}")
        return {"status": "Error", "message": str(e), "is_uptrend": False}

def get_btc_data_for_bot() -> Optional[pd.DataFrame]:
    logger.info("ℹ️ [بيانات BTC] جاري جلب بيانات البيتكوين لحساب المؤشرات...")
    btc_data = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
    if btc_data is None:
        logger.error("❌ [بيانات BTC] فشل جلب بيانات البيتكوين. سيتخطى البوت الارتباط.")
        return None
    btc_data['btc_returns'] = btc_data['close'].pct_change()
    return btc_data

def main_loop():
    logger.info("[الحلقة الرئيسية] انتظار اكتمال التهيئة الأولية...")
    time.sleep(15) 
    if not validated_symbols_to_scan:
        log_and_notify("critical", "لا توجد رموز معتمدة للمسح. لن يستمر البوت في العمل.", "SYSTEM"); return
    log_and_notify("info", f"بدء حلقة المسح الرئيسية لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")
    
    all_symbols = list(validated_symbols_to_scan)

    while True:
        try:
            for i in range(0, len(all_symbols), MODEL_BATCH_SIZE):
                symbol_batch = all_symbols[i:i + MODEL_BATCH_SIZE]
                
                num_batches = -(-len(all_symbols) // MODEL_BATCH_SIZE)
                logger.info(f"🧠 [إدارة الذاكرة] بدء معالجة الدفعة ({i // MODEL_BATCH_SIZE + 1}/{num_batches}).")
                logger.info("🧠 [إدارة الذاكرة] مسح ذاكرة التخزين المؤقت للنماذج السابقة...")
                
                ml_models_cache.clear()
                gc.collect()
                
                logger.info(f"🧠 [إدارة الذاكرة] تم مسح الذاكرة المؤقتة. الرموز في هذه الدفعة: {symbol_batch}")

                if USE_BTC_TREND_FILTER:
                    trend_data = get_btc_trend()
                    if not trend_data.get("is_uptrend"):
                        logger.warning(f"⚠️ [إيقاف المسح] تم إيقاف البحث بسبب الاتجاه الهابط للبيتكوين. {trend_data.get('message')}")
                        time.sleep(300)
                        break

                with signal_cache_lock: open_count = len(open_signals_cache)
                if open_count >= MAX_OPEN_TRADES:
                    logger.info(f"ℹ️ [إيقاف مؤقت] تم الوصول للحد الأقصى للصفقات ({open_count}/{MAX_OPEN_TRADES}).")
                    time.sleep(60)
                    break 

                slots_available = MAX_OPEN_TRADES - open_count
                if slots_available <= 0:
                    break

                logger.info(f"ℹ️ [بدء المسح] بدء دورة مسح للدفعة الحالية. المراكز المتاحة: {slots_available}")
                
                btc_data = get_btc_data_for_bot()
                if btc_data is None:
                    logger.error("❌ فشل حاسم في جلب بيانات BTC. سيتم تخطي دورة المسح هذه."); 
                    time.sleep(120)
                    continue
                
                for symbol in symbol_batch:
                    if slots_available <= 0: break
                    with signal_cache_lock:
                        if symbol in open_signals_cache: continue
                    
                    try:
                        df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        if df_15m is None or df_15m.empty or df_4h is None or df_4h.empty: continue
                        
                        sr_levels = fetch_sr_levels_from_db(symbol)
                        ichimoku_data = fetch_ichimoku_features_from_db(symbol, SIGNAL_GENERATION_TIMEFRAME)
                        
                        strategy = TradingStrategy(symbol)
                        df_features = strategy.get_features(df_15m, df_4h, btc_data, sr_levels, ichimoku_data)
                        
                        del df_15m, df_4h, sr_levels, ichimoku_data; gc.collect()
                        
                        if df_features is None or df_features.empty: continue
                        
                        potential_signal = strategy.generate_signal(df_features)
                        if potential_signal and redis_client:
                            current_price_str = redis_client.hget(REDIS_PRICES_HASH_NAME, symbol)
                            if not current_price_str:
                                 logger.warning(f"⚠️ {symbol}: لا يمكن الحصول على السعر الحالي من Redis. سيتم التخطي."); continue
                            current_price = float(current_price_str)

                            # --- ✨ تعديل: تطبيق فلتر السرعة والتوقيت ---
                            passes_speed_filter = True
                            if USE_SPEED_FILTER:
                                last_features = df_features.iloc[-1]
                                adx = last_features.get('adx', 0)
                                rel_vol = last_features.get('relative_volume', 0)
                                rsi = last_features.get('rsi', 0)

                                # التحقق من شروط الفلتر
                                cond1 = adx >= SPEED_FILTER_ADX_THRESHOLD
                                cond2 = rel_vol >= SPEED_FILTER_REL_VOL_THRESHOLD
                                cond3 = SPEED_FILTER_RSI_MIN <= rsi < SPEED_FILTER_RSI_MAX

                                if not (cond1 and cond2 and cond3):
                                    passes_speed_filter = False
                                    logger.info(f"ℹ️ [{symbol}] تم تخطي الإشارة بسبب فلتر السرعة. "
                                                f"ADX: {adx:.2f} (Req: >{SPEED_FILTER_ADX_THRESHOLD}), "
                                                f"RelVol: {rel_vol:.2f} (Req: >{SPEED_FILTER_REL_VOL_THRESHOLD}), "
                                                f"RSI: {rsi:.2f} (Req: {SPEED_FILTER_RSI_MIN}-{SPEED_FILTER_RSI_MAX})")
                            
                            if passes_speed_filter:
                                logger.info(f"✅ [{symbol}] الإشارة مرت من فلتر السرعة. جاري حساب الأهداف...")
                                potential_signal['entry_price'] = current_price
                                if USE_DYNAMIC_SL_TP:
                                    atr_value = df_features['atr'].iloc[-1]
                                    potential_signal['stop_loss'] = current_price - (atr_value * ATR_SL_MULTIPLIER)
                                    potential_signal['target_price'] = current_price + (atr_value * ATR_TP_MULTIPLIER)
                                else:
                                    potential_signal['target_price'] = current_price * 1.02
                                    potential_signal['stop_loss'] = current_price * 0.985
                                
                                entry = potential_signal.get('entry_price', 0)
                                target = potential_signal.get('target_price', 0)

                                if entry > 0 and target > entry:
                                    profit_percentage = ((target / entry) - 1) * 100
                                    if profit_percentage >= MIN_PROFIT_PERCENTAGE_FILTER:
                                        logger.info(f"✅ [{symbol}] الإشارة مرت من فلتر الربح بنسبة {profit_percentage:.2f}%. جاري الحفظ...")
                                        saved_signal = insert_signal_into_db(potential_signal)
                                        if saved_signal:
                                            with signal_cache_lock: open_signals_cache[saved_signal['symbol']] = saved_signal
                                            send_new_signal_alert(saved_signal)
                                            slots_available -= 1
                                    else:
                                        logger.info(f"ℹ️ [{symbol}] تم تخطي الإشارة. الربح المتوقع {profit_percentage:.2f}% وهو أقل من الحد الأدنى المطلوب ({MIN_PROFIT_PERCENTAGE_FILTER}%).")
                                else:
                                    logger.warning(f"⚠️ [{symbol}] سعر دخول أو هدف غير صالح لحساب الربح. الدخول: {entry}, الهدف: {target}. تم تخطي الإشارة.")

                    except Exception as e:
                        logger.error(f"❌ [خطأ في المعالجة] حدث خطأ أثناء معالجة العملة {symbol}: {e}", exc_info=True)

                logger.info(f"✅ [نهاية الدفعة] انتهت معالجة الدفعة. انتظار قصير قبل الدفعة التالية...")
                time.sleep(10)

            logger.info("ℹ️ [نهاية الدورة] انتهت دورة المسح الكاملة لجميع الدفعات. في انتظار الدورة التالية..."); 
            time.sleep(60)

        except (KeyboardInterrupt, SystemExit): break
        except Exception as main_err:
            log_and_notify("error", f"خطأ غير متوقع في الحلقة الرئيسية: {main_err}", "SYSTEM"); time.sleep(120)

# ---------------------- واجهة برمجة تطبيقات Flask (تبقى كما هي) ----------------------
app = Flask(__name__)
CORS(app)

def get_fear_and_greed_index() -> Dict[str, Any]:
    classification_translation = {"Extreme Fear": "خوف شديد", "Fear": "خوف", "Neutral": "محايد", "Greed": "طمع", "Extreme Greed": "طمع شديد", "Error": "خطأ"}
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        response.raise_for_status()
        data = response.json()['data'][0]; original = data['value_classification']
        return {"value": int(data['value']), "classification": classification_translation.get(original, original)}
    except Exception as e:
        logger.error(f"❌ [مؤشر الخوف والطمع] فشل الاتصال بالـ API: {e}")
        return {"value": -1, "classification": classification_translation["Error"]}

@app.route('/')
def home():
    try:
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, 'index.html')
        with open(file_path, 'r', encoding='utf-8') as f: return render_template_string(f.read())
    except FileNotFoundError: return "<h1>ملف لوحة التحكم (index.html) غير موجود.</h1>", 404

@app.route('/api/market_status')
def get_market_status(): return jsonify({"btc_trend": get_btc_trend(), "fear_and_greed": get_fear_and_greed_index()})

@app.route('/api/stats')
def get_stats():
    if not check_db_connection() or not conn:
        return jsonify({"error": "فشل الاتصال بقاعدة البيانات"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT status, profit_percentage FROM signals;")
            all_signals = cur.fetchall()

        with signal_cache_lock:
            open_trades_count = len(open_signals_cache)

        closed_trades = [s for s in all_signals if s.get('status') != 'open' and s.get('profit_percentage') is not None]
        targets_hit_all_time = sum(1 for s in closed_trades if s.get('profit_percentage', 0) > 0)
        stops_hit_all_time = len(closed_trades) - targets_hit_all_time
        total_profit_pct = sum(s['profit_percentage'] for s in closed_trades)

        return jsonify({
            "open_trades_count": open_trades_count,
            "total_profit_pct": total_profit_pct,
            "targets_hit_all_time": targets_hit_all_time,
            "stops_hit_all_time": stops_hit_all_time,
            "total_closed_trades": len(closed_trades)
        })
    except Exception as e:
        logger.error(f"❌ [API إحصائيات] خطأ: {e}")
        return jsonify({"error": "تعذر جلب الإحصائيات"}), 500

@app.route('/api/signals')
def get_signals():
    if not check_db_connection() or not conn or not redis_client:
        return jsonify({"error": "فشل الاتصال بالخدمات الأساسية (DB أو Redis)"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals ORDER BY CASE WHEN status = 'open' THEN 0 ELSE 1 END, id DESC;")
            all_signals = cur.fetchall()
        
        open_symbols = [s['symbol'] for s in all_signals if s['status'] == 'open']
        
        current_prices = {}
        if open_symbols:
            prices_list = redis_client.hmget(REDIS_PRICES_HASH_NAME, open_symbols)
            current_prices = {symbol: float(price) if price else None for symbol, price in zip(open_symbols, prices_list)}

        for s in all_signals:
            if s.get('closed_at'):
                s['closed_at'] = s['closed_at'].isoformat()
            
            if s['status'] == 'open':
                current_price = current_prices.get(s['symbol'])
                s['current_price'] = current_price
                if current_price and s.get('entry_price') and s['entry_price'] > 0:
                    pnl = ((current_price / s['entry_price']) - 1) * 100
                    s['pnl_pct'] = pnl
                else:
                    s['pnl_pct'] = 0
                    
        return jsonify(all_signals)
    except Exception as e:
        logger.error(f"❌ [API إشارات] خطأ: {e}")
        return jsonify({"error": "تعذر جلب الإشارات"}), 500

@app.route('/api/close/<int:signal_id>', methods=['POST'])
def manual_close_signal(signal_id):
    if not redis_client or not client:
        return jsonify({"error": "الخدمات الأساسية غير متاحة (Redis or Binance Client)"}), 500
    
    logger.info(f"ℹ️ [API إغلاق] تم استلام طلب إغلاق يدوي للإشارة ID: {signal_id}")
    
    with closure_lock:
        if signal_id in signals_pending_closure:
            return jsonify({"error": "الإشارة قيد الإغلاق حالياً."}), 409
    
    if not check_db_connection() or not conn:
        return jsonify({"error": "فشل الاتصال بقاعدة البيانات"}), 500
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE id = %s AND status = 'open';", (signal_id,))
            signal_to_close = cur.fetchone()
    except Exception as e:
        logger.error(f"❌ [API إغلاق] خطأ في البحث عن الإشارة {signal_id} في قاعدة البيانات: {e}")
        return jsonify({"error": "خطأ في قاعدة البيانات"}), 500

    if not signal_to_close:
        return jsonify({"error": "لم يتم العثور على الإشارة أو أنها ليست مفتوحة."}), 404
        
    symbol_to_close = signal_to_close['symbol']
    closing_price = None
    try:
        ticker = client.get_symbol_ticker(symbol=symbol_to_close)
        closing_price = float(ticker['price'])
    except Exception as e:
         return jsonify({"error": f"تعذر الحصول على السعر الحالي لـ {symbol_to_close} من API: {e}"}), 500
    
    initiate_signal_closure(symbol_to_close, dict(signal_to_close), 'manual_close', closing_price)
    
    return jsonify({"message": f"جاري إغلاق الإشارة {signal_id} لـ {symbol_to_close}."})

@app.route('/api/notifications')
def get_notifications():
    with notifications_lock: return jsonify(list(notifications_cache))

def run_flask():
    host, port = "0.0.0.0", int(os.environ.get('PORT', 10000))
    log_and_notify("info", f"بدء تشغيل لوحة التحكم على {host}:{port}", "SYSTEM")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        logger.warning("⚠️ [Flask] مكتبة 'waitress' غير موجودة, سيتم استخدام خادم التطوير.")
        app.run(host=host, port=port)

# ---------------------- نقطة انطلاق البرنامج ----------------------
def initialize_bot_services():
    global client, validated_symbols_to_scan
    logger.info("🤖 [خدمات البوت] بدء التهيئة في الخلفية...")
    try:
        client = Client(API_KEY, API_SECRET)
        logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
        
        init_db()
        init_redis()
        
        load_open_signals_to_cache(); load_notifications_to_cache()
        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan:
            logger.critical("❌ لا توجد رموز معتمدة للمسح. الحلقات لن تبدأ."); return
        
        Thread(target=run_websocket_manager, daemon=True).start()
        Thread(target=trade_monitoring_loop, daemon=True).start()
        Thread(target=main_loop, daemon=True).start()
        
        logger.info("✅ [خدمات البوت] تم بدء جميع خدمات الخلفية بنجاح.")
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حاسم أثناء التهيئة: {e}", "SYSTEM")
        exit(1)

if __name__ == "__main__":
    logger.info(f"🚀 بدء تشغيل بوت التداول - إصدار {BASE_ML_MODEL_NAME} (مع دعم Redis وإدارة الذاكرة)...")
    initialization_thread = Thread(target=initialize_bot_services, daemon=True)
    initialization_thread.start()
    run_flask()
    logger.info("👋 [إيقاف] تم إيقاف تشغيل البوت."); os._exit(0)
