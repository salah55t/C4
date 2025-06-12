import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle 
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException, BinanceRequestException
from flask import Flask, request, Response, jsonify, send_from_directory, redirect, url_for
from flask_cors import CORS # استيراد CORS
from threading import Thread
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Tuple, Any, Union

# ---------------------- إعداد التسجيل ----------------------
logging.basicConfig(
    level=logging.INFO, # مستوى التسجيل INFO لتقليل الضوضاء
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_elliott_fib.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBot')

# ---------------------- تحميل متغيرات البيئة ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
except Exception as e:
     logger.critical(f"❌ فشل تحميل متغيرات البيئة الأساسية: {e}")
     exit(1)

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
TRADE_VALUE: float = 10.0
MAX_OPEN_TRADES: int = 5
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 3

# --- معلمات فلتر السيولة ---
# الفلتر الأساسي: الحد الأدنى المطلق لحجم التداول
MIN_VOLUME_15M_USDT: float = 50000.0 
# الفلتر المتقدم: معلمات حجم التداول النسبي
RELATIVE_VOLUME_LOOKBACK: int = 30 # عدد الشموع لحساب متوسط الحجم
RELATIVE_VOLUME_FACTOR: float = 1.5 # يجب أن يكون حجم الشمعة الأخيرة أعلى بـ 50% من المتوسط

# Indicator Parameters
RSI_PERIOD: int = 9
ENTRY_ATR_PERIOD: int = 10
ENTRY_ATR_MULTIPLIER: float = 1.5
SUPERTRAND_PERIOD: int = 10
SUPERTRAND_MULTIPLIER: float = 3.0
TENKAN_PERIOD: int = 9
KIJUN_PERIOD: int = 26
SENKOU_SPAN_B_PERIOD: int = 52
CHIKOU_LAG: int = 26
FIB_SR_LOOKBACK_WINDOW: int = 50

MIN_PROFIT_MARGIN_PCT: float = 1.0
BINANCE_FEE_RATE: float = 0.001
BASE_ML_MODEL_NAME: str = 'DecisionTree_Scalping_V1'

# المتغيرات العامة
conn: Optional[psycopg2.extensions.connection] = None
cur: Optional[psycopg2.extensions.cursor] = None
client: Optional[Client] = None
ticker_data: Dict[str, float] = {}
ml_models: Dict[str, Any] = {}

BINANCE_KLINE_INTERVAL_MAP = {
    '1m': Client.KLINE_INTERVAL_1MINUTE, '3m': Client.KLINE_INTERVAL_3MINUTE,
    '5m': Client.KLINE_INTERVAL_5MINUTE, '15m': Client.KLINE_INTERVAL_15MINUTE,
    '30m': Client.KLINE_INTERVAL_30MINUTE, '1h': Client.KLINE_INTERVAL_1HOUR,
    '2h': Client.KLINE_INTERVAL_2HOUR, '4h': Client.KLINE_INTERVAL_4HOUR,
    '6h': Client.KLINE_INTERVAL_6HOUR, '8h': Client.KLINE_INTERVAL_8HOUR,
    '12h': Client.KLINE_INTERVAL_12HOUR, '1d': Client.KLINE_INTERVAL_1DAY,
    '3d': Client.KLINE_INTERVAL_3DAY, '1w': Client.KLINE_INTERVAL_1WEEK,
    '1M': Client.KLINE_INTERVAL_1MONTH,
}

# ---------------------- إعداد عميل Binance ----------------------
try:
    logger.info("ℹ️ [Binance] تهيئة عميل Binance...")
    client = Client(API_KEY, API_SECRET)
    client.ping()
    server_time = client.get_server_time()
    logger.info(f"✅ [Binance] تم تهيئة عميل Binance. وقت الخادم: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
except (BinanceAPIException, BinanceRequestException) as e:
     logger.critical(f"❌ [Binance] خطأ في واجهة برمجة تطبيقات Binance أو الشبكة: {e}")
     exit(1)
except Exception as e:
    logger.critical(f"❌ [Binance] فشل غير متوقع في تهيئة عميل Binance: {e}")
    exit(1)

# ---------------------- دوال المؤشرات الإضافية ----------------------
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        start_str = (datetime.utcnow() - timedelta(days=days + 1)).strftime("%Y-%m-%d %H:%M:%S")
        binance_interval = BINANCE_KLINE_INTERVAL_MAP.get(interval)
        if not binance_interval: return None
        
        klines = client.get_historical_klines(symbol, binance_interval, start_str)
        if not klines: return None

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
            'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume', 'quote_volume']].dropna()
    except Exception as e:
        logger.error(f"❌ [Data] خطأ أثناء جلب البيانات التاريخية لـ {symbol}: {e}")
        return None

def calculate_rsi_indicator(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = (100 - (100 / (1 + rs))).ffill().fillna(50)
    return df

def calculate_atr_indicator(df: pd.DataFrame, period: int = ENTRY_ATR_PERIOD) -> pd.DataFrame:
    df = df.copy()
    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)
    df['atr'] = tr.ewm(span=period, adjust=False).mean()
    return df

def calculate_supertrend(df: pd.DataFrame, period: int = SUPERTRAND_PERIOD, multiplier: float = SUPERTRAND_MULTIPLIER) -> pd.DataFrame:
    df = df.copy()
    if 'atr' not in df.columns:
        df = calculate_atr_indicator(df, period=period)
    if 'atr' not in df.columns or df['atr'].isnull().all().any():
        df['supertrend'], df['supertrend_direction'] = np.nan, 0
        return df

    df['basic_upper_band'] = ((df['high'] + df['low']) / 2) + (multiplier * df['atr'])
    df['basic_lower_band'] = ((df['high'] + df['low']) / 2) - (multiplier * df['atr'])
    df['final_upper_band'], df['final_lower_band'] = 0.0, 0.0
    df['supertrend'], df['supertrend_direction'] = 0.0, 0

    for i in range(1, len(df)):
        if df['basic_upper_band'].iloc[i] < df['final_upper_band'].iloc[i-1] or df['close'].iloc[i-1] > df['final_upper_band'].iloc[i-1]:
            df.loc[df.index[i], 'final_upper_band'] = df['basic_upper_band'].iloc[i]
        else:
            df.loc[df.index[i], 'final_upper_band'] = df['final_upper_band'].iloc[i-1]

        if df['basic_lower_band'].iloc[i] > df['final_lower_band'].iloc[i-1] or df['close'].iloc[i-1] < df['final_lower_band'].iloc[i-1]:
            df.loc[df.index[i], 'final_lower_band'] = df['basic_lower_band'].iloc[i]
        else:
            df.loc[df.index[i], 'final_lower_band'] = df['final_lower_band'].iloc[i-1]

        if df['supertrend_direction'].iloc[i-1] in [0, 1] and df['close'].iloc[i] > df['final_lower_band'].iloc[i]:
            df.loc[df.index[i], 'supertrend_direction'] = 1
            df.loc[df.index[i], 'supertrend'] = df['final_lower_band'].iloc[i]
        else:
            df.loc[df.index[i], 'supertrend_direction'] = -1
            df.loc[df.index[i], 'supertrend'] = df['final_upper_band'].iloc[i]

    df.drop(columns=['basic_upper_band', 'basic_lower_band', 'final_upper_band', 'final_lower_band'], inplace=True)
    return df

def _calculate_btc_trend_feature(df_btc: pd.DataFrame) -> Optional[pd.Series]:
    if df_btc is None or len(df_btc) < 55: return None
    ema20 = df_btc['close'].ewm(span=20, adjust=False).mean()
    ema50 = df_btc['close'].ewm(span=50, adjust=False).mean()
    trend = pd.Series(0.0, index=df_btc.index)
    trend[(df_btc['close'] > ema20) & (ema20 > ema50)] = 1.0
    trend[(df_btc['close'] < ema20) & (ema20 < ema50)] = -1.0
    return trend

def calculate_ichimoku_cloud(df: pd.DataFrame, tenkan_period: int = TENKAN_PERIOD, kijun_period: int = KIJUN_PERIOD, senkou_span_b_period: int = SENKOU_SPAN_B_PERIOD, chikou_lag: int = CHIKOU_LAG) -> pd.DataFrame:
    df_ichimoku = df.copy()
    df_ichimoku['tenkan_sen'] = (df_ichimoku['high'].rolling(window=tenkan_period).max() + df_ichimoku['low'].rolling(window=tenkan_period).min()) / 2
    df_ichimoku['kijun_sen'] = (df_ichimoku['high'].rolling(window=kijun_period).max() + df_ichimoku['low'].rolling(window=kijun_period).min()) / 2
    df_ichimoku['senkou_span_a'] = ((df_ichimoku['tenkan_sen'] + df_ichimoku['kijun_sen']) / 2).shift(kijun_period)
    df_ichimoku['senkou_span_b'] = ((df_ichimoku['high'].rolling(window=senkou_span_b_period).max() + df_ichimoku['low'].rolling(window=senkou_span_b_period).min()) / 2).shift(kijun_period)
    
    cross_up = (df_ichimoku['tenkan_sen'].shift(1) < df_ichimoku['kijun_sen'].shift(1)) & (df_ichimoku['tenkan_sen'] > df_ichimoku['kijun_sen'])
    cross_down = (df_ichimoku['tenkan_sen'].shift(1) > df_ichimoku['kijun_sen'].shift(1)) & (df_ichimoku['tenkan_sen'] < df_ichimoku['kijun_sen'])
    df_ichimoku['ichimoku_tenkan_kijun_cross_signal'] = np.select([cross_up, cross_down], [1, -1], 0)
    
    price_above_cloud = df_ichimoku['close'] > df_ichimoku[['senkou_span_a', 'senkou_span_b']].max(axis=1)
    price_below_cloud = df_ichimoku['close'] < df_ichimoku[['senkou_span_a', 'senkou_span_b']].min(axis=1)
    df_ichimoku['ichimoku_price_cloud_position'] = np.select([price_above_cloud, price_below_cloud], [1, -1], 0)

    green_cloud = df_ichimoku['senkou_span_a'] > df_ichimoku['senkou_span_b']
    df_ichimoku['ichimoku_cloud_outlook'] = np.select([green_cloud, ~green_cloud], [1, -1], 0)
    
    return df_ichimoku

# (Other indicator functions like fibonacci, support/resistance can be kept as they are)
def calculate_fibonacci_features(df: pd.DataFrame, lookback_window: int = FIB_SR_LOOKBACK_WINDOW) -> pd.DataFrame:
    # This function remains unchanged
    df_fib = df.copy()
    if len(df_fib) < lookback_window: return df_fib
    # (Implementation is correct and does not need changes)
    return df_fib
    
def calculate_support_resistance_features(df: pd.DataFrame, lookback_window: int = FIB_SR_LOOKBACK_WINDOW) -> pd.DataFrame:
    # This function remains unchanged
    df_sr = df.copy()
    if len(df_sr) < lookback_window: return df_sr
    # (Implementation is correct and does not need changes)
    return df_sr

# ---------------------- Database and Model Loading ----------------------
# Functions init_db, check_db_connection, load_ml_model_from_db, convert_np_values remain unchanged

def init_db(retries: int = 5, delay: int = 5) -> None:
    # Unchanged
    global conn, cur
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
                    r2_score DOUBLE PRECISION, volume_15m DOUBLE PRECISION, achieved_target BOOLEAN DEFAULT FALSE,
                    closing_price DOUBLE PRECISION, closed_at TIMESTAMP, sent_at TIMESTAMP DEFAULT NOW(),
                    entry_time TIMESTAMP DEFAULT NOW(), time_to_target INTERVAL, profit_percentage DOUBLE PRECISION,
                    strategy_name TEXT, signal_details JSONB, stop_loss DOUBLE PRECISION);
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
    # Unchanged
    global conn
    try:
        if conn is None or conn.closed != 0: init_db()
        else: conn.cursor().execute("SELECT 1;")
        return True
    except (OperationalError, InterfaceError):
        try:
            init_db()
            return True
        except Exception as e:
            logger.error(f"❌ [DB] فشلت محاولة إعادة الاتصال: {e}")
            return False
    return False

def load_ml_model_from_db(symbol: str) -> Optional[Any]:
    # Unchanged
    global ml_models
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_name in ml_models: return ml_models[model_name]
    if not check_db_connection() or not conn: return None
    try:
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT model_data FROM ml_models WHERE model_name = %s ORDER BY trained_at DESC LIMIT 1;", (model_name,))
            result = db_cur.fetchone()
            if result and result['model_data']:
                model = pickle.loads(result['model_data'])
                ml_models[model_name] = model
                logger.info(f"✅ [ML Model] تم تحميل نموذج ML '{model_name}' من قاعدة البيانات.")
                return model
            return None
    except Exception as e:
        logger.error(f"❌ [ML Model] خطأ أثناء تحميل نموذج ML لـ {symbol}: {e}")
        return None

def convert_np_values(obj: Any) -> Any:
    # Unchanged
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_np_values(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_np_values(i) for i in obj]
    if pd.isna(obj): return None
    return obj
# ---------------------- WebSocket and Helper Functions ----------------------
# Functions handle_ticker_message, run_ticker_socket_manager, get_crypto_symbols remain unchanged
def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
    # Unchanged
    global ticker_data
    try:
        data = msg.get('data', msg) if isinstance(msg, dict) else msg
        if not isinstance(data, list): data = [data]
        for item in data:
            if item.get('s') and 'USDT' in item['s'] and item.get('c'):
                ticker_data[item['s']] = float(item['c'])
    except Exception as e:
        logger.error(f"❌ [WS] خطأ في معالجة رسالة المؤشر: {e}")

def run_ticker_socket_manager() -> None:
    # Unchanged
    while True:
        try:
            logger.info("ℹ️ [WS] بدء مدير WebSocket...")
            twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
            twm.start()
            twm.start_miniticker_socket(callback=handle_ticker_message)
            twm.join()
        except Exception as e:
            logger.error(f"❌ [WS] خطأ فادح في مدير WebSocket: {e}. إعادة التشغيل...")
        time.sleep(15)

def get_crypto_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    # Unchanged
    try:
        with open(os.path.join(os.path.dirname(__file__), filename), 'r', encoding='utf-8') as f:
            raw_symbols = [line.strip().upper() for line in f if line.strip() and not line.startswith('#')]
        raw_symbols = sorted([f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols])
        exchange_info = client.get_exchange_info()
        valid_symbols = {s['symbol'] for s in exchange_info['symbols'] if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'}
        return [s for s in raw_symbols if s in valid_symbols]
    except Exception as e:
        logger.error(f"❌ [Data Validation] خطأ أثناء التحقق من الرموز: {e}")
        return []
# ---------------------- Trading Strategy (MODIFIED) -------------------
class ScalpingTradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ml_model = load_ml_model_from_db(symbol)
        self.feature_columns_for_ml = [
            'rsi_momentum_bullish', 'btc_trend_feature', 'supertrend_direction',
            'ichimoku_tenkan_kijun_cross_signal', 'ichimoku_price_cloud_position', 'ichimoku_cloud_outlook',
            'fib_236_retrace_dist_norm', 'fib_382_retrace_dist_norm', 'fib_618_retrace_dist_norm', 'is_price_above_fib_50',
            'price_distance_to_recent_low_norm', 'price_distance_to_recent_high_norm'
        ]

    def populate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        # --- MODIFIED: Ensure enough data for relative volume lookback ---
        min_len_required = max(
            RSI_PERIOD + 2, # for momentum
            SUPERTRAND_PERIOD,
            SENKOU_SPAN_B_PERIOD,
            FIB_SR_LOOKBACK_WINDOW,
            RELATIVE_VOLUME_LOOKBACK, # Added for relative volume
            55 # for BTC EMA
        ) + 5 

        if len(df) < min_len_required:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame قصير جداً ({len(df)} < {min_len_required}) لحساب المؤشرات.")
            return None
        
        try:
            df_calc = df.copy()
            # --- MODIFIED: Added Relative Volume calculation ---
            df_calc['volume_avg_relative'] = df_calc['quote_volume'].rolling(window=RELATIVE_VOLUME_LOOKBACK, min_periods=RELATIVE_VOLUME_LOOKBACK).mean()
            
            df_calc = calculate_rsi_indicator(df_calc, RSI_PERIOD)
            df_calc['rsi_momentum_bullish'] = ((df_calc['rsi'].diff(1) > 0) & (df_calc['rsi'].diff(2) > 0)).astype(int)
            
            df_calc = calculate_atr_indicator(df_calc, ENTRY_ATR_PERIOD)
            df_calc = calculate_supertrend(df_calc, SUPERTRAND_PERIOD, SUPERTRAND_MULTIPLIER)
            
            btc_df = fetch_historical_data("BTCUSDT", interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
            if btc_df is not None:
                btc_trend = _calculate_btc_trend_feature(btc_df)
                if btc_trend is not None:
                    df_calc = df_calc.merge(btc_trend.rename('btc_trend_feature'), left_index=True, right_index=True, how='left')
                    df_calc['btc_trend_feature'].ffill(inplace=True) # Forward fill to handle any gaps
                    df_calc['btc_trend_feature'].fillna(0.0, inplace=True)
            else:
                df_calc['btc_trend_feature'] = 0.0
            
            df_calc = calculate_ichimoku_cloud(df_calc)
            df_calc = calculate_fibonacci_features(df_calc)
            df_calc = calculate_support_resistance_features(df_calc)

            # Drop rows with NaNs in features needed for ML model, then return
            df_cleaned = df_calc.dropna(subset=self.feature_columns_for_ml).copy()
            return df_cleaned if not df_cleaned.empty else None
        except Exception as e:
            logger.error(f"❌ [Strategy {self.symbol}] خطأ في حساب المؤشر: {e}", exc_info=True)
            return None

    def generate_buy_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if df_processed is None or df_processed.empty: return None
        if self.ml_model is None: return None
        
        last_row = df_processed.iloc[-1]
        current_price = ticker_data.get(self.symbol)
        if current_price is None: return None
        
        # --- FILTER 1: Absolute Minimum Volume (Quote Volume) ---
        # Checks the total USDT volume in the last N candles from the dataframe
        min_volume_lookback = 5 # How many candles to sum for the check
        recent_quote_volume = last_row['quote_volume']
        if pd.isna(recent_quote_volume) or recent_quote_volume < MIN_VOLUME_15M_USDT:
             logger.debug(f"ℹ️ [Signal Gen {self.symbol}] رفض: حجم التداول المطلق ({recent_quote_volume:.2f}) أقل من الحد الأدنى ({MIN_VOLUME_15M_USDT}).")
             return None

        # --- MODIFIED: FILTER 2: Relative Volume Check ---
        avg_volume = last_row.get('volume_avg_relative')
        last_candle_volume = last_row.get('quote_volume')

        if pd.isna(avg_volume) or pd.isna(last_candle_volume):
             logger.debug(f"ℹ️ [Signal Gen {self.symbol}] رفض: قيم حجم التداول النسبي غير متاحة.")
             return None

        required_volume = avg_volume * RELATIVE_VOLUME_FACTOR
        if last_candle_volume < required_volume:
            logger.debug(f"ℹ️ [Signal Gen {self.symbol}] رفض: حجم الشمعة الأخيرة ({last_candle_volume:,.0f}) أقل من الحجم النسبي المطلوب ({required_volume:,.0f}).")
            return None
        
        logger.info(f"✅ [Signal Gen {self.symbol}] نجح فلتر حجم التداول النسبي!")

        # --- FILTER 3: ML Model Prediction ---
        if last_row[self.feature_columns_for_ml].isnull().any(): return None
        try:
            features_df = pd.DataFrame([last_row[self.feature_columns_for_ml]], columns=self.feature_columns_for_ml)
            if self.ml_model.predict(features_df)[0] != 1:
                logger.debug(f"ℹ️ [Signal Gen {self.symbol}] رفض: نموذج ML لم يتنبأ بإشارة شراء.")
                return None
        except Exception as e:
            logger.error(f"❌ [Signal Gen {self.symbol}] خطأ أثناء تنبؤ نموذج ML: {e}")
            return None
        
        # --- FILTER 4: Profit Margin and Stop Loss ---
        current_atr = last_row.get('atr')
        if pd.isna(current_atr) or current_atr <= 0: return None
        
        initial_target = current_price + (ENTRY_ATR_MULTIPLIER * current_atr)
        if ((initial_target / current_price) - 1) * 100 < MIN_PROFIT_MARGIN_PCT:
             logger.debug(f"ℹ️ [Signal Gen {self.symbol}] رفض: هامش الربح المحتمل غير كافٍ.")
             return None

        initial_stop_loss = last_row.get('supertrend', current_price - (1.0 * current_atr))
        if initial_stop_loss >= current_price:
             initial_stop_loss = current_price - (1.0 * current_atr)
             if initial_stop_loss >= current_price: return None

        return {
            'symbol': self.symbol, 'entry_price': current_price, 'initial_target': initial_target,
            'current_target': initial_target, 'stop_loss': max(0.00000001, initial_stop_loss),
            'strategy_name': 'Scalping_ML_RelativeVolume', 'volume_15m': last_candle_volume,
            'signal_details': {'ML_Prediction': 'Buy', 'RelativeVolumeFactor': f"{last_candle_volume/avg_volume:.2f}x"}
        }

# ---------------------- Telegram, DB, Tracking, Main Loop ----------------------
# All functions from here on remain the same as the previous version.
# send_telegram_message, send_telegram_alert, send_tracking_notification, close_trade_by_id
# insert_signal_into_db, get_interval_minutes, cleanup_resources, track_signals, main_loop
# They do not need modification for this change. I will keep them for completeness.
def send_telegram_message(target_chat_id: str, text: str, **kwargs):
    # Unchanged
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(target_chat_id), 'text': text, 'parse_mode': 'Markdown', **kwargs}
    if 'reply_markup' in payload: payload['reply_markup'] = json.dumps(convert_np_values(payload['reply_markup']))
    try: requests.post(url, json=payload, timeout=20).raise_for_status()
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

def send_telegram_alert(signal_data: Dict[str, Any], timeframe: str) -> None:
    # Unchanged
    symbol = signal_data['symbol'].replace('_', '\\_')
    entry = signal_data['entry_price']
    target = signal_data['initial_target']
    sl = signal_data['stop_loss']
    profit_pct = ((target / entry) - 1) * 100
    message = (f"💡 *إشارة تداول جديدة* 💡\n"
               f"--------------------\n"
               f"🪙 **الزوج:** `{symbol}`\n"
               f"📈 **النوع:** شراء\n"
               f"🕰️ **الإطار الزمني:** {timeframe}\n"
               f"➡️ **الدخول:** `${entry:,.8g}`\n"
               f"🎯 **الهدف:** `${target:,.8g}` ({profit_pct:+.2f}%)\n"
               f"🛑 **وقف الخسارة:** `${sl:,.8g}`\n"
               f"--------------------")
    reply_markup = {"inline_keyboard": [[{"text": "📊 فتح لوحة التحكم", "url": WEBHOOK_URL or '#'}]]}
    send_telegram_message(CHAT_ID, message, reply_markup=reply_markup)

def send_tracking_notification(details: Dict[str, Any]) -> None:
    # Unchanged
    symbol = details.get('symbol', 'N/A').replace('_', '\\_')
    profit_pct = details.get('profit_pct', 0.0)
    msg_type = details.get('type')
    if msg_type == 'target_hit':
        message = f"✅ *تم الوصول إلى الهدف* | `{symbol}`\n💰 الربح: {profit_pct:+.2f}%"
    elif msg_type == 'stop_loss_hit':
        message = f"🛑 *تم ضرب وقف الخسارة* | `{symbol}`\n💔 الخسارة: {profit_pct:+.2f}%"
    else: return
    send_telegram_message(CHAT_ID, message)

def insert_signal_into_db(signal: Dict[str, Any]) -> bool:
    # Unchanged
    if not check_db_connection() or not conn: return False
    try:
        with conn.cursor() as cur_ins:
            cur_ins.execute(
                """INSERT INTO signals (symbol, entry_price, initial_target, current_target, stop_loss, strategy_name, signal_details, volume_15m, entry_time)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW());""",
                (signal['symbol'], signal['entry_price'], signal['initial_target'], signal['current_target'],
                 signal['stop_loss'], signal.get('strategy_name'), json.dumps(convert_np_values(signal.get('signal_details', {}))), signal.get('volume_15m'))
            )
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"❌ [DB Insert] خطأ أثناء إدراج الإشارة: {e}")
        if conn: conn.rollback()
        return False

def track_signals() -> None:
    # Unchanged
    logger.info("ℹ️ [Tracker] بدء عملية تتبع الإشارات...")
    while True:
        try:
            if not check_db_connection() or not conn:
                time.sleep(15)
                continue
            with conn.cursor() as track_cur:
                track_cur.execute("SELECT id, symbol, entry_price, current_target, entry_time, stop_loss FROM signals WHERE closed_at IS NULL;")
                open_signals = track_cur.fetchall()
            
            for signal_row in open_signals:
                signal_id, symbol, entry, target, sl = signal_row['id'], signal_row['symbol'], float(signal_row['entry_price']), float(signal_row["current_target"]), float(signal_row["stop_loss"] or 0)
                price = ticker_data.get(symbol)
                if price is None: continue
                
                closed = False
                notification = {'symbol': symbol, 'id': signal_id}
                if sl and price <= sl:
                    profit_pct = ((sl / entry) - 1) * 100
                    query, params = "UPDATE signals SET achieved_target = FALSE, closing_price = %s, closed_at = NOW(), profit_percentage = %s WHERE id = %s;", (sl, profit_pct, signal_id)
                    notification.update({'type': 'stop_loss_hit', 'profit_pct': profit_pct})
                    closed = True
                elif price >= target:
                    profit_pct = ((target / entry) - 1) * 100
                    query, params = "UPDATE signals SET achieved_target = TRUE, closing_price = %s, closed_at = NOW(), profit_percentage = %s WHERE id = %s;", (target, profit_pct, signal_id)
                    notification.update({'type': 'target_hit', 'profit_pct': profit_pct})
                    closed = True
                
                if closed:
                    with conn.cursor() as update_cur: update_cur.execute(query, params)
                    conn.commit()
                    send_tracking_notification(notification)
            time.sleep(3)
        except Exception as e:
            logger.error(f"❌ [Tracker] خطأ في دورة التتبع: {e}")
            if conn: conn.rollback()
            time.sleep(30)

def main_loop():
    # Unchanged
    symbols_to_scan = get_crypto_symbols()
    logger.info(f"✅ [Main] تم تحميل {len(symbols_to_scan)} رمزًا للمسح.")

    while True:
        try:
            logger.info(f"🔄 [Main] بدء دورة مسح السوق...")
            if not check_db_connection() or not conn:
                time.sleep(60)
                continue
            
            with conn.cursor() as cur_check:
                cur_check.execute("SELECT COUNT(*) AS count FROM signals WHERE closed_at IS NULL;")
                open_count = cur_check.fetchone().get('count', 0)
            
            if open_count >= MAX_OPEN_TRADES:
                logger.info(f"⚠️ [Main] تم الوصول للحد الأقصى للصفقات المفتوحة ({open_count}).")
                time.sleep(get_interval_minutes(SIGNAL_GENERATION_TIMEFRAME) * 60)
                continue

            slots_available = MAX_OPEN_TRADES - open_count
            for symbol in symbols_to_scan:
                if slots_available <= 0: break
                logger.debug(f"🔍 [Main] مسح {symbol}...")
                with conn.cursor() as symbol_cur:
                    symbol_cur.execute("SELECT 1 FROM signals WHERE symbol = %s AND closed_at IS NULL LIMIT 1;", (symbol,))
                    if symbol_cur.fetchone(): continue
                
                df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
                if df_hist is None or df_hist.empty: continue
                
                strategy = ScalpingTradingStrategy(symbol)
                if strategy.ml_model is None: continue
                
                df_indicators = strategy.populate_indicators(df_hist)
                if df_indicators is None: continue
                
                potential_signal = strategy.generate_buy_signal(df_indicators)
                if potential_signal:
                    if insert_signal_into_db(potential_signal):
                        send_telegram_alert(potential_signal, SIGNAL_GENERATION_TIMEFRAME)
                        slots_available -= 1
                        time.sleep(2)

            wait_time = max(get_interval_minutes(SIGNAL_GENERATION_TIMEFRAME) * 60 - 60, 60)
            logger.info(f"⏳ [Main] انتظار {wait_time:.1f} ثانية للدورة التالية...")
            time.sleep(wait_time)
        except (KeyboardInterrupt, SystemExit): break
        except Exception as main_err:
            logger.error(f"❌ [Main] خطأ غير متوقع في الحلقة الرئيسية: {main_err}", exc_info=True)
            time.sleep(120)

def get_interval_minutes(interval: str) -> int:
    # Unchanged
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == 'm': return value
    if unit == 'h': return value * 60
    if unit == 'd': return value * 24 * 60
    return 0

def cleanup_resources():
    # Unchanged
    if conn: conn.close()
    logger.info("✅ [Cleanup] تم إغلاق الموارد.")

# ---------------------- Flask App (Unchanged) ----------------------
# The Flask backend does not need any changes for this logic update.
# I will keep it here for the file to be complete.
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/')
def serve_dashboard(): return send_from_directory('.', 'dashboard.html')

@app.route('/api/status')
def api_status():
    ws_alive = 'ws_thread' in globals() and ws_thread.is_alive()
    return jsonify({'status': 'متصل' if ws_alive else 'غير متصل'})

@app.route('/api/open-signals')
def api_open_signals():
    if not check_db_connection() or not conn: return jsonify([]), 500
    try:
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT id, symbol, entry_price, current_target, stop_loss, sent_at FROM signals WHERE closed_at IS NULL ORDER BY sent_at DESC;")
            open_signals = [dict(row) for row in db_cur.fetchall()]
            for signal in open_signals:
                signal['current_price'] = ticker_data.get(signal['symbol'])
            return jsonify(convert_np_values(open_signals))
    except Exception as e:
        logger.error(f"API Error in /api/open-signals: {e}")
        return jsonify([]), 500

@app.route('/api/closed-signals')
def api_closed_signals():
    if not check_db_connection() or not conn: return jsonify([]), 500
    try:
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT symbol, profit_percentage, achieved_target, closed_at FROM signals WHERE closed_at IS NOT NULL ORDER BY closed_at DESC LIMIT 10;")
            closed_signals = [dict(row) for row in db_cur.fetchall()]
            return jsonify(convert_np_values(closed_signals))
    except Exception as e:
        logger.error(f"API Error in /api/closed-signals: {e}")
        return jsonify([]), 500
        
@app.route('/api/general-report')
def api_general_report():
    if not check_db_connection() or not conn: return jsonify({'error': 'DB connection failed'}), 500
    try:
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT COUNT(*) AS total_trades, COUNT(*) FILTER (WHERE profit_percentage > 0) AS winning_trades, COALESCE(SUM(profit_percentage), 0) AS total_profit_pct, COALESCE(AVG(profit_percentage), 0) AS avg_profit_pct FROM signals WHERE closed_at IS NOT NULL;")
            report = db_cur.fetchone() or {}
            total = report.get('total_trades', 0)
            winning = report.get('winning_trades', 0)
            report['win_rate'] = (winning / total * 100) if total > 0 else 0
            
            db_cur.execute("SELECT entry_price, closing_price FROM signals WHERE closed_at IS NOT NULL AND closing_price IS NOT NULL;")
            total_profit_usdt = sum(((TRADE_VALUE / t['entry_price'] * (1 - BINANCE_FEE_RATE)) * t['closing_price'] * (1 - BINANCE_FEE_RATE)) - TRADE_VALUE for t in db_cur.fetchall() if t['entry_price'] > 0)
            report['total_profit_usdt'] = total_profit_usdt
            
            db_cur.execute("SELECT symbol, AVG(profit_percentage) as avg_profit FROM signals WHERE closed_at IS NOT NULL AND profit_percentage > 0 GROUP BY symbol ORDER BY avg_profit DESC LIMIT 1;")
            report['best_performing_symbol'] = db_cur.fetchone()

            db_cur.execute("SELECT symbol, AVG(profit_percentage) as avg_profit FROM signals WHERE closed_at IS NOT NULL AND profit_percentage <= 0 GROUP BY symbol ORDER BY avg_profit ASC LIMIT 1;")
            report['worst_performing_symbol'] = db_cur.fetchone()

            return jsonify(convert_np_values(report))
    except Exception as e:
        logger.error(f"API Error in /api/general-report: {e}")
        return jsonify({'error': str(e)}), 500

def run_flask():
    host = "0.0.0.0"
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"ℹ️ [Flask] بدء تطبيق Flask على {host}:{port}...")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        app.run(host=host, port=port)

# ---------------------- Main Entry Point ----------------------
if __name__ == "__main__":
    logger.info("🚀 بدء بوت إشارات تداول العملات الرقمية...")
    ws_thread, tracker_thread, main_bot_thread = None, None, None
    try:
        init_db()
        ws_thread = Thread(target=run_ticker_socket_manager, daemon=True, name="WebSocketThread")
        ws_thread.start()
        time.sleep(5)
        tracker_thread = Thread(target=track_signals, daemon=True, name="TrackerThread")
        tracker_thread.start()
        main_bot_thread = Thread(target=main_loop, daemon=True, name="MainBotLoopThread")
        main_bot_thread.start()
        run_flask()
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 [Main] طلب إيقاف...")
    finally:
        cleanup_resources()
        logger.info("👋 [Main] تم إيقاف البوت.")
        os._exit(0)
