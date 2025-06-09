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
from flask import Flask, request, Response
from threading import Thread
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Tuple, Any, Union

# ---------------------- إعداد التسجيل ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',\
    handlers=[
        logging.FileHandler('crypto_bot_enhanced.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotEnhanced')

# ---------------------- تحميل المتغيرات البيئية ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
except Exception as e:
     logger.critical(f"❌ فشل في تحميل المتغيرات البيئية الأساسية: {e}")
     exit(1)

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
TRADE_VALUE: float = 10.0
MAX_OPEN_TRADES: int = 5
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 5 # Increased lookback for more context

# Indicator Parameters
RSI_PERIOD: int = 14
VOLUME_LOOKBACK_CANDLES: int = 2
RSI_MOMENTUM_LOOKBACK_CANDLES: int = 3
ENTRY_ATR_PERIOD: int = 14
ENTRY_ATR_MULTIPLIER: float = 2.0
SUPERTREND_PERIOD: int = 10
SUPERTREND_MULTIPLIER: float = 3.0
MACD_FAST_PERIOD: int = 12
MACD_SLOW_PERIOD: int = 26
MACD_SIGNAL_PERIOD: int = 9
BB_PERIOD: int = 20
BB_STD_DEV: int = 2
ADX_PERIOD: int = 14

MIN_PROFIT_MARGIN_PCT: float = 1.0
MIN_VOLUME_15M_USDT: float = 75000.0
TARGET_APPROACH_THRESHOLD_PCT: float = 0.005
BINANCE_FEE_RATE: float = 0.001

BASE_ML_MODEL_NAME: str = 'DecisionTree_Scalping_V2_Enhanced'

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
cur: Optional[psycopg2.extensions.cursor] = None
client: Optional[Client] = None
ticker_data: Dict[str, float] = {}
ml_models: Dict[str, Any] = {}

# ---------------------- Binance Client Setup ----------------------
try:
    logger.info("ℹ️ [Binance] تهيئة عميل Binance...")
    client = Client(API_KEY, API_SECRET)
    client.ping()
    server_time = client.get_server_time()
    logger.info(f"✅ [Binance] تم تهيئة عميل Binance. وقت الخادم: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
except (BinanceAPIException, BinanceRequestException) as e:
    logger.critical(f"❌ [Binance] فشل في تهيئة عميل Binance: {e}")
    exit(1)

# ---------------------- Indicator Functions (including new ones) ----------------------

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client:
        logger.error("❌ [Data] عميل Binance غير مهيأ لجلب البيانات.")
        return None
    try:
        start_str_overall = (datetime.utcnow() - timedelta(days=days + 1)).strftime("%Y-%m-%d %H:%M:%S")
        logger.debug(f"ℹ️ [Data] جلب بيانات {interval} لـ {symbol} من {start_str_overall}...")
        
        interval_map = {
            '15m': Client.KLINE_INTERVAL_15MINUTE, '5m': Client.KLINE_INTERVAL_5MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR, '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY
        }
        binance_interval = interval_map.get(interval)
        if not binance_interval:
            logger.error(f"❌ [Data] فترة زمنية غير مدعومة: {interval}")
            return None

        klines = client.get_historical_klines(symbol, binance_interval, start_str_overall)
        if not klines:
            logger.warning(f"⚠️ [Data] لا توجد بيانات تاريخية ({interval}) لـ {symbol}.")
            return None

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[numeric_cols].dropna()
        df.sort_index(inplace=True)
        
        logger.debug(f"✅ [Data] تم جلب ومعالجة {len(df)} شمعة تاريخية ({interval}) لـ {symbol}.")
        return df
    except (BinanceAPIException, BinanceRequestException) as e:
        logger.error(f"❌ [Data] خطأ Binance أثناء جلب البيانات لـ {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ [Data] خطأ غير متوقع أثناء جلب البيانات التاريخية لـ {symbol}: {e}", exc_info=True)
        return None

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi_indicator(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)
    return df

def calculate_atr_indicator(df: pd.DataFrame, period: int = ENTRY_ATR_PERIOD) -> pd.DataFrame:
    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)
    df['atr'] = tr.ewm(span=period, adjust=False).mean()
    return df

def calculate_supertrend(df: pd.DataFrame, period: int = SUPERTREND_PERIOD, multiplier: float = SUPERTREND_MULTIPLIER) -> pd.DataFrame:
    if 'atr' not in df.columns: df = calculate_atr_indicator(df, period)
    hl2 = (df['high'] + df['low']) / 2
    df['upper_band'] = hl2 + (multiplier * df['atr'])
    df['lower_band'] = hl2 - (multiplier * df['atr'])
    df['in_uptrend'] = True
    for current in range(1, len(df.index)):
        previous = current - 1
        if df['close'].iloc[current] > df['upper_band'].iloc[previous]: df.loc[df.index[current], 'in_uptrend'] = True
        elif df['close'].iloc[current] < df['lower_band'].iloc[previous]: df.loc[df.index[current], 'in_uptrend'] = False
        else:
            df.loc[df.index[current], 'in_uptrend'] = df['in_uptrend'].iloc[previous]
            if df['in_uptrend'].iloc[current] and df['lower_band'].iloc[current] < df['lower_band'].iloc[previous]: df.loc[df.index[current], 'lower_band'] = df['lower_band'].iloc[previous]
            if not df['in_uptrend'].iloc[current] and df['upper_band'].iloc[current] > df['upper_band'].iloc[previous]: df.loc[df.index[current], 'upper_band'] = df['upper_band'].iloc[previous]
    df['supertrend'] = np.where(df['in_uptrend'], df['lower_band'], df['upper_band'])
    df['supertrend_direction'] = np.where(df['in_uptrend'], 1, -1)
    df.drop(['upper_band', 'lower_band', 'in_uptrend'], axis=1, inplace=True, errors='ignore')
    return df

def _calculate_btc_trend_feature(df_btc: pd.DataFrame) -> Optional[pd.Series]:
    min_data_for_ema = 55
    if df_btc is None or len(df_btc) < min_data_for_ema: return pd.Series(index=df_btc.index if df_btc is not None else None, data=0.0)
    ema20 = calculate_ema(df_btc['close'], 20)
    ema50 = calculate_ema(df_btc['close'], 50)
    trend_series = pd.Series(index=ema20.index, data=0.0)
    trend_series[(df_btc['close'] > ema20) & (ema20 > ema50)] = 1.0
    trend_series[(df_btc['close'] < ema20) & (ema20 < ema50)] = -1.0
    return trend_series.reindex(df_btc.index).fillna(0.0)

def calculate_macd(df: pd.DataFrame, fast_period: int = MACD_FAST_PERIOD, slow_period: int = MACD_SLOW_PERIOD, signal_period: int = MACD_SIGNAL_PERIOD) -> pd.DataFrame:
    df['ema_fast'] = calculate_ema(df['close'], span=fast_period)
    df['ema_slow'] = calculate_ema(df['close'], span=slow_period)
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = calculate_ema(df['macd'], span=signal_period)
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df.drop(['ema_fast', 'ema_slow'], axis=1, inplace=True, errors='ignore')
    return df

def calculate_bollinger_bands(df: pd.DataFrame, period: int = BB_PERIOD, std_dev: int = BB_STD_DEV) -> pd.DataFrame:
    df['bb_ma'] = df['close'].rolling(window=period).mean()
    df['bb_std'] = df['close'].rolling(window=period).std()
    df['bb_upper'] = df['bb_ma'] + (df['bb_std'] * std_dev)
    df['bb_lower'] = df['bb_ma'] - (df['bb_std'] * std_dev)
    df.drop(['bb_ma', 'bb_std'], axis=1, inplace=True, errors='ignore')
    return df

def calculate_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    if 'atr' not in df.columns: df = calculate_atr_indicator(df, period)
    df['plus_dm'] = df['high'].diff()
    df['minus_dm'] = df['low'].diff().mul(-1)
    df['plus_dm'] = np.where((df['plus_dm'] > df['minus_dm']) & (df['plus_dm'] > 0), df['plus_dm'], 0)
    df['minus_dm'] = np.where((df['minus_dm'] > df['plus_dm']) & (df['minus_dm'] > 0), df['minus_dm'], 0)
    df['plus_di'] = 100 * (df['plus_dm'].ewm(alpha=1/period, adjust=False).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].ewm(alpha=1/period, adjust=False).mean() / df['atr'])
    df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']).replace(0, 1))
    df['adx'] = df['dx'].ewm(alpha=1/period, adjust=False).mean()
    df.drop(['plus_dm', 'minus_dm', 'plus_di', 'minus_di', 'dx'], axis=1, inplace=True, errors='ignore')
    return df

# ---------------------- Database Connection Setup ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    global conn, cur
    logger.info("[DB] تهيئة قاعدة البيانات...")
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            cur = conn.cursor()
            logger.info("✅ [DB] تم الاتصال بقاعدة البيانات بنجاح.")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION NOT NULL,
                    initial_target DOUBLE PRECISION NOT NULL, current_target DOUBLE PRECISION NOT NULL,
                    r2_score DOUBLE PRECISION, volume_15m DOUBLE PRECISION, achieved_target BOOLEAN DEFAULT FALSE,
                    closing_price DOUBLE PRECISION, closed_at TIMESTAMP, sent_at TIMESTAMP DEFAULT NOW(),
                    entry_time TIMESTAMP DEFAULT NOW(), time_to_target INTERVAL, profit_percentage DOUBLE PRECISION,
                    strategy_name TEXT, signal_details JSONB, stop_loss DOUBLE PRECISION
                );
                CREATE TABLE IF NOT EXISTS ml_models (
                    id SERIAL PRIMARY KEY, model_name TEXT NOT NULL UNIQUE, model_data BYTEA NOT NULL,
                    trained_at TIMESTAMP DEFAULT NOW(), metrics JSONB
                );
            """)
            conn.commit()
            logger.info("✅ [DB] جداول 'signals' و 'ml_models' جاهزة.")
            return
        except (OperationalError, Exception) as e:
            logger.error(f"❌ [DB] فشلت محاولة الاتصال {attempt + 1}: {e}")
            if conn: conn.rollback()
            if attempt == retries - 1:
                 logger.critical("❌ [DB] فشلت جميع محاولات الاتصال بقاعدة البيانات.")
                 raise e
            time.sleep(delay)

def check_db_connection() -> bool:
    global conn, cur
    try:
        if conn is None or conn.closed != 0:
            logger.warning("⚠️ [DB] الاتصال مغلق أو غير موجود. إعادة التهيئة...")
            init_db()
        else:
             with conn.cursor() as check_cur: check_cur.execute("SELECT 1;")
        return True
    except (OperationalError, InterfaceError, Exception) as e:
        logger.error(f"❌ [DB] فقدان الاتصال بقاعدة البيانات ({e}). محاولة إعادة الاتصال...")
        try:
             init_db()
             return True
        except Exception as recon_err:
            logger.error(f"❌ [DB] فشلت محاولة إعادة الاتصال: {recon_err}")
            return False

def load_ml_model_from_db(symbol: str) -> Optional[Any]:
    global ml_models
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_name in ml_models: return ml_models[model_name]
    if not check_db_connection() or not conn:
        logger.error(f"❌ [ML Model] لا يمكن تحميل النموذج لـ {symbol}، مشكلة في اتصال قاعدة البيانات.")
        return None
    try:
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT model_data FROM ml_models WHERE model_name = %s ORDER BY trained_at DESC LIMIT 1;", (model_name,))
            result = db_cur.fetchone()
            if result and result['model_data']:
                model_bundle = pickle.loads(result['model_data'])
                ml_models[model_name] = model_bundle # Store the tuple (model, scaler)
                logger.info(f"✅ [ML Model] تم تحميل النموذج '{model_name}' من قاعدة البيانات.")
                return model_bundle
            else:
                logger.warning(f"⚠️ [ML Model] النموذج '{model_name}' غير موجود في قاعدة البيانات.")
                return None
    except (psycopg2.Error, pickle.UnpicklingError, Exception) as e:
        logger.error(f"❌ [ML Model] فشل تحميل النموذج لـ {symbol}: {e}", exc_info=True)
        return None

def convert_np_values(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int_)): return int(obj)
    if isinstance(obj, (np.floating, np.float_)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_np_values(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_np_values(item) for item in obj]
    if pd.isna(obj): return None
    return obj

# ---------------------- WebSocket & Helpers ----------------------
def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]):
    global ticker_data
    try:
        data_list = msg if isinstance(msg, list) else msg.get('data', [])
        for item in data_list:
            symbol = item.get('s')
            price_str = item.get('c')
            if symbol and 'USDT' in symbol and price_str:
                ticker_data[symbol] = float(price_str)
    except (ValueError, Exception) as e:
        logger.error(f"❌ [WS] خطأ في معالجة رسالة التيكر: {e}", exc_info=True)

def run_ticker_socket_manager():
    while True:
        try:
            logger.info("ℹ️ [WS] بدء إدارة WebSocket...")
            twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
            twm.start()
            stream_name = twm.start_miniticker_socket(callback=handle_ticker_message)
            logger.info(f"✅ [WS] تم بدء بث WebSocket: {stream_name}")
            twm.join()
            logger.warning("⚠️ [WS] توقفت إدارة WebSocket. إعادة التشغيل...")
        except Exception as e:
            logger.error(f"❌ [WS] تعطلت إدارة WebSocket: {e}. إعادة التشغيل في 15 ثانية...", exc_info=True)
        time.sleep(15)

def fetch_recent_volume(symbol: str) -> float:
    if not client: return 0.0
    try:
        klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=VOLUME_LOOKBACK_CANDLES)
        return sum(float(k[7]) for k in klines) if klines else 0.0
    except (BinanceAPIException, BinanceRequestException, Exception) as e:
        logger.error(f"❌ [Volume] فشل جلب الحجم لـ {symbol}: {e}")
        return 0.0

# FIXED: More robust path handling
def get_crypto_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    raw_symbols: List[str] = []
    logger.info(f"ℹ️ [Symbols] قراءة قائمة الرموز من '{filename}'...")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)

        if not os.path.exists(file_path):
            file_path = os.path.abspath(filename)
            if not os.path.exists(file_path):
                 logger.error(f"❌ [Symbols] الملف '{filename}' غير موجود في دليل السكريبت أو الدليل الحالي.")
                 return []

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = [f"{line.strip().upper().replace('USDT', '')}USDT" for line in f if line.strip() and not line.startswith('#')]
        logger.info(f"ℹ️ [Symbols] تم قراءة {len(raw_symbols)} رمزًا مبدئيًا. جاري التحقق من Binance...")

        if not client:
            logger.error("❌ [Symbols] عميل Binance غير جاهز للتحقق من الرموز.")
            return raw_symbols

        exchange_info = client.get_exchange_info()
        valid_symbols = {s['symbol'] for s in exchange_info['symbols'] if s.get('status') == 'TRADING' and s.get('isSpotTradingAllowed')}
        validated_symbols = [s for s in raw_symbols if s in valid_symbols]
        
        removed_count = len(raw_symbols) - len(validated_symbols)
        if removed_count > 0:
            logger.warning(f"⚠️ [Symbols] تم إزالة {removed_count} رمزًا غير صالح للتداول.")
        
        logger.info(f"✅ [Symbols] تم التحقق من {len(validated_symbols)} رمزًا صالحًا للتداول.")
        return validated_symbols
    except Exception as e:
        logger.error(f"❌ [Symbols] فشل قراءة أو التحقق من الرموز من '{filename}': {e}", exc_info=True)
        return []

# ---------------------- Trading Strategy (ENHANCED) -------------------
class EnhancedTradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model_bundle = load_ml_model_from_db(symbol) # (model, scaler)
        if self.model_bundle is None:
            logger.warning(f"⚠️ [Strategy {self.symbol}] لم يتم تحميل نموذج تعلم الآلة. لن تتمكن الإستراتيجية من توليد إشارات.")

        self.feature_columns_for_ml = [
            'volume_15m_avg', 'rsi_momentum_bullish', 'btc_trend_feature', 
            'supertrend_direction', 'macd_hist', 'bb_upper_dist', 'bb_lower_dist', 'adx'
        ]

    def populate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        logger.debug(f"ℹ️ [Strategy {self.symbol}] حساب المؤشرات لنموذج ML...")
        try:
            df_calc = df.copy()
            df_calc = calculate_rsi_indicator(df_calc)
            df_calc = calculate_atr_indicator(df_calc)
            df_calc = calculate_supertrend(df_calc)
            df_calc = calculate_macd(df_calc)
            df_calc = calculate_bollinger_bands(df_calc)
            df_calc = calculate_adx(df_calc)
            df_calc['volume_15m_avg'] = df_calc['volume'].rolling(window=VOLUME_LOOKBACK_CANDLES).mean()
            df_calc['rsi_momentum_bullish'] = ((df_calc['rsi'].diff(RSI_MOMENTUM_LOOKBACK_CANDLES) > 0) & (df_calc['rsi'] > 50)).astype(int)
            df_calc['bb_upper_dist'] = (df_calc['bb_upper'] - df_calc['close']) / df_calc['close']
            df_calc['bb_lower_dist'] = (df_calc['close'] - df_calc['bb_lower']) / df_calc['close']

            btc_df = fetch_historical_data("BTCUSDT", interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
            btc_trend_series = _calculate_btc_trend_feature(btc_df)
            df_calc = df_calc.merge(btc_trend_series.rename('btc_trend_feature'), left_index=True, right_index=True, how='left')
            df_calc['btc_trend_feature'].fillna(0.0, inplace=True)

            for col in self.feature_columns_for_ml:
                if col not in df_calc.columns: df_calc[col] = np.nan
            
            df_cleaned = df_calc.dropna(subset=self.feature_columns_for_ml + ['atr']).copy()
            logger.debug(f"✅ [Strategy {self.symbol}] تم حساب المؤشرات. طول DataFrame النظيف: {len(df_cleaned)}")
            return df_cleaned
        except Exception as e:
            logger.error(f"❌ [Strategy {self.symbol}] خطأ أثناء حساب المؤشرات: {e}", exc_info=True)
            return None

    def generate_buy_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if df_processed is None or df_processed.empty or self.model_bundle is None: return None
        model, scaler = self.model_bundle
        
        last_row = df_processed.iloc[-1]
        current_price = ticker_data.get(self.symbol)
        if current_price is None:
            logger.warning(f"⚠️ [Strategy {self.symbol}] السعر الحالي غير متاح من التيكر.")
            return None

        signal_details = {}
        try:
            features_df = pd.DataFrame([last_row[self.feature_columns_for_ml]], columns=self.feature_columns_for_ml)
            features_scaled = scaler.transform(features_df)
            ml_pred = model.predict(features_scaled)[0]
            ml_is_bullish = ml_pred == 1
            signal_details['ML_Prediction'] = 'Bullish ✅' if ml_is_bullish else 'Bearish ❌'
        except Exception as ml_err:
            logger.error(f"❌ [Strategy {self.symbol}] خطأ في تنبؤ ML: {ml_err}")
            return None

        if not ml_is_bullish:
            logger.debug(f"ℹ️ [Strategy {self.symbol}] تم رفض الإشارة: تنبؤ ML ليس صعوديًا.")
            return None

        supertrend_is_bullish = last_row.get('supertrend_direction') == 1
        macd_hist_positive_rising = last_row.get('macd_hist', -1) > 0 and last_row.get('macd_hist', -1) > df_processed['macd_hist'].iloc[-2]
        adx_strong_trend = last_row.get('adx', 0) > 25
        volume_recent = fetch_recent_volume(self.symbol)
        volume_ok = volume_recent >= MIN_VOLUME_15M_USDT
        current_atr = last_row.get('atr')
        if pd.isna(current_atr) or current_atr <= 0: return None
        initial_target = current_price + (ENTRY_ATR_MULTIPLIER * current_atr)
        initial_stop_loss = last_row.get('supertrend')
        profit_margin_pct = ((initial_target / current_price) - 1) * 100
        profit_margin_ok = profit_margin_pct >= MIN_PROFIT_MARGIN_PCT

        if not all([supertrend_is_bullish, macd_hist_positive_rising, adx_strong_trend, volume_ok, profit_margin_ok]):
            logger.debug(f"ℹ️ [Strategy {self.symbol}] تم رفض الإشارة: فشل في الفلاتر الإضافية.")
            return None

        signal_details.update({
            'Supertrend': 'Pass' if supertrend_is_bullish else 'Fail',
            'MACD': 'Pass' if macd_hist_positive_rising else 'Fail',
            'ADX': f"Pass ({last_row.get('adx',0):.1f})" if adx_strong_trend else 'Fail',
            'Volume': f"Pass ({volume_recent:,.0f})" if volume_ok else 'Fail',
            'Profit Margin': f"Pass ({profit_margin_pct:.2f}%)" if profit_margin_ok else 'Fail'
        })

        signal_output = {
            'symbol': self.symbol, 'entry_price': float(current_price),
            'initial_target': float(initial_target), 'current_target': float(initial_target),
            'stop_loss': float(initial_stop_loss), 'strategy_name': 'Scalping_ML_Enhanced_V2',
            'signal_details': signal_details, 'volume_15m': volume_recent
        }
        logger.info(f"✅ [Strategy {self.symbol}] تم تأكيد إشارة شراء محسنة. السعر: {current_price:.6f}, الهدف: {initial_target:.6f}, وقف الخسارة: {initial_stop_loss:.6f}")
        return signal_output

# ---------------------- Main Loop and Execution ----------------------
# (Placeholder for the rest of the c4.py logic: main_loop, Flask, tracking, etc.)
# You need to integrate the above classes and functions into your existing main execution flow.
def main_loop():
    logger.info("Starting main bot loop...")
    # This is where your original main_loop logic would go, using the new EnhancedTradingStrategy
    pass

if __name__ == "__main__":
    logger.info("🚀 بدء بوت التداول المحسن...")
    init_db()
    ws_thread = Thread(target=run_ticker_socket_manager, daemon=True, name="WebSocketThread")
    ws_thread.start()
    logger.info("✅ [Main] تم بدء إدارة WebSocket. انتظار 5 ثوانٍ للبيانات الأولية...")
    time.sleep(5)
    
    # Placeholder to run the main logic. You should integrate your existing tracker and flask threads here.
    main_loop()

    logger.info("👋 [Main] تم إيقاف البوت.")
