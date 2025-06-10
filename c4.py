import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle # Added for ML model deserialization
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

# ---------------------- Logging Setup ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_elliott_fib.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBot')

# ---------------------- Load Environment Variables ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
    # WEBHOOK_URL is optional, but Flask will always run for Render compatibility
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
except Exception as e:
     logger.critical(f"❌ فشل تحميل متغيرات البيئة الأساسية: {e}")
     exit(1)

logger.info(f"مفتاح API الخاص بـ Binance: {'متاح' if API_KEY else 'غير متاح'}")
logger.info(f"رمز Telegram: {TELEGRAM_TOKEN[:10]}...{'*' * (len(TELEGRAM_TOKEN)-10)}")
logger.info(f"معرف دردشة Telegram: {CHAT_ID}")
logger.info(f"عنوان URL لقاعدة البيانات: {'متاح' if DB_URL else 'غير متاح'}")
logger.info(f"عنوان URL للخطاف الويب: {WEBHOOK_URL if WEBHOOK_URL else 'غير محدد'} (سيتم تشغيل Flask دائمًا لتوافق Render)")

# ---------------------- Constants and Global Variables Setup ----------------------
TRADE_VALUE: float = 10.0
MAX_OPEN_TRADES: int = 5
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 3
SIGNAL_TRACKING_TIMEFRAME: str = '15m'
SIGNAL_TRACKING_LOOKBACK_DAYS: int = 1

# Indicator Parameters (MUST match ml.py)
RSI_PERIOD: int = 9
RSI_OVERSOLD: int = 30
RSI_OVERBOUGHT: int = 70
VOLUME_LOOKBACK_CANDLES: int = 1
RSI_MOMENTUM_LOOKBACK_CANDLES: int = 2

ENTRY_ATR_PERIOD: int = 10
ENTRY_ATR_MULTIPLIER: float = 1.5

SUPERTRAND_PERIOD: int = 10
SUPERTRAND_MULTIPLIER: float = 3.0

# Ichimoku Cloud Parameters (MUST match ml.py)
TENKAN_PERIOD: int = 9
KIJUN_PERIOD: int = 26
SENKOU_SPAN_B_PERIOD: int = 52
CHIKOU_LAG: int = 26

# Fibonacci & S/R Parameters (MUST match ml.py)
FIB_SR_LOOKBACK_WINDOW: int = 50

MIN_PROFIT_MARGIN_PCT: float = 1.0
MIN_VOLUME_15M_USDT: float = 50000.0

TARGET_APPROACH_THRESHOLD_PCT: float = 0.005

BINANCE_FEE_RATE: float = 0.001

BASE_ML_MODEL_NAME: str = 'DecisionTree_Scalping_V1'

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
except BinanceRequestException as req_err:
     logger.critical(f"❌ [Binance] خطأ في طلب Binance (مشكلة في الشبكة أو الطلب): {req_err}")
     exit(1)
except BinanceAPIException as api_err:
     logger.critical(f"❌ [Binance] خطأ في واجهة برمجة تطبيقات Binance (مفاتيح غير صالحة أو مشكلة في الخادم): {api_err}")
     exit(1)
except Exception as e:
    logger.critical(f"❌ [Binance] فشل غير متوقع في تهيئة عميل Binance: {e}")
    exit(1)

# ---------------------- Additional Indicator Functions ----------------------
def get_fear_greed_index() -> str:
    """Fetches the Fear & Greed Index from alternative.me and translates classification to Arabic."""
    classification_translation_ar = {
        "Extreme Fear": "خوف شديد", "Fear": "خوف", "Neutral": "محايد",
        "Greed": "جشع", "Extreme Greed": "جشع شديد",
    }
    url = "https://api.alternative.me/fng/"
    logger.debug(f"ℹ️ [Indicators] جلب مؤشر الخوف والجشع من {url}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        value = int(data["data"][0]["value"])
        classification_en = data["data"][0]["value_classification"]
        classification_ar = classification_translation_ar.get(classification_en, classification_en)
        logger.debug(f"✅ [Indicators] مؤشر الخوف والجشع: {value} ({classification_ar})")
        return f"{value} ({classification_ar})"
    except requests.exceptions.RequestException as e:
         logger.error(f"❌ [Indicators] خطأ في الشبكة أثناء جلب مؤشر الخوف والجشع: {e}")
         return "N/A (خطأ في الشبكة)"
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"❌ [Indicators] خطأ في تنسيق البيانات لمؤشر الخوف والجشع: {e}")
        return "N/A (خطأ في البيانات)"
    except Exception as e:
        logger.error(f"❌ [Indicators] خطأ غير متوقع أثناء جلب مؤشر الخوف والجشع: {e}", exc_info=True)
        return "N/A (خطأ غير معروف)"

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """
    Fetches historical candlestick data from Binance for a specified number of days.
    This function relies on python-binance's get_historical_klines to handle
    internal pagination for large data ranges.
    """
    if not client:
        logger.error("❌ [Data] عميل Binance غير مهيأ لجلب البيانات.")
        return None
    try:
        # Calculate the start date for the entire data range needed
        start_dt = datetime.utcnow() - timedelta(days=days + 1)
        start_str_overall = start_dt.strftime("%Y-%m-%d %H:%M:%S")

        logger.debug(f"ℹ️ [Data] جلب بيانات {interval} لـ {symbol} من {start_str_overall} فصاعدًا...")

        # Map interval string to Binance client constant
        binance_interval = None
        if interval == '15m':
            binance_interval = Client.KLINE_INTERVAL_15MINUTE
        elif interval == '5m':
            binance_interval = Client.KLINE_INTERVAL_5MINUTE
        elif interval == '1h':
            binance_interval = Client.KLINE_INTERVAL_1HOUR
        elif interval == '4h':
            binance_interval = Client.KLINE_INTERVAL_4HOUR
        elif interval == '1d':
            binance_interval = Client.KLINE_INTERVAL_1DAY
        else:
            logger.error(f"❌ [Data] فترة غير مدعومة: {interval}")
            return None

        # Call get_historical_klines for the entire period.
        # The python-binance library is designed to handle internal pagination
        # if the requested range exceeds the API's single-request limit (e.g., 1000 klines).
        klines = client.get_historical_klines(symbol, binance_interval, start_str_overall)

        if not klines:
            logger.warning(f"⚠️ [Data] لا توجد بيانات تاريخية ({interval}) لـ {symbol} للفترة المطلوبة.")
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
        initial_len = len(df)
        df.dropna(subset=numeric_cols, inplace=True)

        if len(df) < initial_len:
            logger.debug(f"ℹ️ [Data] {symbol}: تم حذف {initial_len - len(df)} صفوف بسبب قيم NaN في بيانات OHLCV.")

        if df.empty:
            logger.warning(f"⚠️ [Data] DataFrame لـ {symbol} فارغ بعد إزالة قيم NaN الأساسية.")
            return None

        # Sort by index (timestamp) to ensure chronological order
        df.sort_index(inplace=True)

        logger.debug(f"✅ [Data] تم جلب ومعالجة {len(df)} شمعة تاريخية ({interval}) لـ {symbol}.")
        return df

    except BinanceAPIException as api_err:
         logger.error(f"❌ [Data] خطأ في Binance API أثناء جلب البيانات لـ {symbol}: {api_err}")
         return None
    except BinanceRequestException as req_err:
         logger.error(f"❌ [Data] خطأ في الطلب أو الشبكة أثناء جلب البيانات لـ {symbol}: {req_err}")
         return None
    except Exception as e:
        logger.error(f"❌ [Data] خطأ غير متوقع أثناء جلب البيانات التاريخية لـ {symbol}: {e}", exc_info=True)
        return None


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculates Exponential Moving Average (EMA)."""
    if series is None or series.isnull().all() or len(series) < span:
        return pd.Series(index=series.index if series is not None else None, dtype=float)
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi_indicator(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """Calculates Relative Strength Index (RSI)."""
    df = df.copy()
    if 'close' not in df.columns or df['close'].isnull().all():
        logger.warning("⚠️ [Indicator RSI] عمود 'close' مفقود أو فارغ.")
        df['rsi'] = np.nan
        return df
    if len(df) < period:
        logger.warning(f"⚠️ [Indicator RSI] بيانات غير كافية ({len(df)} < {period}) لحساب RSI.")
        df['rsi'] = np.nan
        return df
    
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi_series = 100 - (100 / (1 + rs))
    df['rsi'] = rsi_series.ffill().fillna(50)

    return df

def calculate_atr_indicator(df: pd.DataFrame, period: int = ENTRY_ATR_PERIOD) -> pd.DataFrame:
    """Calculates Average True Range (ATR)."""
    df = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator ATR] أعمدة 'high', 'low', 'close' مفقودة أو فارغة.")
        df['atr'] = np.nan
        return df
    if len(df) < period + 1:
        logger.warning(f"⚠️ [Indicator ATR] بيانات غير كافية ({len(df)} < {period + 1}) لحساب ATR.")
        df['atr'] = np.nan
        return df

    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()

    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)

    df['atr'] = tr.ewm(span=period, adjust=False).mean()
    return df

def calculate_supertrend(df: pd.DataFrame, period: int = SUPERTRAND_PERIOD, multiplier: float = SUPERTRAND_MULTIPLIER) -> pd.DataFrame:
    """Calculates the Supertrend indicator."""
    df = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator Supertrend] أعمدة 'high', 'low', 'close' مفقودة أو فارغة. لا يمكن حساب Supertrend.")
        df['supertrend'] = np.nan
        df['supertrend_direction'] = 0
        return df

    # Ensure ATR is already calculated
    if 'atr' not in df.columns:
        df = calculate_atr_indicator(df, period=period)
        if 'atr' not in df.columns or df['atr'].isnull().all().any():
            logger.warning("⚠️ [Indicator Supertrend] فشل حساب ATR. لا يمكن حساب Supertrend.")
            df['supertrend'] = np.nan
            df['supertrend_direction'] = 0
            return df

    # Calculate Basic Upper and Lower Bands
    df['basic_upper_band'] = ((df['high'] + df['low']) / 2) + (multiplier * df['atr'])
    df['basic_lower_band'] = ((df['high'] + df['low']) / 2) - (multiplier * df['atr'])

    # Initialize Final Upper and Lower Bands
    df['final_upper_band'] = 0.0
    df['final_lower_band'] = 0.0

    # Initialize Supertrend and Direction
    df['supertrend'] = 0.0
    df['supertrend_direction'] = 0

    # Determine Supertrend value and direction
    for i in range(1, len(df)):
        # Final Upper Band
        if df['basic_upper_band'].iloc[i] < df['final_upper_band'].iloc[i-1] or \
           df['close'].iloc[i-1] > df['final_upper_band'].iloc[i-1]:
            df.loc[df.index[i], 'final_upper_band'] = df['basic_upper_band'].iloc[i]
        else:
            df.loc[df.index[i], 'final_upper_band'] = df['final_upper_band'].iloc[i-1]

        # Final Lower Band
        if df['basic_lower_band'].iloc[i] > df['final_lower_band'].iloc[i-1] or \
           df['close'].iloc[i-1] < df['final_lower_band'].iloc[i-1]:
            df.loc[df.index[i], 'final_lower_band'] = df['basic_lower_band'].iloc[i]
        else:
            df.loc[df.index[i], 'final_lower_band'] = df['final_lower_band'].iloc[i-1]

        # Supertrend logic
        if df['supertrend_direction'].iloc[i-1] == 1:
            if df['close'].iloc[i] < df['final_upper_band'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['final_upper_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = -1
            else:
                df.loc[df.index[i], 'supertrend'] = df['final_lower_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = 1
        elif df['supertrend_direction'].iloc[i-1] == -1:
            if df['close'].iloc[i] > df['final_lower_band'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['final_lower_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = 1
            else:
                df.loc[df.index[i], 'supertrend'] = df['final_upper_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = -1
        else:
            if df['close'].iloc[i] > df['final_lower_band'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['final_lower_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = 1
            elif df['close'].iloc[i] < df['final_upper_band'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['final_upper_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = -1
            else:
                df.loc[df.index[i], 'supertrend'] = df['close'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = 0


    # Drop temporary columns
    df.drop(columns=['basic_upper_band', 'basic_lower_band', 'final_upper_band', 'final_lower_band'], inplace=True, errors='ignore')
    logger.debug(f"✅ [Indicator Supertrend] تم حساب Supertrend.")
    return df

def _calculate_btc_trend_feature(df_btc: pd.DataFrame) -> Optional[pd.Series]:
    """
    Calculates a numerical representation of Bitcoin's trend based on EMA20 and EMA50.
    Returns 1 for bullish (صعودي), -1 for bearish (هبوطي), 0 for neutral/sideways (محايد/تذبذب).
    """
    logger.debug("ℹ️ [Indicators] حساب اتجاه البيتكوين للميزات...")
    # Need enough data for EMA50, plus a few extra candles for robustness
    min_data_for_ema = 50 + 5

    if df_btc is None or df_btc.empty or len(df_btc) < min_data_for_ema:
        logger.warning(f"⚠️ [Indicators] بيانات BTC/USDT غير كافية ({len(df_btc) if df_btc is not None else 0} < {min_data_for_ema}) لحساب اتجاه البيتكوين للميزات.")
        # Return a series of zeros (neutral) with the original index if data is insufficient
        return pd.Series(index=df_btc.index if df_btc is not None else None, data=0.0)

    df_btc_copy = df_btc.copy()
    df_btc_copy['close'] = pd.to_numeric(df_btc_copy['close'], errors='coerce')
    df_btc_copy.dropna(subset=['close'], inplace=True)

    if len(df_btc_copy) < min_data_for_ema:
        logger.warning(f"⚠️ [Indicators] بيانات BTC/USDT غير كافية بعد إزالة NaN لحساب الاتجاه.")
        return pd.Series(index=df_btc.index, data=0.0)

    ema20 = calculate_ema(df_btc_copy['close'], 20)
    ema50 = calculate_ema(df_btc_copy['close'], 50)

    # Combine EMAs and close into a single DataFrame for easier comparison
    ema_df = pd.DataFrame({'ema20': ema20, 'ema50': ema50, 'close': df_btc_copy['close']})
    ema_df.dropna(inplace=True)

    if ema_df.empty:
        logger.warning("⚠️ [Indicators] EMA DataFrame فارغ بعد إزالة NaN. لا يمكن حساب اتجاه البيتكوين.")
        return pd.Series(index=df_btc.index, data=0.0)

    # Initialize trend column with neutral (0.0)
    trend_series = pd.Series(index=ema_df.index, data=0.0)

    # Apply trend logic:
    # Bullish: current_close > ema20 > ema50
    trend_series[(ema_df['close'] > ema_df['ema20']) & (ema_df['ema20'] > ema_df['ema50'])] = 1.0
    # Bearish: current_close < ema20 < ema50
    trend_series[(ema_df['close'] < ema_df['ema20']) & (ema_df['ema20'] < ema_df['ema50'])] = -1.0

    # Reindex to original df_btc index and fill any remaining NaNs with 0 (neutral)
    # This ensures the series has the same index as the altcoin DataFrame for merging
    final_trend_series = trend_series.reindex(df_btc.index).fillna(0.0)
    logger.debug(f"✅ [Indicators] تم حساب ميزة اتجاه البيتكوين. أمثلة: {final_trend_series.tail().tolist()}")
    return final_trend_series


# NEW: Ichimoku Cloud Calculation (Copied from ml.py)
def calculate_ichimoku_cloud(df: pd.DataFrame, tenkan_period: int = TENKAN_PERIOD, kijun_period: int = KIJUN_PERIOD, senkou_span_b_period: int = SENKOU_SPAN_B_PERIOD, chikou_lag: int = CHIKOU_LAG) -> pd.DataFrame:
    """Calculates Ichimoku Cloud components and derived features."""
    df_ichimoku = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df_ichimoku.columns for col in required_cols) or df_ichimoku[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator Ichimoku] أعمدة OHLC مفقودة أو فارغة. لا يمكن حساب Ichimoku.")
        for col in ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span',
                    'ichimoku_tenkan_kijun_cross_signal', 'ichimoku_price_cloud_position', 'ichimoku_cloud_outlook']:
            df_ichimoku[col] = np.nan
        return df_ichimoku

    # Convert to numeric
    for col in required_cols:
        df_ichimoku[col] = pd.to_numeric(df_ichimoku[col], errors='coerce')

    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    df_ichimoku['tenkan_sen'] = (df_ichimoku['high'].rolling(window=tenkan_period, min_periods=1).max() +
                                 df_ichimoku['low'].rolling(window=tenkan_period, min_periods=1).min()) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    df_ichimoku['kijun_sen'] = (df_ichimoku['high'].rolling(window=kijun_period, min_periods=1).max() +
                                df_ichimoku['low'].rolling(window=kijun_period, min_periods=1).min()) / 2

    # Senkou Span A (Leading Span A): (Conversion Line + Base Line) / 2 plotted 26 periods ahead
    df_ichimoku['senkou_span_a'] = ((df_ichimoku['tenkan_sen'] + df_ichimoku['kijun_sen']) / 2).shift(kijun_period)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 plotted 26 periods ahead
    df_ichimoku['senkou_span_b'] = ((df_ichimoku['high'].rolling(window=senkou_span_b_period, min_periods=1).max() +
                                     df_ichimoku['low'].rolling(window=senkou_span_b_period, min_periods=1).min()) / 2).shift(kijun_period)

    # Chikou Span (Lagging Span): Close plotted 26 periods back
    df_ichimoku['chikou_span'] = df_ichimoku['close'].shift(-chikou_lag)

    # --- Derived Ichimoku Features ---
    # Tenkan/Kijun Cross Signal
    df_ichimoku['ichimoku_tenkan_kijun_cross_signal'] = 0
    if len(df_ichimoku) > 1:
        # Bullish cross: Tenkan-sen crosses above Kijun-sen
        df_ichimoku.loc[(df_ichimoku['tenkan_sen'].shift(1) < df_ichimoku['kijun_sen'].shift(1)) & \
                        (df_ichimoku['tenkan_sen'] > df_ichimoku['kijun_sen']), 'ichimoku_tenkan_kijun_cross_signal'] = 1
        # Bearish cross: Tenkan-sen crosses below Kijun-sen
        df_ichimoku.loc[(df_ichimoku['tenkan_sen'].shift(1) > df_ichimoku['kijun_sen'].shift(1)) & \
                        (df_ichimoku['tenkan_sen'] < df_ichimoku['kijun_sen']), 'ichimoku_tenkan_kijun_cross_signal'] = -1

    # Price vs Cloud Position (using current close price vs future cloud)
    df_ichimoku['ichimoku_price_cloud_position'] = 0
    # Price above cloud
    df_ichimoku.loc[(df_ichimoku['close'] > df_ichimoku[['senkou_span_a', 'senkou_span_b']].max(axis=1)), 'ichimoku_price_cloud_position'] = 1
    # Price below cloud
    df_ichimoku.loc[(df_ichimoku['close'] < df_ichimoku[['senkou_span_a', 'senkou_span_b']].min(axis=1)), 'ichimoku_price_cloud_position'] = -1

    # Cloud Outlook (future cloud's color)
    df_ichimoku['ichimoku_cloud_outlook'] = 0
    df_ichimoku.loc[(df_ichimoku['senkou_span_a'] > df_ichimoku['senkou_span_b']), 'ichimoku_cloud_outlook'] = 1
    df_ichimoku.loc[(df_ichimoku['senkou_span_a'] < df_ichimoku['senkou_span_b']), 'ichimoku_cloud_outlook'] = -1

    logger.debug(f"✅ [Indicator Ichimoku] تم حساب مكونات وميزات Ichimoku Cloud.")
    return df_ichimoku


# NEW: Fibonacci Retracement Features (Copied from ml.py)
def calculate_fibonacci_features(df: pd.DataFrame, lookback_window: int = FIB_SR_LOOKBACK_WINDOW) -> pd.DataFrame:
    """
    Calculates Fibonacci Retracement levels from a recent swing (max/min in lookback window)
    and generates features based on current price position relative to these levels.
    """
    df_fib = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df_fib.columns for col in required_cols) or df_fib[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator Fibonacci] أعمدة OHLC مفقودة أو فارغة. لا يمكن حساب ميزات Fibonacci.")
        for col in ['fib_236_retrace_dist_norm', 'fib_382_retrace_dist_norm', 'fib_618_retrace_dist_norm', 'is_price_above_fib_50']:
            df_fib[col] = np.nan
        return df_fib
    if len(df_fib) < lookback_window:
        logger.warning(f"⚠️ [Indicator Fibonacci] بيانات غير كافية ({len(df_fib)} < {lookback_window}) لحساب Fibonacci.")
        for col in ['fib_236_retrace_dist_norm', 'fib_382_retrace_dist_norm', 'fib_618_retrace_dist_norm', 'is_price_above_fib_50']:
            df_fib[col] = np.nan
        return df_fib

    # Convert to numeric
    for col in required_cols:
        df_fib[col] = pd.to_numeric(df_fib[col], errors='coerce')

    df_fib['fib_236_retrace_dist_norm'] = np.nan
    df_fib['fib_382_retrace_dist_norm'] = np.nan
    df_fib['fib_618_retrace_dist_norm'] = np.nan
    df_fib['is_price_above_fib_50'] = 0

    for i in range(lookback_window - 1, len(df_fib)):
        window_df = df_fib.iloc[i - lookback_window + 1 : i + 1]
        swing_high = window_df['high'].max()
        swing_low = window_df['low'].min()
        current_close = df_fib['close'].iloc[i]

        price_range = swing_high - swing_low

        if price_range > 0:
            # For Uptrend Retracement (price drops from high to low)
            # Retracement levels are calculated from (Swing High - (Swing High - Swing Low) * Fib Level)
            fib_0_236 = swing_high - (price_range * 0.236)
            fib_0_382 = swing_high - (price_range * 0.382)
            fib_0_500 = swing_high - (price_range * 0.500)
            fib_0_618 = swing_high - (price_range * 0.618)

            # Features: Normalized distance from current price to key Fib levels
            if price_range != 0:
                df_fib.loc[df_fib.index[i], 'fib_236_retrace_dist_norm'] = (current_close - fib_0_236) / price_range
                df_fib.loc[df_fib.index[i], 'fib_382_retrace_dist_norm'] = (current_close - fib_0_382) / price_range
                df_fib.loc[df_fib.index[i], 'fib_618_retrace_dist_norm'] = (current_close - fib_0_618) / price_range

            # Is price above 0.5 Fibonacci retracement level?
            if current_close > fib_0_500:
                df_fib.loc[df_fib.index[i], 'is_price_above_fib_50'] = 1
            else:
                df_fib.loc[df_fib.index[i], 'is_price_above_fib_50'] = 0

    logger.debug(f"✅ [Indicator Fibonacci] تم حساب ميزات Fibonacci.")
    return df_fib


# NEW: Support and Resistance Features (Copied from ml.py)
def calculate_support_resistance_features(df: pd.DataFrame, lookback_window: int = FIB_SR_LOOKBACK_WINDOW) -> pd.DataFrame:
    """
    Calculates simplified support and resistance features based on the lowest low and highest high
    within a rolling lookback window.
    """
    df_sr = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator S/R] أعمدة OHLC مفقودة أو فارغة. لا يمكن حساب ميزات الدعم/المقاومة.")
        for col in ['price_distance_to_recent_low_norm', 'price_distance_to_recent_high_norm']:
            df_sr[col] = np.nan
        return df_sr
    if len(df_sr) < lookback_window:
        logger.warning(f"⚠️ [Indicator S/R] بيانات غير كافية ({len(df_sr)} < {lookback_window}) لحساب S/R.")
        for col in ['price_distance_to_recent_low_norm', 'price_distance_to_recent_high_norm']:
            df_sr[col] = np.nan
        return df_sr

    # Convert to numeric
    for col in required_cols:
        df_sr[col] = pd.to_numeric(df_sr[col], errors='coerce')

    df_sr['price_distance_to_recent_low_norm'] = np.nan
    df_sr['price_distance_to_recent_high_norm'] = np.nan

    for i in range(lookback_window - 1, len(df_sr)):
        window_df = df_sr.iloc[i - lookback_window + 1 : i + 1]
        recent_high = window_df['high'].max()
        recent_low = window_df['low'].min()
        current_close = df_sr['close'].iloc[i]

        price_range = recent_high - recent_low

        if price_range > 0:
            df_sr.loc[df_sr.index[i], 'price_distance_to_recent_low_norm'] = (current_close - recent_low) / price_range
            df_sr.loc[df_sr.index[i], 'price_distance_to_recent_high_norm'] = (recent_high - current_close) / price_range
        else:
            df_sr.loc[df_sr.index[i], 'price_distance_to_recent_low_norm'] = 0.0
            df_sr.loc[df_sr.index[i], 'price_distance_to_recent_high_norm'] = 0.0

    logger.debug(f"✅ [Indicator S/R] تم حساب ميزات الدعم والمقاومة.")
    return df_sr


# ---------------------- Database Connection Setup ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    """Initializes database connection and creates tables if they don't exist."""
    global conn, cur
    logger.info("[DB] بدء تهيئة قاعدة البيانات...")
    for attempt in range(retries):
        try:
            logger.info(f"[DB] محاولة الاتصال بقاعدة البيانات (المحاولة {attempt + 1}/{retries})...")
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            cur = conn.cursor()
            logger.info("✅ [DB] تم الاتصال بقاعدة البيانات بنجاح.")

            # --- Create or update signals table (Modified schema) ---
            logger.info("[DB] التحقق من/إنشاء جدول 'signals'...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    initial_target DOUBLE PRECISION NOT NULL,
                    current_target DOUBLE PRECISION NOT NULL,
                    r2_score DOUBLE PRECISION,
                    volume_15m DOUBLE PRECISION,
                    achieved_target BOOLEAN DEFAULT FALSE,
                    closing_price DOUBLE PRECISION,
                    closed_at TIMESTAMP,
                    sent_at TIMESTAMP DEFAULT NOW(),
                    entry_time TIMESTAMP DEFAULT NOW(),
                    time_to_target INTERVAL,
                    profit_percentage DOUBLE PRECISION,
                    strategy_name TEXT,
                    signal_details JSONB,
                    stop_loss DOUBLE PRECISION
                );""")
            conn.commit()
            logger.info("✅ [DB] جدول 'signals' موجود أو تم إنشاؤه.")

            # --- Create ml_models table (NEW) ---
            logger.info("[DB] التحقق من/إنشاء جدول 'ml_models'...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ml_models (
                    id SERIAL PRIMARY KEY,
                    model_name TEXT NOT NULL UNIQUE,
                    model_data BYTEA NOT NULL,
                    trained_at TIMESTAMP DEFAULT NOW(),
                    metrics JSONB
                );""")
            conn.commit()
            logger.info("✅ [DB] جدول 'ml_models' موجود أو تم إنشاؤه.")

            # --- Create market_dominance table (if it doesn't exist) ---
            logger.info("[DB] التحقق من/إنشاء جدول 'market_dominance'...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS market_dominance (
                    id SERIAL PRIMARY KEY,
                    recorded_at TIMESTAMP DEFAULT NOW(),
                    btc_dominance DOUBLE PRECISION,
                    eth_dominance DOUBLE PRECISION
                );
            """)
            conn.commit()
            logger.info("✅ [DB] جدول 'market_dominance' موجود أو تم إنشاؤه.")

            logger.info("✅ [DB] تم تهيئة قاعدة البيانات بنجاح.")
            return

        except OperationalError as op_err:
            logger.error(f"❌ [DB] خطأ في التشغيل أثناء الاتصال (المحاولة {attempt + 1}): {op_err}")
            if conn: conn.rollback()
            if attempt == retries - 1:
                 logger.critical("❌ [DB] فشلت جميع محاولات الاتصال بقاعدة البيانات.")
                 raise op_err
            time.sleep(delay)
        except Exception as e:
            logger.critical(f"❌ [DB] فشل غير متوقع في تهيئة قاعدة البيانات (المحاولة {attempt + 1}): {e}", exc_info=True)
            if conn: conn.rollback()
            if attempt == retries - 1:
                 logger.critical("❌ [DB] فشلت جميع محاولات الاتصال بقاعدة البيانات.")
                 raise e
            time.sleep(delay)

    logger.critical("❌ [DB] فشل الاتصال بقاعدة البيانات بعد عدة محاولات.")
    exit(1)


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
                  check_cur.execute("SELECT 1;")
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
        try:
            init_db()
            return True
        except Exception as recon_err:
             logger.error(f"❌ [DB] فشلت محاولة إعادة الاتصال بعد خطأ غير متوقع: {recon_err}")
             return False

def load_ml_model_from_db(symbol: str) -> Optional[Any]:
    """Loads the latest trained ML model for a specific symbol from the database."""
    global ml_models
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"

    if model_name in ml_models:
        logger.debug(f"ℹ️ [ML Model] النموذج '{model_name}' موجود بالفعل في الذاكرة.")
        return ml_models[model_name]

    if not check_db_connection() or not conn:
        logger.error(f"❌ [ML Model] لا يمكن تحميل نموذج ML لـ {symbol} بسبب مشكلة في الاتصال بقاعدة البيانات.")
        return None

    try:
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT model_data FROM ml_models WHERE model_name = %s ORDER BY trained_at DESC LIMIT 1;", (model_name,))
            result = db_cur.fetchone()
            if result and result['model_data']:
                model = pickle.loads(result['model_data'])
                ml_models[model_name] = model
                logger.info(f"✅ [ML Model] تم تحميل نموذج ML '{model_name}' من قاعدة البيانات بنجاح.")
                return model
            else:
                logger.warning(f"⚠️ [ML Model] لم يتم العثور على نموذج ML بالاسم '{model_name}' في قاعدة البيانات. يرجى تدريب النموذج أولاً.")
                return None
    except psycopg2.Error as db_err:
        logger.error(f"❌ [ML Model] خطأ في قاعدة البيانات أثناء تحميل نموذج ML لـ {symbol}: {db_err}", exc_info=True)
        return None
    except pickle.UnpicklingError as unpickle_err:
        logger.error(f"❌ [ML Model] خطأ في فك حزمة نموذج ML لـ {symbol}: {unpickle_err}. قد يكون النموذج تالفًا أو تم حفظه بإصدار مختلف.", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"❌ [ML Model] خطأ غير متوقع أثناء تحميل نموذج ML لـ {symbol}: {e}", exc_info=True)
        return None


def convert_np_values(obj: Any) -> Any:
    """Converts NumPy data types to native Python types for JSON and DB compatibility."""
    if isinstance(obj, dict):
        return {k: convert_np_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_values(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

# ---------------------- WebSocket Management for Ticker Prices ----------------------
def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
    """Handles incoming WebSocket messages for mini-ticker prices."""
    global ticker_data
    try:
        if isinstance(msg, list):
            for ticker_item in msg:
                symbol = ticker_item.get('s')
                price_str = ticker_item.get('c')
                if symbol and 'USDT' in symbol and price_str:
                    try:
                        ticker_data[symbol] = float(price_str)
                    except ValueError:
                         logger.warning(f"⚠️ [WS] قيمة سعر غير صالحة للرمز {symbol}: '{price_str}'")
        elif isinstance(msg, dict):
             if msg.get('e') == 'error':
                 logger.error(f"❌ [WS] رسالة خطأ من WebSocket: {msg.get('m', 'لا توجد تفاصيل خطأ')}")
             elif msg.get('stream') and msg.get('data'):
                 for ticker_item in msg.get('data', []):
                    symbol = ticker_item.get('s')
                    price_str = ticker_item.get('c')
                    if symbol and 'USDT' in symbol and price_str:
                        try:
                            ticker_data[symbol] = float(price_str)
                        except ValueError:
                             logger.warning(f"⚠️ [WS] قيمة سعر غير صالحة للرمز {symbol} في التدفق المدمج: '{price_str}'")
        else:
             logger.warning(f"⚠️ [WS] تم استلام رسالة WebSocket بتنسيق غير متوقع: {type(msg)}")

    except Exception as e:
        logger.error(f"❌ [WS] خطأ في معالجة رسالة المؤشر: {e}", exc_info=True)


def run_ticker_socket_manager() -> None:
    """Runs and manages the WebSocket connection for mini-ticker."""
    while True:
        try:
            logger.info("ℹ️ [WS] بدء مدير WebSocket لأسعار المؤشرات...")
            twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
            twm.start()

            stream_name = twm.start_miniticker_socket(callback=handle_ticker_message)
            logger.info(f"✅ [WS] تم بدء تدفق WebSocket: {stream_name}")

            twm.join()
            logger.warning("⚠️ [WS] تم إيقاف مدير WebSocket. إعادة التشغيل...")

        except Exception as e:
            logger.error(f"❌ [WS] خطأ فادح في مدير WebSocket: {e}. إعادة التشغيل في 15 ثانية...", exc_info=True)

        time.sleep(15)

# ---------------------- Other Helper Functions (Volume) ----------------------
def fetch_recent_volume(symbol: str, interval: str = SIGNAL_GENERATION_TIMEFRAME, num_candles: int = VOLUME_LOOKBACK_CANDLES) -> float:
    """Fetches the trading volume in USDT for the last `num_candles` of the specified `interval`."""
    if not client:
         logger.error(f"❌ [Data Volume] عميل Binance غير مهيأ لجلب حجم التداول لـ {symbol}.")
         return 0.0
    try:
        logger.debug(f"ℹ️ [Data Volume] جلب حجم التداول لآخر {num_candles} شمعة {interval} لـ {symbol}...")

        # Map interval string to Binance client constant
        binance_interval = None
        if interval == '15m':
            binance_interval = Client.KLINE_INTERVAL_15MINUTE
        elif interval == '5m':
            binance_interval = Client.KLINE_INTERVAL_5MINUTE
        elif interval == '1h':
            binance_interval = Client.KLINE_INTERVAL_1HOUR
        elif interval == '4h':
            binance_interval = Client.KLINE_INTERVAL_4HOUR
        elif interval == '1d':
            binance_interval = Client.KLINE_INTERVAL_1DAY
        else:
            logger.error(f"❌ [Data Volume] فترة غير مدعومة: {interval}")
            return 0.0

        klines = client.get_klines(symbol=symbol, interval=binance_interval, limit=num_candles)
        if not klines or len(klines) < num_candles:
             logger.warning(f"⚠️ [Data Volume] بيانات {interval} غير كافية (أقل من {num_candles} شمعة) لـ {symbol}.")
             return 0.0

        # k[7] is the quote asset volume (e.g., USDT volume)
        volume_usdt = sum(float(k[7]) for k in klines if len(k) > 7 and k[7])
        logger.debug(f"✅ [Data Volume] السيولة لآخر {num_candles} شمعة {interval} لـ {symbol}: {volume_usdt:.2f} USDT")
        return volume_usdt
    except (BinanceAPIException, BinanceRequestException) as binance_err:
         logger.error(f"❌ [Data Volume] خطأ في Binance API أو الشبكة أثناء جلب حجم التداول لـ {symbol}: {binance_err}")
         return 0.0
    except Exception as e:
        logger.error(f"❌ [Data Volume] خطأ غير متوقع أثناء جلب حجم التداول لـ {symbol}: {e}", exc_info=True)
        return 0.0

# ---------------------- Reading and Validating Symbols List ----------------------
def get_crypto_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """
    Reads the list of currency symbols from a text file, appends 'USDT' to each,
    then validates them as valid USDT pairs available for Spot trading on Binance.
    """
    raw_symbols: List[str] = []
    logger.info(f"ℹ️ [Data] قراءة قائمة الرموز من الملف '{filename}'...")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)

        if not os.path.exists(file_path):
            file_path = os.path.abspath(filename)
            if not os.path.exists(file_path):
                 logger.error(f"❌ [Data] الملف '{filename}' غير موجود في دليل السكريبت أو الدليل الحالي.")
                 return []
            else:
                 logger.warning(f"⚠️ [Data] الملف '{filename}' غير موجود في دليل السكريبت. استخدام الملف في الدليل الحالي: '{file_path}'")

        with open(file_path, 'r', encoding='utf-8') as f:
            # Append USDT to each symbol if not already present
            raw_symbols = [f"{line.strip().upper()}USDT" if not line.strip().upper().endswith('USDT') else line.strip().upper()
                           for line in f if line.strip() and not line.startswith('#')]
        raw_symbols = sorted(list(set(raw_symbols)))
        logger.info(f"ℹ️ [Data] تم قراءة {len(raw_symbols)} رمزًا مبدئيًا من '{file_path}'.")

    except FileNotFoundError:
         logger.error(f"❌ [Data] الملف '{filename}' غير موجود.")
         return []
    except Exception as e:
        logger.error(f"❌ [Data] خطأ في قراءة الملف '{filename}': {e}", exc_info=True)
        return []

    if not raw_symbols:
         logger.warning("⚠️ [Data] قائمة الرموز الأولية فارغة.")
         return []

    if not client:
        logger.error("❌ [Data Validation] عميل Binance غير مهيأ. لا يمكن التحقق من الرموز.")
        return raw_symbols

    try:
        logger.info("ℹ️ [Data Validation] التحقق من الرموز وحالة التداول من Binance API...")
        exchange_info = client.get_exchange_info()
        valid_trading_usdt_symbols = {
            s['symbol'] for s in exchange_info['symbols']
            if s.get('quoteAsset') == 'USDT' and
               s.get('status') == 'TRADING' and
               s.get('isSpotTradingAllowed') is True
        }
        logger.info(f"ℹ️ [Data Validation] تم العثور على {len(valid_trading_usdt_symbols)} أزواج تداول USDT صالحة في Spot على Binance.")
        validated_symbols = [symbol for symbol in raw_symbols if symbol in valid_trading_usdt_symbols]

        removed_count = len(raw_symbols) - len(validated_symbols)
        if removed_count > 0:
            removed_symbols = set(raw_symbols) - set(validated_symbols)
            logger.warning(f"⚠️ [Data Validation] تم إزالة {removed_count} رمز تداول USDT غير صالح أو غير متاح من القائمة: {', '.join(removed_symbols)}")

        logger.info(f"✅ [Data Validation] تم التحقق من الرموز. استخدام {len(validated_symbols)} رمزًا صالحًا.")
        return validated_symbols

    except (BinanceAPIException, BinanceRequestException) as binance_err:
         logger.error(f"❌ [Data Validation] خطأ في Binance API أو الشبكة أثناء التحقق من الرموز: {binance_err}")
         logger.warning("⚠️ [Data Validation] استخدام القائمة الأولية من الملف بدون تحقق Binance.")
         return raw_symbols
    except Exception as api_err:
         logger.error(f"❌ [Data Validation] خطأ غير متوقع أثناء التحقق من رموز Binance: {api_err}", exc_info=True)
         logger.warning("⚠️ [Data Validation] استخدام القائمة الأولية من الملف بدون تحقق Binance.")
         return raw_symbols

# ---------------------- Comprehensive Performance Report Generation Function ----------------------
def generate_performance_report() -> Tuple[str, Optional[Dict]]:
    """Generates a comprehensive performance report from the database in Arabic, including recent closed trades and USD profit/loss."""
    logger.info("ℹ️ [Report] إنشاء تقرير الأداء...")
    if not check_db_connection() or not conn or not cur:
        logger.error("❌ [Report] لا يمكن إنشاء التقرير، مشكلة في الاتصال بقاعدة البيانات.")
        return "❌ لا يمكن إنشاء التقرير، مشكلة في الاتصال بقاعدة البيانات.", None
    try:
        with conn.cursor() as report_cur:
            # Modify query to include current_target and add current price from ticker_data
            report_cur.execute("SELECT id, symbol, entry_price, current_target, entry_time FROM signals WHERE achieved_target = FALSE ORDER BY entry_time DESC;")
            open_signals = report_cur.fetchall()
            open_signals_count = len(open_signals)

            report_cur.execute("""
                SELECT
                    COUNT(*) AS total_closed,
                    COUNT(*) FILTER (WHERE profit_percentage > 0) AS winning_signals,
                    COUNT(*) FILTER (WHERE profit_percentage <= 0) AS losing_signals,
                    COALESCE(SUM(profit_percentage) FILTER (WHERE profit_percentage > 0), 0) AS gross_profit_pct_sum,
                    COALESCE(SUM(profit_percentage) FILTER (WHERE profit_percentage <= 0), 0) AS gross_loss_pct_sum,
                    COALESCE(AVG(profit_percentage) FILTER (WHERE profit_percentage > 0), 0) AS avg_win_pct,
                    COALESCE(AVG(profit_percentage) FILTER (WHERE profit_percentage <= 0), 0) AS avg_loss_pct
                FROM signals
                WHERE achieved_target = TRUE;
            """)
            closed_stats = report_cur.fetchone() or {}

            total_closed = closed_stats.get('total_closed', 0)
            winning_signals = closed_stats.get('winning_signals', 0)
            losing_signals = closed_stats.get('losing_signals', 0)
            gross_profit_pct_sum = closed_stats.get('gross_profit_pct_sum', 0.0)
            gross_loss_pct_sum = closed_stats.get('gross_loss_pct_sum', 0.0)
            avg_win_pct = closed_stats.get('avg_win_pct', 0.0)
            avg_loss_pct = closed_stats.get('avg_loss_pct', 0.0)

            # Calculate USD profit/loss based on a fixed TRADE_VALUE
            gross_profit_usd = (gross_profit_pct_sum / 100.0) * TRADE_VALUE
            gross_loss_usd = (gross_loss_pct_sum / 100.0) * TRADE_VALUE

            # Total fees for all closed trades (entry and exit)
            total_fees_usd = total_closed * (TRADE_VALUE * BINANCE_FEE_RATE + (TRADE_VALUE * (1 + (avg_win_pct / 100.0 if avg_win_pct > 0 else 0))) * BINANCE_FEE_RATE)

            net_profit_usd = gross_profit_usd + gross_loss_usd - total_fees_usd
            net_profit_pct = (net_profit_usd / (total_closed * TRADE_VALUE)) * 100 if total_closed * TRADE_VALUE > 0 else 0.0


            win_rate = (winning_signals / total_closed) * 100 if total_closed > 0 else 0.0
            profit_factor = float('inf') if gross_loss_pct_sum == 0 else (gross_profit_pct_sum / abs(gross_loss_pct_sum))

        report_text = f"""📊 *تقرير الأداء الشامل:*
_(قيمة التداول المفترضة: ${TRADE_VALUE:,.2f} ورسوم Binance: {BINANCE_FEE_RATE*100:.2f}% لكل تداول)_
——————————————
📈 الإشارات المفتوحة حاليًا: *{open_signals_count}*
"""
        inline_keyboard_buttons = []

        if open_signals:
            report_text += "  • التفاصيل:\n"
            for i, signal in enumerate(open_signals):
                # Corrected: Escape backslashes for f-string to work with Telegram Markdown
                safe_symbol = str(signal['symbol']).replace('_', r'\_').replace('*', r'\*').replace('[', r'\[').replace('`', r'\`')
                entry_time_str = signal['entry_time'].strftime('%Y-%m-%d %H:%M') if signal['entry_time'] else 'N/A'

                # Get current price from ticker data
                current_price = ticker_data.get(signal['symbol'], 0.0)

                # Calculate progress towards target
                progress_pct = 0.0
                if current_price > 0 and signal['entry_price'] > 0 and signal['current_target'] > signal['entry_price']:
                    progress_pct = ((current_price - signal['entry_price']) / (signal['current_target'] - signal['entry_price'])) * 100

                # Determine progress icon based on percentage
                progress_icon = "🔴"
                if progress_pct >= 75:
                    progress_icon = "🟢"
                elif progress_pct >= 50:
                    progress_icon = "🟡"
                elif progress_pct >= 25:
                    progress_icon = "🟠"

                # Add entry price, target, and current price to report in an organized format
                report_text += f"""    *{i+1}. {safe_symbol}*
       💲 *الدخول:* `${signal['entry_price']:.8g}`
       🎯 *الهدف:* `${signal['current_target']:.8g}`
       💵 *السعر الحالي:* `${current_price:.8g}`
       {progress_icon} *التقدم:* `{progress_pct:.1f}%`
       ⏰ *تاريخ الفتح:* `{entry_time_str}`
"""
                # Add an "Exit Trade" button for each open signal
                inline_keyboard_buttons.append([
                    {"text": f"❌ إغلاق {safe_symbol}", "callback_data": f"exit_trade_{signal['id']}"}
                ])
                # Add separator between signals unless it's the last signal
                if i < len(open_signals) - 1:
                    report_text += "       ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\n"
        else:
            report_text += "  • لا توجد إشارات مفتوحة حاليًا.\n"

        report_text += f"""——————————————
📉 *إحصائيات الإشارات المغلقة:* • إجمالي الإشارات المغلقة: *{total_closed}* ✅ الإشارات الرابحة: *{winning_signals}* ({win_rate:.2f}%)
  ❌ الإشارات الخاسرة: *{losing_signals}* ——————————————
💰 *الربحية الإجمالية:* • إجمالي الربح: *{gross_profit_pct_sum:+.2f}%* (≈ *${gross_profit_usd:+.2f}*)
  • إجمالي الخسارة: *{gross_loss_pct_sum:+.2f}%* (≈ *${gross_loss_usd:+.2f}*)
  • إجمالي الرسوم المقدرة: *${total_fees_usd:,.2f}* • *صافي الربح:* *{net_profit_pct:+.2f}%* (≈ *${net_profit_usd:+.2f}*)
  • متوسط الربح لكل صفقة رابحة: *{avg_win_pct:+.2f}%* • متوسط الخسارة لكل صفقة خاسرة: *{avg_loss_pct:+.2f}%* • عامل الربح: *{'∞' if profit_factor == float('inf') else f'{profit_factor:.2f}'}* ——————————————
🕰️ _تم تحديث التقرير في: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"""

        logger.info("✅ [Report] تم إنشاء تقرير الأداء بنجاح.")

        reply_markup = None
        if inline_keyboard_buttons:
            # Add a refresh button at the very bottom
            inline_keyboard_buttons.append([{"text": "🔄 تحديث التقرير", "callback_data": "get_report"}])
            reply_markup = {"inline_keyboard": inline_keyboard_buttons}

        return report_text, reply_markup

    except psycopg2.Error as db_err:
        logger.error(f"❌ [Report] خطأ في قاعدة البيانات أثناء إنشاء تقرير الأداء: {db_err}")
        if conn: conn.rollback()
        return "❌ خطأ في قاعدة البيانات أثناء إنشاء تقرير الأداء.", None
    except Exception as e:
        logger.error(f"❌ [Report] حدث خطأ غير متوقع أثناء إنشاء تقرير الأداء: {e}", exc_info=True)
        return "❌ حدث خطأ غير متوقع أثناء إنشاء تقرير الأداء.", None

# ---------------------- Trading Strategy (Adjusted for ML-Only) -------------------

class ScalpingTradingStrategy:
    """Encapsulates the trading strategy logic, now relying solely on ML model prediction for buy signals."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ml_model = load_ml_model_from_db(symbol)
        if self.ml_model is None:
            logger.warning(f"⚠️ [Strategy {self.symbol}] لم يتم تحميل نموذج ML لـ {symbol}. لن تتمكن الإستراتيجية من إنشاء إشارات.")

        # Updated feature columns to include all new indicators (MUST match ml.py)
        self.feature_columns_for_ml = [
            'volume_15m_avg',
            'rsi_momentum_bullish',
            'btc_trend_feature',
            'supertrend_direction',
            # Ichimoku features
            'ichimoku_tenkan_kijun_cross_signal',
            'ichimoku_price_cloud_position',
            'ichimoku_cloud_outlook',
            # Fibonacci features
            'fib_236_retrace_dist_norm',
            'fib_382_retrace_dist_norm',
            'fib_618_retrace_dist_norm',
            'is_price_above_fib_50',
            # Support/Resistance features
            'price_distance_to_recent_low_norm',
            'price_distance_to_recent_high_norm'
        ]

    def populate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculates all required indicators for the ML model's features."""
        logger.debug(f"ℹ️ [Strategy {self.symbol}] حساب المؤشرات لنموذج ML...")

        # min_len_required should reflect the max lookback of all indicators
        min_len_required = max(
            VOLUME_LOOKBACK_CANDLES,
            RSI_PERIOD,
            RSI_MOMENTUM_LOOKBACK_CANDLES,
            ENTRY_ATR_PERIOD,
            SUPERTRAND_PERIOD,
            TENKAN_PERIOD,
            KIJUN_PERIOD,
            SENKOU_SPAN_B_PERIOD,
            CHIKOU_LAG,
            FIB_SR_LOOKBACK_WINDOW,
            55
        ) + 5

        if len(df) < min_len_required:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame قصير جدًا ({len(df)} < {min_len_required}) لحساب مؤشر ML.")
            return None

        try:
            df_calc = df.copy()
            # Calculate RSI as it's a prerequisite for rsi_momentum_bullish
            df_calc = calculate_rsi_indicator(df_calc, RSI_PERIOD)
            # Calculate ATR for target price calculation and Supertrend
            df_calc = calculate_atr_indicator(df_calc, ENTRY_ATR_PERIOD)
            # Calculate Supertrend
            df_calc = calculate_supertrend(df_calc, SUPERTRAND_PERIOD, SUPERTRAND_MULTIPLIER)

            # Add new features: 15-minute average liquidity volume (1 15m candle)
            df_calc['volume_15m_avg'] = df_calc['volume'].rolling(window=VOLUME_LOOKBACK_CANDLES, min_periods=1).mean()

            # Add bullish RSI Momentum indicator
            df_calc['rsi_momentum_bullish'] = 0
            if len(df_calc) >= RSI_MOMENTUM_LOOKBACK_CANDLES + 1:
                for i in range(RSI_MOMENTUM_LOOKBACK_CANDLES, len(df_calc)):
                    rsi_slice = df_calc['rsi'].iloc[i - RSI_MOMENTUM_LOOKBACK_CANDLES : i + 1]
                    if not rsi_slice.isnull().any() and np.all(np.diff(rsi_slice) > 0) and rsi_slice.iloc[-1] > 50:
                        df_calc.loc[df_calc.index[i], 'rsi_momentum_bullish'] = 1

            # Fetch and calculate BTC trend feature
            btc_df = fetch_historical_data("BTCUSDT", interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
            btc_trend_series = None
            if btc_df is not None and not btc_df.empty:
                btc_trend_series = _calculate_btc_trend_feature(btc_df)
                if btc_trend_series is not None:
                    # Merge BTC trend with the current symbol's DataFrame based on timestamp index
                    df_calc = df_calc.merge(btc_trend_series.rename('btc_trend_feature'),
                                            left_index=True, right_index=True, how='left')
                    df_calc['btc_trend_feature'] = df_calc['btc_trend_feature'].fillna(0.0)
                    logger.debug(f"ℹ️ [Strategy {self.symbol}] تم دمج ميزة اتجاه البيتكوين.")
                else:
                    logger.warning(f"⚠️ [Strategy {self.symbol}] فشل حساب ميزة اتجاه البيتكوين. تعيين 'btc_trend_feature' إلى 0.")
                    df_calc['btc_trend_feature'] = 0.0
            else:
                logger.warning(f"⚠️ [Strategy {self.symbol}] فشل جلب البيانات التاريخية للبيتكوين. تعيين 'btc_trend_feature' إلى 0.")
                df_calc['btc_trend_feature'] = 0.0

            # NEW: Calculate Ichimoku Cloud components and features
            df_calc = calculate_ichimoku_cloud(df_calc, TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_SPAN_B_PERIOD, CHIKOU_LAG)
            logger.debug(f"ℹ️ [Strategy {self.symbol}] تم حساب ميزات Ichimoku Cloud.")

            # NEW: Calculate Fibonacci Retracement features
            df_calc = calculate_fibonacci_features(df_calc, FIB_SR_LOOKBACK_WINDOW)
            logger.debug(f"ℹ️ [Strategy {self.symbol}] تم حساب ميزات Fibonacci Retracement.")

            # NEW: Calculate Support and Resistance features
            df_calc = calculate_support_resistance_features(df_calc, FIB_SR_LOOKBACK_WINDOW)
            logger.debug(f"ℹ️ [Strategy {self.symbol}] تم حساب ميزات الدعم والمقاومة.")


            # Ensure all feature columns for ML exist and are numeric
            for col in self.feature_columns_for_ml:
                if col not in df_calc.columns:
                    logger.warning(f"⚠️ [Strategy {self.symbol}] عمود ميزة مفقود لنموذج ML: {col}")
                    df_calc[col] = np.nan
                else:
                    df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')

            initial_len = len(df_calc)
            # Use all required columns for dropna, including ML features and ATR for target
            all_required_cols = list(set(self.feature_columns_for_ml + [
                'open', 'high', 'low', 'close', 'volume', 'atr', 'supertrend'
            ]))
            df_cleaned = df_calc.dropna(subset=all_required_cols).copy()
            dropped_count = initial_len - len(df_cleaned)

            if dropped_count > 0:
                 logger.debug(f"ℹ️ [Strategy {self.symbol}] تم حذف {dropped_count} صفوف بسبب قيم NaN في المؤشرات.")
            if df_cleaned.empty:
                logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame فارغ بعد إزالة قيم NaN للمؤشرات.")
                return None

            latest = df_cleaned.iloc[-1]
            logger.debug(f"✅ [Strategy {self.symbol}] تم حساب مؤشرات ML. الأحدث: {latest.to_dict()}")
            return df_cleaned

        except KeyError as ke:
             logger.error(f"❌ [Strategy {self.symbol}] خطأ: لم يتم العثور على العمود المطلوب أثناء حساب المؤشر: {ke}", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"❌ [Strategy {self.symbol}] خطأ غير متوقع أثناء حساب المؤشر: {e}", exc_info=True)
            return None


    def generate_buy_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Generates a buy signal based solely on the ML model's bullish prediction,
        followed by essential filters (volume, profit margin).
        """
        logger.debug(f"ℹ️ [Strategy {self.symbol}] إنشاء إشارة شراء (تعتمد على ML فقط)...")

        # min_signal_data_len should reflect the max lookback of all indicators
        min_signal_data_len = max(
            VOLUME_LOOKBACK_CANDLES,
            RSI_PERIOD,
            RSI_MOMENTUM_LOOKBACK_CANDLES,
            ENTRY_ATR_PERIOD,
            SUPERTRAND_PERIOD,
            TENKAN_PERIOD,
            KIJUN_PERIOD,
            SENKOU_SPAN_B_PERIOD,
            CHIKOU_LAG,
            FIB_SR_LOOKBACK_WINDOW,
            55
        ) + 1

        if df_processed is None or df_processed.empty or len(df_processed) < min_signal_data_len:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame فارغ أو قصير جدًا (<{min_signal_data_len})، لا يمكن إنشاء إشارة.")
            return None

        # Ensure all required columns for signal generation, including ML features, are present
        required_cols_for_signal = list(set(self.feature_columns_for_ml + [
            'close', 'atr', 'supertrend'
        ]))
        missing_cols = [col for col in required_cols_for_signal if col not in df_processed.columns]
        if missing_cols:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame يفتقد أعمدة مطلوبة للإشارة: {missing_cols}.")
            return None

        last_row = df_processed.iloc[-1]

        # --- Get current real-time price from ticker_data ---
        current_price = ticker_data.get(self.symbol)
        if current_price == None:
            logger.warning(f"⚠️ [Strategy {self.symbol}] السعر الحالي غير متاح من بيانات المؤشر. لا يمكن إنشاء إشارة.")
            return None

        if last_row[self.feature_columns_for_ml].isnull().values.any() or pd.isna(last_row.get('atr')) or pd.isna(last_row.get('supertrend')):
             logger.warning(f"⚠️ [Strategy {self.symbol}] البيانات التاريخية تحتوي على قيم NaN في أعمدة المؤشرات المطلوبة. لا يمكن إنشاء إشارة.")
             return None

        signal_details = {}

        # --- ML Model Prediction (Primary decision maker) ---
        ml_prediction_result_text = "N/A (لم يتم تحميل النموذج)"
        ml_is_bullish = False

        if self.ml_model:
            try:
                # Ensure the order of features for prediction matches the training order
                features_for_prediction = pd.DataFrame([last_row[self.feature_columns_for_ml].values], columns=self.feature_columns_for_ml)
                ml_pred = self.ml_model.predict(features_for_prediction)[0]
                if ml_pred == 1:
                    ml_is_bullish = True
                    ml_prediction_result_text = 'صعودي ✅'
                    logger.info(f"✨ [Strategy {self.symbol}] توقع نموذج ML صعودي.")
                else:
                    ml_prediction_result_text = 'هبوطي ❌'
                    logger.info(f"ℹ️ [Strategy {self.symbol}] توقع نموذج ML هبوطي. تم رفض الإشارة.")
            except Exception as ml_err:
                logger.error(f"❌ [Strategy {self.symbol}] خطأ في توقع نموذج ML: {ml_err}", exc_info=True)
                ml_prediction_result_text = "خطأ في التوقع (0)"

        signal_details['ML_Prediction'] = ml_prediction_result_text
        # Add values of relevant features to signal_details for logging/reporting
        signal_details['BTC_Trend_Feature_Value'] = last_row.get('btc_trend_feature', 0.0)
        signal_details['Supertrend_Direction_Value'] = last_row.get('supertrend_direction', 0)
        signal_details['Ichimoku_Cross_Signal'] = last_row.get('ichimoku_tenkan_kijun_cross_signal', 0)
        signal_details['Ichimoku_Price_Cloud_Position'] = last_row.get('ichimoku_price_cloud_position', 0)
        signal_details['Ichimoku_Cloud_Outlook'] = last_row.get('ichimoku_cloud_outlook', 0)
        signal_details['Fib_Above_50'] = last_row.get('is_price_above_fib_50', 0)
        signal_details['Dist_to_Recent_Low_Norm'] = last_row.get('price_distance_to_recent_low_norm', np.nan)
        signal_details['Dist_to_Recent_High_Norm'] = last_row.get('price_distance_to_recent_high_norm', np.nan)


        # If ML model is not bullish, or was not loaded, reject the signal early.
        if not ml_is_bullish:
            logger.info(f"ℹ️ [Strategy {self.symbol}] نموذج ML لم يتوقع صعودًا. تم رفض الإشارة.")
            return None

        # --- NEW: Add additional filters (these are hard-coded rules on top of ML prediction) ---
        # Filter 1: Supertrend must be bullish
        current_supertrend_direction = last_row.get('supertrend_direction')
        if current_supertrend_direction != 1:
            logger.info(f"ℹ️ [Strategy {self.symbol}] Supertrend ليس صعوديًا ({current_supertrend_direction}). تم رفض الإشارة.")
            signal_details['Supertrend_Filter'] = f'فشل: Supertrend ليس صعوديًا ({current_supertrend_direction})'
            return None
        else:
            signal_details['Supertrend_Filter'] = f'نجاح: Supertrend صعودي ({current_supertrend_direction})'

        # Filter 2: Bitcoin trend should not be strongly bearish (allow neutral or bullish)
        current_btc_trend = last_row.get('btc_trend_feature')
        if current_btc_trend == -1.0:
            logger.info(f"ℹ️ [Strategy {self.symbol}] اتجاه البيتكوين هبوطي ({current_btc_trend}). تم رفض الإشارة.")
            signal_details['BTC_Trend_Filter'] = f'فشل: اتجاه البيتكوين هبوطي ({current_btc_trend})'
            return None
        else:
            signal_details['BTC_Trend_Filter'] = f'نجاح: اتجاه البيتكوين ليس هبوطيًا ({current_btc_trend})'

        # --- Volume Check (Essential filter) ---
        volume_recent = fetch_recent_volume(self.symbol, interval=SIGNAL_GENERATION_TIMEFRAME, num_candles=VOLUME_LOOKBACK_CANDLES)
        if volume_recent < MIN_VOLUME_15M_USDT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] السيولة ({volume_recent:,.0f} USDT) أقل من الحد الأدنى المطلوب ({MIN_VOLUME_15M_USDT:,.0f} USDT). تم رفض الإشارة.")
            signal_details['Volume_Check'] = f'فشل: سيولة غير كافية ({volume_recent:,.0f} USDT)'
            return None
        else:
            signal_details['Volume_Check'] = f'نجاح: سيولة كافية ({volume_recent:,.0f} USDT)'


        current_atr = last_row.get('atr')
        current_supertrend_value = last_row.get('supertrend')

        if pd.isna(current_atr) or current_atr <= 0 or pd.isna(current_supertrend_value):
             logger.warning(f"⚠️ [Strategy {self.symbol}] قيمة ATR أو Supertrend غير صالحة ({current_atr}, {current_supertrend_value}) لحساب الهدف/الوقف. لا يمكن إنشاء إشارة.")
             return None

        # --- Calculate Initial Target ---
        target_multiplier = ENTRY_ATR_MULTIPLIER
        initial_target = current_price + (target_multiplier * current_atr)

        # --- Profit Margin Check (Essential filter) ---
        profit_margin_pct = ((initial_target / current_price) - 1) * 100 if current_price > 0 else 0
        if profit_margin_pct < MIN_PROFIT_MARGIN_PCT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] هامش الربح ({profit_margin_pct:.2f}%) أقل من الحد الأدنى المطلوب ({MIN_PROFIT_MARGIN_PCT:.2f}%). تم رفض الإشارة.")
            signal_details['Profit_Margin_Check'] = f'فشل: هامش ربح غير كافٍ ({profit_margin_pct:.2f}%)' # Changed profit_pct to profit_margin_pct
            return None
        else:
            signal_details['Profit_Margin_Check'] = f'نجاح: هامش ربح كافٍ ({profit_margin_pct:.2f}%)' # Changed profit_pct to profit_margin_pct

        # --- Calculate Initial Stop Loss ---
        # Ensure Supertrend value is below the current price for a long signal
        if current_supertrend_value < current_price:
            initial_stop_loss = current_supertrend_value
            signal_details['Stop_Loss_Method'] = f'Supertrend ({current_supertrend_value:.8g})'
        else:
            # Fallback to ATR if Supertrend is above entry (shouldn't happen if direction is 1, but as safety)
            stop_loss_atr_multiplier = 1.0
            initial_stop_loss = current_price - (stop_loss_atr_multiplier * current_atr)
            signal_details['Stop_Loss_Method'] = f'ATR Fallback ({initial_stop_loss:.8g})'
            logger.warning(f"⚠️ [Strategy {self.symbol}] Supertrend ({current_supertrend_value:.8g}) أعلى من سعر الدخول ({current_price:.8g}) على الرغم من الاتجاه الصعودي. استخدام ATR لوقف الخسارة.")

        # Ensure stop loss is not negative
        initial_stop_loss = max(0.00000001, initial_stop_loss)


        signal_output = {
            'symbol': self.symbol,
            'entry_price': float(f"{current_price:.8g}"),
            'initial_target': float(f"{initial_target:.8g}"),
            'current_target': float(f"{initial_target:.8g}"),
            'stop_loss': float(f"{initial_stop_loss:.8g}"),
            'r2_score': 1.0,
            'strategy_name': 'Scalping_ML_Enhanced_Filtered',
            'signal_details': signal_details,
            'volume_15m': volume_recent,
            'trade_value': TRADE_VALUE,
            'total_possible_score': 1.0
        }

        logger.info(f"✅ [Strategy {self.symbol}] تم تأكيد إشارة الشراء (ML + المرشحات). السعر: {current_price:.6f}, الهدف: {initial_target:.6f}, وقف الخسارة: {initial_stop_loss:.6f}, ATR: {current_atr:.6f}, الحجم: {volume_recent:,.0f}, توقع ML: {ml_prediction_result_text}, اتجاه BTC: {signal_details.get('BTC_Trend_Feature_Value')}, اتجاه Supertrend: {signal_details.get('Supertrend_Direction_Value')}, تقاطع Ichimoku: {signal_details.get('Ichimoku_Cross_Signal')}, موضع السعر السحابي: {signal_details.get('Ichimoku_Price_Cloud_Position')}, نظرة السحابة: {signal_details.get('Ichimoku_Cloud_Outlook')}, فيبوناتشي فوق 50: {signal_details.get('Fib_Above_50')}") # Changed ml_prediction_status to ml_prediction_result_text
        return signal_output


# ---------------------- Telegram Functions ----------------------
def send_telegram_message(target_chat_id: str, text: str, reply_markup: Optional[Dict] = None, parse_mode: str = 'Markdown', disable_web_page_preview: bool = True, timeout: int = 20) -> Optional[Dict]:
    """Sends a message via Telegram Bot API with improved error handling."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': str(target_chat_id),
        'text': text,
        'parse_mode': parse_mode,
        'disable_web_page_preview': disable_web_page_preview
    }
    if reply_markup:
        try:
            payload['reply_markup'] = json.dumps(convert_np_values(reply_markup))
        except (TypeError, ValueError) as json_err:
             logger.error(f"❌ [Telegram] فشل تحويل reply_markup إلى JSON: {json_err} - Markup: {reply_markup}")
             return None

    logger.debug(f"ℹ️ [Telegram] إرسال رسالة إلى {target_chat_id}...")
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        logger.info(f"✅ [Telegram] تم إرسال الرسالة بنجاح إلى {target_chat_id}.")
        return response.json()
    except requests.exceptions.Timeout:
         logger.error(f"❌ [Telegram] فشل إرسال الرسالة إلى {target_chat_id} (مهلة).")
         return None
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة إلى {target_chat_id} (خطأ HTTP: {http_err.response.status_code}).")
        try:
            error_details = http_err.response.json()
            logger.error(f"❌ [Telegram] تفاصيل خطأ API: {error_details}")
        except json.JSONDecodeError:
            logger.error(f"❌ [Telegram] تعذر فك تشفير استجابة الخطأ: {http_err.response.text}")
        return None
    except requests.exceptions.RequestException as req_err:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة إلى {target_chat_id} (خطأ في الطلب): {req_err}")
        return None
    except Exception as e:
         logger.error(f"❌ [Telegram] خطأ غير متوقع أثناء إرسال الرسالة: {e}", exc_info=True)
         return None

def edit_telegram_message(target_chat_id: str, message_id: int, text: str, reply_markup: Optional[Dict] = None, parse_mode: str = 'Markdown', disable_web_page_preview: bool = True, timeout: int = 20) -> Optional[Dict]:
    """Edits an existing message via Telegram Bot API."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText"
    payload = {
        'chat_id': str(target_chat_id),
        'message_id': message_id,
        'text': text,
        'parse_mode': parse_mode,
        'disable_web_page_preview': disable_web_page_preview
    }
    if reply_markup:
        try:
            payload['reply_markup'] = json.dumps(convert_np_values(reply_markup))
        except (TypeError, ValueError) as json_err:
            logger.error(f"❌ [Telegram] فشل تحويل reply_markup إلى JSON للتعديل: {json_err} - Markup: {reply_markup}")
            return None

    logger.debug(f"ℹ️ [Telegram] تعديل رسالة {message_id} في {target_chat_id}...")
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        logger.info(f"✅ [Telegram] تم تعديل الرسالة بنجاح {message_id} في {target_chat_id}.")
        return response.json()
    except requests.exceptions.Timeout:
        logger.error(f"❌ [Telegram] فشل تعديل الرسالة {message_id} في {target_chat_id} (مهلة).")
        return None
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"❌ [Telegram] فشل تعديل الرسالة {message_id} في {target_chat_id} (خطأ HTTP: {http_err.response.status_code}).")
        try:
            error_details = http_err.response.json()
            logger.error(f"❌ [Telegram] تفاصيل خطأ API: {error_details}")
        except json.JSONDecodeError:
            logger.error(f"❌ [Telegram] تعذر فك تشفير استجابة الخطأ للتعديل: {http_err.response.text}")
        return None
    except requests.exceptions.RequestException as req_err:
        logger.error(f"❌ [Telegram] فشل تعديل الرسالة {message_id} في {target_chat_id} (خطأ في الطلب): {req_err}")
        return None
    except Exception as e:
        logger.error(f"❌ [Telegram] خطأ غير متوقع أثناء تعديل الرسالة: {e}", exc_info=True)
        return None


def send_telegram_alert(signal_data: Dict[str, Any], timeframe: str) -> None:
    """Formats and sends a new trading signal alert to Telegram in Arabic, displaying the ML prediction and new indicator details."""
    logger.debug(f"ℹ️ [Telegram Alert] تنسيق وإرسال تنبيه الإشارة: {signal_data.get('symbol', 'N/A')}")
    try:
        entry_price = float(signal_data['entry_price'])
        target_price = float(signal_data['initial_target'])
        stop_loss_price = float(signal_data['stop_loss'])
        symbol = signal_data['symbol']
        strategy_name = signal_data.get('strategy_name', 'N/A')
        volume_15m = signal_data.get('volume_15m', 0.0)
        trade_value_signal = signal_data.get('trade_value', TRADE_VALUE)
        signal_details = signal_data.get('signal_details', {})

        profit_pct = ((target_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        loss_pct = ((stop_loss_price / entry_price) - 1) * 100 if entry_price > 0 else 0

        entry_fee = trade_value_signal * BINANCE_FEE_RATE
        exit_value_target = trade_value_signal * (1 + profit_pct / 100.0)
        exit_value_stoploss = trade_value_signal * (1 + loss_pct / 100.0)

        exit_fee_target = exit_value_target * BINANCE_FEE_RATE
        exit_fee_stoploss = exit_value_stoploss * BINANCE_FEE_RATE

        total_trade_fees_target = entry_fee + exit_fee_target
        total_trade_fees_stoploss = entry_fee + exit_fee_stoploss

        profit_usdt_gross = trade_value_signal * (profit_pct / 100)
        profit_usdt_net = profit_usdt_gross - total_trade_fees_target

        loss_usdt_gross = trade_value_signal * (loss_pct / 100)
        loss_usdt_net = loss_usdt_gross - total_trade_fees_stoploss

        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Corrected: Escape backslashes for f-string to work with Telegram Markdown
        safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[')
        # Backtick needs special handling if used within an f-string backtick section,
        # but here it's about making `safe_symbol` itself robust for Markdown.
        # If ` symbol ` could contain backticks that need escaping for markdown,
        # then `symbol.replace('`', '\\`')` would be needed, but it's more complex
        # to embed in an f-string that *also* uses backticks for code block.
        # For now, let's assume `safe_symbol` will be inside Markdown's inline code `...`
        # and not have literal backticks that need escaping in the source symbol.
        if '`' in safe_symbol: # A simple check to prevent issues if symbol itself contains backticks
            safe_symbol = safe_symbol.replace('`', '\\`')

        fear_greed = get_fear_greed_index()
        ml_prediction_status = signal_details.get('ML_Prediction', 'N/A')
        btc_trend_feature_value = signal_details.get('BTC_Trend_Feature_Value', 0.0)
        btc_trend_display = "صعودي 📈" if btc_trend_feature_value == 1.0 else ("هبوطي 📉" if btc_trend_feature_value == -1.0 else "محايد 🔄")
        supertrend_direction_value = signal_details.get('Supertrend_Direction_Value', 0)
        supertrend_display = "اتجاه صاعد ⬆️" if supertrend_direction_value == 1 else ("اتجاه هابط ⬇️" if supertrend_direction_value == -1 else "محايد ↔️")

        # Ichimoku display
        ichimoku_cross_signal = signal_details.get('Ichimoku_Cross_Signal', 0)
        ichimoku_cross_display = "تقاطع صعودي (TK) ✅" if ichimoku_cross_signal == 1 else ("تقاطع هبوطي (TK) ❌" if ichimoku_cross_signal == -1 else "لا يوجد تقاطع ↔️")
        ichimoku_price_cloud_pos = signal_details.get('Ichimoku_Price_Cloud_Position', 0)
        ichimoku_price_cloud_display = "فوق السحابة ☁️⬆️" if ichimoku_price_cloud_pos == 1 else ("تحت السحابة ☁️⬇️" if ichimoku_price_cloud_pos == -1 else "داخل السحابة ☁️↔️")
        ichimoku_cloud_outlook = signal_details.get('Ichimoku_Cloud_Outlook', 0)
        ichimoku_cloud_outlook_display = "سحابة صعودية 🟩" if ichimoku_cloud_outlook == 1 else ("سحابة هبوطية 🟥" if ichimoku_cloud_outlook == -1 else "سحابة مسطحة ⬜")

        # Fibonacci & S/R display (simplified, for context)
        fib_above_50 = signal_details.get('Fib_Above_50', 0)
        fib_above_50_display = "فوق 50% فيب 🟢" if fib_above_50 == 1 else "تحت 50% فيب 🔴"
        dist_to_recent_low = signal_details.get('Dist_to_Recent_Low_Norm', np.nan)
        dist_to_recent_high = signal_details.get('Dist_to_Recent_High_Norm', np.nan)

        sr_display_content = ""
        if not pd.isna(dist_to_recent_low) and not pd.isna(dist_to_recent_high):
            # Add newline directly to the string if it's present
            sr_display_content = f"  - المسافة إلى أدنى مستوى حديث: {dist_to_recent_low:.2f} | المسافة إلى أعلى مستوى حديث: {dist_to_recent_high:.2f}\n"

        message = f"""💡 *إشارة تداول جديدة (تعتمد على ML فقط)* 💡
——————————————
🪙 **الزوج:** `{safe_symbol}`
📈 **نوع الإشارة:** شراء (طويل)
🕰️ **الإطار الزمني:** {timeframe}
💧 **السيولة (آخر 15 دقيقة):** {volume_15m:,.0f} USDT
——————————————
➡️ **سعر الدخول المقترح:** `${entry_price:,.8g}`
🎯 **الهدف الأولي:** `${target_price:,.8g}`
🛑 **وقف الخسارة:** `${stop_loss_price:,.8g}`
💰 **الربح المتوقع (الإجمالي):** ({profit_pct:+.2f}% / ≈ ${profit_usdt_gross:+.2f})
💸 **الخسارة المتوقعة (الإجمالية):** ({loss_pct:+.2f}% / ≈ ${loss_usdt_gross:+.2f})
📈 **صافي الربح (المتوقع):** ${profit_usdt_net:+.2f}
📉 **صافي الخسارة (المتوقعة):** ${loss_usdt_net:+.2f}
——————————————
🤖 *توقع نموذج ML:* *{ml_prediction_status}* ✅ *الشروط الإضافية المحققة:* - فحص السيولة: {signal_details.get('Volume_Check', 'N/A')}
  - فحص هامش الربح: {signal_details.get('Profit_Margin_Check', 'N/A')}
  - فلتر Supertrend: {signal_details.get('Supertrend_Filter', 'N/A')}
  - فلتر اتجاه البيتكوين: {signal_details.get('BTC_Trend_Filter', 'N/A')}
——————————————
📊 *رؤى المؤشرات:* - مؤشر الخوف والجشع: {fear_greed}
  - اتجاه البيتكوين: {btc_trend_display}
  - اتجاه Supertrend: {supertrend_display}
  - إيشيموكو تينكان/كيجون: {ichimoku_cross_display}
  - سعر إيشيموكو مقابل السحابة: {ichimoku_price_cloud_display}
  - نظرة سحابة إيشيموكو: {ichimoku_cloud_outlook_display}
  - فيبوناتشي (50%): {fib_above_50_display}
{sr_display_content}——————————————
⏰ {timestamp_str}"""
        # For new signal alerts, we add a single button to view the full report
        reply_markup = {
            "inline_keyboard": [
                [{"text": "📊 عرض تقرير الأداء", "callback_data": "get_report"}]
            ]
        }

        send_telegram_message(CHAT_ID, message, reply_markup=reply_markup, parse_mode='Markdown')

    except KeyError as ke:
        logger.error(f"❌ [Telegram Alert] بيانات إشارة غير مكتملة للرمز {signal_data.get('symbol', 'N/A')}: مفتاح مفقود {ke}", exc_info=True)
    except Exception as e:
        logger.error(f"❌ [Telegram Alert] فشل إرسال تنبيه الإشارة للرمز {signal_data.get('symbol', 'N/A')}: {e}", exc_info=True)

def send_tracking_notification(details: Dict[str, Any]) -> None:
    """Formats and sends enhanced Telegram notifications for tracking events in Arabic."""
    symbol = details.get('symbol', 'N/A')
    signal_id = details.get('id', 'N/A')
    notification_type = details.get('type', 'unknown')
    message = ""
    # Corrected: Escape backslashes for f-string to work with Telegram Markdown
    safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[')
    if '`' in safe_symbol:
        safe_symbol = safe_symbol.replace('`', '\\`')

    closing_price = details.get('closing_price', 0.0)
    profit_pct = details.get('profit_pct', 0.0)
    current_price = details.get('current_price', 0.0)
    time_to_target = details.get('time_to_target', 'N/A')
    old_target = details.get('old_target', 0.0)
    new_target = details.get('new_target', 0.0)
    old_stop_loss = details.get('old_stop_loss', 0.0)
    new_stop_loss = details.get('new_stop_loss', 0.0)


    logger.debug(f"ℹ️ [Notification] تنسيق إشعار التتبع: ID={signal_id}, النوع={notification_type}, الرمز={symbol}")

    if notification_type == 'target_hit':
        message = f"""✅ *تم الوصول إلى الهدف (ID: {signal_id})* ——————————————
🪙 **الزوج:** `{safe_symbol}`
🎯 **سعر الإغلاق (الهدف):** `${closing_price:,.8g}`
💰 **الربح المحقق:** {profit_pct:+.2f}%
⏱️ **الوقت المستغرق:** {time_to_target}"""
    elif notification_type == 'stop_loss_hit':
        message = f"""🛑 *تم ضرب وقف الخسارة (ID: {signal_id})* ——————————————
🪙 **الزوج:** `{safe_symbol}`
📉 **سعر الإغلاق (وقف الخسارة):** `${closing_price:,.8g}`
💔 **الخسارة المحققة:** {profit_pct:+.2f}%
⏱️ **الوقت المستغرق:** {time_to_target}"""
    elif notification_type == 'target_stoploss_updated':
         update_parts_formatted = []
         if 'old_target' in details and 'new_target' in details:
             update_parts_formatted.append(f"  🎯 *الهدف:* `${old_target:,.8g}` -> `${new_target:,.8g}`")
         if 'old_stop_loss' in details and 'new_stop_loss' in details:
             update_parts_formatted.append(f"  🛑 *وقف الخسارة:* `${old_stop_loss:,.8g}` -> `${new_stop_loss:,.8g}`")

         update_block = "\n".join(update_parts_formatted)

         message = f"""🔄 *تحديث الإشارة (ID: {signal_id})* ——————————————
🪙 **الزوج:** `{safe_symbol}`
📈 **السعر الحالي:** `${current_price:,.8g}`
{update_block}
ℹ️ *تم التحديث بناءً على الزخم الصعودي المستمر أو ظروف السوق.*"""
    else:
        logger.warning(f"⚠️ [Notification] نوع إشعار غير معروف: {notification_type} للتفاصيل: {details}")
        return

    if message:
        send_telegram_message(CHAT_ID, message, parse_mode='Markdown')

def close_trade_by_id(signal_id: int, chat_id: str) -> None:
    """Closes a specific trade by ID at the current market price and sends a notification."""
    logger.info(f"ℹ️ [Close Trade] محاولة إغلاق الصفقة ID: {signal_id} بناءً على طلب المستخدم.")

    if not check_db_connection() or not conn:
        send_telegram_message(chat_id, "❌ لا يمكن إغلاق الصفقة، مشكلة في الاتصال بقاعدة البيانات.", parse_mode='Markdown')
        logger.error(f"❌ [Close Trade] لا يمكن إغلاق الصفقة ID: {signal_id} بسبب مشكلة في الاتصال بقاعدة البيانات.")
        return

    try:
        with conn.cursor() as cur_close:
            cur_close.execute("""
                SELECT id, symbol, entry_price, entry_time
                FROM signals
                WHERE id = %s AND achieved_target = FALSE;
            """, (signal_id,))
            signal_data = cur_close.fetchone()

            if not signal_data:
                send_telegram_message(chat_id, f"⚠️ الصفقة ID: *{signal_id}* غير موجودة أو تم إغلاقها بالفعل.", parse_mode='Markdown')
                logger.warning(f"⚠️ [Close Trade] الصفقة ID: {signal_id} غير موجودة أو تم إغلاقها بالفعل.")
                return

            symbol = signal_data['symbol']
            entry_price = float(signal_data['entry_price'])
            entry_time = signal_data['entry_time']

            current_price = ticker_data.get(symbol)
            if current_price == None:
                send_telegram_message(chat_id, f"❌ لا يمكن الحصول على السعر الحالي لـ {symbol}. يرجى المحاولة مرة أخرى.", parse_mode='Markdown')
                logger.error(f"❌ [Close Trade] السعر الحالي غير متاح لـ {symbol} لإغلاق الصفقة ID: {signal_id}.")
                return

            profit_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
            closed_at = datetime.now()
            time_to_close = closed_at - entry_time if entry_time else timedelta(0)
            time_to_close_str = str(time_to_close)

            cur_close.execute("""
                UPDATE signals
                SET achieved_target = TRUE, closing_price = %s, closed_at = %s, profit_percentage = %s, time_to_target = %s
                WHERE id = %s;
            """, (current_price, closed_at, profit_pct, time_to_close, signal_id))
            conn.commit()

            logger.info(f"✅ [Close Trade] تم إغلاق الصفقة ID: {signal_id} لـ {symbol} عند {current_price:.8g} (الربح: {profit_pct:+.2f}%، الوقت: {time_to_close_str}).")

            notification_message = f"""✅ *تم إغلاق الصفقة يدوياً (ID: {signal_id})*
——————————————
🪙 **الزوج:** `{symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[')}`
ℹ️ \\*تم الإغلاق بناءً على طلبك.\\*""" # FIXED: Escaped asterisks to avoid SyntaxError
            # Corrected: escape backticks if present in symbol
            if '`' in notification_message:
                notification_message = notification_message.replace('`', '\\`')

            send_telegram_message(chat_id, notification_message, parse_mode='Markdown')

    except psycopg2.Error as db_err:
        logger.error(f"❌ [Close Trade] خطأ في قاعدة البيانات أثناء إغلاق الصفقة ID: {signal_id}: {db_err}", exc_info=True)
        if conn: conn.rollback()
        send_telegram_message(chat_id, f"❌ خطأ في قاعدة البيانات أثناء إغلاق الصفقة ID: *{signal_id}*.", parse_mode='Markdown')
    except Exception as e:
        logger.error(f"❌ [Close Trade] خطأ غير متوقع أثناء إغلاق الصفقة ID: {signal_id}: {e}", exc_info=True)
        send_telegram_message(chat_id, f"❌ حدث خطأ غير متوقع أثناء إغلاق الصفقة ID: *{signal_id}*.", parse_mode='Markdown')

# ---------------------- Database Functions (Insert and Update) ----------------------
def insert_signal_into_db(signal: Dict[str, Any]) -> bool:
    """Inserts a new signal into the signals table with the weighted score and entry time."""
    if not check_db_connection() or not conn:
        logger.error(f"❌ [DB Insert] فشل إدراج الإشارة {signal.get('symbol', 'N/A')} بسبب مشكلة في الاتصال بقاعدة البيانات.")
        return False

    symbol = signal.get('symbol', 'N/A')
    logger.debug(f"ℹ️ [DB Insert] محاولة إدراج إشارة لـ {symbol}...")
    try:
        signal_prepared = convert_np_values(signal)
        signal_details_json = json.dumps(signal_prepared.get('signal_details', {}))

        with conn.cursor() as cur_ins:
            insert_query = sql.SQL("""
                INSERT INTO signals
                 (symbol, entry_price, initial_target, current_target, stop_loss,
                 r2_score, strategy_name, signal_details, volume_15m, entry_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW());
            """)
            cur_ins.execute(insert_query, (
                signal_prepared['symbol'],
                signal_prepared['entry_price'],
                signal_prepared['initial_target'],
                signal_prepared['current_target'],
                signal_prepared['stop_loss'],
                signal_prepared.get('r2_score'),
                signal_prepared.get('strategy_name', 'unknown'),
                signal_details_json,
                signal_prepared.get('volume_15m')
            ))
        conn.commit()
        logger.info(f"✅ [DB Insert] تم إدراج إشارة لـ {symbol} في قاعدة البيانات (النتيجة: {signal_prepared.get('r2_score')}).")
        return True
    except psycopg2.Error as db_err:
        logger.error(f"❌ [DB Insert] خطأ في قاعدة البيانات أثناء إدراج الإشارة لـ {symbol}: {db_err}")
        if conn: conn.rollback()
        return False
    except (TypeError, ValueError) as convert_err:
         logger.error(f"❌ [DB Insert] خطأ في تحويل بيانات الإشارة قبل الإدراج لـ {symbol}: {convert_err} - بيانات الإشارة: {signal}")
         if conn: conn.rollback()
         return False
    except Exception as e:
        logger.error(f"❌ [DB Insert] خطأ غير متوقع أثناء إدراج الإشارة لـ {symbol}: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

# ---------------------- Open Signal Tracking Function ----------------------
def track_signals() -> None:
    """Tracks open signals and checks targets. Calculates time to target upon hit."""
    logger.info("ℹ️ [Tracker] بدء عملية تتبع الإشارات المفتوحة...")
    while True:
        active_signals_summary: List[str] = []
        processed_in_cycle = 0
        try:
            if not check_db_connection() or not conn:
                logger.warning("⚠️ [Tracker] تخطي دورة التتبع بسبب مشكلة في الاتصال بقاعدة البيانات.")
                time.sleep(15)
                continue

            with conn.cursor() as track_cur:
                 track_cur.execute("""
                    SELECT id, symbol, entry_price, initial_target, current_target, entry_time, stop_loss
                    FROM signals
                    WHERE achieved_target = FALSE AND closing_price is NULL;
                """)
                 open_signals: List[Dict] = track_cur.fetchall()

            if not open_signals:
                time.sleep(10)
                continue

            logger.debug(f"ℹ️ [Tracker] تتبع {len(open_signals)} إشارة مفتوحة...")

            for signal_row in open_signals:
                signal_id = signal_row['id']
                symbol = signal_row['symbol']
                processed_in_cycle += 1
                update_executed = False

                try:
                    entry_price = float(signal_row['entry_price'])
                    entry_time = signal_row['entry_time']
                    current_target = float(signal_row["current_target"])
                    current_stop_loss = float(signal_row["stop_loss"]) if signal_row.get("stop_loss") is not None else None

                    current_price = ticker_data.get(symbol)

                    if current_price == None:
                         logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): السعر الحالي غير متاح في بيانات المؤشر.")
                         continue

                    active_signals_summary.append(f"{symbol}({signal_id}): P={current_price:.4f} T={current_target:.4f} SL={current_stop_loss if current_stop_loss else 'N/A'}")

                    update_query: Optional[sql.SQL] = None
                    update_params: Tuple = ()
                    log_message: Optional[str] = None
                    notification_details: Dict[str, Any] = {
                        'symbol': symbol,
                        'id': signal_id,
                        'current_price': current_price,
                        'entry_price': entry_price,
                        'current_target': current_target,
                        'stop_loss': current_stop_loss
                    }


                    # --- Check and Update Logic ---
                    # 0. Check for Stop Loss Hit (PRIORITY)
                    if current_stop_loss is not None and current_price <= current_stop_loss:
                        profit_pct = ((current_stop_loss / entry_price) - 1) * 100 if entry_price > 0 else 0
                        closed_at = datetime.now()
                        time_to_close = closed_at - entry_time if entry_time else timedelta(0)
                        time_to_close_str = str(time_to_close)

                        update_query = sql.SQL("UPDATE signals SET achieved_target = FALSE, closing_price = %s, closed_at = %s, profit_percentage = %s, time_to_target = %s WHERE id = %s;")
                        update_params = (current_stop_loss, closed_at, profit_pct, time_to_close, signal_id)
                        log_message = f"🛑 [Tracker] {symbol}(ID:{signal_id}): تم ضرب وقف الخسارة عند {current_stop_loss:.8g} (الخسارة: {profit_pct:+.2f}%، الوقت: {time_to_close_str})."
                        notification_details.update({
                            'type': 'stop_loss_hit',
                            'closing_price': current_stop_loss,
                            'profit_pct': profit_pct,
                            'time_to_target': time_to_close_str
                        })
                        update_executed = True

                    # 1. Check for Target Hit (Only if Stop Loss not hit)
                    if not update_executed and current_price >= current_target:
                        profit_pct = ((current_target / entry_price) - 1) * 100 if entry_price > 0 else 0
                        closed_at = datetime.now()
                        time_to_target_duration = closed_at - entry_time if entry_time else timedelta(0)
                        time_to_target_str = str(time_to_target_duration)

                        update_query = sql.SQL("UPDATE signals SET achieved_target = TRUE, closing_price = %s, closed_at = %s, profit_percentage = %s, time_to_target = %s WHERE id = %s;")
                        update_params = (current_target, closed_at, profit_pct, time_to_target_duration, signal_id)
                        log_message = f"🎯 [Tracker] {symbol}(ID:{signal_id}): تم الوصول إلى الهدف عند {current_target:.8g} (الربح: {profit_pct:+.2f}%، الوقت: {time_to_target_str})."
                        notification_details.update({
                            'type': 'target_hit',
                            'closing_price': current_target,
                            'profit_pct': profit_pct,
                            'time_to_target': time_to_target_str
                        })
                        update_executed = True

                    # 2. Check for Target/Stop Loss Update (Dynamic Target & Trailing Stop) (Only if not closed)
                    if not update_executed:
                        # Condition to check for update: e.g., price made significant progress towards target
                        progress_to_target = (current_price - entry_price) / (current_target - entry_price) if (current_target - entry_price) != 0 else 0
                        should_check_update = current_price >= current_target * (1 - TARGET_APPROACH_THRESHOLD_PCT)

                        if should_check_update:
                             logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): السعر قريب من الهدف ({current_price:.8g} مقابل {current_target:.8g}). التحقق من إشارة الاستمرارية لتحديث الهدف/وقف الخسارة...")

                             df_continuation = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)

                             if df_continuation is not None and not df_continuation.empty:
                                 continuation_strategy = ScalpingTradingStrategy(symbol)
                                 if continuation_strategy.ml_model == None:
                                     logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): نموذج ML لم يتم تحميله لإستراتيجية الاستمرارية. تخطي تحديث الهدف/وقف الخسارة.")
                                     continue

                                 df_continuation_indicators = continuation_strategy.populate_indicators(df_continuation)

                                 if df_continuation_indicators is not None and not df_continuation_indicators.empty:
                                     # Use generate_buy_signal to check if conditions *still* hold for a buy
                                     # We don't need the full signal output, just whether it passes filters
                                     continuation_signal_check = continuation_strategy.generate_buy_signal(df_continuation_indicators)

                                     if continuation_signal_check:
                                         latest_row = df_continuation_indicators.iloc[-1]
                                         current_atr_for_update = latest_row.get('atr')
                                         current_supertrend_for_update = latest_row.get('supertrend')

                                         if pd.notna(current_atr_for_update) and current_atr_for_update > 0 and pd.notna(current_supertrend_for_update):
                                             # --- Calculate Potential New Target ---
                                             potential_new_target = current_price + (ENTRY_ATR_MULTIPLIER * current_atr_for_update)

                                             # --- Calculate Potential New Stop Loss (Trailing Stop) ---
                                             # Option 1: Trail using Supertrend
                                             potential_new_stop_loss = current_supertrend_for_update

                                             # Ensure new stop loss is higher than the current one AND below current price
                                             new_stop_loss_valid = potential_new_stop_loss > (current_stop_loss or 0) and potential_new_stop_loss < current_price

                                             # --- Decide whether to update Target and/or Stop Loss ---
                                             update_target = potential_new_target > current_target
                                             update_stop_loss = new_stop_loss_valid

                                             if update_target or update_stop_loss:
                                                 old_target = current_target
                                                 old_stop_loss = current_stop_loss
                                                 new_target = potential_new_target if update_target else current_target
                                                 new_stop_loss = potential_new_stop_loss if update_stop_loss else current_stop_loss

                                                 update_fields = []
                                                 update_params_list = []
                                                 log_parts = []
                                                 notification_details.update({'type': 'target_stoploss_updated'})

                                                 if update_target:
                                                     update_fields.append("current_target = %s")
                                                     update_params_list.append(new_target)
                                                     log_parts.append(f"الهدف من {old_target:.8g} إلى {new_target:.8g}")
                                                     notification_details['old_target'] = old_target
                                                     notification_details['new_target'] = new_target

                                                 if update_stop_loss:
                                                     update_fields.append("stop_loss = %s")
                                                     update_params_list.append(new_stop_loss)
                                                     log_parts.append(f"وقف الخسارة من {old_stop_loss if old_stop_loss else 'N/A'} إلى {new_stop_loss:.8g}")
                                                     notification_details['old_stop_loss'] = old_stop_loss
                                                     notification_details['new_stop_loss'] = new_stop_loss

                                                 update_params_list.append(signal_id)
                                                 update_query = sql.SQL(f"UPDATE signals SET {', '.join(update_fields)} WHERE id = %s;")
                                                 update_params = tuple(update_params_list)
                                                 log_message = f"↔️ [Tracker] {symbol}(ID:{signal_id}): تم تحديث {' و '.join(log_parts)} بناءً على استمرارية الإشارة."
                                                 update_executed = True
                                             else:
                                                 logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): تم اكتشاف إشارة استمرارية، لكن الهدف الجديد ({potential_new_target:.8g}) أو وقف الخسارة الجديد ({potential_new_stop_loss:.8g}) لا يستدعي تحديثًا.")
                                         else:
                                             logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): لا يمكن حساب هدف/وقف خسارة جديد بسبب ATR/Supertrend غير صالحين ({current_atr_for_update}, {current_supertrend_for_update}) من بيانات الاستمرارية.")
                                     else:
                                         logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): السعر قريب من الهدف، لكن إشارة الاستمرارية لم يتم تأكيدها (فشلت المرشحات أو توقع ML). عدم تحديث الهدف/وقف الخسارة.")
                                 else:
                                     logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): فشل في تعبئة المؤشرات للتحقق من الاستمرارية.")
                             else:
                                 logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): تعذر جلب البيانات التاريخية للتحقق من الاستمرارية.")


                    if update_executed and update_query:
                        try:
                             with conn.cursor() as update_cur:
                                  update_cur.execute(update_query, update_params)
                             conn.commit()
                             if log_message: logger.info(log_message)
                             if notification_details.get('type'):
                                send_tracking_notification(notification_details)
                        except psycopg2.Error as db_err:
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ في قاعدة البيانات أثناء التحديث: {db_err}")
                            if conn: conn.rollback()
                        except Exception as exec_err:
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ غير متوقع أثناء تنفيذ التحديث/الإشعار: {exec_err}", exc_info=True)
                            if conn: conn.rollback()

                except (TypeError, ValueError) as convert_err:
                    logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ في تحويل قيم الإشارة الأولية: {convert_err}")
                    continue
                except Exception as inner_loop_err:
                     logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ غير متوقع أثناء معالجة الإشارة: {inner_loop_err}", exc_info=True)
                     continue

            if active_signals_summary:
                logger.debug(f"ℹ️ [Tracker] نهاية حالة الدورة ({processed_in_cycle} معالجة): {'; '.join(active_signals_summary)}")

            time.sleep(3)

        except psycopg2.Error as db_cycle_err:
             logger.error(f"❌ [Tracker] خطأ في قاعدة البيانات في دورة التتبع الرئيسية: {db_cycle_err}. محاولة إعادة الاتصال...")
             if conn: conn.rollback()
             time.sleep(30)
             check_db_connection()
        except Exception as cycle_err:
            logger.error(f"❌ [Tracker] خطأ غير متوقع في دورة تتبع الإشارة: {cycle_err}", exc_info=True)
            time.sleep(30)

def get_interval_minutes(interval: str) -> int:
    """Helper function to convert Binance interval string to minutes."""
    if interval.endswith('m'):
        return int(interval[:-1])
    elif interval.endswith('h'):
        return int(interval[:-1]) * 60
    elif interval.endswith('d'):
        return int(interval[:-1]) * 60 * 24
    return 0


# ---------------------- Flask Service (Optional for Webhook) ----------------------
app = Flask(__name__)

@app.route('/')
def home() -> Response:
    """Simple home page to show the bot is running."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ws_alive = ws_thread.is_alive() if 'ws_thread' in globals() and ws_thread else False
    tracker_alive = tracker_thread.is_alive() if 'tracker_thread' in globals() and tracker_thread else False
    main_bot_alive = main_bot_thread.is_alive() if 'main_bot_thread' in globals() and main_bot_thread else False
    status = "يعمل" if ws_alive and tracker_alive and main_bot_alive else "يعمل جزئياً"
    return Response(f"📈 بوت إشارات العملات الرقمية ({status}) - آخر تحقق: {now}", status=200, mimetype='text/plain')

@app.route('/favicon.ico')
def favicon() -> Response:
    """Handles favicon request to avoid 404 errors in logs."""
    return Response(status=204)

@app.route('/webhook', methods=['POST'])
def webhook() -> Tuple[str, int]:
    """Handles incoming requests from Telegram (like button presses and commands)."""
    # Only process webhook if WEBHOOK_URL is configured
    if not WEBHOOK_URL:
        logger.warning("⚠️ [Flask] تم استلام طلب Webhook، ولكن WEBHOOK_URL غير مكوّن. تجاهل الطلب.")
        return "Webhook غير مكوّن", 200

    if not request.is_json:
        logger.warning("⚠️ [Flask] تم استلام طلب Webhook غير بصيغة JSON.")
        return "تنسيق طلب غير صالح", 400

    try:
        data = request.get_json()
        logger.info(f"✅ [Flask] تم استلام بيانات Webhook. حجم البيانات: {len(json.dumps(data))} بايت.")
        logger.debug(f"ℹ️ [Flask] بيانات Webhook كاملة: {json.dumps(data)}")


        if 'callback_query' in data:
            callback_query = data['callback_query']
            callback_id = callback_query['id']
            callback_data = callback_query.get('data')
            message_info = callback_query.get('message')

            logger.info(f"ℹ️ [Flask] تم استلام استعلام رد الاتصال. المعرف: {callback_id}, البيانات: '{callback_data}'")

            if not message_info or not callback_data:
                 logger.warning(f"⚠️ [Flask] استعلام رد الاتصال (المعرف: {callback_id}) يفتقد الرسالة أو البيانات. تجاهل.")
                 try:
                     ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                     requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
                 except Exception as ack_err:
                     logger.warning(f"⚠️ [Flask] فشل تأكيد استعلام رد الاتصال غير الصالح {callback_id}: {ack_err}")
                 return "OK", 200
            chat_id_callback = message_info.get('chat', {}).get('id')
            if not chat_id_callback:
                 logger.warning(f"⚠️ [Flask] استعلام رد الاتصال (المعرف: {callback_id}) يفتقد معرف الدردشة. تجاهل.")
                 try:
                     ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                     requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
                 except Exception as ack_err:
                     logger.warning(f"⚠️ [Flask] فشل تأكيد استعلام رد الاتصال غير الصالح {callback_id}: {ack_err}")
                 return "OK", 200


            message_id = message_info['message_id']
            user_info = callback_query.get('from', {})
            user_id = user_info.get('id')
            username = user_info.get('username', 'N/A')

            logger.info(f"ℹ️ [Flask] معالجة استعلام رد الاتصال: البيانات='{callback_data}', المستخدم={username}({user_id}), الدردشة={chat_id_callback}")

            try:
                # Always acknowledge the callback query to remove the loading animation from the button
                ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
                logger.debug(f"✅ [Flask] تم تأكيد استعلام رد الاتصال {callback_id}.")
            except Exception as ack_err:
                 logger.warning(f"⚠️ [Flask] فشل تأكيد استعلام رد الاتصال {callback_id}: {ack_err}")

            if callback_data == "get_report":
                logger.info(f"ℹ️ [Flask] تم استلام طلب 'get_report' من الدردشة {chat_id_callback}. إنشاء التقرير...")
                report_content, reply_markup = generate_performance_report()
                if report_content:
                    # Edit the original message that contained the button
                    report_thread = Thread(target=lambda: edit_telegram_message(chat_id_callback, message_id, report_content, reply_markup=reply_markup, parse_mode='Markdown'))
                    report_thread.start()
                    logger.info(f"✅ [Flask] تم بدء مؤشر ترابط تحديث التقرير للدردشة {chat_id_callback}.")
            elif callback_data and callback_data.startswith("exit_trade_"):
                signal_id_str = callback_data.replace("exit_trade_", "")
                try:
                    signal_id = int(signal_id_str)
                    logger.info(f"ℹ️ [Flask] تم استلام طلب 'exit_trade' للصفقة ID: {signal_id} من الدردشة {chat_id_callback}.")
                    # Call close_trade_by_id in a new thread to avoid blocking the webhook
                    close_thread = Thread(target=close_trade_by_id, args=(signal_id, chat_id_callback,))
                    close_thread.start()
                    logger.info(f"✅ [Flask] تم بدء مؤشر ترابط إغلاق الصفقة ID: {signal_id}.")
                    # Immediately re-send updated report to reflect change
                    report_content, reply_markup = generate_performance_report()
                    if report_content:
                        edit_telegram_message(chat_id_callback, message_id, report_content, reply_markup=reply_markup, parse_mode='Markdown')
                except ValueError:
                    logger.error(f"❌ [Flask] callback_data غير صالح لمعرف الصفقة: {callback_data}")
                    send_telegram_message(chat_id_callback, "❌ معرف الصفقة غير صالح.", parse_mode='Markdown')
            else:
                logger.warning(f"⚠️ [Flask] تم استلام بيانات رد اتصال غير معالجة: '{callback_data}'")


        elif 'message' in data:
            message_data = data['message']
            chat_info = message_data.get('chat')
            user_info = message_data.get('from', {})
            text_msg = message_data.get('text', '').strip()

            if not chat_info or not text_msg:
                 logger.debug("ℹ️ [Flask] تم استلام رسالة بدون معلومات دردشة أو نص.")
                 return "OK", 200

            chat_id_msg = chat_info['id']
            user_id = user_info.get('id')
            username = user_info.get('username', 'N/A')

            logger.info(f"ℹ️ [Flask] تم استلام رسالة: النص='{text_msg}', المستخدم={username}({user_id}), الدردشة={chat_id_msg}")

            if text_msg.lower() == '/report':
                 report_content, reply_markup = generate_performance_report()
                 if report_content:
                    report_thread = Thread(target=lambda: send_telegram_message(chat_id_msg, report_content, reply_markup=reply_markup, parse_mode='Markdown'))
                    report_thread.start()
            elif text_msg.lower() == '/status':
                 status_thread = Thread(target=handle_status_command, args=(chat_id_msg,))
                 status_thread.start()

        else:
            logger.debug("ℹ️ [Flask] تم استلام بيانات Webhook بدون 'callback_query' أو 'message'.")

        return "OK", 200
    except Exception as e:
         logger.error(f"❌ [Flask] خطأ في معالجة Webhook: {e}", exc_info=True)
         return "خطأ داخلي في الخادم", 500

def handle_status_command(chat_id_msg: int) -> None:
    """Separate function to handle /status command to avoid blocking the Webhook."""
    logger.info(f"ℹ️ [Flask Status] معالجة أمر /status للدردشة {chat_id_msg}")
    status_msg = "⏳ جلب الحالة..."
    msg_sent = send_telegram_message(chat_id_msg, status_msg)
    if not (msg_sent and msg_sent.get('ok')):
         logger.error(f"❌ [Flask Status] فشل إرسال رسالة الحالة الأولية إلى {chat_id_msg}")
         return
    message_id_to_edit = msg_sent['result']['message_id'] if msg_sent and msg_sent.get('result') else None

    if message_id_to_edit == None:
        logger.error(f"❌ [Flask Status] فشل الحصول على message_id لتحديث الحالة في الدردشة {chat_id_msg}")
        return


    try:
        open_count = 0
        if check_db_connection() and conn:
            with conn.cursor() as status_cur:
                status_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE;")
                open_count = (status_cur.fetchone() or {}).get('count', 0)

        ws_status = 'نشط ✅' if 'ws_thread' in globals() and ws_thread and ws_thread.is_alive() else 'غير نشط ❌'
        tracker_status = 'نشط ✅' if 'tracker_thread' in globals() and tracker_thread and tracker_thread.is_alive() else 'غير نشط ❌'
        main_bot_alive = 'نشط ✅' if 'main_bot_thread' in globals() and main_bot_thread and main_bot_thread.is_alive() else 'غير نشط ❌'
        final_status_msg = f"""🤖 *حالة البوت:* - تتبع الأسعار (WS): {ws_status}
- تتبع الإشارات: {tracker_status}
- حلقة البوت الرئيسية: {main_bot_alive}
- الإشارات النشطة: *{open_count}* / {MAX_OPEN_TRADES}
- وقت الخادم الحالي: {datetime.now().strftime('%H:%M:%S')}"""

        edit_telegram_message(chat_id_msg, message_id_to_edit, final_status_msg, parse_mode='Markdown')

    except Exception as status_err:
        logger.error(f"❌ [Flask Status] خطأ أثناء جلب/تعديل تفاصيل الحالة للدردشة {chat_id_msg}: {status_err}", exc_info=True)
        send_telegram_message(chat_id_msg, "❌ حدث خطأ أثناء جلب تفاصيل الحالة.")


def run_flask() -> None:
    """Runs the Flask application to listen for the Webhook using a production server if available."""
    host = "0.0.0.0"
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"ℹ️ [Flask] بدء تطبيق Flask على {host}:{port}...")
    try:
        from waitress import serve
        logger.info("✅ [Flask] استخدام خادم 'waitress'.")
        serve(app, host=host, port=port, threads=6)
    except ImportError:
         logger.warning("⚠️ [Flask] 'waitress' غير مثبت. الرجوع إلى خادم تطوير Flask (لا يوصى به للإنتاج).")
         try:
             app.run(host=host, port=port)
         except Exception as flask_run_err:
              logger.critical(f"❌ [Flask] فشل بدء خادم التطوير: {flask_run_err}", exc_info=True)
    except Exception as serve_err:
         logger.critical(f"❌ [Flask] فشل بدء الخادم (waitress؟): {serve_err}", exc_info=True)

# ---------------------- Main Loop and Check Function ----------------------
def main_loop() -> None:
    """Main loop to scan pairs and generate signals."""
    symbols_to_scan = get_crypto_symbols()
    if not symbols_to_scan:
        logger.critical("❌ [Main] لم يتم تحميل أو التحقق من أي رموز صالحة. لا يمكن المتابعة.")
        return

    logger.info(f"✅ [Main] تم تحميل {len(symbols_to_scan)} رمزًا صالحًا للمسح.")
    last_full_scan_time = time.time()

    while True:
        try:
            scan_start_time = time.time()
            logger.info("+" + "-"*60 + "+")
            logger.info(f"🔄 [Main] بدء دورة مسح السوق - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("+" + "-"*60 + "+")

            if not check_db_connection() or not conn:
                logger.error("❌ [Main] تخطي دورة المسح بسبب فشل الاتصال بقاعدة البيانات.")
                time.sleep(60)
                continue

            open_count = 0
            try:
                 with conn.cursor() as cur_check:
                    cur_check.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE;")
                    open_count = (cur_check.fetchone() or {}).get('count', 0)
            except psycopg2.Error as db_err:
                 logger.error(f"❌ [Main] خطأ في قاعدة البيانات أثناء التحقق من عدد الإشارات المفتوحة: {db_err}. تخطي الدورة.")
                 if conn: conn.rollback()
                 time.sleep(60)
                 continue

            logger.info(f"ℹ️ [Main] الإشارات المفتوحة حاليًا: {open_count} / {MAX_OPEN_TRADES}")
            if open_count >= MAX_OPEN_TRADES:
                logger.info(f"⚠️ [Main] تم الوصول إلى الحد الأقصى لعدد الإشارات المفتوحة. في انتظار...")
                time.sleep(get_interval_minutes(SIGNAL_GENERATION_TIMEFRAME) * 60)
                continue

            processed_in_loop = 0
            signals_generated_in_loop = 0
            slots_available = MAX_OPEN_TRADES - open_count

            for symbol in symbols_to_scan:
                 if slots_available <= 0:
                      logger.info(f"ℹ️ [Main] تم الوصول إلى الحد الأقصى للتداولات المفتوحة ({MAX_OPEN_TRADES}) أثناء المسح. إيقاف مسح الرموز لهذه الدورة.")
                      break

                 processed_in_loop += 1
                 logger.debug(f"🔍 [Main] مسح {symbol} ({processed_in_loop}/{len(symbols_to_scan)})...")

                 try:
                    with conn.cursor() as symbol_cur:
                        symbol_cur.execute("SELECT 1 FROM signals WHERE symbol = %s AND achieved_target = FALSE LIMIT 1;", (symbol,))
                        if symbol_cur.fetchone():
                            continue

                    df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_hist == None or df_hist.empty:
                        continue

                    strategy = ScalpingTradingStrategy(symbol)
                    # Check if ML model was loaded successfully for this symbol
                    if strategy.ml_model == None:
                        logger.warning(f"⚠️ [Main] تخطي {symbol} لأن نموذج ML الخاص به لم يتم تحميله بنجاح.")
                        continue

                    df_indicators = strategy.populate_indicators(df_hist)
                    if df_indicators == None:
                        continue

                    potential_signal = strategy.generate_buy_signal(df_indicators)

                    if potential_signal:
                        logger.info(f"✨ [Main] تم العثور على إشارة محتملة لـ {symbol}! التحقق النهائي والإدراج...")
                        with conn.cursor() as final_check_cur:
                             final_check_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE;")
                             final_open_count = (final_check_cur.fetchone() or {}).get('count', 0)

                             if final_open_count < MAX_OPEN_TRADES:
                                 if insert_signal_into_db(potential_signal):
                                     send_telegram_alert(potential_signal, SIGNAL_GENERATION_TIMEFRAME)
                                     signals_generated_in_loop += 1
                                     slots_available -= 1
                                     time.sleep(2)
                                 else:
                                     logger.error(f"❌ [Main] فشل إدراج الإشارة لـ {symbol} في قاعدة البيانات.")
                             else:
                                 logger.warning(f"⚠️ [Main] تم الوصول إلى الحد الأقصى للتداولات المفتوحة ({final_open_count}) قبل إدراج الإشارة لـ {symbol}. تم تجاهل الإشارة.")
                                 break

                 except psycopg2.Error as db_loop_err:
                      logger.error(f"❌ [Main] خطأ في قاعدة البيانات أثناء معالجة الرمز {symbol}: {db_loop_err}. الانتقال إلى التالي...")
                      if conn: conn.rollback()
                      continue
                 except Exception as symbol_proc_err:
                      logger.error(f"❌ [Main] خطأ عام في معالجة الرمز {symbol}: {symbol_proc_err}", exc_info=True)
                      continue

                 time.sleep(0.1)

            scan_duration = time.time() - scan_start_time
            logger.info(f"🏁 [Main] انتهت دورة المسح. الإشارات التي تم إنشاؤها: {signals_generated_in_loop}. مدة المسح: {scan_duration:.2f} ثانية.")
            frame_minutes = get_interval_minutes(SIGNAL_GENERATION_TIMEFRAME)
            wait_time = max(frame_minutes * 60, 120 - scan_duration)
            logger.info(f"⏳ [Main] انتظار {wait_time:.1f} ثانية للدورة التالية...")
            time.sleep(wait_time)

        except KeyboardInterrupt:
             logger.info("🛑 [Main] طلب إيقاف (KeyboardInterrupt). إيقاف التشغيل...")
             break
        except psycopg2.Error as db_main_err:
             logger.error(f"❌ [Main] خطأ فادح في قاعدة البيانات في الحلقة الرئيسية: {db_main_err}. محاولة إعادة الاتصال...")
             if conn: conn.rollback()
             time.sleep(60)
             try:
                 init_db()
             except Exception as recon_err:
                 logger.critical(f"❌ [Main] فشل إعادة الاتصال بقاعدة البيانات: {recon_err}. الخروج...")
                 break
        except Exception as main_err:
            logger.error(f"❌ [Main] خطأ غير متوقع في الحلقة الرئيسية: {main_err}", exc_info=True)
            logger.info("ℹ️ [Main] انتظار 120 ثانية قبل إعادة المحاولة...")
            time.sleep(120)

def cleanup_resources() -> None:
    """Closes used resources like the database connection."""
    global conn
    logger.info("ℹ️ [Cleanup] إغلاق الموارد...")
    if conn:
        try:
            conn.close()
            logger.info("✅ [DB] تم إغلاق الاتصال بقاعدة البيانات.")
        except Exception as close_err:
            logger.error(f"⚠️ [DB] خطأ أثناء إغلاق الاتصال بقاعدة البيانات: {close_err}")
    logger.info("✅ [Cleanup] اكتمال تنظيف الموارد.")


# ---------------------- Main Entry Point ----------------------
if __name__ == "__main__":
    logger.info("🚀 بدء بوت إشارات تداول العملات الرقمية...")
    logger.info(f"الوقت المحلي: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | التوقيت العالمي المنسق: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    ws_thread: Optional[Thread] = None
    tracker_thread: Optional[Thread] = None
    flask_thread: Optional[Thread] = None
    main_bot_thread: Optional[Thread] = None

    try:
        # 1. Initialize the database first
        init_db()

        # 2. Start WebSocket Ticker
        ws_thread = Thread(target=run_ticker_socket_manager, daemon=True, name="WebSocketThread")
        ws_thread.start()
        logger.info("✅ [Main] تم بدء مؤشر WebSocket.")
        logger.info("ℹ️ [Main] انتظار 5 ثوانٍ لتهيئة WebSocket...")
        time.sleep(5)
        if not ticker_data:
             logger.warning("⚠️ [Main] لم يتم استلام أي بيانات أولية من WebSocket بعد 5 ثوانٍ.")
        else:
             logger.info(f"✅ [Main] تم استلام بيانات WebSocket الأولية لـ {len(ticker_data)} رمزًا.")


        # 3. Start Signal Tracker
        tracker_thread = Thread(target=track_signals, daemon=True, name="TrackerThread")
        tracker_thread.start()
        logger.info("✅ [Main] تم بدء متتبع الإشارات.")

        # 4. Start the main bot logic in a separate thread
        main_bot_thread = Thread(target=main_loop, daemon=True, name="MainBotLoopThread")
        main_bot_thread.start()
        logger.info("✅ [Main] تم بدء حلقة البوت الرئيسية في مؤشر ترابط منفصل.")

        # 5. Start Flask Server (ALWAYS run, daemon=False so it keeps the main program alive)
        flask_thread = Thread(target=run_flask, daemon=False, name="FlaskThread")
        flask_thread.start()
        logger.info("✅ [Main] تم بدء خادم Flask.")

        # Wait for the Flask thread to finish (it usually won't unless there's an error)
        flask_thread.join()

    except Exception as startup_err:
        logger.critical(f"❌ [Main] حدث خطأ فادح أثناء بدء التشغيل أو في الحلقة الرئيسية: {startup_err}", exc_info=True)
    finally:
        logger.info("🛑 [Main] إيقاف تشغيل البرنامج...")
        cleanup_resources()
        logger.info("👋 [Main] تم إيقاف بوت إشارات تداول العملات الرقمية.")
        os._exit(0)
