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
    level=logging.INFO,
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
    # WEBHOOK_URL يستخدم الآن كعنوان URL عام للوحة التحكم
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
except Exception as e:
     logger.critical(f"❌ فشل تحميل متغيرات البيئة الأساسية: {e}")
     exit(1)

logger.info(f"مفتاح API الخاص بـ Binance: {'متاح' if API_KEY else 'غير متاح'}")
logger.info(f"رمز Telegram: {TELEGRAM_TOKEN[:10]}...{'*' * (len(TELEGRAM_TOKEN)-10)}")
logger.info(f"معرف دردشة Telegram: {CHAT_ID}")
logger.info(f"عنوان URL لقاعدة البيانات: {'متاح' if DB_URL else 'غير متاح'}")
logger.info(f"عنوان URL للوحة التحكم (Dashboard URL): {WEBHOOK_URL if WEBHOOK_URL else 'غير محدد'}")

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
TRADE_VALUE: float = 10.0
MAX_OPEN_TRADES: int = 5
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 3
SIGNAL_TRACKING_TIMEFRAME: str = '15m'
SIGNAL_TRACKING_LOOKBACK_DAYS: int = 1

# Indicator Parameters (يجب أن تتطابق مع ml.py)
RSI_PERIOD: int = 9
RSI_OVERSOLD: int = 30
RSI_OVERBOUGHT: int = 70
VOLUME_LOOKBACK_CANDLES: int = 1
RSI_MOMENTUM_LOOKBACK_CANDLES: int = 2

ENTRY_ATR_PERIOD: int = 10
ENTRY_ATR_MULTIPLIER: float = 1.5

SUPERTRAND_PERIOD: int = 10
SUPERTRAND_MULTIPLIER: float = 3.0

# Ichimoku Cloud Parameters (يجب أن تتطابق مع ml.py)
TENKAN_PERIOD: int = 9
KIJUN_PERIOD: int = 26
SENKOU_SPAN_B_PERIOD: int = 52
CHIKOU_LAG: int = 26

# Fibonacci & S/R Parameters (يجب أن تتطابق مع ml.py)
FIB_SR_LOOKBACK_WINDOW: int = 50

MIN_PROFIT_MARGIN_PCT: float = 1.0
MIN_VOLUME_15M_USDT: float = 50000.0

TARGET_APPROACH_THRESHOLD_PCT: float = 0.005

BINANCE_FEE_RATE: float = 0.001

BASE_ML_MODEL_NAME: str = 'DecisionTree_Scalping_V1'

# المتغيرات العامة
conn: Optional[psycopg2.extensions.connection] = None
cur: Optional[psycopg2.extensions.cursor] = None
client: Optional[Client] = None
ticker_data: Dict[str, float] = {}
ml_models: Dict[str, Any] = {}

# ---------------------- إعداد عميل Binance ----------------------
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

# ---------------------- دوال المؤشرات الإضافية ----------------------
def get_fear_greed_index() -> str:
    """تجلب مؤشر الخوف والجشع من alternative.me وتترجم التصنيف إلى العربية."""
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
    تجلب بيانات الشموع التاريخية من Binance لعدد محدد من الأيام.
    """
    if not client:
        logger.error("❌ [Data] عميل Binance غير مهيأ لجلب البيانات.")
        return None
    try:
        start_dt = datetime.utcnow() - timedelta(days=days + 1)
        start_str_overall = start_dt.strftime("%Y-%m-%d %H:%M:%S")

        logger.debug(f"ℹ️ [Data] جلب بيانات {interval} لـ {symbol} من {start_str_overall} فصاعدًا...")

        binance_interval_map = {
            '15m': Client.KLINE_INTERVAL_15MINUTE, '5m': Client.KLINE_INTERVAL_5MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR, '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY
        }
        binance_interval = binance_interval_map.get(interval)
        if not binance_interval:
            logger.error(f"❌ [Data] فترة غير مدعومة: {interval}")
            return None

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
        df = df[numeric_cols].dropna()

        if df.empty:
            logger.warning(f"⚠️ [Data] DataFrame لـ {symbol} فارغ بعد إزالة قيم NaN الأساسية.")
            return None

        df.sort_index(inplace=True)
        logger.debug(f"✅ [Data] تم جلب ومعالجة {len(df)} شمعة تاريخية ({interval}) لـ {symbol}.")
        return df

    except (BinanceAPIException, BinanceRequestException) as binance_err:
         logger.error(f"❌ [Data] خطأ في Binance API أو الشبكة أثناء جلب البيانات لـ {symbol}: {binance_err}")
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
        logger.warning("⚠️ [Indicator RSI] 'close' column is missing or empty.")
        df['rsi'] = np.nan
        return df
    if len(df) < period:
        logger.warning(f"⚠️ [Indicator RSI] Insufficient data ({len(df)} < {period}) to calculate RSI.")
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
        logger.warning("⚠️ [Indicator ATR] 'high', 'low', 'close' columns are missing or empty.")
        df['atr'] = np.nan
        return df
    if len(df) < period + 1:
        logger.warning(f"⚠️ [Indicator ATR] Insufficient data ({len(df)} < {period + 1}) to calculate ATR.")
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
        logger.warning("⚠️ [Indicator Supertrend] 'high', 'low', 'close' columns are missing or empty. Cannot calculate Supertrend.")
        df['supertrend'] = np.nan
        df['supertrend_direction'] = 0 # Neutral if cannot calculate
        return df

    # Ensure ATR is already calculated
    if 'atr' not in df.columns:
        df = calculate_atr_indicator(df, period=period) # Use Supertrend period for ATR if not already calculated
        if 'atr' not in df.columns or df['atr'].isnull().all().any():
            logger.warning("⚠️ [Indicator Supertrend] ATR calculation failed. Cannot calculate Supertrend.")
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
    df['supertrend_direction'] = 0 # 1 for uptrend, -1 for downtrend, 0 for neutral/flat

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
        if df['supertrend_direction'].iloc[i-1] == 1: # Previous was uptrend
            if df['close'].iloc[i] < df['final_upper_band'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['final_upper_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = -1 # Change to downtrend
            else:
                df.loc[df.index[i], 'supertrend'] = df['final_lower_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = 1 # Remain uptrend
        elif df['supertrend_direction'].iloc[i-1] == -1: # Previous was downtrend
            if df['close'].iloc[i] > df['final_lower_band'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['final_lower_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = 1 # Change to uptrend
            else:
                df.loc[df.index[i], 'supertrend'] = df['final_upper_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = -1 # Remain downtrend
        else: # Initial state or neutral
            if df['close'].iloc[i] > df['final_lower_band'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['final_lower_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = 1
            elif df['close'].iloc[i] < df['final_upper_band'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['final_upper_band'].iloc[i]
                df.loc[df.index[i], 'supertrend_direction'] = -1
            else:
                df.loc[df.index[i], 'supertrend'] = df['close'].iloc[i] # Fallback
                df.loc[df.index[i], 'supertrend_direction'] = 0


    # Drop temporary columns
    df.drop(columns=['basic_upper_band', 'basic_lower_band', 'final_upper_band', 'final_lower_band'], inplace=True, errors='ignore')
    logger.debug(f"✅ [Indicator Supertrend] Supertrend calculated.")
    return df

def _calculate_btc_trend_feature(df_btc: pd.DataFrame) -> Optional[pd.Series]:
    """
    Calculates a numerical representation of Bitcoin's trend based on EMA20 and EMA50.
    Returns 1 for bullish (صعودي), -1 for bearish (هبوطي), 0 for neutral/sideways (محايد/تذبذب).
    """
    logger.debug("ℹ️ [Indicators] Calculating Bitcoin trend for features...")
    # Need enough data for EMA50, plus a few extra candles for robustness
    min_data_for_ema = 50 + 5 # 50 for EMA50, 5 buffer

    if df_btc is None or df_btc.empty or len(df_btc) < min_data_for_ema:
        logger.warning(f"⚠️ [Indicators] Insufficient BTC/USDT data ({len(df_btc) if df_btc is not None else 0} < {min_data_for_ema}) to calculate Bitcoin trend for features.")
        # Return a series of zeros (neutral) with the original index if data is insufficient
        return pd.Series(index=df_btc.index if df_btc is not None else None, data=0.0)

    df_btc_copy = df_btc.copy()
    df_btc_copy['close'] = pd.to_numeric(df_btc_copy['close'], errors='coerce')
    df_btc_copy.dropna(subset=['close'], inplace=True)

    if len(df_btc_copy) < min_data_for_ema:
        logger.warning(f"⚠️ [Indicators] Insufficient BTC/USDT data after NaN removal to calculate trend.")
        return pd.Series(index=df_btc.index, data=0.0) # Return neutral if not enough data after dropna

    ema20 = calculate_ema(df_btc_copy['close'], 20)
    ema50 = calculate_ema(df_btc_copy['close'], 50)

    # Combine EMAs and close into a single DataFrame for easier comparison
    ema_df = pd.DataFrame({'ema20': ema20, 'ema50': ema50, 'close': df_btc_copy['close']})
    ema_df.dropna(inplace=True) # Drop rows where any EMA or close is NaN

    if ema_df.empty:
        logger.warning("⚠️ [Indicators] EMA DataFrame is empty after NaN removal. Cannot calculate Bitcoin trend.")
        return pd.Series(index=df_btc.index, data=0.0) # Return neutral if no valid EMA data

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
    logger.debug(f"✅ [Indicators] Bitcoin trend feature calculated. Examples: {final_trend_series.tail().tolist()}")
    return final_trend_series


# NEW: Ichimoku Cloud Calculation (Copied from ml.py)
def calculate_ichimoku_cloud(df: pd.DataFrame, tenkan_period: int = TENKAN_PERIOD, kijun_period: int = KIJUN_PERIOD, senkou_span_b_period: int = SENKOU_SPAN_B_PERIOD, chikou_lag: int = CHIKOU_LAG) -> pd.DataFrame:
    """Calculates Ichimoku Cloud components and derived features."""
    df_ichimoku = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df_ichimoku.columns for col in required_cols) or df_ichimoku[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator Ichimoku] Missing or empty OHLC columns. Cannot calculate Ichimoku.")
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
        df_ichimoku.loc[(df_ichimoku['tenkan_sen'].shift(1) < df_ichimoku['kijun_sen'].shift(1)) &
                        (df_ichimoku['tenkan_sen'] > df_ichimoku['kijun_sen']), 'ichimoku_tenkan_kijun_cross_signal'] = 1
        # Bearish cross: Tenkan-sen crosses below Kijun-sen
        df_ichimoku.loc[(df_ichimoku['tenkan_sen'].shift(1) > df_ichimoku['kijun_sen'].shift(1)) &
                        (df_ichimoku['tenkan_sen'] < df_ichimoku['kijun_sen']), 'ichimoku_tenkan_kijun_cross_signal'] = -1

    # Price vs Cloud Position (using current close price vs future cloud)
    df_ichimoku['ichimoku_price_cloud_position'] = 0 # 0 for inside, 1 for above, -1 for below
    # Price above cloud
    df_ichimoku.loc[(df_ichimoku['close'] > df_ichimoku[['senkou_span_a', 'senkou_span_b']].max(axis=1)), 'ichimoku_price_cloud_position'] = 1
    # Price below cloud
    df_ichimoku.loc[(df_ichimoku['close'] < df_ichimoku[['senkou_span_a', 'senkou_span_b']].min(axis=1)), 'ichimoku_price_cloud_position'] = -1

    # Cloud Outlook (future cloud's color)
    df_ichimoku['ichimoku_cloud_outlook'] = 0 # 0 for flat/mixed, 1 for bullish (green), -1 for bearish (red)
    df_ichimoku.loc[(df_ichimoku['senkou_span_a'] > df_ichimoku['senkou_span_b']), 'ichimoku_cloud_outlook'] = 1 # Green Cloud
    df_ichimoku.loc[(df_ichimoku['senkou_span_a'] < df_ichimoku['senkou_span_b']), 'ichimoku_cloud_outlook'] = -1 # Red Cloud

    logger.debug(f"✅ [Indicator Ichimoku] Ichimoku Cloud and derived features calculated.")
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
        logger.warning("⚠️ [Indicator Fibonacci] Missing or empty OHLC columns. Cannot calculate Fibonacci features.")
        for col in ['fib_236_retrace_dist_norm', 'fib_382_retrace_dist_norm', 'fib_618_retrace_dist_norm', 'is_price_above_fib_50']:
            df_fib[col] = np.nan
        return df_fib
    if len(df_fib) < lookback_window:
        logger.warning(f"⚠️ [Indicator Fibonacci] Insufficient data ({len(df_fib)} < {lookback_window}) for Fibonacci calculation.")
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

    logger.debug(f"✅ [Indicator Fibonacci] Fibonacci features calculated.")
    return df_fib


# NEW: Support and Resistance Features (Copied from ml.py)
def calculate_support_resistance_features(df: pd.DataFrame, lookback_window: int = FIB_SR_LOOKBACK_WINDOW) -> pd.DataFrame:
    """
    Calculates simplified support and resistance features based on the lowest low and highest high
    within a rolling lookback window.
    """
    df_sr = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df_sr.columns for col in required_cols) or df_sr[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator S/R] Missing or empty OHLC columns. Cannot calculate S/R features.")
        for col in ['price_distance_to_recent_low_norm', 'price_distance_to_recent_high_norm']:
            df_sr[col] = np.nan
        return df_sr
    if len(df_sr) < lookback_window:
        logger.warning(f"⚠️ [Indicator S/R] Insufficient data ({len(df_sr)} < {lookback_window}) for S/R calculation.")
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
            df_sr.loc[df_sr.index[i], 'price_distance_to_recent_low_norm'] = 0.0 # Price is at the low
            df_sr.loc[df_sr.index[i], 'price_distance_to_recent_high_norm'] = 0.0 # Price is at the high (if range is 0)

    logger.debug(f"✅ [Indicator S/R] Support and Resistance features calculated.")
    return df_sr

# ---------------------- إعداد قاعدة البيانات ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
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
                CREATE TABLE IF NOT EXISTS market_dominance (id SERIAL PRIMARY KEY, recorded_at TIMESTAMP DEFAULT NOW(),
                    btc_dominance DOUBLE PRECISION, eth_dominance DOUBLE PRECISION);
            """)
            conn.commit()
            logger.info("✅ [DB] تم تهيئة قاعدة البيانات بنجاح.")
            return
        except (OperationalError, Exception) as e:
            logger.error(f"❌ [DB] خطأ في الاتصال (المحاولة {attempt + 1}): {e}")
            if conn: conn.rollback()
            if attempt == retries - 1:
                logger.critical("❌ [DB] فشلت جميع محاولات الاتصال بقاعدة البيانات.")
                exit(1)
            time.sleep(delay)

def check_db_connection() -> bool:
    global conn, cur
    try:
        if conn is None or conn.closed != 0:
            init_db()
        else:
            with conn.cursor() as check_cur:
                check_cur.execute("SELECT 1;")
        return True
    except (OperationalError, InterfaceError):
        logger.error("❌ [DB] فقد الاتصال بقاعدة البيانات. إعادة التهيئة...")
        try:
            init_db()
            return True
        except Exception as recon_err:
            logger.error(f"❌ [DB] فشلت محاولة إعادة الاتصال: {recon_err}")
            return False
    return False

def load_ml_model_from_db(symbol: str) -> Optional[Any]:
    global ml_models
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_name in ml_models:
        return ml_models[model_name]
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
            logger.warning(f"⚠️ [ML Model] لم يتم العثور على نموذج '{model_name}'.")
            return None
    except Exception as e:
        logger.error(f"❌ [ML Model] خطأ أثناء تحميل نموذج ML لـ {symbol}: {e}", exc_info=True)
        return None

def convert_np_values(obj: Any) -> Any:
    # تم تحديث هذه الدالة لمعالجة أنواع NumPy بشكل صحيح مع إصدارات NumPy 2.0+
    if isinstance(obj, (np.integer, np.int_, np.int64)): # إضافة np.int64
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)): # استبدال np.float_ بـ np.float64
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_np_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_np_values(i) for i in obj]
    if pd.isna(obj):
        return None
    return obj

# ---------------------- إدارة WebSocket لأسعار المؤشرات ----------------------
def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
    global ticker_data
    try:
        data = msg.get('data', msg) if isinstance(msg, dict) else msg
        if not isinstance(data, list): data = [data]
        for item in data:
            symbol = item.get('s')
            price_str = item.get('c')
            if symbol and 'USDT' in symbol and price_str:
                ticker_data[symbol] = float(price_str)
    except Exception as e:
        logger.error(f"❌ [WS] خطأ في معالجة رسالة المؤشر: {e}", exc_info=True)

def run_ticker_socket_manager() -> None:
    while True:
        try:
            logger.info("ℹ️ [WS] بدء مدير WebSocket لأسعار المؤشرات...")
            twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
            twm.start()
            stream_name = twm.start_miniticker_socket(callback=handle_ticker_message)
            logger.info(f"✅ [WS] تم بدء تدفق WebSocket: {stream_name}")
            twm.join()
        except Exception as e:
            logger.error(f"❌ [WS] خطأ فادح في مدير WebSocket: {e}. إعادة التشغيل في 15 ثانية...", exc_info=True)
        time.sleep(15)

# ---------------------- دوال مساعدة أخرى ----------------------
def fetch_recent_volume(symbol: str, interval: str = SIGNAL_GENERATION_TIMEFRAME, num_candles: int = VOLUME_LOOKBACK_CANDLES) -> float:
    if not client: return 0.0
    try:
        binance_interval = getattr(Client, f'KLINE_INTERVAL_{interval.upper()}', None)
        if not binance_interval: return 0.0
        klines = client.get_klines(symbol=symbol, interval=binance_interval, limit=num_candles)
        return sum(float(k[7]) for k in klines if len(k) > 7 and k[7])
    except Exception: return 0.0

def get_crypto_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    raw_symbols: List[str] = []
    try:
        with open(os.path.join(os.path.dirname(__file__), filename), 'r', encoding='utf-8') as f:
            raw_symbols = [line.strip().upper() for line in f if line.strip() and not line.startswith('#')]
        raw_symbols = sorted([f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols])
    except Exception as e:
        logger.error(f"❌ [Data] خطأ في قراءة الملف '{filename}': {e}")
        return []
    if not client or not raw_symbols: return raw_symbols
    try:
        exchange_info = client.get_exchange_info()
        valid_symbols = {s['symbol'] for s in exchange_info['symbols'] if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'}
        return [s for s in raw_symbols if s in valid_symbols]
    except Exception as e:
        logger.error(f"❌ [Data Validation] خطأ أثناء التحقق من الرموز: {e}")
        return raw_symbols

# ---------------------- Trading Strategy (Adjusted for ML-Only) -------------------
class ScalpingTradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ml_model = load_ml_model_from_db(symbol)
        # تحديث قائمة الميزات لتشمل المؤشرات الجديدة التي تم تدريب النموذج عليها
        self.feature_columns_for_ml = [
            'volume_15m_avg', 'rsi_momentum_bullish', 'btc_trend_feature', 'supertrend_direction',
            'ichimoku_tenkan_kijun_cross_signal', 'ichimoku_price_cloud_position', 'ichimoku_cloud_outlook',
            'fib_236_retrace_dist_norm', 'fib_382_retrace_dist_norm', 'fib_618_retrace_dist_norm', 'is_price_above_fib_50',
            'price_distance_to_recent_low_norm', 'price_distance_to_recent_high_norm'
        ]

    def populate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        # الحد الأدنى لطول البيانات المطلوبة لجميع المؤشرات
        min_len_required = max(
            VOLUME_LOOKBACK_CANDLES,
            RSI_PERIOD,
            RSI_MOMENTUM_LOOKBACK_CANDLES,
            ENTRY_ATR_PERIOD,
            SUPERTRAND_PERIOD,
            TENKAN_PERIOD,
            KIJUN_PERIOD,
            SENKOU_SPAN_B_PERIOD,
            CHIKOU_LAG, # للحفاظ على توافق مؤشر تشيكو
            FIB_SR_LOOKBACK_WINDOW,
            55 # لحساب EMA البيتكوين
        ) + 5 # بوفير إضافي

        if len(df) < min_len_required:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame قصير جداً ({len(df)} < {min_len_required}) لحساب المؤشرات.")
            return None
        
        try:
            df_calc = df.copy()
            df_calc['volume_15m_avg'] = df_calc['volume'].rolling(window=VOLUME_LOOKBACK_CANDLES).mean()
            df_calc = calculate_rsi_indicator(df_calc, RSI_PERIOD)
            df_calc['rsi_momentum_bullish'] = 0
            for i in range(RSI_MOMENTUM_LOOKBACK_CANDLES, len(df_calc)):
                rsi_slice = df_calc['rsi'].iloc[i - RSI_MOMENTUM_LOOKBACK_CANDLES : i + 1]
                if not rsi_slice.isnull().any() and np.all(np.diff(rsi_slice) > 0) and rsi_slice.iloc[-1] > 50:
                    df_calc.loc[df_calc.index[i], 'rsi_momentum_bullish'] = 1
            
            df_calc = calculate_atr_indicator(df_calc, ENTRY_ATR_PERIOD)
            df_calc = calculate_supertrend(df_calc, SUPERTRAND_PERIOD, SUPERTRAND_MULTIPLIER)
            
            # جلب وحساب ميزة اتجاه البيتكوين
            btc_df = fetch_historical_data("BTCUSDT", interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
            if btc_df is not None:
                btc_trend = _calculate_btc_trend_feature(btc_df)
                if btc_trend is not None:
                    df_calc = df_calc.merge(btc_trend.rename('btc_trend_feature'), left_index=True, right_index=True, how='left')
                    # FIX: Removed inplace=True to avoid FutureWarning
                    df_calc['btc_trend_feature'] = df_calc['btc_trend_feature'].fillna(0.0)
            else:
                df_calc['btc_trend_feature'] = 0.0 # الافتراضي إذا لم يتم جلب بيانات BTC
            
            # حساب مؤشرات إيشيموكو الجديدة
            df_calc = calculate_ichimoku_cloud(df_calc, TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_SPAN_B_PERIOD, CHIKOU_LAG)
            
            # حساب ميزات فيبوناتشي الجديدة
            df_calc = calculate_fibonacci_features(df_calc, FIB_SR_LOOKBACK_WINDOW)
            
            # حساب ميزات الدعم والمقاومة الجديدة
            df_calc = calculate_support_resistance_features(df_calc, FIB_SR_LOOKBACK_WINDOW)

            for col in self.feature_columns_for_ml:
                if col not in df_calc.columns:
                    df_calc[col] = np.nan # إضافة أي أعمدة مفقودة بقيم NaN
            
            df_cleaned = df_calc.dropna(subset=self.feature_columns_for_ml).copy()
            return df_cleaned if not df_cleaned.empty else None
        except Exception as e:
            logger.error(f"❌ [Strategy {self.symbol}] خطأ في حساب المؤشر: {e}", exc_info=True)
            return None

    def generate_buy_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if df_processed is None or df_processed.empty or self.ml_model is None: return None
        last_row = df_processed.iloc[-1]
        current_price = ticker_data.get(self.symbol)
        # FIX: Changed '===' to 'is' for correct Python syntax
        if current_price is None or last_row[self.feature_columns_for_ml].isnull().any(): return None 
        try:
            features_df = pd.DataFrame([last_row[self.feature_columns_for_ml]], columns=self.feature_columns_for_ml)
            ml_pred = self.ml_model.predict(features_df)[0]
            if ml_pred != 1: return None
        except Exception: return None
        
        signal_details = {col: last_row.get(col, 'N/A') for col in self.feature_columns_for_ml}
        signal_details['ML_Prediction'] = 'صعودي ✅'

        # الشروط الإضافية التي قد تصفى من قبل النموذج المدرب الآن
        # if last_row.get('supertrend_direction') != 1: return None
        # if last_row.get('btc_trend_feature') == -1.0: return None
        
        volume_recent = fetch_recent_volume(self.symbol)
        if volume_recent < MIN_VOLUME_15M_USDT:
            logger.debug(f"⚠️ [Signal Gen] حجم منخفض لـ {self.symbol}: {volume_recent}")
            return None
        
        current_atr = last_row.get('atr')
        if pd.isna(current_atr) or current_atr <= 0:
            logger.debug(f"⚠️ [Signal Gen] ATR غير صالح لـ {self.symbol}.")
            return None
        
        # تحديد الهدف ووقف الخسارة بناءً على ATR مع استخدام ميزات النموذج لفلترة أفضل
        initial_target = current_price + (ENTRY_ATR_MULTIPLIER * current_atr)
        profit_margin_pct = ((initial_target / current_price) - 1) * 100
        if profit_margin_pct < MIN_PROFIT_MARGIN_PCT:
            logger.debug(f"⚠️ [Signal Gen] هامش ربح غير كافٍ لـ {self.symbol}: {profit_margin_pct:.2f}%")
            return None
        
        # استخدام Supertrend كوقف خسارة أولي
        initial_stop_loss = last_row.get('supertrend', current_price - (1.0 * current_atr))
        # التأكد من أن وقف الخسارة تحت سعر الدخول
        if initial_stop_loss >= current_price:
            initial_stop_loss = current_price - (1.0 * current_atr)
        initial_stop_loss = max(0.00000001, initial_stop_loss) # منع وقف الخسارة من أن يكون صفراً أو سالباً

        return {
            'symbol': self.symbol, 'entry_price': current_price, 'initial_target': initial_target,
            'current_target': initial_target, 'stop_loss': initial_stop_loss, 'r2_score': 1.0, # r2_score هنا هو مجرد قيمة وهمية للنموذج الثنائي
            'strategy_name': 'Scalping_ML_Filtered', 'signal_details': signal_details,
            'volume_15m': volume_recent, 'trade_value': TRADE_VALUE
        }

# ---------------------- دوال Telegram ----------------------
def send_telegram_message(target_chat_id: str, text: str, reply_markup: Optional[Dict] = None, parse_mode: str = 'Markdown', disable_web_page_preview: bool = True, timeout: int = 20) -> Optional[Dict]:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(target_chat_id), 'text': text, 'parse_mode': parse_mode, 'disable_web_page_preview': disable_web_page_preview}
    if reply_markup:
        payload['reply_markup'] = json.dumps(convert_np_values(reply_markup))
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")
        return None

def send_telegram_alert(signal_data: Dict[str, Any], timeframe: str) -> None:
    try:
        symbol = signal_data['symbol']
        safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[')
        entry_price = float(signal_data['entry_price'])
        target_price = float(signal_data['initial_target'])
        stop_loss_price = float(signal_data['stop_loss'])
        profit_pct = ((target_price / entry_price) - 1) * 100
        dashboard_url = WEBHOOK_URL if WEBHOOK_URL else 'http://localhost:10000' # Fallback URL
        
        message = f"""💡 *إشارة تداول جديدة* 💡
        --------------------
        🪙 **الزوج:** `{safe_symbol}`
        📈 **نوع الإشارة:** شراء (طويل)
        🕰️ **الإطار الزمني:** {timeframe}
        ➡️ **سعر الدخول:** `${entry_price:,.8g}`
        🎯 **الهدف:** `${target_price:,.8g}` ({profit_pct:+.2f}%)
        🛑 **وقف الخسارة:** `${stop_loss_price:,.8g}`
        --------------------
        """
        
        reply_markup = {
            "inline_keyboard": [[
                {"text": "📊 فتح لوحة التحكم", "url": dashboard_url}
            ]]
        }
        send_telegram_message(CHAT_ID, message, reply_markup=reply_markup, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"❌ [Telegram Alert] فشل إرسال تنبيه الإشارة: {e}", exc_info=True)

def send_tracking_notification(details: Dict[str, Any]) -> None:
    symbol = details.get('symbol', 'N/A')
    safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[')
    notification_type = details.get('type', 'unknown')
    profit_pct = details.get('profit_pct', 0.0)
    closing_price = details.get('closing_price', 0.0)
    time_to_target = details.get('time_to_target', 'N/A')
    message = ""
    if notification_type == 'target_hit':
        message = f"✅ *تم الوصول إلى الهدف* | `{safe_symbol}`\n💰 الربح: {profit_pct:+.2f}% | ⏱️ الوقت: {time_to_target}"
    elif notification_type == 'stop_loss_hit':
        message = f"🛑 *تم ضرب وقف الخسارة* | `{safe_symbol}`\n💔 الخسارة: {profit_pct:+.2f}%"
    elif notification_type == 'target_stoploss_updated':
        message = f"🔄 *تحديث الإشارة* | `{safe_symbol}`\n🎯 الهدف الجديد: `${details.get('new_target', 0):.8g}`\n🛑 وقف الخسارة الجديد: `${details.get('new_stop_loss', 0):.8g}`"
    if message:
        send_telegram_message(CHAT_ID, message, parse_mode='Markdown')

def close_trade_by_id(signal_id: int, chat_id: str) -> None:
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur_close:
            cur_close.execute("SELECT symbol, entry_price, entry_time FROM signals WHERE id = %s AND closed_at IS NULL;", (signal_id,))
            signal_data = cur_close.fetchone()
            if not signal_data: return
            symbol, entry_price, entry_time = signal_data['symbol'], float(signal_data['entry_price']), signal_data['entry_time']
            current_price = ticker_data.get(symbol)
            if current_price is None: return
            profit_pct = ((current_price / entry_price) - 1) * 100
            closed_at = datetime.now()
            time_to_close = closed_at - entry_time if entry_time else timedelta(0)
            cur_close.execute(
                "UPDATE signals SET achieved_target = FALSE, closing_price = %s, closed_at = %s, profit_percentage = %s, time_to_target = %s WHERE id = %s;",
                (current_price, closed_at, profit_pct, time_to_close, signal_id)
            )
            conn.commit()
            safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[')
            send_telegram_message(chat_id, f"✅ *تم إغلاق الصفقة يدوياً* | `{safe_symbol}`", parse_mode='Markdown')
    except Exception as e:
        logger.error(f"❌ [Close Trade] خطأ أثناء إغلاق الصفقة {signal_id}: {e}", exc_info=True)
        if conn: conn.rollback()

# ---------------------- دوال قاعدة البيانات ----------------------
def insert_signal_into_db(signal: Dict[str, Any]) -> bool:
    if not check_db_connection() or not conn: return False
    try:
        signal_prepared = convert_np_values(signal)
        signal_details_json = json.dumps(signal_prepared.get('signal_details', {}))
        with conn.cursor() as cur_ins:
            cur_ins.execute(
                """INSERT INTO signals (symbol, entry_price, initial_target, current_target, stop_loss, r2_score, strategy_name, signal_details, volume_15m, entry_time)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW());""",
                (signal_prepared['symbol'], signal_prepared['entry_price'], signal_prepared['initial_target'],
                 signal_prepared['current_target'], signal_prepared['stop_loss'], signal_prepared.get('r2_score'),
                 signal_prepared.get('strategy_name'), signal_details_json, signal_prepared.get('volume_15m'))
            )
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"❌ [DB Insert] خطأ أثناء إدراج الإشارة: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

# ---------------------- دالة تتبع الإشارات المفتوحة ----------------------
def track_signals() -> None:
    logger.info("ℹ️ [Tracker] بدء عملية تتبع الإشارات المفتوحة...")
    while True:
        try:
            if not check_db_connection() or not conn:
                time.sleep(15)
                continue
            with conn.cursor() as track_cur:
                track_cur.execute("SELECT id, symbol, entry_price, current_target, entry_time, stop_loss FROM signals WHERE closed_at IS NULL;")
                open_signals = track_cur.fetchall()
            if not open_signals:
                time.sleep(10)
                continue
            
            for signal_row in open_signals:
                signal_id, symbol = signal_row['id'], signal_row['symbol']
                entry_price, current_target = float(signal_row['entry_price']), float(signal_row["current_target"])
                current_stop_loss = float(signal_row["stop_loss"]) if signal_row.get("stop_loss") is not None else None
                current_price = ticker_data.get(symbol)
                # FIX: Changed '===' to 'is' for correct Python syntax
                if current_price is None: continue 

                closed = False
                notification_details = {'symbol': symbol, 'id': signal_id}
                
                # Check stop loss first
                if current_stop_loss and current_price <= current_stop_loss:
                    profit_pct = ((current_stop_loss / entry_price) - 1) * 100
                    query = "UPDATE signals SET achieved_target = FALSE, closing_price = %s, closed_at = NOW(), profit_percentage = %s, time_to_target = NOW() - entry_time WHERE id = %s;"
                    params = (current_stop_loss, profit_pct, signal_id)
                    notification_details.update({'type': 'stop_loss_hit', 'closing_price': current_stop_loss, 'profit_pct': profit_pct})
                    closed = True
                # Check target hit
                elif current_price >= current_target:
                    profit_pct = ((current_target / entry_price) - 1) * 100
                    query = "UPDATE signals SET achieved_target = TRUE, closing_price = %s, closed_at = NOW(), profit_percentage = %s, time_to_target = NOW() - entry_time WHERE id = %s;"
                    params = (current_target, profit_pct, signal_id)
                    notification_details.update({'type': 'target_hit', 'closing_price': current_target, 'profit_pct': profit_pct, 'time_to_target': str(datetime.now() - signal_row['entry_time'])})
                    closed = True
                
                if closed:
                    with conn.cursor() as update_cur:
                        update_cur.execute(query, params)
                    conn.commit()
                    send_tracking_notification(notification_details)

            time.sleep(3)
        except Exception as e:
            logger.error(f"❌ [Tracker] خطأ في دورة التتبع: {e}", exc_info=True)
            if conn: conn.rollback()
            time.sleep(30)

# ---------------------- خدمة Flask (الواجهة الخلفية للوحة التحكم) ----------------------
app = Flask(__name__)
CORS(app) # تفعيل CORS لجميع المسارات

@app.route('/')
def serve_dashboard():
    # هذه الدالة تخدم ملف لوحة التحكم
    try:
        # تأكد من أن dashboard.html موجود في نفس الدليل
        return send_from_directory('.', 'dashboard.html')
    except FileNotFoundError:
        logger.error("❌ [Flask] dashboard.html not found!")
        return "Dashboard file not found.", 404

@app.route('/api/status')
def api_status():
    ws_alive = ws_thread.is_alive() if 'ws_thread' in globals() else False
    return jsonify({'status': 'متصل' if ws_alive else 'غير متصل'})

@app.route('/api/performance')
def api_performance():
    if not check_db_connection() or not conn: return jsonify({'error': 'DB connection failed'}), 500
    try:
        with conn.cursor() as db_cur:
            db_cur.execute("""
                SELECT
                    COUNT(*) AS total_trades,
                    COUNT(*) FILTER (WHERE profit_percentage > 0) AS winning_trades,
                    COALESCE(SUM(profit_percentage), 0) AS total_profit_pct
                FROM signals WHERE closed_at IS NOT NULL;
            """)
            stats = db_cur.fetchone() or {}
            total = stats.get('total_trades', 0)
            winning = stats.get('winning_trades', 0)
            stats['win_rate'] = (winning / total * 100) if total > 0 else 0
            return jsonify(convert_np_values(stats))
    except Exception as e:
        logger.error(f"API Error in /api/performance: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/open-signals')
def api_open_signals():
    if not check_db_connection() or not conn: return jsonify([]), 500
    try:
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT id, symbol, entry_price, current_target, sent_at FROM signals WHERE closed_at IS NULL ORDER BY sent_at DESC;")
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
            # General stats
            db_cur.execute("""
                SELECT
                    COUNT(*) AS total_trades,
                    COUNT(*) FILTER (WHERE profit_percentage > 0) AS winning_trades,
                    COUNT(*) FILTER (WHERE profit_percentage <= 0) AS losing_trades,
                    COALESCE(SUM(profit_percentage), 0) AS total_profit_pct,
                    COALESCE(AVG(profit_percentage), 0) AS avg_profit_pct
                FROM signals WHERE closed_at IS NOT NULL;
            """)
            report = db_cur.fetchone() or {}
            total = report.get('total_trades', 0)
            winning = report.get('winning_trades', 0)
            report['win_rate'] = (winning / total * 100) if total > 0 else 0
            
            # Best performing
            db_cur.execute("""
                SELECT symbol, AVG(profit_percentage) as avg_profit, COUNT(id) as trade_count
                FROM signals WHERE closed_at IS NOT NULL AND profit_percentage > 0
                GROUP BY symbol ORDER BY avg_profit DESC LIMIT 1;
            """)
            report['best_performing_symbol'] = db_cur.fetchone()

            # Worst performing
            db_cur.execute("""
                SELECT symbol, AVG(profit_percentage) as avg_profit, COUNT(id) as trade_count
                FROM signals WHERE closed_at IS NOT NULL AND profit_percentage <= 0
                GROUP BY symbol ORDER BY avg_profit ASC LIMIT 1;
            """)
            report['worst_performing_symbol'] = db_cur.fetchone()

            return jsonify(convert_np_values(report))
    except Exception as e:
        logger.error(f"API Error in /api/general-report: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/webhook', methods=['POST'])
def webhook():
    if not WEBHOOK_URL: return "Webhook not configured", 200
    try:
        data = request.get_json()
        if 'callback_query' in data:
            callback_query = data['callback_query']
            callback_data = callback_query.get('data')
            chat_id = callback_query.get('message', {}).get('chat', {}).get('id')
            if not chat_id: return "OK", 200

            # Acknowledge the callback
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery", json={'callback_query_id': callback_query['id']})

            if callback_data and callback_data.startswith("exit_trade_"):
                signal_id = int(callback_data.replace("exit_trade_", ""))
                Thread(target=close_trade_by_id, args=(signal_id, chat_id)).start()
        
        elif 'message' in data:
            message_data = data['message']
            chat_id = message_data.get('chat', {}).get('id')
            text_msg = message_data.get('text', '').strip().lower()
            if not chat_id: return "OK", 200

            if text_msg == '/report':
                dashboard_url = WEBHOOK_URL if WEBHOOK_URL else 'http://localhost:10000'
                message = "📈 لعرض تقرير الأداء الكامل وجميع الصفقات الحية، يرجى زيارة لوحة التحكم."
                reply_markup = {"inline_keyboard": [[{"text": "📊 فتح لوحة التحكم", "url": dashboard_url}]]}
                send_telegram_message(chat_id, message, reply_markup=reply_markup, parse_mode='Markdown')

        return "OK", 200
    except Exception as e:
        logger.error(f"❌ [Webhook] خطأ في معالجة Webhook: {e}", exc_info=True)
        return "Internal Server Error", 500

def run_flask():
    host = "0.0.0.0"
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"ℹ️ [Flask] بدء تطبيق Flask على {host}:{port}...")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        logger.warning("⚠️ [Flask] 'waitress' غير مثبت. استخدام خادم تطوير Flask.")
        app.run(host=host, port=port)

# ---------------------- الحلقة الرئيسية والتحقق ----------------------
def main_loop():
    symbols_to_scan = get_crypto_symbols()
    if not symbols_to_scan:
        logger.critical("❌ [Main] لا توجد رموز صالحة للمتابعة.")
        return
    logger.info(f"✅ [Main] تم تحميل {len(symbols_to_scan)} رمزًا للمسح.")

    while True:
        try:
            logger.info(f"🔄 [Main] بدء دورة مسح السوق - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if not check_db_connection() or not conn:
                time.sleep(60)
                continue
            
            with conn.cursor() as cur_check:
                cur_check.execute("SELECT COUNT(*) AS count FROM signals WHERE closed_at IS NULL;")
                open_count = (cur_check.fetchone() or {}).get('count', 0)
            
            if open_count >= MAX_OPEN_TRADES:
                logger.info(f"⚠️ [Main] تم الوصول للحد الأقصى للصفقات المفتوحة ({open_count}). في انتظار...")
                time.sleep(get_interval_minutes(SIGNAL_GENERATION_TIMEFRAME) * 60)
                continue

            slots_available = MAX_OPEN_TRADES - open_count
            for symbol in symbols_to_scan:
                if slots_available <= 0: break
                logger.debug(f"🔍 [Main] مسح {symbol}...")
                with conn.cursor() as symbol_cur:
                    symbol_cur.execute("SELECT 1 FROM signals WHERE symbol = %s AND closed_at IS NULL LIMIT 1;", (symbol,))
                    if symbol_cur.fetchone():
                        continue
                
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

        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as main_err:
            logger.error(f"❌ [Main] خطأ غير متوقع في الحلقة الرئيسية: {main_err}", exc_info=True)
            time.sleep(120)

def get_interval_minutes(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == 'm': return value
    if unit == 'h': return value * 60
    if unit == 'd': return value * 24 * 60
    return 0

def cleanup_resources():
    if conn: conn.close()
    logger.info("✅ [Cleanup] تم إغلاق الموارد.")

# ---------------------- نقطة الدخول الرئيسية ----------------------
if __name__ == "__main__":
    logger.info("🚀 بدء بوت إشارات تداول العملات الرقمية...")
    ws_thread, tracker_thread, main_bot_thread = None, None, None
    try:
        init_db()
        
        ws_thread = Thread(target=run_ticker_socket_manager, daemon=True, name="WebSocketThread")
        ws_thread.start()
        logger.info("✅ [Main] تم بدء مؤشر WebSocket. انتظار 5 ثوانٍ...")
        time.sleep(5)

        tracker_thread = Thread(target=track_signals, daemon=True, name="TrackerThread")
        tracker_thread.start()
        logger.info("✅ [Main] تم بدء متتبع الإشارات.")

        main_bot_thread = Thread(target=main_loop, daemon=True, name="MainBotLoopThread")
        main_bot_thread.start()
        logger.info("✅ [Main] تم بدء حلقة البوت الرئيسية.")

        # Flask يعمل في المؤشر الرئيسي لإبقاء البرنامج حيًا
        run_flask()

    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 [Main] طلب إيقاف. إيقاف التشغيل...")
    except Exception as startup_err:
        logger.critical(f"❌ [Main] حدث خطأ فادح أثناء بدء التشغيل: {startup_err}", exc_info=True)
    finally:
        cleanup_resources()
        logger.info("👋 [Main] تم إيقاف بوت إشارات تداول العملات الرقمية.")
        os._exit(0)
