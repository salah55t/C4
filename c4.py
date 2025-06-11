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
from threading import Thread, Lock
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Tuple, Any, Union

# ---------------------- إعداد التسجيل ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_live.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBot')

# ---------------------- تحميل المتغيرات البيئية ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
except Exception as e:
     logger.critical(f"❌ فشل تحميل متغيرات البيئة الأساسية: {e}")
     exit(1)

# ---------------------- الثوابت والمتغيرات العامة ----------------------
TRADE_VALUE: float = 10.0
MAX_OPEN_TRADES: int = 5
SIGNAL_TIMEFRAME: str = '15m'
SIGNAL_LOOKBACK_DAYS: int = 5 # More data for indicator stability

# Indicator Parameters (Must match ml.py)
RSI_PERIOD: int = 9
RSI_MOMENTUM_LOOKBACK_CANDLES: int = 2
ENTRY_ATR_PERIOD: int = 10
SUPERTREND_PERIOD: int = 10
SUPERTREND_MULTIPLIER: float = 3.0
TENKAN_PERIOD: int = 9
KIJUN_PERIOD: int = 26
SENKOU_SPAN_B_PERIOD: int = 52
CHIKOU_LAG: int = 26
FIB_SR_LOOKBACK_WINDOW: int = 50

MIN_PROFIT_MARGIN_PCT: float = 1.0 # Minimum profit for a signal to be considered
MIN_VOLUME_15M_USDT: float = 50000.0

# Base model name from training script
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V2'

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
ticker_data: Dict[str, float] = {}
ml_models_cache: Dict[str, Any] = {}
db_lock = Lock() # Lock for thread-safe database operations

# ---------------------- عميل Binance والإعداد ----------------------
try:
    logger.info("ℹ️ [Binance] تهيئة عميل Binance...")
    client = Client(API_KEY, API_SECRET)
    client.ping()
    logger.info(f"✅ [Binance] تم تهيئة عميل Binance بنجاح.")
except (BinanceAPIException, BinanceRequestException) as e:
    logger.critical(f"❌ [Binance] خطأ في واجهة برمجة تطبيقات أو طلب Binance: {e}")
    exit(1)

# ---------------------- دوال المؤشرات الفنية (مطابقة للتدريب) ----------------------
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
        return pd.Series(index=df_btc.index if df_btc is not None else None, data=0.0)

    df_btc_copy = df_btc.copy()
    ema20 = calculate_ema(df_btc_copy['close'], 20)
    ema50 = calculate_ema(df_btc_copy['close'], 50)
    ema_df = pd.DataFrame({'ema20': ema20, 'ema50': ema50, 'close': df_btc_copy['close']}).dropna()

    trend_series = pd.Series(index=ema_df.index, data=0.0)
    trend_series[(ema_df['close'] > ema_df['ema20']) & (ema_df['ema20'] > ema_df['ema50'])] = 1.0
    trend_series[(ema_df['close'] < ema_df['ema20']) & (ema_df['ema20'] < ema_df['ema50'])] = -1.0
    return trend_series.reindex(df_btc.index).fillna(0.0)

# ---------------------- اتصال قاعدة البيانات وتحميل النموذج ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    """Initializes the database connection."""
    global conn
    logger.info("[DB] بدء تهيئة قاعدة البيانات...")
    with db_lock:
        if conn and conn.closed == 0:
            logger.info("[DB] الاتصال بقاعدة البيانات موجود بالفعل ونشط.")
            return
        for attempt in range(retries):
            try:
                conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
                logger.info("✅ [DB] تم الاتصال بقاعدة البيانات بنجاح.")
                return
            except OperationalError as op_err:
                logger.error(f"❌ [DB] خطأ في التشغيل أثناء الاتصال (المحاولة {attempt + 1}): {op_err}")
            except Exception as e:
                logger.critical(f"❌ [DB] فشل غير متوقع في تهيئة قاعدة البيانات (المحاولة {attempt + 1}): {e}", exc_info=True)
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logger.critical("❌ [DB] فشلت جميع محاولات الاتصال بقاعدة البيانات.")
                raise e

def get_db_connection() -> Optional[psycopg2.extensions.connection]:
    """Gets a thread-safe database connection, reconnecting if necessary."""
    global conn
    with db_lock:
        try:
            if conn is None or conn.closed != 0:
                logger.warning("⚠️ [DB] الاتصال مغلق أو غير موجود. إعادة التهيئة...")
                init_db()
            # Perform a simple query to ensure the connection is alive
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
            return conn
        except (OperationalError, InterfaceError, psycopg2.Error) as e:
            logger.error(f"❌ [DB] فقد الاتصال بقاعدة البيانات ({e}). محاولة إعادة الاتصال...")
            try:
                init_db()
                return conn
            except Exception as recon_err:
                logger.error(f"❌ [DB] فشلت محاولة إعادة الاتصال: {recon_err}")
                return None

def load_ml_model_from_db(symbol: str) -> Optional[Dict]:
    """Loads the latest trained ML model and scaler for a symbol from the cache or DB."""
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_name in ml_models_cache:
        logger.debug(f"ℹ️ [ML Model] النموذج '{model_name}' موجود بالفعل في الذاكرة.")
        return ml_models_cache[model_name]

    db_conn = get_db_connection()
    if not db_conn:
        logger.error(f"❌ [ML Model] لا يمكن تحميل نموذج ML لـ {symbol} بسبب مشكلة في الاتصال.")
        return None

    try:
        with db_conn.cursor() as db_cur:
            db_cur.execute("SELECT model_data FROM ml_models WHERE model_name = %s ORDER BY trained_at DESC LIMIT 1;", (model_name,))
            result = db_cur.fetchone()
            if result and result.get('model_data'):
                # The stored object is a dictionary {'model': ..., 'scaler': ...}
                model_and_scaler = pickle.loads(result['model_data'])
                ml_models_cache[model_name] = model_and_scaler
                logger.info(f"✅ [ML Model] تم تحميل نموذج ML والمحول '{model_name}' من قاعدة البيانات بنجاح.")
                return model_and_scaler
            else:
                logger.warning(f"⚠️ [ML Model] لم يتم العثور على نموذج ML بالاسم '{model_name}' في قاعدة البيانات.")
                return None
    except Exception as e:
        logger.error(f"❌ [ML Model] خطأ غير متوقع أثناء تحميل نموذج ML لـ {symbol}: {e}", exc_info=True)
        return None

# ---------------------- WebSocket & Data Fetching ----------------------
def handle_ticker_message(msg: Dict[str, Any]) -> None:
    """Handles incoming WebSocket mini-ticker messages."""
    try:
        if 'e' in msg and msg['e'] == 'error':
            logger.error(f"❌ [WS] رسالة خطأ من WebSocket: {msg.get('m', 'No details')}")
            return
        if isinstance(msg, list):
            for item in msg:
                symbol = item.get('s')
                price_str = item.get('c')
                if symbol and price_str and 'USDT' in symbol:
                    try:
                        ticker_data[symbol] = float(price_str)
                    except (ValueError, TypeError):
                        pass
    except Exception as e:
        logger.error(f"❌ [WS] خطأ في معالجة رسالة المؤشر: {e}", exc_info=True)

def run_ticker_socket_manager() -> None:
    """Manages the WebSocket connection for real-time ticker prices."""
    while True:
        try:
            logger.info("ℹ️ [WS] بدء مدير WebSocket لأسعار المؤشرات...")
            twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
            twm.start()
            stream_name = twm.start_miniticker_socket(callback=handle_ticker_message)
            logger.info(f"✅ [WS] تم بدء تدفق WebSocket: {stream_name}")
            twm.join() # This will block until the socket manager stops
        except Exception as e:
            logger.error(f"❌ [WS] خطأ فادح في مدير WebSocket: {e}. إعادة التشغيل في 15 ثانية...", exc_info=True)
        logger.warning("⚠️ [WS] تم إيقاف مدير WebSocket. إعادة التشغيل...")
        time.sleep(15)

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """Fetches historical data for signal generation."""
    if not client: return None
    try:
        start_str = (datetime.utcnow() - timedelta(days=days)).strftime("%d %b %Y %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None

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
        return df
    except (BinanceAPIException, BinanceRequestException) as e:
        logger.error(f"❌ [Data] خطأ في Binance API أو الشبكة لـ {symbol}: {e}")
        return None

# ---------------------- Trading Strategy & Signal Generation ----------------------
class TradingStrategy:
    """Encapsulates the ML-based trading strategy."""
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model_data = load_ml_model_from_db(symbol)
        if self.model_data:
            self.model = self.model_data['model']
            self.scaler = self.model_data['scaler']
            self.feature_columns = self.model.feature_name_
        else:
            self.model = None
            self.scaler = None
            self.feature_columns = []
            logger.warning(f"⚠️ [Strategy {symbol}] لم يتم تحميل نموذج ML. لن تتمكن الإستراتيجية من إنشاء إشارات.")

    def _prepare_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepares the complete feature set for prediction."""
        if df is None or df.empty: return None
        df_calc = df.copy()

        # Feature Engineering (must be identical to training script)
        df_calc['volume_15m_avg'] = df_calc['volume'].rolling(window=1).mean()
        delta = df_calc['close'].diff()
        gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df_calc['rsi'] = (100 - (100 / (1 + rs))).ffill().fillna(50)
        df_calc['rsi_momentum_bullish'] = ((df_calc['rsi'] > df_calc['rsi'].shift(1)) & (df_calc['rsi'].shift(1) > df_calc['rsi'].shift(2)) & (df_calc['rsi'] > 50)).astype(int)
        df_calc = calculate_atr_indicator(df_calc, ENTRY_ATR_PERIOD)
        df_calc = calculate_supertrend(df_calc, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
        # BTC Trend
        btc_df = fetch_historical_data("BTCUSDT", interval=SIGNAL_TIMEFRAME, days=SIGNAL_LOOKBACK_DAYS)
        if btc_df is not None:
            btc_trend_series = _calculate_btc_trend_feature(btc_df)
            if btc_trend_series is not None:
                df_calc = df_calc.join(btc_trend_series.rename('btc_trend_feature'), how='left')
                df_calc['btc_trend_feature'].fillna(0.0, inplace=True)
        if 'btc_trend_feature' not in df_calc.columns: df_calc['btc_trend_feature'] = 0.0
        # Ichimoku
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
        # S/R and Fib
        rolling_high = df_calc['high'].rolling(window=FIB_SR_LOOKBACK_WINDOW)
        rolling_low = df_calc['low'].rolling(window=FIB_SR_LOOKBACK_WINDOW)
        swing_high = rolling_high.max()
        swing_low = rolling_low.min()
        price_range = swing_high - swing_low
        price_range[price_range == 0] = np.nan
        df_calc['price_distance_to_recent_low_norm'] = (df_calc['close'] - swing_low) / price_range
        df_calc['price_distance_to_recent_high_norm'] = (swing_high - df_calc['close']) / price_range
        fib_50 = swing_high - (price_range * 0.5)
        df_calc['is_price_above_fib_50'] = (df_calc['close'] > fib_50).astype(int)

        return df_calc.dropna(subset=self.feature_columns + ['atr', 'supertrend_direction'])

    def generate_buy_signal(self, df_hist: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generates a buy signal based on ML prediction and additional filters."""
        if not self.model or not self.scaler:
            return None # Cannot generate signal without a model

        df_features = self._prepare_features(df_hist)
        if df_features is None or df_features.empty:
            logger.debug(f"ℹ️ [Strategy {self.symbol}] فشل إعداد الميزات أو كانت فارغة.")
            return None

        last_row = df_features.iloc[-1]
        current_price = ticker_data.get(self.symbol)
        if not current_price:
            logger.warning(f"⚠️ [Strategy {self.symbol}] السعر الحالي غير متاح من المؤشر.")
            return None

        # 1. ML Prediction
        try:
            features_to_predict = last_row[self.feature_columns]
            features_scaled = self.scaler.transform([features_to_predict.values])
            prediction = self.model.predict(features_scaled)[0]
            pred_proba = self.model.predict_proba(features_scaled)[0][1] # Probability of '1' class
        except Exception as e:
            logger.error(f"❌ [Strategy {self.symbol}] خطأ أثناء توقع النموذج: {e}")
            return None

        if prediction != 1:
            logger.debug(f"ℹ️ [Strategy {self.symbol}] نموذج ML لم يتوقع صعودًا (Pred={prediction}). تم رفض الإشارة.")
            return None
        logger.info(f"✨ [Strategy {self.symbol}] توقع نموذج ML صعوديًا (Prob={pred_proba:.2f})... التحقق من المرشحات...")

        # 2. Additional Filters
        if last_row['supertrend_direction'] != 1:
            logger.info(f"ℹ️ [Strategy {self.symbol}] فشل مرشح Supertrend ({last_row['supertrend_direction']}). تم الرفض.")
            return None
        if last_row['btc_trend_feature'] == -1.0:
            logger.info(f"ℹ️ [Strategy {self.symbol}] فشل مرشح اتجاه BTC ({last_row['btc_trend_feature']}). تم الرفض.")
            return None
        volume_15m = last_row.get('volume', 0) * last_row.get('close', 0)
        if volume_15m < MIN_VOLUME_15M_USDT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] فشل مرشح السيولة ({volume_15m:,.0f} USDT). تم الرفض.")
            return None

        # 3. Calculate Target and Stop Loss
        current_atr = last_row['atr']
        initial_target = current_price + (1.5 * current_atr)
        profit_margin = ((initial_target / current_price) - 1) * 100
        if profit_margin < MIN_PROFIT_MARGIN_PCT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] فشل مرشح هامش الربح ({profit_margin:.2f}%). تم الرفض.")
            return None

        initial_stop_loss = last_row['supertrend'] # Use Supertrend line as stop loss

        signal = {
            'symbol': self.symbol,
            'entry_price': current_price,
            'initial_target': initial_target,
            'current_target': initial_target,
            'stop_loss': initial_stop_loss,
            'strategy_name': f"{BASE_ML_MODEL_NAME}_Filtered",
            'volume_15m': volume_15m,
            'signal_details': {'ml_proba': pred_proba}
        }
        logger.info(f"✅ [Strategy {self.symbol}] تم تأكيد إشارة الشراء! السعر: {current_price:.6f}, الهدف: {initial_target:.6f}, وقف الخسارة: {initial_stop_loss:.6f}")
        return signal

# ---------------------- Telegram Functions ----------------------
def send_telegram_message(text: str, reply_markup: Optional[Dict] = None) -> None:
    """Sends a message to the configured Telegram chat."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': CHAT_ID, 'text': text, 'parse_mode': 'Markdown', 'disable_web_page_preview': True}
    if reply_markup:
        payload['reply_markup'] = json.dumps(reply_markup)
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

# ... (Other Telegram functions: send_telegram_alert, send_tracking_notification, etc. would go here, simplified for brevity but essential in the full script)
# ... (Full report generation function would also be here)

# ---------------------- Main Loop and Tracking ----------------------
def track_signals() -> None:
    """Tracks open signals, checking for target or stop-loss hits."""
    logger.info("ℹ️ [Tracker] بدء عملية تتبع الإشارات المفتوحة...")
    while True:
        try:
            db_conn = get_db_connection()
            if not db_conn:
                logger.warning("⚠️ [Tracker] تخطي دورة التتبع بسبب مشكلة في الاتصال بقاعدة البيانات.")
                time.sleep(15)
                continue

            with db_conn.cursor() as track_cur:
                track_cur.execute("SELECT id, symbol, entry_price, current_target, stop_loss, entry_time FROM signals WHERE closed_at IS NULL;")
                open_signals = track_cur.fetchall()

            if not open_signals:
                time.sleep(10)
                continue

            for signal in open_signals:
                current_price = ticker_data.get(signal['symbol'])
                if not current_price: continue

                # Check for Stop Loss Hit
                if signal['stop_loss'] and current_price <= signal['stop_loss']:
                    profit_pct = ((signal['stop_loss'] / signal['entry_price']) - 1) * 100
                    with db_conn.cursor() as update_cur:
                        update_cur.execute("UPDATE signals SET closing_price = %s, closed_at = NOW(), profit_percentage = %s, achieved_target = FALSE WHERE id = %s;",
                                           (signal['stop_loss'], profit_pct, signal['id']))
                    db_conn.commit()
                    logger.info(f"🛑 [Tracker] تم ضرب وقف الخسارة لـ {signal['symbol']} (ID:{signal['id']}).")
                    send_telegram_message(f"🛑 *تم ضرب وقف الخسارة*\n- الزوج: `{signal['symbol']}`\n- الخسارة: `{profit_pct:.2f}%`")
                    continue

                # Check for Target Hit
                if current_price >= signal['current_target']:
                    profit_pct = ((signal['current_target'] / signal['entry_price']) - 1) * 100
                    with db_conn.cursor() as update_cur:
                        update_cur.execute("UPDATE signals SET closing_price = %s, closed_at = NOW(), profit_percentage = %s, achieved_target = TRUE WHERE id = %s;",
                                           (signal['current_target'], profit_pct, signal['id']))
                    db_conn.commit()
                    logger.info(f"🎯 [Tracker] تم الوصول إلى الهدف لـ {signal['symbol']} (ID:{signal['id']}).")
                    send_telegram_message(f"✅ *تم الوصول إلى الهدف*\n- الزوج: `{signal['symbol']}`\n- الربح: `+{profit_pct:.2f}%`")

        except (psycopg2.Error, Exception) as e:
            logger.error(f"❌ [Tracker] خطأ في دورة تتبع الإشارة: {e}", exc_info=True)
            time.sleep(30) # Wait longer on error

        time.sleep(3) # Short sleep between tracking cycles

def main_loop() -> None:
    """Main loop to scan for new trading signals."""
    symbols_to_scan = get_crypto_symbols() # Assume this function is defined
    if not symbols_to_scan:
        logger.critical("❌ [Main] لم يتم تحميل أي رموز صالحة. لا يمكن المتابعة.")
        return

    logger.info(f"✅ [Main] تم تحميل {len(symbols_to_scan)} رمزًا صالحًا للمسح.")

    while True:
        try:
            logger.info(f"🔄 [Main] بدء دورة مسح السوق - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            db_conn = get_db_connection()
            if not db_conn:
                logger.error("❌ [Main] تخطي دورة المسح بسبب فشل الاتصال بقاعدة البيانات.")
                time.sleep(60)
                continue

            with db_conn.cursor() as cur_check:
                cur_check.execute("SELECT COUNT(*) AS count FROM signals WHERE closed_at IS NULL;")
                open_count = (cur_check.fetchone() or {}).get('count', 0)

            if open_count >= MAX_OPEN_TRADES:
                logger.info(f"⚠️ [Main] تم الوصول إلى الحد الأقصى للصفقات ({MAX_OPEN_TRADES}). الانتظار...")
                time.sleep(60 * 5) # Wait 5 minutes
                continue

            for symbol in symbols_to_scan:
                with db_conn.cursor() as symbol_cur:
                    symbol_cur.execute("SELECT 1 FROM signals WHERE symbol = %s AND closed_at IS NULL LIMIT 1;", (symbol,))
                    if symbol_cur.fetchone():
                        logger.debug(f"ℹ️ [Main] تخطي {symbol}، توجد صفقة مفتوحة بالفعل.")
                        continue

                df_hist = fetch_historical_data(symbol, interval=SIGNAL_TIMEFRAME, days=SIGNAL_LOOKBACK_DAYS)
                if df_hist is None or df_hist.empty:
                    continue

                strategy = TradingStrategy(symbol)
                potential_signal = strategy.generate_buy_signal(df_hist)

                if potential_signal:
                    with db_conn.cursor() as insert_cur:
                        insert_cur.execute(
                            sql.SQL("INSERT INTO signals (symbol, entry_price, initial_target, current_target, stop_loss, strategy_name, volume_15m, signal_details) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);"),
                            (
                                potential_signal['symbol'], potential_signal['entry_price'], potential_signal['initial_target'],
                                potential_signal['current_target'], potential_signal['stop_loss'], potential_signal['strategy_name'],
                                potential_signal['volume_15m'], json.dumps(potential_signal['signal_details'])
                            )
                        )
                    db_conn.commit()
                    logger.info(f"✅ [Main] تم إدراج إشارة جديدة لـ {symbol} في قاعدة البيانات.")
                    # Send alert (full function would be here)
                    send_telegram_message(f"💡 *إشارة تداول جديدة*\n- الزوج: `{potential_signal['symbol']}`\n- الدخول: `${potential_signal['entry_price']:.6g}`\n- الهدف: `${potential_signal['initial_target']:.6g}`")
                    time.sleep(2) # Avoid rate limiting

        except Exception as main_err:
            logger.error(f"❌ [Main] خطأ غير متوقع في الحلقة الرئيسية: {main_err}", exc_info=True)
            time.sleep(120)

        logger.info(f"⏳ [Main] انتظار 15 دقيقة للدورة التالية...")
        time.sleep(60 * 15)

# ---------------------- Entry Point ----------------------
if __name__ == "__main__":
    logger.info("🚀 بدء بوت إشارات تداول العملات الرقمية...")
    try:
        init_db()

        # Start WebSocket in a background thread
        ws_thread = Thread(target=run_ticker_socket_manager, daemon=True, name="WebSocketThread")
        ws_thread.start()
        logger.info("✅ [Main] تم بدء مؤشر WebSocket. انتظار 5 ثوانٍ لتهيئة البيانات...")
        time.sleep(5)

        # Start Signal Tracker in a background thread
        tracker_thread = Thread(target=track_signals, daemon=True, name="TrackerThread")
        tracker_thread.start()
        logger.info("✅ [Main] تم بدء متتبع الإشارات.")

        # Start the main bot loop (this will run in the main thread)
        # In a full version with Flask, this would also be in a thread.
        main_loop()

    except Exception as startup_err:
        logger.critical(f"❌ [Main] حدث خطأ فادح أثناء بدء التشغيل: {startup_err}", exc_info=True)
    finally:
        logger.info("🛑 [Main] إيقاف تشغيل البرنامج...")
        if conn: conn.close()
        logger.info("👋 [Main] تم إيقاف بوت إشارات تداول العملات الرقمية.")
        os._exit(0)

