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
from flask import Flask, request, Response, jsonify, send_from_directory
from flask_cors import CORS
from threading import Thread
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler

# ---------------------- إعداد التسجيل ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot.log', encoding='utf-8'),
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
MIN_VOLUME_15M_USDT: float = 50000.0 
RELATIVE_VOLUME_LOOKBACK: int = 30
RELATIVE_VOLUME_FACTOR: float = 1.5
RSI_PERIOD: int = 9
ENTRY_ATR_PERIOD: int = 10
SUPERTRAND_PERIOD: int = 10
SUPERTRAND_MULTIPLIER: float = 3.0
TENKAN_PERIOD: int = 9
KIJUN_PERIOD: int = 26
SENKOU_SPAN_B_PERIOD: int = 52
CHIKOU_LAG: int = 26
FIB_SR_LOOKBACK_WINDOW: int = 50
VOLUME_LOOKBACK_CANDLES: int = 1 
RSI_MOMENTUM_LOOKBACK_CANDLES: int = 2
PRICE_CHANGE_THRESHOLD_FOR_TARGET: float = 0.005
MIN_PROFIT_MARGIN_PCT: float = 1.0

# <-- تغيير اسم النموذج ليطابق الاسم المستخدم في التدريب
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V1'

conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
ticker_data: Dict[str, float] = {}
ml_models_cache: Dict[str, Any] = {}

# ---------------------- Binance Client & DB Setup ----------------------
try:
    logger.info("ℹ️ [Binance] تهيئة عميل Binance...")
    client = Client(API_KEY, API_SECRET)
    client.ping()
except Exception as e:
    logger.critical(f"❌ [Binance] فشل غير متوقع في تهيئة عميل Binance: {e}")
    exit(1)

# (All functions from here down are mostly unchanged, as the model loading is generic)
def init_db(retries: int = 5, delay: int = 5) -> None:
    global conn
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
                    achieved_target BOOLEAN DEFAULT FALSE, closing_price DOUBLE PRECISION, closed_at TIMESTAMP,
                    profit_percentage DOUBLE PRECISION, strategy_name TEXT, signal_details JSONB, stop_loss DOUBLE PRECISION);
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
    global conn
    try:
        if conn is None or conn.closed != 0: init_db()
        else: conn.cursor().execute("SELECT 1;")
        return True
    except (OperationalError, InterfaceError):
        try: init_db()
        except Exception: return False
        return True
    return False

def load_ml_model_bundle_from_db(symbol: str) -> Optional[Dict[str, Any]]:
    global ml_models_cache
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_name in ml_models_cache:
        logger.info(f"✅ [ML Model] تم تحميل حزمة نموذج '{model_name}' من الذاكرة المؤقتة.")
        return ml_models_cache[model_name]
    if not check_db_connection() or not conn: return None
    try:
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT model_data FROM ml_models WHERE model_name = %s ORDER BY trained_at DESC LIMIT 1;", (model_name,))
            result = db_cur.fetchone()
            if result and result['model_data']:
                model_bundle = pickle.loads(result['model_data'])
                ml_models_cache[model_name] = model_bundle
                logger.info(f"✅ [ML Model] تم تحميل حزمة نموذج '{model_name}' من قاعدة البيانات.")
                return model_bundle
            logger.warning(f"⚠️ [ML Model] لم يتم العثور على نموذج لـ {symbol} بالاسم '{model_name}'.")
            return None
    except Exception as e:
        logger.error(f"❌ [ML Model] خطأ أثناء تحميل حزمة نموذج ML لـ {symbol}: {e}", exc_info=True)
        return None
# ...
# ... All other functions (indicators, websocket, trading logic, Flask routes, etc.)
# ... are IDENTICAL to the previous version and are omitted here for brevity.
# ... The key change was the BASE_ML_MODEL_NAME variable above.
# ...
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        start_str = (datetime.utcnow() - timedelta(days=days + 1)).strftime("%Y-%m-%d %H:%M:%S")
        binance_interval = Client.KLINE_INTERVAL_15MINUTE
        klines = client.get_historical_klines(symbol, binance_interval, start_str)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume', 'quote_volume']].dropna()
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        logger.error(f"❌ [Data] خطأ أثناء جلب البيانات التاريخية لـ {symbol}: {e}")
        return None
def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    if series is None or series.isnull().all() or len(series) < span: return pd.Series(index=series.index if series is not None else None, dtype=float)
    return series.ewm(span=span, adjust=False).mean()
def calculate_rsi_indicator(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    df = df.copy(); delta = df['close'].diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0); avg_gain = gain.ewm(com=period - 1, adjust=False).mean(); avg_loss = loss.ewm(com=period - 1, adjust=False).mean(); rs = avg_gain / avg_loss.replace(0, np.nan); df['rsi'] = (100 - (100 / (1 + rs))).ffill().fillna(50); return df
def calculate_atr_indicator(df: pd.DataFrame, period: int = ENTRY_ATR_PERIOD) -> pd.DataFrame:
    df = df.copy(); high_low = df['high'] - df['low']; high_close_prev = (df['high'] - df['close'].shift(1)).abs(); low_close_prev = (df['low'] - df['close'].shift(1)).abs(); tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False); df['atr'] = tr.ewm(span=period, adjust=False).mean(); return df
def calculate_supertrend(df: pd.DataFrame, period: int = SUPERTRAND_PERIOD, multiplier: float = SUPERTRAND_MULTIPLIER) -> pd.DataFrame:
    df = df.copy();
    if 'atr' not in df.columns: df = calculate_atr_indicator(df, period=period)
    df['basic_upper_band'] = ((df['high'] + df['low']) / 2) + (multiplier * df['atr']); df['basic_lower_band'] = ((df['high'] + df['low']) / 2) - (multiplier * df['atr']); df['final_upper_band'], df['final_lower_band'] = 0.0, 0.0; df['supertrend'], df['supertrend_direction'] = 0.0, 0
    for i in range(1, len(df)):
        if df['basic_upper_band'].iloc[i] < df['final_upper_band'].iloc[i-1] or df['close'].iloc[i-1] > df['final_upper_band'].iloc[i-1]: df.loc[df.index[i], 'final_upper_band'] = df['basic_upper_band'].iloc[i]
        else: df.loc[df.index[i], 'final_upper_band'] = df['final_upper_band'].iloc[i-1]
        if df['basic_lower_band'].iloc[i] > df['final_lower_band'].iloc[i-1] or df['close'].iloc[i-1] < df['final_lower_band'].iloc[i-1]: df.loc[df.index[i], 'final_lower_band'] = df['basic_lower_band'].iloc[i]
        else: df.loc[df.index[i], 'final_lower_band'] = df['final_lower_band'].iloc[i-1]
        if df['supertrend_direction'].iloc[i-1] == 1:
            if df['close'].iloc[i] < df['final_upper_band'].iloc[i]: df.loc[df.index[i], 'supertrend'], df.loc[df.index[i], 'supertrend_direction'] = df['final_upper_band'].iloc[i], -1
            else: df.loc[df.index[i], 'supertrend'], df.loc[df.index[i], 'supertrend_direction'] = df['final_lower_band'].iloc[i], 1
        elif df['supertrend_direction'].iloc[i-1] == -1:
            if df['close'].iloc[i] > df['final_lower_band'].iloc[i]: df.loc[df.index[i], 'supertrend'], df.loc[df.index[i], 'supertrend_direction'] = df['final_lower_band'].iloc[i], 1
            else: df.loc[df.index[i], 'supertrend'], df.loc[df.index[i], 'supertrend_direction'] = df['final_upper_band'].iloc[i], -1
        else:
            if df['close'].iloc[i] > df['final_lower_band'].iloc[i]: df.loc[df.index[i], 'supertrend_direction'] = 1
            elif df['close'].iloc[i] < df['final_upper_band'].iloc[i]: df.loc[df.index[i], 'supertrend_direction'] = -1
            df.loc[df.index[i], 'supertrend'] = df.loc[df.index[i], 'final_lower_band'] if df.loc[df.index[i], 'supertrend_direction'] == 1 else df.loc[df.index[i], 'final_upper_band']
    df.drop(columns=['basic_upper_band', 'basic_lower_band', 'final_upper_band', 'final_lower_band'], inplace=True, errors='ignore'); return df
def _calculate_btc_trend_feature(df_btc: pd.DataFrame) -> Optional[pd.Series]:
    if df_btc is None or df_btc.empty or len(df_btc) < 55: return None
    ema20, ema50 = calculate_ema(df_btc['close'], 20), calculate_ema(df_btc['close'], 50); ema_df = pd.DataFrame({'ema20': ema20, 'ema50': ema50, 'close': df_btc['close']}).dropna(); trend_series = pd.Series(index=ema_df.index, data=0.0)
    trend_series[(ema_df['close'] > ema_df['ema20']) & (ema_df['ema20'] > ema_df['ema50'])] = 1.0; trend_series[(ema_df['close'] < ema_df['ema20']) & (ema_df['ema20'] < ema_df['ema50'])] = -1.0; return trend_series.reindex(df_btc.index).fillna(0.0)
def calculate_ichimoku_cloud(df: pd.DataFrame, tenkan_period: int = TENKAN_PERIOD, kijun_period: int = KIJUN_PERIOD, senkou_span_b_period: int = SENKOU_SPAN_B_PERIOD, chikou_lag: int = CHIKOU_LAG) -> pd.DataFrame:
    df_ichimoku = df.copy(); df_ichimoku['tenkan_sen'] = (df_ichimoku['high'].rolling(window=tenkan_period).max() + df_ichimoku['low'].rolling(window=tenkan_period).min()) / 2; df_ichimoku['kijun_sen'] = (df_ichimoku['high'].rolling(window=kijun_period).max() + df_ichimoku['low'].rolling(window=kijun_period).min()) / 2; df_ichimoku['senkou_span_a'] = ((df_ichimoku['tenkan_sen'] + df_ichimoku['kijun_sen']) / 2).shift(kijun_period); df_ichimoku['senkou_span_b'] = ((df_ichimoku['high'].rolling(window=senkou_span_b_period).max() + df_ichimoku['low'].rolling(window=senkou_span_b_period).min()) / 2).shift(kijun_period); df_ichimoku['chikou_span'] = df_ichimoku['close'].shift(-chikou_lag); df_ichimoku['ichimoku_tenkan_kijun_cross_signal'] = np.where((df_ichimoku['tenkan_sen'] > df_ichimoku['kijun_sen']) & (df_ichimoku['tenkan_sen'].shift(1) < df_ichimoku['kijun_sen'].shift(1)), 1, np.where((df_ichimoku['tenkan_sen'] < df_ichimoku['kijun_sen']) & (df_ichimoku['tenkan_sen'].shift(1) > df_ichimoku['kijun_sen'].shift(1)), -1, 0)); df_ichimoku['ichimoku_price_cloud_position'] = np.where(df_ichimoku['close'] > df_ichimoku[['senkou_span_a', 'senkou_span_b']].max(axis=1), 1, np.where(df_ichimoku['close'] < df_ichimoku[['senkou_span_a', 'senkou_span_b']].min(axis=1), -1, 0)); df_ichimoku['ichimoku_cloud_outlook'] = np.where(df_ichimoku['senkou_span_a'] > df_ichimoku['senkou_span_b'], 1, np.where(df_ichimoku['senkou_span_a'] < df_ichimoku['senkou_span_b'], -1, 0)); return df_ichimoku
def calculate_fibonacci_features(df: pd.DataFrame, lookback_window: int = FIB_SR_LOOKBACK_WINDOW) -> pd.DataFrame:
    df_fib = df.copy()
    for i in range(lookback_window - 1, len(df_fib)):
        window_df = df_fib.iloc[i - lookback_window + 1 : i + 1]; swing_high, swing_low = window_df['high'].max(), window_df['low'].min(); price_range = swing_high - swing_low
        if price_range > 0:
            current_close = df_fib['close'].iloc[i]; fib_0_500 = swing_high - (price_range * 0.500)
            df_fib.loc[df_fib.index[i], 'fib_236_retrace_dist_norm'] = (current_close - (swing_high - price_range * 0.236)) / price_range
            df_fib.loc[df_fib.index[i], 'fib_382_retrace_dist_norm'] = (current_close - (swing_high - price_range * 0.382)) / price_range
            df_fib.loc[df_fib.index[i], 'fib_618_retrace_dist_norm'] = (current_close - (swing_high - price_range * 0.618)) / price_range
            df_fib.loc[df_fib.index[i], 'is_price_above_fib_50'] = 1 if current_close > fib_0_500 else 0
    return df_fib
def calculate_support_resistance_features(df: pd.DataFrame, lookback_window: int = FIB_SR_LOOKBACK_WINDOW) -> pd.DataFrame:
    df_sr = df.copy()
    for i in range(lookback_window - 1, len(df_sr)):
        window_df = df_sr.iloc[i - lookback_window + 1 : i + 1]; recent_high, recent_low = window_df['high'].max(), window_df['low'].min(); price_range = recent_high - recent_low
        if price_range > 0:
            current_close = df_sr['close'].iloc[i]
            df_sr.loc[df_sr.index[i], 'price_distance_to_recent_low_norm'] = (current_close - recent_low) / price_range
            df_sr.loc[df_sr.index[i], 'price_distance_to_recent_high_norm'] = (recent_high - current_close) / price_range
    return df_sr
def convert_np_values(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_np_values(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_np_values(i) for i in obj]
    if pd.isna(obj): return None
    return obj
def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
    global ticker_data
    try:
        data = msg.get('data', msg) if isinstance(msg, dict) else msg
        if not isinstance(data, list): data = [data]
        for item in data:
            if item.get('s') and 'USDT' in item['s'] and item.get('c'): ticker_data[item['s']] = float(item['c'])
    except Exception as e: logger.error(f"❌ [WS] خطأ في معالجة رسالة المؤشر: {e}")
def run_ticker_socket_manager() -> None:
    while True:
        try:
            logger.info("ℹ️ [WS] بدء مدير WebSocket..."); twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET); twm.start(); twm.start_miniticker_socket(callback=handle_ticker_message); twm.join()
        except Exception as e: logger.error(f"❌ [WS] خطأ فادح في مدير WebSocket: {e}. إعادة التشغيل..."); time.sleep(15)
def get_crypto_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    try:
        with open(os.path.join(os.path.dirname(__file__), filename), 'r', encoding='utf-8') as f: raw_symbols = [line.strip().upper() for line in f if line.strip() and not line.startswith('#')]
        raw_symbols = sorted([f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols]); exchange_info = client.get_exchange_info()
        valid_symbols = {s['symbol'] for s in exchange_info['symbols'] if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'}; return [s for s in raw_symbols if s in valid_symbols]
    except Exception as e: logger.error(f"❌ [Data Validation] خطأ أثناء التحقق من الرموز: {e}"); return []
class ScalpingTradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol; model_bundle = load_ml_model_bundle_from_db(symbol); self.ml_model = model_bundle.get('model') if model_bundle else None; self.scaler = model_bundle.get('scaler') if model_bundle else None
        self.feature_columns_for_ml = ['volume_15m_avg', 'rsi_momentum_bullish', 'btc_trend_feature', 'supertrend_direction','ichimoku_tenkan_kijun_cross_signal', 'ichimoku_price_cloud_position', 'ichimoku_cloud_outlook','fib_236_retrace_dist_norm', 'fib_382_retrace_dist_norm', 'fib_618_retrace_dist_norm','is_price_above_fib_50', 'price_distance_to_recent_low_norm', 'price_distance_to_recent_high_norm']
    def populate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        min_len_required = max(FIB_SR_LOOKBACK_WINDOW, SENKOU_SPAN_B_PERIOD, 55) + 5
        if len(df) < min_len_required: return None
        df_calc = df.copy(); df_calc['volume_15m_avg'] = df_calc['quote_volume'].rolling(window=VOLUME_LOOKBACK_CANDLES, min_periods=1).mean(); df_calc = calculate_rsi_indicator(df_calc, RSI_PERIOD); df_calc['rsi_momentum_bullish'] = 0
        for i in range(RSI_MOMENTUM_LOOKBACK_CANDLES, len(df_calc)):
            rsi_slice = df_calc['rsi'].iloc[i - RSI_MOMENTUM_LOOKBACK_CANDLES : i + 1]
            if not rsi_slice.isnull().any() and np.all(np.diff(rsi_slice) > 0) and rsi_slice.iloc[-1] > 50: df_calc.loc[df_calc.index[i], 'rsi_momentum_bullish'] = 1
        df_calc = calculate_atr_indicator(df_calc, ENTRY_ATR_PERIOD); df_calc = calculate_supertrend(df_calc, SUPERTRAND_PERIOD, SUPERTRAND_MULTIPLIER)
        btc_df = fetch_historical_data("BTCUSDT", interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
        if btc_df is not None: btc_trend = _calculate_btc_trend_feature(btc_df); df_calc = df_calc.merge(btc_trend.rename('btc_trend_feature'), left_index=True, right_index=True, how='left').fillna(0.0)
        else: df_calc['btc_trend_feature'] = 0.0
        df_calc = calculate_ichimoku_cloud(df_calc); df_calc = calculate_fibonacci_features(df_calc); df_calc = calculate_support_resistance_features(df_calc)
        for col in self.feature_columns_for_ml:
            if col not in df_calc.columns: df_calc[col] = np.nan
        df_calc.dropna(subset=self.feature_columns_for_ml, inplace=True); return df_calc if not df_calc.empty else None
    def generate_buy_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        symbol_log_prefix = f"🔍 [Signal Gen {self.symbol}]"
        if df_processed is None or df_processed.empty: return None
        if self.ml_model is None or self.scaler is None: logger.info(f"{symbol_log_prefix} رفض: نموذج ML أو Scaler غير محمل."); return None
        last_row = df_processed.iloc[-1]; current_price = ticker_data.get(self.symbol)
        if current_price is None: return None
        recent_quote_volume = last_row.get('quote_volume')
        if pd.isna(recent_quote_volume) or recent_quote_volume < MIN_VOLUME_15M_USDT: return None
        avg_volume = last_row.get('volume_15m_avg')
        if pd.isna(avg_volume) or last_row.get('quote_volume') < avg_volume * RELATIVE_VOLUME_FACTOR: return None
        try:
            features_df = pd.DataFrame([last_row[self.feature_columns_for_ml]], columns=self.feature_columns_for_ml)
            features_scaled_array = self.scaler.transform(features_df)
            features_scaled_df = pd.DataFrame(features_scaled_array, columns=self.feature_columns_for_ml)
            ml_prediction = self.ml_model.predict(features_scaled_df)[0]
            if ml_prediction != 1: logger.info(f"{symbol_log_prefix} رفض: تنبؤ النموذج {ml_prediction}."); return None
        except Exception as e: logger.error(f"❌ {symbol_log_prefix} خطأ أثناء التنبؤ: {e}", exc_info=True); return None
        current_atr = last_row.get('atr')
        if pd.isna(current_atr) or current_atr <= 0: return None
        initial_target = current_price * (1 + PRICE_CHANGE_THRESHOLD_FOR_TARGET)
        if ((initial_target / current_price) - 1) * 100 < MIN_PROFIT_MARGIN_PCT: return None
        initial_stop_loss = last_row.get('supertrend', current_price - (1.0 * current_atr))
        if initial_stop_loss >= current_price: initial_stop_loss = current_price - (1.0 * current_atr)
        if initial_stop_loss >= current_price: return None
        return {'symbol': self.symbol, 'entry_price': current_price, 'initial_target': initial_target, 'current_target': initial_target, 'stop_loss': max(1e-8, initial_stop_loss), 'strategy_name': 'Scalping_LGBM_RelativeVolume', 'volume_15m': recent_quote_volume, 'signal_details': {'ML_Prediction': 'Buy'}}
def send_telegram_message(target_chat_id: str, text: str, **kwargs):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"; payload = {'chat_id': str(target_chat_id), 'text': text, 'parse_mode': 'Markdown', **kwargs}
    if 'reply_markup' in payload: payload['reply_markup'] = json.dumps(convert_np_values(payload['reply_markup']))
    try: requests.post(url, json=payload, timeout=20).raise_for_status()
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")
def send_telegram_alert(signal_data: Dict[str, Any], timeframe: str) -> None:
    symbol, entry, target, sl = signal_data['symbol'].replace('_', '\\_'), signal_data['entry_price'], signal_data['initial_target'], signal_data['stop_loss']; profit_pct = ((target / entry) - 1) * 100
    message = (f"💡 *إشارة تداول جديدة (LGBM)* 💡\n--------------------\n" f"🪙 **الزوج:** `{symbol}`\n📈 **النوع:** شراء\n🕰️ **الإطار الزمني:** {timeframe}\n" f"➡️ **الدخول:** `${entry:,.8g}`\n🎯 **الهدف:** `${target:,.8g}` ({profit_pct:+.2f}%)\n" f"🛑 **وقف الخسارة:** `${sl:,.8g}`\n--------------------")
    reply_markup = {"inline_keyboard": [[{"text": "📊 فتح لوحة التحكم", "url": WEBHOOK_URL or '#'}]]}; send_telegram_message(CHAT_ID, message, reply_markup=reply_markup)
def insert_signal_into_db(signal: Dict[str, Any]) -> bool:
    if not check_db_connection() or not conn: return False
    try:
        with conn.cursor() as cur_ins:
            cur_ins.execute("INSERT INTO signals (symbol, entry_price, initial_target, current_target, stop_loss, strategy_name, signal_details) VALUES (%s, %s, %s, %s, %s, %s, %s);", (signal['symbol'], signal['entry_price'], signal['initial_target'], signal['current_target'], signal['stop_loss'], signal.get('strategy_name'), json.dumps(convert_np_values(signal.get('signal_details', {})))))
        conn.commit(); return True
    except Exception as e: logger.error(f"❌ [DB Insert] خطأ أثناء إدراج الإشارة: {e}"); conn.rollback(); return False
def track_signals() -> None:
    logger.info("ℹ️ [Tracker] بدء عملية تتبع الإشارات...")
    while True:
        try:
            if not check_db_connection() or not conn: time.sleep(15); continue
            with conn.cursor() as track_cur: track_cur.execute("SELECT id, symbol, entry_price, current_target, stop_loss FROM signals WHERE closed_at IS NULL;"); open_signals = track_cur.fetchall()
            for signal_row in open_signals:
                signal_id, symbol, entry, target, sl = signal_row['id'], signal_row['symbol'], float(signal_row['entry_price']), float(signal_row["current_target"]), float(signal_row["stop_loss"] or 0); price = ticker_data.get(symbol)
                if price is None: continue
                closed, profit_pct = False, 0.0
                if sl and price <= sl: closed, profit_pct, closing_price, achieved = True, ((sl / entry) - 1) * 100, sl, False
                elif price >= target: closed, profit_pct, closing_price, achieved = True, ((target / entry) - 1) * 100, target, True
                if closed:
                    with conn.cursor() as update_cur: update_cur.execute("UPDATE signals SET achieved_target = %s, closing_price = %s, closed_at = NOW(), profit_percentage = %s WHERE id = %s;", (achieved, closing_price, profit_pct, signal_id))
                    conn.commit(); send_telegram_message(CHAT_ID, f"{'✅' if achieved else '🛑'} *{'Target Hit' if achieved else 'Stop Loss Hit'}* | `{symbol.replace('_', '\\_')}`\n💰 Profit: {profit_pct:+.2f}%")
            time.sleep(3)
        except Exception as e: logger.error(f"❌ [Tracker] خطأ في دورة التتبع: {e}"); conn.rollback(); time.sleep(30)
def get_interval_minutes(interval: str) -> int:
    unit, value = interval[-1], int(interval[:-1]);
    if unit == 'm': return value
    if unit == 'h': return value * 60
    return 0
def main_loop():
    symbols_to_scan = get_crypto_symbols()
    logger.info(f"✅ [Main] تم تحميل {len(symbols_to_scan)} رمزًا للمسح.")
    while True:
        try:
            logger.info(f"🔄 [Main] بدء دورة مسح السوق...")
            if not check_db_connection() or not conn: time.sleep(60); continue
            with conn.cursor() as cur_check: cur_check.execute("SELECT COUNT(*) AS count FROM signals WHERE closed_at IS NULL;"); open_count = cur_check.fetchone().get('count', 0)
            if open_count >= MAX_OPEN_TRADES: logger.info(f"⚠️ [Main] الحد الأقصى للصفقات المفتوحة ({open_count})."); time.sleep(get_interval_minutes(SIGNAL_GENERATION_TIMEFRAME) * 60); continue
            slots_available = MAX_OPEN_TRADES - open_count
            for symbol in symbols_to_scan:
                if slots_available <= 0: break
                logger.info(f"🔍 [Main] مسح {symbol}...")
                with conn.cursor() as symbol_cur: symbol_cur.execute("SELECT 1 FROM signals WHERE symbol = %s AND closed_at IS NULL LIMIT 1;", (symbol,));
                if symbol_cur.fetchone(): continue
                df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
                if df_hist is None or df_hist.empty: continue
                strategy = ScalpingTradingStrategy(symbol)
                if strategy.ml_model is None: continue
                df_indicators = strategy.populate_indicators(df_hist)
                if df_indicators is None: continue
                potential_signal = strategy.generate_buy_signal(df_indicators)
                if potential_signal and insert_signal_into_db(potential_signal): send_telegram_alert(potential_signal, SIGNAL_GENERATION_TIMEFRAME); slots_available -= 1
            wait_time = max(get_interval_minutes(SIGNAL_GENERATION_TIMEFRAME) * 60 - 60, 60)
            logger.info(f"⏳ [Main] انتظار {wait_time:.1f} ثانية للدورة التالية...")
            time.sleep(wait_time)
        except (KeyboardInterrupt, SystemExit): break
        except Exception as main_err: logger.error(f"❌ [Main] خطأ غير متوقع: {main_err}", exc_info=True); time.sleep(120)

def run_flask():
    host, port = "0.0.0.0", int(os.environ.get('PORT', 10000))
    app = Flask(__name__); CORS(app)
    @app.route('/')
    def home(): return "Trading Bot is running"
    # Add other routes as needed
    logger.info(f"ℹ️ [Flask] بدء تطبيق Flask على {host}:{port}...")
    try: from waitress import serve; serve(app, host=host, port=port, threads=8)
    except ImportError: app.run(host=host, port=port)

if __name__ == "__main__":
    logger.info("🚀 بدء بوت إشارات تداول العملات الرقمية (LightGBM)...")
    try:
        init_db()
        Thread(target=run_ticker_socket_manager, daemon=True).start()
        time.sleep(5)
        Thread(target=track_signals, daemon=True).start()
        Thread(target=main_loop, daemon=True).start()
        run_flask()
    except (KeyboardInterrupt, SystemExit): logger.info("🛑 [Main] طلب إيقاف...")
    finally:
        if conn: conn.close()
        logger.info("👋 [Main] تم إيقاف البوت."); os._exit(0)
