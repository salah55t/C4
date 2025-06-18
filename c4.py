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
from binance.exceptions import BinanceAPIException
from flask import Flask, request, Response, jsonify, render_template_string
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler
from collections import deque

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
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
     logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
     exit(1)

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
MAX_OPEN_TRADES: int = 5
TRADE_AMOUNT_USDT: float = 10.0
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 7
MIN_VOLUME_24H_USDT: float = 10_000_000
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V4'
MODEL_PREDICTION_THRESHOLD = 0.70
USE_DYNAMIC_SL_TP = True
ATR_SL_MULTIPLIER = 2.0
ATR_TP_MULTIPLIER = 3.0
USE_TRAILING_STOP = False
USE_BTC_TREND_FILTER = True
BTC_SYMBOL = 'BTCUSDT'
BTC_TREND_TIMEFRAME = '4h'
BTC_TREND_EMA_PERIOD = 10
RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, BBANDS_PERIOD, ATR_PERIOD = 14, 12, 26, 9, 20, 14
BBANDS_STD_DEV: float = 2.0
USE_RSI_FILTER = True
RSI_LOWER_THRESHOLD = 40
RSI_UPPER_THRESHOLD = 69

# --- إعدادات فلتر MACD الجديدة ---
USE_MACD_FILTER: bool = True
MACD_SHORT_PERIOD: int = 12
MACD_LONG_PERIOD: int = 26
MACD_SIGNAL_PERIOD: int = 9
MACD_DIF_CROSSOVER_ONLY: bool = True # فلتر لتقاطع DIF صعوديا فوق DEA

# --- المتغيرات العامة وقفل العمليات ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
ml_models_cache: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
current_prices: Dict[str, float] = {}
prices_lock = Lock()
notifications_cache = deque(maxlen=50)
notifications_lock = Lock()

# --- !!! جديد: متغيرات حالة البوت !!! ---
bot_status_info = {"status": "INITIALIZING", "message": "البوت قيد التشغيل والتهيئة..."}
bot_status_lock = Lock()


# ---------------------- دوال قاعدة البيانات ----------------------
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
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        entry_price DOUBLE PRECISION NOT NULL,
                        target_price DOUBLE PRECISION NOT NULL,
                        stop_loss DOUBLE PRECISION NOT NULL,
                        status TEXT DEFAULT 'open',
                        closing_price DOUBLE PRECISION,
                        closed_at TIMESTAMP,
                        profit_percentage DOUBLE PRECISION,
                        strategy_name TEXT,
                        signal_details JSONB,
                        trailing_stop_price DOUBLE PRECISION
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL,
                        message TEXT NOT NULL,
                        is_read BOOLEAN DEFAULT FALSE
                    );
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
    log_methods = {
        'info': logger.info, 'warning': logger.warning,
        'error': logger.error, 'critical': logger.critical,
    }
    log_methods.get(level.lower(), logger.info)(message)

    if not check_db_connection() or not conn: return
    try:
        new_notification = {"timestamp": datetime.now().isoformat(), "type": notification_type, "message": message}
        with notifications_lock:
            notifications_cache.appendleft(new_notification)
        with conn.cursor() as cur:
            cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [Notify DB] فشل حفظ التنبيه في قاعدة البيانات: {e}")
        if conn: conn.rollback()

# ---------------------- دوال Binance والبيانات ----------------------
def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    logger.info(f"ℹ️ [التحقق] قراءة الرموز من '{filename}' والتحقق منها مع Binance...")
    if not client:
        logger.error("❌ [التحقق] كائن Binance client غير مهيأ. لا يمكن المتابعة.")
        return []
    try:
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols_from_file = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        formatted_symbols = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols_from_file}
        exchange_info = client.get_exchange_info()
        active_binance_symbols = {s['symbol'] for s in exchange_info['symbols'] if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'}
        validated_symbols = sorted(list(formatted_symbols.intersection(active_binance_symbols)))
        logger.info(f"✅ [التحقق] سيقوم البوت بمراقبة {len(validated_symbols)} عملة رقمية معتمدة.")
        return validated_symbols
    except Exception as e:
        logger.error(f"❌ [التحقق] حدث خطأ أثناء التحقق من الرموز: {e}", exc_info=True)
        return []

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        start_str = (datetime.utcnow() - timedelta(days=days + 1)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df[numeric_cols].dropna()
    except Exception as e:
        logger.error(f"❌ [البيانات] خطأ أثناء جلب البيانات التاريخية لـ {symbol}: {e}")
        return None

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df_calc['rsi'] = 100 - (100 / (1 + rs))

    # --- حسابات MACD: DIF و DEA (MACD Signal) و MACD Hist ---
    ema_fast = df_calc['close'].ewm(span=MACD_SHORT_PERIOD, adjust=False).mean() # DIF (MACD Line)
    ema_slow = df_calc['close'].ewm(span=MACD_LONG_PERIOD, adjust=False).mean()  # Slow EMA
    df_calc['macd_line'] = ema_fast - ema_slow # هذا هو DIF
    df_calc['macd_signal_line'] = df_calc['macd_line'].ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean() # هذا هو DEA
    df_calc['macd_hist'] = df_calc['macd_line'] - df_calc['macd_signal_line'] # هذا هو MACD Hist

    # تحديد تقاطع MACD
    # DIF يخترق صعودياً DEA
    df_calc['macd_crossover_up'] = (df_calc['macd_line'] > df_calc['macd_signal_line']) & \
                                   (df_calc['macd_line'].shift(1) <= df_calc['macd_signal_line'].shift(1))

    sma = df_calc['close'].rolling(window=BBANDS_PERIOD).mean()
    std = df_calc['close'].rolling(window=BBANDS_PERIOD).std()
    df_calc['bb_upper'] = sma + (std * BBANDS_STD_DEV)
    df_calc['bb_lower'] = sma - (std * BBANDS_STD_DEV)
    df_calc['bb_width'] = (df_calc['bb_upper'] - df_calc['bb_lower']) / sma
    df_calc['bb_pos'] = (df_calc['close'] - sma) / std.replace(0, np.nan)
    df_calc['day_of_week'] = df_calc.index.dayofweek
    df_calc['hour_of_day'] = df_calc.index.hour
    df_calc['candle_body_size'] = (df_calc['close'] - df_calc['open']).abs()
    df_calc['upper_wick'] = df_calc['high'] - df_calc[['open', 'close']].max(axis=1)
    df_calc['lower_wick'] = df_calc[['open', 'close']].min(axis=1) - df_calc['low']
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=30, min_periods=1).mean() + 1e-9)
    return df_calc.dropna()

def load_ml_model_bundle_from_db(symbol: str) -> Optional[Dict[str, Any]]:
    global ml_models_cache
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_name in ml_models_cache: return ml_models_cache[model_name]
    if not check_db_connection() or not conn: return None
    try:
        with conn.cursor() as db_cur:
            db_cur.execute("SELECT model_data FROM ml_models WHERE model_name = %s ORDER BY trained_at DESC LIMIT 1;", (model_name,))
            result = db_cur.fetchone()
            if result and result['model_data']:
                model_bundle = pickle.loads(result['model_data'])
                if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
                    ml_models_cache[model_name] = model_bundle
                    logger.info(f"✅ [نموذج تعلم الآلة] تم تحميل النموذج '{model_name}' بنجاح من قاعدة البيانات.")
                    return model_bundle
            logger.warning(f"⚠️ [نموذج تعلم الآلة] لم يتم العثور على النموذج '{model_name}' للعملة {symbol} في قاعدة البيانات.")
            return None
    except Exception as e:
        logger.error(f"❌ [نموذج تعلم الآلة] خطأ في تحميل حزمة النموذج للعملة {symbol}: {e}", exc_info=True)
        return None

# ---------------------- دوال WebSocket والاستراتيجية ----------------------
def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
    global open_signals_cache, current_prices
    try:
        data = msg.get('data', msg) if isinstance(msg, dict) else msg
        if not isinstance(data, list): data = [data]
        for item in data:
            symbol = item.get('s')
            if not symbol: continue
            price = float(item.get('c', 0))
            if price == 0: continue
            with prices_lock:
                current_prices[symbol] = price
            signal_to_process, status, closing_price = None, None, None
            with signal_cache_lock:
                if symbol in open_signals_cache:
                    signal = open_signals_cache[symbol]
                    target_price = signal.get('target_price')
                    current_stop_price = signal.get('trailing_stop_price') or signal.get('stop_loss')
                    if not all(isinstance(p, (int, float)) for p in [price, target_price, current_stop_price]): continue
                    if price >= target_price:
                        status, closing_price = 'target_hit', target_price
                        signal_to_process = signal
                    elif price <= current_stop_price:
                        status, closing_price = 'stop_loss_hit', current_stop_price
                        signal_to_process = signal
            if signal_to_process and status:
                logger.info(f"⚡ [المتتبع الفوري] تم تفعيل حدث '{status}' للعملة {symbol} عند سعر {price:.8f}")
                Thread(target=close_signal, args=(signal_to_process, status, closing_price, "auto")).start()
    except Exception as e:
        logger.error(f"❌ [متتبع WebSocket] خطأ في معالجة رسالة السعر الفورية: {e}", exc_info=True)

def run_websocket_manager() -> None:
    logger.info("ℹ️ [WebSocket] بدء مدير WebSocket...")
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    twm.start_ticker_socket(callback=handle_ticker_message)
    logger.info("✅ [WebSocket] تم الاتصال والاستماع بنجاح.")
    twm.join()

class TradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_bundle_from_db(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    def get_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        return calculate_features(df)

    def generate_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]):
            return None
        last_row = df_processed.iloc[-1]
        try:
            if USE_RSI_FILTER:
                current_rsi = last_row.get('rsi')
                if current_rsi is None or not (RSI_LOWER_THRESHOLD <= current_rsi <= RSI_UPPER_THRESHOLD):
                    logger.info(f"ℹ️ [{self.symbol}] فشل فلتر RSI. RSI الحالي: {current_rsi:.2f}")
                    return None
                logger.info(f"✅ [{self.symbol}] نجح فلتر RSI. RSI الحالي: {current_rsi:.2f} (ضمن النطاق {RSI_LOWER_THRESHOLD}-{RSI_UPPER_THRESHOLD})")

            # --- فلتر MACD الجديد ---
            if USE_MACD_FILTER:
                # تحقق من وجود بيانات كافية لـ MACD
                if len(df_processed) < max(MACD_SHORT_PERIOD, MACD_LONG_PERIOD, MACD_SIGNAL_PERIOD) + 1:
                    logger.info(f"ℹ️ [{self.symbol}] بيانات غير كافية لفلتر MACD.")
                    return None
                
                # تحقق من تقاطع DIF صعوديا فوق DEA في الشمعة الأخيرة
                # macd_line هي DIF، macd_signal_line هي DEA
                current_macd_line = last_row.get('macd_line')
                current_macd_signal_line = last_row.get('macd_signal_line')
                prev_macd_line = df_processed.iloc[-2].get('macd_line')
                prev_macd_signal_line = df_processed.iloc[-2].get('macd_signal_line')

                if current_macd_line is None or current_macd_signal_line is None or \
                   prev_macd_line is None or prev_macd_signal_line is None:
                    logger.info(f"ℹ️ [{self.symbol}] قيم MACD غير متوفرة لفلتر MACD.")
                    return None

                macd_crossover_up_triggered = (current_macd_line > current_macd_signal_line) and \
                                              (prev_macd_line <= prev_macd_signal_line)
                
                if MACD_DIF_CROSSOVER_ONLY and not macd_crossover_up_triggered:
                    logger.info(f"ℹ️ [{self.symbol}] فشل فلتر MACD. لم يحدث تقاطع DIF صعوديا فوق DEA.")
                    return None
                elif not MACD_DIF_CROSSOVER_ONLY and not (current_macd_line > current_macd_signal_line):
                    # إذا لم نكن نركز فقط على التقاطع، فنتأكد من أن MACD Line (DIF) فوق Signal Line (DEA)
                    logger.info(f"ℹ️ [{self.symbol}] فشل فلتر MACD. MACD Line ليست فوق Signal Line.")
                    return None
                
                logger.info(f"✅ [{self.symbol}] نجح فلتر MACD. DIF: {current_macd_line:.4f}, DEA: {current_macd_signal_line:.4f}")
                
            features_df = pd.DataFrame([last_row], columns=df_processed.columns)[self.feature_names]
            if features_df.isnull().values.any():
                logger.warning(f"⚠️ [{self.symbol}] يوجد قيم NaN في الميزات. سيتم التخطي.")
                return None
            
            features_scaled = self.scaler.transform(features_df)
            features_scaled_df = pd.DataFrame(features_scaled, columns=self.feature_names)
            prediction_proba = self.ml_model.predict_proba(features_scaled_df)[0][1]
            
            if prediction_proba < MODEL_PREDICTION_THRESHOLD:
                logger.info(f"ℹ️ [{self.symbol}] احتمالية ML منخفضة: {prediction_proba:.2%}. سيتم التخطي.")
                return None
            
            logger.info(f"✅ [العثور على إشارة] {self.symbol}: إشارة محتملة باحتمالية {prediction_proba:.2%}.")
            
            signal_details = {
                'ML_Probability': f"{prediction_proba:.2%}",
                'RSI_Value': f"{last_row.get('rsi'):.2f}",
                'MACD_Line': f"{last_row.get('macd_line'):.4f}",
                'MACD_Signal_Line': f"{last_row.get('macd_signal_line'):.4f}"
            }
            return {'symbol': self.symbol, 'strategy_name': BASE_ML_MODEL_NAME, 'signal_details': signal_details}
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
    rsi_value = signal_data['signal_details'].get('RSI_Value', 'N/A')
    ml_prob = signal_data['signal_details'].get('ML_Probability', 'N/A')
    macd_line = signal_data['signal_details'].get('MACD_Line', 'N/A')
    macd_signal_line = signal_data['signal_details'].get('MACD_Signal_Line', 'N/A')

    message = (f"💡 *إشارة تداول جديدة ({BASE_ML_MODEL_NAME})* 💡\n\n"
               f"🪙 *العملة:* `{safe_symbol}`\n"
               f"📈 *النوع:* شراء (LONG)\n\n"
               f"⬅️ *سعر الدخول:* `${entry:,.8g}`\n"
               f"🎯 *الهدف:* `${target:,.8g}` (ربح متوقع `{profit_pct:+.2f}%`)\n"
               f"🛑 *وقف الخسارة:* `${sl:,.8g}`\n\n"
               f"🔍 *مستوى الثقة (ML):* {ml_prob}\n"
               f"📊 *مؤشر القوة النسبية (RSI):* `{rsi_value}`\n"
               f"📊 *مؤشر MACD (DIF):* `{macd_line}`\n"
               f"📊 *مؤشر MACD (DEA):* `{macd_signal_line}`")

    reply_markup = {"inline_keyboard": [[{"text": "📊 فتح لوحة التحكم", "url": WEBHOOK_URL or '#'}]]}
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(CHAT_ID), 'text': message, 'parse_mode': 'Markdown', 'reply_markup': json.dumps(reply_markup)}
    try: requests.post(url, json=payload, timeout=20).raise_for_status()
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال تنبيه الإشارة الجديدة: {e}")
    log_and_notify('info', f"إشارة جديدة: {signal_data['symbol']} بسعر دخول ${entry:,.8g}", "NEW_SIGNAL")

def insert_signal_into_db(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not check_db_connection() or not conn: return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, trailing_stop_price) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;""",
                (signal['symbol'], float(signal['entry_price']), float(signal['target_price']), float(signal['stop_loss']), 
                 signal.get('strategy_name'), json.dumps(signal.get('signal_details', {})), float(signal['stop_loss']))
            )
            new_id = cur.fetchone()['id']
            signal['id'] = new_id
            signal['trailing_stop_price'] = signal['stop_loss']
        conn.commit()
        logger.info(f"✅ [قاعدة البيانات] تم إدراج الإشارة لـ {signal['symbol']} (ID: {new_id}).")
        return signal
    except Exception as e:
        logger.error(f"❌ [إدراج في قاعدة البيانات] خطأ في إدراج إشارة {signal['symbol']}: {e}", exc_info=True)
        if conn: conn.rollback()
        return None

def close_signal(signal: Dict, status: str, closing_price: float, closed_by: str):
    symbol = signal['symbol']
    with signal_cache_lock:
        if symbol not in open_signals_cache or open_signals_cache[symbol]['id'] != signal['id']: return
    if not check_db_connection() or not conn: return
    try:
        db_closing_price = float(closing_price)
        db_profit_pct = float(((db_closing_price / signal['entry_price']) - 1) * 100)
        with conn.cursor() as update_cur:
            update_cur.execute(
                "UPDATE signals SET status = %s, closing_price = %s, closed_at = NOW(), profit_percentage = %s WHERE id = %s;",
                (status, db_closing_price, db_profit_pct, signal['id'])
            )
        conn.commit()
        with signal_cache_lock: del open_signals_cache[symbol]
        status_map = {'target_hit': '✅ تحقق الهدف', 'stop_loss_hit': '🛑 ضرب وقف الخسارة', 'manual_close': '🖐️ أُغلقت يدوياً'}
        status_message = status_map.get(status, status.replace('_', ' ').title())
        safe_symbol = signal['symbol'].replace('_', '\\_')
        alert_msg_tg = f"*{status_message}*\n`{safe_symbol}` | *الربح:* `{db_profit_pct:+.2f}%`"
        send_telegram_message(CHAT_ID, alert_msg_tg)
        alert_msg_db = f"{status_message}: {signal['symbol']} | الربح: {db_profit_pct:+.2f}%"
        log_and_notify('info', alert_msg_db, 'CLOSE_SIGNAL')
    except Exception as e:
        logger.error(f"❌ [إغلاق قاعدة البيانات] خطأ فادح أثناء إغلاق الإشارة {signal['id']} لـ {signal['symbol']}: {e}", exc_info=True)
        if conn: conn.rollback()

def load_open_signals_to_cache():
    if not check_db_connection() or not conn: return
    logger.info("ℹ️ [تحميل الذاكرة المؤقتة] جاري تحميل الإشارات المفتوحة سابقاً إلى ذاكرة التتبع...")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status = 'open';")
            open_signals = cur.fetchall()
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in open_signals:
                    open_signals_cache[signal['symbol']] = dict(signal)
            logger.info(f"✅ [تحميل الذاكرة المؤقتة] تم تحميل {len(open_signals)} إشارة مفتوحة. يتم الآن تتبع {len(open_signals_cache)} إشارة.")
    except Exception as e:
        logger.error(f"❌ [تحميل الذاكرة المؤقتة] فشل تحميل الإشارات المفتوحة: {e}")

def load_notifications_to_cache():
    if not check_db_connection() or not conn: return
    logger.info("ℹ️ [تحميل الذاكرة المؤقتة] جاري تحميل آخر التنبيهات...")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 50;")
            recent_notifications = cur.fetchall()
            with notifications_lock:
                notifications_cache.clear()
                for n in reversed(recent_notifications): 
                    n['timestamp'] = n['timestamp'].isoformat()
                    notifications_cache.appendleft(dict(n))
            logger.info(f"✅ [تحميل الذاكرة المؤقتة] تم تحميل {len(notifications_cache)} تنبيه.")
    except Exception as e:
        logger.error(f"❌ [تحميل الذاكرة المؤقتة] فشل تحميل التنبيهات: {e}")

# ---------------------- حلقة العمل الرئيسية ----------------------
def get_btc_trend() -> Dict[str, Any]:
    if not client: 
        return {"status": "error", "message": "Binance client not initialized", "is_uptrend": False}
    try:
        klines = client.get_klines(symbol=BTC_SYMBOL, interval=BTC_TREND_TIMEFRAME, limit=BTC_TREND_EMA_PERIOD * 2)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df['close'] = pd.to_numeric(df['close'])
        ema = df['close'].ewm(span=BTC_TREND_EMA_PERIOD, adjust=False).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        if current_price > ema:
            status, message = "Uptrend", f"صاعد (السعر فوق متوسط {BTC_TREND_EMA_PERIOD} على إطار {BTC_TREND_TIMEFRAME})"
        else:
            # تم تصحيح الخطأ الإملائي هنا
            status, message = "Downtrend", f"هابط (السعر تحت متوسط {BTC_TREND_EMA_PERIOD} على إطار {BTC_TREND_TIMEFRAME})"
        return {"status": status, "message": message, "is_uptrend": (status == "Uptrend")}
    except Exception as e:
        logger.error(f"❌ [فلتر BTC] فشل تحديد اتجاه البيتكوين: {e}")
        return {"status": "Error", "message": str(e), "is_uptrend": False}

def set_bot_status(status: str, message: str):
    global bot_status_info
    with bot_status_lock:
        bot_status_info = {"status": status, "message": message}
    logger.info(f"[حالة البوت] {status}: {message}")

def main_loop():
    logger.info("[الحلقة الرئيسية] انتظار اكتمال التهيئة الأولية...")
    time.sleep(15) 
    if not validated_symbols_to_scan:
        message = "لا توجد رموز معتمدة للمسح. لن يستمر البوت في العمل."
        log_and_notify("critical", message, "SYSTEM")
        set_bot_status("ERROR", message)
        return
    
    log_and_notify("info", f"بدء حلقة المسح الرئيسية لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")
    
    while True:
        try:
            if USE_BTC_TREND_FILTER:
                trend_data = get_btc_trend()
                if not trend_data.get("is_uptrend"):
                    message = f"تم إيقاف البحث عن إشارات شراء بسبب الاتجاه الهابط للبيتكوين."
                    set_bot_status("PAUSED_BTC_TREND", message)
                    time.sleep(300)
                    continue

            with signal_cache_lock: open_count = len(open_signals_cache)
            if open_count >= MAX_OPEN_TRADES:
                message = f"تم الوصول للحد الأقصى للصفقات المفتوحة ({open_count}/{MAX_OPEN_TRADES})."
                set_bot_status("PAUSED_MAX_TRADES", message)
                time.sleep(60)
                continue

            slots_available = MAX_OPEN_TRADES - open_count
            set_bot_status("SCANNING", f"جاري البحث عن إشارات... المراكز المفتوحة: {open_count}/{MAX_OPEN_TRADES}")
            
            for symbol in validated_symbols_to_scan:
                if slots_available <= 0: break
                with signal_cache_lock:
                    if symbol in open_signals_cache: continue
                try:
                    df_hist = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_hist is None or df_hist.empty: continue
                    strategy = TradingStrategy(symbol)
                    df_features = strategy.get_features(df_hist)
                    if df_features is None or df_features.empty: continue
                    potential_signal = strategy.generate_signal(df_features)
                    if potential_signal:
                        with prices_lock: current_price = current_prices.get(symbol)
                        if not current_price:
                             logger.warning(f"⚠️ {symbol}: لا يمكن الحصول على السعر الحالي. سيتم التخطي.")
                             continue
                        potential_signal['entry_price'] = current_price
                        if USE_DYNAMIC_SL_TP:
                            atr_value = df_features['atr'].iloc[-1]
                            potential_signal['stop_loss'] = current_price - (atr_value * ATR_SL_MULTIPLIER)
                            potential_signal['target_price'] = current_price + (atr_value * ATR_TP_MULTIPLIER)
                        else:
                            potential_signal['target_price'] = current_price * 1.015
                            potential_signal['stop_loss'] = current_price * 0.99
                        potential_signal['trailing_stop_price'] = potential_signal['stop_loss']
                        saved_signal = insert_signal_into_db(potential_signal)
                        if saved_signal:
                            with signal_cache_lock: open_signals_cache[saved_signal['symbol']] = saved_signal
                            send_new_signal_alert(saved_signal)
                            slots_available -= 1
                            open_count += 1
                            set_bot_status("SCANNING", f"إشارة جديدة! المراكز المفتوحة: {open_count}/{MAX_OPEN_TRADES}")
                except Exception as e:
                    logger.error(f"❌ [خطأ في المعالجة] حدث خطأ أثناء معالجة العملة {symbol}: {e}", exc_info=True)
            
            time.sleep(60)
        except (KeyboardInterrupt, SystemExit): break
        except Exception as main_err:
            message = f"خطأ غير متوقع في الحلقة الرئيسية: {main_err}"
            log_and_notify("error", message, "SYSTEM")
            set_bot_status("ERROR", message)
            time.sleep(120)

# ---------------------- واجهة برمجة تطبيقات Flask للوحة التحكم ----------------------
app = Flask(__name__)
CORS(app)

def get_fear_and_greed_index() -> Dict[str, Any]:
    classification_translation = {"Extreme Fear": "خوف شديد", "Fear": "خوف", "Neutral": "محايد", "Greed": "طمع", "Extreme Greed": "طمع شديد", "Error": "خطأ"}
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        response.raise_for_status()
        data = response.json()['data'][0]
        original_classification = data['value_classification']
        return {"value": int(data['value']), "classification": classification_translation.get(original_classification, original_classification)}
    except Exception as e:
        logger.error(f"❌ [مؤشر الخوف والطمع] فشل الاتصال بالـ API: {e}")
        return {"value": -1, "classification": classification_translation["Error"]}

@app.route('/')
def home():
    try:
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, 'index.html')
        with open(file_path, 'r', encoding='utf-8') as f:
            return render_template_string(f.read())
    except FileNotFoundError:
        return "<h1>ملف لوحة التحكم (index.html) غير موجود.</h1>", 404

# --- !!! جديد: API لحالة البوت !!! ---
@app.route('/api/bot_status')
def get_bot_status():
    with bot_status_lock:
        return jsonify(bot_status_info)

@app.route('/api/market_status')
def get_market_status():
    btc_trend = get_btc_trend()
    fear_and_greed = get_fear_and_greed_index()
    return jsonify({"btc_trend": btc_trend, "fear_and_greed": fear_and_greed})

@app.route('/api/stats')
def get_stats():
    if not check_db_connection() or not conn:
        return jsonify({"error": "فشل الاتصال بقاعدة البيانات"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT status, profit_percentage FROM signals WHERE status != 'open';")
            closed_signals = cur.fetchall()
        wins = sum(1 for s in closed_signals if s.get('profit_percentage', 0) > 0)
        losses = sum(1 for s in closed_signals if s.get('profit_percentage', 0) <= 0)
        total_closed = len(closed_signals)
        win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
        loss_rate = (losses / total_closed * 100) if total_closed > 0 else 0
        total_profit_usdt = sum(s['profit_percentage'] / 100 * TRADE_AMOUNT_USDT for s in closed_signals if s.get('profit_percentage') is not None)
        return jsonify({"win_rate": win_rate, "loss_rate": loss_rate, "wins": wins, "losses": losses, "total_profit_usdt": total_profit_usdt, "total_closed_trades": total_closed})
    except Exception as e:
        logger.error(f"❌ [API إحصائيات] خطأ: {e}")
        return jsonify({"error": "تعذر جلب الإحصائيات"}), 500

@app.route('/api/signals')
def get_signals():
    if not check_db_connection() or not conn:
        return jsonify({"error": "فشل الاتصال بقاعدة البيانات"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals ORDER BY CASE WHEN status = 'open' THEN 0 ELSE 1 END, id DESC;")
            all_signals = cur.fetchall()
        for s in all_signals:
            if s.get('closed_at'): s['closed_at'] = s['closed_at'].isoformat()
            if s['status'] == 'open':
                with prices_lock: s['current_price'] = current_prices.get(s['symbol'])
        return jsonify(all_signals)
    except Exception as e:
        logger.error(f"❌ [API إشارات] خطأ: {e}")
        return jsonify({"error": "تعذر جلب الإشارات"}), 500

@app.route('/api/close/<int:signal_id>', methods=['POST'])
def manual_close_signal(signal_id):
    logger.info(f"ℹ️ [API إغلاق] تم استلام طلب إغلاق يدوي للإشارة ID: {signal_id}")
    signal_to_close = None
    with signal_cache_lock:
        for signal_data in open_signals_cache.values():
            if signal_data['id'] == signal_id:
                signal_to_close = signal_data.copy()
                break
    if not signal_to_close:
        return jsonify({"error": "لم يتم العثور على الإشارة في ذاكرة الصفقات المفتوحة أو أنها أُغلقت بالفعل."}), 404
    symbol_to_close = signal_to_close['symbol']
    with prices_lock: closing_price = current_prices.get(symbol_to_close)
    if not closing_price:
        return jsonify({"error": f"تعذر الحصول على السعر الحالي للعملة {symbol_to_close} لإتمام الإغلاق."}), 500
    Thread(target=close_signal, args=(signal_to_close, 'manual_close', closing_price, "manual")).start()
    return jsonify({"message": f"جاري إغلاق الإشارة {signal_id} للعملة {symbol_to_close} عند سعر {closing_price}."})

@app.route('/api/notifications')
def get_notifications():
    with notifications_lock:
        return jsonify(list(notifications_cache))

def run_flask():
    host, port = "0.0.0.0", int(os.environ.get('PORT', 10000))
    log_and_notify("info", f"بدء تشغيل لوحة التحكم على {host}:{port}", "SYSTEM")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        logger.warning("⚠️ [Flask] مكتبة 'waitress' غير موجودة, سيتم استخدام خادم التطوير (غير مناسب للإنتاج).")
        app.run(host=host, port=port)

# ---------------------- نقطة انطلاق البرنامج ----------------------
def initialize_bot_services():
    global client, validated_symbols_to_scan
    set_bot_status("INITIALIZING", "بدء تهيئة خدمات البوت...")
    try:
        client = Client(API_KEY, API_SECRET)
        logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
        init_db()
        load_open_signals_to_cache()
        load_notifications_to_cache()
        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan:
            message = "لا توجد رموز معتمدة للمسح. الحلقات لن تبدأ."
            logger.critical(f"❌ {message}")
            set_bot_status("ERROR", message)
            return
        Thread(target=run_websocket_manager, daemon=True).start()
        Thread(target=main_loop, daemon=True).start()
        logger.info("✅ [خدمات البوت] تم بدء جميع خدمات الخلفية بنجاح.")
    except Exception as e:
        message = f"حدث خطأ حاسم أثناء التهيئة: {e}"
        log_and_notify("critical", message, "SYSTEM")
        set_bot_status("ERROR", message)
        pass

if __name__ == "__main__":
    logger.info("🚀 بدء تشغيل تطبيق بوت التداول...")
    initialization_thread = Thread(target=initialize_bot_services, daemon=True)
    initialization_thread.start()
    run_flask()
    logger.info("👋 [إيقاف] تم إيقاف تشغيل البوت.")
    os._exit(0)
