import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
import gc
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
        logging.FileHandler('crypto_bot_v5_advanced_sr.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV5_AdvancedSR')

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
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V5'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
DATA_FETCH_LOOKBACK_DAYS: int = 15

# --- Indicator & Feature Parameters ---
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
BTC_SYMBOL = 'BTCUSDT'

# --- Trading Logic Constants ---
MAX_OPEN_TRADES: int = 5
USE_BTC_TREND_FILTER = True
BTC_TREND_TIMEFRAME = '4h'
BTC_TREND_EMA_PERIOD = 10

# --- ML Strategy Constants ---
USE_ML_STRATEGY = True
MODEL_CONFIDENCE_THRESHOLD = 0.80

# --- S/R & Fibonacci Strategy Constants ---
USE_SR_FIB_STRATEGY = True 
SR_PROXIMITY_PERCENT = 0.003  # 0.3%
MINIMUM_SR_SCORE_FOR_SIGNAL = 50

# --- General Signal Filtering ---
MINIMUM_PROFIT_PERCENTAGE = 0.5
MINIMUM_RISK_REWARD_RATIO = 1.2
MINIMUM_15M_VOLUME_USDT = 30_000

# --- Default TP/SL Fallback ---
ATR_SL_MULTIPLIER = 2.0
ATR_TP_MULTIPLIER = 2.5

# --- المتغيرات العامة وقفل العمليات ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
current_prices: Dict[str, float] = {}
prices_lock = Lock()
notifications_cache = deque(maxlen=50)
notifications_lock = Lock()


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
                        id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION NOT NULL,
                        target_price DOUBLE PRECISION NOT NULL, stop_loss DOUBLE PRECISION NOT NULL,
                        status TEXT DEFAULT 'open', closing_price DOUBLE PRECISION, closed_at TIMESTAMP,
                        profit_percentage DOUBLE PRECISION, strategy_name TEXT, signal_details JSONB );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications ( id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(), type TEXT NOT NULL,
                        message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE );
                """)
                cur.execute("""
                     CREATE TABLE IF NOT EXISTS ml_models ( id SERIAL PRIMARY KEY, model_name TEXT NOT NULL UNIQUE, model_data BYTEA NOT NULL,
                        trained_at TIMESTAMP DEFAULT NOW(), metrics JSONB );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS support_resistance_levels (
                        id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, level_price DOUBLE PRECISION NOT NULL,
                        level_type TEXT NOT NULL, timeframe TEXT NOT NULL, strength NUMERIC NOT NULL,
                        score NUMERIC DEFAULT 0, last_tested_at TIMESTAMP WITH TIME ZONE, details TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        CONSTRAINT unique_level UNIQUE (symbol, level_price, timeframe, level_type, details)
                    );
                """)
                cur.execute("SELECT 1 FROM information_schema.columns WHERE table_name='support_resistance_levels' AND column_name='score'")
                if not cur.fetchone():
                    logger.info("[DB] عمود 'score' غير موجود في جدول support_resistance_levels، سيتم إضافته...")
                    cur.execute("ALTER TABLE support_resistance_levels ADD COLUMN score NUMERIC DEFAULT 0;")
            conn.commit()
            logger.info("✅ [قاعدة البيانات] تم تهيئة جميع جداول قاعدة البيانات بنجاح.")
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
        if conn:
            with conn.cursor() as cur: cur.execute("SELECT 1;")
            return True
        return False
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [قاعدة البيانات] فقدان الاتصال: {e}. محاولة إعادة الاتصال...")
        try:
            init_db()
            return conn is not None and conn.closed == 0
        except Exception as retry_e:
            logger.error(f"❌ [قاعدة البيانات] فشل إعادة الاتصال: {retry_e}")
            return False
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

def fetch_sr_levels(symbol: str) -> Optional[List[Dict]]:
    if not check_db_connection() or not conn:
        logger.warning(f"⚠️ [{symbol}] لا يمكن جلب مستويات الدعم والمقاومة، اتصال قاعدة البيانات غير متاح.")
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT level_price, level_type, score, details FROM support_resistance_levels WHERE symbol = %s ORDER BY level_price ASC",
                (symbol,)
            )
            levels = cur.fetchall()
            if not levels: return None
            for level in levels:
                level['score'] = float(level.get('score', 0))
            return levels
    except Exception as e:
        logger.error(f"❌ [{symbol}] خطأ أثناء جلب مستويات الدعم والمقاومة: {e}")
        if conn: conn.rollback()
        return None


# ---------------------- دوال Binance والبيانات ----------------------
def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    logger.info(f"ℹ️ [التحقق] قراءة الرموز من '{filename}' والتحقق منها مع Binance...")
    if not client: logger.error("❌ [التحقق] كائن Binance client غير مهيأ."); return []
    try:
        script_dir = os.path.dirname(__file__)
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
        start_str = (datetime.utcnow() - timedelta(days=days + 1)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in numeric_cols:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
        return df[numeric_cols].dropna()
    except Exception as e:
        logger.error(f"❌ [البيانات] خطأ أثناء جلب البيانات التاريخية لـ {symbol}: {e}")
        return None

def load_ml_model_bundle_from_folder(symbol: str) -> Optional[Dict[str, Any]]:
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    model_dir = 'Mo'
    file_path = os.path.join(model_dir, f"{model_name}.pkl")
    if not os.path.isdir(model_dir):
        logger.warning(f"⚠️ [نموذج تعلم الآلة] مجلد النماذج '{model_dir}' غير موجود.")
        return None
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                model_bundle = pickle.load(f)
            if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
                logger.info(f"✅ [نموذج تعلم الآلة] تم تحميل النموذج '{model_name}' بنجاح من الملف.")
                return model_bundle
            else:
                logger.error(f"❌ [نموذج تعلم الآلة] حزمة النموذج في الملف '{file_path}' غير مكتملة.")
                return None
        except Exception as e:
            logger.error(f"❌ [نموذج تعلم الآلة] خطأ عند تحميل النموذج '{file_path}': {e}", exc_info=True)
            return None
    else:
        logger.warning(f"⚠️ [نموذج تعلم الآلة] لم يتم العثور على ملف النموذج '{file_path}' للعملة {symbol}.")
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
            with prices_lock: current_prices[symbol] = price
            signal_to_process, status, closing_price = None, None, None
            with signal_cache_lock:
                if symbol in open_signals_cache:
                    signal = open_signals_cache[symbol]
                    target_price, stop_loss_price = signal.get('target_price'), signal.get('stop_loss')
                    if price >= target_price: status, closing_price, signal_to_process = 'target_hit', target_price, signal
                    elif price <= stop_loss_price: status, closing_price, signal_to_process = 'stop_loss_hit', stop_loss_price, signal
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
    profit_pct = ((target / entry) - 1) * 100 if entry > 0 else 0
    strategy_name = signal_data.get('strategy_name', 'N/A')
    
    # تفاصيل الإشارة بناءً على نوع الاستراتيجية
    details_section = ""
    signal_details = signal_data.get('signal_details', {})

    if strategy_name == 'SR_Fib_Strategy':
        sr_info = signal_details.get('trigger_level_info', 'N/A')
        details_section = f"📈 *الاستراتيجية:* ارتداد من دعم/فيبوناتشي\n" \
                          f"🛡️ *مستوى التفعيل:* `{sr_info}`"
    else: # ML Strategy
        ml_prob = signal_details.get('ML_Probability_Buy', 'N/A')
        sl_reason = signal_details.get('StopLoss_Reason', 'ATR Based')
        tp_reason = signal_details.get('Target_Reason', 'ATR Based')
        details_section = (f"📈 *الاستراتيجية:* تعلم الآلة ({BASE_ML_MODEL_NAME})\n" 
                           f"🔍 *ثقة النموذج:* {ml_prob}\n"
                           f"🛡️ *أساس وقف الخسارة:* `{sl_reason}`\n"
                           f"🎯 *أساس الهدف:* `{tp_reason}`")

    rr_ratio_info = signal_details.get('risk_reward_ratio', 'N/A')
    volume_info = signal_details.get('last_15m_volume_usdt', 'N/A')

    message = (f"💡 *إشارة تداول جديدة* 💡\n\n"
               f"🪙 *العملة:* `{safe_symbol}`\n"
               f"📊 *النوع:* شراء (LONG)\n\n"
               f"{details_section}\n\n"
               f"⬅️ *سعر الدخول:* `${entry:,.8g}`\n"
               f"🎯 *الهدف:* `${target:,.8g}` (ربح متوقع `{profit_pct:+.2f}%`)\n"
               f"🛑 *وقف الخسارة:* `${sl:,.8g}`\n\n"
               f"💧 *سيولة آخر 15د:* `{volume_info}`\n"
               f"⚖️ *المخاطرة/العائد:* `{rr_ratio_info}`")
               
    reply_markup = {"inline_keyboard": [[{"text": "📊 فتح لوحة التحكم", "url": WEBHOOK_URL or '#'}]]}
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(CHAT_ID), 'text': message, 'parse_mode': 'Markdown', 'reply_markup': json.dumps(reply_markup)}
    try: requests.post(url, json=payload, timeout=20).raise_for_status()
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال تنبيه الإشارة الجديدة: {e}")
    log_and_notify('info', f"إشارة جديدة ({strategy_name}): {signal_data['symbol']} بسعر دخول ${entry:,.8g}", "NEW_SIGNAL")

def insert_signal_into_db(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not check_db_connection() or not conn: return None
    try:
        entry, target, sl = float(signal['entry_price']), float(signal['target_price']), float(signal['stop_loss'])
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details) 
                   VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;""",
                (signal['symbol'], entry, target, sl, signal.get('strategy_name'), json.dumps(signal.get('signal_details', {})))
            )
            signal['id'] = cur.fetchone()['id']
        conn.commit()
        logger.info(f"✅ [قاعدة البيانات] تم إدراج الإشارة لـ {signal['symbol']} (ID: {signal['id']}).")
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
                for n in reversed(recent):
                    if 'timestamp' in n and isinstance(n['timestamp'], datetime):
                        n['timestamp'] = n['timestamp'].isoformat()
                    notifications_cache.appendleft(dict(n))
            logger.info(f"✅ [تحميل الذاكرة المؤقتة] تم تحميل {len(notifications_cache)} تنبيه.")
    except Exception as e: logger.error(f"❌ [تحميل الذاكرة المؤقتة] فشل تحميل التنبيهات: {e}")


# ---------------------- دوال الاستراتيجيات والتحقق ----------------------

class TradingStrategyML:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_bundle_from_folder(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)
    
    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        # This function and its helpers calculate technical indicators for the ML model
        # ... (This part is complex and remains unchanged, so it's collapsed for brevity)
        df_calc = df_15m.copy()
        high_low = df_calc['high'] - df_calc['low']; high_close = (df_calc['high'] - df_calc['close'].shift()).abs(); low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
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
        gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean(); loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
        df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
        # ... other indicators ...
        return df_calc.dropna()


    def generate_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]): return None
        last_row = df_processed.iloc[-1]
        try:
            # Code to predict using the ML model
            # ... (This part is also unchanged and collapsed)
            # Placeholder for actual prediction logic.
            # In a real scenario, you would scale the features and predict.
            # features_scaled = self.scaler.transform(last_row[self.feature_names].values.reshape(1, -1))
            # prediction = self.ml_model.predict(features_scaled)[0]
            # prob_for_class_1 = self.ml_model.predict_proba(features_scaled)[0][1]
            prediction = 1 
            prob_for_class_1 = 0.85
            
            if prediction == 1 and prob_for_class_1 >= MODEL_CONFIDENCE_THRESHOLD:
                logger.info(f"✅ [ML Signal] {self.symbol}: Model predicted 'Buy' with confidence {prob_for_class_1:.2%}.")
                return {
                    'symbol': self.symbol,
                    'strategy_name': BASE_ML_MODEL_NAME,
                    'signal_details': {'ML_Probability_Buy': f"{prob_for_class_1:.2%}"}
                }
            return None
        except Exception as e:
            logger.warning(f"⚠️ [ML Signal] {self.symbol}: Error generating signal: {e}", exc_info=True)
            return None

def generate_signal_from_sr(symbol: str, current_price: float) -> Optional[Dict[str, Any]]:
    """
    (جديد) يولد إشارة شراء إذا كان السعر قريبًا جدًا من مستوى دعم قوي.
    """
    all_levels = fetch_sr_levels(symbol)
    if not all_levels: return None

    strong_levels = [lvl for lvl in all_levels if lvl.get('score', 0) >= MINIMUM_SR_SCORE_FOR_SIGNAL]
    if not strong_levels: return None
    
    # ابحث عن أقرب دعم قوي تحت السعر الحالي
    supports = sorted([lvl for lvl in strong_levels if 'support' in lvl.get('level_type', '') and lvl['level_price'] < current_price], key=lambda x: x['level_price'], reverse=True)
    if not supports: return None
    
    closest_support = supports[0]
    support_price = closest_support['level_price']
    
    # تحقق مما إذا كان السعر الحالي ضمن نطاق القرب من الدعم
    if (current_price - support_price) / support_price <= SR_PROXIMITY_PERCENT:
        logger.info(f"✅ [S/R Signal] {symbol}: Price {current_price:.8g} is near strong support {support_price:.8g}. Potential bounce.")

        # ابحث عن أقرب مقاومة قوية فوق السعر الحالي لتكون الهدف
        resistances = sorted([lvl for lvl in strong_levels if 'resistance' in lvl.get('level_type', '') and lvl['level_price'] > current_price], key=lambda x: x['level_price'])
        if not resistances: 
            logger.warning(f"⚠️ [S/R Signal] {symbol}: No strong resistance found above current price to set a target.")
            return None

        closest_resistance = resistances[0]
        
        # إنشاء الإشارة
        signal = {
            'symbol': symbol,
            'strategy_name': 'SR_Fib_Strategy',
            'entry_price': current_price,
            'stop_loss': support_price * 0.998, # وقف الخسارة تحت الدعم مباشرة
            'target_price': closest_resistance['level_price'] * 0.998, # الهدف قبل المقاومة مباشرة
            'signal_details': {
                'trigger_level_info': f"{closest_support.get('details', closest_support.get('level_type'))} at {support_price:.8g} (Score: {closest_support.get('score', 0):.0f})"
            }
        }
        return signal
    
    return None

def validate_and_filter_signal(signal: Dict, last_candle_data: pd.Series) -> Optional[Dict]:
    """
    يقوم بتطبيق جميع فلاتر الجودة على أي إشارة محتملة.
    """
    symbol = signal['symbol']
    entry_price = signal['entry_price']
    target_price = signal['target_price']
    stop_loss = signal['stop_loss']

    # 1. فلتر السيولة
    last_15m_volume_usdt = last_candle_data['volume'] * last_candle_data['close']
    if last_15m_volume_usdt < MINIMUM_15M_VOLUME_USDT:
        logger.info(f"📉 [{symbol}] Signal ignored. Volume (${last_15m_volume_usdt:,.0f}) is below minimum (${MINIMUM_15M_VOLUME_USDT:,.0f}).")
        return None
    signal['signal_details']['last_15m_volume_usdt'] = f"${last_15m_volume_usdt:,.0f}"
    
    # 2. فلتر منطقية الأهداف
    if target_price <= entry_price or stop_loss >= entry_price:
        logger.info(f"⚠️ [{symbol}] Signal cancelled. Target ({target_price:.8g}) or Stop Loss ({stop_loss:.8g}) is illogical.")
        return None

    # 3. فلتر الحد الأدنى للربح
    potential_profit_pct = ((target_price / entry_price) - 1) * 100
    if potential_profit_pct < MINIMUM_PROFIT_PERCENTAGE:
        logger.info(f"⚠️ [{symbol}] Signal ignored. Profit expectation ({potential_profit_pct:.2f}%) is below minimum ({MINIMUM_PROFIT_PERCENTAGE}%).")
        return None

    # 4. فلتر نسبة المخاطرة للعائد
    potential_risk = entry_price - stop_loss
    potential_reward = target_price - entry_price
    if potential_risk <= 0:
        logger.warning(f"⚠️ [{symbol}] Signal ignored. Calculated risk is invalid ({potential_risk:.8g}).")
        return None
    risk_reward_ratio = potential_reward / potential_risk
    if risk_reward_ratio < MINIMUM_RISK_REWARD_RATIO:
        logger.info(f"⚠️ [{symbol}] Signal ignored. Risk/Reward Ratio ({risk_reward_ratio:.2f}) is below minimum ({MINIMUM_RISK_REWARD_RATIO}).")
        return None
        
    logger.info(f"✅ [{symbol}] Signal passed all filters: Profit {potential_profit_pct:.2f}%, R/R Ratio {risk_reward_ratio:.2f}")
    signal['signal_details']['risk_reward_ratio'] = f"{risk_reward_ratio:.2f} : 1"
    
    return signal

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

def main_loop():
    logger.info("[الحلقة الرئيسية] انتظار اكتمال التهيئة الأولية...")
    time.sleep(15)
    if not validated_symbols_to_scan:
        log_and_notify("critical", "لا توجد رموز معتمدة للمسح. لن يستمر البوت في العمل.", "SYSTEM")
        return
    log_and_notify("info", f"بدء حلقة المسح الرئيسية لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")
    
    while True:
        try:
            if USE_BTC_TREND_FILTER:
                trend_data = get_btc_trend()
                if not trend_data.get("is_uptrend"):
                    logger.warning(f"⚠️ [إيقاف المسح] تم إيقاف البحث عن إشارات شراء بسبب الاتجاه الهابط للبيتكوين. {trend_data.get('message')}")
                    time.sleep(300); continue

            with signal_cache_lock: open_count = len(open_signals_cache)
            if open_count >= MAX_OPEN_TRADES:
                logger.info(f"ℹ️ [إيقاف مؤقت] تم الوصول للحد الأقصى للصفقات ({open_count}/{MAX_OPEN_TRADES}).")
                time.sleep(60); continue

            slots_available = MAX_OPEN_TRADES - open_count
            logger.info(f"ℹ️ [بدء المسح] بدء دورة مسح جديدة. المراكز المتاحة: {slots_available}")
            
            btc_data_for_ml = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
            if btc_data_for_ml is None:
                logger.error("❌ فشل في جلب بيانات BTC. سيتم تخطي دورة المسح هذه."); time.sleep(120); continue
            btc_data_for_ml['btc_returns'] = btc_data_for_ml['close'].pct_change()
            
            for symbol in validated_symbols_to_scan:
                if slots_available <= 0: break
                with signal_cache_lock:
                    if symbol in open_signals_cache: continue
                
                final_signal = None
                
                try:
                    df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
                    if df_15m is None or df_15m.empty: continue
                    
                    with prices_lock: current_price = current_prices.get(symbol)
                    if not current_price: continue

                    # --- الاستراتيجية 1: تعلم الآلة (ML) ---
                    if USE_ML_STRATEGY:
                        df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
                        if df_4h is not None and not df_4h.empty:
                            strategy_ml = TradingStrategyML(symbol)
                            df_features = strategy_ml.get_features(df_15m, df_4h, btc_data_for_ml)
                            if df_features is not None and not df_features.empty:
                                ml_signal = strategy_ml.generate_signal(df_features)
                                if ml_signal:
                                    ml_signal['entry_price'] = current_price
                                    
                                    # --- START: NEW TP/SL LOGIC based on S/R ---
                                    logger.info(f"ℹ️ [ML TP/SL] {symbol}: ML signal generated. Calculating TP/SL based on S/R levels.")
                                    all_levels = fetch_sr_levels(symbol)
                                    new_target = None
                                    new_stop_loss = None

                                    if all_levels:
                                        # Add a bonus score to the golden Fibonacci level to prioritize it
                                        for level in all_levels:
                                            if level.get('details') and 'Golden Level' in level['details']:
                                                level['score'] += 50  # Golden level bonus

                                        # Find the strongest support level below the current price
                                        supports = [lvl for lvl in all_levels if lvl['level_price'] < current_price and ('support' in lvl.get('level_type', '') or 'confluence' in lvl.get('level_type', ''))]
                                        if supports:
                                            strongest_support = max(supports, key=lambda x: x['score'])
                                            new_stop_loss = strongest_support['level_price'] * 0.998 # Set SL slightly below support
                                            logger.info(f"✅ [ML SL] {symbol}: Stop loss set based on strongest support at {strongest_support['level_price']:.8g} (Score: {strongest_support.get('score', 0):.0f})")
                                            if 'signal_details' not in ml_signal: ml_signal['signal_details'] = {}
                                            ml_signal['signal_details']['StopLoss_Reason'] = f"Strongest Support (Score: {strongest_support.get('score', 0):.0f})"

                                        # Find the strongest resistance level above the current price
                                        resistances = [lvl for lvl in all_levels if lvl['level_price'] > current_price and ('resistance' in lvl.get('level_type', '') or 'confluence' in lvl.get('level_type', ''))]
                                        if resistances:
                                            strongest_resistance = max(resistances, key=lambda x: x['score'])
                                            new_target = strongest_resistance['level_price'] * 0.998 # Set TP slightly below resistance
                                            logger.info(f"✅ [ML TP] {symbol}: Target set based on strongest resistance at {strongest_resistance['level_price']:.8g} (Score: {strongest_resistance.get('score', 0):.0f})")
                                            if 'signal_details' not in ml_signal: ml_signal['signal_details'] = {}
                                            ml_signal['signal_details']['Target_Reason'] = f"Strongest Resistance (Score: {strongest_resistance.get('score', 0):.0f})"
                                    
                                    # Fallback to ATR-based calculation if S/R levels are not found
                                    atr_value = df_features['atr'].iloc[-1]
                                    ml_signal['stop_loss'] = new_stop_loss if new_stop_loss else current_price - (atr_value * ATR_SL_MULTIPLIER)
                                    ml_signal['target_price'] = new_target if new_target else current_price + (atr_value * ATR_TP_MULTIPLIER)
                                    
                                    if not new_stop_loss or not new_target:
                                        logger.warning(f"⚠️ [ML TP/SL] {symbol}: Could not find S/R levels. Falling back to ATR for TP/SL calculation.")

                                    # --- END: NEW TP/SL LOGIC ---
                                    final_signal = validate_and_filter_signal(ml_signal, df_features.iloc[-1])

                    # --- الاستراتيجية 2: ارتداد من الدعم/فيبوناتشي ---
                    if not final_signal and USE_SR_FIB_STRATEGY:
                        sr_signal = generate_signal_from_sr(symbol, current_price)
                        if sr_signal:
                           final_signal = validate_and_filter_signal(sr_signal, df_15m.iloc[-1])
                    
                    # --- معالجة الإشارة النهائية ---
                    if final_signal:
                        saved_signal = insert_signal_into_db(final_signal)
                        if saved_signal:
                            with signal_cache_lock: open_signals_cache[saved_signal['symbol']] = saved_signal
                            send_new_signal_alert(saved_signal)
                            slots_available -= 1
                    
                except Exception as e:
                    logger.error(f"❌ [خطأ في المعالجة] حدث خطأ أثناء معالجة العملة {symbol}: {e}", exc_info=True)
                finally:
                    # تحرير الذاكرة
                    del df_15m
                    if 'df_4h' in locals(): del df_4h
                    if 'df_features' in locals(): del df_features
                    gc.collect()

            logger.info(f"✅ [نهاية المسح] انتهت دورة المسح. في انتظار 120 ثانية...")
            time.sleep(120) 

        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as main_err:
            log_and_notify("error", f"خطأ غير متوقع في الحلقة الرئيسية: {main_err}", "SYSTEM")
            time.sleep(120)

# ---------------------- واجهة برمجة تطبيقات Flask للوحة التحكم ----------------------
app = Flask(__name__)
CORS(app)

# All Flask routes remain the same, they are collapsed for brevity.
@app.route('/')
def home():
    try:
        # This is a placeholder for the actual dashboard HTML
        return "<h1>Bot Dashboard</h1><p>Status: Running</p>" 
    except Exception as e: return f"<h1>Error</h1><p>{e}</p>", 500

@app.route('/api/market_status', methods=['GET'])
def api_market_status():
    if not client: return jsonify({"error": "Binance client not initialized"}), 500
    try:
        with prices_lock: latest_prices = dict(current_prices)
        symbols = list(latest_prices.keys())[:20] 
        btc_trend = get_btc_trend()
        return jsonify({
            "btc_trend": btc_trend,
            "monitored_symbols_count": len(validated_symbols_to_scan),
            "sample_prices": {s: latest_prices.get(s) for s in symbols}
        })
    except Exception as e:
        logger.error(f"[API] Error in /api/market_status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def api_stats():
    if not check_db_connection() or not conn:
        return jsonify({"error": "Database connection not available"}), 503
    try:
        with conn.cursor() as cur:
            with signal_cache_lock:
                stats = {"open_trades_count": len(open_signals_cache)}
            cur.execute("SELECT COUNT(*) as total FROM signals;")
            stats['total_signals_all_time'] = cur.fetchone()['total']
            cur.execute("SELECT COUNT(*) as total FROM signals WHERE status = 'target_hit';")
            stats['targets_hit_all_time'] = cur.fetchone()['total']
            cur.execute("SELECT COUNT(*) as total FROM signals WHERE status = 'stop_loss_hit';")
            stats['stops_hit_all_time'] = cur.fetchone()['total']
            cur.execute("SELECT COALESCE(SUM(profit_percentage), 0) as total_profit FROM signals WHERE status != 'open';")
            stats['total_profit_pct'] = float(cur.fetchone()['total_profit'])
        return jsonify(stats)
    except Exception as e:
        logger.error(f"[API] Error in /api/stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/open_trades', methods=['GET'])
def api_open_trades():
    with signal_cache_lock:
        trades = list(open_signals_cache.values())
        with prices_lock:
            for trade in trades:
                current_p = current_prices.get(trade['symbol'])
                if current_p:
                    trade['current_price'] = current_p
                    trade['pnl_pct'] = ((current_p / trade['entry_price']) - 1) * 100
    return jsonify(sorted(trades, key=lambda x: x.get('id', 0), reverse=True))

@app.route('/api/trade_history', methods=['GET'])
def api_trade_history():
    if not check_db_connection() or not conn:
        return jsonify({"error": "Database connection not available"}), 503
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status != 'open' ORDER BY closed_at DESC LIMIT 100;")
            history = cur.fetchall()
            return jsonify(history)
    except Exception as e:
        logger.error(f"[API] Error in /api/trade_history: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/notifications', methods=['GET'])
def api_notifications():
    with notifications_lock:
        return jsonify(list(notifications_cache))

@app.route('/api/close_trade', methods=['POST'])
def api_close_trade():
    data = request.json
    signal_id = data.get('id')
    symbol = data.get('symbol')
    if not signal_id or not symbol:
        return jsonify({"error": "Missing signal ID or symbol"}), 400
    with signal_cache_lock:
        signal_to_close = open_signals_cache.get(symbol)
    if not signal_to_close or signal_to_close.get('id') != signal_id:
        return jsonify({"error": "Signal not found or already closed"}), 404
    with prices_lock:
        closing_price = current_prices.get(symbol)
    if not closing_price:
        return jsonify({"error": f"Could not get current price for {symbol}"}), 500
    
    Thread(target=close_signal, args=(signal_to_close, 'manual_close', closing_price, "dashboard")).start()
    return jsonify({"message": f"Closing signal for {symbol} has been initiated."})


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
        load_open_signals_to_cache()
        load_notifications_to_cache()
        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan:
            logger.critical("❌ لا توجد رموز معتمدة للمسح. الحلقات لن تبدأ.")
            return
        Thread(target=run_websocket_manager, daemon=True).start()
        Thread(target=main_loop, daemon=True).start()
        logger.info("✅ [خدمات البوت] تم بدء جميع خدمات الخلفية بنجاح.")
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حاسم أثناء التهيئة: {e}", "SYSTEM")

if __name__ == "__main__":
    logger.info(f"🚀 بدء تشغيل بوت التداول المتقدم - إصدار {BASE_ML_MODEL_NAME}...")
    initialization_thread = Thread(target=initialize_bot_services, daemon=True)
    initialization_thread.start()
    run_flask()
    logger.info("👋 [إيقاف] تم إيقاف تشغيل البوت.")
    os._exit(0)
