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
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Union
from sklearn.preprocessing import StandardScaler
from collections import deque

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_pending_logic.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotPendingLogic')

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
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V5_Pending'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
DATA_FETCH_LOOKBACK_DAYS: int = 15

# --- Trading Logic Constants ---
MODEL_CONFIDENCE_THRESHOLD = 0.80
MAX_OPEN_TRADES: int = 5 # الحد الأقصى للصفقات المفتوحة فعلياً
ATR_SL_MULTIPLIER = 2.0
ATR_TP_MULTIPLIER = 2.5
MINIMUM_RISK_REWARD_RATIO = 1.2
MINIMUM_15M_VOLUME_USDT = 200_000

# --- المتغيرات العامة وقفل العمليات ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
validated_symbols_to_scan: List[str] = []
# --- **جديد**: فصل الذاكرة المؤقتة للصفقات ---
open_signals_cache: Dict[str, Dict] = {}
pending_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
current_prices: Dict[str, float] = {}
prices_lock = Lock()
notifications_cache = deque(maxlen=100)
notifications_lock = Lock()

# (بقية دوال الإعداد والمساعدة مثل init_db, log_and_notify, get_validated_symbols, fetch_historical_data, etc. تبقى كما هي في معظمها)
# ... سيتم إدراج الدوال التي لم تتغير هنا للاختصار ...
# --- دوال Binance والبيانات (معظمها بدون تغيير) ---
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
        return df[numeric_cols].dropna()
    except BinanceAPIException as e:
        logger.warning(f"⚠️ [API Binance] خطأ في جلب بيانات {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ [البيانات] خطأ أثناء جلب البيانات التاريخية لـ {symbol}: {e}")
        return None
# ---------------------- دوال قاعدة البيانات (مُعدَّلة) ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    global conn
    logger.info("[قاعدة البيانات] بدء تهيئة الاتصال...")
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False # Important for transaction management
            with conn.cursor() as cur:
                # The table structure is kept flexible with JSONB
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        status TEXT DEFAULT 'pending', -- **جديد**: القيمة الافتراضية هي pending
                        generation_price DOUBLE PRECISION, -- **جديد**: سعر توليد التوصية
                        entry_price DOUBLE PRECISION, -- سعر الدخول الفعلي (null for pending)
                        target_price DOUBLE PRECISION NOT NULL, -- الهدف الأصلي / الهدف الثاني
                        stop_loss DOUBLE PRECISION NOT NULL, -- وقف الخسارة للتوصية / للصفقة
                        closing_price DOUBLE PRECISION,
                        closed_at TIMESTAMP,
                        profit_percentage DOUBLE PRECISION,
                        strategy_name TEXT,
                        signal_details JSONB, -- لتخزين كل التفاصيل الإضافية
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                """)
                # A simple check for a potentially missing column from old versions
                cur.execute("SELECT 1 FROM information_schema.columns WHERE table_name='signals' AND column_name='generation_price'")
                if not cur.fetchone():
                    logger.info("[DB] Adding 'generation_price' column to 'signals' table.")
                    cur.execute("ALTER TABLE signals ADD COLUMN generation_price DOUBLE PRECISION;")

                cur.execute("""
                     CREATE TABLE IF NOT EXISTS notifications (
                         id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                         type TEXT NOT NULL, message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE
                     );
                """)

            conn.commit()
            logger.info("✅ [قاعدة البيانات] تم تهيئة قاعدة البيانات بنجاح.")
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
            with conn.cursor() as cur:
                 cur.execute("SELECT 1;")
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

# --- **جديد**: دالة لإدراج توصية "قيد الانتظار" ---
def insert_pending_signal_into_db(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not check_db_connection() or not conn: return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO signals (symbol, status, generation_price, target_price, stop_loss, strategy_name, signal_details)
                   VALUES (%s, 'pending', %s, %s, %s, %s, %s) RETURNING *;""",
                (
                    signal['symbol'],
                    signal['generation_price'],
                    signal['original_target'],
                    signal['trigger_price'], # وقف الخسارة الأولي هو سعر التفعيل
                    signal.get('strategy_name'),
                    json.dumps(signal.get('signal_details', {}))
                )
            )
            inserted_signal = dict(cur.fetchone())
        conn.commit()
        logger.info(f"✅ [DB] تم إدراج توصية قيد الانتظار لـ {signal['symbol']} (ID: {inserted_signal['id']}).")
        return inserted_signal
    except Exception as e:
        logger.error(f"❌ [DB Insert] خطأ في إدراج توصية {signal['symbol']}: {e}", exc_info=True)
        if conn: conn.rollback()
        return None

# --- **جديد**: تحميل جميع التوصيات الفعالة (مفتوحة وقيد الانتظار) ---
def load_active_signals_to_cache():
    if not check_db_connection() or not conn: return
    logger.info("ℹ️ [Cache] جاري تحميل جميع التوصيات الفعالة (المفتوحة وقيد الانتظار)...")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status IN ('open', 'pending');")
            active_signals = cur.fetchall()
            with signal_cache_lock:
                open_signals_cache.clear()
                pending_signals_cache.clear()
                for signal in active_signals:
                    signal_dict = dict(signal)
                    if signal_dict['status'] == 'open':
                        open_signals_cache[signal_dict['symbol']] = signal_dict
                    elif signal_dict['status'] == 'pending':
                        pending_signals_cache[signal_dict['symbol']] = signal_dict
            logger.info(f"✅ [Cache] تم تحميل {len(open_signals_cache)} صفقة مفتوحة و {len(pending_signals_cache)} توصية قيد الانتظار.")
    except Exception as e:
        logger.error(f"❌ [Cache Load] فشل تحميل التوصيات الفعالة: {e}")

# (TradingStrategy class and feature calculation functions remain unchanged)
class TradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = self.load_ml_model_bundle_from_folder(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    def load_ml_model_bundle_from_folder(self, symbol: str) -> Optional[Dict[str, Any]]:
        model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
        model_dir = 'Mo'
        file_path = os.path.join(model_dir, f"{model_name}.pkl")
        if not os.path.isdir(model_dir): return None
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            except Exception: return None
        return None

    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        # This function would be the same as in the original file
        # For brevity, assuming calculate_all_features exists and is correct
        try:
            # Placeholder for the actual feature calculation logic from your file
            df_calc = df_15m.copy()
            df_calc['atr'] = (df_calc['high'] - df_calc['low']).rolling(window=14).mean()
            # ... add all other feature calculations
            return df_calc.dropna()
        except Exception:
            return None


    def generate_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]): return None
        last_row = df_processed.iloc[-1]
        try:
            features_df = pd.DataFrame([last_row], columns=df_processed.columns)[self.feature_names]
            if features_df.isnull().values.any(): return None
            features_scaled = self.scaler.transform(features_df)
            prob_for_class_1 = self.ml_model.predict_proba(features_scaled)[0][1]
            if prob_for_class_1 >= MODEL_CONFIDENCE_THRESHOLD:
                logger.info(f"✅ [توليد توصية] {self.symbol}: تنبأ النموذج 'شراء' بثقة {prob_for_class_1:.2%}.")
                return {'symbol': self.symbol, 'strategy_name': BASE_ML_MODEL_NAME, 'signal_details': {'ML_Probability_Buy': f"{prob_for_class_1:.2%}"}}
            return None
        except Exception: return None

# --- **مُعدَّل**: معالج WebSocket لمراقبة كل من الصفقات المفتوحة والتوصيات قيد الانتظار ---
def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
    global open_signals_cache, pending_signals_cache, current_prices
    try:
        data = msg.get('data', msg) if isinstance(msg, dict) else msg
        if not isinstance(data, list): data = [data]

        for item in data:
            symbol = item.get('s')
            if not symbol: continue
            price = float(item.get('c', 0))
            if price == 0: continue
            with prices_lock: current_prices[symbol] = price

            # --- القسم الأول: مراقبة الصفقات المفتوحة (للوقف والهدف) ---
            signal_to_close, status, closing_price = None, None, None
            with signal_cache_lock:
                if symbol in open_signals_cache:
                    signal = open_signals_cache[symbol]
                    # Note: You might have multiple targets now. This logic checks the final target.
                    if price >= signal.get('target_price'):
                        status, closing_price, signal_to_close = 'target_hit', signal.get('target_price'), signal
                    elif price <= signal.get('stop_loss'):
                        status, closing_price, signal_to_close = 'stop_loss_hit', signal.get('stop_loss'), signal

            if signal_to_close and status:
                logger.info(f"⚡ [Tracker] تم تفعيل حدث إغلاق '{status}' للصفقة {symbol} عند سعر {price:.8f}")
                Thread(target=close_signal, args=(signal_to_close, status, closing_price, "auto")).start()
                continue # Move to next ticker item

            # --- القسم الثاني: مراقبة التوصيات قيد الانتظار (للتفعيل) ---
            signal_to_activate = None
            with signal_cache_lock:
                if symbol in pending_signals_cache:
                    signal = pending_signals_cache[symbol]
                    # trigger_price is the original stop_loss
                    if price <= signal.get('stop_loss'):
                        signal_to_activate = signal

            if signal_to_activate:
                logger.info(f"⚡ [Tracker] تم تفعيل الدخول للتوصية {symbol} عند سعر {price:.8f}")
                # The activation logic is now handled in its own function
                Thread(target=activate_pending_signal, args=(signal_to_activate, price)).start()

    except Exception as e:
        logger.error(f"❌ [WebSocket] خطأ في معالجة رسالة السعر: {e}", exc_info=True)


# --- **جديد**: دالة لتفعيل توصية قيد الانتظار وتحويلها لصفقة مفتوحة ---
def activate_pending_signal(signal_to_activate: Dict, activation_price: float):
    symbol = signal_to_activate['symbol']

    with signal_cache_lock:
        # تأكد من أن التوصية لا تزال قيد الانتظار ولم تتم معالجتها بالفعل
        if symbol not in pending_signals_cache or pending_signals_cache[symbol]['id'] != signal_to_activate['id']:
            logger.warning(f"⚠️ [{symbol}] تم تخطي التفعيل، ربما تمت معالجتها بالفعل.")
            return

    logger.info(f"🚀 [{symbol}] بدء عملية تفعيل التوصية (ID: {signal_to_activate['id']})...")

    # 1. استرجاع التفاصيل الأصلية
    generation_price = signal_to_activate['generation_price']
    original_target = signal_to_activate['target_price']
    atr_at_generation = signal_to_activate.get('signal_details', {}).get('atr_at_generation')

    if not atr_at_generation:
        logger.error(f"❌ [{symbol}] لا يمكن تفعيل التوصية. قيمة ATR عند التوليد مفقودة!")
        return

    # 2. حساب الأهداف ووقف الخسارة الجديد
    new_entry_price = activation_price
    new_target_1 = generation_price # الهدف الأول هو سعر التوليد
    new_target_2 = original_target  # الهدف الثاني هو الهدف الأصلي
    new_stop_loss = new_entry_price - (atr_at_generation * ATR_SL_MULTIPLIER)

    # 3. تحديث تفاصيل الصفقة
    updated_signal = signal_to_activate.copy()
    updated_signal['status'] = 'open'
    updated_signal['entry_price'] = new_entry_price
    updated_signal['target_price'] = new_target_2 # The main target is the second one
    updated_signal['stop_loss'] = new_stop_loss
    updated_signal['signal_details']['activated_at'] = datetime.now().isoformat()
    updated_signal['signal_details']['target_1'] = new_target_1
    updated_signal['signal_details']['original_entry_price'] = generation_price

    # 4. تحديث قاعدة البيانات
    if not check_db_connection() or not conn:
        logger.error(f"❌ [{symbol}] فشل تفعيل التوصية، لا يوجد اتصال بقاعدة البيانات.")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE signals
                SET status = 'open', entry_price = %s, target_price = %s, stop_loss = %s, signal_details = %s
                WHERE id = %s AND status = 'pending'
            """, (
                new_entry_price, new_target_2, new_stop_loss,
                json.dumps(updated_signal['signal_details']),
                signal_to_activate['id']
            ))
        conn.commit()
        logger.info(f"✅ [DB] تم تحديث حالة التوصية {symbol} إلى 'open' بنجاح.")
    except Exception as e:
        logger.error(f"❌ [DB Update] خطأ أثناء تفعيل التوصية {symbol}: {e}")
        if conn: conn.rollback()
        return

    # 5. تحديث الذاكرة المؤقتة وإرسال التنبيهات
    with signal_cache_lock:
        del pending_signals_cache[symbol]
        open_signals_cache[symbol] = updated_signal

    log_and_notify('info', f"تم تفعيل صفقة شراء لـ {symbol} بسعر دخول {new_entry_price:.8f}", "TRADE_ACTIVATED")
    send_trade_activated_alert(updated_signal)


# --- دوال التنبيهات (معدلة) ---
def send_trade_activated_alert(signal_data: Dict[str, Any]):
    safe_symbol = signal_data['symbol'].replace('_', '\\_')
    entry, target1, target2, sl = signal_data['entry_price'], signal_data['signal_details']['target_1'], signal_data['target_price'], signal_data['stop_loss']

    message = (f"✅ *تم تفعيل صفقة شراء جديدة*\n\n"
               f"🪙 *العملة:* `{safe_symbol}`\n\n"
               f"⬅️ *سعر الدخول:* `${entry:,.8g}`\n"
               f"🎯 *الهدف الأول:* `${target1:,.8g}`\n"
               f"🎯 *الهدف الثاني:* `${target2:,.8g}`\n"
               f"🛑 *وقف الخسارة:* `${sl:,.8g}`\n\n"
               f"🔍 *ثقة النموذج الأصلية:* {signal_data['signal_details']['ML_Probability_Buy']}")

    send_telegram_message(CHAT_ID, message)

def send_telegram_message(target_chat_id: str, text: str):
    if not TELEGRAM_TOKEN or not target_chat_id: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(target_chat_id), 'text': text, 'parse_mode': 'Markdown'}
    try: requests.post(url, json=payload, timeout=10)
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

# (دالة إغلاق الصفقة close_signal تبقى كما هي)
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

# --- **مُعدَّل**: حلقة العمل الرئيسية ---
def main_loop():
    logger.info("[الحلقة الرئيسية] انتظار اكتمال التهيئة الأولية...")
    time.sleep(10) # Wait for initial connections
    if not validated_symbols_to_scan:
        log_and_notify("critical", "لا توجد رموز معتمدة للمسح. لن يستمر البوت في العمل.", "SYSTEM")
        return
    log_and_notify("info", f"بدء حلقة المسح الرئيسية لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")

    while True:
        try:
            with signal_cache_lock:
                open_count = len(open_signals_cache)
            if open_count >= MAX_OPEN_TRADES:
                logger.info(f"ℹ️ [إيقاف مؤقت] تم الوصول للحد الأقصى للصفقات المفتوحة ({open_count}/{MAX_OPEN_TRADES}).")
                time.sleep(60)
                continue

            logger.info("ℹ️ [بدء المسح] بدء دورة مسح جديدة للبحث عن توصيات قيد الانتظار...")
            btc_data_cycle = fetch_historical_data('BTCUSDT', SIGNAL_GENERATION_TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
            if btc_data_cycle is None:
                logger.error("❌ فشل في جلب بيانات BTC. سيتم تخطي دورة المسح هذه.")
                time.sleep(120)
                continue
            
            for symbol in validated_symbols_to_scan:
                with signal_cache_lock:
                    if symbol in open_signals_cache or symbol in pending_signals_cache:
                        continue # تخطي إذا كانت هناك صفقة مفتوحة أو توصية قيد الانتظار بالفعل

                try:
                    df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
                    df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, DATA_FETCH_LOOKBACK_DAYS)
                    if df_15m is None or df_15m.empty or df_4h is None or df_4h.empty: continue

                    strategy = TradingStrategy(symbol)
                    df_features = strategy.get_features(df_15m, df_4h, btc_data_cycle)
                    if df_features is None or df_features.empty: continue

                    potential_signal = strategy.generate_signal(df_features)
                    if potential_signal:
                        last_candle = df_features.iloc[-1]
                        last_15m_volume_usdt = last_candle['volume'] * last_candle['close']
                        if last_15m_volume_usdt < MINIMUM_15M_VOLUME_USDT:
                            logger.info(f"📉 [{symbol}] تم تجاهل التوصية. حجم السيولة (${last_15m_volume_usdt:,.0f}) أقل من الحد الأدنى.")
                            continue

                        with prices_lock: current_price = current_prices.get(symbol)
                        if not current_price: continue

                        atr_value = df_features['atr'].iloc[-1]
                        stop_loss_price = current_price - (atr_value * ATR_SL_MULTIPLIER)
                        target_price = current_price + (atr_value * ATR_TP_MULTIPLIER)

                        # --- بناء كائن التوصية قيد الانتظار ---
                        pending_recommendation = {
                            'symbol': symbol,
                            'generation_price': current_price,
                            'original_target': target_price,
                            'trigger_price': stop_loss_price, # وقف الخسارة هو سعر التفعيل
                            'strategy_name': BASE_ML_MODEL_NAME,
                            'signal_details': {
                                'ML_Probability_Buy': potential_signal['signal_details']['ML_Probability_Buy'],
                                'atr_at_generation': atr_value,
                                'risk_reward_ratio_original': f"{((target_price - current_price) / (current_price - stop_loss_price)):.2f}:1"
                            }
                        }

                        saved_signal = insert_pending_signal_into_db(pending_recommendation)
                        if saved_signal:
                            with signal_cache_lock:
                                pending_signals_cache[saved_signal['symbol']] = saved_signal
                            log_and_notify('info', f"توصية جديدة قيد الانتظار لـ {symbol} بسعر توليد ${current_price:.8f}", "NEW_PENDING_SIGNAL")
                
                except Exception as e:
                    logger.error(f"❌ [خطأ في المعالجة] حدث خطأ أثناء معالجة العملة {symbol}: {e}", exc_info=True)
                finally:
                    gc.collect()

            logger.info("ℹ️ [نهاية المسح] انتهت دورة المسح. في انتظار 120 ثانية...")
            time.sleep(120)

        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as main_err:
            log_and_notify("error", f"خطأ غير متوقع في الحلقة الرئيسية: {main_err}", "SYSTEM")
            time.sleep(120)

# --- Flask API (معدل) ---
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(script_dir, 'index.html')
        with open(file_path, 'r', encoding='utf-8') as f:
            return render_template_string(f.read())
    except Exception as e:
        logger.error(f"Error rendering homepage: {e}", exc_info=True)
        return "<h1>Error loading dashboard file (index.html).</h1>", 500

@app.route('/api/signals')
def get_signals():
    if not check_db_connection() or not conn: return jsonify({"error": "فشل الاتصال بقاعدة البيانات"}), 500
    try:
        with conn.cursor() as cur:
            # جلب كل التوصيات وترتيبها حسب الحالة ثم التاريخ
            cur.execute("""
                SELECT *,
                       CASE
                           WHEN status = 'pending' THEN 1
                           WHEN status = 'open' THEN 2
                           ELSE 3
                       END as status_order
                FROM signals
                ORDER BY status_order ASC, created_at DESC;
            """)
            all_signals = [dict(s) for s in cur.fetchall()]

        for s in all_signals:
            # تحويل التواريخ إلى صيغة نصية للـ JSON
            for key in ['created_at', 'closed_at']:
                if s.get(key) and isinstance(s[key], datetime):
                    s[key] = s[key].isoformat()

            if s['status'] == 'open':
                with prices_lock:
                    s['current_price'] = current_prices.get(s['symbol'])
        return jsonify(all_signals)
    except Exception as e:
        logger.error(f"❌ [API إشارات] خطأ: {e}"); return jsonify({"error": "تعذر جلب الإشارات"}), 500

@app.route('/api/stats')
def get_stats():
    if not check_db_connection() or not conn: return jsonify({"error": "فشل الاتصال بقاعدة البيانات"}), 500
    try:
        with conn.cursor() as cur:
            # الإحصائيات تحسب فقط من الصفقات المغلقة
            cur.execute("SELECT profit_percentage FROM signals WHERE status NOT IN ('open', 'pending');")
            closed = cur.fetchall()
        wins = sum(1 for s in closed if s.get('profit_percentage', 0) > 0)
        losses = len(closed) - wins
        total_closed = len(closed)
        win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
        return jsonify({"win_rate": win_rate, "wins": wins, "losses": losses, "total_closed_trades": total_closed})
    except Exception as e:
        logger.error(f"❌ [API إحصائيات] خطأ: {e}"); return jsonify({"error": "تعذر جلب الإحصائيات"}), 500

# (بقية دوال Flask مثل manual_close و get_notifications تبقى كما هي)
@app.route('/api/close/<int:signal_id>', methods=['POST'])
def manual_close_signal(signal_id):
    logger.info(f"ℹ️ [API إغلاق] تم استلام طلب إغلاق يدوي للإشارة ID: {signal_id}")
    signal_to_close = None
    with signal_cache_lock:
        for s in open_signals_cache.values():
            if s['id'] == signal_id: signal_to_close = s.copy(); break
    if not signal_to_close: return jsonify({"error": "لم يتم العثور على الصفقة المفتوحة."}), 404
    
    with prices_lock: closing_price = current_prices.get(signal_to_close['symbol'])
    if not closing_price: return jsonify({"error": f"تعذر الحصول على السعر الحالي."}), 500
    
    Thread(target=close_signal, args=(signal_to_close, 'manual_close', closing_price, "manual")).start()
    return jsonify({"message": f"جاري إغلاق الصفقة {signal_id}."})

@app.route('/api/notifications')
def get_notifications():
    with notifications_lock: return jsonify(list(notifications_cache))

def run_flask():
    host, port = "0.0.0.0", int(os.environ.get('PORT', 10000))
    log_and_notify("info", f"بدء تشغيل لوحة التحكم على http://{host}:{port}", "SYSTEM")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        app.run(host=host, port=port)

# ---------------------- نقطة انطلاق البرنامج ----------------------
def initialize_bot_services():
    global client, validated_symbols_to_scan
    logger.info("🤖 [خدمات البوت] بدء التهيئة في الخلفية...")
    try:
        client = Client(API_KEY, API_SECRET)
        logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
        init_db()
        load_active_signals_to_cache() # تحميل التوصيات المفتوحة وقيد الانتظار
        # load_notifications_to_cache() # يمكنك تفعيلها إذا أردت
        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan:
            logger.critical("❌ لا توجد رموز معتمدة للمسح. الحلقات لن تبدأ.")
            return

        # Start WebSocket and main loop in background threads
        Thread(target=run_websocket_manager, daemon=True).start()
        Thread(target=main_loop, daemon=True).start()
        logger.info("✅ [خدمات البوت] تم بدء جميع خدمات الخلفية بنجاح.")
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حاسم أثناء التهيئة: {e}", "SYSTEM")

if __name__ == "__main__":
    logger.info(f"🚀 بدء تشغيل بوت التداول بمنطق التفعيل المؤجل...")
    Thread(target=initialize_bot_services, daemon=True).start()
    run_flask() # Flask runs in the main thread
    logger.info("👋 [إيقاف] تم إيقاف تشغيل البوت.")
    os._exit(0)
