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
        logging.FileHandler('crypto_bot_v5.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV5')

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
# --- !!! تحديثات رئيسية للنموذج V5 !!! ---
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V5'
# عتبة الثقة في تنبؤ النموذج (للصنف 1 فقط)
MODEL_CONFIDENCE_THRESHOLD = 0.55
# معلمات تحديد الهدف (يجب أن تطابق ملف التدريب)
TP_ATR_MULTIPLIER: float = 2.0
SL_ATR_MULTIPLIER: float = 1.5

# --- إعدادات البوت العامة ---
MAX_OPEN_TRADES: int = 5
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 7 # فترة قصيرة لجلب البيانات بسرعة للمسح

# --- معلمات المؤشرات (للتوافق مع التدريب) ---
RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, ATR_PERIOD = 14, 12, 26, 9, 14
EMA_SLOW_PERIOD, EMA_FAST_PERIOD, BTC_CORR_PERIOD = 200, 50, 30

# --- المتغيرات العامة والأقفال ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
ml_models_cache: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
current_prices: Dict[str, float] = {}
btc_data_cache: Optional[pd.DataFrame] = None
signal_cache_lock = Lock()
prices_lock = Lock()
btc_data_lock = Lock()
notifications_cache = deque(maxlen=50)
notifications_lock = Lock()

# ---------------------- دوال قاعدة البيانات والتنبيهات (بدون تغيير) ----------------------
def init_db():
    global conn
    # ... (الكود موجود في الملف الأصلي ولم يتغير)
    try:
        conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION NOT NULL,
                    target_price DOUBLE PRECISION NOT NULL, stop_loss DOUBLE PRECISION NOT NULL,
                    status TEXT DEFAULT 'open', closing_price DOUBLE PRECISION, closed_at TIMESTAMP,
                    profit_percentage DOUBLE PRECISION, strategy_name TEXT, signal_details JSONB,
                    trailing_stop_price DOUBLE PRECISION);""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS notifications (
                    id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    type TEXT NOT NULL, message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE);""")
        conn.commit()
        logger.info("✅ [DB] تم تهيئة جداول قاعدة البيانات بنجاح.")
    except Exception as e:
        logger.critical(f"❌ [DB] خطأ في الاتصال: {e}"); exit(1)

def check_db_connection():
    # ... (الكود موجود في الملف الأصلي ولم يتغير)
    global conn
    if conn is None or conn.closed != 0:
        logger.warning("[DB] الاتصال مغلق، محاولة إعادة الاتصال...")
        init_db()
    try:
        if conn: conn.cursor().execute("SELECT 1;"); return True
        return False
    except (OperationalError, InterfaceError):
        logger.error("[DB] فقدان الاتصال. محاولة إعادة الاتصال...")
        try: init_db(); return conn is not None and conn.closed == 0
        except Exception as retry_e: logger.error(f"[DB] فشل إعادة الاتصال: {retry_e}"); return False
    return False

def log_and_notify(level: str, message: str, notification_type: str):
    # ... (الكود موجود في الملف الأصلي ولم يتغير)
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error}
    log_methods.get(level.lower(), logger.info)(message)
    if not check_db_connection() or not conn: return
    try:
        with notifications_lock: notifications_cache.appendleft({"timestamp": datetime.now().isoformat(), "type": notification_type, "message": message})
        with conn.cursor() as cur: cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [Notify DB] فشل حفظ التنبيه: {e}"); conn.rollback()

# ---------------------- دوال Binance والبيانات (مع تحديثات) ----------------------
def get_validated_symbols(filename: str = 'crypto_list.txt'):
    # ... (الكود موجود في الملف الأصلي ولم يتغير)
    try:
        with open(os.path.join(os.path.dirname(__file__), filename), 'r', encoding='utf-8') as f:
            symbols = {s.strip().upper() for s in f if s.strip() and not s.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in symbols}
        info = client.get_exchange_info()
        active = {s['symbol'] for s in info['symbols'] if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT'}
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ [Validation] سيتم مراقبة {len(validated)} عملة.")
        return validated
    except Exception as e:
        logger.error(f"❌ [Validation] خطأ في التحقق من الرموز: {e}"); return []

def fetch_historical_data(symbol: str, interval: str, days: int):
    # ... (الكود موجود في الملف الأصلي ولم يتغير)
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
        logger.error(f"❌ [Data] خطأ جلب البيانات لـ {symbol}: {e}"); return None

# --- !!! جديد: دوال إدارة بيانات البيتكوين !!! ---
def update_btc_data_cache():
    """تجلب بيانات البيتكوين وتحدث الذاكرة المؤقتة."""
    logger.info("ℹ️ [BTC Data] تحديث بيانات البيتكوين...")
    # Fetch a bit more data for indicator calculation
    temp_btc_df = fetch_historical_data('BTCUSDT', SIGNAL_GENERATION_TIMEFRAME, days=15)
    if temp_btc_df is not None:
        with btc_data_lock:
            global btc_data_cache
            temp_btc_df['btc_returns'] = temp_btc_df['close'].pct_change()
            btc_data_cache = temp_btc_df
            logger.info(f"✅ [BTC Data] تم تحديث ذاكرة البيتكوين. آخر سجل: {btc_data_cache.index[-1]}")
    else:
        logger.warning("⚠️ [BTC Data] فشل تحديث بيانات البيتكوين في هذه الدورة.")

def btc_cache_updater_loop():
    """حلقة تعمل في الخلفية لتحديث بيانات البيتكوين بشكل دوري."""
    while True:
        try:
            update_btc_data_cache()
            # Update every 15 minutes
            time.sleep(900)
        except Exception as e:
            logger.error(f"❌ [BTC Loop] خطأ في حلقة تحديث البيتكوين: {e}")
            time.sleep(60)

# --- !!! تحديث: دالة حساب الميزات لتتوافق مع V5 !!! ---
def calculate_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df_calc = df.copy()

    # Standard Indicators
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    ema_fast = df_calc['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df_calc['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df_calc['macd_hist'] = (ema_fast - ema_slow) - (ema_fast - ema_slow).ewm(span=MACD_SIGNAL, adjust=False).mean()

    # V5 Trend Features
    ema_fast_trend = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    ema_slow_trend = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    df_calc['price_vs_ema50'] = (df_calc['close'] / ema_fast_trend) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / ema_slow_trend) - 1
    
    # V5 BTC Correlation Feature
    if btc_df is not None:
        df_calc['returns'] = df_calc['close'].pct_change()
        merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
        df_calc['btc_correlation'] = merged_df['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    else:
        df_calc['btc_correlation'] = 0 # Default value if BTC data is missing

    # Other Features
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=30, min_periods=1).mean() + 1e-9)
    df_calc['hour_of_day'] = df_calc.index.hour
    
    return df_calc.dropna()

def load_ml_model_bundle_from_db(symbol: str):
    # ... (الكود موجود في الملف الأصلي ولم يتغير)
    global ml_models_cache
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_name in ml_models_cache: return ml_models_cache[model_name]
    if not check_db_connection() or not conn: return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT model_data FROM ml_models WHERE model_name = %s LIMIT 1;", (model_name,))
            res = cur.fetchone()
            if res and res['model_data']:
                bundle = pickle.loads(res['model_data'])
                ml_models_cache[model_name] = bundle
                logger.info(f"✅ [ML] تم تحميل النموذج '{model_name}' من قاعدة البيانات.")
                return bundle
        logger.warning(f"⚠️ [ML] لم يتم العثور على النموذج '{model_name}'.")
        return None
    except Exception as e:
        logger.error(f"❌ [ML] خطأ في تحميل النموذج لـ {symbol}: {e}"); return None

# ---------------------- دوال WebSocket والاستراتيجية (مع تحديثات) ----------------------
class TradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_bundle_from_db(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    def get_features(self, df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        return calculate_features(df, btc_df)

    # --- !!! تحديث: منطق توليد الإشارة لنموذج V5 متعدد الفئات !!! ---
    def generate_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]):
            return None
        
        last_row = df_processed.iloc[-1:]
        try:
            features_df = last_row[self.feature_names]
            if features_df.isnull().values.any(): return None
                
            features_scaled = self.scaler.transform(features_df)
            
            # التنبؤ بالصنف (1, 0, -1)
            prediction = self.ml_model.predict(features_scaled)[0]
            
            # نريد التداول فقط عندما يتوقع النموذج ربحاً (الصنف 1)
            if prediction != 1:
                return None
                
            # التحقق من ثقة النموذج في هذا التنبؤ
            prediction_proba = self.ml_model.predict_proba(features_scaled)[0]
            confidence_for_class_1 = prediction_proba[np.where(self.ml_model.classes_ == 1)[0][0]]
            
            if confidence_for_class_1 < MODEL_CONFIDENCE_THRESHOLD:
                return None
            
            logger.info(f"✅ [Signal Found] {self.symbol}: إشارة شراء محتملة (الصنف 1) بثقة {confidence_for_class_1:.2%}.")
            return {
                'symbol': self.symbol,
                'strategy_name': BASE_ML_MODEL_NAME,
                'signal_details': {'ML_Confidence': f"{confidence_for_class_1:.2%}"}
            }
        except Exception as e:
            logger.warning(f"⚠️ [Signal Gen] {self.symbol}: خطأ: {e}")
            return None

def handle_ticker_message(msg):
    # ... (الكود موجود في الملف الأصلي ولم يتغير، يعالج إغلاق الصفقات)
    # ملاحظة: منطق وقف الخسارة المتحرك يعتمد على السعر، لذا لا يحتاج لتغيير.
    pass # The logic here is for closing trades and remains the same.

def run_websocket_manager():
    # ... (الكود موجود في الملف الأصلي ولم يتغير)
    pass # This function starts the websocket.

# ---------------------- دوال الإدارة (بدون تغيير كبير) ----------------------
def insert_signal_into_db(signal):
    # ... (الكود موجود في الملف الأصلي ولم يتغير)
    pass

def close_signal(signal, status, closing_price, closed_by):
    # ... (الكود موجود في الملف الأصلي ولم يتغير)
    pass

def load_open_signals_to_cache():
    # ... (الكود موجود في الملف الأصلي ولم يتغير)
    pass

# ---------------------- حلقة العمل الرئيسية (مع تحديثات) ----------------------
def main_loop():
    logger.info("[Main Loop] انتظار اكتمال التهيئة الأولية...")
    time.sleep(15) 
    
    if not validated_symbols_to_scan:
        log_and_notify("critical", "لا توجد رموز معتمدة للمسح.", "SYSTEM"); return
    
    log_and_notify("info", f"بدء حلقة المسح لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")
    
    while True:
        try:
            # --- !!! إزالة فلتر BTC القديم، أصبح مدمجاً في النموذج !!! ---

            with signal_cache_lock: open_count = len(open_signals_cache)
            if open_count >= MAX_OPEN_TRADES:
                logger.info(f"ℹ️ [Pause] تم الوصول للحد الأقصى للصفقات ({open_count}).")
                time.sleep(60); continue
            
            slots_available = MAX_OPEN_TRADES - open_count
            logger.info(f"ℹ️ [Scan] بدء دورة مسح جديدة. المراكز المتاحة: {slots_available}")
            
            with btc_data_lock:
                current_btc_data = btc_data_cache
            
            if current_btc_data is None:
                logger.warning("⚠️ [Scan] بيانات البيتكوين غير متاحة. سيتم تخطي هذه الدورة.")
                time.sleep(60); continue

            for symbol in validated_symbols_to_scan:
                if slots_available <= 0: break
                with signal_cache_lock:
                    if symbol in open_signals_cache: continue
                
                try:
                    df_hist = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_hist is None or df_hist.empty: continue
                    
                    strategy = TradingStrategy(symbol)
                    # --- تمرير بيانات البيتكوين لحساب الميزات ---
                    df_features = strategy.get_features(df_hist, current_btc_data)
                    if df_features is None or df_features.empty: continue
                    
                    potential_signal = strategy.generate_signal(df_features)
                    if potential_signal:
                        with prices_lock: current_price = current_prices.get(symbol)
                        if not current_price: continue

                        potential_signal['entry_price'] = current_price
                        
                        # --- !!! تحديث: تحديد SL/TP بناءً على ATR !!! ---
                        atr_value = df_features['atr'].iloc[-1]
                        potential_signal['stop_loss'] = current_price - (atr_value * SL_ATR_MULTIPLIER)
                        potential_signal['target_price'] = current_price + (atr_value * TP_ATR_MULTIPLIER)
                        potential_signal['trailing_stop_price'] = potential_signal['stop_loss'] # Initial TSL

                        saved_signal = insert_signal_into_db(potential_signal)
                        if saved_signal:
                            with signal_cache_lock: open_signals_cache[saved_signal['symbol']] = saved_signal
                            # send_new_signal_alert(saved_signal) # Assumed function
                            slots_available -= 1
                except Exception as e:
                    logger.error(f"❌ [Processing Error] {symbol}: {e}", exc_info=True)

            logger.info("ℹ️ [Scan End] انتهت دورة المسح.")
            time.sleep(90) # A bit longer sleep
        except Exception as main_err:
            log_and_notify("error", f"خطأ في الحلقة الرئيسية: {main_err}", "SYSTEM")
            time.sleep(120)

# ---------------------- واجهة API والتشغيل (مع تحديثات) ----------------------
app = Flask(__name__)
CORS(app)
# ... The Flask routes from the original file can be copied here without change ...

def run_flask():
    host, port = "0.0.0.0", int(os.environ.get('PORT', 10000))
    log_and_notify("info", f"بدء تشغيل لوحة التحكم على {host}:{port}", "SYSTEM")
    # ... (use waitress or app.run)

def initialize_bot_services():
    global client, validated_symbols_to_scan
    logger.info("🤖 [Init] بدء تهيئة خدمات البوت V5...")
    try:
        client = Client(API_KEY, API_SECRET)
        init_db()
        load_open_signals_to_cache()
        # load_notifications_to_cache()
        
        # --- بدء محدث بيانات البيتكوين في الخلفية ---
        Thread(target=btc_cache_updater_loop, daemon=True).start()
        logger.info("... انتظار أول جلب لبيانات البيتكوين ...")
        time.sleep(10) # Give it time to fetch first data

        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan:
            logger.critical("❌ لا توجد رموز معتمدة للمسح.")
            return

        Thread(target=run_websocket_manager, daemon=True).start()
        Thread(target=main_loop, daemon=True).start()
        logger.info("✅ [Init] تم بدء جميع خدمات الخلفية بنجاح.")
    except Exception as e:
        log_and_notify("critical", f"خطأ حاسم أثناء التهيئة: {e}", "SYSTEM")

if __name__ == "__main__":
    logger.info("🚀 بدء تشغيل تطبيق بوت التداول V5...")
    initialization_thread = Thread(target=initialize_bot_services, daemon=True)
    initialization_thread.start()
    run_flask()
