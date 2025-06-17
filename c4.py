import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
# --- !!! تعديل: استيراد مجمع الاتصالات !!! ---
from psycopg2 import pool, sql, OperationalError, InterfaceError
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
# --- waitress هو خادم ويب أفضل من خادم فلاسك الافتراضي للبيئة الإنتاجية ---
from waitress import serve

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
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V5'
MODEL_CONFIDENCE_THRESHOLD = 0.55
TP_ATR_MULTIPLIER: float = 2.0
SL_ATR_MULTIPLIER: float = 1.5
MAX_OPEN_TRADES: int = 5
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 7
RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, ATR_PERIOD = 14, 12, 26, 9, 14
EMA_SLOW_PERIOD, EMA_FAST_PERIOD, BTC_CORR_PERIOD = 200, 50, 30

# --- !!! تعديل: استخدام مجمع الاتصالات بدلاً من اتصال واحد !!! ---
db_pool: Optional[pool.SimpleConnectionPool] = None

# --- المتغيرات العامة والأقفال ---
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

# ---------------------- دوال قاعدة البيانات والتنبيهات (مُعدَّلة) ----------------------
def init_db():
    """تهيئة مجمع اتصالات قاعدة البيانات لضمان الأمان في بيئة الخيوط المتعددة."""
    global db_pool
    if db_pool:
        return
    try:
        db_pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,  # السماح بـ 10 اتصالات كحد أقصى لجميع الخيوط
            dsn=DB_URL,
            cursor_factory=RealDictCursor
        )
        conn = db_pool.getconn()
        with conn.cursor() as cur:
            # التأكد من وجود جميع الجداول اللازمة
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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ml_models (
                    id SERIAL PRIMARY KEY, model_name TEXT NOT NULL UNIQUE,
                    model_data BYTEA NOT NULL, trained_at TIMESTAMP DEFAULT NOW(), metrics JSONB);""")
        conn.commit()
        db_pool.putconn(conn)
        logger.info("✅ [DB Pool] تم تهيئة مجمع اتصالات قاعدة البيانات بنجاح.")
    except Exception as e:
        logger.critical(f"❌ [DB Pool] خطأ في تهيئة مجمع الاتصالات: {e}", exc_info=True)
        exit(1)

def execute_db_query(query, params=None, fetch=None):
    """دالة مركزية لتنفيذ استعلامات قاعدة البيانات باستخدام المجمع."""
    if not db_pool:
        logger.error("❌ [DB] مجمع الاتصالات غير متاح.")
        return None
    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cur:
            cur.execute(query, params)
            if fetch == 'one':
                return cur.fetchone()
            if fetch == 'all':
                return cur.fetchall()
            conn.commit()
            # للحصول على القيمة المرجعة بعد الإدخال (مثل ID)
            if 'RETURNING' in query.upper():
                 return cur.fetchone()
            return True
    except Exception as e:
        logger.error(f"❌ [DB Query] فشل تنفيذ الاستعلام: {e}", exc_info=True)
        if conn: conn.rollback()
        return None
    finally:
        if conn: db_pool.putconn(conn)


def log_and_notify(level: str, message: str, notification_type: str):
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error}
    log_methods.get(level.lower(), logger.info)(message)
    with notifications_lock:
        notifications_cache.appendleft({"timestamp": datetime.now().isoformat(), "type": notification_type, "message": message})
    query = "INSERT INTO notifications (type, message) VALUES (%s, %s);"
    execute_db_query(query, (notification_type, message))

# ---------------------- دوال Binance والبيانات (مع تحديثات طفيفة) ----------------------
def get_validated_symbols(filename: str = 'crypto_list.txt'):
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

def update_btc_data_cache():
    logger.info("ℹ️ [BTC Data] تحديث بيانات البيتكوين...")
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
    while True:
        try:
            update_btc_data_cache()
            time.sleep(900)
        except Exception as e:
            logger.error(f"❌ [BTC Loop] خطأ في حلقة تحديث البيتكوين: {e}")
            time.sleep(60)

def calculate_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df_calc = df.copy()
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
    ema_fast_trend = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    ema_slow_trend = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    df_calc['price_vs_ema50'] = (df_calc['close'] / ema_fast_trend) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / ema_slow_trend) - 1
    if btc_df is not None:
        df_calc['returns'] = df_calc['close'].pct_change()
        merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
        df_calc['btc_correlation'] = merged_df['returns'].rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    else:
        df_calc['btc_correlation'] = 0
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=30, min_periods=1).mean() + 1e-9)
    df_calc['hour_of_day'] = df_calc.index.hour
    return df_calc.dropna()

def load_ml_model_bundle_from_db(symbol: str):
    global ml_models_cache
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_name in ml_models_cache: return ml_models_cache[model_name]
    
    query = "SELECT model_data FROM ml_models WHERE model_name = %s LIMIT 1;"
    res = execute_db_query(query, (model_name,), fetch='one')
    
    if res and res.get('model_data'):
        bundle = pickle.loads(res['model_data'])
        ml_models_cache[model_name] = bundle
        logger.info(f"✅ [ML] تم تحميل النموذج '{model_name}' من قاعدة البيانات.")
        return bundle
    
    logger.warning(f"⚠️ [ML] لم يتم العثور على النموذج '{model_name}'.")
    return None

# ---------------------- دوال WebSocket والاستراتيجية ----------------------
class TradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_bundle_from_db(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    def get_features(self, df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        return calculate_features(df, btc_df)

    def generate_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]): return None
        last_row = df_processed.iloc[-1:]
        try:
            features_df = last_row[self.feature_names]
            if features_df.isnull().values.any(): return None
            features_scaled = self.scaler.transform(features_df)
            prediction = self.ml_model.predict(features_scaled)[0]
            if prediction != 1: return None
            prediction_proba = self.ml_model.predict_proba(features_scaled)[0]
            confidence_for_class_1 = prediction_proba[np.where(self.ml_model.classes_ == 1)[0][0]]
            if confidence_for_class_1 < MODEL_CONFIDENCE_THRESHOLD: return None
            logger.info(f"✅ [Signal Found] {self.symbol}: إشارة شراء محتملة بثقة {confidence_for_class_1:.2%}.")
            return {'symbol': self.symbol, 'strategy_name': BASE_ML_MODEL_NAME, 'signal_details': {'ML_Confidence': f"{confidence_for_class_1:.2%}"}}
        except Exception as e:
            logger.warning(f"⚠️ [Signal Gen] {self.symbol}: خطأ: {e}"); return None

def close_signal(signal, status, closing_price, closed_by):
    entry_price = signal['entry_price']
    profit = ((closing_price - entry_price) / entry_price) * 100
    
    query = """
        UPDATE signals SET status = %s, closing_price = %s, closed_at = NOW(), profit_percentage = %s
        WHERE id = %s;
    """
    params = (status, closing_price, profit, signal['id'])
    execute_db_query(query, params)

    log_and_notify('info', (
        f"🔴 [{closed_by}] {signal['symbol']}: Closed at ${closing_price:.4f}. "
        f"Profit: {profit:.2f}% (Entry: ${entry_price:.4f})"
    ), "TRADE_CLOSE")
    
    with signal_cache_lock:
        if signal['symbol'] in open_signals_cache:
            del open_signals_cache[signal['symbol']]

def handle_ticker_message(msg):
    if msg.get('e') != '24hrTicker': return
    symbol = msg['s']
    price = float(msg['c'])
    with prices_lock: current_prices[symbol] = price
    with signal_cache_lock:
        signal = open_signals_cache.get(symbol)
        if not signal: return
    
    if price >= signal['target_price']:
        close_signal(signal, 'closed_tp', signal['target_price'], 'TP')
    elif price <= signal['stop_loss']:
        close_signal(signal, 'closed_sl', signal['stop_loss'], 'SL')

def run_websocket_manager():
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    symbols_for_ws = [s.lower()+'@ticker' for s in validated_symbols_to_scan]
    if symbols_for_ws:
        twm.start_multiplex_socket(callback=handle_ticker_message, streams=symbols_for_ws)
        logger.info(f"✅ [WebSocket] تم بدء مراقبة Ticker لـ {len(symbols_for_ws)} عملة.")
    twm.join()

# ---------------------- دوال الإدارة ----------------------
def insert_signal_into_db(signal):
    query = """
        INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, trailing_stop_price)
        VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id, symbol, entry_price, target_price, stop_loss, trailing_stop_price;
    """
    params = (
        signal['symbol'], signal['entry_price'], signal['target_price'], signal['stop_loss'],
        signal['strategy_name'], json.dumps(signal.get('signal_details', {})), signal.get('trailing_stop_price')
    )
    new_signal_record = execute_db_query(query, params, fetch='one')
    if new_signal_record:
        log_and_notify('info', (
            f"🚀 [New Signal] {signal['symbol']}: Entry: ${signal['entry_price']:.4f}, "
            f"TP: ${signal['target_price']:.4f}, SL: ${signal['stop_loss']:.4f}"
        ), "NEW_TRADE")
    return new_signal_record

def load_open_signals_to_cache():
    logger.info("ℹ️ [Cache] تحميل الصفقات المفتوحة إلى الذاكرة المؤقتة...")
    query = "SELECT * FROM signals WHERE status = 'open';"
    open_signals = execute_db_query(query, fetch='all')
    if open_signals is not None:
        with signal_cache_lock:
            open_signals_cache.clear()
            for s in open_signals:
                open_signals_cache[s['symbol']] = dict(s)
        logger.info(f"✅ [Cache] تم تحميل {len(open_signals)} صفقة مفتوحة.")

# ---------------------- حلقة العمل الرئيسية ----------------------
def main_loop():
    logger.info("[Main Loop] انتظار اكتمال التهيئة الأولية...")
    time.sleep(15) 
    if not validated_symbols_to_scan:
        log_and_notify("critical", "لا توجد رموز معتمدة للمسح.", "SYSTEM"); return
    
    log_and_notify("info", f"بدء حلقة المسح لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")
    
    while True:
        try:
            with signal_cache_lock: open_count = len(open_signals_cache)
            if open_count >= MAX_OPEN_TRADES:
                logger.info(f"ℹ️ [Pause] تم الوصول للحد الأقصى للصفقات ({open_count})."); time.sleep(60); continue
            
            slots_available = MAX_OPEN_TRADES - open_count
            logger.info(f"ℹ️ [Scan] بدء دورة مسح جديدة. المراكز المتاحة: {slots_available}")
            with btc_data_lock: current_btc_data = btc_data_cache
            if current_btc_data is None:
                logger.warning("⚠️ [Scan] بيانات البيتكوين غير متاحة. سيتم تخطي هذه الدورة."); time.sleep(60); continue

            for symbol in validated_symbols_to_scan:
                if slots_available <= 0: break
                with signal_cache_lock:
                    if symbol in open_signals_cache: continue
                
                try:
                    df_hist = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_hist is None or df_hist.empty: continue
                    
                    strategy = TradingStrategy(symbol)
                    df_features = strategy.get_features(df_hist, current_btc_data)
                    if df_features is None or df_features.empty: continue
                    
                    potential_signal = strategy.generate_signal(df_features)
                    if potential_signal:
                        with prices_lock: current_price = current_prices.get(symbol)
                        if not current_price: continue
                        potential_signal['entry_price'] = current_price
                        atr_value = df_features['atr'].iloc[-1]
                        potential_signal['stop_loss'] = current_price - (atr_value * SL_ATR_MULTIPLIER)
                        potential_signal['target_price'] = current_price + (atr_value * TP_ATR_MULTIPLIER)
                        potential_signal['trailing_stop_price'] = potential_signal['stop_loss']
                        saved_signal = insert_signal_into_db(potential_signal)
                        if saved_signal:
                            with signal_cache_lock: open_signals_cache[saved_signal['symbol']] = saved_signal
                            slots_available -= 1
                except Exception as e:
                    logger.error(f"❌ [Processing Error] {symbol}: {e}", exc_info=True)
            logger.info("ℹ️ [Scan End] انتهت دورة المسح."); time.sleep(90)
        except Exception as main_err:
            log_and_notify("error", f"خطأ في الحلقة الرئيسية: {main_err}", "SYSTEM"); time.sleep(120)

# ---------------------- واجهة API والتشغيل ----------------------
app = Flask(__name__)
CORS(app)

@app.route('/')
def home(): return "Crypto Trading Bot V5 is running.", 200
@app.route('/status')
def get_status():
    with signal_cache_lock: open_trades = list(open_signals_cache.values())
    with prices_lock: prices = dict(current_prices)
    return jsonify({'open_trades': open_trades, 'current_prices': prices, 'max_trades': MAX_OPEN_TRADES})
@app.route('/notifications')
def get_notifications():
    with notifications_lock: notifs = list(notifications_cache)
    return jsonify(notifs)

def initialize_bot_services():
    global client, validated_symbols_to_scan
    logger.info("🤖 [Init] بدء تهيئة خدمات البوت V5...")
    try:
        init_db()
        client = Client(API_KEY, API_SECRET)
        load_open_signals_to_cache()
        
        Thread(target=btc_cache_updater_loop, daemon=True).start()
        logger.info("... انتظار أول جلب لبيانات البيتكوين ...")
        time.sleep(10)

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
    
    # --- التشغيل باستخدام خادم إنتاجي ---
    host = "0.0.0.0"
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"🌍 بدء تشغيل خادم الويب على {host}:{port}")
    serve(app, host=host, port=port)
