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
        logging.FileHandler('crypto_bot_v6_reversal_entry.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV6_ReversalEntry')

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
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V6_Reversal'
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
MODEL_CONFIDENCE_THRESHOLD = 0.80
MAX_OPEN_TRADES: int = 5
USE_SR_LEVELS = True
MINIMUM_SR_SCORE = 30
ATR_SL_MULTIPLIER_ON_ENTRY = 1.5
ATR_TP_MULTIPLIER_ON_ENTRY = 2.0
USE_BTC_TREND_FILTER = True
BTC_TREND_TIMEFRAME = '4h'
BTC_TREND_EMA_PERIOD = 10

# --- ثوابت فلترة التوصيات ---
MINIMUM_PROFIT_PERCENTAGE = 0.5
MINIMUM_RISK_REWARD_RATIO = 1.2
MINIMUM_15M_VOLUME_USDT = 200_000

# --- المتغيرات العامة وقفل العمليات ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
validated_symbols_to_scan: List[str] = []
pending_recommendations_cache: Dict[str, Dict] = {}
recommendations_cache_lock = Lock()
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
current_prices: Dict[str, float] = {}
prices_lock = Lock()
notifications_cache = deque(maxlen=50)
notifications_lock = Lock()


# ---------------------- دوال قاعدة البيانات (مُعدَّلة) ----------------------
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
                    CREATE TABLE IF NOT EXISTS recommendations (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL UNIQUE,
                        generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        original_entry_price DOUBLE PRECISION NOT NULL,
                        original_target_price DOUBLE PRECISION NOT NULL,
                        entry_trigger_price DOUBLE PRECISION NOT NULL,
                        status TEXT DEFAULT 'waiting',
                        signal_details JSONB,
                        triggered_at TIMESTAMP WITH TIME ZONE
                    );
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
                        level_type TEXT NOT NULL, timeframe TEXT NOT NULL, strength NUMERIC NOT NULL, score NUMERIC DEFAULT 0,
                        last_tested_at TIMESTAMP WITH TIME ZONE, details TEXT, created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        CONSTRAINT unique_level UNIQUE (symbol, level_price, timeframe, level_type)
                    );
                """)
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

def fetch_sr_levels(symbol: str) -> Optional[List[Dict]]:
    if not check_db_connection() or not conn: return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT level_price, level_type, score FROM support_resistance_levels WHERE symbol = %s ORDER BY level_price ASC",
                (symbol,)
            )
            levels = cur.fetchall()
            if not levels: return None
            for level in levels: level['score'] = float(level.get('score', 0))
            return levels
    except Exception as e:
        logger.error(f"❌ [{symbol}] خطأ أثناء جلب مستويات الدعم والمقاومة: {e}")
        if conn: conn.rollback()
        return None

# ---------------------- دوال Binance والبيانات ----------------------
def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
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
    except Exception as e:
        logger.error(f"❌ [البيانات] خطأ أثناء جلب البيانات التاريخية لـ {symbol}: {e}")
    return None

# --- Other data and ML functions (calculate_all_features, load_ml_model_bundle_from_folder, etc.) remain here ---
# These functions are unchanged and are omitted for brevity.
def calculate_all_features(df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    # This function remains unchanged
    df_calc = df_15m.copy()
    high_low = df_calc['high'] - df_calc['low']; high_close = (df_calc['high'] - df_calc['close'].shift()).abs(); low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    # ... rest of the calculations
    return df_calc.dropna()

def load_ml_model_bundle_from_folder(symbol: str) -> Optional[Dict[str, Any]]:
    # This function remains unchanged
    # ... implementation
    return None
    
class TradingStrategy:
    # This class remains unchanged
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_bundle_from_folder(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        return calculate_all_features(df_15m, df_4h, btc_df)

    def generate_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        # ... implementation
        return None

# ---------------------- دوال WebSocket والاستراتيجية ----------------------
def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
    global open_signals_cache, current_prices, pending_recommendations_cache
    try:
        data = msg.get('data', msg) if isinstance(msg, dict) else msg
        if not isinstance(data, list): data = [data]

        for item in data:
            symbol = item.get('s')
            if not symbol: continue
            price = float(item.get('c', 0))
            if price == 0: continue
            with prices_lock: current_prices[symbol] = price

            # Check active trades
            signal_to_process, status, closing_price = None, None, None
            with signal_cache_lock:
                if symbol in open_signals_cache:
                    signal = open_signals_cache[symbol]
                    if price >= signal['target_price']: status, closing_price, signal_to_process = 'target_hit', signal['target_price'], signal
                    elif price <= signal['stop_loss']: status, closing_price, signal_to_process = 'stop_loss_hit', signal['stop_loss'], signal
            if signal_to_process:
                Thread(target=close_signal, args=(signal_to_process, status, closing_price, "auto")).start()
                continue

            # Check pending recommendations
            rec_to_trigger = None
            with recommendations_cache_lock:
                if symbol in pending_recommendations_cache:
                    if price <= pending_recommendations_cache[symbol]['entry_trigger_price']:
                        rec_to_trigger = pending_recommendations_cache[symbol]
            if rec_to_trigger:
                Thread(target=open_trade_from_recommendation, args=(rec_to_trigger, price)).start()

    except Exception as e:
        logger.error(f"❌ [متتبع WebSocket] خطأ في معالجة رسالة السعر: {e}", exc_info=True)


def run_websocket_manager() -> None:
    logger.info("ℹ️ [WebSocket] بدء مدير WebSocket...")
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    streams = [f"{s.lower()}@ticker" for s in validated_symbols_to_scan] if validated_symbols_to_scan else []
    if not streams:
        logger.error("❌ [WebSocket] لا توجد رموز صالحة للاستماع إليها.")
        return
    twm.start_multiplex_socket(callback=handle_ticker_message, streams=streams)
    logger.info(f"✅ [WebSocket] تم الاتصال والاستماع لـ {len(streams)} عملة.")
    twm.join()
    
# ---------------------- Management & Alerting Functions ----------------------
# All functions like send_telegram_message, save_or_update_recommendation_in_db,
# insert_signal_into_db, open_trade_from_recommendation, close_signal, load_data_to_cache
# remain here. They are unchanged and omitted for brevity.
# ...

# ---------------------- Main Loop ----------------------
def get_btc_trend() -> Dict[str, Any]:
    if not client: return {"status": "Error", "message": "Binance client not initialized", "is_uptrend": False}
    try:
        klines = client.get_klines(symbol=BTC_SYMBOL, interval=BTC_TREND_TIMEFRAME, limit=BTC_TREND_EMA_PERIOD * 2)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df['close'] = pd.to_numeric(df['close'])
        ema = df['close'].ewm(span=BTC_TREND_EMA_PERIOD, adjust=False).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        is_uptrend = current_price > ema
        status = "Uptrend" if is_uptrend else "Downtrend"
        message = f"صاعد (السعر فوق EMA {BTC_TREND_EMA_PERIOD})" if is_uptrend else f"هابط (السعر تحت EMA {BTC_TREND_EMA_PERIOD})"
        return {"status": status, "message": message, "is_uptrend": is_uptrend}
    except Exception as e:
        logger.error(f"❌ [فلتر BTC] فشل تحديد اتجاه البيتكوين: {e}")
        return {"status": "Error", "message": str(e), "is_uptrend": False}

# The main_loop function remains here, unchanged, and is omitted for brevity.
# ...

# ---------------------- Flask API (FIXED) ----------------------
app = Flask(__name__)
CORS(app)

# **FIXED**: Re-implemented the missing function
def get_fear_and_greed_index() -> Dict[str, Any]:
    """
    Fetches the Fear and Greed Index from alternative.me API.
    """
    classification_translation = {
        "Extreme Fear": "خوف شديد", "Fear": "خوف", "Neutral": "محايد", 
        "Greed": "طمع", "Extreme Greed": "طمع شديد", "Error": "خطأ"
    }
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        response.raise_for_status()
        data = response.json()['data'][0]
        original_classification = data['value_classification']
        return {
            "value": int(data['value']),
            "classification": classification_translation.get(original_classification, original_classification)
        }
    except Exception as e:
        logger.error(f"❌ [مؤشر الخوف والطمع] فشل الاتصال بالـ API: {e}")
        return {"value": -1, "classification": classification_translation["Error"]}

@app.route('/')
def home():
    try:
        # It's better to read the file from the filesystem to allow easy edits
        return render_template_string(open('index.html', 'r', encoding='utf-8').read())
    except FileNotFoundError:
        return "<h1>ملف لوحة التحكم (index.html) غير موجود.</h1><p>تأكد من وجود الملف في نفس مجلد السكريبت.</p>", 404
    except Exception as e:
        return f"<h1>خطأ في تحميل لوحة التحكم:</h1><p>{e}</p>", 500

# **FIXED**: Correctly structured the API response
@app.route('/api/market_status')
def get_market_status():
    """
    Provides market status including BTC trend and Fear & Greed index.
    This is the fixed endpoint that caused the error.
    """
    btc_trend_data = get_btc_trend()
    fear_greed_data = get_fear_and_greed_index()
    
    # The response is now a clean dictionary that jsonify can handle.
    return jsonify({
        "btc_trend": btc_trend_data,
        "fear_and_greed": fear_greed_data
    })

@app.route('/api/stats')
def get_stats():
    if not check_db_connection() or not conn: return jsonify({"error": "فشل الاتصال بقاعدة البيانات"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT status, profit_percentage FROM signals WHERE status != 'open';")
            closed = cur.fetchall()
        wins = sum(1 for s in closed if s.get('profit_percentage', 0) > 0)
        losses = len(closed) - wins
        total_closed = len(closed)
        win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
        total_profit = sum(s['profit_percentage'] for s in closed if s.get('profit_percentage') is not None)
        return jsonify({
            "win_rate": win_rate, "wins": wins, "losses": losses,
            "total_profit_percent": total_profit, "total_closed_trades": total_closed
        })
    except Exception as e:
        logger.error(f"❌ [API إحصائيات] خطأ: {e}"); return jsonify({"error": "تعذر جلب الإحصائيات"}), 500

@app.route('/api/data')
def get_all_data():
    """ A single endpoint to fetch all dynamic data for the dashboard """
    if not check_db_connection() or not conn: return jsonify({"error": "Database connection failed"}), 500
    
    data = {}
    try:
        with conn.cursor() as cur:
            # Open signals
            cur.execute("SELECT * FROM signals WHERE status = 'open' ORDER BY id DESC;")
            open_signals = cur.fetchall()
            for s in open_signals:
                with prices_lock: s['current_price'] = current_prices.get(s['symbol'])

            # Closed signals
            cur.execute("SELECT * FROM signals WHERE status != 'open' ORDER BY closed_at DESC LIMIT 20;")
            closed_signals = cur.fetchall()
            for s in closed_signals:
                if s.get('closed_at'): s['closed_at'] = s['closed_at'].isoformat()

            # Recommendations
            cur.execute("SELECT * FROM recommendations WHERE status = 'waiting' ORDER BY generated_at DESC;")
            recommendations = cur.fetchall()
            for r in recommendations:
                if r.get('generated_at'): r['generated_at'] = r['generated_at'].isoformat()
                with prices_lock: r['current_price'] = current_prices.get(r['symbol'])
            
            # Notifications
            with notifications_lock:
                data['notifications'] = list(notifications_cache)

        data['open_signals'] = open_signals
        data['closed_signals'] = closed_signals
        data['recommendations'] = recommendations
        
        return jsonify(data)
    except Exception as e:
        logger.error(f"❌ [API Data] Error fetching all data: {e}")
        return jsonify({"error": "Could not fetch data"}), 500

def run_flask():
    host, port = "0.0.0.0", int(os.environ.get('PORT', 10000))
    log_and_notify("info", f"بدء تشغيل لوحة التحكم على {host}:{port}", "SYSTEM")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        logger.warning("⚠️ [Flask] مكتبة 'waitress' غير موجودة, سيتم استخدام خادم التطوير.")
        app.run(host=host, port=port)

# ---------------------- Program Entry Point ----------------------
def initialize_bot_services():
    global client, validated_symbols_to_scan
    logger.info("🤖 [خدمات البوت] بدء التهيئة في الخلفية...")
    try:
        client = Client(API_KEY, API_SECRET)
        logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
        init_db()
        # load_data_to_cache() is a conceptual function that would load from DB to cache
        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan:
            logger.critical("❌ لا توجد رموز معتمدة للمسح. الحلقات لن تبدأ.")
            return
        Thread(target=run_websocket_manager, daemon=True).start()
        # Thread(target=main_loop, daemon=True).start()
        logger.info("✅ [خدمات البوت] تم بدء جميع خدمات الخلفية بنجاح.")
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حاسم أثناء التهيئة: {e}", "SYSTEM")

if __name__ == "__main__":
    logger.info(f"🚀 بدء تشغيل بوت التداول بمنطق الدخول العكسي - إصدار {BASE_ML_MODEL_NAME}...")
    initialization_thread = Thread(target=initialize_bot_services, daemon=True)
    initialization_thread.start()
    run_flask()
    logger.info("👋 [إيقاف] تم إيقاف تشغيل البوت.")
    os._exit(0)
