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
from flask import Flask, jsonify, Response
from flask_cors import CORS # استيراد CORS
from threading import Thread, Lock
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Tuple, Any, Union

# ---------------------- إعداد التسجيل ----------------------
# (نفس الكود بدون تغيير)
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
# (نفس الكود بدون تغيير)
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
# (نفس الكود بدون تغيير)
MAX_OPEN_TRADES: int = 5
SIGNAL_TIMEFRAME: str = '15m'
SIGNAL_LOOKBACK_DAYS: int = 5
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V2'
# ... (باقي الثوابت)
RSI_PERIOD: int = 9
ENTRY_ATR_PERIOD: int = 10
SUPERTREND_PERIOD: int = 10
SUPERTREND_MULTIPLIER: float = 3.0
TENKAN_PERIOD: int = 9
KIJUN_PERIOD: int = 26
SENKOU_SPAN_B_PERIOD: int = 52
FIB_SR_LOOKBACK_WINDOW: int = 50
MIN_PROFIT_MARGIN_PCT: float = 1.0
MIN_VOLUME_15M_USDT: float = 50000.0


# Global variables
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
ticker_data: Dict[str, float] = {}
ml_models_cache: Dict[str, Any] = {}
db_lock = Lock()
bot_status: Dict[str, Any] = {"status": "Initializing", "open_trades": 0, "last_scan": None}

# ---------------------- عميل Binance والإعداد ----------------------
# (نفس الكود بدون تغيير)
try:
    logger.info("ℹ️ [Binance] تهيئة عميل Binance...")
    client = Client(API_KEY, API_SECRET)
    client.ping()
    logger.info(f"✅ [Binance] تم تهيئة عميل Binance بنجاح.")
except (BinanceAPIException, BinanceRequestException) as e:
    logger.critical(f"❌ [Binance] خطأ في واجهة برمجة تطبيقات أو طلب Binance: {e}")
    exit(1)

# ===================================================================
# ======================= واجهة برمجة التطبيقات (API) =======================
# ===================================================================

# إنشاء تطبيق Flask
app = Flask(__name__)
# السماح للواجهة الأمامية بالوصول إلى ה-API (مهم جداً)
CORS(app)

def default_converter(o):
    """محول لمساعده تحويل الأنواع غير المعروفة في JSON."""
    if isinstance(o, (datetime, timedelta)):
        return o.__str__()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()

@app.route('/api/status', methods=['GET'])
def get_status():
    """نقطة نهاية للحصول على الحالة العامة للبوت."""
    db_conn = get_db_connection()
    if not db_conn:
        return jsonify({"error": "Database connection failed"}), 500
    with db_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS count FROM signals WHERE closed_at IS NULL;")
        open_count = (cur.fetchone() or {}).get('count', 0)
        bot_status['open_trades'] = open_count
    return jsonify(bot_status)

@app.route('/api/open-signals', methods=['GET'])
def get_open_signals():
    """نقطة نهاية للحصول على جميع الصفقات المفتوحة حالياً."""
    db_conn = get_db_connection()
    if not db_conn:
        return jsonify({"error": "Database connection failed"}), 500
    with db_conn.cursor() as cur:
        cur.execute("SELECT id, symbol, entry_price, current_target, stop_loss, sent_at, strategy_name FROM signals WHERE closed_at IS NULL ORDER BY sent_at DESC;")
        signals = cur.fetchall()
    return Response(json.dumps(signals, default=default_converter), mimetype='application/json')

@app.route('/api/closed-signals', methods=['GET'])
def get_closed_signals():
    """نقطة نهاية للحصول على آخر 50 صفقة مغلقة."""
    db_conn = get_db_connection()
    if not db_conn:
        return jsonify({"error": "Database connection failed"}), 500
    with db_conn.cursor() as cur:
        cur.execute("SELECT id, symbol, entry_price, closing_price, profit_percentage, achieved_target, closed_at FROM signals WHERE closed_at IS NOT NULL ORDER BY closed_at DESC LIMIT 50;")
        signals = cur.fetchall()
    return Response(json.dumps(signals, default=default_converter), mimetype='application/json')

@app.route('/api/performance', methods=['GET'])
def get_performance():
    """نقطة نهاية للحصول على إحصائيات الأداء."""
    db_conn = get_db_connection()
    if not db_conn:
        return jsonify({"error": "Database connection failed"}), 500
    with db_conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN achieved_target = TRUE THEN 1 ELSE 0 END) as winning_trades,
                SUM(profit_percentage) as total_profit_pct
            FROM signals WHERE closed_at IS NOT NULL;
        """)
        perf = cur.fetchone()
        if perf and perf['total_trades'] > 0:
            perf['win_rate'] = (perf['winning_trades'] / perf['total_trades']) * 100
        else:
            perf = {'total_trades': 0, 'winning_trades': 0, 'total_profit_pct': 0, 'win_rate': 0}
    return jsonify(perf)

def run_api_service():
    """دالة لتشغيل خادم الـ API في خيط منفصل."""
    host = "0.0.0.0"
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"ℹ️ [API] بدء خادم API على http://{host}:{port}...")
    # استخدام Waitress لخادم انتاجي بدلاً من خادم التطوير الخاص بـ Flask
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=4)
    except ImportError:
        logger.warning("⚠️ [API] 'waitress' غير مثبت. الرجوع إلى خادم تطوير Flask.")
        app.run(host=host, port=port)
    except Exception as e:
        logger.critical(f"❌ [API] فشل بدء خادم API: {e}", exc_info=True)


# ===================================================================
# ======================= منطق البوت الأساسي (بدون تغيير كبير) =================
# ===================================================================

# (جميع دوال البوت الأخرى مثل init_db, load_ml_model_from_db, TradingStrategy, etc. تبقى كما هي)
# ...
# ... (انسخ والصق جميع دوال البوت الأخرى من ملف c4.py الأصلي هنا)
# ...
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
def load_ml_model_from_db(symbol: str) -> Optional[Dict]: return {} # Placeholder
def handle_ticker_message(msg: Dict[str, Any]) -> None: pass # Placeholder
def run_ticker_socket_manager() -> None: pass # Placeholder
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]: return None # Placeholder
class TradingStrategy:
    def __init__(self, symbol: str): self.symbol = symbol
    def generate_buy_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]: return None # Placeholder
def send_telegram_message(text: str, reply_markup: Optional[Dict] = None) -> None: pass # Placeholder
def get_crypto_symbols(filename: str = 'crypto_list.txt') -> List[str]: return ["BTCUSDT", "ETHUSDT"] # Placeholder
def track_signals() -> None:
    """Tracks open signals, updating bot_status."""
    logger.info("ℹ️ [Tracker] بدء عملية تتبع الإشارات المفتوحة...")
    while True:
        try:
            bot_status['status'] = "Tracking signals"
            time.sleep(10) # Placeholder for tracking logic
        except Exception as e:
            logger.error(f"❌ [Tracker] خطأ في دورة تتبع الإشارة: {e}", exc_info=True)
            bot_status['status'] = f"Tracker Error: {e}"
            time.sleep(30)


def main_loop():
    """Main loop to scan for new trading signals."""
    symbols_to_scan = get_crypto_symbols()
    if not symbols_to_scan:
        logger.critical("❌ [Main] لم يتم تحميل أي رموز صالحة. لا يمكن المتابعة.")
        bot_status['status'] = "Error: No symbols loaded"
        return

    logger.info(f"✅ [Main] تم تحميل {len(symbols_to_scan)} رمزًا صالحًا للمسح.")

    while True:
        try:
            bot_status['status'] = "Scanning market..."
            bot_status['last_scan'] = datetime.now().isoformat()
            logger.info(f"🔄 [Main] بدء دورة مسح السوق - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            # ... (باقي منطق الحلقة الرئيسية)
            logger.info(f"⏳ [Main] انتظار 15 دقيقة للدورة التالية...")
            time.sleep(60 * 15)
        except Exception as main_err:
            logger.error(f"❌ [Main] خطأ غير متوقع في الحلقة الرئيسية: {main_err}", exc_info=True)
            bot_status['status'] = f"Main loop error: {main_err}"
            time.sleep(120)

# ---------------------- Entry Point ----------------------
if __name__ == "__main__":
    logger.info("🚀 بدء بوت إشارات تداول العملات الرقمية...")
    try:
        init_db()

        # تشغيل خادم الـ API في خيط منفصل
        api_thread = Thread(target=run_api_service, daemon=True, name="APIThread")
        api_thread.start()

        # Start WebSocket in a background thread
        ws_thread = Thread(target=run_ticker_socket_manager, daemon=True, name="WebSocketThread")
        ws_thread.start()
        logger.info("✅ [Main] تم بدء مؤشر WebSocket. انتظار 5 ثوانٍ لتهيئة البيانات...")
        time.sleep(5)

        # Start Signal Tracker in a background thread
        tracker_thread = Thread(target=track_signals, daemon=True, name="TrackerThread")
        tracker_thread.start()
        logger.info("✅ [Main] تم بدء متتبع الإشارات.")

        # تشغيل الحلقة الرئيسية للبوت في الخيط الرئيسي
        main_loop()

    except Exception as startup_err:
        logger.critical(f"❌ [Main] حدث خطأ فادح أثناء بدء التشغيل: {startup_err}", exc_info=True)
        bot_status['status'] = f"Fatal startup error: {startup_err}"
    finally:
        logger.info("🛑 [Main] إيقاف تشغيل البرنامج...")
        if conn: conn.close()
        logger.info("👋 [Main] تم إيقاف بوت إشارات تداول العملات الرقمية.")
        os._exit(0)
