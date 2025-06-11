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
MAX_OPEN_TRADES: int = 5
SIGNAL_TIMEFRAME: str = '15m'
# ... (باقي الثوابت كما هي)

# Global variables
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
ticker_data: Dict[str, float] = {}
ml_models_cache: Dict[str, Any] = {}
db_lock = Lock()
bot_status: Dict[str, Any] = {"status": "Initializing", "open_trades": 0, "last_scan": None}

# ---------------------- عميل Binance والإعداد ----------------------
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

app = Flask(__name__)
CORS(app) # السماح بالوصول من نطاقات مختلفة

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
    if isinstance(o, (psycopg2.extras.RealDictRow, dict)):
        return dict(o)

@app.route('/api/status', methods=['GET'])
def get_status():
    """نقطة نهاية للحصول على الحالة العامة للبوت."""
    db_conn = get_db_connection()
    if not db_conn:
        return jsonify({"error": "Database connection failed"}), 500
    try:
        with db_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS count FROM signals WHERE closed_at IS NULL;")
            open_count = (cur.fetchone() or {}).get('count', 0)
            bot_status['open_trades'] = open_count
    except Exception as e:
        logger.error(f"[API] Error getting status: {e}")
        return jsonify({"error": str(e)}), 500
    return jsonify(bot_status)

@app.route('/api/open-signals', methods=['GET'])
def get_open_signals():
    """نقطة نهاية للحصول على جميع الصفقات المفتوحة حالياً."""
    db_conn = get_db_connection()
    if not db_conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with db_conn.cursor() as cur:
            cur.execute("SELECT id, symbol, entry_price, current_target, stop_loss, sent_at, strategy_name FROM signals WHERE closed_at IS NULL ORDER BY sent_at DESC;")
            signals = cur.fetchall()
        return Response(json.dumps(signals, default=default_converter), mimetype='application/json')
    except Exception as e:
        logger.error(f"[API] Error getting open signals: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/closed-signals', methods=['GET'])
def get_closed_signals():
    """نقطة نهاية للحصول على آخر 50 صفقة مغلقة."""
    db_conn = get_db_connection()
    if not db_conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with db_conn.cursor() as cur:
            cur.execute("SELECT id, symbol, entry_price, closing_price, profit_percentage, achieved_target, closed_at FROM signals WHERE closed_at IS NOT NULL ORDER BY closed_at DESC LIMIT 50;")
            signals = cur.fetchall()
        return Response(json.dumps(signals, default=default_converter), mimetype='application/json')
    except Exception as e:
        logger.error(f"[API] Error getting closed signals: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance', methods=['GET'])
def get_performance():
    """نقطة نهاية للحصول على إحصائيات الأداء الأساسية."""
    db_conn = get_db_connection()
    if not db_conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN achieved_target = TRUE THEN 1 ELSE 0 END) as winning_trades,
                    SUM(profit_percentage) as total_profit_pct
                FROM signals WHERE closed_at IS NOT NULL;
            """)
            perf = cur.fetchone()
            if perf and perf.get('total_trades') and perf['total_trades'] > 0:
                perf['win_rate'] = (perf.get('winning_trades', 0) / perf['total_trades']) * 100
            else:
                perf = {'total_trades': 0, 'winning_trades': 0, 'total_profit_pct': 0, 'win_rate': 0}
        return jsonify(perf)
    except Exception as e:
        logger.error(f"[API] Error getting performance: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/general-report', methods=['GET'])
def get_general_report():
    """
    نقطة نهاية جديدة للحصول على تقرير إحصائي عام ومفصل.
    """
    db_conn = get_db_connection()
    if not db_conn:
        return jsonify({"error": "Database connection failed"}), 500

    report = {}
    try:
        with db_conn.cursor() as cur:
            # 1. إحصائيات أساسية
            cur.execute("""
                SELECT
                    COUNT(*) AS total_trades,
                    COALESCE(SUM(CASE WHEN achieved_target = TRUE THEN 1 ELSE 0 END), 0) AS winning_trades,
                    COALESCE(SUM(CASE WHEN achieved_target = FALSE THEN 1 ELSE 0 END), 0) AS losing_trades,
                    COALESCE(SUM(profit_percentage), 0) AS total_profit_pct,
                    COALESCE(AVG(profit_percentage), 0) AS avg_profit_pct
                FROM signals WHERE closed_at IS NOT NULL;
            """)
            base_stats = cur.fetchone()
            report.update(base_stats)

            if report.get('total_trades', 0) > 0:
                report['win_rate'] = (report.get('winning_trades', 0) / report['total_trades']) * 100
            else:
                report['win_rate'] = 0

            # 2. أفضل عملة من حيث متوسط الربح
            cur.execute("""
                SELECT symbol, AVG(profit_percentage) AS avg_profit, COUNT(*) as trade_count
                FROM signals
                WHERE closed_at IS NOT NULL AND profit_percentage IS NOT NULL
                GROUP BY symbol
                ORDER BY avg_profit DESC
                LIMIT 1;
            """)
            best_symbol = cur.fetchone()
            report['best_performing_symbol'] = best_symbol if best_symbol else {}

            # 3. أسوأ عملة من حيث متوسط الربح
            cur.execute("""
                SELECT symbol, AVG(profit_percentage) AS avg_profit, COUNT(*) as trade_count
                FROM signals
                WHERE closed_at IS NOT NULL AND profit_percentage IS NOT NULL
                GROUP BY symbol
                ORDER BY avg_profit ASC
                LIMIT 1;
            """)
            worst_symbol = cur.fetchone()
            report['worst_performing_symbol'] = worst_symbol if worst_symbol else {}

        return Response(json.dumps(report, default=default_converter), mimetype='application/json')

    except Exception as e:
        logger.error(f"❌ [API] خطأ في إنشاء التقرير العام: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate report: {e}"}), 500

def run_api_service():
    """دالة لتشغيل خادم الـ API في خيط منفصل."""
    host = "0.0.0.0"
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"ℹ️ [API] بدء خادم API على http://{host}:{port}...")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        logger.warning("⚠️ [API] 'waitress' غير مثبت. الرجوع إلى خادم تطوير Flask.")
        app.run(host=host, port=port)
    except Exception as e:
        logger.critical(f"❌ [API] فشل بدء خادم API: {e}", exc_info=True)


# ===================================================================
# ======================= منطق البوت الأساسي (بدون تغيير كبير) =================
# ===================================================================

def get_db_connection() -> Optional[psycopg2.extensions.connection]:
    """Gets a thread-safe database connection, reconnecting if necessary."""
    global conn
    with db_lock:
        try:
            if conn is None or conn.closed != 0:
                logger.warning("⚠️ [DB] الاتصال مغلق أو غير موجود. إعادة التهيئة...")
                init_db()
            # Test the connection before returning
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
            return conn
        except (OperationalError, InterfaceError, psycopg2.Error) as e:
            logger.error(f"❌ [DB] فقد الاتصال بقاعدة البيانات ({e}). محاولة إعادة الاتصال...")
            try:
                # Close the faulty connection object before re-initializing
                if conn:
                    conn.close()
                init_db()
                return conn
            except Exception as recon_err:
                logger.error(f"❌ [DB] فشلت محاولة إعادة الاتصال: {recon_err}")
                return None

def init_db(retries: int = 5, delay: int = 5) -> None:
    """Initializes the database connection."""
    global conn
    logger.info("[DB] بدء تهيئة قاعدة البيانات...")
    attempt_conn = None
    for attempt in range(retries):
        try:
            # Use a local variable for the connection attempt
            attempt_conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            logger.info("✅ [DB] تم الاتصال بقاعدة البيانات بنجاح.")
            conn = attempt_conn # Assign to global only on success
            return
        except OperationalError as op_err:
            logger.error(f"❌ [DB] خطأ في التشغيل أثناء الاتصال (المحاولة {attempt + 1}): {op_err}")
        except Exception as e:
            logger.critical(f"❌ [DB] فشل غير متوقع في تهيئة قاعدة البيانات (المحاولة {attempt + 1}): {e}", exc_info=True)
        
        if attempt_conn:
            attempt_conn.close()

        if attempt < retries - 1:
            time.sleep(delay)
        else:
            logger.critical("❌ [DB] فشلت جميع محاولات الاتصال بقاعدة البيانات.")
            raise Exception("Database connection failed after multiple retries.")


# ... (جميع دوال البوت الأخرى مثل load_ml_model_from_db, TradingStrategy, etc. تبقى كما هي)
# ... Placeholder functions to make the script runnable for demonstration
def load_ml_model_from_db(symbol: str) -> Optional[Dict]: return {}
def handle_ticker_message(msg: Dict[str, Any]) -> None: pass
def run_ticker_socket_manager() -> None:
    logger.info("ℹ️ [WebSocket] Ticker manager started (simulation).")
    time.sleep(3600) # Simulate running
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]: return None
class TradingStrategy:
    def __init__(self, symbol: str): self.symbol = symbol
    def generate_buy_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]: return None
def send_telegram_message(text: str, reply_markup: Optional[Dict] = None) -> None: pass
def get_crypto_symbols(filename: str = 'crypto_list.txt') -> List[str]: return ["BTCUSDT", "ETHUSDT"]
def track_signals() -> None:
    logger.info("ℹ️ [Tracker] بدء عملية تتبع الإشارات المفتوحة...")
    while True:
        try:
            db_conn = get_db_connection()
            if db_conn:
                with db_conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) AS count FROM signals WHERE closed_at IS NULL;")
                    open_count = (cur.fetchone() or {}).get('count', 0)
                    bot_status['open_trades'] = open_count
                bot_status['status'] = "Tracking signals"
            else:
                bot_status['status'] = "Tracker DB connection error"
            time.sleep(30)
        except Exception as e:
            logger.error(f"❌ [Tracker] خطأ في دورة تتبع الإشارة: {e}", exc_info=True)
            bot_status['status'] = f"Tracker Error: {e}"
            time.sleep(60)

def main_loop():
    logger.info("🔄 [Main] بدء الحلقة الرئيسية للبوت (وضع المحاكاة)...")
    bot_status['status'] = "Running (Simulated Mode)"
    while True:
        logger.info(f"🔄 [Main] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ...محاكاة دورة الفحص")
        bot_status['last_scan'] = datetime.now().isoformat()
        time.sleep(60 * 5) # Simulate scanning every 5 minutes

# ---------------------- Entry Point ----------------------
if __name__ == "__main__":
    logger.info("🚀 بدء بوت إشارات تداول العملات الرقمية...")
    try:
        init_db()

        api_thread = Thread(target=run_api_service, daemon=True, name="APIThread")
        api_thread.start()

        ws_thread = Thread(target=run_ticker_socket_manager, daemon=True, name="WebSocketThread")
        ws_thread.start()
        logger.info("✅ [Main] تم بدء مؤشر WebSocket.")
        
        tracker_thread = Thread(target=track_signals, daemon=True, name="TrackerThread")
        tracker_thread.start()
        logger.info("✅ [Main] تم بدء متتبع الإشارات.")

        main_loop()

    except Exception as startup_err:
        logger.critical(f"❌ [Main] حدث خطأ فادح أثناء بدء التشغيل: {startup_err}", exc_info=True)
        bot_status['status'] = f"Fatal startup error: {startup_err}"
    finally:
        logger.info("🛑 [Main] إيقاف تشغيل البرنامج...")
        if conn: conn.close()
        logger.info("👋 [Main] تم إيقاف بوت إشارات تداول العملات الرقمية.")
        # os._exit(0) # This can be problematic, letting threads exit gracefully is better.

