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
from datetime import datetime, timedelta, UTC
from decouple import config
from typing import List, Dict, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler
from collections import deque

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v6.1_reversal_entry.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV6.1_ReversalEntry')

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


# ---------------------- دوال قاعدة البيانات ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    # This function is a placeholder. You should have your own implementation.
    global conn
    logger.info("[قاعدة البيانات] بدء تهيئة الاتصال...")
    pass 

def check_db_connection() -> bool:
    # This function is a placeholder. You should have your own implementation.
    return True 

def log_and_notify(level: str, message: str, notification_type: str):
    # This function is a placeholder. You should have your own implementation.
    pass 

def fetch_sr_levels(symbol: str) -> Optional[List[Dict]]:
    # This function is a placeholder. You should have your own implementation.
    return None

# ---------------------- دوال Binance والبيانات ----------------------
def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """
    Reads a list of symbols from a file (e.g., "BTC"), appends "USDT" to them,
    validates them against Binance, and returns a list of tradable USDT symbols.
    """
    logger.info(f"[التحقق] بدء التحقق من الرموز من الملف: {filename}")
    validated_symbols = []
    
    if not os.path.exists(filename):
        logger.error(f"❌ [التحقق] ملف الرموز '{filename}' غير موجود. يرجى إنشاء الملف وإضافة رموز (مثل BTC).")
        return []

    if not client:
        logger.error("❌ [التحقق] عميل Binance غير مهيأ. لا يمكن التحقق من الرموز.")
        return []

    try:
        exchange_info = client.get_exchange_info()
        all_symbols = {s['symbol'] for s in exchange_info['symbols']}
        logger.info(f"✅ [Binance] تم جلب {len(all_symbols)} رمزًا من البورصة.")

        with open(filename, 'r', encoding='utf-8') as f:
            # Read base symbols from file (e.g., BTC, ETH)
            base_symbols_from_file = [line.strip().upper() for line in f if line.strip()]

        logger.info(f"🔎 [التحقق] تم العثور على {len(base_symbols_from_file)} رمزًا أساسيًا في الملف '{filename}'.")

        for base_symbol in base_symbols_from_file:
            # **MODIFIED**: Automatically append 'USDT' to the base symbol
            symbol_to_check = f"{base_symbol}USDT"
            
            if symbol_to_check in all_symbols:
                symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol_to_check), None)
                if symbol_info and symbol_info['status'] == 'TRADING':
                    validated_symbols.append(symbol_to_check)
                else:
                    logger.warning(f"⚠️ [التحقق] تم تخطي الرمز '{symbol_to_check}' لأنه غير متاح للتداول.")
            else:
                logger.warning(f"⚠️ [التحقق] لم يتم العثور على الرمز '{symbol_to_check}' في Binance.")

        logger.info(f"✅ [التحقق] اكتمل التحقق. تم العثور على {len(validated_symbols)} رمزًا صالحًا للتداول.")
        return validated_symbols

    except BinanceAPIException as e:
        logger.error(f"❌ [API Binance] خطأ أثناء التحقق من الرموز: {e}")
    except FileNotFoundError:
        logger.error(f"❌ [التحقق] ملف الرموز '{filename}' غير موجود.")
    except Exception as e:
        logger.error(f"❌ [التحقق] حدث خطأ غير متوقع أثناء التحقق من الرموز: {e}")
        
    return []


def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """
    Fetches historical kline data from Binance and returns it as a pandas DataFrame.
    """
    if not client: return None
    try:
        start_str = (datetime.now(UTC) - timedelta(days=days + 1)).strftime("%Y-%m-%d %H:%M:%S")
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

# --- Other data and ML functions are placeholders ---
# ...

# ---------------------- دوال WebSocket والاستراتيجية ----------------------
def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
    # Placeholder
    pass

def run_websocket_manager() -> None:
    # Placeholder
    pass
    
# ---------------------- Management & Alerting Functions ----------------------
# Placeholders
# ...

# ---------------------- Main Loop ----------------------
def get_btc_trend() -> Dict[str, Any]:
    # Placeholder
    return {}

def main_loop():
    # Placeholder
    pass


# ---------------------- Flask API ----------------------
app = Flask(__name__)
CORS(app)

def get_fear_and_greed_index() -> Dict[str, Any]:
    # Placeholder
    return {}

@app.route('/')
def home():
    try:
        # It is better to have an absolute path or ensure the HTML file is in the correct directory
        return render_template_string(open('index.html', 'r', encoding='utf-8').read())
    except FileNotFoundError:
        return "<h1>ملف لوحة التحكم (index.html) غير موجود.</h1>", 404
    except Exception as e:
        return f"<h1>خطأ في تحميل لوحة التحكم:</h1><p>{e}</p>", 500

@app.route('/api/market_status')
def get_market_status():
    btc_trend_data = get_btc_trend()
    fear_greed_data = get_fear_and_greed_index()
    return jsonify({
        "btc_trend": btc_trend_data,
        "fear_and_greed": fear_greed_data
    })

# Other API endpoints are placeholders
# ...

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
        
        # The core fix is here: calling the now-functional get_validated_symbols
        validated_symbols_to_scan = get_validated_symbols()
        
        if not validated_symbols_to_scan:
            logger.critical("❌ لا توجد رموز معتمدة للمسح. تأكد من وجود ملف 'crypto_list.txt' وأنه يحتوي على رموز صالحة. الحلقات لن تبدأ.")
            return # Stop initialization if no symbols are found
            
        Thread(target=run_websocket_manager, daemon=True).start()
        Thread(target=main_loop, daemon=True).start()
        logger.info(f"✅ [خدمات البوت] تم بدء جميع خدمات الخلفية بنجاح لـ {len(validated_symbols_to_scan)} رمزًا.")
        
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حاسم أثناء التهيئة: {e}", "SYSTEM")

if __name__ == "__main__":
    logger.info(f"🚀 بدء تشغيل بوت التداول بمنطق الدخول العكسي - إصدار {BASE_ML_MODEL_NAME}...")
    initialization_thread = Thread(target=initialize_bot_services, daemon=True)
    initialization_thread.start()
    run_flask()
    logger.info("👋 [إيقاف] تم إيقاف تشغيل البوت.")
    os._exit(0)
