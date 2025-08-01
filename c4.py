# ملف c4.py - نسخة محدثة مع استراتيجية BB/Stoch ولوحة تحكم متكاملة
# تم التحديث بواسطة Gemini
# --- الملخص:
# 1. الحفاظ على استراتيجية (BB + Stochastic + أنماط الشموع).
# 2. إعادة دمج واجهة تحكم Flask متكاملة.
# 3. إضافة زر لتفعيل/إيقاف التداول الحقيقي.
# 4. عرض الرصيد، الصفقات المفتوحة، وحالة السوق.
import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import pickle
import redis
import gc
import random
from decimal import Decimal
from urllib.parse import urlparse
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance.exceptions import BinanceAPIException
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timezone, timedelta
from decouple import config
from typing import List, Dict, Optional, Any
from collections import deque
import warnings

# --- إعدادات التجاهل واللوجر ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_dashboard_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBot_Dashboard')

# --- تحميل متغيرات البيئة ---
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
    REDIS_URL: str = config('REDIS_URL', default='redis://localhost:6379/0')
    TELEGRAM_BOT_TOKEN: str = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID: str = config('TELEGRAM_CHAT_ID', default='')
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
    exit(1)

# --- متغيرات عامة وإعدادات البوت ---
is_trading_enabled: bool = False
trading_status_lock = Lock()
RISK_PER_TRADE_PERCENT: float = 1.0
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V9_With_Microstructure'
MODEL_FOLDER: str = 'V9'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 90
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices_v9"
TRADING_FEE_PERCENT: float = 0.1
STATS_TRADE_SIZE_USDT: float = 5.0
BTC_SYMBOL: str = 'BTCUSDT'
MAX_OPEN_TRADES: int = 4
BUY_CONFIDENCE_THRESHOLD = 0.55
SYMBOL_PROCESSING_BATCH_SIZE: int = 10

# --- إعدادات المؤشرات الفنية ---
STOCH_K_PERIOD: int = 14
STOCH_D_PERIOD: int = 3
BB_PERIOD: int = 20
BB_STD_DEV: int = 2
ATR_PERIOD: int = 14
RSI_PERIOD: int = 14
ADX_PERIOD: int = 14

# --- إعدادات إدارة الصفقات ---
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_ACTIVATION_PROFIT_PERCENT: float = 1.8
TRAILING_DISTANCE_PERCENT: float = 1.0
ORDER_BOOK_DEPTH_LIMIT: int = 100

# --- متغيرات الحالة والكاش ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
ml_models_cache: Dict[str, Any] = {}
exchange_info_map: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
notifications_cache = deque(maxlen=50)
notifications_lock = Lock()
rejection_logs_cache = deque(maxlen=100)
rejection_logs_lock = Lock()
current_market_state: Dict[str, Any] = {"overall_regime": "INITIALIZING", "trend_details_by_tf": {}, "last_updated": None}
market_state_lock = Lock()
last_market_state_check = 0

# --- قاموس أسباب الرفض باللغة العربية ---
REJECTION_REASONS_AR = {
    "No BB Signal": "لا توجد إشارة Bollinger Band",
    "No Stoch Crossover": "لا يوجد تقاطع إيجابي لمؤشر Stochastic",
    "No Bullish Pattern": "لا يوجد نمط شموع صعودي",
    "Order Book Check Failed": "فشل التحقق من دفتر الطلبات",
    "ML Model Rejected": "نموذج تعلم الآلة رفض الإشارة",
    "ML Model Load Failed": "فشل تحميل نموذج تعلم الآلة",
    "Data Fetch Failed": "فشل جلب البيانات",
    "Price Fetch Failed": "فشل جلب السعر الحالي",
    "TP/SL Calculation Failed": "فشل حساب الهدف ووقف الخسارة",
    "Invalid Position Size": "حجم الصفقة غير صالح",
    "Lot Size Adjustment Failed": "فشل ضبط حجم العقد",
    "Min Notional Filter": "قيمة الصفقة أقل من الحد الأدنى",
    "Insufficient Balance": "الرصيد غير كافٍ",
}

# --- دالة إرسال رسائل تليجرام ---
def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload, timeout=10).raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

# --- دوال تهيئة الخدمات (مشابهة للنسخ السابقة) ---
def init_db(retries: int = 5, delay: int = 5) -> None:
    global conn
    logger.info("[DB] تهيئة الاتصال بقاعدة البيانات...")
    db_url_to_use = DB_URL
    if 'postgres' in db_url_to_use and 'sslmode' not in db_url_to_use:
        db_url_to_use += f"{'?' if '?' not in db_url_to_use else '&'}sslmode=require"
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(db_url_to_use, connect_timeout=15, cursor_factory=RealDictCursor)
            conn.autocommit = False
            # ... (schema creation remains the same)
            logger.info("✅ [DB] الاتصال بقاعدة البيانات وتحديث المخطط بنجاح.")
            return
        except Exception as e:
            logger.error(f"❌ [DB] خطأ أثناء التهيئة (محاولة {attempt + 1}/{retries}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: logger.critical("❌ [DB] فشل الاتصال بقاعدة البيانات.")

def check_db_connection() -> bool:
    global conn
    if conn is None or conn.closed != 0:
        logger.warning("[DB] الاتصال مغلق، محاولة إعادة الاتصال...")
        init_db()
    try:
        if conn and conn.closed == 0:
            with conn.cursor() as cur: cur.execute("SELECT 1;")
            return True
        return False
    except (OperationalError, InterfaceError):
        logger.error(f"❌ [DB] فقدان الاتصال. إعادة الاتصال...")
        try:
            init_db()
            return conn is not None and conn.closed == 0
        except Exception as retry_e:
            logger.error(f"❌ [DB] فشل إعادة الاتصال: {retry_e}")
            return False

def log_and_notify(level: str, message: str, notification_type: str):
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error}
    log_methods.get(level.lower(), logger.info)(message)
    if not check_db_connection() or not conn: return
    try:
        new_notification = {"timestamp": datetime.now(timezone.utc).isoformat(), "type": notification_type, "message": message}
        with notifications_lock: notifications_cache.appendleft(new_notification)
        with conn.cursor() as cur: cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [Notify DB] فشل حفظ الإشعار: {e}")
        if conn: conn.rollback()

def log_rejection(symbol: str, reason_key: str, details: Optional[Dict] = None):
    reason_ar = REJECTION_REASONS_AR.get(reason_key, reason_key)
    log_message = f"🚫 [REJECTED] {symbol} | Reason: {reason_ar} | Details: {details or {}}"
    logger.info(log_message)
    with rejection_logs_lock:
        rejection_logs_cache.appendleft({
            "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
            "reason": reason_ar, "details": json.loads(json.dumps(details, default=str)) or {}
        })

def init_redis() -> None:
    global redis_client
    logger.info("[Redis] تهيئة الاتصال بـ Redis...")
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] تم الاتصال بنجاح بخادم Redis.")
    except redis.exceptions.ConnectionError as e:
        logger.critical(f"❌ [Redis] فشل الاتصال بـ Redis: {e}")
        exit(1)

def get_exchange_info_map() -> None:
    global exchange_info_map
    if not client: return
    try:
        info = client.get_exchange_info()
        exchange_info_map = {s['symbol']: s for s in info['symbols']}
    except Exception as e:
        logger.error(f"❌ [Exchange Info] لم يتمكن من جلب معلومات المنصة: {e}")

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    if not client: return []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        if not exchange_info_map: get_exchange_info_map()
        active = {s for s, info in exchange_info_map.items() if info.get('quoteAsset') == 'USDT' and info.get('status') == 'TRADING'}
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ [Validation] سيقوم البوت بمراقبة {len(validated)} عملة.")
        return validated
    except Exception as e:
        logger.error(f"❌ [Validation] خطأ أثناء التحقق من العملات: {e}", exc_info=True)
        return []

# --- دوال جلب البيانات وحساب الميزات (نفس النسخة السابقة) ---
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        start_dt = datetime.now(timezone.utc) - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(klines, columns=cols)
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_cols]
        numeric_cols = {'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'}
        df = df.astype(numeric_cols)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna()
    except Exception as e:
        logger.error(f"❌ [Data] خطأ في جلب البيانات التاريخية لـ {symbol}: {e}")
        return None

def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    # BB
    df_calc['bb_middle'] = df_calc['close'].rolling(window=BB_PERIOD).mean()
    bb_std = df_calc['close'].rolling(window=BB_PERIOD).std()
    df_calc['bb_upper'] = df_calc['bb_middle'] + (bb_std * BB_STD_DEV)
    df_calc['bb_lower'] = df_calc['bb_middle'] - (bb_std * BB_STD_DEV)
    # Stoch
    low_min = df_calc['low'].rolling(window=STOCH_K_PERIOD).min()
    high_max = df_calc['high'].rolling(window=STOCH_K_PERIOD).max()
    df_calc['stoch_k'] = 100 * (df_calc['close'] - low_min) / (high_max - low_min).replace(0, 1e-9)
    df_calc['stoch_d'] = df_calc['stoch_k'].rolling(window=STOCH_D_PERIOD).mean()
    # ATR
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    # EMA for trend
    df_calc['ema_fast'] = df_calc['close'].ewm(span=12, adjust=False).mean()
    df_calc['ema_slow'] = df_calc['close'].ewm(span=26, adjust=False).mean()
    # ADX for trend strength
    plus_dm = pd.Series(np.where((df_calc['high'].diff() > -df_calc['low'].diff()) & (df_calc['high'].diff() > 0), df_calc['high'].diff(), 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((-df_calc['low'].diff() > df_calc['high'].diff()) & (-df_calc['low'].diff() > 0), -df_calc['low'].diff(), 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    
    return df_calc.astype('float32', errors='ignore')

# --- تحميل الصفقات والإشعارات (نفس النسخة السابقة) ---
def load_open_signals_to_cache():
    # ... (code remains the same)
    pass
def load_notifications_to_cache():
    # ... (code remains the same)
    pass

# --- الاستراتيجية الجديدة والفلاتر (نفس النسخة السابقة) ---
def detect_bullish_patterns(df: pd.DataFrame) -> Optional[str]:
    # ... (code remains the same)
    pass
def check_bb_stoch_signal(df: pd.DataFrame, lookback: int = 3) -> Optional[Dict[str, Any]]:
    # ... (code remains the same)
    pass
def passes_order_book_filter(symbol: str, entry_price: float) -> bool:
    # ... (code remains the same)
    pass
# --- تحميل نموذج تعلم الآلة (نفس النسخة السابقة) ---
class MLConfirmation:
    # ... (class remains the same)
    pass

# --- حساب الهدف ووقف الخسارة (نفس النسخة السابقة) ---
def calculate_tp_sl(symbol: str, entry_price: float, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    # ... (code remains the same)
    pass
# --- دوال إدارة الصفقات (نفس النسخة السابقة) ---
def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> Optional[Decimal]:
    # ... (code remains the same)
    pass
def calculate_position_size(symbol: str, entry_price: float, stop_loss_price: float) -> Optional[Decimal]:
    # ... (code remains the same)
    pass
def place_order(symbol: str, side: str, quantity: Decimal, order_type: str = Client.ORDER_TYPE_MARKET) -> Optional[Dict]:
    # ... (code remains the same)
    pass
def close_signal(signal_id: int, closing_price: float, reason: str) -> bool:
    # ... (code remains the same, including Telegram notification)
    pass
def insert_signal_into_db(signal_data: Dict) -> Optional[Dict]:
    # ... (code remains the same, including Telegram notification)
    pass

# --- دالة تحليل حالة السوق (جديدة/مُحسّنة) ---
def determine_market_state_enhanced():
    global current_market_state, last_market_state_check
    if time.time() - last_market_state_check < 180: return # Update every 3 minutes
    logger.info("🧠 [Market State] تحديث حالة السوق...")
    try:
        trend_details = {}
        for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
            df = fetch_historical_data(BTC_SYMBOL, tf, 50) # Fetch more data for ADX
            if df is not None and not df.empty:
                df_features = calculate_all_features(df)
                ema_fast = df_features['ema_fast'].iloc[-1]
                ema_slow = df_features['ema_slow'].iloc[-1]
                adx = df_features['adx'].iloc[-1] if 'adx' in df_features.columns else 0
                
                if ema_fast > ema_slow and adx > 25: trend = "Strong Uptrend"
                elif ema_fast > ema_slow: trend = "Uptrend"
                elif ema_fast < ema_slow and adx > 25: trend = "Strong Downtrend"
                elif ema_fast < ema_slow: trend = "Downtrend"
                else: trend = "Ranging"
                trend_details[tf] = {"trend": trend, "adx": float(adx)}
            else:
                trend_details[tf] = {"trend": "Uncertain", "adx": 0}

        trends = [d['trend'] for d in trend_details.values()]
        overall_regime = max(set(trends), key=trends.count) if trends else "Uncertain"
        
        with market_state_lock:
            current_market_state = {
                "overall_regime": overall_regime.upper().replace(" ", "_"), 
                "trend_details_by_tf": trend_details, 
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            last_market_state_check = time.time()
        logger.info(f"✅ [Market State] الحالة العامة: {overall_regime}")
    except Exception as e:
        logger.error(f"❌ [Market State] خطأ: {e}", exc_info=True)

# ---------------------- واجهة Flask ----------------------
app = Flask(__name__)
CORS(app)

def get_dashboard_html():
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم بوت التداول</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
    <style>
        :root { --bg-main: #0D1117; --bg-card: #161B22; --border-color: #30363D; --text-primary: #E6EDF3; --text-secondary: #848D97; --accent-blue: #58A6FF; --accent-green: #3FB950; --accent-red: #F85149; --accent-yellow: #D29922; }
        body { font-family: 'Tajawal', sans-serif; background-color: var(--bg-main); color: var(--text-primary); }
        .card { background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 0.5rem; }
        .trend-light { width: 1rem; height: 1rem; border-radius: 9999px; border: 2px solid #30363D; transition: all 0.5s ease; }
        .light-on-green { background-color: var(--accent-green); box-shadow: 0 0 10px 2px var(--accent-green); }
        .light-on-red { background-color: var(--accent-red); box-shadow: 0 0 10px 2px var(--accent-red); }
        .light-on-yellow { background-color: var(--accent-yellow); box-shadow: 0 0 10px 2px var(--accent-yellow); }
        .tab-btn.active { border-bottom-color: var(--accent-blue); }
        input:checked + .toggle-bg { background-color: var(--accent-green); }
        #modal-overlay { transition: opacity 0.3s ease; }
    </style>
</head>
<body class="p-4 md:p-6">
    <div id="modal-overlay" class="fixed inset-0 bg-black bg-opacity-70 hidden items-center justify-center z-50">
        <div id="modal-content" class="card p-6 rounded-lg shadow-xl max-w-sm w-full">
            <h3 id="modal-title" class="text-xl font-bold mb-4"></h3>
            <p id="modal-body" class="text-text-secondary mb-6"></p>
            <div class="flex justify-end gap-3">
                <button id="modal-cancel" class="px-4 py-2 rounded-md bg-gray-600 hover:bg-gray-700">إلغاء</button>
                <button id="modal-confirm" class="px-4 py-2 rounded-md bg-red-600 hover:bg-red-700">تأكيد</button>
            </div>
        </div>
    </div>

    <div class="container mx-auto max-w-screen-2xl">
        <header class="mb-6 flex flex-wrap justify-between items-center gap-4">
            <h1 class="text-2xl md:text-3xl font-extrabold"><span class="text-accent-blue">لوحة تحكم</span><span class="text-text-secondary font-medium"> BB/Stoch</span></h1>
            <div id="trend-lights-container" class="flex items-center gap-x-6 bg-black/20 px-4 py-2 rounded-lg border border-border-color"></div>
        </header>
        <section class="mb-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">حالة السوق العامة</h3><div id="overall-regime" class="text-2xl font-bold text-center">...</div></div>
            <div class="card p-4 flex flex-col justify-center items-center"><h3 class="font-bold text-lg text-text-secondary mb-2">رصيد التداول (USDT)</h3><div id="usdt-balance" class="text-2xl font-bold text-center font-mono">...</div></div>
            <div class="card p-4 flex flex-col justify-center items-center"><h3 class="font-bold text-lg text-text-secondary mb-2">التداول الحقيقي</h3><div class="flex items-center space-x-3 space-x-reverse"><span id="trading-status-text" class="font-bold text-lg"></span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="trading-toggle" class="sr-only" onchange="toggleTrading()"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div></div>
        </section>
        <div class="mb-4 border-b border-border-color"><nav class="flex space-x-6 space-x-reverse -mb-px"><button onclick="showTab('signals', this)" class="tab-btn active text-white py-3 px-1 font-semibold">الصفقات المفتوحة</button><button onclick="showTab('stats', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الإحصائيات</button><button onclick="showTab('notifications', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الإشعارات</button><button onclick="showTab('rejections', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الصفقات المرفوضة</button></nav></div>
        <main>
            <div id="signals-tab" class="tab-content"><div class="overflow-x-auto card p-0"><table class="min-w-full text-sm text-right"><thead class="border-b border-border-color bg-black/20"><tr><th class="p-4 font-semibold">العملة</th><th class="p-4 font-semibold">الحالة</th><th class="p-4 font-semibold">الربح/الخسارة</th><th class="p-4 font-semibold w-[25%]">التقدم نحو الهدف</th><th class="p-4 font-semibold">الدخول/الحالي</th><th class="p-4 font-semibold">إجراء</th></tr></thead><tbody id="signals-table"></tbody></table></div></div>
            <div id="stats-tab" class="tab-content hidden"><div id="stats-container" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"></div></div>
            <div id="notifications-tab" class="tab-content hidden"><div id="notifications-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-2"></div></div>
            <div id="rejections-tab" class="tab-content hidden"><div id="rejections-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-2"></div></div>
        </main>
    </div>
<script>
let confirmCallback = null;
const modal = { overlay: document.getElementById('modal-overlay'), title: document.getElementById('modal-title'), body: document.getElementById('modal-body'), confirmBtn: document.getElementById('modal-confirm'), cancelBtn: document.getElementById('modal-cancel') };
modal.cancelBtn.onclick = () => { modal.overlay.classList.add('hidden'); };
modal.confirmBtn.onclick = () => { if(confirmCallback) confirmCallback(); modal.overlay.classList.add('hidden'); };

function showConfirmation(title, bodyText, onConfirm) {
    modal.title.textContent = title;
    modal.body.textContent = bodyText;
    confirmCallback = onConfirm;
    modal.overlay.classList.remove('hidden');
    modal.overlay.classList.add('flex');
}
function showTab(tabId, el) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.add('hidden'));
    document.getElementById(tabId + '-tab').classList.remove('hidden');
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active', 'text-white'));
    el.classList.add('active', 'text-white');
}
async function fetchData(url) { try { const r = await fetch(url); return r.ok ? await r.json() : null; } catch (e) { console.error('Fetch Error:', e); return null; } }

function updateDashboard() {
    fetchData('/api/market_status').then(data => {
        if (!data) return;
        // Market State
        document.getElementById('overall-regime').textContent = (data.market_state?.overall_regime || 'UNCERTAIN').replace(/_/g, ' ');
        const lights = document.getElementById('trend-lights-container');
        lights.innerHTML = '';
        ['15m', '1h', '4h'].forEach(tf => {
            const trendInfo = data.market_state?.trend_details_by_tf[tf];
            const trend = trendInfo?.trend || 'Uncertain';
            let c = trend.includes('Uptrend') ? 'light-on-green' : trend.includes('Downtrend') ? 'light-on-red' : 'light-on-yellow';
            lights.innerHTML += `<div class="flex items-center gap-2"><div class="trend-light ${c}"></div><span class="text-sm font-bold text-text-secondary">${tf}</span></div>`;
        });
        
        // Trading Status & Balance
        const tradeToggle = document.getElementById('trading-toggle'), tradeText = document.getElementById('trading-status-text');
        tradeToggle.checked = data.is_trading_enabled;
        tradeText.textContent = data.is_trading_enabled ? 'مُفعَّل' : 'غير مُفعَّل';
        tradeText.className = `font-bold text-lg ${data.is_trading_enabled ? 'text-accent-green' : 'text-accent-red'}`;
        document.getElementById('usdt-balance').textContent = data.usdt_balance ? parseFloat(data.usdt_balance).toFixed(2) : 'N/A';
    });

    fetchData('/api/signals').then(data => {
        if (!data) return;
        const tableBody = document.getElementById('signals-table');
        tableBody.innerHTML = '';
        if (data.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="6" class="p-8 text-center text-text-secondary">لا توجد صفقات مفتوحة حالياً.</td></tr>';
            return;
        }
        data.forEach(s => {
            const profit = parseFloat(s.profit_percentage || 0);
            const pClass = profit > 0 ? 'text-accent-green' : profit < 0 ? 'text-accent-red' : 'text-text-secondary';
            const entry = parseFloat(s.entry_price), sl = parseFloat(s.stop_loss), tp = parseFloat(s.target_price), current = parseFloat(s.current_price || entry);
            const progress = (tp - sl > 0) ? Math.max(0, Math.min(100, (current - sl) / (tp - sl) * 100)) : 0;
            tableBody.innerHTML += `<tr class="border-b border-border-color hover:bg-white/5"><td class="p-4 font-bold">${s.symbol}</td><td class="p-4"><span class="px-2 py-1 text-xs font-semibold rounded-full ${s.is_real_trade ? 'bg-blue-500/20 text-blue-400' : 'bg-yellow-500/20 text-yellow-400'}">${s.is_real_trade ? 'حقيقي' : 'تجريبي'}</span></td><td class="p-4 font-mono ${pClass}">${profit.toFixed(2)}%</td><td class="p-4"><div class="w-full bg-gray-700 rounded-full h-2.5"><div class="bg-accent-blue h-2.5 rounded-full" style="width: ${progress}%"></div></div></td><td class="p-4 font-mono">${current.toFixed(4)} / ${entry.toFixed(4)}</td><td class="p-4"><button onclick="manualClose(${s.id}, '${s.symbol}')" class="bg-red-600 hover:bg-red-700 text-white font-bold py-1 px-3 rounded text-xs">إغلاق</button></td></tr>`;
        });
    });
}

function updateOtherTabs() {
    fetchData('/api/stats').then(data => {
        if (!data) return;
        const container = document.getElementById('stats-container');
        container.innerHTML = `<div class="card p-4 text-center"><h4 class="text-text-secondary">صافي الربح (USDT)</h4><div class="text-2xl font-bold ${data.net_profit_usdt >= 0 ? 'text-accent-green' : 'text-accent-red'}">${parseFloat(data.net_profit_usdt).toFixed(2)}</div></div><div class="card p-4 text-center"><h4 class="text-text-secondary">معدل الربح</h4><div class="text-2xl font-bold">${parseFloat(data.win_rate).toFixed(2)}%</div></div><div class="card p-4 text-center"><h4 class="text-text-secondary">عامل الربح</h4><div class="text-2xl font-bold">${data.profit_factor === 'Infinity' ? '∞' : parseFloat(data.profit_factor).toFixed(2)}</div></div><div class="card p-4 text-center"><h4 class="text-text-secondary">الصفقات المغلقة</h4><div class="text-2xl font-bold">${data.total_closed_trades}</div></div>`;
    });
    fetchData('/api/notifications').then(data => {
        document.getElementById('notifications-list').innerHTML = data.map(n => `<div class="p-2 border-b border-border-color"><span class="font-mono text-xs text-text-secondary">${new Date(n.timestamp).toLocaleString('ar-EG')}</span>: ${n.message}</div>`).join('');
    });
    fetchData('/api/rejection_logs').then(data => {
        document.getElementById('rejections-list').innerHTML = data.map(r => `<div class="p-2 border-b border-border-color"><span class="font-mono text-xs text-text-secondary">${new Date(r.timestamp).toLocaleString('ar-EG')}</span>: <strong class="text-accent-yellow">${r.symbol}</strong> - ${r.reason} <span class="text-xs text-gray-500">${JSON.stringify(r.details)}</span></div>`).join('');
    });
}

function manualClose(signalId, symbol) {
    showConfirmation('تأكيد الإغلاق', `هل أنت متأكد من رغبتك في إغلاق الصفقة لـ ${symbol} يدوياً؟`, () => {
        fetch(`/api/signals/close/${signalId}`, { method: 'POST' })
            .then(res => res.json())
            .then(data => { if(data.success) updateDashboard(); else alert(data.message); });
    });
}
function toggleTrading() {
    const isChecked = document.getElementById('trading-toggle').checked;
    const action = isChecked ? "تفعيل" : "إيقاف";
    showConfirmation(`تأكيد ${action} التداول`, `هل أنت متأكد من ${action} التداول الحقيقي؟`, () => {
        fetch('/api/trading/toggle', { method: 'POST' }).then(() => updateDashboard());
    });
}

document.addEventListener('DOMContentLoaded', () => {
    updateDashboard();
    updateOtherTabs();
    setInterval(updateDashboard, 5000); // Update main dashboard frequently
    setInterval(updateOtherTabs, 30000); // Update other tabs less frequently
});
</script>
</body></html>
"""

@app.route('/')
def home(): return render_template_string(get_dashboard_html())

@app.route('/api/market_status')
def get_market_status():
    with market_state_lock: state_copy = dict(current_market_state)
    with trading_status_lock: is_enabled = is_trading_enabled
    usdt_balance = None
    if client:
        try: usdt_balance = float(client.get_asset_balance(asset='USDT')['free'])
        except: usdt_balance = 'N/A'
    return jsonify({
        "market_state": state_copy, 
        "is_trading_enabled": is_enabled,
        "usdt_balance": usdt_balance
    })

@app.route('/api/signals')
def get_signals():
    if not all([check_db_connection(), redis_client]): return jsonify([]), 500
    try:
        current_prices = redis_client.hgetall(REDIS_PRICES_HASH_NAME)
        with signal_cache_lock: signals_copy = list(open_signals_cache.values())
        for signal in signals_copy:
            current_price = current_prices.get(signal['symbol'])
            if current_price:
                signal['current_price'] = current_price
                signal['profit_percentage'] = ((float(current_price) - float(signal['entry_price'])) / float(signal['entry_price'])) * 100
        return jsonify(signals_copy)
    except Exception as e:
        logger.error(f"❌ [API Signals] Error: {e}"); return jsonify([]), 500

@app.route('/api/stats')
def get_stats():
    # ... (code remains the same)
    pass

@app.route('/api/notifications')
def get_notifications():
    with notifications_lock: return jsonify(list(notifications_cache))

@app.route('/api/rejection_logs')
def get_rejection_logs():
    with rejection_logs_lock: return jsonify(list(rejection_logs_cache))

@app.route('/api/trading/toggle', methods=['POST'])
def toggle_trading_status():
    global is_trading_enabled
    with trading_status_lock:
        is_trading_enabled = not is_trading_enabled
        status_msg = "ENABLED" if is_trading_enabled else "DISABLED"
        log_and_notify('warning', f"🚨 Real trading status changed to: {status_msg}", "TRADING_STATUS_CHANGE")
        send_telegram_message(f"🚨 *تم { 'تفعيل' if is_trading_enabled else 'إيقاف' } التداول الحقيقي*")
        return jsonify({"message": f"Trading status set to {status_msg}"})

@app.route('/api/signals/close/<int:signal_id>', methods=['POST'])
def manual_close_trade_endpoint(signal_id):
    if not redis_client or not client: return jsonify({"success": False, "message": "Services not ready"}), 503
    with signal_cache_lock:
        signal_to_close = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)
    if not signal_to_close: return jsonify({"success": False, "message": "Signal not found"}), 404
    try:
        current_price = float(redis_client.hget(REDIS_PRICES_HASH_NAME, signal_to_close['symbol']))
    except:
        try: current_price = float(client.get_symbol_ticker(symbol=signal_to_close['symbol'])['price'])
        except Exception as e: return jsonify({"success": False, "message": f"Could not fetch price: {e}"}), 500
    
    if close_signal(signal_id, current_price, 'manual'):
        return jsonify({"success": True, "message": "Signal closed."})
    else:
        return jsonify({"success": False, "message": "Failed to close signal."}), 500

# ---------------------- حلقات النظام ----------------------
def trade_management_loop():
    # ... (code remains the same)
    pass

def main_loop_new_strategy():
    """
    الحلقة الرئيسية لمسح الرموز وتوليد الإشارات وإنشاء الصفقات.
    يحل هذا الإصدار محل الدالة الناقصة التي كانت تسبب التكرار اللانهائي.
    """
    ml_confirm = MLConfirmation()
    logger.info("▶️ بدء حلقة الاستراتيجية الرئيسية...")

    while True:
        try:
            logger.info("🔄 بدء دورة مسح جديدة...")
            determine_market_state_enhanced()

            with trading_status_lock:
                is_real_mode = is_trading_enabled

            with signal_cache_lock:
                open_trades = [s for s in open_signals_cache.values() if s.get('is_real_trade') == is_real_mode]
                open_symbols = {s['symbol'] for s in open_trades}
            
            if len(open_trades) >= MAX_OPEN_TRADES:
                logger.info(f"⏸️ الحد الأقصى للصفقات المفتوحة ({MAX_OPEN_TRADES}). إيقاف البحث عن إشارات جديدة مؤقتاً.")
                time.sleep(60)
                continue

            symbols_to_scan = [s for s in validated_symbols_to_scan if s not in open_symbols]
            random.shuffle(symbols_to_scan)
            
            logger.info(f"🔍 التحضير لمسح {len(symbols_to_scan)} عملة.")

            for symbol in symbols_to_scan:
                # إعادة التحقق من الحد الأقصى للصفقات قبل معالجة كل رمز
                with signal_cache_lock:
                    if len([s for s in open_signals_cache.values() if s.get('is_real_trade') == is_real_mode]) >= MAX_OPEN_TRADES:
                        logger.info("⏸️ تم الوصول للحد الأقصى للصفقات المفتوحة أثناء المسح. إيقاف الدورة الحالية.")
                        break

                try:
                    df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df is None or len(df) < BB_PERIOD:
                        continue # لا توجد بيانات كافية

                    df_features = calculate_all_features(df)
                    
                    signal_details = check_bb_stoch_signal(df_features)
                    if not signal_details:
                        continue # لا توجد إشارة أولية

                    entry_price = signal_details['price']
                    
                    bullish_pattern = detect_bullish_patterns(df_features)
                    if not bullish_pattern:
                        log_rejection(symbol, "No Bullish Pattern")
                        continue

                    if not passes_order_book_filter(symbol, entry_price):
                        log_rejection(symbol, "Order Book Check Failed", {"price": entry_price})
                        continue

                    ml_prediction = ml_confirm.get_prediction(symbol, df_features)
                    if ml_prediction is None:
                        log_rejection(symbol, "ML Model Load Failed")
                        continue
                    if ml_prediction < BUY_CONFIDENCE_THRESHOLD:
                        log_rejection(symbol, "ML Model Rejected", {"prediction": f"{ml_prediction:.2f}"})
                        continue

                    tp_sl_data = calculate_tp_sl(symbol, entry_price, df_features)
                    if not tp_sl_data:
                        log_rejection(symbol, "TP/SL Calculation Failed")
                        continue

                    if is_real_mode:
                        quantity = calculate_position_size(symbol, entry_price, tp_sl_data['stop_loss'])
                    else:
                        quantity = adjust_quantity_to_lot_size(symbol, STATS_TRADE_SIZE_USDT / entry_price)

                    if not quantity or quantity <= 0:
                        log_rejection(symbol, "Invalid Position Size", {"quantity": quantity})
                        continue
                    
                    signal_data = {
                        "symbol": symbol, "entry_price": entry_price, "quantity": float(quantity),
                        "target_price": tp_sl_data['target_price'], "stop_loss": tp_sl_data['stop_loss'],
                        "trailing_stop_loss": USE_TRAILING_STOP_LOSS, "trailing_activation_price": tp_sl_data.get('trailing_activation_price'),
                        "trailing_distance": TRAILING_DISTANCE_PERCENT, "is_real_trade": is_real_mode,
                        "ml_model_name": ml_confirm.get_model_name(symbol), "ml_prediction_score": float(ml_prediction),
                        "strategy_name": "BB_STOCH_CANDLE_ML_V9", "bullish_pattern_detected": bullish_pattern
                    }
                    
                    # هذه الدالة ستتعامل مع إدخال البيانات في قاعدة البيانات وتنفيذ الأمر
                    insert_signal_into_db(signal_data)
                    time.sleep(1) # تأخير بسيط بعد العثور على إشارة

                except Exception as e:
                    logger.error(f"❌ خطأ أثناء معالجة العملة {symbol}: {e}", exc_info=False)

            logger.info(f"✅ دورة المسح اكتملت. الانتظار لمدة {60} ثانية.")
            time.sleep(60)

        except KeyboardInterrupt:
            logger.info("🛑 تم إيقاف البوت يدوياً.")
            break
        except Exception as main_err:
            logger.error(f"❌ خطأ فادح في الحلقة الرئيسية: {main_err}", exc_info=True)
            log_and_notify('error', f"Critical error in main loop: {main_err}", "SYSTEM_ERROR")
            time.sleep(120)


def price_update_loop():
    # ... (code remains the same)
    pass

def initialize_bot_services():
    global client, validated_symbols_to_scan
    logger.info("🤖 [Bot Services] بدء التهيئة...")
    try:
        client = Client(API_KEY, API_SECRET)
        init_db()
        init_redis()
        get_exchange_info_map()
        load_open_signals_to_cache()
        load_notifications_to_cache()
        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan: logger.critical("❌ لا توجد عملات صالحة للمسح."); return
        
        Thread(target=main_loop_new_strategy, daemon=True).start()
        Thread(target=price_update_loop, daemon=True).start()
        Thread(target=trade_management_loop, daemon=True).start()
        
        logger.info("✅ [Bot Services] تم بدء جميع الخدمات الخلفية بنجاح.")
        send_telegram_message("✅ *البوت ولوحة التحكم قيد التشغيل الآن*")
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حرج أثناء التهيئة: {e}", "SYSTEM"); exit(1)

# ---------------------- نقطة الانطلاق ----------------------
if __name__ == "__main__":
    logger.info("🚀 إطلاق بوت التداول مع لوحة التحكم 🚀")
    Thread(target=initialize_bot_services, daemon=True).start()
    port = int(os.environ.get('PORT', 10000))
    host = "0.0.0.0"
    logger.info(f"✅ بدء لوحة التحكم على http://{host}:{port}")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        app.run(host=host, port=port)
    logger.info("👋 [Shutdown] تم إيقاف تشغيل التطبيق.")
