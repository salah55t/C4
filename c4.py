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
import re
import gc
import random
from decimal import Decimal, ROUND_DOWN
from urllib.parse import urlparse
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException
from flask import Flask, request, Response, jsonify, render_template_string
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timedelta, timezone
from decouple import config
from typing import List, Dict, Optional, Any, Set, Tuple
from sklearn.preprocessing import StandardScaler
from collections import deque
import warnings

# --- تجاهل التحذيرات غير الهامة ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ---------------------- إعداد نظام التسجيل (Logging) - V27.5 (API Optimization) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v27_5_arabic_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV27.5')

# ---------------------- تحميل متغيرات البيئة ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
    REDIS_URL: str = config('REDIS_URL', default='redis://localhost:6379/0')
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
    exit(1)

# ---------------------- ملفات الفلاتر الديناميكية ----------------------
FILTER_PROFILES: Dict[str, Dict[str, Any]] = {
    "STRONG_UPTREND": {
        "description": "اتجاه صاعد قوي (مستخلص من البيانات)", "strategy": "MOMENTUM",
        "filters": {
            "adx": 30.0, "rel_vol": 0.5, "rsi_range": (55, 95), "roc": 0.1, "slope": 0.01,
            "min_rrr": 1.5, "min_volatility_pct": 0.40, "min_btc_correlation": 0.5, "min_bid_ask_ratio": 1.2
        }},
    "UPTREND": {
        "description": "اتجاه صاعد (مستخلص من البيانات)", "strategy": "MOMENTUM",
        "filters": {
            "adx": 22.0, "rel_vol": 0.3, "rsi_range": (50, 90), "roc": 0.0, "slope": 0.0,
            "min_rrr": 1.4, "min_volatility_pct": 0.30, "min_btc_correlation": 0.3, "min_bid_ask_ratio": 1.1
        }},
    "RANGING": {
        "description": "اتجاه عرضي/محايد", "strategy": "MOMENTUM",
        "filters": {
            "adx": 18.0, "rel_vol": 0.2, "rsi_range": (45, 75), "roc": 0.05, "slope": 0.0,
            "min_rrr": 1.5, "min_volatility_pct": 0.25, "min_btc_correlation": -0.2, "min_bid_ask_ratio": 1.2
        }},
    "DOWNTREND": {
        "description": "اتجاه هابط (مراقبة الانعكاس)", "strategy": "REVERSAL",
        "filters": {
            "min_rrr": 2.0, "min_volatility_pct": 0.5, "min_btc_correlation": -0.5,
            "min_relative_volume": 1.5, "min_bid_ask_ratio": 1.5
        }},
    "STRONG_DOWNTREND": { "description": "اتجاه هابط قوي (التداول متوقف)", "strategy": "DISABLED", "filters": {} },
    "WEEKEND": {
        "description": "سيولة منخفضة (عطلة نهاية الأسبوع)", "strategy": "MOMENTUM",
        "filters": {
            "adx": 17.0, "rel_vol": 0.2, "rsi_range": (40, 70), "roc": 0.1, "slope": 0.0,
            "min_rrr": 1.5, "min_volatility_pct": 0.25, "min_btc_correlation": -0.4, "min_bid_ask_ratio": 1.4
        }}
}

# ---------------------- الثوابت والمتغيرات العامة ----------------------
is_trading_enabled: bool = False; trading_status_lock = Lock()
force_momentum_strategy: bool = False; force_momentum_lock = Lock()
RISK_PER_TRADE_PERCENT: float = 1.0
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V8_With_Momentum'
MODEL_FOLDER: str = 'V8'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
TIMEFRAMES_FOR_TREND_ANALYSIS: List[str] = ['15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 30
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices_v8"
TRADING_FEE_PERCENT: float = 0.1
STATS_TRADE_SIZE_USDT: float = 10.0
BTC_SYMBOL: str = 'BTCUSDT'
SYMBOL_PROCESSING_BATCH_SIZE: int = 50
ADX_PERIOD: int = 14; RSI_PERIOD: int = 14; ATR_PERIOD: int = 14
EMA_PERIODS: List[int] = [21, 50, 200]
REL_VOL_PERIOD: int = 30; MOMENTUM_PERIOD: int = 12; EMA_SLOPE_PERIOD: int = 5
MAX_OPEN_TRADES: int = 4
BUY_CONFIDENCE_THRESHOLD = 0.80
ATR_FALLBACK_SL_MULTIPLIER: float = 1.5
ATR_FALLBACK_TP_MULTIPLIER: float = 2.2
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_ACTIVATION_PROFIT_PERCENT: float = 1.0
TRAILING_DISTANCE_PERCENT: float = 0.8
USE_PEAK_FILTER: bool = True
PEAK_LOOKBACK_PERIOD: int = 50
PEAK_DISTANCE_THRESHOLD_PCT: float = 0.995
DYNAMIC_FILTER_ANALYSIS_INTERVAL: int = 900
ORDER_BOOK_DEPTH_LIMIT: int = 100
ORDER_BOOK_WALL_MULTIPLIER: float = 10.0
ORDER_BOOK_ANALYSIS_RANGE_PCT: float = 0.02

# --- [جديد] متغيرات التخزين المؤقت للبيانات التاريخية ---
DATA_CACHE_TTL_SECONDS: int = 60 * 10  # 10 دقائق
historical_data_cache: Dict[str, Dict[str, Any]] = {}
data_cache_lock = Lock()

# --- متغيرات حالة الحظر من واجهة برمجة التطبيقات ---
is_api_rate_limited: bool = False
rate_limit_lock = Lock()
rate_limit_until: float = 0

conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
ml_models_cache: Dict[str, Any] = {}; exchange_info_map: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}; signal_cache_lock = Lock()
notifications_cache = deque(maxlen=50); notifications_lock = Lock()
signals_pending_closure: Set[int] = set(); closure_lock = Lock()
rejection_logs_cache = deque(maxlen=100); rejection_logs_lock = Lock()
last_market_state_check = 0
current_market_state: Dict[str, Any] = {"trend_score": 0, "trend_label": "INITIALIZING", "details_by_tf": {}, "last_updated": None}; market_state_lock = Lock()
dynamic_filter_profile_cache: Dict[str, Any] = {}; last_dynamic_filter_analysis_time: float = 0; dynamic_filter_lock = Lock()

REJECTION_REASONS_AR = {
    "Filters Not Loaded": "الفلاتر غير محملة", "Low Volatility": "تقلب منخفض جداً", "BTC Correlation": "ارتباط ضعيف بالبيتكوين",
    "RRR Filter": "نسبة المخاطرة/العائد غير كافية", "Peak Filter": "فلتر القمة (السعر قريب من القمة)", "Invalid ATR for TP/SL": "ATR غير صالح لحساب الأهداف",
    "Momentum ADX": "فلتر الزخم (ADX ضعيف)", "Momentum Rel Vol": "فلتر الزخم (حجم نسبي منخفض)", "Momentum RSI": "فلتر الزخم (RSI خارج النطاق)",
    "Momentum ROC": "فلتر الزخم (ROC سلبي أو ضعيف)", "Momentum Slope": "فلتر الزخم (ميل EMA سلبي)", "Reversal Volume Filter": "فوليوم الانعكاس ضعيف",
    "Reversal Signal Rejected by ML Model": "نموذج التعلم الآلي رفض إشارة الانعكاس", "Invalid Position Size": "حجم الصفقة غير صالح (الوقف تحت الدخول)",
    "Lot Size Adjustment Failed": "فشل ضبط حجم العقد (LOT_SIZE)", "Min Notional Filter": "قيمة الصفقة أقل من الحد الأدنى",
    "Insufficient Balance": "الرصيد غير كافٍ", "Order Book Fetch Failed": "فشل جلب دفتر الطلبات", "Order Book Imbalance": "اختلال توازن دفتر الطلبات (ضغط بيع)",
    "Large Sell Wall Detected": "تم كشف جدار بيع ضخم", "API Rate Limited": "تم تجاوز حدود الطلبات (API)"
}


# --- الدوال المساعدة ---
fng_cache: Dict[str, Any] = {"value": -1, "classification": "فشل التحميل", "last_updated": 0}
FNG_CACHE_DURATION: int = 3600

def get_fear_and_greed_index() -> Dict[str, Any]:
    global fng_cache
    now = time.time()
    if now - fng_cache["last_updated"] < FNG_CACHE_DURATION: return fng_cache
    logger.info("ℹ️ [F&G Index] Fetching new Fear and Greed index data...")
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        response.raise_for_status()
        data = response.json().get('data', [])
        if data:
            value, classification = int(data[0]['value']), data[0]['value_classification']
            fng_cache = {"value": value, "classification": classification, "last_updated": now}
            logger.info(f"✅ [F&G Index] Updated: {value} ({classification})")
        else: raise ValueError("No data in F&G API response")
    except (requests.RequestException, ValueError) as e:
        logger.error(f"❌ [F&G Index] Could not fetch F&G Index: {e}")
        if fng_cache["value"] == -1: fng_cache["last_updated"] = now
    return fng_cache

def get_session_state() -> Tuple[List[str], str, str]:
    now_utc = datetime.now(timezone.utc)
    current_time, current_weekday = now_utc.time(), now_utc.weekday()
    sessions = {"Tokyo": ("00:00", "09:00"), "London": ("08:00", "17:00"), "New York": ("13:00", "22:00")}
    active_sessions = [name for name, (start, end) in sessions.items() if datetime.strptime(start, "%H:%M").time() <= current_time < datetime.strptime(end, "%H:%M").time()]
    if current_weekday >= 5: return active_sessions, "WEEKEND", "سيولة منخفضة (عطلة نهاية الأسبوع)"
    if "London" in active_sessions and "New York" in active_sessions: return active_sessions, "HIGH", "سيولة مرتفعة (تداخل لندن ونيويورك)"
    if active_sessions: return active_sessions, "NORMAL", f"سيولة عادية ({', '.join(active_sessions)})"
    return active_sessions, "LOW", "سيولة منخفضة (خارج ساعات التداول الرئيسية)"

# --- منظم ذكي لمعالجة أخطاء Binance API والحظر ---
def handle_binance_api_errors(func):
    def wrapper(*args, **kwargs):
        global is_api_rate_limited, rate_limit_until
        with rate_limit_lock:
            if is_api_rate_limited and time.time() < rate_limit_until:
                logger.warning(f"API is rate-limited. Skipping call to {func.__name__}.")
                symbol = args[0] if args and isinstance(args[0], str) else 'N/A'
                if func.__name__ != 'check_api_status': log_rejection(symbol, "API Rate Limited", {"function": func.__name__})
                return None
        try:
            return func(*args, **kwargs)
        except BinanceAPIException as e:
            if e.code == -1003:
                with rate_limit_lock:
                    if not is_api_rate_limited:
                        ban_duration_minutes = 30
                        rate_limit_until = time.time() + (ban_duration_minutes * 60)
                        is_api_rate_limited = True
                        logger.critical(f"🚨 IP BANNED by Binance (Code -1003). Pausing all API requests for {ban_duration_minutes} minutes.")
                        log_and_notify("critical", f"IP BANNED by Binance. Pausing API requests for {ban_duration_minutes} minutes.", "API_BAN")
                        def unban_task():
                            global is_api_rate_limited
                            time.sleep(ban_duration_minutes * 60 + 5)
                            with rate_limit_lock: is_api_rate_limited = False; logger.info("✅ API rate-limit ban has been lifted. Resuming API calls.")
                        Thread(target=unban_task, daemon=True).start()
            logger.error(f"❌ Binance API Error in {func.__name__}: {e}", exc_info=False)
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected Error in {func.__name__}: {e}", exc_info=True)
            return None
    return wrapper

# ---------------------- دالة HTML للوحة التحكم ----------------------
def get_dashboard_html():
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم التداول V27.5</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon@3.4.4/build/global/luxon.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.3.1/dist/chartjs-adapter-luxon.umd.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
    <style>
        :root { --bg-main: #0D1117; --bg-card: #161B22; --border-color: #30363D; --text-primary: #E6EDF3; --text-secondary: #848D97; --accent-blue: #58A6FF; --accent-green: #3FB950; --accent-red: #F85149; --accent-yellow: #D29922; }
        body { font-family: 'Tajawal', sans-serif; background-color: var(--bg-main); color: var(--text-primary); }
        .card { background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 0.5rem; transition: all 0.3s ease; }
        .card:hover { border-color: var(--accent-blue); }
        .skeleton { animation: pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite; background-color: #21262d; border-radius: 0.5rem; }
        @keyframes pulse { 50% { opacity: .6; } }
        .tab-btn { position: relative; transition: color 0.2s ease; }
        .tab-btn.active { color: var(--text-primary); }
        .tab-btn.active::after { content: ''; position: absolute; bottom: -1px; left: 0; right: 0; height: 2px; background-color: var(--accent-blue); border-radius: 2px; }
        .toggle-bg:after { content: ''; position: absolute; top: 2px; left: 2px; background: white; border-radius: 9999px; height: 1.25rem; width: 1.25rem; transition: transform 0.2s ease-in-out; }
        input:checked + .toggle-bg:after { transform: translateX(100%); }
        input:checked + .toggle-bg { background-color: var(--accent-green); }
        .trend-light { width: 1rem; height: 1rem; border-radius: 9999px; border: 2px solid rgba(255, 255, 255, 0.1); transition: background-color 0.5s ease, box-shadow 0.5s ease; box-shadow: inset 0 1px 2px rgba(0,0,0,0.3); }
        .light-on-green { background-color: var(--accent-green); box-shadow: inset 0 1px 2px rgba(0,0,0,0.3), 0 0 10px 2px rgba(63, 185, 80, 0.6); }
        .light-on-red { background-color: var(--accent-red); box-shadow: inset 0 1px 2px rgba(0,0,0,0.3), 0 0 10px 2px rgba(248, 81, 73, 0.6); }
        .light-on-yellow { background-color: var(--accent-yellow); box-shadow: inset 0 1px 2px rgba(0,0,0,0.3), 0 0 10px 2px rgba(210, 153, 34, 0.6); }
        .light-off { background-color: #30363D; }
    </style>
</head>
<body class="p-4 md:p-6">
    <div class="container mx-auto max-w-screen-2xl">
        <header class="mb-6 flex flex-wrap justify-between items-center gap-4">
            <h1 class="text-2xl md:text-3xl font-extrabold text-white">
                <span class="text-accent-blue">لوحة تحكم التداول</span>
                <span class="text-text-secondary font-medium">V27.5</span>
            </h1>
            <div id="trend-lights-container" class="flex items-center gap-x-6 bg-black/20 px-4 py-2 rounded-lg border border-border-color">
                <div class="flex items-center gap-2" title="اتجاه فريم 15 دقيقة"><div id="trend-light-15m" class="trend-light skeleton"></div><span class="text-sm font-bold text-text-secondary">15د</span></div>
                <div class="flex items-center gap-2" title="اتجاه فريم ساعة"><div id="trend-light-1h" class="trend-light skeleton"></div><span class="text-sm font-bold text-text-secondary">1س</span></div>
                <div class="flex items-center gap-2" title="اتجاه فريم 4 ساعات"><div id="trend-light-4h" class="trend-light skeleton"></div><span class="text-sm font-bold text-text-secondary">4س</span></div>
            </div>
            <div id="connection-status" class="flex items-center gap-3 text-sm">
                <div class="flex items-center gap-2"><div id="db-status-light" class="w-2.5 h-2.5 rounded-full bg-gray-600 animate-pulse"></div><span class="text-text-secondary">DB</span></div>
                <div class="flex items-center gap-2"><div id="api-status-light" class="w-2.5 h-2.5 rounded-full bg-gray-600 animate-pulse"></div><span class="text-text-secondary">API</span></div>
            </div>
        </header>
        <!-- ... (بقية كود HTML لم يتغير) ... -->
        <main>
            <div id="signals-tab" class="tab-content">...</div>
            <div id="stats-tab" class="tab-content hidden">...</div>
            <div id="notifications-tab" class="tab-content hidden">...</div>
            <div id="rejections-tab" class="tab-content hidden">...</div>
            <div id="filters-tab" class="tab-content hidden">...</div>
        </main>
    </div>
<script>
// ... (كود JavaScript لم يتغير بشكل كبير) ...
function manualCloseSignal(signalId) {
    if (confirm(`هل أنت متأكد من رغبتك في إغلاق الصفقة #${signalId} يدوياً؟`)) {
        fetch(`/api/close/${signalId}`, { method: 'POST' }).then(res => res.json()).then(data => {
            alert(data.message || data.error);
            refreshData();
        });
    }
}
function refreshData() {
    // ...
    updateList('/api/rejection_logs', 'rejections-list', log => {
        const details = log.details ? Object.entries(log.details).map(([key, value]) => `${key}: ${value}`).join(', ') : 'لا توجد تفاصيل';
        return `<div class="p-3 rounded-md bg-gray-900/50 text-sm">[${new Date(log.timestamp).toLocaleString('fr-CA', { timeZone: 'UTC' })}] <strong>${log.symbol}</strong>: ${log.reason} - <span class="font-mono text-xs text-text-secondary">${details}</span></div>`;
    });
}
setInterval(refreshData, 5000);
window.onload = refreshData;
</script>
</body>
</html>
    """

# ---------------------- دوال قاعدة البيانات ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    global conn
    logger.info("[DB] Initializing database connection...")
    db_url_to_use = DB_URL
    if 'postgres' in db_url_to_use and 'sslmode' not in db_url_to_use: db_url_to_use += ('&' if '?' in db_url_to_use else '?') + "sslmode=require"
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(db_url_to_use, connect_timeout=15, cursor_factory=RealDictCursor)
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION NOT NULL,
                        target_price DOUBLE PRECISION NOT NULL, stop_loss DOUBLE PRECISION NOT NULL, status TEXT DEFAULT 'open',
                        closing_price DOUBLE PRECISION, closed_at TIMESTAMP, profit_percentage DOUBLE PRECISION,
                        strategy_name TEXT, signal_details JSONB, current_peak_price DOUBLE PRECISION,
                        is_real_trade BOOLEAN DEFAULT FALSE, quantity DOUBLE PRECISION, order_id TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_signals_status ON signals (status);
                    CREATE TABLE IF NOT EXISTS notifications (id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(), type TEXT NOT NULL, message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE);
                """)
            conn.commit()
            logger.info("✅ [DB] Database connection and schema are up-to-date.")
            return
        except Exception as e:
            logger.error(f"❌ [DB] Error during initialization (Attempt {attempt + 1}/{retries}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: logger.critical("❌ [DB] Failed to connect to the database.")

def check_db_connection() -> bool:
    global conn
    if conn is None or conn.closed != 0:
        logger.warning("[DB] Connection is closed, attempting to reconnect...")
        init_db()
    try:
        if conn and conn.closed == 0:
            with conn.cursor() as cur: cur.execute("SELECT 1;")
            return True
        return False
    except (OperationalError, InterfaceError):
        logger.error(f"❌ [DB] Connection lost. Reconnecting...")
        try: init_db(); return conn is not None and conn.closed == 0
        except Exception as retry_e: logger.error(f"❌ [DB] Reconnect failed: {retry_e}"); return False
    return False

def log_and_notify(level: str, message: str, notification_type: str):
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error, 'critical': logger.critical}
    log_methods.get(level.lower(), logger.info)(message)
    if not check_db_connection() or not conn: return
    try:
        new_notification = {"timestamp": datetime.now().isoformat(), "type": notification_type, "message": message}
        with notifications_lock: notifications_cache.appendleft(new_notification)
        with conn.cursor() as cur: cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
    except Exception as e: logger.error(f"❌ [Notify DB] Failed to save notification: {e}"); conn.rollback()

def log_rejection(symbol: str, reason_key: str, details: Optional[Dict] = None):
    reason_ar = REJECTION_REASONS_AR.get(reason_key, reason_key)
    details_str = " | ".join([f"{k}: {v}" for k, v in (details or {}).items()])
    logger.info(f"🚫 [REJECTED] {symbol} | {reason_ar} ({reason_key}) | {details_str}")
    with rejection_logs_lock:
        rejection_logs_cache.appendleft({"timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol, "reason": reason_ar, "details": details or {}})

def init_redis() -> None:
    global redis_client
    logger.info("[Redis] Initializing Redis connection...")
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] Successfully connected to Redis server.")
    except redis.exceptions.ConnectionError as e: logger.critical(f"❌ [Redis] Failed to connect to Redis: {e}"); exit(1)

# ---------------------- دوال Binance والبيانات ----------------------
@handle_binance_api_errors
def get_exchange_info_map_call() -> Optional[Dict]: return client.get_exchange_info()

def get_exchange_info_map() -> None:
    global exchange_info_map
    logger.info("ℹ️ [Exchange Info] Fetching exchange trading rules...")
    info = get_exchange_info_map_call()
    if info: exchange_info_map = {s['symbol']: s for s in info['symbols']}; logger.info(f"✅ [Exchange Info] Loaded rules for {len(exchange_info_map)} symbols.")
    else: logger.error("❌ [Exchange Info] Could not fetch exchange info due to API error.")

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    if not client: return []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f: raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        if not exchange_info_map: get_exchange_info_map()
        active = {s for s, info in exchange_info_map.items() if info.get('quoteAsset') == 'USDT' and info.get('status') == 'TRADING'}
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ [Validation] Bot will monitor {len(validated)} symbols.")
        return validated
    except Exception as e: logger.error(f"❌ [Validation] Error during symbol validation: {e}", exc_info=True); return []

@handle_binance_api_errors
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    limit = int((days * 24 * 60) / int(re.sub('[a-zA-Z]', '', interval)))
    klines = client.get_historical_klines(symbol, interval, limit=min(limit, 1000))
    if not klines: return None
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.astype({'open': np.float32, 'high': np.float32, 'low': np.float32, 'close': np.float32, 'volume': np.float32})
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    return df.dropna()

# --- [جديد] دالة جلب البيانات مع التخزين المؤقت ---
def get_cached_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    cache_key = f"{symbol}_{interval}"
    now = time.time()
    with data_cache_lock:
        if cache_key in historical_data_cache:
            cached_item = historical_data_cache[cache_key]
            if now - cached_item['timestamp'] < DATA_CACHE_TTL_SECONDS:
                logger.debug(f"✅ [Cache HIT] Using cached data for {cache_key}.")
                return cached_item['data'].copy()
    
    logger.info(f"⏳ [Cache MISS] Fetching new historical data for {cache_key}.")
    df = fetch_historical_data(symbol, interval, days)
    
    if df is not None and not df.empty:
        with data_cache_lock:
            historical_data_cache[cache_key] = {'timestamp': now, 'data': df}
            logger.info(f"💾 [Cache SET] Stored new data for {cache_key}.")
        return df.copy()
    return None

@handle_binance_api_errors
def analyze_order_book(symbol: str, entry_price: float) -> Optional[Dict[str, Any]]:
    # ... (لم يتغير)
    return None

# ---------------------- دوال حساب الميزات وتحديد الاتجاه ----------------------
def calculate_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    # ... (لم يتغير)
    return df

def determine_market_trend_score():
    global current_market_state, last_market_state_check
    with market_state_lock:
        if time.time() - last_market_state_check < 300: return
    logger.info("🧠 [Market Score] Updating multi-timeframe trend score...")
    try:
        total_score, details, tf_weights = 0, {}, {'15m': 0.2, '1h': 0.3, '4h': 0.5}
        for tf in TIMEFRAMES_FOR_TREND_ANALYSIS:
            days = 5 if tf == '15m' else (15 if tf == '1h' else 50)
            # --- [مُعدَّل] استخدام الدالة الجديدة مع التخزين المؤقت ---
            df = get_cached_historical_data(BTC_SYMBOL, tf, days)
            if df is None or len(df) < EMA_PERIODS[-1]:
                details[tf] = {"score": 0, "label": "غير واضح", "reason": "بيانات غير كافية"}; continue
            for p in EMA_PERIODS: df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()
            last = df.iloc[-1]
            close, ema21, ema50, ema200 = last['close'], last['ema_21'], last['ema_50'], last['ema_200']
            tf_score = (1 if close > ema21 else -1) + (1 if ema21 > ema50 else -1) + (1 if ema50 > ema200 else -1)
            label = "صاعد" if tf_score >= 2 else ("هابط" if tf_score <= -2 else "محايد")
            details[tf] = {"score": tf_score, "label": label, "reason": f"E21:{ema21:.2f},E50:{ema50:.2f},E200:{ema200:.2f}"}
            total_score += tf_score * tf_weights[tf]
        final_score = round(total_score)
        trend_label = "صاعد قوي" if final_score >= 4 else ("صاعد" if final_score >= 1 else ("هابط قوي" if final_score <= -4 else ("هابط" if final_score <= -1 else "محايد")))
        with market_state_lock:
            current_market_state = {"trend_score": final_score, "trend_label": trend_label, "details_by_tf": details, "last_updated": datetime.now(timezone.utc).isoformat()}
            last_market_state_check = time.time()
        logger.info(f"✅ [Market Score] New State: Score={final_score}, Label='{trend_label}'")
    except Exception as e:
        logger.error(f"❌ [Market Score] Failed to determine market state: {e}", exc_info=True)
        with market_state_lock: current_market_state.update({'trend_score': 0, 'trend_label': "غير واضح"})

def analyze_market_and_create_dynamic_profile():
    # ... (لم يتغير)
    pass

def get_current_filter_profile() -> Dict[str, Any]:
    with dynamic_filter_lock: return dict(dynamic_filter_profile_cache)

def load_ml_model_bundle_from_folder(symbol: str) -> Optional[Dict[str, Any]]:
    # ... (لم يتغير)
    return None

# ---------------------- دوال الاستراتيجية والتداول الحقيقي ----------------------
def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> Optional[Decimal]:
    # ... (لم يتغير)
    return None

@handle_binance_api_errors
def get_asset_balance_call(asset: str) -> Optional[Dict]:
    return client.get_asset_balance(asset=asset)

def calculate_position_size(symbol: str, entry_price: float, stop_loss_price: float) -> Optional[Decimal]:
    # ... (لم يتغير)
    return None

@handle_binance_api_errors
def place_order(symbol: str, side: str, quantity: Decimal, order_type: str = Client.ORDER_TYPE_MARKET) -> Optional[Dict]:
    # ... (لم يتغير)
    return None

class TradingStrategy:
    # ... (لم يتغير)
    pass

def passes_filters(symbol: str, last_features: pd.Series, profile: Dict[str, Any], entry_price: float, tp_sl_data: Dict, df_15m: pd.DataFrame) -> bool:
    # ... (لم يتغير)
    return True

def passes_order_book_check(symbol: str, order_book_analysis: Dict, profile: Dict) -> bool:
    # ... (لم يتغير)
    return True

def calculate_tp_sl(symbol: str, entry_price: float, last_atr: float) -> Optional[Dict[str, Any]]:
    # ... (لم يتغير)
    return None

def handle_price_update_message(msg: List[Dict[str, Any]]) -> None:
    if not isinstance(msg, list) or not redis_client: return
    try:
        price_updates = {item.get('s'): float(item.get('c', 0)) for item in msg if item.get('s') and item.get('c')}
        if price_updates: redis_client.hset(REDIS_PRICES_HASH_NAME, mapping=price_updates)
    except Exception as e: logger.error(f"❌ [WebSocket Price Updater] Error: {e}", exc_info=True)

def initiate_signal_closure(symbol: str, signal_to_close: Dict, status: str, closing_price: float):
    # ... (لم يتغير)
    pass

def trade_monitoring_loop():
    # ... (لم يتغير)
    pass

def send_telegram_message(target_chat_id: str, text: str):
    # ... (لم يتغير)
    pass

def send_new_signal_alert(signal_data: Dict[str, Any]):
    # ... (لم يتغير)
    pass

def insert_signal_into_db(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # ... (لم يتغير)
    return None

def close_signal(signal: Dict, status: str, closing_price: float):
    # ... (لم يتغير)
    pass

def load_open_signals_to_cache():
    # ... (لم يتغير)
    pass

def load_notifications_to_cache():
    # ... (لم يتغير)
    pass

def get_btc_data_for_bot() -> Optional[pd.DataFrame]:
    # --- [مُعدَّل] استخدام الدالة الجديدة مع التخزين المؤقت ---
    btc_data = get_cached_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
    if btc_data is not None: btc_data['btc_returns'] = btc_data['close'].pct_change()
    return btc_data

def perform_end_of_cycle_cleanup():
    logger.info("🧹 [Cleanup] Starting end-of-cycle cleanup...")
    try:
        # لا تقم بمسح ذاكرة التخزين المؤقت للبيانات التاريخية هنا، دعها تحتفظ بالبيانات
        ml_models_cache.clear()
        collected = gc.collect()
        logger.info(f"🧹 [Cleanup] ML model cache cleared. Collected {collected} objects.")
    except Exception as e: logger.error(f"❌ [Cleanup] An error occurred during cleanup: {e}", exc_info=True)

# ---------------------- حلقة العمل الرئيسية ----------------------
def main_loop():
    logger.info("[Main Loop] Waiting for initialization...")
    time.sleep(15)
    if not validated_symbols_to_scan:
        log_and_notify("critical", "No validated symbols to scan. Bot will not start.", "SYSTEM"); return
    
    log_and_notify("info", f"✅ Starting main scan loop for {len(validated_symbols_to_scan)} symbols.", "SYSTEM")
    
    while True:
        try:
            logger.info("🔄 Starting new main cycle...")
            determine_market_trend_score()
            analyze_market_and_create_dynamic_profile()
            filter_profile = get_current_filter_profile()
            active_strategy_type = filter_profile.get("strategy")

            if not active_strategy_type or active_strategy_type == "DISABLED":
                logger.warning(f"🛑 Trading disabled by profile: '{filter_profile.get('name')}'. Skipping cycle."); time.sleep(300); continue

            if redis_client:
                num_prices = redis_client.hlen(REDIS_PRICES_HASH_NAME)
                if num_prices < len(validated_symbols_to_scan) * 0.7:
                    logger.warning(f"⚠️ [Main Loop] Redis price cache is not fully populated ({num_prices}/{len(validated_symbols_to_scan)}). Waiting for WebSocket...")
                    time.sleep(30); continue
            
            btc_data = get_btc_data_for_bot()
            if btc_data is None: logger.warning("⚠️ Could not get BTC data, some features will be disabled."); time.sleep(60); continue
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            all_symbols_with_models = [s for s in validated_symbols_to_scan if os.path.exists(os.path.join(script_dir, MODEL_FOLDER, f"{BASE_ML_MODEL_NAME}_{s}.pkl"))]
            if not all_symbols_with_models: logger.warning("⚠️ No symbols with models found. Skipping scan cycle."); time.sleep(300); continue

            random.shuffle(all_symbols_with_models)
            total_batches = (len(all_symbols_with_models) + SYMBOL_PROCESSING_BATCH_SIZE - 1) // SYMBOL_PROCESSING_BATCH_SIZE

            for i in range(0, len(all_symbols_with_models), SYMBOL_PROCESSING_BATCH_SIZE):
                batch_symbols = all_symbols_with_models[i:i + SYMBOL_PROCESSING_BATCH_SIZE]
                batch_num = (i // SYMBOL_PROCESSING_BATCH_SIZE) + 1
                logger.info(f"🔄 Processing Batch {batch_num}/{total_batches} with {len(batch_symbols)} symbols.")

                for symbol in batch_symbols:
                    try:
                        with signal_cache_lock:
                            if symbol in open_signals_cache or len(open_signals_cache) >= MAX_OPEN_TRADES: continue
                        
                        model_bundle = load_ml_model_bundle_from_folder(symbol)
                        if not model_bundle: continue

                        # --- [مُعدَّل] استخدام الدالة الجديدة مع التخزين المؤقت ---
                        df_15m = get_cached_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        if df_15m is None or df_15m.empty: continue
                        
                        if not redis_client: continue
                        entry_price_str = redis_client.hget(REDIS_PRICES_HASH_NAME, symbol)
                        if not entry_price_str: logger.debug(f"[{symbol}] Price not in Redis cache. Skipping."); continue
                        entry_price = float(entry_price_str)
                        
                        df_features = calculate_features(df_15m, btc_data)
                        if df_features is None or df_features.empty: continue
                        
                        strategy = TradingStrategy(symbol)
                        ml_signal = strategy.generate_buy_signal(df_features)
                        if not ml_signal or ml_signal['confidence'] < BUY_CONFIDENCE_THRESHOLD: continue
                        
                        last_features = df_features.iloc[-1]
                        tp_sl_data = calculate_tp_sl(symbol, entry_price, last_features.get('atr', 0))
                        if not tp_sl_data or not passes_filters(symbol, last_features, filter_profile, entry_price, tp_sl_data, df_15m): continue
                        
                        order_book_analysis = analyze_order_book(symbol, entry_price)
                        if not order_book_analysis or not passes_order_book_check(symbol, order_book_analysis, filter_profile): continue
                        
                        # ... (بقية منطق إنشاء الصفقة لم يتغير) ...

                    except Exception as e:
                        logger.error(f"❌ [Processing Error] for symbol {symbol}: {e}", exc_info=True)
                    finally: time.sleep(0.5)
                
                logger.info(f"🧹 [Batch Cleanup] Cleaning up memory after batch {batch_num}/{total_batches}...")
                ml_models_cache.clear(); gc.collect()

            logger.info("✅ [End of Cycle] Full scan of all batches finished. Waiting for 60 seconds..."); time.sleep(60)

        except (KeyboardInterrupt, SystemExit):
            log_and_notify("info", "Bot is shutting down by user request.", "SYSTEM"); break
        except Exception as main_err:
            log_and_notify("error", f"Critical error in main loop: {main_err}", "SYSTEM"); time.sleep(120)

# ---------------------- واجهة برمجة تطبيقات Flask ----------------------
app = Flask(__name__)
# ... (بقية دوال Flask لم تتغير)

def run_flask():
    port = int(os.environ.get('PORT', 10000))
    host = "0.0.0.0"
    logger.info(f"✅ Preparing to start dashboard on {host}:{port}")
    try: from waitress import serve; serve(app, host=host, port=port, threads=8)
    except ImportError: logger.warning("⚠️ 'waitress' not found. Using Flask's development server."); app.run(host=host, port=port)

# ---------------------- نقطة انطلاق البرنامج ----------------------
def run_websocket_manager():
    if not client or not validated_symbols_to_scan:
        logger.error("❌ [WebSocket] Cannot start: Client or symbols not initialized."); return
    logger.info("📡 [WebSocket] Starting WebSocket Manager...")
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    streams = [f"{s.lower()}@miniTicker" for s in validated_symbols_to_scan]
    twm.start_multiplex_socket(callback=handle_price_update_message, streams=streams)
    logger.info(f"✅ [WebSocket] Subscribed to {len(streams)} price streams.")
    twm.join()

def initialize_bot_services():
    global client, validated_symbols_to_scan
    logger.info("🤖 [Bot Services] Starting background initialization...")
    try:
        client = Client(API_KEY, API_SECRET)
        init_db(); init_redis()
        get_exchange_info_map()
        load_open_signals_to_cache(); load_notifications_to_cache()
        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan:
            logger.critical("❌ No validated symbols to scan. Bot will not start."); return
        
        Thread(target=run_websocket_manager, daemon=True).start()
        # --- [مُعدَّل] زيادة وقت الانتظار للسماح لـ WebSocket بالعمل ---
        logger.info("⏳ Giving WebSocket Manager time to populate Redis (20 seconds)...")
        time.sleep(20)

        Thread(target=determine_market_trend_score, daemon=True).start()
        Thread(target=trade_monitoring_loop, daemon=True).start()
        Thread(target=main_loop, daemon=True).start()
        logger.info("✅ [Bot Services] All background services started successfully.")
    except Exception as e:
        log_and_notify("critical", f"A critical error occurred during initialization: {e}", "SYSTEM")
        exit(1)

if __name__ == "__main__":
    logger.info("🚀 LAUNCHING TRADING BOT & DASHBOARD (V27.5 - API Optimization) 🚀")
    initialization_thread = Thread(target=initialize_bot_services, daemon=True)
    initialization_thread.start()
    run_flask()
    logger.info("👋 [Shutdown] Application has been shut down."); os._exit(0)
