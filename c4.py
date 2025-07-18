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

# ---------------------- إعداد نظام التسجيل (Logging) - V27.3 (Rate Limit Handling) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v27_3_arabic_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV27.3')

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

# ---------------------- ملفات الفلاتر الديناميكية (القيم المثلى) ----------------------
FILTER_PROFILES: Dict[str, Dict[str, Any]] = {
    "STRONG_UPTREND": {
        "description": "اتجاه صاعد قوي (مستخلص من البيانات)", "strategy": "MOMENTUM",
        "filters": {
            "adx": 35.0, "rel_vol": 0.8, "rsi_range": (60, 90), "roc": 0.5, "slope": 0.05, # قيم مثلى للسوق الصاعد القوي
            "min_rrr": 1.8, "min_volatility_pct": 0.50, "min_btc_correlation": 0.6, "min_bid_ask_ratio": 1.5,
            "ml_confidence": 0.90 # إضافة ثقة نموذج التعلم الآلي كفلتر
        }},
    "UPTREND": {
        "description": "اتجاه صاعد (مستخلص من البيانات)", "strategy": "MOMENTUM",
        "filters": {
            "adx": 25.0, "rel_vol": 0.5, "rsi_range": (55, 85), "roc": 0.2, "slope": 0.02, # قيم مثلى للسوق الصاعد
            "min_rrr": 1.5, "min_volatility_pct": 0.40, "min_btc_correlation": 0.4, "min_bid_ask_ratio": 1.3,
            "ml_confidence": 0.85
        }},
    "RANGING": {
        "description": "اتجاه عرضي/محايد", "strategy": "MOMENTUM", # يمكن تغيير الاستراتيجية هنا إذا كان هناك استراتيجية خاصة بالرينج
        "filters": {
            "adx": 20.0, "rel_vol": 0.3, "rsi_range": (45, 70), "roc": 0.0, "slope": 0.0, # قيم مثلى للسوق المحايد
            "min_rrr": 1.5, "min_volatility_pct": 0.30, "min_btc_correlation": -0.1, "min_bid_ask_ratio": 1.2,
            "ml_confidence": 0.80
        }},
    "DOWNTREND": {
        "description": "اتجاه هابط (مراقبة الانعكاس)", "strategy": "REVERSAL", # استراتيجية الانعكاس
        "filters": {
            "adx": 30.0, "rel_vol": 1.0, "rsi_range": (10, 40), "roc": -0.5, "slope": -0.05, # قيم مثلى للسوق الهابط (للانعكاس)
            "min_rrr": 2.0, "min_volatility_pct": 0.6, "min_btc_correlation": -0.3, # الارتباط قد يكون سلبياً
            "min_bid_ask_ratio": 1.8, # نسبة العرض/الطلب أعلى للانعكاس الصاعد
            "ml_confidence": 0.92 # ثقة أعلى للانعكاس
        }},
    "STRONG_DOWNTREND": { "description": "اتجاه هابط قوي (التداول متوقف)", "strategy": "DISABLED", "filters": {} },
    "WEEKEND": {
        "description": "سيولة منخفضة (عطلة نهاية الأسبوع)", "strategy": "MOMENTUM",
        "filters": {
            "adx": 17.0, "rel_vol": 0.2, "rsi_range": (40, 70), "roc": 0.1, "slope": 0.0,
            "min_rrr": 1.5, "min_volatility_pct": 0.25, "min_btc_correlation": -0.4, "min_bid_ask_ratio": 1.4,
            "ml_confidence": 0.75
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
SYMBOL_PROCESSING_BATCH_SIZE: int = 50 # حجم دفعة النماذج
ADX_PERIOD: int = 14; RSI_PERIOD: int = 14; ATR_PERIOD: int = 14
EMA_PERIODS: List[int] = [21, 50, 200]
REL_VOL_PERIOD: int = 30; MOMENTUM_PERIOD: int = 12; EMA_SLOPE_PERIOD: int = 5
MAX_OPEN_TRADES: int = 4
# BUY_CONFIDENCE_THRESHOLD = 0.80 # تم نقل هذا إلى FILTER_PROFILES
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

# --- [جديد] --- متغيرات حالة الحظر من واجهة برمجة التطبيقات
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
rejection_logs_cache = deque(maxlen=100); rejection_logs_lock = Lock() # سجلات الرفض
last_market_state_check = 0
current_market_state: Dict[str, Any] = {"trend_score": 0, "trend_label": "INITIALIZING", "details_by_tf": {}, "last_updated": None}; market_state_lock = Lock()
dynamic_filter_profile_cache: Dict[str, Any] = {}; last_dynamic_filter_analysis_time: float = 0; dynamic_filter_lock = Lock()

REJECTION_REASONS_AR = {
    "Filters Not Loaded": "الفلاتر غير محملة", "Low Volatility": "تقلب منخفض جداً", "BTC Correlation": "ارتباط ضعيف بالبيتكوين",
    "RRR Filter": "نسبة المخاطرة/العائد غير كافية", "Reversal Volume Filter": "فوليوم الانعكاس ضعيف", "Momentum/Strength Filter": "فلتر الزخم والقوة",
    "Peak Filter": "فلتر القمة (السعر قريب جداً من القمة الأخيرة)", "Invalid ATR for TP/SL": "ATR غير صالح لحساب الأهداف",
    "ML Model Predicted No Buy": "نموذج التعلم الآلي لم يتنبأ بالشراء", # تحديث سبب الرفض ليكون أكثر وضوحاً
    "ML Confidence Too Low": "ثقة نموذج التعلم الآلي منخفضة جداً", # سبب رفض جديد
    "Invalid Position Size": "حجم الصفقة غير صالح (الوقف تحت الدخول)",
    "Lot Size Adjustment Failed": "فشل ضبط حجم العقد (LOT_SIZE)", "Min Notional Filter": "قيمة الصفقة أقل من الحد الأدنى",
    "Insufficient Balance": "الرصيد غير كافٍ", "Order Book Fetch Failed": "فشل جلب دفتر الطلبات", "Order Book Imbalance": "اختلال توازن دفتر الطلبات (ضغط بيع)",
    "Large Sell Wall Detected": "تم كشف جدار بيع ضخم", "API Rate Limited": "تم تجاوز حدود الطلبات (API)",
    "No Historical Data": "لا توجد بيانات تاريخية كافية للعملة", # سبب رفض جديد
    "No Current Price": "لا يوجد سعر حالي للعملة في ذاكرة التخزين المؤقت", # سبب رفض جديد
    "Feature Calculation Failed": "فشل حساب الميزات الفنية", # سبب رفض جديد
    "Redis Cache Empty": "ذاكرة التخزين المؤقت لـ Redis فارغة" # سبب رفض جديد
}

# --- [إضافة] --- الدوال المساعدة المفقودة
fng_cache: Dict[str, Any] = {"value": -1, "classification": "فشل التحميل", "last_updated": 0}
FNG_CACHE_DURATION: int = 3600 # تحديث كل ساعة

def get_fear_and_greed_index() -> Dict[str, Any]:
    """
    Fetches the Fear and Greed Index from alternative.me API with caching.
    """
    global fng_cache
    now = time.time()
    if now - fng_cache["last_updated"] < FNG_CACHE_DURATION:
        return fng_cache

    logger.info("ℹ️ [F&G Index] Fetching new Fear and Greed index data...")
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        response.raise_for_status()
        data = response.json().get('data', [])
        if data:
            value = int(data[0]['value'])
            classification = data[0]['value_classification']
            fng_cache = {
                "value": value,
                "classification": classification,
                "last_updated": now
            }
            logger.info(f"✅ [F&G Index] Updated: {value} ({classification})")
        else:
            raise ValueError("No data in F&G API response")
    except (requests.RequestException, ValueError) as e:
        logger.error(f"❌ [F&G Index] Could not fetch F&G Index: {e}")
        if fng_cache["value"] == -1:
             fng_cache["last_updated"] = now # Prevent rapid retries on failure
    return fng_cache

def get_session_state() -> Tuple[List[str], str, str]:
    """
    Determines the current active trading sessions and overall liquidity state.
    Returns a tuple: (active_sessions_list, liquidity_state, description).
    """
    now_utc = datetime.now(timezone.utc)
    current_time = now_utc.time()
    current_weekday = now_utc.weekday() # Monday is 0, Sunday is 6

    sessions = {
        "Tokyo": {"start": "00:00", "end": "09:00"},
        "London": {"start": "08:00", "end": "17:00"},
        "New York": {"start": "13:00", "end": "22:00"}
    }

    active_sessions = []
    for name, times in sessions.items():
        start_time = datetime.strptime(times["start"], "%H:%M").time()
        end_time = datetime.strptime(times["end"], "%H:%M").time()
        if start_time <= current_time < end_time:
            active_sessions.append(name)

    is_weekend = current_weekday >= 5 # Saturday or Sunday
    is_london_ny_overlap = "London" in active_sessions and "New York" in active_sessions

    if is_weekend:
        liquidity_state = "WEEKEND"
        description = "سيولة منخفضة (عطلة نهاية الأسبوع)"
    elif is_london_ny_overlap:
        liquidity_state = "HIGH"
        description = "سيولة مرتفعة (تداخل لندن ونيويورك)"
    elif active_sessions:
        liquidity_state = "NORMAL"
        description = f"سيولة عادية ({', '.join(active_sessions)})"
    else:
        liquidity_state = "LOW"
        description = "سيولة منخفضة (خارج ساعات التداول الرئيسية)"

    return active_sessions, liquidity_state, description


# --- [جديد] --- منظم ذكي لمعالجة أخطاء Binance API والحظر
def handle_binance_api_errors(func):
    def wrapper(*args, **kwargs):
        global is_api_rate_limited, rate_limit_until
        with rate_limit_lock:
            if is_api_rate_limited and time.time() < rate_limit_until:
                logger.warning(f"API is rate-limited. Skipping call to {func.__name__}.")
                # إضافة سجل رفض بسبب الحظر
                symbol = args[0] if args and isinstance(args[0], str) else 'N/A'
                if func.__name__ != 'check_api_status': # تجنب تسجيل الرفض لعمليات التحقق البسيطة
                    log_rejection(symbol, "API Rate Limited", {"function": func.__name__})
                return None
        try:
            return func(*args, **kwargs)
        except BinanceAPIException as e:
            if e.code == -1003:
                with rate_limit_lock:
                    if not is_api_rate_limited: # تفعيل الحظر فقط إذا لم يكن مفعلاً بالفعل
                        ban_duration_minutes = 30
                        rate_limit_until = time.time() + (ban_duration_minutes * 60)
                        is_api_rate_limited = True
                        logger.critical(f"🚨 IP BANNED by Binance (Code -1003). Pausing all API requests for {ban_duration_minutes} minutes.")
                        log_and_notify("critical", f"IP BANNED by Binance. Pausing API requests for {ban_duration_minutes} minutes.", "API_BAN")
                        
                        def unban_task():
                            global is_api_rate_limited
                            time.sleep(ban_duration_minutes * 60 + 5) # إضافة 5 ثوانٍ احتياطية
                            with rate_limit_lock:
                                is_api_rate_limited = False
                                logger.info("✅ API rate-limit ban has been lifted. Resuming API calls.")
                        Thread(target=unban_task, daemon=True).start()
            logger.error(f"❌ Binance API Error in {func.__name__}: {e}", exc_info=False) # تقليل الضوضاء في السجل
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected Error in {func.__name__}: {e}", exc_info=True)
            return None
    return wrapper

# ---------------------- دالة HTML للوحة التحكم ----------------------
def get_dashboard_html():
    # ... (كود HTML للوحة التحكم لم يتغير)
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم التداول V27.3</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon@3.4.4/build/global/luxon.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.3.1/dist/chartjs-adapter-luxon.umd.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-main: #0D1117; --bg-card: #161B22; --border-color: #30363D;
            --text-primary: #E6EDF3; --text-secondary: #848D97;
            --accent-blue: #58A6FF; --accent-green: #3FB950; --accent-red: #F85149; --accent-yellow: #D29922;
        }
        body { font-family: 'Tajawal', sans-serif; background-color: var(--bg-main); color: var(--text-primary); }
        .card { background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 0.5rem; transition: all 0.3s ease; }
        .card:hover { border-color: var(--accent-blue); }
        .skeleton { animation: pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite; background-color: #21262d; border-radius: 0.5rem; }
        @keyframes pulse { 50% { opacity: .6; } }
        .progress-bar-container { position: relative; width: 100%; height: 0.75rem; background-color: #30363d; border-radius: 999px; overflow: hidden; }
        .progress-bar { height: 100%; transition: width 0.5s ease-in-out; border-radius: 999px; }
        .progress-labels { display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--text-secondary); padding: 0 2px; margin-top: 4px; }
        #needle { transition: transform 1s cubic-bezier(0.68, -0.55, 0.27, 1.55); }
        .tab-btn { position: relative; transition: color 0.2s ease; }
        .tab-btn.active { color: var(--text-primary); }
        .tab-btn.active::after { content: ''; position: absolute; bottom: -1px; left: 0; right: 0; height: 2px; background-color: var(--accent-blue); border-radius: 2px; }
        .table-row:hover { background-color: #1a2029; }
        .toggle-bg:after { content: ''; position: absolute; top: 2px; left: 2px; background: white; border-radius: 9999px; height: 1.25rem; width: 1.25rem; transition: transform 0.2s ease-in-out; }
        input:checked + .toggle-bg:after { transform: translateX(100%); }
        input:checked + .toggle-bg { background-color: var(--accent-green); }
        .trend-light {
            width: 1rem; height: 1rem; border-radius: 9999px;
            border: 2px solid rgba(255, 255, 255, 0.1);
            transition: background-color 0.5s ease, box-shadow 0.5s ease;
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.3);
        }
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
                <span class="text-text-secondary font-medium">V27.3</span>
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

        <!-- قسم التحكم والمعلومات -->
        <section class="mb-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-5">
            <div class="card p-4">
                 <h3 class="font-bold mb-3 text-lg text-text-secondary">حالة السوق (BTC)</h3>
                 <div class="grid grid-cols-2 gap-4 text-center">
                     <div><h4 class="text-sm font-medium text-text-secondary">تقييم الاتجاه</h4><div id="overall-regime" class="text-2xl font-bold skeleton h-8 w-3/4 mx-auto mt-1"></div></div>
                     <div><h4 class="text-sm font-medium text-text-secondary">نقاط الاتجاه</h4><div id="trend-score" class="text-3xl font-bold skeleton h-9 w-1/2 mx-auto"></div></div>
                 </div>
            </div>
            <div class="card p-4">
                 <h3 class="font-bold mb-3 text-lg text-text-secondary">ملف الفلاتر والاستراتيجية</h3>
                 <div class="text-center">
                     <div id="filter-profile-name" class="text-xl font-bold skeleton h-7 w-full mx-auto mt-1"></div>
                     <div id="active-strategy" class="text-base text-text-secondary skeleton h-5 w-2/3 mx-auto mt-2"></div>
                 </div>
            </div>
            <div class="card p-4">
                 <h3 class="font-bold mb-3 text-lg text-text-secondary">البورصات النشطة</h3>
                 <div id="active-sessions-list" class="flex flex-wrap gap-2 items-center justify-center pt-2 skeleton h-12 w-full"></div>
            </div>
            <div class="card p-4 flex flex-col justify-center items-center">
                <h3 class="font-bold text-lg text-text-secondary mb-2">التحكم بالتداول الحقيقي</h3>
                <div class="flex items-center space-x-3 space-x-reverse">
                    <span id="trading-status-text" class="font-bold text-lg text-accent-red">غير مُفعَّل</span>
                    <label for="trading-toggle" class="flex items-center cursor-pointer">
                        <div class="relative"><input type="checkbox" id="trading-toggle" class="sr-only" onchange="toggleTrading()"><div class="toggle-bg block bg-accent-red w-12 h-7 rounded-full"></div></div>
                    </label>
                </div>
                 <div class="mt-2 text-xs text-text-secondary">رصيد USDT: <span id="usdt-balance" class="font-mono skeleton w-20 inline-block"></span></div>
            </div>
            <div class="card p-4 flex flex-col justify-center items-center bg-blue-900/20 border-accent-blue">
                <h3 class="font-bold text-lg text-text-secondary mb-2">التحكم بالاستراتيجية</h3>
                <div class="flex items-center space-x-3 space-x-reverse">
                    <span id="force-momentum-text" class="font-bold text-lg text-text-secondary">تلقائي</span>
                    <label for="force-momentum-toggle" class="flex items-center cursor-pointer">
                        <div class="relative"><input type="checkbox" id="force-momentum-toggle" class="sr-only" onchange="toggleMomentumStrategy()"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div>
                    </label>
                </div>
                 <div id="force-momentum-desc" class="mt-2 text-xs text-text-secondary text-center">فرض استراتيجية الزخم</div>
            </div>
        </section>

        <section class="mb-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
            <div class="card p-4 flex flex-col justify-center items-center text-center">
                <h3 class="font-bold text-text-secondary text-lg">صفقات مفتوحة</h3>
                <div id="open-trades-value" class="text-5xl font-black text-accent-blue mt-2 skeleton h-12 w-1/2"></div>
            </div>
            <div class="card p-4 flex flex-col justify-center items-center">
                 <h3 class="font-bold mb-2 text-lg text-text-secondary">الخوف والطمع</h3>
                 <div id="fear-greed-gauge" class="relative w-full max-w-[150px] aspect-square"></div>
                 <div id="fear-greed-value" class="text-3xl font-bold mt-[-20px] skeleton h-10 w-1/2"></div>
                 <div id="fear-greed-text" class="text-md text-text-secondary skeleton h-6 w-3/4 mt-1"></div>
            </div>
            <div id="profit-chart-card" class="card lg:col-span-2 p-4">
                <div class="flex justify-between items-center mb-3">
                    <h3 class="font-bold text-lg text-text-secondary">منحنى الربح التراكمي (%)</h3>
                    <div id="net-profit-usdt" class="text-2xl font-bold skeleton h-8 w-1/3"></div>
                </div>
                <div class="relative h-80">
                    <canvas id="profitChart"></canvas>
                    <div id="profit-chart-loader" class="absolute inset-0 flex items-center justify-center bg-bg-card z-10"><div class="skeleton w-full h-full"></div></div>
                </div>
            </div>
        </section>

        <div class="mb-4 border-b border-border-color">
            <nav class="flex space-x-6 -mb-px" aria-label="Tabs">
                <button onclick="showTab('signals', this)" class="tab-btn active text-white py-3 px-1 font-semibold">الصفقات</button>
                <button onclick="showTab('stats', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الإحصائيات</button>
                <button onclick="showTab('notifications', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الإشعارات</button>
                <button onclick="showTab('rejections', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الصفقات المرفوضة</button>
                <button onclick="showTab('filters', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الفلاتر الحالية</button>
            </nav>
        </div>

        <main>
            <div id="signals-tab" class="tab-content"><div class="overflow-x-auto card p-0"><table class="min-w-full text-sm text-right"><thead class="border-b border-border-color bg-black/20"><tr><th class="p-4 font-semibold text-text-secondary">العملة</th><th class="p-4 font-semibold text-text-secondary">الحالة</th><th class="p-4 font-semibold text-text-secondary">الكمية</th><th class="p-4 font-semibold text-text-secondary">الربح/الخسارة</th><th class="p-4 font-semibold text-text-secondary w-[25%]">التقدم</th><th class="p-4 font-semibold text-text-secondary">الدخول/الحالي</th><th class="p-4 font-semibold text-text-secondary">إجراء</th></tr></thead><tbody id="signals-table"></tbody></table></div></div>
            <div id="stats-tab" class="tab-content hidden"><div id="stats-container" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4"></div></div>
            <div id="notifications-tab" class="tab-content hidden"><div id="notifications-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-2"></div></div>
            <div id="rejections-tab" class="tab-content hidden"><div id="rejections-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-2"></div></div>
            <div id="filters-tab" class="tab-content hidden"><div id="filters-display" class="card p-4 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4"></div></div>
        </main>
    </div>

<script>
let profitChartInstance;
const TREND_STYLES = {
    "صاعد قوي": { color: "text-accent-green" }, "صاعد": { color: "text-green-400" }, "محايد": { color: "text-accent-yellow" },
    "هابط": { color: "text-red-400" }, "هابط قوي": { color: "text-accent-red" }, "غير واضح": { color: "text-text-secondary" }, "تهيئة...": { color: "text-accent-blue" }
};
const STRATEGY_STYLES = {
    "MOMENTUM": { text: "متابعة الزخم", color: "text-accent-blue" }, "REVERSAL": { text: "بحث عن انعكاس", color: "text-accent-yellow" },
    "DISABLED": { text: "متوقف (غير مناسب)", color: "text-text-secondary" }, "FORCED_MOMENTUM": { text: "زخم (يدوي)", color: "text-cyan-400" }
};
const TREND_LIGHT_COLORS = { "صاعد": "light-on-green", "هابط": "light-on-red", "محايد": "light-on-yellow", "غير واضح": "light-off" };

function formatNumber(num, digits = 2) {
    if (num === null || num === undefined || isNaN(num)) return 'N/A';
    return num.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
function showTab(tabName, element) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.add('hidden'));
    document.getElementById(`${tabName}-tab`).classList.remove('hidden');
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active', 'text-white'));
    element.classList.add('active', 'text-white');
}
async function apiFetch(url, options = {}) {
    try {
        const response = await fetch(url, options);
        if (!response.ok) { console.error(`API Error ${response.status}`); return { error: `HTTP Error ${response.status}` }; }
        return await response.json();
    } catch (error) { console.error(`Fetch error for ${url}:`, error); return { error: "Network or fetch error" }; }
}
function getFngColor(value) {
    if (value < 25) return 'var(--accent-red)'; if (value < 45) return '#F97316';
    if (value < 55) return 'var(--accent-yellow)'; if (value < 75) return '#84CC16';
    return 'var(--accent-green)';
}
function renderFearGreedGauge(value, classification) {
    const container = document.getElementById('fear-greed-gauge');
    const valueEl = document.getElementById('fear-greed-value');
    const textEl = document.getElementById('fear-greed-text');
    [valueEl, textEl].forEach(el => el.classList.remove('skeleton', 'h-10', 'w-1/2', 'h-6', 'w-3/4'));

    if (value === -1) {
        container.innerHTML = `<div class="text-center text-text-secondary">خطأ</div>`;
        valueEl.textContent = 'N/A'; textEl.textContent = 'فشل التحميل';
        return;
    }
    valueEl.textContent = value; textEl.textContent = classification;
    const angle = -90 + (value / 100) * 180;
    const color = getFngColor(value);
    valueEl.style.color = color;
    container.innerHTML = `<svg viewBox="0 0 100 57" class="w-full h-full"><defs><linearGradient id="g"><stop offset="0%" stop-color="#F85149"/><stop offset="50%" stop-color="#D29922"/><stop offset="100%" stop-color="#3FB950"/></linearGradient></defs><path d="M10 50 A 40 40 0 0 1 90 50" stroke="url(#g)" stroke-width="10" fill="none" stroke-linecap="round"/><g transform="rotate(${angle} 50 50)"><path d="M50 45 L 47 15 Q 50 10 53 15 L 50 45" fill="${color}" id="needle"/></g><circle cx="50" cy="50" r="4" fill="${color}"/></svg>`;
}
function updateMarketStatus() {
    apiFetch('/api/market_status').then(data => {
        if (!data || data.error) return;
        const apiStatusLight = document.getElementById('api-status-light');
        if (data.is_rate_limited) {
            apiStatusLight.className = 'w-2.5 h-2.5 rounded-full bg-orange-500 animate-pulse';
            apiStatusLight.title = `API Rate Limited until ${new Date(data.rate_limit_until * 1000).toLocaleTimeString()}`;
        } else {
            apiStatusLight.className = `w-2.5 h-2.5 rounded-full ${data.api_ok ? 'bg-green-500' : 'bg-red-500'}`;
            apiStatusLight.title = `API Status: ${data.api_ok ? 'OK' : 'Error'}`;
        }
        document.getElementById('db-status-light').className = `w-2.5 h-2.5 rounded-full ${data.db_ok ? 'bg-green-500' : 'bg-red-500'}`;
        
        updateMomentumToggle(data.force_momentum_enabled);

        const state = data.market_state;
        const trendLabel = state.trend_label || "غير واضح";
        const trendStyle = TREND_STYLES[trendLabel] || TREND_STYLES["غير واضح"];
        
        document.getElementById('overall-regime').textContent = trendLabel;
        document.getElementById('overall-regime').className = `text-2xl font-bold ${trendStyle.color}`;
        document.getElementById('trend-score').textContent = state.trend_score;
        document.getElementById('trend-score').className = `text-3xl font-bold ${trendStyle.color}`;

        const trendDetails = state.details_by_tf || {};
        ['15m', '1h', '4h'].forEach(tf => {
            const lightEl = document.getElementById(`trend-light-${tf}`);
            if (lightEl) {
                const trendInfo = trendDetails[tf];
                const trend = trendInfo ? trendInfo.label : "غير واضح";
                const colorClass = TREND_LIGHT_COLORS[trend] || TREND_LIGHT_COLORS["غير واضح"];
                lightEl.className = `trend-light ${colorClass}`;
            }
        });

        const profile = data.filter_profile;
        let strategy = profile.strategy || "DISABLED";
        if (data.force_momentum_enabled) strategy = "FORCED_MOMENTUM";
        const strategyStyle = STRATEGY_STYLES[strategy] || STRATEGY_STYLES["DISABLED"];
        document.getElementById('active-strategy').textContent = strategyStyle.text;
        document.getElementById('active-strategy').className = `text-base font-bold ${strategyStyle.color}`;
        document.getElementById('filter-profile-name').textContent = profile.name;
        document.getElementById('filter-profile-name').className = `text-xl font-bold ${trendStyle.color}`;

        const sessions = data.active_sessions;
        const sessionsDiv = document.getElementById('active-sessions-list');
        sessionsDiv.innerHTML = '';
        sessionsDiv.classList.remove('skeleton', 'h-12');
        if (sessions && sessions.length > 0) {
            const sessionColors = { 'London': 'bg-blue-500/20 text-blue-300', 'New York': 'bg-green-500/20 text-green-300', 'Tokyo': 'bg-red-500/20 text-red-300' };
            sessions.forEach(session => {
                const colorClass = sessionColors[session] || 'bg-gray-500/20 text-gray-300';
                sessionsDiv.innerHTML += `<span class="${colorClass} text-sm font-semibold px-3 py-1 rounded-full">${session}</span>`;
            });
        } else {
            sessionsDiv.innerHTML = '<span class="text-text-secondary text-sm">لا توجد بورصات رئيسية مفتوحة</span>';
        }

        renderFearGreedGauge(data.fear_and_greed.value, data.fear_and_greed.classification);
        
        const usdtBalanceEl = document.getElementById('usdt-balance');
        usdtBalanceEl.textContent = data.usdt_balance ? `$${formatNumber(data.usdt_balance, 2)}` : 'N/A';
        usdtBalanceEl.classList.remove('skeleton', 'w-20');

        const filtersDisplay = document.getElementById('filters-display');
        filtersDisplay.innerHTML = '';
        if(profile && profile.filters && Object.keys(profile.filters).length > 0) {
            for (const [key, value] of Object.entries(profile.filters)) {
                let displayValue = value;
                if (typeof value === 'number') displayValue = formatNumber(value, 4);
                if (Array.isArray(value)) displayValue = `(${formatNumber(value[0])} - ${formatNumber(value[1])})`;
                filtersDisplay.innerHTML += `<div class="bg-gray-900/50 p-3 rounded-lg text-center"><div class="text-sm text-text-secondary uppercase">${key.replace(/_/g, ' ')}</div><div class="text-xl font-bold text-accent-blue font-mono">${displayValue}</div></div>`;
            }
        } else {
            filtersDisplay.innerHTML = '<p class="text-text-secondary col-span-full text-center">لا توجد فلاتر نشطة (التداول متوقف).</p>';
        }
    });
}
function updateTradingStatus() {
    apiFetch('/api/trading/status').then(data => {
        if (!data || data.error) return;
        const toggle = document.getElementById('trading-toggle');
        const text = document.getElementById('trading-status-text');
        toggle.checked = data.is_enabled;
        text.textContent = data.is_enabled ? 'مُفعَّل' : 'غير مُفعَّل';
        text.className = `font-bold text-lg ${data.is_enabled ? 'text-accent-green' : 'text-accent-red'}`;
    });
}
function updateMomentumToggle(is_forced) {
    const toggle = document.getElementById('force-momentum-toggle');
    const text = document.getElementById('force-momentum-text');
    const bg = toggle.nextElementSibling;
    toggle.checked = is_forced;
    if (is_forced) {
        text.textContent = 'زخم يدوي'; text.className = 'font-bold text-lg text-cyan-400';
        bg.classList.remove('bg-gray-600'); bg.classList.add('bg-accent-blue');
    } else {
        text.textContent = 'تلقائي'; text.className = 'font-bold text-lg text-text-secondary';
        bg.classList.remove('bg-accent-blue'); bg.classList.add('bg-gray-600');
    }
}
function toggleTrading() {
    const toggle = document.getElementById('trading-toggle');
    const msg = toggle.checked ? "هل أنت متأكد من تفعيل التداول بأموال حقيقية؟" : "هل أنت متأكد من إيقاف التداول الحقيقي؟";
    # استخدام نافذة مودال مخصصة بدلاً من confirm()
    showCustomConfirm(msg, () => {
        apiFetch('/api/trading/toggle', { method: 'POST' }).then(data => {
            if (data.message) { showCustomAlert(data.message); updateTradingStatus(); } 
            else if (data.error) { showCustomAlert(`خطأ: ${data.error}`); updateTradingStatus(); }
        });
    }, () => {
        toggle.checked = !toggle.checked; # إعادة التبديل إذا تم الإلغاء
    });
}
function toggleMomentumStrategy() {
    const toggle = document.getElementById('force-momentum-toggle');
    const msg = toggle.checked ? "هل أنت متأكد من فرض استراتيجية الزخم؟" : "هل أنت متأكد من العودة إلى الوضع التلقائي؟";
    # استخدام نافذة مودال مخصصة بدلاً من confirm()
    showCustomConfirm(msg, () => {
        apiFetch('/api/strategy/force_momentum/toggle', { method: 'POST' }).then(data => {
            if (data.message) { showCustomAlert(data.message); updateMomentumToggle(data.is_forced); } 
            else if (data.error) { showCustomAlert(`خطأ: ${data.error}`); updateMomentumToggle(!toggle.checked); }
        });
    }, () => {
        toggle.checked = !toggle.checked; # إعادة التبديل إذا تم الإلغاء
    });
}

# Custom Alert/Confirm Modals (بدلاً من alert() و confirm())
function showCustomAlert(message) {
    const modalHtml = `
        <div id="customAlertModal" class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
            <div class="bg-gray-800 p-6 rounded-lg shadow-xl max-w-sm w-full text-center border border-gray-700">
                <p class="text-lg text-white mb-4">${message}</p>
                <button onclick="document.getElementById('customAlertModal').remove()" class="bg-accent-blue hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-md">
                    حسناً
                </button>
            </div>
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', modalHtml);
}

function showCustomConfirm(message, onConfirm, onCancel) {
    const modalHtml = `
        <div id="customConfirmModal" class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
            <div class="bg-gray-800 p-6 rounded-lg shadow-xl max-w-sm w-full text-center border border-gray-700">
                <p class="text-lg text-white mb-4">${message}</p>
                <div class="flex justify-center space-x-4 space-x-reverse">
                    <button id="confirmBtn" class="bg-accent-green hover:bg-green-700 text-white font-bold py-2 px-4 rounded-md">
                        تأكيد
                    </button>
                    <button id="cancelBtn" class="bg-gray-600 hover:bg-gray-700 text-white font-bold py-2 px-4 rounded-md">
                        إلغاء
                    </button>
                </div>
            </div>
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', modalHtml);

    document.getElementById('confirmBtn').onclick = () => {
        document.getElementById('customConfirmModal').remove();
        if (onConfirm) onConfirm();
    };
    document.getElementById('cancelBtn').onclick = () => {
        document.getElementById('customConfirmModal').remove();
        if (onCancel) onCancel();
    };
}


function updateStats() {
    apiFetch('/api/stats').then(data => {
        if (!data || data.error) return;
        const profitFactorDisplay = data.profit_factor === 'Infinity' ? '∞' : formatNumber(data.profit_factor);
        document.getElementById('open-trades-value').textContent = formatNumber(data.open_trades_count, 0);
        document.getElementById('open-trades-value').classList.remove('skeleton', 'h-12', 'w-1/2');
        const netProfitEl = document.getElementById('net-profit-usdt');
        netProfitEl.textContent = `$${formatNumber(data.net_profit_usdt)}`;
        netProfitEl.className = `text-2xl font-bold ${data.net_profit_usdt >= 0 ? 'text-accent-green' : 'text-accent-red'}`;
        document.getElementById('stats-container').innerHTML = `
            <div class="card text-center p-4"><div class="text-sm text-text-secondary mb-1">نسبة النجاح</div><div class="text-3xl font-bold text-accent-blue">${formatNumber(data.win_rate)}%</div></div>
            <div class="card text-center p-4"><div class="text-sm text-text-secondary mb-1">عامل الربح</div><div class="text-3xl font-bold text-accent-yellow">${profitFactorDisplay}</div></div>
            <div class="card text-center p-4"><div class="text-sm text-text-secondary mb-1">إجمالي الصفقات</div><div class="text-3xl font-bold">${formatNumber(data.total_closed_trades, 0)}</div></div>
            <div class="card text-center p-4"><div class="text-sm text-text-secondary mb-1">متوسط الربح %</div><div class="text-3xl font-bold text-accent-green">${formatNumber(data.average_win_pct)}%</div></div>
            <div class="card text-center p-4"><div class="text-sm text-text-secondary mb-1">متوسط الخسارة %</div><div class="text-3xl font-bold text-accent-red">${formatNumber(data.average_loss_pct)}%</div></div>`;
    });
}
function updateProfitChart() {
    const loader = document.getElementById('profit-chart-loader');
    const canvas = document.getElementById('profitChart');
    const chartCard = document.getElementById('profit-chart-card');
    apiFetch('/api/profit_curve').then(data => {
        loader.style.display = 'none';
        const existingMsg = chartCard.querySelector('.no-data-msg');
        if(existingMsg) existingMsg.remove();
        if (!data || data.error || data.length <= 1) {
            canvas.style.display = 'none';
            if (!existingMsg) chartCard.insertAdjacentHTML('beforeend', '<p class="no-data-msg text-center text-text-secondary mt-8">لا توجد صفقات كافية.</p>');
            return;
        }
        canvas.style.display = 'block';
        const ctx = canvas.getContext('2d');
        const chartData = data.map(d => ({ x: luxon.DateTime.fromISO(d.timestamp).valueOf(), y: d.cumulative_profit }));
        const lastProfit = chartData[chartData.length - 1].y;
        const gradient = ctx.createLinearGradient(0, 0, 0, ctx.canvas.height);
        if (lastProfit >= 0) { gradient.addColorStop(0, 'rgba(63, 185, 80, 0.4)'); gradient.addColorStop(1, 'rgba(63, 185, 80, 0)'); } 
        else { gradient.addColorStop(0, 'rgba(248, 81, 73, 0.4)'); gradient.addColorStop(1, 'rgba(248, 81, 73, 0)'); }
        const config = { type: 'line', data: { datasets: [{ label: 'الربح %', data: chartData, borderColor: lastProfit >= 0 ? 'var(--accent-green)' : 'var(--accent-red)', backgroundColor: gradient, fill: true, tension: 0.4, pointRadius: 0 }] }, options: { responsive: true, maintainAspectRatio: false, scales: { x: { type: 'time', time: { unit: 'day' }, grid: { display: false }, ticks: { color: 'var(--text-secondary)' } }, y: { position: 'right', grid: { color: 'var(--border-color)' }, ticks: { color: 'var(--text-secondary)', callback: v => v + '%' } } }, plugins: { legend: { display: false } } } };
        if (profitChartInstance) {
            profitChartInstance.data.datasets[0].data = chartData;
            profitChartInstance.data.datasets[0].borderColor = lastProfit >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';
            profitChartInstance.data.datasets[0].backgroundColor = gradient;
            profitChartInstance.update('none');
        } else { profitChartInstance = new Chart(ctx, config); }
    });
}
function renderProgressBar(signal) {
    const { entry_price, stop_loss, target_price, current_price } = signal;
    if ([entry_price, stop_loss, target_price, current_price].some(v => v === null || v === undefined)) return '';
    const [entry, sl, tp, current] = [entry_price, stop_loss, target_price, current_price].map(parseFloat);
    const totalDist = tp - sl;
    if (totalDist <= 0) return '';
    const progressPct = Math.max(0, Math.min(100, ((current - sl) / totalDist) * 100));
    return `<div class="flex flex-col w-full"><div class="progress-bar-container"><div class="progress-bar ${current >= entry ? 'bg-accent-green' : 'bg-accent-red'}" style="width: ${progressPct}%"></div></div><div class="progress-labels"><span title="SL">${sl.toPrecision(4)}</span><span title="TP">${tp.toPrecision(4)}</span></div></div>`;
}
function updateSignals() {
    apiFetch('/api/signals').then(data => {
        const tableBody = document.getElementById('signals-table');
        if (!data || data.error) { tableBody.innerHTML = '<tr><td colspan="7" class="p-8 text-center text-text-secondary">فشل تحميل الصفقات.</td></tr>'; return; }
        if (data.length === 0) { tableBody.innerHTML = '<tr><td colspan="7" class="p-8 text-center text-text-secondary">لا توجد صفقات لعرضها.</td></tr>'; return; }
        tableBody.innerHTML = data.map(signal => {
            const pnlPct = (signal.status === 'open' || signal.status === 'updated') ? signal.pnl_pct : signal.profit_percentage;
            const pnlDisplay = pnlPct !== null && pnlPct !== undefined ? `${formatNumber(pnlPct)}%` : 'N/A';
            const pnlColor = pnlPct === null || pnlPct === undefined ? 'text-text-secondary' : (pnlPct >= 0 ? 'text-accent-green' : 'text-accent-red');
            let statusClass = 'text-gray-400'; let statusText = signal.status;
            if (signal.status === 'open') { statusClass = 'text-yellow-400'; statusText = 'مفتوحة'; }
            else if (signal.status === 'updated') { statusClass = 'text-blue-400'; statusText = 'تم تحديثها'; }
            const quantityDisplay = signal.quantity ? formatNumber(signal.quantity, 4) : '-';
            const realTradeIndicator = signal.is_real_trade ? '<span class="text-accent-green" title="صفقة حقيقية">●</span>' : '';
            const strategyName = signal.strategy_name || '';
            let strategyBadge = strategyName.includes('Reversal') ? '<span class="bg-yellow-500/20 text-yellow-300 text-xs font-bold px-2 py-1 rounded-md ml-2">انعكاس</span>' : '<span class="bg-blue-500/20 text-blue-300 text-xs font-bold px-2 py-1 rounded-md ml-2">زخم</span>';
            return `<tr class="table-row border-b border-border-color">
                    <td class="p-4 font-mono font-semibold">${realTradeIndicator} ${signal.symbol} ${strategyBadge}</td>
                    <td class="p-4 font-bold ${statusClass}">${statusText}</td>
                    <td class="p-4 font-mono text-text-secondary">${quantityDisplay}</td>
                    <td class="p-4 font-mono font-bold ${pnlColor}">${pnlDisplay}</td>
                    <td class="p-4">${(signal.status === 'open' || signal.status === 'updated') ? renderProgressBar(signal) : '-'}</td>
                    <td class="p-4 font-mono text-xs"><div>${formatNumber(signal.entry_price, 5)}</div><div class="text-text-secondary">${formatNumber(signal.current_price, 5)}</div></td>
                    <td class="p-4">${(signal.status === 'open' || signal.status === 'updated') ? `<button onclick="manualCloseSignal(${signal.id})" class="bg-red-600/80 hover:bg-red-600 text-white text-xs py-1 px-3 rounded-md">إغلاق</button>` : ''}</td>
                </tr>`;
        }).join('');
    });
}
function updateList(endpoint, listId, formatter) {
    apiFetch(endpoint).then(data => {
        if (!data || data.error) return;
        document.getElementById(listId).innerHTML = data.map(formatter).join('') || `<div class="p-4 text-center text-text-secondary">لا توجد بيانات.</div>`;
    });
}
function manualCloseSignal(signalId) {
    # استخدام نافذة مودال مخصصة بدلاً من confirm()
    showCustomConfirm(`هل أنت متأكد من رغبتك في إغلاق الصفقة #${signalId} يدوياً؟`, () => {
        fetch(`/api/close/${signalId}`, { method: 'POST' }).then(res => res.json()).then(data => {
            showCustomAlert(data.message || data.error);
            refreshData();
        });
    });
}
function refreshData() {
    updateMarketStatus();
    updateTradingStatus();
    updateStats();
    updateProfitChart();
    updateSignals();
    const dateLocaleOptions = { timeZone: 'UTC', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false };
    const locale = 'fr-CA';
    updateList('/api/notifications', 'notifications-list', n => `<div class="p-3 rounded-md bg-gray-900/50 text-sm">[${new Date(n.timestamp).toLocaleString(locale, dateLocaleOptions)}] ${n.message}</div>`);
    updateList('/api/rejection_logs', 'rejections-list', log => `<div class="p-3 rounded-md bg-gray-900/50 text-sm">[${new Date(log.timestamp).toLocaleString(locale, dateLocaleOptions)}] <strong>${log.symbol}</strong>: ${log.reason} - <span class="font-mono text-xs text-text-secondary">${JSON.stringify(log.details)}</span></div>`);
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
    if 'postgres' in db_url_to_use and 'sslmode' not in db_url_to_use:
        db_url_to_use += ('&' if '?' in db_url_to_use else '?') + "sslmode=require"
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
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL, message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE
                    );
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
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [DB] Connection lost: {e}. Reconnecting...")
        try:
            init_db()
            return conn is not None and conn.closed == 0
        except Exception as retry_e:
            logger.error(f"❌ [DB] Reconnect failed: {retry_e}")
            return False
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
    except Exception as e:
        logger.error(f"❌ [Notify DB] Failed to save notification: {e}")
        if conn: conn.rollback()

def log_rejection(symbol: str, reason_key: str, details: Optional[Dict] = None):
    reason_ar = REJECTION_REASONS_AR.get(reason_key, reason_key)
    logger.info(f"🚫 [REJECTED] {symbol} | Reason: {reason_key} | Details: {details or {}}")
    with rejection_logs_lock:
        rejection_logs_cache.appendleft({
            "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
            "reason": reason_ar, "details": details or {}
        })

def init_redis() -> None:
    global redis_client
    logger.info("[Redis] Initializing Redis connection...")
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] Successfully connected to Redis server.")
    except redis.exceptions.ConnectionError as e:
        logger.critical(f"❌ [Redis] Failed to connect to Redis: {e}")
        exit(1)

# ---------------------- دوال Binance والبيانات ----------------------
@handle_binance_api_errors
def get_exchange_info_map_call() -> Optional[Dict]:
    return client.get_exchange_info()

def get_exchange_info_map() -> None:
    global exchange_info_map
    logger.info("ℹ️ [Exchange Info] Fetching exchange trading rules...")
    info = get_exchange_info_map_call()
    if info:
        exchange_info_map = {s['symbol']: s for s in info['symbols']}
        logger.info(f"✅ [Exchange Info] Loaded rules for {len(exchange_info_map)} symbols.")
    else:
        logger.error("❌ [Exchange Info] Could not fetch exchange info due to API error.")

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
        logger.info(f"✅ [Validation] Bot will monitor {len(validated)} symbols.")
        return validated
    except Exception as e:
        logger.error(f"❌ [Validation] Error during symbol validation: {e}", exc_info=True)
        return []

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

@handle_binance_api_errors
def analyze_order_book(symbol: str, entry_price: float) -> Optional[Dict[str, Any]]:
    order_book = client.get_order_book(symbol=symbol, limit=ORDER_BOOK_DEPTH_LIMIT)
    if not order_book: return None
    bids = pd.DataFrame(order_book['bids'], columns=['price', 'qty'], dtype=float)
    asks = pd.DataFrame(order_book['asks'], columns=['price', 'qty'], dtype=float)
    price_range = entry_price * ORDER_BOOK_ANALYSIS_RANGE_PCT
    relevant_bids_vol = bids[bids['price'] >= entry_price - price_range]['qty'].sum()
    relevant_asks_vol = asks[asks['price'] <= entry_price + price_range]['qty'].sum()
    bid_ask_ratio = relevant_bids_vol / relevant_asks_vol if relevant_asks_vol > 0 else float('inf')
    avg_ask_qty = asks['qty'].mean()
    sell_wall_threshold = avg_ask_qty * ORDER_BOOK_WALL_MULTIPLIER
    nearby_asks = asks[asks['price'].between(entry_price, entry_price * 1.05)]
    large_sell_walls = nearby_asks[nearby_asks['qty'] > sell_wall_threshold]
    analysis_result = {
        "bid_ask_ratio": bid_ask_ratio, "has_large_sell_wall": not large_sell_walls.empty,
        "wall_details": large_sell_walls.to_dict('records') if not large_sell_walls.empty else []
    }
    logger.info(f"📖 [{symbol}] Order Book Analysis: Ratio={bid_ask_ratio:.2f}, Has Wall={analysis_result['has_large_sell_wall']}")
    return analysis_result

# ---------------------- دوال حساب الميزات وتحديد الاتجاه ----------------------
def calculate_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df_calc = df.copy()
    for period in EMA_PERIODS: df_calc[f'ema_{period}'] = df_calc['close'].ewm(span=period, adjust=False).mean()
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    up_move = df_calc['high'].diff(); down_move = -df_calc['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)
    if btc_df is not None and not btc_df.empty:
        merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
        df_calc['btc_correlation'] = df_calc['close'].pct_change().rolling(window=30).corr(merged_df['btc_returns'])
    else:
        df_calc['btc_correlation'] = 0.0
    df_calc[f'roc_{MOMENTUM_PERIOD}'] = (df_calc['close'] / df_calc['close'].shift(MOMENTUM_PERIOD) - 1) * 100
    ema_slope = df_calc['close'].ewm(span=EMA_SLOPE_PERIOD, adjust=False).mean()
    df_calc[f'ema_slope_{EMA_SLOPE_PERIOD}'] = (ema_slope - ema_slope.shift(1)) / ema_slope.shift(1).replace(0, 1e-9) * 100
    return df_calc.astype('float32', errors='ignore')

def determine_market_trend_score():
    global current_market_state, last_market_state_check
    with market_state_lock:
        if time.time() - last_market_state_check < 300: return
    logger.info("🧠 [Market Score] Updating multi-timeframe trend score...")
    try:
        total_score, details, tf_weights = 0, {}, {'15m': 0.2, '1h': 0.3, '4h': 0.5}
        for tf in TIMEFRAMES_FOR_TREND_ANALYSIS:
            days = 5 if tf == '15m' else (15 if tf == '1h' else 50)
            df = fetch_historical_data(BTC_SYMBOL, tf, days)
            if df is None or len(df) < EMA_PERIODS[-1]:
                details[tf] = {"score": 0, "label": "غير واضح", "reason": "بيانات غير كافية"}; continue
            for p in EMA_PERIODS: df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()
            last = df.iloc[-1]
            close, ema21, ema50, ema200 = last['close'], last['ema_21'], last['ema_50'], last['ema_200']
            tf_score = (1 if close > ema21 else -1) + (1 if ema21 > ema50 else -1) + (1 if ema50 > ema200 else -1)
            label = "صاعد" if tf_score >= 2 else ("هابط" if tf_score <= -2 else "محايد")
            details[tf] = {"score": tf_score, "label": label, "reason": f"E21:{ema21:.2f},E50:{ema50:.2f},E200:{ema200:.2f}"}
            total_score += tf_score * tf_weights[tf]
            time.sleep(0.2)
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
    global dynamic_filter_profile_cache, last_dynamic_filter_analysis_time
    with dynamic_filter_lock:
        if time.time() - last_dynamic_filter_analysis_time < DYNAMIC_FILTER_ANALYSIS_INTERVAL: return
    logger.info("🔬 [Dynamic Filter] Generating profile...")
    with force_momentum_lock: is_forced = force_momentum_strategy
    if is_forced:
        logger.warning(" BOLD [OVERRIDE] Manual momentum strategy is active.")
        base_profile = FILTER_PROFILES["UPTREND"].copy()
        liquidity_desc = "استراتيجية الزخم مفروضة يدوياً"
    else:
        active_sessions, liquidity_state, liquidity_desc = get_session_state()
        with market_state_lock: market_label = current_market_state.get("trend_label", "محايد")
        profile_key_map = {"صاعد قوي": "STRONG_UPTREND", "صاعد": "UPTREND", "هابط قوي": "STRONG_DOWNTREND", "هابط": "DOWNTREND"}
        profile_key = profile_key_map.get(market_label, "RANGING")
        base_profile = FILTER_PROFILES["WEEKEND"].copy() if liquidity_state == "WEEKEND" else FILTER_PROFILES.get(profile_key, FILTER_PROFILES["RANGING"]).copy()
    with dynamic_filter_lock:
        dynamic_filter_profile_cache = {
            "name": base_profile['description'], "description": liquidity_desc, "strategy": base_profile.get("strategy", "DISABLED"),
            "filters": base_profile.get("filters", {}), "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        last_dynamic_filter_analysis_time = time.time()
    logger.info(f"✅ [Dynamic Filter] New profile: '{base_profile['description']}' | Strategy: '{base_profile['strategy']}' | Manual Force: {is_forced}")

def get_current_filter_profile() -> Dict[str, Any]:
    with dynamic_filter_lock: return dict(dynamic_filter_profile_cache)

def load_ml_model_bundle_from_folder(symbol: str) -> Optional[Dict[str, Any]]:
    global ml_models_cache
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_name in ml_models_cache: return ml_models_cache[model_name]
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FOLDER, f"{model_name}.pkl")
    if not os.path.exists(model_path): return None
    try:
        with open(model_path, 'rb') as f: model_bundle = pickle.load(f)
        if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
            ml_models_cache[model_name] = model_bundle
            return model_bundle
        return None
    except Exception as e:
        logger.error(f"❌ [ML Model] Error loading model for {symbol}: {e}", exc_info=True)
        return None

# ---------------------- دوال الاستراتيجية والتداول الحقيقي ----------------------
def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> Optional[Decimal]:
    try:
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info: return None
        for f in symbol_info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                step_size = Decimal(f['stepSize'])
                return (Decimal(str(quantity)) // step_size) * step_size
        return Decimal(str(quantity))
    except Exception as e:
        logger.error(f"[{symbol}] Error adjusting quantity to lot size: {e}")
        return None

@handle_binance_api_errors
def get_asset_balance_call(asset: str) -> Optional[Dict]:
    return client.get_asset_balance(asset=asset)

def calculate_position_size(symbol: str, entry_price: float, stop_loss_price: float) -> Optional[Decimal]:
    balance_response = get_asset_balance_call(asset='USDT')
    if not balance_response: return None
    try:
        available_balance = Decimal(balance_response['free'])
        risk_amount_usdt = available_balance * (Decimal(str(RISK_PER_TRADE_PERCENT)) / Decimal('100'))
        risk_per_coin = Decimal(str(entry_price)) - Decimal(str(stop_loss_price))
        if risk_per_coin <= 0:
            log_rejection(symbol, "Invalid Position Size", {"detail": "Stop loss must be below entry price."}); return None
        initial_quantity = risk_amount_usdt / risk_per_coin
        adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(initial_quantity))
        if adjusted_quantity is None or adjusted_quantity <= 0:
            log_rejection(symbol, "Lot Size Adjustment Failed", {"quantity": f"{adjusted_quantity}"}); return None
        notional_value = adjusted_quantity * Decimal(str(entry_price))
        symbol_info = exchange_info_map.get(symbol)
        if symbol_info:
            for f in symbol_info['filters']:
                if f['filterType'] in ('MIN_NOTIONAL', 'NOTIONAL'):
                    min_notional = Decimal(f.get('minNotional', f.get('notional', '0')))
                    if notional_value < min_notional:
                        log_rejection(symbol, "Min Notional Filter", {"value": f"{notional_value:.2f}", "required": f"{min_notional}"}); return None
        if notional_value > available_balance:
            log_rejection(symbol, "Insufficient Balance", {"required": f"{notional_value:.2f}", "available": f"{available_balance:.2f}"}); return None
        logger.info(f"✅ [{symbol}] Calculated position size: {adjusted_quantity} | Risk: ${risk_amount_usdt:.2f}")
        return adjusted_quantity
    except Exception as e:
        logger.error(f"❌ [{symbol}] Error in calculate_position_size: {e}", exc_info=True)
        return None

@handle_binance_api_errors
def place_order(symbol: str, side: str, quantity: Decimal, order_type: str = Client.ORDER_TYPE_MARKET) -> Optional[Dict]:
    logger.info(f"➡️ [{symbol}] Attempting to place a REAL {side} order for {quantity} units.")
    order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=float(quantity))
    if order:
        logger.info(f"✅ [{symbol}] REAL {side} order placed successfully! Order ID: {order['orderId']}")
        log_and_notify('info', f"REAL TRADE: Placed {side} order for {quantity} {symbol}.", "REAL_TRADE")
    else:
        log_and_notify('error', f"REAL TRADE FAILED: {symbol}", "REAL_TRADE_ERROR")
    return order

class TradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_bundle_from_folder(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    def generate_buy_signal(self, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]) or df_features.empty: return None
        try:
            last_row = df_features.iloc[[-1]][self.feature_names]
            features_scaled = self.scaler.transform(last_row)
            prediction = self.ml_model.predict(features_scaled)[0]
            confidence = float(np.max(self.ml_model.predict_proba(features_scaled)[0]))
            
            # التحقق من تنبؤ النموذج وثقته هنا
            if prediction != 1: # إذا لم يتنبأ النموذج بـ "شراء"
                log_rejection(self.symbol, "ML Model Predicted No Buy", {"prediction": int(prediction), "confidence": f"{confidence:.2%}"})
                return None
            
            logger.debug(f"ℹ️ [{self.symbol}] ML Model predicted 'BUY' with {confidence:.2%} confidence.")
            return {'prediction': int(prediction), 'confidence': confidence}
        except Exception as e:
            logger.warning(f"⚠️ [{self.symbol}] ML Signal Generation Error: {e}")
            return None

def passes_filters(symbol: str, last_features: pd.Series, profile: Dict[str, Any], entry_price: float, tp_sl_data: Dict, df_15m: pd.DataFrame, ml_confidence: float) -> bool:
    filters = profile.get("filters", {})
    if not filters: log_rejection(symbol, "Filters Not Loaded", {"profile": profile.get('name')}); return False
    
    # فلتر ثقة نموذج التعلم الآلي
    if ml_confidence < filters.get('ml_confidence', 0.0):
        log_rejection(symbol, "ML Confidence Too Low", {"confidence": f"{ml_confidence:.2%}", "min": f"{filters.get('ml_confidence', 0.0):.2%}"}); return False

    volatility = (last_features.get('atr', 0) / entry_price * 100) if entry_price > 0 else 0
    if volatility < filters.get('min_volatility_pct', 0.0):
        log_rejection(symbol, "Low Volatility", {"volatility": f"{volatility:.2f}%", "min": f"{filters.get('min_volatility_pct', 0.0):.2f}%"}); return False
    
    correlation = last_features.get('btc_correlation', 0)
    if correlation < filters.get('min_btc_correlation', -1.0):
        log_rejection(symbol, "BTC Correlation", {"corr": f"{correlation:.2f}", "min": f"{filters.get('min_btc_correlation', -1.0)}"}); return False

    risk, reward = entry_price - float(tp_sl_data['stop_loss']), float(tp_sl_data['target_price']) - entry_price
    if risk <= 0 or reward <= 0 or (reward / risk) < filters.get('min_rrr', 0.0):
        log_rejection(symbol, "RRR Filter", {"rrr": f"{(reward/risk):.2f}" if risk > 0 else "N/A", "min": f"{filters.get('min_rrr', 0.0):.2f}"}); return False

    if profile.get("strategy") == "MOMENTUM":
        adx, rel_vol, rsi = last_features.get('adx', 0), last_features.get('relative_volume', 0), last_features.get('rsi', 0)
        roc, slope = last_features.get(f'roc_{MOMENTUM_PERIOD}', 0), last_features.get(f'ema_slope_{EMA_SLOPE_PERIOD}', 0)
        rsi_min, rsi_max = filters.get('rsi_range', (0, 100))
        if not (adx >= filters.get('adx', 0) and rel_vol >= filters.get('rel_vol', 0) and rsi_min <= rsi < rsi_max and roc > filters.get('roc', -100) and slope > filters.get('slope', -100)):
            log_rejection(symbol, "Momentum/Strength Filter", {"ADX": f"{adx:.2f}", "Vol": f"{rel_vol:.2f}", "RSI": f"{rsi:.2f}", "ROC": f"{roc:.2f}", "Slope": f"{slope:.6f}"}); return False
        if USE_PEAK_FILTER and df_15m is not None and len(df_15m) >= PEAK_LOOKBACK_PERIOD:
            lookback_data = df_15m.iloc[-PEAK_LOOKBACK_PERIOD:-1]
            if not lookback_data.empty:
                highest_high = lookback_data['high'].max()
                if entry_price >= (highest_high * PEAK_DISTANCE_THRESHOLD_PCT):
                    log_rejection(symbol, "Peak Filter", {"entry": f"{entry_price:.4f}", "peak_limit": f"{highest_high * PEAK_DISTANCE_THRESHOLD_PCT:.4f}"}); return False
    elif profile.get("strategy") == "REVERSAL":
        # فلاتر خاصة باستراتيجية الانعكاس (يمكن تعديلها بناءً على التحليل)
        adx, rel_vol, rsi = last_features.get('adx', 0), last_features.get('relative_volume', 0), last_features.get('rsi', 0)
        roc, slope = last_features.get(f'roc_{MOMENTUM_PERIOD}', 0), last_features.get(f'ema_slope_{EMA_SLOPE_PERIOD}', 0)
        rsi_min, rsi_max = filters.get('rsi_range', (0, 100))
        # مثال: قد تتطلب استراتيجية الانعكاس RSI منخفضاً جداً و ADX مرتفعاً
        if not (adx >= filters.get('adx', 0) and rel_vol >= filters.get('rel_vol', 0) and rsi_min <= rsi < rsi_max and roc < filters.get('roc', 100) and slope < filters.get('slope', 100)):
             log_rejection(symbol, "Reversal Signal Filter", {"ADX": f"{adx:.2f}", "Vol": f"{rel_vol:.2f}", "RSI": f"{rsi:.2f}", "ROC": f"{roc:.2f}", "Slope": f"{slope:.6f}"}); return False
    return True

def passes_order_book_check(symbol: str, order_book_analysis: Dict, profile: Dict) -> bool:
    filters = profile.get("filters", {})
    if order_book_analysis.get('has_large_sell_wall', True):
        log_rejection(symbol, "Large Sell Wall Detected", {"details": order_book_analysis.get('wall_details')}); return False
    bid_ask_ratio = order_book_analysis.get('bid_ask_ratio', 0)
    if bid_ask_ratio < filters.get('min_bid_ask_ratio', 1.0):
        log_rejection(symbol, "Order Book Imbalance", {"ratio": f"{bid_ask_ratio:.2f}", "min_required": filters.get('min_bid_ask_ratio', 1.0)}); return False
    return True

def calculate_tp_sl(symbol: str, entry_price: float, last_atr: float) -> Optional[Dict[str, Any]]:
    if last_atr <= 0: log_rejection(symbol, "Invalid ATR for TP/SL", {"atr": last_atr}); return None
    tp = entry_price + (last_atr * ATR_FALLBACK_TP_MULTIPLIER)
    sl = entry_price - (last_atr * ATR_FALLBACK_SL_MULTIPLIER)
    return {'target_price': tp, 'stop_loss': sl, 'source': 'ATR_Fallback'}

def handle_price_update_message(msg: List[Dict[str, Any]]) -> None:
    if not isinstance(msg, list) or not redis_client: return
    try:
        price_updates = {item.get('s'): float(item.get('c', 0)) for item in msg if item.get('s') and item.get('c')}
        if price_updates:
            redis_client.hset(REDIS_PRICES_HASH_NAME, mapping=price_updates)
            logger.debug(f"✅ [WebSocket Price Updater] Stored {len(price_updates)} price updates in Redis.")
    except Exception as e: logger.error(f"❌ [WebSocket Price Updater] Error: {e}", exc_info=True)

def initiate_signal_closure(symbol: str, signal_to_close: Dict, status: str, closing_price: float):
    signal_id = signal_to_close.get('id')
    if not signal_id: return
    with closure_lock:
        if signal_id in signals_pending_closure: return
        signals_pending_closure.add(signal_id)
    with signal_cache_lock: open_signals_cache.pop(symbol, None)
    logger.info(f"ℹ️ [Closure] Starting closure thread for signal {signal_id} ({symbol}) with status '{status}'.")
    Thread(target=close_signal, args=(signal_to_close, status, closing_price)).start()

def trade_monitoring_loop():
    logger.info("✅ [Trade Monitor] Starting trade monitoring loop.")
    while True:
        try:
            with signal_cache_lock: signals_to_check = dict(open_signals_cache)
            if not signals_to_check or not redis_client: time.sleep(1); continue
            
            symbols_to_fetch = list(signals_to_check.keys())
            redis_prices_list = redis_client.hmget(REDIS_PRICES_HASH_NAME, symbols_to_fetch)
            redis_prices = {symbol: price for symbol, price in zip(symbols_to_fetch, redis_prices_list)}
            
            for symbol, signal in signals_to_check.items():
                signal_id = signal.get('id')
                if not signal_id: continue
                with closure_lock:
                    if signal_id in signals_pending_closure: continue
                
                price_str = redis_prices.get(symbol)
                if not price_str: continue
                try: price = float(price_str)
                except (ValueError, TypeError): continue
                
                entry_price = float(signal['entry_price'])
                with signal_cache_lock:
                    if symbol in open_signals_cache:
                        open_signals_cache[symbol]['current_price'] = price
                        open_signals_cache[symbol]['pnl_pct'] = ((price / entry_price) - 1) * 100
                
                target_price, original_stop_loss = float(signal.get('target_price', 0)), float(signal.get('stop_loss', 0))
                effective_stop_loss = original_stop_loss
                
                if USE_TRAILING_STOP_LOSS and price > entry_price * (1 + TRAILING_ACTIVATION_PROFIT_PERCENT / 100):
                    current_peak = float(signal.get('current_peak_price', entry_price))
                    if price > current_peak:
                        with signal_cache_lock:
                            if symbol in open_signals_cache: open_signals_cache[symbol]['current_peak_price'] = price
                        current_peak = price
                    trailing_stop_price = current_peak * (1 - TRAILING_DISTANCE_PERCENT / 100)
                    if trailing_stop_price > effective_stop_loss: effective_stop_loss = trailing_stop_price

                status_to_set = 'target_hit' if price >= target_price else ('stop_loss_hit' if price <= effective_stop_loss else None)
                if status_to_set:
                    logger.info(f"✅ [TRIGGER] ID:{signal_id} | {symbol} | '{status_to_set}' at price {price}.")
                    initiate_signal_closure(symbol, signal, status_to_set, price)
            time.sleep(0.2)
        except Exception as e:
            logger.error(f"❌ [Trade Monitor] Critical error: {e}", exc_info=True)
            time.sleep(5)

def send_telegram_message(target_chat_id: str, text: str):
    if not TELEGRAM_TOKEN or not target_chat_id: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(target_chat_id), 'text': text, 'parse_mode': 'Markdown'}
    try: requests.post(url, json=payload, timeout=10)
    except requests.exceptions.RequestException as e: logger.error(f"❌ [Telegram] Request failed: {e}")

def send_new_signal_alert(signal_data: Dict[str, Any]):
    symbol, entry, target, sl = signal_data['symbol'], float(signal_data['entry_price']), float(signal_data['target_price']), float(signal_data['stop_loss'])
    profit_pct = ((target / entry) - 1) * 100 if entry > 0 else 0
    risk_pct = abs(((entry / sl) - 1) * 100) if sl > 0 else 0
    rrr = profit_pct / risk_pct if risk_pct > 0 else 0
    strategy_name = signal_data.get('strategy_name', '')
    title = "💫 *توصية شراء (انعكاس)*" if 'Reversal' in strategy_name else "💡 *توصية شراء جديدة*"
    details = signal_data.get('signal_details', {})
    confidence = details.get('ML_Confidence_Display', 'N/A')
    profile_name = details.get('Filter_Profile', 'N/A')
    ratio = f"{details.get('Bid_Ask_Ratio', 0):.2f}"
    trade_type_msg = f"\n*🔥 صفقة حقيقية 🔥*\n*الكمية:* `{signal_data.get('quantity')}`\n" if signal_data.get('is_real_trade') else ""
    message = (f"{title}\n\n*العملة:* `{symbol}`\n*ملف الفلتر:* `{profile_name}`\n{trade_type_msg}"
               f"*الدخول:* `{entry:.8f}`\n*الهدف:* `{target:.8f}`\n*وقف الخسارة:* `{sl:.8f}`\n\n"
               f"*الربح المتوقع:* `{profit_pct:.2f}%`\n*المخاطرة/العائد:* `1:{rrr:.2f}`\n\n"
               f"*تأكيد النموذج:* `{confidence}`\n*نسبة الشراء/البيع:* `{ratio}`")
    send_telegram_message(CHAT_ID, message)
    log_and_notify('info', f"New Signal: {symbol} via {strategy_name}. Real: {signal_data.get('is_real_trade', False)}", "NEW_SIGNAL")

def insert_signal_into_db(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not check_db_connection() or not conn: return None
    try:
        entry, target, sl = float(signal['entry_price']), float(signal['target_price']), float(signal['stop_loss'])
        is_real, quantity, order_id = signal.get('is_real_trade', False), signal.get('quantity'), signal.get('order_id')
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, current_peak_price, is_real_trade, quantity, order_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
                """, (signal['symbol'], entry, target, sl, signal.get('strategy_name'), json.dumps(signal.get('signal_details', {})), entry, is_real, quantity, order_id))
            signal['id'] = cur.fetchone()['id']
        conn.commit()
        logger.info(f"✅ [DB] Inserted signal {signal['id']} for {signal['symbol']}. Real: {is_real}")
        return signal
    except Exception as e:
        logger.error(f"❌ [Insert] Error inserting signal for {signal['symbol']}: {e}", exc_info=True)
        if conn: conn.rollback()
        return None

def close_signal(signal: Dict, status: str, closing_price: float):
    signal_id, symbol, is_real = signal.get('id'), signal.get('symbol'), signal.get('is_real_trade', False)
    with trading_status_lock: is_enabled = is_trading_enabled
    if is_real and is_enabled:
        try:
            base_asset = exchange_info_map.get(symbol, {}).get('baseAsset')
            if not base_asset: raise ValueError("Base asset unknown.")
            balance_info = get_asset_balance_call(asset=base_asset)
            if not balance_info: raise ValueError("Could not get balance.")
            quantity_to_sell = adjust_quantity_to_lot_size(symbol, float(balance_info['free']))
            if quantity_to_sell and quantity_to_sell > 0:
                if not place_order(symbol, Client.SIDE_SELL, quantity_to_sell):
                    logger.critical(f"🚨 CRITICAL: FAILED TO PLACE SELL ORDER FOR {signal_id}. MANUAL ACTION NEEDED.")
                    log_and_notify('critical', f"CRITICAL: FAILED TO SELL {symbol} for signal {signal_id}.", "REAL_TRADE_ERROR")
            else: logger.warning(f"⚠️ [{symbol}] No sellable balance found. Closing virtually.")
        except Exception as e:
            logger.critical(f"🚨 CRITICAL: Exception closing real trade for {symbol}: {e}", exc_info=True)
            with signal_cache_lock:
                if symbol not in open_signals_cache: open_signals_cache[symbol] = signal
            with closure_lock: signals_pending_closure.discard(signal_id)
            return
    elif is_real and not is_enabled:
        logger.warning(f"⚠️ [{symbol}] Real trade {signal_id} triggered closure, but master switch is OFF. Closing virtually.")
    try:
        if not check_db_connection() or not conn: raise OperationalError("DB connection failed.")
        db_closing_price, entry_price = float(closing_price), float(signal['entry_price'])
        profit_pct = ((db_closing_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        with conn.cursor() as cur:
            cur.execute("UPDATE signals SET status = %s, closing_price = %s, closed_at = NOW(), profit_percentage = %s WHERE id = %s AND status IN ('open', 'updated');",
                        (status, db_closing_price, profit_pct, signal_id))
            if cur.rowcount == 0: logger.warning(f"⚠️ [DB Close] Signal {signal_id} was already closed."); return
        conn.commit()
        status_map = {'target_hit': '✅ تحقق الهدف', 'stop_loss_hit': '🛑 ضرب وقف الخسارة', 'manual_close': '🖐️ إغلاق يدوي'}
        status_message = status_map.get(status, status)
        real_trade_tag = "🔥 REAL" if is_real else "👻 VIRTUAL"
        alert_msg = f"*{status_message} ({real_trade_tag})*\n*العملة:* `{symbol}`\n*الربح:* `{profit_pct:+.2f}%`"
        send_telegram_message(CHAT_ID, alert_msg)
        log_and_notify('info', f"{status_message}: {symbol} | Profit: {profit_pct:+.2f}% | Real: {is_real}", 'CLOSE_SIGNAL')
        logger.info(f"✅ [DB Close] Signal {signal_id} closed successfully.")
    except Exception as e:
        logger.error(f"❌ [DB Close] Critical error closing signal {signal_id}: {e}", exc_info=True)
        if conn: conn.rollback()
        if symbol:
            with signal_cache_lock:
                if symbol not in open_signals_cache: open_signals_cache[symbol] = signal
    finally:
        with closure_lock: signals_pending_closure.discard(signal_id)

def load_open_signals_to_cache():
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status IN ('open', 'updated');")
            open_signals = cur.fetchall()
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in open_signals: open_signals_cache[signal['symbol']] = dict(signal)
            logger.info(f"✅ [Loading] Loaded {len(open_signals)} open signals.")
    except Exception as e: logger.error(f"❌ [Loading] Failed to load open signals: {e}")

def load_notifications_to_cache():
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 50;")
            recent = cur.fetchall()
            with notifications_lock:
                notifications_cache.clear()
                for n in reversed(recent): n['timestamp'] = n['timestamp'].isoformat(); notifications_cache.appendleft(dict(n))
            logger.info(f"✅ [Loading] Loaded {len(notifications_cache)} notifications.")
    except Exception as e: logger.error(f"❌ [Loading] Failed to load notifications: {e}")

def get_btc_data_for_bot() -> Optional[pd.DataFrame]:
    btc_data = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
    if btc_data is not None: btc_data['btc_returns'] = btc_data['close'].pct_change()
    return btc_data

def perform_end_of_cycle_cleanup():
    logger.info("🧹 [Cleanup] Starting end-of-cycle cleanup...")
    try:
        if redis_client: redis_client.delete(REDIS_PRICES_HASH_NAME)
        ml_models_cache.clear()
        collected = gc.collect()
        logger.info(f"🧹 [Cleanup] Final garbage collection complete. Collected {collected} objects.")
    except Exception as e:
        logger.error(f"❌ [Cleanup] An error occurred during cleanup: {e}", exc_info=True)

# ---------------------- حلقة العمل الرئيسية ----------------------
def main_loop():
    logger.info("[Main Loop] Waiting for initialization...")
    time.sleep(30) # زيادة وقت الانتظار الأولي

    if not validated_symbols_to_scan:
        log_and_notify("critical", "No validated symbols to scan. Bot will not start.", "SYSTEM")
        logger.critical("❌ [Main Loop] No validated symbols to scan. Check 'crypto_list.txt' and exchange info.")
        time.sleep(300) # Sleep to prevent rapid logging
        return
    
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

            # التحقق من أن Redis يحتوي على أسعار قبل البدء في معالجة الرموز
            if redis_client:
                # محاولة جلب بعض الأسعار للتأكد من أن Redis ليس فارغًا تمامًا
                test_prices = redis_client.hgetall(REDIS_PRICES_HASH_NAME)
                if not test_prices or len(test_prices) < len(validated_symbols_to_scan) * 0.5: # على الأقل نصف الرموز لديها أسعار
                    logger.warning(f"⚠️ [Main Loop] Redis price cache is largely empty ({len(test_prices)}/{len(validated_symbols_to_scan)} symbols have prices). Waiting for WebSocket data.")
                    log_rejection("N/A", "Redis Cache Empty", {"detail": "Waiting for WebSocket to populate prices."})
                    time.sleep(60) # انتظر أكثر لـ WebSocket
                    continue
            else:
                logger.error("❌ [Main Loop] Redis client is not initialized. Cannot proceed.")
                time.sleep(60)
                continue


            btc_data = get_btc_data_for_bot()
            if btc_data is None: 
                logger.warning("⚠️ [Main Loop] Could not get BTC data, some features might be disabled or incorrect. Retrying in 60s.");
                time.sleep(60); continue
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            all_symbols_with_models = [
                s for s in validated_symbols_to_scan 
                if os.path.exists(os.path.join(script_dir, MODEL_FOLDER, f"{BASE_ML_MODEL_NAME}_{s}.pkl"))
            ]
            
            if not all_symbols_with_models:
                logger.warning("⚠️ [Main Loop] No symbols with ML models found in the specified folder. Skipping scan cycle.");
                logger.warning(f"💡 Ensure '{MODEL_FOLDER}' folder exists and contains '{BASE_ML_MODEL_NAME}_SYMBOL.pkl' files for your symbols.")
                time.sleep(300); continue

            random.shuffle(all_symbols_with_models)
            total_batches = (len(all_symbols_with_models) + SYMBOL_PROCESSING_BATCH_SIZE - 1) // SYMBOL_PROCESSING_BATCH_SIZE

            for i in range(0, len(all_symbols_with_models), SYMBOL_PROCESSING_BATCH_SIZE):
                batch_symbols = all_symbols_with_models[i:i + SYMBOL_PROCESSING_BATCH_SIZE]
                batch_num = (i // SYMBOL_PROCESSING_BATCH_SIZE) + 1
                logger.info(f"🔄 Processing Batch {batch_num}/{total_batches} with {len(batch_symbols)} symbols.")

                for symbol in batch_symbols:
                    try:
                        with signal_cache_lock:
                            if symbol in open_signals_cache or len(open_signals_cache) >= MAX_OPEN_TRADES:
                                logger.debug(f"[{symbol}] Skipping: Already open or max open trades reached.")
                                continue
                        
                        # تحميل النموذج فقط عند الحاجة
                        model_bundle = load_ml_model_bundle_from_folder(symbol)
                        if not model_bundle:
                            logger.debug(f"[{symbol}] Could not load model bundle, skipping.")
                            # لا نسجل رفض هنا لأنه قد لا يكون هناك نموذج لهذه العملة أصلاً
                            continue

                        df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        if df_15m is None or df_15m.empty:
                            log_rejection(symbol, "No Historical Data", {"timeframe": SIGNAL_GENERATION_TIMEFRAME, "days": SIGNAL_GENERATION_LOOKBACK_DAYS})
                            continue
                        
                        if not redis_client:
                            logger.error(f"[{symbol}] Redis client not initialized. Cannot fetch current price.")
                            continue # لا نسجل رفض هنا، المشكلة أعمق
                        
                        entry_price_str = redis_client.hget(REDIS_PRICES_HASH_NAME, symbol)
                        if not entry_price_str:
                            log_rejection(symbol, "No Current Price", {"detail": "Price not in Redis cache."})
                            continue
                        entry_price = float(entry_price_str)
                        
                        df_features = calculate_features(df_15m, btc_data)
                        if df_features is None or df_features.empty:
                            log_rejection(symbol, "Feature Calculation Failed", {"detail": "Could not calculate features from historical data."})
                            continue
                        
                        strategy = TradingStrategy(symbol) # يتم تحميل النموذج داخل هذه الفئة
                        ml_signal = strategy.generate_buy_signal(df_features) # هذا سيقوم بتسجيل الرفض إذا لم يتنبأ بـ "شراء"
                        
                        # التحقق من إشارة ML وثقتها بعد توليدها
                        if not ml_signal: # إذا لم يتم توليد إشارة ML أصلاً (تم رفضها داخل generate_buy_signal)
                            continue
                        
                        last_features = df_features.iloc[-1]
                        tp_sl_data = calculate_tp_sl(symbol, entry_price, last_features.get('atr', 0))
                        
                        # تمرير ثقة ML إلى دالة passes_filters
                        if not tp_sl_data or not passes_filters(symbol, last_features, filter_profile, entry_price, tp_sl_data, df_15m, ml_signal['confidence']):
                            # يتم تسجيل الرفض داخل passes_filters
                            continue
                        
                        order_book_analysis = analyze_order_book(symbol, entry_price)
                        if not order_book_analysis or not passes_order_book_check(symbol, order_book_analysis, filter_profile):
                            # يتم تسجيل الرفض داخل passes_order_book_check
                            continue
                        
                        new_signal = {
                            'symbol': symbol, 'strategy_name': f"{active_strategy_type}_ML",
                            'signal_details': {
                                'ML_Confidence_Display': f"{ml_signal['confidence']:.2%}", 'Filter_Profile': f"{filter_profile['name']}",
                                'Bid_Ask_Ratio': order_book_analysis.get('bid_ask_ratio', 0)
                            }, 'entry_price': entry_price, **tp_sl_data
                        }
                        with trading_status_lock: is_enabled = is_trading_enabled
                        if is_enabled:
                            quantity = calculate_position_size(symbol, entry_price, new_signal['stop_loss'])
                            if quantity and quantity > 0:
                                order_result = place_order(symbol, Client.SIDE_BUY, quantity)
                                if order_result: new_signal.update({'is_real_trade': True, 'quantity': float(quantity), 'order_id': order_result['orderId']})
                                else: continue
                            else: continue
                        else: new_signal['is_real_trade'] = False

                        saved_signal = insert_signal_into_db(new_signal)
                        if saved_signal:
                            with signal_cache_lock: open_signals_cache[saved_signal['symbol']] = saved_signal
                            send_new_signal_alert(saved_signal)
                    except Exception as e:
                        logger.error(f"❌ [Processing Error] for symbol {symbol}: {e}", exc_info=True)
                    finally:
                        time.sleep(0.75) 
                
                # تنظيف الذاكرة بعد كل دفعة
                logger.info(f"🧹 [Batch Cleanup] Cleaning up memory after batch {batch_num}/{total_batches}...")
                ml_models_cache.clear()
                collected = gc.collect()
                logger.info(f"🧹 [Batch Cleanup] Garbage Collector freed {collected} objects. Model cache cleared.")

            logger.info("✅ [End of Cycle] Full scan of all batches finished."); 
            logger.info("⏳ Waiting for 60 seconds before next full scan..."); 
            time.sleep(60)

        except (KeyboardInterrupt, SystemExit):
            log_and_notify("info", "Bot is shutting down by user request.", "SYSTEM"); break
        except Exception as main_err:
            log_and_notify("error", f"Critical error in main loop: {main_err}", "SYSTEM"); time.sleep(120)


# ---------------------- واجهة برمجة تطبيقات Flask ----------------------
app = Flask(__name__)
CORS(app)

@handle_binance_api_errors
def check_api_status_call() -> bool:
    client.ping(); return True

@app.route('/api/market_status')
def get_market_status():
    with market_state_lock: state_copy = dict(current_market_state)
    with force_momentum_lock: is_forced = force_momentum_strategy
    with rate_limit_lock: is_limited, limit_until = is_api_rate_limited, rate_limit_until
    profile_copy = get_current_filter_profile()
    active_sessions, _, _ = get_session_state()
    return jsonify({
        "fear_and_greed": get_fear_and_greed_index(), "market_state": state_copy, "filter_profile": profile_copy,
        "active_sessions": active_sessions, "db_ok": check_db_connection(), "api_ok": check_api_status_call() is not None,
        "usdt_balance": get_usdt_balance(), "force_momentum_enabled": is_forced,
        "is_rate_limited": is_limited, "rate_limit_until": limit_until
    })

# ... (بقية دوال Flask لم تتغير بشكل كبير)
def get_usdt_balance() -> Optional[float]:
    balance_info = get_asset_balance_call(asset='USDT')
    return float(balance_info['free']) if balance_info else None

@app.route('/')
def home(): return render_template_string(get_dashboard_html())

@app.route('/api/stats')
def get_stats():
    if not check_db_connection(): return jsonify({"error": "DB connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT status, profit_percentage, is_real_trade, quantity, entry_price FROM signals;")
            all_signals = cur.fetchall()
        open_trades_count = sum(1 for s in all_signals if s.get('status') in ['open', 'updated'])
        closed_trades = [s for s in all_signals if s.get('status') not in ['open', 'updated'] and s.get('profit_percentage') is not None]
        total_net_profit_usdt = 0.0
        if closed_trades:
            for t in closed_trades:
                profit_pct = float(t['profit_percentage']) - (2 * TRADING_FEE_PERCENT)
                trade_size = float(t['quantity']) * float(t['entry_price']) if t.get('is_real_trade') and t.get('quantity') and t.get('entry_price') else STATS_TRADE_SIZE_USDT
                total_net_profit_usdt += (profit_pct / 100) * trade_size
        wins = [float(s['profit_percentage']) for s in closed_trades if float(s['profit_percentage']) > 0]
        losses = [float(s['profit_percentage']) for s in closed_trades if float(s['profit_percentage']) < 0]
        return jsonify({
            "open_trades_count": open_trades_count, "net_profit_usdt": total_net_profit_usdt,
            "win_rate": (len(wins) / len(closed_trades) * 100) if closed_trades else 0.0,
            "profit_factor": sum(wins) / abs(sum(losses)) if abs(sum(losses)) > 0 else "Infinity",
            "total_closed_trades": len(closed_trades), "average_win_pct": sum(wins) / len(wins) if wins else 0.0,
            "average_loss_pct": sum(losses) / len(losses) if losses else 0.0
        })
    except Exception as e:
        logger.error(f"❌ [API Stats] Error: {e}", exc_info=True); return jsonify({"error": "Internal error"}), 500

@app.route('/api/profit_curve')
def get_profit_curve():
    if not check_db_connection(): return jsonify({"error": "DB connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT closed_at, profit_percentage FROM signals WHERE status NOT IN ('open', 'updated') AND profit_percentage IS NOT NULL AND closed_at IS NOT NULL ORDER BY closed_at ASC;")
            trades = cur.fetchall()
        start_time = (trades[0]['closed_at'] - timedelta(seconds=1)).isoformat() if trades else datetime.now(timezone.utc).isoformat()
        curve_data = [{"timestamp": start_time, "cumulative_profit": 0.0}]
        cumulative_profit = 0.0
        for trade in trades:
            cumulative_profit += float(trade['profit_percentage'])
            curve_data.append({"timestamp": trade['closed_at'].isoformat(), "cumulative_profit": cumulative_profit})
        return jsonify(curve_data)
    except Exception as e:
        logger.error(f"❌ [API Profit Curve] Error: {e}", exc_info=True); return jsonify({"error": "Error fetching curve"}), 500

@app.route('/api/signals')
def get_signals():
    if not all([check_db_connection(), redis_client]): return jsonify({"error": "Service connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals ORDER BY CASE WHEN status IN ('open', 'updated') THEN 0 ELSE 1 END, id DESC;")
            all_signals = [dict(s) for s in cur.fetchall()]
        open_signals_symbols = [s['symbol'] for s in all_signals if s['status'] in ('open', 'updated')]
        if open_signals_symbols:
            prices_from_redis = redis_client.hmget(REDIS_PRICES_HASH_NAME, open_signals_symbols)
            redis_prices_map = {symbol: p for symbol, p in zip(open_signals_symbols, prices_from_redis)}
            for s in all_signals:
                if s['status'] in ('open', 'updated'):
                    price_str = redis_prices_map.get(s['symbol'])
                    s['current_price'] = float(price_str) if price_str else None
                    if s['current_price'] and s.get('entry_price'):
                        s['pnl_pct'] = ((s['current_price'] / float(s['entry_price'])) - 1) * 100
        return jsonify(all_signals)
    except Exception as e:
        logger.error(f"❌ [API Signals] Error: {e}", exc_info=True); return jsonify({"error": str(e)}), 500

@app.route('/api/close/<int:signal_id>', methods=['POST'])
def manual_close_signal_api(signal_id):
    with closure_lock:
        if signal_id in signals_pending_closure: return jsonify({"error": "Signal is already being closed"}), 409
    if not check_db_connection(): return jsonify({"error": "DB connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE id = %s AND status IN ('open', 'updated');", (signal_id,))
            signal_to_close = cur.fetchone()
        if not signal_to_close: return jsonify({"error": "Signal not found or already closed"}), 404
        symbol = dict(signal_to_close)['symbol']
        price_str = redis_client.hget(REDIS_PRICES_HASH_NAME, symbol)
        if not price_str: return jsonify({"error": f"Could not fetch price for {symbol}"}), 500
        initiate_signal_closure(symbol, dict(signal_to_close), 'manual_close', float(price_str))
        return jsonify({"message": f"تم إرسال طلب إغلاق الصفقة {signal_id}..."})
    except Exception as e:
        logger.error(f"❌ [API Close] Error: {e}", exc_info=True); return jsonify({"error": str(e)}), 500

@app.route('/api/trading/status', methods=['GET'])
def get_trading_status():
    with trading_status_lock: return jsonify({"is_enabled": is_trading_enabled})

@app.route('/api/trading/toggle', methods=['POST'])
def toggle_trading_status():
    global is_trading_enabled
    with trading_status_lock:
        is_trading_enabled = not is_trading_enabled
        status_msg = "ENABLED" if is_trading_enabled else "DISABLED"
        log_and_notify('warning', f"🚨 Real trading status changed to: {status_msg}", "TRADING_STATUS_CHANGE")
        return jsonify({"message": f"Trading status set to {status_msg}", "is_enabled": is_trading_enabled})

@app.route('/api/strategy/force_momentum/toggle', methods=['POST'])
def toggle_force_momentum():
    global force_momentum_strategy
    with force_momentum_lock:
        force_momentum_strategy = not force_momentum_strategy
        status_msg = "FORCED MOMENTUM" if force_momentum_strategy else "AUTOMATIC"
        log_and_notify('warning', f"⚙️ Strategy mode changed to: {status_msg}", "STRATEGY_MODE_CHANGE")
        Thread(target=analyze_market_and_create_dynamic_profile).start()
        return jsonify({"message": f"Strategy mode set to {status_msg}", "is_forced": force_momentum_strategy})

@app.route('/api/notifications')
def get_notifications():
    with notifications_lock: return jsonify(list(notifications_cache))

@app.route('/api/rejection_logs')
def get_rejection_logs():
    with rejection_logs_lock: return jsonify(list(rejection_logs_cache))

def run_flask():
    port = int(os.environ.get('PORT', 10000))
    host = "0.0.0.0"
    logger.info(f"✅ Preparing to start dashboard on {host}:{port}")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        logger.warning("⚠️ 'waitress' not found. Using Flask's development server.")
        app.run(host=host, port=port)

# ---------------------- نقطة انطلاق البرنامج ----------------------
def run_websocket_manager():
    if not client or not validated_symbols_to_scan:
        logger.error("❌ [WebSocket] Cannot start: Client or symbols not initialized."); return
    logger.info("📡 [WebSocket] Starting WebSocket Manager...")
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    streams = [f"{s.lower()}@miniTicker" for s in validated_symbols_to_scan]
    
    # التحقق مما إذا كان WebSocket Manager قد بدأ بنجاح
    if not twm.is_alive():
        logger.critical("❌ [WebSocket] ThreadedWebsocketManager failed to start. Check API keys and network.")
        return

    twm.start_multiplex_socket(callback=handle_price_update_message, streams=streams)
    logger.info(f"✅ [WebSocket] Subscribed to {len(streams)} price streams. Waiting for data...")
    twm.join() # هذا سيجعل الثريد يعمل في الخلفية

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
        
        # تشغيل WebSocket Manager في ثريد منفصل
        Thread(target=run_websocket_manager, daemon=True).start()
        
        # إعطاء بعض الوقت لـ WebSocket Manager لبدء جلب البيانات
        logger.info("⏳ Giving WebSocket Manager some time to populate Redis with initial prices (10 seconds)...")
        time.sleep(10) # انتظار إضافي هنا
        
        Thread(target=determine_market_trend_score, daemon=True).start()
        Thread(target=trade_monitoring_loop, daemon=True).start()
        Thread(target=main_loop, daemon=True).start()
        logger.info("✅ [Bot Services] All background services started successfully.")
    except Exception as e:
        log_and_notify("critical", f"A critical error occurred during initialization: {e}", "SYSTEM")
        exit(1)

if __name__ == "__main__":
    logger.info("🚀 LAUNCHING TRADING BOT & DASHBOARD (V27.3 - Rate Limit Handling) 🚀")
    initialization_thread = Thread(target=initialize_bot_services, daemon=True)
    initialization_thread.start()
    run_flask()
    logger.info("👋 [Shutdown] Application has been shut down."); os._exit(0)

