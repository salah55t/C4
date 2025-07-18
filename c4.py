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
from binance.exceptions import BinanceAPIException, BinanceRequestException
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

# ---------------------- إعداد نظام التسجيل (Logging) - V27.7 (Dashboard Update) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v27_7_arabic_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV27.7')

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
        "filters": { "adx": 30.0, "rel_vol": 0.5, "rsi_range": (55, 95), "roc": 0.1, "slope": 0.01, "min_rrr": 1.5, "min_volatility_pct": 0.40, "min_btc_correlation": 0.5, "min_bid_ask_ratio": 1.2 }},
    "UPTREND": {
        "description": "اتجاه صاعد (مستخلص من البيانات)", "strategy": "MOMENTUM",
        "filters": { "adx": 22.0, "rel_vol": 0.3, "rsi_range": (50, 90), "roc": 0.0, "slope": 0.0, "min_rrr": 1.4, "min_volatility_pct": 0.30, "min_btc_correlation": 0.3, "min_bid_ask_ratio": 1.1 }},
    "RANGING": {
        "description": "اتجاه عرضي/محايد", "strategy": "MOMENTUM",
        "filters": { "adx": 18.0, "rel_vol": 0.2, "rsi_range": (45, 75), "roc": 0.05, "slope": 0.0, "min_rrr": 1.5, "min_volatility_pct": 0.25, "min_btc_correlation": -0.2, "min_bid_ask_ratio": 1.2 }},
    "DOWNTREND": {
        "description": "اتجاه هابط (مراقبة الانعكاس)", "strategy": "REVERSAL",
        "filters": { "min_rrr": 2.0, "min_volatility_pct": 0.5, "min_btc_correlation": -0.5, "min_relative_volume": 1.5, "min_bid_ask_ratio": 1.5 }},
    "STRONG_DOWNTREND": { "description": "اتجاه هابط قوي (التداول متوقف)", "strategy": "DISABLED", "filters": {} },
    "WEEKEND": {
        "description": "سيولة منخفضة (عطلة نهاية الأسبوع)", "strategy": "MOMENTUM",
        "filters": { "adx": 17.0, "rel_vol": 0.2, "rsi_range": (40, 70), "roc": 0.1, "slope": 0.0, "min_rrr": 1.5, "min_volatility_pct": 0.25, "min_btc_correlation": -0.4, "min_bid_ask_ratio": 1.4 }}
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
MONITOR_API_CHECK_INTERVAL: int = 60

DATA_CACHE_TTL_SECONDS: int = 60 * 10
historical_data_cache: Dict[str, Dict[str, Any]] = {}
data_cache_lock = Lock()

API_WEIGHT_LIMIT_PER_MINUTE: int = 1100
api_weight_used: int = 0
api_weight_period_start: float = time.time()
api_weight_lock = Lock()

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
last_monitor_api_check: float = 0

REJECTION_REASONS_AR = {
    "Filters Not Loaded": "الفلاتر غير محملة", "Low Volatility": "تقلب منخفض جداً", "BTC Correlation": "ارتباط ضعيف بالبيتكوين",
    "RRR Filter": "نسبة المخاطرة/العائد غير كافية", "Peak Filter": "فلتر القمة (السعر قريب من القمة)", "Invalid ATR for TP/SL": "ATR غير صالح لحساب الأهداف",
    "Momentum ADX": "فلتر الزخم (ADX ضعيف)", "Momentum Rel Vol": "فلتر الزخم (حجم نسبي منخفض)", "Momentum RSI": "فلتر الزخم (RSI خارج النطاق)",
    "Momentum ROC": "فلتر الزخم (ROC سلبي أو ضعيف)", "Momentum Slope": "فلتر الزخم (ميل EMA سلبي)", "Reversal Volume Filter": "فوليوم الانعكاس ضعيف",
    "Reversal Signal Rejected by ML Model": "نموذج التعلم الآلي رفض إشارة الانعكاس", "Invalid Position Size": "حجم الصفقة غير صالح (الوقف تحت الدخول)",
    "Lot Size Adjustment Failed": "فشل ضبط حجم العقد (LOT_SIZE)", "Min Notional Filter": "قيمة الصفقة أقل من الحد الأدنى",
    "Insufficient Balance": "الرصيد غير كافٍ", "Order Book Fetch Failed": "فشل جلب دفتر الطلبات", "Order Book Imbalance": "اختلال توازن دفتر الطلبات (ضغط بيع)",
    "Large Sell Wall Detected": "تم كشف جدار بيع ضخم", "API Rate Limited": "تم تجاوز حدود الطلبات (API)", "Price not in Cache": "السعر غير موجود في الذاكرة المؤقتة"
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

def check_and_wait_for_rate_limit(weight: int):
    global api_weight_used, api_weight_period_start
    with api_weight_lock:
        now = time.time()
        if now - api_weight_period_start > 60:
            logger.debug(f"Resetting API weight counter. Last minute usage: {api_weight_used}")
            api_weight_used = 0
            api_weight_period_start = now
        if api_weight_used + weight > API_WEIGHT_LIMIT_PER_MINUTE:
            sleep_time = 60 - (now - api_weight_period_start) + 1
            logger.warning(f"Approaching API rate limit ({api_weight_used}+{weight}). Pausing for {sleep_time:.2f} seconds.")
            time.sleep(sleep_time)
            api_weight_used = weight
            api_weight_period_start = time.time()
        else:
            api_weight_used += weight
        logger.debug(f"API weight used: {api_weight_used}/{API_WEIGHT_LIMIT_PER_MINUTE}")

def handle_binance_api_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (BinanceAPIException, BinanceRequestException) as e:
            logger.error(f"❌ Binance API Error in {func.__name__}: {e.code} - {e.message}", exc_info=False)
            if e.code == -1003:
                logger.critical(f"🚨 IP BANNED by Binance (Code -1003). The bot will continue running using WebSockets but API calls will fail.")
                log_and_notify("critical", "IP BANNED by Binance. API calls are blocked.", "API_BAN")
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
    <title>لوحة تحكم التداول V27.7</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
    <style>
        :root { 
            --bg-main: #0D1117; 
            --bg-card: #161B22; 
            --border-color: #30363D; 
            --text-primary: #E6EDF3; 
            --text-secondary: #848D97; 
            --accent-blue: #58A6FF; 
            --accent-green: #3FB950; 
            --accent-red: #F85149; 
            --accent-yellow: #D29922;
            --shadow-green: rgba(63, 185, 80, 0.5);
            --shadow-yellow: rgba(210, 153, 34, 0.5);
            --shadow-red: rgba(248, 81, 73, 0.5);
        }
        body { font-family: 'Tajawal', sans-serif; background-color: var(--bg-main); color: var(--text-primary); }
        .card { background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 0.75rem; }
        .skeleton { animation: pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite; background-color: #21262d; border-radius: 0.5rem; }
        @keyframes pulse { 50% { opacity: .6; } }
        .tab-btn { position: relative; transition: color 0.2s ease; color: var(--text-secondary); }
        .tab-btn.active { color: var(--accent-blue); }
        .tab-btn.active::after { content: ''; position: absolute; bottom: -8px; left: 0; right: 0; height: 2px; background-color: var(--accent-blue); border-radius: 2px; }
        .toggle-bg:after { content: ''; position: absolute; top: 2px; left: 2px; background: white; border-radius: 9999px; height: 1.25rem; width: 1.25rem; transition: transform 0.2s ease-in-out; }
        input:checked + .toggle-bg:after { transform: translateX(100%); }
        input:checked + .toggle-bg { background-color: var(--accent-green); }
        .market-light { width: 1.25rem; height: 1.25rem; border-radius: 9999px; transition: background-color 0.5s ease, box-shadow 0.5s ease; }
        .light-green { background-color: var(--accent-green); box-shadow: 0 0 12px 3px var(--shadow-green); }
        .light-red { background-color: var(--accent-red); box-shadow: 0 0 12px 3px var(--shadow-red); }
        .light-yellow { background-color: var(--accent-yellow); box-shadow: 0 0 12px 3px var(--shadow-yellow); }
        .light-off { background-color: #30363D; }
        .status-dot { width: 0.75rem; height: 0.75rem; border-radius: 9999px; }
        .tooltip { position: relative; display: inline-block; }
        .tooltip .tooltiptext { visibility: hidden; width: 220px; background-color: var(--bg-card); color: var(--text-primary); text-align: center; border-radius: 6px; padding: 8px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -110px; opacity: 0; transition: opacity 0.3s; border: 1px solid var(--border-color); }
        .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
    </style>
</head>
<body class="p-4 md:p-6">
    <div class="max-w-7xl mx-auto">
        <!-- Header -->
        <header class="flex flex-col md:flex-row justify-between items-center mb-6">
            <h1 class="text-3xl font-bold text-white mb-4 md:mb-0">لوحة تحكم التداول <span class="text-sm font-mono text-gray-500">v27.7</span></h1>
            <div class="card p-3 flex items-center space-x-4 space-x-reverse">
                <div id="main-status-light" class="status-dot bg-gray-500"></div>
                <span id="main-status-text" class="text-lg font-semibold">جاري التهيئة...</span>
                <div class="w-px h-6 bg-gray-600"></div>
                <label for="toggleTrading" class="flex items-center cursor-pointer">
                    <span class="mr-3 text-sm font-medium text-gray-300">التداول</span>
                    <div class="relative">
                        <input type="checkbox" id="toggleTrading" class="sr-only">
                        <div class="block bg-gray-600 w-10 h-6 rounded-full toggle-bg"></div>
                    </div>
                </label>
            </div>
        </header>

        <!-- Key Metrics -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <div class="card p-4 flex flex-col justify-between">
                <div class="flex justify-between items-center">
                    <h2 class="text-lg font-semibold text-gray-400">حالة السوق</h2>
                    <div id="market-trend-light" class="market-light light-off"></div>
                </div>
                <p id="market-trend-label" class="text-2xl font-bold mt-2">--</p>
                <p id="market-trend-details" class="text-sm text-gray-500 mt-1">&nbsp;</p>
            </div>
            <div class="card p-4 flex flex-col justify-between">
                <h2 class="text-lg font-semibold text-gray-400">الخوف والطمع</h2>
                <div class="relative w-full h-24">
                    <canvas id="fng-chart"></canvas>
                    <div id="fng-value" class="absolute inset-0 flex items-center justify-center text-3xl font-bold">--</div>
                </div>
                <p id="fng-classification" class="text-center text-sm text-gray-500 mt-1">&nbsp;</p>
            </div>
            <div class="card p-4">
                <h2 class="text-lg font-semibold text-gray-400">سيولة السوق</h2>
                <p id="session-state-label" class="text-2xl font-bold mt-2">--</p>
                <p id="session-active-sessions" class="text-sm text-gray-500 mt-1">&nbsp;</p>
            </div>
            <div class="card p-4">
                <h2 class="text-lg font-semibold text-gray-400">الصفقات المفتوحة</h2>
                <p id="open-trades-count" class="text-5xl font-bold text-center mt-2">--</p>
                <p class="text-center text-sm text-gray-500">من أصل <span id="max-open-trades"></span></p>
            </div>
        </div>

        <!-- Main Content -->
        <div class="card p-4">
            <div class="border-b border-gray-700 mb-4">
                <nav class="flex space-x-8 space-x-reverse -mb-px" id="tabs">
                    <button data-tab="open-trades" class="tab-btn py-4 px-1 text-base font-medium active">الصفقات المفتوحة</button>
                    <button data-tab="rejections" class="tab-btn py-4 px-1 text-base font-medium">الإشارات المرفوضة</button>
                    <button data-tab="notifications" class="tab-btn py-4 px-1 text-base font-medium">إشعارات النظام</button>
                    <button data-tab="config" class="tab-btn py-4 px-1 text-base font-medium">الإعدادات</button>
                </nav>
            </div>
            <div id="tab-content">
                <!-- Open Trades Tab -->
                <div id="open-trades-content" class="tab-pane">
                    <div class="overflow-x-auto">
                        <table class="w-full text-sm text-left text-gray-400">
                            <thead class="text-xs text-gray-300 uppercase bg-gray-700/20">
                                <tr>
                                    <th scope="col" class="px-4 py-3">الرمز</th>
                                    <th scope="col" class="px-4 py-3">سعر الدخول</th>
                                    <th scope="col" class="px-4 py-3">السعر الحالي</th>
                                    <th scope="col" class="px-4 py-3">الربح/الخسارة %</th>
                                    <th scope="col" class="px-4 py-3">الهدف</th>
                                    <th scope="col" class="px-4 py-3">وقف الخسارة</th>
                                </tr>
                            </thead>
                            <tbody id="open-trades-table">
                                <!-- Rows will be inserted here by JS -->
                            </tbody>
                        </table>
                    </div>
                </div>
                <!-- Rejections Tab -->
                <div id="rejections-content" class="tab-pane hidden h-96 overflow-y-auto">
                    <ul id="rejection-logs-list" class="space-y-3">
                        <!-- Items will be inserted here by JS -->
                    </ul>
                </div>
                <!-- Notifications Tab -->
                <div id="notifications-content" class="tab-pane hidden h-96 overflow-y-auto">
                     <ul id="notifications-list" class="space-y-3">
                        <!-- Items will be inserted here by JS -->
                    </ul>
                </div>
                 <!-- Config Tab -->
                <div id="config-content" class="tab-pane hidden">
                    <div id="config-list" class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 text-sm">
                        <!-- Items will be inserted here by JS -->
                    </div>
                </div>
            </div>
        </div>
    </div>

<script>
let fngChart = null;

function initializeFngChart() {
    const ctx = document.getElementById('fng-chart').getContext('2d');
    fngChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [50, 50],
                backgroundColor: ['#4A5568', '#2D3748'],
                borderColor: 'var(--bg-card)',
                circumference: 180,
                rotation: 270,
                cutout: '70%',
                borderRadius: 5,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { tooltip: { enabled: false }, legend: { display: false } },
            animation: { animateRotate: false }
        }
    });
}

function updateFngChart(value) {
    if (!fngChart) return;
    const percentage = value / 100;
    fngChart.data.datasets[0].data = [value, 100 - value];
    
    const green = [63, 185, 80];
    const red = [248, 81, 73];
    const color = red.map((c, i) => Math.round(c + (green[i] - c) * percentage));
    
    fngChart.data.datasets[0].backgroundColor[0] = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    fngChart.update('none');
}

function updateUI(data) {
    // Main Status
    const statusLight = document.getElementById('main-status-light');
    const statusText = document.getElementById('main-status-text');
    const toggleTrading = document.getElementById('toggleTrading');
    
    if (data.is_bot_running) {
        statusLight.className = 'status-dot bg-green-500 animate-pulse';
        statusText.textContent = data.is_trading_enabled ? 'التداول نشط' : 'البوت يعمل';
        statusText.className = data.is_trading_enabled ? 'text-lg font-semibold text-green-400' : 'text-lg font-semibold text-yellow-400';
        toggleTrading.checked = data.is_trading_enabled;
    } else {
        statusLight.className = 'status-dot bg-red-500';
        statusText.textContent = 'البوت متوقف';
        statusText.className = 'text-lg font-semibold text-red-500';
        toggleTrading.checked = false;
    }

    // Market Trend
    const trendLight = document.getElementById('market-trend-light');
    const trendLabel = document.getElementById('market-trend-label');
    const trendDetails = document.getElementById('market-trend-details');
    trendLabel.textContent = data.market_state.trend_label_ar || '--';
    let trendColorClass = 'light-off';
    if (data.market_state.trend_label.toLowerCase().includes('up')) {
        trendColorClass = 'light-green';
    } else if (data.market_state.trend_label.toLowerCase().includes('down')) {
        trendColorClass = 'light-red';
    } else if (data.market_state.trend_label.toLowerCase().includes('ranging')) {
        trendColorClass = 'light-yellow';
    }
    trendLight.className = `market-light ${trendColorClass}`;
    const tfDetails = Object.entries(data.market_state.details_by_tf)
        .map(([tf, details]) => `${tf}: ${details.label_ar}`)
        .join(' | ');
    trendDetails.textContent = tfDetails || 'جاري تحليل الأطر الزمنية...';

    // F&G Index
    document.getElementById('fng-value').textContent = data.fng_index.value > 0 ? data.fng_index.value : '--';
    document.getElementById('fng-classification').textContent = data.fng_index.classification_ar || 'غير متاح';
    if(data.fng_index.value > 0) updateFngChart(data.fng_index.value);

    // Session State
    document.getElementById('session-state-label').textContent = data.session_state.description_ar || '--';
    document.getElementById('session-active-sessions').textContent = data.session_state.active_sessions.length > 0 ? `الجلسات النشطة: ${data.session_state.active_sessions.join(', ')}` : 'لا توجد جلسات رئيسية';

    // Open Trades
    document.getElementById('open-trades-count').textContent = data.open_trades.length;
    document.getElementById('max-open-trades').textContent = data.config.max_open_trades;
    const tradesTable = document.getElementById('open-trades-table');
    tradesTable.innerHTML = ''; // Clear old data
    if (data.open_trades.length === 0) {
        tradesTable.innerHTML = '<tr><td colspan="6" class="text-center py-8 text-gray-500">لا توجد صفقات مفتوحة حاليًا.</td></tr>';
    } else {
        data.open_trades.forEach(trade => {
            const pnl = trade.pnl_pct || 0;
            const pnlColor = pnl >= 0 ? 'text-green-400' : 'text-red-400';
            const row = `
                <tr class="border-b border-gray-700 hover:bg-gray-800/50">
                    <td class="px-4 py-3 font-bold">${trade.symbol}</td>
                    <td class="px-4 py-3">${parseFloat(trade.entry_price).toFixed(4)}</td>
                    <td class="px-4 py-3">${parseFloat(trade.current_price || 0).toFixed(4)}</td>
                    <td class="px-4 py-3 font-semibold ${pnlColor}">${pnl.toFixed(2)}%</td>
                    <td class="px-4 py-3">${parseFloat(trade.target_price).toFixed(4)}</td>
                    <td class="px-4 py-3">${parseFloat(trade.stop_loss).toFixed(4)}</td>
                </tr>`;
            tradesTable.innerHTML += row;
        });
    }

    // Rejection Logs
    const rejectionsList = document.getElementById('rejection-logs-list');
    rejectionsList.innerHTML = '';
    if (data.rejection_logs.length === 0) {
        rejectionsList.innerHTML = '<li class="text-center py-8 text-gray-500">لا توجد سجلات رفض.</li>';
    } else {
        data.rejection_logs.forEach(log => {
            const details = Object.entries(log.details).map(([k, v]) => `${k}: ${v}`).join(', ');
            const item = `
                <li class="flex items-start space-x-3 space-x-reverse">
                    <div class="text-red-500 mt-1">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" /></svg>
                    </div>
                    <div class="flex-1">
                        <p class="font-semibold">${log.symbol} - <span class="text-red-400">${log.reason}</span></p>
                        <p class="text-xs text-gray-400">${details}</p>
                        <p class="text-xs text-gray-500">${new Date(log.timestamp).toLocaleString('ar-EG')}</p>
                    </div>
                </li>`;
            rejectionsList.innerHTML += item;
        });
    }

    // Notifications
    const notificationsList = document.getElementById('notifications-list');
    notificationsList.innerHTML = '';
     if (data.notifications.length === 0) {
        notificationsList.innerHTML = '<li class="text-center py-8 text-gray-500">لا توجد إشعارات.</li>';
    } else {
        data.notifications.forEach(n => {
            let icon, color;
            switch(n.type.toLowerCase()) {
                case 'system': icon = '<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd" /></svg>'; color = 'text-blue-400'; break;
                case 'signal_open': icon = '<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-8.707l-3-3a1 1 0 00-1.414 0l-3 3a1 1 0 001.414 1.414L9 9.414V13a1 1 0 102 0V9.414l1.293 1.293a1 1 0 001.414-1.414z" clip-rule="evenodd" /></svg>'; color = 'text-green-400'; break;
                case 'signal_close': icon = '<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-11a1 1 0 10-2 0v3.586L7.707 9.293a1 1 0 00-1.414 1.414l3 3a1 1 0 001.414 0l3-3a1 1 0 00-1.414-1.414L11 10.586V7z" clip-rule="evenodd" /></svg>'; color = 'text-yellow-400'; break;
                default: icon = '<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path d="M2.003 5.884L10 2.882l7.997 3.002A2 2 0 0119 7.818V12a2 2 0 01-2 2h-2.167l-3.2-3.2a1 1 0 00-1.414 1.414l4.242 4.242a1 1 0 01.293.707V17a2 2 0 01-2 2H5a2 2 0 01-2-2v-1.172a1 1 0 01.293-.707l4.242-4.242a1 1 0 10-1.414-1.414L3.167 14H1V7.818a2 2 0 011.003-1.934z" /></svg>'; color = 'text-gray-400';
            }
            const item = `
                <li class="flex items-start space-x-3 space-x-reverse">
                    <div class="${color} mt-1">${icon}</div>
                    <div class="flex-1">
                        <p class="text-sm">${n.message}</p>
                        <p class="text-xs text-gray-500">${new Date(n.timestamp).toLocaleString('ar-EG')}</p>
                    </div>
                </li>`;
            notificationsList.innerHTML += item;
        });
    }

    // Config
    const configList = document.getElementById('config-list');
    configList.innerHTML = '';
    for (const [key, value] of Object.entries(data.config)) {
        const item = `
            <div class="bg-gray-800/50 p-3 rounded-lg">
                <p class="text-xs text-gray-400">${key}</p>
                <p class="font-mono font-bold text-blue-300">${value}</p>
            </div>`;
        configList.innerHTML += item;
    }
}

async function fetchData() {
    try {
        const response = await fetch('/api/dashboard_data');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        updateUI(data);
    } catch (error) {
        console.error("Could not fetch dashboard data:", error);
        const statusText = document.getElementById('main-status-text');
        statusText.textContent = 'خطأ في الاتصال';
        statusText.className = 'text-lg font-semibold text-red-500';
    }
}

async function toggleTradingAction() {
    try {
        const response = await fetch('/api/toggle_trading', { method: 'POST' });
        const result = await response.json();
        console.log(result.message);
        fetchData(); // Refresh data immediately after action
    } catch (error) {
        console.error("Could not toggle trading status:", error);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initializeFngChart();
    fetchData();
    setInterval(fetchData, 3000);

    document.getElementById('toggleTrading').addEventListener('change', toggleTradingAction);
    
    const tabs = document.getElementById('tabs');
    const tabContent = document.getElementById('tab-content');
    tabs.addEventListener('click', (e) => {
        if (e.target.tagName === 'BUTTON') {
            const tabName = e.target.dataset.tab;
            
            // Update button styles
            tabs.querySelectorAll('button').forEach(btn => btn.classList.remove('active'));
            e.target.classList.add('active');

            // Update content visibility
            tabContent.querySelectorAll('.tab-pane').forEach(pane => pane.classList.add('hidden'));
            document.getElementById(`${tabName}-content`).classList.remove('hidden');
        }
    });
});
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
        new_notification = {"timestamp": datetime.now(timezone.utc).isoformat(), "type": notification_type, "message": message}
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

# --- [مُضافة] الدوال المفقودة لتحميل البيانات الأولية ---
def load_open_signals_to_cache():
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status IN ('open', 'updated');")
            open_signals = cur.fetchall()
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in open_signals: open_signals_cache[signal['symbol']] = dict(signal)
            logger.info(f"✅ [Loading] Loaded {len(open_signals)} open signals from DB.")
    except Exception as e: logger.error(f"❌ [Loading] Failed to load open signals: {e}")

def load_notifications_to_cache():
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 50;")
            recent = cur.fetchall()
            with notifications_lock:
                notifications_cache.clear()
                for n in reversed(recent):
                    n['timestamp'] = n['timestamp'].isoformat()
                    notifications_cache.appendleft(dict(n))
            logger.info(f"✅ [Loading] Loaded {len(notifications_cache)} notifications from DB.")
    except Exception as e: logger.error(f"❌ [Loading] Failed to load notifications: {e}")

# ---------------------- دوال Binance والبيانات ----------------------
@handle_binance_api_errors
def get_exchange_info_map_call() -> Optional[Dict]:
    check_and_wait_for_rate_limit(weight=10)
    return client.get_exchange_info()

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
def fetch_historical_data_from_api(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    check_and_wait_for_rate_limit(weight=1)
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

def get_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    cache_key = f"{symbol}_{interval}"
    now = time.time()
    with data_cache_lock:
        if cache_key in historical_data_cache:
            cached_item = historical_data_cache[cache_key]
            if now - cached_item['timestamp'] < DATA_CACHE_TTL_SECONDS:
                logger.debug(f"✅ [Cache HIT] Using cached data for {cache_key}.")
                return cached_item['data'].copy()
    
    logger.debug(f"⏳ [Cache MISS] Fetching new historical data for {cache_key}.")
    df = fetch_historical_data_from_api(symbol, interval, days)
    
    if df is not None and not df.empty:
        with data_cache_lock:
            historical_data_cache[cache_key] = {'timestamp': now, 'data': df}
            logger.debug(f"💾 [Cache SET] Stored new data for {cache_key}.")
        return df.copy()
    return None

@handle_binance_api_errors
def analyze_order_book(symbol: str, entry_price: float) -> Optional[Dict[str, Any]]:
    check_and_wait_for_rate_limit(weight=1)
    ob = client.get_order_book(symbol=symbol, limit=ORDER_BOOK_DEPTH_LIMIT)
    bids = pd.DataFrame(ob['bids'], columns=['price', 'quantity'], dtype=float)
    asks = pd.DataFrame(ob['asks'], columns=['price', 'quantity'], dtype=float)
    price_range = ORDER_BOOK_ANALYSIS_RANGE_PCT * entry_price
    nearby_bids_vol = bids[bids['price'] >= entry_price - price_range]['quantity'].sum()
    nearby_asks_vol = asks[asks['price'] <= entry_price + price_range]['quantity'].sum()
    bid_ask_ratio = nearby_bids_vol / nearby_asks_vol if nearby_asks_vol > 0 else float('inf')
    avg_ask_size = asks['quantity'].mean()
    large_sell_walls = asks[asks['quantity'] > avg_ask_size * ORDER_BOOK_WALL_MULTIPLIER]
    return { "bid_ask_ratio": bid_ask_ratio, "has_large_sell_wall": not large_sell_walls.empty, "wall_details": large_sell_walls.to_dict('records') if not large_sell_walls.empty else [] }

# ---------------------- دوال حساب الميزات وتحديد الاتجاه ----------------------
def calculate_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    df['returns'] = df['close'].pct_change()
    try:
        import talib
        df['adx'] = pd.Series(talib.ADX(df['high'], df['low'], df['close'], timeperiod=ADX_PERIOD))
        df['rsi'] = pd.Series(talib.RSI(df['close'], timeperiod=RSI_PERIOD))
        df['atr'] = pd.Series(talib.ATR(df['high'], df['low'], df['close'], timeperiod=ATR_PERIOD))
        df['roc'] = pd.Series(talib.ROC(df['close'], timeperiod=MOMENTUM_PERIOD))
    except ImportError:
        logger.warning("TA-Lib not found. Some features will be unavailable.")
        for col in ['adx', 'rsi', 'atr', 'roc']: df[col] = 0.0
    df['rel_vol'] = df['volume'] / df['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean()
    ema21 = df['close'].ewm(span=21, adjust=False).mean()
    df['slope'] = (ema21 - ema21.shift(EMA_SLOPE_PERIOD)) / EMA_SLOPE_PERIOD
    if btc_df is not None and 'btc_returns' in btc_df.columns:
        df = df.join(btc_df['btc_returns'])
        df['btc_correlation'] = df['returns'].rolling(window=50).corr(df['btc_returns'])
    else:
        df['btc_correlation'] = 0.0
    return df.dropna()

def determine_market_trend_score():
    global current_market_state, last_market_state_check
    with market_state_lock:
        if time.time() - last_market_state_check < 300: return
    logger.info("🧠 [Market Score] Updating multi-timeframe trend score...")
    try:
        total_score, details, tf_weights = 0, {}, {'15m': 0.2, '1h': 0.3, '4h': 0.5}
        for tf in TIMEFRAMES_FOR_TREND_ANALYSIS:
            days = 5 if tf == '15m' else (15 if tf == '1h' else 50)
            df = get_historical_data(BTC_SYMBOL, tf, days)
            if df is None or len(df) < EMA_PERIODS[-1]:
                details[tf] = {"score": 0, "label_ar": "غير واضح", "reason": "بيانات غير كافية"}; continue
            for p in EMA_PERIODS: df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()
            last = df.iloc[-1]
            close, ema21, ema50, ema200 = last['close'], last['ema_21'], last['ema_50'], last['ema_200']
            tf_score = (1 if close > ema21 else -1) + (1 if ema21 > ema50 else -1) + (1 if ema50 > ema200 else -1)
            label_ar = "صاعد" if tf_score >= 2 else ("هابط" if tf_score <= -2 else "محايد")
            details[tf] = {"score": tf_score, "label_ar": label_ar, "reason": f"E21:{ema21:.2f},E50:{ema50:.2f},E200:{ema200:.2f}"}
            total_score += tf_score * tf_weights[tf]
        final_score = round(total_score)
        
        trend_label, trend_label_ar = "RANGING", "محايد"
        if final_score >= 4: trend_label, trend_label_ar = "STRONG_UPTREND", "صاعد قوي"
        elif final_score >= 1: trend_label, trend_label_ar = "UPTREND", "صاعد"
        elif final_score <= -4: trend_label, trend_label_ar = "STRONG_DOWNTREND", "هابط قوي"
        elif final_score <= -1: trend_label, trend_label_ar = "DOWNTREND", "هابط"

        with market_state_lock:
            current_market_state = {"trend_score": final_score, "trend_label": trend_label, "trend_label_ar": trend_label_ar, "details_by_tf": details, "last_updated": datetime.now(timezone.utc).isoformat()}
            last_market_state_check = time.time()
        logger.info(f"✅ [Market Score] New State: Score={final_score}, Label='{trend_label_ar}'")
    except Exception as e:
        logger.error(f"❌ [Market Score] Failed to determine market state: {e}", exc_info=True)
        with market_state_lock: current_market_state.update({'trend_score': 0, 'trend_label': "UNKNOWN", 'trend_label_ar': "غير واضح"})

# --- دوال الاستراتيجية والتداول الحقيقي (ستتم إضافتها هنا) ---
# ...
def get_btc_data_for_bot() -> Optional[pd.DataFrame]:
    btc_data = get_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
    if btc_data is not None: btc_data['btc_returns'] = btc_data['close'].pct_change()
    return btc_data

def load_ml_model_bundle_from_folder(symbol: str) -> Optional[Dict[str, Any]]:
    global ml_models_cache
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_name in ml_models_cache:
        return ml_models_cache[model_name]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir_path = os.path.join(script_dir, MODEL_FOLDER)
    model_path = os.path.join(model_dir_path, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        logger.debug(f"⚠️ [ML Model] Model file not found: '{model_path}'.")
        return None
    try:
        with open(model_path, 'rb') as f:
            model_bundle = pickle.load(f)
        if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
            ml_models_cache[model_name] = model_bundle
            return model_bundle
        return None
    except Exception as e:
        logger.error(f"❌ [ML Model] Error loading model for {symbol}: {e}", exc_info=True)
        return None

class TradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_bundle_from_folder(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    def generate_buy_signal(self, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]) or df_features.empty: return None
        try:
            last_row_ordered_df = df_features.iloc[[-1]][self.feature_names]
            features_scaled_np = self.scaler.transform(last_row_ordered_df)
            features_scaled_df = pd.DataFrame(features_scaled_np, columns=self.feature_names)
            prediction = self.ml_model.predict(features_scaled_df)[0]
            if prediction != 1: return None
            
            prediction_proba = self.ml_model.predict_proba(features_scaled_df)
            confidence = float(np.max(prediction_proba[0]))
            logger.debug(f"ℹ️ [{self.symbol}] ML Model predicted 'BUY' with {confidence:.2%} confidence.")
            return {'prediction': int(prediction), 'confidence': confidence}
        except Exception as e:
            logger.warning(f"⚠️ [{self.symbol}] ML Signal Generation Error: {e}")
            return None

def calculate_tp_sl(symbol: str, entry_price: float, last_atr: float) -> Optional[Dict[str, Any]]:
    if last_atr <= 0:
        log_rejection(symbol, "Invalid ATR for TP/SL", {"atr": last_atr})
        return None
    tp = entry_price + (last_atr * ATR_FALLBACK_TP_MULTIPLIER)
    sl = entry_price - (last_atr * ATR_FALLBACK_SL_MULTIPLIER)
    return {'target_price': tp, 'stop_loss': sl, 'source': 'ATR_Fallback'}

def passes_filters(symbol: str, last_features: pd.Series, profile: Dict[str, Any], entry_price: float, tp_sl_data: Dict, df_15m: pd.DataFrame) -> bool:
    filters = profile.get("filters", {})
    if not filters:
        log_rejection(symbol, "Filters Not Loaded", {"profile": profile.get('name')})
        return False

    volatility = (last_features.get('atr', 0) / entry_price * 100) if entry_price > 0 else 0
    if volatility < filters.get('min_volatility_pct', 0.0):
        log_rejection(symbol, "Low Volatility", {"volatility": f"{volatility:.2f}%", "min": f"{filters.get('min_volatility_pct', 0.0):.2f}%"})
        return False

    correlation = last_features.get('btc_correlation', 0)
    if correlation < filters.get('min_btc_correlation', -1.0):
        log_rejection(symbol, "BTC Correlation", {"corr": f"{correlation:.2f}", "min": f"{filters.get('min_btc_correlation', -1.0)}"})
        return False

    risk, reward = entry_price - float(tp_sl_data['stop_loss']), float(tp_sl_data['target_price']) - entry_price
    if risk <= 0 or reward <= 0 or (reward / risk) < filters.get('min_rrr', 0.0):
        log_rejection(symbol, "RRR Filter", {"rrr": f"{(reward/risk):.2f}" if risk > 0 else "N/A", "min": f"{filters.get('min_rrr', 0.0):.2f}"})
        return False

    if profile.get("strategy") == "REVERSAL":
        rel_vol = last_features.get('relative_volume', 0)
        if rel_vol < filters.get('min_relative_volume', 1.5):
            log_rejection(symbol, "Reversal Volume Filter", {"RelVol": f"{rel_vol:.2f}", "min": filters.get('min_relative_volume', 1.5)})
            return False

    elif profile.get("strategy") == "MOMENTUM":
        adx, rel_vol, rsi, roc, slope = last_features.get('adx',0), last_features.get('rel_vol',0), last_features.get('rsi',0), last_features.get('roc',0), last_features.get('slope',0)
        rsi_min, rsi_max = filters.get('rsi_range', (0, 100))
        if not (adx >= filters.get('adx', 0) and rel_vol >= filters.get('rel_vol', 0) and rsi_min <= rsi < rsi_max and roc > filters.get('roc', -100) and slope > filters.get('slope', -100)):
            log_rejection(symbol, "Momentum/Strength Filter", {"ADX":f"{adx:.2f}", "Vol":f"{rel_vol:.2f}", "RSI":f"{rsi:.2f}", "ROC":f"{roc:.2f}", "Slope":f"{slope:.4f}"})
            return False
    return True

def passes_order_book_check(symbol: str, order_book_analysis: Dict, profile: Dict) -> bool:
    filters = profile.get("filters", {})
    min_ratio = filters.get('min_bid_ask_ratio', 1.0)
    if order_book_analysis.get('has_large_sell_wall', True):
        log_rejection(symbol, "Large Sell Wall Detected", {"details": order_book_analysis.get('wall_details')})
        return False
    bid_ask_ratio = order_book_analysis.get('bid_ask_ratio', 0)
    if bid_ask_ratio < min_ratio:
        log_rejection(symbol, "Order Book Imbalance", {"ratio": f"{bid_ask_ratio:.2f}", "min_required": min_ratio})
        return False
    return True

def initiate_signal_closure(symbol: str, signal_to_close: Dict, status: str, closing_price: float):
    signal_id = signal_to_close.get('id')
    if not signal_id: return
    with closure_lock:
        if signal_id in signals_pending_closure: return
        signals_pending_closure.add(signal_id)
    with signal_cache_lock: open_signals_cache.pop(symbol, None)
    logger.info(f"ℹ️ [Closure] Starting closure thread for signal {signal_id} ({symbol}) with status '{status}'.")
    Thread(target=close_signal, args=(signal_to_close, status, closing_price)).start()

def close_signal(signal: Dict, status: str, closing_price: float):
    # ... (This function remains largely the same)
    pass

# ---------------------- WebSocket & Main Loops ----------------------
def handle_price_update_message(msg: Dict[str, Any]) -> None:
    if not redis_client or 'e' not in msg or msg['e'] != '24hrMiniTicker': return
    data = msg.get('data')
    if not isinstance(data, list): return
    try:
        pipeline = redis_client.pipeline()
        for item in data:
            symbol = item.get('s')
            close_price = item.get('c')
            if symbol and close_price:
                pipeline.hset(REDIS_PRICES_HASH_NAME, symbol, float(close_price))
        pipeline.execute()
    except Exception as e:
        logger.error(f"❌ [WebSocket Price Updater] Error processing message: {e}", exc_info=False)

def trade_monitoring_loop():
    global last_monitor_api_check
    logger.info("✅ [Trade Monitor] Starting trade monitoring loop.")
    while True:
        try:
            with signal_cache_lock:
                if not open_signals_cache:
                    time.sleep(1); continue
                signals_to_check = dict(open_signals_cache)

            if not redis_client or not client:
                time.sleep(1); continue

            symbols_to_fetch = list(signals_to_check.keys())
            redis_prices_list = redis_client.hmget(REDIS_PRICES_HASH_NAME, symbols_to_fetch)
            current_prices = {symbol: price for symbol, price in zip(symbols_to_fetch, redis_prices_list)}

            perform_direct_api_check = (time.time() - last_monitor_api_check) > MONITOR_API_CHECK_INTERVAL
            if perform_direct_api_check:
                logger.debug("[Trade Monitor] Performing periodic direct API price check...")
                try:
                    check_and_wait_for_rate_limit(weight=1)
                    tickers = client.get_symbol_ticker()
                    api_prices = {t['symbol']: t['price'] for t in tickers if t['symbol'] in symbols_to_fetch}
                    current_prices.update(api_prices)
                    last_monitor_api_check = time.time()
                except Exception as e:
                    logger.warning(f"⚠️ [Trade Monitor] Could not perform API price check: {e}")

            for symbol, signal in signals_to_check.items():
                signal_id = signal.get('id')
                if not signal_id: continue
                with closure_lock:
                    if signal_id in signals_pending_closure: continue

                price_str = current_prices.get(symbol)
                if not price_str: continue
                try:
                    price = float(price_str)
                except (ValueError, TypeError):
                    continue
                
                entry_price = float(signal['entry_price'])
                with signal_cache_lock:
                    if symbol in open_signals_cache:
                        open_signals_cache[symbol]['current_price'] = price
                        open_signals_cache[symbol]['pnl_pct'] = ((price / entry_price) - 1) * 100
                
                target_price = float(signal.get('target_price', 0))
                original_stop_loss = float(signal.get('stop_loss', 0))
                effective_stop_loss = original_stop_loss
                
                if USE_TRAILING_STOP_LOSS:
                    activation_price = entry_price * (1 + TRAILING_ACTIVATION_PROFIT_PERCENT / 100)
                    if price > activation_price:
                        current_peak = float(signal.get('current_peak_price', entry_price))
                        if price > current_peak:
                            with signal_cache_lock:
                                if symbol in open_signals_cache: open_signals_cache[symbol]['current_peak_price'] = price
                            current_peak = price
                        trailing_stop_price = current_peak * (1 - TRAILING_DISTANCE_PERCENT / 100)
                        if trailing_stop_price > effective_stop_loss:
                            effective_stop_loss = trailing_stop_price

                status_to_set = None
                if price >= target_price: status_to_set = 'target_hit'
                elif price <= effective_stop_loss: status_to_set = 'stop_loss_hit'
                
                if status_to_set:
                    logger.info(f"✅ [TRIGGER] ID:{signal_id} | {symbol} | Condition '{status_to_set}' met at price {price}.")
                    initiate_signal_closure(symbol, signal, status_to_set, price)
            
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"❌ [Trade Monitor] Critical error: {e}", exc_info=True)
            time.sleep(5)

def main_loop():
    logger.info("[Main Loop] Waiting for initialization...")
    time.sleep(20)
    if not validated_symbols_to_scan:
        log_and_notify("critical", "No validated symbols to scan. Bot will not start.", "SYSTEM"); return
    
    log_and_notify("info", f"✅ Starting main scan loop for {len(validated_symbols_to_scan)} symbols.", "SYSTEM")
    
    while True:
        try:
            logger.info("🔄 Starting new main cycle...")
            determine_market_trend_score()
            # analyze_market_and_create_dynamic_profile() # Assuming this function exists
            filter_profile = FILTER_PROFILES["UPTREND"] # Simplified for now
            active_strategy_type = filter_profile.get("strategy")

            if not active_strategy_type or active_strategy_type == "DISABLED":
                logger.warning(f"🛑 Trading disabled by profile. Skipping cycle."); time.sleep(300); continue

            if redis_client:
                num_prices = redis_client.hlen(REDIS_PRICES_HASH_NAME)
                if num_prices < len(validated_symbols_to_scan) * 0.7:
                    logger.warning(f"⚠️ [Main Loop] Redis price cache is not fully populated ({num_prices}/{len(validated_symbols_to_scan)}). Waiting...")
                    time.sleep(30); continue
            
            btc_data = get_btc_data_for_bot()
            if btc_data is None: logger.warning("⚠️ Could not get BTC data, some features will be disabled."); time.sleep(60); continue
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            all_symbols_with_models = [s for s in validated_symbols_to_scan if os.path.exists(os.path.join(script_dir, MODEL_FOLDER, f"{BASE_ML_MODEL_NAME}_{s}.pkl"))]
            if not all_symbols_with_models: logger.warning("⚠️ No symbols with models found. Skipping scan cycle."); time.sleep(300); continue

            random.shuffle(all_symbols_with_models)
            
            for symbol in all_symbols_with_models:
                try:
                    with trading_status_lock:
                        if not is_trading_enabled: continue
                    with signal_cache_lock:
                        if symbol in open_signals_cache or len(open_signals_cache) >= MAX_OPEN_TRADES: continue
                    
                    model_bundle = load_ml_model_bundle_from_folder(symbol)
                    if not model_bundle: continue

                    df_15m = get_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_15m is None or df_15m.empty: continue
                    
                    if not redis_client: continue
                    entry_price_str = redis_client.hget(REDIS_PRICES_HASH_NAME, symbol)
                    if not entry_price_str:
                        log_rejection(symbol, "Price not in Cache", {"detail": "WebSocket might be lagging."})
                        continue
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
                    
                    # ... (Logic for creating and placing orders) ...

                except Exception as e:
                    logger.error(f"❌ [Processing Error] for symbol {symbol}: {e}", exc_info=True)
                finally:
                    time.sleep(0.1)
            
            logger.info("✅ [End of Cycle] Full scan of all symbols finished. Waiting for 90 seconds..."); time.sleep(90)

        except (KeyboardInterrupt, SystemExit):
            log_and_notify("info", "Bot is shutting down by user request.", "SYSTEM"); break
        except Exception as main_err:
            log_and_notify("error", f"Critical error in main loop: {main_err}", "SYSTEM"); time.sleep(120)

# ---------------------- واجهة برمجة تطبيقات Flask ----------------------
app = Flask(__name__)
CORS(app)

@app.route('/')
def dashboard():
    return get_dashboard_html()

@app.route('/api/dashboard_data')
def get_dashboard_data():
    with signal_cache_lock:
        open_trades_list = list(open_signals_cache.values())
    
    with notifications_lock:
        notifications_list = list(notifications_cache)

    with rejection_logs_lock:
        rejection_logs_list = list(rejection_logs_cache)

    with market_state_lock:
        market_state_copy = dict(current_market_state)
    
    fng_index = get_fear_and_greed_index()
    fng_index_ar = fng_index.copy()
    fng_index_ar['classification_ar'] = {
        "Extreme Fear": "خوف شديد", "Fear": "خوف", "Neutral": "محايد",
        "Greed": "طمع", "Extreme Greed": "طمع شديد"
    }.get(fng_index.get('classification'), fng_index.get('classification'))

    sessions, _, session_desc = get_session_state()
    session_state_data = {
        "active_sessions": sessions,
        "description_ar": session_desc
    }
    
    # Sort trades by symbol for consistent display
    open_trades_list.sort(key=lambda x: x['symbol'])

    data = {
        "is_bot_running": client is not None,
        "is_trading_enabled": is_trading_enabled,
        "market_state": market_state_copy,
        "fng_index": fng_index_ar,
        "session_state": session_state_data,
        "open_trades": open_trades_list,
        "notifications": notifications_list,
        "rejection_logs": rejection_logs_list,
        "config": {
            "max_open_trades": MAX_OPEN_TRADES,
            "risk_per_trade": f"{RISK_PER_TRADE_PERCENT}%",
            "buy_confidence": f"{BUY_CONFIDENCE_THRESHOLD:.0%}",
            "trailing_stop": "يعمل" if USE_TRAILING_STOP_LOSS else "متوقف",
            "monitored_symbols": len(validated_symbols_to_scan),
        }
    }
    return jsonify(data)

@app.route('/api/toggle_trading', methods=['POST'])
def toggle_trading():
    global is_trading_enabled
    with trading_status_lock:
        is_trading_enabled = not is_trading_enabled
        status = "ENABLED" if is_trading_enabled else "DISABLED"
        message = f"Trading has been manually {status} from the dashboard."
        log_and_notify("info", message, "SYSTEM")
        return jsonify({"status": "success", "trading_enabled": is_trading_enabled, "message": message})

# ---------------------- نقطة انطلاق البرنامج ----------------------
def run_websocket_manager():
    if not client or not validated_symbols_to_scan:
        logger.error("❌ [WebSocket] Cannot start: Client or symbols not initialized.")
        return
    logger.info("📡 [WebSocket] Starting WebSocket Manager...")
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    streams = ['!miniTicker@arr'] 
    twm.start_multiplex_socket(callback=handle_price_update_message, streams=streams)
    logger.info(f"✅ [WebSocket] Subscribed to combined mini-ticker stream for all symbols.")
    twm.join()

def initialize_bot_services():
    global client, validated_symbols_to_scan
    logger.info("🤖 [Bot Services] Starting background initialization...")
    try:
        client = Client(API_KEY, API_SECRET)
        init_db(); init_redis()
        get_exchange_info_map()
        load_open_signals_to_cache()
        load_notifications_to_cache()
        
        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan:
            logger.critical("❌ No validated symbols to scan. Bot will not start."); return
        
        Thread(target=run_websocket_manager, daemon=True).start()
        
        logger.info("⏳ Giving WebSocket Manager time to populate Redis (20 seconds)...")
        time.sleep(20)

        # Start main loops
        Thread(target=determine_market_trend_score, daemon=True).start()
        Thread(target=trade_monitoring_loop, daemon=True).start()
        Thread(target=main_loop, daemon=True).start()
        logger.info("✅ [Bot Services] All background services started successfully.")
    except Exception as e:
        log_and_notify("critical", f"A critical error occurred during initialization: {e}", "SYSTEM")
        logger.critical("Critical error during initialization", exc_info=True)
        exit(1)

if __name__ == "__main__":
    logger.info("🚀 LAUNCHING TRADING BOT & DASHBOARD (V27.7 - Dashboard Update) 🚀")
    initialization_thread = Thread(target=initialize_bot_services, daemon=True)
    initialization_thread.start()
    
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"🌍 Dashboard will be available at http://0.0.0.0:{port}")
    
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=port)
    except ImportError:
        logger.warning("Waitress not found, using Flask's development server. Not recommended for production.")
        app.run(host="0.0.0.0", port=port)
