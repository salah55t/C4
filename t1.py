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

# --- Ignore uncritical warnings ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ---------------------- Logging Setup - V27.2 (Improved Logic) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v27_2_arabic_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV27.2')

# ---------------------- Load Environment Variables ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
    REDIS_URL: str = config('REDIS_URL', default='redis://localhost:6379/0')
except Exception as e:
    logger.critical(f"❌ Critical failure loading essential environment variables: {e}")
    exit(1)

# --- [Improvement] ---
# ---------------------- Dynamic Filter Profiles (Adjusted for new logic) ----------------------
# Filters have been adjusted to be more realistic and logical, especially momentum filters.
FILTER_PROFILES: Dict[str, Dict[str, Any]] = {
    "STRONG_UPTREND": {
        "description": "Strong Uptrend (4+ points)",
        "strategy": "MOMENTUM",
        "filters": {
            "adx": 25.0, "rel_vol": 0.2, "rsi_range": (50, 95), "roc": 0.0,
            "slope": 0.0, "min_rrr": 1.3, "min_volatility_pct": 0.25,
            "min_btc_correlation": -0.1, "min_bid_ask_ratio": 1.1
        }
    },
    "UPTREND": {
        "description": "Uptrend (1-3 points)",
        "strategy": "MOMENTUM",
        "filters": {
            "adx": 20.0,
            "rel_vol": 0.1,
            "rsi_range": (48, 90),
            # ROC and Slope adjusted to allow for minor corrections but not sharp declines
            "roc": -0.5,
            "slope": -0.05,
            "min_rrr": 1.4,
            "min_volatility_pct": 0.20,
            "min_btc_correlation": -0.2,
            "min_bid_ask_ratio": 1.1
        }
    },
    "RANGING": {
        "description": "Sideways/Neutral Trend (0 points)",
        "strategy": "MOMENTUM", # Can look for short-term momentum
        "filters": {
            "adx": 18.0, "rel_vol": 0.2, "rsi_range": (45, 75), "roc": 0.05,
            "slope": 0.0, "min_rrr": 1.5, "min_volatility_pct": 0.25,
            "min_btc_correlation": -0.2, "min_bid_ask_ratio": 1.2
        }
    },
    "DOWNTREND": {
        "description": "Downtrend (Reversal Monitoring)",
        "strategy": "REVERSAL",
        "filters": {
            "min_rrr": 2.0, "min_volatility_pct": 0.5, "min_btc_correlation": -0.5,
            "min_relative_volume": 1.5, "min_bid_ask_ratio": 1.5
        }
    },
    "STRONG_DOWNTREND": { "description": "Strong Downtrend (Trading Halted)", "strategy": "DISABLED", "filters": {} },
    "WEEKEND": {
        "description": "Low Liquidity (Weekend)",
        "strategy": "MOMENTUM",
        "filters": {
            "adx": 17.0, "rel_vol": 0.2, "rsi_range": (40, 70), "roc": 0.1,
            "slope": 0.0, "min_rrr": 1.5, "min_volatility_pct": 0.25,
            "min_btc_correlation": -0.4, "min_bid_ask_ratio": 1.4
        }
    }
}


# ---------------------- Constants and Global Variables ----------------------
is_trading_enabled: bool = False
trading_status_lock = Lock()
force_momentum_strategy: bool = False
force_momentum_lock = Lock()
RISK_PER_TRADE_PERCENT: float = 1.0
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V8_With_Momentum'
MODEL_FOLDER: str = 'V8'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
# --- [Improvement] --- Timeframes for trend analysis
TIMEFRAMES_FOR_TREND_ANALYSIS: List[str] = ['15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 30
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices_v8"
DIRECT_API_CHECK_INTERVAL: int = 10
TRADING_FEE_PERCENT: float = 0.1
STATS_TRADE_SIZE_USDT: float = 10.0
BTC_SYMBOL: str = 'BTCUSDT'
SYMBOL_PROCESSING_BATCH_SIZE: int = 50
ADX_PERIOD: int = 14; RSI_PERIOD: int = 14; ATR_PERIOD: int = 14
# --- [Improvement] --- Definition of averages used in trend analysis
EMA_PERIODS: List[int] = [21, 50, 200]
REL_VOL_PERIOD: int = 30; MOMENTUM_PERIOD: int = 12; EMA_SLOPE_PERIOD: int = 5
MAX_OPEN_TRADES: int = 4
BUY_CONFIDENCE_THRESHOLD = 0.80
MIN_CONFIDENCE_INCREASE_FOR_UPDATE = 0.05
ATR_FALLBACK_SL_MULTIPLIER: float = 1.5
ATR_FALLBACK_TP_MULTIPLIER: float = 2.2
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_ACTIVATION_PROFIT_PERCENT: float = 1.0
TRAILING_DISTANCE_PERCENT: float = 0.8
LAST_PEAK_UPDATE_TIME: Dict[int, float] = {}
PEAK_UPDATE_COOLDOWN: int = 60
# --- [Improvement] --- Simplified peak filter settings
USE_PEAK_FILTER: bool = True
PEAK_LOOKBACK_PERIOD: int = 50 # Number of candles to look back
PEAK_DISTANCE_THRESHOLD_PCT: float = 0.995 # Do not buy if price is within the top 0.5% of the peak
DYNAMIC_FILTER_ANALYSIS_INTERVAL: int = 900
ORDER_BOOK_DEPTH_LIMIT: int = 100
ORDER_BOOK_WALL_MULTIPLIER: float = 10.0
ORDER_BOOK_ANALYSIS_RANGE_PCT: float = 0.02

conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
ml_models_cache: Dict[str, Any] = {}; exchange_info_map: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}; signal_cache_lock = Lock()
notifications_cache = deque(maxlen=50); notifications_lock = Lock()
signals_pending_closure: Set[int] = set(); closure_lock = Lock()
last_api_check_time = time.time()
rejection_logs_cache = deque(maxlen=100); rejection_logs_lock = Lock()
last_market_state_check = 0
# --- [Improvement] --- New market state structure
current_market_state: Dict[str, Any] = {
    "trend_score": 0,
    "trend_label": "INITIALIZING",
    "details_by_tf": {},
    "last_updated": None
}
market_state_lock = Lock()
dynamic_filter_profile_cache: Dict[str, Any] = {}
last_dynamic_filter_analysis_time: float = 0
dynamic_filter_lock = Lock()

REJECTION_REASONS_AR = {
    "Filters Not Loaded": "الفلاتر غير محملة",
    "Low Volatility": "تقلب منخفض جداً",
    "BTC Correlation": "ارتباط ضعيف بالبيتكوين",
    "RRR Filter": "نسبة المخاطرة/العائد غير كافية",
    "Reversal Volume Filter": "فوليوم الانعكاس ضعيف",
    "Momentum/Strength Filter": "فلتر الزخم والقوة",
    "Peak Filter": "فلتر القمة (السعر قريب جداً من القمة الأخيرة)",
    "Invalid ATR for TP/SL": "ATR غير صالح لحساب الأهداف",
    "Reversal Signal Rejected by ML Model": "نموذج التعلم الآلي رفض إشارة الانعكاس",
    "Invalid Position Size": "حجم الصفقة غير صالح (الوقف تحت الدخول)",
    "Lot Size Adjustment Failed": "فشل ضبط حجم العقد (LOT_SIZE)",
    "Min Notional Filter": "قيمة الصفقة أقل من الحد الأدنى",
    "Insufficient Balance": "الرصيد غير كافٍ",
    "Order Book Fetch Failed": "فشل جلب دفتر الطلبات",
    "Order Book Imbalance": "اختلال توازن دفتر الطلبات (ضغط بيع)",
    "Large Sell Wall Detected": "تم كشف جدار بيع ضخم",
}

# ---------------------- HTML Function for Dashboard ----------------------
def get_dashboard_html():
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم التداول V27.2</title>
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
                <span class="text-text-secondary font-medium">V27.2</span>
            </h1>
            <div class="flex items-center gap-4">
                <button onclick="downloadBacktestReport()" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-md transition duration-300">
                    تحميل تقرير الاختبار الخلفي
                </button>
                <div id="trend-lights-container" class="flex items-center gap-x-6 bg-black/20 px-4 py-2 rounded-lg border border-border-color">
                    <div class="flex items-center gap-2" title="اتجاه فريم 15 دقيقة"><div id="trend-light-15m" class="trend-light skeleton"></div><span class="text-sm font-bold text-text-secondary">15د</span></div>
                    <div class="flex items-center gap-2" title="اتجاه فريم ساعة"><div id="trend-light-1h" class="trend-light skeleton"></div><span class="text-sm font-bold text-text-secondary">1س</span></div>
                    <div class="flex items-center gap-2" title="اتجاه فريم 4 ساعات"><div id="trend-light-4h" class="trend-light skeleton"></div><span class="text-sm font-bold text-text-secondary">4س</span></div>
                </div>
                <div id="connection-status" class="flex items-center gap-3 text-sm">
                    <div class="flex items-center gap-2"><div id="db-status-light" class="w-2.5 h-2.5 rounded-full bg-gray-600 animate-pulse"></div><span class="text-text-secondary">DB</span></div>
                    <div class="flex items-center gap-2"><div id="api-status-light" class="w-2.5 h-2.5 rounded-full bg-gray-600 animate-pulse"></div><span class="text-text-secondary">API</span></div>
                </div>
            </div>
        </header>

        <!-- Control and Information Section -->
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
// --- [Improvement] --- Styles updated to fit the new point logic
const TREND_STYLES = {
    "صاعد قوي": { color: "text-accent-green" },
    "صاعد": { color: "text-green-400" },
    "محايد": { color: "text-accent-yellow" },
    "هابط": { color: "text-red-400" },
    "هابط قوي": { color: "text-accent-red" },
    "غير واضح": { color: "text-text-secondary" },
    "تهيئة...": { color: "text-accent-blue" }
};
const STRATEGY_STYLES = {
    "MOMENTUM": { text: "متابعة الزخم", color: "text-accent-blue" },
    "REVERSAL": { text: "بحث عن انعكاس", color: "text-accent-yellow" },
    "DISABLED": { text: "متوقف (غير مناسب)", color: "text-text-secondary" },
    "FORCED_MOMENTUM": { text: "زخم (يدوي)", color: "text-cyan-400" }
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
        document.getElementById('db-status-light').className = `w-2.5 h-2.5 rounded-full ${data.db_ok ? 'bg-green-500' : 'bg-red-500'}`;
        document.getElementById('api-status-light').className = `w-2.5 h-2.5 rounded-full ${data.api_ok ? 'bg-green-500' : 'bg-red-500'}`;
        
        updateMomentumToggle(data.force_momentum_enabled);

        // --- [Improvement] --- Update UI based on new point logic
        const state = data.market_state;
        const trendLabel = state.trend_label || "غير واضح";
        const trendStyle = TREND_STYLES[trendLabel] || TREND_STYLES["غير واضح"];
        
        const overallDiv = document.getElementById('overall-regime');
        overallDiv.textContent = trendLabel;
        overallDiv.className = `text-2xl font-bold ${trendStyle.color}`;

        const scoreDiv = document.getElementById('trend-score');
        scoreDiv.textContent = state.trend_score;
        scoreDiv.className = `text-3xl font-bold ${trendStyle.color}`;

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
        if (data.force_momentum_enabled) {
            strategy = "FORCED_MOMENTUM";
        }
        const strategyStyle = STRATEGY_STYLES[strategy] || STRATEGY_STYLES["DISABLED"];
        const strategyDiv = document.getElementById('active-strategy');
        strategyDiv.textContent = strategyStyle.text;
        strategyDiv.className = `text-base font-bold ${strategyStyle.color}`;

        const profileNameDiv = document.getElementById('filter-profile-name');
        profileNameDiv.textContent = profile.name;
        profileNameDiv.className = `text-xl font-bold ${trendStyle.color}`;

        const sessions = data.active_sessions;
        const sessionsDiv = document.getElementById('active-sessions-list');
        sessionsDiv.innerHTML = '';
        sessionsDiv.classList.remove('skeleton', 'h-12');
        if (sessions && sessions.length > 0) {
            const sessionColors = { 'London': 'bg-blue-500/20 text-blue-300', 'New York': 'bg-green-500/20 text-green-300', 'Tokyo': 'bg-red-500/20 text-red-300' };
            sessions.forEach(session => {
                const colorClass = sessionColors[session] || 'bg-gray-500/20 text-gray-300';
                const badge = `<span class="${colorClass} text-sm font-semibold px-3 py-1 rounded-full">${session}</span>`;
                sessionsDiv.innerHTML += badge;
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
                const item = `<div class="bg-gray-900/50 p-3 rounded-lg text-center"><div class="text-sm text-text-secondary uppercase">${key.replace(/_/g, ' ')}</div><div class="text-xl font-bold text-accent-blue font-mono">${displayValue}</div></div>`;
                filtersDisplay.innerHTML += item;
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
        if (data.is_enabled) {
            text.textContent = 'مُفعَّل';
            text.className = 'font-bold text-lg text-accent-green';
        } else {
            text.textContent = 'غير مُفعَّل';
            text.className = 'font-bold text-lg text-accent-red';
        }
    });
}

function updateMomentumToggle(is_forced) {
    const toggle = document.getElementById('force-momentum-toggle');
    const text = document.getElementById('force-momentum-text');
    const bg = toggle.nextElementSibling;

    toggle.checked = is_forced;
    if (is_forced) {
        text.textContent = 'زخم يدوي';
        text.className = 'font-bold text-lg text-cyan-400';
        bg.classList.remove('bg-gray-600');
        bg.classList.add('bg-accent-blue');
    } else {
        text.textContent = 'تلقائي';
        text.className = 'font-bold text-lg text-text-secondary';
        bg.classList.remove('bg-accent-blue');
        bg.classList.add('bg-gray-600');
    }
}

function toggleTrading() {
    const toggle = document.getElementById('trading-toggle');
    const confirmationMessage = toggle.checked
        ? "هل أنت متأكد من تفعيل التداول بأموال حقيقية؟"
        : "هل أنت متأكد من إيقاف التداول الحقيقي؟";

    if (confirm(confirmationMessage)) {
        apiFetch('/api/trading/toggle', { method: 'POST' }).then(data => {
            if (data.message) { alert(data.message); updateTradingStatus(); } 
            else if (data.error) { alert(`خطأ: ${data.error}`); updateTradingStatus(); }
        });
    } else { toggle.checked = !toggle.checked; }
}

function toggleMomentumStrategy() {
    const toggle = document.getElementById('force-momentum-toggle');
    const confirmationMessage = toggle.checked
        ? "هل أنت متأكد من فرض استراتيجية الزخم؟ سيتجاهل البوت ظروف السوق."
        : "هل أنت متأكد من العودة إلى الوضع التلقائي لاختيار الاستراتيجية؟";

    if (confirm(confirmationMessage)) {
        apiFetch('/api/strategy/force_momentum/toggle', { method: 'POST' }).then(data => {
            if (data.message) {
                alert(data.message);
                updateMomentumToggle(data.is_forced);
            } else if (data.error) {
                alert(`خطأ: ${data.error}`);
                updateMomentumToggle(!toggle.checked); // Revert UI on error
            }
        });
    } else {
        toggle.checked = !toggle.checked;
    }
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
        netProfitEl.classList.remove('skeleton', 'h-8', 'w-1/3');

        const statsContainer = document.getElementById('stats-container');
        statsContainer.innerHTML = `
            <div class="card text-center p-4 flex flex-col justify-center"><div class="text-sm text-text-secondary mb-1">نسبة النجاح</div><div class="text-3xl font-bold text-accent-blue">${formatNumber(data.win_rate)}%</div></div>
            <div class="card text-center p-4 flex flex-col justify-center"><div class="text-sm text-text-secondary mb-1">عامل الربح</div><div class="text-3xl font-bold text-accent-yellow">${profitFactorDisplay}</div></div>
            <div class="card text-center p-4 flex flex-col justify-center"><div class="text-sm text-text-secondary mb-1">إجمالي الصفقات المغلقة</div><div class="text-3xl font-bold text-text-primary">${formatNumber(data.total_closed_trades, 0)}</div></div>
            <div class="card text-center p-4 flex flex-col justify-center"><div class="text-sm text-text-secondary mb-1">متوسط الربح %</div><div class="text-3xl font-bold text-accent-green">${formatNumber(data.average_win_pct)}%</div></div>
            <div class="card text-center p-4 flex flex-col justify-center"><div class="text-sm text-text-secondary mb-1">متوسط الخسارة %</div><div class="text-3xl font-bold text-accent-red">${formatNumber(data.average_loss_pct)}%</div></div>
        `;
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
            if (!existingMsg) chartCard.insertAdjacentHTML('beforeend', '<p class="no-data-msg text-center text-text-secondary mt-8">لا توجد صفقات كافية لعرض المنحنى.</p>');
            return;
        }
        
        canvas.style.display = 'block';
        const ctx = canvas.getContext('2d');
        const chartData = data.map(d => ({ x: luxon.DateTime.fromISO(d.timestamp).valueOf(), y: d.cumulative_profit }));
        const lastProfit = chartData[chartData.length - 1].y;
        const gradient = ctx.createLinearGradient(0, 0, 0, ctx.canvas.height);
        if (lastProfit >= 0) {
            gradient.addColorStop(0, 'rgba(63, 185, 80, 0.4)'); gradient.addColorStop(1, 'rgba(63, 185, 80, 0)');
        } else {
            gradient.addColorStop(0, 'rgba(248, 81, 73, 0.4)'); gradient.addColorStop(1, 'rgba(248, 81, 73, 0)');
        }

        const config = {
            type: 'line',
            data: { datasets: [{
                label: 'الربح التراكمي %', data: chartData,
                borderColor: lastProfit >= 0 ? 'var(--accent-green)' : 'var(--accent-red)',
                backgroundColor: gradient, fill: true, tension: 0.4, pointRadius: 0, pointHoverRadius: 6,
                pointBackgroundColor: lastProfit >= 0 ? 'var(--accent-green)' : 'var(--accent-red)',
            }]},
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    x: { type: 'time', time: { unit: 'day', tooltipFormat: 'MMM dd, yyyy HH:mm' }, grid: { display: false }, ticks: { color: 'var(--text-secondary)', maxRotation: 0, autoSkip: true, maxTicksLimit: 7 } }
                    ,
                    y: { position: 'right', beginAtZero: true, grid: { color: 'var(--border-color)', drawBorder: false }, ticks: { color: 'var(--text-secondary)', callback: v => formatNumber(v) + '%' } }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        mode: 'index', intersect: false, backgroundColor: '#0D1117', titleFont: { weight: 'bold', family: 'Tajawal' },
                        bodyFont: { family: 'Tajawal' }, displayColors: false,
                        callbacks: { label: (ctx) => `الربح التراكمي: ${formatNumber(ctx.raw.y)}%` }
                    }
                },
                interaction: { mode: 'index', intersect: false }
            }
        };

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
    if ([entry_price, stop_loss, target_price, current_price].some(v => v === null || v === undefined)) return '<span class="text-xs text-text-secondary">لا تتوفر بيانات</span>';
    const [entry, sl, tp, current] = [entry_price, stop_loss, target_price, current_price].map(parseFloat);
    const totalDist = tp - sl;
    if (totalDist <= 0) return '<span class="text-xs text-text-secondary">بيانات غير صالحة</span>';
    const progressPct = Math.max(0, Math.min(100, ((current - sl) / totalDist) * 100));
    return `<div class="flex flex-col w-full"><div class="progress-bar-container"><div class="progress-bar ${current >= entry ? 'bg-accent-green' : 'bg-accent-red'}" style="width: ${progressPct}%"></div></div><div class="progress-labels"><span title="وقف الخسارة">${sl.toPrecision(4)}</span><span title="الهدف">${tp.toPrecision(4)}</span></div></div>`;
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
            
            let statusClass = 'text-gray-400';
            let statusText = signal.status;
            if (signal.status === 'open') { statusClass = 'text-yellow-400'; statusText = 'مفتوحة'; }
            else if (signal.status === 'updated') { statusClass = 'text-blue-400'; statusText = 'تم تحديثها'; }

            const quantityDisplay = signal.quantity ? formatNumber(signal.quantity, 4) : '-';
            const realTradeIndicator = signal.is_real_trade ? '<span class="text-accent-green" title="صفقة حقيقية">●</span>' : '';
            
            const strategyName = signal.strategy_name || '';
            let strategyBadge = '';
            if (strategyName.includes('Reversal')) {
                strategyBadge = '<span class="bg-yellow-500/20 text-yellow-300 text-xs font-bold px-2 py-1 rounded-md ml-2">انعكاس</span>';
            } else {
                strategyBadge = '<span class="bg-blue-500/20 text-blue-300 text-xs font-bold px-2 py-1 rounded-md ml-2">زخم</span>';
            }

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
    if (confirm(`هل أنت متأكد من رغبتك في إغلاق الصفقة #${signalId} يدوياً؟`)) {
        fetch(`/api/close/${signalId}`, { method: 'POST' }).then(res => res.json()).then(data => {
            alert(data.message || data.error);
            refreshData();
        });
    }
}

async function downloadBacktestReport() {
    try {
        const data = await apiFetch('/api/backtest_report');
        if (data.error) {
            alert(`خطأ في تحميل التقرير: ${data.error}`);
            return;
        }
        if (data.length === 0) {
            alert('لا توجد صفقات اختبار خلفي متاحة للتحميل.');
            return;
        }

        // Define CSV headers
        const headers = [
            "ID", "Symbol", "Status", "Entry Price", "Closing Price", "Profit %",
            "Entry Time", "Close Time", "Strategy", "ML Confidence", "Technical Reason",
            "Bid/Ask Ratio", "Filter Profile Name", "ADX Filter", "Relative Volume Filter",
            "RSI Range Filter Min", "RSI Range Filter Max", "ROC Filter", "Slope Filter",
            "Min RRR Filter", "Min Volatility % Filter", "Min BTC Correlation Filter",
            "Min Bid/Ask Ratio Filter", "Market Trend Score", "Market Trend Label",
            "Market Trend Details (15m Score)", "Market Trend Details (1h Score)",
            "Market Trend Details (4h Score)"
        ];

        // Map data to CSV rows
        const rows = data.map(trade => {
            const signalDetails = trade.signal_details || {};
            const backtestFilters = trade.backtest_filters || {};
            const backtestMarketTrend = trade.backtest_market_trend || {};
            const marketTrendDetailsByTf = backtestMarketTrend.details_by_tf || {};

            return [
                trade.id,
                trade.symbol,
                trade.status,
                trade.entry_price,
                trade.closing_price || '',
                trade.profit_percentage || '',
                trade.signal_details.timestamp || '',
                trade.closed_at || '',
                trade.strategy_name,
                signalDetails.ML_Confidence_Display || '',
                signalDetails.Technical_Reason || '',
                signalDetails.Bid_Ask_Ratio || '',
                signalDetails.Filter_Profile || '',
                backtestFilters.adx || '',
                backtestFilters.rel_vol || '',
                Array.isArray(backtestFilters.rsi_range) ? backtestFilters.rsi_range[0] : '',
                Array.isArray(backtestFilters.rsi_range) ? backtestFilters.rsi_range[1] : '',
                backtestFilters.roc || '',
                backtestFilters.slope || '',
                backtestFilters.min_rrr || '',
                backtestFilters.min_volatility_pct || '',
                backtestFilters.min_btc_correlation || '',
                backtestFilters.min_bid_ask_ratio || '',
                backtestMarketTrend.trend_score || '',
                backtestMarketTrend.trend_label || '',
                marketTrendDetailsByTf['15m'] ? marketTrendDetailsByTf['15m'].score : '',
                marketTrendDetailsByTf['1h'] ? marketTrendDetailsByTf['1h'].score : '',
                marketTrendDetailsByTf['4h'] ? marketTrendDetailsByTf['4h'].score : ''
            ].map(item => {
                // Handle commas and quotes in CSV
                if (typeof item === 'string' && item.includes(',')) {
                    return `"${item.replace(/"/g, '""')}"`;
                }
                return item;
            }).join(',');
        });

        const csvContent = [headers.join(','), ...rows].join('\\n');
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `backtest_report_${luxon.DateTime.now().toFormat('yyyyMMdd_HHmmss')}.csv`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(link.href);
        alert('تم تحميل تقرير الاختبار الخلفي بنجاح!');

    } catch (error) {
        console.error('Error downloading backtest report:', error);
        alert('حدث خطأ أثناء تحميل تقرير الاختبار الخلفي.');
    }
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

# ---------------------- Database Functions ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    global conn
    logger.info("[DB] Initializing database connection...")
    db_url_to_use = DB_URL
    if 'postgres' in db_url_to_use and 'sslmode' not in db_url_to_use:
        separator = '&' if '?' in db_url_to_use else '?'
        db_url_to_use += f"{separator}sslmode=require"
    
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(db_url_to_use, connect_timeout=15, cursor_factory=RealDictCursor)
            conn.autocommit = False
            
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION NOT NULL,
                        target_price DOUBLE PRECISION NOT NULL, stop_loss DOUBLE PRECISION NOT NULL,
                        status TEXT DEFAULT 'open', closing_price DOUBLE PRECISION, closed_at TIMESTAMP,
                        profit_percentage DOUBLE PRECISION, strategy_name TEXT, signal_details JSONB,
                        current_peak_price DOUBLE PRECISION, is_real_trade BOOLEAN DEFAULT FALSE,
                        quantity DOUBLE PRECISION, order_id TEXT,
                        backtest_filters JSONB, backtest_market_trend JSONB
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals (status);")
                cur.execute("""
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
    log_message = f"🚫 [REJECTED] {symbol} | Reason: {reason_key} | Details: {details or {}}"
    logger.info(log_message)
    with rejection_logs_lock:
        rejection_logs_cache.appendleft({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "reason": reason_ar,
            "details": details or {}
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

# ---------------------- Binance and Data Functions ----------------------
def get_exchange_info_map() -> None:
    global exchange_info_map
    if not client: return
    logger.info("ℹ️ [Exchange Info] Fetching exchange trading rules...")
    try:
        info = client.get_exchange_info()
        exchange_info_map = {s['symbol']: s for s in info['symbols']}
        logger.info(f"✅ [Exchange Info] Loaded rules for {len(exchange_info_map)} symbols.")
    except Exception as e:
        logger.error(f"❌ [Exchange Info] Could not fetch exchange info: {e}")

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

def fetch_historical_data(symbol: str, interval: str, days: int, end_str: Optional[str] = None) -> Optional[pd.DataFrame]:
    # This function is modified to accept an end_str for backtesting
    if not client: return None
    try:
        # Calculate limit based on interval and days
        interval_minutes = int(re.sub('[a-zA-Z]', '', interval))
        # Binance API max limit is 1000, so we might need multiple calls for very long periods
        # For backtesting 5 days at 15m, 480 candles, so one call is enough.
        limit = int((days * 24 * 60) / interval_minutes)

        # Fetch klines with an optional end_str
        klines = client.get_historical_klines(symbol, interval, f"{days} days ago UTC", end_str)
        
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.astype({'open': np.float32, 'high': np.float32, 'low': np.float32, 'close': np.float32, 'volume': np.float32})
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna()
    except Exception as e:
        logger.error(f"❌ [Data] Error fetching historical data for {symbol}: {e}")
        return None

def analyze_order_book(symbol: str, entry_price: float) -> Optional[Dict[str, Any]]:
    # This function is not directly used in backtesting as order book data is not historical.
    # For backtesting, we might simulate this or make assumptions.
    # For now, it will return a default positive response for backtesting purposes.
    return {
        "bid_ask_ratio": 1.5, # Assume favorable ratio for backtesting
        "has_large_sell_wall": False,
        "wall_details": []
    }

# ---------------------- Feature Calculation and Trend Determination Functions ----------------------
def calculate_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df_calc = df.copy()
    
    # --- [Improvement] --- Add averages used in trend analysis
    for period in EMA_PERIODS:
        df_calc[f'ema_{period}'] = df_calc['close'].ewm(span=period, adjust=False).mean()

    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    
    up_move = df_calc['high'].diff()
    down_move = -df_calc['low'].diff()
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
    
    kc_ema = df_calc['close'].ewm(span=20, adjust=False).mean()
    kc_atr = df_calc['atr'].ewm(span=10, adjust=False).mean()
    df_calc['kc_upper'] = kc_ema + (kc_atr * 2)
    df_calc['kc_lower'] = kc_ema - (kc_atr * 2)
    df_calc['kc_middle'] = kc_ema

    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)
    df_calc['price_vs_ema50'] = (df_calc['close'] / df_calc['ema_50']) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / df_calc['ema_200']) - 1
    
    if btc_df is not None and not btc_df.empty:
        merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
        df_calc['btc_correlation'] = df_calc['close'].pct_change().rolling(window=30).corr(merged_df['btc_returns'])
    else:
        df_calc['btc_correlation'] = 0.0
        
    df_calc[f'roc_{MOMENTUM_PERIOD}'] = (df_calc['close'] / df_calc['close'].shift(MOMENTUM_PERIOD) - 1) * 100
    df_calc['roc_acceleration'] = df_calc[f'roc_{MOMENTUM_PERIOD}'].diff()
    ema_slope = df_calc['close'].ewm(span=EMA_SLOPE_PERIOD, adjust=False).mean()
    df_calc[f'ema_slope_{EMA_SLOPE_PERIOD}'] = (ema_slope - ema_slope.shift(1)) / ema_slope.shift(1).replace(0, 1e-9) * 100
    df_calc['hour_of_day'] = df_calc.index.hour
    
    return df_calc.astype('float32', errors='ignore')

# --- [Improvement] --- New function to determine trend based on point system
def determine_market_trend_score_backtest(btc_data_15m: pd.DataFrame, btc_data_1h: pd.DataFrame, btc_data_4h: pd.DataFrame):
    # This function is modified for backtesting to take pre-fetched BTC data
    logger.debug("🧠 [Market Score] Updating multi-timeframe trend score for backtest...")
    
    total_score = 0
    details = {}
    tf_weights = {'15m': 0.2, '1h': 0.3, '4h': 0.5} # Give more weight to larger timeframes

    for tf, df in [('15m', btc_data_15m), ('1h', btc_data_1h), ('4h', btc_data_4h)]:
        if df is None or len(df) < EMA_PERIODS[-1]:
            details[tf] = {"score": 0, "label": "غير واضح", "reason": "بيانات غير كافية"}
            continue

        # Add averages
        for period in EMA_PERIODS:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        last_candle = df.iloc[-1]
        close = last_candle['close']
        ema21 = last_candle['ema_21']
        ema50 = last_candle['ema_50']
        ema200 = last_candle['ema_200']

        tf_score = 0
        # Point 1: Price above/below EMA21
        if close > ema21: tf_score += 1
        elif close < ema21: tf_score -= 1
        
        # Point 2: EMA21 above/below EMA50
        if ema21 > ema50: tf_score += 1
        elif ema21 < ema50: tf_score -= 1

        # Point 3: EMA50 above/below EMA200
        if ema50 > ema200: tf_score += 1
        elif ema50 < ema200: tf_score -= 1

        label = "محايد"
        if tf_score >= 2: label = "صاعد"
        elif tf_score <= -2: label = "هابط"
        
        details[tf] = {"score": tf_score, "label": label, "reason": f"EMA21:{ema21:.2f}, EMA50:{ema50:.2f}, EMA200:{ema200:.2f}"}
        total_score += tf_score * tf_weights[tf]
    
    # Round the final result
    final_score = round(total_score)
    
    trend_label = "محايد"
    if final_score >= 4: trend_label = "صاعد قوي"
    elif final_score >= 1: trend_label = "صاعد"
    elif final_score <= -4: trend_label = "هابط قوي"
    elif final_score <= -1: trend_label = "هابط"

    market_state = {
        "trend_score": final_score,
        "trend_label": trend_label,
        "details_by_tf": details,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
    logger.debug(f"✅ [Market Score] Backtest State: Score={final_score}, Label='{trend_label}'")
    return market_state


def get_session_state_backtest(timestamp: datetime) -> Tuple[List[str], str, str]:
    # This function is modified for backtesting to take a specific timestamp
    sessions = {"London": (8, 17), "New York": (13, 22), "Tokyo": (0, 9)}
    active_sessions = []
    now_utc = timestamp
    current_hour = now_utc.hour
    if now_utc.weekday() >= 5: # Saturday or Sunday
        return [], "WEEKEND", "Very low liquidity (weekend)"
    for session, (start, end) in sessions.items():
        if start <= current_hour < end:
            active_sessions.append(session)
    if "London" in active_sessions and "New York" in active_sessions:
        return active_sessions, "HIGH_LIQUIDITY", "High liquidity (London/New York overlap)"
    elif len(active_sessions) >= 1:
        return active_sessions, "NORMAL_LIQUIDITY", f"Normal liquidity ({', '.join(active_sessions)})"
    else:
        return [], "LOW_LIQUIDITY", "Low liquidity (off-peak hours)"

def analyze_market_and_create_dynamic_profile_backtest(market_state: Dict[str, Any], timestamp: datetime) -> Dict[str, Any]:
    # This function is modified for backtesting to take market_state and timestamp
    logger.debug("🔬 [Dynamic Filter] Generating profile for backtest...")
    
    # In backtesting, we don't have a 'force_momentum_strategy' toggle.
    # We assume it's always automatic unless specified otherwise for backtesting.
    is_forced = False 

    active_sessions, liquidity_state, liquidity_desc = get_session_state_backtest(timestamp)
    market_label = market_state.get("trend_label", "محايد")

    # --- [Improvement] --- Select filter profile based on new trend classification
    profile_key = "RANGING" # Default
    if "صاعد قوي" in market_label: profile_key = "STRONG_UPTREND"
    elif "صاعد" in market_label: profile_key = "UPTREND"
    elif "هابط قوي" in market_label: profile_key = "STRONG_DOWNTREND"
    elif "هابط" in market_label: profile_key = "DOWNTREND"

    if liquidity_state == "WEEKEND":
        base_profile = FILTER_PROFILES["WEEKEND"].copy()
    else:
        base_profile = FILTER_PROFILES.get(profile_key, FILTER_PROFILES["RANGING"]).copy()

    dynamic_filter_profile = {
        "name": base_profile['description'],
        "description": liquidity_desc,
        "strategy": base_profile.get("strategy", "DISABLED"),
        "filters": base_profile.get("filters", {}),
        "last_updated": timestamp.isoformat(),
    }
    
    logger.debug(f"✅ [Dynamic Filter] Backtest profile: '{base_profile['description']}' | Strategy: '{base_profile['strategy']}'")
    return dynamic_filter_profile

def get_current_filter_profile() -> Dict[str, Any]:
    # This function is for live trading. For backtesting, we use the _backtest version.
    with dynamic_filter_lock:
        return dict(dynamic_filter_profile_cache)

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

# ---------------------- Strategy and Live Trading Functions ----------------------

def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> Optional[Decimal]:
    # This function is primarily for live trading with Binance API.
    # For backtesting, we can simplify or assume perfect lot size.
    # For now, return Decimal(str(quantity))
    return Decimal(str(quantity))

def calculate_position_size(symbol: str, entry_price: float, stop_loss_price: float) -> Optional[Decimal]:
    # This function is primarily for live trading with Binance API.
    # For backtesting, we can use a fixed trade size for simplicity.
    # For backtesting, we'll use a fixed STATS_TRADE_SIZE_USDT for the notional value.
    try:
        risk_per_coin = Decimal(str(entry_price)) - Decimal(str(stop_loss_price))
        if risk_per_coin <= 0:
            logger.warning(f"[{symbol}] Invalid position size calculation: SL above entry or zero risk.")
            return None
        
        # In backtesting, we use a fixed USDT value for trade size
        quantity = Decimal(str(STATS_TRADE_SIZE_USDT)) / Decimal(str(entry_price))
        adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(quantity))
        
        if adjusted_quantity is None or adjusted_quantity <= 0:
            logger.warning(f"[{symbol}] Backtest lot size adjustment failed or quantity is zero.")
            return None
        
        return adjusted_quantity
    except Exception as e:
        logger.error(f"❌ [{symbol}] Error in calculate_position_size for backtest: {e}", exc_info=True)
        return None

def place_order(symbol: str, side: str, quantity: Decimal, order_type: str = Client.ORDER_TYPE_MARKET) -> Optional[Dict]:
    # This function is for live trading. In backtesting, we simulate orders.
    logger.debug(f"➡️ [{symbol}] Simulating {side} order for {quantity} units.")
    return {"orderId": f"BACKTEST_ORDER_{random.randint(1000, 9999)}", "status": "FILLED"}

class TradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_bundle_from_folder(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.feature_names is None: return None
        try:
            df_featured = calculate_features(df_15m, btc_df)
            df_4h_features = calculate_features(df_4h, None)
            df_4h_features = df_4h_features.rename(columns=lambda c: f"{c}_4h", inplace=False)
            required_4h_cols = ['rsi_4h', 'price_vs_ema50_4h']
            
            # Ensure df_4h_features has enough data to perform join
            if len(df_4h_features) < len(df_featured):
                # If 4h data is shorter, reindex df_4h_features to match df_featured's index
                # and then fill NaNs. This ensures proper alignment for feature combination.
                df_4h_features = df_4h_features.reindex(df_featured.index, method='ffill')

            df_featured = df_featured.join(df_4h_features[required_4h_cols], how='left') # Changed to left join
            df_featured.fillna(method='ffill', inplace=True) # Fill NaNs after join
            for col in self.feature_names:
                if col not in df_featured.columns: df_featured[col] = 0.0
            df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_featured.dropna(subset=self.feature_names)
        except Exception as e:
            logger.error(f"❌ [{self.symbol}] Feature engineering failed: {e}", exc_info=True)
            return None

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

def find_crazy_reversal_signal(df_featured: pd.DataFrame) -> Optional[Dict[str, Any]]:
    try:
        if len(df_featured) < 30: return None
        
        last_candle = df_featured.iloc[-1]
        if last_candle['close'] <= last_candle['kc_upper']:
            return None
        
        lookback_period = 25
        relevant_data = df_featured.iloc[-lookback_period:-1]
        
        if relevant_data.empty: return None # Ensure there's data to analyze after slicing

        price_low_idx = relevant_data['low'].idxmin()
        price_low_val = relevant_data.loc[price_low_idx, 'low']
        rsi_at_price_low = relevant_data.loc[price_low_idx, 'rsi']

        current_price_low = last_candle['low']
        if current_price_low <= price_low_val:
            return None

        current_rsi = last_candle['rsi']
        if current_rsi >= rsi_at_price_low:
            return None

        if last_candle['relative_volume'] < 1.5:
            return None

        logger.info(f"✅ [CRAZY REVERSAL] Signal detected for {df_featured.name}!")
        return {
            "signal_type": "CRAZY_REVERSAL",
            "reason": f"Hidden Divergence, Keltner Breakout, Volume Spike (RelVol: {last_candle['relative_volume']:.2f})"
        }

    except Exception as e:
        symbol_name = df_featured.name if hasattr(df_featured, 'name') else 'Unknown'
        logger.error(f"❌ [{symbol_name}] Error in find_crazy_reversal_signal: {e}")
        return None

def passes_filters(symbol: str, last_features: pd.Series, profile: Dict[str, Any], entry_price: float, tp_sl_data: Dict, df_15m: pd.DataFrame) -> bool:
    filters = profile.get("filters", {})
    if not filters:
        log_rejection(symbol, "Filters Not Loaded", {"profile": profile.get('name')})
        return False

    # --- [Improvement] --- No need to modify filters here as they come ready from the dynamic profile
    final_filters = filters.copy()

    volatility = (last_features.get('atr', 0) / entry_price * 100) if entry_price > 0 else 0
    if volatility < final_filters.get('min_volatility_pct', 0.0):
        log_rejection(symbol, "Low Volatility", {"volatility": f"{volatility:.2f}%", "min": f"{final_filters.get('min_volatility_pct', 0.0):.2f}%"})
        return False

    correlation = last_features.get('btc_correlation', 0)
    if correlation < final_filters.get('min_btc_correlation', -1.0):
        log_rejection(symbol, "BTC Correlation", {"corr": f"{correlation:.2f}", "min": f"{final_filters.get('min_btc_correlation', -1.0)}"})
        return False

    risk, reward = entry_price - float(tp_sl_data['stop_loss']), float(tp_sl_data['target_price']) - entry_price
    if risk <= 0 or reward <= 0 or (reward / risk) < final_filters.get('min_rrr', 0.0):
        log_rejection(symbol, "RRR Filter", {"rrr": f"{(reward/risk):.2f}" if risk > 0 else "N/A", "min": f"{final_filters.get('min_rrr', 0.0):.2f}"})
        return False

    if profile.get("strategy") == "REVERSAL":
        rel_vol = last_features.get('relative_volume', 0)
        if rel_vol < final_filters.get('min_relative_volume', 1.5):
            log_rejection(symbol, "Reversal Volume Filter", {"RelVol": f"{rel_vol:.2f}", "min": final_filters.get('min_relative_volume', 1.5)})
            return False

    elif profile.get("strategy") == "MOMENTUM":
        adx = last_features.get('adx', 0)
        rel_vol = last_features.get('relative_volume', 0)
        rsi = last_features.get('rsi', 0)
        roc = last_features.get(f'roc_{MOMENTUM_PERIOD}', 0)
        slope = last_features.get(f'ema_slope_{EMA_SLOPE_PERIOD}', 0)
        
        rsi_min, rsi_max = final_filters.get('rsi_range', (0, 100))

        if not (adx >= final_filters.get('adx', 0) and 
                rel_vol >= final_filters.get('rel_vol', 0) and 
                rsi_min <= rsi < rsi_max and
                roc > final_filters.get('roc', -100) and
                slope > final_filters.get('slope', -100)):
            log_rejection(symbol, "Momentum/Strength Filter", {
                "ADX": f"{adx:.2f}", "Volume": f"{rel_vol:.2f}", "RSI": f"{rsi:.2f}",
                "ROC": f"{roc:.2f}", "Slope": f"{slope:.6f}"
            })
            return False
        
        # --- [Improvement] --- Simplified and effective peak filter
        if USE_PEAK_FILTER:
            if df_15m is None or len(df_15m) < PEAK_LOOKBACK_PERIOD:
                 logger.warning(f"⚠️ [{symbol}] Not enough 15m data for peak filter. Skipping peak check.")
            else:
                lookback_data = df_15m.iloc[-PEAK_LOOKBACK_PERIOD:-1] # Exclude current candle
                if not lookback_data.empty:
                    highest_high = lookback_data['high'].max()
                    if entry_price >= (highest_high * PEAK_DISTANCE_THRESHOLD_PCT):
                        log_rejection(symbol, "Peak Filter", {"entry": f"{entry_price:.4f}", "peak_limit": f"{highest_high * PEAK_DISTANCE_THRESHOLD_PCT:.4f}"})
                        return False

    return True


def passes_order_book_check(symbol: str, order_book_analysis: Dict, profile: Dict) -> bool:
    # In backtesting, we assume order book is favorable based on the simplified analyze_order_book
    return True

def calculate_tp_sl(symbol: str, entry_price: float, last_atr: float) -> Optional[Dict[str, Any]]:
    if last_atr <= 0:
        log_rejection(symbol, "Invalid ATR for TP/SL", {"atr": last_atr})
        return None
    tp = entry_price + (last_atr * ATR_FALLBACK_TP_MULTIPLIER)
    sl = entry_price - (last_atr * ATR_FALLBACK_SL_MULTIPLIER)
    return {'target_price': tp, 'stop_loss': sl, 'source': 'ATR_Fallback'}

def handle_price_update_message(msg: List[Dict[str, Any]]) -> None:
    # This function is for live trading. Not used in backtesting.
    pass

def initiate_signal_closure(symbol: str, signal_to_close: Dict, status: str, closing_price: float):
    # This function is for live trading. In backtesting, closure is handled synchronously.
    pass

def update_signal_peak_price_in_db(signal_id: int, new_peak_price: float):
    # This function is for live trading. Not directly used in backtesting.
    pass

def trade_monitoring_loop():
    # This function is for live trading. Not used in backtesting.
    pass

def send_telegram_message(target_chat_id: str, text: str, reply_markup: Optional[Dict] = None) -> bool:
    # This function is for live trading. Not used in backtesting.
    return True

def send_new_signal_alert(signal_data: Dict[str, Any]):
    # This function is for live trading. Not used in backtesting.
    pass


def insert_signal_into_db(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Modified to save backtest_filters and backtest_market_trend
    if not check_db_connection() or not conn: return None
    try:
        entry, target, sl = float(signal['entry_price']), float(signal['target_price']), float(signal['stop_loss'])
        is_real, quantity, order_id = signal.get('is_real_trade', False), signal.get('quantity'), signal.get('order_id')
        
        backtest_filters = json.dumps(signal.get('backtest_filters', {}))
        backtest_market_trend = json.dumps(signal.get('backtest_market_trend', {}))

        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price, stop_loss, status, closed_at, profit_percentage, 
                                     strategy_name, signal_details, current_peak_price, is_real_trade, quantity, order_id,
                                     backtest_filters, backtest_market_trend)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
                """,
                (signal['symbol'], entry, target, sl, signal.get('status', 'open'), signal.get('closed_at'), signal.get('profit_percentage'),
                 signal.get('strategy_name'), json.dumps(signal.get('signal_details', {})), entry, is_real, quantity, order_id,
                 backtest_filters, backtest_market_trend))
            signal['id'] = cur.fetchone()['id']
        conn.commit()
        logger.info(f"✅ [DB] Inserted signal {signal['id']} for {signal['symbol']}. Real: {is_real}")
        return signal
    except Exception as e:
        logger.error(f"❌ [Insert] Error inserting signal for {signal['symbol']}: {e}", exc_info=True)
        if conn: conn.rollback()
        return None

def close_signal(signal: Dict, status: str, closing_price: float):
    # This function is for live trading. In backtesting, closure is handled synchronously.
    pass


def load_open_signals_to_cache():
    # This function is for live trading. Not used in backtesting.
    pass

def load_notifications_to_cache():
    # This function is for live trading. Not used in backtesting.
    pass

def get_btc_data_for_bot() -> Optional[pd.DataFrame]:
    # This function is for live trading. For backtesting, we fetch BTC data for specific timestamps.
    return None

def perform_end_of_cycle_cleanup():
    logger.info("🧹 [Cleanup] Starting end-of-cycle cleanup...")
    try:
        if redis_client: redis_client.delete(REDIS_PRICES_HASH_NAME)
        ml_models_cache.clear()
        collected = gc.collect()
        logger.info(f"🧹 [Cleanup] Final garbage collection complete. Collected {collected} objects.")
    except Exception as e:
        logger.error(f"❌ [Cleanup] An error occurred during cleanup: {e}", exc_info=True)

# ---------------------- Main Work Loop ----------------------
def main_loop():
    # This is the main live trading loop. We will create a separate backtesting loop.
    logger.info("[Main Loop] Waiting for initialization...")
    time.sleep(15)
    if not validated_symbols_to_scan:
        log_and_notify("critical", "No validated symbols to scan. Bot will not start.", "SYSTEM")
        return
    log_and_notify("info", f"✅ Starting main scan loop for {len(validated_symbols_to_scan)} symbols.", "SYSTEM")

    while True:
        try:
            logger.info("🔄 Starting new main cycle...")
            ml_models_cache.clear(); gc.collect()

            # --- [Improvement] --- Use the new trend determination function
            determine_market_trend_score()
            analyze_market_and_create_dynamic_profile()
            
            filter_profile = get_current_filter_profile()
            active_strategy_type = filter_profile.get("strategy")
            
            if not active_strategy_type or active_strategy_type == "DISABLED":
                logger.warning(f"🛑 Trading is disabled by profile: '{filter_profile.get('name')}'. Skipping cycle.")
                time.sleep(300)
                continue

            btc_data = get_btc_data_for_bot()
            
            symbols_with_models = []
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir_path = os.path.join(script_dir, MODEL_FOLDER)
            for symbol in validated_symbols_to_scan:
                if os.path.exists(os.path.join(model_dir_path, f"{BASE_ML_MODEL_NAME}_{symbol}.pkl")):
                    symbols_with_models.append(symbol)
            
            if not symbols_with_models:
                logger.warning("⚠️ No symbols with models found. Skipping scan cycle.")
                time.sleep(300)
                continue
                
            logger.info(f"✅ Found {len(symbols_with_models)} symbols with models. Active Strategy: {active_strategy_type}")
            symbols_to_process = random.sample(symbols_with_models, len(symbols_with_models))
            
            processed_count = 0
            for symbol in symbols_to_process:
                strategy, df_15m, df_4h, df_features = None, None, None, None
                try:
                    with signal_cache_lock:
                        if symbol in open_signals_cache or len(open_signals_cache) >= MAX_OPEN_TRADES:
                            continue
                    
                    df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_15m is None or df_15m.empty: continue
                    df_15m.name = symbol

                    df_features_with_indicators = calculate_features(df_15m, btc_data)
                    if df_features_with_indicators is None or df_features_with_indicators.empty: continue
                    df_features_with_indicators.name = symbol
                    
                    technical_signal = None
                    if active_strategy_type == "REVERSAL":
                        technical_signal = find_crazy_reversal_signal(df_features_with_indicators)
                    elif active_strategy_type == "MOMENTUM":
                        technical_signal = {"signal_type": "MOMENTUM"}

                    if not technical_signal:
                        continue
                    
                    strategy = TradingStrategy(symbol)
                    if not all([strategy.ml_model, strategy.scaler, strategy.feature_names]): continue
                    
                    df_4h = fetch_historical_data(symbol, '4h', SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_4h is None or df_4h.empty: continue
                    
                    df_features = strategy.get_features(df_15m, df_4h, btc_data)
                    if df_features is None or df_features.empty: continue
                    
                    ml_signal = strategy.generate_buy_signal(df_features)
                    
                    if not ml_signal or ml_signal['confidence'] < BUY_CONFIDENCE_THRESHOLD:
                        if "REVERSAL" in technical_signal['signal_type']:
                            log_rejection(symbol, "Reversal Signal Rejected by ML Model", {"ML_confidence": ml_signal.get('confidence') if ml_signal else 'N/A'})
                        continue
                    
                    try:
                        entry_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                    except Exception as e:
                        logger.error(f"❌ [{symbol}] Could not fetch entry price: {e}. Skipping.")
                        continue

                    last_features = df_features.iloc[-1]
                    last_atr = last_features.get('atr', 0)
                    tp_sl_data = calculate_tp_sl(symbol, entry_price, last_atr)
                    
                    if not tp_sl_data or not passes_filters(symbol, last_features, filter_profile, entry_price, tp_sl_data, df_15m):
                        continue

                    order_book_analysis = analyze_order_book(symbol, entry_price)
                    if not order_book_analysis or not passes_order_book_check(symbol, order_book_analysis, filter_profile):
                        continue
                    
                    strategy_name_for_db = f"Reversal_ML" if active_strategy_type == "REVERSAL" else f"Momentum_ML"
                    new_signal = {
                        'symbol': symbol, 'strategy_name': strategy_name_for_db,
                        'signal_details': {
                            'ML_Confidence': ml_signal['confidence'],
                            'ML_Confidence_Display': f"{ml_signal['confidence']:.2%}",
                            'Filter_Profile': f"{filter_profile['name']}",
                            'Technical_Reason': technical_signal.get('reason', 'N/A'),
                            'Bid_Ask_Ratio': order_book_analysis.get('bid_ask_ratio', 0)
                        },
                        'entry_price': entry_price, **tp_sl_data
                    }
                    
                    with trading_status_lock: is_enabled = is_trading_enabled
                    if is_enabled:
                        quantity = calculate_position_size(symbol, entry_price, new_signal['stop_loss'])
                        if quantity and quantity > 0:
                            order_result = place_order(symbol, Client.SIDE_BUY, quantity)
                            if order_result:
                                new_signal.update({'is_real_trade': True, 'quantity': float(quantity), 'order_id': order_result['orderId']})
                            else: continue
                        else: continue
                    else:
                        new_signal['is_real_trade'] = False

                    saved_signal = insert_signal_into_db(new_signal)
                    if saved_signal:
                        with signal_cache_lock: open_signals_cache[saved_signal['symbol']] = saved_signal
                        send_new_signal_alert(saved_signal)

                except Exception as e:
                    logger.error(f"❌ [Processing Error] for symbol {symbol}: {e}", exc_info=True)
                finally:
                    del strategy, df_15m, df_4h, df_features
                    processed_count += 1
                    if processed_count % SYMBOL_PROCESSING_BATCH_SIZE == 0:
                        logger.info("📦 Processed batch. Running memory cleanup...")
                        ml_models_cache.clear(); gc.collect()
                        time.sleep(2)
            
            logger.info("✅ [End of Cycle] Full scan cycle finished.")
            perform_end_of_cycle_cleanup()
            logger.info(f"⏳ [End of Cycle] Waiting for 60 seconds...")
            time.sleep(60)

        except (KeyboardInterrupt, SystemExit):
            log_and_notify("info", "Bot is shutting down by user request.", "SYSTEM"); break
        except Exception as main_err:
            log_and_notify("error", f"Critical error in main loop: {main_err}", "SYSTEM"); time.sleep(120)


# ---------------------- Flask API Interface ----------------------
app = Flask(__name__)
CORS(app)

def get_fear_and_greed_index() -> Dict[str, Any]:
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10).json()
        return {"value": int(response['data'][0]['value']), "classification": response['data'][0]['value_classification']}
    except Exception: return {"value": -1, "classification": "Error"}

def check_api_status() -> bool:
    if not client: return False
    try: client.ping(); return True
    except Exception: return False

def get_usdt_balance() -> Optional[float]:
    if not client: return None
    try:
        return float(client.get_asset_balance(asset='USDT')['free'])
    except Exception: return None

@app.route('/')
def home(): return render_template_string(get_dashboard_html())

@app.route('/api/market_status')
def get_market_status():
    with market_state_lock: state_copy = dict(current_market_state)
    with force_momentum_lock: is_forced = force_momentum_strategy
    profile_copy = get_current_filter_profile()
    active_sessions, _, _ = get_session_state()
    return jsonify({
        "fear_and_greed": get_fear_and_greed_index(), "market_state": state_copy,
        "filter_profile": profile_copy, "active_sessions": active_sessions,
        "db_ok": check_db_connection(), "api_ok": check_api_status(),
        "usdt_balance": get_usdt_balance(),
        "force_momentum_enabled": is_forced
    })

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
                trade_size = STATS_TRADE_SIZE_USDT
                if t.get('is_real_trade') and t.get('quantity') and t.get('entry_price'):
                    trade_size = float(t['quantity']) * float(t['entry_price'])
                total_net_profit_usdt += (profit_pct / 100) * trade_size
        
        wins_list = [float(s['profit_percentage']) for s in closed_trades if float(s['profit_percentage']) > 0]
        losses_list = [float(s['profit_percentage']) for s in closed_trades if float(s['profit_percentage']) < 0]
        win_rate = (len(wins_list) / len(closed_trades) * 100) if closed_trades else 0.0
        avg_win = sum(wins_list) / len(wins_list) if wins_list else 0.0
        avg_loss = sum(losses_list) / len(losses_list) if losses_list else 0.0
        profit_factor_val = sum(wins_list) / abs(sum(losses_list)) if abs(sum(losses_list)) > 0 else "Infinity"

        return jsonify({
            "open_trades_count": open_trades_count, "net_profit_usdt": total_net_profit_usdt,
            "win_rate": win_rate, "profit_factor": profit_factor_val, "total_closed_trades": len(closed_trades),
            "average_win_pct": avg_win, "average_loss_pct": avg_loss
        })
    except Exception as e:
        logger.error(f"❌ [API Stats] Error: {e}", exc_info=True)
        if conn: conn.rollback()
        return jsonify({"error": "Internal error in stats"}), 500

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
        logger.error(f"❌ [API Profit Curve] Error: {e}", exc_info=True)
        if conn: conn.rollback()
        return jsonify({"error": "Error fetching profit curve"}), 500

@app.route('/api/signals')
def get_signals():
    if not all([check_db_connection(), redis_client, client]):
        return jsonify({"error": "Service connection failed"}), 500
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
                    symbol = s['symbol']
                    price = None
                    try:
                        price = float(redis_prices_map.get(symbol))
                    except (ValueError, TypeError, AttributeError):
                        try:
                            price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                            logger.info(f"[{symbol}] Fetched price directly from API for dashboard.")
                        except Exception as api_e:
                            logger.warning(f"[{symbol}] Could not fetch price from API for dashboard: {api_e}")
                    
                    s['current_price'] = price
                    if price and s.get('entry_price'):
                        s['pnl_pct'] = ((price / float(s['entry_price'])) - 1) * 100
        return jsonify(all_signals)
    except Exception as e:
        logger.error(f"❌ [API Signals] Error: {e}", exc_info=True)
        if conn: conn.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/api/close/<int:signal_id>', methods=['POST'])
def manual_close_signal_api(signal_id):
    if not client: return jsonify({"error": "Binance Client not available"}), 500
    with closure_lock:
        if signal_id in signals_pending_closure: return jsonify({"error": "Signal is already being closed"}), 409
    if not check_db_connection(): return jsonify({"error": "DB connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE id = %s AND status IN ('open', 'updated');", (signal_id,))
            signal_to_close = cur.fetchone()
        if not signal_to_close: return jsonify({"error": "Signal not found or already closed"}), 404
        symbol = dict(signal_to_close)['symbol']
        try:
            price = float(client.get_symbol_ticker(symbol=symbol)['price'])
        except Exception as e:
            return jsonify({"error": f"Could not fetch price for {symbol}: {e}"}), 500
        initiate_signal_closure(symbol, dict(signal_to_close), 'manual_close', price)
        return jsonify({"message": f"تم إرسال طلب إغلاق الصفقة {signal_id}..."})
    except Exception as e:
        logger.error(f"❌ [API Close] Error: {e}", exc_info=True)
        if conn: conn.rollback()
        return jsonify({"error": str(e)}), 500

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
        # Trigger analysis immediately to reflect the change
        Thread(target=analyze_market_and_create_dynamic_profile).start()
        return jsonify({"message": f"Strategy mode set to {status_msg}", "is_forced": force_momentum_strategy})

@app.route('/api/notifications')
def get_notifications():
    with notifications_lock: return jsonify(list(notifications_cache))

@app.route('/api/rejection_logs')
def get_rejection_logs():
    with rejection_logs_lock: return jsonify(list(rejection_logs_cache))

@app.route('/api/backtest_report')
def get_backtest_report():
    """
    API endpoint to fetch backtest trade data.
    """
    if not check_db_connection():
        return jsonify({"error": "DB connection failed"}), 500
    try:
        with conn.cursor() as cur:
            # Fetch only backtest trades (is_real_trade = FALSE)
            cur.execute("""
                SELECT id, symbol, entry_price, target_price, stop_loss, status, closing_price, closed_at, 
                       profit_percentage, strategy_name, signal_details, backtest_filters, backtest_market_trend
                FROM signals 
                WHERE is_real_trade = FALSE 
                ORDER BY closed_at ASC;
            """)
            backtest_trades = [dict(row) for row in cur.fetchall()]

        # Convert datetime objects to ISO format for JSON serialization
        for trade in backtest_trades:
            if trade['closed_at']:
                trade['closed_at'] = trade['closed_at'].isoformat()
            # signal_details, backtest_filters, backtest_market_trend are already JSONB in DB,
            # so they are retrieved as Python dicts. No need to re-parse.
            # However, if they contain datetime objects, they need to be converted.
            # For simplicity, assuming they are simple JSON objects.
            # If signal_details has timestamp, ensure it's ISO formatted if it's a datetime object
            if trade.get('signal_details') and isinstance(trade['signal_details'], dict) and trade['signal_details'].get('timestamp'):
                if isinstance(trade['signal_details']['timestamp'], datetime):
                    trade['signal_details']['timestamp'] = trade['signal_details']['timestamp'].isoformat()

        return jsonify(backtest_trades)
    except Exception as e:
        logger.error(f"❌ [API Backtest Report] Error: {e}", exc_info=True)
        if conn: conn.rollback()
        return jsonify({"error": "Internal error fetching backtest report"}), 500


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

# ---------------------- Backtesting Implementation ----------------------

def run_backtest(symbol: str, days: int = 5, interval: str = '15m'):
    """
    Runs a backtest for a single symbol over a specified number of days.
    """
    logger.info(f"🚀 Starting backtest for {symbol} over {days} days with {interval} interval.")

    # Clear existing signals in DB for a clean backtest run
    if check_db_connection():
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM signals WHERE is_real_trade = FALSE;") # Delete only backtest signals
            conn.commit()
            logger.info("🧹 Cleared previous backtest signals from the database.")
        except Exception as e:
            logger.error(f"❌ Error clearing backtest signals: {e}")
            if conn: conn.rollback()

    # Fetch historical data for the symbol
    df = fetch_historical_data(symbol, interval, days)
    if df is None or df.empty:
        logger.error(f"❌ No historical data found for {symbol} for backtesting.")
        return

    # Fetch BTC data for trend analysis for the same period
    # Add buffer for indicators, ensuring enough data for EMA200 etc.
    btc_df_15m = fetch_historical_data(BTC_SYMBOL, '15m', days + 5) 
    btc_df_1h = fetch_historical_data(BTC_SYMBOL, '1h', days + 15)
    btc_df_4h = fetch_historical_data(BTC_SYMBOL, '4h', days + 50)

    if btc_df_15m is not None: btc_df_15m['btc_returns'] = btc_df_15m['close'].pct_change()
    if btc_df_1h is not None: btc_df_1h['btc_returns'] = btc_df_1h['close'].pct_change()
    if btc_df_4h is not None: btc_df_4h['btc_returns'] = btc_df_4h['close'].pct_change()

    strategy = TradingStrategy(symbol)
    if not all([strategy.ml_model, strategy.scaler, strategy.feature_names]):
        logger.error(f"❌ ML model or scaler not loaded for {symbol}. Cannot run backtest.")
        return

    backtest_trades = []
    open_backtest_signal = None # Only one open signal at a time for simplicity

    # Iterate through historical data candle by candle
    for i in range(len(df)):
        current_candle_data = df.iloc[i:i+1]
        current_timestamp = current_candle_data.index[0]
        current_price = current_candle_data['close'].iloc[0]

        # Ensure enough historical data for feature calculation
        # This check should be more robust, considering all periods used in calculate_features
        # For simplicity, we'll assume enough initial data.
        if i < max(EMA_PERIODS + [REL_VOL_PERIOD, MOMENTUM_PERIOD, ATR_PERIOD, ADX_PERIOD, PEAK_LOOKBACK_PERIOD]):
            continue

        # --- Simulate Market Trend and Filter Profile at this timestamp ---
        # Get relevant BTC data up to current_timestamp
        # Ensure enough data for indicators to be calculated for BTC slices
        min_btc_data_points_needed = max(EMA_PERIODS) # Max EMA period for BTC
        
        btc_15m_slice = btc_df_15m.loc[btc_df_15m.index <= current_timestamp] if btc_df_15m is not None else None
        if btc_15m_slice is not None and len(btc_15m_slice) < min_btc_data_points_needed: btc_15m_slice = None

        btc_1h_slice = btc_df_1h.loc[btc_df_1h.index <= current_timestamp] if btc_df_1h is not None else None
        if btc_1h_slice is not None and len(btc_1h_slice) < min_btc_data_points_needed: btc_1h_slice = None

        btc_4h_slice = btc_df_4h.loc[btc_df_4h.index <= current_timestamp] if btc_df_4h is not None else None
        if btc_4h_slice is not None and len(btc_4h_slice) < min_btc_data_points_needed: btc_4h_slice = None

        current_market_trend = determine_market_trend_score_backtest(btc_15m_slice, btc_1h_slice, btc_4h_slice)
        current_filter_profile = analyze_market_and_create_dynamic_profile_backtest(current_market_trend, current_timestamp)
        active_strategy_type = current_filter_profile.get("strategy")

        if active_strategy_type == "DISABLED":
            logger.debug(f"🛑 Trading disabled by profile at {current_timestamp}. Skipping.")
            if open_backtest_signal:
                # If trading is disabled, close any open positions immediately
                profit_pct = ((current_price / open_backtest_signal['entry_price']) - 1) * 100
                open_backtest_signal.update({
                    'status': 'disabled_close',
                    'closing_price': current_price,
                    'closed_at': current_timestamp,
                    'profit_percentage': profit_pct
                })
                insert_signal_into_db(open_backtest_signal)
                backtest_trades.append(open_backtest_signal)
                open_backtest_signal = None
            continue

        # --- Check for signal closure if a trade is open ---
        if open_backtest_signal:
            entry_price = open_backtest_signal['entry_price']
            target_price = open_backtest_signal['target_price']
            stop_loss = open_backtest_signal['stop_loss']

            # Update current price and PnL for open signal in backtest
            open_backtest_signal['current_price'] = current_price
            open_backtest_signal['pnl_pct'] = ((current_price / entry_price) - 1) * 100

            # Trailing Stop Loss logic (simplified for backtest)
            effective_stop_loss = stop_loss
            if USE_TRAILING_STOP_LOSS:
                activation_price = entry_price * (1 + TRAILING_ACTIVATION_PROFIT_PERCENT / 100)
                if current_price > activation_price:
                    current_peak = open_backtest_signal.get('current_peak_price', entry_price)
                    if current_price > current_peak:
                        open_backtest_signal['current_peak_price'] = current_price
                        current_peak = current_price
                    
                    trailing_stop_price = current_peak * (1 - TRAILING_DISTANCE_PERCENT / 100)
                    if trailing_stop_price > effective_stop_loss:
                        effective_stop_loss = trailing_stop_price

            status_to_set = None
            if current_price >= target_price:
                status_to_set = 'target_hit'
            elif current_price <= effective_stop_loss:
                status_to_set = 'stop_loss_hit'
            
            if status_to_set:
                profit_pct = ((current_price / entry_price) - 1) * 100
                open_backtest_signal.update({
                    'status': status_to_set,
                    'closing_price': current_price,
                    'closed_at': current_timestamp,
                    'profit_percentage': profit_pct
                })
                insert_signal_into_db(open_backtest_signal)
                backtest_trades.append(open_backtest_signal)
                open_backtest_signal = None
                logger.info(f"✅ Backtest: Signal closed for {symbol} at {current_timestamp} due to {status_to_set}.")
                continue # Move to next candle after closing trade

        # --- Check for new buy signal if no trade is open ---
        if not open_backtest_signal:
            # Prepare data for feature calculation up to the current candle
            df_slice_15m = df.iloc[:i+1]
            if len(df_slice_15m) < SIGNAL_GENERATION_LOOKBACK_DAYS * (24*60 / int(re.sub('[a-zA-Z]', '', SIGNAL_GENERATION_TIMEFRAME))):
                continue # Not enough data for lookback period

            df_features_with_indicators = calculate_features(df_slice_15m, btc_15m_slice)
            if df_features_with_indicators is None or df_features_with_indicators.empty:
                continue
            df_features_with_indicators.name = symbol

            technical_signal = None
            if active_strategy_type == "REVERSAL":
                technical_signal = find_crazy_reversal_signal(df_features_with_indicators)
            elif active_strategy_type == "MOMENTUM":
                technical_signal = {"signal_type": "MOMENTUM"}

            if not technical_signal:
                continue
            
            # Fetch 4h data slice for features
            # Adjust end_str to current_timestamp for accurate historical slice
            df_4h_slice = fetch_historical_data(symbol, '4h', SIGNAL_GENERATION_LOOKBACK_DAYS, end_str=current_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'))
            if df_4h_slice is None or df_4h_slice.empty:
                continue

            df_features = strategy.get_features(df_slice_15m, df_4h_slice, btc_15m_slice)
            if df_features is None or df_features.empty:
                continue
            
            ml_signal = strategy.generate_buy_signal(df_features)
            
            if not ml_signal or ml_signal['confidence'] < BUY_CONFIDENCE_THRESHOLD:
                logger.debug(f"Backtest: ML signal rejected for {symbol} at {current_timestamp}.")
                continue
            
            entry_price = current_price # Entry price is the close of the current candle
            last_features = df_features.iloc[-1]
            last_atr = last_features.get('atr', 0)
            tp_sl_data = calculate_tp_sl(symbol, entry_price, last_atr)
            
            if not tp_sl_data or not passes_filters(symbol, last_features, current_filter_profile, entry_price, tp_sl_data, df_slice_15m):
                logger.debug(f"Backtest: Filters not passed for {symbol} at {current_timestamp}.")
                continue

            order_book_analysis = analyze_order_book(symbol, entry_price) # Simplified for backtest
            if not order_book_analysis or not passes_order_book_check(symbol, order_book_analysis, current_filter_profile):
                logger.debug(f"Backtest: Order book check failed for {symbol} at {current_timestamp}.")
                continue
            
            strategy_name_for_db = f"Reversal_ML" if active_strategy_type == "REVERSAL" else f"Momentum_ML"
            new_signal = {
                'symbol': symbol, 'strategy_name': strategy_name_for_db,
                'signal_details': {
                    'ML_Confidence': ml_signal['confidence'],
                    'ML_Confidence_Display': f"{ml_signal['confidence']:.2%}",
                    'Filter_Profile': f"{current_filter_profile['name']}",
                    'Technical_Reason': technical_signal.get('reason', 'N/A'),
                    'Bid_Ask_Ratio': order_book_analysis.get('bid_ask_ratio', 0),
                    'timestamp': current_timestamp.isoformat() # Add signal timestamp
                },
                'entry_price': entry_price, 
                'current_price': entry_price, # Initialize current price to entry
                'current_peak_price': entry_price, # Initialize peak price
                'is_real_trade': False, # Backtest trades are not real
                'quantity': float(calculate_position_size(symbol, entry_price, tp_sl_data['stop_loss'])),
                'order_id': f"BACKTEST_ORDER_{random.randint(1000, 9999)}",
                'status': 'open',
                'closed_at': None,
                'profit_percentage': None,
                'backtest_filters': current_filter_profile['filters'], # Save filter values
                'backtest_market_trend': current_market_trend, # Save market trend state
                **tp_sl_data
            }
            
            # Simulate placing the order
            place_order(symbol, Client.SIDE_BUY, Decimal(str(new_signal['quantity'])))
            
            open_backtest_signal = new_signal
            logger.info(f"✅ Backtest: New signal opened for {symbol} at {current_timestamp}.")

    # Close any remaining open signals at the end of the backtest period
    if open_backtest_signal:
        profit_pct = ((current_price / open_backtest_signal['entry_price']) - 1) * 100
        open_backtest_signal.update({
            'status': 'closed_end_of_backtest',
            'closing_price': current_price,
            'closed_at': df.index[-1], # Use the last timestamp of the data
            'profit_percentage': profit_pct
        })
        insert_signal_into_db(open_backtest_signal)
        backtest_trades.append(open_backtest_signal)
        logger.info(f"✅ Backtest: Remaining signal closed for {symbol} at end of backtest.")

    logger.info(f"🏁 Backtest for {symbol} finished. Total trades: {len(backtest_trades)}")
    generate_backtest_report(backtest_trades)


def generate_backtest_report(trades: List[Dict[str, Any]]):
    """
    Generates and prints a detailed report for backtest trades.
    """
    logger.info("📊 Generating Backtest Report...")
    if not trades:
        logger.info("No trades were executed during the backtest period.")
        return

    total_profit_pct = 0.0
    total_trades = len(trades)
    winning_trades = [t for t in trades if t['profit_percentage'] is not None and t['profit_percentage'] > 0]
    losing_trades = [t for t in trades if t['profit_percentage'] is not None and t['profit_percentage'] < 0]

    total_wins = len(winning_trades)
    total_losses = len(losing_trades)

    win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    avg_win_pct = sum(t['profit_percentage'] for t in winning_trades) / total_wins if total_wins > 0 else 0
    avg_loss_pct = sum(t['profit_percentage'] for t in losing_trades) / total_losses if total_losses > 0 else 0
    
    gross_profit = sum(t['profit_percentage'] for t in winning_trades)
    gross_loss = sum(abs(t['profit_percentage']) for t in losing_trades)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    report_lines = [
        "\n--- Backtest Report ---",
        f"Symbol: {trades[0]['symbol'] if trades else 'N/A'}",
        f"Total Trades: {total_trades}",
        f"Winning Trades: {total_wins} ({win_rate:.2f}%)",
        f"Losing Trades: {total_losses}",
        f"Average Win %: {avg_win_pct:.2f}%",
        f"Average Loss %: {avg_loss_pct:.2f}%",
        f"Gross Profit %: {gross_profit:.2f}%",
        f"Gross Loss %: {gross_loss:.2f}%",
        f"Profit Factor: {profit_factor:.2f}" if profit_factor != float('inf') else "Profit Factor: Infinity",
        "\n--- Detailed Trades ---"
    ]

    for trade in trades:
        entry_time = trade['signal_details'].get('timestamp', 'N/A')
        close_time = trade['closed_at'].isoformat() if trade['closed_at'] else 'N/A'
        
        filter_values_str = json.dumps(trade.get('backtest_filters', {}), indent=2)
        market_trend_str = json.dumps(trade.get('backtest_market_trend', {}), indent=2)

        report_lines.append(f"\nTrade ID: {trade['id']}")
        report_lines.append(f"  Symbol: {trade['symbol']}")
        report_lines.append(f"  Status: {trade['status']}")
        report_lines.append(f"  Entry Price: {trade['entry_price']:.8f}")
        report_lines.append(f"  Closing Price: {trade['closing_price']:.8f}" if trade['closing_price'] else "  Closing Price: N/A")
        report_lines.append(f"  Profit %: {trade['profit_percentage']:.2f}%" if trade['profit_percentage'] is not None else "  Profit %: N/A")
        report_lines.append(f"  Entry Time: {entry_time}")
        report_lines.append(f"  Close Time: {close_time}")
        report_lines.append(f"  Strategy: {trade['strategy_name']}")
        report_lines.append(f"  Filters at Signal: \n{filter_values_str}")
        report_lines.append(f"  Market Trend at Signal: \n{market_trend_str}")
        report_lines.append("-" * 30)

    for line in report_lines:
        logger.info(line)

# ---------------------- Program Entry Point ----------------------
def run_websocket_manager():
    # This function is for live trading. Not used in backtesting.
    pass

def initialize_bot_services():
    global client, validated_symbols_to_scan
    logger.info("🤖 [Bot Services] Starting background initialization...")
    try:
        # Initialize Binance client for fetching historical data
        client = Client(API_KEY, API_SECRET)
        init_db()
        # No need for Redis for backtesting
        get_exchange_info_map()
        # No need to load open signals or notifications for backtesting
        
        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan:
            logger.critical("❌ No validated symbols to scan. Bot will not start."); return
        
        # In a real application, you'd run these in separate threads.
        # For backtesting, we call run_backtest directly.
        # Thread(target=determine_market_trend_score, daemon=True).start()
        # Thread(target=run_websocket_manager, daemon=True).start()
        # Thread(target=trade_monitoring_loop, daemon=True).start()
        # Thread(target=main_loop, daemon=True).start()
        logger.info("✅ [Bot Services] All background services started successfully for backtest setup.")
    except Exception as e:
        log_and_notify("critical", f"A critical error occurred during initialization: {e}", "SYSTEM")
        exit(1)

if __name__ == "__main__":
    logger.info("🚀 LAUNCHING TRADING BOT & DASHBOARD (V27.2 - Improved Logic) 🚀")
    
    # Initialize services (DB, Binance Client, etc.)
    initialize_bot_services()

    # --- Run the Backtest ---
    # You can change the symbol and number of days here
    backtest_symbol = "ETHUSDT" # Example symbol
    backtest_days = 5
    backtest_interval = '15m' # Use 15m interval for backtest candles

    if client and backtest_symbol in validated_symbols_to_scan:
        run_backtest(backtest_symbol, backtest_days, backtest_interval)
    else:
        logger.error(f"❌ Cannot run backtest. Either Binance client is not initialized or {backtest_symbol} is not a validated symbol.")

    # The Flask app is still available if you want to view the dashboard
    # However, the backtest results are primarily logged to console/file and DB.
    # To view backtest results in the dashboard, you'd need to fetch them from DB.
    # For this script, the focus is on the console report.
    run_flask() # Uncomment this if you want to run the Flask dashboard alongside the backtest
    logger.info("👋 [Shutdown] Application has been shut down."); os._exit(0)

