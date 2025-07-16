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
# [FIX] Import requests session management tools
from requests.adapters import HTTPAdapter
from requests import Session


# --- تجاهل التحذيرات غير الهامة ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ---------------------- إعداد نظام التسجيل (Logging) - V28 (Market Context Model) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v28_arabic_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV28')
# Silence noisy connection pool warnings as we are managing it now
logging.getLogger("urllib3").setLevel(logging.ERROR)


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

# ---------------------- V28: تعريفات فلاتر جديدة تتوافق مع أنظمة السوق ----------------------
FILTER_PROFILES: Dict[str, Dict[str, Any]] = {
    "PRIME_BULLISH": {
        "description": "أفضل حالة للشراء (زخم قوي مع اتجاه عام صاعد)",
        "strategy": "MOMENTUM",
        "filters": {
            "adx": 25.0, "rel_vol": 0.2, "rsi_range": (55, 95), "roc": 0.08,
            "slope": 0.04, "min_rrr": 1.3, "min_volatility_pct": 0.20,
            "min_btc_correlation": -0.1, "min_bid_ask_ratio": 1.15
        }
    },
    "PULLBACK_BULLISH": {
        "description": "شراء التصحيحات (اتجاه عام صاعد مع ضعف مؤقت)",
        "strategy": "MOMENTUM",
        "filters": {
            "adx": 20.0, "rel_vol": 0.1, "rsi_range": (45, 90), "roc": -0.5,
            "slope": -0.1, "min_rrr": 1.2, "min_volatility_pct": 0.15,
            "min_btc_correlation": -0.2, "min_bid_ask_ratio": 1.1
        }
    },
    "CAUTIOUS_BULLISH": {
        "description": "شراء بحذر (اتجاه عام مختلط مع زخم لحظي قوي)",
        "strategy": "MOMENTUM",
        "filters": {
            "adx": 28.0, "rel_vol": 0.3, "rsi_range": (60, 95), "roc": 0.1,
            "slope": 0.05, "min_rrr": 1.6, "min_volatility_pct": 0.25,
            "min_btc_correlation": 0.0, "min_bid_ask_ratio": 1.2
        }
    },
    "DISABLED": {
        "description": "التداول متوقف (السوق غير مناسب)",
        "strategy": "DISABLED",
        "filters": {}
    }
}


# ---------------------- الثوابت والمتغيرات العامة ----------------------
is_trading_enabled: bool = False
trading_status_lock = Lock()
RISK_PER_TRADE_PERCENT: float = 1.0
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V8_With_Momentum'
MODEL_FOLDER: str = 'V8'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
CONTEXT_TIMEFRAMES: Dict[str, int] = {'4h': 100, '1h': 75}
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices_v28"
TRADING_FEE_PERCENT: float = 0.1
STATS_TRADE_SIZE_USDT: float = 10.0
BTC_SYMBOL: str = 'BTCUSDT'
# Reduced batch size to prevent connection pool overflow and websocket queue issues.
SYMBOL_PROCESSING_BATCH_SIZE: int = 10
ADX_PERIOD: int = 14; RSI_PERIOD: int = 14; ATR_PERIOD: int = 14
EMA_FAST_PERIOD: int = 50; EMA_SLOW_PERIOD: int = 200
REL_VOL_PERIOD: int = 30; MOMENTUM_PERIOD: int = 12; EMA_SLOPE_PERIOD: int = 5
MAX_OPEN_TRADES: int = 4
BUY_CONFIDENCE_THRESHOLD = 0.80
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_ACTIVATION_PROFIT_PERCENT: float = 1.0
TRAILING_DISTANCE_PERCENT: float = 0.8
MARKET_CONTEXT_CHECK_INTERVAL: int = 300 # 5 دقائق

conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
ml_models_cache: Dict[str, Any] = {}; exchange_info_map: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}; signal_cache_lock = Lock()
notifications_cache = deque(maxlen=50); notifications_lock = Lock()
signals_pending_closure: Set[int] = set(); closure_lock = Lock()
rejection_logs_cache = deque(maxlen=100); rejection_logs_lock = Lock()
last_market_context_check_time: float = 0
current_market_context: Dict[str, Any] = {"regime": "INITIALIZING", "details": {}}
market_context_lock = Lock()
dynamic_filter_profile_cache: Dict[str, Any] = {}

# [FIX] Batching mechanism for Redis price updates
price_updates_batch: Dict[str, str] = {}
price_updates_lock = Lock()

# [FIX FOR BALANCE DISPLAY] Caching mechanism for USDT balance
usdt_balance_cache: Optional[float] = None
usdt_balance_lock = Lock()


REJECTION_REASONS_AR = {
    "Filters Not Loaded": "الفلاتر غير محملة",
    "Low Volatility": "تقلب منخفض جداً",
    "ADX Filter": "فلتر ADX لم يتحقق",
    "RSI Filter": "فلتر RSI خارج النطاق",
    "ROC Filter": "فلتر ROC لم يتحقق",
    "Slope Filter": "فلتر الميل لم يتحقق",
    "BTC Correlation": "ارتباط ضعيف بالبيتكوين",
    "RRR Filter": "نسبة المخاطرة/العائد غير كافية",
    "Momentum/Strength Filter": "فلتر الزخم والقوة",
    "Invalid ATR for TP/SL": "ATR غير صالح لحساب الأهداف",
    "ML Model Rejected": "نموذج التعلم الآلي رفض الإشارة",
    "Invalid Position Size": "حجم الصفقة غير صالح (الوقف تحت الدخول)",
    "Lot Size Adjustment Failed": "فشل ضبط حجم العقد (LOT_SIZE)",
    "Min Notional Filter": "قيمة الصفقة أقل من الحد الأدنى",
    "Insufficient Balance": "الرصيد غير كافٍ",
    "Order Book Fetch Failed": "فشل جلب دفتر الطلبات",
    "Order Book Imbalance": "اختلال توازن دفتر الطلبات (ضغط بيع)",
    "Max Open Trades Reached": "تم الوصول للحد الأقصى للصفقات المفتوحة",
    "Data Fetch Failed": "فشل جلب بيانات العملة",
    "Signal Already Open": "توجد صفقة مفتوحة بالفعل لهذه العملة"
}

# ---------------------- دالة HTML للوحة التحكم (تبقى كما هي) ----------------------
def get_dashboard_html():
    # This function remains unchanged.
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم التداول V28</title>
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
        .card { background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 0.5rem; }
        .skeleton { animation: pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite; background-color: #21262d; border-radius: 0.5rem; }
        @keyframes pulse { 50% { opacity: .6; } }
        .tab-btn { position: relative; transition: color 0.2s ease; }
        .tab-btn.active { color: var(--text-primary); }
        .tab-btn.active::after { content: ''; position: absolute; bottom: -1px; left: 0; right: 0; height: 2px; background-color: var(--accent-blue); border-radius: 2px; }
        .toggle-bg:after { content: ''; position: absolute; top: 2px; left: 2px; background: white; border-radius: 9999px; height: 1.25rem; width: 1.25rem; transition: transform 0.2s ease-in-out; }
        input:checked + .toggle-bg:after { transform: translateX(100%); }
        input:checked + .toggle-bg { background-color: var(--accent-green); }
    </style>
</head>
<body class="p-4 md:p-6">
    <div class="container mx-auto max-w-screen-2xl">
        <header class="mb-6 flex flex-wrap justify-between items-center gap-4">
            <h1 class="text-2xl md:text-3xl font-extrabold text-white">
                <span class="text-accent-blue">لوحة تحكم التداول</span>
                <span class="text-text-secondary font-medium">V28</span>
            </h1>
            <div id="connection-status" class="flex items-center gap-3 text-sm">
                <div class="flex items-center gap-2"><div id="db-status-light" class="w-2.5 h-2.5 rounded-full bg-gray-600 animate-pulse"></div><span class="text-text-secondary">DB</span></div>
                <div class="flex items-center gap-2"><div id="api-status-light" class="w-2.5 h-2.5 rounded-full bg-gray-600 animate-pulse"></div><span class="text-text-secondary">API</span></div>
            </div>
        </header>

        <section class="mb-6 card p-5">
            <h2 class="text-xl font-bold mb-4 text-center">تحليل سياق السوق (BTC)</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-5">
                <div class="bg-black/20 p-4 rounded-lg border border-border-color">
                    <h3 class="font-bold text-text-secondary text-center mb-3">1. التحيز العام (HTF Bias)</h3>
                    <div id="htf-bias-container" class="space-y-3">
                        <div class="skeleton h-8 w-full"></div>
                        <div class="skeleton h-8 w-full"></div>
                    </div>
                </div>
                <div class="bg-black/20 p-4 rounded-lg border border-border-color">
                    <h3 class="font-bold text-text-secondary text-center mb-3">2. الحالة الفورية (15m Condition)</h3>
                    <div id="condition-15m-container" class="space-y-3">
                        <div class="skeleton h-8 w-full"></div>
                        <div class="skeleton h-8 w-full"></div>
                        <div class="skeleton h-8 w-full"></div>
                    </div>
                </div>
                <div class="bg-blue-900/20 p-4 rounded-lg border border-accent-blue flex flex-col justify-center items-center">
                    <h3 class="font-bold text-text-secondary text-center mb-3">3. القرار النهائي</h3>
                    <div id="final-regime" class="text-3xl font-extrabold text-center skeleton h-10 w-3/4"></div>
                    <div id="active-strategy" class="text-lg text-accent-blue text-center mt-2 skeleton h-7 w-full"></div>
                </div>
            </div>
        </section>

        <section class="mb-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
            <div class="card p-4 flex flex-col justify-center items-center">
                <h3 class="font-bold text-lg text-text-secondary mb-2">التحكم بالتداول</h3>
                <div class="flex items-center space-x-3 space-x-reverse">
                    <span id="trading-status-text" class="font-bold text-lg text-accent-red">غير مُفعَّل</span>
                    <label for="trading-toggle" class="flex items-center cursor-pointer">
                        <div class="relative"><input type="checkbox" id="trading-toggle" class="sr-only" onchange="toggleTrading()"><div class="toggle-bg block bg-accent-red w-12 h-7 rounded-full"></div></div>
                    </label>
                </div>
                 <div class="mt-2 text-xs text-text-secondary">رصيد USDT: <span id="usdt-balance" class="font-mono skeleton w-20 inline-block"></span></div>
            </div>
            <div class="card p-4 flex flex-col justify-center items-center text-center">
                <h3 class="font-bold text-text-secondary text-lg">صفقات مفتوحة</h3>
                <div id="open-trades-value" class="text-5xl font-black text-accent-blue mt-2 skeleton h-12 w-1/2"></div>
            </div>
            <div id="profit-chart-card" class="card lg:col-span-2 p-4">
                <div class="flex justify-between items-center mb-3">
                    <h3 class="font-bold text-lg text-text-secondary">منحنى الربح التراكمي (%)</h3>
                    <div id="net-profit-usdt" class="text-2xl font-bold skeleton h-8 w-1/3"></div>
                </div>
                <div class="relative h-60"><canvas id="profitChart"></canvas></div>
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
            <div id="signals-tab" class="tab-content"><div class="overflow-x-auto card p-0"><table class="min-w-full text-sm text-right"><thead class="border-b border-border-color bg-black/20"><tr><th class="p-4 font-semibold text-text-secondary">العملة</th><th class="p-4 font-semibold text-text-secondary">الحالة</th><th class="p-4 font-semibold text-text-secondary">الربح/الخسارة</th><th class="p-4 font-semibold text-text-secondary">الدخول/الحالي</th><th class="p-4 font-semibold text-text-secondary">إجراء</th></tr></thead><tbody id="signals-table"></tbody></table></div></div>
            <div id="stats-tab" class="tab-content hidden"><div id="stats-container" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4"></div></div>
            <div id="notifications-tab" class="tab-content hidden"><div id="notifications-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-2"></div></div>
            <div id="rejections-tab" class="tab-content hidden"><div id="rejections-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-2"></div></div>
            <div id="filters-tab" class="tab-content hidden"><div id="filters-display" class="card p-4 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4"></div></div>
        </main>
    </div>

<script>
let profitChartInstance;
const REGIME_STYLES = {
    "PRIME_BULLISH": { text: "صاعد مثالي", color: "text-accent-green" },
    "PULLBACK_BULLISH": { text: "تصحيح صاعد", color: "text-green-400" },
    "CAUTIOUS_BULLISH": { text: "صاعد بحذر", color: "text-cyan-400" },
    "RANGING_NEUTRAL": { text: "عرضي/محايد", color: "text-accent-yellow" },
    "HIGH_RISK_CHOP": { text: "خطر عالٍ/فوضى", color: "text-orange-500" },
    "BEARISH": { text: "هابط", color: "text-accent-red" },
    "INITIALIZING": { text: "تهيئة...", color: "text-accent-blue" }
};
const BIAS_STYLES = {
    "Bullish": { text: "صاعد", color: "bg-green-500/20 text-green-300", icon: "▲" },
    "Bearish": { text: "هابط", color: "bg-red-500/20 text-red-300", icon: "▼" },
    "Mixed": { text: "مختلط", color: "bg-yellow-500/20 text-yellow-300", icon: "↔" },
    "Neutral": { text: "محايد", color: "bg-gray-500/20 text-gray-300", icon: "—" }
};
const CONDITION_STYLES = {
    "Healthy": { text: "صحي", color: "text-green-400", icon: "✓" },
    "Weak": { text: "ضعيف", color: "text-yellow-400", icon: "!" },
    "Unhealthy": { text: "غير صحي", color: "text-red-400", icon: "✗" }
};

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

function renderStatus(title, value, style) {
    return `<div class="flex items-center justify-between text-lg p-2 rounded-md ${style.color}">
                <span class="font-bold">${title}</span>
                <span class="flex items-center gap-2">${value} <span class="text-xl">${style.icon}</span></span>
            </div>`;
}

function updateMarketStatus() {
    apiFetch('/api/market_status').then(data => {
        if (!data || data.error) return;
        
        document.getElementById('db-status-light').className = `w-2.5 h-2.5 rounded-full ${data.db_ok ? 'bg-green-500' : 'bg-red-500'}`;
        document.getElementById('api-status-light').className = `w-2.5 h-2.5 rounded-full ${data.api_ok ? 'bg-green-500' : 'bg-red-500'}`;
        
        const usdtBalanceEl = document.getElementById('usdt-balance');
        usdtBalanceEl.textContent = data.usdt_balance ? `$${formatNumber(data.usdt_balance, 2)}` : 'N/A';
        usdtBalanceEl.classList.remove('skeleton', 'w-20');

        const context = data.market_context;
        const details = context.details || {};
        
        const htfContainer = document.getElementById('htf-bias-container');
        const bias4h = details.bias_4h || "Neutral";
        const bias1h = details.bias_1h || "Neutral";
        htfContainer.innerHTML = `
            ${renderStatus('4 ساعات', BIAS_STYLES[bias4h].text, BIAS_STYLES[bias4h])}
            ${renderStatus('1 ساعة', BIAS_STYLES[bias1h].text, BIAS_STYLES[bias1h])}
        `;

        const conditionContainer = document.getElementById('condition-15m-container');
        const structure = details.structure_15m || { state: "Unhealthy" };
        const momentum = details.momentum_15m || { state: "Unhealthy" };
        const volatility = details.volatility_15m || { state: "Unhealthy" };
        conditionContainer.innerHTML = `
            ${renderStatus('الهيكل', structure.state, CONDITION_STYLES[structure.state])}
            ${renderStatus('الزخم', momentum.state, CONDITION_STYLES[momentum.state])}
            ${renderStatus('التقلب', volatility.state, CONDITION_STYLES[volatility.state])}
        `;
        
        const regime = context.regime || "INITIALIZING";
        const regimeStyle = REGIME_STYLES[regime] || REGIME_STYLES["BEARISH"];
        const finalRegimeEl = document.getElementById('final-regime');
        finalRegimeEl.textContent = regimeStyle.text;
        finalRegimeEl.className = `text-3xl font-extrabold text-center ${regimeStyle.color}`;

        const profile = data.filter_profile;
        const activeStrategyEl = document.getElementById('active-strategy');
        activeStrategyEl.textContent = profile.description || "---";
        activeStrategyEl.className = `text-lg text-accent-blue text-center mt-2`;

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
            <div class="card text-center p-4 flex flex-col justify-center"><div class="text-sm text-text-secondary mb-1">إجمالي الصفقات</div><div class="text-3xl font-bold text-text-primary">${formatNumber(data.total_closed_trades, 0)}</div></div>
        `;
    });
}

function updateProfitChart() {
    apiFetch('/api/profit_curve').then(data => {
        if (!data || data.error || data.length <= 1) { 
            document.getElementById('profit-chart-card').classList.add('hidden');
            return; 
        }
        document.getElementById('profit-chart-card').classList.remove('hidden');
        
        const ctx = document.getElementById('profitChart').getContext('2d');
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
                backgroundColor: gradient, fill: true, tension: 0.4, pointRadius: 0
            }]},
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    x: { type: 'time', time: { unit: 'day' }, grid: { display: false }, ticks: { color: 'var(--text-secondary)'} },
                    y: { position: 'right', grid: { color: 'var(--border-color)' }, ticks: { color: 'var(--text-secondary)', callback: v => v + '%' } }
                },
                plugins: { legend: { display: false } }
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

function updateSignals() {
    apiFetch('/api/signals').then(data => {
        const tableBody = document.getElementById('signals-table');
        if (!data || data.error) { tableBody.innerHTML = '<tr><td colspan="5" class="p-8 text-center text-text-secondary">فشل تحميل الصفقات.</td></tr>'; return; }
        if (data.length === 0) { tableBody.innerHTML = '<tr><td colspan="5" class="p-8 text-center text-text-secondary">لا توجد صفقات لعرضها.</td></tr>'; return; }
        
        tableBody.innerHTML = data.map(signal => {
            const pnlPct = (signal.status === 'open' || signal.status === 'updated') ? signal.pnl_pct : signal.profit_percentage;
            const pnlDisplay = pnlPct !== null && pnlPct !== undefined ? `${formatNumber(pnlPct)}%` : 'N/A';
            const pnlColor = pnlPct === null || pnlPct === undefined ? 'text-text-secondary' : (pnlPct >= 0 ? 'text-accent-green' : 'text-accent-red');
            let statusClass = 'text-yellow-400';
            let statusText = 'مفتوحة';
            if (signal.status !== 'open' && signal.status !== 'updated') {
                statusClass = 'text-gray-400';
                statusText = 'مغلقة';
            }
            const realTradeIndicator = signal.is_real_trade ? '<span class="text-accent-green" title="صفقة حقيقية">●</span>' : '';
            
            return `<tr class="border-b border-border-color">
                    <td class="p-4 font-mono font-semibold">${realTradeIndicator} ${signal.symbol}</td>
                    <td class="p-4 font-bold ${statusClass}">${statusText}</td>
                    <td class="p-4 font-mono font-bold ${pnlColor}">${pnlDisplay}</td>
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

function refreshData() {
    updateMarketStatus();
    updateTradingStatus();
    updateStats();
    updateProfitChart();
    updateSignals();
    const dateLocaleOptions = { timeZone: 'UTC', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false };
    const locale = 'fr-CA'; // Use a neutral locale like Canadian French for YYYY-MM-DD format
    updateList('/api/notifications', 'notifications-list', n => `<div class="p-3 rounded-md bg-gray-900/50 text-sm">[${new Date(n.timestamp).toLocaleString(locale, dateLocaleOptions)}] ${n.message}</div>`);
    updateList('/api/rejection_logs', 'rejections-list', log => `<div class="p-3 rounded-md bg-gray-900/50 text-sm">[${new Date(log.timestamp).toLocaleString(locale, dateLocaleOptions)}] <strong>${log.symbol}</strong>: ${log.reason} - <span class="font-mono text-xs text-text-secondary">${JSON.stringify(log.details)}</span></div>`);
}

setInterval(refreshData, 5000);
window.onload = refreshData;
</script>
</body>
</html>
    """

# ---------------------- دوال قاعدة البيانات (تبقى كما هي) ----------------------
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
                        status TEXT DEFAULT 'open', closing_price DOUBLE PRECISION, closed_at TIMESTAMP WITH TIME ZONE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(), profit_percentage DOUBLE PRECISION,
                        strategy_name TEXT, signal_details JSONB, current_peak_price DOUBLE PRECISION,
                        is_real_trade BOOLEAN DEFAULT FALSE, quantity DOUBLE PRECISION, order_id TEXT
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL, message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals (status);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals (created_at DESC);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_closed_at ON signals (closed_at);")
            conn.commit()
            logger.info("✅ [DB] Database connection and schema are up-to-date.")
            return
        except Exception as e:
            logger.error(f"❌ [DB] Error during initialization (Attempt {attempt + 1}/{retries}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else:
                logger.critical("❌ [DB] Failed to connect to the database after multiple retries.")
                return 

def check_db_connection() -> bool:
    global conn
    if conn is None or conn.closed != 0:
        logger.warning("[DB] Connection is not available or closed, attempting to reconnect...")
        init_db()
    if conn is None or conn.closed != 0:
        return False
    try:
        with conn.cursor() as cur: cur.execute("SELECT 1;")
        return True
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [DB] Connection lost: {e}. Reconnecting...")
        init_db()
        return conn is not None and conn.closed == 0
    return False

def log_and_notify(level: str, message: str, notification_type: str):
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error, 'critical': logger.critical}
    log_methods.get(level.lower(), logger.info)(message)
    
    new_notification = {"timestamp": datetime.now(timezone.utc).isoformat(), "type": notification_type, "message": message}
    with notifications_lock:
        notifications_cache.appendleft(new_notification)

    if not check_db_connection() or not conn:
        logger.warning(f"[Notify DB] Could not save notification, DB connection unavailable. Message: {message}")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [Notify DB] Failed to save notification: {e}")
        if conn: conn.rollback()

def log_rejection(symbol: str, reason_key: str, details: Optional[Dict] = None):
    reason_ar = REJECTION_REASONS_AR.get(reason_key, reason_key)
    log_message = f"🚫 [REJECTED] {symbol} | Reason: {reason_ar} | Details: {details or {}}"
    # Use a lower level log to avoid spamming the main log file
    logger.debug(log_message) 
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

# ---------------------- دوال Binance والبيانات (تبقى كما هي) ----------------------
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

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        interval_in_minutes = {'1d': 1440, '4h': 240, '1h': 60, '15m': 15}
        limit = int((days * 1440) / interval_in_minutes.get(interval, 15))
        
        klines = client.get_historical_klines(symbol, interval, limit=min(limit, 1000))
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
        logger.error(f"❌ [Data] Error fetching historical data for {symbol} on {interval}: {e}")
        return None

# ---------------------- دوال تحليل سياق السوق (تبقى كما هي) ----------------------
def calculate_context_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['ema_fast'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=200, adjust=False).mean()
    df['ema_15m_fast'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_15m_slow'] = df['close'].ewm(span=50, adjust=False).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(com=14 - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=14 - 1, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    bb_period = 20
    df['bb_mid'] = df['close'].rolling(window=bb_period).mean()
    df['bb_std'] = df['close'].rolling(window=bb_period).std()
    df['bb_width'] = ((df['bb_mid'] + 2 * df['bb_std']) - (df['bb_mid'] - 2 * df['bb_std'])) / df['bb_mid']
    return df

def get_htf_bias(data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    bias = {}
    for tf, df in data.items():
        if df is None or df.empty:
            bias[f'bias_{tf}'] = "Neutral"
            continue
        last = df.iloc[-1]
        if last['close'] > last['ema_fast'] and last['ema_fast'] > last['ema_slow']:
            bias[f'bias_{tf}'] = "Bullish"
        elif last['close'] < last['ema_fast'] and last['ema_fast'] < last['ema_slow']:
            bias[f'bias_{tf}'] = "Bearish"
        else:
            bias[f'bias_{tf}'] = "Neutral"
    return bias

def get_15m_condition(df_15m: pd.DataFrame) -> Dict:
    if df_15m is None or df_15m.empty: return {}
    last = df_15m.iloc[-1]
    structure_state = "Unhealthy"
    if last['ema_15m_fast'] > last['ema_15m_slow']: structure_state = "Healthy"
    momentum_state = "Weak"
    if last['rsi'] > 60: momentum_state = "Healthy"
    elif last['rsi'] < 45: momentum_state = "Unhealthy"
    volatility_state = "Weak"
    bbw_avg = df_15m['bb_width'].rolling(window=50).mean().iloc[-1]
    if last['bb_width'] > bbw_avg * 0.8 and last['bb_width'] < bbw_avg * 2.5:
        volatility_state = "Healthy"
    elif last['bb_width'] >= bbw_avg * 2.5:
        volatility_state = "Unhealthy"
    return {
        "structure_15m": {"state": structure_state}, "momentum_15m": {"state": momentum_state},
        "volatility_15m": {"state": volatility_state}
    }

def determine_market_context_for_scalping():
    global current_market_context, last_market_context_check_time
    with market_context_lock:
        if time.time() - last_market_context_check_time < MARKET_CONTEXT_CHECK_INTERVAL: return
    logger.info("🧠 [Market Context V28] Updating market context for 15m trading...")
    try:
        htf_data = {tf: fetch_historical_data(BTC_SYMBOL, tf, days) for tf, days in CONTEXT_TIMEFRAMES.items()}
        df_15m = fetch_historical_data(BTC_SYMBOL, '15m', 5)
        processed_htf_data = {tf: calculate_context_indicators(df) for tf, df in htf_data.items() if df is not None}
        processed_15m_data = calculate_context_indicators(df_15m) if df_15m is not None else None
        htf_bias = get_htf_bias(processed_htf_data)
        condition_15m = get_15m_condition(processed_15m_data)
        bias_4h = htf_bias.get('bias_4h', 'Neutral'); bias_1h = htf_bias.get('bias_1h', 'Neutral')
        struct_15m = condition_15m.get('structure_15m', {}).get('state')
        momentum_15m = condition_15m.get('momentum_15m', {}).get('state')
        volatility_15m = condition_15m.get('volatility_15m', {}).get('state')
        regime = "BEARISH"
        if bias_4h == "Bearish" or bias_1h == "Bearish": regime = "BEARISH"
        elif bias_4h == "Bullish" and bias_1h == "Bullish":
            if struct_15m == "Healthy" and momentum_15m == "Healthy" and volatility_15m == "Healthy": regime = "PRIME_BULLISH"
            elif struct_15m == "Healthy" and volatility_15m == "Healthy": regime = "PULLBACK_BULLISH"
            else: regime = "HIGH_RISK_CHOP"
        elif bias_4h == "Bullish" or bias_1h == "Bullish":
            if struct_15m == "Healthy" and momentum_15m == "Healthy" and volatility_15m == "Healthy": regime = "CAUTIOUS_BULLISH"
            else: regime = "HIGH_RISK_CHOP"
        elif bias_4h == "Neutral" and bias_1h == "Neutral":
             if volatility_15m != "Unhealthy": regime = "RANGING_NEUTRAL"
             else: regime = "HIGH_RISK_CHOP"
        else: regime = "HIGH_RISK_CHOP"
        with market_context_lock:
            current_market_context = {
                "regime": regime, "details": {**htf_bias, **condition_15m},
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            last_market_context_check_time = time.time()
        logger.info(f"✅ [Market Context V28] New context: Regime='{regime}', Bias(4h:{bias_4h}, 1h:{bias_1h})")
    except Exception as e:
        logger.error(f"❌ [Market Context V28] Failed to determine market context: {e}", exc_info=True)
        with market_context_lock:
            current_market_context['regime'] = "BEARISH"
            current_market_context['details'] = {}

def analyze_and_select_filter_profile() -> None:
    global dynamic_filter_profile_cache
    with market_context_lock:
        market_regime = current_market_context.get("regime", "BEARISH")
    regime_to_profile_map = {
        "PRIME_BULLISH": "PRIME_BULLISH", "PULLBACK_BULLISH": "PULLBACK_BULLISH",
        "CAUTIOUS_BULLISH": "CAUTIOUS_BULLISH", "RANGING_NEUTRAL": "DISABLED",
        "HIGH_RISK_CHOP": "DISABLED", "BEARISH": "DISABLED", "INITIALIZING": "DISABLED"
    }
    profile_key = regime_to_profile_map.get(market_regime, "DISABLED")
    base_profile = FILTER_PROFILES[profile_key].copy()
    dynamic_filter_profile_cache = {
        "name": market_regime, "description": base_profile['description'],
        "strategy": base_profile.get("strategy", "DISABLED"), "filters": base_profile.get("filters", {}),
    }
    logger.info(f"🔬 [Filter Profile] Selected profile: '{profile_key}' based on market regime '{market_regime}'")

# ---------------------- دوال إدارة الصفقات والذاكرة المؤقتة (تبقى كما هي) ----------------------
def get_current_prices_from_redis(symbols: List[str]) -> Dict[str, Optional[float]]:
    if not redis_client or not symbols: return {s: None for s in symbols}
    try:
        prices = redis_client.hmget(REDIS_PRICES_HASH_NAME, symbols)
        return {symbol: float(price) if price else None for symbol, price in zip(symbols, prices)}
    except Exception as e:
        logger.error(f"❌ [Redis Price] Failed to get prices from Redis: {e}")
        return {s: None for s in symbols}

def load_open_signals_to_cache():
    global open_signals_cache
    if not check_db_connection() or not conn:
        logger.error("❌ [Cache Load] Cannot load open signals, no DB connection.")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status = 'open' OR status = 'updated';")
            open_signals = cur.fetchall()
            with signal_cache_lock:
                open_signals_cache = {signal['symbol']: dict(signal) for signal in open_signals}
            logger.info(f"✅ [Cache Load] Loaded {len(open_signals_cache)} open signals from DB into cache.")
    except Exception as e:
        logger.error(f"❌ [Cache Load] Error loading open signals into cache: {e}")
        if conn: conn.rollback()

def load_notifications_to_cache():
    if not check_db_connection() or not conn:
        logger.error("❌ [Cache Load] Cannot load notifications, no DB connection.")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 50;")
            recent_notifications = [dict(row) for row in cur.fetchall()]
        with notifications_lock:
            notifications_cache.clear()
            for notification in reversed(recent_notifications):
                notifications_cache.appendleft(notification)
        logger.info(f"✅ [Cache Load] Loaded {len(notifications_cache)} recent notifications into cache.")
    except Exception as e:
        logger.error(f"❌ [Cache Load] Error loading notifications into cache: {e}")
        if conn: conn.rollback()

def _close_trade_logic(signal_id: int, reason: str = "Automatic Closure") -> Tuple[bool, str]:
    if not check_db_connection() or not conn:
        return False, "Database connection is not available."
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE id = %s AND (status = 'open' OR status = 'updated');", (signal_id,))
            signal = cur.fetchone()
            if not signal: return False, f"Signal ID {signal_id} not found or is already closed."
            symbol = signal['symbol']
            is_real_trade = signal.get('is_real_trade', False)
            quantity_to_sell = signal.get('quantity')
            current_price = get_current_prices_from_redis([symbol]).get(symbol)
            if not current_price: return False, f"Could not retrieve current price for {symbol}."
            closing_price = current_price
            binance_order_details = {}
            if is_real_trade:
                if not client: return False, "Binance client not initialized."
                if not quantity_to_sell or quantity_to_sell <= 0: return False, "Invalid quantity for real trade."
                try:
                    logger.info(f"Attempting to place MARKET SELL order for {quantity_to_sell} of {symbol} for signal {signal_id}.")
                    order = client.create_order(symbol=symbol, side=Client.SIDE_SELL, type=Client.ORDER_TYPE_MARKET, quantity=quantity_to_sell)
                    logger.info(f"✅ Binance MARKET SELL order successful: {order}")
                    if order and 'fills' in order and len(order['fills']) > 0:
                        total_cost = sum(float(f['price']) * float(f['qty']) for f in order['fills'])
                        total_qty = sum(float(f['qty']) for f in order['fills'])
                        closing_price = total_cost / total_qty if total_qty > 0 else current_price
                    else:
                        closing_price = float(order.get('cummulativeQuoteQty', current_price * quantity_to_sell)) / float(order.get('executedQty', quantity_to_sell))
                    binance_order_details = order
                except BinanceAPIException as e:
                    error_msg = f"Binance API error on closing {symbol}: {e}"
                    log_and_notify('error', error_msg, "TRADE_ERROR")
                    return False, error_msg
                except Exception as e:
                    error_msg = f"Unexpected error during Binance order for {symbol}: {e}"
                    log_and_notify('error', error_msg, "TRADE_ERROR")
                    return False, error_msg
            profit_percentage = ((closing_price / signal['entry_price']) - 1) * 100
            cur.execute("""
                UPDATE signals SET status = 'closed', closing_price = %s, closed_at = %s, profit_percentage = %s,
                signal_details = signal_details || %s::jsonb WHERE id = %s;
            """, (closing_price, datetime.now(timezone.utc), profit_percentage, json.dumps({"closure_reason": reason, "binance_sell_order": binance_order_details}), signal_id))
        conn.commit()
        with signal_cache_lock:
            if symbol in open_signals_cache and open_signals_cache[symbol]['id'] == signal_id:
                del open_signals_cache[symbol]
        success_msg = f"✅ Successfully closed trade for {symbol} (ID: {signal_id}). Reason: {reason}. PnL: {profit_percentage:.2f}%"
        log_and_notify('info', success_msg, "TRADE_CLOSE")
        return True, success_msg
    except Exception as e:
        logger.error(f"❌ Critical error in _close_trade_logic for signal {signal_id}: {e}", exc_info=True)
        if conn: conn.rollback()
        return False, f"A server error occurred: {e}"

def trade_monitoring_loop():
    logger.info("🚦 [Monitor] Starting trade monitoring loop...")
    while True:
        try:
            time.sleep(10)
            with signal_cache_lock:
                if not open_signals_cache: continue
                cached_signals_copy = list(open_signals_cache.values())
            current_prices = get_current_prices_from_redis([s['symbol'] for s in cached_signals_copy])
            for signal in cached_signals_copy:
                symbol = signal['symbol']
                signal_id = signal['id']
                current_price = current_prices.get(symbol)
                if not current_price:
                    logger.warning(f"[Monitor] No current price for {symbol}, skipping check.")
                    continue
                if current_price <= signal['stop_loss']:
                    logger.info(f"🚨 [SL HIT] {symbol} at price {current_price} (SL: {signal['stop_loss']})")
                    _close_trade_logic(signal_id, reason=f"Stop-Loss Hit at {current_price}")
                    continue
                if current_price >= signal['target_price']:
                    logger.info(f"🎯 [TP HIT] {symbol} at price {current_price} (TP: {signal['target_price']})")
                    _close_trade_logic(signal_id, reason=f"Take-Profit Hit at {current_price}")
                    continue
        except Exception as e:
            logger.error(f"❌ [Monitor] Error in trade monitoring loop: {e}", exc_info=True)
            time.sleep(60)

# ---------------------- دوال البحث عن الصفقات والتحليل (تبقى كما هي) ----------------------

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates all necessary technical indicators for signal generation."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    plus_di = 100 * (plus_dm.ewm(alpha=1/ADX_PERIOD).mean() / df['atr'])
    minus_di = 100 * (abs(minus_dm.ewm(alpha=1/ADX_PERIOD).mean()) / df['atr'])
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df['adx'] = dx.ewm(alpha=1/ADX_PERIOD).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rel_vol'] = df['volume'] / df['volume'].rolling(window=REL_VOL_PERIOD).mean().replace(0, 1e-9)
    df['roc'] = df['close'].pct_change(periods=MOMENTUM_PERIOD) * 100
    ema_slope = df['close'].ewm(span=EMA_SLOPE_PERIOD, adjust=False).mean()
    df['slope'] = np.degrees(np.arctan(ema_slope.diff() / df['close']))
    df['volatility_pct'] = (df['high'] - df['low']) / df['low'] * 100
    return df

def adjust_to_step_size(quantity: float, step_size: str) -> float:
    """Adjusts a quantity to the step size defined by Binance."""
    decimal_step = Decimal(step_size)
    precision = abs(decimal_step.as_tuple().exponent)
    return float(Decimal(quantity).quantize(Decimal('1e-' + str(precision)), rounding=ROUND_DOWN))

def generate_and_execute_signal(symbol: str, df: pd.DataFrame, filters: Dict, strategy_name: str):
    """Generates and executes a trade signal after passing all filters."""
    current_price = df.iloc[-1]['close']
    atr = df.iloc[-1]['atr']
    if not atr or atr <= 0:
        log_rejection(symbol, "Invalid ATR for TP/SL", {"atr": atr})
        return
    stop_loss_price = current_price - (2 * atr)
    target_price = current_price + (2.5 * atr)
    risk = current_price - stop_loss_price
    reward = target_price - current_price
    if risk <= 0:
        log_rejection(symbol, "Invalid Position Size", {"risk": risk})
        return
    rrr = reward / risk
    if rrr < filters.get('min_rrr', 1.0):
        log_rejection(symbol, "RRR Filter", {"rrr": round(rrr, 2), "min_rrr": filters['min_rrr']})
        return
    with trading_status_lock:
        is_real = is_trading_enabled
    if is_real:
        usdt_balance = get_usdt_balance()
        if not usdt_balance or usdt_balance < 20:
            log_rejection(symbol, "Insufficient Balance", {"balance": usdt_balance})
            return
        risk_amount_usdt = (usdt_balance * RISK_PER_TRADE_PERCENT) / 100
        quantity = risk_amount_usdt / risk
    else:
        quantity = STATS_TRADE_SIZE_USDT / current_price
    symbol_info = exchange_info_map.get(symbol)
    if not symbol_info:
        logger.error(f"Could not find exchange info for {symbol}")
        return
    step_size = next((f['stepSize'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
    if step_size:
        quantity = adjust_to_step_size(quantity, step_size)
    min_notional = next((float(f['minNotional']) for f in symbol_info['filters'] if f['filterType'] == 'MIN_NOTIONAL'), 10.0)
    if quantity * current_price < min_notional:
        log_rejection(symbol, "Min Notional Filter", {"value": quantity * current_price, "min_notional": min_notional})
        return
    if quantity <= 0:
        log_rejection(symbol, "Lot Size Adjustment Failed", {"quantity": quantity})
        return
    order_id = f"paper_{int(time.time())}"
    binance_order = {}
    if is_real:
        try:
            logger.info(f"Placing REAL BUY order for {quantity} of {symbol}")
            order = client.create_order(symbol=symbol, side=Client.SIDE_BUY, type=Client.ORDER_TYPE_MARKET, quantity=quantity)
            order_id = order['orderId']
            binance_order = order
            log_and_notify('info', f"✅ REAL TRADE EXECUTED: BUY {quantity} {symbol} @ market", "REAL_TRADE")
        except Exception as e:
            log_and_notify('error', f"❌ REAL TRADE FAILED for {symbol}: {e}", "TRADE_ERROR")
            return
    else:
        log_and_notify('info', f"📝 PAPER TRADE: BUY {quantity} {symbol} @ {current_price}", "PAPER_TRADE")
    if not check_db_connection() or not conn:
        log_and_notify('error', f"DB connection lost. Failed to save signal for {symbol}", "DB_ERROR")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals 
                (symbol, entry_price, target_price, stop_loss, status, strategy_name, signal_details, is_real_trade, quantity, order_id, current_peak_price)
                VALUES (%s, %s, %s, %s, 'open', %s, %s, %s, %s, %s, %s) RETURNING *;
            """, (
                symbol, current_price, target_price, stop_loss_price, strategy_name,
                json.dumps({"filters_passed": filters, "binance_buy_order": binance_order}),
                is_real, quantity, order_id, current_price
            ))
            new_signal = dict(cur.fetchone())
        conn.commit()
        with signal_cache_lock:
            open_signals_cache[symbol] = new_signal
        logger.info(f"✅ Signal for {symbol} saved to DB and cache. ID: {new_signal['id']}")
    except Exception as e:
        logger.error(f"❌ DB Error saving signal for {symbol}: {e}")
        if conn: conn.rollback()

def process_symbol(symbol: str, filter_profile: Dict):
    """Processes a single symbol against the current filter profile."""
    filters = filter_profile.get("filters", {})
    if not filters: return
    with signal_cache_lock:
        if symbol in open_signals_cache:
            # log_rejection(symbol, "Signal Already Open") # This can be noisy, maybe disable
            return
    df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, days=10)
    if df is None or df.empty or len(df) < 100:
        log_rejection(symbol, "Data Fetch Failed")
        return
    df = calculate_technical_indicators(df)
    last = df.iloc[-1].to_dict()
    if last.get('volatility_pct', 0) < filters.get('min_volatility_pct', 0):
        log_rejection(symbol, "Low Volatility", {"value": round(last.get('volatility_pct', 0), 2), "required": filters['min_volatility_pct']})
        return
    if last.get('adx', 0) < filters.get('adx', 0):
        log_rejection(symbol, "ADX Filter", {"value": round(last.get('adx', 0), 2), "required": filters['adx']})
        return
    rsi_min, rsi_max = filters.get('rsi_range', (0, 100))
    if not (rsi_min <= last.get('rsi', 50) <= rsi_max):
        log_rejection(symbol, "RSI Filter", {"value": round(last.get('rsi', 50), 2), "range": (rsi_min, rsi_max)})
        return
    if last.get('roc', 0) < filters.get('roc', -100):
        log_rejection(symbol, "ROC Filter", {"value": round(last.get('roc', 0), 2), "required": filters['roc']})
        return
    if last.get('slope', 0) < filters.get('slope', -100):
        log_rejection(symbol, "Slope Filter", {"value": round(last.get('slope', 0), 4), "required": filters['slope']})
        return
    logger.info(f"✅ [PASS] {symbol} passed all filters. Generating signal...")
    generate_and_execute_signal(symbol, df, filters, filter_profile.get("strategy", "UNKNOWN"))


# ---------------------- حلقة العمل الرئيسية (تبقى كما هي) ----------------------
def main_loop():
    """Main loop now actively scans for signals."""
    logger.info("[Main Loop] Waiting for initialization...")
    time.sleep(15)
    if not validated_symbols_to_scan:
        log_and_notify("critical", "No validated symbols to scan. Bot will not start.", "SYSTEM")
        return
    log_and_notify("info", f"✅ Starting main scan loop for {len(validated_symbols_to_scan)} symbols.", "SYSTEM")
    
    while True:
        try:
            determine_market_context_for_scalping()
            analyze_and_select_filter_profile()
            filter_profile = dynamic_filter_profile_cache
            active_strategy_type = filter_profile.get("strategy")
            if not active_strategy_type or active_strategy_type == "DISABLED":
                logger.warning(f"🛑 Trading is disabled by market regime: '{filter_profile.get('name')}'. Skipping scan cycle.")
                time.sleep(60)
                continue
            with signal_cache_lock:
                open_trades_count = len(open_signals_cache)
            if open_trades_count >= MAX_OPEN_TRADES:
                logger.info(f"Max open trades ({MAX_OPEN_TRADES}) reached. Pausing new signal search.")
                time.sleep(30)
                continue
            logger.info(f"🔍 Starting scan cycle with profile '{filter_profile.get('name')}' for {len(validated_symbols_to_scan)} symbols...")
            random.shuffle(validated_symbols_to_scan)
            for i in range(0, len(validated_symbols_to_scan), SYMBOL_PROCESSING_BATCH_SIZE):
                batch = validated_symbols_to_scan[i:i+SYMBOL_PROCESSING_BATCH_SIZE]
                threads = []
                for symbol in batch:
                    with signal_cache_lock:
                        if len(open_signals_cache) >= MAX_OPEN_TRADES:
                            log_rejection(symbol, "Max Open Trades Reached")
                            break
                    thread = Thread(target=process_symbol, args=(symbol, filter_profile))
                    threads.append(thread)
                    thread.start()
                for thread in threads:
                    thread.join()
                if len(open_signals_cache) >= MAX_OPEN_TRADES:
                    break
                time.sleep(2)
            logger.info("✅ Scan cycle finished.")
            time.sleep(30) # Reduced wait time between full cycles
        except (KeyboardInterrupt, SystemExit):
            log_and_notify("info", "Bot is shutting down by user request.", "SYSTEM"); break
        except Exception as main_err:
            log_and_notify("error", f"Critical error in main loop: {main_err}", "SYSTEM"); time.sleep(120)

# ---------------------- واجهة برمجة تطبيقات Flask (مُعدَّلة) ----------------------
app = Flask(__name__)
CORS(app)

def check_api_status() -> bool:
    if not client: return False
    try: client.ping(); return True
    except Exception: return False

def get_usdt_balance() -> Optional[float]:
    """
    [FIXED] This function now reads from a cache updated by a background thread.
    It no longer makes direct API calls, making it faster and more reliable.
    """
    with usdt_balance_lock:
        return usdt_balance_cache

@app.route('/')
def home(): return render_template_string(get_dashboard_html())

@app.route('/api/market_status')
def get_market_status():
    with market_context_lock: context_copy = dict(current_market_context)
    return jsonify({
        "market_context": context_copy, "filter_profile": dynamic_filter_profile_cache,
        "db_ok": check_db_connection(), "api_ok": check_api_status(),
        "usdt_balance": get_usdt_balance(),
    })

@app.route('/api/stats')
def get_stats():
    if not check_db_connection() or not conn:
        return jsonify({"error": "Database connection failed"}), 500
    with signal_cache_lock:
        open_trades_count = len(open_signals_cache)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit_percentage > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN profit_percentage > 0 THEN profit_percentage ELSE 0 END) as gross_profit,
                    SUM(CASE WHEN profit_percentage < 0 THEN profit_percentage ELSE 0 END) as gross_loss
                FROM signals WHERE status = 'closed' AND is_real_trade = TRUE;
            """)
            stats = cur.fetchone()
            cur.execute("""
                SELECT SUM( (closing_price / entry_price - 1) * quantity * entry_price ) as net_profit_usdt
                FROM signals WHERE status = 'closed' AND is_real_trade = TRUE AND entry_price > 0;
            """)
            profit_sum = cur.fetchone()
        if not stats or stats['total_trades'] == 0:
            return jsonify({"win_rate": 0, "profit_factor": 0, "total_closed_trades": 0, "open_trades_count": open_trades_count, "net_profit_usdt": 0})
        win_rate = (stats['winning_trades'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0
        gross_profit = stats['gross_profit'] or 0
        gross_loss = abs(stats['gross_loss'] or 0)
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 'Infinity'
        net_profit_usdt = profit_sum['net_profit_usdt'] if profit_sum and profit_sum['net_profit_usdt'] is not None else 0
        return jsonify({"win_rate": win_rate, "profit_factor": profit_factor, "total_closed_trades": stats['total_trades'], "open_trades_count": open_trades_count, "net_profit_usdt": net_profit_usdt})
    except Exception as e:
        logger.error(f"❌ [API Stats] Error fetching stats: {e}")
        if conn: conn.rollback()
        return jsonify({"error": "Failed to fetch stats from database"}), 500

@app.route('/api/profit_curve')
def get_profit_curve():
    if not check_db_connection() or not conn:
        return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT closed_at as timestamp, SUM(profit_percentage) OVER (ORDER BY closed_at) as cumulative_profit
                FROM signals WHERE status = 'closed' AND is_real_trade = TRUE AND closed_at IS NOT NULL ORDER BY closed_at;
            """)
            profit_data = [dict(row) for row in cur.fetchall()]
        if profit_data:
            start_time = profit_data[0]['timestamp'] - timedelta(seconds=1)
            profit_data.insert(0, {'timestamp': start_time, 'cumulative_profit': 0})
        else:
            profit_data.insert(0, {'timestamp': datetime.now(timezone.utc), 'cumulative_profit': 0})
        for item in profit_data:
            if isinstance(item['timestamp'], datetime):
                item['timestamp'] = item['timestamp'].isoformat()
        return jsonify(profit_data)
    except Exception as e:
        logger.error(f"❌ [API Profit Curve] Error fetching profit curve data: {e}")
        if conn: conn.rollback()
        return jsonify({"error": "Failed to fetch profit curve data"}), 500

@app.route('/api/signals')
def get_signals():
    if not check_db_connection() or not conn:
        return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals ORDER BY created_at DESC, id DESC LIMIT 50;")
            signals = [dict(row) for row in cur.fetchall()]
        open_symbols = [s['symbol'] for s in signals if s['status'] in ['open', 'updated']]
        current_prices = get_current_prices_from_redis(open_symbols)
        for signal in signals:
            if signal['status'] in ['open', 'updated']:
                current_price = current_prices.get(signal['symbol'])
                signal['current_price'] = current_price
                if current_price and signal.get('entry_price'):
                    pnl = ((current_price / signal['entry_price']) - 1) * 100
                    signal['pnl_pct'] = pnl
                else:
                    signal['pnl_pct'] = 0
            else:
                signal['current_price'] = signal.get('closing_price')
                signal['pnl_pct'] = signal.get('profit_percentage')
        return jsonify(signals)
    except Exception as e:
        logger.error(f"❌ [API Signals] Error fetching signals for dashboard: {e}")
        if conn: conn.rollback()
        return jsonify({"error": "Failed to fetch signals from database"}), 500

@app.route('/api/close/<int:signal_id>', methods=['POST'])
def manual_close_signal_api(signal_id):
    logger.warning(f"Manual close requested from dashboard for signal ID {signal_id}.")
    success, message = _close_trade_logic(signal_id, reason="Manual Closure from Dashboard")
    if success:
        return jsonify({"message": message})
    else:
        return jsonify({"error": message}), 500

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

@app.route('/api/notifications')
def get_notifications():
    with notifications_lock: return jsonify(list(notifications_cache))

@app.route('/api/rejection_logs')
def get_rejection_logs():
    with rejection_logs_lock: return jsonify(list(rejection_logs_cache))

def run_flask():
    # Use the PORT environment variable provided by many hosting platforms, default to 5000
    port = int(os.environ.get('PORT', 5000))
    host = "0.0.0.0"
    logger.info(f"✅ Preparing to start dashboard on {host}:{port}")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        logger.warning("⚠️ 'waitress' not found. Using Flask's development server.")
        app.run(host=host, port=port)

# ---------------------- نقطة انطلاق البرنامج (مُعدَّلة) ----------------------
def handle_price_update(msg):
    """
    [FIXED] This function is now extremely lightweight.
    It only adds the received price to an in-memory dictionary.
    """
    if 'data' not in msg or not msg.get('stream'):
        return
    try:
        symbol = msg['stream'].split('@')[0].upper()
        price = msg['data']['c']
        # Update the batch dictionary. This is a very fast, in-memory operation.
        with price_updates_lock:
            price_updates_batch[symbol] = price
    except (KeyError, TypeError) as e:
        logger.warning(f"⚠️ [WebSocket] Could not process price update: {e} | MSG: {msg}")

def redis_batch_writer():
    """
    [NEW] This function runs in a dedicated thread.
    It periodically writes the collected price updates to Redis in a single batch.
    """
    logger.info("💾 [Redis Writer] Starting Redis batch writer thread...")
    while True:
        # Define how often to write to Redis. 1 second is a good balance.
        time.sleep(1)
        
        with price_updates_lock:
            if not price_updates_batch:
                continue
            # Create a local copy of the batch and clear the global one immediately.
            # This minimizes the time the lock is held.
            local_batch = price_updates_batch.copy()
            price_updates_batch.clear()

        if redis_client and local_batch:
            try:
                # Use Redis's HMSET (via hset with mapping) for efficient batch updates.
                redis_client.hset(REDIS_PRICES_HASH_NAME, mapping=local_batch)
                # Optional: Uncomment for debugging to see batch sizes.
                # logger.debug(f"[Redis Writer] Wrote {len(local_batch)} price updates to Redis.")
            except redis.exceptions.RedisError as e:
                logger.error(f"❌ [Redis Writer] Failed to write batch to Redis: {e}")

def balance_update_loop():
    """
    [NEW] This function runs in a dedicated thread to periodically update the USDT balance cache.
    This prevents frequent API calls from the dashboard and makes the balance display more resilient.
    """
    global usdt_balance_cache
    logger.info("💰 [Balance] Starting balance update loop...")
    while True:
        if not client:
            time.sleep(10) # Wait for client to be initialized
            continue
        
        try:
            balance_info = client.get_asset_balance(asset='USDT')
            with usdt_balance_lock:
                usdt_balance_cache = float(balance_info['free'])
            # Optional: log successful update for debugging
            # logger.info(f"[Balance] Successfully updated USDT balance cache: {usdt_balance_cache}")
        except BinanceAPIException as e:
            logger.error(f"❌ [Balance Loop] Binance API error while fetching USDT balance: {e}")
        except Exception as e:
            logger.error(f"❌ [Balance Loop] Unexpected error while fetching USDT balance: {e}", exc_info=False)
        
        # Update balance every 60 seconds. It's not a critical real-time value.
        time.sleep(60)

def run_websocket_manager():
    if not client or not validated_symbols_to_scan:
        logger.error("❌ [WebSocket] Cannot start: Client or symbols not initialized.")
        return
    logger.info("📡 [WebSocket] Starting WebSocket Manager...")
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    streams = [f"{s.lower()}@miniTicker" for s in validated_symbols_to_scan]
    
    # The callback 'handle_price_update' is now very fast, preventing queue overflow.
    twm.start_multiplex_socket(callback=handle_price_update, streams=streams)
    
    logger.info(f"✅ [WebSocket] Subscribed to {len(streams)} price streams.")
    twm.join()

def initialize_bot_services():
    global client, validated_symbols_to_scan
    logger.info("🤖 [Bot Services] Starting background initialization...")
    try:
        # [FIXED] Correctly initialize Binance client with a custom session.
        client = Client(API_KEY, API_SECRET)
        # Create a custom session with a larger connection pool
        custom_session = Session()
        adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100)
        # Mount the adapter to handle all https requests
        custom_session.mount('https://', adapter)
        # Replace the default session in the client with our custom one
        client.session = custom_session

        init_db()
        init_redis()
        get_exchange_info_map()
        
        load_open_signals_to_cache()
        load_notifications_to_cache()
        
        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan:
            log_and_notify("critical", "No validated symbols to scan. Bot will not start.", "SYSTEM")
            return

        # [NEW] Start the dedicated Redis writer thread.
        Thread(target=redis_batch_writer, daemon=True).start()
        # [NEW] Start the dedicated balance update thread.
        Thread(target=balance_update_loop, daemon=True).start()
        
        Thread(target=run_websocket_manager, daemon=True).start()
        Thread(target=trade_monitoring_loop, daemon=True).start()
        Thread(target=main_loop, daemon=True).start()
        logger.info("✅ [Bot Services] All background services started successfully.")
    except Exception as e:
        log_and_notify("critical", f"A critical error occurred during initialization: {e}", "SYSTEM")
        # Don't exit here, let Flask run so the error is visible on the dashboard if possible
        # exit(1)

if __name__ == "__main__":
    logger.info("🚀 LAUNCHING TRADING BOT & DASHBOARD (V28 - Market Context Model) 🚀")
    initialization_thread = Thread(target=initialize_bot_services, daemon=True)
    initialization_thread.start()
    run_flask()
    logger.info("👋 [Shutdown] Application has been shut down."); os._exit(0)
