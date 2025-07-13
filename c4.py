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
from typing import List, Dict, Optional, Any, Set
from sklearn.preprocessing import StandardScaler
from collections import deque
import warnings

# --- تجاهل التحذيرات غير الهامة ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ---------------------- [معدل] إعداد نظام التسجيل (Logging) - V3 ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_smc_v3.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotSMC_V3')

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

# ---------------------- [معدل] إعداد الثوابت والمتغيرات العامة - V3 ----------------------
# --- إعدادات التداول الحقيقي ---
is_trading_enabled: bool = False
trading_status_lock = Lock()
RISK_PER_TRADE_PERCENT: float = 1.0

# --- ثوابت عامة لاستخدام النموذج الجديد ---
BASE_ML_MODEL_NAME: str = 'SMC_Scalping_V3_With_SR_Sentiment_OB'
MODEL_FOLDER: str = 'V10_SMC_SR_OB' # [معدل] مجلد جديد للنماذج المتقدمة
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 30
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices_smc_v3"
DIRECT_API_CHECK_INTERVAL: int = 10
TRADING_FEE_PERCENT: float = 0.1
STATS_TRADE_SIZE_USDT: float = 10.0

# --- مؤشرات فنية ---
ADX_PERIOD: int = 14; RSI_PERIOD: int = 14; ATR_PERIOD: int = 14
EMA_FAST_PERIOD: int = 50; EMA_SLOW_PERIOD: int = 200
REL_VOL_PERIOD: int = 30; MOMENTUM_PERIOD: int = 12; EMA_SLOPE_PERIOD: int = 5

# --- إدارة الصفقات ---
MAX_OPEN_TRADES: int = 4
BUY_CONFIDENCE_THRESHOLD = 0.75
MIN_CONFIDENCE_INCREASE_FOR_UPDATE = 0.05

# --- إعدادات الهدف ووقف الخسارة ---
ATR_FALLBACK_SL_MULTIPLIER: float = 1.5
ATR_FALLBACK_TP_MULTIPLIER: float = 2.0

# --- إعدادات وقف الخسارة المتحرك ---
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_ACTIVATION_PROFIT_PERCENT: float = 1.0
TRAILING_DISTANCE_PERCENT: float = 0.8
LAST_PEAK_UPDATE_TIME: Dict[int, float] = {}
PEAK_UPDATE_COOLDOWN: int = 60

# --- إعدادات الفلاتر ---
USE_BTC_TREND_FILTER: bool = True; BTC_SYMBOL: str = 'BTCUSDT'
USE_SPEED_FILTER: bool = True; USE_RRR_FILTER: bool = True; MIN_RISK_REWARD_RATIO: float = 1.5
USE_BTC_CORRELATION_FILTER: bool = True; MIN_BTC_CORRELATION: float = 0.2
USE_MIN_VOLATILITY_FILTER: bool = True; MIN_VOLATILITY_PERCENT: float = 0.5
USE_MOMENTUM_FILTER: bool = True

# --- المتغيرات العامة وقفل العمليات ---
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
current_market_state: Dict[str, Any] = {"overall_regime": "INITIALIZING", "details": {}, "last_updated": None}
market_state_lock = Lock()
fg_data_cache: Optional[pd.DataFrame] = None # [جديد] ذاكرة تخزين لمؤشر الخوف والطمع
last_fg_fetch_time: float = 0 # [جديد] وقت آخر جلب لمؤشر الخوف والطمع

# ---------------------- دالة HTML للوحة التحكم (مع تحديث العنوان) ----------------------
def get_dashboard_html():
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم التداول V3 - نموذج متقدم</title>
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
    </style>
</head>
<body class="p-4 md:p-6">
    <div class="container mx-auto max-w-screen-2xl">
        <header class="mb-6 flex flex-wrap justify-between items-center gap-4">
            <h1 class="text-2xl md:text-3xl font-extrabold text-white">
                <span class="text-accent-blue">لوحة التحكم</span>
                <span class="text-text-secondary font-medium">V3 - Advanced Model</span>
            </h1>
            <div id="connection-status" class="flex items-center gap-3 text-sm">
                <div class="flex items-center gap-2"><div id="db-status-light" class="w-2.5 h-2.5 rounded-full bg-gray-600 animate-pulse"></div><span class="text-text-secondary">DB</span></div>
                <div class="flex items-center gap-2"><div id="api-status-light" class="w-2.5 h-2.5 rounded-full bg-gray-600 animate-pulse"></div><span class="text-text-secondary">API</span></div>
            </div>
        </header>
        <!-- بقية محتوى HTML للوحة التحكم يبقى كما هو -->
        <section class="mb-6 grid grid-cols-1 md:grid-cols-3 gap-5">
            <div class="card p-4 md:col-span-2">
                 <h3 class="font-bold mb-3 text-lg text-text-secondary">حالة السوق (BTC)</h3>
                 <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
                     <div><h4 class="text-sm font-medium text-text-secondary">الاتجاه العام</h4><div id="overall-regime" class="text-2xl font-bold skeleton h-8 w-3/4 mx-auto mt-1"></div></div>
                     <div><h4 class="text-sm font-medium text-text-secondary">15 دقيقة</h4><div id="tf-15m-status" class="text-xl font-bold skeleton h-7 w-2/3 mx-auto mt-1"></div></div>
                     <div><h4 class="text-sm font-medium text-text-secondary">ساعة</h4><div id="tf-1h-status" class="text-xl font-bold skeleton h-7 w-2/3 mx-auto mt-1"></div></div>
                     <div><h4 class="text-sm font-medium text-text-secondary">4 ساعات</h4><div id="tf-4h-status" class="text-xl font-bold skeleton h-7 w-2/3 mx-auto mt-1"></div></div>
                 </div>
            </div>
            <div class="card p-4 flex flex-col justify-center items-center">
                <h3 class="font-bold text-lg text-text-secondary mb-2">التحكم بالتداول الحقيقي</h3>
                <div class="flex items-center space-x-3 space-x-reverse">
                    <span id="trading-status-text" class="font-bold text-lg text-accent-red">غير مُفعّل</span>
                    <label for="trading-toggle" class="flex items-center cursor-pointer">
                        <div class="relative">
                            <input type="checkbox" id="trading-toggle" class="sr-only" onchange="toggleTrading()">
                            <div class="toggle-bg block bg-accent-red w-12 h-7 rounded-full"></div>
                        </div>
                    </label>
                </div>
                 <div class="mt-2 text-xs text-text-secondary">رصيد USDT: <span id="usdt-balance" class="font-mono skeleton w-20 inline-block"></span></div>
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
            </nav>
        </div>
        <main>
            <div id="signals-tab" class="tab-content"><div class="overflow-x-auto card p-0"><table class="min-w-full text-sm text-right"><thead class="border-b border-border-color bg-black/20"><tr><th class="p-4 font-semibold text-text-secondary">العملة</th><th class="p-4 font-semibold text-text-secondary">الحالة</th><th class="p-4 font-semibold text-text-secondary">الكمية</th><th class="p-4 font-semibold text-text-secondary">الربح/الخسارة</th><th class="p-4 font-semibold text-text-secondary w-[25%]">التقدم</th><th class="p-4 font-semibold text-text-secondary">الدخول/الحالي</th><th class="p-4 font-semibold text-text-secondary">إجراء</th></tr></thead><tbody id="signals-table"></tbody></table></div></div>
            <div id="stats-tab" class="tab-content hidden"><div id="stats-container" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4"></div></div>
            <div id="notifications-tab" class="tab-content hidden"><div id="notifications-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-2"></div></div>
            <div id="rejections-tab" class="tab-content hidden"><div id="rejections-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-2"></div></div>
        </main>
    </div>
    <script>
        // كود JavaScript للوحة التحكم يبقى كما هو
        let profitChartInstance;
        const REGIME_STYLES = { "STRONG UPTREND": { text: "صاعد قوي", color: "text-accent-green" }, "UPTREND": { text: "صاعد", color: "text-green-400" }, "RANGING": { text: "عرضي", color: "text-accent-yellow" }, "DOWNTREND": { text: "هابط", color: "text-red-400" }, "STRONG DOWNTREND": { text: "هابط قوي", color: "text-accent-red" }, "UNCERTAIN": { text: "غير واضح", color: "text-text-secondary" }, "INITIALIZING": { text: "تهيئة...", color: "text-accent-blue" } };
        const TF_STATUS_STYLES = { "Uptrend": { text: "صاعد", icon: "▲", color: "text-accent-green" }, "Downtrend": { text: "هابط", icon: "▼", color: "text-accent-red" }, "Ranging": { text: "عرضي", icon: "↔", color: "text-accent-yellow" }, "Uncertain": { text: "غير واضح", icon: "?", color: "text-text-secondary" } };
        function formatNumber(num, digits = 2) { if (num === null || num === undefined || isNaN(num)) return 'N/A'; return num.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits }); }
        function showTab(tabName, element) { document.querySelectorAll('.tab-content').forEach(tab => tab.classList.add('hidden')); document.getElementById(`${tabName}-tab`).classList.remove('hidden'); document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active', 'text-white')); element.classList.add('active', 'text-white'); }
        async function apiFetch(url, options = {}) { try { const response = await fetch(url, options); if (!response.ok) { console.error(`API Error ${response.status}`); return { error: `HTTP Error ${response.status}` }; } return await response.json(); } catch (error) { console.error(`Fetch error for ${url}:`, error); return { error: "Network or fetch error" }; } }
        function getFngColor(value) { if (value < 25) return 'var(--accent-red)'; if (value < 45) return '#F97316'; if (value < 55) return 'var(--accent-yellow)'; if (value < 75) return '#84CC16'; return 'var(--accent-green)'; }
        function renderFearGreedGauge(value, classification) { const container = document.getElementById('fear-greed-gauge'); const valueEl = document.getElementById('fear-greed-value'); const textEl = document.getElementById('fear-greed-text'); [valueEl, textEl].forEach(el => el.classList.remove('skeleton', 'h-10', 'w-1/2', 'h-6', 'w-3/4')); if (value === -1) { container.innerHTML = `<div class="text-center text-text-secondary">خطأ</div>`; valueEl.textContent = 'N/A'; textEl.textContent = 'فشل التحميل'; return; } valueEl.textContent = value; textEl.textContent = classification; const angle = -90 + (value / 100) * 180; const color = getFngColor(value); valueEl.style.color = color; container.innerHTML = `<svg viewBox="0 0 100 57" class="w-full h-full"><defs><linearGradient id="g"><stop offset="0%" stop-color="#F85149"/><stop offset="50%" stop-color="#D29922"/><stop offset="100%" stop-color="#3FB950"/></linearGradient></defs><path d="M10 50 A 40 40 0 0 1 90 50" stroke="url(#g)" stroke-width="10" fill="none" stroke-linecap="round"/><g transform="rotate(${angle} 50 50)"><path d="M50 45 L 47 15 Q 50 10 53 15 L 50 45" fill="${color}" id="needle"/></g><circle cx="50" cy="50" r="4" fill="${color}"/></svg>`; }
        function updateMarketStatus() { apiFetch('/api/market_status').then(data => { if (!data || data.error) return; document.getElementById('db-status-light').className = `w-2.5 h-2.5 rounded-full ${data.db_ok ? 'bg-green-500' : 'bg-red-500'}`; document.getElementById('api-status-light').className = `w-2.5 h-2.5 rounded-full ${data.api_ok ? 'bg-green-500' : 'bg-red-500'}`; const state = data.market_state; const overallRegime = state.overall_regime || "UNCERTAIN"; const regimeStyle = REGIME_STYLES[overallRegime.toUpperCase()] || REGIME_STYLES["UNCERTAIN"]; const overallDiv = document.getElementById('overall-regime'); overallDiv.textContent = regimeStyle.text; overallDiv.className = `text-2xl font-bold ${regimeStyle.color}`; ['15m', '1h', '4h'].forEach(tf => { const tfData = state.details[tf]; const statusDiv = document.getElementById(`tf-${tf}-status`); statusDiv.classList.remove('skeleton', 'h-7', 'w-2/3', 'mx-auto', 'mt-1'); if (tfData) { const style = TF_STATUS_STYLES[tfData.trend] || TF_STATUS_STYLES["Uncertain"]; statusDiv.innerHTML = `<span class="${style.color}">${style.icon} ${style.text}</span>`; } else { statusDiv.textContent = 'N/A'; } }); renderFearGreedGauge(data.fear_and_greed.value, data.fear_and_greed.classification); const usdtBalanceEl = document.getElementById('usdt-balance'); usdtBalanceEl.textContent = data.usdt_balance ? `$${formatNumber(data.usdt_balance, 2)}` : 'N/A'; usdtBalanceEl.classList.remove('skeleton', 'w-20'); }); }
        function updateTradingStatus() { apiFetch('/api/trading/status').then(data => { if (!data || data.error) return; const toggle = document.getElementById('trading-toggle'); const text = document.getElementById('trading-status-text'); const bg = toggle.nextElementSibling; toggle.checked = data.is_enabled; if (data.is_enabled) { text.textContent = 'مُفعّل'; text.className = 'font-bold text-lg text-accent-green'; bg.classList.remove('bg-accent-red'); bg.classList.add('bg-accent-green'); } else { text.textContent = 'غير مُفعّل'; text.className = 'font-bold text-lg text-accent-red'; bg.classList.remove('bg-accent-green'); bg.classList.add('bg-accent-red'); } }); }
        function toggleTrading() { const toggle = document.getElementById('trading-toggle'); const confirmationMessage = toggle.checked ? "هل أنت متأكد من تفعيل التداول بأموال حقيقية؟ هذا الإجراء يحمل مخاطر." : "هل أنت متأكد من إيقاف التداول الحقيقي؟ لن يتم فتح أو إغلاق أي صفقات جديدة."; if (confirm(confirmationMessage)) { apiFetch('/api/trading/toggle', { method: 'POST' }).then(data => { if (data.message) { alert(data.message); updateTradingStatus(); } else if (data.error) { alert(`خطأ: ${data.error}`); updateTradingStatus(); } }); } else { toggle.checked = !toggle.checked; } }
        function updateStats() { apiFetch('/api/stats').then(data => { if (!data || data.error) { console.error("Failed to fetch stats:", data ? data.error : "No data"); return; } const profitFactorDisplay = data.profit_factor === 'Infinity' ? '∞' : formatNumber(data.profit_factor); document.getElementById('open-trades-value').textContent = formatNumber(data.open_trades_count, 0); document.getElementById('open-trades-value').classList.remove('skeleton', 'h-12', 'w-1/2'); const netProfitEl = document.getElementById('net-profit-usdt'); netProfitEl.textContent = `$${formatNumber(data.net_profit_usdt)}`; netProfitEl.className = `text-2xl font-bold ${data.net_profit_usdt >= 0 ? 'text-accent-green' : 'text-accent-red'}`; netProfitEl.classList.remove('skeleton', 'h-8', 'w-1/3'); const statsContainer = document.getElementById('stats-container'); statsContainer.innerHTML = ` <div class="card text-center p-4 flex flex-col justify-center"> <div class="text-sm text-text-secondary mb-1">نسبة النجاح</div> <div class="text-3xl font-bold text-accent-blue">${formatNumber(data.win_rate)}%</div> </div> <div class="card text-center p-4 flex flex-col justify-center"> <div class="text-sm text-text-secondary mb-1">عامل الربح</div> <div class="text-3xl font-bold text-accent-yellow">${profitFactorDisplay}</div> </div> <div class="card text-center p-4 flex flex-col justify-center"> <div class="text-sm text-text-secondary mb-1">إجمالي الصفقات المغلقة</div> <div class="text-3xl font-bold text-text-primary">${formatNumber(data.total_closed_trades, 0)}</div> </div> <div class="card text-center p-4 flex flex-col justify-center"> <div class="text-sm text-text-secondary mb-1">متوسط الربح %</div> <div class="text-3xl font-bold text-accent-green">${formatNumber(data.average_win_pct)}%</div> </div> <div class="card text-center p-4 flex flex-col justify-center"> <div class="text-sm text-text-secondary mb-1">متوسط الخسارة %</div> <div class="text-3xl font-bold text-accent-red">${formatNumber(data.average_loss_pct)}%</div> </div> `; }); }
        function updateProfitChart() { const loader = document.getElementById('profit-chart-loader'); const canvas = document.getElementById('profitChart'); const chartCard = document.getElementById('profit-chart-card'); apiFetch('/api/profit_curve').then(data => { loader.style.display = 'none'; const existingMsg = chartCard.querySelector('.no-data-msg'); if(existingMsg) existingMsg.remove(); if (!data || data.error || data.length <= 1) { canvas.style.display = 'none'; if (!existingMsg) { chartCard.insertAdjacentHTML('beforeend', '<p class="no-data-msg text-center text-text-secondary mt-8">لا توجد صفقات كافية لعرض المنحنى.</p>'); } return; } canvas.style.display = 'block'; const ctx = canvas.getContext('2d'); const chartData = data.map(d => ({ x: luxon.DateTime.fromISO(d.timestamp).valueOf(), y: d.cumulative_profit })); const lastProfit = chartData[chartData.length - 1].y; const gradient = ctx.createLinearGradient(0, 0, 0, ctx.canvas.height); if (lastProfit >= 0) { gradient.addColorStop(0, 'rgba(63, 185, 80, 0.4)'); gradient.addColorStop(1, 'rgba(63, 185, 80, 0)'); } else { gradient.addColorStop(0, 'rgba(248, 81, 73, 0.4)'); gradient.addColorStop(1, 'rgba(248, 81, 73, 0)'); } const config = { type: 'line', data: { datasets: [{ label: 'الربح التراكمي %', data: chartData, borderColor: lastProfit >= 0 ? 'var(--accent-green)' : 'var(--accent-red)', backgroundColor: gradient, fill: true, tension: 0.4, pointRadius: 0, pointHoverRadius: 6, pointBackgroundColor: lastProfit >= 0 ? 'var(--accent-green)' : 'var(--accent-red)', }] }, options: { responsive: true, maintainAspectRatio: false, scales: { x: { type: 'time', time: { unit: 'day', tooltipFormat: 'MMM dd, yyyy HH:mm' }, grid: { display: false }, ticks: { color: 'var(--text-secondary)', maxRotation: 0, autoSkip: true, maxTicksLimit: 7 } }, y: { position: 'right', beginAtZero: true, grid: { color: 'var(--border-color)', drawBorder: false }, ticks: { color: 'var(--text-secondary)', callback: v => formatNumber(v) + '%' } } }, plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false, backgroundColor: '#0D1117', titleFont: { weight: 'bold', family: 'Tajawal' }, bodyFont: { family: 'Tajawal' }, displayColors: false, callbacks: { label: (ctx) => `الربح التراكمي: ${formatNumber(ctx.raw.y)}%` } } }, interaction: { mode: 'index', intersect: false } } }; if (profitChartInstance) { profitChartInstance.data.datasets[0].data = chartData; profitChartInstance.data.datasets[0].borderColor = lastProfit >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'; profitChartInstance.data.datasets[0].backgroundColor = gradient; profitChartInstance.update('none'); } else { profitChartInstance = new Chart(ctx, config); } }); }
        function renderProgressBar(signal) { const { entry_price, stop_loss, target_price, current_price } = signal; if ([entry_price, stop_loss, target_price, current_price].some(v => v === null || v === undefined)) return '<span class="text-xs text-text-secondary">لا تتوفر بيانات</span>'; const [entry, sl, tp, current] = [entry_price, stop_loss, target_price, current_price].map(parseFloat); const totalDist = tp - sl; if (totalDist <= 0) return '<span class="text-xs text-text-secondary">بيانات غير صالحة</span>'; const progressPct = Math.max(0, Math.min(100, ((current - sl) / totalDist) * 100)); return `<div class="flex flex-col w-full"><div class="progress-bar-container"><div class="progress-bar ${current >= entry ? 'bg-accent-green' : 'bg-accent-red'}" style="width: ${progressPct}%"></div></div><div class="progress-labels"><span title="وقف الخسارة">${sl.toPrecision(4)}</span><span title="الهدف">${tp.toPrecision(4)}</span></div></div>`; }
        function updateSignals() { apiFetch('/api/signals').then(data => { const tableBody = document.getElementById('signals-table'); if (!data || data.error) { tableBody.innerHTML = '<tr><td colspan="7" class="p-8 text-center text-text-secondary">فشل تحميل الصفقات.</td></tr>'; return; } if (data.length === 0) { tableBody.innerHTML = '<tr><td colspan="7" class="p-8 text-center text-text-secondary">لا توجد صفقات لعرضها.</td></tr>'; return; } tableBody.innerHTML = data.map(signal => { const pnlPct = (signal.status === 'open' || signal.status === 'updated') ? signal.pnl_pct : signal.profit_percentage; const pnlDisplay = pnlPct !== null && pnlPct !== undefined ? `${formatNumber(pnlPct)}%` : 'N/A'; const pnlColor = pnlPct === null || pnlPct === undefined ? 'text-text-secondary' : (pnlPct >= 0 ? 'text-accent-green' : 'text-accent-red'); let statusClass = 'text-gray-400'; let statusText = signal.status; if (signal.status === 'open') { statusClass = 'text-yellow-400'; statusText = 'مفتوحة'; } else if (signal.status === 'updated') { statusClass = 'text-blue-400'; statusText = 'تم تحديثها'; } const quantityDisplay = signal.quantity ? formatNumber(signal.quantity, 4) : '-'; const realTradeIndicator = signal.is_real_trade ? '<span class="text-accent-green" title="صفقة حقيقية">●</span>' : ''; return `<tr class="table-row border-b border-border-color"> <td class="p-4 font-mono font-semibold">${realTradeIndicator} ${signal.symbol}</td> <td class="p-4 font-bold ${statusClass}">${statusText}</td> <td class="p-4 font-mono text-text-secondary">${quantityDisplay}</td> <td class="p-4 font-mono font-bold ${pnlColor}">${pnlDisplay}</td> <td class="p-4">${(signal.status === 'open' || signal.status === 'updated') ? renderProgressBar(signal) : '-'}</td> <td class="p-4 font-mono text-xs"><div>${formatNumber(signal.entry_price, 5)}</div><div class="text-text-secondary">${formatNumber(signal.current_price, 5)}</div></td> <td class="p-4">${(signal.status === 'open' || signal.status === 'updated') ? `<button onclick="manualCloseSignal(${signal.id})" class="bg-red-600/80 hover:bg-red-600 text-white text-xs py-1 px-3 rounded-md">إغلاق</button>` : ''}</td> </tr>`; }).join(''); }); }
        function updateList(endpoint, listId, formatter) { apiFetch(endpoint).then(data => { if (!data || data.error) return; document.getElementById(listId).innerHTML = data.map(formatter).join('') || `<div class="p-4 text-center text-text-secondary">لا توجد بيانات.</div>`; }); }
        function manualCloseSignal(signalId) { if (confirm(`هل أنت متأكد من رغبتك في إغلاق الصفقة #${signalId} يدوياً؟ سيتم بيع الكمية بسعر السوق إذا كان التداول الحقيقي مُفعّلاً.`)) { fetch(`/api/close/${signalId}`, { method: 'POST' }).then(res => res.json()).then(data => { alert(data.message || data.error); refreshData(); }); } }
        function refreshData() { updateMarketStatus(); updateTradingStatus(); updateStats(); updateProfitChart(); updateSignals(); const dateLocaleOptions = { timeZone: 'UTC', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }; const locale = 'fr-CA'; updateList('/api/notifications', 'notifications-list', n => `<div class="p-3 rounded-md bg-gray-900/50 text-sm">[${new Date(n.timestamp).toLocaleString(locale, dateLocaleOptions)}] ${n.message}</div>`); updateList('/api/rejection_logs', 'rejections-list', log => `<div class="p-3 rounded-md bg-gray-900/50 text-sm">[${new Date(log.timestamp).toLocaleString(locale, dateLocaleOptions)}] <strong>${log.symbol}</strong>: ${log.reason} - <span class="font-mono text-xs text-text-secondary">${JSON.stringify(log.details)}</span></div>`); }
        setInterval(refreshData, 5000);
        window.onload = refreshData;
    </script>
</body>
</html>
    """

# ---------------------- دوال قاعدة البيانات (مع إصلاح الترقية) ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    # ... الكود يبقى كما هو ...
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
                        current_peak_price DOUBLE PRECISION
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals (status);")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL, message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE
                    );
                """)
                
                logger.info("[DB Migration] Checking for necessary schema upgrades...")
                
                cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'signals' AND table_schema = 'public';")
                existing_columns = [row['column_name'] for row in cur.fetchall()]
                
                required_columns = {
                    'is_real_trade': 'BOOLEAN DEFAULT FALSE',
                    'quantity': 'DOUBLE PRECISION',
                    'order_id': 'TEXT'
                }
                
                for col_name, col_type in required_columns.items():
                    if col_name not in existing_columns:
                        logger.warning(f"[DB Migration] Column '{col_name}' is missing. Adding it now...")
                        cur.execute(sql.SQL("ALTER TABLE signals ADD COLUMN {} {}").format(
                            sql.Identifier(col_name), sql.SQL(col_type)
                        ))
                        logger.info(f"[DB Migration] Successfully added column '{col_name}'.")
                
            conn.commit()
            logger.info("✅ [DB] Database connection and schema are up-to-date.")
            return

        except Exception as e:
            logger.error(f"❌ [DB] Error during initialization/migration (Attempt {attempt + 1}/{retries}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: logger.critical("❌ [DB] Failed to connect/migrate the database after multiple retries.")

def check_db_connection() -> bool:
    # ... الكود يبقى كما هو ...
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
        logger.error(f"❌ [DB] Connection lost: {e}. Attempting to reconnect...")
        try:
            init_db()
            return conn is not None and conn.closed == 0
        except Exception as retry_e:
            logger.error(f"❌ [DB] Reconnect failed: {retry_e}")
            return False
    return False

def log_and_notify(level: str, message: str, notification_type: str):
    # ... الكود يبقى كما هو ...
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error, 'critical': logger.critical}
    log_methods.get(level.lower(), logger.info)(message)
    if not check_db_connection() or not conn: return
    try:
        new_notification = {"timestamp": datetime.now().isoformat(), "type": notification_type, "message": message}
        with notifications_lock: notifications_cache.appendleft(new_notification)
        with conn.cursor() as cur: cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [Notify DB] Failed to save notification to DB: {e}")
        if conn: conn.rollback()

def log_rejection(symbol: str, reason: str, details: Optional[Dict] = None):
    # ... الكود يبقى كما هو ...
    log_message = f"🚫 [REJECTED] {symbol} | Reason: {reason} | Details: {details or {}}"
    logger.info(log_message)
    with rejection_logs_lock:
        rejection_logs_cache.appendleft({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "reason": reason,
            "details": details or {}
        })

def init_redis() -> None:
    # ... الكود يبقى كما هو ...
    global redis_client
    logger.info("[Redis] Initializing Redis connection...")
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] Successfully connected to Redis server.")
    except redis.exceptions.ConnectionError as e:
        logger.critical(f"❌ [Redis] Failed to connect to Redis. Error: {e}")
        exit(1)

# ---------------------- دوال Binance والبيانات (مع إضافة دوال الميزات الجديدة) ----------------------
def get_exchange_info_map() -> None:
    # ... الكود يبقى كما هو ...
    global exchange_info_map
    if not client: return
    logger.info("ℹ️ [Exchange Info] Fetching exchange trading rules...")
    try:
        info = client.get_exchange_info()
        exchange_info_map = {s['symbol']: s for s in info['symbols']}
        logger.info(f"✅ [Exchange Info] Successfully loaded trading rules for {len(exchange_info_map)} symbols.")
    except Exception as e:
        logger.error(f"❌ [Exchange Info] Could not fetch exchange info: {e}")

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    # ... الكود يبقى كما هو ...
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
        logger.info(f"✅ [Validation] Bot will monitor {len(validated)} validated symbols.")
        return validated
    except Exception as e:
        logger.error(f"❌ [Validation] An error occurred during symbol validation: {e}", exc_info=True)
        return []

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    # ... الكود يبقى كما هو ...
    if not client: return None
    try:
        limit = int((days * 24 * 60) / int(re.sub('[a-zA-Z]', '', interval)))
        klines = client.get_historical_klines(symbol, interval, limit=min(limit, 1000))
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna()
    except Exception as e:
        logger.error(f"❌ [Data] Error fetching historical data for {symbol}: {e}")
        return None

# --- [جديد] دوال جلب الميزات الإضافية ---
def fetch_fear_and_greed_data() -> None:
    """جلب بيانات مؤشر الخوف والطمع وتخزينها مؤقتاً."""
    global fg_data_cache, last_fg_fetch_time
    # تحديث كل ساعة لتجنب استدعاء API بشكل مفرط
    if time.time() - last_fg_fetch_time < 3600 and fg_data_cache is not None:
        return
    
    logger.info("ℹ️ [F&G] Attempting to fetch Fear & Greed index data...")
    try:
        url = "https://api.alternative.me/fng/?limit=90&format=json" # جلب بيانات 90 يوم
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()['data']
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.normalize()
        df = df[['timestamp', 'value']]
        df.rename(columns={'value': 'fear_greed_value'}, inplace=True)
        df['fear_greed_value'] = pd.to_numeric(df['fear_greed_value'])
        df.set_index('timestamp', inplace=True)
        
        fg_data_cache = df
        last_fg_fetch_time = time.time()
        logger.info(f"✅ [F&G] Successfully fetched and cached {len(df)} records.")
    except Exception as e:
        logger.error(f"❌ [F&G] Failed to fetch Fear & Greed data: {e}")
        # في حالة الفشل، ننشئ DataFrame فارغ لتجنب تكرار المحاولة فوراً
        if fg_data_cache is None:
             fg_data_cache = pd.DataFrame()
        last_fg_fetch_time = time.time()

def fetch_sr_levels(symbol: str) -> Optional[pd.DataFrame]:
    """جلب مستويات الدعم والمقاومة لعملة معينة من قاعدة البيانات."""
    if not check_db_connection() or not conn: return None
    query = sql.SQL("SELECT level_price, level_type, score FROM support_resistance_levels WHERE symbol = %s AND score > 20;")
    try:
        with conn.cursor() as cur:
            cur.execute(query, (symbol,))
            levels = cur.fetchall()
        if not levels: return None
        df_levels = pd.DataFrame(levels)
        df_levels['level_price'] = df_levels['level_price'].astype(float)
        df_levels['score'] = df_levels['score'].astype(float)
        return df_levels
    except Exception as e:
        logger.error(f"❌ [S/R Fetch] Error fetching S/R levels for {symbol}: {e}")
        if conn: conn.rollback()
        return None

def calculate_order_book_features(symbol: str) -> Dict[str, float]:
    """حساب معالم من دفتر الأوامر الحالي للعملة."""
    default_features = {'bid_ask_spread': 0.0, 'order_book_imbalance': 0.0, 'liquidity_density': 0.0}
    if not client: return default_features
    try:
        order_book = client.get_order_book(symbol=symbol, limit=20)
        if not order_book or not order_book['bids'] or not order_book['asks']: return default_features
        bids = pd.DataFrame(order_book['bids'], columns=['price', 'qty'], dtype=float)
        asks = pd.DataFrame(order_book['asks'], columns=['price', 'qty'], dtype=float)
        best_bid, best_ask = bids['price'].iloc[0], asks['price'].iloc[0]
        spread = (best_ask - best_bid) / best_ask if best_ask > 0 else 0.0
        total_bid_volume = (bids['qty']).sum()
        total_ask_volume = (asks['qty']).sum()
        total_volume = total_bid_volume + total_ask_volume
        imbalance = (total_bid_volume - total_ask_volume) / total_volume if total_volume > 0 else 0.0
        density = (bids['qty'].sum() + asks['qty'].sum()) / (len(bids) + len(asks))
        return {'bid_ask_spread': spread, 'order_book_imbalance': imbalance, 'liquidity_density': density}
    except Exception as e:
        logger.warning(f"⚠️ [Order Book] Could not calculate order book features for {symbol}: {e}")
        return default_features

# ---------------------- [معدل] دوال حساب الميزات وتحديد الاتجاه ----------------------
def calculate_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    # دالة حساب المؤشرات الفنية الأساسية (تبقى كما هي)
    df_calc = df.copy()
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
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)
    df_calc['price_vs_ema50'] = (df_calc['close'] / df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()) - 1
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

def get_trend_for_timeframe(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    # ... الكود يبقى كما هو ...
    if df is None or len(df) < 26: return {"trend": "Uncertain", "rsi": -1, "adx": -1}
    try:
        close_series = df['close']
        delta = close_series.diff()
        gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
        rsi = (100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))).iloc[-1]
        high_low = df['high'] - df['low']
        high_close = (df['high'] - close_series.shift()).abs()
        low_close = (df['low'] - close_series.shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(span=ADX_PERIOD, adjust=False).mean()
        up_move = df['high'].diff(); down_move = -df['low'].diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
        plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / atr.replace(0, 1e-9)
        minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / atr.replace(0, 1e-9)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
        adx = dx.ewm(span=ADX_PERIOD, adjust=False).mean().iloc[-1]
        ema_fast = close_series.ewm(span=12, adjust=False).mean().iloc[-1]
        ema_slow = close_series.ewm(span=26, adjust=False).mean().iloc[-1]
        trend = "Ranging"
        if adx > 20:
            if ema_fast > ema_slow and rsi > 50: trend = "Uptrend"
            elif ema_fast < ema_slow and rsi < 50: trend = "Downtrend"
        return {"trend": trend, "rsi": float(rsi), "adx": float(adx)}
    except Exception as e:
        logger.error(f"Error in get_trend_for_timeframe: {e}")
        return {"trend": "Uncertain", "rsi": -1, "adx": -1}

def determine_market_state():
    # ... الكود يبقى كما هو ...
    global current_market_state, last_market_state_check
    with market_state_lock:
        if time.time() - last_market_state_check < 300: return
    logger.info("🧠 [Market State] Updating market state...")
    try:
        df_15m = fetch_historical_data(BTC_SYMBOL, '15m', 2)
        df_1h = fetch_historical_data(BTC_SYMBOL, '1h', 5)
        df_4h = fetch_historical_data(BTC_SYMBOL, '4h', 15)
        state_15m = get_trend_for_timeframe(df_15m)
        state_1h = get_trend_for_timeframe(df_1h)
        state_4h = get_trend_for_timeframe(df_4h)
        trends = [state_15m['trend'], state_1h['trend'], state_4h['trend']]
        uptrends = trends.count("Uptrend")
        downtrends = trends.count("Downtrend")
        overall_regime = "RANGING"
        if uptrends == 3: overall_regime = "STRONG UPTREND"
        elif uptrends >= 2 and downtrends == 0: overall_regime = "UPTREND"
        elif downtrends == 3: overall_regime = "STRONG DOWNTREND"
        elif downtrends >= 2 and uptrends == 0: overall_regime = "DOWNTREND"
        elif "Uncertain" in trends: overall_regime = "UNCERTAIN"
        with market_state_lock:
            current_market_state = {
                "overall_regime": overall_regime,
                "details": {"15m": state_15m, "1h": state_1h, "4h": state_4h},
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            last_market_state_check = time.time()
        logger.info(f"✅ [Market State] New state: {overall_regime} (15m: {state_15m['trend']}, 1h: {state_1h['trend']}, 4h: {state_4h['trend']})")
    except Exception as e:
        logger.error(f"❌ [Market State] Failed to determine market state: {e}", exc_info=True)
        with market_state_lock: current_market_state['overall_regime'] = "UNCERTAIN"

def load_ml_model_bundle_from_folder(symbol: str) -> Optional[Dict[str, Any]]:
    # ... الكود يبقى كما هو (مع تحديث اسم النموذج والمجلد) ...
    global ml_models_cache
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_name in ml_models_cache:
        return ml_models_cache[model_name]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir_path = os.path.join(script_dir, MODEL_FOLDER)
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
        logger.info(f"📁 Created model directory: {model_dir_path}")
    model_path = os.path.join(model_dir_path, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        logger.warning(f"⚠️ [ML Model] Model file not found at '{model_path}'.")
        return None
    try:
        with open(model_path, 'rb') as f:
            model_bundle = pickle.load(f)
        if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
            ml_models_cache[model_name] = model_bundle
            logger.info(f"✅ [ML Model] Loaded model '{model_name}' successfully.")
            return model_bundle
        else:
            logger.error(f"❌ [ML Model] Model bundle at '{model_path}' is incomplete.")
            return None
    except Exception as e:
        logger.error(f"❌ [ML Model] Error loading model for symbol {symbol}: {e}", exc_info=True)
        return None

# ---------------------- [معدل] دوال الاستراتيجية والتداول الحقيقي ----------------------
def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> Optional[Decimal]:
    # ... الكود يبقى كما هو ...
    try:
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info:
            logger.error(f"[{symbol}] Could not find exchange info for lot size adjustment.")
            return None
        
        for f in symbol_info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                step_size_str = f['stepSize']
                step_size = Decimal(step_size_str)
                
                quantity_dec = Decimal(str(quantity))
                adjusted_quantity = (quantity_dec // step_size) * step_size
                
                logger.debug(f"[{symbol}] Adjusted quantity from {quantity} to {adjusted_quantity} with step size {step_size}")
                return adjusted_quantity
        return Decimal(str(quantity))
    except Exception as e:
        logger.error(f"[{symbol}] Error adjusting quantity to lot size: {e}")
        return None

def calculate_position_size(symbol: str, entry_price: float, stop_loss_price: float) -> Optional[Decimal]:
    # ... الكود يبقى كما هو ...
    if not client: return None
    try:
        balance_response = client.get_asset_balance(asset='USDT')
        available_balance = Decimal(balance_response['free'])
        logger.info(f"[{symbol}] Available USDT balance: {available_balance:.2f}")

        risk_amount_usdt = available_balance * (Decimal(str(RISK_PER_TRADE_PERCENT)) / Decimal('100'))
        
        risk_per_coin = Decimal(str(entry_price)) - Decimal(str(stop_loss_price))
        if risk_per_coin <= 0:
            log_rejection(symbol, "Invalid Position Size", {"detail": "Stop loss must be below entry price."})
            return None
            
        initial_quantity = risk_amount_usdt / risk_per_coin
        
        adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(initial_quantity))
        if adjusted_quantity is None or adjusted_quantity <= 0:
            log_rejection(symbol, "Lot Size Adjustment Failed", {"detail": f"Adjusted quantity is zero or invalid: {adjusted_quantity}"})
            return None

        notional_value = adjusted_quantity * Decimal(str(entry_price))
        symbol_info = exchange_info_map.get(symbol)
        if symbol_info:
            for f in symbol_info['filters']:
                if f['filterType'] == 'MIN_NOTIONAL' or f['filterType'] == 'NOTIONAL':
                    min_notional = Decimal(f.get('minNotional', f.get('notional', '0')))
                    if notional_value < min_notional:
                        log_rejection(symbol, "Min Notional Filter", {"value": f"{notional_value:.2f}", "required": f"{min_notional}"})
                        return None
        
        if notional_value > available_balance:
            log_rejection(symbol, "Insufficient Balance", {"required": f"{notional_value:.2f}", "available": f"{available_balance:.2f}"})
            return None

        logger.info(f"✅ [{symbol}] Calculated position size: {adjusted_quantity} | Risk: ${risk_amount_usdt:.2f} | Notional: ${notional_value:.2f}")
        return adjusted_quantity

    except BinanceAPIException as e:
        logger.error(f"❌ [{symbol}] Binance API error during position size calculation: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ [{symbol}] Unexpected error in calculate_position_size: {e}", exc_info=True)
        return None

def place_order(symbol: str, side: str, quantity: Decimal, order_type: str = Client.ORDER_TYPE_MARKET) -> Optional[Dict]:
    # ... الكود يبقى كما هو ...
    if not client: return None
    logger.info(f"➡️ [{symbol}] Attempting to place a REAL {side} order for {quantity} units.")
    try:
        order = client.create_order(
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=float(quantity)
        )
        logger.info(f"✅ [{symbol}] REAL {side} order placed successfully! Order ID: {order['orderId']}")
        log_and_notify('info', f"REAL TRADE: Placed {side} order for {quantity} {symbol}.", "REAL_TRADE")
        return order
    except BinanceAPIException as e:
        logger.error(f"❌ [{symbol}] Binance API Exception on order placement: {e}")
        log_and_notify('error', f"REAL TRADE FAILED: {symbol} | {e}", "REAL_TRADE_ERROR")
        return None
    except Exception as e:
        logger.error(f"❌ [{symbol}] Unexpected error on order placement: {e}", exc_info=True)
        return None

class TradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_bundle_from_folder(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """[معدل] دالة تجميع كل الميزات المطلوبة للنموذج الجديد."""
        if self.feature_names is None: return None
        try:
            # 1. حساب المؤشرات الفنية
            df_featured = calculate_features(df_15m, btc_df)

            # 2. إضافة معالم الدعم والمقاومة
            sr_levels = fetch_sr_levels(self.symbol)
            if sr_levels is not None and not sr_levels.empty:
                support_prices = np.sort(sr_levels[sr_levels['level_type'].str.contains('support|poc|confluence', case=False, na=False)]['level_price'].unique())
                resistance_prices = np.sort(sr_levels[sr_levels['level_type'].str.contains('resistance|poc|confluence', case=False, na=False)]['level_price'].unique())
                price_to_score = sr_levels.set_index('level_price')['score'].to_dict()
                
                # حساب المسافة لأقرب دعم ومقاومة
                last_price = df_featured['close'].iloc[-1]
                
                nearest_support_price = support_prices[np.searchsorted(support_prices, last_price) - 1] if np.searchsorted(support_prices, last_price) > 0 else 0
                dist_to_support = (last_price - nearest_support_price) / last_price if nearest_support_price > 0 else 1.0
                score_of_support = price_to_score.get(nearest_support_price, 0)

                nearest_resistance_price = resistance_prices[np.searchsorted(resistance_prices, last_price)] if np.searchsorted(resistance_prices, last_price) < len(resistance_prices) else float('inf')
                dist_to_resistance = (nearest_resistance_price - last_price) / last_price if last_price > 0 else 1.0
                score_of_resistance = price_to_score.get(nearest_resistance_price, 0)
                
                df_featured['dist_to_support'] = dist_to_support
                df_featured['score_of_support'] = score_of_support
                df_featured['dist_to_resistance'] = dist_to_resistance
                df_featured['score_of_resistance'] = score_of_resistance
            else:
                df_featured['dist_to_support'] = 1.0; df_featured['score_of_support'] = 0
                df_featured['dist_to_resistance'] = 1.0; df_featured['score_of_resistance'] = 0

            # 3. إضافة معلم الخوف والطمع
            if fg_data_cache is not None and not fg_data_cache.empty:
                # دمج بناءً على التاريخ فقط
                df_featured = pd.merge(df_featured, fg_data_cache, left_on=df_featured.index.date, right_on=fg_data_cache.index.date, how='left')
                df_featured.index = df_15m.index # استعادة الفهرس الأصلي
                df_featured['fear_greed_value'].fillna(method='ffill', inplace=True)
                df_featured['fear_greed_value'].fillna(method='bfill', inplace=True)
            df_featured['fear_greed_value'] = df_featured.get('fear_greed_value', 50.0) # قيمة افتراضية

            # 4. إضافة معالم دفتر الأوامر
            ob_features = calculate_order_book_features(self.symbol)
            for key, value in ob_features.items():
                df_featured[key] = value

            # 5. إضافة معالم الفريم الأعلى
            df_4h_features = calculate_features(df_4h, None)
            df_4h_features = df_4h_features.rename(columns={'rsi': 'rsi_4h', 'price_vs_ema50': 'price_vs_ema50_4h'})
            required_4h_cols = ['rsi_4h', 'price_vs_ema50_4h']
            df_featured = df_featured.join(df_4h_features[required_4h_cols])
            df_featured.fillna(method='ffill', inplace=True)

            # التأكد من وجود كل الأعمدة المطلوبة
            for col in self.feature_names:
                if col not in df_featured.columns:
                    logger.warning(f"[{self.symbol}] Missing feature '{col}', filling with 0.")
                    df_featured[col] = 0.0
            
            df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_featured.dropna(subset=self.feature_names)
        except Exception as e:
            logger.error(f"❌ [{self.symbol}] Feature engineering failed: {e}", exc_info=True)
            return None

    def generate_signal(self, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        # ... الكود يبقى كما هو ...
        if not all([self.ml_model, self.scaler, self.feature_names]) or df_features.empty: return None
        try:
            last_row_ordered_df = df_features.iloc[[-1]][self.feature_names]
            features_scaled_np = self.scaler.transform(last_row_ordered_df)
            
            prediction = self.ml_model.predict(features_scaled_np)[0]
            prediction_proba = self.ml_model.predict_proba(features_scaled_np)
            
            confidence = float(np.max(prediction_proba[0]))
            
            logger.info(f"ℹ️ [{self.symbol}] Advanced Model predicted '{'BUY' if prediction == 1 else 'SELL/HOLD'}' with {confidence:.2%} confidence.")
            return {'prediction': int(prediction), 'confidence': confidence}
        except Exception as e:
            logger.warning(f"⚠️ [{self.symbol}] Signal Generation Error: {e}")
            return None

# ... بقية الدوال (الفلاتر، إدارة الصفقات، إلخ) تبقى كما هي ...
def passes_momentum_filter(last_features: pd.Series) -> bool:
    symbol = last_features.name
    roc = last_features.get(f'roc_{MOMENTUM_PERIOD}', 0)
    accel = last_features.get('roc_acceleration', 0)
    slope = last_features.get(f'ema_slope_{EMA_SLOPE_PERIOD}', 0)
    if roc > 0 and accel >= 0 and slope > 0:
        return True
    log_rejection(symbol, "Momentum Filter", {
        "ROC": f"{roc:.2f} (Req: > 0)",
        "Acceleration": f"{accel:.4f} (Req: >= 0)",
        "Slope": f"{slope:.6f} (Req: > 0)"
    })
    return False

def passes_speed_filter(last_features: pd.Series) -> bool:
    symbol = last_features.name
    with market_state_lock: regime = current_market_state.get("overall_regime", "RANGING")
    if regime in ["DOWNTREND", "STRONG DOWNTREND"]:
        log_rejection(symbol, "Speed Filter", {"detail": f"Disabled due to market regime: {regime}"})
        return True
    if regime == "STRONG UPTREND": adx_threshold, rel_vol_threshold, rsi_min, rsi_max = (30.0, 0.1, 50.0, 75.0)
    elif regime == "UPTREND": adx_threshold, rel_vol_threshold, rsi_min, rsi_max = (25.0, 0.05, 45.0, 75.0)
    else: adx_threshold, rel_vol_threshold, rsi_min, rsi_max = (20.0, 0.02, 35.0, 65.0)
    adx, rel_vol, rsi = last_features.get('adx', 0), last_features.get('relative_volume', 0), last_features.get('rsi', 0)
    if (adx >= adx_threshold and rel_vol >= rel_vol_threshold and rsi_min <= rsi < rsi_max): return True
    log_rejection(symbol, "Speed Filter", {"Regime": regime, "ADX": f"{adx:.2f} (Req: >{adx_threshold})", "Volume": f"{rel_vol:.2f} (Req: >{rel_vol_threshold})", "RSI": f"{rsi:.2f} (Req: {rsi_min}-{rsi_max})"})
    return False

def calculate_tp_sl(symbol: str, entry_price: float, last_atr: float) -> Optional[Dict[str, Any]]:
    if last_atr <= 0:
        log_rejection(symbol, "Invalid ATR for Fallback", {"detail": "ATR is zero or negative"})
        return None
    fallback_tp = entry_price + (last_atr * ATR_FALLBACK_TP_MULTIPLIER)
    fallback_sl = entry_price - (last_atr * ATR_FALLBACK_SL_MULTIPLIER)
    return {'target_price': fallback_tp, 'stop_loss': fallback_sl, 'source': 'ATR_Fallback'}

def handle_price_update_message(msg: List[Dict[str, Any]]) -> None:
    if not isinstance(msg, list) or not redis_client: return
    try:
        price_updates = {item.get('s'): float(item.get('c', 0)) for item in msg if item.get('s') and item.get('c')}
        if price_updates: redis_client.hset(REDIS_PRICES_HASH_NAME, mapping=price_updates)
    except Exception as e: logger.error(f"❌ [WebSocket Price Updater] Error: {e}", exc_info=True)

def initiate_signal_closure(symbol: str, signal_to_close: Dict, status: str, closing_price: float):
    signal_id = signal_to_close.get('id')
    if not signal_id:
        logger.error(f"❌ [Closure] Attempted to close a signal without an ID for symbol {symbol}")
        return
    with closure_lock:
        if signal_id in signals_pending_closure:
            logger.warning(f"⚠️ [Closure] Closure for signal {signal_id} ({symbol}) already in progress.")
            return
        signals_pending_closure.add(signal_id)
    with signal_cache_lock: open_signals_cache.pop(symbol, None)
    logger.info(f"ℹ️ [Closure] Starting closure thread for signal {signal_id} ({symbol}) with status '{status}'.")
    Thread(target=close_signal, args=(signal_to_close, status, closing_price)).start()

def update_signal_peak_price_in_db(signal_id: int, new_peak_price: float):
    if not check_db_connection() or not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE signals SET current_peak_price = %s WHERE id = %s;", (new_peak_price, signal_id))
        conn.commit()
        logger.debug(f"💾 [DB Peak Update] Saved new peak price {new_peak_price} for signal {signal_id}.")
    except Exception as e:
        logger.error(f"❌ [DB Peak Update] Failed to update peak price for signal {signal_id}: {e}")
        if conn: conn.rollback()

def trade_monitoring_loop():
    global last_api_check_time
    logger.info("✅ [Trade Monitor] Starting trade monitoring loop.")
    while True:
        try:
            with signal_cache_lock:
                signals_to_check = dict(open_signals_cache)
            if not signals_to_check or not redis_client or not client:
                time.sleep(1); continue
            
            perform_direct_api_check = (time.time() - last_api_check_time) > DIRECT_API_CHECK_INTERVAL
            if perform_direct_api_check:
                last_api_check_time = time.time()
            
            symbols_to_fetch = list(signals_to_check.keys())
            redis_prices_list = redis_client.hmget(REDIS_PRICES_HASH_NAME, symbols_to_fetch)
            redis_prices = {symbol: price for symbol, price in zip(symbols_to_fetch, redis_prices_list)}
            
            for symbol, signal in signals_to_check.items():
                signal_id = signal.get('id')
                if not signal_id: continue
                with closure_lock:
                    if signal_id in signals_pending_closure: continue
                
                price = None
                if perform_direct_api_check:
                    try: price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                    except Exception: pass
                if not price and redis_prices.get(symbol):
                    try: price = float(redis_prices[symbol])
                    except (ValueError, TypeError): continue
                if not price: continue
                
                with signal_cache_lock:
                    if symbol in open_signals_cache:
                        open_signals_cache[symbol]['current_price'] = price
                        open_signals_cache[symbol]['pnl_pct'] = ((price / float(signal['entry_price'])) - 1) * 100
                
                target_price = float(signal.get('target_price', 0))
                original_stop_loss = float(signal.get('stop_loss', 0))
                effective_stop_loss = original_stop_loss
                
                if USE_TRAILING_STOP_LOSS:
                    entry_price = float(signal.get('entry_price', 0))
                    activation_price = entry_price * (1 + TRAILING_ACTIVATION_PROFIT_PERCENT / 100)
                    if price > activation_price:
                        current_peak = float(signal.get('current_peak_price', entry_price))
                        if price > current_peak:
                            with signal_cache_lock:
                                if symbol in open_signals_cache:
                                    open_signals_cache[symbol]['current_peak_price'] = price
                            now = time.time()
                            if now - LAST_PEAK_UPDATE_TIME.get(signal_id, 0) > PEAK_UPDATE_COOLDOWN:
                                update_signal_peak_price_in_db(signal_id, price)
                                LAST_PEAK_UPDATE_TIME[signal_id] = now
                            current_peak = price
                        
                        trailing_stop_price = current_peak * (1 - TRAILING_DISTANCE_PERCENT / 100)
                        if trailing_stop_price > effective_stop_loss:
                            logger.info(f"📈 [Trailing SL] {symbol} new peak: {current_peak:.4f}. Adjusted SL to: {trailing_stop_price:.4f}")
                            effective_stop_loss = trailing_stop_price
                
                status_to_set = None
                if price >= target_price: status_to_set = 'target_hit'
                elif price <= effective_stop_loss: status_to_set = 'stop_loss_hit'
                
                if status_to_set:
                    logger.info(f"✅ [TRIGGER] ID:{signal_id} | {symbol} | Condition '{status_to_set}' met at price {price}.")
                    initiate_signal_closure(symbol, signal, status_to_set, price)
            time.sleep(0.2)
        except Exception as e:
            logger.error(f"❌ [Trade Monitor] Critical error: {e}", exc_info=True)
            time.sleep(5)

def send_telegram_message(target_chat_id: str, text: str, reply_markup: Optional[Dict] = None) -> bool:
    if not TELEGRAM_TOKEN or not target_chat_id: return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(target_chat_id), 'text': text, 'parse_mode': 'Markdown'}
    if reply_markup: payload['reply_markup'] = json.dumps(reply_markup)
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200: return True
        else: logger.error(f"❌ [Telegram] Failed to send message. Status: {response.status_code}, Response: {response.text}"); return False
    except requests.exceptions.RequestException as e: logger.error(f"❌ [Telegram] Request failed: {e}"); return False

def send_new_signal_alert(signal_data: Dict[str, Any]):
    symbol = signal_data['symbol']; entry = float(signal_data['entry_price']); target = float(signal_data['target_price']); sl = float(signal_data['stop_loss'])
    profit_pct = ((target / entry) - 1) * 100
    risk_pct = abs(((entry / sl) - 1) * 100) if sl > 0 else 0
    rrr = profit_pct / risk_pct if risk_pct > 0 else 0
    with market_state_lock: market_regime = current_market_state.get('overall_regime', 'N/A')
    confidence_display = signal_data['signal_details'].get('ML_Confidence_Display', 'N/A')
    
    trade_type_msg = ""
    if signal_data.get('is_real_trade'):
        quantity = signal_data.get('quantity')
        trade_type_msg = f"\n*🔥 صفقة حقيقية 🔥*\n*الكمية:* `{quantity}`\n"

    message = (f"💡 *توصية تداول جديدة (Advanced V3)* 💡\n\n*العملة:* `{symbol}`\n*حالة السوق:* `{market_regime}`\n{trade_type_msg}\n"
               f"*الدخول:* `{entry:.8f}`\n*الهدف:* `{target:.8f}`\n*وقف الخسارة:* `{sl:.8f}`\n\n"
               f"*الربح المتوقع:* `{profit_pct:.2f}%`\n*المخاطرة/العائد:* `1:{rrr:.2f}`\n\n"
               f"*ثقة النموذج:* `{confidence_display}`")
    reply_markup = {"inline_keyboard": [[{"text": "📊 فتح لوحة التحكم", "url": WEBHOOK_URL or '#'}]]}
    if send_telegram_message(CHAT_ID, message, reply_markup):
        log_and_notify('info', f"New Advanced Signal: {symbol} in {market_regime} market. Real Trade: {signal_data.get('is_real_trade', False)}", "NEW_SIGNAL")

def send_trade_update_alert(signal_data: Dict[str, Any], old_signal_data: Dict[str, Any]):
    symbol = signal_data['symbol']
    old_target = float(old_signal_data['target_price']); new_target = float(signal_data['target_price'])
    old_sl = float(old_signal_data['stop_loss']); new_sl = float(signal_data['stop_loss'])
    old_conf = old_signal_data['signal_details'].get('ML_Confidence_Display', 'N/A')
    new_conf = signal_data['signal_details'].get('ML_Confidence_Display', 'N/A')
    
    message = (f"🔄 *تحديث صفقة (تعزيز) - Advanced V3* 🔄\n\n"
               f"*العملة:* `{symbol}`\n\n"
               f"*الثقة:* `{old_conf}` ⬅️ `{new_conf}`\n"
               f"*الهدف:* `{old_target:.8f}` ⬅️ `{new_target:.8f}`\n"
               f"*الوقف:* `{old_sl:.8f}` ⬅️ `{new_sl:.8f}`\n\n"
               f"تم تحديث الصفقة بناءً على إشارة شراء أقوى.")
    reply_markup = {"inline_keyboard": [[{"text": "📊 فتح لوحة التحكم", "url": WEBHOOK_URL or '#'}]]}
    if send_telegram_message(CHAT_ID, message, reply_markup):
        log_and_notify('info', f"Updated Advanced Signal: {symbol} due to stronger signal.", "UPDATE_SIGNAL")

def insert_signal_into_db(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not check_db_connection() or not conn: return None
    try:
        entry = float(signal['entry_price']); target = float(signal['target_price']); sl = float(signal['stop_loss'])
        is_real = signal.get('is_real_trade', False)
        quantity = signal.get('quantity')
        order_id = signal.get('order_id')

        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, current_peak_price, is_real_trade, quantity, order_id) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
                """,
                (signal['symbol'], entry, target, sl, signal.get('strategy_name'), json.dumps(signal.get('signal_details', {})), entry, is_real, quantity, order_id))
            signal['id'] = cur.fetchone()['id']
        conn.commit()
        logger.info(f"✅ [DB] Inserted signal {signal['id']} for {signal['symbol']}. Real Trade: {is_real}")
        return signal
    except Exception as e:
        logger.error(f"❌ [Insert] Error inserting signal for {signal['symbol']}: {e}", exc_info=True)
        if conn: conn.rollback()
        return None

def update_signal_in_db(signal_id: int, new_data: Dict[str, Any]) -> bool:
    if not check_db_connection() or not conn: return False
    try:
        target = float(new_data['target_price'])
        sl = float(new_data['stop_loss'])
        details = json.dumps(new_data.get('signal_details', {}))
        
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE signals 
                SET target_price = %s, stop_loss = %s, signal_details = %s, status = 'updated'
                WHERE id = %s AND status IN ('open', 'updated');
            """, (target, sl, details, signal_id))
            if cur.rowcount == 0:
                logger.warning(f"⚠️ [DB Update] Signal {signal_id} not found or already closed. No update performed.")
                return False
        conn.commit()
        logger.info(f"✅ [DB] Updated signal {signal_id} for {new_data['symbol']}.")
        return True
    except Exception as e:
        logger.error(f"❌ [Update] Error updating signal {signal_id}: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

def close_signal(signal: Dict, status: str, closing_price: float):
    signal_id = signal.get('id')
    symbol = signal.get('symbol')
    logger.info(f"Initiating closure for signal {signal_id} ({symbol}) with status '{status}'")
    
    is_real = signal.get('is_real_trade', False)
    
    with trading_status_lock:
        is_enabled = is_trading_enabled

    if is_real and is_enabled:
        try:
            base_asset = exchange_info_map.get(symbol, {}).get('baseAsset')
            if not base_asset or not client:
                raise ValueError(f"Could not determine base asset for {symbol} or client not ready.")
            balance_info = client.get_asset_balance(asset=base_asset)
            actual_free_balance = float(balance_info['free'])
            logger.info(f"🔥 [{symbol}] REAL TRADE CLOSURE. Actual free balance for {base_asset}: {actual_free_balance}")
            quantity_to_sell_adjusted = adjust_quantity_to_lot_size(symbol, actual_free_balance)
            if quantity_to_sell_adjusted and quantity_to_sell_adjusted > 0:
                sell_order = place_order(symbol, Client.SIDE_SELL, quantity_to_sell_adjusted)
                if not sell_order:
                    logger.critical(f"🚨 CRITICAL: FAILED TO PLACE SELL ORDER FOR REAL TRADE {signal_id} ({symbol}). THE POSITION REMAINS OPEN. MANUAL INTERVENTION REQUIRED.")
                    log_and_notify('critical', f"CRITICAL: FAILED TO SELL {symbol} for signal {signal_id}. MANUAL ACTION NEEDED.", "REAL_TRADE_ERROR")
            else:
                logger.warning(f"⚠️ [{symbol}] No sellable balance ({actual_free_balance}) found for asset {base_asset}. Closing signal virtually.")
        
        except Exception as e:
            logger.critical(f"🚨 CRITICAL: An exception occurred while preparing the sell order for {symbol}: {e}", exc_info=True)
            log_and_notify('critical', f"CRITICAL: FAILED TO PREPARE SELL for {symbol} due to error: {e}. MANUAL ACTION NEEDED.", "REAL_TRADE_ERROR")
            with signal_cache_lock:
                if symbol not in open_signals_cache:
                    open_signals_cache[symbol] = signal
            with closure_lock:
                signals_pending_closure.discard(signal_id)
            return
    elif is_real and not is_enabled:
        logger.warning(f"⚠️ [{symbol}] Real trade signal {signal_id} triggered closure, but master trading switch is OFF. Closing virtually.")

    try:
        if not check_db_connection() or not conn: raise OperationalError("DB connection failed.")
        db_closing_price = float(closing_price); entry_price = float(signal['entry_price'])
        profit_pct = ((db_closing_price / entry_price) - 1) * 100
        with conn.cursor() as cur:
            cur.execute("UPDATE signals SET status = %s, closing_price = %s, closed_at = NOW(), profit_percentage = %s WHERE id = %s AND status IN ('open', 'updated');",
                        (status, db_closing_price, profit_pct, signal_id))
            if cur.rowcount == 0: logger.warning(f"⚠️ [DB Close] Signal {signal_id} was already closed or not found."); return
        conn.commit()
        
        status_map = {'target_hit': '✅ تحقق الهدف', 'stop_loss_hit': '🛑 ضرب وقف الخسارة', 'manual_close': '🖐️ إغلاق يدوي'}
        status_message = status_map.get(status, status)
        real_trade_tag = "🔥 REAL" if is_real else "👻 VIRTUAL"
        
        alert_msg = (f"*{status_message} ({real_trade_tag})*\n*العملة:* `{symbol}`\n*الربح:* `{profit_pct:+.2f}%`")
        send_telegram_message(CHAT_ID, alert_msg)
        log_and_notify('info', f"{status_message}: {symbol} | Profit: {profit_pct:+.2f}% | Real: {is_real}", 'CLOSE_SIGNAL')
        logger.info(f"✅ [DB Close] Signal {signal_id} closed successfully in DB.")
    except Exception as e:
        logger.error(f"❌ [DB Close] Critical error closing signal {signal_id} in DB: {e}", exc_info=True)
        if conn: conn.rollback()
        if symbol:
            with signal_cache_lock:
                if symbol not in open_signals_cache: open_signals_cache[symbol] = signal
    finally:
        with closure_lock: signals_pending_closure.discard(signal_id)

def load_open_signals_to_cache():
    # ... الكود يبقى كما هو ...
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
    # ... الكود يبقى كما هو ...
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
    # ... الكود يبقى كما هو ...
    btc_data = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
    if btc_data is not None: btc_data['btc_returns'] = btc_data['close'].pct_change()
    return btc_data

def perform_end_of_cycle_cleanup():
    # ... الكود يبقى كما هو ...
    logger.info("🧹 [Cleanup] Starting end-of-cycle cleanup...")
    try:
        if redis_client:
            deleted_keys = redis_client.delete(REDIS_PRICES_HASH_NAME)
            logger.info(f"🧹 [Cleanup] Cleared Redis price cache '{REDIS_PRICES_HASH_NAME}'. Keys deleted: {deleted_keys}.")
        
        model_cache_size = len(ml_models_cache)
        ml_models_cache.clear()
        logger.info(f"🧹 [Cleanup] Cleared {model_cache_size} ML models from in-memory cache.")

        collected = gc.collect()
        logger.info(f"🧹 [Cleanup] Garbage collector ran. Collected {collected} objects.")
        
        logger.info("✅ [Cleanup] End-of-cycle cleanup finished successfully.")

    except Exception as e:
        logger.error(f"❌ [Cleanup] An error occurred during cleanup: {e}", exc_info=True)

# ---------------------- [معدل] حلقة العمل الرئيسية ----------------------
def main_loop():
    logger.info("[Main Loop] Waiting for initialization...")
    time.sleep(15)
    if not validated_symbols_to_scan:
        log_and_notify("critical", "No validated symbols to scan. Bot will not start.", "SYSTEM")
        return
    log_and_notify("info", f"✅ Starting main scan loop for {len(validated_symbols_to_scan)} symbols with Advanced V3 model.", "SYSTEM")

    batch_size = 50 

    while True:
        try:
            logger.info("🌀 Starting new main cycle...")
            
            # [جديد] جلب بيانات الخوف والطمع مرة واحدة في بداية كل دورة
            fetch_fear_and_greed_data()

            symbols_to_process = list(validated_symbols_to_scan)
            
            for i in range(0, len(symbols_to_process), batch_size):
                symbol_batch = symbols_to_process[i:i + batch_size]
                logger.info(f"🔹 [Batch Processing] Starting batch {i//batch_size + 1}, symbols {i+1}-{i+len(symbol_batch)} of {len(symbols_to_process)}")

                determine_market_state()
                with market_state_lock:
                    market_regime = current_market_state.get("overall_regime", "UNCERTAIN")

                if USE_BTC_TREND_FILTER and market_regime in ["DOWNTREND", "STRONG DOWNTREND"]:
                    log_rejection("ALL", "BTC Trend Filter", {"detail": f"Scan paused for this batch due to market regime: {market_regime}"})
                else:
                    btc_data = get_btc_data_for_bot()

                    for symbol in symbol_batch:
                        try:
                            with signal_cache_lock:
                                open_trade = open_signals_cache.get(symbol)
                                open_trade_count = len(open_signals_cache)

                            strategy = TradingStrategy(symbol)
                            if not all([strategy.ml_model, strategy.scaler, strategy.feature_names]):
                                continue

                            df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                            if df_15m is None or df_15m.empty: continue
                            df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                            if df_4h is None or df_4h.empty: continue
                            
                            # [معدل] استدعاء الدالة الجديدة لتجميع كل الميزات
                            df_features = strategy.get_features(df_15m, df_4h, btc_data)
                            if df_features is None or df_features.empty: continue
                            
                            signal_info = strategy.generate_signal(df_features)
                            if not signal_info: continue
                            
                            prediction, confidence = signal_info['prediction'], signal_info['confidence']
                            
                            if prediction == 1 and confidence >= BUY_CONFIDENCE_THRESHOLD:
                                last_features = df_features.iloc[-1]
                                last_features.name = symbol
                                
                                try:
                                    entry_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                                except Exception as e:
                                    logger.error(f"❌ [{symbol}] Could not fetch fresh entry price via API: {e}. Skipping signal.")
                                    continue

                                if open_trade:
                                    old_confidence_raw = open_trade.get('signal_details', {}).get('ML_Confidence', 0.0)
                                    try:
                                        old_confidence = float(str(old_confidence_raw).strip().replace('%', '')) / 100.0 if isinstance(old_confidence_raw, str) else float(old_confidence_raw)
                                    except (ValueError, TypeError): old_confidence = 0.0
                                    
                                    if confidence > old_confidence + MIN_CONFIDENCE_INCREASE_FOR_UPDATE:
                                        logger.info(f"✅ [{symbol}] Reinforcement condition met. Old: {old_confidence:.2%}, New: {confidence:.2%}. Evaluating update...")
                                        if USE_SPEED_FILTER and not passes_speed_filter(last_features): continue
                                        if USE_MOMENTUM_FILTER and not passes_momentum_filter(last_features): continue
                                        
                                        last_atr = last_features.get('atr', 0)
                                        tp_sl_data = calculate_tp_sl(symbol, entry_price, last_atr)
                                        if not tp_sl_data: continue

                                        updated_signal_data = {
                                            'symbol': symbol, 'target_price': tp_sl_data['target_price'], 'stop_loss': tp_sl_data['stop_loss'],
                                            'signal_details': { 'ML_Confidence': confidence, 'ML_Confidence_Display': f"{confidence:.2%}", 'Update_Reason': 'Reinforcement Signal' }
                                        }
                                        
                                        if update_signal_in_db(open_trade['id'], updated_signal_data):
                                            with signal_cache_lock:
                                                open_signals_cache[symbol].update(updated_signal_data)
                                                open_signals_cache[symbol]['status'] = 'updated'
                                            send_trade_update_alert(updated_signal_data, open_trade)
                                    continue

                                if open_trade_count >= MAX_OPEN_TRADES:
                                    log_rejection(symbol, "Max Open Trades", {"count": open_trade_count, "max": MAX_OPEN_TRADES}); continue

                                if USE_SPEED_FILTER and not passes_speed_filter(last_features): continue
                                if USE_MOMENTUM_FILTER and not passes_momentum_filter(last_features): continue
                                
                                last_atr = last_features.get('atr', 0)
                                volatility = (last_atr / entry_price * 100) if entry_price > 0 else 0
                                if USE_MIN_VOLATILITY_FILTER and volatility < MIN_VOLATILITY_PERCENT:
                                    log_rejection(symbol, "Low Volatility", {"volatility": f"{volatility:.2f}%", "min": f"{MIN_VOLATILITY_PERCENT}%"}); continue
                                
                                if USE_BTC_CORRELATION_FILTER and market_regime in ["UPTREND", "STRONG UPTREND"]:
                                    correlation = last_features.get('btc_correlation', 0)
                                    if correlation < MIN_BTC_CORRELATION:
                                        log_rejection(symbol, "BTC Correlation", {"corr": f"{correlation:.2f}", "min": f"{MIN_BTC_CORRELATION}"}); continue
                                
                                tp_sl_data = calculate_tp_sl(symbol, entry_price, last_atr)
                                if not tp_sl_data: continue
                                
                                new_signal = {
                                    'symbol': symbol, 'strategy_name': BASE_ML_MODEL_NAME, 
                                    'signal_details': {'ML_Confidence': confidence, 'ML_Confidence_Display': f"{confidence:.2%}"}, 
                                    'entry_price': entry_price, **tp_sl_data
                                }

                                if USE_RRR_FILTER:
                                    risk = entry_price - float(new_signal['stop_loss'])
                                    reward = float(new_signal['target_price']) - entry_price
                                    if risk <= 0 or reward <= 0 or (reward / risk) < MIN_RISK_REWARD_RATIO:
                                        log_rejection(symbol, "RRR Filter", {"rrr": f"{(reward/risk):.2f}" if risk > 0 else "N/A"}); continue
                                
                                with trading_status_lock:
                                    is_enabled = is_trading_enabled

                                if is_enabled:
                                    logger.info(f"🔥 [{symbol}] Real trading is ENABLED. Calculating position size...")
                                    quantity = calculate_position_size(symbol, entry_price, new_signal['stop_loss'])
                                    if quantity and quantity > 0:
                                        order_result = place_order(symbol, Client.SIDE_BUY, quantity)
                                        if order_result:
                                            actual_entry_price = float(order_result['fills'][0]['price']) if order_result.get('fills') else entry_price
                                            new_signal['entry_price'] = actual_entry_price
                                            new_signal['is_real_trade'] = True
                                            new_signal['quantity'] = float(order_result['executedQty'])
                                            new_signal['order_id'] = order_result['orderId']
                                        else:
                                            logger.error(f"[{symbol}] Failed to place real order. Skipping signal.")
                                            continue
                                    else:
                                        logger.warning(f"[{symbol}] Could not calculate a valid position size. Skipping real trade.")
                                        continue
                                else:
                                    logger.info(f"👻 [{symbol}] Real trading is DISABLED. Logging as a virtual signal.")
                                    new_signal['is_real_trade'] = False

                                saved_signal = insert_signal_into_db(new_signal)
                                if saved_signal:
                                    with signal_cache_lock:
                                        open_signals_cache[saved_signal['symbol']] = saved_signal
                                    send_new_signal_alert(saved_signal)
                            
                            time.sleep(2)
                        except Exception as e: 
                            logger.error(f"❌ [Processing Error] {symbol}: {e}", exc_info=True)
                
                logger.info(f"🧹 [Batch Cleanup] Finished batch {i//batch_size + 1}. Performing cleanup to free memory.")
                perform_end_of_cycle_cleanup()
                logger.info(f"⏸️ [Batch Cooldown] Waiting for 10 seconds before next batch...")
                time.sleep(10)

            logger.info("✅ [End of Cycle] Full scan cycle finished.")
            logger.info(f"⏳ [End of Cycle] Waiting for 300 seconds before next full cycle...")
            time.sleep(300)

        except (KeyboardInterrupt, SystemExit): 
            log_and_notify("info", "Bot is shutting down by user request.", "SYSTEM")
            break
        except Exception as main_err: 
            log_and_notify("error", f"Critical error in main loop: {main_err}", "SYSTEM")
            time.sleep(120)

# ---------------------- واجهة برمجة تطبيقات Flask (تبقى كما هي) ----------------------
app = Flask(__name__)
CORS(app)

def get_fear_and_greed_index() -> Dict[str, Any]:
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10).json()
        return {"value": int(response['data'][0]['value']), "classification": response['data'][0]['value_classification']}
    except Exception as e:
        logger.warning(f"⚠️ [F&G Index] Could not fetch Fear & Greed index: {e}")
        return {"value": -1, "classification": "Error"}

def check_api_status() -> bool:
    if not client: return False
    try: client.ping(); return True
    except Exception: return False

def get_usdt_balance() -> Optional[float]:
    if not client: return None
    try:
        balance = client.get_asset_balance(asset='USDT')
        return float(balance['free'])
    except Exception as e:
        logger.error(f"❌ Could not fetch USDT balance: {e}")
        return None

@app.route('/')
def home():
    return render_template_string(get_dashboard_html())

@app.route('/api/market_status')
def get_market_status():
    with market_state_lock: state_copy = dict(current_market_state)
    return jsonify({
        "fear_and_greed": get_fear_and_greed_index(), "market_state": state_copy,
        "db_ok": check_db_connection(), "api_ok": check_api_status(),
        "usdt_balance": get_usdt_balance()
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
        win_rate = 0.0
        profit_factor_val = 0.0
        avg_win = 0.0
        avg_loss = 0.0

        if closed_trades:
            total_net_profit_usdt = sum(
                (((float(t['profit_percentage']) - (2 * TRADING_FEE_PERCENT)) / 100) * (float(t['quantity']) * float(t['entry_price']) if t.get('is_real_trade') and t.get('quantity') and t.get('entry_price') else STATS_TRADE_SIZE_USDT))
                for t in closed_trades if t.get('profit_percentage') is not None
            )
            
            wins_list = [float(s['profit_percentage']) for s in closed_trades if s.get('profit_percentage') is not None and float(s['profit_percentage']) > 0]
            losses_list = [float(s['profit_percentage']) for s in closed_trades if s.get('profit_percentage') is not None and float(s['profit_percentage']) < 0]
            
            win_rate = (len(wins_list) / len(closed_trades) * 100) if closed_trades else 0.0
            avg_win = sum(wins_list) / len(wins_list) if wins_list else 0.0
            avg_loss = sum(losses_list) / len(losses_list) if losses_list else 0.0
            
            total_profit_from_wins = sum(wins_list)
            total_loss_from_losses = abs(sum(losses_list))
            
            if total_loss_from_losses > 0:
                profit_factor_val = total_profit_from_wins / total_loss_from_losses
            elif total_profit_from_wins > 0:
                profit_factor_val = "Infinity"
        
        return jsonify({
            "open_trades_count": open_trades_count,
            "net_profit_usdt": total_net_profit_usdt,
            "win_rate": win_rate,
            "profit_factor": profit_factor_val,
            "total_closed_trades": len(closed_trades),
            "average_win_pct": avg_win,
            "average_loss_pct": avg_loss
        })
    except Exception as e:
        logger.error(f"❌ [API Stats] Critical error: {e}", exc_info=True)
        if conn: conn.rollback()
        return jsonify({"error": "An internal error occurred in stats"}), 500

@app.route('/api/profit_curve')
def get_profit_curve():
    if not check_db_connection(): 
        return jsonify({"error": "DB connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT closed_at, profit_percentage FROM signals 
                WHERE status NOT IN ('open', 'updated') AND profit_percentage IS NOT NULL AND closed_at IS NOT NULL 
                ORDER BY closed_at ASC;
            """)
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
    if not check_db_connection() or not redis_client: 
        return jsonify({"error": "Service connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals ORDER BY CASE WHEN status IN ('open', 'updated') THEN 0 ELSE 1 END, id DESC;")
            all_signals = [dict(s) for s in cur.fetchall()]
        
        open_signals_to_process = [s for s in all_signals if s['status'] in ('open', 'updated')]
        
        if open_signals_to_process:
            symbols = [s['symbol'] for s in open_signals_to_process]
            prices_from_redis_list = redis_client.hmget(REDIS_PRICES_HASH_NAME, symbols)
            redis_prices = {symbol: p for symbol, p in zip(symbols, prices_from_redis_list)}

            for s in all_signals:
                if s['status'] in ('open', 'updated'):
                    symbol = s['symbol']
                    price = None
                    s['current_price'] = None
                    s['pnl_pct'] = None
                    if redis_prices.get(symbol):
                        try: price = float(redis_prices[symbol])
                        except (ValueError, TypeError): price = None
                    if price is None and client:
                        try: price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                        except Exception as e: price = None
                    s['current_price'] = price
                    if price and s.get('entry_price'):
                        s['pnl_pct'] = ((price / float(s['entry_price'])) - 1) * 100
        return jsonify(all_signals)
    except Exception as e:
        logger.error(f"❌ [API Signals] Critical error in get_signals: {e}", exc_info=True)
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
            logger.error(f"❌ [API Close] Could not fetch price for {symbol}: {e}")
            return jsonify({"error": f"Could not fetch price for {symbol}"}), 500
        initiate_signal_closure(symbol, dict(signal_to_close), 'manual_close', price)
        return jsonify({"message": f"تم إرسال طلب إغلاق الصفقة {signal_id}..."})
    except Exception as e:
        logger.error(f"❌ [API Close] Error: {e}", exc_info=True)
        if conn: conn.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/api/trading/status', methods=['GET'])
def get_trading_status():
    with trading_status_lock:
        return jsonify({"is_enabled": is_trading_enabled})

@app.route('/api/trading/toggle', methods=['POST'])
def toggle_trading_status():
    global is_trading_enabled
    with trading_status_lock:
        is_trading_enabled = not is_trading_enabled
        status_msg = "ENABLED" if is_trading_enabled else "DISABLED"
        log_and_notify('warning', f"🚨 Real trading status has been manually changed to: {status_msg}", "TRADING_STATUS_CHANGE")
        return jsonify({"message": f"Trading status set to {status_msg}", "is_enabled": is_trading_enabled})

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
    # ... الكود يبقى كما هو ...
    if not client or not validated_symbols_to_scan:
        logger.error("❌ [WebSocket] Cannot start: Client or symbols not initialized.")
        return
    logger.info("📈 [WebSocket] Starting WebSocket Manager...")
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    streams = [f"{s.lower()}@miniTicker" for s in validated_symbols_to_scan]
    twm.start_multiplex_socket(callback=handle_price_update_message, streams=streams)
    logger.info(f"✅ [WebSocket] Subscribed to {len(streams)} price streams.")
    twm.join()

def initialize_bot_services():
    # ... الكود يبقى كما هو ...
    global client, validated_symbols_to_scan
    logger.info("🤖 [Bot Services] Starting background initialization...")
    try:
        client = Client(API_KEY, API_SECRET)
        init_db()
        init_redis()
        get_exchange_info_map()
        load_open_signals_to_cache()
        load_notifications_to_cache()
        Thread(target=determine_market_state, daemon=True).start()
        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan:
            logger.critical("❌ No validated symbols to scan. Bot will not start."); return
        Thread(target=run_websocket_manager, daemon=True).start()
        Thread(target=trade_monitoring_loop, daemon=True).start()
        Thread(target=main_loop, daemon=True).start()
        logger.info("✅ [Bot Services] All background services started successfully.")
    except Exception as e:
        log_and_notify("critical", f"A critical error occurred during initialization: {e}", "SYSTEM")
        exit(1)

if __name__ == "__main__":
    logger.info("🚀 LAUNCHING TRADING BOT & DASHBOARD (V3 - Advanced Model) 🚀")
    initialization_thread = Thread(target=initialize_bot_services, daemon=True)
    initialization_thread.start()
    run_flask()
    logger.info("👋 [Shutdown] Application has been shut down."); os._exit(0)
