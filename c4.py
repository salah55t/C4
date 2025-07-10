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
import gc
from decimal import Decimal, ROUND_DOWN

# --- تجاهل التحذيرات غير الهامة ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v17_real.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV17_RealTrading')

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

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
# --- START: REAL TRADING CONFIGURATION ---
# 🔴 !! تحذير !! 🔴
# تغيير هذا المتغير إلى True سيمكن البوت من إجراء عمليات تداول حقيقية بأموال حقيقية.
# لا تقم بتفعيله إلا إذا كنت تفهم الكود تمامًا وتقبل المخاطر.
ENABLE_REAL_TRADING: bool = False # 👈 *** مفتاح التداول الحقيقي ***

# النسبة المئوية من رصيد USDT لاستخدامها في كل صفقة
TRADE_BALANCE_PERCENT: float = 2.0 # مثال: 2.0 تعني استخدام 2% من الرصيد

# --- END: REAL TRADING CONFIGURATION ---

BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V8_With_Momentum'
MODEL_FOLDER: str = 'V8'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 30
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices_v8"
MODEL_BATCH_SIZE: int = 5
DIRECT_API_CHECK_INTERVAL: int = 10
TRADING_FEE_PERCENT: float = 0.1
HYPOTHETICAL_TRADE_SIZE_USDT: float = 100.0

# --- مؤشرات فنية ---
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
EMA_FAST_PERIOD: int = 50
EMA_SLOW_PERIOD: int = 200
REL_VOL_PERIOD: int = 30
MOMENTUM_PERIOD: int = 12
EMA_SLOPE_PERIOD: int = 5


# --- إدارة الصفقات ---
MAX_OPEN_TRADES: int = 10
BUY_CONFIDENCE_THRESHOLD = 0.65
SELL_CONFIDENCE_THRESHOLD = 0.70
MIN_PROFIT_FOR_SELL_CLOSE_PERCENT = 0.2
MIN_CONFIDENCE_INCREASE_FOR_UPDATE = 0.05

# --- إعدادات الهدف ووقف الخسارة ---
ATR_FALLBACK_SL_MULTIPLIER: float = 1.5
ATR_FALLBACK_TP_MULTIPLIER: float = 2.0
SL_BUFFER_ATR_PERCENT: float = 0.25

# --- إعدادات وقف الخسارة المتحرك (Trailing Stop-Loss) ---
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_ACTIVATION_PROFIT_PERCENT: float = 1.0
TRAILING_DISTANCE_PERCENT: float = 0.8
LAST_PEAK_UPDATE_TIME: Dict[int, float] = {}
PEAK_UPDATE_COOLDOWN: int = 60

# --- إعدادات الفلاتر المحسّنة ---
USE_BTC_TREND_FILTER: bool = True
BTC_SYMBOL: str = 'BTCUSDT'
BTC_TREND_TIMEFRAME: str = '4h'
BTC_TREND_EMA_PERIOD: int = 50
USE_SPEED_FILTER: bool = True
USE_RRR_FILTER: bool = True
MIN_RISK_REWARD_RATIO: float = 1.1
USE_BTC_CORRELATION_FILTER: bool = True
MIN_BTC_CORRELATION: float = 0.1
USE_MIN_VOLATILITY_FILTER: bool = True
MIN_VOLATILITY_PERCENT: float = 0.3
USE_MOMENTUM_FILTER: bool = True


# --- المتغيرات العامة وقفل العمليات ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
ml_models_cache: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
notifications_cache = deque(maxlen=50)
notifications_lock = Lock()
signals_pending_closure: Set[int] = set()
closure_lock = Lock()
last_api_check_time = time.time()
rejection_logs_cache = deque(maxlen=100)
rejection_logs_lock = Lock()
last_market_state_check = 0
current_market_state: Dict[str, Any] = {
    "overall_regime": "INITIALIZING",
    "details": {},
    "last_updated": None
}
market_state_lock = Lock()
# --- Real Trading Cache ---
exchange_info_cache: Dict[str, Any] = {}
exchange_info_lock = Lock()


# ---------------------- دالة HTML للوحة التحكم (تم الإصلاح) ----------------------
def get_dashboard_html():
    """
    لوحة تحكم محسنة مع شارت أرباح بتصميم الشموع اليابانية (شلال) وتصميم متجاوب.
    """
    # ... (الكود الخاص بواجهة HTML لم يتغير)
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم بوت التداول V8</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon@3.4.4/build/global/luxon.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.3.1/dist/chartjs-adapter-luxon.umd.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;700;900&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #111827; --bg-card: #1F2937; --border-color: #374151;
            --text-primary: #F9FAFB; --text-secondary: #9CA3AF;
            --accent-blue: #3B82F6; --accent-green: #22C55E; --accent-red: #EF4444; --accent-yellow: #EAB308;
        }
        body { font-family: 'Cairo', sans-serif; background-color: var(--bg-dark); color: var(--text-primary); }
        .card { background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 0.75rem; transition: all 0.3s ease-in-out; padding: 1rem; }
        .card:hover { transform: translateY(-4px); box-shadow: 0 8px 25px rgba(0,0,0,0.2); border-color: var(--accent-blue); }
        .skeleton { animation: pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite; background-color: #374151; border-radius: 0.5rem; }
        @keyframes pulse { 50% { opacity: .5; } }
        .progress-bar-container { position: relative; width: 100%; height: 1.25rem; background-color: #374151; border-radius: 0.5rem; overflow: hidden; display: flex; align-items: center; }
        .progress-bar { height: 100%; transition: width 0.5s ease-in-out; }
        .progress-point { position: absolute; top: 50%; transform: translateY(-50%); width: 8px; height: 8px; border-radius: 50%; border: 2px solid white; }
        .entry-point { background-color: var(--accent-blue); }
        .current-point { background-color: var(--accent-yellow); }
        .progress-labels { display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--text-secondary); padding: 0 2px; margin-top: 2px; }
        #needle { transition: transform 1s cubic-bezier(0.68, -0.55, 0.27, 1.55); }
        .tab-btn.active { border-bottom-color: var(--accent-blue); color: var(--text-primary); }
    </style>
</head>
<body class="p-4 md:p-6">
    <div class="container mx-auto max-w-7xl">
        <header class="mb-6 flex flex-wrap justify-between items-center gap-4">
            <h1 class="text-3xl md:text-4xl font-black text-white">لوحة تحكم التداول</h1>
            <div id="connection-status" class="flex items-center gap-2 text-sm">
                <div id="db-status-light" class="w-3 h-3 rounded-full bg-gray-500 animate-pulse"></div><span class="text-text-secondary">DB</span>
                <div id="api-status-light" class="w-3 h-3 rounded-full bg-gray-500 animate-pulse"></div><span class="text-text-secondary">API</span>
            </div>
        </header>

        <section class="mb-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
            <div class="card lg:col-span-2">
                <h3 class="font-bold mb-3 text-lg">اتجاه السوق (BTC)</h3>
                <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
                    <div>
                        <h4 class="text-sm font-semibold text-text-secondary">الاتجاه العام</h4>
                        <div id="overall-regime" class="text-2xl font-bold skeleton h-8 w-3/4 mx-auto mt-1"></div>
                    </div>
                    <div>
                        <h4 class="text-sm font-semibold text-text-secondary">إطار 15 دقيقة</h4>
                        <div id="tf-15m-status" class="text-xl font-bold skeleton h-7 w-2/3 mx-auto mt-1"></div>
                        <div id="tf-15m-details" class="text-xs text-text-secondary skeleton h-4 w-1/2 mx-auto mt-1"></div>
                    </div>
                    <div>
                        <h4 class="text-sm font-semibold text-text-secondary">إطار ساعة</h4>
                        <div id="tf-1h-status" class="text-xl font-bold skeleton h-7 w-2/3 mx-auto mt-1"></div>
                        <div id="tf-1h-details" class="text-xs text-text-secondary skeleton h-4 w-1/2 mx-auto mt-1"></div>
                    </div>
                    <div>
                        <h4 class="text-sm font-semibold text-text-secondary">إطار 4 ساعات</h4>
                        <div id="tf-4h-status" class="text-xl font-bold skeleton h-7 w-2/3 mx-auto mt-1"></div>
                        <div id="tf-4h-details" class="text-xs text-text-secondary skeleton h-4 w-1/2 mx-auto mt-1"></div>
                    </div>
                </div>
            </div>
            <div class="card flex flex-col justify-center items-center">
                 <h3 class="font-bold mb-2 text-lg">مؤشر الخوف والطمع</h3>
                 <div id="fear-greed-gauge" class="relative w-full max-w-[180px] aspect-square"></div>
                 <div id="fear-greed-value" class="text-3xl font-bold mt-[-25px] skeleton h-10 w-1/2"></div>
                 <div id="fear-greed-text" class="text-md text-text-secondary skeleton h-6 w-3/4 mt-1"></div>
            </div>
            <div class="card flex flex-col justify-center items-center text-center">
                <h3 class="font-bold text-text-secondary text-lg">صفقات مفتوحة</h3>
                <div id="open-trades-value" class="text-5xl font-black text-accent-blue mt-2 skeleton h-12 w-1/2"></div>
            </div>
        </section>

        <section class="mb-6 grid grid-cols-1 lg:grid-cols-3 gap-5">
            <div id="profit-chart-card" class="card lg:col-span-2">
                <h3 class="font-bold mb-3">أداء الصفقات (الربح التراكمي %)</h3>
                <div class="relative h-80 md:h-96">
                    <canvas id="profitChart"></canvas>
                </div>
            </div>
            <div id="other-stats-container" class="grid grid-cols-1 sm:grid-cols-3 lg:grid-cols-1 gap-4">
                <div class="card text-center flex flex-col justify-center">
                    <div class="text-sm text-text-secondary mb-1">صافي الربح (USDT)</div>
                    <div id="net-profit-usdt" class="text-2xl font-bold skeleton h-8 w-3/4 mx-auto"></div>
                </div>
                <div class="card text-center flex flex-col justify-center">
                    <div class="text-sm text-text-secondary mb-1">نسبة النجاح</div>
                    <div id="win-rate" class="text-2xl font-bold skeleton h-8 w-1/2 mx-auto"></div>
                </div>
                <div class="card text-center flex flex-col justify-center">
                    <div class="text-sm text-text-secondary mb-1">عامل الربح</div>
                    <div id="profit-factor" class="text-2xl font-bold skeleton h-8 w-1/2 mx-auto"></div>
                </div>
            </div>
        </section>

        <div class="mb-4 border-b border-border-color">
            <nav class="flex space-x-4 -mb-px" aria-label="Tabs">
                <button onclick="showTab('signals', this)" class="tab-btn active text-white border-b-2 py-3 px-4 font-semibold">الصفقات</button>
                <button onclick="showTab('notifications', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-4">الإشعارات</button>
                <button onclick="showTab('rejections', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-4">الصفقات المرفوضة</button>
            </nav>
        </div>

        <main>
            <div id="signals-tab" class="tab-content"><div class="overflow-x-auto card p-0"><table class="min-w-full text-sm text-right"><thead class="border-b border-border-color"><tr><th class="p-4 font-semibold">العملة</th><th class="p-4 font-semibold">الحالة</th><th class="p-4 font-semibold">الربح/الخسارة</th><th class="p-4 font-semibold w-[35%]">التقدم نحو الهدف</th><th class="p-4 font-semibold">الدخول / الحالي</th><th class="p-4 font-semibold">إجراء</th></tr></thead><tbody id="signals-table"></tbody></table></div></div>
            <div id="notifications-tab" class="tab-content hidden"><div id="notifications-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-2"></div></div>
            <div id="rejections-tab" class="tab-content hidden"><div id="rejections-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-2"></div></div>
        </main>
    </div>

<script>
let profitChartInstance;
const REGIME_STYLES = {
    "STRONG UPTREND": { text: "صاعد قوي", color: "text-accent-green" }, "UPTREND": { text: "صاعد", color: "text-green-400" },
    "RANGING": { text: "عرضي", color: "text-accent-yellow" }, "DOWNTREND": { text: "هابط", color: "text-red-400" },
    "STRONG DOWNTREND": { text: "هابط قوي", color: "text-accent-red" }, "UNCERTAIN": { text: "غير واضح", color: "text-text-secondary" },
    "INITIALIZING": { text: "تهيئة...", color: "text-accent-blue" }
};
const TF_STATUS_STYLES = {
    "Uptrend": { text: "صاعد", icon: "▲", color: "text-accent-green" }, "Downtrend": { text: "هابط", icon: "▼", color: "text-accent-red" },
    "Ranging": { text: "عرضي", icon: "↔", color: "text-accent-yellow" }, "Uncertain": { text: "غير واضح", icon: "?", color: "text-text-secondary" }
};

function formatNumber(num, digits = 2) {
    if (num === null || num === undefined || isNaN(num)) return 'N/A';
    return num.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function showTab(tabName, element) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.add('hidden'));
    document.getElementById(`${tabName}-tab`).classList.remove('hidden');
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active', 'text-white', 'border-accent-blue', 'font-semibold');
        btn.classList.add('text-text-secondary', 'border-transparent');
    });
    element.classList.add('active', 'text-white', 'border-accent-blue', 'font-semibold');
    element.classList.remove('text-text-secondary', 'border-transparent');
}

async function apiFetch(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) { 
            console.error(`API Error ${response.status}`); 
            try {
                return await response.json(); // Try to get error message from body
            } catch (e) {
                return { error: `HTTP Error ${response.status}` };
            }
        }
        return await response.json();
    } catch (error) { 
        console.error(`Fetch error for ${url}:`, error); 
        return { error: "Network or fetch error" };
    }
}

function getFngColor(value) {
    if (value < 25) return '#EF4444'; if (value < 45) return '#F97316';
    if (value < 55) return '#EAB308'; if (value < 75) return '#84CC16';
    return '#22C55E';
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
    container.innerHTML = `<svg viewBox="0 0 100 57" class="w-full h-full"><defs><linearGradient id="g"><stop offset="0%" stop-color="#EF4444"/><stop offset="50%" stop-color="#EAB308"/><stop offset="100%" stop-color="#22C55E"/></linearGradient></defs><path d="M10 50 A 40 40 0 0 1 90 50" stroke="url(#g)" stroke-width="10" fill="none" stroke-linecap="round"/><g transform="rotate(${angle} 50 50)"><path d="M50 45 L 47 15 Q 50 10 53 15 L 50 45" fill="${color}" id="needle"/></g><circle cx="50" cy="50" r="4" fill="${color}"/></svg>`;
}

function updateMarketStatus() {
    apiFetch('/api/market_status').then(data => {
        if (!data || data.error) return;
        document.getElementById('db-status-light').className = `w-3 h-3 rounded-full ${data.db_ok ? 'bg-green-500' : 'bg-red-500'}`;
        document.getElementById('api-status-light').className = `w-3 h-3 rounded-full ${data.api_ok ? 'bg-green-500' : 'bg-red-500'}`;
        
        const state = data.market_state;
        const overallRegime = state.overall_regime || "UNCERTAIN";
        const regimeStyle = REGIME_STYLES[overallRegime.toUpperCase()] || REGIME_STYLES["UNCERTAIN"];
        const overallDiv = document.getElementById('overall-regime');
        overallDiv.textContent = regimeStyle.text;
        overallDiv.className = `text-2xl font-bold ${regimeStyle.color}`;
        overallDiv.classList.remove('skeleton', 'h-8', 'w-3/4', 'mx-auto', 'mt-1');

        ['15m', '1h', '4h'].forEach(tf => {
            const tfData = state.details[tf];
            const statusDiv = document.getElementById(`tf-${tf}-status`);
            const detailsDiv = document.getElementById(`tf-${tf}-details`);
            [statusDiv, detailsDiv].forEach(el => el.classList.remove('skeleton', 'h-7', 'w-2/3', 'h-4', 'w-1/2', 'mx-auto', 'mt-1'));
            if (tfData) {
                const style = TF_STATUS_STYLES[tfData.trend] || TF_STATUS_STYLES["Uncertain"];
                statusDiv.innerHTML = `<span class="${style.color}">${style.icon} ${style.text}</span>`;
                detailsDiv.textContent = `RSI: ${formatNumber(tfData.rsi, 1)} | ADX: ${formatNumber(tfData.adx, 1)}`;
            } else {
                statusDiv.textContent = 'N/A'; detailsDiv.textContent = '';
            }
        });
        renderFearGreedGauge(data.fear_and_greed.value, data.fear_and_greed.classification);
    });
}

function updateStats() {
    apiFetch('/api/stats').then(data => {
        const stat_ids = ['open-trades-value', 'net-profit-usdt', 'win-rate', 'profit-factor'];
        if (!data || data.error) {
            console.error("Failed to fetch stats:", data ? data.error : "No data");
            stat_ids.forEach(id => {
                const el = document.getElementById(id);
                if (el) {
                    el.textContent = 'خطأ';
                    el.classList.remove('skeleton');
                }
            });
            return;
        }

        const profitFactorDisplay = data.profit_factor === 'Infinity' ? '∞' : formatNumber(data.profit_factor);

        const fields = {
            'open-trades-value': formatNumber(data.open_trades_count, 0),
            'net-profit-usdt': `$${formatNumber(data.net_profit_usdt)}`,
            'win-rate': `${formatNumber(data.win_rate)}%`,
            'profit-factor': profitFactorDisplay
        };

        for (const [id, value] of Object.entries(fields)) {
            const el = document.getElementById(id);
            if (el) {
                el.textContent = value;
                el.classList.remove('skeleton', 'h-12', 'h-8', 'w-1/2', 'w-3/4', 'mx-auto');
                if (id === 'net-profit-usdt') {
                    el.className = `text-2xl font-bold ${data.net_profit_usdt >= 0 ? 'text-accent-green' : 'text-accent-red'}`;
                }
            }
        }
    });
}

function updateProfitChart() {
    const chartCard = document.getElementById('profit-chart-card');
    const canvas = document.getElementById('profitChart');

    apiFetch('/api/profit_curve').then(data => {
        const existingMsg = chartCard.querySelector('.no-data-msg, .error-msg');
        if(existingMsg) existingMsg.remove();

        if (!data || data.error) { 
            canvas.style.display = 'none'; 
            chartCard.insertAdjacentHTML('beforeend', '<p class="error-msg text-center text-text-secondary mt-8">حدث خطأ أثناء تحميل بيانات الرسم البياني.</p>');
            return; 
        }
        if (data.length <= 1) { 
            canvas.style.display = 'none'; 
            chartCard.insertAdjacentHTML('beforeend', '<p class="no-data-msg text-center text-text-secondary mt-8">لا توجد صفقات كافية لعرض الرسم البياني.</p>');
            return;
        }
        
        canvas.style.display = 'block';
        const ctx = canvas.getContext('2d');
        const labels = data.map((d, i) => i > 0 ? `صفقة ${i}` : 'البداية');
        const chartData = data.map(d => d.profit_range);
        
        const config = {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'الربح/الخسارة للصفقة',
                    data: chartData,
                    backgroundColor: (ctx) => {
                        if (ctx.raw === null || ctx.raw === undefined) return 'var(--border-color)';
                        const [start, end] = ctx.raw;
                        return end >= start ? 'rgba(34, 197, 94, 0.7)' : 'rgba(239, 68, 68, 0.7)';
                    },
                    borderColor: (ctx) => {
                        if (ctx.raw === null || ctx.raw === undefined) return 'var(--border-color)';
                        const [start, end] = ctx.raw;
                        return end >= start ? 'var(--accent-green)' : 'var(--accent-red)';
                    },
                    borderWidth: 1,
                    barPercentage: 0.8,
                    categoryPercentage: 0.9,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    x: { ticks: { display: false }, grid: { display: false } },
                    y: { beginAtZero: false, ticks: { color: 'var(--text-secondary)', callback: v => formatNumber(v) + '%' }, grid: { color: '#37415180' } }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        mode: 'index', intersect: false, backgroundColor: 'var(--bg-card)',
                        titleFont: { weight: 'bold' }, bodyFont: { family: 'Cairo' },
                        callbacks: {
                            title: (ctx) => ctx[0] ? ctx[0].label : '',
                            label: (ctx) => {
                                if (ctx.dataIndex === 0) return `البداية: ${formatNumber(ctx.raw[1])}%`;
                                const tradeData = data[ctx.dataIndex];
                                const profit = tradeData.profit_change;
                                const cumulative = tradeData.profit_range[1];
                                return [`ربح الصفقة: ${formatNumber(profit)}%`, `الربح التراكمي: ${formatNumber(cumulative)}%`];
                            }
                        }
                    }
                }
            }
        };

        if (profitChartInstance) {
            profitChartInstance.data.labels = labels;
            profitChartInstance.data.datasets[0].data = chartData;
            profitChartInstance.update('none');
        } else {
            profitChartInstance = new Chart(ctx, config);
        }
    });
}

function renderProgressBar(signal) {
    const { entry_price, stop_loss, target_price, current_price } = signal;
    if ([entry_price, stop_loss, target_price, current_price].some(v => v === null)) return '<span>لا تتوفر بيانات</span>';
    const [entry, sl, tp, current] = [entry_price, stop_loss, target_price, current_price].map(parseFloat);
    const totalDist = tp - sl;
    if (totalDist <= 0) return '<span>بيانات غير صالحة</span>';
    const progressPct = Math.max(0, Math.min(100, ((current - sl) / totalDist) * 100));
    const entryPointPct = Math.max(0, Math.min(100, ((entry - sl) / totalDist) * 100));
    return `<div class="flex flex-col w-full"><div class="progress-bar-container"><div class="progress-bar ${current >= entry ? 'bg-accent-green' : 'bg-accent-red'}" style="width: ${progressPct}%"></div><div class="progress-point entry-point" style="left: ${entryPointPct}%" title="الدخول: ${entry.toFixed(4)}"></div></div><div class="progress-labels"><span title="وقف الخسارة">${sl.toFixed(4)}</span><span title="الهدف">${tp.toFixed(4)}</span></div></div>`;
}

function updateSignals() {
    apiFetch('/api/signals').then(data => {
        const tableBody = document.getElementById('signals-table');
        if (!data || data.error) { tableBody.innerHTML = '<tr><td colspan="6" class="p-8 text-center">فشل تحميل الصفقات.</td></tr>'; return; }
        if (data.length === 0) { tableBody.innerHTML = '<tr><td colspan="6" class="p-8 text-center">لا توجد صفقات لعرضها.</td></tr>'; return; }
        tableBody.innerHTML = data.map(signal => {
            const pnlPct = signal.status === 'open' || signal.status === 'updated' ? (signal.pnl_pct || 0) : (signal.profit_percentage || 0);
            const statusClass = signal.status === 'open' ? 'text-yellow-400' : (signal.status === 'updated' ? 'text-blue-400' : 'text-gray-400');
            const statusText = signal.status === 'updated' ? 'تم تحديثها' : signal.status;
            return `<tr class="border-b border-border-color hover:bg-gray-800/50 transition-colors">
                    <td class="p-4 font-mono font-semibold">${signal.symbol}</td>
                    <td class="p-4 font-bold ${statusClass}">${statusText}</td>
                    <td class="p-4 font-mono font-bold ${pnlPct >= 0 ? 'text-accent-green' : 'text-accent-red'}">${formatNumber(pnlPct)}%</td>
                    <td class="p-4">${signal.status === 'open' || signal.status === 'updated' ? renderProgressBar(signal) : '-'}</td>
                    <td class="p-4 font-mono text-xs"><div>${formatNumber(signal.entry_price, 5)}</div><div class="text-text-secondary">${signal.current_price ? formatNumber(signal.current_price, 5) : 'N/A'}</div></td>
                    <td class="p-4">${signal.status === 'open' || signal.status === 'updated' ? `<button onclick="manualCloseSignal(${signal.id})" class="bg-red-600 hover:bg-red-700 text-white text-xs py-1 px-3 rounded-md">إغلاق</button>` : ''}</td>
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
    updateStats();
    updateProfitChart();
    updateSignals();
    const dateLocaleOptions = { timeZone: 'UTC', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false };
    const locale = 'fr-CA';
    updateList('/api/notifications', 'notifications-list', n => `<div class="p-3 rounded-md bg-gray-900/50 text-sm">[${new Date(n.timestamp).toLocaleString(locale, dateLocaleOptions)}] ${n.message}</div>`);
    updateList('/api/rejection_logs', 'rejections-list', log => `<div class="p-3 rounded-md bg-gray-900/50 text-sm">[${new Date(log.timestamp).toLocaleString(locale, dateLocaleOptions)}] <strong>${log.symbol}</strong>: ${log.reason} - <span class="font-mono text-xs text-text-secondary">${JSON.stringify(log.details)}</span></div>`);
}

setInterval(refreshData, 8000);
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
        separator = '&' if '?' in db_url_to_use else '?'
        db_url_to_use += f"{separator}sslmode=require"
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(db_url_to_use, connect_timeout=15, cursor_factory=RealDictCursor)
            conn.autocommit = False
            with conn.cursor() as cur:
                # --- تعديل جدول الصفقات لدعم التداول الحقيقي ---
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        entry_price DOUBLE PRECISION NOT NULL,
                        target_price DOUBLE PRECISION NOT NULL,
                        stop_loss DOUBLE PRECISION NOT NULL,
                        status TEXT DEFAULT 'open',
                        closing_price DOUBLE PRECISION,
                        closed_at TIMESTAMP,
                        profit_percentage DOUBLE PRECISION,
                        strategy_name TEXT,
                        signal_details JSONB,
                        current_peak_price DOUBLE PRECISION,
                        -- أعمدة جديدة للتداول الحقيقي
                        buy_order_id TEXT,
                        oco_order_id TEXT,
                        quantity DOUBLE PRECISION,
                        commission DOUBLE PRECISION
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
            logger.info("✅ [DB] Database connection successful and tables (re)initialized for real trading.")
            return
        except Exception as e:
            logger.error(f"❌ [DB] Connection error (Attempt {attempt + 1}/{retries}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: logger.critical("❌ [DB] Failed to connect to the database after multiple retries.")


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
        logger.error(f"❌ [DB] Connection lost: {e}. Attempting to reconnect...")
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
        logger.error(f"❌ [Notify DB] Failed to save notification to DB: {e}")
        if conn: conn.rollback()

def log_rejection(symbol: str, reason: str, details: Optional[Dict] = None):
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
    global redis_client
    logger.info("[Redis] Initializing Redis connection...")
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] Successfully connected to Redis server.")
    except redis.exceptions.ConnectionError as e:
        logger.critical(f"❌ [Redis] Failed to connect to Redis. Error: {e}")
        exit(1)

# ---------------------- دوال Binance والبيانات ----------------------
def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    if not client: return []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        exchange_info = client.get_exchange_info()
        active = {s['symbol'] for s in exchange_info['symbols'] if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'}
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ [Validation] Bot will monitor {len(validated)} validated symbols.")
        return validated
    except Exception as e:
        logger.error(f"❌ [Validation] An error occurred during symbol validation: {e}", exc_info=True)
        return []

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
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

# --- START: REAL TRADING HELPER FUNCTIONS ---

def get_exchange_info_for_symbol(symbol: str) -> Optional[Dict]:
    """
    Fetches and caches exchange information for a specific symbol.
    This is crucial for getting trading rules like LOT_SIZE and MIN_NOTIONAL.
    """
    with exchange_info_lock:
        if symbol in exchange_info_cache:
            return exchange_info_cache[symbol]
    if not client: return None
    try:
        info = client.get_exchange_info()
        symbol_info = next((s for s in info['symbols'] if s['symbol'] == symbol), None)
        if symbol_info:
            with exchange_info_lock:
                exchange_info_cache[symbol] = symbol_info
            return symbol_info
        return None
    except Exception as e:
        logger.error(f"❌ [Exchange Info] Could not fetch info for {symbol}: {e}")
        return None

def adjust_value_by_step_size(value: float, step_size: str) -> float:
    """
    Adjusts a value down to the nearest multiple of the step size.
    """
    decimal_value = Decimal(str(value))
    decimal_step = Decimal(str(step_size))
    adjusted = (decimal_value // decimal_step) * decimal_step
    return float(adjusted)

def get_usdt_balance() -> float:
    """
    Fetches the free (available) USDT balance from the Binance account.
    """
    if not client: return 0.0
    try:
        balance_info = client.get_asset_balance(asset='USDT')
        return float(balance_info['free'])
    except Exception as e:
        logger.error(f"❌ [Balance] Could not get USDT balance: {e}")
        return 0.0

# --- END: REAL TRADING HELPER FUNCTIONS ---

# ---------------------- دوال حساب الميزات وتحديد الاتجاه ----------------------
def calculate_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
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
            df_4h_features = df_4h_features.rename(columns=lambda c: f"{c}_4h" if c not in ['atr', 'volume'] else c, inplace=False)
            required_4h_cols = ['rsi_4h', 'price_vs_ema50_4h']
            df_featured = df_featured.join(df_4h_features[required_4h_cols], how='outer')
            df_featured.fillna(method='ffill', inplace=True)
            for col in self.feature_names:
                if col not in df_featured.columns: df_featured[col] = 0.0
            df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_featured.dropna(subset=self.feature_names)
        except Exception as e:
            logger.error(f"❌ [{self.symbol}] Feature engineering failed: {e}", exc_info=True)
            return None

    def generate_signal(self, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]) or df_features.empty: return None
        try:
            last_row_ordered_df = df_features.iloc[[-1]][self.feature_names]
            features_scaled_np = self.scaler.transform(last_row_ordered_df)
            features_scaled_df = pd.DataFrame(features_scaled_np, columns=self.feature_names)
            prediction = self.ml_model.predict(features_scaled_df)[0]
            prediction_proba = self.ml_model.predict_proba(features_scaled_df)
            confidence = float(np.max(prediction_proba[0]))
            logger.info(f"ℹ️ [{self.symbol}] Model predicted '{'BUY' if prediction == 1 else 'SELL/HOLD'}' with {confidence:.2%} confidence.")
            return {'prediction': int(prediction), 'confidence': confidence}
        except Exception as e:
            logger.warning(f"⚠️ [{self.symbol}] Signal Generation Error: {e}")
            return None

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
    if regime == "STRONG UPTREND": adx_threshold, rel_vol_threshold, rsi_min, rsi_max = (25.0, 0.6, 45.0, 85.0)
    elif regime == "UPTREND": adx_threshold, rel_vol_threshold, rsi_min, rsi_max = (22.0, 0.5, 40.0, 80.0)
    else: adx_threshold, rel_vol_threshold, rsi_min, rsi_max = (18.0, 0.2, 30.0, 80.0)
    adx, rel_vol, rsi = last_features.get('adx', 0), last_features.get('relative_volume', 0), last_features.get('rsi', 0)
    if (adx >= adx_threshold and rel_vol >= rel_vol_threshold and rsi_min <= rsi < rsi_max): return True
    log_rejection(symbol, "Speed Filter", {"Regime": regime, "ADX": f"{adx:.2f} (Req: >{adx_threshold})", "Volume": f"{rel_vol:.2f} (Req: >{rel_vol_threshold})", "RSI": f"{rsi:.2f} (Req: {rsi_min}-{rsi_max})"})
    return False

def calculate_tp_sl(symbol: str, entry_price: float, last_atr: float) -> Optional[Dict[str, Any]]:
    if last_atr <= 0:
        log_rejection(symbol, "Invalid ATR for Fallback", {"detail": "ATR is zero or negative"})
        return None
    
    # --- تعديل لحساب TP/SL ليتوافق مع قوانين المنصة ---
    symbol_info = get_exchange_info_for_symbol(symbol)
    if not symbol_info:
        log_rejection(symbol, "TP/SL Calculation", {"detail": "Failed to get exchange info"})
        return None

    tick_size = None
    for f in symbol_info['filters']:
        if f['filterType'] == 'PRICE_FILTER':
            tick_size = f['tickSize']
            break
    
    if not tick_size:
        log_rejection(symbol, "TP/SL Calculation", {"detail": "Could not find tickSize"})
        return None

    fallback_tp = entry_price + (last_atr * ATR_FALLBACK_TP_MULTIPLIER)
    fallback_sl = entry_price - (last_atr * ATR_FALLBACK_SL_MULTIPLIER)

    # تعديل الأسعار لتتوافق مع tickSize
    adjusted_tp = adjust_value_by_step_size(fallback_tp, tick_size)
    adjusted_sl = adjust_value_by_step_size(fallback_sl, tick_size)

    return {'target_price': adjusted_tp, 'stop_loss': adjusted_sl, 'source': 'ATR_Fallback'}


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
    
    # --- تعديل: إلغاء أمر OCO عند الإغلاق اليدوي ---
    if status == 'manual_close' and ENABLE_REAL_TRADING and client:
        oco_order_id = signal_to_close.get('oco_order_id')
        if oco_order_id:
            try:
                logger.info(f"ℹ️ [Manual Close] Attempting to cancel OCO order list {oco_order_id} for {symbol}.")
                client.cancel_order(symbol=symbol, orderListId=int(oco_order_id))
                logger.info(f"✅ [Manual Close] Successfully cancelled OCO order list {oco_order_id}.")
            except BinanceAPIException as e:
                # قد يكون الأمر تم تنفيذه بالفعل، وهذا طبيعي
                if e.code == -2011: # Unknown order sent.
                     logger.warning(f"⚠️ [Manual Close] OCO order list {oco_order_id} for {symbol} not found on exchange. It might have already been filled or cancelled.")
                else:
                    logger.error(f"❌ [Manual Close] Failed to cancel OCO order list {oco_order_id} for {symbol}: {e}")
                    # لا نوقف العملية، قد نرغب في بيع الكمية المتبقية يدويًا
            except Exception as e:
                logger.error(f"❌ [Manual Close] An unexpected error occurred while cancelling OCO order {oco_order_id}: {e}")


    with closure_lock:
        if signal_id in signals_pending_closure:
            logger.warning(f"⚠️ [Closure] Closure for signal {signal_id} ({symbol}) already in progress.")
            return
        signals_pending_closure.add(signal_id)
    with signal_cache_lock: open_signals_cache.pop(symbol, None)
    logger.info(f"ℹ️ [Closure] Starting closure thread for signal {signal_id} ({symbol}) with status '{status}'.")
    Thread(target=close_signal, args=(signal_to_close, status, closing_price, "initiator")).start()

def update_signal_peak_price_in_db(signal_id: int, new_peak_price: float):
    if not check_db_connection() or not conn:
        logger.error(f"❌ [DB Peak Update] Cannot update peak for signal {signal_id}, DB connection is down.")
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
                time.sleep(1)
                continue
            
            # --- منطق مراقبة الصفقات لم يعد يعتمد على وضع أوامر البيع ---
            # --- الآن دوره الأساسي هو تحديث الأسعار الحالية وحساب الربح/الخسارة للعرض ---
            # --- وتفعيل وقف الخسارة المتحرك (إذا كان مفعلاً) ---

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
                    if signal_id in signals_pending_closure:
                        continue
                
                price = None
                if perform_direct_api_check:
                    try: price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                    except Exception: pass
                if not price and redis_prices.get(symbol):
                    price = float(redis_prices[symbol])
                if not price: continue

                with signal_cache_lock:
                    if symbol in open_signals_cache:
                        open_signals_cache[symbol]['current_price'] = price
                        open_signals_cache[symbol]['pnl_pct'] = ((price / float(signal['entry_price'])) - 1) * 100

                # --- منطق وقف الخسارة المتحرك ---
                # هذا الجزء يحتاج إلى تعديل معقد لإلغاء أمر OCO ووضع أمر جديد.
                # سنبقيه كما هو في الوقت الحالي للتركيز على الوظائف الأساسية.
                # في التداول الحقيقي، هذا الجزء سيقوم بإلغاء OCO ووضع أمر جديد.
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
                            last_update = LAST_PEAK_UPDATE_TIME.get(signal_id, 0)
                            if now - last_update > PEAK_UPDATE_COOLDOWN:
                                update_signal_peak_price_in_db(signal_id, price)
                                LAST_PEAK_UPDATE_TIME[signal_id] = now
                            current_peak = price
                        
                        # trailing_stop_price = current_peak * (1 - TRAILING_DISTANCE_PERCENT / 100)
                        # logger.info(f"📈 [Trailing SL] {symbol} new peak: {current_peak:.4f}. New potential SL: {trailing_stop_price:.4f}")
                        # ملاحظة: هنا يجب وضع منطق إلغاء OCO القديم ووضع OCO جديد مع وقف الخسارة المحدث.

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

def send_new_signal_alert(signal_data: Dict[str, Any], is_real_trade: bool = False):
    symbol = signal_data['symbol']; entry = float(signal_data['entry_price']); target = float(signal_data['target_price']); sl = float(signal_data['stop_loss'])
    profit_pct = ((target / entry) - 1) * 100
    risk_pct = abs(((entry / sl) - 1) * 100) if sl > 0 else 0
    rrr = profit_pct / risk_pct if risk_pct > 0 else 0
    with market_state_lock: market_regime = current_market_state.get('overall_regime', 'N/A')
    confidence_display = signal_data['signal_details'].get('ML_Confidence_Display', 'N/A')
    
    # --- تعديل رسالة التلغرام لتعكس نوع الصفقة (حقيقية أم إشارة) ---
    trade_type_header = "🔥 *صفقة حقيقية جديدة* 🔥" if is_real_trade else "💡 *توصية تداول جديدة* 💡"
    quantity_line = f"\n*الكمية:* `{signal_data.get('quantity', 'N/A'):.8g}`" if is_real_trade else ""

    message = (f"{trade_type_header}\n\n*العملة:* `{symbol}`\n*حالة السوق:* `{market_regime}`\n"
               f"{quantity_line}\n"
               f"*الدخول:* `{entry:,.8g}`\n*الهدف:* `{target:,.8g}`\n*وقف الخسارة:* `{sl:,.8g}`\n\n"
               f"*الربح المتوقع:* `{profit_pct:.2f}%`\n*المخاطرة/العائد:* `1:{rrr:.2f}`\n\n"
               f"*ثقة النموذج:* `{confidence_display}`")
    reply_markup = {"inline_keyboard": [[{"text": "📊 فتح لوحة التحكم", "url": WEBHOOK_URL or '#'}]]}
    if send_telegram_message(CHAT_ID, message, reply_markup):
        log_and_notify('info', f"New {'Real Trade' if is_real_trade else 'Signal'}: {symbol} in {market_regime} market", "NEW_SIGNAL")

def send_trade_update_alert(signal_data: Dict[str, Any], old_signal_data: Dict[str, Any]):
    symbol = signal_data['symbol']
    old_target = float(old_signal_data['target_price'])
    new_target = float(signal_data['target_price'])
    old_sl = float(old_signal_data['stop_loss'])
    new_sl = float(signal_data['stop_loss'])
    old_conf = old_signal_data['signal_details'].get('ML_Confidence_Display', 'N/A')
    new_conf = signal_data['signal_details'].get('ML_Confidence_Display', 'N/A')
    
    message = (f"🔄 *تحديث صفقة (تعزيز)* 🔄\n\n"
               f"*العملة:* `{symbol}`\n\n"
               f"*الثقة:* `{old_conf}` ⬅️ `{new_conf}`\n"
               f"*الهدف:* `{old_target:,.8g}` ⬅️ `{new_target:,.8g}`\n"
               f"*الوقف:* `{old_sl:,.8g}` ⬅️ `{new_sl:,.8g}`\n\n"
               f"تم تحديث الصفقة بناءً على إشارة شراء أقوى.")
    reply_markup = {"inline_keyboard": [[{"text": "📊 فتح لوحة التحكم", "url": WEBHOOK_URL or '#'}]]}
    if send_telegram_message(CHAT_ID, message, reply_markup):
        log_and_notify('info', f"Updated Signal: {symbol} due to stronger signal.", "UPDATE_SIGNAL")

def insert_signal_into_db(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not check_db_connection() or not conn: return None
    try:
        entry = float(signal['entry_price']); target = float(signal['target_price']); sl = float(signal['stop_loss'])
        with conn.cursor() as cur:
            # --- تعديل دالة الإدخال لتشمل بيانات التداول الحقيقي ---
            cur.execute("""
                INSERT INTO signals (
                    symbol, entry_price, target_price, stop_loss, strategy_name, 
                    signal_details, current_peak_price, buy_order_id, oco_order_id, quantity
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
                """,
                (
                    signal['symbol'], entry, target, sl, signal.get('strategy_name'), 
                    json.dumps(signal.get('signal_details', {})), entry,
                    signal.get('buy_order_id'), signal.get('oco_order_id'), signal.get('quantity')
                )
            )
            signal['id'] = cur.fetchone()['id']
        conn.commit()
        logger.info(f"✅ [DB] Inserted signal {signal['id']} for {signal['symbol']}.")
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

def close_signal(signal: Dict, status: str, closing_price: float, closed_by: str):
    signal_id = signal.get('id'); symbol = signal.get('symbol')
    logger.info(f"Initiating closure for signal {signal_id} ({symbol}) with status '{status}'")
    try:
        if not check_db_connection() or not conn: raise OperationalError("DB connection failed.")
        db_closing_price = float(closing_price); entry_price = float(signal['entry_price'])
        profit_pct = ((db_closing_price / entry_price) - 1) * 100
        with conn.cursor() as cur:
            cur.execute("UPDATE signals SET status = %s, closing_price = %s, closed_at = NOW(), profit_percentage = %s WHERE id = %s AND status IN ('open', 'updated');",
                        (status, db_closing_price, profit_pct, signal_id))
            if cur.rowcount == 0: logger.warning(f"⚠️ [DB Close] Signal {signal_id} was already closed or not found."); return
        conn.commit()
        status_map = {'target_hit': '✅ تحقق الهدف', 'stop_loss_hit': '🛑 ضرب وقف الخسارة', 'manual_close': '🖐️ إغلاق يدوي', 'closed_by_sell_signal': '🔴 إغلاق بإشارة بيع'}
        status_message = status_map.get(status, status)
        alert_msg = (f"*{status_message}*\n*العملة:* `{symbol}`\n*الربح:* `{profit_pct:+.2f}%`")
        send_telegram_message(CHAT_ID, alert_msg)
        log_and_notify('info', f"{status_message}: {symbol} | Profit: {profit_pct:+.2f}%", 'CLOSE_SIGNAL')
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

def main_loop():
    logger.info("[Main Loop] Waiting for initialization...")
    time.sleep(15)
    if not validated_symbols_to_scan: log_and_notify("critical", "No validated symbols to scan.", "SYSTEM"); return
    
    if ENABLE_REAL_TRADING:
        log_and_notify("warning", "🔴 REAL TRADING MODE IS ACTIVE. THE BOT WILL EXECUTE TRADES WITH REAL MONEY. 🔴", "SYSTEM")
    else:
        log_and_notify("info", "🔵 Paper Trading Mode is active. The bot will only generate signals, not execute trades. 🔵", "SYSTEM")

    log_and_notify("info", f"Starting scan loop for {len(validated_symbols_to_scan)} symbols.", "SYSTEM")
    
    while True:
        try:
            determine_market_state()
            with market_state_lock: market_regime = current_market_state.get("overall_regime", "UNCERTAIN")
            if USE_BTC_TREND_FILTER and market_regime in ["DOWNTREND", "STRONG DOWNTREND"]:
                log_rejection("ALL", "BTC Trend Filter", {"detail": f"Scan paused due to market regime: {market_regime}"})
                time.sleep(300); continue
            
            btc_data = get_btc_data_for_bot()
            for symbol in validated_symbols_to_scan:
                try:
                    with signal_cache_lock:
                        open_trade = open_signals_cache.get(symbol)
                        open_trade_count = len(open_signals_cache)

                    if open_trade:
                        # لا تقم بمسح عملة لديها صفقة مفتوحة بالفعل
                        continue

                    df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_15m is None: continue
                    strategy = TradingStrategy(symbol)
                    if not all([strategy.ml_model, strategy.scaler, strategy.feature_names]): continue
                    df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS * 4)
                    if df_4h is None: continue
                    df_features = strategy.get_features(df_15m, df_4h, btc_data)
                    if df_features is None or df_features.empty: continue
                    signal_info = strategy.generate_signal(df_features)
                    if not signal_info or not redis_client: continue
                    current_price_str = redis_client.hget(REDIS_PRICES_HASH_NAME, symbol)
                    if not current_price_str: continue
                    current_price = float(current_price_str)
                    prediction, confidence = signal_info['prediction'], signal_info['confidence']
                    
                    if prediction == 1 and confidence >= BUY_CONFIDENCE_THRESHOLD:
                        last_features = df_features.iloc[-1]; last_features.name = symbol
                        
                        if open_trade_count >= MAX_OPEN_TRADES:
                            log_rejection(symbol, "Max Open Trades", {"count": open_trade_count, "max": MAX_OPEN_TRADES}); continue
                        
                        if USE_SPEED_FILTER and not passes_speed_filter(last_features): continue
                        if USE_MOMENTUM_FILTER and not passes_momentum_filter(last_features): continue
                        
                        last_atr = last_features.get('atr', 0)
                        volatility = (last_atr / current_price * 100)
                        if USE_MIN_VOLATILITY_FILTER and volatility < MIN_VOLATILITY_PERCENT:
                            log_rejection(symbol, "Low Volatility", {"volatility": f"{volatility:.2f}%", "min": f"{MIN_VOLATILITY_PERCENT}%"}); continue
                        if USE_BTC_CORRELATION_FILTER and market_regime in ["UPTREND", "STRONG UPTREND"]:
                            correlation = last_features.get('btc_correlation', 0)
                            if correlation < MIN_BTC_CORRELATION:
                                log_rejection(symbol, "BTC Correlation", {"corr": f"{correlation:.2f}", "min": f"{MIN_BTC_CORRELATION}"}); continue
                        
                        tp_sl_data = calculate_tp_sl(symbol, current_price, last_atr)
                        if not tp_sl_data: continue
                        
                        new_signal = {
                            'symbol': symbol, 'strategy_name': BASE_ML_MODEL_NAME, 
                            'signal_details': {'ML_Confidence': confidence, 'ML_Confidence_Display': f"{confidence:.2%}"}, 
                            'entry_price': current_price, **tp_sl_data
                        }

                        if USE_RRR_FILTER:
                            risk = current_price - float(new_signal['stop_loss']); reward = float(new_signal['target_price']) - current_price
                            if risk <= 0 or reward <= 0 or (reward / risk) < MIN_RISK_REWARD_RATIO:
                                log_rejection(symbol, "RRR Filter", {"rrr": f"{(reward/risk):.2f}" if risk > 0 else "N/A"}); continue
                        
                        # --- START: REAL TRADING EXECUTION LOGIC ---
                        if ENABLE_REAL_TRADING:
                            try:
                                # 1. الحصول على معلومات التداول للعملة
                                symbol_info = get_exchange_info_for_symbol(symbol)
                                if not symbol_info:
                                    log_rejection(symbol, "Real Trade Execution", {"detail": "Failed to get exchange info."})
                                    continue

                                lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                                min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'MIN_NOTIONAL'), None)
                                
                                if not lot_size_filter or not min_notional_filter:
                                    log_rejection(symbol, "Real Trade Execution", {"detail": "LOT_SIZE or MIN_NOTIONAL filter not found."})
                                    continue

                                step_size = lot_size_filter['stepSize']
                                min_notional = float(min_notional_filter['minNotional'])

                                # 2. حساب الرصيد وتحديد حجم الصفقة
                                usdt_balance = get_usdt_balance()
                                usdt_to_spend = usdt_balance * (TRADE_BALANCE_PERCENT / 100)
                                
                                if usdt_to_spend < min_notional:
                                    log_rejection(symbol, "Insufficient Balance for Min Notional", {"required": min_notional, "available_to_spend": usdt_to_spend})
                                    continue
                                
                                # 3. حساب وتعديل الكمية
                                quantity = usdt_to_spend / current_price
                                adjusted_quantity = adjust_value_by_step_size(quantity, step_size)

                                if adjusted_quantity == 0:
                                    log_rejection(symbol, "Quantity Too Small", {"calculated_qty": quantity, "adjusted_qty": adjusted_quantity})
                                    continue

                                # 4. تنفيذ أمر الشراء
                                logger.info(f"💰 [REAL TRADE] Attempting to BUY {adjusted_quantity} of {symbol} at market price.")
                                buy_order = client.create_order(
                                    symbol=symbol,
                                    side=Client.SIDE_BUY,
                                    type=Client.ORDER_TYPE_MARKET,
                                    quantity=adjusted_quantity
                                )
                                logger.info(f"✅ [REAL TRADE] Market BUY order placed successfully for {symbol}. Order ID: {buy_order['orderId']}")
                                
                                # 5. حساب سعر الدخول الفعلي والكمية من استجابة الأمر
                                filled_quantity = float(buy_order['executedQty'])
                                total_cost = float(buy_order['cummulativeQuoteQty'])
                                actual_entry_price = total_cost / filled_quantity
                                
                                new_signal['entry_price'] = actual_entry_price
                                new_signal['quantity'] = filled_quantity
                                new_signal['buy_order_id'] = buy_order['orderId']

                                # 6. إعادة حساب TP/SL بناءً على سعر الدخول الفعلي
                                new_tp_sl = calculate_tp_sl(symbol, actual_entry_price, last_atr)
                                if not new_tp_sl:
                                    log_and_notify('error', f"CRITICAL: Bought {symbol} but failed to calculate TP/SL for OCO. MANUAL INTERVENTION NEEDED.", "CRITICAL_ERROR")
                                    continue
                                
                                new_signal.update(new_tp_sl)

                                # 7. تنفيذ أمر OCO للبيع (جني الأرباح ووقف الخسارة)
                                logger.info(f"🛡️ [REAL TRADE] Placing OCO SELL order for {symbol}. TP: {new_signal['target_price']}, SL: {new_signal['stop_loss']}")
                                oco_order = client.create_oco_order(
                                    symbol=symbol,
                                    side=Client.SIDE_SELL,
                                    quantity=filled_quantity,
                                    price=f"{new_signal['target_price']:.8f}".rstrip('0').rstrip('.'), # Take Profit
                                    stopPrice=f"{new_signal['stop_loss']:.8f}".rstrip('0').rstrip('.'), # Stop Loss Trigger
                                    stopLimitPrice=f"{new_signal['stop_loss']:.8f}".rstrip('0').rstrip('.'), # Stop Loss Limit
                                    stopLimitTimeInForce=Client.TIME_IN_FORCE_GTC
                                )
                                logger.info(f"✅ [REAL TRADE] OCO order placed successfully for {symbol}. List Order ID: {oco_order['orderListId']}")
                                new_signal['oco_order_id'] = oco_order['orderListId']

                                # 8. حفظ الصفقة الحقيقية في قاعدة البيانات
                                saved_signal = insert_signal_into_db(new_signal)
                                if saved_signal:
                                    with signal_cache_lock:
                                        open_signals_cache[saved_signal['symbol']] = saved_signal
                                    send_new_signal_alert(saved_signal, is_real_trade=True)

                            except BinanceAPIException as e:
                                log_and_notify('error', f"Binance API Error during real trade for {symbol}: {e}", "TRADE_ERROR")
                            except Exception as e:
                                log_and_notify('error', f"General Error during real trade for {symbol}: {e}", "TRADE_ERROR")
                        else:
                            # وضع التداول الورقي (Paper Trading)
                            saved_signal = insert_signal_into_db(new_signal)
                            if saved_signal:
                                with signal_cache_lock:
                                    open_signals_cache[saved_signal['symbol']] = saved_signal
                                send_new_signal_alert(saved_signal, is_real_trade=False)
                        # --- END: REAL TRADING EXECUTION LOGIC ---
                    time.sleep(2)
                except Exception as e: logger.error(f"❌ [Processing Error] {symbol}: {e}", exc_info=True)
            logger.info("ℹ️ [End of Cycle] Scan cycle finished. Waiting..."); time.sleep(300)
        except (KeyboardInterrupt, SystemExit): break
        except Exception as main_err: log_and_notify("error", f"Error in main loop: {main_err}", "SYSTEM"); time.sleep(120)

# ---------------------- واجهة برمجة تطبيقات Flask (تم الإصلاح) ----------------------
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

@app.route('/')
def home():
    return render_template_string(get_dashboard_html())

@app.route('/api/market_status')
def get_market_status():
    with market_state_lock: state_copy = dict(current_market_state)
    return jsonify({
        "fear_and_greed": get_fear_and_greed_index(), "market_state": state_copy,
        "db_ok": check_db_connection(), "api_ok": check_api_status()
    })

@app.route('/api/stats')
def get_stats():
    if not check_db_connection() or not conn:
        logger.error("❌ [API Stats] DB connection check failed.")
        return jsonify({"error": "DB connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT status, profit_percentage, quantity, entry_price FROM signals;")
            all_signals = cur.fetchall()
        open_trades_count = sum(1 for s in all_signals if s.get('status') in ['open', 'updated'])
        closed_trades = [s for s in all_signals if s.get('status') not in ['open', 'updated'] and s.get('profit_percentage') is not None]
        
        # --- تعديل حساب الربح ليعكس الصفقات الحقيقية أو الافتراضية ---
        total_net_profit_usdt = 0.0
        for t in closed_trades:
            profit_pct = float(t['profit_percentage'])
            # إذا كانت الصفقة حقيقية (لديها كمية مسجلة)، استخدم حجمها الفعلي
            if t.get('quantity') and t.get('entry_price'):
                trade_size_usdt = float(t['quantity']) * float(t['entry_price'])
            else: # وإلا، استخدم الحجم الافتراضي
                trade_size_usdt = HYPOTHETICAL_TRADE_SIZE_USDT
            
            # افترض أن رسوم الدخول والخروج تمثل 2 * TRADING_FEE_PERCENT
            net_profit_pct = profit_pct - (2 * TRADING_FEE_PERCENT)
            total_net_profit_usdt += (net_profit_pct / 100) * trade_size_usdt

        win_rate = 0.0
        profit_factor_val = 0.0
        if closed_trades:
            wins = sum(1 for s in closed_trades if float(s['profit_percentage']) > 0)
            win_rate = (wins / len(closed_trades) * 100) if closed_trades else 0.0
            total_profit_from_wins = sum(float(s['profit_percentage']) for s in closed_trades if float(s['profit_percentage']) > 0)
            total_loss_from_losses = abs(sum(float(s['profit_percentage']) for s in closed_trades if float(s['profit_percentage']) < 0))
            if total_loss_from_losses > 0:
                profit_factor_val = total_profit_from_wins / total_loss_from_losses
            elif total_profit_from_wins > 0:
                profit_factor_val = "Infinity"

        return jsonify({
            "open_trades_count": open_trades_count,
            "net_profit_usdt": total_net_profit_usdt,
            "win_rate": win_rate,
            "profit_factor": profit_factor_val
        })
    except Exception as e:
        logger.error(f"❌ [API Stats] Critical error: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred while calculating stats."}), 500

@app.route('/api/profit_curve')
def get_profit_curve():
    if not check_db_connection(): return jsonify({"error": "DB connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT closed_at, profit_percentage FROM signals WHERE status NOT IN ('open', 'updated') AND profit_percentage IS NOT NULL AND closed_at IS NOT NULL ORDER BY closed_at ASC;")
            trades = cur.fetchall()
        
        curve_data = []
        cumulative_profit = 0.0
        
        start_time = (trades[0]['closed_at'] - timedelta(minutes=1)).isoformat() if trades else datetime.now(timezone.utc).isoformat()
        curve_data.append({"timestamp": start_time, "profit_range": [0.0, 0.0], "profit_change": 0.0})

        for trade in trades:
            profit_change = float(trade['profit_percentage'])
            start_profit = cumulative_profit
            end_profit = cumulative_profit + profit_change
            curve_data.append({
                "timestamp": trade['closed_at'].isoformat(),
                "profit_range": [start_profit, end_profit],
                "profit_change": profit_change
            })
            cumulative_profit = end_profit
            
        return jsonify(curve_data)
    except Exception as e:
        logger.error(f"❌ [API Profit Curve] Error: {e}", exc_info=True)
        return jsonify({"error": "Error fetching profit curve data"}), 500

@app.route('/api/signals')
def get_signals():
    if not check_db_connection() or not redis_client: return jsonify({"error": "Service connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals ORDER BY CASE WHEN status IN ('open', 'updated') THEN 0 ELSE 1 END, id DESC;")
            all_signals = [dict(s) for s in cur.fetchall()]
        open_symbols = [s['symbol'] for s in all_signals if s['status'] in ('open', 'updated')]
        if open_symbols:
            prices_list = redis_client.hmget(REDIS_PRICES_HASH_NAME, open_symbols)
            current_prices = {symbol: float(p) if p else None for symbol, p in zip(open_symbols, prices_list)}
            for s in all_signals:
                if s['status'] in ('open', 'updated'):
                    price = current_prices.get(s['symbol'])
                    s['current_price'] = price
                    if price and s.get('entry_price'): s['pnl_pct'] = ((price / float(s['entry_price'])) - 1) * 100
        return jsonify(all_signals)
    except Exception as e: return jsonify({"error": str(e)}), 500

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
        
        signal_dict = dict(signal_to_close)
        symbol = signal_dict['symbol']
        
        # --- تعديل الإغلاق اليدوي ليعمل مع الصفقات الحقيقية ---
        if ENABLE_REAL_TRADING and signal_dict.get('quantity'):
            # إذا كانت صفقة حقيقية، قم ببيع الكمية بسعر السوق
            quantity_to_sell = signal_dict['quantity']
            logger.info(f"💰 [MANUAL CLOSE] Attempting to SELL {quantity_to_sell} of {symbol} at market price.")
            try:
                # أولاً، قم بإلغاء أمر OCO الموجود
                initiate_signal_closure(symbol, signal_dict, 'manual_close', 0) # استدعاء لإلغاء OCO
                time.sleep(1) # انتظر قليلاً للتأكد من الإلغاء
                
                sell_order = client.create_order(
                    symbol=symbol,
                    side=Client.SIDE_SELL,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity_to_sell
                )
                closing_price = float(sell_order['cummulativeQuoteQty']) / float(sell_order['executedQty'])
                logger.info(f"✅ [MANUAL CLOSE] Market SELL order executed for {symbol} at price {closing_price}.")
                # سيتم تحديث قاعدة البيانات عبر stream أو webhooks لاحقًا
                return jsonify({"message": f"تم إرسال أمر بيع سوقي للصفقة {signal_id}."})
            except BinanceAPIException as e:
                 logger.error(f"❌ [API Manual Close] Binance error selling {symbol}: {e}")
                 return jsonify({"error": f"Binance error selling {symbol}: {e.message}"}), 500
            except Exception as e:
                 logger.error(f"❌ [API Manual Close] General error selling {symbol}: {e}")
                 return jsonify({"error": f"An error occurred: {str(e)}"}), 500
        else:
            # إذا كانت صفقة ورقية، أغلقها كالمعتاد
            try:
                price = float(client.get_symbol_ticker(symbol=symbol)['price'])
            except Exception as e:
                logger.error(f"❌ [API Close] Could not fetch price for {symbol}: {e}")
                return jsonify({"error": f"Could not fetch price for {symbol}"}), 500
            initiate_signal_closure(symbol, signal_dict, 'manual_close', price)
            return jsonify({"message": f"تم إرسال طلب إغلاق الصفقة {signal_id}..."})

    except Exception as e:
        logger.error(f"❌ [API Close] Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/notifications')
def get_notifications():
    with notifications_lock: return jsonify(list(notifications_cache))

@app.route('/api/rejection_logs')
def get_rejection_logs():
    with rejection_logs_lock: return jsonify(list(rejection_logs_cache))

def run_flask():
    port_str = os.environ.get('PORT', '10000')
    try:
        port = int(port_str)
    except (ValueError, TypeError):
        logger.error(f"❌ Invalid PORT environment variable: '{port_str}'. Defaulting to 10000.")
        port = 10000
    host = "0.0.0.0"
    logger.info(f"✅ Preparing to start dashboard on {host}:{port}")
    logger.info("🤖 Starting background bot services in a separate thread...")
    initialization_thread = Thread(target=initialize_bot_services, daemon=True)
    initialization_thread.start()
    logger.info("🌐 Starting web server...")
    try:
        from waitress import serve
        logger.info("✅ Found 'waitress', starting production server...")
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        logger.warning("⚠️ 'waitress' not found. Using Flask's development server (NOT recommended for production).")
        app.run(host=host, port=port)

# ---------------------- نقطة انطلاق البرنامج ----------------------
def run_websocket_manager():
    """Manages the WebSocket connection for real-time price updates."""
    if not client or not validated_symbols_to_scan:
        logger.error("❌ [WebSocket] Cannot start WebSocket manager: Client or symbols not initialized.")
        return
    logger.info("📈 [WebSocket] Starting WebSocket Manager for price streams...")
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    
    chunk_size = 50 
    symbol_chunks = [validated_symbols_to_scan[i:i + chunk_size] for i in range(0, len(validated_symbols_to_scan), chunk_size)]
    
    for i, chunk in enumerate(symbol_chunks):
        streams = [f"{s.lower()}@miniTicker" for s in chunk]
        twm.start_multiplex_socket(callback=handle_price_update_message, streams=streams)
        logger.info(f"✅ [WebSocket] Subscribed to price stream chunk {i+1}/{len(symbol_chunks)}.")

    # --- إضافة stream لبيانات الحساب لمراقبة تنفيذ الأوامر ---
    if ENABLE_REAL_TRADING:
        logger.info("📈 [WebSocket] Starting User Data Stream...")
        # twm.start_user_socket(callback=handle_user_data_message) # تحتاج لدالة معالجة جديدة
        logger.info("✅ [WebSocket] User Data Stream started.")

    twm.join()

def initialize_bot_services():
    global client, validated_symbols_to_scan
    logger.info("🤖 [Bot Services] Starting background initialization...")
    try:
        client = Client(API_KEY, API_SECRET)
        init_db()
        init_redis()
        load_open_signals_to_cache()
        load_notifications_to_cache()
        Thread(target=determine_market_state, daemon=True).start()
        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan:
            logger.critical("❌ No validated symbols to scan. Loops will not start."); return
        Thread(target=run_websocket_manager, daemon=True).start()
        Thread(target=trade_monitoring_loop, daemon=True).start()
        Thread(target=main_loop, daemon=True).start()
        logger.info("✅ [Bot Services] All background services started successfully.")
    except Exception as e:
        log_and_notify("critical", f"A critical error occurred during initialization: {e}", "SYSTEM")
        exit(1)

if __name__ == "__main__":
    logger.info("======================================================")
    logger.info("🚀 LAUNCHING TRADING BOT & DASHBOARD APPLICATION 🚀")
    logger.info("======================================================")
    run_flask()
    logger.info("👋 [Shutdown] Application has been shut down."); os._exit(0)
