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

# --- تجاهل التحذيرات غير الهامة ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v10.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV10_Improved')

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
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V7_With_Ichimoku'
MODEL_FOLDER: str = 'V7'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 30
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices"
MODEL_BATCH_SIZE: int = 5
DIRECT_API_CHECK_INTERVAL: int = 10

# --- مؤشرات فنية ---
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
EMA_FAST_PERIOD: int = 50
EMA_SLOW_PERIOD: int = 200

# --- إدارة الصفقات ---
MAX_OPEN_TRADES: int = 10
BUY_CONFIDENCE_THRESHOLD = 0.65
SELL_CONFIDENCE_THRESHOLD = 0.70
MIN_PROFIT_FOR_SELL_CLOSE_PERCENT = 0.2
TRADE_AMOUNT_USDT: float = 10.0  # حجم الصفقة الافتراضي بالدولار
BINANCE_FEE_RATE: float = 0.001 # رسوم Binance (0.1%)

# --- إعدادات الهدف ووقف الخسارة ---
ATR_FALLBACK_SL_MULTIPLIER: float = 1.5
ATR_FALLBACK_TP_MULTIPLIER: float = 2.0
SL_BUFFER_ATR_PERCENT: float = 0.25 

# --- إعدادات وقف الخسارة المتحرك (Trailing Stop-Loss) ---
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_ACTIVATION_PROFIT_PERCENT: float = 1.0
TRAILING_DISTANCE_PERCENT: float = 0.8

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

# --- متغيرات حالة السوق المحسّنة ---
last_market_state_check = 0
current_market_state: Dict[str, Any] = {
    "overall_regime": "INITIALIZING",
    "details": {},
    "last_updated": None
}
market_state_lock = Lock()


# ---------------------- دوال HTML للوحة التحكم (نسخة محسنة V3) ----------------------
def get_dashboard_html_v3():
    # This function returns the full HTML for the new dashboard.
    # It includes TailwindCSS, Chart.js, and custom styling for a modern look.
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم بوت التداول V3</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #030712; /* gray-950 */
            color: #f9fafb; /* gray-50 */
        }
        .card {
            background-color: #111827; /* gray-900 */
            border: 1px solid #1f2937; /* gray-800 */
            transition: all 0.3s ease;
        }
        .card:hover {
            border-color: #374151; /* gray-700 */
            transform: translateY(-2px);
        }
        .tab-btn.active {
            color: #2563eb; /* blue-600 */
            border-bottom-color: #2563eb;
        }
        .progress-container {
            position: relative;
            height: 10px;
            background-color: #374151; /* gray-700 */
            border-radius: 9999px;
        }
        .progress-bar {
            height: 100%;
            border-radius: 9999px;
            transition: width 0.5s ease-in-out;
        }
        .progress-point {
            position: absolute;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 12px;
            height: 12px;
            border-radius: 50%;
            border: 2px solid #111827; /* gray-900 */
        }
        .progress-point.current {
             width: 16px; height: 16px; z-index: 10;
        }
        /* Chart.js tooltip custom styles */
        .chartjs-tooltip {
            background: rgba(31, 41, 55, 0.8);
            border-radius: 0.5rem;
            color: white;
            padding: 0.5rem 1rem;
            pointer-events: none;
            position: absolute;
            transition: all .1s ease;
            backdrop-filter: blur(4px);
        }
        .fade-in {
            animation: fadeIn 0.5s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body class="p-4 sm:p-6 lg:p-8">
    <div class="container mx-auto max-w-screen-2xl">
        <!-- Header -->
        <header class="mb-8 flex flex-wrap justify-between items-center gap-4">
            <h1 class="text-3xl font-bold text-white">لوحة التحكم الاحترافية</h1>
            <div id="last-updated" class="text-sm text-gray-400">آخر تحديث: ...</div>
        </header>

        <!-- Main Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            
            <!-- Left Column: Stats & Chart -->
            <div class="lg:col-span-1 xl:col-span-1 flex flex-col gap-6">
                <!-- Stats Cards -->
                <div class="grid grid-cols-2 gap-4">
                    <div class="card rounded-xl p-4 fade-in">
                        <h3 class="text-sm font-medium text-gray-400">صافي الربح (USDT)</h3>
                        <p id="net-profit-usdt" class="text-2xl font-semibold mt-1">...</p>
                    </div>
                    <div class="card rounded-xl p-4 fade-in" style="animation-delay: 0.1s;">
                        <h3 class="text-sm font-medium text-gray-400">نسبة النجاح</h3>
                        <p id="win-rate" class="text-2xl font-semibold mt-1">...</p>
                    </div>
                    <div class="card rounded-xl p-4 fade-in" style="animation-delay: 0.2s;">
                        <h3 class="text-sm font-medium text-gray-400">عامل الربح</h3>
                        <p id="profit-factor" class="text-2xl font-semibold mt-1">...</p>
                    </div>
                    <div class="card rounded-xl p-4 fade-in" style="animation-delay: 0.3s;">
                        <h3 class="text-sm font-medium text-gray-400">إجمالي الرسوم (USDT)</h3>
                        <p id="total-fees" class="text-2xl font-semibold mt-1">...</p>
                    </div>
                </div>

                <!-- Equity Curve Chart -->
                <div class="card rounded-xl p-4 h-80 fade-in" style="animation-delay: 0.4s;">
                    <h3 class="text-lg font-semibold mb-2">منحنى الربح التراكمي</h3>
                    <div class="relative h-full w-full">
                        <canvas id="equityCurveChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- Right Column: Trades & Info -->
            <div class="lg:col-span-2 xl:col-span-3 flex flex-col gap-6">
                <!-- Tabs -->
                <div class="card rounded-xl p-2 fade-in" style="animation-delay: 0.5s;">
                    <nav class="flex space-x-2 sm:space-x-4">
                        <button onclick="showTab('open-signals')" class="tab-btn active text-gray-300 hover:text-white py-2 px-4 font-semibold border-b-2 border-transparent transition-colors duration-200">الصفقات المفتوحة</button>
                        <button onclick="showTab('closed-signals')" class="tab-btn text-gray-400 hover:text-white py-2 px-4 font-semibold border-b-2 border-transparent transition-colors duration-200">سجل الصفقات</button>
                        <button onclick="showTab('notifications')" class="tab-btn text-gray-400 hover:text-white py-2 px-4 font-semibold border-b-2 border-transparent transition-colors duration-200">الإشعارات</button>
                    </nav>
                </div>

                <!-- Tab Content -->
                <main class="card rounded-xl p-4 min-h-[60vh] fade-in" style="animation-delay: 0.6s;">
                    <div id="open-signals-tab" class="tab-content">
                        <div id="open-signals-list" class="space-y-4"></div>
                    </div>
                    <div id="closed-signals-tab" class="tab-content hidden">
                        <div class="overflow-x-auto">
                            <table class="min-w-full text-sm text-right">
                                <thead class="border-b border-gray-700">
                                    <tr>
                                        <th class="p-3 font-semibold text-gray-400">العملة</th>
                                        <th class="p-3 font-semibold text-gray-400">تاريخ الإغلاق</th>
                                        <th class="p-3 font-semibold text-gray-400">الربح %</th>
                                        <th class="p-3 font-semibold text-gray-400">الربح USDT</th>
                                        <th class="p-3 font-semibold text-gray-400">الحالة</th>
                                    </tr>
                                </thead>
                                <tbody id="closed-signals-table-body"></tbody>
                            </table>
                        </div>
                    </div>
                    <div id="notifications-tab" class="tab-content hidden">
                        <div id="notifications-list" class="space-y-2 max-h-[70vh] overflow-y-auto"></div>
                    </div>
                </main>
            </div>
        </div>
    </div>
    
    <!-- Modal for manual close -->
    <div id="confirmation-modal" class="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50 transition-opacity duration-300 opacity-0 pointer-events-none">
        <div class="card rounded-xl p-6 w-full max-w-sm transform scale-95 transition-transform duration-300">
            <h3 id="modal-title" class="text-xl font-bold mb-4"></h3>
            <p id="modal-body" class="text-gray-300 mb-6"></p>
            <div class="flex justify-end gap-3">
                <button id="modal-cancel" class="py-2 px-4 rounded-lg bg-gray-600 hover:bg-gray-500 transition-colors">إلغاء</button>
                <button id="modal-confirm" class="py-2 px-4 rounded-lg bg-red-600 hover:bg-red-500 text-white transition-colors">تأكيد الإغلاق</button>
            </div>
        </div>
    </div>

<script>
let equityChart = null;

// --- Chart.js Configuration ---
const chartConfig = {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'الربح التراكمي (USDT)',
            data: [],
            borderColor: '#22d3ee', // cyan-400
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 6,
            pointBackgroundColor: '#22d3ee',
            tension: 0.3,
            fill: true,
            backgroundColor: (context) => {
                const ctx = context.chart.ctx;
                const gradient = ctx.createLinearGradient(0, 0, 0, 200);
                gradient.addColorStop(0, 'rgba(34, 211, 238, 0.3)');
                gradient.addColorStop(1, 'rgba(34, 211, 238, 0)');
                return gradient;
            },
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                ticks: { color: '#9ca3af' }, // gray-400
                grid: { color: 'rgba(55, 65, 81, 0.5)' } // gray-700
            },
            y: {
                ticks: { color: '#9ca3af', callback: (value) => '$' + value },
                grid: { color: 'rgba(55, 65, 81, 0.5)' }
            }
        },
        plugins: {
            legend: { display: false },
            tooltip: {
                enabled: false,
                external: (context) => {
                    // Custom Tooltip
                    let tooltipEl = document.getElementById('chartjs-tooltip');
                    if (!tooltipEl) {
                        tooltipEl = document.createElement('div');
                        tooltipEl.id = 'chartjs-tooltip';
                        tooltipEl.className = 'chartjs-tooltip';
                        document.body.appendChild(tooltipEl);
                    }
                    const tooltipModel = context.tooltip;
                    if (tooltipModel.opacity === 0) {
                        tooltipEl.style.opacity = 0;
                        return;
                    }
                    if (tooltipModel.body) {
                        const date = tooltipModel.title || [];
                        const value = tooltipModel.dataPoints[0].formattedValue;
                        tooltipEl.innerHTML = `<div>${date}</div><div><strong>${value}</strong></div>`;
                    }
                    const position = context.chart.canvas.getBoundingClientRect();
                    tooltipEl.style.opacity = 1;
                    tooltipEl.style.left = position.left + window.pageXOffset + tooltipModel.caretX + 'px';
                    tooltipEl.style.top = position.top + window.pageYOffset + tooltipModel.caretY + 'px';
                }
            }
        }
    }
};

// --- API & Data Handling ---
async function fetchData(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`API Error: ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error(`Failed to fetch from ${url}:`, error);
        return null;
    }
}

function updateStats() {
    fetchData('/api/stats_v2').then(data => {
        if (!data) return;
        const profitEl = document.getElementById('net-profit-usdt');
        profitEl.textContent = `${data.total_net_profit_usdt.toFixed(2)} $`;
        profitEl.className = `text-2xl font-semibold mt-1 ${data.total_net_profit_usdt >= 0 ? 'text-green-400' : 'text-red-400'}`;
        
        document.getElementById('win-rate').textContent = `${data.win_rate.toFixed(2)} %`;
        document.getElementById('profit-factor').textContent = data.profit_factor.toFixed(2);
        document.getElementById('total-fees').textContent = `${data.total_fees_usdt.toFixed(2)} $`;
    });
}

function updateEquityCurve() {
    fetchData('/api/equity_curve').then(data => {
        if (!data) return;
        const chartCanvas = document.getElementById('equityCurveChart');
        if (!chartCanvas) return;

        const labels = data.map(d => new Date(d.timestamp).toLocaleDateString('ar-EG'));
        const values = data.map(d => d.cumulative_profit);

        if (!equityChart) {
            equityChart = new Chart(chartCanvas, chartConfig);
        }
        equityChart.data.labels = labels;
        equityChart.data.datasets[0].data = values;
        equityChart.update();
    });
}

function updateSignals() {
    fetchData('/api/signals_v2').then(data => {
        if (!data) return;
        
        const openSignals = data.filter(s => s.status === 'open');
        const closedSignals = data.filter(s => s.status !== 'open');

        const openList = document.getElementById('open-signals-list');
        openList.innerHTML = openSignals.length > 0 ? openSignals.map(createOpenSignalCard).join('') : '<p class="text-gray-400 text-center">لا توجد صفقات مفتوحة حالياً.</p>';

        const closedBody = document.getElementById('closed-signals-table-body');
        closedBody.innerHTML = closedSignals.map(createClosedSignalRow).join('');
    });
}

function createOpenSignalCard(signal) {
    const pnlPct = signal.pnl_pct || 0;
    const pnlColor = pnlPct >= 0 ? 'text-green-400' : 'text-red-400';
    
    const { sl, entry, tp, current } = {
        sl: parseFloat(signal.stop_loss),
        entry: parseFloat(signal.entry_price),
        tp: parseFloat(signal.target_price),
        current: parseFloat(signal.current_price)
    };

    let progress = 0;
    const range = tp - sl;
    if (range > 0) {
        progress = ((current - sl) / range) * 100;
    }
    progress = Math.max(0, Math.min(100, progress));

    const slPos = 0;
    const entryPos = ((entry - sl) / range) * 100;
    const tpPos = 100;

    return `
        <div class="card rounded-lg p-4 border-l-4 ${pnlPct >= 0 ? 'border-green-500' : 'border-red-500'}">
            <div class="flex justify-between items-center mb-3">
                <div class="flex items-center gap-3">
                    <span class="font-bold text-lg">${signal.symbol}</span>
                    <span class="text-xs font-mono px-2 py-1 rounded bg-gray-700">${signal.strategy_name}</span>
                </div>
                <div class="${pnlColor} font-semibold text-lg">${pnlPct.toFixed(2)}%</div>
            </div>

            <div class="space-y-2 text-sm mb-4">
                <div class="flex justify-between">
                    <span class="text-gray-400">الدخول:</span>
                    <span class="font-mono">${entry.toFixed(5)}</span>
                </div>
                <div class="flex justify-between">
                    <span class="text-gray-400">السعر الحالي:</span>
                    <span class="font-mono">${current.toFixed(5)}</span>
                </div>
            </div>

            <div class="progress-container my-2">
                <div class="progress-bar ${pnlPct >= 0 ? 'bg-green-500' : 'bg-red-500'}" style="width: ${progress}%"></div>
                <!-- Points -->
                <div class="progress-point bg-red-500" style="left: ${slPos}%" title="Stop Loss: ${sl.toFixed(5)}"></div>
                <div class="progress-point bg-gray-400" style="left: ${entryPos}%" title="Entry: ${entry.toFixed(5)}"></div>
                <div class="progress-point current bg-white" style="left: ${progress}%" title="Current: ${current.toFixed(5)}"></div>
                <div class="progress-point bg-green-500" style="left: ${tpPos}%" title="Take Profit: ${tp.toFixed(5)}"></div>
            </div>
            <div class="flex justify-between text-xs text-gray-400 mt-1">
                <span>SL: ${sl.toFixed(5)}</span>
                <span>TP: ${tp.toFixed(5)}</span>
            </div>

            <div class="mt-4 text-right">
                <button onclick="confirmCloseSignal(${signal.id}, '${signal.symbol}')" class="bg-red-700 hover:bg-red-600 text-white text-xs py-1.5 px-3 rounded-md transition-colors">إغلاق يدوي</button>
            </div>
        </div>
    `;
}

function createClosedSignalRow(signal) {
    const profitPct = signal.profit_percentage || 0;
    const netProfitUsdt = signal.net_profit_usdt || 0;
    const pnlColor = profitPct >= 0 ? 'text-green-400' : 'text-red-400';
    return `
        <tr class="border-b border-gray-800 hover:bg-gray-800/50">
            <td class="p-3 font-semibold">${signal.symbol}</td>
            <td class="p-3 text-gray-400">${new Date(signal.closed_at).toLocaleString('ar-EG')}</td>
            <td class="p-3 font-mono ${pnlColor}">${profitPct.toFixed(2)}%</td>
            <td class="p-3 font-mono ${pnlColor}">${netProfitUsdt.toFixed(3)} $</td>
            <td class="p-3 text-gray-300">${signal.status}</td>
        </tr>
    `;
}

function createNotificationItem(notification) {
    return `<div class="p-3 rounded-md bg-gray-800/50 text-sm text-gray-300">[${new Date(notification.timestamp).toLocaleString('ar-EG')}] ${notification.message}</div>`;
}

// --- UI Interaction ---
function showTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.add('hidden'));
    document.getElementById(`${tabId}-tab`).classList.remove('hidden');
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
}

function confirmCloseSignal(id, symbol) {
    const modal = document.getElementById('confirmation-modal');
    document.getElementById('modal-title').textContent = `تأكيد إغلاق الصفقة`;
    document.getElementById('modal-body').textContent = `هل أنت متأكد من رغبتك في إغلاق الصفقة #${id} للعملة ${symbol} يدوياً؟`;
    
    modal.classList.remove('opacity-0', 'pointer-events-none');
    modal.querySelector('.card').classList.remove('scale-95');

    const confirmBtn = document.getElementById('modal-confirm');
    const cancelBtn = document.getElementById('modal-cancel');

    const close = () => {
        modal.classList.add('opacity-0', 'pointer-events-none');
        modal.querySelector('.card').classList.add('scale-95');
    };

    confirmBtn.onclick = () => {
        fetch(`/api/close/${id}`, { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                if(data.error) console.error("Close Error:", data.error);
                refreshData();
            });
        close();
    };
    cancelBtn.onclick = close;
}

// --- Initial Load & Refresh ---
function refreshData() {
    updateStats();
    updateEquityCurve();
    updateSignals();
    fetchData('/api/notifications').then(data => {
        if(data) document.getElementById('notifications-list').innerHTML = data.map(createNotificationItem).join('');
    });
    document.getElementById('last-updated').textContent = `آخر تحديث: ${new Date().toLocaleTimeString('ar-EG')}`;
}

window.onload = () => {
    refreshData();
    setInterval(refreshData, 10000); // Refresh every 10 seconds
};

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
        logger.info("[DB] 'sslmode=require' was automatically added to the database URL.")
    
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(db_url_to_use, connect_timeout=15, cursor_factory=RealDictCursor)
            conn.autocommit = False
            with conn.cursor() as cur:
                # Add new columns for USDT profit calculation
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
                        trade_amount_usdt DOUBLE PRECISION,
                        fees_usdt DOUBLE PRECISION,
                        net_profit_usdt DOUBLE PRECISION
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL,
                        message TEXT NOT NULL,
                        is_read BOOLEAN DEFAULT FALSE
                    );
                """)
            conn.commit()
            logger.info("✅ [DB] Database connection successful and tables initialized.")
            return
        except Exception as e:
            logger.error(f"❌ [DB] Connection error (Attempt {attempt + 1}/{retries}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logger.critical("❌ [DB] Failed to connect after multiple retries.")
                exit(1)

def check_db_connection() -> bool:
    global conn
    if conn is None or conn.closed != 0:
        logger.warning("[DB] Connection is closed, attempting to reconnect...")
        init_db()
    try:
        if conn and conn.closed == 0:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
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
        with notifications_lock:
            notifications_cache.appendleft(new_notification)
        with conn.cursor() as cur:
            cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [Notify DB] Failed to save notification to DB: {e}")
        if conn: conn.rollback()

def log_rejection(symbol: str, reason: str, details: Optional[Dict] = None):
    details_str = f" | {details}" if details else ""
    logger.info(f"ℹ️ [{symbol}] Signal rejected. Reason: {reason}{details_str}")
    with rejection_logs_lock:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "reason": reason,
            "details": details or {}
        }
        rejection_logs_cache.appendleft(log_entry)

def init_redis() -> None:
    global redis_client
    logger.info("[Redis] Initializing Redis connection...")
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] Successfully connected to Redis server.")
    except redis.exceptions.ConnectionError as e:
        logger.critical(f"❌ [Redis] Failed to connect to Redis at {REDIS_URL}. Error: {e}")
        exit(1)

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    logger.info(f"ℹ️ [Validation] Reading symbols from '{filename}' and validating with Binance...")
    if not client:
        logger.error("❌ [Validation] Binance client is not initialized.")
        return []
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
        limit = int((days * 24 * 60) / int(interval[:-1])) if 'm' in interval else int((days * 24) / int(interval[:-1]))
        limit = min(limit, 1000)
        
        klines = client.get_historical_klines(symbol, interval, limit=limit)
        if not klines: return None
        
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        numeric_cols = {'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'float32'}
        df = df.astype(numeric_cols)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna()
    except BinanceAPIException as e:
        logger.warning(f"⚠️ [Binance API] Error fetching data for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ [Data] Error during historical data fetch for {symbol}: {e}")
        return None

def fetch_sr_levels_from_db(symbol: str) -> pd.DataFrame:
    if not check_db_connection() or not conn: return pd.DataFrame()
    query = "SELECT level_price, level_type FROM support_resistance_levels WHERE symbol = %s"
    try:
        df_levels = pd.read_sql(query, conn, params=(symbol,))
        if not df_levels.empty:
            df_levels['level_price'] = pd.to_numeric(df_levels['level_price'], errors='coerce')
            df_levels.dropna(subset=['level_price'], inplace=True)
        return df_levels
    except Exception as e:
        logger.error(f"❌ [S/R Fetch Bot] Could not fetch S/R levels for {symbol}: {e}")
        if conn: conn.rollback()
        return pd.DataFrame()

def fetch_ichimoku_features_from_db(symbol: str, timeframe: str) -> pd.DataFrame:
    if not check_db_connection() or not conn: return pd.DataFrame()
    query = """
        SELECT timestamp, tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b
        FROM ichimoku_features
        WHERE symbol = %s AND timeframe = %s
        ORDER BY timestamp DESC LIMIT 1;
    """
    try:
        df_ichimoku = pd.read_sql(query, conn, params=(symbol, timeframe), index_col='timestamp', parse_dates=['timestamp'])
        
        if df_ichimoku.empty:
            return df_ichimoku

        if not df_ichimoku.index.tz:
             df_ichimoku.index = df_ichimoku.index.tz_localize('UTC')

        if (datetime.now(timezone.utc) - df_ichimoku.index[-1]) > (pd.to_timedelta(timeframe) * 3):
            logger.warning(f"[{symbol}] ⚠️ Stale Ichimoku data found. Discarding.")
            return pd.DataFrame()

        for col in ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']:
            if col in df_ichimoku.columns:
                df_ichimoku[col] = pd.to_numeric(df_ichimoku[col], errors='coerce')
        
        return df_ichimoku.dropna()
    except Exception as e:
        logger.error(f"❌ [Ichimoku Fetch Bot] Could not fetch Ichimoku features for {symbol}: {e}")
        if conn: conn.rollback()
        return pd.DataFrame()

def calculate_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
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
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=30, min_periods=1).mean() + 1e-9)
    ema_fast_trend = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    ema_slow_trend = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    df_calc['price_vs_ema50'] = (df_calc['close'] / ema_fast_trend) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / ema_slow_trend) - 1
    df_calc['returns'] = df_calc['close'].pct_change()
    if btc_df is not None and not btc_df.empty:
        merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
        df_calc['btc_correlation'] = merged_df['returns'].rolling(window=30).corr(merged_df['btc_returns'])
    else:
        df_calc['btc_correlation'] = 0.0
    return df_calc.astype('float32', errors='ignore')

def load_ml_model_bundle_from_folder(symbol: str) -> Optional[Dict[str, Any]]:
    global ml_models_cache
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_name in ml_models_cache:
        return ml_models_cache[model_name]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FOLDER, f"{model_name}.pkl")
    if not os.path.exists(model_path):
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

def get_trend_for_timeframe(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or len(df) < 26:
        return {"trend": "Uncertain", "rsi": -1, "adx": -1}
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=13, adjust=False).mean()
    rsi = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr.replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr.replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    adx = dx.ewm(span=14, adjust=False).mean()
    ema_fast = df['close'].ewm(span=12, adjust=False).mean().iloc[-1]
    ema_slow = df['close'].ewm(span=26, adjust=False).mean().iloc[-1]
    trend = "Ranging"
    if adx.iloc[-1] > 20:
        if ema_fast > ema_slow and rsi.iloc[-1] > 50: trend = "Uptrend"
        elif ema_fast < ema_slow and rsi.iloc[-1] < 50: trend = "Downtrend"
    return {"trend": trend, "rsi": rsi.iloc[-1], "adx": adx.iloc[-1]}

def determine_market_state():
    global current_market_state, last_market_state_check
    with market_state_lock:
        if time.time() - last_market_state_check < 300: return current_market_state
    logger.info("🧠 [Market State] Updating market state using MTA...")
    try:
        df_1h = fetch_historical_data(BTC_SYMBOL, '1h', 5)
        df_4h = fetch_historical_data(BTC_SYMBOL, '4h', 15)
        if df_1h is None or df_4h is None:
            logger.warning("⚠️ [Market State] Could not fetch all required BTC data.")
            return current_market_state
        state_1h = get_trend_for_timeframe(df_1h)
        state_4h = get_trend_for_timeframe(df_4h)
        trends = [state_1h['trend'], state_4h['trend']]
        uptrends = trends.count("Uptrend"); downtrends = trends.count("Downtrend")
        overall_regime = "RANGING"
        if uptrends == 2: overall_regime = "STRONG UPTREND"
        elif uptrends == 1 and downtrends == 0: overall_regime = "UPTREND"
        elif downtrends == 2: overall_regime = "STRONG DOWNTREND"
        elif downtrends == 1 and uptrends == 0: overall_regime = "DOWNTREND"
        with market_state_lock:
            current_market_state = {
                "overall_regime": overall_regime,
                "details": {"1h": state_1h, "4h": state_4h},
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            last_market_state_check = time.time()
        logger.info(f"✅ [Market State] New state: {current_market_state['overall_regime']}")
        return current_market_state
    except Exception as e:
        logger.error(f"❌ [Market State] Failed to determine market state: {e}", exc_info=True)
        return current_market_state

# --- [RESTORED] Dynamic Speed Filter ---
def passes_speed_filter(last_features: pd.Series) -> bool:
    symbol = last_features.name
    with market_state_lock: regime = current_market_state.get("overall_regime", "RANGING")
    
    # Dynamic thresholds based on market regime
    if regime in ["DOWNTREND", "STRONG DOWNTREND"]:
        log_rejection(symbol, "Speed Filter", {"detail": f"Trading disabled in market regime: {regime}"})
        return False # Disable trading in downtrends as per original logic
    elif regime == "STRONG UPTREND": 
        adx_threshold, rel_vol_threshold, rsi_min, rsi_max = (25.0, 0.6, 45.0, 85.0)
    elif regime == "UPTREND": 
        adx_threshold, rel_vol_threshold, rsi_min, rsi_max = (22.0, 0.5, 40.0, 80.0)
    else: # RANGING or UNCERTAIN
        adx_threshold, rel_vol_threshold, rsi_min, rsi_max = (18.0, 0.2, 30.0, 80.0)
        
    adx = last_features.get('adx', 0)
    rel_vol = last_features.get('relative_volume', 0)
    rsi = last_features.get('rsi', 0)
    
    if (adx >= adx_threshold and rel_vol >= rel_vol_threshold and rsi_min <= rsi < rsi_max): 
        return True
        
    log_rejection(symbol, "Speed Filter", {
        "Regime": regime, 
        "ADX": f"{adx:.2f} (Req: >{adx_threshold})", 
        "Volume": f"{rel_vol:.2f} (Req: >{rel_vol_threshold})", 
        "RSI": f"{rsi:.2f} (Req: {rsi_min}-{rsi_max})"
    })
    return False

def calculate_tp_sl(symbol: str, entry_price: float, last_atr: float) -> Optional[Dict[str, Any]]:
    logger.info(f"[{symbol}] 🧠 Calculating TP/SL. Entry: {entry_price:.4f}, ATR: {last_atr:.4f}")
    sr_levels_df = fetch_sr_levels_from_db(symbol)
    ichimoku_df = fetch_ichimoku_features_from_db(symbol, SIGNAL_GENERATION_TIMEFRAME)
    db_levels = []
    if not sr_levels_df.empty: db_levels.extend(sr_levels_df['level_price'].astype(float).tolist())
    if not ichimoku_df.empty:
        last_ichi = ichimoku_df.iloc[-1]
        ichi_levels = [last_ichi.get(k) for k in ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']]
        db_levels.extend([float(lvl) for lvl in ichi_levels if pd.notna(lvl)])
    if db_levels:
        unique_levels = sorted(list(set(db_levels)))
        resistances = [lvl for lvl in unique_levels if lvl > entry_price]
        supports = [lvl for lvl in unique_levels if lvl < entry_price]
        if resistances and supports:
            target_price = min(resistances)
            stop_loss_base = max(supports)
            min_distance = last_atr * 0.5
            if (target_price - entry_price) < min_distance:
                 higher_resistances = [r for r in resistances if (r - entry_price) >= min_distance]
                 target_price = min(higher_resistances) if higher_resistances else None
            if stop_loss_base and (entry_price - stop_loss_base) < min_distance:
                lower_supports = [s for s in supports if (entry_price - s) >= min_distance]
                stop_loss_base = max(lower_supports) if lower_supports else None
            if target_price and stop_loss_base:
                final_stop_loss = stop_loss_base - (last_atr * SL_BUFFER_ATR_PERCENT)
                logger.info(f"✅ [{symbol}] DB-driven levels selected: TP={target_price:.4f}, SL={final_stop_loss:.4f}")
                return {'target_price': target_price, 'stop_loss': final_stop_loss, 'source': 'Database_Improved'}
    logger.warning(f"[{symbol}] ⚠️ Could not determine TP/SL from DB. Using ATR Fallback.")
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

    with signal_cache_lock:
        open_signals_cache.pop(symbol, None)

    logger.info(f"ℹ️ [Closure] Starting closure thread for signal {signal_id} ({symbol}) with status '{status}'.")
    Thread(target=close_signal, args=(signal_to_close, status, closing_price, "initiator")).start()

def run_websocket_manager() -> None:
    logger.info("ℹ️ [WebSocket] Starting WebSocket Manager...")
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    twm.start_miniticker_socket(callback=handle_price_update_message)
    logger.info("✅ [WebSocket] Connection successful.")
    twm.join()

class TradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_bundle_from_folder(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.feature_names is None: return None
        try:
            df_featured = calculate_features(df_15m, btc_df)
            delta_4h = df_4h['close'].diff()
            gain_4h = delta_4h.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
            loss_4h = -delta_4h.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
            df_4h['rsi_4h'] = 100 - (100 / (1 + (gain_4h / loss_4h.replace(0, 1e-9))))
            ema_fast_4h = df_4h['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
            df_4h['price_vs_ema50_4h'] = (df_4h['close'] / ema_fast_4h) - 1
            mtf_features = df_4h[['rsi_4h', 'price_vs_ema50_4h']]
            df_featured = df_featured.join(mtf_features)
            df_featured[['rsi_4h', 'price_vs_ema50_4h']] = df_featured[['rsi_4h', 'price_vs_ema50_4h']].fillna(method='ffill')
            for col in self.feature_names:
                if col not in df_featured.columns: df_featured[col] = 0.0
            df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_featured.dropna()
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
            logger.info(f"ℹ️ [{self.symbol}] Model predicted '{'BUY' if prediction == 1 else 'SELL'}' with {confidence:.2%} confidence.")
            return {'prediction': int(prediction), 'confidence': confidence}
        except Exception as e:
            logger.warning(f"⚠️ [{self.symbol}] Signal Generation Error: {e}")
            return None

def trade_monitoring_loop():
    global last_api_check_time
    logger.info("✅ [Trade Monitor] Starting trade monitoring loop.")
    while True:
        try:
            with signal_cache_lock: signals_to_check = dict(open_signals_cache)
            if not signals_to_check or not redis_client or not client: time.sleep(1); continue
            perform_direct_api_check = (time.time() - last_api_check_time) > DIRECT_API_CHECK_INTERVAL
            if perform_direct_api_check: last_api_check_time = time.time()
            symbols_to_fetch = list(signals_to_check.keys())
            redis_prices_list = redis_client.hmget(REDIS_PRICES_HASH_NAME, symbols_to_fetch)
            redis_prices = {symbol: price for symbol, price in zip(symbols_to_fetch, redis_prices_list)}
            for symbol, signal in signals_to_check.items():
                signal_id = signal.get('id')
                with closure_lock:
                    if signal_id in signals_pending_closure: continue
                price = None
                if perform_direct_api_check:
                    try: price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                    except Exception: pass
                if not price and redis_prices.get(symbol): price = float(redis_prices[symbol])
                if not price: continue
                target_price = float(signal.get('target_price', 0))
                original_stop_loss = float(signal.get('stop_loss', 0))
                effective_stop_loss = original_stop_loss
                if USE_TRAILING_STOP_LOSS:
                    entry_price = float(signal.get('entry_price', 0))
                    activation_price = entry_price * (1 + TRAILING_ACTIVATION_PROFIT_PERCENT / 100)
                    if price > activation_price:
                        current_peak = float(signal.get('current_peak_price', entry_price))
                        if price > current_peak: 
                            signal['current_peak_price'] = price
                            current_peak = price
                        trailing_stop_price = current_peak * (1 - TRAILING_DISTANCE_PERCENT / 100)
                        effective_stop_loss = max(original_stop_loss, trailing_stop_price)
                status_to_set = None
                if price >= target_price: status_to_set = 'target_hit'
                elif price <= effective_stop_loss: status_to_set = 'stop_loss_hit'
                if status_to_set:
                    logger.info(f"✅ [TRIGGER] ID:{signal_id} | {symbol} | Condition '{status_to_set}' met.")
                    initiate_signal_closure(symbol, signal, status_to_set, price)
            time.sleep(0.2)
        except Exception as e:
            logger.error(f"❌ [Trade Monitor] Critical error: {e}", exc_info=True)
            time.sleep(5)

def send_telegram_message(target_chat_id: str, text: str) -> bool:
    if not TELEGRAM_TOKEN or not target_chat_id:
        logger.error("❌ [Telegram] Token or Chat ID is missing.")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(target_chat_id), 'text': text}
        
    logger.info(f"ℹ️ [Telegram] Attempting to send plain text message to Chat ID: {target_chat_id}")
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info("✅ [Telegram] Message sent successfully.")
            return True
        else:
            logger.error(f"❌ [Telegram] Failed to send message. Status: {response.status_code}, Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [Telegram] Request failed: {e}")
        return False

def send_new_signal_alert(signal_data: Dict[str, Any]):
    message = (
        f"💡 توصية تداول جديدة 💡\n\n"
        f"العملة: {signal_data['symbol']}\n"
        f"الدخول: {float(signal_data['entry_price']):.8g}\n"
        f"الهدف: {float(signal_data['target_price']):.8g}\n"
        f"وقف الخسارة: {float(signal_data['stop_loss']):.8g}"
    )
    send_telegram_message(CHAT_ID, message)

def insert_signal_into_db(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not check_db_connection() or not conn: return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, current_peak_price, trade_amount_usdt)
                   VALUES (%(symbol)s, %(entry_price)s, %(target_price)s, %(stop_loss)s, %(strategy_name)s, %(signal_details)s, %(entry_price)s, %(trade_amount_usdt)s) RETURNING id;""",
                {
                    'symbol': signal['symbol'],
                    'entry_price': float(signal['entry_price']),
                    'target_price': float(signal['target_price']),
                    'stop_loss': float(signal['stop_loss']),
                    'strategy_name': signal.get('strategy_name'),
                    'signal_details': json.dumps(signal.get('signal_details', {})),
                    'trade_amount_usdt': float(signal.get('trade_amount_usdt'))
                }
            )
            signal['id'] = cur.fetchone()['id']
        conn.commit()
        logger.info(f"✅ [DB] Inserted signal {signal['id']} for {signal['symbol']}.")
        return signal
    except Exception as e:
        logger.error(f"❌ [DB Insert] Error inserting signal for {signal['symbol']}: {e}", exc_info=True)
        if conn: conn.rollback()
        return None

def close_signal(signal: Dict, status: str, closing_price: float, closed_by: str):
    signal_id = signal.get('id'); symbol = signal.get('symbol')
    logger.info(f"Closing signal {signal_id} ({symbol}) with status '{status}' by {closed_by}")
    try:
        if not check_db_connection() or not conn: raise OperationalError("DB connection failed.")
        
        entry_price = float(signal['entry_price'])
        trade_amount = float(signal.get('trade_amount_usdt', TRADE_AMOUNT_USDT))
        profit_pct = ((closing_price / entry_price) - 1) * 100
        
        pnl_usdt = trade_amount * (profit_pct / 100)
        entry_fee = trade_amount * BINANCE_FEE_RATE
        exit_fee = (trade_amount + pnl_usdt) * BINANCE_FEE_RATE
        total_fees = entry_fee + exit_fee
        net_profit_usdt = pnl_usdt - total_fees

        with conn.cursor() as cur:
            cur.execute(
                """UPDATE signals 
                   SET status = %s, closing_price = %s, closed_at = NOW(), profit_percentage = %s,
                       fees_usdt = %s, net_profit_usdt = %s
                   WHERE id = %s AND status = 'open';""",
                (status, closing_price, profit_pct, total_fees, net_profit_usdt, signal_id)
            )
            if cur.rowcount == 0: 
                logger.warning(f"⚠️ [DB Close] Signal {signal_id} already closed or not found."); 
                return
        conn.commit()

        status_map = {'target_hit': '✅ تحقق الهدف', 'stop_loss_hit': '🛑 ضرب وقف الخسارة', 'manual_close': '🖐️ إغلاق يدوي', 'closed_by_sell_signal': '🔴 إغلاق بإشارة بيع'}
        status_message = status_map.get(status, status)
        
        alert_msg = f"{status_message}\nالعملة: {symbol}\nالربح: {net_profit_usdt:+.2f} USDT ({profit_pct:+.2f}%)"
        send_telegram_message(CHAT_ID, alert_msg)
        log_and_notify('info', f"{status_message}: {symbol} | Net Profit: {net_profit_usdt:+.2f} USDT", 'CLOSE_SIGNAL')
        logger.info(f"✅ [DB Close] Signal {signal_id} closed. Net Profit: {net_profit_usdt:.2f} USDT")

    except Exception as e:
        logger.error(f"❌ [DB Close] Critical error closing signal {signal_id}: {e}", exc_info=True)
        if conn: conn.rollback()
    finally:
        with closure_lock: signals_pending_closure.discard(signal_id)

def load_open_signals_to_cache():
    if not check_db_connection() or not conn: return
    logger.info("ℹ️ [Loading] Loading open signals...")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status = 'open';")
            open_signals = cur.fetchall()
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in open_signals: open_signals_cache[signal['symbol']] = dict(signal)
            logger.info(f"✅ [Loading] Loaded {len(open_signals)} open signals.")
    except Exception as e: logger.error(f"❌ [Loading] Failed to load open signals: {e}")

def load_notifications_to_cache():
    if not check_db_connection() or not conn: return
    logger.info("ℹ️ [Loading] Loading latest notifications...")
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
    if btc_data is None: logger.error("❌ [BTC Data] Failed to fetch Bitcoin data."); return None
    btc_data['btc_returns'] = btc_data['close'].pct_change()
    return btc_data

# --- [RESTORED] Main Loop with Original Filters ---
def main_loop():
    logger.info("[Main Loop] Waiting for initialization...")
    time.sleep(15)
    if not validated_symbols_to_scan: log_and_notify("critical", "No validated symbols to scan.", "SYSTEM"); return
    log_and_notify("info", f"Starting scan loop for {len(validated_symbols_to_scan)} symbols.", "SYSTEM")
    all_symbols = list(validated_symbols_to_scan)
    while True:
        try:
            market_state = determine_market_state()
            market_regime = market_state.get("overall_regime", "UNCERTAIN")
            for i in range(0, len(all_symbols), MODEL_BATCH_SIZE):
                symbol_batch = all_symbols[i:i + MODEL_BATCH_SIZE]
                ml_models_cache.clear(); gc.collect()
                if USE_BTC_TREND_FILTER and market_regime in ["DOWNTREND", "STRONG DOWNTREND"]:
                    log_rejection("ALL", "BTC Trend Filter", {"detail": f"Scan paused due to market regime: {market_regime}"})
                    time.sleep(300); break
                with signal_cache_lock: open_count = len(open_signals_cache)
                if open_count >= MAX_OPEN_TRADES:
                    logger.info(f"ℹ️ [Pause] Max open trades limit reached."); time.sleep(60); break
                slots_available = MAX_OPEN_TRADES - open_count
                btc_data = get_btc_data_for_bot()
                if btc_data is None: time.sleep(120); continue
                for symbol in symbol_batch:
                    try:
                        df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        if df_15m is None: continue
                        strategy = TradingStrategy(symbol)
                        if not all([strategy.ml_model, strategy.scaler, strategy.feature_names]): continue
                        df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        if df_4h is None: continue
                        df_features = strategy.get_features(df_15m, df_4h, btc_data)
                        if df_features is None or df_features.empty: continue
                        signal_info = strategy.generate_signal(df_features)
                        if not signal_info or not redis_client: continue
                        current_price_str = redis_client.hget(REDIS_PRICES_HASH_NAME, symbol)
                        if not current_price_str: continue
                        current_price = float(current_price_str)
                        prediction, confidence = signal_info['prediction'], signal_info['confidence']
                        with signal_cache_lock: is_trade_open = symbol in open_signals_cache
                        if is_trade_open:
                            open_signal = open_signals_cache[symbol]
                            if prediction == -1 and confidence >= SELL_CONFIDENCE_THRESHOLD:
                                profit_check_price = float(open_signal['entry_price']) * (1 + MIN_PROFIT_FOR_SELL_CLOSE_PERCENT / 100)
                                if current_price >= profit_check_price:
                                    logger.info(f"✅ [Action] Closing open trade for {symbol} due to new SELL signal.")
                                    initiate_signal_closure(symbol, open_signal, 'closed_by_sell_signal', current_price)
                            elif prediction == 1 and confidence >= BUY_CONFIDENCE_THRESHOLD:
                                last_atr = df_features.iloc[-1].get('atr', 0)
                                tp_sl_data = calculate_tp_sl(symbol, current_price, last_atr)
                                if tp_sl_data and float(tp_sl_data['target_price']) > float(open_signal['target_price']):
                                    new_tp, new_sl = float(tp_sl_data['target_price']), float(tp_sl_data['stop_loss'])
                                    if update_signal_target_in_db(open_signal['id'], new_tp, new_sl):
                                        open_signals_cache[symbol]['target_price'] = new_tp
                                        open_signals_cache[symbol]['stop_loss'] = new_sl
                                        send_telegram_message(CHAT_ID, f"🔼 تحديث الهدف\nالعملة: {symbol}\nالهدف الجديد: {new_tp:,.8g}\nالوقف الجديد: {new_sl:,.8g}")
                        elif not is_trade_open and prediction == 1 and confidence >= BUY_CONFIDENCE_THRESHOLD:
                            if slots_available <= 0: continue
                            last_features = df_features.iloc[-1]; last_features.name = symbol
                            if USE_SPEED_FILTER and not passes_speed_filter(last_features): continue
                            last_atr = last_features.get('atr', 0)
                            volatility = (last_atr / current_price * 100)
                            if USE_MIN_VOLATILITY_FILTER and volatility < MIN_VOLATILITY_PERCENT:
                                log_rejection(symbol, "Low Volatility", {"volatility": f"{volatility:.2f}%", "min": f"{MIN_VOLATILITY_PERCENT}%"})
                                continue
                            if USE_BTC_CORRELATION_FILTER and market_regime in ["UPTREND", "STRONG UPTREND"]:
                                correlation = last_features.get('btc_correlation', 0)
                                if correlation < MIN_BTC_CORRELATION:
                                    log_rejection(symbol, "BTC Correlation", {"corr": f"{correlation:.2f}", "min": f"{MIN_BTC_CORRELATION}"})
                                    continue
                            tp_sl_data = calculate_tp_sl(symbol, current_price, last_atr)
                            if not tp_sl_data: continue
                            new_signal = {
                                'symbol': symbol, 
                                'strategy_name': BASE_ML_MODEL_NAME, 
                                'signal_details': {'ML_Confidence': f"{confidence:.2%}", 'TP_SL_Source': tp_sl_data['source']}, 
                                'entry_price': current_price, 
                                'trade_amount_usdt': TRADE_AMOUNT_USDT,
                                **tp_sl_data
                            }
                            if USE_RRR_FILTER:
                                risk = current_price - float(new_signal['stop_loss'])
                                reward = float(new_signal['target_price']) - current_price
                                if risk <= 0 or reward <= 0: 
                                    log_rejection(symbol, "Invalid R/R", {}); continue
                                if (reward / risk) < MIN_RISK_REWARD_RATIO: 
                                    log_rejection(symbol, "RRR Filter", {"rrr": f"{(reward/risk):.2f}"}); continue
                            logger.info(f"✅ [{symbol}] Signal passed all filters. Saving...")
                            saved_signal = insert_signal_into_db(new_signal)
                            if saved_signal:
                                with signal_cache_lock: open_signals_cache[saved_signal['symbol']] = saved_signal
                                send_new_signal_alert(saved_signal)
                                slots_available -= 1
                        del df_15m, df_4h, df_features; gc.collect()
                    except Exception as e: logger.error(f"❌ [Processing Error] {symbol}: {e}", exc_info=True)
                time.sleep(10)
            logger.info("ℹ️ [End of Cycle] Scan cycle finished. Waiting..."); time.sleep(60)
        except (KeyboardInterrupt, SystemExit): break
        except Exception as main_err: log_and_notify("error", f"Error in main loop: {main_err}", "SYSTEM"); time.sleep(120)

# ---------------------- واجهة برمجة تطبيقات Flask (محسنة V3) ----------------------
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template_string(get_dashboard_html_v3())

@app.route('/api/stats_v2')
def get_stats_v2():
    if not check_db_connection() or not conn: return jsonify({"error": "DB connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT net_profit_usdt, fees_usdt FROM signals WHERE status != 'open' AND net_profit_usdt IS NOT NULL;")
            closed_trades = cur.fetchall()
            
            total_net_profit = sum(s['net_profit_usdt'] for s in closed_trades)
            total_fees = sum(s['fees_usdt'] for s in closed_trades)
            
            wins = [s for s in closed_trades if s['net_profit_usdt'] > 0]
            win_rate = (len(wins) / len(closed_trades) * 100) if closed_trades else 0
            
            total_profit_from_wins = sum(s['net_profit_usdt'] for s in wins)
            total_loss_from_losses = abs(sum(s['net_profit_usdt'] for s in closed_trades if s['net_profit_usdt'] <= 0))
            profit_factor = (total_profit_from_wins / total_loss_from_losses) if total_loss_from_losses > 0 else float('inf')

        return jsonify({
            "total_net_profit_usdt": total_net_profit,
            "total_fees_usdt": total_fees,
            "win_rate": win_rate,
            "profit_factor": profit_factor
        })
    except Exception as e:
        logger.error(f"❌ [API Stats V2] Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/equity_curve')
def get_equity_curve():
    if not check_db_connection() or not conn: return jsonify({"error": "DB connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT closed_at, net_profit_usdt 
                FROM signals 
                WHERE status != 'open' AND net_profit_usdt IS NOT NULL 
                ORDER BY closed_at ASC;
            """)
            trades = cur.fetchall()
        
        cumulative_profit = 0
        equity_data = []
        for trade in trades:
            cumulative_profit += trade['net_profit_usdt']
            equity_data.append({
                "timestamp": trade['closed_at'].isoformat(),
                "cumulative_profit": cumulative_profit
            })
        
        return jsonify(equity_data)
    except Exception as e:
        logger.error(f"❌ [API Equity Curve] Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/signals_v2')
def get_signals_v2():
    if not check_db_connection() or not conn or not redis_client: return jsonify({"error": "Service connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals ORDER BY CASE WHEN status = 'open' THEN 0 ELSE 1 END, closed_at DESC, id DESC;")
            all_signals = [dict(s) for s in cur.fetchall()]
        
        open_symbols = [s['symbol'] for s in all_signals if s['status'] == 'open']
        if open_symbols:
            prices_list = redis_client.hmget(REDIS_PRICES_HASH_NAME, open_symbols)
            current_prices = {symbol: float(p) if p else None for symbol, p in zip(open_symbols, prices_list)}
            for s in all_signals:
                if s['status'] == 'open':
                    price = current_prices.get(s['symbol'])
                    s['current_price'] = price
                    if price and s.get('entry_price'): 
                        s['pnl_pct'] = ((price / float(s['entry_price'])) - 1) * 100
        
        return jsonify(all_signals)
    except Exception as e: 
        logger.error(f"❌ [API Signals V2] Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/close/<int:signal_id>', methods=['POST'])
def manual_close_signal(signal_id):
    if not client: return jsonify({"error": "Binance Client not available"}), 500
    with closure_lock:
        if signal_id in signals_pending_closure: return jsonify({"error": "Signal is already being closed"}), 409
    if not check_db_connection() or not conn: return jsonify({"error": "DB connection failed"}), 500
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE id = %s AND status = 'open';", (signal_id,))
            signal_to_close = cur.fetchone()
        
        if not signal_to_close:
            return jsonify({"error": "Signal not found or already closed"}), 404
        
        signal_data = dict(signal_to_close)
        symbol = signal_data['symbol']

        try:
            price = float(client.get_symbol_ticker(symbol=symbol)['price'])
        except Exception as e:
            logger.error(f"❌ [API Close] Could not fetch price for {symbol}: {e}")
            return jsonify({"error": f"Could not fetch price for {symbol}"}), 500

        initiate_signal_closure(symbol, signal_data, 'manual_close', price)
        
        return jsonify({"message": f"تم إرسال طلب إغلاق الصفقة {signal_id}..."})
    except Exception as e:
        logger.error(f"❌ [API Close] Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/notifications')
def get_notifications():
    with notifications_lock: return jsonify(list(notifications_cache))

def run_flask():
    host, port = "0.0.0.0", int(os.environ.get('PORT', 10000))
    logger.info(f"Dashboard V3 is starting on http://{host}:{port}")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        logger.warning("⚠️ [Flask] 'waitress' not found, using development server.")
        app.run(host=host, port=port)

# ---------------------- نقطة انطلاق البرنامج ----------------------
def initialize_bot_services():
    global client, validated_symbols_to_scan
    logger.info("🤖 [Bot Services] Starting background initialization...")
    try:
        client = Client(API_KEY, API_SECRET)
        logger.info("✅ [Binance] Connected to Binance API successfully.")
        init_db()
        init_redis()
        load_open_signals_to_cache()
        load_notifications_to_cache()
        determine_market_state()
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
    logger.info(f"🚀 Starting Trading Bot - Dashboard V3 with Original Filters...")
    initialization_thread = Thread(target=initialize_bot_services, daemon=True)
    initialization_thread.start()
    run_flask()
    logger.info("👋 [Shutdown] Bot has been shut down."); os._exit(0)
