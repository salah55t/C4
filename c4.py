# ملف c4.py - نسخة V34.2.0 (تحسين جلب الرصيد وحساب الصفقات)
# --- وصف التعديلات:
# 1. [Backend] إضافة دالة جديدة `get_available_balance` للاستعلام بشكل صريح عن الرصيد المتاح قبل فتح الصفقات الحقيقية.
# 2. [Backend] إعادة هيكلة دالة `calculate_position_size` لتوضيح منطق التحقق من الرصيد وقواعد المنصة.
# 3. [Backend] تحسين تسجيل الأخطاء عند فشل حساب حجم الصفقة لتقديم سبب واضح للرفض.
# 4. [لوحة التحكم] تعديل مسمى عرض الرصيد إلى "الرصيد المتاح (USDT)" لزيادة الوضوح.

import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import redis
import statistics
from decimal import Decimal, ROUND_DOWN, getcontext
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from flask_sock import Sock
from threading import Thread, Lock
from datetime import datetime, timezone, timedelta
from decouple import config
from typing import List, Dict, Optional, Any
from collections import deque
import warnings
from scipy.signal import argrelextrema

# --- إعدادات التجاهل واللوجر ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ضبط دقة النوع Decimal
getcontext().prec = 18

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v34_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV34.2.0')

# --- المشفر المخصص لأنواع بيانات NumPy ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, Decimal): return float(obj)
        if isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat()
        return super(NpEncoder, self).default(obj)

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
paper_trading_mode: bool = True
trading_mode_lock = Lock()
usdt_balance: float = 0.0
balance_lock = Lock()
cooldowns_by_symbol = {}
cooldowns_lock = Lock()
consecutive_losses_by_symbol = {}
consecutive_losses_lock = Lock()
COOLDOWN_MINUTES_AFTER_SL = 30
PAPER_TRADE_INITIAL_BALANCE = 1000.0

# --- المتغيرات القابلة للتعديل ---
FIXED_TRADE_AMOUNT_USDT: float = 5.0
fixed_trade_amount_lock = Lock()
MAX_OPEN_TRADES: int = 3
TRAILING_STOP_ACTIVATION_PROFIT_PERCENT: float = 1.0
MIN_SIGNAL_QUALITY: int = 70
AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE: bool = True
min_quality_lock = Lock()

# --- مفاتيح تفعيل الاستراتيجيات ---
USE_BB_STOCH_STRATEGY: bool = True
USE_MACD_EMA_STRATEGY: bool = True
USE_EMA_RSI_STRATEGY: bool = True
USE_PULLBACK_STRATEGY: bool = True
USE_MOMENTUM_VOLATILITY_STRATEGY: bool = True
USE_ELLIOTT_WAVE_STRATEGY: bool = True
USE_RANGE_REVERSAL_STRATEGY: bool = True

# --- إعدادات الفلاتر الديناميكية للاستراتيجيات ---
STRATEGY_NAMES = {
    "BB_Stoch_Strategy": "BB+Stoch (ارتداد مبكر)",
    "MACD_EMA_Strategy": "MACD+SMA (زخم وتقاطع)",
    "EMA_RSI_Strategy": "EMA+RSI (ارتداد سريع)",
    "Pullback_Strategy": "Pullback (ارتداد بحجم تداول)",
    "Momentum_Volatility_Strategy": "Momentum (زخم متزايد)",
    "Elliott_Wave_Strategy": "Elliott Wave (موجات إليوت)",
    "Range_Reversal_Strategy": "Range Reversal (انعكاس نطاقي)"
}
strategy_filters_lock = Lock()

# --- إعدادات عامة ---
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '1h'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 15
BTC_SYMBOL: str = 'BTCUSDT'
API_REQUEST_DELAY: float = 1

# --- متغيرات الحالة والكاش ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
ws_manager: Optional[ThreadedWebsocketManager] = None
live_prices: Dict[str, float] = {}
live_prices_lock = Lock()
exchange_info_map: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
notifications_cache = deque(maxlen=20)
notifications_lock = Lock()
rejection_logs_cache = deque(maxlen=30)
rejection_logs_lock = Lock()
current_market_state: Dict[str, Any] = {"trend_details_by_tf": {}}
market_state_lock = Lock()

# --- قاموس أسباب الرفض باللغة العربية ---
REJECTION_REASONS_AR = {
    # General Filters
    "Market Volatility Filter Failed": "فلتر تقلب السوق رفض الدخول",
    "Insufficient Historical Data": "بيانات تاريخية غير كافية للفحص",
    "MinNotional Filter Failed": "قيمة الصفقة أقل من الحد الأدنى للمنصة",
    "LOT_SIZE Filter Failed": "فشل تعديل حجم الصفقة",
    "Insufficient Balance": "الرصيد غير كافي لتنفيذ الصفقة",
    "Low Quality Signal": "جودة الإشارة منخفضة",
    "Invalid Position Size": "حجم الصفقة غير صالح (الوقف أعلى من الدخول)",
    "News Filter Failed": "فلتر الأخبار: تجنب التداول وقت الأخبار",
    "Liquidity Filter Failed": "فلتر السيولة: تجنب التداول في أوقات السيولة المنخفضة",
    "Correlation Filter Failed": "فلتر الارتباط: توجد صفقة مفتوحة على عملة مرتبطة",

    # Dynamic Filters Rejections
    "DYN_BB_WIDTH_LOW": "ديناميكي: عرض البولينجر ضيق جدًا",
    "DYN_STOCH_LOW": "ديناميكي: ستوكاستيك منخفض جدًا للسوق المتقلب",
    "DYN_VOLUME_LOW": "ديناميكي: حجم التداول منخفض بالنسبة للتقلبات",
    "DYN_ADX_LOW": "ديناميكي: قوة الاتجاه (ADX) ضعيفة للسوق الحالي",
    "DYN_MACD_MOMENTUM_LOW": "ديناميكي: زخم الماكد لا يتزايد بقوة كافية",
    "DYN_RSI_OOR": "ديناميكي: مؤشر القوة النسبية خارج النطاق المطلوب للاتجاه الحالي",
    "DYN_EMA_SPREAD_LOW": "ديناميكي: تباعد المتوسطات المتحركة ضعيف",
    "DYN_PULLBACK_SHALLOW": "ديناميكي: الارتداد ضحل جدًا للسوق المتقلب",
    "DYN_RECOVERY_FAIL": "ديناميكي: فشل السعر في التعافي بعد الارتداد",
    "DYN_VOLATILITY_OOR": "ديناميكي: التقلب خارج النطاق الأمثل للزخم",
    "DYN_MOMENTUM_SCORE_LOW": "ديناميكي: درجة الزخم الإجمالية منخفضة",
    "DYN_FIB_RETRACEMENT_OOR": "ديناميكي: تصحيح فيبوناتشي خارج النطاق المقبول للتقلب الحالي",

    # Strategy Specific Rejections
    "Trend: Not bullish on MTF or long-term": "الاتجاه ليس صاعدًا (لا طويل الأمد ولا على الإطارات القصيرة)",
    "Elliott Wave: Insufficient swing points": "موجات إليوت: نقاط تذبذب غير كافية",
    "Elliott Wave: Error in pattern detection": "موجات إليوت: خطأ في اكتشاف النمط",
    "Range Reversal: Trend too strong": "انعكاس نطاقي: الاتجاه قوي جدًا",
    "Range Reversal: RSI not in oversold zone": "انعكاس نطاقي: RSI ليس في منطقة تشبع بيعي"
}

# --- إعداد تطبيق Flask و WebSocket ---
app = Flask(__name__)
CORS(app)
sock = Sock(app)
ws_clients: List[Any] = []
ws_clients_lock = Lock()

DASHBOARD_TEMPLATE_V2 = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>لوحة التحكم - بوت التداول (V34.2.0)</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <style>
        :root { --bg: #0a0e1a; --panel: #12182c; --accent: #4a72ff; --ok: #15c46a; --warn: #ff9f1a; --bad: #ff4757; --muted: #8aa0c8; --border-color: #202945; }
        * { box-sizing: border-box; }
        body { margin: 0; background: var(--bg); color: #e8f1ff; font-family: 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif; -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
        .container { max-width: 1600px; margin: 0 auto; padding: 1rem; display: grid; gap: 1rem; grid-template-columns: 1fr; }
        @media(min-width: 1200px) { .container { grid-template-columns: 1fr 380px; } }
        header { grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; justify-content: space-between; margin-bottom: 0.5rem; }
        h1 { font-size: 1.5rem; margin: 0; font-weight: 700; color: #fff; }
        .badge { padding: 6px 12px; border-radius: 999px; font-size: 0.8rem; background: var(--panel); border: 1px solid var(--border-color); color: #cce0ff; font-variant-numeric: tabular-nums; }
        .main-column, .sidebar { display: flex; flex-direction: column; gap: 1rem; }
        .card { background: var(--panel); border: 1px solid var(--border-color); border-radius: 16px; box-shadow: 0 8px 30px rgba(0,0,0,.25); overflow: hidden; }
        .card-header { margin: 0; padding: 1rem; border-bottom: 1px solid var(--border-color); font-size: 1rem; color: #cfe2ff; display: flex; justify-content: space-between; align-items: center; font-weight: 600; }
        .card-body { padding: 1rem; }
        .signals-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1rem; }
        .signal-card { background: #1a223d; border: 1px solid var(--border-color); border-radius: 12px; padding: 1rem; display: grid; gap: 0.5rem 1rem; grid-template-areas: "title price" "meta price" "progress progress"; grid-template-columns: 1fr auto; will-change: transform; transition: transform 0.2s; }
        .signal-card:hover { transform: translateY(-2px); }
        .signal-title { grid-area: title; font-weight: 700; font-size: 1.1rem; color: #fff; }
        .signal-meta { grid-area: meta; font-size: 0.8rem; color: var(--muted); }
        .signal-price { grid-area: price; text-align: right; }
        .current-price { font-size: 1.25rem; font-weight: 700; direction: ltr; }
        .price-delta.green { color: var(--ok); } .price-delta.red { color: var(--bad); }
        .signal-progress { grid-area: progress; }
        .progress-bar { height: 8px; background: #2a3352; border-radius: 999px; overflow: hidden; }
        .progress-bar > span { display: block; height: 100%; transition: width 0.3s; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 1rem; }
        .status-item { text-align: center; }
        .status-label { font-size: 0.8rem; color: var(--muted); margin-bottom: 0.5rem; }
        .status-value { font-size: 1.25rem; font-weight: 700; color: #fff; }
        .status-value.green { color: var(--ok); } .status-value.red { color: var(--bad); }
        .trend-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; }
        .trend-pill { background: #1a223d; border-radius: 10px; padding: 0.75rem; text-align: center; }
        .trend-pill b { display: block; font-size: 0.8rem; color: #9fb7ef; margin-bottom: 4px; }
        .trend-pill span { font-size: 0.9rem; font-weight: 600; }
        .table-wrapper { max-height: 250px; overflow-y: auto; }
        .styled-table { width: 100%; border-collapse: collapse; }
        .styled-table th, .styled-table td { padding: 0.75rem; text-align: right; }
        .styled-table th { font-size: 0.8rem; color: #9ab2e2; font-weight: 600; border-bottom: 1px solid var(--border-color); }
        .styled-table td { font-size: 0.9rem; border-bottom: 1px solid #1a223d; white-space: nowrap; }
        .styled-table tbody tr:last-child td { border-bottom: none; }
        .control-group { display: flex; flex-direction: column; gap: 1rem; }
        .control-item { display: flex; justify-content: space-between; align-items: center; }
        .control-label { display: flex; flex-direction: column; }
        .control-label span:first-child { font-weight: 500; }
        .control-label small { color: var(--muted); font-size: 0.8rem; }
        .toggle-switch { display: inline-block; width: 44px; height: 24px; position: relative; }
        .toggle-switch input { opacity: 0; width: 0; height: 0; }
        .slider-track { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #2a3352; transition: .4s; border-radius: 34px; }
        .slider-track:before { position: absolute; content: ""; height: 18px; width: 18px; left: 3px; bottom: 3px; background-color: white; transition: .4s; border-radius: 50%; }
        input:checked + .slider-track { background-color: var(--accent); }
        input:checked + .slider-track:before { transform: translateX(20px); }
        .input-group { display: flex; align-items: center; gap: 0.5rem; }
        .input-group input { width: 70px; background: #1a223d; border: 1px solid var(--border-color); color: #fff; border-radius: 8px; padding: 6px 8px; text-align: center; }
        .chart-container { height: 220px; position: relative; }
        .loading-spinner, .chart-placeholder { display: flex; align-items: center; justify-content: center; position: absolute; top: 0; left: 0; right: 0; bottom: 0; color: var(--muted); }
        .spinner { border: 3px solid rgba(255, 255, 255, 0.1); border-radius: 50%; border-top: 3px solid var(--accent); width: 30px; height: 30px; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .btn-group { display: flex; gap: 0.5rem; }
        .btn { border: 1px solid var(--border-color); background: #1a223d; color: #d9e7ff; padding: 8px 12px; border-radius: 8px; cursor: pointer; font-weight: 600; transition: background-color 0.2s; }
        .btn.active, .btn:hover { background-color: var(--accent); color: #fff; border-color: var(--accent); }
    </style>
</head>
<body>
<div class="container">
    <header><h1>لوحة التحكم • بوت V34.2.0</h1><div class="badge" id="serverTime">—</div></header>
    
    <div class="main-column">
        <div class="card">
            <div class="card-header">الصفقات المفتوحة <span id="openSignalCount" class="badge">0</span></div>
            <div class="card-body">
                <div id="signals" class="signals-grid">
                    <p style="color:var(--muted); text-align:center;">لا توجد صفقات مفتوحة حالياً.</p>
                </div>
            </div>
        </div>
        <div class="card">
            <div class="card-header">مؤشرات الأداء (آخر 30 يوم)</div>
            <div class="card-body">
                <div class="status-grid" id="performanceMetrics">
                    <div class="loading-spinner"><div class="spinner"></div></div>
                </div>
                <div class="chart-container">
                    <div id="chartPlaceholder" class="chart-placeholder">جاري تحميل بيانات الأداء...</div>
                    <canvas id="performanceChart"></canvas>
                </div>
            </div>
        </div>
    </div>

    <div class="sidebar">
        <div class="card">
            <div class="card-header">التحكم والحالة</div>
            <div class="card-body">
                <div class="control-group">
                    <div class="control-item">
                        <div class="control-label"><span>تشغيل التداول</span></div>
                        <label class="toggle-switch"><input id="toggleTrading" type="checkbox"><span class="slider-track"></span></label>
                    </div>
                    <div class="control-item">
                        <div class="control-label"><span>وضع التداول</span></div>
                        <div class="btn-group">
                            <button class="btn" id="modePaper">ورقي</button>
                            <button class="btn" id="modeReal">حقيقي</button>
                        </div>
                    </div>
                </div>
                <hr style="border-color: var(--border-color); margin: 1rem 0;">
                <div class="status-grid">
                    <!-- [تعديل] تم تغيير المسمى هنا -->
                    <div class="status-item"><div class="status-label">الرصيد المتاح (USDT)</div><div id="balance" class="status-value">—</div></div>
                    <div class="status-item"><div class="status-label">صفقات مفتوحة</div><div id="openCount" class="status-value">—</div></div>
                </div>
            </div>
        </div>
        <div class="card">
            <div class="card-header">إعدادات التداول</div>
            <div class="card-body control-group">
                <div class="control-item">
                    <div class="control-label"><span>الحد الأدنى للجودة</span><small>جودة الإشارة المطلوبة لفتح صفقة</small></div>
                    <div class="input-group">
                        <input type="number" id="qualityFilter" min="30" max="90" step="1">
                        <span id="qualityValue" class="badge">70</span>
                    </div>
                </div>
                <div class="control-item">
                    <div class="control-label"><span>حجم الصفقة (USDT)</span><small>المبلغ الثابت لكل صفقة</small></div>
                    <div class="input-group">
                        <input type="number" id="tradeAmount" min="1" max="100" step="1">
                        <span id="tradeAmountValue" class="badge">$5.00</span>
                    </div>
                </div>
            </div>
        </div>
        <div class="card">
            <div class="card-header">حالة السوق</div>
            <div class="card-body">
                <div class="trend-grid" id="marketTrends">
                    <div class="loading-spinner" style="position: relative;"><div class="spinner"></div></div>
                </div>
            </div>
        </div>
        <div class="card">
            <div class="card-header">سجل الرفض</div>
            <div class="table-wrapper">
                <table class="styled-table">
                    <thead><tr><th>الوقت</th><th>الرمز</th><th>السبب</th></tr></thead>
                    <tbody id="rejections"></tbody>
                </table>
            </div>
        </div>
        <div class="card">
            <div class="card-header">سجل الأحداث</div>
            <div class="table-wrapper">
                <table class="styled-table">
                    <thead><tr><th>الوقت</th><th>النوع</th><th>الرسالة</th></tr></thead>
                    <tbody id="events"></tbody>
                </table>
            </div>
        </div>
    </div>
</div>
<script>
document.addEventListener('DOMContentLoaded', () => {
    const qs = s => document.querySelector(s);
    let lastPrices = {};
    let performanceChartInstance = null;
    let openSignals = {};

    const debounce = (func, delay) => {
        let timeout;
        return (...args) => {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), delay);
        };
    };

    function formatNumber(n, decimals = 6) {
        return n == null ? '—' : (+n).toLocaleString('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: decimals
        });
    }

    function renderSignal(signal) {
        const cp = lastPrices[signal.symbol] || signal.entry_price;
        const entry = signal.entry_price;
        const tp1 = signal.target_price_1;
        const sl = signal.stop_loss;
        let progress = 0;
        let color = 'transparent';
        
        if (cp >= entry && tp1 > entry) {
            progress = Math.min(100, ((cp - entry) / (tp1 - entry)) * 100);
            color = 'var(--ok)';
        } else if (cp < entry && entry > sl) {
            progress = Math.min(100, ((entry - cp) / (entry - sl)) * 100);
            color = 'var(--bad)';
        }

        const qualityScore = signal.signal_details?.quality_score || 0;
        const qualityColor = qualityScore > 75 ? 'var(--ok)' : qualityScore > 60 ? 'var(--warn)' : 'var(--bad)';
        const strategyName = (signal.strategy_name || "").replace(/_/g, " ").replace("Strategy", "");
        
        return `
            <div class="signal-card" id="signal-${signal.id}" data-symbol="${signal.symbol}">
                <div class="signal-title">${signal.symbol}</div>
                <div class="signal-meta">${strategyName} | <span style="color: ${qualityColor}; font-weight: bold;">⭐ ${qualityScore}/100</span></div>
                <div class="signal-price">
                    <div class="current-price" id="price-${signal.symbol}">${formatNumber(cp, 4)}</div>
                    <div class="price-delta" id="delta-${signal.symbol}"></div>
                </div>
                <div class="signal-progress">
                    <div class="progress-bar"><span style="width:${progress.toFixed(2)}%; background:${color};"></span></div>
                </div>
            </div>`;
    }

    function renderAllSignals() {
        const container = qs('#signals');
        const signalsArray = Object.values(openSignals);
        qs('#openSignalCount').textContent = signalsArray.length;
        qs('#openCount').textContent = signalsArray.length;

        if (signalsArray.length === 0) {
            container.innerHTML = '<p style="color:var(--muted); text-align:center;">لا توجد صفقات مفتوحة حالياً.</p>';
            return;
        }
        container.innerHTML = signalsArray.map(renderSignal).join('');
    }

    function updatePriceOnSignalCard(symbol, price) {
        const priceEl = document.getElementById(`price-${symbol}`);
        const deltaEl = document.getElementById(`delta-${symbol}`);
        const signalCard = document.querySelector(`.signal-card[data-symbol="${symbol}"]`);

        if (!signalCard) return;

        const prevPrice = lastPrices[symbol] || price;
        const delta = price - prevPrice;
        if (priceEl) priceEl.textContent = formatNumber(price, 4);
        if (deltaEl) {
            deltaEl.className = `price-delta small ${delta > 0 ? 'green' : (delta < 0 ? 'red' : '')}`;
            deltaEl.textContent = delta > 0 ? '▲' : (delta < 0 ? '▼' : '');
        }

        const signalId = signalCard.id.split('-')[1];
        const signalData = openSignals[signalId];
        if (signalData) {
            const entry = signalData.entry_price;
            const tp1 = signalData.target_price_1;
            const sl = signalData.stop_loss;
            let progress = 0, color = 'transparent';
            
            if (price >= entry && tp1 > entry) {
                progress = Math.min(100, ((price - entry) / (tp1 - entry)) * 100);
                color = 'var(--ok)';
            } else if (price < entry && entry > sl) {
                progress = Math.min(100, ((entry - price) / (entry - sl)) * 100);
                color = 'var(--bad)';
            }
            const progressBar = signalCard.querySelector('.progress-bar span');
            if (progressBar) {
                progressBar.style.width = `${progress}%`;
                progressBar.style.background = color;
            }
        }
    }
    
    function updatePrices(priceData) {
        for (const [symbol, price] of Object.entries(priceData)) {
            updatePriceOnSignalCard(symbol, price);
            lastPrices[symbol] = price;
        }
    }

    function addLogEntry(tableBodyId, entry, maxRows) {
        const tbody = qs(tableBodyId);
        const date = new Date(entry.timestamp).toLocaleTimeString('ar-EG', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        let rowHtml;
        if (tableBodyId === '#events') {
             rowHtml = `<tr><td>${date}</td><td>${entry.type || ''}</td><td>${entry.message || ''}</td></tr>`;
        } else {
             rowHtml = `<tr><td>${date}</td><td>${entry.symbol || ''}</td><td>${entry.reason || ''}</td></tr>`;
        }
        tbody.insertAdjacentHTML('afterbegin', rowHtml);
        if (tbody.rows.length > maxRows) tbody.deleteRow(-1);
    }
    
    function updateMarketTrends(marketState) {
        const trendsContainer = qs('#marketTrends');
        trendsContainer.innerHTML = '';
        if (marketState && marketState.trend_details_by_tf) {
            ['15m', '1h', '4h'].forEach(tf => {
                const trend = marketState.trend_details_by_tf[tf];
                if (trend) {
                    let trendClass = 'amber', trendText = 'جانبي';
                    if (trend.trend === 'bullish') { trendClass = 'green'; trendText = 'صاعد'; } 
                    else if (trend.trend === 'bearish') { trendClass = 'red'; trendText = 'هابط'; }
                    trendsContainer.innerHTML += `<div class="trend-pill"><b>${tf}</b><span class="${trendClass}">${trendText}</span></div>`;
                }
            });
        }
    }

    function renderPerformanceMetrics(stats) {
        const container = qs('#performanceMetrics');
        if (!stats || stats.total_trades === 0) {
            container.innerHTML = '<p style="color:var(--muted); text-align:center; grid-column: 1 / -1;">لا توجد بيانات كافية لعرض الإحصائيات.</p>';
            return;
        }
        const winRate = stats.win_rate !== null ? `${stats.win_rate.toFixed(1)}%` : '—';
        const avgProfit = stats.avg_profit !== null ? `${stats.avg_profit.toFixed(2)}%` : '—';
        
        container.innerHTML = `
            <div class="status-item"><div class="status-label">معدل الربح</div><div class="status-value ${stats.win_rate >= 50 ? 'green' : 'red'}">${winRate}</div></div>
            <div class="status-item"><div class="status-label">متوسط الربح</div><div class="status-value ${stats.avg_profit >= 0 ? 'green' : 'red'}">${avgProfit}</div></div>
            <div class="status-item"><div class="status-label">إجمالي الصفقات</div><div class="status-value">${stats.total_trades}</div></div>
            <div class="status-item"><div class="status-label">أكبر ربح</div><div class="status-value green">${stats.max_profit.toFixed(2)}%</div></div>
        `;
    }

    function renderPerformanceChart(history) {
        const placeholder = qs('#chartPlaceholder');
        if (!history || history.length === 0) {
            placeholder.textContent = 'لا توجد بيانات كافية لرسم المخطط.';
            return;
        }
        placeholder.style.display = 'none';
        
        const ctx = qs('#performanceChart').getContext('2d');
        let cumulativeProfit = 0;
        const chartData = history.map(trade => {
            cumulativeProfit += trade.profit;
            return { x: new Date(trade.date), y: cumulativeProfit };
        });

        if (performanceChartInstance) {
            performanceChartInstance.destroy();
        }

        performanceChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'الأرباح التراكمية (%)',
                    data: chartData,
                    borderColor: 'var(--accent)',
                    backgroundColor: 'rgba(74, 114, 255, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    x: { type: 'time', time: { unit: 'day' }, grid: { color: 'rgba(138, 160, 200, 0.1)' }, ticks: { color: 'var(--muted)' } },
                    y: { grid: { color: 'rgba(138, 160, 200, 0.1)' }, ticks: { color: 'var(--muted)', callback: value => `${value.toFixed(1)}%` } }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    async function initializeDashboard() {
        try {
            const [baseRes, signalsRes, perfRes] = await Promise.all([
                fetch('/api/dashboard_data'),
                fetch('/api/open_signals'),
                fetch('/api/performance_stats')
            ]);
            if (!baseRes.ok || !signalsRes.ok || !perfRes.ok) throw new Error('Network response was not ok.');
            
            const baseData = await baseRes.json();
            const signalsData = await signalsRes.json();
            const perfData = await perfRes.json();

            // Base Data
            qs('#serverTime').textContent = new Date(baseData.server_time).toLocaleTimeString('ar-EG');
            qs('#toggleTrading').checked = !!baseData.trading_enabled;
            qs('#balance').textContent = formatNumber(baseData.usdt_balance, 2);
            
            // Trading Mode
            const isPaper = baseData.paper_trading_mode;
            qs('#modePaper').classList.toggle('active', isPaper);
            qs('#modeReal').classList.toggle('active', !isPaper);

            // Settings
            qs('#qualityFilter').value = baseData.min_signal_quality;
            qs('#qualityValue').textContent = baseData.min_signal_quality;
            qs('#tradeAmount').value = parseFloat(baseData.fixed_trade_amount);
            qs('#tradeAmountValue').textContent = `$${parseFloat(baseData.fixed_trade_amount).toFixed(2)}`;

            // Market State & Logs
            updateMarketTrends(baseData.market_state);
            qs('#rejections').innerHTML = '';
            baseData.rejections.forEach(r => addLogEntry('#rejections', r, 30));
            qs('#events').innerHTML = '';
            baseData.notifications.forEach(n => addLogEntry('#events', n, 20));

            // Open Signals
            openSignals = signalsData.signals.reduce((acc, s) => { acc[s.id] = s; return acc; }, {});
            renderAllSignals();
            
            // Performance Data
            renderPerformanceMetrics(perfData);
            renderPerformanceChart(perfData.profit_history);

        } catch (error) {
            console.error("Failed to load initial dashboard data:", error);
            qs('#signals').innerHTML = '<p>فشل تحميل البيانات. حاول تحديث الصفحة.</p>';
        }
    }

    function setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const socket = new WebSocket(`${protocol}//${window.location.host}/ws`);
        socket.onopen = () => console.log("WebSocket connected");
        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            switch (data.type) {
                case 'price_update': updatePrices(data.payload); break;
                case 'new_signal':
                    openSignals[data.payload.id] = data.payload;
                    renderAllSignals();
                    break;
                case 'trade_closed':
                    delete openSignals[data.payload.signal_id];
                    renderAllSignals();
                    break;
                case 'new_notification': addLogEntry('#events', data.payload, 20); break;
                case 'new_rejection': addLogEntry('#rejections', data.payload, 30); break;
                case 'market_state_update': updateMarketTrends(data.payload); break;
                case 'full_dashboard_update':
                    qs('#balance').textContent = formatNumber(data.payload.usdt_balance, 2);
                    break;
            }
        };
        socket.onclose = () => { console.log("WebSocket closed, reconnecting..."); setTimeout(setupWebSocket, 3000); };
        socket.onerror = (error) => console.error("WebSocket error:", error);
    }
    
    // Event Listeners
    qs('#toggleTrading').addEventListener('change', () => fetch('/toggle_trading', { method: 'POST' }));
    
    const setTradingMode = (isPaper) => {
        fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ paper_trading_mode: isPaper })
        });
        qs('#modePaper').classList.toggle('active', isPaper);
        qs('#modeReal').classList.toggle('active', !isPaper);
    };
    qs('#modePaper').addEventListener('click', () => setTradingMode(true));
    qs('#modeReal').addEventListener('click', () => setTradingMode(false));

    const debouncedQualityUpdate = debounce((value) => {
        fetch('/api/signal_quality', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ min_quality: parseInt(value) })
        });
    }, 500);
    qs('#qualityFilter').addEventListener('input', function() {
        qs('#qualityValue').textContent = this.value;
        debouncedQualityUpdate(this.value);
    });
    
    const debouncedAmountUpdate = debounce((value) => {
        fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ FIXED_TRADE_AMOUNT_USDT: parseFloat(value) })
        });
    }, 500);
    qs('#tradeAmount').addEventListener('input', function() {
        qs('#tradeAmountValue').textContent = `$${parseFloat(this.value).toFixed(2)}`;
        debouncedAmountUpdate(this.value);
    });

    initializeDashboard();
    setupWebSocket();
});
</script>
</body>
</html>
"""

# --- دوال WebSocket ---
def broadcast(data: Dict):
    with ws_clients_lock:
        clients_to_remove = []
        for client in ws_clients:
            try:
                client.send(json.dumps(data, cls=NpEncoder))
            except Exception as e:
                logger.warning(f"WebSocket send failed, removing client: {e}")
                clients_to_remove.append(client)
        
        for client in clients_to_remove:
            try:
                ws_clients.remove(client)
            except ValueError:
                pass

def get_dashboard_payload() -> Dict:
    with trading_status_lock: trading_enabled = is_trading_enabled
    with trading_mode_lock: is_paper_mode = paper_trading_mode
    with balance_lock: current_balance = usdt_balance
    with notifications_lock: notifications = list(notifications_cache)
    with rejection_logs_lock: rejections = list(rejection_logs_cache)
    with market_state_lock: market_state = dict(current_market_state)
    with min_quality_lock: min_quality = MIN_SIGNAL_QUALITY
    with fixed_trade_amount_lock: fixed_amount = FIXED_TRADE_AMOUNT_USDT

    return {
        "trading_enabled": trading_enabled,
        "paper_trading_mode": is_paper_mode,
        "usdt_balance": current_balance,
        "notifications": notifications,
        "rejections": rejections,
        "market_state": market_state,
        "min_signal_quality": min_quality,
        "fixed_trade_amount": fixed_amount,
        "server_time": datetime.now(timezone.utc).isoformat()
    }

# --- دوال تهيئة الخدمات وقاعدة البيانات (بدون تغيير) ---
def optimize_database():
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            logger.info("[DB] Optimizing database with indexes...")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol_status ON signals(symbol, status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_notifications_timestamp ON notifications(timestamp);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_status_closed_at ON signals(status, closed_at);")
            conn.commit()
            logger.info("✅ [DB] Database indexes optimized successfully.")
    except Exception as e:
        logger.error(f"❌ [DB] Error optimizing database: {e}")
        if conn: conn.rollback()

def column_exists(cursor, table_name, column_name):
    cursor.execute("SELECT 1 FROM information_schema.columns WHERE table_name = %s AND column_name = %s", (table_name, column_name))
    return cursor.fetchone() is not None

def init_db(retries: int = 5, base_delay: int = 5) -> None:
    global conn
    logger.info("[DB] Initializing database connection...")
    db_url_to_use = DB_URL
    if 'postgres' in db_url_to_use and 'sslmode' not in db_url_to_use:
        db_url_to_use += f"{'?' if '?' not in db_url_to_use else '&'}sslmode=require"
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(db_url_to_use, connect_timeout=15, cursor_factory=RealDictCursor)
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION NOT NULL,
                        stop_loss DOUBLE PRECISION NOT NULL, status TEXT DEFAULT 'open',
                        closing_price DOUBLE PRECISION, closed_at TIMESTAMP, profit_percentage DOUBLE PRECISION,
                        strategy_name TEXT, signal_details JSONB, is_real_trade BOOLEAN DEFAULT FALSE,
                        quantity DOUBLE PRECISION, closing_reason TEXT, order_id TEXT
                    );
                """)
                cur.execute("CREATE TABLE IF NOT EXISTS notifications (id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(), type TEXT NOT NULL, message TEXT NOT NULL);")
                columns_to_add = {
                    "target_price_1": "DOUBLE PRECISION", "target_price_2": "DOUBLE PRECISION",
                    "initial_quantity": "DOUBLE PRECISION"
                }
                for col, col_type in columns_to_add.items():
                    if not column_exists(cur, 'signals', col):
                        cur.execute(sql.SQL("ALTER TABLE signals ADD COLUMN {} {}").format(sql.Identifier(col), sql.SQL(col_type)))
                        logger.info(f"✅ [DB] Added missing column '{col}' to 'signals' table.")
            conn.commit()
            logger.info("✅ [DB] Database connection and schema updated successfully.")
            optimize_database()
            return
        except Exception as e:
            logger.error(f"❌ [DB] Error during initialization (Attempt {attempt + 1}/{retries}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info(f"[DB] Retrying connection in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.critical("❌ [DB] Failed to connect to the database after all retries. Exiting.")
                exit(1)

def check_db_connection() -> bool:
    global conn
    if conn is None or conn.closed != 0:
        logger.warning("[DB] Connection is None or closed. Re-initializing...")
        init_db()
    try:
        if conn and conn.closed == 0:
            with conn.cursor() as cur: cur.execute("SELECT 1;")
            return True
        logger.warning("[DB] Connection check failed. It might still be closed.")
        return False
    except (OperationalError, InterfaceError) as e:
        logger.error(f"[DB] Connection lost ({e}). Attempting to reconnect...")
        init_db()
        return conn is not None and conn.closed == 0

def init_redis() -> None:
    global redis_client
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] Connected successfully.")
    except redis.exceptions.ConnectionError as e:
        logger.warning(f"⚠️ [Redis] Connection failed: {e}.")
        redis_client = None

# --- دوال المساعدة والإشعارات (بدون تغيير) ---
def log_and_notify(level: str, message: str, notification_type: str):
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error}
    log_methods.get(level.lower(), logger.info)(message)
    if not check_db_connection() or not conn: return
    try:
        new_notification = {"timestamp": datetime.now(timezone.utc).isoformat(), "type": notification_type, "message": message}
        with notifications_lock: notifications_cache.appendleft(new_notification)
        with conn.cursor() as cur: cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
        broadcast({"type": "new_notification", "payload": new_notification})
    except Exception as e:
        logger.error(f"❌ [DB] Failed to save notification: {e}")
        if conn: conn.rollback()

def log_rejection(symbol: str, reason_key: str, details: Optional[Dict] = None):
    try:
        reason_ar = REJECTION_REASONS_AR.get(reason_key, reason_key)
        if details:
            details_str = ", ".join([f"{k}: {v}" for k, v in details.items()])
            reason_ar = f"{reason_ar} ({details_str})"
        log_entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol, "reason": reason_ar}
        with rejection_logs_lock: rejection_logs_cache.appendleft(log_entry)
        broadcast({"type": "new_rejection", "payload": log_entry})
    except Exception as e:
        logger.error(f"❌ [Log Rejection] Error logging rejection for {symbol}: {e}", exc_info=True)
        
def send_enhanced_telegram_message(message: str, force: bool = False):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    max_length = 4096
    messages = [message[i:i+max_length] for i in range(0, len(message), max_length)]
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for msg in messages:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown", "disable_web_page_preview": True}
        for attempt in range(3):
            try:
                r = requests.post(url, data=payload, timeout=10)
                if r.status_code == 429:
                    retry_after = int(r.json().get("parameters", {}).get("retry_after", 1))
                    time.sleep(min(5, retry_after)); continue
                if r.ok: break
                else: logger.warning(f"[Telegram] HTTP {r.status_code}: {r.text}")
            except requests.exceptions.RequestException as e:
                if attempt == 2: logger.error(f"❌ [Telegram] Failed to send message after retries: {e}")
                time.sleep(1.5)

def send_trade_open_notification(symbol: str, strategy_name: str, entry_price: float, stop_loss: float,
                                target1: float, target2: float, quantity: float, is_real: bool,
                                quality_score: int, atr_percent: float, notional_value: float):
    trade_type = "حقيقية" if is_real else "ورقية"
    emoji = "🔥" if is_real else "📊"
    message = (
        f"{emoji} *صفقة {trade_type} جديدة*\n\n"
        f"*العملة:* `{symbol}`\n"
        f"*الاستراتيجية:* `{STRATEGY_NAMES.get(strategy_name, strategy_name)}`\n"
        f"*جودة الإشارة:* `{quality_score}/100`\n"
        f"*تقلب السوق:* `{atr_percent:.2f}%`\n\n"
        f"*سعر الدخول:* `{entry_price:.4f}`\n"
        f"*وقف الخسارة:* `{stop_loss:.4f}`\n"
        f"*الهدف الأول:* `{target1:.4f}`\n"
        f"*الهدف الثاني:* `{target2:.4f}`\n\n"
        f"*الكمية:* `{quantity:.4f}`\n"
        f"*قيمة الصفقة:* `${notional_value:.2f}`\n"
        f"*نسبة المخاطرة:* `{((entry_price - stop_loss) / entry_price * 100):.2f}%`\n"
        f"*نسبة الربح المحتملة 1:* `{((target1 - entry_price) / entry_price * 100):.2f}%`\n"
        f"*نسبة الربح المحتملة 2:* `{((target2 - entry_price) / entry_price * 100):.2f}%`"
    )
    send_enhanced_telegram_message(message, force=True)

def handle_socket_message(msg):
    global live_prices
    try:
        if msg and 'e' in msg and msg['e'] == 'error':
            logger.error(f"❌ [WebSocket] Error: {msg['m']}")
            return
        
        if isinstance(msg, list):
            price_updates = {}
            with live_prices_lock:
                for ticker in msg:
                    if 's' in ticker and 'c' in ticker:
                        symbol = ticker['s']
                        try:
                            price = float(ticker['c'])
                            live_prices[symbol] = price
                            price_updates[symbol] = price
                        except (ValueError, TypeError):
                            logger.warning(f"[WebSocket] Invalid price data for {symbol}: {ticker.get('c')}")
            
            if price_updates:
                broadcast({"type": "price_update", "payload": price_updates})
    except Exception as e:
        logger.error(f"❌ [WebSocket] Error processing message: {e}", exc_info=True)

def start_websocket():
    global ws_manager
    ws_manager = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    ws_manager.start()
    ws_manager.start_ticker_socket(callback=handle_socket_message)
    logger.info("✅ [WebSocket] Subscribed to ticker stream.")

def get_exchange_info_map() -> None:
    global exchange_info_map
    try:
        logger.info("[API] Fetching exchange info...")
        exchange_info_map = {s['symbol']: s for s in client.get_exchange_info()['symbols']}
        logger.info(f"[API] Exchange info map created with {len(exchange_info_map)} symbols.")
    except Exception as e:
        logger.error(f"❌ [API] Error fetching exchange info: {e}")

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    try:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        if not os.path.exists(file_path):
            logger.critical(f"❌ Symbol list file '{filename}' not found!"); return []
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        if not exchange_info_map: get_exchange_info_map()
        active = {s for s, info in exchange_info_map.items() if info.get('quoteAsset') == 'USDT' and info.get('status') == 'TRADING'}
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ Found {len(validated)} valid symbols for trading.")
        return validated
    except Exception as e:
        logger.error(f"❌ [Symbols] Error validating symbols: {e}"); return []

def get_available_balance(is_real_trade: bool) -> float:
    """
    [جديد] الاستعلام عن الرصيد المتاح للتداول.
    للتداول الحقيقي: يجلب الرصيد الحر من منصة Binance.
    للتداول الورقي: يستخدم الرصيد المحاكى.
    """
    if is_real_trade:
        try:
            if not client:
                logger.error("❌ Binance client not initialized. Cannot fetch real balance.")
                return 0.0
            balance_info = client.get_asset_balance(asset='USDT')
            balance = float(balance_info.get('free', 0.0))
            logger.info(f"💰 [Real Balance] Fetched available balance: {balance:.2f} USDT")
            return balance
        except Exception as e:
            logger.error(f"❌ Failed to fetch REAL USDT balance: {e}. Returning 0.")
            return 0.0
    else:  # Paper trading
        with balance_lock:
            # For paper mode, use the globally managed simulated balance
            return usdt_balance
# --- نهاية دوال المساعدة ---

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    time.sleep(API_REQUEST_DELAY)
    try:
        klines = client.get_historical_klines(symbol, interval, f"{days} day ago UTC")
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna().astype(float)
    except Exception as e:
        logger.error(f"❌ [Data] Error fetching data for {symbol}: {e}"); return None

# --- حساب المؤشرات (بدون تغيير) ---
def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    
    df_calc['sma7'] = df_calc['close'].rolling(window=7).mean()
    df_calc['sma200'] = df_calc['close'].rolling(window=200).mean()

    df_calc['ema9'] = df_calc['close'].ewm(span=9, adjust=False).mean()
    df_calc['ema13'] = df_calc['close'].ewm(span=13, adjust=False).mean()
    df_calc['ema21'] = df_calc['close'].ewm(span=21, adjust=False).mean()
    df_calc['ema34'] = df_calc['close'].ewm(span=34, adjust=False).mean()
    df_calc['ema50'] = df_calc['close'].ewm(span=50, adjust=False).mean()
    df_calc['ema100'] = df_calc['close'].ewm(span=100, adjust=False).mean()
    df_calc['ema200'] = df_calc['close'].ewm(span=200, adjust=False).mean()
    
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    df_calc['atr'] = tr.ewm(span=14, adjust=False).mean()
    df_calc['atr_percent'] = (df_calc['atr'] / df_calc['close'].replace(0, 1e-9)) * 100
    up_move = df_calc['high'].diff()
    down_move = -df_calc['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=14, adjust=False).mean()
    
    delta = df_calc['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=7).mean()
    avg_loss = loss.rolling(window=7).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df_calc['rsi'] = 100 - (100 / (1 + rs))
    
    bb_middle = df_calc['close'].rolling(window=20).mean()
    bb_std = df_calc['close'].rolling(window=20).std()
    df_calc['bb_middle'] = bb_middle
    df_calc['bb_lower'] = bb_middle - (bb_std * 2)
    df_calc['bb_upper'] = bb_middle + (bb_std * 2)
    df_calc['bb_width'] = (df_calc['bb_upper'] - df_calc['bb_lower']) / df_calc['bb_middle'].replace(0, 1e-9)
    
    exp1 = df_calc['close'].ewm(span=8, adjust=False).mean()
    exp2 = df_calc['close'].ewm(span=17, adjust=False).mean()
    df_calc['macd'] = exp1 - exp2
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()
    df_calc['macd_hist'] = df_calc['macd'] - df_calc['macd_signal']
    
    low_14 = df_calc['low'].rolling(14).min()
    high_14 = df_calc['high'].rolling(14).max()
    high_low_range = high_14 - low_14
    meaningful_range = high_low_range > (df_calc['close'] * 0.0001)
    df_calc['stoch_k'] = np.where(meaningful_range, 100 * ((df_calc['close'] - low_14) / high_low_range.replace(0, 1e-9)), 50)
    df_calc['stoch_d'] = df_calc['stoch_k'].rolling(3).mean()
    
    df_calc['vwap'] = (df_calc['close'] * df_calc['volume']).cumsum() / df_calc['volume'].cumsum()
    return df_calc

# --- إدارة الكاش والإعدادات (بدون تغيير) ---
def load_open_signals_to_cache():
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status IN ('open', 'updated');")
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in cur.fetchall(): open_signals_cache[signal['id']] = dict(signal)
            logger.info(f"✅ [Cache] Loaded {len(open_signals_cache)} open signals.")
    except Exception as e:
        logger.error(f"❌ [Cache] Failed to load open signals: {e}")

def load_settings_from_redis():
    global FIXED_TRADE_AMOUNT_USDT, MAX_OPEN_TRADES, USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY, USE_MOMENTUM_VOLATILITY_STRATEGY, USE_ELLIOTT_WAVE_STRATEGY, USE_RANGE_REVERSAL_STRATEGY, paper_trading_mode, MIN_SIGNAL_QUALITY
    if not redis_client: return
    try:
        settings_data = redis_client.get('trading_settings')
        if settings_data:
            settings = json.loads(settings_data)
            with fixed_trade_amount_lock: FIXED_TRADE_AMOUNT_USDT = settings.get('FIXED_TRADE_AMOUNT_USDT', 5.0)
            MAX_OPEN_TRADES = settings.get('MAX_OPEN_TRADES', 3)
            with trading_mode_lock: paper_trading_mode = settings.get('paper_trading_mode', True)
            
        quality_settings_data = redis_client.get('signal_quality_settings')
        if quality_settings_data:
            quality_settings = json.loads(quality_settings_data)
            with min_quality_lock: MIN_SIGNAL_QUALITY = quality_settings.get('min_quality', 70)

        strategies_data = redis_client.get('strategy_settings')
        if strategies_data:
            strategies = json.loads(strategies_data)
            USE_BB_STOCH_STRATEGY = strategies.get('USE_BB_STOCH_STRATEGY', True)
            USE_MACD_EMA_STRATEGY = strategies.get('USE_MACD_EMA_STRATEGY', True)
            USE_EMA_RSI_STRATEGY = strategies.get('USE_EMA_RSI_STRATEGY', True)
            USE_PULLBACK_STRATEGY = strategies.get('USE_PULLBACK_STRATEGY', True)
            USE_MOMENTUM_VOLATILITY_STRATEGY = strategies.get('USE_MOMENTUM_VOLATILITY_STRATEGY', True)
            USE_ELLIOTT_WAVE_STRATEGY = strategies.get('USE_ELLIOTT_WAVE_STRATEGY', True)
            USE_RANGE_REVERSAL_STRATEGY = strategies.get('USE_RANGE_REVERSAL_STRATEGY', True)

        logger.info("✅ [Redis] Successfully loaded settings from Redis.")
    except Exception as e:
        logger.error(f"❌ [Redis] Error loading settings: {e}")

# --- [جديد] نظام تقييم جودة الإشارة ---
def calculate_signal_quality(df: pd.DataFrame, mtf_trend: Dict) -> int:
    score = 50  # Base score
    last = df.iloc[-1]
    
    # 1. قوة الاتجاه (ADX) - مهم لاستراتيجيات الاتجاه
    if last.get('adx', 0) > 25:
        score += 15
    elif last.get('adx', 0) > 20:
        score += 5
        
    # 2. تأكيد حجم التداول
    volume_ma = df['volume'].rolling(window=20).mean().iloc[-1]
    if last.get('volume', 0) > volume_ma * 1.2:
        score += 15
        
    # 3. مؤشر القوة النسبية (RSI) - تجنب مناطق الشراء المفرط
    if 40 < last.get('rsi', 50) < 70:
        score += 10
        
    # 4. توافق الإطارات الزمنية (MTF)
    if mtf_trend.get('1h') == 'bullish':
        score += 10
    if mtf_trend.get('15m') == 'bullish':
        score += 10
        
    # 5. تأكيد الاتجاه طويل الأمد (SMA200)
    if last.get('close') > last.get('sma200', float('inf')):
        score += 10
        
    return min(100, int(score))

def get_wave_retracement(df: pd.DataFrame) -> float:
    # دالة مساعدة لحساب تصحيح الموجة الأخيرة
    try:
        highs = df['high'].values
        lows = df['low'].values
        peaks_idx = argrelextrema(highs, np.greater, order=5)[0]
        troughs_idx = argrelextrema(lows, np.less, order=5)[0]
        
        if len(peaks_idx) < 1 or len(troughs_idx) < 2: return 999.0
        
        last_trough_idx = troughs_idx[-1]
        prev_peak_idx = peaks_idx[peaks_idx < last_trough_idx][-1]
        prev_trough_idx = troughs_idx[troughs_idx < prev_peak_idx][-1]

        wave_start_price = lows[prev_trough_idx]
        wave_end_price = highs[prev_peak_idx]
        retracement_price = lows[last_trough_idx]

        wave_height = wave_end_price - wave_start_price
        if wave_height <= 0: return 999.0
        
        retracement = (wave_end_price - retracement_price) / wave_height
        return retracement
    except Exception:
        return 999.0

# --- [معدل] الفلاتر الديناميكية ونظام السوق ---
def check_bb_stoch_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    bb_width = df['bb_width']
    dynamic_bb_threshold = bb_width.rolling(20).mean() * 1.1 # تخفيف
    stoch_threshold = 20 if atr_percent > 3.0 else 15 # تخفيف
    volume_ma = df['volume'].rolling(20).mean()
    volume_multiplier = 1.0 + (atr_percent / 120) # تخفيف
    return {
        'bb_width_ok': bb_width.iloc[-1] > dynamic_bb_threshold.iloc[-1],
        'stoch_ok': last_row['stoch_k'] > stoch_threshold,
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * volume_multiplier
    }

def check_macd_ema_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    adx_threshold = 20 if atr_percent > 2.5 else 15 # تخفيف
    volume_ma = df['volume'].rolling(20).mean()
    volatility_adjusted_volume = volume_ma * (1 + atr_percent / 80) # تخفيف
    macd_momentum = df['macd_hist'].diff()
    momentum_threshold = macd_momentum.rolling(10).std() * 0.1 # تخفيف كبير
    return {
        'adx_ok': last_row['adx'] > adx_threshold,
        'volume_ok': last_row['volume'] > volatility_adjusted_volume.iloc[-1],
        'momentum_ok': macd_momentum.iloc[-1] > momentum_threshold.iloc[-1],
    }

def check_ema_rsi_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    adx = last_row.get('adx', 0)
    if adx > 25:
        rsi_lower, rsi_upper = 40, 80 # توسيع النطاق
    else:
        rsi_lower, rsi_upper = 45, 75 # توسيع النطاق
    ema_spread = (df['ema9'] - df['ema21']) / df['ema21'].replace(0, 1e-9)
    dynamic_ema_threshold = ema_spread.rolling(20).std() * 1.5 # تخفيف
    volume_ma = df['volume'].rolling(20).mean()
    trend_strength_multiplier = 1 + (adx / 120) # تخفيف
    return {
        'rsi_ok': rsi_lower < last_row['rsi'] < rsi_upper,
        'ema_ok': ema_spread.iloc[-1] > dynamic_ema_threshold.iloc[-1],
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * trend_strength_multiplier,
    }
    
def check_pullback_dynamic_filters(df: pd.DataFrame, mtf_trend: Dict) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    pullback_depth = 0.038 if atr_percent > 2.0 else 0.022 # زيادة العمق المسموح به قليلاً
    if mtf_trend.get('15m') == 'bullish' and mtf_trend.get('1h') == 'bullish':
        pullback_depth *= 1.3 # زيادة المرونة في الاتجاهات القوية
    recent_low = df['low'].tail(5).min()
    recovery_threshold = recent_low * (1 + pullback_depth)
    volume_ma = df['volume'].rolling(20).mean()
    recovery_volume_multiplier = 1.05 + (atr_percent / 100) # تخفيف
    return {
        'recovery_ok': last_row['close'] > recovery_threshold,
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * recovery_volume_multiplier,
    }
    
def check_momentum_volatility_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = df['atr_percent']
    volatility_ma = atr_percent.rolling(20).mean()
    volatility_std = atr_percent.rolling(20).std()
    
    dynamic_vol_min = volatility_ma.iloc[-1] - (volatility_std.iloc[-1] * 1.5)
    dynamic_vol_max = volatility_ma.iloc[-1] + (volatility_std.iloc[-1] * 1.5)
    
    momentum_indicators = [
        last_row['macd_hist'],
        last_row['rsi'] - 50,
        (last_row['close'] - last_row['ema21']) / last_row['ema21']
    ]
    momentum_score = sum(momentum_indicators) / len(momentum_indicators)
    
    adx_ma = df['adx'].rolling(20).mean()
    dynamic_adx_threshold = adx_ma.iloc[-1] * 0.85
    
    return {
        'volatility_ok': dynamic_vol_min <= atr_percent.iloc[-1] <= dynamic_vol_max,
        'momentum_ok': momentum_score > 0,
        'adx_ok': last_row['adx'] > dynamic_adx_threshold,
    }


def check_elliott_wave_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    atr_percent = last_row.get('atr_percent', 0)
    if atr_percent > 2.5:
        fib_min, fib_max = 0.3, 0.786
    else:
        fib_min, fib_max = 0.2, 0.7 # توسيع النطاق
    volume_ma = df['volume'].rolling(20).mean()
    wave_volume_multiplier = 1.1 + (atr_percent / 50) # تخفيف
    macd_momentum = df['macd_hist'].rolling(5).mean()
    momentum_threshold = macd_momentum.rolling(20).std() * 0.2 # تخفيف
    return {
        'fibonacci_ok': fib_min <= get_wave_retracement(df) <= fib_max,
        'volume_ok': last_row['volume'] > volume_ma.iloc[-1] * wave_volume_multiplier,
        'momentum_ok': macd_momentum.iloc[-1] > momentum_threshold.iloc[-1],
    }

def check_range_reversal_dynamic_filters(df: pd.DataFrame) -> Dict:
    last_row = df.iloc[-1]
    adx_ok = last_row.get('adx', 99) < 25 # توسيع
    atr_percent = last_row.get('atr_percent', 0)
    rsi_threshold = 38 if atr_percent < 2.5 else 42 # توسيع
    rsi_ok = last_row.get('rsi', 50) < rsi_threshold
    return {'adx_ok': adx_ok, 'rsi_ok': rsi_ok}
    
# --- [معدل] الفلاتر العامة ---
def check_market_volatility_filter_enhanced(df: pd.DataFrame, symbol: str = "Unknown") -> bool:
    if 'atr_percent' not in df.columns or df['atr_percent'].isnull().all():
        log_rejection(symbol, "Market Volatility Filter Failed", {"reason": "No ATR data"})
        return False
    
    last_atr_percent = float(df.iloc[-1].get('atr_percent', 0))
    ATR_PERCENT_MIN = 1.2 # [تعديل] تخفيض الحد الأدنى للسماح بالعملات الأقل تقلباً
    ATR_PERCENT_MAX = 7.0 # [تعديل] زيادة الحد الأعلى للسماح بالعملات الأكثر تقلباً
    
    if not (ATR_PERCENT_MIN <= last_atr_percent <= ATR_PERCENT_MAX):
        log_rejection(symbol, "Market Volatility Filter Failed", {
            "atr": f"{last_atr_percent:.2f}%",
            "range": f"({ATR_PERCENT_MIN:.2f}-{ATR_PERCENT_MAX:.2f})%"
        })
        return False
    return True

# --- دوال تحديد وقف الخسارة والهدف (بدون تغيير) ---
def calculate_dynamic_stop_loss(df: pd.DataFrame, entry_price: float, strategy_name: str) -> float:
    last = df.iloc[-1]
    atr_value = last.get('atr', 0)
    if strategy_name == "BB_Stoch_Strategy": stop_loss = min(df['low'].tail(3).min() * 0.995, entry_price - (atr_value * 1.5))
    elif strategy_name == "MACD_EMA_Strategy": stop_loss = min(last['ema21'], entry_price - (atr_value * 2.0))
    elif strategy_name == "EMA_RSI_Strategy": stop_loss = min(last['ema21'], entry_price - (atr_value * 1.8))
    elif strategy_name == "Pullback_Strategy": stop_loss = min(df['low'].tail(5).min() * 0.995, entry_price - (atr_value * 1.5))
    elif strategy_name == "Momentum_Volatility_Strategy": stop_loss = min(last['ema21'], entry_price - (atr_value * 2.2))
    elif strategy_name == "Range_Reversal_Strategy": stop_loss = min(df['low'].tail(5).min() * 0.99, entry_price - (atr_value * 1.2))
    else: stop_loss = entry_price - (atr_value * 2.0)
    max_stop_distance = entry_price * 0.05
    if entry_price - stop_loss > max_stop_distance: stop_loss = entry_price - max_stop_distance
    return stop_loss

def calculate_dynamic_take_profit(df: pd.DataFrame, entry_price: float, stop_loss: float, strategy_name: str) -> tuple:
    risk_amount = entry_price - stop_loss
    if risk_amount <= 0: return (entry_price * 1.02, entry_price * 1.04)
    if strategy_name == "Range_Reversal_Strategy":
        return df.iloc[-1].get('bb_middle', entry_price * 1.02), df.iloc[-1].get('bb_upper', entry_price * 1.04)
    rr_map = {
        "BB_Stoch_Strategy": (2.5, 4.0), "MACD_EMA_Strategy": (2.0, 3.5),
        "EMA_RSI_Strategy": (2.2, 3.8), "Pullback_Strategy": (2.3, 4.0),
        "Momentum_Volatility_Strategy": (1.8, 3.2), "Elliott_Wave_Strategy": (2.5, 4.5)
    }
    rr1, rr2 = rr_map.get(strategy_name, (2.0, 3.5))
    return entry_price + (risk_amount * rr1), entry_price + (risk_amount * rr2)

# --- [معدل] استراتيجيات التداول مع الفلاتر الديناميكية ---
def check_ema_rsi_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 200: return False
    last = df.iloc[-1]
    
    is_mtf_bullish = mtf_trend.get('15m') == 'bullish' or mtf_trend.get('1h') == 'bullish'
    is_long_term_bullish = last['ema50'] > last['ema200'] and last['close'] > last['ema9']
    
    if not (is_mtf_bullish or is_long_term_bullish):
        log_rejection(symbol_name, "Trend: Not bullish on MTF or long-term"); return False

    filters = check_ema_rsi_dynamic_filters(df)
    if not filters['rsi_ok']: log_rejection(symbol_name, "DYN_RSI_OOR"); return False
    if not filters['ema_ok']: log_rejection(symbol_name, "DYN_EMA_SPREAD_LOW"); return False
    if not filters['volume_ok']: log_rejection(symbol_name, "DYN_VOLUME_LOW"); return False
    return True

def check_bb_stoch_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 50: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    
    is_mtf_bullish = mtf_trend.get('15m') == 'bullish' or mtf_trend.get('1h') == 'bullish'
    is_long_term_bullish = last['close'] > last['ema50']
    
    if not (is_mtf_bullish or is_long_term_bullish):
        log_rejection(symbol_name, "Trend: Not bullish on MTF or long-term"); return False

    if not ((df['low'].tail(3) <= df['bb_lower'].tail(3)).any() and last['close'] > last['bb_lower']): return False
    if not ((prev['stoch_k'] < 30) and (last['stoch_k'] > prev['stoch_k'])): return False

    filters = check_bb_stoch_dynamic_filters(df)
    if not filters['bb_width_ok']: log_rejection(symbol_name, "DYN_BB_WIDTH_LOW"); return False
    if not filters['stoch_ok']: log_rejection(symbol_name, "DYN_STOCH_LOW"); return False
    if not filters['volume_ok']: log_rejection(symbol_name, "DYN_VOLUME_LOW"); return False
    return True

def check_macd_ema_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 200: return False
    last = df.iloc[-1]
    
    is_mtf_bullish = mtf_trend.get('15m') == 'bullish' or mtf_trend.get('1h') == 'bullish' # شرط "أو"
    is_long_term_bullish = last['close'] > last['sma200']
    
    if not (is_mtf_bullish or is_long_term_bullish):
        log_rejection(symbol_name, "Trend: Not bullish on MTF or long-term"); return False

    hist = df['macd_hist'].tail(4).values
    if not (last['macd'] > 0 and last['macd_hist'] > 0 and hist[3] > hist[2] > hist[1]): return False
        
    filters = check_macd_ema_dynamic_filters(df)
    if not filters['adx_ok']: log_rejection(symbol_name, "DYN_ADX_LOW", {'adx': f"{last['adx']:.1f}"}); return False
    if not filters['volume_ok']: log_rejection(symbol_name, "DYN_VOLUME_LOW"); return False
    if not filters['momentum_ok']: log_rejection(symbol_name, "DYN_MACD_MOMENTUM_LOW"); return False
    return True

def check_pullback_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 200: return False
    last = df.iloc[-1]
    
    is_mtf_bullish = mtf_trend.get('15m') == 'bullish' or mtf_trend.get('1h') == 'bullish'
    is_long_term_bullish = last['ema21'] > last['ema50'] > last['ema200']
    
    if not (is_mtf_bullish or is_long_term_bullish):
        log_rejection(symbol_name, "Trend: Not bullish on MTF or long-term"); return False

    if not (df['low'].tail(3) <= df['ema21'].tail(3)).any(): return False

    filters = check_pullback_dynamic_filters(df, mtf_trend)
    if not filters['recovery_ok']: log_rejection(symbol_name, "DYN_RECOVERY_FAIL"); return False
    if not filters['volume_ok']: log_rejection(symbol_name, "DYN_VOLUME_LOW"); return False
    return True

def check_momentum_volatility_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 50: return False
    last = df.iloc[-1]
    
    is_mtf_bullish = mtf_trend.get('15m') == 'bullish' or mtf_trend.get('1h') == 'bullish'
    is_long_term_bullish = last['ema9'] > last['ema21'] > last['ema50']
    
    if not (is_mtf_bullish or is_long_term_bullish):
        log_rejection(symbol_name, "Trend: Not bullish on MTF or long-term"); return False
        
    filters = check_momentum_volatility_dynamic_filters(df)
    if not filters['volatility_ok']: log_rejection(symbol_name, "DYN_VOLATILITY_OOR"); return False
    if not filters['momentum_ok']: log_rejection(symbol_name, "DYN_MOMENTUM_SCORE_LOW"); return False
    if not filters['adx_ok']: log_rejection(symbol_name, "DYN_ADX_LOW"); return False
    return True

def check_elliott_wave_strategy_enhanced(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 100: return False
    last = df.iloc[-1]
    
    is_mtf_bullish = mtf_trend.get('15m') == 'bullish' or mtf_trend.get('1h') == 'bullish' # شرط "أو"
    is_long_term_bullish = last['ema50'] > last['ema200']

    if not (is_mtf_bullish or is_long_term_bullish):
        log_rejection(symbol_name, "Trend: Not bullish on MTF or long-term"); return False
    
    if last['adx'] < 20: return False # تخفيف
    if last['macd'] <= 0: return False
        
    filters = check_elliott_wave_dynamic_filters(df)
    if not filters['fibonacci_ok']: log_rejection(symbol_name, "DYN_FIB_RETRACEMENT_OOR"); return False
    if not filters['volume_ok']: log_rejection(symbol_name, "DYN_VOLUME_LOW"); return False
    if not filters['momentum_ok']: log_rejection(symbol_name, "DYN_MACD_MOMENTUM_LOW"); return False
    return True

def check_range_reversal_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    symbol_name = getattr(df, 'name', 'Unknown')
    if len(df) < 50: return False
    last, prev = df.iloc[-1], df.iloc[-2]

    price_crossed_down = prev['low'] <= prev['bb_lower']
    price_rebounded_up = last['close'] > last['bb_lower']
    if not (price_crossed_down and price_rebounded_up): return False
    
    filters = check_range_reversal_dynamic_filters(df)
    if not filters['adx_ok']: log_rejection(symbol_name, "Range Reversal: Trend too strong"); return False
    if not filters['rsi_ok']: log_rejection(symbol_name, "Range Reversal: RSI not in oversold zone"); return False
    return True

# --- [معدل] دوال حساب حجم الصفقة والتداول ---
def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> Optional[Decimal]:
    try:
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info:
            logger.error(f"[{symbol}] No exchange info found for LOT_SIZE adjustment.")
            return None
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if not lot_size_filter:
            return Decimal(str(quantity))
        step_size = Decimal(lot_size_filter['stepSize'])
        min_qty = Decimal(lot_size_filter['minQty'])
        quantity_dec = Decimal(str(quantity))
        if quantity_dec < min_qty:
            log_rejection(symbol, "LOT_SIZE Filter Failed", {"reason": "Below minQty", "qty": f"{quantity_dec}", "min": f"{min_qty}"})
            return None
        adjusted_quantity = (quantity_dec - (quantity_dec % step_size))
        if adjusted_quantity < min_qty:
            log_rejection(symbol, "LOT_SIZE Filter Failed", {"reason": "Adjusted below minQty", "qty": f"{adjusted_quantity}", "min": f"{min_qty}"})
            return None
        return adjusted_quantity
    except Exception as e:
        logger.error(f"❌ [{symbol}] CRITICAL ERROR adjusting quantity: {e}", exc_info=True)
        return None

def calculate_position_size(symbol: str, entry_price: float, available_balance: float) -> Optional[Decimal]:
    """
    [معدل] حساب حجم الصفقة بناءً على مبلغ ثابت مع التحقق من قواعد المنصة.
    
    المنطق:
    1. استخدام قيمة `FIXED_TRADE_AMOUNT_USDT` المحددة في الإعدادات كحجم أساسي للصفقة.
    2. التحقق من أن الرصيد المتاح (`available_balance`) كافٍ لتغطية هذا المبلغ. إذا لم يكن كذلك، يتم رفض الصفقة.
    3. حساب الكمية الأولية (`initial_quantity`) بقسمة المبلغ الثابت على سعر الدخول.
    4. تعديل الكمية لتتوافق مع قواعد `LOT_SIZE` الخاصة بالعملة.
    5. التحقق من أن القيمة النهائية للصفقة (`notional_value`) تتجاوز الحد الأدنى `MIN_NOTIONAL`.
    6. إذا نجحت كل الشروط، يتم إرجاع الكمية المعدلة. وإلا، يتم إرجاع `None` ويتم تسجيل سبب الرفض.
    """
    with fixed_trade_amount_lock: fixed_amount = FIXED_TRADE_AMOUNT_USDT
    try:
        dec_entry = Decimal(str(entry_price))
        dec_balance = Decimal(str(available_balance))
        dec_fixed_amount = Decimal(str(fixed_amount))

        # الخطوة 1: التحقق من أن الرصيد المتاح يغطي حجم الصفقة الثابت المطلوب
        if dec_fixed_amount > dec_balance:
            log_rejection(symbol, "Insufficient Balance", {"required": f"${dec_fixed_amount:.2f}", "available": f"${dec_balance:.2f}"})
            return None
        
        if dec_entry <= 0: return None
        
        # الخطوة 2: حساب الكمية بناءً على المبلغ الثابت
        initial_quantity = dec_fixed_amount / dec_entry
        
        # الخطوة 3: تعديل الكمية حسب قواعد LOT_SIZE
        adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(initial_quantity))
        if adjusted_quantity is None or adjusted_quantity <= 0:
            # يتم تسجيل سبب الرفض داخل `adjust_quantity_to_lot_size`
            return None
            
        notional_value = adjusted_quantity * dec_entry
        
        # الخطوة 4: التحقق من الحد الأدنى لقيمة الصفقة (MIN_NOTIONAL)
        symbol_info = exchange_info_map.get(symbol)
        if symbol_info:
            min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] in ('MIN_NOTIONAL', 'NOTIONAL')), None)
            if min_notional_filter:
                min_notional = Decimal(min_notional_filter.get('minNotional', min_notional_filter.get('notional', '5.0')))
                if notional_value < min_notional:
                    log_rejection(symbol, "MinNotional Filter Failed", {"value": f"{notional_value:.2f}", "required": f"{min_notional}"})
                    return None
                    
        # فحص أخير للتأكد من أن القيمة النهائية لا تتجاوز الرصيد (نادر الحدوث لكنه آمن)
        if notional_value > dec_balance:
            log_rejection(symbol, "Insufficient Balance", {"required": f"{notional_value:.2f}", "available": f"${dec_balance:.2f}"})
            return None
            
        return adjusted_quantity
    except Exception as e:
        logger.error(f"❌ [{symbol}] Unhandled exception in calculate_position_size: {e}", exc_info=True)
        return None

# --- [معدل] دالة إنشاء الإشارة الرئيسية ---
def create_trade_signal(symbol: str, df: pd.DataFrame, strategy_name: str, mtf_trend: Dict):
    df.strategy = strategy_name 
    
    if not check_market_volatility_filter_enhanced(df, symbol): return

    quality_score = calculate_signal_quality(df, mtf_trend)
    with min_quality_lock: min_score = MIN_SIGNAL_QUALITY
    if quality_score < min_score:
        log_rejection(symbol, "Low Quality Signal", {"score": quality_score, "min_required": min_score})
        return
    logger.info(f"⭐ [Signal Quality] {symbol} ({strategy_name}): {quality_score}/100")

    entry_price = df.iloc[-1]['close']
    stop_loss_price = calculate_dynamic_stop_loss(df, entry_price, strategy_name)
    target_price_1, target_price_2 = calculate_dynamic_take_profit(df, entry_price, stop_loss_price, strategy_name)
    
    if stop_loss_price >= entry_price:
        log_rejection(symbol, "Invalid Position Size", {"entry": entry_price, "sl": stop_loss_price})
        return

    with trading_mode_lock: is_real = not paper_trading_mode
    signal_details = {"quality_score": quality_score, "atr_percent": df.iloc[-1].get('atr_percent', 0)}
    trade_levels = {"entry_price": entry_price, "stop_loss": stop_loss_price, "target_price_1": target_price_1, "target_price_2": target_price_2}
    
    # [تعديل] استدعاء دالة الاستعلام عن الرصيد المتاح بشكل صريح
    available_balance = get_available_balance(is_real)
    if available_balance <= 0.1: # تحقق من وجود رصيد كافي
        if is_real: log_rejection(symbol, "Insufficient Balance", {"reason": "Available balance is zero or could not be fetched."})
        return

    quantity_dec = calculate_position_size(symbol, entry_price, available_balance)
    if quantity_dec is None or quantity_dec <= 0:
        logger.error(f"❌ [{symbol}] Position size calculation failed. Trade rejected.")
        return
    
    notional_value = float(quantity_dec) * entry_price
    
    if is_real:
        try:
            logger.info(f"💰 [Real Trade] Placing LIVE MARKET BUY order for {quantity_dec} of {symbol}")
            order = client.create_order(symbol=symbol, side=Client.SIDE_BUY, type=Client.ORDER_TYPE_MARKET, quantity=str(quantity_dec))
            avg_fill_price = sum(Decimal(f['price']) * Decimal(f['qty']) for f in order.get('fills', [])) / max(sum(Decimal(f['qty']) for f in order.get('fills', [])), Decimal('1e-8')) if order.get('fills') else Decimal(str(entry_price))
            final_quantity = Decimal(order.get('executedQty', str(quantity_dec)))
            order_id = order.get('orderId', 'N/A')
            save_signal_to_db(symbol, float(avg_fill_price), trade_levels, strategy_name, True, float(final_quantity), {**signal_details, "avg_fill": float(avg_fill_price)}, order_id)
            send_trade_open_notification(symbol, strategy_name, float(avg_fill_price), stop_loss_price, target_price_1, target_price_2, float(final_quantity), is_real, quality_score, df.iloc[-1].get('atr_percent', 0), notional_value)
        except BinanceAPIException as e:
            logger.error(f"❌ [Real Trade] Binance API Error for {symbol}: {e}")
            send_enhanced_telegram_message(f"❌ *خطأ في صفقة حقيقية لـ {symbol}*\n`{e}`", force=True)
        except Exception as e:
            logger.error(f"❌ [Real Trade] CRITICAL ERROR creating real trade for {symbol}: {e}", exc_info=True)
    else: # Paper Trading
        save_signal_to_db(symbol, entry_price, trade_levels, strategy_name, False, float(quantity_dec), signal_details)
        send_trade_open_notification(symbol, strategy_name, entry_price, stop_loss_price, target_price_1, target_price_2, float(quantity_dec), is_real, quality_score, df.iloc[-1].get('atr_percent', 0), notional_value)

def save_signal_to_db(symbol: str, entry_price: float, trade_levels: Dict, strategy_name: str, is_real: bool, quantity: float, signal_details: Dict, order_id: Optional[str] = None):
    try:
        if not (check_db_connection() and conn): return
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price_1, target_price_2, stop_loss, status,
                                   strategy_name, is_real_trade, quantity, initial_quantity, signal_details, order_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
            """, (symbol, float(entry_price), float(trade_levels['target_price_1']), float(trade_levels['target_price_2']),
                  float(trade_levels['stop_loss']), 'open', strategy_name, is_real, float(quantity), float(quantity),
                  json.dumps(signal_details, cls=NpEncoder), order_id))
            new_id = cur.fetchone()['id']
        conn.commit()
        signal_data = {
            'id': new_id, 'symbol': symbol, 'entry_price': float(entry_price),
            'target_price_1': float(trade_levels['target_price_1']), 'target_price_2': float(trade_levels['target_price_2']),
            'stop_loss': float(trade_levels['stop_loss']), 'status': 'open', 'strategy_name': strategy_name,
            'is_real_trade': is_real, 'quantity': float(quantity), 'initial_quantity': float(quantity),
            'signal_details': signal_details, 'order_id': order_id
        }
        with signal_cache_lock: open_signals_cache[new_id] = signal_data
        broadcast({"type": "new_signal", "payload": signal_data})
    except Exception as e:
        logger.error(f"❌ [DB] CRITICAL ERROR saving signal for {symbol}: {e}", exc_info=True)
        if conn: conn.rollback()


# --- مسارات Flask ---
@app.route('/')
def dashboard(): return render_template_string(DASHBOARD_TEMPLATE_V2)

@app.route('/api/dashboard_data')
def dashboard_data():
    try: return jsonify(get_dashboard_payload())
    except Exception as e:
        logger.error(f"❌ [API Error] Failed to generate dashboard data: {e}", exc_info=True)
        return jsonify({"error": "Failed to load dashboard data."}), 500

@app.route('/api/open_signals')
def get_open_signals():
    if not check_db_connection(): return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor() as cur:
            query = sql.SQL("SELECT * FROM signals WHERE status IN ('open', 'updated') ORDER BY id DESC")
            cur.execute(query)
            signals = cur.fetchall()
        return jsonify({"signals": [dict(s) for s in signals]})
    except Exception as e:
        logger.error(f"Error fetching open signals: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance_stats')
def get_performance_stats():
    if not check_db_connection() or not conn:
        return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor() as cur:
            query = """
                SELECT
                    COUNT(*) AS total_trades,
                    COALESCE(SUM(CASE WHEN profit_percentage > 0 THEN 1 ELSE 0 END), 0) AS winning_trades,
                    COALESCE(AVG(profit_percentage), 0) AS avg_profit,
                    COALESCE(MAX(profit_percentage), 0) AS max_profit,
                    json_agg(
                        json_build_object('date', closed_at, 'profit', profit_percentage) ORDER BY closed_at
                    ) FILTER (WHERE profit_percentage IS NOT NULL) AS profit_history
                FROM signals
                WHERE status = 'closed' AND closed_at >= NOW() - INTERVAL '30 days';
            """
            cur.execute(query)
            stats = cur.fetchone()
            
            if stats['total_trades'] > 0:
                win_rate = (stats['winning_trades'] / stats['total_trades']) * 100 if stats['total_trades'] > 0 else 0
            else:
                win_rate = 0
            
            return jsonify({
                "total_trades": stats['total_trades'],
                "win_rate": win_rate,
                "avg_profit": stats['avg_profit'],
                "max_profit": stats['max_profit'],
                "profit_history": stats['profit_history'] or []
            })
    except Exception as e:
        logger.error(f"Error fetching performance stats: {e}")
        return jsonify({"error": str(e)}), 500

@sock.route('/ws')
def ws(ws_client):
    logger.info("WebSocket client connected.")
    with ws_clients_lock: ws_clients.append(ws_client)
    try:
        ws_client.send(json.dumps({"type": "connection_established"}, cls=NpEncoder))
        while True:
            message = ws_client.receive(timeout=30)
            if message is None: ws_client.send(json.dumps({"type": "ping"}, cls=NpEncoder))
    except Exception: logger.info("WebSocket client disconnected.")
    finally:
        with ws_clients_lock:
            if ws_client in ws_clients: ws_clients.remove(ws_client)

@app.route('/toggle_trading', methods=['POST'])
def toggle_trading():
    global is_trading_enabled
    with trading_status_lock: is_trading_enabled = not is_trading_enabled
    status_msg = "enabled" if is_trading_enabled else "disabled"
    log_and_notify("info", f"Trading has been {status_msg}.", "TRADING_STATUS")
    return jsonify({"status": "success", "trading_enabled": is_trading_enabled})

@app.route('/api/settings', methods=['POST'])
def update_settings():
    global FIXED_TRADE_AMOUNT_USDT, MAX_OPEN_TRADES, paper_trading_mode
    try:
        data = request.json
        if 'FIXED_TRADE_AMOUNT_USDT' in data:
            with fixed_trade_amount_lock:
                FIXED_TRADE_AMOUNT_USDT = float(data['FIXED_TRADE_AMOUNT_USDT'])
        if 'paper_trading_mode' in data:
            with trading_mode_lock:
                paper_trading_mode = bool(data['paper_trading_mode'])
        # In a real app, you would save these to Redis or a database
        return jsonify({"success": True, "message": "Settings updated"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/signal_quality', methods=['POST'])
def update_signal_quality():
    global MIN_SIGNAL_QUALITY
    try:
        data = request.json
        if 'min_quality' in data:
            with min_quality_lock:
                MIN_SIGNAL_QUALITY = int(data['min_quality'])
        return jsonify({"success": True, "message": "Signal quality updated"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# --- Main Loop & Threads ---
def get_mtf_trend(symbol: str) -> Dict[str, str]:
    trends = {}
    timeframes = {'15m': 10, '1h': 10, '4h': 10}
    for tf, days in timeframes.items():
        try:
            df = fetch_historical_data(symbol, tf, days)
            if df is None or len(df) < 50:
                trends[tf] = {'trend': 'unknown', 'adx': 0, 'rsi': 0}; continue
            
            df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

            # ADX
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
            atr = tr.ewm(span=14, adjust=False).mean()
            up_move = df['high'].diff()
            down_move = -df['low'].diff()
            plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
            minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
            plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr.replace(0, 1e-9)
            minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr.replace(0, 1e-9)
            dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
            adx = dx.ewm(span=14, adjust=False).mean()

            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-9)
            rsi = 100 - (100 / (1 + rs))

            last = df.iloc[-1]
            trend_val = 'sideways'
            if last['close'] > last['ema50'] and last['ema21'] > last['ema50']: trend_val = 'bullish'
            elif last['close'] < last['ema50'] and last['ema21'] < last['ema50']: trend_val = 'bearish'

            trends[tf] = {
                'trend': trend_val,
                'adx': adx.iloc[-1],
                'rsi': rsi.iloc[-1]
            }

        except Exception: trends[tf] = {'trend': 'unknown', 'adx': 0, 'rsi': 0}
    return trends
    
def main_bot_loop():
    logger.info("🚀 [Main Loop] Starting signal scanning loop...")
    while True:
        try:
            now = datetime.now(timezone.utc)
            seconds_until_next_candle = (15 - (now.minute % 15)) * 60 - now.second - 5 # Scan 5s before close
            if seconds_until_next_candle < 0:
                seconds_until_next_candle += 15 * 60
            
            logger.info(f"[Main Loop] Next scan in {seconds_until_next_candle:.0f} seconds...")
            time.sleep(seconds_until_next_candle)
            
            with trading_status_lock:
                if not is_trading_enabled: 
                    logger.info("[Main Loop] Trading is disabled, skipping scan cycle.")
                    continue
            
            logger.info("="*20 + " Starting New Scan Cycle " + "="*20)
            with market_state_lock:
                current_market_state["trend_details_by_tf"] = get_mtf_trend(BTC_SYMBOL)
                broadcast({"type": "market_state_update", "payload": current_market_state})

            for symbol in validated_symbols_to_scan:
                with signal_cache_lock:
                    open_signal_symbols = [s['symbol'] for s in open_signals_cache.values()]
                    if len(open_signals_cache) >= MAX_OPEN_TRADES: 
                        logger.warning(f"[Scan] Max open trades ({MAX_OPEN_TRADES}) reached. Stopping scan.")
                        break
                    if symbol in open_signal_symbols:
                        continue
                
                mtf_trend_symbol = get_mtf_trend(symbol)
                df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                if df is None or len(df) < 200:
                    if df is not None: log_rejection(symbol, "Insufficient Historical Data")
                    continue
                df_featured = calculate_all_features(df)
                df_featured.name = symbol
                
                strategy_found = None
                if USE_BB_STOCH_STRATEGY and check_bb_stoch_strategy_enhanced(df_featured, mtf_trend_symbol): strategy_found = "BB_Stoch_Strategy"
                elif USE_MACD_EMA_STRATEGY and check_macd_ema_strategy_enhanced(df_featured, mtf_trend_symbol): strategy_found = "MACD_EMA_Strategy"
                elif USE_EMA_RSI_STRATEGY and check_ema_rsi_strategy_enhanced(df_featured, mtf_trend_symbol): strategy_found = "EMA_RSI_Strategy"
                elif USE_PULLBACK_STRATEGY and check_pullback_strategy_enhanced(df_featured, mtf_trend_symbol): strategy_found = "Pullback_Strategy"
                elif USE_MOMENTUM_VOLATILITY_STRATEGY and check_momentum_volatility_strategy_enhanced(df_featured, mtf_trend_symbol): strategy_found = "Momentum_Volatility_Strategy"
                elif USE_ELLIOTT_WAVE_STRATEGY and check_elliott_wave_strategy_enhanced(df_featured, mtf_trend_symbol): strategy_found = "Elliott_Wave_Strategy"
                elif USE_RANGE_REVERSAL_STRATEGY and check_range_reversal_strategy(df_featured, mtf_trend_symbol): strategy_found = "Range_Reversal_Strategy"

                if strategy_found:
                    create_trade_signal(symbol, df_featured, strategy_found, mtf_trend_symbol)

        except Exception as e:
            logger.error(f"❌ [Main Loop] A critical error occurred: {e}", exc_info=True)
            time.sleep(60)

def trade_management_loop():
    logger.info("🚀 [Trade Manager] Starting advanced trade management loop...")
    while True:
        try:
            with signal_cache_lock:
                if not open_signals_cache:
                    time.sleep(2)
                    continue
                signals_to_monitor = list(open_signals_cache.values())

            for signal in signals_to_monitor:
                symbol = signal['symbol']
                current_price = live_prices.get(symbol)
                if current_price is None:
                    continue

                # Stop-Loss Check
                if current_price <= signal['stop_loss']:
                    logger.info(f"🚨 [SL] Stop-loss triggered for {symbol} at {current_price}")
                    # In a real scenario, you would place a market sell order here
                    # close_trade_in_db(signal['id'], current_price, 'stop_loss')
                    with signal_cache_lock:
                        if signal['id'] in open_signals_cache:
                           del open_signals_cache[signal['id']]
                    broadcast({"type": "trade_closed", "payload": {"signal_id": signal['id']}})
                    continue
                
                # Take-Profit Check (simplified)
                if current_price >= signal['target_price_1']:
                    logger.info(f"✅ [TP1] Take-profit 1 triggered for {symbol} at {current_price}")
                    # Logic for partial close and trailing stop would go here
                    with signal_cache_lock:
                         if signal['id'] in open_signals_cache:
                           del open_signals_cache[signal['id']]
                    broadcast({"type": "trade_closed", "payload": {"signal_id": signal['id']}})
            
            time.sleep(1)
        except Exception as e:
            logger.error(f"❌ [Trade Manager] Loop error: {e}", exc_info=True)
            time.sleep(2)


def update_balance_loop():
    logger.info("🚀 [Balance Updater] Starting balance update loop...")
    while True:
        try:
            with trading_mode_lock:
                is_paper = paper_trading_mode
            
            current_balance = 0.0
            if is_paper:
                # In paper mode, we might simulate balance changes
                current_balance = PAPER_TRADE_INITIAL_BALANCE
            else:
                try:
                    balance_info = client.get_asset_balance(asset='USDT')
                    current_balance = float(balance_info['free'])
                except Exception as e:
                    logger.error(f"❌ [Balance Loop] Failed to fetch real balance: {e}")

            with balance_lock:
                global usdt_balance
                usdt_balance = current_balance
            
            broadcast({"type": "full_dashboard_update", "payload": get_dashboard_payload()})

        except Exception as e: 
            logger.error(f"❌ [Balance Loop] Error: {e}", exc_info=True)
        
        time.sleep(60 * 5) # Update every 5 minutes

# --- نقطة بداية البرنامج ---
if __name__ == '__main__':
    logger.info("="*50 + "\n====== Starting Crypto Trading Bot V34.2.0 (Balance & Sizing Refactor) ======\n" + "="*50)
    init_db()
    init_redis()
    try:
        client = Client(API_KEY, API_SECRET); client.ping()
        logger.info("✅ [Binance] API connection successful.")
    except Exception as e:
        logger.critical(f"❌ [Binance] API connection failed: {e}"); exit(1)
    get_exchange_info_map()
    validated_symbols_to_scan = get_validated_symbols()
    if not validated_symbols_to_scan:
        logger.critical("❌ No valid symbols to scan. Exiting."); exit(1)
    
    load_open_signals_to_cache()
    load_settings_from_redis()
    logger.info("Initial data fetch complete.")
    
    start_websocket()
    Thread(target=main_bot_loop, daemon=True).start()
    Thread(target=trade_management_loop, daemon=True).start()
    Thread(target=update_balance_loop, daemon=True).start()
    
    logger.info("🌐 [Flask] Starting UI on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
