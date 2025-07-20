# ملف c4.py - نسخة كاملة مع لوحة تحكم تفاعلية
# تم التحديث والإصلاح بواسطة Gemini
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
from binance.exceptions import BinanceAPIException
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timezone, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Set, Tuple
from sklearn.preprocessing import StandardScaler
from collections import deque
import warnings

# --- إعدادات التجاهل واللوجر ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v9_dynamic_tp_sl.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV9_DynamicTPSL')

# --- تحميل متغيرات البيئة ---
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL', default='') 
    REDIS_URL: str = config('REDIS_URL', default='redis://localhost:6379/0')
    TELEGRAM_BOT_TOKEN: str = config('TELEGRAM_BOT_TOKEN', default='')
    TELEGRAM_CHAT_ID: str = config('TELEGRAM_CHAT_ID', default='')
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
    API_KEY, API_SECRET = None, None

# --- متغيرات عامة وإعدادات البوت ---
is_trading_enabled: bool = False
trading_status_lock = Lock()
RISK_PER_TRADE_PERCENT: float = 1.0
BTC_SYMBOL: str = 'BTCUSDT'
MAX_OPEN_TRADES: int = 4
SYMBOL_PROCESSING_BATCH_SIZE: int = 20
BATCH_PROCESSING_SLEEP_SECONDS: int = 10
SIGNAL_GENERATION_TIMEFRAME: str = '15m'

# --- متغيرات الحالة والكاش ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
exchange_info_map: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
notifications_cache = deque(maxlen=50)
notifications_lock = Lock()
rejection_logs_cache = deque(maxlen=100)
rejection_logs_lock = Lock()
current_market_state: Dict[str, Any] = {"overall_regime": {"key": "INITIALIZING", "ar": "تهيئة..."}, "trend_details_by_tf": {}, "last_updated": None}
market_state_lock = Lock()
dynamic_filter_profile_cache: Dict[str, Any] = {}
dynamic_filter_lock = Lock()

# --- قواميس الترجمة ---
REJECTION_REASONS_AR = {
    "Dynamic TP/SL Calculation Failed": "فشل حساب الأهداف الديناميكية"
}
TREND_TRANSLATIONS = {
    "STRONG_UPTREND": "اتجاه صاعد قوي", "UPTREND": "اتجاه صاعد",
    "STRONG_DOWNTREND": "اتجاه هابط قوي", "DOWNTREND": "اتجاه هابط",
    "RANGING": "متذبذب (تجميع)", "UNCERTAIN": "غير واضح", "INITIALIZING": "تهيئة..."
}

# --- دوال الخدمات والتهيئة ---
def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload, timeout=10).raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

def log_and_notify(level: str, message: str, component: str = "SYSTEM"):
    log_message = f"[{component}] {message}"
    timestamp = datetime.now(timezone.utc).isoformat()
    
    with notifications_lock:
        notifications_cache.appendleft({"timestamp": timestamp, "level": level.upper(), "message": message, "component": component})

    if level.lower() == "info": logger.info(log_message)
    elif level.lower() == "warning":
        logger.warning(log_message)
        send_telegram_message(f"⚠️ تحذير: {message}")
    elif level.lower() in ["error", "critical"]:
        logger.error(log_message)
        send_telegram_message(f"🛑 خطأ: {message}")

def log_rejection(symbol: str, reason_key: str, details: Dict = None):
    reason_ar = REJECTION_REASONS_AR.get(reason_key, reason_key)
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "reason_ar": reason_ar,
        "details": details or {}
    }
    with rejection_logs_lock:
        rejection_logs_cache.appendleft(log_entry)
    logger.debug(f"🚫 [REJECTED] {symbol}: {reason_ar} | Details: {details}")

def init_binance_client() -> Optional[Client]:
    if not API_KEY or not API_SECRET:
        logger.critical("API_KEY أو API_SECRET غير موجود.")
        return None
    try:
        c = Client(API_KEY, API_SECRET)
        c.ping()
        logger.info("✅ [Binance] تم الاتصال بنجاح.")
        return c
    except Exception as e:
        log_and_notify("error", f"فشل الاتصال بـ Binance: {e}", "BINANCE")
        return None

def get_validated_symbols() -> List[str]:
    if not client: return []
    try:
        info = client.get_exchange_info()
        global exchange_info_map
        exchange_info_map = {item['symbol']: item for item in info['symbols']}
        symbols = [
            s for s, i in exchange_info_map.items()
            if i['status'] == 'TRADING' and i['isSpotTradingAllowed'] and s.endswith('USDT')
            and not any(x in s for x in ['UP', 'DOWN', 'BULL', 'BEAR'])
        ]
        logger.info(f"تم العثور على {len(symbols)} عملة USDT صالحة للتداول.")
        return symbols
    except Exception as e:
        log_and_notify("error", f"فشل جلب معلومات الصرف: {e}", "BINANCE")
        return []

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        start_str = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])[cols]
        for col in cols[1:]: df[col] = pd.to_numeric(df[col])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        return df.set_index('timestamp').dropna()
    except Exception as e:
        logger.error(f"❌ [Data] خطأ في جلب البيانات لـ {symbol}: {e}")
        return None

def calculate_dynamic_tp_sl(df: pd.DataFrame, entry_price: float) -> Tuple[float, float]:
    df_copy = df.copy()
    df_copy['tr'] = pd.DataFrame([df_copy['high'] - df_copy['low'], abs(df_copy['high'] - df_copy['close'].shift()), abs(df_copy['low'] - df_copy['close'].shift())]).max(axis=0)
    atr = df_copy['tr'].ewm(span=14, adjust=False).mean().iloc[-1]
    if atr == 0: raise ValueError("ATR is zero")

    pivot_low = df_copy['low'].iloc[-20:].min()
    sl_price = min(entry_price - (atr * 1.5), pivot_low - (atr * 0.2))
    risk = entry_price - sl_price
    if risk <= 0: raise ValueError("Risk is non-positive")
    
    tp_price = entry_price + (risk * 1.5) # RRR 1.5
    return round(tp_price, 4), round(sl_price, 4)

def determine_market_state_enhanced():
    global current_market_state
    # دالة وهمية لتحديد حالة السوق لأغراض العرض
    states = ["UPTREND", "RANGING", "DOWNTREND"]
    with market_state_lock:
        current_market_state = {
            "overall_regime": {"key": "RANGING", "ar": "متذبذب (تجميع)"},
            "trend_details_by_tf": {
                "15m": {"key": "UPTREND", "ar": TREND_TRANSLATIONS["UPTREND"]},
                "1h": {"key": "RANGING", "ar": TREND_TRANSLATIONS["RANGING"]},
                "4h": {"key": "DOWNTREND", "ar": TREND_TRANSLATIONS["DOWNTREND"]}
            },
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

# ---------------------- واجهة Flask وتوابعها ----------------------
app = Flask(__name__)
CORS(app)

@app.route('/')
def dashboard():
    return render_template_string("""
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم التداول V9.3</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
    <style>
        :root { --bg-main: #0D1117; --bg-card: #161B22; --border-color: #30363D; --text-primary: #E6EDF3; --text-secondary: #848D97; --accent-blue: #58A6FF; --accent-green: #3FB950; --accent-red: #F85149; --accent-yellow: #D29922; --accent-gray: #484F58;}
        body { font-family: 'Tajawal', sans-serif; background-color: var(--bg-main); color: var(--text-primary); scroll-behavior: smooth; }
        .card { background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 0.5rem; }
        .tab-btn { transition: all 0.2s ease-in-out; border-bottom: 2px solid transparent; }
        .tab-btn.active { color: var(--accent-blue); border-bottom-color: var(--accent-blue); }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .trend-light { width: 1rem; height: 1rem; border-radius: 9999px; border: 2px solid #30363D; transition: all 0.5s ease; }
        .light-green { background-color: var(--accent-green); box-shadow: 0 0 8px 1px var(--accent-green); }
        .light-red { background-color: var(--accent-red); box-shadow: 0 0 8px 1px var(--accent-red); }
        .light-yellow { background-color: var(--accent-yellow); box-shadow: 0 0 8px 1px var(--accent-yellow); }
        .light-gray { background-color: var(--accent-gray); }
        .status-badge { padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 600; font-size: 0.8rem; }
        .bg-green-badge { background-color: rgba(63, 185, 80, 0.2); color: var(--accent-green); }
        .bg-red-badge { background-color: rgba(248, 81, 73, 0.2); color: var(--accent-red); }
        .bg-yellow-badge { background-color: rgba(210, 153, 34, 0.2); color: var(--accent-yellow); }
        .toggle-bg:after { content: ''; @apply absolute top-0.5 left-0.5 bg-white border border-gray-300 rounded-full h-5 w-5 transition-all; }
        input:checked + .toggle-bg:after { @apply transform translate-x-full; }
        input:checked + .toggle-bg { @apply bg-green-500; }
    </style>
</head>
<body class="p-4 md:p-6">
    <div class="container mx-auto max-w-screen-xl">
        <!-- Header -->
        <header class="card p-4 mb-6">
            <div class="flex flex-wrap justify-between items-center gap-4">
                <h1 class="text-2xl md:text-3xl font-extrabold"><span class="text-accent-blue">لوحة تحكم</span><span class="text-text-secondary font-medium"> V9.3</span></h1>
                <div class="flex items-center gap-x-4">
                    <div class="flex items-center gap-2">
                        <span class="font-semibold text-text-secondary">التداول:</span>
                        <label for="trading-toggle" class="flex items-center cursor-pointer">
                            <div class="relative">
                                <input type="checkbox" id="trading-toggle" class="sr-only">
                                <div class="toggle-bg block bg-gray-600 w-11 h-6 rounded-full"></div>
                            </div>
                        </label>
                    </div>
                    <div id="bot-status-container" class="flex items-center gap-2"></div>
                </div>
            </div>
            <div class="mt-4 pt-4 border-t border-border-color flex flex-wrap items-center justify-between gap-4">
                <div id="market-state-container" class="flex items-center gap-2"></div>
                <div id="trend-lights-container" class="flex items-center gap-x-4"></div>
            </div>
        </header>

        <!-- Tabs -->
        <div class="card">
            <div class="border-b border-border-color px-4">
                <nav class="-mb-px flex gap-x-6" id="tabs">
                    <button data-tab="trades" class="tab-btn py-3 px-1 text-sm font-medium active">الصفقات المفتوحة (<span id="open-trades-count">0</span>)</button>
                    <button data-tab="rejections" class="tab-btn py-3 px-1 text-sm font-medium">سجل الرفض</button>
                    <button data-tab="notifications" class="tab-btn py-3 px-1 text-sm font-medium">الإشعارات</button>
                    <button data-tab="config" class="tab-btn py-3 px-1 text-sm font-medium">الإعدادات</button>
                </nav>
            </div>
            <div class="p-4">
                <div id="trades-content" class="tab-content active"></div>
                <div id="rejections-content" class="tab-content"></div>
                <div id="notifications-content" class="tab-content"></div>
                <div id="config-content" class="tab-content"></div>
            </div>
        </div>
    </div>

<script>
    const API_URL = '/api/status';
    const CONTROL_URL = '/api/control';

    // --- Tab Handling ---
    const tabs = document.getElementById('tabs');
    const tabContents = document.querySelectorAll('.tab-content');
    tabs.addEventListener('click', (e) => {
        if (e.target.tagName !== 'BUTTON') return;
        
        tabs.querySelector('.active').classList.remove('active');
        e.target.classList.add('active');

        tabContents.forEach(content => content.classList.remove('active'));
        document.getElementById(`${e.target.dataset.tab}-content`).classList.add('active');
    });

    // --- Trading Toggle ---
    const tradingToggle = document.getElementById('trading-toggle');
    tradingToggle.addEventListener('change', async () => {
        try {
            const response = await fetch(CONTROL_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'toggle_trading' })
            });
            if (!response.ok) throw new Error('Network response was not ok.');
            const data = await response.json();
            updateTradingToggle(data.trading_enabled);
        } catch (error) {
            console.error('Error toggling trading:', error);
            // Revert toggle on error
            tradingToggle.checked = !tradingToggle.checked;
        }
    });

    function updateTradingToggle(isEnabled) {
        tradingToggle.checked = isEnabled;
    }

    // --- Data Rendering ---
    function formatTime(isoString) {
        if (!isoString) return 'N/A';
        return new Date(isoString).toLocaleString('ar-EG', {
            hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
        });
    }

    function renderStatus(data) {
        const container = document.getElementById('bot-status-container');
        const statusClass = data.bot_status === 'RUNNING' ? 'bg-green-badge' : 'bg-red-badge';
        const statusText = data.bot_status === 'RUNNING' ? 'يعمل' : 'متوقف';
        container.innerHTML = `<span class="status-badge ${statusClass}">${statusText}</span>`;
        updateTradingToggle(data.trading_enabled);
    }

    function renderMarketState(state) {
        const container = document.getElementById('market-state-container');
        const stateClass = state.overall_regime.key.includes('UP') ? 'text-accent-green' : state.overall_regime.key.includes('DOWN') ? 'text-accent-red' : 'text-accent-yellow';
        container.innerHTML = `<span class="font-semibold text-text-secondary">حالة السوق:</span> <span class="font-bold ${stateClass}">${state.overall_regime.ar || 'N/A'}</span>`;
    }

    function renderTrendLights(state) {
        const container = document.getElementById('trend-lights-container');
        let html = '';
        if (state.trend_details_by_tf) {
            for (const [tf, trend] of Object.entries(state.trend_details_by_tf)) {
                let lightClass = 'light-gray';
                if (trend.key.includes('UP')) lightClass = 'light-green';
                else if (trend.key.includes('DOWN')) lightClass = 'light-red';
                else if (trend.key.includes('RANGING')) lightClass = 'light-yellow';
                html += `<div class="flex items-center gap-2"><div class="trend-light ${lightClass}"></div><span class="text-xs font-medium text-text-secondary">${tf}</span></div>`;
            }
        }
        container.innerHTML = html;
    }

    function renderOpenTrades(trades) {
        const container = document.getElementById('trades-content');
        document.getElementById('open-trades-count').textContent = trades.length;
        if (trades.length === 0) {
            container.innerHTML = '<p class="text-center text-text-secondary py-4">لا توجد صفقات مفتوحة حالياً.</p>';
            return;
        }
        let rows = trades.map(t => `
            <tr class="border-b border-border-color hover:bg-white/5">
                <td class="p-3 font-mono font-bold">${t.symbol}</td>
                <td class="p-3 font-mono text-accent-green">${t.entry_price}</td>
                <td class="p-3 font-mono text-accent-green">${t.target_price}</td>
                <td class="p-3 font-mono text-accent-red">${t.stop_loss}</td>
                <td class="p-3 text-text-secondary">${formatTime(t.timestamp)}</td>
            </tr>
        `).join('');
        container.innerHTML = `
            <div class="overflow-x-auto">
                <table class="w-full text-sm text-right">
                    <thead class="text-xs text-text-secondary uppercase"><tr>
                        <th class="p-3">العملة</th><th class="p-3">سعر الدخول</th><th class="p-3">الهدف</th><th class="p-3">وقف الخسارة</th><th class="p-3">الوقت</th>
                    </tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>`;
    }
    
    function renderRejectionLogs(logs) {
        const container = document.getElementById('rejections-content');
        if (logs.length === 0) {
            container.innerHTML = '<p class="text-center text-text-secondary py-4">لا توجد سجلات رفض.</p>';
            return;
        }
        let rows = logs.slice(0, 50).map(l => `
            <tr class="border-b border-border-color hover:bg-white/5">
                <td class="p-3 text-text-secondary">${formatTime(l.timestamp)}</td>
                <td class="p-3 font-mono font-bold">${l.symbol}</td>
                <td class="p-3 text-accent-yellow">${l.reason_ar}</td>
            </tr>
        `).join('');
        container.innerHTML = `
            <div class="overflow-x-auto max-h-96">
                <table class="w-full text-sm text-right">
                    <thead class="text-xs text-text-secondary uppercase sticky top-0 bg-bg-card"><tr>
                        <th class="p-3">الوقت</th><th class="p-3">العملة</th><th class="p-3">سبب الرفض</th>
                    </tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>`;
    }

    function renderNotifications(items) {
        const container = document.getElementById('notifications-content');
        if (items.length === 0) {
            container.innerHTML = '<p class="text-center text-text-secondary py-4">لا توجد إشعارات.</p>';
            return;
        }
        let levelColors = { 'INFO': 'text-accent-blue', 'WARNING': 'text-accent-yellow', 'ERROR': 'text-accent-red', 'CRITICAL': 'text-accent-red font-bold' };
        let rows = items.map(n => `
            <div class="flex items-start gap-3 p-3 border-b border-border-color">
                <div class="text-xs text-text-secondary whitespace-nowrap">${formatTime(n.timestamp)}</div>
                <div class="w-20 text-center text-xs font-semibold ${levelColors[n.level] || 'text-text-primary'}">[${n.component}]</div>
                <div class="text-sm">${n.message}</div>
            </div>
        `).join('');
        container.innerHTML = `<div class="max-h-96 overflow-y-auto">${rows}</div>`;
    }

    function renderConfig(config) {
        const container = document.getElementById('config-content');
        let items = Object.entries(config).map(([key, value]) => `
            <div class="flex justify-between p-3 border-b border-border-color">
                <span class="text-text-secondary">${key.replace(/_/g, ' ')}</span>
                <span class="font-mono font-bold">${value}</span>
            </div>
        `).join('');
        container.innerHTML = `<div>${items}</div>`;
    }

    // --- Main Fetch and Update Cycle ---
    async function updateDashboard() {
        try {
            const response = await fetch(API_URL);
            if (!response.ok) throw new Error('Network response was not ok.');
            const data = await response.json();

            renderStatus(data);
            renderMarketState(data.market_state);
            renderTrendLights(data.market_state);
            renderOpenTrades(data.open_trades);
            renderRejectionLogs(data.rejection_logs);
            renderNotifications(data.notifications);
            renderConfig(data.config);

        } catch (error) {
            console.error('Error fetching dashboard data:', error);
            document.getElementById('bot-status-container').innerHTML = '<span class="status-badge bg-red-badge">خطأ في الاتصال</span>';
        }
    }

    setInterval(updateDashboard, 3000); // Update every 3 seconds
    updateDashboard(); // Initial load
</script>
</body>
</html>
    """)

@app.route('/api/status')
def api_status():
    with trading_status_lock: is_enabled = is_trading_enabled
    with signal_cache_lock: open_trades = list(open_signals_cache.values())
    with rejection_logs_lock: rejections = list(rejection_logs_cache)
    with market_state_lock: market_state = current_market_state
    with notifications_lock: notifications = list(notifications_cache)

    return jsonify({
        "bot_status": "RUNNING" if is_enabled else "PAUSED",
        "trading_enabled": is_enabled,
        "open_trades_count": len(open_trades),
        "market_state": market_state,
        "open_trades": open_trades,
        "rejection_logs": rejections,
        "notifications": notifications,
        "config": {
            "Risk Per Trade": f"{RISK_PER_TRADE_PERCENT}%",
            "Max Open Trades": MAX_OPEN_TRADES,
            "BTC Symbol": BTC_SYMBOL,
            "Signal Timeframe": SIGNAL_GENERATION_TIMEFRAME,
        }
    })

@app.route('/api/control', methods=['POST'])
def api_control():
    global is_trading_enabled
    data = request.json
    if data and data.get('action') == 'toggle_trading':
        with trading_status_lock:
            is_trading_enabled = not is_trading_enabled
            new_status = is_trading_enabled
        log_and_notify('info', f"Trading has been {'enabled' if new_status else 'disabled'} via dashboard.", "CONTROL")
        return jsonify({"success": True, "trading_enabled": new_status})
    return jsonify({"success": False, "message": "Invalid action"}), 400

# ---------------------- حلقة التداول الرئيسية ----------------------
def main_loop_enhanced():
    log_and_notify("info", "انتظار اكتمال التهيئة...")
    time.sleep(5)
    
    if not validated_symbols_to_scan: 
        log_and_notify("critical", "لا توجد عملات صالحة للمسح. الحلقة الرئيسية لن تبدأ.")
        return
    
    log_and_notify("info", f"بدء حلقة المسح لـ {len(validated_symbols_to_scan)} عملة.")

    while True:
        try:
            with trading_status_lock:
                if not is_trading_enabled:
                    time.sleep(5)
                    continue
            
            determine_market_state_enhanced()
            shuffled_symbols = random.sample(validated_symbols_to_scan, len(validated_symbols_to_scan))
            
            for symbol in shuffled_symbols:
                with trading_status_lock:
                    if not is_trading_enabled: break
                
                with signal_cache_lock:
                    if symbol in open_signals_cache or len(open_signals_cache) >= MAX_OPEN_TRADES:
                        continue
                
                df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, 5)
                if df_15m is None or len(df_15m) < 50: continue
                
                try:
                    entry_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                    tp_price, sl_price = calculate_dynamic_tp_sl(df_15m, entry_price)
                    
                    # Placeholder: Add to open trades for demo
                    with signal_cache_lock:
                        if len(open_signals_cache) < MAX_OPEN_TRADES:
                             open_signals_cache[symbol] = {
                                "symbol": symbol, "entry_price": entry_price, "target_price": tp_price,
                                "stop_loss": sl_price, "timestamp": datetime.now(timezone.utc).isoformat()
                            }
                             log_and_notify('info', f"Signal created for {symbol} at {entry_price}", "SIGNAL")

                except Exception as e:
                    log_rejection(symbol, "Dynamic TP/SL Calculation Failed", {"error": str(e)})
                
                finally:
                    if 'df_15m' in locals(): del df_15m
                    gc.collect()
                    time.sleep(1) # Small delay between symbols
            
            log_and_notify("info", "انتهت دورة المسح. الانتظار 60 ثانية...", "CYCLE")
            time.sleep(60)
            
        except (KeyboardInterrupt, SystemExit):
            log_and_notify("info", "إيقاف البوت.", "SYSTEM"); break
        except Exception as main_err:
            log_and_notify("error", f"خطأ حرج في الحلقة الرئيسية: {main_err}", "SYSTEM"); time.sleep(120)

# --- نقطة الانطلاق ---
if __name__ == "__main__":
    logger.info("🚀 إطلاق بوت التداول ولوحة التحكم (V9.3 - لوحة تحكم تفاعلية) 🚀")
    
    client = init_binance_client()
    if client:
        validated_symbols_to_scan = get_validated_symbols()
    else:
        log_and_notify("critical", "لا يمكن بدء البوت بدون الاتصال بـ Binance.")
        exit(1)

    trading_thread = Thread(target=main_loop_enhanced, daemon=True)
    trading_thread.start()
    logger.info("✅ تم تشغيل حلقة التداول الرئيسية في الخلفية.")

    try:
        port = int(os.environ.get('PORT', 8080))
        logger.info(f"🌍 بدء تشغيل خادم الويب على 0.0.0.0:{port}...")
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        log_and_notify("critical", f"❌ فشل حاسم في تشغيل خادم الويب: {e}", "SYSTEM")
        exit(1)
