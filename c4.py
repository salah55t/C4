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
        logging.FileHandler('crypto_bot_v9_improved.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV9_Improved')

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

# --- إعدادات الهدف ووقف الخسارة ---
USE_DATABASE_SL_TP: bool = True
ATR_FALLBACK_SL_MULTIPLIER: float = 1.5
ATR_FALLBACK_TP_MULTIPLIER: float = 2.0
SL_BUFFER_ATR_PERCENT: float = 0.25 # نسبة من ATR لإضافتها كهامش أمان لوقف الخسارة

# --- إعدادات وقف الخسارة المتحرك (Trailing Stop-Loss) ---
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_ACTIVATION_PROFIT_PERCENT: float = 1.0
TRAILING_DISTANCE_PERCENT: float = 0.8

# --- إعدادات الفلاتر المحسّنة ---
USE_BTC_TREND_FILTER: bool = True
BTC_SYMBOL: str = 'BTCUSDT'
BTC_TREND_TIMEFRAME: str = '4h' # This is now a base, but the new logic uses multiple TFs
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

# --- ✨ IMPROVED: متغيرات حالة السوق المحسّنة ---
last_market_state_check = 0
current_market_state: Dict[str, Any] = {
    "overall_regime": "INITIALIZING",
    "details": {},
    "last_updated": None
}
market_state_lock = Lock()


# ---------------------- دوال HTML للوحة التحكم ----------------------
def get_dashboard_html():
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم بوت التداول</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Cairo', sans-serif; }
        .dark {
            --bg-color: #111827;
            --card-color: #1f2937;
            --text-color: #d1d5db;
            --header-color: #9ca3af;
            --border-color: #374151;
            --green-bg: #052e16;
            --red-bg: #450a0a;
            --yellow-bg: #422006;
        }
        body {
            background-color: var(--bg-color);
            color: var(--text-color);
        }
        .card {
            background-color: var(--card-color);
            border: 1px solid var(--border-color);
        }
        .header { color: var(--header-color); }
        .bg-custom-green { background-color: var(--green-bg); }
        .bg-custom-red { background-color: var(--red-bg); }
        .bg-custom-yellow { background-color: var(--yellow-bg); }
        .blinking { animation: blinker 1.5s linear infinite; }
        @keyframes blinker { 50% { opacity: 0.3; } }
    </style>
</head>
<body class="dark p-4 md:p-6">
    <div class="container mx-auto">
        <header class="mb-6 flex justify-between items-center">
            <h1 class="text-2xl md:text-3xl font-bold text-white">لوحة تحكم بوت التداول</h1>
            <button id="testTelegram" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition-colors">
                اختبار تليجرام
            </button>
        </header>

        <!-- Market Status Section -->
        <section id="market-status" class="mb-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <!-- Overall Regime -->
            <div id="overall-regime-card" class="card rounded-xl p-4 flex flex-col justify-center items-center">
                <h3 class="header text-lg font-semibold mb-2">حالة السوق العامة</h3>
                <div id="overall-regime" class="text-2xl font-bold">جاري التحميل...</div>
            </div>
            <!-- Fear & Greed -->
            <div class="card rounded-xl p-4 flex flex-col justify-center items-center">
                <h3 class="header text-lg font-semibold mb-2">مؤشر الخوف والطمع</h3>
                <div id="fear-greed" class="text-2xl font-bold">...</div>
            </div>
            <!-- Timeframe 1H -->
            <div id="tf-1h-card" class="card rounded-xl p-4">
                <h3 class="header text-center font-semibold mb-2">اتجاه (1 ساعة)</h3>
                <div id="tf-1h-status" class="text-xl font-bold text-center">...</div>
                <div id="tf-1h-details" class="text-sm text-center mt-1">...</div>
            </div>
            <!-- Timeframe 4H -->
            <div id="tf-4h-card" class="card rounded-xl p-4">
                <h3 class="header text-center font-semibold mb-2">اتجاه (4 ساعات)</h3>
                <div id="tf-4h-status" class="text-xl font-bold text-center">...</div>
                <div id="tf-4h-details" class="text-sm text-center mt-1">...</div>
            </div>
        </section>

        <!-- Stats Section -->
        <section id="stats" class="mb-6 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-4 gap-4">
            <!-- Cards will be injected here -->
        </section>

        <!-- Tabs -->
        <div class="mb-4 border-b border-gray-700">
            <nav class="flex space-x-4" aria-label="Tabs">
                <button onclick="showTab('signals')" class="tab-btn active text-white py-2 px-4">التوصيات</button>
                <button onclick="showTab('notifications')" class="tab-btn text-gray-400 py-2 px-4">الإشعارات</button>
                <button onclick="showTab('rejections')" class="tab-btn text-gray-400 py-2 px-4">التوصيات المرفوضة</button>
            </nav>
        </div>

        <!-- Signals Table -->
        <div id="signals-tab" class="tab-content">
            <div class="overflow-x-auto card rounded-lg">
                <table class="min-w-full text-sm text-right">
                    <thead class="bg-gray-700">
                        <tr>
                            <th class="p-3">العملة</th>
                            <th class="p-3">الحالة</th>
                            <th class="p-3">الربح/الخسارة (%)</th>
                            <th class="p-3">سعر الدخول</th>
                            <th class="p-3">السعر الحالي</th>
                            <th class="p-3">الهدف</th>
                            <th class="p-3">وقف الخسارة</th>
                            <th class="p-3">إجراء</th>
                        </tr>
                    </thead>
                    <tbody id="signals-table">
                        <!-- Rows will be injected here -->
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Notifications -->
        <div id="notifications-tab" class="tab-content hidden">
            <div class="card rounded-lg p-4 max-h-96 overflow-y-auto">
                <ul id="notifications-list" class="space-y-2"></ul>
            </div>
        </div>

        <!-- Rejection Logs -->
        <div id="rejections-tab" class="tab-content hidden">
            <div class="card rounded-lg p-4 max-h-96 overflow-y-auto">
                <ul id="rejections-list" class="space-y-2"></ul>
            </div>
        </div>
    </div>

<script>
const REGIME_STYLES = {
    "STRONG UPTREND": { text: "اتجاه صاعد قوي", color: "text-green-400", bg: "bg-custom-green" },
    "UPTREND": { text: "اتجاه صاعد", color: "text-green-500", bg: "bg-custom-green" },
    "RANGING": { text: "عرضي", color: "text-yellow-400", bg: "bg-custom-yellow" },
    "DOWNTREND": { text: "اتجاه هابط", color: "text-red-500", bg: "bg-custom-red" },
    "STRONG DOWNTREND": { text: "اتجاه هابط قوي", color: "text-red-400", bg: "bg-custom-red" },
    "UNCERTAIN": { text: "غير واضح", color: "text-gray-400", bg: "bg-gray-700" },
    "INITIALIZING": { text: "جاري التهيئة...", color: "text-blue-400", bg: "bg-blue-900" }
};

const TF_STATUS_STYLES = {
    "Uptrend": { text: "صاعد", icon: "▲", color: "text-green-400" },
    "Downtrend": { text: "هابط", icon: "▼", color: "text-red-400" },
    "Ranging": { text: "عرضي", icon: " sideways", color: "text-yellow-400" },
};

function updateMarketStatus() {
    fetch('/api/market_status')
        .then(response => response.json())
        .then(data => {
            // Fear & Greed
            const fg = data.fear_and_greed;
            document.getElementById('fear-greed').textContent = `${fg.value} (${fg.classification})`;

            // Overall Market Regime
            const state = data.market_state;
            const overallRegime = state.overall_regime || "UNCERTAIN";
            const regimeStyle = REGIME_STYLES[overallRegime.toUpperCase()] || REGIME_STYLES["UNCERTAIN"];
            
            const overallDiv = document.getElementById('overall-regime');
            overallDiv.textContent = regimeStyle.text;
            overallDiv.className = `text-2xl font-bold ${regimeStyle.color}`;

            const overallCard = document.getElementById('overall-regime-card');
            overallCard.className = `card rounded-xl p-4 flex flex-col justify-center items-center ${regimeStyle.bg}`;

            // Timeframe Details
            updateTimeframeCard('1h', state.details['1h']);
            updateTimeframeCard('4h', state.details['4h']);
        });
}

function updateTimeframeCard(tf, data) {
    const card = document.getElementById(`tf-${tf}-card`);
    const statusDiv = document.getElementById(`tf-${tf}-status`);
    const detailsDiv = document.getElementById(`tf-${tf}-details`);

    if (!data) {
        statusDiv.textContent = 'N/A';
        detailsDiv.textContent = '';
        return;
    }

    const style = TF_STATUS_STYLES[data.trend] || { text: 'N/A', icon: '', color: 'text-gray-400' };
    statusDiv.innerHTML = `<span class="${style.color}">${style.icon} ${style.text}</span>`;
    detailsDiv.textContent = `RSI: ${data.rsi.toFixed(1)} | ADX: ${data.adx.toFixed(1)}`;
    
    let bgColor = 'bg-gray-800';
    if (data.trend === 'Uptrend') bgColor = 'bg-custom-green';
    else if (data.trend === 'Downtrend') bgColor = 'bg-custom-red';
    else if (data.trend === 'Ranging') bgColor = 'bg-custom-yellow';
    card.className = `card rounded-xl p-4 ${bgColor}`;
}

function updateStats() {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            const statsContainer = document.getElementById('stats');
            statsContainer.innerHTML = `
                <div class="card p-4 rounded-lg text-center"><div class="header">صفقات مفتوحة</div><div class="text-2xl font-bold">${data.open_trades_count}</div></div>
                <div class="card p-4 rounded-lg text-center"><div class="header">إجمالي الربح</div><div class="text-2xl font-bold ${data.total_profit_pct >= 0 ? 'text-green-400' : 'text-red-400'}">${data.total_profit_pct.toFixed(2)}%</div></div>
                <div class="card p-4 rounded-lg text-center"><div class="header">نسبة النجاح</div><div class="text-2xl font-bold">${data.win_rate.toFixed(2)}%</div></div>
                <div class="card p-4 rounded-lg text-center"><div class="header">عامل الربح</div><div class="text-2xl font-bold">${data.profit_factor.toFixed(2)}</div></div>
            `;
        });
}

function updateSignals() {
    fetch('/api/signals')
        .then(response => response.json())
        .then(data => {
            const tableBody = document.getElementById('signals-table');
            tableBody.innerHTML = '';
            if (!data || data.error) {
                tableBody.innerHTML = '<tr><td colspan="8" class="text-center p-4">فشل في تحميل البيانات.</td></tr>';
                return;
            }
            data.forEach(signal => {
                const pnlPct = signal.status === 'open' ? (signal.pnl_pct || 0) : (signal.profit_percentage || 0);
                const pnlClass = pnlPct >= 0 ? 'text-green-400' : 'text-red-400';
                const statusClass = signal.status === 'open' ? 'text-yellow-400 blinking' : 'text-gray-400';
                
                const row = `
                    <tr class="border-b border-gray-700 hover:bg-gray-800">
                        <td class="p-3 font-mono">${signal.symbol}</td>
                        <td class="p-3 ${statusClass}">${signal.status}</td>
                        <td class="p-3 font-mono ${pnlClass}">${pnlPct.toFixed(2)}%</td>
                        <td class="p-3 font-mono">${signal.entry_price.toFixed(4)}</td>
                        <td class="p-3 font-mono">${signal.current_price ? signal.current_price.toFixed(4) : 'N/A'}</td>
                        <td class="p-3 font-mono">${signal.target_price.toFixed(4)}</td>
                        <td class="p-3 font-mono">${signal.stop_loss.toFixed(4)}</td>
                        <td class="p-3">
                            ${signal.status === 'open' ? `<button onclick="closeSignal(${signal.id})" class="bg-red-600 hover:bg-red-700 text-white text-xs py-1 px-2 rounded">إغلاق</button>` : ''}
                        </td>
                    </tr>
                `;
                tableBody.innerHTML += row;
            });
        });
}

function updateNotifications() {
    fetch('/api/notifications')
        .then(response => response.json())
        .then(data => {
            const list = document.getElementById('notifications-list');
            list.innerHTML = '';
            data.forEach(n => {
                const item = `<li class="p-2 rounded-md bg-gray-800 text-sm">[${new Date(n.timestamp).toLocaleString('ar-EG')}] ${n.message}</li>`;
                list.innerHTML += item;
            });
        });
}

function updateRejectionLogs() {
    fetch('/api/rejection_logs')
        .then(response => response.json())
        .then(data => {
            const list = document.getElementById('rejections-list');
            list.innerHTML = '';
            data.forEach(log => {
                const details = JSON.stringify(log.details);
                const item = `<li class="p-2 rounded-md bg-gray-800 text-sm">[${new Date(log.timestamp).toLocaleString('ar-EG')}] <strong>${log.symbol}</strong>: ${log.reason} - <span class="font-mono text-xs">${details}</span></li>`;
                list.innerHTML += item;
            });
        });
}

function closeSignal(id) {
    if (!confirm('هل أنت متأكد من رغبتك في إغلاق هذه الصفقة يدوياً؟')) return;
    fetch(`/api/close/${id}`, { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            alert(data.message || data.error);
            updateSignals();
        });
}

function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.add('hidden'));
    document.getElementById(`${tabName}-tab`).classList.remove('hidden');
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('text-white');
        btn.classList.add('text-gray-400');
    });
    event.target.classList.add('text-white');
    event.target.classList.remove('text-gray-400');
}

document.getElementById('testTelegram').addEventListener('click', () => {
    fetch('/api/test_telegram').then(res => res.text()).then(alert);
});

function refreshData() {
    updateMarketStatus();
    updateStats();
    updateSignals();
    updateNotifications();
    updateRejectionLogs();
}

setInterval(refreshData, 5000); // Refresh every 5 seconds
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
        logger.info("[DB] 'sslmode=require' was automatically added to the database URL for cloud compatibility.")
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(db_url_to_use, connect_timeout=15, cursor_factory=RealDictCursor)
            conn.autocommit = False
            with conn.cursor() as cur:
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
                        current_peak_price DOUBLE PRECISION
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
                logger.critical("❌ [DB] Failed to connect to the database after multiple retries.")


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

# ---------------------- دوال Binance والبيانات ----------------------
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
        limit = min(limit, 1000) # Binance limit is 1000 klines per request
        
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

# ---------------------- دوال جلب الميزات من قاعدة البيانات ----------------------
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
        if not df_ichimoku.index.tz:
             df_ichimoku.index = df_ichimoku.index.tz_localize('UTC')
        for col in ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']:
            if col in df_ichimoku.columns:
                df_ichimoku[col] = pd.to_numeric(df_ichimoku[col], errors='coerce')
        return df_ichimoku.dropna()
    except Exception as e:
        logger.error(f"❌ [Ichimoku Fetch Bot] Could not fetch Ichimoku features for {symbol}: {e}")
        if conn: conn.rollback()
        return pd.DataFrame()

# ---------------------- دوال حساب الميزات ----------------------
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

# ---------------------- ✨ IMPROVED: دوال تحديد اتجاه السوق المحسّنة ----------------------

def get_trend_for_timeframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates trend indicators for a single timeframe."""
    if df is None or len(df) < 26:
        return {"trend": "Uncertain", "rsi": -1, "adx": -1}

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=13, adjust=False).mean()
    rsi = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))

    # ADX
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

    # Trend Determination
    ema_fast = df['close'].ewm(span=12, adjust=False).mean().iloc[-1]
    ema_slow = df['close'].ewm(span=26, adjust=False).mean().iloc[-1]
    
    trend = "Uncertain"
    if adx.iloc[-1] > 20:
        if ema_fast > ema_slow and rsi.iloc[-1] > 50:
            trend = "Uptrend"
        elif ema_fast < ema_slow and rsi.iloc[-1] < 50:
            trend = "Downtrend"
        else:
            trend = "Ranging"
    else:
        trend = "Ranging"

    return {
        "trend": trend,
        "rsi": rsi.iloc[-1],
        "adx": adx.iloc[-1]
    }

def determine_market_state():
    """
    Determines the overall market state by analyzing multiple timeframes of BTC.
    This is a more robust approach than the single timeframe analysis.
    """
    global current_market_state, last_market_state_check
    
    # --- Cache check to avoid excessive API calls ---
    with market_state_lock:
        if time.time() - last_market_state_check < 300: # 5 minutes
            return current_market_state

    logger.info("🧠 [Market State] Updating market state using Multi-Timeframe Analysis (MTA)...")
    
    try:
        # --- Fetch data for all required timeframes ---
        df_1h = fetch_historical_data(BTC_SYMBOL, '1h', 5)
        df_4h = fetch_historical_data(BTC_SYMBOL, '4h', 15)
        
        if df_1h is None or df_4h is None:
            logger.warning("⚠️ [Market State] Could not fetch all required BTC data. Using previous state.")
            return current_market_state

        # --- Analyze each timeframe ---
        state_1h = get_trend_for_timeframe(df_1h)
        state_4h = get_trend_for_timeframe(df_4h)
        
        trends = [state_1h['trend'], state_4h['trend']]
        
        # --- Combine results to determine overall regime ---
        overall_regime = "UNCERTAIN"
        uptrends = trends.count("Uptrend")
        downtrends = trends.count("Downtrend")

        if uptrends == 2:
            overall_regime = "STRONG UPTREND"
        elif uptrends == 1 and downtrends == 0:
            overall_regime = "UPTREND"
        elif downtrends == 2:
            overall_regime = "STRONG DOWNTREND"
        elif downtrends == 1 and uptrends == 0:
            overall_regime = "DOWNTREND"
        elif "Ranging" in trends:
            overall_regime = "RANGING"

        # --- Update the global state variable ---
        with market_state_lock:
            current_market_state = {
                "overall_regime": overall_regime,
                "details": {
                    "1h": state_1h,
                    "4h": state_4h,
                },
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            last_market_state_check = time.time()
            
        logger.info(f"✅ [Market State] New state determined: {current_market_state['overall_regime']}")
        logger.info(f"   - 1H: {state_1h['trend']} (RSI: {state_1h['rsi']:.1f}, ADX: {state_1h['adx']:.1f})")
        logger.info(f"   - 4H: {state_4h['trend']} (RSI: {state_4h['rsi']:.1f}, ADX: {state_4h['adx']:.1f})")

        return current_market_state

    except Exception as e:
        logger.error(f"❌ [Market State] Failed to determine market state: {e}", exc_info=True)
        # Return the last known state in case of an error
        return current_market_state

# ---------------------- دوال الفلاتر وحساب الأهداف ----------------------

def passes_speed_filter(last_features: pd.Series) -> bool:
    symbol = last_features.name
    
    # Use the new detailed market state
    with market_state_lock:
        regime = current_market_state.get("overall_regime", "RANGING")

    if regime in ["DOWNTREND", "STRONG DOWNTREND"]:
        log_rejection(symbol, "Speed Filter", {"detail": f"Disabled due to market regime: {regime}"})
        return True # In a downtrend, this filter might not be relevant for buy signals
    
    # Adjust thresholds based on market regime
    if regime == "STRONG UPTREND":
        adx_threshold, rel_vol_threshold, rsi_min, rsi_max = (22.0, 0.5, 40.0, 75.0)
    elif regime == "UPTREND":
        adx_threshold, rel_vol_threshold, rsi_min, rsi_max = (18.0, 0.2, 35.0, 80.0)
    else: # RANGING or UNCERTAIN
        adx_threshold, rel_vol_threshold, rsi_min, rsi_max = (16.0, 0.1, 30.0, 80.0)

    adx, rel_vol, rsi = last_features.get('adx', 0), last_features.get('relative_volume', 0), last_features.get('rsi', 0)
    if (adx >= adx_threshold and rel_vol >= rel_vol_threshold and rsi_min <= rsi < rsi_max):
        return True
    
    log_rejection(symbol, "Speed Filter", {
        "Regime": regime,
        "ADX": f"{adx:.2f} (Req: >{adx_threshold})",
        "Volume": f"{rel_vol:.2f} (Req: >{rel_vol_threshold})",
        "RSI": f"{rsi:.2f} (Req: {rsi_min}-{rsi_max})"
    })
    return False

def calculate_db_driven_tp_sl(symbol: str, entry_price: float, last_atr: float) -> Optional[Dict[str, Any]]:
    logger.info(f"[{symbol}] 🧠 Calculating TP/SL from Database for entry price: {entry_price:.4f}")
    
    sr_levels_df = fetch_sr_levels_from_db(symbol)
    ichimoku_df = fetch_ichimoku_features_from_db(symbol, SIGNAL_GENERATION_TIMEFRAME)
    
    all_levels = []
    if not sr_levels_df.empty:
        all_levels.extend(sr_levels_df['level_price'].astype(float).tolist())
        logger.info(f"[{symbol}] Found {len(sr_levels_df)} S/R & Fibonacci levels in DB.")

    if not ichimoku_df.empty:
        last_ichi = ichimoku_df.iloc[-1]
        ichi_levels = [
            last_ichi.get('tenkan_sen'),
            last_ichi.get('kijun_sen'),
            last_ichi.get('senkou_span_a'),
            last_ichi.get('senkou_span_b')
        ]
        valid_ichi_levels = [float(lvl) for lvl in ichi_levels if pd.notna(lvl)]
        all_levels.extend(valid_ichi_levels)
        logger.info(f"[{symbol}] Found {len(valid_ichi_levels)} Ichimoku levels in DB.")

    if not all_levels:
        log_rejection(symbol, "No DB Levels", {"detail": "No S/R or Ichimoku levels found in the database."})
        return None

    unique_levels = sorted(list(set(all_levels)))
    resistances = [lvl for lvl in unique_levels if lvl > entry_price]
    supports = [lvl for lvl in unique_levels if lvl < entry_price]

    logger.info(f"[{symbol}] Potential Resistances (> {entry_price:.4f}): {resistances}")
    logger.info(f"[{symbol}] Potential Supports (< {entry_price:.4f}): {supports}")

    target_price = min(resistances) if resistances else None
    stop_loss_price = max(supports) if supports else None

    if target_price is None or stop_loss_price is None:
        logger.warning(f"[{symbol}] ⚠️ Could not determine a clear TP or SL from DB levels. Using ATR fallback.")
        log_rejection(symbol, "Insufficient DB Levels", {"detail": "Not enough support/resistance found around entry price."})
        
        fallback_tp = entry_price + (last_atr * ATR_FALLBACK_TP_MULTIPLIER)
        fallback_sl = entry_price - (last_atr * ATR_FALLBACK_SL_MULTIPLIER)
        
        logger.info(f"[{symbol}] ATR Fallback: TP={fallback_tp:.4f}, SL={fallback_sl:.4f}")
        return {'target_price': fallback_tp, 'stop_loss': fallback_sl, 'source': 'ATR_Fallback'}
    
    final_stop_loss = stop_loss_price - (last_atr * SL_BUFFER_ATR_PERCENT)
    
    logger.info(f"✅ [{symbol}] DB-driven levels determined:")
    logger.info(f"   - Target (Closest Resistance): {target_price:.4f}")
    logger.info(f"   - Stop Loss (Closest Support): {stop_loss_price:.4f}")
    logger.info(f"   - Final Stop Loss (with ATR buffer): {final_stop_loss:.4f}")

    return {
        'target_price': target_price,
        'stop_loss': final_stop_loss,
        'source': 'Database'
    }


# ---------------------- WebSocket و TradingStrategy ----------------------
def handle_price_update_message(msg: List[Dict[str, Any]]) -> None:
    if not isinstance(msg, list) or not redis_client: return
    try:
        price_updates = {item.get('s'): float(item.get('c', 0)) for item in msg if item.get('s') and item.get('c')}
        if price_updates:
            redis_client.hset(REDIS_PRICES_HASH_NAME, mapping=price_updates)
    except Exception as e:
        logger.error(f"❌ [WebSocket Price Updater] Error: {e}", exc_info=True)

def initiate_signal_closure(symbol: str, signal_to_close: Dict, status: str, closing_price: float):
    signal_id = signal_to_close.get('id')
    with closure_lock:
        if signal_id in signals_pending_closure: return
        signals_pending_closure.add(signal_id)
    with signal_cache_lock:
        signal_data_for_thread = open_signals_cache.pop(symbol, None)
    if signal_data_for_thread:
        Thread(target=close_signal, args=(signal_data_for_thread, status, closing_price, "auto_monitor")).start()
    else:
        with closure_lock:
            signals_pending_closure.discard(signal_id)

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
        if not all([self.ml_model, self.scaler, self.feature_names]) or df_features.empty:
            return None
        
        try:
            last_row_ordered_df = df_features.iloc[[-1]][self.feature_names]
            features_scaled_np = self.scaler.transform(last_row_ordered_df)
            features_scaled_df = pd.DataFrame(features_scaled_np, columns=self.feature_names)

            prediction = self.ml_model.predict(features_scaled_df)[0]
            prediction_proba = self.ml_model.predict_proba(features_scaled_df)
            confidence = float(np.max(prediction_proba[0]))

            logger.info(f"ℹ️ [Signal Generation] {self.symbol}: Model predicted '{'BUY' if prediction == 1 else 'SELL'}' with {confidence:.2%} confidence.")
            
            return {'prediction': int(prediction), 'confidence': confidence}

        except Exception as e:
            logger.warning(f"⚠️ [Signal Generation] {self.symbol}: Error: {e}")
            return None


# ---------------------- حلقة مراقبة الصفقات ----------------------
def trade_monitoring_loop():
    global last_api_check_time
    logger.info("✅ [Trade Monitor] Starting trade monitoring loop (with Trailing Stop support).")
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

# ---------------------- دوال التنبيهات والإدارة ----------------------
def send_telegram_message(target_chat_id: str, text: str, reply_markup: Optional[Dict] = None) -> bool:
    if not TELEGRAM_TOKEN or not target_chat_id:
        logger.error("❌ [Telegram] Token or Chat ID is missing.")
        return False
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(target_chat_id), 'text': text, 'parse_mode': 'Markdown'}
    if reply_markup: payload['reply_markup'] = json.dumps(reply_markup)
    
    logger.info(f"ℹ️ [Telegram] Attempting to send message to Chat ID: {target_chat_id}")
    try: 
        response = requests.post(url, json=payload, timeout=10)
        logger.info(f"ℹ️ [Telegram] Server Response: {response.status_code} | Text: {response.text}")
        response.raise_for_status()
        logger.info("✅ [Telegram] Message sent successfully.")
        return True
    except requests.exceptions.RequestException as e: 
        logger.error(f"❌ [Telegram] Failed to send message: {e}")
        return False

def send_new_signal_alert(signal_data: Dict[str, Any]):
    safe_symbol = signal_data['symbol'].replace('_', '\\_')
    entry = float(signal_data['entry_price'])
    target = float(signal_data['target_price'])
    sl = float(signal_data['stop_loss'])
    profit_pct = ((target / entry) - 1) * 100
    risk_pct = abs(((entry / sl) - 1) * 100) if sl > 0 else 0
    rrr = profit_pct / risk_pct if risk_pct > 0 else 0
    
    # Add market state to the alert
    with market_state_lock:
        market_regime = current_market_state.get('overall_regime', 'N/A')

    message = (
        f"💡 *توصية تداول جديدة* 💡\n\n"
        f" *العملة:* `{safe_symbol}`\n"
        f" *الاستراتيجية:* `{BASE_ML_MODEL_NAME}`\n"
        f" *حالة السوق:* `{market_regime}`\n\n"
        f" *الدخول:* `${entry:,.8g}`\n"
        f" *الهدف:* `${target:,.8g}`\n"
        f" *وقف الخسارة:* `${sl:,.8g}`\n\n"
        f" *الربح المتوقع:* `{profit_pct:+.2f}%`\n"
        f" *المخاطرة:* `{risk_pct:.2f}%`\n"
        f" *المخاطرة/العائد:* `1:{rrr:.2f}`\n\n"
        f" *ثقة النموذج:* {signal_data['signal_details']['ML_Confidence']}\n"
        f" *مصدر الهدف:* {signal_data['signal_details']['TP_SL_Source']}"
    )
    
    reply_markup = {"inline_keyboard": [[{"text": "📊 فتح لوحة التحكم", "url": WEBHOOK_URL or '#'}]]}
    if send_telegram_message(CHAT_ID, message, reply_markup):
        log_and_notify('info', f"New Signal: {signal_data['symbol']} in {market_regime} market", "NEW_SIGNAL")

def insert_signal_into_db(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not check_db_connection() or not conn: return None
    try:
        entry = float(signal['entry_price'])
        target = float(signal['target_price'])
        sl = float(signal['stop_loss'])
        
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, current_peak_price)
                   VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;""",
                (signal['symbol'], entry, target, sl, signal.get('strategy_name'), json.dumps(signal.get('signal_details', {})), entry)
            )
            signal['id'] = cur.fetchone()['id']
        conn.commit()
        logger.info(f"✅ [DB] Inserted signal {signal['id']} for {signal['symbol']}.")
        return signal
    except Exception as e:
        logger.error(f"❌ [Insert] Error inserting signal for {signal['symbol']}: {e}", exc_info=True)
        if conn: conn.rollback()
        return None

def update_signal_target_in_db(signal_id: int, new_target: float, new_stop_loss: float) -> bool:
    if not check_db_connection() or not conn: return False
    try:
        db_new_target = float(new_target)
        db_new_stop_loss = float(new_stop_loss)
        
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE signals SET target_price = %s, stop_loss = %s WHERE id = %s;",
                (db_new_target, db_new_stop_loss, signal_id)
            )
        conn.commit()
        logger.info(f"✅ [DB Update] Updated TP/SL for signal {signal_id}.")
        return True
    except Exception as e:
        logger.error(f"❌ [DB Update] Error updating signal {signal_id}: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

def close_signal(signal: Dict, status: str, closing_price: float, closed_by: str):
    signal_id = signal.get('id'); symbol = signal.get('symbol')
    logger.info(f"Initiating closure for signal {signal_id} ({symbol}) with status '{status}'")
    try:
        if not check_db_connection() or not conn: raise OperationalError("DB connection failed during signal closure.")
        
        db_closing_price = float(closing_price)
        entry_price = float(signal['entry_price'])
        profit_pct = ((db_closing_price / entry_price) - 1) * 100
        
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE signals SET status = %s, closing_price = %s, closed_at = NOW(), profit_percentage = %s WHERE id = %s AND status = 'open';",
                (status, db_closing_price, profit_pct, signal_id)
            )
            if cur.rowcount == 0: logger.warning(f"⚠️ [DB Close] Signal {signal_id} was already closed or not found."); return
        conn.commit()
        status_map = {
            'target_hit': '✅ تحقق الهدف',
            'stop_loss_hit': '🛑 ضرب وقف الخسارة',
            'manual_close': '🖐️ أُغلقت يدوياً',
            'closed_by_sell_signal': '🔴 أُغلقت بإشارة بيع'
        }
        status_message = status_map.get(status, status)
        alert_msg = f"*{status_message}*\n`{symbol.replace('_', '\\_')}` | *الربح:* `{profit_pct:+.2f}%`"
        send_telegram_message(CHAT_ID, alert_msg)
        log_and_notify('info', f"{status_message}: {symbol} | Profit: {profit_pct:+.2f}%", 'CLOSE_SIGNAL')
        logger.info(f"✅ [DB Close] Signal {signal_id} closed successfully.")
    except Exception as e:
        logger.error(f"❌ [DB Close] Critical error closing signal {signal_id}: {e}", exc_info=True)
        if conn: conn.rollback()
        if symbol:
            with signal_cache_lock:
                if symbol not in open_signals_cache: open_signals_cache[symbol] = signal; logger.info(f"🔄 [Recovery] Signal {signal_id} restored to cache due to error.")
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

# ---------------------- حلقة العمل الرئيسية ----------------------
def get_btc_data_for_bot() -> Optional[pd.DataFrame]:
    btc_data = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
    if btc_data is None: logger.error("❌ [BTC Data] Failed to fetch Bitcoin data."); return None
    btc_data['btc_returns'] = btc_data['close'].pct_change()
    return btc_data

def main_loop():
    logger.info("[Main Loop] Waiting for initialization to complete...")
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
                
                # --- ✨ IMPROVED: BTC Trend Filter using the new market state ---
                if USE_BTC_TREND_FILTER and market_regime in ["DOWNTREND", "STRONG DOWNTREND"]:
                    log_rejection("ALL", "BTC Trend Filter", {"detail": f"Scan paused due to market regime: {market_regime}"})
                    time.sleep(300)
                    break
                
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
                        if not all([strategy.ml_model, strategy.scaler, strategy.feature_names]):
                            continue

                        df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        if df_4h is None: continue

                        df_features = strategy.get_features(df_15m, df_4h, btc_data)
                        if df_features is None or df_features.empty: continue
                        
                        signal_info = strategy.generate_signal(df_features)
                        if not signal_info or not redis_client: continue

                        current_price_str = redis_client.hget(REDIS_PRICES_HASH_NAME, symbol)
                        if not current_price_str: continue
                        current_price = float(current_price_str)
                        
                        prediction = signal_info['prediction']
                        confidence = signal_info['confidence']

                        with signal_cache_lock:
                            is_trade_open = symbol in open_signals_cache

                        if is_trade_open:
                            open_signal = open_signals_cache[symbol]
                            if prediction == -1: # SELL Signal
                                profit_check_price = float(open_signal['entry_price']) * (1 + MIN_PROFIT_FOR_SELL_CLOSE_PERCENT / 100)
                                
                                if confidence < SELL_CONFIDENCE_THRESHOLD:
                                    logger.info(f"ℹ️ [Sell Ignore] {symbol}: Confidence {confidence:.2%} is below threshold {SELL_CONFIDENCE_THRESHOLD:.2%}.")
                                    continue
                                
                                if current_price < profit_check_price:
                                    logger.info(f"ℹ️ [Sell Ignore] {symbol}: Price {current_price} has not reached minimum profit level {profit_check_price}.")
                                    continue

                                logger.info(f"✅ [Action] Closing open trade for {symbol} due to new SELL signal with sufficient confidence and profit.")
                                initiate_signal_closure(symbol, open_signal, 'closed_by_sell_signal', current_price)
                                send_telegram_message(CHAT_ID, f"🔴 *إغلاق بإشارة بيع*\n`{symbol}`\nتم إغلاق الصفقة المفتوحة بسعر السوق لتحقيق الشروط الجديدة.")
                                continue

                            elif prediction == 1 and confidence >= BUY_CONFIDENCE_THRESHOLD: # New BUY signal on open trade
                                logger.info(f"ℹ️ [Action] Checking for TP update for {symbol} due to new BUY signal.")
                                last_atr = df_features.iloc[-1].get('atr', 0)
                                tp_sl_data = calculate_db_driven_tp_sl(symbol, current_price, last_atr)

                                if tp_sl_data and float(tp_sl_data['target_price']) > float(open_signal['target_price']):
                                    new_tp = float(tp_sl_data['target_price'])
                                    new_sl = float(tp_sl_data['stop_loss'])
                                    
                                    if update_signal_target_in_db(open_signal['id'], new_tp, new_sl):
                                        open_signals_cache[symbol]['target_price'] = new_tp
                                        open_signals_cache[symbol]['stop_loss'] = new_sl
                                        send_telegram_message(CHAT_ID, f"🔼 *تحديث الهدف*\n`{symbol}`\n*الهدف الجديد:* `${new_tp:,.8g}`\n*الوقف الجديد:* `${new_sl:,.8g}`")
                                else:
                                    logger.info(f"ℹ️ [Info] New BUY signal for {symbol} did not result in a higher target. No action taken.")
                                continue
                        
                        elif not is_trade_open and prediction == 1 and confidence >= BUY_CONFIDENCE_THRESHOLD:
                            if slots_available <= 0: continue

                            last_features = df_features.iloc[-1]; last_features.name = symbol
                            if USE_SPEED_FILTER and not passes_speed_filter(last_features): continue
                            
                            last_atr = last_features.get('atr', 0)
                            volatility = (last_atr / current_price * 100)
                            if USE_MIN_VOLATILITY_FILTER and volatility < MIN_VOLATILITY_PERCENT:
                                log_rejection(symbol, "Low Volatility Filter", {"volatility": f"{volatility:.2f}%", "min_required": f"{MIN_VOLATILITY_PERCENT}%"})
                                continue

                            if USE_BTC_CORRELATION_FILTER and market_regime in ["UPTREND", "STRONG UPTREND"]:
                                correlation = last_features.get('btc_correlation', 0)
                                if correlation < MIN_BTC_CORRELATION:
                                    log_rejection(symbol, "BTC Correlation Filter", {"correlation": f"{correlation:.2f}", "min_required": f"{MIN_BTC_CORRELATION}"})
                                    continue
                            
                            tp_sl_data = calculate_db_driven_tp_sl(symbol, current_price, last_atr)
                            if not tp_sl_data: continue

                            new_signal = {
                                'symbol': symbol,
                                'strategy_name': BASE_ML_MODEL_NAME,
                                'signal_details': {'ML_Confidence': f"{confidence:.2%}", 'TP_SL_Source': tp_sl_data['source']},
                                'entry_price': current_price,
                                **tp_sl_data
                            }

                            if USE_RRR_FILTER:
                                tp = float(new_signal['target_price'])
                                sl = float(new_signal['stop_loss'])
                                risk = current_price - sl
                                reward = tp - current_price
                                if risk <= 0 or reward <= 0: continue
                                rrr = reward / risk
                                if rrr < MIN_RISK_REWARD_RATIO:
                                    log_rejection(symbol, "Risk/Reward Ratio Filter", {"RRR": f"{rrr:.2f}", "min_required": f"{MIN_RISK_REWARD_RATIO}"})
                                    continue

                            logger.info(f"✅ [{symbol}] Signal passed all filters. Saving...")
                            saved_signal = insert_signal_into_db(new_signal)
                            if saved_signal:
                                with signal_cache_lock: open_signals_cache[saved_signal['symbol']] = saved_signal
                                send_new_signal_alert(saved_signal)
                                slots_available -= 1
                        
                        del df_15m, df_4h, df_features; gc.collect()

                    except Exception as e:
                        logger.error(f"❌ [Processing Error] {symbol}: {e}", exc_info=True)
                time.sleep(10)
            logger.info("ℹ️ [End of Cycle] Scan cycle finished. Waiting..."); 
            time.sleep(60)
        except (KeyboardInterrupt, SystemExit): break
        except Exception as main_err:
            log_and_notify("error", f"Error in main loop: {main_err}", "SYSTEM"); time.sleep(120)

# ---------------------- واجهة برمجة تطبيقات Flask ----------------------
app = Flask(__name__)
CORS(app)

def get_fear_and_greed_index() -> Dict[str, Any]:
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10).json()
        value = int(response['data'][0]['value'])
        classification = response['data'][0]['value_classification']
        return {"value": value, "classification": classification}
    except Exception: return {"value": -1, "classification": "Error"}

@app.route('/')
def home():
    try:
        return render_template_string(get_dashboard_html())
    except Exception as e:
        logger.error(f"Error rendering homepage: {e}")
        return "<h1>An error occurred.</h1>", 500

@app.route('/api/market_status')
def get_market_status():
    with market_state_lock:
        # Make a copy to avoid race conditions during serialization
        state_copy = dict(current_market_state)
    return jsonify({
        "fear_and_greed": get_fear_and_greed_index(),
        "market_state": state_copy
    })

@app.route('/api/stats')
def get_stats():
    if not check_db_connection() or not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT status, profit_percentage FROM signals;")
            all_signals = cur.fetchall()
        
        with signal_cache_lock:
            open_trades_count = len(open_signals_cache)

        closed_trades = [s for s in all_signals if s.get('status') != 'open' and s.get('profit_percentage') is not None]
        
        wins = [s for s in closed_trades if s['status'] == 'target_hit']
        losses = [s for s in closed_trades if s['status'] == 'stop_loss_hit']
        
        total_profit_pct = sum(float(s['profit_percentage']) for s in closed_trades)
        
        win_count = len(wins)
        loss_count = len(losses)
        total_closed = win_count + loss_count
        
        win_rate = (win_count / total_closed * 100) if total_closed > 0 else 0
        
        total_profit_from_wins = sum(float(s['profit_percentage']) for s in wins)
        total_loss_from_losses = abs(sum(float(s['profit_percentage']) for s in losses))
        
        profit_factor = (total_profit_from_wins / total_loss_from_losses) if total_loss_from_losses > 0 else 0
        
        avg_win_pct = (total_profit_from_wins / win_count) if win_count > 0 else 0
        avg_loss_pct = (total_loss_from_losses / loss_count) if loss_count > 0 else 0


        return jsonify({
            "open_trades_count": open_trades_count,
            "total_profit_pct": total_profit_pct,
            "total_closed_trades": len(closed_trades),
            "wins": win_count,
            "losses": loss_count,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win_pct": avg_win_pct,
            "avg_loss_pct": -avg_loss_pct 
        })
    except Exception as e:
        logger.error(f"❌ [API Stats] Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/signals')
def get_signals():
    if not check_db_connection() or not conn or not redis_client: return jsonify({"error": "Failed to connect to services"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals ORDER BY CASE WHEN status = 'open' THEN 0 ELSE 1 END, id DESC;")
            all_signals = cur.fetchall()
        open_symbols = [s['symbol'] for s in all_signals if s['status'] == 'open']
        current_prices = {}
        if open_symbols:
            prices_list = redis_client.hmget(REDIS_PRICES_HASH_NAME, open_symbols)
            current_prices = {symbol: float(p) if p else None for symbol, p in zip(open_symbols, prices_list)}
        for s in all_signals:
            if s.get('closed_at'): s['closed_at'] = s['closed_at'].isoformat()
            if s['status'] == 'open':
                price = current_prices.get(s['symbol'])
                s['current_price'] = price
                if price and s.get('entry_price') and float(s.get('entry_price')) > 0: 
                    s['pnl_pct'] = ((price / float(s['entry_price'])) - 1) * 100
        return jsonify(all_signals)
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/api/close/<int:signal_id>', methods=['POST'])
def manual_close_signal(signal_id):
    if not client: return jsonify({"error": "Binance Client not available"}), 500
    with closure_lock:
        if signal_id in signals_pending_closure: return jsonify({"error": "Signal is already being closed"}), 409
    if not check_db_connection() or not conn: return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE id = %s AND status = 'open';", (signal_id,))
            signal_to_close = cur.fetchone()
        if not signal_to_close: return jsonify({"error": "Signal not found"}), 404
        symbol = signal_to_close['symbol']
        price = float(client.get_symbol_ticker(symbol=symbol)['price'])
        initiate_signal_closure(symbol, dict(signal_to_close), 'manual_close', price)
        return jsonify({"message": f"Closing signal {signal_id}..."})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/api/test_telegram')
def test_telegram():
    logger.info("API: Received request to test Telegram.")
    message = "👋 رسالة اختبار من بوت التداول الخاص بك. إذا رأيت هذه الرسالة، فالاتصال سليم!"
    if send_telegram_message(CHAT_ID, message):
        return "✅ Test message sent successfully. Please check your Telegram."
    else:
        return "❌ Failed to send test message. Please check logs for errors and verify your Token/Chat ID.", 500

@app.route('/api/notifications')
def get_notifications():
    with notifications_lock: return jsonify(list(notifications_cache))

@app.route('/api/rejection_logs')
def get_rejection_logs():
    with rejection_logs_lock:
        return jsonify(list(rejection_logs_cache))

def run_flask():
    host, port = "0.0.0.0", int(os.environ.get('PORT', 10000))
    logger.info(f"Attempting to start dashboard on {host}:{port}")
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
    
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.critical("❌ [Telegram] Token or Chat ID not found in environment variables. Alerts will not be sent.")
    else:
        logger.info("✅ [Telegram] Token and Chat ID loaded successfully.")

    try:
        client = Client(API_KEY, API_SECRET)
        logger.info("✅ [Binance] Connected to Binance API successfully.")
        
        init_db()
        init_redis()
        
        load_open_signals_to_cache()
        load_notifications_to_cache()
        
        # Run initial market state check
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
    logger.info(f"🚀 Starting Trading Bot - Improved Version...")
    
    initialization_thread = Thread(target=initialize_bot_services, daemon=True)
    initialization_thread.start()
    
    run_flask()
    
    logger.info("👋 [Shutdown] Bot has been shut down."); os._exit(0)
