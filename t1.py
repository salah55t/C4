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
from decimal import Decimal
from urllib.parse import urlparse
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timedelta, timezone
from decouple import config
from typing import List, Dict, Optional, Any, Set, Tuple
from sklearn.preprocessing import StandardScaler
from collections import deque
import warnings

# --- Ignore irrelevant warnings ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ---------------------- Logging Setup - V29.0 (Signal Tracker) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_tracker_v29_0_arabic_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoTrackerV29.0')

# ---------------------- Environment Variables Loading ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
    REDIS_URL: str = config('REDIS_URL', default='redis://localhost:6379/0')
except Exception as e:
    logger.critical(f"❌ Critical failure in loading essential environment variables: {e}")
    exit(1)

# ---------------------- Constants and Global Variables ----------------------
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V8_With_Momentum'
MODEL_FOLDER: str = 'V8'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
TIMEFRAMES_FOR_TREND_ANALYSIS: List[str] = ['15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 30
REDIS_PRICES_HASH_NAME: str = "crypto_tracker_current_prices_v29"
BTC_SYMBOL: str = 'BTCUSDT'
ADX_PERIOD: int = 14; RSI_PERIOD: int = 14; ATR_PERIOD: int = 14
EMA_PERIODS: List[int] = [21, 50, 200]
REL_VOL_PERIOD: int = 30; MOMENTUM_PERIOD: int = 12; EMA_SLOPE_PERIOD: int = 5
BUY_CONFIDENCE_THRESHOLD = 0.80 # Threshold for ML model approval
ATR_SL_MULTIPLIER: float = 1.5
ATR_TP_MULTIPLIER: float = 2.2
PEAK_TIMEFRAME: str = '1h'
PEAK_LOOKBACK_CANDLES: int = 20

# --- System Components ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
ml_models_cache: Dict[str, Any] = {}
exchange_info_map: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
tracked_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
closure_lock = Lock()
signals_pending_closure: Set[int] = set()
current_market_state: Dict[str, Any] = {
    "trend_score": 0, "trend_label": "INITIALIZING",
    "details_by_tf": {}, "last_updated": None
}
market_state_lock = Lock()


# ---------------------- Dashboard HTML ----------------------
def get_dashboard_html():
    # UPDATE: Added new buttons and tabs for successful and failed signals.
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تتبع التوصيات V29.0</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Tajawal', sans-serif; background-color: #0D1117; color: #E6EDF3; }
        .card { background-color: #161B22; border: 1px solid #30363D; border-radius: 0.5rem; }
        .tab-btn.active { color: #58A6FF; border-bottom-color: #58A6FF; }
        .details-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 0.75rem; }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap" rel="stylesheet">
</head>
<body class="p-4 md:p-6">
    <div class="container mx-auto max-w-screen-xl">
        <header class="mb-6">
            <h1 class="text-2xl md:text-3xl font-bold text-white">
                لوحة تتبع توصيات النموذج
                <span class="text-base font-medium text-gray-400">V29.0</span>
            </h1>
            <p class="text-gray-400">نظام لرصد وتسجيل أداء توصيات نموذج التعلم الآلي بشكل سلبي.</p>
        </header>

        <div class="mb-4 border-b border-gray-700">
            <nav class="flex flex-wrap -mb-px space-x-6 space-x-reverse">
                <button onclick="showTab('tracking', this)" class="tab-btn active py-3 px-1 border-b-2 font-semibold">توصيات قيد التتبع</button>
                <button onclick="showTab('successful', this)" class="tab-btn py-3 px-1 border-b-2 border-transparent text-gray-400 hover:text-white">التوصيات الناجحة</button>
                <button onclick="showTab('failed', this)" class="tab-btn py-3 px-1 border-b-2 border-transparent text-gray-400 hover:text-white">التوصيات الخاسرة</button>
                <button onclick="showTab('closed', this)" class="tab-btn py-3 px-1 border-b-2 border-transparent text-gray-400 hover:text-white">كل التوصيات المغلقة</button>
            </nav>
        </div>

        <main>
            <div id="tracking-tab" class="tab-content space-y-4"></div>
            <div id="successful-tab" class="tab-content hidden space-y-4"></div>
            <div id="failed-tab" class="tab-content hidden space-y-4"></div>
            <div id="closed-tab" class="tab-content hidden space-y-4"></div>
        </main>
    </div>

<script>
    let activeTab = 'tracking';

    function formatNumber(num, digits = 2) {
        if (num === null || num === undefined || isNaN(num)) return 'N/A';
        return num.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
    }

    function showTab(tabName, element) {
        activeTab = tabName;
        document.querySelectorAll('.tab-content').forEach(tab => tab.classList.add('hidden'));
        document.getElementById(`${tabName}-tab`).classList.remove('hidden');
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
            btn.classList.remove('font-semibold');
        });
        element.classList.add('active', 'font-semibold');
        refreshData();
    }

    function renderSignalCard(signal) {
        const isTracking = signal.status === 'tracking';
        const pnl = isTracking ? ((signal.current_price / signal.entry_price) - 1) * 100 : signal.profit_percentage;
        const pnlColor = pnl >= 0 ? 'text-green-400' : 'text-red-400';
        const borderColor = isTracking ? 'border-yellow-500/30' : (pnl >= 0 ? 'border-green-500/30' : 'border-red-500/30');
        
        const totalDist = signal.target_price - signal.stop_loss;
        const progressPct = totalDist > 0 ? Math.max(0, Math.min(100, ((signal.current_price - signal.stop_loss) / totalDist) * 100)) : 0;

        let detailsHTML = '<div class="details-grid mt-3">';
        const signalDetails = signal.signal_context || {};
        const allFilters = signalDetails.all_filters_data || {};
        
        const itemsToShow = {
            'ثقة النموذج': formatNumber(signalDetails.ml_confidence * 100, 2) + '%',
            'المسافة من قمة 1H': formatNumber(signalDetails.distance_from_peak_1h_pct, 2) + '%',
            'نقاط اتجاه السوق': signalDetails.market_trend_score,
            'ADX': formatNumber(allFilters.adx),
            'RSI': formatNumber(allFilters.rsi),
            'ROC(12)': formatNumber(allFilters.roc_12),
        };

        for (const [key, value] of Object.entries(itemsToShow)) {
            detailsHTML += `<div class="bg-gray-900/50 p-2 rounded text-center"><div class="text-xs text-gray-400 uppercase">${key}</div><div class="font-mono font-semibold">${value !== undefined && value !== null ? value : 'N/A'}</div></div>`;
        }
        detailsHTML += '</div>';

        return `
            <div class="card p-4 border-l-4 ${borderColor}">
                <div class="flex flex-wrap justify-between items-center gap-2">
                    <div>
                        <span class="font-bold text-xl">${signal.symbol}</span>
                        <span class="text-xs text-gray-400 ml-2">${new Date(signal.created_at).toLocaleString('ar-EG')}</span>
                    </div>
                    <div class="text-2xl font-bold ${pnlColor}">${formatNumber(pnl)}%</div>
                </div>
                ${isTracking ? `
                <div class="mt-2 text-xs text-gray-400 grid grid-cols-3 gap-2 text-center">
                    <div>وقف: <span class="font-mono">${formatNumber(signal.stop_loss, 4)}</span></div>
                    <div class="font-semibold">حالي: <span class="font-mono">${formatNumber(signal.current_price, 4)}</span></div>
                    <div>هدف: <span class="font-mono">${formatNumber(signal.target_price, 4)}</span></div>
                </div>
                <div class="mt-2 w-full bg-gray-700 rounded-full h-1.5"><div class="bg-blue-500 h-1.5 rounded-full" style="width: ${progressPct}%"></div></div>
                ` : `<div class="mt-2 text-sm">أغلقت بسعر <span class="font-mono">${formatNumber(signal.closing_price, 4)}</span> بتاريخ ${new Date(signal.closed_at).toLocaleString('ar-EG')}</div>`}
                <div class="mt-2 border-t border-gray-700 pt-2">
                    <h4 class="font-semibold text-gray-300">بيانات التوصية عند الإنشاء:</h4>
                    ${detailsHTML}
                </div>
            </div>
        `;
    }

    function refreshData() {
        // UPDATE: The endpoint now dynamically uses the activeTab variable.
        const endpoint = '/api/signals?status=' + activeTab;
        const container = document.getElementById(`${activeTab}-tab`);
        
        fetch(endpoint)
            .then(res => res.json())
            .then(data => {
                if (!data || data.error) {
                    container.innerHTML = '<p class="text-center text-red-500">فشل تحميل البيانات.</p>';
                    return;
                }
                if (data.length === 0) {
                    container.innerHTML = '<p class="text-center text-gray-400 p-8">لا توجد توصيات لعرضها في هذا القسم.</p>';
                    return;
                }
                container.innerHTML = data.map(renderSignalCard).join('');
            });
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
    db_url = DB_URL + ("?sslmode=require" if 'postgres' in DB_URL and 'sslmode' not in DB_URL else "")
    
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(db_url, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS tracked_signals (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        entry_price DOUBLE PRECISION NOT NULL,
                        target_price DOUBLE PRECISION NOT NULL,
                        stop_loss DOUBLE PRECISION NOT NULL,
                        status TEXT DEFAULT 'tracking', -- tracking, target_hit, stop_loss_hit
                        closing_price DOUBLE PRECISION,
                        closed_at TIMESTAMP,
                        profit_percentage DOUBLE PRECISION,
                        distance_from_peak_1h_pct DOUBLE PRECISION,
                        signal_context JSONB
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_tracked_signals_status ON tracked_signals (status);")
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

def insert_tracked_signal_into_db(signal_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not check_db_connection() or not conn: return None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO tracked_signals (
                    symbol, entry_price, target_price, stop_loss, 
                    distance_from_peak_1h_pct, signal_context
                ) VALUES (%s, %s, %s, %s, %s, %s) RETURNING *;
            """, (
                signal_data['symbol'],
                float(signal_data['entry_price']),
                float(signal_data['target_price']),
                float(signal_data['stop_loss']),
                float(signal_data['distance_from_peak_1h_pct']),
                json.dumps(signal_data.get('signal_context', {}))
            ))
            inserted_signal = dict(cur.fetchone())
        conn.commit()
        logger.info(f"💾 [TRACKER] New signal for {inserted_signal['symbol']} is now being tracked (ID: {inserted_signal['id']}).")
        return inserted_signal
    except Exception as e:
        logger.error(f"❌ [Insert] Error inserting tracked signal for {signal_data['symbol']}: {e}", exc_info=True)
        if conn: conn.rollback()
        return None

def close_tracked_signal(signal: Dict, status: str, closing_price: float):
    signal_id = signal.get('id')
    with closure_lock:
        if signal_id in signals_pending_closure: return
        signals_pending_closure.add(signal_id)
    
    try:
        if not check_db_connection() or not conn: raise OperationalError("DB connection failed.")
        profit_pct = ((float(closing_price) / float(signal['entry_price'])) - 1) * 100
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE tracked_signals 
                SET status = %s, closing_price = %s, closed_at = NOW(), profit_percentage = %s 
                WHERE id = %s AND status = 'tracking';
            """, (status, float(closing_price), profit_pct, signal_id))
            if cur.rowcount == 0:
                logger.warning(f"⚠️ [DB Close] Tracked signal {signal_id} was already closed or not found."); return
        conn.commit()
        
        status_map = {'target_hit': '✅ تحقق الهدف', 'stop_loss_hit': '🛑 ضرب وقف الخسارة'}
        logger.info(f"📈 [TRACKER] Closed signal {signal_id} ({signal['symbol']}) | Status: {status_map.get(status, status)} | PnL: {profit_pct:+.2f}%")
    except Exception as e:
        logger.error(f"❌ [DB Close] Critical error closing tracked signal {signal_id}: {e}", exc_info=True)
        if conn: conn.rollback()
    finally:
        with signal_cache_lock: tracked_signals_cache.pop(signal.get('symbol'), None)
        with closure_lock: signals_pending_closure.discard(signal_id)


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
        limit = int((days * 24 * 60) / int(re.sub('[a-zA-Z]', '', interval)))
        klines = client.get_historical_klines(symbol, interval, limit=min(limit, 1000))
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna()
    except Exception as e:
        logger.error(f"❌ [Data] Error fetching historical data for {symbol}: {e}")
        return None

# ---------------------- Feature and Logic Functions ----------------------

def calculate_distance_from_peak(symbol: str, current_price: float) -> Optional[float]:
    try:
        days_to_fetch = int((PEAK_LOOKBACK_CANDLES * 60) / (24 * 60)) + 2
        df_peak = fetch_historical_data(symbol, PEAK_TIMEFRAME, days=days_to_fetch)
        
        if df_peak is None or len(df_peak) < PEAK_LOOKBACK_CANDLES:
            logger.warning(f"⚠️ [{symbol}] Not enough data for peak calculation ({len(df_peak) if df_peak is not None else 0} candles).")
            return None
        
        peak_high = df_peak.iloc[-PEAK_LOOKBACK_CANDLES-1:-1]['high'].max()
        
        if pd.isna(peak_high) or peak_high <= 0:
            return None
            
        distance_pct = ((current_price / peak_high) - 1) * 100
        return float(distance_pct)

    except Exception as e:
        logger.error(f"❌ [{symbol}] Error in calculate_distance_from_peak: {e}")
        return None

def calculate_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df_calc = df.copy()
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
    global current_market_state, market_state_lock
    with market_state_lock:
        logger.info("🧠 [Market Score] Updating multi-timeframe trend score...")
        try:
            total_score, details, tf_weights = 0, {}, {'15m': 0.2, '1h': 0.3, '4h': 0.5}
            for tf in TIMEFRAMES_FOR_TREND_ANALYSIS:
                days_to_fetch = 5 if tf == '15m' else (15 if tf == '1h' else 50)
                df = fetch_historical_data(BTC_SYMBOL, tf, days_to_fetch)
                if df is None or len(df) < EMA_PERIODS[-1]:
                    details[tf] = {"score": 0, "label": "غير واضح"}; continue
                for period in EMA_PERIODS: df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
                last = df.iloc[-1]
                tf_score = (1 if last.close > last.ema_21 else -1) + (1 if last.ema_21 > last.ema_50 else -1) + (1 if last.ema_50 > last.ema_200 else -1)
                details[tf] = {"score": tf_score, "label": "صاعد" if tf_score >= 2 else ("هابط" if tf_score <= -2 else "محايد")}
                total_score += tf_score * tf_weights[tf]
            final_score = round(total_score)
            trend_label = "صاعد قوي" if final_score >= 4 else ("صاعد" if final_score >= 1 else ("هابط قوي" if final_score <= -4 else ("هابط" if final_score <= -1 else "محايد")))
            current_market_state = {"trend_score": final_score, "trend_label": trend_label, "details_by_tf": details, "last_updated": datetime.now(timezone.utc).isoformat()}
            logger.info(f"✅ [Market Score] New State: Score={final_score}, Label='{trend_label}'")
        except Exception as e:
            logger.error(f"❌ [Market Score] Failed to determine market state: {e}", exc_info=True)

class TradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = self.load_ml_model_bundle_from_folder(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)
    def load_ml_model_bundle_from_folder(self, symbol: str) -> Optional[Dict[str, Any]]:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FOLDER, f"{BASE_ML_MODEL_NAME}_{symbol}.pkl")
        if not os.path.exists(model_path): return None
        try:
            with open(model_path, 'rb') as f: return pickle.load(f)
        except Exception as e:
            logger.error(f"❌ [ML Model] Error loading model for {symbol}: {e}"); return None
    def get_features(self, df_15m: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.feature_names is None: return None
        try:
            df_featured = calculate_features(df_15m, btc_df)
            for col in self.feature_names:
                if col not in df_featured.columns: df_featured[col] = 0.0
            return df_featured.dropna(subset=self.feature_names)
        except Exception: return None
    def generate_buy_signal(self, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]) or df_features.empty: return None
        try:
            features_scaled = self.scaler.transform(df_features.iloc[[-1]][self.feature_names])
            if self.ml_model.predict(features_scaled)[0] != 1: return None
            confidence = float(np.max(self.ml_model.predict_proba(features_scaled)[0]))
            if confidence >= BUY_CONFIDENCE_THRESHOLD:
                return {'prediction': 1, 'confidence': confidence}
            return None
        except Exception: return None

# ---------------------- Main Application Loops ----------------------

def signal_monitoring_loop():
    logger.info("✅ [Signal Monitor] Starting signal monitoring loop.")
    while True:
        try:
            with signal_cache_lock: signals_to_check = dict(tracked_signals_cache)
            if not signals_to_check or not redis_client:
                time.sleep(1); continue
            
            symbols_to_fetch = list(signals_to_check.keys())
            redis_prices_list = redis_client.hmget(REDIS_PRICES_HASH_NAME, symbols_to_fetch)
            
            for symbol, signal in signals_to_check.items():
                price_str = redis_prices_list.pop(0)
                if not price_str: continue
                
                price = float(price_str)
                with signal_cache_lock:
                    if symbol in tracked_signals_cache:
                        tracked_signals_cache[symbol]['current_price'] = price
                
                status_to_set = None
                if price >= float(signal['target_price']): status_to_set = 'target_hit'
                elif price <= float(signal['stop_loss']): status_to_set = 'stop_loss_hit'
                
                if status_to_set:
                    close_tracked_signal(signal, status_to_set, price)
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"❌ [Signal Monitor] Critical error: {e}", exc_info=True)
            time.sleep(5)

def main_scanner_loop():
    logger.info("[Main Scanner] Waiting for initialization...")
    time.sleep(10)
    if not validated_symbols_to_scan:
        logger.critical("No validated symbols to scan. Bot will not start."); return
    logger.info(f"✅ Starting main scanner loop for {len(validated_symbols_to_scan)} symbols.")

    while True:
        try:
            logger.info("🔄 Starting new scanner cycle...")
            determine_market_trend_score()
            
            with signal_cache_lock:
                symbols_already_tracked = set(tracked_signals_cache.keys())
            
            symbols_to_process = [s for s in validated_symbols_to_scan if s not in symbols_already_tracked]
            
            for symbol in random.sample(symbols_to_process, len(symbols_to_process)):
                try:
                    strategy = TradingStrategy(symbol)
                    if not strategy.ml_model: continue
                    
                    df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_15m is None: continue
                    
                    btc_data = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, 5)
                    if btc_data is not None: btc_data['btc_returns'] = btc_data['close'].pct_change()

                    df_features = strategy.get_features(df_15m, btc_data)
                    if df_features is None or df_features.empty: continue
                    
                    ml_signal = strategy.generate_buy_signal(df_features)
                    if not ml_signal: continue
                    
                    logger.info(f"💡 [ML Signal] Model approved BUY for {symbol} with confidence {ml_signal['confidence']:.2%}")
                    
                    entry_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                    last_atr = df_features.iloc[-1].get('atr', 0)
                    if last_atr <= 0: continue

                    distance_pct = calculate_distance_from_peak(symbol, entry_price)
                    if distance_pct is None: 
                        logger.warning(f"[{symbol}] Could not calculate distance from peak. Skipping signal.")
                        continue

                    with market_state_lock: market_state_copy = dict(current_market_state)
                    last_features = df_features.iloc[-1]
                    all_filters_data = {k: (float(v) if pd.notna(v) and np.isfinite(v) else None) for k, v in last_features.items()}

                    signal_context = {
                        'ml_confidence': ml_signal['confidence'],
                        'market_trend_score': market_state_copy.get('trend_score'),
                        'market_trend_details': market_state_copy.get('details_by_tf'),
                        'distance_from_peak_1h_pct': distance_pct,
                        'all_filters_data': all_filters_data
                    }
                    
                    new_signal_data = {
                        'symbol': symbol,
                        'entry_price': entry_price,
                        'target_price': entry_price + (last_atr * ATR_TP_MULTIPLIER),
                        'stop_loss': entry_price - (last_atr * ATR_SL_MULTIPLIER),
                        'distance_from_peak_1h_pct': distance_pct,
                        'signal_context': signal_context
                    }
                    
                    saved_signal = insert_tracked_signal_into_db(new_signal_data)
                    if saved_signal:
                        with signal_cache_lock:
                            tracked_signals_cache[saved_signal['symbol']] = saved_signal
                
                except Exception as e:
                    logger.error(f"❌ [Processing Error] for symbol {symbol}: {e}", exc_info=True)
                finally:
                    time.sleep(2)
            
            logger.info("✅ [End of Cycle] Full scan cycle finished. Waiting for 60 seconds...")
            time.sleep(60)

        except (KeyboardInterrupt, SystemExit):
            logger.info("Bot is shutting down by user request."); break
        except Exception as main_err:
            logger.error(f"Critical error in main loop: {main_err}", exc_info=True); time.sleep(120)

# ---------------------- Flask API ----------------------
app = Flask(__name__)
CORS(app)

@app.route('/')
def home(): return render_template_string(get_dashboard_html())

@app.route('/api/signals')
def get_signals():
    # UPDATE: Added logic to handle 'successful' and 'failed' statuses.
    status = request.args.get('status', 'tracking')
    if not check_db_connection(): return jsonify({"error": "DB connection failed"}), 500
    
    try:
        with conn.cursor() as cur:
            if status == 'successful':
                cur.execute("SELECT * FROM tracked_signals WHERE status = 'target_hit' ORDER BY closed_at DESC;")
            elif status == 'failed':
                cur.execute("SELECT * FROM tracked_signals WHERE status = 'stop_loss_hit' ORDER BY closed_at DESC;")
            elif status == 'closed':
                cur.execute("SELECT * FROM tracked_signals WHERE status IN ('target_hit', 'stop_loss_hit') ORDER BY closed_at DESC;")
            else: # 'tracking'
                cur.execute("SELECT * FROM tracked_signals WHERE status = 'tracking' ORDER BY created_at DESC;")
            
            signals = [dict(s) for s in cur.fetchall()]

        if status == 'tracking':
            symbols = [s['symbol'] for s in signals]
            if symbols and redis_client:
                prices = {s: p for s, p in zip(symbols, redis_client.hmget(REDIS_PRICES_HASH_NAME, symbols)) if p}
                for s in signals:
                    s['current_price'] = float(prices.get(s['symbol'], s['entry_price']))
        
        return jsonify(signals)
    except Exception as e:
        if conn: conn.rollback()
        logger.error(f"❌ [API Error] Failed to get signals for status '{status}': {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ---------------------- Program Startup ----------------------

def handle_price_update_message(msg: Dict[str, Any]):
    global redis_client
    try:
        if msg and 'e' in msg and msg['e'] == 'error':
            logger.error(f"❌ [WebSocket Error] Message: {msg['m']}")
        elif msg and 's' in msg and 'c' in msg:
            symbol = msg['s']
            price = msg['c']
            if redis_client:
                redis_client.hset(REDIS_PRICES_HASH_NAME, symbol, price)
    except Exception as e:
        logger.error(f"❌ [WebSocket Handler] Error processing message: {e} - Data: {msg}")

def run_websocket_manager():
    if not client or not validated_symbols_to_scan: return
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    streams = [f"{s.lower()}@miniTicker" for s in validated_symbols_to_scan]
    twm.start_multiplex_socket(callback=handle_price_update_message, streams=streams)
    logger.info(f"✅ [WebSocket] Subscribed to {len(streams)} price streams.")
    twm.join()

def load_tracked_signals_to_cache():
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM tracked_signals WHERE status = 'tracking';")
            open_signals = cur.fetchall()
            with signal_cache_lock:
                tracked_signals_cache.clear()
                for signal in open_signals:
                    tracked_signals_cache[signal['symbol']] = dict(signal)
            logger.info(f"✅ [Loading] Loaded {len(open_signals)} signals to track.")
    except Exception as e:
        logger.error(f"❌ [Loading] Failed to load tracked signals: {e}")

def initialize_bot_services():
    global client, validated_symbols_to_scan
    logger.info("🤖 [Bot Services] Starting background initialization...")
    try:
        client = Client(API_KEY, API_SECRET)
        init_db()
        init_redis()
        get_exchange_info_map()
        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan:
            logger.critical("❌ No validated symbols to scan. Bot will not start."); return
        
        load_tracked_signals_to_cache()
        
        Thread(target=run_websocket_manager, daemon=True).start()
        Thread(target=signal_monitoring_loop, daemon=True).start()
        Thread(target=main_scanner_loop, daemon=True).start()
        logger.info("✅ [Bot Services] All background services started successfully.")
    except Exception as e:
        logger.critical(f"A critical error occurred during initialization: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    logger.info("🚀 LAUNCHING SIGNAL TRACKER & DASHBOARD (V29.0) 🚀")
    initialization_thread = Thread(target=initialize_bot_services, daemon=True)
    initialization_thread.start()
    port = int(os.environ.get('PORT', 10000))
    app.run(host="0.0.0.0", port=port)
    logger.info("👋 [Shutdown] Application has been shut down."); os._exit(0)
