import time
import os
import json
import logging
import numpy as np
import pandas as pd
import psycopg2
import pickle
import gc
import random
from urllib.parse import urlparse
from psycopg2 import sql
from psycopg2.extras import RealDictCursor, execute_values
from binance.client import Client
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from threading import Thread
from datetime import datetime, timezone
from decouple import config
from typing import List, Dict, Optional, Any, Tuple
from sklearn.preprocessing import StandardScaler
import warnings

# --- تجاهل التحذيرات غير الهامة ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtester_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Backtester')

# ---------------------- تحميل متغيرات البيئة ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
    exit(1)

# ---------------------- الثوابت والمتغيرات العامة للاختبار ----------------------
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V8_With_Momentum'
MODEL_FOLDER: str = 'V8'
TIMEFRAMES_FOR_TREND_ANALYSIS: List[str] = ['15m', '1h', '4h']
BTC_SYMBOL: str = 'BTCUSDT'
ADX_PERIOD: int = 14; RSI_PERIOD: int = 14; ATR_PERIOD: int = 14
EMA_PERIODS: List[int] = [21, 50, 200]
REL_VOL_PERIOD: int = 30; MOMENTUM_PERIOD: int = 12; EMA_SLOPE_PERIOD: int = 5
BUY_CONFIDENCE_THRESHOLD = 0.80
ATR_FALLBACK_SL_MULTIPLIER: float = 1.5
ATR_FALLBACK_TP_MULTIPLIER: float = 2.2
MAX_TRADE_DURATION_CANDLES: int = 96
BACKTEST_BATCH_SIZE: int = 5

# --- متغيرات الاتصال ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
ml_models_cache: Dict[str, Any] = {}
exchange_info_map: Dict[str, Any] = {}

# ---------------------- دالة HTML للوحة التحكم ----------------------
def get_dashboard_html():
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم نتائج الاختبار الخلفي</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-main: #0D1117; --bg-card: #161B22; --border-color: #30363D;
            --text-primary: #E6EDF3; --text-secondary: #848D97;
            --accent-blue: #58A6FF; --accent-green: #3FB950; --accent-red: #F85149;
        }
        body { font-family: 'Tajawal', sans-serif; background-color: var(--bg-main); color: var(--text-primary); }
        .card { background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 0.5rem; }
        .table-container { max-height: 80vh; }
        table { border-collapse: separate; border-spacing: 0; }
        th, td { border-bottom: 1px solid var(--border-color); padding: 0.75rem 1rem; text-align: right; white-space: nowrap; }
        thead th { background-color: #10141a; position: sticky; top: 0; z-index: 10; cursor: pointer; }
        .positive { color: var(--accent-green); }
        .negative { color: var(--accent-red); }
        .loader {
            border: 4px solid var(--border-color);
            border-top: 4px solid var(--accent-blue);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body class="p-4 md:p-6">
    <div class="container mx-auto max-w-screen-3xl">
        <header class="mb-6 flex justify-between items-center">
            <h1 class="text-2xl md:text-3xl font-extrabold text-white">
                <span class="text-accent-blue">نتائج الاختبار الخلفي</span>
            </h1>
            <div class="flex items-center gap-4">
                <button id="load-results-btn" onclick="loadResults()" class="bg-accent-blue text-white font-bold py-2 px-4 rounded-lg hover:bg-blue-500 transition-colors">
                    تحميل النتائج
                </button>
                 <div id="db-status" class="flex items-center gap-2 text-sm">
                    <div id="db-status-light" class="w-2.5 h-2.5 rounded-full bg-gray-600 animate-pulse"></div>
                    <span class="text-text-secondary">DB</span>
                </div>
            </div>
        </header>

        <main id="main-content" class="card p-4">
            <div id="loader-container" class="hidden flex justify-center items-center py-16">
                <div class="loader"></div>
            </div>
            <div id="results-table-container" class="table-container overflow-x-auto">
                <table class="min-w-full text-sm">
                    <thead id="results-table-head"></thead>
                    <tbody id="results-table-body"></tbody>
                </table>
            </div>
             <div id="no-results" class="hidden text-center py-16 text-text-secondary">
                <p>لم يتم العثور على نتائج. قم بتشغيل الاختبار الخلفي أولاً ثم اضغط على "تحميل النتائج".</p>
            </div>
        </main>
    </div>

<script>
    let sortState = { column: 'id', order: 'desc' };

    async function apiFetch(url, options = {}) {
        try {
            const response = await fetch(url, options);
            if (!response.ok) {
                console.error(`API Error ${response.status}`);
                return { error: `HTTP Error ${response.status}` };
            }
            return await response.json();
        } catch (error) {
            console.error(`Fetch error for ${url}:`, error);
            return { error: "Network or fetch error" };
        }
    }

    function formatNumber(num, digits = 2) {
        if (num === null || num === undefined || isNaN(num)) return 'N/A';
        return num.toLocaleString('en-US', {
            minimumFractionDigits: digits,
            maximumFractionDigits: digits
        });
    }

    function renderTable(data) {
        const tableHead = document.getElementById('results-table-head');
        const tableBody = document.getElementById('results-table-body');
        const noResultsDiv = document.getElementById('no-results');
        const tableContainer = document.getElementById('results-table-container');

        if (!data || data.length === 0) {
            tableContainer.classList.add('hidden');
            noResultsDiv.classList.remove('hidden');
            return;
        }

        tableContainer.classList.remove('hidden');
        noResultsDiv.classList.add('hidden');

        // Create headers
        const headers = Object.keys(data[0]);
        tableHead.innerHTML = `<tr>${headers.map(h => `<th onclick="setSort('${h}')">${h.replace(/_/g, ' ')} ${sortState.column === h ? (sortState.order === 'asc' ? '▲' : '▼') : ''}</th>`).join('')}</tr>`;

        // Create rows
        tableBody.innerHTML = data.map(row => {
            let rowHtml = '<tr>';
            for (const key of headers) {
                let value = row[key];
                let cellClass = '';

                if (key === 'pnl_pct') {
                    const pnl = parseFloat(value);
                    if (!isNaN(pnl)) {
                        cellClass = pnl >= 0 ? 'positive' : 'negative';
                        value = `${pnl > 0 ? '+' : ''}${formatNumber(pnl, 2)}%`;
                    }
                } else if (key === 'trade_outcome') {
                    if (value === 'TP_HIT') cellClass = 'positive';
                    if (value === 'SL_HIT') cellClass = 'negative';
                } else if (typeof value === 'number') {
                    value = formatNumber(value, 4);
                } else if (key.includes('timestamp')) {
                    value = new Date(value).toLocaleString('ar-EG');
                }

                rowHtml += `<td class="${cellClass}">${value}</td>`;
            }
            rowHtml += '</tr>';
            return rowHtml;
        }).join('');
    }
    
    function setSort(column) {
        if (sortState.column === column) {
            sortState.order = sortState.order === 'asc' ? 'desc' : 'asc';
        } else {
            sortState.column = column;
            sortState.order = 'desc';
        }
        loadResults();
    }

    async function loadResults() {
        const loader = document.getElementById('loader-container');
        const tableContainer = document.getElementById('results-table-container');
        const noResultsDiv = document.getElementById('no-results');
        
        loader.classList.remove('hidden');
        tableContainer.classList.add('hidden');
        noResultsDiv.classList.add('hidden');
        
        const url = `/api/backtest_results?sort_by=${sortState.column}&order=${sortState.order}`;
        const response = await apiFetch(url);

        loader.classList.add('hidden');

        if (response.error) {
            alert('فشل تحميل النتائج: ' + response.error);
            noResultsDiv.classList.remove('hidden');
            return;
        }

        renderTable(response.data);
    }
    
    async function checkDbStatus() {
        const response = await apiFetch('/api/status');
        const light = document.getElementById('db-status-light');
        if (response && response.db_ok) {
            light.className = 'w-2.5 h-2.5 rounded-full bg-green-500';
        } else {
            light.className = 'w-2.5 h-2.5 rounded-full bg-red-500';
        }
    }

    window.onload = () => {
        checkDbStatus();
        loadResults();
        setInterval(checkDbStatus, 10000);
    };
</script>
</body>
</html>
    """

# ---------------------- دوال قاعدة البيانات ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    global conn
    logger.info("[DB] تهيئة الاتصال بقاعدة البيانات...")
    db_url_to_use = DB_URL
    if 'postgres' in db_url_to_use and 'sslmode' not in db_url_to_use:
        separator = '&' if '?' in db_url_to_use else '?'
        db_url_to_use += f"{separator}sslmode=require"
    
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(db_url_to_use, connect_timeout=15, cursor_factory=RealDictCursor)
            conn.autocommit = True
            logger.info("✅ [DB] تم الاتصال بنجاح بقاعدة البيانات.")
            create_backtest_results_table()
            return
        except Exception as e:
            logger.error(f"❌ [DB] خطأ في التهيئة (محاولة {attempt + 1}/{retries}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: logger.critical("❌ [DB] فشل الاتصال بقاعدة البيانات.")

def create_backtest_results_table():
    if not conn: return
    logger.info("[DB] التحقق من جدول نتائج الاختبار الخلفي...")
    try:
        with conn.cursor() as cur:
            # --- [FIX] --- حذف الجدول أولاً لضمان أن المخطط محدث دائمًا
            logger.warning("سيتم حذف جدول backtest_results الموجود لضمان مخطط جديد...")
            cur.execute("DROP TABLE IF EXISTS backtest_results;")
            
            cur.execute("""
                CREATE TABLE backtest_results (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    signal_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    target_price DOUBLE PRECISION NOT NULL,
                    stop_loss DOUBLE PRECISION NOT NULL,
                    ml_confidence DOUBLE PRECISION,
                    trade_outcome TEXT, -- 'TP_HIT', 'SL_HIT', 'TIMEOUT'
                    outcome_timestamp TIMESTAMP WITH TIME ZONE,
                    pnl_pct DOUBLE PRECISION,
                    max_drawdown_pct DOUBLE PRECISION,
                    max_profit_pct DOUBLE PRECISION,
                    trend_15m TEXT,
                    trend_1h TEXT,
                    trend_4h TEXT,
                    filter_adx DOUBLE PRECISION,
                    filter_rsi DOUBLE PRECISION,
                    filter_relative_volume DOUBLE PRECISION,
                    filter_roc_12 DOUBLE PRECISION,
                    filter_ema_slope_5 DOUBLE PRECISION,
                    filter_btc_correlation DOUBLE PRECISION,
                    filter_volatility_pct DOUBLE PRECISION
                );
            """)
            logger.info("✅ [DB] جدول 'backtest_results' جاهز.")
    except Exception as e:
        logger.error(f"❌ [DB] فشل في إنشاء جدول 'backtest_results': {e}")
        if conn: conn.rollback()

def check_db_connection() -> bool:
    global conn
    if conn is None or conn.closed != 0:
        logger.warning("[DB] الاتصال مغلق، محاولة إعادة الاتصال...")
        init_db()
    try:
        if conn and conn.closed == 0:
            with conn.cursor() as cur: cur.execute("SELECT 1;")
            return True
        return False
    except (psycopg2.OperationalError, psycopg2.InterfaceError):
        logger.error("❌ [DB] فقد الاتصال. إعادة الاتصال...")
        init_db()
        return conn is not None and conn.closed == 0
    return False

# ---------------------- دوال Binance والبيانات ----------------------
def get_exchange_info_map() -> None:
    global exchange_info_map
    if not client: return
    logger.info("ℹ️ [Exchange Info] جلب قواعد التداول...")
    try:
        info = client.get_exchange_info()
        exchange_info_map = {s['symbol']: s for s in info['symbols']}
        logger.info(f"✅ [Exchange Info] تم تحميل القواعد لـ {len(exchange_info_map)} عملة.")
    except Exception as e:
        logger.error(f"❌ [Exchange Info] لم يتمكن من جلب معلومات البورصة: {e}")

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
        logger.info(f"✅ [Validation] سيتم اختبار {len(validated)} عملة.")
        return validated
    except Exception as e:
        logger.error(f"❌ [Validation] خطأ أثناء التحقق من العملات: {e}", exc_info=True)
        return []

def fetch_historical_data(symbol: str, interval: str, start_str: str, end_str: str = None) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        klines = client.get_historical_klines(symbol, interval, start_str, end_str)
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna()
    except Exception as e:
        return None

# ---------------------- دوال حساب الميزات وتحديد الاتجاه ----------------------
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
    
    return df_calc

def determine_trend_for_timestamp(df: pd.DataFrame) -> str:
    if df is None or len(df) < EMA_PERIODS[-1]:
        return "غير واضح"

    last_candle = df.iloc[-1]
    close = last_candle['close']
    ema21 = last_candle['ema_21']
    ema50 = last_candle['ema_50']
    ema200 = last_candle['ema_200']

    score = 0
    if close > ema21: score += 1
    elif close < ema21: score -= 1
    if ema21 > ema50: score += 1
    elif ema21 < ema50: score -= 1
    if ema50 > ema200: score += 1
    elif ema50 < ema200: score -= 1

    if score >= 2: return "صاعد"
    if score <= -2: return "هابط"
    return "محايد"

def load_ml_model_bundle_from_folder(symbol: str) -> Optional[Dict[str, Any]]:
    global ml_models_cache
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_name in ml_models_cache:
        return ml_models_cache[model_name]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir_path = os.path.join(script_dir, MODEL_FOLDER)
    model_path = os.path.join(model_dir_path, f"{model_name}.pkl")
    
    if not os.path.exists(model_path):
        return None
    try:
        with open(model_path, 'rb') as f:
            model_bundle = pickle.load(f)
        if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
            ml_models_cache[model_name] = model_bundle
            return model_bundle
        return None
    except Exception as e:
        logger.error(f"❌ [ML Model] خطأ في تحميل النموذج لـ {symbol}: {e}", exc_info=True)
        return None

# ---------------------- محرك الاختبار الخلفي ----------------------
class BacktestTradingStrategy:
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
            df_featured = df_featured.join(df_4h_features[required_4h_cols], how='outer')
            df_featured.fillna(method='ffill', inplace=True)
            for col in self.feature_names:
                if col not in df_featured.columns: df_featured[col] = 0.0
            df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_featured.dropna(subset=self.feature_names)
        except Exception as e:
            logger.error(f"❌ [{self.symbol}] فشل في هندسة الميزات: {e}", exc_info=True)
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
            return {'prediction': int(prediction), 'confidence': confidence}
        except Exception:
            return None

def calculate_tp_sl(entry_price: float, last_atr: float) -> Optional[Dict[str, Any]]:
    if last_atr <= 0: return None
    tp = entry_price + (last_atr * ATR_FALLBACK_TP_MULTIPLIER)
    sl = entry_price - (last_atr * ATR_FALLBACK_SL_MULTIPLIER)
    return {'target_price': tp, 'stop_loss': sl}

def simulate_trade_outcome(entry_price: float, tp: float, sl: float, future_candles: pd.DataFrame) -> Dict:
    for index, row in future_candles.iterrows():
        if row['low'] <= sl:
            return {'outcome': 'SL_HIT', 'timestamp': index, 'pnl': ((sl / entry_price) - 1) * 100}
        if row['high'] >= tp:
            return {'outcome': 'TP_HIT', 'timestamp': index, 'pnl': ((tp / entry_price) - 1) * 100}
    
    last_candle = future_candles.iloc[-1]
    return {'outcome': 'TIMEOUT', 'timestamp': last_candle.name, 'pnl': ((last_candle['close'] / entry_price) - 1) * 100}

def run_backtest_for_symbol(symbol: str, start_date: str, end_date: str):
    logger.info(f"🚀 بدء الاختبار الخلفي للعملة: {symbol} من {start_date} إلى {end_date}")
    
    strategy = BacktestTradingStrategy(symbol)
    if not strategy.ml_model:
        logger.warning(f"⚠️ لا يوجد نموذج للعملة {symbol}. جاري التخطي.")
        return 0

    df_15m = fetch_historical_data(symbol, '15m', start_date, end_date)
    df_1h = fetch_historical_data(symbol, '1h', start_date, end_date)
    df_4h = fetch_historical_data(symbol, '4h', start_date, end_date)
    btc_df = fetch_historical_data(BTC_SYMBOL, '15m', start_date, end_date)

    if df_15m is None or len(df_15m) < 250:
        logger.warning(f"⚠️ بيانات غير كافية لـ {symbol}. جاري التخطي.")
        return 0
    if btc_df is not None: btc_df['btc_returns'] = btc_df['close'].pct_change()
        
    df_15m_features = calculate_features(df_15m.copy(), btc_df)
    df_1h_features = calculate_features(df_1h.copy(), None) if df_1h is not None else None
    df_4h_features = calculate_features(df_4h.copy(), None) if df_4h is not None else None

    results_to_insert = []
    
    for i in range(250, len(df_15m_features)):
        current_timestamp = df_15m_features.index[i]
        
        df_15m_point_in_time = df_15m.iloc[:i+1]
        df_4h_point_in_time = df_4h[df_4h.index <= current_timestamp] if df_4h is not None else None
        btc_point_in_time = btc_df[btc_df.index <= current_timestamp] if btc_df is not None else None
        
        features_for_model = strategy.get_features(df_15m_point_in_time, df_4h_point_in_time, btc_point_in_time)
        if features_for_model is None or features_for_model.empty:
            continue
            
        ml_signal = strategy.generate_buy_signal(features_for_model)
        
        if ml_signal and ml_signal['confidence'] >= BUY_CONFIDENCE_THRESHOLD:
            last_features = features_for_model.iloc[-1]
            entry_price = last_features['close']
            
            tp_sl_data = calculate_tp_sl(entry_price, last_features.get('atr', 0))
            if not tp_sl_data: continue

            trend_15m = determine_trend_for_timestamp(df_15m_features.iloc[:i+1])
            trend_1h = determine_trend_for_timestamp(df_1h_features[df_1h_features.index <= current_timestamp]) if df_1h_features is not None else "N/A"
            trend_4h = determine_trend_for_timestamp(df_4h_features[df_4h_features.index <= current_timestamp]) if df_4h_features is not None else "N/A"
            
            future_candles = df_15m.iloc[i+1 : i+1+MAX_TRADE_DURATION_CANDLES]
            if future_candles.empty: continue
            
            trade_sim = simulate_trade_outcome(entry_price, tp_sl_data['target_price'], tp_sl_data['stop_loss'], future_candles)
            
            trade_period_candles = df_15m.loc[current_timestamp:trade_sim['timestamp']]
            max_p = ((trade_period_candles['high'].max() / entry_price) - 1) * 100
            max_d = ((trade_period_candles['low'].min() / entry_price) - 1) * 100

            result_row = {
                "symbol": symbol,
                "signal_timestamp": current_timestamp,
                "entry_price": entry_price,
                "target_price": tp_sl_data['target_price'],
                "stop_loss": tp_sl_data['stop_loss'],
                "ml_confidence": ml_signal['confidence'],
                "trade_outcome": trade_sim['outcome'],
                "outcome_timestamp": trade_sim['timestamp'],
                "pnl_pct": trade_sim['pnl'],
                "max_drawdown_pct": max_d,
                "max_profit_pct": max_p,
                "trend_15m": trend_15m,
                "trend_1h": trend_1h,
                "trend_4h": trend_4h,
                "filter_adx": last_features.get('adx'),
                "filter_rsi": last_features.get('rsi'),
                "filter_relative_volume": last_features.get('relative_volume'),
                "filter_roc_12": last_features.get(f'roc_{MOMENTUM_PERIOD}'),
                "filter_ema_slope_5": last_features.get(f'ema_slope_{EMA_SLOPE_PERIOD}'),
                "filter_btc_correlation": last_features.get('btc_correlation'),
                "filter_volatility_pct": (last_features.get('atr', 0) / entry_price * 100) if entry_price > 0 else 0
            }
            
            # --- [FIX] تحويل أنواع بيانات NumPy إلى أنواع Python الأصلية ---
            # تقوم قاعدة البيانات برفض أنواع مثل numpy.float64 مباشرة.
            # نحولها إلى float و int و None قبل إرسالها.
            for key, value in result_row.items():
                if pd.isna(value):
                    result_row[key] = None
                elif isinstance(value, np.generic):
                    result_row[key] = value.item()
            
            results_to_insert.append(result_row)

    if results_to_insert:
        if not check_db_connection() or not conn:
            logger.error("فشل إدراج النتائج، لا يوجد اتصال بقاعدة البيانات.")
            return 0
        
        cols = results_to_insert[0].keys()
        query = sql.SQL("INSERT INTO backtest_results ({}) VALUES %s").format(
            sql.SQL(', ').join(map(sql.Identifier, cols))
        )
        
        values = [[row[col] for col in cols] for row in results_to_insert]
        
        try:
            with conn.cursor() as cur:
                execute_values(cur, query, values)
            logger.info(f"✅ [{symbol}] تم إدراج {len(results_to_insert)} نتيجة في قاعدة البيانات.")
            return len(results_to_insert)
        except Exception as e:
            logger.error(f"❌ [{symbol}] فشل في إدراج النتائج: {e}")
            if conn: conn.rollback()
            return 0
            
    return 0

def main_backtest_loop(start_date: str, end_date: str, symbols: List[str]):
    logger.info(f"====== بدء دورة الاختبار الخلفي الكاملة من {start_date} إلى {end_date} ======")
    total_signals_found = 0
    
    symbols_with_models = []
    logger.info("... التحقق من النماذج المتاحة ...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir_path = os.path.join(script_dir, MODEL_FOLDER)
    for symbol in symbols:
        model_path = os.path.join(model_dir_path, f"{BASE_ML_MODEL_NAME}_{symbol}.pkl")
        if os.path.exists(model_path):
            symbols_with_models.append(symbol)
    logger.info(f"وجد {len(symbols_with_models)} عملة مع نماذج متاحة للاختبار.")

    for i in range(0, len(symbols_with_models), BACKTEST_BATCH_SIZE):
        batch_symbols = symbols_with_models[i:i + BACKTEST_BATCH_SIZE]
        num_batches = (len(symbols_with_models) + BACKTEST_BATCH_SIZE - 1) // BACKTEST_BATCH_SIZE
        logger.info(f"--- بدء معالجة الدفعة {i//BACKTEST_BATCH_SIZE + 1}/{num_batches} ({len(batch_symbols)} عملة) ---")
        
        for symbol_in_batch in batch_symbols:
            try:
                signals_count = run_backtest_for_symbol(symbol_in_batch, start_date, end_date)
                total_signals_found += signals_count
            except Exception as e:
                logger.error(f"❌ حدث خطأ فادح أثناء اختبار {symbol_in_batch}: {e}", exc_info=True)
        
        logger.info(f"--- 🧹 اكتملت الدفعة. بدء تنظيف الذاكرة... ---")
        ml_models_cache.clear()
        collected = gc.collect()
        logger.info(f"--- ✅ تم تنظيف الذاكرة. تم جمع {collected} كائن. ---")
        time.sleep(2)

    logger.info(f"====== اكتملت دورة الاختبار الخلفي. إجمالي الإشارات التي تم العثور عليها وتخزينها: {total_signals_found} ======")

# ---------------------- واجهة برمجة تطبيقات Flask ----------------------
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template_string(get_dashboard_html())

@app.route('/api/status')
def get_status():
    return jsonify({"db_ok": check_db_connection()})

@app.route('/api/backtest_results')
def get_backtest_results():
    if not check_db_connection():
        return jsonify({"error": "DB connection failed"}), 500
    
    sort_by = request.args.get('sort_by', 'id')
    order = request.args.get('order', 'desc').upper()
    
    allowed_columns = [
        'id', 'symbol', 'signal_timestamp', 'entry_price', 'target_price', 'stop_loss',
        'ml_confidence', 'trade_outcome', 'outcome_timestamp', 'pnl_pct', 'max_drawdown_pct',
        'max_profit_pct', 'trend_15m', 'trend_1h', 'trend_4h', 'filter_adx', 'filter_rsi',
        'filter_relative_volume', 'filter_roc_12', 'filter_ema_slope_5',
        'filter_btc_correlation', 'filter_volatility_pct'
    ]
    if sort_by not in allowed_columns:
        sort_by = 'id'
    if order not in ['ASC', 'DESC']:
        order = 'DESC'

    try:
        with conn.cursor() as cur:
            query = sql.SQL("SELECT * FROM backtest_results ORDER BY {} {} NULLS LAST LIMIT 500").format(
                sql.Identifier(sort_by),
                sql.SQL(order)
            )
            cur.execute(query)
            results = cur.fetchall()
        
        for row in results:
            for key, value in row.items():
                if isinstance(value, datetime):
                    row[key] = value.isoformat()

        return jsonify({"data": results})
    except Exception as e:
        logger.error(f"❌ [API Results] خطأ: {e}", exc_info=True)
        if conn: conn.rollback()
        return jsonify({"error": "Internal error fetching results"}), 500

def run_flask():
    port = int(os.environ.get('PORT', 10001))
    host = "0.0.0.0"
    logger.info(f"✅ إعداد لوحة التحكم على {host}:{port}")
    app.run(host=host, port=port)

# ---------------------- نقطة انطلاق البرنامج ----------------------
if __name__ == "__main__":
    logger.info("🚀 إطلاق محرك الاختبار الخلفي ولوحة التحكم 🚀")
    
    client = Client(API_KEY, API_SECRET)
    init_db()
    get_exchange_info_map()
    
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # --- منطقة التحكم في الاختبار ---
    backtest_start_date = "14 July, 2025"
    backtest_end_date = "17 July, 2025"
    symbols_to_test = get_validated_symbols()
    
    logger.info(f"===> سيتم بدء الاختبار الخلفي تلقائيًا للفترة من {backtest_start_date} إلى {backtest_end_date} <===")
    backtest_thread = Thread(target=main_backtest_loop, args=(backtest_start_date, backtest_end_date, symbols_to_test))
    backtest_thread.start()
    
    try:
        backtest_thread.join()
        logger.info("✅✅✅ اكتمل الاختبار الخلفي. يمكنك الآن تحليل النتائج من لوحة التحكم. ✅✅✅")
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("👋 إيقاف تشغيل البرنامج.")
        os._exit(0)
