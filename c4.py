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

# --- تجاهل التحذيرات غير الهامة ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ---------------------- إعداد نظام التسجيل (Logging) - V26.0 (Crazy Reversal) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v26_crazy_reversal.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV26')

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

# ---------------------- إعدادات الفلاتر الديناميكية ----------------------
FILTER_PROFILES: Dict[str, Dict[str, Any]] = {
    "STRONG_UPTREND": {
        "description": "اتجاه صاعد قوي",
        "strategy": "MOMENTUM",
        "filters": {
            "adx": 22.0, "rel_vol": 1.2, "rsi_range": (50, 95), "roc": 0.3, 
            "accel": 0.1, "slope": 0.01, "min_rrr": 1.5, "min_volatility_pct": 0.4, 
            "min_btc_correlation": -0.1
        }
    },
    "UPTREND": {
        "description": "اتجاه صاعد",
        "strategy": "MOMENTUM",
        "filters": {
            "adx": 20.0, "rel_vol": 0.5, "rsi_range": (45, 90), "roc": 0.2, 
            "accel": 0.05, "slope": 0.005, "min_rrr": 1.8, "min_volatility_pct": 0.35, 
            "min_btc_correlation": 0.0
        }
    },
    "RANGING": {
        "description": "اتجاه عرضي",
        "strategy": "MOMENTUM",
        "filters": {
            "adx": 18.0, "rel_vol": 0.3, "rsi_range": (40, 70), "roc": 0.1, 
            "accel": 0.0, "slope": 0.0, "min_rrr": 2.0, "min_volatility_pct": 0.3, 
            "min_btc_correlation": -0.2
        }
    },
    "DOWNTREND": {
        "description": "اتجاه هابط (مراقبة الانعكاس المجنون)",
        "strategy": "CRAZY_REVERSAL", # <--- الاستراتيجية الجديدة
        "filters": {
            "min_rrr": 2.5, # RRR أعلى للانعكاسات
            "min_volatility_pct": 0.5,
            "min_btc_correlation": -0.5,
            "reversal_rsi_divergence_strength": 1.5, # الحد الأدنى لقوة الانحراف
            "reversal_volume_spike_multiplier": 2.0 # مضاعف فوليوم الذروة
        }
    },
    "WEEKEND": {
        "description": "سيولة منخفضة (عطلة نهاية الأسبوع)",
        "strategy": "MOMENTUM",
        "filters": {
            "adx": 17.0, "rel_vol": 0.8, "rsi_range": (30, 70), "roc": 0.1, 
            "accel": -0.05, "slope": 0.0, "min_rrr": 1.5, "min_volatility_pct": 0.25, 
            "min_btc_correlation": -0.4
        }
    }
}

SESSION_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "HIGH_LIQUIDITY": { "adx_mult": 1.1, "rel_vol_mult": 1.1, "rrr_mult": 0.95 },
    "NORMAL_LIQUIDITY": { "adx_mult": 1.0, "rel_vol_mult": 1.0, "rrr_mult": 1.0 },
    "LOW_LIQUIDITY": { "adx_mult": 0.9, "rel_vol_mult": 0.9, "rrr_mult": 1.1 }
}

# ---------------------- الثوابت والمتغيرات العامة ----------------------
is_trading_enabled: bool = False
trading_status_lock = Lock()
RISK_PER_TRADE_PERCENT: float = 1.0
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V8_With_Momentum'
MODEL_FOLDER: str = 'V8'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 45 # زيادة لجودة حساب POC
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices_v8"
DIRECT_API_CHECK_INTERVAL: int = 10
TRADING_FEE_PERCENT: float = 0.1
STATS_TRADE_SIZE_USDT: float = 10.0
BTC_SYMBOL: str = 'BTCUSDT'
SYMBOL_PROCESSING_BATCH_SIZE: int = 50
ADX_PERIOD: int = 14; RSI_PERIOD: int = 14; ATR_PERIOD: int = 14
EMA_FAST_PERIOD: int = 50; EMA_SLOW_PERIOD: int = 200
REL_VOL_PERIOD: int = 30; MOMENTUM_PERIOD: int = 12; EMA_SLOPE_PERIOD: int = 5
MAX_OPEN_TRADES: int = 4
BUY_CONFIDENCE_THRESHOLD = 0.85 # زيادة الثقة المطلوبة للانعكاس
MIN_CONFIDENCE_INCREASE_FOR_UPDATE = 0.05
ATR_FALLBACK_SL_MULTIPLIER: float = 1.8 # زيادة وقف الخسارة للانعكاس
ATR_FALLBACK_TP_MULTIPLIER: float = 3.0 # زيادة الهدف للانعكاس
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_ACTIVATION_PROFIT_PERCENT: float = 1.0
TRAILING_DISTANCE_PERCENT: float = 0.8
LAST_PEAK_UPDATE_TIME: Dict[int, float] = {}
PEAK_UPDATE_COOLDOWN: int = 60
USE_PEAK_FILTER: bool = True
PEAK_CHECK_PERIOD: int = 50
PULLBACK_THRESHOLD_PCT: float = 0.988
BREAKOUT_ALLOWANCE_PCT: float = 1.003
DYNAMIC_FILTER_ANALYSIS_INTERVAL: int = 900
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
dynamic_filter_profile_cache: Dict[str, Any] = {}
last_dynamic_filter_analysis_time: float = 0
dynamic_filter_lock = Lock()


# ---------------------- دالة HTML للوحة التحكم (بدون تغيير) ----------------------
def get_dashboard_html():
    # ... (الكود الخاص بواجهة التحكم لم يتغير)
    # For brevity, the HTML code is omitted here but is the same as in the original file.
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم التداول V26.0 - استراتيجية الانعكاس المجنون</title>
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
    <!-- The rest of the HTML body is the same as the original file -->
</body>
</html>
    """

# ---------------------- دوال قاعدة البيانات (بدون تغيير) ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    # ... (الكود الخاص بقاعدة البيانات لم يتغير)
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
                        quantity DOUBLE PRECISION, order_id TEXT
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

# ... (بقية دوال قاعدة البيانات والاتصال لم تتغير)

# ---------------------- دوال Binance والبيانات (بدون تغيير) ----------------------
def get_exchange_info_map() -> None:
    # ... (الكود لم يتغير)
    global exchange_info_map
    if not client: return
    logger.info("ℹ️ [Exchange Info] Fetching exchange trading rules...")
    try:
        info = client.get_exchange_info()
        exchange_info_map = {s['symbol']: s for s in info['symbols']}
        logger.info(f"✅ [Exchange Info] Loaded rules for {len(exchange_info_map)} symbols.")
    except Exception as e:
        logger.error(f"❌ [Exchange Info] Could not fetch exchange info: {e}")


def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    # ... (الكود لم يتغير)
    if not client: return None
    try:
        limit = int((days * 24 * 60) / int(re.sub('[a-zA-Z]', '', interval)))
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
        logger.error(f"❌ [Data] Error fetching historical data for {symbol}: {e}")
        return None

# ---------------------- دوال حساب الميزات وتحديد الاتجاه (مع إضافة ميزات جديدة) ----------------------
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
    
    # --- ميزات جديدة للاستراتيجية المجنونة ---
    df_calc['volume_ma_30'] = df_calc['volume'].rolling(window=30).mean()
    df_calc['volume_spike_factor'] = df_calc['volume'] / df_calc['volume_ma_30']
    
    return df_calc.astype('float32', errors='ignore')

# ---------------------- دوال الاستراتيجية والتداول الحقيقي (مع التعديلات الجذرية) ----------------------

def get_point_of_control(df: pd.DataFrame, num_bins: int = 50) -> Optional[float]:
    """
    [جديد] تحسب نقطة التحكم (POC) من بيانات الشموع.
    POC هي مستوى السعر الذي حدث عنده أعلى حجم تداول.
    """
    if df.empty:
        return None
    try:
        price_range = df['high'].max() - df['low'].min()
        if price_range == 0:
            return df['close'].iloc[-1]
            
        bins = np.linspace(df['low'].min(), df['high'].max(), num_bins)
        
        # توزيع حجم التداول على مستويات الأسعار
        volume_per_bin = np.zeros(num_bins)
        for _, row in df.iterrows():
            price_indices = np.digitize([row['low'], row['high']], bins)
            start_idx, end_idx = min(price_indices) - 1, max(price_indices) -1
            
            # توزيع حجم الشمعة بالتساوي على البينات التي مرت بها
            num_bins_in_candle = end_idx - start_idx + 1
            volume_per_bin[start_idx:end_idx+1] += row['volume'] / num_bins_in_candle

        # العثور على البين ذو الحجم الأعلى
        max_volume_idx = np.argmax(volume_per_bin)
        poc = (bins[max_volume_idx] + bins[max_volume_idx - 1]) / 2 if max_volume_idx > 0 else bins[0]
        return float(poc)
    except Exception as e:
        logger.error(f"Error calculating Point of Control: {e}")
        return None


def find_bullish_reversal_signal(df_15m: pd.DataFrame, df_4h: pd.DataFrame, filters: Dict) -> Optional[Dict[str, Any]]:
    """
    [مُعدلة بالكامل] هذه هي الاستراتيجية المجنونة الجديدة.
    تبحث عن قاع الذعر والانحراف الإيجابي عند مناطق السيولة.
    """
    try:
        if len(df_15m) < 50: return None # نحتاج بيانات كافية للبحث

        df = df_15m.copy()
        df.name = df_15m.name if hasattr(df_15m, 'name') else 'Unknown'
        
        # --- الركيزة 1: تحديد منطقة السيولة (POC) ---
        poc_4h = get_point_of_control(df_4h)
        if poc_4h is None:
            log_rejection(df.name, "Reversal Signal Rejected", {"reason": "Failed to calculate 4h POC"})
            return None

        # --- الركيزة 2: البحث عن ذروة البيع (Climactic Volume) ---
        # نبحث في آخر 10 شمعات عن شمعة الذروة
        search_window = df.iloc[-10:]
        
        # حساب متوسط الفوليوم المتحرك
        df['volume_ma_30'] = df['volume'].rolling(window=30, min_periods=10).mean()
        
        # البحث عن شمعة حمراء بفوليوم ضخم قريبة من الـ POC
        climactic_candle = None
        climactic_candle_idx = -1
        
        for i in range(len(search_window) - 1, -1, -1):
            candle = search_window.iloc[i]
            is_near_poc = abs(candle['low'] - poc_4h) / poc_4h < 0.02 # قريب بنسبة 2% من الـ POC
            is_red_candle = candle['close'] < candle['open']
            volume_spike_multiplier = filters.get('reversal_volume_spike_multiplier', 2.0)
            is_volume_spike = candle['volume'] > df.loc[candle.name, 'volume_ma_30'] * volume_spike_multiplier
            
            if is_near_poc and is_red_candle and is_volume_spike:
                climactic_candle = candle
                # الحصول على الاندكس من الداتا فريم الأصلي
                climactic_candle_idx = df.index.get_loc(climactic_candle.name)
                break
        
        if climactic_candle is None:
            # لا توجد شمعة ذروة بيع، لا توجد إشارة
            return None
        
        # --- الركيزة 3: البحث عن انحراف إيجابي (Bullish Divergence) بعد شمعة الذروة ---
        # يجب أن يحدث القاع الجديد بعد شمعة الذروة
        data_after_climactic = df.iloc[climactic_candle_idx + 1:]
        if len(data_after_climactic) < 3:
            # لا توجد بيانات كافية بعد شمعة الذروة للبحث عن انحراف
            return None
            
        # إيجاد القاع السعري الجديد الأدنى
        new_low_price = data_after_climactic['low'].min()
        if new_low_price >= climactic_candle['low']:
            # السعر لم يكسر قاع شمعة الذروة، لا يوجد انحراف بعد
            return None
        
        # الحصول على بيانات القيعان
        climactic_low_rsi = df.iloc[climactic_candle_idx]['rsi']
        new_low_candle_idx = data_after_climactic['low'].idxmin()
        new_low_rsi = df.loc[new_low_candle_idx]['rsi']

        # التحقق من شروط الانحراف
        price_makes_lower_low = new_low_price < climactic_candle['low']
        rsi_makes_higher_low = new_low_rsi > climactic_low_rsi

        if price_makes_lower_low and rsi_makes_higher_low:
            # تم العثور على انحراف إيجابي!
            divergence_strength = (new_low_rsi - climactic_low_rsi)
            min_strength = filters.get('reversal_rsi_divergence_strength', 1.5)
            
            if divergence_strength < min_strength:
                log_rejection(df.name, "Reversal Rejected", {"reason": f"Divergence too weak ({divergence_strength:.2f})"})
                return None

            logger.info(f"✅ [CRAZY REVERSAL] Bullish signal detected for {df.name}.")
            return {
                "signal_type": "CRAZY_REVERSAL",
                "reason": "POC + Volume Spike + Bullish Divergence",
                "details": {
                    "poc_4h": poc_4h,
                    "climactic_volume": climactic_candle['volume'],
                    "divergence_strength": divergence_strength
                }
            }

        return None
    except Exception as e:
        logger.error(f"❌ [{df_15m.name if hasattr(df_15m, 'name') else 'Unknown'}] Error in find_bullish_reversal_signal: {e}", exc_info=True)
        return None


def passes_filters(symbol: str, last_features: pd.Series, profile: Dict[str, Any], entry_price: float, tp_sl_data: Dict, df_15m: pd.DataFrame) -> bool:
    # ... (الكود هنا لم يتغير بشكل كبير، فقط يتكيف مع الاستراتيجية الجديدة)
    filters = profile.get("filters", {})
    if not filters:
        log_rejection(symbol, "Filters Not Loaded", {"profile": profile.get('name')})
        return False

    volatility = (last_features.get('atr', 0) / entry_price * 100) if entry_price > 0 else 0
    if volatility < filters['min_volatility_pct']:
        log_rejection(symbol, "Low Volatility", {"volatility": f"{volatility:.2f}%", "min": f"{filters['min_volatility_pct']:.2f}%"})
        return False

    correlation = last_features.get('btc_correlation', 0)
    if correlation < filters['min_btc_correlation']:
        log_rejection(symbol, "BTC Correlation", {"corr": f"{correlation:.2f}", "min": f"{filters['min_btc_correlation']}"})
        return False

    risk, reward = entry_price - float(tp_sl_data['stop_loss']), float(tp_sl_data['target_price']) - entry_price
    if risk <= 0 or reward <= 0 or (reward / risk) < filters['min_rrr']:
        log_rejection(symbol, "RRR Filter", {"rrr": f"{(reward/risk):.2f}" if risk > 0 else "N/A", "min": filters['min_rrr']})
        return False

    # الفلاتر الخاصة باستراتيجية الزخم فقط
    if profile.get("strategy") == "MOMENTUM":
        adx, rel_vol, rsi = last_features.get('adx', 0), last_features.get('relative_volume', 0), last_features.get('rsi', 0)
        rsi_min, rsi_max = filters['rsi_range']
        if not (adx >= filters['adx'] and rel_vol >= filters['rel_vol'] and rsi_min <= rsi < rsi_max):
            log_rejection(symbol, "Speed Filter", {"ADX": f"{adx:.2f}", "Volume": f"{rel_vol:.2f}", "RSI": f"{rsi:.2f}"})
            return False
        # ... (بقية فلاتر الزخم)
    
    # لا توجد فلاتر إضافية خاصة بالانعكاس هنا لأن الشروط مدمجة في دالة الإشارة نفسها
    return True

# ---------------------- حلقة العمل الرئيسية (مع تعديل بسيط لدعم الاستراتيجية الجديدة) ----------------------
def main_loop():
    logger.info("[Main Loop] Waiting for initialization...")
    time.sleep(15)
    if not validated_symbols_to_scan:
        log_and_notify("critical", "No validated symbols to scan. Bot will not start.", "SYSTEM")
        return
    log_and_notify("info", f"✅ Starting main scan loop for {len(validated_symbols_to_scan)} symbols.", "SYSTEM")

    while True:
        try:
            logger.info("🌀 Starting new main cycle...")
            ml_models_cache.clear(); gc.collect()

            determine_market_state()
            analyze_market_and_create_dynamic_profile()
            
            filter_profile = get_current_filter_profile()
            active_strategy_type = filter_profile.get("strategy")
            
            if not active_strategy_type or active_strategy_type == "DISABLED":
                logger.warning(f"🔴 Trading is disabled by profile: '{filter_profile.get('name')}'. Skipping cycle.")
                time.sleep(300)
                continue

            btc_data = get_btc_data_for_bot()
            
            # ... (الكود الخاص بتحميل النماذج لم يتغير)
            
            logger.info(f"✅ Active Strategy: {active_strategy_type}")
            
            for symbol in random.sample(validated_symbols_to_scan, len(validated_symbols_to_scan)):
                # ... (الكود الخاص بالتحقق من الصفقات المفتوحة لم يتغير)
                
                df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                if df_15m is None or df_15m.empty: continue
                df_15m.name = symbol
                
                df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                if df_4h is None or df_4h.empty: continue

                # حساب الميزات الأساسية أولاً
                df_15m = calculate_features(df_15m, btc_data)

                technical_signal = None
                if active_strategy_type == "CRAZY_REVERSAL":
                    technical_signal = find_bullish_reversal_signal(df_15m, df_4h, filter_profile.get("filters", {}))
                elif active_strategy_type == "MOMENTUM":
                    technical_signal = {"signal_type": "MOMENTUM"} # إشارة وهمية للزخم للمتابعة

                if not technical_signal:
                    continue
                
                # --- دمج مع تعلم الآلة ---
                strategy = TradingStrategy(symbol) # تحميل النموذج
                if not all([strategy.ml_model, strategy.scaler, strategy.feature_names]): continue
                
                df_features_for_ml = strategy.get_features(df_15m, df_4h, btc_data)
                if df_features_for_ml is None or df_features_for_ml.empty: continue
                
                ml_signal = strategy.generate_buy_signal(df_features_for_ml)
                
                if not ml_signal or ml_signal['confidence'] < BUY_CONFIDENCE_THRESHOLD:
                    if technical_signal['signal_type'] == "CRAZY_REVERSAL":
                        log_rejection(symbol, "Reversal Signal Rejected by ML Model", {"ML_confidence": ml_signal.get('confidence') if ml_signal else 'N/A'})
                    continue
                
                # ... (بقية الكود الخاص بفتح الصفقة وإرسال الإشعارات لم يتغير)
                # This part includes fetching entry price, calculating TP/SL, checking filters,
                # calculating position size, placing order, and saving to DB.
                # It remains functionally the same as the original script.
                
        except (KeyboardInterrupt, SystemExit): 
            log_and_notify("info", "Bot is shutting down by user request.", "SYSTEM"); break
        except Exception as main_err: 
            log_and_notify("error", f"Critical error in main loop: {main_err}", "SYSTEM"); time.sleep(120)


# ... (بقية دوال Flask و التشغيل الرئيسي لم تتغير)
# The rest of the script (Flask API, initialization, etc.) is omitted for brevity
# but should be included from the original file for the bot to be fully functional.

if __name__ == "__main__":
    logger.info("🚀 LAUNCHING TRADING BOT & DASHBOARD (V26.0 - Crazy Reversal) 🚀")
    # The initialization and startup sequence remains the same.
    # initialize_bot_services()
    # run_flask()

