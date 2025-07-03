# -*- coding: utf-8 -*-
# ==============================================================================
# === Crypto Trading Bot Orchestrator ==========================================
# ==============================================================================
# هذا السكريبت يدمج وينظم عمل ثلاثة مكونات:
# 1. ماسح الدعم والمقاومة (SR Scanner - من c4r.py)
# 2. حاسب مؤشر إيشيموكو (Ichimoku Calculator - من c4i.py)
# 3. بوت التداول الرئيسي (Main Trading Bot - من c4.py)
#
# === V3.0 Update Notes ===
# - حل مشكلة "Port scan timeout reached" على منصات الاستضافة.
# - إعادة هيكلة عملية البدء لتشغيل خادم الويب (Flask) فوراً.
# - نقل التحليل الأولي الطويل والخدمات الخلفية لتعمل في خيط منفصل.
# ==============================================================================

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
import gc
import threading
import http.server
import socketserver
import datetime as dt

from urllib.parse import urlparse
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor, execute_values
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
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# --- تجاهل التحذيرات غير الهامة ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# ------------------------------------------------------------------------------
# --- ⚙️ 1. الإعدادات المركزية والتسجيل (Central Configuration & Logging) ⚙️ ---
# ------------------------------------------------------------------------------

# --- إعداد نظام التسجيل (Logging) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('orchestrator.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoOrchestrator')

# --- تحميل متغيرات البيئة ---
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

# --- ثوابت بوت التداول (Trading Bot Constants) ---
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V7_With_Ichimoku'
MODEL_FOLDER: str = 'V7'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 30
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices"
MODEL_BATCH_SIZE: int = 5
MAX_OPEN_TRADES: int = 5
TRADE_AMOUNT_USDT: float = 10.0
USE_DYNAMIC_SL_TP = True
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.0
USE_BTC_TREND_FILTER = True
BTC_SYMBOL = 'BTCUSDT'
BTC_TREND_TIMEFRAME = '4h'
BTC_TREND_EMA_PERIOD = 10
MIN_PROFIT_PERCENTAGE_FILTER: float = 1.0
MODEL_CONFIDENCE_THRESHOLD = 0.70

# --- ثوابت ماسح الدعم والمقاومة (SR Scanner Constants) ---
SR_RUN_INTERVAL_MINUTES = 60
SR_MAX_WORKERS = 5 
SR_API_RETRY_ATTEMPTS = 3
SR_API_RETRY_DELAY = 5
SR_DATA_FETCH_DAYS_1H = 30
SR_DATA_FETCH_DAYS_15M = 7
SR_DATA_FETCH_DAYS_5M = 3
SR_ATR_PROMINENCE_MULTIPLIER_1H = 0.8
SR_ATR_PROMINENCE_MULTIPLIER_15M = 0.6
SR_ATR_PROMINENCE_MULTIPLIER_5M = 0.5
SR_ATR_PERIOD = 14
SR_WIDTH_1H = 8
SR_WIDTH_15M = 5
SR_WIDTH_5M = 3
SR_VOLUME_CONFIRMATION_ENABLED = True
SR_VOLUME_AVG_PERIOD = 20
SR_VOLUME_SPIKE_FACTOR = 1.6
SR_CLUSTER_EPS_PERCENT = 0.0015
SR_CONFLUENCE_ZONE_PERCENT = 0.002
SR_VOLUME_PROFILE_BINS = 100

# --- ثوابت حاسب إيشيموكو (Ichimoku Calculator Constants) ---
ICHIMOKU_RUN_INTERVAL_HOURS: int = 4
ICHIMOKU_TIMEFRAME: str = '15m'
ICHIMOKU_DATA_LOOKBACK_DAYS: int = 30
ICHIMOKU_TENKAN_PERIOD: int = 9
ICHIMOKU_KIJUN_PERIOD: int = 26
ICHIMOKU_SENKOU_B_PERIOD: int = 52
ICHIMOKU_CHIKOU_SHIFT: int = -26
ICHIMOKU_SENKOU_SHIFT: int = 26

# --- المتغيرات العامة المشتركة والأقفال (Shared Global Variables & Locks) ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
validated_symbols_to_scan: List[str] = []

# متغيرات خاصة بالبوت
ml_models_cache: Dict[str, Any] = {}
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
notifications_cache = deque(maxlen=50)
notifications_lock = Lock()
signals_pending_closure: Set[int] = set()
closure_lock = Lock()


# ------------------------------------------------------------------------------
# --- 🛠️ 2. الدوال المشتركة والخدمات الأساسية (Common Utilities & Core Services) 🛠️ ---
# ------------------------------------------------------------------------------

def init_db(retries: int = 5, delay: int = 5) -> None:
    """Initializes and returns a database connection."""
    global conn
    logger.info("[DB] بدء تهيئة الاتصال بقاعدة البيانات...")
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False # Important for main bot
            logger.info("✅ [DB] تم الاتصال بقاعدة البيانات بنجاح.")
            create_tables_if_not_exist()
            return
        except Exception as e:
            logger.error(f"❌ [DB] خطأ في الاتصال (المحاولة {attempt + 1}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else:
                logger.critical("❌ [DB] فشل الاتصال بعد عدة محاولات.")
                exit(1)

def create_tables_if_not_exist():
    """Creates all necessary tables for all services."""
    if not conn:
        logger.error("[DB] لا يمكن إنشاء الجداول بدون اتصال.")
        return
    try:
        with conn.cursor() as cur:
            # Table for Trading Bot
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, entry_price DOUBLE PRECISION NOT NULL,
                    target_price DOUBLE PRECISION NOT NULL, stop_loss DOUBLE PRECISION NOT NULL,
                    status TEXT DEFAULT 'open', closing_price DOUBLE PRECISION, closed_at TIMESTAMP,
                    profit_percentage DOUBLE PRECISION, strategy_name TEXT, signal_details JSONB );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS notifications ( id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    type TEXT NOT NULL, message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE );
            """)
            # Table for SR Scanner
            cur.execute("""
                CREATE TABLE IF NOT EXISTS support_resistance_levels (
                    id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, level_price DOUBLE PRECISION NOT NULL,
                    level_type TEXT NOT NULL, timeframe TEXT NOT NULL, strength NUMERIC NOT NULL,
                    score NUMERIC DEFAULT 0, last_tested_at TIMESTAMP WITH TIME ZONE, details TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    CONSTRAINT unique_level UNIQUE (symbol, level_price, timeframe, level_type, details)
                );
            """)
            # Table for Ichimoku Calculator
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ichimoku_features (
                    id SERIAL PRIMARY KEY, symbol VARCHAR(20) NOT NULL, timestamp TIMESTAMPTZ NOT NULL,
                    timeframe VARCHAR(10) NOT NULL, tenkan_sen FLOAT, kijun_sen FLOAT,
                    senkou_span_a FLOAT, senkou_span_b FLOAT, chikou_span FLOAT,
                    UNIQUE (symbol, timestamp, timeframe)
                );
                CREATE INDEX IF NOT EXISTS idx_ichimoku_symbol_timestamp ON ichimoku_features (symbol, timestamp DESC);
            """)
        conn.commit()
        logger.info("✅ [DB] تم فحص/إنشاء جميع الجداول اللازمة بنجاح.")
    except Exception as e:
        logger.error(f"❌ [DB] خطأ في إنشاء الجداول: {e}")
        if conn: conn.rollback()

def check_db_connection() -> bool:
    """Checks the database connection and re-initializes if necessary."""
    global conn
    if conn is None or conn.closed != 0:
        logger.warning("[DB] الاتصال مغلق، محاولة إعادة الاتصال...")
        init_db()
    try:
        if conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
            return True
        return False
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [DB] فقدان الاتصال: {e}. محاولة إعادة الاتصال...")
        try:
            init_db()
            return conn is not None and conn.closed == 0
        except Exception as retry_e:
            logger.error(f"❌ [DB] فشل إعادة الاتصال: {retry_e}")
            return False
    return False


def init_redis() -> None:
    """Initializes the Redis client."""
    global redis_client
    logger.info("[Redis] بدء تهيئة الاتصال...")
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] تم الاتصال بنجاح بخادم Redis.")
    except redis.exceptions.ConnectionError as e:
        logger.critical(f"❌ [Redis] فشل الاتصال بـ Redis. الخطأ: {e}")
        exit(1)

def init_binance_client() -> None:
    """Initializes the Binance client."""
    global client
    logger.info("[Binance] بدء تهيئة الاتصال...")
    try:
        client = Client(API_KEY, API_SECRET)
        client.ping()
        logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
    except Exception as e:
        logger.critical(f"❌ [Binance] فشل الاتصال بواجهة برمجة التطبيقات: {e}")
        exit(1)

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """Reads and validates symbols against Binance. A shared function."""
    global validated_symbols_to_scan
    if validated_symbols_to_scan:
        return validated_symbols_to_scan
        
    logger.info(f"ℹ️ [Symbols] قراءة الرموز من '{filename}' والتحقق منها...")
    if not client:
        logger.error("❌ [Symbols] كائن Binance client غير مهيأ.")
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
        logger.info(f"✅ [Symbols] سيتم مراقبة {len(validated)} عملة معتمدة.")
        validated_symbols_to_scan = validated
        return validated
    except Exception as e:
        logger.error(f"❌ [Symbols] حدث خطأ أثناء التحقق من الرموز: {e}", exc_info=True)
        return []

def fetch_historical_data(symbol: str, interval: str, days: int, retries: int = 3, delay: int = 5) -> Optional[pd.DataFrame]:
    """
    Fetches historical kline data from Binance with retries and a built-in intelligent delay
    to respect API rate limits.
    """
    if not client: return None
    for attempt in range(retries):
        try:
            start_dt = datetime.now(timezone.utc) - timedelta(days=days)
            start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # The actual API call
            klines = client.get_historical_klines(symbol, interval, start_str)
            
            # ✨ NEW: Add a small delay AFTER every successful API call to spread out requests
            time.sleep(0.2) 
            
            if not klines: return None
            
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            numeric_cols = {'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'float32'}
            df = df.astype(numeric_cols)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            return df.dropna()
            
        except BinanceAPIException as e:
            logger.warning(f"⚠️ [API Binance] خطأ في جلب بيانات {symbol}: {e} (المحاولة {attempt + 1})")
            # ✨ NEW: If it's a rate limit error, wait for a longer, specific period
            if e.code == -1003:
                logger.warning(f"🕒 [Rate Limit] تم الوصول إلى حد الطلبات. الانتظار لمدة 60 ثانية...")
                time.sleep(60)
            elif attempt < retries - 1:
                time.sleep(delay)
                
        except Exception as e:
            logger.error(f"❌ [Data] خطأ أثناء جلب البيانات لـ {symbol}: {e} (المحاولة {attempt + 1})")
            if attempt < retries - 1:
                time.sleep(delay)
    return None


# ------------------------------------------------------------------------------
# --- 📈 3. دوال ماسح الدعم والمقاومة (SR Scanner Functions - from c4r.py) 📈 ---
# ------------------------------------------------------------------------------

def sr_calculate_level_score(level: Dict) -> int:
    score = 0
    score += float(level.get('strength', 1)) * 10
    last_tested = level.get('last_tested_at')
    if last_tested:
        if isinstance(last_tested, dt.datetime) and last_tested.tzinfo is None:
             last_tested = last_tested.replace(tzinfo=dt.timezone.utc)
        days_since_tested = (dt.datetime.now(dt.timezone.utc) - last_tested).days
        if days_since_tested < 2: score += 30
        elif days_since_tested < 7: score += 15
        elif days_since_tested < 30: score += 5
    if level.get('level_type') == 'confluence':
        num_timeframes = len(level.get('timeframe', '').split(','))
        num_details = len(level.get('details', '').split(','))
        score += (num_timeframes + num_details) * 20
        if 'poc' in level.get('details', ''): score += 25
    if level.get('level_type') == 'poc':
        score += 15
    if 'fib' in level.get('level_type', ''):
        score += 5
        if 'Golden Level' in level.get('details', ''):
            score += 20
    return int(score)

def sr_calculate_atr(df: pd.DataFrame, period: int) -> float:
    if df.empty or len(df) < period: return 0
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr.iloc[-1] if not atr.empty else 0

def sr_find_price_action_levels(df: pd.DataFrame, atr_value: float, prominence_multiplier: float, width: int, cluster_eps_percent: float) -> List[Dict]:
    lows, highs = df['low'].to_numpy(), df['high'].to_numpy()
    dynamic_prominence = atr_value * prominence_multiplier
    if dynamic_prominence == 0: dynamic_prominence = highs.mean() * 0.01 
    low_peaks_indices, _ = find_peaks(-lows, prominence=dynamic_prominence, width=width)
    high_peaks_indices, _ = find_peaks(highs, prominence=dynamic_prominence, width=width)
    if SR_VOLUME_CONFIRMATION_ENABLED and not df.empty:
        df['volume_avg'] = df['volume'].rolling(window=SR_VOLUME_AVG_PERIOD, min_periods=1).mean()
        confirmed_low_indices = [idx for idx in low_peaks_indices if df['volume'].iloc[idx] >= df['volume_avg'].iloc[idx] * SR_VOLUME_SPIKE_FACTOR]
        confirmed_high_indices = [idx for idx in high_peaks_indices if df['volume'].iloc[idx] >= df['volume_avg'].iloc[idx] * SR_VOLUME_SPIKE_FACTOR]
        low_peaks_indices, high_peaks_indices = np.array(confirmed_low_indices), np.array(confirmed_high_indices)
    
    def cluster_and_strengthen(prices: np.ndarray, indices: np.ndarray, level_type: str) -> List[Dict]:
        if len(indices) < 2: return []
        points = prices[indices].reshape(-1, 1)
        eps_value = points.mean() * cluster_eps_percent
        if eps_value == 0: return []
        db = DBSCAN(eps=eps_value, min_samples=2).fit(points)
        clustered_levels = []
        for label in set(db.labels_):
            if label != -1:
                mask = (db.labels_ == label)
                cluster_indices = indices[mask]
                clustered_levels.append({
                    "level_price": float(prices[cluster_indices].mean()), "level_type": level_type,
                    "strength": int(len(cluster_indices)), "last_tested_at": df.index[cluster_indices[-1]].to_pydatetime()
                })
        return clustered_levels
    
    support_levels = cluster_and_strengthen(lows, low_peaks_indices, 'support')
    resistance_levels = cluster_and_strengthen(highs, high_peaks_indices, 'resistance')
    return support_levels + resistance_levels

def sr_analyze_volume_profile(df: pd.DataFrame, bins: int) -> List[Dict]:
    price_min, price_max = df['low'].min(), df['high'].max()
    if price_min >= price_max: return []
    price_bins = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
    volume_by_bin = np.zeros(bins)
    for _, row in df.iterrows():
        low_idx = np.searchsorted(price_bins, row['low'], side='right') - 1
        high_idx = np.searchsorted(price_bins, row['high'], side='left')
        low_idx, high_idx = max(0, low_idx), min(bins, high_idx)
        if high_idx > low_idx:
            volume_per_bin = row['volume'] / (high_idx - low_idx)
            for i in range(low_idx, high_idx): volume_by_bin[i] += volume_per_bin
    if np.sum(volume_by_bin) == 0: return []
    poc_index = np.argmax(volume_by_bin)
    return [{"level_price": float(bin_centers[poc_index]), "level_type": 'poc', "strength": float(volume_by_bin[poc_index]), "last_tested_at": None}]

def sr_calculate_fibonacci_levels(df: pd.DataFrame) -> List[Dict]:
    if df.empty: return []
    max_high, min_low = df['high'].max(), df['low'].min()
    diff = max_high - min_low
    if diff <= 0: return []
    fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    all_fib_levels = []
    for ratio in fib_ratios:
        details_s = f"Fib Support {ratio*100:.1f}%" + (" (Golden Level)" if ratio == 0.618 else "")
        all_fib_levels.append({"level_price": float(max_high - (diff * ratio)), "level_type": 'fib_support', "strength": 20 if ratio == 0.618 else 5, "details": details_s, "last_tested_at": None})
        details_r = f"Fib Resistance {ratio*100:.1f}%" + (" (Golden Level)" if ratio == 0.618 else "")
        all_fib_levels.append({"level_price": float(min_low + (diff * ratio)), "level_type": 'fib_resistance', "strength": 20 if ratio == 0.618 else 5, "details": details_r, "last_tested_at": None})
    return all_fib_levels

def sr_find_confluence_zones(levels: List[Dict], confluence_percent: float) -> Tuple[List[Dict], List[Dict]]:
    if not levels: return [], []
    levels.sort(key=lambda x: x['level_price'])
    tf_weights = {'1h': 3, '15m': 2, '5m': 1}; type_weights = {'poc': 2.5, 'support': 1.5, 'resistance': 1.5, 'fib_support': 1.2, 'fib_resistance': 1.2}
    confluence_zones, used_indices = [], set()
    for i in range(len(levels)):
        if i in used_indices: continue
        current_zone_levels, current_zone_indices = [levels[i]], {i}
        for j in range(i + 1, len(levels)):
            if j in used_indices: continue
            price_i, price_j = levels[i]['level_price'], levels[j]['level_price']
            if price_i > 0 and (abs(price_j - price_i) / price_i) <= confluence_percent:
                current_zone_levels.append(levels[j]); current_zone_indices.add(j)
        if len(current_zone_levels) > 1:
            used_indices.update(current_zone_indices)
            total_strength_for_avg = sum(l['strength'] for l in current_zone_levels)
            if total_strength_for_avg == 0: continue
            avg_price = sum(l['level_price'] * l['strength'] for l in current_zone_levels) / total_strength_for_avg
            total_strength = sum(l['strength'] * tf_weights.get(l.get('timeframe'), 1) * type_weights.get(l.get('level_type'), 1) for l in current_zone_levels)
            timeframes = sorted(list(set(str(l['timeframe']) for l in current_zone_levels)))
            details = sorted(list(set(l['level_type'] for l in current_zone_levels)))
            last_tested = max((l['last_tested_at'] for l in current_zone_levels if l['last_tested_at']), default=None)
            confluence_zones.append({"level_price": avg_price, "level_type": 'confluence', "strength": float(total_strength), "timeframe": ",".join(timeframes), "details": ",".join(details), "last_tested_at": last_tested})
    remaining_levels = [level for i, level in enumerate(levels) if i not in used_indices]
    return confluence_zones, remaining_levels

def sr_analyze_single_symbol(symbol: str) -> List[Dict]:
    logger.info(f"[SR] --- بدء تحليل (سكالبينج) للعملة: {symbol} ---")
    raw_levels = []
    df_15m = fetch_historical_data(symbol, '15m', SR_DATA_FETCH_DAYS_15M)
    if df_15m is not None and not df_15m.empty:
        fib_levels = sr_calculate_fibonacci_levels(df_15m)
        for level in fib_levels: level['timeframe'] = '15m'
        raw_levels.extend(fib_levels)
    
    timeframes_config = {
        '1h':  {'days': SR_DATA_FETCH_DAYS_1H,  'prominence_multiplier': SR_ATR_PROMINENCE_MULTIPLIER_1H,  'width': SR_WIDTH_1H},
        '15m': {'days': SR_DATA_FETCH_DAYS_15M, 'prominence_multiplier': SR_ATR_PROMINENCE_MULTIPLIER_15M, 'width': SR_WIDTH_15M},
        '5m':  {'days': SR_DATA_FETCH_DAYS_5M,  'prominence_multiplier': SR_ATR_PROMINENCE_MULTIPLIER_5M,  'width': SR_WIDTH_5M}
    }
    for tf, config in timeframes_config.items():
        df = df_15m if tf == '15m' and df_15m is not None else fetch_historical_data(symbol, tf, config['days'])
        if df is not None and not df.empty:
            atr_val = sr_calculate_atr(df, period=SR_ATR_PERIOD)
            pa_levels = sr_find_price_action_levels(df, atr_val, config['prominence_multiplier'], config['width'], SR_CLUSTER_EPS_PERCENT)
            vol_levels = sr_analyze_volume_profile(df, bins=SR_VOLUME_PROFILE_BINS)
            all_new = pa_levels + vol_levels
            for level in all_new: level['timeframe'] = tf
            raw_levels.extend(all_new)
        else:
            logger.warning(f"⚠️ [{symbol}-{tf}] تعذر جلب البيانات، سيتم التخطي.")
    
    if not raw_levels: return []
    confluence_zones, remaining_singles = sr_find_confluence_zones(raw_levels, SR_CONFLUENCE_ZONE_PERCENT)
    final_levels = confluence_zones + remaining_singles
    for level in final_levels:
        level['symbol'] = symbol; level['score'] = sr_calculate_level_score(level)
    logger.info(f"[SR] --- ✅ انتهى تحليل {symbol}، تم العثور على {len(final_levels)} مستوى نهائي. ---")
    return final_levels

def run_sr_scanner_full_analysis():
    """Runs a full analysis cycle for all symbols for SR levels."""
    logger.info("🚀 [SR] بدء دورة تحليل الدعم والمقاومة...")
    if not client: logger.error("[SR] Binance client غير متاح."); return
    if not check_db_connection(): logger.error("[SR] اتصال قاعدة البيانات غير متاح."); return
    
    symbols = get_validated_symbols()
    if not symbols: logger.warning("[SR] لا توجد عملات لتحليلها."); return

    all_final_levels = []
    with ThreadPoolExecutor(max_workers=SR_MAX_WORKERS) as executor:
        future_to_symbol = {executor.submit(sr_analyze_single_symbol, symbol): symbol for symbol in symbols}
        for i, future in enumerate(as_completed(future_to_symbol)):
            symbol = future_to_symbol[future]
            try:
                symbol_levels = future.result()
                if symbol_levels: all_final_levels.extend(symbol_levels)
            except Exception as e:
                logger.error(f"❌ [SR] خطأ فادح أثناء تحليل {symbol}: {e}", exc_info=True)

    if all_final_levels and check_db_connection() and conn:
        all_final_levels.sort(key=lambda x: x.get('score', 0), reverse=True)
        try:
            with conn.cursor() as cur:
                symbols_processed = list(set(level['symbol'] for level in all_final_levels))
                cur.execute("DELETE FROM support_resistance_levels WHERE symbol = ANY(%s);", (symbols_processed,))
                insert_query = "INSERT INTO support_resistance_levels (symbol, level_price, level_type, timeframe, strength, score, last_tested_at, details) VALUES %s ON CONFLICT (symbol, level_price, timeframe, level_type, details) DO NOTHING;"
                values = [(l.get('symbol'), l.get('level_price'), l.get('level_type'), l.get('timeframe'), l.get('strength'), l.get('score', 0), l.get('last_tested_at'), l.get('details')) for l in all_final_levels]
                execute_values(cur, insert_query, values)
            conn.commit()
            logger.info(f"✅ [SR DB] تم حفظ {len(all_final_levels)} مستوى بنجاح.")
        except Exception as e:
            logger.error(f"❌ [SR DB] خطأ في الحفظ: {e}", exc_info=True)
            if conn: conn.rollback()
    logger.info("🎉 [SR] اكتملت دورة تحليل الدعم والمقاومة.")


# ------------------------------------------------------------------------------
# --- ☁️ 4. دوال حاسب إيشيموكو (Ichimoku Calculator Functions - from c4i.py) ☁️ ---
# ------------------------------------------------------------------------------

def ichimoku_calculate(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates all Ichimoku Cloud components."""
    high, low, close = df['high'], df['low'], df['close']
    df['tenkan_sen'] = (high.rolling(window=ICHIMOKU_TENKAN_PERIOD).max() + low.rolling(window=ICHIMOKU_TENKAN_PERIOD).min()) / 2
    df['kijun_sen'] = (high.rolling(window=ICHIMOKU_KIJUN_PERIOD).max() + low.rolling(window=ICHIMOKU_KIJUN_PERIOD).min()) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(ICHIMOKU_SENKOU_SHIFT)
    df['senkou_span_b'] = ((high.rolling(window=ICHIMOKU_SENKOU_B_PERIOD).max() + low.rolling(window=ICHIMOKU_SENKOU_B_PERIOD).min()) / 2).shift(ICHIMOKU_SENKOU_SHIFT)
    df['chikou_span'] = close.shift(ICHIMOKU_CHIKOU_SHIFT)
    return df

def ichimoku_save_to_db(symbol: str, df_ichimoku: pd.DataFrame, timeframe: str):
    """Saves the calculated Ichimoku features to the database."""
    if not check_db_connection() or not conn or df_ichimoku.empty: return
    
    df_to_save = df_ichimoku[['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']].copy()
    ichimoku_cols = df_to_save.columns.tolist()
    df_to_save.dropna(subset=ichimoku_cols, how='all', inplace=True)
    if df_to_save.empty: return

    df_to_save.reset_index(inplace=True)
    tuples = [tuple(x) for x in df_to_save[['timestamp'] + ichimoku_cols].to_numpy()]
    data_to_insert = [(symbol, row[0], timeframe) + row[1:] for row in tuples]
    cols = ['symbol', 'timestamp', 'timeframe'] + ichimoku_cols
    update_cols = [f"{col} = EXCLUDED.{col}" for col in ichimoku_cols]
    query = f"INSERT INTO ichimoku_features ({', '.join(cols)}) VALUES %s ON CONFLICT (symbol, timestamp, timeframe) DO UPDATE SET {', '.join(update_cols)};"
    
    try:
        with conn.cursor() as cur:
            execute_values(cur, query, data_to_insert)
        conn.commit()
        logger.info(f"💾 [Ichimoku DB] تم حفظ/تحديث {len(data_to_insert)} سجل لـ {symbol}.")
    except Exception as e:
        logger.error(f"❌ [Ichimoku DB] خطأ في حفظ بيانات {symbol}: {e}")
        if conn: conn.rollback()

def run_ichimoku_calculator_full_analysis():
    """Runs a full analysis cycle for all symbols for Ichimoku features."""
    logger.info("🚀 [Ichimoku] بدء دورة حساب إيشيموكو...")
    if not client: logger.error("[Ichimoku] Binance client غير متاح."); return
    if not check_db_connection(): logger.error("[Ichimoku] اتصال قاعدة البيانات غير متاح."); return

    symbols = get_validated_symbols()
    if not symbols: logger.warning("[Ichimoku] لا توجد عملات لمعالجتها."); return

    for symbol in symbols:
        logger.info(f"--- ⏳ [Ichimoku] Processing {symbol} ---")
        try:
            df_ohlc = fetch_historical_data(symbol, ICHIMOKU_TIMEFRAME, ICHIMOKU_DATA_LOOKBACK_DAYS)
            if df_ohlc is None or df_ohlc.empty:
                logger.warning(f"Could not fetch data for {symbol}. Skipping.")
                continue
            df_with_ichimoku = ichimoku_calculate(df_ohlc)
            ichimoku_save_to_db(symbol, df_with_ichimoku, ICHIMOKU_TIMEFRAME)
        except Exception as e:
            logger.error(f"❌ [Ichimoku] خطأ حرج في معالجة {symbol}: {e}", exc_info=True)
    logger.info("🎉 [Ichimoku] اكتملت دورة حساب إيشيموكو.")


# ----------------------------------------------------------------------------------
# --- 🤖 5. دوال بوت التداول الرئيسي والمنطق (Main Bot Logic Functions - from c4.py) 🤖 ---
# ----------------------------------------------------------------------------------

# --- دوال جلب البيانات الخاصة بالبوت ---
def bot_fetch_sr_levels_from_db(symbol: str) -> pd.DataFrame:
    if not check_db_connection() or not conn: return pd.DataFrame()
    query = "SELECT level_price, level_type, score FROM support_resistance_levels WHERE symbol = %s"
    try:
        with conn.cursor() as cur:
            cur.execute(query, (symbol,))
            levels = cur.fetchall()
        return pd.DataFrame(levels) if levels else pd.DataFrame()
    except Exception as e:
        logger.error(f"❌ [Bot SR Fetch] Could not fetch S/R levels for {symbol}: {e}"); return pd.DataFrame()

def bot_fetch_ichimoku_features_from_db(symbol: str, timeframe: str) -> pd.DataFrame:
    if not check_db_connection() or not conn: return pd.DataFrame()
    query = "SELECT timestamp, tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span FROM ichimoku_features WHERE symbol = %s AND timeframe = %s ORDER BY timestamp;"
    try:
        with conn.cursor() as cur:
            cur.execute(query, (symbol, timeframe))
            features = cur.fetchall()
        if not features: return pd.DataFrame()
        df = pd.DataFrame(features)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        logger.error(f"❌ [Bot Ichimoku Fetch] Could not fetch Ichimoku for {symbol}: {e}"); return pd.DataFrame()

# --- دوال حساب الميزات الخاصة بالبوت ---
def bot_calculate_ichimoku_based_features(df: pd.DataFrame) -> pd.DataFrame:
    # (The code for this function is copied from c4.py)
    df['price_vs_tenkan'] = (df['close'] - df['tenkan_sen']) / df['tenkan_sen']
    df['price_vs_kijun'] = (df['close'] - df['kijun_sen']) / df['kijun_sen']
    df['tenkan_vs_kijun'] = (df['tenkan_sen'] - df['kijun_sen']) / df['kijun_sen']
    df['price_vs_kumo_a'] = (df['close'] - df['senkou_span_a']) / df['senkou_span_a']
    df['price_vs_kumo_b'] = (df['close'] - df['senkou_span_b']) / df['senkou_span_b']
    df['kumo_thickness'] = (df['senkou_span_a'] - df['senkou_span_b']).abs() / df['close']
    kumo_high = df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
    kumo_low = df[['senkou_span_a', 'senkou_span_b']].min(axis=1)
    df['price_above_kumo'] = (df['close'] > kumo_high).astype(int)
    df['price_below_kumo'] = (df['close'] < kumo_low).astype(int)
    df['price_in_kumo'] = ((df['close'] >= kumo_low) & (df['close'] <= kumo_high)).astype(int)
    df['chikou_above_kumo'] = (df['chikou_span'] > kumo_high).astype(int)
    df['chikou_below_kumo'] = (df['chikou_span'] < kumo_low).astype(int)
    cross_up = (df['tenkan_sen'].shift(1) < df['kijun_sen'].shift(1)) & (df['tenkan_sen'] > df['kijun_sen'])
    cross_down = (df['tenkan_sen'].shift(1) > df['kijun_sen'].shift(1)) & (df['tenkan_sen'] < df['kijun_sen'])
    df['tenkan_kijun_cross'] = 0
    df.loc[cross_up, 'tenkan_kijun_cross'] = 1
    df.loc[cross_down, 'tenkan_kijun_cross'] = -1
    return df

def bot_calculate_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    # (The code for this function is copied from c4.py)
    op, hi, lo, cl = df['open'], df['high'], df['low'], df['close']
    body = abs(cl - op); candle_range = hi - lo; candle_range[candle_range == 0] = 1e-9
    upper_wick = hi - pd.concat([op, cl], axis=1).max(axis=1)
    lower_wick = pd.concat([op, cl], axis=1).min(axis=1) - lo
    df['candlestick_pattern'] = 0
    df.loc[(body / candle_range) < 0.05, 'candlestick_pattern'] = 3 # Doji
    df.loc[(body > candle_range * 0.1) & (lower_wick >= body * 2) & (upper_wick < body), 'candlestick_pattern'] = 2 # Hammer
    df.loc[(body > candle_range * 0.1) & (upper_wick >= body * 2) & (lower_wick < body), 'candlestick_pattern'] = -2 # Shooting Star
    df.loc[(cl.shift(1) < op.shift(1)) & (cl > op) & (cl >= op.shift(1)) & (op <= cl.shift(1)) & (body > body.shift(1)), 'candlestick_pattern'] = 1 # Bullish Engulfing
    df.loc[(cl.shift(1) > op.shift(1)) & (cl < op) & (op >= cl.shift(1)) & (cl <= op.shift(1)) & (body > body.shift(1)), 'candlestick_pattern'] = -1 # Bearish Engulfing
    return df

def bot_calculate_sr_features(df: pd.DataFrame, sr_levels_df: pd.DataFrame) -> pd.DataFrame:
    # (The code for this function is copied from c4.py)
    if sr_levels_df.empty:
        df['dist_to_support'] = 0.0; df['dist_to_resistance'] = 0.0
        df['score_of_support'] = 0.0; df['score_of_resistance'] = 0.0
        return df
    supports = sr_levels_df[sr_levels_df['level_type'].str.contains('support|poc|confluence', case=False)]['level_price'].sort_values().to_numpy()
    resistances = sr_levels_df[sr_levels_df['level_type'].str.contains('resistance|poc|confluence', case=False)]['level_price'].sort_values().to_numpy()
    support_scores = sr_levels_df[sr_levels_df['level_type'].str.contains('support|poc|confluence', case=False)].set_index('level_price')['score'].to_dict()
    resistance_scores = sr_levels_df[sr_levels_df['level_type'].str.contains('resistance|poc|confluence', case=False)].set_index('level_price')['score'].to_dict()

    def get_sr_info(price):
        dist_s, score_s, dist_r, score_r = 1.0, 0.0, 1.0, 0.0
        if supports.size > 0:
            idx = np.searchsorted(supports, price, side='right') - 1
            if idx >= 0:
                s_price = supports[idx]
                dist_s = (price - s_price) / price if price > 0 else 0
                score_s = support_scores.get(s_price, 0)
        if resistances.size > 0:
            idx = np.searchsorted(resistances, price, side='left')
            if idx < len(resistances):
                r_price = resistances[idx]
                dist_r = (r_price - price) / price if price > 0 else 0
                score_r = resistance_scores.get(r_price, 0)
        return dist_s, score_s, dist_r, score_r
    results = df['close'].apply(get_sr_info)
    df[['dist_to_support', 'score_of_support', 'dist_to_resistance', 'score_of_resistance']] = pd.DataFrame(results.tolist(), index=df.index)
    return df

def bot_calculate_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    # (This function is a simplified merge of the feature calculation from c4.py)
    # Constants for calculation
    ADX_PERIOD, BBANDS_PERIOD, RSI_PERIOD = 14, 20, 14
    MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
    ATR_PERIOD, EMA_SLOW_PERIOD, EMA_FAST_PERIOD = 14, 200, 50
    STOCH_RSI_PERIOD, STOCH_K, STOCH_D = 14, 3, 3
    
    # ATR
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    # ADX
    plus_dm = pd.Series(np.where((df['high'].diff() > -df['low'].diff()) & (df['high'].diff() > 0), df['high'].diff(), 0.0), index=df.index)
    minus_dm = pd.Series(np.where((-df['low'].diff() > df['high'].diff()) & (-df['low'].diff() > 0), -df['low'].diff(), 0.0), index=df.index)
    plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df['atr']
    minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df['atr']
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    # MACD
    ema_fast = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    df['macd_hist'] = macd_line - macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    # BBands
    sma = df['close'].rolling(window=BBANDS_PERIOD).mean()
    std_dev = df['close'].rolling(window=BBANDS_PERIOD).std()
    df['bb_width'] = ((sma + (std_dev * 2)) - (sma - (std_dev * 2))) / (sma + 1e-9)
    # Stochastic RSI
    min_rsi = df['rsi'].rolling(window=STOCH_RSI_PERIOD).min()
    max_rsi = df['rsi'].rolling(window=STOCH_RSI_PERIOD).max()
    stoch_rsi_val = (df['rsi'] - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-9)
    df['stoch_rsi_k'] = stoch_rsi_val.rolling(window=STOCH_K).mean() * 100
    df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(window=STOCH_D).mean()
    # Other features
    df['price_vs_ema50'] = (df['close'] / df['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()) - 1
    df['price_vs_ema200'] = (df['close'] / df['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()) - 1
    if btc_df is not None and not btc_df.empty:
        merged_df = pd.merge(df, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
        df['btc_correlation'] = merged_df['close'].pct_change().rolling(window=30).corr(merged_df['btc_returns'])
    else:
        df['btc_correlation'] = 0.0
    df['hour_of_day'] = df.index.hour
    df = bot_calculate_candlestick_patterns(df)
    return df.astype('float32', errors='ignore')

# --- دوال تحميل النموذج والمنطق الاستراتيجي ---
def load_ml_model_bundle_from_folder(symbol: str) -> Optional[Dict[str, Any]]:
    global ml_models_cache
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_name in ml_models_cache: return ml_models_cache[model_name]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FOLDER, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        logger.warning(f"⚠️ [ML Model] ملف النموذج '{model_path}' غير موجود لـ {symbol}.")
        return None
    try:
        with open(model_path, 'rb') as f:
            model_bundle = pickle.load(f)
        if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
            ml_models_cache[model_name] = model_bundle
            logger.info(f"✅ [ML Model] تم تحميل النموذج '{model_name}' بنجاح.")
            return model_bundle
        logger.error(f"❌ [ML Model] حزمة النموذج '{model_path}' غير مكتملة.")
        return None
    except Exception as e:
        logger.error(f"❌ [ML Model] خطأ في تحميل النموذج لـ {symbol}: {e}", exc_info=True)
        return None

class TradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_bundle_from_folder(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame, sr_levels_df: pd.DataFrame, ichimoku_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            df_featured = bot_calculate_features(df_15m, btc_df)
            df_featured = bot_calculate_sr_features(df_featured, sr_levels_df)
            if not ichimoku_df.empty:
                df_featured = df_featured.join(ichimoku_df, how='left')
                df_featured = bot_calculate_ichimoku_based_features(df_featured)
            
            # Add MTF features
            df_4h['rsi_4h'] = bot_calculate_features(df_4h, None)['rsi']
            df_4h['price_vs_ema50_4h'] = bot_calculate_features(df_4h, None)['price_vs_ema50']
            df_featured = df_featured.join(df_4h[['rsi_4h', 'price_vs_ema50_4h']]).fillna(method='ffill')
            
            for col in self.feature_names:
                if col not in df_featured.columns: df_featured[col] = 0.0
            
            df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_featured[self.feature_names].dropna()
        except Exception as e:
            logger.error(f"❌ [{self.symbol}] فشل هندسة الميزات: {e}", exc_info=True)
            return None

    def generate_signal(self, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]) or df_features.empty: return None
        try:
            features_scaled = self.scaler.transform(df_features.iloc[[-1]])
            prediction = self.ml_model.predict(features_scaled)[0]
            prediction_proba = self.ml_model.predict_proba(features_scaled)[0]
            class_1_index = list(self.ml_model.classes_).index(1)
            prob_for_class_1 = prediction_proba[class_1_index]
            if prediction == 1 and prob_for_class_1 >= MODEL_CONFIDENCE_THRESHOLD:
                logger.info(f"✅ [Signal Found] {self.symbol}: Buy signal with confidence {prob_for_class_1:.2%}.")
                return {'symbol': self.symbol, 'strategy_name': BASE_ML_MODEL_NAME, 'signal_details': {'ML_Probability_Buy': f"{prob_for_class_1:.2%}"}}
            return None
        except Exception as e:
            logger.warning(f"⚠️ [Signal Gen] {self.symbol}: Error during generation: {e}")
            return None

# --- دوال إدارة الصفقات والتنبيهات ---
def log_and_notify(level: str, message: str, notification_type: str):
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error, 'critical': logger.critical}
    log_methods.get(level.lower(), logger.info)(message)
    if not check_db_connection() or not conn: return
    try:
        new_notification = {"timestamp": datetime.now().isoformat(), "type": notification_type, "message": message}
        with notifications_lock: notifications_cache.appendleft(new_notification)
        with conn.cursor() as cur:
            cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [Notify DB] فشل حفظ التنبيه: {e}");
        if conn: conn.rollback()

def send_telegram_message(target_chat_id: str, text: str):
    if not TELEGRAM_TOKEN or not target_chat_id: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(target_chat_id), 'text': text, 'parse_mode': 'Markdown'}
    try: requests.post(url, json=payload, timeout=10)
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

def send_new_signal_alert(signal_data: Dict[str, Any]):
    entry, target, sl = signal_data['entry_price'], signal_data['target_price'], signal_data['stop_loss']
    profit_pct = ((target / entry) - 1) * 100
    message = (f"💡 *إشارة تداول جديدة ({BASE_ML_MODEL_NAME})* 💡\n"
               f"🪙 *العملة:* `{signal_data['symbol']}`\n"
               f"⬅️ *الدخول:* `${entry:,.8g}`\n"
               f"🎯 *الهدف:* `${target:,.8g}` (`{profit_pct:+.2f}%`)\n"
               f"🛑 *وقف الخسارة:* `${sl:,.8g}`\n"
               f"🔍 *الثقة:* {signal_data['signal_details']['ML_Probability_Buy']}")
    send_telegram_message(CHAT_ID, message)
    log_and_notify('info', f"إشارة جديدة: {signal_data['symbol']} @ ${entry:,.8g}", "NEW_SIGNAL")

def insert_signal_into_db(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not check_db_connection() or not conn: return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;",
                (signal['symbol'], signal['entry_price'], signal['target_price'], signal['stop_loss'], signal.get('strategy_name'), json.dumps(signal.get('signal_details', {})))
            )
            signal['id'] = cur.fetchone()['id']
        conn.commit()
        logger.info(f"✅ [DB] تم إدراج الإشارة لـ {signal['symbol']} (ID: {signal['id']}).")
        return signal
    except Exception as e:
        logger.error(f"❌ [DB Insert] خطأ في إدراج إشارة {signal['symbol']}: {e}", exc_info=True)
        if conn: conn.rollback(); return None

def close_signal(signal: Dict, status: str, closing_price: float, closed_by: str):
    signal_id, symbol = signal.get('id'), signal.get('symbol')
    logger.info(f"Closing signal {signal_id} ({symbol}) with status '{status}'")
    if not check_db_connection() or not conn: return
    try:
        profit_pct = ((closing_price / signal['entry_price']) - 1) * 100
        with conn.cursor() as cur:
            cur.execute("UPDATE signals SET status = %s, closing_price = %s, closed_at = NOW(), profit_percentage = %s WHERE id = %s AND status = 'open';", (status, closing_price, profit_pct, signal_id))
            if cur.rowcount == 0: logger.warning(f"⚠️ [DB Close] Signal {signal_id} not found or already closed."); return
        conn.commit()
        status_map = {'target_hit': '✅ تحقق الهدف', 'stop_loss_hit': '🛑 ضرب وقف الخسارة', 'manual_close': '🖐️ أُغلقت يدوياً'}
        alert_msg = f"*{status_map.get(status, status)}*\n`{symbol}` | *الربح:* `{profit_pct:+.2f}%`"
        send_telegram_message(CHAT_ID, alert_msg)
        log_and_notify('info', f"{status_map.get(status, status)}: {symbol} | Profit: {profit_pct:+.2f}%", 'CLOSE_SIGNAL')
    except Exception as e:
        logger.error(f"❌ [DB Close] خطأ فادح في إغلاق {signal_id}: {e}", exc_info=True)
        if conn: conn.rollback()
        # Recovery mechanism
        if symbol:
            with signal_cache_lock:
                if symbol not in open_signals_cache:
                    open_signals_cache[symbol] = signal
                    logger.info(f"🔄 [Recovery] Signal {signal_id} for {symbol} returned to cache due to closing error.")
    finally:
        with closure_lock: signals_pending_closure.discard(signal_id)

# --- دوال وحلقات العمل في الخلفية ---
def handle_price_update_message(msg: List[Dict[str, Any]]):
    if not isinstance(msg, list) or not redis_client: return
    try:
        price_updates = {item.get('s'): float(item.get('c', 0)) for item in msg if item.get('s') and item.get('c')}
        if price_updates: redis_client.hset(REDIS_PRICES_HASH_NAME, mapping=price_updates)
    except Exception as e:
        logger.error(f"❌ [WebSocket Price] خطأ في معالجة رسالة السعر: {e}", exc_info=True)

def trade_monitoring_loop():
    logger.info("✅ [Trade Monitor] بدء مراقبة الصفقات...")
    while True:
        try:
            with signal_cache_lock: signals_to_check = dict(open_signals_cache)
            if not signals_to_check or not redis_client: time.sleep(1); continue
            
            symbols = list(signals_to_check.keys())
            prices = redis_client.hmget(REDIS_PRICES_HASH_NAME, symbols)
            latest_prices = {symbol: float(p) if p else None for symbol, p in zip(symbols, prices)}

            for symbol, signal in signals_to_check.items():
                price = latest_prices.get(symbol)
                if not price: continue
                
                signal_id = signal.get('id')
                with closure_lock:
                    if signal_id in signals_pending_closure: continue
                
                status, closing_price = None, None
                if price >= signal.get('target_price', float('inf')): status, closing_price = 'target_hit', price
                elif price <= signal.get('stop_loss', 0): status, closing_price = 'stop_loss_hit', price

                if status:
                    with closure_lock:
                        if signal_id in signals_pending_closure: continue
                        signals_pending_closure.add(signal_id)
                    with signal_cache_lock: open_signals_cache.pop(symbol, None)
                    logger.info(f"⚡ [Monitor Trigger] {status} for {symbol}. Initiating close.")
                    Thread(target=close_signal, args=(signal, status, closing_price, "auto_monitor")).start()
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"❌ [Trade Monitor] خطأ فادح: {e}", exc_info=True); time.sleep(5)

def main_scan_loop():
    logger.info("[Main Loop] انتظار 15 ثانية لاكتمال التهيئة...")
    time.sleep(15) 
    if not get_validated_symbols():
        log_and_notify("critical", "لا توجد رموز معتمدة للمسح. إيقاف الحلقة الرئيسية.", "SYSTEM"); return
    
    all_symbols = get_validated_symbols()
    while True:
        try:
            for i in range(0, len(all_symbols), MODEL_BATCH_SIZE):
                symbol_batch = all_symbols[i:i + MODEL_BATCH_SIZE]
                logger.info(f"🧠 [Memory] بدء معالجة الدفعة ({i // MODEL_BATCH_SIZE + 1}/{ -(-len(all_symbols) // MODEL_BATCH_SIZE) }).")
                ml_models_cache.clear(); gc.collect()
                
                if USE_BTC_TREND_FILTER and not get_btc_trend().get("is_uptrend"):
                    logger.warning("⚠️ [Scan Paused] تم إيقاف البحث بسبب اتجاه BTC الهابط."); time.sleep(300); break

                with signal_cache_lock: open_count = len(open_signals_cache)
                if open_count >= MAX_OPEN_TRADES:
                    logger.info(f"ℹ️ [Scan Paused] تم الوصول للحد الأقصى للصفقات ({open_count}/{MAX_OPEN_TRADES})."); time.sleep(60); break
                
                slots_available = MAX_OPEN_TRADES - open_count
                logger.info(f"ℹ️ [Scan Start] بدء دورة المسح. المراكز المتاحة: {slots_available}")
                
                btc_data = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, 90)
                if btc_data is not None: btc_data['btc_returns'] = btc_data['close'].pct_change()

                for symbol in symbol_batch:
                    if slots_available <= 0: break
                    with signal_cache_lock:
                        if symbol in open_signals_cache: continue
                    
                    try:
                        df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        if df_15m is None or df_4h is None: continue
                        
                        sr_levels = bot_fetch_sr_levels_from_db(symbol)
                        ichimoku_data = bot_fetch_ichimoku_features_from_db(symbol, SIGNAL_GENERATION_TIMEFRAME)
                        
                        strategy = TradingStrategy(symbol)
                        df_features = strategy.get_features(df_15m, df_4h, btc_data, sr_levels, ichimoku_data)
                        del df_15m, df_4h, sr_levels, ichimoku_data; gc.collect()
                        
                        if df_features is None or df_features.empty: continue
                        
                        potential_signal = strategy.generate_signal(df_features)
                        if potential_signal and redis_client:
                            current_price = float(redis_client.hget(REDIS_PRICES_HASH_NAME, symbol) or 0)
                            if not current_price: continue

                            potential_signal['entry_price'] = current_price
                            atr_value = df_features['atr'].iloc[-1]
                            potential_signal['stop_loss'] = current_price - (atr_value * ATR_SL_MULTIPLIER)
                            potential_signal['target_price'] = current_price + (atr_value * ATR_TP_MULTIPLIER)
                            
                            profit_percentage = ((potential_signal['target_price'] / current_price) - 1) * 100
                            if profit_percentage >= MIN_PROFIT_PERCENTAGE_FILTER:
                                saved_signal = insert_signal_into_db(potential_signal)
                                if saved_signal:
                                    with signal_cache_lock: open_signals_cache[saved_signal['symbol']] = saved_signal
                                    send_new_signal_alert(saved_signal)
                                    slots_available -= 1
                    except Exception as e:
                        logger.error(f"❌ [Processing Error] خطأ في معالجة {symbol}: {e}", exc_info=True)
                time.sleep(10) # Delay between batches
            logger.info("ℹ️ [Cycle End] انتهت دورة المسح الكاملة. انتظار..."); time.sleep(60)
        except (KeyboardInterrupt, SystemExit): break
        except Exception as main_err:
            log_and_notify("error", f"خطأ غير متوقع في الحلقة الرئيسية: {main_err}", "SYSTEM"); time.sleep(120)

# ------------------------------------------------------------------------------
# --- 🌐 6. واجهة برمجة تطبيقات Flask ولوحة التحكم (Flask API & Dashboard) 🌐 ---
# ------------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)

def get_btc_trend() -> Dict[str, Any]:
    # This is a simplified version for the dashboard
    try:
        klines = client.get_klines(symbol=BTC_SYMBOL, interval=BTC_TREND_TIMEFRAME, limit=2)
        current_price = float(klines[-1][4])
        prev_price = float(klines[-2][4])
        status = "Uptrend" if current_price > prev_price else "Downtrend"
        return {"status": status, "is_uptrend": (status == "Uptrend")}
    except Exception as e:
        return {"status": "Error", "is_uptrend": False}

@app.route('/')
def home():
    try:
        # Assuming index.html is in the same directory
        with open('index.html', 'r', encoding='utf-8') as f: return render_template_string(f.read())
    except FileNotFoundError: return "<h1>Dashboard file (index.html) not found.</h1>", 404

@app.route('/api/stats')
def get_stats():
    if not check_db_connection() or not conn: return jsonify({"error": "DB connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT status, profit_percentage FROM signals WHERE status != 'open';")
            closed_trades = cur.fetchall()
        with signal_cache_lock: open_trades_count = len(open_signals_cache)
        
        total_profit_pct = sum(s['profit_percentage'] for s in closed_trades if s.get('profit_percentage') is not None)
        return jsonify({
            "open_trades_count": open_trades_count,
            "total_profit_pct": total_profit_pct,
            "total_closed_trades": len(closed_trades)
        })
    except Exception as e:
        return jsonify({"error": f"Stats fetch error: {e}"}), 500

@app.route('/api/signals')
def get_signals():
    if not check_db_connection() or not conn or not redis_client: return jsonify({"error": "Service connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals ORDER BY CASE WHEN status = 'open' THEN 0 ELSE 1 END, id DESC;")
            all_signals = [dict(s) for s in cur.fetchall()]
        
        open_symbols = [s['symbol'] for s in all_signals if s['status'] == 'open']
        if open_symbols:
            prices = redis_client.hmget(REDIS_PRICES_HASH_NAME, open_symbols)
            current_prices = {symbol: float(p) if p else 0 for symbol, p in zip(open_symbols, prices)}
            for s in all_signals:
                if s['status'] == 'open':
                    price = current_prices.get(s['symbol'], 0)
                    s['current_price'] = price
                    s['pnl_pct'] = ((price / s['entry_price']) - 1) * 100 if s['entry_price'] > 0 else 0
        return jsonify(all_signals)
    except Exception as e:
        return jsonify({"error": f"Signals fetch error: {e}"}), 500

# Other Flask routes from c4.py can be added here...


# ------------------------------------------------------------------------------
# --- 🚀 7. المنظم الرئيسي ونقطة انطلاق البرنامج (Main Orchestrator & Entry Point) 🚀 ---
# ------------------------------------------------------------------------------

def sr_scanner_scheduler():
    """Background job to run the SR scanner periodically."""
    while True:
        try:
            run_sr_scanner_full_analysis()
        except Exception as e:
            logger.error(f"❌ [SR Scheduler] خطأ فادح في دورة التحليل: {e}", exc_info=True)
        logger.info(f"👍 [SR Scheduler] اكتملت الدورة. انتظار {SR_RUN_INTERVAL_MINUTES} دقيقة.")
        time.sleep(SR_RUN_INTERVAL_MINUTES * 60)

def ichimoku_calculator_scheduler():
    """Background job to run the Ichimoku calculator periodically."""
    while True:
        try:
            run_ichimoku_calculator_full_analysis()
        except Exception as e:
            logger.error(f"❌ [Ichimoku Scheduler] خطأ فادح في دورة الحساب: {e}", exc_info=True)
        logger.info(f"👍 [Ichimoku Scheduler] اكتملت الدورة. انتظار {ICHIMOKU_RUN_INTERVAL_HOURS} ساعات.")
        time.sleep(ICHIMOKU_RUN_INTERVAL_HOURS * 60 * 60)

def initialize_and_load_cache():
    """Loads initial data into cache from the database."""
    if not check_db_connection() or not conn: return
    logger.info("ℹ️ [Cache] جاري تحميل الإشارات المفتوحة والتنبيهات...")
    try:
        # Load open signals
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status = 'open';")
            open_signals = cur.fetchall()
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in open_signals: open_signals_cache[signal['symbol']] = dict(signal)
        # Load notifications
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 50;")
            recent = cur.fetchall()
            with notifications_lock:
                notifications_cache.clear()
                for n in reversed(recent):
                    n['timestamp'] = n['timestamp'].isoformat()
                    notifications_cache.appendleft(dict(n))
        logger.info(f"✅ [Cache] تم تحميل {len(open_signals_cache)} إشارة مفتوحة و {len(notifications_cache)} تنبيه.")
    except Exception as e:
        logger.error(f"❌ [Cache] فشل تحميل البيانات: {e}")

def start_background_services():
    """
    ✨ NEW: This function runs all the time-consuming initial setup and starts
    the background threads. It's designed to be run in a separate thread
    to not block the main Flask web server from starting.
    """
    logger.info("▶️ [Background] بدء التشغيل الأولي للتحليلات (قد يستغرق بعض الوقت)...")
    run_sr_scanner_full_analysis()
    run_ichimoku_calculator_full_analysis()
    logger.info("✅ [Background] اكتمل التشغيل الأولي للتحليلات بنجاح.")

    # تحميل البيانات المبدئية والذاكرة المؤقتة
    initialize_and_load_cache()

    # بدء الخدمات الخلفية للبوت
    logger.info("▶️ [Background] بدء الخدمات الخلفية للبوت...")
    Thread(target=trade_monitoring_loop, daemon=True, name="TradeMonitor").start()
    
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    twm.start_miniticker_socket(callback=handle_price_update_message)
    
    Thread(target=main_scan_loop, daemon=True, name="MainScanLoop").start()
    logger.info("✅ [Background] تم بدء خدمات البوت (WebSocket, Monitor, Main Loop).")

    # بدء المهام الدورية للتحليل في الخلفية
    logger.info("▶️ [Background] بدء المهام الدورية للتحليل في الخلفية...")
    Thread(target=sr_scanner_scheduler, daemon=True, name="SR_Scheduler").start()
    Thread(target=ichimoku_calculator_scheduler, daemon=True, name="Ichimoku_Scheduler").start()
    logger.info("✅ [Background] تم بدء المهام الدورية (SR & Ichimoku).")


if __name__ == "__main__":
    logger.info("======================================================")
    logger.info("=== 🚀 بدء تشغيل منظم بوت التداول (Orchestrator) 🚀 ===")
    logger.info("======================================================")

    # --- 1. تهيئة الخدمات الأساسية ---
    init_db()
    init_redis()
    init_binance_client()
    get_validated_symbols()

    # --- 2. ✨ NEW: بدء جميع مهام الإعداد والتحليل في خيط خلفي ---
    # هذا يضمن أن خادم الويب يمكن أن يبدأ على الفور.
    background_setup_thread = Thread(target=start_background_services, daemon=True, name="BackgroundSetup")
    background_setup_thread.start()

    # --- 3. تشغيل لوحة التحكم (Flask) في الخيط الرئيسي ---
    # سيبدأ هذا على الفور، مما يحل مشكلة انتهاء الوقت.
    host, port = "0.0.0.0", int(os.environ.get('PORT', 10000))
    log_and_notify("info", f"بدء تشغيل لوحة التحكم على http://{host}:{port}", "SYSTEM")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        logger.warning("⚠️ [Flask] مكتبة 'waitress' غير موجودة, سيتم استخدام خادم التطوير.")
        app.run(host=host, port=port)

    logger.info("👋 [إيقاف] تم إيقاف تشغيل المنظم. وداعاً!")
    os._exit(0)
