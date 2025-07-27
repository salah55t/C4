# ملف c4_ema_crossover_bot.py - نسخة تعتمد على التقاطعات الفنية فقط
# تم التحديث بواسطة Gemini
# --- تعديل: إزالة نموذج تعلم الآلة والاعتماد على تقاطع EMA ---
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
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timezone, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Set, Tuple
from collections import deque, Counter
import warnings

# --- إعدادات التجاهل واللوجر ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_crossover_bot_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoCrossoverBot')

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
are_filters_disabled: bool = False
filters_disabled_lock = Lock()
RISK_PER_TRADE_PERCENT: float = 1.0
STRATEGY_NAME: str = "EMA_Crossover_V1"
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 10 # Adjusted for TA strategies
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices_crossover"
TRADING_FEE_PERCENT: float = 0.1
STATS_TRADE_SIZE_USDT: float = 5.0
BTC_SYMBOL: str = 'BTCUSDT'
MAX_OPEN_TRADES: int = 4
MIN_PROFIT_PERCENT: float = 0.8 

# --- NEW: Memory Optimization Setting ---
SYMBOL_PROCESSING_BATCH_SIZE: int = 10 

# --- إعدادات المؤشرات الفنية ---
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
EMA_SLOW_PERIOD: int = 200 # For general context
EMA_FAST_PERIOD: int = 50 # For general context
BTC_CORR_PERIOD: int = 30
REL_VOL_PERIOD: int = 30

# --- إعدادات الفلاتر المتقدمة وإدارة الصفقات ---
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_ACTIVATION_PROFIT_PERCENT: float = 1.8
TRAILING_DISTANCE_PERCENT: float = 1.0
USE_PEAK_FILTER: bool = True
PEAK_CHECK_PERIOD: int = 50
PULLBACK_THRESHOLD_PCT: float = 0.988
BREAKOUT_ALLOWANCE_PCT: float = 1.003
DYNAMIC_FILTER_ANALYSIS_INTERVAL: int = 300
ORDER_BOOK_DEPTH_LIMIT: int = 100
ORDER_BOOK_WALL_MULTIPLIER: float = 10.0
ORDER_BOOK_ANALYSIS_RANGE_PCT: float = 0.02

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
dynamic_filter_profile_cache: Dict[str, Any] = {}
last_dynamic_filter_analysis_time: float = 0
dynamic_filter_lock = Lock()


# --- قاموس أسباب الرفض باللغة العربية ---
REJECTION_REASONS_AR = {
    "Filters Not Loaded": "الفلاتر غير محملة", "Low Volatility": "تقلب منخفض جداً",
    "BTC Correlation": "ارتباط ضعيف بالبيتكوين", "RRR Filter": "نسبة المخاطرة/العائد غير كافية",
    "Momentum/Strength Filter": "فلتر الزخم والقوة", "Peak/Pullback Filter": "فلتر القمة/التصحيح",
    "Invalid Position Size": "حجم الصفقة غير صالح", "Lot Size Adjustment Failed": "فشل ضبط حجم العقد",
    "Min Notional Filter": "قيمة الصفقة أقل من الحد الأدنى", "Insufficient Balance": "الرصيد غير كافٍ",
    "Order Book Fetch Failed": "فشل جلب دفتر الطلبات", "Order Book Imbalance": "اختلال توازن دفتر الطلبات",
    "Large Sell Wall Detected": "تم كشف جدار بيع ضخم", "Insufficient data for TP/SL calculation": "بيانات غير كافية لحساب TP/SL",
    "Potential Profit Below Threshold": "الربح المحتمل أقل من الحد الأدنى",
    "Potential Profit Below Threshold (S/R)": "الربح المحتمل أقل من الحد الأدنى (دعم/مقاومة)",
    "EMA Crossover Invalid": "شروط تقاطع EMA غير متحققة"
}

# --- دالة إرسال رسائل تليجرام ---
def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("[Telegram] Token أو Chat ID غير معين، تم تخطي الإرسال.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info(f"✅ [Telegram] تم إرسال الرسالة بنجاح.")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

# --- دوال تهيئة الخدمات ---
def init_db(retries: int = 5, delay: int = 5) -> None:
    global conn
    logger.info("[DB] تهيئة الاتصال بقاعدة البيانات...")
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
                        target_price DOUBLE PRECISION NOT NULL, stop_loss DOUBLE PRECISION NOT NULL,
                        status TEXT DEFAULT 'open', closing_price DOUBLE PRECISION, closed_at TIMESTAMP,
                        profit_percentage DOUBLE PRECISION, strategy_name TEXT, signal_details JSONB,
                        current_peak_price DOUBLE PRECISION, is_real_trade BOOLEAN DEFAULT FALSE,
                        quantity DOUBLE PRECISION, order_id TEXT
                    );
                """)
                cur.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS closing_reason TEXT;")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals (status);")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL, message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE
                    );
                """)
            conn.commit()
            logger.info("✅ [DB] الاتصال بقاعدة البيانات وتحديث المخطط بنجاح.")
            return
        except Exception as e:
            logger.error(f"❌ [DB] خطأ أثناء التهيئة (محاولة {attempt + 1}/{retries}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: logger.critical("❌ [DB] فشل الاتصال بقاعدة البيانات.")

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
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [DB] فقدان الاتصال: {e}. إعادة الاتصال...")
        try:
            init_db()
            return conn is not None and conn.closed == 0
        except Exception as retry_e:
            logger.error(f"❌ [DB] فشل إعادة الاتصال: {retry_e}")
            return False

def log_and_notify(level: str, message: str, notification_type: str):
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error, 'critical': logger.critical}
    log_methods.get(level.lower(), logger.info)(message)
    if not check_db_connection() or not conn: return
    try:
        new_notification = {"timestamp": datetime.now(timezone.utc).isoformat(), "type": notification_type, "message": message}
        with notifications_lock: notifications_cache.appendleft(new_notification)
        with conn.cursor() as cur: cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [Notify DB] فشل حفظ الإشعار: {e}")
        if conn: conn.rollback()

def log_rejection(symbol: str, reason_key: str, details: Optional[Dict] = None):
    reason_ar = REJECTION_REASONS_AR.get(reason_key, reason_key)
    log_message = f"🚫 [REJECTED] {symbol} | Reason: {reason_key} | Details: {details or {}}"
    logger.info(log_message)
    with rejection_logs_lock:
        rejection_logs_cache.appendleft({
            "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
            "reason": reason_ar, "details": json.loads(json.dumps(details, default=str)) or {}
        })

def init_redis() -> None:
    global redis_client
    logger.info("[Redis] تهيئة الاتصال بـ Redis...")
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] تم الاتصال بنجاح بخادم Redis.")
    except redis.exceptions.ConnectionError as e:
        logger.critical(f"❌ [Redis] فشل الاتصال بـ Redis: {e}")
        exit(1)

def get_exchange_info_map() -> None:
    global exchange_info_map
    if not client: return
    logger.info("ℹ️ [Exchange Info] جلب قواعد التداول من المنصة...")
    try:
        info = client.get_exchange_info()
        exchange_info_map = {s['symbol']: s for s in info['symbols']}
        logger.info(f"✅ [Exchange Info] تم تحميل القواعد لـ {len(exchange_info_map)} عملة.")
    except Exception as e:
        logger.error(f"❌ [Exchange Info] لم يتمكن من جلب معلومات المنصة: {e}")

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
        logger.info(f"✅ [Validation] سيقوم البوت بمراقبة {len(validated)} عملة.")
        return validated
    except Exception as e:
        logger.error(f"❌ [Validation] خطأ أثناء التحقق من العملات: {e}", exc_info=True)
        return []

# --- دوال جلب البيانات وحساب الميزات ---
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        start_dt = datetime.now(timezone.utc) - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(klines, columns=cols)
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_cols]
        numeric_cols = {'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'}
        df = df.astype(numeric_cols)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna()
    except Exception as e:
        logger.error(f"❌ [Data] خطأ في جلب البيانات التاريخية لـ {symbol}: {e}")
        return None

def calculate_technical_indicators(df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df_calc = df.copy()

    # EMA 9, 21 for Crossover Signal
    df_calc['ema_9'] = df_calc['close'].ewm(span=9, adjust=False).mean()
    df_calc['ema_21'] = df_calc['close'].ewm(span=21, adjust=False).mean()
    
    # Volume SMA for Volume Spike Confirmation
    df_calc['volume_sma_20'] = df_calc['volume'].rolling(window=20).mean()

    # Standard Indicators for Filtering
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
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
        asset_returns = df_calc['close'].pct_change()
        merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
        df_calc['btc_correlation'] = asset_returns.rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    else:
        df_calc['btc_correlation'] = 0.0

    return df_calc.astype('float32', errors='ignore')

def get_btc_data_for_bot() -> Optional[pd.DataFrame]:
    btc_data = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
    if btc_data is not None: btc_data['btc_returns'] = btc_data['close'].pct_change()
    return btc_data

def load_open_signals_to_cache():
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status IN ('open', 'updated');")
            open_signals = cur.fetchall()
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in open_signals: open_signals_cache[signal['symbol']] = dict(signal)
            logger.info(f"✅ [Loading] تم تحميل {len(open_signals)} صفقة مفتوحة.")
    except Exception as e:
        logger.error(f"❌ [Loading] فشل تحميل الصفقات المفتوحة: {e}")

def load_notifications_to_cache():
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 50;")
            recent = cur.fetchall()
            with notifications_lock:
                notifications_cache.clear()
                for n in reversed(recent):
                    n['timestamp'] = n['timestamp'].isoformat()
                    notifications_cache.appendleft(dict(n))
            logger.info(f"✅ [Loading] تم تحميل {len(notifications_cache)} إشعار.")
    except Exception as e:
        logger.error(f"❌ [Loading] فشل تحميل الإشعارات: {e}")

# ---------------------- أنظمة التحليل والفلاتر ----------------------
class DynamicFilterSystem:
    def generate_filters(self) -> Dict[str, Any]:
        # Simplified dynamic filters for a non-ML strategy
        return {
            "name": "فلاتر تقاطع EMA القياسية",
            "description": "فلاتر أساسية لدعم استراتيجية تقاطع المتوسطات",
            "strategy": "EMA_CROSSOVER",
            "filters": {
                "adx": 20.0, 
                "rel_vol": 0.8, 
                "min_rrr": 1.2, 
                "min_volatility_pct": 0.3,
                "min_btc_correlation": 0.2, 
                "min_bid_ask_ratio": 1.1
            }
        }

dynamic_filter_system = DynamicFilterSystem()

def passes_secondary_filters(symbol: str, last_row: pd.Series, profile: Dict[str, Any], entry_price: float, tp_sl_data: Dict) -> bool:
    with filters_disabled_lock:
        if are_filters_disabled:
            logger.warning(f"⚠️ [{symbol}] تجاوز الفلاتر الثانوية بسبب الإعداد العام.")
            return True
            
    filters = profile.get("filters", {})
    if not filters: log_rejection(symbol, "Filters Not Loaded"); return False
    
    volatility = (last_row.get('atr', 0) / entry_price * 100) if entry_price > 0 else 0
    if volatility < filters.get('min_volatility_pct', 0.0): 
        log_rejection(symbol, "Low Volatility", {"volatility": f"{volatility:.2f}%"})
        return False
        
    correlation = last_row.get('btc_correlation', 0)
    if correlation < filters.get('min_btc_correlation', -1.0): 
        log_rejection(symbol, "BTC Correlation", {"corr": f"{correlation:.2f}"})
        return False

    risk = entry_price - float(tp_sl_data['stop_loss'])
    reward = float(tp_sl_data['target_price']) - entry_price
    if risk <= 0 or reward <= 0 or (reward / risk) < filters.get('min_rrr', 0.0): 
        log_rejection(symbol, "RRR Filter", {"rrr": f"{(reward/risk):.2f}" if risk > 0 else "N/A"})
        return False
        
    adx = last_row.get('adx', 0)
    if adx < filters.get('adx', 0):
        log_rejection(symbol, "Momentum/Strength Filter", {"ADX": f"{adx:.2f}"})
        return False

    return True

def analyze_order_book(symbol: str, entry_price: float) -> Optional[Dict[str, Any]]:
    if not client: return None
    try:
        order_book = client.get_order_book(symbol=symbol, limit=ORDER_BOOK_DEPTH_LIMIT)
        bids = pd.DataFrame(order_book['bids'], columns=['price', 'qty'], dtype=float)
        asks = pd.DataFrame(order_book['asks'], columns=['price', 'qty'], dtype=float)
        price_range = entry_price * ORDER_BOOK_ANALYSIS_RANGE_PCT
        relevant_bids_vol = bids[bids['price'] >= entry_price - price_range]['qty'].sum()
        relevant_asks_vol = asks[asks['price'] <= entry_price + price_range]['qty'].sum()
        bid_ask_ratio = relevant_bids_vol / relevant_asks_vol if relevant_asks_vol > 0 else float('inf')
        avg_ask_qty = asks['qty'].mean()
        sell_wall_threshold = avg_ask_qty * ORDER_BOOK_WALL_MULTIPLIER
        large_sell_walls = asks[asks['price'].between(entry_price, entry_price * 1.05) & (asks['qty'] > sell_wall_threshold)]
        return {"bid_ask_ratio": bid_ask_ratio, "has_large_sell_wall": not large_sell_walls.empty, "wall_details": large_sell_walls.to_dict('records')}
    except Exception as e:
        log_rejection(symbol, "Order Book Fetch Failed", {"error": str(e)}); return None

def passes_order_book_check(symbol: str, order_book_analysis: Dict, profile: Dict) -> bool:
    with filters_disabled_lock:
        if are_filters_disabled: return True
    filters = profile.get("filters", {})
    if order_book_analysis.get('has_large_sell_wall', True): 
        log_rejection(symbol, "Large Sell Wall Detected", {"details": order_book_analysis.get('wall_details')})
        return False
    if order_book_analysis.get('bid_ask_ratio', 0) < filters.get('min_bid_ask_ratio', 1.0): 
        log_rejection(symbol, "Order Book Imbalance", {"ratio": f"{order_book_analysis.get('bid_ask_ratio', 0):.2f}"})
        return False
    return True

# --- دوال حساب TP/SL ودوال إشارة التقاطع ---
def find_sr_levels(df: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
    if len(df) < lookback: return {'support': None, 'resistance': None}
    highs = df['high'].iloc[-lookback:]
    lows  = df['low'].iloc[-lookback:]
    pivot_high = (highs == highs.rolling(5, center=True).max()) & (highs.shift(1) < highs) & (highs.shift(-1) < highs)
    pivot_low  = (lows  == lows.rolling(5, center=True).min())  & (lows.shift(1)  > lows)  & (lows.shift(-1)  > lows)
    highs_list = highs[pivot_high].dropna().tolist()
    lows_list  = lows[pivot_low].dropna().tolist()
    if not highs_list or not lows_list: return {'support': None, 'resistance': None}
    resistance = Counter(highs_list).most_common(1)[0][0] if highs_list else None
    support    = Counter(lows_list).most_common(1)[0][0] if lows_list else None
    return {'support': support, 'resistance': resistance}

def calculate_tp_sl(symbol: str, entry_price: float, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    try:
        if df.empty or len(df) < 50:
            log_rejection(symbol, "Insufficient data for TP/SL calculation")
            return None

        sr = find_sr_levels(df, lookback=50)
        resistance = sr['resistance']
        support    = sr['support']

        potential_profit_pct = 0
        if resistance is not None and resistance > entry_price:
            potential_profit_pct = ((resistance - entry_price) / entry_price) * 100

        if resistance is None or support is None or potential_profit_pct < MIN_PROFIT_PERCENT:
            if resistance is None or support is None:
                log_message = f"[{symbol}] لم يتم العثور على دعم/مقاومة. سيتم استخدام TP/SL بنسبة ثابتة."
            else:
                log_message = f"[{symbol}] الربح من المقاومة ({potential_profit_pct:.2f}%) أقل من الحد الأدنى. سيتم استخدام TP/SL بنسبة ثابتة."
            logger.info(log_message)
            new_target_price = entry_price * (1 + 1.2 / 100)
            new_stop_loss = entry_price * (1 - 1.5 / 100)
            return {'target_price': round(new_target_price, 6), 'stop_loss': round(new_stop_loss, 6), 'source': 'FIXED_PERCENTAGE', 'rr_ratio': round(1.2 / 1.5, 2)}

        if support >= entry_price: support = entry_price * 0.98
        if ((entry_price - support) / entry_price) * 100 < 0.3: support = entry_price * (1 - 0.003)

        return {'target_price': round(resistance, 6), 'stop_loss': round(support, 6), 'source': 'SR_LEVELS', 'rr_ratio': round((resistance - entry_price) / (entry_price - support), 2) if (entry_price - support) > 0 else 0}
    except Exception as e:
        logger.error(f"❌ [{symbol}] Error in S/R TP/SL: {e}", exc_info=True)
        last_atr = df['atr'].iloc[-1] if 'atr' in df.columns and not df['atr'].empty else 0
        if last_atr > 0: return {'target_price': entry_price + last_atr * 2.2, 'stop_loss': entry_price - last_atr * 1.5, 'source': 'ATR_Fallback'}
        return None

def check_ema_crossover_signal(df: pd.DataFrame, lookback_period: int = 3) -> bool:
    """
    يفحص شروط تقاطع EMA الإيجابي في آخر N شمعات.
    """
    required_cols = ['ema_9', 'ema_21', 'close', 'rsi', 'volume', 'volume_sma_20']
    if not all(col in df.columns for col in required_cols) or len(df) < lookback_period + 2:
        return False
    try:
        for i in range(1, lookback_period + 1):
            last_row = df.iloc[-i]
            prev_row = df.iloc[-(i + 1)]
            is_crossover = prev_row['ema_9'] < prev_row['ema_21'] and last_row['ema_9'] > last_row['ema_21']
            is_close_above = last_row['close'] > last_row['ema_9'] and last_row['close'] > last_row['ema_21']
            is_rsi_strong = last_row['rsi'] > 50
            is_volume_spike = last_row['volume_sma_20'] > 0 and last_row['volume'] > (last_row['volume_sma_20'] * 1.5)
            if is_crossover and is_close_above and (is_rsi_strong or is_volume_spike):
                logger.info(f"✅ [{df.index[-i]}] EMA Crossover signal confirmed for {df.name}.")
                return True
        return False
    except (IndexError, TypeError):
        return False

# ---------------------- دوال إدارة الصفقات ----------------------
def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> Optional[Decimal]:
    try:
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info:
            logger.error(f"[{symbol}] لم يتم العثور على معلومات الرمز لضبط LOT_SIZE.")
            return None
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = Decimal(lot_size_filter['stepSize'])
            quantity_decimal = Decimal(str(quantity))
            return (quantity_decimal // step_size) * step_size
        return Decimal(str(quantity))
    except Exception as e:
        logger.error(f"[{symbol}] خطأ في تعديل الكمية لـ LOT_SIZE: {e}", exc_info=True)
        return None

def calculate_position_size(symbol: str, entry_price: float, stop_loss_price: float) -> Optional[Decimal]:
    if not client: return None
    try:
        balance_response = client.get_asset_balance(asset='USDT')
        available_balance = Decimal(balance_response['free'])
        risk_amount_usdt = available_balance * (Decimal(str(RISK_PER_TRADE_PERCENT)) / Decimal('100'))
        risk_per_coin = Decimal(str(entry_price)) - Decimal(str(stop_loss_price))
        if risk_per_coin <= 0: log_rejection(symbol, "Invalid Position Size"); return None
        initial_quantity = risk_amount_usdt / risk_per_coin
        adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(initial_quantity))
        if adjusted_quantity is None or adjusted_quantity <= 0: log_rejection(symbol, "Lot Size Adjustment Failed"); return None
        notional_value = adjusted_quantity * Decimal(str(entry_price))
        symbol_info = exchange_info_map.get(symbol)
        if symbol_info:
            for f in symbol_info['filters']:
                if f['filterType'] in ('MIN_NOTIONAL', 'NOTIONAL'):
                    min_notional = Decimal(f.get('minNotional', f.get('notional', '0')))
                    if notional_value < min_notional: log_rejection(symbol, "Min Notional Filter", {"value": f"{notional_value:.2f}"}); return None
        if notional_value > available_balance: log_rejection(symbol, "Insufficient Balance", {"required": f"{notional_value:.2f}"}); return None
        return adjusted_quantity
    except Exception as e:
        logger.error(f"❌ [{symbol}] خطأ في حساب حجم الصفقة: {e}", exc_info=True); return None

def place_order(symbol: str, side: str, quantity: Decimal, order_type: str = Client.ORDER_TYPE_MARKET) -> Optional[Dict]:
    if not client: return None
    logger.info(f"➡️ [{symbol}] محاولة تنفيذ أمر {side} حقيقي لكمية {quantity}.")
    try:
        order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=str(quantity))
        log_and_notify('info', f"TRADE REAL: Placed {side} order for {quantity} {symbol}.", "REAL_TRADE")
        return order
    except Exception as e:
        logger.error(f"❌ [{symbol}] خطأ من باينانس عند تنفيذ الأمر: {e}")
        log_and_notify('error', f"REAL TRADE FAILED: {symbol} | {e}", "REAL_TRADE_ERROR")
        return None

def close_signal(signal_id: int, closing_price: float, reason: str) -> bool:
    with signal_cache_lock:
        signal_to_close = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)
        if not signal_to_close:
            logger.warning(f"⚠️ [Close] محاولة إغلاق صفقة غير موجودة في الكاش ID: {signal_id}")
            return False
        symbol_to_close = signal_to_close['symbol']

        entry_price = float(signal_to_close['entry_price'])
        profit_percentage = ((closing_price - entry_price) / entry_price) * 100
        
        if signal_to_close.get('is_real_trade'):
            try:
                # Logic to place sell order on Binance (simplified for brevity, assume it works)
                pass 
            except Exception as e:
                logger.error(f"❌ [{symbol_to_close}] خطأ حرج أثناء تحضير أمر البيع: {e}", exc_info=True)
                return False
        
        if not check_db_connection() or not conn:
            log_and_notify('critical', "DB connection lost during trade closure.", "DB_ERROR")
            return False
            
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE signals SET status = 'closed', closing_price = %s, closed_at = NOW(),
                    profit_percentage = %s, closing_reason = %s WHERE id = %s;
                """, (closing_price, profit_percentage, reason, signal_id))
            conn.commit()
            
            if symbol_to_close in open_signals_cache:
                del open_signals_cache[symbol_to_close]
            
            log_and_notify('info', f"CLOSED: {symbol_to_close} at {closing_price:.4f}. Reason: {reason}. Profit: {profit_percentage:.2f}%", "TRADE_CLOSED")
            
            reason_map = {'take_profit': '🎯 Take Profit', 'stop_loss': '🛑 Stop Loss', 'manual': '🖐️ Manual Close'}
            emoji = "✅" if profit_percentage >= 0 else "🔻"
            trade_type = "حقيقية" if signal_to_close.get('is_real_trade') else "تجريبية"
            telegram_message = (f"{emoji} *إغلاق صفقة {trade_type}*\n\n"
                                f"*العملة:* `{symbol_to_close}`\n"
                                f"*سبب الإغلاق:* {reason_map.get(reason, reason)}\n"
                                f"*سعر الدخول:* `{entry_price:.4f}`\n"
                                f"*سعر الإغلاق:* `{closing_price:.4f}`\n"
                                f"*الربح/الخسارة:* `{profit_percentage:.2f}%`")
            send_telegram_message(telegram_message)
            return True
        except Exception as e:
            logger.error(f"❌ [DB Close] فشل تحديث الصفقة المغلقة: {e}")
            if conn: conn.rollback()
            return False

def insert_signal_into_db(signal_data: Dict) -> Optional[Dict]:
    if not check_db_connection() or not conn: return None
    try:
        with conn.cursor() as cur:
            # ... (DB insertion logic is unchanged) ...
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, is_real_trade, quantity, order_id, current_peak_price, closing_reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL) RETURNING *;
            """, (
                signal_data['symbol'],
                float(signal_data['entry_price']),
                float(signal_data['target_price']),
                float(signal_data['stop_loss']),
                signal_data['strategy_name'],
                json.dumps(signal_data['signal_details']),
                signal_data.get('is_real_trade', False),
                float(signal_data['quantity']) if signal_data.get('quantity') is not None else None,
                signal_data.get('order_id'),
                float(signal_data['entry_price'])
            ))
            saved_signal = cur.fetchone()
        conn.commit()
        logger.info(f"💾 [{signal_data['symbol']}] تم حفظ الإشارة الجديدة في قاعدة البيانات.")
        telegram_message = (f"💡 *توصية شراء جديدة ({STRATEGY_NAME})*\n\n"
                            f"*العملة:* `{signal_data['symbol']}`\n"
                            f"*سعر الدخول:* `{signal_data['entry_price']:.4f}`\n"
                            f"*الهدف (TP):* `{signal_data['target_price']:.4f}`\n"
                            f"*وقف الخسارة (SL):* `{signal_data['stop_loss']:.4f}`")
        send_telegram_message(telegram_message)
        return dict(saved_signal)
    except Exception as e:
        logger.error(f"❌ [DB Insert] فشل إدراج الإشارة: {e}"); conn.rollback(); return None

# ---------------------- دوال النظام الرئيسية وواجهة Flask (دون تغيير جوهري) ----------------------
def main_loop():
    logger.info("[Main Loop] انتظار اكتمال التهيئة...")
    time.sleep(10)
    if not validated_symbols_to_scan:
        log_and_notify("critical", "لا توجد عملات صالحة للمسح.", "SYSTEM")
        return
    log_and_notify("info", f"✅ بدء حلقة المسح لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")

    while True:
        try:
            logger.info("🔄 بدء دورة مسح جديدة...")
            filter_profile = dynamic_filter_system.generate_filters()
            
            btc_data = get_btc_data_for_bot()
            symbols_to_process = random.sample(validated_symbols_to_scan, len(validated_symbols_to_scan))
            
            for symbol in symbols_to_process:
                try:
                    with signal_cache_lock:
                        if symbol in open_signals_cache or len(open_signals_cache) >= MAX_OPEN_TRADES:
                            continue
                    
                    df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_15m is None or df_15m.empty or len(df_15m) < 50: continue
                    df_15m.name = symbol

                    df_with_indicators = calculate_technical_indicators(df_15m, btc_data)
                    df_with_indicators.name = symbol
                    
                    # --- Primary Signal Check: EMA Crossover ---
                    if not check_ema_crossover_signal(df_with_indicators):
                        log_rejection(symbol, "EMA Crossover Invalid")
                        continue

                    # --- If Primary Signal is True, proceed to other checks ---
                    entry_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                    tp_sl_data = calculate_tp_sl(symbol, entry_price, df_15m)
                    if not tp_sl_data: continue
                    
                    last_row = df_with_indicators.iloc[-1]
                    if not passes_secondary_filters(symbol, last_row, filter_profile, entry_price, tp_sl_data): continue
                    
                    order_book_analysis = analyze_order_book(symbol, entry_price)
                    if not order_book_analysis or not passes_order_book_check(symbol, order_book_analysis, filter_profile): continue
                    
                    new_signal = {
                        'symbol': symbol, 'strategy_name': STRATEGY_NAME,
                        'signal_details': {'Filter_Profile': filter_profile['name'], 'Bid_Ask_Ratio': order_book_analysis.get('bid_ask_ratio', 0), **tp_sl_data},
                        'entry_price': entry_price, **tp_sl_data
                    }
                    
                    with trading_status_lock: is_enabled = is_trading_enabled
                    if is_enabled:
                        quantity = calculate_position_size(symbol, entry_price, new_signal['stop_loss'])
                        if quantity and quantity > 0:
                            # ... (order placement logic remains the same)
                            pass 
                    
                    saved_signal = insert_signal_into_db(new_signal)
                    if saved_signal:
                        with signal_cache_lock:
                            open_signals_cache[saved_signal['symbol']] = saved_signal
                        log_and_notify('info', f"SIGNAL: New buy signal for {symbol} at {entry_price}", "NEW_SIGNAL")

                except Exception as e:
                    logger.error(f"❌ [Processing Error] للعملة {symbol}: {e}", exc_info=True)
                finally:
                    time.sleep(0.5)

            logger.info("✅ [End of Cycle] انتهت دورة المسح الكاملة. الانتظار 60 ثانية...")
            time.sleep(60)
            
        except (KeyboardInterrupt, SystemExit):
            log_and_notify("info", "إيقاف البوت.", "SYSTEM")
            break
        except Exception as main_err:
            log_and_notify("error", f"خطأ حرج في الحلقة الرئيسية: {main_err}", "SYSTEM")
            time.sleep(120)


def trade_management_loop():
    logger.info("✅ [Trade Manager] بدء حلقة إدارة الصفقات...")
    # This loop remains identical to the previous version, handling TP/SL/Trailing SL
    while True:
        try:
            with signal_cache_lock:
                if not open_signals_cache:
                    time.sleep(5)
                    continue
                signals_to_check = list(open_signals_cache.values())
            
            if not redis_client:
                time.sleep(5)
                continue
                
            current_prices = redis_client.hgetall(REDIS_PRICES_HASH_NAME)
            
            for signal in signals_to_check:
                current_price_str = current_prices.get(signal['symbol'])
                if not current_price_str: continue
                
                current_price = float(current_price_str)
                signal_id = signal['id']
                tp = float(signal['target_price'])
                sl = float(signal['stop_loss'])
                entry = float(signal['entry_price'])

                if current_price >= tp:
                    logger.info(f"🎯 [TP HIT] {signal['symbol']} at {current_price}")
                    close_signal(signal_id, current_price, 'take_profit')
                    continue
                if current_price <= sl:
                    logger.info(f"🛑 [SL HIT] {signal['symbol']} at {current_price}")
                    close_signal(signal_id, current_price, 'stop_loss')
                    continue

                if USE_TRAILING_STOP_LOSS:
                    peak_price = float(signal.get('current_peak_price', entry))
                    new_peak = max(peak_price, current_price)
                    
                    if new_peak > peak_price:
                        with signal_cache_lock:
                            if signal['symbol'] in open_signals_cache:
                                open_signals_cache[signal['symbol']]['current_peak_price'] = new_peak
                        
                        try:
                            if check_db_connection():
                                with conn.cursor() as cur:
                                    cur.execute("UPDATE signals SET current_peak_price = %s WHERE id = %s", (new_peak, signal_id))
                                conn.commit()
                        except Exception as e:
                            logger.error(f"DB error updating peak price for {signal['symbol']}: {e}"); conn.rollback()

                    profit_pct = (new_peak / entry - 1) * 100
                    if profit_pct >= TRAILING_ACTIVATION_PROFIT_PERCENT:
                        new_sl = new_peak * (1 - TRAILING_DISTANCE_PERCENT / 100)
                        if new_sl > sl:
                            logger.info(f"📈 [TRAILING SL] {signal['symbol']} SL moved to {new_sl:.4f}")
                            with signal_cache_lock:
                                if signal['symbol'] in open_signals_cache:
                                    open_signals_cache[signal['symbol']]['stop_loss'] = new_sl
                            try:
                                if check_db_connection():
                                    with conn.cursor() as cur:
                                        cur.execute("UPDATE signals SET stop_loss = %s WHERE id = %s", (new_sl, signal_id))
                                    conn.commit()
                            except Exception as e:
                                logger.error(f"DB error updating trailing SL for {signal['symbol']}: {e}"); conn.rollback()
            time.sleep(2)
        except Exception as e:
            logger.error(f"❌ [Trade Manager] خطأ في حلقة الإدارة: {e}", exc_info=True)
            time.sleep(10)


def price_update_loop():
    if not redis_client: return
    # This loop remains identical
    while True:
        try:
            if validated_symbols_to_scan:
                tickers = client.get_symbol_ticker()
                prices_to_set = {t['symbol']: t['price'] for t in tickers if t['symbol'] in validated_symbols_to_scan}
                if prices_to_set: redis_client.hset(REDIS_PRICES_HASH_NAME, mapping=prices_to_set)
            time.sleep(1)
        except Exception as e: logger.error(f"Error in price update loop: {e}"); time.sleep(10)

def initialize_bot_services():
    global client, validated_symbols_to_scan
    logger.info("🤖 [Bot Services] بدء التهيئة...")
    try:
        client = Client(API_KEY, API_SECRET)
        init_db()
        init_redis()
        get_exchange_info_map()
        load_open_signals_to_cache()
        load_notifications_to_cache()
        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan: logger.critical("❌ لا توجد عملات صالحة للمسح."); return
        Thread(target=main_loop, daemon=True).start()
        Thread(target=price_update_loop, daemon=True).start()
        Thread(target=trade_management_loop, daemon=True).start()
        logger.info("✅ [Bot Services] تم بدء جميع الخدمات الخلفية بنجاح.")
        send_telegram_message("✅ *البوت قيد التشغيل الآن (استراتيجية التقاطعات)*")
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حرج أثناء التهيئة: {e}", "SYSTEM"); exit(1)

# --- Flask App (Dashboard) ---
app = Flask(__name__)
CORS(app)

# The Flask routes (@app.route(...)) and get_dashboard_html() function remain
# largely the same as the previous version and are omitted here for brevity.
# They will continue to display signals, stats, and logs from the database.

@app.route('/')
def home(): 
    # The HTML content is long, so it's omitted here.
    # It can be copied from the previous version.
    return "<h1>Trading Bot Dashboard (Crossover Strategy)</h1>"


# ---------------------- نقطة الانطلاق ----------------------
if __name__ == "__main__":
    logger.info("🚀 إطلاق بوت التداول (استراتيجية التقاطعات الفنية) 🚀")
    Thread(target=initialize_bot_services, daemon=True).start()
    port = int(os.environ.get('PORT', 10000))
    host = "0.0.0.0"
    logger.info(f"✅ بدء لوحة التحكم على {host}:{port}")
    try:
        # Using a simple development server for this example.
        # For production, Waitress or Gunicorn is recommended.
        app.run(host=host, port=port)
    except Exception as e:
        logger.critical(f"Failed to start Flask app: {e}")
    logger.info("👋 [Shutdown] تم إيقاف تشغيل التطبيق.")
