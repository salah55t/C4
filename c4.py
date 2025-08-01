# ملف c4.py - نسخة معدلة مع استراتيجيات ديناميكية وإدارة مخاطر متقدمة
# تم التحديث بواسطة Gemini بناءً على طلب المستخدم
# --- التعديلات الرئيسية ---
# 1. أطوال EMA ديناميكية تتكيف مع تقلبات السوق (ATR).
# 2. فلتر وحيد يعتمد على دفتر الطلبات (Bids > 1.3 * Asks).
# 3. وقف خسارة ديناميكي يعتمد على ATR وحالة السوق.
# 4. جني أرباح جزئي عند 1:1 R/R وتحريك وقف الخسارة لنقطة الدخول.

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
from sklearn.preprocessing import StandardScaler
from collections import deque, Counter
import warnings

# --- إعدادات التجاهل واللوجر ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_dynamic_v2_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBot_DynamicV2')

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

# --- متغيرات عامة وإعدادات البوت (محدثة) ---
is_trading_enabled: bool = False
trading_status_lock = Lock()
# --- MODIFICATION: Disable all old filters as requested ---
are_filters_disabled: bool = True # Kept for UI consistency, but logic is now hardcoded to bypass old filters
filters_disabled_lock = Lock()
RISK_PER_TRADE_PERCENT: float = 1.0
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V9_With_Microstructure'
MODEL_FOLDER: str = 'V9'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 90
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices_v9_dynamic"
TRADING_FEE_PERCENT: float = 0.1
STATS_TRADE_SIZE_USDT: float = 5.0
BTC_SYMBOL: str = 'BTCUSDT'
MAX_OPEN_TRADES: int = 4
BUY_CONFIDENCE_THRESHOLD = 0.55

# --- NEW: Dynamic Risk Management Settings ---
PARTIAL_TP_CLOSE_PERCENT: float = 0.5 # Close 50% of the position at TP1
FINAL_TP_RR_RATIO: float = 2.0 # Final TP will be at 2:1 Risk/Reward
ATR_SL_MULTIPLIER_CALM: float = 1.2
ATR_SL_MULTIPLIER_TURBULENT: float = 1.5

# --- NEW: Order Book Filter Settings ---
ORDER_BOOK_FILTER_RATIO: float = 1.3 # Bids must be > 1.3 * Asks
ORDER_BOOK_PRICE_RANGE_PCT: float = 0.005 # ±0.5% around current price

# --- NEW: Dynamic EMA Settings ---
BASE_EMA_FAST: int = 9
BASE_EMA_SLOW: int = 21
VOLATILITY_LOOKBACK: int = 20 # Lookback for ATR normalization

# --- إعدادات المؤشرات الفنية (مطابقة لملف التدريب V9) ---
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
BTC_CORR_PERIOD: int = 30
MOMENTUM_PERIOD: int = 12

# --- متغيرات الحالة والكاش ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
ml_models_cache: Dict[str, Any] = {}
exchange_info_map: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
notifications_cache = deque(maxlen=50)
notifications_lock = Lock()
rejection_logs_cache = deque(maxlen=100)
rejection_logs_lock = Lock()
current_market_state: Dict[str, Any] = {"overall_regime": "INITIALIZING", "volatility_regime": "NORMAL", "trend_details_by_tf": {}, "last_updated": None}
market_state_lock = Lock()
last_market_state_check = 0

# --- قاموس أسباب الرفض باللغة العربية (مبسط) ---
REJECTION_REASONS_AR = {
    "ML Model Rejected Signal": "نموذج التعلم الآلي رفض الإشارة",
    "Invalid Position Size": "حجم الصفقة غير صالح",
    "Lot Size Adjustment Failed": "فشل ضبط حجم العقد",
    "Min Notional Filter": "قيمة الصفقة أقل من الحد الأدنى",
    "Insufficient Balance": "الرصيد غير كافٍ",
    "Order Book Filter Failed": "فشل فلتر دفتر الطلبات (Bids < 1.3 * Asks)",
    "Invalid TP/SL Calculation": "فشل حساب الأهداف ووقف الخسارة",
    "No Valid Technical Signal": "لم يتحقق أي شرط فني للدخول"
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
                # --- NEW: Updated table schema for partial TP ---
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        status TEXT DEFAULT 'open', -- open, partially_closed, closed
                        entry_price DOUBLE PRECISION NOT NULL,
                        stop_loss DOUBLE PRECISION NOT NULL,
                        tp1 DOUBLE PRECISION, -- Partial TP
                        tp2 DOUBLE PRECISION, -- Final TP
                        closing_price DOUBLE PRECISION,
                        closed_at TIMESTAMP,
                        profit_percentage DOUBLE PRECISION,
                        strategy_name TEXT,
                        signal_details JSONB,
                        is_real_trade BOOLEAN DEFAULT FALSE,
                        initial_quantity DOUBLE PRECISION,
                        current_quantity DOUBLE PRECISION,
                        order_id TEXT,
                        closing_reason TEXT
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
    log_message = f"🚫 [REJECTED] {symbol} | Reason: {reason_ar} | Details: {details or {}}"
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
        if not os.path.exists(file_path):
            logger.critical(f"❌ [Validation] ملف العملات '{filename}' غير موجود!")
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        if not exchange_info_map: get_exchange_info_map()
        active = {s for s, info in exchange_info_map.items() if info.get('quoteAsset') == 'USDT' and info.get('status') == 'TRADING'}
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ [Validation] تم العثور على {len(validated)} عملة صالحة للتداول من ملفك.")
        return validated
    except Exception as e:
        logger.error(f"❌ [Validation] خطأ أثناء التحقق من العملات: {e}", exc_info=True)
        return []

# --- دوال جلب البيانات وحساب الميزات (محدثة) ---
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        start_dt = datetime.now(timezone.utc) - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(klines, columns=cols)
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base']
        df = df[required_cols]
        numeric_cols = {'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float', 'quote_volume': 'float', 'taker_buy_base': 'float'}
        df = df.astype(numeric_cols)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna()
    except Exception as e:
        logger.error(f"❌ [Data] خطأ في جلب البيانات التاريخية لـ {symbol}: {e}")
        return None

def calculate_advanced_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    highest_high = df['high'].rolling(window=14).max()
    lowest_low = df['low'].rolling(window=14).min()
    df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low).replace(0, 1e-9)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    bb_period = 20
    df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
    bb_std = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    return df

# --- NEW: Dynamic EMA Calculation ---
def get_dynamic_ema_lengths(df: pd.DataFrame) -> Tuple[int, int]:
    """Calculates dynamic EMA lengths based on volatility (ATR)."""
    if 'atr' not in df.columns or df['atr'].isnull().all():
        return BASE_EMA_FAST, BASE_EMA_SLOW

    # Normalize ATR by its moving average to get a volatility factor
    atr_ma = df['atr'].rolling(window=VOLATILITY_LOOKBACK, min_periods=5).mean()
    volatility_factor = (df['atr'].iloc[-1] / atr_ma.iloc[-1]) if atr_ma.iloc[-1] > 0 else 1.0
    
    # Clamp the factor to avoid extreme values
    volatility_factor = np.clip(volatility_factor, 0.5, 2.0)

    # Adjust EMA lengths: higher volatility -> shorter EMA (more responsive)
    fast_ema_len = int(round(BASE_EMA_FAST / volatility_factor))
    slow_ema_len = int(round(BASE_EMA_SLOW / volatility_factor))

    # Ensure lengths are reasonable
    fast_ema_len = max(3, fast_ema_len)
    slow_ema_len = max(fast_ema_len + 5, slow_ema_len)

    return fast_ema_len, slow_ema_len

def calculate_all_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df_calc = df.copy()

    # --- ATR and RSI (needed for dynamic calculations) ---
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))

    # --- NEW: Calculate Dynamic EMAs ---
    fast_len, slow_len = get_dynamic_ema_lengths(df_calc)
    df_calc['ema_fast_dynamic'] = df_calc['close'].ewm(span=fast_len, adjust=False).mean()
    df_calc['ema_slow_dynamic'] = df_calc['close'].ewm(span=slow_len, adjust=False).mean()
    # Store the lengths used for this calculation for reference
    df_calc['dynamic_ema_fast_len'] = fast_len
    df_calc['dynamic_ema_slow_len'] = slow_len

    # --- Other features for ML model ---
    if btc_df is not None and not btc_df.empty:
        asset_returns = df_calc['close'].pct_change()
        merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
        df_calc['btc_correlation'] = asset_returns.rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    else:
        df_calc['btc_correlation'] = 0.0
    
    df_calc = calculate_advanced_momentum_features(df_calc)

    # --- Fill NaNs and convert types ---
    return df_calc.astype('float32', errors='ignore')

def get_btc_data_for_bot() -> Optional[pd.DataFrame]:
    btc_data = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
    if btc_data is not None: btc_data['btc_returns'] = btc_data['close'].pct_change()
    return btc_data

def load_open_signals_to_cache():
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status IN ('open', 'partially_closed');")
            open_signals = cur.fetchall()
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in open_signals: open_signals_cache[signal['symbol']] = dict(signal)
            logger.info(f"✅ [Loading] تم تحميل {len(open_signals)} صفقة مفتوحة أو مفتوحة جزئياً.")
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

# --- NEW: Modified Strategy and Filter Functions ---

# --- شموع مساعدة ---
def is_hammer(candle: pd.Series) -> bool:
    body = abs(candle['close'] - candle['open'])
    lower_wick = candle['open'] - candle['low'] if candle['close'] > candle['open'] else candle['close'] - candle['low']
    upper_wick = candle['high'] - candle['close'] if candle['close'] > candle['open'] else candle['high'] - candle['open']
    return body > 0 and lower_wick > 2 * body and upper_wick < body

def is_bullish_engulfing(current: pd.Series, previous: pd.Series) -> bool:
    return (current['close'] > current['open'] and previous['open'] > previous['close'] and
            current['close'] > previous['open'] and current['open'] < previous['close'])

def check_bb_stoch_reversal_strategy(df: pd.DataFrame) -> bool:
    required_cols = ['low', 'close', 'open', 'high', 'bb_lower', 'stoch_k', 'stoch_d']
    if not all(col in df.columns for col in required_cols) or len(df) < 2:
        return False
    try:
        last = df.iloc[-1]; prev = df.iloc[-2]
        stoch_crossed = prev['stoch_k'] <= prev['stoch_d'] and last['stoch_k'] > last['stoch_d']
        stoch_low = last['stoch_k'] < 20 and last['stoch_d'] < 20
        price_at_band = last['low'] <= last['bb_lower']
        candle_confirm = is_hammer(last) or is_bullish_engulfing(last, prev)
        if stoch_crossed and stoch_low and price_at_band and candle_confirm:
            logger.info(f"  -> [{df.name}] ✅ إشارة شراء من استراتيجية BB/Stoch Reversal.")
            return True
        return False
    except Exception: return False

def check_ema_stoch_momentum_strategy(df: pd.DataFrame) -> bool:
    required_cols = ['ema_fast_dynamic', 'ema_slow_dynamic', 'stoch_k', 'stoch_d']
    if not all(col in df.columns for col in required_cols) or len(df) < 2:
        return False
    try:
        last = df.iloc[-1]; prev = df.iloc[-2]
        ema_crossed = prev['ema_fast_dynamic'] <= prev['ema_slow_dynamic'] and last['ema_fast_dynamic'] > last['ema_slow_dynamic']
        stoch_confirm = last['stoch_k'] > last['stoch_d']
        if ema_crossed and stoch_confirm:
            logger.info(f"  -> [{df.name}] ✅ إشارة شراء من استراتيجية EMA الديناميكية (أطوال: {int(last['dynamic_ema_fast_len'])}/{int(last['dynamic_ema_slow_len'])}).")
            return True
        return False
    except Exception: return False

# --- NEW: The ONLY filter to be used ---
def passes_final_order_book_check(symbol: str, entry_price: float) -> bool:
    if not client: return False
    try:
        order_book = client.get_order_book(symbol=symbol, limit=100)
        bids = pd.DataFrame(order_book['bids'], columns=['price', 'qty'], dtype=float)
        asks = pd.DataFrame(order_book['asks'], columns=['price', 'qty'], dtype=float)
        
        price_limit_upper = entry_price * (1 + ORDER_BOOK_PRICE_RANGE_PCT)
        price_limit_lower = entry_price * (1 - ORDER_BOOK_PRICE_RANGE_PCT)

        relevant_bids_vol = bids[bids['price'] >= price_limit_lower]['qty'].sum()
        relevant_asks_vol = asks[asks['price'] <= price_limit_upper]['qty'].sum()

        if relevant_asks_vol == 0: return True # Avoid division by zero, bullish sign

        ratio = relevant_bids_vol / relevant_asks_vol
        
        if ratio > ORDER_BOOK_FILTER_RATIO:
            logger.info(f"  -> [{symbol}] ✅ فلتر دفتر الطلبات: نجح (Ratio: {ratio:.2f})")
            return True
        else:
            log_rejection(symbol, "Order Book Filter Failed", {"ratio": f"{ratio:.2f}"})
            return False
    except Exception as e:
        log_rejection(symbol, "Order Book Filter Failed", {"error": str(e)})
        return False

# --- NEW: Dynamic Risk Management TP/SL Calculation ---
def calculate_dynamic_tp_sl(symbol: str, entry_price: float, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    try:
        if df.empty or 'atr' not in df.columns:
            log_rejection(symbol, "Invalid TP/SL Calculation", {"reason": "No ATR data"})
            return None

        last_atr = df['atr'].iloc[-1]
        if not (last_atr > 0):
            log_rejection(symbol, "Invalid TP/SL Calculation", {"reason": "Invalid ATR value"})
            return None

        with market_state_lock:
            volatility_regime = current_market_state.get("volatility_regime", "NORMAL")

        sl_multiplier = ATR_SL_MULTIPLIER_TURBULENT if volatility_regime == "HIGH" else ATR_SL_MULTIPLIER_CALM
        
        stop_loss_price = entry_price - (last_atr * sl_multiplier)
        risk_per_share = entry_price - stop_loss_price

        if risk_per_share <= 0:
            log_rejection(symbol, "Invalid TP/SL Calculation", {"reason": "Risk is zero or negative"})
            return None

        # TP1 is at 1:1 Risk/Reward
        tp1_price = entry_price + risk_per_share
        
        # Final TP is at the specified R/R ratio
        tp2_price = entry_price + (risk_per_share * FINAL_TP_RR_RATIO)
        
        logger.info(f"  -> [{symbol}] أهداف محسوبة: SL={stop_loss_price:.4f}, TP1={tp1_price:.4f}, TP2={tp2_price:.4f} (ATR={last_atr:.4f}, Volatility={volatility_regime})")

        return {
            'stop_loss': round(stop_loss_price, 6),
            'tp1': round(tp1_price, 6),
            'tp2': round(tp2_price, 6),
            'source': f'DYNAMIC_ATR_{volatility_regime}'
        }
    except Exception as e:
        logger.error(f"❌ [{symbol}] Error in dynamic TP/SL: {e}", exc_info=True)
        return None

# --- دوال إدارة الصفقات (محدثة) ---
def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> Optional[Decimal]:
    try:
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info: return None
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = Decimal(lot_size_filter['stepSize'])
            return (Decimal(str(quantity)) // step_size) * step_size
        return Decimal(str(quantity))
    except Exception: return None

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

def close_signal(signal_id: int, closing_price: float, reason: str, is_partial_close: bool = False, remaining_quantity: float = 0.0) -> bool:
    with signal_cache_lock:
        signal_to_close = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)
        if not signal_to_close:
            logger.warning(f"⚠️ [Close] محاولة إغلاق صفقة غير موجودة في الكاش ID: {signal_id}")
            return False
        
        symbol = signal_to_close['symbol']
        entry_price = float(signal_to_close['entry_price'])
        profit_percentage = ((closing_price - entry_price) / entry_price) * 100

        if signal_to_close.get('is_real_trade'):
            try:
                quantity_to_sell_str = signal_to_close['current_quantity']
                if is_partial_close:
                    quantity_to_sell_str = float(signal_to_close['initial_quantity']) * PARTIAL_TP_CLOSE_PERCENT
                
                quantity_to_sell = adjust_quantity_to_lot_size(symbol, quantity_to_sell_str)
                if quantity_to_sell is None or quantity_to_sell <= 0:
                     logger.error(f"❌ [{symbol}] فشل تعديل كمية البيع أو الكمية صفر.")
                     return False

                sell_order = place_order(symbol, Client.SIDE_SELL, quantity_to_sell)
                if not sell_order:
                    log_and_notify('error', f"CRITICAL: Sell order placement failed for {symbol}. Trade remains open.", "TRADE_ERROR")
                    return False
            except Exception as e:
                logger.error(f"❌ [{symbol}] خطأ حرج أثناء تحضير أمر البيع: {e}", exc_info=True)
                return False

        if not check_db_connection() or not conn: return False

        try:
            with conn.cursor() as cur:
                if is_partial_close:
                    new_stop_loss = entry_price # Move SL to breakeven
                    cur.execute("""
                        UPDATE signals SET status = 'partially_closed', current_quantity = %s, stop_loss = %s
                        WHERE id = %s;
                    """, (remaining_quantity, new_stop_loss, signal_id))
                    
                    with signal_cache_lock:
                        if symbol in open_signals_cache:
                            open_signals_cache[symbol]['status'] = 'partially_closed'
                            open_signals_cache[symbol]['current_quantity'] = remaining_quantity
                            open_signals_cache[symbol]['stop_loss'] = new_stop_loss
                    
                    log_and_notify('info', f"PARTIAL CLOSE: {symbol} at {closing_price:.4f}. Reason: {reason}. SL moved to Breakeven.", "TRADE_PARTIAL_CLOSE")
                    telegram_message = (
                        f"💰 *جني أرباح جزئي*\n\n"
                        f"*العملة:* `{symbol}`\n"
                        f"*السبب:* {reason}\n"
                        f"*سعر الإغلاق الجزئي:* `{closing_price:.4f}`\n"
                        f"تم نقل وقف الخسارة إلى نقطة الدخول: `{entry_price:.4f}`"
                    )
                    send_telegram_message(telegram_message)
                else: # Full close
                    cur.execute("""
                        UPDATE signals SET status = 'closed', closing_price = %s, closed_at = NOW(),
                        profit_percentage = %s, closing_reason = %s, current_quantity = 0 WHERE id = %s;
                    """, (closing_price, profit_percentage, reason, signal_id))
                    
                    with signal_cache_lock:
                        if symbol in open_signals_cache:
                            del open_signals_cache[symbol]
                    
                    log_and_notify('info', f"CLOSED: {symbol} at {closing_price:.4f}. Reason: {reason}. Profit: {profit_percentage:.2f}%", "TRADE_CLOSED")
                    emoji = "✅" if profit_percentage >= 0 else "🔻"
                    trade_type = "حقيقية" if signal_to_close.get('is_real_trade') else "تجريبية"
                    telegram_message = (
                        f"{emoji} *إغلاق صفقة {trade_type}*\n\n"
                        f"*العملة:* `{symbol}`\n"
                        f"*سبب الإغلاق:* {reason}\n"
                        f"*سعر الدخول:* `{entry_price:.4f}`\n"
                        f"*سعر الإغلاق:* `{closing_price:.4f}`\n"
                        f"*الربح/الخسارة:* `{profit_percentage:.2f}%`"
                    )
                    send_telegram_message(telegram_message)
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"❌ [DB Close] فشل تحديث الصفقة: {e}"); conn.rollback(); return False

def insert_signal_into_db(signal_data: Dict) -> Optional[Dict]:
    if not check_db_connection() or not conn: return None
    try:
        with conn.cursor() as cur:
            quantity = float(signal_data['quantity']) if signal_data.get('quantity') is not None else None
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, stop_loss, tp1, tp2, strategy_name, signal_details, is_real_trade, initial_quantity, current_quantity, order_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *;
            """, (
                signal_data['symbol'], signal_data['entry_price'], signal_data['stop_loss'],
                signal_data.get('tp1'), signal_data.get('tp2'),
                signal_data['strategy_name'], json.dumps(signal_data['signal_details']),
                signal_data.get('is_real_trade', False), quantity, quantity,
                signal_data.get('order_id')
            ))
            saved_signal = cur.fetchone()
            conn.commit()
            logger.info(f"💾 [{signal_data['symbol']}] تم حفظ الإشارة الجديدة في قاعدة البيانات.")

            trade_type = "حقيقية" if signal_data.get('is_real_trade') else "تجريبية"
            telegram_message = (
                f"💡 *توصية شراء {trade_type} جديدة*\n\n"
                f"*العملة:* `{signal_data['symbol']}`\n"
                f"*الاستراتيجية:* `{signal_data['strategy_name']}`\n"
                f"*سعر الدخول:* `{signal_data['entry_price']:.4f}`\n"
                f"*الهدف الأول (TP1):* `{signal_data.get('tp1'):.4f}`\n"
                f"*الهدف النهائي (TP2):* `{signal_data.get('tp2'):.4f}`\n"
                f"*وقف الخسارة (SL):* `{signal_data['stop_loss']:.4f}`"
            )
            send_telegram_message(telegram_message)
            return dict(saved_signal)
    except Exception as e:
        logger.error(f"❌ [DB Insert] فشل إدراج الإشارة: {e}"); conn.rollback(); return None

# --- دوال النظام الرئيسية (محدثة) ---
def determine_market_state_enhanced():
    global current_market_state, last_market_state_check
    if time.time() - last_market_state_check < 180: return
    logger.info("🧠 [Market State] تحديث حالة السوق...")
    try:
        btc_data_1h = fetch_historical_data(BTC_SYMBOL, '1h', 30)
        if btc_data_1h is None: return
        
        # Volatility Regime
        btc_data_1h['atr'] = (btc_data_1h['high'] - btc_data_1h['low']).rolling(14).mean()
        volatility = btc_data_1h['atr'].iloc[-1] / btc_data_1h['close'].iloc[-1] * 100
        volatility_regime = "HIGH" if volatility > 0.5 else "NORMAL" # Simple threshold

        # Trend Regime
        trend_details = {}
        for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
            df = fetch_historical_data(BTC_SYMBOL, tf, 20)
            if df is not None and not df.empty:
                ema_fast = df['close'].ewm(span=12, adjust=False).mean().iloc[-1]
                ema_slow = df['close'].ewm(span=26, adjust=False).mean().iloc[-1]
                trend = "Uptrend" if ema_fast > ema_slow else "Downtrend"
                trend_details[tf] = {"trend": trend}
            else: trend_details[tf] = {"trend": "Uncertain"}
        
        trends = [d['trend'] for d in trend_details.values()]
        overall_regime = max(set(trends), key=trends.count) if trends else "Uncertain"
        
        with market_state_lock:
            current_market_state = {
                "overall_regime": overall_regime.upper().replace(" ", "_"),
                "volatility_regime": volatility_regime,
                "trend_details_by_tf": trend_details,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            last_market_state_check = time.time()
        logger.info(f"✅ [Market State] الحالة العامة: {overall_regime}, التقلب: {volatility_regime}")
    except Exception as e:
        logger.error(f"❌ [Market State] خطأ: {e}", exc_info=True)

# --- واجهة Flask (محدثة) ---
app = Flask(__name__)
CORS(app)

def get_dashboard_html():
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم التداول V10 (ديناميكي)</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
    <style>
        :root { --bg-main: #0D1117; --bg-card: #161B22; --border-color: #30363D; --text-primary: #E6EDF3; --text-secondary: #848D97; --accent-blue: #58A6FF; --accent-green: #3FB950; --accent-red: #F85149; --accent-yellow: #D29922; }
        body { font-family: 'Tajawal', sans-serif; background-color: var(--bg-main); color: var(--text-primary); }
        .card { background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 0.5rem; }
        .tab-btn.active { border-bottom-color: var(--accent-blue); }
        input:checked + .toggle-bg { background-color: var(--accent-green); }
    </style>
</head>
<body class="p-4 md:p-6">
    <div class="container mx-auto max-w-screen-2xl">
        <header class="mb-6 flex flex-wrap justify-between items-center gap-4">
            <h1 class="text-2xl md:text-3xl font-extrabold"><span class="text-accent-blue">لوحة تحكم ديناميكية</span></h1>
            <div id="status-container" class="flex items-center gap-x-6 bg-black/20 px-4 py-2 rounded-lg border border-border-color"></div>
        </header>
        <section class="mb-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">حالة السوق</h3><div id="overall-regime" class="text-2xl font-bold text-center">...</div></div>
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">مستوى التقلب</h3><div id="volatility-regime" class="text-2xl font-bold text-center">...</div></div>
            <div class="card p-4 flex flex-col justify-center items-center"><h3 class="font-bold text-lg text-text-secondary mb-2">التداول الحقيقي</h3><div class="flex items-center space-x-3 space-x-reverse"><span id="trading-status-text" class="font-bold text-lg"></span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="trading-toggle" class="sr-only" onchange="toggleTrading()"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div><div class="mt-2 text-xs text-text-secondary">رصيد USDT: <span id="usdt-balance" class="font-mono">...</span></div></div>
        </section>
        <div class="mb-4 border-b border-border-color"><nav class="flex space-x-6 space-x-reverse -mb-px"><button onclick="showTab('signals', this)" class="tab-btn active text-white py-3 px-1 font-semibold">الصفقات</button><button onclick="showTab('rejections', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الصفقات المرفوضة</button></nav></div>
        <main>
            <div id="signals-tab" class="tab-content"><div class="overflow-x-auto card p-0"><table class="min-w-full text-sm text-right"><thead class="border-b border-border-color bg-black/20"><tr><th class="p-4 font-semibold">العملة</th><th class="p-4 font-semibold">الحالة</th><th class="p-4 font-semibold">الربح/الخسارة</th><th class="p-4 font-semibold w-[25%]">التقدم</th><th class="p-4 font-semibold">الدخول/الحالي</th><th class="p-4 font-semibold">إجراء</th></tr></thead><tbody id="signals-table"></tbody></table></div></div>
            <div id="rejections-tab" class="tab-content hidden"><div id="rejections-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-2"></div></div>
        </main>
    </div>
<script>
function showTab(tabId, el) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.add('hidden'));
    document.getElementById(tabId + '-tab').classList.remove('hidden');
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active', 'text-white'));
    el.classList.add('active', 'text-white');
}
async function fetchData(url) { try { const r = await fetch(url); return r.ok ? await r.json() : null; } catch (e) { console.error('Fetch Error:', e); return null; } }
function updateStatus() {
    fetchData('/api/market_status').then(data => {
        if (!data) return;
        document.getElementById('overall-regime').textContent = (data.market_state?.overall_regime || 'UNCERTAIN').replace(/_/g, ' ');
        document.getElementById('volatility-regime').textContent = data.market_state?.volatility_regime || 'UNKNOWN';
        const tradeToggle = document.getElementById('trading-toggle'), tradeText = document.getElementById('trading-status-text');
        tradeToggle.checked = data.is_trading_enabled;
        tradeText.textContent = data.is_trading_enabled ? 'مُفعَّل' : 'غير مُفعَّل';
        tradeText.className = `font-bold text-lg ${data.is_trading_enabled ? 'text-accent-green' : 'text-accent-red'}`;
        document.getElementById('usdt-balance').textContent = data.usdt_balance ? parseFloat(data.usdt_balance).toFixed(2) : 'N/A';
    });
}
function updateSignals() {
    fetchData('/api/signals').then(data => {
        if (!data) return;
        const tableBody = document.getElementById('signals-table');
        tableBody.innerHTML = '';
        data.forEach(s => {
            const profit = parseFloat(s.profit_percentage || 0);
            const pClass = profit > 0 ? 'text-accent-green' : profit < 0 ? 'text-accent-red' : 'text-text-secondary';
            const entry = parseFloat(s.entry_price), sl = parseFloat(s.stop_loss), current = parseFloat(s.current_price || entry);
            let tp, progress_target;
            // Determine which TP to use for progress bar
            if (s.status === 'open') {
                tp = parseFloat(s.tp1);
                progress_target = 'TP1';
            } else { // partially_closed
                tp = parseFloat(s.tp2);
                progress_target = 'TP2';
            }
            const progress = (tp - sl > 0) ? Math.max(0, Math.min(100, (current - sl) / (tp - sl) * 100)) : 0;
            
            let statusBadge;
            if (s.status === 'open') {
                statusBadge = `<span class="px-2 py-1 text-xs font-semibold rounded-full bg-blue-500/20 text-blue-400">مفتوحة</span>`;
            } else if (s.status === 'partially_closed') {
                statusBadge = `<span class="px-2 py-1 text-xs font-semibold rounded-full bg-green-500/20 text-green-400">مؤمَّنة</span>`;
            }

            tableBody.innerHTML += `<tr class="border-b border-border-color hover:bg-white/5"><td class="p-4 font-bold">${s.symbol}</td><td class="p-4">${statusBadge}</td><td class="p-4 font-mono ${pClass}">${profit.toFixed(2)}%</td><td class="p-4"><div class="text-xs text-text-secondary mb-1">التقدم نحو ${progress_target}</div><div class="w-full bg-gray-700 rounded-full h-2.5"><div class="bg-accent-blue h-2.5 rounded-full" style="width: ${progress}%"></div></div></td><td class="p-4 font-mono">${current.toFixed(4)} / ${entry.toFixed(4)}</td><td class="p-4"><button onclick="manualClose(${s.id})" class="bg-red-600 hover:bg-red-700 text-white font-bold py-1 px-3 rounded text-xs">إغلاق</button></td></tr>`;
        });
    });
}
function updateRejections() {
    fetchData('/api/rejection_logs').then(data => {
        if (!data) return;
        document.getElementById('rejections-list').innerHTML = data.map(r => `<div class="p-2 border-b border-border-color"><span class="font-mono text-xs text-text-secondary">${new Date(r.timestamp).toLocaleString('ar-EG')}</span>: <strong class="text-accent-yellow">${r.symbol}</strong> - ${r.reason}</div>`).join('');
    });
}
function manualClose(signalId) {
    if (confirm('هل أنت متأكد من رغبتك في إغلاق الصفقة يدوياً؟')) {
        fetch(`/api/signals/close/${signalId}`, { method: 'POST' }).then(() => updateSignals());
    }
}
function toggleTrading() { fetch('/api/trading/toggle', { method: 'POST' }).then(() => updateStatus()); }
document.addEventListener('DOMContentLoaded', () => {
    ['Status', 'Signals', 'Rejections'].forEach(f => window[`update${f}`]());
    setInterval(updateStatus, 5000); setInterval(updateSignals, 7000); setInterval(updateRejections, 15000);
});
</script>
</body></html>
"""

@app.route('/')
def home(): return render_template_string(get_dashboard_html())

@app.route('/api/market_status')
def get_market_status():
    with market_state_lock: state_copy = dict(current_market_state)
    with trading_status_lock: is_enabled = is_trading_enabled
    usdt_balance = None
    if client:
        try: usdt_balance = float(client.get_asset_balance(asset='USDT')['free'])
        except: usdt_balance = 'N/A'
    return jsonify({
        "market_state": state_copy,
        "is_trading_enabled": is_enabled,
        "usdt_balance": usdt_balance
    })

@app.route('/api/signals')
def get_signals():
    if not all([check_db_connection(), redis_client]): return jsonify({"error": "Service connection failed"}), 500
    try:
        current_prices = redis_client.hgetall(REDIS_PRICES_HASH_NAME)
        with signal_cache_lock: signals_copy = list(open_signals_cache.values())
        for signal in signals_copy:
            current_price = current_prices.get(signal['symbol'])
            if current_price:
                signal['current_price'] = current_price
                signal['profit_percentage'] = ((float(current_price) - float(signal['entry_price'])) / float(signal['entry_price'])) * 100
        return jsonify(signals_copy)
    except Exception as e:
        logger.error(f"❌ [API Signals] Error: {e}"); return jsonify({"error": str(e)}), 500

@app.route('/api/rejection_logs')
def get_rejection_logs():
    with rejection_logs_lock:
        return jsonify(list(rejection_logs_cache))

@app.route('/api/trading/toggle', methods=['POST'])
def toggle_trading_status():
    global is_trading_enabled
    with trading_status_lock:
        is_trading_enabled = not is_trading_enabled
        status_msg = "ENABLED" if is_trading_enabled else "DISABLED"
        log_and_notify('warning', f"🚨 Real trading status changed to: {status_msg}", "TRADING_STATUS_CHANGE")
        return jsonify({"message": f"Trading status set to {status_msg}"})

@app.route('/api/signals/close/<int:signal_id>', methods=['POST'])
def manual_close_trade_endpoint(signal_id):
    if not redis_client or not client: return jsonify({"success": False, "message": "Services not ready"}), 503
    with signal_cache_lock:
        signal = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)
    if not signal: return jsonify({"success": False, "message": "Signal not found"}), 404
    try:
        price = float(redis_client.hget(REDIS_PRICES_HASH_NAME, signal['symbol']))
    except:
        price = float(client.get_symbol_ticker(symbol=signal['symbol'])['price'])
    
    if close_signal(signal_id, price, 'manual_full_close'):
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "message": "Failed to close signal"}), 500

# --- حلقات النظام (محدثة) ---
def trade_management_loop():
    logger.info("✅ [Trade Manager] بدء حلقة إدارة الصفقات...")
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
                status = signal['status']
                
                # 1. Check Stop Loss (for all open/partially_closed trades)
                sl = float(signal['stop_loss'])
                if current_price <= sl:
                    reason = 'stop_loss_hit' if status == 'open' else 'breakeven_stop_hit'
                    logger.info(f"🛑 [{reason.upper()}] {signal['symbol']} at {current_price}")
                    close_signal(signal_id, current_price, reason)
                    continue

                # 2. Check for Partial Take Profit (TP1)
                if status == 'open':
                    tp1 = float(signal['tp1'])
                    if current_price >= tp1:
                        logger.info(f"💰 [TP1 HIT] {signal['symbol']} at {current_price}")
                        initial_qty = float(signal['initial_quantity'])
                        remaining_qty = initial_qty * (1 - PARTIAL_TP_CLOSE_PERCENT)
                        close_signal(signal_id, current_price, 'partial_take_profit', is_partial_close=True, remaining_quantity=remaining_qty)
                        continue

                # 3. Check for Final Take Profit (TP2)
                if status == 'partially_closed':
                    tp2 = float(signal['tp2'])
                    if current_price >= tp2:
                        logger.info(f"🎯 [TP2 HIT] {signal['symbol']} at {current_price}")
                        close_signal(signal_id, current_price, 'final_take_profit')
                        continue

            time.sleep(2)
        except Exception as e:
            logger.error(f"❌ [Trade Manager] خطأ في حلقة الإدارة: {e}", exc_info=True)
            time.sleep(10)

def main_loop_enhanced():
    logger.info("[Main Loop] انتظار اكتمال التهيئة...")
    time.sleep(15)
    if not validated_symbols_to_scan:
        log_and_notify("critical", "قائمة العملات للمسح فارغة.", "SYSTEM_ERROR")
        return
    log_and_notify("info", f"✅ بدء حلقة المسح لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")

    while True:
        try:
            logger.info("🔄 [Main Loop] بدء دورة مسح جديدة...")
            determine_market_state_enhanced()
            btc_data = get_btc_data_for_bot()
            
            for symbol in random.sample(validated_symbols_to_scan, len(validated_symbols_to_scan)):
                try:
                    logger.info(f"---===[ 🔍 تحليل {symbol} ]===---")
                    with signal_cache_lock:
                        if symbol in open_signals_cache or len(open_signals_cache) >= MAX_OPEN_TRADES:
                            continue
                    
                    strategy = EnhancedTradingStrategy(symbol) # Helper class from old code
                    if not strategy.ml_model: continue

                    df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_15m is None or df_15m.empty: continue

                    df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_4h is None or df_4h.empty: continue
                    
                    df_features = strategy.get_features(df_15m, df_4h, btc_data)
                    if df_features is None or df_features.empty: continue
                    df_features.name = symbol

                    # --- NEW DYNAMIC STRATEGY LOGIC ---
                    strategy_signal_found = False
                    strategy_name = None
                    if check_ema_stoch_momentum_strategy(df_features):
                        strategy_signal_found = True
                        strategy_name = "Dynamic_EMA_Momentum"
                    elif check_bb_stoch_reversal_strategy(df_features):
                        strategy_signal_found = True
                        strategy_name = "BB_Stoch_Reversal"
                    
                    if not strategy_signal_found:
                        log_rejection(symbol, "No Valid Technical Signal")
                        continue
                    
                    ml_signal = strategy.generate_buy_signal(df_features)
                    if not ml_signal or ml_signal['confidence'] < BUY_CONFIDENCE_THRESHOLD:
                        log_rejection(symbol, "ML Model Rejected Signal", {"confidence": ml_signal['confidence'] if ml_signal else 'N/A'})
                        continue
                    
                    logger.info(f"  -> [{symbol}] ✅ النموذج يؤكد الإشارة (Confidence: {ml_signal['confidence']:.2%}).")
                    
                    try:
                        entry_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                    except Exception as e:
                        logger.error(f"❌ [{symbol}] فشل جلب سعر الدخول: {e}."); continue
                    
                    # --- NEW: Apply the ONLY filter ---
                    if not passes_final_order_book_check(symbol, entry_price):
                        continue
                    
                    tp_sl_data = calculate_dynamic_tp_sl(symbol, entry_price, df_features)
                    if not tp_sl_data: continue

                    new_signal = {
                        'symbol': symbol, 'strategy_name': strategy_name,
                        'signal_details': {'ML_Confidence': f"{ml_signal['confidence']:.2%}"},
                        'entry_price': entry_price, **tp_sl_data
                    }

                    with trading_status_lock: is_enabled = is_trading_enabled
                    if is_enabled:
                        quantity = calculate_position_size(symbol, entry_price, new_signal['stop_loss'])
                        if quantity and quantity > 0:
                            order_result = place_order(symbol, Client.SIDE_BUY, quantity)
                            if order_result:
                                new_signal.update({'is_real_trade': True, 'quantity': float(quantity), 'order_id': order_result['orderId']})
                            else: continue
                        else: continue

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
            log_and_notify("info", "إيقاف البوت.", "SYSTEM"); break
        except Exception as main_err:
            log_and_notify("error", f"خطأ حرج في الحلقة الرئيسية: {main_err}", "SYSTEM"); time.sleep(120)

# --- Helper classes and functions from old code needed for ML ---
class EnhancedTradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = self._load_ml_model_from_file(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    def _load_ml_model_from_file(self, symbol: str) -> Optional[Dict[str, Any]]:
        model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
        if model_name in ml_models_cache: return ml_models_cache[model_name]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir_path = os.path.join(script_dir, MODEL_FOLDER)
        model_path = os.path.join(model_dir_path, f"{model_name}.pkl")
        if not os.path.exists(model_path): return None
        try:
            with open(model_path, 'rb') as f: model_bundle = pickle.load(f)
            ml_models_cache[model_name] = model_bundle
            return model_bundle
        except Exception: return None

    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.feature_names is None: return None
        try:
            df_featured = calculate_all_features_for_ml(df_15m, btc_df) # Use a dedicated feature function for ML
            df_4h_features = calculate_all_features_for_ml(df_4h, None).rename(columns=lambda c: f"{c}_4h")
            df_featured = df_featured.join(df_4h_features[['rsi_4h']], how='left').fillna(method='ffill')
            for col in self.feature_names:
                if col not in df_featured.columns: df_featured[col] = 0.0
            return df_featured.dropna(subset=self.feature_names)
        except Exception: return None

    def generate_buy_signal(self, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]) or df_features.empty: return None
        try:
            last_row = df_features.iloc[[-1]][self.feature_names]
            features_scaled = self.scaler.transform(last_row)
            prediction = self.ml_model.predict(features_scaled)[0]
            if prediction != 1: return None
            confidence = float(np.max(self.ml_model.predict_proba(features_scaled)[0]))
            return {'prediction': int(prediction), 'confidence': confidence}
        except Exception: return None

def calculate_all_features_for_ml(df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    # This is a simplified version of the old feature calculation,
    # just to ensure the ML model gets the features it was trained on.
    # The dynamic EMAs are NOT used here, only for the strategy signal itself.
    df_calc = df.copy()
    df_calc['ema_9'] = df_calc['close'].ewm(span=9, adjust=False).mean()
    df_calc['ema_21'] = df_calc['close'].ewm(span=21, adjust=False).mean()
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    if btc_df is not None and not btc_df.empty:
        asset_returns = df_calc['close'].pct_change()
        merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
        df_calc['btc_correlation'] = asset_returns.rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    else:
        df_calc['btc_correlation'] = 0.0
    # Add other necessary features for the model if they are missing
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=30, min_periods=1).mean() + 1e-9)
    df_calc['price_vs_ema50'] = (df_calc['close'] / df_calc['close'].ewm(span=50, adjust=False).mean()) - 1
    return df_calc

def price_update_loop():
    if not redis_client: return
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
        Thread(target=main_loop_enhanced, daemon=True).start()
        Thread(target=price_update_loop, daemon=True).start()
        Thread(target=trade_management_loop, daemon=True).start()
        logger.info("✅ [Bot Services] تم بدء جميع الخدمات الخلفية بنجاح.")
        send_telegram_message("✅ *البوت الديناميكي قيد التشغيل الآن*")
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حرج أثناء التهيئة: {e}", "SYSTEM"); exit(1)

# ---------------------- نقطة الانطلاق ----------------------
if __name__ == "__main__":
    logger.info("🚀 إطلاق بوت التداول الديناميكي V10 🚀")
    Thread(target=initialize_bot_services, daemon=True).start()
    port = int(os.environ.get('PORT', 10000))
    host = "0.0.0.0"
    logger.info(f"✅ بدء لوحة التحكم على {host}:{port}")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        app.run(host=host, port=port)
    logger.info("👋 [Shutdown] تم إيقاف تشغيل التطبيق.")
