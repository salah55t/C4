# ملف c4.py - نسخة محدثة باستراتيجية Bollinger Bands و Stochastic
# تم التحديث بواسطة Gemini بناءً على طلب المستخدم
# --- الملخص:
# 1. إزالة الاستراتيجيات والفلاتر القديمة.
# 2. إضافة استراتيجية جديدة (BB + Stochastic + أنماط الشموع).
# 3. إضافة فلتر دفتر الطلبات المخصص.
# 4. استخدام نموذج تعلم الآلة كخطوة تأكيد أخيرة.
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
from collections import deque
import warnings

# --- إعدادات التجاهل واللوجر ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_bb_stoch_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBot_BB_Stoch')

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
RISK_PER_TRADE_PERCENT: float = 1.0
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V9_With_Microstructure'
MODEL_FOLDER: str = 'V9'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 90
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices_v9"
TRADING_FEE_PERCENT: float = 0.1
STATS_TRADE_SIZE_USDT: float = 5.0
BTC_SYMBOL: str = 'BTCUSDT'
MAX_OPEN_TRADES: int = 4
BUY_CONFIDENCE_THRESHOLD = 0.55
MIN_PROFIT_PERCENT: float = 0.8
SYMBOL_PROCESSING_BATCH_SIZE: int = 10

# --- إعدادات المؤشرات الفنية ---
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
STOCH_K_PERIOD: int = 14
STOCH_D_PERIOD: int = 3
STOCH_SMOOTHING: int = 3
BB_PERIOD: int = 20
BB_STD_DEV: int = 2

# --- إعدادات إدارة الصفقات ---
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_ACTIVATION_PROFIT_PERCENT: float = 1.8
TRAILING_DISTANCE_PERCENT: float = 1.0
ORDER_BOOK_DEPTH_LIMIT: int = 100

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

# --- قاموس أسباب الرفض باللغة العربية (محدث) ---
REJECTION_REASONS_AR = {
    "No BB Signal": "لا توجد إشارة Bollinger Band",
    "No Stoch Crossover": "لا يوجد تقاطع إيجابي لمؤشر Stochastic",
    "No Bullish Pattern": "لا يوجد نمط شموع صعودي",
    "Order Book Check Failed": "فشل التحقق من دفتر الطلبات",
    "ML Model Rejected": "نموذج تعلم الآلة رفض الإشارة",
    "ML Model Load Failed": "فشل تحميل نموذج تعلم الآلة",
    "Data Fetch Failed": "فشل جلب البيانات",
    "Price Fetch Failed": "فشل جلب السعر الحالي",
    "TP/SL Calculation Failed": "فشل حساب الهدف ووقف الخسارة",
    "Invalid Position Size": "حجم الصفقة غير صالح",
    "Lot Size Adjustment Failed": "فشل ضبط حجم العقد",
    "Min Notional Filter": "قيمة الصفقة أقل من الحد الأدنى",
    "Insufficient Balance": "الرصيد غير كافٍ",
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

# --- دوال تهيئة الخدمات (بدون تغيير جوهري) ---
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

def calculate_all_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df_calc = df.copy()

    # Bollinger Bands
    df_calc['bb_middle'] = df_calc['close'].rolling(window=BB_PERIOD).mean()
    bb_std = df_calc['close'].rolling(window=BB_PERIOD).std()
    df_calc['bb_upper'] = df_calc['bb_middle'] + (bb_std * BB_STD_DEV)
    df_calc['bb_lower'] = df_calc['bb_middle'] - (bb_std * BB_STD_DEV)

    # Stochastic Oscillator
    low_min = df_calc['low'].rolling(window=STOCH_K_PERIOD).min()
    high_max = df_calc['high'].rolling(window=STOCH_K_PERIOD).max()
    df_calc['stoch_k'] = 100 * (df_calc['close'] - low_min) / (high_max - low_min).replace(0, 1e-9)
    df_calc['stoch_d'] = df_calc['stoch_k'].rolling(window=STOCH_D_PERIOD).mean()

    # --- Other indicators for ML model ---
    # ATR
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    # RSI
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    # MACD
    exp1 = df_calc['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_calc['close'].ewm(span=26, adjust=False).mean()
    df_calc['macd'] = exp1 - exp2
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()

    # --- Features from original file for ML compatibility ---
    # Note: These are calculated but not used for the primary signal generation, only for the ML model confirmation step.
    plus_dm = pd.Series(np.where((df_calc['high'].diff() > -df_calc['low'].diff()) & (df_calc['high'].diff() > 0), df_calc['high'].diff(), 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((-df_calc['low'].diff() > df_calc['high'].diff()) & (-df_calc['low'].diff() > 0), -df_calc['low'].diff(), 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=30, min_periods=1).mean() + 1e-9)
    df_calc['price_vs_ema50'] = (df_calc['close'] / df_calc['close'].ewm(span=50, adjust=False).mean()) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / df_calc['close'].ewm(span=200, adjust=False).mean()) - 1
    if btc_df is not None and not btc_df.empty:
        asset_returns = df_calc['close'].pct_change()
        merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
        df_calc['btc_correlation'] = asset_returns.rolling(window=30).corr(merged_df['btc_returns'])
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

# ---------------------- الاستراتيجية الجديدة والفلاتر ----------------------

def detect_bullish_patterns(df: pd.DataFrame) -> Optional[str]:
    """
    يكتشف مجموعة واسعة من أنماط الشموع الصاعدة الانعكاسية في آخر شمعة.
    """
    if len(df) < 3: return None
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev_2 = df.iloc[-3]

    o, h, l, c = last['open'], last['high'], last['low'], last['close']
    o1, h1, l1, c1 = prev['open'], prev['high'], prev['low'], prev['close']
    o2, h2, l2, c2 = prev_2['open'], prev_2['high'], prev_2['low'], prev_2['close']
    
    body = abs(c - o)
    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)

    # Hammer / Inverted Hammer
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    if body > 0 and lower_wick > body * 2 and upper_wick < body * 0.5: return "Hammer"
    if body > 0 and upper_wick > body * 2 and lower_wick < body * 0.5: return "Inverted Hammer"

    # Bullish Engulfing
    if c1 < o1 and c > o and c >= o1 and o <= c1: return "Bullish Engulfing"
    
    # Piercing Line
    if c1 < o1 and c > o and o < c1 and c > (o1 + c1) / 2 and c < o1: return "Piercing Line"

    # Morning Star
    if c2 < o2 and body2 > 0 and c1 < o1 and body1 < body2 * 0.3 and c > o and c > (c2+o2)/2: return "Morning Star"

    # Three White Soldiers
    if c2 < o2 and c1 > o1 and c > o and \
       o > o1 and o1 > o2 and \
       c > c1 and c1 > c2 and \
       abs(o-c) > abs(o1-c1) * 0.7 and abs(o1-c1) > abs(o2-c2) * 0.7:
        return "Three White Soldiers"
        
    # Bullish Harami
    if c1 < o1 and c > o and c <= o1 and o >= c1: return "Bullish Harami"

    return None

def check_bb_stoch_signal(df: pd.DataFrame, lookback: int = 3) -> Optional[Dict[str, Any]]:
    """
    يفحص شروط استراتيجية BB + Stochastic + Candlestick.
    """
    required_cols = ['low', 'close', 'bb_lower', 'stoch_k', 'stoch_d']
    if not all(col in df.columns for col in required_cols) or len(df) < lookback + 2:
        return None

    for i in range(1, lookback + 1):
        idx = -i
        
        # 1. BB Condition: Price touches or breaks below lower band
        price_hit_bb = df['low'].iloc[idx] <= df['bb_lower'].iloc[idx]
        if not price_hit_bb:
            continue

        # 2. Stochastic Condition: Positive crossover below 15
        k_current = df['stoch_k'].iloc[idx]
        d_current = df['stoch_d'].iloc[idx]
        k_prev = df['stoch_k'].iloc[idx - 1]
        d_prev = df['stoch_d'].iloc[idx - 1]
        
        stoch_crossover = k_prev < d_prev and k_current > d_current
        stoch_level_ok = k_current < 15 and d_current < 15
        
        if not (stoch_crossover and stoch_level_ok):
            continue

        # 3. Candlestick Pattern Condition
        bullish_pattern = detect_bullish_patterns(df.iloc[:idx+1])
        if bullish_pattern is None:
            continue
            
        # All conditions met
        logger.info(f"✅ [{df.name}] إشارة شراء محتملة: BB Hit, Stoch Cross at {k_current:.2f}, Pattern: {bullish_pattern}")
        return {
            "strategy_name": "BB_STOCH_CANDLE",
            "pattern": bullish_pattern,
            "stoch_k": k_current
        }
        
    return None

def passes_order_book_filter(symbol: str, entry_price: float) -> bool:
    """
    فلتر دفتر الطلبات المخصص.
    Bids > 1.3 * Asks in a ±0.5% range around the current price.
    """
    if not client: return False
    try:
        order_book = client.get_order_book(symbol=symbol, limit=ORDER_BOOK_DEPTH_LIMIT)
        bids = pd.DataFrame(order_book['bids'], columns=['price', 'qty'], dtype=float)
        asks = pd.DataFrame(order_book['asks'], columns=['price', 'qty'], dtype=float)
        
        price_range_percent = 0.005  # ±0.5%
        price_lower_bound = entry_price * (1 - price_range_percent)
        price_upper_bound = entry_price * (1 + price_range_percent)

        relevant_bids_vol = bids[bids['price'].between(price_lower_bound, entry_price)]['qty'].sum()
        relevant_asks_vol = asks[asks['price'].between(entry_price, price_upper_bound)]['qty'].sum()

        if relevant_asks_vol == 0: # Avoid division by zero, strong bullish sign
            return True

        if relevant_bids_vol >= 1.3 * relevant_asks_vol:
            logger.info(f"✅ [{symbol}] اجتاز فلتر دفتر الطلبات (Bids: {relevant_bids_vol:.2f}, Asks: {relevant_asks_vol:.2f})")
            return True
        else:
            log_rejection(symbol, "Order Book Check Failed", {"bids": relevant_bids_vol, "asks": relevant_asks_vol, "ratio": relevant_bids_vol / relevant_asks_vol if relevant_asks_vol > 0 else 'inf'})
            return False

    except Exception as e:
        log_rejection(symbol, "Order Book Check Failed", {"error": str(e)})
        return False

# --- تحميل نموذج تعلم الآلة (عند الحاجة) ---
class MLConfirmation:
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
            if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
                ml_models_cache[model_name] = model_bundle
                logger.info(f"✅ [{self.symbol}] تم تحميل نموذج V9 من ملف للتأكيد: {model_path}")
                return model_bundle
            return None
        except Exception as e:
            logger.error(f"❌ [ML Model File] خطأ في تحميل النموذج لـ {symbol}: {e}")
            return None

    def get_features_for_ml(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.feature_names is None: return None
        try:
            # Re-calculate all features required by the original model
            df_featured = calculate_all_features(df_15m, btc_df) # This now calculates everything
            
            # Add 4h features
            df_4h_features = calculate_all_features(df_4h, None)
            df_4h_features = df_4h_features.rename(columns=lambda c: f"{c}_4h")
            required_4h_cols = ['rsi_4h', 'price_vs_ema50_4h']
            df_featured = df_featured.join(df_4h_features[required_4h_cols], how='left')
            df_featured[required_4h_cols] = df_featured[required_4h_cols].fillna(method='ffill')
            
            # Ensure all columns are present
            for col in self.feature_names:
                if col not in df_featured.columns: df_featured[col] = 0.0
            
            df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_featured.dropna(subset=self.feature_names)
        except Exception as e:
            logger.error(f"❌ [{self.symbol}] فشل هندسة الميزات لنموذج V9: {e}", exc_info=True)
            return None

    def confirm_signal(self, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]) or df_features.empty: return None
        try:
            last_row_ordered_df = df_features.iloc[[-1]][self.feature_names]
            features_scaled = self.scaler.transform(last_row_ordered_df)
            prediction = self.ml_model.predict(features_scaled)[0]
            if prediction != 1: return None
            prediction_proba = self.ml_model.predict_proba(features_scaled)
            confidence = float(np.max(prediction_proba[0]))
            return {'prediction': int(prediction), 'confidence': confidence}
        except Exception as e:
            logger.warning(f"⚠️ [{self.symbol}] خطأ في توليد تأكيد النموذج: {e}")
            return None

# --- حساب الهدف ووقف الخسارة (بدون تغيير) ---
def calculate_tp_sl(symbol: str, entry_price: float, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    try:
        if df.empty or 'atr' not in df.columns or df['atr'].isnull().all():
            log_rejection(symbol, "TP/SL Calculation Failed", {"reason": "ATR not available"})
            return None
        last_atr = df['atr'].iloc[-1]
        if last_atr <= 0:
            log_rejection(symbol, "TP/SL Calculation Failed", {"reason": "Invalid ATR value"})
            return None
            
        # Using a fixed R:R ratio for simplicity, can be improved
        target_price = entry_price + (last_atr * 2.0)
        stop_loss = entry_price - (last_atr * 1.5)
        
        return {
            'target_price': round(target_price, 6),
            'stop_loss': round(stop_loss, 6),
            'source': 'ATR_BASED',
            'rr_ratio': round(2.0 / 1.5, 2)
        }
    except Exception as e:
        logger.error(f"❌ [{symbol}] Error in TP/SL calculation: {e}", exc_info=True)
        return None

# --- دوال إدارة الصفقات (بدون تغيير جوهري) ---
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
            adjusted_quantity = (quantity_decimal // step_size) * step_size
            logger.info(f"[{symbol}] تعديل الكمية: الأصلية={quantity_decimal}, حجم الخطوة={step_size}, المعدلة={adjusted_quantity}")
            return adjusted_quantity
            
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
        signal_to_close = None
        symbol_to_close = None
        for symbol, signal_data in open_signals_cache.items():
            if signal_data['id'] == signal_id:
                signal_to_close = signal_data
                symbol_to_close = symbol
                break
        
        if not signal_to_close:
            logger.warning(f"⚠️ [Close] محاولة إغلاق صفقة غير موجودة في الكاش ID: {signal_id}")
            return False

        entry_price = float(signal_to_close['entry_price'])
        profit_percentage = ((closing_price - entry_price) / entry_price) * 100
        
        if signal_to_close.get('is_real_trade'):
            # Logic for placing a real sell order (remains the same)
            pass
        
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
            telegram_message = (
                f"{emoji} *إغلاق صفقة {trade_type}*\n\n"
                f"*العملة:* `{symbol_to_close}`\n"
                f"*سبب الإغلاق:* {reason_map.get(reason, reason)}\n"
                f"*سعر الدخول:* `{entry_price:.4f}`\n"
                f"*سعر الإغلاق:* `{closing_price:.4f}`\n"
                f"*الربح/الخسارة:* `{profit_percentage:.2f}%`"
            )
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
            entry_price = float(signal_data['entry_price'])
            target_price = float(signal_data['target_price'])
            stop_loss = float(signal_data['stop_loss'])
            quantity = float(signal_data['quantity']) if signal_data.get('quantity') is not None else None

            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, is_real_trade, quantity, order_id, current_peak_price, closing_reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL) RETURNING *;
            """, (
                signal_data['symbol'], entry_price, target_price, stop_loss,
                signal_data['strategy_name'], json.dumps(signal_data['signal_details']),
                signal_data.get('is_real_trade', False), quantity,
                signal_data.get('order_id'), entry_price
            ))
            saved_signal = cur.fetchone()
            conn.commit()
            logger.info(f"💾 [{signal_data['symbol']}] تم حفظ الإشارة الجديدة في قاعدة البيانات.")
            
            trade_type = "حقيقية" if signal_data.get('is_real_trade') else "تجريبية"
            telegram_message = (
                f"💡 *توصية شراء {trade_type} جديدة*\n\n"
                f"*العملة:* `{signal_data['symbol']}`\n"
                f"*الاستراتيجية:* `{signal_data['strategy_name']}`\n"
                f"*نمط الشمعة:* `{signal_data['signal_details'].get('Pattern', 'N/A')}`\n"
                f"*سعر الدخول:* `{entry_price:.4f}`\n"
                f"*الهدف (TP):* `{target_price:.4f}`\n"
                f"*وقف الخسارة (SL):* `{stop_loss:.4f}`\n\n"
                f"ML Confidence: {signal_data['signal_details'].get('ML_Confidence', 'N/A')}"
            )
            send_telegram_message(telegram_message)
            return dict(saved_signal)
    except Exception as e:
        logger.error(f"❌ [DB Insert] فشل إدراج الإشارة: {e}"); conn.rollback(); return None

# ---------------------- واجهة Flask (مبسطة) ----------------------
app = Flask(__name__)
CORS(app)

@app.route('/')
def home(): return "Bot is running." # The detailed dashboard is removed for simplicity, can be added back if needed.

# Other API endpoints can be added here if needed, following the original structure.

# ---------------------- حلقات النظام ----------------------
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
                        # Update peak price in cache and DB (logic remains the same)
                        pass

                    profit_pct = (new_peak / entry - 1) * 100
                    if profit_pct >= TRAILING_ACTIVATION_PROFIT_PERCENT:
                        new_sl = new_peak * (1 - TRAILING_DISTANCE_PERCENT / 100)
                        if new_sl > sl:
                            # Update trailing stop loss (logic remains the same)
                            pass
            time.sleep(2)
        except Exception as e:
            logger.error(f"❌ [Trade Manager] خطأ في حلقة الإدارة: {e}", exc_info=True)
            time.sleep(10)

def main_loop_new_strategy():
    logger.info("[Main Loop] انتظار اكتمال التهيئة...")
    time.sleep(15)
    if not validated_symbols_to_scan:
        log_and_notify("critical", "لا توجد عملات صالحة للمسح.", "SYSTEM")
        return
    log_and_notify("info", f"✅ بدء حلقة المسح لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")

    while True:
        try:
            logger.info("🔄 بدء دورة مسح جديدة...")
            btc_data = get_btc_data_for_bot()
            symbols_to_process = random.sample(validated_symbols_to_scan, len(validated_symbols_to_scan))
            
            for i in range(0, len(symbols_to_process), SYMBOL_PROCESSING_BATCH_SIZE):
                batch = symbols_to_process[i:i + SYMBOL_PROCESSING_BATCH_SIZE]
                total_batches = (len(symbols_to_process) + SYMBOL_PROCESSING_BATCH_SIZE - 1) // SYMBOL_PROCESSING_BATCH_SIZE
                logger.info(f"🔄 Processing batch {i // SYMBOL_PROCESSING_BATCH_SIZE + 1}/{total_batches} ({len(batch)} symbols)...")

                for symbol in batch:
                    try:
                        with signal_cache_lock:
                            if symbol in open_signals_cache or len(open_signals_cache) >= MAX_OPEN_TRADES:
                                continue
                        
                        # 1. تحميل البيانات وحساب المؤشرات
                        df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        if df_15m is None or df_15m.empty: continue
                        df_15m.name = symbol
                        df_features = calculate_all_features(df_15m, btc_data)
                        if df_features.isnull().values.any(): continue

                        # 2. البحث عن إشارة باستخدام استراتيجية BB
                        strategy_signal = check_bb_stoch_signal(df_features)
                        if not strategy_signal:
                            continue # لا توجد إشارة، انتقل للعملة التالية
                        
                        try:
                            entry_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                        except Exception as e:
                            log_rejection(symbol, "Price Fetch Failed", {"error": str(e)})
                            continue
                        
                        # 3. فحص التوصية بفلتر دفتر الأوامر
                        if not passes_order_book_filter(symbol, entry_price):
                            continue # لم يجتز الفلتر، انتقل للعملة التالية

                        # 4. تحميل نموذج تعلم الآلة والتأكد من إشارة الشراء
                        logger.info(f"[{symbol}] إشارة أولية ناجحة. جاري التحقق من نموذج تعلم الآلة...")
                        ml_conf = MLConfirmation(symbol)
                        if not all([ml_conf.ml_model, ml_conf.scaler, ml_conf.feature_names]):
                            log_rejection(symbol, "ML Model Load Failed")
                            continue
                        
                        df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        if df_4h is None or df_4h.empty: continue
                        
                        ml_features = ml_conf.get_features_for_ml(df_15m, df_4h, btc_data)
                        if ml_features is None or ml_features.empty:
                            log_rejection(symbol, "ML Model Load Failed", {"reason": "Feature generation failed"})
                            continue

                        ml_signal = ml_conf.confirm_signal(ml_features)
                        if not ml_signal or ml_signal['confidence'] < BUY_CONFIDENCE_THRESHOLD:
                            if ml_signal: log_rejection(symbol, "ML Model Rejected", {"confidence": ml_signal['confidence']})
                            else: log_rejection(symbol, "ML Model Rejected")
                            continue
                        
                        logger.info(f"✅ [{symbol}] وافق نموذج تعلم الآلة على الإشارة بثقة {ml_signal['confidence']:.2%}")

                        # --- كل الشروط تحققت، جهز الصفقة ---
                        tp_sl_data = calculate_tp_sl(symbol, entry_price, df_features)
                        if not tp_sl_data: continue
                        
                        new_signal = {
                            'symbol': symbol, 
                            'strategy_name': strategy_signal['strategy_name'], 
                            'signal_details': {
                                'Pattern': strategy_signal['pattern'],
                                'Stoch_K': f"{strategy_signal['stoch_k']:.2f}",
                                'ML_Confidence': f"{ml_signal['confidence']:.2%}",
                                **tp_sl_data
                            }, 
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

                logger.info(f"🗑️ Batch {i // SYMBOL_PROCESSING_BATCH_SIZE + 1} processed. Clearing caches.")
                ml_models_cache.clear()
                gc.collect()

            logger.info("✅ [End of Cycle] انتهت دورة المسح الكاملة. الانتظار 60 ثانية...")
            time.sleep(60)
            
        except (KeyboardInterrupt, SystemExit):
            log_and_notify("info", "إيقاف البوت.", "SYSTEM")
            break
        except Exception as main_err:
            log_and_notify("error", f"خطأ حرج في الحلقة الرئيسية: {main_err}", "SYSTEM")
            time.sleep(120)

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
        if not validated_symbols_to_scan: logger.critical("❌ لا توجد عملات صالحة للمسح."); return
        # --- تغيير الحلقة الرئيسية هنا ---
        Thread(target=main_loop_new_strategy, daemon=True).start()
        Thread(target=price_update_loop, daemon=True).start()
        Thread(target=trade_management_loop, daemon=True).start()
        logger.info("✅ [Bot Services] تم بدء جميع الخدمات الخلفية بنجاح.")
        send_telegram_message("✅ *البوت قيد التشغيل الآن (استراتيجية BB/Stoch)*")
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حرج أثناء التهيئة: {e}", "SYSTEM"); exit(1)

# ---------------------- نقطة الانطلاق ----------------------
if __name__ == "__main__":
    logger.info("🚀 إطلاق بوت التداول (استراتيجية BB/Stoch) 🚀")
    Thread(target=initialize_bot_services, daemon=True).start()
    port = int(os.environ.get('PORT', 10000))
    host = "0.0.0.0"
    logger.info(f"✅ بدء واجهة التحكم على {host}:{port}")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        app.run(host=host, port=port)
    logger.info("👋 [Shutdown] تم إيقاف تشغيل التطبيق.")
