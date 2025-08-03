# ملف c4.py - نسخة V5 مع خط تجميع إشارة واضح
# تم التحديث بواسطة Gemini لتوضيح تطبيق الفلاتر النهائية على جميع الاستراتيجيات
# --- التغييرات الرئيسية (V5):
# 1. (هيكلة جديدة) إعادة هيكلة الحلقة الرئيسية لتتبع "خط تجميع إشارة" (Signal Pipeline) واضح ومكون من 4 مراحل.
# 2. (تأكيد منطقي) التأكيد على أن مخرجات كلتا الاستراتيجيتين (الانعكاس ومتابعة الاتجاه) تمر بنفس فلاتر الجودة (النموذج الآلي ودفتر الطلبات).
# 3. (تحسين التسجيل) إضافة رسائل سجل أكثر تفصيلاً لتتبع تقدم الإشارة عبر كل مرحلة من مراحل التحقق.

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
        logging.FileHandler('crypto_bot_v10_dynamic_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV10_Dynamic')

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
are_filters_disabled: bool = True
filters_disabled_lock = Lock()
RISK_PER_TRADE_PERCENT: float = 1.0
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V9_With_Microstructure'
MODEL_FOLDER: str = 'V9'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 90
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices_v10"
TRADING_FEE_PERCENT: float = 0.1
STATS_TRADE_SIZE_USDT: float = 5.0
BTC_SYMBOL: str = 'BTCUSDT'
MAX_OPEN_TRADES: int = 4
BUY_CONFIDENCE_THRESHOLD = 0.55
MIN_PROFIT_PERCENT: float = 0.8
SYMBOL_PROCESSING_BATCH_SIZE: int = 10

# --- إعدادات المؤشرات الفنية ---
EMA_FAST_PERIOD: int = 50
EMA_SLOW_PERIOD: int = 120
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
BTC_CORR_PERIOD: int = 30
REL_VOL_PERIOD: int = 30
MOMENTUM_PERIOD: int = 12
EMA_SLOPE_PERIOD: int = 5
SUPERTREND_ATR_PERIOD: int = 10
SUPERTREND_MULTIPLIER: float = 3.0

# --- إعدادات الفلاتر المتقدمة وإدارة الصفقات ---
ORDER_BOOK_DEPTH_LIMIT: int = 100
ORDER_BOOK_ANALYSIS_RANGE_PCT: float = 0.005 # ±0.5%
ORDER_BOOK_MIN_BID_ASK_RATIO: float = 1.3 # Bids must be > 30% of Asks
USE_ATR_TRAILING_STOP: bool = True
ATR_TS_PERIOD: int = 14
ATR_TS_MULTIPLIER: float = 2.5

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
current_market_state: Dict[str, Any] = {"overall_regime": "INITIALIZING", "trend_details_by_tf": {}, "last_updated": None}
market_state_lock = Lock()
last_market_state_check = 0
technical_signals_cache: Dict[str, Dict] = {}
TECHNICAL_SIGNAL_CACHE_DURATION: int = 60 * 5

# --- قاموس أسباب الرفض باللغة العربية (موسع للاستراتيجيتين) ---
REJECTION_REASONS_AR = {
    # أسباب عامة
    "Strategy Signal Not Found": "لم تتحقق شروط أي استراتيجية",
    "ML Model Rejected Signal": "نموذج التعلم الآلي رفض الإشارة",
    "ML Model Load Failed": "فشل تحميل نموذج التعلم الآلي",
    "Order Book Filter Failed": "فشل فلتر دفتر الطلبات (Bids/Asks)",
    "Invalid Position Size": "حجم الصفقة غير صالح",
    "Lot Size Adjustment Failed": "فشل ضبط حجم العقد",
    "Min Notional Filter": "قيمة الصفقة أقل من الحد الأدنى",
    "Insufficient Balance": "الرصيد غير كافٍ",
    "Order Book Fetch Failed": "فشل جلب دفتر الطلبات",
    "Insufficient data for TP/SL calculation": "بيانات غير كافية لحساب TP/SL",
    "Invalid Market Regime for Any Strategy": "حالة السوق غير مناسبة لأي استراتيجية متاحة",

    # أسباب استراتيجية الانعكاس (Divergence)
    "No Bullish Divergence": "لم يتم العثور على انفراج إيجابي",
    "Divergence not from Oversold": "الانفراج لم يبدأ من منطقة ذروة بيع",
    "No Stochastic Crossover": "لم يحدث تقاطع إيجابي لمؤشر ستوكاستيك",
    
    # أسباب استراتيجية متابعة الاتجاه (Pullback)
    "Not an Uptrend": "السوق ليس في اتجاه صاعد",
    "No Pullback to EMA": "لم يحدث تصحيح لمنطقة الدعم (EMA)",
    "RSI below threshold": "مؤشر القوة النسبية ضعيف",

    # أسباب مشتركة
    "No Confirmation Candle": "لم تظهر شمعة تأكيد صاعدة قوية",
    "Low Volume on Signal Candle": "حجم تداول ضعيف على شمعة الإشارة",
    "Low Buy Pressure": "ضغط الشراء ضعيف",
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
            logger.critical(f"❌ [Validation] ملف العملات '{filename}' غير موجود! يرجى إنشاء الملف وإضافة رموز العملات.")
            return []
            
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}

        if not raw_symbols:
            logger.warning(f"⚠️ [Validation] ملف العملات '{filename}' فارغ. لن يتم مسح أي عملات.")
            return []

        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        if not exchange_info_map: get_exchange_info_map()
        active = {s for s, info in exchange_info_map.items() if info.get('quoteAsset') == 'USDT' and info.get('status') == 'TRADING'}
        validated = sorted(list(formatted.intersection(active)))
        
        logger.info(f"✅ [Validation] تم العثور على {len(validated)} عملة صالحة للتداول من ملفك.")
        if not validated:
             logger.warning(f"⚠️ [Validation] لم تتطابق أي من العملات في ملفك مع العملات المتاحة للتداول على Binance.")
        else:
            logger.info(f"🔍 [Validation] عينة من العملات التي ستتم مراقبتها: {validated[:5]}")

        return validated
    except Exception as e:
        logger.error(f"❌ [Validation] خطأ أثناء التحقق من العملات: {e}", exc_info=True)
        return []

# --- دوال جلب البيانات وحساب الميزات (محدثة لـ V10) ---
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
    df['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low).replace(0, 1e-9)
    df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low).replace(0, 1e-9)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    bb_period = 20
    df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
    bb_std = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, 1e-9)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    money_ratio = positive_flow / negative_flow.replace(0, 1e-9)
    df['mfi'] = 100 - (100 / (1 + money_ratio))
    return df

def calculate_market_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ['taker_buy_base', 'volume', 'quote_volume', 'high', 'low', 'open', 'close']
    if not all(col in df.columns for col in required_cols): return df
    df['buy_pressure'] = df['taker_buy_base'] / df['volume'].replace(0, 1e-9)
    volume_ma = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / volume_ma.replace(0, 1e-9)
    df['price_impact'] = df['quote_volume'] / df['volume'].replace(0, 1e-9)
    log_hl = np.log(df['high'] / df['low'].replace(0, 1e-9))
    log_co = np.log(df['close'] / df['open'].replace(0, 1e-9))
    gk_vol_sq = (0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)).clip(lower=0)
    df['garman_klass_vol'] = np.sqrt(gk_vol_sq)
    log_hc = np.log(df['high'] / df['close'].replace(0, 1e-9))
    log_ho = np.log(df['high'] / df['open'].replace(0, 1e-9))
    log_lc = np.log(df['low'] / df['close'].replace(0, 1e-9))
    log_lo = np.log(df['low'] / df['open'].replace(0, 1e-9))
    rs_vol_sq = (log_hc * log_ho + log_lc * log_lo).clip(lower=0)
    df['rogers_satchell_vol'] = np.sqrt(rs_vol_sq)
    return df

def calculate_advanced_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    high_low = df['high'] - df['low']
    ema_high_low = high_low.ewm(span=10, adjust=False).mean()
    ema_high_low_shifted = ema_high_low.shift(10)
    df['chaikin_volatility'] = (ema_high_low - ema_high_low_shifted) / ema_high_low_shifted.replace(0, 1e-9) * 100
    period = 14
    max_close = df['close'].rolling(window=period).max()
    percentage_drawdown = 100 * (df['close'] - max_close) / max_close.replace(0, 1e-9)
    df['ulcer_index'] = np.sqrt((percentage_drawdown ** 2).rolling(window=period).mean())
    if 'atr' not in df.columns: return df
    high_low_tr = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift()).abs()
    low_close_prev = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low_tr, high_close_prev, low_close_prev], axis=1).max(axis=1)
    for p in [5, 10, 20]:
        atr_p = tr.ewm(span=p, adjust=False).mean()
        df[f'atr_ratio_{p}'] = df['atr'] / atr_p.replace(0, 1e-9)
    return df

def calculate_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['asia_session'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
    df['london_session'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
    df['ny_session'] = ((df.index.hour >= 13) & (df.index.hour < 21)).astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    return df

def calculate_supertrend(df: pd.DataFrame, atr_period: int, multiplier: float) -> pd.DataFrame:
    high = df['high']
    low = df['low']
    close = df['close']
    high_low = high - low
    high_close_prev = np.abs(high - close.shift(1))
    low_close_prev = np.abs(low - close.shift(1))
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = tr.ewm(com=atr_period - 1, min_periods=atr_period, adjust=False).mean()
    hl2 = (high + low) / 2
    final_upper_band = upper_band = hl2 + (multiplier * atr)
    final_lower_band = lower_band = hl2 - (multiplier * atr)
    supertrend = pd.Series(np.nan, index=df.index)
    supertrend_direction = pd.Series(np.nan, index=df.index)
    for i in range(1, len(df)):
        curr, prev = i, i - 1
        if close[curr] > final_upper_band[prev]:
            supertrend_direction[curr] = 1
        elif close[curr] < final_lower_band[prev]:
            supertrend_direction[curr] = -1
        else:
            supertrend_direction[curr] = supertrend_direction[prev]
            if supertrend_direction[curr] == -1 and final_upper_band[curr] < final_upper_band[prev]:
                final_upper_band[curr] = final_upper_band[curr]
            if supertrend_direction[curr] == 1 and final_lower_band[curr] > final_lower_band[prev]:
                final_lower_band[curr] = final_lower_band[curr]
        if supertrend_direction[curr] == 1:
            supertrend[curr] = final_lower_band[curr]
        else:
            supertrend[curr] = final_upper_band[curr]
    df['supertrend'] = supertrend
    df['supertrend_direction'] = supertrend_direction
    return df

def calculate_all_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df_calc = df.copy()
    df_calc['ema_9'] = df_calc['close'].ewm(span=9, adjust=False).mean()
    df_calc['ema_21'] = df_calc['close'].ewm(span=21, adjust=False).mean()
    df_calc['sma_50'] = df_calc['close'].rolling(window=50).mean()
    df_calc['sma_200'] = df_calc['close'].rolling(window=200).mean()
    df_calc['volume_sma_20'] = df_calc['volume'].rolling(window=20).mean()
    df_calc['ema_50'] = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    df_calc['ema_120'] = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
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
    STOCH_RSI_PERIOD = 14
    STOCH_RSI_K_PERIOD = 3
    STOCH_RSI_D_PERIOD = 3
    rsi = df_calc['rsi']
    stoch_rsi_val = (rsi - rsi.rolling(STOCH_RSI_PERIOD).min()) / (rsi.rolling(STOCH_RSI_PERIOD).max() - rsi.rolling(STOCH_RSI_PERIOD).min()).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(STOCH_RSI_K_PERIOD).mean() * 100
    df_calc['stoch_rsi_d'] = df_calc['stoch_rsi_k'].rolling(STOCH_RSI_D_PERIOD).mean()
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)
    df_calc['price_vs_ema50'] = (df_calc['close'] / df_calc['ema_50']) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / df_calc['close'].ewm(span=200, adjust=False).mean()) - 1
    if btc_df is not None and not btc_df.empty:
        asset_returns = df_calc['close'].pct_change()
        merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
        df_calc['btc_correlation'] = asset_returns.rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    else:
        df_calc['btc_correlation'] = 0.0
    df_calc = calculate_advanced_momentum_features(df_calc)
    df_calc = calculate_market_microstructure_features(df_calc)
    df_calc = calculate_advanced_volatility_features(df_calc)
    df_calc = calculate_temporal_features(df_calc)
    df_calc = calculate_supertrend(df_calc, SUPERTREND_ATR_PERIOD, SUPERTREND_MULTIPLIER)
    df_calc[f'roc_{MOMENTUM_PERIOD}'] = (df_calc['close'] / df_calc['close'].shift(MOMENTUM_PERIOD) - 1) * 100
    df_calc['roc_acceleration'] = df_calc[f'roc_{MOMENTUM_PERIOD}'].diff()
    ema_slope = df_calc['close'].ewm(span=EMA_SLOPE_PERIOD, adjust=False).mean()
    df_calc[f'ema_slope_{EMA_SLOPE_PERIOD}'] = (ema_slope - ema_slope.shift(1)) / ema_slope.shift(1).replace(0, 1e-9) * 100
    return df_calc.astype('float32', errors='ignore')

def get_session_state() -> Tuple[List[str], str, str]:
    sessions = {"London": (8, 17), "New York": (13, 22), "Tokyo": (0, 9)}
    active_sessions = []
    now_utc = datetime.now(timezone.utc)
    current_hour = now_utc.hour
    if now_utc.weekday() >= 5: return [], "WEEKEND", "عطلة نهاية الأسبوع"
    for session, (start, end) in sessions.items():
        if start <= current_hour < end: active_sessions.append(session)
    if "London" in active_sessions and "New York" in active_sessions: return active_sessions, "HIGH_LIQUIDITY", "تداخل لندن/نيويورك"
    elif len(active_sessions) >= 1: return active_sessions, "NORMAL_LIQUIDITY", f"{', '.join(active_sessions)}"
    else: return [], "LOW_LIQUIDITY", "خارج أوقات الذروة"

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

# ---------------------- استراتيجية التداول والفلاتر (محدثة لـ V10) ----------------------
class EnhancedTradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ml_model, self.scaler, self.feature_names = None, None, None

    def load_model(self) -> bool:
        model_name = f"{BASE_ML_MODEL_NAME}_{self.symbol}"
        if model_name in ml_models_cache:
            model_bundle = ml_models_cache[model_name]
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir_path = os.path.join(script_dir, MODEL_FOLDER)
            model_path = os.path.join(model_dir_path, f"{model_name}.pkl")

            if not os.path.exists(model_path):
                logger.warning(f"  -> [{self.symbol}] 🛑 ملف النموذج غير موجود في '{model_path}'.")
                return False
            try:
                with open(model_path, 'rb') as f:
                    model_bundle = pickle.load(f)
                ml_models_cache[model_name] = model_bundle
            except Exception as e:
                logger.error(f"❌ [ML Model File] خطأ في تحميل النموذج لـ {self.symbol}: {e}")
                return False
        
        if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
            self.ml_model = model_bundle['model']
            self.scaler = model_bundle['scaler']
            self.feature_names = model_bundle['feature_names']
            logger.info(f"  -> [{self.symbol}] ✅ تم تحميل النموذج بنجاح.")
            return True
        else:
            logger.error(f"  -> [{self.symbol}] 🛑 ملف النموذج غير مكتمل.")
            return False

    def get_features_for_model(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.feature_names is None:
            logger.error(f"  -> [{self.symbol}] 🛑 لا يمكن إعداد الميزات لأن أسماء الميزات غير محملة.")
            return None
        try:
            df_featured = calculate_all_features(df_15m, btc_df)
            df_4h_features = calculate_all_features(df_4h, None)
            df_4h_features = df_4h_features.rename(columns=lambda c: f"{c}_4h")
            
            required_4h_cols = [f for f in self.feature_names if f.endswith('_4h')]
            
            df_featured = df_featured.join(df_4h_features[required_4h_cols], how='left')
            df_featured[required_4h_cols] = df_featured[required_4h_cols].fillna(method='ffill')

            for col in self.feature_names:
                if col not in df_featured.columns:
                    df_featured[col] = 0.0
            
            df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_featured.dropna(subset=self.feature_names)
        except Exception as e:
            logger.error(f"❌ [{self.symbol}] فشل هندسة الميزات للنموذج: {e}", exc_info=True)
            return None

    def generate_buy_signal(self, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]) or df_features.empty:
            return None
        try:
            last_row_ordered_df = df_features.iloc[[-1]][self.feature_names]
            features_scaled = self.scaler.transform(last_row_ordered_df)
            prediction = self.ml_model.predict(features_scaled)[0]
            if prediction != 1:
                return None
            prediction_proba = self.ml_model.predict_proba(features_scaled)
            confidence = float(np.max(prediction_proba[0]))
            return {'prediction': int(prediction), 'confidence': confidence}
        except Exception as e:
            logger.warning(f"⚠️ [{self.symbol}] خطأ في توليد إشارة النموذج: {e}")
            return None

def passes_final_order_book_check(symbol: str, entry_price: float) -> bool:
    if not client:
        log_rejection(symbol, "Order Book Fetch Failed", {"error": "Client not initialized"})
        return False
    try:
        order_book = client.get_order_book(symbol=symbol, limit=ORDER_BOOK_DEPTH_LIMIT)
        bids = pd.DataFrame(order_book['bids'], columns=['price', 'qty'], dtype=float)
        asks = pd.DataFrame(order_book['asks'], columns=['price', 'qty'], dtype=float)

        price_range_upper = entry_price * (1 + ORDER_BOOK_ANALYSIS_RANGE_PCT)
        price_range_lower = entry_price * (1 - ORDER_BOOK_ANALYSIS_RANGE_PCT)

        relevant_bids_vol = bids[bids['price'].between(price_range_lower, entry_price)]['qty'].sum()
        relevant_asks_vol = asks[asks['price'].between(entry_price, price_range_upper)]['qty'].sum()

        if relevant_asks_vol == 0:
            return True

        bid_ask_ratio = relevant_bids_vol / relevant_asks_vol
        
        if bid_ask_ratio >= ORDER_BOOK_MIN_BID_ASK_RATIO:
            return True
        else:
            log_rejection(symbol, "Order Book Filter Failed", {"ratio": f"{bid_ask_ratio:.2f}", "required": f"{ORDER_BOOK_MIN_BID_ASK_RATIO}"})
            return False

    except Exception as e:
        log_rejection(symbol, "Order Book Fetch Failed", {"error": str(e)})
        return False

# --- START: S/R BASED TP/SL FUNCTIONS ---
SR_LOOKBACK_CANDLES = 50
SR_MIN_BOUNCES      = 2

def find_sr_levels(df: pd.DataFrame, lookback: int = 50, min_bounces: int = 2) -> Dict[str, float]:
    if len(df) < lookback:
        return {'support': None, 'resistance': None}

    candle_body_tops = df[['open', 'close']].max(axis=1)
    highs = candle_body_tops.iloc[-lookback:]
    lows  = df['low'].iloc[-lookback:]

    pivot_high = (highs == highs.rolling(5, center=True).max()) & (highs.shift(1) < highs) & (highs.shift(-1) < highs)
    pivot_low  = (lows  == lows.rolling(5, center=True).min())  & (lows.shift(1)  > lows)  & (lows.shift(-1)  > lows)

    highs_list = highs[pivot_high].dropna().tolist()
    lows_list  = lows[pivot_low].dropna().tolist()

    if not highs_list or not lows_list:
        return {'support': None, 'resistance': None}

    resistance = Counter(highs_list).most_common(1)[0][0] if highs_list else None
    support    = Counter(lows_list).most_common(1)[0][0] if lows_list else None

    return {'support': support, 'resistance': resistance}


def calculate_tp_sl(symbol: str, entry_price: float, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    try:
        if df.empty or len(df) < 50:
            log_rejection(symbol, "Insufficient data for TP/SL calculation")
            return None

        sr = find_sr_levels(df, lookback=SR_LOOKBACK_CANDLES, min_bounces=SR_MIN_BOUNCES)
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

            return {
                'target_price': round(new_target_price, 6),
                'stop_loss':    round(new_stop_loss, 6),
                'source':       'FIXED_PERCENTAGE',
                'rr_ratio':     round(1.2 / 1.5, 2)
            }

        if support >= entry_price:
            support = entry_price * 0.98

        risk_pct = ((entry_price - support) / entry_price) * 100
        if risk_pct < 0.3:
            support = entry_price * (1 - 0.003)

        return {
            'target_price': round(resistance, 6),
            'stop_loss':    round(support, 6),
            'source':       'SR_LEVELS',
            'rr_ratio':     round((resistance - entry_price) / (entry_price - support), 2) if (entry_price - support) > 0 else 0
        }

    except Exception as e:
        logger.error(f"❌ [{symbol}] Error in S/R TP/SL: {e}", exc_info=True)
        last_atr = df['atr'].iloc[-1] if 'atr' in df.columns and not df['atr'].empty else 0
        if last_atr > 0:
            return {'target_price': entry_price + last_atr * 2.2, 'stop_loss': entry_price - last_atr * 1.5, 'source': 'ATR_Fallback'}
        return None

# --- START: STRATEGY FUNCTIONS (V5 - Multi-Strategy) ---

def is_strong_bullish_candle(candle: pd.Series, prev_candle: pd.Series) -> bool:
    """
    التحقق من وجود شمعة صاعدة قوية وأكثر مرونة.
    """
    is_bullish = candle['close'] > candle['open']
    range_val = candle['high'] - candle['low']
    if range_val == 0: return False
    
    close_position_in_range = (candle['close'] - candle['low']) / range_val
    body_size = abs(candle['close'] - candle['open'])
    prev_body_size = abs(prev_candle['close'] - prev_candle['open'])
    
    return is_bullish and close_position_in_range > 0.7 and body_size > prev_body_size

# --- استراتيجية 1: الانعكاس (Divergence Reversal) ---
def find_bullish_divergence(df: pd.DataFrame, lookback: int = 40, pivot_window: int = 5) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    if len(df) < lookback: return None
    data = df.iloc[-lookback:].copy()
    data['price_pivot'] = data['low'].rolling(window=pivot_window*2+1, center=True).min()
    price_lows = data[data['low'] == data['price_pivot']].dropna()
    if len(price_lows) < 2: return None
    last_price_low = price_lows.iloc[-1]
    prev_price_low = price_lows.iloc[-2]
    if last_price_low['low'] >= prev_price_low['low']: return None
    rsi_low_at_last_price_low = data['rsi'].loc[last_price_low.name]
    rsi_low_at_prev_price_low = data['rsi'].loc[prev_price_low.name]
    if rsi_low_at_last_price_low > rsi_low_at_prev_price_low:
        logger.info(f"  -> [{df.name}] ✅ تم العثور على انفراج إيجابي بين {prev_price_low.name} و {last_price_low.name}")
        return prev_price_low.name, last_price_low.name
    return None

def calculate_dynamic_rsi_level(rsi_series: pd.Series, period: int = 20, std_dev: float = 2.0) -> float:
    if len(rsi_series) < period: return 30.0
    rsi_ma = rsi_series.rolling(window=period).mean()
    rsi_std = rsi_series.rolling(window=period).std()
    rsi_bb_lower = rsi_ma - (rsi_std * std_dev)
    return rsi_bb_lower.iloc[-1]

def check_reversal_strategy(df: pd.DataFrame) -> Tuple[bool, str, Optional[Dict]]:
    df.name = df.name if hasattr(df, 'name') else 'DataFrame'
    divergence_points = find_bullish_divergence(df)
    if not divergence_points:
        return False, "No Bullish Divergence", {}
    
    first_low_ts, _ = divergence_points
    dynamic_oversold_level = calculate_dynamic_rsi_level(df['rsi'])
    rsi_at_first_low = df.loc[first_low_ts]['rsi']
    if rsi_at_first_low > dynamic_oversold_level:
        details = {"rsi_at_low": f"{rsi_at_first_low:.2f}", "dynamic_level": f"{dynamic_oversold_level:.2f}"}
        return False, "Divergence not from Oversold", details
        
    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    stoch_k_cross_above_d = prev_candle['stoch_k'] <= prev_candle['stoch_d'] and last_candle['stoch_k'] > last_candle['stoch_d']
    if not stoch_k_cross_above_d:
        return False, "No Stochastic Crossover", {}

    if not is_strong_bullish_candle(last_candle, prev_candle):
        return False, "No Confirmation Candle", {}
            
    avg_volume = df['volume'].iloc[-11:-1].mean()
    if last_candle['volume'] < avg_volume:
        details = {"signal_volume": f"{last_candle['volume']:.2f}", "avg_volume": f"{avg_volume:.2f}"}
        return False, "Low Volume on Signal Candle", details
    
    if last_candle['buy_pressure'] < 0.55:
        return False, "Low Buy Pressure", {"buy_pressure": f"{last_candle['buy_pressure']:.2f}"}

    return True, "Divergence_Reversal", {}

# --- استراتيجية 2: متابعة الاتجاه (Trend Continuation) ---
def check_trend_continuation_strategy(df: pd.DataFrame) -> Tuple[bool, str, Optional[Dict]]:
    df.name = df.name if hasattr(df, 'name') else 'DataFrame'
    
    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    
    # --- START: MODIFICATION ---
    # تم تخفيف الشرط للتركيز على هيكل الاتجاه العام (ema_21 > ema_50)
    # والسماح للسعر بالانخفاض مؤقتًا تحت ema_21 كجزء من التصحيح.
    if not (last_candle['ema_21'] > last_candle['ema_50']):
        return False, "Not an Uptrend", {}
    # --- END: MODIFICATION ---
        
    pullback_occurred = (prev_candle['low'] <= prev_candle['ema_21']) or (last_candle['low'] <= last_candle['ema_21'])
    if not pullback_occurred:
        return False, "No Pullback to EMA", {}
        
    if last_candle['rsi'] < 45:
        return False, "RSI below threshold", {"rsi": f"{last_candle['rsi']:.2f}"}
        
    if not is_strong_bullish_candle(last_candle, prev_candle):
        return False, "No Confirmation Candle", {}
        
    avg_volume = df['volume'].iloc[-11:-1].mean()
    if last_candle['volume'] < avg_volume * 1.1:
        details = {"signal_volume": f"{last_candle['volume']:.2f}", "avg_volume": f"{avg_volume:.2f}"}
        return False, "Low Volume on Signal Candle", details

    return True, "Uptrend_Pullback", {}

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
            adjusted_quantity = (quantity_decimal // step_size) * step_size
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

def verify_order_filled(symbol: str, order_id: str, timeout_seconds: int = 30) -> bool:
    if not client:
        return False
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            order_status = client.get_order(symbol=symbol, orderId=order_id)
            if order_status['status'] == 'FILLED':
                logger.info(f"✅ [{symbol}] Order {order_id} confirmed as FILLED.")
                return True
            elif order_status['status'] in ['CANCELED', 'EXPIRED', 'REJECTED']:
                logger.error(f"❌ [{symbol}] Order {order_id} has failed status: {order_status['status']}.")
                return False
            time.sleep(2)
        except BinanceAPIException as e:
            logger.error(f"❌ [{symbol}] API Error while verifying order {order_id}: {e}")
            time.sleep(5)
        except Exception as e:
            logger.error(f"❌ [{symbol}] Unexpected error while verifying order {order_id}: {e}", exc_info=True)
            return False

    logger.error(f"⌛️ [{symbol}] Timeout reached while waiting for order {order_id} to be filled.")
    return False

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
            try:
                symbol_info = exchange_info_map.get(symbol_to_close)
                if not symbol_info:
                    logger.error(f"❌ [{symbol_to_close}] لا يمكن العثور على معلومات العملة لإغلاق الصفقة.")
                    return False
                base_asset = symbol_info['baseAsset']

                balance_response = client.get_asset_balance(asset=base_asset)
                available_balance = Decimal(balance_response['free'])
                logger.info(f"ℹ️ [{symbol_to_close}] الرصيد المتاح للبيع من {base_asset}: {available_balance}")

                if available_balance <= 0:
                    logger.warning(f"⚠️ [{symbol_to_close}] لا يوجد رصيد متاح للبيع من {base_asset}. سيتم إغلاق الصفقة في النظام فقط.")
                else:
                    quantity_to_sell = adjust_quantity_to_lot_size(symbol_to_close, float(available_balance))
                    if quantity_to_sell is None or quantity_to_sell <= 0:
                        logger.error(f"❌ [{symbol_to_close}] فشل تعديل كمية البيع أو الكمية المعدلة صفر. الكمية المتاحة: {available_balance}")
                        return False

                    notional_value = quantity_to_sell * Decimal(str(closing_price))
                    min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] in ('MIN_NOTIONAL', 'NOTIONAL')), None)

                    min_notional_ok = True
                    if min_notional_filter:
                        min_notional = Decimal(min_notional_filter.get('minNotional', min_notional_filter.get('notional', '0')))
                        if notional_value < min_notional:
                            logger.warning(f"⚠️ [{symbol_to_close}] قيمة البيع ({notional_value:.2f}) أقل من الحد الأدنى ({min_notional}). سيتم إغلاق الصفقة في النظام فقط.")
                            min_notional_ok = False

                    if min_notional_ok:
                        sell_order = place_order(symbol_to_close, Client.SIDE_SELL, quantity_to_sell)
                        if not sell_order:
                            log_and_notify('error', f"CRITICAL: Sell order placement failed for {symbol_to_close}. Trade remains open.", "TRADE_ERROR")
                            return False

                        order_id = sell_order.get('orderId')
                        if not verify_order_filled(symbol_to_close, order_id):
                            log_and_notify(
                                'critical',
                                f"🚨 MANUAL ACTION REQUIRED: Sell order for {symbol_to_close} (ID: {order_id}) was placed but NOT confirmed as FILLED. The trade remains open in the bot. Please verify on Binance.",
                                "CRITICAL_SELL_FAILURE"
                            )
                            send_telegram_message(
                                f"🚨 *فشل إغلاق صفقة حرجة*\n\n"
                                f"*العملة:* `{symbol_to_close}`\n"
                                f"*المشكلة:* تم إرسال أمر البيع لكن لم يتم تأكيد تنفيذه.\n"
                                f"الصفقة لا تزال مفتوحة في البوت. *الرجاء التحقق من حسابك في باينانس يدوياً!*"
                            )
                            return False

                        logger.info(f"✅ [{symbol_to_close}] Sell order {order_id} successfully filled and verified.")

            except Exception as e:
                logger.error(f"❌ [{symbol_to_close}] خطأ حرج أثناء تحضير أمر البيع: {e}", exc_info=True)
                return False

        if not check_db_connection() or not conn:
            log_and_notify('critical', "DB connection lost during trade closure. Data might be inconsistent.", "DB_ERROR")
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

            reason_map = {
                'take_profit': '🎯 Take Profit',
                'stop_loss': '🛑 Stop Loss',
                'manual': '🖐️ Manual Close',
                'atr_trailing_stop': '🛡️ وقف خسارة متحرك (ATR)'
            }
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
                signal_data['symbol'],
                entry_price,
                target_price,
                stop_loss,
                signal_data['strategy_name'],
                json.dumps(signal_data['signal_details']),
                signal_data.get('is_real_trade', False),
                quantity,
                signal_data.get('order_id'),
                entry_price
            ))
            saved_signal = cur.fetchone()
            conn.commit()
            logger.info(f"💾 [{signal_data['symbol']}] تم حفظ الإشارة الجديدة في قاعدة البيانات.")

            trade_type = "حقيقية" if signal_data.get('is_real_trade') else "تجريبية"
            telegram_message = (
                f"💡 *توصية شراء {trade_type} جديدة*\n\n"
                f"*العملة:* `{signal_data['symbol']}`\n"
                f"*الاستراتيجية:* `{signal_data['strategy_name'].replace('_', ' ')}`\n"
                f"*سعر الدخول:* `{entry_price:.4f}`\n"
                f"*الهدف (TP):* `{target_price:.4f}`\n"
                f"*وقف الخسارة (SL):* `{stop_loss:.4f}`\n\n"
                f"Confidence: {signal_data['signal_details'].get('ML_Confidence', 'N/A')}"
            )
            send_telegram_message(telegram_message)

            return dict(saved_signal)
    except Exception as e:
        logger.error(f"❌ [DB Insert] فشل إدراج الإشارة: {e}"); conn.rollback(); return None

# ---------------------- دوال النظام الرئيسية ----------------------
def determine_market_state_enhanced():
    global current_market_state, last_market_state_check
    if time.time() - last_market_state_check < 180: return
    logger.info("🧠 [Market State] تحديث حالة السوق...")
    try:
        trend_details = {}
        for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
            df = fetch_historical_data(BTC_SYMBOL, tf, 20)
            if df is not None and not df.empty:
                ema_fast = df['close'].ewm(span=12, adjust=False).mean().iloc[-1]
                ema_slow = df['close'].ewm(span=26, adjust=False).mean().iloc[-1]
                adx_features = calculate_all_features(df, None)
                adx = adx_features['adx'].iloc[-1] if not adx_features.empty else 0
                if ema_fast > ema_slow and adx > 25: trend = "Strong Uptrend"
                elif ema_fast > ema_slow: trend = "Uptrend"
                elif ema_fast < ema_slow and adx > 25: trend = "Strong Downtrend"
                elif ema_fast < ema_slow: trend = "Downtrend"
                else: trend = "Ranging"
                trend_details[tf] = {"trend": trend, "adx": float(adx)}
            else: trend_details[tf] = {"trend": "Uncertain", "adx": 0}
        trends = [d['trend'] for d in trend_details.values()]
        overall_regime = max(set(trends), key=trends.count) if trends else "Uncertain"
        with market_state_lock:
            current_market_state = {"overall_regime": overall_regime.upper().replace(" ", "_"), "trend_details_by_tf": trend_details, "last_updated": datetime.now(timezone.utc).isoformat()}
            last_market_state_check = time.time()
        logger.info(f"✅ [Market State] الحالة العامة: {overall_regime}")
    except Exception as e:
        logger.error(f"❌ [Market State] خطأ: {e}", exc_info=True)

# ---------------------- واجهة Flask ----------------------
app = Flask(__name__)
CORS(app)

def get_dashboard_html():
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم التداول V10 - نظام متعدد الاستراتيجيات</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
    <style>
        :root { --bg-main: #0D1117; --bg-card: #161B22; --border-color: #30363D; --text-primary: #E6EDF3; --text-secondary: #848D97; --accent-blue: #58A6FF; --accent-green: #3FB950; --accent-red: #F85149; --accent-yellow: #D29922; }
        body { font-family: 'Tajawal', sans-serif; background-color: var(--bg-main); color: var(--text-primary); }
        .card { background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 0.5rem; }
        .trend-light { width: 1rem; height: 1rem; border-radius: 9999px; border: 2px solid #30363D; transition: all 0.5s ease; }
        .light-on-green { background-color: var(--accent-green); box-shadow: 0 0 10px 2px var(--accent-green); }
        .light-on-red { background-color: var(--accent-red); box-shadow: 0 0 10px 2px var(--accent-red); }
        .light-on-yellow { background-color: var(--accent-yellow); box-shadow: 0 0 10px 2px var(--accent-yellow); }
        .tab-btn.active { border-bottom-color: var(--accent-blue); }
        input:checked + .toggle-bg { background-color: var(--accent-green); }
        #modal-overlay { transition: opacity 0.3s ease; }
    </style>
</head>
<body class="p-4 md:p-6">
    <div id="modal-overlay" class="fixed inset-0 bg-black bg-opacity-70 hidden items-center justify-center z-50">
        <div id="modal-content" class="card p-6 rounded-lg shadow-xl max-w-sm w-full">
            <h3 id="modal-title" class="text-xl font-bold mb-4"></h3>
            <p id="modal-body" class="text-text-secondary mb-6"></p>
            <div class="flex justify-end gap-3">
                <button id="modal-cancel" class="px-4 py-2 rounded-md bg-gray-600 hover:bg-gray-700">إلغاء</button>
                <button id="modal-confirm" class="px-4 py-2 rounded-md bg-red-600 hover:bg-red-700">تأكيد</button>
            </div>
        </div>
    </div>

    <div class="container mx-auto max-w-screen-2xl">
        <header class="mb-6 flex flex-wrap justify-between items-center gap-4">
            <h1 class="text-2xl md:text-3xl font-extrabold"><span class="text-accent-blue">لوحة تحكم</span><span class="text-text-secondary font-medium"> V5 (Multi-Strategy)</span></h1>
            <div id="trend-lights-container" class="flex items-center gap-x-6 bg-black/20 px-4 py-2 rounded-lg border border-border-color"></div>
        </header>
        <section class="mb-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">حالة السوق</h3><div id="overall-regime" class="text-2xl font-bold text-center">...</div></div>
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">الجلسات النشطة</h3><div id="active-sessions-list" class="flex flex-wrap gap-2 items-center justify-center pt-2">...</div></div>
            <div class="card p-4 flex flex-col justify-center items-center"><h3 class="font-bold text-lg text-text-secondary mb-2">التداول الحقيقي</h3><div class="flex items-center space-x-3 space-x-reverse"><span id="trading-status-text" class="font-bold text-lg"></span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="trading-toggle" class="sr-only" onchange="toggleTrading()"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div><div class="mt-2 text-xs text-text-secondary">رصيد USDT: <span id="usdt-balance" class="font-mono">...</span></div></div>
            <div class="card p-4 flex flex-col justify-center items-center"><h3 class="font-bold text-lg text-text-secondary mb-2">حالة الفلاتر</h3><div class="flex items-center space-x-3 space-x-reverse"><span id="disable-filters-text" class="font-bold text-lg"></span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="disable-filters-toggle" class="sr-only" onchange="toggleFilters()"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div><div class="mt-2 text-xs text-text-secondary">فلتر دفتر الطلبات نشط دائمًا</div></div>
        </section>
        <div class="mb-4 border-b border-border-color"><nav class="flex space-x-6 space-x-reverse -mb-px"><button onclick="showTab('signals', this)" class="tab-btn active text-white py-3 px-1 font-semibold">الصفقات</button><button onclick="showTab('stats', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الإحصائيات</button><button onclick="showTab('notifications', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الإشعارات</button><button onclick="showTab('rejections', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الصفقات المرفوضة</button></nav></div>
        <main>
            <div id="signals-tab" class="tab-content"><div class="overflow-x-auto card p-0"><table class="min-w-full text-sm text-right"><thead class="border-b border-border-color bg-black/20"><tr><th class="p-4 font-semibold">العملة</th><th class="p-4 font-semibold">الاستراتيجية</th><th class="p-4 font-semibold">الربح/الخسارة</th><th class="p-4 font-semibold w-[25%]">التقدم</th><th class="p-4 font-semibold">الدخول/الحالي</th><th class="p-4 font-semibold">إجراء</th></tr></thead><tbody id="signals-table"></tbody></table></div></div>
            <div id="stats-tab" class="tab-content hidden"><div id="stats-container" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"></div></div>
            <div id="notifications-tab" class="tab-content hidden"><div id="notifications-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-2"></div></div>
            <div id="rejections-tab" class="tab-content hidden"><div id="rejections-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-2"></div></div>
        </main>
    </div>
<script>
let confirmCallback = null;
const modal = {
    overlay: document.getElementById('modal-overlay'),
    title: document.getElementById('modal-title'),
    body: document.getElementById('modal-body'),
    confirmBtn: document.getElementById('modal-confirm'),
    cancelBtn: document.getElementById('modal-cancel'),
};
modal.cancelBtn.onclick = () => { modal.overlay.classList.add('hidden'); };
modal.confirmBtn.onclick = () => { if(confirmCallback) confirmCallback(); modal.overlay.classList.add('hidden'); };

function showConfirmation(title, bodyText, onConfirm) {
    modal.title.textContent = title;
    modal.body.textContent = bodyText;
    confirmCallback = onConfirm;
    modal.overlay.classList.remove('hidden');
    modal.overlay.classList.add('flex');
}
function showTab(tabId, el) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.add('hidden'));
    document.getElementById(tabId + '-tab').classList.remove('hidden');
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active', 'text-white'));
    el.classList.add('active', 'text-white');
}
async function fetchData(url) { try { const r = await fetch(url); return r.ok ? await r.json() : null; } catch (e) { console.error('Fetch Error:', e); return null; } }
function updateMarketStatus() {
    fetchData('/api/market_status').then(data => {
        if (!data) return;
        document.getElementById('overall-regime').textContent = (data.market_state?.overall_regime || 'UNCERTAIN').replace(/_/g, ' ');
        const lights = document.getElementById('trend-lights-container');
        lights.innerHTML = '';
        ['15m', '1h', '4h'].forEach(tf => {
            const trendInfo = data.market_state?.trend_details_by_tf[tf];
            const trend = trendInfo?.trend || 'Uncertain';
            let c = trend.includes('Uptrend') ? 'light-on-green' : trend.includes('Downtrend') ? 'light-on-red' : 'light-on-yellow';
            lights.innerHTML += `<div class="flex items-center gap-2"><div class="trend-light ${c}"></div><span class="text-sm font-bold text-text-secondary">${tf}</span></div>`;
        });
        const sessions = document.getElementById('active-sessions-list');
        sessions.innerHTML = data.active_sessions.length > 0 ? data.active_sessions.map(s => `<span class="bg-accent-blue/20 text-accent-blue text-xs font-bold px-2 py-1 rounded">${s}</span>`).join('') : `<span class="bg-gray-700 text-text-secondary text-xs font-bold px-2 py-1 rounded">لا توجد</span>`;

        const tradeToggle = document.getElementById('trading-toggle'), tradeText = document.getElementById('trading-status-text');
        tradeToggle.checked = data.is_trading_enabled;
        tradeText.textContent = data.is_trading_enabled ? 'مُفعَّل' : 'غير مُفعَّل';
        tradeText.className = `font-bold text-lg ${data.is_trading_enabled ? 'text-accent-green' : 'text-accent-red'}`;
        document.getElementById('usdt-balance').textContent = data.usdt_balance ? parseFloat(data.usdt_balance).toFixed(2) : 'N/A';

        const filtersToggle = document.getElementById('disable-filters-toggle'), filtersText = document.getElementById('disable-filters-text');
        filtersToggle.checked = data.are_filters_disabled;
        filtersText.textContent = data.are_filters_disabled ? 'معطلة' : 'مفعلة';
        filtersText.className = `font-bold text-lg ${data.are_filters_disabled ? 'text-accent-red' : 'text-accent-green'}`;
    });
}
function updateSignals() {
    fetchData('/api/signals').then(data => {
        if (!data) return;
        const tableBody = document.getElementById('signals-table');
        tableBody.innerHTML = '';
        data.filter(s => ['open', 'updated'].includes(s.status)).forEach(s => {
            const profit = parseFloat(s.profit_percentage || 0);
            const pClass = profit > 0 ? 'text-accent-green' : profit < 0 ? 'text-accent-red' : 'text-text-secondary';
            const entry = parseFloat(s.entry_price), sl = parseFloat(s.stop_loss), tp = parseFloat(s.target_price), current = parseFloat(s.current_price || entry);
            const progress = (tp - sl > 0) ? Math.max(0, Math.min(100, (current - sl) / (tp - sl) * 100)) : 0;
            const strategyName = s.strategy_name || 'N/A';
            let strategyClass = 'bg-gray-700 text-gray-200';
            if (strategyName.includes('Pullback')) {
                strategyClass = 'bg-green-500/20 text-green-400';
            } else if (strategyName.includes('Reversal')) {
                strategyClass = 'bg-yellow-500/20 text-yellow-400';
            }
            tableBody.innerHTML += `<tr class="border-b border-border-color hover:bg-white/5"><td class="p-4 font-bold">${s.symbol}</td><td class="p-4"><span class="px-2 py-1 text-xs font-semibold rounded-full ${strategyClass}">${strategyName.replace(/_/g, ' ')}</span></td><td class="p-4 font-mono ${pClass}">${profit.toFixed(2)}%</td><td class="p-4"><div class="w-full bg-gray-700 rounded-full h-2.5"><div class="bg-accent-blue h-2.5 rounded-full" style="width: ${progress}%"></div></div></td><td class="p-4 font-mono">${current.toFixed(4)} / ${entry.toFixed(4)}</td><td class="p-4"><button onclick="manualClose(${s.id}, '${s.symbol}')" class="bg-red-600 hover:bg-red-700 text-white font-bold py-1 px-3 rounded text-xs">إغلاق</button></td></tr>`;
        });
    });
}
function updateStats() {
    fetchData('/api/stats').then(data => {
        if (!data) return;
        const container = document.getElementById('stats-container');
        if (data.error) {
            container.innerHTML = `<div class="card p-4 text-center col-span-full text-accent-red">${data.error}</div>`;
            return;
        }
        container.innerHTML = `<div class="card p-4 text-center"><h4 class="text-text-secondary">صافي الربح</h4><div class="text-2xl font-bold ${data.net_profit_usdt >= 0 ? 'text-accent-green' : 'text-accent-red'}">${parseFloat(data.net_profit_usdt).toFixed(2)}</div></div><div class="card p-4 text-center"><h4 class="text-text-secondary">معدل الربح</h4><div class="text-2xl font-bold">${parseFloat(data.win_rate).toFixed(2)}%</div></div><div class="card p-4 text-center"><h4 class="text-text-secondary">عامل الربح</h4><div class="text-2xl font-bold">${data.profit_factor === 'Infinity' ? '∞' : parseFloat(data.profit_factor).toFixed(2)}</div></div><div class="card p-4 text-center"><h4 class="text-text-secondary">الصفقات المغلقة</h4><div class="text-2xl font-bold">${data.total_closed_trades}</div></div>`;
    });
}
function updateNotifications() {
    fetchData('/api/notifications').then(data => {
        if (!data) return;
        document.getElementById('notifications-list').innerHTML = data.map(n => `<div class="p-2 border-b border-border-color"><span class="font-mono text-xs text-text-secondary">${new Date(n.timestamp).toLocaleString('ar-EG')}</span>: ${n.message}</div>`).join('');
    });
}
function updateRejections() {
    fetchData('/api/rejection_logs').then(data => {
        if (!data) return;
        document.getElementById('rejections-list').innerHTML = data.map(r => `<div class="p-2 border-b border-border-color"><span class="font-mono text-xs text-text-secondary">${new Date(r.timestamp).toLocaleString('ar-EG')}</span>: <strong class="text-accent-yellow">${r.symbol}</strong> - ${r.reason} <span class="text-xs text-gray-500">${JSON.stringify(r.details)}</span></div>`).join('');
    });
}
function manualClose(signalId, symbol) {
    showConfirmation('تأكيد الإغلاق', `هل أنت متأكد من رغبتك في إغلاق الصفقة لـ ${symbol} يدوياً؟`, () => {
        fetch(`/api/signals/close/${signalId}`, { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                if(data.success) {
                    updateSignals();
                } else {
                    alert(data.message);
                }
            });
    });
}
function toggleTrading() { fetch('/api/trading/toggle', { method: 'POST' }).then(() => updateMarketStatus()); }
function toggleFilters() { fetch('/api/filters/disable/toggle', { method: 'POST' }).then(() => updateMarketStatus()); }
document.addEventListener('DOMContentLoaded', () => {
    ['MarketStatus', 'Signals', 'Stats', 'Notifications', 'Rejections'].forEach(f => window[`update${f}`]());
    setInterval(updateMarketStatus, 5000); setInterval(updateSignals, 7000); setInterval(updateStats, 60000);
    setInterval(updateNotifications, 15000); setInterval(updateRejections, 15000);
});
</script>
</body></html>
"""

@app.route('/')
def home(): return render_template_string(get_dashboard_html())

@app.route('/api/market_status')
def get_market_status():
    with market_state_lock: state_copy = dict(current_market_state)
    with filters_disabled_lock: is_disabled = are_filters_disabled
    with trading_status_lock: is_enabled = is_trading_enabled
    active_sessions, _, _ = get_session_state()
    usdt_balance = None
    if client:
        try: usdt_balance = float(client.get_asset_balance(asset='USDT')['free'])
        except: usdt_balance = 'N/A'
    return jsonify({
        "market_state": state_copy,
        "active_sessions": active_sessions,
        "usdt_balance": usdt_balance,
        "is_trading_enabled": is_enabled,
        "are_filters_disabled": is_disabled
    })

@app.route('/api/stats')
def get_stats():
    if not check_db_connection() or not conn:
        return jsonify({"error": "DB connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT profit_percentage, is_real_trade, quantity, entry_price FROM signals WHERE status = 'closed';")
            closed_trades = cur.fetchall()

        if not closed_trades:
            return jsonify({"net_profit_usdt": 0, "win_rate": 0, "profit_factor": 0, "total_closed_trades": 0})

        total_net_profit_usdt = sum(
            ((float(t['profit_percentage']) - (2 * TRADING_FEE_PERCENT)) / 100) * (float(t['quantity']) * float(t['entry_price']) if t.get('is_real_trade') and t.get('quantity') and t.get('entry_price') else STATS_TRADE_SIZE_USDT)
            for t in closed_trades
        )
        wins = [float(s['profit_percentage']) for s in closed_trades if float(s['profit_percentage']) > 0]
        losses = [float(s['profit_percentage']) for s in closed_trades if float(s['profit_percentage']) < 0]
        win_rate = (len(wins) / len(closed_trades) * 100) if closed_trades else 0.0
        total_loss = abs(sum(losses))
        profit_factor = sum(wins) / total_loss if total_loss > 0 else "Infinity"

        return jsonify({
            "net_profit_usdt": total_net_profit_usdt,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_closed_trades": len(closed_trades)
        })
    except Exception as e:
        logger.error(f"❌ [API Stats] Error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error fetching stats"}), 500

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

@app.route('/api/notifications')
def get_notifications():
    with notifications_lock:
        return jsonify(list(notifications_cache))

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

@app.route('/api/filters/disable/toggle', methods=['POST'])
def toggle_disable_filters():
    global are_filters_disabled
    with filters_disabled_lock:
        are_filters_disabled = not are_filters_disabled
        status_msg = "DISABLED" if are_filters_disabled else "ENABLED"
        log_and_notify('warning', f"⚙️ Filters status changed to: {status_msg}", "FILTER_STATUS_CHANGE")
        return jsonify({"message": f"Filters status set to {status_msg}"})

@app.route('/api/signals/close/<int:signal_id>', methods=['POST'])
def manual_close_trade_endpoint(signal_id):
    if not redis_client or not client: return jsonify({"success": False, "message": "Services not ready"}), 503

    with signal_cache_lock:
        signal_to_close = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)

    if not signal_to_close: return jsonify({"success": False, "message": "Signal not found or already closed"}), 404

    try:
        current_price = float(redis_client.hget(REDIS_PRICES_HASH_NAME, signal_to_close['symbol']))
    except (TypeError, ValueError):
        try: current_price = float(client.get_symbol_ticker(symbol=signal_to_close['symbol'])['price'])
        except Exception as e: return jsonify({"success": False, "message": f"Could not fetch current price: {e}"}), 500

    if close_signal(signal_id, current_price, 'manual'):
        return jsonify({"success": True, "message": f"Signal for {signal_to_close['symbol']} closed successfully."})
    else:
        return jsonify({"success": False, "message": "Failed to close signal. Check logs."}), 500

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
                symbol = signal['symbol']
                tp = float(signal['target_price'])
                sl = float(signal['stop_loss'])
                entry = float(signal['entry_price'])

                if current_price >= tp:
                    logger.info(f"🎯 [TP HIT] {symbol} at {current_price}")
                    close_signal(signal_id, current_price, 'take_profit')
                    continue

                peak_price = float(signal.get('current_peak_price', entry))
                new_peak = max(peak_price, current_price)
                if new_peak > peak_price:
                    with signal_cache_lock:
                        if symbol in open_signals_cache:
                            open_signals_cache[symbol]['current_peak_price'] = new_peak
                    try:
                        if check_db_connection():
                            with conn.cursor() as cur:
                                cur.execute("UPDATE signals SET current_peak_price = %s WHERE id = %s", (float(new_peak), signal_id))
                            conn.commit()
                    except Exception as e:
                        logger.error(f"DB error updating peak price for {symbol}: {e}"); conn.rollback()
                    signal['current_peak_price'] = new_peak

                if USE_ATR_TRAILING_STOP:
                    try:
                        df_atr = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, 3)
                        if df_atr is not None and len(df_atr) > ATR_TS_PERIOD:
                            high_low = df_atr['high'] - df_atr['low']
                            high_close_prev = (df_atr['high'] - df_atr['close'].shift()).abs()
                            low_close_prev = (df_atr['low'] - df_atr['close'].shift()).abs()
                            tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)
                            latest_atr = tr.ewm(span=ATR_TS_PERIOD, adjust=False).mean().iloc[-1]

                            if latest_atr > 0:
                                new_trailing_stop_price = new_peak - (latest_atr * ATR_TS_MULTIPLIER)
                                if new_trailing_stop_price > sl:
                                    logger.info(f"📈 [ATR TRAILING SL] {symbol} SL moved up to {new_trailing_stop_price:.4f}")
                                    sl = new_trailing_stop_price
                                    with signal_cache_lock:
                                        if symbol in open_signals_cache:
                                            open_signals_cache[symbol]['stop_loss'] = new_trailing_stop_price
                                    try:
                                        if check_db_connection():
                                            with conn.cursor() as cur:
                                                cur.execute("UPDATE signals SET stop_loss = %s WHERE id = %s", (float(new_trailing_stop_price), signal_id))
                                            conn.commit()
                                    except Exception as e:
                                        logger.error(f"DB error updating ATR trailing SL for {symbol}: {e}"); conn.rollback()
                    except Exception as e:
                        logger.error(f"❌ [{symbol}] Error during ATR Trailing Stop calculation: {e}")

                if current_price <= sl:
                    reason = 'atr_trailing_stop' if USE_ATR_TRAILING_STOP and sl > float(signal['stop_loss']) else 'stop_loss'
                    logger.info(f"🛑 [{reason.upper()} HIT] {symbol} at {current_price}")
                    close_signal(signal_id, current_price, reason)
                    continue

            time.sleep(2)
        except Exception as e:
            logger.error(f"❌ [Trade Manager] خطأ في حلقة الإدارة: {e}", exc_info=True)
            time.sleep(10)

def main_loop_enhanced():
    """
    الحلقة الرئيسية بنظام متعدد الاستراتيجيات وخط تجميع واضح (V5).
    """
    global technical_signals_cache
    logger.info("[Main Loop] انتظار اكتمال التهيئة...")
    time.sleep(15)
    
    if not validated_symbols_to_scan:
        log_and_notify("critical", "قائمة العملات للمسح فارغة. يرجى التحقق من ملف 'crypto_list.txt'.", "SYSTEM_ERROR")
        return

    log_and_notify("info", f"✅ بدء حلقة المسح لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")

    while True:
        try:
            logger.info("🔄 [Main Loop] بدء دورة مسح جديدة...")
            determine_market_state_enhanced()
            with market_state_lock:
                market_regime = current_market_state.get("overall_regime", "UNCERTAIN")

            btc_data = get_btc_data_for_bot()
            symbols_to_process = random.sample(validated_symbols_to_scan, len(validated_symbols_to_scan))

            for i in range(0, len(symbols_to_process), SYMBOL_PROCESSING_BATCH_SIZE):
                batch = symbols_to_process[i:i + SYMBOL_PROCESSING_BATCH_SIZE]
                total_batches = (len(symbols_to_process) + SYMBOL_PROCESSING_BATCH_SIZE - 1) // SYMBOL_PROCESSING_BATCH_SIZE
                logger.info(f"🔄 Processing batch {i // SYMBOL_PROCESSING_BATCH_SIZE + 1}/{total_batches} ({len(batch)} symbols)...")

                for symbol in batch:
                    try:
                        logger.info(f"---===[ 🔍 تحليل {symbol} | حالة السوق: {market_regime} ]===---")

                        with signal_cache_lock:
                            if symbol in open_signals_cache or len(open_signals_cache) >= MAX_OPEN_TRADES:
                                continue
                        
                        df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        if df_15m is None or df_15m.empty:
                            continue
                        
                        df_with_indicators = calculate_all_features(df_15m, btc_data)
                        if df_with_indicators is None or df_with_indicators.empty:
                            continue
                        
                        # =================================================================
                        # ============== خط تجميع الإشارة (Signal Pipeline) ==============
                        # =================================================================

                        # --- المرحلة 1: الاكتشاف الفني (Technical Discovery) ---
                        strategy_signal_found = False
                        strategy_name = ""
                        rejection_key = "Strategy Signal Not Found"
                        details = {}

                        if market_regime in ["UPTREND", "STRONG_UPTREND"]:
                            logger.info(f"  -> [مرحلة 1] تفعيل استراتيجية متابعة الاتجاه (Pullback)...")
                            strategy_signal_found, rejection_key, details = check_trend_continuation_strategy(df_with_indicators)
                        elif market_regime in ["DOWNTREND", "RANGING"]:
                            logger.info(f"  -> [مرحلة 1] تفعيل استراتيجية الانعكاس (Reversal)...")
                            strategy_signal_found, rejection_key, details = check_reversal_strategy(df_with_indicators)
                        else:
                            rejection_key = "Invalid Market Regime for Any Strategy"
                            details = {"regime": market_regime}

                        if not strategy_signal_found:
                            log_rejection(symbol, rejection_key, details)
                            continue
                        
                        strategy_name = rejection_key # اسم الاستراتيجية الناجحة
                        logger.info(f"  -> [مرحلة 1] ✅ نجاح! إشارة فنية مكتشفة (استراتيجية: {strategy_name}).")
                        
                        # --- المرحلة 2: التحقق من النموذج الآلي (ML Validation) ---
                        logger.info(f"  -> [مرحلة 2] الانتقال إلى التحقق من النموذج الآلي...")
                        strategy = EnhancedTradingStrategy(symbol)
                        if not strategy.load_model():
                            log_rejection(symbol, "ML Model Load Failed")
                            continue

                        df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        if df_4h is None or df_4h.empty:
                            continue

                        df_features_for_model = strategy.get_features_for_model(df_15m, df_4h, btc_data)
                        if df_features_for_model is None or df_features_for_model.empty:
                            continue
                        
                        ml_signal = strategy.generate_buy_signal(df_features_for_model)
                        if not ml_signal or ml_signal['confidence'] < BUY_CONFIDENCE_THRESHOLD:
                            log_rejection(symbol, "ML Model Rejected Signal", {"confidence": ml_signal['confidence'] if ml_signal else 'N/A'})
                            continue
                        
                        logger.info(f"  -> [مرحلة 2] ✅ نجاح! النموذج يؤكد الإشارة (Confidence: {ml_signal['confidence']:.2%}).")

                        # --- المرحلة 3: التحقق من دفتر الطلبات (Order Book Validation) ---
                        logger.info(f"  -> [مرحلة 3] الانتقال إلى التحقق من دفتر الطلبات...")
                        try:
                            entry_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                        except Exception as e:
                            logger.error(f"❌ [{symbol}] فشل جلب سعر الدخول: {e}.")
                            continue

                        if not passes_final_order_book_check(symbol, entry_price):
                            # سيتم تسجيل الرفض داخل الدالة نفسها
                            continue
                        
                        logger.info(f"  -> [مرحلة 3] ✅ نجاح! دفتر الطلبات يؤكد الإشارة.")

                        # --- المرحلة 4: التنفيذ (Execution) ---
                        logger.info(f"  -> [مرحلة 4] ✅ تم تجاوز جميع الشروط. تحضير الصفقة...")
                        tp_sl_data = calculate_tp_sl(symbol, entry_price, df_15m)
                        if not tp_sl_data: continue

                        signal_details = {
                            'ML_Confidence': f"{ml_signal['confidence']:.2%}",
                            'Pattern': strategy_name,
                            'Signal_Type': 'Multi-Strategy System',
                            **tp_sl_data
                        }

                        new_signal = {
                            'symbol': symbol,
                            'strategy_name': strategy_name,
                            'signal_details': signal_details,
                            'entry_price': entry_price,
                            **tp_sl_data
                        }

                        with trading_status_lock: is_enabled = is_trading_enabled
                        if is_enabled:
                            quantity = calculate_position_size(symbol, entry_price, new_signal['stop_loss'])
                            if quantity and quantity > 0:
                                order_result = place_order(symbol, Client.SIDE_BUY, quantity)
                                if order_result:
                                    new_signal.update({'is_real_trade': True, 'quantity': float(quantity), 'order_id': order_result['orderId']})
                                else:
                                    continue
                            else:
                                continue
                        
                        saved_signal = insert_signal_into_db(new_signal)
                        if saved_signal:
                            with signal_cache_lock:
                                open_signals_cache[saved_signal['symbol']] = saved_signal
                            log_and_notify('info', f"SIGNAL: New buy signal for {symbol} at {entry_price}", "NEW_SIGNAL")

                    except Exception as e:
                        logger.error(f"❌ [Processing Error] للعملة {symbol}: {e}", exc_info=True)
                    finally:
                        time.sleep(0.5)

                logger.info(f"🗑️ Batch {i // SYMBOL_PROCESSING_BATCH_SIZE + 1} processed. Clearing caches and collecting garbage.")
                ml_models_cache.clear()
                gc.collect()
                logger.info("✅ Memory cleanup for batch complete.")

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
        Thread(target=main_loop_enhanced, daemon=True).start()
        Thread(target=price_update_loop, daemon=True).start()
        Thread(target=trade_management_loop, daemon=True).start()
        logger.info("✅ [Bot Services] تم بدء جميع الخدمات الخلفية بنجاح.")
        send_telegram_message("✅ *البوت قيد التشغيل الآن (نسخة V5 - متعدد الاستراتيجيات)*")
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حرج أثناء التهيئة: {e}", "SYSTEM"); exit(1)

# ---------------------- نقطة الانطلاق ----------------------
if __name__ == "__main__":
    logger.info("🚀 إطلاق بوت التداول ولوحة التحكم (V5 - Multi-Strategy System) 🚀")
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
