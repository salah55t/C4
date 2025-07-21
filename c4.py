# ملف c4_complete_v9_final_telegram.py - نسخة محدثة مع تحسين الذاكرة ومنطق الدعم والمقاومة
# تم التحديث بواسطة Gemini
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
from scipy.signal import argrelextrema
from tenacity import retry, stop_after_attempt, wait_exponential
import warnings

# --- إعدادات التجاهل واللوجر ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v9_telegram_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV9_Telegram')

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

# --- متغيرات عامة وإعدادات البوت (محدثة لـ V9) ---
is_trading_enabled: bool = False
trading_status_lock = Lock()
are_filters_disabled: bool = False
filters_disabled_lock = Lock()
RISK_PER_TRADE_PERCENT: float = 1.0
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V9_With_Microstructure'
MODEL_FOLDER: str = 'V9'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['15m', '1h', '4h']
# --- MEMORY OPTIMIZATION: Reduced lookback period significantly ---
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 10 # <-- تخفيض المدة من 90 إلى 10 أيام
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices_v9"
TRADING_FEE_PERCENT: float = 0.1
STATS_TRADE_SIZE_USDT: float = 10.0
BTC_SYMBOL: str = 'BTCUSDT'
MAX_OPEN_TRADES: int = 4
BUY_CONFIDENCE_THRESHOLD = 0.50
MIN_PROFIT_PERCENT: float = 1.0

# --- MEMORY OPTIMIZATION: Processing batch size ---
SYMBOL_PROCESSING_BATCH_SIZE: int = 15 # معالجة 15 عملة في كل دفعة

# --- إعدادات المؤشرات الفنية (مطابقة لملف التدريب V9) ---
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
EMA_SLOW_PERIOD: int = 200
EMA_FAST_PERIOD: int = 50
BTC_CORR_PERIOD: int = 30
REL_VOL_PERIOD: int = 30
MOMENTUM_PERIOD: int = 12
EMA_SLOPE_PERIOD: int = 5

# --- إعدادات الفلاتر المتقدمة وإدارة الصفقات ---
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_ACTIVATION_PROFIT_PERCENT: float = 1.0
TRAILING_DISTANCE_PERCENT: float = 0.8
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
enhanced_client: 'EnhancedClient' = None
redis_client: Optional[redis.Redis] = None
# --- MEMORY OPTIMIZATION: Removed ML models cache ---
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
dynamic_filter_profile_cache: Dict[str, Any] = {}
last_dynamic_filter_analysis_time: float = 0
dynamic_filter_lock = Lock()
last_market_state_check = 0

# --- قاموس أسباب الرفض باللغة العربية ---
REJECTION_REASONS_AR = {
    "Filters Not Loaded": "الفلاتر غير محملة", "Low Volatility": "تقلب منخفض جداً",
    "BTC Correlation": "ارتباط ضعيف بالبيتكوين", "RRR Filter": "نسبة المخاطرة/العائد غير كافية",
    "Momentum/Strength Filter": "فلتر الزخم والقوة", "Peak/Pullback Filter": "فلتر القمة/التصحيح",
    "Invalid ATR for TP/SL": "ATR غير صالح لحساب الأهداف", "ML Model Rejected Signal": "نموذج التعلم الآلي رفض الإشارة",
    "Invalid Position Size": "حجم الصفقة غير صالح", "Lot Size Adjustment Failed": "فشل ضبط حجم العقد",
    "Min Notional Filter": "قيمة الصفقة أقل من الحد الأدنى", "Insufficient Balance": "الرصيد غير كافٍ",
    "Order Book Fetch Failed": "فشل جلب دفتر الطلبات", "Order Book Imbalance": "اختلال توازن دفتر الطلبات",
    "Large Sell Wall Detected": "تم كشف جدار بيع ضخم", "Insufficient data for TP/SL calculation": "بيانات غير كافية لحساب TP/SL",
    "Potential Profit Below Threshold": "الربح المحتمل أقل من الحد الأدنى",
    "Potential Profit Below Threshold (S/R)": "الربح المحتمل أقل من الحد الأدنى (دعم/مقاومة)"
}

# --- NEW: EnhancedClient with Retry Mechanism ---
class EnhancedClient:
    def __init__(self, api_key: str, api_secret: str):
        self.client = Client(api_key, api_secret)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def safe_get_symbol_ticker(self, **kwargs) -> Dict:
        return self.client.get_symbol_ticker(**kwargs)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
    def safe_get_historical_klines(self, **kwargs) -> List:
        return self.client.get_historical_klines(**kwargs)
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def safe_get_exchange_info(self) -> Dict:
        return self.client.get_exchange_info()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def safe_get_asset_balance(self, **kwargs) -> Dict:
        return self.client.get_asset_balance(**kwargs)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def safe_create_order(self, **kwargs) -> Dict:
        return self.client.create_order(**kwargs)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def safe_get_order_book(self, **kwargs) -> Dict:
        return self.client.get_order_book(**kwargs)

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
    if not enhanced_client: return
    logger.info("ℹ️ [Exchange Info] جلب قواعد التداول من المنصة...")
    try:
        info = enhanced_client.safe_get_exchange_info()
        exchange_info_map = {s['symbol']: s for s in info['symbols']}
        exchange_info_map['_timestamp'] = time.time()
        logger.info(f"✅ [Exchange Info] تم تحميل القواعد لـ {len(exchange_info_map)} عملة.")
    except Exception as e:
        logger.error(f"❌ [Exchange Info] لم يتمكن من جلب معلومات المنصة: {e}")

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    if not enhanced_client: return []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        
        if not exchange_info_map or time.time() - exchange_info_map.get('_timestamp', 0) > 3600:
            get_exchange_info_map()

        active = {
            s for s, info in exchange_info_map.items() 
            if isinstance(info, dict) and info.get('quoteAsset') == 'USDT' and info.get('status') == 'TRADING'
        }
        
        validated = sorted(list(formatted.intersection(active)))
        
        if not validated:
            logger.warning("⚠️ [Validation] لم يتم العثور على عملات صالحة بعد الفلترة.")
        else:
            logger.info(f"✅ [Validation] سيقوم البوت بمراقبة {len(validated)} عملة.")
        
        return validated
        
    except FileNotFoundError:
        logger.error(f"❌ [Validation] ملف العملات '{filename}' غير موجود في المسار: {file_path}")
        return []
    except Exception as e:
        logger.error(f"❌ [Validation] خطأ أثناء التحقق من العملات: {e}", exc_info=True)
        return []

# --- دوال جلب البيانات وحساب الميزات ---
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not enhanced_client: return None
    try:
        start_dt = datetime.now(timezone.utc) - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        klines = enhanced_client.safe_get_historical_klines(symbol=symbol, interval=interval, start_str=start_str)
        if not klines: return None
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(klines, columns=cols)
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base']
        df = df[required_cols]
        numeric_cols = {'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float', 'quote_volume': 'float', 'taker_buy_base': 'float'}
        df = df.astype(numeric_cols)
        df = MemoryManager.optimize_pandas_df(df)
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

def calculate_all_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df_calc = df.copy()
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
    df_calc['price_vs_ema50'] = (df_calc['close'] / df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()) - 1
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

# ---------------------- الفئات الجديدة المضافة ----------------------

class MemoryManager:
    @staticmethod
    def optimize_pandas_df(df: pd.DataFrame) -> pd.DataFrame:
        """تحسين استخدام الذاكرة لداتافريم pandas"""
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        return df

class AdvancedStrategies:
    @staticmethod
    def calculate_fibonacci_levels(df: pd.DataFrame) -> Dict[str, float]:
        if df.empty or len(df) < 50: return {}
        recent_high = df['high'].tail(50).max()
        recent_low = df['low'].tail(50).min()
        diff = recent_high - recent_low
        if diff == 0: return {}
        return {
            '0.0': recent_low, '23.6': recent_low + diff * 0.236,
            '38.2': recent_low + diff * 0.382, '50.0': recent_low + diff * 0.5,
            '61.8': recent_low + diff * 0.618, '78.6': recent_low + diff * 0.786,
            '100.0': recent_high
        }
    
    @staticmethod
    def detect_divergences(df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty or 'close' not in df.columns or 'rsi' not in df.columns:
            return {'divergences': []}
        divergences = []
        price_peaks = argrelextrema(df['close'].values, np.greater, order=5)[0]
        rsi_peaks = argrelextrema(df['rsi'].values, np.greater, order=5)[0]
        if len(price_peaks) < 2 or len(rsi_peaks) < 2:
            return {'divergences': divergences}
        last_price_peak_idx, prev_price_peak_idx = price_peaks[-1], price_peaks[-2]
        last_rsi_peak_idx, prev_rsi_peak_idx = rsi_peaks[-1], rsi_peaks[-2]
        if df['close'].iloc[last_price_peak_idx] > df['close'].iloc[prev_price_peak_idx] and \
           df['rsi'].iloc[last_rsi_peak_idx] < df['rsi'].iloc[prev_rsi_peak_idx]:
            divergences.append({
                'type': 'bearish_rsi_divergence', 'strength': 'strong',
                'price_peak_1': df['close'].iloc[prev_price_peak_idx], 'price_peak_2': df['close'].iloc[last_price_peak_idx],
                'rsi_peak_1': df['rsi'].iloc[prev_rsi_peak_idx], 'rsi_peak_2': df['rsi'].iloc[last_rsi_peak_idx]
            })
        return {'divergences': divergences}
    
    @staticmethod
    def volume_profile_analysis(df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty or 'close' not in df.columns or 'volume' not in df.columns: return {}
        try:
            price_bins = pd.cut(df['close'], bins=20)
            volume_profile = df.groupby(price_bins)['volume'].sum()
            if volume_profile.empty: return {}
            poc_interval = volume_profile.idxmax()
            poc = poc_interval.mid
            total_volume = volume_profile.sum()
            if total_volume == 0: return {'poc': poc}
            sorted_profile = volume_profile.sort_values(ascending=False)
            cumulative_volume = sorted_profile.cumsum()
            value_area_mask = cumulative_volume <= total_volume * 0.7
            value_area_intervals = sorted_profile[value_area_mask].index.tolist()
            if not value_area_intervals: return {'poc': poc}
            value_area_high = max(interval.right for interval in value_area_intervals)
            value_area_low = min(interval.left for interval in value_area_intervals)
            return {
                'poc': poc, 'value_area_high': value_area_high, 'value_area_low': value_area_low,
                'imbalances': [interval.mid for interval in volume_profile[volume_profile < volume_profile.mean() * 0.5].index.tolist()]
            }
        except Exception as e:
            logger.error(f"Error in volume profile analysis: {e}")
            return {}

class RiskManagementSystem:
    def __init__(self):
        self.daily_loss_limit = 0.05
        self.max_positions_per_sector = 2
        self.correlation_threshold = 0.8
    
    def calculate_portfolio_risk(self) -> Dict[str, float]:
        if not redis_client or not enhanced_client:
            return {'total_exposure': 0, 'daily_pnl': 0, 'risk_score': 0}
        try:
            with signal_cache_lock:
                open_positions = [p for p in open_signals_cache.values() if p.get('is_real_trade')]
            if not open_positions:
                return {'total_exposure': 0, 'daily_pnl': 0, 'risk_score': 0}
            current_prices = redis_client.hgetall(REDIS_PRICES_HASH_NAME)
            total_exposure, daily_pnl = 0, 0
            for pos in open_positions:
                current_price_str = current_prices.get(pos['symbol'])
                if not current_price_str: continue
                current_price = float(current_price_str)
                entry_price = float(pos['entry_price'])
                quantity = float(pos['quantity'])
                position_value = quantity * current_price
                total_exposure += position_value
                daily_pnl += (current_price - entry_price) * quantity
            usdt_balance = float(enhanced_client.safe_get_asset_balance(asset='USDT')['free'])
            total_portfolio_value = usdt_balance + total_exposure
            risk_score = 0
            if total_portfolio_value > 0 and daily_pnl < 0:
                loss_percentage = abs(daily_pnl) / total_portfolio_value
                risk_score = min(loss_percentage / self.daily_loss_limit, 1.0)
            return {'total_exposure': total_exposure, 'daily_pnl': daily_pnl, 'risk_score': risk_score}
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return {'total_exposure': 0, 'daily_pnl': 0, 'risk_score': 0}
    
    def adjust_position_size_based_on_risk(self, symbol: str, base_size: Decimal) -> Decimal:
        risk_metrics = self.calculate_portfolio_risk()
        risk_score = risk_metrics.get('risk_score', 0)
        if risk_score > 0.7:
            logger.warning(f"RISK-ADJUST: High risk ({risk_score:.2f}). Reducing size for {symbol} by 50%.")
            return base_size * Decimal('0.5')
        elif risk_score > 0.4:
            logger.info(f"RISK-ADJUST: Moderate risk ({risk_score:.2f}). Reducing size for {symbol} by 25%.")
            return base_size * Decimal('0.75')
        return base_size

# ---------------------- أنظمة التحليل المتقدمة ----------------------
class MarketConditionsAnalyzer:
    def __init__(self):
        self.conditions_cache = {}
        self.last_analysis = 0
    def analyze_conditions(self) -> Dict[str, Any]:
        if time.time() - self.last_analysis < 300: return self.conditions_cache
        try:
            conditions = {
                'volatility_regime': self._get_volatility_regime(), 
                'volume_regime': self._get_volume_regime(), 
                'correlation_regime': self._get_correlation_regime(), 
                'session_type': self._get_session_type()
            }
            self.conditions_cache = conditions; self.last_analysis = time.time()
            return conditions
        except Exception as e:
            logger.error(f"❌ [Market Conditions] خطأ: {e}"); return self._get_default_conditions()
    def _get_volatility_regime(self) -> str:
        try:
            btc_data = fetch_historical_data(BTC_SYMBOL, '1h', 7)
            if btc_data is None: return "normal"
            volatility = btc_data['close'].pct_change().rolling(24).std().iloc[-1] * np.sqrt(24 * 365) * 100
            if volatility < 20: return "low"
            elif volatility < 60: return "normal"
            else: return "high"
        except: return "normal"
    def _get_volume_regime(self) -> str:
        try:
            btc_data = fetch_historical_data(BTC_SYMBOL, '1h', 7)
            if btc_data is None: return "normal"
            ratio = btc_data['volume'].iloc[-1] / btc_data['volume'].rolling(24).mean().iloc[-1]
            if ratio < 0.7: return "low"
            elif ratio < 1.5: return "normal"
            else: return "high"
        except: return "normal"
    def _get_correlation_regime(self) -> str: return "normal"
    def _get_session_type(self) -> str: return get_session_state()[1]
    def _get_default_conditions(self) -> Dict[str, Any]: return {'volatility_regime': 'normal', 'volume_regime': 'normal', 'correlation_regime': 'normal', 'session_type': 'NORMAL_LIQUIDITY'}

class EnhancedFilterSystem:
    def __init__(self): self.analyzer = MarketConditionsAnalyzer()
    def generate_filters(self) -> Dict[str, Any]:
        conditions = self.analyzer.analyze_conditions()
        base_profile = {"adx": 25.0, "rel_vol": 0.4, "rsi_range": (52, 88), "roc": 0.05, "slope": 0.01, "min_rrr": 1.4, "min_volatility_pct": 0.35, "min_btc_correlation": 0.4, "min_bid_ask_ratio": 1.15}
        if conditions['volatility_regime'] == "low": base_profile['min_volatility_pct'] *= 0.7; base_profile['min_rrr'] *= 1.2
        elif conditions['volatility_regime'] == "high": base_profile['min_volatility_pct'] *= 1.3; base_profile['min_rrr'] *= 0.8
        if conditions['volume_regime'] == "low": base_profile['rel_vol'] *= 0.5
        elif conditions['volume_regime'] == "high": base_profile['rel_vol'] *= 1.2
        return {"name": f"فلاتر متكيفة - {conditions['volatility_regime']}", "description": f"نظام متكيف: {conditions['volatility_regime']}/{conditions['volume_regime']}", "strategy": "MOMENTUM", "filters": base_profile, "conditions": conditions}

enhanced_filter_system = EnhancedFilterSystem()
risk_management_system = RiskManagementSystem()

# ---------------------- استراتيجية التداول والفلاتر (محدثة لـ V9) ----------------------
class EnhancedTradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = self._load_ml_model_from_file(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    def _load_ml_model_from_file(self, symbol: str) -> Optional[Dict[str, Any]]:
        # --- MEMORY OPTIMIZATION: Load model directly without caching ---
        model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir_path = os.path.join(script_dir, MODEL_FOLDER)
        model_path = os.path.join(model_dir_path, f"{model_name}.pkl")
        
        if not os.path.exists(model_path): return None
        
        try:
            with open(model_path, 'rb') as f: model_bundle = pickle.load(f)
            if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
                logger.info(f"✅ [{self.symbol}] تم تحميل نموذج V9 من ملف (بدون كاش).")
                return model_bundle
            return None
        except Exception as e:
            logger.error(f"❌ [ML Model File] خطأ في تحميل النموذج لـ {symbol}: {e}")
            return None

    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.feature_names is None: return None
        try:
            df_featured = calculate_all_features(df_15m, btc_df)
            df_4h_features = calculate_all_features(df_4h, None)
            df_4h_features = df_4h_features.rename(columns=lambda c: f"{c}_4h")
            required_4h_cols = ['rsi_4h', 'price_vs_ema50_4h']
            df_featured = df_featured.join(df_4h_features[required_4h_cols], how='left')
            df_featured[required_4h_cols] = df_featured[required_4h_cols].fillna(method='ffill')
            for col in self.feature_names:
                if col not in df_featured.columns: df_featured[col] = 0.0
            df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_featured.dropna(subset=self.feature_names)
        except Exception as e:
            logger.error(f"❌ [{self.symbol}] فشل هندسة الميزات لـ V9: {e}", exc_info=True)
            return None

    def generate_buy_signal(self, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
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
            logger.warning(f"⚠️ [{self.symbol}] خطأ في توليد إشارة النموذج: {e}")
            return None

def passes_filters(symbol: str, last_features: pd.Series, profile: Dict[str, Any], entry_price: float, tp_sl_data: Dict, df_15m: pd.DataFrame) -> bool:
    with filters_disabled_lock:
        if are_filters_disabled:
            logger.warning(f"⚠️ [{symbol}] تجاوز الفلاتر بسبب الإعداد العام.")
            return True
            
    filters = profile.get("filters", {})
    if not filters: log_rejection(symbol, "Filters Not Loaded"); return False
    volatility = (last_features.get('atr', 0) / entry_price * 100) if entry_price > 0 else 0
    if volatility < filters.get('min_volatility_pct', 0.0): log_rejection(symbol, "Low Volatility", {"volatility": f"{volatility:.2f}%"}); return False
    correlation = last_features.get('btc_correlation', 0)
    if correlation < filters.get('min_btc_correlation', -1.0): log_rejection(symbol, "BTC Correlation", {"corr": f"{correlation:.2f}"}); return False
    risk = entry_price - float(tp_sl_data['stop_loss']); reward = float(tp_sl_data['target_price']) - entry_price
    if risk <= 0 or reward <= 0 or (reward / risk) < filters.get('min_rrr', 0.0): log_rejection(symbol, "RRR Filter", {"rrr": f"{(reward/risk):.2f}" if risk > 0 else "N/A"}); return False
    adx, rel_vol, rsi, roc, slope = last_features.get('adx', 0), last_features.get('relative_volume', 0), last_features.get('rsi', 0), last_features.get(f'roc_{MOMENTUM_PERIOD}', 0), last_features.get(f'ema_slope_{EMA_SLOPE_PERIOD}', 0)
    rsi_min, rsi_max = filters.get('rsi_range', (0, 100))
    if not (adx >= filters.get('adx', 0) and rel_vol >= filters.get('rel_vol', 0) and rsi_min <= rsi < rsi_max and roc > filters.get('roc', -100) and slope > filters.get('slope', -100)):
        log_rejection(symbol, "Momentum/Strength Filter", {"ADX": f"{adx:.2f}", "RSI": f"{rsi:.2f}"}); return False
    if USE_PEAK_FILTER and df_15m is not None and len(df_15m) >= PEAK_CHECK_PERIOD:
        recent_candles = df_15m.iloc[-PEAK_CHECK_PERIOD:-1]
        if not recent_candles.empty:
            highest_high = recent_candles['high'].max()
            with market_state_lock: is_strong_uptrend = (current_market_state.get("overall_regime") == "STRONG_UPTREND")
            price_limit = highest_high * (BREAKOUT_ALLOWANCE_PCT if is_strong_uptrend else PULLBACK_THRESHOLD_PCT)
            if not (entry_price <= price_limit): log_rejection(symbol, "Peak/Pullback Filter", {"entry": f"{entry_price:.4f}", "limit": f"{price_limit:.4f}"}); return False
    return True

def analyze_order_book(symbol: str, entry_price: float) -> Optional[Dict[str, Any]]:
    if not enhanced_client: return None
    try:
        order_book = enhanced_client.safe_get_order_book(symbol=symbol, limit=ORDER_BOOK_DEPTH_LIMIT)
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
        if are_filters_disabled:
            return True
    filters = profile.get("filters", {})
    if order_book_analysis.get('has_large_sell_wall', True): log_rejection(symbol, "Large Sell Wall Detected", {"details": order_book_analysis.get('wall_details')}); return False
    if order_book_analysis.get('bid_ask_ratio', 0) < filters.get('min_bid_ask_ratio', 1.0): log_rejection(symbol, "Order Book Imbalance", {"ratio": f"{order_book_analysis.get('bid_ask_ratio', 0):.2f}"}); return False
    return True

SR_LOOKBACK_CANDLES = 50
SR_SWING_THRESHOLD  = 0.02

def find_sr_levels(df: pd.DataFrame, lookback: int = 50, swing_threshold: float = 0.02) -> Dict[str, Optional[float]]:
    if len(df) < lookback: return {'support': None, 'resistance': None}
    df = df.iloc[-lookback:].copy()
    highs, lows = df['high'], df['low']
    resistance_candidates, support_candidates = [], []
    for i in range(2, len(df) - 2):
        if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
            if (highs.iloc[i] - max(highs.iloc[i-2], highs.iloc[i+2])) / highs.iloc[i] > swing_threshold:
                resistance_candidates.append(highs.iloc[i])
        if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
            if (min(lows.iloc[i-2], lows.iloc[i+2]) - lows.iloc[i]) / lows.iloc[i] > swing_threshold:
                support_candidates.append(lows.iloc[i])
    current_price = df['close'].iloc[-1]
    resistance, support = None, None
    if resistance_candidates:
        above = [r for r in resistance_candidates if r > current_price]
        resistance = min(above) if above else max(resistance_candidates)
    if support_candidates:
        below = [s for s in support_candidates if s < current_price]
        support = max(below) if below else min(support_candidates)
    return {'support': support, 'resistance': resistance}

def calculate_tp_sl(symbol: str, entry_price: float, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    try:
        if df.empty or len(df) < 50:
            log_rejection(symbol, "Insufficient data for TP/SL calculation")
            return None
        sr = find_sr_levels(df, lookback=SR_LOOKBACK_CANDLES, swing_threshold=SR_SWING_THRESHOLD)
        resistance, support = sr['resistance'], sr['support']
        if resistance is None or support is None:
            last_atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0
            if last_atr <= 0: return None
            resistance = entry_price + last_atr * 2.5
            support = entry_price - last_atr * 1.5
        potential_profit_pct = ((resistance - entry_price) / entry_price) * 100
        if potential_profit_pct < MIN_PROFIT_PERCENT:
            log_rejection(symbol, "Potential Profit Below Threshold (S/R)", {"potential_profit": f"{potential_profit_pct:.2f}%"})
            return None
        if support >= entry_price: support = entry_price * 0.98
        if ((entry_price - support) / entry_price) * 100 < 0.3: support = entry_price * 0.997
        return {
            'target_price': round(resistance, 6), 'stop_loss': round(support, 6),
            'source': 'SR_LEVELS_V2',
            'rr_ratio': round((resistance - entry_price) / (entry_price - support), 2) if (entry_price - support) > 0 else 0
        }
    except Exception as e:
        logger.error(f"❌ [{symbol}] Error in S/R TP/SL: {e}", exc_info=True)
        last_atr = df['atr'].iloc[-1] if 'atr' in df.columns and not df['atr'].empty else 0
        if last_atr > 0:
            return {'target_price': entry_price + last_atr * 2.2, 'stop_loss': entry_price - last_atr * 1.5, 'source': 'ATR_Fallback'}
        return None

# ---------------------- دوال إدارة الصفقات ----------------------
def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> Optional[Decimal]:
    try:
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info: return None
        for f in symbol_info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                step_size = Decimal(f['stepSize'])
                return (Decimal(str(quantity)) // step_size) * step_size
        return Decimal(str(quantity))
    except Exception as e:
        logger.error(f"[{symbol}] خطأ في تعديل الكمية: {e}"); return None

def calculate_position_size(symbol: str, entry_price: float, stop_loss_price: float) -> Optional[Decimal]:
    if not enhanced_client: return None
    try:
        balance_response = enhanced_client.safe_get_asset_balance(asset='USDT')
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
    if not enhanced_client: return None
    logger.info(f"➡️ [{symbol}] محاولة تنفيذ أمر {side} حقيقي لكمية {quantity}.")
    try:
        order = enhanced_client.safe_create_order(symbol=symbol, side=side, type=order_type, quantity=float(quantity))
        log_and_notify('info', f"TRADE REAL: Placed {side} order for {quantity} {symbol}.", "REAL_TRADE")
        return order
    except Exception as e:
        logger.error(f"❌ [{symbol}] خطأ من باينانس عند تنفيذ الأمر: {e}")
        log_and_notify('error', f"REAL TRADE FAILED: {symbol} | {e}", "REAL_TRADE_ERROR")
        return None

def close_signal(signal_id: int, closing_price: float, reason: str) -> bool:
    with signal_cache_lock:
        signal_to_close, symbol_to_close = None, None
        for symbol, signal_data in open_signals_cache.items():
            if signal_data['id'] == signal_id:
                signal_to_close, symbol_to_close = signal_data, symbol
                break
        if not signal_to_close:
            logger.warning(f"⚠️ [Close] محاولة إغلاق صفقة غير موجودة في الكاش ID: {signal_id}")
            return False
        entry_price = float(signal_to_close['entry_price'])
        profit_percentage = ((closing_price - entry_price) / entry_price) * 100
        if signal_to_close.get('is_real_trade'):
            quantity_to_sell = Decimal(str(signal_to_close.get('quantity')))
            if quantity_to_sell > 0:
                if not place_order(symbol_to_close, Client.SIDE_SELL, quantity_to_sell):
                    log_and_notify('error', f"CRITICAL: Failed to place SELL order for {symbol_to_close}. Signal remains open.", "TRADE_ERROR")
                    return False
        if not check_db_connection() or not conn:
            log_and_notify('critical', "DB connection lost during trade closure.", "DB_ERROR")
            return False
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE signals SET status = 'closed', closing_price = %s, closed_at = NOW(), profit_percentage = %s, closing_reason = %s WHERE id = %s;", (closing_price, profit_percentage, reason, signal_id))
            conn.commit()
            if symbol_to_close in open_signals_cache: del open_signals_cache[symbol_to_close]
            log_and_notify('info', f"CLOSED: {symbol_to_close} at {closing_price:.4f}. Reason: {reason}. Profit: {profit_percentage:.2f}%", "TRADE_CLOSED")
            reason_map = {'take_profit': '🎯 Take Profit', 'stop_loss': '🛑 Stop Loss', 'manual': '🖐️ Manual Close'}
            emoji = "✅" if profit_percentage >= 0 else "🔻"
            trade_type = "حقيقية" if signal_to_close.get('is_real_trade') else "تجريبية"
            telegram_message = (f"{emoji} *إغلاق صفقة {trade_type}*\n\n"
                              f"*العملة:* `{symbol_to_close}`\n*سبب الإغلاق:* {reason_map.get(reason, reason)}\n"
                              f"*سعر الدخول:* `{entry_price:.4f}`\n*سعر الإغلاق:* `{closing_price:.4f}`\n"
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
            entry_price = float(signal_data['entry_price'])
            target_price = float(signal_data['target_price'])
            stop_loss = float(signal_data['stop_loss'])
            quantity = float(signal_data['quantity']) if signal_data.get('quantity') is not None else None
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, is_real_trade, quantity, order_id, current_peak_price, closing_reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL) RETURNING *;
            """, (signal_data['symbol'], entry_price, target_price, stop_loss, signal_data['strategy_name'], json.dumps(signal_data['signal_details']), signal_data.get('is_real_trade', False), quantity, signal_data.get('order_id'), entry_price))
            saved_signal = cur.fetchone()
            conn.commit()
            logger.info(f"💾 [{signal_data['symbol']}] تم حفظ الإشارة الجديدة في قاعدة البيانات.")
            trade_type = "حقيقية" if signal_data.get('is_real_trade') else "تجريبية"
            telegram_message = (f"💡 *توصية شراء {trade_type} جديدة*\n\n"
                              f"*العملة:* `{signal_data['symbol']}`\n*سعر الدخول:* `{entry_price:.4f}`\n"
                              f"*الهدف (TP):* `{target_price:.4f}`\n*وقف الخسارة (SL):* `{stop_loss:.4f}`\n\n"
                              f"Confidence: {signal_data['signal_details'].get('ML_Confidence', 'N/A')}")
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
        trend_details, advanced_analysis = {}, {}
        tf_for_advanced = '4h'
        df_advanced = fetch_historical_data(BTC_SYMBOL, tf_for_advanced, 90)
        if df_advanced is not None and not df_advanced.empty:
            df_advanced_features = calculate_all_features(df_advanced, None)
            advanced_analysis['fibonacci'] = AdvancedStrategies.calculate_fibonacci_levels(df_advanced_features)
            advanced_analysis['divergences'] = AdvancedStrategies.detect_divergences(df_advanced_features)
            advanced_analysis['volume_profile'] = AdvancedStrategies.volume_profile_analysis(df_advanced_features)
            logger.info(f"📊 [Advanced Analysis] Fib: {advanced_analysis['fibonacci'] is not None}, Div: {len(advanced_analysis['divergences']['divergences']) > 0}, VP: {advanced_analysis['volume_profile'] is not None}")
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
            current_market_state = {
                "overall_regime": overall_regime.upper().replace(" ", "_"), 
                "trend_details_by_tf": trend_details, 
                "advanced_analysis": json.loads(json.dumps(advanced_analysis, default=str)),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            last_market_state_check = time.time()
        logger.info(f"✅ [Market State] الحالة العامة: {overall_regime}")
    except Exception as e:
        logger.error(f"❌ [Market State] خطأ: {e}", exc_info=True)

def analyze_market_and_create_dynamic_profile_enhanced():
    global dynamic_filter_profile_cache, last_dynamic_filter_analysis_time
    if time.time() - last_dynamic_filter_analysis_time < DYNAMIC_FILTER_ANALYSIS_INTERVAL: return
    logger.info("🔬 [Filter] توليد فلاتر متكيفة...")
    enhanced_profile = enhanced_filter_system.generate_filters()
    with dynamic_filter_lock:
        dynamic_filter_profile_cache = {
            "name": enhanced_profile['description'], "description": enhanced_profile['description'],
            "strategy": "MOMENTUM", "filters": enhanced_profile['filters'],
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        last_dynamic_filter_analysis_time = time.time()
    logger.info(f"✅ [Filter] تم توليد فلاتر جديدة: {enhanced_profile['description']}")

# ---------------------- واجهة Flask ----------------------
app = Flask(__name__)
CORS(app)

def get_dashboard_html():
    # The HTML content remains the same as it's for the frontend.
    # No changes needed here for memory optimization.
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم التداول V9</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root { --bg-main: #0D1117; --bg-card: #161B22; --border-color: #30363D; --text-primary: #E6EDF3; --text-secondary: #848D97; --accent-blue: #58A6FF; --accent-green: #3FB950; --accent-red: #F85149; --accent-yellow: #D29922; }
        body { font-family: 'Tajawal', sans-serif; background-color: var(--bg-main); color: var(--text-primary); }
        .card { background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 0.5rem; }
        .trend-light { width: 1rem; height: 1rem; border-radius: 9999px; border: 2px solid #30363D; transition: all 0.5s ease; }
        .light-on-green { background-color: var(--accent-green); box-shadow: 0 0 10px 2px var(--accent-green); }
        .light-on-red { background-color: var(--accent-red); box-shadow: 0 0 10px 2px var(--accent-red); }
        .light-on-yellow { background-color: var(--accent-yellow); box-shadow: 0 0 10px 2px var(--accent-yellow); }
        .tab-btn.active { border-bottom-color: var(--accent-blue); color: var(--text-primary); }
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
            <h1 class="text-2xl md:text-3xl font-extrabold"><span class="text-accent-blue">لوحة تحكم</span><span class="text-text-secondary font-medium"> V9</span></h1>
            <div id="trend-lights-container" class="flex items-center gap-x-6 bg-black/20 px-4 py-2 rounded-lg border border-border-color"></div>
        </header>
        <section class="mb-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-5">
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">حالة السوق</h3><div id="overall-regime" class="text-2xl font-bold text-center">...</div></div>
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">ملف الفلاتر</h3><div id="filter-profile-name" class="text-xl font-bold text-center">...</div></div>
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">الجلسات النشطة</h3><div id="active-sessions-list" class="flex flex-wrap gap-2 items-center justify-center pt-2">...</div></div>
            <div class="card p-4 flex flex-col justify-center items-center"><h3 class="font-bold text-lg text-text-secondary mb-2">التداول الحقيقي</h3><div class="flex items-center space-x-3 space-x-reverse"><span id="trading-status-text" class="font-bold text-lg"></span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="trading-toggle" class="sr-only" onchange="toggleTrading()"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div><div class="mt-2 text-xs text-text-secondary">رصيد USDT: <span id="usdt-balance" class="font-mono">...</span></div></div>
            <div class="card p-4 flex flex-col justify-center items-center"><h3 class="font-bold text-lg text-text-secondary mb-2">تعطيل الفلاتر</h3><div class="flex items-center space-x-3 space-x-reverse"><span id="disable-filters-text" class="font-bold text-lg"></span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="disable-filters-toggle" class="sr-only" onchange="toggleFilters()"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div></div>
        </section>
        <div class="mb-4 border-b border-border-color"><nav class="flex space-x-6 space-x-reverse -mb-px">
            <button onclick="showTab('signals', this)" class="tab-btn active text-white py-3 px-1 font-semibold">الصفقات</button>
            <button onclick="showTab('stats', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الإحصائيات</button>
            <button onclick="showTab('advanced-analytics', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">التحليلات المتقدمة</button>
            <button onclick="showTab('notifications', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الإشعارات</button>
            <button onclick="showTab('rejections', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الصفقات المرفوضة</button>
            <button onclick="showTab('filters', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الفلاتر الحالية</button>
        </nav></div>
        <main>
            <div id="signals-tab" class="tab-content"><div class="overflow-x-auto card p-0"><table class="min-w-full text-sm text-right"><thead class="border-b border-border-color bg-black/20"><tr><th class="p-4 font-semibold">العملة</th><th class="p-4 font-semibold">الحالة</th><th class="p-4 font-semibold">الربح/الخسارة</th><th class="p-4 font-semibold w-[25%]">التقدم</th><th class="p-4 font-semibold">الدخول/الحالي</th><th class="p-4 font-semibold">إجراء</th></tr></thead><tbody id="signals-table"></tbody></table></div></div>
            <div id="stats-tab" class="tab-content hidden"><div id="stats-container" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"></div></div>
            <div id="notifications-tab" class="tab-content hidden"><div id="notifications-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-2"></div></div>
            <div id="rejections-tab" class="tab-content hidden"><div id="rejections-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-2"></div></div>
            <div id="filters-tab" class="tab-content hidden"><div id="filters-display" class="card p-4 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4"></div></div>
            <div id="advanced-analytics-tab" class="tab-content hidden">
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
                    <div class="card p-4"><h3 class="font-bold mb-2">الأداء اليومي (USDT)</h3><div style="height: 250px;"><canvas id="dailyPnlChart"></canvas></div></div>
                    <div class="card p-4"><h3 class="font-bold mb-2">توزيع المحفظة</h3><div style="height: 250px;"><canvas id="portfolioDistribution"></canvas></div></div>
                    <div class="card p-4"><h3 class="font-bold mb-2">مؤشرات السوق</h3><div id="marketIndicators" class="space-y-4 pt-4">
                        <div class="flex justify-between items-center"><span class="text-text-secondary">مؤشر الخوف والطمع:</span><span id="fearGreedIndex" class="font-bold text-xl">-</span></div>
                        <div class="flex justify-between items-center"><span class="text-text-secondary">تقلب البيتكوين (30 يوم):</span><span id="btcVolatility" class="font-bold text-xl">-</span></div>
                        <div class="flex justify-between items-center"><span class="text-text-secondary">مخاطر المحفظة الحالية:</span><span id="portfolioRisk" class="font-bold text-xl">-</span></div>
                    </div></div>
                </div>
            </div>
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
    document.querySelectorAll('.tab-btn').forEach(b => { b.classList.remove('active', 'text-white'); b.classList.add('text-text-secondary'); });
    el.classList.add('active', 'text-white');
    el.classList.remove('text-text-secondary');
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
        document.getElementById('filter-profile-name').textContent = data.filter_profile?.name || 'غير متاح';
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
            tableBody.innerHTML += `<tr class="border-b border-border-color hover:bg-white/5"><td class="p-4 font-bold">${s.symbol}</td><td class="p-4"><span class="px-2 py-1 text-xs font-semibold rounded-full ${s.is_real_trade ? 'bg-blue-500/20 text-blue-400' : 'bg-yellow-500/20 text-yellow-400'}">${s.is_real_trade ? 'حقيقي' : 'تجريبي'}</span></td><td class="p-4 font-mono ${pClass}">${profit.toFixed(2)}%</td><td class="p-4"><div class="w-full bg-gray-700 rounded-full h-2.5"><div class="bg-accent-blue h-2.5 rounded-full" style="width: ${progress}%"></div></div></td><td class="p-4 font-mono">${current.toFixed(4)} / ${entry.toFixed(4)}</td><td class="p-4"><button onclick="manualClose(${s.id}, '${s.symbol}')" class="bg-red-600 hover:bg-red-700 text-white font-bold py-1 px-3 rounded text-xs">إغلاق</button></td></tr>`;
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
function updateFilters() {
     fetchData('/api/market_status').then(data => {
        if (!data?.filter_profile?.filters) return;
        document.getElementById('filters-display').innerHTML = Object.entries(data.filter_profile.filters).map(([k, v]) => `<div class="card p-3 bg-black/20"><div class="text-sm text-text-secondary">${k}</div><div class="font-bold text-lg text-accent-blue">${Array.isArray(v) ? `(${v.join(', ')})` : v}</div></div>`).join('');
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

class AdvancedDashboard {
    constructor() {
        this.charts = {};
        this.initCharts();
    }
    
    initCharts() {
        const pnlCtx = document.getElementById('dailyPnlChart').getContext('2d');
        this.charts.dailyPnl = new Chart(pnlCtx, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'الربح اليومي (USDT)', data: [], borderColor: '#3FB950', backgroundColor: 'rgba(63, 185, 80, 0.1)', fill: true, tension: 0.4 }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { ticks: { color: '#848D97' } }, x: { ticks: { color: '#848D97' } } } }
        });
        
        const portfolioCtx = document.getElementById('portfolioDistribution').getContext('2d');
        this.charts.portfolio = new Chart(portfolioCtx, {
            type: 'doughnut',
            data: { labels: [], datasets: [{ data: [], backgroundColor: ['#58A6FF', '#3FB950', '#F85149', '#D29922', '#A371F7', '#E6EDF3'] }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { color: '#848D97' } } } }
        });
    }
    
    async updateAdvancedMetrics() {
        const data = await fetchData('/api/advanced_metrics');
        if (!data) return;
        
        document.getElementById('fearGreedIndex').textContent = data.fear_greed ? `${data.fear_greed.value} (${data.fear_greed.value_classification})` : '-';
        document.getElementById('btcVolatility').textContent = data.btc_volatility ? `${data.btc_volatility.toFixed(2)}%` : '-';
        document.getElementById('portfolioRisk').textContent = data.portfolio_risk ? `${(data.portfolio_risk.risk_score * 100).toFixed(1)}%` : '-';

        if (data.daily_pnl_history) {
            this.charts.dailyPnl.data.labels = data.daily_pnl_history.map(d => d.date);
            this.charts.dailyPnl.data.datasets[0].data = data.daily_pnl_history.map(d => d.pnl);
            this.charts.dailyPnl.update();
        }
        
        if (data.portfolio_distribution && Object.keys(data.portfolio_distribution).length > 0) {
            this.charts.portfolio.data.labels = Object.keys(data.portfolio_distribution);
            this.charts.portfolio.data.datasets[0].data = Object.values(data.portfolio_distribution);
            this.charts.portfolio.update();
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    ['MarketStatus', 'Signals', 'Stats', 'Notifications', 'Rejections', 'Filters'].forEach(f => window[`update${f}`]());
    setInterval(updateMarketStatus, 5000); 
    setInterval(updateSignals, 7000); 
    setInterval(updateStats, 60000);
    setInterval(updateNotifications, 15000); 
    setInterval(updateRejections, 15000); 
    setInterval(updateFilters, 60000);

    const advancedDashboard = new AdvancedDashboard();
    advancedDashboard.updateAdvancedMetrics();
    setInterval(() => advancedDashboard.updateAdvancedMetrics(), 30000);
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
    with dynamic_filter_lock: profile_copy = dict(dynamic_filter_profile_cache)
    active_sessions, _, _ = get_session_state()
    usdt_balance = None
    if enhanced_client:
        try: usdt_balance = float(enhanced_client.safe_get_asset_balance(asset='USDT')['free'])
        except: usdt_balance = 'N/A'
    return jsonify({
        "market_state": state_copy, "filter_profile": profile_copy, 
        "active_sessions": active_sessions, "usdt_balance": usdt_balance, 
        "is_trading_enabled": is_enabled, "are_filters_disabled": is_disabled
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
            for t in closed_trades)
        wins = [float(s['profit_percentage']) for s in closed_trades if float(s['profit_percentage']) > 0]
        losses = [float(s['profit_percentage']) for s in closed_trades if float(s['profit_percentage']) < 0]
        win_rate = (len(wins) / len(closed_trades) * 100) if closed_trades else 0.0
        total_loss = abs(sum(losses))
        profit_factor = sum(wins) / total_loss if total_loss > 0 else "Infinity"
        return jsonify({
            "net_profit_usdt": total_net_profit_usdt, "win_rate": win_rate, 
            "profit_factor": profit_factor, "total_closed_trades": len(closed_trades)
        })
    except Exception as e:
        logger.error(f"❌ [API Stats] Error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error fetching stats"}), 500

@app.route('/api/advanced_metrics')
def get_advanced_metrics():
    fear_greed, btc_volatility, daily_pnl_history = None, None, []
    try:
        response = requests.get('https://api.alternative.me/fng/?limit=1', timeout=5)
        response.raise_for_status()
        fear_greed = response.json()['data'][0]
    except Exception as e: logger.warning(f"Could not fetch Fear & Greed Index: {e}")
    try:
        btc_data = fetch_historical_data(BTC_SYMBOL, '1d', 30)
        if btc_data is not None: btc_volatility = btc_data['close'].pct_change().std() * np.sqrt(365) * 100
    except Exception as e: logger.warning(f"Could not calculate BTC volatility: {e}")
    if check_db_connection():
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DATE(closed_at) as pnl_date, SUM((profit_percentage / 100) * CASE WHEN is_real_trade = true AND quantity IS NOT NULL AND entry_price IS NOT NULL 
                    THEN quantity * entry_price ELSE %s END) as daily_pnl
                    FROM signals WHERE status = 'closed' AND closed_at IS NOT NULL
                    GROUP BY pnl_date ORDER BY pnl_date DESC LIMIT 15;
                """, (STATS_TRADE_SIZE_USDT,))
                pnl_data = cur.fetchall()
                daily_pnl_history = [{"date": r['pnl_date'].strftime('%Y-%m-%d'), "pnl": float(r['daily_pnl'])} for r in reversed(pnl_data)]
        except Exception as e: logger.error(f"Error fetching PNL history: {e}")
    portfolio_distribution = {}
    if redis_client:
        try:
            with signal_cache_lock: open_positions = [p for p in open_signals_cache.values() if p.get('is_real_trade')]
            current_prices = redis_client.hgetall(REDIS_PRICES_HASH_NAME)
            for pos in open_positions:
                current_price_str = current_prices.get(pos['symbol'])
                if current_price_str: portfolio_distribution[pos['symbol']] = float(pos['quantity']) * float(current_price_str)
        except Exception as e: logger.error(f"Error fetching portfolio distribution: {e}")
    return jsonify({
        "fear_greed": fear_greed, "btc_volatility": btc_volatility,
        "daily_pnl_history": daily_pnl_history,
        "portfolio_distribution": portfolio_distribution,
        "portfolio_risk": risk_management_system.calculate_portfolio_risk()
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

@app.route('/api/notifications')
def get_notifications():
    with notifications_lock: return jsonify(list(notifications_cache))

@app.route('/api/rejection_logs')
def get_rejection_logs():
    with rejection_logs_lock: return jsonify(list(rejection_logs_cache))

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
    if not redis_client or not enhanced_client: return jsonify({"success": False, "message": "Services not ready"}), 503
    with signal_cache_lock:
        signal_to_close = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)
    if not signal_to_close: return jsonify({"success": False, "message": "Signal not found"}), 404
    try:
        current_price = float(redis_client.hget(REDIS_PRICES_HASH_NAME, signal_to_close['symbol']))
    except (TypeError, ValueError):
        try: current_price = float(enhanced_client.safe_get_symbol_ticker(symbol=signal_to_close['symbol'])['price'])
        except Exception as e: return jsonify({"success": False, "message": f"Could not fetch price: {e}"}), 500
    if close_signal(signal_id, current_price, 'manual'):
        return jsonify({"success": True, "message": "Signal closed."})
    else:
        return jsonify({"success": False, "message": "Failed to close signal."}), 500

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
                signal_id, tp, sl, entry = signal['id'], float(signal['target_price']), float(signal['stop_loss']), float(signal['entry_price'])
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
                            if signal['symbol'] in open_signals_cache: open_signals_cache[signal['symbol']]['current_peak_price'] = new_peak
                        try:
                            if check_db_connection():
                                with conn.cursor() as cur: cur.execute("UPDATE signals SET current_peak_price = %s WHERE id = %s", (new_peak, signal_id)); conn.commit()
                        except Exception as e: logger.error(f"DB error updating peak price for {signal['symbol']}: {e}"); conn.rollback()
                    if (new_peak / entry - 1) * 100 >= TRAILING_ACTIVATION_PROFIT_PERCENT:
                        new_sl = new_peak * (1 - TRAILING_DISTANCE_PERCENT / 100)
                        if new_sl > sl:
                            logger.info(f"📈 [TRAILING SL] {signal['symbol']} SL moved to {new_sl:.4f}")
                            with signal_cache_lock:
                                if signal['symbol'] in open_signals_cache: open_signals_cache[signal['symbol']]['stop_loss'] = new_sl
                            try:
                                if check_db_connection():
                                    with conn.cursor() as cur: cur.execute("UPDATE signals SET stop_loss = %s WHERE id = %s", (new_sl, signal_id)); conn.commit()
                            except Exception as e: logger.error(f"DB error updating trailing SL for {signal['symbol']}: {e}"); conn.rollback()
            time.sleep(2)
        except Exception as e:
            logger.error(f"❌ [Trade Manager] خطأ في حلقة الإدارة: {e}", exc_info=True)
            time.sleep(10)

def main_loop_enhanced():
    logger.info("[Main Loop] انتظار اكتمال التهيئة...")
    time.sleep(15)
    if not validated_symbols_to_scan:
        log_and_notify("critical", "لا توجد عملات صالحة للمسح.", "SYSTEM")
        return
    log_and_notify("info", f"✅ بدء حلقة المسح لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")

    while True:
        try:
            logger.info("🔄 بدء دورة مسح جديدة...")
            determine_market_state_enhanced()
            analyze_market_and_create_dynamic_profile_enhanced()
            
            with dynamic_filter_lock: filter_profile = dynamic_filter_profile_cache
            if not filter_profile:
                logger.warning("🛑 لم يتم تحميل ملف الفلاتر. الانتظار..."); time.sleep(60); continue

            btc_data = get_btc_data_for_bot()
            symbols_to_process = random.sample(validated_symbols_to_scan, len(validated_symbols_to_scan))
            
            for i in range(0, len(symbols_to_process), SYMBOL_PROCESSING_BATCH_SIZE):
                batch = symbols_to_process[i:i + SYMBOL_PROCESSING_BATCH_SIZE]
                total_batches = (len(symbols_to_process) + SYMBOL_PROCESSING_BATCH_SIZE - 1) // SYMBOL_PROCESSING_BATCH_SIZE
                logger.info(f"🔄 Processing batch {i // SYMBOL_PROCESSING_BATCH_SIZE + 1}/{total_batches} ({len(batch)} symbols)...")

                for symbol in batch:
                    # --- MEMORY OPTIMIZATION: Define variables to be cleaned up ---
                    strategy, df_15m, df_4h, df_features = None, None, None, None
                    try:
                        with signal_cache_lock:
                            if symbol in open_signals_cache or len(open_signals_cache) >= MAX_OPEN_TRADES: continue
                        
                        strategy = EnhancedTradingStrategy(symbol)
                        if not all([strategy.ml_model, strategy.scaler, strategy.feature_names]): continue
                        
                        df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        if df_15m is None or df_15m.empty: continue
                        
                        df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        if df_4h is None or df_4h.empty: continue
                        
                        df_features = strategy.get_features(df_15m, df_4h, btc_data)
                        if df_features is None or df_features.empty: continue
                        
                        ml_signal = strategy.generate_buy_signal(df_features)
                        if not ml_signal or ml_signal['confidence'] < BUY_CONFIDENCE_THRESHOLD:
                            if ml_signal: log_rejection(symbol, "ML Model Rejected Signal", {"confidence": ml_signal['confidence']})
                            continue
                        
                        entry_price = float(enhanced_client.safe_get_symbol_ticker(symbol=symbol)['price'])
                        tp_sl_data = calculate_tp_sl(symbol, entry_price, df_15m)
                        if not tp_sl_data: continue
                        
                        if not passes_filters(symbol, df_features.iloc[-1], filter_profile, entry_price, tp_sl_data, df_15m): continue
                        
                        order_book_analysis = analyze_order_book(symbol, entry_price)
                        if not order_book_analysis or not passes_order_book_check(symbol, order_book_analysis, filter_profile): continue
                        
                        new_signal = {'symbol': symbol, 'strategy_name': "Momentum_ML_V9", 'signal_details': {'ML_Confidence': f"{ml_signal['confidence']:.2%}", 'Filter_Profile': f"{filter_profile['name']}", 'Bid_Ask_Ratio': order_book_analysis.get('bid_ask_ratio', 0), **tp_sl_data}, 'entry_price': entry_price, **tp_sl_data}
                        
                        with trading_status_lock: is_enabled = is_trading_enabled
                        if is_enabled:
                            quantity = calculate_position_size(symbol, entry_price, new_signal['stop_loss'])
                            if quantity and quantity > 0:
                                quantity = risk_management_system.adjust_position_size_based_on_risk(symbol, quantity)
                                order_result = place_order(symbol, Client.SIDE_BUY, quantity)
                                if order_result:
                                    new_signal.update({'is_real_trade': True, 'quantity': float(quantity), 'order_id': order_result['orderId']})
                                else: continue
                            else: continue
                        
                        saved_signal = insert_signal_into_db(new_signal)
                        if saved_signal:
                            with signal_cache_lock: open_signals_cache[saved_signal['symbol']] = saved_signal
                            log_and_notify('info', f"SIGNAL: New buy signal for {symbol} at {entry_price}", "NEW_SIGNAL")
                    except Exception as e:
                        logger.error(f"❌ [Processing Error] للعملة {symbol}: {e}", exc_info=True)
                    finally:
                        # --- MEMORY OPTIMIZATION: Aggressive cleanup after each symbol ---
                        del strategy, df_15m, df_4h, df_features
                        gc.collect()

            logger.info("✅ [End of Cycle] انتهت دورة المسح الكاملة. الانتظار 60 ثانية...")
            time.sleep(60)
            
        except (KeyboardInterrupt, SystemExit):
            log_and_notify("info", "إيقاف البوت.", "SYSTEM"); break
        except Exception as main_err:
            log_and_notify("error", f"خطأ حرج في الحلقة الرئيسية: {main_err}", "SYSTEM"); time.sleep(120)

def price_update_loop():
    if not redis_client or not enhanced_client: return
    all_symbols = []
    while True:
        try:
            if not all_symbols: # Fetch all symbols once
                all_symbols = [s['symbol'] for s in enhanced_client.safe_get_exchange_info()['symbols'] if s['quoteAsset'] == 'USDT']
            
            if all_symbols:
                tickers = enhanced_client.client.get_symbol_ticker()
                prices_to_set = {t['symbol']: t['price'] for t in tickers if t['symbol'] in all_symbols}
                if prices_to_set: redis_client.hset(REDIS_PRICES_HASH_NAME, mapping=prices_to_set)
            time.sleep(2) # Update prices every 2 seconds
        except Exception as e: 
            logger.error(f"Error in price update loop: {e}"); time.sleep(10)

def initialize_bot_services():
    global client, enhanced_client, validated_symbols_to_scan
    logger.info("🤖 [Bot Services] بدء التهيئة...")
    try:
        enhanced_client = EnhancedClient(API_KEY, API_SECRET)
        client = enhanced_client.client
        init_db()
        init_redis()
        load_open_signals_to_cache()
        load_notifications_to_cache()
        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan: 
            log_and_notify("critical", "لا توجد عملات صالحة للمسح. سيتم إيقاف البوت.", "SYSTEM")
            return
        
        Thread(target=main_loop_enhanced, daemon=True).start()
        Thread(target=price_update_loop, daemon=True).start()
        Thread(target=trade_management_loop, daemon=True).start()
        
        logger.info("✅ [Bot Services] تم بدء جميع الخدمات الخلفية بنجاح.")
        send_telegram_message("✅ *البوت قيد التشغيل الآن*")
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حرج أثناء التهيئة: {e}", "SYSTEM"); exit(1)

# ---------------------- نقطة الانطلاق ----------------------
if __name__ == "__main__":
    logger.info("🚀 إطلاق بوت التداول ولوحة التحكم (V9 - Final with Telegram) 🚀")
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
