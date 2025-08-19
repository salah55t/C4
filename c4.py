# ملف c4.py - نسخة V12.3.1 (إصلاح خطأ المسافات البادئة)
# --- التغييرات الرئيسية (V12.3.1):
# 1. [إصلاح حاسم] تم إصلاح خطأ المسافات البادئة في الدالة `update_market_state`.
# 2. [تحسين] تم التأكد من صحة المسافات البادئة لجميع الدوال المضافة حديثاً.
# 3. [ميزة جديدة] إضافة استراتيجية دخول جديدة: Volume Profile.
# 4. [تحسين] تحسين دالة وقف الخسارة المتحرك (Trailing Stop Loss) بناءً على ATR ونسبة الربح.
# 5. [تحسين] تحسين دالة أخذ الربح الجزئي (Partial Take Profit) مع الأخذ في الاعتبار قوة الاتجاه.
# 6. [ميزة جديدة] إضافة استراتيجية خروج جديدة بناءً على تغير هيكل السوق (Market Structure Change).
# 7. [ميزة جديدة] إضافة استراتيجية خروج جديدة بناءً على إشارات الانعكاس (Reversal Signal).
# 8. [تعريب] تعريب تقييم حالة السوق.

import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import redis
import gc
from decimal import Decimal
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timezone
from decouple import config
from typing import List, Dict, Optional, Any
from collections import deque
import warnings

# --- إعدادات التجاهل واللوجر ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v12_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV12.3.1')

# --- المشفر المخصص لأنواع بيانات NumPy ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, Decimal): return float(obj)
        if isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat()
        return super(NpEncoder, self).default(obj)

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
paper_trading_mode: bool = True

# --- المتغيرات القابلة للتعديل ---
RISK_PER_TRADE_PERCENT: float = 0.85
risk_per_trade_lock = Lock()
BUY_CONFIDENCE_THRESHOLD = 0.53
buy_confidence_lock = Lock()
MAX_OPEN_TRADES: int = 3
MIN_PROFIT_PERCENT: float = 0.8
PAPER_TRADE_SIZE_USDT: float = 10.0

# --- إعدادات إدارة الصفقات المتقدمة ---
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_STOP_TRIGGER_PERCENT: float = 0.4
TRAILING_STOP_DISTANCE_PERCENT: float = 0.5
USE_PARTIAL_TAKE_PROFIT: bool = True
PARTIAL_TP_RSI_THRESHOLD: float = 60
USE_VOLUME_PROFILE_STRATEGY: bool = True
volume_profile_strategy_lock = Lock()

# --- مفاتيح تفعيل الاستراتيجيات ---
USE_BB_STOCH_STRATEGY: bool = True
bb_stoch_strategy_lock = Lock()
USE_MACD_EMA_STRATEGY: bool = True
macd_ema_strategy_lock = Lock()
USE_EMA_RSI_STRATEGY: bool = True
ema_rsi_strategy_lock = Lock()
USE_PULLBACK_STRATEGY: bool = True
pullback_strategy_lock = Lock()

# --- إعدادات عامة ---
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '1h'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 15
BTC_SYMBOL: str = 'BTCUSDT'
SYMBOL_PROCESSING_BATCH_SIZE: int = 5
ATR_TS_MULTIPLIER: float = 2.2
TRADING_FEE_PERCENT: float = 0.1
API_REQUEST_DELAY: float = 0.5
API_RETRY_COUNT: int = 3
API_RETRY_DELAY: float = 5.0

# --- إعدادات المؤشرات الفنية ---
EMA_FAST_PERIOD: int = 12
EMA_SLOW_PERIOD: int = 26
ADX_PERIOD: int = 10
RSI_PERIOD: int = 10
ATR_PERIOD: int = 10
MOMENTUM_PERIOD: int = 5

# --- متغيرات الحالة والكاش ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
ws_manager: Optional[ThreadedWebsocketManager] = None
live_prices: Dict[str, float] = {}
live_prices_lock = Lock()
exchange_info_map: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
notifications_cache = deque(maxlen=20)
notifications_lock = Lock()
rejection_logs_cache = deque(maxlen=30)
rejection_logs_lock = Lock()
current_market_state: Dict[str, Any] = {"overall_regime": "INITIALIZING", "trend_details_by_tf": {}, "last_updated": "N/A"}
market_state_lock = Lock()

# --- قاموس أسباب الرفض باللغة العربية ---
REJECTION_REASONS_AR = {
    "Market Volatility Filter Failed": "فلتر تقلب السوق رفض الدخول",
    "Trend Strength Filter Failed": "فلتر قوة الاتجاه رفض الدخول",
    "HTF Trend Confirmation Failed": "فشل تأكيد الترند على الفريم الأعلى",
    "Bullish Reversal Candle Pattern Failed": "لم يظهر نمط شمعة انعكاسية صاعدة",
    "Insufficient Historical Data": "بيانات تاريخية غير كافية للفحص",
    "Volume Profile Strategy Failed": "استراتيجية ملف الحجم رفضت الدخول"
}

# --- إعداد تطبيق Flask ---
app = Flask(__name__)
CORS(app)

# --- دوال تهيئة الخدمات ---
def init_db(retries: int = 5, delay: int = 5) -> None:
    """تهيئة الاتصال بقاعدة البيانات PostgreSQL وإنشاء الجداول اللازمة."""
    global conn
    logger.info("[DB] Initializing database connection...")
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
                        is_real_trade BOOLEAN DEFAULT FALSE, quantity DOUBLE PRECISION, closing_reason TEXT,
                        target_price_2 DOUBLE PRECISION, initial_quantity DOUBLE PRECISION
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL, message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE
                    );
                """)
            conn.commit()
            logger.info("✅ [DB] Database connection and schema updated successfully.")
            return
        except Exception as e:
            logger.error(f"❌ [DB] Error during initialization (Attempt {attempt + 1}/{retries}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: logger.critical("❌ [DB] Failed to connect to the database. Exiting.")

def check_db_connection() -> bool:
    """يتحقق من حالة الاتصال بقاعدة البيانات ويعيد الاتصال إذا لزم الأمر."""
    global conn
    if conn is None or conn.closed != 0:
        logger.warning("[DB] Connection is closed. Attempting to reconnect...")
        init_db()
    try:
        if conn and conn.closed == 0:
            with conn.cursor() as cur: cur.execute("SELECT 1;")
            return True
        return False
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [DB] Connection lost: {e}. Reconnecting...")
        init_db()
        return conn is not None and conn.closed == 0

def init_redis() -> None:
    """تهيئة الاتصال بذاكرة التخزين المؤقت Redis."""
    global redis_client
    logger.info("[Redis] Initializing connection...")
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] Connected successfully.")
    except redis.exceptions.ConnectionError as e:
        logger.warning(f"⚠️ [Redis] Connection failed: {e}. The bot will run without Redis.")
        redis_client = None

# --- دوال المساعدة والإشعارات ---
def log_and_notify(level: str, message: str, notification_type: str):
    """يسجل رسالة في السجل ويحفظها كإشعار في قاعدة البيانات."""
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error}
    log_methods.get(level.lower(), logger.info)(message)
    if not check_db_connection() or not conn:
        logger.error(f"[DB] Could not save notification due to DB connection issue: {message}")
        return
    try:
        new_notification = {"timestamp": datetime.now(timezone.utc).isoformat(), "type": notification_type, "message": message}
        with notifications_lock: notifications_cache.appendleft(new_notification)
        with conn.cursor() as cur: cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [DB] Failed to save notification: {e}")
        if conn: conn.rollback()

def log_rejection(symbol: str, reason_key: str, details: Optional[Dict] = None):
    """يسجل سبب رفض إشارة دخول في الذاكرة المؤقتة."""
    reason_ar = REJECTION_REASONS_AR.get(reason_key, reason_key)
    with rejection_logs_lock:
        rejection_logs_cache.appendleft({
            "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
            "reason": reason_ar, "details": json.loads(json.dumps(details or {}, cls=NpEncoder))
        })

def send_telegram_message(message: str):
    """يرسل رسالة إلى تيليجرام."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [Telegram] Failed to send message: {e}")

# --- WebSocket Handler ---
def handle_socket_message(msg):
    """يعالج الرسائل الواردة من WebSocket لتحديث الأسعار الحية."""
    global live_prices
    if msg and 'e' in msg and msg['e'] == 'error':
        logger.error(f"❌ [WebSocket] Error: {msg['m']}")
        return
    if isinstance(msg, list):
        with live_prices_lock:
            for ticker in msg:
                if 's' in ticker and 'c' in ticker:
                    live_prices[ticker['s']] = float(ticker['c'])

def start_websocket():
    """يبدأ مدير WebSocket للاستماع إلى تحديثات الأسعار."""
    global ws_manager
    logger.info("🚀 [WebSocket] Starting WebSocket manager...")
    ws_manager = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    ws_manager.start()
    ws_manager.start_ticker_socket(callback=handle_socket_message)
    logger.info("✅ [WebSocket] Successfully subscribed to ticker stream (!ticker@arr).")

# --- دوال جلب البيانات وحساب المؤشرات ---
def get_exchange_info_map() -> None:
    """يجلب معلومات البورصة وينشئ خريطة للرموز."""
    global exchange_info_map
    if not client: return
    try:
        logger.info("[API] Fetching exchange info...")
        info = client.get_exchange_info()
        exchange_info_map = {s['symbol']: s for s in info['symbols']}
        logger.info(f"[API] Exchange info map created with {len(exchange_info_map)} symbols.")
    except BinanceAPIException as e:
        logger.error(f"❌ [API] Binance error fetching exchange info: {e}")
    except Exception as e:
        logger.error(f"❌ [API] Generic error fetching exchange info: {e}")

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """يقرأ قائمة الرموز ويتحقق من صحتها مقابل البورصة."""
    if not client: return []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        if not os.path.exists(file_path):
            logger.critical(f"❌ Symbol list file '{filename}' not found!"); return []
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        if not exchange_info_map: get_exchange_info_map()
        active = {s for s, info in exchange_info_map.items() if info.get('quoteAsset') == 'USDT' and info.get('status') == 'TRADING'}
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ Found {len(validated)} valid symbols for trading.")
        return validated
    except Exception as e:
        logger.error(f"❌ [Symbols] Error validating symbols: {e}"); return []

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """يجلب البيانات التاريخية للشموع من Binance."""
    if not client: return None
    time.sleep(API_REQUEST_DELAY)
    try:
        lookback_str = f"{days} day ago UTC"
        klines = client.get_historical_klines(symbol, interval, lookback_str)
        if not klines: return None
        processed_klines = [kline[:6] for kline in klines]
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(processed_klines, columns=cols)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna().astype(float)
    except BinanceAPIException as e:
        logger.error(f"❌ [API] Binance error fetching data for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ [Data] Generic error fetching data for {symbol}: {e}"); return None

def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """تحسب جميع المؤشرات الفنية الأساسية (EMA, ATR, ADX, RSI, MACD)."""
    df_calc = df.copy()
    df_calc['ema_9'] = df_calc['close'].ewm(span=9, adjust=False).mean()
    df_calc['ema_12'] = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    df_calc['ema_26'] = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
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
    rsi_val = df_calc['rsi']
    stoch_rsi = (rsi_val - rsi_val.rolling(14).min()) / (rsi_val.rolling(14).max() - rsi_val.rolling(14).min()).replace(0, 1e-9)
    df_calc['stoch_rsi'] = stoch_rsi.rolling(3).mean() * 100
    df_calc['macd'] = df_calc['ema_12'] - df_calc['ema_26']
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()
    df_calc['macd_hist'] = df_calc['macd'] - df_calc['macd_signal']
    df_calc['lower_band'] = df_calc['close'].rolling(20).mean() - (df_calc['close'].rolling(20).std() * 2)
    df_calc['upper_band'] = df_calc['close'].rolling(20).mean() + (df_calc['close'].rolling(20).std() * 2)
    return df_calc.dropna().astype(float)

def calculate_market_trend(df: pd.DataFrame) -> str:
    """تحسب اتجاه السوق بناءً على EMA و ADX."""
    last = df.iloc[-1]
    last_but_one = df.iloc[-2]
    is_up_trend = (last['ema_9'] > last['ema_26'] and
                   last_but_one['ema_9'] > last_but_one['ema_26'] and
                   last['adx'] > 20)
    is_down_trend = (last['ema_9'] < last['ema_26'] and
                     last_but_one['ema_9'] < last_but_one['ema_26'] and
                     last['adx'] > 20)
    if is_up_trend: return 'اتجاه صاعد'
    if is_down_trend: return 'اتجاه هابط'
    return 'سوق عرضي'

def is_bullish_reversal_pattern(df: pd.DataFrame) -> bool:
    """تتحقق من وجود نمط شمعة انعكاسية صاعدة (مثال: مطرقة، ابتلاع صاعد)."""
    if len(df) < 2: return False
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # نمط المطرقة (Hammer)
    hammer = (last['close'] > last['open'] and  # شمعة صاعدة
              last['low'] < last['open'] * 0.99 and  # ظل سفلي طويل
              (last['open'] - last['low']) > 2 * (last['close'] - last['open']))
    
    # نمط الابتلاع الصاعد (Bullish Engulfing)
    engulfing = (prev['close'] < prev['open'] and  # شمعة سابقة هابطة
                 last['close'] > last['open'] and  # شمعة حالية صاعدة
                 last['close'] > prev['open'] and  # الشمعة الحالية تغطي السابقة
                 last['open'] < prev['close'])
    
    return hammer or engulfing

def is_bearish_reversal_pattern(df: pd.DataFrame) -> bool:
    """تتحقق من وجود نمط شمعة انعكاسية هابطة (مثال: شهاب، ابتلاع هابط)."""
    if len(df) < 2: return False
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # نمط الشهاب (Shooting Star)
    shooting_star = (last['close'] < last['open'] and  # شمعة هابطة
                     last['high'] > last['close'] * 1.01 and # ظل علوي طويل
                     (last['high'] - last['close']) > 2 * (last['open'] - last['close']))
    
    # نمط الابتلاع الهابط (Bearish Engulfing)
    engulfing = (prev['close'] > prev['open'] and  # شمعة سابقة صاعدة
                 last['close'] < last['open'] and  # شمعة حالية هابطة
                 last['close'] < prev['open'] and  # الشمعة الحالية تغطي السابقة
                 last['open'] > prev['close'])
                 
    return shooting_star or engulfing

# --- استراتيجيات الدخول (جديد) ---
def calculate_volume_profile(df: pd.DataFrame):
    """
    يحسب ملف الحجم (Volume Profile) ونقطة التحكم (POC) ومنطقة القيمة (VA).
    هذه نسخة مبسطة لغرض العرض.
    """
    price_bins = pd.cut(df['close'], bins=20, labels=False)
    volume_by_bin = df.groupby(price_bins)['volume'].sum()
    
    # نقطة التحكم (Point of Control - POC): أعلى حجم
    poc_bin = volume_by_bin.idxmax()
    poc = df['close'].iloc[price_bins[price_bins == poc_bin].index].mean()
    
    # منطقة القيمة (Value Area - VA): 70% من الحجم
    total_volume = volume_by_bin.sum()
    sorted_bins = volume_by_bin.sort_values(ascending=False)
    
    cumulative_volume = 0
    value_area_bins = []
    
    for bin_label, volume in sorted_bins.items():
        cumulative_volume += volume
        value_area_bins.append(bin_label)
        if cumulative_volume >= total_volume * 0.7:
            break
            
    value_area_prices = df['close'].iloc[price_bins[price_bins.isin(value_area_bins)].index]
    value_area_high = value_area_prices.max()
    value_area_low = value_area_prices.min()
    
    return poc, value_area_high, value_area_low

def check_volume_profile_strategy(df: pd.DataFrame) -> bool:
    """
    استراتيجية ملف الحجم: تتحقق من أن السعر فوق منطقة القيمة، مع حجم تداول وقوة اتجاه عالية.
    """
    last = df.iloc[-1]
    
    # حساب نقطة التحكم (Point of Control) وملف الحجم
    poc, value_area_high, value_area_low = calculate_volume_profile(df)
    
    # تحقق من أن السعر فوق منطقة القيمة
    above_value_area = last['close'] > value_area_high
    
    # تحقق من حجم التداول العالي
    high_volume = last['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.5
    
    # تحقق من إغلاق الشمعة فوق قمة الشمعة السابقة
    price_action = last['close'] > df['high'].iloc[-2]
    
    # تحقق من قوة الاتجاه
    trend_strength = last['adx'] > 18
    
    return above_value_area and high_volume and price_action and trend_strength
    
# --- استراتيجيات الخروج المتقدمة (جديد) ---
def check_market_structure_change(df: pd.DataFrame, signal: Dict) -> bool:
    """
    استراتيجية خروج: تتحقق من تغير في هيكل السوق مثل كسر قاع مهم أو ظهور نمط انعكاسي قوي.
    """
    last = df.iloc[-1]
    
    # تحقق من كسر هيكل السوق (تشكيل قمة أدنى أو قاع أدنى)
    market_structure_broken = False
    
    # حساب القمم والقيعان المحلية
    highs = df['high'].rolling(5).max()
    lows = df['low'].rolling(5).min()
    
    # تحقق من كسر قاع محلي مهم
    if last['close'] < lows.iloc[-2] * 0.995:  # كسر بنسبة 0.5%
        market_structure_broken = True
    
    # تحقق من ظهور نمط انعكاسي قوي
    if is_bearish_reversal_pattern(df):
        market_structure_broken = True
    
    # تحقق من تغير قوة الاتجاه
    if last['adx'] < 15:  # اتجاه ضعيف جداً
        market_structure_broken = True
    
    return market_structure_broken

def check_reversal_signal_exit(df: pd.DataFrame, signal: Dict) -> bool:
    """
    استراتيجية خروج: تتحقق من ظهور إشارات عكسية متعددة مثل تقاطع MACD أو ذروة شراء في RSI.
    """
    last = df.iloc[-1]
    
    # تحقق من ظهور إشارات عكسية قوية
    
    # 1. تقاطع MACD عكسي قوي
    macd_bearish_cross = (df['macd'].iloc[-2] > df['macd_signal'].iloc[-2] and
                          last['macd'] < last['macd_signal'] and
                          last['macd'] > 0)  # فوق خط الصفر
    
    # 2. RSI في منطقة ذروة الشراء
    rsi_overbought = last['rsi'] > 70
    
    # 3. نمط شمعة انعكاسية هابطة
    bearish_candle = is_bearish_reversal_pattern(df)
    
    # 4. حجم تداول عالي مع حركة سعرية عكسية
    high_volume_reversal = (last['volume'] > df['volume'].rolling(10).mean().iloc[-1] * 1.5 and
                           last['close'] < last['open'] and
                           (last['close'] - last['low']) > (last['high'] - last['close']) * 2)
    
    # إذا ظهرت على الأقل إشارتان من الإشارات المذكورة، قم بالخروج
    reversal_signals = [macd_bearish_cross, rsi_overbought, bearish_candle, high_volume_reversal]
    
    return sum(reversal_signals) >= 2

# --- دوال إدارة الصفقات (تحسين) ---
def calculate_trailing_stop_loss(df: pd.DataFrame, entry_price: float, initial_stop: float, highest_price: float) -> float:
    """
    يحسب وقف الخسارة المتحرك بناءً على ATR ونسبة الربح المحققة.
    """
    if not USE_TRAILING_STOP_LOSS:
        return initial_stop
        
    last = df.iloc[-1]
    atr = last['atr']
    
    # حساب نسبة الربح الحالية
    profit_percent = ((highest_price - entry_price) / entry_price) * 100
    
    # إذا وصلنا إلى نسبة ربح معينة، قم بتفعيل وقف الخسارة المتحرك
    if profit_percent >= TRAILING_STOP_TRIGGER_PERCENT:
        # حساب المسافة المتحركة بناءً على ATR
        trailing_distance = atr * TRAILING_STOP_DISTANCE_PERCENT
        
        # وقف الخسارة المتحرك هو أعلى سعر - المسافة المتحركة
        trailing_stop = highest_price - trailing_distance
        
        # تأكد من أن وقف الخسارة المتحرك ليس أقل من وقف الخسارة الأولي
        return max(trailing_stop, initial_stop)
    
    return initial_stop

def check_partial_take_profit(df: pd.DataFrame, signal: Dict, current_price: float) -> Dict:
    """
    يتحقق من شروط أخذ الربح الجزئي، مع الأخذ في الاعتبار قوة الاتجاه عبر RSI.
    """
    if not USE_PARTIAL_TAKE_PROFIT:
        return {"action": "hold"}
        
    last = df.iloc[-1]
    entry_price = signal['entry_price']
    target_price_1 = signal['target_price']
    target_price_2 = signal['target_price_2']
    initial_quantity = signal['initial_quantity']
    current_quantity = signal['quantity']
    
    # إذا وصل السعر إلى الهدف الأول
    if current_price >= target_price_1 and current_quantity == initial_quantity:
        # تحقق من قوة الاتجاه باستخدام RSI
        if last['rsi'] >= PARTIAL_TP_RSI_THRESHOLD:
            # الاتجاه لا يزال قوياً، نأخذ نصف الربح ونستمر للهدف الثاني
            new_quantity = initial_quantity * 0.5
            
            # تحديث الإشارة في الذاكرة المؤقتة
            with signal_cache_lock:
                open_signals_cache[signal['symbol']]['quantity'] = new_quantity
            
            # إرسال إشعار
            message = (f"📊 *أخذ ربح جزئي*\\n"
                       f"💱 *العملة:* `{signal['symbol']}`\\n"
                       f"🎯 *الهدف 1 مُحقق:* `{target_price_1:.4f}`\\n"
                       f"📈 *الهدف 2:* `{target_price_2:.4f}`\\n"
                       f"📉 *الكمية المتبقية:* `{new_quantity:.4f}`")
            send_telegram_message(message)
            log_and_notify("info", f"تم أخذ ربح جزئي لـ {signal['symbol']} والاستمرار للهدف الثاني", "PARTIAL_TP")
            
            return {"action": "partial_tp", "new_quantity": new_quantity}
        else:
            # الاتجاه ضعيف، نخرج من الصفقة بالكامل
            return {"action": "close_all", "reason": "weak_trend"}
    
    return {"action": "hold"}
    
def check_for_close_conditions(df: pd.DataFrame, signal: Dict, current_price: float) -> bool:
    """
    يتحقق من جميع شروط الخروج الجديدة.
    """
    # تحقق من تغير هيكل السوق
    if check_market_structure_change(df, signal):
        logger.warning(f"❌ [Close Signal] إشارة خروج لـ {signal['symbol']} بسبب تغير هيكل السوق.")
        return True
    
    # تحقق من إشارة انعكاس قوية
    if check_reversal_signal_exit(df, signal):
        logger.warning(f"❌ [Close Signal] إشارة خروج لـ {signal['symbol']} بسبب ظهور إشارة عكسية.")
        return True

    # شروط الخروج الأصلية
    stop_loss_hit = current_price <= signal['stop_loss']
    take_profit_hit = current_price >= signal['target_price']
    take_profit_2_hit = signal['target_price_2'] and current_price >= signal['target_price_2']
    
    if stop_loss_hit:
        logger.warning(f"❌ [Close Signal] إشارة خروج لـ {signal['symbol']} بسبب ضرب وقف الخسارة.")
        return True
    
    if take_profit_hit:
        if take_profit_2_hit:
            logger.info(f"✅ [Close Signal] إشارة خروج لـ {signal['symbol']} بسبب تحقيق الهدف الثاني.")
            return True
        else:
            partial_tp_result = check_partial_take_profit(df, signal, current_price)
            if partial_tp_result['action'] == 'close_all':
                logger.info(f"✅ [Close Signal] إشارة خروج لـ {signal['symbol']} بسبب ضعف الترند عند الهدف الأول.")
                return True
    
    return False

# --- دوال التعامل مع قاعدة البيانات (تحديثات طفيفة) ---
def create_signal_db(signal_data: Dict) -> bool:
    """ينشئ صفقة جديدة في قاعدة البيانات."""
    if not check_db_connection(): return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, is_real_trade, quantity, target_price_2, initial_quantity)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
            """, (signal_data['symbol'], signal_data['entry_price'], signal_data['target_price'],
                  signal_data['stop_loss'], signal_data['strategy'], json.dumps(signal_data['details'], cls=NpEncoder),
                  not paper_trading_mode, signal_data['quantity'], signal_data['target_price_2'], signal_data['initial_quantity']))
            signal_id = cur.fetchone()['id']
            conn.commit()
            log_and_notify("info", f"✅ [DB] تم حفظ إشارة جديدة ID: {signal_id} لـ {signal_data['symbol']}", "DB_ACTION")
            return True
    except Exception as e:
        logger.error(f"❌ [DB] فشل حفظ الإشارة: {e}")
        if conn: conn.rollback()
        return False

def get_open_signals() -> List[Dict]:
    """يجلب جميع الإشارات المفتوحة من قاعدة البيانات."""
    if not check_db_connection(): return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status = 'open';")
            signals = cur.fetchall()
        return signals
    except Exception as e:
        logger.error(f"❌ [DB] فشل جلب الإشارات المفتوحة: {e}"); return []

def update_signal_db(signal_id: int, updates: Dict) -> bool:
    """يحدث صفقة موجودة في قاعدة البيانات."""
    if not check_db_connection(): return False
    try:
        with conn.cursor() as cur:
            sql_query = sql.SQL("UPDATE signals SET {} WHERE id = %s").format(
                sql.SQL(", ").join([
                    sql.Identifier(key) for key in updates.keys()
                ])
            )
            cur.execute(sql_query, list(updates.values()) + [signal_id])
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"❌ [DB] فشل تحديث الإشارة ID {signal_id}: {e}")
        if conn: conn.rollback()
        return False

def close_signal_db(signal: Dict, closing_price: float, closing_reason: str):
    """يغلق صفقة في قاعدة البيانات ويحسب الربح."""
    if not check_db_connection(): return
    try:
        entry_price = signal['entry_price']
        profit_percentage = ((closing_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        profit_percentage_after_fees = profit_percentage - (TRADING_FEE_PERCENT * 2)
        
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE signals SET status = 'closed', closing_price = %s, closed_at = NOW(),
                profit_percentage = %s, closing_reason = %s WHERE id = %s;
            """, (closing_price, profit_percentage_after_fees, closing_reason, signal['id']))
            conn.commit()
            log_and_notify("info", f"✅ [DB] تم إغلاق الإشارة ID: {signal['id']} لـ {signal['symbol']} بنجاح.", "DB_ACTION")
    except Exception as e:
        logger.error(f"❌ [DB] فشل إغلاق الإشارة ID {signal['id']}: {e}")
        if conn: conn.rollback()

def load_open_signals_to_cache() -> None:
    """يحمل الإشارات المفتوحة من قاعدة البيانات إلى الذاكرة المؤقتة عند بدء التشغيل."""
    global open_signals_cache
    if not check_db_connection(): return
    try:
        signals = get_open_signals()
        with signal_cache_lock:
            open_signals_cache = {s['symbol']: s for s in signals}
        logger.info(f"✅ [Cache] تم تحميل {len(signals)} إشارة مفتوحة إلى الذاكرة المؤقتة.")
    except Exception as e:
        logger.error(f"❌ [Cache] فشل تحميل الإشارات المفتوحة: {e}")

def load_notifications_to_cache() -> None:
    """يحمل الإشعارات الأخيرة من قاعدة البيانات إلى الذاكرة المؤقتة عند بدء التشغيل."""
    if not check_db_connection(): return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 20;")
            notifs = cur.fetchall()
        with notifications_lock:
            notifications_cache.extendleft(notifs)
        logger.info(f"✅ [Cache] تم تحميل {len(notifs)} إشعارًا إلى الذاكرة المؤقتة.")
    except Exception as e:
        logger.error(f"❌ [Cache] فشل تحميل الإشعارات: {e}")

# --- دوال إعدادات Redis (جديد) ---
def save_settings_to_redis() -> None:
    """يحفظ إعدادات البوت القابلة للتغيير إلى Redis."""
    if not redis_client: return
    try:
        settings = {
            'is_trading_enabled': is_trading_enabled,
            'paper_trading_mode': paper_trading_mode,
            'RISK_PER_TRADE_PERCENT': RISK_PER_TRADE_PERCENT,
            'BUY_CONFIDENCE_THRESHOLD': BUY_CONFIDENCE_THRESHOLD,
            'USE_TRAILING_STOP_LOSS': USE_TRAILING_STOP_LOSS,
            'USE_PARTIAL_TAKE_PROFIT': USE_PARTIAL_TAKE_PROFIT,
            'USE_VOLUME_PROFILE_STRATEGY': USE_VOLUME_PROFILE_STRATEGY,
            'USE_BB_STOCH_STRATEGY': USE_BB_STOCH_STRATEGY,
            'USE_MACD_EMA_STRATEGY': USE_MACD_EMA_STRATEGY,
            'USE_EMA_RSI_STRATEGY': USE_EMA_RSI_STRATEGY,
            'USE_PULLBACK_STRATEGY': USE_PULLBACK_STRATEGY,
        }
        redis_client.set('bot_settings', json.dumps(settings))
        logger.info("✅ [Redis] تم حفظ الإعدادات بنجاح.")
    except Exception as e:
        logger.error(f"❌ [Redis] فشل حفظ الإعدادات: {e}")

def load_settings_from_redis() -> None:
    """يحمل إعدادات البوت من Redis عند بدء التشغيل."""
    global is_trading_enabled, paper_trading_mode, RISK_PER_TRADE_PERCENT, BUY_CONFIDENCE_THRESHOLD
    global USE_TRAILING_STOP_LOSS, USE_PARTIAL_TAKE_PROFIT, USE_VOLUME_PROFILE_STRATEGY
    global USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY
    if not redis_client: return
    try:
        settings_json = redis_client.get('bot_settings')
        if settings_json:
            settings = json.loads(settings_json)
            is_trading_enabled = settings.get('is_trading_enabled', is_trading_enabled)
            paper_trading_mode = settings.get('paper_trading_mode', paper_trading_mode)
            RISK_PER_TRADE_PERCENT = settings.get('RISK_PER_TRADE_PERCENT', RISK_PER_TRADE_PERCENT)
            BUY_CONFIDENCE_THRESHOLD = settings.get('BUY_CONFIDENCE_THRESHOLD', BUY_CONFIDENCE_THRESHOLD)
            USE_TRAILING_STOP_LOSS = settings.get('USE_TRAILING_STOP_LOSS', USE_TRAILING_STOP_LOSS)
            USE_PARTIAL_TAKE_PROFIT = settings.get('USE_PARTIAL_TAKE_PROFIT', USE_PARTIAL_TAKE_PROFIT)
            USE_VOLUME_PROFILE_STRATEGY = settings.get('USE_VOLUME_PROFILE_STRATEGY', USE_VOLUME_PROFILE_STRATEGY)
            USE_BB_STOCH_STRATEGY = settings.get('USE_BB_STOCH_STRATEGY', USE_BB_STOCH_STRATEGY)
            USE_MACD_EMA_STRATEGY = settings.get('USE_MACD_EMA_STRATEGY', USE_MACD_EMA_STRATEGY)
            USE_EMA_RSI_STRATEGY = settings.get('USE_EMA_RSI_STRATEGY', USE_EMA_RSI_STRATEGY)
            USE_PULLBACK_STRATEGY = settings.get('USE_PULLBACK_STRATEGY', USE_PULLBACK_STRATEGY)
            logger.info("✅ [Redis] تم تحميل الإعدادات بنجاح.")
    except Exception as e:
        logger.error(f"❌ [Redis] فشل تحميل الإعدادات: {e}")

# --- دوال الواجهة (API) ---
@app.route('/status', methods=['GET'])
def get_status():
    """نقطة نهاية للحصول على حالة البوت الحالية."""
    with trading_status_lock:
        is_enabled = is_trading_enabled
    return jsonify({
        'status': 'Running',
        'trading_enabled': is_enabled,
        'paper_trading_mode': paper_trading_mode,
        'open_trades_count': len(open_signals_cache),
        'max_open_trades': MAX_OPEN_TRADES,
        'risk_per_trade_percent': RISK_PER_TRADE_PERCENT,
        'buy_confidence_threshold': BUY_CONFIDENCE_THRESHOLD,
        'market_state': current_market_state,
        'strategies_enabled': {
            'bb_stoch': USE_BB_STOCH_STRATEGY,
            'macd_ema': USE_MACD_EMA_STRATEGY,
            'ema_rsi': USE_EMA_RSI_STRATEGY,
            'pullback': USE_PULLBACK_STRATEGY,
            'volume_profile': USE_VOLUME_PROFILE_STRATEGY
        },
        'advanced_features_enabled': {
            'trailing_stop_loss': USE_TRAILING_STOP_LOSS,
            'partial_take_profit': USE_PARTIAL_TAKE_PROFIT
        }
    })

@app.route('/open_trades', methods=['GET'])
def get_open_trades():
    """نقطة نهاية للحصول على الصفقات المفتوحة."""
    with signal_cache_lock:
        trades = list(open_signals_cache.values())
    return jsonify({'open_trades': trades})

@app.route('/rejection_logs', methods=['GET'])
def get_rejection_logs():
    """نقطة نهاية للحصول على سجلات الرفض."""
    with rejection_logs_lock:
        logs = list(rejection_logs_cache)
    return jsonify({'rejection_logs': logs})

@app.route('/notifications', methods=['GET'])
def get_notifications():
    """نقطة نهاية للحصول على الإشعارات."""
    with notifications_lock:
        notifs = list(notifications_cache)
    return jsonify({'notifications': notifs})

@app.route('/settings', methods=['POST'])
def update_settings():
    """نقطة نهاية لتحديث إعدادات البوت."""
    global is_trading_enabled, paper_trading_mode, RISK_PER_TRADE_PERCENT, BUY_CONFIDENCE_THRESHOLD
    global USE_TRAILING_STOP_LOSS, USE_PARTIAL_TAKE_PROFIT, USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY, USE_VOLUME_PROFILE_STRATEGY

    try:
        data = request.json
        with trading_status_lock:
            is_trading_enabled = data.get('trading_enabled', is_trading_enabled)
        paper_trading_mode = data.get('paper_trading_mode', paper_trading_mode)
        RISK_PER_TRADE_PERCENT = data.get('risk_per_trade_percent', RISK_PER_TRADE_PERCENT)
        BUY_CONFIDENCE_THRESHOLD = data.get('buy_confidence_threshold', BUY_CONFIDENCE_THRESHOLD)
        
        USE_TRAILING_STOP_LOSS = data.get('use_trailing_stop_loss', USE_TRAILING_STOP_LOSS)
        USE_PARTIAL_TAKE_PROFIT = data.get('use_partial_take_profit', USE_PARTIAL_TAKE_PROFIT)
        USE_VOLUME_PROFILE_STRATEGY = data.get('use_volume_profile_strategy', USE_VOLUME_PROFILE_STRATEGY)
        
        USE_BB_STOCH_STRATEGY = data.get('use_bb_stoch_strategy', USE_BB_STOCH_STRATEGY)
        USE_MACD_EMA_STRATEGY = data.get('use_macd_ema_strategy', USE_MACD_EMA_STRATEGY)
        USE_EMA_RSI_STRATEGY = data.get('use_ema_rsi_strategy', USE_EMA_RSI_STRATEGY)
        USE_PULLBACK_STRATEGY = data.get('use_pullback_strategy', USE_PULLBACK_STRATEGY)
        
        save_settings_to_redis()
        
        log_and_notify("info", f"✅ [API] تم تحديث إعدادات البوت بنجاح.", "SETTINGS_UPDATE")
        return jsonify({"success": True, "message": "Settings updated successfully."})
    except Exception as e:
        logger.error(f"❌ [API] فشل تحديث الإعدادات: {e}")
        return jsonify({"success": False, "message": "Failed to update settings."}), 500

@app.route('/')
def home():
    """نقطة نهاية لصفحة الويب الرئيسية."""
    return render_template_string("""
        <!doctype html>
        <html lang="ar">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
            <title>Crypto Bot Dashboard</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { background-color: #f8f9fa; color: #333; }
                .container { max-width: 1100px; margin-top: 30px; }
                .card { border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.05); }
                .card-header { background-color: #007bff; color: white; border-top-left-radius: 15px; border-top-right-radius: 15px; }
                .status-dot { height: 12px; width: 12px; background-color: #bbb; border-radius: 50%; display: inline-block; margin-right: 8px; }
                .status-dot.online { background-color: #28a745; }
                .status-dot.offline { background-color: #dc3545; }
                .status-dot.warning { background-color: #ffc107; }
                .btn-toggle { transition: background-color 0.3s; }
                .btn-toggle.on { background-color: #28a745; }
                .btn-toggle.off { background-color: #dc3545; }
                .list-group-item.success { background-color: #d4edda; }
                .list-group-item.danger { background-color: #f8d7da; }
                .list-group-item.warning { background-color: #fff3cd; }
                .table-responsive { max-height: 400px; overflow-y: auto; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="text-center mb-4">
                    <h1 class="display-4">لوحة تحكم البوت</h1>
                </div>

                <!-- Status Card -->
                <div class="card mb-4">
                    <div class="card-header">
                        <h5 class="mb-0">الحالة العامة للبوت</h5>
                    </div>
                    <div class="card-body">
                        <div class="d-flex align-items-center mb-3">
                            <span id="botStatusDot" class="status-dot"></span>
                            <h6 id="botStatusText" class="mb-0">جاري الاتصال...</h6>
                        </div>
                        <div class="row text-center">
                            <div class="col-md-4">
                                <p class="mb-1">وضع التداول</p>
                                <h4 id="tradingModeText"></h4>
                            </div>
                            <div class="col-md-4">
                                <p class="mb-1">الصفقات المفتوحة</p>
                                <h4 id="openTradesCount"></h4>
                            </div>
                            <div class="col-md-4">
                                <p class="mb-1">قوة الاتجاه العام (BTC)</p>
                                <h4 id="marketTrendText"></h4>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Settings Card -->
                <div class="card mb-4">
                    <div class="card-header">
                        <h5 class="mb-0">الإعدادات والتحكم</h5>
                    </div>
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <span class="fw-bold">تفعيل التداول</span>
                            <button id="toggleTradingBtn" class="btn btn-lg btn-toggle btn-outline-secondary">إيقاف</button>
                        </div>
                        <hr>
                        <div class="mb-3">
                            <label for="riskPerTrade" class="form-label">نسبة المخاطرة لكل صفقة (%)</label>
                            <input type="number" step="0.1" class="form-control" id="riskPerTrade" value="0.85">
                        </div>
                        <div class="mb-3">
                            <label for="buyConfidence" class="form-label">عتبة ثقة الشراء</label>
                            <input type="number" step="0.01" class="form-control" id="buyConfidence" value="0.53">
                        </div>
                        <div class="form-check form-switch mb-2">
                            <input class="form-check-input" type="checkbox" id="trailingStopLossToggle">
                            <label class="form-check-label" for="trailingStopLossToggle">تفعيل وقف الخسارة المتحرك</label>
                        </div>
                        <div class="form-check form-switch mb-2">
                            <input class="form-check-input" type="checkbox" id="partialTakeProfitToggle">
                            <label class="form-check-label" for="partialTakeProfitToggle">تفعيل أخذ الربح الجزئي</label>
                        </div>
                        <hr>
                        <h6 class="mt-4">تفعيل الاستراتيجيات</h6>
                        <div class="row">
                            <div class="col-md-6">
                                <div class="form-check form-switch">
                                    <input class="form-check-input" type="checkbox" id="bbStochStrategy">
                                    <label class="form-check-label" for="bbStochStrategy">استراتيجية BB & Stoch</label>
                                </div>
                                <div class="form-check form-switch">
                                    <input class="form-check-input" type="checkbox" id="macdEmaStrategy">
                                    <label class="form-check-label" for="macdEmaStrategy">استراتيجية MACD & EMA</label>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="form-check form-switch">
                                    <input class="form-check-input" type="checkbox" id="emaRsiStrategy">
                                    <label class="form-check-label" for="emaRsiStrategy">استراتيجية EMA & RSI</label>
                                </div>
                                <div class="form-check form-switch">
                                    <input class="form-check-input" type="checkbox" id="pullbackStrategy">
                                    <label class="form-check-label" for="pullbackStrategy">استراتيجية Pullback</label>
                                </div>
                            </div>
                        </div>
                        <div class="d-grid mt-4">
                            <button id="saveSettingsBtn" class="btn btn-primary">حفظ الإعدادات</button>
                        </div>
                    </div>
                </div>

                <!-- Trades and Logs Card -->
                <div class="card mb-4">
                    <div class="card-header">
                        <h5 class="mb-0">الصفقات المفتوحة وسجلات الرفض</h5>
                    </div>
                    <div class="card-body">
                        <ul class="nav nav-tabs" id="myTab" role="tablist">
                            <li class="nav-item" role="presentation">
                                <button class="nav-link active" id="open-trades-tab" data-bs-toggle="tab" data-bs-target="#open-trades" type="button" role="tab" aria-controls="open-trades" aria-selected="true">الصفقات المفتوحة</button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="rejection-logs-tab" data-bs-toggle="tab" data-bs-target="#rejection-logs" type="button" role="tab" aria-controls="rejection-logs" aria-selected="false">سجلات الرفض</button>
                            </li>
                        </ul>
                        <div class="tab-content" id="myTabContent">
                            <div class="tab-pane fade show active" id="open-trades" role="tabpanel" aria-labelledby="open-trades-tab">
                                <div class="table-responsive mt-3">
                                    <table class="table table-striped table-hover">
                                        <thead>
                                            <tr>
                                                <th>الرمز</th>
                                                <th>سعر الدخول</th>
                                                <th>سعر الإيقاف</th>
                                                <th>سعر الهدف</th>
                                                <th>الكمية</th>
                                            </tr>
                                        </thead>
                                        <tbody id="openTradesTableBody">
                                            <tr><td colspan="5" class="text-center">لا توجد صفقات مفتوحة.</td></tr>
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                            <div class="tab-pane fade" id="rejection-logs" role="tabpanel" aria-labelledby="rejection-logs-tab">
                                <ul class="list-group mt-3" id="rejectionLogsList">
                                    <li class="list-group-item text-center">لا توجد سجلات رفض حديثة.</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Notifications Card -->
                <div class="card mb-4">
                    <div class="card-header">
                        <h5 class="mb-0">الإشعارات الأخيرة</h5>
                    </div>
                    <div class="card-body">
                        <ul class="list-group" id="notificationsList">
                            <li class="list-group-item text-center">لا توجد إشعارات.</li>
                        </ul>
                    </div>
                </div>

            </div>

            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
            <script>
                const fetchStatus = async () => {
                    try {
                        const response = await fetch('/status');
                        const data = await response.json();
                        document.getElementById('botStatusDot').className = data.trading_enabled ? 'status-dot online' : 'status-dot offline';
                        document.getElementById('botStatusText').innerText = data.trading_enabled ? 'قيد التشغيل' : 'متوقف';
                        
                        const tradingModeText = data.paper_trading_mode ? 'تجريبي' : 'فعلي';
                        document.getElementById('tradingModeText').innerText = tradingModeText;

                        document.getElementById('openTradesCount').innerText = data.open_trades_count;
                        document.getElementById('marketTrendText').innerText = data.market_state.overall_regime || 'غير متوفر';

                        document.getElementById('toggleTradingBtn').innerText = data.trading_enabled ? 'إيقاف' : 'تفعيل';
                        document.getElementById('toggleTradingBtn').classList.toggle('on', data.trading_enabled);
                        document.getElementById('toggleTradingBtn').classList.toggle('off', !data.trading_enabled);

                        document.getElementById('riskPerTrade').value = data.risk_per_trade_percent;
                        document.getElementById('buyConfidence').value = data.buy_confidence_threshold;
                        document.getElementById('trailingStopLossToggle').checked = data.advanced_features_enabled.trailing_stop_loss;
                        document.getElementById('partialTakeProfitToggle').checked = data.advanced_features_enabled.partial_take_profit;
                        document.getElementById('bbStochStrategy').checked = data.strategies_enabled.bb_stoch;
                        document.getElementById('macdEmaStrategy').checked = data.strategies_enabled.macd_ema;
                        document.getElementById('emaRsiStrategy').checked = data.strategies_enabled.ema_rsi;
                        document.getElementById('pullbackStrategy').checked = data.strategies_enabled.pullback;

                    } catch (error) {
                        console.error('Error fetching status:', error);
                        document.getElementById('botStatusDot').className = 'status-dot offline';
                        document.getElementById('botStatusText').innerText = 'خطأ في الاتصال';
                    }
                };

                const fetchOpenTrades = async () => {
                    try {
                        const response = await fetch('/open_trades');
                        const data = await response.json();
                        const tableBody = document.getElementById('openTradesTableBody');
                        tableBody.innerHTML = '';
                        if (data.open_trades.length === 0) {
                            tableBody.innerHTML = '<tr><td colspan="5" class="text-center">لا توجد صفقات مفتوحة.</td></tr>';
                            return;
                        }
                        data.open_trades.forEach(trade => {
                            const row = document.createElement('tr');
                            row.innerHTML = `
                                <td>${trade.symbol}</td>
                                <td>$${trade.entry_price.toFixed(4)}</td>
                                <td>$${trade.stop_loss.toFixed(4)}</td>
                                <td>$${trade.target_price.toFixed(4)}</td>
                                <td>${trade.quantity.toFixed(4)}</td>
                            `;
                            tableBody.appendChild(row);
                        });
                    } catch (error) {
                        console.error('Error fetching open trades:', error);
                    }
                };

                const fetchRejectionLogs = async () => {
                    try {
                        const response = await fetch('/rejection_logs');
                        const data = await response.json();
                        const list = document.getElementById('rejectionLogsList');
                        list.innerHTML = '';
                        if (data.rejection_logs.length === 0) {
                            list.innerHTML = '<li class="list-group-item text-center">لا توجد سجلات رفض حديثة.</li>';
                            return;
                        }
                        data.rejection_logs.forEach(log => {
                            const listItem = document.createElement('li');
                            listItem.className = 'list-group-item danger';
                            listItem.innerHTML = `<strong>${log.symbol}</strong>: ${log.reason} (${new Date(log.timestamp).toLocaleTimeString()})`;
                            list.appendChild(listItem);
                        });
                    } catch (error) {
                        console.error('Error fetching rejection logs:', error);
                    }
                };

                const fetchNotifications = async () => {
                    try {
                        const response = await fetch('/notifications');
                        const data = await response.json();
                        const list = document.getElementById('notificationsList');
                        list.innerHTML = '';
                        if (data.notifications.length === 0) {
                            list.innerHTML = '<li class="list-group-item text-center">لا توجد إشعارات.</li>';
                            return;
                        }
                        data.notifications.forEach(notif => {
                            const listItem = document.createElement('li');
                            let statusClass = '';
                            if (notif.type.includes('SIGNAL')) statusClass = 'success';
                            else if (notif.type.includes('CLOSE')) statusClass = 'danger';
                            else if (notif.type.includes('TP') || notif.type.includes('SETTINGS')) statusClass = 'warning';
                            listItem.className = `list-group-item ${statusClass}`;
                            listItem.innerHTML = `[${new Date(notif.timestamp).toLocaleTimeString()}] <strong>${notif.type}</strong>: ${notif.message}`;
                            list.appendChild(listItem);
                        });
                    } catch (error) {
                        console.error('Error fetching notifications:', error);
                    }
                };

                document.getElementById('toggleTradingBtn').addEventListener('click', async () => {
                    const isEnabled = document.getElementById('toggleTradingBtn').classList.contains('on');
                    await updateSettings({ trading_enabled: !isEnabled });
                });

                document.getElementById('saveSettingsBtn').addEventListener('click', async () => {
                    const settings = {
                        trading_enabled: document.getElementById('toggleTradingBtn').classList.contains('on'),
                        risk_per_trade_percent: parseFloat(document.getElementById('riskPerTrade').value),
                        buy_confidence_threshold: parseFloat(document.getElementById('buyConfidence').value),
                        use_trailing_stop_loss: document.getElementById('trailingStopLossToggle').checked,
                        use_partial_take_profit: document.getElementById('partialTakeProfitToggle').checked,
                        use_bb_stoch_strategy: document.getElementById('bbStochStrategy').checked,
                        use_macd_ema_strategy: document.getElementById('macdEmaStrategy').checked,
                        use_ema_rsi_strategy: document.getElementById('emaRsiStrategy').checked,
                        use_pullback_strategy: document.getElementById('pullbackStrategy').checked,
                    };
                    await updateSettings(settings);
                });

                const updateSettings = async (settings) => {
                    try {
                        const response = await fetch('/settings', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(settings)
                        });
                        const data = await response.json();
                        if (data.success) {
                            await fetchStatus();
                            alert(data.message);
                        } else {
                            alert(data.message);
                        }
                    } catch (error) {
                        console.error('Error updating settings:', error);
                        alert('Failed to update settings. Please check console for details.');
                    }
                };

                const refreshDashboard = () => {
                    fetchStatus();
                    fetchOpenTrades();
                    fetchRejectionLogs();
                    fetchNotifications();
                };

                setInterval(refreshDashboard, 5000);
                document.addEventListener('DOMContentLoaded', refreshDashboard);
            </script>
        </body>
        </html>
    """)

def start_flask_app():
    """يبدأ خادم Flask."""
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# --- وظائف البوت الرئيسية ---
def process_symbol(symbol: str):
    """
    الدالة الرئيسية التي تحلل البيانات وتطبق استراتيجيات التداول.
    """
    logger.info(f"✨ [Scanner] جاري فحص الرمز: {symbol}")
    
    # 1. جلب وتحليل البيانات التاريخية
    df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
    if df is None or len(df) < 50:
        log_rejection(symbol, "Insufficient Historical Data")
        return
    df_with_features = calculate_all_features(df)
    last = df_with_features.iloc[-1]
    
    # 2. فحص الفلاتر الأساسية
    if last['adx'] < 18 or last['atr'] < (last['close'] * 0.005):
        log_rejection(symbol, "Market Volatility Filter Failed")
        return
        
    # 3. فحص تأكيد الترند على الفريم الأعلى
    htf_df = fetch_historical_data(symbol, HIGHER_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
    if htf_df is None or len(htf_df) < 50:
        log_rejection(symbol, "Insufficient Historical Data")
        return
    htf_df_with_features = calculate_all_features(htf_df)
    htf_trend = calculate_market_trend(htf_df_with_features)
    if htf_trend != 'اتجاه صاعد':
        log_rejection(symbol, "HTF Trend Confirmation Failed")
        return
        
    # 4. تطبيق استراتيجيات الدخول
    signal_found = False
    strategy_name = "N/A"
    signal_details = {}
    
    if USE_VOLUME_PROFILE_STRATEGY and check_volume_profile_strategy(df_with_features):
        signal_found = True
        strategy_name = "Volume Profile"
        signal_details = {"poc": calculate_volume_profile(df_with_features)[0]}
        
    elif USE_BB_STOCH_STRATEGY and last['close'] < last['lower_band'] and last['stoch_rsi'] < 20:
        signal_found = True
        strategy_name = "BB & Stoch"
        
    elif USE_MACD_EMA_STRATEGY and last['macd_hist'] > 0 and last['macd'] > last['macd_signal'] and last['ema_12'] > last['ema_26']:
        signal_found = True
        strategy_name = "MACD & EMA"
        
    elif USE_EMA_RSI_STRATEGY and last['ema_12'] > last['ema_26'] and last['rsi'] > 50 and last['close'] > last['ema_26']:
        signal_found = True
        strategy_name = "EMA & RSI"
        
    elif USE_PULLBACK_STRATEGY and last['close'] > last['ema_26'] and df_with_features.iloc[-2]['close'] < df_with_features.iloc[-2]['ema_26']:
        signal_found = True
        strategy_name = "Pullback"
        
    # 5. إذا تم العثور على إشارة، قم بإنشاء الصفقة
    if signal_found:
        entry_price = float(last['close'])
        stop_loss = entry_price - (entry_price * RISK_PER_TRADE_PERCENT / 100)
        target_price = entry_price + (entry_price * MIN_PROFIT_PERCENT / 100)
        target_price_2 = entry_price + (entry_price * (MIN_PROFIT_PERCENT + 0.5) / 100)
        
        quantity = PAPER_TRADE_SIZE_USDT / entry_price if paper_trading_mode else 0
        
        signal_data = {
            "symbol": symbol, "entry_price": entry_price, "target_price": target_price,
            "stop_loss": stop_loss, "strategy": strategy_name, "quantity": quantity,
            "is_real_trade": not paper_trading_mode, "details": signal_details,
            "target_price_2": target_price_2, "initial_quantity": quantity
        }
        
        if create_signal_db(signal_data):
            with signal_cache_lock:
                open_signals_cache[symbol] = signal_data
            message = (f"📈 *إشارة شراء جديدة!*\\n"
                       f"💱 *العملة:* `{symbol}`\\n"
                       f"🛒 *سعر الدخول:* `{entry_price:.4f}`\\n"
                       f"🛑 *وقف الخسارة:* `{stop_loss:.4f}`\\n"
                       f"🎯 *الهدف 1:* `{target_price:.4f}`\\n"
                       f"🎯 *الهدف 2:* `{target_price_2:.4f}`\\n"
                       f"🧠 *الاستراتيجية:* `{strategy_name}`")
            send_telegram_message(message)

def run_signal_scanner():
    """
    الخيط الرئيسي الذي يفحص الرموز بحثًا عن إشارات جديدة.
    """
    while True:
        try:
            with trading_status_lock:
                if not is_trading_enabled:
                    time.sleep(60)
                    continue

            logger.info("🕵️ [Scanner] بدء دورة المسح...")
            
            with signal_cache_lock:
                open_trades_count = len(open_signals_cache)
                
            if open_trades_count >= MAX_OPEN_TRADES:
                logger.warning(f"⚠️ [Scanner] تجاوز الحد الأقصى للصفقات المفتوحة ({open_trades_count}/{MAX_OPEN_TRADES}). جاري التخطي.")
                time.sleep(60)
                continue

            symbols_to_process = [s for s in validated_symbols_to_scan if s not in open_signals_cache]
            if not symbols_to_process:
                logger.info("💤 [Scanner] لا توجد رموز جديدة للفحص. جاري الانتظار.")
                time.sleep(60)
                continue
            
            for i in range(0, len(symbols_to_process), SYMBOL_PROCESSING_BATCH_SIZE):
                batch = symbols_to_process[i:i + SYMBOL_PROCESSING_BATCH_SIZE]
                for symbol in batch:
                    try:
                        process_symbol(symbol)
                    except Exception as e:
                        logger.error(f"❌ [Scanner] خطأ في معالجة الرمز {symbol}: {e}")
                time.sleep(60) # تأخير بين الدفعات لتجنب حظر API

            logger.info("✅ [Scanner] انتهت دورة المسح بنجاح.")
            
        except Exception as e:
            logger.error(f"❌ [Scanner] حدث خطأ حرج: {e}", exc_info=True)
            time.sleep(60)
        finally:
            gc.collect()

def run_trade_manager():
    """
    الخيط الرئيسي الذي يدير الصفقات المفتوحة.
    """
    while True:
        try:
            with trading_status_lock:
                if not is_trading_enabled:
                    time.sleep(60)
                    continue

            logger.info("🔄 [Manager] بدء دورة إدارة الصفقات...")
            
            with signal_cache_lock:
                symbols_to_manage = list(open_signals_cache.keys())
            
            if not symbols_to_manage:
                logger.info("💤 [Manager] لا توجد صفقات مفتوحة لإدارتها. جاري الانتظار.")
                time.sleep(60)
                continue
                
            for symbol in symbols_to_manage:
                with signal_cache_lock:
                    if symbol not in open_signals_cache:
                        continue
                    signal = open_signals_cache[symbol]
                    
                with live_prices_lock:
                    current_price = live_prices.get(symbol)
                    
                if current_price is None:
                    logger.warning(f"⚠️ [Manager] السعر الحي لـ {symbol} غير متوفر. جاري التخطي.")
                    continue
                    
                # 1. تحديث وقف الخسارة المتحرك
                df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, 1) # جلب شمعة واحدة
                if df is not None and len(df) > 0:
                    df_with_features = calculate_all_features(df)
                    last_price = df_with_features.iloc[-1]['close']
                    # تحديث أعلى سعر تم تحقيقه
                    highest_price = max(signal.get('highest_price', signal['entry_price']), last_price)
                    
                    new_stop_loss = calculate_trailing_stop_loss(df_with_features, signal['entry_price'], signal['stop_loss'], highest_price)
                    if new_stop_loss > signal['stop_loss']:
                        logger.info(f"✅ [Manager] تم تحديث وقف الخسارة المتحرك لـ {symbol} من {signal['stop_loss']:.4f} إلى {new_stop_loss:.4f}.")
                        update_signal_db(signal['id'], {'stop_loss': new_stop_loss})
                        with signal_cache_lock:
                            open_signals_cache[symbol]['stop_loss'] = new_stop_loss
                            open_signals_cache[symbol]['highest_price'] = highest_price

                # 2. فحص شروط الخروج
                if check_for_close_conditions(df, signal, current_price):
                    close_signal_db(signal, current_price, "Exit condition met")
                    with signal_cache_lock:
                        open_signals_cache.pop(symbol, None)
                        
            logger.info("✅ [Manager] انتهت دورة إدارة الصفقات بنجاح.")
            
        except Exception as e:
            logger.error(f"❌ [Manager] حدث خطأ حرج: {e}", exc_info=True)
        finally:
            gc.collect()
            time.sleep(30) # تأخير ثابت بين الدورات

def update_market_state():
    """تحدث حالة السوق العامة واتجاه الترند على الفريمات المختلفة."""
    global current_market_state
    while True:
        try:
            btc_data = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
            if btc_data is None or len(btc_data) < 20:
                logger.warning("⚠️ [Market State] بيانات BTC غير كافية لتقييم حالة السوق.")
                time.sleep(60)
                continue
            
            btc_data_with_features = calculate_all_features(btc_data)
            overall_regime = calculate_market_trend(btc_data_with_features)
            
            trend_details = {}
            for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
                tf_data = fetch_historical_data(BTC_SYMBOL, tf, SIGNAL_GENERATION_LOOKBACK_DAYS)
                if tf_data is not None:
                    tf_data_with_features = calculate_all_features(tf_data)
                    trend_details[tf] = calculate_market_trend(tf_data_with_features)
            
            logger.info(f"✅ [Market State] تم تحديث حالة السوق: الاتجاه العام: {overall_regime}")
            with market_state_lock:
                current_market_state.update({'overall_regime': overall_regime, 'trend_details_by_tf': trend_details, 'last_updated': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')})
            time.sleep(60 * 5)
        except Exception as e:
            logger.error(f"❌ [Market State] A critical error occurred: {e}", exc_info=True)
            time.sleep(60)

# --- نقطة بداية البرنامج ---
if __name__ == '__main__':
    logger.info("="*50 + "\n====== Starting Crypto Trading Bot V12.3.1 ======\n" + "="*50)
    init_db()
    init_redis()
    try:
        client = Client(API_KEY, API_SECRET); client.ping()
        logger.info("✅ [Binance] API connection successful.")
    except Exception as e:
        logger.critical(f"❌ [Binance] API connection failed: {e}"); exit(1)
    get_exchange_info_map()
    validated_symbols_to_scan = get_validated_symbols()
    if not validated_symbols_to_scan:
        logger.critical("❌ No valid symbols to scan. The bot will exit."); exit(1)
    load_open_signals_to_cache()
    load_notifications_to_cache()
    load_settings_from_redis()
    start_websocket()
    
    # بدء الخيوط
    Thread(target=run_signal_scanner, daemon=True).start()
    Thread(target=run_trade_manager, daemon=True).start()
    Thread(target=update_market_state, daemon=True).start()
    Thread(target=start_flask_app, daemon=True).start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("👋 [Bot] Shutting down bot gracefully...")
    finally:
        if ws_manager: ws_manager.stop()
        if conn: conn.close()
        logger.info("✅ [Bot] Bot has been shut down.")
        
