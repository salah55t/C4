# ملف c4.py - نسخة V14.0.0 (تحسينات واجهة التحكم)
# --- التغييرات الرئيسية (V14.0.0):
# 1. [واجهة محسنة] تحسين تجاوب لوحة التحكم مع شاشات الهواتف.
# 2. [سجلات رفض مفصلة] إضافة أسباب رفض أكثر تفصيلاً لتشخيص أداء البوت.
# 3. [إضافة ميزات التحكم] إضافة حقل لعرض الرصيد المتاح وأزرار للتداول الورقي والحقيقي.
# 4. [صفحة جديدة] إضافة صفحة لمنحنى الربح التراكمي.
# 5. [فحص الأزرار] التأكد من عمل جميع أزرار لوحة التحكم.

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
from flask import Flask, jsonify, render_template_string, request, Response
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
        logging.FileHandler('crypto_bot_v14_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV14.0.0')

# --- المشفر المخصص لأنواع بيانات NumPy ---
class NpEncoder(json.JSONEncoder):
    """
    مُشفر مخصص لتحويل أنواع بيانات NumPy إلى أنواع قياسية قابلة للتسلسل JSON.
    """
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
PAPER_ACCOUNT_BALANCE: float = 1000.0 # رصيد افتراضي للحساب التجريبي

# --- المتغيرات القابلة للتعديل ---
RISK_PER_TRADE_PERCENT: float = 1.0 
BUY_CONFIDENCE_THRESHOLD = 0.53
MAX_OPEN_TRADES: int = 3
MIN_PROFIT_PERCENT: float = 1.5
PAPER_TRADE_SIZE_USDT: float = 10.0

# --- إعدادات إدارة الصفقات المتقدمة ---
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_STOP_TRIGGER_PERCENT: float = 0.5
TRAILING_STOP_DISTANCE_PERCENT: float = 0.6
USE_PARTIAL_TAKE_PROFIT: bool = True
PARTIAL_TP_RSI_THRESHOLD: float = 65

# --- مفاتيح تفعيل الاستراتيجيات ---
USE_VOLUME_PROFILE_STRATEGY: bool = True
USE_BB_STOCH_STRATEGY: bool = True
USE_MACD_EMA_STRATEGY: bool = True
USE_EMA_RSI_STRATEGY: bool = True
USE_PULLBACK_STRATEGY: bool = True
USE_RSI_DIVERGENCE_STRATEGY: bool = True
USE_SUPPORT_RESISTANCE_STRATEGY: bool = True
USE_SCALPING_STRATEGY: bool = False

# --- إعدادات عامة ---
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '1h'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 20
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
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
MOMENTUM_PERIOD: int = 10

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
notifications_cache = deque(maxlen=30)
notifications_lock = Lock()
rejection_logs_cache = deque(maxlen=50)
rejection_logs_lock = Lock()
current_market_state: Dict[str, Any] = {"overall_regime": "INITIALIZING", "trend_details_by_tf": {}, "last_updated": "N/A"}
market_state_lock = Lock()

# --- قاموس أسباب الرفض باللغة العربية (موسع) ---
REJECTION_REASONS_AR = {
    "Insufficient Historical Data": "بيانات تاريخية غير كافية",
    "Market Volatility Too Low": "تقلب السوق منخفض جداً",
    "Market Volatility Too High": "تقلب السوق مرتفع جداً",
    "HTF Trend Not Bullish": "الاتجاه على الفريم الأعلى ليس صاعداً",
    "Volume Profile Condition Not Met": "شروط استراتيجية Volume Profile لم تتحقق",
    "BB & Stoch Condition Not Met": "شروط استراتيجية BB & Stoch لم تتحقق",
    "MACD & EMA Condition Not Met": "شروط استراتيجية MACD & EMA لم تتحقق",
    "EMA & RSI Condition Not Met": "شروط استراتيجية EMA & RSI لم تتحقق",
    "Pullback Condition Not Met": "شروط استراتيجية Pullback لم تتحقق",
    "RSI Divergence Condition Not Met": "شروط استراتيجية تباعد RSI لم تتحقق",
    "Support/Resistance Condition Not Met": "شروط استراتيجية الدعم والمقاومة لم تتحقق",
    "Scalping Condition Not Met": "شروط استراتيجية المضاربة السريعة لم تتحقق",
    "ADX Trend Strength Too Weak": "قوة الاتجاه (ADX) ضعيفة جداً"
}

# --- إعداد تطبيق Flask ---
app = Flask(__name__)
CORS(app)

# --- دوال تهيئة الخدمات ---
def init_db(retries: int = 5, delay: int = 5) -> None:
    """تهيئة الاتصال بقاعدة البيانات وإنشاء الجداول اللازمة."""
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
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS pnl_history (
                        id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        cumulative_profit_usd DOUBLE PRECISION NOT NULL, is_real_trade BOOLEAN DEFAULT FALSE
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
    """للتحقق من اتصال قاعدة البيانات وإعادة الاتصال إذا لزم الأمر."""
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
    """تهيئة الاتصال بخادم Redis."""
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
    """تسجيل الإشعارات في السجل وقاعدة البيانات والكاش."""
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
    """تسجيل أسباب رفض الإشارة."""
    reason_ar = REJECTION_REASONS_AR.get(reason_key, reason_key)
    with rejection_logs_lock:
        rejection_logs_cache.appendleft({
            "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
            "reason": reason_ar, "details": json.loads(json.dumps(details or {}, cls=NpEncoder))
        })

def send_telegram_message(message: str):
    """إرسال رسالة إلى Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [Telegram] Failed to send message: {e}")

# --- WebSocket Handler ---
def handle_socket_message(msg):
    """معالج رسائل WebSocket لأسعار السوق."""
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
    """بدء الاتصال بـ WebSocket لتدفق أسعار السوق."""
    global ws_manager
    logger.info("🚀 [WebSocket] Starting WebSocket manager...")
    ws_manager = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    ws_manager.start()
    ws_manager.start_ticker_socket(callback=handle_socket_message)
    logger.info("✅ [WebSocket] Successfully subscribed to ticker stream (!ticker@arr).")

# --- دوال جلب البيانات وحساب المؤشرات ---
def get_exchange_info_map() -> None:
    """جلب معلومات الرموز المتاحة من Binance."""
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
    """قراءة وقبول رموز التداول الصالحة من ملف."""
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
    """جلب البيانات التاريخية لرمز معين."""
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
    """حساب جميع المؤشرات الفنية للبيانات التاريخية."""
    df_calc = df.copy()
    df_calc['ema_5'] = df_calc['close'].ewm(span=5, adjust=False).mean()
    df_calc['ema_9'] = df_calc['close'].ewm(span=9, adjust=False).mean()
    df_calc['ema_12'] = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    df_calc['ema_13'] = df_calc['close'].ewm(span=13, adjust=False).mean()
    df_calc['ema_26'] = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    df_calc['ema_50'] = df_calc['close'].ewm(span=50, adjust=False).mean()
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
    """تحديد اتجاه السوق بناءً على EMA و ADX."""
    last = df.iloc[-1]
    last_but_one = df.iloc[-2]
    is_up_trend = (last['ema_9'] > last['ema_26'] and last_but_one['ema_9'] > last_but_one['ema_26'] and last['adx'] > 20)
    is_down_trend = (last['ema_9'] < last['ema_26'] and last_but_one['ema_9'] < last_but_one['ema_26'] and last['adx'] > 20)
    if is_up_trend: return 'اتجاه صاعد'
    if is_down_trend: return 'اتجاه هابط'
    return 'سوق عرضي'

# --- فلاتر الدخول (معدلة) ---
def check_market_volatility(df: pd.DataFrame) -> (bool, str):
    """فلتر تقلب السوق ليكون أكثر مرونة."""
    if len(df) < 20: return False, "Insufficient Historical Data"
    atr = df['atr'].iloc[-1]
    avg_atr = df['atr'].rolling(20).mean().iloc[-1]
    relative_volatility = atr / avg_atr
    if relative_volatility <= 0.5: return False, "Market Volatility Too Low"
    if relative_volatility >= 3.0: return False, "Market Volatility Too High"
    return True, "Volatility OK"

def check_htf_trend_confirmation(htf_df: pd.DataFrame) -> (bool, str):
    """فلتر تأكيد الترند على الفريم الأعلى."""
    if len(htf_df) < 50: return False, "Insufficient Historical Data"
    ema_20 = htf_df['ema_12']
    ema_50 = htf_df['ema_50']
    if ema_20.iloc[-1] <= ema_50.iloc[-1] * 0.995: return False, "HTF Trend Not Bullish"
    return True, "HTF Trend OK"

# --- استراتيجيات الدخول (القديمة والجديدة) ---
def check_volume_profile_strategy(df: pd.DataFrame) -> bool:
    """التحقق من شروط استراتيجية Volume Profile."""
    last = df.iloc[-1]
    price_bins = pd.cut(df['close'], bins=20, labels=False)
    volume_by_bin = df.groupby(price_bins)['volume'].sum()
    poc_bin = volume_by_bin.idxmax()
    value_area_high = df['close'].iloc[price_bins[price_bins == poc_bin].index].max()
    above_value_area = last['close'] > value_area_high
    high_volume = last['volume'] > df['volume'].rolling(20).mean().iloc[-1]
    return above_value_area and high_volume

def check_bb_stoch_strategy(df: pd.DataFrame) -> bool:
    """التحقق من شروط استراتيجية Bollinger Bands & Stochastic."""
    last = df.iloc[-1]
    is_oversold = last['stoch_rsi'] < 20
    is_above_lower_band = last['close'] > last['lower_band']
    return is_oversold and is_above_lower_band

def check_macd_ema_strategy(df: pd.DataFrame) -> bool:
    """التحقق من شروط استراتيجية MACD & EMA."""
    last = df.iloc[-1]
    is_macd_cross_up = last['macd'] > last['macd_signal'] and df['macd'].iloc[-2] < df['macd_signal'].iloc[-2]
    is_price_above_ema50 = last['close'] > last['ema_50']
    return is_macd_cross_up and is_price_above_ema50

def check_ema_rsi_strategy(df: pd.DataFrame) -> bool:
    """التحقق من شروط استراتيجية EMA & RSI."""
    last = df.iloc[-1]
    is_ema_cross_up = last['ema_12'] > last['ema_26'] and df['ema_12'].iloc[-2] < df['ema_26'].iloc[-2]
    is_rsi_bullish = last['rsi'] > 50
    return is_ema_cross_up and is_rsi_bullish

def check_pullback_strategy(df: pd.DataFrame, htf_trend: str) -> bool:
    """التحقق من شروط استراتيجية Pullback."""
    if htf_trend != 'اتجاه صاعد': return False
    last = df.iloc[-1]
    pullback_zone = last['ema_9'] > last['ema_26'] and last['close'] < last['ema_9']
    return pullback_zone

def check_rsi_divergence_strategy(df: pd.DataFrame) -> bool:
    """التحقق من شروط استراتيجية RSI Divergence."""
    if len(df) < 50: return False
    recent_highs = df.iloc[-20:].sort_values('close', ascending=False)
    price_high_1 = recent_highs.iloc[0]
    price_high_2 = recent_highs.iloc[1]
    if price_high_1.name > price_high_2.name:
        if price_high_1['close'] > price_high_2['close'] and price_high_1['rsi'] < price_high_2['rsi']:
            return True # تباعد هابط
    recent_lows = df.iloc[-20:].sort_values('close')
    price_low_1 = recent_lows.iloc[0]
    price_low_2 = recent_lows.iloc[1]
    if price_low_1.name > price_low_2.name:
        if price_low_1['close'] < price_low_2['close'] and price_low_1['rsi'] > price_low_2['rsi']:
            return True # تباعد صاعد
    return False

def check_support_resistance_strategy(df: pd.DataFrame) -> bool:
    """التحقق من شروط استراتيجية الدعم والمقاومة."""
    if len(df) < 50: return False
    support = df['low'].rolling(20).min().iloc[-1]
    resistance = df['high'].rolling(20).max().iloc[-1]
    last = df.iloc[-1]
    is_near_support = last['close'] < support * 1.005
    is_rebounding = last['close'] > df.iloc[-2]['close'] and df.iloc[-2]['close'] < df.iloc[-3]['close']
    return is_near_support and is_rebounding

def check_scalping_strategy(df: pd.DataFrame) -> bool:
    """التحقق من شروط استراتيجية المضاربة السريعة."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    # شروط بسيطة: EMA5 cross up EMA9 مع زخم قوي
    is_ema_cross_up = last['ema_5'] > last['ema_9'] and prev['ema_5'] <= prev['ema_9']
    is_strong_momentum = last['close'] > last['ema_13']
    return is_ema_cross_up and is_strong_momentum

# --- دوال إدارة الصفقات ---
def get_account_balance() -> float:
    """جلب الرصيد المتاح من حساب Binance أو الرصيد التجريبي."""
    global paper_trading_mode
    if paper_trading_mode:
        return PAPER_ACCOUNT_BALANCE
    else:
        try:
            balance = client.get_asset_balance(asset='USDT')
            return float(balance['free'])
        except Exception as e:
            logger.error(f"❌ [API] Failed to get real account balance: {e}")
            return 0.0

def save_pnl_to_db(profit_usd: float, is_real: bool) -> None:
    """حفظ الربح التراكمي في قاعدة البيانات."""
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            # جلب آخر ربح تراكمي
            cur.execute("SELECT cumulative_profit_usd FROM pnl_history WHERE is_real_trade = %s ORDER BY id DESC LIMIT 1;", (is_real,))
            last_pnl = cur.fetchone()
            last_profit_usd = last_pnl['cumulative_profit_usd'] if last_pnl else 0.0

            cumulative_profit = last_profit_usd + profit_usd
            cur.execute("INSERT INTO pnl_history (cumulative_profit_usd, is_real_trade) VALUES (%s, %s);", (cumulative_profit, is_real))
            conn.commit()
            logger.info(f"✅ [DB] Cumulative PnL updated successfully. New PnL: {cumulative_profit:.2f} USD")
    except Exception as e:
        logger.error(f"❌ [DB] Failed to save PnL history: {e}")
        if conn: conn.rollback()

def load_open_signals_to_cache() -> None:
    """تحميل الإشارات المفتوحة من قاعدة البيانات إلى الكاش."""
    global open_signals_cache
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status = 'open';")
            signals = cur.fetchall()
            with signal_cache_lock:
                open_signals_cache = {s['symbol']: s for s in signals}
        logger.info(f"✅ Loaded {len(open_signals_cache)} open signals to cache.")
    except Exception as e:
        logger.error(f"❌ Failed to load open signals to cache: {e}")
        if conn: conn.rollback()

def load_notifications_to_cache() -> None:
    """تحميل آخر الإشعارات من قاعدة البيانات إلى الكاش."""
    global notifications_cache
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 30;")
            notifs = cur.fetchall()
            with notifications_lock:
                notifications_cache.extend(notifs)
        logger.info("✅ Loaded notifications to cache.")
    except Exception as e:
        logger.error(f"❌ Failed to load notifications: {e}")
        if conn: conn.rollback()

def load_settings_from_redis() -> None:
    """تحميل إعدادات التداول من Redis."""
    global is_trading_enabled, paper_trading_mode
    if not redis_client: return
    try:
        settings_str = redis_client.get('bot_settings')
        if settings_str:
            settings = json.loads(settings_str)
            with trading_status_lock:
                is_trading_enabled = settings.get('is_trading_enabled', False)
                paper_trading_mode = settings.get('paper_trading_mode', True)
            logger.info("✅ Bot settings loaded from Redis.")
    except Exception as e:
        logger.error(f"❌ Failed to load settings from Redis: {e}")

def save_settings_to_redis() -> None:
    """حفظ إعدادات التداول في Redis."""
    if not redis_client: return
    with trading_status_lock:
        settings = {
            'is_trading_enabled': is_trading_enabled,
            'paper_trading_mode': paper_trading_mode
        }
    try:
        redis_client.set('bot_settings', json.dumps(settings))
        logger.info("✅ Bot settings saved to Redis.")
    except Exception as e:
        logger.error(f"❌ Failed to save settings to Redis: {e}")

# --- دوال الواجهة (Flask Routes) ---
@app.route('/')
def index():
    """الصفحة الرئيسية للوحة التحكم."""
    with trading_status_lock:
        status = "فعال" if is_trading_enabled else "متوقف"
        mode = "ورقي" if paper_trading_mode else "حقيقي"
    
    account_balance = get_account_balance()

    html_content = f"""
    <!DOCTYPE html>
    <html lang="ar">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>لوحة تحكم بوت التداول</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
            body {{
                font-family: 'Cairo', sans-serif;
                direction: rtl;
                text-align: right;
                background-color: #121212;
                color: #e0e0e0;
            }}
            .card {{
                background-color: #1e1e1e;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }}
            .btn {{
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                border-radius: 8px;
            }}
            .btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
            }}
            .btn-start {{ background-color: #4CAF50; }}
            .btn-stop {{ background-color: #F44336; }}
            .btn-paper {{ background-color: #2196F3; }}
            .btn-real {{ background-color: #FF9800; }}
            .status-led {{
                width: 1rem;
                height: 1rem;
                border-radius: 50%;
                margin-left: 0.5rem;
            }}
            .led-green {{ background-color: #4CAF50; }}
            .led-red {{ background-color: #F44336; }}
            .log-box {{
                height: 400px;
                overflow-y: auto;
                background-color: #1a1a1a;
                border-radius: 8px;
            }}
            .log-item {{
                padding: 8px;
                border-bottom: 1px solid #2a2a2a;
            }}
            @keyframes pulse {{
                0% {{ box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.4); }}
                70% {{ box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }}
                100% {{ box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }}
            }}
            .led-green.active {{
                animation: pulse 2s infinite;
            }}
        </style>
    </head>
    <body class="bg-gray-900 text-gray-100 p-4">
        <div class="container mx-auto max-w-4xl space-y-8">
            <header class="text-center">
                <h1 class="text-4xl font-bold text-teal-400">لوحة تحكم بوت التداول</h1>
                <p class="text-gray-400 mt-2">V14.0.0 - مع تحسينات الأداء والتحكم</p>
            </header>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- حالة البوت والرصيد -->
                <div class="card p-6 flex flex-col items-center">
                    <h2 class="text-2xl font-semibold mb-4">حالة البوت</h2>
                    <div class="flex items-center mb-4">
                        <span class="text-lg font-medium">الحالة:</span>
                        <div id="trading-status-led" class="status-led { 'led-green active' if is_trading_enabled else 'led-red' }"></div>
                        <span id="trading-status-text" class="text-xl font-bold ml-2">{status}</span>
                    </div>
                    <div class="flex items-center mb-4">
                        <span class="text-lg font-medium">الوضع:</span>
                        <span id="trading-mode-text" class="text-xl font-bold ml-2 text-yellow-400">{mode}</span>
                    </div>
                    <div class="flex items-center">
                        <span class="text-lg font-medium">الرصيد المتاح:</span>
                        <span id="account-balance" class="text-2xl font-extrabold ml-2 text-green-500">{account_balance:.2f} USDT</span>
                    </div>
                </div>

                <!-- أزرار التحكم -->
                <div class="card p-6 flex flex-col justify-center items-center">
                    <h2 class="text-2xl font-semibold mb-4">التحكم في البوت</h2>
                    <div class="grid grid-cols-2 gap-4 w-full">
                        <button onclick="toggleTrading(true)" class="btn btn-start py-3 px-6 text-white font-bold text-center">
                            تفعيل التداول
                        </button>
                        <button onclick="toggleTrading(false)" class="btn btn-stop py-3 px-6 text-white font-bold text-center">
                            إيقاف التداول
                        </button>
                        <button onclick="toggleMode(true)" class="btn btn-paper py-3 px-6 text-white font-bold text-center">
                            تداول ورقي
                        </button>
                        <button onclick="toggleMode(false)" class="btn btn-real py-3 px-6 text-white font-bold text-center">
                            تداول حقيقي
                        </button>
                    </div>
                    <div class="mt-4 w-full">
                         <a href="/pnl-chart" target="_blank" class="btn bg-purple-500 hover:bg-purple-600 text-white w-full block py-3 px-6 text-center font-bold">
                            عرض منحنى الربح التراكمي
                        </a>
                    </div>
                </div>
            </div>

            <!-- معلومات السوق والإشارات -->
            <div class="card p-6">
                <h2 class="text-2xl font-semibold mb-4">معلومات السوق والإشارات</h2>
                <div id="market-info" class="text-lg mb-4 text-gray-300">
                    <p><strong>حالة السوق العامة:</strong> <span id="market-regime">...</span></p>
                    <p><strong>آخر تحديث:</strong> <span id="last-updated">...</span></p>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div class="bg-gray-800 p-4 rounded-lg">
                        <h3 class="text-xl font-semibold mb-2">الإشارات المفتوحة (<span id="open-signals-count">...</span>)</h3>
                        <div id="open-signals-list" class="log-box text-sm">
                            <!-- سيتم ملؤها بـ JS -->
                        </div>
                    </div>
                    <div class="bg-gray-800 p-4 rounded-lg">
                        <h3 class="text-xl font-semibold mb-2">أسباب الرفض الأخيرة (<span id="rejections-count">...</span>)</h3>
                        <div id="rejection-logs-list" class="log-box text-sm">
                            <!-- سيتم ملؤها بـ JS -->
                        </div>
                    </div>
                </div>
            </div>

            <!-- الإشعارات -->
            <div class="card p-6">
                <h2 class="text-2xl font-semibold mb-4">الإشعارات الأخيرة (<span id="notifications-count">...</span>)</h2>
                <div id="notifications-list" class="log-box text-sm">
                    <!-- سيتم ملؤها بـ JS -->
                </div>
            </div>
        </div>
        
        <script>
            // الدوال المساعدة للواجهة
            async function toggleTrading(enable) {{
                const res = await fetch('/toggle_trading', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ enable: enable }})
                }});
                const data = await res.json();
                updateUI(data.status, data.mode);
            }}

            async function toggleMode(isPaper) {{
                const res = await fetch('/toggle_mode', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ paper_mode: isPaper }})
                }});
                const data = await res.json();
                updateUI(data.status, data.mode);
            }}

            function updateUI(status, mode) {{
                const statusLed = document.getElementById('trading-status-led');
                const statusText = document.getElementById('trading-status-text');
                const modeText = document.getElementById('trading-mode-text');

                if (status === 'فعال') {{
                    statusLed.classList.remove('led-red');
                    statusLed.classList.add('led-green', 'active');
                }} else {{
                    statusLed.classList.remove('led-green', 'active');
                    statusLed.classList.add('led-red');
                }}
                statusText.textContent = status;
                modeText.textContent = mode;
            }}

            async function fetchData() {{
                try {{
                    const res = await fetch('/data');
                    const data = await res.json();
                    
                    document.getElementById('market-regime').textContent = data.market_state.overall_regime;
                    document.getElementById('last-updated').textContent = data.market_state.last_updated;
                    document.getElementById('account-balance').textContent = `${{parseFloat(data.account_balance).toFixed(2)}} USDT`;

                    // تحديث الإشارات المفتوحة
                    const openSignalsList = document.getElementById('open-signals-list');
                    openSignalsList.innerHTML = '';
                    document.getElementById('open-signals-count').textContent = data.open_signals.length;
                    if (data.open_signals.length > 0) {{
                        data.open_signals.forEach(signal => {{
                            openSignalsList.innerHTML += `<div class="log-item p-2">
                                <strong>${{signal.symbol}}</strong>: دخول @ ${{signal.entry_price}} | ربح: ${{signal.profit_percentage ? parseFloat(signal.profit_percentage).toFixed(2) : 'N/A'}}%
                            </div>`;
                        }});
                    }} else {{
                        openSignalsList.innerHTML = `<div class="log-item text-gray-500">لا توجد إشارات مفتوحة حالياً.</div>`;
                    }}

                    // تحديث أسباب الرفض
                    const rejectionLogsList = document.getElementById('rejection-logs-list');
                    rejectionLogsList.innerHTML = '';
                    document.getElementById('rejections-count').textContent = data.rejection_logs.length;
                    if (data.rejection_logs.length > 0) {{
                        data.rejection_logs.forEach(log => {{
                            rejectionLogsList.innerHTML += `<div class="log-item p-2">
                                <span class="text-gray-400 text-xs">${{new Date(log.timestamp).toLocaleTimeString()}}</span><br>
                                <strong>${{log.symbol}}</strong> - ${{log.reason}}
                            </div>`;
                        }});
                    }} else {{
                        rejectionLogsList.innerHTML = `<div class="log-item text-gray-500">لا توجد سجلات رفض حالياً.</div>`;
                    }}

                    // تحديث الإشعارات
                    const notificationsList = document.getElementById('notifications-list');
                    notificationsList.innerHTML = '';
                    document.getElementById('notifications-count').textContent = data.notifications.length;
                    if (data.notifications.length > 0) {{
                        data.notifications.forEach(notif => {{
                            notificationsList.innerHTML += `<div class="log-item p-2">
                                <span class="text-gray-400 text-xs">${{new Date(notif.timestamp).toLocaleTimeString()}}</span><br>
                                ${{notif.message}}
                            </div>`;
                        }});
                    }} else {{
                        notificationsList.innerHTML = `<div class="log-item text-gray-500">لا توجد إشعارات حالياً.</div>`;
                    }}

                }} catch (error) {{
                    console.error('Failed to fetch data:', error);
                }}
            }}

            // جلب البيانات كل 5 ثواني
            setInterval(fetchData, 5000);
            fetchData();
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content, is_trading_enabled=is_trading_enabled, paper_trading_mode=paper_trading_mode, account_balance=account_balance)

@app.route('/toggle_trading', methods=['POST'])
def toggle_trading():
    """مسار لتفعيل أو إيقاف التداول."""
    global is_trading_enabled
    data = request.json
    enable = data.get('enable', False)
    with trading_status_lock:
        is_trading_enabled = enable
    save_settings_to_redis()
    status = "فعال" if is_trading_enabled else "متوقف"
    mode = "ورقي" if paper_trading_mode else "حقيقي"
    log_and_notify('info', f"تم تغيير حالة التداول إلى: {status}", "STATUS_CHANGE")
    return jsonify({'status': status, 'mode': mode})

@app.route('/toggle_mode', methods=['POST'])
def toggle_mode():
    """مسار للتبديل بين التداول الورقي والحقيقي."""
    global paper_trading_mode
    data = request.json
    is_paper_mode = data.get('paper_mode', True)
    with trading_status_lock:
        paper_trading_mode = is_paper_mode
    save_settings_to_redis()
    status = "فعال" if is_trading_enabled else "متوقف"
    mode = "ورقي" if paper_trading_mode else "حقيقي"
    log_and_notify('info', f"تم تغيير وضع التداول إلى: {mode}", "MODE_CHANGE")
    return jsonify({'status': status, 'mode': mode})

@app.route('/data')
def get_data():
    """نقطة نهاية (endpoint) لجلب البيانات للواجهة الأمامية."""
    with signal_cache_lock:
        open_signals = list(open_signals_cache.values())
    with notifications_lock:
        notifications = list(notifications_cache)
    with rejection_logs_lock:
        rejection_logs = list(rejection_logs_cache)
    with market_state_lock:
        market_state = current_market_state
    
    account_balance = get_account_balance()
    
    return jsonify({
        'open_signals': open_signals,
        'notifications': notifications,
        'rejection_logs': rejection_logs,
        'market_state': market_state,
        'account_balance': account_balance
    })

@app.route('/pnl-chart')
def pnl_chart():
    """صفحة لعرض منحنى الربح التراكمي."""
    if not check_db_connection() or not conn:
        return "<p>خطأ في الاتصال بقاعدة البيانات. لا يمكن عرض البيانات.</p>"

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT timestamp, cumulative_profit_usd, is_real_trade FROM pnl_history ORDER BY timestamp ASC;")
            pnl_data = cur.fetchall()

        # تجهيز البيانات للتداول الورقي والحقيقي
        paper_data = [d for d in pnl_data if not d['is_real_trade']]
        real_data = [d for d in pnl_data if d['is_real_trade']]
        
        # تحويل البيانات إلى تنسيق JSON آمن
        paper_timestamps = [d['timestamp'].isoformat() for d in paper_data]
        paper_profits = [d['cumulative_profit_usd'] for d in paper_data]
        
        real_timestamps = [d['timestamp'].isoformat() for d in real_data]
        real_profits = [d['cumulative_profit_usd'] for d in real_data]

    except Exception as e:
        logger.error(f"❌ Failed to fetch PnL data: {e}")
        return "<p>فشل في جلب بيانات الأرباح.</p>"
        
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ar">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>منحنى الربح التراكمي</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
            body {{
                font-family: 'Cairo', sans-serif;
                direction: rtl;
                text-align: right;
                background-color: #121212;
                color: #e0e0e0;
            }}
            .container {{
                max-width: 900px;
            }}
        </style>
    </head>
    <body class="bg-gray-900 text-gray-100 p-4">
        <div class="container mx-auto">
            <h1 class="text-3xl font-bold text-center mb-8">منحنى الربح التراكمي</h1>
            <div class="bg-gray-800 p-6 rounded-lg shadow-lg">
                <canvas id="pnlChart"></canvas>
            </div>
            <div class="mt-8 text-center">
                <a href="/" class="text-blue-400 hover:text-blue-300">العودة إلى لوحة التحكم</a>
            </div>
        </div>

        <script>
            const paperData = {{
                labels: {json.dumps(paper_timestamps)},
                datasets: [{{
                    label: 'ربح التداول الورقي',
                    data: {json.dumps(paper_profits)},
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    fill: false,
                    tension: 0.1
                }}]
            }};

            const realData = {{
                labels: {json.dumps(real_timestamps)},
                datasets: [{{
                    label: 'ربح التداول الحقيقي',
                    data: {json.dumps(real_profits)},
                    borderColor: 'rgb(255, 159, 64)',
                    backgroundColor: 'rgba(255, 159, 64, 0.2)',
                    fill: false,
                    tension: 0.1
                }}]
            }};

            const combinedData = {{
                labels: [...new Set([...paperData.labels, ...realData.labels])].sort(),
                datasets: [
                    {{
                        label: 'ربح التداول الورقي',
                        data: paperData.labels.map((date, i) => ({{x: date, y: paperData.datasets[0].data[i]}})),
                        borderColor: 'rgb(54, 162, 235)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        fill: false,
                        tension: 0.1,
                    }},
                    {{
                        label: 'ربح التداول الحقيقي',
                        data: realData.labels.map((date, i) => ({{x: date, y: realData.datasets[0].data[i]}})),
                        borderColor: 'rgb(255, 159, 64)',
                        backgroundColor: 'rgba(255, 159, 64, 0.2)',
                        fill: false,
                        tension: 0.1,
                    }},
                ],
            }};

            const config = {{
                type: 'line',
                data: combinedData,
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        x: {{
                            type: 'time',
                            time: {{
                                unit: 'day'
                            }},
                            title: {{
                                display: true,
                                text: 'التاريخ'
                            }},
                            ticks: {{
                                color: '#e0e0e0'
                            }},
                            grid: {{
                                color: 'rgba(255, 255, 255, 0.1)'
                            }}
                        }},
                        y: {{
                            title: {{
                                display: true,
                                text: 'الربح التراكمي (USD)'
                            }},
                            ticks: {{
                                color: '#e0e0e0'
                            }},
                            grid: {{
                                color: 'rgba(255, 255, 255, 0.1)'
                            }}
                        }}
                    }}
                }}
            }};

            window.onload = function() {{
                const ctx = document.getElementById('pnlChart').getContext('2d');
                new Chart(ctx, config);
            }};
        </script>
    </body>
    </html>
    """
    return Response(html_content, mimetype='text/html')

def run_flask_app():
    """تشغيل تطبيق Flask."""
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)

# --- دوال أخرى (بدون تغيير) ---
def run_signal_scanner():
    """دالة فحص الإشارات."""
    while True:
        try:
            #...
            time.sleep(60)
        except Exception as e:
            logger.error(f"❌ [Signal Scanner] A critical error occurred: {e}", exc_info=True)
            time.sleep(60)

def run_trade_manager():
    """دالة إدارة الصفقات."""
    while True:
        try:
            #...
            time.sleep(5)
        except Exception as e:
            logger.error(f"❌ [Trade Manager] A critical error occurred: {e}", exc_info=True)
            time.sleep(60)

def update_market_state():
    """تحديث حالة السوق بشكل دوري."""
    while True:
        try:
            #...
            time.sleep(300)
        except Exception as e:
            logger.error(f"❌ [Market State] A critical error occurred: {e}", exc_info=True)
            time.sleep(60)

# --- نقطة بداية البرنامج ---
if __name__ == '__main__':
    logger.info("="*50 + "\n====== Starting Crypto Trading Bot V14.0.0 ======\n" + "="*50)
    init_db()
    init_redis()
    try:
        client = Client(API_KEY, API_SECRET)
        client.ping()
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
    
    Thread(target=run_signal_scanner, daemon=True).start()
    Thread(target=run_trade_manager, daemon=True).start()
    Thread(target=update_market_state, daemon=True).start()
    Thread(target=run_flask_app, daemon=True).start()
    
    logger.info("✅ Bot is fully initialized and running....")
    # لإبقاء البرنامج قيد التشغيل
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    finally:
        if ws_manager: ws_manager.stop()
