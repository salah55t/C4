# ملف c4.py - نسخة V10.0.0 (تكامل WebSocket وتصميم ليلي)
# --- التغييرات الرئيسية (V10.0.0):
# 1. [ميزة] إضافة تكامل Binance Websocket للحصول على أسعار فورية وتجنب حظر IP.
# 2. [تحسين] تعديل حلقة إدارة الصفقات للاعتماد على بيانات WebSocket بدلاً من طلبات API المتكررة.
# 3. [تصميم] إعادة تصميم كاملة لواجهة التحكم (Dashboard) إلى وضع ليلي احترافي.
# 4. [استقرار] تحسينات عامة على الأداء واستقرار الكود.

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
        logging.FileHandler('crypto_bot_v10_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV10.0.0')

# --- المشفر المخصص لأنواع بيانات NumPy ---
class NpEncoder(json.JSONEncoder):
    """ مشفر مخصص لأنواع بيانات NumPy """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
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

# --- مفاتيح تفعيل الاستراتيجيات ---
USE_BB_STOCH_STRATEGY: bool = True
bb_stoch_strategy_lock = Lock()
USE_MACD_EMA_STRATEGY: bool = True
macd_ema_strategy_lock = Lock()
USE_EMA_RSI_STRATEGY: bool = True
ema_rsi_strategy_lock = Lock()
USE_PULLBACK_STRATEGY: bool = True
pullback_strategy_lock = Lock()
USE_BB_SQUEEZE_STRATEGY: bool = True
USE_BULLISH_MOMENTUM_STRATEGY: bool = True
USE_SR_BREAKOUT_STRATEGY: bool = True

# --- إعدادات عامة ---
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '1h'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 15
BTC_SYMBOL: str = 'BTCUSDT'
SYMBOL_PROCESSING_BATCH_SIZE: int = 5
ATR_TS_MULTIPLIER: float = 2.2
TRADING_FEE_PERCENT: float = 0.1

# --- إعدادات المؤشرات الفنية ---
EMA_FAST_PERIOD: int = 12
EMA_SLOW_PERIOD: int = 26
ADX_PERIOD: int = 10
RSI_PERIOD: int = 10
ATR_PERIOD: int = 10
BTC_CORR_PERIOD: int = 30
REL_VOL_PERIOD: int = 30
MOMENTUM_PERIOD: int = 5
EMA_SLOPE_PERIOD: int = 5
SUPERTREND_ATR_PERIOD: int = 7
SUPERTREND_MULTIPLIER: float = 3.0

# --- إعدادات التحكم في معدل الطلبات (لغير الأسعار) ---
API_REQUEST_DELAY: float = 0.5
API_RETRY_COUNT: int = 3
API_RETRY_DELAY: float = 5.0

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
    "BB_Stoch Strategy Conditions Not Met": "شروط استراتيجية BB+Stoch لم تتحقق",
    "MACD_EMA Strategy Conditions Not Met": "شروط استراتيجية MACD+EMA لم تتحقق",
    "EMA_RSI Strategy Conditions Not Met": "شروط استراتيجية EMA+RSI لم تتحقق",
    "Pullback Strategy Conditions Not Met": "شروط استراتيجية Pullback لم تتحقق",
    "BB Squeeze Strategy Conditions Not Met": "شروط استراتيجية BB Squeeze لم تتحقق",
    "SR Breakout Strategy Conditions Not Met": "شروط استراتيجية اختراق الدعم/المقاومة لم تتحقق",
}

# --- إعداد تطبيق Flask ---
app = Flask(__name__)
CORS(app)

# --- دوال تهيئة الخدمات ---
def init_db(retries: int = 5, delay: int = 5) -> None:
    global conn
    logger.info("[قاعدة البيانات] تهيئة الاتصال...")
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
                        is_real_trade BOOLEAN DEFAULT FALSE, quantity DOUBLE PRECISION, closing_reason TEXT
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL, message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE
                    );
                """)
            conn.commit()
            logger.info("✅ [قاعدة البيانات] الاتصال وتحديث المخطط بنجاح.")
            return
        except Exception as e:
            logger.error(f"❌ [قاعدة البيانات] خطأ أثناء التهيئة (محاولة {attempt + 1}/{retries}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: logger.critical("❌ [قاعدة البيانات] فشل الاتصال.")

def check_db_connection() -> bool:
    global conn
    if conn is None or conn.closed != 0:
        logger.warning("[قاعدة البيانات] الاتصال مغلق، محاولة إعادة الاتصال...")
        init_db()
    try:
        if conn and conn.closed == 0:
            with conn.cursor() as cur: cur.execute("SELECT 1;")
            return True
        return False
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [قاعدة البيانات] فقدان الاتصال: {e}. إعادة الاتصال...")
        try:
            init_db()
            return conn is not None and conn.closed == 0
        except Exception as retry_e:
            logger.error(f"❌ [قاعدة البيانات] فشل إعادة الاتصال: {retry_e}")
            return False

def init_redis() -> None:
    global redis_client
    logger.info("[Redis] تهيئة الاتصال...")
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] تم الاتصال بنجاح.")
    except redis.exceptions.ConnectionError as e:
        logger.warning(f"⚠️ [Redis] فشل الاتصال بـ Redis: {e}. سيتم العمل بدون Redis.")
        redis_client = None

# --- دوال المساعدة والإشعارات ---
def log_and_notify(level: str, message: str, notification_type: str):
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error}
    log_methods.get(level.lower(), logger.info)(message)
    if not check_db_connection() or not conn: return
    try:
        new_notification = {"timestamp": datetime.now(timezone.utc).isoformat(), "type": notification_type, "message": message}
        with notifications_lock: notifications_cache.appendleft(new_notification)
        with conn.cursor() as cur: cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [قاعدة البيانات] فشل حفظ الإشعار: {e}")
        if conn: conn.rollback()

def log_rejection(symbol: str, reason_key: str, details: Optional[Dict] = None):
    reason_ar = REJECTION_REASONS_AR.get(reason_key, reason_key)
    log_message = f"  -> 🚫 [{symbol}] تم الرفض | السبب: {reason_ar} | تفاصيل: {details or {}}"
    logger.info(log_message)
    with rejection_logs_lock:
        rejection_logs_cache.appendleft({
            "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
            "reason": reason_ar, "details": json.loads(json.dumps(details, cls=NpEncoder)) or {}
        })

def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [تليجرام] فشل إرسال الرسالة: {e}")

# --- WebSocket Handler ---
def handle_socket_message(msg):
    """معالج رسائل WebSocket لتحديث الأسعار."""
    global live_prices
    if msg and 'e' in msg and msg['e'] == 'error':
        logger.error(f"❌ [WebSocket] خطأ: {msg['m']}")
        return

    if isinstance(msg, list):
        with live_prices_lock:
            for ticker in msg:
                if 's' in ticker and 'c' in ticker:
                    live_prices[ticker['s']] = float(ticker['c'])

def start_websocket():
    """بدء وإدارة اتصال WebSocket."""
    global ws_manager
    logger.info("🚀 [WebSocket] بدء مدير WebSocket...")
    ws_manager = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    ws_manager.start()
    
    # الاشتراك في بث جميع العملات
    ws_manager.start_ticker_socket(callback=handle_socket_message)
    logger.info("✅ [WebSocket] تم الاشتراك بنجاح في بث الأسعار (!ticker@arr).")

# --- دوال جلب البيانات وحساب المؤشرات ---
def get_exchange_info_map() -> None:
    global exchange_info_map
    if not client: return
    logger.info("ℹ️ [معلومات المنصة] جاري جلب قواعد التداول...")
    try:
        info = client.get_exchange_info()
        exchange_info_map = {s['symbol']: s for s in info['symbols']}
        logger.info(f"✅ [معلومات المنصة] تم تحميل القواعد لـ {len(exchange_info_map)} عملة.")
    except Exception as e:
        logger.error(f"❌ [معلومات المنصة] فشل جلب المعلومات: {e}")

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    if not client: return []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        if not os.path.exists(file_path):
            logger.critical(f"❌ [التحقق من الرموز] ملف العملات '{filename}' غير موجود!")
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        if not raw_symbols:
            logger.warning(f"⚠️ [التحقق من الرموز] ملف العملات '{filename}' فارغ.")
            return []
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        if not exchange_info_map: get_exchange_info_map()
        active = {s for s, info in exchange_info_map.items() if info.get('quoteAsset') == 'USDT' and info.get('status') == 'TRADING'}
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ [التحقق من الرموز] تم العثور على {len(validated)} عملة صالحة للتداول.")
        return validated
    except Exception as e:
        logger.error(f"❌ [التحقق من الرموز] خطأ: {e}", exc_info=True)
        return []

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    time.sleep(API_REQUEST_DELAY)
    for attempt in range(API_RETRY_COUNT):
        try:
            lookback_str = f"{days} day ago UTC"
            klines = client.get_historical_klines(symbol, interval, lookback_str)
            if not klines: return None
            
            klines = [kline[:6] for kline in klines[:-1]]
            cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = pd.DataFrame(klines, columns=cols)
            numeric_cols = {'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'float32'}
            df = df.astype(numeric_cols)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            return df.dropna()
        except BinanceAPIException as e:
            logger.error(f"❌ [جلب البيانات] خطأ API لـ {symbol}: {e}")
            if attempt < API_RETRY_COUNT - 1: time.sleep(API_RETRY_DELAY)
            else: return None
        except Exception as e:
            logger.error(f"❌ [جلب البيانات] خطأ عام لـ {symbol}: {e}")
            if attempt < API_RETRY_COUNT - 1: time.sleep(API_RETRY_DELAY)
            else: return None
    return None

def calculate_all_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df_calc = df.copy()
    df_calc['ema_9'] = df_calc['close'].ewm(span=9, adjust=False).mean()
    df_calc['ema_12'] = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    df_calc['ema_26'] = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    df_calc['sma_50'] = df_calc['close'].rolling(window=50).mean()
    df_calc['volume_sma_20'] = df_calc['volume'].rolling(window=20).mean()
    
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
    df_calc['plus_di'] = plus_di
    df_calc['minus_di'] = minus_di
    
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    
    STOCH_RSI_PERIOD = 14
    rsi_val = df_calc['rsi']
    stoch_rsi = (rsi_val - rsi_val.rolling(STOCH_RSI_PERIOD).min()) / (rsi_val.rolling(STOCH_RSI_PERIOD).max() - rsi_val.rolling(STOCH_RSI_PERIOD).min()).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi.rolling(3).mean() * 100
    
    bb_period = 20
    df_calc['bb_middle'] = df_calc['close'].rolling(window=bb_period).mean()
    bb_std = df_calc['close'].rolling(window=bb_period).std()
    df_calc['bb_upper'] = df_calc['bb_middle'] + (bb_std * 2)
    df_calc['bb_lower'] = df_calc['bb_middle'] - (bb_std * 2)
    df_calc['bb_width'] = (df_calc['bb_upper'] - df_calc['bb_lower']) / df_calc['bb_middle'].replace(0, 1e-9)
    
    exp1 = df_calc['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_calc['close'].ewm(span=26, adjust=False).mean()
    df_calc['macd'] = exp1 - exp2
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()
    
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)
    df_calc[f'roc_{MOMENTUM_PERIOD}'] = (df_calc['close'] / df_calc['close'].shift(MOMENTUM_PERIOD) - 1) * 100
    
    return df_calc.astype('float32', errors='ignore')

# --- دوال تحميل البيانات الأولية ---
def load_open_signals_to_cache():
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status IN ('open', 'updated');")
            open_signals = cur.fetchall()
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in open_signals: open_signals_cache[signal['symbol']] = dict(signal)
            logger.info(f"✅ [تحميل] تم تحميل {len(open_signals)} صفقة مفتوحة إلى الذاكرة المؤقتة.")
    except Exception as e:
        logger.error(f"❌ [تحميل] فشل تحميل الصفقات المفتوحة: {e}")

def load_notifications_to_cache():
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 20;")
            recent = cur.fetchall()
            with notifications_lock:
                notifications_cache.clear()
                for n in reversed(recent):
                    n['timestamp'] = n['timestamp'].isoformat()
                    notifications_cache.appendleft(dict(n))
            logger.info(f"✅ [تحميل] تم تحميل {len(notifications_cache)} إشعار إلى الذاكرة المؤقتة.")
    except Exception as e:
        logger.error(f"❌ [تحميل] فشل تحميل الإشعارات: {e}")

# --- منطق التداول والفلاتر ---
def check_market_volatility_filter(df: pd.DataFrame) -> bool:
    if len(df) < 50: return False
    last = df.iloc[-1]
    if 'atr' not in last or 'close' not in last or last['close'] == 0: return False
    atr_percent = (last['atr'] / last['close']) * 100
    if atr_percent < 0.5 or atr_percent > 5.0:
        log_rejection(df.name, "Market Volatility Filter Failed", {"atr_percent": f"{atr_percent:.2f}"})
        return False
    return True

def check_trend_strength_filter(df: pd.DataFrame) -> bool:
    if len(df) < 50: return False
    last = df.iloc[-1]
    if 'adx' not in last or f'roc_{MOMENTUM_PERIOD}' not in last: return False
    if last['adx'] < 18:
        log_rejection(df.name, "Trend Strength Filter Failed", {"reason": "ADX too low", "adx": f"{last['adx']:.2f}"})
        return False
    return True

def is_htf_bullish_confirmation(symbol: str, htf: str = '1h', lookback: int = 200) -> bool:
    try:
        df = fetch_historical_data(symbol, htf, days=40) 
        if df is None or len(df) < lookback: return False
        df['ema50']  = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        last = df.iloc[-1]
        return last['close'] > last['ema50'] and last['ema50'] > last['ema200']
    except Exception as e:
        logger.error(f"❌ [HTF Confirm] خطأ في {symbol}: {e}")
        return False

# --- استراتيجيات التداول ---
def check_bb_stoch_strategy_revised(df: pd.DataFrame) -> bool:
    if len(df) < 15: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    return (prev['low'] <= prev['bb_lower'] * 1.001 and
            last['close'] > last['open'] and
            last['stoch_rsi_k'] < 35 and last['stoch_rsi_k'] > prev['stoch_rsi_k'] and
            last['rsi'] > 25 and
            last['volume'] > df['volume'].rolling(10).mean().iloc[-1] * 1.2)

def check_macd_ema_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 3: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    return (prev['macd'] < prev['macd_signal'] and last['macd'] > last['macd_signal'] and
            last['close'] > last['ema_12'] and
            last['adx'] > 18)

def check_ema_rsi_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 2: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    return (prev['ema_9'] < prev['ema_12'] and last['ema_9'] > last['ema_12'] and
            last['rsi'] > 52 and
            last['close'] > last['ema_26'])

def check_pullback_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 2: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    return (last['close'] > last['ema_12'] and last['ema_12'] > last['ema_26'] and
            prev['macd'] < prev['macd_signal'] and last['macd'] > last['macd_signal'])

def check_bb_squeeze_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 100: return False
    last = df.iloc[-1]
    squeeze_threshold = df['bb_width'].rolling(100).quantile(0.20).iloc[-1]
    return (last['bb_width'] < squeeze_threshold and
            last['close'] > last['bb_upper'] and
            last.get('relative_volume', 0) > 1.25)

# --- أنماط الشموع ---
def is_bullish_reversal_pattern(df: pd.DataFrame) -> bool:
    if len(df) < 2: return False
    c2, c3 = df.iloc[-2], df.iloc[-1]
    # Hammer
    body = abs(c3['open'] - c3['close'])
    if body > 0:
        lower_wick = c3['close'] - c3['low'] if c3['open'] < c3['close'] else c3['open'] - c3['low']
        upper_wick = c3['high'] - c3['close'] if c3['open'] < c3['close'] else c3['high'] - c3['open']
        if lower_wick > 2 * body and upper_wick < body: return True
    # Bullish Engulfing
    if (c2['close'] < c2['open'] and c3['close'] > c3['open'] and
        c3['close'] > c2['open'] and c3['open'] < c2['close']): return True
    return False

# --- دوال إنشاء الصفقات ---
def create_paper_trade_signal(symbol: str, df: pd.DataFrame, strategy_name: str) -> None:
    try:
        last = df.iloc[-1]
        entry_price = last['close']
        atr = last['atr']
        stop_loss = entry_price - (atr * ATR_TS_MULTIPLIER)
        
        if entry_price <= stop_loss:
            logger.error(f"❌ [{symbol}] لا يمكن حساب حجم الصفقة، وقف الخسارة ({stop_loss}) غير صحيح.")
            return

        risk_per_unit = entry_price - stop_loss
        target_price = entry_price + (risk_per_unit * 1.5) 
        
        message = f"📊 *فتح صفقة ورقية جديدة*\n💱 *العملة:* `{symbol}`\n📈 *الاستراتيجية:* {strategy_name}\n" \
                  f"📌 *الدخول:* `{entry_price:.4f}`\n🎯 *الهدف:* `{target_price:.4f}`\n🛑 *الوقف:* `{stop_loss:.4f}`"
        send_telegram_message(message)
        log_and_notify("info", f"تم فتح صفقة ورقية لـ {symbol} باستراتيجية {strategy_name}", "PAPER_TRADE_OPEN")
        
        if check_db_connection() and conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO signals (symbol, entry_price, target_price, stop_loss, status, 
                                           strategy_name, is_real_trade, signal_details) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
                    """, (
                        symbol, float(entry_price), float(target_price), float(stop_loss), 'open',
                        strategy_name, False, json.dumps({
                            "atr": float(atr), "rsi": float(last['rsi']), "timestamp": datetime.now(timezone.utc).isoformat()
                        }, cls=NpEncoder)
                    ))
                    new_id = cur.fetchone()['id']
                conn.commit()
                with signal_cache_lock:
                    open_signals_cache[symbol] = {
                        'id': new_id, 'symbol': symbol, 'entry_price': float(entry_price), 
                        'target_price': float(target_price), 'stop_loss': float(stop_loss), 
                        'status': 'open', 'strategy_name': strategy_name, 'is_real_trade': False,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                logger.info(f"✅ تم حفظ الصفقة الورقية لـ {symbol} في قاعدة البيانات.")
            except Exception as e:
                logger.error(f"❌ خطأ في حفظ الصفقة الورقية لـ {symbol}: {e}", exc_info=True)
                if conn: conn.rollback()
    except Exception as e:
        logger.error(f"❌ خطأ في إنشاء الصفقة الورقية لـ {symbol}: {e}", exc_info=True)

# --- قوالب HTML (تصميم ليلي) ---
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم بوت التداول</title>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #121212; --bg-surface: #1e1e1e; --primary: #BB86FC;
            --primary-variant: #3700B3; --secondary: #03DAC6; --text-light: #e0e0e0;
            --text-medium: #a0a0a0; --success: #4CAF50; --danger: #F44336;
            --warning: #FFC107; --bullish: #26a69a; --bearish: #ef5350;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg-dark); color: var(--text-light);
            font-family: 'Tajawal', sans-serif; line-height: 1.6;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        header {
            background-color: var(--bg-surface); padding: 15px 25px;
            border-radius: 12px; margin-bottom: 25px; display: flex;
            justify-content: space-between; align-items: center;
            border: 1px solid #2a2a2a;
        }
        .header-title { font-size: 24px; font-weight: 700; color: var(--primary); }
        .status-indicator { display: flex; align-items: center; gap: 15px; }
        .status-dot { width: 12px; height: 12px; border-radius: 50%; background-color: var(--danger);
                      box-shadow: 0 0 8px var(--danger); }
        .status-dot.active { background-color: var(--success); box-shadow: 0 0 8px var(--success); }
        .btn {
            background-color: var(--primary-variant); color: white; border: none;
            padding: 10px 20px; border-radius: 8px; cursor: pointer;
            transition: background-color 0.3s, transform 0.2s;
            font-weight: 700; text-decoration: none;
        }
        .btn:hover { background-color: var(--primary); transform: translateY(-2px); }
        .btn.stop { background-color: var(--danger); }
        .btn.stop:hover { background-color: #d32f2f; }
        .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 20px; }
        .card {
            background-color: var(--bg-surface); border-radius: 12px;
            padding: 20px; border: 1px solid #2a2a2a;
            display: flex; flex-direction: column;
        }
        .card-header {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #333;
        }
        .card-title { font-size: 18px; font-weight: 700; color: var(--text-light); }
        .scrollable-content { overflow-y: auto; max-height: 400px; padding-right: 10px; }
        .scrollable-content::-webkit-scrollbar { width: 6px; }
        .scrollable-content::-webkit-scrollbar-track { background: #2a2a2a; }
        .scrollable-content::-webkit-scrollbar-thumb { background: var(--primary); border-radius: 3px; }
        .item {
            padding: 12px; border-radius: 8px; margin-bottom: 10px;
            border-left: 4px solid var(--primary); background-color: #252525;
        }
        .item-header { display: flex; justify-content: space-between; margin-bottom: 5px; }
        .item-title { font-weight: 700; }
        .item-time { font-size: 12px; color: var(--text-medium); }
        .item-content { font-size: 14px; color: var(--text-light); }
        .state-item { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px; }
        .state-label { font-weight: bold; color: var(--text-medium); }
        .state-value.Bullish { color: var(--bullish); font-weight: 700; }
        .state-value.Bearish { color: var(--bearish); font-weight: 700; }
        .state-value.Sideways { color: var(--text-medium); }
        .signal-item.paper { border-left-color: var(--secondary); }
        .notification-item.info { border-left-color: var(--primary); }
        .notification-item.warning { border-left-color: var(--warning); }
        .notification-item.error, .notification-item.trading_status { border-left-color: var(--danger); }
        .rejection-item { border-left-color: var(--warning); }
        .progress-bar-container { background-color: #333; border-radius: 10px; height: 10px; overflow: hidden; margin-top: 10px; direction: ltr; }
        .progress-bar { height: 100%; transition: width 0.4s ease-in-out; border-radius: 10px; }
        .footer { text-align: center; margin-top: 30px; padding: 15px; color: var(--text-medium); font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-title">بوت التداول V10.0</div>
            <div class="status-indicator">
                <div class="status-dot {{ 'active' if trading_enabled else '' }}"></div>
                <span>{{ 'نشط' if trading_enabled else 'متوقف' }}</span>
                <button class="btn {{ 'stop' if trading_enabled else '' }}" onclick="toggleTrading()">{{ 'إيقاف' if trading_enabled else 'تشغيل' }}</button>
                <a href="/settings" class="btn">الإعدادات</a>
            </div>
        </header>
        <div class="dashboard-grid">
            <div class="card">
                <div class="card-header"><div class="card-title">حالة السوق (BTC)</div></div>
                <div class="item">
                    <div class="state-item">
                        <span class="state-label">النظام العام:</span>
                        <span class="state-value {{ market_state.get('overall_regime', 'N/A') }}">{{ market_state.get('overall_regime', 'N/A') }}</span>
                    </div>
                    {% for tf, details in market_state.get('trend_details_by_tf', {}).items() %}
                    <div class="state-item">
                        <span class="state-label">اتجاه {{ tf }}:</span>
                        <span class="state-value {{ details.get('trend', 'N/A') }}">{{ details.get('trend', 'N/A') }} (RSI: {{ "%.1f"|format(details.get('rsi', 0)) }})</span>
                    </div>
                    {% endfor %}
                    <div class="state-item" style="margin-top:10px; font-size: 12px; color: var(--text-medium);">
                        <span>آخر تحديث:</span><span>{{ market_state.get('last_updated', 'N/A') }}</span>
                    </div>
                </div>
            </div>
            <div class="card">
                <div class="card-header"><div class="card-title">الإشارات المفتوحة ({{ open_signals|length }})</div></div>
                <div class="scrollable-content">
                {% if open_signals %}{% for symbol, signal in open_signals.items() %}
                <div class="item signal-item {{ 'paper' if not signal.get('is_real_trade') else '' }}">
                    <div class="item-header">
                        <div class="item-title">{{ symbol }} <span style="font-size:12px; color: var(--text-medium);">({{ 'ورقية' if not signal.get('is_real_trade') else 'حقيقية' }})</span></div>
                        <div class="item-time">{{ signal.get('strategy_name', '') }}</div>
                    </div>
                    <div class="item-content" style="font-size: 13px;">
                        دخول: {{ "%.4f"|format(signal.get('entry_price', 0)) }} | حالي: {{ "%.4f"|format(signal.get('current_price', 0)) }}<br>
                        هدف: {{ "%.4f"|format(signal.get('target_price', 0)) }} | وقف: {{ "%.4f"|format(signal.get('stop_loss', 0)) }}
                    </div>
                    <div class="progress-bar-container">
                        {% set progress = signal.get('progress', 0) %}
                        {% if progress >= 0 %}
                            <div class="progress-bar" style="width: {{ [progress, 100]|min }}%; background-color: var(--success);"></div>
                        {% else %}
                            <div class="progress-bar" style="width: {{ [progress|abs, 100]|min }}%; background-color: var(--danger); float: right;"></div>
                        {% endif %}
                    </div>
                </div>
                {% endfor %}{% else %}<div style="text-align: center; padding: 20px; color: var(--text-medium);">لا توجد إشارات مفتوحة</div>{% endif %}
                </div>
            </div>
            <div class="card">
                <div class="card-header"><div class="card-title">الإشعارات الأخيرة</div></div>
                <div class="scrollable-content">
                {% if notifications %}{% for notif in notifications %}
                <div class="item notification-item {{ notif.get('type', 'info').lower() }}">
                    <div class="item-header"><div class="item-title">{{ notif.get('type', 'INFO') }}</div><div class="item-time">{{ notif.get('timestamp', '')[:16] }}</div></div>
                    <div class="item-content">{{ notif.get('message', '') }}</div>
                </div>
                {% endfor %}{% else %}<div style="text-align: center; padding: 20px; color: var(--text-medium);">لا توجد إشعارات</div>{% endif %}
                </div>
            </div>
            <div class="card">
                <div class="card-header"><div class="card-title">سجل الرفض</div></div>
                <div class="scrollable-content">
                {% if rejections %}{% for rej in rejections %}
                <div class="item rejection-item">
                    <div class="item-header"><div class="item-title">{{ rej.get('symbol', 'N/A') }}</div><div class="item-time">{{ rej.get('timestamp', '')[:16] }}</div></div>
                    <div class="item-content">{{ rej.get('reason', 'N/A') }}</div>
                </div>
                {% endfor %}{% else %}<div style="text-align: center; padding: 20px; color: var(--text-medium);">لا يوجد رفض</div>{% endif %}
                </div>
            </div>
        </div>
        <div class="footer"><div>بوت التداول الإلكتروني V10.0 - فريم 15 دقيقة</div></div>
    </div>
    <script>
        function showAlert(message, type = 'info') {
            const alertBox = document.createElement('div');
            const bgColor = type === 'success' ? 'var(--success)' : type === 'error' ? 'var(--danger)' : 'var(--primary)';
            Object.assign(alertBox.style, {
                position: 'fixed', bottom: '20px', left: '20px', padding: '15px 25px',
                borderRadius: '8px', color: 'white', zIndex: '1000', backgroundColor: bgColor,
                boxShadow: '0 4px 15px rgba(0,0,0,0.2)', transform: 'translateY(100px)', opacity: '0',
                transition: 'transform 0.4s ease, opacity 0.4s ease'
            });
            alertBox.innerText = message;
            document.body.appendChild(alertBox);
            setTimeout(() => {
                alertBox.style.transform = 'translateY(0)';
                alertBox.style.opacity = '1';
            }, 10);
            setTimeout(() => {
                alertBox.style.transform = 'translateY(100px)';
                alertBox.style.opacity = '0';
                setTimeout(() => alertBox.remove(), 400);
            }, 4000);
        }
        function toggleTrading() {
            fetch('/toggle_trading', { method: 'POST' })
            .then(res => res.json()).then(data => {
                showAlert(data.message, data.success ? 'success' : 'error');
                if(data.success) setTimeout(() => location.reload(), 1500);
            }).catch(error => showAlert('خطأ في الاتصال بالخادم', 'error'));
        }
        setInterval(() => location.reload(), 60000);
    </script>
</body>
</html>
"""

# --- مسارات Flask ---
@app.route('/')
def dashboard():
    with signal_cache_lock: open_signals = dict(sorted(open_signals_cache.items()))
    with market_state_lock: market_state = current_market_state.copy()
    with trading_status_lock: trading_enabled = is_trading_enabled
    with notifications_lock: notifications = list(notifications_cache)
    with rejection_logs_lock: rejections = list(rejection_logs_cache)
    return render_template_string(DASHBOARD_TEMPLATE, 
                                market_state=market_state,
                                trading_enabled=trading_enabled,
                                open_signals=open_signals,
                                notifications=notifications,
                                rejections=rejections)

@app.route('/toggle_trading', methods=['POST'])
def toggle_trading():
    global is_trading_enabled
    with trading_status_lock:
        is_trading_enabled = not is_trading_enabled
        status = "مفعل" if is_trading_enabled else "معطل"
        mode = "حقيقي" if not paper_trading_mode else "ورقي"
        log_and_notify("info", f"تم {status} التداول (الوضع: {mode})", "TRADING_STATUS")
        send_telegram_message(f"⚙️ تم {status} التداول (الوضع: {mode})")
        return jsonify({"success": True, "message": f"تم {status} التداول"})

# --- حلقات العمل الخلفية ---
def main_bot_loop():
    logger.info("🚀 [الحلقة الرئيسية] بدء حلقة البحث عن الإشارات...")
    btc_df_cache = None
    last_btc_fetch = 0

    while True:
        try:
            with trading_status_lock:
                if not is_trading_enabled:
                    time.sleep(10)
                    continue

            logger.info("="*20 + " بدء دورة فحص جديدة " + "="*20)

            if time.time() - last_btc_fetch > 300: # تحديث بيانات BTC كل 5 دقائق
                btc_df_cache = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                last_btc_fetch = time.time()
                if btc_df_cache is None:
                    logger.warning("⚠️ [الحلقة الرئيسية] لا يمكن جلب بيانات BTC، سيتم تخطي دورة الفحص.")
                    time.sleep(60)
                    continue

            for i in range(0, len(validated_symbols_to_scan), SYMBOL_PROCESSING_BATCH_SIZE):
                batch = validated_symbols_to_scan[i:i + SYMBOL_PROCESSING_BATCH_SIZE]
                for symbol in batch:
                    logger.info(f"--- تحليل [{symbol}] ---")
                    with signal_cache_lock:
                        if symbol in open_signals_cache:
                            logger.info(f"  -> تخطي، توجد صفقة مفتوحة بالفعل.")
                            continue
                    
                    df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df is None or df.empty:
                        log_rejection(symbol, "Insufficient Historical Data")
                        continue
                    
                    df_featured = calculate_all_features(df, btc_df_cache)
                    df_featured.name = symbol

                    if not check_market_volatility_filter(df_featured): continue
                    if not check_trend_strength_filter(df_featured): continue
                    if not is_htf_bullish_confirmation(symbol, HIGHER_TIMEFRAME):
                        log_rejection(symbol, "HTF Trend Confirmation Failed")
                        continue

                    strategy_found = None
                    active_strategies = {
                        "BB+Stoch": (USE_BB_STOCH_STRATEGY, check_bb_stoch_strategy_revised),
                        "MACD+EMA": (USE_MACD_EMA_STRATEGY, check_macd_ema_strategy),
                        "EMA+RSI": (USE_EMA_RSI_STRATEGY, check_ema_rsi_strategy),
                        "Pullback": (USE_PULLBACK_STRATEGY, check_pullback_strategy),
                        "BB Squeeze": (USE_BB_SQUEEZE_STRATEGY, check_bb_squeeze_strategy),
                    }
                    
                    for name, (is_active, func) in active_strategies.items():
                        if is_active and func(df_featured):
                            strategy_found = name
                            break 
                    
                    if strategy_found:
                        logger.info(f"  -> 🌟 [{symbol}] إشارة مؤكدة! الاستراتيجية: {strategy_found}")
                        if is_bullish_reversal_pattern(df_featured):
                            create_paper_trade_signal(symbol, df_featured, strategy_found)
                        else:
                            log_rejection(symbol, "Bullish Reversal Candle Pattern Failed")
            
            logger.info("="*20 + " اكتملت دورة الفحص " + "="*20)
            time.sleep(60 * 5) # انتظار 5 دقائق قبل الدورة التالية

        except Exception as e:
            logger.error(f"❌ [الحلقة الرئيسية] حدث خطأ فادح: {e}", exc_info=True)
            time.sleep(60)

def close_signal(signal: Dict, closing_price: float, reason: str):
    symbol = signal['symbol']
    entry_price = signal['entry_price']
    profit_percentage = ((closing_price - entry_price) / entry_price) * 100
    if signal.get('is_real_trade', False):
        profit_percentage -= 2 * TRADING_FEE_PERCENT

    logger.info(f"🔔 [إغلاق صفقة] {symbol} | السبب: {reason} | الربح: {profit_percentage:.2f}%")
    
    if check_db_connection() and conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE signals SET status = 'closed', closing_price = %s, closed_at = %s, 
                                      profit_percentage = %s, closing_reason = %s
                    WHERE id = %s
                """, (closing_price, datetime.now(timezone.utc), profit_percentage, reason, signal['id']))
            conn.commit()
            log_and_notify("info", f"تم إغلاق صفقة {symbol} بربح {profit_percentage:.2f}%", "TRADE_CLOSED")
            result_emoji = "✅" if profit_percentage >= 0 else "🔻"
            reason_text = "تحقيق الهدف" if reason == "TP_HIT" else "وقف الخسارة"
            send_telegram_message(f"{result_emoji} *إغلاق صفقة ورقية*\n💱 *العملة:* `{symbol}`\n*السبب:* {reason_text}\n*الربح:* `{profit_percentage:.2f}%`")
        except Exception as e:
            logger.error(f"❌ [قاعدة البيانات] فشل تحديث إغلاق الصفقة لـ {symbol}: {e}")
            if conn: conn.rollback()

    with signal_cache_lock:
        if symbol in open_signals_cache:
            del open_signals_cache[symbol]

def manage_open_trades_loop():
    logger.info("🚀 [إدارة الصفقات] بدء حلقة إدارة الصفقات المفتوحة...")
    while True:
        try:
            with signal_cache_lock:
                open_signals_copy = list(open_signals_cache.values())

            if not open_signals_copy:
                time.sleep(5)
                continue
            
            with live_prices_lock:
                current_prices = live_prices.copy()

            for signal in open_signals_copy:
                symbol = signal.get('symbol')
                if not symbol: continue

                current_price = current_prices.get(symbol)
                if not current_price: continue

                entry_price = signal.get('entry_price', 0)
                target_price = signal.get('target_price', 0)
                stop_loss = signal.get('stop_loss', 0)

                # حساب نسبة التقدم وتحديث الكاش
                total_tp_dist = target_price - entry_price
                total_sl_dist = entry_price - stop_loss
                current_dist = current_price - entry_price
                
                progress = 0
                if current_dist >= 0 and total_tp_dist > 0:
                    progress = (current_dist / total_tp_dist) * 100
                elif current_dist < 0 and total_sl_dist > 0:
                    progress = (current_dist / total_sl_dist) * 100

                with signal_cache_lock:
                    if symbol in open_signals_cache:
                        open_signals_cache[symbol]['current_price'] = current_price
                        open_signals_cache[symbol]['progress'] = progress

                # التحقق من الهدف والوقف
                if current_price >= target_price:
                    close_signal(signal, target_price, "TP_HIT")
                    continue 
                if current_price <= stop_loss:
                    close_signal(signal, stop_loss, "SL_HIT")
                    continue
            
            time.sleep(1) # فحص سريع كل ثانية

        except Exception as e:
            logger.error(f"❌ [إدارة الصفقات] حدث خطأ: {e}", exc_info=True)
            time.sleep(10)

def update_market_state_loop():
    logger.info("🚀 [حالة السوق] بدء حلقة تحديث حالة السوق...")
    while True:
        try:
            trend_details = {}
            bullish_count = 0
            for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
                days = 10 if tf == '15m' else 30 if tf == '1h' else 90
                btc_df = fetch_historical_data(BTC_SYMBOL, tf, days)
                if btc_df is None or len(btc_df) < 50:
                    trend_details[tf] = {"trend": "Unknown", "rsi": 50}
                    continue
                btc_df['ema_fast'] = btc_df['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
                btc_df['ema_slow'] = btc_df['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
                delta = btc_df['close'].diff()
                gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
                loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
                rsi = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
                last = btc_df.iloc[-1]
                trend = "Sideways"
                if last['close'] > last['ema_slow'] and last['ema_fast'] > last['ema_slow'] and rsi.iloc[-1] > 55:
                    trend = "Bullish"; bullish_count += 1
                elif last['close'] < last['ema_slow'] and last['ema_fast'] < last['ema_slow'] and rsi.iloc[-1] < 45:
                    trend = "Bearish"
                trend_details[tf] = {"trend": trend, "rsi": rsi.iloc[-1]}

            overall_regime = "Sideways"
            if bullish_count >= 2: overall_regime = "Bullish"
            elif bullish_count == 0: overall_regime = "Bearish"

            with market_state_lock:
                current_market_state['overall_regime'] = overall_regime
                current_market_state['trend_details_by_tf'] = trend_details
                current_market_state['last_updated'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            
            logger.info(f"✅ [حالة السوق] تم التحديث. النظام العام: {overall_regime}")
            time.sleep(60 * 5)
        except Exception as e:
            logger.error(f"❌ [حالة السوق] حدث خطأ: {e}", exc_info=True)
            time.sleep(60)

# --- نقطة بداية البرنامج ---
if __name__ == '__main__':
    logger.info("="*50)
    logger.info("====== بدء تشغيل بوت التداول الإلكتروني V10.0 ======")
    logger.info("="*50)

    init_db()
    init_redis()
    try:
        client = Client(API_KEY, API_SECRET)
        client.ping()
        logger.info("✅ [Binance] الاتصال بالمنصة ناجح.")
    except Exception as e:
        logger.critical(f"❌ [Binance] فشل الاتصال بالمنصة: {e}")
        exit(1)

    get_exchange_info_map()
    validated_symbols_to_scan = get_validated_symbols()
    if not validated_symbols_to_scan:
        logger.critical("❌ لا توجد عملات صالحة للمسح. سيتم إيقاف البوت.")
        exit(1)
    
    load_open_signals_to_cache()
    load_notifications_to_cache()

    # بدء الـ WebSocket
    start_websocket()

    # بدء حلقات العمل
    main_loop_thread = Thread(target=main_bot_loop, daemon=True)
    manage_trades_thread = Thread(target=manage_open_trades_loop, daemon=True)
    market_state_thread = Thread(target=update_market_state_loop, daemon=True)
    
    main_loop_thread.start()
    manage_trades_thread.start()
    market_state_thread.start()

    logger.info("🌐 [Flask] بدء تشغيل واجهة المستخدم على http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
