# ملف c4.py - نسخة V11.0.0 (ميزات متقدمة لإدارة الصفقات)
# --- التغييرات الرئيسية (V11.0.0):
# 1. [ميزة] إضافة زر إغلاق يدوي لكل صفقة في واجهة التحكم.
# 2. [ميزة] تطبيق آلية وقف خسارة متحرك (Trailing Stop-Loss) للصفقات الرابحة.
# 3. [ميزة] تطبيق آلية أخذ ربح جزئي عند الهدف الأول مع تحليل قوة الاتجاه (RSI) لتحديد إمكانية الاستمرار للهدف الثاني.
# 4. [تحسين] حساب كمية الصفقات الورقية بناءً على حجم ثابت (10 USDT).
# 5. [تحسين] تحديث مخطط قاعدة البيانات لاستيعاب الحقول الجديدة (مثل الهدف الثاني ووقف الخسارة المتحرك).

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
        logging.FileHandler('crypto_bot_v11_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV11.0.0')

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
PAPER_TRADE_SIZE_USDT: float = 10.0 # حجم الصفقة الورقية بالـ USDT

# --- إعدادات إدارة الصفقات المتقدمة ---
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_STOP_TRIGGER_PERCENT: float = 0.4 # نسبة الربح لتفعيل الوقف المتحرك
TRAILING_STOP_DISTANCE_PERCENT: float = 0.5 # المسافة التي يتبعها الوقف المتحرك خلف السعر
USE_PARTIAL_TAKE_PROFIT: bool = True
PARTIAL_TP_RSI_THRESHOLD: float = 60 # حد الـ RSI للاستمرار للهدف الثاني

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
                # إنشاء الجداول الأساسية
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
                # إضافة الأعمدة الجديدة للإصدار 11 (إذا لم تكن موجودة)
                alter_commands = [
                    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS target_price_2 DOUBLE PRECISION;",
                    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS initial_quantity DOUBLE PRECISION;",
                ]
                for command in alter_commands:
                    cur.execute(command)
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
    except (OperationalError, InterfaceError):
        logger.error(f"❌ [قاعدة البيانات] فقدان الاتصال. إعادة الاتصال...")
        init_db()
        return conn is not None and conn.closed == 0

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
    with rejection_logs_lock:
        rejection_logs_cache.appendleft({
            "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
            "reason": reason_ar, "details": json.loads(json.dumps(details or {}, cls=NpEncoder))
        })

def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [تليجرام] فشل إرسال الرسالة: {e}")

# --- WebSocket Handler ---
def handle_socket_message(msg):
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
    global ws_manager
    logger.info("🚀 [WebSocket] بدء مدير WebSocket...")
    ws_manager = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    ws_manager.start()
    ws_manager.start_ticker_socket(callback=handle_socket_message)
    logger.info("✅ [WebSocket] تم الاشتراك بنجاح في بث الأسعار (!ticker@arr).")

# --- دوال جلب البيانات وحساب المؤشرات ---
def get_exchange_info_map() -> None:
    global exchange_info_map
    if not client: return
    try:
        info = client.get_exchange_info()
        exchange_info_map = {s['symbol']: s for s in info['symbols']}
    except Exception as e:
        logger.error(f"❌ [معلومات المنصة] فشل جلب المعلومات: {e}")

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    if not client: return []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        if not os.path.exists(file_path):
            logger.critical(f"❌ ملف العملات '{filename}' غير موجود!"); return []
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        if not exchange_info_map: get_exchange_info_map()
        active = {s for s, info in exchange_info_map.items() if info.get('quoteAsset') == 'USDT' and info.get('status') == 'TRADING'}
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ تم العثور على {len(validated)} عملة صالحة للتداول.")
        return validated
    except Exception as e:
        logger.error(f"❌ [التحقق من الرموز] خطأ: {e}"); return []

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
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
    except Exception as e:
        logger.error(f"❌ [جلب البيانات] خطأ لـ {symbol}: {e}"); return None

def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
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
    df_calc['stoch_rsi_k'] = stoch_rsi.rolling(3).mean() * 100
    bb_period = 20
    df_calc['bb_middle'] = df_calc['close'].rolling(window=bb_period).mean()
    bb_std = df_calc['close'].rolling(window=bb_period).std()
    df_calc['bb_upper'] = df_calc['bb_middle'] + (bb_std * 2)
    df_calc['bb_lower'] = df_calc['bb_middle'] - (bb_std * 2)
    exp1 = df_calc['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_calc['close'].ewm(span=26, adjust=False).mean()
    df_calc['macd'] = exp1 - exp2
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()
    df_calc[f'roc_{MOMENTUM_PERIOD}'] = (df_calc['close'] / df_calc['close'].shift(MOMENTUM_PERIOD) - 1) * 100
    return df_calc

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
    except Exception as e:
        logger.error(f"❌ [تحميل] فشل تحميل الإشعارات: {e}")

def load_settings_from_redis():
    global RISK_PER_TRADE_PERCENT, BUY_CONFIDENCE_THRESHOLD, MAX_OPEN_TRADES, MIN_PROFIT_PERCENT
    global USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY
    if not redis_client: return
    try:
        settings_data = redis_client.get('trading_settings')
        if settings_data:
            settings = json.loads(settings_data)
            with risk_per_trade_lock: RISK_PER_TRADE_PERCENT = settings.get('RISK_PER_TRADE_PERCENT', 0.85)
            with buy_confidence_lock: BUY_CONFIDENCE_THRESHOLD = settings.get('BUY_CONFIDENCE_THRESHOLD', 0.53)
            MAX_OPEN_TRADES = settings.get('MAX_OPEN_TRADES', 3)
            MIN_PROFIT_PERCENT = settings.get('MIN_PROFIT_PERCENT', 0.8)
        strategies_data = redis_client.get('strategy_settings')
        if strategies_data:
            strategies = json.loads(strategies_data)
            with bb_stoch_strategy_lock: USE_BB_STOCH_STRATEGY = strategies.get('USE_BB_STOCH_STRATEGY', True)
            with macd_ema_strategy_lock: USE_MACD_EMA_STRATEGY = strategies.get('USE_MACD_EMA_STRATEGY', True)
            with ema_rsi_strategy_lock: USE_EMA_RSI_STRATEGY = strategies.get('USE_EMA_RSI_STRATEGY', True)
            with pullback_strategy_lock: USE_PULLBACK_STRATEGY = strategies.get('USE_PULLBACK_STRATEGY', True)
        logger.info("✅ تم تحميل الإعدادات المحفوظة من Redis بنجاح.")
    except Exception as e:
        logger.error(f"❌ خطأ في تحميل الإعدادات من Redis: {e}")

# --- منطق التداول والفلاتر ---
def check_market_volatility_filter(df: pd.DataFrame) -> bool:
    last = df.iloc[-1]
    atr_percent = (last['atr'] / last['close']) * 100
    if atr_percent < 0.5 or atr_percent > 5.0:
        log_rejection(df.name, "Market Volatility Filter Failed", {"atr_percent": f"{atr_percent:.2f}"})
        return False
    return True

def check_trend_strength_filter(df: pd.DataFrame) -> bool:
    last = df.iloc[-1]
    if last['adx'] < 18:
        log_rejection(df.name, "Trend Strength Filter Failed", {"adx": f"{last['adx']:.2f}"})
        return False
    return True

def is_htf_bullish_confirmation(symbol: str, htf: str = '1h') -> bool:
    try:
        df = fetch_historical_data(symbol, htf, days=40) 
        if df is None or len(df) < 200: return False
        df['ema50']  = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        last = df.iloc[-1]
        return last['close'] > last['ema50'] and last['ema50'] > last['ema200']
    except Exception: return False

# --- استراتيجيات التداول ---
def check_bb_stoch_strategy(df: pd.DataFrame) -> bool:
    last, prev = df.iloc[-1], df.iloc[-2]
    return (prev['low'] <= prev['bb_lower'] * 1.001 and last['close'] > last['open'] and
            last['stoch_rsi_k'] < 35 and last['stoch_rsi_k'] > prev['stoch_rsi_k'] and
            last['rsi'] > 25)

def check_macd_ema_strategy(df: pd.DataFrame) -> bool:
    last, prev = df.iloc[-1], df.iloc[-2]
    return (prev['macd'] < prev['macd_signal'] and last['macd'] > last['macd_signal'] and
            last['close'] > last['ema_12'] and last['adx'] > 18)

def check_ema_rsi_strategy(df: pd.DataFrame) -> bool:
    last, prev = df.iloc[-1], df.iloc[-2]
    return (prev['ema_9'] < prev['ema_12'] and last['ema_9'] > last['ema_12'] and
            last['rsi'] > 52 and last['close'] > last['ema_26'])

def check_pullback_strategy(df: pd.DataFrame) -> bool:
    last, prev = df.iloc[-1], df.iloc[-2]
    return (last['close'] > last['ema_12'] and last['ema_12'] > last['ema_26'] and
            prev['macd'] < prev['macd_signal'] and last['macd'] > last['macd_signal'])

# --- أنماط الشموع ---
def is_bullish_reversal_pattern(df: pd.DataFrame) -> bool:
    c2, c3 = df.iloc[-2], df.iloc[-3] # Check last two closed candles
    last = df.iloc[-1] # Current candle
    
    # Hammer on c3
    body = abs(c3['open'] - c3['close'])
    if body > 0:
        lower_wick = c3['close'] - c3['low'] if c3['open'] < c3['close'] else c3['open'] - c3['low']
        upper_wick = c3['high'] - c3['close'] if c3['open'] < c3['close'] else c3['high'] - c3['open']
        if lower_wick > 2 * body and upper_wick < body and last['close'] > c3['close']: return True
        
    # Bullish Engulfing on c3
    if (c2['close'] < c2['open'] and c3['close'] > c3['open'] and
        c3['close'] > c2['open'] and c3['open'] < c2['close'] and last['close'] > c3['close']): return True
        
    return False

# --- دوال إنشاء الصفقات ---
def create_paper_trade_signal(symbol: str, df: pd.DataFrame, strategy_name: str) -> None:
    try:
        last = df.iloc[-1]
        entry_price = last['close']
        atr = last['atr']
        stop_loss = entry_price - (atr * ATR_TS_MULTIPLIER)
        if entry_price <= stop_loss: return
        
        risk_per_unit = entry_price - stop_loss
        target_price_1 = entry_price + (risk_per_unit * 1.5) 
        target_price_2 = entry_price + (risk_per_unit * 3.0)
        
        quantity = PAPER_TRADE_SIZE_USDT / entry_price

        message = (f"📊 *فتح صفقة ورقية جديدة*\n"
                   f"💱 *العملة:* `{symbol}`\n"
                   f"📈 *الاستراتيجية:* {strategy_name}\n"
                   f"📌 *الدخول:* `{entry_price:.4f}`\n"
                   f"💰 *الكمية:* `{quantity:.4f}`\n"
                   f"🎯 *الهدف 1:* `{target_price_1:.4f}`\n"
                   f"🎯 *الهدف 2:* `{target_price_2:.4f}`\n"
                   f"🛑 *الوقف:* `{stop_loss:.4f}`")
        send_telegram_message(message)
        log_and_notify("info", f"تم فتح صفقة ورقية لـ {symbol}", "PAPER_TRADE_OPEN")
        
        if check_db_connection() and conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO signals (symbol, entry_price, target_price, target_price_2, stop_loss, status, 
                                       strategy_name, is_real_trade, quantity, initial_quantity, signal_details) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
                """, (symbol, float(entry_price), float(target_price_1), float(target_price_2), float(stop_loss), 'open',
                      strategy_name, False, float(quantity), float(quantity), json.dumps({"atr": float(atr)}, cls=NpEncoder)))
                new_id = cur.fetchone()['id']
            conn.commit()
            with signal_cache_lock:
                open_signals_cache[symbol] = {
                    'id': new_id, 'symbol': symbol, 'entry_price': float(entry_price), 
                    'target_price': float(target_price_1), 'target_price_2': float(target_price_2),
                    'stop_loss': float(stop_loss), 'status': 'open', 'strategy_name': strategy_name, 
                    'is_real_trade': False, 'quantity': float(quantity), 'initial_quantity': float(quantity)
                }
    except Exception as e:
        logger.error(f"❌ خطأ في إنشاء الصفقة الورقية لـ {symbol}: {e}")
        if conn: conn.rollback()

# --- قوالب HTML ---
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
        body { background-color: var(--bg-dark); color: var(--text-light); font-family: 'Tajawal', sans-serif; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        header { background-color: var(--bg-surface); padding: 15px 25px; border-radius: 12px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: center; border: 1px solid #2a2a2a; }
        .header-title { font-size: 24px; font-weight: 700; color: var(--primary); }
        .status-indicator { display: flex; align-items: center; gap: 15px; }
        .status-dot { width: 12px; height: 12px; border-radius: 50%; background-color: var(--danger); box-shadow: 0 0 8px var(--danger); }
        .status-dot.active { background-color: var(--success); box-shadow: 0 0 8px var(--success); }
        .btn { background-color: var(--primary-variant); color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; transition: background-color 0.3s, transform 0.2s; font-weight: 700; text-decoration: none; }
        .btn:hover { background-color: var(--primary); transform: translateY(-2px); }
        .btn.stop { background-color: var(--danger); }
        .btn.stop:hover { background-color: #d32f2f; }
        .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 20px; }
        .card { background-color: var(--bg-surface); border-radius: 12px; padding: 20px; border: 1px solid #2a2a2a; display: flex; flex-direction: column; }
        .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #333; }
        .card-title { font-size: 18px; font-weight: 700; color: var(--text-light); }
        .scrollable-content { overflow-y: auto; max-height: 400px; padding-right: 10px; }
        .scrollable-content::-webkit-scrollbar { width: 6px; }
        .scrollable-content::-webkit-scrollbar-track { background: #2a2a2a; }
        .scrollable-content::-webkit-scrollbar-thumb { background: var(--primary); border-radius: 3px; }
        .item { padding: 12px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid var(--primary); background-color: #252525; }
        .item-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
        .item-title { font-weight: 700; }
        .item-time { font-size: 12px; color: var(--text-medium); }
        .item-content { font-size: 13px; color: var(--text-light); line-height: 1.6; }
        .state-item { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px; }
        .state-label { font-weight: bold; color: var(--text-medium); }
        .state-value.Bullish { color: var(--bullish); font-weight: 700; }
        .state-value.Bearish { color: var(--bearish); font-weight: 700; }
        .state-value.Sideways { color: var(--text-medium); }
        .signal-item.paper { border-left-color: var(--secondary); }
        .signal-item.updated { border-left-color: var(--warning); }
        .notification-item.info { border-left-color: var(--primary); }
        .notification-item.warning { border-left-color: var(--warning); }
        .notification-item.error, .notification-item.trading_status { border-left-color: var(--danger); }
        .rejection-item { border-left-color: var(--warning); }
        .progress-bar-container { background-color: #333; border-radius: 10px; height: 10px; overflow: hidden; margin-top: 10px; direction: ltr; }
        .progress-bar { height: 100%; transition: width 0.4s ease-in-out; border-radius: 10px; }
        .footer { text-align: center; margin-top: 30px; padding: 15px; color: var(--text-medium); font-size: 14px; }
        .signal-actions { margin-top: 10px; display: flex; justify-content: flex-end; }
        .btn-close { background-color: var(--danger); color: white; border: none; padding: 5px 12px; font-size: 12px; border-radius: 6px; cursor: pointer; transition: background-color 0.2s; }
        .btn-close:hover { background-color: #c0392b; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-title">بوت التداول V11.0.0</div>
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
                    <div class="state-item"><span class="state-label">النظام العام:</span><span class="state-value {{ market_state.get('overall_regime', 'N/A') }}">{{ market_state.get('overall_regime', 'N/A') }}</span></div>
                    {% for tf, details in market_state.get('trend_details_by_tf', {}).items() %}
                    <div class="state-item"><span class="state-label">اتجاه {{ tf }}:</span><span class="state-value {{ details.get('trend', 'N/A') }}">{{ details.get('trend', 'N/A') }} (RSI: {{ "%.1f"|format(details.get('rsi', 0)) }})</span></div>
                    {% endfor %}
                    <div class="state-item" style="margin-top:10px; font-size: 12px; color: var(--text-medium);"><span>آخر تحديث:</span><span>{{ market_state.get('last_updated', 'N/A') }}</span></div>
                </div>
            </div>
            <div class="card">
                <div class="card-header"><div class="card-title">الإشارات المفتوحة ({{ open_signals|length }})</div></div>
                <div class="scrollable-content">
                {% if open_signals %}{% for symbol, signal in open_signals.items() %}
                <div class="item signal-item {{ 'paper' if not signal.get('is_real_trade') else '' }} {{ signal.get('status', 'open') }}">
                    <div class="item-header">
                        <div class="item-title">{{ symbol }} 
                            <span style="font-size:12px; color: var(--text-medium);">({{ 'ورقية' if not signal.get('is_real_trade') else 'حقيقية' }})</span>
                            {% if signal.get('status') == 'updated' %}<span style="font-size:11px; color: var(--warning); font-weight: bold;">(ربح جزئي)</span>{% endif %}
                        </div>
                        <div class="item-time">{{ signal.get('strategy_name', '') }}</div>
                    </div>
                    <div class="item-content">
                        دخول: {{ "%.4f"|format(signal.get('entry_price', 0)) }} | حالي: {{ "%.4f"|format(signal.get('current_price', 0)) }}<br>
                        هدف: {{ "%.4f"|format(signal.get('target_price', 0)) }} | وقف: {{ "%.4f"|format(signal.get('stop_loss', 0)) }}
                    </div>
                    <div class="progress-bar-container">
                        {% set progress = signal.get('progress', 0) %}{% if progress >= 0 %}<div class="progress-bar" style="width: {{ [progress, 100]|min }}%; background-color: var(--success);"></div>{% else %}<div class="progress-bar" style="width: {{ [progress|abs, 100]|min }}%; background-color: var(--danger); float: right;"></div>{% endif %}
                    </div>
                    <div class="signal-actions">
                        <button class="btn-close" onclick="manualClose({{ signal.id }})">إغلاق</button>
                    </div>
                </div>
                {% endfor %}{% else %}<div style="text-align: center; padding: 20px; color: var(--text-medium);">لا توجد إشارات مفتوحة</div>{% endif %}
                </div>
            </div>
            <div class="card">
                <div class="card-header"><div class="card-title">الإشعارات الأخيرة</div></div>
                <div class="scrollable-content">
                {% for notif in notifications %}<div class="item notification-item {{ notif.get('type', 'info').lower() }}"><div class="item-header"><div class="item-title">{{ notif.get('type', 'INFO') }}</div><div class="item-time">{{ notif.get('timestamp', '')[:16] }}</div></div><div class="item-content">{{ notif.get('message', '') }}</div></div>{% endfor %}
                </div>
            </div>
            <div class="card">
                <div class="card-header"><div class="card-title">سجل الرفض</div></div>
                <div class="scrollable-content">
                {% for rej in rejections %}<div class="item rejection-item"><div class="item-header"><div class="item-title">{{ rej.get('symbol', 'N/A') }}</div><div class="item-time">{{ rej.get('timestamp', '')[:16] }}</div></div><div class="item-content">{{ rej.get('reason', 'N/A') }}</div></div>{% endfor %}
                </div>
            </div>
        </div>
        <div class="footer"><div>بوت التداول الإلكتروني V11.0.0</div></div>
    </div>
    <script>
        function showAlert(message, type = 'info') {
            const alertBox = document.createElement('div');
            const bgColor = type === 'success' ? 'var(--success)' : type === 'error' ? 'var(--danger)' : 'var(--primary)';
            Object.assign(alertBox.style, { position: 'fixed', bottom: '20px', left: '20px', padding: '15px 25px', borderRadius: '8px', color: 'white', zIndex: '1000', backgroundColor: bgColor, boxShadow: '0 4px 15px rgba(0,0,0,0.2)', transform: 'translateY(100px)', opacity: '0', transition: 'transform 0.4s ease, opacity 0.4s ease' });
            alertBox.innerText = message;
            document.body.appendChild(alertBox);
            setTimeout(() => { alertBox.style.transform = 'translateY(0)'; alertBox.style.opacity = '1'; }, 10);
            setTimeout(() => { alertBox.style.transform = 'translateY(100px)'; alertBox.style.opacity = '0'; setTimeout(() => alertBox.remove(), 400); }, 4000);
        }
        function toggleTrading() {
            fetch('/toggle_trading', { method: 'POST' }).then(res => res.json()).then(data => {
                showAlert(data.message, data.success ? 'success' : 'error');
                if(data.success) setTimeout(() => location.reload(), 1500);
            });
        }
        function manualClose(signalId) {
            if (!confirm('هل أنت متأكد من رغبتك في إغلاق هذه الصفقة يدويًا؟')) return;
            fetch('/close_signal/' + signalId, { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    showAlert(data.message, data.success ? 'success' : 'error');
                    if(data.success) setTimeout(() => location.reload(), 1500);
                });
        }
        setInterval(() => location.reload(), 60000);
    </script>
</body>
</html>
"""

SETTINGS_TEMPLATE = """
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>إعدادات البوت</title>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #121212; --bg-surface: #1e1e1e; --primary: #BB86FC;
            --primary-variant: #3700B3; --secondary: #03DAC6; --text-light: #e0e0e0;
            --text-medium: #a0a0a0; --danger: #F44336;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background-color: var(--bg-dark); color: var(--text-light); font-family: 'Tajawal', sans-serif; }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; }
        header { background-color: var(--bg-surface); padding: 15px 25px; border-radius: 12px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: center; border: 1px solid #2a2a2a; }
        .header-title { font-size: 24px; font-weight: 700; color: var(--primary); }
        .btn { background-color: var(--primary-variant); color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; transition: all 0.3s; font-weight: 700; text-decoration: none; }
        .btn:hover { background-color: var(--primary); transform: translateY(-2px); }
        .settings-form { background-color: var(--bg-surface); border-radius: 12px; padding: 25px; margin-bottom: 20px; border: 1px solid #2a2a2a; }
        .form-section-title { font-size: 20px; font-weight: 700; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid #333; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; margin-bottom: 8px; font-weight: bold; color: var(--text-medium); }
        .form-group input[type="number"] { width: 100%; padding: 12px; border: 1px solid #333; border-radius: 8px; background-color: #252525; color: var(--text-light); font-size: 16px; }
        .form-actions { display: flex; justify-content: space-between; align-items: center; margin-top: 25px; gap: 15px; }
        .btn-secondary { background-color: #333; }
        .btn-secondary:hover { background-color: #444; }
        .checkbox-group { display: flex; align-items: center; gap: 10px; padding: 10px; border-radius: 8px; transition: background-color 0.2s; }
        .checkbox-group:hover { background-color: #252525; }
        .checkbox-group input { width: 18px; height: 18px; accent-color: var(--primary); }
        .checkbox-group label { margin-bottom: 0; cursor: pointer; }
        .toggle-section { text-align: center; padding: 20px; background-color: var(--bg-surface); border-radius: 12px; margin-bottom: 20px; border: 1px solid #2a2a2a; }
        .toggle-switch { display: inline-flex; align-items: center; gap: 10px; background-color: #252525; padding: 5px; border-radius: 20px; cursor: pointer; }
        .toggle-switch .label { padding: 5px 15px; border-radius: 15px; font-weight: bold; transition: all 0.3s; }
        #paper-trading-toggle:checked ~ .labels .paper { background-color: var(--primary); color: white; }
        #paper-trading-toggle:not(:checked) ~ .labels .real { background-color: var(--secondary); color: var(--bg-dark); }
        #paper-trading-toggle { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-title">إعدادات البوت</div>
            <a href="/" class="btn">العودة للرئيسية</a>
        </header>
        <div class="toggle-section">
            <h3 class="form-section-title">وضع التداول</h3>
            <label class="toggle-switch">
                <input type="checkbox" id="paper-trading-toggle" {{ 'checked' if paper_trading_mode else '' }}>
                <div class="labels">
                    <span class="label real">حقيقي</span>
                    <span class="label paper">ورقي</span>
                </div>
            </label>
        </div>
        <div class="settings-form">
            <h3 class="form-section-title">إعدادات التداول</h3>
            <form id="settings-form">
                <div class="form-group"><label for="risk-per-trade">نسبة المخاطرة للصفقة (%)</label><input type="number" id="risk-per-trade" name="risk_per_trade" step="0.1" value="{{ RISK_PER_TRADE_PERCENT }}"></div>
                <div class="form-group"><label for="buy-confidence">حد الثقة للشراء</label><input type="number" id="buy-confidence" name="buy_confidence" step="0.01" value="{{ BUY_CONFIDENCE_THRESHOLD }}"></div>
                <div class="form-group"><label for="max-trades">الحد الأقصى للصفقات المفتوحة</label><input type="number" id="max-trades" name="max_trades" value="{{ MAX_OPEN_TRADES }}"></div>
                <div class="form-group"><label for="min-profit">الحد الأدنى للربح (%)</label><input type="number" id="min-profit" name="min_profit" step="0.1" value="{{ MIN_PROFIT_PERCENT }}"></div>
                <div class="form-actions"><button type="button" class="btn btn-secondary" onclick="resetSettings()">إعادة الافتراضي</button><button type="submit" class="btn">حفظ الإعدادات</button></div>
            </form>
        </div>
        <div class="settings-form">
            <h3 class="form-section-title">تفعيل الاستراتيجيات</h3>
            <form id="strategies-form">
                <div class="form-group checkbox-group"><input type="checkbox" id="use_bb_stoch" name="use_bb_stoch" {{ 'checked' if USE_BB_STOCH_STRATEGY else '' }}><label for="use_bb_stoch">استراتيجية BB+Stoch</label></div>
                <div class="form-group checkbox-group"><input type="checkbox" id="use_macd_ema" name="use_macd_ema" {{ 'checked' if USE_MACD_EMA_STRATEGY else '' }}><label for="use_macd_ema">استراتيجية MACD+EMA</label></div>
                <div class="form-group checkbox-group"><input type="checkbox" id="use_ema_rsi" name="use_ema_rsi" {{ 'checked' if USE_EMA_RSI_STRATEGY else '' }}><label for="use_ema_rsi">استراتيجية EMA+RSI</label></div>
                <div class="form-group checkbox-group"><input type="checkbox" id="use_pullback" name="use_pullback" {{ 'checked' if USE_PULLBACK_STRATEGY else '' }}><label for="use_pullback">استراتيجية Pullback</label></div>
                <div class="form-actions"><button type="submit" class="btn">حفظ الاستراتيجيات</button></div>
            </form>
        </div>
    </div>
    <script>
        function showAlert(message, type = 'info') {
            const alertBox = document.createElement('div');
            const bgColor = type === 'success' ? 'var(--success)' : type === 'error' ? 'var(--danger)' : 'var(--primary)';
            Object.assign(alertBox.style, { position: 'fixed', bottom: '20px', left: '20px', padding: '15px 25px', borderRadius: '8px', color: 'white', zIndex: '1000', backgroundColor: bgColor, boxShadow: '0 4px 15px rgba(0,0,0,0.2)', transform: 'translateY(100px)', opacity: '0', transition: 'transform 0.4s ease, opacity 0.4s ease' });
            alertBox.innerText = message; document.body.appendChild(alertBox);
            setTimeout(() => { alertBox.style.transform = 'translateY(0)'; alertBox.style.opacity = '1'; }, 10);
            setTimeout(() => { alertBox.style.transform = 'translateY(100px)'; alertBox.style.opacity = '0'; setTimeout(() => alertBox.remove(), 400); }, 4000);
        }
        document.getElementById('paper-trading-toggle').addEventListener('change', function() {
            fetch('/toggle_paper_trading', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ paper_trading_mode: this.checked }) }).then(res => res.json()).then(data => showAlert(data.message, data.success ? 'success' : 'error'));
        });
        document.getElementById('settings-form').addEventListener('submit', function(e) {
            e.preventDefault();
            const data = { risk_per_trade: this.risk_per_trade.value, buy_confidence: this.buy_confidence.value, max_trades: this.max_trades.value, min_profit: this.min_profit.value };
            fetch('/update_settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }).then(res => res.json()).then(data => showAlert(data.message, data.success ? 'success' : 'error'));
        });
        document.getElementById('strategies-form').addEventListener('submit', function(e) {
            e.preventDefault();
            const data = { use_bb_stoch: this.use_bb_stoch.checked, use_macd_ema: this.use_macd_ema.checked, use_ema_rsi: this.use_ema_rsi.checked, use_pullback: this.use_pullback.checked };
            fetch('/update_strategies', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }).then(res => res.json()).then(data => showAlert(data.message, data.success ? 'success' : 'error'));
        });
        function resetSettings() {
            const modalHTML = `<div id="confirm-modal" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); display: flex; align-items: center; justify-content: center; z-index: 2000;"><div style="background: var(--bg-surface); padding: 25px; border-radius: 12px; text-align: center; border: 1px solid #333;"><p style="margin-bottom: 20px;">هل أنت متأكد من إعادة تعيين كافة الإعدادات؟</p><button id="confirm-yes" class="btn">نعم</button><button id="confirm-no" class="btn btn-secondary" style="margin-right: 10px;">لا</button></div></div>`;
            document.body.insertAdjacentHTML('beforeend', modalHTML);
            document.getElementById('confirm-yes').onclick = () => {
                fetch('/reset_settings', { method: 'POST' }).then(res => res.json()).then(data => { showAlert(data.message, data.success ? 'success' : 'error'); if(data.success) setTimeout(() => location.reload(), 1500); });
                document.getElementById('confirm-modal').remove();
            };
            document.getElementById('confirm-no').onclick = () => { document.getElementById('confirm-modal').remove(); };
        }
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

@app.route('/settings')
def settings():
    with risk_per_trade_lock: risk_val = RISK_PER_TRADE_PERCENT
    with buy_confidence_lock: buy_conf = BUY_CONFIDENCE_THRESHOLD
    with bb_stoch_strategy_lock: use_bb = USE_BB_STOCH_STRATEGY
    with macd_ema_strategy_lock: use_macd = USE_MACD_EMA_STRATEGY
    with ema_rsi_strategy_lock: use_ema = USE_EMA_RSI_STRATEGY
    with pullback_strategy_lock: use_pullback = USE_PULLBACK_STRATEGY
    return render_template_string(SETTINGS_TEMPLATE,
                                paper_trading_mode=paper_trading_mode,
                                RISK_PER_TRADE_PERCENT=risk_val,
                                BUY_CONFIDENCE_THRESHOLD=buy_conf,
                                MAX_OPEN_TRADES=MAX_OPEN_TRADES,
                                MIN_PROFIT_PERCENT=MIN_PROFIT_PERCENT,
                                USE_BB_STOCH_STRATEGY=use_bb,
                                USE_MACD_EMA_STRATEGY=use_macd,
                                USE_EMA_RSI_STRATEGY=use_ema,
                                USE_PULLBACK_STRATEGY=use_pullback)

@app.route('/toggle_trading', methods=['POST'])
def toggle_trading():
    global is_trading_enabled
    with trading_status_lock:
        is_trading_enabled = not is_trading_enabled
        status = "مفعل" if is_trading_enabled else "معطل"
        log_and_notify("info", f"تم {status} التداول", "TRADING_STATUS")
        send_telegram_message(f"⚙️ تم {status} التداول")
        return jsonify({"success": True, "message": f"تم {status} التداول"})

@app.route('/close_signal/<int:signal_id>', methods=['POST'])
def manual_close_signal_route(signal_id):
    with signal_cache_lock:
        signal_to_close = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)
    
    if not signal_to_close:
        return jsonify({"success": False, "message": "لم يتم العثور على الصفقة"}), 404

    symbol = signal_to_close['symbol']
    with live_prices_lock:
        current_price = live_prices.get(symbol)

    if not current_price:
        return jsonify({"success": False, "message": "لا يوجد سعر حالي متاح للإغلاق"}), 500

    close_signal(signal_to_close, current_price, "MANUAL_CLOSE")
    return jsonify({"success": True, "message": f"تم إرسال أمر إغلاق لصفقة {symbol}"})


@app.route('/toggle_paper_trading', methods=['POST'])
def toggle_paper_trading():
    global paper_trading_mode
    try:
        data = request.json
        paper_trading_mode = data.get('paper_trading_mode', True)
        mode = "ورقي" if paper_trading_mode else "حقيقي"
        log_and_notify("info", f"تم تغيير وضع التداول إلى: {mode}", "TRADING_MODE")
        return jsonify({"success": True, "message": f"تم تغيير وضع التداول إلى: {mode}"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/update_settings', methods=['POST'])
def update_settings():
    global RISK_PER_TRADE_PERCENT, BUY_CONFIDENCE_THRESHOLD, MAX_OPEN_TRADES, MIN_PROFIT_PERCENT
    try:
        data = request.json
        with risk_per_trade_lock: RISK_PER_TRADE_PERCENT = float(data['risk_per_trade'])
        with buy_confidence_lock: BUY_CONFIDENCE_THRESHOLD = float(data['buy_confidence'])
        MAX_OPEN_TRADES = int(data['max_trades'])
        MIN_PROFIT_PERCENT = float(data['min_profit'])
        if redis_client:
            redis_client.set('trading_settings', json.dumps({
                'RISK_PER_TRADE_PERCENT': RISK_PER_TRADE_PERCENT, 'BUY_CONFIDENCE_THRESHOLD': BUY_CONFIDENCE_THRESHOLD,
                'MAX_OPEN_TRADES': MAX_OPEN_TRADES, 'MIN_PROFIT_PERCENT': MIN_PROFIT_PERCENT
            }))
        log_and_notify("info", "تم تحديث إعدادات التداول", "SETTINGS_UPDATE")
        return jsonify({"success": True, "message": "تم تحديث الإعدادات بنجاح"})
    except Exception as e:
        return jsonify({"success": False, "message": "خطأ في تحديث الإعدادات"}), 500

@app.route('/update_strategies', methods=['POST'])
def update_strategies():
    global USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY
    try:
        data = request.json
        with bb_stoch_strategy_lock: USE_BB_STOCH_STRATEGY = data['use_bb_stoch']
        with macd_ema_strategy_lock: USE_MACD_EMA_STRATEGY = data['use_macd_ema']
        with ema_rsi_strategy_lock: USE_EMA_RSI_STRATEGY = data['use_ema_rsi']
        with pullback_strategy_lock: USE_PULLBACK_STRATEGY = data['use_pullback']
        if redis_client:
            redis_client.set('strategy_settings', json.dumps({
                'USE_BB_STOCH_STRATEGY': USE_BB_STOCH_STRATEGY, 'USE_MACD_EMA_STRATEGY': USE_MACD_EMA_STRATEGY,
                'USE_EMA_RSI_STRATEGY': USE_EMA_RSI_STRATEGY, 'USE_PULLBACK_STRATEGY': USE_PULLBACK_STRATEGY
            }))
        log_and_notify("info", "تم تحديث تفعيل الاستراتيجيات", "STRATEGY_UPDATE")
        return jsonify({"success": True, "message": "تم تحديث الاستراتيجيات بنجاح"})
    except Exception as e:
        return jsonify({"success": False, "message": "خطأ في تحديث الاستراتيجيات"}), 500

@app.route('/reset_settings', methods=['POST'])
def reset_settings():
    global RISK_PER_TRADE_PERCENT, BUY_CONFIDENCE_THRESHOLD, MAX_OPEN_TRADES, MIN_PROFIT_PERCENT
    global USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY
    try:
        with risk_per_trade_lock: RISK_PER_TRADE_PERCENT = 0.85
        with buy_confidence_lock: BUY_CONFIDENCE_THRESHOLD = 0.53
        MAX_OPEN_TRADES = 3
        MIN_PROFIT_PERCENT = 0.8
        with bb_stoch_strategy_lock: USE_BB_STOCH_STRATEGY = True
        with macd_ema_strategy_lock: USE_MACD_EMA_STRATEGY = True
        with ema_rsi_strategy_lock: USE_EMA_RSI_STRATEGY = True
        with pullback_strategy_lock: USE_PULLBACK_STRATEGY = True
        if redis_client:
            redis_client.delete('trading_settings', 'strategy_settings')
        log_and_notify("info", "تمت إعادة تعيين الإعدادات إلى القيم الافتراضية", "SETTINGS_RESET")
        return jsonify({"success": True, "message": "تمت إعادة تعيين الإعدادات"})
    except Exception as e:
        return jsonify({"success": False, "message": "خطأ في إعادة تعيين الإعدادات"}), 500

# --- حلقات العمل الخلفية ---
def main_bot_loop():
    logger.info("🚀 [الحلقة الرئيسية] بدء حلقة البحث عن الإشارات...")
    while True:
        try:
            with trading_status_lock:
                if not is_trading_enabled:
                    time.sleep(10); continue
            
            with signal_cache_lock:
                if len(open_signals_cache) >= MAX_OPEN_TRADES:
                    logger.info(f"وصل الحد الأقصى للصفقات ({MAX_OPEN_TRADES}). إيقاف البحث مؤقتًا.")
                    time.sleep(60 * 2)
                    continue

            logger.info("="*20 + " بدء دورة فحص جديدة " + "="*20)
            for i in range(0, len(validated_symbols_to_scan), SYMBOL_PROCESSING_BATCH_SIZE):
                batch = validated_symbols_to_scan[i:i + SYMBOL_PROCESSING_BATCH_SIZE]
                for symbol in batch:
                    with signal_cache_lock:
                        if symbol in open_signals_cache: continue
                    df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df is None or df.empty or len(df) < 50:
                        log_rejection(symbol, "Insufficient Historical Data"); continue
                    df_featured = calculate_all_features(df); df_featured.name = symbol
                    if not check_market_volatility_filter(df_featured): continue
                    if not check_trend_strength_filter(df_featured): continue
                    if not is_htf_bullish_confirmation(symbol, HIGHER_TIMEFRAME):
                        log_rejection(symbol, "HTF Trend Confirmation Failed"); continue
                    
                    strategy_found = None
                    if USE_BB_STOCH_STRATEGY and check_bb_stoch_strategy(df_featured): strategy_found = "BB+Stoch"
                    elif USE_MACD_EMA_STRATEGY and check_macd_ema_strategy(df_featured): strategy_found = "MACD+EMA"
                    elif USE_EMA_RSI_STRATEGY and check_ema_rsi_strategy(df_featured): strategy_found = "EMA+RSI"
                    elif USE_PULLBACK_STRATEGY and check_pullback_strategy(df_featured): strategy_found = "Pullback"

                    if strategy_found and is_bullish_reversal_pattern(df_featured):
                        logger.info(f"🌟 [{symbol}] إشارة مؤكدة! الاستراتيجية: {strategy_found}")
                        create_paper_trade_signal(symbol, df_featured, strategy_found)
            logger.info("="*20 + " اكتملت دورة الفحص " + "="*20)
            time.sleep(60 * 5)
        except Exception as e:
            logger.error(f"❌ [الحلقة الرئيسية] حدث خطأ فادح: {e}", exc_info=True)
            time.sleep(60)

def close_signal(signal: Dict, closing_price: float, reason: str):
    symbol = signal['symbol']
    entry_price = signal['entry_price']
    initial_quantity = signal.get('initial_quantity', signal.get('quantity', 0))
    
    # حساب الربح بناءً على سعر الدخول الأصلي والكمية الأصلية
    profit = ((closing_price - entry_price) / entry_price) * 100
    
    if check_db_connection() and conn:
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE signals SET status = 'closed', closing_price = %s, closed_at = %s, profit_percentage = %s, closing_reason = %s WHERE id = %s",
                            (closing_price, datetime.now(timezone.utc), profit, reason, signal['id']))
            conn.commit()
            log_and_notify("info", f"تم إغلاق صفقة {symbol} بسبب: {reason}. الربح: {profit:.2f}%", "TRADE_CLOSED")
            send_telegram_message(f"✅ *إغلاق صفقة {symbol}*\n*السبب:* {reason}\n*الربح:* `{profit:.2f}%`")
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
            with signal_cache_lock: open_signals_copy = list(open_signals_cache.values())
            if not open_signals_copy:
                time.sleep(5)
                continue
            
            with live_prices_lock: current_prices = live_prices.copy()
            
            for signal in open_signals_copy:
                symbol = signal.get('symbol')
                current_price = current_prices.get(symbol)
                if not symbol or not current_price: continue

                entry, target1, stop = signal.get('entry_price', 0), signal.get('target_price', 0), signal.get('stop_loss', 0)
                
                # تحديث السعر الحالي والتقدم في الكاش
                progress = 0
                if current_price >= entry and target1 > entry: progress = ((current_price - entry) / (target1 - entry)) * 100
                elif current_price < entry and entry > stop: progress = ((current_price - entry) / (entry - stop)) * 100
                with signal_cache_lock:
                    if symbol in open_signals_cache:
                        open_signals_cache[symbol]['current_price'] = current_price
                        open_signals_cache[symbol]['progress'] = progress

                # 1. التحقق من ضرب وقف الخسارة (له الأولوية القصوى)
                if current_price <= stop:
                    close_signal(signal, stop, "SL_HIT")
                    continue

                # 2. آلية أخذ الربح الجزئي عند الهدف الأول
                if USE_PARTIAL_TAKE_PROFIT and signal['status'] == 'open' and current_price >= target1:
                    df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, 5)
                    if df is not None and not df.empty:
                        df = calculate_all_features(df)
                        last_rsi = df['rsi'].iloc[-1]
                        
                        if last_rsi >= PARTIAL_TP_RSI_THRESHOLD: # قوة شرائية، استمر للهدف الثاني
                            new_quantity = signal['quantity'] / 2
                            target2 = signal['target_price_2']
                            
                            if check_db_connection() and conn:
                                with conn.cursor() as cur:
                                    cur.execute("UPDATE signals SET status = 'updated', target_price = %s, quantity = %s WHERE id = %s",
                                                (target2, new_quantity, signal['id']))
                                conn.commit()
                            with signal_cache_lock:
                                if symbol in open_signals_cache:
                                    open_signals_cache[symbol]['status'] = 'updated'
                                    open_signals_cache[symbol]['target_price'] = target2
                                    open_signals_cache[symbol]['quantity'] = new_quantity
                            
                            msg = f"📈 *أخذ ربح جزئي لـ {symbol}*\nتم بيع نصف الكمية عند `{target1:.4f}`.\nالهدف الجديد: `{target2:.4f}`"
                            log_and_notify("info", msg, "PARTIAL_TP")
                            send_telegram_message(msg)
                            continue # انتقل للصفقة التالية
                        else: # لا توجد قوة، أغلق الصفقة كاملة
                            close_signal(signal, target1, "TP1_HIT_NO_MOMENTUM")
                            continue
                    else: # فشل جلب البيانات، أغلق كإجراء وقائي
                        close_signal(signal, target1, "TP1_HIT_DATA_FAIL")
                        continue

                # 3. التحقق من ضرب الهدف النهائي (بعد أخذ الربح الجزئي)
                if signal['status'] == 'updated' and current_price >= signal['target_price']:
                    close_signal(signal, signal['target_price'], "TP2_HIT")
                    continue

                # 4. آلية وقف الخسارة المتحرك
                if USE_TRAILING_STOP_LOSS and entry > 0:
                    current_profit_percent = ((current_price - entry) / entry) * 100
                    if current_profit_percent > TRAILING_STOP_TRIGGER_PERCENT:
                        new_stop_loss = current_price * (1 - (TRAILING_STOP_DISTANCE_PERCENT / 100))
                        if new_stop_loss > stop:
                            if check_db_connection() and conn:
                                with conn.cursor() as cur:
                                    cur.execute("UPDATE signals SET stop_loss = %s WHERE id = %s", (new_stop_loss, signal['id']))
                                conn.commit()
                            with signal_cache_lock:
                                if symbol in open_signals_cache:
                                    open_signals_cache[symbol]['stop_loss'] = new_stop_loss
                            log_and_notify("info", f"تم تحديث وقف الخسارة لـ {symbol} إلى {new_stop_loss:.4f}", "TSL_UPDATE")

            time.sleep(1)
        except Exception as e:
            logger.error(f"❌ [إدارة الصفقات] حدث خطأ: {e}", exc_info=True)
            time.sleep(10)

def update_market_state_loop():
    logger.info("🚀 [حالة السوق] بدء حلقة تحديث حالة السوق...")
    while True:
        try:
            trend_details, bullish_count = {}, 0
            for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
                days = 10 if tf == '15m' else 30 if tf == '1h' else 90
                btc_df = fetch_historical_data(BTC_SYMBOL, tf, days)
                if btc_df is None or len(btc_df) < 50:
                    trend_details[tf] = {"trend": "Unknown", "rsi": 50}; continue
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
                current_market_state.update({'overall_regime': overall_regime, 'trend_details_by_tf': trend_details, 'last_updated': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')})
            time.sleep(60 * 5)
        except Exception as e:
            logger.error(f"❌ [حالة السوق] حدث خطأ: {e}", exc_info=True)
            time.sleep(60)

# --- نقطة بداية البرنامج ---
if __name__ == '__main__':
    logger.info("="*50 + "\n====== بدء تشغيل بوت التداول الإلكتروني V11.0.0 ======\n" + "="*50)
    init_db()
    init_redis()
    try:
        client = Client(API_KEY, API_SECRET); client.ping()
        logger.info("✅ [Binance] الاتصال بالمنصة ناجح.")
    except Exception as e:
        logger.critical(f"❌ [Binance] فشل الاتصال بالمنصة: {e}"); exit(1)
    get_exchange_info_map()
    validated_symbols_to_scan = get_validated_symbols()
    if not validated_symbols_to_scan:
        logger.critical("❌ لا توجد عملات صالحة للمسح. سيتم إيقاف البوت."); exit(1)
    load_open_signals_to_cache()
    load_notifications_to_cache()
    load_settings_from_redis()
    start_websocket()
    Thread(target=main_bot_loop, daemon=True).start()
    Thread(target=manage_open_trades_loop, daemon=True).start()
    Thread(target=update_market_state_loop, daemon=True).start()
    logger.info("🌐 [Flask] بدء تشغيل واجهة المستخدم على http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
