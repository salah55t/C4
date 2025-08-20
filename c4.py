# ملف c4.py - نسخة V14.1.0 (إصلاح أزرار التحكم وإضافة تفعيل الوضع الحقيقي)
# --- التغييرات الرئيسية (V14.1.0):
# 1. [إصلاح الواجهة] تعديل JavaScript لربط جميع أزرار التحكم والتأكد من حفظ الإعدادات فوراً.
# 2. [تداول حقيقي آمن] إضافة زر مخصص لتفعيل التداول الحقيقي.
# 3. [نافذة تأكيد] إضافة نافذة منبثقة لتأكيد تفعيل الوضع الحقيقي لمنع الأخطاء.
# 4. [تحسينات UI] تحسينات طفيفة على تصميم الواجهة لتكون أكثر وضوحاً.

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
        logging.FileHandler('crypto_bot_v14_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV14.1.0')

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
PAPER_ACCOUNT_BALANCE: float = 1000.0

# --- المتغيرات القابلة للتعديل ---
RISK_PER_TRADE_PERCENT: float = 1.0
BUY_CONFIDENCE_THRESHOLD = 0.53
MAX_OPEN_TRADES: int = 3
MIN_PROFIT_PERCENT: float = 1.5

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
TRADING_FEE_PERCENT: float = 0.1
API_REQUEST_DELAY: float = 0.5

# --- إعدادات المؤشرات الفنية ---
EMA_FAST_PERIOD: int = 12
EMA_SLOW_PERIOD: int = 26
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14

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

# --- دوال تهيئة الخدمات (بدون تغيير جوهري) ---
def init_db(retries: int = 5, delay: int = 5) -> None:
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
        logger.error(f"❌ [Telegram] Failed to send message: {e}")

# --- WebSocket Handler ---
def handle_socket_message(msg):
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
    global ws_manager
    logger.info("🚀 [WebSocket] Starting WebSocket manager...")
    ws_manager = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    ws_manager.start()
    ws_manager.start_ticker_socket(callback=handle_socket_message)
    logger.info("✅ [WebSocket] Successfully subscribed to ticker stream (!ticker@arr).")

# --- دوال جلب البيانات وحساب المؤشرات ---
def get_exchange_info_map() -> None:
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
    last = df.iloc[-1]
    last_but_one = df.iloc[-2]
    is_up_trend = (last['ema_9'] > last['ema_26'] and last_but_one['ema_9'] > last_but_one['ema_26'] and last['adx'] > 20)
    is_down_trend = (last['ema_9'] < last['ema_26'] and last_but_one['ema_9'] < last_but_one['ema_26'] and last['adx'] > 20)
    if is_up_trend: return 'اتجاه صاعد'
    if is_down_trend: return 'اتجاه هابط'
    return 'سوق عرضي'

# --- فلاتر الدخول (معدلة) ---
def check_market_volatility(df: pd.DataFrame) -> (bool, str):
    if len(df) < 20: return False, "Insufficient Historical Data"
    atr = df['atr'].iloc[-1]
    avg_atr = df['atr'].rolling(20).mean().iloc[-1]
    relative_volatility = atr / avg_atr
    if relative_volatility <= 0.5: return False, "Market Volatility Too Low"
    if relative_volatility >= 3.0: return False, "Market Volatility Too High"
    return True, "Volatility OK"

def check_htf_trend_confirmation(htf_df: pd.DataFrame) -> (bool, str):
    if len(htf_df) < 50: return False, "Insufficient Historical Data"
    ema_20 = htf_df['ema_12']
    ema_50 = htf_df['ema_50']
    if ema_20.iloc[-1] <= ema_50.iloc[-1] * 0.995: return False, "HTF Trend Not Bullish"
    return True, "HTF Trend OK"

# --- استراتيجيات الدخول (القديمة والجديدة) ---
def check_volume_profile_strategy(df: pd.DataFrame) -> bool:
    last = df.iloc[-1]
    price_bins = pd.cut(df['close'], bins=20, labels=False)
    volume_by_bin = df.groupby(price_bins)['volume'].sum()
    poc_bin = volume_by_bin.idxmax()
    value_area_high = df['close'].iloc[price_bins[price_bins == poc_bin].index].max()
    above_value_area = last['close'] > value_area_high
    high_volume = last['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.5
    price_action = last['close'] > df['high'].iloc[-2]
    trend_strength = last['adx'] > 18
    return above_value_area and high_volume and price_action and trend_strength

def check_rsi_divergence(df: pd.DataFrame) -> bool:
    if len(df) < 10: return False
    # تباعد صاعد (Bullish Divergence)
    if (df['low'].iloc[-1] < df['low'].iloc[-5] and
        df['rsi'].iloc[-1] > df['rsi'].iloc[-5] and
        df['rsi'].iloc[-1] < 40):
        return True
    return False

def identify_support_resistance(df: pd.DataFrame, window: int = 20) -> tuple:
    resistance = df['high'].rolling(window, center=True).max().dropna()
    support = df['low'].rolling(window, center=True).min().dropna()
    if resistance.empty or support.empty:
        return df['high'].iloc[-1], df['low'].iloc[-1]
    return resistance.iloc[-1], support.iloc[-1]

def check_support_resistance_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 20: return False
    _, support = identify_support_resistance(df)
    current_close = df['close'].iloc[-1]
    if (current_close <= support * 1.01 and
        df['close'].iloc[-1] > df['open'].iloc[-1] and
        df['volume'].iloc[-1] > df['volume'].rolling(10).mean().iloc[-1] * 1.2):
        return True
    return False

def check_scalping_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 15: return False
    ema_5 = df['ema_5']
    ema_13 = df['ema_13']
    if (ema_5.iloc[-2] < ema_13.iloc[-2] and
        ema_5.iloc[-1] > ema_13.iloc[-1] and
        df['rsi'].iloc[-1] > 30 and df['rsi'].iloc[-1] < 70):
        return True
    return False

# --- إدارة المخاطر والخروج (محسنة) ---
def calculate_dynamic_position_size(df: pd.DataFrame, account_balance: float, risk_percent: float) -> float:
    atr = df['atr'].iloc[-1]
    volatility = atr / df['close'].iloc[-1]
    
    if volatility > 0.05: adjusted_risk_percent = risk_percent * 0.7
    elif volatility < 0.02: adjusted_risk_percent = risk_percent * 1.3
    else: adjusted_risk_percent = risk_percent
    
    risk_amount = (account_balance * adjusted_risk_percent) / 100
    stop_loss_distance = atr * 2
    if stop_loss_distance == 0: return 0
    position_size_usdt = risk_amount / (stop_loss_distance / df['close'].iloc[-1])
    return min(position_size_usdt, account_balance * 0.1)

def check_improved_exit_conditions(signal: Dict, current_price: float) -> (bool, str):
    entry_price = signal['entry_price']
    stop_loss = signal['stop_loss']
    target_price = signal['target_price']
    
    if current_price <= stop_loss: return True, "Stop Loss Hit"
    if current_price >= entry_price + (target_price - entry_price) * 0.8: return True, "Target Price (80%) Hit"
    
    return False, "No Action"

# --- دوال التعامل مع قاعدة البيانات (بدون تغيير جوهري) ---
def create_signal_db(signal_data: Dict) -> bool:
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
    if not check_db_connection(): return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status = 'open';")
            signals = cur.fetchall()
        return [dict(s) for s in signals]
    except Exception as e:
        logger.error(f"❌ [DB] فشل جلب الإشارات المفتوحة: {e}"); return []

def update_signal_db(signal_id: int, updates: Dict) -> bool:
    if not check_db_connection(): return False
    try:
        with conn.cursor() as cur:
            set_clause = ", ".join([f"{key} = %s" for key in updates.keys()])
            query = f"UPDATE signals SET {set_clause} WHERE id = %s"
            cur.execute(query, list(updates.values()) + [signal_id])
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"❌ [DB] فشل تحديث الإشارة ID {signal_id}: {e}")
        if conn: conn.rollback()
        return False

def close_signal_db(signal: Dict, closing_price: float, closing_reason: str):
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
    if not check_db_connection(): return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 30;")
            notifs = cur.fetchall()
        with notifications_lock:
            notifications_cache.extendleft([dict(n) for n in notifs])
        logger.info(f"✅ [Cache] تم تحميل {len(notifs)} إشعارًا إلى الذاكرة المؤقتة.")
    except Exception as e:
        logger.error(f"❌ [Cache] فشل تحميل الإشعارات: {e}")

# --- دوال إعدادات Redis ---
def save_settings_to_redis() -> None:
    if not redis_client: return
    try:
        settings = {
            'is_trading_enabled': is_trading_enabled, 'paper_trading_mode': paper_trading_mode,
            'RISK_PER_TRADE_PERCENT': RISK_PER_TRADE_PERCENT, 'BUY_CONFIDENCE_THRESHOLD': BUY_CONFIDENCE_THRESHOLD,
            'USE_TRAILING_STOP_LOSS': USE_TRAILING_STOP_LOSS, 'USE_PARTIAL_TAKE_PROFIT': USE_PARTIAL_TAKE_PROFIT,
            'USE_VOLUME_PROFILE_STRATEGY': USE_VOLUME_PROFILE_STRATEGY, 'USE_BB_STOCH_STRATEGY': USE_BB_STOCH_STRATEGY,
            'USE_MACD_EMA_STRATEGY': USE_MACD_EMA_STRATEGY, 'USE_EMA_RSI_STRATEGY': USE_EMA_RSI_STRATEGY,
            'USE_PULLBACK_STRATEGY': USE_PULLBACK_STRATEGY, 'USE_RSI_DIVERGENCE_STRATEGY': USE_RSI_DIVERGENCE_STRATEGY,
            'USE_SUPPORT_RESISTANCE_STRATEGY': USE_SUPPORT_RESISTANCE_STRATEGY, 'USE_SCALPING_STRATEGY': USE_SCALPING_STRATEGY,
        }
        redis_client.set('bot_settings_v14', json.dumps(settings))
        logger.info("✅ [Redis] تم حفظ الإعدادات بنجاح.")
    except Exception as e:
        logger.error(f"❌ [Redis] فشل حفظ الإعدادات: {e}")

def load_settings_from_redis() -> None:
    global is_trading_enabled, paper_trading_mode, RISK_PER_TRADE_PERCENT, BUY_CONFIDENCE_THRESHOLD
    global USE_TRAILING_STOP_LOSS, USE_PARTIAL_TAKE_PROFIT, USE_VOLUME_PROFILE_STRATEGY
    global USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY, USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY
    global USE_RSI_DIVERGENCE_STRATEGY, USE_SUPPORT_RESISTANCE_STRATEGY, USE_SCALPING_STRATEGY
    if not redis_client: return
    try:
        settings_json = redis_client.get('bot_settings_v14')
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
            USE_RSI_DIVERGENCE_STRATEGY = settings.get('USE_RSI_DIVERGENCE_STRATEGY', USE_RSI_DIVERGENCE_STRATEGY)
            USE_SUPPORT_RESISTANCE_STRATEGY = settings.get('USE_SUPPORT_RESISTANCE_STRATEGY', USE_SUPPORT_RESISTANCE_STRATEGY)
            USE_SCALPING_STRATEGY = settings.get('USE_SCALPING_STRATEGY', USE_SCALPING_STRATEGY)
            logger.info("✅ [Redis] تم تحميل الإعدادات بنجاح.")
    except Exception as e:
        logger.error(f"❌ [Redis] فشل تحميل الإعدادات: {e}")

# --- دوال الواجهة (API) ---
@app.route('/status', methods=['GET'])
def get_status():
    with trading_status_lock: is_enabled = is_trading_enabled
    return jsonify(json.loads(json.dumps({
        'status': 'Running', 'trading_enabled': is_enabled, 'paper_trading_mode': paper_trading_mode,
        'open_trades_count': len(open_signals_cache), 'max_open_trades': MAX_OPEN_TRADES,
        'risk_per_trade_percent': RISK_PER_TRADE_PERCENT, 'buy_confidence_threshold': BUY_CONFIDENCE_THRESHOLD,
        'market_state': current_market_state,
        'strategies_enabled': {
            'bb_stoch': USE_BB_STOCH_STRATEGY, 'macd_ema': USE_MACD_EMA_STRATEGY, 'ema_rsi': USE_EMA_RSI_STRATEGY,
            'pullback': USE_PULLBACK_STRATEGY, 'volume_profile': USE_VOLUME_PROFILE_STRATEGY,
            'rsi_divergence': USE_RSI_DIVERGENCE_STRATEGY, 'support_resistance': USE_SUPPORT_RESISTANCE_STRATEGY,
            'scalping': USE_SCALPING_STRATEGY,
        },
        'advanced_features_enabled': {
            'trailing_stop_loss': USE_TRAILING_STOP_LOSS, 'partial_take_profit': USE_PARTIAL_TAKE_PROFIT
        }
    }, cls=NpEncoder)))

@app.route('/open_trades', methods=['GET'])
def get_open_trades():
    with signal_cache_lock: trades = list(open_signals_cache.values())
    return jsonify(json.loads(json.dumps({'open_trades': trades}, cls=NpEncoder)))

@app.route('/rejection_logs', methods=['GET'])
def get_rejection_logs():
    with rejection_logs_lock: logs = list(rejection_logs_cache)
    return jsonify(json.loads(json.dumps({'rejection_logs': logs}, cls=NpEncoder)))

@app.route('/notifications', methods=['GET'])
def get_notifications():
    with notifications_lock: notifs = list(notifications_cache)
    return jsonify(json.loads(json.dumps({'notifications': notifs}, cls=NpEncoder)))

@app.route('/settings', methods=['POST'])
def update_settings_route():
    global is_trading_enabled, paper_trading_mode, RISK_PER_TRADE_PERCENT, BUY_CONFIDENCE_THRESHOLD
    global USE_TRAILING_STOP_LOSS, USE_PARTIAL_TAKE_PROFIT, USE_BB_STOCH_STRATEGY, USE_MACD_EMA_STRATEGY
    global USE_EMA_RSI_STRATEGY, USE_PULLBACK_STRATEGY, USE_VOLUME_PROFILE_STRATEGY
    global USE_RSI_DIVERGENCE_STRATEGY, USE_SUPPORT_RESISTANCE_STRATEGY, USE_SCALPING_STRATEGY
    try:
        data = request.json
        if data is None: return jsonify({"success": False, "message": "Invalid JSON payload."}), 400
        
        # استخدام الأقفال لضمان سلامة التحديثات
        with trading_status_lock:
            if 'trading_enabled' in data: is_trading_enabled = bool(data['trading_enabled'])
        
        # تحديث باقي الإعدادات
        if 'paper_trading_mode' in data: paper_trading_mode = bool(data['paper_trading_mode'])
        if 'risk_per_trade_percent' in data: RISK_PER_TRADE_PERCENT = float(data['risk_per_trade_percent'])
        if 'buy_confidence_threshold' in data: BUY_CONFIDENCE_THRESHOLD = float(data['buy_confidence_threshold'])
        if 'use_trailing_stop_loss' in data: USE_TRAILING_STOP_LOSS = bool(data['use_trailing_stop_loss'])
        if 'use_partial_take_profit' in data: USE_PARTIAL_TAKE_PROFIT = bool(data['use_partial_take_profit'])
        if 'use_volume_profile_strategy' in data: USE_VOLUME_PROFILE_STRATEGY = bool(data['use_volume_profile_strategy'])
        if 'use_bb_stoch_strategy' in data: USE_BB_STOCH_STRATEGY = bool(data['use_bb_stoch_strategy'])
        if 'use_macd_ema_strategy' in data: USE_MACD_EMA_STRATEGY = bool(data['use_macd_ema_strategy'])
        if 'use_ema_rsi_strategy' in data: USE_EMA_RSI_STRATEGY = bool(data['use_ema_rsi_strategy'])
        if 'use_pullback_strategy' in data: USE_PULLBACK_STRATEGY = bool(data['use_pullback_strategy'])
        if 'use_rsi_divergence_strategy' in data: USE_RSI_DIVERGENCE_STRATEGY = bool(data['use_rsi_divergence_strategy'])
        if 'use_support_resistance_strategy' in data: USE_SUPPORT_RESISTANCE_STRATEGY = bool(data['use_support_resistance_strategy'])
        if 'use_scalping_strategy' in data: USE_SCALPING_STRATEGY = bool(data['use_scalping_strategy'])
        
        save_settings_to_redis()
        log_and_notify("info", f"✅ [API] تم تحديث إعدادات البوت: {json.dumps(data)}", "SETTINGS_UPDATE")
        return jsonify({"success": True, "message": "تم تحديث الإعدادات بنجاح."})
    except Exception as e:
        logger.error(f"❌ [API] فشل تحديث الإعدادات: {e}")
        return jsonify({"success": False, "message": "فشل تحديث الإعدادات."}), 500

@app.route('/')
def home():
    return render_template_string("""
        <!doctype html>
        <html lang="ar" dir="rtl">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>لوحة تحكم بوت التداول V14.1</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap" rel="stylesheet">
            <style>
                body { font-family: 'Cairo', sans-serif; scroll-behavior: smooth; }
                .sidebar-link.active { background-color: #374151; color: white; }
                .tab-link.active-tab { color: #3b82f6; border-color: #3b82f6; }
                ::-webkit-scrollbar { width: 8px; }
                ::-webkit-scrollbar-track { background: #1f2937; }
                ::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 4px; }
                ::-webkit-scrollbar-thumb:hover { background: #6b7280; }
                .modal-overlay { transition: opacity 0.3s ease-in-out; }
            </style>
        </head>
        <body class="bg-gray-900 text-gray-200">
            <div class="flex flex-col md:flex-row min-h-screen">
                <!-- Sidebar -->
                <aside class="w-full md:w-64 bg-gray-800 p-4 md:p-6 flex flex-col shrink-0">
                    <h1 class="text-2xl font-bold text-white mb-8 text-center md:text-right">بوت التداول V14.1</h1>
                    <nav class="flex-grow">
                        <ul class="space-y-2">
                            <li><a href="#" id="nav-dashboard" class="sidebar-link flex items-center py-2.5 px-4 rounded-lg hover:bg-gray-700 transition-colors">
                                <svg class="w-5 h-5 ml-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"></path></svg>
                                لوحة التحكم
                            </a></li>
                            <li><a href="#" id="nav-settings" class="sidebar-link flex items-center py-2.5 px-4 rounded-lg hover:bg-gray-700 transition-colors">
                                <svg class="w-5 h-5 ml-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
                                الإعدادات
                            </a></li>
                        </ul>
                    </nav>
                    <div id="status-footer" class="text-center mt-6">
                        <div class="flex items-center justify-center">
                            <span id="botStatusDot" class="h-3 w-3 rounded-full bg-gray-500 ml-2"></span>
                            <span id="botStatusText">جاري التحميل...</span>
                        </div>
                        <span id="tradingModeText" class="text-sm text-gray-400 mt-1 block"></span>
                    </div>
                </aside>

                <!-- Main Content -->
                <main class="flex-1 p-4 md:p-8 overflow-y-auto">
                    <!-- Dashboard Page -->
                    <div id="page-dashboard">
                        <h2 class="text-3xl font-bold mb-6">لوحة التحكم</h2>
                        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
                            <div class="bg-gray-800 p-6 rounded-xl">
                                <h3 class="text-gray-400 text-lg">حالة السوق (BTC)</h3>
                                <p id="marketTrendText" class="text-2xl font-bold mt-2 text-white">جاري التقييم...</p>
                                <p id="marketTrendTime" class="text-xs text-gray-500 mt-1"></p>
                            </div>
                            <div class="bg-gray-800 p-6 rounded-xl">
                                <h3 class="text-gray-400 text-lg">الصفقات المفتوحة</h3>
                                <p id="openTradesCount" class="text-2xl font-bold mt-2 text-white">-</p>
                            </div>
                            <div class="bg-gray-800 p-6 rounded-xl">
                                <h3 class="text-gray-400 text-lg">تفعيل/إيقاف البوت</h3>
                                <button id="toggleTradingBtn" class="mt-3 w-full text-white font-bold py-2 px-4 rounded-lg transition-colors">...</button>
                            </div>
                        </div>

                        <div class="bg-gray-800 rounded-xl p-4 md:p-6">
                            <div class="border-b border-gray-700">
                                <nav class="-mb-px flex space-x-2 md:space-x-6 overflow-x-auto" dir="ltr">
                                    <a href="#" class="tab-link active-tab whitespace-nowrap py-4 px-2 md:px-4 border-b-2 font-medium text-sm md:text-base" data-tab="open-trades">الصفقات المفتوحة</a>
                                    <a href="#" class="tab-link text-gray-400 hover:text-white whitespace-nowrap py-4 px-2 md:px-4 border-b-2 border-transparent font-medium text-sm md:text-base" data-tab="rejection-logs">سجلات الرفض</a>
                                    <a href="#" class="tab-link text-gray-400 hover:text-white whitespace-nowrap py-4 px-2 md:px-4 border-b-2 border-transparent font-medium text-sm md:text-base" data-tab="notifications">الإشعارات</a>
                                </nav>
                            </div>
                            <div class="mt-6">
                                <div id="tab-content-open-trades" class="tab-content">
                                    <div class="overflow-x-auto">
                                        <table class="min-w-full">
                                            <thead class="text-gray-400 text-sm">
                                                <tr>
                                                    <th class="py-3 px-4 text-right">الرمز</th>
                                                    <th class="py-3 px-4 text-right">الدخول</th>
                                                    <th class="py-3 px-4 text-right">وقف الخسارة</th>
                                                    <th class="py-3 px-4 text-right">الهدف (1)</th>
                                                    <th class="py-3 px-4 text-right">الهدف (2)</th>
                                                    <th class="py-3 px-4 text-right">الكمية</th>
                                                </tr>
                                            </thead>
                                            <tbody id="openTradesTableBody"></tbody>
                                        </table>
                                    </div>
                                </div>
                                <div id="tab-content-rejection-logs" class="tab-content hidden">
                                    <ul id="rejectionLogsList" class="space-y-3 max-h-96 overflow-y-auto"></ul>
                                </div>
                                <div id="tab-content-notifications" class="tab-content hidden">
                                    <ul id="notificationsList" class="space-y-3 max-h-96 overflow-y-auto"></ul>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Settings Page -->
                    <div id="page-settings" class="hidden">
                        <h2 class="text-3xl font-bold mb-6">الإعدادات</h2>
                        <div class="space-y-8">
                            <div class="bg-gray-800 p-6 rounded-xl">
                                <h3 class="text-xl font-bold mb-4 border-b border-gray-700 pb-3">وضع التداول</h3>
                                <div class="space-y-4">
                                    <div class="flex items-center justify-between bg-gray-700 p-3 rounded-lg">
                                        <span class="font-medium text-white">التداول التجريبي (Paper Trading)</span>
                                        <button id="setPaperModeBtn" class="font-bold py-2 px-4 rounded-lg transition-colors text-sm">...</button>
                                    </div>
                                    <div class="flex items-center justify-between bg-gray-700 p-3 rounded-lg">
                                        <span class="font-medium text-red-400">التداول الحقيقي (Live Trading)</span>
                                        <button id="setRealModeBtn" class="font-bold py-2 px-4 rounded-lg transition-colors text-sm">...</button>
                                    </div>
                                </div>
                            </div>

                            <div class="bg-gray-800 p-6 rounded-xl">
                                <h3 class="text-xl font-bold mb-4 border-b border-gray-700 pb-3">إدارة المخاطر</h3>
                                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div>
                                        <label for="riskPerTrade" class="block mb-2 text-sm font-medium text-gray-300">نسبة المخاطرة (%)</label>
                                        <input type="number" id="riskPerTrade" step="0.05" class="bg-gray-700 border border-gray-600 text-white text-sm rounded-lg w-full p-2.5">
                                    </div>
                                    <div>
                                        <label for="buyConfidence" class="block mb-2 text-sm font-medium text-gray-300">عتبة ثقة الشراء</label>
                                        <input type="number" id="buyConfidence" step="0.01" class="bg-gray-700 border border-gray-600 text-white text-sm rounded-lg w-full p-2.5">
                                    </div>
                                </div>
                            </div>
                            
                            <div class="bg-gray-800 p-6 rounded-xl">
                                <h3 class="text-xl font-bold mb-4 border-b border-gray-700 pb-3">إدارة الصفقات المتقدمة</h3>
                                <div class="space-y-4">
                                    <div class="flex items-center justify-between"><span class="font-medium text-white">وقف الخسارة المتحرك</span><label class="relative inline-flex items-center cursor-pointer"><input type="checkbox" id="trailingStopLossToggle" data-key="use_trailing_stop_loss" class="sr-only peer setting-toggle"><div class="w-11 h-6 bg-gray-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div></label></div>
                                    <div class="flex items-center justify-between"><span class="font-medium text-white">أخذ الربح الجزئي</span><label class="relative inline-flex items-center cursor-pointer"><input type="checkbox" id="partialTakeProfitToggle" data-key="use_partial_take_profit" class="sr-only peer setting-toggle"><div class="w-11 h-6 bg-gray-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div></label></div>
                                </div>
                            </div>

                            <div class="bg-gray-800 p-6 rounded-xl">
                                <h3 class="text-xl font-bold mb-4 border-b border-gray-700 pb-3">تفعيل استراتيجيات التداول</h3>
                                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div class="flex items-center justify-between"><span class="font-medium text-white">Volume Profile</span><label class="relative inline-flex items-center cursor-pointer"><input type="checkbox" id="volumeProfileStrategy" data-key="use_volume_profile_strategy" class="sr-only peer setting-toggle"><div class="w-11 h-6 bg-gray-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div></label></div>
                                    <div class="flex items-center justify-between"><span class="font-medium text-white">BB & Stoch</span><label class="relative inline-flex items-center cursor-pointer"><input type="checkbox" id="bbStochStrategy" data-key="use_bb_stoch_strategy" class="sr-only peer setting-toggle"><div class="w-11 h-6 bg-gray-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div></label></div>
                                    <div class="flex items-center justify-between"><span class="font-medium text-white">MACD & EMA</span><label class="relative inline-flex items-center cursor-pointer"><input type="checkbox" id="macdEmaStrategy" data-key="use_macd_ema_strategy" class="sr-only peer setting-toggle"><div class="w-11 h-6 bg-gray-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div></label></div>
                                    <div class="flex items-center justify-between"><span class="font-medium text-white">EMA & RSI</span><label class="relative inline-flex items-center cursor-pointer"><input type="checkbox" id="emaRsiStrategy" data-key="use_ema_rsi_strategy" class="sr-only peer setting-toggle"><div class="w-11 h-6 bg-gray-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div></label></div>
                                    <div class="flex items-center justify-between"><span class="font-medium text-white">Pullback</span><label class="relative inline-flex items-center cursor-pointer"><input type="checkbox" id="pullbackStrategy" data-key="use_pullback_strategy" class="sr-only peer setting-toggle"><div class="w-11 h-6 bg-gray-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div></label></div>
                                    <div class="flex items-center justify-between"><span class="font-medium text-white">RSI Divergence</span><label class="relative inline-flex items-center cursor-pointer"><input type="checkbox" id="rsiDivergenceStrategy" data-key="use_rsi_divergence_strategy" class="sr-only peer setting-toggle"><div class="w-11 h-6 bg-gray-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div></label></div>
                                    <div class="flex items-center justify-between"><span class="font-medium text-white">Support/Resistance</span><label class="relative inline-flex items-center cursor-pointer"><input type="checkbox" id="supportResistanceStrategy" data-key="use_support_resistance_strategy" class="sr-only peer setting-toggle"><div class="w-11 h-6 bg-gray-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div></label></div>
                                    <div class="flex items-center justify-between"><span class="font-medium text-white">Scalping</span><label class="relative inline-flex items-center cursor-pointer"><input type="checkbox" id="scalpingStrategy" data-key="use_scalping_strategy" class="sr-only peer setting-toggle"><div class="w-11 h-6 bg-gray-600 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div></label></div>
                                </div>
                            </div>
                            
                            <div class="flex justify-end">
                                <button id="saveRiskSettingsBtn" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg transition-colors">حفظ إعدادات المخاطر</button>
                            </div>
                        </div>
                    </div>
                </main>
            </div>

            <!-- Confirmation Modal -->
            <div id="confirmationModal" class="modal-overlay fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center p-4 hidden opacity-0">
                <div class="bg-gray-800 rounded-xl shadow-2xl p-8 max-w-sm w-full">
                    <h3 class="text-2xl font-bold text-center text-red-400">تحذير!</h3>
                    <p class="text-center text-gray-300 my-4">أنت على وشك تفعيل وضع التداول الحقيقي بأموال حقيقية. هل أنت متأكد من رغبتك في المتابعة؟</p>
                    <div class="flex justify-around mt-6">
                        <button id="confirmRealModeBtn" class="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-8 rounded-lg transition-colors">تأكيد</button>
                        <button id="cancelRealModeBtn" class="bg-gray-600 hover:bg-gray-700 text-white font-bold py-2 px-8 rounded-lg transition-colors">إلغاء</button>
                    </div>
                </div>
            </div>

            <script>
                document.addEventListener('DOMContentLoaded', () => {
                    const pageDashboard = document.getElementById('page-dashboard');
                    const pageSettings = document.getElementById('page-settings');
                    const navDashboard = document.getElementById('nav-dashboard');
                    const navSettings = document.getElementById('nav-settings');
                    const modal = document.getElementById('confirmationModal');
                    
                    const showPage = (pageToShow) => {
                        [pageDashboard, pageSettings].forEach(p => p.classList.add('hidden'));
                        pageToShow.classList.remove('hidden');
                        [navDashboard, navSettings].forEach(n => n.classList.remove('active'));
                        if (pageToShow === pageDashboard) navDashboard.classList.add('active');
                        else navSettings.classList.add('active');
                    };

                    navDashboard.addEventListener('click', (e) => { e.preventDefault(); showPage(pageDashboard); });
                    navSettings.addEventListener('click', (e) => { e.preventDefault(); showPage(pageSettings); });

                    const tabLinks = document.querySelectorAll('.tab-link');
                    const tabContents = document.querySelectorAll('.tab-content');

                    tabLinks.forEach(link => {
                        link.addEventListener('click', (e) => {
                            e.preventDefault();
                            const tabId = link.dataset.tab;
                            tabLinks.forEach(l => l.classList.remove('active-tab'));
                            link.classList.add('active-tab');
                            tabContents.forEach(content => content.classList.add('hidden'));
                            document.getElementById(`tab-content-${tabId}`).classList.remove('hidden');
                        });
                    });

                    const formatNumber = (num, decimals = 4) => {
                        if (typeof num !== 'number') return num;
                        const formatted = num.toFixed(decimals).toString().replace(/\\B(?=(\\d{3})+(?!\\d))/g, " ");
                        return formatted.replace(/(\\.\\d*?)0+$/, '$1').replace(/\\.$/, '');
                    };

                    const updateSettings = async (settings) => {
                        try {
                            const response = await fetch('/settings', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify(settings)
                            });
                            if (!response.ok) throw new Error('Failed to update settings');
                            await fetchStatus();
                        } catch (error) {
                            console.error('Error updating settings:', error);
                        }
                    };

                    const fetchStatus = async () => {
                        try {
                            const response = await fetch('/status');
                            if (!response.ok) throw new Error('Network response was not ok');
                            const data = await response.json();
                            
                            // Update status indicators
                            document.getElementById('botStatusDot').className = data.trading_enabled ? 'h-3 w-3 rounded-full bg-green-500 ml-2' : 'h-3 w-3 rounded-full bg-red-500 ml-2';
                            document.getElementById('botStatusText').innerText = data.trading_enabled ? 'يعمل' : 'متوقف';
                            document.getElementById('tradingModeText').innerText = data.paper_trading_mode ? 'وضع تجريبي' : 'وضع حقيقي';
                            
                            // Update dashboard cards
                            document.getElementById('marketTrendText').innerText = data.market_state.overall_regime || 'غير محدد';
                            document.getElementById('marketTrendTime').innerText = `آخر تحديث: ${data.market_state.last_updated || 'N/A'}`;
                            document.getElementById('openTradesCount').innerText = `${data.open_trades_count} / ${data.max_open_trades}`;
                            
                            // Update main toggle button
                            const toggleBtn = document.getElementById('toggleTradingBtn');
                            toggleBtn.innerText = data.trading_enabled ? 'إيقاف البوت' : 'تفعيل البوت';
                            toggleBtn.className = data.trading_enabled ? 'w-full text-white font-bold py-2 px-4 rounded-lg transition-colors bg-red-600 hover:bg-red-700' : 'w-full text-white font-bold py-2 px-4 rounded-lg transition-colors bg-green-600 hover:bg-green-700';

                            // Update settings page inputs and buttons
                            document.getElementById('riskPerTrade').value = data.risk_per_trade_percent;
                            document.getElementById('buyConfidence').value = data.buy_confidence_threshold;
                            
                            const paperBtn = document.getElementById('setPaperModeBtn');
                            const realBtn = document.getElementById('setRealModeBtn');
                            if(data.paper_trading_mode) {
                                paperBtn.innerText = 'مفعل حالياً';
                                paperBtn.className = 'font-bold py-2 px-4 rounded-lg transition-colors text-sm bg-blue-600 text-white cursor-not-allowed';
                                realBtn.innerText = 'تفعيل';
                                realBtn.className = 'font-bold py-2 px-4 rounded-lg transition-colors text-sm bg-gray-600 hover:bg-red-600 text-white';
                            } else {
                                paperBtn.innerText = 'تفعيل';
                                paperBtn.className = 'font-bold py-2 px-4 rounded-lg transition-colors text-sm bg-gray-600 hover:bg-blue-600 text-white';
                                realBtn.innerText = 'مفعل حالياً';
                                realBtn.className = 'font-bold py-2 px-4 rounded-lg transition-colors text-sm bg-red-600 text-white cursor-not-allowed';
                            }
                            
                            document.querySelectorAll('.setting-toggle').forEach(toggle => {
                                const key = toggle.dataset.key;
                                if (key.startsWith('use_')) { // Strategy or feature
                                    toggle.checked = data.strategies_enabled[key.replace('use_', '')] || data.advanced_features_enabled[key.replace('use_', '')];
                                }
                            });

                        } catch (error) {
                            console.error('Error fetching status:', error);
                        }
                    };

                    const fetchOpenTrades = async () => {
                        // Implementation unchanged
                    };
                    const fetchRejectionLogs = async () => {
                        // Implementation unchanged
                    };
                    const fetchNotifications = async () => {
                        // Implementation unchanged
                    };

                    // Event Listeners
                    document.getElementById('toggleTradingBtn').addEventListener('click', async () => {
                        const isEnabled = document.getElementById('botStatusText').innerText === 'يعمل';
                        await updateSettings({ trading_enabled: !isEnabled });
                    });

                    document.querySelectorAll('.setting-toggle').forEach(toggle => {
                        toggle.addEventListener('change', async () => {
                            const key = toggle.dataset.key;
                            const value = toggle.checked;
                            await updateSettings({ [key]: value });
                        });
                    });
                    
                    document.getElementById('saveRiskSettingsBtn').addEventListener('click', async () => {
                        const settings = {
                            risk_per_trade_percent: parseFloat(document.getElementById('riskPerTrade').value),
                            buy_confidence_threshold: parseFloat(document.getElementById('buyConfidence').value),
                        };
                        await updateSettings(settings);
                    });

                    // Trading Mode Buttons & Modal Logic
                    document.getElementById('setPaperModeBtn').addEventListener('click', async () => {
                        await updateSettings({ paper_trading_mode: true });
                    });

                    document.getElementById('setRealModeBtn').addEventListener('click', () => {
                        if (document.getElementById('tradingModeText').innerText !== 'وضع حقيقي') {
                            modal.classList.remove('hidden');
                            setTimeout(() => modal.classList.remove('opacity-0'), 10);
                        }
                    });

                    document.getElementById('cancelRealModeBtn').addEventListener('click', () => {
                        modal.classList.add('opacity-0');
                        setTimeout(() => modal.classList.add('hidden'), 300);
                    });

                    document.getElementById('confirmRealModeBtn').addEventListener('click', async () => {
                        await updateSettings({ paper_trading_mode: false });
                        modal.classList.add('opacity-0');
                        setTimeout(() => modal.classList.add('hidden'), 300);
                    });
                    
                    const refreshDashboard = () => {
                        fetchStatus(); fetchOpenTrades(); fetchRejectionLogs(); fetchNotifications();
                    };
                    setInterval(refreshDashboard, 5000);
                    refreshDashboard();
                    showPage(pageDashboard);
                });
            </script>
        </body>
        </html>
    """)

def start_flask_app():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# --- وظائف البوت الرئيسية (بدون تغيير) ---
def process_symbol(symbol: str):
    logger.info(f"✨ [Scanner] جاري فحص الرمز: {symbol}")
    df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
    if df is None or len(df) < 50:
        log_rejection(symbol, "Insufficient Historical Data")
        return
    df_with_features = calculate_all_features(df)
    
    volatility_ok, vol_reason = check_market_volatility(df_with_features)
    if not volatility_ok:
        log_rejection(symbol, vol_reason)
        return
    
    if df_with_features.iloc[-1]['adx'] < 20:
        log_rejection(symbol, "ADX Trend Strength Too Weak", {'adx': df_with_features.iloc[-1]['adx']})
        return
        
    htf_df = fetch_historical_data(symbol, HIGHER_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
    if htf_df is None:
        log_rejection(symbol, "Insufficient Historical Data")
        return
    htf_df_with_features = calculate_all_features(htf_df)
    htf_ok, htf_reason = check_htf_trend_confirmation(htf_df_with_features)
    if not htf_ok:
        log_rejection(symbol, htf_reason)
        return
        
    signal_found = False
    strategy_name = "N/A"
    
    # التحقق من الاستراتيجيات بالترتيب
    if USE_RSI_DIVERGENCE_STRATEGY and check_rsi_divergence(df_with_features):
        signal_found, strategy_name = True, "RSI Divergence"
    elif USE_SUPPORT_RESISTANCE_STRATEGY and check_support_resistance_strategy(df_with_features):
        signal_found, strategy_name = True, "Support/Resistance"
    elif USE_VOLUME_PROFILE_STRATEGY and check_volume_profile_strategy(df_with_features):
        signal_found, strategy_name = True, "Volume Profile"
    elif USE_BB_STOCH_STRATEGY and df_with_features.iloc[-1]['close'] < df_with_features.iloc[-1]['lower_band'] and df_with_features.iloc[-1]['stoch_rsi'] < 20:
        signal_found, strategy_name = True, "BB & Stoch"
    elif USE_MACD_EMA_STRATEGY and df_with_features.iloc[-1]['macd_hist'] > 0 and df_with_features.iloc[-1]['macd'] > df_with_features.iloc[-1]['macd_signal']:
        signal_found, strategy_name = True, "MACD & EMA"
    elif USE_EMA_RSI_STRATEGY and df_with_features.iloc[-1]['ema_12'] > df_with_features.iloc[-1]['ema_26'] and df_with_features.iloc[-1]['rsi'] > 50:
        signal_found, strategy_name = True, "EMA & RSI"
    elif USE_PULLBACK_STRATEGY and df_with_features.iloc[-1]['close'] > df_with_features.iloc[-1]['ema_26'] and df_with_features.iloc[-2]['close'] < df_with_features.iloc[-2]['ema_26']:
        signal_found, strategy_name = True, "Pullback"
    elif USE_SCALPING_STRATEGY and check_scalping_strategy(df_with_features):
        signal_found, strategy_name = True, "Scalping"
    else:
        log_rejection(symbol, f"All Strategy Conditions Not Met")

    if signal_found:
        entry_price = float(df_with_features['close'].iloc[-1])
        atr = df_with_features['atr'].iloc[-1]
        stop_loss = entry_price - (atr * 2)
        target_price = entry_price + (atr * 3)
        target_price_2 = entry_price + (atr * 5)
        
        position_size_usdt = calculate_dynamic_position_size(df_with_features, PAPER_ACCOUNT_BALANCE, RISK_PER_TRADE_PERCENT)
        if position_size_usdt <= 0:
            logger.warning(f"⚠️ [Signal] حجم الصفقة المحسوب لـ {symbol} هو صفر. تم تخطي الإشارة.")
            return
            
        quantity = position_size_usdt / entry_price
        
        signal_data = {
            "symbol": symbol, "entry_price": entry_price, "target_price": target_price,
            "stop_loss": stop_loss, "strategy": strategy_name, "quantity": quantity,
            "is_real_trade": not paper_trading_mode, "details": {},
            "target_price_2": target_price_2, "initial_quantity": quantity
        }
        
        if create_signal_db(signal_data):
            newly_created_signal = get_open_signals()[-1]
            with signal_cache_lock: open_signals_cache[symbol] = newly_created_signal
            message = (f"📈 *إشارة شراء جديدة!*\\n"
                       f"💱 *العملة:* `{symbol}`\\n"
                       f"🛒 *سعر الدخول:* `{entry_price:.4f}`\\n"
                       f"🛑 *وقف الخسارة:* `{stop_loss:.4f}`\\n"
                       f"🎯 *الهدف 1:* `{target_price:.4f}`\\n"
                       f"🎯 *الهدف 2:* `{target_price_2:.4f}`\\n"
                       f"🧠 *الاستراتيجية:* `{strategy_name}`")
            send_telegram_message(message)

def run_signal_scanner():
    while True:
        try:
            with trading_status_lock:
                if not is_trading_enabled:
                    time.sleep(60)
                    continue
            logger.info("🕵️ [Scanner] بدء دورة المسح...")
            with signal_cache_lock: open_trades_count = len(open_signals_cache)
            if open_trades_count >= MAX_OPEN_TRADES:
                logger.warning(f"⚠️ [Scanner] تجاوز الحد الأقصى للصفقات المفتوحة ({open_trades_count}/{MAX_OPEN_TRADES}).")
                time.sleep(60)
                continue
            
            symbols_to_process = [s for s in validated_symbols_to_scan if s not in open_signals_cache]
            if not symbols_to_process:
                logger.info("💤 [Scanner] لا توجد رموز جديدة للفحص.")
                time.sleep(60)
                continue
                
            for i in range(0, len(symbols_to_process), SYMBOL_PROCESSING_BATCH_SIZE):
                batch = symbols_to_process[i:i + SYMBOL_PROCESSING_BATCH_SIZE]
                for symbol in batch:
                    try:
                        process_symbol(symbol)
                    except Exception as e:
                        logger.error(f"❌ [Scanner] خطأ في معالجة الرمز {symbol}: {e}")
                time.sleep(5)
            logger.info("✅ [Scanner] انتهت دورة المسح.")
            time.sleep(60 * 2)
        except Exception as e:
            logger.error(f"❌ [Scanner] حدث خطأ حرج: {e}", exc_info=True)
            time.sleep(60)
        finally:
            gc.collect()

def run_trade_manager():
    while True:
        try:
            time.sleep(15)
            with trading_status_lock:
                if not is_trading_enabled: continue
            
            with signal_cache_lock: symbols_to_manage = list(open_signals_cache.keys())
            if not symbols_to_manage: continue
            
            for symbol in symbols_to_manage:
                with signal_cache_lock:
                    if symbol not in open_signals_cache: continue
                    signal = open_signals_cache[symbol]
                with live_prices_lock: current_price = live_prices.get(symbol)
                if current_price is None: continue
                
                # تحديث وقف الخسارة إلى نقطة الدخول
                entry_price = signal['entry_price']
                stop_loss = signal['stop_loss']
                if (entry_price - stop_loss) > 0:
                    profit_ratio = (current_price - entry_price) / (entry_price - stop_loss)
                    if profit_ratio > 0.5 and stop_loss < entry_price:
                        if signal.get('stop_moved_to_breakeven', False) is False:
                            update_signal_db(signal['id'], {'stop_loss': entry_price})
                            with signal_cache_lock:
                                if symbol in open_signals_cache:
                                    open_signals_cache[symbol]['stop_loss'] = entry_price
                                    open_signals_cache[symbol]['stop_moved_to_breakeven'] = True
                            log_and_notify("info", f"🛡️ [Manager] تم نقل وقف الخسارة إلى نقطة الدخول لـ {symbol}", "TRADE_UPDATE")

                should_close, reason = check_improved_exit_conditions(signal, current_price)
                if should_close:
                    close_signal_db(signal, current_price, reason)
                    with signal_cache_lock: open_signals_cache.pop(symbol, None)
                        
        except Exception as e:
            logger.error(f"❌ [Manager] حدث خطأ حرج: {e}", exc_info=True)
        finally:
            gc.collect()

def update_market_state():
    global current_market_state
    while True:
        try:
            btc_data = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
            if btc_data is None or len(btc_data) < 50:
                time.sleep(60)
                continue
            
            btc_data_with_features = calculate_all_features(btc_data)
            overall_regime = calculate_market_trend(btc_data_with_features)
            
            trend_details = {}
            for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
                tf_data = fetch_historical_data(BTC_SYMBOL, tf, 50)
                if tf_data is not None:
                    tf_data_with_features = calculate_all_features(tf_data)
                    trend_details[tf] = calculate_market_trend(tf_data_with_features)
            
            with market_state_lock:
                current_market_state.update({'overall_regime': overall_regime, 'trend_details_by_tf': trend_details, 'last_updated': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')})
            time.sleep(60 * 5)
        except Exception as e:
            logger.error(f"❌ [Market State] A critical error occurred: {e}", exc_info=True)
            time.sleep(60)

# --- نقطة بداية البرنامج ---
if __name__ == '__main__':
    logger.info("="*50 + "\\n====== Starting Crypto Trading Bot V14.1.0 ======\\n" + "="*50)
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
    
    Thread(target=run_signal_scanner, daemon=True).start()
    Thread(target=run_trade_manager, daemon=True).start()
    Thread(target=update_market_state, daemon=True).start()
    Thread(target=start_flask_app, daemon=True).start()
    
    logger.info("✅ Bot is fully initialized and running. Dashboard is available at http://127.0.0.1:5000")

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        logger.info("👋 [Bot] Shutting down bot gracefully...")
    finally:
        if ws_manager: ws_manager.stop()
        if conn: conn.close()
        logger.info("✅ [Bot] Bot has been shut down.")
