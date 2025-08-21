# ملف c4.py - نسخة V17.3.1 (إصلاح خطأ اتصال قاعدة البيانات)
# --- وصف الإصدار:
# هذا الإصدار يضيف دالة check_db_connection المفقودة التي كانت تسبب خطأ NameError عند بدء التشغيل.
# 1.  [جديد] إضافة دالة check_db_connection للتحقق من حالة الاتصال بقاعدة البيانات وإعادة الاتصال عند الحاجة.
# 2.  [إصلاح] استدعاء الدالة الجديدة في جميع الأماكن التي تتطلب عمليات قاعدة البيانات لضمان استقرار الاتصال.
# 3.  [محسن] تحسين طفيف في منطق استدعاء الدالة ليكون أكثر إيجازًا.

import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import redis
from decimal import Decimal, ROUND_DOWN
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
        logging.FileHandler('crypto_bot_v17_3_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV17.3.1')

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
trading_mode_lock = Lock()
usdt_balance: float = 0.0
balance_lock = Lock()


# --- المتغيرات القابلة للتعديل ---
RISK_PER_TRADE_PERCENT: float = 0.85
risk_per_trade_lock = Lock()
MAX_OPEN_TRADES: int = 3
PAPER_TRADE_SIZE_USDT: float = 10.0
TRAILING_STOP_ACTIVATION_PROFIT_PERCENT: float = 1.4

# --- مفاتيح تفعيل الاستراتيجيات ---
USE_BB_STOCH_STRATEGY: bool = True
USE_MACD_EMA_STRATEGY: bool = True
USE_EMA_RSI_STRATEGY: bool = True
USE_PULLBACK_STRATEGY: bool = True
USE_MOMENTUM_VOLATILITY_STRATEGY: bool = True

# --- إعدادات الفلاتر الديناميكية للاستراتيجيات ---
STRATEGY_NAMES = {
    "BB_Stoch_Strategy": "BB+Stoch (انعكاسية)", "MACD_EMA_Strategy": "MACD+EMA (اتجاهية)",
    "EMA_RSI_Strategy": "EMA+RSI (مختلطة)", "Pullback_Strategy": "Pullback (انعكاسية)",
    "Momentum_Volatility_Strategy": "Momentum (زخم)"
}
STRATEGY_FILTER_CONFIG = {
    "BB_Stoch_Strategy": {"profile": "Reversal", "adx_threshold": 18, "htf_confirmation_mode": "Disabled"},
    "MACD_EMA_Strategy": {"profile": "Strict", "adx_threshold": 22, "htf_confirmation_mode": "Strict"},
    "EMA_RSI_Strategy": {"profile": "Moderate", "adx_threshold": 20, "htf_confirmation_mode": "Relaxed"},
    "Pullback_Strategy": {"profile": "Reversal", "adx_threshold": 18, "htf_confirmation_mode": "Relaxed"},
    "Momentum_Volatility_Strategy": {"profile": "Strict", "adx_threshold": 25, "htf_confirmation_mode": "Strict"},
}
strategy_filters_lock = Lock()
BASE_FILTER_ADX_THRESHOLD = 20

# --- إعدادات عامة ---
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '1h'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 15
BTC_SYMBOL: str = 'BTCUSDT'
API_REQUEST_DELAY: float = 1

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
current_market_state: Dict[str, Any] = {"trend_details_by_tf": {}}
market_state_lock = Lock()

# --- قاموس أسباب الرفض باللغة العربية ---
REJECTION_REASONS_AR = {
    "Market Volatility Filter Failed": "فلتر تقلب السوق رفض الدخول",
    "Trend Strength Filter Failed": "فلتر قوة الاتجاه رفض الدخول",
    "HTF Trend Confirmation Failed": "فشل تأكيد الترند على الفريم الأعلى",
    "Insufficient Historical Data": "بيانات تاريخية غير كافية للفحص",
    "MinNotional Filter Failed": "قيمة الصفقة أقل من الحد الأدنى للمنصة",
    "Insufficient Balance": "الرصيد غير كافي لتنفيذ الصفقة",
    "Bullish Confirmation Failed": "فشل تأكيد الشمعة الصعودية",
}

# --- إعداد تطبيق Flask ---
app = Flask(__name__)
CORS(app)

# --- دوال تهيئة الخدمات وقاعدة البيانات ---
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
                        stop_loss DOUBLE PRECISION NOT NULL, status TEXT DEFAULT 'open', 
                        closing_price DOUBLE PRECISION, closed_at TIMESTAMP, profit_percentage DOUBLE PRECISION, 
                        strategy_name TEXT, signal_details JSONB, is_real_trade BOOLEAN DEFAULT FALSE, 
                        quantity DOUBLE PRECISION, closing_reason TEXT, order_id TEXT,
                        target_price_1 DOUBLE PRECISION, target_price_2 DOUBLE PRECISION,
                        initial_quantity DOUBLE PRECISION
                    );
                """)
                cur.execute("CREATE TABLE IF NOT EXISTS notifications (id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(), type TEXT NOT NULL, message TEXT NOT NULL);")
            conn.commit()
            logger.info("✅ [DB] Database connection and schema verified successfully.")
            return
        except Exception as e:
            logger.error(f"❌ [DB] Error during initialization (Attempt {attempt + 1}/{retries}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: logger.critical("❌ [DB] Failed to connect to the database. Exiting.")

def init_redis() -> None:
    global redis_client
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] Connected successfully.")
    except redis.exceptions.ConnectionError as e:
        logger.warning(f"⚠️ [Redis] Connection failed: {e}.")
        redis_client = None

# [FIX] Added the missing database connection check function
def check_db_connection() -> bool:
    """Checks if the database connection is alive and tries to reconnect if not."""
    global conn
    try:
        # If conn is None or closed, attempt to reconnect
        if conn is None or conn.closed != 0:
            logger.warning("⚠️ [DB] Connection lost or not established. Attempting to reconnect...")
            init_db()
            # Check again after trying to initialize
            if conn is None or conn.closed != 0:
                logger.error("❌ [DB] Reconnection failed.")
                return False
            logger.info("✅ [DB] Reconnection successful.")
        return True
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [DB] Connection check failed with an exception: {e}. Attempting to reconnect...")
        init_db()
        if conn is None or conn.closed != 0:
            logger.error("❌ [DB] Reconnection after exception failed.")
            return False
        logger.info("✅ [DB] Reconnection after exception was successful.")
        return True


# --- دوال المساعدة والإشعارات ---
def log_and_notify(level: str, message: str, notification_type: str):
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error}
    log_methods.get(level.lower(), logger.info)(message)
    try:
        new_notification = {"timestamp": datetime.now(timezone.utc).isoformat(), "type": notification_type, "message": message}
        with notifications_lock: notifications_cache.appendleft(new_notification)
    except Exception as e:
        logger.error(f"❌ [Cache] Failed to save notification to cache: {e}")

# --- WebSocket & Data Fetching ---
def handle_socket_message(msg):
    global live_prices
    if msg and 'e' in msg and msg['e'] == 'error': logger.error(f"❌ [WebSocket] Error: {msg['m']}"); return
    if isinstance(msg, list):
        with live_prices_lock:
            for ticker in msg:
                if 's' in ticker and 'c' in ticker: live_prices[ticker['s']] = float(ticker['c'])

def start_websocket():
    global ws_manager
    ws_manager = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    ws_manager.start()
    ws_manager.start_ticker_socket(callback=handle_socket_message)
    logger.info("✅ [WebSocket] Subscribed to ticker stream.")

def get_exchange_info_map() -> None:
    global exchange_info_map
    try:
        logger.info("[API] Fetching exchange info...")
        exchange_info_map = {s['symbol']: s for s in client.get_exchange_info()['symbols']}
    except Exception as e:
        logger.error(f"❌ [API] Error fetching exchange info: {e}")

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    try:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
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
    time.sleep(API_REQUEST_DELAY)
    try:
        klines = client.get_historical_klines(symbol, interval, f"{days} day ago UTC")
        if not klines: return None
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna().astype(float)
    except Exception as e:
        logger.error(f"❌ [Data] Error fetching data for {symbol}: {e}"); return None

def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    df_calc['ema9'] = df_calc['close'].ewm(span=9, adjust=False).mean()
    df_calc['ema21'] = df_calc['close'].ewm(span=21, adjust=False).mean()
    df_calc['ema50'] = df_calc['close'].ewm(span=50, adjust=False).mean()
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    df_calc['atr'] = tr.ewm(span=14, adjust=False).mean()
    up_move = df_calc['high'].diff(); down_move = -df_calc['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=14, adjust=False).mean()
    delta = df_calc['close'].diff(); gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean(); avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df_calc['rsi'] = 100 - (100 / (1 + rs))
    rsi_val = df_calc['rsi']
    stoch_rsi = (rsi_val - rsi_val.rolling(14).min()) / (rsi_val.rolling(14).max() - rsi_val.rolling(14).min()).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi.rolling(3).mean() * 100
    bb_middle = df_calc['close'].rolling(window=20).mean()
    bb_std = df_calc['close'].rolling(window=20).std()
    df_calc['bb_lower'] = bb_middle - (bb_std * 2)
    exp1 = df_calc['close'].ewm(span=12, adjust=False).mean(); exp2 = df_calc['close'].ewm(span=26, adjust=False).mean()
    df_calc['macd'] = exp1 - exp2
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()
    df_calc['atr_percent'] = (df_calc['atr'] / df_calc['close']) * 100
    return df_calc

# --- Data Loading ---
def load_open_signals_to_cache():
    # [FIX] Use the new check_db_connection function
    if not check_db_connection(): return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status IN ('open', 'updated');")
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in cur.fetchall(): open_signals_cache[signal['symbol']] = dict(signal)
            logger.info(f"✅ [Cache] Loaded {len(open_signals_cache)} open signals.")
    except Exception as e:
        logger.error(f"❌ [Cache] Failed to load open signals: {e}")

# --- Filters & Strategies ---
def check_bb_stoch_strategy_enhanced(df: pd.DataFrame) -> bool:
    if len(df) < 21: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    signal_condition = (prev['close'] < prev['bb_lower']) and (last['close'] > last['bb_lower']) and (last['stoch_rsi_k'] < 30)
    bullish_confirmation = last['close'] > last['open']
    return signal_condition and bullish_confirmation

def check_pullback_strategy_enhanced(df: pd.DataFrame) -> bool:
    if len(df) < 50: return False
    last = df.iloc[-1]
    signal_condition = (last['close'] > last['ema50']) and (last['low'] < last['ema21']) and (last['close'] > last['ema21'])
    bullish_confirmation = last['close'] > last['open']
    return signal_condition and bullish_confirmation

# (Other strategy checks remain the same)
def check_macd_ema_strategy_enhanced(df: pd.DataFrame) -> bool:
    if len(df) < 30: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    return (prev['macd'] < prev['macd_signal']) and (last['macd'] > last['macd_signal']) and (last['close'] > last['ema50'])

def check_ema_rsi_strategy_enhanced(df: pd.DataFrame) -> bool:
    if len(df) < 30: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    return (prev['ema9'] < prev['ema21']) and (last['ema9'] > last['ema21']) and (50 < last['rsi'] < 65)

def check_momentum_volatility_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 50: return False
    last = df.iloc[-1]
    return (last['atr_percent'] > df['atr_percent'].rolling(14).mean().iloc[-1] * 1.5) and (last['close'] > last['ema9'])

# --- نظام إدارة الصفقات ---
def calculate_trade_levels(df: pd.DataFrame) -> Dict[str, Any]:
    last = df.iloc[-1]
    atr = last['atr']
    entry_price = last['close']
    stop_loss = entry_price - (atr * 1.5)
    target_price_1 = entry_price + (atr * 2.0)
    target_price_2 = entry_price + (atr * 3.5)
    trailing_stop_distance = atr * 1.5
    return {
        "entry_price": entry_price, "stop_loss": stop_loss, "target_price_1": target_price_1,
        "target_price_2": target_price_2, "atr": atr,
        "trailing_stop_distance": trailing_stop_distance
    }

def adjust_quantity_to_step_size(quantity: float, step_size: str) -> float:
    return float(Decimal(quantity).quantize(Decimal(step_size), rounding=ROUND_DOWN))

def create_trade_signal(symbol: str, df: pd.DataFrame, strategy_name: str):
    with trading_mode_lock:
        is_real = not paper_trading_mode

    trade_levels = calculate_trade_levels(df)
    entry_price = trade_levels['entry_price']
    
    if is_real:
        with balance_lock: current_usdt_balance = usdt_balance
        with risk_per_trade_lock: trade_size_usdt = current_usdt_balance * (RISK_PER_TRADE_PERCENT / 100)

        if trade_size_usdt <= 0: return

        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info: return

        min_notional = float(next((f['minNotional'] for f in symbol_info['filters'] if f['filterType'] == 'NOTIONAL'), '0.0'))
        if trade_size_usdt < min_notional: return

        quantity = trade_size_usdt / entry_price
        step_size = next((f['stepSize'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), '0.000001')
        adjusted_quantity = adjust_quantity_to_step_size(quantity, step_size)

        if adjusted_quantity <= 0: return

        try:
            logger.info(f"💰 [Real Trade] Placing LIVE MARKET BUY order for {adjusted_quantity} of {symbol}")
            order = client.create_order(symbol=symbol, side=Client.SIDE_BUY, type=Client.ORDER_TYPE_MARKET, quantity=adjusted_quantity)
            
            avg_fill_price = sum(float(f['price']) * float(f['qty']) for f in order.get('fills', [])) / sum(float(f['qty']) for f in order.get('fills', [])) if order.get('fills') else entry_price
            final_quantity = float(order.get('executedQty', adjusted_quantity))
            order_id = order.get('orderId', 'N/A')
            
            save_signal_to_db(symbol, avg_fill_price, trade_levels, strategy_name, True, final_quantity, order_id)

        except Exception as e:
            logger.error(f"❌ [Real Trade] CRITICAL ERROR creating real trade for {symbol}: {e}", exc_info=True)

    else: # Paper Trading
        quantity = PAPER_TRADE_SIZE_USDT / entry_price
        save_signal_to_db(symbol, entry_price, trade_levels, strategy_name, False, quantity)

def save_signal_to_db(symbol: str, entry_price: float, trade_levels: Dict, strategy_name: str, is_real: bool, quantity: float, order_id: Optional[str] = None):
    try:
        # [FIX] Use the new check_db_connection function
        if not check_db_connection(): return
        signal_details = {
            "atr": trade_levels['atr'], "is_trailing_active": False,
            "trailing_stop_distance": trade_levels['trailing_stop_distance']
        }
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price_1, target_price_2, stop_loss, status, 
                                   strategy_name, is_real_trade, quantity, initial_quantity, signal_details, order_id) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
            """, (symbol, float(entry_price), float(trade_levels['target_price_1']), float(trade_levels['target_price_2']), 
                  float(trade_levels['stop_loss']), 'open', strategy_name, is_real, float(quantity), float(quantity), 
                  json.dumps(signal_details, cls=NpEncoder), order_id))
            new_id = cur.fetchone()['id']
        conn.commit()
        
        signal_data = {
            'id': new_id, 'symbol': symbol, 'entry_price': float(entry_price), 
            'target_price_1': float(trade_levels['target_price_1']), 'target_price_2': float(trade_levels['target_price_2']),
            'stop_loss': float(trade_levels['stop_loss']), 'status': 'open', 'strategy_name': strategy_name, 
            'is_real_trade': is_real, 'quantity': float(quantity), 'initial_quantity': float(quantity),
            'signal_details': signal_details, 'order_id': order_id
        }
        with signal_cache_lock: open_signals_cache[symbol] = signal_data
        log_and_notify("info", f"Opened {'REAL' if is_real else 'PAPER'} trade for {symbol}", "TRADE_OPEN")

    except Exception as e:
        logger.error(f"❌ [DB] CRITICAL ERROR saving signal for {symbol}: {e}", exc_info=True)
        if conn: conn.rollback()

# --- قوالب HTML ---
DASHBOARD_TEMPLATE = """
<!DOCTYPE html><html dir="rtl" lang="ar"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>لوحة تحكم بوت التداول</title><link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap" rel="stylesheet"><style>:root{--bg-dark:#121212;--bg-surface:#1e1e1e;--primary:#BB86FC;--primary-variant:#3700B3;--text-light:#e0e0e0;--text-medium:#a0a0a0;--success:#4CAF50;--danger:#F44336;--warning:#FFC107;--info:#2196F3;}body{background-color:var(--bg-dark);color:var(--text-light);font-family:'Tajawal',sans-serif;margin:0;padding:20px;box-sizing:border-box;}.container{max-width:1400px;margin:0 auto;}header{background-color:var(--bg-surface);padding:15px 25px;border-radius:12px;margin-bottom:25px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:15px;}.header-title{font-size:24px;font-weight:700;color:var(--primary);}.status-indicator{display:flex;align-items:center;gap:15px;}.status-dot{width:12px;height:12px;border-radius:50%;background-color:var(--danger);transition:background-color 0.5s ease;}.status-dot.active{background-color:var(--success);}.btn{background-color:var(--primary-variant);color:white;border:none;padding:10px 20px;border-radius:8px;cursor:pointer;font-size:14px;}.btn-small{padding:5px 10px;font-size:12px;}.btn.stop{background-color:var(--danger);}.dashboard-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px;}.card{background-color:var(--bg-surface);border-radius:12px;padding:20px;display:flex;flex-direction:column;}.card-title{font-size:18px;font-weight:700;margin:0 0 15px 0;padding-bottom:10px;border-bottom:1px solid #333;}.scrollable-content{overflow-y:auto;max-height:400px;flex-grow:1;}.item{padding:12px;border-radius:8px;margin-bottom:10px;border-left:4px solid var(--primary);background-color:#252525;}.item.real-trade-item{border-left-color:var(--info);}.item-header{display:flex;justify-content:space-between;align-items:center;}.item-title{font-weight:700;}.item-content{font-size:13px;margin-top:5px;}.trend-container{display:flex;justify-content:space-around;align-items:center;padding:15px 0;}.trend-item{text-align:center;}.trend-label{font-size:14px;color:var(--text-medium);margin-bottom:8px;}.trend-status{font-size:18px;font-weight:700;}.trend-up{color:var(--success);}.trend-down{color:var(--danger);}.trend-sideways{color:var(--warning);}.progress-bar-container{width:100%;background-color:#3c3c3c;border-radius:5px;height:10px;margin:8px 0;overflow:hidden;}.progress-bar{height:100%;transition:width 0.4s ease-in-out;}.progress-bar.profit{background-color:var(--success);}.progress-bar.loss{background-color:var(--danger);}.item-footer{display:flex;justify-content:space-between;font-size:12px;color:var(--text-medium);margin-top:4px;}.trade-mode-card{grid-column:1/-1;display:flex;justify-content:space-between;align-items:center;}.trade-mode-status span{font-weight:700;padding:4px 12px;border-radius:8px;}.trade-mode-paper{color:var(--warning);background-color:rgba(255,193,7,0.1);}.trade-mode-real{color:var(--info);background-color:rgba(33,150,243,0.1);}</style></head><body><div class="container"><header><div class="header-title">بوت التداول V17.3.1</div><div class="status-indicator"><div id="status-dot" class="status-dot"></div><span id="status-text">متوقف</span><button id="toggle-trading-btn" class="btn">تشغيل</button></div></header><div class="dashboard-grid"><div class="card trade-mode-card"><div id="trade-mode-status"></div><div id="balance-display"></div><button id="toggle-real-trading-btn" class="btn"></button></div><div class="card"><div class="card-title">اتجاه السوق (BTC)</div><div id="market-trend-container" class="trend-container"></div></div><div class="card"><div class="card-title" id="open-signals-title">الإشارات المفتوحة (0)</div><div id="open-signals-container" class="scrollable-content"></div></div><div class="card"><div class="card-title">الإشعارات</div><div id="notifications-container" class="scrollable-content"></div></div></div></div>
<script>
// مخزن لحفظ بيانات الصفقات الثابتة لتجنب إعادة طلبها
let openSignalsData = {};

function toggleTrading() {
    fetch('/toggle_trading', { method: 'POST' })
        .then(res => res.json())
        .then(data => updateStaticUI(data.state))
        .catch(err => console.error('Error toggling trading:', err));
}

function toggleRealTrading() {
    if (confirm('تحذير: أنت على وشك تغيير وضع التداول. هل أنت متأكد؟')) {
        fetch('/toggle_real_trading', { method: 'POST' })
            .then(res => res.json())
            .then(data => updateStaticUI(data.state))
            .catch(err => console.error('Error toggling real trading:', err));
    }
}

function closeTrade(signalId, symbol) {
    if (confirm(`هل أنت متأكد من رغبتك في إغلاق الصفقة لـ ${symbol} يدويًا؟`)) {
        fetch(`/close_trade/${signalId}`, { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                alert(data.message);
                fetchInitialData(); // إعادة تحميل البيانات بعد الإغلاق
            })
            .catch(err => alert('حدث خطأ أثناء محاولة إغلاق الصفقة.'));
    }
}

// دالة لتحديث الأجزاء الثابتة من الواجهة
function updateStaticUI(state) {
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const toggleTradingBtn = document.getElementById('toggle-trading-btn');
    if (state.trading_enabled) {
        statusDot.classList.add('active');
        statusText.textContent = 'نشط';
        toggleTradingBtn.textContent = 'إيقاف';
        toggleTradingBtn.classList.add('stop');
    } else {
        statusDot.classList.remove('active');
        statusText.textContent = 'متوقف';
        toggleTradingBtn.textContent = 'تشغيل';
        toggleTradingBtn.classList.remove('stop');
    }

    const tradeModeStatus = document.getElementById('trade-mode-status');
    const toggleRealBtn = document.getElementById('toggle-real-trading-btn');
    if (state.paper_trading_mode) {
        tradeModeStatus.innerHTML = 'وضع التداول: <span class="trade-mode-paper">ورقي</span>';
        toggleRealBtn.textContent = 'تفعيل التداول الحقيقي';
        toggleRealBtn.className = 'btn real-mode';
    } else {
        tradeModeStatus.innerHTML = 'وضع التداول: <span class="trade-mode-real">حقيقي (LIVE)</span>';
        toggleRealBtn.textContent = 'العودة للتداول الورقي';
        toggleRealBtn.className = 'btn stop';
    }

    document.getElementById('balance-display').innerHTML = `الرصيد: <b>$${state.usdt_balance.toFixed(2)}</b>`;
    
    const marketTrendContainer = document.getElementById('market-trend-container');
    marketTrendContainer.innerHTML = '';
    if (state.market_state && state.market_state.trend_details_by_tf) {
        Object.entries(state.market_state.trend_details_by_tf).forEach(([tf, trend_data]) => {
            let trendClass = 'trend-sideways';
            let trendText = 'متذبذب';
            if (trend_data.trend === 'Bullish') { trendClass = 'trend-up'; trendText = 'صاعد'; }
            else if (trend_data.trend === 'Bearish') { trendClass = 'trend-down'; trendText = 'هابط'; }
            marketTrendContainer.innerHTML += `<div class="trend-item"><div class="trend-label">${tf}</div><div class="trend-status ${trendClass}">${trendText}</div></div>`;
        });
    }

    const notificationsContainer = document.getElementById('notifications-container');
    notificationsContainer.innerHTML = '';
    state.notifications.forEach(n => {
        notificationsContainer.innerHTML += `<div class="item"><div class="item-content">${n.message}</div></div>`;
    });

    // بناء هيكل الصفقات المفتوحة
    openSignalsData = state.open_signals;
    const signalsContainer = document.getElementById('open-signals-container');
    signalsContainer.innerHTML = '';
    document.getElementById('open-signals-title').textContent = `الإشارات المفتوحة (${Object.keys(openSignalsData).length})`;

    Object.entries(openSignalsData).forEach(([symbol, signal]) => {
        const isReal = signal.is_real_trade ? 'real-trade-item' : '';
        signalsContainer.innerHTML += `
            <div class="item ${isReal}" id="signal-${symbol}">
                <div class="item-header">
                    <div class="item-title">${symbol}</div>
                    <button class="btn btn-small stop" onclick="closeTrade(${signal.id},'${symbol}')">إغلاق</button>
                </div>
                <div class="item-content">
                    <span>الدخول: ${signal.entry_price.toFixed(4)}</span> | <span id="price-${symbol}">الحالي: ...</span>
                </div>
                <div class="progress-bar-container" id="progress-${symbol}">
                    <div class="progress-bar"></div>
                </div>
                <div class="item-footer">
                    <span>الوقف: ${signal.stop_loss.toFixed(4)}</span>
                    <span>الهدف: ${signal.target_price_1.toFixed(4)}</span>
                </div>
            </div>`;
    });
}

// دالة لتحديث الأجزاء المتحركة (الأسعار والتقدم)
function updateDynamicData(updates) {
    Object.entries(updates.prices).forEach(([symbol, currentPrice]) => {
        const signal = openSignalsData[symbol];
        if (!signal) return;

        const priceEl = document.getElementById(`price-${symbol}`);
        if (priceEl) {
            priceEl.textContent = `الحالي: ${currentPrice.toFixed(4)}`;
        }

        const progressContainer = document.getElementById(`progress-${symbol}`);
        if (progressContainer) {
            let progressBarHtml = '';
            const entryPrice = signal.entry_price;
            const stopLoss = signal.stop_loss;
            const targetPrice1 = signal.target_price_1;

            if (currentPrice > entryPrice && targetPrice1 > entryPrice) {
                const progress = Math.min(((currentPrice - entryPrice) / (targetPrice1 - entryPrice)) * 100, 100);
                progressBarHtml = `<div class="progress-bar profit" style="width:${progress}%;"></div>`;
            } else if (currentPrice < entryPrice && entryPrice > stopLoss) {
                const progress = Math.min(((entryPrice - currentPrice) / (entryPrice - stopLoss)) * 100, 100);
                progressBarHtml = `<div class="progress-bar loss" style="width:${progress}%;"></div>`;
            }
            progressContainer.innerHTML = progressBarHtml;
        }
    });
}

async function fetchInitialData() {
    try {
        const response = await fetch('/api/dashboard_data');
        const data = await response.json();
        updateStaticUI(data);
        updateDynamicData({ prices: data.live_prices }); // تحديث الأسعار الأولية
    } catch (error) {
        console.error('Failed to fetch initial data:', error);
    }
}

async function fetchUpdates() {
    if (Object.keys(openSignalsData).length === 0) return; // لا تطلب تحديثات إذا لم تكن هناك صفقات
    try {
        const response = await fetch('/api/dashboard_updates');
        const data = await response.json();
        updateDynamicData(data);
    } catch (error) {
        console.error('Failed to fetch updates:', error);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('toggle-trading-btn').onclick = toggleTrading;
    document.getElementById('toggle-real-trading-btn').onclick = toggleRealTrading;
    fetchInitialData();
    setInterval(fetchUpdates, 2000); // تحديث الأسعار كل ثانيتين
});
</script></body></html>
"""

# --- مسارات Flask ---
@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_TEMPLATE)

# [مُعدل] هذا المسار الآن للتحميل الأولي فقط
@app.route('/api/dashboard_data')
def dashboard_data():
    try:
        with trading_status_lock: trading_enabled = is_trading_enabled
        with trading_mode_lock: is_paper_mode = paper_trading_mode
        with balance_lock: current_balance = usdt_balance
        with notifications_lock: notifications = list(notifications_cache)
        with market_state_lock: market_state = dict(current_market_state)
        with signal_cache_lock: open_signals = dict(open_signals_cache)
        
        # إرسال الأسعار الحالية مع البيانات الأولية لظهورها فوراً
        with live_prices_lock:
            live_prices_for_open_signals = {
                symbol: live_prices.get(symbol) 
                for symbol in open_signals.keys() 
                if live_prices.get(symbol) is not None
            }

        payload = {
            "trading_enabled": trading_enabled,
            "paper_trading_mode": is_paper_mode,
            "usdt_balance": current_balance,
            "open_signals": open_signals,
            "notifications": notifications,
            "market_state": market_state,
            "live_prices": live_prices_for_open_signals
        }
        return app.response_class(
            response=json.dumps(payload, cls=NpEncoder),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        logger.error(f"❌ [API Error] Failed to generate initial dashboard data: {e}", exc_info=True)
        return jsonify({"error": "Failed to load dashboard data."}), 500

# [جديد] مسار خفيف جداً لتحديث الأسعار فقط
@app.route('/api/dashboard_updates')
def dashboard_updates():
    try:
        with signal_cache_lock:
            open_symbols = list(open_signals_cache.keys())
        
        with live_prices_lock:
            prices = {
                symbol: live_prices.get(symbol) 
                for symbol in open_symbols 
                if live_prices.get(symbol) is not None
            }
        return jsonify({"prices": prices})
    except Exception as e:
        logger.error(f"❌ [API Error] Failed to generate price updates: {e}", exc_info=True)
        return jsonify({"prices": {}}), 500

# [مُعدل] مسارات التحكم الآن تُرجع فقط الحالة العامة
@app.route('/toggle_trading', methods=['POST'])
def toggle_trading():
    global is_trading_enabled
    with trading_status_lock: is_trading_enabled = not is_trading_enabled
    log_and_notify("info", f"Trading has been {'enabled' if is_trading_enabled else 'disabled'}.", "TRADING_STATUS")
    return jsonify(state=get_current_bot_state())

@app.route('/toggle_real_trading', methods=['POST'])
def toggle_real_trading():
    global paper_trading_mode
    with trading_mode_lock:
        with trading_status_lock:
            if is_trading_enabled and not paper_trading_mode:
                return jsonify({"success": False, "message": "يجب إيقاف البوت أولاً للعودة للوضع الورقي"})
        
        paper_trading_mode = not paper_trading_mode
        log_and_notify("info", f"Trading mode switched to {'Paper' if paper_trading_mode else 'Real'}.", "TRADING_MODE_SWITCH")
    return jsonify(state=get_current_bot_state())

def get_current_bot_state():
    """دالة مساعدة لجلب الحالة الحالية للبوت"""
    with trading_status_lock: trading_enabled = is_trading_enabled
    with trading_mode_lock: is_paper_mode = paper_trading_mode
    with balance_lock: current_balance = usdt_balance
    with market_state_lock: market_state = dict(current_market_state)
    with notifications_lock: notifications = list(notifications_cache)
    with signal_cache_lock: open_signals = dict(open_signals_cache)
    return {
        "trading_enabled": trading_enabled,
        "paper_trading_mode": is_paper_mode,
        "usdt_balance": current_balance,
        "market_state": market_state,
        "notifications": notifications,
        "open_signals": open_signals
    }

@app.route('/close_trade/<int:signal_id>', methods=['POST'])
def manual_close_trade(signal_id):
    with signal_cache_lock:
        signal_to_close = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)

    if not signal_to_close:
        return jsonify({"success": False, "message": "لم يتم العثور على الصفقة."}), 404

    symbol = signal_to_close['symbol']
    with live_prices_lock: current_price = live_prices.get(symbol)

    if not current_price:
        return jsonify({"success": False, "message": "لا يمكن الحصول على السعر الحالي للإغلاق."}), 500

    try:
        close_signal(signal_to_close, current_price, "MANUAL_CLOSE")
        return jsonify({"success": True, "message": f"تم إرسال أمر إغلاق لصفقة {symbol} بنجاح."})
    except Exception as e:
        logger.error(f"❌ [Manual Close] Error closing signal {signal_id}: {e}", exc_info=True)
        return jsonify({"success": False, "message": "حدث خطأ أثناء إغلاق الصفقة."}), 500

# --- Main Loop & Threads ---
def main_bot_loop():
    logger.info("🚀 [Main Loop] Starting signal scanning loop...")
    while True:
        try:
            with trading_status_lock:
                if not is_trading_enabled: time.sleep(10); continue
            with signal_cache_lock:
                if len(open_signals_cache) >= 3: time.sleep(120); continue
            
            for symbol in validated_symbols_to_scan:
                with signal_cache_lock:
                    if symbol in open_signals_cache: continue
                
                df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                if df is None or len(df) < 50: continue
                
                df_featured = calculate_all_features(df); df_featured.name = symbol
                
                strategy_found = None
                if USE_BB_STOCH_STRATEGY and check_bb_stoch_strategy_enhanced(df_featured): strategy_found = "BB_Stoch_Strategy"
                elif USE_MACD_EMA_STRATEGY and check_macd_ema_strategy_enhanced(df_featured): strategy_found = "MACD_EMA_Strategy"
                elif USE_EMA_RSI_STRATEGY and check_ema_rsi_strategy_enhanced(df_featured): strategy_found = "EMA_RSI_Strategy"
                elif USE_PULLBACK_STRATEGY and check_pullback_strategy_enhanced(df_featured): strategy_found = "Pullback_Strategy"
                elif USE_MOMENTUM_VOLATILITY_STRATEGY and check_momentum_volatility_strategy(df_featured): strategy_found = "Momentum_Volatility_Strategy"
                
                if strategy_found:
                    logger.info(f"🌟 [Signal Found] for {symbol}! Strategy: {strategy_found}")
                    create_trade_signal(symbol, df_featured, strategy_found)
            
            time.sleep(60 * 5)
        except Exception as e:
            logger.error(f"❌ [Main Loop] A critical error occurred: {e}", exc_info=True)
            time.sleep(60)

def update_signal_in_db(signal_id, updates):
    # [FIX] Use the new check_db_connection function
    if not check_db_connection(): return False
    try:
        with conn.cursor() as cur:
            set_clause = sql.SQL(', ').join(sql.SQL("{} = %s").format(sql.Identifier(k)) for k in updates.keys())
            values = list(updates.values())
            query = sql.SQL("UPDATE signals SET {} WHERE id = %s").format(set_clause)
            values.append(signal_id)
            cur.execute(query, values)
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"❌ [DB] Failed to update signal {signal_id}: {e}")
        if conn: conn.rollback()
        return False

def close_signal(signal: Dict, closing_price: float, reason: str):
    symbol, signal_id, entry_price = signal['symbol'], signal['id'], signal['entry_price']
    
    if signal.get('is_real_trade'):
        try:
            asset = symbol.replace("USDT", "")
            balance = client.get_asset_balance(asset=asset)
            quantity_to_sell = float(balance['free'])
            
            if quantity_to_sell > 0:
                symbol_info = exchange_info_map.get(symbol)
                step_size = next((f['stepSize'] for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), '0.000001')
                adjusted_quantity = adjust_quantity_to_step_size(quantity_to_sell, step_size)

                if adjusted_quantity > 0:
                    client.create_order(symbol=symbol, side=Client.SIDE_SELL, type=Client.ORDER_TYPE_MARKET, quantity=adjusted_quantity)
        except Exception as e:
            logger.error(f"❌ [Real Close] CRITICAL ERROR closing real trade for {symbol}: {e}", exc_info=True)

    profit = ((closing_price - entry_price) / entry_price) * 100
    update_signal_in_db(signal_id, {"status": "closed", "closing_price": closing_price, "closed_at": datetime.now(timezone.utc), "profit_percentage": profit, "closing_reason": reason})
    
    log_and_notify("info", f"Closed trade for {symbol}. Profit: {profit:.2f}%", "TRADE_CLOSED")
    
    with signal_cache_lock:
        if symbol in open_signals_cache: del open_signals_cache[symbol]

def trade_management_loop():
    logger.info("🚀 [Trade Manager] Starting advanced trade management loop...")
    while True:
        try:
            with signal_cache_lock:
                if not open_signals_cache: time.sleep(2); continue
                signals_to_monitor = list(open_signals_cache.values())

            for signal in signals_to_monitor:
                symbol = signal['symbol']
                with live_prices_lock: current_price = live_prices.get(symbol)
                if not current_price: continue

                signal_details = signal.get('signal_details', {})
                stop_loss = signal['stop_loss']
                entry_price = signal['entry_price']

                if current_price <= stop_loss:
                    reason = "TRAILING_SL_HIT" if signal_details.get('is_trailing_active') else "SL_HIT"
                    close_signal(signal, stop_loss, reason)
                    continue

                if not signal_details.get('is_trailing_active'):
                    profit_percent = ((current_price - entry_price) / entry_price) * 100
                    if profit_percent >= TRAILING_STOP_ACTIVATION_PROFIT_PERCENT:
                        new_stop_loss = entry_price
                        signal_details['is_trailing_active'] = True
                        
                        updates = {"stop_loss": new_stop_loss, "status": "updated", "signal_details": json.dumps(signal_details, cls=NpEncoder)}
                        if update_signal_in_db(signal['id'], updates):
                            signal.update({"stop_loss": new_stop_loss, "status": "updated", "signal_details": signal_details})
                            log_and_notify("info", f"Trailing stop activated for {symbol}. New SL at entry: {new_stop_loss:.4f}", "TRAIL_ACTIVATED")
                        continue

                if signal_details.get('is_trailing_active'):
                    trailing_distance = signal_details.get('trailing_stop_distance', 0)
                    if trailing_distance > 0:
                        potential_new_sl = current_price - trailing_distance
                        if potential_new_sl > stop_loss:
                            if update_signal_in_db(signal['id'], {"stop_loss": potential_new_sl}):
                                signal['stop_loss'] = potential_new_sl
            time.sleep(1)
        except Exception as e:
            logger.error(f"❌ [Trade Manager] A critical error occurred: {e}", exc_info=True)
            time.sleep(10)

def update_market_state_loop():
    logger.info("🚀 [Market State] Starting market state update loop...")
    while True:
        try:
            trend_details = {}
            for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
                btc_df = fetch_historical_data(BTC_SYMBOL, tf, 30)
                if btc_df is None or btc_df.empty: continue
                btc_df_featured = calculate_all_features(btc_df)
                last = btc_df_featured.iloc[-1]
                rsi_value = last['rsi']
                trend = "Sideways"
                if rsi_value > 55: trend = "Bullish"
                elif rsi_value < 45: trend = "Bearish"
                trend_details[tf] = {"trend": trend, "rsi": round(rsi_value, 2)}

            with market_state_lock:
                current_market_state['trend_details_by_tf'] = trend_details
        except Exception as e:
            logger.error(f"❌ [Market State] Error: {e}", exc_info=True)
        time.sleep(60 * 5)

def update_balance_loop():
    logger.info("🚀 [Balance Updater] Starting balance update loop...")
    while True:
        try:
            balance_info = client.get_asset_balance(asset='USDT')
            with balance_lock:
                global usdt_balance
                usdt_balance = float(balance_info['free'])
        except Exception as e:
            logger.error(f"❌ [Balance] Could not update USDT balance: {e}")
        time.sleep(60 * 10) 

# --- نقطة بداية البرنامج ---
if __name__ == '__main__':
    logger.info("="*50 + "\n====== Starting Crypto Trading Bot V17.3.1 ======\n" + "="*50)
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
        logger.critical("❌ No valid symbols to scan. Exiting."); exit(1)
    
    load_open_signals_to_cache()
    
    start_websocket()
    Thread(target=main_bot_loop, daemon=True).start()
    Thread(target=trade_management_loop, daemon=True).start()
    Thread(target=update_market_state_loop, daemon=True).start()
    Thread(target=update_balance_loop, daemon=True).start() 
    
    logger.info("🌐 [Flask] Starting UI on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
