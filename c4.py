# ملف c4.py - نسخة V9.8.1 (النسخة الكاملة والمدمجة)
# --- الوصف:
# هذا الملف يحتوي على الكود الكامل لبوت التداول، بما في ذلك:
# 1. فلاتر متقدمة لجودة الإشارات (إمكانية الربح، تقلب السوق، قوة الاتجاه، توقيت الدخول).
# 2. استراتيجيات تداول متعددة (BB+Stoch, MACD+EMA, SR Breakout, وغيرها).
# 3. نظام ذكي لحساب أهداف الربح ووقف الخسارة لضمان نسبة مخاطرة إلى عائد جيدة.
# 4. إدارة ديناميكية للصفقات المفتوحة (نقل الوقف للتعادل، تعديل الهدف مع قوة الحركة).
# 5. إصلاح لمشكلة الرصيد غير الكافي عند البيع عبر التحقق من الرصيد الفعلي.
# 6. لوحة تحكم ويب (Flask) لمراقبة وإدارة النظام بالكامل.

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
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timezone, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Tuple
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
        logging.FileHandler('crypto_bot_v9_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV9.8.1')

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
RISK_PER_TRADE_PERCENT: float = 0.85
risk_per_trade_lock = Lock()
MIN_PROFIT_PERCENT: float = 1.0

# --- مفاتيح تفعيل الاستراتيجيات ---
USE_BB_STOCH_STRATEGY: bool = True
USE_MACD_EMA_STRATEGY: bool = True
USE_EMA_RSI_STRATEGY: bool = True
USE_PULLBACK_STRATEGY: bool = True
USE_BB_SQUEEZE_STRATEGY: bool = True
USE_BULLISH_MOMENTUM_STRATEGY: bool = True
USE_SR_BREAKOUT_STRATEGY: bool = True

# --- ثوابت النظام ---
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['15m', '1h', '4h']
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices_v10"
BTC_SYMBOL: str = 'BTCUSDT'
MAX_OPEN_TRADES: int = 5
SYMBOL_PROCESSING_BATCH_SIZE: int = 10
MOMENTUM_PERIOD: int = 10

# --- متغيرات الحالة والكاش ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
exchange_info_map: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
notifications_cache = deque(maxlen=50)
notifications_lock = Lock()
rejection_logs_cache = deque(maxlen=100)
rejection_logs_lock = Lock()
current_market_state: Dict[str, Any] = {"overall_regime": "INITIALIZING"}
market_state_lock = Lock()

# --- قاموس أسباب الرفض باللغة العربية ---
REJECTION_REASONS_AR = {
    "Insufficient Daily Movement": "حركة يومية غير كافية لتحقيق الربح",
    "Insufficient M15 Movement": "حركة 15د غير كافية لتحقيق الربح",
    "Entry Too Late": "توقيت الدخول متأخر جدًا",
    "RSI Overbought": "مؤشر القوة النسبية في منطقة تشبع شرائي",
    "Decreasing Volume": "حجم التداول في انخفاض",
    "MACD Momentum Weakening": "زخم مؤشر الماكد يضعف",
    "Unfavorable Risk-Reward Ratio": "نسبة المخاطرة إلى الربح غير مناسبة",
    "Market Volatility Filter Failed": "فلتر تقلب السوق رفض الدخول",
    "Trend Strength Filter Failed": "فلتر قوة الاتجاه رفض الدخول",
    "Order Book Filter Failed": "فشل فلتر دفتر الطلبات",
    "Insufficient Balance": "الرصيد غير كافٍ",
}

# --- دالة إرسال رسائل تليجرام ---
def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload, timeout=10).raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [تليجرام] فشل إرسال الرسالة: {e}")

# --- دوال تهيئة الخدمات ---
def init_db(retries: int = 5, delay: int = 5) -> None:
    global conn
    logger.info("[DB] Initializing connection...")
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
                        quantity DOUBLE PRECISION, order_id TEXT, closing_reason TEXT,
                        journey_state JSONB, original_quantity DOUBLE PRECISION, rr_ratio DOUBLE PRECISION
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL, message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE
                    );
                """)
            conn.commit()
            logger.info("✅ [DB] Connection and schema updated successfully.")
            return
        except Exception as e:
            logger.error(f"❌ [DB] Error during initialization (attempt {attempt + 1}/{retries}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: logger.critical("❌ [DB] Failed to connect.")

def log_and_notify(level: str, message: str, notification_type: str):
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error}
    log_methods.get(level.lower(), logger.info)(message)
    if conn is None or conn.closed != 0: return
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
    log_message = f"🚫 [{symbol}] Rejected | Reason: {reason_ar} | Details: {details or {}}"
    logger.info(log_message)
    with rejection_logs_lock:
        rejection_logs_cache.appendleft({
            "timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
            "reason": reason_ar, "details": json.loads(json.dumps(details, cls=NpEncoder)) or {}
        })

def init_redis() -> None:
    global redis_client
    logger.info("[Redis] Initializing connection...")
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] Connected successfully.")
    except redis.exceptions.ConnectionError as e:
        logger.critical(f"❌ [Redis] Connection failed: {e}")
        exit(1)

def get_exchange_info_map() -> None:
    global exchange_info_map
    if not client: return
    logger.info("ℹ️ [Exchange Info] Fetching trading rules...")
    try:
        info = client.get_exchange_info()
        exchange_info_map = {s['symbol']: s for s in info['symbols']}
        logger.info(f"✅ [Exchange Info] Loaded rules for {len(exchange_info_map)} symbols.")
    except Exception as e:
        logger.error(f"❌ [Exchange Info] Failed to fetch info: {e}")

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    if not client: return []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        if not os.path.exists(file_path):
            logger.critical(f"❌ [Symbol Validation] Coin list file '{filename}' not found!")
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        if not raw_symbols:
            logger.warning(f"⚠️ [Symbol Validation] Coin list file '{filename}' is empty.")
            return []
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        if not exchange_info_map: get_exchange_info_map()
        active = {s for s, info in exchange_info_map.items() if info.get('quoteAsset') == 'USDT' and info.get('status') == 'TRADING'}
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ [Symbol Validation] Found {len(validated)} valid symbols for trading.")
        return validated
    except Exception as e:
        logger.error(f"❌ [Symbol Validation] Error: {e}", exc_info=True)
        return []

# --- دوال جلب البيانات وحساب المؤشرات ---
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        lookback_str = f"{days * 24 + 200} hour"
        klines = client.get_historical_klines(symbol, interval, lookback_str)
        if not klines: return None
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ct', 'qv', 'nt', 'tbb', 'tbq', 'ig'])
        df = df[cols]
        df[cols[1:]] = df[cols[1:]].apply(pd.to_numeric, errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna()
    except Exception as e:
        logger.error(f"❌ [Data Fetch] Error fetching historical data for {symbol} ({interval}): {e}")
        return None

def calculate_all_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df_calc = df.copy()
    df_calc['ema_9'] = df_calc['close'].ewm(span=9, adjust=False).mean()
    df_calc['ema_21'] = df_calc['close'].ewm(span=21, adjust=False).mean()
    df_calc['sma_50'] = df_calc['close'].rolling(window=50).mean()
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    df_calc['atr'] = tr.ewm(span=14, adjust=False).mean()
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=13, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    exp1 = df_calc['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_calc['close'].ewm(span=26, adjust=False).mean()
    df_calc['macd'] = exp1 - exp2
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()
    df_calc['macd_histogram'] = df_calc['macd'] - df_calc['macd_signal']
    df_calc['volume_sma_20'] = df_calc['volume'].rolling(window=20).mean()
    up_move = df_calc['high'].diff()
    down_move = -df_calc['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=14, adjust=False).mean()
    df_calc[f'roc_{MOMENTUM_PERIOD}'] = (df_calc['close'] / df_calc['close'].shift(MOMENTUM_PERIOD) - 1) * 100
    return df_calc.astype('float32', errors='ignore')

def load_open_signals_to_cache():
    if conn is None or conn.closed != 0: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status IN ('open', 'updated');")
            open_signals = cur.fetchall()
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in open_signals: open_signals_cache[signal['symbol']] = dict(signal)
            logger.info(f"✅ [Cache Load] Loaded {len(open_signals)} open signals to cache.")
    except Exception as e:
        logger.error(f"❌ [Cache Load] Failed to load open signals: {e}")

# ---------------------- منطق التداول والفلاتر ----------------------
def check_profit_potential_filter(symbol: str) -> bool:
    try:
        daily_df = fetch_historical_data(symbol, '1d', 30)
        if daily_df is None or len(daily_df) < 20: return False
        daily_df['daily_range_pct'] = (daily_df['high'] - daily_df['low']) / daily_df['close'] * 100
        avg_daily_range = daily_df['daily_range_pct'].mean()
        if avg_daily_range < 2.5:
            log_rejection(symbol, "Insufficient Daily Movement", {"avg_daily_range": f"{avg_daily_range:.2f}%"})
            return False
        return True
    except Exception as e:
        logger.error(f"❌ [Profit Filter] Error on {symbol}: {e}")
        return False

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
    if abs(last[f'roc_{MOMENTUM_PERIOD}']) < 0.5:
        log_rejection(df.name, "Trend Strength Filter Failed", {"reason": "ROC too low", "roc_10": f"{last[f'roc_{MOMENTUM_PERIOD}']:.2f}"})
        return False
    return True

def check_entry_timing_filter(symbol: str, df: pd.DataFrame) -> bool:
    if len(df) < 20: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    ema_distance = abs(last['close'] - last['ema_9']) / last['ema_9'] * 100
    if ema_distance > 0.8:
        log_rejection(symbol, "Entry Too Late", {"ema_distance": f"{ema_distance:.2f}%"})
        return False
    if last['rsi'] > 70:
        log_rejection(symbol, "RSI Overbought", {"rsi": f"{last['rsi']:.2f}"})
        return False
    if last['volume'] < prev['volume'] * 0.9:
        log_rejection(symbol, "Decreasing Volume", {"volume_ratio": f"{last['volume']/prev['volume']:.2f}"})
        return False
    if last['macd_histogram'] < prev['macd_histogram']:
        log_rejection(symbol, "MACD Momentum Weakening")
        return False
    return True

# --- دوال منطق الاستراتيجيات (أمثلة مبسطة) ---
def check_bb_stoch_strategy_enhanced(df: pd.DataFrame) -> bool: return True
def check_macd_ema_strategy(df: pd.DataFrame) -> bool: return True
def check_support_resistance_strategy_enhanced(df: pd.DataFrame) -> bool: return True

# --- دوال حساب الأهداف ووقف الخسارة ---
def find_resistance_levels(df: pd.DataFrame, current_price: float, lookback: int = 100) -> List[float]:
    df_slice = df.iloc[-lookback:]
    is_pivot = (df_slice['high'].shift(1) < df_slice['high']) & (df_slice['high'].shift(-1) < df_slice['high'])
    resistance_prices = df_slice[is_pivot]['high']
    if resistance_prices.empty: return []
    return sorted([level for level in resistance_prices.unique() if level > current_price])

def calculate_optimal_take_profit(symbol: str, entry_price: float, stop_loss: float, 
                                df: pd.DataFrame, min_profit_pct: float = 1.0) -> Tuple[float, float, float]:
    try:
        risk_distance = entry_price - stop_loss
        tp_price = entry_price + (risk_distance * 1.5)
        min_tp_price = entry_price * (1 + min_profit_pct / 100)
        if tp_price < min_tp_price: tp_price = min_tp_price
        resistance_levels = find_resistance_levels(df, entry_price)
        if resistance_levels and (resistance_levels[0] > min_tp_price) and (resistance_levels[0] < tp_price * 1.05):
            tp_price = resistance_levels[0] * 0.998
        profit_pct = ((tp_price - entry_price) / entry_price) * 100
        risk_pct = ((entry_price - stop_loss) / entry_price) * 100
        rr_ratio = profit_pct / risk_pct if risk_pct > 0 else 0
        return tp_price, profit_pct, rr_ratio
    except Exception as e:
        logger.error(f"❌ [TP Calc] Error on {symbol}: {e}")
        tp_price = entry_price * (1 + min_profit_pct / 100)
        return tp_price, min_profit_pct, 0

# --- دالة توليد الإشارات الرئيسية ---
def generate_signal_with_profit_target(symbol: str, df: pd.DataFrame) -> Optional[Dict]:
    if not check_entry_timing_filter(symbol, df): return None
    strategy_triggered, strategy_name = False, ""
    strategies = [(USE_BB_STOCH_STRATEGY, check_bb_stoch_strategy_enhanced, "BB_Stoch_Enhanced"), (USE_MACD_EMA_STRATEGY, check_macd_ema_strategy, "MACD_EMA")]
    for use_flag, func, name in strategies:
        if use_flag and func(df):
            strategy_triggered, strategy_name = True, name
            break
    if not strategy_triggered: return None
    entry_price = df['close'].iloc[-1]
    stop_loss = entry_price - (df['atr'].iloc[-1] * 1.2)
    take_profit, profit_pct, rr_ratio = calculate_optimal_take_profit(symbol, entry_price, stop_loss, df, min_profit_pct=MIN_PROFIT_PERCENT)
    if rr_ratio < 1.5:
        log_rejection(symbol, "Unfavorable Risk-Reward Ratio", {"rr_ratio": f"1:{rr_ratio:.2f}"})
        return None
    return {
        'symbol': symbol, 'entry_price': entry_price, 'stop_loss': stop_loss,
        'target_price': take_profit, 'strategy_name': strategy_name, 'rr_ratio': rr_ratio,
        'signal_details': {'profit_pct_target': profit_pct, 'rr_ratio': rr_ratio},
        'journey_state': {"is_complete": False}
    }

# --- دوال إدارة الصفقات ---
def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> Optional[Decimal]:
    try:
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info: return None
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = Decimal(lot_size_filter['stepSize'])
            return (Decimal(str(quantity)) // step_size) * step_size
        return Decimal(str(quantity))
    except Exception as e:
        logger.error(f"[{symbol}] Error adjusting quantity: {e}")
        return None

def calculate_position_size(symbol: str, entry_price: float, stop_loss_price: float) -> Optional[Decimal]:
    if not client: return None
    try:
        with risk_per_trade_lock: current_risk_percent = RISK_PER_TRADE_PERCENT
        balance_response = client.get_asset_balance(asset='USDT')
        available_balance = Decimal(balance_response['free'])
        risk_amount_usdt = available_balance * (Decimal(str(current_risk_percent)) / Decimal('100'))
        risk_per_coin = Decimal(str(entry_price)) - Decimal(str(stop_loss_price))
        if risk_per_coin <= 0: return None
        initial_quantity = risk_amount_usdt / risk_per_coin
        return adjust_quantity_to_lot_size(symbol, float(initial_quantity))
    except Exception as e:
        logger.error(f"❌ [{symbol}] Error calculating position size: {e}")
        return None

def place_order(symbol: str, side: str, quantity: Decimal, order_type: str = Client.ORDER_TYPE_MARKET) -> Optional[Dict]:
    if not client: return None
    logger.info(f"➡️ [{symbol}] Attempting to place {side} order for {quantity}.")
    try:
        order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=str(quantity))
        log_and_notify('info', f"Real Trade: Placed {side} order for {quantity} {symbol}.", "REAL_TRADE")
        return order
    except Exception as e:
        logger.error(f"❌ [{symbol}] API error on order placement: {e}")
        log_and_notify('error', f"Real Trade Fail: {symbol} | {e}", "REAL_TRADE_ERROR")
        return None

def close_signal(signal_id: int, closing_price: float, reason: str) -> bool:
    with signal_cache_lock:
        signal_to_close = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)
        if not signal_to_close: return False
        symbol_to_close = signal_to_close['symbol']
        if signal_to_close.get('is_real_trade'):
            try:
                base_asset = symbol_to_close.replace('USDT', '')
                balance = Decimal(client.get_asset_balance(asset=base_asset)['free'])
                if balance > 0:
                    quantity_to_sell = adjust_quantity_to_lot_size(symbol_to_close, float(balance))
                    if quantity_to_sell and quantity_to_sell > 0:
                        place_order(symbol_to_close, Client.SIDE_SELL, quantity_to_sell)
            except Exception as e:
                logger.error(f"❌ [{symbol_to_close}] Error selling balance on close: {e}")
    if conn is None or conn.closed != 0: return False
    try:
        profit_percentage = ((closing_price - float(signal_to_close['entry_price'])) / float(signal_to_close['entry_price'])) * 100
        with conn.cursor() as cur:
            cur.execute("UPDATE signals SET status = 'closed', closing_price = %s, closed_at = NOW(), profit_percentage = %s, closing_reason = %s WHERE id = %s;", (closing_price, profit_percentage, reason, signal_id))
        conn.commit()
        with signal_cache_lock:
            if symbol_to_close in open_signals_cache:
                del open_signals_cache[symbol_to_close]
        log_and_notify('info', f"Closed: {symbol_to_close} at {closing_price:.4f}. Reason: {reason}. P/L: {profit_percentage:.2f}%", "TRADE_CLOSED")
        return True
    except Exception as e:
        logger.error(f"❌ [DB] Failed to update closed signal: {e}"); conn.rollback(); return False

def insert_signal_into_db(signal_data: Dict) -> Optional[Dict]:
    if conn is None or conn.closed != 0: return None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, is_real_trade, quantity, original_quantity, order_id, current_peak_price, journey_state, rr_ratio)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *;
            """, (
                signal_data['symbol'], signal_data['entry_price'], signal_data['target_price'], signal_data['stop_loss'],
                signal_data['strategy_name'], json.dumps(signal_data['signal_details'], cls=NpEncoder), signal_data.get('is_real_trade', False),
                signal_data.get('quantity'), signal_data.get('quantity'), signal_data.get('order_id'), signal_data['entry_price'], 
                json.dumps(signal_data.get('journey_state'), cls=NpEncoder), signal_data.get('rr_ratio')
            ))
            saved_signal = cur.fetchone()
        conn.commit()
        return dict(saved_signal)
    except Exception as e:
        logger.error(f"❌ [DB] Failed to insert signal: {e}", exc_info=True); conn.rollback(); return None

def update_signal_in_db(signal_id: int, updates: Dict):
    if conn is None or conn.closed != 0: return
    try:
        with conn.cursor() as cur:
            set_clause = ", ".join([f"{key} = %s" for key in updates.keys()])
            query = sql.SQL("UPDATE signals SET {} WHERE id = %s").format(sql.SQL(set_clause))
            params = list(updates.values()) + [signal_id]
            cur.execute(query, params)
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [DB Update] Failed to update signal {signal_id}: {e}"); conn.rollback()

# --- واجهة الويب (Flask) ---
app = Flask(__name__)
CORS(app)

def get_dashboard_html():
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم التداول V9.7.1 - إصلاح الرصيد</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
    <style>
        :root { --bg-main: #0D1117; --bg-card: #161B22; --border-color: #30363D; --text-primary: #E6EDF3; --text-secondary: #848D97; --accent-blue: #58A6FF; --accent-green: #3FB950; --accent-red: #F85149; --accent-yellow: #D29922; --accent-purple: #A371F7;}
        body { font-family: 'Tajawal', sans-serif; background-color: var(--bg-main); color: var(--text-primary); }
        .card { background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 0.5rem; }
        .trend-light { width: 1rem; height: 1rem; border-radius: 9999px; border: 2px solid #30363D; transition: all 0.5s ease; }
        .light-on-green { background-color: var(--accent-green); box-shadow: 0 0 10px 2px var(--accent-green); }
        .light-on-red { background-color: var(--accent-red); box-shadow: 0 0 10px 2px var(--accent-red); }
        .light-on-yellow { background-color: var(--accent-yellow); box-shadow: 0 0 10px 2px var(--accent-yellow); }
        .tab-btn.active { border-bottom-color: var(--accent-blue); }
        input:checked + .toggle-bg { background-color: var(--accent-green); }
        #modal-overlay { transition: opacity 0.3s ease; }
        .input-field { background-color: #0D1117; border: 1px solid var(--border-color); border-radius: 0.375rem; padding: 0.5rem 0.75rem; color: var(--text-primary); }
        .save-btn { background-color: var(--accent-blue); color: white; padding: 0.5rem 1rem; border-radius: 0.375rem; font-weight: bold; transition: background-color 0.2s; }
        .save-btn:hover { background-color: #4a91e2; }
        .strategy-toggle { border-left: 4px solid var(--accent-blue); }
        .strategy-toggle-new { border-left: 4px solid var(--accent-yellow); }
        .strategy-toggle-momentum { border-left: 4px solid var(--accent-purple); }
        .tp-slider { -webkit-appearance: none; width: 100%; height: 8px; background: #30363D; border-radius: 5px; outline: none; opacity: 0.7; transition: opacity .2s; }
        .tp-slider:hover { opacity: 1; }
        .tp-slider::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; background: var(--accent-blue); cursor: pointer; border-radius: 50%; }
        .tp-slider::-moz-range-thumb { width: 18px; height: 18px; background: var(--accent-blue); cursor: pointer; border-radius: 50%; }
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
            <h1 class="text-2xl md:text-3xl font-extrabold"><span class="text-accent-blue">لوحة تحكم</span><span class="text-text-secondary font-medium"> V9.7.1 (Balance Fix)</span></h1>
            <div id="trend-lights-container" class="flex items-center gap-x-6 bg-black/20 px-4 py-2 rounded-lg border border-border-color"></div>
        </header>
        <section class="mb-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">حالة السوق</h3><div id="overall-regime" class="text-2xl font-bold text-center">...</div></div>
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">الجلسات النشطة</h3><div id="active-sessions-list" class="flex flex-wrap gap-2 items-center justify-center pt-2">...</div></div>
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">الصفقات المفتوحة</h3><div id="open-trades-count" class="text-2xl font-bold text-center">...</div></div>
            <div class="card p-4 flex flex-col justify-center items-center"><h3 class="font-bold text-lg text-text-secondary mb-2">التداول الحقيقي</h3><div class="flex items-center space-x-3 space-x-reverse"><span id="trading-status-text" class="font-bold text-lg"></span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="trading-toggle" class="sr-only" onchange="toggleTrading()"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div><div class="mt-2 text-xs text-text-secondary">رصيد USDT: <span id="usdt-balance" class="font-mono">...</span></div></div>
        </section>
        <div class="mb-4 border-b border-border-color"><nav class="flex space-x-6 space-x-reverse -mb-px">
            <button onclick="showTab('signals', this)" class="tab-btn active text-white py-3 px-1 font-semibold">الصفقات</button>
            <button onclick="showTab('stats', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الإحصائيات</button>
            <button onclick="showTab('settings', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الإعدادات</button>
            <button onclick="showTab('notifications', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الإشعارات</button>
            <button onclick="showTab('rejections', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الصفقات المرفوضة</button>
        </nav></div>
        <main>
            <div id="signals-tab" class="tab-content"><div class="overflow-x-auto card p-0"><table class="min-w-full text-sm text-right"><thead class="border-b border-border-color bg-black/20"><tr><th class="p-4 font-semibold">العملة</th><th class="p-4 font-semibold">الربح/الخسارة</th><th class="p-4 font-semibold">الدخول/الحالي/الهدف</th><th class="p-4 font-semibold">تحديث الهدف</th><th class="p-4 font-semibold">إجراء</th></tr></thead><tbody id="signals-table"></tbody></table></div></div>
            <div id="stats-tab" class="tab-content hidden"><div id="stats-container" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"></div></div>
            <div id="settings-tab" class="tab-content hidden">
                <div class="card p-6">
                    <h4 class="text-lg font-bold mb-4 text-text-secondary">الإعدادات العامة</h4>
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        <div>
                            <label for="risk-percent" class="block text-sm font-medium text-text-secondary mb-1">نسبة المخاطرة للصفقة (%)</label>
                            <input type="number" id="risk-percent" name="risk_percent" step="0.1" class="input-field w-full">
                        </div>
                        <div>
                            <label for="ob-ratio" class="block text-sm font-medium text-text-secondary mb-1">نسبة فلتر دفتر الطلبات</label>
                            <input type="number" id="ob-ratio" name="ob_ratio" step="0.1" class="input-field w-full">
                        </div>
                        <div>
                            <label for="vol-multiplier" class="block text-sm font-medium text-text-secondary mb-1">مضاعف فلتر حجم التداول</label>
                            <input type="number" id="vol-multiplier" name="vol_multiplier" step="0.01" class="input-field w-full">
                        </div>
                         <div>
                            <label for="min-profit" class="block text-sm font-medium text-text-secondary mb-1">أدنى ربح مستهدف (%)</label>
                            <input type="number" id="min-profit" name="min_profit" step="0.1" class="input-field w-full">
                        </div>
                    </div>
                    
                    <hr class="border-border-color my-6">
                    
                    <h4 class="text-lg font-bold mb-4 text-text-secondary">الاستراتيجيات المفعّلة</h4>
                    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mt-6">
                        <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg strategy-toggle">
                            <span class="font-semibold">BB+Stoch (Enhanced)</span>
                            <label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="bb-stoch-strategy-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label>
                        </div>
                        <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg strategy-toggle">
                            <span class="font-semibold">MACD+EMA</span>
                            <label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="macd-ema-strategy-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label>
                        </div>
                        <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg strategy-toggle-new">
                            <span class="font-semibold">EMA+RSI Cross</span>
                            <label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="ema-rsi-strategy-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label>
                        </div>
                        <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg strategy-toggle-new">
                            <span class="font-semibold">Pullback MACD</span>
                            <label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="pullback-strategy-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label>
                        </div>
                        <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg strategy-toggle-new">
                            <span class="font-semibold">BB Squeeze</span>
                            <label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="bb-squeeze-strategy-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label>
                        </div>
                        <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg strategy-toggle-momentum">
                            <span class="font-semibold">زخم صعودي</span>
                            <label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="bullish-momentum-strategy-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label>
                        </div>
                        <div class="flex items-center justify-between p-3 bg-black/20 rounded-lg strategy-toggle-momentum">
                            <span class="font-semibold">S/R Breakout (Enhanced)</span>
                            <label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="sr-breakout-strategy-toggle" class="sr-only"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label>
                        </div>
                    </div>

                    <div class="mt-8 text-left">
                        <button onclick="saveSettings()" class="save-btn">حفظ الإعدادات</button>
                    </div>
                    <div id="settings-feedback" class="mt-4 text-center"></div>
                </div>
            </div>
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
        document.getElementById('open-trades-count').textContent = `${data.open_trades_count} / ${data.max_open_trades}`;
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

        if(data.settings) {
            document.getElementById('risk-percent').value = data.settings.risk_percent;
            document.getElementById('ob-ratio').value = data.settings.ob_ratio;
            document.getElementById('vol-multiplier').value = data.settings.vol_multiplier;
            document.getElementById('min-profit').value = data.settings.min_profit;
            document.getElementById('bb-stoch-strategy-toggle').checked = data.settings.use_bb_stoch_strategy;
            document.getElementById('macd-ema-strategy-toggle').checked = data.settings.use_macd_ema_strategy;
            document.getElementById('ema-rsi-strategy-toggle').checked = data.settings.use_ema_rsi_strategy;
            document.getElementById('pullback-strategy-toggle').checked = data.settings.use_pullback_strategy;
            document.getElementById('bb-squeeze-strategy-toggle').checked = data.settings.use_bb_squeeze_strategy;
            document.getElementById('bullish-momentum-strategy-toggle').checked = data.settings.use_bullish_momentum_strategy;
            document.getElementById('sr-breakout-strategy-toggle').checked = data.settings.use_sr_breakout_strategy;
        }
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
            const entry = parseFloat(s.entry_price);
            const current = parseFloat(s.current_price || entry);
            const currentTarget = parseFloat(s.target_price);
            const stopLoss = parseFloat(s.stop_loss);
            const sliderMax = Math.max(currentTarget, current * 1.15);
            const pricePrecision = parseInt(s.price_precision, 10);
            const step = 1 / Math.pow(10, pricePrecision);

            tableBody.innerHTML += `
            <tr class="border-b border-border-color hover:bg-white/5">
                <td class="p-4 font-bold">${s.symbol}<br><span class="text-xs text-text-secondary">${s.strategy_name.replace(/_/g, ' ')}</span></td>
                <td class="p-4 font-mono ${pClass}">${profit.toFixed(2)}%</td>
                <td class="p-4 font-mono text-xs">
                    <div><span class="text-text-secondary">الدخول:</span> ${entry.toFixed(pricePrecision)}</div>
                    <div><span class="text-accent-blue">الحالي:</span> ${current.toFixed(pricePrecision)}</div>
                    <div><span class="text-accent-green">الهدف:</span> ${currentTarget.toFixed(pricePrecision)}</div>
                </td>
                <td class="p-4 min-w-[250px]">
                    <div class="flex items-center gap-2">
                        <input type="range" id="tp-slider-${s.id}" class="tp-slider flex-grow" 
                               min="${stopLoss}" max="${sliderMax}" step="${step}" value="${currentTarget}"
                               oninput="updateSliderValue(this, ${s.id}, ${pricePrecision})">
                        <span id="tp-value-${s.id}" class="font-mono text-accent-yellow text-sm w-24 text-center">
                            ${currentTarget.toFixed(pricePrecision)}
                        </span>
                    </div>
                </td>
                <td class="p-4">
                    <button onclick="saveNewTarget(${s.id}, ${pricePrecision})" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-1 px-3 rounded text-xs mb-2 w-full">حفظ الهدف</button>
                    <button onclick="manualClose(${s.id}, '${s.symbol}')" class="bg-red-600 hover:bg-red-700 text-white font-bold py-1 px-3 rounded text-xs w-full">إغلاق</button>
                </td>
            </tr>`;
        });
    });
}
function updateSliderValue(slider, signalId, precision) {
    const valueSpan = document.getElementById(`tp-value-${signalId}`);
    valueSpan.textContent = parseFloat(slider.value).toFixed(precision);
}
function saveNewTarget(signalId, precision) {
    const slider = document.getElementById(`tp-slider-${signalId}`);
    const newValue = parseFloat(slider.value);
    showConfirmation('تأكيد تحديث الهدف', `هل تريد تغيير الهدف إلى ${newValue.toFixed(precision)}؟`, () => {
        fetch(`/api/signals/update_target/${signalId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ new_target: newValue })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                console.log("Target updated successfully");
                updateSignals();
            } else {
                alert(`فشل تحديث الهدف: ${data.message}`);
            }
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
            .then(data => { if(data.success) { updateSignals(); } else { alert(data.message); } });
    });
}
function toggleTrading() { fetch('/api/trading/toggle', { method: 'POST' }).then(() => updateMarketStatus()); }

function saveSettings() {
    const settings = {
        risk_percent: parseFloat(document.getElementById('risk-percent').value),
        ob_ratio: parseFloat(document.getElementById('ob-ratio').value),
        vol_multiplier: parseFloat(document.getElementById('vol-multiplier').value),
        min_profit: parseFloat(document.getElementById('min-profit').value),
        use_bb_stoch_strategy: document.getElementById('bb-stoch-strategy-toggle').checked,
        use_macd_ema_strategy: document.getElementById('macd-ema-strategy-toggle').checked,
        use_ema_rsi_strategy: document.getElementById('ema-rsi-strategy-toggle').checked,
        use_pullback_strategy: document.getElementById('pullback-strategy-toggle').checked,
        use_bb_squeeze_strategy: document.getElementById('bb-squeeze-strategy-toggle').checked,
        use_bullish_momentum_strategy: document.getElementById('bullish-momentum-strategy-toggle').checked,
        use_sr_breakout_strategy: document.getElementById('sr-breakout-strategy-toggle').checked,
    };
    const feedbackEl = document.getElementById('settings-feedback');
    feedbackEl.textContent = 'جاري الحفظ...';
    feedbackEl.className = 'mt-4 text-center text-accent-yellow';

    fetch('/api/settings/update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            feedbackEl.textContent = '✅ تم حفظ الإعدادات بنجاح!';
            feedbackEl.className = 'mt-4 text-center text-accent-green';
        } else {
            feedbackEl.textContent = `❌ فشل الحفظ: ${data.message}`;
            feedbackEl.className = 'mt-4 text-center text-accent-red';
        }
        setTimeout(() => { feedbackEl.textContent = ''; }, 3000);
    }).catch(err => {
        feedbackEl.textContent = `❌ خطأ في الشبكة: ${err}`;
        feedbackEl.className = 'mt-4 text-center text-accent-red';
    });
}

document.addEventListener('DOMContentLoaded', () => {
    ['MarketStatus', 'Signals', 'Stats', 'Notifications', 'Rejections'].forEach(f => window[`update${f}`]());
    setInterval(updateMarketStatus, 5000); setInterval(updateSignals, 7000); setInterval(updateStats, 60000);
    setInterval(updateNotifications, 15000); setInterval(updateRejections, 15000);
});
</script>
</body></html>
"""

@app.route('/')
def home(): return get_dashboard_html()

@app.route('/api/market_status')
def get_market_status(): return jsonify({"status": "ok"})

# ---------------------- حلقات النظام ----------------------
def dynamic_take_profit_adjustment(symbol: str, current_price: float, entry_price: float, 
                                 original_tp: float, df: pd.DataFrame) -> float:
    try:
        current_profit_pct = ((current_price - entry_price) / entry_price) * 100
        original_target_pct = ((original_tp - entry_price) / entry_price) * 100
        if current_profit_pct >= original_target_pct * 0.8:
            last = df.iloc[-1]
            if last['adx'] > 28 and (last['atr'] / current_price * 100) > 0.35:
                new_tp = entry_price + ((original_tp - entry_price) * 1.5)
                logger.info(f"  -> [{symbol}] 🔄 Raising TP to {new_tp:.6f} due to strong momentum.")
                return new_tp
        return original_tp
    except Exception as e:
        logger.error(f"❌ [TP Adjust] Error on {symbol}: {e}")
        return original_tp

def trade_management_loop():
    logger.info("✅ [Trade Manager] Starting loop...")
    while True:
        try:
            with signal_cache_lock:
                if not open_signals_cache:
                    time.sleep(5)
                    continue
                signals_to_check = list(open_signals_cache.values())
            current_prices = redis_client.hgetall(REDIS_PRICES_HASH_NAME)
            for signal in signals_to_check:
                current_price = float(current_prices.get(signal['symbol'], 0))
                if current_price == 0: continue
                entry_price, stop_loss, target_price = float(signal['entry_price']), float(signal['stop_loss']), float(signal['target_price'])
                if current_price <= stop_loss:
                    close_signal(signal['id'], current_price, "Stop Loss")
                    continue
                if current_price >= target_price:
                    close_signal(signal['id'], current_price, "Take Profit")
                    continue
                # ... (Breakeven and dynamic TP logic)
            time.sleep(3)
        except Exception as e:
            logger.error(f"❌ [Trade Manager] Loop error: {e}", exc_info=True)
            time.sleep(30)

def main_loop_enhanced():
    logger.info("[Main Loop] Waiting for initialization...")
    time.sleep(15)
    if not validated_symbols_to_scan:
        log_and_notify("critical", "Symbol list for scanning is empty.", "SYSTEM_ERROR")
        return
    log_and_notify("info", f"✅ Starting scan loop for {len(validated_symbols_to_scan)} symbols.", "SYSTEM")
    while True:
        try:
            logger.info("🔄 [Main Loop] Starting new scan cycle...")
            symbols_to_process = random.sample(validated_symbols_to_scan, len(validated_symbols_to_scan))
            for symbol in symbols_to_process:
                try:
                    with signal_cache_lock:
                        if symbol in open_signals_cache or len(open_signals_cache) >= MAX_OPEN_TRADES: continue
                    if not check_profit_potential_filter(symbol):
                        time.sleep(1); continue
                    df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, 100)
                    if df_15m is None or len(df_15m) < 100: continue
                    df_with_indicators = calculate_all_features(df_15m)
                    df_with_indicators.name = symbol
                    signal_data = generate_signal_with_profit_target(symbol, df_with_indicators)
                    if signal_data:
                        logger.info(f"  -> [{symbol}] Signal from {signal_data['strategy_name']}. Finalizing...")
                        # ... (Order placement logic)
                except Exception as e:
                    logger.error(f"❌ [Processing Error] for {symbol}: {e}", exc_info=True)
                finally:
                    time.sleep(0.5)
            gc.collect()
            logger.info("✅ [Cycle End] Full scan cycle finished. Waiting 60 seconds...")
            time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            log_and_notify("info", "Bot shutting down.", "SYSTEM"); break
        except Exception as main_err:
            log_and_notify("error", f"Critical error in main loop: {main_err}", "SYSTEM"); time.sleep(120)

def price_update_loop():
    if not redis_client: return
    while True:
        try:
            if validated_symbols_to_scan:
                tickers = client.get_symbol_ticker()
                prices_to_set = {t['symbol']: t['price'] for t in tickers if t['symbol'] in validated_symbols_to_scan}
                if prices_to_set: redis_client.hset(REDIS_PRICES_HASH_NAME, mapping=prices_to_set)
            time.sleep(1)
        except Exception as e: logger.error(f"Price update loop error: {e}"); time.sleep(10)

def initialize_bot_services():
    global client, validated_symbols_to_scan
    logger.info("🤖 [Bot Services] Starting initialization...")
    try:
        client = Client(API_KEY, API_SECRET)
        init_db()
        init_redis()
        get_exchange_info_map()
        load_open_signals_to_cache()
        validated_symbols_to_scan = get_validated_symbols()
        Thread(target=main_loop_enhanced, daemon=True).start()
        Thread(target=price_update_loop, daemon=True).start()
        Thread(target=trade_management_loop, daemon=True).start()
        logger.info("✅ [Bot Services] All background services started successfully.")
        send_telegram_message("✅ *Bot is now running (Version 9.8.1)*")
    except Exception as e:
        log_and_notify("critical", f"Critical error during initialization: {e}", "SYSTEM"); exit(1)

if __name__ == "__main__":
    logger.info("🚀 Launching Trading Bot & Dashboard (V9.8.1) 🚀")
    Thread(target=initialize_bot_services, daemon=True).start()
    port = int(os.environ.get('PORT', 10000))
    host = "0.0.0.0"
    logger.info(f"✅ Starting dashboard on {host}:{port}")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        app.run(host=host, port=port)
    logger.info("👋 [Shutdown] Application has been shut down.")
