# ملف c4.py - نسخة نهائية مع لوحة تحكم كاملة واستراتيجية ديناميكية (StochRSI + Candlesticks + Volume Spike)
# تم التحديث بواسطة Gemini
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
        logging.FileHandler('crypto_bot_full_dynamic_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotFullDynamic')

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
SAFETY_NET_STOP_LOSS_PCT: float = 5.0
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 10
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices_full_dynamic"
STATS_TRADE_SIZE_USDT: float = 5.0
BTC_SYMBOL: str = 'BTCUSDT'
MAX_OPEN_TRADES: int = 4
SYMBOL_PROCESSING_BATCH_SIZE: int = 10

# --- إعدادات المؤشرات الفنية والاستراتيجية ---
RSI_PERIOD: int = 14
STOCH_RSI_PERIOD: int = 14
STOCH_RSI_K_PERIOD: int = 3
STOCH_RSI_D_PERIOD: int = 3
OVERSOLD_THRESHOLD: int = 30
OVERBOUGHT_THRESHOLD: int = 70
VOLUME_SPIKE_MULTIPLIER: float = 1.5 # حجم التداول الحالي يجب أن يكون 1.5 مرة أكبر من السابق

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
rejection_logs_cache = deque(maxlen=100) # تم إعادته لدعم الواجهة
rejection_logs_lock = Lock()
current_market_state: Dict[str, Any] = {"overall_regime": "INITIALIZING", "trend_details_by_tf": {}, "last_updated": None}
market_state_lock = Lock()
last_market_state_check = 0

# --- دالة إرسال رسائل تليجرام ---
def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload, timeout=10)
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
                        target_price DOUBLE PRECISION, stop_loss DOUBLE PRECISION,
                        status TEXT DEFAULT 'open', closing_price DOUBLE PRECISION, closed_at TIMESTAMP,
                        profit_percentage DOUBLE PRECISION, strategy_name TEXT, signal_details JSONB,
                        current_peak_price DOUBLE PRECISION, is_real_trade BOOLEAN DEFAULT FALSE,
                        quantity DOUBLE PRECISION, order_id TEXT, closing_reason TEXT
                    );
                """)
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
            logger.critical(f"❌ [Validation] ملف العملات '{filename}' غير موجود!")
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        if not raw_symbols:
            logger.warning(f"⚠️ [Validation] ملف العملات '{filename}' فارغ.")
            return []
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        if not exchange_info_map: get_exchange_info_map()
        active = {s for s, info in exchange_info_map.items() if info.get('quoteAsset') == 'USDT' and info.get('status') == 'TRADING'}
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ [Validation] تم العثور على {len(validated)} عملة صالحة للتداول.")
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
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(klines, columns=cols + [''] * (len(klines[0]) - len(cols)))
        df = df[cols]
        numeric_cols = {'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'}
        df = df.astype(numeric_cols)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna()
    except Exception as e:
        logger.error(f"❌ [Data] خطأ في جلب البيانات التاريخية لـ {symbol}: {e}")
        return None

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    # 1. RSI
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    rsi = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    df_calc['rsi'] = rsi
    # 2. Stochastic RSI
    stoch_rsi_val = (rsi - rsi.rolling(STOCH_RSI_PERIOD).min()) / \
                    (rsi.rolling(STOCH_RSI_PERIOD).max() - rsi.rolling(STOCH_RSI_PERIOD).min()).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(STOCH_RSI_K_PERIOD).mean() * 100
    df_calc['stoch_rsi_d'] = df_calc['stoch_rsi_k'].rolling(STOCH_RSI_D_PERIOD).mean()
    # 3. ADX (for market state analysis)
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    atr = tr.ewm(span=14, adjust=False).mean()
    up_move = df_calc['high'].diff()
    down_move = -df_calc['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr.replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr.replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=14, adjust=False).mean()

    return df_calc.dropna()

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

# --- مكتبة أنماط الشموع ---
def get_candlestick_pattern(df_slice: pd.DataFrame) -> Dict[str, str]:
    if len(df_slice) < 3: return {"type": "none", "name": "No Pattern"}
    c1, c2, c3 = df_slice.iloc[-1], df_slice.iloc[-2], df_slice.iloc[-3]
    body1 = abs(c1['close'] - c1['open'])
    # -- أنماط صاعدة --
    if c1['close'] > c1['open']:
        if body1 > 0 and (c1['open'] - c1['low']) > 2 * body1 and (c1['high'] - c1['close']) < body1: return {"type": "bullish", "name": "Hammer"}
        if c2['open'] > c2['close'] and c1['close'] > c2['open'] and c1['open'] < c2['close']: return {"type": "bullish", "name": "Bullish Engulfing"}
    if c2['open'] > c2['close'] and c1['close'] > c1['open'] and c1['open'] < c2['low'] and c1['close'] > (c2['open'] - abs(c2['close']-c2['open'])/2) and c1['close'] < c2['open']: return {"type": "bullish", "name": "Piercing Line"}
    body2, body3 = abs(c2['close']-c2['open']), abs(c3['close']-c3['open'])
    if c3['open'] > c3['close'] and body2 < body3 and c2['close'] < c3['close'] and c1['close'] > c1['open'] and c1['open'] > c2['close'] and c1['close'] > (c3['open'] - body3/2): return {"type": "bullish", "name": "Morning Star"}
    # -- أنماط هابطة --
    if c1['close'] < c1['open']:
        if body1 > 0 and (c1['high'] - c1['open']) > 2 * body1 and (c1['close'] - c1['low']) < body1: return {"type": "bearish", "name": "Shooting Star"}
        if c2['close'] > c2['open'] and c1['open'] > c2['close'] and c1['close'] < c2['open']: return {"type": "bearish", "name": "Bearish Engulfing"}
    if c3['close'] > c3['open'] and body2 < body3 and c2['close'] > c3['close'] and c1['close'] < c1['open'] and c1['open'] < c2['close'] and c1['close'] < (c3['open'] + body3/2): return {"type": "bearish", "name": "Evening Star"}
    if c2['close'] > c2['open'] and c1['close'] < c1['open'] and c1['open'] > c2['high'] and c1['close'] < (c2['close'] - abs(c2['close']-c2['open'])/2) and c1['close'] > c2['open']: return {"type": "bearish", "name": "Dark Cloud Cover"}
    return {"type": "none", "name": "No Pattern"}

# --- استراتيجية التداول ---
def check_entry_strategy(df: pd.DataFrame) -> Optional[str]:
    required_cols = ['stoch_rsi_k', 'stoch_rsi_d', 'volume', 'open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols) or len(df) < 3: return None
    try:
        last, prev = df.iloc[-1], df.iloc[-2]
        is_bullish_crossover = prev['stoch_rsi_k'] < prev['stoch_rsi_d'] and last['stoch_rsi_k'] > last['stoch_rsi_d']
        is_oversold = last['stoch_rsi_k'] < OVERSOLD_THRESHOLD and last['stoch_rsi_d'] < OVERSOLD_THRESHOLD
        volume_spike = last['volume'] > prev['volume'] * VOLUME_SPIKE_MULTIPLIER
        if is_bullish_crossover and is_oversold and volume_spike:
            pattern_info = get_candlestick_pattern(df.iloc[-3:])
            if pattern_info["type"] == "bullish":
                logger.info(f"  -> [{df.name}] ✅ إشارة شراء: StochRSI Cross + Oversold + Volume Spike + Bullish Pattern ({pattern_info['name']}).")
                return pattern_info['name']
        return None
    except Exception as e:
        logger.error(f"  -> [{df.name}] ❌ خطأ أثناء التحقق من استراتيجية الدخول: {e}")
        return None

def check_dynamic_exit_conditions(df: pd.DataFrame) -> Optional[str]:
    required_cols = ['stoch_rsi_k', 'stoch_rsi_d', 'open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols) or len(df) < 3: return None
    try:
        last, prev = df.iloc[-1], df.iloc[-2]
        is_bearish_crossover = prev['stoch_rsi_k'] > prev['stoch_rsi_d'] and last['stoch_rsi_k'] < last['stoch_rsi_d']
        is_overbought = last['stoch_rsi_k'] > OVERBOUGHT_THRESHOLD
        if is_bearish_crossover and is_overbought:
            pattern_info = get_candlestick_pattern(df.iloc[-3:])
            if pattern_info["type"] == "bearish":
                logger.info(f"  -> [{df.name}]  EXIT SIGNAL: StochRSI Cross + Overbought + Bearish Pattern ({pattern_info['name']}).")
                return pattern_info['name']
        return None
    except Exception as e:
        logger.error(f"  -> [{df.name}] ❌ خطأ أثناء التحقق من شروط الخروج: {e}")
        return None

# --- دوال إدارة الصفقات ---
def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> Optional[Decimal]:
    try:
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info: return None
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = Decimal(lot_size_filter['stepSize'])
            quantity_decimal = Decimal(str(quantity))
            return (quantity_decimal // step_size) * step_size
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
        if risk_per_coin <= 0: return None
        initial_quantity = risk_amount_usdt / risk_per_coin
        adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(initial_quantity))
        if adjusted_quantity is None or adjusted_quantity <= 0: return None
        notional_value = adjusted_quantity * Decimal(str(entry_price))
        symbol_info = exchange_info_map.get(symbol)
        if symbol_info:
            for f in symbol_info['filters']:
                if f['filterType'] in ('MIN_NOTIONAL', 'NOTIONAL'):
                    min_notional = Decimal(f.get('minNotional', f.get('notional', '0')))
                    if notional_value < min_notional: return None
        if notional_value > available_balance: return None
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
        signal_to_close = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)
        if not signal_to_close:
            logger.warning(f"⚠️ [Close] محاولة إغلاق صفقة غير موجودة في الكاش ID: {signal_id}")
            return False
        
        symbol_to_close = signal_to_close['symbol']
        entry_price = float(signal_to_close['entry_price'])
        profit_percentage = ((closing_price - entry_price) / entry_price) * 100

        if signal_to_close.get('is_real_trade'):
            # ... (منطق بيع الصفقة الحقيقية)
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
            
            reason_map = {'dynamic_exit': '🎯 خروج ديناميكي', 'safety_net_sl': '🛡️ وقف خسارة أمان', 'manual': '🖐️ إغلاق يدوي'}
            emoji = "✅" if profit_percentage >= 0 else "🔻"
            trade_type = "حقيقية" if signal_to_close.get('is_real_trade') else "تجريبية"
            telegram_message = (f"{emoji} *إغلاق صفقة {trade_type}*\n\n*العملة:* `{symbol_to_close}`\n*سبب الإغلاق:* {reason_map.get(reason, reason)}\n*سعر الدخول:* `{entry_price:.4f}`\n*سعر الإغلاق:* `{closing_price:.4f}`\n*الربح/الخسارة:* `{profit_percentage:.2f}%`")
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
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, is_real_trade, quantity, order_id, current_peak_price)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *;
            """, (
                signal_data['symbol'], signal_data['entry_price'], None,
                signal_data['stop_loss'], signal_data['strategy_name'],
                json.dumps(signal_data['signal_details']), signal_data.get('is_real_trade', False),
                signal_data.get('quantity'), signal_data.get('order_id'), signal_data['entry_price']
            ))
            saved_signal = cur.fetchone()
        conn.commit()
        logger.info(f"💾 [{signal_data['symbol']}] تم حفظ الإشارة الجديدة في قاعدة البيانات.")
        trade_type = "حقيقية" if signal_data.get('is_real_trade') else "تجريبية"
        telegram_message = (f"💡 *توصية شراء {trade_type} جديدة*\n\n*العملة:* `{signal_data['symbol']}`\n*الاستراتيجية:* `{signal_data['strategy_name']}`\n*نمط الدخول:* `{signal_data['signal_details']['entry_pattern']}`\n*سعر الدخول:* `{signal_data['entry_price']:.4f}`\n*وقف الأمان (SL):* `{signal_data['stop_loss']:.4f}`\n\n_الخروج سيكون ديناميكياً._")
        send_telegram_message(telegram_message)
        return dict(saved_signal)
    except Exception as e:
        logger.error(f"❌ [DB Insert] فشل إدراج الإشارة: {e}"); conn.rollback(); return None

# --- دوال النظام الرئيسية ---
def determine_market_state():
    global current_market_state, last_market_state_check
    if time.time() - last_market_state_check < 180: return
    logger.info("🧠 [Market State] تحديث حالة السوق...")
    try:
        trend_details = {}
        for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
            df = fetch_historical_data(BTC_SYMBOL, tf, 50)
            if df is not None and not df.empty:
                df_features = calculate_features(df)
                ema_fast = df_features['close'].ewm(span=12, adjust=False).mean().iloc[-1]
                ema_slow = df_features['close'].ewm(span=26, adjust=False).mean().iloc[-1]
                adx = df_features['adx'].iloc[-1]
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
# ... (دوال الواجهة الكاملة من الإصدار القديم)
def get_dashboard_html():
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم التداول الديناميكية</title>
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
    </style>
</head>
<body class="p-4 md:p-6">
    <div class="container mx-auto max-w-screen-2xl">
        <header class="mb-6 flex flex-wrap justify-between items-center gap-4">
            <h1 class="text-2xl md:text-3xl font-extrabold"><span class="text-accent-blue">لوحة تحكم</span><span class="text-text-secondary font-medium"> Dynamic StochRSI</span></h1>
            <div id="trend-lights-container" class="flex items-center gap-x-6 bg-black/20 px-4 py-2 rounded-lg border border-border-color"></div>
        </header>
        <section class="mb-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">حالة السوق العامة</h3><div id="overall-regime" class="text-2xl font-bold text-center">...</div></div>
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">صفقات مفتوحة</h3><div id="open-trades-count" class="text-2xl font-bold text-center">...</div></div>
            <div class="card p-4 flex flex-col justify-center items-center"><h3 class="font-bold text-lg text-text-secondary mb-2">التداول الحقيقي</h3><div class="flex items-center space-x-3 space-x-reverse"><span id="trading-status-text" class="font-bold text-lg"></span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="trading-toggle" class="sr-only" onchange="toggleTrading()"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div><div class="mt-2 text-xs text-text-secondary">رصيد USDT: <span id="usdt-balance" class="font-mono">...</span></div></div>
        </section>
        <div class="mb-4 border-b border-border-color"><nav class="flex space-x-6 space-x-reverse -mb-px"><button onclick="showTab('signals', this)" class="tab-btn active text-white py-3 px-1 font-semibold">الصفقات</button><button onclick="showTab('stats', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الإحصائيات</button><button onclick="showTab('notifications', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الإشعارات</button></nav></div>
        <main>
            <div id="signals-tab" class="tab-content"><div class="overflow-x-auto card p-0"><table class="min-w-full text-sm text-right"><thead class="border-b border-border-color bg-black/20"><tr><th class="p-4 font-semibold">العملة</th><th class="p-4 font-semibold">الحالة</th><th class="p-4 font-semibold">الربح/الخسارة</th><th class="p-4 font-semibold">الدخول/الحالي</th><th class="p-4 font-semibold">نمط الدخول</th><th class="p-4 font-semibold">إجراء</th></tr></thead><tbody id="signals-table"></tbody></table></div></div>
            <div id="stats-tab" class="tab-content hidden"><div id="stats-container" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"></div></div>
            <div id="notifications-tab" class="tab-content hidden"><div id="notifications-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-2"></div></div>
        </main>
    </div>
<script>
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
        document.getElementById('open-trades-count').textContent = data.open_trades_count;
        const tradeToggle = document.getElementById('trading-toggle'), tradeText = document.getElementById('trading-status-text');
        tradeToggle.checked = data.is_trading_enabled;
        tradeText.textContent = data.is_trading_enabled ? 'مُفعَّل' : 'غير مُفعَّل';
        tradeText.className = `font-bold text-lg ${data.is_trading_enabled ? 'text-accent-green' : 'text-accent-red'}`;
        document.getElementById('usdt-balance').textContent = data.usdt_balance ? parseFloat(data.usdt_balance).toFixed(2) : 'N/A';
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
            const entry = parseFloat(s.entry_price), current = parseFloat(s.current_price || entry);
            const entryPattern = s.signal_details?.entry_pattern || 'N/A';
            tableBody.innerHTML += `<tr class="border-b border-border-color hover:bg-white/5"><td class="p-4 font-bold">${s.symbol}</td><td class="p-4"><span class="px-2 py-1 text-xs font-semibold rounded-full ${s.is_real_trade ? 'bg-blue-500/20 text-blue-400' : 'bg-yellow-500/20 text-yellow-400'}">${s.is_real_trade ? 'حقيقي' : 'تجريبي'}</span></td><td class="p-4 font-mono ${pClass}">${profit.toFixed(2)}%</td><td class="p-4 font-mono">${current.toFixed(4)} / ${entry.toFixed(4)}</td><td class="p-4 text-text-secondary">${entryPattern}</td><td class="p-4"><button onclick="manualClose(${s.id}, '${s.symbol}')" class="bg-red-600 hover:bg-red-700 text-white font-bold py-1 px-3 rounded text-xs">إغلاق</button></td></tr>`;
        });
    });
}
function updateStats() {
    fetchData('/api/stats').then(data => {
        if (!data) return;
        const container = document.getElementById('stats-container');
        if (data.error) { container.innerHTML = `<div class="card p-4 text-center col-span-full text-accent-red">${data.error}</div>`; return; }
        container.innerHTML = `<div class="card p-4 text-center"><h4 class="text-text-secondary">صافي الربح</h4><div class="text-2xl font-bold ${data.net_profit_usdt >= 0 ? 'text-accent-green' : 'text-accent-red'}">${parseFloat(data.net_profit_usdt).toFixed(2)}</div></div><div class="card p-4 text-center"><h4 class="text-text-secondary">معدل الربح</h4><div class="text-2xl font-bold">${parseFloat(data.win_rate).toFixed(2)}%</div></div><div class="card p-4 text-center"><h4 class="text-text-secondary">عامل الربح</h4><div class="text-2xl font-bold">${data.profit_factor === 'Infinity' ? '∞' : parseFloat(data.profit_factor).toFixed(2)}</div></div><div class="card p-4 text-center"><h4 class="text-text-secondary">الصفقات المغلقة</h4><div class="text-2xl font-bold">${data.total_closed_trades}</div></div>`;
    });
}
function updateNotifications() {
    fetchData('/api/notifications').then(data => {
        if (!data) return;
        document.getElementById('notifications-list').innerHTML = data.map(n => `<div class="p-2 border-b border-border-color"><span class="font-mono text-xs text-text-secondary">${new Date(n.timestamp).toLocaleString('ar-EG')}</span>: ${n.message}</div>`).join('');
    });
}
function manualClose(signalId, symbol) { if (confirm(\`هل أنت متأكد من رغبتك في إغلاق الصفقة لـ \${symbol} يدوياً؟\`)) { fetch(\`/api/signals/close/\${signalId}\`, { method: 'POST' }).then(() => updateSignals()); } }
function toggleTrading() { fetch('/api/trading/toggle', { method: 'POST' }).then(() => updateMarketStatus()); }
document.addEventListener('DOMContentLoaded', () => {
    ['MarketStatus', 'Signals', 'Stats', 'Notifications'].forEach(f => window[\`update\${f}\`]());
    setInterval(updateMarketStatus, 5000); setInterval(updateSignals, 7000); setInterval(updateStats, 60000); setInterval(updateNotifications, 15000);
});
</script>
</body></html>
"""

@app.route('/')
def home(): return render_template_string(get_dashboard_html())

@app.route('/api/market_status')
def get_market_status():
    with market_state_lock: state_copy = dict(current_market_state)
    with trading_status_lock: is_enabled = is_trading_enabled
    with signal_cache_lock: open_count = len(open_signals_cache)
    usdt_balance = None
    if client:
        try: usdt_balance = float(client.get_asset_balance(asset='USDT')['free'])
        except: usdt_balance = 'N/A'
    return jsonify({
        "market_state": state_copy,
        "is_trading_enabled": is_enabled,
        "open_trades_count": open_count,
        "usdt_balance": usdt_balance
    })

@app.route('/api/stats')
def get_stats():
    if not check_db_connection() or not conn: return jsonify({"error": "DB connection failed"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT profit_percentage, is_real_trade, quantity, entry_price FROM signals WHERE status = 'closed';")
            closed_trades = cur.fetchall()
        if not closed_trades: return jsonify({"net_profit_usdt": 0, "win_rate": 0, "profit_factor": 0, "total_closed_trades": 0})
        total_net_profit_usdt = sum( ( (float(t['profit_percentage']) - (2*0.1)) / 100) * (float(t['quantity']) * float(t['entry_price']) if t.get('is_real_trade') and t.get('quantity') and t.get('entry_price') else STATS_TRADE_SIZE_USDT) for t in closed_trades )
        wins = [float(s['profit_percentage']) for s in closed_trades if float(s['profit_percentage']) > 0]
        losses = [float(s['profit_percentage']) for s in closed_trades if float(s['profit_percentage']) < 0]
        win_rate = (len(wins) / len(closed_trades) * 100) if closed_trades else 0.0
        total_loss = abs(sum(losses))
        profit_factor = sum(wins) / total_loss if total_loss > 0 else "Infinity"
        return jsonify({"net_profit_usdt": total_net_profit_usdt, "win_rate": win_rate, "profit_factor": profit_factor, "total_closed_trades": len(closed_trades)})
    except Exception as e:
        logger.error(f"❌ [API Stats] Error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/signals')
def get_signals():
    if not all([check_db_connection(), redis_client]): return jsonify([]), 500
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
        logger.error(f"❌ [API Signals] Error: {e}"); return jsonify([]), 500

@app.route('/api/notifications')
def get_notifications():
    with notifications_lock: return jsonify(list(notifications_cache))

@app.route('/api/trading/toggle', methods=['POST'])
def toggle_trading_status():
    global is_trading_enabled
    with trading_status_lock:
        is_trading_enabled = not is_trading_enabled
        status_msg = "ENABLED" if is_trading_enabled else "DISABLED"
        log_and_notify('warning', f"🚨 Real trading status changed to: {status_msg}", "TRADING_STATUS_CHANGE")
        return jsonify({"message": f"Trading status set to {status_msg}"})

@app.route('/api/signals/close/<int:signal_id>', methods=['POST'])
def manual_close_trade_endpoint(signal_id):
    if not redis_client or not client: return jsonify({"success": False, "message": "Services not ready"}), 503
    with signal_cache_lock: signal_to_close = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)
    if not signal_to_close: return jsonify({"success": False, "message": "Signal not found"}), 404
    try:
        current_price = float(redis_client.hget(REDIS_PRICES_HASH_NAME, signal_to_close['symbol']))
    except:
        try: current_price = float(client.get_symbol_ticker(symbol=signal_to_close['symbol'])['price'])
        except Exception as e: return jsonify({"success": False, "message": f"Could not fetch price: {e}"}), 500
    if close_signal(signal_id, current_price, 'manual'):
        return jsonify({"success": True, "message": "Signal closed."})
    else:
        return jsonify({"success": False, "message": "Failed to close signal."}), 500

# ---------------------- حلقات النظام ----------------------
def trade_management_loop():
    logger.info("✅ [Trade Manager] بدء حلقة إدارة الصفقات الديناميكية...")
    while True:
        try:
            with signal_cache_lock:
                if not open_signals_cache:
                    time.sleep(5); continue
                signals_to_check = list(open_signals_cache.values())
            if not redis_client:
                time.sleep(5); continue
            current_prices = redis_client.hgetall(REDIS_PRICES_HASH_NAME)
            for signal in signals_to_check:
                current_price_str = current_prices.get(signal['symbol'])
                if not current_price_str: continue
                current_price = float(current_price_str)
                signal_id, symbol, sl = signal['id'], signal['symbol'], float(signal['stop_loss'])
                df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, 20)
                if df is not None:
                    df_features = calculate_features(df)
                    df_features.name = symbol
                    exit_pattern = check_dynamic_exit_conditions(df_features)
                    if exit_pattern:
                        logger.info(f"🎯 [DYNAMIC EXIT] {symbol} at {current_price}. Reason: {exit_pattern}")
                        close_signal(signal_id, current_price, 'dynamic_exit')
                        continue
                if current_price <= sl:
                    logger.info(f"🛡️ [SAFETY NET SL HIT] {symbol} at {current_price}")
                    close_signal(signal_id, current_price, 'safety_net_sl')
                    continue
            time.sleep(5)
        except Exception as e:
            logger.error(f"❌ [Trade Manager] خطأ في حلقة الإدارة: {e}", exc_info=True)
            time.sleep(10)

def main_loop():
    logger.info("[Main Loop] انتظار اكتمال التهيئة...")
    time.sleep(10)
    if not validated_symbols_to_scan:
        log_and_notify("critical", "قائمة العملات للمسح فارغة.", "SYSTEM_ERROR")
        return
    log_and_notify("info", f"✅ بدء حلقة المسح لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")
    while True:
        try:
            logger.info("🔄 [Main Loop] بدء دورة مسح جديدة...")
            determine_market_state()
            symbols_to_process = random.sample(validated_symbols_to_scan, len(validated_symbols_to_scan))
            for symbol in symbols_to_process:
                try:
                    with signal_cache_lock:
                        if symbol in open_signals_cache or len(open_signals_cache) >= MAX_OPEN_TRADES: continue
                    df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, 20)
                    if df is None or len(df) < 20: continue
                    df_features = calculate_features(df)
                    df_features.name = symbol
                    entry_pattern = check_entry_strategy(df_features)
                    if entry_pattern:
                        logger.info(f"  -> [{symbol}] ✅ تم العثور على إشارة، المتابعة...")
                        try:
                            entry_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                        except Exception as e:
                            logger.error(f"❌ [{symbol}] فشل جلب سعر الدخول: {e}."); continue
                        stop_loss_price = entry_price * (1 - SAFETY_NET_STOP_LOSS_PCT / 100)
                        new_signal = {'symbol': symbol, 'strategy_name': "Dynamic StochRSI", 'signal_details': {'entry_pattern': entry_pattern, 'stoch_k': f"{df_features.iloc[-1]['stoch_rsi_k']:.2f}", 'stoch_d': f"{df_features.iloc[-1]['stoch_rsi_d']:.2f}"}, 'entry_price': entry_price, 'stop_loss': stop_loss_price}
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
                            with signal_cache_lock: open_signals_cache[saved_signal['symbol']] = saved_signal
                            log_and_notify('info', f"SIGNAL: New buy signal for {symbol}", "NEW_SIGNAL")
                except Exception as e:
                    logger.error(f"❌ [Processing Error] للعملة {symbol}: {e}", exc_info=True)
                finally:
                    time.sleep(1) 
            gc.collect()
            logger.info("✅ [End of Cycle] انتهت دورة المسح الكاملة. الانتظار 60 ثانية...")
            time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            log_and_notify("info", "إيقاف البوت.", "SYSTEM"); break
        except Exception as main_err:
            log_and_notify("error", f"خطأ حرج في الحلقة الرئيسية: {main_err}", "SYSTEM"); time.sleep(120)

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
        Thread(target=main_loop, daemon=True).start()
        Thread(target=price_update_loop, daemon=True).start()
        Thread(target=trade_management_loop, daemon=True).start()
        logger.info("✅ [Bot Services] تم بدء جميع الخدمات الخلفية بنجاح.")
        send_telegram_message("✅ *بوت StochRSI الديناميكي (مع لوحة تحكم) قيد التشغيل الآن*")
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حرج أثناء التهيئة: {e}", "SYSTEM"); exit(1)

# ---------------------- نقطة الانطلاق ----------------------
if __name__ == "__main__":
    logger.info("🚀 إطلاق بوت التداول الديناميكي (StochRSI + Candlesticks + Volume) 🚀")
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

