# ملف c4.py - نسخة V12 مع واجهة محدثة (رصيد + مصابيح اتجاه)
# تم التحديث بواسطة Gemini بناءً على طلب المستخدم
# --- تعديل: إعادة إضافة عرض الرصيد ومصابيح الاتجاه إلى لوحة التحكم ---
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
        logging.FileHandler('crypto_bot_v12_full_ui.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV12_FullUI')

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

# --- متغيرات عامة وإعدادات البوت (V12) ---
is_trading_enabled: bool = False
trading_status_lock = Lock()
RISK_PER_TRADE_PERCENT: float = 1.0
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V9_With_Microstructure'
MODEL_FOLDER: str = 'V9'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 90
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices_v12"
TRADING_FEE_PERCENT: float = 0.1
BTC_SYMBOL: str = 'BTCUSDT'
MAX_OPEN_TRADES: int = 4
BUY_CONFIDENCE_THRESHOLD = 0.55
PARTIAL_TP_RR_RATIO: float = 1.0
FINAL_TP_RR_RATIO: float = 2.0

# --- إعدادات المؤشرات الفنية (V12) ---
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
ATR_PERIOD_FOR_SL: int = 14

# --- إعدادات فلتر دفتر الطلبات (V12) ---
ORDER_BOOK_DEPTH_LIMIT: int = 100
ORDER_BOOK_ANALYSIS_RANGE_PCT: float = 0.005
MIN_BID_ASK_STRENGTH_RATIO: float = 1.3

# --- متغيرات الحالة والكاش ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
exchange_info_map: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
rejection_logs_cache = deque(maxlen=100)
rejection_logs_lock = Lock()
current_market_state: Dict[str, Any] = {"overall_regime": "INITIALIZING", "volatility_regime": "normal", "trend_details_by_tf": {}, "last_updated": None}
market_state_lock = Lock()
last_market_state_check = 0

# --- قاموس أسباب الرفض باللغة العربية (مبسط) ---
REJECTION_REASONS_AR = {
    "Order Book Strength": "ضعف في دفتر الطلبات (Bids < 1.3 * Asks)",
    "Invalid ATR for TP/SL": "ATR غير صالح لحساب الأهداف",
    "Invalid Position Size": "حجم الصفقة غير صالح",
    "Lot Size Adjustment Failed": "فشل ضبط حجم العقد",
    "Min Notional Filter": "قيمة الصفقة أقل من الحد الأدنى",
    "Insufficient Balance": "الرصيد غير كافٍ",
    "Order Book Fetch Failed": "فشل جلب دفتر الطلبات",
}

# --- دالة إرسال رسائل تليجرام ---
def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
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
                        id SERIAL PRIMARY KEY, symbol TEXT NOT NULL, status TEXT DEFAULT 'open',
                        strategy_name TEXT, entry_price DOUBLE PRECISION NOT NULL,
                        stop_loss DOUBLE PRECISION NOT NULL, target_price DOUBLE PRECISION NOT NULL,
                        partial_tp_price DOUBLE PRECISION, closing_price DOUBLE PRECISION,
                        closed_at TIMESTAMP, profit_percentage DOUBLE PRECISION, closing_reason TEXT,
                        signal_details JSONB, is_real_trade BOOLEAN DEFAULT FALSE,
                        initial_quantity DOUBLE PRECISION, remaining_quantity DOUBLE PRECISION,
                        order_id TEXT, current_peak_price DOUBLE PRECISION
                    );
                """)
                alter_commands = [
                    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS partial_tp_price DOUBLE PRECISION;",
                    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS initial_quantity DOUBLE PRECISION;",
                    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS remaining_quantity DOUBLE PRECISION;"
                ]
                for command in alter_commands: cur.execute(command)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals (status);")
            conn.commit()
            logger.info("✅ [DB] الاتصال بقاعدة البيانات وتحديث المخطط بنجاح (V12).")
            return
        except Exception as e:
            logger.error(f"❌ [DB] خطأ أثناء التهيئة (محاولة {attempt + 1}/{retries}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: logger.critical("❌ [DB] فشل الاتصال بقاعدة البيانات.")

def check_db_connection() -> bool:
    global conn
    if conn is None or conn.closed != 0:
        init_db()
    try:
        if conn and conn.closed == 0:
            with conn.cursor() as cur: cur.execute("SELECT 1;")
            return True
        return False
    except (OperationalError, InterfaceError):
        try:
            init_db()
            return conn is not None and conn.closed == 0
        except Exception as retry_e:
            logger.error(f"❌ [DB] فشل إعادة الاتصال: {retry_e}")
            return False

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
        logger.info(f"✅ [Validation] تم العثور على {len(validated)} عملة صالحة للتداول من ملفك.")
        return validated
    except Exception as e:
        logger.error(f"❌ [Validation] خطأ أثناء التحقق من العملات: {e}", exc_info=True)
        return []

# --- دوال جلب البيانات وحساب الميزات (V12) ---
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        start_dt = datetime.now(timezone.utc) - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(klines, columns=cols)
        numeric_cols = {'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'}
        df = df.astype(numeric_cols)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna()
    except Exception as e:
        logger.error(f"❌ [Data] خطأ في جلب البيانات التاريخية لـ {symbol}: {e}")
        return None

def get_dynamic_ema_periods(df: pd.DataFrame) -> Tuple[int, int]:
    if 'atr' not in df.columns or df.empty: return (9, 21)
    relative_volatility = (df['atr'].iloc[-1] / df['close'].iloc[-1]) * 100
    if relative_volatility > 2.5: return (7, 18)
    elif relative_volatility > 1.5: return (8, 20)
    elif relative_volatility < 0.8: return (10, 24)
    else: return (9, 21)

def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD_FOR_SL, adjust=False).mean()
    
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))

    up_move = df_calc['high'].diff()
    down_move = -df_calc['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    
    fast_ema_period, slow_ema_period = get_dynamic_ema_periods(df_calc)
    df_calc['ema_fast'] = df_calc['close'].ewm(span=fast_ema_period, adjust=False).mean()
    df_calc['ema_slow'] = df_calc['close'].ewm(span=slow_ema_period, adjust=False).mean()
    df_calc['dynamic_ema_fast_period'] = fast_ema_period
    df_calc['dynamic_ema_slow_period'] = slow_ema_period
    
    highest_high = df_calc['high'].rolling(window=14).max()
    lowest_low = df_calc['low'].rolling(window=14).min()
    df_calc['stoch_k'] = 100 * (df_calc['close'] - lowest_low) / (highest_high - lowest_low).replace(0, 1e-9)
    df_calc['stoch_d'] = df_calc['stoch_k'].rolling(3).mean()
    
    bb_period = 20
    df_calc['bb_middle'] = df_calc['close'].rolling(window=bb_period).mean()
    bb_std = df_calc['close'].rolling(window=bb_period).std()
    df_calc['bb_upper'] = df_calc['bb_middle'] + (bb_std * 2)
    df_calc['bb_lower'] = df_calc['bb_middle'] - (bb_std * 2)
    
    return df_calc.astype('float32', errors='ignore')

def load_open_signals_to_cache():
    if not check_db_connection() or not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status IN ('open', 'partially_closed');")
            open_signals = cur.fetchall()
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in open_signals: open_signals_cache[signal['symbol']] = dict(signal)
            logger.info(f"✅ [Loading] تم تحميل {len(open_signals)} صفقة مفتوحة أو مغلقة جزئيًا.")
    except Exception as e:
        logger.error(f"❌ [Loading] فشل تحميل الصفقات المفتوحة: {e}")

# --- NEW: دالة تحليل حالة السوق مع المصابيح ---
def determine_market_state_enhanced():
    global current_market_state, last_market_state_check
    if time.time() - last_market_state_check < 180: return
    
    logger.info("🧠 [Market State] تحديث حالة السوق...")
    try:
        trend_details = {}
        volatility_regime = "normal"
        
        # تحليل التقلب من البيتكوين
        btc_data_1h = fetch_historical_data(BTC_SYMBOL, '1h', 7)
        if btc_data_1h is not None and not btc_data_1h.empty:
            btc_data_1h = calculate_all_features(btc_data_1h)
            atr_percent = (btc_data_1h['atr'].iloc[-1] / btc_data_1h['close'].iloc[-1]) * 100
            if atr_percent < 0.8: volatility_regime = "low"
            elif atr_percent > 2.0: volatility_regime = "high"

        # تحليل الاتجاه للفريمات المختلفة
        for tf in TIMEFRAMES_FOR_TREND_LIGHTS:
            df = fetch_historical_data(BTC_SYMBOL, tf, 50) # بيانات كافية لـ EMA و ADX
            if df is not None and not df.empty:
                df = calculate_all_features(df)
                ema_fast = df['ema_fast'].iloc[-1]
                ema_slow = df['ema_slow'].iloc[-1]
                adx = df['adx'].iloc[-1]
                
                if ema_fast > ema_slow and adx > 22: trend = "Strong Uptrend"
                elif ema_fast > ema_slow: trend = "Uptrend"
                elif ema_fast < ema_slow and adx > 22: trend = "Strong Downtrend"
                elif ema_fast < ema_slow: trend = "Downtrend"
                else: trend = "Ranging"
                trend_details[tf] = {"trend": trend, "adx": float(adx)}
            else:
                trend_details[tf] = {"trend": "Uncertain", "adx": 0}
                
        trends = [d['trend'] for d in trend_details.values()]
        overall_regime = max(set(trends), key=trends.count) if trends else "Uncertain"
        
        with market_state_lock:
            current_market_state = {
                "overall_regime": overall_regime.upper().replace(" ", "_"),
                "volatility_regime": volatility_regime,
                "trend_details_by_tf": trend_details,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            last_market_state_check = time.time()
        logger.info(f"✅ [Market State] الحالة العامة: {overall_regime}, التقلب: {volatility_regime}")
    except Exception as e:
        logger.error(f"❌ [Market State] خطأ: {e}", exc_info=True)

# --- استراتيجية التداول والفلاتر (V12) ---
def check_ema_stoch_momentum_strategy(df: pd.DataFrame) -> bool:
    required_cols = ['ema_fast', 'ema_slow', 'stoch_k', 'stoch_d']
    if not all(col in df.columns for col in required_cols): return False
    if len(df) < 4: return False
    try:
        current, previous = df.iloc[-1], df.iloc[-2]
        ema_crossover = previous['ema_fast'] <= previous['ema_slow'] and current['ema_fast'] > current['ema_slow']
        stochastic_momentum = current['stoch_k'] > current['stoch_d']
        if ema_crossover and stochastic_momentum:
            logger.info(f"  -> [{df.name}] ✅ إشارة من EMA ديناميكي ({int(current['dynamic_ema_fast_period'])}/{int(current['dynamic_ema_slow_period'])})/Stoch.")
            return True
        return False
    except Exception: return False

def check_bb_stoch_reversal_strategy(df: pd.DataFrame) -> tuple[bool, str]:
    required_cols = ['low', 'close', 'open', 'high', 'bb_lower', 'stoch_k', 'stoch_d']
    if not all(col in df.columns for col in required_cols): return False, ""
    if len(df) < 3: return False, ""
    try:
        last_candle, prev_candle = df.iloc[-1], df.iloc[-2]
        stoch_k_cross_above_d = prev_candle['stoch_k'] <= prev_candle['stoch_d'] and last_candle['stoch_k'] > last_candle['stoch_d']
        stoch_below_15 = last_candle['stoch_k'] < 15 and last_candle['stoch_d'] < 15
        price_at_lower_band = last_candle['low'] <= last_candle['bb_lower']
        if stoch_k_cross_above_d and stoch_below_15 and price_at_lower_band:
            patterns = { "Hammer": is_hammer(last_candle), "Bullish Engulfing": is_bullish_engulfing(last_candle, prev_candle) }
            for name, found in patterns.items():
                if found:
                    logger.info(f"  -> [{df.name}] ✅ إشارة من BB/Stoch Reversal. (Pattern: {name})")
                    return True, name
        return False, ""
    except Exception: return False, ""

def is_hammer(c): body = abs(c['close'] - c['open']); lw = (c['open'] if c['close'] > c['open'] else c['close']) - c['low']; uw = c['high'] - (c['close'] if c['close'] > c['open'] else c['open']); return body > 0 and lw > 2 * body and uw < body
def is_bullish_engulfing(cur, prev): return cur['close'] > cur['open'] and prev['open'] > prev['close'] and cur['close'] > prev['open'] and cur['open'] < prev['close']

def analyze_order_book(symbol: str, entry_price: float) -> Optional[Dict[str, Any]]:
    if not client: return None
    try:
        order_book = client.get_order_book(symbol=symbol, limit=ORDER_BOOK_DEPTH_LIMIT)
        bids = pd.DataFrame(order_book['bids'], columns=['price', 'qty'], dtype=float)
        asks = pd.DataFrame(order_book['asks'], columns=['price', 'qty'], dtype=float)
        price_range = entry_price * ORDER_BOOK_ANALYSIS_RANGE_PCT
        relevant_bids_vol = bids[bids['price'].between(entry_price - price_range, entry_price)]['qty'].sum()
        relevant_asks_vol = asks[asks['price'].between(entry_price, entry_price + price_range)]['qty'].sum()
        strength_ok = relevant_bids_vol > (relevant_asks_vol * MIN_BID_ASK_STRENGTH_RATIO)
        return { "passes_strength_check": strength_ok, "strength_ratio": relevant_bids_vol / relevant_asks_vol if relevant_asks_vol > 0 else float('inf') }
    except Exception as e:
        log_rejection(symbol, "Order Book Fetch Failed", {"error": str(e)}); return None

def passes_order_book_check(symbol: str, order_book_analysis: Dict) -> bool:
    if not order_book_analysis.get("passes_strength_check", False):
        log_rejection(symbol, "Order Book Strength", {"ratio": f"{order_book_analysis.get('strength_ratio', 0):.2f}"})
        return False
    logger.info(f"  -> [{symbol}] ✅ فلتر دفتر الطلبات: ناجح (Ratio: {order_book_analysis.get('strength_ratio', 0):.2f}).")
    return True

def calculate_dynamic_tp_sl(symbol: str, entry_price: float, df: pd.DataFrame, volatility_regime: str) -> Optional[Dict[str, Any]]:
    try:
        if df.empty or 'atr' not in df.columns or df['atr'].isnull().all():
            log_rejection(symbol, "Invalid ATR for TP/SL"); return None
        last_atr = df['atr'].iloc[-1]
        sl_multiplier = 1.5 if volatility_regime == 'high' else 1.2
        risk_per_coin = last_atr * sl_multiplier
        stop_loss_price = entry_price - risk_per_coin
        partial_tp_price = entry_price + risk_per_coin
        final_tp_price = entry_price + (risk_per_coin * FINAL_TP_RR_RATIO)
        return {
            'stop_loss': round(stop_loss_price, 6), 'partial_tp_price': round(partial_tp_price, 6),
            'target_price': round(final_tp_price, 6), 'source': f'DYNAMIC_ATR_{volatility_regime.upper()}'
        }
    except Exception as e:
        logger.error(f"❌ [{symbol}] Error in dynamic TP/SL: {e}", exc_info=True); return None

# --- دوال إدارة الصفقات (V12) ---
def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> Optional[Decimal]:
    try:
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info: return None
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_size_filter:
            step_size = Decimal(lot_size_filter['stepSize'])
            return (Decimal(str(quantity)) // step_size) * step_size
        return Decimal(str(quantity))
    except Exception: return None

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
        return order
    except Exception as e:
        logger.error(f"❌ [{symbol}] خطأ من باينانس عند تنفيذ الأمر: {e}")
        return None

def close_signal(signal_id: int, closing_price: float, reason: str) -> bool:
    with signal_cache_lock:
        signal_to_close = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)
        if not signal_to_close: return False
        symbol_to_close = signal_to_close['symbol']
        entry_price = float(signal_to_close['entry_price'])
        profit_percentage = ((closing_price - entry_price) / entry_price) * 100
        if signal_to_close.get('is_real_trade'):
            quantity_to_sell_dec = Decimal(str(signal_to_close['remaining_quantity']))
            if quantity_to_sell_dec > 0:
                sell_order = place_order(symbol_to_close, Client.SIDE_SELL, quantity_to_sell_dec)
                if not sell_order: return False
    if not check_db_connection() or not conn: return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE signals SET status = 'closed', closing_price = %s, closed_at = NOW(),
                profit_percentage = %s, closing_reason = %s, remaining_quantity = 0 WHERE id = %s;
            """, (closing_price, profit_percentage, reason, signal_id))
        conn.commit()
        with signal_cache_lock:
            if symbol_to_close in open_signals_cache: del open_signals_cache[symbol_to_close]
        emoji = "✅" if profit_percentage >= 0 else "🔻"
        send_telegram_message(f"{emoji} *إغلاق صفقة نهائي*\n\n*العملة:* `{symbol_to_close}`\n*السبب:* {reason}\n*الربح النهائي:* `{profit_percentage:.2f}%`")
        return True
    except Exception as e:
        logger.error(f"❌ [DB Close] فشل تحديث الصفقة المغلقة: {e}"); conn.rollback(); return False

def insert_signal_into_db(signal_data: Dict) -> Optional[Dict]:
    if not check_db_connection() or not conn: return None
    try:
        with conn.cursor() as cur:
            quantity = float(signal_data.get('quantity', 0))
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price, stop_loss, partial_tp_price, strategy_name, signal_details, is_real_trade, initial_quantity, remaining_quantity, order_id, current_peak_price)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *;
            """, (
                signal_data['symbol'], signal_data['entry_price'], signal_data['target_price'],
                signal_data['stop_loss'], signal_data['partial_tp_price'], signal_data['strategy_name'],
                json.dumps(signal_data['signal_details']), signal_data.get('is_real_trade', False),
                quantity, quantity, signal_data.get('order_id'), signal_data['entry_price']
            ))
            saved_signal = cur.fetchone()
            conn.commit()
            logger.info(f"💾 [{signal_data['symbol']}] تم حفظ الإشارة الجديدة في قاعدة البيانات (V12).")
            send_telegram_message(
                f"💡 *توصية شراء جديدة*\n\n"
                f"*العملة:* `{signal_data['symbol']}`\n*الاستراتيجية:* `{signal_data['strategy_name']}`\n"
                f"*سعر الدخول:* `{signal_data['entry_price']:.4f}`\n*وقف الخسارة (SL):* `{signal_data['stop_loss']:.4f}`\n"
                f"*هدف جزئي (TP1):* `{signal_data['partial_tp_price']:.4f}`\n*هدف نهائي (TP2):* `{signal_data['target_price']:.4f}`"
            )
            return dict(saved_signal)
    except Exception as e:
        logger.error(f"❌ [DB Insert] فشل إدراج الإشارة: {e}"); conn.rollback(); return None

# ---------------------- واجهة Flask (V12) ----------------------
app = Flask(__name__)
CORS(app)

def get_dashboard_html():
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم التداول V12</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
    <style>:root { --bg-main: #0D1117; --bg-card: #161B22; --border-color: #30363D; --text-primary: #E6EDF3; --text-secondary: #848D97; --accent-blue: #58A6FF; --accent-green: #3FB950; --accent-red: #F85149; --accent-yellow: #D29922; } body { font-family: 'Tajawal', sans-serif; background-color: var(--bg-main); color: var(--text-primary); } .card { background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 0.5rem; } .progress-container { background-color: #30363D; border-radius: 9999px; overflow: hidden; } .progress-bar { height: 100%; transition: width 0.5s ease-in-out; } .progress-bar-partial { background-color: var(--accent-yellow); z-index: 10; } .progress-bar-full { background-color: var(--accent-blue); } input:checked + .toggle-bg { background-color: var(--accent-green); }</style>
</head>
<body class="p-4 md:p-6">
    <div class="container mx-auto max-w-screen-2xl">
        <header class="mb-6 flex flex-wrap justify-between items-center gap-4">
            <h1 class="text-2xl md:text-3xl font-extrabold"><span class="text-accent-blue">لوحة تحكم</span><span class="text-text-secondary font-medium"> V12</span></h1>
            <div id="trend-lights-container" class="flex items-center gap-x-6 bg-black/20 px-4 py-2 rounded-lg border border-border-color"></div>
        </header>
        <section class="mb-6 grid grid-cols-1 md:grid-cols-3 gap-5">
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">الاتجاه العام</h3><div id="overall-regime" class="text-2xl font-bold text-center">...</div></div>
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">تقلب السوق</h3><div id="volatility-regime" class="text-2xl font-bold text-center">...</div></div>
            <div class="card p-4 flex flex-col justify-center items-center">
                <h3 class="font-bold text-lg text-text-secondary mb-2">التداول الحقيقي</h3>
                <div class="flex items-center space-x-3 space-x-reverse">
                    <span id="trading-status-text" class="font-bold text-lg"></span>
                    <label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="trading-toggle" class="sr-only" onchange="toggleTrading()"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label>
                </div>
                <div class="mt-2 text-xs text-text-secondary">الرصيد المتاح: <span id="usdt-balance" class="font-mono">...</span> USDT</div>
            </div>
        </section>
        <main>
            <div class="overflow-x-auto card p-0"><table class="min-w-full text-sm text-right"><thead class="border-b border-border-color bg-black/20"><tr><th class="p-4 font-semibold">العملة</th><th class="p-4 font-semibold">الحالة</th><th class="p-4 font-semibold">الربح/الخسارة</th><th class="p-4 font-semibold w-[30%]">التقدم للأهداف</th><th class="p-4 font-semibold">الكمية (المتبقية/الأصلية)</th><th class="p-4 font-semibold">إجراء</th></tr></thead><tbody id="signals-table"></tbody></table></div>
        </main>
    </div>
<script>
function toggleTrading() { fetch('/api/trading/toggle', { method: 'POST' }).then(() => updateStatus()); }
function manualClose(signalId, symbol) { if(confirm(`هل أنت متأكد من إغلاق صفقة ${symbol}؟`)) { fetch(`/api/signals/close/${signalId}`, { method: 'POST' }); } }
function updateStatus() {
    fetch('/api/market_status').then(r => r.json()).then(data => {
        if (!data) return;
        document.getElementById('overall-regime').textContent = (data.market_state?.overall_regime || 'UNCERTAIN').replace(/_/g, ' ');
        document.getElementById('volatility-regime').textContent = (data.market_state?.volatility_regime || 'normal').toUpperCase();
        
        const balanceEl = document.getElementById('usdt-balance');
        if (data.usdt_balance !== 'N/A' && data.usdt_balance !== null) {
            balanceEl.textContent = parseFloat(data.usdt_balance).toFixed(2);
        } else {
            balanceEl.textContent = 'غير متاح';
        }

        const lightsContainer = document.getElementById('trend-lights-container');
        lightsContainer.innerHTML = '';
        const timeframes = ['15m', '1h', '4h'];
        timeframes.forEach(tf => {
            const trendInfo = data.market_state?.trend_details_by_tf[tf];
            const trend = trendInfo?.trend || 'Uncertain';
            let lightClass = 'bg-gray-500';
            if (trend.includes('Uptrend')) { lightClass = 'bg-accent-green shadow-[0_0_8px_var(--accent-green)]'; }
            else if (trend.includes('Downtrend')) { lightClass = 'bg-accent-red shadow-[0_0_8px_var(--accent-red)]'; }
            else if (trend.includes('Ranging')) { lightClass = 'bg-accent-yellow shadow-[0_0_8px_var(--accent-yellow)]'; }
            lightsContainer.innerHTML += `<div class="flex items-center gap-2"><div class="w-3 h-3 rounded-full ${lightClass}"></div><span class="text-sm font-bold text-text-secondary">${tf}</span></div>`;
        });

        const tradeToggle = document.getElementById('trading-toggle'), tradeText = document.getElementById('trading-status-text');
        tradeToggle.checked = data.is_trading_enabled;
        tradeText.textContent = data.is_trading_enabled ? 'مُفعَّل' : 'غير مُفعَّل';
        tradeText.className = `font-bold text-lg ${data.is_trading_enabled ? 'text-accent-green' : 'text-accent-red'}`;
    });
}
function updateSignals() {
    fetch('/api/signals').then(r => r.json()).then(data => {
        if (!data) return;
        const tableBody = document.getElementById('signals-table');
        tableBody.innerHTML = '';
        data.forEach(s => {
            const profit = parseFloat(s.profit_percentage || 0);
            const pClass = profit > 0 ? 'text-accent-green' : profit < 0 ? 'text-accent-red' : 'text-text-secondary';
            const entry = parseFloat(s.entry_price), sl = parseFloat(s.stop_loss), current = parseFloat(s.current_price || entry);
            const partial_tp = parseFloat(s.partial_tp_price), final_tp = parseFloat(s.target_price);
            const progressToPartial = (partial_tp - sl > 0) ? Math.max(0, Math.min(100, (current - sl) / (partial_tp - sl) * 100)) : 0;
            const progressToFinal = (final_tp - partial_tp > 0) ? Math.max(0, Math.min(100, (current - partial_tp) / (final_tp - partial_tp) * 100)) : 0;
            let statusText, statusClass;
            if (s.status === 'open') { statusText = 'مفتوحة'; statusClass = 'bg-blue-500/20 text-blue-400'; }
            else if (s.status === 'partially_closed') { statusText = 'مغلقة جزئياً'; statusClass = 'bg-yellow-500/20 text-yellow-400'; }
            else { statusText = s.status; statusClass = 'bg-gray-500/20 text-gray-400'; }
            tableBody.innerHTML += `
                <tr class="border-b border-border-color hover:bg-white/5">
                    <td class="p-4 font-bold">${s.symbol}</td>
                    <td class="p-4"><span class="px-2 py-1 text-xs font-semibold rounded-full ${statusClass}">${statusText}</span></td>
                    <td class="p-4 font-mono ${pClass}">${profit.toFixed(2)}%</td>
                    <td class="p-4">
                        <div class="text-xs text-text-secondary mb-1">TP1: ${progressToPartial.toFixed(0)}% | TP2: ${s.status === 'partially_closed' ? progressToFinal.toFixed(0) + '%' : '...'}</div>
                        <div class="w-full progress-container h-2.5 relative">
                            <div class="progress-bar ${s.status === 'partially_closed' ? 'progress-bar-full' : 'progress-bar-partial'}" style="width: ${s.status === 'partially_closed' ? progressToFinal : progressToPartial}%"></div>
                        </div>
                    </td>
                    <td class="p-4 font-mono">${parseFloat(s.remaining_quantity || 0).toFixed(4)} / ${parseFloat(s.initial_quantity || 0).toFixed(4)}</td>
                    <td class="p-4"><button onclick="manualClose(${s.id}, '${s.symbol}')" class="bg-red-600 hover:bg-red-700 text-white font-bold py-1 px-3 rounded text-xs">إغلاق</button></td>
                </tr>`;
        });
    });
}
document.addEventListener('DOMContentLoaded', () => {
    updateStatus(); updateSignals();
    setInterval(updateStatus, 5000); setInterval(updateSignals, 7000);
});
</script>
</body></html>
"""

@app.route('/')
def home(): return render_template_string(get_dashboard_html())

@app.route('/api/market_status')
def get_market_status_api():
    with market_state_lock: state_copy = dict(current_market_state)
    with trading_status_lock: is_enabled = is_trading_enabled
    usdt_balance = None
    if client:
        try:
            usdt_balance = float(client.get_asset_balance(asset='USDT')['free'])
        except Exception as e:
            logger.warning(f"Could not fetch USDT balance: {e}")
            usdt_balance = 'N/A'
    return jsonify({"market_state": state_copy, "is_trading_enabled": is_enabled, "usdt_balance": usdt_balance})

@app.route('/api/signals')
def get_signals_api():
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

@app.route('/api/trading/toggle', methods=['POST'])
def toggle_trading_status():
    global is_trading_enabled
    with trading_status_lock:
        is_trading_enabled = not is_trading_enabled
        status_msg = "ENABLED" if is_trading_enabled else "DISABLED"
        send_telegram_message(f"🚨 Real trading status changed to: *{status_msg}*")
        return jsonify({"message": f"Trading status set to {status_msg}"})

@app.route('/api/signals/close/<int:signal_id>', methods=['POST'])
def manual_close_trade_endpoint(signal_id):
    if not redis_client or not client: return jsonify({"success": False, "message": "Services not ready"}), 503
    with signal_cache_lock:
        signal_to_close = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)
    if not signal_to_close: return jsonify({"success": False, "message": "Signal not found"}), 404
    try:
        current_price = float(redis_client.hget(REDIS_PRICES_HASH_NAME, signal_to_close['symbol']))
    except:
        current_price = float(client.get_symbol_ticker(symbol=signal_to_close['symbol'])['price'])
    if close_signal(signal_id, current_price, 'manual'):
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "message": "Failed to close signal"}), 500

# ---------------------- حلقات النظام (V12) ----------------------
def execute_partial_close(signal: Dict) -> bool:
    symbol = signal['symbol']
    signal_id = signal['id']
    logger.info(f"💰 [PARTIAL TP] {symbol} hit 1:1 R/R target. Executing partial close.")
    with signal_cache_lock:
        if signal_id not in [s['id'] for s in open_signals_cache.values()] or open_signals_cache[symbol]['status'] != 'open':
            return False
        initial_quantity = Decimal(str(signal['initial_quantity']))
        quantity_to_sell = initial_quantity / Decimal('2')
        adjusted_qty_to_sell = adjust_quantity_to_lot_size(symbol, float(quantity_to_sell))
        if adjusted_qty_to_sell is None or adjusted_qty_to_sell <= 0: return False
        if signal.get('is_real_trade'):
            sell_order = place_order(symbol, Client.SIDE_SELL, adjusted_qty_to_sell)
            if not sell_order: return False
    if not check_db_connection() or not conn: return False
    try:
        new_remaining_qty = float(initial_quantity - adjusted_qty_to_sell)
        new_stop_loss = float(signal['entry_price'])
        with conn.cursor() as cur:
            cur.execute("UPDATE signals SET status = 'partially_closed', stop_loss = %s, remaining_quantity = %s WHERE id = %s;", (new_stop_loss, new_remaining_qty, signal_id))
        conn.commit()
        with signal_cache_lock:
            if symbol in open_signals_cache:
                open_signals_cache[symbol]['status'] = 'partially_closed'
                open_signals_cache[symbol]['stop_loss'] = new_stop_loss
                open_signals_cache[symbol]['remaining_quantity'] = new_remaining_qty
        send_telegram_message(f"💰 *جني ربح جزئي*\n\n*العملة:* `{symbol}`\n*الإجراء:* تم بيع 50% من الكمية ونقل وقف الخسارة إلى نقطة الدخول.")
        return True
    except Exception as e:
        logger.error(f"❌ [DB Partial Close] فشل تحديث الصفقة: {e}"); conn.rollback(); return False

def trade_management_loop():
    logger.info("✅ [Trade Manager] بدء حلقة إدارة الصفقات (V12)...")
    while True:
        try:
            with signal_cache_lock:
                if not open_signals_cache: time.sleep(5); continue
                signals_to_check = list(open_signals_cache.values())
            if not redis_client: time.sleep(5); continue
            current_prices = redis_client.hgetall(REDIS_PRICES_HASH_NAME)
            for signal in signals_to_check:
                current_price_str = current_prices.get(signal['symbol'])
                if not current_price_str: continue
                current_price = float(current_price_str)
                if signal['status'] == 'open' and signal.get('partial_tp_price') and current_price >= float(signal['partial_tp_price']):
                    execute_partial_close(signal); continue
                if current_price >= float(signal['target_price']):
                    close_signal(signal['id'], current_price, 'take_profit_final'); continue
                if current_price <= float(signal['stop_loss']):
                    close_signal(signal['id'], current_price, 'stop_loss'); continue
            time.sleep(2)
        except Exception as e:
            logger.error(f"❌ [Trade Manager] خطأ في حلقة الإدارة: {e}", exc_info=True); time.sleep(10)

def main_loop_enhanced():
    logger.info("[Main Loop] انتظار اكتمال التهيئة...")
    time.sleep(15)
    if not validated_symbols_to_scan:
        send_telegram_message("قائمة العملات للمسح فارغة. يتوقف البوت."); return
    send_telegram_message(f"✅ بدء حلقة المسح لـ {len(validated_symbols_to_scan)} عملة.")
    
    while True:
        try:
            logger.info("🔄 [Main Loop] بدء دورة مسح جديدة (V12)...")
            determine_market_state_enhanced()
            with market_state_lock: volatility_regime = current_market_state['volatility_regime']
            
            symbols_to_process = random.sample(validated_symbols_to_scan, len(validated_symbols_to_scan))
            for symbol in symbols_to_process:
                try:
                    with signal_cache_lock:
                        if symbol in open_signals_cache or len(open_signals_cache) >= MAX_OPEN_TRADES: continue
                    
                    df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, 90)
                    if df_15m is None or df_15m.empty: continue
                    
                    df_features = calculate_all_features(df_15m)
                    if df_features is None or df_features.empty: continue
                    df_features.name = symbol

                    strategy_signal_found, strategy_name, pattern_name = False, None, ""
                    if check_ema_stoch_momentum_strategy(df_features):
                        strategy_signal_found, strategy_name = True, "Dynamic_EMA_Stoch_V12"
                    else:
                        bb_stoch_signal, pattern_name = check_bb_stoch_reversal_strategy(df_features)
                        if bb_stoch_signal:
                            strategy_signal_found, strategy_name = True, "BB_Stoch_Reversal_V12"
                    
                    if not strategy_signal_found: continue
                    
                    entry_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                    
                    order_book_analysis = analyze_order_book(symbol, entry_price)
                    if not order_book_analysis or not passes_order_book_check(symbol, order_book_analysis): continue
                    
                    tp_sl_data = calculate_dynamic_tp_sl(symbol, entry_price, df_features, volatility_regime)
                    if not tp_sl_data: continue
                    
                    signal_details = { **order_book_analysis, **tp_sl_data }
                    if pattern_name: signal_details['Pattern'] = pattern_name

                    new_signal = { 'symbol': symbol, 'strategy_name': strategy_name, 'signal_details': signal_details, 'entry_price': entry_price, **tp_sl_data }

                    with trading_status_lock: is_enabled = is_trading_enabled
                    if is_enabled:
                        quantity = calculate_position_size(symbol, entry_price, new_signal['stop_loss'])
                        if quantity and quantity > 0:
                            order_result = place_order(symbol, Client.SIDE_BUY, quantity)
                            if order_result: new_signal.update({'is_real_trade': True, 'quantity': float(quantity), 'order_id': order_result['orderId']})
                            else: continue
                        else: continue

                    saved_signal = insert_signal_into_db(new_signal)
                    if saved_signal:
                        with signal_cache_lock: open_signals_cache[saved_signal['symbol']] = saved_signal

                except Exception as e:
                    logger.error(f"❌ [Processing Error] للعملة {symbol}: {e}", exc_info=True)
                finally:
                    time.sleep(0.5)
            
            gc.collect()
            logger.info("✅ [End of Cycle] انتهت دورة المسح الكاملة. الانتظار 60 ثانية...")
            time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            send_telegram_message("إيقاف البوت يدوياً."); break
        except Exception as main_err:
            send_telegram_message(f"خطأ حرج في الحلقة الرئيسية: {main_err}"); time.sleep(120)

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
    logger.info("🤖 [Bot Services] بدء التهيئة (V12)...")
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
        logger.info("✅ [Bot Services] تم بدء جميع الخدمات الخلفية بنجاح.")
        send_telegram_message("✅ *البوت قيد التشغيل الآن (V12)*")
    except Exception as e:
        send_telegram_message(f"حدث خطأ حرج أثناء التهيئة: {e}"); exit(1)

# ---------------------- نقطة الانطلاق ----------------------
if __name__ == "__main__":
    logger.info("🚀 إطلاق بوت التداول ولوحة التحكم (V12) 🚀")
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
