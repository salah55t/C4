# ملف c4_v11.1_local_list.py - نسخة V11.1 "Local List"
# --- نسخة محدثة لتقرأ قائمة العملات من ملف محلي ---
# التحديث الرئيسي في هذا الإصدار:
# 1. تعديل دالة `fetch_and_validate_symbols` لتقوم بقراءة الرموز من ملف `crypto_list.txt`.
# 2. إضافة لاحقة "USDT" تلقائيًا لكل رمز.
# 3. التحقق من صلاحية كل رمز من الملف مقابل بيانات Binance الحية.

import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import redis
import traceback
import gc
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timezone, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Tuple
from functools import wraps

# --- إعدادات التجاهل واللوجر ---
warnings = __import__('warnings')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v11.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV11-Integrated')

# --- مشفر JSON مخصص ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, Decimal): return str(obj)
        if isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat()
        return super(NpEncoder, self).default(obj)

# --- تحميل متغيرات البيئة ---
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
    REDIS_URL: str = config('REDIS_URL', default='redis://localhost:6379/0')
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية. تأكد من وجود ملف .env: {e}")
    exit(1)

# --- متغيرات عامة وأقفال ---
is_trading_enabled: bool = False
trading_status_lock = Lock()

# --- المتغيرات القابلة للتعديل ---
RISK_PER_TRADE_PERCENT: float = 1.0
risk_per_trade_lock = Lock()

# --- مفاتيح تفعيل الاستراتيجيات ---
STRATEGY_CONFIG = {
    "BB_Stoch": {"enabled": True, "lock": Lock(), "display_name": "BB+Stoch"},
    "MACD_EMA": {"enabled": True, "lock": Lock(), "display_name": "MACD+EMA"},
    "SR_Breakout": {"enabled": True, "lock": Lock(), "display_name": "S/R Breakout"},
    "Triple_Confirmation": {"enabled": True, "lock": Lock(), "display_name": "Triple Confirmation"},
    "VWAP_Reversal": {"enabled": True, "lock": Lock(), "display_name": "VWAP Reversal"},
    "Fibonacci": {"enabled": True, "lock": Lock(), "display_name": "Fibonacci Levels"},
}

# --- إعدادات الفلاتر القابلة للتعديل ---
FILTER_CONFIG = {
    "ADX_THRESHOLD": {"value": 20, "lock": Lock(), "display_name": "حد مؤشر ADX"},
    "BB_STOCH_VOLUME_MULT": {"value": 1.1, "lock": Lock(), "display_name": "مضاعف فوليوم (BB Stoch)"},
    "SR_BREAKOUT_VOLUME_MULT": {"value": 1.3, "lock": Lock(), "display_name": "مضاعف فوليوم (SR Breakout)"},
    "TRIPLE_CONF_VOLUME_MULT": {"value": 1.1, "lock": Lock(), "display_name": "مضاعف فوليوم (Triple Conf)"},
    "VWAP_VOLUME_MULT": {"value": 1.2, "lock": Lock(), "display_name": "مضاعف فوليوم (VWAP Reversal)"},
    "FIB_VOLUME_MULT": {"value": 1.2, "lock": Lock(), "display_name": "مضاعف فوليوم (Fibonacci)"},
    "TRIPLE_CONF_MODE": {"value": "relaxed", "lock": Lock(), "display_name": "وضع (Triple Conf)"},
    "VWAP_REVERSAL_MODE": {"value": "relaxed", "lock": Lock(), "display_name": "وضع (VWAP Reversal)"},
}

# --- إعدادات المؤشرات ---
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 90
MAX_OPEN_TRADES: int = 5
ATR_PERIOD: int = 14
ADX_PERIOD: int = 14
BATCH_SIZE: int = 10
CACHE_EXPIRATION_SECONDS: int = 60 * 10 # 10 دقائق

# --- متغيرات الحالة والكاش ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
exchange_info_map: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
rejection_logs_cache: List[Dict] = []
rejection_logs_lock = Lock()
notifications_cache: List[Dict] = []
notifications_lock = Lock()
current_market_state: Dict[str, Any] = {"status": "INITIALIZING"}
market_state_lock = Lock()

# --- دوال مساعدة ---
def log_rejection(symbol: str, reason: str, details: Optional[Dict] = None):
    with rejection_logs_lock:
        rejection_logs_cache.insert(0, {"timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol, "reason": reason, "details": details or {}})
        if len(rejection_logs_cache) > 50: rejection_logs_cache.pop()

def log_and_notify(level: str, message: str, category: str):
    log_func = getattr(logger, level, logger.info)
    log_func(message)
    with notifications_lock:
        notifications_cache.insert(0, {"timestamp": datetime.now(timezone.utc).isoformat(), "level": level.upper(), "message": message, "category": category})
        if len(notifications_cache) > 50: notifications_cache.pop()

# --- دوال التهيئة ---
def initialize_database():
    global conn
    try:
        conn = psycopg2.connect(DB_URL)
        conn.autocommit = True
        log_and_notify("info", "✅ تم الاتصال بقاعدة البيانات بنجاح.", "INITIALIZATION")
    except OperationalError as e:
        log_and_notify("critical", f"❌ فشل الاتصال بقاعدة البيانات: {e}", "INITIALIZATION")
        conn = None

def initialize_redis():
    global redis_client
    try:
        redis_client = redis.from_url(REDIS_URL)
        redis_client.ping()
        log_and_notify("info", "✅ تم الاتصال بـ Redis بنجاح.", "INITIALIZATION")
    except redis.exceptions.ConnectionError as e:
        log_and_notify("critical", f"❌ فشل الاتصال بـ Redis: {e}", "INITIALIZATION")
        redis_client = None

def initialize_binance_client() -> Optional[Client]:
    try:
        c = Client(API_KEY, API_SECRET)
        c.ping()
        log_and_notify("info", "✅ تم الاتصال بـ Binance API بنجاح.", "INITIALIZATION")
        return c
    except (BinanceAPIException, BinanceRequestException, requests.exceptions.ConnectionError) as e:
        log_and_notify("critical", f"❌ فشل الاتصال بـ Binance API: {e}", "INITIALIZATION")
        return None

# --- الدالة المعدلة ---
def fetch_and_validate_symbols():
    """
    تقوم هذه الدالة بقراءة قائمة العملات من ملف `crypto_list.txt`،
    ثم تتحقق من صلاحيتها للتداول على Binance.
    """
    global exchange_info_map, validated_symbols_to_scan
    if not client: return

    # الخطوة 1: جلب جميع معلومات الصرف مرة واحدة كمرجع
    try:
        info = client.get_exchange_info()
        for s in info.get('symbols', []):
            exchange_info_map[s['symbol']] = s
        log_and_notify("info", f"🔍 تم جلب معلومات الصرف لـ {len(exchange_info_map)} رمز.", "INITIALIZATION")
    except Exception as e:
        log_and_notify("error", f"❌ فشل في جلب معلومات العملات من Binance: {e}", "INITIALIZATION")
        return

    # الخطوة 2: قراءة الملف المحلي والتحقق من الرموز
    validated_symbols = []
    try:
        with open('crypto_list.txt', 'r') as f:
            symbols_from_file = [line.strip().upper() for line in f if line.strip()]
        
        log_and_notify("info", f"📖 تم العثور على {len(symbols_from_file)} رمز في ملف crypto_list.txt.", "INITIALIZATION")

        for base_asset in symbols_from_file:
            symbol = f"{base_asset}USDT"
            
            # التحقق من وجود الرمز وصلاحيته
            if symbol in exchange_info_map:
                symbol_info = exchange_info_map[symbol]
                if (symbol_info['status'] == 'TRADING' and 
                    'SPOT' in symbol_info['permissions']):
                    validated_symbols.append(symbol)
                else:
                    logger.warning(f"⚠️ الرمز {symbol} من الملف غير صالح للتداول (الحالة: {symbol_info['status']}).")
            else:
                logger.warning(f"⚠️ الرمز {symbol} من الملف غير موجود في Binance.")

        validated_symbols_to_scan = validated_symbols
        log_and_notify("info", f"✅ تم التحقق من صلاحية {len(validated_symbols_to_scan)} عملة من الملف المحلي.", "INITIALIZATION")

    except FileNotFoundError:
        log_and_notify("critical", "❌ لم يتم العثور على ملف `crypto_list.txt`. يرجى إنشاء الملف ووضع رموز العملات فيه.", "INITIALIZATION")
    except Exception as e:
        log_and_notify("error", f"❌ حدث خطأ أثناء قراءة ملف العملات: {e}", "INITIALIZATION")


# --- دوال جلب البيانات وحساب المؤشرات ---
def get_data_for_symbol(symbol: str, timeframe: str, lookback_days: int) -> Optional[pd.DataFrame]:
    if not client or not redis_client: return None
    cache_key = f"klines:{symbol}:{timeframe}:{lookback_days}"
    
    try:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            df = pd.read_json(cached_data.decode('utf-8'))
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            return df
    except Exception as e:
        logger.warning(f"⚠️ فشل في قراءة الكاش لـ {symbol}: {e}")

    try:
        start_str = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, timeframe, start_str)
        
        if not klines: return None
        
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        
        redis_client.set(cache_key, df.to_json(), ex=CACHE_EXPIRATION_SECONDS)
        return df
    except (BinanceAPIException, BinanceRequestException) as e:
        logger.error(f"❌ خطأ API عند جلب بيانات {symbol}: {e}")
    except Exception as e:
        logger.error(f"❌ خطأ غير متوقع عند جلب بيانات {symbol}: {e}", exc_info=True)
    return None

def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    df_calc = df.copy()
    
    df_calc['ema_50'] = df_calc['close'].ewm(span=50, adjust=False).mean()
    df_calc['ema_200'] = df_calc['close'].ewm(span=200, adjust=False).mean()
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
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=14 - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=14 - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    bb_period = 20
    df_calc['bb_middle'] = df_calc['close'].rolling(window=bb_period).mean()
    bb_std = df_calc['close'].rolling(window=bb_period).std()
    df_calc['bb_upper'] = df_calc['bb_middle'] + (bb_std * 2)
    df_calc['bb_lower'] = df_calc['bb_middle'] - (bb_std * 2)
    df_calc['bb_width'] = (df_calc['bb_upper'] - df_calc['bb_lower']) / df_calc['bb_middle'].replace(0, 1e-9)
    rsi = df_calc['rsi']
    stoch_rsi_val = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min()).replace(0, 1e-9)
    df_calc['stoch_rsi_k'] = stoch_rsi_val.rolling(3).mean() * 100
    df_calc['stoch_rsi_d'] = df_calc['stoch_rsi_k'].rolling(3).mean()
    exp1 = df_calc['close'].ewm(span=12, adjust=False).mean()
    exp2 = df_calc['close'].ewm(span=26, adjust=False).mean()
    df_calc['macd'] = exp1 - exp2
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()
    q = df_calc['volume']
    p = (df_calc['high'] + df_calc['low'] + df_calc['close']) / 3
    df_calc['vwap'] = (p * q).cumsum() / q.cumsum()
    
    if len(df_calc) >= 50:
        last_50_candles = df_calc.iloc[-50:]
        high_point = last_50_candles['high'].max()
        low_point = last_50_candles['low'].min()
        df_calc['fib_high_point'] = high_point
        df_calc['fib_low_point'] = low_point

    return df_calc.dropna()

# --- دوال الاستراتيجيات الكاملة ---
def check_bb_stoch_strategy_enhanced(df: pd.DataFrame) -> bool:
    if len(df) < 2: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    adx_threshold = FILTER_CONFIG["ADX_THRESHOLD"]["value"]
    vol_mult = FILTER_CONFIG["BB_STOCH_VOLUME_MULT"]["value"]
    
    buy_condition = (last['stoch_rsi_k'] > last['stoch_rsi_d'] and prev['stoch_rsi_k'] <= prev['stoch_rsi_d'] and last['stoch_rsi_k'] < 30 and last['close'] < last['bb_lower'])
    if not buy_condition: return False

    if not (last['adx'] > adx_threshold):
        log_rejection(df.name, "BB_Stoch: ADX Filter Failed", {"adx": round(last['adx'],2)})
        return False
    if not (last['volume'] > (last['volume_sma_20'] * vol_mult)):
        log_rejection(df.name, "BB_Stoch: Volume Filter Failed")
        return False
    
    log_and_notify("debug", f"  -> [{df.name}] ✅ إشارة BB+Stoch.", "STRATEGY")
    return True

def check_macd_ema_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 2: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    
    trend_ok = last['ema_50'] > last['ema_200']
    crossover = last['macd'] > last['macd_signal'] and prev['macd'] <= prev['macd_signal']
    rsi_ok = last['rsi'] < 70

    if trend_ok and crossover and rsi_ok:
        log_and_notify("debug", f"  -> [{df.name}] ✅ إشارة MACD+EMA.", "STRATEGY")
        return True
    return False

def check_sr_breakout_strategy_enhanced(df: pd.DataFrame) -> bool:
    if len(df) < 50: return False
    last = df.iloc[-1]
    vol_mult = FILTER_CONFIG["SR_BREAKOUT_VOLUME_MULT"]["value"]
    
    resistance_level = df['high'].iloc[-51:-1].max()
    breakout = last['close'] > resistance_level
    volume_ok = last['volume'] > (last['volume_sma_20'] * vol_mult)
    
    if breakout and volume_ok:
        log_and_notify("debug", f"  -> [{df.name}] ✅ إشارة اختراق مقاومة.", "STRATEGY")
        return True
    if breakout and not volume_ok:
        log_rejection(df.name, "SR_Breakout: Volume Filter Failed")
    return False

def check_triple_confirmation_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 2: return False
    last, prev = df.iloc[-1], df.iloc[-2]
    mode = FILTER_CONFIG["TRIPLE_CONF_MODE"]["value"]
    vol_mult = FILTER_CONFIG["TRIPLE_CONF_VOLUME_MULT"]["value"]
    
    ema_bullish = last['ema_50'] > last['ema_200']
    macd_bullish = last['macd'] > last['macd_signal'] and prev['macd'] <= prev['macd_signal']
    rsi_ok = last['rsi'] < 70
    volume_ok = last['volume'] > (last['volume_sma_20'] * vol_mult)
    
    if mode == 'strict' and (ema_bullish and macd_bullish and rsi_ok and volume_ok):
        log_and_notify("debug", f"  -> [{df.name}] ✅ إشارة تأكيد ثلاثي (Strict).", "STRATEGY")
        return True
    if mode == 'relaxed' and (ema_bullish and macd_bullish and volume_ok):
        log_and_notify("debug", f"  -> [{df.name}] ✅ إشارة تأكيد ثلاثي (Relaxed).", "STRATEGY")
        return True
    return False

def check_vwap_reversal_strategy(df: pd.DataFrame) -> bool:
    if len(df) < 2 or 'vwap' not in df.columns: return False
    last = df.iloc[-1]
    mode = FILTER_CONFIG["VWAP_REVERSAL_MODE"]["value"]
    vol_mult = FILTER_CONFIG["VWAP_VOLUME_MULT"]["value"]
    
    vwap_deviation = ((last['close'] - last['vwap']) / last['vwap']) * 100
    reversal = vwap_deviation < -1.5 and last['close'] > df.iloc[-2]['close']
    volume_ok = last['volume'] > (last['volume_sma_20'] * vol_mult)
    
    if reversal and volume_ok:
        if mode == 'strict' and last['rsi'] < 40:
            log_and_notify("debug", f"  -> [{df.name}] ✅ إشارة ارتداد VWAP (Strict).", "STRATEGY")
            return True
        if mode == 'relaxed':
            log_and_notify("debug", f"  -> [{df.name}] ✅ إشارة ارتداد VWAP (Relaxed).", "STRATEGY")
            return True
    return False

def calculate_fibonacci_levels(high_point: float, low_point: float) -> Tuple[Dict, Dict]:
    price_range = high_point - low_point
    if price_range == 0: return {}, {}
    res = {f'{p:.1f}%': high_point - (price_range * (p/100)) for p in [0, 23.6, 38.2, 50, 61.8, 78.6, 100]}
    sup = {f'{p:.1f}%': low_point + (price_range * (p/100)) for p in [0, 23.6, 38.2, 50, 61.8, 78.6, 100]}
    return res, sup

def check_fibonacci_strategy(df: pd.DataFrame) -> Optional[str]:
    if len(df) < 50 or 'fib_high_point' not in df.columns: return None
    last, prev = df.iloc[-1], df.iloc[-2]
    high, low = last['fib_high_point'], last['fib_low_point']
    res, sup = calculate_fibonacci_levels(high, low)
    if not res: return None

    vol_mult = FILTER_CONFIG["FIB_VOLUME_MULT"]["value"]
    volume_ok = last['volume'] > (last['volume_sma_20'] * vol_mult)
    
    support_bounce = last['close'] > sup['61.8%'] and prev['close'] <= sup['61.8%']
    
    if support_bounce and volume_ok:
        log_and_notify("debug", f"  -> [{df.name}] ✅ إشارة فيبوناتشي (شراء عند الدعم).", "STRATEGY")
        return "BUY"
    if support_bounce and not volume_ok:
        log_rejection(df.name, "Fibonacci: Volume Filter Failed")
    return None

# --- دوال إدارة الصفقات والطلبات ---
def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> Optional[Decimal]:
    try:
        info = exchange_info_map[symbol]
        lot_size_filter = next((f for f in info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if not lot_size_filter: return Decimal(str(quantity))

        step_size = Decimal(lot_size_filter['stepSize'])
        min_qty = Decimal(lot_size_filter['minQty'])
        
        if quantity < min_qty:
            log_rejection(symbol, "Quantity less than minQty", {"quantity": quantity, "minQty": min_qty})
            return None
            
        adjusted_quantity = (Decimal(str(quantity)) // step_size) * step_size
        return adjusted_quantity
    except Exception as e:
        log_and_notify("error", f"❌ [{symbol}] فشل في تعديل حجم العقد: {e}", "TRADE_MGMT")
        return None

def calculate_position_size(symbol: str, entry_price: float, stop_loss_price: float) -> Optional[Decimal]:
    if not client: return None
    try:
        with risk_per_trade_lock: current_risk_percent = RISK_PER_TRADE_PERCENT
        balance_response = client.get_asset_balance(asset='USDT')
        available_balance = Decimal(balance_response['free'])
        
        risk_amount_usdt = available_balance * (Decimal(str(current_risk_percent)) / Decimal('100'))
        risk_per_coin = Decimal(str(entry_price)) - Decimal(str(stop_loss_price))
        
        if risk_per_coin <= 0:
            log_rejection(symbol, "Invalid risk per coin (SL too high)")
            return None

        quantity = risk_amount_usdt / risk_per_coin
        adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(quantity))

        if not adjusted_quantity or adjusted_quantity <= 0:
            log_rejection(symbol, "Lot size adjustment resulted in zero quantity")
            return None

        notional_value = adjusted_quantity * Decimal(str(entry_price))
        info = exchange_info_map.get(symbol)
        min_notional_filter = next((f for f in info['filters'] if f['filterType'] == 'MIN_NOTIONAL'), None)
        min_notional = Decimal(min_notional_filter['minNotional']) if min_notional_filter else Decimal('0')
        
        if notional_value < min_notional:
            log_rejection(symbol, "Min Notional Filter failed", {"value": notional_value, "min": min_notional})
            return None
        if notional_value > available_balance:
            log_rejection(symbol, "Insufficient Balance", {"required": notional_value, "available": available_balance})
            return None
            
        return adjusted_quantity
    except Exception as e:
        log_and_notify("error", f"❌ [{symbol}] خطأ في حساب حجم الصفقة: {e}", "TRADE_MGMT")
        return None

def place_order(symbol: str, side: str, quantity: Decimal) -> Optional[Dict]:
    if not client: return None
    try:
        log_and_notify("info", f"➡️ [{symbol}] محاولة وضع طلب {side} لكمية {quantity}", "ORDER")
        # لإلغاء تفعيل وضع الطلبات الحقيقية، قم بإلغاء التعليق على السطر التالي
        # return {"orderId": f"test_{int(time.time())}", "symbol": symbol, "status": "FILLED"}
        order = client.create_order(symbol=symbol, side=side, type=Client.ORDER_TYPE_MARKET, quantity=float(quantity))
        log_and_notify("info", f"✅ [{symbol}] تم وضع الطلب بنجاح. ID: {order.get('orderId')}", "ORDER")
        return order
    except (BinanceAPIException, BinanceRequestException) as e:
        log_and_notify("error", f"❌ [{symbol}] فشل وضع الطلب: {e.message}", "ORDER")
        return None

# --- الحلقة الرئيسية ---
def main_loop():
    log_and_notify("info", "[الحلقة الرئيسية] انتظار اكتمال التهيئة...", "SYSTEM")
    time.sleep(10)
    if not validated_symbols_to_scan: 
        log_and_notify("critical", "قائمة العملات فارغة. لا يمكن بدء المسح.", "SYSTEM_ERROR")
        return
    log_and_notify("info", f"✅ بدء حلقة المسح لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")
    
    while True:
        try:
            with trading_status_lock:
                if not is_trading_enabled:
                    time.sleep(30)
                    continue

            symbols_to_process = random.sample(validated_symbols_to_scan, len(validated_symbols_to_scan))
            
            for i in range(0, len(symbols_to_process), BATCH_SIZE):
                batch = symbols_to_process[i:i + BATCH_SIZE]
                logger.info(f"--- 🔄 بدء الدفعة | فحص {len(batch)} عملة ---")

                for symbol in batch:
                    with signal_cache_lock:
                        if symbol in open_signals_cache or len(open_signals_cache) >= MAX_OPEN_TRADES:
                            continue
                    
                    df = get_data_for_symbol(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df is None or len(df) < 201: continue
                        
                    df_indicators = calculate_all_features(df)
                    df_indicators.name = symbol
                    
                    signal_type, strategy_used = None, None
                    strategies_to_check = [
                        ('BB_Stoch', check_bb_stoch_strategy_enhanced), 
                        ('MACD_EMA', check_macd_ema_strategy),
                        ('SR_Breakout', check_sr_breakout_strategy_enhanced), 
                        ('Triple_Confirmation', check_triple_confirmation_strategy), 
                        ('VWAP_Reversal', check_vwap_reversal_strategy),
                        ('Fibonacci', check_fibonacci_strategy)
                    ]
                    
                    for key, func in strategies_to_check:
                        if STRATEGY_CONFIG[key]['enabled']:
                            result = func(df_indicators)
                            if result:
                                signal_type = "BUY" if isinstance(result, bool) else result
                                strategy_used = key
                                break
                            
                    if signal_type == "BUY":
                        log_and_notify("info", f"🎯 [{symbol}] إشارة شراء ناجحة من {strategy_used}. جاري التحقق النهائي...", "SIGNAL")
                        
                        try: 
                            entry_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                        except Exception as e: 
                            log_and_notify("error", f"❌ [{symbol}] فشل جلب سعر الدخول: {e}.", "SIGNAL")
                            continue
                        
                        last_atr = df_indicators.iloc[-1]['atr']
                        stop_loss_price = entry_price - (last_atr * 2)
                        
                        quantity = calculate_position_size(symbol, entry_price, stop_loss_price)
                        if not quantity: continue

                        new_signal = {
                            'symbol': symbol, 'strategy_name': strategy_used, 'entry_price': entry_price,
                            'stop_loss': stop_loss_price, 'timestamp': datetime.now(timezone.utc).isoformat(),
                            'quantity': float(quantity)
                        }
                        
                        order_result = place_order(symbol, Client.SIDE_BUY, quantity)
                        if order_result:
                            new_signal['order_id'] = order_result['orderId']
                            with signal_cache_lock:
                                open_signals_cache[symbol] = new_signal
                
                gc.collect()
                time.sleep(5)

            log_and_notify("info", f"✅ [نهاية الدورة] انتهت دورة المسح. الانتظار 1 دقيقة...", "SYSTEM");
            time.sleep(60)

        except (KeyboardInterrupt, SystemExit):
            log_and_notify("info", "إيقاف البوت.", "SYSTEM")
            break
        except Exception as main_err:
            log_and_notify("error", f"خطأ حرج في الحلقة الرئيسية: {main_err}", "SYSTEM_ERROR")
            traceback.print_exc()
            time.sleep(120)

# --- إعداد تطبيق Flask ---
app = Flask(__name__)
CORS(app)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم بوت التداول V11.1</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <style>
        body { font-family: 'Cairo', sans-serif; background-color: #111827; color: #d1d5db; }
        .card { background-color: #1f2937; border: 1px solid #374151; }
        .btn { transition: all 0.2s ease-in-out; }
        .btn-green { background-color: #10b981; color: white; }
        .btn-green:hover { background-color: #059669; }
        .btn-red { background-color: #ef4444; color: white; }
        .btn-red:hover { background-color: #dc2626; }
        .toggle-bg:after { content: ''; position: absolute; top: 2px; left: 2px; background: white; border-radius: 9999px; height: 1.25rem; width: 1.25rem; transition: 0.3s; }
        input:checked + .toggle-bg:after { transform: translateX(100%); }
        input:checked + .toggle-bg { background-color: #10b981; }
    </style>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap" rel="stylesheet">
</head>
<body x-data="botDashboard()" x-init="init()">
    <div class="container mx-auto p-4 md:p-6">
        <header class="flex justify-between items-center mb-6">
            <h1 class="text-3xl font-bold text-white">لوحة تحكم البوت V11.1</h1>
            <div class="flex items-center space-x-4 space-x-reverse">
                <span class="text-sm" x-text="`آخر تحديث: ${lastUpdated}`"></span>
                <div :class="status.is_bot_running ? 'bg-green-500' : 'bg-red-500'" class="w-4 h-4 rounded-full animate-pulse"></div>
            </div>
        </header>

        <!-- قسم التحكم الرئيسي -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div class="card p-4 rounded-lg flex flex-col justify-center items-center">
                <h2 class="text-lg font-semibold mb-2">حالة التداول</h2>
                <button @click="toggleTrading()" :class="status.is_trading_enabled ? 'btn-red' : 'btn-green'" class="btn w-full py-2 rounded-md font-bold" x-text="status.is_trading_enabled ? 'إيقاف التداول' : 'تفعيل التداول'"></button>
            </div>
            <div class="card p-4 rounded-lg text-center">
                <h2 class="text-lg font-semibold mb-2">الصفقات المفتوحة</h2>
                <p class="text-4xl font-bold text-white" x-text="status.open_signals_count"></p>
            </div>
            <div class="card p-4 rounded-lg text-center">
                <h2 class="text-lg font-semibold mb-2">نسبة المخاطرة (%)</h2>
                <div class="flex items-center justify-center">
                    <input type="number" step="0.1" x-model.number="status.risk_percent" class="bg-gray-700 border text-center border-gray-600 text-white text-lg rounded-lg w-24 p-2">
                    <button @click="updateRisk()" class="p-2.5 ms-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700">حفظ</button>
                </div>
            </div>
        </div>

        <!-- قسم الاستراتيجيات والفلاتر -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <div class="card p-6 rounded-lg">
                <h2 class="text-xl font-bold mb-4 text-white">الاستراتيجيات</h2>
                <div class="space-y-3">
                    <template x-for="(strategy, key) in status.strategies" :key="key">
                        <div class="flex justify-between items-center bg-gray-800 p-3 rounded-md">
                            <span x-text="strategy.display_name"></span>
                            <div class="relative inline-block w-10 mr-2 align-middle select-none">
                                <input type="checkbox" :id="key" :checked="strategy.enabled" @change="toggleStrategy(key)" class="toggle-checkbox absolute block w-6 h-6 rounded-full bg-white border-4 appearance-none cursor-pointer"/>
                                <label :for="key" class="toggle-label block overflow-hidden h-6 rounded-full bg-gray-600 cursor-pointer toggle-bg"></label>
                            </div>
                        </div>
                    </template>
                </div>
            </div>
            <div class="card p-6 rounded-lg">
                <h2 class="text-xl font-bold mb-4 text-white">الفلاتر</h2>
                <div class="space-y-4">
                     <template x-for="(filter, key) in status.filters" :key="key">
                        <div class="flex justify-between items-center">
                            <label :for="`filter_${key}`" class="text-gray-300" x-text="filter.display_name"></label>
                            <div class="flex items-center w-1/2">
                                <input :type="typeof filter.value === 'number' ? 'number' : 'text'" :id="`filter_${key}`" x-model="filter.value" class="bg-gray-700 border border-gray-600 text-white text-sm rounded-lg w-full p-2.5">
                                <button @click="updateFilter(key, filter.value)" class="p-2.5 ms-2 text-sm text-white bg-blue-600 rounded-lg hover:bg-blue-700">حفظ</button>
                            </div>
                        </div>
                    </template>
                </div>
            </div>
        </div>
        
        <!-- الصفقات المفتوحة -->
        <div class="card p-6 rounded-lg mb-6">
            <h2 class="text-xl font-bold mb-4 text-white">الصفقات المفتوحة</h2>
            <div class="overflow-x-auto">
                <table class="w-full text-sm text-right text-gray-400">
                    <thead class="text-xs uppercase bg-gray-700 text-gray-400">
                        <tr>
                            <th class="px-6 py-3">العملة</th><th class="px-6 py-3">الاستراتيجية</th><th class="px-6 py-3">سعر الدخول</th>
                            <th class="px-6 py-3">وقف الخسارة</th><th class="px-6 py-3">الكمية</th><th class="px-6 py-3">وقت الإشارة</th>
                        </tr>
                    </thead>
                    <tbody>
                        <template x-if="!status.open_signals || status.open_signals.length === 0">
                            <tr><td colspan="6" class="text-center py-4">لا توجد صفقات مفتوحة</td></tr>
                        </template>
                        <template x-for="signal in status.open_signals" :key="signal.symbol">
                            <tr class="border-b bg-gray-800 border-gray-700">
                                <th class="px-6 py-4 font-medium text-white" x-text="signal.symbol"></th>
                                <td class="px-6 py-4" x-text="signal.strategy_name"></td>
                                <td class="px-6 py-4" x-text="parseFloat(signal.entry_price).toFixed(4)"></td>
                                <td class="px-6 py-4 text-red-400" x-text="parseFloat(signal.stop_loss).toFixed(4)"></td>
                                <td class="px-6 py-4" x-text="parseFloat(signal.quantity).toFixed(6)"></td>
                                <td class="px-6 py-4" x-text="new Date(signal.timestamp).toLocaleString()"></td>
                            </tr>
                        </template>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- السجلات -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="card p-6 rounded-lg">
                <h2 class="text-xl font-bold mb-4 text-white">آخر سجلات الرفض</h2>
                <div class="space-y-2 max-h-96 overflow-y-auto">
                    <template x-for="log in status.rejection_logs" :key="log.timestamp">
                        <div class="bg-gray-800 p-2 rounded-md text-sm">
                            <p><span class="font-bold text-yellow-400" x-text="log.symbol"></span> - <span x-text="log.reason"></span></p>
                            <p class="text-xs text-gray-500" x-text="new Date(log.timestamp).toLocaleTimeString()"></p>
                        </div>
                    </template>
                </div>
            </div>
            <div class="card p-6 rounded-lg">
                <h2 class="text-xl font-bold mb-4 text-white">آخر الإشعارات</h2>
                <div class="space-y-2 max-h-96 overflow-y-auto">
                    <template x-for="n in status.notifications" :key="n.timestamp">
                        <div class="bg-gray-800 p-2 rounded-md text-sm" :class="{ 'border-r-4 border-green-500': n.level === 'INFO', 'border-r-4 border-yellow-500': n.level === 'WARNING', 'border-r-4 border-red-500': n.level.includes('ERROR') }">
                            <p x-text="n.message"></p>
                            <p class="text-xs text-gray-500" x-text="new Date(n.timestamp).toLocaleTimeString()"></p>
                        </div>
                    </template>
                </div>
            </div>
        </div>
    </div>
    <script>
        function botDashboard() {
            return {
                status: {}, lastUpdated: 'N/A',
                fetchStatus() {
                    fetch('/api/status').then(res => res.json()).then(data => {
                        this.status = data; this.lastUpdated = new Date().toLocaleTimeString();
                    }).catch(err => console.error('Error fetching status:', err));
                },
                postRequest(url, body) {
                    return fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
                },
                toggleTrading() { this.postRequest('/api/toggle_trading').then(() => this.fetchStatus()); },
                toggleStrategy(key) { this.postRequest('/api/toggle_strategy', { strategy: key }).then(() => this.fetchStatus()); },
                updateFilter(key, value) { this.postRequest('/api/update_filter', { filter: key, value: value }).then(() => this.fetchStatus()); },
                updateRisk() { this.postRequest('/api/update_risk', { risk: this.status.risk_percent }).then(() => this.fetchStatus()); },
                init() { this.fetchStatus(); setInterval(() => this.fetchStatus(), 5000); }
            }
        }
    </script>
</body>
</html>
"""

# --- نقاط نهاية Flask API ---
@app.route('/')
def index(): return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def api_status():
    return jsonify({
        "is_bot_running": True, "is_trading_enabled": is_trading_enabled,
        "open_signals_count": len(open_signals_cache), "risk_percent": RISK_PER_TRADE_PERCENT,
        "strategies": {k: {"enabled": v["enabled"], "display_name": v["display_name"]} for k, v in STRATEGY_CONFIG.items()},
        "filters": {k: {"value": v["value"], "display_name": v["display_name"]} for k, v in FILTER_CONFIG.items()},
        "open_signals": list(open_signals_cache.values()),
        "rejection_logs": rejection_logs_cache, "notifications": notifications_cache
    })

@app.route('/api/toggle_trading', methods=['POST'])
def toggle_trading():
    global is_trading_enabled
    with trading_status_lock: is_trading_enabled = not is_trading_enabled
    log_and_notify("info", f"تم تغيير حالة التداول إلى: {'مفعّل' if is_trading_enabled else 'متوقف'}", "CONTROL")
    return jsonify({"success": True})

@app.route('/api/toggle_strategy', methods=['POST'])
def toggle_strategy():
    key = request.json.get('strategy')
    if key in STRATEGY_CONFIG:
        with STRATEGY_CONFIG[key]['lock']: STRATEGY_CONFIG[key]['enabled'] = not STRATEGY_CONFIG[key]['enabled']
        log_and_notify("info", f"تم {'تفعيل' if STRATEGY_CONFIG[key]['enabled'] else 'إيقاف'} استراتيجية {key}", "CONTROL")
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Strategy not found"}), 404

@app.route('/api/update_filter', methods=['POST'])
def update_filter():
    key, value = request.json.get('filter'), request.json.get('value')
    if key in FILTER_CONFIG:
        try:
            original_type = type(FILTER_CONFIG[key]['value'])
            with FILTER_CONFIG[key]['lock']: FILTER_CONFIG[key]['value'] = original_type(value)
            log_and_notify("info", f"تم تحديث الفلتر {key} إلى {value}", "CONTROL")
            return jsonify({"success": True})
        except Exception as e: return jsonify({"success": False, "error": str(e)}), 400
    return jsonify({"success": False, "error": "Filter not found"}), 404

@app.route('/api/update_risk', methods=['POST'])
def update_risk():
    global RISK_PER_TRADE_PERCENT
    risk_val = request.json.get('risk')
    try:
        new_risk = float(risk_val)
        if 0.1 <= new_risk <= 5.0:
            with risk_per_trade_lock: RISK_PER_TRADE_PERCENT = new_risk
            log_and_notify("info", f"تم تحديث نسبة المخاطرة إلى {new_risk}%", "CONTROL")
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "Risk must be between 0.1 and 5.0"}), 400
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Invalid risk value"}), 400

# --- نقطة الدخول الرئيسية ---
if __name__ == "__main__":
    log_and_notify("info", "🚀 إطلاق بوت التداول المتكامل V11.1 🚀", "SYSTEM")
    
    initialize_database()
    initialize_redis()
    client = initialize_binance_client()
    
    if client:
        fetch_and_validate_symbols()
        main_loop_thread = Thread(target=main_loop, daemon=True)
        main_loop_thread.start()
    else:
        log_and_notify("critical", "فشل تهيئة عميل Binance. لن تبدأ حلقة التداول.", "SYSTEM_ERROR")

    log_and_notify("info", "🚀 تشغيل خادم Flask للوحة التحكم على http://127.0.0.1:5000 🚀", "SYSTEM")
    app.run(host='0.0.0.0', port=5000, debug=False)
