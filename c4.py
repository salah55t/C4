# ملف c4_enhanced_v10.7_fibonacci.py - نسخة V10.7 "Fibonacci"
# --- نسخة معدلة مع استراتيجية فيبوناتشي المتكاملة ---
# هذا الإصدار يضيف استراتيجية تداول قائمة على مستويات فيبوناتشي
# مع تحديد أهداف الربح ووقف الخسارة بشكل ديناميكي.
# --- تحديثات v10.7 ---
# 1. إضافة استراتيجية فيبوناتشي للدعم والمقاومة.
# 2. حساب أهداف الربح ووقف الخسارة تلقائيًا بناءً على مستويات فيبوناتشي.
# 3. تعديل حجم دفعة فحص العملات إلى 10 لتحسين إدارة الذاكرة.

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
import gc # مكتبة جامع القمامة
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException, BinanceRequestException
from flask import Flask, jsonify, render_template_string, request, abort
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timezone, timedelta
from decouple import config
from typing import List, Dict, Optional, Any, Tuple
from functools import wraps
import random

# --- إعدادات التجاهل واللوجر ---
warnings = __import__('warnings')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v10.7_fibonacci.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV10.7-Fibonacci')

# --- مشفر مخصص لأنواع بيانات NumPy والعشرية ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
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

# --- متغيرات عامة وأقفال ---
is_trading_enabled: bool = False
trading_status_lock = Lock()

# --- المتغيرات القابلة للتعديل ---
RISK_PER_TRADE_PERCENT: float = 0.85
risk_per_trade_lock = Lock()

# --- مفاتيح تفعيل الاستراتيجيات ---
STRATEGY_CONFIG = {
    "BB_Stoch": {"enabled": True, "lock": Lock(), "display_name": "BB+Stoch (Enhanced)"},
    "MACD_EMA": {"enabled": True, "lock": Lock(), "display_name": "MACD+EMA (Enhanced)"},
    "SR_Breakout": {"enabled": True, "lock": Lock(), "display_name": "S/R Breakout (Enhanced)"},
    "Triple_Confirmation": {"enabled": True, "lock": Lock(), "display_name": "Triple Confirmation (New)"},
    "VWAP_Reversal": {"enabled": True, "lock": Lock(), "display_name": "VWAP Reversal (New)"},
    "Fibonacci": {"enabled": True, "lock": Lock(), "display_name": "Fibonacci Levels (New)"},
}

# --- إعدادات الفلاتر القابلة للتعديل ---
FILTER_CONFIG = {
    "ADX_THRESHOLD": {"value": 20, "lock": Lock(), "display_name": "حد مؤشر ADX"},
    "BB_STOCH_VOLUME_MULT": {"value": 1.1, "lock": Lock(), "display_name": "مضاعف فوليوم (BB Stoch)"},
    "SR_BREAKOUT_VOLUME_MULT": {"value": 1.3, "lock": Lock(), "display_name": "مضاعف فوليوم (SR Breakout)"},
    "TRIPLE_CONF_VOLUME_MULT": {"value": 1.1, "lock": Lock(), "display_name": "مضاعف فوليوم (Triple Conf)"},
    "VWAP_VOLUME_MULT": {"value": 1.2, "lock": Lock(), "display_name": "مضاعف فوليوم (VWAP Reversal)"},
    "FIB_VOLUME_MULT": {"value": 1.2, "lock": Lock(), "display_name": "مضاعف فوليوم (Fibonacci)"},
    "TRIPLE_CONF_MODE": {"value": "relaxed", "lock": Lock(), "display_name": "وضع (Triple Conf)"}, # 'strict' or 'relaxed'
    "VWAP_REVERSAL_MODE": {"value": "relaxed", "lock": Lock(), "display_name": "وضع (VWAP Reversal)"}, # 'strict' or 'relaxed'
    "TIME_BASED_EXIT_CANDLES": {"value": 20, "lock": Lock(), "display_name": "إغلاق الصفقة بعد (شمعة)"}
}

# --- إعدادات المؤشرات الفنية والإطارات الزمنية ---
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 90
BTC_SYMBOL: str = 'BTCUSDT'
MAX_OPEN_TRADES: int = 5
ATR_PERIOD: int = 14
ADX_PERIOD: int = 14
CACHE_EXPIRATION_MINUTES: int = 15
BATCH_SIZE: int = 10 # <-- تم تعديل حجم الدفعة إلى 10

# --- إعدادات إدارة المخاطر والخروج ---
USE_SMART_EXIT_SYSTEM: bool = True
TAKE_PROFIT_LEVELS = { # هذه الأهداف تستخدم للاستراتيجيات التي لا تحدد أهدافها بنفسها
    1: {"atr_multiplier": 1.5, "exit_percentage": 0.50},
    2: {"atr_multiplier": 3.0, "exit_percentage": 0.30},
    3: {"atr_multiplier": 5.0, "exit_percentage": 0.20}
}
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_STOP_ACTIVATION_ATR: float = 2.0

# --- متغيرات الحالة والكاش ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
exchange_info_map: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
rejection_logs_cache = []
rejection_logs_lock = Lock()
notifications_cache = []
notifications_lock = Lock()
current_market_state: Dict[str, Any] = {"status": "INITIALIZING"}
market_state_lock = Lock()

# --- قاموس أسباب الرفض باللغة العربية ---
REJECTION_REASONS_AR = {
    "Market Status Filter: BTC Downtrend (5m)": "فلتر السوق: اتجاه البيتكوين هابط (5 دقائق)",
    "Market Status Filter: BTC Downtrend (4h)": "فلتر السوق: اتجاه البيتكوين هابط (4 ساعات)",
    "Market Status Filter: Low Liquidity": "فلتر السوق: سيولة منخفضة",
    "BB_Stoch: ADX Filter Failed": "BB_Stoch: فلتر قوة الاتجاه ADX",
    "BB_Stoch: BBW Filter Failed": "BB_Stoch: فلتر توسع البولينجر BBW",
    "BB_Stoch: Volume Filter Failed": "BB_Stoch: فلتر تأكيد حجم التداول",
    "MACD_EMA: RSI Filter Failed": "MACD_EMA: فلتر RSI",
    "MACD_EMA: Trend Filter Failed": "MACD_EMA: فلتر تأكيد الاتجاه",
    "SR_Breakout: Retest Failed": "SR_Breakout: فشل إعادة اختبار المستوى",
    "Triple_Confirmation: Conditions Not Met": "Triple Confirmation: لم تتحقق الشروط",
    "VWAP_Reversal: Conditions Not Met": "VWAP Reversal: لم تتحقق الشروط",
    "Fibonacci: Volume Filter Failed (Resistance)": "Fibonacci: فلتر حجم التداول (مقاومة)",
    "Fibonacci: Volume Filter Failed (Support)": "Fibonacci: فلتر حجم التداول (دعم)",
    "Insufficient Balance": "الرصيد غير كافٍ",
    "Min Notional Filter": "قيمة الصفقة أقل من الحد الأدنى",
    "Invalid Position Size": "حجم الصفقة غير صالح",
    "Lot Size Adjustment Failed": "فشل تعديل حجم العقد"
}

# --- آلية تنظيم الطلبات المتقدمة (Token Bucket) ---
class RequestThrottler:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_rate = float(refill_rate)
        self.last_refill_time = time.time()
        self.lock = Lock()
        self.total_weight_used_minute = 0
        self.minute_start_time = time.time()

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill_time
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill_time = now
        if now - self.minute_start_time > 60:
            self.total_weight_used_minute = 0
            self.minute_start_time = now

    def consume(self, weight: int) -> None:
        with self.lock:
            self._refill()
            if weight > self.tokens:
                wait_time = (weight - self.tokens) / self.refill_rate
                logger.warning(f"🚦 [Throttler] الوزن المطلوب ({weight}) أعلى من المتاح ({self.tokens:.2f}). الانتظار {wait_time:.2f} ثانية.")
                time.sleep(wait_time)
            self._refill()
            self.tokens -= weight
            self.total_weight_used_minute += weight

throttler = RequestThrottler(capacity=5900, refill_rate=100) 

def rate_limiter(weight=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            throttler.consume(weight)
            try:
                response = func(*args, **kwargs)
                return response
            except (BinanceAPIException, BinanceRequestException) as e:
                if e.status_code in [429, 418]:
                    logger.critical("🚨 [API BAN] تم الوصول إلى حد الطلبات (HTTP 429/418). سيتم الانتظار لمدة 10 دقائق.")
                    send_telegram_message("🚨 *تحذير حظر API!* 🚨\nتم الوصول إلى حد الطلبات. سيتوقف البوت مؤقتاً.")
                    time.sleep(600)
                raise
        return wrapper
    return decorator

# --- دوال مساعدة ---
def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("[تليجرام] Token أو Chat ID غير معين، تم تخطي الإرسال.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("[تليجرام] تم إرسال الرسالة بنجاح.")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [تليجرام] فشل إرسال الرسالة: {e}")

# ... (بقية دوال التهيئة والاتصال بقاعدة البيانات و Redis تبقى كما هي) ...
# --- دوال جلب البيانات وحساب المؤشرات ---
# ... (دوال جلب البيانات تبقى كما هي) ...

def calculate_fibonacci_levels(df: pd.DataFrame, high_point: float, low_point: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    حساب مستويات فيبوناتشي للدعم والمقاومة بناءً على قمة وقاع محددين.
    """
    price_range = high_point - low_point
    if price_range == 0: # تجنب القسمة على صفر
        return {}, {}

    # مستويات المقاومة (ترتد الأسعار للأسفل)
    resistance_levels = {
        '0%': high_point,
        '23.6%': high_point - (price_range * 0.236),
        '38.2%': high_point - (price_range * 0.382),
        '50%': high_point - (price_range * 0.5),
        '61.8%': high_point - (price_range * 0.618),
        '78.6%': high_point - (price_range * 0.786),
        '100%': low_point
    }
    
    # مستويات الدعم (ترتد الأسعار للأعلى)
    support_levels = {
        '100%': high_point,
        '78.6%': low_point + (price_range * 0.214), # 1 - 0.786
        '61.8%': low_point + (price_range * 0.382), # 1 - 0.618
        '50%': low_point + (price_range * 0.5),
        '38.2%': low_point + (price_range * 0.618), # 1 - 0.382
        '23.6%': low_point + (price_range * 0.764), # 1 - 0.236
        '0%': low_point
    }
    
    return resistance_levels, support_levels

def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    df_calc = df.copy()
    
    # ... (الكود الحالي لحساب المؤشرات الأخرى يبقى كما هو) ...
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
    
    # إضافة حساب مستويات فيبوناتشي
    if len(df_calc) >= 50:
        last_50_candles = df_calc.iloc[-50:]
        high_point = last_50_candles['high'].max()
        low_point = last_50_candles['low'].min()
        
        # تخزين القمة والقاع في DataFrame لسهولة الوصول
        df_calc['fib_high_point'] = high_point
        df_calc['fib_low_point'] = low_point

    return df_calc.dropna()

# --- دوال الاستراتيجيات ---

# ... (بقية دوال الاستراتيجيات تبقى كما هي) ...

def check_fibonacci_strategy(df: pd.DataFrame) -> Optional[str]:
    """
    التحقق من استراتيجية فيبوناتشي وتحديد نوع الإشارة (شراء أو بيع).
    
    العائد:
    'BUY' أو 'SELL' إذا تحققت الشروط، None خلاف ذلك.
    """
    if len(df) < 50 or 'fib_high_point' not in df.columns: 
        return None
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # الحصول على القمة والقاع من DataFrame
    high_point = last['fib_high_point']
    low_point = last['fib_low_point']
    
    # حساب مستويات فيبوناتشي
    resistance_levels, support_levels = calculate_fibonacci_levels(df, high_point, low_point)
    
    if not resistance_levels or not support_levels:
        return None

    # شروط استراتيجية المقاومة (للبيع)
    # السعر يرتد من مستوى مقاومة مهم (مثل 61.8%)
    resistance_condition = (
        last['close'] < resistance_levels['61.8%'] and
        prev['close'] >= resistance_levels['61.8%'] and
        last['close'] > resistance_levels['78.6%'] # تأكيد عدم الهبوط السريع
    )
    
    # شروط استراتيجية الدعم (للشراء)
    # السعر يرتد من مستوى دعم مهم (مثل 38.2% أو 61.8%)
    support_condition = (
        last['close'] > support_levels['61.8%'] and
        prev['close'] <= support_levels['61.8%'] and
        last['close'] < support_levels['50%'] # تأكيد عدم الصعود السريع
    )
    
    # التحقق من حجم التداول
    with FILTER_CONFIG["FIB_VOLUME_MULT"]["lock"]:
        vol_mult = FILTER_CONFIG["FIB_VOLUME_MULT"]["value"]
    
    volume_confirmed = last['volume'] > (last['volume_sma_20'] * vol_mult)
    
    if resistance_condition and volume_confirmed:
        logger.info(f"  -> [{df.name}] ✅ إشارة فيبوناتشي (بيع عند المقاومة).")
        return "SELL"
    elif support_condition and volume_confirmed:
        logger.info(f"  -> [{df.name}] ✅ إشارة فيبوناتشي (شراء عند الدعم).")
        return "BUY"
    
    if resistance_condition and not volume_confirmed:
        log_rejection(df.name, "Fibonacci: Volume Filter Failed (Resistance)", {"vol_multiplier": vol_mult})
    elif support_condition and not volume_confirmed:
        log_rejection(df.name, "Fibonacci: Volume Filter Failed (Support)", {"vol_multiplier": vol_mult})
    
    return None

# ... (بقية دوال إدارة الصفقات تبقى كما هي) ...

# --- الحلقة الرئيسية ---
def main_loop_enhanced():
    logger.info("[الحلقة الرئيسية] انتظار اكتمال التهيئة...")
    time.sleep(15)
    if not validated_symbols_to_scan: 
        log_and_notify("critical", "قائمة العملات فارغة.", "SYSTEM_ERROR")
        return
    log_and_notify("info", f"✅ بدء حلقة المسح لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")
    
    while True:
        try:
            determine_market_state_enhanced()
            if not passes_comprehensive_market_filter():
                logger.info("⏸️ [الحلقة الرئيسية] السوق في حالة غير مناسبة. الانتظار 5 دقائق...")
                time.sleep(300)
                continue

            symbols_to_process = random.sample(validated_symbols_to_scan, len(validated_symbols_to_scan))
            num_batches = (len(symbols_to_process) + BATCH_SIZE - 1) // BATCH_SIZE

            for i in range(num_batches):
                batch_symbols = symbols_to_process[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                logger.info(f"--- 🔄 بدء الدفعة {i+1}/{num_batches} | فحص {len(batch_symbols)} عملة ---")

                for symbol in batch_symbols:
                    with signal_cache_lock:
                        if symbol in open_signals_cache or len(open_signals_cache) >= MAX_OPEN_TRADES:
                            continue
                    
                    df_signal = get_data_for_symbol(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_signal is None or len(df_signal) < 201:
                        continue
                        
                    df_with_indicators = calculate_all_features(df_signal)
                    df_with_indicators.name = symbol
                    
                    signal_found = None
                    strategy_used = None
                    signal_type = None

                    # --- فحص الاستراتيجيات ---
                    strategies_to_check = [
                        ('BB_Stoch', lambda df: "BUY" if check_bb_stoch_strategy_enhanced(df) else None), 
                        ('SR_Breakout', lambda df: "BUY" if check_sr_breakout_strategy_enhanced(df) else None), 
                        ('Triple_Confirmation', lambda df: "BUY" if check_triple_confirmation_strategy(df) else None), 
                        ('VWAP_Reversal', lambda df: "BUY" if check_vwap_reversal_strategy(df) else None),
                        ('Fibonacci', check_fibonacci_strategy) # هذه الدالة تعيد 'BUY' أو 'SELL'
                    ]
                    
                    for key, func in strategies_to_check:
                        with STRATEGY_CONFIG[key]['lock']: 
                            is_enabled = STRATEGY_CONFIG[key]['enabled']
                        if is_enabled:
                            result = func(df_with_indicators)
                            if result:
                                signal_found = True
                                strategy_used = key
                                signal_type = result
                                break
                            
                    if signal_found:
                        logger.info(f"  -> [{symbol}] إشارة ناجحة ({signal_type}) من {strategy_used}. جاري التحقق النهائي...")
                        try: 
                            entry_price_str = redis_client.hget("crypto_bot_prices", symbol)
                            if not entry_price_str:
                                logger.warning(f"⚠️ [{symbol}] لم يتم العثور على السعر في Redis. سيتم جلبه عبر API.")
                                entry_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                            else:
                                entry_price = float(entry_price_str)
                        except Exception as e: 
                            logger.error(f"❌ [{symbol}] فشل جلب سعر الدخول: {e}.")
                            continue
                        
                        # --- حساب وقف الخسارة والأهداف ---
                        stop_loss_price = None
                        exit_levels = {}
                        last_atr = df_with_indicators.iloc[-1]['atr']

                        if strategy_used == 'Fibonacci':
                            high_point = df_with_indicators.iloc[-1]['fib_high_point']
                            low_point = df_with_indicators.iloc[-1]['fib_low_point']
                            resistance_levels, support_levels = calculate_fibonacci_levels(df_with_indicators, high_point, low_point)
                            
                            if signal_type == 'BUY':
                                stop_loss_price = support_levels['0%'] # أدنى قاع
                                # الأهداف هي مستويات المقاومة الأعلى
                                exit_levels = {
                                    "1": {"target_price": resistance_levels['61.8%'], "exit_percentage": 0.50, "is_hit": False},
                                    "2": {"target_price": resistance_levels['38.2%'], "exit_percentage": 0.30, "is_hit": False},
                                    "3": {"target_price": resistance_levels['0%'], "exit_percentage": 0.20, "is_hit": False}
                                }
                            elif signal_type == 'SELL':
                                # ملاحظة: البوت الحالي لا يدعم البيع على المكشوف (Short Selling)
                                # هذا الجزء نظري ويمكن تفعيله في حال إضافة دعم البيع على المكشوف
                                logger.info(f"  -> [{symbol}] تم العثور على إشارة بيع فيبوناتشي (غير مدعومة حاليًا).")
                                continue # تخطي إشارة البيع

                        else: # للاستراتيجيات الأخرى، استخدم نظام ATR
                            size_result = calculate_dynamic_position_size(symbol, entry_price, last_atr)
                            if not size_result: continue
                            quantity, stop_loss_price = size_result
                            for level, config in TAKE_PROFIT_LEVELS.items():
                                exit_levels[str(level)] = {
                                    "target_price": entry_price + (last_atr * config['atr_multiplier']),
                                    "exit_percentage": config['exit_percentage'],
                                    "is_hit": False
                                }
                        
                        if not stop_loss_price:
                            logger.error(f"❌ [{symbol}] فشل حساب وقف الخسارة.")
                            continue

                        # --- حساب حجم الصفقة ---
                        # يتم استدعاء هذه الدالة بعد تحديد وقف الخسارة
                        risk_per_coin = abs(Decimal(str(entry_price)) - Decimal(str(stop_loss_price)))
                        size_result = calculate_position_size_from_sl(symbol, entry_price, risk_per_coin)
                        if not size_result:
                            continue
                        quantity, _ = size_result # تم حساب وقف الخسارة مسبقًا

                        new_signal = {
                            'symbol': symbol, 'strategy_name': strategy_used, 'entry_price': entry_price,
                            'stop_loss': stop_loss_price, 'exit_levels': exit_levels,
                            'signal_details': {'atr': last_atr, 'fib_res': resistance_levels, 'fib_sup': support_levels} if strategy_used == 'Fibonacci' else {'atr': last_atr}
                        }
                        
                        with trading_status_lock: 
                            is_enabled = is_trading_enabled
                        if is_enabled:
                            order_result = place_order(symbol, Client.SIDE_BUY, quantity)
                            if order_result:
                                new_signal.update({'is_real_trade': True, 'quantity': float(quantity), 'order_id': order_result['orderId']})
                            else:
                                continue
                                
                        saved_signal = insert_signal_into_db(new_signal)
                        if saved_signal:
                            with signal_cache_lock:
                                open_signals_cache[saved_signal['symbol']] = saved_signal
                
                logger.info(f"--- ✅ انتهت الدفعة {i+1}/{num_batches}. جاري تحرير الذاكرة... ---")
                gc.collect()
                logger.info("--- 🗑️ تم استدعاء جامع القمامة. ---")
                time.sleep(10) # فاصل زمني بين الدفعات

            logger.info(f"✅ [نهاية الدورة] انتهت دورة المسح الكاملة. الانتظار 3 دقائق...");
            time.sleep(180)

        except (KeyboardInterrupt, SystemExit):
            log_and_notify("info", "إيقاف البوت.", "SYSTEM")
            break
        except Exception as main_err:
            log_and_notify("error", f"خطأ حرج في الحلقة الرئيسية: {main_err}", "SYSTEM")
            traceback.print_exc()
            time.sleep(120)


# --- دالة مساعدة جديدة لحساب حجم الصفقة بناءً على وقف الخسارة ---
def calculate_position_size_from_sl(symbol: str, entry_price: float, risk_per_coin: Decimal) -> Optional[Tuple[Decimal, float]]:
    if not client: return None
    try:
        with risk_per_trade_lock: current_risk_percent = RISK_PER_TRADE_PERCENT
        balance_response = client.get_asset_balance(asset='USDT')
        available_balance = Decimal(balance_response['free'])
        risk_amount_usdt = available_balance * (Decimal(str(current_risk_percent)) / Decimal('100'))
        
        if risk_per_coin <= 0:
            log_rejection(symbol, "Invalid Position Size", {"details": "Risk per coin is zero or negative."})
            return None

        initial_quantity = risk_amount_usdt / risk_per_coin
        adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(initial_quantity))
        
        if adjusted_quantity is None or adjusted_quantity <= 0:
            log_rejection(symbol, "Lot Size Adjustment Failed")
            return None

        notional_value = adjusted_quantity * Decimal(str(entry_price))
        symbol_info = exchange_info_map.get(symbol)
        if symbol_info:
            min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] in ('MIN_NOTIONAL', 'NOTIONAL')), None)
            if min_notional_filter:
                min_notional = Decimal(min_notional_filter.get('minNotional', min_notional_filter.get('notional', '0')))
                if notional_value < min_notional:
                    log_rejection(symbol, "Min Notional Filter", {"value": f"{notional_value:.2f}", "min": f"{min_notional}"})
                    return None
        
        if notional_value > available_balance:
            log_rejection(symbol, "Insufficient Balance", {"required": f"{notional_value:.2f}", "available": f"{available_balance:.2f}"})
            return None
            
        # الدالة تعيد الكمية فقط، حيث أن وقف الخسارة تم حسابه مسبقًا
        return adjusted_quantity, 0.0 
    except Exception as e:
        logger.error(f"❌ [{symbol}] خطأ في حساب حجم الصفقة: {e}", exc_info=True)
        return None

# ... (بقية الكود: دوال WebSocket، التهيئة، ونقطة الدخول الرئيسية تبقى كما هي) ...

if __name__ == "__main__":
    logger.info("🚀 إطلاق بوت التداول V10.7 'Fibonacci' مع لوحة التحكم 🚀")
    # ... (الكود المتبقي لتشغيل الخدمات و Flask) ...
