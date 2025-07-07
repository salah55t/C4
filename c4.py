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
from urllib.parse import urlparse
from psycopg2 import sql, OperationalError, InterfaceError
from psycopg2.extras import RealDictCursor
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException
from flask import Flask, request, Response, jsonify, render_template_string
from flask_cors import CORS
from threading import Thread, Lock
from datetime import datetime, timedelta, timezone
from decouple import config
from typing import List, Dict, Optional, Any, Set
from sklearn.preprocessing import StandardScaler
from collections import deque
import warnings
import gc

# --- تجاهل التحذيرات غير الهامة ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v8_db_sl_tp.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV8_DB_SL_TP')

# ---------------------- تحميل متغيرات البيئة ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
    REDIS_URL: str = config('REDIS_URL', default='redis://localhost:6379/0')
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
    exit(1)

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V7_With_Ichimoku'
MODEL_FOLDER: str = 'V7' 
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 30
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices"
MODEL_BATCH_SIZE: int = 5
DIRECT_API_CHECK_INTERVAL: int = 10

# --- مؤشرات فنية ---
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
EMA_FAST_PERIOD: int = 50
EMA_SLOW_PERIOD: int = 200

# --- إدارة الصفقات ---
MAX_OPEN_TRADES: int = 10
MODEL_CONFIDENCE_THRESHOLD = 0.65 

# --- إعدادات الهدف ووقف الخسارة ---
USE_DATABASE_SL_TP: bool = True
ATR_FALLBACK_SL_MULTIPLIER: float = 1.5
ATR_FALLBACK_TP_MULTIPLIER: float = 2.0
SL_BUFFER_ATR_PERCENT: float = 0.25

# --- إعدادات وقف الخسارة المتحرك (Trailing Stop-Loss) ---
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_ACTIVATION_PROFIT_PERCENT: float = 1.0
TRAILING_DISTANCE_PERCENT: float = 0.8

# --- إعدادات الفلاتر المحسّنة ---
USE_BTC_TREND_FILTER: bool = True
BTC_SYMBOL: str = 'BTCUSDT'
BTC_TREND_TIMEFRAME: str = '4h'
BTC_TREND_EMA_PERIOD: int = 50

USE_SPEED_FILTER: bool = True
USE_MOMENTUM_ACCELERATION_FILTER: bool = True
ACCELERATION_LOOKBACK_PERIOD: int = 3
ACCELERATION_MIN_RSI_INCREASE: float = 2.0
ACCELERATION_MIN_ADX_INCREASE: float = 1.0

USE_RRR_FILTER: bool = True
MIN_RISK_REWARD_RATIO: float = 1.1

USE_BTC_CORRELATION_FILTER: bool = True
MIN_BTC_CORRELATION: float = 0.1

USE_MIN_VOLATILITY_FILTER: bool = True
MIN_VOLATILITY_PERCENT: float = 0.3


# --- المتغيرات العامة وقفل العمليات ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
ml_models_cache: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
notifications_cache = deque(maxlen=50)
notifications_lock = Lock()
signals_pending_closure: Set[int] = set()
closure_lock = Lock()
last_api_check_time = time.time()
last_market_regime_check = 0
current_market_regime = "RANGING"

# --- ✨ إضافة: ذاكرة مؤقتة لتخزين سجلات الرفض ---
rejection_logs_cache = deque(maxlen=100)
rejection_logs_lock = Lock()


# ---------------------- دوال قاعدة البيانات (تبقى كما هي) ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    global conn
    logger.info("[قاعدة البيانات] بدء تهيئة الاتصال...")
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        entry_price DOUBLE PRECISION NOT NULL,
                        target_price DOUBLE PRECISION NOT NULL,
                        stop_loss DOUBLE PRECISION NOT NULL,
                        status TEXT DEFAULT 'open',
                        closing_price DOUBLE PRECISION,
                        closed_at TIMESTAMP,
                        profit_percentage DOUBLE PRECISION,
                        strategy_name TEXT,
                        signal_details JSONB,
                        current_peak_price DOUBLE PRECISION
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL,
                        message TEXT NOT NULL,
                        is_read BOOLEAN DEFAULT FALSE
                    );
                """)
            conn.commit()
            logger.info("✅ [قاعدة البيانات] تم تهيئة جداول قاعدة البيانات بنجاح.")
            return
        except Exception as e:
            logger.error(f"❌ [قاعدة البيانات] خطأ في الاتصال (المحاولة {attempt + 1}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: logger.critical("❌ [قاعدة البيانات] فشل الاتصال بعد عدة محاولات."); exit(1)

def check_db_connection() -> bool:
    global conn
    if conn is None or conn.closed != 0:
        logger.warning("[قاعدة البيانات] الاتصال مغلق، محاولة إعادة الاتصال...")
        init_db()
    try:
        if conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
            return True
        return False
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [قاعدة البيانات] فقدان الاتصال: {e}. محاولة إعادة الاتصال...")
        try:
            init_db()
            return conn is not None and conn.closed == 0
        except Exception as retry_e:
            logger.error(f"❌ [قاعدة البيانات] فشل إعادة الاتصال: {retry_e}")
            return False
    return False

def log_and_notify(level: str, message: str, notification_type: str):
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error, 'critical': logger.critical}
    log_methods.get(level.lower(), logger.info)(message)
    if not check_db_connection() or not conn: return
    try:
        new_notification = {"timestamp": datetime.now().isoformat(), "type": notification_type, "message": message}
        with notifications_lock:
            notifications_cache.appendleft(new_notification)
        with conn.cursor() as cur:
            cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [Notify DB] فشل حفظ التنبيه في قاعدة البيانات: {e}")
        if conn: conn.rollback()

# --- ✨ إضافة: دالة لتسجيل سبب الرفض في الذاكرة المؤقتة والملف ---
def log_rejection(symbol: str, reason: str, details: Optional[Dict] = None):
    """Logs a signal rejection to the console, file, and a deque for the API."""
    details_str = f" | {details}" if details else ""
    logger.info(f"ℹ️ [{symbol}] تم رفض الإشارة. السبب: {reason}{details_str}")
    
    with rejection_logs_lock:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "reason": reason,
            "details": details or {}
        }
        rejection_logs_cache.appendleft(log_entry)

def init_redis() -> None:
    global redis_client
    logger.info("[Redis] بدء تهيئة الاتصال...")
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] تم الاتصال بنجاح بخادم Redis.")
    except redis.exceptions.ConnectionError as e:
        logger.critical(f"❌ [Redis] فشل الاتصال بـ Redis على {REDIS_URL}. الخطأ: {e}")
        exit(1)

# ---------------------- دوال Binance والبيانات (تبقى كما هي في الغالب) ----------------------
def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    logger.info(f"ℹ️ [التحقق] قراءة الرموز من '{filename}' والتحقق منها مع Binance...")
    if not client:
        logger.error("❌ [التحقق] كائن Binance client غير مهيأ.")
        return []
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        
        exchange_info = client.get_exchange_info()
        active = {s['symbol'] for s in exchange_info['symbols'] if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'}
        
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ [التحقق] سيقوم البوت بمراقبة {len(validated)} عملة معتمدة.")
        return validated
    except Exception as e:
        logger.error(f"❌ [التحقق] حدث خطأ أثناء التحقق من الرموز: {e}", exc_info=True)
        return []

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        start_dt = datetime.now(timezone.utc) - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines: return None
        
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        numeric_cols = {'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'float32'}
        df = df.astype(numeric_cols)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna()
    except BinanceAPIException as e:
        logger.warning(f"⚠️ [API Binance] خطأ في جلب بيانات {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ [البيانات] خطأ أثناء جلب البيانات التاريخية لـ {symbol}: {e}")
        return None

# ---------------------- دوال جلب الميزات من قاعدة البيانات (تبقى كما هي) ----------------------
def fetch_sr_levels_from_db(symbol: str) -> pd.DataFrame:
    if not check_db_connection() or not conn: return pd.DataFrame()
    query = "SELECT level_price, level_type FROM support_resistance_levels WHERE symbol = %s"
    try:
        with conn.cursor() as cur:
            cur.execute(query, (symbol,))
            levels = cur.fetchall()
            if not levels: return pd.DataFrame()
            return pd.DataFrame(levels)
    except Exception as e:
        logger.error(f"❌ [S/R Fetch Bot] Could not fetch S/R levels for {symbol}: {e}")
        if conn: conn.rollback()
        return pd.DataFrame()

def fetch_ichimoku_features_from_db(symbol: str, timeframe: str) -> pd.DataFrame:
    if not check_db_connection() or not conn: return pd.DataFrame()
    query = """
        SELECT timestamp, tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b
        FROM ichimoku_features
        WHERE symbol = %s AND timeframe = %s
        ORDER BY timestamp;
    """
    try:
        df_ichimoku = pd.read_sql(query, conn, params=(symbol, timeframe), index_col='timestamp', parse_dates=['timestamp'])
        if not df_ichimoku.index.tz:
             df_ichimoku.index = df_ichimoku.index.tz_localize('UTC')
        return df_ichimoku
    except Exception as e:
        logger.error(f"❌ [Ichimoku Fetch Bot] Could not fetch Ichimoku features for {symbol}: {e}")
        if conn: conn.rollback()
        return pd.DataFrame()

# ---------------------- دوال حساب الميزات (تبقى كما هي) ----------------------
def calculate_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    # ATR, ADX
    high_low = df_calc['high'] - df_calc['low']
    high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
    low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_calc['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    up_move = df_calc['high'].diff(); down_move = -df_calc['low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=ADX_PERIOD, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()
    # RSI
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    # Other simple features
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=30, min_periods=1).mean() + 1e-9)
    ema_fast_trend = df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    ema_slow_trend = df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()
    df_calc['price_vs_ema50'] = (df_calc['close'] / ema_fast_trend) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / ema_slow_trend) - 1
    df_calc['returns'] = df_calc['close'].pct_change()
    if btc_df is not None and not btc_df.empty:
        merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
        df_calc['btc_correlation'] = merged_df['returns'].rolling(window=30).corr(merged_df['btc_returns'])
    else:
        df_calc['btc_correlation'] = 0.0
    return df_calc.astype('float32', errors='ignore')

def load_ml_model_bundle_from_folder(symbol: str) -> Optional[Dict[str, Any]]:
    global ml_models_cache
    model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
    if model_name in ml_models_cache:
        return ml_models_cache[model_name]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FOLDER, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        return None
    try:
        with open(model_path, 'rb') as f:
            model_bundle = pickle.load(f)
        if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
            ml_models_cache[model_name] = model_bundle
            logger.info(f"✅ [نموذج تعلم الآلة] تم تحميل النموذج '{model_name}' بنجاح.")
            return model_bundle
        else:
            logger.error(f"❌ [نموذج تعلم الآلة] حزمة النموذج في '{model_path}' غير مكتملة.")
            return None
    except Exception as e:
        logger.error(f"❌ [نموذج تعلم الآلة] خطأ في تحميل النموذج للعملة {symbol}: {e}", exc_info=True)
        return None

# ---------------------- دوال الفلاتر وحساب الأهداف (تبقى كما هي) ----------------------

def determine_market_regime():
    global current_market_regime, last_market_regime_check
    if time.time() - last_market_regime_check < 300: return current_market_regime
    logger.info("ℹ️ [نظام السوق] تحديث حالة السوق (BTC)...")
    try:
        btc_data = fetch_historical_data(BTC_SYMBOL, '4h', 10)
        if btc_data is None or len(btc_data) < 50:
            logger.warning("⚠️ [نظام السوق] بيانات BTC غير كافية، سيتم استخدام النظام السابق.")
            return current_market_regime
        ema_fast = btc_data['close'].ewm(span=12, adjust=False).mean()
        ema_slow = btc_data['close'].ewm(span=26, adjust=False).mean()
        high_low = btc_data['high'] - btc_data['low']
        high_close = (btc_data['high'] - btc_data['close'].shift()).abs()
        low_close = (btc_data['low'] - btc_data['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()
        up_move = btc_data['high'].diff(); down_move = -btc_data['low'].diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=btc_data.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=btc_data.index)
        plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr.replace(0, 1e-9)
        minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr.replace(0, 1e-9)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
        adx = dx.ewm(span=14, adjust=False).mean()

        if adx.iloc[-1] > 25:
            current_market_regime = "UPTREND" if ema_fast.iloc[-1] > ema_slow.iloc[-1] else "DOWNTREND"
        else:
            current_market_regime = "RANGING"
        last_market_regime_check = time.time()
        logger.info(f"✅ [نظام السوق] تم تحديد الحالة: {current_market_regime} (ADX: {adx.iloc[-1]:.2f})")
        return current_market_regime
    except Exception as e:
        logger.error(f"❌ [نظام السوق] فشل تحديد نظام السوق: {e}")
        return current_market_regime

def passes_speed_filter(last_features: pd.Series) -> bool:
    symbol = last_features.name
    regime = determine_market_regime()
    if regime == "DOWNTREND":
        log_rejection(symbol, "فلتر السرعة", {"detail": "تم التعطيل بسبب السوق الهابط"})
        return True
    
    adx_threshold, rel_vol_threshold, rsi_min, rsi_max, log_msg = (22.0, 0.9, 40.0, 90.0, "صارمة (UPTREND)") if regime == "UPTREND" else (18.0, 0.8, 30.0, 80.0, "مخففة (RANGING)")

    adx, rel_vol, rsi = last_features.get('adx', 0), last_features.get('relative_volume', 0), last_features.get('rsi', 0)
    if (adx >= adx_threshold and rel_vol >= rel_vol_threshold and rsi_min <= rsi < rsi_max):
        return True
    
    log_rejection(symbol, "فلتر السرعة", {
        "ADX": f"{adx:.2f} (Req: >{adx_threshold})",
        "Volume": f"{rel_vol:.2f} (Req: >{rel_vol_threshold})",
        "RSI": f"{rsi:.2f} (Req: {rsi_min}-{rsi_max})"
    })
    return False

def calculate_db_driven_tp_sl(symbol: str, entry_price: float, sr_levels_df: pd.DataFrame, ichimoku_df: pd.DataFrame, last_atr: float) -> Optional[Dict[str, float]]:
    resistances, supports = [], []
    if not sr_levels_df.empty:
        for _, row in sr_levels_df.iterrows():
            level_price = row['level_price']
            if 'resist' in row['level_type'].lower() or 'poc' in row['level_type'].lower(): resistances.append(level_price)
            if 'supp' in row['level_type'].lower() or 'poc' in row['level_type'].lower(): supports.append(level_price)
    if not ichimoku_df.empty:
        last_ichi = ichimoku_df.iloc[-1]
        for level in [last_ichi.get('kijun_sen'), last_ichi.get('senkou_span_a'), last_ichi.get('senkou_span_b')]:
            if pd.notna(level):
                if level > entry_price: resistances.append(level)
                else: supports.append(level)
    
    potential_tps = sorted([r for r in resistances if r > entry_price])
    target_price = potential_tps[0] if potential_tps else None
    potential_sls = sorted([s for s in supports if s < entry_price], reverse=True)
    stop_loss_price = potential_sls[0] if potential_sls else None

    if target_price is None or stop_loss_price is None:
        log_rejection(symbol, "عدم وجود مستويات دعم/مقاومة", {"detail": "العودة إلى طريقة ATR"})
        fallback_tp = entry_price + (last_atr * ATR_FALLBACK_TP_MULTIPLIER)
        fallback_sl = entry_price - (last_atr * ATR_FALLBACK_SL_MULTIPLIER)
        return {'target_price': fallback_tp, 'stop_loss': fallback_sl, 'source': 'ATR_Fallback'}
    
    final_stop_loss = stop_loss_price - (last_atr * SL_BUFFER_ATR_PERCENT)
    return {'target_price': target_price, 'stop_loss': final_stop_loss, 'source': 'Database'}

# ---------------------- WebSocket و TradingStrategy ----------------------
def handle_price_update_message(msg: List[Dict[str, Any]]) -> None:
    if not isinstance(msg, list) or not redis_client: return
    try:
        price_updates = {item.get('s'): float(item.get('c', 0)) for item in msg if item.get('s') and item.get('c')}
        if price_updates:
            redis_client.hset(REDIS_PRICES_HASH_NAME, mapping=price_updates)
    except Exception as e:
        logger.error(f"❌ [WebSocket Price Updater] خطأ: {e}", exc_info=True)

def initiate_signal_closure(symbol: str, signal_to_close: Dict, status: str, closing_price: float):
    signal_id = signal_to_close.get('id')
    with closure_lock:
        if signal_id in signals_pending_closure: return
        signals_pending_closure.add(signal_id)
    with signal_cache_lock:
        signal_data_for_thread = open_signals_cache.pop(symbol, None)
    if signal_data_for_thread:
        Thread(target=close_signal, args=(signal_data_for_thread, status, closing_price, "auto_monitor")).start()
    else:
        with closure_lock:
            signals_pending_closure.discard(signal_id)

def run_websocket_manager() -> None:
    logger.info("ℹ️ [WebSocket] بدء مدير WebSocket...")
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    twm.start_miniticker_socket(callback=handle_price_update_message)
    logger.info("✅ [WebSocket] تم الاتصال بنجاح.")
    twm.join()

class TradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = load_ml_model_bundle_from_folder(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.feature_names is None: return None
        try:
            df_featured = calculate_features(df_15m, btc_df)
            delta_4h = df_4h['close'].diff()
            gain_4h = delta_4h.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
            loss_4h = -delta_4h.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
            df_4h['rsi_4h'] = 100 - (100 / (1 + (gain_4h / loss_4h.replace(0, 1e-9))))
            ema_fast_4h = df_4h['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
            df_4h['price_vs_ema50_4h'] = (df_4h['close'] / ema_fast_4h) - 1
            mtf_features = df_4h[['rsi_4h', 'price_vs_ema50_4h']]
            df_featured = df_featured.join(mtf_features)
            df_featured[['rsi_4h', 'price_vs_ema50_4h']] = df_featured[['rsi_4h', 'price_vs_ema50_4h']].fillna(method='ffill')
            for col in self.feature_names:
                if col not in df_featured.columns: df_featured[col] = 0.0
            df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_featured.dropna()
        except Exception as e:
            logger.error(f"❌ [{self.symbol}] فشل هندسة الميزات: {e}", exc_info=True)
            return None

    def generate_signal(self, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]) or df_features.empty: return None
        
        try:
            last_row_ordered_df = df_features.iloc[[-1]][self.feature_names]
            features_scaled_np = self.scaler.transform(last_row_ordered_df)
            features_scaled_df = pd.DataFrame(features_scaled_np, columns=self.feature_names)

            prediction = self.ml_model.predict(features_scaled_df)[0]
            prediction_proba = self.ml_model.predict_proba(features_scaled_df)[0]
            
            try:
                class_1_index = list(self.ml_model.classes_).index(1)
            except ValueError:
                return None
            
            prob_for_class_1 = prediction_proba[class_1_index]
            
            if prediction == 1 and prob_for_class_1 >= MODEL_CONFIDENCE_THRESHOLD:
                logger.info(f"✅ [العثور على إشارة] {self.symbol}: تنبأ النموذج 'شراء' بثقة {prob_for_class_1:.2%}.")
                return {'symbol': self.symbol, 'strategy_name': BASE_ML_MODEL_NAME, 'signal_details': {'ML_Probability_Buy': f"{prob_for_class_1:.2%}"}}
            
            # ✨ تعديل: تسجيل الرفض إذا لم تتحقق الشروط
            if prediction != 1:
                log_rejection(self.symbol, "تنبؤ النموذج ليس 'شراء'", {"prediction": prediction})
            elif prob_for_class_1 < MODEL_CONFIDENCE_THRESHOLD:
                log_rejection(self.symbol, "ثقة النموذج منخفضة", {"confidence": f"{prob_for_class_1:.2%}", "threshold": f"{MODEL_CONFIDENCE_THRESHOLD:.2%}"})

            return None
        except Exception as e:
            logger.warning(f"⚠️ [توليد إشارة] {self.symbol}: خطأ: {e}")
            return None

# ---------------------- حلقة مراقبة الصفقات (تبقى كما هي) ----------------------
def trade_monitoring_loop():
    global last_api_check_time
    logger.info("✅ [Trade Monitor] بدء مراقبة الصفقات (مع دعم الوقف المتحرك).")
    while True:
        try:
            with signal_cache_lock: signals_to_check = dict(open_signals_cache)
            if not signals_to_check or not redis_client or not client: time.sleep(1); continue
            perform_direct_api_check = (time.time() - last_api_check_time) > DIRECT_API_CHECK_INTERVAL
            if perform_direct_api_check: last_api_check_time = time.time()
            symbols_to_fetch = list(signals_to_check.keys())
            redis_prices_list = redis_client.hmget(REDIS_PRICES_HASH_NAME, symbols_to_fetch)
            redis_prices = {symbol: price for symbol, price in zip(symbols_to_fetch, redis_prices_list)}
            for symbol, signal in signals_to_check.items():
                signal_id = signal.get('id')
                with closure_lock:
                    if signal_id in signals_pending_closure: continue
                price = None
                if perform_direct_api_check:
                    try: price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                    except Exception: pass
                if not price and redis_prices.get(symbol): price = float(redis_prices[symbol])
                if not price: continue
                target_price = float(signal.get('target_price', 0))
                original_stop_loss = float(signal.get('stop_loss', 0))
                effective_stop_loss = original_stop_loss
                if USE_TRAILING_STOP_LOSS:
                    entry_price = float(signal.get('entry_price', 0))
                    activation_price = entry_price * (1 + TRAILING_ACTIVATION_PROFIT_PERCENT / 100)
                    if price > activation_price:
                        current_peak = float(signal.get('current_peak_price', entry_price))
                        if price > current_peak: signal['current_peak_price'] = price; current_peak = price
                        trailing_stop_price = current_peak * (1 - TRAILING_DISTANCE_PERCENT / 100)
                        effective_stop_loss = max(original_stop_loss, trailing_stop_price)
                
                status_to_set = None
                if price >= target_price: status_to_set = 'target_hit'
                elif price <= effective_stop_loss: status_to_set = 'stop_loss_hit'
                if status_to_set:
                    logger.info(f"✅ [TRIGGER] ID:{signal_id} | {symbol} | Condition '{status_to_set}' met.")
                    initiate_signal_closure(symbol, signal, status_to_set, price)
            time.sleep(0.2)
        except Exception as e:
            logger.error(f"❌ [Trade Monitor] خطأ فادح: {e}", exc_info=True)
            time.sleep(5)

# ---------------------- دوال التنبيهات والإدارة (تبقى كما هي) ----------------------
def send_telegram_message(target_chat_id: str, text: str, reply_markup: Optional[Dict] = None):
    if not TELEGRAM_TOKEN or not target_chat_id: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': str(target_chat_id), 'text': text, 'parse_mode': 'Markdown'}
    if reply_markup: payload['reply_markup'] = json.dumps(reply_markup)
    try: requests.post(url, json=payload, timeout=10)
    except Exception as e: logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

def send_new_signal_alert(signal_data: Dict[str, Any]):
    safe_symbol = signal_data['symbol'].replace('_', '\\_')
    entry, target, sl = signal_data['entry_price'], signal_data['target_price'], signal_data['stop_loss']
    profit_pct = ((target / entry) - 1) * 100
    risk_pct = ((entry / sl) - 1) * 100 if sl > 0 else 0
    rrr = profit_pct / risk_pct if risk_pct > 0 else 0
    message = (f"💡 *إشارة تداول جديدة ({BASE_ML_MODEL_NAME})* 💡\n\n"
               f"🪙 *العملة:* `{safe_symbol}`\n"
               f"⬅️ *الدخول:* `${entry:,.8g}`\n"
               f"🎯 *الهدف:* `${target:,.8g}` (`{profit_pct:+.2f}%`)\n"
               f"🛑 *الوقف:* `${sl:,.8g}` (`{risk_pct:.2f}%`)\n"
               f"📈 *مخاطرة/عائد:* `1:{rrr:.2f}`\n\n"
               f"🔍 *الثقة:* {signal_data['signal_details']['ML_Probability_Buy']}\n"
               f"⚙️ *مصدر الهدف:* {signal_data['signal_details']['TP_SL_Source']}")
    reply_markup = {"inline_keyboard": [[{"text": "📊 فتح لوحة التحكم", "url": WEBHOOK_URL or '#'}]]}
    send_telegram_message(CHAT_ID, message, reply_markup)
    log_and_notify('info', f"إشارة جديدة: {signal_data['symbol']}", "NEW_SIGNAL")

def insert_signal_into_db(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not check_db_connection() or not conn: return None
    try:
        entry, target, sl = float(signal['entry_price']), float(signal['target_price']), float(signal['stop_loss'])
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, current_peak_price)
                   VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;""",
                (signal['symbol'], entry, target, sl, signal.get('strategy_name'), json.dumps(signal.get('signal_details', {})), entry)
            )
            signal['id'] = cur.fetchone()['id']
        conn.commit()
        logger.info(f"✅ [قاعدة البيانات] تم إدراج الإشارة {signal['id']} لـ {signal['symbol']}.")
        return signal
    except Exception as e:
        logger.error(f"❌ [إدراج] خطأ في إدراج إشارة {signal['symbol']}: {e}", exc_info=True)
        if conn: conn.rollback()
        return None

def close_signal(signal: Dict, status: str, closing_price: float, closed_by: str):
    signal_id = signal.get('id'); symbol = signal.get('symbol')
    logger.info(f"بدء عملية إغلاق الإشارة {signal_id} ({symbol}) بحالة '{status}'")
    try:
        if not check_db_connection() or not conn: raise OperationalError("فشل الاتصال بقاعدة البيانات عند إغلاق الإشارة.")
        profit_pct = ((closing_price / signal['entry_price']) - 1) * 100
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE signals SET status = %s, closing_price = %s, closed_at = NOW(), profit_percentage = %s WHERE id = %s AND status = 'open';",
                (status, closing_price, profit_pct, signal_id)
            )
            if cur.rowcount == 0: logger.warning(f"⚠️ [DB Close] الإشارة {signal_id} مغلقة بالفعل أو غير موجودة."); return 
        conn.commit()
        status_map = {'target_hit': '✅ تحقق الهدف', 'stop_loss_hit': '🛑 ضرب وقف الخسارة', 'manual_close': '🖐️ أُغلقت يدوياً'}
        status_message = status_map.get(status, status)
        alert_msg = f"*{status_message}*\n`{symbol.replace('_', '\\_')}` | *الربح:* `{profit_pct:+.2f}%`"
        send_telegram_message(CHAT_ID, alert_msg)
        log_and_notify('info', f"{status_message}: {symbol} | الربح: {profit_pct:+.2f}%", 'CLOSE_SIGNAL')
        logger.info(f"✅ [DB Close] تم إغلاق الإشارة {signal_id} بنجاح.")
    except Exception as e:
        logger.error(f"❌ [DB Close] خطأ حاسم أثناء إغلاق الإشارة {signal_id}: {e}", exc_info=True)
        if conn: conn.rollback()
        if symbol:
            with signal_cache_lock:
                if symbol not in open_signals_cache: open_signals_cache[symbol] = signal; logger.info(f"🔄 [Recovery] تمت إعادة الإشارة {signal_id} للذاكرة المؤقتة بسبب خطأ.")
    finally:
        with closure_lock: signals_pending_closure.discard(signal_id)

def load_open_signals_to_cache():
    if not check_db_connection() or not conn: return
    logger.info("ℹ️ [تحميل] جاري تحميل الإشارات المفتوحة...")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status = 'open';")
            open_signals = cur.fetchall()
            with signal_cache_lock:
                open_signals_cache.clear()
                for signal in open_signals: open_signals_cache[signal['symbol']] = dict(signal)
            logger.info(f"✅ [تحميل] تم تحميل {len(open_signals)} إشارة مفتوحة.")
    except Exception as e: logger.error(f"❌ [تحميل] فشل تحميل الإشارات المفتوحة: {e}")

def load_notifications_to_cache():
    if not check_db_connection() or not conn: return
    logger.info("ℹ️ [تحميل] جاري تحميل آخر التنبيهات...")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 50;")
            recent = cur.fetchall()
            with notifications_lock:
                notifications_cache.clear()
                for n in reversed(recent): n['timestamp'] = n['timestamp'].isoformat(); notifications_cache.appendleft(dict(n))
            logger.info(f"✅ [تحميل] تم تحميل {len(notifications_cache)} تنبيه.")
    except Exception as e: logger.error(f"❌ [تحميل] فشل تحميل التنبيهات: {e}")

# ---------------------- حلقة العمل الرئيسية (تبقى كما هي) ----------------------
def get_btc_trend() -> Dict[str, Any]:
    if not client: return {"status": "error", "is_uptrend": False}
    try:
        klines = client.get_klines(symbol=BTC_SYMBOL, interval=BTC_TREND_TIMEFRAME, limit=BTC_TREND_EMA_PERIOD * 2)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df['close'] = pd.to_numeric(df['close'])
        ema = df['close'].ewm(span=BTC_TREND_EMA_PERIOD, adjust=False).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        is_uptrend = bool(current_price > ema)
        return {"status": "Uptrend" if is_uptrend else "Downtrend", "is_uptrend": is_uptrend}
    except Exception as e:
        logger.error(f"❌ [فلتر BTC] فشل تحديد اتجاه البيتكوين: {e}")
        return {"status": "Error", "is_uptrend": False}

def get_btc_data_for_bot() -> Optional[pd.DataFrame]:
    btc_data = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
    if btc_data is None: logger.error("❌ [بيانات BTC] فشل جلب بيانات البيتكوين."); return None
    btc_data['btc_returns'] = btc_data['close'].pct_change()
    return btc_data

def main_loop():
    logger.info("[الحلقة الرئيسية] انتظار اكتمال التهيئة...")
    time.sleep(15) 
    if not validated_symbols_to_scan: log_and_notify("critical", "لا توجد رموز معتمدة للمسح.", "SYSTEM"); return
    log_and_notify("info", f"بدء حلقة المسح لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")
    all_symbols = list(validated_symbols_to_scan)
    while True:
        try:
            determine_market_regime()
            
            for i in range(0, len(all_symbols), MODEL_BATCH_SIZE):
                symbol_batch = all_symbols[i:i + MODEL_BATCH_SIZE]
                ml_models_cache.clear(); gc.collect()
                
                btc_trend_info = get_btc_trend()
                if USE_BTC_TREND_FILTER and not btc_trend_info.get("is_uptrend"):
                    log_rejection("ALL", "فلتر اتجاه BTC", {"detail": "تم إيقاف المسح بسبب اتجاه BTC الهابط"})
                    time.sleep(300)
                    break
                
                with signal_cache_lock: open_count = len(open_signals_cache)
                if open_count >= MAX_OPEN_TRADES:
                    logger.info(f"ℹ️ [إيقاف مؤقت] تم الوصول للحد الأقصى للصفقات."); time.sleep(60); break 
                
                slots_available = MAX_OPEN_TRADES - open_count
                if slots_available <= 0: break
                
                btc_data = get_btc_data_for_bot()
                if btc_data is None: time.sleep(120); continue
                
                for symbol in symbol_batch:
                    if slots_available <= 0: break
                    with signal_cache_lock:
                        if symbol in open_signals_cache: continue
                    try:
                        df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                        if df_15m is None or df_4h is None: continue
                        
                        strategy = TradingStrategy(symbol)
                        if not all([strategy.ml_model, strategy.scaler, strategy.feature_names]):
                            # log_rejection(symbol, "نموذج التعلم الآلي غير موجود") # This can be noisy
                            continue

                        df_features = strategy.get_features(df_15m, df_4h, btc_data)
                        if df_features is None or df_features.empty: continue
                        
                        potential_signal = strategy.generate_signal(df_features)
                        if not potential_signal or not redis_client: continue
                        
                        current_price_str = redis_client.hget(REDIS_PRICES_HASH_NAME, symbol)
                        if not current_price_str: continue
                        current_price = float(current_price_str)
                        
                        last_features = df_features.iloc[-1]; last_features.name = symbol
                        
                        if USE_SPEED_FILTER and not passes_speed_filter(last_features): continue
                        
                        last_atr = last_features.get('atr', 0)
                        volatility = (last_atr / current_price * 100)
                        if USE_MIN_VOLATILITY_FILTER and volatility < MIN_VOLATILITY_PERCENT:
                            log_rejection(symbol, "فلتر التقلب المنخفض", {"volatility": f"{volatility:.2f}%", "min_required": f"{MIN_VOLATILITY_PERCENT}%"})
                            continue

                        if USE_BTC_CORRELATION_FILTER and btc_trend_info.get("is_uptrend"):
                            correlation = last_features.get('btc_correlation', 0)
                            if correlation < MIN_BTC_CORRELATION:
                                log_rejection(symbol, "فلتر الارتباط مع BTC", {"correlation": f"{correlation:.2f}", "min_required": f"{MIN_BTC_CORRELATION}"})
                                continue
                        
                        sr_levels = fetch_sr_levels_from_db(symbol)
                        ichimoku_data = fetch_ichimoku_features_from_db(symbol, SIGNAL_GENERATION_TIMEFRAME)
                        
                        tp_sl_data = calculate_db_driven_tp_sl(symbol, current_price, sr_levels, ichimoku_data, last_atr)
                        if not tp_sl_data: continue

                        potential_signal.update(tp_sl_data)
                        potential_signal['entry_price'] = current_price
                        potential_signal['signal_details']['TP_SL_Source'] = tp_sl_data['source']

                        if USE_RRR_FILTER:
                            tp, sl = potential_signal['target_price'], potential_signal['stop_loss']
                            risk, reward = current_price - sl, tp - current_price
                            if risk <= 0 or reward <= 0: continue
                            rrr = reward / risk
                            if rrr < MIN_RISK_REWARD_RATIO:
                                log_rejection(symbol, "فلتر المخاطرة/العائد", {"RRR": f"{rrr:.2f}", "min_required": f"{MIN_RISK_REWARD_RATIO}"})
                                continue

                        logger.info(f"✅ [{symbol}] الإشارة مرت من جميع الفلاتر. جاري الحفظ...")
                        saved_signal = insert_signal_into_db(potential_signal)
                        if saved_signal:
                            with signal_cache_lock: open_signals_cache[saved_signal['symbol']] = saved_signal
                            send_new_signal_alert(saved_signal)
                            slots_available -= 1
                        
                        del df_15m, df_4h, sr_levels, ichimoku_data, df_features; gc.collect()

                    except Exception as e:
                        logger.error(f"❌ [خطأ معالجة] {symbol}: {e}", exc_info=True)
                time.sleep(10)
            logger.info("ℹ️ [نهاية الدورة] انتهت دورة المسح. انتظار..."); 
            time.sleep(60)
        except (KeyboardInterrupt, SystemExit): break
        except Exception as main_err:
            log_and_notify("error", f"خطأ في الحلقة الرئيسية: {main_err}", "SYSTEM"); time.sleep(120)

# ---------------------- واجهة برمجة تطبيقات Flask (تبقى كما هي) ----------------------
app = Flask(__name__)
CORS(app)

def get_fear_and_greed_index() -> Dict[str, Any]:
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10).json()
        value = int(response['data'][0]['value'])
        classification = response['data'][0]['value_classification']
        return {"value": value, "classification": classification}
    except Exception: return {"value": -1, "classification": "Error"}

@app.route('/')
def home():
    try:
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, 'index.html')
        with open(file_path, 'r', encoding='utf-8') as f: return render_template_string(f.read())
    except FileNotFoundError: return "<h1>ملف index.html غير موجود.</h1>", 404

@app.route('/api/market_status')
def get_market_status(): return jsonify({"btc_trend": get_btc_trend(), "fear_and_greed": get_fear_and_greed_index(), "market_regime": current_market_regime})

@app.route('/api/stats')
def get_stats():
    if not check_db_connection() or not conn: return jsonify({"error": "فشل الاتصال بقاعدة البيانات"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT status, profit_percentage FROM signals;")
            all_signals = cur.fetchall()
        
        with signal_cache_lock:
            open_trades_count = len(open_signals_cache)

        closed_trades = [s for s in all_signals if s.get('status') != 'open' and s.get('profit_percentage') is not None]
        
        wins = [s for s in closed_trades if s['status'] == 'target_hit']
        losses = [s for s in closed_trades if s['status'] == 'stop_loss_hit']
        
        total_profit_pct = sum(s['profit_percentage'] for s in closed_trades)
        
        win_count = len(wins)
        loss_count = len(losses)
        total_closed = win_count + loss_count
        
        win_rate = (win_count / total_closed * 100) if total_closed > 0 else 0
        
        total_profit_from_wins = sum(s['profit_percentage'] for s in wins)
        total_loss_from_losses = abs(sum(s['profit_percentage'] for s in losses))
        
        profit_factor = (total_profit_from_wins / total_loss_from_losses) if total_loss_from_losses > 0 else 0
        
        avg_win_pct = (total_profit_from_wins / win_count) if win_count > 0 else 0
        avg_loss_pct = (total_loss_from_losses / loss_count) if loss_count > 0 else 0


        return jsonify({
            "open_trades_count": open_trades_count,
            "total_profit_pct": total_profit_pct,
            "total_closed_trades": len(closed_trades),
            "wins": win_count,
            "losses": loss_count,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win_pct": avg_win_pct,
            "avg_loss_pct": -avg_loss_pct 
        })
    except Exception as e:
        logger.error(f"❌ [API Stats] Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/signals')
def get_signals():
    if not check_db_connection() or not conn or not redis_client: return jsonify({"error": "فشل الاتصال بالخدمات"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals ORDER BY CASE WHEN status = 'open' THEN 0 ELSE 1 END, id DESC;")
            all_signals = cur.fetchall()
        open_symbols = [s['symbol'] for s in all_signals if s['status'] == 'open']
        current_prices = {}
        if open_symbols:
            prices_list = redis_client.hmget(REDIS_PRICES_HASH_NAME, open_symbols)
            current_prices = {symbol: float(p) if p else None for symbol, p in zip(open_symbols, prices_list)}
        for s in all_signals:
            if s.get('closed_at'): s['closed_at'] = s['closed_at'].isoformat()
            if s['status'] == 'open':
                price = current_prices.get(s['symbol'])
                s['current_price'] = price
                if price and s.get('entry_price') > 0: s['pnl_pct'] = ((price / s['entry_price']) - 1) * 100
        return jsonify(all_signals)
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/api/close/<int:signal_id>', methods=['POST'])
def manual_close_signal(signal_id):
    if not client: return jsonify({"error": "Binance Client غير متاح"}), 500
    with closure_lock:
        if signal_id in signals_pending_closure: return jsonify({"error": "الإشارة قيد الإغلاق"}), 409
    if not check_db_connection() or not conn: return jsonify({"error": "فشل الاتصال بقاعدة البيانات"}), 500
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE id = %s AND status = 'open';", (signal_id,))
            signal_to_close = cur.fetchone()
        if not signal_to_close: return jsonify({"error": "لم يتم العثور على الإشارة"}), 404
        symbol = signal_to_close['symbol']
        price = float(client.get_symbol_ticker(symbol=symbol)['price'])
        initiate_signal_closure(symbol, dict(signal_to_close), 'manual_close', price)
        return jsonify({"message": f"جاري إغلاق الإشارة {signal_id}."})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/api/notifications')
def get_notifications():
    with notifications_lock: return jsonify(list(notifications_cache))

# --- ✨ إضافة: نقطة نهاية جديدة لجلب سجلات الرفض ---
@app.route('/api/rejection_logs')
def get_rejection_logs():
    with rejection_logs_lock:
        return jsonify(list(rejection_logs_cache))

def run_flask():
    host, port = "0.0.0.0", int(os.environ.get('PORT', 10000))
    log_and_notify("info", f"بدء تشغيل لوحة التحكم على {host}:{port}", "SYSTEM")
    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        logger.warning("⚠️ [Flask] مكتبة 'waitress' غير موجودة, سيتم استخدام خادم التطوير.")
        app.run(host=host, port=port)

# ---------------------- نقطة انطلاق البرنامج ----------------------
def initialize_bot_services():
    global client, validated_symbols_to_scan
    logger.info("🤖 [خدمات البوت] بدء التهيئة في الخلفية...")
    try:
        client = Client(API_KEY, API_SECRET)
        logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
        init_db()
        init_redis()
        load_open_signals_to_cache(); load_notifications_to_cache()
        validated_symbols_to_scan = get_validated_symbols()
        if not validated_symbols_to_scan:
            logger.critical("❌ لا توجد رموز معتمدة للمسح. الحلقات لن تبدأ."); return
        Thread(target=run_websocket_manager, daemon=True).start()
        Thread(target=trade_monitoring_loop, daemon=True).start()
        Thread(target=main_loop, daemon=True).start()
        logger.info("✅ [خدمات البوت] تم بدء جميع خدمات الخلفية بنجاح.")
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حاسم أثناء التهيئة: {e}", "SYSTEM")
        exit(1)

if __name__ == "__main__":
    logger.info(f"🚀 بدء تشغيل بوت التداول - إصدار {BASE_ML_MODEL_NAME}...")
    initialization_thread = Thread(target=initialize_bot_services, daemon=True)
    initialization_thread.start()
    run_flask()
    logger.info("👋 [إيقاف] تم إيقاف تشغيل البوت."); os._exit(0)
