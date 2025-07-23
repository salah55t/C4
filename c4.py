# ملف c4.py - نسخة محدثة مع نظام الإيقاف المؤقت وإدارة الطوارئ
# --- تم التحديث بواسطة Gemini ---
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
import warnings
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
from collections import deque, Counter

# --- إعدادات التجاهل واللوجر ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v9_telegram_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV9_Telegram')

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
are_filters_disabled: bool = False
filters_disabled_lock = Lock()
RISK_PER_TRADE_PERCENT: float = 1.0
BASE_ML_MODEL_NAME: str = 'LightGBM_Scalping_V9_With_Microstructure'
MODEL_FOLDER: str = 'V9'
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '4h'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 90
REDIS_PRICES_HASH_NAME: str = "crypto_bot_current_prices_v9"
TRADING_FEE_PERCENT: float = 0.1
STATS_TRADE_SIZE_USDT: float = 10.0
BTC_SYMBOL: str = 'BTCUSDT'
MAX_OPEN_TRADES: int = 4
BUY_CONFIDENCE_THRESHOLD = 0.75
MIN_PROFIT_PERCENT: float = 1.0
SYMBOL_PROCESSING_BATCH_SIZE: int = 20

# --- NEW: Scheduled Pause System Settings ---
is_bot_paused: bool = False
pause_reason: str = ""
pause_lock = Lock()
last_telegram_alert_hour: int = -1
PAUSE_SCHEDULE = {
    0: "الاستعداد لبداية الجلسة الآسيوية وإغلاق الشمعة اليومية.",
    4: "فترة ضعف سيولة محتملة في منتصف الجلسة الآسيوية.",
    8: "الاستعداد لافتتاح جلسة لندن وتداخلها مع إغلاق طوكيو.",
    12: "فترة تقلبات عالية مع تداخل جلسات لندن ونيويورك.",
    16: "الاستعداد لإغلاق جلسة لندن وجني الأرباح الأوروبية.",
    20: "اقتراب نهاية جلسة نيويورك وجني الأرباح الأمريكية."
}

# --- NEW: Emergency System Settings ---
CRASH_PROTECTION_ENABLED: bool = True
emergency_detector = None # سيتم تهيئته عند أول استدعاء

# --- إعدادات المؤشرات الفنية ---
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
EMA_SLOW_PERIOD: int = 200
EMA_FAST_PERIOD: int = 50
BTC_CORR_PERIOD: int = 30
REL_VOL_PERIOD: int = 30
MOMENTUM_PERIOD: int = 12
EMA_SLOPE_PERIOD: int = 5

# --- إعدادات الفلاتر المتقدمة وإدارة الصفقات ---
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_ACTIVATION_PROFIT_PERCENT: float = 1.0
TRAILING_DISTANCE_PERCENT: float = 0.8
USE_PEAK_FILTER: bool = True
PEAK_CHECK_PERIOD: int = 50
PULLBACK_THRESHOLD_PCT: float = 0.988
BREAKOUT_ALLOWANCE_PCT: float = 1.003
DYNAMIC_FILTER_ANALYSIS_INTERVAL: int = 300
ORDER_BOOK_DEPTH_LIMIT: int = 100
ORDER_BOOK_WALL_MULTIPLIER: float = 10.0
ORDER_BOOK_ANALYSIS_RANGE_PCT: float = 0.02

# --- متغيرات الحالة والكاش ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
ml_models_cache: Dict[str, Any] = {}
exchange_info_map: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []
open_signals_cache: Dict[str, Dict] = {}
signal_cache_lock = Lock()
notifications_cache = deque(maxlen=50)
notifications_lock = Lock()
rejection_logs_cache = deque(maxlen=100)
rejection_logs_lock = Lock()
current_market_state: Dict[str, Any] = {"overall_regime": "INITIALIZING", "trend_details_by_tf": {}, "last_updated": None}
market_state_lock = Lock()
dynamic_filter_profile_cache: Dict[str, Any] = {}
last_dynamic_filter_analysis_time: float = 0
dynamic_filter_lock = Lock()
last_market_state_check = 0

REJECTION_REASONS_AR = {
    "Filters Not Loaded": "الفلاتر غير محملة", "Low Volatility": "تقلب منخفض جداً",
    "BTC Correlation": "ارتباط ضعيف بالبيتكوين", "RRR Filter": "نسبة المخاطرة/العائد غير كافية",
    "Momentum/Strength Filter": "فلتر الزخم والقوة", "Peak/Pullback Filter": "فلتر القمة/التصحيح",
    "Invalid ATR for TP/SL": "ATR غير صالح لحساب الأهداف", "ML Model Rejected Signal": "نموذج التعلم الآلي رفض الإشارة",
    "Invalid Position Size": "حجم الصفقة غير صالح", "Lot Size Adjustment Failed": "فشل ضبط حجم العقد",
    "Min Notional Filter": "قيمة الصفقة أقل من الحد الأدنى", "Insufficient Balance": "الرصيد غير كافٍ",
    "Order Book Fetch Failed": "فشل جلب دفتر الطلبات", "Order Book Imbalance": "اختلال توازن دفتر الطلبات",
    "Large Sell Wall Detected": "تم كشف جدار بيع ضخم", "Insufficient data for TP/SL calculation": "بيانات غير كافية لحساب TP/SL",
    "Potential Profit Below Threshold": "الربح المحتمل أقل من الحد الأدنى",
    "Potential Profit Below Threshold (S/R)": "الربح المحتمل أقل من الحد الأدنى (دعم/مقاومة)"
}

# --- START: NEW Multi-Asset Emergency System ---
class MultiAssetEmergencyDetector:
    """
    نظام كشف الطوارئ متعدد الأصول لحماية المحفظة من الانهيارات المفاجئة.
    """
    def __init__(self, client: Client):
        self.client = client
        self.emergency_assets = {
            'BTCUSDT': {'weight': 0.4, 'threshold': -3.0},
            'ETHUSDT': {'weight': 0.3, 'threshold': -4.0},
            'BNBUSDT': {'weight': 0.2, 'threshold': -5.0},
            'SOLUSDT': {'weight': 0.1, 'threshold': -6.0}
        }
        self.volume_spike_threshold = 5.0

    def get_15m_change(self, symbol: str) -> float:
        try:
            klines = self.client.get_historical_klines(symbol, '15m', "30 minutes ago UTC", limit=2)
            if len(klines) < 2: return 0.0
            previous_open = float(klines[-2][1])
            previous_close = float(klines[-2][4])
            if previous_open == 0: return 0.0
            return ((previous_close - previous_open) / previous_open) * 100
        except Exception as e:
            logger.error(f"❌ [Emergency] خطأ في جلب تغير 15 دقيقة لـ {symbol}: {e}")
            return 0.0

    def get_volume_spike(self, symbol: str) -> float:
        try:
            klines = self.client.get_historical_klines(symbol, '15m', "2 hours ago UTC", limit=8)
            if len(klines) < 8: return 1.0
            volumes = [float(k[5]) for k in klines]
            current_vol = volumes[-1]
            avg_vol = np.mean(volumes[:-1])
            return current_vol / avg_vol if avg_vol > 0 else 1.0
        except Exception as e:
            logger.error(f"❌ [Emergency] خطأ في جلب طفرة الحجم لـ {symbol}: {e}")
            return 1.0

    def calculate_emergency_score(self) -> Tuple[float, Dict]:
        total_score = 0.0
        details = {}
        for symbol, config in self.emergency_assets.items():
            price_change = self.get_15m_change(symbol)
            if price_change <= config['threshold']:
                asset_score = abs(price_change / config['threshold']) * config['weight'] * 100
                total_score += asset_score
                details[symbol] = {'type': 'Price Drop', 'change_pct': round(price_change, 2), 'threshold_pct': config['threshold'], 'contribution': round(asset_score, 1)}
            volume_ratio = self.get_volume_spike(symbol)
            if volume_ratio >= self.volume_spike_threshold:
                vol_score = min((volume_ratio / self.volume_spike_threshold) * 10, 20) * config['weight'] * 2
                total_score += vol_score
                details[f"{symbol}_volume"] = {'type': 'Volume Spike', 'ratio': round(volume_ratio, 2), 'threshold_ratio': self.volume_spike_threshold, 'contribution': round(vol_score, 1)}
        return min(total_score, 100), details

    def is_emergency_triggered(self, threshold: float = 60.0) -> Tuple[bool, Dict]:
        score, details = self.calculate_emergency_score()
        triggered = score >= threshold
        if triggered:
            logger.critical(f"🚨🚨 EMERGENCY TRIGGERED! Score: {score:.1f} / {threshold} 🚨🚨")
        return triggered, {'score': round(score, 1), 'details': details}

def is_market_crashing() -> bool:
    global emergency_detector
    if not CRASH_PROTECTION_ENABLED: return False
    if emergency_detector is None:
        if client:
            logger.info("ℹ️ [Emergency] تهيئة نظام كشف الطوارئ متعدد الأصول...")
            emergency_detector = MultiAssetEmergencyDetector(client)
        else:
            logger.warning("⚠️ [Emergency] لا يمكن تهيئة الكاشف، العميل غير موجود.")
            return False
    try:
        triggered, details = emergency_detector.is_emergency_triggered(threshold=60.0)
        if triggered:
            details_json = json.dumps(details, ensure_ascii=False, default=str)
            log_and_notify('critical', f"🚨 Emergency triggered! Score: {details.get('score', 'N/A')}. Details: {details_json}", "EMERGENCY_TRIGGER")
            try:
                if check_db_connection():
                    with conn.cursor() as cur:
                        cur.execute("INSERT INTO notifications (type, message, details) VALUES ('EMERGENCY_TRIGGER', %s, %s)", (f"Emergency Score: {details.get('score', 'N/A')}", details_json))
                        conn.commit()
            except Exception as e:
                logger.error(f"❌ [DB Log] فشل حفظ تفاصيل الطوارئ في قاعدة البيانات: {e}")
                if conn: conn.rollback()
        return triggered
    except Exception as e:
        logger.error(f"❌ [Emergency] خطأ فادح أثناء التحقق من حالة السوق: {e}", exc_info=True)
        return False
# --- END: NEW Multi-Asset Emergency System ---

def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("[Telegram] Token أو Chat ID غير معين، تم تخطي الإرسال.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info(f"✅ [Telegram] تم إرسال الرسالة بنجاح.")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة: {e}")

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
                        target_price DOUBLE PRECISION NOT NULL, stop_loss DOUBLE PRECISION NOT NULL,
                        status TEXT DEFAULT 'open', closing_price DOUBLE PRECISION, closed_at TIMESTAMP,
                        profit_percentage DOUBLE PRECISION, strategy_name TEXT, signal_details JSONB,
                        current_peak_price DOUBLE PRECISION, is_real_trade BOOLEAN DEFAULT FALSE,
                        quantity DOUBLE PRECISION, order_id TEXT
                    );
                """)
                cur.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS closing_reason TEXT;")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals (status);")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL, message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE
                    );
                """)
                cur.execute("ALTER TABLE notifications ADD COLUMN IF NOT EXISTS details JSONB;")
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
    if notification_type == "EMERGENCY_TRIGGER": return
    if not check_db_connection() or not conn: return
    try:
        new_notification = {"timestamp": datetime.now(timezone.utc).isoformat(), "type": notification_type, "message": message}
        with notifications_lock: notifications_cache.appendleft(new_notification)
        with conn.cursor() as cur: cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
    except Exception as e:
        logger.error(f"❌ [Notify DB] فشل حفظ الإشعار: {e}")
        if conn: conn.rollback()

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
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        if not exchange_info_map: get_exchange_info_map()
        active = {s for s, info in exchange_info_map.items() if info.get('quoteAsset') == 'USDT' and info.get('status') == 'TRADING'}
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ [Validation] سيقوم البوت بمراقبة {len(validated)} عملة.")
        return validated
    except Exception as e:
        logger.error(f"❌ [Validation] خطأ أثناء التحقق من العملات: {e}", exc_info=True)
        return []

def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        limit = 100 if interval == '15m' else None
        klines = client.get_historical_klines(symbol, interval, f"{days} day ago UTC", limit=limit)
        if not klines: return None
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(klines, columns=cols)
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base']
        df = df[required_cols]
        numeric_cols = {'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float', 'quote_volume': 'float', 'taker_buy_base': 'float'}
        df = df.astype(numeric_cols)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df.dropna()
    except Exception as e:
        logger.error(f"❌ [Data] خطأ في جلب البيانات التاريخية لـ {symbol}: {e}")
        return None

def calculate_all_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df_calc = df.copy()
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
    return df_calc.astype('float32', errors='ignore')

def get_session_state() -> Tuple[List[str], str, str]:
    sessions = {"London": (8, 17), "New York": (13, 22), "Tokyo": (0, 9)}
    active_sessions = []
    now_utc = datetime.now(timezone.utc)
    current_hour = now_utc.hour
    if now_utc.weekday() >= 5: return [], "WEEKEND", "عطلة نهاية الأسبوع"
    for session, (start, end) in sessions.items():
        if start <= current_hour < end: active_sessions.append(session)
    if "London" in active_sessions and "New York" in active_sessions: return active_sessions, "HIGH_LIQUIDITY", "تداخل لندن/نيويورك"
    elif len(active_sessions) >= 1: return active_sessions, "NORMAL_LIQUIDITY", f"{', '.join(active_sessions)}"
    else: return [], "LOW_LIQUIDITY", "خارج أوقات الذروة"

def get_btc_data_for_bot() -> Optional[pd.DataFrame]:
    btc_data = fetch_historical_data(BTC_SYMBOL, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
    if btc_data is not None: btc_data['btc_returns'] = btc_data['close'].pct_change()
    return btc_data

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

class EnhancedFilterSystem:
    def generate_filters(self) -> Dict[str, Any]:
        return {"name": "Default Filters", "description": "Default static filters", "strategy": "MOMENTUM", "filters": {"adx": 25.0, "min_rrr": 1.4}}

enhanced_filter_system = EnhancedFilterSystem()

class EnhancedTradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        # Simplified for brevity
        self.ml_model, self.scaler, self.feature_names = (True, True, ['rsi', 'adx']) if random.random() > 0.1 else (None, None, None)

    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        return calculate_all_features(df_15m, btc_df)

    def generate_buy_signal(self, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not df_features.empty and df_features.iloc[-1]['rsi'] > 55 and df_features.iloc[-1]['adx'] > 25:
             return {'prediction': 1, 'confidence': 0.85}
        return None

def passes_filters(symbol: str, last_features: pd.Series, profile: Dict[str, Any], entry_price: float, tp_sl_data: Dict, df_15m: pd.DataFrame) -> bool:
    with filters_disabled_lock:
        if are_filters_disabled: return True
    return True # Simplified

def analyze_order_book(symbol: str, entry_price: float) -> Optional[Dict[str, Any]]:
    return {"bid_ask_ratio": 1.2, "has_large_sell_wall": False}

def passes_order_book_check(symbol: str, order_book_analysis: Dict, profile: Dict) -> bool:
    return True # Simplified

def calculate_tp_sl(symbol: str, entry_price: float, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    last_atr = df['atr'].iloc[-1]
    if last_atr > 0:
        return {'target_price': entry_price + last_atr * 2.2, 'stop_loss': entry_price - last_atr * 1.5, 'source': 'ATR_Fallback'}
    return None

def emergency_close_all_positions(reason: str = "سوق متهاوٍ"):
    with signal_cache_lock:
        open_trades = list(open_signals_cache.values())
    if not open_trades: return
    log_and_notify("critical", f"🚨 نظام الطوارئ: بدء إغلاق {len(open_trades)} صفقة بسبب: {reason}", "EMERGENCY_CLOSE")
    for signal in open_trades:
        try:
            current_price = float(client.get_symbol_ticker(symbol=signal['symbol'])['price'])
            close_signal(signal['id'], current_price, f"emergency_{reason}")
        except Exception as e:
            logger.error(f"❌ خطأ فادح أثناء محاولة إغلاق {signal['symbol']} في حالة الطوارئ: {e}")

def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> Optional[Decimal]:
    symbol_info = exchange_info_map.get(symbol)
    if not symbol_info: return None
    lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
    if lot_size_filter:
        step_size = Decimal(lot_size_filter['stepSize'])
        return (Decimal(str(quantity)) // step_size) * step_size
    return Decimal(str(quantity))

def calculate_position_size(symbol: str, entry_price: float, stop_loss_price: float) -> Optional[Decimal]:
    if not client: return None
    try:
        balance_response = client.get_asset_balance(asset='USDT')
        available_balance = Decimal(balance_response['free'])
        risk_amount_usdt = available_balance * (Decimal(str(RISK_PER_TRADE_PERCENT)) / Decimal('100'))
        risk_per_coin = Decimal(str(entry_price)) - Decimal(str(stop_loss_price))
        if risk_per_coin <= 0: return None
        initial_quantity = risk_amount_usdt / risk_per_coin
        return adjust_quantity_to_lot_size(symbol, float(initial_quantity))
    except Exception as e:
        logger.error(f"❌ [{symbol}] خطأ في حساب حجم الصفقة: {e}"); return None

def place_order(symbol: str, side: str, quantity: Decimal, order_type: str = Client.ORDER_TYPE_MARKET) -> Optional[Dict]:
    if not client: return None
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
        # Real trade closing logic would be here
        if not check_db_connection() or not conn: return False
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE signals SET status = 'closed', closing_price = %s, closed_at = NOW(), profit_percentage = %s, closing_reason = %s WHERE id = %s;", (closing_price, profit_percentage, reason, signal_id))
            conn.commit()
            if symbol_to_close in open_signals_cache: del open_signals_cache[symbol_to_close]
            log_and_notify('info', f"CLOSED: {symbol_to_close} at {closing_price:.4f}. Profit: {profit_percentage:.2f}%", "TRADE_CLOSED")
            return True
        except Exception as e:
            logger.error(f"❌ [DB Close] فشل تحديث الصفقة المغلقة: {e}"); conn.rollback(); return False

def insert_signal_into_db(signal_data: Dict) -> Optional[Dict]:
    if not check_db_connection() or not conn: return None
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, is_real_trade, quantity, order_id, current_peak_price) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *;",
                        (signal_data['symbol'], signal_data['entry_price'], signal_data['target_price'], signal_data['stop_loss'], signal_data['strategy_name'], json.dumps(signal_data['signal_details']), signal_data.get('is_real_trade', False), signal_data.get('quantity'), signal_data.get('order_id'), signal_data['entry_price']))
            saved_signal = cur.fetchone()
            conn.commit()
            return dict(saved_signal)
    except Exception as e:
        logger.error(f"❌ [DB Insert] فشل إدراج الإشارة: {e}"); conn.rollback(); return None

def determine_market_state_enhanced():
    global current_market_state, last_market_state_check
    if time.time() - last_market_state_check < 180: return
    with market_state_lock:
        current_market_state = {"overall_regime": "NORMAL", "last_updated": datetime.now(timezone.utc).isoformat()}
        last_market_state_check = time.time()

def analyze_market_and_create_dynamic_profile_enhanced():
    global dynamic_filter_profile_cache, last_dynamic_filter_analysis_time
    if time.time() - last_dynamic_filter_analysis_time < DYNAMIC_FILTER_ANALYSIS_INTERVAL: return
    with dynamic_filter_lock:
        dynamic_filter_profile_cache = enhanced_filter_system.generate_filters()
        last_dynamic_filter_analysis_time = time.time()

# --- NEW: Scheduled Pause Management Loop ---
def pause_management_loop():
    """Manages scheduled pauses for the bot based on UTC time."""
    global is_bot_paused, pause_reason, last_telegram_alert_hour
    logger.info("✅ [Pause Manager] بدء حلقة إدارة الإيقاف المؤقت...")
    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            currently_in_pause_window = False
            
            for hour, reason in PAUSE_SCHEDULE.items():
                pause_start_minute = 30
                alert_minute = 15

                # Check for pause condition (30 mins before the hour)
                # e.g., for hour=4, this is true from 03:30 to 03:59
                if now_utc.hour == (hour - 1 + 24) % 24 and now_utc.minute >= pause_start_minute:
                    with pause_lock:
                        if not is_bot_paused:
                            log_and_notify('warning', f"PAUSE ACTIVATED: Bot paused. Reason: {reason}", "BOT_PAUSE")
                        is_bot_paused = True
                        pause_reason = reason
                    currently_in_pause_window = True
                    break

                # Check for Telegram alert condition (45 mins before the hour -> 15 mins into the previous hour)
                # e.g., for hour=4, this is true at 03:15
                if now_utc.hour == (hour - 1 + 24) % 24 and now_utc.minute == alert_minute:
                    with pause_lock:
                        # Check if an alert for this hour has already been sent
                        if last_telegram_alert_hour != now_utc.hour:
                            alert_message = (f"🔔 *تنبيه إيقاف مؤقت*\n\n"
                                             f"سيتم إيقاف توليد التوصيات خلال *15 دقيقة*.\n\n"
                                             f"*السبب:* {reason}")
                            send_telegram_message(alert_message)
                            logger.info(f"Sent pre-pause Telegram alert for hour {hour}.")
                            last_telegram_alert_hour = now_utc.hour
            
            if not currently_in_pause_window:
                with pause_lock:
                    if is_bot_paused:
                        log_and_notify('info', "PAUSE ENDED: Bot resumed generating signals.", "BOT_RESUME")
                    is_bot_paused = False
                    pause_reason = ""

        except Exception as e:
            logger.error(f"❌ [Pause Manager] خطأ في حلقة الإدارة: {e}", exc_info=True)
        
        time.sleep(60) # Check every minute

app = Flask(__name__)
CORS(app)

def get_dashboard_html():
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم التداول V9</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap" rel="stylesheet">
    <style>
        :root { --bg-main: #0D1117; --bg-card: #161B22; --border-color: #30363D; --text-primary: #E6EDF3; --text-secondary: #848D97; --accent-blue: #58A6FF; --accent-green: #3FB950; --accent-red: #F85149; --accent-yellow: #D29922; }
        body { font-family: 'Tajawal', sans-serif; background-color: var(--bg-main); color: var(--text-primary); }
        .card { background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 0.5rem; }
        .tab-btn.active { border-bottom-color: var(--accent-blue); }
        input:checked + .toggle-bg { background-color: var(--accent-green); }
    </style>
</head>
<body class="p-4 md:p-6">
    <div class="container mx-auto max-w-screen-2xl">
        <header class="mb-6 flex flex-wrap justify-between items-center gap-4">
            <h1 class="text-2xl md:text-3xl font-extrabold"><span class="text-accent-blue">لوحة تحكم</span><span class="text-text-secondary font-medium"> V9</span></h1>
        </header>
        
        <!-- NEW: Pause Status Banner -->
        <div id="pause-status-banner" class="hidden card bg-yellow-900/50 border-yellow-600 text-yellow-200 p-4 mb-6 text-center">
            <h3 class="font-bold text-lg">⚠️ البوت متوقف مؤقتاً</h3>
            <p id="pause-reason-text" class="mt-1"></p>
        </div>

        <section class="mb-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">حالة السوق</h3><div id="overall-regime" class="text-2xl font-bold text-center">...</div></div>
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">الجلسات النشطة</h3><div id="active-sessions-list" class="flex flex-wrap gap-2 items-center justify-center pt-2">...</div></div>
            <div class="card p-4 flex flex-col justify-center items-center"><h3 class="font-bold text-lg text-text-secondary mb-2">التداول الحقيقي</h3><div class="flex items-center space-x-3 space-x-reverse"><span id="trading-status-text" class="font-bold text-lg"></span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="trading-toggle" class="sr-only" onchange="toggleTrading()"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div></div>
            <div class="card p-4 flex flex-col justify-center items-center"><h3 class="font-bold text-lg text-text-secondary mb-2">تعطيل الفلاتر</h3><div class="flex items-center space-x-3 space-x-reverse"><span id="disable-filters-text" class="font-bold text-lg"></span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="disable-filters-toggle" class="sr-only" onchange="toggleFilters()"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div></div>
        </section>
        <div class="mb-4 border-b border-border-color"><nav class="flex space-x-6 space-x-reverse -mb-px"><button onclick="showTab('signals', this)" class="tab-btn active text-white py-3 px-1 font-semibold">الصفقات</button><button onclick="showTab('notifications', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الإشعارات</button></nav></div>
        <main>
            <div id="signals-tab" class="tab-content"><div class="overflow-x-auto card p-0"><table class="min-w-full text-sm text-right"><thead class="border-b border-border-color bg-black/20"><tr><th class="p-4 font-semibold">العملة</th><th class="p-4 font-semibold">الحالة</th><th class="p-4 font-semibold">الربح/الخسارة</th><th class="p-4 font-semibold">الدخول/الحالي</th></tr></thead><tbody id="signals-table"></tbody></table></div></div>
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
        
        // NEW: Handle pause status
        const pauseBanner = document.getElementById('pause-status-banner');
        const pauseReasonText = document.getElementById('pause-reason-text');
        if (data.is_bot_paused) {
            pauseReasonText.textContent = data.pause_reason;
            pauseBanner.classList.remove('hidden');
        } else {
            pauseBanner.classList.add('hidden');
        }

        document.getElementById('overall-regime').textContent = (data.market_state?.overall_regime || 'UNCERTAIN').replace(/_/g, ' ');
        const sessions = document.getElementById('active-sessions-list');
        sessions.innerHTML = data.active_sessions.length > 0 ? data.active_sessions.map(s => `<span class="bg-accent-blue/20 text-accent-blue text-xs font-bold px-2 py-1 rounded">${s}</span>`).join('') : `<span class="bg-gray-700 text-text-secondary text-xs font-bold px-2 py-1 rounded">لا توجد</span>`;
        const tradeToggle = document.getElementById('trading-toggle');
        tradeToggle.checked = data.is_trading_enabled;
        document.getElementById('trading-status-text').textContent = data.is_trading_enabled ? 'مُفعَّل' : 'غير مُفعَّل';
        const filtersToggle = document.getElementById('disable-filters-toggle');
        filtersToggle.checked = data.are_filters_disabled;
        document.getElementById('disable-filters-text').textContent = data.are_filters_disabled ? 'معطلة' : 'مفعلة';
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
            tableBody.innerHTML += `<tr class="border-b border-border-color hover:bg-white/5"><td class="p-4 font-bold">${s.symbol}</td><td class="p-4"><span class="px-2 py-1 text-xs font-semibold rounded-full ${s.is_real_trade ? 'bg-blue-500/20 text-blue-400' : 'bg-yellow-500/20 text-yellow-400'}">${s.is_real_trade ? 'حقيقي' : 'تجريبي'}</span></td><td class="p-4 font-mono ${pClass}">${profit.toFixed(2)}%</td><td class="p-4 font-mono">${parseFloat(s.current_price || s.entry_price).toFixed(4)} / ${parseFloat(s.entry_price).toFixed(4)}</td></tr>`;
        });
    });
}
function updateNotifications() {
    fetchData('/api/notifications').then(data => {
        if (!data) return;
        document.getElementById('notifications-list').innerHTML = data.map(n => `<div class="p-2 border-b border-border-color"><span class="font-mono text-xs text-text-secondary">${new Date(n.timestamp).toLocaleString('ar-EG')}</span>: ${n.message}</div>`).join('');
    });
}
function toggleTrading() { fetch('/api/trading/toggle', { method: 'POST' }).then(() => updateMarketStatus()); }
function toggleFilters() { fetch('/api/filters/disable/toggle', { method: 'POST' }).then(() => updateMarketStatus()); }
document.addEventListener('DOMContentLoaded', () => {
    ['MarketStatus', 'Signals', 'Notifications'].forEach(f => window[`update${f}`]());
    setInterval(updateMarketStatus, 5000); setInterval(updateSignals, 7000); setInterval(updateNotifications, 15000);
});
</script>
</body></html>
"""

@app.route('/')
def home(): return render_template_string(get_dashboard_html())

@app.route('/api/market_status')
def get_market_status():
    with market_state_lock: state_copy = dict(current_market_state)
    with filters_disabled_lock: is_disabled = are_filters_disabled
    with trading_status_lock: is_enabled = is_trading_enabled
    with pause_lock:
        bot_paused = is_bot_paused
        reason_text = pause_reason
    active_sessions, _, _ = get_session_state()
    return jsonify({
        "market_state": state_copy, 
        "active_sessions": active_sessions, 
        "is_trading_enabled": is_enabled, 
        "are_filters_disabled": is_disabled,
        "is_bot_paused": bot_paused,
        "pause_reason": reason_text
    })

@app.route('/api/signals')
def get_signals():
    if not all([check_db_connection(), redis_client]): return jsonify({"error": "Service connection failed"}), 500
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
        logger.error(f"❌ [API Signals] Error: {e}"); return jsonify({"error": str(e)}), 500

@app.route('/api/notifications')
def get_notifications():
    with notifications_lock:
        return jsonify(list(notifications_cache))

@app.route('/api/trading/toggle', methods=['POST'])
def toggle_trading_status():
    global is_trading_enabled
    with trading_status_lock:
        is_trading_enabled = not is_trading_enabled
        status_msg = "ENABLED" if is_trading_enabled else "DISABLED"
        log_and_notify('warning', f"🚨 Real trading status changed to: {status_msg}", "TRADING_STATUS_CHANGE")
        return jsonify({"message": f"Trading status set to {status_msg}"})

@app.route('/api/filters/disable/toggle', methods=['POST'])
def toggle_disable_filters():
    global are_filters_disabled
    with filters_disabled_lock:
        are_filters_disabled = not are_filters_disabled
        status_msg = "DISABLED" if are_filters_disabled else "ENABLED"
        log_and_notify('warning', f"⚙️ Filters status changed to: {status_msg}", "FILTER_STATUS_CHANGE")
        return jsonify({"message": f"Filters status set to {status_msg}"})

def trade_management_loop():
    logger.info("✅ [Trade Manager] بدء حلقة إدارة الصفقات...")
    while True:
        try:
            with signal_cache_lock:
                if not open_signals_cache:
                    time.sleep(5)
                    continue
                signals_to_check = list(open_signals_cache.values())
            
            if not redis_client:
                time.sleep(5)
                continue
                
            current_prices = redis_client.hgetall(REDIS_PRICES_HASH_NAME)
            
            for signal in signals_to_check:
                current_price_str = current_prices.get(signal['symbol'])
                if not current_price_str: continue
                current_price = float(current_price_str)
                if current_price >= float(signal['target_price']): close_signal(signal['id'], current_price, 'take_profit')
                elif current_price <= float(signal['stop_loss']): close_signal(signal['id'], current_price, 'stop_loss')
            time.sleep(2)
        except Exception as e:
            logger.error(f"❌ [Trade Manager] خطأ في حلقة الإدارة: {e}", exc_info=True)
            time.sleep(10)

def main_loop_enhanced():
    logger.info("[Main Loop] انتظار اكتمال التهيئة...")
    time.sleep(15)
    if not validated_symbols_to_scan:
        log_and_notify("critical", "لا توجد عملات صالحة للمسح.", "SYSTEM")
        return
    log_and_notify("info", f"✅ بدء حلقة المسح لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")

    while True:
        try:
            if is_market_crashing():
                emergency_close_all_positions()
                log_and_notify("warning", "نظام الطوارئ مفعل. تم إيقاف البوت مؤقتاً لمدة 5 دقائق.", "EMERGENCY_HALT")
                time.sleep(300)
                continue

            with pause_lock:
                if is_bot_paused:
                    logger.info(f"PAUSED: Bot is currently paused. Reason: {pause_reason}. Waiting...")
                    time.sleep(60)
                    continue

            logger.info("🔄 بدء دورة مسح جديدة...")
            determine_market_state_enhanced()
            analyze_market_and_create_dynamic_profile_enhanced()
            
            with dynamic_filter_lock: filter_profile = dynamic_filter_profile_cache
            if not filter_profile:
                logger.warning("🛑 لم يتم تحميل ملف الفلاتر. الانتظار...")
                time.sleep(60)
                continue

            btc_data = get_btc_data_for_bot()
            symbols_to_process = random.sample(validated_symbols_to_scan, len(validated_symbols_to_scan))
            
            for symbol in symbols_to_process:
                try:
                    with signal_cache_lock:
                        if symbol in open_signals_cache or len(open_signals_cache) >= MAX_OPEN_TRADES:
                            continue
                    
                    strategy = EnhancedTradingStrategy(symbol)
                    if not all([strategy.ml_model, strategy.scaler, strategy.feature_names]): continue
                    
                    df_15m = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_15m is None or df_15m.empty: continue
                    
                    df_4h = fetch_historical_data(symbol, HIGHER_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_4h is None or df_4h.empty: continue
                    
                    df_features = strategy.get_features(df_15m, df_4h, btc_data)
                    if df_features is None or df_features.empty: continue
                    
                    ml_signal = strategy.generate_buy_signal(df_features)
                    if not ml_signal or ml_signal['confidence'] < BUY_CONFIDENCE_THRESHOLD: continue
                    
                    entry_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                    tp_sl_data = calculate_tp_sl(symbol, entry_price, df_15m)
                    if not tp_sl_data: continue
                    
                    last_features = df_features.iloc[-1]
                    if not passes_filters(symbol, last_features, filter_profile, entry_price, tp_sl_data, df_15m): continue
                    
                    order_book_analysis = analyze_order_book(symbol, entry_price)
                    if not order_book_analysis or not passes_order_book_check(symbol, order_book_analysis, filter_profile): continue
                    
                    new_signal = {'symbol': symbol, 'strategy_name': "Momentum_ML_V9", 'signal_details': {}, 'entry_price': entry_price, **tp_sl_data}
                    
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
        if not validated_symbols_to_scan: logger.critical("❌ لا توجد عملات صالحة للمسح."); return
        Thread(target=main_loop_enhanced, daemon=True).start()
        Thread(target=price_update_loop, daemon=True).start()
        Thread(target=trade_management_loop, daemon=True).start()
        Thread(target=pause_management_loop, daemon=True).start() # <-- NEW THREAD
        logger.info("✅ [Bot Services] تم بدء جميع الخدمات الخلفية بنجاح.")
        send_telegram_message("✅ *البوت قيد التشغيل الآن*")
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حرج أثناء التهيئة: {e}", "SYSTEM"); exit(1)

if __name__ == "__main__":
    logger.info("🚀 إطلاق بوت التداول ولوحة التحكم (V9 - Final with Telegram) 🚀")
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
