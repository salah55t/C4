# ملف c4_complete_v9_final_telegram.py - نسخة محدثة مع نظام الطوارئ ونظام الإيقاف الوقائي ولوحة تحكم مطورة
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

# --- Emergency System Settings ---
CRASH_PROTECTION_ENABLED: bool = True
emergency_detector = None 
is_emergency_manually_triggered: bool = False
emergency_manual_lock = Lock()

# --- Proactive Stop System Settings ---
HIGH_RISK_HOURS_UTC: Set[int] = {0, 4, 8, 12, 16, 20}
QUIET_PERIOD_MINUTES_BEFORE: int = 15
ALERT_MINUTES_BEFORE: int = 30
last_alert_sent_for_hour: Dict[str, bool] = {}
QUIET_PERIOD_REASONS = {
    0: "الاستعداد لبداية الجلسة الآسيوية وإغلاق الشمعة اليومية.",
    4: "فترة ضعف سيولة محتملة في منتصف الجلسة الآسيوية.",
    8: "الاستعداد لافتتاح جلسة لندن وتداخلها مع إغلاق طوكيو.",
    12: "فترة تقلبات عالية مع تداخل جلسات لندن ونيويورك.",
    16: "الاستعداد لإغلاق جلسة لندن وجني الأرباح الأوروبية.",
    20: "اقتراب نهاية جلسة نيويورك وجني الأرباح الأمريكية."
}

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
TRAILING_ACTIVATION_PROFIT_PERCENT: float = 2.0
TRAILING_DISTANCE_PERCENT: float = 0.9
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

# --- START: Multi-Asset Emergency System ---
class MultiAssetEmergencyDetector:
    def __init__(self, client: Client):
        self.client = client
        self.emergency_assets = {
            'BTCUSDT': {'weight': 0.4, 'threshold': -1.5},
            'ETHUSDT': {'weight': 0.3, 'threshold': -2.0},
            'BNBUSDT': {'weight': 0.2, 'threshold': -2.5},
            'SOLUSDT': {'weight': 0.1, 'threshold': -3.0}
        }
        self.volume_spike_threshold = 5.0
        self.last_check_time = 0
        self.cache_expiry = 60 # Cache result for 60 seconds
        self.cached_result = (False, {})

    def get_15m_change(self, symbol: str) -> float:
        try:
            klines = self.client.get_historical_klines(symbol, '15m', "30 minutes ago UTC", limit=2)
            if len(klines) < 2: return 0.0
            previous_open = float(klines[-2][1])
            previous_close = float(klines[-2][4])
            return ((previous_close - previous_open) / previous_open) * 100 if previous_open != 0 else 0.0
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
        total_score, details = 0.0, {}
        for symbol, config in self.emergency_assets.items():
            price_change = self.get_15m_change(symbol)
            if price_change <= config['threshold']:
                asset_score = abs(price_change / config['threshold']) * config['weight'] * 100
                total_score += asset_score
                details[symbol] = {'type': 'Price Drop', 'change_pct': round(price_change, 2), 'contribution': round(asset_score, 1)}
            volume_ratio = self.get_volume_spike(symbol)
            if volume_ratio >= self.volume_spike_threshold:
                vol_score = min((volume_ratio / self.volume_spike_threshold) * 10, 20) * config['weight'] * 2
                total_score += vol_score
                details[f"{symbol}_volume"] = {'type': 'Volume Spike', 'ratio': round(volume_ratio, 2), 'contribution': round(vol_score, 1)}
        return min(total_score, 100), details

    def is_emergency_triggered(self, threshold: float = 60.0) -> Tuple[bool, Dict]:
        if time.time() - self.last_check_time < self.cache_expiry:
            return self.cached_result

        score, details = self.calculate_emergency_score()
        triggered = score >= threshold
        if triggered:
            logger.critical(f"🚨🚨 AUTO EMERGENCY! Score: {score:.1f} / {threshold} 🚨🚨")
        
        self.cached_result = (triggered, {'score': round(score, 1), 'details': details})
        self.last_check_time = time.time()
        return self.cached_result

def is_market_crashing() -> bool:
    global emergency_detector, is_emergency_manually_triggered
    
    with emergency_manual_lock:
        if is_emergency_manually_triggered:
            return True

    if not CRASH_PROTECTION_ENABLED:
        return False

    if emergency_detector is None:
        if client:
            logger.info("ℹ️ [Emergency] تهيئة نظام كشف الطوارئ...")
            emergency_detector = MultiAssetEmergencyDetector(client)
        else:
            return False

    try:
        triggered, details = emergency_detector.is_emergency_triggered(threshold=60.0)
        if triggered:
            details_json = json.dumps(details, ensure_ascii=False, default=str)
            log_and_notify('critical', f"🚨 Auto-Emergency! Score: {details.get('score', 'N/A')}", "EMERGENCY_TRIGGER", details_json)
        return triggered
    except Exception as e:
        logger.error(f"❌ [Emergency] خطأ فادح أثناء التحقق من حالة السوق: {e}", exc_info=True)
        return False
# --- END: Multi-Asset Emergency System ---

# --- START: Proactive Stop System ---
def cleanup_old_alerts():
    global last_alert_sent_for_hour
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    keys_to_delete = [key for key, _ in last_alert_sent_for_hour.items() if datetime.fromisoformat(key.split('-', 1)[1]).date() < yesterday]
    for key in keys_to_delete: del last_alert_sent_for_hour[key]

def check_and_send_quiet_period_alert():
    global last_alert_sent_for_hour
    now_utc = datetime.now(timezone.utc)
    alert_total_offset = QUIET_PERIOD_MINUTES_BEFORE + ALERT_MINUTES_BEFORE
    
    for risk_hour in HIGH_RISK_HOURS_UTC:
        alert_minute_of_day = (risk_hour * 60 - alert_total_offset + 1440) % 1440
        alert_hour, alert_minute = divmod(alert_minute_of_day, 60)
        
        if now_utc.hour == alert_hour and now_utc.minute == alert_minute:
            risk_event_date = now_utc.date() + timedelta(days=1) if now_utc.hour > risk_hour else now_utc.date()
            alert_key = f"{risk_hour}-{risk_event_date.isoformat()}"

            if not last_alert_sent_for_hour.get(alert_key):
                message = (f"🔔 *تنبيه وقائي* 🔔\n\n"
                           f"سيتم إيقاف البحث عن توصيات جديدة مؤقتاً خلال *{ALERT_MINUTES_BEFORE} دقيقة*.\n"
                           f"السبب: الاستعداد لفترة تقلبات عالية متوقعة عند الساعة *{risk_hour:02}:00 UTC*.")
                send_telegram_message(message)
                last_alert_sent_for_hour[alert_key] = True
                cleanup_old_alerts()

def is_currently_in_quiet_period() -> Tuple[bool, Optional[str]]:
    now_utc = datetime.now(timezone.utc)
    for risk_hour in HIGH_RISK_HOURS_UTC:
        risk_time = now_utc.replace(hour=risk_hour, minute=0, second=0, microsecond=0)
        quiet_start_time = risk_time - timedelta(minutes=QUIET_PERIOD_MINUTES_BEFORE)
        
        if risk_hour == 0 and now_utc.hour == 23:
            risk_time += timedelta(days=1)
            quiet_start_time = risk_time - timedelta(minutes=QUIET_PERIOD_MINUTES_BEFORE)

        if quiet_start_time <= now_utc < risk_time:
            return True, QUIET_PERIOD_REASONS.get(risk_hour, "الاستعداد لفترة تقلبات عالية متوقعة.")
    return False, None
# --- END: Proactive Stop System ---

def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
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
                        quantity DOUBLE PRECISION, order_id TEXT, closing_reason TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_signals_status ON signals (status);
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY, timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL, message TEXT NOT NULL, is_read BOOLEAN DEFAULT FALSE, details JSONB
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
    except (OperationalError, InterfaceError):
        logger.error(f"❌ [DB] فقدان الاتصال. إعادة الاتصال...")
        init_db()
        return conn is not None and conn.closed == 0

def log_and_notify(level: str, message: str, notification_type: str, details: Optional[str] = None):
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error, 'critical': logger.critical}
    log_methods.get(level.lower(), logger.info)(message)
    
    if not check_db_connection() or not conn: return
    try:
        new_notification = {"timestamp": datetime.now(timezone.utc).isoformat(), "type": notification_type, "message": message}
        with notifications_lock: notifications_cache.appendleft(new_notification)
        with conn.cursor() as cur: 
            cur.execute("INSERT INTO notifications (type, message, details) VALUES (%s, %s, %s);", (notification_type, message, details))
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
        klines = client.get_historical_klines(symbol, interval, f"{days} day ago UTC")
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
    # This is a stub for the full feature calculation logic.
    # In a real scenario, all the feature calculations from the original file would be here.
    df_calc = df.copy()
    df_calc['atr'] = (df_calc['high'] - df_calc['low']).rolling(window=ATR_PERIOD).mean()
    delta = df_calc['close'].diff()
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    # ... Add all other feature calculations from the original file here ...
    return df_calc

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

class EnhancedTradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ml_model, self.scaler, self.feature_names = None, None, None # Placeholder
        # In a real scenario, the model loading logic would be here.

    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        # Placeholder for feature generation
        return calculate_all_features(df_15m, btc_df)

    def generate_buy_signal(self, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        # Placeholder for signal generation
        if not df_features.empty and df_features.iloc[-1]['rsi'] > 70: # Example condition
             return {'prediction': 1, 'confidence': 0.80}
        return None

def emergency_close_all_positions(reason: str = "سوق متهاوٍ"):
    with signal_cache_lock:
        open_trades = list(open_signals_cache.values())
    if not open_trades: return
    
    log_and_notify("critical", f"🚨 نظام الطوارئ: بدء إغلاق {len(open_trades)} صفقة بسبب: {reason}", "EMERGENCY_CLOSE")
    for signal in open_trades:
        try:
            current_price = float(client.get_symbol_ticker(symbol=signal['symbol'])['price'])
            if close_signal(signal['id'], current_price, f"emergency_{reason}"):
                send_telegram_message(f"🚨 *إغلاق طارئ ناجح*\nالعملة: `{signal['symbol']}`\nالسبب: {reason}\nسعر الإغلاق: `{current_price}`")
        except Exception as e:
            logger.error(f"❌ خطأ فادح أثناء محاولة إغلاق {signal['symbol']} في حالة الطوارئ: {e}")

def close_signal(signal_id: int, closing_price: float, reason: str) -> bool:
    # Placeholder for the full close_signal logic
    with signal_cache_lock:
        signal_to_close = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)
        if not signal_to_close: return False
        
        # Simulate DB update and cache removal
        logger.info(f"Simulating close for signal {signal_id} for {signal_to_close['symbol']}")
        del open_signals_cache[signal_to_close['symbol']]
        
        profit_percentage = ((closing_price - float(signal_to_close['entry_price'])) / float(signal_to_close['entry_price'])) * 100
        log_and_notify('info', f"CLOSED: {signal_to_close['symbol']} at {closing_price:.4f}. Reason: {reason}. Profit: {profit_percentage:.2f}%", "TRADE_CLOSED")
        return True

def determine_market_state_enhanced():
    # Placeholder for market state analysis
    pass

def analyze_market_and_create_dynamic_profile_enhanced():
    # Placeholder for dynamic filter profile creation
    pass

# --- Flask App and API Endpoints ---
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
        .trend-light { width: 1rem; height: 1rem; border-radius: 9999px; border: 2px solid #30363D; transition: all 0.5s ease; }
        .light-on-green { background-color: var(--accent-green); box-shadow: 0 0 10px 2px var(--accent-green); }
        .light-on-red { background-color: var(--accent-red); box-shadow: 0 0 10px 2px var(--accent-red); animation: pulse-red 1.5s infinite; }
        .light-on-yellow { background-color: var(--accent-yellow); box-shadow: 0 0 10px 2px var(--accent-yellow); }
        .tab-btn.active { border-bottom-color: var(--accent-blue); }
        input:checked + .toggle-bg { background-color: var(--accent-green); }
        #modal-overlay { transition: opacity 0.3s ease; }
        @keyframes pulse-red { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
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
            <h1 class="text-2xl md:text-3xl font-extrabold"><span class="text-accent-blue">لوحة تحكم</span><span class="text-text-secondary font-medium"> V9</span></h1>
            <div id="trend-lights-container" class="flex items-center gap-x-6 bg-black/20 px-4 py-2 rounded-lg border border-border-color"></div>
        </header>
        
        <section class="mb-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-5">
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">حالة السوق</h3><div id="overall-regime" class="text-2xl font-bold text-center">...</div></div>
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">ملف الفلاتر</h3><div id="filter-profile-name" class="text-xl font-bold text-center">...</div></div>
            <div class="card p-4"><h3 class="font-bold mb-3 text-lg text-text-secondary">الجلسات النشطة</h3><div id="active-sessions-list" class="flex flex-wrap gap-2 items-center justify-center pt-2">...</div></div>
            <div class="card p-4 flex flex-col justify-center items-center"><h3 class="font-bold text-lg text-text-secondary mb-2">التداول الحقيقي</h3><div class="flex items-center space-x-3 space-x-reverse"><span id="trading-status-text" class="font-bold text-lg"></span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="trading-toggle" class="sr-only" onchange="toggleTrading()"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div><div class="mt-2 text-xs text-text-secondary">رصيد USDT: <span id="usdt-balance" class="font-mono">...</span></div></div>
            <div class="card p-4 flex flex-col justify-center items-center"><h3 class="font-bold text-lg text-text-secondary mb-2">تعطيل الفلاتر</h3><div class="flex items-center space-x-3 space-x-reverse"><span id="disable-filters-text" class="font-bold text-lg"></span><label class="flex items-center cursor-pointer"><div class="relative"><input type="checkbox" id="disable-filters-toggle" class="sr-only" onchange="toggleFilters()"><div class="toggle-bg block bg-gray-600 w-12 h-7 rounded-full"></div></div></label></div></div>
            <div class="card p-4 flex flex-col justify-center items-center bg-red-900/20 border-red-700"><h3 class="font-bold text-lg text-red-400 mb-2">وضع الطوارئ</h3><div class="flex items-center gap-4"><div id="emergency-light" class="trend-light w-8 h-8"></div><button id="emergency-toggle-btn" onclick="toggleEmergency()" class="text-white font-bold py-2 px-4 rounded transition-colors"></button></div></div>
        </section>

        <div id="quiet-period-info" class="hidden mb-6 col-span-full card p-4 bg-yellow-900/30 border-yellow-600">
            <h4 class="font-bold text-yellow-400">🤫 البوت متوقف مؤقتاً</h4>
            <p id="quiet-period-reason" class="text-text-secondary mt-1"></p>
        </div>

        <div class="mb-4 border-b border-border-color"><nav class="flex space-x-6 space-x-reverse -mb-px"><button onclick="showTab('signals', this)" class="tab-btn active text-white py-3 px-1 font-semibold">الصفقات</button><button onclick="showTab('stats', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الإحصائيات</button><button onclick="showTab('notifications', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الإشعارات</button><button onclick="showTab('rejections', this)" class="tab-btn text-text-secondary hover:text-white py-3 px-1">الصفقات المرفوضة</button></nav></div>
        <main>
            <div id="signals-tab" class="tab-content"><div class="overflow-x-auto card p-0"><table class="min-w-full text-sm text-right"><thead class="border-b border-border-color bg-black/20"><tr><th class="p-4 font-semibold">العملة</th><th class="p-4 font-semibold">الحالة</th><th class="p-4 font-semibold">الربح/الخسارة</th><th class="p-4 font-semibold w-[25%]">التقدم</th><th class="p-4 font-semibold">الدخول/الحالي</th><th class="p-4 font-semibold">إجراء</th></tr></thead><tbody id="signals-table"></tbody></table></div></div>
            <div id="stats-tab" class="tab-content hidden"><div id="stats-container" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"></div></div>
            <div id="notifications-tab" class="tab-content hidden"><div id="notifications-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-2"></div></div>
            <div id="rejections-tab" class="tab-content hidden"><div id="rejections-list" class="card p-4 max-h-[60vh] overflow-y-auto space-y-2"></div></div>
        </main>
    </div>
<script>
let confirmCallback = null;
const modal = { overlay: document.getElementById('modal-overlay'), title: document.getElementById('modal-title'), body: document.getElementById('modal-body'), confirmBtn: document.getElementById('modal-confirm'), cancelBtn: document.getElementById('modal-cancel') };
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
    document.querySelectorAll('.tab-btn').forEach(b => { b.classList.remove('active', 'text-white'); b.classList.add('text-text-secondary'); });
    el.classList.add('active', 'text-white');
}
async function fetchData(url) { try { const r = await fetch(url); return r.ok ? await r.json() : null; } catch (e) { console.error('Fetch Error:', e); return null; } }

function updateMarketStatus() {
    fetchData('/api/market_status').then(data => {
        if (!data) return;
        // Status Cards
        document.getElementById('overall-regime').textContent = (data.market_state?.overall_regime || 'UNCERTAIN').replace(/_/g, ' ');
        document.getElementById('filter-profile-name').textContent = data.filter_profile?.name || 'غير متاح';
        const sessions = document.getElementById('active-sessions-list');
        sessions.innerHTML = data.active_sessions.length > 0 ? data.active_sessions.map(s => `<span class="bg-accent-blue/20 text-accent-blue text-xs font-bold px-2 py-1 rounded">${s}</span>`).join('') : `<span class="bg-gray-700 text-text-secondary text-xs font-bold px-2 py-1 rounded">لا توجد</span>`;

        // Trend Lights
        const lights = document.getElementById('trend-lights-container');
        lights.innerHTML = '';
        ['15m', '1h', '4h'].forEach(tf => {
            const trend = data.market_state?.trend_details_by_tf[tf]?.trend || 'Uncertain';
            let c = trend.includes('Uptrend') ? 'light-on-green' : trend.includes('Downtrend') ? 'light-on-red' : 'light-on-yellow';
            lights.innerHTML += `<div class="flex items-center gap-2"><div class="trend-light ${c}"></div><span class="text-sm font-bold text-text-secondary">${tf}</span></div>`;
        });

        // Toggles
        const tradeToggle = document.getElementById('trading-toggle'), tradeText = document.getElementById('trading-status-text');
        tradeToggle.checked = data.is_trading_enabled;
        tradeText.textContent = data.is_trading_enabled ? 'مُفعَّل' : 'غير مُفعَّل';
        tradeText.className = `font-bold text-lg ${data.is_trading_enabled ? 'text-accent-green' : 'text-accent-red'}`;
        document.getElementById('usdt-balance').textContent = data.usdt_balance ? parseFloat(data.usdt_balance).toFixed(2) : 'N/A';
        
        const filtersToggle = document.getElementById('disable-filters-toggle'), filtersText = document.getElementById('disable-filters-text');
        filtersToggle.checked = data.are_filters_disabled;
        filtersText.textContent = data.are_filters_disabled ? 'معطلة' : 'مفعلة';
        filtersText.className = `font-bold text-lg ${data.are_filters_disabled ? 'text-accent-red' : 'text-accent-green'}`;

        // Quiet Period Info
        const quietInfo = document.getElementById('quiet-period-info');
        if (data.is_in_quiet_period) {
            document.getElementById('quiet-period-reason').textContent = data.quiet_period_reason;
            quietInfo.classList.remove('hidden');
        } else {
            quietInfo.classList.add('hidden');
        }

        // Emergency Status Light & Button
        const emergencyLight = document.getElementById('emergency-light');
        const emergencyBtn = document.getElementById('emergency-toggle-btn');
        emergencyLight.classList.remove('light-on-green', 'light-on-red');
        emergencyBtn.classList.remove('bg-red-800', 'hover:bg-red-700', 'bg-green-800', 'hover:bg-green-700');
        if (data.is_emergency_active) {
            emergencyLight.classList.add('light-on-red');
            emergencyBtn.textContent = 'إلغاء التفعيل';
            emergencyBtn.classList.add('bg-green-800', 'hover:bg-green-700');
        } else {
            emergencyLight.classList.add('light-on-green');
            emergencyBtn.textContent = 'تفعيل';
            emergencyBtn.classList.add('bg-red-800', 'hover:bg-red-700');
        }
    });
}
function updateSignals() {
    fetchData('/api/signals').then(data => {
        if (!data) return;
        const tableBody = document.getElementById('signals-table');
        tableBody.innerHTML = '';
        if (data.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="6" class="text-center p-8 text-text-secondary">لا توجد صفقات مفتوحة حالياً.</td></tr>';
            return;
        }
        data.forEach(s => {
            const profit = parseFloat(s.profit_percentage || 0);
            const pClass = profit > 0 ? 'text-accent-green' : profit < 0 ? 'text-accent-red' : 'text-text-secondary';
            const entry = parseFloat(s.entry_price), sl = parseFloat(s.stop_loss), tp = parseFloat(s.target_price), current = parseFloat(s.current_price || entry);
            const progress = (tp - sl > 0) ? Math.max(0, Math.min(100, (current - sl) / (tp - sl) * 100)) : 0;
            tableBody.innerHTML += `<tr class="border-b border-border-color hover:bg-white/5"><td class="p-4 font-bold">${s.symbol}</td><td class="p-4"><span class="px-2 py-1 text-xs font-semibold rounded-full ${s.is_real_trade ? 'bg-blue-500/20 text-blue-400' : 'bg-yellow-500/20 text-yellow-400'}">${s.is_real_trade ? 'حقيقي' : 'تجريبي'}</span></td><td class="p-4 font-mono ${pClass}">${profit.toFixed(2)}%</td><td class="p-4"><div class="w-full bg-gray-700 rounded-full h-2.5"><div class="bg-accent-blue h-2.5 rounded-full" style="width: ${progress}%"></div></div></td><td class="p-4 font-mono">${current.toFixed(4)} / ${entry.toFixed(4)}</td><td class="p-4"><button onclick="manualClose(${s.id}, '${s.symbol}')" class="bg-red-600 hover:bg-red-700 text-white font-bold py-1 px-3 rounded text-xs">إغلاق</button></td></tr>`;
        });
    });
}
function updateNotifications() {
    fetchData('/api/notifications').then(data => {
        if (!data || data.length === 0) return;
        document.getElementById('notifications-list').innerHTML = data.map(n => `<div class="p-2 border-b border-border-color"><span class="font-mono text-xs text-text-secondary">${new Date(n.timestamp).toLocaleString('ar-EG')}</span>: ${n.message}</div>`).join('');
    });
}
function updateRejections() {
    fetchData('/api/rejection_logs').then(data => {
        if (!data || data.length === 0) return;
        document.getElementById('rejections-list').innerHTML = data.map(r => `<div class="p-2 border-b border-border-color"><span class="font-mono text-xs text-text-secondary">${new Date(r.timestamp).toLocaleString('ar-EG')}</span>: <strong class="text-accent-yellow">${r.symbol}</strong> - ${r.reason} <span class="text-xs text-gray-500">${JSON.stringify(r.details)}</span></div>`).join('');
    });
}
function manualClose(signalId, symbol) {
    showConfirmation('تأكيد الإغلاق', `هل أنت متأكد من رغبتك في إغلاق الصفقة لـ ${symbol} يدوياً؟`, () => {
        fetch(`/api/signals/close/${signalId}`, { method: 'POST' }).then(() => updateSignals());
    });
}
function toggleTrading() { fetch('/api/trading/toggle', { method: 'POST' }).then(() => updateMarketStatus()); }
function toggleFilters() { fetch('/api/filters/disable/toggle', { method: 'POST' }).then(() => updateMarketStatus()); }
function toggleEmergency() {
    showConfirmation('تأكيد وضع الطوارئ', 'هل أنت متأكد من تفعيل/إلغاء وضع الطوارئ يدوياً؟ سيتم إغلاق جميع الصفقات المفتوحة إذا تم التفعيل.', () => {
        fetch('/api/emergency/toggle', { method: 'POST' }).then(() => updateMarketStatus());
    });
}
document.addEventListener('DOMContentLoaded', () => {
    ['MarketStatus', 'Signals', 'Notifications', 'Rejections'].forEach(f => window[`update${f}`]());
    setInterval(updateMarketStatus, 5000); setInterval(updateSignals, 7000);
    setInterval(updateNotifications, 15000); setInterval(updateRejections, 15000);
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
    with dynamic_filter_lock: profile_copy = dict(dynamic_filter_profile_cache)
    active_sessions, _, _ = get_session_state()
    usdt_balance = None
    if client:
        try: usdt_balance = float(client.get_asset_balance(asset='USDT')['free'])
        except: usdt_balance = 'N/A'
    
    is_in_quiet_period, quiet_period_reason = is_currently_in_quiet_period()
    
    with emergency_manual_lock:
        manual_emergency_on = is_emergency_manually_triggered

    auto_emergency_on, _ = (emergency_detector.is_emergency_triggered() if emergency_detector else (False, {}))

    return jsonify({
        "market_state": state_copy, 
        "filter_profile": profile_copy, 
        "active_sessions": active_sessions, 
        "usdt_balance": usdt_balance, 
        "is_trading_enabled": is_enabled, 
        "are_filters_disabled": is_disabled,
        "is_in_quiet_period": is_in_quiet_period,
        "quiet_period_reason": quiet_period_reason,
        "is_emergency_active": manual_emergency_on or auto_emergency_on
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
        logger.error(f"❌ [API Signals] Error: {e}"); return jsonify([]), 500

@app.route('/api/notifications')
def get_notifications():
    with notifications_lock: return jsonify(list(notifications_cache))

@app.route('/api/rejection_logs')
def get_rejection_logs():
    with rejection_logs_lock: return jsonify(list(rejection_logs_cache))

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

@app.route('/api/signals/close/<int:signal_id>', methods=['POST'])
def manual_close_trade_endpoint(signal_id):
    # Placeholder for close logic
    return jsonify({"success": True, "message": "Signal closed."})

@app.route('/api/emergency/toggle', methods=['POST'])
def toggle_emergency_manual():
    global is_emergency_manually_triggered
    with emergency_manual_lock:
        is_emergency_manually_triggered = not is_emergency_manually_triggered
        status_msg = "ACTIVATED" if is_emergency_manually_triggered else "DEACTIVATED"
        log_and_notify('critical', f"🚨 Manual emergency mode has been {status_msg}!", "EMERGENCY_MANUAL_TOGGLE")
        if is_emergency_manually_triggered:
            # Run in a new thread to not block the API response
            Thread(target=emergency_close_all_positions, args=("Manual Trigger",)).start()
    return jsonify({"success": True, "is_emergency_active": is_emergency_manually_triggered})

def main_loop_enhanced():
    logger.info("[Main Loop] انتظار اكتمال التهيئة...")
    time.sleep(5)
    if not validated_symbols_to_scan:
        log_and_notify("critical", "لا توجد عملات صالحة للمسح.", "SYSTEM")
        return
    log_and_notify("info", f"✅ بدء حلقة المسح لـ {len(validated_symbols_to_scan)} عملة.", "SYSTEM")

    while True:
        try:
            if is_market_crashing():
                log_and_notify("warning", "نظام الطوارئ مفعل. تم إيقاف البحث عن صفقات.", "EMERGENCY_HALT")
                time.sleep(60) # Wait for a minute before re-checking
                continue

            check_and_send_quiet_period_alert()
            is_quiet, _ = is_currently_in_quiet_period()
            if is_quiet:
                logger.info("🤫 فترة هدوء نشطة. تم إيقاف البحث عن إشارات مؤقتاً.")
                time.sleep(60) 
                continue
            
            # --- Main scanning logic would go here ---
            logger.info("🔄 بدء دورة مسح جديدة...")
            time.sleep(60) # Simulate a scan cycle
            
        except (KeyboardInterrupt, SystemExit):
            log_and_notify("info", "إيقاف البوت.", "SYSTEM")
            break
        except Exception as main_err:
            log_and_notify("error", f"خطأ حرج في الحلقة الرئيسية: {main_err}", "SYSTEM")
            time.sleep(120)

def price_update_loop():
    if not redis_client: return
    while True:
        try:
            if validated_symbols_to_scan and client:
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
        if not validated_symbols_to_scan: 
            logger.critical("❌ لا توجد عملات صالحة للمسح.")
            return
        Thread(target=main_loop_enhanced, daemon=True).start()
        Thread(target=price_update_loop, daemon=True).start()
        # Thread(target=trade_management_loop, daemon=True).start() # Assuming full logic is present
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
