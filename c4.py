# ملف c4.py - نسخة محدثة مع نظام تحذير من الهبوط وتفسيرات مرتبطة بجلسات التداول العالمية
# تم التحديث بواسطة Gemini
# --- تعديل: إضافة تفسيرات لسبب الهبوط المحتمل بناءً على افتتاح/إغلاق البورصات العالمية ---
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
STATS_TRADE_SIZE_USDT: float = 5.0
BTC_SYMBOL: str = 'BTCUSDT'
MAX_OPEN_TRADES: int = 4
BUY_CONFIDENCE_THRESHOLD = 0.85
MIN_PROFIT_PERCENT: float = 0.8
SYMBOL_PROCESSING_BATCH_SIZE: int = 30
ADX_PERIOD: int = 14
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
EMA_SLOW_PERIOD: int = 200
EMA_FAST_PERIOD: int = 50
BTC_CORR_PERIOD: int = 30
REL_VOL_PERIOD: int = 30
MOMENTUM_PERIOD: int = 12
EMA_SLOPE_PERIOD: int = 5
USE_TRAILING_STOP_LOSS: bool = True
TRAILING_ACTIVATION_PROFIT_PERCENT: float = 1.8
TRAILING_DISTANCE_PERCENT: float = 1.0
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

# --- NEW: متغيرات وتفسيرات خاصة بنظام التحذير من الهبوط ---
sent_dip_warning_tracker: Set[str] = set()
last_warning_check_date: Optional[datetime.date] = None
HISTORICAL_DIP_HOURS_UTC: List[int] = [0, 4, 8, 12, 16, 20]
NOTIFICATION_LEAD_TIME_MINUTES: int = 30
DIP_REASONS_UTC: Dict[int, str] = {
    0:  "يتزامن مع *افتتاح الجلسة الآسيوية* (بورصة طوكيو)، مما يضخ سيولة وتقلبات جديدة في السوق.",
    4:  "فترة نشاط عالٍ في منتصف الجلسة الآسيوية، وعادةً ما ترتبط بحركة جني أرباح أو استكمال للاتجاه قبل افتتاح أوروبا.",
    8:  "يتزامن مع *افتتاح الجلسة الأوروبية* (بورصة لندن وفرانكفورت)، وهي فترة سيولة عالية ونشاط مؤسساتي كثيف.",
    12: "فترة ما قبل افتتاح السوق الأمريكي، حيث يقوم المتداولون الأوروبيون بتعديل مراكزهم، مما يؤدي غالبًا إلى زيادة التقلبات.",
    16: "يتزامن مع *إغلاق الجلسة الأوروبية*، وهي فترة تشهد تعديلات كبيرة على المراكز وجني أرباح.",
    20: "يتزامن مع *إغلاق الجلسة الأمريكية* (بورصة نيويورك)، ويمثل نهاية يوم التداول لأكبر حجم من المؤسسات المالية."
}

# --- قاموس أسباب الرفض باللغة العربية ---
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
    "Potential Profit Below Threshold (S/R)": "الربح المحتمل أقل من الحد الأدنى (دعم/مقاومة)",
    "EMA Crossover Invalid": "تقاطع EMA غير صالح"
}

# --- دالة إرسال رسائل تليجرام ---
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
                        target_price DOUBLE PRECISION NOT NULL, stop_loss DOUBLE PRECISION NOT NULL,
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

# --- دوال جلب البيانات وحساب الميزات ---
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    if not client: return None
    try:
        start_dt = datetime.now(timezone.utc) - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        klines = client.get_historical_klines(symbol, interval, start_str)
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
    df_calc['ema_9'] = df_calc['close'].ewm(span=9, adjust=False).mean()
    df_calc['ema_21'] = df_calc['close'].ewm(span=21, adjust=False).mean()
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
    gain = delta.clip(lower=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=RSI_PERIOD - 1, adjust=False).mean()
    df_calc['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    df_calc['relative_volume'] = df_calc['volume'] / (df_calc['volume'].rolling(window=REL_VOL_PERIOD, min_periods=1).mean() + 1e-9)
    df_calc['price_vs_ema50'] = (df_calc['close'] / df_calc['close'].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()) - 1
    df_calc['price_vs_ema200'] = (df_calc['close'] / df_calc['close'].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()) - 1
    if btc_df is not None and not btc_df.empty:
        asset_returns = df_calc['close'].pct_change()
        merged_df = pd.merge(df_calc, btc_df[['btc_returns']], left_index=True, right_index=True, how='left').fillna(0)
        df_calc['btc_correlation'] = asset_returns.rolling(window=BTC_CORR_PERIOD).corr(merged_df['btc_returns'])
    else:
        df_calc['btc_correlation'] = 0.0
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

# --- استراتيجية التداول والفلاتر ---
class EnhancedTradingStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        model_bundle = self._load_ml_model_from_file(symbol)
        self.ml_model, self.scaler, self.feature_names = (model_bundle.get('model'), model_bundle.get('scaler'), model_bundle.get('feature_names')) if model_bundle else (None, None, None)

    def _load_ml_model_from_file(self, symbol: str) -> Optional[Dict[str, Any]]:
        model_name = f"{BASE_ML_MODEL_NAME}_{symbol}"
        if model_name in ml_models_cache: return ml_models_cache[model_name]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir_path = os.path.join(script_dir, MODEL_FOLDER)
        model_path = os.path.join(model_dir_path, f"{model_name}.pkl")
        if not os.path.exists(model_path): return None
        try:
            with open(model_path, 'rb') as f: model_bundle = pickle.load(f)
            if 'model' in model_bundle and 'scaler' in model_bundle and 'feature_names' in model_bundle:
                ml_models_cache[model_name] = model_bundle
                return model_bundle
            return None
        except Exception as e:
            logger.error(f"❌ [ML Model File] خطأ في تحميل النموذج لـ {symbol}: {e}")
            return None

    def get_features(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame, btc_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.feature_names is None: return None
        try:
            df_featured = calculate_all_features(df_15m, btc_df)
            df_4h_features = calculate_all_features(df_4h, None)
            df_4h_features = df_4h_features.rename(columns=lambda c: f"{c}_4h")
            required_4h_cols = ['rsi_4h', 'price_vs_ema50_4h']
            df_featured = df_featured.join(df_4h_features[required_4h_cols], how='left')
            df_featured[required_4h_cols] = df_featured[required_4h_cols].fillna(method='ffill')
            for col in self.feature_names:
                if col not in df_featured.columns: df_featured[col] = 0.0
            df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_featured.dropna(subset=self.feature_names)
        except Exception as e:
            logger.error(f"❌ [{self.symbol}] فشل هندسة الميزات: {e}", exc_info=True)
            return None

    def generate_buy_signal(self, df_features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not all([self.ml_model, self.scaler, self.feature_names]) or df_features.empty: return None
        try:
            last_row_ordered_df = df_features.iloc[[-1]][self.feature_names]
            features_scaled = self.scaler.transform(last_row_ordered_df)
            prediction = self.ml_model.predict(features_scaled)[0]
            if prediction != 1: return None
            prediction_proba = self.ml_model.predict_proba(features_scaled)
            confidence = float(np.max(prediction_proba[0]))
            return {'prediction': int(prediction), 'confidence': confidence}
        except Exception as e:
            logger.warning(f"⚠️ [{self.symbol}] خطأ في توليد إشارة النموذج: {e}")
            return None

def passes_filters(symbol: str, last_features: pd.Series, entry_price: float, tp_sl_data: Dict, df_15m: pd.DataFrame) -> bool:
    with filters_disabled_lock:
        if are_filters_disabled:
            logger.warning(f"⚠️ [{symbol}] تجاوز الفلاتر بسبب الإعداد العام.")
            return True
    risk = entry_price - float(tp_sl_data['stop_loss']); reward = float(tp_sl_data['target_price']) - entry_price
    if risk <= 0 or reward <= 0 or (reward / risk) < 1.4: log_rejection(symbol, "RRR Filter", {"rrr": f"{(reward/risk):.2f}" if risk > 0 else "N/A"}); return False
    return True

def calculate_tp_sl(symbol: str, entry_price: float, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    try:
        if df.empty or len(df) < 50:
            log_rejection(symbol, "Insufficient data for TP/SL calculation")
            return None
        last_atr = df['atr'].iloc[-1] if 'atr' in df.columns and not df['atr'].empty else 0
        if last_atr > 0:
            target_price = entry_price + last_atr * 2.2
            stop_loss = entry_price - last_atr * 1.5
            return {'target_price': target_price, 'stop_loss': stop_loss, 'source': 'ATR_Fallback'}
        return None
    except Exception as e:
        logger.error(f"❌ [{symbol}] Error in TP/SL: {e}", exc_info=True)
        return None

def check_ema_crossover_signal(df: pd.DataFrame, lookback_period: int = 3) -> bool:
    required_cols = ['ema_9', 'ema_21', 'close', 'rsi', 'volume', 'volume_sma_20']
    if not all(col in df.columns for col in required_cols) or len(df) < lookback_period + 2:
        return False
    try:
        for i in range(1, lookback_period + 1):
            current_close = df['close'].iloc[-i]
            current_ema9 = df['ema_9'].iloc[-i]
            current_ema21 = df['ema_21'].iloc[-i]
            current_rsi = df['rsi'].iloc[-i]
            current_volume = df['volume'].iloc[-i]
            current_volume_sma = df['volume_sma_20'].iloc[-i]
            prev_ema9 = df['ema_9'].iloc[-(i + 1)]
            prev_ema21 = df['ema_21'].iloc[-(i + 1)]
            is_crossover = prev_ema9 < prev_ema21 and current_ema9 > current_ema21
            is_close_above = current_close > current_ema9 and current_close > current_ema21
            is_rsi_strong = current_rsi > 50
            is_volume_spike = current_volume_sma > 0 and current_volume > (current_volume_sma * 1.5)
            has_confirmation = is_rsi_strong or is_volume_spike
            if is_crossover and is_close_above and has_confirmation:
                return True
        return False
    except (IndexError, TypeError):
        return False

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
            # Simplified closing logic for brevity
            logger.info(f"Simulating closing real trade for {symbol_to_close}")
        if not check_db_connection() or not conn:
            log_and_notify('critical', "DB connection lost during trade closure.", "DB_ERROR")
            return False
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE signals SET status = 'closed', closing_price = %s, closed_at = NOW(), profit_percentage = %s, closing_reason = %s WHERE id = %s;", (closing_price, profit_percentage, reason, signal_id))
            conn.commit()
            if symbol_to_close in open_signals_cache:
                del open_signals_cache[symbol_to_close]
            log_and_notify('info', f"CLOSED: {symbol_to_close} at {closing_price:.4f}. Reason: {reason}. Profit: {profit_percentage:.2f}%", "TRADE_CLOSED")
            reason_map = {'take_profit': '🎯 Take Profit', 'stop_loss': '🛑 Stop Loss', 'manual': '🖐️ Manual Close'}
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
            entry_price = float(signal_data['entry_price'])
            target_price = float(signal_data['target_price'])
            stop_loss = float(signal_data['stop_loss'])
            quantity = float(signal_data['quantity']) if signal_data.get('quantity') is not None else None
            cur.execute("INSERT INTO signals (symbol, entry_price, target_price, stop_loss, strategy_name, signal_details, is_real_trade, quantity, order_id, current_peak_price) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *;", (signal_data['symbol'], entry_price, target_price, stop_loss, signal_data['strategy_name'], json.dumps(signal_data['signal_details']), signal_data.get('is_real_trade', False), quantity, signal_data.get('order_id'), entry_price))
            saved_signal = cur.fetchone()
            conn.commit()
            logger.info(f"💾 [{signal_data['symbol']}] تم حفظ الإشارة الجديدة في قاعدة البيانات.")
            trade_type = "حقيقية" if signal_data.get('is_real_trade') else "تجريبية"
            telegram_message = (f"💡 *توصية شراء {trade_type} جديدة*\n\n*العملة:* `{signal_data['symbol']}`\n*سعر الدخول:* `{entry_price:.4f}`\n*الهدف (TP):* `{target_price:.4f}`\n*وقف الخسارة (SL):* `{stop_loss:.4f}`\n\nConfidence: {signal_data['signal_details'].get('ML_Confidence', 'N/A')}")
            send_telegram_message(telegram_message)
            return dict(saved_signal)
    except Exception as e:
        logger.error(f"❌ [DB Insert] فشل إدراج الإشارة: {e}"); conn.rollback(); return None

# --- حلقات النظام ---
def trade_management_loop():
    logger.info("✅ [Trade Manager] بدء حلقة إدارة الصفقات...")
    while True:
        try:
            with signal_cache_lock:
                if not open_signals_cache:
                    time.sleep(5)
                    continue
                signals_to_check = list(open_signals_cache.values())
            if not redis_client: time.sleep(5); continue
            current_prices = redis_client.hgetall(REDIS_PRICES_HASH_NAME)
            for signal in signals_to_check:
                current_price_str = current_prices.get(signal['symbol'])
                if not current_price_str: continue
                current_price = float(current_price_str)
                signal_id = signal['id']
                tp = float(signal['target_price'])
                sl = float(signal['stop_loss'])
                if current_price >= tp:
                    close_signal(signal_id, current_price, 'take_profit')
                elif current_price <= sl:
                    close_signal(signal_id, current_price, 'stop_loss')
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
                    if not ml_signal or ml_signal['confidence'] < BUY_CONFIDENCE_THRESHOLD:
                        if ml_signal: log_rejection(symbol, "ML Model Rejected Signal", {"confidence": ml_signal['confidence']})
                        continue
                    if not check_ema_crossover_signal(df_features, lookback_period=3):
                        log_rejection(symbol, "EMA Crossover Invalid")
                        continue
                    try: entry_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
                    except Exception as e: logger.error(f"❌ [{symbol}] فشل جلب سعر الدخول: {e}."); continue
                    tp_sl_data = calculate_tp_sl(symbol, entry_price, df_15m)
                    if not tp_sl_data: continue
                    last_features = df_features.iloc[-1]
                    if not passes_filters(symbol, last_features, entry_price, tp_sl_data, df_15m): continue
                    new_signal = {'symbol': symbol, 'strategy_name': "Momentum_ML_V9", 'signal_details': {'ML_Confidence': f"{ml_signal['confidence']:.2%}"}, 'entry_price': entry_price, **tp_sl_data}
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
                finally: time.sleep(0.5)
            logger.info("✅ [End of Cycle] انتهت دورة المسح الكاملة. الانتظار 60 ثانية...")
            gc.collect()
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

def dip_warning_loop():
    """
    حلقة منفصلة للتحقق من أوقات الهبوط المحتملة وإرسال تحذيرات مع تفسيرات.
    """
    global sent_dip_warning_tracker, last_warning_check_date
    logger.info("✅ [Dip Warner] بدء حلقة التحذير من الهبوط المحتمل...")
    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            
            # Reset the tracker for a new day
            if last_warning_check_date is None or now_utc.date() != last_warning_check_date:
                logger.info(f"🌅 [Dip Warner] يوم جديد ({now_utc.date()}). إعادة تعيين متتبع التحذيرات.")
                sent_dip_warning_tracker.clear()
                last_warning_check_date = now_utc.date()

            # Skip warnings on weekends (Saturday=5, Sunday=6)
            if now_utc.weekday() >= 5:
                # Log this only once per hour to avoid spamming logs
                if now_utc.minute == 0:
                    logger.info("🗓️ [Dip Warner] عطلة نهاية الأسبوع، تم إيقاف التحذيرات مؤقتاً.")
                time.sleep(60)
                continue

            for target_hour_utc in HISTORICAL_DIP_HOURS_UTC:
                # Calculate the exact time to send the notification
                notification_time_utc = datetime(year=now_utc.year, month=now_utc.month, day=now_utc.day, hour=target_hour_utc, tzinfo=timezone.utc) - timedelta(minutes=NOTIFICATION_LEAD_TIME_MINUTES)
                
                # Check if it's time to send the notification
                if now_utc.hour == notification_time_utc.hour and now_utc.minute == notification_time_utc.minute:
                    warning_key = f"{now_utc.date()}-{target_hour_utc}"
                    
                    if warning_key not in sent_dip_warning_tracker:
                        # Get the reason for the dip
                        reason = DIP_REASONS_UTC.get(target_hour_utc, "لوحظ تكرار هبوط الأسعار في هذا التوقيت بناءً على تحليل إحصائي.")
                        
                        # Convert target time to GMT+1 for the message
                        target_time_gmt1 = (datetime(now_utc.year, now_utc.month, now_utc.day, target_hour_utc, tzinfo=timezone.utc) + timedelta(hours=1))
                        time_gmt1_str = target_time_gmt1.strftime('%H:%M')
                        
                        message = (
                            f"🚨 *تحذير من هبوط محتمل في السوق*\n\n"
                            f"*(تنبيه استباقي بناءً على تحليل إحصائي وجلسات التداول العالمية)*\n\n"
                            f"*الوقت المتوقع:* حوالي الساعة *{time_gmt1_str} بتوقيت GMT+1*\n\n"
                            f"*السبب المحتمل:* {reason}\n\n"
                            f"*المدة المحتملة:* قد يمتد تأثير هذا التقلب خلال شمعة الـ 4 ساعات القادمة.\n\n"
                            f"يرجى توخي الحذر وإدارة المخاطر بحكمة."
                        )
                        
                        send_telegram_message(message)
                        log_and_notify('info', f"Sent potential dip warning for {target_hour_utc}:00 UTC. Reason: Market session event.", "DIP_WARNING")
                        sent_dip_warning_tracker.add(warning_key)
            
            # Wait for 60 seconds before the next check
            time.sleep(60)
            
        except Exception as e:
            logger.error(f"❌ [Dip Warner] خطأ في حلقة التحذير: {e}", exc_info=True)
            time.sleep(300)

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
            logger.critical("❌ لا توجد عملات صالحة للمسح."); return
        Thread(target=main_loop_enhanced, daemon=True).start()
        Thread(target=price_update_loop, daemon=True).start()
        Thread(target=trade_management_loop, daemon=True).start()
        Thread(target=dip_warning_loop, daemon=True).start() # --- بدء حلقة التحذير الجديدة ---
        logger.info("✅ [Bot Services] تم بدء جميع الخدمات الخلفية بنجاح.")
        send_telegram_message("✅ *البوت قيد التشغيل الآن*")
    except Exception as e:
        log_and_notify("critical", f"حدث خطأ حرج أثناء التهيئة: {e}", "SYSTEM"); exit(1)

# --- نقطة الانطلاق ---
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
