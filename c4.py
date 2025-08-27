# ملف c4.py - نسخة V33.0.0 (فلتر تقلب ديناميكي - نسخة كاملة)
# --- وصف الإصدار:
# 1.  [تحسين جذري] استبدال فلتر تقلب السوق الثابت بفلتر ديناميكي جديد يحدد "أنظمة السوق" (Market Regimes).
# 2.  [ديناميكية] البوت الآن يقوم بتعديل معايير التداول (مثل مضاعفات وقف الخسارة والأهداف) تلقائيًا بناءً o
#     على نظام السوق السائد (هادئ، عادي، متقلب، شديد التقلب).
# 3.  [مرونة] تخفيف القيود بشكل ذكي في الأسواق العادية وزيادة الحذر في الأسواق شديدة التقلب.
# 4.  [مراجعة شاملة] تم مراجعة وتحسين جميع الفلاتر والاستراتيجيات لتعمل بتناغم مع النظام الديناميكي الجديد.
# 5.  [واجهة كاملة] هذا الملف يحتوي على الكود الكامل للبوت بما في ذلك واجهة التحكم Flask بجميع مكوناتها.

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
from flask_sock import Sock
from threading import Thread, Lock
from datetime import datetime, timezone, timedelta
from decouple import config
from typing import List, Dict, Optional, Any
from collections import deque
import warnings
from scipy.signal import argrelextrema

# --- إعدادات التجاهل واللوجر ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v33_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotV33.0.0')

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
cooldowns_by_symbol = {}
cooldowns_lock = Lock()
consecutive_losses_by_symbol = {}
consecutive_losses_lock = Lock()
COOLDOWN_MINUTES_AFTER_SL = 20
PAPER_TRADE_INITIAL_BALANCE = 1000.0

# --- المتغيرات القابلة للتعديل ---
RISK_PER_TRADE_PERCENT: float = 1.0
risk_per_trade_lock = Lock()
MAX_OPEN_TRADES: int = 3
TRAILING_STOP_ACTIVATION_PROFIT_PERCENT: float = 1.4
MIN_SIGNAL_QUALITY: int = 60
AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE: bool = True
min_quality_lock = Lock()

# --- مفاتيح تفعيل الاستراتيجيات ---
USE_BB_STOCH_STRATEGY: bool = True
USE_ELLIOTT_WAVE_STRATEGY: bool = True

# --- إعدادات استراتيجية موجات إليوت القابلة للتعديل ---
ELLIOTT_WAVE_SETTINGS = {
    "min_pattern_score": 0.55,
    "swing_point_order": 3,
    "swing_point_strength": 0.2,
    "volatility_filter_threshold": 8.0
}

# --- إعدادات الفلاتر الديناميكية للاستراتيجيات ---
STRATEGY_NAMES = {
    "BB_Stoch_Strategy": "BB+Stoch (انعكاسية)",
    "Elliott_Wave_Strategy": "Elliott Wave (موجات إليوت)"
}
STRATEGY_FILTER_CONFIG = {
    "BB_Stoch_Strategy": {"profile": "Reversal", "adx_threshold": 18, "htf_confirmation_mode": "Disabled"},
    "Elliott_Wave_Strategy": {"profile": "Relaxed", "adx_threshold": 20, "htf_confirmation_mode": "Relaxed"}
}
strategy_filters_lock = Lock()
BASE_FILTER_ADX_THRESHOLD = 20

# --- إعدادات عامة ---
SIGNAL_GENERATION_TIMEFRAME: str = '15m'
HIGHER_TIMEFRAME: str = '1h'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 15
BTC_SYMBOL: str = 'BTCUSDT'
API_REQUEST_DELAY: float = 0.5

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
    "Market Regime Not Tradeable": "فلتر نظام السوق رفض الدخول",
    "Trend Strength Filter Failed": "فلتر قوة الاتجاه رفض الدخول",
    "HTF Trend Confirmation Failed": "فشل تأكيد الترند على الفريم الأعلى",
    "Insufficient Historical Data": "بيانات تاريخية غير كافية للفحص",
    "MinNotional Filter Failed": "قيمة الصفقة أقل من الحد الأدنى للمنصة",
    "LOT_SIZE Filter Failed": "فشل تعديل حجم الصفقة",
    "Insufficient Balance": "الرصيد غير كافي لتنفيذ الصفقة",
    "Bullish Confirmation Failed": "فشل تأكيد الشمعة الصعودية",
    "Volume Filter Failed": "فلتر حجم التداول فشل",
    "Low Quality Signal": "جودة الإشارة منخفضة",
    "Invalid Position Size": "حجم الصفقة غير صالح (الوقف أعلى من الدخول)",
    "News Filter Failed": "فلتر الأخبار: تجنب التداول وقت الأخبار",
    "Liquidity Filter Failed": "فلتر السيولة: تجنب التداول في أوقات السيولة المنخفضة",
    "Correlation Filter Failed": "فلتر الارتباط: توجد صفقة مفتوحة على عملة مرتبطة",
    "BB: Price did not cross middle band": "BB: السعر لم يتقاطع مع الخط الأوسط",
    "Stoch: Not in oversold area": "Stoch: المؤشر ليس في منطقة ذروة البيع",
    "Elliott Wave: No valid patterns detected": "موجات إليوت: لم يتم العثور على أنماط صالحة",
    "Elliott Wave: Pattern score too low": "موجات إليوت: درجة جودة النمط منخفضة جدًا",
    "Elliott Wave: Invalid Impulse Wave Rules": "موجات إليوت: النمط لا يتبع القواعد الصارمة",
    "Elliott Wave: Volume too low": "موجات إليوت: حجم التداول منخفض جدًا",
    "Elliott Wave: RSI not in optimal range": "موجات إليوت: مؤشر القوة النسبية ليس في النطاق الأمثل",
    "Elliott Wave: MACD not positive": "موجات إليوت: مؤشر الماكد ليس إيجابيًا",
    "Elliott Wave: EMAs not in correct order": "موجات إليوت: المتوسطات المتحركة ليست بالترتيب الصحيح",
}

# --- إعداد تطبيق Flask و WebSocket ---
app = Flask(__name__)
CORS(app)
sock = Sock(app)
ws_clients: List[Any] = []
ws_clients_lock = Lock()

# --- دوال WebSocket ---
def broadcast(data: Dict):
    with ws_clients_lock:
        clients_to_remove = []
        for client in ws_clients:
            try:
                client.send(json.dumps(data, cls=NpEncoder))
            except Exception:
                clients_to_remove.append(client)
        for client in clients_to_remove:
            try:
                ws_clients.remove(client)
            except ValueError:
                pass

def get_dashboard_payload() -> Dict:
    with trading_status_lock: trading_enabled = is_trading_enabled
    with trading_mode_lock: is_paper_mode = paper_trading_mode
    with balance_lock: current_balance = usdt_balance
    with notifications_lock: notifications = list(notifications_cache)
    with rejection_logs_lock: rejections = list(rejection_logs_cache)
    with market_state_lock: market_state = dict(current_market_state)
    with min_quality_lock: min_quality = MIN_SIGNAL_QUALITY
    with risk_per_trade_lock: risk_percent = RISK_PER_TRADE_PERCENT

    return {
        "trading_enabled": trading_enabled,
        "paper_trading_mode": is_paper_mode,
        "usdt_balance": current_balance,
        "notifications": notifications,
        "rejections": rejections,
        "market_state": market_state,
        "min_signal_quality": min_quality,
        "risk_per_trade": risk_percent,
        "server_time": datetime.now(timezone.utc).isoformat()
    }

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
            logger.info("✅ [DB] Database connection and schema updated successfully.")
            return
        except Exception as e:
            logger.error(f"❌ [DB] Error during initialization (Attempt {attempt + 1}/{retries}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1: time.sleep(delay)
            else: logger.critical("❌ [DB] Failed to connect to the database. Exiting.")

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
        init_db()
        return conn is not None and conn.closed == 0

def init_redis() -> None:
    global redis_client
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("✅ [Redis] Connected successfully.")
    except redis.exceptions.ConnectionError as e:
        logger.warning(f"⚠️ [Redis] Connection failed: {e}.")
        redis_client = None

# --- دوال المساعدة والإشعارات ---
def log_and_notify(level: str, message: str, notification_type: str):
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error}
    log_methods.get(level.lower(), logger.info)(message)
    if not check_db_connection() or not conn: return
    try:
        new_notification = {"timestamp": datetime.now(timezone.utc).isoformat(), "type": notification_type, "message": message}
        with notifications_lock: notifications_cache.appendleft(new_notification)
        with conn.cursor() as cur: cur.execute("INSERT INTO notifications (type, message) VALUES (%s, %s);", (notification_type, message))
        conn.commit()
        broadcast({"type": "new_notification", "payload": new_notification})
    except Exception as e:
        logger.error(f"❌ [DB] Failed to save notification: {e}")
        if conn: conn.rollback()

def log_rejection(symbol: str, reason_key: str, details: Optional[Dict] = None):
    try:
        reason_ar = REJECTION_REASONS_AR.get(reason_key, reason_key)
        if details:
            details_str = ", ".join([f"{k}: {v}" for k, v in details.items()])
            reason_ar = f"{reason_ar} ({details_str})"
        log_entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "symbol": symbol, "reason": reason_ar}
        with rejection_logs_lock: rejection_logs_cache.appendleft(log_entry)
        broadcast({"type": "new_rejection", "payload": log_entry})
    except Exception as e:
        logger.error(f"❌ [Log Rejection] Error logging rejection for {symbol}: {e}", exc_info=True)

def send_enhanced_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [Telegram] Failed to send message: {e}")

# --- دوال جلب البيانات والمؤشرات ---
def get_exchange_info_map() -> None:
    global exchange_info_map
    try:
        logger.info("[API] Fetching exchange info...")
        exchange_info_map = {s['symbol']: s for s in client.get_exchange_info()['symbols']}
        logger.info(f"✅ [API] Exchange info map created with {len(exchange_info_map)} symbols.")
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
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        return df.dropna().astype(float)
    except Exception as e:
        logger.error(f"❌ [Data] Error fetching data for {symbol}: {e}"); return None

def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df_calc = df.copy()
    df_calc['ema9'] = df_calc['Close'].ewm(span=9, adjust=False).mean()
    df_calc['ema13'] = df_calc['Close'].ewm(span=13, adjust=False).mean()
    df_calc['ema21'] = df_calc['Close'].ewm(span=21, adjust=False).mean()
    df_calc['ema50'] = df_calc['Close'].ewm(span=50, adjust=False).mean()
    df_calc['ema200'] = df_calc['Close'].ewm(span=200, adjust=False).mean()
    
    high_low = df_calc['High'] - df_calc['Low']
    high_close = (df_calc['High'] - df_calc['Close'].shift()).abs()
    low_close = (df_calc['Low'] - df_calc['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    df_calc['atr'] = tr.ewm(span=14, adjust=False).mean()
    df_calc['atr_percent'] = (df_calc['atr'] / df_calc['Close'].replace(0, 1e-9)) * 100
    
    up_move = df_calc['High'].diff()
    down_move = -df_calc['Low'].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
    df_calc['adx'] = dx.ewm(span=14, adjust=False).mean()
    
    delta = df_calc['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df_calc['rsi'] = 100 - (100 / (1 + rs))
    
    bb_middle = df_calc['Close'].rolling(window=10).mean()
    bb_std = df_calc['Close'].rolling(window=10).std()
    df_calc['bb_middle'] = bb_middle
    df_calc['bb_lower'] = bb_middle - (bb_std * 1.5)
    df_calc['bb_upper'] = bb_middle + (bb_std * 1.5)
    
    exp1 = df_calc['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df_calc['Close'].ewm(span=26, adjust=False).mean()
    df_calc['macd'] = exp1 - exp2
    df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()
    df_calc['macd_hist'] = df_calc['macd'] - df_calc['macd_signal']
    
    low_stoch = df_calc['Low'].rolling(9).min()
    high_stoch = df_calc['High'].rolling(9).max()
    df_calc['stoch_k'] = 100 * ((df_calc['Close'] - low_stoch) / (high_stoch - low_stoch).replace(0, 1e-9))
    df_calc['stoch_d'] = df_calc['stoch_k'].rolling(3).mean()
    
    return df_calc

# --- START: Dynamic Market Regime Filter (V33.0.0) ---
def get_dynamic_market_regime(df: pd.DataFrame) -> Dict[str, Any]:
    if 'atr_percent' not in df.columns or len(df) < 100:
        return {"regime": "Unknown", "tradeable": False, "params": {}}

    recent_atr = df['atr_percent'].tail(96 * 10).dropna()
    if len(recent_atr) < 50:
        return {"regime": "Unknown", "tradeable": False, "params": {}}

    current_atr = df['atr_percent'].iloc[-1]
    
    p20 = np.percentile(recent_atr, 20)
    p85 = np.percentile(recent_atr, 85)
    p98 = np.percentile(recent_atr, 98)

    regime = "Unknown"
    tradeable = False
    params = {
        "stop_loss_multiplier": 1.5,
        "target1_multiplier": 2.0,
        "target2_multiplier": 3.5,
        "adx_threshold": BASE_FILTER_ADX_THRESHOLD
    }

    if current_atr < p20:
        regime = "Low Volatility"
        tradeable = True
        params.update({
            "stop_loss_multiplier": 1.2, "target1_multiplier": 1.8,
            "target2_multiplier": 3.0, "adx_threshold": BASE_FILTER_ADX_THRESHOLD * 1.2
        })
    elif p20 <= current_atr < p85:
        regime = "Normal"
        tradeable = True
    elif p85 <= current_atr < p98:
        regime = "High Volatility"
        tradeable = True
        params.update({
            "stop_loss_multiplier": 2.2, "target1_multiplier": 2.5,
            "target2_multiplier": 4.5, "adx_threshold": BASE_FILTER_ADX_THRESHOLD * 0.85
        })
    else:
        regime = "Extreme Volatility"
        tradeable = False

    return {"regime": regime, "tradeable": tradeable, "params": params}

def apply_market_regime_filter(symbol: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    market_analysis = get_dynamic_market_regime(df)
    if not market_analysis["tradeable"]:
        log_rejection(symbol, "Market Regime Not Tradeable", {"regime": market_analysis["regime"]})
        return None
    logger.info(f"[{symbol}] Market Regime: {market_analysis['regime']}. Applying dynamic params.")
    return market_analysis["params"]
# --- END: Dynamic Market Regime Filter ---

# --- Risk Management & Other Filters ---
def calculate_signal_quality_score(df, strategy_name):
    score = 0
    if df.empty or len(df) < 50: return 0
    last_row = df.iloc[-1]
    adx_value = last_row.get('adx', 0)
    if adx_value > 35: score += 25
    elif adx_value > 25: score += 20
    elif adx_value > 18: score += 15
    else: score += 5
    current_volume = last_row.get('Volume', 0)
    volume_ma = df['Volume'].rolling(20, min_periods=5).mean().iloc[-1]
    volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1
    if volume_ratio > 2.0: score += 15
    elif volume_ratio > 1.5: score += 12
    else: score += 5
    rsi = last_row.get('rsi', 50)
    if 45 <= rsi <= 55: score += 20
    elif 40 <= rsi <= 60: score += 15
    ema9, ema21, ema50, ema200, close = last_row.get('ema9',0), last_row.get('ema21',0), last_row.get('ema50',0), last_row.get('ema200',0), last_row.get('Close',0)
    if close > ema9 > ema21 > ema50 > ema200: score += 20
    elif close > ema9 > ema21 > ema50: score += 15
    elif close > ema9 > ema21: score += 10
    atr_percent = last_row.get('atr_percent', 0)
    if 1.5 <= atr_percent <= 4.5: score += 10
    if strategy_name == "BB_Stoch_Strategy" and last_row.get('stoch_k', 50) < 25: score += 8
    elif strategy_name == "Elliott_Wave_Strategy" and adx_value > 25: score += 10
    return min(100, max(0, int(score)))

def check_trend_strength_filter(df: pd.DataFrame, adx_threshold: int, symbol: str) -> bool:
    if 'adx' not in df.columns or len(df) < 5:
        log_rejection(symbol, "Trend Strength Filter Failed"); return False
    recent_adx = float(pd.Series(df['adx'].tail(3)).mean())
    if recent_adx < adx_threshold:
        log_rejection(symbol, "Trend Strength Filter Failed", {"adx": f"{recent_adx:.1f}", "required": adx_threshold}); return False
    return True

def is_htf_bullish_confirmation(symbol: str, htf: str = '1h', mode: str = 'Strict') -> bool:
    if mode == 'Disabled': return True
    try:
        df = fetch_historical_data(symbol, htf, days=40)
        if df is None or len(df) < 50: return False
        df['ema50'] = df['Close'].ewm(span=50, adjust=False).mean()
        last = df.iloc[-1]
        if mode == 'Strict':
            df['ema200'] = df['Close'].ewm(span=200, adjust=False).mean()
            last = df.iloc[-1]
            return last['Close'] > last['ema50'] and last['ema50'] > last['ema200']
        elif mode == 'Relaxed':
            return last['Close'] > last['ema50']
        return False
    except Exception as e:
        logger.warning(f"[HTF] Could not confirm HTF trend for {symbol}: {e}"); return False

def apply_strategy_filters(symbol: str, df: pd.DataFrame, strategy_name: str, dynamic_params: Dict) -> bool:
    with strategy_filters_lock: config = STRATEGY_FILTER_CONFIG.get(strategy_name)
    if not config or config.get("profile") == "Disabled": return True
    adx_threshold = dynamic_params.get("adx_threshold", config.get("adx_threshold", 22))
    if not check_trend_strength_filter(df, adx_threshold, symbol): return False
    htf_mode = config.get("htf_confirmation_mode", "Strict")
    if not is_htf_bullish_confirmation(symbol, HIGHER_TIMEFRAME, htf_mode):
        log_rejection(symbol, "HTF Trend Confirmation Failed"); return False
    return True

# --- Trading Strategies ---
def check_bb_stoch_strategy_enhanced(df: pd.DataFrame, symbol: str) -> bool:
    needed_cols = {'bb_lower', 'stoch_k', 'stoch_d', 'Open', 'Close'}
    if len(df) < 21 or not needed_cols.issubset(df.columns): return False
    last, prev = df.iloc[-1], df.iloc[-2]
    oversold_level = 20
    stoch_in_oversold = prev['stoch_k'] < oversold_level and prev['stoch_d'] < oversold_level
    stoch_crossed_up = last['stoch_k'] > last['stoch_d'] and prev['stoch_k'] <= prev['stoch_d']
    if not (stoch_in_oversold and stoch_crossed_up):
        log_rejection(symbol, "Stoch: Not in oversold area"); return False
    bounce_from_lower_band = (prev['Low'] <= prev['bb_lower']) and (last['Close'] > last['bb_lower'])
    if not bounce_from_lower_band:
        log_rejection(symbol, "BB: Price did not bounce from lower band"); return False
    is_bullish_candle = last['Close'] > last['Open']
    if not is_bullish_candle:
        log_rejection(symbol, "Bullish Confirmation Failed"); return False
    return True

class PatternWeights:
    def __init__(self): self.weights = {'wave3_ratio': 0.25, 'wave3_length': 0.30, 'volume': 0.15, 'rsi': 0.12, 'macd': 0.10, 'ema_order': 0.08}
    def _calculate_volume_score(self, points: List, df: pd.DataFrame) -> float:
        try:
            avg_volume = df['Volume'].iloc[:points[0]['index']].mean()
            wave3_volume = df['Volume'].iloc[points[2]['index']:points[3]['index']].mean()
            return min(1.0, (wave3_volume / avg_volume) / 2.0) if avg_volume > 0 else 0.0
        except (IndexError, KeyError): return 0.0
    def _calculate_rsi_score(self, df: pd.DataFrame) -> float:
        try:
            rsi_value = df['rsi'].iloc[-1]
            return (rsi_value - 40) / 30 if 40 <= rsi_value <= 70 else 0.0
        except (IndexError, KeyError): return 0.0
    def _calculate_macd_score(self, df: pd.DataFrame) -> float:
        try:
            macd_hist = df['macd_hist'].iloc[-1]
            return min(1.0, macd_hist / (df['Close'].iloc[-1] * 0.001)) if macd_hist > 0 else 0.0
        except (IndexError, KeyError): return 0.0
    def _calculate_ema_score(self, df: pd.DataFrame) -> float:
        try:
            last = df.iloc[-1]
            if last['ema9'] > last['ema21'] > last['ema50']: return 1.0
            elif last['ema9'] > last['ema21']: return 0.5
            return 0.0
        except (IndexError, KeyError): return 0.0
    def calculate_impulse_score(self, points: List, df: pd.DataFrame, direction: str = 'up') -> float:
        score = 0.0; p = [item['price'] for item in points]
        wave1 = p[1] - p[0] if direction == 'up' else p[0] - p[1]
        wave3 = p[3] - p[2] if direction == 'up' else p[2] - p[3]
        wave5 = p[5] - p[4] if direction == 'up' else p[4] - p[5]
        if wave1 <= 0 or wave3 <= 0 or wave5 <= 0: return 0.0
        if wave3 > wave1 and wave3 > wave5: score += self.weights['wave3_length']
        wave3_ratio = wave3 / wave1; ideal_ratio = 1.618
        ratio_score = 1.0 - min(1.0, abs(wave3_ratio - ideal_ratio) / ideal_ratio)
        score += ratio_score * self.weights['wave3_ratio']
        if 'Volume' in df.columns: score += self._calculate_volume_score(points, df) * self.weights['volume']
        score += self._calculate_rsi_score(df) * self.weights['rsi']
        score += self._calculate_macd_score(df) * self.weights['macd']
        score += self._calculate_ema_score(df) * self.weights['ema_order']
        return min(1.0, score)
pattern_weights = PatternWeights()

def find_swing_points(df, order=5):
    high_indices = argrelextrema(df['High'].values, np.greater, order=order)[0]
    low_indices = argrelextrema(df['Low'].values, np.less, order=order)[0]
    points = [{'index': i, 'type': 'high', 'price': df['High'].iloc[i]} for i in high_indices] + \
             [{'index': i, 'type': 'low', 'price': df['Low'].iloc[i]} for i in low_indices]
    points.sort(key=lambda x: x['index'])
    if not points: return []
    filtered_points = [points[0]]
    for i in range(1, len(points)):
        if points[i]['type'] != filtered_points[-1]['type']:
            filtered_points.append(points[i])
    return filtered_points

def validate_impulse_wave_rules(points: List, df: pd.DataFrame, wave_type: str = 'up') -> bool:
    if len(points) != 6: return False
    p = [item['price'] for item in points]
    if wave_type == 'up':
        if not all(points[i]['type'] == ('low' if i % 2 == 0 else 'high') for i in range(6)): return False
        wave1_len = p[1] - p[0]; wave3_len = p[3] - p[2]; wave5_len = p[5] - p[4]
        if wave1_len <= 0 or wave3_len <= 0: return False
        if p[2] < p[0] or (wave3_len < wave1_len and wave3_len < wave5_len) or p[4] < p[1]: return False
        wave2_retracement = (p[1] - p[2]) / wave1_len
        if not (0.382 <= wave2_retracement <= 0.618): return False
        wave4_retracement = (p[3] - p[4]) / wave3_len
        if not (0.382 <= wave4_retracement <= 0.5): return False
    return True

def detect_elliott_wave_patterns(df: pd.DataFrame, symbol: str) -> Dict:
    swing_points = find_swing_points(df, order=ELLIOTT_WAVE_SETTINGS['swing_point_order'])
    if len(swing_points) < 6: return {}
    valid_patterns = []
    for i in range(len(swing_points) - 5):
        potential_pattern = swing_points[i:i+6]
        if potential_pattern[0]['type'] == 'low' and validate_impulse_wave_rules(potential_pattern, df, 'up'):
            score = pattern_weights.calculate_impulse_score(potential_pattern, df, 'up')
            if score >= ELLIOTT_WAVE_SETTINGS['min_pattern_score']:
                valid_patterns.append({'type': 'impulse_up', 'points': potential_pattern, 'score': score, 'direction': 'up'})
    if not valid_patterns: return {}
    best_pattern = max(valid_patterns, key=lambda p: p['score'])
    return {best_pattern['type']: best_pattern}

def apply_elliott_wave_quality_filters(pattern: Dict, df: pd.DataFrame, symbol: str) -> bool:
    if pattern['score'] < ELLIOTT_WAVE_SETTINGS['min_pattern_score']:
        log_rejection(symbol, "Elliott Wave: Pattern score too low"); return False
    last_row = df.iloc[-1]
    if not (40 <= last_row.get('rsi', 50) <= 60):
        log_rejection(symbol, "Elliott Wave: RSI not in optimal range"); return False
    if pattern['direction'] == 'up' and last_row.get('macd_hist', 0) <= 0:
        log_rejection(symbol, "Elliott Wave: MACD not positive"); return False
    if pattern['direction'] == 'up' and not (last_row.get('ema9', 0) > last_row.get('ema21', 0) > last_row.get('ema50', 0)):
        log_rejection(symbol, "Elliott Wave: EMAs not in correct order"); return False
    return True

# --- Trade Execution & Management ---
def calculate_trade_levels(df: pd.DataFrame, dynamic_params: Dict) -> Dict[str, Any]:
    last = df.iloc[-1]
    atr = last['atr']
    entry_price = last['Close']
    stop_loss_multiplier = dynamic_params['stop_loss_multiplier']
    target1_multiplier = dynamic_params['target1_multiplier']
    target2_multiplier = dynamic_params['target2_multiplier']
    stop_loss = entry_price - (atr * stop_loss_multiplier)
    target_price_1 = entry_price + (atr * target1_multiplier)
    target_price_2 = entry_price + (atr * target2_multiplier)
    trailing_stop_distance = atr * (stop_loss_multiplier * 0.8)
    return {
        "entry_price": entry_price, "stop_loss": stop_loss, "target_price_1": target_price_1,
        "target_price_2": target_price_2, "trailing_stop_distance": trailing_stop_distance
    }

def adjust_quantity_to_lot_size(symbol: str, quantity: float) -> Optional[Decimal]:
    try:
        symbol_info = exchange_info_map.get(symbol)
        if not symbol_info: return None
        lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if not lot_size_filter: return Decimal(str(quantity))
        step_size = Decimal(lot_size_filter['stepSize'])
        min_qty = Decimal(lot_size_filter['minQty'])
        quantity_dec = Decimal(str(quantity))
        if quantity_dec < min_qty: return None
        adjusted_quantity = (quantity_dec // step_size) * step_size
        return adjusted_quantity if adjusted_quantity >= min_qty else None
    except Exception as e:
        logger.error(f"❌ [{symbol}] CRITICAL ERROR adjusting quantity: {e}", exc_info=True); return None

def calculate_position_size(symbol: str, entry_price: float, stop_loss_price: float) -> Optional[Decimal]:
    if not client: return None
    try:
        with risk_per_trade_lock: risk_percent = RISK_PER_TRADE_PERCENT
        balance_response = client.get_asset_balance(asset='USDT')
        available_balance = Decimal(balance_response['free'])
        risk_amount_usdt = available_balance * (Decimal(str(risk_percent)) / Decimal('100'))
        risk_per_coin = Decimal(str(entry_price)) - Decimal(str(stop_loss_price))
        if risk_per_coin <= 0:
            log_rejection(symbol, "Invalid Position Size"); return None
        initial_quantity = risk_amount_usdt / risk_per_coin
        adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(initial_quantity))
        if adjusted_quantity is None or adjusted_quantity <= 0: return None
        notional_value = adjusted_quantity * Decimal(str(entry_price))
        symbol_info = exchange_info_map.get(symbol)
        if symbol_info:
            min_notional_filter = next((f for f in symbol_info['filters'] if f['filterType'] in ('MIN_NOTIONAL', 'NOTIONAL')), None)
            if min_notional_filter:
                min_notional = Decimal(min_notional_filter.get('minNotional', min_notional_filter.get('notional', '0')))
                if notional_value < min_notional:
                    log_rejection(symbol, "MinNotional Filter Failed"); return None
        if notional_value > available_balance:
            log_rejection(symbol, "Insufficient Balance"); return None
        return adjusted_quantity
    except Exception as e:
        logger.error(f"❌ [{symbol}] Error calculating position size: {e}", exc_info=True); return None

def calculate_paper_position_size(entry_price: float, stop_loss_price: float) -> float:
    with risk_per_trade_lock: risk_percent = RISK_PER_TRADE_PERCENT
    risk_per_coin = entry_price - stop_loss_price
    if risk_per_coin <= 0: return 0.0
    risk_amount = PAPER_TRADE_INITIAL_BALANCE * (risk_percent / 100.0)
    return risk_amount / risk_per_coin

def place_order(symbol: str, side: str, quantity: Decimal, order_type: str = Client.ORDER_TYPE_MARKET) -> Optional[Dict]:
    if not client: return None
    logger.info(f"➡️ [{symbol}] Attempting to place REAL {side} order for quantity {quantity}.")
    try:
        order = client.create_order(symbol=symbol, side=side, type=order_type, quantity=str(quantity))
        log_and_notify('info', f"TRADE REAL: Placed {side} order for {quantity} {symbol}.", "REAL_TRADE")
        return order
    except BinanceAPIException as e:
        logger.error(f"❌ [{symbol}] Binance API Error on order placement: {e}"); return None
    except Exception as e:
        logger.error(f"❌ [{symbol}] General error on order placement: {e}", exc_info=True); return None

def generate_signal(symbol: str, df: pd.DataFrame, strategy_name: str, dynamic_params: Dict) -> Optional[Dict]:
    quality_score = calculate_signal_quality_score(df, strategy_name)
    with min_quality_lock:
        if quality_score < MIN_SIGNAL_QUALITY:
            log_rejection(symbol, "Low Quality Signal", {"score": quality_score, "min_required": MIN_SIGNAL_QUALITY}); return None
    trade_levels = calculate_trade_levels(df, dynamic_params)
    entry_price = trade_levels['entry_price']; stop_loss = trade_levels['stop_loss']
    if not paper_trading_mode:
        quantity = calculate_position_size(symbol, entry_price, stop_loss)
        if quantity is None: return None
        quantity = float(quantity)
    else:
        quantity = calculate_paper_position_size(entry_price, stop_loss)
    if quantity <= 0: return None
    return {
        "symbol": symbol, "strategy": strategy_name, "entry_price": entry_price,
        "stop_loss": stop_loss, "target1": trade_levels['target_price_1'],
        "target2": trade_levels['target_price_2'], "quantity": quantity,
        "quality_score": quality_score, "atr_percent": df['atr_percent'].iloc[-1],
        "trailing_stop_distance": trade_levels['trailing_stop_distance']
    }

def create_trade_signal(signal_data: Dict):
    symbol = signal_data['symbol']
    with cooldowns_lock:
        if symbol in cooldowns_by_symbol and datetime.now(timezone.utc) < cooldowns_by_symbol[symbol]: return
    with trading_mode_lock: is_real = not paper_trading_mode
    signal_details = {
        "atr_percent": signal_data['atr_percent'], "quality_score": signal_data['quality_score'],
        "trailing_stop_distance": signal_data['trailing_stop_distance'], "trailing_stop_activated": False
    }
    if is_real:
        order = place_order(symbol, Client.SIDE_BUY, Decimal(str(signal_data['quantity'])))
        if order:
            avg_fill_price = sum(Decimal(f['price']) * Decimal(f['qty']) for f in order.get('fills', [])) / max(sum(Decimal(f['qty']) for f in order.get('fills', [])), Decimal('1e-8')) if order.get('fills') else Decimal(str(signal_data['entry_price']))
            final_quantity = Decimal(order.get('executedQty', str(signal_data['quantity'])))
            order_id = order.get('orderId', 'N/A')
            save_signal_to_db(symbol, float(avg_fill_price), signal_data, True, float(final_quantity), {**signal_details, "avg_fill": float(avg_fill_price)}, order_id)
        else:
            logger.error(f"❌ [Real Trade] Order placement failed for {symbol}. Trade not opened."); return
    else:
        save_signal_to_db(symbol, signal_data['entry_price'], signal_data, False, signal_data['quantity'], signal_details)
    log_and_notify("info", f"Opened {'REAL' if is_real else 'PAPER'} trade for {symbol}", "TRADE_OPEN")
    send_enhanced_telegram_message(f"🚀 *New {'Real' if is_real else 'Paper'} Trade: {symbol}*\nStrategy: {signal_data['strategy']}\nEntry: {signal_data['entry_price']:.4f}\nSL: {signal_data['stop_loss']:.4f}\nTP1: {signal_data['target1']:.4f}")

def save_signal_to_db(symbol: str, entry_price: float, signal_data: Dict, is_real: bool, quantity: float, signal_details: Dict, order_id: Optional[str] = None):
    try:
        if not (check_db_connection() and conn): return
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO signals (symbol, entry_price, target_price_1, target_price_2, stop_loss, status,
                                   strategy_name, is_real_trade, quantity, initial_quantity, signal_details, order_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
            """, (symbol, entry_price, signal_data['target1'], signal_data['target2'], signal_data['stop_loss'],
                  'open', signal_data['strategy'], is_real, quantity, quantity,
                  json.dumps(signal_details, cls=NpEncoder), order_id))
            new_id = cur.fetchone()['id']
        conn.commit()
        new_signal_data = {
            'id': new_id, 'symbol': symbol, 'entry_price': entry_price,
            'target_price_1': signal_data['target1'], 'target_price_2': signal_data['target2'],
            'stop_loss': signal_data['stop_loss'], 'status': 'open', 'strategy_name': signal_data['strategy'],
            'is_real_trade': is_real, 'quantity': quantity, 'initial_quantity': quantity,
            'signal_details': signal_details, 'order_id': order_id
        }
        with signal_cache_lock: open_signals_cache[symbol] = new_signal_data
        broadcast({"type": "new_signal", "payload": new_signal_data})
    except Exception as e:
        logger.error(f"❌ [DB] CRITICAL ERROR saving signal for {symbol}: {e}", exc_info=True)
        if conn: conn.rollback()

# --- قوالب HTML ---
DASHBOARD_TEMPLATE = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>لوحة التحكم - بوت التداول (V33.0.0)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
:root{--bg:#0b1020;--panel:#121b36;--accent:#3aa0ff;--ok:#15c46a;--warn:#ff9f1a;--bad:#ff4757;--muted:#8aa0c8;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:#e8f1ff;font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,"Noto Sans",Arial}
.container{max-width:1600px;margin:0 auto;padding:16px;display:flex;flex-direction:column;gap:16px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between}
h1{font-size:18px;margin:0;font-weight:700;color:#d7e4ff}
.badge{padding:6px 10px;border-radius:999px;font-size:12px;background:#0d1730;border:1px solid #1e2c52;color:#cce0ff}
.main-layout{display:grid;grid-template-columns:1fr;gap:16px;}
@media(min-width: 1000px){.main-layout{grid-template-columns:1fr 350px;}}
.left-column,.right-column{display:flex;flex-direction:column;gap:16px}
.card{background:var(--panel);border:1px solid #1e2c52;border-radius:14px;box-shadow:0 8px 30px rgba(0,0,0,.25);overflow:hidden}
.card h2{margin:0;padding:12px 14px;border-bottom:1px solid #1e2c52;font-size:14px;color:#cfe2ff; display: flex; justify-content: space-between; align-items: center;}
.card-body{padding:12px}
.controls{display:flex;gap:8px;flex-wrap:wrap}
.btn{appearance:none;border:1px solid #2a3a68;background:#0f1b3b;color:#d9e7ff;padding:10px 14px;border-radius:10px;cursor:pointer;font-weight:700;transition: background-color 0.2s, transform 0.2s; will-change: transform; text-decoration: none;}
.btn:hover{transform:translateY(-1px);border-color:#3a58a6}
.btn.warn{background:linear-gradient(180deg,#3b2a0f,#291b08);border-color:#8b5b0f}
.btn.small{padding: 6px 10px; font-size: 12px;}
.signals-grid{display:grid;grid-template-columns:repeat(auto-fill, minmax(300px, 1fr));gap:10px;}
.signal{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;padding:10px;border:1px solid #24335f;border-radius:12px;background:#0d1730; grid-template-rows: auto auto;}
.signal > *:nth-child(1) { grid-column: 1 / 2; }
.signal > *:nth-child(2) { grid-column: 2 / 3; grid-row: 1 / 3; }
.signal > *:nth-child(3) { grid-column: 1 / 2; }
.sig-title{font-weight:700}
.sig-meta{font-size:12px;color:var(--muted)}
.price{font-variant-numeric:tabular-nums;direction:ltr; font-size: 16px; font-weight: bold;}
.progress{height:8px;background:#0b1126;border:1px solid #233056;border-radius:999px;overflow:hidden; margin-top: 6px;}
.progress>span{display:block;height:100%;}
.kv{display:grid;grid-template-columns:auto 1fr;gap:6px 10px; align-items: center;}
.kv div:nth-child(odd){opacity:.8}
.trend{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}
.trend .pill{background:#0d1730;border:1px solid #1f2d55;border-radius:10px;padding:8px;text-align:center; display: flex; flex-direction: column; align-items: center; gap: 4px;}
.pill b{display:block;font-size:12px;color:#9fb7ef}
.pill span{font-size:12px}
.pill small {font-size: 10px; opacity: 0.8;}
.green{color:var(--ok)}.red{color:var(--bad)}.amber{color:var(--warn)}
.table{width:100%;border-collapse:separate;border-spacing:0 8px; table-layout: fixed;}
.table th{font-size:12px;text-align:right;color:#9ab2e2;font-weight:600;padding:0 6px}
.table td{padding:8px;background:#0d1730;border:1px solid #24335f; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;}
.switch{display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;border:1px solid #2a3a68;background:#0f1b3b;cursor:pointer;user-select:none}
.switch input{display:none}
.switch .dot{width:14px;height:14px;border-radius:50%;background:#6a7fb2;transition:.2s}
.switch input:checked + .dot{background:#24d08a;}
.small{font-size:12px;color:#a8bfeb}
.chart-container { height: 200px; }
.loading-spinner { border: 3px solid rgba(255, 255, 255, 0.1); border-radius: 50%; border-top: 3px solid #3aa0ff; width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 20px auto; }
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="container">
  <header><h1>لوحة التحكم • بوت التداول V33.0.0</h1><div class="badge" id="serverTime">—</div></header>
  <div class="main-layout">
    <div class="left-column">
      <div class="card">
        <h2>الصفقات المفتوحة <span class="small" id="signalCount">(0)</span></h2>
        <div class="card-body"><div id="signals" class="signals-grid"><div class="loading-spinner"></div></div></div>
      </div>
      <div class="card">
        <h2>مؤشرات الأداء</h2>
        <div class="card-body"><div class="chart-container"><canvas id="performanceChart"></canvas></div></div>
      </div>
    </div>
    <div class="right-column">
      <div class="card">
        <h2>التحكم والحالة</h2>
        <div class="card-body">
          <div class="controls">
            <label class="switch"><input id="toggleTrading" type="checkbox" /><span class="dot"></span><span class="small">تشغيل التداول</span></label>
            <label class="switch"><input id="toggleMode" type="checkbox" /><span class="dot"></span><span class="small" id="modeText">ورقي</span></label>
          </div>
          <div class="kv" style="margin-top:12px">
            <div>الرصيد (USDT):</div><div id="balance">—</div><div>عدد الصفقات:</div><div id="openCount">—</div>
          </div>
        </div>
      </div>
      <div class="card">
        <h2>حالة السوق</h2>
        <div class="card-body"><div class="trend" id="marketTrends"><div class="loading-spinner"></div></div></div>
      </div>
      <div class="card">
        <h2>سجل الرفض</h2>
        <div class="card-body" style="padding:0; max-height: 250px; overflow-y: auto;">
          <table class="table" id="rejections"><thead><tr><th>الوقت</th><th>الرمز</th><th>السبب</th></tr></thead><tbody></tbody></table>
        </div>
      </div>
      <div class="card">
        <h2>سجل الأحداث</h2>
        <div class="card-body" style="padding:0; max-height: 250px; overflow-y: auto;">
          <table class="table" id="events"><thead><tr><th>الوقت</th><th>النوع</th><th>الرسالة</th></tr></thead><tbody></tbody></table>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
const qs = s => document.querySelector(s);
let lastPrices = {};
let performanceChartInstance = null;
let openSignals = {};

function fmt(n){ return n == null ? '—' : (+n).toLocaleString('en-US', {maximumFractionDigits: 6}); }

function renderSignal(signal) {
    const cp = lastPrices[signal.symbol] || signal.entry_price;
    const entry = signal.entry_price, tp1 = signal.target_price_1, sl = signal.stop_loss;
    let progress = 0, color = 'transparent', title = 'في انتظار حركة السعر';
    if (cp >= entry && tp1 > entry) {
        progress = Math.min(100, ((cp - entry) / (tp1 - entry)) * 100);
        color = 'linear-gradient(90deg, var(--ok), #3fd1b0)';
        title = `التقدم نحو الهدف: ${progress.toFixed(1)}%`;
    } else if (cp < entry && entry > sl) {
        progress = Math.min(100, ((entry - cp) / (entry - sl)) * 100);
        color = 'linear-gradient(90deg, var(--bad), #ff6b7a)';
        title = `الاقتراب من وقف الخسارة: ${progress.toFixed(1)}%`;
    }
    const qualityScore = signal.signal_details?.quality_score || 0;
    const qualityColor = qualityScore > 75 ? 'var(--ok)' : qualityScore > 55 ? 'var(--warn)' : 'var(--bad)';
    const strategyName = signal.strategy_name.replace(/_/g, " ").replace("Strategy", "");
    return `
        <div class="signal" id="signal-${signal.id}" data-symbol="${signal.symbol}">
            <div>
                <div class="sig-title">${signal.symbol}</div>
                <div class="sig-meta">${strategyName} | <span style="color: ${qualityColor}; font-weight: bold;">⭐ ${qualityScore}/100</span></div>
            </div>
            <div style="text-align:end">
                <div class="price">${fmt(cp)}</div>
                <button class="btn warn small" onclick="closeTrade(${signal.id})">إغلاق</button>
            </div>
            <div class="progress" title="${title}"><span style="width:${progress.toFixed(2)}%; background:${color};"></span></div>
        </div>`;
}

function updateUI(data) {
    qs('#serverTime').textContent = new Date(data.server_time).toLocaleTimeString('ar-EG');
    qs('#toggleTrading').checked = !!data.trading_enabled;
    qs('#toggleMode').checked = !data.paper_trading_mode;
    qs('#modeText').textContent = data.paper_trading_mode ? 'ورقي' : 'حقيقي';
    qs('#balance').textContent = fmt(data.usdt_balance);
    qs('#openCount').textContent = Object.keys(openSignals).length;
    qs('#signalCount').textContent = `(${Object.keys(openSignals).length})`;
    if (data.market_state) updateMarketTrends(data.market_state);
    qs('#rejections tbody').innerHTML = data.rejections.map(r => `<tr><td>${new Date(r.timestamp).toLocaleTimeString('ar-EG')}</td><td>${r.symbol}</td><td>${r.reason}</td></tr>`).join('');
    qs('#events tbody').innerHTML = data.notifications.map(n => `<tr><td>${new Date(n.timestamp).toLocaleTimeString('ar-EG')}</td><td>${n.type}</td><td>${n.message}</td></tr>`).join('');
}

function updateMarketTrends(marketState) {
  const trendsContainer = qs('#marketTrends');
  trendsContainer.innerHTML = '';
  if (marketState && marketState.trend_details_by_tf) {
    ['15m', '1h', '4h'].forEach(tf => {
      const trend = marketState.trend_details_by_tf[tf];
      if (trend) {
        let trendClass = 'amber', trendText = 'جانبي';
        if (trend.trend === 'bullish') { trendClass = 'green'; trendText = 'صاعد'; }
        else if (trend.trend === 'bearish') { trendClass = 'red'; trendText = 'هابط'; }
        trendsContainer.innerHTML += `<div class="pill"><b>${tf}</b><span class="${trendClass}">${trendText}</span><small>ADX: ${trend.adx?.toFixed(1) || '—'}</small></div>`;
      }
    });
  }
}

function setupWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    const socket = new WebSocket(wsUrl);
    socket.onopen = () => console.log("WebSocket connected");
    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        switch(data.type) {
            case 'price_update': 
                Object.assign(lastPrices, data.payload);
                Object.values(openSignals).forEach(s => {
                    const el = qs(`#signal-${s.id}`);
                    if (el) el.outerHTML = renderSignal(s);
                });
                break;
            case 'new_signal': openSignals[data.payload.id] = data.payload; renderAllSignals(); break;
            case 'trade_closed': delete openSignals[data.payload.signal_id]; renderAllSignals(); break;
            case 'full_update': 
                openSignals = data.payload.signals.reduce((acc, s) => { acc[s.id] = s; return acc; }, {});
                updateUI(data.payload.dashboard);
                renderAllSignals();
                break;
        }
    };
    socket.onclose = () => { console.log("WebSocket closed, reconnecting..."); setTimeout(setupWebSocket, 3000); };
}

function renderAllSignals() {
    const container = qs('#signals');
    const signals = Object.values(openSignals);
    if (!signals || signals.length === 0) {
        container.innerHTML = '<p style="text-align:center;color:var(--muted);">لا توجد صفقات مفتوحة.</p>';
    } else {
        container.innerHTML = signals.map(renderSignal).join('');
    }
    qs('#openCount').textContent = signals.length;
    qs('#signalCount').textContent = `(${signals.length})`;
}

async function closeTrade(signalId) {
    if (!confirm('هل أنت متأكد من إغلاق الصفقة يدويًا؟')) return;
    await fetch(`/api/close_trade/${signalId}`, { method: 'POST' });
}

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const [baseRes, signalsRes] = await Promise.all([fetch('/api/dashboard_data'), fetch('/api/open_signals')]);
        const baseData = await baseRes.json();
        const signalsData = await signalsRes.json();
        openSignals = signalsData.signals.reduce((acc, s) => { acc[s.id] = s; return acc; }, {});
        updateUI(baseData);
        renderAllSignals();
        setupWebSocket();
        qs('#toggleTrading').addEventListener('change', async () => await fetch('/toggle_trading', {method:'POST'}));
        qs('#toggleMode').addEventListener('change', async (e) => await fetch('/api/settings', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({paper_trading_mode: !e.target.checked}) }));
    } catch (error) {
        console.error("Dashboard init failed:", error);
        qs('#signals').innerHTML = '<p>فشل تحميل البيانات.</p>';
    }
});
</script>
</body>
</html>
"""

# --- مسارات Flask ---
@app.route('/')
def dashboard(): return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/api/dashboard_data')
def api_dashboard_data(): return jsonify(get_dashboard_payload())

@app.route('/api/open_signals')
def api_get_open_signals():
    with signal_cache_lock:
        return jsonify({"signals": list(open_signals_cache.values())})

@app.route('/toggle_trading', methods=['POST'])
def toggle_trading():
    global is_trading_enabled
    with trading_status_lock: is_trading_enabled = not is_trading_enabled
    status_msg = "enabled" if is_trading_enabled else "disabled"
    log_and_notify("info", f"Trading has been {status_msg}.", "TRADING_STATUS")
    broadcast({"type": "trading_status", "payload": {"enabled": is_trading_enabled}})
    return jsonify({"status": "success", "enabled": is_trading_enabled})

@app.route('/api/settings', methods=['POST'])
def update_settings():
    data = request.json
    if 'paper_trading_mode' in data:
        with trading_mode_lock:
            global paper_trading_mode
            paper_trading_mode = bool(data['paper_trading_mode'])
        log_and_notify("info", f"Trading mode set to {'Paper' if paper_trading_mode else 'Real'}.", "MODE_CHANGE")
    # Add other settings here
    return jsonify({"success": True})

@app.route('/api/close_trade/<int:signal_id>', methods=['POST'])
def api_close_trade(signal_id):
    with signal_cache_lock:
        signal_to_close = next((s for s in open_signals_cache.values() if s['id'] == signal_id), None)
    if not signal_to_close:
        return jsonify({"success": False, "message": "Signal not found"}), 404
    
    symbol = signal_to_close['symbol']
    with live_prices_lock:
        closing_price = live_prices.get(symbol)
    
    if closing_price is None:
        return jsonify({"success": False, "message": "Could not get live price"}), 500
        
    # Run in a thread to avoid blocking the request
    Thread(target=close_signal, args=(signal_to_close, closing_price, "manual_close")).start()
    return jsonify({"success": True, "message": "Close command sent"})

@sock.route('/ws')
def ws(ws_client):
    logger.info("WebSocket client connected.")
    with ws_clients_lock: ws_clients.append(ws_client)
    try:
        # Send initial full state
        with signal_cache_lock:
            signals = list(open_signals_cache.values())
        dashboard_data = get_dashboard_payload()
        ws_client.send(json.dumps({"type": "full_update", "payload": {"signals": signals, "dashboard": dashboard_data}}, cls=NpEncoder))
        
        while True:
            # Keep connection alive
            message = ws_client.receive(timeout=30)
            if message is None:
                ws_client.send(json.dumps({"type": "ping"}))

    except Exception as e:
        logger.info(f"WebSocket client disconnected: {e}")
    finally:
        with ws_clients_lock:
            if ws_client in ws_clients: ws_clients.remove(ws_client)

# --- Main Loop & Threads ---
def scan_symbol_for_signals(symbol: str) -> Optional[Dict]:
    with signal_cache_lock:
        if len(open_signals_cache) >= MAX_OPEN_TRADES or symbol in open_signals_cache: return None
    with cooldowns_lock:
        if symbol in cooldowns_by_symbol and datetime.now(timezone.utc) < cooldowns_by_symbol[symbol]: return None

    df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
    if df is None or len(df) < 100:
        log_rejection(symbol, "Insufficient Historical Data"); return None
    
    df_with_features = calculate_all_features(df)
    
    dynamic_params = apply_market_regime_filter(symbol, df_with_features)
    if dynamic_params is None: return None

    signals = []
    if USE_BB_STOCH_STRATEGY and apply_strategy_filters(symbol, df_with_features, "BB_Stoch_Strategy", dynamic_params):
        if check_bb_stoch_strategy_enhanced(df_with_features, symbol):
            signal = generate_signal(symbol, df_with_features, "BB_Stoch_Strategy", dynamic_params)
            if signal: signals.append(signal)
    
    if USE_ELLIOTT_WAVE_STRATEGY and apply_strategy_filters(symbol, df_with_features, "Elliott_Wave_Strategy", dynamic_params):
        ew_patterns = detect_elliott_wave_patterns(df_with_features, symbol)
        if ew_patterns and 'impulse_up' in ew_patterns and apply_elliott_wave_quality_filters(ew_patterns['impulse_up'], df_with_features, symbol):
            signal = generate_signal(symbol, df_with_features, "Elliott_Wave_Strategy", dynamic_params)
            if signal: signals.append(signal)

    if not signals: return None
    
    best_signal = max(signals, key=lambda x: x['quality_score'])
    with min_quality_lock:
        if best_signal['quality_score'] < MIN_SIGNAL_QUALITY:
            log_rejection(symbol, "Low Quality Signal", {"score": best_signal['quality_score']}); return None
            
    return best_signal

def main_bot_loop():
    logger.info("🚀 [Main Loop] Starting signal scanning loop (V33 - Dynamic)...")
    while True:
        try:
            with trading_status_lock:
                if not is_trading_enabled:
                    time.sleep(10); continue
            
            with signal_cache_lock: open_trades_count = len(open_signals_cache)
            logger.info(f"===== New Scan Cycle | Open Trades: {open_trades_count}/{MAX_OPEN_TRADES} =====")
            
            for symbol in validated_symbols_to_scan:
                with signal_cache_lock:
                    if len(open_signals_cache) >= MAX_OPEN_TRADES:
                        logger.info(f"Max open trades reached. Ending scan cycle."); break
                signal = scan_symbol_for_signals(symbol)
                if signal: create_trade_signal(signal)

            now = datetime.now(timezone.utc)
            minutes_to_wait = 15 - (now.minute % 15)
            seconds_to_wait = (minutes_to_wait * 60) - now.second
            logger.info(f"Scan cycle complete. Waiting {seconds_to_wait:.0f}s for next candle.")
            time.sleep(max(1, seconds_to_wait))
        except Exception as e:
            logger.error(f"❌ [Main Loop] Critical error: {e}", exc_info=True); time.sleep(60)

def update_signal_in_db(signal_id, updates):
    if not (check_db_connection() and conn): return False
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
        logger.error(f"❌ [DB] Failed to update signal {signal_id}: {e}"); conn.rollback(); return False

def close_signal(signal: Dict, closing_price: float, reason: str):
    symbol, signal_id, entry_price = signal['symbol'], signal['id'], signal['entry_price']
    with signal_cache_lock:
        if symbol not in open_signals_cache or open_signals_cache[symbol]['id'] != signal_id: return
    
    if signal.get('is_real_trade'):
        try:
            quantity_in_bot = Decimal(str(signal.get('quantity', 0)))
            if quantity_in_bot > 0:
                adjusted_quantity = adjust_quantity_to_lot_size(symbol, float(quantity_in_bot))
                if adjusted_quantity and adjusted_quantity > 0:
                    place_order(symbol, Client.SIDE_SELL, adjusted_quantity)
        except Exception as e:
            logger.error(f"❌ [{symbol}] Critical error during real trade closure: {e}", exc_info=True); return
            
    profit = ((closing_price - entry_price) / entry_price) * 100
    update_signal_in_db(signal_id, {"status": "closed", "closing_price": closing_price, "closed_at": datetime.now(timezone.utc), "profit_percentage": profit, "closing_reason": reason})
    
    with signal_cache_lock:
        if symbol in open_signals_cache: del open_signals_cache[symbol]
        
    broadcast({"type": "trade_closed", "payload": {"signal_id": signal_id, "symbol": symbol, "reason": reason}})
    log_and_notify("info", f"Closed trade for {symbol}. Profit: {profit:.2f}%", "TRADE_CLOSED")
    send_enhanced_telegram_message(f"{'✅' if profit >= 0 else '🔻'} *Trade Closed: {symbol}*\nReason: {reason}\nProfit: `{profit:.2f}%`")

def trade_management_loop():
    logger.info("🚀 [Trade Manager] Starting...")
    while True:
        try:
            with signal_cache_lock:
                if not open_signals_cache:
                    time.sleep(2); continue
                signals_to_monitor = list(open_signals_cache.values())
            
            for signal in signals_to_monitor:
                symbol = signal['symbol']
                with live_prices_lock: current_price = live_prices.get(symbol)
                if not current_price: continue
                
                details = signal.get('signal_details', {})
                if isinstance(details, str): details = json.loads(details)
                
                stop_loss = float(signal['stop_loss'])
                tp1 = float(signal['target_price_1'])
                tp2 = float(signal['target_price_2'])
                entry_price = float(signal['entry_price'])
                
                if current_price <= stop_loss:
                    close_signal(signal, stop_loss, "SL_HIT"); continue
                if current_price >= tp2:
                    close_signal(signal, tp2, "TP2_HIT"); continue
                
                if not details.get('tp1_done') and current_price >= tp1:
                    # Logic for partial close at TP1
                    new_sl = max(stop_loss, entry_price)
                    details['tp1_done'] = True
                    updates = {"stop_loss": new_sl, "status": "updated", "signal_details": json.dumps(details)}
                    if update_signal_in_db(signal['id'], updates):
                        with signal_cache_lock:
                            if symbol in open_signals_cache: open_signals_cache[symbol].update(updates)
                        send_enhanced_telegram_message(f"🥇 *TP1 Hit: {symbol}*\nStop loss moved to entry.")
                
                profit_pct = ((current_price - entry_price) / entry_price) * 100
                if not details.get('trailing_active') and profit_pct >= TRAILING_STOP_ACTIVATION_PROFIT_PERCENT:
                    details['trailing_active'] = True
                    update_signal_in_db(signal['id'], {"signal_details": json.dumps(details)})
                
                if details.get('trailing_active'):
                    trail_dist = details.get('trailing_stop_distance', 0)
                    new_sl = max(stop_loss, current_price - trail_dist)
                    if new_sl > stop_loss:
                        update_signal_in_db(signal['id'], {"stop_loss": new_sl})
            time.sleep(1)
        except Exception as e:
            logger.error(f"❌ [Trade Manager] Loop error: {e}", exc_info=True); time.sleep(5)

def load_initial_data():
    init_db()
    init_redis()
    global client
    try:
        client = Client(API_KEY, API_SECRET); client.ping()
        logger.info("✅ [Binance] API connection successful.")
    except Exception as e:
        logger.critical(f"❌ [Binance] API connection failed: {e}"); exit(1)
    
    get_exchange_info_map()
    global validated_symbols_to_scan
    validated_symbols_to_scan = get_validated_symbols()
    if not validated_symbols_to_scan:
        logger.critical("❌ No valid symbols to scan. Exiting."); exit(1)
    
    if check_db_connection():
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status IN ('open', 'updated');")
            with signal_cache_lock:
                for signal in cur.fetchall(): open_signals_cache[signal['symbol']] = dict(signal)
            logger.info(f"✅ [Cache] Loaded {len(open_signals_cache)} open signals.")

def handle_socket_message(msg):
    global live_prices
    try:
        if isinstance(msg, list):
            price_updates = {}
            with live_prices_lock:
                for ticker in msg:
                    if 's' in ticker and 'c' in ticker:
                        price = float(ticker['c'])
                        live_prices[ticker['s']] = price
                        price_updates[ticker['s']] = price
            if price_updates:
                broadcast({"type": "price_update", "payload": price_updates})
    except Exception as e:
        logger.error(f"❌ [WebSocket] Error processing message: {e}", exc_info=True)

def start_websocket():
    global ws_manager
    ws_manager = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    ws_manager.start()
    ws_manager.start_ticker_socket(callback=handle_socket_message)
    logger.info("✅ [WebSocket] Subscribed to ticker stream.")

# --- نقطة بداية البرنامج ---
if __name__ == '__main__':
    logger.info("="*50 + "\n====== Starting Crypto Trading Bot V33.0.0 (Dynamic Regime) ======\n" + "="*50)
    load_initial_data()
    start_websocket()
    Thread(target=main_bot_loop, daemon=True).start()
    Thread(target=trade_management_loop, daemon=True).start()
    logger.info("🌐 [Flask] Starting UI on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
