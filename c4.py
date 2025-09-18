# ملف crypto_bot_enhanced.py - النسخة المحسنة والمطورة
# --- وصف التطويرات:
# 1. [تحسين الاستراتيجيات] منطق استراتيجيات أكثر ذكاءً ودقة
# 2. [إدارة المخاطر المتقدمة] نظام إدارة مخاطر متطور
# 3. [واجهة محسنة] تصميم عصري وألوان متطورة
# 4. [تحليلات متقدمة] تحليلات أعمق وإحصائيات مفصلة
# 5. [إشعارات محسنة] نظام إشعارات أكثر ذكاءً
# 6. [أداء محسن] تحسينات في الأداء والاستقرار

import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
import redis
import statistics
import random
import math
from decimal import Decimal, ROUND_DOWN, getcontext
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
from typing import List, Dict, Optional, Any, Tuple
from collections import deque
import warnings
from scipy.signal import argrelextrema
import talib

# --- إعدادات التجاهل واللوجر ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ضبط دقة النوع Decimal
getcontext().prec = 18

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_enhanced_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBotEnhanced')

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
COOLDOWN_MINUTES_AFTER_SL = 30

# --- المتغيرات القابلة للتعديل (محسنة) ---
PAPER_TRADE_FIXED_AMOUNT_USDT: float = 15.0  # زيادة القيمة الافتراضية
FIXED_TRADE_AMOUNT_MIN_USDT: float = 5.0
FIXED_TRADE_AMOUNT_MAX_USDT: float = 8.0
trade_amount_lock = Lock()
MAX_OPEN_TRADES: int = 4  # زيادة عدد الصفقات المتاحة
TRAILING_STOP_ACTIVATION_PROFIT_PERCENT: float = 0.8
MIN_SIGNAL_QUALITY: int = 65  # تقليل قليل للحصول على إشارات أكثر
AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE: bool = True
min_quality_lock = Lock()

# إدارة المخاطر المتقدمة
MAX_DAILY_LOSS_PERCENT: float = 5.0
MAX_TOTAL_EXPOSURE_PERCENT: float = 15.0
MIN_RISK_REWARD_RATIO: float = 1.2
MAX_CORRELATION_THRESHOLD: float = 0.7
risk_management_lock = Lock()

# --- مفاتيح تفعيل الاستراتيجيات المحسنة ---
USE_ENHANCED_BB_STOCH_STRATEGY: bool = True
USE_ENHANCED_MACD_EMA_STRATEGY: bool = True
USE_ENHANCED_EMA_RSI_STRATEGY: bool = True
USE_ENHANCED_PULLBACK_STRATEGY: bool = True
USE_ENHANCED_MOMENTUM_STRATEGY: bool = True
USE_ENHANCED_ELLIOTT_WAVE_STRATEGY: bool = True
USE_ENHANCED_RANGE_REVERSAL_STRATEGY: bool = True
USE_ADVANCED_BREAKOUT_STRATEGY: bool = True  # استراتيجية جديدة
USE_VOLUME_PROFILE_STRATEGY: bool = True  # استراتيجية جديدة
USE_ICHIMOKU_STRATEGY: bool = True  # استراتيجية جديدة

# --- إعدادات الفلاتر الديناميكية للاستراتيجيات المحسنة ---
ENHANCED_STRATEGY_NAMES = {
    "Enhanced_BB_Stoch_Strategy": "BB+Stoch المحسن (ارتداد ذكي)",
    "Enhanced_MACD_EMA_Strategy": "MACD+SMA المحسن (زخم متقدم)",
    "Enhanced_EMA_RSI_Strategy": "EMA+RSI المحسن (ارتداد سريع)",
    "Enhanced_Pullback_Strategy": "Pullback المحسن (ارتداد بحجم ذكي)",
    "Enhanced_Momentum_Strategy": "Momentum المحسن (زخم متطور)",
    "Enhanced_Elliott_Wave_Strategy": "Elliott Wave المحسن (موجات ذكية)",
    "Enhanced_Range_Reversal_Strategy": "Range Reversal المحسن (انعكاس نطاقي)",
    "Advanced_Breakout_Strategy": "Breakout المتقدم (كسر المستويات)",
    "Volume_Profile_Strategy": "Volume Profile (تحليل الحجم)",
    "Ichimoku_Strategy": "Ichimoku (السحب اليابانية)"
}

strategy_filters_lock = Lock()

# --- إعدادات عامة محسنة ---
SIGNAL_GENERATION_TIMEFRAME: str = '5m'
HIGHER_TIMEFRAME: str = '15m'
MACRO_TIMEFRAME: str = '1h'
TIMEFRAMES_FOR_TREND_LIGHTS: List[str] = ['5m', '15m', '1h', '4h']
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 10
BTC_SYMBOL: str = 'BTCUSDT'
ETH_SYMBOL: str = 'ETHUSDT'
API_REQUEST_DELAY: float = 0.8

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
notifications_cache = deque(maxlen=30)
notifications_lock = Lock()
rejection_logs_cache = deque(maxlen=50)
rejection_logs_lock = Lock()
current_market_state: Dict[str, Any] = {"trend_details_by_tf": {}}
market_state_lock = Lock()

# Cache for performance optimization
indicator_cache = {}
indicator_cache_lock = Lock()

# --- أسباب الرفض المحسنة ---
ENHANCED_REJECTION_REASONS_AR = {
    # General Filters
    "Market Volatility Filter Failed": "فلتر تقلب السوق رفض الدخول",
    "Insufficient Historical Data": "بيانات تاريخية غير كافية للفحص",
    "MinNotional Filter Failed": "قيمة الصفقة أقل من الحد الأدنى للمنصة",
    "LOT_SIZE Filter Failed": "فشل تعديل حجم الصفقة",
    "Insufficient Balance": "الرصيد غير كافي لتنفيذ الصفقة",
    "Low Quality Signal": "جودة الإشارة منخفضة",
    "Invalid Position Size": "حجم الصفقة غير صالح",
    "News Filter Failed": "فلتر الأخبار: تجنب التداول وقت الأخبار",
    "Liquidity Filter Failed": "فلتر السيولة: تجنب التداول في أوقات السيولة المنخفضة",
    "Correlation Filter Failed": "فلتر الارتباط: توجد صفقة مفتوحة على عملة مرتبطة",
    
    # Risk Management
    "Max Daily Loss Reached": "تم الوصول للحد الأقصى للخسارة اليومية",
    "Max Exposure Exceeded": "تجاوز الحد الأقصى للتعرض",
    "Poor Risk Reward Ratio": "نسبة المخاطرة للعائد ضعيفة",
    "High Correlation Risk": "مخاطر ارتباط عالية",
    
    # Enhanced Dynamic Filters
    "ENHANCED_VOLATILITY_EXTREME": "محسن: التقلب متطرف جداً",
    "ENHANCED_VOLUME_ANOMALY": "محسن: شذوذ في حجم التداول",
    "ENHANCED_MOMENTUM_WEAK": "محسن: الزخم ضعيف جداً",
    "ENHANCED_TREND_UNCERTAIN": "محسن: الاتجاه غير واضح",
    "ENHANCED_SUPPORT_RESISTANCE_WEAK": "محسن: مستويات الدعم والمقاومة ضعيفة",
    
    # Strategy Specific Enhanced
    "Enhanced BB: Multiple timeframe conflict": "BB المحسن: تضارب في الإطارات الزمنية",
    "Enhanced MACD: Divergence detected": "MACD المحسن: تباعد مكتشف",
    "Enhanced RSI: Overbought conditions": "RSI المحسن: ظروف تشبع شرائي",
    "Enhanced Pullback: Insufficient retracement": "Pullback المحسن: تصحيح غير كافي",
    "Advanced Breakout: False breakout risk": "Breakout المتقدم: مخاطر كسر وهمي"
}

# --- إعداد تطبيق Flask و WebSocket ---
app = Flask(__name__)
CORS(app)
sock = Sock(app)
ws_clients: List[Any] = []
ws_clients_lock = Lock()

# --- دوال WebSocket محسنة ---
def broadcast(data: Dict):
    with ws_clients_lock:
        clients_to_remove = []
        for client in ws_clients:
            try:
                client.send(json.dumps(data, cls=NpEncoder))
            except Exception as e:
                logger.warning(f"WebSocket send failed, removing client: {e}")
                clients_to_remove.append(client)
        
        for client in clients_to_remove:
            try:
                ws_clients.remove(client)
            except ValueError:
                pass

# --- دوال تهيئة الخدمات وقاعدة البيانات المحسنة ---
def optimize_database():
    if not check_db_connection() or not conn:
        return
    try:
        with conn.cursor() as cur:
            logger.info("[DB] Optimizing database with enhanced indexes...")
            # Existing indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol_status ON signals(symbol, status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_notifications_timestamp ON notifications(timestamp);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_status_closed_at ON signals(status, closed_at);")
            
            # Enhanced indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_strategy_name ON signals(strategy_name);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_profit_percentage ON signals(profit_percentage);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals(created_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_is_real_trade ON signals(is_real_trade);")
            
            conn.commit()
            logger.info("✅ [DB] Enhanced database indexes optimized successfully.")
    except Exception as e:
        logger.error(f"❌ [DB] Error optimizing database: {e}")
        if conn: conn.rollback()

def column_exists(cursor, table_name, column_name):
    cursor.execute("SELECT 1 FROM information_schema.columns WHERE table_name = %s AND column_name = %s", (table_name, column_name))
    return cursor.fetchone() is not None

def init_db(retries: int = 5, base_delay: int = 5) -> None:
    global conn
    logger.info("[DB] Initializing enhanced database connection...")
    db_url_to_use = DB_URL
    if 'postgres' in db_url_to_use and 'sslmode' not in db_url_to_use:
        db_url_to_use += f"{'?' if '?' not in db_url_to_use else '&'}sslmode=require"
    
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(db_url_to_use, connect_timeout=15, cursor_factory=RealDictCursor)
            conn.autocommit = False
            
            with conn.cursor() as cur:
                # Enhanced signals table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        entry_price DOUBLE PRECISION NOT NULL,
                        stop_loss DOUBLE PRECISION NOT NULL,
                        status TEXT DEFAULT 'open',
                        closing_price DOUBLE PRECISION,
                        closed_at TIMESTAMP,
                        profit_percentage DOUBLE PRECISION,
                        strategy_name TEXT,
                        signal_details JSONB,
                        is_real_trade BOOLEAN DEFAULT FALSE,
                        quantity DOUBLE PRECISION,
                        closing_reason TEXT,
                        order_id TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                """)
                
                # Enhanced notifications table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL,
                        message TEXT NOT NULL,
                        priority INTEGER DEFAULT 1,
                        read_status BOOLEAN DEFAULT FALSE
                    );
                """)
                
                # Enhanced performance summary table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS performance_summary (
                        id SERIAL PRIMARY KEY,
                        trade_id INTEGER REFERENCES signals(id),
                        profit_percentage DOUBLE PRECISION,
                        drawdown DOUBLE PRECISION,
                        date DATE,
                        strategy_name TEXT,
                        market_conditions JSONB
                    );
                """)
                
                # Risk management table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS risk_metrics (
                        id SERIAL PRIMARY KEY,
                        date DATE,
                        daily_pnl DOUBLE PRECISION,
                        max_drawdown DOUBLE PRECISION,
                        total_exposure DOUBLE PRECISION,
                        var_95 DOUBLE PRECISION,
                        sharpe_ratio DOUBLE PRECISION,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                """)
                
                # Market analysis table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS market_analysis (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        analysis_data JSONB,
                        trend_strength DOUBLE PRECISION,
                        volatility_rank DOUBLE PRECISION,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                """)
                
                # Add missing columns with enhanced features
                columns_to_add = {
                    "target_price_1": "DOUBLE PRECISION",
                    "target_price_2": "DOUBLE PRECISION", 
                    "target_price_3": "DOUBLE PRECISION",  # New target
                    "initial_quantity": "DOUBLE PRECISION",
                    "max_profit_reached": "DOUBLE PRECISION",  # Track max profit
                    "risk_score": "INTEGER",  # Risk assessment score
                    "market_sentiment": "TEXT",  # Market sentiment at entry
                    "technical_score": "INTEGER"  # Technical analysis score
                }
                
                for col, col_type in columns_to_add.items():
                    if not column_exists(cur, 'signals', col):
                        cur.execute(sql.SQL("ALTER TABLE signals ADD COLUMN {} {}").format(sql.Identifier(col), sql.SQL(col_type)))
                        logger.info(f"✅ [DB] Added enhanced column '{col}' to 'signals' table.")
                
            conn.commit()
            logger.info("✅ [DB] Enhanced database connection and schema updated successfully.")
            optimize_database()
            return
            
        except Exception as e:
            logger.error(f"❌ [DB] Error during initialization (Attempt {attempt + 1}/{retries}): {e}")
            if conn: conn.rollback()
            if attempt < retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info(f"[DB] Retrying connection in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.critical("❌ [DB] Failed to connect to the database after all retries. Exiting.")
                exit(1)

def check_db_connection() -> bool:
    global conn
    if conn is None or conn.closed != 0:
        logger.warning("[DB] Connection is None or closed. Re-initializing...")
        init_db()
    try:
        if conn and conn.closed == 0:
            with conn.cursor() as cur: 
                cur.execute("SELECT 1;")
            return True
        logger.warning("[DB] Connection check failed. It might still be closed.")
        return False
    except (OperationalError, InterfaceError) as e:
        logger.error(f"[DB] Connection lost ({e}). Attempting to reconnect...")
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

# --- دوال المساعدة والإشعارات المحسنة ---
def log_and_notify(level: str, message: str, notification_type: str, priority: int = 1):
    log_methods = {'info': logger.info, 'warning': logger.warning, 'error': logger.error}
    log_methods.get(level.lower(), logger.info)(message)
    
    if not check_db_connection() or not conn: 
        return
    
    try:
        new_notification = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": notification_type,
            "message": message,
            "priority": priority
        }
        
        with notifications_lock: 
            notifications_cache.appendleft(new_notification)
        
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO notifications (type, message, priority) VALUES (%s, %s, %s);",
                (notification_type, message, priority)
            )
        conn.commit()
        broadcast({"type": "new_notification", "payload": new_notification})
        
    except Exception as e:
        logger.error(f"❌ [DB] Failed to save notification: {e}")
        if conn: conn.rollback()

def log_rejection(symbol: str, reason_key: str, details: Optional[Dict] = None):
    try:
        reason_ar = ENHANCED_REJECTION_REASONS_AR.get(reason_key, reason_key)
        if details:
            details_str = ", ".join([f"{k}: {v}" for k, v in details.items()])
            reason_ar = f"{reason_ar} ({details_str})"
        
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "reason": reason_ar,
            "severity": "medium" if "محسن" in reason_ar else "low"
        }
        
        with rejection_logs_lock: 
            rejection_logs_cache.appendleft(log_entry)
        
        broadcast({"type": "new_rejection", "payload": log_entry})
        
    except Exception as e:
        logger.error(f"❌ [Log Rejection] Error logging rejection for {symbol}: {e}", exc_info=True)

# --- محسن دوال البيانات والمؤشرات ---
def fetch_historical_data(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    cache_key = f"{symbol}_{interval}_{days}"
    
    # Check cache first
    with indicator_cache_lock:
        if cache_key in indicator_cache:
            cache_time, cached_df = indicator_cache[cache_key]
            if datetime.now() - cache_time < timedelta(minutes=2):
                return cached_df.copy()
    
    time.sleep(API_REQUEST_DELAY)
    try:
        klines = client.get_historical_klines(symbol, interval, f"{days} day ago UTC")
        if not klines: 
            return None
            
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'number_of_trades', 
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df = df.dropna().astype(float)
        
        # Cache the result
        with indicator_cache_lock:
            indicator_cache[cache_key] = (datetime.now(), df.copy())
            # Keep cache size manageable
            if len(indicator_cache) > 100:
                oldest_key = min(indicator_cache.keys(), key=lambda k: indicator_cache[k][0])
                del indicator_cache[oldest_key]
        
        return df
        
    except Exception as e:
        logger.error(f"❌ [Data] Error fetching data for {symbol}: {e}")
        return None

def calculate_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """حساب المؤشرات التقنية المحسنة والمتقدمة"""
    df_calc = df.copy()
    
    try:
        # --- Moving Averages (Enhanced) ---
        for period in [5, 9, 13, 21, 34, 50, 100, 200]:
            df_calc[f'sma{period}'] = df_calc['close'].rolling(window=period).mean()
            df_calc[f'ema{period}'] = df_calc['close'].ewm(span=period, adjust=False).mean()
        
        # --- Advanced Volatility Indicators ---
        df_calc['atr'] = talib.ATR(df_calc['high'].values, df_calc['low'].values, df_calc['close'].values, timeperiod=14)
        df_calc['natr'] = talib.NATR(df_calc['high'].values, df_calc['low'].values, df_calc['close'].values, timeperiod=14)
        
        # Bollinger Bands with multiple periods
        for period in [20, 50]:
            upper, middle, lower = talib.BBANDS(df_calc['close'].values, timeperiod=period, nbdevup=2, nbdevdn=2)
            df_calc[f'bb_upper_{period}'] = upper
            df_calc[f'bb_middle_{period}'] = middle  
            df_calc[f'bb_lower_{period}'] = lower
            df_calc[f'bb_width_{period}'] = (upper - lower) / middle * 100
            df_calc[f'bb_position_{period}'] = (df_calc['close'] - lower) / (upper - lower) * 100
        
        # --- Enhanced Momentum Indicators ---
        df_calc['rsi'] = talib.RSI(df_calc['close'].values, timeperiod=14)
        df_calc['rsi_fast'] = talib.RSI(df_calc['close'].values, timeperiod=7)
        df_calc['rsi_slow'] = talib.RSI(df_calc['close'].values, timeperiod=21)
        
        # MACD with multiple timeframes
        macd, macdsignal, macdhist = talib.MACD(df_calc['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
        df_calc['macd'] = macd
        df_calc['macd_signal'] = macdsignal
        df_calc['macd_hist'] = macdhist
        
        # --- Stochastic Oscillators ---
        slowk, slowd = talib.STOCH(df_calc['high'].values, df_calc['low'].values, df_calc['close'].values,
                                  fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        df_calc['stoch_k'] = slowk
        df_calc['stoch_d'] = slowd
        
        # --- ADX and Directional Movement ---
        df_calc['adx'] = talib.ADX(df_calc['high'].values, df_calc['low'].values, df_calc['close'].values, timeperiod=14)
        df_calc['plus_di'] = talib.PLUS_DI(df_calc['high'].values, df_calc['low'].values, df_calc['close'].values, timeperiod=14)
        df_calc['minus_di'] = talib.MINUS_DI(df_calc['high'].values, df_calc['low'].values, df_calc['close'].values, timeperiod=14)
        
        # --- Volume Indicators (Enhanced) ---
        df_calc['volume_sma'] = df_calc['volume'].rolling(window=20).mean()
        df_calc['volume_ratio'] = df_calc['volume'] / df_calc['volume_sma']
        df_calc['ad'] = talib.AD(df_calc['high'].values, df_calc['low'].values, df_calc['close'].values, df_calc['volume'].values)
        df_calc['obv'] = talib.OBV(df_calc['close'].values, df_calc['volume'].values)
        
        # --- Price Action Indicators ---
        df_calc['price_change'] = df_calc['close'].pct_change() * 100
        df_calc['price_velocity'] = df_calc['price_change'].rolling(window=5).mean()
        df_calc['price_acceleration'] = df_calc['price_velocity'].diff()
        
        # --- Support and Resistance Levels ---
        df_calc['pivot'] = (df_calc['high'] + df_calc['low'] + df_calc['close']) / 3
        df_calc['resistance1'] = 2 * df_calc['pivot'] - df_calc['low']
        df_calc['support1'] = 2 * df_calc['pivot'] - df_calc['high']
        
        # --- Enhanced Trend Indicators ---
        df_calc['ema_spread'] = ((df_calc['ema9'] - df_calc['ema21']) / df_calc['ema21']) * 100
        df_calc['trend_strength'] = df_calc['adx'] * (1 if df_calc['plus_di'].iloc[-1] > df_calc['minus_di'].iloc[-1] else -1)
        
        # --- Market Structure ---
        df_calc['higher_high'] = df_calc['high'] > df_calc['high'].shift(1)
        df_calc['higher_low'] = df_calc['low'] > df_calc['low'].shift(1)
        df_calc['lower_high'] = df_calc['high'] < df_calc['high'].shift(1)
        df_calc['lower_low'] = df_calc['low'] < df_calc['low'].shift(1)
        
        # --- Ichimoku Cloud Components ---
        nine_period_high = df_calc['high'].rolling(window=9).max()
        nine_period_low = df_calc['low'].rolling(window=9).min()
        df_calc['tenkan_sen'] = (nine_period_high + nine_period_low) / 2
        
        twenty_six_period_high = df_calc['high'].rolling(window=26).max()
        twenty_six_period_low = df_calc['low'].rolling(window=26).min()
        df_calc['kijun_sen'] = (twenty_six_period_high + twenty_six_period_low) / 2
        
        df_calc['senkou_span_a'] = ((df_calc['tenkan_sen'] + df_calc['kijun_sen']) / 2).shift(26)
        
        fifty_two_period_high = df_calc['high'].rolling(window=52).max()
        fifty_two_period_low = df_calc['low'].rolling(window=52).min()
        df_calc['senkou_span_b'] = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)
        
        df_calc['chikou_span'] = df_calc['close'].shift(-26)
        
        # --- Fibonacci Levels (Dynamic) ---
        recent_high = df_calc['high'].rolling(window=50).max().iloc[-1]
        recent_low = df_calc['low'].rolling(window=50).min().iloc[-1]
        fib_diff = recent_high - recent_low
        
        df_calc['fib_236'] = recent_high - (fib_diff * 0.236)
        df_calc['fib_382'] = recent_high - (fib_diff * 0.382)
        df_calc['fib_500'] = recent_high - (fib_diff * 0.500)
        df_calc['fib_618'] = recent_high - (fib_diff * 0.618)
        df_calc['fib_786'] = recent_high - (fib_diff * 0.786)
        
        # --- Advanced Pattern Recognition ---
        df_calc['doji'] = talib.CDLDOJI(df_calc['open'].values, df_calc['high'].values, df_calc['low'].values, df_calc['close'].values)
        df_calc['hammer'] = talib.CDLHAMMER(df_calc['open'].values, df_calc['high'].values, df_calc['low'].values, df_calc['close'].values)
        df_calc['shooting_star'] = talib.CDLSHOOTINGSTAR(df_calc['open'].values, df_calc['high'].values, df_calc['low'].values, df_calc['close'].values)
        
        # --- Volatility Rank ---
        rolling_vol = df_calc['price_change'].rolling(window=20).std()
        vol_percentile = rolling_vol.rolling(window=252).rank(pct=True) * 100
        df_calc['volatility_rank'] = vol_percentile
        
        # --- Market Regime Detection ---
        short_vol = df_calc['price_change'].rolling(window=10).std()
        long_vol = df_calc['price_change'].rolling(window=50).std()
        df_calc['vol_regime'] = short_vol / long_vol
        
        # --- Momentum Quality Score ---
        momentum_factors = [
            (df_calc['rsi'] > 50).astype(int),
            (df_calc['macd'] > df_calc['macd_signal']).astype(int),
            (df_calc['close'] > df_calc['ema21']).astype(int),
            (df_calc['adx'] > 20).astype(int),
            (df_calc['volume_ratio'] > 1.2).astype(int)
        ]
        df_calc['momentum_score'] = sum(momentum_factors) / len(momentum_factors) * 100
        
        return df_calc.fillna(method='bfill').fillna(method='ffill')
        
    except Exception as e:
        logger.error(f"❌ [Indicators] Error calculating enhanced features: {e}")
        return df

# --- استراتيجيات التداول المحسنة ---
def check_enhanced_bb_stoch_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """استراتيجية البولينجر باند والستوكاستيك المحسنة"""
    try:
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # التحقق من توفر البيانات المطلوبة
        required_cols = ['bb_lower_20', 'bb_upper_20', 'bb_position_20', 'stoch_k', 'stoch_d', 'rsi', 'volume_ratio', 'adx']
        if not all(col in df.columns for col in required_cols):
            return False
        
        # شروط متعددة الإطارات الزمنية
        if mtf_trend.get('5m', {}).get('trend') == 'bearish' and mtf_trend.get('15m', {}).get('trend') == 'bearish':
            log_rejection(df.name, "Enhanced BB: Multiple timeframe bearish conflict")
            return False
        
        # شروط البولينجر باند المحسنة
        price_near_lower_bb = latest['bb_position_20'] < 25  # السعر قريب من الحد السفلي
        bb_squeeze = latest['bb_width_20'] < df['bb_width_20'].rolling(50).quantile(0.3)  # ضغط البولينجر
        
        # شروط الستوكاستيك المحسنة
        stoch_oversold = latest['stoch_k'] < 25 and latest['stoch_d'] < 25
        stoch_turning_up = latest['stoch_k'] > prev['stoch_k'] and latest['stoch_d'] > prev['stoch_d']
        
        # شروط إضافية للجودة
        rsi_oversold = 25 < latest['rsi'] < 45  # RSI في منطقة جذابة
        volume_confirmation = latest['volume_ratio'] > 1.1  # تأكيد بالحجم
        trend_strength_ok = latest['adx'] > 15  # قوة اتجاه مقبولة
        
        # شروط الدخول المحسنة
        entry_conditions = [
            price_near_lower_bb,
            stoch_oversold,
            stoch_turning_up,
            rsi_oversold,
            volume_confirmation,
            trend_strength_ok,
            not bb_squeeze  # تجنب فترات الضغط الشديد
        ]
        
        # حساب نقاط الجودة
        quality_score = sum(entry_conditions) / len(entry_conditions) * 100
        
        if quality_score >= 0.75:  # 75% من الشروط يجب أن تتحقق
            return True
        else:
            log_rejection(df.name, "Enhanced BB: Quality score insufficient", {"score": f"{quality_score:.1f}%"})
            return False
            
    except Exception as e:
        logger.error(f"❌ [Enhanced BB Strategy] Error for {df.name}: {e}")
        return False

def check_enhanced_macd_ema_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """استراتيجية MACD وEMA المحسنة"""
    try:
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        # التحقق من توفر البيانات
        required_cols = ['macd', 'macd_signal', 'macd_hist', 'ema9', 'ema21', 'ema50', 'rsi', 'adx', 'volume_ratio']
        if not all(col in df.columns for col in required_cols):
            return False
        
        # تحليل الاتجاه متعدد الإطارات
        bullish_timeframes = sum(1 for tf_data in mtf_trend.values() if tf_data.get('trend') == 'bullish')
        if bullish_timeframes < 2:
            log_rejection(df.name, "Enhanced MACD: Insufficient bullish timeframes")
            return False
        
        # شروط MACD المحسنة
        macd_cross_up = latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']
        macd_hist_increasing = latest['macd_hist'] > prev['macd_hist'] > prev2['macd_hist']
        macd_above_zero = latest['macd'] > 0  # تفضيل MACD فوق الصفر
        
        # شروط EMA المحسنة
        ema_bullish_order = latest['ema9'] > latest['ema21'] > latest['ema50']
        price_above_ema9 = latest['close'] > latest['ema9']
        ema_spread_good = ((latest['ema9'] - latest['ema21']) / latest['ema21'] * 100) > 0.5
        
        # شروط الزخم والحجم
        rsi_momentum = 45 < latest['rsi'] < 75  # RSI في منطقة زخم جيدة
        volume_surge = latest['volume_ratio'] > 1.3  # زيادة كبيرة في الحجم
        trend_strength = latest['adx'] > 20  # قوة اتجاه جيدة
        
        # حتأكد من عدم وجود تباعد سلبي
        price_trend = latest['close'] > df['close'].iloc[-5]  # السعر في اتجاه صاعد
        macd_trend = latest['macd'] > df['macd'].iloc[-5]  # MACD في اتجاه صاعد
        no_divergence = price_trend == macd_trend
        
        # شروط الدخول
        entry_conditions = [
            macd_cross_up or macd_hist_increasing,
            ema_bullish_order,
            price_above_ema9,
            rsi_momentum,
            volume_surge,
            trend_strength,
            no_divergence,
            ema_spread_good
        ]
        
        quality_score = sum(entry_conditions) / len(entry_conditions) * 100
        
        if quality_score >= 0.70:
            return True
        else:
            log_rejection(df.name, "Enhanced MACD: Quality score insufficient", {"score": f"{quality_score:.1f}%"})
            return False
            
    except Exception as e:
        logger.error(f"❌ [Enhanced MACD Strategy] Error for {df.name}: {e}")
        return False

def check_advanced_breakout_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """استراتيجية الكسر المتقدمة - جديدة"""
    try:
        latest = df.iloc[-1]
        
        # التحقق من توفر البيانات
        required_cols = ['high', 'low', 'close', 'volume_ratio', 'atr', 'adx', 'rsi']
        if not all(col in df.columns for col in required_cols):
            return False
        
        # حساب مستويات الكسر
        resistance_period = 20
        support_period = 20
        
        resistance_level = df['high'].rolling(window=resistance_period).max().iloc[-2]  # مستوى المقاومة
        support_level = df['low'].rolling(window=support_period).min().iloc[-2]  # مستوى الدعم
        
        # شروط كسر المقاومة
        price_breakout = latest['close'] > resistance_level
        volume_confirmation = latest['volume_ratio'] > 1.5  # حجم قوي يؤكد الكسر
        
        # تأكيد قوة الكسر
        breakout_strength = (latest['close'] - resistance_level) / resistance_level * 100
        strong_breakout = breakout_strength > 0.5  # كسر بقوة أكثر من 0.5%
        
        # شروط الزخم
        rsi_momentum = 50 < latest['rsi'] < 80  # RSI يظهر زخم صاعد
        trend_strength = latest['adx'] > 25  # قوة اتجاه عالية
        
        # تجنب الكسرات الوهمية
        atr_filter = latest['atr'] > df['atr'].rolling(20).mean().iloc[-1] * 0.8  # تقلب كافي
        
        # شروط متعددة الإطارات
        higher_tf_bullish = mtf_trend.get('15m', {}).get('trend') in ['bullish', 'sideways']
        
        # شروط الدخول
        entry_conditions = [
            price_breakout,
            volume_confirmation,
            strong_breakout,
            rsi_momentum,
            trend_strength,
            atr_filter,
            higher_tf_bullish
        ]
        
        quality_score = sum(entry_conditions) / len(entry_conditions) * 100
        
        if quality_score >= 0.75:
            return True
        else:
            log_rejection(df.name, "Advanced Breakout: Quality score insufficient", {"score": f"{quality_score:.1f}%"})
            return False
            
    except Exception as e:
        logger.error(f"❌ [Advanced Breakout Strategy] Error for {df.name}: {e}")
        return False

def check_volume_profile_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """استراتيجية تحليل الحجم المتقدمة - جديدة"""
    try:
        latest = df.iloc[-1]
        
        # التحقق من توفر البيانات
        required_cols = ['volume', 'volume_ratio', 'close', 'obv', 'ad', 'rsi']
        if not all(col in df.columns for col in required_cols):
            return False
        
        # تحليل الحجم المتقدم
        volume_ma_short = df['volume'].rolling(window=5).mean().iloc[-1]
        volume_ma_long = df['volume'].rolling(window=20).mean().iloc[-1]
        volume_trend = volume_ma_short > volume_ma_long  # اتجاه الحجم صاعد
        
        # شروط الحجم المتقدمة
        volume_spike = latest['volume_ratio'] > 2.0  # ارتفاع حاد في الحجم
        sustained_volume = df['volume_ratio'].tail(3).mean() > 1.3  # حجم مستمر
        
        # تحليل OBV (On Balance Volume)
        obv_trend = latest['obv'] > df['obv'].rolling(20).mean().iloc[-1]
        obv_momentum = latest['obv'] > df['obv'].iloc[-5]  # OBV في زيادة
        
        # تحليل A/D Line (Accumulation/Distribution)
        ad_positive = latest['ad'] > df['ad'].iloc[-5]  # تراكم إيجابي
        
        # شروط السعر والزخم
        price_momentum = latest['close'] > df['close'].rolling(5).mean().iloc[-1]
        rsi_healthy = 40 < latest['rsi'] < 75  # RSI في منطقة صحية
        
        # تأكيد الاتجاه
        bullish_volume_price = (latest['close'] > df['close'].iloc[-1]) and (latest['volume'] > df['volume'].iloc[-1])
        
        # شروط الدخول
        entry_conditions = [
            volume_trend,
            volume_spike or sustained_volume,
            obv_trend,
            obv_momentum,
            ad_positive,
            price_momentum,
            rsi_healthy,
            bullish_volume_price
        ]
        
        quality_score = sum(entry_conditions) / len(entry_conditions) * 100
        
        if quality_score >= 0.70:
            return True
        else:
            log_rejection(df.name, "Volume Profile: Quality score insufficient", {"score": f"{quality_score:.1f}%"})
            return False
            
    except Exception as e:
        logger.error(f"❌ [Volume Profile Strategy] Error for {df.name}: {e}")
        return False

def check_ichimoku_strategy(df: pd.DataFrame, mtf_trend: Dict) -> bool:
    """استراتيجية الإيشيموكو المتقدمة - جديدة"""
    try:
        latest = df.iloc[-1]
        
        # التحقق من توفر البيانات
        required_cols = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'close', 'rsi']
        if not all(col in df.columns for col in required_cols):
            return False
        
        # شروط الإيشيموكو الأساسية
        tenkan_above_kijun = latest['tenkan_sen'] > latest['kijun_sen']  # Tenkan فوق Kijun
        price_above_cloud = latest['close'] > max(latest['senkou_span_a'], latest['senkou_span_b'])  # السعر فوق السحابة
        
        # قوة السحابة
        cloud_thickness = abs(latest['senkou_span_a'] - latest['senkou_span_b'])
        cloud_strong = cloud_thickness > latest['close'] * 0.005  # سماكة السحابة > 0.5% من السعر
        
        # اتجاه السحابة
        cloud_bullish = latest['senkou_span_a'] > latest['senkou_span_b']  # السحابة صاعدة
        
        # تقاطع Tenkan و Kijun
        prev = df.iloc[-2]
        golden_cross = (latest['tenkan_sen'] > latest['kijun_sen'] and 
                       prev['tenkan_sen'] <= prev['kijun_sen'])  # التقاطع الذهبي
        
        # قوة الزخم
        rsi_momentum = 45 < latest['rsi'] < 75  # RSI في منطقة زخم جيدة
        
        # المسافة من السحابة
        distance_from_cloud = min(abs(latest['close'] - latest['senkou_span_a']), 
                                 abs(latest['close'] - latest['senkou_span_b']))
        safe_distance = distance_from_cloud > latest['close'] * 0.002  # مسافة آمنة من السحابة
        
        # شروط الدخول
        entry_conditions = [
            tenkan_above_kijun,
            price_above_cloud,
            cloud_strong,
            cloud_bullish,
            rsi_momentum,
            safe_distance
        ]
        
        # إضافة نقاط إضافية للتقاطع الذهبي
        if golden_cross:
            entry_conditions.append(True)
        
        quality_score = sum(entry_conditions) / len(entry_conditions) * 100
        
        if quality_score >= 0.75:
            return True
        else:
            log_rejection(df.name, "Ichimoku: Quality score insufficient", {"score": f"{quality_score:.1f}%"})
            return False
            
    except Exception as e:
        logger.error(f"❌ [Ichimoku Strategy] Error for {df.name}: {e}")
        return False

# --- نظام إدارة المخاطر المتقدم ---
def calculate_position_size(symbol: str, entry_price: float, stop_loss: float, account_balance: float) -> float:
    """حساب حجم الصفقة بناءً على إدارة المخاطر المتقدمة"""
    try:
        # حساب المخاطرة لكل صفقة (2% من رأس المال)
        risk_per_trade = account_balance * 0.02
        
        # حساب المسافة للستوب لوس
        stop_distance = abs(entry_price - stop_loss) / entry_price
        
        # حساب حجم الصفقة
        notional_value = risk_per_trade / stop_distance
        
        # تطبيق حدود إضافية
        max_position_value = account_balance * 0.10  # حد أقصى 10% من رأس المال لكل صفقة
        notional_value = min(notional_value, max_position_value)
        
        return max(notional_value, FIXED_TRADE_AMOUNT_MIN_USDT)
        
    except Exception as e:
        logger.error(f"❌ [Position Size] Error calculating position size for {symbol}: {e}")
        return FIXED_TRADE_AMOUNT_MIN_USDT

def check_risk_limits() -> bool:
    """فحص حدود المخاطرة"""
    try:
        if not check_db_connection() or not conn:
            return True
        
        with conn.cursor() as cur:
            # فحص الخسارة اليومية
            today = datetime.now(timezone.utc).date()
            cur.execute("""
                SELECT COALESCE(SUM(profit_percentage), 0) as daily_pnl
                FROM signals 
                WHERE DATE(closed_at) = %s AND status = 'closed'
            """, (today,))
            
            daily_pnl = cur.fetchone()['daily_pnl'] or 0
            
            if daily_pnl <= -MAX_DAILY_LOSS_PERCENT:
                log_rejection("SYSTEM", "Max Daily Loss Reached", {"daily_pnl": f"{daily_pnl:.2f}%"})
                return False
            
            # فحص إجمالي التعرض
            cur.execute("""
                SELECT COUNT(*) as open_trades
                FROM signals 
                WHERE status IN ('open', 'updated')
            """)
            
            open_trades = cur.fetchone()['open_trades'] or 0
            
            with balance_lock:
                total_exposure = open_trades * FIXED_TRADE_AMOUNT_MAX_USDT
                exposure_percent = (total_exposure / usdt_balance * 100) if usdt_balance > 0 else 0
                
                if exposure_percent > MAX_TOTAL_EXPOSURE_PERCENT:
                    log_rejection("SYSTEM", "Max Exposure Exceeded", {"exposure": f"{exposure_percent:.1f}%"})
                    return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ [Risk Limits] Error checking risk limits: {e}")
        return True

def calculate_signal_quality_score(df: pd.DataFrame, strategy_name: str) -> int:
    """حساب نقاط جودة الإشارة المحسن"""
    try:
        latest = df.iloc[-1]
        score = 0
        max_score = 100
        
        # نقاط الاتجاه (20 نقطة)
        if latest.get('ema9', 0) > latest.get('ema21', 0) > latest.get('ema50', 0):
            score += 20
        elif latest.get('ema9', 0) > latest.get('ema21', 0):
            score += 10
        
        # نقاط الزخم (20 نقطة)
        rsi = latest.get('rsi', 50)
        if 45 <= rsi <= 75:
            score += 20
        elif 40 <= rsi <= 80:
            score += 10
        
        # نقاط الحجم (15 نقطة)
        volume_ratio = latest.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            score += 15
        elif volume_ratio > 1.2:
            score += 10
        elif volume_ratio > 1.0:
            score += 5
        
        # نقاط قوة الاتجاه (15 نقطة)
        adx = latest.get('adx', 0)
        if adx > 25:
            score += 15
        elif adx > 20:
            score += 10
        elif adx > 15:
            score += 5
        
        # نقاط التقلب (10 نقطة)
        volatility_rank = latest.get('volatility_rank', 50)
        if 30 <= volatility_rank <= 70:
            score += 10
        elif 20 <= volatility_rank <= 80:
            score += 5
        
        # نقاط خاصة بالاستراتيجية (20 نقطة)
        if "Enhanced_BB_Stoch" in strategy_name:
            if latest.get('bb_position_20', 50) < 25 and latest.get('stoch_k', 50) < 30:
                score += 20
        elif "Enhanced_MACD" in strategy_name:
            if latest.get('macd', 0) > latest.get('macd_signal', 0) and latest.get('macd_hist', 0) > 0:
                score += 20
        elif "Advanced_Breakout" in strategy_name:
            if latest.get('volume_ratio', 1) > 1.5 and latest.get('adx', 0) > 25:
                score += 20
        
        return min(score, max_score)
        
    except Exception as e:
        logger.error(f"❌ [Quality Score] Error calculating quality score: {e}")
        return 50

# --- دوال التداول المحسنة ---
def get_exchange_info_map() -> None:
    global exchange_info_map
    try:
        logger.info("[API] Fetching enhanced exchange info...")
        exchange_info_map = {s['symbol']: s for s in client.get_exchange_info()['symbols']}
        logger.info(f"✅ [API] Enhanced exchange info map created with {len(exchange_info_map)} symbols.")
    except Exception as e:
        logger.error(f"❌ [API] Error fetching exchange info: {e}")

def get_validated_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    try:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        if not os.path.exists(file_path):
            # إنشاء ملف العملات الافتراضي إذا لم يوجد
            default_symbols = [
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT',
                'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'XLMUSDT', 'XRPUSDT',
                'EOSUSDT', 'TRXUSDT', 'ETCUSDT', 'XMRUSDT', 'DASHUSDT',
                'ZECUSDT', 'ATOMUSDT', 'NEOUSDT', 'IOTAUSDT', 'ALGOUSDT'
            ]
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(default_symbols))
            logger.info(f"✅ Created default symbol list file: {filename}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        
        if not exchange_info_map: 
            get_exchange_info_map()
        
        active = {s for s, info in exchange_info_map.items() 
                 if info.get('quoteAsset') == 'USDT' and info.get('status') == 'TRADING'}
        
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ Found {len(validated)} valid symbols for enhanced trading.")
        return validated
        
    except Exception as e:
        logger.error(f"❌ [Symbols] Error validating symbols: {e}")
        return []

def get_mtf_trend(symbol: str) -> Dict:
    """تحليل الاتجاه متعدد الإطارات الزمنية"""
    try:
        mtf_trend = {}
        
        for tf in ['5m', '15m', '1h']:
            df = fetch_historical_data(symbol, tf, 5)
            if df is None or len(df) < 50:
                continue
            
            df = calculate_enhanced_features(df)
            latest = df.iloc[-1]
            
            # تحديد الاتجاه بناءً على عدة مؤشرات
            trend_indicators = []
            
            # EMA Trend
            if latest.get('ema9', 0) > latest.get('ema21', 0) > latest.get('ema50', 0):
                trend_indicators.append('bullish')
            elif latest.get('ema9', 0) < latest.get('ema21', 0) < latest.get('ema50', 0):
                trend_indicators.append('bearish')
            else:
                trend_indicators.append('sideways')
            
            # MACD Trend
            if latest.get('macd', 0) > latest.get('macd_signal', 0) and latest.get('macd_hist', 0) > 0:
                trend_indicators.append('bullish')
            elif latest.get('macd', 0) < latest.get('macd_signal', 0) and latest.get('macd_hist', 0) < 0:
                trend_indicators.append('bearish')
            else:
                trend_indicators.append('sideways')
            
            # ADX Trend
            adx = latest.get('adx', 0)
            plus_di = latest.get('plus_di', 0)
            minus_di = latest.get('minus_di', 0)
            
            if adx > 20:
                if plus_di > minus_di:
                    trend_indicators.append('bullish')
                else:
                    trend_indicators.append('bearish')
            else:
                trend_indicators.append('sideways')
            
            # تحديد الاتجاه النهائي
            bullish_count = trend_indicators.count('bullish')
            bearish_count = trend_indicators.count('bearish')
            
            if bullish_count >= 2:
                final_trend = 'bullish'
            elif bearish_count >= 2:
                final_trend = 'bearish'
            else:
                final_trend = 'sideways'
            
            mtf_trend[tf] = {
                'trend': final_trend,
                'strength': adx,
                'rsi': latest.get('rsi', 50)
            }
        
        return mtf_trend
        
    except Exception as e:
        logger.error(f"❌ [MTF Trend] Error analyzing trend for {symbol}: {e}")
        return {}

# --- واجهة المستخدم المحسنة ---
ENHANCED_DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة التحكم المحسنة - بوت التداول</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --success: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            --danger: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            --warning: linear-gradient(135deg, #f7b733 0%, #fc4a1a 100%);
            --info: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
            --dark: #2c3e50;
            --light: #ecf0f1;
            --background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            --surface: rgba(255, 255, 255, 0.1);
            --surface-light: rgba(255, 255, 255, 0.2);
            --text: #ffffff;
            --text-muted: rgba(255, 255, 255, 0.7);
            --border: rgba(255, 255, 255, 0.2);
            --shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            --glow: 0 0 20px rgba(116, 185, 255, 0.3);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--background);
            color: var(--text);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            animation: fadeInDown 1s ease;
        }
        
        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(45deg, #74b9ff, #00cec9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            text-shadow: var(--glow);
        }
        
        .header p {
            color: var(--text-muted);
            font-size: 1.1rem;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: var(--surface);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            transition: all 0.3s ease;
            animation: fadeInUp 0.8s ease;
            position: relative;
            overflow: hidden;
        }
        
        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--primary);
            transition: all 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: var(--shadow);
            border-color: rgba(255, 255, 255, 0.4);
        }
        
        .stat-card:hover::before {
            height: 5px;
            box-shadow: var(--glow);
        }
        
        .stat-icon {
            font-size: 2.5rem;
            margin-bottom: 15px;
            background: var(--primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 5px;
            color: var(--text);
        }
        
        .stat-label {
            color: var(--text-muted);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 12px 25px;
            border: none;
            border-radius: 50px;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
            position: relative;
            overflow: hidden;
            min-width: 140px;
        }
        
        .btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }
        
        .btn:hover::before {
            left: 100%;
        }
        
        .btn-primary {
            background: var(--primary);
            color: white;
        }
        
        .btn-success {
            background: var(--success);
            color: white;
        }
        
        .btn-danger {
            background: var(--danger);
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }
        
        .content-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .panel {
            background: var(--surface);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border);
            border-radius: 15px;
            padding: 25px;
            animation: fadeIn 1s ease;
        }
        
        .panel-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 20px;
            color: var(--text);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .panel-title i {
            color: #74b9ff;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 20px;
        }
        
        .table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        .table th,
        .table td {
            padding: 12px;
            text-align: right;
            border-bottom: 1px solid var(--border);
        }
        
        .table th {
            background: var(--surface-light);
            font-weight: 600;
            color: var(--text);
        }
        
        .table td {
            color: var(--text-muted);
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-left: 8px;
        }
        
        .status-active {
            background: #00b894;
            animation: pulse 2s infinite;
        }
        
        .status-inactive {
            background: #e17055;
        }
        
        .profit-positive {
            color: #00b894;
            font-weight: 600;
        }
        
        .profit-negative {
            color: #e17055;
            font-weight: 600;
        }
        
        .notification {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            border-right: 4px solid #74b9ff;
            animation: slideInRight 0.5s ease;
        }
        
        .notification-time {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-bottom: 5px;
        }
        
        .notification-message {
            color: var(--text);
        }
        
        .market-lights {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }
        
        .market-light {
            text-align: center;
            padding: 10px;
        }
        
        .light-circle {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            margin: 0 auto 8px;
            animation: pulse 2s infinite;
        }
        
        .light-bullish {
            background: #00b894;
        }
        
        .light-bearish {
            background: #e17055;
        }
        
        .light-sideways {
            background: #fdcb6e;
        }
        
        .scrollable {
            max-height: 400px;
            overflow-y: auto;
            scrollbar-width: thin;
            scrollbar-color: var(--border) transparent;
        }
        
        .scrollable::-webkit-scrollbar {
            width: 6px;
        }
        
        .scrollable::-webkit-scrollbar-track {
            background: transparent;
        }
        
        .scrollable::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 3px;
        }
        
        .floating-action {
            position: fixed;
            bottom: 30px;
            left: 30px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: var(--primary);
            border: none;
            color: white;
            font-size: 1.5rem;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: var(--shadow);
            z-index: 1000;
        }
        
        .floating-action:hover {
            transform: scale(1.1);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        @media (max-width: 768px) {
            .content-grid {
                grid-template-columns: 1fr;
            }
            
            .stats-grid {
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            }
            
            .controls {
                flex-direction: column;
                align-items: center;
            }
            
            .header h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-robot"></i> بوت التداول المحسن</h1>
            <p>نظام تداول ذكي ومتطور للعملات المشفرة</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-wallet"></i></div>
                <div class="stat-value" id="balance">$0.00</div>
                <div class="stat-label">الرصيد</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-chart-line"></i></div>
                <div class="stat-value" id="open-trades">0</div>
                <div class="stat-label">الصفقات المفتوحة</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-percentage"></i></div>
                <div class="stat-value" id="win-rate">0%</div>
                <div class="stat-label">معدل الربح</div>
            </div>
            
            <div class="stat-card">
                <div class="stat-icon"><i class="fas fa-trophy"></i></div>
                <div class="stat-value" id="total-profit">0%</div>
                <div class="stat-label">إجمالي الربح</div>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn btn-primary" onclick="toggleTrading()">
                <i class="fas fa-power-off"></i> <span id="trading-btn-text">تفعيل التداول</span>
            </button>
            <button class="btn btn-success" onclick="toggleMode()">
                <i class="fas fa-exchange-alt"></i> <span id="mode-btn-text">وضع ورقي</span>
            </button>
            <button class="btn btn-danger" onclick="showSettings()">
                <i class="fas fa-cog"></i> الإعدادات
            </button>
        </div>
        
        <div class="panel">
            <div class="panel-title">
                <i class="fas fa-chart-area"></i>
                حالة السوق
            </div>
            <div class="market-lights" id="market-lights">
                <div class="market-light">
                    <div class="light-circle light-sideways"></div>
                    <div>5 دقائق</div>
                </div>
                <div class="market-light">
                    <div class="light-circle light-sideways"></div>
                    <div>15 دقيقة</div>
                </div>
                <div class="market-light">
                    <div class="light-circle light-sideways"></div>
                    <div>ساعة</div>
                </div>
                <div class="market-light">
                    <div class="light-circle light-sideways"></div>
                    <div>4 ساعات</div>
                </div>
            </div>
        </div>
        
        <div class="content-grid">
            <div class="panel">
                <div class="panel-title">
                    <i class="fas fa-chart-line"></i>
                    الأداء
                </div>
                <div class="chart-container">
                    <canvas id="performanceChart"></canvas>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-title">
                    <i class="fas fa-bell"></i>
                    الإشعارات
                </div>
                <div class="scrollable" id="notifications">
                    <!-- Notifications will be populated here -->
                </div>
            </div>
        </div>
        
        <div class="content-grid">
            <div class="panel">
                <div class="panel-title">
                    <i class="fas fa-list"></i>
                    الصفقات المفتوحة
                </div>
                <div class="scrollable">
                    <table class="table" id="open-trades-table">
                        <thead>
                            <tr>
                                <th>العملة</th>
                                <th>السعر</th>
                                <th>الربح</th>
                                <th>الاستراتيجية</th>
                            </tr>
                        </thead>
                        <tbody>
                            <!-- Open trades will be populated here -->
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-title">
                    <i class="fas fa-times-circle"></i>
                    سجل الرفض
                </div>
                <div class="scrollable" id="rejections">
                    <!-- Rejections will be populated here -->
                </div>
            </div>
        </div>
    </div>
    
    <button class="floating-action" onclick="refreshData()">
        <i class="fas fa-sync-alt"></i>
    </button>
    
    <script>
        let socket = null;
        let performanceChart = null;
        
        function initializeWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            socket = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            socket.onopen = function() {
                console.log('WebSocket connected');
            };
            
            socket.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            socket.onclose = function() {
                console.log('WebSocket disconnected, attempting to reconnect...');
                setTimeout(initializeWebSocket, 3000);
            };
            
            socket.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }
        
        function handleWebSocketMessage(data) {
            switch(data.type) {
                case 'new_notification':
                    addNotification(data.payload);
                    break;
                case 'new_rejection':
                    addRejection(data.payload);
                    break;
                case 'market_state_update':
                    updateMarketState(data.payload);
                    break;
                case 'trade_closed':
                case 'signal_update':
                    refreshData();
                    break;
            }
        }
        
        function addNotification(notification) {
            const container = document.getElementById('notifications');
            const div = document.createElement('div');
            div.className = 'notification';
            div.innerHTML = `
                <div class="notification-time">${new Date(notification.timestamp).toLocaleString('ar-EG')}</div>
                <div class="notification-message">${notification.message}</div>
            `;
            container.insertBefore(div, container.firstChild);
            
            // Keep only latest 20 notifications
            while(container.children.length > 20) {
                container.removeChild(container.lastChild);
            }
        }
        
        function addRejection(rejection) {
            const container = document.getElementById('rejections');
            const div = document.createElement('div');
            div.className = 'notification';
            div.innerHTML = `
                <div class="notification-time">${new Date(rejection.timestamp).toLocaleString('ar-EG')}</div>
                <div class="notification-message">${rejection.symbol}: ${rejection.reason}</div>
            `;
            container.insertBefore(div, container.firstChild);
            
            // Keep only latest 30 rejections
            while(container.children.length > 30) {
                container.removeChild(container.lastChild);
            }
        }
        
        function updateMarketState(state) {
            const lights = document.getElementById('market-lights');
            const timeframes = ['5m', '15m', '1h', '4h'];
            
            timeframes.forEach((tf, index) => {
                const light = lights.children[index].querySelector('.light-circle');
                const trend = state.trend_details_by_tf[tf]?.trend || 'sideways';
                
                light.className = 'light-circle';
                light.classList.add(`light-${trend}`);
            });
        }
        
        function toggleTrading() {
            fetch('/toggle_trading', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    const btn = document.getElementById('trading-btn-text');
                    btn.textContent = data.trading_enabled ? 'إيقاف التداول' : 'تفعيل التداول';
                    refreshData();
                });
        }
        
        function toggleMode() {
            // Implementation for toggling trading mode
            console.log('Toggle mode clicked');
        }
        
        function showSettings() {
            window.location.href = '/settings';
        }
        
        function refreshData() {
            fetch('/api/dashboard')
                .then(response => response.json())
                .then(data => {
                    updateDashboard(data);
                })
                .catch(error => console.error('Error fetching data:', error));
        }
        
        function updateDashboard(data) {
            // Update stats
            document.getElementById('balance').textContent = `$${data.usdt_balance.toFixed(2)}`;
            document.getElementById('open-trades').textContent = data.open_trades.length;
            
            // Update notifications
            const notificationsContainer = document.getElementById('notifications');
            notificationsContainer.innerHTML = '';
            data.notifications.forEach(notification => {
                addNotification(notification);
            });
            
            // Update rejections
            const rejectionsContainer = document.getElementById('rejections');
            rejectionsContainer.innerHTML = '';
            data.rejections.forEach(rejection => {
                addRejection(rejection);
            });
            
            // Update market state
            if (data.market_state) {
                updateMarketState(data.market_state);
            }
            
            // Update open trades table
            updateOpenTradesTable(data.open_trades);
            
            // Update performance chart
            updatePerformanceChart(data.chart_data);
        }
        
        function updateOpenTradesTable(trades) {
            const tbody = document.querySelector('#open-trades-table tbody');
            tbody.innerHTML = '';
            
            trades.forEach(trade => {
                const row = document.createElement('tr');
                const profit = ((getCurrentPrice(trade.symbol) - trade.entry_price) / trade.entry_price * 100).toFixed(2);
                const profitClass = profit >= 0 ? 'profit-positive' : 'profit-negative';
                
                row.innerHTML = `
                    <td>${trade.symbol}</td>
                    <td>$${trade.entry_price.toFixed(4)}</td>
                    <td class="${profitClass}">${profit}%</td>
                    <td>${trade.strategy_name}</td>
                `;
                tbody.appendChild(row);
            });
        }
        
        function getCurrentPrice(symbol) {
            // This would be updated via WebSocket in real implementation
            return Math.random() * 100; // Placeholder
        }
        
        function updatePerformanceChart(chartData) {
            const ctx = document.getElementById('performanceChart').getContext('2d');
            
            if (performanceChart) {
                performanceChart.destroy();
            }
            
            performanceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['يناير', 'فبراير', 'مارس', 'أبريل', 'مايو', 'يونيو'],
                    datasets: [{
                        label: 'الأداء',
                        data: [0, 2.5, 1.8, 3.2, 4.1, 5.2],
                        borderColor: '#74b9ff',
                        backgroundColor: 'rgba(116, 185, 255, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            ticks: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            }
                        },
                        y: {
                            ticks: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            }
                        }
                    }
                }
            });
        }
        
        // Initialize everything when page loads
        document.addEventListener('DOMContentLoaded', function() {
            initializeWebSocket();
            refreshData();
            
            // Refresh data every 30 seconds
            setInterval(refreshData, 30000);
        });
    </script>
</body>
</html>
"""

# --- إضافة الدوال المطلوبة للتشغيل ---
def get_notification_settings() -> Dict:
    defaults = {
        'telegram_enabled': True,
        'email_enabled': False,
        'min_profit_notification': 1.0,
        'max_loss_notification': -1.0
    }
    if not redis_client:
        return defaults
    try:
        settings_data = redis_client.get('notification_settings')
        if settings_data:
            settings = json.loads(settings_data)
            for key, value in defaults.items():
                settings.setdefault(key, value)
            return settings
        return defaults
    except Exception as e:
        logger.error(f"❌ [Redis] Failed to get notification settings: {e}")
        return defaults

def send_enhanced_telegram_message(message: str, force: bool = False):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    
    settings = get_notification_settings()
    if not settings.get('telegram_enabled') and not force:
        return
    
    max_length = 4096
    messages = [message[i:i+max_length] for i in range(0, len(message), max_length)]
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    for msg in messages:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        for attempt in range(3):
            try:
                r = requests.post(url, data=payload, timeout=10)
                if r.status_code == 429:
                    retry_after = int(r.json().get("parameters", {}).get("retry_after", 1))
                    time.sleep(min(5, retry_after))
                    continue
                if r.ok:
                    break
                else:
                    logger.warning(f"[Telegram] HTTP {r.status_code}: {r.text}")
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    logger.error(f"❌ [Telegram] Failed to send message after retries: {e}")
                time.sleep(1.5)

def handle_socket_message(msg):
    global live_prices
    try:
        if msg and 'e' in msg and msg['e'] == 'error':
            logger.error(f"❌ [WebSocket] Error: {msg['m']}")
            return
        
        if isinstance(msg, list):
            price_updates = {}
            with live_prices_lock:
                for ticker in msg:
                    if 's' in ticker and 'c' in ticker:
                        symbol = ticker['s']
                        try:
                            price = float(ticker['c'])
                            live_prices[symbol] = price
                            price_updates[symbol] = price
                        except (ValueError, TypeError):
                            logger.warning(f"[WebSocket] Invalid price data for {symbol}: {ticker.get('c')}")
            
            if price_updates:
                broadcast({"type": "price_update", "payload": price_updates})
    except Exception as e:
        logger.error(f"❌ [WebSocket] Error processing message: {e}", exc_info=True)

def start_websocket():
    global ws_manager
    try:
        ws_manager = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
        ws_manager.start()
        ws_manager.start_ticker_socket(callback=handle_socket_message)
        logger.info("✅ [WebSocket] Enhanced WebSocket started successfully.")
    except Exception as e:
        logger.error(f"❌ [WebSocket] Error starting WebSocket: {e}")

# دوال إضافية مطلوبة للتشغيل
def load_open_signals_to_cache():
    """تحميل الإشارات المفتوحة إلى الكاش"""
    global open_signals_cache
    if not check_db_connection() or not conn:
        return
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE status IN ('open', 'updated')")
            signals = cur.fetchall()
            
        with signal_cache_lock:
            open_signals_cache.clear()
            for signal in signals:
                signal_dict = dict(signal)
                symbol = signal_dict['symbol']
                open_signals_cache[symbol] = signal_dict
                
        logger.info(f"✅ [Cache] Loaded {len(open_signals_cache)} open signals to cache.")
        
    except Exception as e:
        logger.error(f"❌ [Cache] Error loading open signals: {e}")

def load_notifications_to_cache():
    """تحميل الإشعارات إلى الكاش"""
    global notifications_cache
    if not check_db_connection() or not conn:
        return
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM notifications ORDER BY timestamp DESC LIMIT 30")
            notifications = cur.fetchall()
            
        with notifications_lock:
            notifications_cache.clear()
            for notification in notifications:
                notification_dict = dict(notification)
                notification_dict['timestamp'] = notification_dict['timestamp'].isoformat()
                notifications_cache.append(notification_dict)
                
        logger.info(f"✅ [Cache] Loaded {len(notifications_cache)} notifications to cache.")
        
    except Exception as e:
        logger.error(f"❌ [Cache] Error loading notifications: {e}")

def save_settings_to_redis():
    """حفظ الإعدادات في Redis"""
    if not redis_client:
        return
    
    try:
        settings = {
            'FIXED_TRADE_AMOUNT_MIN_USDT': FIXED_TRADE_AMOUNT_MIN_USDT,
            'FIXED_TRADE_AMOUNT_MAX_USDT': FIXED_TRADE_AMOUNT_MAX_USDT,
            'MAX_OPEN_TRADES': MAX_OPEN_TRADES,
            'MIN_SIGNAL_QUALITY': MIN_SIGNAL_QUALITY,
            'paper_trading_mode': paper_trading_mode,
            'strategies': {
                'USE_ENHANCED_BB_STOCH_STRATEGY': USE_ENHANCED_BB_STOCH_STRATEGY,
                'USE_ENHANCED_MACD_EMA_STRATEGY': USE_ENHANCED_MACD_EMA_STRATEGY,
                'USE_ENHANCED_EMA_RSI_STRATEGY': USE_ENHANCED_EMA_RSI_STRATEGY,
                'USE_ENHANCED_PULLBACK_STRATEGY': USE_ENHANCED_PULLBACK_STRATEGY,
                'USE_ENHANCED_MOMENTUM_STRATEGY': USE_ENHANCED_MOMENTUM_STRATEGY,
                'USE_ENHANCED_ELLIOTT_WAVE_STRATEGY': USE_ENHANCED_ELLIOTT_WAVE_STRATEGY,
                'USE_ENHANCED_RANGE_REVERSAL_STRATEGY': USE_ENHANCED_RANGE_REVERSAL_STRATEGY,
                'USE_ADVANCED_BREAKOUT_STRATEGY': USE_ADVANCED_BREAKOUT_STRATEGY,
                'USE_VOLUME_PROFILE_STRATEGY': USE_VOLUME_PROFILE_STRATEGY,
                'USE_ICHIMOKU_STRATEGY': USE_ICHIMOKU_STRATEGY
            }
        }
        
        redis_client.set('bot_settings', json.dumps(settings, cls=NpEncoder))
        logger.info("✅ [Redis] Settings saved successfully.")
        
    except Exception as e:
        logger.error(f"❌ [Redis] Error saving settings: {e}")

def load_settings_from_redis():
    """تحميل الإعدادات من Redis"""
    global FIXED_TRADE_AMOUNT_MIN_USDT, FIXED_TRADE_AMOUNT_MAX_USDT, MAX_OPEN_TRADES
    global MIN_SIGNAL_QUALITY, paper_trading_mode
    global USE_ENHANCED_BB_STOCH_STRATEGY, USE_ENHANCED_MACD_EMA_STRATEGY
    global USE_ENHANCED_EMA_RSI_STRATEGY, USE_ENHANCED_PULLBACK_STRATEGY
    global USE_ENHANCED_MOMENTUM_STRATEGY, USE_ENHANCED_ELLIOTT_WAVE_STRATEGY
    global USE_ENHANCED_RANGE_REVERSAL_STRATEGY, USE_ADVANCED_BREAKOUT_STRATEGY
    global USE_VOLUME_PROFILE_STRATEGY, USE_ICHIMOKU_STRATEGY
    
    if not redis_client:
        return
    
    try:
        settings_data = redis_client.get('bot_settings')
        if settings_data:
            settings = json.loads(settings_data)
            
            # Load basic settings
            with trade_amount_lock:
                FIXED_TRADE_AMOUNT_MIN_USDT = settings.get('FIXED_TRADE_AMOUNT_MIN_USDT', FIXED_TRADE_AMOUNT_MIN_USDT)
                FIXED_TRADE_AMOUNT_MAX_USDT = settings.get('FIXED_TRADE_AMOUNT_MAX_USDT', FIXED_TRADE_AMOUNT_MAX_USDT)
            
            MAX_OPEN_TRADES = settings.get('MAX_OPEN_TRADES', MAX_OPEN_TRADES)
            
            with min_quality_lock:
                MIN_SIGNAL_QUALITY = settings.get('MIN_SIGNAL_QUALITY', MIN_SIGNAL_QUALITY)
            
            with trading_mode_lock:
                paper_trading_mode = settings.get('paper_trading_mode', paper_trading_mode)
            
            # Load strategy settings
            strategies = settings.get('strategies', {})
            USE_ENHANCED_BB_STOCH_STRATEGY = strategies.get('USE_ENHANCED_BB_STOCH_STRATEGY', USE_ENHANCED_BB_STOCH_STRATEGY)
            USE_ENHANCED_MACD_EMA_STRATEGY = strategies.get('USE_ENHANCED_MACD_EMA_STRATEGY', USE_ENHANCED_MACD_EMA_STRATEGY)
            USE_ENHANCED_EMA_RSI_STRATEGY = strategies.get('USE_ENHANCED_EMA_RSI_STRATEGY', USE_ENHANCED_EMA_RSI_STRATEGY)
            USE_ENHANCED_PULLBACK_STRATEGY = strategies.get('USE_ENHANCED_PULLBACK_STRATEGY', USE_ENHANCED_PULLBACK_STRATEGY)
            USE_ENHANCED_MOMENTUM_STRATEGY = strategies.get('USE_ENHANCED_MOMENTUM_STRATEGY', USE_ENHANCED_MOMENTUM_STRATEGY)
            USE_ENHANCED_ELLIOTT_WAVE_STRATEGY = strategies.get('USE_ENHANCED_ELLIOTT_WAVE_STRATEGY', USE_ENHANCED_ELLIOTT_WAVE_STRATEGY)
            USE_ENHANCED_RANGE_REVERSAL_STRATEGY = strategies.get('USE_ENHANCED_RANGE_REVERSAL_STRATEGY', USE_ENHANCED_RANGE_REVERSAL_STRATEGY)
            USE_ADVANCED_BREAKOUT_STRATEGY = strategies.get('USE_ADVANCED_BREAKOUT_STRATEGY', USE_ADVANCED_BREAKOUT_STRATEGY)
            USE_VOLUME_PROFILE_STRATEGY = strategies.get('USE_VOLUME_PROFILE_STRATEGY', USE_VOLUME_PROFILE_STRATEGY)
            USE_ICHIMOKU_STRATEGY = strategies.get('USE_ICHIMOKU_STRATEGY', USE_ICHIMOKU_STRATEGY)
            
            logger.info("✅ [Redis] Settings loaded successfully.")
        
    except Exception as e:
        logger.error(f"❌ [Redis] Error loading settings: {e}")

def get_open_trades_details():
    """الحصول على تفاصيل الصفقات المفتوحة"""
    try:
        with signal_cache_lock:
            return list(open_signals_cache.values())
    except Exception as e:
        logger.error(f"❌ [Open Trades] Error getting open trades details: {e}")
        return []

def get_strategy_performance_stats():
    """إحصائيات أداء الاستراتيجيات"""
    if not check_db_connection() or not conn:
        return {}
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    strategy_name,
                    COUNT(*) as total_trades,
                    AVG(profit_percentage) as avg_profit,
                    SUM(CASE WHEN profit_percentage > 0 THEN 1 ELSE 0 END) as winning_trades
                FROM signals 
                WHERE status = 'closed' AND closed_at >= NOW() - INTERVAL '30 days'
                GROUP BY strategy_name
            """)
            
            stats = cur.fetchall()
            
            result = {}
            for stat in stats:
                strategy = stat['strategy_name']
                total = stat['total_trades'] or 0
                winning = stat['winning_trades'] or 0
                
                result[strategy] = {
                    'total_trades': total,
                    'win_rate': (winning / total * 100) if total > 0 else 0,
                    'avg_profit': stat['avg_profit'] or 0
                }
            
            return result
        
    except Exception as e:
        logger.error(f"❌ [Strategy Stats] Error getting strategy performance: {e}")
        return {}

def update_balance():
    """تحديث الرصيد"""
    try:
        balance_info = client.get_asset_balance(asset='USDT')
        with balance_lock:
            global usdt_balance
            usdt_balance = float(balance_info['free'])
        logger.info(f"✅ [Balance] Updated balance: ${usdt_balance:.2f}")
    except Exception as e:
        logger.error(f"❌ [Balance] Could not update USDT balance: {e}")

def update_balance_loop():
    """حلقة تحديث الرصيد"""
    logger.info("🚀 [Balance Updater] Starting enhanced balance update loop...")
    while True:
        try:
            update_balance()
        except Exception as e:
            logger.error(f"❌ [Balance Loop] Error: {e}", exc_info=True)
        time.sleep(60 * 3)  # تحديث كل 3 دقائق

# إضافة دوال إنشاء الصفقات والفحص
def create_trade_signal(symbol: str, df: pd.DataFrame, strategy_name: str):
    """إنشاء إشارة تداول محسنة"""
    try:
        latest = df.iloc[-1]
        
        # فحص الشروط الأساسية
        if not check_risk_limits():
            return
        
        # حساب السعر والأهداف
        entry_price = latest['close']
        atr = latest.get('atr', entry_price * 0.02)
        
        # حساب وقف الخسارة بناءً على ATR
        stop_loss = entry_price - (atr * 2)
        
        # حساب الأهداف
        target_1 = entry_price + (atr * 2)
        target_2 = entry_price + (atr * 4)
        target_3 = entry_price + (atr * 6)
        
        # حساب جودة الإشارة
        quality_score = calculate_signal_quality_score(df, strategy_name)
        
        if quality_score < MIN_SIGNAL_QUALITY:
            log_rejection(symbol, "Low Quality Signal", {"score": quality_score})
            return
        
        # حساب حجم الصفقة
        with balance_lock:
            current_balance = usdt_balance
        
        # تحديد قيمة الصفقة
        if paper_trading_mode:
            notional_value = PAPER_TRADE_FIXED_AMOUNT_USDT
            is_real = False
        else:
            notional_value = calculate_position_size(symbol, entry_price, stop_loss, current_balance)
            is_real = True
            
            # التحقق من الرصيد
            if notional_value > current_balance:
                log_rejection(symbol, "Insufficient Balance", {"required": notional_value, "available": current_balance})
                return
        
        # حساب الكمية
        quantity = notional_value / entry_price
        
        # إدخال الصفقة في قاعدة البيانات
        if not check_db_connection() or not conn:
            logger.error(f"❌ [Signal Creation] Database connection failed for {symbol}")
            return
        
        try:
            with conn.cursor() as cur:
                signal_details = {
                    'quality_score': quality_score,
                    'atr_percent': (atr / entry_price) * 100,
                    'strategy_confidence': quality_score,
                    'market_conditions': get_mtf_trend(symbol),
                    'risk_reward_ratio': (target_2 - entry_price) / (entry_price - stop_loss)
                }
                
                cur.execute("""
                    INSERT INTO signals (
                        symbol, entry_price, stop_loss, target_price_1, target_price_2, target_price_3,
                        strategy_name, signal_details, is_real_trade, quantity, initial_quantity,
                        risk_score, technical_score
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    symbol, entry_price, stop_loss, target_1, target_2, target_3,
                    strategy_name, json.dumps(signal_details, cls=NpEncoder), is_real,
                    quantity, quantity, quality_score, quality_score
                ))
                
                signal_id = cur.fetchone()['id']
                
            conn.commit()
            
            # إضافة إلى الكاش
            with signal_cache_lock:
                open_signals_cache[symbol] = {
                    'id': signal_id,
                    'symbol': symbol,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'target_price_1': target_1,
                    'target_price_2': target_2,
                    'target_price_3': target_3,
                    'strategy_name': strategy_name,
                    'signal_details': signal_details,
                    'is_real_trade': is_real,
                    'quantity': quantity,
                    'status': 'open'
                }
            
            # إرسال الإشعارات
            trade_type = "حقيقية" if is_real else "ورقية"
            log_and_notify("info", f"New {trade_type} trade opened for {symbol} using {strategy_name}", "TRADE_OPENED", priority=2)
            
            message = (
                f"🚀 *صفقة {trade_type} جديدة محسنة*\n\n"
                f"*العملة:* `{symbol}`\n"
                f"*الاستراتيجية:* `{ENHANCED_STRATEGY_NAMES.get(strategy_name, strategy_name)}`\n"
                f"*جودة الإشارة:* `{quality_score}/100`\n"
                f"*نقاط التقنية:* `{quality_score}/100`\n\n"
                f"*سعر الدخول:* `{entry_price:.4f}`\n"
                f"*وقف الخسارة:* `{stop_loss:.4f}`\n"
                f"*الهدف الأول:* `{target_1:.4f}`\n"
                f"*الهدف الثاني:* `{target_2:.4f}`\n"
                f"*الهدف الثالث:* `{target_3:.4f}`\n\n"
                f"*الكمية:* `{quantity:.4f}`\n"
                f"*قيمة الصفقة:* `${notional_value:.2f}`\n"
                f"*نسبة المخاطرة:* `{((entry_price - stop_loss) / entry_price * 100):.2f}%`\n"
                f"*نسبة الربح المحتملة:* `{((target_2 - entry_price) / entry_price * 100):.2f}%`"
            )
            
            send_enhanced_telegram_message(message, force=True)
            
            broadcast({
                "type": "new_signal", 
                "payload": open_signals_cache[symbol]
            })
            
            logger.info(f"✅ [Signal Created] {trade_type} signal created for {symbol} with quality {quality_score}/100")
            
        except Exception as e:
            logger.error(f"❌ [Signal Creation] Database error for {symbol}: {e}")
            if conn: conn.rollback()
            
    except Exception as e:
        logger.error(f"❌ [Signal Creation] Error creating signal for {symbol}: {e}", exc_info=True)

def main_bot_loop():
    """الحلقة الرئيسية المحسنة للبوت"""
    logger.info("🚀 [Main Loop] Starting enhanced main bot loop...")
    
    while True:
        try:
            with trading_status_lock:
                if not is_trading_enabled:
                    time.sleep(30)
                    continue
            
            logger.info("="*25 + " بدء دورة مسح محسنة " + "="*25)
            
            # فحص حدود المخاطرة قبل البدء
            if not check_risk_limits():
                logger.warning("⚠️ [Risk Management] Risk limits exceeded, skipping scan cycle")
                time.sleep(300)  # انتظار 5 دقائق قبل المحاولة مرة أخرى
                continue
            
            for symbol in validated_symbols_to_scan:
                try:
                    # فحص الحد الأقصى للصفقات المفتوحة
                    with signal_cache_lock:
                        if len(open_signals_cache) >= MAX_OPEN_TRADES:
                            logger.info(f"تم الوصول للحد الأقصى من الصفقات ({MAX_OPEN_TRADES}). توقف المسح.")
                            break
                        
                        # تجنب تكرار الصفقات لنفس العملة
                        if symbol in open_signals_cache:
                            continue
                    
                    # تحليل الاتجاه متعدد الإطارات
                    mtf_trend = get_mtf_trend(symbol)
                    
                    # جلب البيانات التاريخية
                    df = fetch_historical_data(symbol, SIGNAL_GENERATION_TIMEFRAME, SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df is None or len(df) < 200:
                        if df is not None:
                            log_rejection(symbol, "Insufficient Historical Data")
                        continue
                    
                    # حساب المؤشرات المحسنة
                    df_featured = calculate_enhanced_features(df)
                    df_featured.name = symbol
                    
                    # فحص الاستراتيجيات المحسنة
                    strategy_found = None
                    
                    if USE_ENHANCED_BB_STOCH_STRATEGY and check_enhanced_bb_stoch_strategy(df_featured, mtf_trend):
                        strategy_found = "Enhanced_BB_Stoch_Strategy"
                    elif USE_ENHANCED_MACD_EMA_STRATEGY and check_enhanced_macd_ema_strategy(df_featured, mtf_trend):
                        strategy_found = "Enhanced_MACD_EMA_Strategy"
                    elif USE_ADVANCED_BREAKOUT_STRATEGY and check_advanced_breakout_strategy(df_featured, mtf_trend):
                        strategy_found = "Advanced_Breakout_Strategy"
                    elif USE_VOLUME_PROFILE_STRATEGY and check_volume_profile_strategy(df_featured, mtf_trend):
                        strategy_found = "Volume_Profile_Strategy"
                    elif USE_ICHIMOKU_STRATEGY and check_ichimoku_strategy(df_featured, mtf_trend):
                        strategy_found = "Ichimoku_Strategy"
                    
                    # إنشاء إشارة تداول إذا تم العثور على استراتيجية مناسبة
                    if strategy_found:
                        create_trade_signal(symbol, df_featured, strategy_found)
                        time.sleep(2)  # انتظار قصير بين الصفقات
                
                except Exception as e:
                    logger.error(f"❌ [Main Loop] Error processing {symbol}: {e}")
                    continue
                
                # انتظار قصير بين العملات لتجنب تحميل API
                time.sleep(0.5)
            
            # انتظار قبل الدورة التالية
            logger.info("✅ [Main Loop] Scan cycle completed. Waiting for next cycle...")
            time.sleep(60)  # انتظار دقيقة واحدة بين الدورات
            
        except Exception as e:
            logger.error(f"❌ [Main Loop] Critical error in main loop: {e}", exc_info=True)
            time.sleep(60)

# --- مسارات Flask المحسنة ---
@app.route('/')
def dashboard():
    return render_template_string(ENHANCED_DASHBOARD_TEMPLATE)

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    try:
        with trading_status_lock:
            trading_enabled = is_trading_enabled
        with trading_mode_lock:
            is_paper_mode = paper_trading_mode
        with balance_lock:
            current_balance = usdt_balance
        with notifications_lock:
            notifications = list(notifications_cache)
        with rejection_logs_lock:
            rejections = list(rejection_logs_cache)
        with market_state_lock:
            market_state = dict(current_market_state)
        with min_quality_lock:
            min_quality = MIN_SIGNAL_QUALITY
        with trade_amount_lock:
            trade_amount_min = FIXED_TRADE_AMOUNT_MIN_USDT
            trade_amount_max = FIXED_TRADE_AMOUNT_MAX_USDT
        
        # الحصول على تفاصيل الصفقات المفتوحة
        open_trades = get_open_trades_details()
        
        # الحصول على إحصائيات أداء الاستراتيجيات
        strategy_stats = get_strategy_performance_stats()
        
        # بيانات الرسم البياني (بيانات تجريبية)
        chart_data = {
            "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
            "datasets": [{
                "label": "الأداء",
                "data": [0, 2.5, 1.8, 3.2, 4.1, 5.2],
                "borderColor": "#74b9ff",
                "backgroundColor": "rgba(116, 185, 255, 0.1)"
            }]
        }
        
        return jsonify({
            "trading_enabled": trading_enabled,
            "paper_trading_mode": is_paper_mode,
            "usdt_balance": current_balance,
            "notifications": notifications,
            "rejections": rejections,
            "market_state": market_state,
            "min_signal_quality": min_quality,
            "trade_amount_min": trade_amount_min,
            "trade_amount_max": trade_amount_max,
            "open_trades": open_trades,
            "strategy_stats": strategy_stats,
            "chart_data": chart_data,
            "server_time": datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ [API Dashboard] Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/toggle_trading', methods=['POST'])
def toggle_trading():
    global is_trading_enabled
    try:
        with trading_status_lock:
            is_trading_enabled = not is_trading_enabled
        
        status_msg = "enabled" if is_trading_enabled else "disabled"
        log_and_notify("info", f"Trading has been {status_msg}.", "TRADING_STATUS", priority=2)
        
        return jsonify({
            "status": "success",
            "trading_enabled": is_trading_enabled,
            "message": f"التداول تم {'تفعيله' if is_trading_enabled else 'إيقافه'}"
        })
        
    except Exception as e:
        logger.error(f"❌ [Toggle Trading] Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/health')
def api_health():
    try:
        with trading_status_lock:
            trading_enabled = is_trading_enabled
        with trading_mode_lock:
            is_paper = paper_trading_mode
        
        return jsonify({
            "status": "ok",
            "trading_enabled": trading_enabled,
            "mode": "PAPER" if is_paper else "REAL",
            "open_signals": len(open_signals_cache),
            "ws": {"connected": True},
            "version": "Enhanced 1.0.0"
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@sock.route('/ws')
def ws(ws_client):
    logger.info("Enhanced WebSocket client connected.")
    with ws_clients_lock:
        ws_clients.append(ws_client)
    
    try:
        ws_client.send(json.dumps({"type": "connection_established", "version": "Enhanced 1.0.0"}, cls=NpEncoder))
        
        while True:
            message = ws_client.receive(timeout=30)
            if message is None:
                ws_client.send(json.dumps({"type": "ping"}, cls=NpEncoder))
    except Exception:
        logger.info("Enhanced WebSocket client disconnected.")
    finally:
        with ws_clients_lock:
            if ws_client in ws_clients:
                ws_clients.remove(ws_client)

# --- نقطة بداية البرنامج المحسنة ---
if __name__ == '__main__':
    logger.info("="*60)
    logger.info("====== Starting Enhanced Crypto Trading Bot V2.0 ======")
    logger.info("="*60)
    
    # تهيئة قاعدة البيانات المحسنة
    init_db()
    
    # تهيئة Redis
    init_redis()
    
    # تهيئة Binance API
    try:
        client = Client(API_KEY, API_SECRET)
        client.ping()
        logger.info("✅ [Binance] Enhanced API connection successful.")
    except Exception as e:
        logger.critical(f"❌ [Binance] API connection failed: {e}")
        exit(1)
    
    # تحضير بيانات السوق
    get_exchange_info_map()
    validated_symbols_to_scan = get_validated_symbols()
    
    if not validated_symbols_to_scan:
        logger.critical("❌ No valid symbols to scan. Exiting.")
        exit(1)
    
    # تحميل البيانات المحفوظة
    load_open_signals_to_cache()
    load_notifications_to_cache() 
    load_settings_from_redis()
    
    # تحديث الرصيد الأولي
    logger.info("Fetching initial enhanced account balance...")
    update_balance()
    with balance_lock:
        logger.info(f"Initial enhanced balance fetched: ${usdt_balance:.2f}")
    
    logger.info("Enhanced initial data fetch complete.")
    
    # بدء الخدمات في الخلفية
    start_websocket()
    Thread(target=main_bot_loop, daemon=True).start()
    Thread(target=update_balance_loop, daemon=True).start()
    
    # بدء خادم Flask
    logger.info("🌐 [Flask] Starting Enhanced UI on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)