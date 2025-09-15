# =================================================================================================
# بوت تداول متكامل V37 - النسخة المحسنة والمُصححة
#
# الوصف: هذا الملف يحتوي على الكود البرمجي المحسن للبوت مع إصلاحات شاملة
# التحسينات:
# - إصلاح الأخطاء المنطقية والبرمجية
# - تحسين إدارة الذاكرة والأقفال
# - تطوير لوحة تحكم محسنة
# - إضافة استراتيجيات مكتملة
# - تحسين نظام إدارة المخاطر
# - إصلاح: تفعيل استراتيجيات (Pullback, Momentum) التي كانت غير نشطة
# - تحسين: إضافة استراتيجيات هيكلية (Elliott Wave, Range Reversal) لتسهيل التطوير المستقبلي
# - تحسين: جعل منفذ التشغيل (Port) قابلاً للتعديل عبر متغيرات البيئة
# =================================================================================================

import time
import os
import json
import logging
import threading
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any, Tuple
from collections import deque
from decimal import Decimal, ROUND_DOWN, getcontext
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

# External libraries with proper error handling
try:
    import requests
    import psycopg2
    import redis
    from psycopg2 import sql, OperationalError, InterfaceError
    from psycopg2.extras import RealDictCursor
    from binance.client import Client
    from binance import ThreadedWebsocketManager
    from binance.exceptions import BinanceAPIException
    from flask import Flask, jsonify, render_template_string, request
    from flask_cors import CORS
    from flask_sock import Sock
    from decouple import config
    from scipy.signal import argrelextrema
    import plotly.graph_objects as go
    import plotly.utils
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("Install with: pip install binance-python python-decouple flask flask-cors flask-sock psycopg2-binary redis pandas numpy requests scipy plotly")
    exit(1)

# --- إعدادات أساسية ---
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
getcontext().prec = 18

# إعداد نظام اللوجز المحسن
class ColoredFormatter(logging.Formatter):
    """مصنع ألوان لوجز تحسن قابلية القراءة"""
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m'  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_v37_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# تطبيق ألوان على console handler فقط
console_handler = logging.getLogger().handlers[1]
console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger('CryptoBotV37')

# --- كلاسات البيانات ---
@dataclass
class TradingSignal:
    """كلاس لتمثيل إشارة التداول"""
    symbol: str
    entry_price: float
    stop_loss: float
    target_price_1: float
    target_price_2: float
    strategy_name: str
    quality_score: int
    signal_details: Dict[str, Any]
    timestamp: datetime

    def to_dict(self) -> Dict:
        """تحويل الإشارة إلى قاموس"""
        return asdict(self)

@dataclass
class MarketState:
    """حالة السوق"""
    trend_5m: str = "unknown"
    trend_15m: str = "unknown"
    trend_1h: str = "unknown"
    adx_5m: float = 0.0
    adx_15m: float = 0.0
    adx_1h: float = 0.0
    rsi_5m: float = 50.0
    rsi_15m: float = 50.0
    rsi_1h: float = 50.0
    volatility: float = 0.0
    volume_trend: str = "normal"

# --- مشفر JSON محسن ---
class EnhancedJSONEncoder(json.JSONEncoder):
    """مشفر JSON محسن للتعامل مع أنواع البيانات المختلفة"""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, Decimal): return float(obj)
        if isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat()
        if hasattr(obj, 'to_dict'): return obj.to_dict()
        return super().default(obj)

# --- تحميل متغيرات البيئة مع معالجة أفضل للأخطاء ---
class ConfigManager:
    """مدير الإعدادات"""
    def __init__(self):
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, str]:
        """تحميل الإعدادات مع معالجة الأخطاء"""
        try:
            return {
                'API_KEY': config('BINANCE_API_KEY'),
                'API_SECRET': config('BINANCE_API_SECRET'),
                'DB_URL': config('DATABASE_URL'),
                'REDIS_URL': config('REDIS_URL', default='redis://localhost:6379/0'),
                'TELEGRAM_BOT_TOKEN': config('TELEGRAM_BOT_TOKEN', default=''),
                'TELEGRAM_CHAT_ID': config('TELEGRAM_CHAT_ID', default=''),
                'FLASK_SECRET_KEY': config('FLASK_SECRET_KEY', default='your-secret-key'),
                'ENVIRONMENT': config('ENVIRONMENT', default='development')
            }
        except Exception as e:
            logger.critical(f"❌ فشل في تحميل متغيرات البيئة: {e}")
            raise

config_manager = ConfigManager()

# --- مدير الحالة العامة للبوت ---
class BotState:
    """مدير الحالة العامة للبوت مع thread safety"""
    def __init__(self):
        self._lock = threading.RLock()
        self._trading_enabled = False
        self._paper_trading_mode = True
        self._usdt_balance = 0.0
        self._open_signals = {}
        self._live_prices = {}
        self._market_state = MarketState()
        self._notifications = deque(maxlen=100)
        self._rejections = deque(maxlen=200)

    @property
    def trading_enabled(self) -> bool:
        with self._lock:
            return self._trading_enabled

    @trading_enabled.setter
    def trading_enabled(self, value: bool):
        with self._lock:
            self._trading_enabled = value

    @property
    def paper_trading_mode(self) -> bool:
        with self._lock:
            return self._paper_trading_mode

    @paper_trading_mode.setter
    def paper_trading_mode(self, value: bool):
        with self._lock:
            self._paper_trading_mode = value

    @property
    def usdt_balance(self) -> float:
        with self._lock:
            return self._usdt_balance

    @usdt_balance.setter
    def usdt_balance(self, value: float):
        with self._lock:
            self._usdt_balance = value

    def get_open_signals(self) -> Dict:
        with self._lock:
            return self._open_signals.copy()

    def set_open_signal(self, symbol: str, signal: Dict):
        with self._lock:
            self._open_signals[symbol] = signal

    def remove_open_signal(self, symbol: str):
        with self._lock:
            self._open_signals.pop(symbol, None)

    def update_live_price(self, symbol: str, price: float):
        with self._lock:
            self._live_prices[symbol] = price

    def get_live_price(self, symbol: str) -> Optional[float]:
        with self._lock:
            return self._live_prices.get(symbol)

    def add_notification(self, notification: Dict):
        with self._lock:
            self._notifications.appendleft(notification)

    def add_rejection(self, rejection: Dict):
        with self._lock:
            self._rejections.appendleft(rejection)

    def get_notifications(self) -> List[Dict]:
        with self._lock:
            return list(self._notifications)

    def get_rejections(self) -> List[Dict]:
        with self._lock:
            return list(self._rejections)

# إنشاء مثيل حالة البوت
bot_state = BotState()

# --- إعدادات البوت القابلة للتعديل ---
class TradingConfig:
    """إعدادات التداول"""
    PAPER_TRADE_FIXED_AMOUNT_USDT: float = 10.0
    FIXED_TRADE_AMOUNT_MIN_USDT: float = 4.5
    FIXED_TRADE_AMOUNT_MAX_USDT: float = 6.5
    MAX_OPEN_TRADES: int = 3
    TRAILING_STOP_ACTIVATION_PROFIT_PERCENT: float = 1.0
    MIN_SIGNAL_QUALITY: int = 70
    AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE: bool = True

    # إعدادات الاستراتيجيات
    USE_BB_STOCH_STRATEGY: bool = True
    USE_MACD_EMA_STRATEGY: bool = True
    USE_EMA_RSI_STRATEGY: bool = True
    USE_PULLBACK_STRATEGY: bool = True
    USE_MOMENTUM_STRATEGY: bool = True # Renamed from MOMENTUM_VOLATILITY_STRATEGY
    USE_ELLIOTT_WAVE_STRATEGY: bool = True
    USE_RANGE_REVERSAL_STRATEGY: bool = True

    # إعدادات الأطر الزمنية
    SIGNAL_GENERATION_TIMEFRAME: str = '5m'
    SIGNAL_GENERATION_LOOKBACK_DAYS: int = 7
    BTC_SYMBOL: str = 'BTCUSDT'
    API_REQUEST_DELAY: float = 0.5

trading_config = TradingConfig()

# أسماء الاستراتيجيات
STRATEGY_NAMES = {
    "BB_Stoch_Strategy": "BB+Stoch (ارتداد مبكر)",
    "MACD_EMA_Strategy": "MACD+SMA (زخم وتقاطع)",
    "EMA_RSI_Strategy": "EMA+RSI (ارتداد سريع)",
    "Pullback_Strategy": "Pullback (ارتداد بحجم تداول)",
    "Momentum_Strategy": "Momentum (زخم متزايد)",
    "Elliott_Wave_Strategy": "Elliott Wave (موجات إليوت)",
    "Range_Reversal_Strategy": "Range Reversal (انعكاس نطاقي)"
}

# أسباب الرفض
REJECTION_REASONS_AR = {
    "Market Volatility Filter Failed": "فلتر تقلب السوق",
    "Insufficient Historical Data": "بيانات تاريخية غير كافية",
    "MinNotional Filter Failed": "قيمة الصفقة أقل من الحد الأدنى",
    "Insufficient Balance": "الرصيد غير كافي",
    "Low Quality Signal": "جودة الإشارة منخفضة",
    "Invalid Position Size": "حجم الصفقة غير صالح",
    "News Filter Failed": "فلتر الأخبار",
    "Liquidity Filter Failed": "فلتر السيولة",
    "Correlation Filter Failed": "فلتر الارتباط",
    "Trend Strength Filter Failed": "فلتر قوة الاتجاه",
    "DYN_BB_WIDTH_LOW": "ديناميكي: عرض البولينجر ضيق",
    "DYN_STOCH_LOW": "ديناميكي: ستوكاستيك منخفض",
    "DYN_VOLUME_LOW": "ديناميكي: حجم التداول منخفض",
    "DYN_ADX_LOW": "ديناميكي: قوة الاتجاه ضعيفة (ADX)",
    "DYN_MACD_MOMENTUM_LOW": "ديناميكي: زخم الماكد ضعيف",
    "DYN_RSI_OOR": "ديناميكي: RSI خارج النطاق",
    "EMA_RSI: Bearish long-term trend": "EMA_RSI: اتجاه هابط",
    "BB: Price below EMA50 (bearish trend)": "BB: السعر تحت EMA50",
    "MACD: Bearish trend": "MACD: اتجاه هابط",
}

# --- متغيرات النظام ---
conn: Optional[psycopg2.extensions.connection] = None
client: Optional[Client] = None
redis_client: Optional[redis.Redis] = None
ws_manager: Optional[ThreadedWebsocketManager] = None
exchange_info_map: Dict[str, Any] = {}
validated_symbols_to_scan: List[str] = []

# --- إعداد Flask ---
app = Flask(__name__)
app.secret_key = config_manager.config['FLASK_SECRET_KEY']
CORS(app)
sock = Sock(app)
ws_clients: List[Any] = []
ws_clients_lock = threading.Lock()

# ==============================================================================
# SECTION 1: قاعدة البيانات والخدمات الأساسية
# ==============================================================================

class DatabaseManager:
    """مدير قاعدة البيانات"""

    def __init__(self):
        self.conn = None
        self._init_connection()

    def _init_connection(self):
        """تهيئة الاتصال بقاعدة البيانات"""
        try:
            db_url = config_manager.config['DB_URL']
            if 'postgres' in db_url and 'sslmode' not in db_url:
                db_url += f"{'?' if '?' not in db_url else '&'}sslmode=require"

            self.conn = psycopg2.connect(db_url, connect_timeout=15, cursor_factory=RealDictCursor)
            self.conn.autocommit = False
            self._create_tables()
            logger.info("✅ [DB] Database connection initialized successfully.")
        except Exception as e:
            logger.critical(f"❌ [DB] Database initialization failed: {e}")
            raise

    def _create_tables(self):
        """إنشاء الجداول المطلوبة"""
        try:
            with self.conn.cursor() as cur:
                # جدول الإشارات
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id SERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        entry_price DOUBLE PRECISION NOT NULL,
                        stop_loss DOUBLE PRECISION NOT NULL,
                        target_price_1 DOUBLE PRECISION,
                        target_price_2 DOUBLE PRECISION,
                        status TEXT DEFAULT 'open',
                        closing_price DOUBLE PRECISION,
                        closed_at TIMESTAMP WITH TIME ZONE,
                        profit_percentage DOUBLE PRECISION,
                        strategy_name TEXT,
                        signal_details JSONB,
                        is_real_trade BOOLEAN DEFAULT FALSE,
                        quantity DOUBLE PRECISION,
                        initial_quantity DOUBLE PRECISION,
                        closing_reason TEXT,
                        order_id TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                """)

                # جدول الإشعارات
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS notifications (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        type TEXT NOT NULL,
                        message TEXT NOT NULL,
                        metadata JSONB
                    );
                """)

                # جدول إعدادات البوت
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS bot_settings (
                        id SERIAL PRIMARY KEY,
                        setting_key TEXT UNIQUE NOT NULL,
                        setting_value JSONB NOT NULL,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                """)

                # إنشاء فهارس لتحسين الأداء
                cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_notifications_timestamp ON notifications(timestamp);")

            self.conn.commit()
            logger.info("✅ [DB] Database schema created/verified successfully.")
        except Exception as e:
            logger.error(f"❌ [DB] Error creating tables: {e}")
            self.conn.rollback()
            raise

    def check_connection(self) -> bool:
        """فحص الاتصال بقاعدة البيانات"""
        if self.conn is None or self.conn.closed:
            logger.warning("[DB] Connection is closed. Reconnecting...")
            self._init_connection()
        return self.conn is not None and not self.conn.closed

    def execute_query(self, query: str, params: tuple = None, fetch: bool = False):
        """تنفيذ استعلام قاعدة البيانات"""
        if not self.check_connection():
            raise Exception("Database connection failed")

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                if fetch:
                    return cur.fetchall()
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ [DB] Query execution failed: {e}")
            raise

# إنشاء مثيل مدير قاعدة البيانات
db_manager = DatabaseManager()

class RedisManager:
    """مدير Redis للتخزين المؤقت"""

    def __init__(self):
        self.client = None
        self._init_connection()

    def _init_connection(self):
        """تهيئة الاتصال بـ Redis"""
        try:
            self.client = redis.from_url(config_manager.config['REDIS_URL'], decode_responses=True)
            self.client.ping()
            logger.info("✅ [Redis] Connected successfully.")
        except Exception as e:
            logger.warning(f"⚠️ [Redis] Could not connect to Redis: {e}. Running without cache.")
            self.client = None

    def set_value(self, key: str, value: Any, expire: int = None):
        """حفظ قيمة في Redis"""
        if self.client:
            try:
                serialized_value = json.dumps(value, cls=EnhancedJSONEncoder)
                self.client.set(key, serialized_value, ex=expire)
            except Exception as e:
                logger.error(f"❌ [Redis] Error setting value: {e}")

    def get_value(self, key: str) -> Any:
        """استرجاع قيمة من Redis"""
        if self.client:
            try:
                value = self.client.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.error(f"❌ [Redis] Error getting value: {e}")
        return None

# إنشاء مثيل مدير Redis
redis_manager = RedisManager()

class NotificationManager:
    """مدير الإشعارات"""

    def __init__(self):
        self.telegram_token = config_manager.config['TELEGRAM_BOT_TOKEN']
        self.telegram_chat_id = config_manager.config['TELEGRAM_CHAT_ID']

    def log_and_notify(self, level: str, message: str, notification_type: str, metadata: Dict = None):
        """إضافة لوج وإرسال إشعار"""
        # إضافة لوج
        log_methods = {
            'info': logger.info,
            'warning': logger.warning,
            'error': logger.error,
            'critical': logger.critical
        }
        log_methods.get(level.lower(), logger.info)(message)

        # إنشاء الإشعار
        notification = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": notification_type,
            "message": message,
            "metadata": metadata or {}
        }

        # إضافة للذاكرة المؤقتة
        bot_state.add_notification(notification)

        # حفظ في قاعدة البيانات
        try:
            db_manager.execute_query(
                "INSERT INTO notifications (type, message, metadata) VALUES (%s, %s, %s);",
                (notification_type, message, json.dumps(metadata, cls=EnhancedJSONEncoder))
            )
        except Exception as e:
            logger.error(f"❌ [DB] Failed to save notification: {e}")

        # إرسال عبر WebSocket
        self._broadcast_ws({"type": "new_notification", "payload": notification})

        # إرسال Telegram للإشعارات المهمة
        if level in ['warning', 'error', 'critical']:
            self._send_telegram_message(message)

    def log_rejection(self, symbol: str, reason_key: str, details: Optional[Dict] = None):
        """تسجيل رفض إشارة"""
        reason_ar = REJECTION_REASONS_AR.get(reason_key, reason_key)
        if details:
            details_str = ", ".join([f"{k}: {v}" for k, v in details.items()])
            reason_ar = f"{reason_ar} ({details_str})"

        rejection = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "reason": reason_ar,
            "details": details or {}
        }

        bot_state.add_rejection(rejection)
        self._broadcast_ws({"type": "new_rejection", "payload": rejection})

    def _send_telegram_message(self, message: str):
        """إرسال رسالة Telegram"""
        if not self.telegram_token or not self.telegram_chat_id:
            return

        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            requests.post(url, data=payload, timeout=10)
        except Exception as e:
            logger.error(f"❌ [Telegram] Failed to send message: {e}")

    def _broadcast_ws(self, data: Dict):
        """إرسال البيانات عبر WebSocket"""
        with ws_clients_lock:
            clients_to_remove = []
            for client in ws_clients:
                try:
                    if hasattr(client, 'send'):
                        client.send(json.dumps(data, cls=EnhancedJSONEncoder))
                except Exception:
                    clients_to_remove.append(client)

            # إزالة العملاء المنقطعين
            for client in clients_to_remove:
                if client in ws_clients:
                    ws_clients.remove(client)

# إنشاء مثيل مدير الإشعارات
notification_manager = NotificationManager()

# ==============================================================================
# SECTION 2: جلب البيانات والتحليل الفني
# ==============================================================================

class DataManager:
    """مدير البيانات"""

    def __init__(self):
        self.client = None
        self._init_client()

    def _init_client(self):
        """تهيئة عميل Binance"""
        try:
            self.client = Client(
                config_manager.config['API_KEY'],
                config_manager.config['API_SECRET']
            )
            self.client.ping()
            logger.info("✅ [API] Binance client initialized successfully.")
        except Exception as e:
            logger.critical(f"❌ [API] Failed to initialize Binance client: {e}")
            raise

    def get_exchange_info(self) -> Dict[str, Any]:
        """الحصول على معلومات البورصة"""
        try:
            exchange_info = self.client.get_exchange_info()
            return {s['symbol']: s for s in exchange_info['symbols']}
        except Exception as e:
            logger.error(f"❌ [API] Error fetching exchange info: {e}")
            return {}

    def get_validated_symbols(self, filename: str = 'crypto_list.txt') -> List[str]:
        """الحصول على قائمة الرموز المفعلة"""
        try:
            # قراءة الرموز من الملف
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    symbols = [line.strip().upper() + 'USDT' for line in f if line.strip()]
            else:
                # قائمة افتراضية
                symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT',
                          'DOGEUSDT', 'MATICUSDT', 'AVAXUSDT', 'LINKUSDT']

            # فلترة الرموز المفعلة
            if not exchange_info_map:
                exchange_info_map.update(self.get_exchange_info())

            valid_symbols = [
                s for s in symbols
                if s in exchange_info_map and exchange_info_map[s]['status'] == 'TRADING'
            ]

            logger.info(f"✅ Found {len(valid_symbols)} valid symbols for scanning.")
            return valid_symbols
        except Exception as e:
            logger.error(f"❌ Error getting validated symbols: {e}")
            return ['BTCUSDT', 'ETHUSDT']  # fallback

    def fetch_historical_data(self, symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
        """جلب البيانات التاريخية"""
        try:
            klines = self.client.get_historical_klines(
                symbol, interval, f"{days} day ago UTC"
            )
            if not klines:
                return None

            # تحويل إلى DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            # الاحتفاظ بالأعمدة المهمة فقط
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            # تحويل الأنواع
            for col in df.columns:
                if col != 'timestamp':
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)

            return df.dropna()
        except Exception as e:
            logger.error(f"❌ [Data] Error fetching data for {symbol}: {e}")
            return None

    def update_balance(self):
        """تحديث رصيد USDT"""
        if bot_state.paper_trading_mode:
            bot_state.usdt_balance = 1000.0  # رصيد وهمي
            return

        try:
            balance_info = self.client.get_asset_balance(asset='USDT')
            bot_state.usdt_balance = float(balance_info['free']) if balance_info else 0.0
        except Exception as e:
            logger.error(f"❌ [API] Failed to update balance: {e}")
            bot_state.usdt_balance = 0.0

# إنشاء مثيل مدير البيانات
data_manager = DataManager()

class TechnicalAnalysis:
    """محلل التحليل الفني"""

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """حساب المؤشرات الفنية"""
        df_calc = df.copy()

        # EMAs
        for span in [9, 21, 50, 200]:
            df_calc[f'ema{span}'] = df_calc['close'].ewm(span=span, adjust=False).mean()

        # ATR
        high_low = df_calc['high'] - df_calc['low']
        high_close = (df_calc['high'] - df_calc['close'].shift()).abs()
        low_close = (df_calc['low'] - df_calc['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df_calc['atr'] = tr.ewm(span=14, adjust=False).mean()
        df_calc['atr_percent'] = (df_calc['atr'] / df_calc['close']) * 100

        # ADX
        up_move = df_calc['high'].diff()
        down_move = -df_calc['low'].diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df_calc.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df_calc.index)
        plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
        minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / df_calc['atr'].replace(0, 1e-9)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
        df_calc['adx'] = dx.ewm(span=14, adjust=False).mean()

        # RSI
        delta = df_calc['close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss.replace(0, 1)
        df_calc['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        bb_middle = df_calc['close'].rolling(window=20).mean()
        bb_std = df_calc['close'].rolling(window=20).std()
        df_calc['bb_middle'] = bb_middle
        df_calc['bb_lower'] = bb_middle - (bb_std * 2)
        df_calc['bb_upper'] = bb_middle + (bb_std * 2)
        df_calc['bb_width'] = (df_calc['bb_upper'] - df_calc['bb_lower']) / bb_middle

        # MACD
        exp1 = df_calc['close'].ewm(span=12, adjust=False).mean()
        exp2 = df_calc['close'].ewm(span=26, adjust=False).mean()
        df_calc['macd'] = exp1 - exp2
        df_calc['macd_signal'] = df_calc['macd'].ewm(span=9, adjust=False).mean()
        df_calc['macd_hist'] = df_calc['macd'] - df_calc['macd_signal']

        # Stochastic
        low_14 = df_calc['low'].rolling(14).min()
        high_14 = df_calc['high'].rolling(14).max()
        df_calc['stoch_k'] = 100 * ((df_calc['close'] - low_14) / (high_14 - low_14).replace(0, 1))
        df_calc['stoch_d'] = df_calc['stoch_k'].rolling(3).mean()

        # Volume indicators
        df_calc['volume_sma'] = df_calc['volume'].rolling(20).mean()
        df_calc['volume_ratio'] = df_calc['volume'] / df_calc['volume_sma']

        return df_calc.dropna()

    @staticmethod
    def get_multi_timeframe_trend(symbol: str) -> MarketState:
        """الحصول على اتجاه السوق متعدد الأطر الزمنية"""
        market_state = MarketState()
        timeframes = {'5m': 7, '15m': 10, '1h': 15}

        for tf, days in timeframes.items():
            try:
                df = data_manager.fetch_historical_data(symbol, tf, days)
                if df is None or len(df) < 50:
                    continue

                df_calc = TechnicalAnalysis.calculate_indicators(df)
                last = df_calc.iloc[-1]

                # تحديد الاتجاه
                if last['close'] > last['ema50'] and last['ema21'] > last['ema50']:
                    trend = 'bullish'
                elif last['close'] < last['ema50'] and last['ema21'] < last['ema50']:
                    trend = 'bearish'
                else:
                    trend = 'sideways'

                # تحديث حالة السوق
                setattr(market_state, f'trend_{tf}', trend)
                setattr(market_state, f'adx_{tf}', float(last.get('adx', 0)))
                setattr(market_state, f'rsi_{tf}', float(last.get('rsi', 50)))

            except Exception as e:
                logger.debug(f"Error getting trend for {symbol} {tf}: {e}")

        # حساب التقلب العام
        try:
            df = data_manager.fetch_historical_data(symbol, '5m', 1)
            if df is not None and len(df) > 0:
                df_calc = TechnicalAnalysis.calculate_indicators(df)
                market_state.volatility = float(df_calc.iloc[-1].get('atr_percent', 0))
        except Exception:
            pass

        return market_state

# إنشاء مثيل المحلل الفني
technical_analysis = TechnicalAnalysis()

# ==============================================================================
# SECTION 3: الاستراتيجيات والفلاتر
# ==============================================================================

class TradingFilters:
    """فلاتر التداول"""

    @staticmethod
    def market_volatility_filter(df: pd.DataFrame, symbol: str) -> bool:
        """فلتر تقلبات السوق"""
        if 'atr_percent' not in df.columns or df['atr_percent'].isnull().all():
            notification_manager.log_rejection(symbol, "Market Volatility Filter Failed", {"reason": "No ATR data"})
            return False

        last_atr_percent = float(df.iloc[-1].get('atr_percent', 0))
        ATR_PERCENT_MIN, ATR_PERCENT_MAX = 0.7, 2.5

        # فحص النطاق الأساسي
        if not (ATR_PERCENT_MIN <= last_atr_percent <= ATR_PERCENT_MAX):
            notification_manager.log_rejection(symbol, "Market Volatility Filter Failed", {"atr": f"{last_atr_percent:.2f}%"})
            return False

        # فحص التغيير المفاجئ في التقلبات
        if len(df) >= 20:
            atr_ma = df['atr_percent'].rolling(20).mean()
            atr_change = abs(last_atr_percent - atr_ma.iloc[-1])
            atr_change_threshold = df['atr_percent'].rolling(10).std().iloc[-1] * 2.0

            if atr_change > atr_change_threshold:
                notification_manager.log_rejection(symbol, "Market Volatility Filter Failed", {"reason": "Sudden volatility change"})
                return False

        return True

    @staticmethod
    def trend_strength_filter(df: pd.DataFrame, market_state: MarketState, symbol: str) -> bool:
        """فلتر قوة الاتجاه"""
        # يجب أن يكون الاتجاه صاعد في الأطر الزمنية القصيرة
        if market_state.trend_5m != 'bullish' or market_state.trend_15m != 'bullish':
            notification_manager.log_rejection(symbol, "Trend Strength Filter Failed", {"reason": "5m or 15m not bullish"})
            return False

        # إذا كان الاتجاه هابط في الساعة، نحتاج ADX قوي في 15 دقيقة
        if market_state.trend_1h == 'bearish' and market_state.adx_15m < 25:
            notification_manager.log_rejection(symbol, "Trend Strength Filter Failed", {"reason": "Weak 15m ADX against 1h bearish"})
            return False

        return True

    @staticmethod
    def liquidity_filter() -> bool:
        """فلتر السيولة (أوقات التداول)"""
        now = datetime.now(timezone.utc)

        # تجنب عطلات نهاية الأسبوع
        if now.weekday() >= 5:
            return False

        # تجنب الأوقات ضعيفة السيولة
        if now.hour >= 22 or now.hour <= 2:
            return False

        # تجنب تغيير اليوم UTC
        if (now.hour == 0 and now.minute <= 30) or (now.hour == 23 and now.minute >= 30):
            return False

        return True

    @staticmethod
    def volume_filter(df: pd.DataFrame, symbol: str, multiplier: float = 1.2) -> bool:
        """فلتر حجم التداول"""
        if len(df) < 20:
            return False

        last_volume = df.iloc[-1]['volume']
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]

        if last_volume < avg_volume * multiplier:
            notification_manager.log_rejection(symbol, "DYN_VOLUME_LOW", {"volume_ratio": f"{last_volume/avg_volume:.2f}"})
            return False

        return True

class TradingStrategies:
    """استراتيجيات التداول"""

    @staticmethod
    def bb_stoch_strategy(df: pd.DataFrame, market_state: MarketState, symbol: str) -> Tuple[bool, int]:
        """استراتيجية Bollinger Bands + Stochastic"""
        if len(df) < 50:
            return False, 0

        last, prev, prev2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]

        # شروط الاتجاه
        if not (last['close'] > last['ema50'] and last['ema21'] > last['ema50']):
            notification_manager.log_rejection(symbol, "BB: Price below EMA50 (bearish trend)")
            return False, 0

        # شرط الارتداد من البولينجر السفلي
        if not ((df['low'].tail(3) <= df['bb_lower'].tail(3)).any() and last['close'] > last['bb_lower']):
            return False, 0

        # شرط الستوكاستيك
        if not (prev2['stoch_k'] < 25 and prev['stoch_k'] < 30 and last['stoch_k'] > prev['stoch_k'] > 30):
            notification_manager.log_rejection(symbol, "DYN_STOCH_LOW")
            return False, 0

        # شرط عرض البولينجر
        if not (df['bb_width'].iloc[-1] > df['bb_width'].rolling(20).mean().iloc[-1] * 1.3):
            notification_manager.log_rejection(symbol, "DYN_BB_WIDTH_LOW")
            return False, 0

        # فلتر الحجم
        volume_multiplier = 1.2 + (last.get('atr_percent', 0) / 100)
        if not TradingFilters.volume_filter(df, symbol, volume_multiplier):
            return False, 0

        return True, 75

    @staticmethod
    def macd_ema_strategy(df: pd.DataFrame, market_state: MarketState, symbol: str) -> Tuple[bool, int]:
        """استراتيجية MACD + EMA"""
        if len(df) < 200:
            return False, 0

        last, prev = df.iloc[-1], df.iloc[-2]

        # شروط الاتجاه
        if not (last['ema21'] > last['ema50'] > last['ema200']):
            notification_manager.log_rejection(symbol, "MACD: Bearish trend")
            return False, 0

        # شرط تقاطع الماكد
        if not (prev['macd'] <= prev['macd_signal'] and last['macd'] > last['macd_signal'] and last['macd_hist'] > 0):
            return False, 0

        # شرط ADX
        adx_threshold = 22 if last.get('atr_percent', 0) > 1.5 else 18
        if not (last['adx'] > adx_threshold):
            notification_manager.log_rejection(symbol, "DYN_ADX_LOW")
            return False, 0

        # فلتر الحجم المتكيف
        volume_multiplier = 1.2 + (last.get('atr_percent', 0) / 75)
        if not TradingFilters.volume_filter(df, symbol, volume_multiplier):
            return False, 0

        # شرط زخم الماكد
        macd_momentum = df['macd_hist'].diff()
        momentum_threshold = macd_momentum.rolling(10).std().iloc[-1] * 0.4
        if not (macd_momentum.iloc[-1] > momentum_threshold):
            notification_manager.log_rejection(symbol, "DYN_MACD_MOMENTUM_LOW")
            return False, 0

        return True, 80

    @staticmethod
    def ema_rsi_strategy(df: pd.DataFrame, market_state: MarketState, symbol: str) -> Tuple[bool, int]:
        """استراتيجية EMA + RSI"""
        if len(df) < 200:
            return False, 0

        last = df.iloc[-1]

        # شروط الاتجاه طويل المدى
        if not (last['ema50'] > last['ema200'] and last['ema21'] > last['ema50'] and last['close'] > last['ema9']):
            notification_manager.log_rejection(symbol, "EMA_RSI: Bearish long-term trend")
            return False, 0

        # شرط الماكد الإيجابي
        if not (last['macd_hist'] > 0 and last['macd'] > last['macd_signal']):
            return False, 0

        # شرط RSI في النطاق المطلوب
        if not (35 < last['rsi'] < 70):
            notification_manager.log_rejection(symbol, "DYN_RSI_OOR")
            return False, 0

        # فلتر الحجم
        if not TradingFilters.volume_filter(df, symbol, 1.2):
            return False, 0

        return True, 78

    @staticmethod
    def pullback_strategy(df: pd.DataFrame, market_state: MarketState, symbol: str) -> Tuple[bool, int]:
        """استراتيجية الارتداد"""
        if len(df) < 50:
            return False, 0

        last = df.iloc[-1]

        # شرط الاتجاه العام صاعد
        if not (last['ema21'] > last['ema50'] and last['close'] > last['ema21']):
            return False, 0

        # شرط الارتداد: السعر لامس أو اقترب من EMA21
        price_to_ema21_distance = abs(last['close'] - last['ema21']) / last['close']
        if price_to_ema21_distance > 0.02:  # أكثر من 2%
            return False, 0

        # شرط RSI ليس في ذروة الشراء
        if last['rsi'] > 70:
            return False, 0

        # شرط الحجم العالي
        if not TradingFilters.volume_filter(df, symbol, 1.5):
            return False, 0

        return True, 72

    @staticmethod
    def momentum_strategy(df: pd.DataFrame, market_state: MarketState, symbol: str) -> Tuple[bool, int]:
        """استراتيجية الزخم"""
        if len(df) < 50:
            return False, 0

        last = df.iloc[-1]

        # شرط الاتجاه الصاعد
        if not (last['close'] > last['ema9'] > last['ema21']):
            return False, 0

        # شرط زخم قوي
        if not (last['macd_hist'] > 0 and df['macd_hist'].diff().iloc[-1] > 0):
            return False, 0

        # شرط ADX قوي
        if last['adx'] < 25:
            return False, 0

        # شرط RSI في منطقة الزخم
        if not (45 < last['rsi'] < 75):
            return False, 0

        # حجم تداول عالي
        if not TradingFilters.volume_filter(df, symbol, 1.8):
            return False, 0

        return True, 85
        
    @staticmethod
    def elliott_wave_strategy(df: pd.DataFrame, market_state: MarketState, symbol: str) -> Tuple[bool, int]:
        """(Placeholder) استراتيجية موجات إليوت"""
        # Note: A proper Elliott Wave implementation is highly complex and requires advanced pattern recognition.
        # This is a placeholder for future development.
        logger.debug(f"Elliott Wave strategy for {symbol} is not implemented yet.")
        return False, 0

    @staticmethod
    def range_reversal_strategy(df: pd.DataFrame, market_state: MarketState, symbol: str) -> Tuple[bool, int]:
        """(Placeholder) استراتيجية الانعكاس النطاقي"""
        # Note: This is a placeholder for future development.
        logger.debug(f"Range Reversal strategy for {symbol} is not implemented yet.")
        return False, 0

# ==============================================================================
# SECTION 4: إدارة المخاطر والصفقات
# ==============================================================================

class RiskManager:
    """مدير المخاطر"""

    @staticmethod
    def calculate_dynamic_stop_loss(df: pd.DataFrame, entry_price: float, strategy_name: str) -> float:
        """حساب وقف الخسارة الديناميكي"""
        last = df.iloc[-1]
        atr_value = last.get('atr', 0)
        atr_percent = last.get('atr_percent', 0)

        # مضاعف ATR حسب الاستراتيجية
        if strategy_name == "BB_Stoch_Strategy":
            atr_multiplier = 2.0 if atr_percent > 2.0 else 1.7
            support_level = df['low'].tail(5).min() * 0.992
            stop_loss = min(support_level, entry_price - (atr_value * atr_multiplier))
        elif strategy_name == "MACD_EMA_Strategy":
            atr_multiplier = 2.2 if atr_percent > 2.0 else 1.9
            stop_loss = min(last['ema21'], entry_price - (atr_value * atr_multiplier))
        elif strategy_name == "EMA_RSI_Strategy":
            atr_multiplier = 1.8
            stop_loss = entry_price - (atr_value * atr_multiplier)
        else:
            # افتراضي
            atr_multiplier = 2.0 if atr_percent > 2.0 else 1.7
            stop_loss = entry_price - (atr_value * atr_multiplier)

        # حد أقصى للخسارة
        max_loss_percent = 0.06 if atr_percent > 2.5 else 0.05
        max_stop_distance = entry_price * max_loss_percent

        return max(stop_loss, entry_price - max_stop_distance)

    @staticmethod
    def calculate_dynamic_take_profit(df: pd.DataFrame, entry_price: float, stop_loss: float, strategy_name: str) -> Tuple[float, float]:
        """حساب أهداف جني الأرباح الديناميكية"""
        risk_amount = entry_price - stop_loss
        if risk_amount <= 0:
            return (entry_price * 1.015, entry_price * 1.025)

        atr_percent = df.iloc[-1].get('atr_percent', 0)

        # مضاعف Risk/Reward حسب التقلبات
        rr_multiplier = 1.2 if atr_percent < 1.5 else 0.8 if atr_percent > 2.5 else 1.0

        # نسب Risk/Reward حسب الاستراتيجية
        strategy_rr = {
            "BB_Stoch_Strategy": (1.8, 3.2),
            "MACD_EMA_Strategy": (1.6, 2.8),
            "EMA_RSI_Strategy": (1.5, 2.5),
            "Pullback_Strategy": (2.0, 3.5),
            "Momentum_Strategy": (2.2, 4.0)
        }

        rr1, rr2 = strategy_rr.get(strategy_name, (1.6, 2.8))
        rr1 *= rr_multiplier
        rr2 *= rr_multiplier

        tp1 = entry_price + (risk_amount * rr1)
        tp2 = entry_price + (risk_amount * rr2)

        return tp1, tp2

    @staticmethod
    def calculate_position_size(entry_price: float, stop_loss: float, risk_percent: float = 2.0) -> float:
        """حساب حجم الصفقة"""
        if bot_state.paper_trading_mode:
            return trading_config.PAPER_TRADE_FIXED_AMOUNT_USDT / entry_price

        balance = bot_state.usdt_balance
        risk_amount = balance * (risk_percent / 100)
        price_risk = entry_price - stop_loss

        if price_risk <= 0:
            return 0.0

        position_value = risk_amount / (price_risk / entry_price)

        # حدود الصفقة
        min_value = trading_config.FIXED_TRADE_AMOUNT_MIN_USDT
        max_value = trading_config.FIXED_TRADE_AMOUNT_MAX_USDT
        position_value = max(min_value, min(max_value, position_value))

        return position_value / entry_price

class TradeManager:
    """مدير الصفقات"""

    @staticmethod
    def create_trading_signal(symbol: str, df: pd.DataFrame, strategy_name: str, quality_score: int) -> TradingSignal:
        """إنشاء إشارة تداول"""
        entry_price = float(df.iloc[-1]['close'])
        stop_loss = RiskManager.calculate_dynamic_stop_loss(df, entry_price, strategy_name)
        tp1, tp2 = RiskManager.calculate_dynamic_take_profit(df, entry_price, stop_loss, strategy_name)

        signal_details = {
            'quality_score': quality_score,
            'atr_percent': float(df.iloc[-1].get('atr_percent', 0)),
            'rsi': float(df.iloc[-1].get('rsi', 50)),
            'adx': float(df.iloc[-1].get('adx', 0)),
            'volume_ratio': float(df.iloc[-1]['volume'] / df['volume'].rolling(20).mean().iloc[-1])
        }

        return TradingSignal(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price_1=tp1,
            target_price_2=tp2,
            strategy_name=strategy_name,
            quality_score=quality_score,
            signal_details=signal_details,
            timestamp=datetime.now(timezone.utc)
        )

    @staticmethod
    def save_signal_to_db(signal: TradingSignal) -> int:
        """حفظ الإشارة في قاعدة البيانات"""
        try:
            query = """
                INSERT INTO signals (symbol, entry_price, stop_loss, target_price_1, target_price_2,
                                   strategy_name, signal_details, is_real_trade, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
            """

            result = db_manager.execute_query(
                query,
                (
                    signal.symbol, signal.entry_price, signal.stop_loss,
                    signal.target_price_1, signal.target_price_2, signal.strategy_name,
                    json.dumps(signal.signal_details, cls=EnhancedJSONEncoder),
                    not bot_state.paper_trading_mode, signal.timestamp
                ),
                fetch=True
            )

            signal_id = result[0]['id']

            # إضافة للذاكرة المؤقتة
            signal_dict = signal.to_dict()
            signal_dict['id'] = signal_id
            signal_dict['status'] = 'open'
            bot_state.set_open_signal(signal.symbol, signal_dict)

            return signal_id
        except Exception as e:
            logger.error(f"❌ Error saving signal to DB: {e}")
            raise

    @staticmethod
    def update_trailing_stops():
        """تحديث وقف الخسارة المتحرك"""
        open_signals = bot_state.get_open_signals()

        for symbol, signal in open_signals.items():
            try:
                current_price = bot_state.get_live_price(symbol)
                if not current_price:
                    continue

                entry_price = signal['entry_price']
                current_stop = signal['stop_loss']

                # حساب الربح الحالي
                profit_percent = ((current_price - entry_price) / entry_price) * 100

                # تفعيل الوقف المتحرك عند تحقيق ربح معين
                if profit_percent >= trading_config.TRAILING_STOP_ACTIVATION_PROFIT_PERCENT:
                    # جلب ATR الحالي
                    df = data_manager.fetch_historical_data(symbol, trading_config.SIGNAL_GENERATION_TIMEFRAME, 2)
                    if df is not None and len(df) >= 15:
                        df_calc = technical_analysis.calculate_indicators(df)
                        atr_value = df_calc.iloc[-1].get('atr', 0)

                        # حساب الوقف الجديد
                        new_stop = current_price - (atr_value * 1.5)

                        # تحديث إذا كان الوقف الجديد أفضل
                        if new_stop > current_stop:
                            # تحديث في قاعدة البيانات
                            db_manager.execute_query(
                                "UPDATE signals SET stop_loss = %s WHERE id = %s;",
                                (new_stop, signal['id'])
                            )

                            # تحديث في الذاكرة المؤقتة
                            signal['stop_loss'] = new_stop
                            bot_state.set_open_signal(symbol, signal)

                            logger.info(f"✅ [Trailing Stop] Updated SL for {symbol} to {new_stop:.4f}")
                            notification_manager._send_telegram_message(
                                f"🔄 *تحديث وقف متحرك لـ {symbol}*\nالوقف الجديد: `{new_stop:.4f}` | الربح الحالي: `{profit_percent:.2f}%`"
                            )

            except Exception as e:
                logger.error(f"❌ Error updating trailing stop for {symbol}: {e}")

    @staticmethod
    def manage_open_trades():
        """إدارة الصفقات المفتوحة"""
        open_signals = bot_state.get_open_signals()

        for symbol, signal in open_signals.items():
            try:
                current_price = bot_state.get_live_price(symbol)
                if not current_price:
                    continue

                entry_price = signal['entry_price']
                stop_loss = signal['stop_loss']
                tp1 = signal['target_price_1']
                tp2 = signal['target_price_2']
                status = signal.get('status', 'open')

                profit_percent = ((current_price - entry_price) / entry_price) * 100
                close_reason = None

                # فحص وقف الخسارة
                if current_price <= stop_loss:
                    close_reason = "stop_loss"

                # فحص الهدف الثاني (للصفقات المحدثة)
                elif status == 'updated' and current_price >= tp2:
                    close_reason = "take_profit_2"

                # فحص الخروج المبكر للصفقات الرابحة
                elif profit_percent > 0.5:
                    df = data_manager.fetch_historical_data(symbol, trading_config.SIGNAL_GENERATION_TIMEFRAME, 1)
                    if df is not None and len(df) >= 20:
                        df_calc = technical_analysis.calculate_indicators(df)
                        last = df_calc.iloc[-1]

                        # شروط الخروج المبكر
                        if (last['close'] < last['ema21'] and
                            last['macd_hist'] < 0 and
                            last['rsi'] < 45):
                            close_reason = "trend_change"

                # جني ربح جزئي عند الهدف الأول
                if status == 'open' and current_price >= tp1:
                    TradeManager._partial_take_profit(signal, current_price)
                    continue

                # إغلاق الصفقة
                if close_reason:
                    TradeManager._close_trade(signal, current_price, close_reason, profit_percent)

            except Exception as e:
                logger.error(f"❌ Error managing trade for {symbol}: {e}")

    @staticmethod
    def _partial_take_profit(signal: Dict, current_price: float):
        """جني ربح جزئي"""
        try:
            # تحديث الكمية إلى النصف وتحريك الوقف للدخول
            new_quantity = signal.get('quantity', 0) / 2
            new_stop = signal['entry_price']

            db_manager.execute_query("""
                UPDATE signals SET quantity = %s, stop_loss = %s, status = 'updated',
                closing_reason = 'take_profit_1_partial' WHERE id = %s;
            """, (new_quantity, new_stop, signal['id']))

            # تحديث في الذاكرة المؤقتة
            signal.update({
                'quantity': new_quantity,
                'stop_loss': new_stop,
                'status': 'updated'
            })
            bot_state.set_open_signal(signal['symbol'], signal)

            # إشعار
            notification_manager._send_telegram_message(
                f"✅ *جني ربح جزئي لـ {signal['symbol']}*\nتم إغلاق 50% وتحريك الوقف للدخول."
            )
            logger.info(f"✅ [TP1] Partial close for {signal['symbol']}")

        except Exception as e:
            logger.error(f"❌ Error in partial take profit: {e}")

    @staticmethod
    def _close_trade(signal: Dict, closing_price: float, close_reason: str, profit_percent: float):
        """إغلاق الصفقة"""
        try:
            # تحديث في قاعدة البيانات
            db_manager.execute_query("""
                UPDATE signals SET status = 'closed', closing_price = %s, closed_at = NOW(),
                profit_percentage = %s, closing_reason = %s WHERE id = %s;
            """, (closing_price, profit_percent, close_reason, signal['id']))

            # إزالة من الذاكرة المؤقتة
            bot_state.remove_open_signal(signal['symbol'])

            # إشعار
            status_icon = "✅" if profit_percent > 0 else "❌"
            notification_manager._send_telegram_message(
                f"{status_icon} *إغلاق صفقة {signal['symbol']}*\n"
                f"السبب: {close_reason}\n"
                f"الربح: `{profit_percent:.2f}%`"
            )

            logger.info(f"✅ [Trade Closed] {signal['symbol']} due to {close_reason}, P&L: {profit_percent:.2f}%")

        except Exception as e:
            logger.error(f"❌ Error closing trade: {e}")

# ==============================================================================
# SECTION 5: مولد الإشارات
# ==============================================================================

class SignalGenerator:
    """مولد إشارات التداول"""

    def __init__(self):
        # FIX: Added all implemented strategies to the dictionary
        self.strategies = {
            'BB_Stoch_Strategy': TradingStrategies.bb_stoch_strategy,
            'MACD_EMA_Strategy': TradingStrategies.macd_ema_strategy,
            'EMA_RSI_Strategy': TradingStrategies.ema_rsi_strategy,
            'Pullback_Strategy': TradingStrategies.pullback_strategy,
            'Momentum_Strategy': TradingStrategies.momentum_strategy,
            'Elliott_Wave_Strategy': TradingStrategies.elliott_wave_strategy,
            'Range_Reversal_Strategy': TradingStrategies.range_reversal_strategy,
        }

    def generate_signals(self):
        """توليد الإشارات"""
        if not bot_state.trading_enabled:
            return

        # فحص الرصيد والتحويل للورقي إذا لزم الأمر
        if not bot_state.paper_trading_mode:
            if bot_state.usdt_balance < trading_config.FIXED_TRADE_AMOUNT_MIN_USDT:
                if trading_config.AUTO_FALLBACK_TO_PAPER_ON_LOW_BALANCE:
                    bot_state.paper_trading_mode = True
                    notification_manager.log_and_notify(
                        'warning', "الرصيد منخفض، التحويل للتداول الورقي", "balance_warning"
                    )
                else:
                    return

        # البحث عن إشارات في الرموز المتاحة
        for symbol in validated_symbols_to_scan:
            try:
                # تخطي إذا كان لدينا صفقة مفتوحة أو وصلنا للحد الأقصى
                open_signals = bot_state.get_open_signals()
                if symbol in open_signals or len(open_signals) >= trading_config.MAX_OPEN_TRADES:
                    continue

                # جلب البيانات
                df = data_manager.fetch_historical_data(
                    symbol,
                    trading_config.SIGNAL_GENERATION_TIMEFRAME,
                    trading_config.SIGNAL_GENERATION_LOOKBACK_DAYS
                )

                if df is None or len(df) < 200:
                    continue

                # حساب المؤشرات
                df_calc = technical_analysis.calculate_indicators(df)

                # الحصول على حالة السوق
                market_state = technical_analysis.get_multi_timeframe_trend(symbol)

                # تطبيق الفلاتر العامة
                if not self._apply_general_filters(df_calc, market_state, symbol):
                    continue

                # فحص الاستراتيجيات
                strategy_result = self._check_strategies(df_calc, market_state, symbol)

                if strategy_result:
                    strategy_name, quality_score = strategy_result

                    # فحص جودة الإشارة
                    if quality_score < trading_config.MIN_SIGNAL_QUALITY:
                        notification_manager.log_rejection(
                            symbol, "Low Quality Signal", {"quality": quality_score}
                        )
                        continue

                    # إنشاء وحفظ الإشارة
                    signal = TradeManager.create_trading_signal(symbol, df_calc, strategy_name, quality_score)

                    # فحص صحة وقف الخسارة
                    if signal.stop_loss >= signal.entry_price:
                        notification_manager.log_rejection(symbol, "Invalid Position Size")
                        continue

                    # حفظ الإشارة
                    signal_id = TradeManager.save_signal_to_db(signal)

                    # إشعار
                    notification_manager.log_and_notify(
                        "info",
                        f"إشارة جديدة لـ {symbol} بواسطة {STRATEGY_NAMES.get(strategy_name, strategy_name)}",
                        "SIGNAL_FOUND",
                        {"symbol": symbol, "strategy": strategy_name, "quality": quality_score}
                    )

                    logger.info(f"✅ [Signal Generated] {symbol} | Strategy: {strategy_name} | Quality: {quality_score}")

            except Exception as e:
                logger.error(f"❌ Error generating signal for {symbol}: {e}")

    def _apply_general_filters(self, df: pd.DataFrame, market_state: MarketState, symbol: str) -> bool:
        """تطبيق الفلاتر العامة"""
        # فلتر السيولة
        if not TradingFilters.liquidity_filter():
            return False

        # فلتر التقلبات
        if not TradingFilters.market_volatility_filter(df, symbol):
            return False

        # فلتر قوة الاتجاه
        if not TradingFilters.trend_strength_filter(df, market_state, symbol):
            return False

        return True

    def _check_strategies(self, df: pd.DataFrame, market_state: MarketState, symbol: str) -> Optional[Tuple[str, int]]:
        """فحص الاستراتيجيات"""
        strategy_config = {
            'BB_Stoch_Strategy': trading_config.USE_BB_STOCH_STRATEGY,
            'MACD_EMA_Strategy': trading_config.USE_MACD_EMA_STRATEGY,
            'EMA_RSI_Strategy': trading_config.USE_EMA_RSI_STRATEGY,
            'Pullback_Strategy': trading_config.USE_PULLBACK_STRATEGY,
            'Momentum_Strategy': trading_config.USE_MOMENTUM_STRATEGY,
            'Elliott_Wave_Strategy': trading_config.USE_ELLIOTT_WAVE_STRATEGY,
            'Range_Reversal_Strategy': trading_config.USE_RANGE_REVERSAL_STRATEGY,
        }

        for strategy_name, is_enabled in strategy_config.items():
            if not is_enabled:
                continue

            strategy_func = self.strategies.get(strategy_name)
            if strategy_func:
                try:
                    success, quality = strategy_func(df, market_state, symbol)
                    if success:
                        return strategy_name, quality
                except Exception as e:
                    logger.error(f"❌ Error in strategy {strategy_name} for {symbol}: {e}")

        return None

# إنشاء مثيل مولد الإشارات
signal_generator = SignalGenerator()

# ==============================================================================
# SECTION 6: واجهة المستخدم (Flask)
# ==============================================================================

# قالب لوحة التحكم المحسن
DASHBOARD_TEMPLATE = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>لوحة التحكم - بوت التداول المحسن V37</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
:root{
  --bg: #0a0e1a; --panel: #1a1f2e; --accent: #00d4ff; --success: #00ff88;
  --warning: #ffb800; --danger: #ff5555; --muted: #8892b0; --border: #233554;
  --text: #e6f1ff; --text-dim: #a8b2d1;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: var(--bg); color: var(--text); line-height: 1.6;
  background-image:
    radial-gradient(circle at 20% 20%, rgba(0, 212, 255, 0.1) 0%, transparent 50%),
    radial-gradient(circle at 80% 80%, rgba(0, 255, 136, 0.1) 0%, transparent 50%);
}

.container {
  max-width: 1600px; margin: 0 auto; padding: 20px;
  display: flex; flex-direction: column; gap: 20px;
}

.header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 20px; background: var(--panel); border-radius: 16px;
  border: 1px solid var(--border); backdrop-filter: blur(10px);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.header h1 {
  font-size: 24px; font-weight: 700;
  background: linear-gradient(135deg, var(--accent), var(--success));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

.status-badges {
  display: flex; gap: 12px; align-items: center;
}

.badge {
  padding: 8px 16px; border-radius: 8px; font-size: 14px; font-weight: 600;
  border: 1px solid; display: flex; align-items: center; gap: 8px;
}

.badge.online { background: rgba(0, 255, 136, 0.2); border-color: var(--success); color: var(--success); }
.badge.offline { background: rgba(255, 85, 85, 0.2); border-color: var(--danger); color: var(--danger); }
.badge.paper { background: rgba(255, 184, 0, 0.2); border-color: var(--warning); color: var(--warning); }

.main-grid {
  display: grid; grid-template-columns: 1fr; gap: 20px;
}

@media (min-width: 1200px) {
  .main-grid { grid-template-columns: 2fr 1fr; }
}

.card {
  background: var(--panel); border: 1px solid var(--border); border-radius: 16px;
  backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
  overflow: hidden; transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.card:hover {
  transform: translateY(-2px); box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
}

.card-header {
  padding: 20px; border-bottom: 1px solid var(--border);
  display: flex; justify-content: space-between; align-items: center;
}

.card-title {
  font-size: 18px; font-weight: 600; color: var(--text);
}

.card-body { padding: 20px; }

.controls {
  display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;
}

.btn {
  padding: 10px 20px; border-radius: 8px; border: 1px solid var(--border);
  background: rgba(255, 255, 255, 0.05); color: var(--text);
  font-weight: 600; cursor: pointer; transition: all 0.2s ease;
  text-decoration: none; display: inline-flex; align-items: center; gap: 8px;
}

.btn:hover {
  background: rgba(255, 255, 255, 0.1); transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.btn.primary {
  background: linear-gradient(135deg, var(--accent), #0099cc);
  border-color: var(--accent); color: white;
}

.btn.success {
  background: linear-gradient(135deg, var(--success), #00cc6a);
  border-color: var(--success); color: white;
}

.signals-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 16px; min-height: 200px;
}

.signal-card {
  background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border);
  border-radius: 12px; padding: 20px; position: relative; overflow: hidden;
  transition: all 0.2s ease;
}

.signal-card:hover {
  background: rgba(255, 255, 255, 0.06); border-color: var(--accent);
}

.signal-header {
  display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;
}

.signal-symbol {
  font-size: 20px; font-weight: 700; color: var(--accent);
}

.signal-strategy {
  font-size: 12px; color: var(--muted); background: rgba(255, 255, 255, 0.1);
  padding: 4px 8px; border-radius: 4px;
}

.signal-price {
  font-size: 24px; font-weight: 700; text-align: center; margin: 16px 0;
}

.signal-progress {
  margin: 16px 0;
}

.progress-bar {
  height: 8px; background: rgba(255, 255, 255, 0.1); border-radius: 4px; overflow: hidden;
}

.progress-fill {
  height: 100%; transition: width 0.3s ease;
}

.signal-actions {
  display: flex; gap: 8px; justify-content: center; margin-top: 16px;
}

.metrics-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px; margin-bottom: 20px;
}

.metric-card {
  background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border);
  border-radius: 12px; padding: 20px; text-align: center;
}

.metric-value {
  font-size: 28px; font-weight: 700; margin-bottom: 8px;
}

.metric-label {
  font-size: 14px; color: var(--muted);
}

.switch {
  position: relative; display: inline-block; width: 60px; height: 34px;
}

.switch input { opacity: 0; width: 0; height: 0; }

.slider {
  position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
  background-color: #333; transition: .4s; border-radius: 34px;
}

.slider:before {
  position: absolute; content: ""; height: 26px; width: 26px; left: 4px; bottom: 4px;
  background-color: white; transition: .4s; border-radius: 50%;
}

input:checked + .slider { background-color: var(--success); }
input:checked + .slider:before { transform: translateX(26px); }

.loading {
  display: flex; align-items: center; justify-content: center; padding: 40px;
  color: var(--muted);
}

.spinner {
  width: 40px; height: 40px; border: 4px solid rgba(0, 212, 255, 0.3);
  border-top: 4px solid var(--accent); border-radius: 50%;
  animation: spin 1s linear infinite; margin-right: 16px;
}

@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

.table-container {
  max-height: 400px; overflow-y: auto; border-radius: 8px;
  border: 1px solid var(--border);
}

.table {
  width: 100%; border-collapse: collapse; font-size: 14px;
}

.table th, .table td {
  padding: 12px; text-align: right; border-bottom: 1px solid var(--border);
}

.table th {
  background: rgba(255, 255, 255, 0.05); font-weight: 600; color: var(--muted);
  position: sticky; top: 0; z-index: 1;
}

.table tr:hover {
  background: rgba(255, 255, 255, 0.03);
}

.text-success { color: var(--success); }
.text-danger { color: var(--danger); }
.text-warning { color: var(--warning); }
.text-muted { color: var(--muted); }

@media (max-width: 768px) {
  .container { padding: 16px; }
  .header { flex-direction: column; gap: 16px; text-align: center; }
  .controls { justify-content: center; }
  .signals-grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<div class="container">
  <div class="header">
    <h1>🚀 بوت التداول المحسن V37</h1>
    <div class="status-badges">
      <div class="badge" id="connectionStatus">
        <span>⚡</span>
        <span id="statusText">متصل</span>
      </div>
      <div class="badge" id="tradingMode">
        <span id="modeIcon">📄</span>
        <span id="modeText">ورقي</span>
      </div>
    </div>
  </div>

  <div class="main-grid">
    <div class="left-section">
      <div class="card">
        <div class="card-header">
          <h2 class="card-title">الصفقات المفتوحة</h2>
          <span class="badge" id="openTradesCount">0</span>
        </div>
        <div class="card-body">
          <div class="controls">
            <button class="btn" onclick="refreshSignals()">🔄 تحديث</button>
            <button class="btn" onclick="closeAllTrades()">❌ إغلاق الكل</button>
          </div>
          <div id="signalsContainer" class="signals-grid">
            <div class="loading">
              <div class="spinner"></div>
              <span>جاري تحميل الصفقات...</span>
            </div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <h2 class="card-title">أداء البوت</h2>
        </div>
        <div class="card-body">
          <div class="metrics-grid">
            <div class="metric-card">
              <div class="metric-value text-success" id="winRate">--</div>
              <div class="metric-label">معدل النجاح</div>
            </div>
            <div class="metric-card">
              <div class="metric-value" id="avgProfit">--</div>
              <div class="metric-label">متوسط الربح</div>
            </div>
            <div class="metric-card">
              <div class="metric-value" id="totalTrades">--</div>
              <div class="metric-label">إجمالي الصفقات</div>
            </div>
            <div class="metric-card">
              <div class="metric-value text-warning" id="currentBalance">--</div>
              <div class="metric-label">الرصيد الحالي</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="right-section">
      <div class="card">
        <div class="card-header">
          <h2 class="card-title">إعدادات التحكم</h2>
        </div>
        <div class="card-body">
          <div style="margin-bottom: 20px;">
            <label style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
              <span style="flex: 1;">تفعيل التداول</span>
              <label class="switch">
                <input type="checkbox" id="tradingToggle">
                <span class="slider"></span>
              </label>
            </label>

            <label style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
              <span style="flex: 1;">وضع التداول الحقيقي</span>
              <label class="switch">
                <input type="checkbox" id="realTradingToggle">
                <span class="slider"></span>
              </label>
            </label>
          </div>

          <div style="margin-bottom: 20px;">
            <label style="display: block; margin-bottom: 8px; color: var(--muted);">
              الحد الأدنى لجودة الإشارة: <span id="qualityValue">70</span>
            </label>
            <input type="range" id="qualitySlider" min="30" max="90" value="70"
                   style="width: 100%; accent-color: var(--accent);">
          </div>

          <div class="controls">
            <a href="/settings" class="btn">⚙️ الإعدادات</a>
            <a href="/backtest" class="btn">📊 اختبار خلفي</a>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <h2 class="card-title">سجل الأحداث</h2>
        </div>
        <div class="card-body">
          <div class="table-container">
            <table class="table">
              <thead>
                <tr><th>الوقت</th><th>النوع</th><th>الحدث</th></tr>
              </thead>
              <tbody id="eventsTable">
                <tr><td colspan="3" class="text-muted" style="text-align: center;">لا توجد أحداث</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <h2 class="card-title">أسباب الرفض</h2>
        </div>
        <div class="card-body">
          <div class="table-container">
            <table class="table">
              <thead>
                <tr><th>الوقت</th><th>الرمز</th><th>السبب</th></tr>
              </thead>
              <tbody id="rejectionsTable">
                <tr><td colspan="3" class="text-muted" style="text-align: center;">لا توجد رفضات</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
class TradingDashboard {
  constructor() {
    this.ws = null;
    this.signals = {};
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.init();
  }

  init() {
    this.loadInitialData();
    this.setupEventListeners();
    this.connectWebSocket();
    setInterval(() => this.loadInitialData(), 30000); // Refresh every 30 seconds
  }

  async loadInitialData() {
    try {
      const [dashboardData, signalsData, metricsData] = await Promise.all([
        fetch('/api/dashboard_data').then(r => r.json()),
        fetch('/api/open_signals').then(r => r.json()),
        fetch('/api/performance_metrics').then(r => r.json())
      ]);

      this.updateDashboardData(dashboardData);
      this.updateSignals(signalsData.signals || []);
      this.updateMetrics(metricsData);
      this.updateBalance(dashboardData.usdt_balance || 0);
    } catch (error) {
      console.error('خطأ في تحميل البيانات:', error);
      this.showError('فشل في تحميل البيانات الأولية');
    }
  }

  updateDashboardData(data) {
    document.getElementById('tradingToggle').checked = data.trading_enabled || false;
    document.getElementById('realTradingToggle').checked = !data.paper_trading_mode;
    document.getElementById('qualitySlider').value = data.min_signal_quality || 70;
    document.getElementById('qualityValue').textContent = data.min_signal_quality || 70;

    // Update trading mode badge
    const modeElement = document.getElementById('tradingMode');
    const modeIcon = document.getElementById('modeIcon');
    const modeText = document.getElementById('modeText');

    if (data.paper_trading_mode) {
      modeElement.className = 'badge paper';
      modeIcon.textContent = '📄';
      modeText.textContent = 'ورقي';
    } else {
      modeElement.className = 'badge online';
      modeIcon.textContent = '💰';
      modeText.textContent = 'حقيقي';
    }

    this.updateEventsTable(data.notifications || []);
    this.updateRejectionsTable(data.rejections || []);
  }
  
  updateBalance(balance) {
      document.getElementById('currentBalance').textContent = `$${parseFloat(balance).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
  }

  updateSignals(signals) {
    this.signals = {};
    signals.forEach(signal => {
      this.signals[signal.id] = signal;
    });

    const container = document.getElementById('signalsContainer');
    const countElement = document.getElementById('openTradesCount');

    countElement.textContent = signals.length;

    if (signals.length === 0) {
      container.innerHTML = `
        <div style="grid-column: 1 / -1; text-align: center; padding: 40px; color: var(--muted);">
          <div style="font-size: 48px; margin-bottom: 16px;">📊</div>
          <div>لا توجد صفقات مفتوحة حالياً</div>
        </div>
      `;
      return;
    }

    container.innerHTML = signals.map(signal => this.renderSignalCard(signal)).join('');
  }

  renderSignalCard(signal) {
    const currentPrice = signal.current_price || signal.entry_price;
    const profitPercent = ((currentPrice - signal.entry_price) / signal.entry_price * 100);
    const profitClass = profitPercent >= 0 ? 'text-success' : 'text-danger';
    const profitIcon = profitPercent >= 0 ? '📈' : '📉';

    // Calculate progress
    let progress = 0;
    let progressColor = 'rgba(0, 212, 255, 0.3)';

    if (currentPrice >= signal.entry_price && signal.target_price_1 > signal.entry_price) {
      progress = Math.min(100, ((currentPrice - signal.entry_price) / (signal.target_price_1 - signal.entry_price)) * 100);
      progressColor = 'linear-gradient(90deg, var(--success), #00cc6a)';
    } else if (currentPrice < signal.entry_price && signal.entry_price > signal.stop_loss) {
      progress = Math.min(100, ((signal.entry_price - currentPrice) / (signal.entry_price - signal.stop_loss)) * 100);
      progressColor = 'linear-gradient(90deg, var(--danger), #cc0000)';
    }

    const strategyName = signal.strategy_name?.replace(/_/g, ' ').replace('Strategy', '') || 'غير محدد';
    const qualityScore = signal.signal_details?.quality_score || 0;

    return `
      <div class="signal-card" data-signal-id="${signal.id}">
        <div class="signal-header">
          <div class="signal-symbol">${signal.symbol}</div>
          <div class="signal-strategy">${strategyName}</div>
        </div>

        <div class="signal-price ${profitClass}">
          ${profitIcon} ${this.formatPrice(currentPrice)}
        </div>

        <div style="text-align: center; margin: 8px 0;">
          <span class="${profitClass}" style="font-weight: 600;">
            ${profitPercent >= 0 ? '+' : ''}${profitPercent.toFixed(2)}%
          </span>
        </div>

        <div class="signal-progress">
          <div class="progress-bar">
            <div class="progress-fill" style="width: ${progress}%; background: ${progressColor};"></div>
          </div>
          <div style="display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); margin-top: 4px;">
            <span>SL: ${this.formatPrice(signal.stop_loss)}</span>
            <span>TP: ${this.formatPrice(signal.target_price_1)}</span>
          </div>
        </div>

        <div style="text-align: center; margin: 12px 0; font-size: 12px; color: var(--muted);">
          جودة الإشارة: <span style="color: ${qualityScore > 75 ? 'var(--success)' : qualityScore > 60 ? 'var(--warning)' : 'var(--danger)'}">${qualityScore}/100</span>
        </div>

        <div class="signal-actions">
          <button class="btn" onclick="dashboard.closeSignal(${signal.id})">❌ إغلاق</button>
          <button class="btn" onclick="dashboard.viewSignalDetails(${signal.id})">👁️ تفاصيل</button>
        </div>
      </div>
    `;
  }

  updateMetrics(metrics) {
    document.getElementById('winRate').textContent = `${metrics.win_rate?.toFixed(1) || '--'}%`;
    document.getElementById('avgProfit').textContent = `${metrics.avg_profit?.toFixed(2) || '--'}%`;
    document.getElementById('totalTrades').textContent = metrics.total_trades || '--';

    // Update profit color
    const avgProfitEl = document.getElementById('avgProfit');
    const avgProfit = metrics.avg_profit || 0;
    avgProfitEl.className = `metric-value ${avgProfit >= 0 ? 'text-success' : 'text-danger'}`;
  }

  updateEventsTable(events) {
    const tbody = document.getElementById('eventsTable');
    if (!events || events.length === 0) {
      tbody.innerHTML = '<tr><td colspan="3" class="text-muted" style="text-align: center;">لا توجد أحداث</td></tr>';
      return;
    }

    tbody.innerHTML = events.slice(0, 20).map(event => `
      <tr>
        <td>${this.formatTime(event.timestamp)}</td>
        <td><span class="badge" style="font-size: 10px; padding: 2px 6px;">${event.type}</span></td>
        <td>${event.message}</td>
      </tr>
    `).join('');
  }

  updateRejectionsTable(rejections) {
    const tbody = document.getElementById('rejectionsTable');
    if (!rejections || rejections.length === 0) {
      tbody.innerHTML = '<tr><td colspan="3" class="text-muted" style="text-align: center;">لا توجد رفضات</td></tr>';
      return;
    }

    tbody.innerHTML = rejections.slice(0, 20).map(rejection => `
      <tr>
        <td>${this.formatTime(rejection.timestamp)}</td>
        <td><code style="background: rgba(255,255,255,0.1); padding: 2px 4px; border-radius: 4px;">${rejection.symbol}</code></td>
        <td>${rejection.reason}</td>
      </tr>
    `).join('');
  }

  connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log('WebSocket متصل');
      this.reconnectAttempts = 0;
      document.getElementById('connectionStatus').className = 'badge online';
      document.getElementById('statusText').textContent = 'متصل';
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.handleWebSocketMessage(data);
      } catch (error) {
        console.error('خطأ في معالجة رسالة WebSocket:', error);
      }
    };

    this.ws.onclose = () => {
      console.log('WebSocket منقطع');
      document.getElementById('connectionStatus').className = 'badge offline';
      document.getElementById('statusText').textContent = 'منقطع';

      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        setTimeout(() => {
          this.reconnectAttempts++;
          this.connectWebSocket();
        }, 5000);
      }
    };
  }

  handleWebSocketMessage(data) {
    switch (data.type) {
      case 'price_update':
        this.updatePrices(data.payload);
        break;
      case 'new_signal':
        this.signals[data.payload.id] = data.payload;
        this.updateSignals(Object.values(this.signals));
        break;
      case 'signal_update':
        if (this.signals[data.payload.id]) {
          this.signals[data.payload.id] = { ...this.signals[data.payload.id], ...data.payload };
          this.updateSignals(Object.values(this.signals));
        }
        break;
      case 'trade_closed':
        delete this.signals[data.payload.signal_id];
        this.updateSignals(Object.values(this.signals));
        break;
      case 'new_notification':
        this.addNotification(data.payload);
        break;
      case 'new_rejection':
        this.addRejection(data.payload);
        break;
    }
  }

  updatePrices(prices) {
    Object.entries(prices).forEach(([symbol, price]) => {
      // Update signals with new prices
      Object.values(this.signals).forEach(signal => {
        if (signal.symbol === symbol) {
          signal.current_price = price;
        }
      });
    });

    // Re-render signals with updated prices
    this.updateSignals(Object.values(this.signals));
  }

  addNotification(notification) {
    const tbody = document.getElementById('eventsTable');
    const newRow = document.createElement('tr');
    newRow.innerHTML = `
      <td>${this.formatTime(notification.timestamp)}</td>
      <td><span class="badge" style="font-size: 10px; padding: 2px 6px;">${notification.type}</span></td>
      <td>${notification.message}</td>
    `;

    if (tbody.firstChild?.textContent?.includes('لا توجد أحداث')) {
      tbody.innerHTML = '';
    }

    tbody.insertBefore(newRow, tbody.firstChild);

    // Keep only last 20 rows
    while (tbody.children.length > 20) {
      tbody.removeChild(tbody.lastChild);
    }
  }

  addRejection(rejection) {
    const tbody = document.getElementById('rejectionsTable');
    const newRow = document.createElement('tr');
    newRow.innerHTML = `
      <td>${this.formatTime(rejection.timestamp)}</td>
      <td><code style="background: rgba(255,255,255,0.1); padding: 2px 4px; border-radius: 4px;">${rejection.symbol}</code></td>
      <td>${rejection.reason}</td>
    `;

    if (tbody.firstChild?.textContent?.includes('لا توجد رفضات')) {
      tbody.innerHTML = '';
    }

    tbody.insertBefore(newRow, tbody.firstChild);

    // Keep only last 20 rows
    while (tbody.children.length > 20) {
      tbody.removeChild(tbody.lastChild);
    }
  }

  setupEventListeners() {
    // Trading toggle
    document.getElementById('tradingToggle').addEventListener('change', async (e) => {
      try {
        await fetch('/toggle_trading', { method: 'POST' });
      } catch (error) {
        console.error('خطأ في تغيير حالة التداول:', error);
        e.target.checked = !e.target.checked; // Revert on error
      }
    });

    // Real trading toggle
    document.getElementById('realTradingToggle').addEventListener('change', async (e) => {
      if (e.target.checked && !confirm('هل أنت متأكد من التبديل إلى التداول الحقيقي؟ هذا سيستخدم أموالاً حقيقية.')) {
        e.target.checked = false;
        return;
      }

      try {
        const response = await fetch('/api/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ paper_trading_mode: !e.target.checked })
        });

        if (!response.ok) {
          throw new Error('فشل في تحديث الإعدادات');
        }
      } catch (error) {
        console.error('خطأ في تغيير وضع التداول:', error);
        e.target.checked = !e.target.checked; // Revert on error
        this.showError('فشل في تغيير وضع التداول');
      }
    });

    // Quality slider
    document.getElementById('qualitySlider').addEventListener('input', (e) => {
      document.getElementById('qualityValue').textContent = e.target.value;
      this.debounceQualityUpdate(e.target.value);
    });
  }

  debounceQualityUpdate = this.debounce(async (value) => {
    try {
      await fetch('/api/signal_quality', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ min_quality: parseInt(value) })
      });
    } catch (error) {
      console.error('خطأ في تحديث جودة الإشارة:', error);
    }
  }, 500);

  debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  async closeSignal(signalId) {
    if (!confirm('هل أنت متأكد من إغلاق هذه الصفقة؟')) return;

    try {
      const response = await fetch(`/api/close_trade/${signalId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      const result = await response.json();
      if (result.success) {
        this.showSuccess('تم إغلاق الصفقة بنجاح');
      } else {
        this.showError(result.message || 'فشل في إغلاق الصفقة');
      }
    } catch (error) {
      console.error('خطأ في إغلاق الصفقة:', error);
      this.showError('خطأ في الاتصال بالخادم');
    }
  }

  viewSignalDetails(signalId) {
    const signal = this.signals[signalId];
    if (!signal) return;

    const details = `
      الرمز: ${signal.symbol}
      الاستراتيجية: ${signal.strategy_name}
      سعر الدخول: ${this.formatPrice(signal.entry_price)}
      وقف الخسارة: ${this.formatPrice(signal.stop_loss)}
      الهدف الأول: ${this.formatPrice(signal.target_price_1)}
      الهدف الثاني: ${this.formatPrice(signal.target_price_2)}
      جودة الإشارة: ${signal.signal_details?.quality_score || 'غير محدد'}/100
      تاريخ الإنشاء: ${this.formatTime(signal.timestamp || signal.created_at)}
    `;

    alert(details);
  }

  async closeAllTrades() {
    if (!confirm('هل أنت متأكد من إغلاق جميع الصفقات المفتوحة؟')) return;

    const promises = Object.keys(this.signals).map(signalId =>
      fetch(`/api/close_trade/${signalId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
    );

    try {
      await Promise.all(promises);
      this.showSuccess('تم إغلاق جميع الصفقات');
    } catch (error) {
      console.error('خطأ في إغلاق الصفقات:', error);
      this.showError('فشل في إغلاق بعض الصفقات');
    }
  }

  async refreshSignals() {
    try {
      const response = await fetch('/api/open_signals');
      const data = await response.json();
      this.updateSignals(data.signals || []);
      this.showSuccess('تم تحديث الصفقات');
    } catch (error) {
      console.error('خطأ في تحديث الصفقات:', error);
      this.showError('فشل في تحديث الصفقات');
    }
  }

  formatPrice(price) {
    if (price === null || price === undefined) return 'N/A';
    return parseFloat(price).toLocaleString('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 6
    });
  }

  formatTime(timestamp) {
    if (!timestamp) return 'N/A';
    return new Date(timestamp).toLocaleTimeString('ar-EG', {
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  showSuccess(message) {
    this.showNotification(message, 'success');
  }

  showError(message) {
    this.showNotification(message, 'error');
  }

  showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
      position: fixed; top: 20px; right: 20px; z-index: 10000;
      padding: 16px 20px; border-radius: 8px; color: white; font-weight: 600;
      max-width: 400px; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
      backdrop-filter: blur(10px); animation: slideIn 0.3s ease;
    `;

    const colors = {
      success: 'var(--success)',
      error: 'var(--danger)',
      warning: 'var(--warning)',
      info: 'var(--accent)'
    };

    notification.style.background = `rgba(${type === 'success' ? '0,255,136' : type === 'error' ? '255,85,85' : '0,212,255'}, 0.8)`;
    notification.textContent = message;

    document.body.appendChild(notification);

    // Auto remove after 5 seconds
    setTimeout(() => {
      notification.style.animation = 'slideOut 0.3s ease forwards';
      setTimeout(() => notification.remove(), 300);
    }, 5000);
  }
}

// Global functions for inline handlers
let dashboard;

window.addEventListener('DOMContentLoaded', () => {
  dashboard = new TradingDashboard();
});

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
  @keyframes slideIn {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
  }

  @keyframes slideOut {
    from { transform: translateX(0); opacity: 1; }
    to { transform: translateX(100%); opacity: 0; }
  }
`;
document.head.appendChild(style);
</script>

</body>
</html>
"""

# مسارات Flask
@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/api/dashboard_data')
def dashboard_data():
    try:
        # تحديث الرصيد
        data_manager.update_balance()

        return jsonify({
            "server_time": datetime.now(timezone.utc).isoformat(),
            "trading_enabled": bot_state.trading_enabled,
            "paper_trading_mode": bot_state.paper_trading_mode,
            "usdt_balance": bot_state.usdt_balance,
            "min_signal_quality": trading_config.MIN_SIGNAL_QUALITY,
            "trade_amount_min": trading_config.FIXED_TRADE_AMOUNT_MIN_USDT,
            "trade_amount_max": trading_config.FIXED_TRADE_AMOUNT_MAX_USDT,
            "notifications": bot_state.get_notifications(),
            "rejections": bot_state.get_rejections()
        })
    except Exception as e:
        logger.error(f"❌ [API Error] Failed to generate dashboard data: {e}")
        return jsonify({"error": "Failed to load dashboard data."}), 500

@app.route('/api/open_signals')
def get_open_signals():
    try:
        signals = db_manager.execute_query(
            "SELECT *, NOW() as current_db_time FROM signals WHERE status IN ('open', 'updated') ORDER BY id DESC;",
            fetch=True
        )
        # Add current price to each signal
        signals_with_price = []
        for row in signals:
            signal = dict(row)
            signal['current_price'] = bot_state.get_live_price(signal['symbol']) or signal['entry_price']
            signals_with_price.append(signal)

        return jsonify({"signals": signals_with_price})
    except Exception as e:
        logger.error(f"❌ [API Error] Failed to fetch open signals: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance_metrics')
def get_performance_metrics():
    try:
        # حساب مؤشرات الأداء من قاعدة البيانات
        closed_trades = db_manager.execute_query("""
            SELECT profit_percentage FROM signals
            WHERE status = 'closed' AND closed_at >= NOW() - INTERVAL '30 days'
        """, fetch=True)

        if not closed_trades:
            return jsonify({
                "win_rate": 0,
                "avg_profit": 0,
                "total_trades": 0
            })

        profits = [float(trade['profit_percentage'] or 0) for trade in closed_trades]
        winning_trades = [p for p in profits if p > 0]

        win_rate = (len(winning_trades) / len(profits)) * 100 if profits else 0
        avg_profit = sum(profits) / len(profits) if profits else 0

        return jsonify({
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "total_trades": len(profits)
        })
    except Exception as e:
        logger.error(f"❌ [API Error] Failed to calculate performance metrics: {e}")
        return jsonify({
            "win_rate": 0,
            "avg_profit": 0,
            "total_trades": 0
        })

@app.route('/api/close_trade/<int:signal_id>', methods=['POST'])
def close_trade_manually(signal_id):
    try:
        # البحث عن الصفقة
        signal = db_manager.execute_query(
            "SELECT * FROM signals WHERE id = %s AND status IN ('open', 'updated');",
            (signal_id,),
            fetch=True
        )

        if not signal:
            return jsonify({"success": False, "message": "Trade not found or already closed."}), 404

        signal = signal[0]
        symbol = signal['symbol']
        current_price = bot_state.get_live_price(symbol) or signal['entry_price']
        profit = ((current_price - signal['entry_price']) / signal['entry_price']) * 100

        # تحديث الصفقة
        db_manager.execute_query("""
            UPDATE signals SET status = 'closed', closing_price = %s, closed_at = NOW(),
            profit_percentage = %s, closing_reason = 'manual_close' WHERE id = %s;
        """, (current_price, profit, signal_id))

        # إزالة من الذاكرة المؤقتة
        bot_state.remove_open_signal(symbol)

        # إشعار
        notification_manager.log_and_notify(
            "info", f"Trade {symbol} closed manually with {profit:.2f}% profit", "TRADE_CLOSED"
        )

        return jsonify({"success": True, "message": "Trade closed successfully."})

    except Exception as e:
        logger.error(f"❌ Error closing trade manually: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/toggle_trading', methods=['POST'])
def toggle_trading():
    bot_state.trading_enabled = not bot_state.trading_enabled
    status_msg = "enabled" if bot_state.trading_enabled else "disabled"
    notification_manager.log_and_notify("info", f"Trading has been {status_msg}.", "TRADING_STATUS")
    return jsonify({"status": "success", "trading_enabled": bot_state.trading_enabled})

@app.route('/api/signal_quality', methods=['POST'])
def update_signal_quality():
    try:
        data = request.json
        new_quality = int(data['min_quality'])
        if 30 <= new_quality <= 90:
            trading_config.MIN_SIGNAL_QUALITY = new_quality
            return jsonify({"success": True})
        return jsonify({"success": False, "message": "Invalid quality value"}), 400
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/settings', methods=['POST'])
def update_settings():
    try:
        data = request.json

        if 'paper_trading_mode' in data:
            bot_state.paper_trading_mode = bool(data['paper_trading_mode'])

        if 'FIXED_TRADE_AMOUNT_MIN_USDT' in data:
            trading_config.FIXED_TRADE_AMOUNT_MIN_USDT = float(data['FIXED_TRADE_AMOUNT_MIN_USDT'])

        if 'FIXED_TRADE_AMOUNT_MAX_USDT' in data:
            trading_config.FIXED_TRADE_AMOUNT_MAX_USDT = float(data['FIXED_TRADE_AMOUNT_MAX_USDT'])

        return jsonify({"success": True, "message": "Settings updated successfully"})

    except Exception as e:
        logger.error(f"❌ Error updating settings: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# WebSocket endpoint
@sock.route('/ws')
def websocket(ws):
    with ws_clients_lock:
        ws_clients.append(ws)

    try:
        while True:
            # إبقاء الاتصال مفتوح
            # This will block until a message is received or the connection is closed.
            data = ws.receive(timeout=60) # Add a timeout
            if data is None: # Timeout occurred, send a ping
                 ws.send(json.dumps({"type": "ping"}))
    except Exception:
        pass # Handle disconnection
    finally:
        with ws_clients_lock:
            if ws in ws_clients:
                ws_clients.remove(ws)

# ==============================================================================
# SECTION 7: النظام الرئيسي للتشغيل
# ==============================================================================

class TradingSystem:
    """النظام الرئيسي للتداول"""

    def __init__(self):
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=4)

    def start(self):
        """بدء النظام"""
        logger.info("🚀 Starting Enhanced Trading Bot System V37...")

        try:
            # تهيئة المكونات
            self._initialize_components()

            # تحميل البيانات الأولية
            self._load_initial_data()

            # بدء المهام الخلفية
            self._start_background_tasks()

            # بدء خادم Flask
            self._start_flask_server()

        except Exception as e:
            logger.critical(f"❌ Failed to start trading system: {e}")
            raise

    def _initialize_components(self):
        """تهيئة المكونات الأساسية"""
        # تهيئة قاعدة البيانات (تمت بالفعل)
        logger.info("✅ Database manager initialized")

        # تهيئة Redis (تمت بالفعل)
        logger.info("✅ Redis manager initialized")

        # تهيئة مدير البيانات (تمت بالفعل)
        logger.info("✅ Data manager initialized")

        # الحصول على معلومات البورصة
        global exchange_info_map, validated_symbols_to_scan
        exchange_info_map = data_manager.get_exchange_info()
        validated_symbols_to_scan = data_manager.get_validated_symbols()

        if not validated_symbols_to_scan:
            raise Exception("No valid symbols found for trading")

        logger.info(f"✅ Found {len(validated_symbols_to_scan)} valid symbols")

    def _load_initial_data(self):
        """تحميل البيانات الأولية"""
        # تحميل الصفقات المفتوحة
        try:
            open_trades = db_manager.execute_query(
                "SELECT * FROM signals WHERE status IN ('open', 'updated');",
                fetch=True
            )

            for trade in open_trades:
                trade_dict = dict(trade)
                bot_state.set_open_signal(trade_dict['symbol'], trade_dict)

            logger.info(f"✅ Loaded {len(open_trades)} open trades")
        except Exception as e:
            logger.error(f"❌ Error loading open trades: {e}")

        # تحديث الرصيد
        data_manager.update_balance()
        logger.info(f"✅ Balance updated: {bot_state.usdt_balance:.2f} USDT")

    def _start_background_tasks(self):
        """بدء المهام الخلفية"""
        self.running = True

        # مولد الإشارات
        self.executor.submit(self._signal_generation_loop)

        # إدارة الصفقات
        self.executor.submit(self._trade_management_loop)

        # تحديث الأسعار
        self.executor.submit(self._price_update_loop)

        # تحديث الرصيد
        self.executor.submit(self._balance_update_loop)

        logger.info("✅ Background tasks started")

    def _signal_generation_loop(self):
        """حلقة توليد الإشارات"""
        last_signal_check = 0

        while self.running:
            try:
                now = time.time()

                # توليد الإشارات كل 5 دقائق
                if now - last_signal_check >= 300:
                    signal_generator.generate_signals()
                    last_signal_check = now

                time.sleep(60)  # فحص كل دقيقة

            except Exception as e:
                logger.error(f"❌ Error in signal generation loop: {e}")
                time.sleep(60)

    def _trade_management_loop(self):
        """حلقة إدارة الصفقات"""
        while self.running:
            try:
                # إدارة الصفقات المفتوحة
                TradeManager.manage_open_trades()

                # تحديث وقف الخسارة المتحرك
                TradeManager.update_trailing_stops()

                time.sleep(10)  # فحص كل 10 ثواني

            except Exception as e:
                logger.error(f"❌ Error in trade management loop: {e}")
                time.sleep(30)

    def _price_update_loop(self):
        """حلقة تحديث الأسعار"""
        while self.running:
            try:
                # تحديث أسعار الرموز المفتوحة
                open_signals = bot_state.get_open_signals()
                symbols = list(set([signal['symbol'] for signal in open_signals.values()]))

                if symbols:
                    # جلب الأسعار الحالية
                    tickers = data_manager.client.get_all_tickers()
                    price_updates = {}

                    for ticker in tickers:
                        if ticker['symbol'] in symbols:
                            price_updates[ticker['symbol']] = float(ticker['price'])
                            bot_state.update_live_price(ticker['symbol'], float(ticker['price']))

                    # إرسال التحديثات عبر WebSocket
                    if price_updates:
                        notification_manager._broadcast_ws({
                            "type": "price_update",
                            "payload": price_updates
                        })

                time.sleep(5)  # تحديث كل 5 ثواني

            except Exception as e:
                logger.error(f"❌ Error in price update loop: {e}")
                time.sleep(30)

    def _balance_update_loop(self):
        """حلقة تحديث الرصيد"""
        while self.running:
            try:
                data_manager.update_balance()
                time.sleep(300)  # تحديث كل 5 دقائق

            except Exception as e:
                logger.error(f"❌ Error in balance update loop: {e}")
                time.sleep(300)

    def _start_flask_server(self):
        """بدء خادم Flask"""
        # IMPROVEMENT: Use environment variable for port
        port = int(os.environ.get("PORT", 5000))
        logger.info(f"🌐 Starting Flask web server on http://0.0.0.0:{port}")

        try:
            # Use a production-ready WSGI server like waitress instead of app.run for better performance
            from waitress import serve
            serve(app, host='0.0.0.0', port=port, threads=8)
        except ImportError:
            logger.warning("waitress not found, falling back to Flask's development server. Install with: pip install waitress")
            app.run(
                host='0.0.0.0',
                port=port,
                threaded=True,
                debug=False  # تعطيل debug في الإنتاج
            )
        except Exception as e:
            logger.critical(f"❌ Failed to start Flask server: {e}")
            raise

    def stop(self):
        """إيقاف النظام"""
        logger.info("🛑 Stopping trading system...")
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("✅ Trading system stopped")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """الدالة الرئيسية"""
    trading_system = TradingSystem()

    try:
        trading_system.start()
    except KeyboardInterrupt:
        logger.info("🛑 Received shutdown signal")
        trading_system.stop()
    except Exception as e:
        logger.critical(f"❌ Critical error in main: {e}", exc_info=True)
        trading_system.stop()
        exit(1) # Exit with a non-zero code to indicate an error

if __name__ == "__main__":
    main()
