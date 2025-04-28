
import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql, OperationalError, InterfaceError # لاستخدام استعلامات آمنة وأخطاء محددة
from psycopg2.extras import RealDictCursor # للحصول على النتائج كقواميس
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException, BinanceRequestException # أخطاء Binance المحددة
from flask import Flask, request, Response
from threading import Thread
from datetime import datetime, timedelta
from decouple import config
from typing import List, Dict, Optional, Tuple, Any, Union # لإضافة Type Hinting

# ---------------------- إعداد التسجيل ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # إضافة اسم المسجل
    handlers=[
        logging.FileHandler('crypto_bot_elliott_fib.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
# استخدام اسم محدد للمسجل بدلاً من الجذر
logger = logging.getLogger('CryptoBot')

# ---------------------- تحميل المتغيرات البيئية ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    TELEGRAM_TOKEN: str = config('TELEGRAM_BOT_TOKEN')
    CHAT_ID: str = config('TELEGRAM_CHAT_ID')
    DB_URL: str = config('DATABASE_URL')
    # استخدام قيمة افتراضية None إذا لم يكن المتغير موجودًا
    WEBHOOK_URL: Optional[str] = config('WEBHOOK_URL', default=None)
except Exception as e:
     logger.critical(f"❌ فشل في تحميل المتغيرات البيئية الأساسية: {e}")
     exit(1) # استخدام رمز خروج غير صفري للإشارة إلى خطأ

logger.info(f"مفتاح Binance API: {'موجود' if API_KEY else 'غير موجود'}")
logger.info(f"توكن تليجرام: {TELEGRAM_TOKEN[:10]}...{'*' * (len(TELEGRAM_TOKEN)-10)}")
logger.info(f"معرف دردشة تليجرام: {CHAT_ID}")
logger.info(f"رابط قاعدة البيانات: {'موجود' if DB_URL else 'غير موجود'}")
logger.info(f"عنوان Webhook: {WEBHOOK_URL if WEBHOOK_URL else 'غير محدد'}")

# ---------------------- إعداد الثوابت والمتغيرات العامة ----------------------
TRADE_VALUE: float = 10.0         # قيمة الصفقة الافتراضية بالدولار
MAX_OPEN_TRADES: int = 4          # الحد الأقصى للصفقات المفتوحة في نفس الوقت
SIGNAL_GENERATION_TIMEFRAME: str = '30m' # الإطار الزمني لتوليد الإشارة
SIGNAL_GENERATION_LOOKBACK_DAYS: int = 5 # عدد الأيام للبيانات التاريخية لتوليد الإشارة
SIGNAL_TRACKING_TIMEFRAME: str = '30m' # الإطار الزمني لتتبع الإشارة وتحديث وقف الخسارة
SIGNAL_TRACKING_LOOKBACK_DAYS: int = 5   # عدد الأيام للبيانات التاريخية لتتبع الإشارة

# =============================================================================
# --- تعديل قيم المؤشرات (مثال) ---
# يمكنك تعديل هذه القيم لتناسب استراتيجيتك بشكل أفضل
# =============================================================================
RSI_PERIOD: int = 14          # فترة RSI (القيمة الأصلية: 14)
RSI_OVERSOLD: int = 35        # حد التشبع البيعي (القيمة الأصلية: 30) - رفع الحد قليلاً
RSI_OVERBOUGHT: int = 65      # حد التشبع الشرائي (القيمة الأصلية: 70) - خفض الحد قليلاً
EMA_PERIOD: int = 26          # فترة EMA للترند (القيمة الأصلية: 21) - زيادة الفترة لتقليل الحساسية
SWING_ORDER: int = 5          # ترتيب تحديد القمم والقيعان
FIB_LEVELS_TO_CHECK: List[float] = [0.382, 0.5, 0.618]
FIB_TOLERANCE: float = 0.007
LOOKBACK_FOR_SWINGS: int = 100
ENTRY_ATR_PERIOD: int = 14     # فترة ATR للدخول
ENTRY_ATR_MULTIPLIER: float = 1.5 # مضاعف ATR للهدف/الوقف الأولي (القيمة الأصلية: 1.2) - زيادة المضاعف
BOLLINGER_WINDOW: int = 20     # فترة Bollinger Bands
BOLLINGER_STD_DEV: int = 2       # الانحراف المعياري لـ Bollinger Bands
MACD_FAST: int = 12            # فترة MACD السريعة
MACD_SLOW: int = 26            # فترة MACD البطيئة
MACD_SIGNAL: int = 9             # فترة خط إشارة MACD
ADX_PERIOD: int = 14            # فترة ADX
SUPERTREND_PERIOD: int = 10     # فترة SuperTrend
SUPERTREND_MULTIPLIER: float = 3.0 # مضاعف SuperTrend

# وقف الخسارة المتحرك
TRAILING_STOP_ACTIVATION_PROFIT_PCT: float = 0.02 د # نسبة الربح لتفعيل الوقف المتحرك (1.5%)
TRAILING_STOP_ATR_MULTIPLIER: float = 2.5        # مضاعف ATR للوقف المتحرك (القيمة الأصلية: 2.5) - تقليل المضاعف ليكون أضيق
TRAILING_STOP_MOVE_INCREMENT_PCT: float = 0.002  # نسبة الزيادة في السعر لتحريك الوقف المتحرك (0.2%)

# شروط إضافية للإشارة
MIN_PROFIT_MARGIN_PCT: float = 1.5 # الحد الأدنى لنسبة الربح المستهدف المئوية
MIN_VOLUME_15M_USDT: float = 100000.0 # الحد الأدنى للسيولة في آخر 15 دقيقة بالدولار
# =============================================================================
# --- نهاية تعديل قيم المؤشرات ---
# =============================================================================

# متغيرات عالمية (سيتم تهيئتها لاحقًا)
conn: Optional[psycopg2.extensions.connection] = None
cur: Optional[psycopg2.extensions.cursor] = None
client: Optional[Client] = None
ticker_data: Dict[str, float] = {} # قاموس لتخزين أحدث أسعار الإغلاق للرموز

# ---------------------- إعداد عميل Binance ----------------------
try:
    logger.info("ℹ️ [Binance] تهيئة عميل Binance...")
    client = Client(API_KEY, API_SECRET)
    client.ping() # التحقق من الاتصال وصحة المفاتيح
    server_time = client.get_server_time()
    logger.info(f"✅ [Binance] تم تهيئة عميل Binance. وقت الخادم: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
except BinanceRequestException as req_err:
     logger.critical(f"❌ [Binance] خطأ في طلب Binance (قد يكون مشكلة شبكة أو طلب): {req_err}")
     exit(1)
except BinanceAPIException as api_err:
     logger.critical(f"❌ [Binance] خطأ من Binance API (قد تكون المفاتيح غير صالحة أو مشكلة في الخادم): {api_err}")
     exit(1)
except Exception as e:
    logger.critical(f"❌ [Binance] فشل غير متوقع في تهيئة عميل Binance: {e}")
    exit(1)

# ---------------------- دوال المؤشرات الإضافية ----------------------
def get_fear_greed_index() -> str:
    """يجلب مؤشر الخوف والطمع من alternative.me ويترجم التصنيف إلى العربية."""
    classification_translation_ar = {
        "Extreme Fear": "خوف شديد", "Fear": "خوف", "Neutral": "محايد",
        "Greed": "جشع", "Extreme Greed": "جشع شديد",
    }
    url = "https://api.alternative.me/fng/"
    logger.debug(f"ℹ️ [Indicators] جلب مؤشر الخوف والطمع من {url}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        value = int(data["data"][0]["value"])
        classification_en = data["data"][0]["value_classification"]
        classification_ar = classification_translation_ar.get(classification_en, classification_en)
        logger.debug(f"✅ [Indicators] مؤشر الخوف والطمع: {value} ({classification_ar})")
        return f"{value} ({classification_ar})"
    except requests.exceptions.RequestException as e:
         logger.error(f"❌ [Indicators] خطأ في الشبكة عند جلب مؤشر الخوف والطمع: {e}")
         return "N/A (خطأ في الشبكة)"
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"❌ [Indicators] خطأ في تنسيق بيانات مؤشر الخوف والطمع: {e}")
        return "N/A (خطأ في البيانات)"
    except Exception as e:
        logger.error(f"❌ [Indicators] خطأ غير متوقع في جلب مؤشر الخوف والطمع: {e}", exc_info=True)
        return "N/A (خطأ غير معروف)"

def fetch_historical_data(symbol: str, interval: str = SIGNAL_GENERATION_TIMEFRAME, days: int = SIGNAL_GENERATION_LOOKBACK_DAYS) -> Optional[pd.DataFrame]:
    """جلب البيانات التاريخية للشموع من Binance."""
    if not client:
        logger.error("❌ [Data] عميل Binance غير مهيأ لجلب البيانات.")
        return None
    try:
        start_dt = datetime.utcnow() - timedelta(days=days + 1) # إضافة يوم إضافي كاحتياط
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        logger.debug(f"ℹ️ [Data] جلب بيانات {interval} للزوج {symbol} منذ {start_str} (حد 1000 شمعة)...")

        klines = client.get_historical_klines(symbol, interval, start_str, limit=1000)

        if not klines:
            logger.warning(f"⚠️ [Data] لا توجد بيانات تاريخية ({interval}) للزوج {symbol} للفترة المطلوبة.")
            return None

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])

        # تحديد الأعمدة الرقمية الأساسية
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce') # coerce تحول القيم غير الصالحة إلى NaN

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # اختيار الأعمدة المطلوبة فقط
        df = df[numeric_cols]

        initial_len = len(df)
        df.dropna(subset=numeric_cols, inplace=True) # حذف الصفوف التي تحتوي على NaN في الأعمدة الأساسية

        if len(df) < initial_len:
            logger.debug(f"ℹ️ [Data] {symbol}: تم حذف {initial_len - len(df)} صف بسبب NaN في بيانات OHLCV.")

        if df.empty:
            logger.warning(f"⚠️ [Data] DataFrame للزوج {symbol} فارغ بعد إزالة NaN الأساسية.")
            return None

        logger.debug(f"✅ [Data] تم جلب ومعالجة {len(df)} شمعة تاريخية ({interval}) للزوج {symbol}.")
        return df

    except BinanceAPIException as api_err:
         logger.error(f"❌ [Data] خطأ API من Binance عند جلب بيانات {symbol}: {api_err}")
         return None
    except BinanceRequestException as req_err:
         logger.error(f"❌ [Data] خطأ طلب أو شبكة عند جلب بيانات {symbol}: {req_err}")
         return None
    except Exception as e:
        logger.error(f"❌ [Data] خطأ غير متوقع في جلب البيانات التاريخية للزوج {symbol}: {e}", exc_info=True)
        return None


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """حساب المتوسط المتحرك الأسي (EMA)."""
    if series is None or series.isnull().all() or len(series) < span:
        # إرجاع سلسلة فارغة بنفس الفهرس إذا أمكن للحفاظ على التوافق
        return pd.Series(index=series.index if series is not None else None, dtype=float)
    return series.ewm(span=span, adjust=False).mean()


def get_btc_trend_4h() -> str:
    """يحسب ترند البيتكوين على فريم 4 ساعات باستخدام EMA20 وEMA50."""
    # ملاحظة: هذه الدالة لا تزال تستخدم EMA20 و EMA50 داخليًا، قد ترغب في توحيدها مع EMA_PERIOD العام إذا أردت
    logger.debug("ℹ️ [Indicators] حساب ترند البيتكوين 4 ساعات...")
    try:
        df = fetch_historical_data("BTCUSDT", interval=Client.KLINE_INTERVAL_4HOUR, days=10) # طلب أيام أكثر قليلاً
        if df is None or df.empty or len(df) < 50 + 1: # التأكد من وجود بيانات كافية لـ EMA50
            logger.warning("⚠️ [Indicators] بيانات BTC/USDT 4H غير كافية لحساب الترند.")
            return "N/A (بيانات غير كافية)"

        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df.dropna(subset=['close'], inplace=True)
        if len(df) < 50:
             logger.warning("⚠️ [Indicators] بيانات BTC/USDT 4H غير كافية بعد إزالة NaN.")
             return "N/A (بيانات غير كافية)"

        ema20 = calculate_ema(df['close'], 20).iloc[-1] # لا يزال يستخدم 20 هنا
        ema50 = calculate_ema(df['close'], 50).iloc[-1] # لا يزال يستخدم 50 هنا
        current_close = df['close'].iloc[-1]

        if pd.isna(ema20) or pd.isna(ema50) or pd.isna(current_close):
            logger.warning("⚠️ [Indicators] قيم EMA أو السعر الحالي لـ BTC هي NaN.")
            return "N/A (خطأ حسابي)"

        diff_ema20_pct = abs(current_close - ema20) / current_close if current_close > 0 else 0

        if current_close > ema20 > ema50:
            trend = "صعود 📈"
        elif current_close < ema20 < ema50:
            trend = "هبوط 📉"
        elif diff_ema20_pct < 0.005: # أقل من 0.5% فرق، يعتبر استقرار
            trend = "استقرار 🔄"
        else: # تقاطع أو تباعد غير واضح
            trend = "تذبذب 🔀"

        logger.debug(f"✅ [Indicators] ترند البيتكوين 4H: {trend}")
        return trend
    except Exception as e:
        logger.error(f"❌ [Indicators] خطأ في حساب ترند البيتكوين على أربع ساعات: {e}", exc_info=True)
        return "N/A (خطأ)"

# ---------------------- إعداد الاتصال بقاعدة البيانات ----------------------
def init_db(retries: int = 5, delay: int = 5) -> None:
    """تهيئة الاتصال بقاعدة البيانات وإنشاء الجداول إذا لم تكن موجودة."""
    global conn, cur
    logger.info("[DB] بدء تهيئة قاعدة البيانات...")
    for attempt in range(retries):
        try:
            logger.info(f"[DB] محاولة الاتصال بقاعدة البيانات (محاولة {attempt + 1}/{retries})...")
            conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
            conn.autocommit = False # التحكم اليدوي بالـ commit/rollback
            cur = conn.cursor()
            logger.info("✅ [DB] تم الاتصال بقاعدة البيانات بنجاح.")

            # --- إنشاء أو تحديث جدول signals ---
            logger.info("[DB] التحقق/إنشاء جدول 'signals'...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    initial_target DOUBLE PRECISION NOT NULL,
                    initial_stop_loss DOUBLE PRECISION NOT NULL,
                    current_target DOUBLE PRECISION NOT NULL,
                    current_stop_loss DOUBLE PRECISION NOT NULL,
                    r2_score DOUBLE PRECISION, -- يمثل الآن درجة الإشارة الموزونة
                    volume_15m DOUBLE PRECISION,
                    achieved_target BOOLEAN DEFAULT FALSE,
                    hit_stop_loss BOOLEAN DEFAULT FALSE,
                    closing_price DOUBLE PRECISION,
                    closed_at TIMESTAMP,
                    sent_at TIMESTAMP DEFAULT NOW(),
                    profit_percentage DOUBLE PRECISION,
                    profitable_stop_loss BOOLEAN DEFAULT FALSE,
                    is_trailing_active BOOLEAN DEFAULT FALSE,
                    strategy_name TEXT,
                    signal_details JSONB,
                    last_trailing_update_price DOUBLE PRECISION
                );""")
            conn.commit()
            logger.info("✅ [DB] جدول 'signals' موجود أو تم إنشاؤه.")

            # --- التحقق وإضافة الأعمدة الناقصة (إذا لزم الأمر) ---
            required_columns = {
                "symbol", "entry_price", "initial_target", "initial_stop_loss",
                "current_target", "current_stop_loss", "r2_score", "volume_15m",
                "achieved_target", "hit_stop_loss", "closing_price", "closed_at",
                "sent_at", "profit_percentage", "profitable_stop_loss",
                "is_trailing_active", "strategy_name", "signal_details",
                "last_trailing_update_price"
            }
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'signals' AND table_schema = 'public';")
            existing_columns = {row['column_name'] for row in cur.fetchall()}
            missing_columns = required_columns - existing_columns

            if missing_columns:
                logger.warning(f"⚠️ [DB] الأعمدة التالية مفقودة في جدول 'signals': {missing_columns}. محاولة إضافتها...")
                # (الكود الأصلي لإضافة الأعمدة كان جيدًا، يمكن الاحتفاظ به أو تحسينه هنا إذا لزم الأمر)
                # ... (يمكن إضافة كود ALTER TABLE هنا إذا كنت تتوقع تغييرات مستقبلية) ...
                logger.warning("⚠️ [DB] لم يتم تنفيذ إضافة الأعمدة المفقودة تلقائيًا في هذا الإصدار المحسن. يرجى التحقق يدويًا إذا لزم الأمر.")
            else:
                logger.info("✅ [DB] جميع الأعمدة المطلوبة موجودة في جدول 'signals'.")

            # --- إنشاء جدول market_dominance (إذا لم يكن موجودًا) ---
            logger.info("[DB] التحقق/إنشاء جدول 'market_dominance'...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS market_dominance (
                    id SERIAL PRIMARY KEY,
                    recorded_at TIMESTAMP DEFAULT NOW(),
                    btc_dominance DOUBLE PRECISION,
                    eth_dominance DOUBLE PRECISION
                );
            """)
            conn.commit()
            logger.info("✅ [DB] جدول 'market_dominance' موجود أو تم إنشاؤه.")

            logger.info("✅ [DB] تهيئة قاعدة البيانات تمت بنجاح.")
            return # نجح الاتصال والتهيئة

        except OperationalError as op_err:
            logger.error(f"❌ [DB] خطأ تشغيلي في الاتصال (المحاولة {attempt + 1}): {op_err}")
            if conn: conn.rollback()
            if attempt == retries - 1:
                 logger.critical("❌ [DB] فشلت جميع محاولات الاتصال بقاعدة البيانات.")
                 raise op_err # إعادة رفع الخطأ بعد فشل كل المحاولات
            time.sleep(delay)
        except Exception as e:
            logger.critical(f"❌ [DB] فشل غير متوقع في تهيئة قاعدة البيانات (المحاولة {attempt + 1}): {e}", exc_info=True)
            if conn: conn.rollback()
            if attempt == retries - 1:
                 logger.critical("❌ [DB] فشلت جميع محاولات الاتصال بقاعدة البيانات.")
                 raise e
            time.sleep(delay)

    # إذا وصل الكود إلى هنا، فقد فشلت كل المحاولات
    logger.critical("❌ [DB] فشل الاتصال بقاعدة البيانات بعد عدة محاولات.")
    exit(1)


def check_db_connection() -> bool:
    """التحقق من حالة الاتصال بقاعدة البيانات وإعادة التهيئة إذا لزم الأمر."""
    global conn, cur
    try:
        if conn is None or conn.closed != 0:
            logger.warning("⚠️ [DB] الاتصال مغلق أو غير موجود. إعادة التهيئة...")
            init_db() # محاولة إعادة الاتصال والتهيئة
            return True # نفترض نجاح التهيئة (init_db سترفع خطأ إذا فشلت)
        else:
             # التحقق من أن الاتصال لا يزال يعمل بإرسال استعلام بسيط
             with conn.cursor() as check_cur: # استخدام cursor مؤقت
                  check_cur.execute("SELECT 1;")
                  check_cur.fetchone()
             # logger.debug("[DB] الاتصال نشط.") # إلغاء التعليق للتحقق المتكرر
             return True
    except (OperationalError, InterfaceError) as e:
        logger.error(f"❌ [DB] فقدان الاتصال بقاعدة البيانات ({e}). إعادة التهيئة...")
        try:
             init_db()
             return True
        except Exception as recon_err:
            logger.error(f"❌ [DB] فشلت محاولة إعادة الاتصال بعد فقدان الاتصال: {recon_err}")
            return False
    except Exception as e:
        logger.error(f"❌ [DB] خطأ غير متوقع أثناء التحقق من الاتصال: {e}", exc_info=True)
        # محاولة إعادة الاتصال كإجراء وقائي
        try:
            init_db()
            return True
        except Exception as recon_err:
             logger.error(f"❌ [DB] فشلت محاولة إعادة الاتصال بعد خطأ غير متوقع: {recon_err}")
             return False


def convert_np_values(obj: Any) -> Any:
    """تحويل أنواع بيانات NumPy إلى أنواع Python الأصلية للتوافق مع JSON و DB."""
    if isinstance(obj, dict):
        return {k: convert_np_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_values(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int_)): # np.int_ قديم لكن لا يزال يعمل
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)): # تم استخدام np.float64 مباشرة
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif pd.isna(obj): # معالجة NaT من Pandas أيضًا
        return None
    else:
        return obj

# ---------------------- قراءة قائمة الأزواج والتحقق منها ----------------------
def get_crypto_symbols(filename: str = 'crypto_list.txt') -> List[str]:
    """
    قراءة قائمة رموز العملات من ملف نصي، ثم التحقق من صلاحيتها
    وكونها أزواج USDT متاحة للتداول على Binance Spot.
    """
    raw_symbols: List[str] = []
    logger.info(f"ℹ️ [Data] قراءة قائمة الرموز من الملف '{filename}'...")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)

        if not os.path.exists(file_path):
            file_path = os.path.abspath(filename) # جرب المسار الحالي إذا لم يوجد بجانب السكربت
            if not os.path.exists(file_path):
                 logger.error(f"❌ [Data] الملف '{filename}' غير موجود في مجلد السكربت أو المجلد الحالي.")
                 return [] # إرجاع قائمة فارغة إذا لم يتم العثور على الملف
            else:
                 logger.warning(f"⚠️ [Data] الملف '{filename}' غير موجود في مجلد السكربت. استخدام الملف في المجلد الحالي: '{file_path}'")

        with open(file_path, 'r', encoding='utf-8') as f:
            # تنظيف وتنسيق الرموز: إزالة الفراغات، تحويل لأحرف كبيرة، التأكد من انتهاء بـ USDT
            raw_symbols = [f"{line.strip().upper().replace('USDT', '')}USDT"
                           for line in f if line.strip() and not line.startswith('#')]
        raw_symbols = sorted(list(set(raw_symbols))) # إزالة التكرارات والترتيب
        logger.info(f"ℹ️ [Data] تم قراءة {len(raw_symbols)} رمز مبدئي من '{file_path}'.")

    except FileNotFoundError:
         logger.error(f"❌ [Data] الملف '{filename}' غير موجود.")
         return []
    except Exception as e:
        logger.error(f"❌ [Data] خطأ في قراءة الملف '{filename}': {e}", exc_info=True)
        return [] # إرجاع قائمة فارغة في حالة حدوث خطأ

    if not raw_symbols:
         logger.warning("⚠️ [Data] القائمة الأولية للرموز فارغة.")
         return []

    # --- التحقق من الرموز مقابل Binance API ---
    if not client:
        logger.error("❌ [Data Validation] عميل Binance غير مهيأ. لا يمكن التحقق من الرموز.")
        return raw_symbols # إرجاع القائمة غير المفلترة إذا لم يكن العميل جاهزًا

    try:
        logger.info("ℹ️ [Data Validation] التحقق من صلاحية الرموز وحالة التداول من Binance API...")
        exchange_info = client.get_exchange_info()
        # بناء مجموعة (set) برموز USDT الصالحة للتداول الفوري لتسريع البحث
        valid_trading_usdt_symbols = {
            s['symbol'] for s in exchange_info['symbols']
            if s.get('quoteAsset') == 'USDT' and    # التأكد من أن العملة المقابلة هي USDT
               s.get('status') == 'TRADING' and         # التأكد من أن الحالة هي TRADING
               s.get('isSpotTradingAllowed') is True    # التأكد من أنه مسموح بالتداول الفوري
        }
        logger.info(f"ℹ️ [Data Validation] تم العثور على {len(valid_trading_usdt_symbols)} زوج USDT صالح للتداول الفوري على Binance.")

        # فلترة القائمة المقروءة من الملف بناءً على القائمة الصالحة من Binance
        validated_symbols = [symbol for symbol in raw_symbols if symbol in valid_trading_usdt_symbols]

        removed_count = len(raw_symbols) - len(validated_symbols)
        if removed_count > 0:
            removed_symbols = set(raw_symbols) - set(validated_symbols)
            logger.warning(f"⚠️ [Data Validation] تم إزالة {removed_count} رمز غير صالح أو غير متاح للتداول الفوري USDT من القائمة: {', '.join(removed_symbols)}")

        logger.info(f"✅ [Data Validation] تم التحقق من الرموز. سيتم استخدام {len(validated_symbols)} رمز صالح.")
        return validated_symbols

    except (BinanceAPIException, BinanceRequestException) as binance_err:
         logger.error(f"❌ [Data Validation] خطأ من Binance API أو الشبكة عند التحقق من الرموز: {binance_err}")
         logger.warning("⚠️ [Data Validation] سيتم استخدام القائمة الأولية من الملف بدون تحقق Binance.")
         return raw_symbols # إرجاع القائمة غير المفلترة في حالة خطأ API
    except Exception as api_err:
         logger.error(f"❌ [Data Validation] خطأ غير متوقع أثناء التحقق من رموز Binance: {api_err}", exc_info=True)
         logger.warning("⚠️ [Data Validation] سيتم استخدام القائمة الأولية من الملف بدون تحقق Binance.")
         return raw_symbols # إرجاع القائمة غير المفلترة في حالة خطأ API


# ---------------------- إدارة WebSocket لأسعار Ticker ----------------------
def handle_ticker_message(msg: Union[List[Dict[str, Any]], Dict[str, Any]]) -> None:
    """معالجة رسائل WebSocket الواردة لأسعار mini-ticker."""
    global ticker_data
    try:
        if isinstance(msg, list):
            for ticker_item in msg:
                symbol = ticker_item.get('s')
                price_str = ticker_item.get('c') # سعر الإغلاق الأخير كـ string
                if symbol and 'USDT' in symbol and price_str:
                    try:
                        ticker_data[symbol] = float(price_str)
                    except ValueError:
                         logger.warning(f"⚠️ [WS] قيمة سعر غير صالحة للرمز {symbol}: '{price_str}'")
        elif isinstance(msg, dict):
             if msg.get('e') == 'error':
                 logger.error(f"❌ [WS] رسالة خطأ من WebSocket: {msg.get('m', 'لا يوجد تفاصيل خطأ')}")
             elif msg.get('stream') and msg.get('data'): # Handle combined streams format
                 for ticker_item in msg.get('data', []):
                    symbol = ticker_item.get('s')
                    price_str = ticker_item.get('c')
                    if symbol and 'USDT' in symbol and price_str:
                        try:
                            ticker_data[symbol] = float(price_str)
                        except ValueError:
                             logger.warning(f"⚠️ [WS] قيمة سعر غير صالحة للرمز {symbol} في combined stream: '{price_str}'")
        else:
             logger.warning(f"⚠️ [WS] تم استلام رسالة WebSocket بتنسيق غير متوقع: {type(msg)}")

    except Exception as e:
        logger.error(f"❌ [WS] خطأ في معالجة رسالة ticker: {e}", exc_info=True)


def run_ticker_socket_manager() -> None:
    """تشغيل وإدارة اتصال WebSocket لـ mini-ticker."""
    while True:
        try:
            logger.info("ℹ️ [WS] بدء تشغيل WebSocket Manager لأسعار Ticker...")
            twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
            twm.start() # بدء المدير

            # استخدام start_miniticker_socket يغطي جميع الرموز وهو مناسب هنا
            stream_name = twm.start_miniticker_socket(callback=handle_ticker_message)
            logger.info(f"✅ [WS] تم بدء WebSocket stream: {stream_name}")

            twm.join() # الانتظار حتى يتوقف المدير (عادة بسبب خطأ أو إيقاف)
            logger.warning("⚠️ [WS] مدير WebSocket توقف. إعادة التشغيل...")

        except Exception as e:
            logger.error(f"❌ [WS] خطأ فادح في WebSocket Manager: {e}. إعادة التشغيل خلال 15 ثانية...", exc_info=True)

        # الانتظار قبل إعادة المحاولة لتجنب استهلاك الموارد أو حظر الـ IP
        time.sleep(15)

# ---------------------- دوال المؤشرات الفنية ----------------------

def calculate_rsi_indicator(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """حساب مؤشر القوة النسبية (RSI)."""
    df = df.copy()
    if 'close' not in df.columns or df['close'].isnull().all():
        logger.warning("⚠️ [Indicator RSI] عمود 'close' مفقود أو فارغ.")
        df['rsi'] = np.nan
        return df
    if len(df) < period:
        logger.warning(f"⚠️ [Indicator RSI] بيانات غير كافية ({len(df)} < {period}) لحساب RSI.")
        df['rsi'] = np.nan
        return df

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # استخدام ewm لحساب المتوسط المتحرك الأسي للمكاسب والخسائر
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    # حساب RS وتجنب القسمة على صفر
    rs = avg_gain / avg_loss.replace(0, np.nan) # استبدال الصفر بـ NaN لتجنب القسمة عليه

    # حساب RSI
    rsi_series = 100 - (100 / (1 + rs))

    # ملء القيم NaN الأولية (الناتجة عن diff أو avg_loss=0) بالقيمة 50 (محايد)
    # واستخدام forward fill لسد الفجوات إن وجدت (نادر الحدوث مع adjust=False)
    df['rsi'] = rsi_series.ffill().fillna(50)

    return df

def calculate_atr_indicator(df: pd.DataFrame, period: int = ENTRY_ATR_PERIOD) -> pd.DataFrame:
    """حساب مؤشر متوسط المدى الحقيقي (ATR)."""
    df = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator ATR] أعمدة 'high', 'low', 'close' مفقودة أو فارغة.")
        df['atr'] = np.nan
        return df
    if len(df) < period + 1: # نحتاج إلى شمعة واحدة إضافية لحساب shift(1)
        logger.warning(f"⚠️ [Indicator ATR] بيانات غير كافية ({len(df)} < {period + 1}) لحساب ATR.")
        df['atr'] = np.nan
        return df

    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()

    # حساب True Range (TR) - تجاهل NaN أثناء حساب الحد الأقصى
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)

    # حساب ATR باستخدام EMA (باستخدام span يعطي نتيجة أقرب لـ TradingView من com=period-1)
    df['atr'] = tr.ewm(span=period, adjust=False).mean()
    return df


def calculate_bollinger_bands(df: pd.DataFrame, window: int = BOLLINGER_WINDOW, num_std: int = BOLLINGER_STD_DEV) -> pd.DataFrame:
    """حساب نطاقات بولينجر."""
    df = df.copy()
    if 'close' not in df.columns or df['close'].isnull().all():
        logger.warning("⚠️ [Indicator BB] عمود 'close' مفقود أو فارغ.")
        df['bb_middle'] = np.nan
        df['bb_upper'] = np.nan
        df['bb_lower'] = np.nan
        return df
    if len(df) < window:
         logger.warning(f"⚠️ [Indicator BB] بيانات غير كافية ({len(df)} < {window}) لحساب BB.")
         df['bb_middle'] = np.nan
         df['bb_upper'] = np.nan
         df['bb_lower'] = np.nan
         return df

    df['bb_middle'] = df['close'].rolling(window=window).mean()
    df['bb_std'] = df['close'].rolling(window=window).std()
    df['bb_upper'] = df['bb_middle'] + num_std * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - num_std * df['bb_std']
    return df


def calculate_macd(df: pd.DataFrame, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> pd.DataFrame:
    """حساب مؤشر MACD وخط الإشارة والهيستوجرام."""
    df = df.copy()
    if 'close' not in df.columns or df['close'].isnull().all():
        logger.warning("⚠️ [Indicator MACD] عمود 'close' مفقود أو فارغ.")
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['macd_hist'] = np.nan
        return df
    min_len = max(fast, slow, signal)
    if len(df) < min_len:
        logger.warning(f"⚠️ [Indicator MACD] بيانات غير كافية ({len(df)} < {min_len}) لحساب MACD.")
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['macd_hist'] = np.nan
        return df

    ema_fast = calculate_ema(df['close'], fast)
    ema_slow = calculate_ema(df['close'], slow)
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = calculate_ema(df['macd'], signal) # حساب EMA للـ MACD line
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df


def calculate_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    """حساب مؤشر ADX و DI+ و DI-."""
    df_calc = df.copy() # العمل على نسخة
    required_cols = ['high', 'low', 'close']
    if not all(col in df_calc.columns for col in required_cols) or df_calc[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator ADX] أعمدة 'high', 'low', 'close' مفقودة أو فارغة.")
        df_calc['adx'] = np.nan
        df_calc['di_plus'] = np.nan
        df_calc['di_minus'] = np.nan
        return df_calc
    # يتطلب ADX فترة + فترة إضافية للـ smoothing
    if len(df_calc) < period * 2:
        logger.warning(f"⚠️ [Indicator ADX] بيانات غير كافية ({len(df_calc)} < {period * 2}) لحساب ADX.")
        df_calc['adx'] = np.nan
        df_calc['di_plus'] = np.nan
        df_calc['di_minus'] = np.nan
        return df_calc

    # حساب True Range (TR)
    df_calc['high-low'] = df_calc['high'] - df_calc['low']
    df_calc['high-prev_close'] = abs(df_calc['high'] - df_calc['close'].shift(1))
    df_calc['low-prev_close'] = abs(df_calc['low'] - df_calc['close'].shift(1))
    df_calc['tr'] = df_calc[['high-low', 'high-prev_close', 'low-prev_close']].max(axis=1, skipna=False)

    # حساب Directional Movement (+DM, -DM)
    df_calc['up_move'] = df_calc['high'] - df_calc['high'].shift(1)
    df_calc['down_move'] = df_calc['low'].shift(1) - df_calc['low']
    df_calc['+dm'] = np.where((df_calc['up_move'] > df_calc['down_move']) & (df_calc['up_move'] > 0), df_calc['up_move'], 0)
    df_calc['-dm'] = np.where((df_calc['down_move'] > df_calc['up_move']) & (df_calc['down_move'] > 0), df_calc['down_move'], 0)

    # استخدام EMA لحساب القيم الملساء (alpha = 1/period)
    alpha = 1 / period
    df_calc['tr_smooth'] = df_calc['tr'].ewm(alpha=alpha, adjust=False).mean()
    df_calc['+dm_smooth'] = df_calc['+dm'].ewm(alpha=alpha, adjust=False).mean()
    df_calc['-dm_smooth'] = df_calc['-dm'].ewm(alpha=alpha, adjust=False).mean()

    # حساب Directional Indicators (DI+, DI-) وتجنب القسمة على صفر
    df_calc['di_plus'] = np.where(df_calc['tr_smooth'] > 0, 100 * (df_calc['+dm_smooth'] / df_calc['tr_smooth']), 0)
    df_calc['di_minus'] = np.where(df_calc['tr_smooth'] > 0, 100 * (df_calc['-dm_smooth'] / df_calc['tr_smooth']), 0)

    # حساب Directional Movement Index (DX)
    di_sum = df_calc['di_plus'] + df_calc['di_minus']
    df_calc['dx'] = np.where(di_sum > 0, 100 * abs(df_calc['di_plus'] - df_calc['di_minus']) / di_sum, 0)

    # حساب Average Directional Index (ADX) باستخدام EMA
    df_calc['adx'] = df_calc['dx'].ewm(alpha=alpha, adjust=False).mean()

    # إرجاع DataFrame مع الأعمدة الجديدة فقط (أو يمكن دمجها مع الأصلي)
    return df_calc[['adx', 'di_plus', 'di_minus']]


def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """حساب متوسط السعر المرجح بالحجم (VWAP) - يعاد تعيينه يوميًا."""
    df = df.copy()
    required_cols = ['high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator VWAP] أعمدة 'high', 'low', 'close', 'volume' مفقودة أو فارغة.")
        df['vwap'] = np.nan
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            # محاولة تحويل الفهرس إذا لم يكن DatetimeIndex
            df.index = pd.to_datetime(df.index)
            logger.warning("⚠️ [Indicator VWAP] تم تحويل الفهرس إلى DatetimeIndex.")
        except Exception:
            logger.error("❌ [Indicator VWAP] فشل تحويل الفهرس إلى DatetimeIndex، لا يمكن حساب VWAP اليومي.")
            df['vwap'] = np.nan
            return df

    df['date'] = df.index.date
    # حساب السعر النموذجي والحجم * السعر النموذجي
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol'] = df['typical_price'] * df['volume']

    # حساب المجاميع التراكمية ضمن كل يوم
    try:
        # Group by date and calculate cumulative sums
        df['cum_tp_vol'] = df.groupby('date')['tp_vol'].cumsum()
        df['cum_volume'] = df.groupby('date')['volume'].cumsum()
    except KeyError as e:
        logger.error(f"❌ [Indicator VWAP] خطأ في تجميع البيانات حسب التاريخ: {e}. قد يكون الفهرس غير صحيح.")
        df['vwap'] = np.nan
        # حذف الأعمدة المؤقتة إذا كانت موجودة
        df.drop(columns=['date', 'typical_price', 'tp_vol', 'cum_tp_vol', 'cum_volume'], inplace=True, errors='ignore')
        return df
    except Exception as e:
         logger.error(f"❌ [Indicator VWAP] خطأ غير متوقع في تجميع VWAP: {e}", exc_info=True)
         df['vwap'] = np.nan
         df.drop(columns=['date', 'typical_price', 'tp_vol', 'cum_tp_vol', 'cum_volume'], inplace=True, errors='ignore')
         return df


    # حساب VWAP وتجنب القسمة على صفر
    df['vwap'] = np.where(df['cum_volume'] > 0, df['cum_tp_vol'] / df['cum_volume'], np.nan)

    # ملء قيم NaN الأولية في بداية كل يوم باستخدام القيمة التالية (backfill)
    # حيث أن VWAP اليومي يتراكم، أول قيمة قد تكون NaN، نستخدم القيمة المحسوبة التالية
    df['vwap'] = df['vwap'].bfill()

    # إزالة الأعمدة المساعدة
    df.drop(columns=['date', 'typical_price', 'tp_vol', 'cum_tp_vol', 'cum_volume'], inplace=True, errors='ignore')
    return df


def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """حساب مؤشر حجم التداول المتوازن (On-Balance Volume - OBV)."""
    df = df.copy()
    required_cols = ['close', 'volume']
    if not all(col in df.columns for col in required_cols) or df[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator OBV] أعمدة 'close' أو 'volume' مفقودة أو فارغة.")
        df['obv'] = np.nan
        return df
    if not pd.api.types.is_numeric_dtype(df['close']) or not pd.api.types.is_numeric_dtype(df['volume']):
        logger.warning("⚠️ [Indicator OBV] الأعمدة 'close' أو 'volume' ليست رقمية.")
        df['obv'] = np.nan
        return df

    obv = np.zeros(len(df), dtype=np.float64) # استخدام numpy array أسرع
    close = df['close'].values
    volume = df['volume'].values

    # حساب التغيرات في الإغلاق مرة واحدة
    close_diff = df['close'].diff().values

    for i in range(1, len(df)):
        if np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(close_diff[i]):
            obv[i] = obv[i-1] # الحفاظ على القيمة السابقة في حالة وجود NaN
            continue

        if close_diff[i] > 0: # السعر ارتفع
            obv[i] = obv[i-1] + volume[i]
        elif close_diff[i] < 0: # السعر انخفض
             obv[i] = obv[i-1] - volume[i]
        else: # السعر لم يتغير
             obv[i] = obv[i-1]

    df['obv'] = obv
    return df


def calculate_supertrend(df: pd.DataFrame, period: int = SUPERTREND_PERIOD, multiplier: float = SUPERTREND_MULTIPLIER) -> pd.DataFrame:
    """حساب مؤشر SuperTrend."""
    df_st = df.copy()
    required_cols = ['high', 'low', 'close']
    if not all(col in df_st.columns for col in required_cols) or df_st[required_cols].isnull().all().any():
        logger.warning("⚠️ [Indicator SuperTrend] أعمدة 'high', 'low', 'close' مفقودة أو فارغة.")
        df_st['supertrend'] = np.nan
        df_st['supertrend_trend'] = 0 # 0: غير معروف, 1: صاعد, -1: هابط
        return df_st

    # التأكد من وجود عمود ATR أو حسابه
    if 'atr' not in df_st.columns or df_st['atr'].isnull().all():
        logger.debug(f"ℹ️ [Indicator SuperTrend] حساب ATR (period={period}) لـ SuperTrend...")
        # استخدام فترة ATR الخاصة بـ SuperTrend هنا
        df_st = calculate_atr_indicator(df_st, period=period)

    if 'atr' not in df_st.columns or df_st['atr'].isnull().all():
         logger.warning("⚠️ [Indicator SuperTrend] لا يمكن حساب SuperTrend بسبب عدم وجود قيم ATR صالحة.")
         df_st['supertrend'] = np.nan
         df_st['supertrend_trend'] = 0
         return df_st
    if len(df_st) < period:
        logger.warning(f"⚠️ [Indicator SuperTrend] بيانات غير كافية ({len(df_st)} < {period}) لحساب SuperTrend.")
        df_st['supertrend'] = np.nan
        df_st['supertrend_trend'] = 0
        return df_st

    # حساب النطاقات العلوية والسفلية الأساسية
    hl2 = (df_st['high'] + df_st['low']) / 2
    df_st['basic_ub'] = hl2 + multiplier * df_st['atr']
    df_st['basic_lb'] = hl2 - multiplier * df_st['atr']

    # تهيئة الأعمدة النهائية
    df_st['final_ub'] = 0.0
    df_st['final_lb'] = 0.0
    df_st['supertrend'] = np.nan
    df_st['supertrend_trend'] = 0 # 1 for uptrend, -1 for downtrend

    # استخدام .values للوصول الأسرع داخل الحلقة
    close = df_st['close'].values
    basic_ub = df_st['basic_ub'].values
    basic_lb = df_st['basic_lb'].values
    final_ub = df_st['final_ub'].values # سيتم تعديله داخل الحلقة
    final_lb = df_st['final_lb'].values # سيتم تعديله داخل الحلقة
    st = df_st['supertrend'].values     # سيتم تعديله داخل الحلقة
    st_trend = df_st['supertrend_trend'].values # سيتم تعديله داخل الحلقة

    # البدء من الشمعة الثانية (index 1) لأننا نقارن مع السابق
    for i in range(1, len(df_st)):
        # التعامل مع NaN في المدخلات الأساسية لهذه الشمعة
        if pd.isna(basic_ub[i]) or pd.isna(basic_lb[i]) or pd.isna(close[i]):
            # في حالة NaN، احتفظ بالقيم السابقة للـ final bands والـ supertrend والاتجاه
            final_ub[i] = final_ub[i-1]
            final_lb[i] = final_lb[i-1]
            st[i] = st[i-1]
            st_trend[i] = st_trend[i-1]
            continue

        # حساب Final Upper Band
        if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]

        # حساب Final Lower Band
        if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]

        # تحديد خط SuperTrend والاتجاه
        if st[i-1] == final_ub[i-1]: # إذا كان الاتجاه السابق هابطًا
            if close[i] <= final_ub[i]: # استمر في الهبوط
                st[i] = final_ub[i]
                st_trend[i] = -1
            else: # تغير الاتجاه إلى صاعد
                st[i] = final_lb[i]
                st_trend[i] = 1
        elif st[i-1] == final_lb[i-1]: # إذا كان الاتجاه السابق صاعدًا
            if close[i] >= final_lb[i]: # استمر في الصعود
                st[i] = final_lb[i]
                st_trend[i] = 1
            else: # تغير الاتجاه إلى هابط
                st[i] = final_ub[i]
                st_trend[i] = -1
        else: # الحالة الأولية (أو إذا كانت القيمة السابقة NaN)
             if close[i] > final_ub[i]: # بداية اتجاه صاعد
                 st[i] = final_lb[i]
                 st_trend[i] = 1
             elif close[i] < final_lb[i]: # بداية اتجاه هابط
                  st[i] = final_ub[i]
                  st_trend[i] = -1
             else: # إذا كان السعر بين النطاقين في البداية (نادر)
                  st[i] = np.nan # أو يمكن استخدام قيمة سابقة إن وجدت
                  st_trend[i] = 0


    # إعادة تعيين القيم المحسوبة إلى DataFrame
    df_st['final_ub'] = final_ub
    df_st['final_lb'] = final_lb
    df_st['supertrend'] = st
    df_st['supertrend_trend'] = st_trend

    # إزالة الأعمدة المساعدة
    df_st.drop(columns=['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, errors='ignore')

    return df_st


# ---------------------- نماذج الشموع اليابانية ----------------------

def is_hammer(row: pd.Series) -> int:
    """التحقق من نموذج المطرقة (إشارة صعودية)."""
    o, h, l, c = row.get('open'), row.get('high'), row.get('low'), row.get('close')
    if pd.isna([o, h, l, c]).any(): return 0
    body = abs(c - o)
    candle_range = h - l
    if candle_range == 0: return 0
    lower_shadow = min(o, c) - l
    upper_shadow = h - max(o, c)
    is_small_body = body < (candle_range * 0.35) # تسامح أكبر قليلاً للجسم
    is_long_lower_shadow = lower_shadow >= 1.8 * body if body > 0 else lower_shadow > candle_range * 0.6
    is_small_upper_shadow = upper_shadow <= body * 0.6 if body > 0 else upper_shadow < candle_range * 0.15
    return 100 if is_small_body and is_long_lower_shadow and is_small_upper_shadow else 0

def is_shooting_star(row: pd.Series) -> int:
    """التحقق من نموذج الشهاب (إشارة هبوطية)."""
    o, h, l, c = row.get('open'), row.get('high'), row.get('low'), row.get('close')
    if pd.isna([o, h, l, c]).any(): return 0
    body = abs(c - o)
    candle_range = h - l
    if candle_range == 0: return 0
    lower_shadow = min(o, c) - l
    upper_shadow = h - max(o, c)
    is_small_body = body < (candle_range * 0.35)
    is_long_upper_shadow = upper_shadow >= 1.8 * body if body > 0 else upper_shadow > candle_range * 0.6
    is_small_lower_shadow = lower_shadow <= body * 0.6 if body > 0 else lower_shadow < candle_range * 0.15
    return -100 if is_small_body and is_long_upper_shadow and is_small_lower_shadow else 0 # إشارة سالبة

def is_doji(row: pd.Series) -> int:
    """التحقق من نموذج دوجي (عدم يقين)."""
    o, h, l, c = row.get('open'), row.get('high'), row.get('low'), row.get('close')
    if pd.isna([o, h, l, c]).any(): return 0
    candle_range = h - l
    if candle_range == 0: return 0
    return 100 if abs(c - o) <= (candle_range * 0.1) else 0 # الجسم صغير جدًا

def compute_engulfing(df: pd.DataFrame, idx: int) -> int:
    """التحقق من نموذج الابتلاع الصعودي أو الهبوطي."""
    if idx == 0: return 0
    prev = df.iloc[idx - 1]
    curr = df.iloc[idx]
    # تحقق من NaN في القيم المطلوبة
    if pd.isna([prev['close'], prev['open'], curr['close'], curr['open']]).any():
        return 0

    # ابتلاع صعودي: شمعة سابقة هابطة، حالية صاعدة تبتلع جسم السابقة
    is_bullish = (prev['close'] < prev['open'] and curr['close'] > curr['open'] and
                  curr['open'] <= prev['close'] and curr['close'] >= prev['open'])
    # ابتلاع هبوطي: شمعة سابقة صاعدة، حالية هابطة تبتلع جسم السابقة
    is_bearish = (prev['close'] > prev['open'] and curr['close'] < curr['open'] and
                  curr['open'] >= prev['close'] and curr['close'] <= prev['open'])

    if is_bullish: return 100
    if is_bearish: return -100
    return 0

def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """تطبيق دوال اكتشاف نماذج الشموع على DataFrame."""
    df = df.copy()
    logger.debug("ℹ️ [Indicators] اكتشاف نماذج الشموع...")
    # تطبيق النماذج التي تعتمد على صف واحد
    df['Hammer'] = df.apply(is_hammer, axis=1)
    df['ShootingStar'] = df.apply(is_shooting_star, axis=1)
    df['Doji'] = df.apply(is_doji, axis=1)
    # df['SpinningTop'] = df.apply(is_spinning_top, axis=1) # يمكن إضافته إذا لزم الأمر

    # حساب الابتلاع يتطلب الوصول للصف السابق
    engulfing_values = [compute_engulfing(df, i) for i in range(len(df))]
    df['Engulfing'] = engulfing_values

    # تجميع إشارات الشموع الإيجابية والسلبية القوية
    # لاحظ: قيمة الإشارة هنا هي 100 أو 0، الوزن سيطبق لاحقًا في الاستراتيجية
    df['BullishCandleSignal'] = df.apply(lambda row: 1 if (row['Hammer'] == 100 or row['Engulfing'] == 100) else 0, axis=1)
    df['BearishCandleSignal'] = df.apply(lambda row: 1 if (row['ShootingStar'] == -100 or row['Engulfing'] == -100) else 0, axis=1)

    # حذف أعمدة النماذج الفردية إذا لم تكن مطلوبة لاحقًا
    # df.drop(columns=['Hammer', 'ShootingStar', 'Doji', 'Engulfing'], inplace=True, errors='ignore')
    logger.debug("✅ [Indicators] تم اكتشاف نماذج الشموع.")
    return df

# ---------------------- دوال مساعدة أخرى (Elliott, Swings, Volume) ----------------------
def detect_swings(prices: np.ndarray, order: int = SWING_ORDER) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """اكتشاف نقاط التأرجح (القمم والقيعان) في سلسلة زمنية (numpy array)."""
    n = len(prices)
    if n < 2 * order + 1: return [], []

    maxima_indices = []
    minima_indices = []

    # تحسين الأداء بتجنب الحلقة على الأطراف غير الضرورية
    for i in range(order, n - order):
        window = prices[i - order : i + order + 1]
        center_val = prices[i]

        # التحقق من NaN في النافذة
        if np.isnan(window).any(): continue

        is_max = np.all(center_val >= window) # هل هو أكبر أو يساوي الكل؟
        is_min = np.all(center_val <= window) # هل هو أصغر أو يساوي الكل؟
        # التأكد أنه القمة/القاع الوحيد في النافذة (لتجنب التكرار في المناطق المسطحة)
        is_unique_max = is_max and (np.sum(window == center_val) == 1)
        is_unique_min = is_min and (np.sum(window == center_val) == 1)

        if is_unique_max:
            # ضمان عدم وجود قمة قريبة جدًا (ضمن مسافة order)
            if not maxima_indices or i > maxima_indices[-1] + order:
                 maxima_indices.append(i)
        elif is_unique_min:
            # ضمان عدم وجود قاع قريب جدًا
            if not minima_indices or i > minima_indices[-1] + order:
                minima_indices.append(i)

    maxima = [(idx, prices[idx]) for idx in maxima_indices]
    minima = [(idx, prices[idx]) for idx in minima_indices]
    return maxima, minima

def detect_elliott_waves(df: pd.DataFrame, order: int = SWING_ORDER) -> List[Dict[str, Any]]:
    """محاولة بسيطة لتحديد موجات إليوت بناءً على تأرجحات هيستوجرام MACD."""
    if 'macd_hist' not in df.columns or df['macd_hist'].isnull().all():
        logger.warning("⚠️ [Elliott] عمود 'macd_hist' غير موجود أو فارغ لحساب موجات إليوت.")
        return []

    # استخدام القيم غير الفارغة فقط
    macd_values = df['macd_hist'].dropna().values
    if len(macd_values) < 2 * order + 1:
         logger.warning("⚠️ [Elliott] بيانات MACD hist غير كافية بعد إزالة NaN.")
         return []

    maxima, minima = detect_swings(macd_values, order=order)

    # دمج وترتيب جميع نقاط التأرجح حسب الفهرس الأصلي
    # (تحتاج إلى ربط الفهرس الأصلي من df بعد إزالة NaN)
    df_nonan_macd = df['macd_hist'].dropna()
    all_swings = sorted(
        [(df_nonan_macd.index[idx], val, 'max') for idx, val in maxima] +
        [(df_nonan_macd.index[idx], val, 'min') for idx, val in minima],
        key=lambda x: x[0] # الترتيب حسب الزمن (الفهرس الأصلي)
    )

    waves = []
    wave_number = 1
    for timestamp, val, typ in all_swings:
        # التصنيف بسيط جدًا هنا، قد لا يتبع قواعد إليوت بدقة
        wave_type = "Impulse" if (typ == 'max' and val > 0) or (typ == 'min' and val >= 0) else "Correction"
        waves.append({
            "wave": wave_number,
            "timestamp": str(timestamp),
            "macd_hist_value": float(val),
            "swing_type": typ,
            "classified_type": wave_type
        })
        wave_number += 1
    return waves


def fetch_recent_volume(symbol: str) -> float:
    """جلب حجم التداول بالـ USDT لآخر 15 دقيقة للرمز المحدد."""
    if not client:
         logger.error(f"❌ [Data Volume] عميل Binance غير مهيأ لجلب حجم التداول لـ {symbol}.")
         return 0.0
    try:
        logger.debug(f"ℹ️ [Data Volume] جلب حجم التداول (15 دقيقة) لـ {symbol}...")
        # جلب بيانات الدقيقة الواحدة لآخر 15 دقيقة
        klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=15)
        if not klines or len(klines) < 15:
             logger.warning(f"⚠️ [Data Volume] بيانات 1m غير كافية (أقل من 15 شمعة) للزوج {symbol}.")
             return 0.0

        # حجم التداول بالعملة المقابلة (Quote Asset Volume) هو الحقل الثامن (index 7)
        volume_usdt = sum(float(k[7]) for k in klines if len(k) > 7 and k[7])
        logger.debug(f"✅ [Data Volume] السيولة آخر 15 دقيقة للزوج {symbol}: {volume_usdt:.2f} USDT")
        return volume_usdt
    except (BinanceAPIException, BinanceRequestException) as binance_err:
         logger.error(f"❌ [Data Volume] خطأ من Binance API أو الشبكة عند جلب حجم التداول لـ {symbol}: {binance_err}")
         return 0.0
    except Exception as e:
        logger.error(f"❌ [Data Volume] خطأ غير متوقع في جلب حجم التداول للزوج {symbol}: {e}", exc_info=True)
        return 0.0

# ---------------------- دالة توليد تقرير الأداء الشامل ----------------------
def generate_performance_report() -> str:
    """توليد تقرير أداء شامل ومفصل من قاعدة البيانات."""
    logger.info("ℹ️ [Report] توليد تقرير الأداء...")
    if not check_db_connection() or not conn or not cur:
        return "❌ لا يمكن توليد التقرير، مشكلة في الاتصال بقاعدة البيانات."
    try:
        # استخدام cursor جديد داخل الدالة لضمان عدم التداخل
        with conn.cursor() as report_cur: # يستخدم RealDictCursor
            # 1. الإشارات المفتوحة
            report_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;")
            open_signals_count = (report_cur.fetchone() or {}).get('count', 0)

            # 2. إحصائيات الإشارات المغلقة
            report_cur.execute("""
                SELECT
                    COUNT(*) AS total_closed,
                    COUNT(CASE WHEN profit_percentage > 0 THEN 1 END) AS winning_signals,
                    COUNT(CASE WHEN profit_percentage < 0 THEN 1 END) AS losing_signals,
                    COUNT(CASE WHEN profit_percentage = 0 THEN 1 END) AS neutral_signals,
                    COALESCE(SUM(profit_percentage), 0) AS total_profit_pct,
                    COALESCE(AVG(profit_percentage), 0) AS avg_profit_pct,
                    COALESCE(SUM(CASE WHEN profit_percentage > 0 THEN profit_percentage ELSE 0 END), 0) AS gross_profit_pct,
                    COALESCE(SUM(CASE WHEN profit_percentage < 0 THEN profit_percentage ELSE 0 END), 0) AS gross_loss_pct,
                    COALESCE(AVG(CASE WHEN profit_percentage > 0 THEN profit_percentage END), 0) AS avg_win_pct,
                    COALESCE(AVG(CASE WHEN profit_percentage < 0 THEN profit_percentage END), 0) AS avg_loss_pct
                FROM signals
                WHERE achieved_target = TRUE OR hit_stop_loss = TRUE;
            """)
            closed_stats = report_cur.fetchone() or {} # التعامل مع حالة عدم وجود نتائج

            total_closed = closed_stats.get('total_closed', 0)
            winning_signals = closed_stats.get('winning_signals', 0)
            losing_signals = closed_stats.get('losing_signals', 0)
            total_profit_pct = closed_stats.get('total_profit_pct', 0.0)
            gross_profit_pct = closed_stats.get('gross_profit_pct', 0.0)
            gross_loss_pct = closed_stats.get('gross_loss_pct', 0.0) # ستكون سالبة أو صفر
            avg_win_pct = closed_stats.get('avg_win_pct', 0.0)
            avg_loss_pct = closed_stats.get('avg_loss_pct', 0.0) # ستكون سالبة أو صفر

            # 3. حساب المقاييس المشتقة
            win_rate = (winning_signals / total_closed * 100) if total_closed > 0 else 0.0
             # Profit Factor: Total Profit / Absolute Total Loss
            profit_factor = (gross_profit_pct / abs(gross_loss_pct)) if gross_loss_pct != 0 else float('inf')

        # 4. تنسيق التقرير
        report = (
            f"📊 *تقرير الأداء الشامل:*\n"
            f"——————————————\n"
            f"📈 الإشارات المفتوحة حاليًا: *{open_signals_count}*\n"
            f"——————————————\n"
            f"📉 *إحصائيات الإشارات المغلقة:*\n"
            f"  • إجمالي الإشارات المغلقة: *{total_closed}*\n"
            f"  ✅ إشارات رابحة: *{winning_signals}*\n"
            f"  ❌ إشارات خاسرة: *{losing_signals}*\n"
            f"  • معدل الربح (Win Rate): *{win_rate:.2f}%*\n"
            f"——————————————\n"
            f"💰 *الربحية:*\n"
            f"  • صافي الربح/الخسارة (إجمالي %): *{total_profit_pct:+.2f}%*\n"
            f"  • إجمالي ربح (%): *{gross_profit_pct:+.2f}%*\n"
            f"  • إجمالي خسارة (%): *{gross_loss_pct:.2f}%*\n"
            f"  • متوسط ربح الصفقة الرابحة: *{avg_win_pct:+.2f}%*\n"
            f"  • متوسط خسارة الصفقة الخاسرة: *{avg_loss_pct:.2f}%*\n"
            f"  • معامل الربح (Profit Factor): *{'∞' if profit_factor == float('inf') else f'{profit_factor:.2f}'}*\n"
            f"——————————————\n"
            f"🕰️ _التقرير حتى: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        )
        logger.info("✅ [Report] تم توليد تقرير الأداء بنجاح.")
        return report

    except psycopg2.Error as db_err:
        logger.error(f"❌ [Report] خطأ في قاعدة البيانات عند توليد تقرير الأداء: {db_err}")
        if conn: conn.rollback() # تراجع عن أي معاملة قد تكون مفتوحة
        return "❌ خطأ في قاعدة البيانات عند توليد تقرير الأداء."
    except Exception as e:
        logger.error(f"❌ [Report] خطأ غير متوقع في توليد تقرير الأداء: {e}", exc_info=True)
        return "❌ خطأ غير متوقع في توليد تقرير الأداء."

# ---------------------- استراتيجية التداول المحافظة (المعدلة) ----------------------
class ConservativeTradingStrategy:
    """تغليف منطق استراتيجية التداول والمؤشرات المرتبطة بها مع نظام نقاط."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        # الأعمدة المطلوبة لحساب المؤشرات
        self.required_cols_indicators = [
            'open', 'high', 'low', 'close', 'volume', # أساسية
            'ema_trend', 'rsi', 'atr', 'bb_upper', 'bb_lower', 'bb_middle',
            'macd', 'macd_signal', 'macd_hist',
            'adx', 'di_plus', 'di_minus',
            'vwap', 'obv', 'supertrend', 'supertrend_trend',
            'BullishCandleSignal', 'BearishCandleSignal' # من نماذج الشموع
        ]
        # الأعمدة المطلوبة لتوليد إشارة الشراء (مع OBV السابق)
        self.required_cols_buy_signal = [
            'close', 'ema_trend', 'rsi', 'atr', 'macd', 'macd_signal',
            'supertrend_trend', 'adx', 'di_plus', 'di_minus', 'vwap', 'bb_upper',
            'BullishCandleSignal', 'obv'
        ]

        # =====================================================================
        # --- نظام النقاط (الأوزان) لشروط الشراء ---
        # قم بتعديل هذه الأوزان لتعكس أهمية كل شرط في استراتيجيتك
        # =====================================================================
        self.condition_weights = {
            'ema_up': 2.0,          # السعر فوق EMA
            'supertrend_up': 2.5,   # SuperTrend صاعد (أهم)
            'above_vwap': 1.5,      # السعر فوق VWAP
            'macd_bullish': 2.0,    # تقاطع MACD إيجابي
            'adx_trending_bullish': 2.0, # ADX قوي و DI+ أعلى
            'rsi_ok': 1.0,          # RSI في منطقة مقبولة (ليس شراء مفرط)
            'bullish_candle': 1.5,  # وجود شمعة ابتلاع أو مطرقة
            'not_bb_extreme': 0.5,  # السعر ليس عند نطاق بولينجر العلوي (أقل أهمية)
            'obv_rising': 2.0       # OBV يرتفع (تأكيد حجم التداول)
        }
        # =====================================================================

        # حساب إجمالي النقاط الممكنة
        self.total_possible_score = sum(self.condition_weights.values())

        # =====================================================================
        # --- عتبة درجة الإشارة المطلوبة (كنسبة مئوية) ---
        # مثال: 70% تعني أن الإشارة يجب أن تحقق 70% من إجمالي النقاط الممكنة
        # =====================================================================
        self.min_score_threshold_pct = 0.70 # 70%
        self.min_signal_score = self.total_possible_score * self.min_score_threshold_pct
        # =====================================================================

    def populate_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """حساب جميع المؤشرات المطلوبة للاستراتيجية."""
        logger.debug(f"ℹ️ [Strategy {self.symbol}] حساب المؤشرات...")
        # الحد الأدنى لعدد الصفوف يعتمد على أطول فترة مطلوبة للمؤشرات
        min_len_required = max(EMA_PERIOD, RSI_PERIOD, ENTRY_ATR_PERIOD, BOLLINGER_WINDOW, MACD_SLOW, ADX_PERIOD*2, SUPERTREND_PERIOD, LOOKBACK_FOR_SWINGS) + 5 # إضافة هامش

        if len(df) < min_len_required:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame قصير جدًا ({len(df)} < {min_len_required}) لحساب المؤشرات.")
            return None

        try:
            df_calc = df.copy()
            # ---- تسلسل حساب المؤشرات مهم (الاعتماديات) ----
            # ATR مطلوب لـ SuperTrend و وقف الخسارة/الهدف
            df_calc = calculate_atr_indicator(df_calc, ENTRY_ATR_PERIOD)
            # SuperTrend يحتاج ATR محسوب بفترته الخاصة
            df_calc = calculate_supertrend(df_calc, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
            # باقي المؤشرات
            df_calc['ema_trend'] = calculate_ema(df_calc['close'], EMA_PERIOD)
            df_calc = calculate_rsi_indicator(df_calc, RSI_PERIOD)
            df_calc = calculate_bollinger_bands(df_calc, BOLLINGER_WINDOW, BOLLINGER_STD_DEV)
            df_calc = calculate_macd(df_calc, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            adx_df = calculate_adx(df_calc, ADX_PERIOD) # حساب ADX في DataFrame منفصل مؤقتًا
            df_calc = df_calc.join(adx_df) # ضم النتائج
            df_calc = calculate_vwap(df_calc)
            df_calc = calculate_obv(df_calc)
            df_calc = detect_candlestick_patterns(df_calc) # يحسب BullishCandleSignal

            # --- التحقق من الأعمدة المطلوبة بعد الحساب ---
            missing_cols = [col for col in self.required_cols_indicators if col not in df_calc.columns]
            if missing_cols:
                 logger.error(f"❌ [Strategy {self.symbol}] أعمدة مؤشرات مطلوبة مفقودة بعد الحساب: {missing_cols}")
                 logger.debug(f"Columns present: {df_calc.columns.tolist()}")
                 return None

            # --- التعامل مع NaN بعد حساب *كل* المؤشرات ---
            initial_len = len(df_calc)
            # حذف الصفوف التي تحتوي على NaN في أي من الأعمدة المطلوبة
            df_cleaned = df_calc.dropna(subset=self.required_cols_indicators).copy()
            dropped_count = initial_len - len(df_cleaned)

            if dropped_count > 0:
                 logger.debug(f"ℹ️ [Strategy {self.symbol}] تم حذف {dropped_count} صف بسبب NaN في المؤشرات.")
            if df_cleaned.empty:
                logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame فارغ بعد إزالة NaN من المؤشرات.")
                return None

            latest = df_cleaned.iloc[-1]
            logger.debug(f"✅ [Strategy {self.symbol}] تم حساب المؤشرات. آخر اتجاه SuperTrend: {latest.get('supertrend_trend', 'N/A')}, ADX: {latest.get('adx', np.nan):.2f}")
            return df_cleaned

        except KeyError as ke:
             logger.error(f"❌ [Strategy {self.symbol}] خطأ: العمود المطلوب غير موجود أثناء حساب المؤشرات: {ke}", exc_info=True)
             return None
        except Exception as e:
            logger.error(f"❌ [Strategy {self.symbol}] خطأ غير متوقع أثناء حساب المؤشرات: {e}", exc_info=True)
            return None

    def generate_buy_signal(self, df_processed: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        توليد إشارة شراء بناءً على DataFrame المعالج ونظام النقاط.
        """
        logger.debug(f"ℹ️ [Strategy {self.symbol}] توليد إشارة الشراء...")

        # 1. التحقق من صحة DataFrame المدخل والأعمدة المطلوبة للإشارة
        if df_processed is None or df_processed.empty or len(df_processed) < 2: # نحتاج صفين للمقارنة (OBV)
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame فارغ أو قصير جدًا (<2)، لا يمكن توليد إشارة.")
            return None
        missing_cols = [col for col in self.required_cols_buy_signal if col not in df_processed.columns]
        if missing_cols:
            logger.warning(f"⚠️ [Strategy {self.symbol}] DataFrame يفتقد أعمدة مطلوبة للإشارة: {missing_cols}.")
            return None

        # 2. فحص ترند البيتكوين (شرط أولي)
        btc_trend = get_btc_trend_4h()
        if "هبوط" in btc_trend:
            logger.info(f"ℹ️ [Strategy {self.symbol}] التداول متوقف مؤقتًا بسبب ترند البيتكوين الهابط ({btc_trend}).")
            return None
        elif "N/A" in btc_trend:
             logger.warning(f"⚠️ [Strategy {self.symbol}] لا يمكن تحديد ترند البيتكوين، سيتم تجاهل هذا الشرط.")

        # 3. استخلاص بيانات الشمعة الأخيرة والسابقة والتحقق من NaN
        last_row = df_processed.iloc[-1]
        prev_row = df_processed.iloc[-2] # نحصل على الصف السابق لمقارنة OBV

        # التحقق من NaN في الأعمدة المطلوبة للصف الأخير
        last_row_check = last_row[self.required_cols_buy_signal]
        if last_row_check.isnull().any():
            nan_cols = last_row_check[last_row_check.isnull()].index.tolist()
            logger.warning(f"⚠️ [Strategy {self.symbol}] الصف الأخير يحتوي على NaN في أعمدة مطلوبة للإشارة: {nan_cols}.")
            return None
        # التحقق من NaN في OBV للصف السابق
        if pd.isna(prev_row['obv']):
           logger.warning(f"⚠️ [Strategy {self.symbol}] قيمة OBV السابقة هي NaN. لا يمكن التحقق من اتجاه OBV.")
           return None

        # 4. تطبيق شروط الشراء وحساب الدرجة بناءً على الأوزان
        signal_details = {}
        current_score = 0.0

        # التحقق من كل شرط وإضافة وزنه إلى الدرجة إذا تحقق
        if last_row['close'] > last_row['ema_trend']:
            current_score += self.condition_weights['ema_up']
            signal_details['EMA'] = f'Above {EMA_PERIOD} EMA (+{self.condition_weights["ema_up"]})'
        if last_row['supertrend_trend'] == 1:
            current_score += self.condition_weights['supertrend_up']
            signal_details['SuperTrend'] = f'Up Trend (+{self.condition_weights["supertrend_up"]})'
        if last_row['close'] > last_row['vwap']:
            current_score += self.condition_weights['above_vwap']
            signal_details['VWAP'] = f'Above VWAP (+{self.condition_weights["above_vwap"]})'
        if last_row['macd'] > last_row['macd_signal']:
            current_score += self.condition_weights['macd_bullish']
            signal_details['MACD'] = f'Bullish Cross (+{self.condition_weights["macd_bullish"]})'
        if last_row['adx'] > 20 and last_row['di_plus'] > last_row['di_minus']:
            current_score += self.condition_weights['adx_trending_bullish']
            signal_details['ADX/DI'] = f'Trending Bullish (ADX:{last_row["adx"]:.1f}, DI+>DI-) (+{self.condition_weights["adx_trending_bullish"]})'
        if last_row['rsi'] < RSI_OVERBOUGHT and last_row['rsi'] > RSI_OVERSOLD: # التأكد أنه ليس شراء أو بيع مفرط
            current_score += self.condition_weights['rsi_ok']
            signal_details['RSI'] = f'OK ({RSI_OVERSOLD}<{last_row["rsi"]:.1f}<{RSI_OVERBOUGHT}) (+{self.condition_weights["rsi_ok"]})'
        if last_row['BullishCandleSignal'] == 1: # القيمة الآن 0 أو 1
            current_score += self.condition_weights['bullish_candle']
            signal_details['Candle'] = f'Bullish Pattern (+{self.condition_weights["bullish_candle"]})'
        if last_row['close'] < last_row['bb_upper']:
            current_score += self.condition_weights['not_bb_extreme']
            signal_details['Bollinger'] = f'Not at Upper Band (+{self.condition_weights["not_bb_extreme"]})'
        if last_row['obv'] > prev_row['obv']:
            current_score += self.condition_weights['obv_rising']
            signal_details['OBV'] = f'Rising (+{self.condition_weights["obv_rising"]})'

        # --- قرار الشراء النهائي بناءً على الدرجة ---
        if current_score < self.min_signal_score:
            logger.debug(f"ℹ️ [Strategy {self.symbol}] لم تتحقق درجة الإشارة المطلوبة (Score: {current_score:.2f} / {self.total_possible_score:.2f}, Threshold: {self.min_signal_score:.2f}).")
            return None # لم تتحقق الشروط

        # 5. فحص حجم التداول (السيولة) - يبقى كما هو
        volume_recent = fetch_recent_volume(self.symbol)
        if volume_recent < MIN_VOLUME_15M_USDT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] السيولة ({volume_recent:,.0f} USDT) أقل من الحد الأدنى ({MIN_VOLUME_15M_USDT:,.0f} USDT). تم رفض الإشارة.")
            return None

        # 6. حساب الهدف ووقف الخسارة الأولي بناءً على ATR - يبقى كما هو
        current_price = last_row['close']
        current_atr = last_row.get('atr')

        # تعديل المضاعفات بناءً على قوة ADX (يمكن الإبقاء عليه أو تعديله)
        adx_val_sig = last_row.get('adx', 0)
        if adx_val_sig > 25: # ترند قوي
            target_multiplier = ENTRY_ATR_MULTIPLIER # استخدام القيمة المعدلة
            stop_loss_multiplier = ENTRY_ATR_MULTIPLIER * 0.8 # يمكن تعديل مضاعف الوقف للترند القوي
            signal_details['SL_Target_Mode'] = f'Strong Trend (ADX {adx_val_sig:.1f})'
        else: # ترند أضعف أو غير واضح
            target_multiplier = ENTRY_ATR_MULTIPLIER
            stop_loss_multiplier = ENTRY_ATR_MULTIPLIER
            signal_details['SL_Target_Mode'] = f'Standard (ADX {adx_val_sig:.1f})'

        initial_target = current_price + (target_multiplier * current_atr)
        initial_stop_loss = current_price - (stop_loss_multiplier * current_atr)

        # ضمان أن وقف الخسارة لا يساوي صفرًا أو سالبًا
        if initial_stop_loss <= 0:
            min_sl_price = current_price * (1 - 0.10) # مثال: 10% كحد أقصى للخسارة الأولية
            initial_stop_loss = max(min_sl_price, current_price * 0.001)
            logger.warning(f"⚠️ [Strategy {self.symbol}] وقف الخسارة المحسوب ({initial_stop_loss}) غير صالح. تم تعديله إلى {initial_stop_loss:.8f}")
            signal_details['Warning'] = f'Initial SL adjusted (was <= 0, set to {initial_stop_loss:.8f})'

        # 7. فحص هامش الربح الأدنى - يبقى كما هو
        profit_margin_pct = ((initial_target / current_price) - 1) * 100 if current_price > 0 else 0
        if profit_margin_pct < MIN_PROFIT_MARGIN_PCT:
            logger.info(f"ℹ️ [Strategy {self.symbol}] هامش الربح ({profit_margin_pct:.2f}%) أقل من الحد الأدنى المطلوب ({MIN_PROFIT_MARGIN_PCT:.2f}%). تم رفض الإشارة.")
            return None

        # 8. تجميع بيانات الإشارة النهائية مع الدرجة الموزونة
        signal_output = {
            'symbol': self.symbol,
            'entry_price': float(f"{current_price:.8g}"),
            'initial_target': float(f"{initial_target:.8g}"),
            'initial_stop_loss': float(f"{initial_stop_loss:.8g}"),
            'current_target': float(f"{initial_target:.8g}"),
            'current_stop_loss': float(f"{initial_stop_loss:.8g}"),
            'r2_score': float(f"{current_score:.2f}"), # تخزين الدرجة الموزونة هنا
            'strategy_name': 'Conservative_Weighted', # تغيير اسم الاستراتيجية ليعكس التعديل
            'signal_details': signal_details, # تفاصيل الشروط المحققة وأوزانها
            'volume_15m': volume_recent,
            'trade_value': TRADE_VALUE,
            'total_possible_score': float(f"{self.total_possible_score:.2f}") # إضافة إجمالي النقاط الممكنة
        }

        logger.info(f"✅ [Strategy {self.symbol}] إشارة شراء مؤكدة. السعر: {current_price:.6f}, Score: {current_score:.2f}/{self.total_possible_score:.2f}, ATR: {current_atr:.6f}, Volume: {volume_recent:,.0f}")
        return signal_output

# ---------------------- دوال Telegram ----------------------
def send_telegram_message(target_chat_id: str, text: str, reply_markup: Optional[Dict] = None, parse_mode: str = 'Markdown', disable_web_page_preview: bool = True, timeout: int = 20) -> Optional[Dict]:
    """إرسال رسالة عبر Telegram Bot API مع معالجة أفضل للأخطاء."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': str(target_chat_id),
        'text': text,
        'parse_mode': parse_mode,
        'disable_web_page_preview': disable_web_page_preview
    }
    if reply_markup:
        try:
            payload['reply_markup'] = json.dumps(convert_np_values(reply_markup))
        except (TypeError, ValueError) as json_err:
             logger.error(f"❌ [Telegram] فشل تحويل reply_markup إلى JSON: {json_err} - Markup: {reply_markup}")
             return None

    logger.debug(f"ℹ️ [Telegram] إرسال رسالة إلى {target_chat_id}...")
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        logger.info(f"✅ [Telegram] تم إرسال الرسالة بنجاح إلى {target_chat_id}.")
        return response.json()
    except requests.exceptions.Timeout:
         logger.error(f"❌ [Telegram] فشل إرسال الرسالة إلى {target_chat_id} (Timeout).")
         return None
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة إلى {target_chat_id} (HTTP Error: {http_err.response.status_code}).")
        try:
            error_details = http_err.response.json()
            logger.error(f"❌ [Telegram] تفاصيل خطأ API: {error_details}")
        except json.JSONDecodeError:
            logger.error(f"❌ [Telegram] لم يتمكن من فك تشفير استجابة الخطأ: {http_err.response.text}")
        return None
    except requests.exceptions.RequestException as req_err:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة إلى {target_chat_id} (Request Error): {req_err}")
        return None
    except Exception as e:
         logger.error(f"❌ [Telegram] خطأ غير متوقع أثناء إرسال الرسالة: {e}", exc_info=True)
         return None

def send_telegram_alert(signal_data: Dict[str, Any], timeframe: str) -> None:
    """تنسيق وإرسال تنبيه إشارة تداول جديدة إلى Telegram مع عرض الدرجة."""
    logger.debug(f"ℹ️ [Telegram Alert] تنسيق وإرسال تنبيه للإشارة: {signal_data.get('symbol', 'N/A')}")
    try:
        entry_price = float(signal_data['entry_price'])
        target_price = float(signal_data['initial_target'])
        stop_loss_price = float(signal_data['initial_stop_loss'])
        symbol = signal_data['symbol']
        strategy_name = signal_data.get('strategy_name', 'N/A')
        signal_score = signal_data.get('r2_score', 0.0) # الدرجة الموزونة
        total_possible_score = signal_data.get('total_possible_score', 10.0) # القيمة الافتراضية قد تحتاج للتعديل
        volume_15m = signal_data.get('volume_15m', 0.0)
        trade_value_signal = signal_data.get('trade_value', TRADE_VALUE)

        profit_pct = ((target_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        loss_pct = ((stop_loss_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        profit_usdt = trade_value_signal * (profit_pct / 100)
        loss_usdt = abs(trade_value_signal * (loss_pct / 100))

        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')

        fear_greed = get_fear_greed_index()
        btc_trend = get_btc_trend_4h()

        # بناء الرسالة مع الدرجة الموزونة
        message = (
            f"💡 *إشارة تداول جديدة ({strategy_name.replace('_', ' ').title()})* 💡\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"📈 **نوع الإشارة:** شراء (Long)\n"
            f"🕰️ **الإطار الزمني:** {timeframe}\n"
            # --- تعديل لعرض الدرجة ---
            f"📊 **قوة الإشارة (Score):** *{signal_score:.1f} / {total_possible_score:.1f}*\n"
            f"💧 **سيولة (15 دقيقة):** {volume_15m:,.0f} USDT\n"
            f"——————————————\n"
            f"➡️ **سعر الدخول المقترح:** `${entry_price:,.8g}`\n"
            f"🎯 **الهدف الأولي:** `${target_price:,.8g}` ({profit_pct:+.2f}% / ≈ ${profit_usdt:+.2f})\n"
            f"🛑 **وقف الخسارة الأولي:** `${stop_loss_price:,.8g}` ({loss_pct:.2f}% / ≈ ${loss_usdt:.2f})\n"
            f"——————————————\n"
            f"😨/🤑 **مؤشر الخوف والطمع:** {fear_greed}\n"
            f"₿ **اتجاه البيتكوين (4H):** {btc_trend}\n"
            f"——————————————\n"
            f"⏰ {timestamp_str}"
        )

        reply_markup = {
            "inline_keyboard": [
                [{"text": "📊 عرض تقرير الأداء", "callback_data": "get_report"}]
            ]
        }

        send_telegram_message(CHAT_ID, message, reply_markup=reply_markup, parse_mode='Markdown')

    except KeyError as ke:
        logger.error(f"❌ [Telegram Alert] بيانات الإشارة غير كاملة للزوج {signal_data.get('symbol', 'N/A')}: مفتاح مفقود {ke}", exc_info=True)
    except Exception as e:
        logger.error(f"❌ [Telegram Alert] فشل إرسال تنبيه الإشارة للزوج {signal_data.get('symbol', 'N/A')}: {e}", exc_info=True)

def send_tracking_notification(details: Dict[str, Any]) -> None:
    """تنسيق وإرسال تنبيهات تليجرام المحسّنة لحالات التتبع."""
    symbol = details.get('symbol', 'N/A')
    signal_id = details.get('id', 'N/A')
    notification_type = details.get('type', 'unknown')
    message = ""
    safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')
    closing_price = details.get('closing_price', 0.0)
    profit_pct = details.get('profit_pct', 0.0)
    current_price = details.get('current_price', 0.0)
    atr_value = details.get('atr_value', 0.0)
    new_stop_loss = details.get('new_stop_loss', 0.0)
    old_stop_loss = details.get('old_stop_loss', 0.0)

    logger.debug(f"ℹ️ [Notification] تنسيق إشعار تتبع: ID={signal_id}, Type={notification_type}, Symbol={symbol}")

    if notification_type == 'target_hit':
        message = (
            f"✅ *الهدف تحقق (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"🎯 **سعر الإغلاق (الهدف):** `${closing_price:,.8g}`\n"
            f"💰 **الربح المحقق:** {profit_pct:+.2f}%"
        )
    elif notification_type == 'stop_loss_hit':
        sl_type_msg = details.get('sl_type', 'بخسارة ❌') # القيمة المحسوبة من دالة التتبع
        message = (
            f"🛑 *وصل وقف الخسارة (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"🚫 **سعر الإغلاق (الوقف):** `${closing_price:,.8g}`\n"
            f"📉 **النتيجة:** {profit_pct:.2f}% ({sl_type_msg})"
        )
    elif notification_type == 'trailing_activated':
        activation_profit_pct = details.get('activation_profit_pct', TRAILING_STOP_ACTIVATION_PROFIT_PCT * 100)
        message = (
            f"⬆️ *تفعيل الوقف المتحرك (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"📈 **السعر الحالي (عند التفعيل):** `${current_price:,.8g}` (ربح > {activation_profit_pct:.1f}%)\n"
            f"📊 **قيمة ATR ({ENTRY_ATR_PERIOD}):** `{atr_value:,.8g}` (المضاعف: {TRAILING_STOP_ATR_MULTIPLIER})\n"
            f"🛡️ **وقف الخسارة الجديد:** `${new_stop_loss:,.8g}`"
        )
    elif notification_type == 'trailing_updated':
        trigger_price_increase_pct = details.get('trigger_price_increase_pct', TRAILING_STOP_MOVE_INCREMENT_PCT * 100)
        message = (
            f"➡️ *تحديث الوقف المتحرك (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"📈 **السعر الحالي (عند التحديث):** `${current_price:,.8g}` (+{trigger_price_increase_pct:.1f}% منذ آخر تحديث)\n"
            f"📊 **قيمة ATR ({ENTRY_ATR_PERIOD}):** `{atr_value:,.8g}` (المضاعف: {TRAILING_STOP_ATR_MULTIPLIER})\n"
            f"🔒 **الوقف السابق:** `${old_stop_loss:,.8g}`\n"
            f"🛡️ **وقف الخسارة الجديد:** `${new_stop_loss:,.8g}`"
        )
    else:
        logger.warning(f"⚠️ [Notification] نوع تنبيه غير معروف: {notification_type} للبيانات: {details}")
        return # لا ترسل شيئًا إذا كان النوع غير معروف

    if message:
        send_telegram_message(CHAT_ID, message, parse_mode='Markdown')

# ---------------------- دوال قاعدة البيانات (إدراج وتحديث) ----------------------
def insert_signal_into_db(signal: Dict[str, Any]) -> bool:
    """إدراج إشارة جديدة في جدول signals مع الدرجة الموزونة."""
    if not check_db_connection() or not conn:
        logger.error(f"❌ [DB Insert] فشل إدراج الإشارة {signal.get('symbol', 'N/A')} بسبب مشكلة في اتصال DB.")
        return False

    symbol = signal.get('symbol', 'N/A')
    logger.debug(f"ℹ️ [DB Insert] محاولة إدراج إشارة للزوج {symbol}...")
    try:
        signal_prepared = convert_np_values(signal)
        # تحويل تفاصيل الإشارة إلى JSON (تأكد من أنها لا تحتوي على أنواع numpy)
        signal_details_json = json.dumps(signal_prepared.get('signal_details', {}))

        with conn.cursor() as cur_ins:
            insert_query = sql.SQL("""
                INSERT INTO signals
                 (symbol, entry_price, initial_target, initial_stop_loss, current_target, current_stop_loss,
                 r2_score, strategy_name, signal_details, last_trailing_update_price, volume_15m)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """)
            cur_ins.execute(insert_query, (
                signal_prepared['symbol'],
                signal_prepared['entry_price'],
                signal_prepared['initial_target'],
                signal_prepared['initial_stop_loss'],
                signal_prepared['current_target'],
                signal_prepared['current_stop_loss'],
                signal_prepared.get('r2_score'), # الدرجة الموزونة
                signal_prepared.get('strategy_name', 'unknown'),
                signal_details_json,
                None, # last_trailing_update_price
                signal_prepared.get('volume_15m')
            ))
        conn.commit()
        logger.info(f"✅ [DB Insert] تم إدراج إشارة للزوج {symbol} في قاعدة البيانات (Score: {signal_prepared.get('r2_score')}).")
        return True
    except psycopg2.Error as db_err:
        logger.error(f"❌ [DB Insert] خطأ في قاعدة البيانات عند إدراج الإشارة للزوج {symbol}: {db_err}")
        if conn: conn.rollback()
        return False
    except (TypeError, ValueError) as convert_err:
         logger.error(f"❌ [DB Insert] خطأ تحويل بيانات الإشارة قبل الإدراج للزوج {symbol}: {convert_err} - Signal Data: {signal}")
         if conn: conn.rollback()
         return False
    except Exception as e:
        logger.error(f"❌ [DB Insert] خطأ غير متوقع في إدراج الإشارة للزوج {symbol}: {e}", exc_info=True)
        if conn: conn.rollback()
        return False

# ---------------------- دالة تتبع الإشارات المفتوحة ----------------------
def track_signals() -> None:
    """تتبع الإشارات المفتوحة، التحقق من الأهداف ووقف الخسارة، وتطبيق الوقف المتحرك."""
    logger.info("ℹ️ [Tracker] بدء عملية تتبع الإشارات المفتوحة...")
    while True:
        active_signals_summary: List[str] = []
        processed_in_cycle = 0
        try:
            if not check_db_connection() or not conn:
                logger.warning("⚠️ [Tracker] تخطي دورة التتبع بسبب مشكلة في اتصال DB.")
                time.sleep(15) # انتظار أطول قليلاً قبل إعادة المحاولة
                continue

            # استخدام cursor مع context manager لجلب الإشارات المفتوحة
            with conn.cursor() as track_cur: # يستخدم RealDictCursor
                 track_cur.execute("""
                    SELECT id, symbol, entry_price, initial_stop_loss, current_target, current_stop_loss,
                           is_trailing_active, last_trailing_update_price
                    FROM signals
                    WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;
                """)
                 open_signals: List[Dict] = track_cur.fetchall()

            if not open_signals:
                # logger.debug("ℹ️ [Tracker] لا توجد إشارات مفتوحة للتتبع.")
                time.sleep(10) # انتظر أقل إذا لم تكن هناك إشارات
                continue

            logger.debug(f"ℹ️ [Tracker] تتبع {len(open_signals)} إشارة مفتوحة...")

            for signal_row in open_signals:
                signal_id = signal_row['id']
                symbol = signal_row['symbol']
                processed_in_cycle += 1
                update_executed = False # لتتبع ما إذا تم تحديث هذه الإشارة في الدورة الحالية

                try:
                    # استخلاص وتحويل آمن للبيانات الرقمية
                    entry_price = float(signal_row['entry_price'])
                    initial_stop_loss = float(signal_row['initial_stop_loss'])
                    current_target = float(signal_row['current_target'])
                    current_stop_loss = float(signal_row['current_stop_loss'])
                    is_trailing_active = signal_row['is_trailing_active']
                    last_update_px = signal_row['last_trailing_update_price']
                    last_trailing_update_price = float(last_update_px) if last_update_px is not None else None

                    # الحصول على السعر الحالي من بيانات WebSocket Ticker
                    current_price = ticker_data.get(symbol)

                    if current_price is None:
                         logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): لا يتوفر سعر حالي في بيانات Ticker.")
                         continue # تخطي هذه الإشارة في هذه الدورة

                    active_signals_summary.append(f"{symbol}({signal_id}): P={current_price:.4f} T={current_target:.4f} SL={current_stop_loss:.4f} Trail={'On' if is_trailing_active else 'Off'}")

                    update_query: Optional[sql.SQL] = None
                    update_params: Tuple = ()
                    log_message: Optional[str] = None
                    notification_details: Dict[str, Any] = {'symbol': symbol, 'id': signal_id}

                    # --- منطق التحقق والتحديث ---
                    # 1. التحقق من الوصول للهدف
                    if current_price >= current_target:
                        profit_pct = ((current_target / entry_price) - 1) * 100 if entry_price > 0 else 0
                        update_query = sql.SQL("UPDATE signals SET achieved_target = TRUE, closing_price = %s, closed_at = NOW(), profit_percentage = %s WHERE id = %s;")
                        update_params = (current_target, profit_pct, signal_id)
                        log_message = f"🎯 [Tracker] {symbol}(ID:{signal_id}): وصل الهدف عند {current_target:.8g} (ربح: {profit_pct:+.2f}%)."
                        notification_details.update({'type': 'target_hit', 'closing_price': current_target, 'profit_pct': profit_pct})
                        update_executed = True

                    # 2. التحقق من الوصول لوقف الخسارة (يجب أن يكون بعد التحقق من الهدف)
                    elif current_price <= current_stop_loss:
                        loss_pct = ((current_stop_loss / entry_price) - 1) * 100 if entry_price > 0 else 0
                        profitable_sl = current_stop_loss > entry_price
                        sl_type_msg = "بربح ✅" if profitable_sl else "بخسارة ❌"
                        update_query = sql.SQL("UPDATE signals SET hit_stop_loss = TRUE, closing_price = %s, closed_at = NOW(), profit_percentage = %s, profitable_stop_loss = %s WHERE id = %s;")
                        update_params = (current_stop_loss, loss_pct, profitable_sl, signal_id)
                        log_message = f"🔻 [Tracker] {symbol}(ID:{signal_id}): وصل وقف الخسارة ({sl_type_msg}) عند {current_stop_loss:.8g} (نسبة: {loss_pct:.2f}%)."
                        notification_details.update({'type': 'stop_loss_hit', 'closing_price': current_stop_loss, 'profit_pct': loss_pct, 'sl_type': sl_type_msg})
                        update_executed = True

                    # 3. التحقق من تفعيل أو تحديث الوقف المتحرك (فقط إذا لم يتم ضرب الهدف أو الوقف)
                    else:
                        activation_threshold_price = entry_price * (1 + TRAILING_STOP_ACTIVATION_PROFIT_PCT)
                        # أ. تفعيل الوقف المتحرك
                        if not is_trailing_active and current_price >= activation_threshold_price:
                            logger.info(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): السعر {current_price:.8g} وصل لعتبة تفعيل الوقف ({activation_threshold_price:.8g}). جلب ATR...")
                            # استخدام الإطار الزمني المحدد للتتبع
                            df_atr = fetch_historical_data(symbol, interval=SIGNAL_TRACKING_TIMEFRAME, days=SIGNAL_TRACKING_LOOKBACK_DAYS)
                            if df_atr is not None and not df_atr.empty:
                                # استخدام فترة ATR المخصصة للدخول/التتبع
                                df_atr = calculate_atr_indicator(df_atr, period=ENTRY_ATR_PERIOD)
                                if not df_atr.empty and 'atr' in df_atr.columns and pd.notna(df_atr['atr'].iloc[-1]):
                                    current_atr_val = df_atr['atr'].iloc[-1]
                                    if current_atr_val > 0:
                                         new_stop_loss_calc = current_price - (TRAILING_STOP_ATR_MULTIPLIER * current_atr_val)
                                         new_stop_loss = max(new_stop_loss_calc, current_stop_loss, entry_price * (1 + 0.005)) # نضمن ربح بسيط جداً أو الحفاظ على الوقف الحالي

                                         if new_stop_loss > current_stop_loss: # فقط إذا كان الوقف الجديد أعلى فعلاً
                                            update_query = sql.SQL("UPDATE signals SET is_trailing_active = TRUE, current_stop_loss = %s, last_trailing_update_price = %s WHERE id = %s;")
                                            update_params = (new_stop_loss, current_price, signal_id)
                                            log_message = f"⬆️✅ [Tracker] {symbol}(ID:{signal_id}): تفعيل الوقف المتحرك. السعر={current_price:.8g}, ATR={current_atr_val:.8g}. الوقف الجديد: {new_stop_loss:.8g}"
                                            notification_details.update({'type': 'trailing_activated', 'current_price': current_price, 'atr_value': current_atr_val, 'new_stop_loss': new_stop_loss, 'activation_profit_pct': TRAILING_STOP_ACTIVATION_PROFIT_PCT * 100})
                                            update_executed = True
                                         else:
                                            logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): الوقف المتحرك المحسوب ({new_stop_loss:.8g}) ليس أعلى من الوقف الحالي ({current_stop_loss:.8g}). لن يتم التفعيل.")
                                    else: logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): قيمة ATR غير صالحة ({current_atr_val}) لتفعيل الوقف المتحرك.")
                                else: logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): لا يمكن حساب ATR لتفعيل الوقف المتحرك.")
                            else: logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): لا يمكن جلب بيانات لحساب ATR لتفعيل الوقف المتحرك.")

                        # ب. تحديث الوقف المتحرك
                        elif is_trailing_active and last_trailing_update_price is not None:
                            update_threshold_price = last_trailing_update_price * (1 + TRAILING_STOP_MOVE_INCREMENT_PCT)
                            if current_price >= update_threshold_price:
                                logger.info(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): السعر {current_price:.8g} وصل لعتبة تحديث الوقف ({update_threshold_price:.8g}). جلب ATR...")
                                df_recent = fetch_historical_data(symbol, interval=SIGNAL_TRACKING_TIMEFRAME, days=SIGNAL_TRACKING_LOOKBACK_DAYS)
                                if df_recent is not None and not df_recent.empty:
                                    df_recent = calculate_atr_indicator(df_recent, period=ENTRY_ATR_PERIOD)
                                    if not df_recent.empty and 'atr' in df_recent.columns and pd.notna(df_recent['atr'].iloc[-1]):
                                         current_atr_val_update = df_recent['atr'].iloc[-1]
                                         if current_atr_val_update > 0:
                                             potential_new_stop_loss = current_price - (TRAILING_STOP_ATR_MULTIPLIER * current_atr_val_update)
                                             if potential_new_stop_loss > current_stop_loss:
                                                new_stop_loss_update = potential_new_stop_loss
                                                update_query = sql.SQL("UPDATE signals SET current_stop_loss = %s, last_trailing_update_price = %s WHERE id = %s;")
                                                update_params = (new_stop_loss_update, current_price, signal_id)
                                                log_message = f"➡️🔼 [Tracker] {symbol}(ID:{signal_id}): تحديث الوقف المتحرك. السعر={current_price:.8g}, ATR={current_atr_val_update:.8g}. القديم={current_stop_loss:.8g}, الجديد: {new_stop_loss_update:.8g}"
                                                notification_details.update({'type': 'trailing_updated', 'current_price': current_price, 'atr_value': current_atr_val_update, 'old_stop_loss': current_stop_loss, 'new_stop_loss': new_stop_loss_update, 'trigger_price_increase_pct': TRAILING_STOP_MOVE_INCREMENT_PCT * 100})
                                                update_executed = True
                                             else:
                                                 logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): الوقف المتحرك المحسوب ({potential_new_stop_loss:.8g}) ليس أعلى من الحالي ({current_stop_loss:.8g}). لن يتم التحديث.")
                                         else: logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): قيمة ATR غير صالحة ({current_atr_val_update}) لتحديث الوقف.")
                                    else: logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): لا يمكن حساب ATR لتحديث الوقف.")
                                else: logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): لا يمكن جلب بيانات لحساب ATR لتحديث الوقف.")

                    # --- تنفيذ التحديث في قاعدة البيانات وإرسال التنبيه ---
                    if update_executed and update_query:
                        try:
                             with conn.cursor() as update_cur:
                                  update_cur.execute(update_query, update_params)
                             conn.commit()
                             if log_message: logger.info(log_message)
                             if notification_details.get('type'):
                                send_tracking_notification(notification_details)
                        except psycopg2.Error as db_err:
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ DB أثناء التحديث: {db_err}")
                            if conn: conn.rollback()
                        except Exception as exec_err:
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ غير متوقع أثناء تنفيذ التحديث/الإشعار: {exec_err}", exc_info=True)
                            if conn: conn.rollback()

                except (TypeError, ValueError) as convert_err:
                    logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ في تحويل قيم الإشارة الأولية: {convert_err}")
                    continue
                except Exception as inner_loop_err:
                     logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ غير متوقع في معالجة الإشارة: {inner_loop_err}", exc_info=True)
                     continue

            if active_signals_summary:
                logger.debug(f"ℹ️ [Tracker] حالة نهاية الدورة ({processed_in_cycle} معالج): {'; '.join(active_signals_summary)}")

            time.sleep(3) # الانتظار بين دورات التتبع

        except psycopg2.Error as db_cycle_err:
             logger.error(f"❌ [Tracker] خطأ قاعدة بيانات في دورة التتبع الرئيسية: {db_cycle_err}. محاولة إعادة الاتصال...")
             if conn: conn.rollback()
             time.sleep(30)
             check_db_connection()
        except Exception as cycle_err:
            logger.error(f"❌ [Tracker] خطأ غير متوقع في دورة تتبع الإشارات: {cycle_err}", exc_info=True)
            time.sleep(30)


# ---------------------- خدمة Flask (اختياري للـ Webhook) ----------------------
app = Flask(__name__)

@app.route('/')
def home() -> Response:
    """صفحة رئيسية بسيطة لإظهار أن البوت يعمل."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ws_alive = ws_thread.is_alive() if 'ws_thread' in globals() and ws_thread else False
    tracker_alive = tracker_thread.is_alive() if 'tracker_thread' in globals() and tracker_thread else False
    status = "running" if ws_alive and tracker_alive else "partially running"
    return Response(f"📈 Crypto Signal Bot ({status}) - Last Check: {now}", status=200, mimetype='text/plain')

@app.route('/favicon.ico')
def favicon() -> Response:
    """معالجة طلب أيقونة المفضلة لتجنب الخطأ 404 في السجلات."""
    return Response(status=204) # No Content

@app.route('/webhook', methods=['POST'])
def webhook() -> Tuple[str, int]:
    """معالجة الطلبات الواردة من Telegram (مثل ضغط الأزرار والأوامر)."""
    if not request.is_json:
        logger.warning("⚠️ [Flask] Received non-JSON webhook request.")
        return "Invalid request format", 400 # Bad Request

    try:
        data = request.get_json()
        logger.debug(f"ℹ️ [Flask] Received webhook data: {json.dumps(data)[:200]}...") # تسجيل جزء من البيانات فقط

        # معالجة ردود الأزرار (Callback Queries)
        if 'callback_query' in data:
            callback_query = data['callback_query']
            callback_id = callback_query['id']
            callback_data = callback_query.get('data')
            message_info = callback_query.get('message')
            if not message_info or not callback_data:
                 logger.warning(f"⚠️ [Flask] Callback query (ID: {callback_id}) missing message or data.")
                 try:
                     ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                     requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
                 except Exception as ack_err:
                     logger.warning(f"⚠️ [Flask] Failed to acknowledge invalid callback query {callback_id}: {ack_err}")
                 return "OK", 200

            chat_id_callback = message_info['chat']['id']
            message_id = message_info['message_id']
            user_info = callback_query.get('from', {})
            user_id = user_info.get('id')
            username = user_info.get('username', 'N/A')

            logger.info(f"ℹ️ [Flask] Received callback query: Data='{callback_data}', User={username}({user_id}), Chat={chat_id_callback}")

            # إرسال تأكيد الاستلام بسرعة
            try:
                ack_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                requests.post(ack_url, json={'callback_query_id': callback_id}, timeout=5)
            except Exception as ack_err:
                 logger.warning(f"⚠️ [Flask] Failed to acknowledge callback query {callback_id}: {ack_err}")

            # معالجة البيانات المستلمة من الزر
            if callback_data == "get_report":
                report_thread = Thread(target=lambda: send_telegram_message(chat_id_callback, generate_performance_report(), parse_mode='Markdown'))
                report_thread.start()
            else:
                logger.warning(f"⚠️ [Flask] Received unhandled callback data: '{callback_data}'")


        # معالجة الرسائل النصية (الأوامر)
        elif 'message' in data:
            message_data = data['message']
            chat_info = message_data.get('chat')
            user_info = message_data.get('from', {})
            text_msg = message_data.get('text', '').strip()

            if not chat_info or not text_msg:
                 logger.debug("ℹ️ [Flask] Received message without chat info or text.")
                 return "OK", 200

            chat_id_msg = chat_info['id']
            user_id = user_info.get('id')
            username = user_info.get('username', 'N/A')

            logger.info(f"ℹ️ [Flask] Received message: Text='{text_msg}', User={username}({user_id}), Chat={chat_id_msg}")

            # معالجة الأوامر المعروفة
            if text_msg.lower() == '/report':
                 report_thread = Thread(target=lambda: send_telegram_message(chat_id_msg, generate_performance_report(), parse_mode='Markdown'))
                 report_thread.start()
            elif text_msg.lower() == '/status':
                 status_thread = Thread(target=handle_status_command, args=(chat_id_msg,))
                 status_thread.start()

        else:
            logger.debug("ℹ️ [Flask] Received webhook data without 'callback_query' or 'message'.")

        return "OK", 200
    except Exception as e:
         logger.error(f"❌ [Flask] Error processing webhook: {e}", exc_info=True)
         return "Internal Server Error", 500

def handle_status_command(chat_id_msg: int) -> None:
    """دالة منفصلة لمعالجة أمر /status لتجنب حظر Webhook."""
    logger.info(f"ℹ️ [Flask Status] Handling /status command for chat {chat_id_msg}")
    status_msg = "⏳ جارٍ جلب الحالة..."
    msg_sent = send_telegram_message(chat_id_msg, status_msg)
    if not (msg_sent and msg_sent.get('ok')):
         logger.error(f"❌ [Flask Status] Failed to send initial status message to {chat_id_msg}")
         return

    message_id_to_edit = msg_sent['result']['message_id']
    try:
        open_count = 0
        if check_db_connection() and conn:
            with conn.cursor() as status_cur:
                status_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;")
                open_count = (status_cur.fetchone() or {}).get('count', 0)

        # التحقق من وجود المتغيرات قبل الوصول إليها
        ws_status = 'نشط ✅' if 'ws_thread' in globals() and ws_thread and ws_thread.is_alive() else 'غير نشط ❌'
        tracker_status = 'نشط ✅' if 'tracker_thread' in globals() and tracker_thread and tracker_thread.is_alive() else 'غير نشط ❌'
        final_status_msg = (
            f"🤖 *حالة البوت:*\n"
            f"- تتبع الأسعار (WS): {ws_status}\n"
            f"- تتبع الإشارات: {tracker_status}\n"
            f"- الإشارات النشطة: *{open_count}* / {MAX_OPEN_TRADES}\n"
            f"- وقت الخادم الحالي: {datetime.now().strftime('%H:%M:%S')}"
        )
        # تعديل الرسالة الأصلية
        edit_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText"
        edit_payload = {
            'chat_id': chat_id_msg,
             'message_id': message_id_to_edit,
            'text': final_status_msg,
            'parse_mode': 'Markdown'
        }
        response = requests.post(edit_url, json=edit_payload, timeout=10)
        response.raise_for_status()
        logger.info(f"✅ [Flask Status] Status updated for chat {chat_id_msg}")

    except Exception as status_err:
        logger.error(f"❌ [Flask Status] Error getting/editing status details for chat {chat_id_msg}: {status_err}", exc_info=True)
        send_telegram_message(chat_id_msg, "❌ حدث خطأ أثناء جلب تفاصيل الحالة.")


def run_flask() -> None:
    """تشغيل تطبيق Flask لسماع الـ Webhook باستخدام خادم إنتاجي إذا كان متاحًا."""
    if not WEBHOOK_URL:
        logger.info("ℹ️ [Flask] Webhook URL not configured. Flask server will not start.")
        return

    host = "0.0.0.0"
    port = int(config('PORT', default=10000)) # استخدام متغير بيئة PORT أو قيمة افتراضية
    logger.info(f"ℹ️ [Flask] Starting Flask app on {host}:{port}...")
    try:
        from waitress import serve
        logger.info("✅ [Flask] Using 'waitress' server.")
        serve(app, host=host, port=port, threads=6)
    except ImportError:
         logger.warning("⚠️ [Flask] 'waitress' not installed. Falling back to Flask development server (NOT recommended for production).")
         try:
             app.run(host=host, port=port)
         except Exception as flask_run_err:
              logger.critical(f"❌ [Flask] Failed to start development server: {flask_run_err}", exc_info=True)
    except Exception as serve_err:
         logger.critical(f"❌ [Flask] Failed to start server (waitress?): {serve_err}", exc_info=True)

# ---------------------- الحلقة الرئيسية ودالة الفحص ----------------------
def main_loop() -> None:
    """الحلقة الرئيسية لفحص الأزواج وتوليد الإشارات."""
    symbols_to_scan = get_crypto_symbols()
    if not symbols_to_scan:
        logger.critical("❌ [Main] لم يتم تحميل أو التحقق من أي رموز صالحة. لا يمكن المتابعة.")
        return

    logger.info(f"✅ [Main] تم تحميل {len(symbols_to_scan)} رمز صالح للفحص.")
    last_full_scan_time = time.time()

    while True:
        try:
            scan_start_time = time.time()
            logger.info("+" + "-"*60 + "+")
            logger.info(f"🔄 [Main] بدء دورة فحص السوق - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("+" + "-"*60 + "+")

            if not check_db_connection() or not conn:
                logger.error("❌ [Main] تخطي دورة الفحص بسبب فشل الاتصال بقاعدة البيانات.")
                time.sleep(60)
                continue

            # 1. التحقق من عدد الإشارات المفتوحة حاليًا
            open_count = 0
            try:
                 with conn.cursor() as cur_check:
                    cur_check.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;")
                    open_count = (cur_check.fetchone() or {}).get('count', 0)
            except psycopg2.Error as db_err:
                 logger.error(f"❌ [Main] خطأ DB أثناء التحقق من عدد الإشارات المفتوحة: {db_err}. تخطي الدورة.")
                 if conn: conn.rollback()
                 time.sleep(60)
                 continue

            logger.info(f"ℹ️ [Main] الإشارات المفتوحة حاليًا: {open_count} / {MAX_OPEN_TRADES}")
            if open_count >= MAX_OPEN_TRADES:
                logger.info(f"⚠️ [Main] تم الوصول للحد الأقصى للإشارات المفتوحة. الانتظار...")
                time.sleep(60)
                continue

            # 2. المرور على قائمة الرموز وفحصها
            processed_in_loop = 0
            signals_generated_in_loop = 0
            slots_available = MAX_OPEN_TRADES - open_count

            for symbol in symbols_to_scan:
                 if slots_available <= 0:
                      logger.info(f"ℹ️ [Main] تم الوصول للحد الأقصى ({MAX_OPEN_TRADES}) أثناء الفحص. إيقاف فحص الرموز لهذه الدورة.")
                      break

                 processed_in_loop += 1
                 logger.debug(f"🔍 [Main] فحص {symbol} ({processed_in_loop}/{len(symbols_to_scan)})...")

                 try:
                    # أ. التحقق من وجود إشارة مفتوحة بالفعل لهذا الرمز
                    with conn.cursor() as symbol_cur:
                        symbol_cur.execute("SELECT 1 FROM signals WHERE symbol = %s AND achieved_target = FALSE AND hit_stop_loss = FALSE LIMIT 1;", (symbol,))
                        if symbol_cur.fetchone():
                            continue

                    # ب. جلب البيانات التاريخية
                    df_hist = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
                    if df_hist is None or df_hist.empty:
                        continue

                    # ج. تطبيق الاستراتيجية وتوليد الإشارة
                    strategy = ConservativeTradingStrategy(symbol) # استخدام الاستراتيجية المعدلة
                    df_indicators = strategy.populate_indicators(df_hist)
                    if df_indicators is None:
                        continue

                    potential_signal = strategy.generate_buy_signal(df_indicators)

                    # د. إدراج الإشارة وإرسال التنبيه
                    if potential_signal:
                        logger.info(f"✨ [Main] تم العثور على إشارة محتملة لـ {symbol}! (Score: {potential_signal.get('r2_score', 0):.2f}) التحقق النهائي وإدراج...")
                        with conn.cursor() as final_check_cur:
                             final_check_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;")
                             final_open_count = (final_check_cur.fetchone() or {}).get('count', 0)

                             if final_open_count < MAX_OPEN_TRADES:
                                 if insert_signal_into_db(potential_signal):
                                     send_telegram_alert(potential_signal, SIGNAL_GENERATION_TIMEFRAME)
                                     signals_generated_in_loop += 1
                                     slots_available -= 1
                                     time.sleep(2)
                                 else:
                                     logger.error(f"❌ [Main] فشل إدراج الإشارة لـ {symbol} في قاعدة البيانات.")
                             else:
                                 logger.warning(f"⚠️ [Main] تم الوصول للحد الأقصى ({final_open_count}) قبل إدراج إشارة {symbol}. تم تجاهل الإشارة.")
                                 break

                 except psycopg2.Error as db_loop_err:
                      logger.error(f"❌ [Main] خطأ DB أثناء معالجة الرمز {symbol}: {db_loop_err}. الانتقال للتالي...")
                      if conn: conn.rollback()
                      continue
                 except Exception as symbol_proc_err:
                      logger.error(f"❌ [Main] خطأ عام أثناء معالجة الرمز {symbol}: {symbol_proc_err}", exc_info=True)
                      continue

                 time.sleep(0.3)

            # 3. انتظار قبل بدء الدورة التالية
            scan_duration = time.time() - scan_start_time
            logger.info(f"🏁 [Main] انتهاء دورة الفحص. الإشارات المولدة: {signals_generated_in_loop}. مدة الفحص: {scan_duration:.2f} ثانية.")
            wait_time = max(60, 300 - scan_duration) # انتظار 5 دقائق إجمالاً أو دقيقة على الأقل
            logger.info(f"⏳ [Main] الانتظار {wait_time:.1f} ثانية للدورة التالية...")
            time.sleep(wait_time)

        except KeyboardInterrupt:
             logger.info("🛑 [Main] تم استقبال طلب إيقاف (KeyboardInterrupt). إغلاق...")
             break
        except psycopg2.Error as db_main_err:
             logger.error(f"❌ [Main] خطأ فادح في قاعدة البيانات في الحلقة الرئيسية: {db_main_err}. محاولة إعادة الاتصال...")
             if conn: conn.rollback()
             time.sleep(60)
             try:
                 init_db()
             except Exception as recon_err:
                 logger.critical(f"❌ [Main] فشلت محاولة إعادة الاتصال بقاعدة البيانات: {recon_err}. الخروج...")
                 break
        except Exception as main_err:
            logger.error(f"❌ [Main] خطأ غير متوقع في الحلقة الرئيسية: {main_err}", exc_info=True)
            logger.info("ℹ️ [Main] انتظار 120 ثانية قبل إعادة المحاولة...")
            time.sleep(120)

def cleanup_resources() -> None:
    """إغلاق الموارد المستخدمة مثل اتصال قاعدة البيانات."""
    global conn
    logger.info("ℹ️ [Cleanup] إغلاق الموارد...")
    if conn:
        try:
            conn.close()
            logger.info("✅ [DB] تم إغلاق اتصال قاعدة البيانات.")
        except Exception as close_err:
            logger.error(f"⚠️ [DB] خطأ أثناء إغلاق اتصال قاعدة البيانات: {close_err}")
    logger.info("✅ [Cleanup] تم الانتهاء من تنظيف الموارد.")


# ---------------------- نقطة الدخول الرئيسية ----------------------
if __name__ == "__main__":
    logger.info("🚀 بدء تشغيل بوت إشارات التداول...")
    logger.info(f"Local Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | UTC Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

    # تهيئة Threads لتكون متاحة كمتغيرات عامة
    ws_thread: Optional[Thread] = None
    tracker_thread: Optional[Thread] = None
    flask_thread: Optional[Thread] = None

    try:
        # 1. تهيئة قاعدة البيانات أولاً
        init_db()

        # 2. بدء WebSocket Ticker
        ws_thread = Thread(target=run_ticker_socket_manager, daemon=True, name="WebSocketThread")
        ws_thread.start()
        logger.info("✅ [Main] تم بدء خيط WebSocket Ticker.")
        logger.info("ℹ️ [Main] الانتظار 5 ثوانٍ لتهيئة WebSocket...")
        time.sleep(5)
        if not ticker_data:
             logger.warning("⚠️ [Main] لم يتم استلام بيانات أولية من WebSocket بعد 5 ثوانٍ.")
        else:
             logger.info(f"✅ [Main] تم استلام بيانات أولية من WebSocket لـ {len(ticker_data)} رمز.")


        # 3. بدء متتبع الإشارات
        tracker_thread = Thread(target=track_signals, daemon=True, name="TrackerThread")
        tracker_thread.start()
        logger.info("✅ [Main] تم بدء خيط تتبع الإشارات.")

        # 4. بدء خادم Flask (إذا تم تكوين Webhook)
        if WEBHOOK_URL:
            flask_thread = Thread(target=run_flask, daemon=True, name="FlaskThread")
            flask_thread.start()
            logger.info("✅ [Main] تم بدء خيط Flask Webhook.")
        else:
             logger.info("ℹ️ [Main] لم يتم تكوين Webhook URL، لن يتم بدء خادم Flask.")

        # 5. بدء الحلقة الرئيسية
        main_loop()

    except Exception as startup_err:
        logger.critical(f"❌ [Main] حدث خطأ فادح أثناء بدء التشغيل أو في الحلقة الرئيسية: {startup_err}", exc_info=True)
    finally:
        logger.info("🛑 [Main] البرنامج في طور الإغلاق...")
        # send_telegram_message(CHAT_ID, "⚠️ تنبيه: بوت التداول قيد الإيقاف الآن.") # يمكن إلغاء التعليق لإرسال تنبيه عند الإيقاف
        cleanup_resources()
        logger.info("👋 [Main] تم إيقاف بوت إشارات التداول.")
        os._exit(0)
