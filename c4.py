#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os
import json
import logging
import requests
import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql # لاستخدام استعلامات آمنة
from psycopg2.extras import RealDictCursor # للحصول على النتائج كقواميس
from binance.client import Client
from binance import ThreadedWebsocketManager
from flask import Flask, request, Response
from threading import Thread
from datetime import datetime, timedelta
from decouple import config
# from apscheduler.schedulers.background import BackgroundScheduler # تم تعليقه - غير مستخدم حالياً

# ---------------------- إعداد التسجيل ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot_elliott_fib.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------- تحميل المتغيرات البيئية ----------------------
try:
    api_key = config('BINANCE_API_KEY')
    api_secret = config('BINANCE_API_SECRET')
    telegram_token = config('TELEGRAM_BOT_TOKEN')
    chat_id = config('TELEGRAM_CHAT_ID')
    db_url = config('DATABASE_URL')
    webhook_url = config('WEBHOOK_URL', default=None) # عنوان الـ webhook سيتم تدوينه في ملف env (اختياري الآن)
except Exception as e:
    logger.critical(f"❌ فشل في تحميل المتغيرات البيئية: {e}")
    exit()

logger.info(f"مفتاح Binance API: {'موجود' if api_key else 'غير موجود'}")
logger.info(f"توكن تليجرام: {telegram_token[:10]}...{'*' * (len(telegram_token)-10)}")
logger.info(f"معرف دردشة تليجرام: {chat_id}")
logger.info(f"رابط قاعدة البيانات: {'موجود' if db_url else 'غير موجود'}")
logger.info(f"عنوان Webhook: {webhook_url if webhook_url else 'غير محدد'}")

# ---------------------- إعداد الثوابت ----------------------
TRADE_VALUE = 10         # قيمة الصفقة الافتراضية بالدولار
MAX_OPEN_TRADES = 4      # الحد الأقصى للصفقات المفتوحة في نفس الوقت
SIGNAL_GENERATION_TIMEFRAME = '30m' # الإطار الزمني لتوليد الإشارة
SIGNAL_GENERATION_LOOKBACK_DAYS = 5 # عدد الأيام للبيانات التاريخية لتوليد الإشارة
SIGNAL_TRACKING_TIMEFRAME = '30m' # الإطار الزمني لتتبع الإشارة وتحديث وقف الخسارة
SIGNAL_TRACKING_LOOKBACK_DAYS = 5   # عدد الأيام للبيانات التاريخية لتتبع الإشارة

# نطاقات RSI
RSI_PERIOD = 14          # فترة RSI
RSI_OVERSOLD = 30        # حد التشبع البيعي
RSI_OVERBOUGHT = 70      # حد التشبع الشرائي

EMA_PERIOD = 21          # فترة EMA للترند
SWING_ORDER = 5          # ترتيب تحديد القمم والقيعان (لـ Elliott Wave - غير مستخدم حالياً في منطق الدخول)
FIB_LEVELS_TO_CHECK = [0.382, 0.5, 0.618] # مستويات فيبوناتشي (غير مستخدم حالياً في منطق الدخول)
FIB_TOLERANCE = 0.007     # التسامح عند التحقق من مستويات فيبوناتشي
LOOKBACK_FOR_SWINGS = 100 # عدد الشموع للبحث عن القمم والقيعان

ENTRY_ATR_PERIOD = 14     # فترة ATR
ENTRY_ATR_MULTIPLIER = 1.2 # مضاعف ATR لتحديد الهدف ووقف الخسارة الأولي

# وقف الخسارة المتحرك (القيم المعدلة)
TRAILING_STOP_ACTIVATION_PROFIT_PCT = 0.015 # نسبة الربح لتفعيل الوقف المتحرك (1%)
TRAILING_STOP_ATR_MULTIPLIER = 2.6        # مضاعف ATR للوقف المتحرك (تمت زيادته لإعطاء مساحة أكبر ضد التقلبات)
TRAILING_STOP_MOVE_INCREMENT_PCT = 0.002  # نسبة الزيادة في السعر لتحريك الوقف المتحرك (0.3%)

MIN_PROFIT_MARGIN_PCT = 1.5 # الحد الأدنى لنسبة الربح المستهدف المئوية مقارنة بسعر الدخول
MIN_VOLUME_15M_USDT = 100000 # الحد الأدنى للسيولة في آخر 15 دقيقة بالدولار

# ---------------------- دوال المؤشرات الإضافية ----------------------
def get_fear_greed_index():
    """يجلب مؤشر الخوف والطمع من alternative.me ويترجم التصنيف إلى العربية"""
    # قاموس لترجمة التصنيفات
    classification_translation_ar = {
        "Extreme Fear": "خوف شديد",
        "Fear": "خوف",
        "Neutral": "محايد",
        "Greed": "جشع",
        "Extreme Greed": "جشع شديد",
        # أضف أي تصنيفات أخرى قد تظهر من الـ API هنا
    }
    try:
        response = requests.get("https://api.alternative.me/fng/", timeout=10)
        response.raise_for_status() # Check for HTTP errors
        data = response.json()
        value = int(data["data"][0]["value"])
        classification_en = data["data"][0]["value_classification"]
        # ترجمة التصنيف إلى العربية، استخدم الإنجليزية كبديل إذا لم توجد ترجمة
        classification_ar = classification_translation_ar.get(classification_en, classification_en)
        return f"{value} ({classification_ar})" # استخدام التصنيف العربي
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [Indicators] خطأ في الشبكة عند جلب مؤشر الخوف والطمع: {e}")
        return "N/A (خطأ في الشبكة)" # رسالة خطأ بالعربية
    except (KeyError, IndexError, ValueError) as e:
        logger.error(f"❌ [Indicators] خطأ في تنسيق بيانات مؤشر الخوف والطمع: {e}")
        return "N/A (خطأ في البيانات)" # رسالة خطأ بالعربية
    except Exception as e:
        logger.error(f"❌ [Indicators] خطأ غير متوقع في جلب مؤشر الخوف والطمع: {e}")
        return "N/A (خطأ غير معروف)" # رسالة خطأ بالعربية

def get_btc_trend_4h():
    """
    يحسب ترند البيتكوين على فريم 4 ساعات باستخدام EMA20 وEMA50.
    """
    try:
        df = fetch_historical_data("BTCUSDT", interval=Client.KLINE_INTERVAL_4HOUR, days=9)
        if df is None or df.empty or len(df) < 50: # تأكد من وجود بيانات كافية
            logger.warning("⚠️ [Indicators] بيانات BTC/USDT غير كافية لحساب الترند.")
            return "N/A (بيانات غير كافية)"
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df.dropna(subset=['close'], inplace=True) # إزالة أي NaN في الإغلاق
        if len(df) < 50:
             logger.warning("⚠️ [Indicators] بيانات BTC/USDT غير كافية بعد إزالة NaN.")
             return "N/A (بيانات غير كافية)"

        ema20 = calculate_ema(df['close'], 20).iloc[-1]
        ema50 = calculate_ema(df['close'], 50).iloc[-1]
        current_close = df['close'].iloc[-1]

        if pd.isna(ema20) or pd.isna(ema50) or pd.isna(current_close):
            logger.warning("⚠️ [Indicators] قيم EMA أو السعر الحالي لـ BTC هي NaN.")
            return "N/A (خطأ حسابي)"

        diff_ema20_pct = abs(current_close - ema20) / current_close if current_close > 0 else 0

        if current_close > ema20 and ema20 > ema50:
            trend = "صعود 📈"
        elif current_close < ema20 and ema20 < ema50:
            trend = "هبوط 📉"
        elif diff_ema20_pct < 0.005: # أقل من 0.5% فرق
            trend = "استقرار 🔄"
        else:
            trend = "تذبذب 🔀"
        return trend
    except Exception as e:
        logger.error(f"❌ [Indicators] خطأ في حساب ترند البيتكوين على أربع ساعات: {e}", exc_info=True)
        return "N/A (خطأ)"

# ---------------------- إعداد الاتصال بقاعدة البيانات ----------------------
conn = None
cur = None

def init_db():
    """تهيئة الاتصال بقاعدة البيانات وإنشاء الجداول إذا لم تكن موجودة."""
    global conn, cur
    retries = 5
    delay = 5
    for i in range(retries):
        try:
            logger.info(f"[DB] محاولة الاتصال بقاعدة البيانات (محاولة {i+1}/{retries})...")
            conn = psycopg2.connect(db_url, connect_timeout=10, cursor_factory=RealDictCursor) # Use RealDictCursor
            conn.autocommit = False # التحكم اليدوي بالـ commit/rollback
            cur = conn.cursor()

            # إنشاء جدول الإشارات
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    initial_target DOUBLE PRECISION NOT NULL,
                    initial_stop_loss DOUBLE PRECISION NOT NULL,
                    current_target DOUBLE PRECISION NOT NULL,
                    current_stop_loss DOUBLE PRECISION NOT NULL,
                    r2_score DOUBLE PRECISION, -- قد يعاد تسميته أو استخدامه بشكل مختلف مع الاستراتيجية الجديدة
                    volume_15m DOUBLE PRECISION, -- تمت إضافته لتخزين حجم السيولة عند الإنشاء (اختياري)
                    achieved_target BOOLEAN DEFAULT FALSE,
                    hit_stop_loss BOOLEAN DEFAULT FALSE,
                    closing_price DOUBLE PRECISION,
                    closed_at TIMESTAMP,
                    sent_at TIMESTAMP DEFAULT NOW(),
                    profit_percentage DOUBLE PRECISION,
                    profitable_stop_loss BOOLEAN DEFAULT FALSE, -- هل تم ضرب وقف الخسارة بربح؟
                    is_trailing_active BOOLEAN DEFAULT FALSE,
                    strategy_name TEXT,
                    signal_details JSONB, -- لتخزين تفاصيل إضافية عن الإشارة
                    last_trailing_update_price DOUBLE PRECISION -- آخر سعر تم عنده تحديث الوقف المتحرك
                )
            """)
            conn.commit()
            logger.info("✅ [DB] جدول 'signals' موجود أو تم إنشاؤه.")

            # التحقق وإضافة الأعمدة الناقصة (أكثر قوة)
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'signals' AND table_schema = 'public'")
            existing_columns = {row['column_name'] for row in cur.fetchall()}
            required_columns = {
                # تأكد من وجود جميع الأعمدة المطلوبة هنا، حتى لو كانت موجودة في CREATE TABLE
                "id": "SERIAL PRIMARY KEY", "symbol": "TEXT NOT NULL", "entry_price": "DOUBLE PRECISION NOT NULL",
                "initial_target": "DOUBLE PRECISION NOT NULL", "initial_stop_loss": "DOUBLE PRECISION NOT NULL",
                "current_target": "DOUBLE PRECISION NOT NULL", "current_stop_loss": "DOUBLE PRECISION NOT NULL",
                "r2_score": "DOUBLE PRECISION", "volume_15m": "DOUBLE PRECISION", "achieved_target": "BOOLEAN DEFAULT FALSE",
                "hit_stop_loss": "BOOLEAN DEFAULT FALSE", "closing_price": "DOUBLE PRECISION", "closed_at": "TIMESTAMP",
                "sent_at": "TIMESTAMP DEFAULT NOW()", "profit_percentage": "DOUBLE PRECISION", "profitable_stop_loss": "BOOLEAN DEFAULT FALSE",
                "is_trailing_active": "BOOLEAN DEFAULT FALSE", "strategy_name": "TEXT", "signal_details": "JSONB",
                "last_trailing_update_price": "DOUBLE PRECISION"
            }
            table_changed = False
            for col_name, col_def in required_columns.items():
                 if col_name not in existing_columns:
                    try:
                        # استخراج نوع العمود فقط (تجنب إعادة إضافة PRIMARY KEY أو NOT NULL إذا تم إضافته بالفعل)
                        col_type = col_def.split(" ")[0]
                        # استخدام psycopg2.sql لتمرير أسماء الأعمدة بأمان
                        alter_query = sql.SQL("ALTER TABLE signals ADD COLUMN {} {}").format(
                            sql.Identifier(col_name), sql.SQL(col_type) # لا نمرر DEFAULT أو NOT NULL هنا، يمكن إضافتها بـ ALTER COLUMN SET DEFAULT لاحقًا إذا لزم الأمر
                        )
                        cur.execute(alter_query)
                        conn.commit() # Commit after each ALTER TABLE
                        table_changed = True
                        logger.info(f"✅ [DB] تمت إضافة العمود '{col_name}'.")
                    except psycopg2.Error as db_err:
                        logger.error(f"❌ [DB] خطأ إضافة العمود '{col_name}': {db_err}")
                        conn.rollback() # Rollback on error for this specific column
                        # Decide whether to raise or continue trying other columns
                        # For now, we log and continue

            if table_changed:
                logger.info("✅ [DB] تم تحديث بنية جدول 'signals'.")

            # إنشاء جدول هيمنة السوق (إذا لم يكن موجودًا)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS market_dominance (
                    id SERIAL PRIMARY KEY,
                    recorded_at TIMESTAMP DEFAULT NOW(),
                    btc_dominance DOUBLE PRECISION,
                    eth_dominance DOUBLE PRECISION
                )
            """)
            conn.commit()
            logger.info("✅ [DB] جدول 'market_dominance' موجود أو تم إنشاؤه.")
            return # نجح الاتصال وإنشاء/تحديث الجداول

        except psycopg2.OperationalError as op_err:
             logger.error(f"❌ [DB] خطأ تشغيلي في الاتصال بقاعدة البيانات (المحاولة {i+1}): {op_err}")
             if conn: conn.rollback() # تأكد من التراجع
             if i == retries - 1:
                 logger.critical("❌ [DB] فشلت جميع محاولات الاتصال بقاعدة البيانات.")
                 raise # ارفع الخطأ بعد فشل كل المحاولات
             time.sleep(delay)
        except Exception as e:
            logger.critical(f"❌ [DB] فشل غير متوقع في تهيئة قاعدة البيانات (المحاولة {i+1}): {e}")
            if conn: conn.rollback()
            if i == retries - 1:
                 logger.critical("❌ [DB] فشلت جميع محاولات الاتصال بقاعدة البيانات.")
                 raise
            time.sleep(delay)
    # إذا وصل هنا، فقد فشلت كل المحاولات
    logger.critical("❌ [DB] فشل الاتصال بقاعدة البيانات بعد عدة محاولات.")
    exit()


def check_db_connection():
    """التحقق من حالة الاتصال بقاعدة البيانات وإعادة التهيئة إذا لزم الأمر."""
    global conn, cur
    try:
        # طريقة بسيطة للتحقق: تنفيذ استعلام بسيط
        if conn is None or conn.closed != 0:
             logger.warning("⚠️ [DB] الاتصال مغلق أو غير موجود. إعادة التهيئة...")
             init_db()
        else:
             # التحقق من أن الاتصال لا يزال يعمل
             cur.execute("SELECT 1;")
             cur.fetchone()
             # logger.debug("[DB] الاتصال نشط.") # يمكن إلغاء التعليق للتحقق المتكرر
    except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
        logger.error(f"❌ [DB] فقدان الاتصال بقاعدة البيانات ({e}). إعادة التهيئة...")
        init_db()
    except Exception as e:
        logger.error(f"❌ [DB] خطأ غير متوقع أثناء التحقق من الاتصال: {e}")
        # قد نحتاج إلى محاولة إعادة الاتصال هنا أيضًا
        init_db()

# ---------------------- دالة تحويل قيم numpy إلى بايثون (مصححة لـ NumPy 2.0) ----------------------
def convert_np_values(obj):
    """تحويل أنواع بيانات NumPy إلى أنواع Python الأصلية للتوافق مع JSON و DB."""
    if isinstance(obj, dict):
        return {k: convert_np_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_values(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int_)): # np.int_ قديم لكن لا يزال يعمل في بعض الإصدارات، يمكن إزالته إذا أردت
        return int(obj)
    # --- السطر المصحح ---
    elif isinstance(obj, (np.floating, np.float64)): # استخدام np.float64 بدلاً من np.float_
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif pd.isna(obj):
        return None # تحويل NaN إلى None
    else:
        return obj

# ---------------------- دالة حساب Bollinger Bands ----------------------
def calculate_bollinger_bands(df, window=20, num_std=2):
    """حساب نطاقات بولينجر."""
    df = df.copy() # تجنب SettingWithCopyWarning
    df['bb_middle'] = df['close'].rolling(window=window).mean()
    df['bb_std'] = df['close'].rolling(window=window).std()
    df['bb_upper'] = df['bb_middle'] + num_std * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - num_std * df['bb_std']
    return df

# ---------------------- قراءة قائمة الأزواج والتحقق منها ----------------------
def get_crypto_symbols(filename='crypto_list.txt'):
    """
    قراءة قائمة رموز العملات من ملف نصي، ثم التحقق من صلاحيتها
    وكونها أزواج USDT متاحة للتداول على Binance Spot.
    """
    raw_symbols = []
    try:
        # محاولة تحديد مسار الملف بالنسبة لمجلد السكربت الحالي
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)

        # إذا لم يوجد في مجلد السكربت، جرب المسار الحالي
        if not os.path.exists(file_path):
            file_path = os.path.abspath(filename)
            if os.path.exists(file_path):
                 logger.warning(f"⚠️ [Data] الملف '{filename}' غير موجود في مجلد السكربت. استخدام الملف في المجلد الحالي.")
            else:
                logger.error(f"❌ [Data] الملف '{filename}' غير موجود في مجلد السكربت أو المجلد الحالي.")
                return []

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = [f"{line.strip().upper().replace('USDT', '')}USDT" for line in f if line.strip() and not line.startswith('#')]
        raw_symbols = sorted(list(set(raw_symbols))) # إزالة التكرارات والترتيب
        logger.info(f"ℹ️ [Data] تم قراءة {len(raw_symbols)} رمز مبدئي من '{file_path}'.")

    except FileNotFoundError:
         logger.error(f"❌ [Data] الملف '{filename}' غير موجود.")
         return []
    except Exception as e:
        logger.error(f"❌ [Data] خطأ في قراءة الملف '{filename}': {e}")
        return []

    if not raw_symbols:
        return [] # لا حاجة للمتابعة إذا كانت القائمة فارغة

    # --- التحقق من الرموز مقابل Binance API ---
    try:
        logger.info("ℹ️ [Data] التحقق من صلاحية الرموز وحالة التداول من Binance API...")
        # تأكد من وجود عميل Binance مهيأ
        if client is None:
             logger.error("❌ [Data Validation] عميل Binance غير مهيأ. لا يمكن التحقق من الرموز.")
             return raw_symbols # أو إرجاع قائمة فارغة حسب المنطق المطلوب

        exchange_info = client.get_exchange_info()
        # بناء مجموعة (set) برموز USDT الصالحة للتداول الفوري لتسريع البحث
        valid_trading_usdt_symbols = {
            s['symbol'] for s in exchange_info['symbols']
            if s.get('quoteAsset') == 'USDT' and    # التأكد من أن العملة المقابلة هي USDT
               s.get('status') == 'TRADING' and         # التأكد من أن الحالة هي TRADING
               s.get('isSpotTradingAllowed') is True    # التأكد من أنه مسموح بالتداول الفوري
        }
        logger.info(f"ℹ️ [Data] تم العثور على {len(valid_trading_usdt_symbols)} زوج USDT صالح للتداول الفوري على Binance.")

        # فلترة القائمة المقروءة من الملف بناءً على القائمة الصالحة من Binance
        validated_symbols = [symbol for symbol in raw_symbols if symbol in valid_trading_usdt_symbols]

        removed_count = len(raw_symbols) - len(validated_symbols)
        if removed_count > 0:
            # تسجيل الرموز المحذوفة (اختياري، قد يكون كثيرًا إذا كانت القائمة كبيرة)
            # removed_symbols = set(raw_symbols) - set(validated_symbols)
            # logger.warning(f"⚠️ [Data] الرموز المحذوفة: {', '.join(removed_symbols)}")
            logger.warning(f"⚠️ [Data] تم إزالة {removed_count} رمز غير صالح أو غير متاح للتداول الفوري USDT من القائمة.")

        logger.info(f"✅ [Data] تم التحقق من الرموز. سيتم استخدام {len(validated_symbols)} رمز صالح.")
        return validated_symbols

    except requests.exceptions.RequestException as req_err:
         logger.error(f"❌ [Data Validation] خطأ في الشبكة عند جلب معلومات الصرف من Binance: {req_err}")
         logger.warning("⚠️ [Data Validation] سيتم استخدام القائمة الأولية من الملف بدون تحقق Binance.")
         return raw_symbols # إرجاع القائمة غير المفلترة في حالة خطأ API
    except Exception as api_err:
         logger.error(f"❌ [Data Validation] خطأ غير متوقع أثناء التحقق من رموز Binance: {api_err}", exc_info=True)
         logger.warning("⚠️ [Data Validation] سيتم استخدام القائمة الأولية من الملف بدون تحقق Binance.")
         return raw_symbols # إرجاع القائمة غير المفلترة في حالة خطأ API


# ---------------------- إعداد عميل Binance ----------------------
try:
    client = Client(api_key, api_secret)
    client.ping() # التحقق من الاتصال وصحة المفاتيح
    server_time = client.get_server_time()
    logger.info(f"✅ [Binance] تم تهيئة عميل Binance. وقت الخادم: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
except requests.exceptions.RequestException as req_err:
     logger.critical(f"❌ [Binance] خطأ في الشبكة عند الاتصال بـ Binance: {req_err}")
     exit()
except Exception as e: # يمكن تخصيص معالجة لأخطاء Binance المحددة إذا لزم الأمر
    logger.critical(f"❌ [Binance] فشل تهيئة عميل Binance: {e}")
    exit()

# ---------------------- إدارة WebSocket لأسعار Ticker ----------------------
ticker_data = {} # قاموس لتخزين أحدث أسعار الإغلاق للرموز

def handle_ticker_message(msg):
    """معالجة رسائل WebSocket الواردة لأسعار mini-ticker."""
    global ticker_data
    try:
        # أحيانًا تأتي الرسائل كقائمة وأحيانًا ككائن خطأ
        if isinstance(msg, list):
            for ticker_item in msg:
                symbol = ticker_item.get('s')
                price_str = ticker_item.get('c') # سعر الإغلاق الأخير كـ string
                if symbol and 'USDT' in symbol and price_str:
                    try:
                        ticker_data[symbol] = float(price_str)
                    except ValueError:
                         logger.warning(f"⚠️ [WS] قيمة سعر غير صالحة للرمز {symbol}: {price_str}")
        elif isinstance(msg, dict) and msg.get('e') == 'error':
            logger.error(f"❌ [WS] رسالة خطأ من WebSocket: {msg.get('m')}")
    except Exception as e:
        logger.error(f"❌ [WS] خطأ في معالجة رسالة ticker: {e}", exc_info=True)


def run_ticker_socket_manager():
    """تشغيل وإدارة اتصال WebSocket لـ mini-ticker."""
    while True:
        try:
            logger.info("ℹ️ [WS] بدء تشغيل WebSocket لأسعار Ticker...")
            twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
            twm.start() # بدء المدير
            # استخدام start_symbol_miniticker_socket يتطلب قائمة رموز محددة
            # start_miniticker_socket يغطي جميع الرموز
            twm.start_miniticker_socket(callback=handle_ticker_message)
            logger.info("✅ [WS] تم توصيل WebSocket بنجاح.")
            twm.join() # الانتظار حتى يتوقف المدير (عادة بسبب خطأ)
            logger.warning("⚠️ [WS] مدير WebSocket توقف. إعادة التشغيل...")
        except Exception as e:
            logger.error(f"❌ [WS] خطأ فادح في WebSocket Manager: {e}. إعادة التشغيل خلال 15 ثانية...")
        # الانتظار قبل إعادة المحاولة لتجنب استهلاك الموارد
        time.sleep(15)


# ---------------------- دوال المؤشرات الفنية ----------------------
def calculate_ema(series, span):
    """حساب المتوسط المتحرك الأسي (EMA)."""
    if series is None or series.isnull().all() or len(series) < span:
        return pd.Series(index=series.index if series is not None else None, dtype=float)
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi_indicator(df, period=RSI_PERIOD):
    """حساب مؤشر القوة النسبية (RSI) مع معالجة تحذيرات Pandas."""
    df = df.copy() # اعمل على نسخة لتجنب تحذيرات SettingWithCopyWarning بشكل عام
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    # التعامل مع حالة avg_loss == 0 لتجنب القسمة على صفر
    rs = avg_gain / avg_loss.replace(0, np.nan) # استبدل الصفر بـ NaN مؤقتًا
    rsi_series = 100 - (100 / (1 + rs)) # احسب السلسلة أولاً

    # تصحيح الطريقة لمعالجة التحذيرات:
    # 1. استخدم ffill() مباشرة بدلاً من fillna(method='ffill')
    # 2. قم بتعيين السلسلة المعدلة مرة أخرى إلى عمود DataFrame بدلاً من استخدام inplace=True على السلسلة
    rsi_series = rsi_series.ffill() # تطبيق forward fill

    # 3. ملء أي قيم NaN متبقية (عادة في البداية) بالقيمة 50
    rsi_series = rsi_series.fillna(50) # تطبيق fillna بدون inplace

    df['rsi'] = rsi_series # تعيين السلسلة المعالجة بالكامل إلى العمود

    return df

def calculate_atr_indicator(df, period=ENTRY_ATR_PERIOD):
    """حساب مؤشر متوسط المدى الحقيقي (ATR)."""
    df = df.copy()
    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()

    # حساب True Range (TR)
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)

    # حساب ATR باستخدام EMA
    df['atr'] = tr.ewm(span=period, adjust=False).mean() # استخدام span بدلاً من com للحصول على نفس نتيجة TradingView تقريبًا
    return df


def calculate_adx(df, period=14):
    """حساب مؤشر ADX و DI+ و DI-."""
    df_calc = df.copy() # اعمل على نسخة داخل الدالة
    df_calc['high-low'] = df_calc['high'] - df_calc['low']
    df_calc['high-prev_close'] = abs(df_calc['high'] - df_calc['close'].shift(1))
    df_calc['low-prev_close'] = abs(df_calc['low'] - df_calc['close'].shift(1))

    df_calc['tr'] = df_calc[['high-low', 'high-prev_close', 'low-prev_close']].max(axis=1)

    df_calc['up_move'] = df_calc['high'] - df_calc['high'].shift(1)
    df_calc['down_move'] = df_calc['low'].shift(1) - df_calc['low']

    df_calc['+dm'] = np.where((df_calc['up_move'] > df_calc['down_move']) & (df_calc['up_move'] > 0), df_calc['up_move'], 0)
    df_calc['-dm'] = np.where((df_calc['down_move'] > df_calc['up_move']) & (df_calc['down_move'] > 0), df_calc['down_move'], 0)

    # استخدام EMA لحساب القيم الملساء (أكثر شيوعًا وتوافقًا مع TradingView)
    df_calc['tr_smooth'] = df_calc['tr'].ewm(alpha=1/period, adjust=False).mean()
    df_calc['+dm_smooth'] = df_calc['+dm'].ewm(alpha=1/period, adjust=False).mean()
    df_calc['-dm_smooth'] = df_calc['-dm'].ewm(alpha=1/period, adjust=False).mean()

    # تجنب القسمة على صفر
    df_calc['di_plus'] = np.where(df_calc['tr_smooth'] > 0, 100 * df_calc['+dm_smooth'] / df_calc['tr_smooth'], 0)
    df_calc['di_minus'] = np.where(df_calc['tr_smooth'] > 0, 100 * df_calc['-dm_smooth'] / df_calc['tr_smooth'], 0)

    # حساب DX
    di_sum = df_calc['di_plus'] + df_calc['di_minus']
    df_calc['dx'] = np.where(di_sum > 0, 100 * abs(df_calc['di_plus'] - df_calc['di_minus']) / di_sum, 0)

    # حساب ADX باستخدام EMA
    df_calc['adx'] = df_calc['dx'].ewm(alpha=1/period, adjust=False).mean()

    # إرجاع الأعمدة النهائية المطلوبة
    return df_calc['adx'], df_calc['di_plus'], df_calc['di_minus']


def calculate_vwap(df):
    """حساب متوسط السعر المرجح بالحجم (VWAP) - يومي."""
    df = df.copy()
    # إعادة تعيين VWAP لكل يوم جديد
    # التأكد من أن الفهرس هو DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("❌ [VWAP] Index is not DatetimeIndex, cannot extract date.")
        df['vwap'] = np.nan # Return NaN if index is wrong type
        return df

    df['date'] = df.index.date
    # حساب السعر النموذجي والحجم * السعر النموذجي
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_vol'] = df['typical_price'] * df['volume']

    # حساب المجاميع التراكمية ضمن كل يوم
    try:
        df['cum_tp_vol'] = df.groupby('date')['tp_vol'].cumsum()
        df['cum_volume'] = df.groupby('date')['volume'].cumsum()
    except KeyError as e:
        logger.error(f"❌ [VWAP] Error grouping by date (maybe index is not datetime?): {e}")
        df['vwap'] = np.nan
        df.drop(columns=['date', 'typical_price', 'tp_vol'], inplace=True, errors='ignore')
        return df

    # إضافة العمود إلى DataFrame
    df['vwap'] = np.where(df['cum_volume'] > 0, df['cum_tp_vol'] / df['cum_volume'], np.nan)
    # ملء قيم NaN الأولية في بداية كل يوم (إذا لزم الأمر)
    df['vwap'] = df['vwap'].bfill() # Use bfill() instead of fillna(method='bfill')

    # إزالة الأعمدة المساعدة
    df.drop(columns=['date', 'typical_price', 'tp_vol', 'cum_tp_vol', 'cum_volume'], inplace=True, errors='ignore')
    return df # إرجاع DataFrame الكامل مع عمود vwap المضاف

def calculate_obv(df):
    """حساب مؤشر حجم التداول المتوازن (On-Balance Volume - OBV)."""
    df = df.copy()
    obv = [0] * len(df) # ابدأ بـ 0 أو بقيمة أولية إذا لزم الأمر
    # التأكد من أن الأعمدة المطلوبة موجودة وليست كلها NaN
    if 'close' not in df.columns or df['close'].isnull().all() or \
       'volume' not in df.columns or df['volume'].isnull().all():
        logger.warning("⚠️ [OBV] الأعمدة 'close' أو 'volume' مفقودة أو فارغة. لا يمكن حساب OBV.")
        df['obv'] = np.nan
        return df

    # التحقق من أن الأعمدة رقمية
    if not pd.api.types.is_numeric_dtype(df['close']) or not pd.api.types.is_numeric_dtype(df['volume']):
        logger.warning("⚠️ [OBV] الأعمدة 'close' أو 'volume' ليست رقمية. لا يمكن حساب OBV.")
        df['obv'] = np.nan
        return df


    for i in range(1, len(df)):
        # التأكد من أن القيم في الصف الحالي والسابق صالحة
        if pd.isna(df['close'].iloc[i]) or pd.isna(df['close'].iloc[i-1]) or pd.isna(df['volume'].iloc[i]):
            obv[i] = obv[i-1] # الحفاظ على القيمة السابقة في حالة وجود NaN
            continue

        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv[i] = obv[i-1] + df['volume'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv[i] = obv[i-1] - df['volume'].iloc[i]
        else:
            obv[i] = obv[i-1] # يبقى كما هو إذا لم يتغير السعر

    # إضافة العمود إلى DataFrame
    df['obv'] = obv
    return df # إرجاع DataFrame الكامل مع عمود obv المضاف


def calculate_supertrend(df, period=10, multiplier=3):
    """حساب مؤشر SuperTrend."""
    df = df.copy()
    # تأكد من وجود عمود ATR أولاً
    if 'atr' not in df.columns:
         df = calculate_atr_indicator(df, period=period) # استخدم نفس الفترة
    elif df['atr'].isnull().all(): # إذا كان موجودًا ولكنه فارغ
         df = calculate_atr_indicator(df, period=period)

    if 'atr' not in df.columns or df['atr'].isnull().all(): # التحقق مرة أخرى بعد المحاولة
         logger.warning("⚠️ [SuperTrend] لا يمكن حساب SuperTrend بسبب عدم وجود قيم ATR صالحة.")
         df['supertrend'] = np.nan
         df['trend'] = "unknown"
         return df

    # حساب النطاقات العلوية والسفلية الأساسية
    hl2 = (df['high'] + df['low']) / 2
    df['basic_ub'] = hl2 + multiplier * df['atr']
    df['basic_lb'] = hl2 - multiplier * df['atr']

    # حساب النطاقات النهائية
    df['final_ub'] = 0.0
    df['final_lb'] = 0.0
    for i in range(period, len(df)): # ابدأ من فترة ATR لتجنب أخطاء NaN الأولية
        if pd.isna(df['basic_ub'].iloc[i]) or pd.isna(df['basic_lb'].iloc[i]):
             # استخدام .loc لتجنب تحذيرات Chained Assignment
             idx = df.index[i]
             idx_prev = df.index[i-1]
             df.loc[idx, 'final_ub'] = df.loc[idx_prev, 'final_ub'] # حافظ على القيمة السابقة إذا كانت البيانات الحالية NaN
             df.loc[idx, 'final_lb'] = df.loc[idx_prev, 'final_lb']
             continue

        # استخدام .loc هنا أيضًا
        idx = df.index[i]
        idx_prev = df.index[i-1]
        if df.loc[idx, 'basic_ub'] < df.loc[idx_prev, 'final_ub'] or df.loc[idx_prev, 'close'] > df.loc[idx_prev, 'final_ub']:
            df.loc[idx, 'final_ub'] = df.loc[idx, 'basic_ub']
        else:
            df.loc[idx, 'final_ub'] = df.loc[idx_prev, 'final_ub']

        if df.loc[idx, 'basic_lb'] > df.loc[idx_prev, 'final_lb'] or df.loc[idx_prev, 'close'] < df.loc[idx_prev, 'final_lb']:
            df.loc[idx, 'final_lb'] = df.loc[idx, 'basic_lb']
        else:
            df.loc[idx, 'final_lb'] = df.loc[idx_prev, 'final_lb']

    # حساب خط SuperTrend وتحديد الاتجاه
    df['supertrend'] = np.nan
    df['trend'] = "unknown" # قيمة افتراضية
    trend = [] # قائمة لتخزين الاتجاه لكل شمعة

    for i in range(period, len(df)):
        idx = df.index[i]
        idx_prev = df.index[i-1]
        current_close = df.loc[idx, 'close']
        prev_supertrend = df.loc[idx_prev, 'supertrend'] # استخدم .loc
        curr_final_lb = df.loc[idx, 'final_lb']
        curr_final_ub = df.loc[idx, 'final_ub']

        if pd.isna(current_close) or pd.isna(curr_final_lb) or pd.isna(curr_final_ub):
             if i > period and trend:
                 current_trend = trend[-1]
             else:
                 current_trend = "unknown"
             df.loc[idx, 'supertrend'] = prev_supertrend
             trend.append(current_trend)
             df.loc[idx, 'trend'] = current_trend
             continue

        if len(trend) == 0:
             if current_close > curr_final_ub:
                 current_trend = "up"
                 df.loc[idx, 'supertrend'] = curr_final_lb
             else:
                 current_trend = "down"
                 df.loc[idx, 'supertrend'] = curr_final_ub
        else:
             prev_trend = trend[-1]
             if prev_trend == "up":
                 if current_close > curr_final_lb:
                     current_trend = "up"
                     df.loc[idx, 'supertrend'] = max(curr_final_lb, prev_supertrend if not pd.isna(prev_supertrend) else curr_final_lb)
                 else:
                     current_trend = "down"
                     df.loc[idx, 'supertrend'] = curr_final_ub
             elif prev_trend == "down":
                 if current_close < curr_final_ub:
                     current_trend = "down"
                     df.loc[idx, 'supertrend'] = min(curr_final_ub, prev_supertrend if not pd.isna(prev_supertrend) else curr_final_ub)
                 else:
                     current_trend = "up"
                     df.loc[idx, 'supertrend'] = curr_final_lb
             else: # prev_trend == "unknown"
                 if current_close > curr_final_ub:
                     current_trend = "up"
                     df.loc[idx, 'supertrend'] = curr_final_lb
                 else:
                     current_trend = "down"
                     df.loc[idx, 'supertrend'] = curr_final_ub

        trend.append(current_trend)
        df.loc[idx, 'trend'] = current_trend

    # إزالة فقط الأعمدة المساعدة الداخلية لهذه الدالة
    columns_to_drop_supertrend = ['basic_ub', 'basic_lb', 'final_ub', 'final_lb']
    df.drop(columns=columns_to_drop_supertrend, errors='ignore', inplace=True)

    return df # إرجاع DataFrame الكامل

# ---------------------- نماذج الشموع اليابانية ----------------------
def is_hammer(row):
    """التحقق من نموذج المطرقة (إشارة صعودية)."""
    open_price, high, low, close = row['open'], row['high'], row['low'], row['close']
    if None in [open_price, high, low, close] or pd.isna([open_price, high, low, close]).any():
        return 0
    body = abs(close - open_price)
    candle_range = high - low
    if candle_range == 0: return 0 # تجنب القسمة على صفر

    lower_shadow = min(open_price, close) - low
    upper_shadow = high - max(open_price, close)

    # شروط المطرقة: جسم صغير، ظل سفلي طويل (ضعف الجسم على الأقل)، ظل علوي قصير جدًا
    is_small_body = body < (candle_range * 0.3) # الجسم أقل من 30% من المدى
    is_long_lower_shadow = lower_shadow >= 2 * body if body > 0 else lower_shadow > candle_range * 0.6 # ظل سفلي ضعف الجسم أو أكثر من 60% من المدى إذا كان الجسم صغيرًا جدًا
    is_small_upper_shadow = upper_shadow <= body * 0.5 if body > 0 else upper_shadow < candle_range * 0.1 # ظل علوي أقل من نصف الجسم أو أقل من 10% من المدى

    # يجب أن تظهر في اتجاه هابط (يمكن إضافة هذا الشرط إذا لزم الأمر بتحليل الشموع السابقة)
    return 100 if is_small_body and is_long_lower_shadow and is_small_upper_shadow else 0

def is_shooting_star(row):
    """التحقق من نموذج الشهاب (إشارة هبوطية)."""
    open_price, high, low, close = row['open'], row['high'], row['low'], row['close']
    if None in [open_price, high, low, close] or pd.isna([open_price, high, low, close]).any():
        return 0
    body = abs(close - open_price)
    candle_range = high - low
    if candle_range == 0: return 0

    lower_shadow = min(open_price, close) - low
    upper_shadow = high - max(open_price, close)

    # شروط الشهاب: جسم صغير، ظل علوي طويل (ضعف الجسم على الأقل)، ظل سفلي قصير جدًا
    is_small_body = body < (candle_range * 0.3)
    is_long_upper_shadow = upper_shadow >= 2 * body if body > 0 else upper_shadow > candle_range * 0.6
    is_small_lower_shadow = lower_shadow <= body * 0.5 if body > 0 else lower_shadow < candle_range * 0.1

    # يجب أن تظهر في اتجاه صاعد (يمكن إضافة هذا الشرط)
    return -100 if is_small_body and is_long_upper_shadow and is_small_lower_shadow else 0 # إشارة سالبة لأنها هبوطية


def is_doji(row):
    """التحقق من نموذج دوجي (عدم يقين)."""
    open_price, high, low, close = row['open'], row['high'], row['low'], row['close']
    if None in [open_price, high, low, close] or pd.isna([open_price, high, low, close]).any():
        return 0
    candle_range = high - low
    if candle_range == 0: return 0 # إذا لم يكن هناك مدى، لا يمكن أن تكون دوجي بالمعنى التقليدي
    # الجسم صغير جدًا مقارنة بالمدى الكلي
    return 100 if abs(close - open_price) < (candle_range * 0.1) else 0 # الجسم أقل من 10% من المدى


def is_spinning_top(row):
    """التحقق من نموذج القمة الدوارة (عدم يقين)."""
    open_price, high, low, close = row['open'], row['high'], row['low'], row['close']
    if None in [open_price, high, low, close] or pd.isna([open_price, high, low, close]).any():
        return 0
    body = abs(close - open_price)
    candle_range = high - low
    if candle_range == 0 or body == 0: return 0 # لا يمكن أن يكون Spinning Top إذا لم يكن هناك مدى أو جسم

    upper_shadow = high - max(open_price, close)
    lower_shadow = min(open_price, close) - low

    # جسم صغير وظلال علوية وسفلية أكبر من الجسم
    is_small_body = body < (candle_range * 0.3) # جسم صغير
    has_upper_shadow = upper_shadow > body
    has_lower_shadow = lower_shadow > body

    return 100 if is_small_body and has_upper_shadow and has_lower_shadow else 0


def compute_engulfing(df, idx):
    """التحقق من نموذج الابتلاع الصعودي أو الهبوطي."""
    if idx == 0: return 0 # لا يمكن التحقق من أول شمعة
    prev = df.iloc[idx - 1]
    curr = df.iloc[idx]

    # التحقق من صلاحية البيانات
    if pd.isna(prev['close']) or pd.isna(prev['open']) or pd.isna(curr['close']) or pd.isna(curr['open']):
        return 0

    # ابتلاع صعودي: شمعة سابقة هابطة، شمعة حالية صاعدة تبتلع جسم الشمعة السابقة
    is_bullish_engulfing = (prev['close'] < prev['open'] and # شمعة سابقة هابطة
                            curr['close'] > curr['open'] and # شمعة حالية صاعدة
                            curr['open'] <= prev['close'] and # افتتاح الحالية أقل أو يساوي إغلاق السابقة
                            curr['close'] >= prev['open'])   # إغلاق الحالية أكبر أو يساوي افتتاح السابقة

    # ابتلاع هبوطي: شمعة سابقة صاعدة، شمعة حالية هابطة تبتلع جسم الشمعة السابقة
    is_bearish_engulfing = (prev['close'] > prev['open'] and # شمعة سابقة صاعدة
                            curr['close'] < curr['open'] and # شمعة حالية هابطة
                            curr['open'] >= prev['close'] and # افتتاح الحالية أكبر أو يساوي إغلاق السابقة
                            curr['close'] <= prev['open'])   # إغلاق الحالية أقل أو يساوي افتتاح السابقة

    if is_bullish_engulfing: return 100
    if is_bearish_engulfing: return -100
    return 0


def detect_candlestick_patterns(df):
    """تطبيق دوال اكتشاف نماذج الشموع على DataFrame."""
    df = df.copy()
    df['Hammer'] = df.apply(is_hammer, axis=1)
    df['ShootingStar'] = df.apply(is_shooting_star, axis=1)
    df['Doji'] = df.apply(is_doji, axis=1)
    df['SpinningTop'] = df.apply(is_spinning_top, axis=1)

    # حساب الابتلاع يتطلب الوصول للصف السابق، لذا نعالجه بشكل منفصل
    if len(df) > 1:
        # إعادة الفهرسة مؤقتًا لتسهيل الوصول بـ iloc
        df_reset = df.reset_index(drop=True)
        engulfing_values = [compute_engulfing(df_reset, i) for i in range(len(df_reset))]
        # إعادة تعيين الفهرس الأصلي عند تعيين السلسلة الجديدة
        df['Engulfing'] = pd.Series(engulfing_values, index=df.index)
    else:
        df['Engulfing'] = 0 # لا يمكن حساب الابتلاع لشمعة واحدة

    # تجميع إشارات الشموع الصعودية والهبوطية (يمكن تخصيصها أكثر)
    # إشارة صعودية قوية: مطرقة أو ابتلاع صعودي
    df['BullishCandleSignal'] = df.apply(lambda row: 100 if (row['Hammer'] == 100 or row['Engulfing'] == 100) else 0, axis=1)
    # إشارة هبوطية قوية: شهاب أو ابتلاع هبوطي
    df['BearishCandleSignal'] = df.apply(lambda row: 100 if (row['ShootingStar'] == -100 or row['Engulfing'] == -100) else 0, axis=1) # استخدام 100 للإشارة لوجودها

    return df


# ---------------------- دوال MACD وموجات إليوت (للتفاصيل الإضافية) ----------------------
def calculate_macd(df, fast=12, slow=26, signal=9):
    """حساب مؤشر MACD وخط الإشارة والهيستوجرام."""
    df = df.copy()
    df['ema_fast'] = calculate_ema(df['close'], fast)
    df['ema_slow'] = calculate_ema(df['close'], slow)
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = calculate_ema(df['macd'], signal)
    df['macd_hist'] = df['macd'] - df['macd_signal']
    # إزالة الأعمدة المساعدة إذا لم تكن مطلوبة في مكان آخر
    df.drop(columns=['ema_fast', 'ema_slow'], inplace=True, errors='ignore')
    return df

# دوال Elliott Wave (detect_swings, detect_elliott_waves) تبقى كما هي لأنها تستخدم فقط لإضافة تفاصيل ولا تؤثر على منطق الدخول الرئيسي.
def detect_swings(prices, order=5):
    """اكتشاف نقاط التأرجح (القمم والقيعان) في سلسلة زمنية."""
    maxima_indices = []
    minima_indices = []
    n = len(prices)
    if n < 2 * order + 1: return [], [] # بيانات غير كافية

    # تأكد من أن prices هو numpy array
    if not isinstance(prices, np.ndarray): prices = np.array(prices)

    for i in range(order, n - order):
        # التأكد من أن الفهرس ضمن الحدود قبل الوصول إلى window
        if i - order < 0 or i + order + 1 > n: continue

        window = prices[i - order: i + order + 1]
        center = prices[i]

        # التحقق من أن النافذة لا تحتوي على NaN وأن center ليس NaN
        if np.isnan(window).any() or np.isnan(center): continue

        # Check if the center value is the maximum in the window
        if center == np.max(window) and np.argmax(window) == order:
            if not maxima_indices or i > maxima_indices[-1] + order:
                maxima_indices.append(i)

        # Check if the center value is the minimum in the window
        if center == np.min(window) and np.argmin(window) == order:
            if not minima_indices or i > minima_indices[-1] + order:
                minima_indices.append(i)

    maxima = [(idx, prices[idx]) for idx in maxima_indices]
    minima = [(idx, prices[idx]) for idx in minima_indices]
    return maxima, minima

def detect_elliott_waves(df, order=SWING_ORDER):
    """محاولة بسيطة لتحديد موجات إليوت بناءً على تأرجحات هيستوجرام MACD."""
    if 'macd_hist' not in df.columns or df['macd_hist'].isnull().all():
        logger.warning("⚠️ [Elliott] عمود 'macd_hist' غير موجود أو فارغ لحساب موجات إليوت.")
        return []

    macd_values = df['macd_hist'].values
    maxima, minima = detect_swings(macd_values, order=order)

    # دمج وترتيب جميع نقاط التأرجح
    all_swings = sorted(
        [(idx, val, 'max') for idx, val in maxima] +
        [(idx, val, 'min') for idx, val in minima],
        key=lambda x: x[0] # الترتيب حسب المؤشر (الزمن)
    )

    waves = []
    wave_number = 1
    # المنطق هنا لتصنيف الموجات يمكن أن يكون أكثر تعقيدًا ويتطلب قواعد إليوت القياسية
    # هذا التنفيذ المبسط يحدد فقط نقاط التأرجح وتصنيفًا أوليًا (اندفاع/تصحيح)
    for idx, val, typ in all_swings:
        # Ensure index is within bounds of df.index
        if idx < 0 or idx >= len(df.index): continue

        # التصنيف بسيط جدًا هنا، قد لا يتبع قواعد إليوت بدقة
        wave_type = "Impulse" if (typ == 'max' and val > 0) or (typ == 'min' and val >= 0) else "Correction"
        waves.append({
            "wave": wave_number,
            "timestamp": str(df.index[idx]), # استخدام الفهرس الأصلي للـ DataFrame
            "macd_hist_value": float(val), # قيمة هيستوجرام MACD عند التأرجح
            "swing_type": typ, # 'max' or 'min'
            "classified_type": wave_type # التصنيف الأولي
        })
        wave_number += 1
    return waves


# ---------------------- دالة لجلب السيولة لآخر 15 دقيقة ----------------------
def fetch_recent_volume(symbol):
    """جلب حجم التداول بالـ USDT لآخر 15 دقيقة للرمز المحدد."""
    try:
        # جلب بيانات الدقيقة الواحدة لآخر 15 دقيقة
        klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=15)
        if not klines or len(klines) < 15:
            logger.warning(f"⚠️ [Data] بيانات 1m غير كافية (أقل من 15 شمعة) للزوج {symbol} لحساب السيولة.")
            return 0.0

        # حجم التداول بالعملة المقابلة (Quote Asset Volume) هو الحقل الثامن (index 7)
        volume_usdt = sum(float(k[7]) for k in klines if len(k) > 7 and k[7])
        # logger.debug(f"ℹ️ [Data] السيولة آخر 15 دقيقة للزوج {symbol}: {volume_usdt:.2f} USDT")
        return volume_usdt
    except Exception as e:
        logger.error(f"❌ [Data] خطأ في جلب السيولة للزوج {symbol}: {e}")
        return 0.0


# ---------------------- دالة توليد تقرير الأداء الشامل (محسّنة) ----------------------
def generate_performance_report():
    """توليد تقرير أداء شامل ومفصل من قاعدة البيانات."""
    try:
        check_db_connection()
        with conn.cursor() as report_cur: # يستخدم RealDictCursor المحدد في init_db
            # 1. الإشارات المفتوحة
            report_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;")
            open_signals_count = report_cur.fetchone()['count'] or 0

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
            closed_stats = report_cur.fetchone()

            total_closed = closed_stats['total_closed'] or 0
            winning_signals = closed_stats['winning_signals'] or 0
            losing_signals = closed_stats['losing_signals'] or 0
            # neutral_signals = closed_stats['neutral_signals'] or 0 # يمكن إضافتها إذا كانت مهمة
            total_profit_pct = closed_stats['total_profit_pct'] or 0.0
            gross_profit_pct = closed_stats['gross_profit_pct'] or 0.0
            gross_loss_pct = closed_stats['gross_loss_pct'] or 0.0 # ستكون سالبة أو صفر
            avg_win_pct = closed_stats['avg_win_pct'] or 0.0
            avg_loss_pct = closed_stats['avg_loss_pct'] or 0.0 # ستكون سالبة أو صفر

            # 3. حساب المقاييس المشتقة
            win_rate = (winning_signals / total_closed * 100) if total_closed > 0 else 0.0
            # Profit Factor: Total Profit / Absolute Total Loss
            profit_factor = (gross_profit_pct / abs(gross_loss_pct)) if gross_loss_pct != 0 else float('inf') # تعني أرباح لا نهائية إذا لم تكن هناك خسائر

        # 4. تنسيق التقرير
        report = (
            "📊 *تقرير الأداء الشامل:*\n"
            "——————————————\n"
            f"📈 الإشارات المفتوحة حاليًا: {open_signals_count}\n"
            "——————————————\n"
            "📉 *إحصائيات الإشارات المغلقة:*\n"
            f" * إجمالي الإشارات المغلقة: {total_closed}\n"
            f" ✅ إشارات رابحة: {winning_signals}\n"
            f" ❌ إشارات خاسرة: {losing_signals}\n"
            f" * معدل الربح (Win Rate): {win_rate:.2f}%\n"
            "——————————————\n"
            "💰 *الربحية:*\n"
            f" * صافي الربح/الخسارة (إجمالي %): {total_profit_pct:+.2f}%\n"
            f" * إجمالي ربح (%): {gross_profit_pct:+.2f}%\n"
            f" * إجمالي خسارة (%): {gross_loss_pct:.2f}%\n"
            f" * متوسط ربح الصفقة الرابحة: {avg_win_pct:+.2f}%\n"
            f" * متوسط خسارة الصفقة الخاسرة: {avg_loss_pct:.2f}%\n"
            f" * معامل الربح (Profit Factor): {'∞' if profit_factor == float('inf') else f'{profit_factor:.2f}'}\n"
            "——————————————\n"
            f"🕰️ _التقرير حتى: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        )
        return report

    except psycopg2.Error as db_err:
        logger.error(f"❌ [Report] خطأ في قاعدة البيانات عند توليد تقرير الأداء: {db_err}")
        conn.rollback() # تراجع عن أي معاملة قد تكون مفتوحة
        return "❌ خطأ في قاعدة البيانات عند توليد تقرير الأداء."
    except Exception as e:
        logger.error(f"❌ [Report] خطأ غير متوقع في توليد تقرير الأداء: {e}", exc_info=True)
        return "❌ خطأ غير متوقع في توليد تقرير الأداء."

# ---------------------- استراتيجية التداول المحافظة (المعدلة) ----------------------
class ElliottFibCandleStrategy:
    def __init__(self):
        pass

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """حساب جميع المؤشرات المطلوبة للاستراتيجية."""
        min_len_required = max(EMA_PERIOD, RSI_PERIOD, ENTRY_ATR_PERIOD, 14, 10, SWING_ORDER * 2 + 1, LOOKBACK_FOR_SWINGS)
        if len(df) < min_len_required:
            logger.warning(f"⚠️ [Strategy] DataFrame قصير جدًا ({len(df)} شمعة، مطلوب {min_len_required}) لحساب المؤشرات.")
            return pd.DataFrame()

        try:
            df = df.copy()
            # ---- تسلسل حساب المؤشرات مهم ----
            df['ema_trend'] = calculate_ema(df['close'], EMA_PERIOD)
            df = calculate_rsi_indicator(df, RSI_PERIOD) # <-- تم إصلاحه
            df = calculate_atr_indicator(df, ENTRY_ATR_PERIOD)
            df = calculate_bollinger_bands(df)
            df = calculate_macd(df)

            # حساب ADX وتعيين الأعمدة
            adx_val, di_plus_val, di_minus_val = calculate_adx(df.copy(), period=14) # Use copy to avoid modifying df inside calculate_adx if it does
            df['adx'] = adx_val
            df['di_plus'] = di_plus_val
            df['di_minus'] = di_minus_val


            # حساب VWAP و OBV (تأكد من أنها تعيد DataFrame)
            df = calculate_vwap(df)   # <-- تم إصلاحه
            df = calculate_obv(df)    # <-- تم إصلاحه

            # حساب SuperTrend (تأكد من أنها تعيد DataFrame ولا تحذف الأعمدة المطلوبة)
            df = calculate_supertrend(df, period=10, multiplier=3) # <-- تم إصلاحه

            # حساب نماذج الشموع
            df = detect_candlestick_patterns(df)


            # --- التعامل مع NaN بعد حساب *كل* المؤشرات ---
            initial_len = len(df)
            required_indicator_cols = [
                'ema_trend', 'rsi', 'atr', 'bb_upper', 'bb_lower',
                'macd', 'macd_signal', 'macd_hist', # macd_hist is used by detect_elliott_waves
                'adx', 'di_plus', 'di_minus', # الآن يجب أن تكون موجودة
                'vwap', 'obv', 'trend', 'supertrend'
            ]
            # تحقق أولاً من وجود الأعمدة قبل محاولة dropna
            missing_cols_final = [col for col in required_indicator_cols if col not in df.columns]
            if missing_cols_final:
                 logger.error(f"❌ [Strategy] أعمدة مطلوبة لا تزال مفقودة قبل dropna: {missing_cols_final}")
                 # Log columns that *are* present for debugging
                 logger.debug(f"Columns present: {df.columns.tolist()}")
                 return pd.DataFrame() # فشل حاسم

            # استخدام dropna مع inplace=False (أكثر أمانًا)
            df_cleaned = df.dropna(subset=required_indicator_cols).copy() # Add .copy() here
            dropped_count = initial_len - len(df_cleaned)

            if dropped_count > 0:
                logger.debug(f"ℹ️ [Strategy] تم حذف {dropped_count} صف بسبب NaN في المؤشرات الأساسية.")
            if df_cleaned.empty:
                logger.warning("⚠️ [Strategy] DataFrame فارغ بعد إزالة NaN من المؤشرات الأساسية.")
                return pd.DataFrame()

            # إرجاع DataFrame النظيف
            df = df_cleaned # Reassign df to the cleaned version
            latest = df.iloc[-1]
            logger.info(f"✅ [Strategy] تم حساب المؤشرات بنجاح. آخر اتجاه SuperTrend: {latest.get('trend', 'N/A')}")
            return df

        except KeyError as ke:
             logger.error(f"❌ [Strategy] خطأ: العمود المطلوب غير موجود أثناء حساب المؤشرات: {ke}", exc_info=True)
             return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ [Strategy] خطأ غير متوقع أثناء حساب المؤشرات: {e}", exc_info=True)
            return pd.DataFrame()

    def populate_buy_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """تحديد إشارات الشراء بناءً على الاستراتيجية المحافظة."""
        # الأعمدة المطلوبة لاتخاذ قرار الشراء
        required_cols = [
            'close', 'ema_trend', 'rsi', 'BullishCandleSignal', 'atr', 'macd', 'macd_signal',
            'trend', 'adx', 'di_plus', 'di_minus', 'vwap', 'bb_upper', 'obv'
         ]

        # التحقق الأولي من الـ DataFrame والأعمدة المطلوبة
        if df.empty:
             logger.warning("⚠️ [Strategy Buy] DataFrame فارغ، لا يمكن حساب إشارة الشراء.")
             df['buy'] = 0
             df['buy_signal_score'] = 0.0
             df['signal_details_json'] = None
             return df
        if not all(col in df.columns for col in required_cols):
             missing_cols = [col for col in required_cols if col not in df.columns]
             logger.warning(f"⚠️ [Strategy Buy] DataFrame يفتقد أعمدة مطلوبة: {missing_cols}. لا يمكن حساب إشارة الشراء.")
             df['buy'] = 0
             df['buy_signal_score'] = 0.0
             df['signal_details_json'] = None
             return df
        # التحقق من وجود NaN في الصف الأخير للأعمدة المطلوبة
        last_row_check = df.iloc[-1][required_cols]
        if last_row_check.isnull().any():
            nan_cols = last_row_check[last_row_check.isnull()].index.tolist()
            logger.warning(f"⚠️ [Strategy Buy] الصف الأخير يحتوي على NaN في أعمدة مطلوبة: {nan_cols}. لا يمكن حساب إشارة الشراء.")
            # Assign default values to the last row to avoid errors, but no signal will be generated.
            last_idx = df.index[-1]
            df.loc[last_idx, 'buy'] = 0
            df.loc[last_idx, 'buy_signal_score'] = 0.0
            df.loc[last_idx, 'signal_details_json'] = None
            # Ensure columns exist before returning
            if 'buy' not in df.columns: df['buy'] = 0
            if 'buy_signal_score' not in df.columns: df['buy_signal_score'] = 0.0
            if 'signal_details_json' not in df.columns: df['signal_details_json'] = None
            return df


        # إعداد قيم أولية للأعمدة الجديدة إذا لم تكن موجودة
        if 'buy' not in df.columns: df['buy'] = 0
        if 'buy_signal_score' not in df.columns: df['buy_signal_score'] = 0.0
        if 'signal_details_json' not in df.columns: df['signal_details_json'] = None

        # العمل على نسخة لتجنب التحذيرات عند التعيين
        df = df.copy()

        # الحصول على بيانات آخر شمعة مكتملة
        last_idx = df.index[-1]
        last_row = df.loc[last_idx]
        signal_details = {}
        conditions_met_count = 0

        # --- تعريف وتحقق من شروط الشراء المحافظة ---

        # 1. تأكيد الاتجاه الصاعد (EMA + SuperTrend + VWAP)
        cond_ema_up = last_row['close'] > last_row['ema_trend']
        cond_supertrend_up = last_row['trend'] == 'up'
        cond_above_vwap = last_row['close'] > last_row['vwap']
        is_uptrend_confirmed = cond_ema_up and cond_supertrend_up and cond_above_vwap
        if is_uptrend_confirmed:
            conditions_met_count += 3
            signal_details['Trend'] = 'Confirmed Up (EMA, Supertrend, VWAP)'

        # 2. تأكيد الزخم الإيجابي (MACD + ADX/DI)
        cond_macd_bullish = last_row['macd'] > last_row['macd_signal']
        cond_adx_trending_bullish = last_row['adx'] > 20 and last_row['di_plus'] > last_row['di_minus']
        is_momentum_confirmed = cond_macd_bullish and cond_adx_trending_bullish
        if is_momentum_confirmed:
            conditions_met_count += 2
            signal_details['Momentum'] = 'Confirmed Bullish (MACD, ADX/DI)'

        # 3. مؤشر القوة النسبية (RSI) في منطقة صحية
        cond_rsi_ok = last_row['rsi'] < RSI_OVERBOUGHT and last_row['rsi'] > 40
        if cond_rsi_ok:
            conditions_met_count += 1
            signal_details['RSI'] = f'OK ({last_row["rsi"]:.1f})'

        # 4. تأكيد من نموذج شمعة إيجابي (اختياري لكن مقوي)
        cond_bullish_candle = last_row['BullishCandleSignal'] == 100
        if cond_bullish_candle:
            conditions_met_count += 1
            signal_details['Candle'] = 'Bullish Pattern'

        # 5. السعر ليس عند قمة متطرفة (بالنسبة لـ Bollinger Bands)
        cond_not_bb_extreme = last_row['close'] < last_row['bb_upper']
        if cond_not_bb_extreme:
            conditions_met_count += 1
            signal_details['BB'] = 'Not Extreme High'

        # --- قرار الشراء النهائي ---
        buy_signal_triggered = False
        MIN_CONDITIONS_FOR_SIGNAL = 7 # مثال: تتطلب 6 شروط على الأقل بما فيها الاتجاه والزخم
        # الشرط الأساسي: اتجاه وزخم إيجابي + RSI مقبول + ليس عند قمة BB
        core_conditions_met = is_uptrend_confirmed and is_momentum_confirmed and cond_rsi_ok and cond_not_bb_extreme

        if core_conditions_met and conditions_met_count >= MIN_CONDITIONS_FOR_SIGNAL :
             buy_signal_triggered = True
             if cond_bullish_candle:
                 signal_details['Strength'] = 'Very Strong (Core Conditions + Candle)'
             else:
                 signal_details['Strength'] = 'Strong (Core Conditions Met)'

        # تحديث آخر صف في DataFrame بالنتيجة باستخدام .loc
        final_buy_signal = 1 if buy_signal_triggered else 0
        final_score = float(conditions_met_count)

        df.loc[last_idx, 'buy'] = final_buy_signal
        df.loc[last_idx, 'buy_signal_score'] = final_score
        if buy_signal_triggered:
            try:
                # التأكد من تحويل القيم قبل JSON dump
                details_converted = convert_np_values(signal_details)
                df.loc[last_idx, 'signal_details_json'] = json.dumps(details_converted)
                logger.info(f"✅ [Strategy Buy] {last_idx} - إشارة شراء محافظة (Score: {final_score}). التفاصيل: {details_converted}")
            except TypeError as json_err:
                 logger.error(f"❌ [Strategy Buy] خطأ تحويل تفاصيل الإشارة إلى JSON: {json_err} - Details: {signal_details}")
                 df.loc[last_idx, 'signal_details_json'] = json.dumps({'error': 'serialization_failed'}) # وضع علامة خطأ
        else:
             df.loc[last_idx, 'signal_details_json'] = None # لا توجد تفاصيل إذا لم تكن هناك إشارة


        return df

# ---------------------- دالة جلب البيانات التاريخية ----------------------
def fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS):
    """جلب البيانات التاريخية للشموع من Binance."""
    try:
        start_dt = datetime.utcnow() - timedelta(days=days + 1)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")

        # logger.debug(f"ℹ️ [Data] جلب بيانات {interval} للزوج {symbol} منذ {start_str}...")
        klines = client.get_historical_klines(symbol, interval, start_str, limit=1000)

        if not klines:
            # logger.warning(f"⚠️ [Data] لا توجد بيانات تاريخية ({interval}) للزوج {symbol} للفترة المطلوبة.")
            return None

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])

        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        df = df[['open', 'high', 'low', 'close', 'volume']]

        initial_len = len(df)
        df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True) # Include volume here
        if len(df) < initial_len:
            logger.debug(f"ℹ️ [Data] تم حذف {initial_len - len(df)} صف بسبب NaN في بيانات OHLCV للزوج {symbol}.")

        if df.empty:
            # logger.warning(f"⚠️ [Data] DataFrame للزوج {symbol} فارغ بعد إزالة NaN الأساسية.")
            return None

        # logger.info(f"✅ [Data] تم جلب ومعالجة {len(df)} شمعة تاريخية ({interval}) للزوج {symbol}.") # يمكن تقليل هذا اللوغ
        return df

    except Exception as e:
        logger.error(f"❌ [Data] خطأ في جلب البيانات التاريخية للزوج {symbol}: {e}", exc_info=True)
        return None

# ---------------------- دوال Telegram ----------------------
def send_telegram_message(chat_id_target, text, reply_markup=None, parse_mode='Markdown', disable_web_page_preview=True, timeout=20):
    """إرسال رسالة عبر Telegram Bot API."""
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {
        'chat_id': str(chat_id_target),
        'text': text,
        'parse_mode': parse_mode,
        'disable_web_page_preview': disable_web_page_preview
    }
    if reply_markup:
        payload['reply_markup'] = json.dumps(reply_markup)

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        logger.info(f"✅ [Telegram] تم إرسال رسالة إلى {chat_id_target}.")
        return response.json()
    except requests.exceptions.Timeout:
         logger.error(f"❌ [Telegram] فشل إرسال الرسالة إلى {chat_id_target} (Timeout).")
         return None
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ [Telegram] فشل إرسال الرسالة إلى {chat_id_target}: {e}")
        if e.response is not None:
             try:
                 error_details = e.response.json()
                 logger.error(f"❌ [Telegram] تفاصيل خطأ API: {error_details}")
             except json.JSONDecodeError:
                 logger.error(f"❌ [Telegram] لم يتمكن من فك تشفير استجابة الخطأ: {e.response.text}")
        return None
    except Exception as e:
         logger.error(f"❌ [Telegram] خطأ غير متوقع أثناء إرسال الرسالة: {e}")
         return None


def send_telegram_alert(signal_data, volume_15m, timeframe):
    """تنسيق وإرسال تنبيه إشارة تداول جديدة إلى Telegram."""
    try:
        entry_price = float(signal_data['entry_price'])
        target_price = float(signal_data['initial_target'])
        stop_loss_price = float(signal_data['initial_stop_loss'])
        symbol = signal_data['symbol']
        strategy_name = signal_data.get('strategy', 'N/A')
        signal_details = signal_data.get('signal_details', {})
        r2_score = signal_data.get('r2_score', 0.0) # استخدام buy_signal_score الآن

        profit_pct = ((target_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        loss_pct = ((stop_loss_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        profit_usdt = TRADE_VALUE * (profit_pct / 100)
        loss_usdt = abs(TRADE_VALUE * (loss_pct / 100))

        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')

        fear_greed = get_fear_greed_index()
        btc_trend = get_btc_trend_4h()

        message = (
            f"💡 *إشارة تداول جديدة ({strategy_name.replace('_', ' ').title()})* 💡\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"📈 **نوع الإشارة:** شراء (Long)\n"
            f"🕰️ **الإطار الزمني:** {timeframe}\n"
            f"📊 **قوة الإشارة (Score/8):** {r2_score:.1f}\n" # استخدام r2_score كتمثيل لـ buy_signal_score
            f"💧 **سيولة (15 دقيقة):** {volume_15m:,.0f} USDT\n"
            f"——————————————\n"
            f"➡️ **سعر الدخول المقترح:** `${entry_price:,.8f}`\n"
            f"🎯 **الهدف الأولي:** `${target_price:,.8f}` ({profit_pct:+.2f}% / ≈ ${profit_usdt:+.2f})\n"
            f"🛑 **وقف الخسارة الأولي:** `${stop_loss_price:,.8f}` ({loss_pct:.2f}% / ≈ ${loss_usdt:.2f})\n"
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

        send_telegram_message(chat_id, message, reply_markup=reply_markup, parse_mode='Markdown')

    except KeyError as ke:
        logger.error(f"❌ [Telegram Alert] بيانات الإشارة غير كاملة للزوج {signal_data.get('symbol', 'N/A')}: مفتاح مفقود {ke}")
    except Exception as e:
        logger.error(f"❌ [Telegram Alert] فشل إرسال تنبيه الإشارة للزوج {signal_data.get('symbol', 'N/A')}: {e}", exc_info=True)


# --- دالة جديدة لتنسيق وإرسال التنبيهات المحسّنة لحالات التتبع ---
def send_improved_telegram_notification(details):
    """تنسيق وإرسال تنبيهات تليجرام المحسّنة لحالات مختلفة."""
    symbol = details.get('symbol', 'N/A')
    signal_id = details.get('id', 'N/A')
    notification_type = details.get('type', 'unknown')
    message = ""
    safe_symbol = symbol.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace('`', '\\`')

    if notification_type == 'target_hit':
        closing_price = details.get('closing_price', 0.0)
        profit_pct = details.get('profit_pct', 0.0)
        message = (
            f"✅ *الهدف تحقق (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"🎯 **سعر الإغلاق (الهدف):** `${closing_price:,.8f}`\n"
            f"💰 **الربح المحقق:** {profit_pct:+.2f}%"
        )
    elif notification_type == 'stop_loss_hit':
        closing_price = details.get('closing_price', 0.0)
        profit_pct = details.get('profit_pct', 0.0)
        sl_type = details.get('sl_type', 'بخسارة ❌') # استخدام القيمة المحسوبة
        message = (
            f"🛑 *وصل وقف الخسارة (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"🚫 **سعر الإغلاق (الوقف):** `${closing_price:,.8f}`\n"
            f"📉 **النتيجة:** {profit_pct:.2f}% ({sl_type})"
        )
    elif notification_type == 'trailing_activated':
        current_price = details.get('current_price', 0.0)
        atr_value = details.get('atr_value', 0.0)
        new_stop_loss = details.get('new_stop_loss', 0.0)
        activation_profit_pct = details.get('activation_profit_pct', 0.0)
        message = (
            f"⬆️ *تفعيل الوقف المتحرك (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"📈 **السعر الحالي (عند التفعيل):** `${current_price:,.8f}` (ربح > {activation_profit_pct:.1f}%)\n"
            f"📊 **قيمة ATR المستخدمة:** `{atr_value:,.8f}` (Multiplier: {TRAILING_STOP_ATR_MULTIPLIER})\n" # إضافة المضاعف للتوضيح
            f"🛡️ **وقف الخسارة الجديد:** `${new_stop_loss:,.8f}`"
        )
    elif notification_type == 'trailing_updated':
        current_price = details.get('current_price', 0.0)
        atr_value = details.get('atr_value', 0.0)
        old_stop_loss = details.get('old_stop_loss', 0.0)
        new_stop_loss = details.get('new_stop_loss', 0.0)
        trigger_price_increase_pct = details.get('trigger_price_increase_pct', 0.0)
        message = (
            f"➡️ *تحديث الوقف المتحرك (ID: {signal_id})*\n"
            f"——————————————\n"
            f"🪙 **الزوج:** `{safe_symbol}`\n"
            f"📈 **السعر الحالي (عند التحديث):** `${current_price:,.8f}` (+{trigger_price_increase_pct:.1f}% منذ آخر تحديث)\n"
            f"📊 **قيمة ATR المستخدمة:** `{atr_value:,.8f}` (Multiplier: {TRAILING_STOP_ATR_MULTIPLIER})\n" # إضافة المضاعف للتوضيح
            f"🔒 **الوقف السابق:** `${old_stop_loss:,.8f}`\n"
            f"🛡️ **وقف الخسارة الجديد:** `${new_stop_loss:,.8f}`"
        )
    else:
        logger.warning(f"⚠️ [Notification] نوع تنبيه غير معروف: {notification_type} للبيانات: {details}")
        return # لا ترسل شيئًا إذا كان النوع غير معروف

    if message:
        # يمكنك إضافة أزرار هنا إذا أردت، مثل زر لعرض تفاصيل الإشارة المحددة
        reply_markup = None
        # مثال لإضافة زر (يتطلب تعديل معالج webhook للتعامل معه)
        # reply_markup = {
        #     "inline_keyboard": [
        #         [{"text": f"🔍 تفاصيل الإشارة {signal_id}", "callback_data": f"signal_details_{signal_id}"}]
        #     ]
        # }
        send_telegram_message(chat_id, message, parse_mode='Markdown', reply_markup=reply_markup)

# ---------------------- دوال قاعدة البيانات (إدراج وتحديث) ----------------------
def insert_signal_into_db(signal):
    """إدراج إشارة جديدة في جدول signals."""
    try:
        check_db_connection()
        signal_prepared = convert_np_values(signal) # <-- التأكد من التحويل قبل JSON dump
        signal_details_json = json.dumps(signal_prepared.get('signal_details', {}))
        volume_15m = signal_prepared.get('volume_15m')

        with conn.cursor() as cur_ins:
            insert_query = sql.SQL("""
                INSERT INTO signals
                 (symbol, entry_price, initial_target, initial_stop_loss, current_target, current_stop_loss,
                 r2_score, strategy_name, signal_details, last_trailing_update_price, volume_15m)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """)
            cur_ins.execute(insert_query, (
                signal_prepared['symbol'],
                signal_prepared['entry_price'],
                signal_prepared['initial_target'],
                signal_prepared['initial_stop_loss'],
                signal_prepared['current_target'], # Initially same as initial_target
                signal_prepared['initial_stop_loss'], # Initially same as initial_stop_loss
                signal_prepared.get('r2_score'), # This is buy_signal_score
                signal_prepared.get('strategy', 'conservative_combo'), # Use the actual strategy name
                signal_details_json,
                None, # last_trailing_update_price is initially NULL
                volume_15m
            ))
        conn.commit()
        logger.info(f"✅ [DB] تم إدراج إشارة للزوج {signal_prepared['symbol']} في قاعدة البيانات.")
    except psycopg2.Error as db_err:
        logger.error(f"❌ [DB] خطأ في قاعدة البيانات عند إدراج الإشارة للزوج {signal.get('symbol', 'N/A')}: {db_err}")
        conn.rollback()
    except TypeError as json_err:
         logger.error(f"❌ [DB] خطأ تحويل تفاصيل الإشارة إلى JSON قبل الإدراج: {json_err} - Signal: {signal.get('symbol', 'N/A')}")
         if conn: conn.rollback()
    except Exception as e:
        logger.error(f"❌ [DB] خطأ غير متوقع في إدراج الإشارة للزوج {signal.get('symbol', 'N/A')}: {e}")
        if conn: conn.rollback()


# ---------------------- دالة توليد الإشارة الرئيسية ----------------------
def generate_signal_elliott_fib_candle(df_input, symbol):
    """
    توليد إشارة شراء بناءً على DataFrame المعالج باستخدام استراتيجية ElliottFibCandleStrategy.
    تتضمن فحص حجم التداول، هامش الربح، ترند البيتكوين، وحساب الهدف/الوقف.
    """
    # 1. فحص ترند البيتكوين (شرط أولي)
    btc_trend = get_btc_trend_4h()
    if "هبوط" in btc_trend:
        logger.info(f"ℹ️ [Signal Gen] {symbol}: التداول متوقف مؤقتًا بسبب ترند البيتكوين الهابط ({btc_trend}).")
        return None
    elif "N/A" in btc_trend:
         logger.warning(f"⚠️ [Signal Gen] {symbol}: لا يمكن تحديد ترند البيتكوين، سيتم تجاهل هذا الشرط.")

    # 2. التحقق من صحة DataFrame المدخل
    if df_input is None or df_input.empty:
        logger.warning(f"⚠️ [Signal Gen] DataFrame فارغ أو غير صالح للزوج {symbol}.")
        return None

    # 3. تطبيق الاستراتيجية (حساب المؤشرات وتحديد إشارة الشراء)
    strategy = ElliottFibCandleStrategy()
    df_processed = strategy.populate_indicators(df_input.copy())
    if df_processed is None or df_processed.empty: # التحقق من None أيضًا
        logger.warning(f"⚠️ [Signal Gen] DataFrame فارغ بعد حساب المؤشرات للزوج {symbol}.")
        return None

    df_with_signals = strategy.populate_buy_trend(df_processed)
    if df_with_signals is None or df_with_signals.empty or 'buy' not in df_with_signals.columns: # التحقق من None
         logger.warning(f"⚠️ [Signal Gen] لم يتم العثور على عمود 'buy' بعد تطبيق الاستراتيجية لـ {symbol}.")
         return None

    # 4. التحقق من وجود إشارة شراء في آخر شمعة
    if df_with_signals['buy'].iloc[-1] != 1:
        # logger.debug(f"ℹ️ [Signal Gen] {symbol}: لا توجد إشارة شراء في آخر شمعة.")
        return None

    # 5. استخلاص بيانات الشمعة الأخيرة
    last_signal_row = df_with_signals.iloc[-1]
    current_price = last_signal_row['close']
    current_atr = last_signal_row.get('atr')
    buy_score = last_signal_row.get('buy_signal_score', 0.0)
    signal_details_json = last_signal_row.get('signal_details_json') # الحصول على JSON مباشرة
    try:
         # محاولة فك التشفير للتحقق، ولكن سنستخدم JSON مباشرة في الإشارة
         signal_details = json.loads(signal_details_json) if signal_details_json else {}
    except (json.JSONDecodeError, TypeError) as e:
         logger.warning(f"⚠️ [Signal Gen] فشل فك تشفير signal_details_json لـ {symbol}: {e}. استخدام قاموس فارغ.")
         signal_details = {} # استخدم قاموس فارغ إذا فشل التحويل

    if pd.isna(current_price) or current_price <= 0 or pd.isna(current_atr) or current_atr <= 0:
        logger.warning(f"⚠️ [Signal Gen] بيانات سعر ({current_price}) أو ATR ({current_atr}) غير صالحة للزوج {symbol}.")
        return None

    # 6. فحص حجم التداول (السيولة)
    volume_recent = fetch_recent_volume(symbol)
    if volume_recent < MIN_VOLUME_15M_USDT:
        logger.info(f"ℹ️ [Signal Gen] {symbol}: السيولة ({volume_recent:,.0f} USDT) أقل من الحد الأدنى ({MIN_VOLUME_15M_USDT:,.0f} USDT).")
        return None

    # 7. حساب الهدف ووقف الخسارة الأولي بناءً على ATR
    # التأكد من وجود القيم قبل استخدامها
    adx_val_sig = last_signal_row.get('adx', 0)
    di_plus_sig = last_signal_row.get('di_plus', 0)
    di_minus_sig = last_signal_row.get('di_minus', 0)
    if pd.isna(adx_val_sig): adx_val_sig = 0
    if pd.isna(di_plus_sig): di_plus_sig = 0
    if pd.isna(di_minus_sig): di_minus_sig = 0

    if adx_val_sig > 25 and di_plus_sig > di_minus_sig:
        target_multiplier = ENTRY_ATR_MULTIPLIER * 1.2 # زيادة الهدف في الترند القوي
        stop_loss_multiplier = ENTRY_ATR_MULTIPLIER * 0.8 # تضييق الوقف في الترند القوي
        if 'SL_Target_Mode' not in signal_details: signal_details['SL_Target_Mode'] = 'Strong Trend Adjustment'
    else:
         target_multiplier = ENTRY_ATR_MULTIPLIER
         stop_loss_multiplier = ENTRY_ATR_MULTIPLIER
         if 'SL_Target_Mode' not in signal_details: signal_details['SL_Target_Mode'] = 'Standard ATR Multiplier'

    initial_target = current_price + (target_multiplier * current_atr)
    initial_stop_loss = current_price - (stop_loss_multiplier * current_atr)

    if initial_stop_loss <= 0:
        # ضمان وجود وقف خسارة صالح حتى لو كان ATR كبيرًا جدًا أو السعر منخفضًا
        min_sl_price = current_price * (1 - 0.05) # مثال: 5% كحد أقصى للخسارة الأولية
        initial_stop_loss = max(min_sl_price, 1e-9) # تجنب الصفر أو القيم السالبة
        logger.warning(f"⚠️ [Signal Gen] وقف الخسارة المحسوب ({initial_stop_loss}) غير صالح للزوج {symbol}. تم تعديله إلى {initial_stop_loss:.8f}")
        if 'Warning' not in signal_details: signal_details['Warning'] = 'Initial Stop Loss Adjusted (was <= 0)'

    # 8. فحص هامش الربح الأدنى
    profit_margin_pct = ((initial_target / current_price) - 1) * 100 if current_price > 0 else 0
    if profit_margin_pct < MIN_PROFIT_MARGIN_PCT:
        logger.info(f"ℹ️ [Signal Gen] {symbol}: هامش الربح ({profit_margin_pct:.2f}%) أقل من الحد الأدنى المطلوب ({MIN_PROFIT_MARGIN_PCT:.2f}%).")
        return None

    # 10. تجميع بيانات الإشارة النهائية
    signal = {
        'symbol': symbol,
        'entry_price': float(f"{current_price:.8f}"), # تنسيق الدقة
        'initial_target': float(f"{initial_target:.8f}"),
        'initial_stop_loss': float(f"{initial_stop_loss:.8f}"),
        'current_target': float(f"{initial_target:.8f}"), # عند الإنشاء، الهدف الحالي هو الهدف الأولي
        'current_stop_loss': float(f"{initial_stop_loss:.8f}"), # عند الإنشاء، الوقف الحالي هو الوقف الأولي
        'r2_score': buy_score, # استخدام اسم الحقل الصحيح في DB
        'trade_value': TRADE_VALUE,
        'strategy': 'Conservative_Combo', # اسم الاستراتيجية المستخدمة
        'signal_details': signal_details, # تمرير القاموس مباشرة
        'volume_15m': volume_recent
    }

    logger.info(f"✅ [Signal Gen] {symbol}: إشارة شراء مؤكدة عند {current_price:.8f} (Score: {buy_score:.1f}, ATR: {current_atr:.8f})")
    return signal


# ---------------------- دالة تتبع الإشارات المفتوحة (محسّنة) ----------------------
def track_signals():
    """تتبع الإشارات المفتوحة، التحقق من الأهداف ووقف الخسارة، وتطبيق الوقف المتحرك المحسّن."""
    logger.info("ℹ️ [Tracker] بدء عملية تتبع الإشارات المفتوحة...")
    while True:
        active_signals_details = []
        try:
            check_db_connection()
            # استخدام cursor context manager لضمان إغلاقه
            with conn.cursor() as track_cur: # يستخدم RealDictCursor
                track_cur.execute("""
                    SELECT id, symbol, entry_price, initial_stop_loss, current_target, current_stop_loss,
                           is_trailing_active, last_trailing_update_price
                    FROM signals
                    WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;
                """)
                open_signals = track_cur.fetchall()

                if not open_signals:
                    time.sleep(10) # انتظر أقل إذا لم تكن هناك إشارات
                    continue

                for signal_row in open_signals:
                    # استخلاص البيانات الأساسية
                    signal_id = signal_row['id']
                    symbol = signal_row['symbol']
                    # التحويل الآمن للقيم الرقمية
                    try:
                        entry_price = float(signal_row['entry_price'])
                        initial_stop_loss = float(signal_row['initial_stop_loss']) # جلب الوقف الأولي للمقارنة
                        current_target = float(signal_row['current_target'])
                        current_stop_loss = float(signal_row['current_stop_loss'])
                        is_trailing_active = signal_row['is_trailing_active']
                        last_trailing_update_price = float(signal_row['last_trailing_update_price']) if signal_row['last_trailing_update_price'] is not None else None
                    except (TypeError, ValueError) as convert_err:
                        logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ في تحويل قيم الإشارة: {convert_err}")
                        continue # تخطي هذه الإشارة

                    # الحصول على السعر الحالي من بيانات WebSocket
                    current_price = ticker_data.get(symbol)

                    if current_price is None:
                        logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): لا يتوفر سعر حالي في بيانات Ticker.")
                        continue

                    # تسجيل حالة الإشارة النشطة (للتتبع)
                    active_signals_details.append(f"{symbol}({signal_id}): P={current_price:.4f}, T={current_target:.4f}, SL={current_stop_loss:.4f}, Trail={'On' if is_trailing_active else 'Off'}")

                    update_query = None
                    update_params = ()
                    log_message = None
                    notification_details = {'symbol': symbol, 'id': signal_id} # لتمريرها إلى دالة التنبيه

                    # 1. التحقق من الوصول للهدف
                    if current_price >= current_target:
                        profit_pct = ((current_target / entry_price) - 1) * 100 if entry_price > 0 else 0
                        update_query = sql.SQL("""
                            UPDATE signals
                            SET achieved_target = TRUE, closing_price = %s, closed_at = NOW(), profit_percentage = %s
                            WHERE id = %s;
                        """)
                        update_params = (current_target, profit_pct, signal_id)
                        log_message = f"🎯 [Tracker] {symbol}(ID:{signal_id}): وصل الهدف عند {current_target:.8f} (ربح: {profit_pct:+.2f}%)."
                        notification_details.update({
                            'type': 'target_hit',
                            'closing_price': current_target,
                            'profit_pct': profit_pct
                        })

                    # 2. التحقق من الوصول لوقف الخسارة
                    elif current_price <= current_stop_loss:
                        loss_pct = ((current_stop_loss / entry_price) - 1) * 100 if entry_price > 0 else 0
                        profitable_sl = current_stop_loss > entry_price
                        sl_type_msg = "بربح ✅" if profitable_sl else "بخسارة ❌"

                        update_query = sql.SQL("""
                            UPDATE signals
                            SET hit_stop_loss = TRUE, closing_price = %s, closed_at = NOW(),
                                profit_percentage = %s, profitable_stop_loss = %s
                            WHERE id = %s;
                        """)
                        update_params = (current_stop_loss, loss_pct, profitable_sl, signal_id)
                        log_message = f"🔻 [Tracker] {symbol}(ID:{signal_id}): وصل وقف الخسارة ({sl_type_msg.split(' ')[0]}) عند {current_stop_loss:.8f} (نسبة: {loss_pct:.2f}%)."
                        notification_details.update({
                            'type': 'stop_loss_hit',
                            'closing_price': current_stop_loss,
                            'profit_pct': loss_pct,
                            'sl_type': sl_type_msg
                        })

                    # 3. التحقق من تفعيل أو تحديث وقف الخسارة المتحرك
                    else:
                        # أ. تفعيل الوقف المتحرك
                        activation_threshold_price = entry_price * (1 + TRAILING_STOP_ACTIVATION_PROFIT_PCT)
                        if not is_trailing_active and current_price >= activation_threshold_price:
                            logger.info(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): السعر {current_price:.8f} وصل لعتبة تفعيل الوقف المتحرك ({activation_threshold_price:.8f}). جلب ATR...")
                            df_atr = fetch_historical_data(symbol, interval=SIGNAL_TRACKING_TIMEFRAME, days=SIGNAL_TRACKING_LOOKBACK_DAYS)
                            if df_atr is not None and not df_atr.empty:
                                df_atr = calculate_atr_indicator(df_atr, period=ENTRY_ATR_PERIOD)
                                if not df_atr.empty and 'atr' in df_atr.columns and pd.notna(df_atr['atr'].iloc[-1]):
                                    current_atr_val = df_atr['atr'].iloc[-1]
                                    if current_atr_val > 0:
                                        new_stop_loss_calc = current_price - (TRAILING_STOP_ATR_MULTIPLIER * current_atr_val)
                                        # نضمن أنه أعلى من الوقف الأولي وأعلى بقليل من سعر الدخول
                                        new_stop_loss = max(new_stop_loss_calc, initial_stop_loss, entry_price * (1 + 0.01)) # نضمن ربح بسيط جداً على الأقل
                                        # تأكد من أن الوقف الجديد أعلى فعلاً من الوقف الحالي (الأولي في هذه الحالة)
                                        if new_stop_loss > current_stop_loss:
                                            update_query = sql.SQL("""
                                                UPDATE signals
                                                SET is_trailing_active = TRUE, current_stop_loss = %s, last_trailing_update_price = %s
                                                WHERE id = %s;
                                            """)
                                            update_params = (new_stop_loss, current_price, signal_id)
                                            log_message = f"📈✅ [Tracker] {symbol}(ID:{signal_id}): تفعيل الوقف المتحرك. السعر الحالي={current_price:.8f}, ATR({ENTRY_ATR_PERIOD})={current_atr_val:.8f}. الوقف الجديد: {new_stop_loss:.8f}"
                                            notification_details.update({
                                                'type': 'trailing_activated',
                                                'current_price': current_price,
                                                'atr_value': current_atr_val,
                                                'new_stop_loss': new_stop_loss,
                                                'activation_profit_pct': TRAILING_STOP_ACTIVATION_PROFIT_PCT * 100
                                            })
                                        else:
                                            logger.info(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): الوقف المتحرك المحسوب ({new_stop_loss:.8f}) ليس أعلى من الوقف الحالي ({current_stop_loss:.8f}). لن يتم التفعيل الآن.")
                                    else:
                                        logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): قيمة ATR غير صالحة ({current_atr_val}) لتفعيل الوقف المتحرك.")
                                else:
                                    logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): لا يمكن حساب ATR لتفعيل الوقف المتحرك.")
                            else:
                                logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): لا يمكن جلب بيانات لحساب ATR لتفعيل الوقف المتحرك.")


                        # ب. تحديث الوقف المتحرك
                        elif is_trailing_active and last_trailing_update_price is not None:
                            update_threshold_price = last_trailing_update_price * (1 + TRAILING_STOP_MOVE_INCREMENT_PCT)
                            if current_price >= update_threshold_price:
                                logger.info(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): السعر {current_price:.8f} وصل لعتبة تحديث الوقف المتحرك (آخر سعر تحديث {last_trailing_update_price:.8f} * {1 + TRAILING_STOP_MOVE_INCREMENT_PCT:.4f} = {update_threshold_price:.8f}). جلب ATR...")
                                df_recent = fetch_historical_data(symbol, interval=SIGNAL_TRACKING_TIMEFRAME, days=SIGNAL_TRACKING_LOOKBACK_DAYS)
                                if df_recent is not None and not df_recent.empty:
                                    df_recent = calculate_atr_indicator(df_recent, period=ENTRY_ATR_PERIOD)
                                    if not df_recent.empty and 'atr' in df_recent.columns and pd.notna(df_recent['atr'].iloc[-1]):
                                        current_atr_val_update = df_recent['atr'].iloc[-1]
                                        if current_atr_val_update > 0:
                                            potential_new_stop_loss = current_price - (TRAILING_STOP_ATR_MULTIPLIER * current_atr_val_update)
                                            # فقط نحدث إذا كان الوقف الجديد المحسوب أعلى من الوقف الحالي
                                            if potential_new_stop_loss > current_stop_loss:
                                                new_stop_loss = potential_new_stop_loss # تم التأكد أنه أعلى
                                                update_query = sql.SQL("""
                                                    UPDATE signals
                                                    SET current_stop_loss = %s, last_trailing_update_price = %s
                                                    WHERE id = %s;
                                                """)
                                                update_params = (new_stop_loss, current_price, signal_id)
                                                log_message = f"🔼 [Tracker] {symbol}(ID:{signal_id}): تحديث الوقف المتحرك. السعر الحالي={current_price:.8f}, ATR({ENTRY_ATR_PERIOD})={current_atr_val_update:.8f}. الوقف القديم={current_stop_loss:.8f}, الوقف الجديد: {new_stop_loss:.8f}"
                                                notification_details.update({
                                                    'type': 'trailing_updated',
                                                    'current_price': current_price,
                                                    'atr_value': current_atr_val_update,
                                                    'old_stop_loss': current_stop_loss,
                                                    'new_stop_loss': new_stop_loss,
                                                    'trigger_price_increase_pct': TRAILING_STOP_MOVE_INCREMENT_PCT * 100
                                                })
                                            else:
                                                logger.debug(f"ℹ️ [Tracker] {symbol}(ID:{signal_id}): الوقف المتحرك المحسوب ({potential_new_stop_loss:.8f}) ليس أعلى من الوقف الحالي ({current_stop_loss:.8f}). لن يتم التحديث.")
                                        else:
                                             logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): قيمة ATR غير صالحة ({current_atr_val_update}) لتحديث الوقف المتحرك.")
                                    else:
                                        logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): لا يمكن حساب ATR لتحديث الوقف المتحرك.")
                                else:
                                    logger.warning(f"⚠️ [Tracker] {symbol}(ID:{signal_id}): لا يمكن جلب بيانات لحساب ATR لتحديث الوقف المتحرك.")

                    # تنفيذ التحديث في قاعدة البيانات وإرسال التنبيه
                    if update_query:
                        try:
                             # استخدام cursor context manager
                             with conn.cursor() as update_cur:
                                update_cur.execute(update_query, update_params)
                             conn.commit()
                             if log_message: logger.info(log_message)
                             # إرسال التنبيه المحسّن
                             if notification_details.get('type'): # التأكد من وجود نوع التنبيه
                                send_improved_telegram_notification(notification_details)
                        except psycopg2.Error as db_err:
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ DB أثناء التحديث: {db_err}")
                            conn.rollback()
                        except Exception as e:
                            logger.error(f"❌ [Tracker] {symbol}(ID:{signal_id}): خطأ غير متوقع أثناء تحديث DB/إرسال تنبيه: {e}")
                            if conn: conn.rollback() # التأكد من وجود الاتصال قبل التراجع

                if active_signals_details:
                    # logger.debug(f"ℹ️ [Tracker] حالة الإشارات النشطة: {'; '.join(active_signals_details)}")
                    pass # يمكن إلغاء التعليق للتحقق

            # تقليل مدة الانتظار بين الدورات لتتبع أسرع
            time.sleep(3) # تقليل الانتظار إلى 3 ثواني

        except psycopg2.Error as db_cycle_err:
             logger.error(f"❌ [Tracker] خطأ قاعدة بيانات في دورة التتبع الرئيسية: {db_cycle_err}")
             if conn: conn.rollback()
             time.sleep(30) # انتظار أطول عند خطأ DB
             # محاولة إعادة الاتصال
             check_db_connection()
        except Exception as cycle_err:
            logger.error(f"❌ [Tracker] خطأ غير متوقع في دورة تتبع الإشارات: {cycle_err}", exc_info=True)
            time.sleep(30) # انتظار أطول عند خطأ غير متوقع


# ---------------------- خدمة Flask (اختياري للـ Webhook) ----------------------
app = Flask(__name__)

@app.route('/')
def home():
    """صفحة رئيسية بسيطة."""
    # إضافة وقت التشغيل الحالي للتحقق
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return Response(f"📈 Crypto Signal Bot is running... Current Time: {now}", status=200, mimetype='text/plain')

@app.route('/favicon.ico')
def favicon():
    """معالجة طلب أيقونة المفضلة."""
    return Response(status=204)

@app.route('/webhook', methods=['POST'])
def webhook():
    """معالجة الطلبات الواردة من Telegram (مثل ضغط الأزرار)."""
    if not request.is_json:
        logger.warning("⚠️ [Flask] Received non-JSON webhook request.")
        return "Invalid request", 400
    try:
        data = request.get_json()
        # logger.info(f"ℹ️ [Flask] Received webhook data: {json.dumps(data, indent=2)}") # يمكن أن يكون اللوغ كبيرًا جدًا

        if 'callback_query' in data:
            callback_query = data['callback_query']
            callback_data = callback_query.get('data')
            chat_id_callback = callback_query['message']['chat']['id']
            message_id = callback_query['message']['message_id']

            try:
                # إرسال تأكيد الاستلام بسرعة
                requests.post(f"https://api.telegram.org/bot{telegram_token}/answerCallbackQuery",
                     json={'callback_query_id': callback_query['id']}, timeout=5)
            except Exception as ack_err:
                 logger.error(f"❌ [Flask] Failed to acknowledge callback query {callback_query['id']}: {ack_err}")

            if callback_data == "get_report":
                report_text = generate_performance_report()
                send_telegram_message(chat_id_callback, report_text, parse_mode='Markdown')
            # يمكنك إضافة معالجة لـ callback_data أخرى هنا، مثل "signal_details_{signal_id}"

        elif 'message' in data:
            message_data = data['message']
            chat_id_msg = message_data['chat']['id']
            text_msg = message_data.get('text', '')
            # logger.info(f"ℹ️ [Flask] Received message from {chat_id_msg}: {text_msg}")
            if text_msg.lower() == '/report':
                report_text = generate_performance_report()
                send_telegram_message(chat_id_msg, report_text, parse_mode='Markdown')
            elif text_msg.lower() == '/status':
                 # مثال لإضافة أمر يعرض حالة البوت أو عدد الإشارات النشطة
                 try:
                     check_db_connection()
                     with conn.cursor() as status_cur:
                         status_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;")
                         open_count = status_cur.fetchone()['count'] or 0
                     status_msg = f"🤖 حالة البوت:\n- الإشارات النشطة: {open_count}/{MAX_OPEN_TRADES}\n- تتبع الأسعار: {'نشط ✅' if ticker_data else 'غير نشط ❌'}\n- وقت الخادم: {datetime.now().strftime('%H:%M:%S')}"
                     send_telegram_message(chat_id_msg, status_msg)
                 except Exception as status_err:
                     logger.error(f"❌ [Flask] Error getting status: {status_err}")
                     send_telegram_message(chat_id_msg, "❌ حدث خطأ أثناء جلب الحالة.")


        return "OK", 200
    except Exception as e:
         logger.error(f"❌ [Flask] Error processing webhook: {e}", exc_info=True)
         return "Error", 500


def run_flask():
    """تشغيل تطبيق Flask لسماع الـ Webhook."""
    if webhook_url:
        logger.info(f"ℹ️ [Flask] Starting Flask app on 0.0.0.0:10000")
        try:
            from waitress import serve
            serve(app, host="0.0.0.0", port=10000, threads=6) # Use waitress with multiple threads
        except ImportError:
             logger.warning("⚠️ [Flask] 'waitress' not installed. Falling back to Flask development server (not recommended for production).")
             app.run(host="0.0.0.0", port=10000)
    else:
         logger.info("ℹ️ [Flask] Webhook URL not configured. Flask server will not start.")


# ---------------------- الدالة الرئيسية ودورة الفحص ----------------------
def main_loop():
    """الحلقة الرئيسية لفحص الأزواج وتوليد الإشارات."""
    # استدعاء الدالة الجديدة التي تتحقق من الرموز
    symbols = get_crypto_symbols() # الآن هذه القائمة تحتوي فقط على رموز صالحة
    if not symbols:
        logger.error("❌ [Main] لم يتم تحميل أو التحقق من أي رموز صالحة. الخروج...")
        return

    logger.info(f"ℹ️ [Main] بدء دورة فحص السوق لـ {len(symbols)} رمزًا صالحًا...")
    last_full_scan_time = time.time()

    while True:
        try:
            check_db_connection()

            # 1. التحقق من عدد التوصيات المفتوحة حاليًا
            open_count = 0
            try:
                 with conn.cursor() as cur_check:
                    cur_check.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;")
                    result = cur_check.fetchone()
                    open_count = result['count'] if result else 0
            except psycopg2.Error as db_err:
                 logger.error(f"❌ [Main] خطأ DB أثناء التحقق من عدد الإشارات المفتوحة: {db_err}")
                 conn.rollback()
                 time.sleep(60)
                 continue

            if open_count >= MAX_OPEN_TRADES:
                logger.info(f"ℹ️ [Main] تم الوصول للحد الأقصى للإشارات المفتوحة ({open_count}/{MAX_OPEN_TRADES}). الانتظار...")
                time.sleep(60) # انتظار دقيقة قبل إعادة التحقق
                continue

            # 2. المرور على قائمة الرموز **الصالحة** وفحصها
            logger.info(f"ℹ️ [Main] بدء فحص الرموز ({len(symbols)})... العدد المفتوح حاليًا: {open_count}")
            processed_count = 0
            symbols_to_process = symbols[:] # العمل على نسخة من القائمة الصالحة

            for symbol in symbols_to_process:
                 # التحقق من الحد الأقصى داخل الحلقة أيضًا
                 try:
                     with conn.cursor() as cur_recheck:
                        cur_recheck.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;")
                        result = cur_recheck.fetchone()
                        current_open_count = result['count'] if result else 0
                     if current_open_count >= MAX_OPEN_TRADES:
                         logger.info(f"ℹ️ [Main] تم الوصول للحد الأقصى ({current_open_count}) أثناء الفحص. إيقاف الفحص مؤقتًا لهذه الدورة.")
                         break # الخروج من حلقة فحص الرموز الحالية

                     # التحقق مما إذا كان هناك إشارة مفتوحة لهذا الرمز المحدد
                     with conn.cursor() as symbol_cur:
                         symbol_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE symbol = %s AND achieved_target = FALSE AND hit_stop_loss = FALSE;", (symbol,))
                         result_symbol = symbol_cur.fetchone()
                         count_symbol_open = result_symbol['count'] if result_symbol else 0
                     if count_symbol_open > 0:
                         # logger.debug(f"ℹ️ [Main] تخطي {symbol}، توجد إشارة مفتوحة بالفعل.")
                         continue

                 except psycopg2.Error as db_err:
                      logger.error(f"❌ [Main] خطأ DB أثناء التحقق من الرمز {symbol}: {db_err}")
                      conn.rollback()
                      continue # الانتقال للرمز التالي
                 except Exception as check_err: # التقاط أخطاء عامة أثناء التحقق
                      logger.error(f"❌ [Main] خطأ عام أثناء التحقق من الرمز {symbol}: {check_err}")
                      continue # الانتقال للرمز التالي


                 # جلب البيانات خارج كتلة try..except الخاصة بالـ DB
                 try:
                      # استخدام إطار زمني أقصر للبيانات إذا لزم الأمر، أو الحفاظ عليه
                      df = fetch_historical_data(symbol, interval=SIGNAL_GENERATION_TIMEFRAME, days=SIGNAL_GENERATION_LOOKBACK_DAYS)
                 except Exception as fetch_err:
                      logger.error(f"❌ [Main] فشل جلب البيانات للرمز {symbol}: {fetch_err}")
                      df = None # تعيين df إلى None في حالة الفشل

                 if df is None or df.empty:
                      # logger.warning(f"⚠️ [Main] لا توجد بيانات أو فشل جلبها للرمز {symbol}.")
                      continue # الانتقال للرمز التالي

                 # توليد الإشارة
                 try:
                      signal = generate_signal_elliott_fib_candle(df, symbol)
                 except Exception as gen_err:
                      logger.error(f"❌ [Main] فشل توليد الإشارة للرمز {symbol}: {gen_err}", exc_info=True)
                      signal = None # تعيين signal إلى None في حالة الفشل


                 if signal:
                     # التأكد مرة أخرى من عدم تجاوز الحد الأقصى قبل الإدراج
                     try:
                         with conn.cursor() as final_check_cur:
                              final_check_cur.execute("SELECT COUNT(*) AS count FROM signals WHERE achieved_target = FALSE AND hit_stop_loss = FALSE;")
                              result_final = final_check_cur.fetchone()
                              final_open_count = result_final['count'] if result_final else 0
                         if final_open_count < MAX_OPEN_TRADES:
                              insert_signal_into_db(signal)
                              vol = signal.get('volume_15m', 0)
                              send_telegram_alert(signal, vol, SIGNAL_GENERATION_TIMEFRAME)
                              processed_count += 1
                              time.sleep(2) # فاصل بسيط بين إرسال الإشارات لتجنب قيود Telegram
                         else:
                              logger.warning(f"⚠️ [Main] تم الوصول للحد الأقصى ({final_open_count}) قبل إدراج إشارة {symbol}. تم تجاهل الإشارة.")
                              # بما أن الحد الأقصى تم الوصول إليه، لا داعي لفحص بقية الرموز في هذه الدورة
                              break
                     except psycopg2.Error as db_err:
                          logger.error(f"❌ [Main] خطأ DB أثناء التحقق النهائي أو إدراج إشارة {symbol}: {db_err}")
                          conn.rollback()
                          # قد يكون من الأفضل التوقف المؤقت هنا لتجنب مشاكل متكررة
                          time.sleep(30)
                          break
                     except Exception as insert_err:
                          logger.error(f"❌ [Main] خطأ عام أثناء إدراج/إرسال إشارة {symbol}: {insert_err}")
                          # قد يكون من الأفضل التوقف المؤقت هنا أيضًا
                          time.sleep(30)
                          break

                 # فاصل قصير بين فحص كل رمز لتخفيف العبء على Binance API
                 time.sleep(0.5)


            # 3. انتظار قبل بدء الدورة التالية
            logger.info(f"ℹ️ [Main] انتهاء دورة الفحص. تم معالجة/إرسال {processed_count} إشارة جديدة (إن وجدت).")
            scan_duration = time.time() - last_full_scan_time
            wait_time = max(60, 300 - scan_duration) # انتظر على الأقل دقيقة، أو أكمل إلى 5 دقائق
            logger.info(f"ℹ️ [Main] مدة الفحص: {scan_duration:.1f} ثانية. الانتظار {wait_time:.1f} ثانية للدورة التالية.")
            time.sleep(wait_time) # الفاصل الزمني بين دورات الفحص الكاملة
            last_full_scan_time = time.time()

        except KeyboardInterrupt:
             logger.info("🛑 [Main] تم استقبال طلب إيقاف (KeyboardInterrupt). إغلاق...")
             break
        except psycopg2.Error as db_main_err:
             logger.error(f"❌ [Main] خطأ فادح في قاعدة البيانات في الحلقة الرئيسية: {db_main_err}. محاولة إعادة الاتصال...")
             if conn: conn.rollback()
             time.sleep(60)
             try:
                 init_db() # محاولة إعادة تهيئة الاتصال
             except Exception as recon_err:
                 logger.critical(f"❌ [Main] فشلت محاولة إعادة الاتصال بقاعدة البيانات: {recon_err}. الخروج...")
                 break
        except Exception as main_err:
            logger.error(f"❌ [Main] خطأ غير متوقع في الحلقة الرئيسية: {main_err}", exc_info=True)
            logger.info("ℹ️ [Main] انتظار 120 ثانية قبل إعادة المحاولة...")
            time.sleep(120)


# ---------------------- نقطة الدخول الرئيسية ----------------------
if __name__ == "__main__":
    logger.info("🚀 بدء تشغيل بوت إشارات التداول...")
    logger.info(f"Current Time (Local): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Current Time (UTC):   {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")


    # 1. تهيئة قاعدة البيانات أولاً
    try:
        init_db()
    except Exception as e:
        logger.critical(f"❌ [Main] فشل تهيئة قاعدة البيانات عند البدء. لا يمكن المتابعة.")
        exit()

    # 2. بدء WebSocket Ticker في خيط منفصل
    ws_thread = Thread(target=run_ticker_socket_manager, daemon=True, name="WebSocketThread")
    ws_thread.start()
    logger.info("✅ [Main] تم بدء خيط WebSocket Ticker.")
    # انتظر قليلاً للسماح لـ WebSocket بالاتصال وتلقي بعض البيانات الأولية
    time.sleep(5)

    # 3. بدء متتبع الإشارات في خيط منفصل
    tracker_thread = Thread(target=track_signals, daemon=True, name="TrackerThread")
    tracker_thread.start()
    logger.info("✅ [Main] تم بدء خيط تتبع الإشارات.")

    # 4. بدء خادم Flask (إذا تم تكوين Webhook) في خيط منفصل
    if webhook_url:
        flask_thread = Thread(target=run_flask, daemon=True, name="FlaskThread")
        flask_thread.start()
        logger.info("✅ [Main] تم بدء خيط Flask Webhook.")
    else:
         logger.info("ℹ️ [Main] لم يتم تكوين Webhook URL، لن يتم بدء خادم Flask.")

    # 5. بدء الحلقة الرئيسية في الخيط الرئيسي
    try:
         main_loop()
    except Exception as final_err:
         logger.critical(f"❌ [Main] حدث خطأ فادح غير معالج في الحلقة الرئيسية: {final_err}", exc_info=True)
    finally:
         logger.info("🛑 [Main] البرنامج في طور الإغلاق...")
         # يمكنك هنا إرسال رسالة تليجرام لإعلامك بالإغلاق
         # send_telegram_message(chat_id, "⚠️ تنبيه: بوت التداول قيد الإيقاف الآن.")
         if conn:
             try:
                 conn.close()
                 logger.info("✅ [DB] تم إغلاق اتصال قاعدة البيانات.")
             except Exception as close_err:
                 logger.error(f"⚠️ [DB] خطأ أثناء إغلاق اتصال قاعدة البيانات: {close_err}")
         logger.info("👋 [Main] تم إيقاف بوت إشارات التداول.")
         # تأكد من إنهاء العملية بالكامل
         os._exit(0) # طريقة لضمان الخروج حتى لو كانت هناك خيوط عالقة (استخدم بحذر)
