import psycopg2
import pandas as pd
import os
from datetime import datetime
from decouple import config
import logging

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_backtest_logs.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('BacktestDownloader')

# ---------------------- تحميل متغيرات البيئة ----------------------
try:
    DB_URL: str = config('DATABASE_URL')
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
    exit(1)

# ---------------------- دالة الاتصال بقاعدة البيانات ----------------------
def get_db_connection(retries: int = 5, delay: int = 5):
    """
    يحاول الاتصال بقاعدة البيانات مع إعادة المحاولة.
    """
    conn = None
    db_url_to_use = DB_URL
    if 'postgres' in db_url_to_use and 'sslmode' not in db_url_to_use:
        separator = '&' if '?' in db_url_to_use else '?'
        db_url_to_use += f"{separator}sslmode=require"

    for attempt in range(retries):
        try:
            logger.info(f"[DB] محاولة الاتصال بقاعدة البيانات (محاولة {attempt + 1}/{retries})...")
            conn = psycopg2.connect(db_url_to_use, connect_timeout=15)
            conn.autocommit = True # For simple read operations, autocommit is fine
            logger.info("✅ [DB] تم الاتصال بقاعدة البيانات بنجاح.")
            return conn
        except Exception as e:
            logger.error(f"❌ [DB] خطأ أثناء الاتصال بقاعدة البيانات: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logger.critical("❌ [DB] فشل حاسم في الاتصال بقاعدة البيانات بعد عدة محاولات.")
                return None
    return None

# ---------------------- دالة جلب البيانات ----------------------
def fetch_backtest_results() -> Optional[pd.DataFrame]:
    """
    يجلب جميع البيانات من جدول backtest_signals_data.
    """
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        logger.info("📊 جلب نتائج الاختبار الخلفي من قاعدة البيانات...")
        query = "SELECT * FROM backtest_signals_data ORDER BY signal_timestamp ASC;"
        df = pd.read_sql(query, conn)
        logger.info(f"✅ تم جلب {len(df)} صفًا من نتائج الاختبار الخلفي.")
        return df
    except Exception as e:
        logger.error(f"❌ خطأ أثناء جلب نتائج الاختبار الخلفي: {e}", exc_info=True)
        return None
    finally:
        if conn:
            conn.close()
            logger.info("👋 تم إغلاق اتصال قاعدة البيانات.")

# ---------------------- دالة حفظ إلى ملف CSV ----------------------
def save_to_csv(df: pd.DataFrame, filename: str = None):
    """
    يحفظ DataFrame إلى ملف CSV.
    """
    if df.empty:
        logger.warning("⚠️ لا توجد بيانات لحفظها في ملف CSV.")
        return

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_results_{timestamp}.csv"
    
    try:
        # Use utf-8-sig for better compatibility with Excel for Arabic characters
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"✅ تم حفظ نتائج الاختبار الخلفي بنجاح إلى: {filename}")
    except Exception as e:
        logger.error(f"❌ خطأ أثناء حفظ ملف CSV: {e}", exc_info=True)

# ---------------------- نقطة انطلاق السكريبت ----------------------
if __name__ == "__main__":
    logger.info("🚀 بدء سكريبت تحميل نتائج الاختبار الخلفي 🚀")
    
    results_df = fetch_backtest_results()
    
    if results_df is not None and not results_df.empty:
        save_to_csv(results_df)
    else:
        logger.info("ℹ️ لم يتم العثور على نتائج اختبار خلفي أو حدث خطأ أثناء الجلب.")
    
    logger.info("👋 اكتمل تشغيل سكريبت تحميل نتائج الاختبار الخلفي.")

