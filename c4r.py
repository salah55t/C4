import os
import time
import logging
import psycopg2
import numpy as np
import pandas as pd
from decouple import config
from binance.client import Client
from psycopg2.extras import RealDictCursor
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from typing import List, Dict, Optional, Tuple

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sr_scanner.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SR_Scanner')

# ---------------------- تحميل متغيرات البيئة ----------------------
# هذا السكريبت يستخدم نفس متغيرات البيئة الخاصة بالبوت الرئيسي
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
    exit(1)

# ---------------------- إعداد الثوابت ----------------------
# كمية البيانات التاريخية التي سيتم تحليلها
DATA_FETCH_DAYS_4H = 200
DATA_FETCH_DAYS_15M = 30

# معايير تحديد القمم والقيعان (يمكن تعديلها حسب الحاجة)
# prominence: مدى بروز القمة/القاع مقارنة بما حوله
# width: عرض القمة/القاع
PROMINENCE_4H = 0.015  # 1.5%
WIDTH_4H = 5

PROMINENCE_15M = 0.008 # 0.8%
WIDTH_15M = 10

# معايير تجميع المستويات (Clustering)
# eps: المسافة القصوى بين نقطتين ليتم اعتبارهما في نفس المجموعة (نسبة مئوية من السعر)
CLUSTER_EPS_PERCENT = 0.005 # 0.5%

# ---------------------- دوال Binance والبيانات ----------------------
def get_binance_client() -> Optional[Client]:
    """يقوم بتهيئة والتحقق من الاتصال مع Binance."""
    try:
        client = Client(API_KEY, API_SECRET)
        client.ping()
        logger.info("✅ [Binance] تم الاتصال بواجهة برمجة تطبيقات Binance بنجاح.")
        return client
    except Exception as e:
        logger.critical(f"❌ [Binance] فشل الاتصال بواجهة برمجة التطبيقات: {e}")
        return None

def fetch_historical_data(client: Client, symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """جلب البيانات التاريخية لعملة معينة."""
    try:
        start_str = (pd.to_datetime('today') - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
        logger.info(f"⏳ [البيانات] جاري جلب بيانات {symbol} على فريم {interval} لآخر {days} يوم...")
        klines = client.get_historical_klines(symbol, interval, start_str)
        if not klines:
            logger.warning(f"⚠️ [البيانات] لم يتم العثور على بيانات لـ {symbol} على فريم {interval}.")
            return None
        
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        logger.info(f"✅ [البيانات] تم جلب {len(df)} شمعة بنجاح.")
        return df[numeric_cols].dropna()
    except Exception as e:
        logger.error(f"❌ [البيانات] خطأ أثناء جلب البيانات التاريخية لـ {symbol}: {e}")
        return None

def get_validated_symbols(client: Client, filename: str = 'crypto_list.txt') -> List[str]:
    """قراءة قائمة العملات والتحقق منها مع Binance."""
    logger.info(f"ℹ️ [التحقق] قراءة الرموز من '{filename}' والتحقق منها...")
    try:
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_symbols = {line.strip().upper() for line in f if line.strip() and not line.startswith('#')}
        
        formatted = {f"{s}USDT" if not s.endswith('USDT') else s for s in raw_symbols}
        exchange_info = client.get_exchange_info()
        active = {s['symbol'] for s in exchange_info['symbols'] if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING'}
        
        validated = sorted(list(formatted.intersection(active)))
        logger.info(f"✅ [التحقق] سيتم تحليل {len(validated)} عملة معتمدة.")
        return validated
    except Exception as e:
        logger.error(f"❌ [التحقق] حدث خطأ أثناء التحقق من الرموز: {e}", exc_info=True)
        return []

# ---------------------- دوال قاعدة البيانات ----------------------
def init_db() -> Optional[psycopg2.extensions.connection]:
    """تهيئة الاتصال بقاعدة البيانات وإنشاء الجدول إذا لم يكن موجوداً."""
    logger.info("[قاعدة البيانات] بدء تهيئة الاتصال...")
    try:
        conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS support_resistance_levels (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    level_price DOUBLE PRECISION NOT NULL,
                    level_type TEXT NOT NULL, -- 'support' or 'resistance'
                    timeframe TEXT NOT NULL, -- '15m', '4h', etc.
                    strength INTEGER NOT NULL, -- Number of touches
                    last_tested_at TIMESTAMP,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    CONSTRAINT unique_level UNIQUE (symbol, level_price, timeframe, level_type)
                );
            """)
        conn.commit()
        logger.info("✅ [قاعدة البيانات] تم تهيئة جدول 'support_resistance_levels' بنجاح.")
        return conn
    except Exception as e:
        logger.critical(f"❌ [قاعدة البيانات] فشل الاتصال أو تهيئة الجدول: {e}")
        return None

def save_levels_to_db(conn: psycopg2.extensions.connection, symbol: str, levels: List[Dict]):
    """حفظ المستويات المكتشفة في قاعدة البيانات."""
    if not levels:
        logger.info(f"ℹ️ [{symbol}] لا توجد مستويات جديدة ليتم حفظها.")
        return

    logger.info(f"⏳ [{symbol}] جاري حفظ {len(levels)} مستوى في قاعدة البيانات...")
    try:
        with conn.cursor() as cur:
            # حذف المستويات القديمة أولاً لضمان تحديث البيانات
            cur.execute("DELETE FROM support_resistance_levels WHERE symbol = %s;", (symbol,))
            logger.info(f"🗑️ [{symbol}] تم حذف المستويات القديمة.")

            # إدراج المستويات الجديدة
            insert_query = """
                INSERT INTO support_resistance_levels 
                (symbol, level_price, level_type, timeframe, strength, last_tested_at) 
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, level_price, timeframe, level_type) DO NOTHING; 
            """
            for level in levels:
                cur.execute(insert_query, (
                    symbol,
                    level['level_price'],
                    level['level_type'],
                    level['timeframe'],
                    level['strength'],
                    level['last_tested_at']
                ))
        conn.commit()
        logger.info(f"✅ [{symbol}] تم حفظ جميع المستويات الجديدة بنجاح.")
    except Exception as e:
        logger.error(f"❌ [{symbol}] حدث خطأ أثناء الحفظ في قاعدة البيانات: {e}")
        conn.rollback()


# ---------------------- دوال التحليل وتحديد المستويات ----------------------
def find_and_cluster_levels(df: pd.DataFrame, prominence: float, width: int, cluster_eps_percent: float) -> Tuple[List[Dict], List[Dict]]:
    """تحديد القمم والقيعان وتجميعها لتحديد مناطق الدعم والمقاومة."""
    
    # تحديد القيعان (الدعوم)
    lows = df['low'].to_numpy()
    low_peaks_indices, _ = find_peaks(-lows, prominence=lows.mean() * prominence, width=width)
    
    # تحديد القمم (المقاومات)
    highs = df['high'].to_numpy()
    high_peaks_indices, _ = find_peaks(highs, prominence=highs.mean() * prominence, width=width)

    def cluster_and_strengthen(prices: np.ndarray, indices: np.ndarray, level_type: str) -> List[Dict]:
        if len(indices) == 0:
            return []
        
        points = prices[indices].reshape(-1, 1)
        # تحديد المسافة القصوى للتجميع بناءً على متوسط السعر
        eps_value = points.mean() * cluster_eps_percent
        
        db = DBSCAN(eps=eps_value, min_samples=2, metric='euclidean').fit(points)
        
        clustered_levels = []
        unique_labels = set(db.labels_)
        
        for label in unique_labels:
            if label == -1: # تجاهل النقاط التي لا تنتمي لأي مجموعة (Noise)
                continue
            
            class_member_mask = (db.labels_ == label)
            cluster_points_indices = indices[class_member_mask]
            
            if len(cluster_points_indices) > 0:
                cluster_prices = prices[cluster_points_indices]
                mean_price = cluster_prices.mean()
                strength = len(cluster_prices)
                last_tested_timestamp = df.index[cluster_points_indices[-1]]
                
                clustered_levels.append({
                    "level_price": float(mean_price),
                    "level_type": level_type,
                    "strength": int(strength),
                    "last_tested_at": last_tested_timestamp.to_pydatetime()
                })
        
        return clustered_levels

    support_levels = cluster_and_strengthen(lows, low_peaks_indices, 'support')
    resistance_levels = cluster_and_strengthen(highs, high_peaks_indices, 'resistance')
    
    return support_levels, resistance_levels

# ---------------------- حلقة العمل الرئيسية ----------------------
def main():
    """الدالة الرئيسية لتشغيل السكريبت."""
    logger.info("🚀 بدء تشغيل محلل الدعوم والمقاومات...")
    
    client = get_binance_client()
    if not client:
        return
        
    conn = init_db()
    if not conn:
        return

    symbols_to_scan = get_validated_symbols(client)
    if not symbols_to_scan:
        logger.warning("⚠️ لا توجد عملات لتحليلها. سيتم إيقاف التشغيل.")
        return

    logger.info(f"🌀 سيتم تحليل {len(symbols_to_scan)} عملة. هذه العملية قد تستغرق وقتاً طويلاً.")

    for i, symbol in enumerate(symbols_to_scan):
        logger.info(f"--- ({i+1}/{len(symbols_to_scan)}) بدء تحليل العملة: {symbol} ---")
        all_symbol_levels = []

        # --- تحليل فريم 4 ساعات ---
        df_4h = fetch_historical_data(client, symbol, '4h', DATA_FETCH_DAYS_4H)
        if df_4h is not None and not df_4h.empty:
            supports_4h, resistances_4h = find_and_cluster_levels(df_4h, PROMINENCE_4H, WIDTH_4H, CLUSTER_EPS_PERCENT)
            for level in supports_4h + resistances_4h:
                level['timeframe'] = '4h'
            all_symbol_levels.extend(supports_4h)
            all_symbol_levels.extend(resistances_4h)
            logger.info(f"🔍 [{symbol}-4h] تم العثور على {len(supports_4h)} مستوى دعم و {len(resistances_4h)} مستوى مقاومة.")
        else:
            logger.warning(f"⚠️ [{symbol}-4h] تعذر جلب البيانات أو تحليلها.")
        
        time.sleep(1) # استراحة قصيرة لتجنب إغراق الـ API

        # --- تحليل فريم 15 دقيقة ---
        df_15m = fetch_historical_data(client, symbol, '15m', DATA_FETCH_DAYS_15M)
        if df_15m is not None and not df_15m.empty:
            supports_15m, resistances_15m = find_and_cluster_levels(df_15m, PROMINENCE_15M, WIDTH_15M, CLUSTER_EPS_PERCENT)
            for level in supports_15m + resistances_15m:
                level['timeframe'] = '15m'
            all_symbol_levels.extend(supports_15m)
            all_symbol_levels.extend(resistances_15m)
            logger.info(f"🔍 [{symbol}-15m] تم العثور على {len(supports_15m)} مستوى دعم و {len(resistances_15m)} مستوى مقاومة.")
        else:
            logger.warning(f"⚠️ [{symbol}-15m] تعذر جلب البيانات أو تحليلها.")
            
        # حفظ جميع المستويات المكتشفة للعملة الحالية
        if all_symbol_levels:
            save_levels_to_db(conn, symbol, all_symbol_levels)
        
        logger.info(f"--- ✅ انتهى تحليل {symbol} ---")
        time.sleep(2) # استراحة أطول بين العملات

    conn.close()
    logger.info("🎉🎉🎉 اكتملت عملية تحليل وحفظ جميع المستويات لجميع العملات بنجاح! 🎉🎉🎉")


if __name__ == "__main__":
    main()
