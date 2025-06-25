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
import threading
import http.server
import socketserver

# ---------------------- إعداد نظام التسجيل (Logging) ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sr_scanner_v3.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SR_Scanner_V3')

# ---------------------- تحميل متغيرات البيئة ----------------------
try:
    API_KEY: str = config('BINANCE_API_KEY')
    API_SECRET: str = config('BINANCE_API_SECRET')
    DB_URL: str = config('DATABASE_URL')
except Exception as e:
    logger.critical(f"❌ فشل حاسم في تحميل متغيرات البيئة الأساسية: {e}")
    exit(1)

# ---------------------- إعداد الثوابت ----------------------
ANALYSIS_INTERVAL_HOURS = 4  # الفاصل الزمني بين كل دورة تحليل (بالساعات)

# كمية البيانات التاريخية
DATA_FETCH_DAYS_1D = 600
DATA_FETCH_DAYS_4H = 200
DATA_FETCH_DAYS_15M = 30

# معايير تحديد القمم والقيعان
PROMINENCE_1D = 0.025
WIDTH_1D = 10
PROMINENCE_4H = 0.015
WIDTH_4H = 5
PROMINENCE_15M = 0.008
WIDTH_15M = 10

# معايير التجميع والدمج
CLUSTER_EPS_PERCENT = 0.005
CONFLUENCE_ZONE_PERCENT = 0.005

# معايير تحليل بروفايل الحجم
VOLUME_PROFILE_BINS = 100

# ---------------------- قسم خادم الويب (للتوافق مع المنصة) ----------------------
class WebServerHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        """الاستجابة لطلبات GET بصفحة HTML بسيطة."""
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        html_content = """
        <!DOCTYPE html>
        <html lang="ar" dir="rtl">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>حالة الماسح</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; color: #333; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
                .container { text-align: center; padding: 40px; background-color: white; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
                h1 { color: #0056b3; }
                p { font-size: 1.2rem; }
                .status { font-weight: bold; color: #28a745; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 ماسح الدعم والمقاومة</h1>
                <p>الخدمة <span class="status">تعمل</span> في الخلفية.</p>
                <p>يتم إجراء التحليل بشكل دوري.</p>
            </div>
        </body>
        </html>
        """
        self.wfile.write(html_content.encode('utf-8'))

def run_web_server():
    """تشغيل خادم الويب على المنفذ المحدد من قبل المنصة."""
    PORT = int(os.environ.get("PORT", 8080))
    with socketserver.TCPServer(("", PORT), WebServerHandler) as httpd:
        logger.info(f"🌐 خادم الويب يعمل على المنفذ {PORT}")
        httpd.serve_forever()

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
        script_dir = os.path.dirname(os.path.abspath(__file__))
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
    """تهيئة الاتصال بقاعدة البيانات وتحديث بنية الجدول تلقائيًا."""
    logger.info("[قاعدة البيانات] بدء تهيئة الاتصال...")
    conn = None
    try:
        conn = psycopg2.connect(DB_URL, connect_timeout=10, cursor_factory=RealDictCursor)
        with conn.cursor() as cur:
            # الخطوة 1: إنشاء الجدول باستخدام النوع NUMERIC للقوة لتجنب الأخطاء
            cur.execute("""
                CREATE TABLE IF NOT EXISTS support_resistance_levels (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    level_price DOUBLE PRECISION NOT NULL,
                    level_type TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    strength NUMERIC NOT NULL, -- استخدام NUMERIC للأرقام الكبيرة جداً
                    last_tested_at TIMESTAMP,
                    details TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    CONSTRAINT unique_level UNIQUE (symbol, level_price, timeframe, level_type)
                );
            """)
            conn.commit()

            # الخطوة 2: التحقق من وجود عمود 'details' وإضافته (للتوافق مع الإصدارات القديمة)
            cur.execute("""
                SELECT 1 FROM information_schema.columns 
                WHERE table_name='support_resistance_levels' AND column_name='details';
            """)
            if cur.fetchone() is None:
                logger.info("[قاعدة البيانات] العمود 'details' غير موجود. جاري إضافته...")
                cur.execute("ALTER TABLE support_resistance_levels ADD COLUMN details TEXT;")
                conn.commit()
                logger.info("✅ [قاعدة البيانات] تم إضافة العمود 'details' بنجاح.")

            # ---- التعديل الرئيسي ----
            # الخطوة 3: التحقق من نوع عمود 'strength' وتحديثه إلى NUMERIC إذا كان integer أو bigint
            cur.execute("""
                SELECT data_type FROM information_schema.columns 
                WHERE table_name = 'support_resistance_levels' AND column_name = 'strength';
            """)
            result = cur.fetchone()
            # التحقق مما إذا كان النوع الحالي هو أحد أنواع الأعداد الصحيحة التي قد تسبب الخطأ
            if result and result['data_type'] in ('bigint', 'integer'):
                logger.info(f"[قاعدة البيانات] العمود 'strength' من نوع {result['data_type']}. جاري التحديث إلى NUMERIC...")
                cur.execute("ALTER TABLE support_resistance_levels ALTER COLUMN strength TYPE NUMERIC USING strength::numeric;")
                conn.commit()
                logger.info("✅ [قاعدة البيانات] تم تحديث نوع العمود 'strength' إلى NUMERIC بنجاح.")

        logger.info("✅ [قاعدة البيانات] تم تهيئة وتحديث جدول 'support_resistance_levels' بنجاح.")
        return conn
    except Exception as e:
        logger.critical(f"❌ [قاعدة البيانات] فشل الاتصال أو تهيئة الجدول: {e}")
        if conn:
            conn.rollback()
        return None

def save_levels_to_db(conn: psycopg2.extensions.connection, symbol: str, levels: List[Dict]):
    """حفظ المستويات النهائية في قاعدة البيانات."""
    if not levels:
        logger.info(f"ℹ️ [{symbol}] لا توجد مستويات نهائية ليتم حفظها.")
        return
    logger.info(f"⏳ [{symbol}] جاري حفظ {len(levels)} مستوى في قاعدة البيانات...")
    try:
        with conn.cursor() as cur:
            # تم حذف المستويات القديمة للعملة لضمان تحديث البيانات
            cur.execute("DELETE FROM support_resistance_levels WHERE symbol = %s;", (symbol,))
            insert_query = """
                INSERT INTO support_resistance_levels 
                (symbol, level_price, level_type, timeframe, strength, last_tested_at, details) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, level_price, timeframe, level_type) DO NOTHING; 
            """
            for level in levels:
                cur.execute(insert_query, (
                    symbol, level.get('level_price'), level.get('level_type'),
                    level.get('timeframe'), level.get('strength'),
                    level.get('last_tested_at'), level.get('details')
                ))
        conn.commit()
        logger.info(f"✅ [{symbol}] تم حفظ جميع المستويات بنجاح.")
    except Exception as e:
        logger.error(f"❌ [{symbol}] حدث خطأ أثناء الحفظ في قاعدة البيانات: {e}")
        conn.rollback()

# ---------------------- دوال التحليل وتحديد المستويات ----------------------
def find_price_action_levels(df: pd.DataFrame, prominence: float, width: int, cluster_eps_percent: float) -> List[Dict]:
    """تحديد القمم والقيعان وتجميعها لتحديد مناطق الدعم والمقاومة."""
    lows = df['low'].to_numpy()
    highs = df['high'].to_numpy()
    low_peaks_indices, _ = find_peaks(-lows, prominence=lows.mean() * prominence, width=width)
    high_peaks_indices, _ = find_peaks(highs, prominence=highs.mean() * prominence, width=width)

    def cluster_and_strengthen(prices: np.ndarray, indices: np.ndarray, level_type: str) -> List[Dict]:
        if len(indices) < 2: return []
        points = prices[indices].reshape(-1, 1)
        eps_value = points.mean() * cluster_eps_percent
        db = DBSCAN(eps=eps_value, min_samples=2).fit(points)
        clustered_levels = []
        for label in set(db.labels_):
            if label != -1:
                mask = (db.labels_ == label)
                cluster_indices = indices[mask]
                clustered_levels.append({
                    "level_price": float(prices[cluster_indices].mean()),
                    "level_type": level_type,
                    "strength": int(len(cluster_indices)),
                    "last_tested_at": df.index[cluster_indices[-1]].to_pydatetime()
                })
        return clustered_levels

    support_levels = cluster_and_strengthen(lows, low_peaks_indices, 'support')
    resistance_levels = cluster_and_strengthen(highs, high_peaks_indices, 'resistance')
    return support_levels + resistance_levels

def analyze_volume_profile(df: pd.DataFrame, bins: int) -> List[Dict]:
    """تحليل بروفايل الحجم لتحديد نقطة التحكم (POC)."""
    price_min, price_max = df['low'].min(), df['high'].max()
    if price_min >= price_max:
        logger.warning("[Volume Profile] النطاق السعري غير صالح. يتم التخطي.")
        return []

    price_bins = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
    volume_by_bin = np.zeros(bins)

    for _, row in df.iterrows():
        low_idx = np.searchsorted(price_bins, row['low'], side='right') - 1
        high_idx = np.searchsorted(price_bins, row['high'], side='left')
        
        low_idx = max(0, low_idx)
        high_idx = min(bins, high_idx)

        if high_idx > low_idx:
            num_bins_spanned = high_idx - low_idx
            volume_per_bin = row['volume'] / num_bins_spanned if num_bins_spanned > 0 else row['volume']
            for i in range(low_idx, high_idx):
                 volume_by_bin[i] += volume_per_bin

    if np.sum(volume_by_bin) == 0:
        logger.warning("[Volume Profile] لم يتم حساب أي حجم.")
        return []
    
    poc_index = np.argmax(volume_by_bin)
    poc_price = bin_centers[poc_index]
    poc_volume = volume_by_bin[poc_index]
    
    # ---- التعديل الرئيسي ----
    # استخدام float للقوة المستندة إلى الحجم لتجنب تجاوز حدود الأعداد الصحيحة
    return [{"level_price": float(poc_price), "level_type": 'poc', "strength": float(poc_volume), "last_tested_at": None}]


def find_confluence_zones(levels: List[Dict], confluence_percent: float) -> Tuple[List[Dict], List[Dict]]:
    """تحديد مناطق التوافق (Confluence) عن طريق دمج المستويات المتقاربة."""
    if not levels: return [], []
    levels.sort(key=lambda x: x['level_price'])
    
    tf_weights = {'1d': 3, '4h': 2, '15m': 1}
    type_weights = {'poc': 2.5, 'support': 1.5, 'resistance': 1.5, 'hvn': 1, 'confluence': 4}

    confluence_zones = []
    used_indices = set()
    
    for i in range(len(levels)):
        if i in used_indices: continue
        
        current_zone_levels = [levels[i]]
        current_zone_indices = {i}
        
        for j in range(i + 1, len(levels)):
            if j in used_indices: continue
            
            price_i = levels[i]['level_price']
            price_j = levels[j]['level_price']

            if (abs(price_j - price_i) / price_i) <= confluence_percent:
                current_zone_levels.append(levels[j])
                current_zone_indices.add(j)

        if len(current_zone_levels) > 1:
            used_indices.update(current_zone_indices)
            
            avg_price = sum(l['level_price'] * l['strength'] for l in current_zone_levels) / sum(l['strength'] for l in current_zone_levels)
            total_strength = 0
            for l in current_zone_levels:
                tf_w = tf_weights.get(l['timeframe'], 1)
                type_w = type_weights.get(l['level_type'], 1)
                total_strength += l['strength'] * tf_w * type_w

            timeframes = sorted(list(set(l['timeframe'] for l in current_zone_levels)))
            details = sorted(list(set(l['level_type'] for l in current_zone_levels)))
            last_tested = max((l['last_tested_at'] for l in current_zone_levels if l['last_tested_at']), default=None)

            confluence_zones.append({
                "level_price": avg_price,
                "level_type": 'confluence',
                # ---- التعديل الرئيسي ----
                # ضمان أن القوة النهائية هي float لتكون متوافقة مع الأنواع الأخرى
                "strength": float(total_strength), 
                "timeframe": ",".join(timeframes),
                "details": ",".join(details),
                "last_tested_at": last_tested
            })

    remaining_levels = [level for i, level in enumerate(levels) if i not in used_indices]
    logger.info(f"🤝 [Confluence] تم العثور على {len(confluence_zones)} منطقة توافق و {len(remaining_levels)} مستوى فردي متبقي.")
    return confluence_zones, remaining_levels

# ---------------------- حلقة العمل الرئيسية للتحليل ----------------------
def run_full_analysis():
    """الدالة الرئيسية التي تحتوي على منطق التحليل الكامل."""
    logger.info("🚀 بدء تشغيل محلل الدعوم والمقاومات...")
    
    client = get_binance_client()
    if not client: return
        
    conn = init_db()
    if not conn: return

    symbols_to_scan = get_validated_symbols(client, 'crypto_list.txt')
    if not symbols_to_scan:
        logger.warning("⚠️ لا توجد عملات لتحليلها. سيتم إيقاف التشغيل.")
        conn.close()
        return

    logger.info(f"🌀 سيتم تحليل {len(symbols_to_scan)} عملة.")

    timeframes_config = {
        '1d':  {'days': DATA_FETCH_DAYS_1D,  'prominence': PROMINENCE_1D,  'width': WIDTH_1D},
        '4h':  {'days': DATA_FETCH_DAYS_4H,  'prominence': PROMINENCE_4H,  'width': WIDTH_4H},
        '15m': {'days': DATA_FETCH_DAYS_15M, 'prominence': PROMINENCE_15M, 'width': WIDTH_15M}
    }

    for i, symbol in enumerate(symbols_to_scan):
        logger.info(f"--- ({i+1}/{len(symbols_to_scan)}) بدء تحليل العملة: {symbol} ---")
        raw_levels = []

        for tf, config in timeframes_config.items():
            df = fetch_historical_data(client, symbol, tf, config['days'])
            if df is not None and not df.empty:
                pa_levels = find_price_action_levels(df, config['prominence'], config['width'], CLUSTER_EPS_PERCENT)
                vol_levels = analyze_volume_profile(df, bins=VOLUME_PROFILE_BINS)
                
                for level in pa_levels + vol_levels:
                    level['timeframe'] = tf
                raw_levels.extend(pa_levels + vol_levels)
            else:
                logger.warning(f"⚠️ [{symbol}-{tf}] تعذر جلب البيانات.")
            time.sleep(1) 
            
        if raw_levels:
            confluence_zones, remaining_singles = find_confluence_zones(raw_levels, CONFLUENCE_ZONE_PERCENT)
            final_levels = confluence_zones + remaining_singles
            save_levels_to_db(conn, symbol, final_levels)
        else:
            logger.info(f"ℹ️ [{symbol}] لم يتم العثور على أي مستويات أولية لتحليلها.")
        
        logger.info(f"--- ✅ انتهى تحليل {symbol} ---")
        time.sleep(2)

    conn.close()
    logger.info("🎉🎉🎉 اكتملت عملية تحليل جميع المستويات لجميع العملات! 🎉🎉🎉")

def analysis_scheduler():
    """تقوم بجدولة وتشغيل دورة التحليل بشكل دوري."""
    while True:
        try:
            run_full_analysis()
        except Exception as e:
            logger.error(f"❌ حدث خطأ فادح في دورة التحليل الرئيسية: {e}", exc_info=True)
        
        sleep_duration_seconds = ANALYSIS_INTERVAL_HOURS * 60 * 60
        logger.info(f"👍 اكتملت دورة التحليل. سيتم الانتظار لمدة {ANALYSIS_INTERVAL_HOURS} ساعات.")
        time.sleep(sleep_duration_seconds)

# ---------------------- نقطة انطلاق البرنامج ----------------------
if __name__ == "__main__":
    # إنشاء وتشغيل خيط خادم الويب
    web_server_thread = threading.Thread(target=run_web_server)
    web_server_thread.daemon = True
    web_server_thread.start()

    # إنشاء وتشغيل خيط التحليل المجدول
    analysis_thread = threading.Thread(target=analysis_scheduler)
    analysis_thread.daemon = True
    analysis_thread.start()

    # إبقاء الخيط الرئيسي حيًا للسماح للخيوط الأخرى بالعمل
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("🛑 تم طلب إيقاف البرنامج. وداعاً!")
