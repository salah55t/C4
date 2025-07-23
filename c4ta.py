# -*- coding: utf-8 -*-
"""
هذا السكريبت يقوم بتحليل بيانات البيتكوين (BTCUSDT) التاريخية على إطار زمني مدته 4 ساعات
لآخر 10 أيام لتحديد الساعات التي ينخفض فيها السعر بشكل متكرر.

للتشغيل:
1. تأكد من تثبيت المكتبات المطلوبة: pip install python-binance pandas
2. قم بوضع مفتاح API والمفتاح السري الخاص بك في المتغيرات أدناه.
3. قم بتشغيل السكريبت.
"""

import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta

# --- الإعدادات الرئيسية ---
# هام: قم بإدخال مفاتيح API الخاصة بك هنا
API_KEY = "VTqU8kmPmXbWcabah7DOvNhPiMFN92Q8WtLjt75AziDX1Adp8snmHBqqqKBo01N8"  # أدخل مفتاح API الخاص بك هنا
API_SECRET = "h8aL7je0HOIJ4tWucNaLgvLcIp3gVvzeGRhLN9F1TyfS1EcXDMOFZUS2v23oXEpG" # أدخل المفتاح السري الخاص بك هنا

SYMBOL = 'BTCUSDT'
TIMEFRAME = Client.KLINE_INTERVAL_4HOUR
DAYS_AGO = 10

# --- نهاية الإعدادات ---

def analyze_btc_drop_times():
    """
    تقوم هذه الدالة الرئيسية بجلب البيانات وتحليلها وطباعة النتائج.
    """
    print("🚀 بدء تحليل أوقات هبوط البيتكوين...")

    # التحقق من وجود مفاتيح API
    if API_KEY == "YOUR_API_KEY" or API_SECRET == "YOUR_API_SECRET":
        print("🛑 خطأ: الرجاء إدخال مفتاح API والمفتاح السري الخاص بك في السكريبت.")
        return

    # تهيئة عميل Binance
    try:
        client = Client(API_KEY, API_SECRET)
        # اختبار الاتصال بالخادم
        client.ping()
        print("✅ تم الاتصال بنجاح بمنصة Binance.")
    except Exception as e:
        print(f"❌ فشل الاتصال بـ Binance: {e}")
        return

    # حساب تاريخ البدء (قبل 10 أيام من الآن)
    start_date = (datetime.utcnow() - timedelta(days=DAYS_AGO)).strftime("%d %b, %Y")

    # جلب البيانات التاريخية
    print(f"📊 جلب البيانات التاريخية للعملة {SYMBOL} على إطار {TIMEFRAME} منذ {start_date}...")
    try:
        klines = client.get_historical_klines(SYMBOL, TIMEFRAME, start_str=start_date)
        if not klines:
            print("لم يتم العثور على بيانات للفترة المحددة.")
            return
    except Exception as e:
        print(f"❌ حدث خطأ أثناء جلب البيانات: {e}")
        return

    # تحويل البيانات إلى DataFrame باستخدام pandas لتسهيل التحليل
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    df = pd.DataFrame(klines, columns=columns)

    # --- معالجة البيانات ---
    # تحويل الأعمدة الرقمية إلى أرقام عشرية
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    # تحويل وقت فتح الشمعة إلى تاريخ ووقت قابل للقراءة (بتوقيت UTC)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)

    # حساب التغير في السعر لكل شمعة (إغلاق - فتح)
    df['price_change'] = df['close'] - df['open']

    # فلترة البيانات لإظهار الشموع التي انخفض فيها السعر فقط
    drops = df[df['price_change'] < 0].copy()

    if drops.empty:
        print("\nلم يتم تسجيل أي انخفاضات في السعر خلال الفترة المحددة.")
        return

    # استخراج "ساعة" بداية الشمعة من عمود الوقت
    # هذه هي الساعة التي بدأ فيها الانخفاض
    drops['hour_of_day'] = drops['open_time'].dt.hour

    # --- عرض النتائج ---
    print("\n" + "="*50)
    print("📈 نتيجة التحليل: الساعات الأكثر تكراراً لهبوط السعر")
    print("="*50)
    print(f"(خلال آخر {DAYS_AGO} أيام، على إطار 4 ساعات، التوقيت العالمي المنسق UTC)")

    # حساب عدد مرات الهبوط في كل ساعة وترتيبها من الأكثر للأقل
    drop_counts = drops['hour_of_day'].value_counts()

    # طباعة النتائج بشكل منظم
    print("\n{:<15} | {:<20}".format("ساعة بداية الشمعة", "عدد مرات الهبوط"))
    print("-"*38)
    for hour, count in drop_counts.items():
        # تنسيق الساعة لتكون دائماً من رقمين (e.g., 04:00)
        hour_str = f"{hour:02d}:00"
        print("{:<15} | {:<20}".format(hour_str, count))

    # تحديد الساعة الأكثر هبوطاً
    most_frequent_hour = drop_counts.idxmax()
    print("\n" + "-"*50)
    print(f"💡 الخلاصة: الساعة الأكثر تكراراً للهبوط هي الساعة {most_frequent_hour:02d}:00 بتوقيت UTC.")
    print("="*50)


if __name__ == "__main__":
    analyze_btc_drop_times()
