import pandas as pd
import yfinance as yf
import pandas_ta as ta
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings

# تجاهل التحذيرات غير الضرورية لتحسين قراءة المخرجات
warnings.filterwarnings('ignore')

def train_and_predict_for_ticker(ticker):
    """
    تقوم هذه الدالة بتحميل البيانات، حساب المؤشرات، تدريب نموذج LightGBM،
    وتقييمه وعمل تنبؤ لأصل مالي معين.
    
    Args:
        ticker (str): رمز الأصل المالي (مثال: 'BTC-USD').
        
    Returns:
        None: تقوم بطباعة النتائج والتنبؤات مباشرة.
    """
    print(f"--- بدء المعالجة لـ: {ticker} ---")
    
    # 1. تحميل البيانات التاريخية لآخر 5 سنوات
    try:
        data = yf.download(ticker, period="5y", interval="1d")
        if data.empty:
            print(f"لم يتم العثور على بيانات لـ {ticker}. قد يكون الرمز غير صحيح أو لا توجد بيانات تاريخية.")
            print("-" * 30 + "\n")
            return
    except Exception as e:
        print(f"حدث خطأ أثناء تحميل البيانات لـ {ticker}: {e}")
        print("-" * 30 + "\n")
        return

    # 2. حساب المؤشرات الفنية باستخدام مكتبة pandas_ta
    # EMA (12, 26)
    data.ta.ema(length=12, append=True)
    data.ta.ema(length=26, append=True)
    
    # ADX (14)
    data.ta.adx(length=14, append=True)
    
    # RSI (14)
    data.ta.rsi(length=14, append=True)
    
    # MACD (12, 26, 9)
    data.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    # 3. تحديد المتغير المستهدف (Target)
    # الهدف هو معرفة ما إذا كان سعر الإغلاق في اليوم التالي سيرتفع أم سينخفض
    # 1 = ارتفاع (صعود), 0 = انخفاض (هبوط)
    data['future_close'] = data['Close'].shift(-1)
    data['target'] = (data['future_close'] > data['Close']).astype(int)
    
    # 4. إعداد البيانات للنموذج
    # إزالة الصفوف التي تحتوي على قيم فارغة (NaN) الناتجة عن حساب المؤشرات
    data.dropna(inplace=True)
    
    # تحديد الميزات (Features) التي سيتم استخدامها في التدريب
    features = [
        'EMA_12', 'EMA_26', 
        'ADX_14', 
        'RSI_14', 
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'Volume'
    ]
    
    X = data[features]
    y = data['target']
    
    # التأكد من وجود بيانات كافية للتدريب
    if len(X) < 50:
        print(f"لا توجد بيانات كافية لتدريب النموذج لـ {ticker} بعد التنظيف.")
        print("-" * 30 + "\n")
        return
        
    # 5. تقسيم البيانات إلى مجموعة تدريب ومجموعة اختبار
    # 80% للتدريب و 20% للاختبار
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # 6. تدريب نموذج LightGBM
    print("بدء تدريب النموذج...")
    model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=1000, # عدد الأشجار
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )
    
    # استخدام خاصية التوقف المبكر لتحسين الأداء وتجنب الـ Overfitting
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='logloss',
        callbacks=[lgb.early_stopping(100, verbose=False)] # إيقاف التدريب إذا لم يتحسن الأداء لـ 100 جولة
    )
    
    # 7. تقييم أداء النموذج
    print("\n--- تقييم النموذج ---")
    y_pred = model.predict(X_test)
    
    print("مصفوفة الارتباك (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nتقرير التصنيف (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=['هبوط (0)', 'صعود (1)']))
    
    # 8. عمل تنبؤ للاتجاه المستقبلي
    # استخدام آخر صف من البيانات للتنبؤ باليوم التالي
    last_row = X.iloc[[-1]]
    future_prediction = model.predict(last_row)
    prediction_proba = model.predict_proba(last_row)

    print("\n--- التنبؤ المستقبلي ---")
    direction = "صعود" if future_prediction[0] == 1 else "هبوط"
    confidence = prediction_proba[0][future_prediction[0]] * 100
    
    print(f"اتجاه السوق المتوقع لـ {ticker} في اليوم التالي هو: {direction}")
    print(f"بنسبة ثقة تبلغ: {confidence:.2f}%")
    print("-" * 30 + "\n")


# قائمة الأصول المطلوب تحليلها
tickers_list = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'RUB=X']

# تنفيذ الدالة لكل أصل في القائمة
for ticker in tickers_list:
    train_and_predict_for_ticker(ticker)
