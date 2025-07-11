import os
import logging
import pickle
import io
from flask import Flask, Response, make_response, render_template_string, send_from_directory
import psycopg2
from psycopg2.extras import RealDictCursor
from decouple import config

# ==============================================================================
# ------------------------------ الإعدادات الأساسية ------------------------------
# ==============================================================================

# إعداد نظام التسجيل (Logging) لعرض المعلومات والأخطاء
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('ModelDownloader')

# إنشاء تطبيق Flask
app = Flask(__name__)

# --- ✨ إضافة جديدة: تحديد مجلد النماذج المحلي ---
# تأكد من أن هذا الاسم يطابق المتغير MODEL_FOLDER في سكريبت التدريب
MODEL_FOLDER = 'SMC_V1'
MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FOLDER)


# تحميل رابط قاعدة البيانات من ملف .env
try:
    DB_URL = config('DATABASE_URL')
    logger.info("✅ تم تحميل رابط قاعدة البيانات بنجاح.")
except Exception as e:
    logger.critical(f"❌ لم يتم العثور على متغير DATABASE_URL في ملف .env. تأكد من وجود الملف والمتغير. الخطأ: {e}")
    # لا نخرج من البرنامج، قد يرغب المستخدم في تحميل الملفات المحلية فقط
    DB_URL = None

# ==============================================================================
# ----------------------------- دوال مساعدة للاتصال -----------------------------
# ==============================================================================

def get_db_connection():
    """
    تقوم بإنشاء وإرجاع اتصال جديد بقاعدة البيانات.
    ترجع None في حال فشل الاتصال أو عدم توفر الرابط.
    """
    if not DB_URL:
        return None
    try:
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        logger.error(f"❌ [DB] فشل إنشاء اتصال جديد بقاعدة البيانات: {e}")
        return None

# ==============================================================================
# --------------------------------- واجهة الويب --------------------------------
# ==============================================================================

# --- ✨ تحديث القالب: لعرض قائمتين للملفات ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>تحميل النماذج وملفات الاختبار</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; background-color: #f4f4f9; color: #333; margin: 0; padding: 20px; }
        .container { max-width: 900px; margin: 20px auto; background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 20px; }
        h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-top: 30px; }
        ul { list-style-type: none; padding: 0; }
        li { background-color: #ecf0f1; margin-bottom: 10px; border-radius: 5px; transition: all 0.3s ease; display: flex; align-items: center; }
        li:hover { background-color: #bdc3c7; transform: translateX(5px); }
        a { display: block; padding: 15px 20px; color: #2980b9; text-decoration: none; font-weight: 500; font-size: 1.1em; flex-grow: 1; }
        a:hover { color: #1c587f; }
        .file-icon { margin-right: 15px; font-size: 1.2em; }
        .json-icon { color: #f1c40f; }
        .pkl-icon { color: #3498db; }
        .error { color: #c0392b; background-color: #f2dede; border: 1px solid #ebccd1; padding: 15px; border-radius: 5px; }
        .empty { color: #7f8c8d; text-align: center; font-size: 1.2em; padding: 40px 0; }
        .info { background-color: #eaf2f8; border: 1px solid #aed6f1; padding: 15px; border-radius: 5px; color: #2874a6; margin-bottom: 20px;}
    </style>
</head>
<body>
    <div class="container">
        <h1>لوحة تحميل النماذج</h1>
        
        <!-- قسم النماذج من قاعدة البيانات -->
        <h2><span class="file-icon">🗄️</span>نماذج من قاعدة البيانات</h2>
        {% if db_error %}
            <p class="error"><b>خطأ في قاعدة البيانات:</b> {{ db_error }}</p>
        {% elif db_models %}
            <ul>
                {% for model in db_models %}
                <li>
                    <span class="file-icon pkl-icon">📦</span>
                    <a href="/download_db/{{ model.model_name }}" download>{{ model.model_name }}</a>
                </li>
                {% endfor %}
            </ul>
        {% else %}
            <p class="empty">لا توجد نماذج في قاعدة البيانات أو تعذر الاتصال.</p>
        {% endif %}

        <!-- قسم الملفات المحلية -->
        <h2><span class="file-icon">🖥️</span>ملفات محلية من مجلد ({{ model_folder_name }})</h2>
        {% if local_files_error %}
             <p class="error"><b>خطأ في الملفات المحلية:</b> {{ local_files_error }}</p>
        {% elif local_files %}
            <p class="info">تم العثور على {{ local_files|length }} ملف محلي.</p>
            <ul>
                {% for file in local_files %}
                <li>
                    {% if file.endswith('.json') %}
                        <span class="file-icon json-icon">📊</span>
                    {% else %}
                         <span class="file-icon pkl-icon">📦</span>
                    {% endif %}
                    <a href="/download_local/{{ file }}" download>{{ file }}</a>
                </li>
                {% endfor %}
            </ul>
        {% else %}
            <p class="empty">لم يتم العثور على ملفات في المجلد المحلي.</p>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    """
    الصفحة الرئيسية التي تعرض قائمة بالملفات من قاعدة البيانات والملفات المحلية.
    """
    # --- جلب النماذج من قاعدة البيانات ---
    db_models_list = []
    db_error_msg = None
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT ON (model_name) model_name
                    FROM ml_models ORDER BY model_name, trained_at DESC;
                """)
                db_models_list = cur.fetchall()
                logger.info(f"✅ [DB] تم العثور على {len(db_models_list)} نموذج فريد في قاعدة البيانات.")
        except Exception as e:
            logger.error(f"❌ [DB] خطأ أثناء جلب قائمة النماذج: {e}")
            db_error_msg = str(e)
        finally:
            conn.close()
    else:
        db_error_msg = "لم يتم تكوين رابط قاعدة البيانات (DATABASE_URL)."

    # --- ✨ إضافة جديدة: جلب الملفات المحلية ---
    local_files_list = []
    local_files_error_msg = None
    try:
        if os.path.exists(MODELS_PATH):
            # جلب كل ملفات .pkl و .json
            files = [f for f in os.listdir(MODELS_PATH) if f.endswith(('.pkl', '.json'))]
            local_files_list = sorted(files)
            logger.info(f"✅ [Local] تم العثور على {len(local_files_list)} ملف في المجلد '{MODEL_FOLDER}'.")
        else:
            local_files_error_msg = f"المجلد '{MODEL_FOLDER}' غير موجود."
            logger.warning(f"⚠️ [Local] {local_files_error_msg}")
    except Exception as e:
        local_files_error_msg = f"خطأ أثناء قراءة المجلد المحلي: {e}"
        logger.error(f"❌ [Local] {local_files_error_msg}")

    return render_template_string(
        HTML_TEMPLATE,
        db_models=db_models_list,
        db_error=db_error_msg,
        local_files=local_files_list,
        local_files_error=local_files_error_msg,
        model_folder_name=MODEL_FOLDER
    )

@app.route('/download_db/<model_name>')
def download_db_model(model_name):
    """
    تقوم بتحميل بيانات النموذج المحدد من قاعدة البيانات وإرسالها كملف.
    """
    conn = get_db_connection()
    if not conn:
        return "خطأ: فشل الاتصال بقاعدة البيانات.", 500

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT model_data FROM ml_models WHERE model_name = %s ORDER BY trained_at DESC LIMIT 1;",
                (model_name,)
            )
            result = cur.fetchone()

            if not result or 'model_data' not in result:
                logger.warning(f"⚠️ [DB] لم يتم العثور على النموذج '{model_name}'.")
                return f"Model '{model_name}' not found in database.", 404

            model_data = result['model_data']
            logger.info(f"✅ [DB] بدء تحميل النموذج '{model_name}'.")
            
            response = make_response(model_data)
            response.headers.set('Content-Type', 'application/octet-stream')
            response.headers.set('Content-Disposition', 'attachment', filename=f"{model_name}.pkl")
            return response

    except Exception as e:
        logger.error(f"❌ [DB] خطأ أثناء تحميل النموذج '{model_name}': {e}")
        return "حدث خطأ أثناء معالجة طلبك.", 500
    finally:
        if conn:
            conn.close()

# --- ✨ إضافة جديدة: مسار لتحميل الملفات المحلية ---
@app.route('/download_local/<path:filename>')
def download_local_file(filename):
    """
    تقوم بتحميل ملف محلي من مجلد النماذج.
    """
    logger.info(f"✅ [Local] بدء تحميل الملف '{filename}'.")
    return send_from_directory(MODELS_PATH, filename, as_attachment=True)


# ==============================================================================
# --------------------------------- نقطة البداية --------------------------------
# ==============================================================================
if __name__ == '__main__':
    # التأكد من وجود المجلد المحلي
    if not os.path.exists(MODELS_PATH):
        logger.warning(f"⚠️ المجلد المحلي '{MODEL_FOLDER}' غير موجود. سيتم إنشاؤه.")
        try:
            os.makedirs(MODELS_PATH)
        except Exception as e:
            logger.error(f"❌ فشل في إنشاء المجلد المحلي: {e}")

    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🌍 لبدء التحميل، افتح الرابط التالي في متصفحك: http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
