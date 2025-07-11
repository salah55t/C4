import os
import logging
import json
import io
from flask import Flask, Response, make_response, render_template_string
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

# تحميل رابط قاعدة البيانات من ملف .env
try:
    DB_URL = config('DATABASE_URL')
    logger.info("✅ تم تحميل رابط قاعدة البيانات بنجاح.")
except Exception as e:
    logger.critical(f"❌ لم يتم العثور على متغير DATABASE_URL في ملف .env. تأكد من وجود الملف والمتغير. الخطأ: {e}")
    exit(1)

# ==============================================================================
# ----------------------------- دوال مساعدة للاتصال -----------------------------
# ==============================================================================

def get_db_connection():
    """
    تقوم بإنشاء وإرجاع اتصال جديد بقاعدة البيانات.
    ترجع None في حال فشل الاتصال.
    """
    try:
        conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        logger.error(f"❌ [DB] فشل إنشاء اتصال جديد بقاعدة البيانات: {e}")
        return None

# ==============================================================================
# --------------------------------- واجهة الويب --------------------------------
# ==============================================================================

# --- ✨ تحديث القالب: لعرض روابط للملفات والنماذج ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>تحميل النماذج وملفات الاختبار</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; background-color: #f4f4f9; color: #333; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: 20px auto; background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .model-entry { background-color: #ecf0f1; margin-bottom: 15px; border-radius: 5px; padding: 15px 20px; transition: all 0.3s ease; }
        .model-entry:hover { background-color: #dfe6e9; }
        .model-name { font-weight: bold; font-size: 1.2em; color: #2c3e50; margin-bottom: 10px; }
        .download-links a {
            display: inline-block;
            margin-left: 10px;
            margin-top: 5px;
            padding: 8px 15px;
            background-color: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-size: 0.9em;
            transition: background-color 0.2s;
        }
        .download-links a.metrics-link { background-color: #f39c12; }
        .download-links a:hover { background-color: #2980b9; }
        .download-links a.metrics-link:hover { background-color: #e67e22; }
        .error { color: #c0392b; background-color: #f2dede; border: 1px solid #ebccd1; padding: 15px; border-radius: 5px; }
        .empty { color: #7f8c8d; text-align: center; font-size: 1.2em; padding: 40px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>النماذج المتاحة للتحميل من قاعدة البيانات</h1>
        {% if error %}
            <p class="error"><b>خطأ:</b> {{ error }}</p>
        {% elif models %}
            {% for model in models %}
            <div class="model-entry">
                <div class="model-name">{{ model.model_name }}</div>
                <div class="download-links">
                    <a href="/download_model/{{ model.model_name }}" download>📦 تحميل النموذج (.pkl)</a>
                    {% if model.has_metrics %}
                    <a href="/download_metrics/{{ model.model_name }}" class="metrics-link" download>📊 تحميل ملف الاختبار (.json)</a>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        {% else %}
            <p class="empty">لا توجد نماذج في قاعدة البيانات.</p>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    """
    الصفحة الرئيسية التي تعرض قائمة بجميع النماذج وملفات الاختبار المتاحة.
    """
    conn = get_db_connection()
    if not conn:
        return render_template_string(HTML_TEMPLATE, error="فشل الاتصال بقاعدة البيانات.")

    models_list = []
    try:
        with conn.cursor() as cur:
            # جلب أحدث نسخة من كل نموذج مع التحقق من وجود حقل المقاييس
            cur.execute("""
                SELECT DISTINCT ON (model_name) 
                       model_name, 
                       (metrics IS NOT NULL AND metrics::text != 'null') as has_metrics
                FROM ml_models
                ORDER BY model_name, trained_at DESC;
            """)
            models_list = cur.fetchall()
            logger.info(f"✅ تم العثور على {len(models_list)} نموذج فريد في قاعدة البيانات.")
    except Exception as e:
        logger.error(f"❌ خطأ أثناء جلب قائمة النماذج: {e}")
        return render_template_string(HTML_TEMPLATE, error=str(e))
    finally:
        if conn:
            conn.close()

    return render_template_string(HTML_TEMPLATE, models=models_list)

@app.route('/download_model/<model_name>')
def download_model(model_name):
    """
    تقوم بتحميل بيانات النموذج (ملف pkl) المحدد من قاعدة البيانات.
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
                logger.warning(f"⚠️ لم يتم العثور على بيانات النموذج '{model_name}'.")
                return f"Model data for '{model_name}' not found.", 404

            model_data = result['model_data']
            logger.info(f"✅ بدء تحميل النموذج '{model_name}'.")
            
            response = make_response(model_data)
            response.headers.set('Content-Type', 'application/octet-stream')
            response.headers.set('Content-Disposition', 'attachment', filename=f"{model_name}.pkl")
            return response

    except Exception as e:
        logger.error(f"❌ خطأ أثناء تحميل النموذج '{model_name}': {e}")
        return "حدث خطأ أثناء معالجة طلبك.", 500
    finally:
        if conn:
            conn.close()

# --- ✨ إضافة جديدة: مسار لتحميل ملفات الاختبار (Metrics) ---
@app.route('/download_metrics/<model_name>')
def download_metrics(model_name):
    """
    تقوم بتحميل بيانات الاختبار (ملف json) للنموذج المحدد من قاعدة البيانات.
    """
    conn = get_db_connection()
    if not conn:
        return "خطأ: فشل الاتصال بقاعدة البيانات.", 500

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT metrics FROM ml_models WHERE model_name = %s ORDER BY trained_at DESC LIMIT 1;",
                (model_name,)
            )
            result = cur.fetchone()

            if not result or 'metrics' not in result or result['metrics'] is None:
                logger.warning(f"⚠️ لم يتم العثور على بيانات الاختبار للنموذج '{model_name}'.")
                return f"Metrics for '{model_name}' not found.", 404

            metrics_data = result['metrics']
            metrics_json_string = json.dumps(metrics_data, indent=4, ensure_ascii=False)
            
            logger.info(f"✅ بدء تحميل ملف الاختبار للنموذج '{model_name}'.")
            
            response = make_response(metrics_json_string)
            response.headers.set('Content-Type', 'application/json; charset=utf-8')
            response.headers.set('Content-Disposition', 'attachment', filename=f"{model_name}_metrics.json")
            return response

    except Exception as e:
        logger.error(f"❌ خطأ أثناء تحميل ملف الاختبار للنموذج '{model_name}': {e}")
        return "حدث خطأ أثناء معالجة طلبك.", 500
    finally:
        if conn:
            conn.close()

# ==============================================================================
# --------------------------------- نقطة البداية --------------------------------
# ==============================================================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🌍 لبدء التحميل، افتح الرابط التالي في متصفحك: http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
