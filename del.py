import os
import psycopg2
from decouple import config
import logging
from flask import Flask, request, flash, redirect, url_for, get_flashed_messages
from markupsafe import Markup

# --- إعداد التطبيق والتسجيل ---
app = Flask(__name__)
app.secret_key = os.urandom(24) # مطلوب لاستخدام الرسائل الفورية (flash messages)

# إعداد نظام التسجيل
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DBCleanupWebApp')

# --- تحميل متغيرات البيئة ---
try:
    DB_URL = config('DATABASE_URL')
    logger.info("Successfully loaded DATABASE_URL.")
except Exception as e:
    logger.critical("❌ Critical Failure: Could not load 'DATABASE_URL'. Ensure a .env file exists.")
    DB_URL = None

# --- كود الواجهة (HTML & CSS) ---
# تم دمج الواجهة هنا داخل متغير نصي متعدد الأسطر
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>لوحة تحكم قاعدة البيانات</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; 
            background-color: #f4f4f9; 
            color: #333; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            height: 100vh; 
            margin: 0; 
        }
        .container { 
            background: #fff; 
            padding: 2rem 3rem; 
            border-radius: 10px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.1); 
            text-align: center; 
            max-width: 600px; 
        }
        .warning-box { 
            background-color: #fff3cd; 
            border: 1px solid #ffeeba; 
            color: #856404; 
            padding: 1rem; 
            border-radius: 8px; 
            margin-bottom: 1.5rem; 
        }
        h1 { 
            color: #dc3545; 
            font-size: 1.8rem; 
        }
        p { 
            line-height: 1.6; 
        }
        .confirmation-phrase { 
            background: #e9ecef; 
            padding: 0.3rem 0.6rem; 
            border-radius: 5px; 
            font-weight: bold; 
            color: #495057; 
        }
        form { 
            margin-top: 1.5rem; 
        }
        input[type="text"] { 
            width: 100%; 
            padding: 0.8rem; 
            margin-bottom: 1rem; 
            border: 1px solid #ced4da; 
            border-radius: 5px; 
            box-sizing: border-box; 
            text-align: center; 
        }
        button { 
            background-color: #dc3545; 
            color: white; 
            border: none; 
            padding: 0.8rem 1.5rem; 
            font-size: 1rem; 
            border-radius: 5px; 
            cursor: pointer; 
            transition: background-color 0.3s; 
            width: 100%; 
        }
        button:hover { 
            background-color: #c82333; 
        }
        .messages { 
            list-style: none; 
            padding: 0; 
            margin-top: 1.5rem; 
        }
        .messages li { 
            padding: 1rem; 
            border-radius: 5px; 
            margin-bottom: 1rem; 
        }
        .messages .success { 
            background-color: #d4edda; 
            color: #155724; 
            border: 1px solid #c3e6cb; 
        }
        .messages .danger { 
            background-color: #f8d7da; 
            color: #721c24; 
            border: 1px solid #f5c6cb; 
        }
        .messages .warning { 
            background-color: #fff3cd; 
            color: #856404; 
            border: 1px solid #ffeeba; 
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="warning-box">
            <h1>!!! تحذير خطير / DANGEROUS WARNING !!!</h1>
            <p>سيقوم هذا الإجراء بحذف جميع البيانات بشكل نهائي من جدولي <strong>signals</strong> و <strong>notifications</strong>. هذا الإجراء لا يمكن التراجع عنه.</p>
        </div>
        
        {{ messages_placeholder }}

        <form action="/clear-data" method="post">
            <p>للمتابعة، يرجى كتابة الجملة التالية بالضبط: <code class="confirmation-phrase">تأكيد الحذف</code></p>
            <input type="text" id="confirmation" name="confirmation" required autocomplete="off">
            <button type="submit">⚠️ حذف جميع البيانات الآن ⚠️</button>
        </form>
    </div>
</body>
</html>
"""

def get_db_connection():
    """إنشاء وإرجاع اتصال بقاعدة البيانات."""
    try:
        conn = psycopg2.connect(DB_URL)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def truncate_tables():
    """
    الدالة الأساسية لحذف البيانات. ترجع رسالة نجاح أو خطأ.
    """
    if not DB_URL:
        return "Error: DATABASE_URL is not configured.", "danger"

    conn = get_db_connection()
    if not conn:
        return "Error: Could not connect to the database.", "danger"

    try:
        with conn.cursor() as cur:
            logger.info("Executing TRUNCATE command...")
            cur.execute("TRUNCATE TABLE signals, notifications RESTART IDENTITY CASCADE;")
            conn.commit()
            logger.info("✅ Data successfully deleted.")
            return "Success! All data has been permanently deleted from the tables.", "success"
    except Exception as e:
        logger.error(f"❌ An error occurred: {e}")
        conn.rollback()
        return f"An error occurred: {e}", "danger"
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

def render_page_with_messages():
    """تجهيز كود HTML لعرض الرسائل."""
    messages_html = ""
    # استرجاع الرسائل التي تم إرسالها عبر flash
    flashed_messages = get_flashed_messages(with_categories=True)
    if flashed_messages:
        messages_html += '<ul class="messages">'
        for category, message in flashed_messages:
            messages_html += f'<li class="{category}">{message}</li>'
        messages_html += '</ul>'
    
    # استبدال العنصر النائب في القالب بالرسائل الفعلية
    return Markup(HTML_TEMPLATE.replace('{{ messages_placeholder }}', messages_html))

# --- مسارات (Routes) تطبيق الويب ---

@app.route('/')
def index():
    """عرض صفحة التحكم الرئيسية."""
    with app.app_context():
        return render_page_with_messages()

@app.route('/clear-data', methods=['POST'])
def clear_data():
    """معالجة طلب حذف البيانات."""
    confirmation_text = request.form.get('confirmation')
    
    if confirmation_text == "تأكيد الحذف":
        message, category = truncate_tables()
        flash(message, category)
    else:
        flash("Confirmation text did not match. Operation cancelled.", "warning")

    return redirect(url_for('index'))

if __name__ == '__main__':
    # للتشغيل المحلي، استخدم هذا الأمر: python app_single_file.py
    # للنشر، استخدم Gunicorn: gunicorn --workers 3 --bind 0.0.0.0:8000 app_single_file:app
    app.run(debug=True, port=5001)
